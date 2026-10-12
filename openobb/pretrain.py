"""Backbone pretraining by feature distillation (OpenOBB's own implementation).

OpenOBB6's backbone is new, so no ImageNet / COCO weights exist for it, and training it from scratch on a small
labelled set fails (research/LEDGER.md F25). Instead it learns general features from unlabelled images by matching
a strong self-supervised teacher, DINOv2 (Apache-2.0): at every location of the stride-8 / 16 / 32 maps, a small
projection of the student's features must point in the same direction as the teacher's patch features
(1 - cosine). This converges far faster than supervised pretraining on COCO and needs no labels; the images can be
anything (COCO, DOTA, drawings): the more varied, the more general the backbone.

    openobb pretrain model=openobb6x data=/content/coco/train2017,/content/drawings epochs=20 imgsz=640
    -> runs/pretrain/train/weights/last.pt;  then:
    openobb train model=openobb6x weights=runs/pretrain/train/weights/last.pt data=data.yaml ...
"""
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from openobb.data.imgio import IMG_EXT, imread


def _images(data):
    dirs = data if isinstance(data, (list, tuple)) else [d for d in str(data).split(",") if d.strip()]
    files = []
    for d in dirs:
        p = Path(str(d).strip())
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT]
        elif p.suffix.lower() in IMG_EXT:
            files.append(p)
        elif p.suffix.lower() == ".txt" and p.exists():
            files += [Path(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not files:
        raise SystemExit(f"no images found in {data}")
    return sorted(files)


class Crops(Dataset):
    """Random resized crops (area 15-100% of the image, aspect 3:4-4:3), flips and 90-degree turns, RGB uint8."""

    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = imread(self.files[i])
        if im is None or min(im.shape[:2]) < 16:
            im = np.full((self.size, self.size, 3), 255, np.uint8)
        h, w = im.shape[:2]
        for _ in range(10):
            area = random.uniform(0.15, 1.0) * h * w
            ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
            if 0 < cw <= w and 0 < ch <= h:
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                im = im[y0:y0 + ch, x0:x0 + cw]
                break
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if random.random() < 0.5:
            im = im[:, ::-1]
        if random.random() < 0.5:
            im = im[::-1]
        im = np.rot90(im, random.randint(0, 3))
        return torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1)


class Projector(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, d, 1), nn.BatchNorm2d(d), nn.GELU(), nn.Conv2d(d, d, 1))

    def forward(self, x):
        return self.net(x)


def _cos_loss(s, t):
    t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
    return (1 - F.cosine_similarity(s.float(), t.float(), dim=1)).mean()


def pretrain(model="openobb6x", data=None, epochs=20, imgsz=640, batch=32, teacher="dinov2_b", teacher_imgsz=None,
             lr=2e-3, wd=0.05, warmup_epochs=1, workers=8, project="runs/pretrain", name="train", exist_ok=False,
             device=None, limit=None, seed=0, v6_off=None):
    """Distil DINOv2 features into the OpenOBB6 backbone on unlabelled images; returns the weights path."""
    from openobb.cli import MODELS, increment_path
    from openobb.models.openobb6 import OpenOBB6
    from openobb.models.vit import DINOV2, DinoV2Backbone
    if data is None:
        raise SystemExit("openobb pretrain needs data=<folder(s) of images, comma-separated>")
    arch, size = MODELS.get(str(model).lower(), (None, None))
    if arch != "v6":
        raise SystemExit("openobb pretrain is for the OpenOBB6 backbone: model=openobb6n/s/m/l/x")
    random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device if device not in (None, "") else ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(device, int) or (isinstance(device, str) and str(device).isdigit()):
        dev = torch.device(f"cuda:{int(device)}")
    from openobb.models.openobb6 import parse_off
    student_off = parse_off(v6_off)
    files = _images(data)
    if limit:
        files = files[:int(limit)]
    save_dir = increment_path(project, name, exist_ok)
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)

    student = OpenOBB6(size, num_classes=1, img_size=imgsz, off=v6_off).backbone.to(dev)
    tch = DinoV2Backbone(teacher).to(dev)
    tch.load_pretrained()
    tch.eval().requires_grad_(False)
    d = DINOV2[teacher][0]
    layers = [tch.layers[1], tch.layers[3], tch.layers[3]]          # middle -> stride 8, last -> 16 and 32
    c = student.channels
    proj = nn.ModuleList([Projector(c[1], d), Projector(c[2], d), Projector(c[3], d)]).to(dev)
    weights = (0.5, 1.0, 0.5)
    t_size = int(teacher_imgsz or round(imgsz * 0.7 / 14) * 14)
    t_size = max(14, t_size // 14 * 14)

    params = [p for p in list(student.parameters()) + list(proj.parameters()) if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.999))
    dl = DataLoader(Crops(files, imgsz), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=workers > 0)
    steps = max(1, len(dl)) * epochs
    warm = max(1, len(dl) * warmup_epochs)
    amp = dev.type == "cuda"
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    print(f"[pretrain] OpenOBB6-{size} backbone <- DINOv2 {teacher} | {len(files)} images, {epochs} epochs, "
          f"student {imgsz} px, teacher {t_size} px, batch {batch}, {dev} -> {save_dir}", flush=True)
    log = []
    it = 0
    for ep in range(epochs):
        student.train()
        proj.train()
        t0, tot, n = time.time(), 0.0, 0
        for x in dl:
            x = x.to(dev, non_blocking=True).float().div_(255)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (it + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(it / steps, 1.0))))
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                xt = F.interpolate((x - mean) / std, size=(t_size, t_size), mode="bilinear", align_corners=False)
                tf = tch.vit(xt, layers)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                _, f8, f16, f32 = student(x)
                loss = sum(w * _cos_loss(p(f), t) for w, p, f, t in zip(weights, proj, (f8, f16, f32), tf))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
            it += 1
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        log.append(rec)
        print(f"[pretrain] epoch {ep + 1}/{epochs}  distill_loss {rec['loss']:.4f}  {rec['time_s']:.0f}s", flush=True)
        state = {f"backbone.{k}": v for k, v in student.state_dict().items()}
        torch.save({"model": state, "classes": [], "epoch": ep + 1,
                    "args": {"arch": "v6", "size": size, "img": imgsz, "pretrain": "dinov2_distill", "teacher": teacher,
                             "v6_off": ",".join(sorted(student_off))}},
                   save_dir / "weights" / "last.pt")
        (save_dir / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    print(f"[pretrain] done: {save_dir / 'weights' / 'last.pt'}  (use as weights= for openobb train "
          f"model=openobb6{size})", flush=True)
    return str(save_dir / "weights" / "last.pt")
