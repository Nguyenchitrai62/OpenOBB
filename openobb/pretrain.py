"""Backbone pretraining by feature distillation (OpenOBB's own implementation).

A new core (OpenOBB6, OpenOBB9) has no ImageNet / COCO weights, and training it from scratch on a small labelled
set fails (research/LEDGER.md F25, F40). Instead it learns from unlabelled images by matching frozen teachers: at
every location of the student's maps, a small projection (1x1 - BN - GELU - 1x1, discarded afterwards) must point
in the same direction as the teacher's features (loss 1 - cosine).

OpenOBB6: one teacher, DINOv2 (Apache-2.0), on backbone strides 8 / 16 / 32.

OpenOBB9 (docs/OPENOBB9.md sections 2, 3, 6): two teachers on the neck outputs P3 / P4 / P5.
  * "dfine":  the D-FINE COCO core (HGNetv2 + hybrid encoder, Apache-2.0; the core of OpenOBB3 / OpenOBB8) at the
              student's resolution and input (RGB in [0, 1]); its P3 / P4 / P5 are the targets of the student's.
              Proposition 1: if the student matches it (cosine high), OpenOBB8's head can do on the student what it
              does on OpenOBB3's core.
  * "dinov2": DINOv2 last-layer patch features (ImageNet mean / std) -> targets for P4 and P5.
  Every level P3 / P4 / P5 gets a distillation loss (the guard against dead levels, OPENOBB9 section 5).
  Gate T2: with val_data (held-out drawings), the mean cosine per (teacher, level) is measured after every epoch;
  T2 passes when the dfine cosine is >= t2_min (0.8) at P3 and P4.

    openobb pretrain model=openobb9x data=/content/coco,/content/drawings val_data=/content/heldout epochs=10 imgsz=1024
    -> runs/pretrain/train/weights/last.pt  (keys backbone.* and neck.* only);  then:
    openobb train model=openobb9x weights=runs/pretrain/train/weights/last.pt data=data.yaml ...
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

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TEACHERS = ("dfine", "dinov2")


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


class CenterCrops(Dataset):
    """Deterministic validation view: the centre square of the image resized to `size`, RGB uint8, no augmentation."""

    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = imread(self.files[i])
        if im is None or min(im.shape[:2]) < 16:
            im = np.full((self.size, self.size, 3), 255, np.uint8)
        h, w = im.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        im = cv2.resize(im[y0:y0 + s, x0:x0 + s], (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1)


class Projector(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, d, 1), nn.BatchNorm2d(d), nn.GELU(), nn.Conv2d(d, d, 1))

    def forward(self, x):
        return self.net(x)


def _cos_map(s, t):
    """Per-location cosine (B, H, W) between student map s and teacher map t resized (bilinear) to s's grid."""
    t = F.interpolate(t.float(), size=s.shape[-2:], mode="bilinear", align_corners=False)
    return F.cosine_similarity(s.float(), t, dim=1)


def _cos_loss(s, t):
    return (1 - _cos_map(s, t)).mean()


def _device(device):
    if isinstance(device, int) or (isinstance(device, str) and str(device).isdigit()):
        return torch.device(f"cuda:{int(device)}")
    return torch.device(device if device not in (None, "") else ("cuda" if torch.cuda.is_available() else "cpu"))


def _names(v):
    if v is None:
        return []
    items = v if isinstance(v, (list, tuple)) else str(v).split(",")
    return [str(x).strip().lower() for x in items if str(x).strip()]


def _floats(v, n, what):
    if isinstance(v, (int, float)):
        vals = [float(v)]
    elif isinstance(v, str):
        vals = [float(x) for x in v.strip("[]()").split(",") if x.strip()]
    else:
        vals = [float(x) for x in v]
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        raise SystemExit(f"{what} needs {n} comma-separated numbers (or one for all), got {v}")
    return vals


def pretrain(model="openobb9x", data=None, val_data=None, teachers=None, epochs=20, imgsz=640, batch=32,
             teacher=None, teacher_imgsz=None, teacher_dfine="x", dino_layers="last", w_dfine=1.0, w_dino=0.5,
             t2_min=0.8, val_limit=300, lr=2e-3, wd=0.05, warmup_epochs=1, workers=8, project="runs/pretrain",
             name="train", exist_ok=False, device=None, limit=None, seed=0, channels_last=True, v6_off=None,
             v9_off=None):
    """Distil frozen teachers into a new core on unlabelled images; returns the weights path (weights/last.pt).

    model         openobb9{s,m,l,x} (two teachers, neck P3-P5) or openobb6{n,s,m,l,x} (DINOv2 only, backbone)
    data          image folder(s) / files / .txt lists, comma-separated (no labels needed)
    val_data      OpenOBB9: held-out image folder(s) for gate T2 (mean cosine per teacher and level, every epoch)
    teachers      OpenOBB9: "dfine,dinov2" (default), "dfine" or "dinov2"
    teacher       DINOv2 size: dinov2_s (default for OpenOBB9), dinov2_b (default for OpenOBB6), dinov2_l
    teacher_imgsz DINOv2 input size (multiple of 14; default ~0.7 x imgsz)
    teacher_dfine D-FINE COCO core size used as teacher: x (default; OpenOBB3-x / OpenOBB8-x core), l, m, s
    dino_layers   "last" (default) or "mid,last": DINOv2 layers used as targets for P4 and P5
    w_dfine       loss weights of the dfine targets at P3, P4, P5 (one number = all)
    w_dino        loss weights of the DINOv2 targets at P4, P5 (one number = both)
    t2_min        gate T2 threshold on the held-out dfine cosine at P3 and P4 (docs/OPENOBB9.md section 6)
    val_limit     at most this many val_data images (sorted order) per evaluation
    """
    from openobb.cli import MODELS, increment_path
    if data is None:
        raise SystemExit("openobb pretrain needs data=<folder(s) of images, comma-separated>")
    arch, size = MODELS.get(str(model).lower(), (None, None))
    if arch not in ("v6", "v9"):
        raise SystemExit("openobb pretrain is for the new cores: model=openobb9s/m/l/x or openobb6n/s/m/l/x")
    random.seed(seed)
    torch.manual_seed(seed)
    dev = _device(device)
    files = _images(data)
    if limit:
        files = files[:int(limit)]
    save_dir = increment_path(project, name, exist_ok)
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)
    common = dict(files=files, dev=dev, save_dir=save_dir, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch),
                  teacher_imgsz=teacher_imgsz, lr=float(lr), wd=float(wd), warmup_epochs=warmup_epochs,
                  workers=int(workers))
    if arch == "v6":
        if val_data is not None:
            raise SystemExit("val_data (gate T2) is implemented for openobb9 only")
        if _names(teachers) not in ([], ["dinov2"]):
            raise SystemExit("OpenOBB6 pretraining has one teacher: teachers=dinov2")
        return _pretrain_v6(size, teacher=teacher or "dinov2_b", v6_off=v6_off, **common)
    val_files = None
    if val_data is not None:
        val_files = _images(val_data)
        if val_limit:
            val_files = val_files[:int(val_limit)]
        overlap = len(set(map(str, val_files)) & set(map(str, files)))
        if overlap:
            print(f"[pretrain] WARNING {overlap} val_data images are also in data: gate T2 must use held-out images",
                  flush=True)
    names = _names(teachers) or ["dfine", "dinov2"]
    bad = [t for t in names if t not in TEACHERS]
    if bad or not names:
        raise SystemExit(f"teachers must be a subset of {','.join(TEACHERS)}, got {teachers}")
    return _pretrain_v9(size, model_name=str(model).lower(), val_files=val_files, teachers=names,
                        teacher=teacher or "dinov2_s", teacher_dfine=str(teacher_dfine).lower(),
                        dino_layers=_names(dino_layers), w_dfine=_floats(w_dfine, 3, "w_dfine"),
                        w_dino=_floats(w_dino, 2, "w_dino"), t2_min=float(t2_min),
                        channels_last=bool(channels_last), v9_off=v9_off, **common)


def _loader(ds, batch, workers, dev, train):
    batch = max(1, min(batch, len(ds))) if train else batch
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers, drop_last=train,
                      pin_memory=dev.type == "cuda", persistent_workers=workers > 0)


def _optimizer(params, lr, wd):
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.999))


def _teacher_size(teacher_imgsz, imgsz):
    t_size = int(teacher_imgsz or round(imgsz * 0.7 / 14) * 14)
    return max(14, t_size // 14 * 14)


# ----------------------------------------------------------------------------------------------------- OpenOBB6
def _pretrain_v6(size, files, dev, save_dir, epochs, imgsz, batch, teacher, teacher_imgsz, lr, wd, warmup_epochs,
                 workers, v6_off):
    from openobb.models.openobb6 import OpenOBB6, parse_off
    from openobb.models.vit import DINOV2, DinoV2Backbone
    student_off = parse_off(v6_off)
    student = OpenOBB6(size, num_classes=1, img_size=imgsz, off=v6_off).backbone.to(dev)
    tch = DinoV2Backbone(teacher).to(dev)
    tch.load_pretrained()
    tch.eval().requires_grad_(False)
    d = DINOV2[teacher][0]
    layers = [tch.layers[1], tch.layers[3], tch.layers[3]]          # middle -> stride 8, last -> 16 and 32
    c = student.channels
    proj = nn.ModuleList([Projector(c[1], d), Projector(c[2], d), Projector(c[3], d)]).to(dev)
    weights = (0.5, 1.0, 0.5)
    t_size = _teacher_size(teacher_imgsz, imgsz)

    params = [p for p in list(student.parameters()) + list(proj.parameters()) if p.requires_grad]
    opt = _optimizer(params, lr, wd)
    dl = _loader(Crops(files, imgsz), batch, workers, dev, True)
    steps = max(1, len(dl)) * epochs
    warm = max(1, len(dl) * warmup_epochs)
    amp = dev.type == "cuda"
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    print(f"[pretrain] OpenOBB6-{size} backbone <- DINOv2 {teacher} | {len(files)} images, {epochs} epochs, "
          f"student {imgsz} px, teacher {t_size} px, batch {dl.batch_size}, {dev} -> {save_dir}", flush=True)
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
        dt = time.time() - t0
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"], "time_s": dt,
               "img_per_s": n * dl.batch_size / max(dt, 1e-9)}
        log.append(rec)
        print(f"[pretrain] epoch {ep + 1}/{epochs}  distill_loss {rec['loss']:.4f}  {dt:.0f}s  "
              f"{rec['img_per_s']:.1f} img/s", flush=True)
        state = {f"backbone.{k}": v for k, v in student.state_dict().items()}
        torch.save({"model": state, "classes": [], "epoch": ep + 1,
                    "args": {"arch": "v6", "size": size, "img": imgsz, "pretrain": "dinov2_distill", "teacher": teacher,
                             "v6_off": ",".join(sorted(student_off))}},
                   save_dir / "weights" / "last.pt")
        (save_dir / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    print(f"[pretrain] done: {save_dir / 'weights' / 'last.pt'}  (use as weights= for openobb train "
          f"model=openobb6{size})", flush=True)
    return str(save_dir / "weights" / "last.pt")


# ----------------------------------------------------------------------------------------------------- OpenOBB9
CORE = ("backbone.", "neck.")
LEVELS = ("P3", "P4", "P5")


def _check_core(net, imgsz, dev):
    """features() must only use backbone.* / neck.* parameters: anything else would be trained here but not saved."""
    was = net.training
    net.eval()
    net.zero_grad(set_to_none=True)
    _, feats = net.features(torch.rand(1, 3, imgsz, imgsz, device=dev))
    sum(f.float().mean() for f in feats).backward()
    used = [n for n, p in net.named_parameters() if p.grad is not None and not n.startswith(CORE)]
    net.zero_grad(set_to_none=True)
    net.train(was)
    if used:
        raise RuntimeError(f"OpenOBB9.features uses parameters outside backbone./neck. ({used[:5]}): they would not "
                           "be saved by pretraining; move them into the core")
    if len(feats) != 3 or any(f.shape[1] != net.hid for f in feats):
        raise RuntimeError(f"OpenOBB9.features must return 3 maps with .hid={net.hid} channels, got "
                           f"{[tuple(f.shape) for f in feats]}")


class _Teachers(nn.Module):
    """Frozen teachers. __call__(x in [0, 1]) -> {"dfine": [P3, P4, P5], "dinov2": [one map per DINOv2 layer]}."""

    def __init__(self, names, imgsz, dfine_size, dino_name, dino_layers, t_size, log=print):
        super().__init__()
        self.dfine = self.vit = None
        self.channels = {}
        if "dfine" in names:
            from openobb.models import openobb1
            from openobb.models.openobb3 import OpenOBB3
            core = OpenOBB3(dfine_size, num_classes=1, img_size=imgsz, lsk=False)
            skipped = openobb1.load_dfine_coco(core, dfine_size, log=log)
            missing = [k for k in (skipped or []) if k.startswith(("backbone.", "encoder."))]
            if missing:                              # never distil from a partly random teacher
                raise RuntimeError(f"D-FINE-{dfine_size} COCO weights do not cover the teacher core: {missing[:5]}")
            self.dfine = nn.ModuleDict({"backbone": core.backbone, "encoder": core.encoder})
            self.channels["dfine"] = int(core.encoder.hidden_dim)        # 384 for x, 256 for s / m / l
        if "dinov2" in names:
            from openobb.models.vit import DINOV2, DinoV2Backbone
            dino = DinoV2Backbone(dino_name)
            dino.load_pretrained()
            self.vit = dino.vit
            d, _, depth, _ = DINOV2[dino_name]
            pick = {"mid": depth // 2 - 1, "last": depth - 1}
            bad = [n for n in dino_layers if n not in pick]
            if bad or not dino_layers:
                raise SystemExit(f"dino_layers must be 'last' or 'mid,last', got {dino_layers}")
            order = sorted(set(dino_layers), key=lambda n: pick[n])      # ViT returns maps in block order
            self.dino_names, self.dino_idx = order, [pick[n] for n in order]
            self.channels["dinov2"] = d
            self.t_size = t_size
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.eval().requires_grad_(False)

    def train(self, mode=True):                      # always frozen, BatchNorm statistics included
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        out = {}
        if self.dfine is not None:                   # same input as OpenOBB3 inference: RGB in [0, 1], no mean / std
            out["dfine"] = self.dfine["encoder"](self.dfine["backbone"](x))
        if self.vit is not None:
            xt = F.interpolate((x - self.mean) / self.std, size=(self.t_size, self.t_size), mode="bilinear",
                               align_corners=False)
            out["dinov2"] = self.vit(xt, self.dino_idx)
        return out


def _pairs(teachers, tch, w_dfine, w_dino):
    """(display name, student level 0..2, teacher, teacher map index, weight) for every distillation target."""
    pairs = []
    if tch.dfine is not None:
        pairs += [(f"dfine/{LEVELS[l]}", l, "dfine", l, w_dfine[l]) for l in range(3)]
    if tch.vit is not None:
        for j, n in enumerate(tch.dino_names):
            tag = "dinov2" if n == "last" else f"dinov2_{n}"
            pairs += [(f"{tag}/{LEVELS[l]}", l, "dinov2", j, w_dino[l - 1]) for l in (1, 2)]
    pairs = [p for p in pairs if p[4] > 0]
    for l in range(3):                               # every level must be distilled (OPENOBB9 section 5)
        if not any(p[1] == l for p in pairs):
            if tch.vit is None:
                raise SystemExit(f"{LEVELS[l]} would get no distillation loss: check w_dfine / teachers")
            j = len(tch.dino_names) - 1
            print(f"[pretrain] {LEVELS[l]} has no target from {','.join(teachers)}: adding DINOv2 last layer "
                  f"(weight 0.5)", flush=True)
            pairs.append((f"dinov2/{LEVELS[l]}", l, "dinov2", j, 0.5))
    return pairs


@torch.no_grad()
def _val_cos(net, proj, pairs, tch, dl, dev, amp, channels_last):
    """Mean cosine per (teacher, level) over all locations of the held-out images (student and projectors in eval)."""
    net.eval()
    proj.eval()
    sums, cnt = {p[0]: 0.0 for p in pairs}, {p[0]: 0 for p in pairs}
    for x in dl:
        x = x.to(dev, non_blocking=True).float().div_(255)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            tf = tch(x)
            _, feats = net.features(x)
            for key, l, t, j, _ in pairs:
                c = _cos_map(proj[key.replace("/", "_")](feats[l]), tf[t][j])
                sums[key] += float(c.sum())
                cnt[key] += c.numel()
    return {k: sums[k] / max(cnt[k], 1) for k in sums}


def _t2(cos, t2_min):
    need = ("dfine/P3", "dfine/P4")
    if cos is None or not all(k in cos for k in need):
        return None
    return all(cos[k] >= t2_min for k in need)


def _pretrain_v9(size, files, dev, save_dir, epochs, imgsz, batch, teacher, teacher_imgsz, lr, wd, warmup_epochs,
                 workers, model_name, val_files, teachers, teacher_dfine, dino_layers, w_dfine, w_dino, t2_min,
                 channels_last, v9_off):
    from openobb.models.openobb9 import OpenOBB9
    amp = dev.type == "cuda"
    cl = channels_last and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    net = OpenOBB9(size, num_classes=1, img_size=imgsz, off=v9_off).to(dev)
    _check_core(net, imgsz, dev)
    core = [p for n, p in net.named_parameters() if n.startswith(CORE)]
    for n, p in net.named_parameters():               # the head is not pretrained (and not saved)
        p.requires_grad_(n.startswith(CORE))
    t_size = _teacher_size(teacher_imgsz, imgsz)
    tch = _Teachers(teachers, imgsz, teacher_dfine, teacher, dino_layers, t_size).to(dev)
    pairs = _pairs(teachers, tch, w_dfine, w_dino)
    proj = nn.ModuleDict({k.replace("/", "_"): Projector(net.hid, tch.channels[t]) for k, _, t, _, _ in pairs}).to(dev)
    if cl:
        net = net.to(memory_format=torch.channels_last)
        tch = tch.to(memory_format=torch.channels_last)
        proj = proj.to(memory_format=torch.channels_last)

    params = core + list(proj.parameters())
    opt = _optimizer(params, lr, wd)
    dl = _loader(Crops(files, imgsz), batch, workers, dev, True)
    vdl = _loader(CenterCrops(val_files, imgsz), dl.batch_size, workers, dev, False) if val_files else None
    steps = max(1, len(dl)) * epochs
    warm = max(1, len(dl) * warmup_epochs)
    desc = []
    if "dfine" in teachers:
        desc.append(f"D-FINE-{teacher_dfine} COCO core ({tch.channels['dfine']} ch, {imgsz} px)")
    if "dinov2" in teachers:
        desc.append(f"DINOv2 {teacher} {'+'.join(tch.dino_names)} ({tch.channels['dinov2']} ch, {t_size} px)")
    print(f"[pretrain] OpenOBB9-{size} core (backbone + neck, P3/P4/P5 {net.hid} ch) <- {' + '.join(desc)}", flush=True)
    print(f"[pretrain] targets: {', '.join(f'{k} x{w:g}' for k, _, _, _, w in pairs)}", flush=True)
    print(f"[pretrain] {len(files)} images, {epochs} epochs, {imgsz} px, batch {dl.batch_size}, {len(dl)} steps/epoch, "
          f"{dev}" + (f", gate T2 on {len(val_files)} held-out images" if val_files else ", no val_data (no gate T2)")
          + f" -> {save_dir}", flush=True)
    log, it = [], 0
    for ep in range(epochs):
        net.train()
        proj.train()
        t0, tot, n = time.time(), 0.0, 0
        part = {p[0]: 0.0 for p in pairs}
        for x in dl:
            x = x.to(dev, non_blocking=True).float().div_(255)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (it + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(it / steps, 1.0))))
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                tf = tch(x)
                _, feats = net.features(x)
                terms = {k: _cos_loss(proj[k.replace("/", "_")](feats[l]), tf[t][j]) for k, l, t, j, _ in pairs}
                loss = sum(w * terms[k] for k, _, _, _, w in pairs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.detach())
            for k in part:
                part[k] += float(terms[k].detach())
            n += 1
            it += 1
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dt = time.time() - t0
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "loss_pairs": {k: v / max(n, 1) for k, v in part.items()},
               "lr": opt.param_groups[0]["lr"], "time_s": dt, "img_per_s": n * dl.batch_size / max(dt, 1e-9)}
        cos, vt = None, 0.0
        if vdl is not None:
            v0 = time.time()
            cos = _val_cos(net, proj, pairs, tch, vdl, dev, amp, cl)
            vt = time.time() - v0
            rec.update(val_cos=cos, val_time_s=vt, t2_pass=_t2(cos, t2_min))
        log.append(rec)
        eta = (epochs - ep - 1) * (dt + vt) / 60
        print(f"[pretrain] epoch {ep + 1}/{epochs}  loss {rec['loss']:.4f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in rec["loss_pairs"].items())
              + f"  | {rec['img_per_s']:.1f} img/s  train {dt:.0f}s  val {vt:.0f}s  ETA {eta:.0f} min", flush=True)
        if cos is not None:
            g = rec["t2_pass"]
            print("[pretrain]   held-out cosine  " + "  ".join(f"{k} {v:.3f}" for k, v in cos.items())
                  + ("" if g is None else f"  | T2 (dfine P3, P4 >= {t2_min:g}): {'pass' if g else 'not yet'}"),
                  flush=True)
        state = {k: v.detach().cpu() for k, v in net.state_dict().items() if k.startswith(CORE)}
        torch.save({"model": state, "classes": [], "epoch": ep + 1, "pretrain_val_cos": cos,
                    "args": {"arch": "v9", "size": size, "img": imgsz, "pretrain": "distill", "teachers": teachers,
                             "teacher": teacher if "dinov2" in teachers else None,
                             "teacher_dfine": teacher_dfine if "dfine" in teachers else None,
                             "v9_off": v9_off if isinstance(v9_off, str) else ",".join(sorted(v9_off or []))}},
                   save_dir / "weights" / "last.pt")
        (save_dir / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    last = log[-1] if log else {}
    if "val_cos" in last:
        g = last["t2_pass"]
        if g is None:
            print("[pretrain] gate T2: not applicable (needs the dfine teacher)", flush=True)
        else:
            c = last["val_cos"]
            print(f"[pretrain] gate T2 (docs/OPENOBB9.md section 6): held-out dfine cosine P3 {c['dfine/P3']:.3f}, "
                  f"P4 {c['dfine/P4']:.3f} vs {t2_min:g} -> {'PASSED' if g else 'FAILED'}", flush=True)
    else:
        print("[pretrain] gate T2: not measured (give val_data=<held-out drawings>)", flush=True)
    w = save_dir / "weights" / "last.pt"
    print(f"[pretrain] done: {w}  (openobb train model={model_name} weights={w} ...)", flush=True)
    return str(w)
