import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vrdet.data.dota import DotaPatches
from vrdet.engine import ctx_batch
from vrdet.models.vrdet import VRDet

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def _raw(root, W=2100, H=1500):
    (root / "val" / "images").mkdir(parents=True)
    (root / "val" / "labelTxt").mkdir(parents=True)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([(xx * 255 / W), (yy * 255 / H), ((xx + yy) % 400) * 255 / 400], -1).astype(np.uint8)
    for k in range(12):
        cv2.circle(img, (150 + 160 * k, 200 + 90 * (k % 7)), 60, (255 - 20 * k, 20 * k, 128), -1)
    cv2.imwrite(str(root / "val" / "images" / "P9.png"), img)
    (root / "val" / "labelTxt" / "P9.txt").write_text("100 100 200 100 200 150 100 150 plane 0\n")


def test_context_crop_matches_patch(tmp_path):
    _raw(tmp_path / "raw")
    subprocess.run([sys.executable, str(ROOT / "colab/data/split_dota.py"), "--src", str(tmp_path / "raw"), "--out",
                    str(tmp_path / "split"), "--splits", "val", "--workers", "1"], check=True)
    random.seed(1)
    np.random.seed(1)
    for kw in (dict(augment=False), dict(augment=True, hsv=(0, 0, 0)), dict(augment=True, hsv=(0, 0, 0), rotate_p=1.0)):
        ds = DotaPatches(tmp_path / "split", "val", context=True, **kw)
        for i in range(len(ds)):
            img, t = ds[i]
            assert t["ctx_valid"]
            x0, y0, x1, y1 = [int(round(v)) for v in t["tile"].tolist()]
            th = t["thumb"].float()
            crop = th[:, max(y0, 0):y1, max(x0, 0):x1]
            small = F.interpolate(img[None].float(), size=(y1 - y0, x1 - x0), mode="area")[0]
            small = small[:, max(0, -y0):, max(0, -x0):][:, :crop.shape[1], :crop.shape[2]]
            h, w = crop.shape[1:]
            c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))      # centre: rotation corners differ
            diff = (crop[:, c[0], c[1]] - small[:, c[0], c[1]]).abs().mean().item()
            assert diff < 12, (kw, i, diff)


def test_context_is_identity_at_init_and_trains(tmp_path):
    torch.manual_seed(0)
    m = VRDet("s", img_size=256, context=True, num_denoising=0).eval()
    x = torch.rand(2, 3, 256, 256)
    ctx = {"thumb": torch.rand(2, 3, 128, 128), "tile": torch.tensor([[0., 0., 64., 64.], [32., 32., 96., 96.]]),
           "valid": torch.tensor([True, False])}
    with torch.no_grad():
        a = m(x)["pred_boxes"]
        b = m(x, ctx=ctx)["pred_boxes"]
    assert torch.allclose(a, b, atol=1e-6)
    m.train()
    out = m(x, [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])}] * 2, ctx=ctx)
    out["pred_boxes"].sum().backward()
    assert m.global_ctx.out.weight.grad is not None and torch.isfinite(m.global_ctx.out.weight.grad).all()


def test_ctx_batch():
    t = [{"thumb": torch.zeros(3, 8, 8, dtype=torch.uint8), "tile": torch.zeros(4), "ctx_valid": True}] * 3
    c = ctx_batch(t, torch.device("cpu"))
    assert c["thumb"].shape == (3, 3, 8, 8) and c["valid"].all()
