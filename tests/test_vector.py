import json
import math
import random

import cv2
import numpy as np
import torch

from openobb.data.dota import DotaPatches, collate
from openobb.engine import ctx_batch
from openobb.models.openobb1 import OpenOBB1

torch.set_num_threads(1)
S = 256


def _split(root):
    """One drawing: a rotated rectangle object drawn by 4 line primitives (+ one far-away stray line)."""
    for d in ("images/val", "vectors/val", "meta"):
        (root / d).mkdir(parents=True)
    rect = ((100.0, 120.0), (90.0, 30.0), 25.0)
    corners = cv2.boxPoints(rect)
    img = np.full((S, S, 3), 255, np.uint8)
    cv2.polylines(img, [corners.astype(np.int32)], True, (0, 0, 0), 2)
    cv2.imwrite(str(root / "images/val/d0.png"), img)
    rows = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        pts = np.linspace(a, b, 8) / S
        rows.append(np.r_[0, pts.reshape(-1), 0, 0, 0, 0.5, 7, 0])
    rows.append(np.r_[1, (np.linspace([230, 20], [250, 40], 8) / S).reshape(-1), 1, 0, 0, 0.5, 0, 0])
    np.savez_compressed(root / "vectors/val/d0.npz", tokens=np.array(rows, np.float32), view=np.float32(S))
    meta = {"name": "d0", "file": "d0.png", "src": "d0", "rate": 1.0, "x0": 0, "y0": 0, "w": S, "h": S,
            "img_w": S, "img_h": S, "objs": [[0, 0, 0] + corners.reshape(-1).tolist()]}
    (root / "meta/val.jsonl").write_text(json.dumps(meta) + "\n")


def _inside(pt, box, tol=3.0):
    cx, cy, w, h, t = box
    dx, dy = pt[0] - cx, pt[1] - cy
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    return abs(u) <= w / 2 + tol and abs(v) <= h / 2 + tol


def test_vector_points_follow_every_augmentation(tmp_path):
    _split(tmp_path)
    random.seed(0)
    np.random.seed(0)
    for kw in (dict(augment=False), dict(augment=True), dict(augment=True, rotate_p=1.0),
               dict(augment=True, mosaic_p=1.0), dict(augment=True, scale_jitter=0.5, translate=0.1),
               dict(augment=True, mosaic_p=1.0, mosaic_mode="yolo", scale_jitter=0.5, translate=0.1)):
        ds = DotaPatches(tmp_path, "val", size=S, hsv=(0, 0, 0), vectors=True, **kw)
        for _ in range(12):
            _, t = ds[0]
            vec, boxes = t["vec"].numpy(), t["boxes"].numpy().copy()
            assert vec.shape[1] == 22
            boxes[:, :4] *= S
            lines = vec[vec[:, 0] == 0]
            moved = kw.get("rotate_p") or kw.get("scale_jitter")         # primitives may leave the image
            assert len(lines) == 4 * max(1, len(boxes)) or moved, (kw, len(lines), len(boxes))
            segs = [row[1:17].reshape(8, 2) * S for row in lines]
            if moved:       # truncated objects (iof < 0.7) are dropped while their strokes stay: check boxes -> strokes
                for b in boxes:
                    assert any(all(_inside(p, b) for p in pts) for pts in segs), (kw, b)
                continue
            for pts in segs:
                assert any(all(_inside(p, b) for p in pts) for b in boxes), (kw, pts, boxes)
            if not kw.get("scale_jitter"):
                want = 0.25 / S if kw.get("mosaic_p") else 0.5 / S    # line width halves with the mosaic scale
                assert np.allclose(lines[:, 20], want, atol=1e-7), (kw, lines[:, 20])


def test_vector_branch_identity_at_init_and_trains():
    torch.manual_seed(0)
    m = OpenOBB1("s", img_size=256, vectors=True, num_denoising=0).eval()
    x = torch.rand(2, 3, 256, 256)
    vec = torch.rand(2, 50, 21)
    vec[..., 0] = torch.randint(0, 4, (2, 50)).float()
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[1, 30:] = False
    side = {"vec": vec, "vec_mask": mask}
    with torch.no_grad():
        a = m(x)["pred_boxes"]
        b = m(x, ctx=side)["pred_boxes"]
    assert torch.allclose(a, b, atol=1e-6)
    m.train()
    tg = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])}] * 2
    out = m(x, tg, ctx=side)
    out["pred_logits"].sum().backward()          # D-FINE zero-inits the last box-distribution layer: use logits
    g = m.vector.out[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # an image without any primitive must not break attention (register token keeps every row valid)
    side0 = {"vec": torch.zeros(2, 0, 21), "vec_mask": torch.zeros(2, 0, dtype=torch.bool)}
    assert torch.isfinite(m(x, tg, ctx=side0)["pred_boxes"]).all()


def test_ctx_batch_pads_vectors(tmp_path):
    _split(tmp_path)
    ds = DotaPatches(tmp_path, "val", size=S, vectors=True)
    imgs, tg = collate([ds[0], ds[0]])
    tg[1]["vec"] = tg[1]["vec"][:2]
    c = ctx_batch(tg, torch.device("cpu"))
    assert c["vec"].shape == (2, 5, 22) and c["vec_mask"].sum().item() == 7 and "thumb" not in c
