"""Ceiling test for "box sides snapped to image edges" on Wall_Color val (CPU, no training, no learning).

Two measurements, both at full image resolution (boxes are kept in the 1280-scale frame of the labels):
  1. Label alignment: snap every GROUND-TRUTH box side to the strongest nearby edge and report the IoU with the original.
     IoU ~1 -> labels sit on image edges, an edge-evidence box decoder has a real target; IoU << 1 -> labels are drawn
     off the strokes and no edge-based mechanism can tighten boxes beyond what the labels allow.
  2. Detections: snap the sides of a trained model's detections (dup_raw_*.npz from dup_analysis.py) and re-score with
     the YOLO-style evaluator (openobb.eval.ultra). The gain is a lower bound of what a learned edge decoder could do.

usage: python edge_snap_oracle.py <dup_raw_<model>_53.npz> [nms_iou=0.7] [radius_px_at_1280=2.0]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import evaluate_ultra  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color"
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw_path = sys.argv[1]
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
RAD = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0      # search radius, px at the 1280 scale
STEP = 0.125                                                 # profile step, px at the 1280 scale


def to_xywhr(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return np.array([cx, cy, w, h, np.deg2rad(a)], np.float64)


def snap(img, s, b, rad=RAD):
    """img: full-resolution float image (H, W, 3); s: full px per 1280 px; b: (cx, cy, w, h, th) at 1280 scale.
    Each of the 4 sides moves along its normal to the offset with the strongest colour gradient of the profile averaged
    over the central 80% of the side, with a mild prior toward no move. Returns the snapped box."""
    cx, cy, w, h, th = b
    u = np.array([np.cos(th), np.sin(th)])      # along w
    v = np.array([-u[1], u[0]])                 # along h
    r = min(rad, 0.35 * min(w, h))              # never let a side cross the box centre on tiny objects
    if r < 0.5:
        return b
    t = np.arange(-r - 1, r + 1 + 1e-9, STEP)
    moves = []
    for normal, half, tang, tlen in ((u, w / 2, v, h), (-u, w / 2, v, h), (v, h / 2, u, w), (-v, h / 2, u, w)):
        along = np.linspace(-0.4, 0.4, 9) * tlen
        base = np.array([cx, cy]) + normal * half
        pts = base[None, None] + along[:, None, None] * tang[None, None] + t[None, :, None] * normal[None, None]
        mx = (pts[..., 0] * s).astype(np.float32)
        my = (pts[..., 1] * s).astype(np.float32)
        prof = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(0)   # (T, 3)
        g = np.abs(np.gradient(prof, axis=0)).sum(-1)
        inner = (t >= -r) & (t <= r)
        score = g * np.exp(-0.5 * (t / max(r, 1e-6)) ** 2 * 0.5)
        score[~inner] = -1
        k = int(np.argmax(score))
        moves.append(t[k] if g[k] > 4.0 * STEP else 0.0)   # flat profile (no edge) -> keep the side
    a, b2, c, d = moves                                      # +u side, -u side, +v side, -v side (outward positive)
    w2, h2 = max(w + a + b2, 0.5), max(h + c + d, 0.5)
    ctr = np.array([cx, cy]) + u * (a - b2) / 2 + v * (c - d) / 2
    return np.array([ctr[0], ctr[1], w2, h2, th])


def poly(b):
    return obb2poly(np.asarray(b, np.float64)[None])[0]


def iou_pair(a, b):
    from shapely.geometry import Polygon
    pa, pb = Polygon(poly(a).reshape(4, 2)).buffer(0), Polygon(poly(b).reshape(4, 2)).buffer(0)
    return pa.intersection(pb).area / max(pa.union(pb).area, 1e-9)


raw = dict(np.load(raw_path, allow_pickle=True))
imgs = {p.stem: p for p in (ROOT / "images" / "val").iterdir()}
gts = {}
dets_before = {c: ([], [], []) for c in NAMES}
dets_after = {c: ([], [], []) for c in NAMES}
gt_iou = {c: [] for c in NAMES}
for k, (stem, rec) in enumerate(sorted(raw.items())):
    rec = rec.item()
    im = imread(str(imgs[stem])).astype(np.float32)
    s = max(im.shape[:2]) / 1280.0
    gts[stem] = [(np.asarray(pts, np.float64), n, 0) for pts, n, _ in rec["gt"]]
    for pts, cn, _ in gts[stem]:
        g = to_xywhr(pts)
        gt_iou[cn].append(iou_pair(g, snap(im, s, g)))
    sc, cl, b = rec["raw"]
    if len(sc):
        logits = torch.zeros(1, len(sc), len(NAMES)) - 30
        logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
            torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
        L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=NMS_IOU)
        s2, c2 = L[0].sigmoid().max(-1)
        for c, p, bb in zip(c2.numpy(), s2.numpy(), B[0].numpy().astype(np.float64)):
            for store, box in ((dets_before, bb), (dets_after, snap(im, s, bb))):
                ids, scs, pls = store[NAMES[c]]
                ids.append(stem), scs.append(float(p)), pls.append(poly(box))
    print(f"{k + 1}/{len(raw)}", end="\r", flush=True)


def pack(d):
    return {c: (ids, np.asarray(sc), np.asarray(pl).reshape(-1, 8)) for c, (ids, sc, pl) in d.items()}


A = evaluate_ultra(pack(dets_before), gts, NAMES, max_det=3000)
Bm = evaluate_ultra(pack(dets_after), gts, NAMES, max_det=3000)
print(f"\nmodel {Path(raw_path).name}, NMS {NMS_IOU}, search radius {RAD} px at 1280 (x{s:.2f} sampled at full res)")
print(f"{'class':12s} | {'GT snap IoU med / p10':>22s} | {'before 50 / 50-95':>18s} | {'snapped 50 / 50-95':>18s} | {'d50-95':>7s}")
for c in NAMES:
    gi = np.array(gt_iou[c])
    a, b = A["classes"][c], Bm["classes"][c]
    print(f"{c:12s} | {np.median(gi):10.3f} / {np.percentile(gi, 10):9.3f} | {a['AP50']:7.3f} / {a['AP50_95']:7.3f}  |"
          f" {b['AP50']:7.3f} / {b['AP50_95']:7.3f}  | {b['AP50_95'] - a['AP50_95']:+.3f}")
print(f"{'all':12s} | {'':22s} | {A['mAP50']:7.3f} / {A['mAP50_95']:7.3f}  | {Bm['mAP50']:7.3f} / {Bm['mAP50_95']:7.3f}  |"
      f" {Bm['mAP50_95'] - A['mAP50_95']:+.3f}")
