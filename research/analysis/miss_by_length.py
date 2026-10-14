"""Which labelled objects does a trained model miss, by length and thickness bucket (Wall_Color val, CPU)?
Uses the raw candidates saved by dup_analysis.py (dup_raw_<model>_53.npz) and, optionally, YOLO26x detections
(yolo_dets_wall_color_53.npz from box_quality.py) for the same objects. A miss = no same-class detection with
score >= 0.25 (after NMS 0.7) at ProbIoU >= 0.5. Lengths / thicknesses in px at the 1280 scale.

usage: python miss_by_length.py <dup_raw_<model>_53.npz> [yolo_dets npz]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import probiou  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw = dict(np.load(sys.argv[1], allow_pickle=True))
yolo = dict(np.load(sys.argv[2], allow_pickle=True)) if len(sys.argv) > 2 else {}
L_BUCKETS = [0, 12, 24, 48, 96, 1e9]


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    th = np.deg2rad(a)
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2])


def ours(rec):
    sc, cl, b = rec["raw"]
    if not len(sc):
        return np.zeros(0), np.zeros(0, int), np.zeros((0, 5))
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
    L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=0.7)
    s2, c2 = L[0].sigmoid().max(-1)
    k = s2 >= 0.25
    return s2[k].numpy(), c2[k].numpy(), B[0][k].numpy().astype(np.float64)


stats = {}
for stem, rec in raw.items():
    rec = rec.item()
    g = rec["gt"]
    if not g:
        continue
    gb = np.array([gt_box(p) for p, _, _ in g])
    gc = np.array([NAMES.index(n) for _, n, _ in g])
    dets = {"ours": ours(rec)}
    if stem in yolo:
        y = list(yolo[stem])
        dets["yolo"] = (np.array([t[1] for t in y]), np.array([t[0] for t in y], int),
                        np.array([t[2] for t in y], np.float64).reshape(-1, 5))
    for name, (s, c, b) in dets.items():
        hit = np.zeros(len(g), bool)
        if len(b):
            iou = probiou(gb, b)
            hit = ((iou >= 0.5) & (gc[:, None] == c[None, :])).any(1)
        # neighbour: another labelled object within 2 px of the object's box (touching)
        for j in range(len(g)):
            n_ = NAMES[gc[j]]
            Lb = np.digitize(gb[j, 2], L_BUCKETS) - 1
            key = (n_, Lb, name)
            st = stats.setdefault(key, [0, 0])
            st[0] += 1
            st[1] += int(hit[j])

print("recall at ProbIoU 0.5 (conf 0.25) by object length (px at 1280): ours | yolo  (n)")
for n_ in ("wall", "wall_300", "junction", "slide_door"):
    print(n_)
    for Lb in range(len(L_BUCKETS) - 1):
        o = stats.get((n_, Lb, "ours"))
        y = stats.get((n_, Lb, "yolo"))
        if not o:
            continue
        rng = f"{L_BUCKETS[Lb]:.0f}-{L_BUCKETS[Lb + 1]:.0f}" if L_BUCKETS[Lb + 1] < 1e8 else f">{L_BUCKETS[Lb]:.0f}"
        print(f"  L {rng:>8s}: {o[1] / o[0]:.2f} | {(y[1] / y[0]) if y else float('nan'):.2f}   (n {o[0]})")
