"""Which part of the box costs mAP50-95? For every detection of a trained model, take its best same-class ground truth
(IoU >= 0.1) and replace ONE component of the detection, measured in the ground truth's own frame, by the ground
truth's value; re-score with the YOLO-style evaluator. The gain of each variant is the ceiling of a mechanism that fixes
that component only (CPU, Wall_Color val).

Components: thick (short side), across (centre offset across the axis), ends (both extents along the axis),
angle, and all four (= perfect boxes on the same detections / scores: the classification + ranking ceiling).

usage: python box_error_oracle.py <dup_raw_<model>_53.npz> [nms_iou=0.7]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import evaluate_ultra, probiou  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw_path = sys.argv[1]
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
VARIANTS = ("none", "thick", "across", "ends", "angle", "all")


def canon(b):
    """(cx, cy, w, h, th) with w >= h."""
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


def fix(p, g, what):
    """Replace component `what` of prediction p by the one of ground truth g (both canonical)."""
    if what == "none":
        return p
    if what == "all":
        return g.copy()
    u = np.array([np.cos(g[4]), np.sin(g[4])])
    v = np.array([-u[1], u[0]])
    dth = (p[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2
    d = p[:2] - g[:2]
    along, across, L, T = d @ u, d @ v, p[2], p[3]
    if what == "thick":
        T = g[3]
    elif what == "across":
        across = 0.0
    elif what == "ends":
        along, L = 0.0, g[2]
    elif what == "angle":
        dth = 0.0
    c = g[:2] + u * along + v * across
    return np.array([c[0], c[1], L, T, g[4] + dth])


raw = dict(np.load(raw_path, allow_pickle=True))
gts = {}
dets = {v: {c: ([], [], []) for c in NAMES} for v in VARIANTS}
for k, (stem, rec) in enumerate(sorted(raw.items())):
    rec = rec.item()
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    gb = np.array([gt_box(p) for p, _, _ in gts[stem]]) if gts[stem] else np.zeros((0, 5))
    gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
    sc, cl, b = rec["raw"]
    if not len(sc):
        continue
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
    L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=NMS_IOU)
    s2, c2 = L[0].sigmoid().max(-1)
    pb = np.array([canon(x) for x in B[0].numpy().astype(np.float64)]).reshape(-1, 5)
    iou = probiou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
    for j, (c, s) in enumerate(zip(c2.numpy(), s2.numpy())):
        m = iou[:, j] * (gc == c)
        gi = int(np.argmax(m)) if len(m) else -1
        for var in VARIANTS:
            box = fix(pb[j], gb[gi], var) if gi >= 0 and m[gi] >= 0.1 else pb[j]
            ids, scs, pls = dets[var][NAMES[c]]
            ids.append(stem), scs.append(float(s)), pls.append(obb2poly(box[None])[0])
    print(f"{k + 1}/{len(raw)}", end="\r", flush=True)

res = {}
for var in VARIANTS:
    d = {c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in dets[var].items()}
    res[var] = evaluate_ultra(d, gts, NAMES, max_det=3000)
print(f"\n{Path(raw_path).name}, NMS {NMS_IOU}: mAP50-95 when ONE component of every matched detection is set to the GT's")
print(f"{'class':12s} " + " ".join(f"{v:>8s}" for v in VARIANTS))
for c in NAMES:
    base = res["none"]["classes"][c]["AP50_95"]
    print(f"{c:12s} {base:8.3f} " + " ".join(f"{res[v]['classes'][c]['AP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
base = res["none"]["mAP50_95"]
print(f"{'all':12s} {base:8.3f} " + " ".join(f"{res[v]['mAP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
print(f"{'all mAP50':12s} {res['none']['mAP50']:8.3f} " + " ".join(f"{res[v]['mAP50'] - res['none']['mAP50']:+8.3f}"
                                                         for v in VARIANTS[1:]))
