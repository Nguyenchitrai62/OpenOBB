"""Drawing-level geometry consensus (no training): the label thickness of wall / wall_300 / door and the size of
junctions is almost constant WITHIN a drawing (median deviation from the image's own median: junction 0.20 px, door
0.43, wall_300 0.44, wall 0.58 at the 1280 scale) but varies across drawings (1-5 px). Every detector box is predicted
on its own, so this per-drawing convention is noise for it. Here, after NMS, each box's short side (and, for near-square
classes, both sides) is pulled towards the median of the CONFIDENT detections (score >= SEED) of the same class in the
same image: new = (1 - lam) x own + lam x median. Ceiling ("gt"): the same with the median of the image's labels.
Scored with the YOLO-style evaluator on the raw candidates of dup_analysis.py. CPU, Wall_Color val.

usage: python consensus_probe.py <dup_raw_<model>_53.npz> [nms_iou=0.5]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import evaluate_ultra  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
SQUARE = {"junction"}                          # both sides follow the consensus
THIN = {"wall", "wall_300", "door", "slide_door", "double_door", "junction"}
SEED = 0.4
raw_path = sys.argv[1]
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
VARIANTS = ["none", "lam0.5", "lam0.8", "lam1.0", "gt1.0"]


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


raw = dict(np.load(raw_path, allow_pickle=True))
gts = {}
dets = {v: {c: ([], [], []) for c in NAMES} for v in VARIANTS}
for k, (stem, rec) in enumerate(sorted(raw.items())):
    rec = rec.item()
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    gb = np.array([gt_box(p) for p, _, _ in gts[stem]]) if gts[stem] else np.zeros((0, 5))
    gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
    sc, cl, b = rec["raw"]
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6)).float()
    L, B = fast_nms(logits, torch.as_tensor(b)[None].float(), max_det=3000, conf=0.05, iou=NMS_IOU)
    s2, c2 = L[0].sigmoid().max(-1)
    s2, c2 = s2.numpy().astype(np.float64), c2.numpy()
    pb = np.array([canon(x) for x in B[0].numpy().astype(np.float64)]).reshape(-1, 5)
    med, gmed = {}, {}
    for ci, n in enumerate(NAMES):
        sel = (c2 == ci) & (s2 >= SEED)
        if sel.sum() >= 3:
            med[ci] = (np.median(pb[sel, 2]), np.median(pb[sel, 3]))
        g = gc == ci
        if g.sum() >= 3:
            gmed[ci] = (np.median(gb[g, 2]), np.median(gb[g, 3]))
    for j, (c, s) in enumerate(zip(c2, s2)):
        n = NAMES[c]
        for var in VARIANTS:
            box = pb[j].copy()
            ref = gmed if var.startswith("gt") else med
            lam = float(var[3:] if var.startswith("lam") else var[2:]) if var != "none" else 0.0
            if lam and n in THIN and c in ref:
                box[3] = (1 - lam) * box[3] + lam * ref[c][1]
                if n in SQUARE:
                    box[2] = (1 - lam) * box[2] + lam * ref[c][0]
            ids, scs, pls = dets[var][n]
            ids.append(stem), scs.append(float(s)), pls.append(obb2poly(box[None])[0])
    print(f"{k + 1}/{len(raw)}", end="\r", flush=True)

res = {v: evaluate_ultra({c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in dets[v].items()},
                         gts, NAMES, max_det=3000) for v in VARIANTS}
print(f"\n{Path(raw_path).name}, NMS {NMS_IOU}, seeds score >= {SEED}: delta mAP50-95 (base absolute)")
print(f"{'class':12s} " + " ".join(f"{v:>8s}" for v in VARIANTS))
for c in NAMES:
    base = res["none"]["classes"][c]["AP50_95"]
    print(f"{c:12s} {base:8.3f} " + " ".join(f"{res[v]['classes'][c]['AP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
base = res["none"]["mAP50_95"]
print(f"{'all':12s} {base:8.3f} " + " ".join(f"{res[v]['mAP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
print(f"{'all mAP50':12s} {res['none']['mAP50']:8.3f} " + " ".join(f"{res[v]['mAP50'] - res['none']['mAP50']:+8.3f}"
                                                         for v in VARIANTS[1:]))
