"""Does the score know how tight the box is? For OpenOBB (raw candidates -> NMS as calibrated) and YOLO26x (its
detections), on the same Wall_Color val objects, detections with conf >= 0.25: per class, the Spearman correlation of
score and ProbIoU to the best same-class label among matched detections (ProbIoU >= 0.5), and the mAP50-95 change when
the score is multiplied by that ProbIoU (unmatched detections: by the median matched ProbIoU) -> how much a perfect
localisation-quality factor would add on top of each model's own score. CPU.

usage: python score_iou_corr.py <dup_raw_<model>_53.npz> <yolo_dets npz> [nms_iou=0.5]
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
raw = dict(np.load(sys.argv[1], allow_pickle=True))
yolo = dict(np.load(sys.argv[2], allow_pickle=True))
NMS_IOU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


def spearman(a, b):
    if len(a) < 5:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


gts, models = {}, {"openobb": {}, "yolo": {}}
for stem, rec in sorted(raw.items()):
    rec = rec.item()
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    sc, cl, b = rec["raw"]
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6)).float()
    L, B = fast_nms(logits, torch.as_tensor(b)[None].float(), max_det=3000, conf=0.25, iou=NMS_IOU)
    s2, c2 = L[0].sigmoid().max(-1)
    models["openobb"][stem] = [(int(c), float(s), canon(x)) for c, s, x in
                               zip(c2.numpy(), s2.numpy(), B[0].numpy().astype(np.float64)) if s >= 0.25]
    models["yolo"][stem] = [(int(c), float(s), canon(np.asarray(x, np.float64))) for c, s, x in yolo[stem] if s >= 0.25]

for name, dets in models.items():
    rows = {c: ([], []) for c in NAMES}
    out = {v: {c: ([], [], []) for c in NAMES} for v in ("base", "s*q")}
    for stem, ds in dets.items():
        gb = np.array([gt_box(p) for p, _, _ in gts[stem]]) if gts[stem] else np.zeros((0, 5))
        gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
        pb = np.array([d[2] for d in ds]).reshape(-1, 5)
        iou = probiou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
        best = np.array([(iou[:, j] * (gc == c)).max() if len(gb) else 0.0 for j, (c, _, _) in enumerate(ds)])
        q_med = float(np.median(best[best >= 0.1])) if (best >= 0.1).any() else 0.5
        for j, (c, s, box) in enumerate(ds):
            if best[j] >= 0.5:
                rows[NAMES[c]][0].append(s), rows[NAMES[c]][1].append(best[j])
            for v, sc in (("base", s), ("s*q", s * (best[j] if best[j] >= 0.1 else q_med))):
                ids, scs, pls = out[v][NAMES[c]]
                ids.append(stem), scs.append(sc), pls.append(obb2poly(box[None])[0])
    res = {v: evaluate_ultra({c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in out[v].items()},
                             gts, NAMES, max_det=3000) for v in out}
    print(f"\n{name}: conf >= 0.25 | Spearman(score, ProbIoU) over matched (>= 0.5) | AP50-95 base -> x ProbIoU")
    for c in NAMES:
        s, q = rows[c]
        b0, b1 = res["base"]["classes"][c]["AP50_95"], res["s*q"]["classes"][c]["AP50_95"]
        print(f"  {c:12s} n {len(s):5d}  rho {spearman(np.array(s), np.array(q)):+.3f}   {b0:.3f} -> {b1:.3f} ({b1 - b0:+.3f})")
    allr = [x for c in NAMES for x in zip(*rows[c])]
    print(f"  {'all':12s} n {len(allr):5d}  rho {spearman(np.array([a for a, _ in allr]), np.array([b for _, b in allr])):+.3f}"
          f"   {res['base']['mAP50_95']:.3f} -> {res['s*q']['mAP50_95']:.3f} ({res['s*q']['mAP50_95'] - res['base']['mAP50_95']:+.3f})")
