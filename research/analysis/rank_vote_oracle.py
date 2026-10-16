"""After the box error decomposition (box_error_oracle.py): is the remaining mAP50-95 gap in the SCORES (a tight box
ranked below a loose one) or in the BOX each NMS cluster keeps (the cluster holds a tighter candidate)? CPU, Wall_Color val.

Variants on the model's raw candidates (dup_raw_<model>_53.npz from dup_analysis.py), NMS as calibrated:
  rank=iou      every kept detection matched to a same-class label (ProbIoU >= 0.1) is re-scored with that ProbIoU;
                unmatched detections keep their score (ceiling of a perfect localisation-quality score)
  rank=s*iou    score x ProbIoU (a quality head that multiplies the class score); unmatched keep their score
  rank=s*q      as s*iou, but unmatched detections are multiplied by the median matched ProbIoU (fair: a quality head
                does not know which boxes are false)
  vote t / p    no labels: each kept box is replaced by the score^p-weighted mean of the raw candidates of its class with
                ProbIoU >= t to it (centre, extents in the kept box's frame, double-angle mean); scores unchanged
  pick          ceiling of choosing, inside each cluster (ProbIoU >= 0.6), the candidate closest to the label

usage: python rank_vote_oracle.py <dup_raw_<model>_53.npz> [nms_iou=0.5]
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
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
VOTES = [(0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1), (0.6, 4), (0.7, 4)]
VARIANTS = ["none", "rank=iou", "rank=s*iou", "rank=s*q"] + [f"vote {t} p{p}" for t, p in VOTES] + ["pick"]


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


def vote(k, cands, wts):
    """Weighted mean of candidate boxes expressed in the frame of the kept box k (all canonical)."""
    out, acc = np.zeros(4), np.zeros(2)
    for c, wt in zip(cands, wts):
        d = (c[4] - k[4] + np.pi / 2) % np.pi - np.pi / 2
        w, h = c[2], c[3]
        if abs(d) > np.pi / 4:                       # the candidate's long side is the kept box's short side
            w, h, d = h, w, (d + np.pi) % np.pi - np.pi / 2
        out += wt * np.array([c[0], c[1], w, h])
        acc += wt * np.array([np.cos(2 * d), np.sin(2 * d)])
    out /= wts.sum()
    return np.array([out[0], out[1], out[2], out[3], k[4] + 0.5 * np.arctan2(acc[1], acc[0])])


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
    sc, cl = np.asarray(sc, np.float64), np.asarray(cl, int)
    rb = np.array([canon(x) for x in np.asarray(b, np.float64)]).reshape(-1, 5)
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6)).float()
    L, B = fast_nms(logits, torch.as_tensor(b)[None].float(), max_det=3000, conf=0.05, iou=NMS_IOU)
    s2, c2 = L[0].sigmoid().max(-1)
    pb = np.array([canon(x) for x in B[0].numpy().astype(np.float64)]).reshape(-1, 5)
    iou = probiou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
    keep_raw = probiou(pb, rb) if len(pb) else np.zeros((0, len(rb)))          # kept x raw
    iou_raw_gt = probiou(gb, rb) if len(gb) else np.zeros((0, len(rb)))
    best = np.array([(iou[:, j] * (gc == c)).max() if len(gb) else 0.0 for j, c in enumerate(c2.numpy())])
    q_med = float(np.median(best[best >= 0.1])) if (best >= 0.1).any() else 0.5
    for j, (c, s) in enumerate(zip(c2.numpy(), s2.numpy().astype(np.float64))):
        m = iou[:, j] * (gc == c) if len(gb) else np.zeros(0)
        gi = int(np.argmax(m)) if len(m) else -1
        matched = gi >= 0 and m[gi] >= 0.1
        same = (cl == c) & (sc >= 0.05)
        for var in VARIANTS:
            box, score = pb[j], float(s)
            if var == "rank=iou" and matched:
                score = float(m[gi])
            elif var == "rank=s*iou" and matched:
                score = float(s * m[gi])
            elif var == "rank=s*q":
                score = float(s * (m[gi] if matched else q_med))
            elif var.startswith("vote"):
                t, p = float(var.split()[1]), int(var.split()[2][1:])
                sel = same & (keep_raw[j] >= t)
                if sel.sum() > 1:
                    box = vote(pb[j], rb[sel], sc[sel] ** p)
            elif var == "pick" and matched:
                sel = np.where(same & (keep_raw[j] >= 0.6))[0]
                if len(sel):
                    box = rb[sel[np.argmax(iou_raw_gt[gi, sel])]]
                    if probiou(gb[gi:gi + 1], box[None])[0, 0] < m[gi]:
                        box = pb[j]
            ids, scs, pls = dets[var][NAMES[c]]
            ids.append(stem), scs.append(score), pls.append(obb2poly(box[None])[0])
    print(f"{k + 1}/{len(raw)}", end="\r", flush=True)

res = {}
for var in VARIANTS:
    d = {c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in dets[var].items()}
    res[var] = evaluate_ultra(d, gts, NAMES, max_det=3000)
short = {"none": "base", "rank=iou": "rk=iou", "rank=s*iou": "rk=s*iou", "rank=s*q": "rk=s*q", "pick": "pick"}
heads = [short.get(v, v.replace("vote ", "v").replace(" p", "/p")) for v in VARIANTS]
print(f"\n{Path(raw_path).name}, NMS {NMS_IOU}: delta mAP50-95 (base absolute)")
print(f"{'class':12s} " + " ".join(f"{h:>8s}" for h in heads))
for c in NAMES:
    base = res["none"]["classes"][c]["AP50_95"]
    print(f"{c:12s} {base:8.3f} " + " ".join(f"{res[v]['classes'][c]['AP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
base = res["none"]["mAP50_95"]
print(f"{'all':12s} {base:8.3f} " + " ".join(f"{res[v]['mAP50_95'] - base:+8.3f}" for v in VARIANTS[1:]))
print(f"{'all mAP50':12s} {res['none']['mAP50']:8.3f} " + " ".join(f"{res[v]['mAP50'] - res['none']['mAP50']:+8.3f}"
                                                         for v in VARIANTS[1:]))
