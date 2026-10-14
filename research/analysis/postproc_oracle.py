"""Post-processing variants on a trained model's raw candidates (dup_raw_<model>_53.npz from dup_analysis.py), CPU:
  * ranking: per labelled object, IoU of the top-scored same-class candidate vs the best candidate available;
  * NMS where the overlap test inflates every box to at least `m` px on each side (thin-aware NMS): two 5.6 px wall
    boxes 2 px apart overlap at IoU ~0.5 and both survive a 0.7 NMS; inflated they are recognised as one object;
  * box voting: each kept box becomes the score^p-weighted mean of the same-class candidates with ProbIoU >= v to it.
Scored with the YOLO-style evaluator; duplicates counted as in the trainer's report (conf 0.25, same class, IoU 0.5).

usage: python postproc_oracle.py <dup_raw_<model>_53.npz>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import evaluate_ultra, probiou  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw = dict(np.load(sys.argv[1], allow_pickle=True))
data = {}
for stem, rec in sorted(raw.items()):
    rec = rec.item()
    sc, cl, b = rec["raw"]
    gts = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    data[stem] = (np.asarray(sc, np.float64), np.asarray(cl, int), np.asarray(b, np.float64).reshape(-1, 5), gts)


def gt_xywhr(pts):
    import cv2
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return np.array([cx, cy, w, h, np.deg2rad(a)])


def inflate(b, m):
    b = b.copy()
    b[:, 2:4] = np.maximum(b[:, 2:4], m)
    return b


def nms(sc, cl, b, iou=0.7, m=0.0):
    """Class-wise greedy NMS on ProbIoU of boxes inflated to >= m px; -> kept indices (score order)."""
    keep = []
    bi = inflate(b, m) if m > 0 else b
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]
        idx = idx[np.argsort(-sc[idx])]
        if not len(idx):
            continue
        ov = probiou(bi[idx], bi[idx])
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(idx[i])
            alive &= ~(ov[i] > iou)
            alive[i] = False
    keep = np.array(keep, int)
    return keep[np.argsort(-sc[keep])] if len(keep) else keep


def vote(sc, cl, b, keep, v=0.6, p=1.0):
    """Kept box <- score^p weighted mean of same-class candidates with ProbIoU >= v to it (angle: axis of the kept
    box; members are expressed in its frame so w / h do not swap)."""
    out = b[keep].copy()
    for j, k in enumerate(keep):
        same = np.where(cl == cl[k])[0]
        ov = probiou(b[k:k + 1], b[same])[0]
        mem = same[ov >= v]
        if len(mem) < 2:
            continue
        w = sc[mem] ** p
        th = b[k, 4]
        u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        L, T = [], []
        for q in mem:
            dth = (b[q, 4] - th + np.pi / 2) % np.pi - np.pi / 2
            swap = abs(dth) > np.pi / 4          # same box written with w / h swapped (theta + 90 deg)
            L.append(b[q, 3] if swap else b[q, 2])
            T.append(b[q, 2] if swap else b[q, 3])
        d = b[mem, :2] - b[k, :2]
        c = b[k, :2] + u * np.average(d @ u, weights=w) + n * np.average(d @ n, weights=w)
        out[j] = [c[0], c[1], np.average(L, weights=w), np.average(T, weights=w), th]
    return out


def run(m=0.0, iou=0.7, v=None, p=1.0):
    dets = {c: ([], [], []) for c in NAMES}
    gts, dup = {}, 0
    n_det = 0
    for stem, (sc, cl, b, g) in data.items():
        gts[stem] = g
        keep = nms(sc, cl, b, iou, m)
        kb = vote(sc, cl, b, keep, v, p) if v is not None else b[keep]
        for k, box in zip(keep, kb):
            ids, s_, pl = dets[NAMES[cl[k]]]
            ids.append(stem), s_.append(sc[k]), pl.append(obb2poly(box[None])[0])
        # duplicates as in the trainer: conf >= 0.25, same class, ProbIoU >= 0.5 to an object already detected
        ok = sc[keep] >= 0.25
        if ok.any() and g:
            gb = np.array([gt_xywhr(p_) for p_, _, _ in g])
            gc = np.array([NAMES.index(n_) for _, n_, _ in g])
            iou_g = probiou(kb[ok], gb)
            hit = np.zeros(len(g), bool)
            for r, k in enumerate(keep[ok]):
                cand = np.where((gc == cl[k]) & (iou_g[r] >= 0.5))[0]
                if not len(cand):
                    continue
                j = cand[np.argmax(iou_g[r, cand])]
                if hit[j]:
                    dup += 1
                hit[j] = True
            n_det += int(ok.sum())
    d = {c: (i, np.asarray(s), np.asarray(pl).reshape(-1, 8)) for c, (i, s, pl) in dets.items()}
    res = evaluate_ultra(d, gts, NAMES, max_det=3000)
    return res, dup / max(n_det, 1)


def ranking():
    """Median over objects: IoU of the top-scored same-class candidate and of the best candidate (score >= 0.05)."""
    out = {c: ([], []) for c in NAMES}
    for sc, cl, b, g in data.values():
        if not g:
            continue
        gb = np.array([gt_xywhr(p_) for p_, _, _ in g])
        ov = probiou(gb, b)
        for j, (_, n_, _) in enumerate(g):
            m = (cl == NAMES.index(n_)) & (ov[j] >= 0.3)
            if not m.any():
                continue
            idx = np.where(m)[0]
            top = idx[np.argmax(sc[idx])]
            out[n_][0].append(ov[j, top])
            out[n_][1].append(ov[j, idx].max())
    print("ranking (ProbIoU to the object): top-scored candidate | best candidate | best - top (median)")
    for c in NAMES:
        t, bst = np.array(out[c][0]), np.array(out[c][1])
        if len(t):
            print(f"  {c:12s} {np.median(t):.3f} | {np.median(bst):.3f} | {np.median(bst - t):+.3f}  (n {len(t)})")


ranking()
rows = [("NMS 0.7 (baseline)", dict()), ("NMS 0.5", dict(iou=0.5)),
        ("thin-aware NMS 0.7, m=8 px", dict(m=8.0)), ("thin-aware NMS 0.7, m=12 px", dict(m=12.0)),
        ("vote v=0.6", dict(v=0.6)), ("vote v=0.5 p=2", dict(v=0.5, p=2.0)),
        ("thin-aware m=8 + vote v=0.6", dict(m=8.0, v=0.6)), ("thin-aware m=12 + vote v=0.5 p=2", dict(m=12.0, v=0.5, p=2.0))]
print(f"\n{'variant':34s} {'mAP50':>6s} {'50-95':>6s} {'dup':>6s} | " + " ".join(f"{c[:8]:>8s}" for c in NAMES) + "  (AP50-95)")
for name, kw in rows:
    r, dup = run(**kw)
    print(f"{name:34s} {r['mAP50']:6.3f} {r['mAP50_95']:6.3f} {100 * dup:5.1f}% | "
          + " ".join(f"{r['classes'][c]['AP50_95']:8.3f}" for c in NAMES), flush=True)
