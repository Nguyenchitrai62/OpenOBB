"""One table for several trained versions on Wall_Color val: mAP50 / mAP50-95 (YOLO-style), duplicate boxes, and the
ORIENTATION / POSITION tails that mAP hides (a long wall tilted 3 deg keeps IoU > 0.5).

Inputs are cached raw candidates (dup_analysis.py, DUP_HEAD=o2m) -> class-wise NMS at --nms; YOLO26x from the
box_quality.py cache (conf >= 0.25, its own NMS 0.7; its mAP comes from F59, not recomputed here).
Box stats on detections with conf >= 0.25: every label is matched to its best same-class detection (ProbIoU >= 0.5 =
found); |angle error| share > 1 / 2 / 5 deg, END OFFSET = largest distance of the detection's end centres from the
label's centre line (tilt x length + sideways shift); for wall / wall_300 also by label length (px at 1280).
Tilted detections (no label needed): conf >= 0.25 wall / wall_300 boxes >= 100 px more than 1 deg off the nearest axis
(val labels: 100% of such walls lie within 0.5 deg of an axis).

usage: python eval_versions.py name=dup_raw_X.npz ... [yolo=yolo_dets_Y_53.npz] [--nms 0.5]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import confusion, evaluate_ultra, probiou  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
args = sys.argv[1:]
NMS = float(args[args.index("--nms") + 1]) if "--nms" in args else 0.5
specs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]
HERE = Path(__file__).parent


def canon(b):
    b = np.asarray(b, np.float64).reshape(-1, 5).copy()
    sw = b[:, 2] < b[:, 3]
    b[sw, 2], b[sw, 3] = b[sw, 3], b[sw, 2].copy()
    b[sw, 4] += np.pi / 2
    return b


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])[0]


def axis_dev(b):
    return np.abs(np.rad2deg((b[:, 4] + np.pi / 4) % (np.pi / 2) - np.pi / 4))


def load_ours(path):
    raw = dict(np.load(HERE / path, allow_pickle=True))
    out, gts = {}, {}
    for stem, rec in raw.items():
        rec = rec.item()
        gts[stem] = rec["gt"]
        sc, cl, b = rec["raw"]
        if not len(sc):
            out[stem] = (np.zeros(0), np.zeros(0, int), np.zeros((0, 5)))
            continue
        lg = torch.zeros(1, len(sc), len(NAMES)) - 30
        lg[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
        L, B = fast_nms(lg, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=NMS)
        s2, c2 = L[0].sigmoid().max(-1)
        k = (s2 > 0.05).numpy()
        out[stem] = (s2.numpy()[k], c2.numpy()[k], B[0].numpy()[k].astype(np.float64))
    return out, gts


def load_yolo(path):
    y = dict(np.load(HERE / path, allow_pickle=True))
    out = {}
    for stem, v in y.items():
        v = list(v)
        out[stem] = (np.array([d[1] for d in v]), np.array([d[0] for d in v], int),
                     np.array([d[2] for d in v], np.float64).reshape(-1, 5))
    return out


results, GTS = {}, None
for name, path in specs:
    dets = load_yolo(path) if name.lower().startswith("yolo") else None
    if dets is None:
        dets, g = load_ours(path)
        GTS = GTS or g
    results[name] = dets
gts = {k: [(np.asarray(p, np.float64), n, 0) for p, n, _ in v] for k, v in GTS.items()}

q = lambda x, p: float(np.quantile(x, p)) if len(x) else float("nan")  # noqa: E731
pc = lambda x: f"{np.mean(x):6.1%}" if len(x) else "   n/a"          # noqa: E731
summary = []
for name, dets in results.items():
    is_yolo = name.lower().startswith("yolo")
    ev, per_img = {c: ([], [], []) for c in NAMES}, {}
    rows, tilt = [], []
    for stem in gts:
        sc, cl, b = dets.get(stem, (np.zeros(0), np.zeros(0, int), np.zeros((0, 5))))
        pl = obb2poly(b) if len(b) else np.zeros((0, 8))
        per_img[stem] = [(float(s), int(c), p) for s, c, p in zip(sc, cl, pl)]
        for s, c, p in zip(sc, cl, pl):
            ids, ss, pp = ev[NAMES[int(c)]]
            ids.append(stem), ss.append(float(s)), pp.append(p)
        keep = sc >= 0.25
        pb, pcl = canon(b[keep]), cl[keep]
        lw = np.isin(pcl, [0, 5]) & (pb[:, 2] >= 100)
        tilt += list(axis_dev(pb[lw]))
        gb = np.array([gt_box(p) for p, _, _ in gts[stem]]).reshape(-1, 5)
        gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
        iou = probiou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
        for i in range(len(gb)):
            m = iou[i] * (pcl == gc[i]) if len(pb) else np.zeros(0)
            j = int(np.argmax(m)) if len(m) else -1
            if j < 0 or m[j] < 0.5:
                continue
            g, p = gb[i], pb[j]
            # near-square labels (junction, many doors): the long axis is arbitrary, so the angle is taken modulo 90 deg
            cands = [p] + ([np.array([p[0], p[1], p[3], p[2], p[4] + np.pi / 2])] if g[2] < 1.5 * g[3] else [])
            best = None
            for pp in cands:
                dth = abs(np.rad2deg((pp[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2))
                up, vg = np.array([np.cos(pp[4]), np.sin(pp[4])]), np.array([-np.sin(g[4]), np.cos(g[4])])
                off = max(abs((pp[:2] + s * up * pp[2] / 2 - g[:2]) @ vg) for s in (-1, 1))
                if best is None or dth < best[0]:
                    best = (dth, off)
            rows.append((gc[i], g[2], best[0], best[1]))
    r = np.array(rows, np.float64).reshape(-1, 4)
    if is_yolo:
        m50, m95 = 0.746, 0.499                            # F59 (Colab, conf 0.001); cache only holds conf >= 0.25
    else:
        d = {c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in ev.items()}
        res = evaluate_ultra(d, gts, NAMES, max_det=3000)
        m50, m95 = res["mAP50"], res["mAP50_95"]
    cm = confusion(per_img, gts, NAMES)
    ndup, ndet = sum(cm["duplicates"].values()), cm["detections"]
    a, e = r[:, 2], r[:, 3]
    wl = np.isin(r[:, 0], [0, 5]) & (r[:, 1] >= 100)
    t = np.array(tilt)
    summary.append((name, m50, m95, ndup / max(ndet, 1), len(r), a, e, r, wl, t))

print(f"Wall_Color val (53 images), NMS {NMS} on cached candidates; box stats at conf 0.25 on labels found (ProbIoU >= 0.5)")
print(f"{'model':16s} {'mAP50':>6s} {'50-95':>6s} {'dup':>6s} | {'found':>5s} {'ang>1':>6s} {'ang>2':>6s} {'ang>5':>6s} {'ang p99':>7s} | "
      f"{'off>2px':>7s} {'off>4px':>7s} {'off p99':>7s} || long walls >=100px: {'n':>4s} {'ang>1':>6s} {'off>4px':>7s} | "
      f"tilted dets: {'n':>4s} {'>1deg':>6s} {'>3deg':>6s}")
for name, m50, m95, dr, n, a, e, r, wl, t in summary:
    print(f"{name:16s} {m50:6.3f} {m95:6.3f} {dr:6.1%} | {n:5d} {pc(a > 1)} {pc(a > 2)} {pc(a > 5)} {q(a, .99):7.2f} | "
          f"{pc(e > 2):>7s} {pc(e > 4):>7s} {q(e, .99):7.2f} || {'':20s}{int(wl.sum()):4d} {pc(a[wl] > 1)} {pc(e[wl] > 4):>7s} | "
          f"{'':13s}{len(t):4d} {pc(t > 1)} {pc(t > 3)}")
print("\nper class: share of found labels with |angle error| > 2 deg / end offset > 4 px")
print(f"{'class':12s} " + " ".join(f"{s[0]:>18s}" for s in summary))
for c in range(len(NAMES)):
    cells = []
    for _, _, _, _, _, a, e, r, _, _ in summary:
        k = r[:, 0] == c
        cells.append(f"{pc(a[k] > 2)} / {pc(e[k] > 4)}" if k.sum() else "n/a")
    print(f"{NAMES[c]:12s} " + " ".join(f"{x:>18s}" for x in cells))
print("\nwall / wall_300 by label length: share |angle| > 1 deg / end offset > 4 px")
for lo, hi in ((0, 40), (40, 100), (100, 200), (200, 10**5)):
    cells = []
    for _, _, _, _, _, a, e, r, _, _ in summary:
        k = np.isin(r[:, 0], [0, 5]) & (r[:, 1] >= lo) & (r[:, 1] < hi)
        cells.append(f"{pc(a[k] > 1)} / {pc(e[k] > 4)} (n {int(k.sum())})" if k.sum() else "n/a")
    print(f"{lo:4d}-{hi if hi < 10**5 else '':<6} " + " ".join(f"{x:>26s}" for x in cells))
