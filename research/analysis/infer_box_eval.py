"""Box quality AT INFERENCE (product predict path, conf 0.25), the part mAP hides: the tails of the orientation and
position errors, by object length. A long wall tilted by 3 deg still has IoU > 0.5 with its label, but its ends sit
several px off the drawn line.

Labelled images (Wall_Color val): every label is matched to the best same-class detection (ProbIoU >= 0.1); per class
and length bucket (px at the 1280 scale): recall at ProbIoU 0.5 / 0.75, |angle error| p50 / p90 / p99 and share > 1 / 3
deg, END OFFSET = largest distance of the detection's two end centres from the label's centre line (tilt x length +
sideways shift) p50 / p90 / p99, and share > 3 px.
Unlabelled drawings: drawing walls are mostly axis-aligned, so for wall / wall_300 detections >= 100 px the angle to
the nearest multiple of 90 deg is a label-free tilt check (its validity is printed from the val labels).

usage: python infer_box_eval.py <model.pt | yolo:<yolo .pt>> ... [--extra img ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import probiou  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
CONF = 0.25
BUCKETS = ((0, 40), (40, 100), (100, 200), (200, 10**5))
args = sys.argv[1:]
extra = args[args.index("--extra") + 1:] if "--extra" in args else []
models = args[:args.index("--extra")] if "--extra" in args else args


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def predictor(spec):
    """-> f(path) = (cls, conf, boxes (N, 5) canonical, px at the 1280 scale)."""
    if spec.startswith("yolo:"):
        from ultralytics import YOLO                  # research baseline only, never inside openobb/
        y = YOLO(spec[5:])

        def run(path):
            img = imread(str(path))
            s = 1280 / max(img.shape[:2])
            r = y.predict(img, imgsz=1280, conf=CONF, iou=0.7, verbose=False, max_det=3000)[0]
            b = r.obb.xywhr.cpu().numpy().astype(np.float64)
            b[:, :4] *= s
            return r.obb.cls.cpu().numpy().astype(int), r.obb.conf.cpu().numpy(), np.array([canon(x) for x in b]).reshape(-1, 5)
        return run
    from openobb import OpenOBB
    m = OpenOBB(spec)

    def run(path):
        img = imread(str(path))
        s = 1280 / max(img.shape[:2])
        r = m.predict(str(path), conf=CONF, save=False, verbose=False)[0]
        b = r.obb.xywhr.cpu().numpy().astype(np.float64)
        b[:, :4] *= s
        return r.obb.cls.cpu().numpy().astype(int), r.obb.conf.cpu().numpy(), np.array([canon(x) for x in b]).reshape(-1, 5)
    return run


def labels(path):
    img = imread(str(path))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    out = []
    lab = ROOT / "labels" / "val" / f"{path.stem}.txt"
    for line in lab.read_text().splitlines() if lab.exists() else []:
        v = line.split()
        if len(v) >= 9:
            pts = (np.array(v[1:9], float) * np.tile([w0 * s, h0 * s], 4)).reshape(4, 2).astype(np.float32)
            (cx, cy), (w, h), a = cv2.minAreaRect(pts)
            out.append((int(v[0]), canon([cx, cy, w, h, np.deg2rad(a)])))
    return out


def errors(g, p):
    """label g, detection p (canonical) -> |angle| deg, end offset px (max distance of p's end centres from g's line)."""
    dth = abs(np.rad2deg((p[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2))
    up = np.array([np.cos(p[4]), np.sin(p[4])])
    vg = np.array([-np.sin(g[4]), np.cos(g[4])])
    ends = [p[:2] + up * p[2] / 2, p[:2] - up * p[2] / 2]
    return dth, max(abs((e - g[:2]) @ vg) for e in ends)


def axis_dev(b):
    return np.abs(np.rad2deg((b[:, 4] + np.pi / 4) % (np.pi / 2) - np.pi / 4))


val = sorted((ROOT / "images" / "val").iterdir())
gts = {p.stem: labels(p) for p in val}
# validity of the label-free check: share of long labelled walls within 0.5 deg of an axis
gl = np.array([g for v in gts.values() for c, g in v if NAMES[c] in ("wall", "wall_300") and g[2] >= 100]).reshape(-1, 5)
print(f"val labels, wall / wall_300 >= 100 px: {len(gl)}, within 0.5 deg of an axis: {(axis_dev(gl) <= 0.5).mean():.1%}")

q = lambda x, p: float(np.quantile(x, p)) if len(x) else float("nan")  # noqa: E731
for spec in models:
    run = predictor(spec)
    rows = []          # (class, L, probiou, angle, end offset)
    for k, p in enumerate(val):
        cl, cf, pb = run(p)
        g = gts[p.stem]
        if not g:
            continue
        gb = np.array([x for _, x in g])
        gc = np.array([c for c, _ in g])
        iou = probiou(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        for i in range(len(gb)):
            m = iou[i] * (cl == gc[i]) if len(pb) else np.zeros(0)
            j = int(np.argmax(m)) if len(m) else -1
            if j < 0 or m[j] < 0.1:
                rows.append((gc[i], gb[i, 2], 0.0, np.nan, np.nan))
                continue
            a, e = errors(gb[i], pb[j])
            rows.append((gc[i], gb[i, 2], m[j], a, e))
        print(f"{Path(spec).name}: {k + 1}/{len(val)}", end="\r", flush=True)
    r = np.array(rows, np.float64)
    print(f"\n=== {spec} (val, conf {CONF}) ===")
    print(f"{'class':12s} {'length':>9s} {'n':>5s} {'R@.5':>5s} {'R@.75':>5s} | {'angle p50/p90/p99 deg':>22s} {'>1':>5s} {'>3':>5s} | "
          f"{'end offset p50/p90/p99 px':>26s} {'>3px':>5s}")
    for c in range(len(NAMES)):
        for lo, hi in ((0, 10**5),) + (BUCKETS if NAMES[c] in ("wall", "wall_300") else ()):
            k = (r[:, 0] == c) & (r[:, 1] >= lo) & (r[:, 1] < hi)
            if k.sum() < 5:
                continue
            x = r[k]
            f = x[x[:, 2] >= 0.5]                     # boxes that count as found
            a, e = f[:, 3], f[:, 4]
            lab = "all" if hi == 10**5 and lo == 0 else f"{lo}-{hi if hi < 10**5 else ''}"
            print(f"{NAMES[c]:12s} {lab:>9s} {int(k.sum()):5d} {(x[:, 2] >= .5).mean():5.2f} {(x[:, 2] >= .75).mean():5.2f} | "
                  f"{q(a, .5):6.2f} {q(a, .9):6.2f} {q(a, .99):7.2f}  {(a > 1).mean() if len(a) else 0:5.1%} {(a > 3).mean() if len(a) else 0:5.1%} | "
                  f"{q(e, .5):7.2f} {q(e, .9):7.2f} {q(e, .99):8.2f}  {(e > 3).mean() if len(e) else 0:5.1%}")
    for p in extra:
        cl, cf, pb = run(p)
        k = np.isin(cl, [NAMES.index("wall"), NAMES.index("wall_300")]) & (pb[:, 2] >= 100)
        d = axis_dev(pb[k])
        print(f"{Path(p).name[:40]:40s} wall/wall_300 >= 100 px: {int(k.sum()):4d} | axis deviation p50/p90/max "
              f"{q(d, .5):5.2f} / {q(d, .9):5.2f} / {(d.max() if len(d) else float('nan')):5.2f} deg | > 1 deg {(d > 1).mean() if len(d) else 0:5.1%}")
