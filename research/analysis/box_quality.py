"""Box quality of a trained OpenOBB vs YOLO26x on Wall_Color val: for every ground-truth object, the best same-class
detection (conf >= 0.25) and the decomposition of its error in the object's own frame. Research only (Ultralytics is
used here as the baseline, never inside openobb/). CPU.

usage: python box_quality.py <dup_raw_<model>_53.npz from dup_analysis.py> <yolo .pt> [nms_iou of the OpenOBB model]
Coordinates: px of the image resized to long side 1280 (the training scale of both models).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb.data.imgio import imread  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color"
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw_path, yolo_path = sys.argv[1], sys.argv[2]
nms_iou = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
CONF = 0.25


def frame(pts):
    """8-point polygon -> (cx, cy, L, T, theta of the long side)."""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    th = np.deg2rad(a)
    if w < h:
        w, h, th = h, w, th + np.pi / 2
    return np.array([cx, cy, w, h, th])


def poly_xywhr(b):
    return Polygon(obb2poly(np.asarray(b, np.float64)[None])[0].reshape(4, 2)).buffer(0)


def openobb_dets(stem, rec):
    sc, cl, b = rec["raw"]
    if not len(sc):
        return []
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
    L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=nms_iou)
    s2, c2 = L[0].sigmoid().max(-1)
    k = s2 >= CONF
    return [(int(c), float(s), bb) for c, s, bb in zip(c2[k].numpy(), s2[k].numpy(), B[0][k].numpy())]


def yolo_dets(model, path):
    img = imread(str(path))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    r = model.predict(img, imgsz=1280, conf=CONF, iou=0.7, verbose=False, max_det=3000)[0]
    xywhr = r.obb.xywhr.cpu().numpy().copy()
    xywhr[:, :4] *= s
    return [(int(c), float(p), bb) for c, p, bb in zip(r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy(), xywhr)]


def decompose(g, p):
    """g, p: (cx, cy, L, T, theta) -> errors of p in g's frame (px, degrees)."""
    u = np.array([np.cos(g[4]), np.sin(g[4])])
    v = np.array([-u[1], u[0]])
    d = p[:2] - g[:2]
    dth = (p[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2
    along, across = d @ u, d @ v
    # ends of the prediction projected on the object's axis (its own long side, angle difference included)
    half = abs(p[2] / 2 * np.cos(dth)) + abs(p[3] / 2 * np.sin(dth))
    e1 = (along - half) - (-g[2] / 2)
    e2 = (along + half) - (g[2] / 2)
    return dict(angle=abs(np.rad2deg(dth)), across=abs(across), thick=p[3] - g[3], ends=(abs(e1) + abs(e2)) / 2,
                short_end=min(abs(e1), abs(e2)), long_end=max(abs(e1), abs(e2)))


def evaluate(dets_by_img, gts_by_img):
    stats = {n: {"n": 0, "iou": [], "err": []} for n in NAMES}
    for stem, gts in gts_by_img.items():
        dets = dets_by_img.get(stem, [])
        dp = [(c, s, bb, poly_xywhr(bb)) for c, s, bb in dets]
        for pts, ci in gts:
            st = stats[NAMES[ci]]
            st["n"] += 1
            gpoly = Polygon(np.asarray(pts).reshape(4, 2)).buffer(0)
            best, bb_best = 0.0, None
            for c, s, bb, pp in dp:
                if c != ci or not gpoly.intersects(pp):
                    continue
                inter = gpoly.intersection(pp).area
                iou = inter / max(gpoly.union(pp).area, 1e-9)
                if iou > best:
                    best, bb_best = iou, bb
            st["iou"].append(best)
            if bb_best is not None and best >= 0.1:
                p = np.array(bb_best, float)
                if p[2] < p[3]:
                    p = np.array([p[0], p[1], p[3], p[2], p[4] + np.pi / 2])
                st["err"].append(decompose(frame(pts), p))
    out = {}
    for n, st in stats.items():
        iou = np.array(st["iou"])
        errs = st["err"]
        med = lambda k: float(np.median([e[k] for e in errs])) if errs else float("nan")  # noqa: E731
        out[n] = dict(n=st["n"], r50=float((iou >= 0.5).mean()), r75=float((iou >= 0.75).mean()),
                      iou_med=float(np.median(iou[iou >= 0.5])) if (iou >= 0.5).any() else float("nan"),
                      ends=med("ends"), long_end=med("long_end"), thick=float(np.median([abs(e["thick"]) for e in errs]))
                      if errs else float("nan"), thick_bias=med("thick"), across=med("across"), angle=med("angle"))
    return out


raw = dict(np.load(raw_path, allow_pickle=True))
gts, ours = {}, {}
for stem, rec in raw.items():
    rec = rec.item()
    gts[stem] = [(pts, NAMES.index(name)) for pts, name, _ in rec["gt"]]
    ours[stem] = openobb_dets(stem, rec)

YCACHE = Path(__file__).with_name(f"yolo_dets_{Path(yolo_path).stem}_{len(gts)}.npz")
if YCACHE.exists():
    theirs = {k: list(v) for k, v in np.load(YCACHE, allow_pickle=True).items()}
else:
    from ultralytics import YOLO  # noqa: E402  (baseline only)
    yolo = YOLO(yolo_path)
    theirs = {}
    imgs = {p.stem: p for p in (ROOT / "images" / "val").iterdir()}
    for k, stem in enumerate(sorted(gts)):
        theirs[stem] = yolo_dets(yolo, imgs[stem])
        print(f"yolo {k + 1}/{len(gts)}", flush=True)
    np.savez(YCACHE, **{k: np.array(v, dtype=object) for k, v in theirs.items()})

res = {"openobb": evaluate(ours, gts), "yolo": evaluate(theirs, gts)}
hdr = f"{'class':12s} {'n':>5s} | {'R@.5':>11s} | {'R@.75':>11s} | {'IoU med':>11s} | {'ends px':>11s} | {'long end':>11s} | {'|thick| px':>11s} | {'thick bias':>11s} | {'across px':>11s} | {'angle deg':>11s}"
print(f"\nOpenOBB (left) vs YOLO26x (right); px at 1280 scale; medians over matched objects\n{hdr}")
for n in NAMES:
    a, b = res["openobb"][n], res["yolo"][n]
    cell = lambda k: f"{a[k]:5.2f} {b[k]:5.2f}"  # noqa: E731
    print(f"{n:12s} {a['n']:5d} | " + " | ".join(cell(k) for k in
          ("r50", "r75", "iou_med", "ends", "long_end", "thick", "thick_bias", "across", "angle")))
out = Path(__file__).with_name(f"box_quality_{Path(raw_path).stem}.json")
out.write_text(json.dumps(res, indent=1))
print("saved", out)
