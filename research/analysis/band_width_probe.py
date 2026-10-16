"""Is the label thickness of a wall / wall_300 readable from the image? Wall_Color walls are COLOURED BANDS (pink, cyan..)
about 8 native px wide; the label box covers the band. For every labelled wall / wall_300 (length >= 6 x thickness), at
7 positions along its axis, the saturation profile across the box (native resolution, bilinear, 0.25 px steps over
+-1.5 x the label thickness) gives a band width (full width at half the profile's max above its floor, sub-pixel) and a
band centre. Then label thickness ~ a x width + b (least squares per class) and label centre vs band centre:
residuals in px at the 1280 scale, to compare with the thickness / across errors of the detectors
(box_quality.py: v23 |thick| 0.70 px, across 0.31 px; YOLO26x 0.54 / 0.28 px on walls). CPU, Wall_Color val.

usage: python band_width_probe.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.imgio import imread  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
CLS = {0: "wall", 5: "wall_300"}


def bilinear(a, xy):
    x, y = xy[..., 0], xy[..., 1]
    return cv2.remap(a, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fwhm(v, t):
    """Profile v over positions t -> (width, centre) at half height above the floor, sub-sample; None if flat."""
    lo, hi = np.percentile(v, 10), v.max()
    if hi - lo < 0.08:
        return None
    half = lo + 0.5 * (hi - lo)
    above = v >= half
    i = int(np.argmax(v))
    l = i
    while l > 0 and above[l - 1]:
        l -= 1
    r = i
    while r < len(v) - 1 and above[r + 1]:
        r += 1
    if l == 0 or r == len(v) - 1:
        return None
    tl = t[l - 1] + (half - v[l - 1]) / (v[l] - v[l - 1] + 1e-9) * (t[l] - t[l - 1])
    tr = t[r] + (half - v[r]) / (v[r + 1] - v[r] + 1e-9) * (t[r + 1] - t[r])
    return tr - tl, 0.5 * (tl + tr)


rows = {c: [] for c in CLS.values()}
for p in sorted((ROOT / "images" / "val").iterdir()):
    img = imread(str(p))
    h, w = img.shape[:2]
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255.0
    s = 1280 / max(h, w)
    for line in (ROOT / "labels" / "val" / f"{p.stem}.txt").read_text().splitlines():
        v = line.split()
        c = int(v[0])
        if c not in CLS:
            continue
        q = (np.array(v[1:9], float) * np.tile([w, h], 4)).reshape(4, 2).astype(np.float32)
        (cx, cy), (bw, bh), a = cv2.minAreaRect(q)
        L, T, th = (bw, bh, np.deg2rad(a)) if bw >= bh else (bh, bw, np.deg2rad(a) + np.pi / 2)
        if L < 6 * T or T < 2:
            continue
        ax, nx = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        t = np.arange(-1.5 * T, 1.5 * T + 1e-6, 0.25)
        ws, cs = [], []
        for u in np.linspace(-0.35, 0.35, 7) * L:
            pts = np.array([cx, cy]) + u * ax + t[:, None] * nx
            r = fwhm(bilinear(sat, pts[None]).ravel(), t)
            if r is not None:
                ws.append(r[0]), cs.append(r[1])
        if len(ws) >= 3:
            rows[CLS[c]].append((T * s, np.median(ws) * s, np.median(cs) * s))

print("px at the 1280 scale; label thickness T vs band width (FWHM of saturation), median of 7 profiles per object")
for c, r in rows.items():
    r = np.array(r)
    if not len(r):
        continue
    T, W, C = r[:, 0], r[:, 1], r[:, 2]
    A = np.stack([W, np.ones_like(W)], 1)
    (a, b), *_ = np.linalg.lstsq(A, T, rcond=None)
    res = T - (a * W + b)
    print(f"{c:9s} n {len(r):4d} | label T median {np.median(T):.2f} | band W median {np.median(W):.2f} | corr {np.corrcoef(T, W)[0, 1]:.3f}"
          f" | T = {a:.2f} W + {b:+.2f}: |residual| median {np.median(np.abs(res)):.2f}, p75 {np.percentile(np.abs(res), 75):.2f}"
          f" | |T - W| median {np.median(np.abs(T - W)):.2f} | band centre vs label centre |d| median {np.median(np.abs(C)):.2f}")
