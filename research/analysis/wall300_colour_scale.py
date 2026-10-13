"""How much of wall_300's distinguishing colour survives the resize to imgsz (1280) vs native, compared with junction.
Per object: fraction of 'pinkish' ink pixels (hue magenta, saturated) inside the box, at native size and after the
INTER_AREA / INTER_LINEAR resize to 1280 long side; plus the colour contrast between wall_300 and junction boxes."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.imgio import imread  # noqa: E402

root = Path("datasets/dataset_wall_color")


def pink(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return ((h >= 140) | (h <= 5)) & (s >= 60) & (v >= 80)


res = {5: {"native": [], "area": [], "linear": []}, 6: {"native": [], "area": [], "linear": []}}
for p in sorted((root / "images" / "val").glob("*.*")):
    lab = root / "labels" / "val" / (p.stem + ".txt")
    im = imread(str(p)); h, w = im.shape[:2]; r = 1280 / max(h, w)
    small = {"area": cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA),
             "linear": cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR)}
    for line in lab.read_text().splitlines():
        v = line.split()
        c = int(v[0])
        if c not in res:
            continue
        pts = np.array(v[1:9], float).reshape(4, 2) * [w, h]
        x0, y0 = np.maximum(np.floor(pts.min(0)), 0).astype(int); x1, y1 = np.ceil(pts.max(0)).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        res[c]["native"].append(pink(im[y0:y1 + 1, x0:x1 + 1]).mean())
        for k, s in small.items():
            a0, b0 = int(x0 * r), int(y0 * r); a1, b1 = int(np.ceil(x1 * r)), int(np.ceil(y1 * r))
            res[c][k].append(pink(s[b0:b1 + 1, a0:a1 + 1]).mean())
for k in ("native", "area", "linear"):
    a, b = np.array(res[5][k]), np.array(res[6][k])
    thr = 0.05
    print(f"{k:7s} pink fraction median wall_300 {np.median(a):.3f} junction {np.median(b):.3f} | "
          f"objects with >{thr:.0%} pink: wall_300 {np.mean(a > thr):.0%}, junction {np.mean(b > thr):.0%}")
