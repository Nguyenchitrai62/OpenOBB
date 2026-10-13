"""Wall_Color: native sizes of wall_300 / junction / wall and what separates wall_300 from junction (colour, size, context).
Writes research/analysis/wall300_crops.png (wall_300 top rows, junction bottom rows, native pixels, 3x zoom)."""
import collections, glob, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.imgio import imread  # noqa: E402

root = Path("datasets/dataset_wall_color")
names = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
sizes, stats, crops = collections.Counter(), collections.defaultdict(list), {5: [], 6: []}
near = []
for split in ("train", "val"):
    for p in sorted((root / "images" / split).glob("*.*")):
        lab = root / "labels" / split / (p.stem + ".txt")
        if not lab.exists():
            continue
        im = imread(str(p))
        h, w = im.shape[:2]
        sizes[(w // 500 * 500, h // 500 * 500)] += 1
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        objs = []
        for r in rows:
            c = int(r[0]); pts = np.array(r[1:9], float).reshape(4, 2) * [w, h]
            (cx, cy), (bw, bh), _ = cv2.minAreaRect(pts.astype(np.float32))
            L, T = max(bw, bh), min(bw, bh)
            objs.append((c, cx, cy, L, T, pts))
            if c in (0, 5, 6):
                x0, y0 = int(max(cx - L / 2, 0)), int(max(cy - L / 2, 0))
                patch = im[y0:int(cy + L / 2) + 1, x0:int(cx + L / 2) + 1]
                ink = patch.reshape(-1, 3)[(patch.reshape(-1, 3) < 200).any(1)] if patch.size else np.zeros((0, 3))
                stats[c].append((L, T, w, *(np.median(ink, 0) if len(ink) else (-1, -1, -1))))
            if c in crops and len(crops[c]) < 24 and split == "train":
                s = int(max(L, 12) * 2.5)
                x0, y0 = int(cx - s), int(cy - s)
                if x0 >= 0 and y0 >= 0 and x0 + 2 * s < w and y0 + 2 * s < h:
                    crops[c].append(cv2.resize(im[y0:y0 + 2 * s, x0:x0 + 2 * s], (96, 96), interpolation=cv2.INTER_NEAREST))
        js = [o for o in objs if o[0] == 6]
        for o in objs:
            if o[0] == 5 and js:
                d = min(np.hypot(o[1] - j[1], o[2] - j[2]) for j in js)
                near.append(d / max(o[3], 1))
print("image sizes (w, h rounded to 500):", sizes.most_common(6))
for c in (0, 5, 6):
    a = np.array(stats[c])
    print(f"{names[c]:9s} n={len(a)} length px med {np.median(a[:, 0]):.1f} thick med {np.median(a[:, 1]):.1f} "
          f"| at 1280 long side: length {np.median(a[:, 0] / a[:, 2] * 1280):.1f} thick {np.median(a[:, 1] / a[:, 2] * 1280):.1f} "
          f"| ink BGR median {np.median(a[:, 3:6], 0)}")
print("wall_300: distance to nearest junction / own length, median", round(float(np.median(near)), 2))
rows = [np.hstack(crops[c][i:i + 12]) for c in (5, 6) for i in (0, 12) if len(crops[c]) >= i + 12]
cv2.imwrite("research/analysis/wall300_crops.png", np.vstack(rows))
