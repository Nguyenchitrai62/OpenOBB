"""Two checkpoints on one drawing: detections per class, then for same-class pairs (ProbIoU >= 0.5) the differences
in angle, centre shift across / along the axis, length and thickness; plots both. CPU, product predict path.

usage: python compare_two.py <a.pt> <b.pt> <image> [out_dir]
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread, imwrite  # noqa: E402
from openobb.eval.ultra import probiou  # noqa: E402

a_pt, b_pt, img_path = sys.argv[1:4]
out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).with_name("compare_two_out")
out.mkdir(exist_ok=True)
img = imread(img_path)
print("image", img.shape)
res = {}
for pt in (a_pt, b_pt):
    m = OpenOBB(pt)
    r = m.predict(img_path, save=False, verbose=False)[0]
    xywhr = r.obb.xywhr.cpu().numpy().astype(np.float64)
    cls = r.obb.cls.cpu().numpy().astype(int)
    conf = r.obb.conf.cpu().numpy()
    names = m.names
    res[pt] = (xywhr, cls, conf, names)
    print(f"{Path(pt).name}: {len(cls)} boxes", dict(Counter(names[c] for c in cls)), "| head", m.model.__class__.__name__,
          "| infer", getattr(m, "ckpt_args", {}).get("infer_head") if hasattr(m, "ckpt_args") else "")
    vis = r.plot()
    imwrite(str(out / f"{Path(pt).stem}.jpg"), vis)


def canon(b):
    b = b.copy()
    sw = b[:, 2] < b[:, 3]
    b[sw, 2], b[sw, 3] = b[sw, 3], b[sw, 2].copy()
    b[sw, 4] += np.pi / 2
    return b


(A, ca, sa, names), (B, cb, sb, _) = res[a_pt], res[b_pt]
A, B = canon(A), canon(B)
iou = probiou(A, B)
rows = []
for i in range(len(A)):
    m = iou[i] * (cb == ca[i])
    j = int(np.argmax(m)) if len(m) else -1
    if j < 0 or m[j] < 0.5:
        continue
    u = np.array([np.cos(A[i, 4]), np.sin(A[i, 4])])
    d = B[j, :2] - A[i, :2]
    dth = np.rad2deg((B[j, 4] - A[i, 4] + np.pi / 2) % np.pi - np.pi / 2)
    rows.append((ca[i], abs(dth), abs(d @ np.array([-u[1], u[0]])), abs(d @ u), B[j, 2] - A[i, 2], B[j, 3] - A[i, 3]))
r = np.array(rows)
print(f"\nmatched pairs {len(r)} of {len(A)} / {len(B)}; per class median |angle| deg, |across| px, |along| px, dL, dT (b - a)")
for c in sorted(set(r[:, 0].astype(int))):
    x = r[r[:, 0] == c]
    print(f"{names[c]:12s} n {len(x):5d}  angle {np.median(x[:, 1]):5.2f} (p90 {np.quantile(x[:, 1], .9):5.2f})  "
          f"across {np.median(x[:, 2]):5.2f}  along {np.median(x[:, 3]):5.2f}  dL {np.median(x[:, 4]):+6.2f}  dT {np.median(x[:, 5]):+5.2f}")
print("plots in", out)
