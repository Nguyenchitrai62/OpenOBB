"""Label-free tilt check on real drawings (product predict path, conf 0.25): wall / wall_300 detections >= 100 px
(1280 scale) more than 1 / 3 deg off the nearest axis (val labels: 100% of such walls lie within 0.5 deg of an axis;
a drawing with true diagonal walls shows up as a floor, the same for every model).

usage: python tilt_real.py name=model.pt ... [yolo=yolo.pt] --images img ...
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb.data.imgio import imread  # noqa: E402

args = sys.argv[1:]
imgs = args[args.index("--images") + 1:]
specs = [a.split("=", 1) for a in args[:args.index("--images")]]


def axis_dev(th):
    return np.abs(np.rad2deg((th + np.pi / 4) % (np.pi / 2) - np.pi / 4))


print(f"{'model':12s} " + " ".join(f"{Path(p).stem[:22]:>30s}" for p in imgs))
for name, pt in specs:
    if name.lower().startswith("yolo"):
        from ultralytics import YOLO                   # research baseline only
        y = YOLO(pt)
        run = lambda p: y.predict(imread(p), imgsz=1280, conf=0.25, iou=0.7, verbose=False, max_det=3000)[0]  # noqa: E731
        walls = [0, 5]
    else:
        from openobb import OpenOBB
        m = OpenOBB(pt)
        run = lambda p: m.predict(p, conf=0.25, save=False, verbose=False)[0]  # noqa: E731
        walls = [k for k, v in m.names.items() if v in ("wall", "wall_300")]
    cells = []
    for p in imgs:
        s = 1280 / max(imread(p).shape[:2])
        r = run(p)
        b = r.obb.xywhr.cpu().numpy().astype(np.float64)
        c = r.obb.cls.cpu().numpy().astype(int)
        L = np.maximum(b[:, 2], b[:, 3]) * s
        th = np.where(b[:, 2] >= b[:, 3], b[:, 4], b[:, 4] + np.pi / 2)
        k = np.isin(c, walls) & (L >= 100)
        d = axis_dev(th[k])
        cells.append(f"n {int(k.sum()):3d} >1° {np.mean(d > 1) if len(d) else 0:5.1%} >3° {np.mean(d > 3) if len(d) else 0:5.1%}")
    print(f"{name:12s} " + " ".join(f"{x:>30s}" for x in cells), flush=True)
