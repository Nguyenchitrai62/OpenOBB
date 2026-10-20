"""One model, one page, 1 CPU thread: preprocess / inference / postprocess from Results.speed + end-to-end.
usage: python speed_page_bench.py <model.pt> <image> [yolo]   ('yolo' = Ultralytics baseline, research only)"""
import os
import sys
from pathlib import Path
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.data.imgio import imread  # noqa: E402

path, img_path = sys.argv[1], sys.argv[2]
img = imread(img_path)
if len(sys.argv) > 3 and sys.argv[3] == "yolo":
    sys.path.insert(0, os.environ["UL_PATH"])
    from ultralytics import YOLO
    m = YOLO(path)
else:
    from openobb import OpenOBB
    m = OpenOBB(path)
fn = lambda: m.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]  # noqa: E731
fn()
best = None
for _ in range(2):
    t = time.perf_counter()
    r = fn()
    wall = (time.perf_counter() - t) * 1e3
    if best is None or wall < best[0]:
        best = (wall, dict(r.speed), len(r.obb))
wall, sp, n = best
print(f"{os.path.basename(path):18s} preprocess {sp['preprocess']:7.1f} ms | inference {sp['inference']:8.1f} ms | "
      f"postprocess {sp['postprocess']:7.1f} ms | end-to-end {wall:8.1f} ms | {n} boxes")
