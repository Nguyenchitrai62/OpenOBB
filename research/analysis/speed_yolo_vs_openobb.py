"""YOLO (Ultralytics baseline, research only) vs OpenOBB checkpoints: raw network forward, same rect input
928 x 1280, 1 CPU thread, ONE process, alternating runs (machine drift hits every model equally); median of 3.
usage: UL_PATH=<ultralytics src> python speed_yolo_vs_openobb.py <yolo.pt> <openobb.pt> [<openobb.pt> ...]"""
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch

sys.path.insert(0, os.environ["UL_PATH"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from ultralytics import YOLO  # noqa: E402

from openobb import OpenOBB  # noqa: E402

nets = {"YOLO " + os.path.basename(sys.argv[1]): YOLO(sys.argv[1]).model.fuse().eval()}
for p in sys.argv[2:]:
    m = OpenOBB(p).model.eval()
    m.nms_conf = 0.25
    nets[os.path.basename(p)] = m
x = torch.rand(1, 3, 928, 1280)
res = {k: [] for k in nets}
with torch.inference_mode():
    for n in nets.values():
        n(x)
    for _ in range(3):
        for k, n in nets.items():
            t = time.perf_counter()
            n(x)
            res[k].append(time.perf_counter() - t)
base = np.median(res[next(iter(res))])
for k, v in res.items():
    print(f"{k:24s} {np.median(v) * 1e3:8.0f} ms  ({np.median(v) / base:.2f}x YOLO)")
