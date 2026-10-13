"""Relative CPU speed (research only; absolute CPU times on this machine drift up to 3x, F47, so only the ratios in ONE
process count): YOLO (Ultralytics baseline) vs OpenOBB checkpoints, network forward incl. the model's own
postprocessing (OpenOBB: decode + NMS / top-k; YOLO26: end-to-end head), input 928 x 1280, alternating rounds, median.
usage: python speed_cpu_v12.py <yolo.pt> <name=openobb.pt[:o2o|o2m]> ... [--threads 4] [--rounds 5]"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yolo")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO  # baseline only

    from openobb import OpenOBB
    from openobb.predict import apply_head
    nets = {"YOLO26x": YOLO(a.yolo).model.fuse().eval()}
    for spec in a.models:
        name, path = spec.split("=", 1)
        head = None
        if path.endswith((":o2o", ":o2m")):
            path, head = path.rsplit(":", 1)
        m = OpenOBB(path)
        m.predict(np.full((64, 64, 3), 255, np.uint8), verbose=False, save=False)   # loads with calibrated settings
        net = m.model.eval()
        net.nms_conf = 0.25
        if head:
            apply_head(net, head)
        nets[name] = net
    x = torch.rand(1, 3, 928, 1280)
    res = {k: [] for k in nets}
    with torch.inference_mode():
        for n in nets.values():
            n(x)
        for r in range(a.rounds):
            for k, n in nets.items():
                t = time.perf_counter()
                n(x)
                res[k].append(time.perf_counter() - t)
            print(f"round {r + 1}/{a.rounds}", flush=True)
    base = np.median(res["YOLO26x"])
    print(f"\nCPU {a.threads} threads, input 928 x 1280, median of {a.rounds} alternating rounds")
    for k, v in res.items():
        print(f"{k:28s} {np.median(v) * 1e3:8.0f} ms  ({np.median(v) / base:.2f}x YOLO26x)")


if __name__ == "__main__":
    main()
