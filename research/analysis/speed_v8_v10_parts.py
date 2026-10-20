"""v8 vs v10 in ONE process, alternating runs (cancels machine drift), 1 CPU thread, same rect input of the page.
Breakdown per forward: core (backbone + LSK + encoder), head (+ decode), refiner stage, NMS.
usage: python speed_v8_v10_parts.py <v8.pt> <v10.pt> <image>"""
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.predict import PAD_BGR, rect_shape  # noqa: E402
from openobb.speed import nms_timer  # noqa: E402

img = imread(sys.argv[3])
s = 1280 / max(img.shape[:2])
img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
H, W = rect_shape(*img.shape[:2], 1280)
x = np.full((H, W, 3), PAD_BGR, np.uint8)
x[:img.shape[0], :img.shape[1]] = img
xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)

nets = {}
for name, p in (("v8", sys.argv[1]), ("v10", sys.argv[2])):
    net = OpenOBB(p).model.eval()
    net.nms_conf = 0.25
    nets[name] = net


def timed(net):
    t = {}
    enc, head = net.encode, net.head.forward

    def e(xx):
        a = time.perf_counter()
        r = enc(xx)
        t["core"] = time.perf_counter() - a
        return r

    def h(f):
        a = time.perf_counter()
        r = head(f)
        t["head"] = time.perf_counter() - a
        return r
    net.encode, net.head.forward = e, h
    try:
        with nms_timer(None) as nms, torch.inference_mode():
            a = time.perf_counter()
            out = net(xt)
            t["total"] = time.perf_counter() - a
        t["nms"] = nms[0]
    finally:
        del net.encode
        net.head.forward = head
    t["refiner+rest"] = t["total"] - t["core"] - t["head"] - t["nms"]
    t["boxes"] = int((out["pred_logits"].amax(-1).sigmoid() > 0.25).sum())
    return t


print(f"input {H}x{W}, 1 thread")
res = {k: [] for k in nets}
for name in nets:
    timed(nets[name])                                   # warm-up
for _ in range(3):
    for name in nets:
        res[name].append(timed(nets[name]))
for name, rs in res.items():
    keys = ("total", "core", "head", "refiner+rest", "nms")
    med = {k: np.median([r[k] for r in rs]) * 1e3 for k in keys}
    print(f"{name:4s} " + " | ".join(f"{k} {med[k]:7.0f} ms" for k in keys) + f" | {rs[-1]['boxes']} boxes")
