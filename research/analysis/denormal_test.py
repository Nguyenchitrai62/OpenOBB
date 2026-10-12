"""Is a checkpoint slow on CPU because of denormal floats? Times the raw forward (928 x 1280, 1 thread) with
flush-denormal off / on, alternating, and counts denormal values in the backbone stage outputs.
usage: python denormal_test.py <openobb.pt>"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"F:\Source_code\NEW_architecture")
torch.set_num_threads(1)
from openobb import OpenOBB  # noqa: E402

net = OpenOBB(sys.argv[1]).model.eval()
net.nms_conf = 0.25
x = torch.rand(1, 3, 928, 1280)
tiny = torch.finfo(torch.float32).tiny
with torch.inference_mode():
    h = net.backbone.stem(x)
    for i, st in enumerate(net.backbone.stages):
        h = st(h)
        d = ((h != 0) & (h.abs() < tiny)).float().mean().item()
        print(f"stage {i}: mean|x| {h.abs().mean().item():.3g}, denormal share {d * 100:.4f}%")
    res = {False: [], True: []}
    for _ in range(2):
        for flush in (False, True):
            torch.set_flush_denormal(flush)
            net(x)
            t = time.perf_counter()
            net(x)
            res[flush].append(time.perf_counter() - t)
for flush, v in res.items():
    print(f"flush_denormal={flush}: {np.median(v) * 1e3:.0f} ms")
