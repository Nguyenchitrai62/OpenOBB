"""Is the stride-32 backbone path (stage 4 -> LSK P5 -> encoder input_proj P5) used by a trained checkpoint?
Runs the page normally and with the backbone P5 map set to 0 (before LSK), compares detections (conf 0.25).
usage: python p5_zero_test.py <checkpoint.pt> <image>   (1 CPU thread, one model per process)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402

ck, path = sys.argv[1], sys.argv[2]
m = OpenOBB(ck)
net = m.model
img = imread(path)
stats = {}


def run(zero):
    bb = net.backbone
    st4 = bb.stages[-1]
    orig = st4.forward

    def fwd(x):
        y = orig(x)
        stats["p5_mean_abs"] = float(y.abs().mean())
        return y * 0 if zero else y
    st4.forward = fwd
    try:
        r = m.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
    finally:
        st4.forward = orig
    return r.obb.xywhr.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()


b0, c0, s0 = run(False)
p5 = stats["p5_mean_abs"]
b1, c1, s1 = run(True)
names = m.names
d = np.linalg.norm(b0[:, None, :2] - b1[None, :, :2], axis=-1) + 1e6 * (c0[:, None] != c1[None])
match = d.min(1) < 2 if len(b1) else np.zeros(len(b0), bool)
print(f"{ck}: backbone P5 mean|x| {p5:.3g}")
print(f"  normal {len(b0)} boxes, P5=0 {len(b1)} boxes, normal boxes kept (same class, centre < 2 px): "
      f"{match.mean() * 100:.1f}%")
for k in sorted(set(c0.tolist()) | set(c1.tolist())):
    print(f"  {names[int(k)]:12s} {int((c0 == k).sum()):4d} -> {int((c1 == k).sum()):4d}")
if len(b1) and match.any():
    j = d.argmin(1)[match]
    print(f"  score change on kept boxes: median {np.median(np.abs(s0[match] - s1[j])):.3f}, "
          f"max {np.abs(s0[match] - s1[j]).max():.3f}")
