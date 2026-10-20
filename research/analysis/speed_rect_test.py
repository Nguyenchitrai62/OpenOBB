"""OpenOBB3-x: square 1280x1280 (current) vs rectangular 928x1280 input on the real page. 1 thread."""
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
from openobb.predict import PAD_BGR, postprocess  # noqa: E402

m = OpenOBB(sys.argv[1] if len(sys.argv) > 1 else r"F:\Downloads\openobb3x.pt")
net = m.model.eval()
net.nms_conf = 0.25
img = imread(r"F:\Downloads\9F作図資料（新規作図） - trang 1 @2x.png")
s = 1280 / max(img.shape[:2])
img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
h, w = img.shape[:2]


def run(H, W):
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:h, :w] = img
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    with torch.inference_mode():
        net(xt)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            out = net(xt)
            ts.append(time.perf_counter() - t)
        sc, lab, b = postprocess(out, 900, 1280)
    keep = sc[0] >= 0.25
    return min(ts) * 1e3, lab[0][keep].numpy(), b[0][keep].numpy(), sc[0][keep].numpy()


t_sq, l_sq, b_sq, s_sq = run(1280, 1280)
enc = net.encoder
enc.eval_spatial_size = None                         # sincos position embedding built for the actual grid
H = int(np.ceil(h / 32) * 32)
t_re, l_re, b_re, s_re = run(H, 1280)
print(f"square 1280x1280: {t_sq:7.0f} ms  {len(l_sq)} boxes")
print(f"rect   {H}x1280 : {t_re:7.0f} ms  {len(l_re)} boxes  ({t_re / t_sq:.2f}x time)")
# match: same class, centre within 2 px
d = np.linalg.norm(b_sq[:, None, :2] - b_re[None, :, :2], axis=-1) + 1e6 * (l_sq[:, None] != l_re[None])
print(f"square boxes with a rect twin (same class, centre < 2 px): {(d.min(1) < 2).mean() * 100:.1f}%"
      if len(l_sq) and len(l_re) else "no boxes")
