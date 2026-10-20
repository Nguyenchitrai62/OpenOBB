"""How much does the OpenOBB20 edge refiner rotate / move the candidate boxes? Records, for every refined candidate
(top-`ref_k`), the change applied by the refiner: angle (deg), across-axis centre shift (px), end moves (px),
thickness ratio. Val images (in distribution) vs other drawings (out of distribution). CPU.

usage: python refiner_shift.py <openobb20 .pt> <n_val> [image ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.predict import PAD_BGR, rect_shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color"
ck, n_val, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
m = OpenOBB(ck)
net = m.model
rec = []
_fwd = net.edge.forward


def hook(f4, p3, b):
    r = _fwd(f4, p3, b)
    rec.append((b.detach().float().clone(), r.detach().float().clone()))
    return r


net.edge.forward = hook


def stats(pairs, conf_scores=None):
    b = torch.cat([p[0].reshape(-1, 5) for p in pairs])
    r = torch.cat([p[1].reshape(-1, 5) for p in pairs])
    dth = torch.rad2deg((r[:, 4] - b[:, 4] + torch.pi / 2) % torch.pi - torch.pi / 2).abs()
    d = r[:, :2] - b[:, :2]
    u = torch.stack([b[:, 4].cos(), b[:, 4].sin()], -1)
    across = (d[:, 0] * -u[:, 1] + d[:, 1] * u[:, 0]).abs()
    dl = (r[:, 2] - b[:, 2]).abs()
    tr = r[:, 3] / b[:, 3].clamp(min=1e-3)
    q = lambda x: " ".join(f"{float(x.quantile(p)):6.2f}" for p in (0.5, 0.9, 0.99))  # noqa: E731
    s = (f"n {len(b):6d} | angle deg p50/90/99 {q(dth)} | across px {q(across)} | length px {q(dl)} | "
         f"thick ratio {q(tr)} | angle > 3 deg {float((dth > 3).float().mean()):.1%}")
    # by length: the sideways move of a box end = across shift + (L / 2) sin(rotation)
    L = torch.maximum(b[:, 2], b[:, 3])
    swing = across + L / 2 * torch.deg2rad(dth).sin()
    for lo, hi in ((0, 40), (40, 100), (100, 200), (200, 10**5)):
        k = (L >= lo) & (L < hi)
        if k.sum() > 10:
            s += (f"\n    L {lo:4d}-{hi:<6d} n {int(k.sum()):6d}  angle p50/p90 {float(dth[k].median()):5.2f}/"
                  f"{float(dth[k].quantile(.9)):5.2f} deg  end sideways move p50/p90 {float(swing[k].median()):5.2f}/"
                  f"{float(swing[k].quantile(.9)):6.2f} px")
    return s


def run(path):
    img = imread(str(path))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = rect_shape(*img.shape[:2], 1280)
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    rec.clear()
    with torch.inference_mode():
        net(xt)
    return list(rec)


val = sorted((ROOT / "images" / "val").iterdir())[:n_val]
allv = []
for p in val:
    allv += run(p)
print("val (in distribution):", stats(allv))
for p in extra:
    print(f"{Path(p).name[:40]:40s}:", stats(run(p)))
