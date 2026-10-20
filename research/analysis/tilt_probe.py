"""Where does the tilt of long boxes come from inside a dense OpenOBB (v17-v20)? Hooks snap_topk on val images and,
for confident long candidates (score >= 0.25, length >= 100 px and >= 10 x thickness; the val labels of such walls are
all within 0.5 deg of an axis), reports the deviation from the nearest axis:
  - per stage: head output, after the edge refiner (v20), after boundary snapping;
  - by pyramid level of the anchor that predicted the box (P3 / P4 / P5);
  - by the anchor's position along the box (|u| = distance from the box centre / half length: 0 centre, 1 end);
  - by the box length; and the raw angle range (bf16 resolution is 0.45 deg on [1, 2) rad, 0.9 deg on [2, 4) rad).
usage: python tilt_probe.py <checkpoint.pt> [n_images]
"""
import os
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
ck, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
if os.environ.get("PROBE_NOFUSE"):                    # load without fusing BN / RepConv branches
    import openobb.predict as _pr
    import openobb.models.openobb7 as _o7
    _pr.fuse_model = lambda m: m
    for _c in __import__("openobb.models.openobb18", fromlist=["x"]).__dict__.values():
        if isinstance(_c, type) and hasattr(_c, "fuse"):
            _c.fuse = lambda self: self
net = OpenOBB(ck).model
rec = []
_snap_topk = net.snap_topk


def snap_topk(field, cls, boxes, k, f4=None, p3=None):
    out, idx = _snap_topk(field, cls, boxes, k, f4, p3)
    cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
    stages = net.refine_stages(f4, p3, field, cand)
    sc = cls.gather(1, idx[..., None].expand(-1, -1, cls.shape[-1])).sigmoid().amax(-1)
    rec.append({"score": sc[0].float().numpy(), "idx": idx[0].numpy(), "head": cand[0].float().numpy(),
                "stages": [s[1][0].float().detach().numpy() for s in stages]})
    return out, idx


net.snap_topk = snap_topk
_anchors = net.anchors
anc = {}


def anchors(feats):
    pts, st = _anchors(feats)
    anc["pts"], anc["st"] = pts.float().numpy(), st.float().numpy()
    return pts, st


net.anchors = anchors


def dev(th):
    return np.abs(np.rad2deg((th + np.pi / 4) % (np.pi / 2) - np.pi / 4))


rows = []
for p in sorted((ROOT / "images" / os.environ.get("PROBE_SPLIT", "val")).iterdir())[:n]:
    img = imread(str(p))
    if os.environ.get("PROBE_ROT90"):                  # rotate the drawing: horizontal walls become vertical
        img = np.ascontiguousarray(np.rot90(img))
    s = 1280 / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    H, W = rect_shape(*img.shape[:2], 1280)
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    rec.clear()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(os.environ.get("PROBE_BF16"))):
        net(torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255))
    r = rec[0]
    b = r["head"]
    L, T = np.maximum(b[:, 2], b[:, 3]), np.minimum(b[:, 2], b[:, 3])
    th = np.where(b[:, 2] >= b[:, 3], b[:, 4], b[:, 4] + np.pi / 2)
    keep = (r["score"] >= 0.25) & (L >= 100) & (L >= 10 * T)
    ai = r["idx"] % len(anc["pts"])                       # grouped models: candidates are (group, anchor)
    pts, st = anc["pts"][ai], anc["st"][ai]
    ax = np.stack([np.cos(th), np.sin(th)], -1)
    u = np.abs(((pts - b[:, :2]) * ax).sum(-1)) / (L / 2)
    for i in np.nonzero(keep)[0]:
        st_dev = [dev(np.array([s_[i, 4] if s_[i, 2] >= s_[i, 3] else s_[i, 4] + np.pi / 2]))[0] for s_ in r["stages"]]
        rows.append((dev(th[i:i + 1])[0], *st_dev, st[i], u[i], L[i], b[i, 4], abs(np.sin(th[i]))))
    print(f"{p.stem}: {int(keep.sum())} long candidates ({int((keep & (np.abs(np.sin(th)) > 0.7)).sum())} vertical), input {x.shape[:2]}", flush=True)

a = np.array(rows)
vert = a[:, -1] > 0.7                                  # wall vertical in the image
a = a[:, :-1]
k_head, k_last = 0, a.shape[1] - 5
pc = lambda x: f"{np.mean(x):5.1%}" if len(x) else "  n/a"  # noqa: E731
print(f"\n{Path(ck).name}: {len(a)} confident long candidates (score >= 0.25, L >= 100 px, L >= 10 T) on {n} val images")
names = ["head"] + [f"stage {j + 1}" for j in range(a.shape[1] - 5)]
for j, nm in enumerate(names):
    print(f"  {nm:8s} > 1 deg {pc(a[:, j] > 1)}  > 3 deg {pc(a[:, j] > 3)}  median {np.median(a[:, j]):.2f}")
lv = a[:, -4]
for s_ in np.unique(lv):
    k = lv == s_
    print(f"  stride {int(s_):2d}: n {int(k.sum()):4d} head > 1 deg {pc(a[k, 0] > 1)}  final > 1 deg {pc(a[k, k_last] > 1)}")
for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 2)):
    k = (a[:, -3] >= lo) & (a[:, -3] < hi)
    print(f"  anchor |u| {lo:.2f}-{hi:.2f}: n {int(k.sum()):4d} head > 1 deg {pc(a[k, 0] > 1)}  final > 1 deg {pc(a[k, k_last] > 1)}")
for lo, hi in ((100, 200), (200, 400), (400, 10**5)):
    k = (a[:, -2] >= lo) & (a[:, -2] < hi)
    print(f"  length {lo}-{hi if hi < 10**5 else ''}: n {int(k.sum()):4d} head > 1 deg {pc(a[k, 0] > 1)}  final > 1 deg {pc(a[k, k_last] > 1)}")
raw = np.abs(a[:, -1])
print(f"  vertical in the image: n {int(vert.sum())} head > 1 deg {pc(a[vert, 0] > 1)} | horizontal: n {int((~vert).sum())} "
      f"head > 1 deg {pc(a[~vert, 0] > 1)} | head angle near +-pi/2 among vertical walls {pc(raw[vert] >= 1)}")
for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 100)):
    k = (raw >= lo) & (raw < hi)
    print(f"  |raw angle| {lo}-{hi} rad: n {int(k.sum()):4d} head > 1 deg {pc(a[k, 0] > 1)}")
# vertical walls (raw angle near +-pi/2, the edge of the le90 range): is the head's angle pulled towards 0 (|raw| < pi/2)?
k = (raw >= 1) & (raw < 2)
if k.any():
    inside = raw[k] < np.pi / 2
    print(f"  vertical (|raw| 1-2 rad): |raw| < pi/2 (pulled towards 0) {pc(inside)}; of the tilted > 1 deg: "
          f"{pc(inside[a[k, 0] > 1])}; median |raw| - pi/2 = {np.rad2deg(np.median(raw[k] - np.pi / 2)):+.2f} deg")
# two basins: a vertical object is right at +pi/2 and at -pi/2; does the head split its anchors between the two, and do
# the two sides tilt in opposite directions (the X pairs seen on the drawings)?
k = (raw >= 1) & (raw < 2)
if k.any():
    sgn = np.sign(a[k, -1])
    signed = np.rad2deg(np.abs(a[k, -1]) - np.pi / 2) * sgn              # >0: rotated towards +, in image terms
    print(f"  vertical: raw angle > 0 on {pc(sgn > 0)} of the candidates; tilt (deg, image sense) median on the + side "
          f"{np.median(signed[sgn > 0]) if (sgn > 0).any() else float('nan'):+.2f}, on the - side "
          f"{np.median(signed[sgn < 0]) if (sgn < 0).any() else float('nan'):+.2f}")
