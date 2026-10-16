"""Inference head x duplicate suppression on Wall_Color val with one checkpoint (product predict path, conf 0.001):
mAP50 / mAP50-95, duplicate boxes, box geometry, and the CPU wall time of the post-processing. F95.

usage: python postproc_compare.py <checkpoint.pt> head:nms ...   (e.g. o2m:0.5 o2o:peak o2m:peak o2o:none)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import confusion, duplicate_line, evaluate_ultra, geometry, geometry_line  # noqa: E402
from openobb.models import openobb17  # noqa: E402
from openobb.predict import apply_head, apply_nms  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
ck, combos = sys.argv[1], sys.argv[2:]
m = OpenOBB(ck)
import os  # noqa: E402
if os.environ.get("ROT_GAIN"):                   # oracle of OpenOBB23's rotational snapping on a trained field
    from openobb.models.openobb18 import BoundarySnapRot
    from openobb.models.openobb17 import GAIN_SCALE
    snap = BoundarySnapRot()
    snap.load_state_dict(m.model.snap.state_dict(), strict=False)
    with torch.no_grad():
        snap.rot_gain.fill_(float(os.environ["ROT_GAIN"]) / GAIN_SCALE)
    m.model.snap = snap.to(next(m.model.parameters()).device)
    print(f"rotational snapping on, gain {os.environ['ROT_GAIN']}")
names = [m.names[i] for i in range(len(m.names))]
imgs = sorted((ROOT / "images" / "val").iterdir())
gts = {}
for p in imgs:
    h, w = imread(str(p)).shape[:2]
    lab = ROOT / "labels" / "val" / f"{p.stem}.txt"
    rows = [r.split() for r in lab.read_text().splitlines() if len(r.split()) >= 9]
    gts[p.stem] = [(np.array(r[1:9], float) * np.tile([w, h], 4), names[int(r[0])], 0) for r in rows]

post_t = [0.0]
_nms, _peak = openobb17.fast_nms, openobb17.peak_mask


def timed(fn):
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        post_t[0] += time.perf_counter() - t
        return r
    return w


openobb17.fast_nms, openobb17.peak_mask = timed(_nms), timed(_peak)
for combo in combos:
    head, nms = combo.split(":")
    apply_head(m.model, head)
    apply_nms(m.model, nms)
    post_t[0] = 0.0
    dets, per_img = {c: ([], [], []) for c in names}, {}
    for p in imgs:
        r = m.predict(str(p), conf=0.001, max_det=1000, iou=nms, save=False, verbose=False)[0].obb   # iou= overrides the calibrated setting
        sc, cl = r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)
        pl = r.xyxyxyxy.cpu().numpy().reshape(-1, 8).astype(np.float64)
        per_img[p.stem] = [(float(s), int(c), q) for s, c, q in zip(sc, cl, pl)]
        for s, c, q in zip(sc, cl, pl):
            ids, ss, pp = dets[names[c]]
            ids.append(p.stem), ss.append(float(s)), pp.append(q)
    d = {c: (i, np.asarray(s), np.asarray(pp).reshape(-1, 8)) for c, (i, s, pp) in dets.items()}
    u = evaluate_ultra(d, gts, names, max_det=1000)
    print(f"\n{combo}: mAP50 {u['mAP50']:.3f}  mAP50-95 {u['mAP50_95']:.3f}  | post-processing CPU {post_t[0] / len(imgs) * 1000:.0f} ms / image")
    print("  " + duplicate_line(confusion(per_img, gts, names)))
    print("  " + geometry_line(geometry(per_img, gts, names)), flush=True)
