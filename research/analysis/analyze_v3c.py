"""Third probe: (1) zero the backbone's P5 map -> do detections change? (2) FLOPs per sub-block."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(4)
from openobb.data.imgio import imread  # noqa: E402
from openobb.models.openobb3 import decode  # noqa: E402
from openobb.ops.obb_torch import probiou  # noqa: E402
from openobb.predict import PAD_BGR, load_model  # noqa: E402

model, names, targs = load_model(r"F:\Downloads\vrdet3x_1.pt", torch.device("cpu"), 1000)
S, C = int(targs["img"]), len(names)
im0 = imread(r"F:\Downloads\wall_train_0151.png")
sc = S / max(im0.shape[:2])
im = cv2.resize(im0, (round(im0.shape[1] * sc), round(im0.shape[0] * sc)), interpolation=cv2.INTER_AREA)
c = np.empty((S, S, 3), np.uint8)
c[:] = PAD_BGR
c[:im.shape[0], :im.shape[1]] = im
x = torch.from_numpy(c[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def run(mode):
    bb = model.backbone(x)
    if mode == "zero_p5":
        bb[2] = torch.zeros_like(bb[2])
    lsk = [m(f) for m, f in zip(model.lsk, bb)]
    if mode == "zero_p5":
        lsk[2] = torch.zeros_like(lsk[2])
    feats = model.encoder(lsk)
    pts, st = model.anchors(feats)
    cls, reg = model.head(feats)
    return cls, decode(reg, pts, st)


def detect(cls, boxes, conf=0.25):
    p = cls[0].sigmoid()
    s, k = p.max(-1)
    out = []
    for j in range(C):
        idx = ((k == j) & (s >= conf)).nonzero(as_tuple=True)[0]
        if len(idx):
            idx = idx[s[idx].argsort(descending=True)]
            b = boxes[0, idx]
            out += idx[torch.triu(probiou(b[:, None], b[None]), 1).amax(0) <= 0.7].tolist()
    return out


with torch.no_grad():
    ca, ba = run("normal")
    cz, bz = run("zero_p5")
da, dz = detect(ca, ba), detect(cz, bz)
la, lz = ca[0].sigmoid().argmax(-1), cz[0].sigmoid().argmax(-1)
res = {}
for j, n in enumerate(names):
    ia = [i for i in da if la[i] == j]
    iz = [i for i in dz if lz[i] == j]
    same = 0
    for i in ia:
        if iz and float(probiou(ba[0, i][None], bz[0, iz]).max()) >= 0.5:
            same += 1
    res[n] = {"normal": len(ia), "p5_zeroed": len(iz), "same_objects": same}
print("zero backbone P5:", json.dumps(res))
ds = (ca[0].sigmoid() - cz[0].sigmoid()).abs().max()
print("max |score change| over all anchors:", float(ds))

# FLOPs per module (keys are module paths of the traced call)
with torch.no_grad(), FlopCounterMode(display=False, depth=4) as fc:
    run("normal")
cnt = fc.get_flop_counts()
rows = {k: sum(v.values()) / 1e9 for k, v in cnt.items()}
tot = rows.get("Global", sum(rows.values()))
keys = sorted(k for k in rows if k != "Global")
print("total GFLOPs", round(tot, 1))
for k in keys:
    if rows[k] >= 1:
        print(f"{k:55s} {rows[k]:8.1f}  {100 * rows[k] / tot:5.1f}%")
