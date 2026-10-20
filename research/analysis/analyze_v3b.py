"""Second probe: the P5 path (LSK / input projection) magnitudes in every v3 checkpoint, what the detections lose when
the P5 LSK is bypassed, the wall thickness distributions, and FLOPs per sub-block."""
import json
import os
import sys
from pathlib import Path
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(4)
from openobb.data.imgio import imread  # noqa: E402
from openobb.models.openobb3 import ALONG_BINS, decode  # noqa: E402
from openobb.ops.obb_torch import probiou  # noqa: E402
from openobb.predict import PAD_BGR, load_model  # noqa: E402

IMG = r"F:\Downloads\wall_train_0151.png"
OUT = os.path.join(os.path.dirname(__file__), "analysis_v3b.json")
R = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def prep(S):
    im0 = imread(IMG)
    sc = S / max(im0.shape[:2])
    im = cv2.resize(im0, (round(im0.shape[1] * sc), round(im0.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    c = np.empty((S, S, 3), np.uint8)
    c[:] = PAD_BGR
    c[:im.shape[0], :im.shape[1]] = im
    return torch.from_numpy(c[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def detect(model, cls, boxes, C, conf=0.25):
    p = cls[0].sigmoid()
    s, c = p.max(-1)
    out = []
    for k in range(C):
        idx = ((c == k) & (s >= conf)).nonzero(as_tuple=True)[0]
        if len(idx):
            idx = idx[s[idx].argsort(descending=True)]
            b = boxes[0, idx]
            keep = torch.triu(probiou(b[:, None], b[None]), 1).amax(0) <= 0.7
            out += idx[keep].tolist()
    return out


def fwd(model, x, bypass_p5_lsk=False):
    bb = model.backbone(x)
    lsk = [m(f) for m, f in zip(model.lsk, bb)]
    if bypass_p5_lsk:
        lsk[2] = bb[2]
    feats = model.encoder(lsk)
    pts, st = model.anchors(feats)
    cls, reg = model.head(feats)
    return bb, lsk, feats, pts, st, cls, reg, decode(reg, pts, st)


# ---------------------------------------------------------------- P5 path in every checkpoint
x = None
for name in ("vrdet3x_1.pt", "vrdet3x_2.pt", "vrdet3x_3.pt"):
    model, names, targs = load_model(rf"F:\Downloads\{name}", torch.device("cpu"), 1000)
    S = int(targs.get("img", 1280))
    x = prep(S)
    with torch.no_grad():
        bb, lsk, feats, pts, st, cls, reg, boxes = fwd(model, x)
        proj = [model.encoder.input_proj[i](f) for i, f in enumerate(lsk)]
    row = {}
    for i in range(3):
        bn = model.encoder.input_proj[i][1] if isinstance(model.encoder.input_proj[i], torch.nn.Sequential) else None
        row[f"P{3 + i}"] = {"backbone_abs": float(bb[i].abs().mean()), "after_lsk_abs": float(lsk[i].abs().mean()),
                            "lsk_out_weight_norm": float(model.lsk[i].out.weight.norm()),
                            "input_proj_abs": float(proj[i].abs().mean()),
                            "encoder_out_abs": float(feats[i].abs().mean()),
                            "proj_bn_running_var_max": float(bn.running_var.max()) if bn is not None and hasattr(bn, "running_var") else None}
    R[name] = row
    log(name, json.dumps(row))

# ---------------------------------------------------------------- bypass the P5 LSK (best checkpoint)
model, names, targs = load_model(r"F:\Downloads\vrdet3x_1.pt", torch.device("cpu"), 1000)
S, C = int(targs.get("img", 1280)), len(names)
x = prep(S)
with torch.no_grad():
    a = fwd(model, x)
    b = fwd(model, x, bypass_p5_lsk=True)
da, db = detect(model, a[5], a[7], C), detect(model, b[5], b[7], C)
ca = a[5][0].sigmoid().argmax(-1)
cb = b[5][0].sigmoid().argmax(-1)
cnt_a = {n: int((ca[da] == i).sum()) for i, n in enumerate(names)}
cnt_b = {n: int((cb[db] == i).sum()) for i, n in enumerate(names)}
# per detection of the normal pass: is the same object found (same class, ProbIoU >= 0.5) without the P5 LSK?
kept = defaultdict(lambda: [0, 0])
for i in da:
    n = names[int(ca[i])]
    kept[n][1] += 1
    m = cb[db] == ca[i]
    if m.any() and float(probiou(a[7][0, i][None], b[7][0, db][m]).max()) >= 0.5:
        kept[n][0] += 1
R["bypass_p5_lsk"] = {"count_normal": cnt_a, "count_bypass": cnt_b,
                      "same_objects_found": {n: f"{v[0]}/{v[1]}" for n, v in kept.items()}}
log("bypass P5 LSK", json.dumps(R["bypass_p5_lsk"]))

# ---------------------------------------------------------------- thickness distribution shape of walls
reg, st = a[6][0], a[4]
peaks = defaultdict(list)
for i in da:
    n = names[int(ca[i])]
    pt = reg[i, 2 + 2 * ALONG_BINS:].softmax(-1)
    p = pt.numpy()
    loc = [j for j in range(1, len(p) - 1) if p[j] > p[j - 1] and p[j] >= p[j + 1] and p[j] > 0.08]
    peaks[n].append((len(loc), float(p.max()), [round(float(j * st[i] / 2), 1) for j in loc]))
R["thickness_peaks"] = {n: {"share_multi_peak": round(float(np.mean([v[0] >= 2 for v in vs])), 2),
                            "median_peak_prob": round(float(np.median([v[1] for v in vs])), 2),
                            "examples_px": [v[2] for v in vs[:6]]} for n, vs in peaks.items()}
log("thickness", json.dumps(R["thickness_peaks"]))

# ---------------------------------------------------------------- FLOPs per sub-block at the training size
flops = {}
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    fwd(model, x)
counts = fc.get_flop_counts()
want = ["OpenOBB3.backbone.stem"] + [f"OpenOBB3.backbone.stages.{i}" for i in range(4)] + \
       [f"OpenOBB3.lsk.{i}" for i in range(3)] + \
       [f"OpenOBB3.encoder.input_proj.{i}" for i in range(3)] + ["OpenOBB3.encoder.encoder"] + \
       [f"OpenOBB3.encoder.{k}" for k in ("lateral_convs", "fpn_blocks", "downsample_convs", "pan_blocks")] + \
       ["OpenOBB3.head.reg", "OpenOBB3.head.cls"]
tot = sum(counts["Global"].values())
for k in want:
    v = sum(counts.get(k, {}).values())
    flops[k.replace("OpenOBB3.", "")] = round(v / 1e9, 1)            # 2 x MAC, as Ultralytics "GFLOPs"
flops["total"] = round(tot / 1e9, 1)
R["gflops"] = flops
log("GFLOPs", json.dumps(flops))
json.dump(R, open(OUT, "w"), indent=1)
log("saved")
