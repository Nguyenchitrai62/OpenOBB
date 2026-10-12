"""Measure what each part of a trained OpenOBB3-x actually does on a real Wall_Color page (CPU).

Parts: HGNetv2-B5 backbone, LSK adapters, hybrid encoder (AIFI on P5 + FPN/PAN), dense oriented head (cls tower,
DFL regression), assignment level usage. Output: JSON + printed tables.
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\Source_code\NEW_architecture")
torch.set_num_threads(int(os.environ.get("T", "4")))
from openobb.data.imgio import imread  # noqa: E402
from openobb.models.openobb3 import ALONG_BINS, decode  # noqa: E402
from openobb.ops.obb_torch import probiou  # noqa: E402
from openobb.predict import PAD_BGR, load_model  # noqa: E402

CK = os.environ.get("CK", r"F:\Downloads\vrdet3x_1.pt")
IMG = os.environ.get("IMG", r"F:\Downloads\wall_train_0151.png")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "analysis_v3.json"))
R = {}
t_start = time.time()


def log(*a):
    print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)


model, names, targs = load_model(CK, torch.device("cpu"), 1000)
model.eval()
for _p in model.parameters():
    _p.requires_grad_(False)
S = int(targs.get("img", 1280))
C = len(names)
im0 = imread(IMG)
sc = S / max(im0.shape[:2])
im = cv2.resize(im0, (round(im0.shape[1] * sc), round(im0.shape[0] * sc)), interpolation=cv2.INTER_AREA)
log("image", im0.shape, "->", im.shape, "classes", names)


def canvas(dx=0, dy=0):
    c = np.empty((S, S, 3), np.uint8)
    c[:] = PAD_BGR
    h, w = im.shape[:2]
    c[dy:dy + h, dx:dx + w] = im[:S - dy, :S - dx]
    return c


def to_x(c):
    return torch.from_numpy(c[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


ink = None


def parts(x, hooks=None):
    bb = model.backbone(x)
    lsk = [m(f) for m, f in zip(model.lsk, bb)]
    feats = model.encoder(lsk)
    pts, st = model.anchors(feats)
    cls, reg = model.head(feats)
    boxes = decode(reg, pts, st)
    return bb, lsk, feats, pts, st, cls, reg, boxes


def detect(cls, boxes, conf=0.25, iou=0.7):
    p = cls[0].sigmoid()
    s, c = p.max(-1)
    keep_all = []
    for k in range(C):
        idx = ((c == k) & (s >= conf)).nonzero(as_tuple=True)[0]
        if not len(idx):
            continue
        idx = idx[s[idx].argsort(descending=True)]
        b = boxes[0, idx]
        ov = probiou(b[:, None], b[None])
        keep = torch.triu(ov, 1).amax(0) <= iou
        keep_all += idx[keep].tolist()
    return keep_all


# ------------------------------------------------------------------ 1. cost per part
def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


R["params_M"] = {"backbone": count(model.backbone), "lsk": count(model.lsk), "encoder": count(model.encoder),
                 "head": count(model.head)}
log("params M", R["params_M"])

# ------------------------------------------------------------------ 2. full-image pass with probes
attn_store = []
orig_mha = model.encoder.encoder[0].layers[0].self_attn.forward


def mha_probe(*a, **k):
    k["need_weights"], k["average_attn_weights"] = True, True
    out, w = orig_mha(*a, **k)
    attn_store.append(w.detach())
    return out, w


model.encoder.encoder[0].layers[0].self_attn.forward = mha_probe
lsk_store = []
for li, m in enumerate(model.lsk):
    def gate_hook(mod, inp, out, li=li):
        lsk_store.append((li, out.detach()))
    m.gate.register_forward_hook(gate_hook)
x0 = to_x(canvas())
with torch.no_grad():
    t = time.time()
    bb, lsk, feats, pts, st, cls, reg, boxes = parts(x0)
    log(f"forward {time.time() - t:.1f}s")
model.encoder.encoder[0].layers[0].self_attn.forward = orig_mha
dets = detect(cls, boxes)
n3, n4 = (S // 8) ** 2, (S // 16) ** 2
lvl = lambda a: 0 if a < n3 else (1 if a < n3 + n4 else 2)  # noqa: E731
prob = cls[0].sigmoid()
per = {n: {"n": 0, "levels": [0, 0, 0], "margin": [], "second": {}, "thick_px": [], "thick_std_px": [],
           "along_sat": [], "len_px": []} for n in names}
for a in dets:
    p = prob[a]
    top = p.topk(2)
    c1, c2 = int(top.indices[0]), int(top.indices[1])
    d = per[names[c1]]
    d["n"] += 1
    d["levels"][lvl(a)] += 1
    d["margin"].append(float(top.values[0] - top.values[1]))
    d["second"][names[c2]] = d["second"].get(names[c2], 0) + 1
    s = float(st[a])
    r = reg[0, a]
    pl, pr = r[2:2 + ALONG_BINS].softmax(-1), r[2 + ALONG_BINS:2 + 2 * ALONG_BINS].softmax(-1)
    pt = r[2 + 2 * ALONG_BINS:].softmax(-1)
    bins = torch.arange(pt.numel(), dtype=torch.float32)
    et = float((pt * bins).sum())
    d["thick_px"].append(et * s / 2)
    d["thick_std_px"].append(float(((pt * (bins - et) ** 2).sum()).sqrt()) * s / 2)
    d["along_sat"].append(float(pl[-2:].sum() + pr[-2:].sum()) / 2)
    d["len_px"].append(float(boxes[0, a, 2]))
summ = {}
for n, d in per.items():
    if not d["n"]:
        continue
    summ[n] = {"n": d["n"], "level_share_P3_P4_P5": [round(v / d["n"], 2) for v in d["levels"]],
               "median_margin_top1_top2": round(float(np.median(d["margin"])), 3),
               "low_margin_share(<0.2)": round(float(np.mean(np.array(d["margin"]) < 0.2)), 2),
               "second_class": dict(sorted(d["second"].items(), key=lambda kv: -kv[1])[:3]),
               "median_len_px": round(float(np.median(d["len_px"])), 1),
               "median_thick_px": round(float(np.median(d["thick_px"])), 2),
               "median_thick_uncert_px": round(float(np.median(d["thick_std_px"])), 2),
               "dfl_end_saturation(mass in last 2 bins)": round(float(np.mean(d["along_sat"])), 3)}
R["detections"] = summ
log("detections", json.dumps(summ, indent=1))

# AIFI attention distance (P5, stride 32)
w = attn_store[0][0]                                      # (N, N)
h5 = w5 = S // 32
yy, xx = torch.meshgrid(torch.arange(h5), torch.arange(w5), indexing="ij")
pos = torch.stack([xx.flatten(), yy.flatten()], -1).float() * 32
dist = torch.cdist(pos, pos)
mean_d = float((w * dist).sum(-1).mean())
unif_d = float(dist.mean())
far = float((w * (dist > 160)).sum(-1).mean())
ent = float(-(w.clamp_min(1e-12).log() * w).sum(-1).mean() / math.log(w.shape[-1]))
R["aifi"] = {"tokens": int(w.shape[0]), "mean_attention_distance_px": round(mean_d, 1),
             "uniform_baseline_px": round(unif_d, 1), "mass_beyond_160px": round(far, 3),
             "normalised_entropy": round(ent, 3)}
log("AIFI", R["aifi"])

# LSK: gate on ink vs blank and size of the modulation x * m
gray = cv2.cvtColor(canvas(), cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(canvas(), cv2.COLOR_BGR2HSV)
ink_px = ((gray < 160) | (hsv[..., 1] > 80)).astype(np.float32)
lsk_r = []
for (li, g), xin, xout in zip(lsk_store, bb, lsk):
    s = 8 * 2 ** li
    m = torch.from_numpy(cv2.resize(ink_px, (S // s, S // s), interpolation=cv2.INTER_AREA))
    on, off = m > 0.05, m == 0
    gs = g[0].sigmoid()
    rel = ((xout - xin).abs().mean(1)[0] / (xin.abs().mean(1)[0] + 1e-6))
    lsk_r.append({"level": f"P{3 + li}", "gate_far_on_ink": round(float(gs[1][on].mean()), 3),
                  "gate_far_on_blank": round(float(gs[1][off].mean()), 3),
                  "gate_near_on_ink": round(float(gs[0][on].mean()), 3),
                  "relative_change_on_ink": round(float(rel[on].mean()), 3),
                  "relative_change_on_blank": round(float(rel[off].mean()), 3)})
R["lsk"] = lsk_r
log("LSK", lsk_r)

# encoder: how much it changes the features it receives (P3 output vs its own input projection)
with torch.no_grad():
    proj = [model.encoder.input_proj[i](f) for i, f in enumerate(lsk)]
cos = [round(float(F.cosine_similarity(p.flatten(2), f.flatten(2), dim=1).mean()), 3) for p, f in zip(proj, feats)]
R["encoder_cos_in_out_P3_P4_P5"] = cos
log("encoder cos(in, out)", cos)

# ------------------------------------------------------------------ 3. shift equivariance (stride aliasing)
base = [(a, int(prob[a].argmax()), boxes[0, a].clone()) for a in dets]
jit = {n: {"cx": [], "cy": [], "len": [], "thick": [], "found": 0, "total": 0} for n in names}
shifted = {}
for k in range(1, 8):
    with torch.no_grad():
        _, _, _, _, _, c2, _, b2 = parts(to_x(canvas(k, k)))
    d2 = detect(c2, b2)
    shifted[k] = (c2[0].sigmoid().argmax(-1)[d2], b2[0, d2].clone())
    log("shift", k, len(d2), "dets")
for a, c, b in base:
    n = names[c]
    errs = []
    for k, (cl2, bx2) in shifted.items():
        jit[n]["total"] += 1
        m = cl2 == c
        if not m.any():
            continue
        bb2 = bx2[m].clone()
        bb2[:, :2] -= k
        iou = probiou(b[None], bb2)
        j = int(iou.argmax())
        if iou[j] < 0.5:
            continue
        jit[n]["found"] += 1
        errs.append(bb2[j] - b)
    if len(errs) >= 3:
        e = torch.stack(errs)
        jit[n]["cx"].append(float(e[:, 0].std()))
        jit[n]["cy"].append(float(e[:, 1].std()))
        jit[n]["len"].append(float(e[:, 2].std()))
        jit[n]["thick"].append(float(e[:, 3].std()))
R["shift_jitter_px"] = {n: {"redetected_share": round(d["found"] / max(d["total"], 1), 2),
                            "center_std": round(float(np.median(d["cx"] + d["cy"])), 2) if d["cx"] else None,
                            "length_std": round(float(np.median(d["len"])), 2) if d["len"] else None,
                            "thick_std": round(float(np.median(d["thick"])), 2) if d["thick"] else None}
                        for n, d in jit.items() if d["total"]}
log("shift jitter", json.dumps(R["shift_jitter_px"], indent=1))

# ------------------------------------------------------------------ 4. effective receptive field of the class logit
model.encoder.eval_spatial_size = None                   # crops: positional embedding computed for their size
CROP = 512
erf = {}
want = ["wall_300", "junction", "wall", "door"]
for n in want:
    ci = names.index(n)
    cand = [(a, b) for a, c, b in base if c == ci][:3]
    rows = []
    for a, b in cand:
        cx, cy = float(b[0]), float(b[1])
        x0, y0 = int(min(max(cx - CROP / 2, 0), S - CROP)), int(min(max(cy - CROP / 2, 0), S - CROP))
        crop = canvas()[y0:y0 + CROP, x0:x0 + CROP]
        x = to_x(crop).requires_grad_(True)
        _, _, fe, p_, s_, c_, r_, b_ = parts(x)
        # the anchor of this object in the crop: same class, best overlap with the full-page box
        bl = b.clone()
        bl[0] -= x0
        bl[1] -= y0
        ov = probiou(bl[None], b_[0].detach()) * (c_[0].detach().sigmoid().argmax(-1) == ci)
        aa = int(ov.argmax())
        if ov[aa] < 0.5:
            continue
        g, = torch.autograd.grad(c_[0, aa, ci], x)
        gm = g[0].abs().sum(0).numpy()
        gm /= gm.sum()
        ys, xs = np.mgrid[:CROP, :CROP]
        px, py = float(p_[aa, 0]), float(p_[aa, 1])
        r = np.hypot(xs - px, ys - py).ravel()
        o = np.argsort(r)
        cm = np.cumsum(gm.ravel()[o])
        r50, r90 = float(r[o][np.searchsorted(cm, 0.5)]), float(r[o][np.searchsorted(cm, 0.9)])
        # share of the evidence outside the object's own box (grown by 2 px)
        bb_ = b_[0, aa].detach()
        c0, s0 = math.cos(float(bb_[4])), math.sin(float(bb_[4]))
        u = (xs - float(bb_[0])) * c0 + (ys - float(bb_[1])) * s0
        v = -(xs - float(bb_[0])) * s0 + (ys - float(bb_[1])) * c0
        inside = (np.abs(u) <= float(bb_[2]) / 2 + 2) & (np.abs(v) <= float(bb_[3]) / 2 + 2)
        rows.append({"len_px": round(float(bb_[2]), 1), "thick_px": round(float(bb_[3]), 1),
                     "level": f"P{int(math.log2(float(s_[aa])))}",
                     "r50_px": round(r50, 1), "r90_px": round(r90, 1),
                     "evidence_outside_box": round(float(gm[~inside].sum()), 3)})
        log("ERF", n, rows[-1])
    erf[n] = rows
R["erf_class_logit"] = erf

json.dump(R, open(OUT, "w"), indent=1)
log("saved", OUT)
