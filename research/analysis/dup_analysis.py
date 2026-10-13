"""Where do the duplicate boxes of a trained dense model come from, and which suppression removes them?
Val images of Wall_Color, the model's raw boxes before NMS (model NMS disabled), then several suppression rules,
each scored with the Ultralytics-style mAP and the duplicate count (conf 0.25). CPU inference only.

usage: python dup_analysis.py <checkpoint.pt> [n_images]   (DUP_HEAD=o2m: dual-head models use the one-to-many head)
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import confusion, evaluate_ultra  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402
from openobb.predict import PAD_BGR, rect_shape  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
ck, n_img = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
m = OpenOBB(ck)
net, names = m.model, [m.names[i] for i in range(len(m.names))]
net.nms_iou, net.nms_conf, net.max_det = 1.01, 0.05, 3000          # model returns its raw candidates (no class NMS)
if hasattr(net, "final_nms"):                 # OpenOBB12 / 13: one-to-one head output, no fallback NMS
    net.final_nms = None
    if os.environ.get("DUP_HEAD") == "o2m":        # same checkpoint, one-to-many head (what v10 used) instead
        net.head_o2o = net.head
        if hasattr(net, "p2_head_o2o"):
            net.p2_head_o2o = net.p2_head

if os.environ.get("DUP_DECODE") == "peak":       # DFL read around the distribution's peak, not its full mean
    from openobb.models import openobb3 as _m3

    def _peak(logits, width):
        p = logits.softmax(-1)
        n = p.shape[-1]
        raw_i = p.argmax(-1, keepdim=True) + torch.arange(-1, 2, device=p.device)   # bins peak-1, peak, peak+1
        valid = (raw_i >= 0) & (raw_i < n)
        idx = raw_i.clamp(0, n - 1)
        w = p.gather(-1, idx) * valid
        return (w * idx.to(p.dtype)).sum(-1) / w.sum(-1).clamp(min=1e-9) * width

    _m3._expect = _peak                                             # decode() looks _expect up at call time
raw, gts = {}, {}
imgs = sorted((ROOT / "images" / "val").iterdir())[:n_img]
t0 = time.time()
CACHE = Path(__file__).with_name(f"dup_raw_{Path(ck).stem}{'_' + os.environ['DUP_HEAD'] if os.environ.get('DUP_HEAD') else ''}{'_' + os.environ['DUP_DECODE'] if os.environ.get('DUP_DECODE') else ''}_{len(imgs)}.npz")
cached = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
for k, p in enumerate(imgs):
    if p.stem in cached:
        raw[p.stem], gts[p.stem] = cached[p.stem].item()["raw"], cached[p.stem].item()["gt"]
        continue
    img = imread(str(p))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = rect_shape(*img.shape[:2], 1280)
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    with torch.inference_mode():
        out = net(xt)
    lg, bx = out["pred_logits"][0].float(), out["pred_boxes"][0].float()
    sc, cl = lg.sigmoid().max(-1)
    keep = sc > 0.05
    b = bx[keep].clone()
    b[:, :4] *= net.img_size
    raw[p.stem] = (sc[keep].numpy(), cl[keep].numpy(), b.numpy())
    lab = ROOT / "labels" / "val" / f"{p.stem}.txt"
    g = []
    for line in lab.read_text().splitlines() if lab.exists() else []:
        v = line.split()
        if len(v) >= 9:
            g.append((np.array(v[1:9], float) * np.tile([w0 * s, h0 * s], 4), names[int(v[0])], 0))
    gts[p.stem] = g
    print(f"{k + 1}/{len(imgs)} {p.stem}: {int(keep.sum())} raw boxes ({time.time() - t0:.0f}s)", flush=True)
if not cached:
    np.savez(CACHE, **{k: {"raw": raw[k], "gt": gts[k]} for k in raw})


def poly(b):
    return Polygon(obb2poly(np.asarray(b, np.float64)[None])[0].reshape(4, 2)).buffer(0)


def suppress(sc, cl, b, iou=0.7, min_side=0.0, ioa=None, max_det=300):
    """class-wise Fast-NMS on ProbIoU (the model's), then optional containment suppression: a box whose area lies
    >= ioa inside a higher-scored kept box of the same class is dropped."""
    if not len(sc):
        return sc, cl, b
    logits = torch.zeros(1, len(sc), len(names)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
    L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=iou, min_side=min_side)
    s2, c2 = L[0].sigmoid().max(-1)
    k = s2 > 0.05
    sc, cl, b = s2[k].numpy(), c2[k].numpy(), B[0][k].numpy()
    if ioa is not None:
        order = np.argsort(-sc)
        polys = [poly(x) for x in b]
        kept = []
        for i in order:
            pi = polys[i]
            if pi.area <= 0 or any(cl[j] == cl[i] and pi.intersection(polys[j]).area / pi.area >= ioa for j in kept):
                continue
            kept.append(i)
        kept = np.array(kept, int)
        sc, cl, b = sc[kept], cl[kept], b[kept]
    o = np.argsort(-sc)[:max_det]
    return sc[o], cl[o], b[o]


def oracle(stem, sc, cl, b):
    """Perfect duplicate removal: a detection matched (poly IoU >= 0.5, same class) to an object already taken by a
    higher-scored detection is dropped; everything else (true detections, false positives) is kept."""
    g = gts[stem]
    gp = [Polygon(np.asarray(q).reshape(4, 2)).buffer(0) for q, _, _ in g]
    taken = set()
    keep = []
    for i in np.argsort(-sc):
        pi = poly(b[i])
        best, bj, frag = 0.0, -1, False
        for j, (q, c, _) in enumerate(g):
            if names.index(c) != cl[i] or not pi.intersects(gp[j]):
                continue
            inter = pi.intersection(gp[j]).area
            u = pi.area + gp[j].area - inter
            if u > 0 and inter / u > best:
                best, bj = inter / u, j
            if j in taken and pi.area > 0 and inter / pi.area >= 0.5:
                frag = True                              # mostly inside an object already found: a fragment / copy
        if best >= 0.5 and bj not in taken:
            taken.add(bj)
        elif (best >= 0.5 and bj in taken) or frag:
            continue
        keep.append(i)
    keep = np.array(keep, int)
    return sc[keep], cl[keep], b[keep]


def score(rule):
    dets, per_img = {}, {}
    for stem, (sc, cl, b) in raw.items():
        rule = dict(rule)
        orc = rule.pop("oracle", False)
        s2, c2, b2 = suppress(sc, cl, b, **rule)
        if orc:
            s2, c2, b2 = oracle(stem, s2, c2, b2)
        pl = obb2poly(b2.astype(np.float64)) if len(b2) else np.zeros((0, 8))
        per_img[stem] = [(float(s), int(c), p) for s, c, p in zip(s2, c2, pl)]
        for s, c, p in zip(s2, c2, pl):
            ids, ss, pp = dets.setdefault(names[int(c)], ([], [], []))
            ids.append(stem), ss.append(float(s)), pp.append(p)
    dets = {k: (v[0], np.array(v[1]), np.array(v[2])) for k, v in dets.items()}
    r = evaluate_ultra(dets, gts, names)
    cm = confusion(per_img, gts, names)
    return r, cm


rules = {"model default (ProbIoU NMS 0.7)": {}, "NMS 0.6": {"iou": 0.6}, "NMS 0.5": {"iou": 0.5}, "NMS 0.4": {"iou": 0.4}, "NMS 0.3": {"iou": 0.3}, "NMS 0.5 + containment 0.8": {"iou": 0.5, "ioa": 0.8}, "NMS 0.7 + floor 12 px": {"min_side": 12.0},
         "NMS 0.7 + containment 0.8": {"ioa": 0.8}, "ORACLE: model NMS + perfect duplicate removal": {"oracle": True}, "NMS 0.7 + floor 12 + containment 0.8": {"min_side": 12.0, "ioa": 0.8}}
res = {}
for name, rule in rules.items():
    r, cm = score(rule)
    res[name] = {"mAP50": round(r["mAP50"], 4), "mAP50_95": round(r["mAP50_95"], 4), "dups": cm["duplicates"],
                 "detections": cm["detections"],
                 "classes": {c: (round(v["AP50"], 3), round(v["AP50_95"], 3)) for c, v in r["classes"].items()}}
    print(f"{name:42s} mAP50 {r['mAP50']:.4f} mAP50-95 {r['mAP50_95']:.4f} | dups {sum(cm['duplicates'].values())} "
          f"of {cm['detections']} | {cm['duplicates']}", flush=True)

# typology of duplicate pairs left by the model's own NMS: same class, conf >= 0.25, sharing area
pairs = {"contained (short inside long)": 0, "near-identical (IoU >= 0.5)": 0, "partial overlap": 0}
for stem, (sc, cl, b) in raw.items():
    s2, c2, b2 = suppress(sc, cl, b)
    k = s2 >= 0.25
    s2, c2, b2 = s2[k], c2[k], b2[k]
    P = [poly(x) for x in b2]
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if c2[i] != c2[j] or not P[i].intersects(P[j]):
                continue
            inter = P[i].intersection(P[j]).area
            small = min(P[i].area, P[j].area)
            if small <= 0 or inter / small < 0.3:
                continue
            iou = inter / (P[i].area + P[j].area - inter)
            pairs["near-identical (IoU >= 0.5)" if iou >= 0.5 else
                  "contained (short inside long)" if inter / small >= 0.8 else "partial overlap"] += 1
print("same-class overlapping pairs after the model's NMS:", pairs, flush=True)
out = Path(__file__).with_name(f"dup_analysis_{Path(ck).stem}{'_' + os.environ['DUP_HEAD'] if os.environ.get('DUP_HEAD') else ''}{'_' + os.environ['DUP_DECODE'] if os.environ.get('DUP_DECODE') else ''}.json")
out.write_text(json.dumps({"rules": res, "pairs": pairs, "images": len(raw)}, indent=1))
print("saved", out)
