"""Can the trained model tell a tight box from a loose one WITHOUT new training? (rank_vote_oracle.py: a tighter
candidate sits in the NMS cluster, +0.052 mAP50-95 if picked; a localisation-quality factor on the score: +0.044.)
Two cues the model already computes but never uses to score:
  field   the boundary field (stride 2, learnt from the labelled sides): sigmoid of the field sampled along the 4 sides
          of the FINAL box (ends -> channel 0, long sides -> channel 1, near-square boxes -> max of both), mean over
          N_S points per side, then the mean (and the min) over the 4 sides: a box whose sides sit on boundaries
  dfl     sharpness of the head's three distance distributions (left end, right end, thickness): mean of the top-2
          bin probabilities (GFLv2's cue)
For the one-to-many head + refiner + snapping + NMS as deployed (o2m 0.5): Spearman of each cue with ProbIoU to the
matched label, and mAP50-95 when (a) the score becomes score x cue^beta, (b) the NMS keeps, in each cluster, the box
with the best score x cue^beta (IoU-guided NMS). CPU, Wall_Color val.

usage: python quality_probe.py <checkpoint.pt> [n_images]
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import evaluate_ultra, probiou  # noqa: E402
from openobb.models.openobb3 import ALONG_BINS, REG_CH, fast_nms  # noqa: E402
from openobb.models.openobb17 import FIELD_STRIDE, _sample  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402
from openobb.predict import PAD_BGR, rect_shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color"
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
N_S = 7
NMS_IOU = 0.5
BETAS = (0.5, 1.0, 2.0)
ck, n_img = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
m = OpenOBB(ck)
net = m.model.eval()
CACHE = Path(__file__).with_name(f"quality_{Path(ck).stem}_{min(n_img, 53)}.npz")


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


def field_cue(field, b):
    """field (1, 2, H, W) logits, b (K, 5) px -> (mean over sides, min over sides) of the side-wise mean probability."""
    c, L, T, th = b[:, :2], b[:, 2].abs(), b[:, 3].abs(), b[:, 4]
    ax = torch.stack([th.cos(), th.sin()], -1)
    nx = torch.stack([-th.sin(), th.cos()], -1)
    q = torch.linspace(-0.4, 0.4, N_S)
    sides = []
    for at, along, half, d_ax, run_ax, is_end in ((L / 2, ax, T, ax, nx, True), (-L / 2, ax, T, ax, nx, True),
                                                  (T / 2, nx, L, nx, ax, False), (-T / 2, nx, L, nx, ax, False)):
        pos = c[:, None] + at[:, None, None] * d_ax[:, None] + (q[None] * half[:, None])[..., None] * run_ax[:, None]
        s = _sample(field, pos.reshape(1, -1, 2), FIELD_STRIDE).reshape(len(b), N_S, 2).sigmoid()
        long_box = L >= 1.3 * T
        ch = torch.where(long_box[:, None], s[..., 0] if is_end else s[..., 1], s.amax(-1))
        sides.append(ch.mean(-1))
    sides = torch.stack(sides, -1)
    return sides.mean(-1), sides.amin(-1)


def dfl_cue(reg_g):
    """reg rows (K, REG_CH) of the candidates' group -> mean top-2 probability of the left / right / thickness bins."""
    parts = (reg_g[:, 2:2 + ALONG_BINS], reg_g[:, 2 + ALONG_BINS:2 + 2 * ALONG_BINS], reg_g[:, 2 + 2 * ALONG_BINS:])
    return torch.stack([p.softmax(-1).topk(2, -1).values.sum(-1) for p in parts], -1).mean(-1)


if CACHE.exists():
    data = {k: v.item() for k, v in np.load(CACHE, allow_pickle=True).items()}
else:
    data = {}
    imgs = sorted((ROOT / "images" / "val").iterdir())[:n_img]
    t0 = time.time()
    for k, p in enumerate(imgs):
        img = imread(str(p))
        h0, w0 = img.shape[:2]
        s = 1280 / max(h0, w0)
        img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        H, W = rect_shape(*img.shape[:2], 1280)
        x = np.full((H, W, 3), PAD_BGR, np.uint8)
        x[:img.shape[0], :img.shape[1]] = img
        xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
        with torch.inference_mode():
            f4, feats = net.encode(xt)
            pts, st = net.anchors(feats)
            field = net.field(f4, feats[0])
            cls, reg = net.run_head(feats, False)                                  # one-to-many head
            cc, boxes = net.candidates(cls, reg, pts, st)
            qv = None
            if getattr(net, "quality", None) is not None:          # OpenOBB24: deployed path (snap_skip) + its learnt q
                k_ = min(int(net.ref_k), cc.shape[1])
                idx = cc.sigmoid().amax(-1).topk(k_, dim=1).indices
                cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
                grp = idx // reg.shape[1] if net.n_groups > 1 else torch.zeros_like(idx)
                stages, hid = net.edge_snap(f4, feats[0], field, cand, grp)
                qv = net.quality_logits(hid, field, stages, net.head_rows(reg, idx)).sigmoid()[0]
                boxes = boxes.scatter(1, idx[..., None].expand(-1, -1, 5), stages[-1][1].to(boxes.dtype))
            else:
                boxes, idx = net.snap_topk(field, cc, boxes, net.ref_k, f4, feats[0])
            idx = idx[0]
            A = reg.shape[1]
            G = net.n_groups
            a_i, g_i = idx % A, idx // A
            R = reg.shape[-1] // G
            reg_g = reg[0].view(A, G, R)[a_i, g_i]
            b = boxes[0, idx].float()
            fm, fmin = field_cue(field, b)
            dq = dfl_cue(reg_g.float())
            sc, cl = cc[0, idx].sigmoid().max(-1)
        lab = ROOT / "labels" / "val" / f"{p.stem}.txt"
        g = []
        for line in lab.read_text().splitlines() if lab.exists() else []:
            v = line.split()
            if len(v) >= 9:
                g.append((np.array(v[1:9], float) * np.tile([w0 * s, h0 * s], 4), NAMES[int(v[0])], 0))
        keep = sc > 0.05
        data[p.stem] = {"sc": sc[keep].numpy(), "cl": cl[keep].numpy(), "b": b[keep].numpy(), "fm": fm[keep].numpy(),
                        "fmin": fmin[keep].numpy(), "dq": dq[keep].numpy(), "gt": g,
                        **({} if qv is None else {"q": qv[keep].numpy()})}
        print(f"{k + 1}/{len(imgs)} {time.time() - t0:.0f}s", end="\r", flush=True)
    np.savez(CACHE, **data)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 4 else float("nan")


def nms_keep(score, cl, b, rank):
    """class-wise NMS (ProbIoU >= NMS_IOU) visiting boxes by `rank`, returns kept indices."""
    logits = torch.zeros(1, len(score), len(NAMES)) - 30
    logits[0, torch.arange(len(score)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(rank).clamp(1e-6, 1 - 1e-6)).float()
    tag = torch.arange(len(score), dtype=torch.float32)
    bb = torch.as_tensor(b)[None].float()
    L, B = fast_nms(logits, bb, max_det=3000, conf=0.05, iou=NMS_IOU)
    kept = L[0].sigmoid().amax(-1) > 0.05
    out = []
    for box in B[0][kept]:
        d = (bb[0] - box).abs().sum(-1)
        out.append(int(d.argmin()))
    return np.array(out, int)


CUES = ("fm", "fmin", "dq") + (("q",) if all("q" in d for d in data.values()) else ())
variants = {"base": ("score", None)}
for cue in CUES:
    for beta in BETAS:
        variants[f"{cue}^{beta}"] = ("score", (cue, beta))
        variants[f"nms {cue}^{beta}"] = ("nms", (cue, beta))
dets = {v: {c: ([], [], []) for c in NAMES} for v in variants}
corr = {c: {k: [] for k in ("s",) + CUES + ("iou",)} for c in NAMES}
gts = {}
for stem, d in sorted(data.items()):
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in d["gt"]]
    gb = np.array([gt_box(p) for p, _, _ in gts[stem]]) if gts[stem] else np.zeros((0, 5))
    gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
    sc, cl, b = d["sc"].astype(np.float64), d["cl"].astype(int), d["b"].astype(np.float64)
    for name, (mode, q) in variants.items():
        rank = sc if q is None else sc * np.clip(d[q[0]], 1e-4, 1) ** q[1]
        keep = nms_keep(sc, cl, b, rank)
        out_s = sc[keep] if mode == "nms" or q is None else rank[keep]
        for j, (c, s_) in zip(keep, zip(cl[keep], out_s)):
            ids, scs, pls = dets[name][NAMES[c]]
            ids.append(stem), scs.append(float(s_)), pls.append(obb2poly(canon(b[j])[None])[0])
        if name == "base" and len(keep) and len(gb):
            pb = np.array([canon(x) for x in b[keep]])
            iou = probiou(gb, pb)
            for t, j in enumerate(keep):
                mm = iou[:, t] * (gc == cl[j])
                if mm.max() >= 0.5 and sc[j] >= 0.25:
                    r = corr[NAMES[cl[j]]]
                    r["s"].append(sc[j]), r["iou"].append(mm.max())
                    for k_ in CUES:
                        r[k_].append(d[k_][j])

res = {v: evaluate_ultra({c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in dets[v].items()},
                         gts, NAMES, max_det=3000) for v in variants}
print(f"\n{Path(ck).name}: Spearman with ProbIoU (matched >= 0.5, score >= 0.25)")
print(f"{'class':12s} {'n':>5s} {'score':>7s} {'field':>7s} {'fmin':>7s} {'dfl':>7s}" + (f" {'q':>7s}" if "q" in CUES else ""))
for c in NAMES:
    r = corr[c]
    if len(r["iou"]) < 5:
        continue
    print(f"{c:12s} {len(r['iou']):5d} " + " ".join(f"{spearman(np.array(r[k]), np.array(r['iou'])):+7.3f}"
                                                  for k in ("s",) + CUES))
names = list(variants)
print(f"\nmAP50-95 delta vs base (o2m + refine + snap + NMS {NMS_IOU}); 'nms x' = cue only chooses the kept box")
print(f"{'class':12s} " + " ".join(f"{n:>11s}" for n in names))
for c in [c for c in NAMES if c in res["base"]["classes"]] + ["all"]:
    base = res["base"]["mAP50_95"] if c == "all" else res["base"]["classes"][c]["AP50_95"]
    vals = [(res[n]["mAP50_95"] if c == "all" else res[n]["classes"][c]["AP50_95"]) for n in names]
    print(f"{c:12s} {base:11.3f} " + " ".join(f"{v - base:+11.3f}" for v in vals[1:]))
print(f"{'all mAP50':12s} {res['base']['mAP50']:11.3f} " + " ".join(f"{res[n]['mAP50'] - res['base']['mAP50']:+11.3f}"
                                                                for n in names[1:]))
