"""Do neighbouring detections know where an object ends? (F64: 56-85% of label ends lie on an edge of the adjacent
object.) No ground truth is used to move boxes: after the model's NMS, every end of every detection is pulled to the
closest crossing of its axis with an edge of another confident detection (conf >= NEIGH_CONF) if that crossing is
within R px of the end; the edge must cross the axis at >= 30 deg. Variants: radius R, and how far to move
(blend 1 = onto the neighbour's edge, 0.5 = half way). Scored with the YOLO-style evaluator; "ends(gt)" is the
ceiling of a perfect end fix on the same detections (box_error_oracle.py).

Diagnostic on matched detections (same class, ProbIoU >= 0.5): for ends that find a neighbour edge within R, the
along-axis error to the label end of the detection's own end vs. of the neighbour's edge.

usage: python shared_end_oracle.py <dup_raw_<model>_53.npz> [nms_iou=0.5]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_num_threads(1)
from openobb.eval.ultra import evaluate_ultra, probiou  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw_path = sys.argv[1]
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
NEIGH_CONF = 0.25
VARIANTS = [("R2", 2.0, 1.0), ("R3", 3.0, 1.0), ("R4", 4.0, 1.0), ("R6", 6.0, 1.0), ("R3 half", 3.0, 0.5),
            ("R4 half", 4.0, 0.5)]
DIAG_R = 4.0


def canon(b):
    cx, cy, w, h, th = b
    return np.array([cx, cy, w, h, th] if w >= h else [cx, cy, h, w, th + np.pi / 2], np.float64)


def gt_box(pts):
    import cv2
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32).reshape(4, 2))
    return canon([cx, cy, w, h, np.deg2rad(a)])


def fix_ends(p, g):
    """box_error_oracle.py 'ends': both extents along the label's axis set to the label's, the rest kept."""
    u = np.array([np.cos(g[4]), np.sin(g[4])])
    v = np.array([-u[1], u[0]])
    dth = (p[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2
    c = g[:2] + v * ((p[:2] - g[:2]) @ v)
    return np.array([c[0], c[1], g[2], p[3], g[4] + dth])


def crossings(b, polys, idx):
    """Axis of canonical box b (centre c, direction u) against the 4 edges of each polygon in polys[idx]:
    -> along-axis positions t (relative to the centre) of the crossings at >= 30 deg inside the edge (2 px slack)."""
    c, u = b[:2], np.array([np.cos(b[4]), np.sin(b[4])])
    if not len(idx):
        return np.zeros(0)
    P = polys[idx]                                              # (n, 4, 2)
    A, Bp = P, np.roll(P, -1, axis=1)
    e = Bp - A                                                  # (n, 4, 2)
    el = np.linalg.norm(e, axis=-1).clip(1e-6)
    cross_ue = u[0] * e[..., 1] - u[1] * e[..., 0]              # |u x e| = |e| sin(angle)
    ok = np.abs(cross_ue) / el >= 0.5
    # c + t u = A + s e  ->  t = ((A - c) x e) / (u x e),  s = ((A - c) x u) / (u x e)
    d = A - c
    den = np.where(ok, cross_ue, 1.0)
    t = (d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]) / den
    s = (d[..., 0] * u[1] - d[..., 1] * u[0]) / den
    slack = 2.0 / el
    ok &= (s >= -slack) & (s <= 1 + slack)
    return t[ok]


def snap(b, ts, R, blend):
    """Move each end of canonical box b to the nearest crossing within R (blend of the way)."""
    L = b[2]
    lo, hi = -L / 2, L / 2
    new = []
    for end in (lo, hi):
        if len(ts):
            dd = np.abs(ts - end)
            j = int(np.argmin(dd))
            new.append(end + blend * (ts[j] - end) if dd[j] <= R else end)
        else:
            new.append(end)
    lo2, hi2 = new
    if hi2 - lo2 < 1.0:
        return b, (lo, hi), (lo, hi)
    u = np.array([np.cos(b[4]), np.sin(b[4])])
    c = b[:2] + u * (lo2 + hi2) / 2
    return np.array([c[0], c[1], hi2 - lo2, b[3], b[4]]), (lo, hi), (lo2, hi2)


raw = dict(np.load(raw_path, allow_pickle=True))
gts = {}
names_v = ["none", "ends(gt)"] + [v[0] for v in VARIANTS] + ["R4 gate(gt)"]
dets = {v: {c: ([], [], []) for c in NAMES} for v in names_v}
diag = {c: [] for c in NAMES}           # (own error, neighbour-edge error) per snappable matched end
n_end, n_snap = {c: 0 for c in NAMES}, {c: 0 for c in NAMES}
for k, (stem, rec) in enumerate(sorted(raw.items())):
    rec = rec.item()
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    gb = np.array([gt_box(p) for p, _, _ in gts[stem]]) if gts[stem] else np.zeros((0, 5))
    gc = np.array([NAMES.index(n) for _, n, _ in gts[stem]], int)
    sc, cl, b = rec["raw"]
    if not len(sc):
        continue
    logits = torch.zeros(1, len(sc), len(NAMES)) - 30
    logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
        torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
    Lg, Bx = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=NMS_IOU)
    s2, c2 = Lg[0].sigmoid().max(-1)
    keep = (s2 > 0.05).numpy()
    s2, c2 = s2.numpy()[keep], c2.numpy()[keep]
    pb = np.array([canon(x) for x in Bx[0].numpy().astype(np.float64)[keep]]).reshape(-1, 5)
    polys = obb2poly(pb).reshape(-1, 4, 2)
    conf_ok = s2 >= NEIGH_CONF
    reach = pb[:, 2] / 2 + 8.0
    iou = probiou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
    for j in range(len(pb)):
        dist = np.linalg.norm(pb[:, :2] - pb[j, :2], axis=1)
        idx = np.nonzero(conf_ok & (dist <= reach + reach[j]) & (np.arange(len(pb)) != j))[0]
        ts = crossings(pb[j], polys, idx)
        c = int(c2[j])
        m = iou[:, j] * (gc == c)
        gi = int(np.argmax(m)) if len(m) else -1
        matched = gi >= 0 and m[gi] >= 0.1
        out = {"none": pb[j]}
        out["ends(gt)"] = fix_ends(pb[j], gb[gi]) if matched else pb[j]
        for name, R, blend in VARIANTS:
            out[name] = snap(pb[j], ts, R, blend)[0]
        # ceiling of a perfect gate: per end, take the neighbour crossing (R 4) only when it is closer to the label end
        out["R4 gate(gt)"] = pb[j]
        if matched and len(ts):
            g = gb[gi]
            ug = np.array([np.cos(g[4]), np.sin(g[4])])
            g_ends = np.array([-g[2] / 2, g[2] / 2]) + (g[:2] @ ug)
            u = np.array([np.cos(pb[j, 4]), np.sin(pb[j, 4])])
            new = []
            for end in (-pb[j, 2] / 2, pb[j, 2] / 2):
                jj = int(np.argmin(np.abs(ts - end)))
                e_own = (pb[j, :2] + u * end) @ ug
                e_nb = (pb[j, :2] + u * ts[jj]) @ ug
                ge = g_ends[np.argmin(np.abs(g_ends - e_own))]
                new.append(ts[jj] if abs(ts[jj] - end) <= 4.0 and abs(e_nb - ge) < abs(e_own - ge) else end)
            if new[1] - new[0] >= 1.0:
                c0 = pb[j, :2] + u * (new[0] + new[1]) / 2
                out["R4 gate(gt)"] = np.array([c0[0], c0[1], new[1] - new[0], pb[j, 3], pb[j, 4]])
        # diagnostic: own end vs neighbour edge, against the label end (along the label's axis)
        if gi >= 0 and m[gi] >= 0.5:
            g = gb[gi]
            ug = np.array([np.cos(g[4]), np.sin(g[4])])
            g_ends = np.array([-g[2] / 2, g[2] / 2]) + (g[:2] @ ug)
            u = np.array([np.cos(pb[j, 4]), np.sin(pb[j, 4])])
            for end in (-pb[j, 2] / 2, pb[j, 2] / 2):
                n_end[NAMES[c]] += 1
                p_own = pb[j, :2] + u * end
                e_own = p_own @ ug
                ge = g_ends[np.argmin(np.abs(g_ends - e_own))]
                if len(ts):
                    jj = int(np.argmin(np.abs(ts - end)))
                    if abs(ts[jj] - end) <= DIAG_R:
                        n_snap[NAMES[c]] += 1
                        e_nb = (pb[j, :2] + u * ts[jj]) @ ug
                        diag[NAMES[c]].append((abs(e_own - ge), abs(e_nb - ge)))
        for v in names_v:
            ids, scs, pls = dets[v][NAMES[c]]
            ids.append(stem), scs.append(float(s2[j])), pls.append(obb2poly(out[v][None])[0])
    print(f"{k + 1}/{len(raw)}", end="\r", flush=True)

res = {}
for v in names_v:
    d = {c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in dets[v].items()}
    res[v] = evaluate_ultra(d, gts, NAMES, max_det=3000)
print(f"\n{Path(raw_path).name}, NMS {NMS_IOU}, neighbours conf >= {NEIGH_CONF}: delta mAP50-95 (no labels used except ends(gt))")
print(f"{'class':12s} {'base':>8s} " + " ".join(f"{v:>9s}" for v in names_v[1:]))
for c in NAMES:
    base = res["none"]["classes"][c]["AP50_95"]
    print(f"{c:12s} {base:8.3f} " + " ".join(f"{res[v]['classes'][c]['AP50_95'] - base:+9.3f}" for v in names_v[1:]))
base = res["none"]["mAP50_95"]
print(f"{'all':12s} {base:8.3f} " + " ".join(f"{res[v]['mAP50_95'] - base:+9.3f}" for v in names_v[1:]))
print(f"{'all mAP50':12s} {res['none']['mAP50']:8.3f} " + " ".join(f"{res[v]['mAP50'] - res['none']['mAP50']:+9.3f}"
                                                          for v in names_v[1:]))
print(f"\nDiagnostic (matched ProbIoU >= 0.5, neighbour edge within {DIAG_R:g} px): along-axis error to the label end, px")
print(f"{'class':12s} {'ends':>6s} {'w/ edge':>8s} {'own med':>8s} {'edge med':>9s} {'edge closer':>12s}")
for c in NAMES:
    a = np.array(diag[c]).reshape(-1, 2)
    if not len(a):
        print(f"{c:12s} {n_end[c]:6d} {0:8d}")
        continue
    print(f"{c:12s} {n_end[c]:6d} {n_snap[c]:8d} {np.median(a[:, 0]):8.2f} {np.median(a[:, 1]):9.2f} "
          f"{(a[:, 1] < a[:, 0]).mean():12.2f}")
