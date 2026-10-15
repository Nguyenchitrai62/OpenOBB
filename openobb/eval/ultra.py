"""Second metric with the conventions of the Ultralytics OBB validator, so OpenOBB numbers can be compared with the
mAP that YOLO prints (OpenOBB's own code, written from the validator's documented behaviour; no Ultralytics code).

Differences from the DOTA devkit protocol in openobb.eval.dota:
  * IoU = ProbIoU between oriented boxes (min-area rectangles of the polygons), not polygon IoU. ProbIoU is more
    lenient on thin objects (3 px wall, 1 px across error: polygon IoU 0.50, ProbIoU ~0.61);
  * matching per image and IoU threshold: each detection keeps its best-IoU GT of the same class, then each GT keeps
    the highest-scored of those detections (not greedy in score order against the best-IoU GT);
  * AP = 101-point interpolation of the monotone precision envelope (precision starts at 1), not VOC07 11-point;
  * detections: score >= 0.001, at most 300 per image (validator defaults).
"""
import numpy as np

from openobb.data.dota import polys_to_obb


def probiou(a, b, eps=1e-7):
    """(N, 5) x (M, 5) oriented boxes (cx, cy, w, h, theta) -> (N, M) ProbIoU (1 - Hellinger distance)."""
    def cov(x):
        va, vb = x[:, 2] ** 2 / 12, x[:, 3] ** 2 / 12
        c, s = np.cos(x[:, 4]), np.sin(x[:, 4])
        return va * c * c + vb * s * s, va * s * s + vb * c * c, (va - vb) * c * s
    a1, b1, c1 = (v[:, None] for v in cov(a))
    a2, b2, c2 = (v[None, :] for v in cov(b))
    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    den = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
    t1 = ((a1 + a2) * dy ** 2 + (b1 + b2) * dx ** 2) / (den + eps) * 0.25
    t2 = ((c1 + c2) * (-dx) * dy) / (den + eps) * 0.5
    t3 = 0.5 * np.log(den / (4 * np.sqrt(np.clip(a1 * b1 - c1 ** 2, 0, None) * np.clip(a2 * b2 - c2 ** 2, 0, None))
                             + eps) + eps)
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    return 1.0 - np.sqrt(1.0 - np.exp(-bd) + eps)


def _match(iou, thr):
    """iou (n_gt, n_det), detections sorted by score desc -> bool (n_det,) true positives at this threshold."""
    tp = np.zeros(iou.shape[1], dtype=bool)
    g, d = np.nonzero(iou >= thr)
    if not len(g):
        return tp
    v = iou[g, d]
    o = np.argsort(-v, kind="stable")
    g, d = g[o], d[o]
    _, first = np.unique(d, return_index=True)          # each detection: its best-IoU GT
    g, d = g[first], d[first]                            # now ordered by detection index (= score order)
    _, first = np.unique(g, return_index=True)          # each GT: the highest-scored detection left
    tp[d[first]] = True
    return tp


def _ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2))


def evaluate_ultra(dets, gts, classes, conf=0.001, max_det=300, iou_thrs=None):
    """dets {class_name: (image_ids, scores, polys)}, gts {image_id: [(poly, cls, difficult)]} (as openobb.eval.dota)
    -> {"mAP50", "mAP75", "mAP50_95", "P", "R", "classes": {name: {"AP50", "AP75", "AP50_95", "P", "R"}}}.
    P / R are taken at the confidence of the best mean F1 (IoU 0.5). Difficult objects count as normal GT."""
    thrs = np.round(np.arange(0.5, 0.96, 0.05), 2) if iou_thrs is None else np.asarray(iou_thrs)
    per_img = {}
    for ci, c in enumerate(classes):
        ids, sc, pl = dets.get(c, ([], np.zeros(0), np.zeros((0, 8))))
        sc, pl = np.asarray(sc, np.float64), np.asarray(pl, np.float64).reshape(-1, 8)
        for i, s, p in zip(ids, sc, pl):
            if s >= conf and np.isfinite(s) and np.isfinite(p).all():
                per_img.setdefault(str(i), []).append((s, ci, p))
    tp_all, sc_all, cl_all = [], [], []
    npos = np.zeros(len(classes), int)
    for img, objs in gts.items():
        gcls = np.array([classes.index(o[1]) for o in objs if o[1] in classes], int)
        gpl = np.array([o[0] for o in objs if o[1] in classes], np.float64).reshape(-1, 8)
        np.add.at(npos, gcls, 1)
        d = sorted(per_img.get(img, []), key=lambda t: -t[0])[:max_det]
        if not d:
            continue
        ds = np.array([t[0] for t in d])
        dc = np.array([t[1] for t in d], int)
        dp = np.stack([t[2] for t in d])
        tp = np.zeros((len(d), len(thrs)), bool)
        if len(gcls):
            iou = probiou(polys_to_obb(gpl.astype(np.float32)).astype(np.float64),
                          polys_to_obb(dp.astype(np.float32)).astype(np.float64))
            iou = iou * (gcls[:, None] == dc[None, :])
            for t, thr in enumerate(thrs):
                tp[:, t] = _match(iou, thr)
        tp_all.append(tp)
        sc_all.append(ds)
        cl_all.append(dc)
    res = {"classes": {}}
    grid, pr_p, pr_r = np.linspace(0, 1, 1000), [], []
    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, len(thrs)), bool)
    sc = np.concatenate(sc_all) if sc_all else np.zeros(0)
    cl = np.concatenate(cl_all) if cl_all else np.zeros(0, int)
    for ci, c in enumerate(classes):
        if not npos[ci]:
            continue
        m = cl == ci
        o = np.argsort(-sc[m], kind="stable")
        t = tp[m][o]
        tpc = np.cumsum(t, 0)
        fpc = np.cumsum(~t, 0)
        aps = [_ap(tpc[:, k] / (npos[ci] + 1e-16), tpc[:, k] / np.maximum(tpc[:, k] + fpc[:, k], 1e-16))
               if len(t) else 0.0 for k in range(len(thrs))]
        k75 = int(np.argmin(np.abs(thrs - 0.75)))
        res["classes"][c] = {"AP50": aps[0], "AP75": aps[k75], "AP50_95": float(np.mean(aps))}
        if len(t):                                  # P / R curves over confidence at IoU 0.5 (as Ultralytics)
            conf_c = sc[m][o]
            pr_r.append(np.interp(-grid, -conf_c, tpc[:, 0] / (npos[ci] + 1e-16), left=0))
            pr_p.append(np.interp(-grid, -conf_c, tpc[:, 0] / np.maximum(tpc[:, 0] + fpc[:, 0], 1e-16), left=1))
        else:
            pr_r.append(np.zeros_like(grid))
            pr_p.append(np.zeros_like(grid))
    if pr_p:                                        # P and R at the confidence of the best mean F1
        p_, r_ = np.stack(pr_p), np.stack(pr_r)
        f1 = 2 * p_ * r_ / (p_ + r_ + 1e-16)
        i = int(_smooth(f1.mean(0), 0.1).argmax())
        for c, pc, rc in zip(res["classes"], p_[:, i], r_[:, i]):
            res["classes"][c]["P"], res["classes"][c]["R"] = float(pc), float(rc)
        res["conf_best_f1"] = float(grid[i])
    res["confusion"] = confusion(per_img, gts, classes)
    res["geometry"] = geometry(per_img, gts, classes)
    vals = list(res["classes"].values())
    res["mAP50"] = float(np.mean([v["AP50"] for v in vals])) if vals else 0.0
    res["mAP75"] = float(np.mean([v["AP75"] for v in vals])) if vals else 0.0
    res["mAP50_95"] = float(np.mean([v["AP50_95"] for v in vals])) if vals else 0.0
    res["P"] = float(np.mean([v.get("P", 0.0) for v in vals])) if vals else 0.0
    res["R"] = float(np.mean([v.get("R", 0.0) for v in vals])) if vals else 0.0
    return res


def _smooth(y, f=0.05):
    """Box filter of fraction f (the F1-curve smoothing of the Ultralytics metrics)."""
    nf = round(len(y) * f * 2) // 2 + 1
    pad = np.ones(nf // 2)
    yp = np.concatenate((pad * y[0], y, pad * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def confusion(per_img, gts, classes, conf=0.25, iou_thr=0.5):
    """Class-agnostic matching at ProbIoU >= iou_thr of detections with score >= conf (YOLO's confusion-matrix
    defaults). Rows = true class (+ background), columns = predicted class (+ background)."""
    n = len(classes)
    cm = np.zeros((n + 1, n + 1), int)
    dup = np.zeros(n, int)                  # same class, overlapping an object another box already detected
    for img, objs in gts.items():
        gc = np.array([classes.index(o[1]) for o in objs if o[1] in classes], int)
        gp = np.array([o[0] for o in objs if o[1] in classes], np.float64).reshape(-1, 8)
        d = sorted([t for t in per_img.get(img, []) if t[0] >= conf], key=lambda t: -t[0])
        dc = np.array([t[1] for t in d], int)
        taken = np.zeros(len(gc), bool)
        if len(d) and len(gc):
            iou = probiou(polys_to_obb(np.stack([t[2] for t in d]).astype(np.float32)).astype(np.float64),
                          polys_to_obb(gp.astype(np.float32)).astype(np.float64))
            for i in range(len(d)):
                j = int(np.argmax(np.where(taken, -1.0, iou[i])))
                if iou[i, j] >= iou_thr and not taken[j]:
                    taken[j] = True
                    cm[gc[j], dc[i]] += 1
                else:
                    cm[n, dc[i]] += 1
                    same = taken & (gc == dc[i])
                    if same.any() and iou[i][same].max() >= iou_thr:
                        dup[dc[i]] += 1
        elif len(d):
            np.add.at(cm[n], dc, 1)
        np.add.at(cm[:, n], gc[~taken], 1)
    return {"classes": list(classes) + ["background"], "matrix": cm.tolist(),
            "duplicates": {c: int(v) for c, v in zip(classes, dup)}, "detections": int(cm[:, :n].sum())}


def _canon(b):
    b = np.asarray(b, np.float64).reshape(-1, 5).copy()
    sw = b[:, 2] < b[:, 3]
    b[sw, 2], b[sw, 3] = b[sw, 3], b[sw, 2].copy()
    b[sw, 4] += np.pi / 2
    return b


def _axis_dev(th):
    """degrees to the nearest multiple of 90 deg"""
    return np.abs(np.rad2deg((th + np.pi / 4) % (np.pi / 2) - np.pi / 4))


def geometry(per_img, gts, classes, conf=0.25, iou_thr=0.5, long_ratio=10.0):
    """Orientation / position quality that mAP hides (a long thin box tilted by 2 deg keeps IoU > 0.5 but leaves its
    line): every label takes its best same-class detection (score >= conf, ProbIoU >= iou_thr = found), then
      angle: |angle error| (modulo 90 deg for near-square labels, whose long axis is arbitrary);
      end:   largest distance of the detection's two end centres from the label's centre line, in units of the
             label's thickness (> 0.5 = an end lies outside the drawn object);
    over all found labels and over long ones (length / thickness >= long_ratio). Label-free: share of long
    detections more than 1 deg off the nearest axis, next to the same share of the long labels (drawings are mostly
    axis-aligned; when the labels are, a tilted detection is wrong whatever its IoU)."""
    ang, end, lng = [], [], []
    det_dev, lab_dev = [], []
    for img, objs in gts.items():
        objs = [o for o in objs if o[1] in classes]
        gb = _canon(polys_to_obb(np.array([o[0] for o in objs], np.float32).reshape(-1, 8)))
        gc = np.array([classes.index(o[1]) for o in objs], int)
        d = [t for t in per_img.get(img, []) if t[0] >= conf]
        pb = _canon(polys_to_obb(np.stack([t[2] for t in d]).astype(np.float32))) if d else np.zeros((0, 5))
        pc = np.array([t[1] for t in d], int)
        lab_dev += list(_axis_dev(gb[gb[:, 2] >= long_ratio * np.maximum(gb[:, 3], 1e-6), 4]))
        det_dev += list(_axis_dev(pb[pb[:, 2] >= long_ratio * np.maximum(pb[:, 3], 1e-6), 4]))
        if not len(gb) or not len(pb):
            continue
        iou = probiou(gb, pb) * (gc[:, None] == pc[None, :])
        for i in range(len(gb)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] < iou_thr:
                continue
            g, p = gb[i], pb[j]
            best = None
            for q in ([p] + ([np.array([p[0], p[1], p[3], p[2], p[4] + np.pi / 2])] if g[2] < 1.5 * g[3] else [])):
                dth = abs(np.rad2deg((q[4] - g[4] + np.pi / 2) % np.pi - np.pi / 2))
                u, v = np.array([np.cos(q[4]), np.sin(q[4])]), np.array([-np.sin(g[4]), np.cos(g[4])])
                off = max(abs((q[:2] + s * u * q[2] / 2 - g[:2]) @ v) for s in (-1, 1)) / max(g[3], 1e-6)
                if best is None or dth < best[0]:
                    best = (dth, off)
            ang.append(best[0]), end.append(best[1]), lng.append(g[2] >= long_ratio * g[3])
    ang, end, lng = np.array(ang), np.array(end), np.array(lng, bool)
    share = lambda x: float(np.mean(x)) if len(x) else 0.0  # noqa: E731
    det_dev, lab_dev = np.array(det_dev), np.array(lab_dev)
    return {"found": int(len(ang)), "angle_gt1": share(ang > 1), "angle_gt2": share(ang > 2), "end_out": share(end > 0.5),
            "long": int(lng.sum()), "long_angle_gt1": share(ang[lng] > 1), "long_end_out": share(end[lng] > 0.5),
            "long_dets": int(len(det_dev)), "long_dets_tilted": share(det_dev > 1), "long_labels_tilted": share(lab_dev > 1)}


def geometry_line(g):
    """'Box geometry ...' summary of geometry()."""
    if not g or not g.get("found"):
        return ""
    return (f"Box geometry (conf 0.25, {g['found']} found objects): angle error > 1 deg {100 * g['angle_gt1']:.1f}%, "
            f"> 2 deg {100 * g['angle_gt2']:.1f}%, an end off the object (> 0.5 thickness) {100 * g['end_out']:.1f}% | "
            f"long objects (length >= 10 x thickness, {g['long']}): angle > 1 deg {100 * g['long_angle_gt1']:.1f}%, "
            f"end off {100 * g['long_end_out']:.1f}% | long detections > 1 deg off the axes {100 * g['long_dets_tilted']:.1f}% "
            f"of {g['long_dets']} (labels {100 * g['long_labels_tilted']:.1f}%)")


def duplicate_line(cm):
    """'Duplicate boxes ...' summary of confusion() (TIDE's duplicate error at conf 0.25, ProbIoU 0.5)."""
    d = cm.get("duplicates")
    if d is None:
        return ""
    tot, det = sum(d.values()), max(cm.get("detections", 0), 1)
    top = ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v)
    return f"Duplicate boxes (same class on an already detected object, conf 0.25): {tot} ({100 * tot / det:.1f}% of "            f"detections){': ' + top if top else ''}"


def top_confusions(cm, k=6):
    """-> lines 'true -> predicted: n (share of that true class)' for the largest off-diagonal cells."""
    m = np.asarray(cm["matrix"])
    names = cm["classes"]
    n = len(names) - 1
    rows = m[:n].sum(1)
    cells = [(m[i, j], i, j) for i in range(n) for j in range(n + 1) if i != j and m[i, j] > 0]
    cells.sort(reverse=True)
    return [f"{names[i]} -> {names[j]}: {v} ({100 * v / max(rows[i], 1):.0f}% of {names[i]})" for v, i, j in cells[:k]]
