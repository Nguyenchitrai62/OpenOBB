"""Second metric with the conventions of the Ultralytics OBB validator, so VRDet numbers can be compared with the
mAP that YOLO prints (VRDet's own code, written from the validator's documented behaviour; no Ultralytics code).

Differences from the DOTA devkit protocol in vrdet.eval.dota:
  * IoU = ProbIoU between oriented boxes (min-area rectangles of the polygons), not polygon IoU. ProbIoU is more
    lenient on thin objects (3 px wall, 1 px across error: polygon IoU 0.50, ProbIoU ~0.61);
  * matching per image and IoU threshold: each detection keeps its best-IoU GT of the same class, then each GT keeps
    the highest-scored of those detections (not greedy in score order against the best-IoU GT);
  * AP = 101-point interpolation of the monotone precision envelope (precision starts at 1), not VOC07 11-point;
  * detections: score >= 0.001, at most 300 per image (validator defaults).
"""
import numpy as np

from vrdet.data.dota import polys_to_obb


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
    """dets {class_name: (image_ids, scores, polys)}, gts {image_id: [(poly, cls, difficult)]} (as vrdet.eval.dota)
    -> {"mAP50", "mAP50_95", "classes": {name: {"AP50", "AP50_95"}}}. Difficult objects count as normal GT."""
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
        res["classes"][c] = {"AP50": aps[0], "AP50_95": float(np.mean(aps))}
    res["confusion"] = confusion(per_img, gts, classes)
    vals = list(res["classes"].values())
    res["mAP50"] = float(np.mean([v["AP50"] for v in vals])) if vals else 0.0
    res["mAP50_95"] = float(np.mean([v["AP50_95"] for v in vals])) if vals else 0.0
    return res


def confusion(per_img, gts, classes, conf=0.25, iou_thr=0.5):
    """Class-agnostic matching at ProbIoU >= iou_thr of detections with score >= conf (YOLO's confusion-matrix
    defaults). Rows = true class (+ background), columns = predicted class (+ background)."""
    n = len(classes)
    cm = np.zeros((n + 1, n + 1), int)
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
        elif len(d):
            np.add.at(cm[n], dc, 1)
        np.add.at(cm[:, n], gc[~taken], 1)
    return {"classes": list(classes) + ["background"], "matrix": cm.tolist()}


def top_confusions(cm, k=6):
    """-> lines 'true -> predicted: n (share of that true class)' for the largest off-diagonal cells."""
    m = np.asarray(cm["matrix"])
    names = cm["classes"]
    n = len(names) - 1
    rows = m[:n].sum(1)
    cells = [(m[i, j], i, j) for i in range(n) for j in range(n + 1) if i != j and m[i, j] > 0]
    cells.sort(reverse=True)
    return [f"{names[i]} -> {names[j]}: {v} ({100 * v / max(rows[i], 1):.0f}% of {names[i]})" for v, i, j in cells[:k]]
