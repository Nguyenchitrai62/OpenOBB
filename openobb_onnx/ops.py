"""Box geometry and NMS in numpy (no PyTorch): the post-processing that follows the ONNX graph.

Boxes are (cx, cy, w, h, theta) in pixels, theta in radians from +x towards +y (image coordinates); polygons are
(N, 8) x1, y1, ..., x4, y4. fast_nms gives the result of the network's class-wise Fast-NMS on ProbIoU, nms_poly the
greedy polygon NMS that merges detections (same rules as the PyTorch package).
"""
import math

import numpy as np

NEAR_MARGIN = 1.5    # pairs farther apart than 1.5 x (sum of half-diagonals) cannot overlap: never scored
EPS = 1e-7


def obb2poly(obb):
    obb = np.asarray(obb, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, t = obb.T
    c, s = np.cos(t), np.sin(t)
    wx, wy = w / 2 * c, w / 2 * s
    hx, hy = -h / 2 * s, h / 2 * c
    return np.stack([cx - wx - hx, cy - wy - hy, cx + wx - hx, cy + wy - hy,
                     cx + wx + hx, cy + wy + hy, cx - wx + hx, cy - wy + hy], axis=1)


def _gauss(b):
    w2 = np.maximum(b[..., 2], 1e-4) ** 2 / 12
    h2 = np.maximum(b[..., 3], 1e-4) ** 2 / 12
    c, s = np.cos(b[..., 4]), np.sin(b[..., 4])
    return b[..., 0], b[..., 1], w2 * c * c + h2 * s * s, w2 * s * s + h2 * c * c, (w2 - h2) * c * s


def probiou(b1, b2):
    """ProbIoU (1 - Hellinger distance of the boxes' Gaussians), elementwise over broadcast shapes."""
    x1, y1, a1, bb1, c1 = _gauss(b1)
    x2, y2, a2, bb2, c2 = _gauss(b2)
    a, b, c = a1 + a2, bb1 + bb2, c1 + c2
    den = np.maximum(a * b - c * c, 1e-30)
    t1 = (a * (y1 - y2) ** 2 + b * (x1 - x2) ** 2) / den * 0.25
    t2 = (c * (x2 - x1) * (y1 - y2)) / den * 0.5
    det1 = np.maximum(a1 * bb1 - c1 * c1, 1e-30)
    det2 = np.maximum(a2 * bb2 - c2 * c2, 1e-30)
    t3 = 0.5 * (np.log(den) - math.log(4.0) - 0.5 * (np.log(det1) + np.log(det2)))
    bd = np.clip(t1 + t2 + t3, EPS, 100.0)
    return 1 - np.sqrt(1.0 - np.exp(-bd) + EPS)


def nms_keep(box, lab, iou):
    """Fast-NMS keep mask of boxes (N, 5) in descending ranking order, labels (N,): a box is dropped when a
    higher-ranked box of its class overlaps it with ProbIoU > iou (only near pairs are scored)."""
    n = len(box)
    worst = np.zeros(n, np.float32)
    r = 0.5 * np.sqrt(box[:, 2] ** 2 + box[:, 3] ** 2)
    for c in np.unique(lab):
        ids = np.nonzero(lab == c)[0]
        if len(ids) < 2:
            continue
        p, rr = box[ids, :2], r[ids]
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        near = np.triu(d <= NEAR_MARGIN * (rr[:, None] + rr[None, :]), 1)      # j (higher rank) < i
        j, i = np.nonzero(near)
        if len(i):
            np.maximum.at(worst, ids[i], probiou(box[ids[j]], box[ids[i]]).astype(np.float32))
    return worst <= iou


def fast_nms(rank, score, label, box, conf=0.001, iou=0.7, max_det=1000):
    """Graph outputs of one image (pairs in descending ranking order) -> (score, label, box) of the kept pairs,
    in ranking order, at most max_det. conf applies to the output score; iou=None: no NMS."""
    keep = score > conf
    score, label, box = score[keep], label[keep], box[keep]
    if iou is not None and len(score):
        keep = nms_keep(box.astype(np.float32), label, float(iou))
        score, label, box = score[keep], label[keep], box[keep]
    return score[:max_det], label[:max_det], box[:max_det]


# ---------------------------------------------------------------- polygon NMS (detection merge), shapely

def _geoms(poly):
    import shapely
    p = np.nan_to_num(np.asarray(poly, dtype=np.float64).reshape(-1, 4, 2), nan=0.0, posinf=0.0, neginf=0.0)
    g = shapely.polygons(p)
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.convex_hull(g[bad])
    return g


def _hbb(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 8)
    return np.stack([p[:, 0::2].min(1), p[:, 1::2].min(1), p[:, 0::2].max(1), p[:, 1::2].max(1)], 1)


def _pair_iou(g, ia, ib):
    import shapely
    if len(ia) == 0:
        return np.zeros(0)
    area = shapely.area(g)
    try:
        inter = shapely.area(shapely.intersection(g[ia], g[ib]))
    except shapely.errors.GEOSException:
        inter = np.zeros(len(ia))
        for k, (i, j) in enumerate(zip(ia, ib)):
            try:
                inter[k] = g[i].intersection(g[j]).area
            except shapely.errors.GEOSException:
                inter[k] = g[i].buffer(0).intersection(g[j].buffer(0)).area
    union = area[ia] + area[ib] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def nms_poly(poly, scores, thr):
    """Greedy polygon NMS -> kept indices in descending score order."""
    import shapely
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 8)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(poly)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    g = _geoms(poly)
    h = _hbb(poly)
    tree = shapely.STRtree(shapely.box(h[:, 0], h[:, 1], h[:, 2], h[:, 3]))
    qi, qj = tree.query(shapely.box(h[:, 0], h[:, 1], h[:, 2], h[:, 3]), predicate="intersects")
    m = qi < qj
    qi, qj = qi[m], qj[m]
    m = _pair_iou(g, qi, qj) > thr
    qi, qj = qi[m], qj[m]
    src, dst = np.concatenate([qi, qj]), np.concatenate([qj, qi])
    srt = np.argsort(src, kind="stable")
    src, dst = src[srt], dst[srt]
    start = np.searchsorted(src, np.arange(n + 1))
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[dst[start[i]:start[i + 1]]] = True
    return np.asarray(keep, dtype=np.int64)


def merge(dets, iou):
    """[(cls, score, poly8)] -> same, after class-wise polygon NMS (iou=None: none), descending score."""
    out = []
    for c in sorted({d[0] for d in dets}):
        items = [d for d in dets if d[0] == c and np.isfinite(d[2]).all() and np.isfinite(d[1])]
        if not items:
            continue
        sc = np.array([d[1] for d in items])
        pl = np.stack([d[2] for d in items])
        keep = nms_poly(pl, sc, iou) if iou is not None else np.argsort(-sc, kind="stable")
        out += [(c, float(sc[k]), pl[k]) for k in keep]
    return sorted(out, key=lambda d: -d[1])


# ---------------------------------------------------------------- output boxes

def polys_to_obb(polys):
    """(N, 8) pixel polygons -> (N, 5) boxes (w >= h, theta in [-pi/2, pi/2)) via min-area rectangles."""
    import cv2
    out = np.zeros((len(polys), 5), dtype=np.float32)
    for i, p in enumerate(polys):
        (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(p).reshape(4, 2).astype(np.float32))
        t = math.radians(a)
        if w < h:
            w, h, t = h, w, t + math.pi / 2
        t = (t + math.pi / 2) % math.pi - math.pi / 2
        out[i] = cx, cy, w, h, t
    return out


def regularize_xywhr(b):
    """Rotation in [0, pi/2), w and h swapped when needed (the Ultralytics OBB output convention)."""
    t = b[:, 4] % math.pi
    swap = t >= math.pi / 2
    w = np.where(swap, b[:, 3], b[:, 2])
    h = np.where(swap, b[:, 2], b[:, 3])
    return np.stack([b[:, 0], b[:, 1], w, h, t % (math.pi / 2)], -1)


def xywhr_to_corners(b):
    """(N, 5) -> (N, 4, 2) corners in the Ultralytics corner order."""
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    ux, uy = b[:, 2] / 2 * c, b[:, 2] / 2 * s
    vx, vy = -b[:, 3] / 2 * s, b[:, 3] / 2 * c
    cx, cy = b[:, 0], b[:, 1]
    pts = [(cx + ux + vx, cy + uy + vy), (cx + ux - vx, cy + uy - vy),
           (cx - ux - vx, cy - uy - vy), (cx - ux + vx, cy - uy + vy)]
    return np.stack([np.stack(p, -1) for p in pts], 1)
