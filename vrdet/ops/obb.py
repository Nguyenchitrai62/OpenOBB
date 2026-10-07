"""Oriented-box geometry on the CPU (numpy + shapely).

Polygons are float arrays of shape (N, 8): x1, y1, ..., x4, y4 in pixels.
OBB convention ("le90"): (cx, cy, w, h, theta), w >= h, theta in [-pi/2, pi/2) radians,
theta measured from the +x axis towards +y (image coordinates, y pointing down).
"""
import numpy as np
import shapely

EPS = 1e-9


def norm_obb(obb):
    """Bring (cx, cy, w, h, theta) to the le90 convention: w >= h, theta in [-pi/2, pi/2)."""
    obb = np.array(obb, dtype=np.float64).reshape(-1, 5)
    swap = obb[:, 2] < obb[:, 3]
    obb[swap, 2], obb[swap, 3] = obb[swap, 3], obb[swap, 2]
    obb[swap, 4] += np.pi / 2
    obb[:, 4] = (obb[:, 4] + np.pi / 2) % np.pi - np.pi / 2
    return obb


def obb2poly(obb):
    obb = np.asarray(obb, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, t = obb.T
    c, s = np.cos(t), np.sin(t)
    wx, wy = w / 2 * c, w / 2 * s          # half extent along the width axis
    hx, hy = -h / 2 * s, h / 2 * c         # half extent along the height axis
    return np.stack([cx - wx - hx, cy - wy - hy,
                     cx + wx - hx, cy + wy - hy,
                     cx + wx + hx, cy + wy + hy,
                     cx - wx + hx, cy - wy + hy], axis=1)


def poly2obb(poly):
    """Minimum-area rectangle of each quadrilateral, in le90."""
    import cv2
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 4, 2)
    out = np.zeros((len(poly), 5))
    for i, p in enumerate(poly):
        (cx, cy), (w, h), a = cv2.minAreaRect(p)
        out[i] = cx, cy, w, h, np.deg2rad(a)
    return norm_obb(out)


def to_geoms(poly):
    """(N, 8) -> shapely polygons. Self-intersecting quads are replaced by their convex hull."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 4, 2)
    g = shapely.polygons(p)
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.convex_hull(g[bad])
    return g


def hbb(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 8)
    return np.stack([p[:, 0::2].min(1), p[:, 1::2].min(1), p[:, 0::2].max(1), p[:, 1::2].max(1)], 1)


def _pair_iou(ga, gb, ia, ib, area_a, area_b):
    if len(ia) == 0:
        return np.zeros(0)
    try:
        inter = shapely.area(shapely.intersection(ga[ia], gb[ib]))
    except shapely.errors.GEOSException:   # rare topology failure: fall back pair by pair
        inter = np.zeros(len(ia))
        for k, (i, j) in enumerate(zip(ia, ib)):
            try:
                inter[k] = ga[i].intersection(gb[j]).area
            except shapely.errors.GEOSException:
                inter[k] = ga[i].buffer(0).intersection(gb[j].buffer(0)).area
    union = area_a[ia] + area_b[ib] - inter
    return np.where(union > EPS, inter / np.maximum(union, EPS), 0.0)


def poly_iou(a, b):
    """Dense (len(a), len(b)) polygon IoU matrix. Only pairs whose bounding boxes overlap are computed."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 8)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 8)
    out = np.zeros((len(a), len(b)))
    if len(a) == 0 or len(b) == 0:
        return out
    ha, hb_ = hbb(a), hbb(b)
    ov = ((np.minimum(ha[:, None, 2], hb_[None, :, 2]) > np.maximum(ha[:, None, 0], hb_[None, :, 0])) &
          (np.minimum(ha[:, None, 3], hb_[None, :, 3]) > np.maximum(ha[:, None, 1], hb_[None, :, 1])))
    ia, ib = np.nonzero(ov)
    ga, gb = to_geoms(a), to_geoms(b)
    out[ia, ib] = _pair_iou(ga, gb, ia, ib, shapely.area(ga), shapely.area(gb))
    return out


def nms_poly(poly, scores, thr):
    """Greedy polygon NMS. Returns kept indices sorted by descending score.

    Neighbour pairs come from an STR-tree on bounding boxes, so it scales to tens of
    thousands of boxes (a big DOTA image merged from many patches)."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 8)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(poly)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    g = to_geoms(poly)
    h = hbb(poly)
    boxes = shapely.box(h[:, 0], h[:, 1], h[:, 2], h[:, 3])
    tree = shapely.STRtree(boxes)
    qi, qj = tree.query(boxes, predicate="intersects")
    m = qi < qj
    qi, qj = qi[m], qj[m]
    iou = _pair_iou(g, g, qi, qj, shapely.area(g), shapely.area(g))
    m = iou > thr
    qi, qj = qi[m], qj[m]
    # symmetric adjacency in CSR form
    src = np.concatenate([qi, qj])
    dst = np.concatenate([qj, qi])
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
