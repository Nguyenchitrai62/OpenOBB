import numpy as np
import pytest

from vrdet.eval.dota import (DOTA1_CLASSES, evaluate, merge_patches, parse_patch_name, read_task1,
                             voc_ap, write_task1)
from vrdet.ops.obb import nms_poly, norm_obb, obb2poly, poly2obb, poly_iou

rng = np.random.default_rng(0)


def rand_obb(n, size=1000):
    w = rng.uniform(10, 120, n)
    h = w * rng.uniform(0.2, 0.8, n)
    return np.stack([rng.uniform(100, size - 100, n), rng.uniform(100, size - 100, n), w, h,
                     rng.uniform(-np.pi / 2, np.pi / 2, n)], 1)


def test_obb_poly_roundtrip():
    obb = norm_obb(rand_obb(200))
    back = poly2obb(obb2poly(obb))
    assert np.allclose(back[:, :4], obb[:, :4], atol=1e-2)
    d = np.abs(back[:, 4] - obb[:, 4])
    assert np.all(np.minimum(d, np.pi - d) < 1e-3)


def test_norm_obb_equivalence():
    a = np.array([[50, 60, 10, 30, 0.3]])
    b = norm_obb(a)
    assert b[0, 2] >= b[0, 3]
    assert -np.pi / 2 <= b[0, 4] < np.pi / 2
    assert poly_iou(obb2poly(a), obb2poly(b))[0, 0] > 0.999


def test_poly_iou_known():
    sq = lambda x, y, s: [x, y, x + s, y, x + s, y + s, x, y + s]
    iou = poly_iou([sq(0, 0, 10)], [sq(5, 0, 10), sq(0, 0, 10), sq(50, 50, 10)])
    assert np.allclose(iou, [[1 / 3, 1.0, 0.0]])
    r = obb2poly([[0, 0, 10, 10, np.pi / 4]])
    assert poly_iou(r, r)[0, 0] == pytest.approx(1.0)
    # 45-degree square vs axis square of same centre/size: inter = octagon
    a = obb2poly([[0, 0, 2, 2, 0.0]])
    inter = 8 * (np.sqrt(2) - 1)          # regular octagon area for these two unit-apothem squares
    assert poly_iou(a, r * 0.2)[0, 0] == pytest.approx(inter / (8 - inter), rel=1e-6)


def test_nms_poly():
    p = obb2poly([[100, 100, 50, 20, 0.1], [101, 100, 50, 20, 0.12], [300, 300, 50, 20, 0.0]])
    keep = nms_poly(p, [0.5, 0.9, 0.7], 0.5)
    assert list(keep) == [1, 2]


def _gt_from_obb(obb, cls_idx, diff=None):
    poly = obb2poly(obb)
    diff = np.zeros(len(obb), int) if diff is None else diff
    return [(list(p), DOTA1_CLASSES[c], int(d)) for p, c, d in zip(poly, cls_idx, diff)]


def test_perfect_predictions_map_is_one(tmp_path):
    gts, dets = {}, {c: ([], [], []) for c in DOTA1_CLASSES}
    for k in range(6):
        obb = rand_obb(30)
        cls = rng.integers(0, len(DOTA1_CLASSES), 30)
        gts[f"P{k:04d}"] = _gt_from_obb(obb, cls)
        for p, c in zip(obb2poly(obb), cls):
            ids, sc, pl = dets[DOTA1_CLASSES[c]]
            ids.append(f"P{k:04d}"); sc.append(rng.uniform(0.1, 1)); pl.append(p)
    dets = {c: (i, np.array(s), np.array(p).reshape(-1, 8)) for c, (i, s, p) in dets.items()}
    write_task1(dets, tmp_path)                     # roundtrip through the submission format
    res = evaluate(read_task1(tmp_path), gts)
    assert res["mAP50"] == pytest.approx(1.0)
    assert res["mAP50_95"] == pytest.approx(1.0, abs=1e-3)   # coords written with 2 decimals


def test_difficult_ignored_and_fp_order():
    obb = np.array([[100, 100, 40, 20, 0.0], [300, 300, 40, 20, 0.5], [600, 600, 40, 20, 1.0]])
    gts = {"A": _gt_from_obb(obb, [0, 0, 0], diff=[0, 0, 1])}
    p = obb2poly(obb)
    fp = obb2poly([[800, 800, 40, 20, 0.0]])
    # highest score on the difficult object (ignored), then an FP, then two TPs
    dets = {"plane": (["A"] * 4, np.array([0.99, 0.95, 0.9, 0.8]), np.concatenate([p[2:3], fp, p[:2]]))}
    r = evaluate(dets, gts)["classes"]["plane"]
    assert r["npos"] == 2
    # flags after dropping the ignored one: FP, TP, TP -> prec [0, .5, .667], rec [0, .5, 1]
    assert r["AP50"] == pytest.approx((6 * 2 / 3 + 5 * 2 / 3) / 11)
    assert r["recall50"] == pytest.approx(1.0)


def test_duplicate_is_fp():
    obb = np.array([[100, 100, 40, 20, 0.0]])
    gts = {"A": _gt_from_obb(obb, [0])}
    p = obb2poly(obb)
    dets = {"plane": (["A", "A"], np.array([0.9, 0.8]), np.concatenate([p, p]))}
    assert evaluate(dets, gts)["classes"]["plane"]["AP50"] == pytest.approx(1.0)
    off = obb2poly([[400, 400, 40, 20, 0.0]])
    dets = {"plane": (["A", "A"], np.array([0.9, 0.8]), np.concatenate([off, p]))}
    assert evaluate(dets, gts)["classes"]["plane"]["AP50"] == pytest.approx(0.5)


def test_voc07_reference():
    rec = np.array([0.1, 0.25, 0.25, 0.45, 0.65])
    prec = np.array([1.0, 1.0, 0.67, 0.8, 0.75])
    # t=0,0.1,0.2 -> 1; 0.3,0.4 -> 0.8; 0.5,0.6 -> 0.75; 0.7..1.0 -> 0
    assert voc_ap(rec, prec, True) == pytest.approx((3 * 1 + 2 * 0.8 + 2 * 0.75) / 11)
    # devkit quirk kept on purpose: np.arange(0, 1.1, 0.1)[6] = 0.6000000000000001 > 0.6
    assert voc_ap(np.array([0.6]), np.array([1.0]), True) == pytest.approx(6 / 11)


def test_parse_patch_name():
    assert parse_patch_name("P0003__1.0__824___0") == ("P0003", 1.0, 824, 0)
    assert parse_patch_name("P1_x__0.5__0___1648") == ("P1_x", 0.5, 0, 1648)
    assert parse_patch_name("plain") == ("plain", 1.0, 0, 0)


def test_merge_patches_recovers_boxes():
    W = H = 2000
    obb = rand_obb(120, size=W)
    obb[:, 2:4] = np.clip(obb[:, 2:4], 5, 60)
    iou = poly_iou(obb2poly(obb), obb2poly(obb))
    keep = [k for k in range(len(obb)) if not any(iou[k, j] > 0 for j in range(k))]
    obb = obb[keep]                     # well separated objects: NMS at 0.1 must not merge them
    poly = obb2poly(obb)
    size, step = 1024, 824
    starts = [0, 824, W - size]
    patch_dets = []
    for x0 in starts:
        for y0 in starts:
            inside = ((poly[:, 0::2].min(1) >= x0) & (poly[:, 0::2].max(1) <= x0 + size) &
                      (poly[:, 1::2].min(1) >= y0) & (poly[:, 1::2].max(1) <= y0 + size))
            for k in np.nonzero(inside)[0]:
                q = poly[k].copy(); q[0::2] -= x0; q[1::2] -= y0
                q += rng.normal(0, 0.3, 8)        # small jitter between patches
                patch_dets.append((f"IMG__1.0__{x0}___{y0}", 0, 0.5 + 0.4 * rng.random(), q))
    merged = merge_patches(patch_dets, iou_thr=0.1)
    ids, sc, pl = merged[0]
    seen = poly_iou(poly, pl).max(1) > 0.8
    covered = np.array([any(((poly[k, 0::2].min() >= x0) & (poly[k, 0::2].max() <= x0 + size) &
                             (poly[k, 1::2].min() >= y0) & (poly[k, 1::2].max() <= y0 + size))
                            for x0 in starts for y0 in starts) for k in range(len(poly))])
    assert seen[covered].all()
    # one detection per object: duplicates from overlapping patches are removed
    assert len(pl) == covered.sum()
