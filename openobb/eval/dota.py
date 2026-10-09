"""DOTA-v1.0 Task1 (oriented boxes) evaluation, written from scratch to follow the
official DOTA devkit protocol (dota_evaluation_task1.py):

* evaluation on ORIGINAL full images, against the original 4-point polygons;
* IoU = polygon IoU between the detected quadrilateral and the GT quadrilateral;
* objects flagged difficult are ignored: they do not count in npos, and a detection
  whose best match is a difficult object is neither TP nor FP;
* greedy matching in descending score; the best-IoU GT already taken -> FP;
* AP50 uses the VOC07 11-point metric (devkit default) -> `mAP50`.

Extra (not part of the official protocol): `mAP50_95` averages all-point AP over IoU
thresholds 0.50:0.05:0.95 with the same matching rules.

Detections use the DOTA submission format: one file per class `Task1_<class>.txt`, lines
`<image_id> <score> x1 y1 x2 y2 x3 y3 x4 y4`. Patch detections are merged back to full
images with `merge_patches` (class-wise polygon NMS, IoU 0.1 like mmrotate's DOTAMetric).

CLI:  python -m openobb.eval.dota --dets DIR_WITH_Task1_FILES --gt LABELTXT_DIR [--out res.json]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from openobb.ops.obb import nms_poly, poly_iou

DOTA1_CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
                 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

PATCH_RE = re.compile(r"^(?P<img>.+)__(?P<rate>[\d.]+)__(?P<x>-?\d+)___(?P<y>-?\d+)$")


# ---------------------------------------------------------------- IO

def parse_dota_txt(path):
    """Original DOTA label file -> list of (poly[8], class_name, difficult)."""
    objs = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue                      # 'imagesource:...', 'gsd:...' headers
        try:
            poly = [float(v) for v in parts[:8]]
        except ValueError:
            continue
        diff = int(parts[9]) if len(parts) > 9 else 0
        objs.append((poly, parts[8], diff))
    return objs


def load_gt_dir(label_dir, image_ids=None):
    """{image_id: list of (poly, cls, difficult)} from a labelTxt directory."""
    gts = {}
    for p in sorted(Path(label_dir).glob("*.txt")):
        if image_ids is None or p.stem in image_ids:
            gts[p.stem] = parse_dota_txt(p)
    return gts


def read_task1(det_dir, classes=DOTA1_CLASSES):
    """{class: (image_ids[list], scores[N], polys[N,8])} from Task1_<class>.txt files."""
    out = {}
    for c in classes:
        p = Path(det_dir) / f"Task1_{c}.txt"
        ids, sc, pl = [], [], []
        if p.exists():
            for line in p.read_text().splitlines():
                parts = line.split()
                if len(parts) < 10:
                    continue
                ids.append(parts[0])
                sc.append(float(parts[1]))
                pl.append([float(v) for v in parts[2:10]])
        out[c] = (ids, np.asarray(sc, dtype=np.float64), np.asarray(pl, dtype=np.float64).reshape(-1, 8))
    return out


def write_task1(dets, det_dir, classes=DOTA1_CLASSES):
    """dets: {class: (image_ids, scores, polys)} -> Task1_<class>.txt files."""
    det_dir = Path(det_dir)
    det_dir.mkdir(parents=True, exist_ok=True)
    for c in classes:
        ids, sc, pl = dets.get(c, ([], np.zeros(0), np.zeros((0, 8))))
        with open(det_dir / f"Task1_{c}.txt", "w") as f:
            for i, s, p in zip(ids, sc, pl):
                f.write(f"{i} {s:.6f} " + " ".join(f"{v:.2f}" for v in p) + "\n")


# ---------------------------------------------------------------- merge

def parse_patch_name(name):
    """'P0003__1.0__824___0' -> ('P0003', 1.0, 824, 0). Plain names map to (name, 1.0, 0, 0)."""
    m = PATCH_RE.match(name)
    if not m:
        return name, 1.0, 0, 0
    return m["img"], float(m["rate"]), int(m["x"]), int(m["y"])


def merge_patches(patch_dets, iou_thr=0.1, max_per_img=None):
    """Merge per-patch detections into full-image detections.

    patch_dets: iterable of (patch_name, cls_index, score, poly[8] in patch pixels).
    Returns {class_index: (image_ids, scores, polys)} in original-image pixels after
    class-wise polygon NMS per image."""
    groups = defaultdict(list)
    for name, c, s, p in patch_dets:
        q = np.asarray(p, dtype=np.float64).copy()
        if not (np.isfinite(q).all() and np.isfinite(s)):     # diverged prediction: never reaches shapely
            continue
        img, rate, x0, y0 = parse_patch_name(name)
        q[0::2] = (q[0::2] + x0) / rate
        q[1::2] = (q[1::2] + y0) / rate
        groups[(img, int(c))].append((float(s), q))
    out = defaultdict(lambda: ([], [], []))
    for (img, c), items in groups.items():
        sc = np.array([s for s, _ in items])
        pl = np.stack([q for _, q in items])
        keep = nms_poly(pl, sc, iou_thr) if iou_thr is not None else np.argsort(-sc)
        if max_per_img:
            keep = keep[:max_per_img]
        ids, ss, pp = out[c]
        ids.extend([img] * len(keep))
        ss.append(sc[keep])
        pp.append(pl[keep])
    return {c: (ids, np.concatenate(ss), np.concatenate(pp)) for c, (ids, ss, pp) in out.items()}


# ---------------------------------------------------------------- metric

def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            ap += (np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.0) / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _match_image(iou, difficult, thr):
    """Greedy devkit matching for one image/class. iou: (n_det sorted by score desc, n_gt).
    Returns per-detection flag: 1 TP, 0 FP, -1 ignored (matched a difficult GT)."""
    n = iou.shape[0]
    flag = np.zeros(n, dtype=np.int8)
    if iou.shape[1] == 0:
        return flag
    taken = np.zeros(iou.shape[1], dtype=bool)
    jmax = iou.argmax(1)
    ovmax = iou[np.arange(n), jmax]
    for d in range(n):
        if ovmax[d] > thr:
            j = jmax[d]
            if difficult[j]:
                flag[d] = -1
            elif not taken[j]:
                flag[d] = 1
                taken[j] = True
    return flag


def evaluate(dets, gts, classes=DOTA1_CLASSES, iou_thrs=None):
    """dets: {class_name: (image_ids, scores, polys)}; gts: {image_id: [(poly, cls, difficult)]}.

    Images present in `gts` define the evaluation set; detections on other images are dropped."""
    iou_thrs = np.round(np.arange(0.5, 0.96, 0.05), 2) if iou_thrs is None else np.asarray(iou_thrs)
    res = {"classes": {}, "n_images": len(gts)}
    grid = np.linspace(0, 1, 101)                         # confidence thresholds for the P / R / F curves
    curves = {}
    for c in classes:
        gt_c = {img: ([o[0] for o in objs if o[1] == c], np.array([o[2] for o in objs if o[1] == c], dtype=bool))
                for img, objs in gts.items()}
        npos = int(sum((~d).sum() for _, d in gt_c.values()))
        ids, sc, pl = dets.get(c, ([], np.zeros(0), np.zeros((0, 8))))
        ids, sc, pl = np.asarray(ids), np.asarray(sc, dtype=np.float64), np.asarray(pl, dtype=np.float64).reshape(-1, 8)
        fin = np.isfinite(sc) & np.isfinite(pl).all(1)
        ids, sc, pl = ids[fin], sc[fin], pl[fin]
        valid = np.isin(ids, list(gts.keys())) if len(ids) else np.zeros(0, dtype=bool)
        ids, sc, pl = ids[valid], sc[valid], pl[valid]
        flags = np.zeros((len(iou_thrs), len(sc)), dtype=np.int8)
        uniq, inv = np.unique(ids, return_inverse=True) if len(ids) else (np.zeros(0), np.zeros(0, int))
        by_img = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])
        for img, idx in zip(uniq, by_img):
            idx = idx[np.argsort(-sc[idx], kind="stable")]
            gpoly, gdiff = gt_c[str(img)]
            iou = poly_iou(pl[idx], np.asarray(gpoly).reshape(-1, 8)) if gpoly else np.zeros((len(idx), 0))
            for t, thr in enumerate(iou_thrs):
                flags[t, idx] = _match_image(iou, gdiff, thr)
        order = np.argsort(-sc, kind="stable")
        i50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
        aps07, aps = [], []
        for t, thr in enumerate(iou_thrs):
            f = flags[t, order]
            f = f[f >= 0]                                  # drop detections on difficult objects
            tp = np.cumsum(f == 1)
            fp = np.cumsum(f == 0)
            rec = tp / max(npos, 1)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            aps07.append(voc_ap(rec, prec, True) if npos else 0.0)
            aps.append(voc_ap(rec, prec, False) if npos else 0.0)
            if t == i50 and npos:                         # P(conf), R(conf) curves at IoU 0.5
                if len(f):
                    s_ord = sc[order][flags[t, order] >= 0]
                    k = np.searchsorted(-s_ord, -grid, side="right")             # detections with score >= conf
                    tp_k = np.where(k > 0, tp[np.maximum(k - 1, 0)], 0)
                    curves[c] = (np.where(k > 0, tp_k / np.maximum(k, 1), 1.0), tp_k / npos)
                else:
                    curves[c] = (np.zeros_like(grid), np.zeros_like(grid))
        res["classes"][c] = {"npos": npos, "ndet": int(len(sc)), "AP50": aps07[i50],
                             "AP50_all": aps[i50], "AP50_95": float(np.mean(aps)),
                             "recall50": float((flags[i50] == 1).sum() / max(npos, 1)),
                             "images": int(sum(1 for _, d in gt_c.values() if (~d).any()))}
    cls_with_gt = [c for c in classes if res["classes"][c]["npos"] > 0]
    # P / R reported at ONE confidence threshold shared by all classes (max mean F1); per class, the threshold that
    # maximises F2 (recall-weighted, the product metric) is kept for inference (conf=auto)
    f1 = np.mean([2 * curves[c][0] * curves[c][1] / np.maximum(curves[c][0] + curves[c][1], 1e-12)
                  for c in cls_with_gt], 0) if cls_with_gt else np.zeros_like(grid)
    j1 = int(np.argmax(f1))
    res["conf_f1"] = float(grid[j1])
    for c in classes:
        r = res["classes"][c]
        if c not in curves:
            r.update(P=0.0, R=0.0, F2=0.0, conf_f2=0.25)
            continue
        P, Rc = curves[c]
        f2 = 5 * P * Rc / np.maximum(4 * P + Rc, 1e-12)
        j2 = int(np.argmax(f2))
        r.update(P=float(P[j1]), R=float(Rc[j1]), F2=float(f2[j2]), conf_f2=float(grid[j2]))

    def mean(key):
        return float(np.mean([res["classes"][c][key] for c in cls_with_gt])) if cls_with_gt else 0.0
    res["mAP50"], res["mAP50_95"], res["mAP50_allpt"] = mean("AP50"), mean("AP50_95"), mean("AP50_all")
    res["P"], res["R"], res["F2"] = mean("P"), mean("R"), mean("F2")
    res["n_instances"] = int(sum(res["classes"][c]["npos"] for c in classes))
    return res


def summary_table(res, classes=DOTA1_CLASSES, per_class=True):
    """Console table: Class / Images / Instances / P / R (one shared conf, max mean F1) / mAP50 (VOC07) / mAP50-95 /
    F2 (per-class best conf)."""
    head = (f"{'Class':>22}{'Images':>11}{'Instances':>11}{'P':>11}{'R':>11}{'mAP50':>11}{'mAP50-95':>11}"
            f"{'F2':>11}")
    rows = [head, f"{'all':>22}{res.get('n_images', 0):>11}{res.get('n_instances', 0):>11}{res.get('P', 0):>11.3f}"
                  f"{res.get('R', 0):>11.3f}{res['mAP50']:>11.3f}{res['mAP50_95']:>11.3f}{res.get('F2', 0):>11.3f}"]
    if per_class:
        for c in classes:
            r = res["classes"][c]
            if r["npos"]:
                rows.append(f"{c[:22]:>22}{r.get('images', 0):>11}{r['npos']:>11}{r.get('P', 0):>11.3f}"
                            f"{r.get('R', 0):>11.3f}{r['AP50']:>11.3f}{r['AP50_95']:>11.3f}{r.get('F2', 0):>11.3f}")
    return "\n".join(rows)


def format_table(res, classes=DOTA1_CLASSES):
    lines = [f"{'class':20s} {'npos':>6s} {'ndet':>8s} {'AP50':>6s} {'AP50:95':>8s} {'R50':>6s}"]
    for c in classes:
        r = res["classes"][c]
        lines.append(f"{c:20s} {r['npos']:6d} {r['ndet']:8d} {100*r['AP50']:6.2f} {100*r['AP50_95']:8.2f} {100*r['recall50']:6.2f}")
    lines.append(f"{'mAP':20s} {'':6s} {'':8s} {100*res['mAP50']:6.2f} {100*res['mAP50_95']:8.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True, help="dir with Task1_<class>.txt (full-image coords)")
    ap.add_argument("--gt", required=True, help="original labelTxt dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(read_task1(a.dets), load_gt_dir(a.gt))
    print(format_table(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
