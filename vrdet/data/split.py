"""Cut large images into fixed-size overlapping tiles with OBB labels (VRDet's own implementation of the usual
DOTA-devkit protocol). Used for DOTA, FloorPlanCAD and labelled OBB datasets (vrdet.data.prepare).

Output (per split):
  {out}/images/{split}/{img}__{rate}__{x0}___{y0}.jpg   tile, padded to size x size (ImageNet-mean gray)
  {out}/labels/{split}/<same>.txt                        'cls x1 y1 ... x4 y4' normalised by `size`
  {out}/meta/{split}.jsonl                               offsets, valid size, pixel polygons + flags per tile
  {out}/gt/{split}/{img}.txt                             full-image labels (DOTA format) for evaluation
  {out}/thumbs/{split}/{img}.jpg                         whole image, long side THUMB px

Object -> tile rule: keep the object if area(obj & tile) / area(obj) >= iof_thr (0.7); truncated objects become
the minimum-area rectangle of their visible part (trunc=1).

fit=True: every image is first resized so its long side equals `size` (one tile per image, any input resolution);
the per-image rate is in the tile name, so evaluation maps detections back to original pixels.
"""
import json
import os
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import shapely

from vrdet.eval.dota import parse_dota_txt

PAD_BGR = (104, 116, 124)       # ImageNet mean, so padding is ~0 after normalisation
THUMB = 512                     # long side of the whole-image thumbnail used for global context


def window_starts(length, size, step):
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    s.append(length - size)
    return s


def _fmt_rate(r):
    return f"{r:g}" if r != 1 else "1.0"


def split_image(item, out, split, size, gap, rates, iof_thr, classes, quality, fit=False):
    img_path, lbl_path = item
    name = Path(img_path).stem
    img0 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img0 is None:
        return [], f"unreadable {img_path}"
    objs0 = parse_dota_txt(lbl_path) if lbl_path and Path(lbl_path).exists() else []
    cls_idx = {c: i for i, c in enumerate(classes)}
    objs0 = [o for o in objs0 if o[1] in cls_idx]
    H0, W0 = img0.shape[:2]
    ts = THUMB / max(H0, W0)
    thumb = cv2.resize(img0, (max(1, round(W0 * ts)), max(1, round(H0 * ts))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "thumbs" / split / f"{name}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    metas = []
    seen = set()                                    # objects kept (whole or truncated) in at least one tile
    for rate in ((size / max(H0, W0),) if fit else rates):
        img = img0 if rate == 1 else cv2.resize(img0, (max(1, round(W0 * rate)), max(1, round(H0 * rate))),
                                                interpolation=cv2.INTER_AREA if rate < 1 else cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        if objs0:
            polys = np.array([o[0] for o in objs0], dtype=np.float64) * rate
            geoms = shapely.polygons(polys.reshape(-1, 4, 2))
            bad = ~shapely.is_valid(geoms)
            if bad.any():
                geoms[bad] = shapely.convex_hull(geoms[bad])
            areas = shapely.area(geoms)
            hx0, hx1 = polys[:, 0::2].min(1), polys[:, 0::2].max(1)
            hy0, hy1 = polys[:, 1::2].min(1), polys[:, 1::2].max(1)
        step = size - gap
        for y0 in window_starts(H, size, step):
            for x0 in window_starts(W, size, step):
                pname = f"{name}__{_fmt_rate(rate)}__{x0}___{y0}"
                patch = img[y0:y0 + size, x0:x0 + size]
                ph, pw = patch.shape[:2]
                if ph < size or pw < size:
                    patch = cv2.copyMakeBorder(patch, 0, size - ph, 0, size - pw, cv2.BORDER_CONSTANT, value=PAD_BGR)
                kept = []
                if objs0:
                    cand = np.nonzero((hx1 > x0) & (hx0 < x0 + pw) & (hy1 > y0) & (hy0 < y0 + ph) & (areas > 0))[0]
                    if len(cand):
                        win = shapely.box(x0, y0, x0 + pw, y0 + ph)
                        inter = shapely.intersection(geoms[cand], win)
                        iof = shapely.area(inter) / areas[cand]
                        for k, g, f in zip(cand, inter, iof):
                            if f < iof_thr:
                                continue
                            if f > 0.999:
                                q = polys[k].copy()
                                trunc = 0
                            else:
                                pts = np.asarray(shapely.get_coordinates(g), dtype=np.float32)
                                q = cv2.boxPoints(cv2.minAreaRect(pts)).reshape(-1).astype(np.float64)
                                trunc = 1
                            q[0::2] = np.clip(q[0::2] - x0, 0, pw)
                            q[1::2] = np.clip(q[1::2] - y0, 0, ph)
                            kept.append([cls_idx[objs0[k][1]], int(objs0[k][2]), trunc] + [round(v, 2) for v in q])
                            seen.add(int(k))
                cv2.imwrite(str(out / "images" / split / f"{pname}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, quality])
                with open(out / "labels" / split / f"{pname}.txt", "w") as fh:
                    for o in kept:
                        fh.write(f"{o[0]} " + " ".join(f"{v / size:.6f}" for v in o[3:]) + "\n")
                metas.append({"name": pname, "src": name, "rate": rate, "x0": x0, "y0": y0, "w": pw, "h": ph,
                              "img_w": W0, "img_h": H0, "objs": kept})
    if metas:
        metas[0]["lost"] = len(objs0) - len(seen)   # objects too large to be >= iof_thr inside any tile
    return metas, None


def split_items(items, out, split, size=1024, gap=200, rates=(1.0,), iof_thr=0.7, classes=(), workers=None,
                quality=95, fit=False):
    """items: list of (image_path, dota_label_path or None). Returns the number of tiles written."""
    out = Path(out)
    for d in ("images", "labels", "thumbs", "gt"):
        (out / d / split).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    fn = partial(split_image, out=out, split=split, size=size, gap=gap, rates=tuple(rates),
                 iof_thr=iof_thr, classes=tuple(classes), quality=quality, fit=fit)
    n_patch = n_obj = n_lost = 0
    with Pool(workers or os.cpu_count()) as pool, open(out / "meta" / f"{split}.jsonl", "w") as mf:
        for metas, err in pool.imap_unordered(fn, items, chunksize=2):
            if err:
                print("WARN", err, flush=True)
            for m in metas:
                n_lost += m.pop("lost", 0)
                mf.write(json.dumps(m) + "\n")
                n_patch += 1
                n_obj += len(m["objs"])
    for img, lbl in items:
        if lbl and Path(lbl).exists():
            shutil.copy(lbl, out / "gt" / split / f"{Path(img).stem}.txt")
    print(f"[split] {split}: {len(items)} images -> {n_patch} tiles, {n_obj} objects", flush=True)
    if n_lost:
        print(f"[split] WARNING {split}: {n_lost} objects are too long/large to be >= {iof_thr:.0%} inside any "
              f"{size}px tile and are missing from this split; use a larger tile (imgsz) or a smaller scale",
              flush=True)
    return n_patch
