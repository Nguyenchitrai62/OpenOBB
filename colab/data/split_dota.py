"""Cut DOTA-style images into fixed-size patches (our own implementation of the usual
DOTA-devkit/mmrotate protocol, no Ultralytics code).

Input  (per split):  {src}/{split}/images/*.png  and  {src}/{split}/labelTxt/*.txt (original DOTA format)
Output (per split):
  {out}/images/{split}/{img}__{rate}__{x0}___{y0}.jpg     patch, padded to size x size (ImageNet-mean gray)
  {out}/labels/{split}/<same>.txt                          'cls x1 y1 ... x4 y4' normalised by `size`
                                                           (YOLO-OBB format, readable by every pipeline)
  {out}/meta/{split}.jsonl                                 one line per patch: offsets, valid size and
                                                           pixel polygons with difficult/truncated flags
  {out}/gt/{split}/*.txt                                   copy of the original full-image labels (for eval)

Object -> patch rule: keep the object if area(obj ∩ patch) / area(obj) >= iof_thr (0.7).
Fully contained objects keep their original polygon; truncated ones are replaced by the
minimum-area rectangle of the visible part (clipped to the patch), flag trunc=1.
"""
import argparse
import json
import os
import shutil
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vrdet.eval.dota import DOTA1_CLASSES, parse_dota_txt  # noqa: E402

PAD_BGR = (104, 116, 124)       # ImageNet mean, so padding is ~0 after normalisation


def window_starts(length, size, step):
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    s.append(length - size)
    return s


def _fmt_rate(r):
    return f"{r:g}" if r != 1 else "1.0"


def split_image(item, out, split, size, gap, rates, iof_thr, classes, quality):
    img_path, lbl_path = item
    name = Path(img_path).stem
    img0 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img0 is None:
        return [], f"unreadable {img_path}"
    objs0 = parse_dota_txt(lbl_path) if lbl_path and Path(lbl_path).exists() else []
    cls_idx = {c: i for i, c in enumerate(classes)}
    objs0 = [o for o in objs0 if o[1] in cls_idx]
    metas = []
    for rate in rates:
        img = img0 if rate == 1 else cv2.resize(img0, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)
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
                cv2.imwrite(str(out / "images" / split / f"{pname}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, quality])
                with open(out / "labels" / split / f"{pname}.txt", "w") as fh:
                    for o in kept:
                        fh.write(f"{o[0]} " + " ".join(f"{v / size:.6f}" for v in o[3:]) + "\n")
                metas.append({"name": pname, "src": name, "rate": rate, "x0": x0, "y0": y0, "w": pw, "h": ph,
                              "objs": kept})
    return metas, None


def split_set(src, out, split, size=1024, gap=200, rates=(1.0,), iof_thr=0.7, classes=DOTA1_CLASSES,
              workers=None, limit=None, quality=95):
    src, out = Path(src), Path(out)
    imgs = sorted((src / split / "images").glob("*.png"))
    if limit:
        imgs = imgs[:limit]
    lbl_dir = src / split / "labelTxt"
    items = [(str(p), str(lbl_dir / f"{p.stem}.txt") if lbl_dir.exists() else None) for p in imgs]
    for d in ("images", "labels"):
        (out / d / split).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    fn = partial(split_image, out=out, split=split, size=size, gap=gap, rates=tuple(rates),
                 iof_thr=iof_thr, classes=classes, quality=quality)
    n_patch = n_obj = 0
    with Pool(workers or os.cpu_count()) as pool, open(out / "meta" / f"{split}.jsonl", "w") as mf:
        for metas, err in pool.imap_unordered(fn, items, chunksize=2):
            if err:
                print("WARN", err, flush=True)
            for m in metas:
                mf.write(json.dumps(m) + "\n")
                n_patch += 1
                n_obj += len(m["objs"])
    if lbl_dir.exists():
        gdir = out / "gt" / split
        gdir.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            lp = lbl_dir / f"{p.stem}.txt"
            if lp.exists():
                shutil.copy(lp, gdir / lp.name)
    print(f"[split] {split}: {len(imgs)} images -> {n_patch} patches, {n_obj} objects", flush=True)
    return n_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--rates", default="1.0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    for s in a.splits.split(","):
        split_set(a.src, a.out, s, a.size, a.gap, [float(r) for r in a.rates.split(",")],
                  workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
