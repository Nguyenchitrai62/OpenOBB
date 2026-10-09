"""Run a trained VRDet checkpoint on images of any size (e.g. full drawing pages rendered from PDF).

Images are prepared exactly as in training: resized so the long side = SIZE (default, whole image per forward pass),
or, for models trained with tile=True, cut into SIZE x SIZE tiles with GAP overlap. Tiles are predicted in batches and
detections are merged back per image with class-wise polygon NMS. Optional vector tokens
(tools/pdf_vectors.py output: {vectors_dir}/{stem}.npz in full-image pixels) are cut per tile for models trained
with --vectors.

Outputs per image in --out:
  {stem}.txt   "class_id x1 y1 x2 y2 x3 y3 x4 y4 score" normalised to the image (same order as the training labels + score)
  {stem}.json  [{"class": name, "score": s, "poly": [8 pixel coords]}]
  {stem}_vis.jpg  (with --vis)

python -m openobb.predict --ckpt runs/myjob/last.pt --src pages/ --out preds/ [--conf 0.25] [--vis]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from openobb.data.vectors import cut_tile
from openobb.eval.dota import merge_patches
from openobb.models.dense_head import dense_predict
from openobb.models.vrdet import VRDet, postprocess
from openobb.ops.obb import obb2poly

PAD_BGR = (104, 116, 124)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def windows(length, size, gap):
    step = size - gap
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    s.append(length - size)
    return s


def load_model(ckpt_path, device, queries=None, classes=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    names = ck.get("classes") or classes
    if not names:
        raise SystemExit("class names unknown: pass --classes (classes.json of the training data or a,b,c)")
    if a.get("arch") == "v2":
        from openobb.models.vrdet2 import VRDet2
        m = VRDet2(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v5":
        from openobb.models.vrdet5 import VRDet5
        m = VRDet5(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024), lsk=a.get("lsk", True),
                   max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), backbone=a.get("backbone", "dinov2_b"),
                   geo_cls=a.get("geo_cls", True), relate=a.get("relate", True), rel_k=a.get("rel_k", 600))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v3":
        from openobb.models.vrdet3 import VRDet3
        m = VRDet3(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024), lsk=a.get("lsk", True),
                   max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), backbone=a.get("backbone", "hgnet"))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    m = VRDet(a.get("size", "s"), num_classes=len(names), num_queries=a.get("queries", 300),
              img_size=a.get("img", 1024), rotate_sampling=not a.get("no_rotate_sampling", False),
              num_denoising=a.get("denoising", 100), dense=a.get("dense", False) or a.get("dense_queries", False),
              strip_k=a.get("strip_k", 0), ortho_heads=a.get("ortho_heads", False), context=a.get("context", False),
              dense_queries=a.get("dense_queries", False), vectors=a.get("vectors", False),
              vec_dim=a.get("vec_dim", 128), vec_layers=a.get("vec_layers", 2), lsk=a.get("lsk", False),
              vec_lfe=a.get("vec_lfe", False), vec_ground=a.get("vec_ground", 0) > 0, p2=a.get("p2", False),
              backbone=a.get("backbone", "hgnet"), dense_v3=a.get("dense_v3", False))
    sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
    m.load_state_dict(sd)
    if queries:
        m.decoder.num_queries = queries
    a = dict(a, conf_thr=ck.get("conf_thr"))
    return m.to(device).eval(), list(names), a


@torch.no_grad()
def predict_image(model, img, size, gap, batch, device, num_top, vec=None, amp=True, union=True, scale=1.0):
    """-> list of (patch_name, cls, score, poly8 in tile pixels) with patch names carrying the tile offsets.
    Hybrid models (dense head present): decoder + dense outputs are pooled ("union", the measured best rule) and
    de-duplicated by the class-wise NMS of the tile merge."""
    if scale != 1.0:
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    tiles = [(x, y) for y in windows(H, size, gap) for x in windows(W, size, gap)]
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = np.empty((len(chunk), size, size, 3), np.uint8)
        x[:] = PAD_BGR
        for j, (x0, y0) in enumerate(chunk):
            t = img[y0:y0 + size, x0:x0 + size]
            x[j, :t.shape[0], :t.shape[1]] = t
        xt = torch.from_numpy(x[..., ::-1].copy()).permute(0, 3, 1, 2).to(device).float().div_(255.0)
        side = None
        if vec is not None:
            toks = [torch.from_numpy(cut_tile(vec, x0, y0, size)) for x0, y0 in chunk]
            M = max(1, max(len(t) for t in toks))
            v = torch.zeros(len(chunk), M, 22)
            msk = torch.zeros(len(chunk), M, dtype=torch.bool)
            for j, t in enumerate(toks):
                v[j, :len(t)] = t
                msk[j, :len(t)] = True
            side = {"vec": v.to(device), "vec_mask": msk.to(device)}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(xt, ctx=side)
        s, l, b = (t.cpu().numpy() for t in postprocess(out, num_top, size))
        for j, (x0, y0) in enumerate(chunk):
            for sc, lab, p in zip(s[j], l[j], obb2poly(b[j])):
                dets.append((f"img__{scale}__{x0}___{y0}", int(lab), float(sc), p))
        if union and "dense_logits" in out:
            for (x0, y0), (ds, dl, db) in zip(chunk, dense_predict(out, img_size=size)):
                for sc, lab, p in zip(ds.cpu().numpy(), dl.cpu().numpy(), obb2poly(db.cpu().numpy())):
                    dets.append((f"img__{scale}__{x0}___{y0}", int(lab), float(sc), p))
    return dets


def conf_thresholds(names, targs, conf):
    """Per-class score thresholds: one value, or conf="auto" for the per-class best-F2 values saved in best.pt."""
    if str(conf).lower() == "auto":
        saved = targs.get("conf_thr") or {}
        if not saved:
            print("[openobb] checkpoint has no per-class thresholds: using conf 0.25")
        return [float(saved.get(n, 0.25)) for n in names]
    return [float(conf)] * len(names)


def merge_dets(dets, names, conf=0.25, iou=0.1):
    """Tile detections (predict_image) -> per-image list [{"class", "class_id", "score", "poly" (8 pixel coords)}]
    after the score threshold and class-wise polygon NMS across tiles."""
    merged = merge_patches([d for d in dets if d[2] >= conf], iou_thr=iou)
    js = []
    for c, (_, sc, pl) in merged.items():
        for s, p in zip(sc, pl):
            js.append({"class": names[c], "class_id": int(c), "score": round(float(s), 4),
                       "poly": [round(float(v), 1) for v in p]})
    return js


def to_label_lines(js, shape, with_score=False):
    """-> 'class_id x1 y1 ... x4 y4 [score]' lines normalised to the image (training label format)."""
    H, W = shape[:2]
    rows = []
    for d in js:
        q = [v / (W if i % 2 == 0 else H) for i, v in enumerate(d["poly"])]
        rows.append(f"{d['class_id']} " + " ".join(f"{v:.6f}" for v in q) + (f" {d['score']:.4f}" if with_score else ""))
    return rows


def run(ckpt, source, out=None, conf=0.25, names=None, size=None, gap=200, queries=900, batch=8, vis=False,
        vectors_dir=None, device=None, verbose=True, scale=None, tile=None, classes=None):
    """Predict every image in `source` (file or folder). Images are resized like in training: long side = tile size
    for models trained with whole-image resize (default), or tiled at `scale` (tile=True / tiled models).
    Returns {image_path: [{"class", "class_id", "score", "poly"}]} with pixel polygons; with `out`, also writes
    {stem}.txt / {stem}.json (and {stem}_vis.jpg)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(names, str):
        p = Path(names)
        names = json.loads(p.read_text()) if p.exists() else [c.strip() for c in names.split(",")]
    model, names, targs = load_model(ckpt, device, queries, names)
    keep = None
    if classes is not None:                         # filter: class ids and/or names
        cl = classes if isinstance(classes, (list, tuple)) else str(classes).split(",")
        keep = {int(c) if str(c).strip().isdigit() else names.index(str(c).strip()) for c in cl}
    thr = conf_thresholds(names, targs, conf)
    size = size or targs.get("img", 1024)
    scale = scale or targs.get("scale", 1.0)
    fit = targs.get("fit", False) if tile is None else not tile
    src = Path(source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
    results = {}
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable {f}")
            continue
        vec = None
        if vectors_dir and targs.get("vectors"):
            vp = Path(vectors_dir) / f"{f.stem}.npz"
            vec = dict(np.load(vp)) if vp.exists() else None
        dets = predict_image(model, img, size, gap, batch, device, queries, vec,
                             scale=size / max(img.shape[:2]) if fit else scale)
        js = merge_dets([d for d in dets if d[2] >= thr[d[1]]], names, 0.0, iou=0.7 if fit else 0.1)
        if keep is not None:
            js = [d for d in js if d["class_id"] in keep]
        results[str(f)] = js
        if out is not None:
            rows = to_label_lines(js, img.shape[:2], with_score=True)
            (out / f"{f.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            (out / f"{f.stem}.json").write_text(json.dumps(js, ensure_ascii=False))
            if vis:
                cv2.imwrite(str(out / f"{f.stem}_vis.jpg"), draw(img, js))
        if verbose:
            print(f"{f.name}: {len(js)} objects")
    return results


def draw(img, dets):
    vis = img.copy()
    for d in dets:
        pts = np.array(d["poly"]).reshape(4, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        cv2.putText(vis, f"{d['class']} {d['score']:.2f}", tuple(int(v) for v in pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True, help="image file or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="classes.json or comma-separated names (if not in ckpt)")
    ap.add_argument("--classes", default=None, help="keep only these class ids / names (comma-separated)")
    ap.add_argument("--vectors-dir", default=None, help="per-image vector npz from tools/pdf_vectors.py")
    ap.add_argument("--size", type=int, default=None, help="tile size (default: training img size)")
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--queries", type=int, default=900, help="inference queries (900 measured best)")
    ap.add_argument("--conf", default="0.25", help="score threshold, or 'auto' (per-class best F2 from best.pt)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--tile", action="store_true", help="tile at native resolution instead of the training resize")
    a = ap.parse_args(argv)
    run(a.ckpt, a.src, a.out, a.conf if a.conf == "auto" else float(a.conf), a.names, a.size, a.gap, a.queries,
        a.batch, a.vis, a.vectors_dir, tile=True if a.tile else None, classes=a.classes)


if __name__ == "__main__":
    main()
