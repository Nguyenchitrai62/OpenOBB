"""Run a trained VRDet checkpoint on images of any size (e.g. full drawing pages rendered from PDF).

Each image is cut into SIZE x SIZE tiles with GAP overlap (same protocol as training), tiles are predicted in
batches, and detections are merged back per image with class-wise polygon NMS. Optional vector tokens
(tools/pdf_vectors.py output: {vectors_dir}/{stem}.npz in full-image pixels) are cut per tile for models trained
with --vectors.

Outputs per image in --out:
  {stem}.txt   "class_id x1 y1 x2 y2 x3 y3 x4 y4 score" normalised to the image (YOLO-OBB order + score)
  {stem}.json  [{"class": name, "score": s, "poly": [8 pixel coords]}]
  {stem}_vis.jpg  (with --vis)

python -m vrdet.predict --ckpt runs/myjob/last.pt --src pages/ --out preds/ [--conf 0.25] [--vis]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from vrdet.data.vectors import cut_tile
from vrdet.eval.dota import merge_patches
from vrdet.models.dense_head import dense_predict
from vrdet.models.vrdet import VRDet, postprocess
from vrdet.ops.obb import obb2poly

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
    m = VRDet(a.get("size", "s"), num_classes=len(names), num_queries=a.get("queries", 300),
              img_size=a.get("img", 1024), rotate_sampling=not a.get("no_rotate_sampling", False),
              num_denoising=a.get("denoising", 100), dense=a.get("dense", False) or a.get("dense_queries", False),
              strip_k=a.get("strip_k", 0), ortho_heads=a.get("ortho_heads", False), context=a.get("context", False),
              dense_queries=a.get("dense_queries", False), vectors=a.get("vectors", False),
              vec_dim=a.get("vec_dim", 128), vec_layers=a.get("vec_layers", 2), lsk=a.get("lsk", False),
              vec_lfe=a.get("vec_lfe", False))
    sd = ck["ema"]["module"] if "ema" in ck else ck["model"]
    m.load_state_dict(sd)
    if queries:
        m.decoder.num_queries = queries
    return m.to(device).eval(), list(names), a


@torch.no_grad()
def predict_image(model, img, size, gap, batch, device, num_top, vec=None, amp=True, union=True):
    """-> list of (patch_name, cls, score, poly8 in tile pixels) with patch names carrying the tile offsets.
    Hybrid models (dense head present): decoder + dense outputs are pooled ("union", the measured best rule) and
    de-duplicated by the class-wise NMS of the tile merge."""
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
                dets.append((f"img__1.0__{x0}___{y0}", int(lab), float(sc), p))
        if union and "dense_logits" in out:
            for (x0, y0), (ds, dl, db) in zip(chunk, dense_predict(out, img_size=size)):
                for sc, lab, p in zip(ds.cpu().numpy(), dl.cpu().numpy(), obb2poly(db.cpu().numpy())):
                    dets.append((f"img__1.0__{x0}___{y0}", int(lab), float(sc), p))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True, help="image file or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=None, help="classes.json or comma-separated names (if not in ckpt)")
    ap.add_argument("--vectors-dir", default=None, help="per-image vector npz from tools/pdf_vectors.py")
    ap.add_argument("--size", type=int, default=None, help="tile size (default: training img size)")
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--queries", type=int, default=900, help="inference queries (900 measured best)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = None
    if a.classes:
        p = Path(a.classes)
        classes = json.loads(p.read_text()) if p.exists() else [c.strip() for c in a.classes.split(",")]
    model, names, targs = load_model(a.ckpt, device, a.queries, classes)
    size = a.size or targs.get("img", 1024)
    src = Path(a.src)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable {f}")
            continue
        vec = None
        if a.vectors_dir and targs.get("vectors"):
            vp = Path(a.vectors_dir) / f"{f.stem}.npz"
            vec = dict(np.load(vp)) if vp.exists() else None
        dets = predict_image(model, img, size, a.gap, a.batch, device, a.queries, vec)
        merged = merge_patches([d for d in dets if d[2] >= a.conf], iou_thr=0.1)
        H, W = img.shape[:2]
        rows, js = [], []
        for c, (_, sc, pl) in merged.items():
            for s, p in zip(sc, pl):
                q = p.copy()
                q[0::2] /= W
                q[1::2] /= H
                rows.append(f"{c} " + " ".join(f"{v:.6f}" for v in q) + f" {s:.4f}")
                js.append({"class": names[c], "class_id": int(c), "score": round(float(s), 4),
                           "poly": [round(float(v), 1) for v in p]})
        (out / f"{f.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        (out / f"{f.stem}.json").write_text(json.dumps(js, ensure_ascii=False))
        if a.vis:
            vis = img.copy()
            for d in js:
                pts = np.array(d["poly"]).reshape(4, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
                cv2.putText(vis, f"{d['class']} {d['score']:.2f}", tuple(int(v) for v in pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(out / f"{f.stem}_vis.jpg"), vis)
        print(f"{f.name}: {len(js)} objects")


if __name__ == "__main__":
    main()
