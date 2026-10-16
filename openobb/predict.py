"""Run a trained OpenOBB checkpoint on images of any size (e.g. full drawing pages rendered from PDF).

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
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from openobb.data.imgio import IMG_EXT, imread, imwrite
from openobb.data.vectors import cut_tile
from openobb.eval.dota import merge_patches
from openobb.models.dense_head import dense_predict
from openobb.models.openobb1 import OpenOBB1, postprocess
from openobb.models.openobb7 import fuse_model
from openobb.ops.obb import obb2poly

PAD_BGR = (104, 116, 124)


def windows(length, size, gap):
    step = size - gap
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    s.append(length - size)
    return s


# architecture labels this package can rebuild (args["arch"] written at training; identical in every build of the
# package whatever its name, so a model trained with one build loads in another of the same or a newer version)
KNOWN_ARCHS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23")


def _ver(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except ValueError:
        return (0,)


def check_arch(ck, path=""):
    """Architecture of a checkpoint dict -> its label; clear error when this package version cannot build it."""
    from openobb import __version__
    a = ck.get("args") or {}
    arch = a.get("arch") or "v1"
    trained = ck.get("version")
    if arch not in KNOWN_ARCHS:
        need = f" (trained with version {trained})" if trained else ""
        raise SystemExit(f"{path}: architecture '{arch}'{need} is not in this package (version {__version__}, "
                         f"architectures {', '.join(KNOWN_ARCHS)}): update the package")
    if trained and _ver(trained) > _ver(__version__):
        print(f"WARNING {path}: trained with version {trained}, this package is {__version__}; update it if the "
              f"model does not load or predicts differently")
    return arch


def parse_nms(v):
    """NMS setting -> None (architecture default: "auto" / missing), "none" (no NMS) or a ProbIoU threshold."""
    if v is None or str(v).lower() in ("auto", ""):
        return None
    if str(v).lower() in ("none", "off", "false"):
        return "none"
    if str(v).lower().startswith("peak"):            # local-peak filter (OpenOBB17+), "peak+0.5": then NMS 0.5 (F95)
        return str(v).lower()
    return float(v)


def apply_nms(model, v):
    """Set a model's duplicate suppression: a threshold -> class-wise NMS at it (OpenOBB11: on its set output),
    "none" -> no NMS, None -> the architecture default. The value is calibrated on val at the end of training
    (args["nms_iou"]) and is the default at inference."""
    v = parse_nms(v)
    if v is None or not hasattr(model, "nms_iou"):
        return model
    if hasattr(model, "final_nms"):
        model.final_nms = None if v == "none" else v
    elif str(v).startswith("peak"):
        return model
    else:
        model.nms_iou = 1.01 if v == "none" else v
    return model


def apply_head(model, h):
    """Dual-head models (OpenOBB12 / 13): the head used at inference ("o2o" / "o2m"), calibrated at training."""
    if h in ("o2o", "o2m") and hasattr(model, "infer_head"):
        model.infer_head = h
    return model


def postprocess_options(model):
    """(inference head, NMS) settings the end-of-training calibration tries for this model."""
    if hasattr(model, "infer_head"):
        peak = [("o2o", "peak"), ("o2o", "peak+0.5"), ("o2m", "peak+0.5")] if hasattr(model, "snap_topk") else []  # F95
        return [("o2o", "none"), ("o2o", 0.7), ("o2o", 0.5), ("o2m", 0.7), ("o2m", 0.5)] + peak
    if hasattr(model, "final_nms"):
        return [(None, "none"), (None, 0.7), (None, 0.5)]
    return [(None, 0.7), (None, 0.6), (None, 0.5)]


DUP_EQ = 0.005    # duplicate rates closer than this (0.5 points) are treated as equal (F76: 2.1% vs 2.2%)
DUP_TOL = 0.01    # fitness margin within which the post-processing with fewer duplicate boxes wins (F69, F74: < half the
                  # ruler noise of a 53-image val, +-0.024)


def dup_rate(u):
    """Share of duplicate boxes (conf 0.25, same class on an already detected object) of an evaluate_ultra result."""
    cm = u.get("confusion") or {}
    d = cm.get("duplicates")
    return None if d is None else sum(d.values()) / max(cm.get("detections", 0), 1)


def choose_postprocess(tried, tol=DUP_TOL):
    """tried: [(fitness, duplicate rate or None, ...)] -> the entry to keep. Among the settings within `tol` of the
    best fitness (0.1 mAP50 + 0.9 mAP50-95), the one with the fewest duplicate boxes: duplicates inflate object counts
    (take-off) and NMS 0.5 vs 0.7 typically costs < 0.005 fitness while cutting duplicates ~6x (F69)."""
    best = max(t[0] for t in tried)
    near = [t for t in tried if t[0] >= best - tol]
    dups = [t[1] for t in near if t[1] is not None]
    if not dups:
        return max(near, key=lambda t: t[0])
    # duplicate rates within DUP_EQ of the lowest count as equal: among those, the most accurate setting
    low = [t for t in near if t[1] is not None and t[1] <= min(dups) + DUP_EQ]
    pick = max(low, key=lambda t: t[0])
    # ...unless an equally accurate / duplicate-free setting tilts its long boxes clearly less (F95: the one-to-one head
    # + NMS 0.5 tilted 5.5% of the long detections vs 15.4% for one-to-many + NMS 0.5, at the same fitness)
    tilt = lambda t: _long_tilt(t[3]) if len(t) > 3 else None  # noqa: E731
    tp = tilt(pick)
    if tp is not None:
        better = [t for t in low if tilt(t) is not None and tilt(t) <= tp - TILT_GAIN]
        if better:
            pick = min(better, key=lambda t: (tilt(t), -t[0]))
    return pick


TILT_GAIN = 0.02  # a calibration setting is preferred for its geometry when it tilts >= 2 points fewer long boxes (F95)


def _long_tilt(r):
    """Share of long detections > 1 deg off the axes in an eval result (openobb.eval.ultra.geometry), or None."""
    if not isinstance(r, dict):
        return None
    g = (r.get("ultra") or r).get("geometry") or {}
    return g.get("long_dets_tilted") if g.get("long_dets") else None


def merge_iou_for(v, fit):
    """Polygon NMS after the network (tile merge): whole-image models follow the model's NMS setting; tiled
    inference keeps 0.1 (overlapping tiles repeat the same object)."""
    v = parse_nms(v)
    if not fit:
        return 0.1
    if isinstance(v, str) and v.startswith("peak"):
        return float(v.split("+")[1]) if "+" in v else None
    return 0.7 if v is None else (None if v == "none" else v)


def fast_layout(m, device):
    """Dense OpenOBB models (v12+) on a GPU: weights in channels_last (NHWC), the layout tensor cores read natively;
    same outputs, trunk + neck 7.2 -> 5.7 ms / image on a G4 at batch 16 (F97). OPENOBB_CHANNELS_LAST=0 turns it off."""
    import os
    if (torch.device(device).type == "cuda" and hasattr(m, "run_head")
            and os.environ.get("OPENOBB_CHANNELS_LAST", "1") != "0"):
        m = m.to(memory_format=torch.channels_last)
    return m


def load_model(ckpt_path, device, queries=None, classes=None):
    if torch.device(device).type == "cpu":
        # flush denormals (identical detections; speed effect not reproducible on the dev machine, F57)
        torch.set_flush_denormal(True)
    m, names, a = _load_model(ckpt_path, device, queries, classes)
    m = fast_layout(m, device)
    if a.get("infer_k") and getattr(m, "refiner", None) is not None:   # calibrated at training
        m.refiner.k = int(a["infer_k"])
    return apply_head(apply_nms(m, a.get("nms_iou")), a.get("infer_head")), names, a


def _load_model(ckpt_path, device, queries=None, classes=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_arch(ck, ckpt_path)
    a = ck.get("args", {})
    names = ck.get("classes") or classes
    if not names:
        raise SystemExit("class names unknown: pass --classes (classes.json of the training data or a,b,c)")
    if a.get("arch") == "v2":
        from openobb.models.openobb2 import OpenOBB2
        m = OpenOBB2(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v9":
        from openobb.models.openobb9 import OpenOBB9
        m = OpenOBB9(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024),
                     max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), off=a.get("v9_off") or None,
                     off8=a.get("v8_off") or None, rel_k=a.get("rel_k", 600))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval().fuse(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") in ("v14", "v15", "v16"):
        from openobb.models.openobb14 import OpenOBB14
        from openobb.models.openobb15 import OpenOBB15
        from openobb.models.openobb16 import OpenOBB16
        cls = {"v14": OpenOBB14, "v15": OpenOBB15, "v16": OpenOBB16}[a["arch"]]
        m = cls(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024),
                      max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), off=a.get("v14_off") or None,
                      box=a.get("box") or "direct", ref_k=int(a.get("ref_k", 1500)), neck=a.get("neck") or "own")
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval().fuse(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") in ("v7", "v8", "v10", "v11", "v12", "v13", "v17", "v18", "v19", "v20", "v21", "v22", "v23"):
        common = dict(num_classes=len(names), img_size=a.get("img", 1024), lsk=a.get("lsk", True),
                      max_det=max(int(a.get("num_top", 1000)), int(queries or 0)))
        if a["arch"] in ("v17", "v18", "v19", "v20", "v21", "v22", "v23"):
            from openobb.models.openobb17 import OpenOBB17
            from openobb.models.openobb18 import OpenOBB18, OpenOBB19, OpenOBB20, OpenOBB21, OpenOBB22, OpenOBB23
            m = {"v17": OpenOBB17, "v18": OpenOBB18, "v19": OpenOBB19, "v20": OpenOBB20, "v21": OpenOBB21,
                 "v22": OpenOBB22, "v23": OpenOBB23}[a["arch"]](
                a.get("size", "x"), off=a.get("v7_off") or None, box=a.get("box") or "dfl",
                ref_k=int(a.get("ref_k", 1500)), groups=a.get("groups") or None, **common)
        elif a["arch"] in ("v12", "v13"):
            from openobb.models.openobb12 import OpenOBB12, OpenOBB13
            m = (OpenOBB13 if a["arch"] == "v13" else OpenOBB12)(a.get("size", "x"), off=a.get("v7_off") or None,
                                                               box=a.get("box") or "dfl", **common)
        elif a["arch"] == "v11":
            from openobb.models.openobb11 import OpenOBB11
            m = OpenOBB11(a.get("size", "x"), off=a.get("v8_off") or None, off7=a.get("v7_off") or None,
                          rel_k=a.get("rel_k", 600), **common)
        elif a["arch"] == "v10":
            from openobb.models.openobb10 import OpenOBB10
            m = OpenOBB10(a.get("size", "x"), off=a.get("v10_off") or None, off8=a.get("v8_off") or None,
                          off7=a.get("v7_off") or None, rel_k=a.get("rel_k", 600), **common)
        elif a["arch"] == "v8":
            from openobb.models.openobb8 import OpenOBB8
            m = OpenOBB8(a.get("size", "x"), off=a.get("v8_off") or None, off7=a.get("v7_off") or None,
                         rel_k=a.get("rel_k", 600), **common)
        else:
            from openobb.models.openobb7 import OpenOBB7
            m = OpenOBB7(a.get("size", "x"), off=a.get("v7_off") or None, **common)
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        m = m.to(device).eval()
        if hasattr(m, "fuse"):
            m.fuse()
        return m, list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v6":
        from openobb.models.openobb6 import OpenOBB6
        m = OpenOBB6(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024),
                     max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), off=a.get("v6_off") or None)
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return m.to(device).eval().fuse(), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v5":
        from openobb.models.openobb5 import OpenOBB5
        m = OpenOBB5(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024), lsk=a.get("lsk", True),
                   max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), backbone=a.get("backbone", "dinov2_b"),
                   geo_cls=a.get("geo_cls", True), relate=a.get("relate", True), rel_k=a.get("rel_k", 600))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return fuse_model(m.to(device).eval()), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    if a.get("arch") == "v3":
        from openobb.models.openobb3 import OpenOBB3
        m = OpenOBB3(a.get("size", "x"), num_classes=len(names), img_size=a.get("img", 1024), lsk=a.get("lsk", True),
                   max_det=max(int(a.get("num_top", 1000)), int(queries or 0)), backbone=a.get("backbone", "hgnet"))
        m.load_state_dict(ck["ema"]["module"] if "ema" in ck else ck["model"])
        return fuse_model(m.to(device).eval()), list(names), dict(a, conf_thr=ck.get("conf_thr"))
    m = OpenOBB1(a.get("size", "s"), num_classes=len(names), num_queries=a.get("queries", 300),
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


def input_size(args, imgsz=None):
    """Network input size for a checkpoint's args: imgsz given by the user means the core's size for every model,
    so OpenOBB13 (core at half of its input, full-resolution detail stream) gets 2x; no imgsz -> the trained size."""
    if not imgsz:
        return int(args.get("img", 1024))
    return int(imgsz) * (2 if args.get("arch") in ("v13", "v15", "v16") else 1)


def rect_shape(h, w, size, stride=32):
    """Network input (H, W) for a whole image of h x w <= size (YOLO letterbox auto / rect): each side padded up to
    a multiple of `stride` (capped at `size`), instead of the full size x size square."""
    return min(math.ceil(h / stride) * stride, size), min(math.ceil(w / stride) * stride, size)


@torch.no_grad()
def predict_image(model, img, size, gap, batch, device, num_top, vec=None, amp=True, union=True, scale=1.0,
                  rect=True, info=None):
    """-> list of (patch_name, cls, score, poly8 in tile pixels) with patch names carrying the tile offsets.
    Hybrid models (dense head present): decoder + dense outputs are pooled ("union", the measured best rule) and
    de-duplicated by the class-wise NMS of the tile merge.
    rect: an image that fits in one tile (whole-image resize) is padded only to multiples of 32 (rect_shape), top-left
    anchored like the square tile; larger images keep square size x size tiles. info (dict): gets "shape", the
    network input shape (B, 3, H, W) of the last batch."""
    if torch.device(device).type == "cpu":
        # flush denormals (identical detections; harmless). Its speed effect on the dev machine was NOT reproducible
        # once runs alternated (3.4-10 s either way, F57). The flag is per thread, so it is set on every call
        torch.set_flush_denormal(True)
    if scale != 1.0:
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    tiles = [(x, y) for y in windows(H, size, gap) for x in windows(W, size, gap)]
    # vector tokens are cut / normalised per square tile: keep the square input for those models
    th, tw = rect_shape(H, W, size) if rect and vec is None and len(tiles) == 1 else (size, size)
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = np.empty((len(chunk), th, tw, 3), np.uint8)
        x[:] = PAD_BGR
        for j, (x0, y0) in enumerate(chunk):
            t = img[y0:y0 + th, x0:x0 + tw]
            x[j, :t.shape[0], :t.shape[1]] = t
        xt = torch.from_numpy(x[..., ::-1].copy()).permute(0, 3, 1, 2).to(device).float().div_(255.0)
        if info is not None:
            info["shape"] = tuple(xt.shape)
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
    model.nms_conf = max(0.001, min(thr))            # pre-NMS score filter (does not change the kept boxes)
    size = input_size(targs, size)
    scale = scale or targs.get("scale", 1.0)
    fit = targs.get("fit", False) if tile is None else not tile
    src = Path(source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
    results = {}
    for f in files:
        img = imread(f)
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
                imwrite(out / f"{f.stem}_vis.jpg", draw(img, js))
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
