"""OpenOBB command line and Python API. Arguments are key=value pairs.

    openobb train   data=data.yaml model=openobb3x epochs=100 imgsz=1280     # -> runs/obb/train, train2, ...
    openobb train   data=data.yaml model=runs/obb/train/weights/best.pt epochs=50   # fine-tune
    openobb train   resume=True                                                # continue the latest unfinished run
    openobb val     model=runs/obb/train/weights/best.pt data=data.yaml
    openobb predict model=runs/obb/train/weights/best.pt source=pages/ conf=0.25   # -> runs/obb/predict, ...
                    (the arguments, defaults and outputs of `yolo obb predict`: save=True writes the annotated
                    images, save_txt=True labels/<image>.txt, save_conf, save_crop, iou, imgsz, max_det, classes...)
    openobb prepare data=data.yaml out=datasets/mydata imgsz=1024
    openobb pretrain model=openobb9x data=<image folders> val_data=<held-out drawings> epochs=20 imgsz=640
                    # OpenOBB9 core by two-teacher distillation, no labels (docs/PRETRAIN.md)
    openobb calibrate model=best.pt data=data.yaml [out=calibrated.pt]   # re-choose the inference head / NMS on val
    openobb strip   model=runs/obb/train/weights/best.pt [out=model.pt half=True]   # deploy checkpoint:
                    # weights + architecture + classes only (no paths, training args or scores)

    from openobb import OpenOBB
    model = OpenOBB("openobb3x")                # or OpenOBB("runs/obb/train/weights/best.pt")
    r = model.train(data="data.yaml", epochs=100, imgsz=1024)    # r.best = runs/obb/train/weights/best.pt
    model.predict("pages/", conf=0.25)

Datasets: data.yaml ('names' + train/val image folders) with labels 'class x1 y1 x2 y2 x3 y3 x4 y4' normalised to
[0, 1] in a sibling 'labels' folder. Images of any size work: each image is resized so its long side = imgsz
(training, validation and prediction alike). tile=True instead cuts large pages into imgsz tiles at native
resolution (tile_scale= to resize first), for very large pages with tiny objects. Without a val split, val_frac of
the train images is held out. Prepared data is cached (~/.cache/openobb). Every train call starts a new run folder
({project}/{name}, then name2, name3...; exist_ok=True reuses it); resume=True continues a run from weights/last.pt.

Training defaults follow the usual one-stage recipe: mosaic=1.0, scale=0.5 (random zoom 0.5-1.5), translate=0.1,
close_mosaic=10, warmup_epochs=3 (3 epochs of batches, at most epochs - 1, every LR rises from 0; warmup_bias_lr=0 as the usual
trainer sets it for AdamW: a bias LR of 0.1 under Adam moves every bias by ~0.1 per step and wrecks a fine-tune),
LR factor per epoch, cosine to lr0 * lrf (lrf=0.01; cos_lr=False for linear, the YOLO default), nbs=64 (gradient
accumulation to a nominal batch of 64, weight decay scaled to it), patience=100,
cache=auto (decoded images in RAM when they fit), batch fitted to the GPU / dataset and halved on out-of-memory.
"""
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

MODES = ("train", "val", "predict", "prepare", "pretrain", "strip", "calibrate", "profile")
SIZES = ("s", "m", "l", "x")
# model names: openobb1{s,m,l,x} (= s/m/l/x, DETR-style OpenOBB), openobb2{n,s,m,l,x} (dense segment-aware OpenOBB2),
# openobb3{s,m,l,x} (pretrained encoder + dense anisotropic oriented head), openobb4x (DINOv2 + hybrid DETR)
MODELS = {**{f"openobb1{s}": ("v1", s) for s in SIZES},
          **{f"openobb2{s}": ("v2", s) for s in ("n", "s", "m", "l", "x")},
          **{f"openobb3{s}": ("v3", s) for s in SIZES}, "openobb4x": ("v4", "x"), "openobb5x": ("v5", "x"),
          **{f"openobb6{s}": ("v6", s) for s in ("n", "s", "m", "l", "x")},
          **{f"openobb8{s}": ("v8", s) for s in SIZES},      # OpenOBB7 = the core of OpenOBB8 (v8_off=refine)
          **{f"openobb9{s}": ("v9", s) for s in SIZES},
          **{f"openobb10{s}": ("v10", s) for s in SIZES},    # OpenOBB8 + multi-slot head (docs/OPENOBB10.md)
          **{f"openobb11{s}": ("v11", s) for s in SIZES},    # NMS-free one-to-one set decoder (docs/OPENOBB11.md)
          **{f"openobb12{s}": ("v12", s) for s in SIZES},    # dual dense head, NMS-free (docs/OPENOBB12.md)
          **{f"openobb13{s}": ("v13", s) for s in SIZES},    # + full-res stream + P2 grid (docs/OPENOBB13.md)
          **{f"openobb14{s}": ("v14", s) for s in SIZES},    # pretrained trunk + own rep-conv neck + edge refiner (docs/OPENOBB14.md)
          **{f"openobb15{s}": ("v15", s) for s in SIZES},    # v14 + full-resolution stream + P2 grid (docs/OPENOBB15.md)
          **{f"openobb16{s}": ("v16", s) for s in SIZES},    # v15 + cascade refiners + fit-aware scores (docs/OPENOBB16.md)
          **{f"openobb17{s}": ("v17", s) for s in SIZES},    # v12 + boundary field + boundary snapping (docs/OPENOBB17.md)
          **{f"openobb18{s}": ("v18", s) for s in SIZES},
          **{f"openobb19{s}": ("v19", s) for s in SIZES},
          **{f"openobb20{s}": ("v20", s) for s in SIZES},
          **{f"openobb21{s}": ("v21", s) for s in SIZES},
          **{f"openobb22{s}": ("v22", s) for s in SIZES},
          **{f"openobb23{s}": ("v23", s) for s in SIZES}}    # v21 + rotational boundary snapping + peak filter (docs/OPENOBB23.md)    # v21 + YOLO26-grade box geometry head (docs/OPENOBB22.md)    # v20 + angle in the double-angle space (docs/OPENOBB21.md)    # v19 + edge refiner on the trunk's stride-4 map (docs/OPENOBB20.md)    # v18 + contention-aware class groups (docs/OPENOBB19.md)    # v17 with an own fast neck distilled from a trained encoder (docs/OPENOBB18.md)

# Best architecture recipe measured on FloorPlanCAD / DOTA (research/LEDGER.md, c6): selective large-kernel
# adapters (LSK), repeat-factor sampling, one-to-many query group, adaptive query denoising, square-aware angle loss,
# IoU-aware matching, 900 inference queries.
RECIPE = {"lsk": True, "rfs": 0.1, "o2m_queries": 900, "o2m_k": 6, "aqd": True, "angle_weight": 1.0, "cost_iou": 0.5,
          "eval_queries": 900, "channels_last": True}     # compile=True to opt in to torch.compile
# augmentation (names as in the common one-stage trainers) -> trainer flags
AUGMENT = {"mosaic": 1.0, "scale": 0.5, "translate": 0.1, "close_mosaic": 10}
AUG_FLAGS = {"mosaic": "mosaic_p", "scale": "scale_jitter", "translate": "translate", "close_mosaic": "mosaic_off"}
SIZE_DEFAULTS = {          # lr, backbone lr multiplier, weight decay, batch, ~GB of GPU memory per image at 1024 px
    "s": dict(lr=1e-4, backbone_mult=0.5, wd=1e-4, batch=16, gb=1.4),
    "m": dict(lr=1e-4, backbone_mult=0.1, wd=1e-4, batch=16, gb=1.8),
    "l": dict(lr=8e-5, backbone_mult=0.05, wd=1.25e-4, batch=8, gb=2.6),
    "x": dict(lr=6e-5, backbone_mult=0.1, wd=1.25e-4, batch=8, gb=4.0),
}
# OpenOBB2 trains from scratch (no pretrained backbone): one LR for every layer, conv-detector regularisation
SIZE_DEFAULTS2 = {
    "n": dict(lr=2e-3, backbone_mult=1.0, wd=0.05, batch=32, gb=0.8),
    "s": dict(lr=2e-3, backbone_mult=1.0, wd=0.05, batch=16, gb=1.4),
    "m": dict(lr=1.5e-3, backbone_mult=1.0, wd=0.05, batch=16, gb=2.5),
    "l": dict(lr=1e-3, backbone_mult=1.0, wd=0.05, batch=8, gb=3.5),
    "x": dict(lr=1e-3, backbone_mult=1.0, wd=0.05, batch=8, gb=4.5),
}
RECIPE2 = {"rfs": 0.1, "channels_last": True, "clip": 10.0, "num_top": 1000}
# OpenOBB3: COCO-pretrained backbone + encoder (lower backbone LR as in OpenOBB1), new dense head; LSK adapters (c6)
SIZE_DEFAULTS3 = {
    "s": dict(lr=5e-4, backbone_mult=0.5, wd=1e-4, batch=16, gb=1.2),
    "m": dict(lr=5e-4, backbone_mult=0.1, wd=1e-4, batch=16, gb=1.6),
    "l": dict(lr=5e-4, backbone_mult=0.1, wd=1e-4, batch=8, gb=2.4),
    "x": dict(lr=5e-4, backbone_mult=0.1, wd=1e-4, batch=8, gb=3.6),
}
RECIPE3 = {"lsk": True, "rfs": 0.1, "channels_last": True, "clip": 10.0, "num_top": 1000}
# OpenOBB4: DINOv2 ViT (self-supervised, Apache-2.0) + D-FINE encoder/decoder (COCO) + OpenOBB3 oriented head as the
# dense branch (auxiliary one-to-many supervision + query proposals); output = decoder + dense branch through NMS
SIZE_DEFAULTS4 = {"x": dict(lr=1e-4, backbone_mult=0.5, wd=1e-4, batch=8, gb=6.0)}      # ViT-B/14, accuracy first
RECIPE4 = dict(RECIPE, dense=True, dense_v3=True, dense_queries=True, primary="union")
BACKBONE4 = {"x": "dinov2_b"}
# OpenOBB5: OpenOBB3's dense head as the output + DINOv2 + geometry-aware classes + relation re-scoring
SIZE_DEFAULTS5 = {"x": dict(lr=5e-4, backbone_mult=0.2, wd=1e-4, batch=8, gb=5.0)}
RECIPE5 = dict(RECIPE3, geo_cls=True, relate=True, rel_k=600)
# OpenOBB6: own backbone / neck / head (no ImageNet / COCO weights): one LR for every layer, conv-detector
# regularisation; start from `openobb pretrain` weights (weights=) when available (docs/OPENOBB6.md)
SIZE_DEFAULTS6 = {
    "n": dict(lr=2e-3, backbone_mult=1.0, wd=0.05, batch=32, gb=0.5),
    "s": dict(lr=2e-3, backbone_mult=1.0, wd=0.05, batch=32, gb=0.8),
    "m": dict(lr=1.5e-3, backbone_mult=1.0, wd=0.05, batch=16, gb=1.2),
    "l": dict(lr=1e-3, backbone_mult=1.0, wd=0.05, batch=16, gb=1.8),
    "x": dict(lr=1e-3, backbone_mult=1.0, wd=0.05, batch=16, gb=2.6),
}
RECIPE6 = {"rfs": 0.1, "channels_last": True, "clip": 10.0, "num_top": 1000}
# OpenOBB7 / OpenOBB8: OpenOBB3's pretrained core (COCO init, lower backbone LR) repaired (+ relational refiner)
RECIPE7 = dict(RECIPE3)
RECIPE8 = dict(RECIPE3, rel_k=600)
RECIPE12 = dict(RECIPE3)                         # OpenOBB12: OpenOBB7 core, dual head
RECIPE13 = dict(RECIPE3)                         # OpenOBB13: OpenOBB12 + full-resolution stream + P2
# OpenOBB14: HGNetv2 trunk (COCO) + own re-parameterised neck + heavy dual head, direct box regression, edge refiner
RECIPE14 = {"rfs": 0.1, "channels_last": True, "clip": 10.0, "num_top": 1000, "box": "direct", "ref_k": 1500}
RECIPE15 = dict(RECIPE14)                        # OpenOBB15: v14 + full-resolution stream + P2 (imgsz = core size, input 2x)
RECIPE16 = dict(RECIPE15)                        # OpenOBB16: v15 + 3-stage cascade of refiners + fit-aware scores
RECIPE17 = dict(RECIPE12)                        # OpenOBB17: v12 + boundary field + boundary snapping
RECIPE18 = dict(RECIPE12)                        # OpenOBB18: v17 + own neck distilled from a trained encoder (teacher=)
RECIPE19 = dict(RECIPE18, groups="auto")         # OpenOBB19: v18 + one box per contention-aware class group
RECIPE20 = dict(RECIPE19)                        # OpenOBB20: v19 + edge refiner reading the trunk's stride-4 map
RECIPE23 = dict(RECIPE20)                        # OpenOBB23: v21 + rotational snapping + local-peak filter
RECIPE22 = dict(RECIPE20)                        # OpenOBB22: v21 + bounded angle from 45 deg + direct boxes
RECIPE21 = dict(RECIPE20)                        # OpenOBB21: v20 + double-angle residual angle head
SIZE_DEFAULTS15 = {s: dict(d, gb=round(d["gb"] * 0.5, 2)) for s, d in SIZE_DEFAULTS3.items()}
# OpenOBB13: imgsz is the full resolution, the core runs at half of it -> memory per image ~0.45x by area
SIZE_DEFAULTS13 = {s: dict(d, gb=round(d["gb"] * 0.45, 2)) for s, d in SIZE_DEFAULTS3.items()}
RECIPE11 = dict(RECIPE8)                         # OpenOBB11: OpenOBB8 recipe, NMS-free set decoder
RECIPE10 = dict(RECIPE8)                         # OpenOBB10: OpenOBB8 + slots / reach / balance
# OpenOBB9: new core pretrained by distillation (openobb pretrain), OpenOBB8's head; no COCO weights for the core
SIZE_DEFAULTS9 = {s: dict(lr=5e-4, backbone_mult=0.2, wd=1e-4, batch=b, gb=g)
                  for s, b, g in (("s", 16, 1.0), ("m", 16, 1.4), ("l", 8, 2.0), ("x", 8, 2.8))}
RECIPE9 = {"rfs": 0.1, "channels_last": True, "clip": 10.0, "num_top": 1000, "rel_k": 600}
# architecture flags a fine-tune inherits from its source checkpoint (weights only load into the same shape)
ARCH_KEYS = ("arch", "size", "p2", "lsk", "strip_k", "ortho_heads", "dense", "dense_queries", "denoising",
             "no_rotate_sampling", "context", "backbone", "dense_v3", "primary", "geo_cls", "relate", "rel_k",
             "v6_off", "v7_off", "v8_off", "v9_off", "v10_off", "box", "v14_off", "ref_k", "neck", "groups")
PRETRAIN_KEYS = ("pretrain", "pretrain_epochs", "pretrain_det_epochs", "pretrain_dir", "pretrain_teacher")
MIN_STEPS, MIN_STEPS_EPOCH = 2000, 25           # optimizer steps for a run / per epoch on small datasets
ALIASES = {"lr0": "lr", "imgsz": "img", "weight_decay": "wd", "val_period": "eval_every", "lrf": "min_lr_ratio",
           "iou": "nms_iou"}      # train: iou= (YOLO's val NMS IoU) = nms_iou, auto by default
IGNORED = ("save_period", "plots", "momentum", "degrees", "shear", "perspective", "mixup", "cutmix",
           "copy_paste", "rect", "multi_scale", "warmup_momentum", "dropout", "cls_pw")


def _value(v):
    if not isinstance(v, str):
        return v
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    if v.startswith("[") and v.endswith("]"):
        return [_value(x.strip().strip("'\"")) for x in v[1:-1].split(",") if x.strip()]
    return v


def parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected key=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip().replace("-", "_")] = _value(v.strip())
    return out


def _gpu_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:  # noqa: BLE001
        pass
    return None


def _set_device(device):
    if device is None or device == "":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(device).lower() == "cpu" else str(device)


def _is_prepared(p):
    p = Path(p)
    return p.is_dir() and (p / "classes.json").exists() and (p / "meta" / "train.jsonl").exists()


def _cache_root(cache):
    return Path(cache or os.environ.get("OPENOBB_CACHE") or Path.home() / ".cache" / "openobb" / "datasets")


def prepare_data(data, imgsz=1024, gap=200, val_frac=0.15, scale=1.0, cache_dir=None, workers=None, seed=0, fit=True):
    """data.yaml / dataset folder -> prepared cache dir (built once, reused). Prepared dirs pass through."""
    if _is_prepared(data):
        return Path(data)
    from openobb.data.prepare import prepare
    src = Path(data).resolve()
    key = hashlib.sha1(f"{src}|{imgsz}|{gap}|{val_frac}|{scale}|{seed}|{fit}".encode()).hexdigest()[:8]
    stem = src.parent.name if src.suffix in (".yaml", ".yml") else src.name
    out = _cache_root(cache_dir) / f"{stem}_{imgsz}_{key}"
    return prepare(src, out, size=imgsz, gap=gap, val_frac=val_frac, seed=seed, workers=workers, scale=scale,
                   fit=fit)


def increment_path(project, name, exist_ok=False):
    """runs/obb/train, runs/obb/train2, train3, ... (exist_ok=True reuses the folder)."""
    base = Path(project)
    d = base / name
    k = 2
    while d.exists() and not exist_ok:
        d = base / f"{name}{k}"
        k += 1
    return d


def _find_last(resume, model, project, name):
    """resume=True: resume=<last.pt> or model=<.../last.pt>; else the newest unfinished run among name, name2, ...
    (when name is given) or under project (like the usual trainer's latest-run lookup)."""
    import re
    if isinstance(resume, (str, Path)) and str(resume).endswith(".pt") and Path(resume).exists():
        return Path(resume)
    if isinstance(model, (str, Path)) and Path(model).name == "last.pt" and Path(model).exists():
        return Path(model)
    runs = [f for f in list(Path(project).glob("**/weights/last.pt")) + list(Path(project).glob("**/last.pt"))
            if not (_run_root(f) / "train_done").exists()]
    if name:
        pat = re.compile(rf"^{re.escape(name)}\d*$")
        runs = [f for f in runs if pat.match(_run_root(f).name)]
    if not runs:
        raise SystemExit(f"resume=True but no unfinished run (weights/last.pt) under {Path(project) / (name or '')}")
    return max(runs, key=lambda f: f.stat().st_mtime)


def _run_root(last):
    return last.parent.parent if last.parent.name == "weights" else last.parent


def _ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:  # noqa: BLE001
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return 16 * 2**30


def _n_lines(p):
    return sum(1 for _ in open(p)) if Path(p).exists() else 0


def _eval_every(prepared, epochs):
    """Validate every epoch unless the val split is large."""
    n = _n_lines(Path(prepared) / "meta" / "val.jsonl")
    return 1 if n <= 2000 else max(1, round(epochs / 10))


def _queries_for(prepared):
    """Decoder queries for training: 300 (D-FINE default) unless dense images need more (one-to-one matching cannot
    supervise more objects than queries; 300 saturated on dense DOTA tiles, research/LEDGER.md F4)."""
    import numpy as np
    counts = [_n_lines(p) for p in (Path(prepared) / "labels" / "train").glob("*.txt")]
    if not counts:
        return 300
    p99 = float(np.percentile(counts, 99))
    return 300 if p99 <= 240 else int(min(900, np.ceil(p99 * 1.5 / 100) * 100))


def page_stats(data, n=40):
    """(median image long side, p99 object long side, p5 object short side) in pixels from a sample of train images."""
    import random

    import numpy as np

    from openobb.data.imgio import imread
    from openobb.data.prepare import find_splits, label_path
    _, splits = find_splits(data)
    imgs = splits["train"]
    sides, longs, shorts = [], [], []
    for p in random.Random(0).sample(imgs, min(n, len(imgs))):
        im = imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        sides.append(max(h, w))
        lp = label_path(p)
        for row in (lp.read_text().splitlines() if lp.exists() else []):
            v = row.split()
            if len(v) >= 9:
                q = np.array([float(t) for t in v[1:9]]).reshape(4, 2) * [w, h]
                a, b = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
                longs.append(max(a, b))
                shorts.append(min(a, b))
    med = float(np.median(sides)) if sides else 0.0
    return (med, float(np.percentile(longs, 99)) if longs else 0.0, float(np.percentile(shorts, 5)) if shorts else 0.0)


def auto_scale(data, imgsz, fit=0.7):
    """Tile mode: the largest page scale <= 1 at which the 99th-percentile object still fits `fit` x tile (objects
    must be >= 70% inside one tile or they are lost); if the scaled page is then <= ~1.6 tiles, use one tile."""
    if _is_prepared(data):
        return 1.0
    med, big, _ = page_stats(data)
    if not med:
        return 1.0
    scale = min(1.0, fit * imgsz / big) if big > 0 else 1.0
    if med * scale <= 1.6 * imgsz:
        scale = min(1.0, imgsz / med)
    scale = math.floor(scale * 1e4) / 1e4
    how = "one tile per page" if med * scale <= imgsz else f"~{math.ceil((med * scale - 200) / (imgsz - 200)) ** 2} tiles per page"
    print(f"[openobb] tile mode: pages ~{med:.0f} px, objects up to ~{big:.0f} px (p99) -> tile_scale {scale}, {how}")
    return scale


def _ckpt_args(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("args", {}) or {}, list(ck.get("classes") or [])


def _to_argv(opts):
    argv = []
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


RESUME_OK = ("workers", "cache", "patience", "eval_every", "verbose", "time")   # may change when resuming
RUN_FILES = ("weights/last.pt", "weights/best.pt", "last.pt", "best.pt", "train_done", "results.csv", "metrics.jsonl", "train_progress.log", "eval_val.txt",
             "eval_val.json", "exitcode")


def _known_keys():
    import inspect

    from openobb.train import build_parser
    keys = set(inspect.signature(train).parameters) - {"extra"}
    keys |= set(ALIASES) | set(IGNORED) | set(AUGMENT) | {"hsv_h", "hsv_s", "hsv_v", "amp", "rot90"}
    keys |= {a.dest for a in build_parser()._actions}
    return keys


def check_keys(kv):
    """Unknown argument -> error with the closest valid names (typos must not pass silently)."""
    import difflib
    known = _known_keys()
    for k in kv:
        if k not in known:
            near = difflib.get_close_matches(k, sorted(known), n=3, cutoff=0.6)
            raise SystemExit(f"'{k}' is not a valid OpenOBB argument." + (f" Similar: {', '.join(near)}" if near else ""))


def _auto_recipe(extra, data_classes, src_classes, n_train, batch, epochs, base_lr, nbs=64):
    """optimizer=auto (the default): settings the user did not pass, chosen from the data and the starting weights, as
    the usual trainer picks its optimizer / LR from the number of iterations and classes (F98).
      lr   classes the starting weights do not know (no weights, or a checkpoint of other classes): their output rows
           start from scratch -> lr0 = max(default, 0.002 * 5 / (4 + nc)), the AdamW fit of the usual trainer
           (1 class: 2e-3, 7: 9.1e-4, 15: 5.3e-4). Same classes (continuing a run): the default.
      nbs  fewer than MIN_STEPS optimizer steps with gradient accumulation to a nominal batch of 64 -> nbs=0 (one step
           per batch: 4x the steps at batch 16 on a small dataset).
    Mutates `extra`; returns one message per change."""
    msgs = []
    if "nbs" not in extra:
        acc = max(round(nbs / batch), 1)
        steps = max(1, n_train // batch) * epochs // acc
        if acc > 1 and steps < MIN_STEPS:
            extra["nbs"] = 0
            msgs.append(f"nbs=0 (small dataset: {steps * acc} optimizer steps instead of {steps}; pass nbs=64 to keep "
                        f"the nominal batch)")
    if "lr" not in extra:
        known = set(src_classes or [])
        new = [c for c in data_classes if c not in known]
        fit = round(0.002 * 5 / (4 + max(len(data_classes), 1)), 6)
        if new and fit > base_lr:
            extra["lr"] = fit
            what = "no starting weights" if src_classes is None else f"new classes {new[:5]}{'...' if len(new) > 5 else ''}"
            msgs.append(f"lr0={fit:g} ({what}: output rows start from scratch; default {base_lr:g} for the same classes; "
                        f"pass lr0= or optimizer=AdamW to keep the default)")
    return msgs


def _batch_from(batch, sd, imgsz, n_train):
    """batch=None/-1: default fitted to the GPU and to the dataset size; 0 < batch < 1: that fraction of VRAM."""
    frac = float(batch) if batch is not None and 0 < float(batch) < 1 else None
    if batch is not None and frac is None and int(batch) > 0:
        return int(batch)
    batch, why = sd["batch"], "default"
    mem = _gpu_mem_gb()
    if mem:                                          # measured ~GB per image at 1024 px, scaled by area
        per = sd["gb"] * (imgsz / 1024) ** 2
        fitb = int(((mem * frac) if frac else (mem - 2.0)) / per) // 2 * 2
        if frac or fitb < batch:
            batch, why = max(2, fitb), f"{int(frac * 100)}% of {mem:.0f} GB" if frac else f"GPU {mem:.0f} GB"
    small = 2 ** int(math.log2(max(4, n_train // MIN_STEPS_EPOCH)))   # small data: >= ~25 optimizer steps / epoch
    if small < batch:
        batch, why = max(4, small), f"{n_train} training images"
    print(f"[openobb] batch {batch} ({why}; set batch= to override)")
    return batch


def _run_trainer(save_dir, prepared, opts, n_train, epochs, warmup_epochs, user_ema):
    """Run the trainer; on CUDA out-of-memory before the first epoch is saved, halve the batch and retry."""
    from openobb import train as trainer
    opts["wdir"] = "weights"
    for attempt in range(4):
        ipe = max(1, n_train // int(opts["batch"]))
        nbs = int(opts.get("nbs") or 0)
        acc = max(round(nbs / int(opts["batch"])), 1) if nbs > 0 else 1
        steps_ep = max(1, ipe // acc)                # optimizer steps per epoch
        total = steps_ep * int(epochs)
        if opts.get("epoch_lr"):                     # usual one-stage trainer (current): round(min(warmup_epochs, epochs - 1)
            # x batches), no 100-batch floor any more (the floor stretched the warmup over most of a small run, F98)
            w_ep = min(warmup_epochs, max(int(epochs) - 1, 0))
            opts["warmup"] = max(round(w_ep * ipe), 1) if w_ep > 0 else -1
        else:                                        # older runs: warmup_epochs epochs (at most epochs - 1)
            opts["warmup"] = max(1, round(min(warmup_epochs, max(int(epochs) - 1, 0)) * ipe))
        opts["ema_warmups"] = max(1, round(max(opts["warmup"], 1) / acc))
        if not user_ema:                             # EMA half-life ~5% of the run (>= 1 epoch, >= 50 steps)
            opts["ema"] = round(min(0.9998, 0.5 ** (1 / max(steps_ep, total // 20, 50))), 6)
        argv = ["--data", str(prepared), "--out", str(save_dir)] + _to_argv(opts)
        args_file = save_dir / "openobb_args.json"
        saved = json.loads(args_file.read_text()) if args_file.exists() else {}
        args_file.write_text(json.dumps({**saved, "prepared": str(prepared), **opts}, indent=1, default=str))
        if opts.get("verbose"):
            print("[openobb] python -m openobb.train " + " ".join(argv))
        try:
            trainer.main(argv)
            return
        except RuntimeError as e:                    # torch.OutOfMemoryError is a RuntimeError
            if ("out of memory" not in str(e).lower() or int(opts["batch"]) <= 2 or attempt == 3
                    or (save_dir / "weights" / "last.pt").exists()):
                raise
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        old = int(opts["batch"])
        opts["batch"] = max(2, old // 2)
        print(f"[openobb] WARNING: CUDA out of memory with batch={old}. Reducing to batch={opts['batch']} and retrying.")


def train(data=None, model="openobb1s", epochs=100, batch=None, imgsz=None, project="runs/obb", name=None,
          exist_ok=False, resume=False, tile=False, tile_scale=None, gap=200, val_frac=0.15, cache="auto", cache_dir=None, workers=None,
          device=None, recipe=True, warmup_epochs=3, patience=100, lrf=0.01, cos_lr=True, seed=0, optimizer="auto",
          **extra):
    """Train (or fine-tune when `model` is a .pt path). Returns SimpleNamespace(save_dir, best, last, metrics).
    pretrain=coco|dota|<image folders> (new cores openobb6* / openobb9*): download, pretrain and cache the starting
    weights first (openobb/auto_pretrain.py); pretrain_epochs=, pretrain_det_epochs= (DOTA stage), pretrain_dir=,
    pretrain_teacher= (an OpenOBB .pt core as distillation teacher) tune it."""
    pre = {k: extra.pop(k) for k in PRETRAIN_KEYS if k in extra}
    check_keys(extra)
    _set_device(device)
    for k in list(extra):
        if k in ALIASES:
            extra[ALIASES[k]] = extra.pop(k)
        elif k in IGNORED:
            extra.pop(k)
            print(f"[openobb] '{k}' is not used by OpenOBB (ignored)")
    if extra.pop("amp", True) is False:
        extra["no_amp"] = True
    if extra.pop("rot90", True) is False:
        extra["no_rot90"] = True
    if batch is not None and float(batch) <= 0:
        batch = None                                 # batch=-1: automatic
    user_ema = "ema" in extra

    # ---- resume=True only (like the usual trainer): continue a run with its saved settings
    if resume:
        last = _find_last(resume, model, project, name)
        save_dir = _run_root(last)
        saved_file = save_dir / "openobb_args.json"
        if not saved_file.exists():
            raise SystemExit(f"{save_dir} has no openobb_args.json: cannot resume it")
        saved = json.loads(saved_file.read_text())
        prepared = Path(saved.get("prepared", ""))
        if not _is_prepared(prepared):               # new machine / VM: rebuild the same prepared data
            prepared = prepare_data(saved.get("data", data), saved["img"], gap, val_frac, saved.get("scale", 1.0),
                                    cache_dir, workers, seed, fit=saved.get("fit", True))
        opts = {k: v for k, v in saved.items() if k not in ("data", "prepared", "warmup", "ema_warmups")}
        given = {"batch": batch, "imgsz": imgsz, **extra, **({"epochs": epochs} if epochs != 100 else {})}
        ignored = [k for k, v in given.items() if v is not None and k not in RESUME_OK
                   and str(opts.get(ALIASES.get(k, k), v)) != str(v)]
        print(f"[openobb] resuming {save_dir} with its saved settings"
              + (f" (ignores {ignored})" if ignored else ""))
        opts.update({k: extra[k] for k in RESUME_OK if k in extra})
        if workers is not None:
            opts["workers"] = int(workers)
        n_train = _n_lines(Path(prepared) / "meta" / "train.jsonl")
        _run_trainer(save_dir, prepared, opts, n_train, opts["epochs"], warmup_epochs, user_ema=True)
        return _summary(save_dir)
    if data is None:
        raise SystemExit("openobb train needs data=<data.yaml or dataset folder>")
    save_dir = increment_path(project, name or "train", exist_ok)   # new run: train, train2, ... (never resumes)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_file = save_dir / "openobb_args.json"
    for f in RUN_FILES:                              # exist_ok=True on an old folder: start over in place
        (save_dir / f).unlink(missing_ok=True)

    aug = {k: extra.pop(k, v) for k, v in AUGMENT.items()}
    hsv = [extra.pop(k, d) for k, d in (("hsv_h", 0.015), ("hsv_s", 0.5), ("hsv_v", 0.3))]
    weights, src_classes = None, None
    m = str(model)
    if m.lower() in MODELS:
        arch, size = MODELS[m.lower()]
        inherited = {"arch": arch} if arch != "v1" else {}
    elif m.endswith(".pt") and Path(m).exists():
        weights = m
        src_args, src_classes = _ckpt_args(m)
        inherited = {k: src_args[k] for k in ARCH_KEYS if k in src_args}
        size = inherited.pop("size", "s")
        imgsz = imgsz or extra.pop("img", None) or src_args.get("img", 1024)
    else:
        raise SystemExit(f"model must be one of {', '.join(MODELS)} or an existing OpenOBB .pt checkpoint, "
                         f"got '{model}'")
    arch = inherited.get("arch", "v1")
    v2, dense_arch = arch == "v2", arch in ("v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23")
    src_img = extra.pop("img", None)                 # a checkpoint's own input size (already the full size for v13)
    core_size = imgsz is not None or src_img is None
    imgsz = int(imgsz or src_img or 1024)
    if imgsz % 32:
        imgsz = int(math.ceil(imgsz / 32) * 32)
        print(f"[openobb] imgsz must be a multiple of 32: using {imgsz}")
    if arch in ("v13", "v15", "v16") and core_size:
        from openobb.predict import input_size
        imgsz = input_size({"arch": arch}, imgsz)
        print(f"[openobb] OpenOBB{arch[1:]}: imgsz {imgsz // 2} is the core's size; images are prepared at {imgsz} px for "
              f"the full-resolution detail stream")

    fit = not tile and tile_scale is None
    if fit:
        med, big, small = page_stats(data) if not _is_prepared(data) else (0, 0, 0)
        if med:
            r = imgsz / med
            print(f"[openobb] images ~{med:.0f} px (median long side) -> resized x{r:.2f} to long side {imgsz} px")
            if small and small * r < 3:
                print(f"[openobb] WARNING: small objects (~{small:.0f} px) become ~{small * r:.1f} px at this size; "
                      f"use a larger imgsz or tile=True")
        scale = 1.0
    else:
        scale = auto_scale(data, imgsz) if tile_scale in (None, "auto") else float(tile_scale)

    sd = {"v2": SIZE_DEFAULTS2, "v3": SIZE_DEFAULTS3, "v4": SIZE_DEFAULTS4,
          "v5": SIZE_DEFAULTS5, "v6": SIZE_DEFAULTS6, "v7": SIZE_DEFAULTS3,
          "v8": SIZE_DEFAULTS3, "v9": SIZE_DEFAULTS9, "v10": SIZE_DEFAULTS3, "v11": SIZE_DEFAULTS3, "v12": SIZE_DEFAULTS3, "v13": SIZE_DEFAULTS13,
          "v14": SIZE_DEFAULTS3, "v15": SIZE_DEFAULTS15, "v16": SIZE_DEFAULTS15,
          "v17": SIZE_DEFAULTS3, "v18": SIZE_DEFAULTS3, "v19": SIZE_DEFAULTS3, "v20": SIZE_DEFAULTS3, "v21": SIZE_DEFAULTS3, "v22": SIZE_DEFAULTS3, "v23": SIZE_DEFAULTS3}.get(arch, SIZE_DEFAULTS)[size]
    if str(optimizer).lower() not in ("auto", "adamw"):
        raise SystemExit("optimizer must be 'auto' or 'AdamW'")   # lr0 given -> used; otherwise the measured LR
    prepared = prepare_data(data, imgsz, gap, val_frac, scale, cache_dir, workers, seed, fit=fit)
    if pre.get("pretrain"):
        if arch not in ("v6", "v9"):
            raise SystemExit(f"pretrain= is for the new cores without public weights (openobb6*, openobb9*); "
                             f"{model} already starts from pretrained COCO weights")
        if weights or extra.get("weights"):
            raise SystemExit("pretrain= builds the starting weights itself: use either pretrain= or weights=")
        from openobb.auto_pretrain import ARCH_SWITCHES, auto_pretrain
        extra["weights"] = auto_pretrain(
            f"openobb{arch[1:]}{size}", pre["pretrain"], Path(prepared) / "images" / "train",
            Path(prepared) / "images" / "val", imgsz=imgsz, epochs=pre.get("pretrain_epochs"),
            det_epochs=pre.get("pretrain_det_epochs"), root=pre.get("pretrain_dir"), teacher=pre.get("pretrain_teacher"),
            device=device, workers=int(workers) if workers is not None else 8,
            switches={k: extra.get(k) for k in ARCH_SWITCHES})
    n_train = _n_lines(Path(prepared) / "meta" / "train.jsonl")
    if cache == "auto":                              # decoded uint8 images in RAM when they take < 25% of it
        cache = n_train * imgsz * imgsz * 3 < 0.25 * _ram_bytes()
    batch = _batch_from(batch, sd, imgsz, n_train)
    if str(optimizer).lower() == "auto" and dense_arch:   # DETR-style OpenOBB1 / 4: LR above ~2e-4 diverges
        if src_classes is None and extra.get("weights") and Path(str(extra["weights"])).exists():
            src_classes = _ckpt_args(extra["weights"])[1]
        from openobb.data.dota import dataset_classes
        for msg in _auto_recipe(extra, dataset_classes(prepared), src_classes, n_train, int(batch), int(epochs),
                                sd["lr"]):
            print(f"[openobb] auto: {msg}")

    nbs_ = int(extra.get("nbs", 64) or 0)
    acc_ = max(round(nbs_ / int(batch)), 1) if nbs_ > 0 else 1        # gradient accumulation (nominal batch)
    ipe_ = max(1, n_train // int(batch))
    steps = ipe_ * int(epochs) // acc_
    if steps < MIN_STEPS and not extra.get("time"):
        acc_txt = f", 1 update per {acc_} batches (nbs={nbs_})" if acc_ > 1 else ""
        print(f"[openobb] WARNING: only {steps} optimizer steps ({ipe_} batches/epoch x {epochs} epochs{acc_txt}); this "
              f"detector needs ~{MIN_STEPS}+ to converge: raise epochs, lower batch"
              f"{' or pass nbs=0' if acc_ > 1 else ''}")
    opts = {"size": size, "img": imgsz, "scale": scale, "fit": fit, "epochs": int(epochs), "batch": int(batch),
            "aug_iof": 0.25, "merge_iou": 0.7 if fit else 0.1, "cache": bool(cache),
            "schedule": "cos" if cos_lr else "linear", "min_lr_ratio": float(lrf),
            "lr": sd["lr"], "backbone_mult": sd["backbone_mult"], "wd": sd["wd"], "hsv": hsv,
            "eval_every": _eval_every(prepared, int(epochs)), "eval_images": 10**9, "patience": int(patience or 0),
            "seed": seed,
            # usual one-stage trainer: LR factor per epoch, warmup over 3 epochs of batches, nominal batch 64
            # (gradient accumulation + weight decay scaled to it). Bias LR warmup off: the usual trainer zeroes
            # warmup_bias_lr whenever it picks AdamW (0.1 is for SGD); under AdamW it moved every bias by ~0.1 per step
            # and dragged fine-tunes down until the warmup ended (F98)
            "epoch_lr": True, "warmup_bias_lr": 0.0, "nbs": 64}
    q = _queries_for(prepared)
    if q > 300 and not dense_arch:
        opts.update(queries=q, num_top=q)
        print(f"[openobb] dense images (p99 objects/image > 240): {q} queries")
    if workers is not None:
        opts["workers"] = int(workers)
    if recipe:
        opts.update({"v2": RECIPE2, "v3": RECIPE3, "v4": RECIPE4, "v5": RECIPE5, "v6": RECIPE6, "v7": RECIPE7,
                     "v8": RECIPE8, "v9": RECIPE9, "v10": RECIPE10, "v11": RECIPE11, "v12": RECIPE12, "v13": RECIPE13,
                     "v14": RECIPE14, "v15": RECIPE15, "v16": RECIPE16,
                     "v17": RECIPE17, "v18": RECIPE18, "v19": RECIPE19, "v20": RECIPE20, "v21": RECIPE21, "v22": RECIPE22, "v23": RECIPE23}.get(arch, RECIPE))
    if arch in ("v4", "v5") and "backbone" not in inherited:
        opts["backbone"] = BACKBONE4[size]
    if aug["mosaic"]:
        opts["mosaic_mode"] = "yolo"
    opts.update({AUG_FLAGS[k]: v for k, v in aug.items()})
    opts.update(inherited)
    if weights:
        opts["weights"] = weights
    if extra.get("v6_off"):                          # OpenOBB6 ablation: v6_off=strip,attn or v6_off=[strip,attn]
        from openobb.models.openobb6 import parse_off
        extra["v6_off"] = ",".join(sorted(parse_off(extra["v6_off"])))
    for k in ("v7_off", "v8_off", "v9_off", "v10_off", "v14_off"):   # OpenOBB7 / 8 / 9 / 10 / 14 ablation switches
        if extra.get(k):
            v = extra[k]
            extra[k] = ",".join(sorted({s.strip() for s in (v.split(",") if isinstance(v, str) else v) if str(s).strip()}))
    opts.update(extra)
    if arch in ("v18", "v19", "v20", "v21", "v22", "v23") and not resume:
        # OpenOBB18 is meant to start from a trained OpenOBB12 / 17 (trunk + head) and to distil its new neck from that
        # model's encoder; without both, the neck learns from scratch (F70: 0.623 / 0.369 instead of ~v12)
        src = opts.get("weights")
        src_arch = _ckpt_args(src)[0].get("arch") if src and Path(str(src)).exists() else None
        if src and not opts.get("teacher") and src_arch in ("v12", "v17"):
            opts["teacher"] = src            # a v18 checkpoint (e.g. pretrained on drawings) already has its own neck
            print(f"[openobb] OpenOBB18: teacher= not given, distilling the neck from weights={src}")
        if not src:
            print("[openobb] WARNING: OpenOBB18 without weights= starts its neck from scratch (much lower accuracy, F70). "
                  "Use weights=<trained openobb12x / 17x .pt> (it is also used as teacher=).")
    if float(opts["lr"]) > 3 * sd["lr"] and not dense_arch:
        print(f"[openobb] WARNING: lr0={opts['lr']:g} is {float(opts['lr']) / sd['lr']:.0f}x the default for OpenOBB1-{size} "
              f"({sd['lr']:g}). DETR-style detectors usually diverge (NaN boxes) above ~2e-4; the 1e-3 of one-stage "
              f"detectors does not transfer.")
    saved_file.write_text(json.dumps({"data": str(data)}, indent=1))
    _run_trainer(save_dir, prepared, opts, n_train, epochs, warmup_epochs, user_ema)
    return _summary(save_dir)


def _summary(save_dir):
    save_dir = Path(save_dir)
    metrics = {}
    for f in ("eval_val_q900.json", "eval_val.json"):
        p = save_dir / f
        if p.exists():
            r = json.loads(p.read_text())
            metrics = {"mAP50": r.get("mAP50"), "mAP50_95": r.get("mAP50_95"), "file": f}
            break
    w = save_dir / "weights" if (save_dir / "weights").exists() else save_dir
    return SimpleNamespace(save_dir=str(save_dir), best=str(w / "best.pt"), last=str(w / "last.pt"), metrics=metrics)


def val(model, data=None, imgsz=None, tile_scale=None, gap=200, val_frac=0.15, batch=8, workers=8, queries=900,
        cache_dir=None, device=None, seed=0, metric="yolo", nms_min_side=0.0, ref_k=None):
    """DOTA-protocol mAP of a checkpoint on the val split of `data` (same hold-out and resizing as training).
    Without `data`, the checkpoint's own prepared data is used when it is available."""
    _set_device(device)
    import torch

    from openobb.data.dota import dataset_classes
    from openobb.engine import eval_dota
    from openobb.eval.dota import summary_table
    from openobb.predict import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, names, targs = load_model(model, dev, queries)
    net.nms_min_side = float(nms_min_side)             # thickness floor for the NMS overlap test (fast_nms)
    if ref_k is not None and hasattr(net, "ref_k"):    # OpenOBB14: edge refiner budget at val (0 = off), no retrain
        net.ref_k = int(ref_k)
    from openobb.predict import input_size
    imgsz = input_size(targs, imgsz)
    fit = bool(targs.get("fit", False)) and tile_scale is None
    scale = float(tile_scale if tile_scale is not None else targs.get("scale", 1.0))
    if data is None:
        if not _is_prepared(targs.get("data", "")):
            raise SystemExit("val needs data=<data.yaml> (the checkpoint's prepared data is not on this machine)")
        prepared = Path(targs["data"])
    else:
        prepared = prepare_data(data, imgsz, gap, val_frac, scale, cache_dir, workers, targs.get("seed", seed),
                                fit=fit)
    classes = list(dataset_classes(prepared))
    if classes != list(names):                       # class ids are positional: a different order means wrong mAP
        raise SystemExit(f"dataset classes {classes} do not match the model's {list(names)} (same names, same order)")
    from openobb.predict import merge_iou_for
    res, _ = eval_dota(net, prepared, dev, None, batch=batch, workers=workers, num_top=queries, img_size=imgsz,
                       merge_iou=merge_iou_for(targs.get("nms_iou"), fit),
                       post=targs.get("post", "flat"), primary=targs.get("primary", "dec"),
                       fusion=targs.get("primary", "dec") != "dec", ultra=True, metric=metric)
    print(summary_table(res, classes))
    from openobb.eval.ultra import duplicate_line, top_confusions
    from openobb.speed import speed_line
    for line in top_confusions(res["ultra"]["confusion"], k=10):
        print("  confusion  " + line)
    if duplicate_line(res["ultra"]["confusion"]):
        print(duplicate_line(res["ultra"]["confusion"]))
    from openobb.eval.ultra import geometry_line
    if geometry_line(res["ultra"].get("geometry")):
        print(geometry_line(res["ultra"]["geometry"]))
    if res.get("speed"):
        print(speed_line(res["speed"]) + f" ({res['speed']['images']} images after a warm-up batch, fused, batch {batch})")
    if res.get("metric") != "yolo":
        u = res["ultra"]
        print(f"Ultralytics-style metric (ProbIoU matching, 101-point AP): mAP50 {u['mAP50']:.3f}  "
              f"mAP50-95 {u['mAP50_95']:.3f}")
    return res


def calibrate(model, data=None, out=None, imgsz=None, batch=8, workers=8, cache_dir=None, device=None, seed=0,
              val_frac=0.15, gap=200):
    """Re-run the end-of-training calibration on val for a trained checkpoint: the inference head of dual-head models
    (one-to-one / one-to-many + NMS) and the NMS threshold, by fitness 0.1 mAP50 + 0.9 mAP50-95 (within 0.01 of the
    best, the setting with the fewest duplicate boxes). Writes the choice
    into the checkpoint's args (in place, or to `out`); everything else is unchanged."""
    _set_device(device)
    import torch

    from openobb.data.dota import dataset_classes
    from openobb.engine import eval_dota
    from openobb import __version__
    from openobb.predict import (DUP_TOL, apply_head, apply_nms, choose_postprocess, dup_rate, input_size, load_model,
                                 merge_iou_for, postprocess_options)
    print(f"[calibrate] OpenOBB {__version__}: best fitness (0.1 mAP50 + 0.9 mAP50-95); within {DUP_TOL:g} of it, the "
          f"setting with the fewest duplicate boxes", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, names, targs = load_model(model, dev)
    imgsz = input_size(targs, imgsz)
    fit = bool(targs.get("fit", False))
    if data is None:
        if not _is_prepared(targs.get("data", "")):
            raise SystemExit("calibrate needs data=<data.yaml> (the checkpoint's prepared data is not on this machine)")
        prepared = Path(targs["data"])
    else:
        prepared = prepare_data(data, imgsz, gap, val_frac, float(targs.get("scale", 1.0)), cache_dir, workers,
                                targs.get("seed", seed), fit=fit)
    if list(dataset_classes(prepared)) != list(names):
        raise SystemExit(f"dataset classes {list(dataset_classes(prepared))} do not match the model's {list(names)}")
    tried = []
    for h, c in postprocess_options(net):
        apply_head(net, h)
        apply_nms(net, c)
        res, _ = eval_dota(net, prepared, dev, None, batch=batch, workers=workers, img_size=imgsz,
                           num_top=int(targs.get("num_top", 1000)),
                           merge_iou=merge_iou_for(c, fit), ultra=True, metric="yolo")
        u = res.get("ultra") or res
        f = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
        q = dup_rate(u)
        tried.append((f, q, h, c))
        print(f"[calibrate] {(h + ' ') if h else ''}nms {c}: mAP50 {u['mAP50']:.4f}  mAP50-95 {u['mAP50_95']:.4f}  "
              f"fitness {f:.4f}" + ("" if q is None else f"  duplicates {100 * q:.1f}%"), flush=True)
    f, _, h, c = choose_postprocess(tried)            # best fitness; within DUP_TOL, the fewest duplicates
    ck = torch.load(model, map_location="cpu", weights_only=False)
    ck["args"] = dict(ck.get("args", {}), nms_iou=c, **({"infer_head": h} if h else {}))
    dst = Path(out or model)
    torch.save(ck, dst.with_suffix(".tmp"))
    os.replace(dst.with_suffix(".tmp"), dst)
    print(f"[calibrate] -> {(h + ' ') if h else ''}nms {c} (fitness {f:.4f}) saved in {dst}")
    return SimpleNamespace(infer_head=h, nms_iou=c, fitness=f, path=str(dst))


def predict(model, source, conf=0.25, save_dir=None, vis=True, imgsz=None, tile=None, tile_scale=None,
            gap=200, queries=900, batch=8, device=None, classes=None, names=None, project="runs/obb", name="predict",
            exist_ok=False, save=True):
    """Detections per image {path: [{"class", "class_id", "score", "poly" (8 pixel coords)}]}; files in save_dir.
    Images are resized exactly as in training; tile=True forces tiling at native resolution (tile_scale).
    conf="auto" uses the per-class thresholds (best F2 on val) stored in best.pt; classes= keeps only these class
    ids or names; names= supplies class names for checkpoints that lack them."""
    _set_device(device)
    from openobb.predict import run
    if save_dir is None and save:
        save_dir = increment_path(project, name, exist_ok)
        print(f"[openobb] results -> {save_dir}")
    return run(model, source, save_dir, conf, names, imgsz, gap, queries, batch, bool(vis) and save_dir is not None,
               scale=tile_scale, tile=True if tile_scale is not None else tile, classes=classes)


class Detector:
    """Convenience wrapper: Detector("s" | "m" | "l" | "x" | "path/to/best.pt")."""

    def __init__(self, model="openobb1s"):
        self.model = str(model)

    def train(self, data, **kw):
        r = train(data, model=self.model, **kw)
        if Path(r.best).exists():
            self.model = r.best
        return r

    def val(self, data, **kw):
        return val(self.model, data, **kw)

    def predict(self, source, **kw):
        return predict(self.model, source, **kw)

    def __call__(self, source, **kw):
        return self.predict(source, **kw)


LEGACY_PREDICT = {"save_dir", "vis", "tile_scale", "gap", "queries", "names"}


def _predict_cli(kv):
    """`openobb predict`: OpenOBB(model).predict(...) with the YOLO CLI defaults (save=True). The old arguments
    (save_dir, vis, tile_scale, gap, queries, names) still run the previous writer (txt + json per image)."""
    if "model" not in kv or "source" not in kv:
        raise SystemExit("openobb predict needs model=<best.pt> and source=<image, folder, glob or URL>")
    if LEGACY_PREDICT & set(kv):
        predict(**kv)
        return 0
    import inspect

    from openobb.model import UNSUPPORTED, OpenOBB
    ok = set(inspect.signature(OpenOBB.predict).parameters) - {"self", "kw"} | UNSUPPORTED
    bad = [k for k in kv if k not in ok and k != "model"]
    if bad:
        import difflib
        near = difflib.get_close_matches(bad[0], sorted(ok), n=3, cutoff=0.6)
        raise SystemExit(f"'{bad[0]}' is not a valid argument for openobb predict."
                         + (f" Similar: {', '.join(near)}" if near else ""))
    if isinstance(kv.get("classes"), str):                      # classes=0,2 or classes=door,window
        kv["classes"] = [_value(c.strip()) for c in kv["classes"].split(",") if c.strip()]
    kv.setdefault("save", True)
    model = OpenOBB(kv.pop("model"))
    for _ in model.predict(stream=True, **kv):
        pass
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help") or argv[0] not in MODES:
        print(__doc__)
        return 0 if (not argv or argv[0] in ("-h", "--help", "help")) else 2
    mode, kv = argv[0], parse_kv(argv[1:])
    from openobb.torch_setup import ensure_torch        # CUDA PyTorch when an NVIDIA GPU is present
    ensure_torch()
    if mode == "predict":
        return _predict_cli(kv)
    if mode == "strip":
        from openobb.strip import strip
        bad = [k for k in kv if k not in ("model", "out", "half")]
        if bad or "model" not in kv:
            raise SystemExit("usage: openobb strip model=<best.pt> [out=<deploy.pt>] [half=True]")
        strip(**kv)
        return 0
    if mode == "profile":
        from openobb.profile import profile_parts
        flags = ("detail", "channels_last", "compile")
        if "model" not in kv or any(k not in ("model", "imgsz", "batch", "runs") + flags for k in kv):
            raise SystemExit("usage: openobb profile model=<best.pt> [imgsz=1280] [batch=16] [runs=20] [detail=True] "
                             "[channels_last=True] [compile=True]  (GPU)")
        on = {f: str(kv.pop(f, "False")).lower() in ("1", "true", "yes") for f in flags}
        profile_parts(kv["model"], **on, **{k: int(v) for k, v in kv.items() if k != "model"})
        return 0
    if mode == "calibrate":
        import inspect
        bad = [k for k in kv if k not in inspect.signature(calibrate).parameters]
        if bad or "model" not in kv:
            raise SystemExit("usage: openobb calibrate model=<best.pt> data=<data.yaml> [out=<file.pt>]")
        calibrate(**kv)
        return 0
    if mode == "pretrain":
        import inspect

        from openobb.pretrain import pretrain
        bad = [k for k in kv if k not in inspect.signature(pretrain).parameters]
        if bad:
            raise SystemExit(f"'{bad[0]}' is not a valid argument for openobb pretrain "
                             f"({', '.join(inspect.signature(pretrain).parameters)})")
        pretrain(**kv)
        return 0
    if mode == "val":
        import inspect
        fn = val
        ok = set(inspect.signature(fn).parameters) | {"save"}
        bad = [k for k in kv if k not in ok]
        if bad:
            import difflib
            near = difflib.get_close_matches(bad[0], sorted(ok), n=3, cutoff=0.6)
            raise SystemExit(f"'{bad[0]}' is not a valid argument for openobb {mode}."
                             + (f" Similar: {', '.join(near)}" if near else ""))
    if mode == "train":
        if "data" not in kv and not kv.get("resume"):
            raise SystemExit("openobb train needs data=<data.yaml or dataset folder> (or resume=True)")
        train(**kv)
    elif mode == "val":
        val(**kv)
    elif mode == "prepare":
        out = kv.pop("out", None)
        tile = bool(kv.get("tile", False))
        if out:
            from openobb.data.prepare import prepare
            prepare(kv["data"], out, size=int(kv.get("imgsz", 1024)), gap=int(kv.get("gap", 200)),
                    val_frac=float(kv.get("val_frac", 0.15)), seed=int(kv.get("seed", 0)),
                    workers=kv.get("workers"), scale=float(kv.get("tile_scale", 1.0)), fit=not tile)
            print(f"[openobb] prepared -> {out}")
        else:
            print(f"[openobb] prepared -> {prepare_data(kv['data'], int(kv.get('imgsz', 1024)), fit=not tile)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
