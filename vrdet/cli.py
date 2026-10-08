"""VRDet command line and Python API. Arguments are key=value pairs.

    vrdet train   data=data.yaml model=s epochs=100 imgsz=1024 project=runs name=exp
    vrdet train   data=data.yaml model=runs/exp/best.pt epochs=50            # fine-tune from a VRDet checkpoint
    vrdet val     model=runs/exp/best.pt data=data.yaml
    vrdet predict model=runs/exp/best.pt source=pages/ conf=0.25 save_dir=preds
    vrdet prepare data=data.yaml out=datasets/mydata imgsz=1024

    from vrdet import Detector
    model = Detector("s")                       # or Detector("runs/exp/best.pt")
    r = model.train(data="data.yaml", epochs=100, imgsz=1024, project="runs", name="exp")
    model.predict("pages/", conf=0.25, save_dir="preds")

Datasets: data.yaml ('names' + train/val image folders) with labels 'class x1 y1 x2 y2 x3 y3 x4 y4' normalised to
[0, 1] in a sibling 'labels' folder. Images of any size work: each image is resized so its long side = imgsz
(training, validation and prediction alike). tile=True instead cuts large pages into imgsz tiles at native
resolution (tile_scale= to resize first), for very large pages with tiny objects. Without a val split, val_frac of
the train images is held out. Prepared data is cached (~/.cache/vrdet). A run resumes from {project}/{name}/last.pt.

Training defaults follow the usual one-stage recipe: mosaic=1.0, scale=0.5 (random zoom 0.5-1.5), translate=0.1,
close_mosaic=10, warmup_epochs=3, LR decayed linearly to lr0 * lrf (lrf=0.01; cos_lr=True for cosine), patience=100,
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

MODES = ("train", "val", "predict", "prepare")
SIZES = ("s", "m", "l", "x")

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
# architecture flags a fine-tune inherits from its source checkpoint (weights only load into the same shape)
ARCH_KEYS = ("size", "p2", "lsk", "strip_k", "ortho_heads", "dense", "dense_queries", "denoising",
             "no_rotate_sampling", "context")
MIN_STEPS, MIN_STEPS_EPOCH = 2000, 25           # optimizer steps for a run / per epoch on small datasets
ALIASES = {"lr0": "lr", "imgsz": "img", "weight_decay": "wd", "val_period": "eval_every", "lrf": "min_lr_ratio"}
IGNORED = ("save_period", "plots", "momentum", "degrees", "shear", "perspective", "mixup", "cutmix",
           "copy_paste", "rect", "multi_scale", "warmup_bias_lr", "warmup_momentum", "nbs", "dropout", "cls_pw")


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
    return Path(cache or os.environ.get("VRDET_CACHE") or Path.home() / ".cache" / "vrdet" / "datasets")


def prepare_data(data, imgsz=1024, gap=200, val_frac=0.15, scale=1.0, cache_dir=None, workers=None, seed=0, fit=True):
    """data.yaml / dataset folder -> prepared cache dir (built once, reused). Prepared dirs pass through."""
    if _is_prepared(data):
        return Path(data)
    from vrdet.data.prepare import prepare
    src = Path(data).resolve()
    key = hashlib.sha1(f"{src}|{imgsz}|{gap}|{val_frac}|{scale}|{seed}|{fit}".encode()).hexdigest()[:8]
    stem = src.parent.name if src.suffix in (".yaml", ".yml") else src.name
    out = _cache_root(cache_dir) / f"{stem}_{imgsz}_{key}"
    return prepare(src, out, size=imgsz, gap=gap, val_frac=val_frac, seed=seed, workers=workers, scale=scale,
                   fit=fit)


def _run_dir(project, name, exist_ok, resume):
    """New run -> fresh dir (name, name2, ...); an unfinished run with last.pt is resumed when resume=True."""
    base = Path(project)
    k = 1
    while True:
        d = base / (name if k == 1 else f"{name}{k}")
        if exist_ok or not d.exists() or not any(d.iterdir()):
            return d, bool(resume) and (d / "last.pt").exists() and not (d / "train_done").exists()
        if resume and (d / "last.pt").exists() and not (d / "train_done").exists():
            return d, True
        k += 1


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

    import cv2
    import numpy as np

    from vrdet.data.prepare import IMG_EXT, _labels_for, find_splits
    _, splits = find_splits(data)
    imgs = sorted(p for p in splits["train"].iterdir() if p.suffix.lower() in IMG_EXT)
    lbl_dir = _labels_for(splits["train"])
    sides, longs, shorts = [], [], []
    for p in random.Random(0).sample(imgs, min(n, len(imgs))):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        sides.append(max(h, w))
        lp = lbl_dir / f"{p.stem}.txt"
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
    print(f"[vrdet] tile mode: pages ~{med:.0f} px, objects up to ~{big:.0f} px (p99) -> tile_scale {scale}, {how}")
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
RUN_FILES = ("last.pt", "best.pt", "train_done", "results.csv", "metrics.jsonl", "train_progress.log", "eval_val.txt",
             "eval_val.json", "exitcode")


def _known_keys():
    import inspect

    from vrdet.train import build_parser
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
            raise SystemExit(f"'{k}' is not a valid VRDet argument." + (f" Similar: {', '.join(near)}" if near else ""))


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
    print(f"[vrdet] batch {batch} ({why}; set batch= to override)")
    return batch


def _run_trainer(save_dir, prepared, opts, n_train, epochs, warmup_epochs, user_ema):
    """Run the trainer; on CUDA out-of-memory before the first epoch is saved, halve the batch and retry."""
    from vrdet import train as trainer
    for attempt in range(4):
        ipe = max(1, n_train // int(opts["batch"]))
        total = ipe * int(epochs)
        # warmup = warmup_epochs epochs of steps (at most epochs - 1), as in the usual one-stage trainer
        opts["warmup"] = opts["ema_warmups"] = max(1, round(min(warmup_epochs, max(int(epochs) - 1, 0)) * ipe))
        if not user_ema:                             # EMA half-life ~5% of the run (>= 1 epoch, >= 50 steps)
            opts["ema"] = round(min(0.9998, 0.5 ** (1 / max(ipe, total // 20, 50))), 6)
        argv = ["--data", str(prepared), "--out", str(save_dir)] + _to_argv(opts)
        args_file = save_dir / "vrdet_args.json"
        saved = json.loads(args_file.read_text()) if args_file.exists() else {}
        args_file.write_text(json.dumps({**saved, "prepared": str(prepared), **opts}, indent=1, default=str))
        if opts.get("verbose"):
            print("[vrdet] python -m vrdet.train " + " ".join(argv))
        try:
            trainer.main(argv)
            return
        except RuntimeError as e:                    # torch.OutOfMemoryError is a RuntimeError
            if ("out of memory" not in str(e).lower() or int(opts["batch"]) <= 2 or attempt == 3
                    or (save_dir / "last.pt").exists()):
                raise
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        old = int(opts["batch"])
        opts["batch"] = max(2, old // 2)
        print(f"[vrdet] WARNING: CUDA out of memory with batch={old}. Reducing to batch={opts['batch']} and retrying.")


def train(data, model="s", epochs=100, batch=None, imgsz=None, project="runs", name="exp", exist_ok=False,
          resume=True, tile=False, tile_scale=None, gap=200, val_frac=0.15, cache="auto", cache_dir=None, workers=None,
          device=None, recipe=True, warmup_epochs=3, patience=100, lrf=0.01, cos_lr=False, seed=0, optimizer="auto",
          **extra):
    """Train (or fine-tune when `model` is a .pt path). Returns SimpleNamespace(save_dir, best, last, metrics)."""
    check_keys(extra)
    _set_device(device)
    for k in list(extra):
        if k in ALIASES:
            extra[ALIASES[k]] = extra.pop(k)
        elif k in IGNORED:
            extra.pop(k)
            print(f"[vrdet] '{k}' is not used by VRDet (ignored)")
    if extra.pop("amp", True) is False:
        extra["no_amp"] = True
    if extra.pop("rot90", True) is False:
        extra["no_rot90"] = True
    if batch is not None and float(batch) <= 0:
        batch = None                                 # batch=-1: automatic
    user_ema = "ema" in extra

    # ---- resume: the run's saved settings win; only a few may change (like the usual trainer)
    save_dir, resuming = _run_dir(project, name, exist_ok, resume)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_file = save_dir / "vrdet_args.json"
    if resuming and saved_file.exists():
        saved = json.loads(saved_file.read_text())
        prepared = Path(saved.get("prepared", ""))
        if not _is_prepared(prepared):               # new machine / VM: rebuild the same prepared data
            prepared = prepare_data(saved.get("data", data), saved["img"], gap, val_frac, saved.get("scale", 1.0),
                                    cache_dir, workers, seed, fit=saved.get("fit", True))
        opts = {k: v for k, v in saved.items() if k not in ("data", "prepared", "warmup", "ema_warmups")}
        given = {"batch": batch, "imgsz": imgsz, **extra, **({"epochs": epochs} if epochs != 100 else {})}
        ignored = [k for k, v in given.items() if v is not None and k not in RESUME_OK
                   and str(opts.get(ALIASES.get(k, k), v)) != str(v)]
        if ignored:
            print(f"[vrdet] resuming {save_dir}: keeps its saved settings, ignores {ignored} "
                  f"(resume=False or a new name to start over)")
        else:
            print(f"[vrdet] resuming {save_dir}")
        opts.update({k: extra[k] for k in RESUME_OK if k in extra})
        if workers is not None:
            opts["workers"] = int(workers)
        n_train = _n_lines(Path(prepared) / "meta" / "train.jsonl")
        _run_trainer(save_dir, prepared, opts, n_train, opts["epochs"], warmup_epochs, user_ema=True)
        return _summary(save_dir)
    if not resuming:                                 # exist_ok=True on an old run: start over in place
        for f in RUN_FILES:
            (save_dir / f).unlink(missing_ok=True)

    aug = {k: extra.pop(k, v) for k, v in AUGMENT.items()}
    hsv = [extra.pop(k, d) for k, d in (("hsv_h", 0.015), ("hsv_s", 0.5), ("hsv_v", 0.3))]
    weights = None
    m = str(model)
    if m.lower() in SIZES:
        size, inherited = m.lower(), {}
    elif m.endswith(".pt") and Path(m).exists():
        weights = m
        src_args, _ = _ckpt_args(m)
        inherited = {k: src_args[k] for k in ARCH_KEYS if k in src_args}
        size = inherited.pop("size", "s")
        imgsz = imgsz or extra.pop("img", None) or src_args.get("img", 1024)
    else:
        raise SystemExit(f"model must be one of {SIZES} or an existing VRDet .pt checkpoint, got '{model}'")
    imgsz = int(imgsz or extra.pop("img", None) or 1024)
    if imgsz % 32:
        imgsz = int(math.ceil(imgsz / 32) * 32)
        print(f"[vrdet] imgsz must be a multiple of 32: using {imgsz}")

    fit = not tile and tile_scale is None
    if fit:
        med, big, small = page_stats(data) if not _is_prepared(data) else (0, 0, 0)
        if med:
            r = imgsz / med
            print(f"[vrdet] images ~{med:.0f} px (median long side) -> resized x{r:.2f} to long side {imgsz} px")
            if small and small * r < 3:
                print(f"[vrdet] WARNING: small objects (~{small:.0f} px) become ~{small * r:.1f} px at this size; "
                      f"use a larger imgsz or tile=True")
        scale = 1.0
    else:
        scale = auto_scale(data, imgsz) if tile_scale in (None, "auto") else float(tile_scale)

    sd = SIZE_DEFAULTS[size]
    if str(optimizer).lower() == "auto":            # like the usual trainer: auto picks the measured LR, ignores lr0
        if "lr" in extra:
            print(f"[vrdet] optimizer=auto: ignoring lr0={extra.pop('lr'):g}, using AdamW lr={sd['lr']:g} (measured for "
                  f"VRDet-{size}); set optimizer=AdamW to use your own lr0")
    elif str(optimizer).lower() != "adamw":
        raise SystemExit("optimizer must be 'auto' or 'AdamW'")
    prepared = prepare_data(data, imgsz, gap, val_frac, scale, cache_dir, workers, seed, fit=fit)
    n_train = _n_lines(Path(prepared) / "meta" / "train.jsonl")
    if cache == "auto":                              # decoded uint8 images in RAM when they take < 25% of it
        cache = n_train * imgsz * imgsz * 3 < 0.25 * _ram_bytes()
    batch = _batch_from(batch, sd, imgsz, n_train)

    steps = max(1, n_train // int(batch)) * int(epochs)
    if steps < MIN_STEPS and not extra.get("time"):
        print(f"[vrdet] WARNING: only {steps} optimizer steps ({max(1, n_train // int(batch))}/epoch x {epochs} epochs); this "
              f"detector needs ~{MIN_STEPS}+ to converge: raise epochs or lower batch")
    opts = {"size": size, "img": imgsz, "scale": scale, "fit": fit, "epochs": int(epochs), "batch": int(batch),
            "aug_iof": 0.25, "merge_iou": 0.7 if fit else 0.1, "cache": bool(cache),
            "schedule": "cos" if cos_lr else "linear", "min_lr_ratio": float(lrf),
            "lr": sd["lr"], "backbone_mult": sd["backbone_mult"], "wd": sd["wd"], "hsv": hsv,
            "eval_every": _eval_every(prepared, int(epochs)), "eval_images": 10**9, "patience": int(patience or 0),
            "seed": seed}
    q = _queries_for(prepared)
    if q > 300:
        opts.update(queries=q, num_top=q)
        print(f"[vrdet] dense images (p99 objects/image > 240): {q} queries")
    if workers is not None:
        opts["workers"] = int(workers)
    if recipe:
        opts.update(RECIPE)
    if aug["mosaic"]:
        opts["mosaic_mode"] = "yolo"
    opts.update({AUG_FLAGS[k]: v for k, v in aug.items()})
    opts.update(inherited)
    if weights:
        opts["weights"] = weights
    opts.update(extra)
    if float(opts["lr"]) > 3 * sd["lr"]:
        print(f"[vrdet] WARNING: lr0={opts['lr']:g} is {float(opts['lr']) / sd['lr']:.0f}x the default for VRDet-{size} "
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
    return SimpleNamespace(save_dir=str(save_dir), best=str(save_dir / "best.pt"), last=str(save_dir / "last.pt"),
                           metrics=metrics)


def val(model, data=None, imgsz=None, tile_scale=None, gap=200, val_frac=0.15, batch=8, workers=8, queries=900,
        cache_dir=None, device=None, seed=0):
    """DOTA-protocol mAP of a checkpoint on the val split of `data` (same hold-out and resizing as training).
    Without `data`, the checkpoint's own prepared data is used when it is available."""
    _set_device(device)
    import torch

    from vrdet.data.dota import dataset_classes
    from vrdet.engine import eval_dota
    from vrdet.eval.dota import summary_table
    from vrdet.predict import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, names, targs = load_model(model, dev, queries)
    imgsz = int(imgsz or targs.get("img", 1024))
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
    res, _ = eval_dota(net, prepared, dev, None, batch=batch, workers=workers, num_top=queries, img_size=imgsz,
                       merge_iou=0.7 if fit else 0.1,
                       post=targs.get("post", "flat"))
    print(summary_table(res, classes))
    return res


def predict(model, source, conf=0.25, save_dir="runs/predict", vis=True, imgsz=None, tile=None, tile_scale=None,
            gap=200, queries=900, batch=8, device=None, classes=None, names=None):
    """Detections per image {path: [{"class", "class_id", "score", "poly" (8 pixel coords)}]}; files in save_dir.
    Images are resized exactly as in training; tile=True forces tiling at native resolution (tile_scale).
    conf="auto" uses the per-class thresholds (best F2 on val) stored in best.pt; classes= keeps only these class
    ids or names; names= supplies class names for checkpoints that lack them."""
    _set_device(device)
    from vrdet.predict import run
    return run(model, source, save_dir, conf, names, imgsz, gap, queries, batch, bool(vis) and save_dir is not None,
               scale=tile_scale, tile=True if tile_scale is not None else tile, classes=classes)


class Detector:
    """Convenience wrapper: Detector("s" | "m" | "l" | "x" | "path/to/best.pt")."""

    def __init__(self, model="s"):
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help") or argv[0] not in MODES:
        print(__doc__)
        return 0 if (not argv or argv[0] in ("-h", "--help", "help")) else 2
    mode, kv = argv[0], parse_kv(argv[1:])
    if mode in ("val", "predict"):
        import inspect
        fn = val if mode == "val" else predict
        ok = set(inspect.signature(fn).parameters) | {"save"}
        bad = [k for k in kv if k not in ok]
        if bad:
            import difflib
            near = difflib.get_close_matches(bad[0], sorted(ok), n=3, cutoff=0.6)
            raise SystemExit(f"'{bad[0]}' is not a valid argument for vrdet {mode}."
                             + (f" Similar: {', '.join(near)}" if near else ""))
    if mode == "train":
        if "data" not in kv:
            raise SystemExit("vrdet train needs data=<data.yaml or dataset folder>")
        train(**kv)
    elif mode == "val":
        val(**kv)
    elif mode == "predict":
        if "save" in kv:
            kv["vis"] = kv.pop("save")
        predict(**kv)
    elif mode == "prepare":
        out = kv.pop("out", None)
        tile = bool(kv.get("tile", False))
        if out:
            from vrdet.data.prepare import prepare
            prepare(kv["data"], out, size=int(kv.get("imgsz", 1024)), gap=int(kv.get("gap", 200)),
                    val_frac=float(kv.get("val_frac", 0.15)), seed=int(kv.get("seed", 0)),
                    workers=kv.get("workers"), scale=float(kv.get("tile_scale", 1.0)), fit=not tile)
            print(f"[vrdet] prepared -> {out}")
        else:
            print(f"[vrdet] prepared -> {prepare_data(kv['data'], int(kv.get('imgsz', 1024)), fit=not tile)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
