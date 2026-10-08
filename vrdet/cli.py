"""VRDet command line and Python API. Arguments are key=value pairs.

    vrdet train   data=data.yaml model=s epochs=50 batch=16 imgsz=1024 project=runs name=exp
    vrdet train   data=data.yaml model=runs/exp/best.pt epochs=30            # fine-tune from a VRDet checkpoint
    vrdet val     model=runs/exp/best.pt data=data.yaml
    vrdet predict model=runs/exp/best.pt source=pages/ conf=0.25 save_dir=preds
    vrdet prepare data=data.yaml out=datasets/mydata imgsz=1024

    from vrdet import Detector
    model = Detector("s")                       # or Detector("runs/exp/best.pt")
    r = model.train(data="data.yaml", epochs=50, imgsz=1024, batch=16, project="runs", name="exp")
    model.predict("pages/", conf=0.25, save_dir="preds")

Datasets: data.yaml ('names' + train/val image folders) with labels 'class x1 y1 x2 y2 x3 y3 x4 y4' normalised
to [0, 1] in a sibling 'labels' folder. Pages of any size are tiled (imgsz tiles, `gap` px overlap); `scale`
resizes pages first. Without a val split, `val_frac` of the train pages is held out. Tiles are cached once
(default ~/.cache/vrdet) and reused. A run resumes automatically from {project}/{name}/last.pt.
"""
import hashlib
import math
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

MODES = ("train", "val", "predict", "prepare")
SIZES = ("s", "m", "l", "x")

# Best recipe measured on FloorPlanCAD / DOTA (research/LEDGER.md, c6): selective large-kernel adapters (LSK),
# repeat-factor sampling, one-to-many query group, adaptive query denoising, square-aware angle loss, IoU-aware
# matching, 900 inference queries.
RECIPE = {"lsk": True, "rfs": 0.1, "o2m_queries": 900, "o2m_k": 6, "aqd": True, "angle_weight": 1.0, "cost_iou": 0.5,
          "eval_queries": 900, "channels_last": True, "compile": True}
SIZE_DEFAULTS = {          # lr, backbone lr multiplier, weight decay, batch, ~GB of GPU memory per image at 1024 px
    "s": dict(lr=1e-4, backbone_mult=0.5, wd=1e-4, batch=16, gb=1.4),
    "m": dict(lr=1e-4, backbone_mult=0.1, wd=1e-4, batch=16, gb=1.8),
    "l": dict(lr=8e-5, backbone_mult=0.05, wd=1.25e-4, batch=8, gb=2.6),
    "x": dict(lr=6e-5, backbone_mult=0.1, wd=1.25e-4, batch=8, gb=4.0),
}
# architecture flags a fine-tune inherits from its source checkpoint (weights only load into the same shape)
ARCH_KEYS = ("size", "p2", "lsk", "strip_k", "ortho_heads", "dense", "dense_queries", "queries", "denoising",
             "no_rotate_sampling", "context")
ALIASES = {"lr0": "lr", "imgsz": "img", "weight_decay": "wd", "val_period": "eval_every", "amp": None,
           "patience": None, "device": None, "save_period": None, "plots": None, "verbose": None}


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


def prepare_data(data, imgsz=1024, gap=200, val_frac=0.15, scale=1.0, cache=None, workers=None, seed=0):
    """data.yaml / dataset folder -> tiled cache dir (built once, reused). Prepared dirs pass through."""
    if _is_prepared(data):
        return Path(data)
    from vrdet.data.prepare import prepare
    src = Path(data).resolve()
    key = hashlib.sha1(f"{src}|{imgsz}|{gap}|{val_frac}|{scale}|{seed}".encode()).hexdigest()[:8]
    stem = src.parent.name if src.suffix in (".yaml", ".yml") else src.name
    out = _cache_root(cache) / f"{stem}_{imgsz}_{key}"
    return prepare(src, out, size=imgsz, gap=gap, val_frac=val_frac, seed=seed, workers=workers, scale=scale)


def _run_dir(project, name, exist_ok, resume):
    """New run -> fresh dir (name, name2, ...); an unfinished run with last.pt is resumed when resume=True."""
    base = Path(project)
    k = 1
    while True:
        d = base / (name if k == 1 else f"{name}{k}")
        if exist_ok or not d.exists() or not any(d.iterdir()):
            return d, (d / "last.pt").exists() and not (d / "train_done").exists()
        if resume and (d / "last.pt").exists() and not (d / "train_done").exists():
            return d, True
        k += 1


def _queries_for(prepared):
    """Decoder queries for training: 300 (D-FINE default) unless dense tiles need more (one-to-one matching cannot
    supervise more objects than queries; 300 saturated on dense DOTA tiles, research/LEDGER.md F4)."""
    import numpy as np
    counts = [sum(1 for _ in open(p)) for p in (Path(prepared) / "labels" / "train").glob("*.txt")]
    if not counts:
        return 300
    p99 = float(np.percentile(counts, 99))
    return 300 if p99 <= 240 else int(min(900, np.ceil(p99 * 1.5 / 100) * 100))


def auto_scale(data, imgsz, n=40):
    """Pages up to ~1.6x the tile size are resized into one tile (like a whole-image resize); larger pages keep their
    native resolution and are tiled, so thin lines survive. Already-tiled data is used as is."""
    if _is_prepared(data):
        return 1.0
    import random

    import cv2
    import numpy as np

    from vrdet.data.prepare import IMG_EXT, find_splits
    _, splits = find_splits(data)
    imgs = sorted(p for p in splits["train"].iterdir() if p.suffix.lower() in IMG_EXT)
    sides = []
    for p in random.Random(0).sample(imgs, min(n, len(imgs))):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None:
            sides.append(max(im.shape[:2]))
    if not sides:
        return 1.0
    med = float(np.median(sides))
    scale = math.floor(imgsz / med * 1e4) / 1e4 if imgsz < med <= 1.6 * imgsz else 1.0   # page <= one tile
    print(f"[vrdet] pages ~{med:.0f} px (median long side) -> scale {scale} "
          f"({'one tile per page' if med * scale <= imgsz else 'tiled at native resolution'}; set scale= to override)")
    return scale


def _ckpt_args(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("args", {}) or {}, list(ck.get("classes") or [])


def train(data, model="s", epochs=50, batch=None, imgsz=None, project="runs", name="exp", exist_ok=False,
          resume=True, gap=200, val_frac=0.15, scale=None, cache=None, workers=None, device=None, recipe=True,
          seed=0, **extra):
    """Train (or fine-tune when `model` is a .pt path). Returns SimpleNamespace(save_dir, best, last, metrics)."""
    _set_device(device)
    for k in list(extra):
        if k not in ALIASES:
            continue
        v = extra.pop(k)
        if ALIASES[k]:
            extra[ALIASES[k]] = v
        elif k == "amp":
            extra["no_amp"] = v is False
        else:
            print(f"[vrdet] '{k}' is not used by VRDet (ignored)")
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
    scale = auto_scale(data, imgsz) if scale in (None, "auto") else float(scale)
    if imgsz % 32:
        raise SystemExit("imgsz must be a multiple of 32")
    sd = SIZE_DEFAULTS[size]
    if batch is None:
        batch = sd["batch"]
        mem = _gpu_mem_gb()
        if mem:                                      # fit the default batch to the GPU (measured ~GB per image)
            fit = int((mem - 2.0) / (sd["gb"] * (imgsz / 1024) ** 2)) // 2 * 2
            batch = max(2, min(batch, fit))
            print(f"[vrdet] batch {batch} (GPU {mem:.0f} GB; set batch= to override)")
    prepared = prepare_data(data, imgsz, gap, val_frac, scale, cache, workers, seed)
    save_dir, resuming = _run_dir(project, name, exist_ok, resume)
    save_dir.mkdir(parents=True, exist_ok=True)
    if resuming:
        print(f"[vrdet] resuming {save_dir} (use resume=False or a new name to start over)")

    opts = {"size": size, "img": imgsz, "scale": scale, "epochs": int(epochs), "batch": int(batch),
            "lr": sd["lr"], "backbone_mult": sd["backbone_mult"], "wd": sd["wd"],
            "eval_every": max(1, round(int(epochs) / 8)), "eval_images": 10**9, "seed": seed}
    q = _queries_for(prepared)
    if q > 300 and "queries" not in inherited:
        opts.update(queries=q, num_top=q)
        print(f"[vrdet] dense tiles (p99 objects/tile > 240): {q} queries")
    if workers is not None:
        opts["workers"] = int(workers)
    if recipe:
        opts.update(RECIPE)
    opts.update(inherited)
    if weights:
        opts["weights"] = weights
    opts.update(extra)
    argv = ["--data", str(prepared), "--out", str(save_dir)]
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
    (save_dir / "vrdet_args.json").write_text(json.dumps({"data": str(data), "prepared": str(prepared), **opts},
                                                         indent=1, default=str))
    print("[vrdet] python -m vrdet.train " + " ".join(argv))
    from vrdet import train as trainer
    trainer.main(argv)
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
    res = SimpleNamespace(save_dir=str(save_dir), best=str(save_dir / "best.pt"), last=str(save_dir / "last.pt"),
                          metrics=metrics)
    if metrics:
        print(f"[vrdet] {save_dir}: val mAP50 {metrics['mAP50']:.4f}  mAP50:95 {metrics['mAP50_95']:.4f}")
    print(f"[vrdet] weights: {res.best} (deploy / fine-tune), {res.last} (resume)")
    return res


def val(model, data, imgsz=None, scale=None, gap=200, val_frac=0.15, batch=8, workers=8, queries=900, cache=None,
        device=None, seed=0):
    """DOTA-protocol mAP of a checkpoint on the val split of `data` (same hold-out as training)."""
    _set_device(device)
    import torch

    from vrdet.data.dota import dataset_classes
    from vrdet.engine import eval_dota
    from vrdet.eval.dota import format_table
    from vrdet.predict import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, names, targs = load_model(model, dev, queries)
    imgsz = int(imgsz or targs.get("img", 1024))
    scale = float(scale if scale is not None else targs.get("scale", 1.0))
    prepared = prepare_data(data, imgsz, gap, val_frac, scale, cache, workers, seed)
    classes = list(dataset_classes(prepared))
    if classes != list(names):
        print(f"[vrdet] WARNING: dataset classes {classes} differ from the model's {list(names)}")
    res, _ = eval_dota(net, prepared, dev, None, batch=batch, workers=workers, num_top=queries, img_size=imgsz,
                       post=targs.get("post", "flat"))
    print(format_table(res, classes))
    return res


def predict(model, source, conf=0.25, save_dir="runs/predict", vis=True, imgsz=None, scale=None, gap=200,
            queries=900, batch=8, device=None, classes=None):
    """Detections per image {path: [{"class", "class_id", "score", "poly" (8 pixel coords)}]}; files in save_dir."""
    _set_device(device)
    from vrdet.predict import run
    return run(model, source, save_dir, conf, classes, imgsz, gap, queries, batch, bool(vis) and save_dir is not None,
               scale=scale)


class Detector:
    """Ultralytics-style convenience wrapper: Detector("s" | "m" | "l" | "x" | "path/to/best.pt")."""

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
        if out:
            from vrdet.data.prepare import prepare
            prepare(kv["data"], out, size=int(kv.get("imgsz", 1024)), gap=int(kv.get("gap", 200)),
                    val_frac=float(kv.get("val_frac", 0.15)), seed=int(kv.get("seed", 0)),
                    workers=kv.get("workers"), scale=float(kv.get("scale", 1.0)))
            print(f"[vrdet] prepared -> {out}")
        else:
            print(f"[vrdet] prepared -> {prepare_data(**kv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
