"""`openobb train ... pretrain=coco|dota`: one command from nothing to a pretrained core, then the usual training.

The new cores (OpenOBB6, OpenOBB9) have no public weights; trained from scratch on a small dataset they stall
(research/LEDGER.md F25, F40, F48: OpenOBB9 0.541 without pretraining). This module does what YOLO users get from a
downloaded checkpoint, on the user's own GPU:

  pretrain=coco   1. COCO val2017 images (5k) + your training images, no labels
                  2. distillation into the core (openobb.pretrain): teachers D-FINE-X COCO core + DINOv2
  pretrain=dota   1. DOTA-v1.0 with labels, cut into 1024 px patches, + your training images
                  2. distillation as above, on the DOTA patches + your images
                  3. supervised OBB training of the whole detector on DOTA (tiny rotated objects, the prior
                     YOLO-OBB checkpoints carry)
  pretrain=<folders>  distillation on your own unlabelled image folders (comma-separated) + your training images
  (sources combine: pretrain=coco,dota)

Every stage is cached under `pretrain_dir` (default ~/.cache/openobb/pretrain; on Colab put it on Drive to reuse it
across sessions) and keyed by model, sources, epochs and image size: a second run with the same settings starts
directly from the cached weights. Held-out images for the distillation gate (T2) are your validation images.
DOTA / COCO are research data: weights pretrained on them are for research; fine-tune on your own labels.
"""
import hashlib
import json
import shutil
from pathlib import Path

PUBLIC = ("coco", "dota")
DISTILL_EPOCHS = {"coco": 20, "dota": 10, "folders": 20}     # defaults; pretrain_epochs= overrides
DOTA_EPOCHS = 12                                              # supervised DOTA stage; pretrain_det_epochs= overrides
ARCH_SWITCHES = ("v6_off", "v8_off", "v9_off")


def default_root():
    import os
    return Path(os.environ.get("OPENOBB_PRETRAIN") or Path.home() / ".cache" / "openobb" / "pretrain")


def parse_sources(spec):
    """'coco' / 'dota,coco' / '/a,/b' / list -> (sorted public names, folder paths)."""
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    public = sorted({s.lower() for s in items if s.lower() in PUBLIC})
    folders = [s for s in items if s.lower() not in PUBLIC]
    bad = [f for f in folders if not Path(f).is_dir()]
    if bad:
        raise SystemExit(f"pretrain= takes coco, dota or existing image folders; not found: {bad}")
    if not public and not folders:
        raise SystemExit("pretrain= is empty: use pretrain=coco, pretrain=dota or pretrain=<image folders>")
    return public, folders


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def auto_pretrain(model, spec, user_train, user_val=None, imgsz=1024, epochs=None, det_epochs=None, root=None,
                  teacher=None, device=None, workers=8, batch=16, switches=None, log=print, _download=None,
                  _distill=None, _train=None):
    """-> path of the weights to start the user's training from (a core, or a DOTA-trained detector).
    model: openobb9{s,m,l,x} / openobb6{n,s,m,l,x}; user_train / user_val: image folders of the user's data."""
    from openobb.data import download
    distill = _distill or __import__("openobb.pretrain", fromlist=["pretrain"]).pretrain
    public, folders = parse_sources(spec)
    root = Path(root) if root else default_root()
    data_root = root / "data"
    switches = {k: v for k, v in (switches or {}).items() if v}
    v6 = str(model).lower().startswith("openobb6")
    if "dota" in public and v6:
        raise SystemExit("pretrain=dota trains the whole detector on DOTA: use it with openobb9*, "
                         "or pretrain=coco / <folders> for openobb6*")
    d_imgsz = min(int(imgsz), 1024)
    kind = "dota" if "dota" in public else ("coco" if public else "folders")
    epochs = int(epochs or DISTILL_EPOCHS[kind])
    det_epochs = int(DOTA_EPOCHS if det_epochs is None else det_epochs)
    key = _key(model, public, [str(Path(f).resolve()) for f in folders], str(Path(user_train).resolve()), epochs,
               d_imgsz, str(teacher or "x"), switches, det_epochs if "dota" in public else 0)
    final = root / f"{model}_{'+'.join(public + (['folders'] if folders else []))}_{key}.pt"
    if final.exists():
        log(f"[pretrain] reusing {final} (same model, sources and settings)")
        return str(final)
    root.mkdir(parents=True, exist_ok=True)

    dl = _download or download
    dirs, dota_dir = [], None
    if "coco" in public:
        dirs.append(str(dl.coco_images(data_root, log=log)))
    if "dota" in public:
        dota_dir = dl.dota(data_root, log=log)
        dirs.append(str(Path(dota_dir) / "images" / "train"))
    dirs += [str(f) for f in folders] + [str(user_train)]
    core = root / f"{model}_core_{key}.pt"
    if not core.exists():
        log(f"[pretrain] stage 1: distilling {model} for {epochs} epochs on {len(dirs)} image folder(s) at {d_imgsz} px")
        kw = dict(teachers="dinov2") if v6 else dict(teachers="dfine,dinov2", teacher_dfine=teacher or "x")
        val = str(user_val) if user_val and Path(user_val).is_dir() and not v6 else None
        w = distill(model, data=",".join(dirs), val_data=val, epochs=epochs, imgsz=d_imgsz, batch=batch,
                    workers=workers, project=str(root / "runs"), name=f"distill_{key}", exist_ok=True, device=device,
                    **{k: v for k, v in switches.items() if k in ("v6_off", "v9_off")}, **kw)
        shutil.copy(w, core)
    if dota_dir is None or det_epochs <= 0:
        shutil.copy(core, final)
        log(f"[pretrain] pretrained core: {final}")
        return str(final)

    log(f"[pretrain] stage 2: training {model} on DOTA-v1.0 (labelled, 1024 px patches) for {det_epochs} epochs")
    run_train = _train or __import__("openobb.cli", fromlist=["train"]).train
    r = run_train(data=str(dota_dir), model=model, weights=str(core), epochs=det_epochs, imgsz=1024, tile=True,
                  tile_scale=1.0, project=str(root / "runs"), name=f"dota_{key}", exist_ok=True, device=device,
                  workers=workers, val_period=max(1, det_epochs), **switches)
    best = Path(r.best) if Path(str(r.best)).exists() else Path(r.last)
    shutil.copy(best, final)
    log(f"[pretrain] DOTA-trained detector: {final}")
    return str(final)
