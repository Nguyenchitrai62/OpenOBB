"""Public pretraining data, fetched on demand and prepared once (marker files make every stage idempotent).

coco_images(root)  COCO val2017 images (5k, ~1 GB, no labels needed: distillation)        -> image folder
dota(root)         DOTA-v1.0 train + val with labels, cut into 1024 / 200 patches
                   (OpenOBB's prepared layout: images / labels / meta / gt + classes.json)   -> prepared folder

DOTA is for research / benchmarking only (DOTA terms); COCO images are under their Flickr licences. Weights pretrained
on them are research weights: fine-tune on your own labelled data for a product.
"""
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

COCO_VAL = "http://images.cocodataset.org/zips/val2017.zip"
DOTA_HF, DOTA_REV = "Last-Bullet/DOTAv1.0", "251502d4b016f95f2688471c32cd79d3764db058"
DOTA_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"   # fallback mirror (data only)


def _download(url, dst, log=print):
    dst = Path(dst)
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"[data] downloading {url}")
    t0, last = time.time(), [0.0]

    def hook(n, bs, total):
        if time.time() - last[0] > 30 and total > 0:
            last[0] = time.time()
            log(f"[data]   {min(n * bs, total) / 2**20:.0f} / {total / 2**20:.0f} MB")
    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dst)
    log(f"[data]   done in {time.time() - t0:.0f}s")
    return dst


def coco_images(root, log=print):
    """COCO val2017 images -> {root}/coco/val2017 (downloaded and unzipped once)."""
    out = Path(root) / "coco"
    d = out / "val2017"
    if (out / ".done").exists():
        return d
    out.mkdir(parents=True, exist_ok=True)
    z = _download(COCO_VAL, out / "val2017.zip", log)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)
    z.unlink()
    (out / ".done").write_text("ok")
    log(f"[data] COCO val2017: {len(list(d.glob('*.jpg')))} images in {d}")
    return d


def _dota_raw(raw, splits, log=print):
    """Original DOTA-v1.0 images (PNG) + labelTxt -> folder with {split}/images and {split}/labelTxt."""
    try:
        from huggingface_hub import snapshot_download
        for attempt in range(4):
            try:
                snapshot_download(DOTA_HF, repo_type="dataset", revision=DOTA_REV,
                                  allow_patterns=[f"DOTA_V1.0/{s}/*" for s in splits], local_dir=str(raw), max_workers=16)
                return raw / "DOTA_V1.0"
            except Exception as e:  # noqa: BLE001  (rate limits, transient network)
                log(f"[data] DOTA from Hugging Face, attempt {attempt + 1} failed: {str(e)[:200]}")
                time.sleep(30 * (attempt + 1))
    except ImportError:
        log("[data] huggingface_hub not installed")
    log("[data] DOTA: falling back to the zip mirror")
    z = _download(DOTA_ZIP, raw / "DOTAv1.zip", log)
    with zipfile.ZipFile(z) as f:
        f.extractall(raw)
    out = raw / "DOTA_from_zip"
    for s in splits:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        for p in (raw / "DOTAv1" / "images" / s).glob("*.jpg"):            # the splitter keeps every image format
            shutil.copy(p, out / s / "images" / p.name)
        shutil.copytree(raw / "DOTAv1" / "labels" / f"{s}_original", out / s / "labelTxt", dirs_exist_ok=True)
    return out


def dota(root, splits=("train", "val"), size=1024, gap=200, workers=None, log=print, fetch=None):
    """DOTA-v1.0 cut into size x size patches (gap overlap) in OpenOBB's prepared layout -> {root}/dota1_{size}.
    fetch(raw, splits) -> folder with {split}/images + {split}/labelTxt (tests inject a local one)."""
    from openobb.data.split import split_items
    from openobb.eval.dota import DOTA1_CLASSES
    root = Path(root)
    out = root / f"dota1_{size}"
    if all((out / f".done_{s}").exists() for s in splits) and (out / "classes.json").exists():
        return out
    raw = root / "dota_raw"
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".fetched"
    if marker.exists():
        src = Path(marker.read_text().strip())
    else:
        src = (fetch or (lambda r, s: _dota_raw(r, s, log)))(raw, list(splits))
        for s in splits:
            n = sum(1 for p in (Path(src) / s / "images").iterdir())
            log(f"[data] DOTA {s}: {n} images")
        marker.write_text(str(src))
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(list(DOTA1_CLASSES)))
    for s in splits:
        if (out / f".done_{s}").exists():
            continue
        shutil.rmtree(out / "images" / s, ignore_errors=True)
        shutil.rmtree(out / "labels" / s, ignore_errors=True)
        imgs = sorted(p for p in (Path(src) / s / "images").iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        lbl = Path(src) / s / "labelTxt"
        items = [(str(p), str(lbl / f"{p.stem}.txt")) for p in imgs]
        t0 = time.time()
        n = split_items(items, out, s, size, gap, (1.0,), 0.7, DOTA1_CLASSES, workers or os.cpu_count(), 95)
        (out / f".done_{s}").write_text("ok")
        log(f"[data] DOTA {s}: {n} patches of {size} px in {time.time() - t0:.0f}s")
    return out
