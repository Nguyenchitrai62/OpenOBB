"""DIOR-R (oriented) -> OpenOBB split layout. Research/benchmark data only.

Standard DIOR-R protocol: train on trainval (11,725 images), evaluate on test (11,738). To reuse every
OpenOBB tool unchanged, the output calls them "train" and "val":
  {out}/images/train = DIOR train+val,  {out}/images/val = DIOR test   (800 x 800 JPEGs, copied as is)
  labels/, meta/, gt/ (DOTA-format polygons + class names), thumbs/, classes.json

Source: HF dataset wxy-yumuxia/henan-data-DIOR (pinned; official DIOR-R layout). Fallback: Kaggle
larbisck/dior-dataset (anonymous download endpoint).

python colab/data/get_dior.py --root /content/datasets
"""
import argparse
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2

REPO, REV = "wxy-yumuxia/henan-data-DIOR", "26f509ad22b11e39f848781ebd2d872b864844c0"
FILES = ["Annotations.zip", "ImageSets.zip", "JPEGImages-trainval.zip", "JPEGImages-test.zip"]
DIOR_CLASSES = ("airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
                "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor",
                "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill")
CORNERS = ("x_left_top", "y_left_top", "x_right_top", "y_right_top", "x_right_bottom", "y_right_bottom",
           "x_left_bottom", "y_left_bottom")


def fetch(raw):
    try:
        from huggingface_hub import hf_hub_download
        for f in FILES:
            hf_hub_download(REPO, f, repo_type="dataset", revision=REV, local_dir=str(raw))
    except Exception as e:  # noqa: BLE001
        print(f"[dior] HF failed ({e}); trying Kaggle", flush=True)
        z = raw / "dior.zip"
        subprocess.run(["curl", "-sL", "-o", str(z),
                        "https://www.kaggle.com/api/v1/datasets/download/larbisck/dior-dataset"], check=True)
    for z in raw.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)


def parse(xml_path):
    objs = []
    for o in ET.parse(xml_path).getroot().iter("object"):
        name = o.findtext("name")
        rb = o.find("robndbox")
        if name not in DIOR_CLASSES or rb is None:
            continue
        poly = [float(rb.findtext(k)) for k in CORNERS]
        objs.append((DIOR_CLASSES.index(name), int(o.findtext("difficult") or 0), poly))
    return objs


def convert(item, out, split, ann_dir, img_dirs):
    iid = item
    src = next((d / f"{iid}.jpg" for d in img_dirs if (d / f"{iid}.jpg").exists()), None)
    if src is None:
        return None
    objs = parse(ann_dir / f"{iid}.xml") if (ann_dir / f"{iid}.xml").exists() else []
    shutil.copy(src, out / "images" / split / f"{iid}.jpg")
    img = cv2.imread(str(src))
    H, W = img.shape[:2]
    cv2.imwrite(str(out / "thumbs" / split / f"{iid}.jpg"), cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA))
    with open(out / "labels" / split / f"{iid}.txt", "w") as f:
        for c, _, p in objs:
            f.write(f"{c} " + " ".join(f"{min(max(v / (W if i % 2 == 0 else H), 0), 1):.6f}" for i, v in enumerate(p)) + "\n")
    with open(out / "gt" / split / f"{iid}.txt", "w") as f:
        for c, d, p in objs:
            f.write(" ".join(f"{v:.1f}" for v in p) + f" {DIOR_CLASSES[c]} {d}\n")
    return {"name": iid, "src": iid, "rate": 1.0, "x0": 0, "y0": 0, "w": W, "h": H, "img_w": W, "img_h": H,
            "objs": [[c, d, 0] + p for c, d, p in objs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/datasets")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    root = Path(a.root)
    raw, out = root / "dior_raw", root / "dior_r"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / ".fetched").exists():
        fetch(raw)
        (raw / ".fetched").write_text("ok")
    ann_dir = next(raw.rglob("Oriented Bounding Boxes"))
    sets = {p.stem: p for p in raw.rglob("Main/*.txt")}
    img_dirs = sorted({p.parent for p in raw.rglob("*.jpg")})
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(list(DIOR_CLASSES)))
    plan = {"train": [l.strip() for k in ("train", "val") for l in sets[k].read_text().split() if l.strip()],
            "val": [l.strip() for l in sets["test"].read_text().split() if l.strip()]}
    for split, ids in plan.items():
        if (out / f".done_{split}").exists():
            continue
        for d in ("images", "labels", "gt", "thumbs"):
            shutil.rmtree(out / d / split, ignore_errors=True)
            (out / d / split).mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(exist_ok=True)
        n = n_obj = 0
        with Pool(a.workers) as pool, open(out / "meta" / f"{split}.jsonl", "w") as mf:
            for m in pool.imap_unordered(partial(convert, out=out, split=split, ann_dir=ann_dir, img_dirs=img_dirs),
                                         ids, chunksize=16):
                if m is None:
                    continue
                mf.write(json.dumps(m) + "\n")
                n += 1
                n_obj += len(m["objs"])
        (out / f".done_{split}").write_text("ok")
        print(f"[dior] {split}: {n} images, {n_obj} objects", flush=True)


if __name__ == "__main__":
    main()
