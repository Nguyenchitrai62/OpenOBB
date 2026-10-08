"""Import a self-labelled YOLO-OBB dataset (e.g. an AI_Takeoff training export) into the VRDet split layout,
so `python -m vrdet.train --data <out>` fine-tunes on it exactly like on DOTA / FloorPlanCAD.

Accepted input layouts (Ultralytics-style, labels 'cls x1 y1 x2 y2 x3 y3 x4 y4' normalised to [0, 1]):
  {src}/images/{split}/*.png|jpg  +  {src}/labels/{split}/*.txt
  {src}/{split}/images/*          +  {src}/{split}/labels/*.txt
Class names come from {src}/data.yaml ('names': list or {id: name}) or --names a,b,c.
Large drawings are cut into SIZE x SIZE tiles with GAP overlap (same protocol as DOTA); evaluation merges the
tiles back per drawing, so mAP is reported per original image.

python tools/import_yolo_obb.py --src D:/exports/fire_obb --out D:/datasets/fire_vrdet --splits train,val
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab" / "data"))
sys.path.insert(0, str(ROOT))
from split_dota import split_set  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def class_names(src, names):
    if names:
        return [n.strip() for n in names.split(",")]
    y = Path(src) / "data.yaml"
    import yaml                                   # PyYAML (MIT)
    d = yaml.safe_load(y.read_text(encoding="utf-8"))["names"]
    return [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)


def split_dirs(src, split):
    src = Path(src)
    for img, lbl in ((src / "images" / split, src / "labels" / split), (src / split / "images", src / split / "labels")):
        if img.is_dir() and any(p.suffix.lower() in IMG_EXT for p in img.iterdir()):
            return img, lbl
    return None, None


def list_images(img_dir):
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT) if img_dir else []


def to_dota_raw(src, split, names, raw, files=None):
    """YOLO-OBB (normalised) -> DOTA raw layout (pixel polygons + class names) under {raw}/{split}.
    `files`: explicit list of (image, label_dir) pairs (used when val is carved out of train)."""
    if files is None:
        img_dir, lbl_dir = split_dirs(src, split)
        files = [(p, lbl_dir) for p in list_images(img_dir)]
    (raw / split / "images").mkdir(parents=True, exist_ok=True)
    (raw / split / "labelTxt").mkdir(parents=True, exist_ok=True)
    safe = [n.replace(" ", "_") for n in names]   # DOTA lines are whitespace separated
    n_img = n_obj = 0
    for p, lbl_dir in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARN unreadable image {p}")
            continue
        H, W = img.shape[:2]
        dst = raw / split / "images" / f"{p.stem}.png"
        if p.suffix.lower() == ".png":
            shutil.copy(p, dst)
        else:
            cv2.imwrite(str(dst), img)
        lines = []
        lp = lbl_dir / f"{p.stem}.txt"
        if lp.exists():
            for row in lp.read_text().split("\n"):
                v = row.split()
                if len(v) < 9:
                    continue
                c, xy = int(v[0]), [float(t) for t in v[1:9]]
                poly = [xy[i] * (W if i % 2 == 0 else H) for i in range(8)]
                lines.append(" ".join(f"{t:.1f}" for t in poly) + f" {safe[c]} 0")
        (raw / split / "labelTxt" / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_obj += len(lines)
    print(f"[import] {split}: {n_img} images, {n_obj} objects")
    return safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--names", default=None, help="comma-separated class names (else data.yaml)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="when the source has no val images, hold out this fraction of train images (whole pages)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = class_names(a.src, a.names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vrdet_import_") as tmp:
        raw = Path(tmp)
        splits = a.splits.split(",")
        plan = {s: None for s in splits}
        tr_img, tr_lbl = split_dirs(a.src, "train")
        va_img, _ = split_dirs(a.src, "val")
        if va_img is None:
            va_img, _ = split_dirs(a.src, "valid")       # AI_Takeoff / Roboflow naming
        if "val" in plan and va_img is None and tr_img is not None:
            import random
            imgs = list_images(tr_img)
            random.Random(a.seed).shuffle(imgs)
            n_val = max(1, int(round(len(imgs) * a.val_frac))) if len(imgs) > 1 else 0
            plan["val"] = [(p, tr_lbl) for p in sorted(imgs[:n_val])]
            plan["train"] = [(p, tr_lbl) for p in sorted(imgs[n_val:])]
            print(f"[import] no val split: holding out {n_val}/{len(imgs)} train images as val")
        elif "val" in plan and va_img is not None and split_dirs(a.src, "val")[0] is None:
            plan["val"] = [(p, split_dirs(a.src, "valid")[1]) for p in list_images(va_img)]
        for split in splits:
            safe = to_dota_raw(a.src, split, names, raw, files=plan[split])
            split_set(raw, out, split, a.size, a.gap, (1.0,), classes=tuple(safe), workers=a.workers)
    (out / "classes.json").write_text(json.dumps(safe))
    print(f"[import] done: {out} ({len(safe)} classes). Fine-tune: python -m vrdet.train --data {out} --size x ...")


if __name__ == "__main__":
    main()
