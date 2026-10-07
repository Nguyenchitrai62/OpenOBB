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
        if img.is_dir():
            return img, lbl
    raise SystemExit(f"no images for split '{split}' under {src}")


def to_dota_raw(src, split, names, raw):
    """YOLO-OBB (normalised) -> DOTA raw layout (pixel polygons + class names) under {raw}/{split}."""
    img_dir, lbl_dir = split_dirs(src, split)
    (raw / split / "images").mkdir(parents=True, exist_ok=True)
    (raw / split / "labelTxt").mkdir(parents=True, exist_ok=True)
    safe = [n.replace(" ", "_") for n in names]   # DOTA lines are whitespace separated
    n_img = n_obj = 0
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
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
    a = ap.parse_args()
    names = class_names(a.src, a.names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vrdet_import_") as tmp:
        raw = Path(tmp)
        for split in a.splits.split(","):
            safe = to_dota_raw(a.src, split, names, raw)
            split_set(raw, out, split, a.size, a.gap, (1.0,), classes=tuple(safe), workers=a.workers)
    (out / "classes.json").write_text(json.dumps(safe))
    print(f"[import] done: {out} ({len(safe)} classes). Fine-tune: python -m vrdet.train --data {out} --size x ...")


if __name__ == "__main__":
    main()
