"""Cut DOTA-style images into fixed-size patches (our own implementation of the usual
DOTA-devkit/mmrotate protocol, no Ultralytics code).

Input  (per split):  {src}/{split}/images/*.png  and  {src}/{split}/labelTxt/*.txt (original DOTA format)
Output (per split):
  {out}/images/{split}/{img}__{rate}__{x0}___{y0}.jpg     patch, padded to size x size (ImageNet-mean gray)
  {out}/labels/{split}/<same>.txt                          'cls x1 y1 ... x4 y4' normalised by `size`
                                                           (YOLO-OBB format, readable by every pipeline)
  {out}/meta/{split}.jsonl                                 one line per patch: offsets, valid size and
                                                           pixel polygons with difficult/truncated flags
  {out}/gt/{split}/*.txt                                   copy of the original full-image labels (for eval)
  {out}/thumbs/{split}/{img}.jpg                           whole image, long side THUMB px (global context, H6)

Object -> patch rule: keep the object if area(obj ∩ patch) / area(obj) >= iof_thr (0.7).
Fully contained objects keep their original polygon; truncated ones are replaced by the
minimum-area rectangle of the visible part (clipped to the patch), flag trunc=1.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.split import PAD_BGR, THUMB, split_image, split_items, window_starts  # noqa: E402,F401
from openobb.eval.dota import DOTA1_CLASSES  # noqa: E402


def split_set(src, out, split, size=1024, gap=200, rates=(1.0,), iof_thr=0.7, classes=DOTA1_CLASSES,
              workers=None, limit=None, quality=95):
    src, out = Path(src), Path(out)
    imgs = sorted(p for p in (src / split / "images").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    if limit:
        imgs = imgs[:limit]
    lbl_dir = src / split / "labelTxt"
    items = [(str(p), str(lbl_dir / f"{p.stem}.txt") if lbl_dir.exists() else None) for p in imgs]
    return split_items(items, out, split, size, gap, rates, iof_thr, classes, workers, quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--rates", default="1.0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    for s in a.splits.split(","):
        split_set(a.src, a.out, s, a.size, a.gap, [float(r) for r in a.rates.split(",")],
                  workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
