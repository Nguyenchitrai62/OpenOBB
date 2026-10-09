"""Labelled OBB dataset -> VRDet tiled training layout.

Accepts a data.yaml (names + train/val image folders; relative paths incl. the "../train/images" form)
or a dataset folder (images/{split} + labels/{split}, or {split}/images + {split}/labels, plus data.yaml).
Labels: 'class_id x1 y1 x2 y2 x3 y3 x4 y4' normalised to [0, 1]. Images are tiled SIZE x SIZE with GAP overlap;
evaluation merges tiles back per original image. When no val split exists, VAL_FRAC of the train images is held out.
"""
import hashlib
import json
import random
import tempfile
from pathlib import Path

import cv2

from openobb.data.split import split_items

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _load_yaml(path):
    import yaml                                     # PyYAML (MIT)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _names(cfg):
    n = cfg.get("names")
    if isinstance(n, dict):
        return [str(n[k]) for k in sorted(n, key=int)]
    if isinstance(n, (list, tuple)):
        return [str(x) for x in n]
    raise SystemExit("data.yaml has no 'names'")


def _resolve(entry, yaml_dir, root):
    """Image folder resolution: absolute, relative to `path`, relative to the yaml, or the '../x' form."""
    if not entry:
        return None
    cands = [Path(entry)]
    for base in (root, yaml_dir):
        if base is not None:
            cands += [base / entry, base / str(entry).lstrip("./").removeprefix("../")]
    for c in cands:
        if c.is_dir() and any(p.suffix.lower() in IMG_EXT for p in c.iterdir()):
            return c.resolve()
    return None


def _labels_for(img_dir):
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):        # last 'images' path component -> 'labels'
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels"


def find_splits(data):
    """-> (names, {"train": img_dir, "val": img_dir or None})"""
    data = Path(data)
    yaml_path = data if data.suffix in (".yaml", ".yml") else next(
        (p for p in (data / "data.yaml", data / "dataset.yaml") if p.exists()), None)
    cfg = _load_yaml(yaml_path) if yaml_path else {}
    yaml_dir = yaml_path.parent if yaml_path else data
    root = Path(cfg["path"]) if cfg.get("path") else None
    if root is not None and not root.is_absolute():
        root = yaml_dir / root
    splits = {"train": _resolve(cfg.get("train"), yaml_dir, root), "val": _resolve(cfg.get("val"), yaml_dir, root)}
    base = data if data.is_dir() else yaml_dir
    for s, alts in (("train", ("train",)), ("val", ("val", "valid"))):
        for a in alts:
            if splits[s] is None:
                splits[s] = _resolve(f"images/{a}", base, None) or _resolve(f"{a}/images", base, None)
    if splits["train"] is None:
        raise SystemExit(f"no training images found from {data}")
    return _names(cfg) if cfg else None, splits


PREP_VERSION = 2             # bump when the prepared layout changes: old caches are rebuilt


def _fingerprint(paths):
    """Content fingerprint of a dataset: (name, size, mtime) of every image and label file, so edited or added labels
    invalidate the prepared cache."""
    h = hashlib.sha1()
    for p in paths:
        try:
            st = p.stat()
            h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{p}|missing\n".encode())
    return h.hexdigest()


def _to_dota(img_paths, lbl_dir, names, out_dir):
    """Normalised OBB txt labels -> polygon label files (pixels + class names); images are NOT copied.
    Unreadable images are skipped with a warning; duplicate label lines are removed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = [n.replace(" ", "_") for n in names]
    items, n_obj = [], 0
    stats = {"backgrounds": 0, "missing labels": 0, "corrupt lines": 0, "unreadable images": 0}
    for p in img_paths:
        lp = lbl_dir / f"{p.stem}.txt"
        dst = out_dir / f"{p.stem}.txt"
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)       # same decoding (EXIF orientation) as the tiling step
        if im is None or min(im.shape[:2]) < 10:
            stats["unreadable images"] += 1
            print(f"[data] WARNING: skipping unreadable or tiny image {p}")
            continue
        h, w = im.shape[:2]
        lines = []
        if not lp.exists():
            stats["missing labels"] += 1
        else:
            rows = [r.split() for r in dict.fromkeys(r.strip() for r in lp.read_text().splitlines()) if r]
            for v in rows:                          # 'class x1 y1 ... x4 y4' with coordinates in [0, 1]
                try:
                    c, xy = int(float(v[0])), [float(t) for t in v[1:9]]
                    ok = len(v) >= 9 and 0 <= c < len(safe) and all(-0.05 <= t <= 1.05 for t in xy)
                except (ValueError, IndexError):
                    ok = False
                if not ok:
                    stats["corrupt lines"] += 1
                    continue
                poly = [min(max(xy[i], 0.0), 1.0) * (w if i % 2 == 0 else h) for i in range(8)]
                lines.append(" ".join(f"{t:.1f}" for t in poly) + f" {safe[c]} 0")
        if not lines:
            stats["backgrounds"] += 1
        dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        items.append((str(p), str(dst)))
        n_obj += len(lines)
    return items, safe, n_obj, stats


def prepare(data, out, size=1024, gap=200, val_frac=0.15, seed=0, names=None, workers=None, scale=1.0,
            fit=False, ext=".png"):
    """Build the tiled layout in `out` (reused while the settings and the dataset files are unchanged).
    fit=True resizes each image so its long side = size (one tile per image, any resolution); otherwise `scale`
    resizes every image before tiling (e.g. 0.8: 1280 px pages -> one 1024 tile). Evaluation maps back."""
    out = Path(out)
    yaml_names, splits = find_splits(data)
    names = list(names) if names else yaml_names
    if not names:
        raise SystemExit("class names unknown: provide data.yaml with 'names'")
    tr = sorted(p for p in splits["train"].iterdir() if p.suffix.lower() in IMG_EXT)
    va = sorted(p for p in splits["val"].iterdir() if p.suffix.lower() in IMG_EXT) if splits["val"] else []
    files = [q for d, imgs in ((splits["train"], tr), (splits["val"], va)) if d is not None
             for p in imgs for q in (p, _labels_for(d) / f"{p.stem}.txt")]
    stamp = {"data": str(Path(data).resolve()), "size": size, "gap": gap, "val_frac": val_frac, "seed": seed,
             "scale": scale, "fit": fit, "ext": ext, "names": names, "version": PREP_VERSION,
             "files": _fingerprint(files)}
    done = out / ".prepared.json"
    if done.exists():
        old = json.loads(done.read_text())
        if {k: v for k, v in old.items() if k != "summary"} == stamp:
            for line in old.get("summary", []):
                print(line)
            print(f"[data] reuse {out}")
            return out
    if splits["val"] is not None:
        plan = {"train": (tr, _labels_for(splits["train"])), "val": (va, _labels_for(splits["val"]))}
    else:
        rng = random.Random(seed)
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(tr) * val_frac))) if len(tr) > 1 else 0
        hold = set(idx[:n_val])
        lbl = _labels_for(splits["train"])
        plan = {"train": ([p for i, p in enumerate(tr) if i not in hold], lbl),
                "val": ([p for i, p in enumerate(tr) if i in hold], lbl)}
        print(f"[data] no val split: holding out {n_val}/{len(tr)} train images")
    summary = []
    with tempfile.TemporaryDirectory(prefix="vrdet_lbl_") as tmp:
        for split, (imgs, lbl_dir) in plan.items():
            items, safe, n_obj, st = _to_dota(imgs, lbl_dir, names, Path(tmp) / split)
            summary.append(f"[data] {split}: {len(items)} images, {n_obj} objects, "
                           + ", ".join(f"{v} {k}" for k, v in st.items()))
            print(summary[-1])
            if st["corrupt lines"]:
                print(f"[data] WARNING {split}: {st['corrupt lines']} label lines skipped (need 'class x1 y1 ... x4 y4', "
                      f"class < {len(safe)}, coordinates in [0, 1])")
            split_items(items, out, split, size, gap, (float(scale),), classes=tuple(safe), workers=workers, fit=fit,
                        ext=ext)
    (out / "classes.json").write_text(json.dumps([n.replace(" ", "_") for n in names]))
    done.write_text(json.dumps({**stamp, "summary": summary}))
    return out
