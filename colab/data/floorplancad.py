"""FloorPlanCAD -> VRDet split (raster patches + OBB labels + vector primitives). Our own SVG parser and
OpenCV renderer (no third-party renderer licences involved). Research/benchmark data: CC BY-NC.

Each drawing (viewBox 0..140) becomes one SIZE x SIZE image (default 1024 px, ~7.3 px per unit).
Objects = the 30 "thing" classes (semanticId 1..30), one OBB per (semanticId, instanceId) = minimum-area
rectangle of densely sampled primitive points. Stuff classes (31..35: row chairs, parking, wall, curtain
wall, railing) are kept only in the vector file for now.

Output (same layout as colab/data/split_dota.py, so every VRDet tool works unchanged):
  {out}/images/{split}/{name}.png, labels/, meta/{split}.jsonl, gt/{split}/{name}.txt (DOTA format),
  thumbs/{split}/{name}.jpg, classes.json, vectors/{split}/{name}.npz (primitive tokens for the vector branch).

python colab/data/floorplancad.py --root /content/datasets [--size 1024]
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

GDRIVE_ID = "1wsOQxIXjsqYzMlUpPNRjyQiMnwgVbtJG"      # FloorPlanCAD SVG ground truth (~132 MB), as used by SymPoint/VecFormer
FPC_CLASSES = ("single-door", "double-door", "sliding-door", "folding-door", "revolving-door", "rolling-door",
               "window", "bay-window", "blind-window", "opening-symbol", "sofa", "bed", "chair", "table",
               "tv-cabinet", "wardrobe", "cabinet", "gas-stove", "sink", "refrigerator", "airconditioner", "bath",
               "bath-tub", "washing-machine", "squat-toilet", "urinal", "toilet", "stairs", "elevator", "escalator")
VIEW = 140.0
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TYPES = {"line": 0, "arc": 1, "circle": 2, "ellipse": 3}


def arc_points(x1, y1, rx, ry, phi, fa, fs, x2, y2, n=12):
    """SVG elliptical arc (endpoint parameterisation, W3C SVG 1.1 F.6.5) -> n points."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return np.array([[x1, y1], [x2, y2]])
    phi = math.radians(phi)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    co = math.sqrt(max(num, 0) / den) if den > 0 else 0.0
    if fa == fs:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx, cy = cp * cxp - sp * cyp + (x1 + x2) / 2, sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    t = t1 + dt * np.linspace(0, 1, n)
    return np.stack([cx + rx * np.cos(t) * cp - ry * np.sin(t) * sp, cy + rx * np.cos(t) * sp + ry * np.sin(t) * cp], 1)


def parse_svg(path):
    """-> list of dict(type, pts (k,2) in viewBox units, rgb, width, sem, inst)."""
    prims = []
    for el in ET.parse(path).iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("path", "circle", "ellipse"):
            continue
        a = el.attrib
        m = re.findall(r"\d+", a.get("stroke", "rgb(0,0,0)"))
        rgb = tuple(int(v) for v in m[:3]) if len(m) >= 3 else (0, 0, 0)
        width = float(a.get("stroke-width", 0.1))
        sem = int(a.get("semanticId", 0) or 0)
        inst = int(a.get("instanceId", -1) or -1)
        base = dict(rgb=rgb, width=width, sem=sem, inst=inst)
        if tag == "path":
            toks = re.findall(r"[MLA]|" + NUM, a.get("d", ""))
            i, cur, start = 0, None, None
            while i < len(toks):
                c = toks[i]
                if c == "M":
                    cur = (float(toks[i + 1]), float(toks[i + 2]))
                    i += 3
                elif c == "L":
                    nxt = (float(toks[i + 1]), float(toks[i + 2]))
                    prims.append(dict(base, type="line", pts=np.array([cur, nxt])))
                    cur = nxt
                    i += 3
                elif c == "A":
                    rx, ry, phi, fa, fs, x, y = (float(v) for v in toks[i + 1:i + 8])
                    prims.append(dict(base, type="arc", pts=arc_points(cur[0], cur[1], rx, ry, phi, int(fa), int(fs), x, y)))
                    cur = (x, y)
                    i += 8
                else:
                    i += 1
        else:
            cx, cy = float(a.get("cx", 0)), float(a.get("cy", 0))
            rx = float(a.get("r", a.get("rx", 0)))
            ry = float(a.get("r", a.get("ry", 0)))
            t = np.linspace(0, 2 * math.pi, 25)
            pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
            rot = re.match(r"rotate\(\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\)", a.get("transform", ""))
            if rot:
                # FloorPlanCAD ellipses carry rotate(a, ox, oy) with (ox, oy) from the source CAD frame (often
                # outside the 140x140 block) while cx, cy are already final: keep only the orientation.
                ang = float(rot.group(1))
                c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                d = pts - (cx, cy)
                pts = np.stack([d[:, 0] * c - d[:, 1] * s + cx, d[:, 0] * s + d[:, 1] * c + cy], 1)
            prims.append(dict(base, type="circle" if tag == "circle" else "ellipse", pts=pts))
    return prims


def render(prims, size):
    k = size / VIEW
    img = np.full((size, size, 3), 255, np.uint8)
    for p in prims:
        pts = np.round(p["pts"] * k * 16).astype(np.int32)          # 4 fractional bits for sub-pixel lines
        th = max(1, int(round(p["width"] * k)))
        col = (int(p["rgb"][2]), int(p["rgb"][1]), int(p["rgb"][0]))
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA, shift=4)
    return img


def objects(prims, size, min_side=2.0):
    k = size / VIEW
    groups = {}
    for p in prims:
        if 1 <= p["sem"] <= 30 and p["inst"] >= 0:
            groups.setdefault((p["sem"], p["inst"]), []).append(p["pts"])
    out = []
    for (sem, _), pts in sorted(groups.items()):
        pts = np.concatenate(pts) * k
        (cx, cy), (w, h), a = cv2.minAreaRect(pts.astype(np.float32))
        w, h = max(w, min_side), max(h, min_side)
        poly = cv2.boxPoints(((cx, cy), (w, h), a)).reshape(-1)
        out.append((sem - 1, np.clip(poly, 0, size)))
    return out


def vector_tokens(prims):
    """Fixed-size primitive tokens: type, 8 points resampled along the primitive (normalised), rgb, width, ids."""
    rows = []
    for p in prims:
        pts = p["pts"]
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        tq = np.linspace(0, seg[-1], 8) if seg[-1] > 0 else np.zeros(8)
        rs = np.stack([np.interp(tq, seg, pts[:, 0]), np.interp(tq, seg, pts[:, 1])], 1) / VIEW
        rows.append(np.r_[TYPES[p["type"]], rs.reshape(-1), np.array(p["rgb"]) / 255.0, p["width"], p["sem"], p["inst"]])
    return np.array(rows, np.float32).reshape(-1, 1 + 16 + 3 + 1 + 2)


def convert_one(svg, out, split, size):
    name = Path(svg).stem
    prims = parse_svg(svg)
    img = render(prims, size)
    objs = objects(prims, size)
    cv2.imwrite(str(out / "images" / split / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(out / "thumbs" / split / f"{name}.jpg"), cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA))
    np.savez_compressed(out / "vectors" / split / f"{name}.npz", tokens=vector_tokens(prims))
    with open(out / "labels" / split / f"{name}.txt", "w") as f:
        for c, q in objs:
            f.write(f"{c} " + " ".join(f"{v / size:.6f}" for v in q) + "\n")
    with open(out / "gt" / split / f"{name}.txt", "w") as f:
        for c, q in objs:
            f.write(" ".join(f"{v:.1f}" for v in q) + f" {FPC_CLASSES[c]} 0\n")
    return {"name": name, "src": name, "rate": 1.0, "x0": 0, "y0": 0, "w": size, "h": size, "img_w": size,
            "img_h": size, "file": f"{name}.png", "objs": [[c, 0, 0] + [round(float(v), 2) for v in q] for c, q in objs]}


def fetch(raw):
    z = raw / "FloorPlanCAD.zip"
    if not z.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run(["gdown", GDRIVE_ID, "-O", str(z)], check=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(raw)
    for inner in list(raw.rglob("*.zip")):
        if inner != z:
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/datasets")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--src", default=None, help="existing dir containing train/val/test SVG folders (skip download)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    root = Path(a.root)
    out = root / f"floorplancad_{a.size}"
    raw = Path(a.src) if a.src else root / "floorplancad_raw"
    if not a.src and not (raw / ".fetched").exists():
        raw.mkdir(parents=True, exist_ok=True)
        fetch(raw)
        (raw / ".fetched").write_text("ok")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(list(FPC_CLASSES)))
    for split in ("train", "val", "test"):
        if (out / f".done_{split}").exists():
            continue
        svgs = sorted(p for p in raw.rglob("*.svg") if split in [q.lower() for q in p.parts[len(raw.parts):-1]])
        if a.limit:
            svgs = svgs[:a.limit]
        if not svgs:
            print(f"[fpc] no SVGs for split {split}", flush=True)
            continue
        for d in ("images", "labels", "gt", "thumbs", "vectors"):
            shutil.rmtree(out / d / split, ignore_errors=True)
            (out / d / split).mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(exist_ok=True)
        with Pool(a.workers) as pool, open(out / "meta" / f"{split}.jsonl", "w") as mf:
            n_obj = 0
            for m in pool.imap_unordered(partial(convert_one, out=out, split=split, size=a.size), map(str, svgs), chunksize=8):
                mf.write(json.dumps(m) + "\n")
                n_obj += len(m["objs"])
        (out / f".done_{split}").write_text("ok")
        print(f"[fpc] {split}: {len(svgs)} drawings, {n_obj} objects", flush=True)


if __name__ == "__main__":
    main()
