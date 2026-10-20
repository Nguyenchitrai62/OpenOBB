"""Wall_Color label statistics for the v10 argument: sizes, anchor contention at P3/P4, overlap between classes."""
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = str(Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color")
NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
S = 1280


def load(split):
    out = []
    for lp in sorted(glob.glob(os.path.join(ROOT, "labels", split, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ims = glob.glob(os.path.join(ROOT, "images", split, stem + ".*"))
        if not ims:
            continue
        im = cv2.imdecode(np.fromfile(ims[0], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        h, w = im.shape[0] * 8, im.shape[1] * 8
        sc = S / max(h, w)
        rows = []
        for line in open(lp):
            v = line.split()
            if len(v) < 9:
                continue
            p = np.array(v[1:9], float).reshape(4, 2) * [w * sc, h * sc]
            (cx, cy), (bw, bh), a = cv2.minAreaRect(p.astype(np.float32))
            if bw < bh:
                bw, bh, a = bh, bw, a + 90
            rows.append((int(v[0]), cx, cy, bw, bh, np.deg2rad(a)))
        out.append((stem, np.array(rows, float).reshape(-1, 6), (h * sc, w * sc)))
    return out


def inside(px, py, g, min_side=8.0, band=0.0):
    c, s = np.cos(g[5]), np.sin(g[5])
    dx, dy = px - g[1], py - g[2]
    u, v = dx * c + dy * s, -dx * s + dy * c
    hw = max(g[3], min_side) / 2
    hh = max(max(g[4], min_side) / 2, band / 2)
    return (np.abs(u) <= hw) & (np.abs(v) <= hh)


for split in ("train", "val"):
    data = load(split)
    allr = np.concatenate([d[1] for d in data])
    print(f"\n== {split}: {len(data)} images, {len(allr)} boxes")
    for k, n in enumerate(NAMES):
        r = allr[allr[:, 0] == k]
        if not len(r):
            continue
        print(f"{n:12s} n={len(r):5d} imgs={sum((d[1][:, 0] == k).any() for d in data):3d} "
              f"len p50={np.median(r[:, 3]):6.1f} thick p50={np.median(r[:, 4]):5.1f} "
              f"len p10/p90={np.percentile(r[:, 3], 10):6.1f}/{np.percentile(r[:, 3], 90):6.1f}")
    if split != "train":
        continue
    # anchor contention: for each GT, points of P3 / P4 inside it (assign rule: min_side 8, across band 1 stride);
    # fraction of those points that are also inside a GT of ANOTHER class (contested), and with which class
    for stride in (8, 16):
        print(f"\n-- stride {stride}: own points / contested share / main rival")
        stats = {k: [0, 0, 0, {}] for k in range(7)}
        for _, r, (H, W) in data:
            if not len(r):
                continue
            xs = (np.arange(int(np.ceil(W / stride))) + 0.5) * stride
            ys = (np.arange(int(np.ceil(H / stride))) + 0.5) * stride
            PX, PY = np.meshgrid(xs, ys)
            masks = [inside(PX, PY, g, band=stride) for g in r]
            for i, g in enumerate(r):
                k = int(g[0])
                m = masks[i]
                stats[k][0] += 1
                stats[k][1] += m.sum()
                for j, g2 in enumerate(r):
                    if j == i or int(g2[0]) == k:
                        continue
                    ov = (m & masks[j]).sum()
                    if ov:
                        stats[k][3][NAMES[int(g2[0])]] = stats[k][3].get(NAMES[int(g2[0])], 0) + ov
                cont = m.copy() & False
                for j, g2 in enumerate(r):
                    if j != i and int(g2[0]) != k:
                        cont |= masks[j]
                stats[k][2] += (m & cont).sum()
        for k, (n, pts, cont, riv) in stats.items():
            if n:
                top = sorted(riv.items(), key=lambda t: -t[1])[:2]
                print(f"{NAMES[k]:12s} {pts / n:7.1f} pts/obj  contested {cont / max(pts, 1) * 100:5.1f}%  {top}")
    # nearest GT of another class (centre distance) and containment in a wall box
    print("\n-- neighbours (train)")
    for k in (3, 4, 5):
        dmin, cont_wall = [], []
        for _, r, _ in data:
            for g in r[r[:, 0] == k]:
                o = r[r[:, 0] != k]
                if len(o):
                    d = np.hypot(o[:, 1] - g[1], o[:, 2] - g[2])
                    j = d.argmin()
                    dmin.append((d[j], NAMES[int(o[j, 0])]))
                walls = r[r[:, 0] == 0]
                c = sum(inside(np.array([g[1]]), np.array([g[2]]), w, min_side=0)[0] for w in walls)
                cont_wall.append(c > 0)
        if dmin:
            ds = np.array([d for d, _ in dmin])
            cls_n = {}
            for _, c in dmin:
                cls_n[c] = cls_n.get(c, 0) + 1
            print(f"{NAMES[k]:12s} nearest other-class centre p50={np.median(ds):5.1f}px p90={np.percentile(ds, 90):5.1f}"
                  f"  nearest class {cls_n}  centre inside a wall box {np.mean(cont_wall) * 100:4.0f}%")
