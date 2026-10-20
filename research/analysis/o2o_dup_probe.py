"""Why does the one-to-one head still give duplicates without NMS (31-37% at conf 0.25, F79-F93) while YOLO26's is
NMS-free? For confident one-to-one candidates (score >= 0.25, before refiner / NMS) matched to the same label
(same class, ProbIoU >= 0.5), classify each duplicate pair: same class group or not, pyramid levels, anchor distance in
strides, score ratio, and the label shape (long / short). Also the one-to-many head for comparison. CPU.

usage: python o2o_dup_probe.py <checkpoint.pt> [n_images]
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import probiou  # noqa: E402
from openobb.predict import PAD_BGR, rect_shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "dataset_wall_color"
ck, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = OpenOBB(ck)
net = m.model
names = [m.names[i] for i in range(len(m.names))]


def canon(b):
    b = np.asarray(b, np.float64).reshape(-1, 5).copy()
    sw = b[:, 2] < b[:, 3]
    b[sw, 2], b[sw, 3] = b[sw, 3], b[sw, 2].copy()
    b[sw, 4] += np.pi / 2
    return b


stats = {h: Counter() for h in ("o2o", "o2m")}
for p in sorted((ROOT / "images" / "val").iterdir())[:n]:
    img = imread(str(p))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = rect_shape(*img.shape[:2], 1280)
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    lab = ROOT / "labels" / "val" / f"{p.stem}.txt"
    g = []
    for line in lab.read_text().splitlines():
        v = line.split()
        if len(v) >= 9:
            pts4 = (np.array(v[1:9], float) * np.tile([w0 * s, h0 * s], 4)).reshape(4, 2).astype(np.float32)
            (cx, cy), (w, h), a = cv2.minAreaRect(pts4)
            g.append((int(v[0]), canon([cx, cy, w, h, np.deg2rad(a)])[0]))
    gb = np.array([b for _, b in g])
    gc = np.array([c for c, _ in g])
    with torch.inference_mode():
        f4, feats = net.encode(xt)
        pts, st = net.anchors(feats)
        A = pts.shape[0]
        for head, o2o in (("o2o", True), ("o2m", False)):
            cls, reg = net.run_head(feats, o2o)
            cl, boxes = net.candidates(cls, reg, pts, st)
            sc, lb = cl[0].sigmoid().max(-1)
            k = (sc >= 0.25).nonzero(as_tuple=True)[0].numpy()
            pb = canon(boxes[0][k].numpy())
            psc, plb = sc[k].numpy(), lb[k].numpy()
            grp, anc = k // A, k % A
            lvl, ap = st[anc].numpy(), pts[anc].numpy()
            iou = probiou(gb, pb) * (gc[:, None] == plb[None, :])
            st_ = stats[head]
            st_["dets"] += len(k)
            for i in range(len(gb)):
                js = np.nonzero(iou[i] >= 0.5)[0]
                if len(js) < 2:
                    continue
                js = js[np.argsort(-psc[js])]
                st_["labels with dup"] += 1
                st_["dup dets"] += len(js) - 1
                L = gb[i, 2] / max(gb[i, 3], 1e-6)
                for j in js[1:]:
                    b = js[0]
                    st_["dup: same group" if grp[j] == grp[b] else "dup: other group"] += 1
                    st_["dup: same level" if lvl[j] == lvl[b] else "dup: other level"] += 1
                    d = np.linalg.norm(ap[j] - ap[b]) / max(lvl[b], lvl[j])
                    st_["dup: anchors <= 1.5 stride apart" if d <= 1.5 else "dup: anchors > 1.5 stride apart"] += 1
                    st_["dup: label long (>= 10x)" if L >= 10 else "dup: label short"] += 1
                    st_["dup: score ratio >= 0.5" if psc[j] >= 0.5 * psc[b] else "dup: score ratio < 0.5"] += 1
    print(p.stem, {h: (stats[h]["dets"], stats[h]["dup dets"]) for h in stats}, flush=True)

for h, c in stats.items():
    d = max(c["dup dets"], 1)
    print(f"\n{h}: {c['dets']} confident candidates, {c['dup dets']} duplicates ({c['dup dets'] / max(c['dets'], 1):.1%}) on "
          f"{c['labels with dup']} labels")
    for key in sorted(k for k in c if k.startswith("dup:")):
        print(f"  {key:36s} {c[key] / d:6.1%}")
