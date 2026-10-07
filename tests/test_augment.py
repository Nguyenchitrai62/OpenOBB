import json
import random

import cv2
import numpy as np
import torch

from vrdet.data.dota import DotaPatches
from vrdet.ops.obb import obb2poly

COL = {0: (40, 40, 230), 6: (40, 220, 40)}


def _make_split(root, n=6, S=256):
    rng = np.random.default_rng(0)
    (root / "images" / "train").mkdir(parents=True)
    (root / "meta").mkdir()
    with open(root / "meta" / "train.jsonl", "w") as f:
        for k in range(n):
            img = np.full((S, S, 3), 120, np.uint8)
            objs = []
            for j, c in enumerate([0, 6, 0]):
                ob = np.array([60 + 70 * j, 60 + 60 * (k % 3), 50, 20, rng.uniform(-1.5, 1.5)])
                p = obb2poly(ob)[0]
                cv2.fillPoly(img, [p.reshape(4, 2).astype(np.int32)], COL[c])
                objs.append([c, 0, 0] + [round(float(v), 2) for v in p])
            name = f"I{k}__1.0__0___0"
            cv2.imwrite(str(root / "images" / "train" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            f.write(json.dumps({"name": name, "src": f"I{k}", "rate": 1.0, "x0": 0, "y0": 0, "w": S, "h": S,
                                "objs": objs}) + "\n")


def _colour_hit(img_chw, boxes, labels, S):
    img = img_chw.permute(1, 2, 0).numpy()[..., ::-1]                 # back to BGR HWC
    hits = []
    for b, c in zip(boxes.numpy(), labels.numpy()):
        bb = b.copy()
        bb[:4] *= S
        bb[2:4] *= 0.6                                                # inner part only (antialiasing)
        mask = np.zeros((S, S), np.uint8)
        cv2.fillPoly(mask, [obb2poly(bb)[0].reshape(4, 2).astype(np.int32)], 1)
        pix = img[mask > 0].astype(float)
        hits.append(np.abs(pix - np.array(COL[int(c)])).sum(1).mean() < 60)
    return np.mean(hits) if hits else 1.0


def test_rotation_and_mosaic_keep_boxes_on_objects(tmp_path):
    _make_split(tmp_path)
    random.seed(0)
    np.random.seed(0)
    for kw in (dict(rotate_p=1.0), dict(mosaic_p=1.0), dict()):
        ds = DotaPatches(tmp_path, "train", size=256, augment=True, hsv=(0, 0, 0), **kw)
        for i in range(len(ds)):
            img, t = ds[i]
            assert img.shape == (3, 256, 256) and img.dtype == torch.uint8
            assert len(t["boxes"]) > 0
            assert _colour_hit(img, t["boxes"], t["labels"], 256) > 0.9, kw
