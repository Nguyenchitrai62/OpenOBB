"""Dataset over the patch split produced by colab/data/split_dota.py.

Targets per image: dict(labels=int64 (N,), boxes=float32 (N, 5) = (cx, cy, w, h, theta) with
cx, cy, w, h normalised by the (square) patch size and theta in le90 radians).
Images are returned as uint8 CHW RGB tensors; normalisation happens on the GPU.
"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)


def polys_to_obb(polys):
    """(N, 8) pixel polygons -> (N, 5) le90 OBBs in pixels via min-area rectangles."""
    out = np.zeros((len(polys), 5), dtype=np.float32)
    for i, p in enumerate(polys):
        (cx, cy), (w, h), a = cv2.minAreaRect(p.reshape(4, 2).astype(np.float32))
        t = math.radians(a)
        if w < h:
            w, h, t = h, w, t + math.pi / 2
        t = (t + math.pi / 2) % math.pi - math.pi / 2
        out[i] = cx, cy, w, h, t
    return out


class DotaPatches(Dataset):
    def __init__(self, root, split, size=1024, augment=False, filter_empty=False, min_size=2.0,
                 hsv=(0.015, 0.5, 0.3), rot90=True, flip=True, limit=None, keep_difficult=True):
        self.root, self.split, self.size = Path(root), split, size
        self.augment, self.min_size, self.hsv, self.rot90, self.flip = augment, min_size, hsv, rot90, flip
        items = [json.loads(l) for l in open(self.root / "meta" / f"{split}.jsonl")]
        items.sort(key=lambda m: m["name"])
        if filter_empty:
            items = [m for m in items if m["objs"]]
        if limit:
            items = items[:limit]
        self.items = items
        self.keep_difficult = keep_difficult

    def __len__(self):
        return len(self.items)

    def _load(self, m):
        img = cv2.imread(str(self.root / "images" / self.split / f"{m['name']}.jpg"), cv2.IMREAD_COLOR)
        objs = m["objs"] if self.keep_difficult else [o for o in m["objs"] if o[1] == 0]
        polys = np.array([o[3:] for o in objs], dtype=np.float32).reshape(-1, 8)
        labels = np.array([o[0] for o in objs], dtype=np.int64)
        return img, polys, labels

    def _augment(self, img, polys):
        S = img.shape[0]
        if self.flip and random.random() < 0.5:
            img = img[:, ::-1]
            polys[:, 0::2] = S - polys[:, 0::2]
        if self.flip and random.random() < 0.5:
            img = img[::-1]
            polys[:, 1::2] = S - polys[:, 1::2]
        if self.rot90:
            k = random.randint(0, 3)
            for _ in range(k):                      # 90 deg counter-clockwise: (x, y) -> (y, S - x)
                img = np.rot90(img)
                x = polys[:, 0::2].copy()
                polys[:, 0::2] = polys[:, 1::2]
                polys[:, 1::2] = S - x
        if self.hsv and any(self.hsv):
            r = np.random.uniform(-1, 1, 3) * self.hsv + 1
            hue, sat, val = cv2.split(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV))
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(np.uint8)
            lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
            lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
            img = cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))),
                               cv2.COLOR_HSV2BGR)
        return img, polys

    def __getitem__(self, i):
        m = self.items[i]
        img, polys, labels = self._load(m)
        if self.augment:
            img, polys = self._augment(img, polys)
        obb = polys_to_obb(polys)
        keep = (obb[:, 2] >= self.min_size) & (obb[:, 3] >= self.min_size) if len(obb) else np.zeros(0, bool)
        obb, labels = obb[keep], labels[keep]
        obb[:, :4] /= self.size
        img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))      # BGR HWC -> RGB CHW
        tgt = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(obb), "name": m["name"]}
        return torch.from_numpy(img), tgt


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]
