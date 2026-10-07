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
import shapely
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)
PAD_BGR = (104, 116, 124)
SQUARE_CLASSES = (9, 11)     # storage-tank, roundabout: only 90-degree rotations (RTMDet-R / O2 practice)


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


def clip_to_window(polys, labels, S, iof_thr=0.7):
    """Keep objects with area(obj & window)/area(obj) >= iof_thr; truncated ones become the min-area rectangle
    of their visible part (same rule as colab/data/split_dota.py)."""
    if len(polys) == 0:
        return polys, labels
    g = shapely.polygons(polys.reshape(-1, 4, 2).astype(np.float64))
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.convex_hull(g[bad])
    area = shapely.area(g)
    inter = shapely.intersection(g, shapely.box(0, 0, S, S))
    iof = np.where(area > 0, shapely.area(inter) / np.maximum(area, 1e-9), 0)
    out_p, out_l = [], []
    for k in np.nonzero(iof >= iof_thr)[0]:
        if iof[k] > 0.999:
            q = polys[k]
        else:
            pts = np.asarray(shapely.get_coordinates(inter[k]), dtype=np.float32)
            q = cv2.boxPoints(cv2.minAreaRect(pts)).reshape(-1)
        out_p.append(np.clip(q, 0, S))
        out_l.append(labels[k])
    if not out_p:
        return np.zeros((0, 8), np.float32), np.zeros(0, np.int64)
    return np.array(out_p, np.float32), np.array(out_l, np.int64)


class DotaPatches(Dataset):
    def __init__(self, root, split, size=1024, augment=False, filter_empty=False, min_size=2.0,
                 hsv=(0.015, 0.5, 0.3), rot90=True, flip=True, limit=None, keep_difficult=True, rotate_p=0.0,
                 mosaic_p=0.0, context=False, thumb=512, ctx_dropout=0.0):
        self.root, self.split, self.size = Path(root), split, size
        self.context, self.thumb, self.ctx_dropout = context, thumb, ctx_dropout
        self.rotate_p, self.mosaic_p = rotate_p, mosaic_p
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
        img = cv2.imread(str(self.root / "images" / self.split / m.get("file", f"{m['name']}.jpg")), cv2.IMREAD_COLOR)
        objs = m["objs"] if self.keep_difficult else [o for o in m["objs"] if o[1] == 0]
        polys = np.array([o[3:] for o in objs], dtype=np.float32).reshape(-1, 8)
        labels = np.array([o[0] for o in objs], dtype=np.int64)
        return img, polys, labels

    def _load_ctx(self, m):
        """Whole-image thumbnail on a THUMB x THUMB canvas + the tile's box on that canvas (global context, H6)."""
        T = self.thumb
        canvas = np.empty((T, T, 3), np.uint8)
        canvas[:] = PAD_BGR
        th = cv2.imread(str(self.root / "thumbs" / self.split / f"{m['src']}.jpg"), cv2.IMREAD_COLOR)
        if th is None or "img_w" not in m:
            return {"thumb": canvas, "tile": np.array([0, 0, T, T], np.float32), "valid": False}
        h, w = th.shape[:2]
        canvas[:min(h, T), :min(w, T)] = th[:T, :T]
        ts = T / max(m["img_w"], m["img_h"])
        r = m["rate"]
        tile = np.array([m["x0"] / r, m["y0"] / r, (m["x0"] + self.size) / r, (m["y0"] + self.size) / r]) * ts
        return {"thumb": canvas, "tile": tile.astype(np.float32), "valid": True}

    @staticmethod
    def _ctx_rot90(ctx):
        T = ctx["thumb"].shape[0]
        ctx["thumb"] = np.rot90(ctx["thumb"])
        x0, y0, x1, y1 = ctx["tile"]
        ctx["tile"] = np.array([y0, T - x1, y1, T - x0], np.float32)     # (x, y) -> (y, T - x)

    def _rotate(self, img, polys, labels, ctx=None):
        """Arbitrary-angle rotation about the centre (images with square-like classes: 90-degree steps only)."""
        S = img.shape[0]
        if np.isin(labels, SQUARE_CLASSES).any():
            ang = 90.0 * random.randint(0, 3)
        else:
            ang = random.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        if ctx is not None:                     # rotate the thumbnail about the tile centre: the tile stays aligned
            T = ctx["thumb"].shape[0]
            c = ((ctx["tile"][0] + ctx["tile"][2]) / 2, (ctx["tile"][1] + ctx["tile"][3]) / 2)
            Mt = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            ctx["thumb"] = cv2.warpAffine(np.ascontiguousarray(ctx["thumb"]), Mt, (T, T), flags=cv2.INTER_LINEAR,
                                          borderValue=PAD_BGR)
        if len(polys):
            pts = polys.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            polys, labels = clip_to_window(pts.reshape(-1, 8).astype(np.float32), labels, S)
        return img, polys, labels

    def _mosaic(self, i):
        """Four patches at half scale, each with its own random flip/rot90 (oriented dense one-to-one, RiO-DETR)."""
        S, h = self.size, self.size // 2
        canvas = np.empty((S, S, 3), np.uint8)
        canvas[:] = PAD_BGR
        P, L = [], []
        for q, j in enumerate([i] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, polys, labels = self._load(self.items[j])
            img, polys = self._augment(img, polys, color=False)
            ox, oy = (q % 2) * h, (q // 2) * h
            canvas[oy:oy + h, ox:ox + h] = cv2.resize(np.ascontiguousarray(img), (h, h), interpolation=cv2.INTER_AREA)
            if len(polys):
                pp = polys * 0.5
                pp[:, 0::2] += ox
                pp[:, 1::2] += oy
                P.append(pp)
                L.append(labels)
        polys = np.concatenate(P) if P else np.zeros((0, 8), np.float32)
        labels = np.concatenate(L) if L else np.zeros(0, np.int64)
        return canvas, polys, labels

    def _augment(self, img, polys, color=True, ctx=None):
        """Flips / rot90 applied identically to the patch and (if given) to the context thumbnail + tile box."""
        S = img.shape[0]
        T = ctx["thumb"].shape[0] if ctx is not None else 0
        if self.flip and random.random() < 0.5:
            img = img[:, ::-1]
            polys[:, 0::2] = S - polys[:, 0::2]
            if ctx is not None:
                ctx["thumb"] = ctx["thumb"][:, ::-1]
                x0, y0, x1, y1 = ctx["tile"]
                ctx["tile"] = np.array([T - x1, y0, T - x0, y1], np.float32)
        if self.flip and random.random() < 0.5:
            img = img[::-1]
            polys[:, 1::2] = S - polys[:, 1::2]
            if ctx is not None:
                ctx["thumb"] = ctx["thumb"][::-1]
                x0, y0, x1, y1 = ctx["tile"]
                ctx["tile"] = np.array([x0, T - y1, x1, T - y0], np.float32)
        if self.rot90:
            k = random.randint(0, 3)
            for _ in range(k):                      # 90 deg counter-clockwise: (x, y) -> (y, S - x)
                img = np.rot90(img)
                x = polys[:, 0::2].copy()
                polys[:, 0::2] = polys[:, 1::2]
                polys[:, 1::2] = S - x
                if ctx is not None:
                    self._ctx_rot90(ctx)
        if color:
            img = self._hsv(img)
        return img, polys

    def _hsv(self, img):
        if not (self.hsv and any(self.hsv)):
            return img
        r = np.random.uniform(-1, 1, 3) * self.hsv + 1
        hue, sat, val = cv2.split(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))),
                            cv2.COLOR_HSV2BGR)

    def __getitem__(self, i):
        m = self.items[i]
        ctx = None
        if self.augment and self.mosaic_p and random.random() < self.mosaic_p:
            img, polys, labels = self._mosaic(i)
            img = self._hsv(img)
            if self.context:                    # a mosaic has no single parent image: no context
                ctx = self._load_ctx(m)
                ctx["valid"] = False
        else:
            img, polys, labels = self._load(m)
            if self.context:
                ctx = self._load_ctx(m)
                if self.augment and self.ctx_dropout and random.random() < self.ctx_dropout:
                    ctx["valid"] = False         # context dropout: do not let scene priors dominate rare classes
            if self.augment:
                img, polys = self._augment(img, polys, ctx=ctx)
                if self.rotate_p and random.random() < self.rotate_p:
                    img, polys, labels = self._rotate(img, polys, labels, ctx=ctx)
        obb = polys_to_obb(polys)
        keep = (obb[:, 2] >= self.min_size) & (obb[:, 3] >= self.min_size) if len(obb) else np.zeros(0, bool)
        obb, labels = obb[keep], labels[keep]
        obb[:, :4] /= self.size
        img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))      # BGR HWC -> RGB CHW
        tgt = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(obb), "name": m["name"]}
        if ctx is not None:
            tgt["thumb"] = torch.from_numpy(np.ascontiguousarray(ctx["thumb"][..., ::-1].transpose(2, 0, 1)))
            tgt["tile"] = torch.from_numpy(np.asarray(ctx["tile"], np.float32))
            tgt["ctx_valid"] = bool(ctx["valid"])
        return torch.from_numpy(img), tgt


def repeat_factors(items, num_classes, t):
    """Repeat-factor sampling (LVIS, Gupta et al. 2019): r_c = max(1, sqrt(t / f_c)), f_c = fraction of patches
    containing class c; a patch is drawn with weight max_c r_c over its classes (1 for empty patches)."""
    import numpy as np
    n = max(len(items), 1)
    has = [set(o[0] for o in m["objs"]) for m in items]
    f = np.array([sum(c in h for h in has) / n for c in range(num_classes)])
    r = np.where(f > 0, np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-12))), 1.0)
    w = np.array([max([r[c] for c in h], default=1.0) for h in has])
    return w, f, r


def dataset_classes(root):
    """Class names of a prepared split ({root}/classes.json), DOTA-v1.0 by default."""
    from vrdet.eval.dota import DOTA1_CLASSES
    p = Path(root) / "classes.json"
    return tuple(json.loads(p.read_text())) if p.exists() else DOTA1_CLASSES


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]
