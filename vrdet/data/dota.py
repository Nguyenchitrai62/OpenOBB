"""Dataset over the patch split produced by colab/data/split_dota.py.

Targets per image: dict(labels=int64 (N,), boxes=float32 (N, 5) = (cx, cy, w, h, theta) with
cx, cy, w, h normalised by the (square) patch size and theta in le90 radians).
Images are returned as uint8 CHW RGB tensors; normalisation happens on the GPU.
With vectors=True (CAD data with {root}/vectors/{split}/{name}.npz) the target also carries "vec" (M, 22):
type, 8 points / image size, rgb, width / image size, CAD layer id (0 = unknown), transformed together with the
image by every augmentation.
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
SQUARE_NAMES = ("storage-tank", "roundabout")    # DOTA: only 90-degree rotations (RTMDet-R / O2 practice)


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
    inside = (polys >= 0).all(1) & (polys <= S).all(1)
    if inside.all():
        return polys.astype(np.float32), labels
    g = shapely.polygons(polys.reshape(-1, 4, 2).astype(np.float64))
    bad = ~shapely.is_valid(g)
    if bad.any():
        g[bad] = shapely.convex_hull(g[bad])
    area = shapely.area(g)
    inter = shapely.intersection(g, shapely.box(0, 0, S, S))
    iof = np.where(area > 0, shapely.area(inter) / np.maximum(area, 1e-9), 0)
    iof[inside] = 1.0
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
                 mosaic_p=0.0, context=False, thumb=512, ctx_dropout=0.0, vectors=False, max_tokens=4096,
                 scale_jitter=0.0, translate=0.0, mosaic_mode="half", layer_drop=0.0, aug_iof=0.7, cache=False,
                 fliplr=0.5, flipud=0.5):
        self.root, self.split, self.size = Path(root), split, size
        # H11 (YOLO recipe idea): random scale in [1 - scale_jitter, 1 + scale_jitter] and translation of
        # +-translate x size on every training sample; mosaic_mode "yolo" = 4 full-resolution patches around a
        # random centre on a 2S canvas, then the same scale/translate warp to S (objects keep their size)
        self.scale_jitter, self.translate, self.mosaic_mode = scale_jitter, translate, mosaic_mode
        # visible fraction an object cut by a mosaic/zoom crop needs to keep its label (a visible but unlabelled
        # part is trained as background; the one-stage recipe keeps even small pieces)
        self.aug_iof = aug_iof
        self.layer_drop = layer_drop    # H16: chance to merge all CAD layers into one (do not over-rely on layers)
        self.vectors, self.max_tokens = vectors, max_tokens
        self.context, self.thumb, self.ctx_dropout = context, thumb, ctx_dropout
        self.rotate_p, self.mosaic_p = rotate_p, mosaic_p
        self.augment, self.min_size, self.hsv, self.rot90, self.flip = augment, min_size, hsv, rot90, flip
        self.fliplr, self.flipud = (fliplr, flipud) if flip else (0.0, 0.0)
        items = [json.loads(l) for l in open(self.root / "meta" / f"{split}.jsonl")]
        items.sort(key=lambda m: m["name"])
        if filter_empty:
            items = [m for m in items if m["objs"]]
        if limit:
            items = items[:limit]
        self.items = items
        try:                                     # by name: class ids of another dataset must not match DOTA's
            self.square_ids = [i for i, n in enumerate(dataset_classes(root)) if n in SQUARE_NAMES]
        except Exception:  # noqa: BLE001
            self.square_ids = []
        self.cache = {}
        if cache:                               # decode once in the main process; forked workers share the pages
            from concurrent.futures import ThreadPoolExecutor
            files = [self.root / "images" / self.split / m.get("file", f"{m['name']}.jpg") for m in self.items]
            with ThreadPoolExecutor(8) as ex:
                imgs = list(ex.map(lambda f: cv2.imread(str(f), cv2.IMREAD_COLOR), files))
            self.cache = {m["name"]: im for m, im in zip(self.items, imgs)}
        self.keep_difficult = keep_difficult

    def __len__(self):
        return len(self.items)

    def _load(self, m):
        img = self.cache.get(m["name"])
        if img is None:
            img = cv2.imread(str(self.root / "images" / self.split / m.get("file", f"{m['name']}.jpg")), cv2.IMREAD_COLOR)
        objs = m["objs"] if self.keep_difficult else [o for o in m["objs"] if o[1] == 0]
        polys = np.array([o[3:] for o in objs], dtype=np.float32).reshape(-1, 8)
        labels = np.array([o[0] for o in objs], dtype=np.int64)
        return img, polys, labels

    def _load_vec(self, m):
        """Primitive tokens in pixel units: pts (M, 16) and attr (M, 6) = type, r, g, b, width, layer id."""
        z = np.load(self.root / "vectors" / self.split / f"{m['name']}.npz")
        t = z["tokens"].astype(np.float32)
        view = float(z["view"]) if "view" in z.files else 140.0     # FloorPlanCAD: widths in viewBox units
        S = self.size
        layer = z["layer"].astype(np.float32)[:, None] + 1 if "layer" in z.files else np.zeros((len(t), 1), np.float32)
        attr = np.concatenate([t[:, :1], t[:, 17:20], t[:, 20:21] / view * S, np.maximum(layer, 0)], 1)
        return {"pts": t[:, 1:17] * S, "attr": attr}

    def _vec_out(self, vec):
        """Drop primitives that left the image, cap the count, normalise -> (M, 22) float32 tensor."""
        S = self.size
        pts, attr = vec["pts"], vec["attr"]
        x, y = pts[:, 0::2], pts[:, 1::2]
        inside = ((x >= 0) & (x <= S) & (y >= 0) & (y <= S)).any(1)
        pts, attr = pts[inside], attr[inside]
        if len(pts) > self.max_tokens:
            sel = (np.sort(np.random.choice(len(pts), self.max_tokens, replace=False)) if self.augment
                   else np.arange(self.max_tokens))
            pts, attr = pts[sel], attr[sel]
        out = np.concatenate([attr[:, :1], pts / S, attr[:, 1:4], attr[:, 4:5] / S, attr[:, 5:6]], 1)
        return torch.from_numpy(out.astype(np.float32).reshape(-1, 22))

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

    def _rotate(self, img, polys, labels, ctx=None, vec=None):
        """Arbitrary-angle rotation about the centre (images with square-like classes: 90-degree steps only)."""
        S = img.shape[0]
        if self.square_ids and np.isin(labels, self.square_ids).any():
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
        if vec is not None and len(vec["pts"]):
            vec["pts"] = (vec["pts"].reshape(-1, 2) @ M[:, :2].T + M[:, 2]).reshape(-1, 16).astype(np.float32)
        if len(polys):
            pts = polys.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            polys, labels = clip_to_window(pts.reshape(-1, 8).astype(np.float32), labels, S)
        return img, polys, labels

    def _warp(self, img, polys, labels, vec, src_centre):
        """Random scale/translate warp of `img` (S x S or a 2S x 2S mosaic canvas) to an S x S output; polygons are
        re-clipped with the iof rule, vector points and line widths follow."""
        S = self.size
        s = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
        tx, ty = (random.uniform(-self.translate, self.translate) * S for _ in range(2))
        M = np.array([[s, 0, S / 2 + tx - s * src_centre], [0, s, S / 2 + ty - s * src_centre]], np.float32)
        if s < 1:                               # zoom-out: area resize first so 1-2 px lines are not broken up
            h, w = img.shape[:2]
            small = cv2.resize(np.ascontiguousarray(img), (max(1, round(w * s)), max(1, round(h * s))),
                               interpolation=cv2.INTER_AREA)
            T = np.array([[1, 0, M[0, 2]], [0, 1, M[1, 2]]], np.float32)
            img = cv2.warpAffine(small, T, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        else:
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=PAD_BGR)
        if vec is not None and len(vec["pts"]):
            vec["pts"] = (vec["pts"].reshape(-1, 2) @ M[:, :2].T + M[:, 2]).reshape(-1, 16).astype(np.float32)
            vec["attr"][:, 4] *= s
        if len(polys):
            pts = polys.reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            polys, labels = clip_to_window(pts.reshape(-1, 8).astype(np.float32), labels, S, self.aug_iof)
        return img, polys, labels

    def _mosaic_yolo(self, i):
        """Four full-resolution patches meeting at a random centre of a 2S canvas, then one scale/translate warp."""
        S = self.size
        canvas = np.empty((2 * S, 2 * S, 3), np.uint8)
        canvas[:] = PAD_BGR
        xc, yc = (int(random.uniform(0.5 * S, 1.5 * S)) for _ in range(2))
        P, L, V = [], [], []
        for q, j in enumerate([i] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, polys, labels = self._load(self.items[j])
            vec = self._load_vec(self.items[j]) if self.vectors else None
            img, polys = self._augment(img, polys, color=False, vec=vec)
            h, w = img.shape[:2]
            if q == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif q == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * S), yc
                x1b, y1b = 0, h - (y2a - y1a)
            elif q == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * S)
                x1b, y1b = w - (x2a - x1a), 0
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * S), min(yc + h, 2 * S)
                x1b, y1b = 0, 0
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
            ox, oy = x1a - x1b, y1a - y1b
            if vec is not None:
                vec["pts"][:, 0::2] += ox
                vec["pts"][:, 1::2] += oy
                vec["attr"][:, 5] += q * 10000 * (vec["attr"][:, 5] > 0)     # keep the 4 sources' layers apart
                V.append(vec)
            if len(polys):
                pp = polys.copy()
                pp[:, 0::2] += ox
                pp[:, 1::2] += oy
                P.append(pp)
                L.append(labels)
        polys = np.concatenate(P) if P else np.zeros((0, 8), np.float32)
        labels = np.concatenate(L) if L else np.zeros(0, np.int64)
        vec = ({"pts": np.concatenate([v["pts"] for v in V]), "attr": np.concatenate([v["attr"] for v in V])}
               if V else None)
        if len(polys):                          # objects cut by the patch borders inside the canvas
            polys, labels = clip_to_window(polys, labels, 2 * S, self.aug_iof)
        img, polys, labels = self._warp(canvas, polys, labels, vec, src_centre=S)
        return img, polys, labels, vec

    def _mosaic(self, i):
        """Four patches at half scale, each with its own random flip/rot90 (oriented dense one-to-one, RiO-DETR)."""
        S, h = self.size, self.size // 2
        canvas = np.empty((S, S, 3), np.uint8)
        canvas[:] = PAD_BGR
        P, L, V = [], [], []
        for q, j in enumerate([i] + [random.randrange(len(self.items)) for _ in range(3)]):
            img, polys, labels = self._load(self.items[j])
            vec = self._load_vec(self.items[j]) if self.vectors else None
            img, polys = self._augment(img, polys, color=False, vec=vec)
            ox, oy = (q % 2) * h, (q // 2) * h
            if vec is not None:
                vp = vec["pts"] * 0.5
                vp[:, 0::2] += ox
                vp[:, 1::2] += oy
                vec["attr"][:, 4] *= 0.5
                vec["attr"][:, 5] += q * 10000 * (vec["attr"][:, 5] > 0)
                V.append({"pts": vp, "attr": vec["attr"]})
            canvas[oy:oy + h, ox:ox + h] = cv2.resize(np.ascontiguousarray(img), (h, h), interpolation=cv2.INTER_AREA)
            if len(polys):
                pp = polys * 0.5
                pp[:, 0::2] += ox
                pp[:, 1::2] += oy
                P.append(pp)
                L.append(labels)
        polys = np.concatenate(P) if P else np.zeros((0, 8), np.float32)
        labels = np.concatenate(L) if L else np.zeros(0, np.int64)
        vec = ({"pts": np.concatenate([v["pts"] for v in V]), "attr": np.concatenate([v["attr"] for v in V])}
               if V else None)
        return canvas, polys, labels, vec

    def _augment(self, img, polys, color=True, ctx=None, vec=None):
        """Flips / rot90 applied identically to the patch, the vector points and the context thumbnail + tile box."""
        vp = vec["pts"] if vec is not None else np.zeros((0, 16), np.float32)
        S = img.shape[0]
        T = ctx["thumb"].shape[0] if ctx is not None else 0
        if random.random() < self.fliplr:
            img = img[:, ::-1]
            polys[:, 0::2] = S - polys[:, 0::2]
            vp[:, 0::2] = S - vp[:, 0::2]
            if ctx is not None:
                ctx["thumb"] = ctx["thumb"][:, ::-1]
                x0, y0, x1, y1 = ctx["tile"]
                ctx["tile"] = np.array([T - x1, y0, T - x0, y1], np.float32)
        if random.random() < self.flipud:
            img = img[::-1]
            polys[:, 1::2] = S - polys[:, 1::2]
            vp[:, 1::2] = S - vp[:, 1::2]
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
                x = vp[:, 0::2].copy()
                vp[:, 0::2] = vp[:, 1::2]
                vp[:, 1::2] = S - x
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
        lut_hue = ((x + (r[0] - 1) * 180) % 180).astype(np.uint8)          # additive hue shift
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))),
                            cv2.COLOR_HSV2BGR)

    def __getitem__(self, i):
        m = self.items[i]
        ctx = None
        vec = None
        if self.augment and self.mosaic_p and random.random() < self.mosaic_p:
            img, polys, labels, vec = self._mosaic_yolo(i) if self.mosaic_mode == "yolo" else self._mosaic(i)
            img = self._hsv(img)
            if self.context:                    # a mosaic has no single parent image: no context
                ctx = self._load_ctx(m)
                ctx["valid"] = False
        else:
            img, polys, labels = self._load(m)
            vec = self._load_vec(m) if self.vectors else None
            if self.context:
                ctx = self._load_ctx(m)
                if self.augment and self.ctx_dropout and random.random() < self.ctx_dropout:
                    ctx["valid"] = False         # context dropout: do not let scene priors dominate rare classes
            if self.augment:
                img, polys = self._augment(img, polys, ctx=ctx, vec=vec)
                if self.rotate_p and random.random() < self.rotate_p:
                    img, polys, labels = self._rotate(img, polys, labels, ctx=ctx, vec=vec)
                if self.scale_jitter or self.translate:
                    img, polys, labels = self._warp(img, polys, labels, vec, src_centre=self.size / 2)
                    if ctx is not None:
                        ctx["valid"] = False     # the thumbnail tile box does not follow the warp
        obb = polys_to_obb(polys)
        # thin lines (1-2 px walls/pipes) keep their label with the short side clamped to min_size; only boxes
        # that vanished in both directions are dropped (deleting them made them permanent misses at eval)
        keep = np.maximum(obb[:, 2], obb[:, 3]) >= self.min_size if len(obb) else np.zeros(0, bool)
        obb, labels = obb[keep], labels[keep]
        obb[:, 2:4] = np.maximum(obb[:, 2:4], self.min_size)
        obb[:, :4] /= self.size
        img = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))      # BGR HWC -> RGB CHW
        tgt = {"labels": torch.from_numpy(labels), "boxes": torch.from_numpy(obb), "name": m["name"]}
        if ctx is not None:
            tgt["thumb"] = torch.from_numpy(np.ascontiguousarray(ctx["thumb"][..., ::-1].transpose(2, 0, 1)))
            tgt["tile"] = torch.from_numpy(np.asarray(ctx["tile"], np.float32))
            tgt["ctx_valid"] = bool(ctx["valid"])
        if self.vectors:
            if self.augment and self.layer_drop and random.random() < self.layer_drop:
                vec["attr"][:, 5] = np.minimum(vec["attr"][:, 5], 1.0)
            tgt["vec"] = self._vec_out(vec)
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
