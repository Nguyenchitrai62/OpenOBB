"""Contention-aware class groups for dense oriented heads (OpenOBB19, docs/OPENOBB19.md, LEDGER F72).

A dense head gives each grid point ONE box: every object must own a point that no other object claims. On drawings,
objects touch and overlap at their ends (wall stubs inside junction boxes, door ends on walls): on Wall_Color 60% of
junctions and 86% of short walls / wall_300 have no point of their own at stride 8 (F72), whatever the grid
(stride 4: still 72%). Giving every point one box PER CLASS GROUP removes the contention between groups. The groups
are chosen from the training labels: classes that claim the same points go to different groups (greedy colouring
of the class contention matrix), so the head adapts to any dataset; with one group the head is the usual one.
"""
import numpy as np
import torch

STRIDE = 8               # the finest grid of the dense heads (P3)
MIN_SIDE = 8.0           # candidate region of an object: its box, each side at least MIN_SIDE px (assign_slots)


def contention_matrix(items, nc, stride=STRIDE, min_side=MIN_SIDE, max_images=400):
    """items: prepared dataset items ({"w", "h", "objs": [[cls, ., ., x1, y1, ..., x4, y4], ...]}) -> (nc, nc) counts:
    W[i, j] = number of (object of class i, grid point in its candidate region also claimed by an object of class j),
    j != the object itself."""
    import cv2
    W = np.zeros((nc, nc), np.float64)
    for m in items[:max_images]:
        objs = np.asarray(m.get("objs") or [], np.float64)
        if not len(objs):
            continue
        w, h = int(m.get("w", 1024)), int(m.get("h", 1024))
        xs = (np.arange(int(np.ceil(w / stride))) + 0.5) * stride
        ys = (np.arange(int(np.ceil(h / stride))) + 0.5) * stride
        gx, gy = np.meshgrid(xs, ys)
        P = np.stack([gx.ravel(), gy.ravel()], 1)
        cls = objs[:, 0].astype(int)
        claims = np.zeros((len(objs), len(P)), bool)
        for k, o in enumerate(objs):
            (cx, cy), (a, b), ang = cv2.minAreaRect(o[3:11].reshape(4, 2).astype(np.float32))
            th = np.deg2rad(ang)
            u = (P[:, 0] - cx) * np.cos(th) + (P[:, 1] - cy) * np.sin(th)
            v = -(P[:, 0] - cx) * np.sin(th) + (P[:, 1] - cy) * np.cos(th)
            claims[k] = (np.abs(u) <= max(a, min_side) / 2) & (np.abs(v) <= max(b, min_side) / 2)
        per_class = np.stack([claims[cls == c].sum(0) for c in range(nc)])            # (nc, P) claim counts
        for k in range(len(objs)):
            pts = claims[k]
            if not pts.any():
                continue
            others = per_class[:, pts].sum(1).astype(np.float64)
            others[cls[k]] -= pts.sum()                                                # not the object itself
            W[cls[k]] += others
    return W


def contention_groups(items, nc, n_groups=3, **kw):
    """-> list of nc group ids: greedy colouring of the symmetric contention matrix, the most contended classes
    first, each class into the group where it collides least with the classes already there."""
    W = contention_matrix(items, nc, **kw)
    S = W + W.T
    np.fill_diagonal(S, 0.0)
    order = np.argsort(-S.sum(1))
    groups = [-1] * nc
    for c in order:
        cost = [sum(S[c, d] for d in range(nc) if groups[d] == g) for g in range(n_groups)]
        size = [sum(1 for d in range(nc) if groups[d] == g) for g in range(n_groups)]
        groups[c] = int(min(range(n_groups), key=lambda g: (cost[g], size[g])))
    return groups


def parse_groups(v, nc):
    """'0,0,1,...' / list / None -> list of nc ints (None: one group)."""
    if v is None or v == "" or v == "none":
        return None
    g = [int(x) for x in (v.split(",") if isinstance(v, str) else v)]
    if len(g) != nc:
        raise ValueError(f"groups needs one id per class ({nc}), got {len(g)}")
    return g if max(g) > 0 else None


def grouped_candidates(cls, boxes_g, class_group):
    """cls (B, A, C) logits, boxes_g (B, A, G, 5) px, class_group (C,) -> (B, G*A, C) logits with the classes of
    other groups at -30, (B, G*A, 5) boxes: every (point, group) becomes one candidate for the class-wise NMS."""
    B, A, G, _ = boxes_g.shape
    mask = class_group[None, :] == torch.arange(G, device=cls.device)[:, None]                 # (G, C)
    c = cls[:, None].expand(B, G, A, cls.shape[-1]).masked_fill(~mask[None, :, None, :], -30.0)
    return c.reshape(B, G * A, -1), boxes_g.permute(0, 2, 1, 3).reshape(B, G * A, 5)
