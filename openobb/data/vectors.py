"""Page-level vector tokens -> per-tile tokens.

Page npz (tools/pdf_vectors.py, or any CAD/SVG source), all in full-image pixel coordinates:
  tokens (N, 21) float32: type, 8 x (x, y) points resampled along the primitive, r, g, b in [0, 1], line width (px)
  layer  (N,)    int32:   CAD layer index per primitive (-1 = none); layer_names (L,) optional
Types: 0 line, 1 arc, 2 circle, 3 ellipse, 4 bezier curve (others reserved).
"""
import numpy as np


def _inside_any(pts, x0, y0, size):
    x, y = pts[:, 0::2] - x0, pts[:, 1::2] - y0
    return ((x >= 0) & (x <= size) & (y >= 0) & (y <= size)).any(1)


def cut_tile(page, x0, y0, size):
    """Model input (M, 22): type, points / size (tile-relative), rgb, width / size, layer + 1 (0 = none)."""
    t = np.asarray(page["tokens"], np.float32)
    if len(t) == 0:
        return np.zeros((0, 22), np.float32)
    keep = _inside_any(t[:, 1:17], x0, y0, size)
    t = t[keep]
    pts = t[:, 1:17].copy()
    pts[:, 0::2] -= x0
    pts[:, 1::2] -= y0
    layer = np.asarray(page.get("layer", np.full(len(keep), -1)), np.float32)[keep]
    return np.concatenate([t[:, :1], pts / size, t[:, 17:20], t[:, 20:21] / size, (layer + 1)[:, None]],
                          1).astype(np.float32)


def tile_npz(page, x0, y0, size, path):
    """Write a tile in the training layout read by DotaPatches(vectors=True) (same as FloorPlanCAD tiles)."""
    t = np.asarray(page["tokens"], np.float32)
    layer = np.asarray(page.get("layer", np.full(len(t), -1)), np.int32)
    keep = _inside_any(t[:, 1:17], x0, y0, size) if len(t) else np.zeros(0, bool)
    t, layer = t[keep], layer[keep]
    pts = t[:, 1:17].copy()
    pts[:, 0::2] -= x0
    pts[:, 1::2] -= y0
    rows = np.concatenate([t[:, :1], pts / size, t[:, 17:20], t[:, 20:21], np.zeros((len(t), 2), np.float32)], 1)
    np.savez_compressed(path, tokens=rows.astype(np.float32).reshape(-1, 23), layer=layer,
                        view=np.float32(size))
