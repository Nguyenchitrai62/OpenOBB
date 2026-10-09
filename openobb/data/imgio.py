"""Image file I/O used everywhere (training data, validation, prediction), so every path decodes images the same way.

Decoding follows the Ultralytics loaders: bytes read with numpy (unicode-safe on Windows) and decoded with
cv2.IMREAD_COLOR, i.e. 3-channel BGR uint8, EXIF orientation applied, alpha dropped, grey expanded to 3 channels,
16-bit images reduced to 8 bits.
"""
from pathlib import Path

import cv2
import numpy as np

# image suffixes accepted as dataset / prediction sources (the formats OpenCV decodes)
IMG_EXT = (".bmp", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm")


def imread(path, flags=cv2.IMREAD_COLOR):
    """-> BGR uint8 image, or None when the file is missing or cannot be decoded (as cv2.imread)."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None):
    """cv2.imwrite that also works with non-ASCII paths on Windows; returns True on success."""
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img, params or [])
    if ok:
        buf.tofile(str(path))
    return bool(ok)
