"""Prediction results with the Ultralytics OBB interface, in numpy:

    r = model.predict("page.png")[0]
    r.obb.xywhr (N, 5) px, rotation in [0, pi/2);  r.obb.xyxyxyxy (N, 4, 2);  r.obb.xyxyxyxyn;  r.obb.xyxy
    r.obb.conf (N,), r.obb.cls (N,), r.obb.data (N, 7) = [xywhr, conf, cls]
    r.names, r.orig_img (BGR), r.orig_shape, r.path, r.speed (ms)
    r.plot(), r.save(), r.save_txt(), r.summary(), r.to_json(), r.verbose()
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .ops import xywhr_to_corners

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
           (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
           (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
           (200, 149, 255), (199, 55, 255)]                                   # BGR, one per class id


def imread(path):
    """BGR uint8 image (unicode-safe paths, EXIF orientation applied), or None."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


class OBB:
    """Oriented boxes of one image (data rows: cx, cy, w, h, r, conf, cls)."""

    def __init__(self, data, orig_shape):
        data = np.asarray(data, np.float32).reshape(-1, 7)
        self.data, self.orig_shape = data, tuple(orig_shape)

    xywhr = property(lambda self: self.data[:, :5])
    conf = property(lambda self: self.data[:, 5])
    cls = property(lambda self: self.data[:, 6])
    xyxyxyxy = property(lambda self: xywhr_to_corners(self.xywhr))
    shape = property(lambda self: self.data.shape)

    @property
    def xyxyxyxyn(self):
        h, w = self.orig_shape[:2]
        return self.xyxyxyxy / np.array([w, h], np.float32)

    @property
    def xyxy(self):
        p = self.xyxyxyxy
        return np.concatenate([p.min(1), p.max(1)], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return OBB(self.data[idx], self.orig_shape)


class Results:
    """Detections of one image."""

    def __init__(self, orig_img, path, names, obb=None, speed=None, save_dir=None):
        self.orig_img, self.path, self.names = orig_img, str(path), dict(names)
        self.orig_shape = tuple(orig_img.shape[:2])
        self.obb = obb if obb is not None else OBB(np.zeros((0, 7)), self.orig_shape)
        self.speed = speed or {"preprocess": None, "inference": None, "postprocess": None}
        self.save_dir = save_dir

    def __len__(self):
        return len(self.obb)

    def __getitem__(self, idx):
        return Results(self.orig_img, self.path, self.names, self.obb[idx], self.speed, self.save_dir)

    def verbose(self):
        if not len(self):
            return "(no detections), "
        counts = np.bincount(self.obb.cls.astype(int), minlength=len(self.names))
        return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts.tolist()) if n > 0)

    def summary(self, normalize=False, decimals=5):
        """[{"name", "class", "confidence", "box": {"x1", "y1", ..., "x4", "y4"}}]"""
        pts = self.obb.xyxyxyxyn if normalize else self.obb.xyxyxyxy
        rows = []
        for p, c, s in zip(pts, self.obb.cls, self.obb.conf):
            box = {}
            for k in range(4):
                box[f"x{k + 1}"] = round(float(p[k, 0]), decimals)
                box[f"y{k + 1}"] = round(float(p[k, 1]), decimals)
            rows.append({"name": self.names[int(c)], "class": int(c), "confidence": round(float(s), decimals),
                         "box": box})
        return rows

    def to_json(self, normalize=False, decimals=5):
        return json.dumps(self.summary(normalize, decimals), indent=2, ensure_ascii=False)

    def save_txt(self, txt_file, save_conf=False):
        """YOLO-OBB label lines 'cls x1 y1 ... x4 y4 [conf]', normalised; appended to txt_file."""
        lines = [f"{int(c)} " + " ".join(f"{v:g}" for v in p) + (f" {float(s):g}" if save_conf else "")
                 for p, c, s in zip(self.obb.xyxyxyxyn.reshape(-1, 8), self.obb.cls, self.obb.conf)]
        if lines:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
        return str(txt_file)

    def plot(self, conf=True, line_width=None, labels=True, boxes=True, img=None, save=False, filename=None):
        """Annotated BGR image with the oriented boxes and 'name conf' labels."""
        im = np.ascontiguousarray((self.orig_img if img is None else img).copy())
        if boxes and len(self):
            lw = int(line_width or max(round(sum(im.shape) / 2 * 0.003), 2))
            tf, fs = max(lw - 1, 1), lw / 3
            for p, c, s in zip(self.obb.xyxyxyxy, self.obb.cls, self.obb.conf):
                col = PALETTE[int(c) % len(PALETTE)]
                pts = np.round(p).astype(np.int32)
                cv2.polylines(im, [pts], True, col, lw, cv2.LINE_AA)
                if labels:
                    text = self.names[int(c)] + (f" {float(s):.2f}" if conf else "")
                    x, y = (int(v) for v in pts[0])
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)
                    y0 = y - th - 3 if y - th - 3 >= 0 else y
                    cv2.rectangle(im, (x, y0), (x + tw, y0 + th + 3), col, -1, cv2.LINE_AA)
                    cv2.putText(im, text, (x, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), tf,
                                cv2.LINE_AA)
        if save:
            imwrite(filename or f"results_{Path(self.path).name}", im)
        return im

    def save(self, filename=None, **kw):
        filename = filename or f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, **kw)
        return str(filename)

    def __repr__(self):
        return (f"Results: {self.path} {self.orig_shape[1]}x{self.orig_shape[0]}, {len(self)} oriented boxes, "
                f"names: {self.names}")
