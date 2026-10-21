"""Prediction results with the interface of the PyTorch package's results (and of Ultralytics OBB results), in numpy, so
inference code written for a .pt model runs unchanged on the exported .onnx:

    r = model.predict("page.png")[0]
    r.obb.xywhr        (N, 5) cx, cy, w, h, rotation in [0, pi/2) radians, pixels of the original image
    r.obb.xyxyxyxy     (N, 4, 2) corners;  r.obb.xyxyxyxyn normalised;  r.obb.xyxy enclosing boxes
    r.obb.conf (N,), r.obb.cls (N,), r.obb.data (N, 7) = [xywhr, conf, cls], r.obb.id = None
    r.names, r.orig_img (BGR), r.orig_shape, r.path, r.speed (ms), r.save_dir
    r.plot(), r.save(), r.show(), r.save_txt(), r.save_crop(), r.summary(), r.to_json(), r.to_df(), r.to_csv(),
    r.verbose(), r.new(), r.update(obb=...), r.cpu() / .numpy() / .cuda() / .to()
The arrays answer the tensor calls such code makes (.cpu(), .numpy(), .cuda(), .to(...), .detach(), .tolist(), .int()...):
`r.obb.xywhr.cpu().numpy()` works as with the .pt results.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .ops import xywhr_to_corners

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
           (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
           (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
           (200, 149, 255), (199, 55, 255)]                                   # BGR, one per class id


class Array(np.ndarray):
    """numpy array that also answers the tensor methods used on PyTorch results: device moves are no-ops, .numpy()
    gives the plain array."""

    def __new__(cls, a, dtype=None):
        return np.asarray(a, dtype=dtype).view(cls)

    def cpu(self):
        return self

    def cuda(self, *_, **__):
        return self

    def to(self, *_, **__):
        return self

    def detach(self):
        return self

    def numpy(self, *_, **__):
        return self.view(np.ndarray)

    def clone(self):
        return self.copy()

    def int(self):
        return self.astype(np.int32)

    def long(self):
        return self.astype(np.int64)

    def float(self):
        return self.astype(np.float32)

    def numel(self):
        return int(self.size)

    def dim(self):
        return self.ndim

    @property
    def device(self):
        return "cpu"

    @property
    def is_cuda(self):
        return False


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
        self.data = Array(np.asarray(data, np.float32).reshape(-1, 7))
        self.orig_shape = tuple(orig_shape)
        self.id, self.is_track = None, False

    xywhr = property(lambda self: self.data[:, :5])
    conf = property(lambda self: self.data[:, 5])
    cls = property(lambda self: self.data[:, 6])
    shape = property(lambda self: self.data.shape)

    @property
    def xyxyxyxy(self):
        return Array(xywhr_to_corners(self.xywhr.numpy()))

    @property
    def xyxyxyxyn(self):
        h, w = self.orig_shape[:2]
        return Array(self.xyxyxyxy.numpy() / np.array([w, h], np.float32))

    @property
    def xyxy(self):
        p = self.xyxyxyxy.numpy()
        return Array(np.concatenate([p.min(1), p.max(1)], -1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return OBB(self.data[idx], self.orig_shape)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def cuda(self):
        return self

    def to(self, *_, **__):
        return self

    def __repr__(self):
        return f"OBB object with {len(self)} boxes\ndata: {self.data.numpy()!r}"


class Results:
    """Detections of one image."""

    def __init__(self, orig_img, path, names, obb=None, speed=None, save_dir=None):
        self.orig_img, self.path, self.names = orig_img, str(path), dict(names)
        self.orig_shape = tuple(orig_img.shape[:2])
        self.obb = obb if obb is not None else OBB(np.zeros((0, 7)), self.orig_shape)
        self.speed = speed or {"preprocess": None, "inference": None, "postprocess": None}
        self.save_dir = save_dir
        self.boxes = self.masks = self.probs = self.keypoints = None        # OBB task: only .obb is set

    def __len__(self):
        return len(self.obb)

    def __getitem__(self, idx):
        return self._new(self.obb[idx])

    def _new(self, obb):
        return Results(self.orig_img, self.path, self.names, obb, self.speed, self.save_dir)

    def new(self):
        """Empty Results of the same image."""
        return Results(self.orig_img, self.path, self.names, None, self.speed, self.save_dir)

    def update(self, obb=None, **_):
        """Replace the boxes: obb is (N, 7) [cx, cy, w, h, r, conf, cls] (array, tensor-like) or an OBB object."""
        if obb is not None:
            self.obb = obb if isinstance(obb, OBB) else OBB(np.asarray(obb), self.orig_shape)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def cuda(self):
        return self

    def to(self, *_, **__):
        return self

    def verbose(self):
        """'3 walls, 1 door, ' (or '(no detections), ')."""
        if not len(self):
            return "(no detections), "
        counts = np.bincount(self.obb.cls.numpy().astype(int), minlength=len(self.names))
        return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts.tolist()) if n > 0)

    def summary(self, normalize=False, decimals=5):
        """[{"name", "class", "confidence", "box": {"x1", "y1", ..., "x4", "y4"}}]"""
        pts = (self.obb.xyxyxyxyn if normalize else self.obb.xyxyxyxy).numpy()
        rows = []
        for p, c, s in zip(pts, self.obb.cls.numpy(), self.obb.conf.numpy()):
            box = {}
            for k in range(4):
                box[f"x{k + 1}"] = round(float(p[k, 0]), decimals)
                box[f"y{k + 1}"] = round(float(p[k, 1]), decimals)
            rows.append({"name": self.names[int(c)], "class": int(c), "confidence": round(float(s), decimals),
                         "box": box})
        return rows

    def to_json(self, normalize=False, decimals=5):
        return json.dumps(self.summary(normalize, decimals), indent=2, ensure_ascii=False)

    def to_df(self, normalize=False, decimals=5):
        import pandas as pd
        return pd.DataFrame(self.summary(normalize, decimals))

    def to_csv(self, normalize=False, decimals=5, *args, **kw):
        return self.to_df(normalize, decimals).to_csv(*args, **kw)

    def save_txt(self, txt_file, save_conf=False):
        """YOLO-OBB label lines 'cls x1 y1 ... x4 y4 [conf]', normalised; appended to txt_file. Nothing is written for
        an image without detections."""
        lines = [f"{int(c)} " + " ".join(f"{v:g}" for v in p) + (f" {float(s):g}" if save_conf else "")
                 for p, c, s in zip(self.obb.xyxyxyxyn.numpy().reshape(-1, 8), self.obb.cls.numpy(),
                                    self.obb.conf.numpy())]
        if lines:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
        return str(txt_file)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """Rotated crop of every box -> save_dir/<class name>/<file_name>.jpg (numbered when the name exists)."""
        for (cx, cy, w, h, r), c in zip(self.obb.xywhr.numpy(), self.obb.cls.numpy()):
            W, H = max(int(round(w)), 1), max(int(round(h)), 1)
            m = cv2.getRotationMatrix2D((float(cx), float(cy)), math.degrees(float(r)), 1.0)
            m[:, 2] += (W / 2 - cx, H / 2 - cy)
            crop = cv2.warpAffine(self.orig_img, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            d = Path(save_dir) / self.names[int(c)]
            d.mkdir(parents=True, exist_ok=True)
            stem, f, k = Path(file_name).stem, d / f"{Path(file_name).stem}.jpg", 1
            while f.exists():
                k += 1
                f = d / f"{stem}{k}.jpg"
            imwrite(f, crop)

    def plot(self, conf=True, line_width=None, font_size=None, font=None, pil=False, img=None, im_gpu=None,
             kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True, show=False, save=False,
             filename=None, color_mode="class", txt_color=(255, 255, 255)):
        """Annotated BGR image with the oriented boxes and 'name conf' labels."""
        im = np.ascontiguousarray((self.orig_img if img is None else img).copy())
        if boxes and len(self):
            lw = int(line_width or max(round(sum(im.shape) / 2 * 0.003), 2))
            tf = max(lw - 1, 1)
            fs = font_size / 10 if font_size else lw / 3
            for i, (p, c, s) in enumerate(zip(self.obb.xyxyxyxy.numpy(), self.obb.cls.numpy(), self.obb.conf.numpy())):
                col = PALETTE[(i if color_mode == "instance" else int(c)) % len(PALETTE)]
                pts = np.round(p).astype(np.int32)
                cv2.polylines(im, [pts], True, col, lw, cv2.LINE_AA)
                if labels:
                    text = self.names[int(c)] + (f" {float(s):.2f}" if conf else "")
                    x, y = (int(v) for v in pts[0])
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)
                    y0 = y - th - 3 if y - th - 3 >= 0 else y
                    cv2.rectangle(im, (x, y0), (x + tw, y0 + th + 3), col, -1, cv2.LINE_AA)
                    cv2.putText(im, text, (x, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, tf, cv2.LINE_AA)
        if show:
            self._show(im)
        if save:
            imwrite(filename or f"results_{Path(self.path).name}", im)
        return im

    def save(self, filename=None, *args, **kw):
        filename = filename or f"results_{Path(self.path).name}"
        self.plot(*args, save=True, filename=filename, **kw)
        return str(filename)

    def show(self, *args, **kw):
        self.plot(*args, show=True, **kw)

    def _show(self, im):
        try:
            from PIL import Image
            Image.fromarray(im[..., ::-1]).show()
        except Exception:  # noqa: BLE001
            cv2.imshow(Path(self.path).name, im)
            cv2.waitKey(0)

    def __repr__(self):
        return (f"Results object: {self.path} {self.orig_shape[1]}x{self.orig_shape[0]}, {len(self)} oriented boxes, "
                f"names: {self.names}")
