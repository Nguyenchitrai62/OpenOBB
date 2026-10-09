"""Prediction results with the same interface as the Ultralytics OBB results, so code written for YOLO reads OpenOBB
output unchanged (OpenOBB's own implementation of that public interface; no Ultralytics code):

    r = model.predict("page.png")[0]
    r.obb.xywhr        (N, 5) cx, cy, w, h, rotation in radians, regularised as Ultralytics: rotation in [0, pi/2)
                       (w and h swapped when needed), pixels of the original image
    r.obb.xyxyxyxy     (N, 4, 2) corner points, pixels;  r.obb.xyxyxyxyn  normalised by the image size
    r.obb.xyxy         (N, 4) axis-aligned enclosing boxes
    r.obb.conf (N,), r.obb.cls (N,), r.obb.data (N, 7) = [xywhr, conf, cls], r.obb.id = None
    r.names {id: name}, r.orig_img (BGR), r.orig_shape (h, w), r.path, r.speed {ms}, r.save_dir
    r.plot(), r.save(), r.show(), r.save_txt(), r.save_crop(), r.summary(), r.to_json(), r.to_df(), r.to_csv(),
    r.verbose(), r.new(), r.update(obb=...), r.cpu() / .numpy() / .cuda() / .to()
Tensors live on the model's device, as with Ultralytics (use .cpu() / .numpy() before numpy code).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from openobb.data.imgio import imwrite

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
           (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
           (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
           (200, 149, 255), (199, 55, 255)]                                   # BGR, one per class id


def _xp(t):
    return torch if isinstance(t, torch.Tensor) else np


def regularize_xywhr(b):
    """(N, 5) cx, cy, w, h, r -> the same boxes with r in [0, pi/2): w and h are swapped when r mod pi >= pi/2
    (the convention of Ultralytics OBB outputs)."""
    xp = _xp(b)
    t = b[:, 4] % math.pi
    swap = t >= math.pi / 2
    w = xp.where(swap, b[:, 3], b[:, 2])
    h = xp.where(swap, b[:, 2], b[:, 3])
    return xp.stack([b[:, 0], b[:, 1], w, h, t % (math.pi / 2)], -1)


def xywhr_to_corners(b):
    """(N, 5) cx, cy, w, h, r -> (N, 4, 2) corners (torch or numpy), in the Ultralytics corner order."""
    xp = _xp(b)
    c, s = xp.cos(b[:, 4]), xp.sin(b[:, 4])
    ux, uy = b[:, 2] / 2 * c, b[:, 2] / 2 * s                 # half length along the box axis
    vx, vy = -b[:, 3] / 2 * s, b[:, 3] / 2 * c                # half width across
    cx, cy = b[:, 0], b[:, 1]
    pts = [(cx + ux + vx, cy + uy + vy), (cx + ux - vx, cy + uy - vy),
           (cx - ux - vx, cy - uy - vy), (cx - ux + vx, cy - uy + vy)]
    return xp.stack([xp.stack(p, -1) for p in pts], 1)


class OBB:
    """Oriented boxes of one image (data rows: cx, cy, w, h, r, conf, cls)."""

    def __init__(self, data, orig_shape):
        if data.ndim == 1:
            data = data[None]
        self.data, self.orig_shape = data, tuple(orig_shape)
        self.id, self.is_track = None, False

    # ---- fields
    @property
    def xywhr(self):
        return self.data[:, :5]

    @property
    def conf(self):
        return self.data[:, 5]

    @property
    def cls(self):
        return self.data[:, 6]

    @property
    def xyxyxyxy(self):
        return xywhr_to_corners(self.xywhr)

    @property
    def xyxyxyxyn(self):
        p = self.xyxyxyxy
        h, w = self.orig_shape[:2]
        scale = p.new_tensor([w, h]) if isinstance(p, torch.Tensor) else np.array([w, h], p.dtype)
        return p / scale

    @property
    def xyxy(self):
        p = self.xyxyxyxy
        if isinstance(p, torch.Tensor):
            return torch.cat([p.amin(1), p.amax(1)], -1)
        return np.concatenate([p.min(1), p.max(1)], -1)

    @property
    def shape(self):
        return self.data.shape

    # ---- container / device helpers
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return OBB(self.data[idx], self.orig_shape)

    def cpu(self):
        return OBB(self.data.cpu(), self.orig_shape) if isinstance(self.data, torch.Tensor) else self

    def numpy(self):
        return OBB(self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data, self.orig_shape)

    def cuda(self):
        return self.to("cuda")

    def to(self, *args, **kw):
        d = self.data if isinstance(self.data, torch.Tensor) else torch.as_tensor(self.data)
        return OBB(d.to(*args, **kw), self.orig_shape)

    def __repr__(self):
        return f"openobb.results.OBB object with {len(self)} boxes\ndata: {self.data!r}"


class Results:
    """Detections of one image."""

    def __init__(self, orig_img, path, names, obb=None, speed=None, save_dir=None):
        self.orig_img, self.path, self.names = orig_img, str(path), dict(names)
        self.orig_shape = tuple(orig_img.shape[:2])
        self.obb = obb if obb is not None else OBB(torch.zeros((0, 7)), self.orig_shape)
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
        """Empty Results of the same image (as Ultralytics)."""
        return Results(self.orig_img, self.path, self.names, None, self.speed, self.save_dir)

    def update(self, obb=None, **_):
        """Replace the boxes: obb is (N, 7) [cx, cy, w, h, r, conf, cls] (tensor or array) or an OBB object."""
        if obb is not None:
            self.obb = obb if isinstance(obb, OBB) else OBB(torch.as_tensor(obb), self.orig_shape)

    def cpu(self):
        return self._new(self.obb.cpu())

    def numpy(self):
        return self._new(self.obb.numpy())

    def cuda(self):
        return self._new(self.obb.cuda())

    def to(self, *args, **kw):
        return self._new(self.obb.to(*args, **kw))

    def verbose(self):
        """'3 walls, 1 door, ' (or '(no detections), ')."""
        if not len(self):
            return "(no detections), "
        cls = np.asarray(self.obb.cpu().cls).astype(int)
        counts = np.bincount(cls, minlength=len(self.names))
        return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts.tolist()) if n > 0)

    def summary(self, normalize=False, decimals=5):
        """[{"name", "class", "confidence", "box": {"x1", "y1", ..., "x4", "y4"}}]"""
        o = self.obb.numpy()
        pts = o.xyxyxyxyn if normalize else o.xyxyxyxy
        rows = []
        for p, c, s in zip(pts, o.cls, o.conf):
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
        """YOLO-OBB label lines 'cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]', coordinates normalised; appended to txt_file.
        Nothing is written for an image without detections (as Ultralytics)."""
        o = self.obb.numpy()
        lines = []
        for p, c, s in zip(o.xyxyxyxyn.reshape(-1, 8), o.cls, o.conf):
            lines.append(f"{int(c)} " + " ".join(f"{v:g}" for v in p) + (f" {float(s):g}" if save_conf else ""))
        if lines:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
        return str(txt_file)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """Rotated crop of every box -> save_dir/<class name>/<file_name>.jpg (numbered when the name exists); the
        crop follows the box rotation and areas outside the image are black."""
        o = self.obb.numpy()
        for (cx, cy, w, h, r), c in zip(o.xywhr, o.cls):
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
        """Annotated BGR image (numpy) with the oriented boxes and 'name conf' labels."""
        im = np.ascontiguousarray((self.orig_img if img is None else img).copy())
        if boxes and len(self):
            o = self.obb.numpy()
            lw = int(line_width or max(round(sum(im.shape) / 2 * 0.003), 2))
            tf = max(lw - 1, 1)
            fs = font_size / 10 if font_size else lw / 3
            for i, (p, c, s) in enumerate(zip(o.xyxyxyxy, o.cls, o.conf)):
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
        return (f"openobb.results.Results object: {self.path} {self.orig_shape[1]}x{self.orig_shape[0]}, "
                f"{len(self)} oriented boxes, names: {self.names}")
