"""Prediction results with the same interface as the Ultralytics OBB results, so code written for YOLO reads VRDet
output unchanged (VRDet's own implementation of that public interface; no Ultralytics code):

    r = model.predict("page.png")[0]
    r.obb.xywhr        (N, 5) cx, cy, w, h, rotation (radians, [0, pi), w >= h), pixels
    r.obb.xyxyxyxy     (N, 4, 2) corner points, pixels;  r.obb.xyxyxyxyn  normalised by the image size
    r.obb.xyxy         (N, 4) axis-aligned enclosing boxes
    r.obb.conf (N,), r.obb.cls (N,), r.obb.data (N, 7) = [xywhr, conf, cls], r.obb.id = None
    r.names {id: name}, r.orig_img (BGR), r.orig_shape (h, w), r.path, r.speed {ms}
    r.plot(), r.save(), r.show(), r.save_txt(), r.summary(), r.to_json(), r.verbose(), r.cpu() / .numpy() / .to()
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
           (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
           (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
           (200, 149, 255), (199, 55, 255)]                                   # BGR, one per class id


def _xp(t):
    return torch if isinstance(t, torch.Tensor) else np


def xywhr_to_corners(b):
    """(N, 5) cx, cy, w, h, r -> (N, 4, 2) corners (torch or numpy)."""
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
        return f"vrdet.results.OBB object with {len(self)} boxes\ndata: {self.data!r}"


class Results:
    """Detections of one image."""

    def __init__(self, orig_img, path, names, obb, speed=None, save_dir=None):
        self.orig_img, self.path, self.names, self.obb = orig_img, str(path), dict(names), obb
        self.orig_shape = tuple(orig_img.shape[:2])
        self.speed = speed or {"preprocess": None, "inference": None, "postprocess": None}
        self.save_dir = save_dir
        self.boxes = self.masks = self.probs = self.keypoints = None        # OBB task: only .obb is set

    def __len__(self):
        return len(self.obb)

    def __getitem__(self, idx):
        return self._new(self.obb[idx])

    def _new(self, obb):
        return Results(self.orig_img, self.path, self.names, obb, self.speed, self.save_dir)

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
        cls = np.asarray(self.obb.cls.cpu() if isinstance(self.obb.cls, torch.Tensor) else self.obb.cls).astype(int)
        out = ""
        for c in np.unique(cls):
            n = int((cls == c).sum())
            out += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return out

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

    def save_txt(self, txt_file, save_conf=False):
        """YOLO-OBB label lines: 'cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]', coordinates normalised."""
        o = self.obb.numpy()
        lines = []
        for p, c, s in zip(o.xyxyxyxyn.reshape(-1, 8), o.cls, o.conf):
            lines.append(f"{int(c)} " + " ".join(f"{v:.6g}" for v in p) + (f" {float(s):.6g}" if save_conf else ""))
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, "a") as f:
            f.writelines(line + "\n" for line in lines)
        return str(txt_file)

    def plot(self, conf=True, labels=True, line_width=None, font_size=None, img=None, boxes=True, **_):
        """Annotated BGR image (numpy) with the oriented boxes and 'name conf' labels."""
        im = np.ascontiguousarray((self.orig_img if img is None else img).copy())
        if not boxes or not len(self):
            return im
        o = self.obb.numpy()
        lw = line_width or max(round(sum(im.shape[:2]) / 2 * 0.002), 2)
        fs = font_size or max(lw / 3, 0.4)
        for p, c, s in zip(o.xyxyxyxy, o.cls, o.conf):
            col = PALETTE[int(c) % len(PALETTE)]
            pts = np.round(p).astype(np.int32)
            cv2.polylines(im, [pts], True, col, lw, cv2.LINE_AA)
            if labels:
                text = self.names[int(c)] + (f" {float(s):.2f}" if conf else "")
                x, y = pts[np.argmin(pts[:, 1])]
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(lw - 1, 1))
                y0 = max(int(y) - th - 4, 0)
                cv2.rectangle(im, (int(x), y0), (int(x) + tw + 2, y0 + th + 4), col, -1)
                cv2.putText(im, text, (int(x) + 1, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                            max(lw - 1, 1), cv2.LINE_AA)
        return im

    def save(self, filename=None, **kw):
        filename = filename or f"results_{Path(self.path).name}"
        cv2.imwrite(str(filename), self.plot(**kw))
        return str(filename)

    def show(self, **kw):
        im = self.plot(**kw)
        try:
            from PIL import Image
            Image.fromarray(im[..., ::-1]).show()
        except Exception:  # noqa: BLE001
            cv2.imshow(Path(self.path).name, im)
            cv2.waitKey(0)

    def __repr__(self):
        return (f"vrdet.results.Results object: {self.path} {self.orig_shape[1]}x{self.orig_shape[0]}, "
                f"{len(self)} oriented boxes, names: {self.names}")
