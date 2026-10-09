"""Ultralytics-style model object for deployment (OpenOBB's own implementation of that public interface).

    from openobb import OpenOBB
    model = OpenOBB("best.pt")                          # any OpenOBB checkpoint (v1..v5)
    results = model.predict("page.png", conf=0.25, iou=0.7, imgsz=1280)   # or model("page.png")
    for r in results:
        r.obb.xyxyxyxy, r.obb.xywhr, r.obb.conf, r.obb.cls, r.names, r.plot(), r.save_txt("out.txt")

`source`: image path, folder, glob, http(s) URL, list of these, numpy BGR image (as cv2.imread / YOLO), PIL image,
or a torch tensor (B, 3, H, W) RGB in [0, 1]. Images are prepared exactly as in training (long side = training
imgsz, or tiles for tile=True models); detections are merged with class-wise polygon NMS.
"""
import glob as _glob
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from openobb.results import OBB, Results

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _to_bgr(im):
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im


def _read(path):
    data = np.fromfile(str(path), np.uint8)              # unicode-safe (Windows paths with Japanese names)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _to_bgr(im)


def load_sources(source):
    """-> list of (name, BGR uint8 image)."""
    if source is None:
        raise ValueError("source is required (image path, folder, URL, numpy image, PIL image or tensor)")
    if isinstance(source, (list, tuple)):
        out = []
        for i, s in enumerate(source):
            out += [(n if n != "image0.jpg" else f"image{i}.jpg", im) for n, im in load_sources(s)]
        return out
    if isinstance(source, np.ndarray):
        return [("image0.jpg", _to_bgr(source))]
    if isinstance(source, torch.Tensor):
        t = source[None] if source.ndim == 3 else source
        arr = (t.detach().float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
        return [(f"image{i}.jpg", np.ascontiguousarray(a)) for i, a in enumerate(arr)]
    try:
        from PIL import Image
        if isinstance(source, Image.Image):
            return [("image0.jpg", np.ascontiguousarray(np.asarray(source.convert("RGB"))[..., ::-1]))]
    except ImportError:
        pass
    s = str(source)
    if s.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(s, timeout=30) as r:
            im = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
        return [(Path(s.split("?")[0]).name or "image0.jpg", im)]
    p = Path(s)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    elif any(ch in s for ch in "*?["):
        files = sorted(Path(f) for f in _glob.glob(s, recursive=True) if Path(f).suffix.lower() in IMG_EXT)
    else:
        if p.suffix.lower() not in IMG_EXT:
            raise NotImplementedError(f"unsupported source {s} (images only: {', '.join(IMG_EXT)})")
        files = [p]
    if not files:
        raise FileNotFoundError(f"no images found in {s}")
    return [(str(f), _read(f)) for f in files]


class _Box:
    def __init__(self, res, names):
        u = res.get("ultra") or res
        ap50_95 = [u["classes"].get(n, {}).get("AP50_95", 0.0) for n in names.values()] if "classes" in u else []
        self.map, self.map50 = res["mAP50_95"], res["mAP50"]
        self.maps = np.array(ap50_95)
        self.mp, self.mr, self.f2 = res.get("P", 0.0), res.get("R", 0.0), res.get("F2", 0.0)


class Metrics:
    """Validation result with YOLO-style fields: metrics.box.map, .map50, .maps, .mp, .mr; metrics.results_dict."""

    def __init__(self, res, names):
        self.raw, self.names, self.box = res, names, _Box(res, names)
        self.results_dict = {"metrics/precision(B)": self.box.mp, "metrics/recall(B)": self.box.mr,
                             "metrics/mAP50(B)": self.box.map50, "metrics/mAP50-95(B)": self.box.map,
                             "fitness": 0.1 * self.box.map50 + 0.9 * self.box.map}


class OpenOBB:
    task = "obb"

    def __init__(self, model="best.pt", task=None, verbose=False):
        self.ckpt_path = str(model)
        self.model, self.device, self.args, self._names = None, None, {}, None
        if self.ckpt_path.endswith(".pt"):
            if not Path(self.ckpt_path).exists():
                raise FileNotFoundError(self.ckpt_path)
            self._load()

    # ---- loading
    def _load(self, device=None):
        from openobb.predict import load_model
        dev = torch.device(device if device not in (None, "") else ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            dev = torch.device(f"cuda:{int(device)}")
        self.model, names, self.args = load_model(self.ckpt_path, dev, 900)        # 900 queries (OpenOBB1), as the CLI
        self.device, self._names = dev, {i: n for i, n in enumerate(names)}

    @property
    def names(self):
        return dict(self._names or {})

    def to(self, device):
        self._load(device)
        return self

    def eval(self):
        return self

    def fuse(self):
        return self

    def info(self, verbose=False):
        n = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        print(f"OpenOBB {self.args.get('arch', 'v1')}-{self.args.get('size', '?')}: {n / 1e6:.2f}M parameters, "
              f"{len(self.names)} classes, imgsz {self.args.get('img')}")

    # ---- inference
    def __call__(self, source=None, stream=False, **kw):
        return self.predict(source, stream=stream, **kw)

    def predict(self, source=None, conf=0.25, iou=None, imgsz=None, device=None, max_det=300, classes=None,
                agnostic_nms=False, verbose=True, save=False, save_txt=False, save_conf=False, project="runs/obb",
                name="predict", exist_ok=False, stream=False, batch=8, tile=None, line_width=None, show=False,
                show_labels=True, show_conf=True, **ignored):
        """-> list[Results] (a generator with stream=True). conf may be 'auto' (per-class thresholds of best.pt)."""
        gen = self._predict(source, conf, iou, imgsz, device, max_det, classes, agnostic_nms, verbose, save,
                            save_txt, save_conf, project, name, exist_ok, batch, tile, line_width, show, show_labels,
                            show_conf)
        return gen if stream else list(gen)

    def _predict(self, source, conf, iou, imgsz, device, max_det, classes, agnostic_nms, verbose, save, save_txt,
                 save_conf, project, name, exist_ok, batch, tile, line_width, show, show_labels, show_conf):
        from openobb.data.dota import polys_to_obb
        from openobb.ops.obb import nms_poly
        from openobb.predict import conf_thresholds, merge_dets, predict_image
        if self.model is None or (device not in (None, "") and str(device) != str(self.device)):
            self._load(device)
        a = self.args
        names = [self._names[i] for i in range(len(self._names))]
        size = int(imgsz or a.get("img", 1024))
        fit = a.get("fit", False) if tile is None else not tile
        if iou is None:
            iou = 0.7 if fit else 0.1
        thr = conf_thresholds(names, a, conf)
        keep_cls = None
        if classes is not None:
            cl = classes if isinstance(classes, (list, tuple)) else [classes]
            keep_cls = {int(c) if str(c).isdigit() else names.index(str(c)) for c in cl}
        save_dir = None
        if save or save_txt:
            from openobb.cli import increment_path
            save_dir = increment_path(project, name, exist_ok)
            save_dir.mkdir(parents=True, exist_ok=True)
        srcs = load_sources(source)
        for k, (path, img) in enumerate(srcs):
            t0 = time.perf_counter()
            scale = size / max(img.shape[:2]) if fit else float(a.get("scale", 1.0))
            t1 = time.perf_counter()
            raw = predict_image(self.model, img, size, 200, batch, self.device, max(max_det, 300) * 3, scale=scale)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            raw = [d for d in raw if d[2] >= thr[d[1]] and (keep_cls is None or d[1] in keep_cls)]
            dets = merge_dets(raw, names, 0.0, iou=iou)
            if agnostic_nms and dets:
                polys = np.array([d["poly"] for d in dets], np.float64)
                keep = nms_poly(polys, np.array([d["score"] for d in dets]), iou)
                dets = [dets[i] for i in keep]
            dets = sorted(dets, key=lambda d: -d["score"])[:max_det]
            if dets:
                obb = polys_to_obb(np.array([d["poly"] for d in dets], np.float32))
                obb[:, 4] = np.mod(obb[:, 4], np.pi)                     # w >= h, rotation in [0, pi)
                data = np.concatenate([obb, np.array([[d["score"], d["class_id"]] for d in dets], np.float32)], 1)
            else:
                data = np.zeros((0, 7), np.float32)
            t3 = time.perf_counter()
            speed = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
            r = Results(img, path, self._names, OBB(torch.from_numpy(data), img.shape[:2]), speed, save_dir)
            if verbose:
                print(f"image {k + 1}/{len(srcs)} {path}: {img.shape[0]}x{img.shape[1]} {r.verbose()}"
                      f"{speed['inference']:.1f}ms")
            if save_dir is not None:
                stem = Path(path).stem
                if save:
                    r.save(str(save_dir / f"{stem}.jpg"), line_width=line_width, labels=show_labels, conf=show_conf)
                if save_txt:
                    (save_dir / "labels").mkdir(exist_ok=True)
                    r.save_txt(str(save_dir / "labels" / f"{stem}.txt"), save_conf=save_conf)
            if show:
                r.show(line_width=line_width, labels=show_labels, conf=show_conf)
            yield r
        if verbose and save_dir is not None:
            print(f"Results saved to {save_dir}")

    # ---- training / validation (same entry points as the CLI)
    def train(self, data=None, **kw):
        from openobb.cli import train
        r = train(data, model=self.ckpt_path, **kw)
        if Path(r.best).exists():
            self.ckpt_path = r.best
            self._load()
        return r

    def val(self, data=None, **kw):
        from openobb.cli import val
        return Metrics(val(self.ckpt_path, data, **kw), self.names)
