"""Ultralytics-style model object for deployment (OpenOBB's own implementation of that public interface).

    from openobb import OpenOBB
    model = OpenOBB("best.pt")                          # any OpenOBB checkpoint (v1..v5)
    results = model.predict("page.png", conf=0.25, iou=0.7, imgsz=1280)   # or model("page.png")
    for r in results:
        r.obb.xyxyxyxy, r.obb.xywhr, r.obb.conf, r.obb.cls, r.names, r.plot(), r.save_txt("out.txt")

`source` (as Ultralytics, images only): image path or pathlib.Path, folder, glob ('pages/*.png', '**' recursive),
http(s) URL, .txt file listing image paths, a list of these, numpy BGR image (as cv2.imread), PIL image (RGB), or a
torch tensor (B, 3, H, W) RGB in [0, 1]. Files are decoded like the Ultralytics loader (cv2.IMREAD_COLOR: BGR, EXIF
orientation applied) and read one at a time. Each image is prepared exactly as in training (long side = imgsz,
padded; or tiles for tile=True models); detections are merged with class-wise polygon NMS.
Videos, webcams and streams are not supported (drawings are still images).
"""
import glob as _glob
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from openobb.data.imgio import IMG_EXT, imread
from openobb.results import OBB, Results, regularize_xywhr

VID_EXT = (".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv", ".webm")
# Ultralytics predict arguments that only concern videos / streams / feature maps: accepted with a warning
UNSUPPORTED = {"vid_stride", "stream_buffer", "visualize", "augment", "embed", "retina_masks", "save_frames",
               "half", "dnn", "rect", "mode", "task", "model", "cfg", "tracker", "persist", "compile"}


def _to_bgr(im):
    import cv2
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    if im.shape[2] == 1:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def _read(path):
    im = imread(path)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return im


def _read_url(url):
    import urllib.request

    import cv2
    with urllib.request.urlopen(url, timeout=30) as r:
        im = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot decode image from {url}")
    return im


def _files(s):
    """Image files of a path-like source (file, folder, glob, .txt list)."""
    p = Path(s)
    if any(ch in s for ch in "*?["):
        files = sorted(Path(f) for f in _glob.glob(s, recursive=True) if Path(f).suffix.lower() in IMG_EXT)
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    elif p.suffix.lower() == ".txt" and p.is_file():          # one source per line
        files = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("http://", "https://")):
                q, rel = Path(line), p.parent / line              # relative entries: relative to the list file
                files += _files(str(q if q.exists() or not rel.exists() else rel))
    elif p.suffix.lower() in VID_EXT or s.isdigit() or s.startswith(("rtsp://", "rtmp://", "tcp://")) \
            or s == "screen" or s.endswith(".streams"):
        raise NotImplementedError(f"{s}: videos, webcams, screens and streams are not supported (images only)")
    elif not p.exists():
        raise FileNotFoundError(f"{s} does not exist")
    elif p.suffix.lower() not in IMG_EXT:
        raise NotImplementedError(f"unsupported source {s} (images only: {', '.join(IMG_EXT)})")
    else:
        files = [p]
    return files


def list_sources(source):
    """-> list of (name, image or loader): numpy/PIL/tensor images are converted now, files and URLs are read lazily
    (one image in memory at a time, as the Ultralytics loader)."""
    if source is None:
        raise ValueError("source is required (image path, folder, URL, numpy image, PIL image or tensor)")
    if isinstance(source, (list, tuple)):
        out = []
        for i, s in enumerate(source):
            out += [(n if n != "image0.jpg" else f"image{i}.jpg", im) for n, im in list_sources(s)]
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
    if isinstance(source, int):
        raise NotImplementedError("webcam sources are not supported (images only)")
    s = str(source)
    if s.startswith(("http://", "https://")):
        name = Path(s.split("?")[0]).name or "image0.jpg"
        if Path(name).suffix.lower() in VID_EXT or "youtube.com" in s or "youtu.be" in s:
            raise NotImplementedError(f"{s}: video URLs are not supported (images only)")
        return [(name, lambda u=s: _read_url(u))]
    files = _files(s)
    if not files:
        raise FileNotFoundError(f"no images found in {s}. Supported formats: {', '.join(IMG_EXT)}")
    return [(str(f), lambda f=f: _read(f)) for f in files]


def load_sources(source):
    """-> list of (name, BGR uint8 image), every image decoded."""
    return [(n, im() if callable(im) else im) for n, im in list_sources(source)]


def _device(device):
    """'cpu' | 0 | '0' | 'cuda' | 'cuda:1' | 'mps' | None (auto) -> torch.device with an explicit cuda index."""
    if device in (None, ""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int) or (isinstance(device, str) and device.strip().isdigit()):
        device = f"cuda:{int(device)}"
    if isinstance(device, str) and "," in device:                 # '0,1': single-process inference uses the first
        device = f"cuda:{int(device.split(',')[0])}"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class _Box:
    """metrics.box with the Ultralytics fields."""

    def __init__(self, res, names):
        cls = res.get("classes", {})
        keys = [n for n in names.values() if n in cls]
        self.map, self.map50 = res["mAP50_95"], res["mAP50"]
        self.map75 = res.get("mAP75", res.get("ultra", {}).get("mAP75", 0.0))
        self.mp, self.mr, self.f2 = res.get("P", 0.0), res.get("R", 0.0), res.get("F2", 0.0)
        self.ap_class_index = np.array([i for i, n in names.items() if n in cls], int)
        self.ap50 = np.array([cls[n]["AP50"] for n in keys])
        self.ap = np.array([cls[n]["AP50_95"] for n in keys])
        self.p = np.array([cls[n].get("P", 0.0) for n in keys])
        self.r = np.array([cls[n].get("R", 0.0) for n in keys])
        self.f1 = 2 * self.p * self.r / (self.p + self.r + 1e-16)
        self.nc = len(names)
        maps = np.full(self.nc, self.map)                           # per class, mAP for classes without labels
        for i, n in zip(self.ap_class_index, keys):
            maps[i] = cls[n]["AP50_95"]
        self.maps = maps

    def mean_results(self):
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]


class Metrics:
    """Validation result with the Ultralytics fields: metrics.box.map / .map50 / .map75 / .maps / .mp / .mr / .p / .r
    / .ap50 / .ap / .ap_class_index, metrics.results_dict, metrics.fitness, metrics.names, metrics.speed."""

    def __init__(self, res, names, speed=None, save_dir=None):
        self.raw, self.names, self.box = res, names, _Box(res, names)
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.save_dir = save_dir
        self.fitness = 0.1 * self.box.map50 + 0.9 * self.box.map
        self.results_dict = {"metrics/precision(B)": self.box.mp, "metrics/recall(B)": self.box.mr,
                             "metrics/mAP50(B)": self.box.map50, "metrics/mAP50-95(B)": self.box.map,
                             "fitness": self.fitness}
        self.keys = list(self.results_dict)[:4]
        self.maps = self.box.maps
        self.ap_class_index = self.box.ap_class_index

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i):
        return self.box.class_result(i)


class OpenOBB:
    task = "obb"

    def __init__(self, model="best.pt", task=None, verbose=False):
        self.ckpt_path = str(model)
        self.model, self.device, self.args, self._names = None, None, {}, None
        self.predictor = None
        if self.ckpt_path.endswith(".pt"):
            if not Path(self.ckpt_path).exists():
                raise FileNotFoundError(self.ckpt_path)
            self._load()

    # ---- loading
    def _load(self, device=None):
        from openobb.predict import load_model
        dev = _device(device)
        self.model, names, self.args = load_model(self.ckpt_path, dev, 900)        # 900 queries (OpenOBB1), as the CLI
        self.device, self._names = dev, {i: n for i, n in enumerate(names)}

    @property
    def names(self):
        return dict(self._names or {})

    @property
    def overrides(self):
        return {"task": "obb", "imgsz": self.args.get("img"), "data": self.args.get("data")}

    def to(self, device):
        self._load(device)
        return self

    def eval(self):
        return self

    def fuse(self):
        return self

    def info(self, detailed=False, verbose=True):
        n = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        line = (f"OpenOBB{self.args.get('arch', 'v1')[1:]}-{self.args.get('size', '?')}: {n / 1e6:.2f}M parameters, "
                f"{len(self.names)} classes, imgsz {self.args.get('img')}")
        if verbose:
            print(line)
        return line

    def track(self, *args, **kw):
        raise NotImplementedError("tracking is not supported: OpenOBB detects on still images")

    def export(self, *args, **kw):
        raise NotImplementedError("export is not implemented yet: deploy the .pt with openobb (PyTorch)")

    # ---- inference
    def __call__(self, source=None, stream=False, **kw):
        return self.predict(source, stream=stream, **kw)

    def predict(self, source=None, stream=False, conf=0.25, iou=None, imgsz=None, device=None, batch=1, max_det=300,
                classes=None, agnostic_nms=False, verbose=True, save=False, save_txt=False, save_conf=False,
                save_crop=False, show=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None,
                project=None, name=None, exist_ok=False, tile=None, **kw):
        """-> list[Results] (a generator with stream=True), with the Ultralytics predict arguments and defaults.
        Extensions: conf='auto' (per-class thresholds of best.pt), classes may also be names, tile=True/False forces
        tiling at native resolution / whole-image resize. iou defaults to 0.7 (0.1 when tiles are merged)."""
        bad = [k for k in kw if k not in UNSUPPORTED]
        if bad:
            raise SyntaxError(f"'{bad[0]}' is not a valid OpenOBB predict argument")
        for k in kw:
            if k not in ("half", "mode", "task", "model", "rect", "compile", "dnn"):
                warnings.warn(f"predict argument '{k}' is not supported by OpenOBB (images only) and is ignored",
                              stacklevel=2)
        gen = self._predict(source, conf, iou, imgsz, device, max(int(batch or 1), 1), max_det, classes,
                            agnostic_nms, verbose, save, save_txt, save_conf, save_crop, show, show_labels, show_conf,
                            show_boxes, line_width, project, name, exist_ok, tile)
        return gen if stream else list(gen)

    def _predict(self, source, conf, iou, imgsz, device, batch, max_det, classes, agnostic_nms, verbose, save,
                 save_txt, save_conf, save_crop, show, show_labels, show_conf, show_boxes, line_width, project, name,
                 exist_ok, tile):
        from openobb.data.dota import polys_to_obb
        from openobb.ops.obb import nms_poly
        from openobb.predict import conf_thresholds, merge_dets, predict_image
        if self.model is None or (device not in (None, "") and _device(device) != self.device):
            self._load(device)
        a = self.args
        names = [self._names[i] for i in range(len(self._names))]
        if isinstance(imgsz, (list, tuple)):
            imgsz = max(int(v) for v in imgsz)
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
        if save or save_txt or save_crop:
            from openobb.cli import increment_path
            save_dir = increment_path(project or "runs/obb", name or "predict", exist_ok)
            save_dir.mkdir(parents=True, exist_ok=True)
        srcs = list_sources(source)
        n_img, t_sum = 0, np.zeros(3)
        for k, (path, img) in enumerate(srcs):
            t0 = time.perf_counter()
            img = img() if callable(img) else img
            scale = size / max(img.shape[:2]) if fit else float(a.get("scale", 1.0))
            t1 = time.perf_counter()
            raw = predict_image(self.model, img, size, 200, 8 * batch, self.device, max(max_det, 300) * 3,
                                scale=scale)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()
            raw = [d for d in raw if d[2] >= thr[d[1]] and (keep_cls is None or d[1] in keep_cls)]
            dets = merge_dets(raw, names, 0.0, iou=iou)
            if agnostic_nms and dets:
                polys = np.array([d["poly"] for d in dets], np.float64)
                keep = nms_poly(polys, np.array([d["score"] for d in dets]), iou)
                dets = [dets[i] for i in keep]
            dets = sorted(dets, key=lambda d: -d["score"])[:max_det]
            if dets:
                obb = regularize_xywhr(polys_to_obb(np.array([d["poly"] for d in dets], np.float32)))
                data = np.concatenate([obb, np.array([[d["score"], d["class_id"]] for d in dets], np.float32)], 1)
            else:
                data = np.zeros((0, 7), np.float32)
            data = torch.from_numpy(np.ascontiguousarray(data, np.float32)).to(self.device)
            t3 = time.perf_counter()
            speed = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
            t_sum += [speed["preprocess"], speed["inference"], speed["postprocess"]]
            n_img += 1
            r = Results(img, path, self._names, OBB(data, img.shape[:2]), speed, str(save_dir) if save_dir else None)
            if verbose:
                print(f"image {k + 1}/{len(srcs)} {path}: {size}x{size} {r.verbose()}{speed['inference']:.1f}ms")
            if save_dir is not None:
                p = Path(path)
                plot_kw = dict(line_width=line_width, labels=show_labels, conf=show_conf, boxes=show_boxes)
                if save:
                    r.save(str(save_dir / (p.stem + (p.suffix or ".jpg"))), **plot_kw)
                if save_txt:
                    r.save_txt(str(save_dir / "labels" / f"{p.stem}.txt"), save_conf=save_conf)
                if save_crop:
                    r.save_crop(save_dir / "crops", p.stem)
            if show:
                r.show(line_width=line_width, labels=show_labels, conf=show_conf, boxes=show_boxes)
            yield r
        if verbose and n_img:
            t = t_sum / n_img
            print(f"Speed: {t[0]:.1f}ms preprocess, {t[1]:.1f}ms inference, {t[2]:.1f}ms postprocess per image at "
                  f"shape (1, 3, {size}, {size})")
        if verbose and save_dir is not None:
            n_lbl = len(list((save_dir / "labels").glob("*.txt"))) if save_txt else 0
            print(f"Results saved to {save_dir}"
                  + (f"\n{n_lbl} label{'s' * (n_lbl != 1)} saved to {save_dir / 'labels'}" if save_txt else ""))

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
        t = time.perf_counter()
        res = val(self.ckpt_path, data, **kw)
        n = max(res.get("n_images", 0), 1)
        return Metrics(res, self.names, {"preprocess": 0.0, "inference": (time.perf_counter() - t) * 1e3 / n,
                                         "loss": 0.0, "postprocess": 0.0})
