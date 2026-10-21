"""OpenOBB model exported to ONNX (`openobb export`), run with ONNX Runtime; no PyTorch.

    from openobb_onnx import OpenOBB
    model = OpenOBB("best.onnx")                      # device=None: GPU (CUDA) when onnxruntime-gpu sees one, else CPU
    results = model.predict("page.png", conf=0.25)    # or model("page.png"); same arguments as the PyTorch package
    for r in results:
        r.obb.xywhr, r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names, r.plot(), r.save_txt("out.txt")

Same interface as the PyTorch package's OpenOBB (constructor, predict arguments, Results with tensor-like arrays), so
inference code written for a .pt runs unchanged with the .onnx path: the handover wheel installs this package under
the PyTorch package's import name.

Images are prepared as in training (long side = imgsz, padded to multiples of 32; tiles for tile-trained models), the
graph runs the network up to the NMS, and the NMS / detection merge run in numpy with the settings calibrated at
training (saved in the model file). Detections match `openobb predict` on the .pt.
"""
import glob as _glob
import json
import math
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from .ops import fast_nms, merge, nms_poly, obb2poly, polys_to_obb, regularize_xywhr
from .results import OBB, Results, imread

IMG_EXT = (".bmp", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm")
PAD_BGR = (104, 116, 124)        # padding colour of training
METADATA_KEY = "obb"
FORMATS = (1,)                   # graph interfaces this runtime reads


def windows(length, size, gap):
    step = size - gap
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    s.append(length - size)
    return s


def no_power_throttling():
    """Windows 11 throttles a process whose window is in the background (EcoQoS: efficiency cores, low clock). A model
    served from a console window (run_app.bat) while the user works in the browser ran 7x slower: 8.6 vs 1.2 s per page
    on an i5-14400. The process opts out (SetProcessInformation, ProcessPowerThrottling, execution speed not throttled)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.SetProcessInformation.restype = wintypes.BOOL
        st = State(1, 1, 0)                       # version 1, control: execution speed, state: not throttled
        return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))
    except Exception:  # noqa: BLE001           - older Windows: nothing to opt out of
        return False


def _providers(device):
    """device: None / "auto" (CUDA when available), "cpu", "cuda", "cuda:1", 0, "trt" (TensorRT, then CUDA)."""
    import onnxruntime as ort
    have = ort.get_available_providers()
    d = "auto" if device in (None, "") else str(device).lower()
    idx = 0
    if d.isdigit():
        d, idx = "cuda", int(d)
    elif d.startswith("cuda:"):
        d, idx = "cuda", int(d.split(":")[1])
    if d == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in have and hasattr(ort, "preload_dlls"):
        try:                                   # CUDA / cuDNN DLLs of the pip packages (onnxruntime-gpu[cuda,cudnn])
            ort.preload_dlls()
        except Exception:  # noqa: BLE001     - system CUDA install: ONNX Runtime finds the DLLs itself
            pass
    # page sizes vary (long side = imgsz, short side any multiple of 32): cuDNN's exhaustive algorithm search would
    # re-run on every new input shape, its heuristic choice does not
    cuda = ("CUDAExecutionProvider", {"device_id": idx, "cudnn_conv_algo_search": "HEURISTIC"})
    if d in ("trt", "tensorrt") and "TensorrtExecutionProvider" in have:
        return [("TensorrtExecutionProvider", {"device_id": idx, "trt_fp16_enable": False}), cuda,
                "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in have and d in ("auto", "cuda", "trt", "tensorrt"):
        return [cuda, "CPUExecutionProvider"]
    if d != "auto":
        print(f"WARNING: device={device!r} needs onnxruntime-gpu (providers here: {have}); running on the CPU")
    return ["CPUExecutionProvider"]


def _files(s):
    p = Path(s)
    if any(ch in s for ch in "*?["):
        return sorted(Path(f) for f in _glob.glob(s, recursive=True) if Path(f).suffix.lower() in IMG_EXT)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    if p.suffix.lower() == ".txt" and p.is_file():
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = Path(line.strip())
                out += _files(str(q if q.is_absolute() or q.exists() else p.parent / q))
        return out
    if not p.exists():
        raise FileNotFoundError(f"{s} does not exist")
    return [p]


def _read_url(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as r:
        im = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot decode image from {url}")
    return im


def list_sources(source):
    """-> [(name, BGR image or loader)]: path, folder, glob, .txt list, http(s) URL, numpy BGR image, PIL image, a
    tensor (B, 3, H, W) RGB in [0, 1] (anything with .detach().cpu().numpy()), or a list of these."""
    if isinstance(source, (list, tuple)):
        out = []
        for i, s in enumerate(source):
            out += [(n if n != "image0.jpg" else f"image{i}.jpg", im) for n, im in list_sources(s)]
        return out
    if isinstance(source, np.ndarray):
        im = source
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return [("image0.jpg", im)]
    if hasattr(source, "detach") and hasattr(source, "permute"):        # torch tensor, without importing torch
        t = np.asarray(source.detach().cpu().numpy(), np.float32)
        t = t[None] if t.ndim == 3 else t
        arr = (np.clip(t, 0, 1) * 255).round().astype(np.uint8).transpose(0, 2, 3, 1)[..., ::-1]
        return [(f"image{i}.jpg", np.ascontiguousarray(a)) for i, a in enumerate(arr)]
    try:
        from PIL import Image
        if isinstance(source, Image.Image):
            return [("image0.jpg", np.ascontiguousarray(np.asarray(source.convert("RGB"))[..., ::-1]))]
    except ImportError:
        pass
    if isinstance(source, int):
        raise NotImplementedError("webcam sources are not supported (images only)")
    if str(source).startswith(("http://", "https://")):
        name = Path(str(source).split("?")[0]).name or "image0.jpg"
        return [(name, lambda u=str(source): _read_url(u))]
    files = _files(str(source))
    if not files:
        raise FileNotFoundError(f"no images found in {source}")
    return [(str(f), lambda f=f: _read(f)) for f in files]


def _read(path):
    im = imread(path)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return im


# predict arguments of the PyTorch package / Ultralytics that only concern videos, streams or feature maps: accepted
# with a warning (as the PyTorch package does)
UNSUPPORTED = {"vid_stride", "stream_buffer", "visualize", "augment", "embed", "retina_masks", "save_frames",
               "half", "dnn", "rect", "mode", "task", "model", "cfg", "tracker", "persist"}


class Device(str):
    """Device name ("cpu", "cuda:0", "tensorrt:0") that also has .type / .index, like torch.device."""

    @property
    def type(self):
        return self.split(":")[0]

    @property
    def index(self):
        return int(self.split(":")[1]) if ":" in self else None


class OpenOBB:
    task = "obb"

    def __init__(self, model="best.onnx", task=None, verbose=False, device=None, threads=None):
        self.path = self.ckpt_path = str(model)
        if not self.path.lower().endswith(".onnx"):
            raise ValueError(f"{self.path}: this package runs exported .onnx models (no PyTorch): export the .pt with "
                             f"`openobb export model=<best.pt>` and pass the .onnx path")
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.threads, self.compile = threads, False
        self._open(device)

    def _open(self, device=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        # worker threads sleep instead of spinning between operators: 1.2 -> 1.1 s per page on the CPU, and 8.6 -> 4.8 s
        # when Windows throttles the process anyway (spinning threads on slow cores wait for each other)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        no_power_throttling()
        if self.threads:
            so.intra_op_num_threads = int(self.threads)
        want = _providers(device)
        self.session = ort.InferenceSession(self.path, so, providers=want)
        if want[0] != "CPUExecutionProvider" and self.session.get_providers()[0] == "CPUExecutionProvider":
            print("WARNING: the GPU could not be used (CUDA / cuDNN libraries or driver), running on the CPU: "
                  "pip install \"onnxruntime-gpu[cuda,cudnn]\" and an NVIDIA driver for CUDA 13 (docs: GPU install)")
        meta = self.session.get_modelmeta().custom_metadata_map
        if METADATA_KEY not in meta:
            raise ValueError(f"{self.path}: not an exported OBB model (no '{METADATA_KEY}' metadata)")
        self.meta = json.loads(meta[METADATA_KEY])
        if self.meta.get("format") not in FORMATS:
            raise ValueError(f"{self.path}: model format {self.meta.get('format')} is newer than this package "
                             f"(reads {FORMATS}): update the package")
        self._names = {i: n for i, n in enumerate(self.meta["names"])}
        self.input = self.session.get_inputs()[0].name
        self.model = self.session                      # set: the model is loaded (as the PyTorch package's .model)
        self._device = device

    @property
    def names(self):
        return dict(self._names)

    @property
    def device(self):
        """Device("cpu") / Device("cuda:0") / Device("tensorrt:0"): a str with .type / .index, like torch.device."""
        p = self.session.get_providers()[0]
        kind = {"CUDAExecutionProvider": "cuda", "TensorrtExecutionProvider": "tensorrt"}.get(p, "cpu")
        if kind == "cpu":
            return Device("cpu")
        opts = self.session.get_provider_options().get(p, {})
        return Device(f"{kind}:{int(opts.get('device_id', 0))}")

    @property
    def args(self):
        """Settings saved in the model (the fields of the PyTorch package's model.args that inference reads)."""
        m = self.meta
        return {"img": m["imgsz"], "imgsz": m["imgsz"], "fit": m.get("fit", True), "scale": m.get("scale", 1.0),
                "nms_iou": m.get("nms"), "conf_thr": dict(m.get("conf_thr") or {}), "task": "obb"}

    @property
    def overrides(self):
        return {"task": "obb", "imgsz": self.meta["imgsz"]}

    def to(self, device):
        self._open(device)
        return self

    def eval(self):
        return self

    def fuse(self):
        return self

    def info(self, detailed=False, verbose=True):
        line = (f"{Path(self.path).name}: {len(self._names)} classes, imgsz {self.meta['imgsz']}, "
                f"NMS {self.meta.get('nms')}, device {self.device}")
        if verbose:
            print(line)
        return line

    def _only_inference(self, what):
        raise NotImplementedError(f"{what}: an exported .onnx model only runs inference (predict); train / val / "
                                  f"export need the .pt and the full package")

    def train(self, *_, **__):
        self._only_inference("train")

    def val(self, *_, **__):
        self._only_inference("val")

    def export(self, *_, **__):
        self._only_inference("export")

    def track(self, *_, **__):
        raise NotImplementedError("tracking is not supported: detection on still images")

    def warmup(self, imgsz=None):
        """One run on a blank page: the first run allocates the memory pools (and on a GPU loads the kernels)."""
        size = int(imgsz or self.meta["imgsz"])
        self._run(np.full((1, size // 32 * 24, size // 32 * 32, 3), 255, np.uint8))
        return self

    def __call__(self, source=None, **kw):
        return self.predict(source, **kw)

    # ---- the network on one batch of tiles
    def _run(self, x):
        return self.session.run(None, {self.input: x})

    def _image_dets(self, img, size, fit, scale, gap, batch, conf, nms, num_top):
        """-> [(cls, score, poly8 in image px)] before the class thresholds (predict_image of the PyTorch package)."""
        m = self.meta
        if scale != 1.0:
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        H, W = img.shape[:2]
        tiles = [(x, y) for y in windows(H, size, gap) for x in windows(W, size, gap)]
        st, lo = int(m.get("stride", 32)), int(m.get("min_side", 256))
        th, tw = (min(math.ceil(H / st) * st, size), min(math.ceil(W / st) * st, size)) if len(tiles) == 1 \
            else (size, size)
        th, tw = max(math.ceil(th / st) * st, lo), max(math.ceil(tw / st) * st, lo)
        self.last_shape = (1, 3, th, tw)
        dets = []
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            x = np.empty((len(chunk), th, tw, 3), np.uint8)
            x[:] = PAD_BGR
            for j, (x0, y0) in enumerate(chunk):
                t = img[y0:y0 + th, x0:x0 + tw]
                x[j, :t.shape[0], :t.shape[1]] = t
            rank, score, label, box = self._run(x)
            for j, (x0, y0) in enumerate(chunk):
                s, lab, b = fast_nms(rank[j], score[j], label[j], box[j], conf, nms, int(m.get("max_det", 1000)))
                top = np.argsort(-s, kind="stable")[:num_top]
                polys = obb2poly(b[top])
                polys[:, 0::2] = (polys[:, 0::2] + x0) / scale
                polys[:, 1::2] = (polys[:, 1::2] + y0) / scale
                dets += [(int(c), float(v), p) for c, v, p in zip(lab[top], s[top], polys)]
        return dets

    def conf_thresholds(self, conf):
        """One value, or "auto": the per-class thresholds (best F2) saved at training."""
        names = self.meta["names"]
        if str(conf).lower() == "auto":
            saved = self.meta.get("conf_thr") or {}
            return [float(saved.get(n, 0.25)) for n in names]
        return [float(conf)] * len(names)

    # ---- inference
    def predict(self, source=None, stream=False, conf=0.25, iou=None, imgsz=None, device=None, batch=1, max_det=300,
                classes=None, agnostic_nms=False, verbose=True, save=False, save_txt=False, save_conf=False,
                save_crop=False, show=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None,
                project=None, name=None, exist_ok=False, tile=None, compile=None, **kw):
        """-> list[Results] (a generator with stream=True): the arguments and defaults of the PyTorch package's predict.
        conf: score threshold or "auto" (per-class thresholds of training); iou: NMS threshold (None: the one calibrated
        at training, "none": no NMS); classes: ids or names to keep; tile: True / False forces tiling / whole-image
        resize; device: moves the model to that device (as the PyTorch package); compile: ignored (ONNX Runtime
        already optimises the graph)."""
        bad = [k for k in kw if k not in UNSUPPORTED]
        if bad:
            raise SyntaxError(f"'{bad[0]}' is not a valid predict argument")
        for k in kw:
            if k not in ("half", "mode", "task", "model", "rect", "dnn"):
                warnings.warn(f"predict argument '{k}' is not supported (images only) and is ignored", stacklevel=2)
        if device not in (None, "") and str(device) != str(self._device):
            self._open(device)
        gen = self._predict(source, conf, iou, imgsz, max_det, classes, agnostic_nms, max(int(batch or 1), 1), tile,
                            verbose, save, save_txt, save_conf, save_crop, show, project, name, exist_ok, line_width,
                            show_labels, show_conf, show_boxes)
        return gen if stream else list(gen)

    def _predict(self, source, conf, iou, imgsz, max_det, classes, agnostic_nms, batch, tile, verbose, save, save_txt,
                 save_conf, save_crop, show, project, name, exist_ok, line_width, show_labels, show_conf, show_boxes):
        m = self.meta
        names = m["names"]
        size = int(max(imgsz) if isinstance(imgsz, (list, tuple)) else (imgsz or m["imgsz"]))
        fit = bool(m.get("fit", True)) if tile is None else not tile
        if iou is None:                                    # the NMS calibrated at training
            nms, merge_iou = m.get("nms"), (m.get("merge_iou") if fit else 0.1)
        elif str(iou).lower() in ("none", "off", "false"):
            nms, merge_iou = None, (None if fit else 0.1)
        else:
            nms = merge_iou = float(iou)
            merge_iou = merge_iou if fit else 0.1
        thr = self.conf_thresholds(conf)
        conf_floor = max(0.001, min(thr))
        keep_cls = None
        if classes is not None:
            cl = classes if isinstance(classes, (list, tuple)) else [classes]
            keep_cls = {int(c) if str(c).isdigit() else names.index(str(c)) for c in cl}
        save_dir = None
        if save or save_txt or save_crop:
            save_dir = _increment(Path(project or "runs/obb") / (name or "predict"), exist_ok)
            save_dir.mkdir(parents=True, exist_ok=True)
        srcs = list_sources(source)
        t_sum, n_img = np.zeros(3), 0
        for k, (path, img) in enumerate(srcs):
            t0 = time.perf_counter()
            img = img() if callable(img) else img
            scale = size / max(img.shape[:2]) if fit else float(m.get("scale", 1.0))
            t1 = time.perf_counter()
            raw = self._image_dets(img, size, fit, scale, 200, 8 * batch, conf_floor, nms, max(max_det, 300) * 3)
            t2 = time.perf_counter()
            raw = [d for d in raw if d[1] >= thr[d[0]] and (keep_cls is None or d[0] in keep_cls)]
            dets = merge(raw, merge_iou)
            if agnostic_nms and dets:
                keep = nms_poly(np.stack([d[2] for d in dets]), np.array([d[1] for d in dets]),
                                0.7 if merge_iou is None else merge_iou)
                dets = [dets[i] for i in keep]
            dets = sorted(dets, key=lambda d: -d[1])[:max_det]
            if dets:
                obb = regularize_xywhr(polys_to_obb(np.stack([d[2] for d in dets]).astype(np.float32)))
                data = np.concatenate([obb, np.array([[d[1], d[0]] for d in dets], np.float32)], 1)
            else:
                data = np.zeros((0, 7), np.float32)
            t3 = time.perf_counter()
            speed = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
            t_sum += [speed["preprocess"], speed["inference"], speed["postprocess"]]
            n_img += 1
            r = Results(img, path, self._names, OBB(data, img.shape[:2]), speed, str(save_dir) if save_dir else None)
            if verbose:
                s = self.last_shape
                print(f"image {k + 1}/{len(srcs)} {path}: {s[2]}x{s[3]} {r.verbose()}{speed['inference']:.1f}ms")
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
            print(f"Speed: {t[0]:.1f}ms preprocess, {t[1]:.1f}ms inference (incl. NMS), {t[2]:.1f}ms postprocess "
                  f"per image ({self.device})")
            if save_dir is not None:
                print(f"Results saved to {save_dir}")


def _increment(path, exist_ok=False):
    """runs/obb/predict, predict2, predict3... (as Ultralytics)."""
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    n = 2
    while (path.parent / f"{path.name}{n}").exists():
        n += 1
    return path.parent / f"{path.name}{n}"
