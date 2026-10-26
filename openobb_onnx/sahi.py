"""Sliced inference (SAHI) with the same calls as a YOLO model.

Code that runs a YOLO model through SAHI changes one import line:

    from sahi import AutoDetectionModel                  # before
    from openobb_onnx.sahi import AutoDetectionModel     # after: OpenOBB .onnx models, anything else -> SAHI as before

    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="best.onnx",
                                               confidence_threshold=0.1, device="cuda")
    result = get_sliced_prediction(image, model, slice_height=1024, slice_width=1024,     # sahi.predict, unchanged
                                   overlap_height_ratio=0.5, overlap_width_ratio=0.5)

`result.object_prediction_list` holds the same `ObjectPrediction`s as for a YOLO OBB model: `.bbox` (axis-aligned box),
`.score`, `.category`, and `.mask.segmentation` = [[x1, y1, ..., x4, y4]] (the 4 corners of the oriented box).

The model is a `sahi.models.ultralytics.UltralyticsDetectionModel` with `is_obb` true (without needing the ultralytics
package), so SAHI treats it exactly as a YOLO OBB model: in `get_sliced_prediction` the predictions of overlapping
slices are combined with NMS (SAHI's rule for OBB models), never merged into polygons of more than 4 corners.

`model_type` may stay "ultralytics": an OpenOBB model is recognised from its file (an exported .onnx carries its own
metadata). `model_type="openobb"` asks for it explicitly. A file that is not an OpenOBB model goes to SAHI's own loader
unchanged, so YOLO models keep working next to OpenOBB ones.

Needs `pip install sahi` (MIT licence). No PyTorch: the model runs on ONNX Runtime; `device` is "cpu", "cuda", "cuda:1"
or None (CUDA when available), as for the model itself.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:
    from sahi import AutoDetectionModel as _SahiAutoDetectionModel
    from sahi.models.base import DetectionModel
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction  # noqa: F401  (re-exported for one import line)
    from sahi.prediction import ObjectPrediction
    from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
except ImportError as e:                                    # pragma: no cover - message only
    raise ImportError("sliced inference needs the SAHI package: pip install sahi") from e

from .model import METADATA_KEY, OpenOBB

MODEL_TYPES = ("openobb",)          # model_type values that always mean an OpenOBB model


def _np(x):
    """Tensor-like (numpy, the runtime's arrays, a torch tensor of the PyTorch package) -> numpy."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


class OpenOBBDetectionModel(UltralyticsDetectionModel):
    """SAHI detection model for OpenOBB. Same constructor as SAHI's models, plus iou (the model's NMS threshold;
    None: the one calibrated at training) and tile (forwarded to predict). Subclass of SAHI's Ultralytics model only so
    that SAHI applies its OBB rules to it; nothing of ultralytics is imported or needed."""

    def __init__(self, *args, iou=None, tile=None, fuse=False, task=None, **kwargs):
        self.iou, self.tile, self.fuse, self.task = iou, tile, False, "obb"
        self.required_packages = []                        # SAHI's Ultralytics model asks for ultralytics
        DetectionModel.__init__(self, *args, **kwargs)

    # SAHI's select_device needs torch for a GPU and sets CUDA_VISIBLE_DEVICES=-1 for "cpu": the model handles the
    # device itself (ONNX Runtime providers)
    def set_device(self, device=None):
        self.device = None if device in (None, "", "auto") else str(device)

    def _load(self, path):
        return OpenOBB(path, device=self.device)

    def load_model(self):
        if not self.model_path:
            raise ValueError("model_path must be given for an OpenOBB model")
        self.set_model(self._load(str(self.model_path)))

    def set_model(self, model, **kwargs):
        self.model = model
        if not self.category_mapping:
            self.category_mapping = {str(i): n for i, n in model.names.items()}

    def unload_model(self):
        self.model = None

    @property
    def category_names(self):
        return list(self.model.names.values())

    @property
    def num_categories(self):
        return len(self.model.names)

    @property
    def has_mask(self):
        return False

    @property
    def is_obb(self):
        return True

    def perform_inference(self, image):
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images):
        """images: list of (H, W, 3) RGB arrays (SAHI's convention); the model reads BGR like a YOLO model."""
        if self.model is None:
            raise ValueError("model is not loaded: call load_model()")
        bgr = [np.ascontiguousarray(np.asarray(im)[..., ::-1]) for im in images]
        kw = {"conf": self.confidence_threshold, "iou": self.iou, "verbose": False}
        if self.image_size is not None:
            kw["imgsz"] = self.image_size
        if self.tile is not None:
            kw["tile"] = self.tile
        preds = []
        for r in self.model.predict(bgr, **kw):
            o = r.obb
            if o is None or not len(o):
                preds.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64),
                              np.zeros((0, 4, 2), np.float32)))
                continue
            preds.append((_np(o.xyxy).astype(np.float32), _np(o.conf).astype(np.float32),
                          _np(o.cls).astype(np.int64), _np(o.xyxyxyxy).astype(np.float32).reshape(-1, 4, 2)))
        self._original_predictions = preds
        self._original_shapes = [np.asarray(im).shape for im in images]

    def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=[[0, 0]],
                                                                  full_shape_list=None):
        shifts = fix_shift_amount_list(shift_amount_list)
        fulls = fix_full_shape_list(full_shape_list)
        per_image = []
        for k, (xyxy, conf, cls, corners) in enumerate(self._original_predictions):
            shift = [int(v) for v in shifts[k]]
            full = None if fulls is None else [int(v) for v in fulls[k]]
            shape = self._original_shapes[k]
            out = []
            for box, s, c, poly in zip(xyxy, conf, cls, corners):
                x1, y1, x2, y2 = (max(0.0, float(v)) for v in box)
                if full is not None:                       # (height, width) of the full image
                    x1, x2 = min(float(full[1]), x1), min(float(full[1]), x2)
                    y1, y2 = min(float(full[0]), y1), min(float(full[0]), y2)
                if not (x1 < x2 and y1 < y2):
                    continue
                out.append(ObjectPrediction(
                    bbox=[x1, y1, x2, y2], category_id=int(c), score=float(s),
                    segmentation=[poly.reshape(-1).astype(float).tolist()],
                    category_name=self.category_mapping[str(int(c))], shift_amount=shift,
                    full_shape=list(shape[:2]) if full is None else full))
            per_image.append(out)
        self._object_prediction_list_per_image = per_image


def _varint(f):
    shift = result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        result |= (b[0] & 0x7F) << shift
        if b[0] < 0x80:
            return result
        shift += 7


def onnx_metadata_keys(path):
    """Keys of an ONNX file's metadata_props (ModelProto field 14), read from the top-level protobuf fields: the
    network (field 7, most of the file) is skipped, not parsed."""
    keys = []
    with open(path, "rb") as f:
        while True:
            try:
                tag = _varint(f)
            except EOFError:
                return keys
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                _varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                n = _varint(f)
                if field != 14:
                    f.seek(n, 1)
                    continue
                entry = io.BytesIO(f.read(n))         # StringStringEntryProto: key = field 1, value = field 2
                while True:
                    try:
                        t = _varint(entry)
                    except EOFError:
                        break
                    if t & 7 != 2:
                        break
                    s = entry.read(_varint(entry))
                    if t >> 3 == 1:
                        keys.append(s.decode("utf-8", "replace"))
            else:
                return keys


def is_model_file(path):
    """An exported OpenOBB model: an .onnx whose metadata has the runtime's entry."""
    p = Path(str(path))
    if p.suffix.lower() != ".onnx" or not p.is_file():
        return False
    try:
        return METADATA_KEY in onnx_metadata_keys(p)
    except Exception:  # noqa: BLE001  - not a readable protobuf: not ours
        return False


class AutoDetectionModel:
    """Drop-in for `sahi.AutoDetectionModel`: OpenOBB models get `OpenOBBDetectionModel`, any other model goes to SAHI."""

    detection_model = OpenOBBDetectionModel
    model_types = MODEL_TYPES
    is_model_file = staticmethod(is_model_file)
    model_class = OpenOBB

    @classmethod
    def from_pretrained(cls, model_type="openobb", model_path=None, model=None, config_path=None, device=None,
                        mask_threshold=0.5, confidence_threshold=0.3, category_mapping=None,
                        category_remapping=None, load_at_init=True, image_size=None, **kwargs):
        ours = (str(model_type).lower() in cls.model_types
                or (model is not None and isinstance(model, cls.model_class))
                or (model is None and model_path is not None and cls.is_model_file(model_path)))
        common = dict(model_path=None if model_path is None else str(model_path), model=model,
                      config_path=config_path, device=device, mask_threshold=mask_threshold,
                      confidence_threshold=confidence_threshold, category_mapping=category_mapping,
                      category_remapping=category_remapping, load_at_init=load_at_init, image_size=image_size)
        if not ours:
            return _SahiAutoDetectionModel.from_pretrained(model_type=model_type, **common, **kwargs)
        return cls.detection_model(**common, **kwargs)


__all__ = ["AutoDetectionModel", "OpenOBBDetectionModel", "get_sliced_prediction", "get_prediction", "is_model_file"]
