"""Sliced inference (SAHI) with the same calls as a YOLO model, for the full package: OpenOBB .pt checkpoints and
exported .onnx models. Same interface as `openobb_onnx.sahi` (docs: that module); code that runs a YOLO model through
SAHI changes one import line:

    from sahi import AutoDetectionModel              # before
    from openobb.sahi import AutoDetectionModel      # after (OpenOBB .pt / .onnx; any other model -> SAHI as before)

Needs `pip install sahi` (MIT licence).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from openobb.model import OpenOBB
from openobb_onnx import OpenOBB as OnnxOpenOBB
from openobb_onnx import sahi as _onnx_sahi
from openobb_onnx.sahi import get_prediction, get_sliced_prediction  # noqa: F401  (re-exported)

MODEL_TYPES = _onnx_sahi.MODEL_TYPES


def is_checkpoint(path):
    """An OpenOBB .pt checkpoint (read from the pickle inside the file, without loading the weights): a data-only
    checkpoint with OpenOBB's fields, not an Ultralytics one (which pickles ultralytics classes)."""
    p = Path(str(path))
    if p.suffix.lower() != ".pt" or not p.is_file() or not zipfile.is_zipfile(p):
        return False
    try:
        with zipfile.ZipFile(p) as z:
            name = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
            if name is None:
                return False
            data = z.read(name)
    except Exception:  # noqa: BLE001
        return False
    return b"ultralytics" not in data and b"arch" in data and b"classes" in data


def is_model_file(path):
    return is_checkpoint(path) or _onnx_sahi.is_model_file(path)


class OpenOBBDetectionModel(_onnx_sahi.OpenOBBDetectionModel):
    """SAHI detection model for an OpenOBB .pt (PyTorch) or .onnx (ONNX Runtime) model."""

    def _load(self, path):
        if path.lower().endswith(".onnx"):
            return OnnxOpenOBB(path, device=self.device)
        m = OpenOBB(path)
        return m.to(self.device) if self.device else m


class AutoDetectionModel(_onnx_sahi.AutoDetectionModel):
    """Drop-in for `sahi.AutoDetectionModel`: OpenOBB .pt / .onnx models get `OpenOBBDetectionModel`, any other model
    goes to SAHI."""

    detection_model = OpenOBBDetectionModel
    is_model_file = staticmethod(is_model_file)
    model_class = (OpenOBB, OnnxOpenOBB)


__all__ = ["AutoDetectionModel", "OpenOBBDetectionModel", "get_sliced_prediction", "get_prediction", "is_model_file"]
