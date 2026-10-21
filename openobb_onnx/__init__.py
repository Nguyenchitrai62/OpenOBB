"""OpenOBB ONNX runtime: oriented-box detection with a model exported by `openobb export` (ONNX Runtime, no PyTorch).

    from openobb_onnx import OpenOBB
    results = OpenOBB("best.onnx").predict("page.png")
"""
from ._version import __version__
from .model import OpenOBB
from .results import OBB, Results

__all__ = ["OpenOBB", "Results", "OBB", "__version__"]
