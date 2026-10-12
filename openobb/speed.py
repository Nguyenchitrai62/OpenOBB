"""Speed reported the way Ultralytics does: per image, split into preprocess / inference / postprocess, with NMS
counted in postprocess, so OpenOBB and YOLO numbers can be compared line by line.

OpenOBB's dense models run their NMS (fast_nms) inside forward(); nms_timer() times those calls so they can be moved
from "inference" to "postprocess".
"""
import contextlib
import importlib
import time

import torch

_NMS_MODULES = ("openobb.models.openobb3", "openobb.models.openobb5", "openobb.models.openobb6",
                "openobb.models.openobb7", "openobb.models.openobb8", "openobb.models.openobb9")


def sync(device):
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def nms_timer(device):
    """Within the block every fast_nms call adds its (synchronised) duration to acc[0] seconds."""
    acc = [0.0]
    saved = []
    for name in _NMS_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "fast_nms", None)
        if fn is None:
            continue

        def timed(*a, _fn=fn, **k):
            sync(device)
            t = time.perf_counter()
            r = _fn(*a, **k)
            sync(device)
            acc[0] += time.perf_counter() - t
            return r
        saved.append((mod, fn))
        mod.fast_nms = timed
    try:
        yield acc
    finally:
        for mod, fn in saved:
            mod.fast_nms = fn


class SpeedMeter:
    """Accumulates per-batch times; .result() -> {"preprocess", "inference", "postprocess"} ms per image."""

    def __init__(self, skip=1):
        self.skip, self.batches, self.images = skip, 0, 0
        self.t = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.shape = None

    def add(self, n, pre, inf, post, shape=None):
        self.batches += 1
        if self.batches <= self.skip:                      # warm-up batch (cudnn autotune, allocations)
            return
        self.images += n
        for k, v in (("preprocess", pre), ("inference", inf), ("postprocess", post)):
            self.t[k] += v
        self.shape = shape or self.shape

    def result(self):
        n = max(self.images, 1)
        return {k: v * 1e3 / n for k, v in self.t.items()} | {"shape": self.shape, "images": self.images}


def speed_line(sp):
    """Ultralytics' wording: 'Speed: 0.3ms preprocess, 9.1ms inference, 0.0ms loss, 1.2ms postprocess per image at
    shape (B, 3, H, W)'."""
    if not sp or not sp.get("images"):
        return ""
    shape = tuple(sp["shape"]) if sp.get("shape") else ""
    return (f"Speed: {sp['preprocess']:.1f}ms preprocess, {sp['inference']:.1f}ms inference, 0.0ms loss, "
            f"{sp['postprocess']:.1f}ms postprocess per image at shape {shape}")
