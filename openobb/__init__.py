"""OpenOBB: oriented-box (OBB) detection library (OpenOBB architectures) for drawings and aerial images.

Deployment (same interface as an Ultralytics OBB model):

    from openobb import OpenOBB
    model = OpenOBB("best.pt")
    for r in model.predict("page.png", conf=0.25):
        print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)

Training: `openobb train data=data.yaml model=openobb5x ...` (CLI) or OpenOBB("openobb5x").train(data="data.yaml").
"""
__version__ = "0.18.0"   # bumped by tools/push_all.py when the package changes


def __getattr__(name):              # lazy: `python -m openobb.cli` must not import openobb.cli twice
    if name in ("Detector", "OpenOBB"):
        from openobb.torch_setup import ensure_torch   # CUDA PyTorch when an NVIDIA GPU is present (OPENOBB_AUTO_TORCH=0: off)
        ensure_torch()
    if name == "Detector":
        from openobb.cli import Detector
        return Detector
    if name in ("OpenOBB", "Metrics"):
        import openobb.model as m
        return getattr(m, name)
    if name in ("Results", "OBB"):
        import openobb.results as r
        return getattr(r, name)
    raise AttributeError(name)
