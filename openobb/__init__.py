"""VRDet: oriented-box (OBB) detector for drawings and aerial images (Apache-2.0 / MIT components only).

Deployment (same interface as an Ultralytics OBB model):

    from openobb import VRDet
    model = VRDet("best.pt")
    for r in model.predict("page.png", conf=0.25):
        print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)

Training: `openobb train data=data.yaml model=vrdet5x ...` (CLI) or VRDet("vrdet5x").train(data="data.yaml").
"""
__version__ = "0.1.0"


def __getattr__(name):              # lazy: `python -m openobb.cli` must not import openobb.cli twice
    if name == "Detector":
        from openobb.cli import Detector
        return Detector
    if name in ("VRDet", "Metrics"):
        import openobb.model as m
        return getattr(m, name)
    if name in ("Results", "OBB"):
        import openobb.results as r
        return getattr(r, name)
    raise AttributeError(name)
