"""VRDet: real-time oriented-box (OBB) detector for drawings and aerial images (Apache-2.0 / MIT components only).

    from vrdet import Detector
    Detector("s").train(data="data.yaml", epochs=50)
"""
__version__ = "0.1.0"


def __getattr__(name):              # lazy: `python -m vrdet.cli` must not import vrdet.cli twice
    if name == "Detector":
        from vrdet.cli import Detector
        return Detector
    raise AttributeError(name)
