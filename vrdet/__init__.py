"""VRDet: real-time oriented-box (OBB) detector for drawings and aerial images (Apache-2.0 / MIT components only).

    from vrdet import Detector
    Detector("s").train(data="data.yaml", epochs=50)
"""
__version__ = "0.1.0"

from vrdet.cli import Detector  # noqa: E402,F401
