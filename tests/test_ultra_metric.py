import numpy as np

from vrdet.eval.dota import evaluate
from vrdet.eval.ultra import evaluate_ultra, probiou


def _poly(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2])


def test_probiou_identity_and_thin_leniency():
    b = np.array([[100.0, 100.0, 200.0, 3.0, 0.0]])
    assert abs(probiou(b, b)[0, 0] - 1.0) < 1e-3
    shifted = np.array([[100.0, 101.0, 200.0, 3.0, 0.0]])      # 1 px across a 3 px wall: polygon IoU 0.5
    v = probiou(b, shifted)[0, 0]
    assert 0.55 < v < 0.7


def test_perfect_and_thin_offset():
    gts = {"a": [(_poly(100, 100, 200, 3).tolist(), "wall", False), (_poly(50, 50, 20, 20).tolist(), "door", False)]}
    perfect = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 100, 200, 3)])),
               "door": (["a"], np.array([0.8]), np.stack([_poly(50, 50, 20, 20)]))}
    u = evaluate_ultra(perfect, gts, ["wall", "door"])
    assert abs(u["mAP50"] - 0.995) < 1e-6 and abs(u["mAP50_95"] - 0.995) < 1e-6   # 101-point AP tops at 0.995
    off = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 101.2, 200, 3)])),     # polygon IoU ~0.43
           "door": (["a"], np.array([0.8]), np.stack([_poly(50, 50, 20, 20)]))}
    d = evaluate(off, gts, ["wall", "door"])
    u = evaluate_ultra(off, gts, ["wall", "door"])
    assert d["classes"]["wall"]["AP50"] == 0.0 and u["classes"]["wall"]["AP50"] > 0.99    # same box, two verdicts
