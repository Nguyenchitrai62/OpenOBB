import numpy as np

from openobb.eval.ultra import geometry, geometry_line
from openobb.ops.obb import obb2poly

CLASSES = ["wall", "junction"]


def _poly(cx, cy, w, h, deg):
    return obb2poly(np.array([[cx, cy, w, h, np.deg2rad(deg)]], np.float64))[0]


def _case(det_deg, det_shift=0.0):
    gts = {"a": [(_poly(100, 100, 200, 6, 0), "wall", 0), (_poly(300, 300, 12, 12, 0), "junction", 0)]}
    per_img = {"a": [(0.9, 0, _poly(100, 100 + det_shift, 200, 6, det_deg)), (0.9, 1, _poly(300, 300, 12, 12, 90))]}
    return geometry(per_img, gts, CLASSES)


def test_exact_boxes_have_no_geometry_error():
    g = _case(0.0)
    assert g["found"] == 2
    assert g["angle_gt1"] == 0 and g["end_out"] == 0 and g["long_dets_tilted"] == 0
    assert geometry_line(g).startswith("Box geometry")


def test_square_label_angle_is_modulo_90():
    assert _case(0.0)["angle_gt2"] == 0.0          # the junction predicted at 90 deg is not an error


def test_tilted_long_box_is_caught_even_when_found():
    g = _case(1.5, det_shift=1.0)                   # ProbIoU stays >= 0.5, an end leaves the 6 px wall
    assert g["long"] == 1 and g["long_angle_gt1"] == 1.0 and g["long_end_out"] == 1.0
    assert g["long_dets_tilted"] == 1.0 and g["long_labels_tilted"] == 0.0
