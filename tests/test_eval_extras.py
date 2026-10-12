"""NMS thickness floor, duplicate counting, YOLO-style speed line."""
import numpy as np
import torch

from openobb.models.openobb3 import fast_nms
from openobb.speed import SpeedMeter, speed_line

torch.set_num_threads(1)


def test_nms_thickness_floor_merges_thin_parallel_duplicates():
    # two 200 x 5 px walls, 2.5 px apart: ProbIoU well below 0.7 -> both kept; with a 12 px floor -> one
    boxes = torch.tensor([[[100.0, 100.0, 200.0, 5.0, 0.0], [100.0, 102.5, 200.0, 5.0, 0.0]]])
    logits = torch.tensor([[[3.0], [2.0]]])
    keep = lambda **k: int((fast_nms(logits, boxes, max_det=10, **k)[0][0, :, 0] > -20).sum())  # noqa: E731
    assert keep() == 2
    assert keep(min_side=12.0) == 1
    # walls 20 px apart stay separate even with the floor
    far = torch.tensor([[[100.0, 100.0, 200.0, 5.0, 0.0], [100.0, 120.0, 200.0, 5.0, 0.0]]])
    assert int((fast_nms(logits, far, max_det=10, min_side=12.0)[0][0, :, 0] > -20).sum()) == 2


def test_confusion_counts_duplicates():
    from openobb.eval.ultra import confusion, duplicate_line
    g = np.array([0, 0, 40, 0, 40, 10, 0, 10], float)
    gts = {"a": [(g, "wall", 0)]}
    per_img = {"a": [(0.9, 0, g), (0.8, 0, g + 0.5)]}                       # second box = duplicate
    cm = confusion(per_img, gts, ["wall"])
    assert cm["duplicates"] == {"wall": 1} and cm["detections"] == 2
    assert "Duplicate boxes" in duplicate_line(cm)


def test_speed_line_like_ultralytics():
    m = SpeedMeter(skip=1)
    m.add(4, 1.0, 1.0, 1.0)                                                # warm-up batch, ignored
    m.add(4, 0.004, 0.04, 0.008, (4, 3, 1280, 1280))
    line = speed_line(m.result())
    assert line.startswith("Speed: 1.0ms preprocess, 10.0ms inference, 0.0ms loss, 2.0ms postprocess per image")
