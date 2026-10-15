import torch

from openobb.models.openobb3 import fast_nms
from openobb.ops.obb_torch import probiou

torch.set_num_threads(1)


def _reference(logits, boxes, max_det=1000, conf=0.001, iou=0.7, pre=4000):
    """The former full-matrix Fast-NMS (overlap of every pair, masked by class)."""
    B, A, C = logits.shape
    out_l = logits.new_full((B, max_det, C), -30.0)
    out_b = boxes.new_zeros((B, max_det, 5))
    out_b[..., 2:4] = 1.0
    for i in range(B):
        sc = logits[i].sigmoid().flatten()
        s, idx = sc.topk(min(pre, sc.numel()))
        k = s > conf
        s, idx = s[k], idx[k]
        lab, b = idx % C, boxes[i][idx // C]
        ov = probiou(b[:, None, :], b[None, :, :]) * (lab[:, None] == lab[None, :])
        k = torch.triu(ov, diagonal=1).amax(0) <= iou
        s, lab, b = s[k][:max_det], lab[k][:max_det], b[k][:max_det]
        n = len(s)
        out_l[i, torch.arange(n), lab] = torch.logit(s.clamp(1e-6, 1 - 1e-6))
        out_b[i, :n] = b
    return out_l, out_b


def test_classwise_nms_equals_the_full_matrix():
    torch.manual_seed(0)
    B, A, C = 2, 600, 7
    logits = torch.randn(B, A, C) * 2
    xy = torch.rand(B, A, 2) * 300
    wh = torch.rand(B, A, 2) * 60 + 2
    th = (torch.rand(B, A, 1) - 0.5) * 3.1
    boxes = torch.cat([xy, wh, th], -1)
    for iou in (0.5, 0.7):
        a = fast_nms(logits, boxes, max_det=300, conf=0.05, iou=iou)
        r = _reference(logits, boxes, max_det=300, conf=0.05, iou=iou)
        assert torch.equal(a[0], r[0]) and torch.equal(a[1], r[1])
