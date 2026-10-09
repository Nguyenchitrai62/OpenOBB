import torch

from openobb.models.openobb1 import postprocess


def test_argmax_post_keeps_one_class_per_query():
    logits = torch.tensor([[[4.0, 1.0, -9.0], [-9.0, -9.0, 2.0], [0.5, 3.0, -9.0]]])     # query 0 also likes class 1
    boxes = torch.rand(1, 3, 5)
    s, l, b = postprocess({"pred_logits": logits, "pred_boxes": boxes}, num_top=10, img_size=100, mode="argmax")
    assert s.shape == (1, 3) and l.tolist() == [[0, 1, 2]]
    s2, l2, _ = postprocess({"pred_logits": logits, "pred_boxes": boxes}, num_top=4, img_size=100)
    assert l2[0].tolist().count(1) == 2                      # flat: query 0 emits a second (class 1) detection
