import math

import torch

from openobb.models.openobb8 import OpenOBB8
from openobb.models.openobb11 import OpenOBB11
from openobb.models.openobb11_loss import OpenOBB11Loss

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_one_to_one_targets_make_copies_and_fragments_negative():
    """Two objects; candidates: the full wall, a fragment of it (2/3 of the length), a shifted copy, the second
    object, background. One positive per object (the best refined box); copies / fragments are negatives; the
    jittered rows are the positives of their own objects."""
    g = torch.tensor([[150.0, 100.0, 300.0, 6.0, 0.0], [60.0, 200.0, 20.0, 20.0, 0.0]])
    rb = torch.tensor([[[150.0, 100.0, 300.0, 6.0, 0.0],        # 0 full wall
                        [100.0, 100.0, 200.0, 6.0, 0.0],        # 1 fragment
                        [160.0, 101.0, 300.0, 6.0, 0.0],        # 2 shifted copy
                        [61.0, 200.0, 20.0, 20.0, 0.0],         # 3 second object
                        [400.0, 400.0, 30.0, 30.0, 0.0],        # 4 background
                        [151.0, 100.0, 290.0, 6.0, 0.0],        # 5 jittered wall (denoising group)
                        [60.0, 201.0, 21.0, 20.0, 0.0]]])       # 6 jittered second object
    lg = torch.full((1, 7, 2), 1.0)
    out = {"ref_dense": 5, "o2o": True, "ref_valid": torch.ones(1, 7, dtype=torch.bool),
           "ref_jit_gt": torch.tensor([[0, 1]])}
    crit = OpenOBB11Loss(img_size=IMG)
    pos, gi = crit.ref_assign(0, lg, rb, rb, g, torch.tensor([0, 1]), out)
    assert pos.tolist() == [True, False, False, True, False, True, True]
    assert gi[[0, 3, 5, 6]].tolist() == [0, 1, 0, 1]


def test_v8_targets_unchanged_for_v8_outputs():
    g = torch.tensor([[150.0, 100.0, 300.0, 6.0, 0.0]])
    rb = torch.tensor([[[150.0, 100.0, 300.0, 6.0, 0.0], [160.0, 101.0, 300.0, 6.0, 0.0]]])
    crit = OpenOBB11Loss(img_size=IMG)
    pos, _ = crit.ref_assign(0, torch.zeros(1, 2, 1), rb, rb, g, torch.tensor([0]), {})
    assert pos.tolist() == [True, True]                     # no "o2o" flag: OpenOBB8's one-to-many rule


def test_v11_loads_v8_weights_and_trains():
    torch.manual_seed(0)
    v8 = OpenOBB8("s", num_classes=NC, img_size=IMG, rel_k=40)
    m = OpenOBB11("s", num_classes=NC, img_size=IMG, rel_k=40, off="refine")       # refine cannot be switched off
    assert m.refiner is not None
    m.load_state_dict(v8.state_dict())                      # same parameter names, strict
    m.train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert out["o2o"] and out["ref_dense"] == 20 and out["ref_jit_gt"].shape == (1, 3)
    losses = OpenOBB11Loss(img_size=IMG)(out, _targets())
    assert {"loss_ref_cls", "loss_ref_box", "loss_cls", "loss_box"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.refiner.cls_out.weight.grad is not None


def test_v11_inference_is_the_set_decoder_output_without_nms():
    torch.manual_seed(0)
    m = OpenOBB11("s", num_classes=NC, img_size=128, rel_k=30, max_det=50).eval()
    with torch.no_grad():
        r = m(torch.rand(2, 3, 128, 128))
    assert r["pred_logits"].shape == (2, 30, NC) and r["pred_boxes"].shape == (2, 30, 5)
    s = r["pred_logits"].sigmoid().amax(-1)
    assert torch.all(s[:, :-1] >= s[:, 1:])                 # best first
    m.final_nms = 0.7                                        # optional safety NMS
    with torch.no_grad():
        r2 = m(torch.rand(1, 3, 128, 128))
    assert r2["pred_logits"].shape[1] == 50
