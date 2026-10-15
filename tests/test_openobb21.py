import math

import torch

from openobb.models import openobb3_loss
from openobb.models.openobb18 import OpenOBB20, OpenOBB21
from openobb.models.openobb18_loss import OpenOBB18Loss
from openobb.models.openobb3_loss import angle_loss

torch.set_num_threads(1)

IMG, NC = 256, 3
G = [0, 1, 1]


def _targets():
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 60 / IMG, 6 / IMG, math.pi / 2],     # vertical wall
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.7, 0.6, 40 / IMG, 6 / IMG, 0.0]])}]


def test_v21_starts_as_v20():
    torch.manual_seed(0)
    a = OpenOBB20("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    b = OpenOBB21("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    with torch.no_grad():                       # non-zero angle outputs, so the identity is tested away from 0
        for m in a.head.reg:
            m[-1].weight.normal_(0, 0.02)
            m[-1].bias.normal_(0, 1.0)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("ang") for k in missing)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        ra, rb = a.run_head(a.features(x), False)[1], b.run_head(b.features(x), False)[1]
        oa, ob = a(x), b(x)
    assert torch.allclose(ra, rb, atol=1e-5)
    assert torch.equal(oa["pred_logits"], ob["pred_logits"])
    assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-5)


def test_v21_trains_its_angle_branch():
    torch.manual_seed(0)
    m = OpenOBB21("s", num_classes=NC, img_size=IMG, groups=G).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.ang[0].weight.grad.abs().sum() > 0 and m.ang_o2o[0].weight.grad.abs().sum() > 0


def test_v21_fuse_keeps_outputs():
    torch.manual_seed(0)
    m = OpenOBB21("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    with torch.no_grad():
        for a in m.ang:
            a.weight.normal_(0, 0.05)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        o1 = m(x)
        o2 = m.fuse()(x)
    assert torch.allclose(o1["pred_boxes"], o2["pred_boxes"], atol=1e-4)


def test_vector_angle_loss_keeps_its_gradient_near_the_optimum():
    t = torch.tensor([math.pi / 2, 0.0])
    p = (t + math.radians(1.0)).requires_grad_(True)
    angle_loss(p, t).sum().backward()
    assert p.grad.abs().min() > 1.5                       # ~2 per radian at 1 deg; the former 1 - cos 2dt: 0.07
    assert float(angle_loss(torch.tensor([math.pi / 2]), torch.tensor([-math.pi / 2]))) < 1e-6   # pi-periodic


def test_former_angle_loss_still_selectable():
    old = openobb3_loss.ANGLE_LOSS
    try:
        openobb3_loss.ANGLE_LOSS = "cos"
        v = angle_loss(torch.tensor([0.1]), torch.tensor([0.0]))
        assert abs(float(v) - (1 - math.cos(0.2))) < 1e-6
    finally:
        openobb3_loss.ANGLE_LOSS = old


def test_v21_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb21x"] == ("v21", "x") and "v21" in KNOWN_ARCHS
