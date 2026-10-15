import math

import torch

from openobb.models.openobb12 import OpenOBB12
from openobb.models.openobb17 import OpenOBB17
from openobb.models.openobb18 import OpenOBB19, OpenOBB20
from openobb.models.openobb18_loss import OpenOBB18Loss, box_distill
from openobb.models.openobb3 import REG_CH

torch.set_num_threads(1)

IMG, NC = 256, 3
G = [0, 1, 1]


def _targets():
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 10 / IMG, 6 / IMG, 0.0],
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.7, 0.6, 40 / IMG, 6 / IMG, 1.2]])}]


def test_v20_starts_as_v19():
    torch.manual_seed(0)
    a = OpenOBB19("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    b = OpenOBB20("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("edge.") for k in missing)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    assert torch.equal(oa["pred_logits"], ob["pred_logits"])
    assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-6)


def test_v20_trains_refiner_then_snap():
    torch.manual_seed(0)
    m = OpenOBB20("s", num_classes=NC, img_size=IMG, groups=G).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert "ref_stages" in out and len(out["ref_stages"]) == 1                # refiner, then snapping
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    assert {"loss_ref_box", "loss_ref_box2", "loss_field"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.edge.mlp[-1].weight.grad.abs().sum() > 0


def test_head_distillation_from_grouped_and_one_group_teachers():
    torch.manual_seed(0)
    x = torch.rand(1, 3, IMG, IMG)
    for teacher in (OpenOBB17("s", num_classes=NC, img_size=IMG, groups=G).eval(),
                    OpenOBB12("s", num_classes=NC, img_size=IMG).eval()):
        m = OpenOBB19("s", num_classes=NC, img_size=IMG, groups=G).train()
        out = m(x, _targets())
        losses = OpenOBB18Loss(img_size=IMG, teacher=teacher)(out, _targets())
        assert {"loss_distill", "loss_ld", "loss_kd_cls"} <= set(losses), type(teacher).__name__
        assert all(math.isfinite(float(losses[k].detach())) for k in ("loss_ld", "loss_kd_cls"))
        sum(losses.values()).backward()
        g = m.head.reg[0][-1].weight.grad
        assert g is not None and g.abs().sum() > 0


def test_box_distill_is_zero_on_identical_outputs():
    r = torch.randn(5, REG_CH)
    assert float(box_distill(r, r).abs().max()) < 1e-4
    assert float(box_distill(r, r + torch.randn(5, REG_CH)).min()) > 0


def test_v20_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb20x"] == ("v20", "x") and "v20" in KNOWN_ARCHS
