import math

import torch

from openobb.models.openobb3 import DIRECT_CH, REG_CH
from openobb.models.openobb18 import OpenOBB21, OpenOBB22
from openobb.models.openobb18_loss import OpenOBB18Loss

torch.set_num_threads(1)

IMG, NC = 256, 3
G = [0, 1, 1]


def _targets():
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 60 / IMG, 6 / IMG, math.pi / 2],
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.7, 0.6, 40 / IMG, 6 / IMG, 0.0]])}]


def test_v22_starts_every_angle_at_45_deg_with_direct_boxes():
    torch.manual_seed(0)
    m = OpenOBB22("s", num_classes=NC, img_size=IMG, groups=G).eval()
    assert m.head.reg[0][-1].out_channels == 2 * DIRECT_CH
    with torch.no_grad():
        _, reg = m.run_head(m.features(torch.rand(1, 3, IMG, IMG)), False)
    th = reg.view(reg.shape[0], reg.shape[1], 2, DIRECT_CH)[..., 0]
    assert torch.allclose(th, torch.full_like(th, math.pi / 4), atol=1e-6)


def test_v22_rebuilds_box_rows_per_group_from_a_dfl_checkpoint(tmp_path):
    torch.manual_seed(0)
    src = OpenOBB21("s", num_classes=NC, img_size=IMG, groups=G)
    with torch.no_grad():
        for m in src.head.reg:
            m[-1].weight.normal_()
            m[-1].bias.normal_()
    path = tmp_path / "v21.pt"
    torch.save({"model": src.state_dict(), "args": {"arch": "v21"}}, path)
    m = OpenOBB22("s", num_classes=NC, img_size=IMG, groups=G)
    assert m.after_weights(str(path))
    w_s, w = src.head.reg[0][-1].weight, m.head.reg[0][-1].weight
    for k in range(2):
        assert torch.equal(w[k * DIRECT_CH + 1], w_s[k * REG_CH + 1])          # across offset of group k
        assert float(w[k * DIRECT_CH].abs().sum()) == 0                          # angle restarts (45 deg)
        assert float(w[k * DIRECT_CH + 2:(k + 1) * DIRECT_CH].abs().sum()) == 0  # ends / thickness restart
    torch.save({"model": m.state_dict(), "args": {"arch": "v22"}}, tmp_path / "v22.pt")
    assert not m.after_weights(str(tmp_path / "v22.pt"))                         # a v22 checkpoint is kept as is


def test_v22_trains():
    torch.manual_seed(0)
    m = OpenOBB22("s", num_classes=NC, img_size=IMG, groups=G).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    g = m.head.reg[0][-1].weight.grad
    assert g is not None and g[0].abs().sum() > 0 and g[2:DIRECT_CH].abs().sum() > 0


def test_v22_fuse_keeps_outputs():
    torch.manual_seed(0)
    m = OpenOBB22("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    with torch.no_grad():
        for r in m.head.reg:
            r[-1].weight.normal_(0, 0.02)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        o1 = m(x)
        o2 = m.fuse()(x)
    assert torch.allclose(o1["pred_boxes"], o2["pred_boxes"], atol=1e-4)


def test_v22_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb22x"] == ("v22", "x") and "v22" in KNOWN_ARCHS
