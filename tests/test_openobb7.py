import pytest
import torch

from openobb.models.common import FrozenBatchNorm2d
from openobb.models.openobb1 import SelectiveKernel
from openobb.models.openobb3 import REG_CH, OpenOBB3
from openobb.models.openobb3_loss import OpenOBB3Loss
from openobb.models.openobb7 import (PARTS7, BoundedSelectiveKernel, OpenOBB7, ScaleGuard, fold_bn, parse_off7,
                                     space_to_depth)

torch.set_num_threads(1)
ALL_OFF = ",".join(PARTS7)


def _targets(n=3, nc=3):
    t = []
    for b in range(2):
        boxes = torch.tensor([[0.3, 0.4, 0.4, 0.03, 0.0], [0.6, 0.6, 0.05, 0.05, 0.0], [0.5, 0.2, 0.3, 0.02, 1.2]])
        t.append({"boxes": boxes[:n], "labels": torch.tensor([0, 1, 2])[:n] % nc})
    return t


def _randomise_bn(m):
    for mod in m.modules():                                 # non-trivial BN statistics / scales
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)


@pytest.mark.parametrize("off", [None, ALL_OFF])
def test_openobb3_weights_give_openobb3_output(off):
    torch.manual_seed(0)
    m3 = OpenOBB3("s", num_classes=3, img_size=256).eval()
    m7 = OpenOBB7("s", num_classes=3, img_size=256, off=off).eval()
    res = m7.load_state_dict(m3.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith(("spd.", "head.ctx.")) for k in res.missing_keys)     # no OpenOBB3 key is missing
    if off:
        assert not res.missing_keys and set(m7.state_dict()) == set(m3.state_dict())
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        a, b = m3(x), m7(x)
        f4, feats = m7.encode(x)
    assert f4.shape[-2:] == (64, 64) and [f.shape[-1] for f in feats] == [32, 16, 8]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_parse_off7():
    assert parse_off7(None) == frozenset() and parse_off7("") == frozenset()
    assert parse_off7("spd, cls_ctx") == {"spd", "cls_ctx"} and parse_off7(["guard"]) == {"guard"}
    with pytest.raises(ValueError):
        parse_off7("spd,nope")


def test_scale_guard():
    g = ScaleGuard(4.0)
    x = torch.randn(2, 8, 16, 16)
    assert torch.equal(g(x), x)                             # healthy map: exactly the identity
    big = x * 1e6
    big[1] = x[1]                                           # per sample: the healthy sample stays untouched
    y = g(big)
    assert torch.equal(y[1], x[1])
    assert abs(float(y[0].pow(2).mean().sqrt()) - 4.0) < 1e-3
    assert torch.isfinite(g(torch.full((1, 4, 8, 8), 1e30))).all()
    xb = (x * 1e4).bfloat16()                               # half precision: stats in float32, dtype preserved
    yb = g(xb)
    assert yb.dtype == torch.bfloat16 and abs(float(yb.float().pow(2).mean().sqrt()) - 4.0) < 0.05
    xr = (x * 1e3).requires_grad_()
    g(xr).sum().backward()
    assert torch.isfinite(xr.grad).all()


def test_bounded_lsk():
    torch.manual_seed(0)
    old, new = SelectiveKernel(16), BoundedSelectiveKernel(16)
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        assert torch.equal(new(x), old(x)) and torch.equal(new(x), x)        # zero-initialised output: identity
        new.out.weight.normal_(0, 50.0)
        new.out.bias.fill_(20.0)
        y = new(x)
    assert torch.isfinite(y).all() and (y.abs() <= 2 * x.abs() + 1e-5).all()  # gain bounded by 2


def test_space_to_depth_odd_edge():
    f4 = torch.randn(1, 3, 9, 10)
    out = space_to_depth(f4, (5, 5))
    assert out.shape == (1, 12, 5, 5)
    assert torch.equal(torch.nn.functional.pixel_shuffle(out, 2)[..., :9, :10], f4)    # lossless, zero-padded edge
    assert space_to_depth(torch.randn(1, 3, 8, 8), (4, 4)).shape == (1, 12, 4, 4)


def test_switches_remove_parts():
    full = OpenOBB7("s", num_classes=3, img_size=128)
    assert full.guard is not None and full.spd is not None and hasattr(full.head, "ctx")
    assert all(isinstance(m, BoundedSelectiveKernel) for m in full.lsk)
    n_full = sum(p.numel() for p in full.parameters())
    x = torch.rand(1, 3, 128, 128)
    for part in PARTS7:
        m = OpenOBB7("s", num_classes=3, img_size=128, off=part).eval()
        n = sum(p.numel() for p in m.parameters())
        keys = set(m.state_dict())
        if part == "guard":
            assert m.guard is None and n == n_full
        elif part == "lsk_bound":
            assert all(type(l) is SelectiveKernel for l in m.lsk) and n == n_full
        elif part == "spd":
            assert m.spd is None and not any(k.startswith("spd.") for k in keys) and n < n_full
        elif part == "cls_ctx":
            assert not hasattr(m.head, "ctx") and not any(".ctx." in k for k in keys) and n < n_full
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out["pred_boxes"]).all()


def test_fold_frozen_bn():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=True)
    bn = FrozenBatchNorm2d(6)
    bn.weight.uniform_(0.5, 1.5)
    bn.bias.uniform_(-0.2, 0.2)
    bn.running_mean.uniform_(-0.5, 0.5)
    bn.running_var.uniform_(0.5, 2.0)
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        assert torch.allclose(fold_bn(conv, bn)(x), bn(conv(x)), atol=1e-5)


def test_fuse_keeps_outputs():
    torch.manual_seed(0)
    m = OpenOBB7("s", num_classes=3, img_size=128)
    with torch.no_grad():
        _randomise_bn(m)
        for blk in m.head.ctx:                              # make the zero-initialised context branch active
            blk.pw2.bn.weight.uniform_(0.5, 1.5)
        m.spd.bn.weight.uniform_(0.5, 1.5)
    m.eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        a = m.head(m.encode(x)[1])
        m.fuse()
        b = m.head(m.encode(x)[1])
        out = m(x)
    assert all(torch.allclose(u, v, atol=1e-3, rtol=1e-3) for u, v in zip(a, b))
    assert not any(isinstance(mod, (torch.nn.BatchNorm2d, FrozenBatchNorm2d)) for mod in m.modules())
    assert torch.isfinite(out["pred_boxes"]).all()


@pytest.mark.parametrize("off", [None, ALL_OFF])
def test_training_forward_loss_backward(off):
    torch.manual_seed(0)
    m = OpenOBB7("s", num_classes=3, img_size=256, off=off).train()
    out = m(torch.rand(2, 3, 256, 256))
    A = (256 // 8) ** 2 + (256 // 16) ** 2 + (256 // 32) ** 2
    assert out["cls"].shape == (2, A, 3) and out["reg"].shape == (2, A, REG_CH)
    losses = OpenOBB3Loss(img_size=256)(out, _targets())
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if not off:                                             # new zero-initialised parts receive a gradient
        assert m.spd.bn.weight.grad is not None and m.head.ctx[0].pw2.bn.weight.grad.abs().sum() > 0
