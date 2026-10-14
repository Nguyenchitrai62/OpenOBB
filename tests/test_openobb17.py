import math

import torch

from openobb.models.openobb12 import OpenOBB12
from openobb.models.openobb17 import FIELD_STRIDE, GAIN_SCALE, BoundarySnap, OpenOBB17
from openobb.models.openobb17_loss import OpenOBB17Loss, field_loss, field_targets

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 6 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v17_trains_field_snap_and_both_heads():
    torch.manual_seed(0)
    m = OpenOBB17("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert out["field"].shape == (1, 2, IMG // FIELD_STRIDE, IMG // FIELD_STRIDE)
    losses = OpenOBB17Loss(img_size=IMG)(out, _targets())
    assert {"loss_cls", "loss_o2o_cls", "loss_ref_box", "loss_field"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    for prefix in ("field.", "snap.gain", "head_o2o."):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), prefix


def test_v17_starts_as_v12():
    """Snapping gains start at zero: same weights -> the same detections as OpenOBB12, for both inference heads."""
    torch.manual_seed(0)
    v12 = OpenOBB12("s", num_classes=NC, img_size=128, max_det=50).eval()
    v17 = OpenOBB17("s", num_classes=NC, img_size=128, max_det=50).eval()
    missing, unexpected = v17.load_state_dict(v12.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("field.", "snap.")) for k in missing)
    x = torch.rand(1, 3, 128, 128)
    for head in ("o2m", "o2o"):
        v12.infer_head = v17.infer_head = head
        with torch.no_grad():
            a, b = v12(x), v17(x)
        assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_field_targets_draw_ends_and_sides():
    t = [{"boxes": torch.tensor([[0.5, 0.5, 40 / 64, 8 / 64, 0.0]])}]      # 40 x 8 px horizontal bar centred at 32
    f = field_targets(t, (32, 32), 64, "cpu")                                # stride 2: cell centres 1, 3, ...
    ends, sides = f[0, 0], f[0, 1]
    # ends at x = 12 and 52 (cells 5.5 / 25.5 -> between 5 and 6 / 25 and 26), sides at y = 28 and 36
    assert ends[16, 5] > 0.5 and ends[16, 26] > 0.5 and ends[16, 16] < 1e-3          # 1 px from the ridge: 0.61
    assert sides[13, 16] > 0.5 and sides[18, 16] > 0.5 and sides[16, 16] < 0.02
    assert ends[2, 5] == 0 and sides[16, 2] < 1e-3                             # limited to the box's extent
    sq = field_targets([{"boxes": torch.tensor([[0.5, 0.5, 10 / 64, 9 / 64, 0.0]])}], (32, 32), 64, "cpu")
    assert torch.allclose(sq[0, 0], sq[0, 1])                                # near-square: all sides in both channels
    assert math.isfinite(float(field_loss(torch.zeros(1, 2, 32, 32), f)))


def test_snap_moves_ends_to_the_ridge():
    """A field with ridges at the true ends / sides pulls a box with wrong ends / thickness onto them."""
    torch.manual_seed(0)
    true = torch.tensor([[[64.0, 64.0, 40.0, 8.0, 0.0]]])
    tgt = field_targets([{"boxes": true[0].clone() / torch.tensor([128, 128, 128, 128, 1.0])}], (64, 64), 128, "cpu")
    field = (tgt * 16 - 8)                                                   # confident logits on the ridges
    snap = BoundarySnap()
    off = torch.tensor([[[65.5, 64.6, 44.0, 9.6, 0.0]]])                     # ends +-2 px off, too thick
    assert torch.equal(snap(field, off), off)                               # gain 0: identity
    with torch.no_grad():
        snap.gain.fill_(1.0 / GAIN_SCALE)
        snap.log_tau.fill_(math.log(0.25))
    out = snap(field, off)[0, 0]
    err_before = (off[0, 0] - true[0, 0]).abs()[:4]
    err_after = (out - true[0, 0]).abs()[:4]
    assert err_after[2] < 0.5 * err_before[2] and err_after[0] < 0.5 * err_before[0]   # length and centre along
    assert err_after[3] < err_before[3]                                                # thickness


def test_v17_field_off_is_v12_and_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    m = OpenOBB17("s", num_classes=NC, img_size=128, off="field")
    assert m.field is None and m.snap is None
    assert MODELS["openobb17x"] == ("v17", "x") and "v17" in KNOWN_ARCHS


def test_v17_fuse_keeps_output():
    torch.manual_seed(0)
    m = OpenOBB17("s", num_classes=NC, img_size=128, max_det=50).eval()
    with torch.no_grad():
        m.snap.gain.fill_(0.03)
        for mod in m.field.modules():                        # non-trivial BN statistics
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.1, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
        x = torch.rand(1, 3, 128, 128)
        a = m(x)
        b = m.fuse()(x)
    assert torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-3)
