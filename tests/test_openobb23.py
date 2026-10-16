import math

import torch

from openobb.models.openobb17 import peak_mask
from openobb.models.openobb18 import BoundarySnapRot, OpenOBB21, OpenOBB23
from openobb.models.openobb18_loss import OpenOBB18Loss

torch.set_num_threads(1)

IMG, NC = 256, 3
G = [0, 1, 1]


def _targets():
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 120 / IMG, 6 / IMG, math.pi / 2],
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.6, 0.6, 140 / IMG, 6 / IMG, 0.0]])}]


def test_v23_starts_as_v21():
    torch.manual_seed(0)
    a = OpenOBB21("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    b = OpenOBB23("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing == ["snap.rot_gain"]
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    assert torch.equal(oa["pred_logits"], ob["pred_logits"])
    assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-6)


def test_rotational_snapping_reads_the_tilt_of_a_long_box():
    """A field whose long-side ridges belong to a box tilted by +2 deg; the predicted box is axis-aligned."""
    stride, H, W = 2, 128, 256                                     # field cells (2 px each) -> 256 x 512 px image
    ys = (torch.arange(H) + 0.5) * stride
    xs = (torch.arange(W) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy, L, T, tilt = 256.0, 128.0, 300.0, 8.0, math.radians(2.0)
    u = (gx - cx) * math.cos(tilt) + (gy - cy) * math.sin(tilt)
    v = -(gx - cx) * math.sin(tilt) + (gy - cy) * math.cos(tilt)
    ridge = torch.exp(-((v.abs() - T / 2) ** 2) / 2.0) * (u.abs() <= L / 2)
    field = torch.full((1, 2, H, W), -8.0)
    field[0, 1] = -8.0 + 16.0 * ridge
    snap = BoundarySnapRot()
    est = snap.rotation(field, torch.tensor([[[cx, cy, L, T, 0.0]]]))
    assert abs(math.degrees(float(est)) - 2.0) < 0.5
    with torch.no_grad():
        snap.rot_gain.fill_(1.0 / 20.0)                            # gain 1
    out = snap(field, torch.tensor([[[cx, cy, L, T, 0.0]]]))
    assert abs(math.degrees(float(out[0, 0, 4])) - 2.0) < 0.5
    short = snap.rotation(field, torch.tensor([[[cx, cy, 20.0, T, 0.0]]]))
    assert float(short) == 0.0                                     # not applied to short boxes (L < 4 T)


def test_v23_trains_the_rotation_gain():
    torch.manual_seed(0)
    m = OpenOBB23("s", num_classes=NC, img_size=IMG, groups=G).train()
    with torch.no_grad():
        m.snap.gain.fill_(0.05)
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.snap.rot_gain.grad is not None


def test_peak_mask_keeps_local_maxima_per_level_and_group():
    sizes, G2 = [(4, 4), (2, 2)], 2
    A = 16 + 4
    cls = torch.full((1, G2 * A, 3), -5.0)
    cls[0, 5, 0], cls[0, 6, 0] = 2.0, 1.5                           # group 0, level 0: two neighbours -> keep the higher
    cls[0, 16, 1] = 1.0                                              # group 0, level 1: alone -> kept
    cls[0, A + 6, 2] = 0.5                                           # group 1 at the same cell: other group -> kept
    keep = peak_mask(cls, sizes, G2)[0]
    assert keep[5] and not keep[6] and keep[16] and keep[A + 6]


def test_v23_peak_inference_runs():
    m = OpenOBB23("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    m.infer_head, m.final_nms = "o2o", "peak"
    with torch.no_grad():
        out = m(torch.rand(1, 3, 128, 128))
    assert out["pred_boxes"].shape == (1, 50, 5)


def test_v23_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb23x"] == ("v23", "x") and "v23" in KNOWN_ARCHS


def test_calibration_prefers_less_tilt_at_equal_accuracy():
    from openobb.predict import choose_postprocess

    def r(tilt):
        return {"ultra": {"geometry": {"long_dets": 600, "long_dets_tilted": tilt}}}
    tried = [(0.5072, 0.018, ("o2m", 0.5), r(0.154), None), (0.5068, 0.019, ("o2o", 0.5), r(0.055), None),
             (0.5100, 0.055, ("o2m", 0.7), r(0.150), None)]
    assert choose_postprocess(tried)[2] == ("o2o", 0.5)
    tried[1] = (0.5068, 0.019, ("o2o", 0.5), r(0.145), None)                   # < 2 points less tilt: accuracy wins
    assert choose_postprocess(tried)[2] == ("o2m", 0.5)
