import math

import torch

from openobb.models.openobb18 import OpenOBB23
from openobb.models.openobb18_loss import OpenOBB18Loss
from openobb.models.openobb24 import OpenOBB24, box_move, dfl_cues, side_profiles
from openobb.models.openobb3 import REG_CH, fast_nms

torch.set_num_threads(1)

IMG, NC = 256, 3
G = [0, 1, 1]


def _targets():
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 120 / IMG, 6 / IMG, math.pi / 2],
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.6, 0.6, 140 / IMG, 6 / IMG, 0.0]])}]


def _pair():
    torch.manual_seed(0)
    a = OpenOBB23("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    b = OpenOBB24("s", num_classes=NC, img_size=128, max_det=50, groups=G).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("quality.") for k in missing)
    return a, b


def test_v24_starts_as_v23():
    """New head constant at init: whatever the calibrated use of q, the boxes are OpenOBB23's (scores too at alpha 0)."""
    a, b = _pair()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        oa = a(x)
        for beta, alpha in ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)):
            b.q_beta, b.q_alpha = beta, alpha
            ob = b(x)
            assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-5)
            assert torch.allclose(oa["pred_logits"].sigmoid(), ob["pred_logits"].sigmoid(), atol=1e-5)
        b.q_beta, b.q_alpha = 1.0, 0.5                     # constant q: same boxes and order, scores scaled
        ob = b(x)
        assert torch.allclose(oa["pred_boxes"], ob["pred_boxes"], atol=1e-5)


def test_quality_chooses_the_kept_box_inside_a_cluster():
    """Two overlapping boxes of one class: NMS by score keeps the higher score; ranked by score x q, the tighter."""
    logits = torch.tensor([[[2.0, -30.0], [1.8, -30.0]]])
    boxes = torch.tensor([[[50.0, 50.0, 40.0, 6.0, 0.0], [50.5, 50.0, 40.0, 6.0, 0.0]]])
    q = torch.tensor([[0.3, 0.9]])
    ls = torch.nn.functional.logsigmoid(logits)
    rank = torch.logit((ls + torch.log(q)[..., None]).exp().clamp(1e-6, 1 - 1e-6))
    _, kept = fast_nms(logits, boxes, max_det=5, iou=0.5)
    l2, kept2 = fast_nms(rank, boxes, max_det=5, iou=0.5, out_logits=logits)
    assert torch.allclose(kept[0, 0], boxes[0, 0]) and torch.allclose(kept2[0, 0], boxes[0, 1])
    assert abs(float(l2[0, 0, 0]) - 1.8) < 1e-4                     # the kept box keeps its own class score


def test_v24_trains_the_quality_head_only_through_its_loss():
    torch.manual_seed(0)
    m = OpenOBB24("s", num_classes=NC, img_size=IMG, groups=G).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert "ref_fit" in out and out["ref_fit"].shape == out["ref_fit_valid"].shape
    assert out["ref_fit_valid"].any() and not out["ref_fit_valid"][:, -1].all()   # jittered rows excluded
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    assert "loss_ref_fit" in losses
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.quality.mlp[-1].weight.grad is not None and m.quality.mlp[-1].weight.grad.abs().sum() > 0
    # inputs are detached: the fit loss alone does not reach the refiner / field / head
    m.zero_grad()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    OpenOBB18Loss(img_size=IMG).fit_loss(out, _targets()).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in m.named_parameters() if not n.startswith("quality."))


def test_quality_cues():
    rows = torch.zeros(2, 4, REG_CH)
    rows[0, :, 2 + 5] = 20.0                                         # sharp left distribution
    c = dfl_cues(rows)
    assert c.shape == (2, 4, 6) and c[0, 0, 0] > 0.99 and c[1, 0, 0] < 0.1
    assert dfl_cues(torch.zeros(1, 2, 5)).abs().sum() == 0            # direct head: no distributions
    a = torch.tensor([[[10.0, 10.0, 20.0, 4.0, 0.0]]])
    b = torch.tensor([[[10.0, 12.0, 20.0, 4.0, 0.0]]])
    mv = box_move(a, b)[0, 0]
    assert abs(float(mv[1]) - 0.5) < 1e-5 and abs(float(mv[0])) < 1e-6
    field = torch.zeros(1, 2, 32, 32)
    assert side_profiles(field, a).shape == (1, 1, 4 * 3 * 5 * 2)


def test_v24_snap_skip_keeps_the_refiner_box():
    _, b = _pair()
    with torch.no_grad():
        b.snap.gain.fill_(0.05)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        f4, feats = b.encode(x)
        field = b.field(f4, feats[0])
        cand = torch.tensor([[[60.0, 60.0, 40.0, 6.0, 0.0], [30.0, 30.0, 8.0, 8.0, 0.0]]])
        grp = torch.tensor([[0, 1]])
        s0, _ = b.edge_snap(f4, feats[0], field, cand, grp)
        b.snap_skip = (1,)
        s1, _ = b.edge_snap(f4, feats[0], field, cand, grp)
    assert torch.allclose(s1[-1][1][0, 1], s1[0][1][0, 1])          # group 1: refiner box
    assert torch.allclose(s1[-1][1][0, 0], s0[-1][1][0, 0])          # group 0: snapped as before


def test_v24_off_quality_is_v23():
    m = OpenOBB24("s", num_classes=NC, img_size=128, max_det=50, groups=G, off24="quality")
    assert m.quality is None
    from openobb.predict import quality_options
    assert quality_options(m) == [(0.0, 0.0)]


def test_v24_registry_strip_and_reload(tmp_path):
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS, load_model
    from openobb.strip import strip
    assert MODELS["openobb24x"] == ("v24", "x") and "v24" in KNOWN_ARCHS
    _, b = _pair()
    args = {"arch": "v24", "size": "s", "img": 128, "num_top": 50, "groups": "0,1,1", "nms_iou": 0.5,
            "infer_head": "o2m", "q_beta": 2.0, "q_alpha": 0.0, "snap_skip": "1"}
    ck = tmp_path / "v24.pt"
    torch.save({"model": b.state_dict(), "args": args, "classes": ["a", "b", "c"]}, ck)
    m, names, a = load_model(str(ck), "cpu")
    assert m.q_beta == 2.0 and m.snap_skip == (1,) and m.infer_head == "o2m" and names == ["a", "b", "c"]
    out = tmp_path / "v24_strip.pt"
    strip(str(ck), str(out))
    m2, _, a2 = load_model(str(out), "cpu")
    assert m2.q_beta == 2.0 and m2.snap_skip == (1,) and a2.get("groups") == "0,1,1"
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(m(x)["pred_boxes"], m2(x)["pred_boxes"], atol=1e-5)
