import math

import pytest
import torch

from openobb.models.openobb7 import OpenOBB7
from openobb.models.openobb8 import PARTS8, OpenOBB8, pair_geometry, parse_off8, refine_boxes
from openobb.models.openobb8_loss import OpenOBB8Loss

torch.set_num_threads(1)

IMG, NC = 256, 3
SMALL = 128      # eval-heavy tests: fast_nms builds a 4000 x 4000 ProbIoU matrix at 256 px (~1 GB of temporaries)


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])},
            {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}]


def _model(off=None, rel_k=40, seed=0, img=IMG):
    torch.manual_seed(seed)
    return OpenOBB8("s", num_classes=NC, img_size=img, off=off, rel_k=rel_k)


def test_parse_off8():
    assert parse_off8(None) == frozenset() and parse_off8("ring, p2") == {"ring", "p2"}
    assert parse_off8(["relate"]) == {"relate"}
    with pytest.raises(ValueError):
        parse_off8("ring,nope")


def test_identity_at_init_equals_openobb7():
    torch.manual_seed(0)
    m7 = OpenOBB7("s", num_classes=NC, img_size=SMALL).eval()
    m8 = _model(rel_k=60, img=SMALL).eval()
    missing, unexpected = m8.load_state_dict(m7.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("refiner.") for k in missing)
    torch.manual_seed(1)
    x = torch.rand(2, 3, SMALL, SMALL)
    with torch.no_grad():
        a, b = m7(x), m8(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_refine_boxes_zero_is_exact_identity():
    b = torch.tensor([[[10.0, 20.0, 0.3, 0.2, 2.5], [100.0, 50.0, 80.0, 4.0, -0.7]]])
    assert torch.equal(refine_boxes(b, torch.zeros(1, 2, 5)), b)


def test_train_loss_backward_with_empty_image():
    m = _model().train()
    crit = OpenOBB8Loss(img_size=IMG)
    x = torch.rand(2, 3, IMG, IMG)
    out = m(x, _targets())
    assert {"ref_logits", "ref_boxes", "ref_cand"} <= set(out)
    K = out["ref_logits"].shape[1]
    assert out["ref_boxes"].shape == (2, K, 5) and out["ref_cand"].shape == (2, K, 5) and K == 20 + 3
    losses = crit(out, _targets())
    assert {"loss_ref_cls", "loss_ref_box"} <= set(losses)
    total = sum(losses.values())
    assert torch.isfinite(total) and losses["loss_ref_box"].item() > 0
    total.backward()
    ref = dict(m.refiner.named_parameters())
    for name in ("cls_out.weight", "box_out.weight"):        # zero-initialised outputs: only they move at step 0
        assert ref[name].grad is not None and ref[name].grad.abs().sum() > 0, name
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    with torch.no_grad():
        for p in m.refiner.parameters():
            if p.grad is not None:
                p -= 0.05 * p.grad.sign()
    m.zero_grad()
    sum(crit(m(x, _targets()), _targets()).values()).backward()
    for name in ("rel.0.qkv.weight", "pair_out.0.weight", "local.0.linear1.weight", "in2.weight", "ring4.weight"):
        assert ref[name].grad is not None and ref[name].grad.abs().sum() > 0, name
    m.zero_grad()
    with torch.no_grad():
        for p in m.refiner.pair_out.parameters():
            p += 0.05
    sum(crit(m(x, _targets()), _targets()).values()).backward()
    assert ref["pair.0.weight"].grad is not None and ref["pair.0.weight"].grad.abs().sum() > 0


def test_train_all_empty_batch():
    m = _model().train()
    crit = OpenOBB8Loss(img_size=IMG)
    t = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}] * 2
    losses = crit(m(torch.rand(2, 3, IMG, IMG), t), t)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()


@pytest.mark.parametrize("part", PARTS8)
def test_each_switch_runs(part):
    m = _model(off=part, img=SMALL)
    if part == "refine":
        assert m.refiner is None
    crit = OpenOBB8Loss(img_size=SMALL)
    m.train()
    out = m(torch.rand(2, 3, SMALL, SMALL), _targets())
    total = sum(crit(out, _targets()).values())
    assert torch.isfinite(total)
    total.backward()
    m.eval()
    with torch.no_grad():
        ev = m(torch.rand(1, 3, SMALL, SMALL))
    assert ev["pred_logits"].shape == (1, 1000, NC) and torch.isfinite(ev["pred_boxes"]).all()


def _rand_boxes(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.rand(1, n, 2, generator=g) * 200
    wh = torch.rand(1, n, 2, generator=g) * 40 + 2
    th = (torch.rand(1, n, 1, generator=g) * 2 - 1) * math.pi
    return torch.cat([c, wh, th], -1).double()


def test_pair_geometry_rotation_and_flip_invariant():
    b = _rand_boxes()
    f = pair_geometry(b)
    assert f.shape == (1, 12, 12, 7)
    phi = 0.83
    R = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]], dtype=b.dtype)
    br = b.clone()
    br[..., :2] = (b[..., :2] - 50) @ R.T + torch.tensor([7.0, -3.0], dtype=b.dtype)
    br[..., 4] = b[..., 4] + phi
    assert torch.allclose(pair_geometry(br), f, atol=1e-4)
    bf = b.clone()
    bf[..., 4] = b[..., 4] + math.pi            # the same boxes, axes pointing the other way
    assert torch.allclose(pair_geometry(bf), f, atol=1e-4)


def _core_accepts(m, x, train):
    """Input sizes the OpenOBB7 core cannot take are not the refiner's business."""
    m.train(train)
    try:
        with torch.no_grad():
            OpenOBB7.forward(m, x)
        return True
    except RuntimeError:
        return False


@pytest.mark.parametrize("hw", [(192, 256), (200, 232)])
def test_odd_input_size(hw):
    m = _model()
    x = torch.rand(1, 3, *hw)
    tr, ev = _core_accepts(m, x, True), _core_accepts(m, x, False)
    if not (tr or ev):
        pytest.skip(f"OpenOBB7 core does not accept {hw}")
    if tr:
        m.train()
        out = m(x, _targets()[:1])
        total = sum(OpenOBB8Loss(img_size=IMG)(out, _targets()[:1]).values())
        assert torch.isfinite(total)
    if ev:
        m.eval()
        with torch.no_grad():
            assert torch.isfinite(m(x)["pred_boxes"]).all()


def test_sample_reads_cell_centres_on_non_square_maps():
    from openobb.models.openobb8 import _sample
    h, w, s = 5, 7, 8
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    fmap = torch.stack([xs, ys])[None]                   # channel 0 = column index, 1 = row index
    pts = torch.tensor([[[(3 + 0.5) * s, (1 + 0.5) * s], [(6 + 0.5) * s, (4 + 0.5) * s], [4.0 * s, 2.5 * s]]])
    out = _sample(fmap, pts, s)[0]
    assert torch.allclose(out, torch.tensor([[3.0, 1.0], [6.0, 4.0], [3.5, 2.0]]), atol=1e-5)


def test_fuse_keeps_output():
    m = _model(img=SMALL).eval()
    with torch.no_grad():
        for p in (m.refiner.cls_out.weight, m.refiner.box_out.weight, m.refiner.pair_out[0].weight):
            p.normal_(0, 0.05)                         # a refiner that actually changes the output
    x = torch.rand(1, 3, SMALL, SMALL)
    with torch.no_grad():
        a = m(x)
        if not hasattr(m, "fuse"):
            pytest.skip("core has no fuse()")
        b = m.fuse()(x)
    keep = a["pred_logits"].amax(-1) > -20
    assert torch.allclose(a["pred_logits"][keep].sigmoid(), b["pred_logits"][keep].sigmoid(), atol=2e-3)


def test_apply_refinement_spreads_to_duplicates():
    from openobb.models.openobb8 import apply_refinement
    boxes = torch.tensor([[[50.0, 50.0, 12.0, 6.0, 0.0], [50.5, 50.0, 12.0, 6.0, 0.02], [150.0, 80.0, 12.0, 6.0, 0.0]]])
    cls = torch.tensor([[[3.0, -1.0], [2.8, -1.2], [2.0, -3.0]]])
    idx = torch.tensor([[0]])
    cand = boxes[:, :1]
    delta = torch.tensor([[[-5.0, 4.0]]])                 # candidate re-classified from class 0 to class 1
    ref_l = cls[:, :1] + delta
    ref_b = cand + torch.tensor([0.3, 0.0, 1.0, -1.0, 0.0])
    c2, b2 = apply_refinement(cls, boxes, idx, cand, delta, ref_l, ref_b)
    assert torch.allclose(c2[0, 0], ref_l[0, 0]) and torch.allclose(b2[0, 0], ref_b[0, 0])
    assert torch.allclose(c2[0, 1], cls[0, 1] + delta[0, 0])        # the duplicate follows its candidate
    assert torch.equal(c2[0, 2], cls[0, 2]) and torch.equal(b2[0, 1:], boxes[0, 1:])
    c3, _ = apply_refinement(cls, boxes, idx, cand, delta, ref_l, ref_b, spread_iou=None)
    assert torch.equal(c3[0, 1], cls[0, 1])
