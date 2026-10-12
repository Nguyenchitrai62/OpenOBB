import io

import pytest
import torch
import torch.nn as nn

from openobb.models.obb_decoder import distinct_topk
from openobb.models.openobb3 import decode
from openobb.models.openobb8_loss import OpenOBB8Loss
from openobb.models.openobb9 import PARTS9, Down, OpenOBB9, RepConv, parse_off9

torch.set_num_threads(1)

IMG, NC = 256, 3
SMALL = 128      # eval-heavy tests: fast_nms builds a 4000 x 4000 ProbIoU matrix at 256 px


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])},
            {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}]


def _model(off=None, off8=None, rel_k=40, seed=0, img=IMG):
    torch.manual_seed(seed)
    return OpenOBB9("s", num_classes=NC, img_size=img, off=off, off8=off8, rel_k=rel_k)


def _perturb_bn(model, seed=0):
    """Non-trivial BatchNorm statistics and affine parameters (fusing must not depend on the defaults)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                c = m.num_features
                m.running_mean.copy_(torch.randn(c, generator=g) * 0.2)
                m.running_var.copy_(torch.rand(c, generator=g) + 0.5)
                m.weight.copy_(torch.rand(c, generator=g) * 0.6 + 0.3)
                m.bias.copy_(torch.randn(c, generator=g) * 0.1)


def _close(a, b, rel=1e-4):
    return (a - b).abs().max() <= rel * a.abs().max() + 1e-5


def test_parse_off9():
    assert parse_off9(None) == frozenset() and parse_off9("rep, attn") == {"rep", "attn"}
    assert parse_off9(["spd"]) == {"spd"}
    with pytest.raises(ValueError):
        parse_off9("rep,nope")
    with pytest.raises(ValueError):
        OpenOBB9("s", NC, IMG, off8="nope")


def test_interface():
    m = _model().eval()
    with torch.no_grad():
        f4, feats = m.features(torch.rand(1, 3, IMG, IMG))
    assert f4.shape == (1, m.c4, IMG // 4, IMG // 4)
    assert [tuple(f.shape) for f in feats] == [(1, m.hid, IMG // s, IMG // s) for s in (8, 16, 32)]
    assert m.strides == (8, 16, 32) and m.arch == "v9"
    assert {k.split(".")[0] for k in m.state_dict()} == {"backbone", "neck", "head", "refiner"}
    for s in "smlx":                                    # widths: multiples of 32, x multiples of 64
        cfg = OpenOBB9(s, NC, IMG, off8="refine")
        div = 64 if s == "x" else 32
        assert all(c % div == 0 for c in cfg.backbone.channels + [cfg.hid])
    assert not any(isinstance(q, nn.Conv2d) and q.groups > 1 for q in m.backbone.modules()
                   if q is not getattr(m.backbone.attn, "pos", None))         # no depthwise in the core
    assert not any(isinstance(q, nn.Conv2d) and q.groups > 1 for q in m.neck.modules())


def test_train_loss_backward_with_empty_image():
    m = _model().train()
    crit = OpenOBB8Loss(img_size=IMG)
    out = m(torch.rand(2, 3, IMG, IMG), _targets())
    assert {"cls", "reg", "points", "strides", "ref_logits", "ref_boxes", "ref_cand", "ref_valid"} <= set(out)
    K = out["ref_logits"].shape[1]
    assert out["ref_boxes"].shape == (2, K, 5) and K == 20 + 3
    losses = crit(out, _targets())
    assert {"loss_ref_cls", "loss_ref_box"} <= set(losses)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    for name, p in list(m.backbone.named_parameters()) + list(m.neck.named_parameters()):
        if name.startswith("attn.") and name not in ("attn.g1", "attn.g2"):
            continue                                    # behind zero-initialised residual scales at step 0
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_train_all_empty_batch():
    m = _model().train()
    t = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}] * 2
    total = sum(OpenOBB8Loss(img_size=IMG)(m(torch.rand(2, 3, IMG, IMG), t), t).values())
    assert torch.isfinite(total)
    total.backward()


@pytest.mark.parametrize("c1,c2,d", [(16, 16, 1), (16, 16, 2), (16, 16, 3), (8, 16, 2)])
def test_repconv_fuse_exact(c1, c2, d):
    torch.manual_seed(0)
    r = RepConv(c1, c2, d)
    assert (r.bn_id is not None) == (c1 == c2)
    _perturb_bn(r)
    r.eval()
    x = torch.randn(2, c1, 13, 11)
    with torch.no_grad():
        a = r(x)
        r.fuse()
        b = r(x)
    assert r.conv1 is None and r.bn_id is None and isinstance(r.bn, nn.Identity) and r.conv.bias is not None
    assert r.conv.dilation == (d, d) and _close(a, b)


def test_fuse_keeps_raw_outputs_and_leaves_no_bn():
    m = _model(img=SMALL, rel_k=60)
    _perturb_bn(m)
    with torch.no_grad():
        for o in m.head.reg:                            # non-zero regression and refiner outputs
            nn.init.normal_(o[-1].weight, std=0.05)
        for p in (m.refiner.cls_out.weight, m.refiner.box_out.weight, m.refiner.pair_out[0].weight):
            p.normal_(0, 0.05)
    m.eval()
    x = torch.rand(1, 3, SMALL, SMALL)
    with torch.no_grad():
        f4, feats = m.features(x)
        cls, reg = m.head(feats)
        pts, st = m.anchors(feats)
        boxes = decode(reg, pts, st)
        idx = distinct_topk(cls.sigmoid().amax(-1), boxes, 60, 120)
        g = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
        cand, base = g(boxes), g(cls)
        ref = m.refiner(f4, feats[0], feats[1], base, cand, SMALL)
        n_rep = sum(isinstance(q, RepConv) for q in m.modules())
        assert m.fuse() is m
        f4b, featsb = m.features(x)
        clsb, regb = m.head(featsb)
        refb = m.refiner(f4b, featsb[0], featsb[1], base, cand, SMALL)
    assert n_rep > 0 and not any(isinstance(q, nn.BatchNorm2d) for q in m.modules())
    assert all(q.conv1 is None and q.bn_id is None for q in m.modules() if isinstance(q, RepConv))
    for a, b in zip([f4, *feats, cls, reg, *ref], [f4b, *featsb, clsb, regb, *refb]):
        assert _close(a, b), (a - b).abs().max()


def test_refiner_identity_at_init():
    full = _model(img=SMALL, rel_k=60).eval()
    plain = _model(img=SMALL, off8="refine").eval()
    assert plain.refiner is None
    missing, unexpected = plain.load_state_dict(full.state_dict(), strict=False)
    assert not missing and unexpected and all(k.startswith("refiner.") for k in unexpected)
    x = torch.rand(2, 3, SMALL, SMALL)
    with torch.no_grad():
        a, b = full(x), plain(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_state_dict_round_trip():
    m = _model(img=SMALL)
    _perturb_bn(m)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = _model(img=SMALL, seed=1)
    m2.load_state_dict(torch.load(buf, weights_only=True))
    x = torch.rand(1, 3, SMALL, SMALL)
    with torch.no_grad():
        a, b = m.eval()(x), m2.eval()(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def _check_removed(m, part):
    reps = [q for q in m.modules() if isinstance(q, RepConv)]
    if part == "rep":
        assert all(q.conv1 is None and q.bn_id is None for q in reps)
    elif part == "dilate":
        assert all(q.conv.dilation == (1, 1) for q in reps)
    elif part == "attn":
        assert isinstance(m.backbone.attn, nn.Identity)
    elif part == "spd":
        assert m.neck.spd is None
    elif part == "maxpool":
        assert not m.backbone.stem.maxpool and not any(q.maxpool for q in m.modules() if isinstance(q, Down))


def test_default_has_every_part():
    m = _model()
    reps = [q for q in m.modules() if isinstance(q, RepConv)]
    assert all(q.conv1 is not None for q in reps) and {q.conv.dilation[0] for q in reps} == {1, 2}  # s: 2 per stage
    assert not isinstance(m.backbone.attn, nn.Identity) and m.neck.spd is not None and m.backbone.stem.maxpool
    assert all(q.maxpool for q in m.modules() if isinstance(q, Down))
    assert {q.conv.dilation[0] for q in OpenOBB9("x", NC, IMG, off8="refine").modules()
            if isinstance(q, RepConv)} == {1, 2, 3}


@pytest.mark.parametrize("off,off8", [(p, None) for p in PARTS9] + [(None, "refine"), (None, "ring")])
def test_each_switch_runs(off, off8):
    m = _model(off=off, off8=off8, img=SMALL)
    if off:
        _check_removed(m, off)
    if off8 == "refine":
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
        m.fuse()
        ev = m(torch.rand(1, 3, SMALL, SMALL))
    assert torch.isfinite(ev["pred_boxes"]).all() and not any(isinstance(q, nn.BatchNorm2d) for q in m.modules())


@pytest.mark.parametrize("hw", [(96, 160), (100, 116)])
def test_odd_input_size(hw):
    m = _model(img=SMALL)
    x = torch.rand(1, 3, *hw)
    m.train()
    out = m(x, _targets()[:1])
    assert torch.isfinite(sum(OpenOBB8Loss(img_size=SMALL)(out, _targets()[:1]).values()))
    m.eval()
    with torch.no_grad():
        f4, feats = m.features(x)
        assert feats[0].shape[-2:] == ((hw[0] + 7) // 8, (hw[1] + 7) // 8)
        assert torch.isfinite(m(x)["pred_boxes"]).all()
