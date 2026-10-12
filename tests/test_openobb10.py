import math

import pytest
import torch

from openobb.models.openobb2_loss import assign
from openobb.models.openobb3 import ALONG_BINS
from openobb.models.openobb7 import fuse_model
from openobb.models.openobb8 import OpenOBB8
from openobb.models.openobb10 import SLOTS, OpenOBB10, parse_off10
from openobb.models.openobb10_loss import OpenOBB10Loss, assign_slots

torch.set_num_threads(1)

IMG, NC = 256, 3
SMALL = 128


def _targets():
    return [{"labels": torch.tensor([0, 1, 2, 1]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2], [0.26, 0.71, 9 / IMG, 6 / IMG, 0.0]])},
            {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}]


def test_parse_off10():
    assert parse_off10(None) == frozenset() and parse_off10("reach, balance") == {"reach", "balance"}
    with pytest.raises(ValueError):
        parse_off10("slots,nope")


def test_identity_at_init_equals_openobb8():
    torch.manual_seed(0)
    m8 = OpenOBB8("s", num_classes=NC, img_size=SMALL, rel_k=60).eval()
    m10 = OpenOBB10("s", num_classes=NC, img_size=SMALL, rel_k=60).eval()
    missing, unexpected = m10.load_state_dict(m8.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(("head.cls_slot.", "head.reg_slot.")) for k in missing)
    m10.head.copy_primary_box()
    # what reaches NMS: slot 0 = OpenOBB8's dense output exactly, extra slots below the NMS confidence (0.001).
    # (Final outputs are not compared: a random-init model has many tied scores and Fast-NMS order among ties
    # depends on the tensor length.)
    import openobb.models.openobb8 as m8mod
    cap, orig = {}, m8mod.fast_nms

    def spy(name):
        def f(cls, boxes, *a, **k):
            cap[name] = (cls.clone(), boxes.clone())
            return orig(cls, boxes, *a, **k)
        return f
    torch.manual_seed(1)
    x = torch.rand(2, 3, SMALL, SMALL)
    try:
        with torch.no_grad():
            m8mod.fast_nms = spy("a")
            m8(x)
            m8mod.fast_nms = spy("b")
            m10(x)
    finally:
        m8mod.fast_nms = orig
    (ca, ba), (cb, bb) = cap["a"], cap["b"]
    A = ca.shape[1]
    assert cb.shape[1] == SLOTS * A
    assert torch.equal(ca, cb[:, :A]) and torch.equal(ba, bb[:, :A])
    assert float(cb[:, A:].sigmoid().max()) < 1e-3


def test_slot_layout_and_fuse():
    torch.manual_seed(0)
    m = OpenOBB10("s", num_classes=NC, img_size=SMALL, rel_k=40).eval()
    with torch.no_grad():
        for p in m.head.cls_slot.parameters():          # make the extra slots visible
            p.normal_(0, 0.05)
        _, feats = m.encode(torch.rand(1, 3, SMALL, SMALL))
        cls, reg = m.head(feats)
        pts, st = m.anchors(feats)
    A = sum(f.shape[-2] * f.shape[-1] for f in feats)
    assert cls.shape[1] == reg.shape[1] == pts.shape[0] == st.shape[0] == SLOTS * A
    assert torch.equal(pts[:A], pts[A:2 * A]) and torch.equal(st[:A], st[2 * A:])
    x = torch.rand(1, 3, SMALL, SMALL)
    m.nms_conf = 0.0
    with torch.no_grad():
        ref = m(x)
        fuse_model(m)
        out = m(x)
    assert torch.allclose(ref["pred_boxes"], out["pred_boxes"], atol=1e-3)


def _random_case(seed, A=300, C=3, M=12):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(A, 2, generator=g) * 200
    st = torch.full((A,), 8.0)
    gb = torch.cat([torch.rand(M, 2, generator=g) * 200, 4 + torch.rand(M, 2, generator=g) * 40,
                    torch.rand(M, 1, generator=g) * 3], 1)
    gl = torch.randint(0, C, (M,), generator=g)
    scores = torch.rand(A, C, generator=g)
    boxes = torch.cat([pts + torch.randn(A, 2, generator=g) * 3, 4 + torch.rand(A, 2, generator=g) * 40,
                       torch.rand(A, 1, generator=g) * 3], 1)
    k = torch.full((M,), 10, dtype=torch.long)
    return scores, boxes, pts, st, gl, gb, k


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_slot_equals_openobb3_assign(seed):
    scores, boxes, pts, st, gl, gb, k = _random_case(seed)
    fg0, gi0, ts0, _ = assign(scores, boxes, pts, gl, gb, k, strides=st, across=1.0)
    fg1, gi1, ts1 = assign_slots(scores, boxes, pts, st, gl, gb, k, slots=1, across=1.0)
    assert torch.equal(fg0, fg1) and torch.equal(gi0[fg0], gi1[fg1]) and torch.allclose(ts0, ts1)


def test_slots_give_overlapping_objects_their_own_points():
    # two tiny overlapping objects of different classes on the same few stride-8 points (wall_300 beside a junction)
    xs, ys = torch.meshgrid(torch.arange(8) * 8.0 + 4, torch.arange(8) * 8.0 + 4, indexing="xy")
    pts = torch.stack([xs.flatten(), ys.flatten()], 1)
    A = len(pts)
    st = torch.full((A,), 8.0)
    gb = torch.tensor([[28.0, 28.0, 9.5, 9.5, 0.0], [30.0, 29.0, 10.0, 6.0, 0.0]])
    gl = torch.tensor([0, 1])
    k = torch.tensor([10, 10])
    box = torch.tensor([29.0, 28.5, 10.0, 8.0, 0.0])
    for S in (1, 3):
        scores = torch.full((S * A, 2), 0.3)
        boxes = box.repeat(S * A, 1)
        fg, gi, ts = assign_slots(scores, boxes, pts.repeat(S, 1), st.repeat(S), gl, gb, k, slots=S)
        got = set(gi[fg].tolist())
        if S == 1:
            assert len(got) == 1                     # one object wins every shared point
        else:
            assert got == {0, 1}                     # both objects are learnt, on different slots of a point
            pos = fg.nonzero()[:, 0]
            assert len(set((pos % A).tolist())) < len(pos)


def test_reach_keeps_only_points_that_can_represent_the_object():
    xs, ys = torch.meshgrid(torch.arange(100) * 8.0 + 4, torch.arange(4) * 8.0 + 4, indexing="xy")
    pts = torch.stack([xs.flatten(), ys.flatten()], 1)
    A = len(pts)
    st = torch.full((A,), 8.0)
    gb = torch.tensor([[400.0, 16.0, 700.0, 6.0, 0.0]])          # longer than 2 x 31 strides: no exact P3 point
    gb2 = torch.tensor([[400.0, 16.0, 400.0, 6.0, 0.0]])         # 400 px: only the middle points reach both ends
    scores = torch.full((A, 1), 0.5)
    boxes = gb2.repeat(A, 1)
    k = torch.tensor([200])
    fg, _, _ = assign_slots(scores, boxes, pts, st, torch.tensor([0]), gb2, k, reach=True)
    u = (pts[fg][:, 0] - 400).abs()
    assert fg.any() and bool((200 + u <= (ALONG_BINS - 1) * 8).all())
    fg_all, _, _ = assign_slots(scores, boxes, pts, st, torch.tensor([0]), gb2, k, reach=False)
    assert fg_all.sum() > fg.sum()
    fg_far, _, _ = assign_slots(scores, gb.repeat(A, 1), pts, st, torch.tensor([0]), gb, k, reach=True)
    assert fg_far.any()                                            # nothing reachable: falls back to the old rule


def test_loss_backward_slots_and_balance():
    torch.manual_seed(0)
    m = OpenOBB10("s", num_classes=NC, img_size=IMG, rel_k=40).train()
    crit = OpenOBB10Loss(img_size=IMG)
    out = m(torch.rand(2, 3, IMG, IMG), _targets())
    assert out["slots"] == SLOTS
    losses = crit(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.head.cls_slot[0][0].weight.grad is not None
    w = crit.pos_count
    assert w is not None and w.shape == (NC,)


def test_balance_weights_rare_classes_only():
    crit = OpenOBB10Loss(img_size=IMG)
    tsc = torch.zeros(1, 1000, 3)
    tsc[0, :900, 0] = 0.5
    tsc[0, 900:990, 1] = 0.5
    tsc[0, 990:, 2] = 0.5
    w = crit.class_weights(tsc)
    assert w[0] == 1.0 and 1.0 < w[1] < w[2] <= 4.0


def test_all_off_equals_openobb8_loss():
    from openobb.models.openobb8_loss import OpenOBB8Loss
    torch.manual_seed(0)
    m = OpenOBB10("s", num_classes=NC, img_size=IMG, rel_k=40, off="slots").train()
    torch.manual_seed(3)
    out = m(torch.rand(2, 3, IMG, IMG), _targets())
    a = OpenOBB10Loss(img_size=IMG, off="slots,reach,balance")(out, _targets())
    b = OpenOBB8Loss(img_size=IMG)(out, _targets())
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k]) for k in a)


def test_cli_name_and_checkpoint_roundtrip(tmp_path):
    from openobb.cli import MODELS
    from openobb.predict import load_model
    assert MODELS["openobb10x"] == ("v10", "x") and MODELS["openobb10s"] == ("v10", "s")
    torch.manual_seed(0)
    m = OpenOBB10("s", num_classes=NC, img_size=SMALL, rel_k=40)
    p = tmp_path / "v10.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v10", "size": "s", "img": SMALL, "rel_k": 40}}, p)
    net, names, _ = load_model(str(p), torch.device("cpu"))
    assert type(net).__name__ == "OpenOBB10" and net.slots == SLOTS and names == ["a", "b", "c"]
