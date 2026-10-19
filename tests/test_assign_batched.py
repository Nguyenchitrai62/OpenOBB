"""The batched dense assignment (F112, training speed) gives exactly the per-image / per-group assignment of
assign_slots, and the same losses through the whole v24 training loss."""
import math

import pytest
import torch

from openobb.models import openobb10_loss as L10
from openobb.models.openobb10_loss import assign_batched, assign_slots, pad_targets

torch.set_num_threads(1)


def _anchors(img=128):
    pts, st = [], []
    for s in (8, 16, 32):
        n = img // s
        y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        pts.append(torch.stack([x.flatten(), y.flatten()], -1).float().add(0.5).mul(s))
        st.append(torch.full((n * n,), float(s)))
    return torch.cat(pts), torch.cat(st)


def _gt(n, img, C, g):
    cx, cy = torch.rand(n, generator=g) * img, torch.rand(n, generator=g) * img
    w = torch.rand(n, generator=g) * 60 + 2
    h = torch.rand(n, generator=g) * 12 + 2
    th = (torch.rand(n, generator=g) - 0.5) * math.pi
    return torch.stack([cx, cy, w, h, th], -1), torch.randint(0, C, (n,), generator=g)


@pytest.mark.parametrize("budget", [None, 3000])          # one block / many blocks over problems
@pytest.mark.parametrize("topk,topk_max", [(10, 48), (1, 1)])
def test_assign_batched_equals_assign_slots(budget, topk, topk_max):
    g = torch.Generator().manual_seed(0)
    img, C, G = 128, 5, 3
    cg = torch.tensor([0, 1, 1, 2, 2])
    pts, st = _anchors(img)
    A = len(pts)
    targets, problems = [], []
    for b, n in enumerate((150, 0, 7, 40)):                  # > 64 objects (legacy chunks), an empty image
        gb, gl = _gt(n, img, C, g)
        targets.append({"boxes": torch.cat([gb[:, :4] / img, gb[:, 4:]], 1), "labels": gl})
    B = len(targets)
    scores = torch.rand(B, A, C, generator=g)
    boxes = torch.stack([torch.stack([_gt(A, img, C, g)[0] for _ in range(G)]) for _ in range(B)])   # (B, G, A, 5)
    pgb, pgl, valid = pad_targets(targets, cg, G, img, "cpu")
    k = (topk + pgb[..., 2] / 16.0).round().clamp(max=topk_max).long()
    fg, gi, ts = assign_batched(scores[:, None].expand(B, G, A, C).reshape(B * G, A, C), boxes.reshape(B * G, A, 5),
                                pts, st, pgl, pgb, valid, k, topk_max, budget=budget)
    for b, t in enumerate(targets):
        gb_all = t["boxes"].clone()
        gb_all[:, :4] *= img
        for gr in range(G):
            sel = cg[t["labels"]] == gr
            gb, gl = gb_all[sel], t["labels"][sel]
            kk = (topk + gb[:, 2] / 16.0).round().clamp(max=topk_max).long()
            f0, g0, t0 = assign_slots(scores[b], boxes[b, gr], pts, st, gl, gb, kk, 1)
            p = b * G + gr
            assert torch.equal(fg[p], f0), (b, gr)
            assert torch.allclose(ts[p], t0, atol=1e-6), (b, gr)
            if f0.any():                                     # same object (indices differ only by padding order)
                assert torch.allclose(pgb[p][gi[p][f0]], gb[g0[f0]])


def test_v24_loss_same_with_batched_assignment():
    from openobb.models.openobb18_loss import OpenOBB18Loss
    from openobb.models.openobb24 import OpenOBB24
    torch.manual_seed(0)
    m = OpenOBB24("s", num_classes=3, img_size=128, max_det=50, groups=[0, 1, 1]).train()
    g = torch.Generator().manual_seed(1)
    targets = []
    for n in (30, 3):
        gb, gl = _gt(n, 128, 3, g)
        targets.append({"boxes": torch.cat([gb[:, :4] / 128, gb[:, 4:]], 1), "labels": gl})
    out = m(torch.rand(2, 3, 128, 128), targets)
    res = {}
    from openobb.models import openobb17_loss as L17
    for mode in ("legacy", "batched"):
        L10.ASSIGN, L17.WINDOWED = mode, mode == "batched"
        try:
            res[mode] = OpenOBB18Loss(img_size=128)(out, targets)
        finally:
            L10.ASSIGN, L17.WINDOWED = "batched", True
    assert res["legacy"].keys() == res["batched"].keys()
    for key in res["legacy"]:
        assert torch.allclose(res["legacy"][key], res["batched"][key], rtol=1e-5, atol=1e-6), key


def test_windowed_field_targets_equal_full_grid():
    from openobb.models import openobb17_loss as L17
    g = torch.Generator().manual_seed(3)
    img = 256
    targets = []
    for n in (60, 0, 5):
        gb, gl = _gt(n, img, 3, g)
        if n >= 3:
            gb[:3, 2] = torch.tensor([400.0, 30.0, 3.0])     # longer than the image, small, tiny
            gb[:3, :2] = torch.tensor([[2.0, 250.0], [255.0, 1.0], [128.0, 128.0]])   # on the borders
        targets.append({"boxes": torch.cat([gb[:, :4] / img, gb[:, 4:]], 1), "labels": gl})
    full = L17._field_targets_full(targets, (img // 2, img // 2), img, "cpu")
    win = L17.field_targets(targets, (img // 2, img // 2), img, "cpu")
    assert full.shape == win.shape
    assert (full - win).abs().max() < 1e-6


def test_profile_train_runs(tmp_path, capsys):
    from openobb.models.openobb24 import OpenOBB24
    from openobb.profile import profile_train
    m = OpenOBB24("s", num_classes=3, img_size=128, max_det=50, groups=[0, 1, 1])
    ck = tmp_path / "v24s.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v24", "size": "s", "img": 128, "groups": "0,1,1"}}, ck)
    for assign in ("batched", "legacy"):
        steps, parts, real = profile_train(str(ck), imgsz=128, batch=1, runs=1, warmup=0, objects=10, assign=assign)
        assert real > 0 and "loss field targets" in parts and "loss   assignment o2m" in parts
    assert L10.ASSIGN == "batched"
    assert "step without syncs" in capsys.readouterr().out


def test_async_saver_snapshots_and_reports_errors(tmp_path):
    from openobb.train import AsyncSaver
    w = torch.zeros(3)
    s = AsyncSaver()
    s.save({"w": w, "opt": {"state": {0: {"m": w}}}, "n": 1}, tmp_path / "a.tmp", tmp_path / "a.pt")
    w += 5                                                 # the live tensor changes while the file is written
    s.wait()
    ck = torch.load(tmp_path / "a.pt")
    assert torch.equal(ck["w"], torch.zeros(3)) and torch.equal(ck["opt"]["state"][0]["m"], torch.zeros(3))
    assert not (tmp_path / "a.tmp").exists()
    s.save({"w": w}, tmp_path / "missing" / "b.tmp", tmp_path / "missing" / "b.pt")
    with pytest.raises(Exception):
        s.wait()


def test_frozen_bn_kernel_matches_expression():
    from openobb.models import common as CM
    bn = CM.FrozenBatchNorm2d(8)
    g = torch.Generator().manual_seed(0)
    for name in ("weight", "bias", "running_mean"):
        getattr(bn, name).copy_(torch.randn(8, generator=g))
    bn.running_var.copy_(torch.rand(8, generator=g) + 0.1)
    x = torch.randn(2, 8, 5, 5, generator=g, requires_grad=True)
    CM.FROZEN_BN_KERNEL = False
    try:
        ref = bn(x)
        (gr,) = torch.autograd.grad(ref.sum(), x)
    finally:
        CM.FROZEN_BN_KERNEL = True
    out = bn(x)
    (go,) = torch.autograd.grad(out.sum(), x)
    assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(go, gr, atol=1e-5)
    assert bn(x.bfloat16()).dtype == torch.bfloat16       # no promotion to fp32 (F112)


def test_compile_training_wraps_what_v24_runs():
    from openobb.models.openobb24 import OpenOBB24
    from openobb.train import compile_training
    torch.manual_seed(0)
    m = OpenOBB24("s", num_classes=3, img_size=128, max_det=50, groups=[0, 1, 1]).train()
    msg = compile_training(m)
    assert "trunk + neck" in msg and "heads" in msg and "boundary field" in msg
    assert "encode" in vars(m) and "run_head" in vars(m)  # instance attributes: the class (and EMA copies) untouched
    assert "encode" not in vars(OpenOBB24)


def test_compile_is_default_for_v18_family(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_cli_v78 import _calls
    a = _calls(tmp_path / "a", monkeypatch, "openobb24x")
    assert "--compile " in a                                   # F114: on by default for v18-v24
    b = _calls(tmp_path / "b", monkeypatch, "openobb24x", compile=False)
    assert "--compile " not in b
    c = _calls(tmp_path / "c", monkeypatch, "openobb12x")
    assert "--compile " not in c                               # older families keep the opt-in


def test_fused_copy_for_validation_keeps_outputs():
    from openobb.models.openobb24 import OpenOBB24
    from openobb.train import fused_copy
    torch.manual_seed(0)
    m = OpenOBB24("s", num_classes=3, img_size=128, max_det=50, groups=[0, 1, 1]).eval()
    vm = fused_copy(m, "cpu")
    assert vm is not m
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        a, b = m(x), vm(x)
    assert torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-3)
    assert any(type(mod).__name__ == "RepConv" for mod in m.modules())   # the EMA model itself stays unfolded
