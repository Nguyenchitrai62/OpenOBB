import math

import torch

from openobb.models.openobb14 import OpenOBB14
from openobb.models.openobb14_loss import OpenOBB14Loss
from openobb.models.openobb15 import OpenOBB15

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v15_trains_with_p2_level():
    torch.manual_seed(0)
    m = OpenOBB15("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    n3 = sum((IMG // s) ** 2 for s in (16, 32, 64))
    assert out["cls"].shape[1] == n3 + (IMG // 8) ** 2 == out["o2o_cls"].shape[1]
    assert sorted(out["strides"].unique().tolist()) == [8.0, 16.0, 32.0, 64.0]
    losses = OpenOBB14Loss(img_size=IMG)(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    for part in ("hires.", "p2_head.", "p2_head_o2o.", "edge.", "neck.", "to_p3."):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(part) and p.grad is not None]
        assert grads, part
    assert any(g.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("hires.") for g in [p.grad])


def test_v15_core_levels_start_as_v14_at_half_resolution():
    """New modules are no-ops for P3-P5 at the start: v15 on a 2x input == v14 on the input halved (same weights)."""
    torch.manual_seed(0)
    v14 = OpenOBB14("s", num_classes=NC, img_size=128).eval()
    m = OpenOBB15("s", num_classes=NC, img_size=256).eval()
    sd = {k: v for k, v in v14.state_dict().items() if not k.startswith("edge.")}   # the refiner reads another map
    missing, unexpected = m.load_state_dict(sd, strict=False)
    new = {"hires", "to_p3", "p2_lat", "p2_down", "p2_fuse", "p2_head", "p2_head_o2o", "edge"}
    assert not unexpected and all(k.split(".")[0] in new for k in missing)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        _, f15 = m.encode(x)
        _, f14 = v14.encode(torch.nn.functional.avg_pool2d(x, 2))
    for a, b in zip(f15[:3], f14):
        assert torch.allclose(a, b, atol=1e-5)


def test_v15_inference_any_size_fused_and_refiner_reads_full_res():
    torch.manual_seed(0)
    m = OpenOBB15("s", num_classes=NC, img_size=256, max_det=40).eval()
    assert m.edge.s4 == 8 and m.edge.s3 == 16 and m.strides == (16, 32, 64, 8)
    with torch.no_grad():
        r = m(torch.rand(1, 3, 200, 136))                   # not a multiple of 64: padded inside
        assert r["pred_logits"].shape == (1, 40, NC) and r["pred_boxes"].shape == (1, 40, 5)
        hr, feats = m.encode(torch.rand(1, 3, 256, 256))
        b = torch.tensor([[[100.0, 80.0, 40.0, 6.0, 0.3]]])
        assert torch.allclose(m.edge(hr, feats[0], b), b, atol=1e-6)       # identity at init
        m.fuse()
        r2 = m(torch.rand(1, 3, 200, 136))
        assert r2["pred_boxes"].shape == (1, 40, 5)
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())


def test_v15_stage4_cut_and_narrow_head_on_both_necks():
    torch.manual_seed(0)
    for neck in ("own", "dfine"):
        m = OpenOBB15("s", num_classes=NC, img_size=IMG, off="stage4,widehead", neck=neck).train()
        assert isinstance(m.backbone.stages[3], torch.nn.Identity)
        assert not any(n.startswith("backbone.stages.3.") for n, _ in m.named_parameters())
        full = OpenOBB15("s", num_classes=NC, img_size=IMG, neck=neck)
        assert sum(p.numel() for p in m.parameters()) < sum(p.numel() for p in full.parameters())
        out = m(torch.rand(1, 3, IMG, IMG), _targets())
        total = sum(OpenOBB14Loss(img_size=IMG)(out, _targets()).values())
        assert math.isfinite(float(total.detach()))
        total.backward()
        m.eval().fuse()
        with torch.no_grad():
            assert m(torch.rand(1, 3, 192, 128))["pred_boxes"].shape[1] == 1000


def test_v15_neck_dfine_and_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS, input_size
    assert MODELS["openobb15x"] == ("v15", "x") and "v15" in KNOWN_ARCHS
    assert input_size({"arch": "v15"}, 1280) == 2560
    torch.manual_seed(0)
    m = OpenOBB15("s", num_classes=NC, img_size=IMG, neck="dfine").train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert math.isfinite(float(sum(OpenOBB14Loss(img_size=IMG)(out, _targets()).values()).detach()))
