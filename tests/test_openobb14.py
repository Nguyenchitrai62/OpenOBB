import math

import torch

from openobb.models.openobb1 import load_dfine_coco
from openobb.models.openobb14 import OpenOBB14, RepConv
from openobb.models.openobb14_loss import OpenOBB14Loss
from openobb.models.openobb3 import OpenOBB3

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v14_trains_every_part():
    torch.manual_seed(0)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert {"cls", "reg", "o2o_cls", "o2o_reg", "points", "strides", "ref_boxes", "ref_cand", "ref_valid"} <= set(out)
    assert out["reg"].shape[-1] == 5                                    # direct box regression by default
    losses = OpenOBB14Loss(img_size=IMG)(out, _targets())
    assert {"loss_cls", "loss_box", "loss_o2o_cls", "loss_ref_box"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    for part in ("neck.", "head.", "head_o2o.", "edge.", "backbone."):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(part) and p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), part


def test_edge_refiner_is_the_identity_at_init():
    torch.manual_seed(0)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG).eval()
    with torch.no_grad():
        f4, feats = m.encode(torch.rand(1, 3, IMG, IMG))
        b = torch.tensor([[[100.0, 80.0, 40.0, 6.0, 0.3], [50.0, 50.0, 10.0, 10.0, 1.0]]])
        assert torch.allclose(m.edge(f4, feats[0], b), b, atol=1e-6)
        pts = m.edge.points(b)                                           # edge rows sit on the two long edges
        c, L, T = b[0, 0, :2], b[0, 0, 2], b[0, 0, 3]
        rel = pts[0, 0] - c
        ax = torch.tensor([math.cos(0.3), math.sin(0.3)])
        nx = torch.tensor([-math.sin(0.3), math.cos(0.3)])
        v = rel @ nx
        assert torch.allclose(v[:6].abs(), torch.full((6,), T / 2)) and torch.allclose(v[6:12].abs(), torch.full((6,), T / 2))
        assert (rel @ ax).abs().max() <= L / 2 + 1e-4


def test_fuse_keeps_outputs_and_removes_every_bn():
    torch.manual_seed(0)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.3, 0.3)
                mod.running_var.uniform_(0.5, 1.5)
                mod.weight.uniform_(0.5, 1.5)
        for t in m.head.box:
            torch.nn.init.normal_(t.out.weight, std=0.05)
        x = torch.rand(1, 3, IMG, IMG)
        f4a, fa = m.encode(x)
        ca, ra = m.head(fa)
        m.fuse()
        f4b, fb = m.encode(x)
        cb, rb = m.head(fb)
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())
    assert all(r.conv1 is None and r.bn_id is None for r in m.modules() if isinstance(r, RepConv))
    assert torch.allclose(ca, cb, atol=1e-3) and torch.allclose(ra, rb, atol=1e-3)


def test_v14_loads_the_pretrained_trunk_as_it_is(tmp_path):
    """The HGNetv2 weights of a D-FINE / OpenOBB3 checkpoint load into backbone.* unchanged; everything else is new."""
    torch.manual_seed(0)
    src = OpenOBB3("s", num_classes=NC, img_size=IMG)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict()}, p)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG)
    skipped = load_dfine_coco(m, str(p), log=lambda *_: None)
    assert not any(k.startswith("backbone.") for k in skipped)
    assert all(k.startswith(("neck.", "head.", "head_o2o.", "edge.")) for k in skipped)
    for (k, v), (k2, v2) in zip(m.backbone.state_dict().items(), src.backbone.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


def test_v14_fine_tune_maps_class_rows_by_name(tmp_path):
    torch.manual_seed(0)
    src = OpenOBB14("s", num_classes=3, img_size=IMG)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict(), "classes": ["a", "b", "c"]}, p)
    m = OpenOBB14("s", num_classes=2, img_size=IMG)
    load_dfine_coco(m, str(p), log=lambda *_: None, class_map={0: 2, 1: 0})     # new classes [c, a]
    for lvl in range(3):
        w_src, w = src.head.cls[lvl].out.weight, m.head.cls[lvl].out.weight
        assert torch.equal(w[0], w_src[2]) and torch.equal(w[1], w_src[0])


def test_inference_shapes_heads_and_nms_settings():
    torch.manual_seed(0)
    m = OpenOBB14("s", num_classes=NC, img_size=256, max_det=40).eval()
    with torch.no_grad():
        r = m(torch.rand(2, 3, 224, 160))                                # rectangular input (multiple of 32)
        assert r["pred_logits"].shape == (2, 40, NC) and r["pred_boxes"].shape == (2, 40, 5)
        assert r["pred_boxes"][..., 4].abs().max() <= math.pi / 2 + 1e-6
        m.infer_head = "o2o"                                             # one-to-one head, NMS-free
        r2 = m(torch.rand(1, 3, 128, 128))
        s = r2["pred_logits"].sigmoid().amax(-1)
        assert torch.all(s[:, :-1] >= s[:, 1:])
        m.final_nms = 0.7
        assert m(torch.rand(1, 3, 128, 128))["pred_logits"].shape[1] == 40
        m.ref_k = 0                                                      # refiner off at inference
        assert m(torch.rand(1, 3, 128, 128))["pred_boxes"].shape == (1, 40, 5)


def test_off_switches_and_dfl_box():
    torch.manual_seed(0)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG, off="attn,edge,dual", box="dfl").train()
    assert m.edge is None and m.head_o2o is None and isinstance(m.neck.attn, torch.nn.Identity)
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert "o2o_cls" not in out and "ref_boxes" not in out and out["reg"].shape[-1] > 5
    total = sum(OpenOBB14Loss(img_size=IMG)(out, _targets()).values())
    assert math.isfinite(float(total.detach()))


def test_neck_dfine_control_loads_the_pretrained_encoder(tmp_path):
    torch.manual_seed(0)
    src = OpenOBB3("s", num_classes=NC, img_size=IMG)
    p = tmp_path / "src.pt"
    torch.save({"model": src.state_dict()}, p)
    m = OpenOBB14("s", num_classes=NC, img_size=IMG, neck="dfine")
    assert m.neck is None and m.hid == src.encoder.hidden_dim
    skipped = load_dfine_coco(m, str(p), log=lambda *_: None)
    assert not any(k.startswith(("backbone.", "encoder.")) for k in skipped)
    m.train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    total = sum(OpenOBB14Loss(img_size=IMG)(out, _targets()).values())
    assert math.isfinite(float(total.detach()))
    m.eval().fuse()
    with torch.no_grad():
        assert m(torch.rand(1, 3, 128, 128))["pred_boxes"].shape[1] == 1000


def test_registry_knows_v14():
    from openobb.cli import MODELS, RECIPE14
    from openobb.predict import KNOWN_ARCHS, check_arch
    assert MODELS["openobb14x"] == ("v14", "x") and "v14" in KNOWN_ARCHS
    assert check_arch({"args": {"arch": "v14"}}) == "v14" and RECIPE14["box"] == "direct"
