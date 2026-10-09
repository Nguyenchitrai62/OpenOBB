import os

import pytest
import torch

from vrdet.models.vrdet3 import REG_CH, OrientedHead
from vrdet.models.vrdet5 import GEO_CH, GeoOrientedHead, VRDet5, geometry_maps, relation_targets

torch.set_num_threads(1)


def _t():
    return [{"labels": torch.tensor([0, 1]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 6 / 256, 0.0], [0.25, 0.7, 12 / 256, 10 / 256, 0.3]])}]


def test_geometry_maps_finite_and_scale_free():
    r = torch.randn(2, REG_CH, 5, 7)
    g = geometry_maps(r, 8.0)
    assert g.shape == (2, GEO_CH, 5, 7) and torch.isfinite(g).all() and g.abs().max() < 4


def test_geo_head_starts_as_vrdet3_head():
    torch.manual_seed(0)
    a, b = OrientedHead([64] * 3, 3), GeoOrientedHead([64] * 3, 3)
    missing = b.load_state_dict(a.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("geo.") for k in missing)          # VRDet3 heads load into VRDet5
    for m in (a, b):
        torch.nn.init.normal_(m.reg[0][-1].weight, std=0.1)
    b.reg[0][-1].weight.data.copy_(a.reg[0][-1].weight.data)
    feats = [torch.randn(1, 64, s, s) for s in (8, 4, 2)]
    a.eval(), b.eval()
    with torch.no_grad():
        (ca, ra), (cb, rb) = a(feats), b(feats)
    assert torch.allclose(ca, cb, atol=1e-5) and torch.allclose(ra, rb, atol=1e-5)


def test_relation_targets_soft_iou():
    boxes = torch.tensor([[[128.0, 76.8, 179.2, 6.0, 0.0], [10.0, 10.0, 5.0, 5.0, 0.0]]])
    t = relation_targets(boxes, _t(), 256, 2)
    assert t[0, 0, 0] > 0.9 and t[0, 1].sum() == 0


@pytest.mark.parametrize("backbone", ["hgnet", "dinov2_s"])
def test_vrdet5_train_eval(backbone):
    from vrdet.models.vrdet5_loss import VRDet5Loss
    torch.manual_seed(0)
    m = VRDet5("s", num_classes=2, img_size=256, backbone=backbone, rel_k=50)
    m.train()
    out = m(torch.rand(1, 3, 256, 256))
    assert out["rel_logits"].shape == (1, 50, 2)
    losses = VRDet5Loss(img_size=256)(out, _t())
    assert {"loss_cls", "loss_box", "loss_dfl", "loss_across", "loss_angle", "loss_rel"} == set(losses)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert m.rescorer.out.weight.grad is not None and m.head.geo[0].weight.grad is not None
    m.eval()
    with torch.no_grad():
        ev = m(torch.rand(1, 3, 256, 256))
    assert ev["pred_logits"].shape == (1, 1000, 2) and torch.isfinite(ev["pred_boxes"]).all()
    assert (ev["pred_logits"] > -20).any(-1).sum() <= 50                    # only re-scored candidates are output


def test_cli_vrdet5_flags(tmp_path, monkeypatch):
    import vrdet.train as trainer
    from test_cli import _dataset
    from vrdet.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), model="vrdet5x", name="a", **kw)
    a = calls[-1] + " "
    for flag in ("--arch v5 ", "--size x ", "--backbone dinov2_b ", "--geo-cls ", "--relate ", "--rel-k 600 ",
                 "--lsk ", "--lr 0.0005 ", "--backbone-mult 0.2 ", "--num-top 1000 "):
        assert flag in a, flag
    train(str(data), model="vrdet5x", name="b", backbone="hgnet", **kw)
    assert "--backbone hgnet " in calls[-1] + " "


def test_predict_loads_v5_checkpoint(tmp_path):
    from vrdet.predict import load_model
    m = VRDet5("s", num_classes=2, img_size=256, backbone="hgnet", rel_k=50)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v5", "size": "s", "img": 256, "lsk": True,
                                                  "backbone": "hgnet", "geo_cls": True, "relate": True, "rel_k": 50},
                "classes": ["door", "window"]}, p)
    net, names, _ = load_model(str(p), torch.device("cpu"))
    assert isinstance(net, VRDet5) and net.rescorer.k == 50


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU pipeline check (set VRDET_SLOW=1)")
def test_vrdet5_cli_smoke(tmp_path):
    from test_cli import _dataset
    from vrdet.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("vrdet5x")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
                cache_dir=str(tmp_path / "cache"), workers=0, max_iters=2, threads=1, no_pretrained=True)
    assert os.path.exists(r.best)
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1
