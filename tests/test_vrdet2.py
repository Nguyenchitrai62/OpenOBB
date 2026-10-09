import math
import os

import pytest
import torch

from vrdet.models.vrdet2 import REG_BINS, VRDet2, decode
from vrdet.models.vrdet2_loss import VRDet2Loss

torch.set_num_threads(1)


def _targets(img=256):
    # one long thin horizontal segment and one small square, normalised (cx, cy, w, h, theta), w = long side
    return [{"labels": torch.tensor([0, 1]),
             "boxes": torch.tensor([[0.5, 0.3, 0.6, 3 / img, 0.0], [0.25, 0.7, 12 / img, 10 / img, 0.3]])}]


def test_decode_inverts_a_known_segment():
    pts = torch.tensor([[40.0, 52.0]])
    st = torch.tensor([8.0])
    L, th = 96.0, math.radians(30)
    reg = torch.zeros(1, 1, 4 + REG_BINS)
    reg[..., 0] = th
    reg[..., 1] = 0.5                                  # point sits halfway to the far end (du = L / 4)
    reg[..., 2] = math.log(L / 8)
    reg[..., 3] = -0.25                                # 2 px across
    reg[..., 4 + 2] = 30.0                             # thickness bin 2 -> 2 * 8 / 2 = 8 px
    b = decode(reg, pts, st)[0, 0]
    du, dv = L / 4, -2.0
    assert torch.allclose(b[2:4], torch.tensor([L, 8.0]), atol=1e-3)
    assert torch.allclose(b[0], torch.tensor(40 - du * math.cos(th) - dv * math.sin(th)), atol=1e-3)
    assert torch.allclose(b[1], torch.tensor(52 - du * math.sin(th) + dv * math.cos(th)), atol=1e-3)


def test_vrdet2_train_eval_shapes_and_finite_loss():
    torch.manual_seed(0)
    m = VRDet2("n", num_classes=2, img_size=256)
    crit = VRDet2Loss(img_size=256)
    x = torch.rand(1, 3, 256, 256)
    m.train()
    out = m(x)
    losses = crit(out, _targets())
    assert {"loss_cls", "loss_box", "loss_end", "loss_across", "loss_dfl", "loss_angle", "loss_cls_o2o"} <= set(losses)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        ev = m(x)
    A = sum((256 // s) ** 2 for s in m.strides)
    assert ev["pred_logits"].shape == (1, A, 2) and ev["pred_boxes"].shape == (1, A, 5)
    from vrdet.models.vrdet import postprocess
    s, l, b = postprocess(ev, 50, 256)
    assert s.shape[1] == 50 and b.shape[-1] == 5


def test_vrdet2_x_size():
    n = sum(p.numel() for p in VRDet2("x", num_classes=15).parameters()) / 1e6
    assert 50 < n < 65                                  # same budget as the largest one-stage OBB models


def test_cli_model_names(tmp_path, monkeypatch):
    import vrdet.train as trainer
    from vrdet.cli import train
    from test_cli import _dataset
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), model="vrdet2x", name="a", **kw)
    a = calls[-1] + " "
    for flag in ("--arch v2 ", "--size x ", "--lr 0.001 ", "--backbone-mult 1.0 ", "--clip 10.0 ", "--num-top 1000 ",
                 "--rfs 0.1 ", "--mosaic-p 1.0 "):
        assert flag in a, flag
    for flag in ("--lsk", "--o2m-queries", "--aqd", "--eval-queries"):
        assert flag not in a, flag
    train(str(data), model="vrdet1x", name="b", **kw)
    b = calls[-1] + " "
    assert "--arch" not in b and "--size x " in b and "--lsk" in b
    train(str(data), model="x", name="c", **kw)
    assert "--arch" not in calls[-1] and "--lsk" in calls[-1]                  # bare sizes stay VRDet1
    with pytest.raises(SystemExit):
        train(str(data), model="vrdet3x", name="d", **kw)


def test_predict_loads_v2_checkpoint(tmp_path):
    from vrdet.predict import load_model
    m = VRDet2("n", num_classes=2, img_size=256)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v2", "size": "n", "img": 256},
                "classes": ["door", "window"], "conf_thr": {"door": 0.3}}, p)
    net, names, a = load_model(str(p), torch.device("cpu"))
    assert isinstance(net, VRDet2) and names == ["door", "window"] and a["conf_thr"] == {"door": 0.3}


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU smoke train (set VRDET_SLOW=1)")
def test_vrdet2_cli_smoke(tmp_path):
    from test_cli import _dataset
    from vrdet.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("vrdet2n")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
                cache_dir=str(tmp_path / "cache"), workers=0, max_iters=2, threads=1)
    assert os.path.exists(r.best)
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1
