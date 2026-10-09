import math
import os

import pytest
import torch

from openobb.models.vrdet3 import ALONG_BINS, REG_CH, THICK_BINS, VRDet3, decode, fast_nms
from openobb.models.vrdet3_loss import VRDet3Loss, frame_targets

torch.set_num_threads(1)


def _onehot_reg(theta, across, left, right, thick, stride):
    reg = torch.zeros(1, 1, REG_CH)
    reg[..., 0], reg[..., 1] = theta, across / stride
    reg[..., 2 + int(left / stride)] = 40.0
    reg[..., 2 + ALONG_BINS + int(right / stride)] = 40.0
    reg[..., 2 + 2 * ALONG_BINS + int(thick / (stride / 2))] = 40.0
    return reg


def test_decode_known_box():
    th, s = math.radians(30), 8.0
    reg = _onehot_reg(th, across=-4.0, left=16.0, right=48.0, thick=12.0, stride=s)
    b = decode(reg, torch.tensor([[100.0, 60.0]]), torch.tensor([s]))[0, 0]
    du, a = (48.0 - 16.0) / 2, -4.0
    assert torch.allclose(b[2:4], torch.tensor([64.0, 12.0]), atol=1e-3)
    assert torch.allclose(b[0], torch.tensor(100 + du * math.cos(th) - a * math.sin(th)), atol=1e-3)
    assert torch.allclose(b[1], torch.tensor(60 + du * math.sin(th) + a * math.cos(th)), atol=1e-3)


def test_frame_targets_flip_wrap_and_square():
    # vertical wall (le90 theta = -pi/2), point 30 px "below" the centre, predicted angle +pi/2 (other wrap side)
    g = torch.tensor([[100.0, 100.0, 200.0, 6.0, -math.pi / 2]])
    p = torch.tensor([[102.0, 130.0]])
    lt, rt, at, L, T, th = frame_targets(p, torch.tensor([math.pi / 2]), g)
    assert torch.allclose(torch.stack([lt, rt]), torch.tensor([[130.0], [70.0]]), atol=1e-4)   # axis points down
    assert abs(float(L) - 200) < 1e-4 and abs(float(T) - 6) < 1e-4 and abs(float(th) - math.pi / 2) < 1e-4
    assert abs(float(at) - 2.0) < 1e-4                    # across axis n = (-1, 0): centre is +2 px along it
    # same point, predicted angle -pi/2: left / right swap, across flips
    lt2, rt2, at2, *_ = frame_targets(p, torch.tensor([-math.pi / 2]), g)
    assert torch.allclose(torch.stack([lt2, rt2]), torch.tensor([[70.0], [130.0]]), atol=1e-4) and abs(float(at2) + 2) < 1e-4
    # near-square door: prediction closer to its short side -> that side becomes the axis
    g2 = torch.tensor([[50.0, 50.0, 40.0, 30.0, 0.0]])
    *_, L2, T2, th2 = frame_targets(torch.tensor([[50.0, 50.0]]), torch.tensor([math.pi / 2 - 0.1]), g2)
    assert abs(float(L2) - 30) < 1e-4 and abs(float(T2) - 40) < 1e-4 and abs(float(th2) - math.pi / 2) < 1e-4
    # elongated wall never switches axis
    *_, L3, _, _ = frame_targets(torch.tensor([[100.0, 100.0]]), torch.tensor([0.0]), g)
    assert abs(float(L3) - 200) < 1e-4


def test_vrdet3_train_eval_and_loss():
    torch.manual_seed(0)
    m = VRDet3("s", num_classes=2, img_size=256)
    crit = VRDet3Loss(img_size=256)
    x = torch.rand(1, 3, 256, 256)
    t = [{"labels": torch.tensor([0, 1]),
          "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / 256, 0.0], [0.25, 0.7, 12 / 256, 10 / 256, 0.3]])}]
    m.train()
    losses = crit(m(x), t)
    assert set(losses) == {"loss_cls", "loss_box", "loss_dfl", "loss_across", "loss_angle"}
    total = sum(losses.values())
    assert torch.isfinite(total) and float(losses["loss_box"]) > 0
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        ev = m(x)
    assert ev["pred_logits"].shape == (1, 1000, 2) and ev["pred_boxes"].shape == (1, 1000, 5)
    from openobb.models.vrdet import postprocess
    s, l, b = postprocess(ev, 100, 256)
    assert torch.isfinite(b).all()


def test_fast_nms_removes_duplicates_keeps_neighbours():
    boxes = torch.tensor([[[100, 100, 200, 6, 0.0], [101, 100, 198, 6, 0.0], [100, 140, 200, 6, 0.0]]])
    logits = torch.tensor([[[3.0], [2.0], [1.0]]])
    lg, bx = fast_nms(logits, boxes, max_det=5)
    kept = (lg[0, :, 0] > -20).sum()
    assert kept == 2 and torch.allclose(bx[0, 1, 1], torch.tensor(140.0))      # parallel wall 40 px away survives


def test_backbone_encoder_match_vrdet1():
    """Same names / shapes as VRDet1: COCO weights and VRDet1 checkpoints load into the backbone, encoder, LSK."""
    from openobb.models.vrdet import VRDet
    a = VRDet("s", num_classes=2, img_size=256, lsk=True).state_dict()
    b = VRDet3("s", num_classes=2, img_size=256).state_dict()
    keys = [k for k in b if k.startswith(("backbone.", "encoder.", "lsk."))]
    assert keys and all(k in a and a[k].shape == b[k].shape for k in keys)


def test_vrdet3_x_size():
    n = sum(p.numel() for p in VRDet3("x", num_classes=7).parameters()) / 1e6
    assert 50 < n < 70


def test_cli_vrdet3_flags(tmp_path, monkeypatch):
    import openobb.train as trainer
    from test_cli import _dataset
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), model="vrdet3x", name="a", **kw)
    a = calls[-1] + " "
    for flag in ("--arch v3 ", "--size x ", "--lr 0.0005 ", "--backbone-mult 0.1 ", "--lsk ", "--clip 10.0 ",
                 "--num-top 1000 ", "--rfs 0.1 "):
        assert flag in a, flag
    for flag in ("--o2m-queries", "--aqd", "--eval-queries", "--no-pretrained"):
        assert flag not in a, flag


def test_predict_loads_v3_checkpoint(tmp_path):
    from openobb.predict import load_model
    m = VRDet3("s", num_classes=2, img_size=256)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v3", "size": "s", "img": 256, "lsk": True},
                "classes": ["door", "window"]}, p)
    net, names, _ = load_model(str(p), torch.device("cpu"))
    assert isinstance(net, VRDet3) and names == ["door", "window"]


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU pipeline check (set VRDET_SLOW=1)")
def test_vrdet3_cli_smoke(tmp_path):
    from test_cli import _dataset
    from openobb.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("vrdet3s")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
                cache_dir=str(tmp_path / "cache"), workers=0, max_iters=2, threads=1, no_pretrained=True)
    assert os.path.exists(r.best)
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1
