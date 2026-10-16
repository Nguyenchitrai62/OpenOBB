import math
import os

import pytest
import torch

from openobb.models.openobb3 import REG_CH
from openobb.models.openobb3_loss import OpenOBB3Loss
from openobb.models.openobb6 import AxisContext, GlobalAttention, OpenOBB6, StripMix

torch.set_num_threads(1)


def _targets(n=3, nc=3):
    t = []
    for b in range(2):
        boxes = torch.tensor([[0.3, 0.4, 0.4, 0.03, 0.0], [0.6, 0.6, 0.05, 0.05, 0.0], [0.5, 0.2, 0.3, 0.02, 1.2]])
        t.append({"boxes": boxes[:n], "labels": torch.tensor([0, 1, 2])[:n] % nc})
    return t


def test_shapes_and_loss_backward():
    m = OpenOBB6("n", num_classes=3, img_size=256).train()
    x = torch.rand(2, 3, 256, 256)
    out = m(x)
    A = (256 // 8) ** 2 + (256 // 16) ** 2 + (256 // 32) ** 2
    assert out["cls"].shape == (2, A, 3) and out["reg"].shape == (2, A, REG_CH)
    losses = OpenOBB3Loss(img_size=256)(out, _targets())
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    g = [p.grad for p in m.backbone.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(v).all() for v in g)
    m.eval()
    with torch.no_grad():
        ev = m(x)
    assert ev["pred_logits"].shape[:2] == ev["pred_boxes"].shape[:2] and ev["pred_boxes"].shape[-1] == 5


def test_residual_blocks_start_as_identity():
    x = torch.randn(2, 32, 16, 16)
    for blk in (StripMix(32, 7), AxisContext(32), GlobalAttention(32, 8)):
        blk.eval()
        with torch.no_grad():
            assert torch.allclose(blk(x), x, atol=1e-6), type(blk).__name__


def test_fuse_keeps_outputs():
    torch.manual_seed(0)
    m = OpenOBB6("n", num_classes=2, img_size=128)
    for mod in m.modules():                                 # non-trivial BN statistics / scales
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    m.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        a = m.head(m.neck(m.backbone(x)))
        m.fuse()
        b = m.head(m.neck(m.backbone(x)))
    assert all(torch.allclose(u, v, atol=1e-3, rtol=1e-3) for u, v in zip(a, b))
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())


def test_odd_input_size():
    m = OpenOBB6("n", num_classes=2, img_size=200).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 200, 232))
    assert torch.isfinite(out["pred_boxes"]).all()


def test_cli_openobb6_flags(tmp_path, monkeypatch):
    import openobb.train as trainer
    from test_cli import _dataset
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1,
              optimizer="AdamW")   # the recipe defaults, without the optimizer=auto choices (F98)
    train(str(data), model="openobb6x", name="a", **kw)
    a = calls[-1] + " "
    for flag in ("--arch v6 ", "--size x ", "--lr 0.001 ", "--backbone-mult 1.0 ", "--wd 0.05 ", "--num-top 1000 "):
        assert flag in a, flag
    assert "--lsk" not in a


def test_predict_loads_v6_checkpoint_fused(tmp_path):
    from openobb.predict import load_model
    m = OpenOBB6("n", num_classes=2, img_size=256)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v6", "size": "n", "img": 256},
                "classes": ["door", "window"]}, p)
    net, names, _ = load_model(str(p), torch.device("cpu"))
    assert isinstance(net, OpenOBB6) and names == ["door", "window"]
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in net.modules())     # fused for inference


def test_pretrain_distillation_and_weights_load(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    import openobb.models.vit as vit
    from openobb.models.openobb1 import load_dfine_coco
    from openobb.pretrain import pretrain
    monkeypatch.setattr(vit.DinoV2Backbone, "load_pretrained", lambda self, log=print: None)   # no download
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(4):
        cv2.imwrite(str(d / f"{i}.png"), np.random.default_rng(i).integers(0, 255, (90, 120, 3), np.uint8))
    w = pretrain("openobb6n", data=str(d), epochs=2, imgsz=64, batch=2, teacher="dinov2_s", teacher_imgsz=56,
                 workers=0, project=str(tmp_path / "runs"), device="cpu")
    ck = torch.load(w, map_location="cpu", weights_only=False)
    assert ck["args"]["arch"] == "v6" and all(k.startswith("backbone.") for k in ck["model"])
    m = OpenOBB6("n", num_classes=5, img_size=256)
    load_dfine_coco(m, w, log=lambda *_: None, class_map={})
    k = next(iter(ck["model"]))
    assert torch.equal(m.state_dict()[k], ck["model"][k])


@pytest.mark.skipif(os.environ.get("OPENOBB_SLOW") != "1", reason="CPU pipeline check (set OPENOBB_SLOW=1)")
def test_openobb6_cli_smoke(tmp_path):
    from test_cli import _dataset
    from openobb import OpenOBB
    from openobb.cli import train
    data = _dataset(tmp_path / "ds")
    r = train(str(data), model="openobb6n", epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
              cache_dir=str(tmp_path / "cache"), workers=0, max_iters=2, threads=1)
    assert os.path.exists(r.best)
    res = OpenOBB(r.best).predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, verbose=False)
    assert len(res) == 1 and math.isfinite(float(res[0].obb.data.sum()))


def test_ablation_switches():
    from openobb.models.openobb6 import PARTS, parse_off
    base = OpenOBB6("n", num_classes=2, img_size=128)
    n0 = sum(p.numel() for p in base.parameters())
    x = torch.rand(1, 3, 128, 128)
    for part in PARTS:
        m = OpenOBB6("n", num_classes=2, img_size=128, off=part).eval()
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out["pred_boxes"]).all(), part
        assert sum(p.numel() for p in m.parameters()) < n0, part          # the part really is gone
    with pytest.raises(ValueError):
        parse_off("strips")
    assert parse_off(["attn", "spd"]) == frozenset({"attn", "spd"})


def test_cli_v6_off_is_passed_and_inherited(tmp_path, monkeypatch):
    import openobb.train as trainer
    from test_cli import _dataset
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1,
              optimizer="AdamW")   # the recipe defaults, without the optimizer=auto choices (F98)
    train(str(data), model="openobb6s", name="a", v6_off=["strip", "attn"], **kw)
    assert "--v6-off attn,strip " in calls[-1] + " "
