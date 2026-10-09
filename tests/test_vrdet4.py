import os

import pytest
import torch

from openobb.models.vit import DinoV2Backbone

torch.set_num_threads(1)


def _v4(nc=2, img=256):
    from openobb.models.vrdet import VRDet
    return VRDet("s", num_classes=nc, img_size=img, backbone="dinov2_s", dense=True, dense_v3=True,
                 dense_queries=True, lsk=True)


def test_vit_names_follow_dinov2_checkpoints():
    sd = DinoV2Backbone("dinov2_s", (256, 512, 1024)).vit.state_dict()
    expect = {"cls_token": (1, 1, 384), "pos_embed": (1, 1370, 384), "register_tokens": (1, 4, 384),
              "mask_token": (1, 384), "patch_embed.proj.weight": (384, 3, 14, 14), "blocks.0.norm1.weight": (384,),
              "blocks.0.attn.qkv.weight": (1152, 384), "blocks.0.attn.proj.weight": (384, 384),
              "blocks.0.ls1.gamma": (384,), "blocks.0.mlp.fc1.weight": (1536, 384), "blocks.11.ls2.gamma": (384,),
              "norm.weight": (384,)}
    for k, shape in expect.items():
        assert tuple(sd[k].shape) == shape, k


def test_backbone_maps_any_size():
    bb = DinoV2Backbone("dinov2_s", (256, 512, 1024)).eval()
    with torch.no_grad():
        p3, p4, p5 = bb(torch.rand(1, 3, 320, 256))           # not multiples of 14: padded inside
    assert p3.shape == (1, 256, 40, 32) and p4.shape == (1, 512, 20, 16) and p5.shape == (1, 1024, 10, 8)


def test_vrdet4_train_eval():
    from openobb.models.dense_head import dense_predict
    from openobb.models.vrdet import build_criterion
    from openobb.models.vrdet3_loss import DenseOrientedCriterion
    torch.manual_seed(0)
    m = _v4()
    t = [{"labels": torch.tensor([0, 1]),
          "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / 256, 0.0], [0.25, 0.7, 12 / 256, 10 / 256, 0.3]])}]
    m.train()
    out = m(torch.rand(1, 3, 256, 256), t)
    losses = build_criterion(num_classes=2)(out, t)
    dl = DenseOrientedCriterion(img_size=256)(out, t)
    assert set(dl) == {"loss_dense_cls", "loss_dense_box", "loss_dense_dfl", "loss_dense_across", "loss_dense_angle"}
    total = sum(losses.values()) + sum(dl.values())
    assert torch.isfinite(total)
    total.backward()
    assert m.backbone.vit.blocks[0].attn.qkv.weight.grad is not None
    m.eval()
    with torch.no_grad():
        ev = m(torch.rand(1, 3, 256, 256))
    assert "pred_logits" in ev and "dense_logits" in ev
    s, lab, b = dense_predict(ev, img_size=256)[0]
    assert torch.isfinite(b).all()


def test_adapter_trains_at_full_lr():
    from openobb.engine import param_groups
    m = _v4()
    groups = {g["name"]: g for g in param_groups(m, 1e-4, 0.5, 1e-4)}
    bb = {id(p) for p in groups["bb"]["params"] + groups["bb_norm"]["params"]}
    assert id(m.backbone.vit.blocks[0].attn.qkv.weight) in bb
    assert id(m.backbone.adapter["fuse"].conv.weight) not in bb


def test_cli_vrdet4_flags(tmp_path, monkeypatch):
    import openobb.train as trainer
    from test_cli import _dataset
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), model="vrdet4x", name="a", **kw)
    a = calls[-1] + " "
    for flag in ("--arch v4 ", "--size x ", "--backbone dinov2_b ", "--dense ", "--dense-v3 ", "--dense-queries ",
                 "--primary union ", "--lsk ", "--lr 0.0001 ", "--backbone-mult 0.5 ", "--o2m-queries 900 "):
        assert flag in a, flag


def test_predict_loads_v4_checkpoint(tmp_path):
    from openobb.predict import load_model
    m = _v4()
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v4", "size": "s", "img": 256, "lsk": True,
                                                  "backbone": "dinov2_s", "dense": True, "dense_v3": True,
                                                  "dense_queries": True, "primary": "union"},
                "classes": ["door", "window"]}, p)
    net, names, _ = load_model(str(p), torch.device("cpu"))
    assert net.backbone_name == "dinov2_s" and names == ["door", "window"]


@pytest.mark.skipif(os.environ.get("VRDET_NET") != "1", reason="downloads DINOv2 ViT-S (set VRDET_NET=1)")
def test_real_dinov2_checkpoint_loads():
    bb = DinoV2Backbone("dinov2_s", (256, 512, 1024))
    bb.load_pretrained(log=print)                           # raises if any tensor does not fit
    assert float(bb.vit.blocks[0].ls1.gamma.abs().mean()) > 1e-4


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU pipeline check (set VRDET_SLOW=1)")
def test_vrdet4_cli_smoke(tmp_path):
    from test_cli import _dataset
    from openobb.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("vrdet4x")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
                cache_dir=str(tmp_path / "cache"), workers=0, max_iters=2, threads=1, no_pretrained=True)
    assert os.path.exists(r.best)
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1
