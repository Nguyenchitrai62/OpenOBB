"""OpenOBB9 two-teacher distillation pretraining (openobb/pretrain.py), CPU, tiny, no downloads."""
import json

import pytest
import torch

from openobb.models.openobb1 import load_dfine_coco      # the real loader (the fixture replaces the module attribute)

torch.set_num_threads(1)


def _imgs(d, n, seed=0):
    import cv2
    import numpy as np
    d.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{i}.png"), np.random.default_rng(seed + i).integers(0, 255, (90, 120, 3), np.uint8))
    return d


@pytest.fixture
def no_downloads(monkeypatch):
    """DINOv2 and D-FINE weights are never downloaded in tests (random teachers); records the D-FINE sizes asked."""
    import openobb.models.openobb1 as o1
    import openobb.models.vit as vit
    asked = []
    monkeypatch.setattr(vit.DinoV2Backbone, "load_pretrained", lambda self, log=print: None)
    monkeypatch.setattr(o1, "load_dfine_coco", lambda model, src, *a, **k: asked.append(src) or [])
    return asked


def test_pretrain9_two_teachers_gate_and_weights_load(tmp_path, no_downloads):
    from openobb.models.openobb9 import OpenOBB9
    from openobb.pretrain import pretrain
    train, val = _imgs(tmp_path / "train", 4), _imgs(tmp_path / "val", 2, seed=100)
    w = pretrain("openobb9s", data=str(train), val_data=str(val), teachers="dfine,dinov2", epochs=2, imgsz=64,
                 batch=2, teacher="dinov2_s", teacher_imgsz=42, teacher_dfine="s", workers=0,
                 project=str(tmp_path / "runs"), device="cpu")
    assert no_downloads == ["s"]                     # teacher size is a parameter (x by default, s here)

    log = json.loads((tmp_path / "runs" / "train" / "pretrain_log.json").read_text())
    assert [r["epoch"] for r in log] == [1, 2]
    want = {"dfine/P3", "dfine/P4", "dfine/P5", "dinov2/P4", "dinov2/P5"}     # every level P3-P5 is distilled
    for r in log:
        assert set(r["loss_pairs"]) == want and set(r["val_cos"]) == want
        assert all(-1.0 <= v <= 1.0 for v in r["val_cos"].values())
        assert r["img_per_s"] > 0 and r["t2_pass"] in (True, False)

    ck = torch.load(w, map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["arch"] == "v9" and a["size"] == "s" and a["img"] == 64 and a["pretrain"] == "distill"
    assert a["teachers"] == ["dfine", "dinov2"] and a["teacher"] == "dinov2_s" and ck["classes"] == []
    keys = list(ck["model"])
    assert keys and all(k.startswith(("backbone.", "neck.")) for k in keys)
    assert any(k.startswith("neck.") for k in keys) and any(k.startswith("backbone.") for k in keys)

    m = OpenOBB9("s", num_classes=5, img_size=128)   # what `openobb train model=openobb9s weights=...` does
    load_dfine_coco(m, w, log=lambda *_: None, class_map={})
    sd = m.state_dict()
    assert all(torch.equal(sd[k], ck["model"][k]) for k in keys)


def test_pretrain9_teacher_channels_follow_the_teacher(tmp_path, no_downloads):
    """The projector output width comes from the teacher model (hidden_dim / DINOv2 width), not a constant."""
    from openobb.models.openobb3 import OpenOBB3
    from openobb.pretrain import _pairs, _Teachers
    tch = _Teachers(["dfine", "dinov2"], 64, "s", "dinov2_s", ["mid", "last"], 42)
    assert tch.channels["dfine"] == OpenOBB3("s", num_classes=1, img_size=64, lsk=False).encoder.hidden_dim == 256
    assert tch.channels["dinov2"] == 384
    x = torch.rand(1, 3, 64, 64)
    out = tch(x)
    assert [f.shape[1] for f in out["dfine"]] == [256] * 3 and [f.shape[-1] for f in out["dfine"]] == [8, 4, 2]
    assert len(out["dinov2"]) == 2 and out["dinov2"][0].shape[1] == 384
    names = {p[0] for p in _pairs(["dfine", "dinov2"], tch, [1, 1, 1], [0.5, 0.5])}
    assert {"dinov2_mid/P4", "dinov2_mid/P5", "dinov2/P4", "dinov2/P5", "dfine/P3"} <= names
    assert not any(p.requires_grad for p in tch.parameters())
    tch.train()
    assert not tch.training                          # teachers stay frozen (BatchNorm statistics too)


def test_pretrain9_every_level_is_distilled_without_dfine(tmp_path, no_downloads):
    from openobb.pretrain import pretrain
    train = _imgs(tmp_path / "train", 2)
    pretrain("openobb9s", data=str(train), teachers="dinov2", epochs=1, imgsz=64, batch=2, teacher="dinov2_s",
             teacher_imgsz=42, workers=0, project=str(tmp_path / "runs"), device="cpu")
    log = json.loads((tmp_path / "runs" / "train" / "pretrain_log.json").read_text())
    assert {k.split("/")[1] for k in log[0]["loss_pairs"]} == {"P3", "P4", "P5"}
    assert no_downloads == []                         # no D-FINE teacher built


def test_pretrain_rejects_bad_options(tmp_path, no_downloads):
    from openobb.pretrain import pretrain
    train = _imgs(tmp_path / "train", 2)
    kw = dict(data=str(train), epochs=1, imgsz=64, batch=2, workers=0, project=str(tmp_path / "runs"), device="cpu")
    with pytest.raises(SystemExit):
        pretrain("openobb9s", teachers="clip", **kw)
    with pytest.raises(SystemExit):
        pretrain("openobb6n", teachers="dfine,dinov2", **kw)
    with pytest.raises(SystemExit):
        pretrain("openobb3x", **kw)


def test_teacher_from_checkpoint_path(tmp_path):
    """teacher_dfine=<OpenOBB .pt> (e.g. a core trained on DOTA): the teacher core is that checkpoint, size read
    from its args, path case kept."""
    from openobb.models.openobb3 import OpenOBB3
    from openobb.pretrain import _Teachers
    torch.manual_seed(0)
    src = OpenOBB3("s", num_classes=3, img_size=64, lsk=False)
    p = tmp_path / "Dota_Core_S.pt"
    torch.save({"model": src.state_dict(), "args": {"size": "s"}, "classes": ["a", "b", "c"]}, p)
    tch = _Teachers(["dfine"], 64, str(p), "dinov2_s", ["last"], 42, log=lambda *_: None)
    assert tch.channels["dfine"] == 256
    src.eval()                                       # teacher = the saved weights, eval statistics
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        want = src.encoder(src.backbone(x))
    got = tch(x)["dfine"]
    assert len(got) == 3 and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(want, got))
    with pytest.raises(SystemExit):
        _Teachers(["dfine"], 64, str(tmp_path / "missing.pt"), "dinov2_s", ["last"], 42)
