import math

import torch

from openobb.models.openobb12 import OpenOBB12
from openobb.models.openobb18 import OpenOBB18, distill_loss, teacher_feats
from openobb.models.openobb18_loss import OpenOBB18Loss

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 6 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v18_replaces_encoder_and_lsk_with_own_neck():
    m = OpenOBB18("s", num_classes=NC, img_size=IMG)
    names = [n for n, _ in m.named_parameters()]
    assert not any(n.startswith(("encoder.", "lsk.", "spd.")) for n in names)
    assert any(n.startswith("neck.") for n in names) and any(n.startswith("field.") for n in names)
    _, feats = m.encode(torch.rand(1, 3, IMG, IMG))
    teacher = OpenOBB12("s", num_classes=NC, img_size=IMG).eval()
    _, tf = teacher.encode(torch.rand(1, 3, IMG, IMG))
    assert [f.shape for f in feats] == [f.shape for f in tf]          # the OpenOBB12 head reads the neck unchanged


def test_v18_loads_trained_v12_trunk_and_head():
    torch.manual_seed(0)
    v12 = OpenOBB12("s", num_classes=NC, img_size=IMG)
    v18 = OpenOBB18("s", num_classes=NC, img_size=IMG)
    missing, unexpected = v18.load_state_dict(v12.state_dict(), strict=False)
    assert all(k.startswith(("neck.", "field.", "snap.")) for k in missing)
    assert all(k.startswith(("encoder.", "lsk.", "spd.")) for k in unexpected)
    for k, v in v12.head.state_dict().items():                         # the trained head is copied as it is
        assert torch.equal(v18.head.state_dict()[k], v), k


def test_v18_trains_with_teacher_distillation():
    torch.manual_seed(0)
    teacher = OpenOBB12("s", num_classes=NC, img_size=IMG).eval()
    m = OpenOBB18("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    crit = OpenOBB18Loss(img_size=IMG, teacher=teacher)
    losses = crit(out, _targets())
    assert {"loss_cls", "loss_o2o_cls", "loss_field", "loss_ref_box", "loss_distill"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    grads = [p.grad for n, p in m.named_parameters() if n.startswith("neck.") and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in teacher.parameters())            # the teacher stays frozen
    assert "loss_distill" not in OpenOBB18Loss(img_size=IMG)(m(torch.rand(1, 3, IMG, IMG), _targets()), _targets())


def test_distill_loss_is_relative_and_zero_on_match():
    t = [torch.randn(1, 4, 8, 8) * 10, torch.randn(1, 4, 4, 4) * 0.1]
    assert float(distill_loss(t, t)) == 0.0
    assert abs(float(distill_loss([torch.zeros_like(x) for x in t], t)) - 1.0) < 1e-6   # scale-free per level
    teacher = OpenOBB12("s", num_classes=NC, img_size=128).eval()
    f = teacher_feats(teacher, torch.rand(1, 3, 128, 128))
    assert len(f) == 3 and not any(x.requires_grad for x in f)


def test_v18_fuse_and_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    torch.manual_seed(0)
    m = OpenOBB18("s", num_classes=NC, img_size=128, max_det=50).eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        a = m(x)
        b = m.fuse()(x)
    assert torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-3)
    assert MODELS["openobb18x"] == ("v18", "x") and "v18" in KNOWN_ARCHS


def test_cli_v18_uses_weights_as_teacher(tmp_path, monkeypatch):
    import openobb.train as trainer
    from openobb.cli import train
    from tests.test_cli import _dataset
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(argv))
    data = _dataset(tmp_path / "ds")
    w = tmp_path / "v12.pt"
    torch.save({"model": {}, "classes": ["door", "window"], "args": {"arch": "v12"}}, w)
    train(str(data), model="openobb18s", epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), name="t",
          cache_dir=str(tmp_path / "cache"), workers=1, weights=str(w))
    argv = " ".join(calls[0])
    assert f"--teacher {w}" in argv and f"--weights {w}" in argv
