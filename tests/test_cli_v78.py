"""CLI / trainer wiring of OpenOBB7 and OpenOBB8 (the models have their own tests)."""
import torch

torch.set_num_threads(1)


def _calls(tmp_path, monkeypatch, model, **extra):
    import openobb.train as trainer
    from test_cli import _dataset
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv) + " "))
    data = _dataset(tmp_path / "ds")
    train(str(data), model=model, name="a", epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"),
          cache_dir=str(tmp_path / "c"), workers=1, **extra)
    return calls[-1]


def test_openobb8_core_switches(tmp_path, monkeypatch):
    a = _calls(tmp_path, monkeypatch, "openobb8x", v7_off=["spd", "cls_ctx"])
    for flag in ("--arch v8 ", "--size x ", "--lr 0.0005 ", "--backbone-mult 0.1 ", "--lsk ", "--num-top 1000 ",
                 "--v7-off cls_ctx,spd "):
        assert flag in a, flag
    assert "--no-pretrained" not in a                   # COCO-pretrained core, as OpenOBB3


def test_openobb7_is_not_a_separate_model():
    from openobb.cli import MODELS
    assert not any(k.startswith("openobb7") for k in MODELS) and "openobb8x" in MODELS


def test_openobb8_flags(tmp_path, monkeypatch):
    a = _calls(tmp_path, monkeypatch, "openobb8x", v8_off="relate")
    for flag in ("--arch v8 ", "--size x ", "--lsk ", "--rel-k 600 ", "--v8-off relate "):
        assert flag in a, flag
