"""compile= of predict / val (as Ultralytics): off by default, True or a torch.compile mode, OPENOBB_COMPILE=1 when not
given; a missing or failing compiler warns and leaves the model eager. torch.compile itself is replaced by a recorder
(a real compile takes minutes on the CPU of the dev machine)."""
import cv2
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture
def fake_compile(monkeypatch):
    """torch.compile -> identity that records (what, mode, dynamic); a C++ compiler counts as present."""
    calls = []

    def fake(fn, mode=None, dynamic=None, **kw):
        calls.append((getattr(fn, "__name__", type(fn).__name__), mode, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake)
    monkeypatch.setattr("torch._inductor.cpp_builder.get_cpp_compiler", lambda *a, **k: None)
    monkeypatch.delenv("OPENOBB_COMPILE", raising=False)
    return calls


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from openobb.models.openobb12 import OpenOBB12
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("m") / "best.pt"
    torch.save({"model": OpenOBB12("s", num_classes=2, img_size=128).state_dict(), "classes": ["a", "b"],
                "args": {"arch": "v12", "size": "s", "img": 128, "fit": True}}, p)
    return p


def test_compile_mode(monkeypatch):
    from openobb.predict import compile_mode
    monkeypatch.delenv("OPENOBB_COMPILE", raising=False)
    assert compile_mode() is None and compile_mode(False) is None and compile_mode(0) is None   # off by default
    assert compile_mode(True) == compile_mode(1) == compile_mode("True") == "default"
    assert compile_mode("reduce-overhead") == "reduce-overhead"
    assert compile_mode("max-autotune-no-cudagraphs") == "max-autotune-no-cudagraphs"
    with pytest.raises(ValueError):
        compile_mode("fastest")
    monkeypatch.setenv("OPENOBB_COMPILE", "1")
    assert compile_mode() == "default"                       # the environment variable when compile= is not given
    assert compile_mode(False) is None                       # an explicit argument wins
    monkeypatch.setenv("OPENOBB_COMPILE", "max-autotune")
    assert compile_mode() == "max-autotune"


def test_compiles_trunk_neck_and_field(fake_compile):
    from openobb.models.openobb17 import OpenOBB17
    from openobb.predict import fast_layout
    m = OpenOBB17("s", num_classes=2, img_size=128).eval()
    assert fast_layout(m, torch.device("cpu")) is m and fake_compile == []          # off by default
    fast_layout(m, torch.device("cpu"), compile="reduce-overhead")
    assert fake_compile == [("encode", "reduce-overhead", True), ("forward", "reduce-overhead", True)]
    assert "encode" in vars(m) and "forward" in vars(m.field)                      # heads stay eager
    keys = set(OpenOBB17("s", num_classes=2, img_size=128).state_dict())
    assert set(m.state_dict()) == keys                                              # checkpoints keep their keys


def test_compile_failure_warns_and_stays_eager(fake_compile, monkeypatch):
    from openobb.models.openobb12 import OpenOBB12
    from openobb.models.openobb3 import OpenOBB3
    from openobb.predict import fast_layout

    def broken(*a, **k):
        raise RuntimeError("no inductor")

    m = OpenOBB12("s", num_classes=2, img_size=128).eval()
    monkeypatch.setattr(torch, "compile", broken)
    with pytest.warns(UserWarning, match="torch.compile failed"):
        fast_layout(m, torch.device("cpu"), compile=True)
    assert "encode" not in vars(m)

    def no_cc(*a, **k):
        raise RuntimeError("cl.exe not found")

    monkeypatch.setattr("torch._inductor.cpp_builder.get_cpp_compiler", no_cc)
    with pytest.warns(UserWarning, match="no C\\+\\+ compiler"):
        fast_layout(m, torch.device("cpu"), compile=True)
    assert "encode" not in vars(m)
    with pytest.warns(UserWarning, match="only dense"):                             # DETR-style / older models
        fast_layout(OpenOBB3("s", num_classes=2, img_size=128).eval(), torch.device("cpu"), compile=True)


def test_training_copies_ignore_the_environment(fake_compile, monkeypatch):
    from openobb.models.openobb12 import OpenOBB12
    from openobb.train import fused_copy
    monkeypatch.setenv("OPENOBB_COMPILE", "1")
    fused_copy(OpenOBB12("s", num_classes=2, img_size=128).eval(), torch.device("cpu"))
    assert fake_compile == []                                 # per-epoch validation stays eager (training: compile=)


def test_predict_compile_argument(fake_compile, ckpt, monkeypatch):
    from openobb import OpenOBB
    img = np.full((96, 128, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 20), (110, 26), (0, 0, 0), -1)
    model = OpenOBB(str(ckpt))
    assert model.compile is False and fake_compile == []                        # YOLO default: eager
    eager = model.predict(img, conf=0.0, verbose=False)[0].obb.data
    r = model.predict(img, conf=0.0, verbose=False, compile=True)[0]
    assert model.compile == "default" and fake_compile == [("encode", "default", True)]
    assert torch.equal(r.obb.data, eager)
    model.predict(img, verbose=False, compile=True)
    assert len(fake_compile) == 1                                                # compiled once, not per call
    model.predict(img, verbose=False, compile=False)
    assert model.compile is False and "encode" not in vars(model.model)          # back to eager
    monkeypatch.setenv("OPENOBB_COMPILE", "1")
    assert OpenOBB(str(ckpt)).compile == "default"                               # environment variable at load
    assert OpenOBB(str(ckpt)).predict(img, verbose=False, compile=False) is not None


def test_cli_accepts_compile():
    import inspect

    from openobb import cli
    from openobb.model import UNSUPPORTED, OpenOBB
    assert "compile" in inspect.signature(cli.val).parameters
    assert "compile" in inspect.signature(cli.predict).parameters
    assert "compile" in inspect.signature(OpenOBB.predict).parameters and "compile" not in UNSUPPORTED
    assert cli.parse_kv(["compile=True", "x=reduce-overhead"]) == {"compile": True, "x": "reduce-overhead"}
