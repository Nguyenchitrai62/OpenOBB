"""`openobb strip`: deploy checkpoint with weights + architecture + classes only."""
import json

import pytest
import torch

from openobb.models.openobb3 import OpenOBB3
from openobb.predict import load_model
from openobb.strip import strip

torch.set_num_threads(1)


def _ckpt(tmp_path):
    torch.manual_seed(0)
    m = OpenOBB3("s", num_classes=2, img_size=128, lsk=True)
    args = {"arch": "v3", "size": "s", "img": 128, "lsk": True, "num_top": 300, "fit": True,
            "data": "/root/.cache/openobb/datasets/x_1280", "out": "runs/obb/train3", "lr": 1e-3, "seed": 0,
            "weights": "/content/drive/MyDrive/openobb8x.pt"}
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "epoch": 12, "args": args, "classes": ["wall", "door"],
                "val_mAP50_95": 0.4, "conf_thr": [0.3, 0.2], "conf_f1": [0.5, 0.6]}, p)
    return p, m


def test_strip_keeps_only_inference_fields_and_loads(tmp_path):
    p, m = _ckpt(tmp_path)
    out = strip(str(p), out=str(tmp_path / "anything.pt"))
    ck = torch.load(out, map_location="cpu", weights_only=True)
    assert set(ck) == {"model", "args", "classes", "conf_thr"}
    assert ck["args"] == {"arch": "v3", "size": "s", "img": 128, "fit": True, "num_top": 300, "lsk": True}
    text = json.dumps({"a": ck["args"], "c": ck["classes"]}).lower()
    assert "openobb" not in text and "/root" not in text and "runs/" not in text
    net, names, _ = load_model(out, torch.device("cpu"))
    assert names == ["wall", "door"]
    a = dict(m.eval().state_dict())
    b = torch.load(out, map_location="cpu", weights_only=True)["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_strip_half_and_refuses_foreign_files(tmp_path):
    p, _ = _ckpt(tmp_path)
    out = strip(str(p), half=True)
    assert out.endswith("best_deploy.pt")
    ck = torch.load(out, map_location="cpu", weights_only=True)
    assert all(v.dtype == torch.float16 for v in ck["model"].values() if v.is_floating_point())
    load_model(out, torch.device("cpu"))                        # float16 weights load into the float32 model
    torch.save({"model": {}}, tmp_path / "x.pt")
    with pytest.raises(SystemExit):
        strip(str(tmp_path / "x.pt"))


def test_check_arch_unknown_and_newer(tmp_path, capsys):
    from openobb import __version__
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS, check_arch
    assert {arch for arch, _ in MODELS.values()} <= set(KNOWN_ARCHS)     # every CLI model can be loaded back
    assert check_arch({"args": {"arch": "v8"}}) == "v8" and check_arch({"args": {}}) == "v1"
    with pytest.raises(SystemExit, match="'v99'.*trained with version 9.9.9.*update"):
        check_arch({"args": {"arch": "v99"}, "version": "9.9.9"}, "m.pt")
    check_arch({"args": {"arch": "v3"}, "version": "99.0.0"}, "m.pt")
    assert "trained with version 99.0.0" in capsys.readouterr().out
    check_arch({"args": {"arch": "v3"}, "version": __version__}, "m.pt")
    assert "WARNING" not in capsys.readouterr().out


def test_cpu_inference_flushes_denormals(tmp_path):
    import torch

    from openobb.models.openobb12 import OpenOBB12
    from openobb.predict import load_model
    p = tmp_path / "m.pt"
    torch.save({"model": OpenOBB12("s", num_classes=2, img_size=128).state_dict(), "classes": ["a", "b"],
                "args": {"arch": "v12", "size": "s", "img": 128}}, p)
    torch.set_flush_denormal(False)
    load_model(str(p), torch.device("cpu"))
    assert torch.tensor([1e-40], dtype=torch.float32).item() == 0.0      # denormal flushed to zero
    torch.set_flush_denormal(False)
