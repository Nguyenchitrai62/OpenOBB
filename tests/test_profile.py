import torch

from openobb.models.openobb18 import OpenOBB22
from openobb.profile import profile_parts

torch.set_num_threads(1)


def test_profile_runs_on_cpu_and_lists_stages(tmp_path):
    m = OpenOBB22("s", num_classes=3, img_size=128, groups=[0, 1, 1])
    p = tmp_path / "v22.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v22", "size": "s", "img": 128, "groups": "0,1,1", "box": "direct"}}, p)
    times, total = profile_parts(str(p), imgsz=128, batch=1, runs=1, warmup=1, device="cpu", detail=True)
    assert {"trunk + neck", "head", "NMS", "  trunk stage 1"} <= set(times) and total > 0
    assert any(k.startswith("  neck.") for k in times)


def test_profile_channels_last_runs_on_cpu(tmp_path):
    m = OpenOBB22("s", num_classes=3, img_size=128, groups=[0, 1, 1])
    p = tmp_path / "v22.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v22", "size": "s", "img": 128, "groups": "0,1,1", "box": "direct"}}, p)
    times, total = profile_parts(str(p), imgsz=128, batch=1, runs=1, warmup=1, device="cpu", channels_last=True)
    assert total > 0
