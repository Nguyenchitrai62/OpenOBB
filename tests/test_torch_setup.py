from openobb import torch_setup as ts


def test_cuda_index_matches_the_driver():
    assert ts.cuda_index((13, 0)) == "cu130"
    assert ts.cuda_index((12, 9)) == "cu128"
    assert ts.cuda_index((12, 4)) == "cu124"
    assert ts.cuda_index((11, 8)) == "cu118"
    assert ts.cuda_index((11, 4)) is None


def test_plan():
    gpu = ("NVIDIA GeForce RTX 3050", (12, 6))
    assert ts.plan(gpu, True, True) == (None, "ok")                     # CUDA build already: nothing to do
    assert ts.plan(gpu, True, False)[0] == "cu126"                      # CPU build on a GPU machine: CUDA build
    assert ts.plan(gpu, False, False)[0] == "cu126"                     # no torch, GPU: CUDA build
    assert ts.plan(None, True, False) == (None, "ok")                   # CPU machine with torch: keep it
    assert ts.plan(None, False, False)[0] == "cpu"                      # CPU machine without torch: CPU build
    assert ts.plan(("old", (10, 2)), True, False) == (None, "ok")       # driver too old: keep the CPU build


def test_ensure_torch_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(ts, "_DONE", None)
    monkeypatch.setenv("OPENOBB_AUTO_TORCH", "0")
    assert ts.ensure_torch(log=lambda *_: None) == "off"


def test_ensure_torch_installs_the_cuda_build_for_a_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(ts, "_DONE", None)
    monkeypatch.delenv("OPENOBB_AUTO_TORCH", raising=False)
    monkeypatch.setattr(ts.shutil, "which", lambda exe: "nvidia-smi" if exe == "nvidia-smi" else None)
    monkeypatch.setattr(ts, "nvidia_gpu", lambda: ("NVIDIA GeForce RTX 3050", (12, 8)))
    monkeypatch.setattr(ts, "_pip_install", lambda tag, log: calls.append(tag) or True)
    import torch
    if torch.version.cuda is not None:                                  # CUDA machine: the fast path applies
        assert ts.ensure_torch(log=lambda *_: None) == "ok" and not calls
    else:                                                               # CPU build loaded: installed for the next run
        assert ts.ensure_torch(log=lambda *_: None) == "restart" and calls == ["cu128"]
