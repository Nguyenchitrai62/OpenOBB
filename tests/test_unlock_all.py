"""unlock_all: nothing stays frozen (the HGNetv2 trunk of sizes l / x), starting exactly at the frozen values."""
import copy

import torch
import torch.nn as nn

from openobb.models.common import FrozenBatchNorm2d, unlock_all
from openobb.models.hgnetv2 import HGNetv2

torch.set_num_threads(1)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HGNetv2(name="B0", return_idx=[1, 2, 3], freeze_at=0, freeze_norm=True, pretrained=False)

    def forward(self, x):
        return self.backbone(x)


def _frozen_net():
    torch.manual_seed(0)
    net = _Net()
    for m in net.modules():                        # non-trivial frozen statistics
        if isinstance(m, FrozenBatchNorm2d):
            m.weight.uniform_(0.5, 1.5)
            m.bias.uniform_(-0.2, 0.2)
            m.running_mean.uniform_(-0.3, 0.3)
            m.running_var.uniform_(0.5, 2.0)
    return net.eval()


def test_unlock_starts_at_the_frozen_model():
    net = _frozen_net()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        ref = net(x)
    n_frozen = sum(isinstance(m, FrozenBatchNorm2d) for m in net.modules())
    assert n_frozen > 0 and not all(p.requires_grad for p in net.backbone.stem.parameters())
    n_bn, n_thaw = unlock_all(net)
    assert n_bn == n_frozen and n_thaw > 0
    assert not any(isinstance(m, FrozenBatchNorm2d) for m in net.modules())
    assert all(p.requires_grad for p in net.parameters())
    with torch.no_grad():
        out = net.eval()(x)
    for a, b in zip(ref, out):
        assert torch.allclose(a, b, atol=1e-5)


def test_unlocked_trunk_trains_bn_and_stem():
    net = _frozen_net()
    unlock_all(net)
    net.train()
    loss = sum(f.square().mean() for f in net(torch.rand(2, 3, 128, 128)))
    loss.backward()
    bn = next(m for m in net.backbone.modules() if isinstance(m, nn.BatchNorm2d))
    assert bn.weight.grad is not None and bn.weight.grad.abs().sum() > 0
    stem_w = next(net.backbone.stem.parameters())
    assert stem_w.grad is not None and stem_w.grad.abs().sum() > 0


def test_checkpoints_load_both_ways():
    frozen = _frozen_net()
    thawed = copy.deepcopy(frozen)
    unlock_all(thawed)
    thawed.train()
    with torch.no_grad():
        thawed(torch.rand(2, 3, 64, 64))           # running statistics move
    sd = thawed.state_dict()
    assert any(k.endswith("num_batches_tracked") for k in sd)
    fresh = _frozen_net()                          # default (frozen) model: inference / old loaders, strict
    fresh.load_state_dict(sd)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        for a, b in zip(thawed.eval()(x), fresh(x)):
            assert torch.allclose(a, b, atol=1e-5)
    again = _frozen_net()                          # resume: unfreeze first, then load the trained state
    unlock_all(again)
    again.load_state_dict(sd)


def test_cli_passes_and_inherits_the_flag(tmp_path, monkeypatch):
    import openobb.train as trainer
    from openobb.cli import ARCH_KEYS, train
    from tests.test_cli import _dataset
    assert "unlock_all" in ARCH_KEYS              # a fine-tune of an unlocked checkpoint stays unlocked
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv) + " "))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), model="openobb3s", name="a", unlock_all=True, **kw)
    assert "--unlock-all " in calls[-1]
    train(str(data), model="openobb3s", name="b", **kw)
    assert "--unlock-all" not in calls[-1]

