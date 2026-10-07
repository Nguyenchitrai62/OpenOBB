import pytest
import torch

from vrdet.models.dense_head import DenseCriterion
from vrdet.models.vrdet import VRDet, build_criterion

torch.set_num_threads(1)     # torch 2.12 CPU kernels race with many threads on this machine


def _targets():
    g = torch.Generator().manual_seed(0)
    b = torch.rand(5, 5, generator=g)
    b[:, 2:4] = b[:, 2:4] * 0.2 + 0.02
    return [{"labels": torch.randint(0, 15, (5,), generator=g), "boxes": b},
            {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)}]


@pytest.mark.parametrize("kw,box_loss", [(dict(), "kld"),
                                         (dict(dense=True, strip_k=11, ortho_heads=True), "probiou"),
                                         (dict(dense_queries=True), "kld")])
def test_variant_trains_one_step(kw, box_loss):
    torch.manual_seed(0)
    m = VRDet("s", img_size=256, **kw)
    crit = build_criterion(box_loss=box_loss)
    t = _targets()
    out = m(torch.rand(2, 3, 256, 256), t)
    losses = crit(out, t)
    if kw.get("dense") or kw.get("dense_queries"):
        losses.update(DenseCriterion()(out, t))
    sum(losses.values()).backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad():
        o = m(torch.rand(1, 3, 256, 256))
    assert o["pred_boxes"].shape == (1, 300, 5)
    if kw.get("dense") or kw.get("dense_queries"):
        assert o["dense_logits"].shape[1] == 32 * 32 + 16 * 16 + 8 * 8
