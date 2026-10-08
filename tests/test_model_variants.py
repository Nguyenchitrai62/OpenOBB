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


def test_lsk_adapter_starts_as_identity():
    import torch
    from vrdet.models.vrdet import VRDet
    torch.manual_seed(0)
    m = VRDet("s", num_classes=3, img_size=256, num_denoising=0, lsk=True).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        a = m(x)["pred_boxes"]
        lsk, m.lsk = m.lsk, None
        b = m(x)["pred_boxes"]
    assert torch.allclose(a, b, atol=1e-6)
    m.lsk = lsk
    m.train()
    out = m(x, [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3]])}])
    out["pred_logits"].sum().backward()
    assert m.lsk[0].out.weight.grad is not None and m.lsk[0].out.weight.grad.abs().sum() > 0
