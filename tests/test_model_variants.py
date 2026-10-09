import pytest
import torch

from openobb.models.dense_head import DenseCriterion
from openobb.models.openobb1 import OpenOBB1, build_criterion

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
    m = OpenOBB1("s", img_size=256, **kw)
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
    from openobb.models.openobb1 import OpenOBB1
    torch.manual_seed(0)
    m = OpenOBB1("s", num_classes=3, img_size=256, num_denoising=0, lsk=True).eval()
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


def test_p2_level_keeps_coco_offsets_per_head():
    import os
    import pytest
    import torch
    from openobb.models.openobb1 import DFINE_URL, OpenOBB1, load_dfine_coco
    ck = os.path.expanduser("~/.cache/torch/hub/checkpoints/dfine_s_coco.pth")
    if not os.path.exists(ck):
        pytest.skip("D-FINE COCO checkpoint not cached")
    src = torch.load(ck, map_location="cpu", weights_only=False)
    src = src["ema"]["module"] if "ema" in src else src.get("model", src)
    m = OpenOBB1("s", num_classes=3, img_size=256, num_denoising=10, p2=True)
    load_dfine_coco(m, ck, class_names=["plane", "ship", "x"])
    k = next(k for k in m.state_dict() if k.endswith("cross_attn.sampling_offsets.weight"))
    new, old = m.state_dict()[k].view(8, -1, 256), src[k].view(8, -1, 256)
    assert new.shape[1] == old.shape[1] + 6                              # 3 extra points x (x, y) per head
    assert torch.equal(new[:, :old.shape[1]], old)                      # every head keeps its COCO offsets
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        assert m.eval()(x)["pred_boxes"].shape == (1, 300, 5)
