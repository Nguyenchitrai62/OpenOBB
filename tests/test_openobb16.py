import math

import torch

from openobb.models.openobb14_loss import OpenOBB14Loss
from openobb.models.openobb15 import OpenOBB15
from openobb.models.openobb16 import OpenOBB16

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v16_trains_three_stages_and_fit():
    torch.manual_seed(0)
    m = OpenOBB16("s", num_classes=NC, img_size=IMG).train()
    assert len(m.stages()) == 3 and all(s.fit is not None for s in m.stages())
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert len(out["ref_stages"]) == 2 and out["ref_stages"][0][2] == 0.6 and out["ref_stages"][1][2] == 0.7
    assert out["ref_fit"].shape == out["ref_cand"].shape[:2]
    losses = OpenOBB14Loss(img_size=IMG)(out, _targets())
    assert {"loss_ref_box", "loss_ref_box2", "loss_ref_box3", "loss_ref_fit"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    for part in ("edge.", "edge_more.0.", "edge_more.1.", "edge_more.1.fit."):
        grads = [p.grad for n, p in m.named_parameters() if n.startswith(part) and p.grad is not None]
        assert grads, part


def test_v16_starts_as_v15():
    """Cascade = identity and fit ~ 1 at the start: same boxes as v15 and the same ranking of the scores."""
    torch.manual_seed(0)
    v15 = OpenOBB15("s", num_classes=NC, img_size=256, max_det=30).eval()
    m = OpenOBB16("s", num_classes=NC, img_size=256, max_det=30).eval()
    missing, unexpected = m.load_state_dict(v15.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("edge_more.", "edge.fit.")) for k in missing)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        a, b = v15(x), m(x)
    assert torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-5)
    assert torch.allclose(a["pred_logits"].sigmoid(), b["pred_logits"].sigmoid(), atol=0.02)


def test_fit_scores_rescale_only_the_refined_candidates():
    cls = torch.full((1, 6, 2), 2.0)
    idx = torch.tensor([[0, 3]])
    fit = torch.tensor([[-4.0, 4.0]])                     # p 0.018 / 0.982
    new = OpenOBB16.fit_scores(cls, idx, fit, 0.5)
    p = new.sigmoid()
    assert p[0, 0, 0] < 0.2 and abs(p[0, 3, 0] - torch.sigmoid(torch.tensor(2.0)) * 0.982 ** 0.5) < 1e-3
    assert torch.equal(new[0, 1], cls[0, 1]) and torch.equal(new[0, 5], cls[0, 5])


def test_registry_knows_v16():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS, input_size
    assert MODELS["openobb16x"] == ("v16", "x") and "v16" in KNOWN_ARCHS and input_size({"arch": "v16"}, 1280) == 2560
