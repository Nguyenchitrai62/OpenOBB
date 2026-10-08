import torch

from vrdet.models.obb_criterion import OBBHungarianMatcher
from vrdet.models.vrdet import VRDet, build_criterion

torch.set_num_threads(1)


def test_one_to_many_assignment():
    m = OBBHungarianMatcher()
    gt = torch.tensor([[0.2, 0.2, 0.1, 0.05, 0.3], [0.7, 0.6, 0.2, 0.1, -0.4]])
    boxes = torch.rand(1, 40, 5) * torch.tensor([1, 1, 0.3, 0.3, 1.0])
    boxes[0, :5] = gt[0] + 0.002 * torch.arange(5)[:, None]      # 5 near-copies of target 0
    boxes[0, 5:8] = gt[1]                                          # 3 copies of target 1
    logits = torch.full((1, 40, 3), -4.0)
    logits[0, :5, 0] = 3.0
    logits[0, 5:8, 2] = 3.0
    out = m.one_to_many({"pred_logits": logits, "pred_boxes": boxes},
                        [{"labels": torch.tensor([0, 2]), "boxes": gt}], k=4)
    q, g = out[0]
    assert len(q) == len(set(q.tolist()))                          # a query serves one target only
    assert (g == 0).sum() <= 4 and (g == 1).sum() <= 4
    assert set(q[g == 1].tolist()) >= {5, 6, 7}                     # exact copies are taken
    assert set(q[g == 0].tolist()) <= set(range(5)) | set(range(8, 40)) and len(set(q[g == 0].tolist()) & set(range(5))) >= 3


def test_o2m_group_is_training_only_and_trains():
    torch.manual_seed(0)
    base = VRDet("s", num_classes=3, img_size=256, num_denoising=10).eval()
    torch.manual_seed(0)
    m = VRDet("s", num_classes=3, img_size=256, num_denoising=10, o2m_queries=120).eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(base(x)["pred_boxes"], m(x)["pred_boxes"], atol=1e-6)    # no extra parameters
    m.train()
    tg = [{"labels": torch.tensor([1, 2]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.2, 0.3, 0.1, 0.1, 0.0]])},
          {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.6, 0.4, 0.3, 0.2, -0.5]])}]
    out = m(x, tg)
    assert out["pred_logits"].shape[1] == 300 and len(out["o2m_outputs"]) == 3
    assert out["o2m_outputs"][-1]["pred_logits"].shape[1] == 120
    crit = build_criterion(num_classes=3, o2m_k=4)
    losses = crit(out, tg)
    o2m = {k: v for k, v in losses.items() if "_o2m_" in k}
    assert o2m and all(torch.isfinite(v) for v in o2m.values())
    sum(losses.values()).backward()
    assert m.decoder.dec_score_head[0].weight.grad is not None


def test_aqd_and_angle_loss_train_step():
    torch.manual_seed(0)
    m = VRDet("s", num_classes=3, img_size=256, num_denoising=20).train()
    x = torch.rand(2, 3, 256, 256)
    tg = [{"labels": torch.tensor([1, 2, 0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.3], [0.2, 0.3, 0.1, 0.1, 0.0],
                                                                     [0.7, 0.7, 0.15, 0.05, -0.6]])},
          {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.6, 0.4, 0.3, 0.2, -0.5]])}]
    out = m(x, tg)
    crit = build_criterion(num_classes=3, aqd=True, angle_weight=0.5)
    ind = crit.aqd_indices(out["dn_outputs"][-1], out["dn_meta"], tg)
    full = crit.get_cdn_matched_indices(out["dn_meta"], tg)
    for (q, g), (qf, gf) in zip(ind, full):                       # AQD keeps a subset of the fixed dn pairs
        assert set(zip(q.tolist(), g.tolist())) <= set(zip(qf.tolist(), gf.tolist()))
    losses = crit(out, tg)
    assert "loss_angle" in losses and all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
