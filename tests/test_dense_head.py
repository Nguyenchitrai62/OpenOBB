import math

import torch

from vrdet.models.dense_head import DenseCriterion, DenseRotatedHead, decode, dense_predict, rotated_tal

torch.set_num_threads(1)     # torch 2.12 CPU kernels race with many threads on this machine


def _gt():
    return torch.tensor([[0.30, 0.30, 0.20, 0.06, 0.4],
                         [0.70, 0.60, 0.05, 0.05, 0.0],
                         [0.50, 0.85, 0.008, 0.004, 1.2]])   # tiny: needs the min-side expansion


def test_decode_identity():
    pts = torch.tensor([[0.5, 0.5]])
    st = torch.tensor([8 / 1024])
    reg = torch.zeros(1, 1, 5)
    b = decode(reg, pts, st)[0, 0]
    assert torch.allclose(b, torch.tensor([0.5, 0.5, 16 / 1024, 16 / 1024, 0.0]), atol=1e-6)


def test_tal_assigns_every_object():
    head = DenseRotatedHead(64, 15, img_size=1024, width=32)
    feats = [torch.randn(1, 64, 128, 128), torch.randn(1, 64, 64, 64), torch.randn(1, 64, 32, 32)]
    out = head(feats)
    gt, lab = _gt(), torch.tensor([0, 9, 4])
    fg, gi, ts = rotated_tal(out["dense_logits"][0].sigmoid(), out["dense_boxes"][0], out["dense_points"], lab, gt)
    assert fg.sum() > 0
    assert set(gi[fg].tolist()) == {0, 1, 2}
    # positives of object 0 lie inside its rotated box
    p = out["dense_points"][fg & (gi == 0)]
    c, s = math.cos(0.4), math.sin(0.4)
    u = (p[:, 0] - 0.3) * c + (p[:, 1] - 0.3) * s
    v = -(p[:, 0] - 0.3) * s + (p[:, 1] - 0.3) * c
    assert (u.abs() <= 0.1 + 1e-6).all() and (v.abs() <= 0.03 + 1e-6).all()
    assert ts[fg].argmax(1).tolist() == lab[gi[fg]].tolist()


def test_dense_loss_backward_and_predict():
    head = DenseRotatedHead(64, 15, img_size=1024, width=32)
    feats = [torch.randn(2, 64, 64, 64, requires_grad=True), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
    head.img_size = 512
    out = head(feats)
    targets = [{"labels": torch.tensor([0, 9, 4]), "boxes": _gt()}, {"labels": torch.zeros(0, dtype=torch.long),
                                                                    "boxes": torch.zeros(0, 5)}]
    losses = DenseCriterion()(out, targets)
    sum(losses.values()).backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    res = dense_predict(out, num_top=100, score_thr=0.0, img_size=512)
    assert len(res) == 2 and res[0][2].shape[1] == 5 and len(res[0][0]) <= 100
