import math

import numpy as np
import torch

from vrdet.ops.obb import obb2poly as obb2poly_np, poly_iou
from vrdet.ops.obb_torch import norm_le90, obb2poly, probiou, probiou_matrix, rotated_iou, rotated_iou_matrix

g = torch.Generator().manual_seed(0)


def rand_obb(n, spread=60.0):
    cx = torch.rand(n, generator=g) * spread
    cy = torch.rand(n, generator=g) * spread
    w = 5 + torch.rand(n, generator=g) * 60
    h = 2 + torch.rand(n, generator=g) * 40
    t = (torch.rand(n, generator=g) - 0.5) * 2 * math.pi
    return torch.stack([cx, cy, w, h, t], 1).double()


def test_poly_matches_numpy():
    o = rand_obb(50)
    assert np.allclose(obb2poly(o).reshape(-1, 8).numpy(), obb2poly_np(o.numpy()))


def test_rotated_iou_vs_shapely():
    a, b = rand_obb(3000), rand_obb(3000)
    b[:1000, :2] = a[:1000, :2] + torch.randn(1000, 2, generator=g).double() * 3   # many real overlaps
    b[1000:1100] = a[1000:1100]                                                    # identical
    b[1100:1200] = a[1100:1200] * torch.tensor([1, 1, 0.5, 0.5, 1]).double()      # contained
    ours = rotated_iou(a, b).numpy()
    ref = np.array([poly_iou(obb2poly_np(x.numpy()), obb2poly_np(y.numpy()))[0, 0] for x, y in zip(a, b)])
    assert np.abs(ours - ref).max() < 1e-6
    assert (ours[1000:1100] > 1 - 1e-9).all()
    assert np.allclose(ours[1100:1200], 0.25, atol=1e-9)


def test_rotated_iou_scale_invariant_and_matrix():
    a, b = rand_obb(200), rand_obb(300)
    m = rotated_iou_matrix(a, b, chunk=7000)
    m2 = rotated_iou_matrix(a * torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1]).double(),
                            b * torch.tensor([1e-3, 1e-3, 1e-3, 1e-3, 1]).double())
    ref = poly_iou(obb2poly_np(a.numpy()), obb2poly_np(b.numpy()))
    assert np.abs(m.numpy() - ref).max() < 1e-6
    assert np.abs(m2.numpy() - ref).max() < 1e-6


def test_probiou_properties():
    a = rand_obb(500)
    assert torch.allclose(probiou(a, a), torch.ones(500, dtype=a.dtype), atol=1e-3)
    sw = a.clone()
    sw[:, 2], sw[:, 3], sw[:, 4] = a[:, 3], a[:, 2], a[:, 4] + math.pi / 2       # same rectangle
    assert torch.allclose(probiou(a, sw), torch.ones(500, dtype=a.dtype), atol=1e-3)
    per = a.clone(); per[:, 4] += math.pi
    assert torch.allclose(probiou(a, per), torch.ones(500, dtype=a.dtype), atol=1e-3)
    far = a.clone(); far[:, 0] += 1e4
    assert (probiou(a, far) < 1e-3).all()
    m = probiou_matrix(a[:20], a[:30])
    assert m.shape == (20, 30) and torch.allclose(m.diagonal(), torch.ones(20, dtype=a.dtype), atol=1e-3)


def test_probiou_grad_finite():
    a = rand_obb(64).float().requires_grad_(True)
    b = rand_obb(64).float()
    (1 - probiou(a, b)).sum().backward()
    assert torch.isfinite(a.grad).all()


def test_norm_le90():
    a = rand_obb(500)
    n = norm_le90(a)
    assert (n[:, 2] >= n[:, 3]).all()
    assert ((n[:, 4] >= -math.pi / 2) & (n[:, 4] < math.pi / 2)).all()
    assert torch.allclose(rotated_iou(a, n), torch.ones(500, dtype=a.dtype), atol=1e-9)
