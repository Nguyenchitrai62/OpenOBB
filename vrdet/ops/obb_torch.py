"""Oriented-box geometry in PyTorch (differentiable where it matters).

Box format: (..., 5) = (cx, cy, w, h, theta), theta in radians; any (w, h, theta) convention works
for the Gaussian functions (they are invariant to w/h swap + 90 deg and to theta periodicity).
"""
import math

import torch

EPS = 1e-7


def obb2poly(obb):
    """(..., 5) -> (..., 4, 2) corners, same order as vrdet.ops.obb.obb2poly."""
    cx, cy, w, h, t = obb.unbind(-1)
    c, s = torch.cos(t), torch.sin(t)
    wx, wy = w / 2 * c, w / 2 * s
    hx, hy = -h / 2 * s, h / 2 * c
    pts = torch.stack([
        torch.stack([cx - wx - hx, cy - wy - hy], -1),
        torch.stack([cx + wx - hx, cy + wy - hy], -1),
        torch.stack([cx + wx + hx, cy + wy + hy], -1),
        torch.stack([cx - wx + hx, cy - wy + hy], -1)], -2)
    return pts


def norm_le90(obb):
    """w >= h, theta in [-pi/2, pi/2). Not differentiable through the swap; use on targets."""
    cx, cy, w, h, t = obb.unbind(-1)
    swap = w < h
    w2 = torch.where(swap, h, w)
    h2 = torch.where(swap, w, h)
    t2 = torch.where(swap, t + math.pi / 2, t)
    t2 = torch.remainder(t2 + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([cx, cy, w2, h2, t2], -1)


# ------------------------------------------------------------------ Gaussian (ProbIoU / KLD)

def _gauss(obb):
    """Mean and covariance entries (a, b, c) with cov = [[a, c], [c, b]] of the uniform-box Gaussian."""
    w2 = obb[..., 2].pow(2) / 12
    h2 = obb[..., 3].pow(2) / 12
    c, s = torch.cos(obb[..., 4]), torch.sin(obb[..., 4])
    a = w2 * c * c + h2 * s * s
    b = w2 * s * s + h2 * c * c
    cc = (w2 - h2) * c * s
    return obb[..., 0], obb[..., 1], a, b, cc


def probiou(obb1, obb2, eps=EPS):
    """ProbIoU (1 - Hellinger distance between Gaussians), elementwise over broadcast shapes."""
    x1, y1, a1, b1, c1 = _gauss(obb1)
    x2, y2, a2, b2, c2 = _gauss(obb2)
    a, b, c = (a1 + a2), (b1 + b2), (c1 + c2)
    den = a * b - c * c
    t1 = (a * (y1 - y2).pow(2) + b * (x1 - x2).pow(2)) / (den + eps) * 0.25
    t2 = (c * (x2 - x1) * (y1 - y2)) / (den + eps) * 0.5
    det1 = (a1 * b1 - c1 * c1).clamp(min=0)
    det2 = (a2 * b2 - c2 * c2).clamp(min=0)
    t3 = torch.log(den / (4 * torch.sqrt(det1 * det2) + eps) + eps) * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd


def probiou_matrix(obb1, obb2):
    """(N, 5) x (M, 5) -> (N, M)."""
    return probiou(obb1[:, None, :], obb2[None, :, :])


# ------------------------------------------------------------------ exact rotated IoU (no grad needed)

def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _inside(pts, poly):
    """pts (..., P, 2) inside convex poly (..., 4, 2) (either orientation)."""
    e0 = poly
    e1 = poly.roll(-1, dims=-2)
    cr = _cross(e0[..., None, :, :], e1[..., None, :, :], pts[..., :, None, :])   # (..., P, 4)
    return (cr >= -1e-6).all(-1) | (cr <= 1e-6).all(-1)


def poly_area(p):
    x, y = p[..., 0], p[..., 1]
    return 0.5 * (x * y.roll(-1, -1) - x.roll(-1, -1) * y).sum(-1).abs()


def rotated_iou(obb1, obb2):
    """Exact IoU of matched pairs: obb1, obb2 (N, 5) -> (N,). Computed in float64, no gradient."""
    with torch.no_grad():
        o1, o2 = obb1.double().clone(), obb2.double().clone()
        n = o1.shape[0]
        if n == 0:
            return obb1.new_zeros(0)
        # similarity-normalise each pair (IoU is invariant) so tolerances are scale free
        ctr0 = (o1[:, :2] + o2[:, :2]) / 2
        scale = torch.stack([o1[:, 2], o1[:, 3], o2[:, 2], o2[:, 3]], 1).abs().amax(1).clamp(min=1e-12)
        for o in (o1, o2):
            o[:, :2] = (o[:, :2] - ctr0) / scale[:, None]
            o[:, 2:4] = o[:, 2:4] / scale[:, None]
        p = obb2poly(o1)
        q = obb2poly(o2)
        # edge-edge intersections (4 x 4)
        a0, a1 = p[:, :, None, :], p.roll(-1, 1)[:, :, None, :]
        b0, b1 = q[:, None, :, :], q.roll(-1, 1)[:, None, :, :]
        da, db = a1 - a0, b1 - b0
        den = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
        diff = b0 - a0
        t = (diff[..., 0] * db[..., 1] - diff[..., 1] * db[..., 0]) / den.where(den.abs() > 1e-12, torch.ones_like(den))
        u = (diff[..., 0] * da[..., 1] - diff[..., 1] * da[..., 0]) / den.where(den.abs() > 1e-12, torch.ones_like(den))
        ok_e = (den.abs() > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        pe = (a0 + t[..., None] * da).reshape(n, 16, 2)
        ok_e = ok_e.reshape(n, 16)
        ok_p = _inside(p, q)
        ok_q = _inside(q, p)
        pts = torch.cat([p, q, pe], 1)                       # (N, 24, 2)
        ok = torch.cat([ok_p, ok_q, ok_e], 1)
        cnt = ok.sum(1)
        ctr = (pts * ok[..., None]).sum(1) / cnt.clamp(min=1)[:, None]
        ang = torch.atan2(pts[..., 1] - ctr[:, None, 1], pts[..., 0] - ctr[:, None, 0])
        ang = torch.where(ok, ang, torch.full_like(ang, 10.0))  # invalid -> sorted last
        order = ang.argsort(1)
        sp = pts.gather(1, order[..., None].expand(-1, -1, 2))
        sok = ok.gather(1, order)
        first = sp[:, :1, :].expand_as(sp)
        sp = torch.where(sok[..., None], sp, first)          # pad with the first valid vertex: zero extra area
        inter = poly_area(sp)
        inter = torch.where(cnt >= 3, inter, torch.zeros_like(inter))
        a1_ = (o1[:, 2] * o1[:, 3]).abs()
        a2_ = (o2[:, 2] * o2[:, 3]).abs()
        iou = inter / (a1_ + a2_ - inter).clamp(min=1e-12)
        return iou.clamp(0, 1).to(obb1.dtype)


def rotated_iou_matrix(obb1, obb2, chunk=4_000_000):
    """(N, 5) x (M, 5) -> (N, M) exact IoU, chunked to bound memory."""
    n, m = obb1.shape[0], obb2.shape[0]
    out = obb1.new_zeros((n, m))
    if n == 0 or m == 0:
        return out
    rows = max(1, chunk // max(m, 1))
    for i in range(0, n, rows):
        a = obb1[i:i + rows, None, :].expand(-1, m, -1).reshape(-1, 5)
        b = obb2[None, :, :].expand(a.shape[0] // m, -1, -1).reshape(-1, 5)
        out[i:i + rows] = rotated_iou(a, b).reshape(-1, m)
    return out
