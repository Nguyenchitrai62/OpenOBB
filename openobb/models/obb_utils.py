"""Oriented-box helpers for the rotated Fine-grained Distribution Refinement (rotated FDR).

`weighting_function` and `translate_gt` follow D-FINE (Apache-2.0, Copyright (c) 2024 The D-FINE
Authors); everything oriented-box specific is VRDet code.

Conventions
  * boxes are (cx, cy, w, h, theta) with cx, cy, w, h normalised by the square input size;
  * a box has 4 equivalent (w, h, theta) representations: (w, h, t + k*pi) and (h, w, t + pi/2 + k*pi).
    `align_to(gt, ref_theta)` picks the one whose angle is closest to `ref_theta`, so the angle
    residual is always in [-pi/4, pi/4]: no boundary discontinuity anywhere;
  * rotated FDR edges (l, t, r, b) are measured from the reference centre along the box axes,
    in units of ref_w / reg_scale (l, r) and ref_h / reg_scale (t, b), exactly like D-FINE's
    axis-aligned edges; the angle residual has its own uniform bin grid over [-pi/4, pi/4].
"""
import math

import torch
import torch.nn.functional as F

ANGLE_RANGE = math.pi / 4


def weighting_function(reg_max, up, reg_scale):
    """D-FINE's non-uniform W(n): reg_max + 1 values, symmetric, dense near 0."""
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def angle_project(reg_max, device=None, dtype=torch.float32):
    return torch.linspace(-ANGLE_RANGE, ANGLE_RANGE, reg_max + 1, device=device, dtype=dtype)


def translate_gt(gt, function_values, reg_max):
    """Continuous targets -> (left bin index, weight_right, weight_left) for the distribution loss (D-FINE)."""
    gt = gt.reshape(-1)
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1
    indices = closest_left_indices.float()
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0
    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1
    return indices, weight_right, weight_left


# ------------------------------------------------------------------ representation

def align_to(box, ref_theta):
    """Equivalent representation of `box` (..., 5) whose angle is closest to ref_theta (...)."""
    cx, cy, w, h, t = box.unbind(-1)
    k = torch.round((ref_theta - t) / (math.pi / 2))          # number of quarter turns
    odd = torch.remainder(k, 2) == 1
    w2 = torch.where(odd, h, w)
    h2 = torch.where(odd, w, h)
    return torch.stack([cx, cy, w2, h2, t + k * (math.pi / 2)], -1)


def le90(box):
    cx, cy, w, h, t = box.unbind(-1)
    swap = w < h
    w2, h2 = torch.where(swap, h, w), torch.where(swap, w, h)
    t2 = torch.where(swap, t + math.pi / 2, t)
    t2 = torch.remainder(t2 + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([cx, cy, w2, h2, t2], -1)


def pos_features(box, with_angle=False):
    """Features for the query positional MLP: le90 (cx, cy, w, h) [+ (cos2t, sin2t)].

    Default leaves the angle out of the positional query: RiO-DETR (arXiv 2603.09411) measured
    +0.7 AP50 on DIOR-R for an angle-free positional query (angle inferred by the content query)."""
    b = le90(box)
    if not with_angle:
        return b[..., :4]
    return torch.cat([b[..., :4], torch.cos(2 * b[..., 4:5]), torch.sin(2 * b[..., 4:5])], -1)


# ------------------------------------------------------------------ rotated FDR

def distance2obb(ref, dist, theta, reg_scale):
    """ref (..., 5) initial reference, dist (..., 4) = (l, t, r, b) in W(n) units, theta (...) final angle."""
    reg_scale = abs(reg_scale)
    L = (0.5 * reg_scale + dist[..., 0]) * (ref[..., 2] / reg_scale)
    T = (0.5 * reg_scale + dist[..., 1]) * (ref[..., 3] / reg_scale)
    R = (0.5 * reg_scale + dist[..., 2]) * (ref[..., 2] / reg_scale)
    B = (0.5 * reg_scale + dist[..., 3]) * (ref[..., 3] / reg_scale)
    c, s = torch.cos(theta), torch.sin(theta)
    du, dv = (R - L) / 2, (B - T) / 2
    cx = ref[..., 0] + du * c - dv * s
    cy = ref[..., 1] + du * s + dv * c
    return torch.stack([cx, cy, (L + R).clamp(min=1e-6), (T + B).clamp(min=1e-6), theta], -1)


def obb2distance(ref, gt, reg_max, reg_scale, up, eps=0.1):
    """Targets for the 4 edge distributions + the angle distribution.

    gt must already be aligned to ref's angle (align_to(gt, ref[..., 4]))."""
    reg_scale = abs(reg_scale)
    c, s = torch.cos(gt[:, 4]), torch.sin(gt[:, 4])
    dx, dy = gt[:, 0] - ref[:, 0], gt[:, 1] - ref[:, 1]
    du, dv = dx * c + dy * s, -dx * s + dy * c                    # centre offset in the gt frame
    L, R = gt[:, 2] / 2 - du, gt[:, 2] / 2 + du
    T, B = gt[:, 3] / 2 - dv, gt[:, 3] / 2 + dv
    uw = ref[:, 2] / reg_scale + 1e-16
    uh = ref[:, 3] / reg_scale + 1e-16
    four = torch.stack([L / uw, T / uh, R / uw, B / uh], -1) - 0.5 * reg_scale
    ang = (gt[:, 4] - ref[:, 4]).clamp(-ANGLE_RANGE, ANGLE_RANGE)
    w_edges = weighting_function(reg_max, up, reg_scale)
    w_angle = angle_project(reg_max, device=gt.device, dtype=w_edges.dtype)
    i4, wr4, wl4 = translate_gt(four, w_edges, reg_max)
    ia, wra, wla = translate_gt(ang, w_angle, reg_max)
    idx = torch.cat([i4.reshape(-1, 4), ia.reshape(-1, 1)], 1).clamp(min=0, max=reg_max - eps)
    wr = torch.cat([wr4.reshape(-1, 4), wra.reshape(-1, 1)], 1)
    wl = torch.cat([wl4.reshape(-1, 4), wla.reshape(-1, 1)], 1)
    return idx.reshape(-1).detach(), wr.reshape(-1).detach(), wl.reshape(-1).detach()


# ------------------------------------------------------------------ Gaussian / corner costs

def gaussian(box, eps=1e-6):
    """(..., 5) -> mean (..., 2), cov entries (a, b, c) of R diag(w^2/4, h^2/4) R^T."""
    w = box[..., 2].clamp(min=eps)
    h = box[..., 3].clamp(min=eps)
    co, si = torch.cos(box[..., 4]), torch.sin(box[..., 4])
    w2, h2 = w * w / 4, h * h / 4
    a = w2 * co * co + h2 * si * si
    b = w2 * si * si + h2 * co * co
    c = (w2 - h2) * co * si
    return box[..., :2], a, b, c


def kld(pred, target, tau=1.0, eps=1e-7):
    """KLD loss of Yang et al. (2021), D = sqrt(KL(N_t || N_p)), loss = 1 - 1 / (tau + log1p(D)). Broadcasts."""
    mp, ap, bp, cp = gaussian(pred)
    mt, at, bt, ct = gaussian(target)
    detp = (ap * bp - cp * cp).clamp(min=eps * eps)
    dett = (at * bt - ct * ct).clamp(min=eps * eps)
    # inverse of Sigma_p = [[bp, -cp], [-cp, ap]] / detp
    dx = mt[..., 0] - mp[..., 0]
    dy = mt[..., 1] - mp[..., 1]
    maha = (bp * dx * dx - 2 * cp * dx * dy + ap * dy * dy) / detp
    trace = (bp * at - 2 * cp * ct + ap * bt) / detp
    kl = 0.5 * (maha + trace - 2 + torch.log(detp / dett))
    d = kl.clamp(min=eps).sqrt()
    return 1 - 1 / (tau + torch.log1p(d))


def corners(box):
    cx, cy, w, h, t = box.unbind(-1)
    c, s = torch.cos(t), torch.sin(t)
    wx, wy = w / 2 * c, w / 2 * s
    hx, hy = -h / 2 * s, h / 2 * c
    return torch.stack([torch.stack([cx - wx - hx, cy - wy - hy], -1), torch.stack([cx + wx - hx, cy + wy - hy], -1),
                        torch.stack([cx + wx + hx, cy + wy + hy], -1), torch.stack([cx - wx + hx, cy - wy + hy], -1)], -2)


def chamfer_matrix(pred, gt):
    """(N, 5) x (M, 5) -> (N, M) symmetric corner Chamfer distance (mean nearest-corner L2, both ways)."""
    p = corners(pred)[:, None, :, None, :]       # N 1 4 1 2
    g = corners(gt)[None, :, None, :, :]         # 1 M 1 4 2
    d = (p - g).norm(dim=-1)                     # N M 4 4
    return d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def unact_to_box(unact):
    """Decoder references are stored as (logit cx, logit cy, logit w, logit h, theta)."""
    return torch.cat([unact[..., :4].sigmoid(), unact[..., 4:]], -1)


def box_to_unact(box):
    return torch.cat([inverse_sigmoid(box[..., :4]), box[..., 4:]], -1)
