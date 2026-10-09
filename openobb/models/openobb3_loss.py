"""OpenOBB3 assignment + losses (OpenOBB's own implementation).

Assignment: rotated task-aligned (openobb.models.openobb2_loss.assign) with a length-adaptive top-k and a per-level
across band of 1 stride (every level has candidates along every object). Points not selected are negatives
(the head is the output head; duplicates are removed by NMS).

Regression targets are expressed in the frame closest to the predicted angle, so nothing jumps at the le90 wrap
(vertical walls) or when the head points the other way along a segment:
  * the axis is the object's long side, flipped to agree with the predicted direction (left / right swap);
  * near-square objects (aspect weight omega > 0.5) may use the short side as axis when the prediction is closer
    to it (their angle is ambiguous by pi/2).
Losses per positive, weighted by the soft target: quality focal (cls), 1 - ProbIoU (box), distribution focal loss on
the two along-axis distances (bin = stride) and the thickness (bin = stride / 2), across-axis offset error relative
to the thickness (thin objects), and 1 - cos(2 dtheta) (angle).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from openobb.ops.obb_torch import probiou

from .openobb2_loss import assign
from .openobb3 import ALONG_BINS, THICK_BINS, decode


def _dfl(logits, t):
    """Distribution focal loss: t (N,) continuous bin index in [0, bins - 1)."""
    lo = t.floor().long()
    wr = t - lo
    logp = F.log_softmax(logits, -1)
    return -(logp.gather(1, lo[:, None])[:, 0] * (1 - wr) + logp.gather(1, (lo + 1)[:, None])[:, 0] * wr)


def frame_targets(pts, theta_p, g, lam=3.0):
    """pts (N, 2), predicted angle (N,), GT boxes (N, 5) px -> left, right, across, length, thickness, theta targets."""
    dg = torch.stack([g[:, 4].cos(), g[:, 4].sin()], -1)
    ng = torch.stack([-dg[:, 1], dg[:, 0]], -1)
    dp = torch.stack([theta_p.cos(), theta_p.sin()], -1)
    omega = torch.exp(-torch.log(g[:, 2].clamp(min=1e-3) / g[:, 3].clamp(min=1e-3)).pow(2) / lam ** 2)
    alt = (omega > 0.5) & ((dp * ng).sum(-1).abs() > (dp * dg).sum(-1).abs())
    ax = torch.where(alt[:, None], ng, dg)
    length = torch.where(alt, g[:, 3], g[:, 2])
    thick = torch.where(alt, g[:, 2], g[:, 3])
    sgn = torch.where((ax * dp).sum(-1) < 0, -1.0, 1.0)
    ax = ax * sgn[:, None]
    nx = torch.stack([-ax[:, 1], ax[:, 0]], -1)
    rel = pts - g[:, :2]
    u, v = (rel * ax).sum(-1), (rel * nx).sum(-1)
    return length / 2 + u, length / 2 - u, -v, length, thick, torch.atan2(ax[:, 1], ax[:, 0])


class OpenOBB3Loss(nn.Module):
    def __init__(self, w_cls=1.0, w_box=2.0, w_dfl=0.5, w_across=1.0, w_angle=0.5, topk=10, topk_len=16.0,
                 topk_max=48, alpha=1.0, beta=6.0, min_side=8.0, across=1.0, img_size=1024):
        super().__init__()
        self.w = dict(cls=w_cls, box=w_box, dfl=w_dfl, across=w_across, angle=w_angle)
        self.topk, self.topk_len, self.topk_max = topk, topk_len, topk_max
        self.alpha, self.beta, self.min_side, self.across, self.img_size = alpha, beta, min_side, across, img_size

    def forward(self, outputs, targets):
        cls, reg, pts, st = outputs["cls"], outputs["reg"], outputs["points"], outputs["strides"]
        boxes = decode(reg, pts, st)
        scores = cls.detach().sigmoid()
        tsc, fgs, tgts = [], [], []
        for b, t in enumerate(targets):
            gb = t["boxes"].float().clone()
            gb[:, :4] *= self.img_size
            gl = t["labels"]
            k = (self.topk + gb[:, 2] / self.topk_len).round().clamp(max=self.topk_max).long()
            fg, gi, ts, _ = assign(scores[b], boxes[b].detach(), pts, gl, gb, k, self.alpha, self.beta,
                                   self.min_side, strides=st, across=self.across)
            tsc.append(ts)
            fgs.append(fg)
            tgts.append(gb[gi] if len(gb) else boxes.new_zeros((len(fg), 5)))
        tsc, fg, tgt = torch.stack(tsc), torch.stack(fgs), torch.stack(tgts)
        norm = tsc.sum().clamp(min=1.0)
        mod = (cls.detach().sigmoid() - tsc).abs().pow(2)                              # quality focal loss
        out = {"cls": (F.binary_cross_entropy_with_logits(cls, tsc, reduction="none") * mod).sum() / norm}
        if not fg.any():
            z = boxes.sum() * 0
            out.update(box=z, dfl=z, across=z, angle=z)
            return {f"loss_{k}": v * self.w[k] for k, v in out.items()}
        w = tsc.sum(-1)[fg]
        r = reg[fg]
        s = st[None].expand(fg.shape)[fg]
        p_pts = pts[None].expand(fg.shape[0], -1, -1)[fg]
        lt, rt, at, length, thick, th_t = frame_targets(p_pts, r[:, 0].detach(), tgt[fg])
        tbox = torch.stack([tgt[fg][:, 0], tgt[fg][:, 1], length, thick, th_t], -1)
        out["box"] = ((1 - probiou(boxes[fg], tbox)) * w).sum() / norm
        hi_a, hi_t = ALONG_BINS - 1 - 1e-3, THICK_BINS - 1 - 1e-3
        dfl = (_dfl(r[:, 2:2 + ALONG_BINS], (lt / s).clamp(0, hi_a))
               + _dfl(r[:, 2 + ALONG_BINS:2 + 2 * ALONG_BINS], (rt / s).clamp(0, hi_a))
               + _dfl(r[:, 2 + 2 * ALONG_BINS:], (thick / (s / 2)).clamp(0, hi_t))) / 3
        out["dfl"] = (dfl * w).sum() / norm
        out["across"] = (torch.log1p((r[:, 1] * s - at).abs() / thick.clamp(min=2.0)) * w).sum() / norm
        out["angle"] = ((1 - torch.cos(2 * (r[:, 0] - th_t))) * w).sum() / norm
        return {f"loss_{k}": v * self.w[k] for k, v in out.items()}


class DenseOrientedCriterion(nn.Module):
    """OpenOBB3Loss on the dense branch of a DETR model (OpenOBB4); keys prefixed loss_dense_*."""

    def __init__(self, img_size=1024):
        super().__init__()
        self.loss = OpenOBB3Loss(img_size=img_size)

    def forward(self, out, targets):
        o = {"cls": out["dense_logits"], "reg": out["dense_reg"], "points": out["dense_pts_px"],
             "strides": out["dense_st_px"]}
        return {k.replace("loss_", "loss_dense_", 1): v for k, v in self.loss(o, targets).items()}
