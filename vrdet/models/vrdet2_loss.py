"""VRDet2 assignment + losses (VRDet's own implementation).

Assignment: rotated task-aligned (TOOD-style metric score^a * IoU^b) with
  * candidates inside the box, the thickness floored to `min_side` px so 1-3 px lines get anchors (YOLO26 STAL idea);
  * a length-adaptive top-k: long objects get more positives along their axis (k = topk + length / topk_len);
  * one-to-many head: points inside an object that are not selected are ignored, not negatives (every point of a
    line looks the same; pushing most of them to background contradicts the few positives);
  * a one-to-one variant (k = 1, same metric times a midpoint prior along the axis) for the NMS-free head
    (YOLOv10 consistent matching idea; the prior makes the single chosen point of a long object stable).
Losses per positive (weighted by the soft target): quality focal (cls), 1 - ProbIoU (shape), endpoint error relative
to the length (along-axis precision of long objects), across-axis centre error relative to the thickness (thin
objects), distribution focal loss on the thickness, and a periodic angle loss (pi for elongated, pi/2 for square).
GPU memory stays bounded: ground truths are processed in chunks, anchors keep a running best assignment.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrdet.models.vrdet2 import REG_BINS, decode
from vrdet.ops.obb_torch import probiou


@torch.no_grad()
def assign(scores, boxes, pts, gl, gb, k_per_gt, alpha=1.0, beta=6.0, min_side=8.0, chunk=64, mid_prior=0.0):
    """scores (A, C) sigmoid, boxes (A, 5) px, pts (A, 2) px; gl (M,), gb (M, 5) px, k_per_gt (M,) int.
    -> fg (A,) bool, gi (A,) long, target_scores (A, C), inside_any (A,) bool (point inside some object)."""
    A, C = scores.shape
    M = gb.shape[0]
    dev = scores.device
    tsc = torch.zeros_like(scores)
    inside_any = torch.zeros(A, dtype=torch.bool, device=dev)
    if M == 0:
        return torch.zeros(A, dtype=torch.bool, device=dev), torch.zeros(A, dtype=torch.long, device=dev), tsc,             inside_any
    best_ov = torch.full((A,), -1.0, device=dev)
    best_g = torch.zeros(A, dtype=torch.long, device=dev)
    best_m = torch.zeros(A, device=dev)
    for j0 in range(0, M, chunk):
        g, lab, kk = gb[j0:j0 + chunk], gl[j0:j0 + chunk], k_per_gt[j0:j0 + chunk]
        c, s = g[:, 4].cos()[:, None], g[:, 4].sin()[:, None]
        dx, dy = pts[None, :, 0] - g[:, 0:1], pts[None, :, 1] - g[:, 1:2]
        u, v = dx * c + dy * s, -dx * s + dy * c
        hw = g[:, 2:3].clamp(min=min_side) / 2
        hh = g[:, 3:4].clamp(min=min_side) / 2
        inside = (u.abs() <= hw) & (v.abs() <= hh)
        inside_any |= inside.any(0)
        mi, ai = inside.nonzero(as_tuple=True)
        ov = torch.zeros(inside.shape, device=dev)
        if len(mi):
            ov[mi, ai] = probiou(boxes[ai].float(), g[mi].float()).clamp(0, 1)
        sc = scores[:, lab].T.float()
        centre = torch.exp(-(u.pow(2) / hw.pow(2) + v.pow(2) / hh.pow(2))) * 1e-6      # tie-break while random
        prior = torch.exp(-mid_prior * u.pow(2) / hw.pow(2)) if mid_prior else 1.0
        metric = (sc.pow(alpha) * ov.pow(beta) * prior + centre) * inside
        kmax = int(kk.max())
        top = metric.topk(min(kmax, A), dim=1).indices
        ok = torch.arange(top.shape[1], device=dev)[None, :] < kk[:, None]
        sel = torch.zeros_like(inside)
        sel.scatter_(1, top, ok)
        sel &= inside
        cand = torch.where(sel, ov + centre, torch.full_like(ov, -1.0))
        bv, bj = cand.max(0)                                                           # this chunk's best per anchor
        upd = bv > best_ov
        best_ov = torch.where(upd, bv, best_ov)
        best_g = torch.where(upd, bj + j0, best_g)
        best_m = torch.where(upd, metric.gather(0, bj[None])[0], best_m)
    fg = best_ov >= 0
    if fg.any():
        gi = best_g[fg]
        gmax_m = torch.zeros(M, device=dev).scatter_reduce(0, gi, best_m[fg], "amax", include_self=True)
        gmax_ov = torch.zeros(M, device=dev).scatter_reduce(0, gi, best_ov[fg].clamp(min=0), "amax", include_self=True)
        norm = best_m[fg] * gmax_ov[gi] / (gmax_m[gi] + 1e-9)
        tsc[fg.nonzero(as_tuple=True)[0], gl[gi]] = norm.to(tsc.dtype)
    return fg, best_g, tsc, inside_any


def _ends(b):
    a = torch.stack([b[:, 4].cos(), b[:, 4].sin()], -1) * (b[:, 2:3] / 2)
    return b[:, :2] - a, b[:, :2] + a


class VRDet2Loss(nn.Module):
    def __init__(self, w_cls=1.0, w_box=2.0, w_end=1.0, w_across=1.0, w_dfl=0.5, w_angle=0.5, w_o2o=1.0,
                 topk=10, topk_len=16.0, topk_max=48, alpha=1.0, beta=6.0, min_side=8.0, lam=3.0, img_size=1024,
                 mid_prior=4.0):
        super().__init__()
        self.w = dict(cls=w_cls, box=w_box, end=w_end, across=w_across, dfl=w_dfl, angle=w_angle)
        self.w_o2o, self.topk, self.topk_len, self.topk_max = w_o2o, topk, topk_len, topk_max
        self.alpha, self.beta, self.min_side, self.lam, self.img_size = alpha, beta, min_side, lam, img_size
        self.mid_prior = mid_prior

    def head_loss(self, cls, reg, pts, st, targets, one2one):
        boxes = decode(reg, pts, st)                                                   # (B, A, 5) px
        scores = cls.detach().sigmoid()
        tsc, fgs, tgts, igns = [], [], [], []
        for b, t in enumerate(targets):
            gb = t["boxes"].float().clone()
            gb[:, :4] *= self.img_size
            gl = t["labels"]
            if one2one:
                k = torch.ones(len(gb), dtype=torch.long, device=gb.device)
            else:
                k = (self.topk + gb[:, 2] / self.topk_len).round().clamp(max=self.topk_max).long()
            fg, gi, ts, ins = assign(scores[b], boxes[b].detach(), pts, gl, gb, k, self.alpha, self.beta,
                                     self.min_side, mid_prior=self.mid_prior if one2one else 0.0)
            tsc.append(ts)
            fgs.append(fg)
            igns.append(ins & ~fg if not one2one else torch.zeros_like(fg))
            tgts.append(gb[gi] if len(gb) else boxes.new_zeros((len(fg), 5)))
        tsc, fg, tgt, ign = torch.stack(tsc), torch.stack(fgs), torch.stack(tgts), torch.stack(igns)
        norm = tsc.sum().clamp(min=1.0)
        mod = (cls.detach().sigmoid() - tsc).abs().pow(2) * (~ign)[..., None]          # quality focal loss
        out = {"cls": (F.binary_cross_entropy_with_logits(cls, tsc, reduction="none") * mod).sum() / norm}
        if one2one:                       # box branch is shared with the one-to-many head (reg is detached here)
            return out
        if not fg.any():
            z = boxes.sum() * 0
            out.update(box=z, end=z, across=z, dfl=z, angle=z)
            return out
        w = tsc.sum(-1)[fg]
        p, g = boxes[fg], tgt[fg]
        s = st[None].expand(fg.shape)[fg]
        r = reg[fg]
        out["box"] = ((1 - probiou(p, g)) * w).sum() / norm
        # endpoints, either ordering, relative to the true length (along-axis precision for long objects)
        pa, pb = _ends(p)
        ga, gb_ = _ends(g)
        e1 = (pa - ga).abs().sum(-1) + (pb - gb_).abs().sum(-1)
        e2 = (pa - gb_).abs().sum(-1) + (pb - ga).abs().sum(-1)
        out["end"] = (torch.minimum(e1, e2) / g[:, 2].clamp(min=1.0) * w).sum() / norm
        # centre offset across the true axis, relative to the true thickness (thin objects: 1 px matters)
        n = torch.stack([-g[:, 4].sin(), g[:, 4].cos()], -1)
        off = ((p[:, :2] - g[:, :2]) * n).sum(-1).abs()
        out["across"] = (torch.log1p(off / g[:, 3].clamp(min=2.0)) * w).sum() / norm
        # thickness distribution (DFL), bins of stride / 2
        tgt_bin = (g[:, 3] / (s / 2)).clamp(0, REG_BINS - 1 - 1e-3)
        lo = tgt_bin.floor().long()
        wr = tgt_bin - lo
        logp = F.log_softmax(r[:, 4:], -1)
        dfl = -(logp.gather(1, lo[:, None])[:, 0] * (1 - wr) + logp.gather(1, (lo + 1)[:, None])[:, 0] * wr)
        out["dfl"] = (dfl * w).sum() / norm
        # angle: period pi for elongated objects, pi/2 for square ones (ProbIoU is angle-blind there)
        dth = p[:, 4] - g[:, 4]
        omega = torch.exp(-torch.log(g[:, 2].clamp(min=1e-3) / g[:, 3].clamp(min=1e-3)).pow(2) / self.lam ** 2)
        ang = (1 - omega) * (1 - torch.cos(2 * dth)) + omega * torch.sin(2 * dth).pow(2)
        out["angle"] = (ang * w).sum() / norm
        return out

    def forward(self, outputs, targets):
        pts, st = outputs["points"], outputs["strides"]
        losses = {}
        for name, one2one, scale in (("o2m", False, 1.0), ("o2o", True, self.w_o2o)):
            cls, reg = outputs[name]
            for k, v in self.head_loss(cls, reg, pts, st, targets, one2one).items():   # o2o: cls only
                losses[f"loss_{k}" + ("" if name == "o2m" else "_o2o")] = v * self.w[k] * scale
        return losses
