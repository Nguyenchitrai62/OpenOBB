"""OpenOBB10 losses: OpenOBB8's (dense head + relational refiner) with a slot-aware, reach-aware assignment and
square-root class balancing of the dense positives. docs/OPENOBB10.md, research/LEDGER.md F45.

  * slots    each point predicts S objects. The task-aligned candidates are selected per object as in OpenOBB3 (the
             metric of a point is its best slot's score^alpha * IoU^beta); a point keeps its S best objects (by IoU)
             and matches them to its slots by the minimum-cost permutation (EMD, CrowdDet). Target scores are
             normalised per object and slot group (slot 0 / extra slots), so the extra slots bootstrap from their own
             predictions exactly as OpenOBB3's head does from zero-initialised boxes.
  * reach    a point is a candidate only if it can represent the whole object: both along-axis distances within
             (ALONG_BINS - 1) strides and the thickness within (THICK_BINS - 1) half-strides. Without it the DFL
             targets are clamped (truncated boxes): at stride 8 that is 78% of the note candidates (39% of notes have
             no exact P3 point), 7% of the wall candidates; a truncated box is a duplicate that NMS keeps. An object
             with no reachable candidate keeps the unfiltered ones.
  * balance  positives of class c weigh w_c = clamp(sqrt(n_max / n_c), 1, 4) in every dense loss term, n_c = running
             count of positive points of class c. Image-level RFS (t = 0.1) gives repeat factor 1 to every Wall_Color
             class (the rarest, slide_door, is in 22% of the images) while the instance imbalance is 54:1
             (junction 6986 / slide_door 130). The most frequent class keeps weight 1; negatives are unchanged.
With slots = 1 and reach / balance off the dense loss is OpenOBB3's exactly (tests/test_openobb10.py).
"""
import itertools

import torch
import torch.nn.functional as F

from openobb.ops.obb_torch import probiou

from .openobb3 import ALONG_BINS, THICK_BINS, decode
from .openobb3_loss import _dfl, frame_targets
from .openobb8_loss import OpenOBB8Loss
from .openobb10 import parse_off10

BAL_MAX = 4.0
BAL_MOM = 0.99


def assign_slots(scores, boxes, pts, strides, gl, gb, k_per_gt, slots=1, alpha=1.0, beta=6.0, min_side=8.0,
                 across=0.0, reach=False, chunk=64):
    """scores (S*A, C) sigmoid, boxes (S*A, 5) px, pts (S*A, 2), strides (S*A,): slot-major (anchors repeated per
    slot); gl (M,), gb (M, 5) px, k_per_gt (M,). -> fg (S*A,) bool, gi (S*A,) long, target scores (S*A, C).
    slots = 1 and reach = False reproduce openobb2_loss.assign (OpenOBB3's) exactly."""
    SA, C = scores.shape
    S = int(slots)
    A = SA // S
    dev = scores.device
    tsc = torch.zeros_like(scores)
    fg = torch.zeros(SA, dtype=torch.bool, device=dev)
    gi = torch.zeros(SA, dtype=torch.long, device=dev)
    M = gb.shape[0]
    if M == 0:
        return fg, gi, tsc
    pts, st = pts[:A], strides[:A]
    sc_s, bx_s = scores.view(S, A, C), boxes.view(S, A, 5)
    P_g, P_a, P_key, P_m, P_ov = [], [], [], [], []
    for j0 in range(0, M, chunk):
        g, lab, kk = gb[j0:j0 + chunk], gl[j0:j0 + chunk], k_per_gt[j0:j0 + chunk]
        c, s = g[:, 4].cos()[:, None], g[:, 4].sin()[:, None]
        dx, dy = pts[None, :, 0] - g[:, 0:1], pts[None, :, 1] - g[:, 1:2]
        u, v = dx * c + dy * s, -dx * s + dy * c
        hw = g[:, 2:3].clamp(min=min_side) / 2
        hh = g[:, 3:4].clamp(min=min_side) / 2
        if across:
            hh = torch.maximum(hh, st[None, :] * (across / 2))
        inside = (u.abs() <= hw) & (v.abs() <= hh)
        if reach:
            ok = ((g[:, 2:3] / 2 + u.abs() <= (ALONG_BINS - 1) * st[None, :])
                  & (g[:, 3:4] <= (THICK_BINS - 1) * st[None, :] / 2))
            ins_r = inside & ok
            inside = torch.where(ins_r.any(1, keepdim=True), ins_r, inside)
        mi, ai = inside.nonzero(as_tuple=True)
        ov = torch.zeros((S,) + inside.shape, device=dev)
        if len(mi):
            for k in range(S):
                ov[k, mi, ai] = probiou(bx_s[k][ai].float(), g[mi].float()).clamp(0, 1)
        sc = sc_s[:, :, lab].permute(0, 2, 1).float()                                   # (S, m, A)
        centre = torch.exp(-(u.pow(2) / hw.pow(2) + v.pow(2) / hh.pow(2))) * 1e-6      # tie-break while random
        met_s = sc.pow(alpha) * ov.pow(beta)
        met_best, s_best = met_s.max(0)
        ov_best = ov.gather(0, s_best[None])[0]
        metric = (met_best + centre) * inside
        kmax = int(kk.max())
        top = metric.topk(min(kmax, A), dim=1).indices
        okk = torch.arange(top.shape[1], device=dev)[None, :] < kk[:, None]
        sel = torch.zeros_like(inside)
        sel.scatter_(1, top, okk)
        sel &= inside
        pg, pa = sel.nonzero(as_tuple=True)
        P_g.append(pg + j0)
        P_a.append(pa)
        P_key.append((ov_best + centre)[pg, pa])
        P_m.append(((met_s + centre[None]) * inside[None])[:, pg, pa].T)                 # (p, S)
        P_ov.append((ov + centre[None])[:, pg, pa].T)
    pg, pa, key = torch.cat(P_g), torch.cat(P_a), torch.cat(P_key)
    if not len(pg):
        return fg, gi, tsc
    pm, pov = torch.cat(P_m), torch.cat(P_ov)
    # each point keeps its S best objects (IoU of the best slot; ties: lower object index, as OpenOBB3)
    o1 = torch.argsort(key, descending=True, stable=True)
    order = o1[torch.argsort(pa[o1], stable=True)]
    pg, pa, pm, pov = pg[order], pa[order], pm[order], pov[order]
    n = len(pa)
    ar = torch.arange(n, device=dev)
    new = torch.ones(n, dtype=torch.bool, device=dev)
    new[1:] = pa[1:] != pa[:-1]
    rank = ar - torch.cummax(torch.where(new, ar, torch.zeros_like(ar)), 0).values
    keep = rank < S
    pg, pa, pm, pov, rank, new = pg[keep], pa[keep], pm[keep], pov[keep], rank[keep], new[keep]
    if S == 1:
        slot = torch.zeros_like(pa)
    else:
        # minimum-cost permutation of the point's (<= S) objects onto its S slots; identity first on ties
        grp = torch.cumsum(new.long(), 0) - 1                                            # point group per pair
        U = int(grp[-1]) + 1
        cost = torch.zeros((U, S, S), device=dev)                                        # (point, rank, slot)
        cost[grp, rank] = -pm
        perms = torch.tensor(list(itertools.permutations(range(S))), device=dev)        # (P, S): rank -> slot
        tot = cost[:, torch.arange(S, device=dev)[None, :], perms].sum(-1)               # (U, P)
        best = tot.argmin(1)
        slot = perms[best[grp], rank]
    m_val = pm.gather(1, slot[:, None])[:, 0]
    ov_val = pov.gather(1, slot[:, None])[:, 0].clamp(min=0)
    flat = slot * A + pa
    fg[flat] = True
    gi[flat] = pg
    grp_t = pg * 2 + (slot > 0).long()                                                   # object x slot group
    gmax_m = torch.zeros(2 * M, device=dev).scatter_reduce(0, grp_t, m_val, "amax", include_self=True)
    gmax_ov = torch.zeros(2 * M, device=dev).scatter_reduce(0, grp_t, ov_val, "amax", include_self=True)
    norm = m_val * gmax_ov[grp_t] / (gmax_m[grp_t] + 1e-9)
    tsc[flat, gl[pg]] = norm.to(tsc.dtype)
    return fg, gi, tsc


class OpenOBB10Loss(OpenOBB8Loss):
    def __init__(self, off=None, **kw):
        super().__init__(**kw)
        off = parse_off10(off)
        self.reach, self.balance = "reach" not in off, "balance" not in off
        self.pos_count = None

    @torch.no_grad()
    def class_weights(self, tsc):
        """Running count of positive points per class -> w_c = clamp(sqrt(n_max / n_c), 1, BAL_MAX)."""
        cnt = (tsc > 0).flatten(0, -2).sum(0).float()
        self.pos_count = cnt if self.pos_count is None else BAL_MOM * self.pos_count.to(cnt) + (1 - BAL_MOM) * cnt
        n = self.pos_count
        return (n.max() / n.clamp(min=1e-6)).sqrt().clamp(1.0, BAL_MAX)

    def dense_losses(self, outputs, targets):
        S = int(outputs.get("slots", 1))
        if S == 1 and not self.reach and not self.balance:
            return super().dense_losses(outputs, targets)
        cls, reg, pts, st = outputs["cls"], outputs["reg"], outputs["points"], outputs["strides"]
        boxes = decode(reg, pts, st)
        scores = cls.detach().sigmoid()
        tsc, fgs, tgts = [], [], []
        for b, t in enumerate(targets):
            gb = t["boxes"].float().clone()
            gb[:, :4] *= self.img_size
            gl = t["labels"]
            k = (self.topk + gb[:, 2] / self.topk_len).round().clamp(max=self.topk_max).long()
            fg, gi, ts = assign_slots(scores[b], boxes[b].detach(), pts, st, gl, gb, k, S, self.alpha, self.beta,
                                      self.min_side, self.across, self.reach)
            tsc.append(ts)
            fgs.append(fg)
            tgts.append(gb[gi] if len(gb) else boxes.new_zeros((len(fg), 5)))
        tsc, fg, tgt = torch.stack(tsc), torch.stack(fgs), torch.stack(tgts)
        norm = tsc.sum().clamp(min=1.0)
        mod = (cls.detach().sigmoid() - tsc).abs().pow(2)                              # quality focal loss
        bce = F.binary_cross_entropy_with_logits(cls, tsc, reduction="none") * mod
        wc = None
        if self.balance:
            wc = self.class_weights(tsc)
            bce = bce * torch.where(tsc > 0, wc.to(bce.dtype), torch.ones_like(bce))     # rare positives count more
        out = {"cls": bce.sum() / norm}
        if not fg.any():
            z = boxes.sum() * 0
            out.update(box=z, dfl=z, across=z, angle=z)
            return {f"loss_{k}": v * self.w[k] for k, v in out.items()}
        w = tsc.sum(-1)[fg]
        if wc is not None:
            w = w * wc[tsc[fg].argmax(-1)]
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
