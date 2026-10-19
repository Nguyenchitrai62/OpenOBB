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
from .openobb3_loss import angle_term, dist_loss, frame_targets
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


ASSIGN = "batched"       # "legacy": assign_slots once per image x class group (A/B of the speed fix, F112)
BUDGET = 1 << 25         # elements of one (problems x objects x anchors) block of the batched assignment


def pad_targets(targets, cg, G, img_size, device):
    """Ground truth of every (image, class group) problem padded to the batch's largest count (ONE device sync) ->
    gb (B G, M, 5) px, gl (B G, M), valid (B G, M). Order inside a problem: the image's label order (stable)."""
    B = len(targets)
    n = [len(t["labels"]) for t in targets]                                   # host-side lengths: no sync
    P = B * G
    if sum(n) == 0:
        z = torch.zeros((P, 0), device=device)
        return z.new_zeros((P, 0, 5)), z.long(), z.bool()
    gb = torch.cat([t["boxes"].float().to(device) for t in targets]).clone()
    gb[:, :4] *= img_size
    gl = torch.cat([t["labels"].to(device) for t in targets]).long()
    bi = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(n, device=device))
    key = bi * G + (cg[gl] if G > 1 else torch.zeros_like(gl))
    order = torch.argsort(key, stable=True)
    key, gb, gl = key[order], gb[order], gl[order]
    cnt = torch.bincount(key, minlength=P)
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(len(key), device=device) - start[key]
    M = int(cnt.max())                                                        # the one sync
    pgb = gb.new_zeros((P, M, 5))
    pgl = gl.new_zeros((P, M))
    valid = torch.zeros((P, M), dtype=torch.bool, device=device)
    pgb[key, rank], pgl[key, rank], valid[key, rank] = gb, gl, True
    return pgb, pgl, valid


@torch.no_grad()
def assign_batched(scores, boxes, pts, strides, gl, gb, valid, k_per_gt, k_max, alpha=1.0, beta=6.0, min_side=8.0,
                   across=0.0, budget=None):
    """assign_slots (slots = 1, no reach) for P independent problems at once: scores (P, A, C) sigmoid, boxes
    (P, A, 5) px, pts (A, 2), strides (A,), gl / k_per_gt (P, M), gb (P, M, 5) px, valid (P, M) (padding False),
    k_max >= every k. -> fg (P, A) bool, gi (P, A) long (object index in M), target scores (P, A, C).
    Same assignment as assign_slots, problem by problem (tests/test_assign_batched.py): dense (problems x objects x
    anchors) blocks instead of a Python loop over images x groups x 64 objects with ~6 device syncs each."""
    P, A, C = scores.shape
    M = gb.shape[1]
    dev = scores.device
    tsc = torch.zeros_like(scores)
    fg = torch.zeros((P, A), dtype=torch.bool, device=dev)
    gi = torch.zeros((P, A), dtype=torch.long, device=dev)
    if M == 0:
        return fg, gi, tsc
    cnt = valid.sum(1)
    host = torch.stack([cnt, k_per_gt.masked_fill(~valid, 0).amax(1)], 1).tolist()     # the one sync
    K = min(max(max(k for _, k in host), 1), int(k_max), A)
    # problems by decreasing object count; each block is cut to its largest count (no padding to the batch's max)
    order = [p for p in sorted(range(P), key=lambda p: -host[p][0]) if host[p][0] > 0]
    budget = budget or BUDGET
    ar = torch.arange(K, device=dev)
    j = 0
    while j < len(order):
        Mb = host[order[j]][0]
        n = max(1, min(len(order) - j, int(budget // max(Mb * A, 1))))
        idx = torch.tensor(order[j:j + n], device=dev)
        j += n
        g, lab, kk, ok = gb[idx, :Mb], gl[idx, :Mb], k_per_gt[idx, :Mb], valid[idx, :Mb]
        Mc = Mb
        c, s = g[..., 4].cos()[..., None], g[..., 4].sin()[..., None]                  # (n, M, 1)
        dx, dy = pts[None, None, :, 0] - g[..., 0:1], pts[None, None, :, 1] - g[..., 1:2]
        u, v = dx * c + dy * s, -dx * s + dy * c                                         # (n, M, A)
        hw = g[..., 2:3].clamp(min=min_side) / 2
        hh = g[..., 3:4].clamp(min=min_side) / 2
        if across:
            hh = torch.maximum(hh, strides[None, None, :] * (across / 2))
        inside = (u.abs() <= hw) & (v.abs() <= hh) & ok[..., None]
        pi, mi, ai = inside.nonzero(as_tuple=True)
        ov = torch.zeros(inside.shape, device=dev)
        bx = boxes[idx]
        if len(pi):
            ov[pi, mi, ai] = probiou(bx[pi, ai].float(), g[pi, mi].float()).clamp(0, 1)
        sc = scores[idx].float().transpose(1, 2).gather(1, lab[..., None].expand(-1, -1, A))   # (n, M, A)
        centre = torch.exp(-(u.pow(2) / hw.pow(2) + v.pow(2) / hh.pow(2))) * 1e-6
        del u, v, dx, dy
        met = sc.pow(alpha) * ov.pow(beta)
        metric = (met + centre) * inside
        top = metric.topk(K, dim=2).indices
        sel = torch.zeros_like(inside).scatter_(2, top, (ar[None, None, :] < kk[..., None]).expand(n, Mc, K))
        sel &= inside
        key = torch.where(sel, ov + centre, torch.full_like(ov, -1.0))
        best = key.argmax(1)                                                            # (n, A): first max = lowest m
        f = sel.any(1)
        m_val = torch.where(f, metric.gather(1, best[:, None])[:, 0], torch.zeros_like(best, dtype=metric.dtype))
        ov_val = torch.where(f, (ov + centre).gather(1, best[:, None])[:, 0].clamp(min=0),
                             torch.zeros_like(m_val))
        gmax_m = torch.zeros((n, Mc), device=dev).scatter_reduce(1, best, m_val, "amax", include_self=True)
        gmax_ov = torch.zeros((n, Mc), device=dev).scatter_reduce(1, best, ov_val, "amax", include_self=True)
        norm = m_val * gmax_ov.gather(1, best) / (gmax_m.gather(1, best) + 1e-9)
        ts = torch.zeros((n, A, C), dtype=tsc.dtype, device=dev)
        ts.scatter_(2, lab.gather(1, best)[..., None], torch.where(f, norm, torch.zeros_like(norm))[..., None].to(tsc.dtype))
        tsc[idx], fg[idx], gi[idx] = ts, f, torch.where(f, best, torch.zeros_like(best))
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

    @torch.no_grad()
    def group_assign(self, cls, boxes, pts, st, targets, cg):
        """Per class group g: the group's objects assigned to the group's boxes (boxes[g] (B, A, 5)) only.
        -> class targets (B, A, C) summed over groups (disjoint columns), per group fg (B, A) and targets (B, A, 5)."""
        if ASSIGN != "legacy":
            return self.group_assign_batched(cls, boxes, pts, st, targets, cg)
        G = len(boxes)
        scores = cls.detach().sigmoid()
        tsc = torch.zeros_like(cls, dtype=torch.float32)
        fgs, tgts = [[] for _ in range(G)], [[] for _ in range(G)]
        for b, t in enumerate(targets):
            gb_all = t["boxes"].float().clone().to(cls.device)
            gb_all[:, :4] *= self.img_size
            gl_all = t["labels"].to(cls.device)
            for g in range(G):
                sel = cg[gl_all] == g
                gb, gl = gb_all[sel], gl_all[sel]
                k = (self.topk + gb[:, 2] / self.topk_len).round().clamp(max=self.topk_max).long()
                fg, gi, ts = assign_slots(scores[b], boxes[g][b].detach(), pts, st, gl, gb, k, 1, self.alpha,
                                          self.beta, self.min_side, self.across, self.reach)
                tsc[b] += ts.float()
                fgs[g].append(fg)
                tgts[g].append(gb[gi] if len(gb) else boxes[g].new_zeros((len(fg), 5)))
        return tsc, [torch.stack(f) for f in fgs], [torch.stack(t) for t in tgts]

    @torch.no_grad()
    def group_assign_batched(self, cls, boxes, pts, st, targets, cg):
        """group_assign with every (image, group) problem in one batched assignment (assign_batched)."""
        G = len(boxes)
        B, A, C = cls.shape
        gb, gl, valid = pad_targets(targets, cg, G, self.img_size, cls.device)            # (B G, M, ...)
        k = (self.topk + gb[..., 2] / self.topk_len).round().clamp(max=self.topk_max).long()
        scores = cls.detach().sigmoid()[:, None].expand(B, G, A, C).reshape(B * G, A, C)
        bx = torch.stack([b.detach() for b in boxes], 1).reshape(B * G, A, 5)
        fg, gi, ts = assign_batched(scores, bx, pts, st, gl, gb, valid, k, self.topk_max, self.alpha, self.beta,
                                    self.min_side, self.across)
        tsc = ts.view(B, G, A, C).sum(1)
        tg = gb.gather(1, gi[..., None].expand(-1, -1, 5)).to(bx.dtype) if gb.shape[1] else bx.new_zeros(bx.shape)
        fg, tg = fg.view(B, G, A), tg.view(B, G, A, 5)
        return tsc, [fg[:, g] for g in range(G)], [tg[:, g] for g in range(G)]

    def grouped_dense_losses(self, outputs, targets):
        """Contention-aware class groups (OpenOBB19): reg holds one box per group at every point. Each group is
        assigned on its own (its classes only, its own boxes), so objects of different groups never compete for a
        point; the class targets of the groups (disjoint columns) are summed into one classification loss."""
        cls, reg, pts, st = outputs["cls"], outputs["reg"], outputs["points"], outputs["strides"]
        cg = outputs["class_group"].to(cls.device)
        G = int(cg.max()) + 1
        R = reg.shape[-1] // G
        regs = reg.view(*reg.shape[:-1], G, R)
        boxes = [decode(regs[..., g, :], pts, st) for g in range(G)]
        tsc, fgs, tgts = self.group_assign(cls, boxes, pts, st, targets, cg)
        norm = tsc.sum().clamp(min=1.0)
        mod = (cls.detach().sigmoid() - tsc).abs().pow(2)                              # quality focal loss
        bce = F.binary_cross_entropy_with_logits(cls, tsc.to(cls.dtype), reduction="none") * mod
        wc = None
        if self.balance:
            wc = self.class_weights(tsc)
            bce = bce * torch.where(tsc > 0, wc.to(bce.dtype), torch.ones_like(bce))
        out = {"cls": bce.sum() / norm}
        z = reg.sum() * 0
        acc = {"box": z, "dfl": z, "across": z, "angle": z}
        for g in range(G):
            fg = fgs[g]
            if not fg.any():
                continue
            tgt = tgts[g]
            tg = tsc[..., cg == g]
            w = tg.sum(-1)[fg]
            if wc is not None:
                w = w * wc[cg == g][tg[fg].argmax(-1)]
            r = regs[..., g, :][fg]
            s = st[None].expand(fg.shape)[fg]
            p_pts = pts[None].expand(fg.shape[0], -1, -1)[fg]
            lt, rt, at, length, thick, th_t = frame_targets(p_pts, r[:, 0].detach(), tgt[fg])
            tbox = torch.stack([tgt[fg][:, 0], tgt[fg][:, 1], length, thick, th_t], -1)
            acc["box"] = acc["box"] + ((1 - probiou(boxes[g][fg], tbox)) * w).sum() / norm
            acc["dfl"] = acc["dfl"] + (dist_loss(r, s, lt, rt, thick) * w).sum() / norm
            acc["across"] = acc["across"] + (torch.log1p((r[:, 1] * s - at).abs() / thick.clamp(min=2.0)) * w).sum() / norm
            acc["angle"] = acc["angle"] + angle_term(r[:, 0], th_t, w, norm)
        out.update(acc)
        return {f"loss_{k}": v * self.w[k] for k, v in out.items()}

    def dense_losses(self, outputs, targets):
        if outputs.get("class_group") is not None:
            return self.grouped_dense_losses(outputs, targets)
        S = int(outputs.get("slots", 1))
        if S == 1 and not self.reach and not self.balance:
            return super().dense_losses(outputs, targets)
        cls, reg, pts, st = outputs["cls"], outputs["reg"], outputs["points"], outputs["strides"]
        boxes = decode(reg, pts, st)
        if S == 1 and not self.reach and ASSIGN != "legacy":
            tsc, fg, tgt = self.group_assign_batched(cls, [boxes], pts, st, targets,
                                                     torch.zeros(cls.shape[-1], dtype=torch.long, device=cls.device))
            return self._dense_terms(cls, reg, boxes, pts, st, tsc, fg[0], tgt[0])
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
        return self._dense_terms(cls, reg, boxes, pts, st, tsc, fg, tgt)

    def _dense_terms(self, cls, reg, boxes, pts, st, tsc, fg, tgt):
        """Dense loss terms of one box per point from the assignment (target scores (B, A, C), fg (B, A), targets)."""
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
        dfl = dist_loss(r, s, lt, rt, thick)
        out["dfl"] = (dfl * w).sum() / norm
        out["across"] = (torch.log1p((r[:, 1] * s - at).abs() / thick.clamp(min=2.0)) * w).sum() / norm
        out["angle"] = angle_term(r[:, 0], th_t, w, norm)
        return {f"loss_{k}": v * self.w[k] for k, v in out.items()}
