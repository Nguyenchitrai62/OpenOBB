"""OpenOBB14 losses: OpenOBB12's dual dense losses (one-to-many + one-to-one, class balancing) on the unrefined boxes,
plus the edge refiner's box loss on the refined candidates.

Refiner targets (one-to-many, as OpenOBB8Loss): each candidate (a dense top-K box or a jittered labelled box) is
matched to the labelled box it overlaps most before refinement; ProbIoU >= ref_thr makes it a positive. Loss per
positive: 1 - ProbIoU(refined, labelled) + w_ref_l1 x L1 of the two end distances (units clamp(L / 10, 2, 32) px),
the log thickness, the across offset (units max(T / 2, 1) px) and the angle, all in the candidate's frame. The dense
heads are not trained through the refiner (candidates are detached): they stay good proposal generators, the
refiner learns the sub-stride correction from the stride-4 samples.
"""
import torch

from openobb.ops.obb_torch import probiou

from . import openobb10_loss as L10
from .openobb12_loss import OpenOBB12Loss
from .openobb3_loss import frame_targets


class OpenOBB14Loss(OpenOBB12Loss):
    def __init__(self, w_ref=2.0, w_ref_l1=0.2, ref_thr=0.5, **kw):
        kw.setdefault("off", "slots,reach")
        super().__init__(**kw)
        self.w_ref, self.w_ref_l1_, self.ref_thr_ = w_ref, w_ref_l1, ref_thr

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if "ref_boxes" in outputs:
            losses["loss_ref_box"] = self.w_ref * self.refine_loss(outputs, targets)
            # cascade (OpenOBB16): stage i is trained on the boxes stage i - 1 produced, with a higher threshold
            for i, (c, r, thr) in enumerate(outputs.get("ref_stages", []), start=2):
                losses[f"loss_ref_box{i}"] = self.w_ref * self.refine_loss(
                    {"ref_boxes": r, "ref_cand": c, "ref_valid": outputs.get("ref_valid")}, targets, thr)
            if "ref_fit" in outputs:
                losses["loss_ref_fit"] = self.fit_loss(outputs, targets)
        return losses

    def _padded(self, targets, device):
        """Labelled boxes of the batch padded to the largest image -> (B, M, 5) px, (B, M) valid (one device sync)."""
        cg = torch.zeros(1, dtype=torch.long, device=device)
        gb, gl, ok = L10.pad_targets([dict(t, labels=t["labels"] * 0) for t in targets], cg, 1, self.img_size, device)
        return gb, ok

    @torch.no_grad()
    def _best_match(self, boxes, targets):
        """boxes (B, K, 5) px -> best ProbIoU (B, K) with a labelled box of the same image (0 if none), its index
        (B, K) and the padded labelled boxes (B, M, 5)."""
        gb, ok = self._padded(targets, boxes.device)
        if gb.shape[1] == 0:
            z = boxes.new_zeros(boxes.shape[:2])
            return z, z.long(), gb
        iou = probiou(boxes[:, :, None, :], gb[:, None, :, :]).clamp(0, 1)
        iou = torch.where(ok[:, None, :], iou, torch.full_like(iou, -1.0))
        best, gi = iou.max(2)
        return best.clamp(min=0), gi, gb

    def fit_loss(self, outputs, targets):
        """BCE of the fit logit against ProbIoU(final refined box, matched object) (0 for unmatched candidates).
        `ref_fit_valid` (OpenOBB24): the rows the fit is learnt on (dense candidates only), else `ref_valid`."""
        fit, rb = outputs["ref_fit"].float(), outputs["ref_fit_boxes"].float().detach()
        valid = outputs.get("ref_fit_valid", outputs.get("ref_valid"))
        if L10.ASSIGN != "legacy":
            best = self._best_match(rb, targets)[0]
            tgt = torch.where(best >= 0.5, best, torch.zeros_like(best))
            l = torch.nn.functional.binary_cross_entropy_with_logits(fit, tgt, reduction="none")
            if valid is not None:
                l = l * valid
                return l.sum() / valid.sum().clamp(min=1.0)
            return l.mean()
        tgt = torch.zeros_like(fit)
        for b, t in enumerate(targets):
            if not len(t["labels"]):
                continue
            g = t["boxes"].float().clone().to(fit.device)
            g[:, :4] *= self.img_size
            with torch.no_grad():
                best = probiou(rb[b][:, None, :], g[None, :, :]).clamp(0, 1).amax(1)
            tgt[b] = torch.where(best >= 0.5, best, torch.zeros_like(best))
        l = torch.nn.functional.binary_cross_entropy_with_logits(fit, tgt, reduction="none")
        if valid is not None:
            l = l * valid
            return l.sum() / valid.sum().clamp(min=1.0)
        return l.mean()

    def refine_loss(self, outputs, targets, thr=None):
        thr = self.ref_thr_ if thr is None else thr
        rb, cand, valid = outputs["ref_boxes"].float(), outputs["ref_cand"].float(), outputs.get("ref_valid")
        if L10.ASSIGN != "legacy":
            return self._refine_loss_batched(rb, cand, valid, targets, thr)
        box_l, n_pos = rb.sum() * 0, 0                      # keeps the graph when there is no positive
        for b, t in enumerate(targets):
            if not len(t["labels"]):
                continue
            g = t["boxes"].float().clone().to(rb.device)
            g[:, :4] *= self.img_size
            with torch.no_grad():
                best, gi = probiou(cand[b][:, None, :], g[None, :, :]).clamp(0, 1).max(1)
                pos = best >= thr
                if valid is not None:
                    pos &= valid[b]
            if not pos.any():
                continue
            pi = pos.nonzero(as_tuple=True)[0]
            gt, r, c = g[gi[pi]], rb[b][pi], cand[b][pi]
            fit = probiou(r, gt).clamp(0, 1)
            lt, rt, at, _, thick, th_t = frame_targets(c[:, :2], c[:, 4], gt)
            ax = torch.stack([c[:, 4].cos(), c[:, 4].sin()], -1)
            nx = torch.stack([-ax[:, 1], ax[:, 0]], -1)
            rel = r[:, :2] - c[:, :2]
            du, dv = (rel * ax).sum(-1), (rel * nx).sum(-1)
            s_al = (c[:, 2] / 10).clamp(2.0, 32.0)
            s_ac = (c[:, 3] / 2).clamp(min=1.0)
            dth = torch.remainder(r[:, 4] - th_t + torch.pi / 2, torch.pi) - torch.pi / 2
            l1 = ((r[:, 2] / 2 - du - lt).abs() / s_al + (r[:, 2] / 2 + du - rt).abs() / s_al
                  + (torch.log(r[:, 3].clamp(min=0.5)) - torch.log(thick.clamp(min=0.5))).abs()
                  + (dv - at).abs() / s_ac + dth.abs() * 5)
            box_l = box_l + ((1 - fit) + self.w_ref_l1_ * l1.clamp(max=20)).sum()
            n_pos += len(pi)
        return box_l / max(n_pos, 1)

    def _refine_loss_batched(self, rb, cand, valid, targets, thr):
        """refine_loss over the whole batch at once (no loop over images): same positives, same per-positive terms,
        same mean over the batch's positives (tests/test_assign_batched.py)."""
        best, gi, gb = self._best_match(cand, targets)
        pos = best >= thr
        if valid is not None:
            pos &= valid
        if gb.shape[1] == 0:
            return rb.sum() * 0
        gt = gb.gather(1, gi[..., None].expand(-1, -1, 5))
        c = cand
        gt = torch.where(pos[..., None], gt, c.detach())                   # non-positives: a finite stand-in
        r = rb
        B, K = pos.shape
        f2 = lambda t: t.reshape(B * K, *t.shape[2:])                      # noqa: E731
        fit = probiou(f2(r), f2(gt)).clamp(0, 1).view(B, K)
        lt, rt, at, _, thick, th_t = (t.view(B, K) for t in frame_targets(f2(c)[:, :2], f2(c)[:, 4], f2(gt)))
        ax = torch.stack([c[..., 4].cos(), c[..., 4].sin()], -1)
        nx = torch.stack([-ax[..., 1], ax[..., 0]], -1)
        rel = r[..., :2] - c[..., :2]
        du, dv = (rel * ax).sum(-1), (rel * nx).sum(-1)
        s_al = (c[..., 2] / 10).clamp(2.0, 32.0)
        s_ac = (c[..., 3] / 2).clamp(min=1.0)
        dth = torch.remainder(r[..., 4] - th_t + torch.pi / 2, torch.pi) - torch.pi / 2
        l1 = ((r[..., 2] / 2 - du - lt).abs() / s_al + (r[..., 2] / 2 + du - rt).abs() / s_al
              + (torch.log(r[..., 3].clamp(min=0.5)) - torch.log(thick.clamp(min=0.5))).abs()
              + (dv - at).abs() / s_ac + dth.abs() * 5)
        per = (1 - fit) + self.w_ref_l1_ * l1.clamp(max=20)
        return torch.where(pos, per, torch.zeros_like(per)).sum() / pos.sum().clamp(min=1)
