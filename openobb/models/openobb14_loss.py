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
        return losses

    def refine_loss(self, outputs, targets):
        rb, cand, valid = outputs["ref_boxes"].float(), outputs["ref_cand"].float(), outputs.get("ref_valid")
        box_l, n_pos = rb.sum() * 0, 0                      # keeps the graph when there is no positive
        for b, t in enumerate(targets):
            if not len(t["labels"]):
                continue
            g = t["boxes"].float().clone().to(rb.device)
            g[:, :4] *= self.img_size
            with torch.no_grad():
                best, gi = probiou(cand[b][:, None, :], g[None, :, :]).clamp(0, 1).max(1)
                pos = best >= self.ref_thr_
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
