"""OpenOBB8 losses: OpenOBB3's dense losses (unchanged) + the oriented relational refiner's losses.

Refiner targets: each candidate is matched to the labelled box it overlaps most (ProbIoU of the candidate before
refinement); >= `ref_thr` makes it a positive of that box. Losses:
  * loss_ref_cls: quality focal loss; the target of a positive is ProbIoU(refined box, labelled box) (detached) for
    the matched class, 0 for every other class and for unmatched candidates (so the refined score also ranks by fit,
    which mAP50-95 rewards);
  * loss_ref_box: for positives, 1 - ProbIoU(refined, labelled) plus a small L1 on the two end points, the log
    thickness, the across offset and the angle, all expressed in the candidate's own frame (the frame refine_boxes
    moves in; openobb3_loss.frame_targets with the candidate centre as the point and the candidate angle).
"""
import torch
import torch.nn.functional as F

from openobb.ops.obb_torch import probiou

from .openobb3_loss import OpenOBB3Loss, frame_targets


class OpenOBB8Loss(OpenOBB3Loss):
    def __init__(self, w_ref_cls=1.0, w_ref_box=2.0, ref_thr=0.5, w_ref_l1=0.2, **kw):
        super().__init__(**kw)
        self.w_ref_cls, self.w_ref_box, self.ref_thr, self.w_ref_l1 = w_ref_cls, w_ref_box, ref_thr, w_ref_l1

    def dense_losses(self, outputs, targets):
        """Losses of the dense head (OpenOBB3's); OpenOBB10Loss overrides this."""
        return super().forward(outputs, targets)

    def ref_assign(self, b, lg, rb, cand, g, labels, outputs):
        """Refiner targets of image b -> (positive mask (N,), object index (N,)). OpenOBB8: one-to-many, every
        candidate whose ProbIoU (before refinement) with an object is >= ref_thr is a positive of that object.
        OpenOBB11Loss overrides this (one-to-one)."""
        best, gi = probiou(cand[b][:, None, :], g[None, :, :]).clamp(0, 1).max(1)
        return best >= self.ref_thr, gi

    def forward(self, outputs, targets):
        losses = self.dense_losses(outputs, targets)
        if "ref_logits" not in outputs:
            return losses
        lg, rb, cand = outputs["ref_logits"].float(), outputs["ref_boxes"].float(), outputs["ref_cand"].float()
        valid = outputs.get("ref_valid")
        tsc = torch.zeros_like(lg)
        box_l, n_pos = rb.sum() * 0, 0                  # keeps the graph when there is no positive
        for b, t in enumerate(targets):
            if not len(t["labels"]):
                continue
            g = t["boxes"].float().clone().to(lg.device)
            g[:, :4] *= self.img_size
            with torch.no_grad():
                pos, gi = self.ref_assign(b, lg, rb, cand, g, t["labels"].to(lg.device), outputs)
                if valid is not None:
                    pos &= valid[b]
            if not pos.any():
                continue
            pi = pos.nonzero(as_tuple=True)[0]
            gt, lab = g[gi[pi]], t["labels"].to(lg.device)[gi[pi]]
            r, c = rb[b][pi], cand[b][pi]
            fit = probiou(r, gt).clamp(0, 1)
            tsc[b, pi, lab] = fit.detach()
            # targets in the candidate frame: back / front end distances from the candidate centre, across offset
            # (gt centre - candidate centre) . normal, length, thickness, axis angle (flipped to the candidate's)
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
            box_l = box_l + ((1 - fit) + self.w_ref_l1 * l1.clamp(max=20)).sum()
            n_pos += len(pi)
        mod = (lg.detach().sigmoid() - tsc).abs().pow(2)
        bce = F.binary_cross_entropy_with_logits(lg, tsc, reduction="none") * mod
        if valid is not None:
            bce = bce * valid[..., None]
        losses["loss_ref_cls"] = self.w_ref_cls * bce.sum() / tsc.sum().clamp(min=1.0)
        losses["loss_ref_box"] = self.w_ref_box * box_l / max(n_pos, 1)
        return losses
