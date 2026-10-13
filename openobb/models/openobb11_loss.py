"""OpenOBB11 losses: the dense head as OpenOBB10 (one-to-many, class balancing; no slots / reach filter) + the
relational stage trained ONE-TO-ONE, so it learns to remove duplicates (docs/OPENOBB11.md).

Refiner targets per image:
  * matching group (the K dense candidates): Hungarian matching between the refined candidates and the labelled
    objects, cost = -(p^alpha * ProbIoU^beta) on the refined predictions (p = refined probability of the object's
    class; alpha / beta as the dense task-aligned assignment). A pair needs ProbIoU >= min_iou. One candidate per
    object is a positive; every other candidate - copies, fragments, shifted boxes - is a negative (target 0);
  * denoising group (jittered labelled boxes, OpenOBB8 jitter): every row is the positive of its own object (DN-DETR);
    the attention mask already keeps the two groups apart.
Classification: quality focal loss with target = ProbIoU(refined box, object) for positives, as OpenOBB8; box losses
for positives only, as OpenOBB8.
"""
import torch
from scipy.optimize import linear_sum_assignment

from openobb.ops.obb_torch import probiou

from .openobb10_loss import OpenOBB10Loss


class OpenOBB11Loss(OpenOBB10Loss):
    def __init__(self, alpha=1.0, beta=6.0, min_iou=0.1, **kw):
        kw.setdefault("off", "slots,reach")          # dense: class balancing only (F47: slots / reach not kept)
        super().__init__(**kw)
        self.m_alpha, self.m_beta, self.min_iou = alpha, beta, min_iou

    def ref_assign(self, b, lg, rb, cand, g, labels, outputs):
        N = lg.shape[1]
        K = int(outputs.get("ref_dense", N))
        pos = torch.zeros(N, dtype=torch.bool, device=lg.device)
        gi = torch.zeros(N, dtype=torch.long, device=lg.device)
        if not outputs.get("o2o", False):                 # an OpenOBB8 output: keep OpenOBB8's targets
            return super().ref_assign(b, lg, rb, cand, g, labels, outputs)
        iou = probiou(rb[b][:K, None, :], g[None, :, :]).clamp(0, 1)                  # (K, M) refined vs labels
        p = lg[b][:K].sigmoid()[:, labels]                                            # (K, M)
        valid = outputs.get("ref_valid")
        ok = iou >= self.min_iou
        if valid is not None:
            ok &= valid[b][:K, None]
        if ok.any():
            metric = p.pow(self.m_alpha) * iou.pow(self.m_beta)
            cost = torch.where(ok, -metric, torch.full_like(metric, 1e6)).cpu().numpy()
            r, c = linear_sum_assignment(cost)
            keep = cost[r, c] < 1e5
            r = torch.as_tensor(r[keep], device=lg.device, dtype=torch.long)
            c = torch.as_tensor(c[keep], device=lg.device, dtype=torch.long)
            pos[r], gi[r] = True, c
        jg = outputs.get("ref_jit_gt")
        if jg is not None and jg.shape[1]:
            j = jg[b]
            m = j >= 0
            rows = K + torch.arange(len(j), device=lg.device)
            pos[rows[m]], gi[rows[m]] = True, j[m]
        return pos, gi

