"""OpenOBB5 losses: OpenOBB3's dense losses + a quality-focal loss on the relation re-scorer's corrected class logits."""
import torch.nn.functional as F

from .openobb3_loss import OpenOBB3Loss
from .openobb5 import relation_targets


class OpenOBB5Loss(OpenOBB3Loss):
    def __init__(self, w_rel=1.0, rel_thr=0.5, **kw):
        super().__init__(**kw)
        self.w_rel, self.rel_thr = w_rel, rel_thr

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if "rel_logits" in outputs:
            lg = outputs["rel_logits"]
            tsc = relation_targets(outputs["rel_boxes"], targets, self.img_size, lg.shape[-1], self.rel_thr)
            mod = (lg.detach().sigmoid() - tsc).abs().pow(2)
            bce = F.binary_cross_entropy_with_logits(lg, tsc, reduction="none")
            losses["loss_rel"] = self.w_rel * (bce * mod).sum() / tsc.sum().clamp(min=1.0)
        return losses
