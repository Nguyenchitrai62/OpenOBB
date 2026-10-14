"""OpenOBB12 / OpenOBB13 losses: the same dense oriented loss on both heads, with consistent assignments.

  * one-to-many head ("cls", "reg"): OpenOBB10's dense loss with class balancing (task-aligned, 10 + length / 16
    positives per object, at most 48);
  * one-to-one head ("o2o_cls", "o2o_reg"): the same loss and the same matching metric (score^alpha x IoU^beta,
    alpha = 1, beta = 6) with ONE positive per object (k = 1): the anchor ranked best for the object; every other
    anchor is a negative, so the head learns to give one box per object (no NMS at inference). Keys loss_o2o_*.
"""
from .openobb10_loss import OpenOBB10Loss


class OpenOBB12Loss(OpenOBB10Loss):
    def __init__(self, w_o2o=1.0, **kw):
        kw.setdefault("off", "slots,reach")               # dense loss of OpenOBB10 with class balancing only
        super().__init__(**kw)
        o2o = dict(kw, topk=1, topk_len=1e9, topk_max=1)
        self.o2o = OpenOBB10Loss(**o2o)
        self.w_o2o = w_o2o

    def forward(self, outputs, targets):
        losses = self.dense_losses(outputs, targets)
        if "o2o_cls" in outputs:
            o = {"cls": outputs["o2o_cls"], "reg": outputs["o2o_reg"], "points": outputs["points"],
                 "strides": outputs["strides"], "class_group": outputs.get("class_group")}
            for k, v in self.o2o.dense_losses(o, targets).items():
                losses[k.replace("loss_", "loss_o2o_", 1)] = v * self.w_o2o
        return losses
