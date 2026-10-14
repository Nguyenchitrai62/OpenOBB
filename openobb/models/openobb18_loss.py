"""OpenOBB18 losses: OpenOBB17's (dual dense heads, boundary field, snapping) + neck distillation from a trained
OpenOBB12 / 17 teacher: relative squared error between the student neck's P3 / P4 / P5 and the teacher encoder's on
the same augmented batch. Without a teacher the loss is OpenOBB17's (the neck then learns from the detection losses
only, the one-factor control)."""
from .openobb17_loss import OpenOBB17Loss
from .openobb18 import distill_loss, teacher_feats


class OpenOBB18Loss(OpenOBB17Loss):
    def __init__(self, teacher=None, w_distill=2.0, **kw):
        super().__init__(**kw)
        self.teacher, self.w_distill = teacher, w_distill

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if self.teacher is not None and "feats" in outputs:
            losses["loss_distill"] = self.w_distill * distill_loss(outputs["feats"],
                                                                   teacher_feats(self.teacher, outputs["input"]))
        return losses
