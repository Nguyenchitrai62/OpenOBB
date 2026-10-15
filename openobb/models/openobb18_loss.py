"""OpenOBB18 / 19 / 20 losses: OpenOBB17's (dual dense heads, boundary field, refinement stages) + distillation from a
trained teacher (`teacher=`, e.g. an OpenOBB12 / 17 checkpoint):
  * neck: relative squared error between the student neck's P3 / P4 / P5 and the teacher encoder's, same augmented batch;
  * head (localization distillation, F77): on the points the teacher scores (max class probability >= LD_MIN), the
    student's box distributions (two along-axis ends, thickness) are pulled to the teacher's by a temperature-softened
    KL divergence (T = 10), the angle and across offset by an L1, and the class probabilities by a quality-focal BCE to
    the teacher's probabilities. With class groups, every student group learns from the teacher's box of that group
    (or the teacher's single box when it has one group), weighted by the teacher's best score within the group.
    Box precision (mAP50-95) is where the fast neck lost to the encoder (F75); the teacher's boxes are tighter.
Without a teacher the loss is OpenOBB17's."""
import torch
import torch.nn.functional as F

from .openobb3 import ALONG_BINS, DIRECT_CH, REG_CH
from .openobb17_loss import OpenOBB17Loss
from .openobb18 import distill_loss, teacher_feats

LD_T = 10.0              # temperature of the box-distribution KL (localization distillation)
LD_MIN = 0.05            # points where the teacher's best class probability is at least this are distilled


@torch.no_grad()
def teacher_outputs(teacher, x):
    """Teacher's neck / encoder outputs and its one-to-many head (cls logits, reg) on the same batch."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        _, feats = teacher.encode(x)
        cls, reg = teacher.run_head(feats, False)
    return [f.detach() for f in feats], cls.float(), reg.float()


def _kl(s, t, T=LD_T):
    """Per-row KL(softmax(t / T) || softmax(s / T)) x T^2 over the last dim."""
    return F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="none").sum(-1) * T * T


def box_distill(reg_s, reg_t):
    """(N, R) student / teacher box outputs of the same layout -> (N,) distillation loss."""
    if reg_s.shape[-1] == DIRECT_CH:
        return (reg_s - reg_t).abs().mean(-1)
    ang = (torch.remainder(reg_s[:, 0] - reg_t[:, 0] + torch.pi / 2, torch.pi) - torch.pi / 2).abs()
    acr = (reg_s[:, 1] - reg_t[:, 1]).abs()
    a, b = 2 + ALONG_BINS, 2 + 2 * ALONG_BINS
    kl = (_kl(reg_s[:, 2:a], reg_t[:, 2:a]) + _kl(reg_s[:, a:b], reg_t[:, a:b]) + _kl(reg_s[:, b:], reg_t[:, b:])) / 3
    return kl + ang * 5 + acr


class OpenOBB18Loss(OpenOBB17Loss):
    def __init__(self, teacher=None, w_distill=2.0, w_ld=0.25, w_kd_cls=0.5, **kw):
        super().__init__(**kw)
        self.teacher, self.w_distill = teacher, w_distill
        self.w_ld, self.w_kd_cls = w_ld, w_kd_cls
        self._warned = False

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if self.teacher is None or "feats" not in outputs:
            return losses
        if (self.w_ld > 0 or self.w_kd_cls > 0) and hasattr(self.teacher, "run_head"):
            tf, cls_t, reg_t = teacher_outputs(self.teacher, outputs["input"])
            losses["loss_distill"] = self.w_distill * distill_loss(outputs["feats"], tf)
            ld = self.head_distill(outputs, cls_t, reg_t)
            if ld is not None:
                losses["loss_ld"], losses["loss_kd_cls"] = self.w_ld * ld[0], self.w_kd_cls * ld[1]
        else:
            losses["loss_distill"] = self.w_distill * distill_loss(outputs["feats"],
                                                                   teacher_feats(self.teacher, outputs["input"]))
        return losses

    def head_distill(self, outputs, cls_t, reg_t):
        cls_s, reg_s = outputs["cls"], outputs["reg"].float()
        cg = outputs.get("class_group")
        G = 1 if cg is None else int(cg.max()) + 1
        R = reg_s.shape[-1] // G
        Rt = reg_t.shape[-1]
        if cls_t.shape != cls_s.shape or R not in (REG_CH, DIRECT_CH) or Rt not in (R, G * R):
            if not self._warned:
                print("[distill] teacher head layout differs from the student's: head distillation off "
                      "(neck distillation only)")
                self._warned = True
            return None
        p_t = cls_t.sigmoid()
        p_s = cls_s.float().sigmoid()
        # classes: quality-focal BCE to the teacher's probabilities (all points), normalised by the teacher's mass
        kd_cls = (F.binary_cross_entropy_with_logits(cls_s.float(), p_t, reduction="none")
                  * (p_t - p_s).abs().pow(2)).sum() / p_t.sum().clamp(min=1.0)
        # boxes: per group, on the points the teacher scores for that group's classes
        ld, wsum = reg_s.sum() * 0, 0.0
        cgv = cg.to(cls_s.device) if cg is not None else torch.zeros(cls_s.shape[-1], dtype=torch.long,
                                                                     device=cls_s.device)
        for g in range(G):
            w = p_t[..., cgv == g].amax(-1)                                            # (B, A)
            m = w >= LD_MIN
            if not m.any():
                continue
            rs = reg_s[..., g * R:(g + 1) * R][m]
            rt = reg_t[..., g * R:(g + 1) * R][m] if Rt == G * R else reg_t[m]
            ld = ld + (box_distill(rs, rt) * w[m]).sum()
            wsum += float(w[m].sum())
        return ld / max(wsum, 1.0), kd_cls
