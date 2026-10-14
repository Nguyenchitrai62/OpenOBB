"""OpenOBB18: the fast CAD core. OpenOBB17 (pretrained HGNetv2 trunk, OpenOBB12's dual dense DFL head, boundary field
+ snapping) with the slow borrowed middle (bounded LSK adapters + D-FINE hybrid encoder) replaced by OpenOBB's own
compute-bound re-parameterised neck, which is DISTILLED from the encoder of a trained OpenOBB12 / 17 on drawings
instead of learning from scratch. docs/OPENOBB18.md.

Why (research/LEDGER.md F37, F50, F58, F60, F64-F66):
  * Speed: the hybrid encoder (38% of OpenOBB12's FLOPs, its P3 FPN block alone 28%) and the LSK adapters (7%) are
    1x1-heavy / depthwise / attention, memory-bound on a GPU (roofline, F58), and the encoder's attention at 1/32 is
    flat on drawings (F37). OpenOBB14's neck of dense 3x3 RepConv (folded into single convolutions at inference) ran
    the whole model 29% faster than OpenOBB12 on the same G4 (10.9 vs 15.3 ms, F60).
  * Why OpenOBB14 was not the answer: its accuracy fell on the semantic / rare classes (note, slide_door, wall_300),
    and OpenOBB14 with D-FINE's encoder was just as bad (F65), so its new head (direct boxes, heavy towers) and
    refiner were at fault, while its neck did well where geometry matters (wall mAP50-95 +0.045, F60). Here the neck
    is OpenOBB14's, the head is OpenOBB12's (the best measured head), copied from a trained OpenOBB12 checkpoint.
  * Priors decide everything on 250 images (F39: COCO init ~2x the mAP50-95 of scratch); every core trained from
    scratch or distilled from a COCO teacher failed (F25, F40, F50). Here the trunk and head keep their trained
    weights and only the neck is new; it learns to reproduce the TRAINED encoder's P3 / P4 / P5 on the drawings
    themselves (in-domain teacher, the same trunk features as input), so the copied head receives what it was trained
    on, plus the detection losses. The teacher is used in training only and is not part of the model.
  * Accuracy: OpenOBB17's boundary field supervises every labelled side densely and snaps box ends / sides (F64:
    ends are the largest box error, +0.116 mAP50-95 ceiling), and the neck brings stride-4 detail by space-to-depth
    and dilated 3x3 context at P3 / P4.
"""
import torch

from .openobb14 import SCALES14, Neck14
from .openobb17 import OpenOBB17


class OpenOBB18(OpenOBB17):
    arch = "v18"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 final_nms=None, box="dfl", ref_k=1500, groups=None, **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off,
                         final_nms=final_nms, box=box, ref_k=ref_k, groups=groups)
        hid, cin = self.encoder.hidden_dim, list(self.encoder.in_channels)
        c4 = self.backbone._out_channels[0]
        # same width as the encoder it replaces: the OpenOBB12 head (and its trained weights) reads it unchanged
        self.neck = Neck14(cin, c4, hid, SCALES14[size]["depth"])
        self.hid = hid
        del self.encoder
        self.lsk, self.spd = None, None

    def encode(self, x):
        """-> (stride-4 trunk map, [P3, P4, P5] from the own neck)."""
        bb = self.backbone
        h = bb.stem(x)
        outs = []
        for stage in bb.stages:
            h = stage(h)
            outs.append(h if self.guard is None else self.guard(h))
        f4, c3, c4, c5 = outs
        return f4, self.neck(f4, c3, c4, c5)


def distill_loss(student, teacher):
    """Mean over levels of the relative squared error ||s - t||^2 / ||t||^2 (scale-free across levels)."""
    total = 0.0
    for s, t in zip(student, teacher):
        s, t = s.float(), t.float()
        total = total + (s - t).pow(2).mean() / t.pow(2).mean().clamp(min=1e-6)
    return total / len(student)


@torch.no_grad()
def teacher_feats(teacher, x):
    """P3 / P4 / P5 of a trained OpenOBB12 / 17 (frozen, eval) on the same batch (bf16 autocast on CUDA)."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        return [f.detach() for f in teacher.encode(x)[1]]


class OpenOBB19(OpenOBB18):
    """OpenOBB18 (fast own neck distilled from a trained encoder, boundary field) + contention-aware class groups:
    one box per class group at every grid point (models/groups.py, docs/OPENOBB19.md). groups = list of one group id
    per class, computed from the training labels by the trainer (groups=auto)."""
    arch = "v19"
