"""OpenOBB12 / OpenOBB13: a dual dense oriented head - one-to-many for learning, one-to-one for inference
(consistent dual assignment, the YOLOv10 / YOLO26 idea), NMS-free, no relational stage. docs/OPENOBB12.md.

Why (research/LEDGER.md F43-F50):
  * Accuracy came from the pretrained core + the one-to-many dense head (OpenOBB3 / 8 / 10: 0.69-0.72); every
    extra stage on top (context, slots, relational refiner; v5 / v7 / v8 / v10 / v11) added no accuracy, and the
    relational refiner took most of the inference time (v11 13.6 ms vs a refiner-free dense model 5.4 ms, F50).
  * Duplicates come from the one-to-many head itself (up to 48 positives per object); NMS on IoU is blind to them on
    thin objects (F49). OpenOBB11 removed them with a one-to-one relational decoder trained by Hungarian matching,
    which learnt slowly on 250 images (recall fell, NMS 0.5 still chosen, F50).
  * Here the one-to-one branch is a second DENSE head of the same design, on the same features, trained with the
    SAME task-aligned metric (score^alpha x IoU^beta) but top-1 per object: the same rule that ranks the one-to-many
    positives picks its single positive (consistent matching), so it learns like the dense head and needs no
    transformer stage. Every other anchor of the object is a negative: at inference the one-to-one head gives one
    box per object, without NMS. The one-to-many head only trains the shared features; it is not run at inference.
  * NMS stays a calibrated fallback (args["nms_iou"]: "none" / 0.7 / 0.5 chosen on val at the end of training).

OpenOBB12 = OpenOBB7 core (pretrained HGNetv2 + bounded LSK + hybrid encoder, repaired) + dual head.
OpenOBB13 = OpenOBB9 fast core (re-parameterised dense 3x3 convolutions, 5.4 ms) + the same dual head; its core needs
pretraining (`pretrain=` with an OpenOBB teacher, e.g. a trained OpenOBB10, as `pretrain_teacher=`).
"""
import copy
import math

import torch

from .openobb3 import decode, fast_nms
from .openobb7 import OpenOBB7
from .openobb9 import OpenOBB9


class DualHeadMixin:
    """`head` (one-to-many, OpenOBB7's ContextOrientedHead) + `head_o2o` (same design, own weights)."""

    def _add_o2o(self, final_nms):
        self.head_o2o = copy.deepcopy(self.head)
        self.final_nms = final_nms            # None: the one-to-one head's output as it is (NMS-free)

    @torch.no_grad()
    def copy_head_to_o2o(self):
        """Warm start of the one-to-one head from the one-to-many head (after loading a single-head checkpoint)."""
        self.head_o2o.load_state_dict(self.head.state_dict())

    def forward(self, x, targets=None, ctx=None):
        _, feats = self.encode(x)
        pts, st = self.anchors(feats)
        if self.training:
            cls, reg = self.head(feats)
            c1, r1 = self.head_o2o(feats)
            return {"cls": cls, "reg": reg, "points": pts, "strides": st, "o2o_cls": c1, "o2o_reg": r1}
        cls, reg = self.head_o2o(feats)       # inference: the one-to-one head only
        boxes = decode(reg, pts, st)
        conf = getattr(self, "nms_conf", 0.001)
        if self.final_nms is not None:
            logits, boxes = fast_nms(cls, boxes, self.max_det, conf=conf, iou=float(self.final_nms))
        else:
            k = min(self.max_det, cls.shape[1])
            order = cls.amax(-1).topk(k, dim=1).indices
            logits = cls.gather(1, order[..., None].expand(-1, -1, cls.shape[-1])).float()
            boxes = boxes.gather(1, order[..., None].expand(-1, -1, 5)).float()
            logits = logits.masked_fill(logits.sigmoid().amax(-1, keepdim=True) < conf, -30.0)
        boxes = boxes.clone()
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}


class OpenOBB12(DualHeadMixin, OpenOBB7):
    arch = "v12"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 final_nms=None, **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off)
        self._add_o2o(final_nms)


class OpenOBB13(DualHeadMixin, OpenOBB9):
    arch = "v13"

    def __init__(self, size="x", num_classes=15, img_size=1024, max_det=1000, nms_iou=0.7, off=None, final_nms=None,
                 **_):
        super().__init__(size, num_classes, img_size, max_det=max_det, nms_iou=nms_iou, off=off, off8="refine")
        self._add_o2o(final_nms)
