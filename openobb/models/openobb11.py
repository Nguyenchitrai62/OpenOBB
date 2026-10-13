"""OpenOBB11: dense proposals + a relational SET decoder trained one-to-one (learned duplicate removal, NMS-free).

Why (research/LEDGER.md F47, F49; docs/OPENOBB11.md):
  * Every dense head of OpenOBB3-10 is trained one-to-many on purpose: 10 + length / 16 positives per object (up to 48),
    each taught to draw the whole object with a high score. A long wall therefore comes out as tens of high-scoring
    copies whose two ends differ (far points see the ends worst). Only NMS removes them.
  * IoU-based NMS is blind to that failure on thin objects: two collinear boxes of equal thickness have IoU = shorter /
    longer length (a 200 px copy of a 300 px wall: 0.67 < 0.7, both survive). OpenOBB10 (3 slots per point) had
    16.7% duplicate detections; the duplicates left by NMS are mostly fragments / shifted copies, not near-identical.
  * OpenOBB8's relational stage sees every candidate in its own rotated frame and compares candidates through a
    relative-geometry attention bias - exactly the information that tells "a copy cut short" from "another wall" -
    but it was trained one-to-MANY (every candidate with ProbIoU >= 0.5 to an object was a positive), so it was taught
    to keep the copies. It never learnt to remove them.
OpenOBB11 trains that stage ONE-TO-ONE (openobb11_loss.py): per image, a Hungarian matching between the refined
candidates and the objects; one candidate per object is a positive, every other candidate - copies and fragments
included - is a negative (Relation Networks used a relation module to replace NMS; DETR / YOLOv10 / YOLO26 use
one-to-one assignment to drop NMS). The dense head keeps its one-to-many loss (fast convergence, rich gradients) and
becomes the proposal generator. Jittered labelled boxes are a separate denoising group (DN-DETR): each is the positive
of its own object, masked from the matching group.

Inference: dense head -> K distinct proposals -> relational stage -> the K refined candidates ARE the output.
No duplicate spreading over 4000 anchors, no class NMS (`final_nms` = an optional ProbIoU threshold, off by default).
Parameter names = OpenOBB8's (an OpenOBB8 / OpenOBB10 checkpoint loads; OpenOBB10's extra slot layers are ignored).
"""
import math

import torch

from .obb_decoder import distinct_topk
from .openobb3 import decode, fast_nms
from .openobb8 import OpenOBB8


class OpenOBB11(OpenOBB8):
    arch = "v11"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 off7=None, rel_k=600, final_nms=None, **_):
        off = ",".join(sorted({*(s.strip() for s in (off or "").split(",") if s.strip())} - {"refine"})) or None
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off, off7=off7,
                         rel_k=rel_k)
        self.final_nms = final_nms            # None: the set decoder's output as it is

    def forward(self, x, targets=None, ctx=None):
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        boxes = decode(reg, pts, st)
        sc = cls.detach().sigmoid().amax(-1)
        k = min(self.refiner.k // 2 if self.training else self.refiner.k, sc.shape[1])
        idx = distinct_topk(sc, boxes.detach(), k, 2 * k)
        gather = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
        cand, base = gather(boxes).detach(), gather(cls)
        block = valid = jit_gt = None
        if self.training and targets is not None and "jitter" not in self.off8:
            cand, base, block, valid, jit_gt = self._with_jitter(cand, base, cls, feats, targets, return_idx=True)
        ref_logits, ref_boxes = self.refiner(f4, feats[0], feats[1], base, cand, self.img_size, block)
        if self.training:
            out = {"cls": cls, "reg": reg, "points": pts, "strides": st, "ref_logits": ref_logits,
                   "ref_boxes": ref_boxes, "ref_cand": cand.detach(), "ref_dense": k, "o2o": True}
            if valid is not None:
                out["ref_valid"], out["ref_jit_gt"] = valid, jit_gt
            return out
        logits, out_boxes = ref_logits.float(), ref_boxes.float()
        if self.final_nms is not None:
            logits, out_boxes = fast_nms(logits, out_boxes, self.max_det, conf=getattr(self, "nms_conf", 0.001),
                                         iou=float(self.final_nms))
        else:                                  # best first, padded / cut to max_det like the NMS output
            order = logits.amax(-1).argsort(1, descending=True)[:, :self.max_det]
            logits = logits.gather(1, order[..., None].expand(-1, -1, logits.shape[-1]))
            out_boxes = out_boxes.gather(1, order[..., None].expand(-1, -1, 5))
            low = logits.sigmoid().amax(-1) < getattr(self, "nms_conf", 0.001)
            logits = logits.masked_fill(low[..., None], -30.0)
        out_boxes = out_boxes.clone()
        out_boxes[..., :4] = out_boxes[..., :4] / self.img_size
        out_boxes[..., 4] = torch.remainder(out_boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": out_boxes}
