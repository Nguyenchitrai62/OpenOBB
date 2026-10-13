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
OpenOBB13 = OpenOBB12 + a full-resolution detail stream and a stride-4 grid (P2) for tiny objects (docs/OPENOBB13.md).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb3 import DIRECT_CH, OrientedHead, decode, fast_nms
from .openobb6 import ConvBN
from .openobb7 import ContextOrientedHead, OpenOBB7, space_to_depth


class DualHeadMixin:
    """`head` (one-to-many, OpenOBB7's ContextOrientedHead) + `head_o2o` (same design, own weights)."""

    def _add_o2o(self, final_nms):
        self.head_o2o = copy.deepcopy(self.head)
        self.final_nms = final_nms            # None: the one-to-one head's output as it is (NMS-free)
        # head used at inference, calibrated on val at the end of training (args["infer_head"]): "o2o" (NMS-free) or
        # "o2m" (the one-to-many head + NMS). F54: on Wall_Color the jointly trained o2m head + NMS 0.5 scored higher.
        self.infer_head = "o2o"

    @torch.no_grad()
    def copy_head_to_o2o(self):
        """Warm start of the one-to-one head from the one-to-many head (after loading a single-head checkpoint)."""
        self.head_o2o.load_state_dict(self.head.state_dict())

    def features(self, x):
        return self.encode(x)[1]

    def run_head(self, feats, o2o):
        return (self.head_o2o if o2o else self.head)(feats)

    def forward(self, x, targets=None, ctx=None):
        feats = self.features(x)
        pts, st = self.anchors(feats)
        if self.training:
            cls, reg = self.run_head(feats, False)
            c1, r1 = self.run_head(feats, True)
            return {"cls": cls, "reg": reg, "points": pts, "strides": st, "o2o_cls": c1, "o2o_reg": r1}
        o2o = self.infer_head != "o2m"
        cls, reg = self.run_head(feats, o2o)
        boxes = decode(reg, pts, st)
        conf = getattr(self, "nms_conf", 0.001)
        nms = self.final_nms if o2o or self.final_nms is not None else self.nms_iou   # one-to-many: always NMS
        if nms is not None:
            logits, boxes = fast_nms(cls, boxes, self.max_det, conf=conf, iou=float(nms))
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


BOX_MODES = ("dfl", "direct")


def make_head(ch, nc, ctx, box):
    """Dense oriented head: OpenOBB7's (context unit in the class tower unless cls_ctx is off); box="direct" regresses
    the distances and thickness as values (YOLO26-style, F54) instead of DFL distributions."""
    kw = {"reg_ch": DIRECT_CH} if box == "direct" else {}
    return ContextOrientedHead(ch, nc, **kw) if ctx else OrientedHead(ch, nc, **kw)


class OpenOBB12(DualHeadMixin, OpenOBB7):
    arch = "v12"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 final_nms=None, box="dfl", **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off)
        if box not in BOX_MODES:
            raise ValueError(f"box must be one of {BOX_MODES}, got {box!r}")
        self.box = box
        if box != "dfl":
            self.head = make_head([self.encoder.hidden_dim] * 3, num_classes, "cls_ctx" not in self.off, box)
        self._add_o2o(final_nms)


class DWRes(nn.Module):
    """x + pw(dw3x3(x)): cheap residual block at full resolution."""

    def __init__(self, c):
        super().__init__()
        self.dw = ConvBN(c, c, 3, g=c)
        self.pw = ConvBN(c, c, 1, act=False, zero=True)

    def forward(self, x):
        return x + self.pw(self.dw(x))


class HiResStream(nn.Module):
    """Light CNN over the full-resolution input (the core sees it at half resolution): a stride-8 map (input px), i.e.
    the stride-4 grid of the core's scale, carrying pixels the core never sees."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(ConvBN(3, 24, 3, 2), ConvBN(24, 48, 3, 2), DWRes(48), DWRes(48),
                                 ConvBN(48, c, 3, 2), DWRes(c), DWRes(c))

    def forward(self, x):
        return self.net(x)


class OpenOBB13(OpenOBB12):
    """Input = full resolution (imgsz, e.g. 2560); the pretrained core runs on it at half resolution (as OpenOBB12 at
    imgsz / 2). New, all starting as no-ops for the core's own levels:
      hires    full-resolution stream -> stride 8 (input px) map
      to_p3    its stride-16 summary added to the core's P3 backbone map before LSK (zero init: P3-P5 = OpenOBB12's)
      p2       P2 level: lateral of the stream + the encoder's P3 brought down to P2 channels and upsampled, then a
               residual block; its own dense head pair (p2_head, p2_head_o2o)
    Strides in input px: P2 8, P3 16, P4 32, P5 64 (= 4 / 8 / 16 / 32 at the core's scale)."""
    arch = "v13"
    PAD = 64                                   # the core halves the input, then needs multiples of 32

    def __init__(self, size="x", num_classes=15, img_size=2048, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 final_nms=None, box="dfl", **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off,
                         final_nms=final_nms, box=box)
        hid, cin = self.encoder.hidden_dim, self.encoder.in_channels
        c2 = max(64, hid // 2)
        self.hires = HiResStream(c2)
        self.to_p3 = ConvBN(c2, cin[0], 3, 2, act=False, zero=True)
        self.p2_lat = ConvBN(c2, c2, 1, act=False)
        self.p2_down = ConvBN(hid, c2, 1, act=False)
        self.p2_fuse = DWRes(c2)
        self.p2_head = make_head([c2], num_classes, True, box)
        self.p2_head_o2o = copy.deepcopy(self.p2_head)
        self.strides = (16, 32, 64, 8)

    @torch.no_grad()
    def copy_head_to_o2o(self):
        super().copy_head_to_o2o()
        self.p2_head_o2o.load_state_dict(self.p2_head.state_dict())

    def features(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.PAD, (-W) % self.PAD
        if ph or pw:                           # bottom / right only: anchor coordinates are unchanged
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        hr = self.hires(x)
        xc = F.avg_pool2d(x, 2)                # the core's input: what OpenOBB12 sees at imgsz / 2
        bb = self.backbone
        h = bb.stem(xc)
        outs = []
        for stage in bb.stages:
            h = stage(h)
            outs.append(h if self.guard is None else self.guard(h))
        f4, feats = outs[0], outs[1:]
        if self.spd is not None:
            feats[0] = feats[0] + self.spd(space_to_depth(f4, feats[0].shape[-2:]))
        feats[0] = feats[0] + self.to_p3(hr)
        if self.lsk is not None:
            feats = [m(f) for m, f in zip(self.lsk, feats)]
        feats = list(self.encoder(feats))
        p2 = self.p2_lat(hr) + F.interpolate(self.p2_down(feats[0]), size=hr.shape[-2:], mode="nearest")
        return feats + [self.p2_fuse(p2)]

    def encode(self, x):
        """-> (None, [P3, P4, P5, P2]) at input-px strides 16 / 32 / 64 / 8."""
        return None, self.features(x)

    def run_head(self, feats, o2o):
        c, r = (self.head_o2o if o2o else self.head)(feats[:3])
        c2, r2 = (self.p2_head_o2o if o2o else self.p2_head)(feats[3:])
        return torch.cat([c, c2], 1), torch.cat([r, r2], 1)
