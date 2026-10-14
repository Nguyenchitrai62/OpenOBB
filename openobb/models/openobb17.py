"""OpenOBB17: OpenOBB12 + a learned boundary field and boundary snapping (the partition detector). docs/OPENOBB17.md.

Why (research/LEDGER.md F64, measured on Wall_Color with the trained OpenOBB12):
  * Box geometry, not classification, holds what is left: perfect boxes on the same detections and scores would give
    0.799 mAP50-95 instead of 0.478. The largest single component is where an object ENDS along its axis (+0.116;
    wall_300 +0.235, note +0.160, junction +0.132, slide_door +0.123), then the across offset (+0.049) and the
    thickness (+0.036); the angle is negligible (+0.003).
  * The labelled boxes are a drawing convention, not visible edges: moving every labelled side to the strongest image
    edge within 2 px leaves a median IoU of 0.51-0.61 with itself on walls / junctions / wall_300 (-0.09 mAP50-95 when
    done on the detections). Looking harder at pixels (OpenOBB15's full-resolution stream) did not help (F63).
  * Ends are shared: 85% of wall_300 ends, 77% of slide_door ends and 66% of wall ends lie within 2 px of a side of
    another labelled object. Walls are continuous strokes partitioned into objects; an end is the boundary with the
    next object.
  Every earlier OpenOBB (and YOLO) regresses each object's extent from one anchor point, independently of its
  neighbours. OpenOBB17 learns the boundaries themselves: a dense two-channel field at stride 2 (object ends, long
  sides), supervised by EVERY labelled side (thousands of supervised pixels per image instead of one box target per
  object), and moves each refined box's ends and sides to the field's peak by a 1-D soft-argmax search in the box's
  own frame. Neighbours read the same ridge, so their shared boundary agrees. The snapping gains start at zero:
  OpenOBB17 begins exactly as OpenOBB12 (test), and the field trains from the first step through its own loss.
  Base: OpenOBB12, the most accurate OpenOBB (0.743 / 0.472, F59); the OpenOBB14 line (direct boxes, own head and
  refiner) stayed below it with either neck (F60, F63, user logs 10-15), so it is not the base.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb3 import decode, fast_nms
from .openobb6 import ConvBN
from .openobb8 import _sample
from .openobb9 import RepConv
from .openobb12 import OpenOBB12
from .openobb14 import OpenOBB14

FIELD_STRIDE = 2         # input px per field cell (computed at stride 4, pixel-shuffled x2)
FIELD_SIGMA = 1.0        # input px, width of the Gaussian ridge drawn on every labelled side
SQUARE = 1.3             # L / T below this: the four sides are drawn in both channels (ends / sides are ambiguous)
N_T = 33                 # samples of every 1-D search profile
N_LINES = 3              # parallel profiles averaged per side
END_R = (3.0, 0.15, 16.0)   # end search radius: clamp(0.15 L, 3, 16) px
SIDE_R = (2.0, 0.5, 6.0)    # side search radius: clamp(0.5 T, 2, 6) px
GAIN_SCALE = 20.0        # snapping gain = GAIN_SCALE x parameter (zero at init): reaches 1 in ~50 Adam steps
REF_K = 1500             # candidates snapped per image at inference (before NMS)
REF_K_TRAIN = 256        # dense candidates snapped per image in training (plus as many jittered labelled boxes)


class BoundaryField(nn.Module):
    """Stride-4 trunk map + P3 (upsampled) -> 1x1 -> two RepConv 3x3 (the second dilated) at stride 4 -> 1x1 to
    2 x 4 logits -> pixel shuffle: two logit maps at stride 2, channel 0 = object ends (short sides), channel 1 = long
    sides. Bias at the 1% prior, so the field starts quiet."""

    def __init__(self, c4, c3, ch=64, prior=0.01):
        super().__init__()
        self.p4 = ConvBN(c4, ch, 1)
        self.p3 = ConvBN(c3, ch, 1)
        self.mix = ConvBN(2 * ch, ch, 1)
        self.c1 = RepConv(ch, ch, 1)
        self.c2 = RepConv(ch, ch, 2)
        self.out = nn.Conv2d(ch, 2 * FIELD_STRIDE ** 2, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.constant_(self.out.bias, -torch.log(torch.tensor((1 - prior) / prior)).item())

    def forward(self, f4, p3):
        up = F.interpolate(self.p3(p3), size=f4.shape[-2:], mode="nearest")
        h = self.c2(self.c1(self.mix(torch.cat([self.p4(f4), up], 1))))
        return F.pixel_shuffle(self.out(h), FIELD_STRIDE).float()          # (B, 2, H / 2, W / 2) logits


class BoundarySnap(nn.Module):
    """Moves each end (along the box axis) and each long side (across it) of boxes (B, K, 5) = (cx, cy, L, T, theta)
    px to the soft-argmax of the field profile sampled around it: N_LINES parallel profiles of N_T samples within the
    search radius, averaged, softmax(logits / tau) -> expected offset t*. The move is gain x conf x t*, conf = peak
    probability of the profile; gain (ends, sides) = GAIN_SCALE x parameter, zero at init -> the identity."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(2))          # ends, sides
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.register_buffer("t", torch.linspace(-1.0, 1.0, N_T), persistent=False)
        self.register_buffer("lines", torch.linspace(-0.25, 0.25, N_LINES), persistent=False)

    def profile(self, field, c, ax, nx, at, across_unit, radius, chan):
        """Sample `field` channel `chan` (B, K) on N_LINES lines parallel to the side at axis offset `at` (B, K) px
        (direction ax), the lines spread across by `across_unit` (B, K) px (direction nx); returns (t (B, K, N_T) px,
        logits (B, K, N_T))."""
        B, K = at.shape
        t = self.t * radius[..., None]                                            # (B, K, N_T)
        pos =(c[..., None, None, :] + (at[..., None, None] + t[:, :, None, :])[..., None] * ax[:, :, None, None, :]
               + (self.lines[None, None, :, None] * across_unit[..., None, None])[..., None] * nx[:, :, None, None, :])
        s = _sample(field, pos.reshape(B, -1, 2), FIELD_STRIDE).reshape(B, K, N_LINES, N_T, 2)
        s = s.mean(2)                                                             # (B, K, N_T, 2)
        logit = torch.where(chan[..., None], s[..., 1], s[..., 0])
        return t, logit

    def forward(self, field, b):
        bd = b.detach().float()
        c, L, T, th = bd[..., :2], bd[..., 2].abs().clamp(min=0.5), bd[..., 3].abs().clamp(min=0.5), bd[..., 4]
        ax = torch.stack([th.cos(), th.sin()], -1)
        nx = torch.stack([-th.sin(), th.cos()], -1)
        tau = self.log_tau.exp()
        long_axis = L >= T                       # the box's own axis runs along its long side: ends = channel 0
        r_end = (END_R[1] * L).clamp(END_R[0], END_R[2])
        r_side = (SIDE_R[1] * T).clamp(SIDE_R[0], SIDE_R[2])
        moves = []
        for sign, length, width, r, along_ax, across_ax, is_end in ((1, L, T, r_end, ax, nx, True),
                                                                    (-1, L, T, r_end, ax, nx, True),
                                                                    (1, T, L, r_side, nx, ax, False),
                                                                    (-1, T, L, r_side, nx, ax, False)):
            chan = ~long_axis if is_end else long_axis          # ends of a long box: channel 0; its sides: channel 1
            t, logit = self.profile(field, c, along_ax, across_ax, sign * length / 2, width, r, chan)
            w = torch.softmax(logit / tau, -1)
            conf = logit.amax(-1).sigmoid()
            moves.append((w * t).sum(-1) * conf)
        g_end, g_side = (GAIN_SCALE * self.gain).unbind()
        m_hi, m_lo, s_hi, s_lo = moves                         # outward along +ax / -ax, +nx / -nx (signed offsets)
        u_hi, u_lo = L / 2 + g_end * m_hi, -L / 2 + g_end * m_lo
        v_hi, v_lo = T / 2 + g_side * s_hi, -T / 2 + g_side * s_lo
        length = (u_hi - u_lo).clamp(min=0.5)
        thick = (v_hi - v_lo).clamp(min=0.5)
        du, dv = (u_hi + u_lo) / 2, (v_hi + v_lo) / 2
        out_c = b[..., :2] + du[..., None] * ax + dv[..., None] * nx
        return torch.cat([out_c, (b[..., 2] + (length - L))[..., None], (b[..., 3] + (thick - T))[..., None],
                          b[..., 4:5]], -1)


class OpenOBB17(OpenOBB12):
    """OpenOBB12 (pretrained HGNetv2 + bounded LSK + hybrid encoder, dual dense head, calibrated head / NMS) + the
    boundary field (stride-4 trunk map + P3) + boundary snapping of the `ref_k` best candidates before NMS."""
    arch = "v17"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 final_nms=None, box="dfl", ref_k=REF_K, **_):
        off_names = [s_.strip() for s_ in (off.split(",") if isinstance(off, str) else (off or [])) if str(s_).strip()]
        self.no_field = "field" in off_names
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou,
                         off=[n for n in off_names if n != "field"] or None, final_nms=final_nms, box=box)
        self.field = None if self.no_field else BoundaryField(self.backbone._out_channels[0], self.encoder.hidden_dim)
        self.snap = None if self.no_field else BoundarySnap()
        self.ref_k, self.ref_k_train = int(ref_k), REF_K_TRAIN

    _with_jitter = OpenOBB14._with_jitter

    def snap_topk(self, field, cls, boxes, k):
        """Snap the k best-scoring boxes (B, A, 5) px in place -> (boxes, idx (B, k))."""
        k = min(int(k), cls.shape[1])
        idx = cls.detach().sigmoid().amax(-1).topk(k, dim=1).indices
        cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
        return boxes.scatter(1, idx[..., None].expand(-1, -1, 5), self.snap(field, cand).to(boxes.dtype)), idx

    def forward(self, x, targets=None, ctx=None):
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        field = None if self.field is None else self.field(f4, feats[0])
        if self.training:
            cls, reg = self.run_head(feats, False)
            c1, r1 = self.run_head(feats, True)
            # feats / input: for losses that compare the neck with a teacher (OpenOBB18); unused otherwise
            out = {"cls": cls, "reg": reg, "points": pts, "strides": st, "o2o_cls": c1, "o2o_reg": r1,
                   "feats": feats, "input": x}
            if field is not None:
                out["field"] = field
                boxes = decode(reg.detach(), pts, st)
                k = min(self.ref_k_train, boxes.shape[1])
                idx = cls.detach().sigmoid().amax(-1).topk(k, dim=1).indices
                cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
                cand, valid = self._with_jitter(cand, targets)
                out.update(ref_boxes=self.snap(field, cand), ref_cand=cand, ref_valid=valid)
            return out
        o2o = self.infer_head != "o2m"
        cls, reg = self.run_head(feats, o2o)
        boxes = decode(reg, pts, st)
        if field is not None and self.ref_k > 0:
            boxes, _ = self.snap_topk(field, cls, boxes, self.ref_k)
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
