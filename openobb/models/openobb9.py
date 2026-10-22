"""OpenOBB9: a new core made of re-parameterised dense 3x3 convolutions + OpenOBB8's head and relational refiner.
The core is pretrained by two-teacher distillation (openobb/pretrain.py). Design argument: docs/OPENOBB9.md.

Why each part (research/LEDGER.md, docs/OPENOBB9.md sections 1-5):
  * dense 3x3 convolutions only, no depthwise (L2, F39): on a GPU a dense 3x3 conv with >= 64 channels is compute-bound
    (~1000 FLOP / byte) while a depthwise conv (~4.5) or a lone BN / add (< 1) is memory-bound. OpenOBB6 (depthwise)
    did 504 GFLOPs in 25.7 ms, OpenOBB3 (909 GFLOPs) 23.1 ms.
  * RepConv (RepVGG idea, own code): 3x3 + 1x1 + identity, each with a trainable BN, summed, SiLU. Linear, so fuse()
    merges it exactly into ONE 3x3 conv with bias: many gradient paths in training, one compute-bound layer at inference.
  * RepCSP (CSP / ELAN idea): half the channels pass untouched, the other half runs n RepConv, every intermediate output
    is concatenated and fused by a 1x1 (many gradient paths, cheap).
  * context without depthwise (L4: class evidence 40-170 px away): dilations 1, 2, 1, 3 in the stride-8 / 16 stages,
    SPPF (max-pool 5, three times) and one small attention layer at stride 32 (zero-initialised residual scales).
  * peaks survive downsampling (N5): a max-pool path beside every strided conv; stride-4 detail joins P3 through
    space-to-depth + 1x1 (lossless).
  * stability (L3, F37): BN trained everywhere and folded at inference; no unbounded x * (1 + m) gate; the distillation
    loss reaches P3, P4 and P5 (every neck output has `hid` channels, as OpenOBB8's head expects).
  * head unchanged: OpenOBB7's ContextOrientedHead + OpenOBB8's oriented relational refiner, so v9 vs v8 compares
    the core only.

Stable interface (used by openobb/pretrain.py and the trainer; do not change):
    OpenOBB9(size="x", num_classes=15, img_size=1024, max_det=1000, nms_iou=0.7, off=None, off8=None, rel_k=600)
    .backbone, .neck            the core (pretrained by distillation: state-dict keys "backbone.*", "neck.*")
    .c4 (int)                   channels of the stride-4 backbone map
    .hid (int)                  channels of every neck output P3 / P4 / P5
    .features(x) -> (f4, [P3, P4, P5])   strides 4 / 8 / 16 / 32
    forward(x, targets=None, ctx=None): training -> dict for openobb.models.openobb8_loss.OpenOBB8Loss
                                        eval -> {"pred_logits", "pred_boxes"} (OpenOBB3 format)
    .fuse() -> self             inference only (re-parameterisation + BatchNorm folding)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb3 import OpenOBB3
from .openobb6 import ConvBN, GlobalAttention, Stem
from .openobb7 import ContextOrientedHead, OpenOBB7, fold_bn, fuse_model, space_to_depth
from .openobb8 import SPREAD_IOU, OpenOBB8, OrientedRelationalRefiner, parse_off8

PARTS9 = ("rep", "dilate", "attn", "spd", "maxpool")
# width multiplier, depth multiplier, channel rounding
SCALES9 = {"s": (0.5, 0.5, 32), "m": (0.75, 0.75, 32), "l": (1.0, 1.0, 32), "x": (1.5, 1.0, 64)}
WIDTHS9 = (112, 224, 384, 384)          # backbone widths at stride 4 / 8 / 16 / 32 for width 1.0
DEPTHS9 = (1, 4, 4, 2)                  # RepConv per backbone stage for depth 1.0
HID9 = 256                              # neck width (P3 / P4 / P5) for width 1.0
NECK_DEPTH9 = 3                         # RepConv per neck block for depth 1.0
DILATIONS = (1, 2, 1, 3)                # cycled in the stride-8 / 16 stages


def parse_off9(off):
    """'rep,attn' / ['rep'] / None -> frozenset of switched-off OpenOBB9 core parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS9]
    if bad:
        raise ValueError(f"unknown OpenOBB9 part(s) {bad}; parts: {', '.join(PARTS9)}")
    return frozenset(names)


def _ch(c, w, div):
    return max(div, int(round(c * w / div)) * div)


class RepConv(nn.Module):
    """SiLU(BN(3x3, dilation d)(x) + BN(1x1)(x) + BN(x)), the identity branch only when c1 == c2.
    rep=False: a plain 3x3 conv + BN. fuse() merges the branches exactly into one 3x3 conv with bias."""

    def __init__(self, c1, c2, d=1, rep=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 1, d, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False) if rep else None
        self.bn1 = nn.BatchNorm2d(c2) if rep else None
        self.bn_id = nn.BatchNorm2d(c2) if rep and c1 == c2 else None
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.bn(self.conv(x))
        if self.conv1 is not None:
            y = y + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return self.act(y)

    @torch.no_grad()
    def fuse(self):
        """Branches -> one 3x3 conv with bias (the 1x1 and the identity sit at the centre tap, any dilation)."""
        if not isinstance(self.bn, nn.BatchNorm2d):
            return
        new = fold_bn(self.conv, self.bn)
        if self.conv1 is not None:
            one = fold_bn(self.conv1, self.bn1)
            new.weight[:, :, 1, 1] += one.weight[:, :, 0, 0]
            new.bias += one.bias
        if self.bn_id is not None:
            eye = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, 1, bias=False).to(new.weight)
            eye.weight.copy_(torch.eye(self.conv.out_channels, dtype=new.weight.dtype)[..., None, None])
            ident = fold_bn(eye, self.bn_id)
            new.weight[:, :, 1, 1] += ident.weight[:, :, 0, 0]
            new.bias += ident.bias
        self.conv, self.bn, self.conv1, self.bn1, self.bn_id = new, nn.Identity(), None, None, None


class RepCSP(nn.Module):
    """1x1 to 2h -> halves; one half through n RepConv (dilations cycled); concat every intermediate output -> 1x1."""

    def __init__(self, c1, c2, n=1, dil=(1,), rep=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = ConvBN(c1, 2 * self.c, 1)
        self.m = nn.ModuleList(RepConv(self.c, self.c, dil[i % len(dil)], rep) for i in range(n))
        self.cv2 = ConvBN((2 + n) * self.c, c2, 1)

    def forward(self, x):
        kept, path = torch.tensor_split(self.cv1(x), 2, dim=1)     # one half bypasses, the other goes through the units
        outs = [kept, path]
        for unit in self.m:
            path = unit(path)
            outs.append(path)                                      # every stage output reaches the fusing 1x1
        return self.cv2(torch.cat(outs, dim=1))


class Down(nn.Module):
    """Stride 2: [dense 3x3 s2 to c2 / 2 || max-pool 2] -> 1x1 to c2 (the max-pool path keeps thin, bright peaks)."""

    def __init__(self, c1, c2, maxpool=True):
        super().__init__()
        self.maxpool = maxpool
        self.cv = ConvBN(c1, c2 // 2, 3, 2)
        self.out = ConvBN(c2 // 2 + (c1 if maxpool else 0), c2, 1)

    def forward(self, x):
        y = self.cv(x)
        if self.maxpool:
            y = torch.cat([y, F.max_pool2d(x, 2, 2, ceil_mode=True)[..., :y.shape[-2], :y.shape[-1]]], 1)
        return self.out(y)


class SPPF(nn.Module):
    """1x1 to c / 2, max-pool 5 three times (fields 5 / 9 / 13), concat, 1x1 back to c."""

    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBN(c, c // 2, 1)
        self.cv2 = ConvBN(2 * c, c, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(F.max_pool2d(y[-1], 5, 1, 2))
        return self.cv2(torch.cat(y, 1))


class Backbone9(nn.Module):
    """-> maps at stride 4, 8, 16, 32 (stem, RepCSP stages, SPPF + one attention layer at stride 32)."""

    def __init__(self, w, d, div, off=frozenset()):
        super().__init__()
        c = [_ch(v, w, div) for v in WIDTHS9]
        n = [max(1, round(v * d)) for v in DEPTHS9]
        rep, mp = "rep" not in off, "maxpool" not in off
        dil = (1,) if "dilate" in off else DILATIONS
        self.channels = c
        self.stem = Stem(c[0], mp)
        self.s4 = RepCSP(c[0], c[0], n[0], rep=rep)
        self.d8, self.s8 = Down(c[0], c[1], mp), RepCSP(c[1], c[1], n[1], dil, rep)
        self.d16, self.s16 = Down(c[1], c[2], mp), RepCSP(c[2], c[2], n[2], dil, rep)
        self.d32, self.s32 = Down(c[2], c[3], mp), RepCSP(c[3], c[3], n[3], rep=rep)
        self.sppf = SPPF(c[3])
        self.attn = nn.Identity() if "attn" in off else GlobalAttention(c[3], heads=8)

    def forward(self, x):
        f4 = self.s4(self.stem(x))
        f8 = self.s8(self.d8(f4))
        f16 = self.s16(self.d16(f8))
        f32 = self.attn(self.sppf(self.s32(self.d32(f16))))
        return f4, f8, f16, f32


class Neck9(nn.Module):
    """Bidirectional PAN of RepCSP blocks; stride-4 detail joins P3 by space-to-depth; every output has `hid` channels."""

    def __init__(self, cb, hid, n, off=frozenset()):
        super().__init__()
        rep, mp = "rep" not in off, "maxpool" not in off
        self.lat5 = ConvBN(cb[3], hid, 1)
        self.td4 = RepCSP(cb[2] + hid, hid, n, rep=rep)
        self.spd = None if "spd" in off else ConvBN(4 * cb[0], hid // 2, 1)
        self.td3 = RepCSP(cb[1] + hid + (0 if self.spd is None else hid // 2), hid, n, rep=rep)
        self.dn3, self.bu4 = Down(hid, hid, mp), RepCSP(2 * hid, hid, n, rep=rep)
        self.dn4, self.bu5 = Down(hid, hid, mp), RepCSP(2 * hid, hid, n, rep=rep)

    def forward(self, f):
        f4, f8, f16, f32 = f
        t5 = self.lat5(f32)
        n4 = self.td4(torch.cat([f16, F.interpolate(t5, size=f16.shape[-2:], mode="nearest")], 1))
        parts = [f8, F.interpolate(n4, size=f8.shape[-2:], mode="nearest")]
        if self.spd is not None:
            parts.append(self.spd(space_to_depth(f4, f8.shape[-2:])))
        p3 = self.td3(torch.cat(parts, 1))
        p4 = self.bu4(torch.cat([self.dn3(p3), n4], 1))
        p5 = self.bu5(torch.cat([self.dn4(p4), t5], 1))
        return [p3, p4, p5]


class OpenOBB9(nn.Module):
    arch = "v9"
    anchors = OpenOBB3.anchors
    _with_jitter = OpenOBB8._with_jitter

    def __init__(self, size="x", num_classes=15, img_size=1024, max_det=1000, nms_iou=0.7, off=None, off8=None,
                 rel_k=600, **_):
        super().__init__()
        w, d, div = SCALES9[size]
        self.size, self.nc, self.img_size, self.max_det, self.nms_iou = size, num_classes, img_size, max_det, nms_iou
        self.off, self.off8 = parse_off9(off), parse_off8(off8)
        self.rel_k = rel_k
        self.backbone = Backbone9(w, d, div, self.off)
        self.c4, self.hid = self.backbone.channels[0], _ch(HID9, w, div)
        self.neck = Neck9(self.backbone.channels, self.hid, max(1, round(NECK_DEPTH9 * d)), self.off)
        self.head = ContextOrientedHead([self.hid] * 3, num_classes)
        self.strides = (8, 16, 32)
        self._grid = {}
        self.spread_iou = SPREAD_IOU                   # None: refined logits reach the candidates only
        self.refiner = None
        if "refine" not in self.off8:
            self.refiner = OrientedRelationalRefiner(self.hid, self.c4, num_classes, k=rel_k, off=self.off8)

    def features(self, x):
        """-> (stride-4 backbone map, [P3, P4, P5] neck outputs)."""
        f = self.backbone(x)
        return f[0], self.neck(f)

    encode = features                               # the name OpenOBB7 / OpenOBB8 forward read the core by

    def forward(self, x, targets=None, ctx=None):
        """OpenOBB8's forward (candidates, refiner, spreading, NMS) on this core; OpenOBB7's when the refiner is off."""
        return (OpenOBB7 if self.refiner is None else OpenOBB8).forward(self, x, targets, ctx)

    @torch.no_grad()
    def fuse(self):
        """Inference only: merge every RepConv into one 3x3 conv, then fold every remaining BatchNorm."""
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse()
        return fuse_model(self)
