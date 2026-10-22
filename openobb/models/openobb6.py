"""OpenOBB6: a general oriented detector designed from the measurements in research/ARCH_ANALYSIS.md
(not assembled from an ImageNet backbone + a DETR encoder). OpenOBB's own design and code.

Principles (ARCH_ANALYSIS section 8) and where they live:
  N1 compute at high resolution: widths grow slowly (stride 4 / 8 / 16 / 32), the stride-32 stage is thin;
     OpenOBB3's HGNetv2 put 2048 channels at stride 32 and that path contributed nothing (F37).
  N2 + N3 wide context early and cheap: every unit mixes a 3x3 depthwise with a separable large kernel
     (depthwise 1 x k then k x 1: a k x k field for 2k weights, any orientation), stages at stride 16 / 32 add
     row / column pooling (context along the whole image), one small attention layer runs at stride 32.
  N4 numerically stable: trainable BatchNorm everywhere (folded into the convolutions for inference, `fuse()`),
     residual branches start at zero, additive context (no unbounded x * (1 + m) gating).
  N5 downsampling keeps peaks: a max-pool path runs beside every strided convolution; stride-4 detail enters P3
     through space-to-depth (no extra stride-4 head).
  N6 a heavier, context-aware class branch (a large-kernel unit inside it); boxes keep OpenOBB3's anisotropic
     oriented regression (end distances + thickness as distributions), which measured well (F37).
  N9 only standard convolutions, depthwise, 1x1, max-pool and one attention layer: exports to ONNX / TensorRT.

Output / loss interface = OpenOBB3 (dense oriented head, NMS on ProbIoU), so openobb.models.openobb3_loss trains it.
Initialisation: no ImageNet / COCO weights. Pretrain the backbone by feature distillation from DINOv2
(`openobb pretrain`, openobb/pretrain.py), then train detection (docs/OPENOBB6.md).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb3 import REG_CH, decode, fast_nms

# width multiplier, depth multiplier (like the n / s / m / l / x scales of one-stage detectors)
SCALES = {"n": (0.25, 0.34), "s": (0.5, 0.34), "m": (0.75, 0.67), "l": (1.0, 1.0), "x": (1.5, 1.0)}
WIDTHS = (96, 192, 384, 512)            # stage widths at stride 4 / 8 / 16 / 32 for width 1.0
DEPTHS = (2, 6, 6, 2)                   # large-kernel units per stage for depth 1.0
KERNELS = (7, 11, 11, 9)                # separable kernel length per stage (field 28 / 88 / 176 / 288 px)
NECK = (256, 384, 512)                  # P3 / P4 / P5 widths of the neck for width 1.0
# parts that can be switched off one at a time to measure what each contributes (docs/OPENOBB6.md, ablation)
PARTS = ("strip", "axis", "attn", "spd", "maxpool", "cls_ctx")


def parse_off(off):
    """'strip,attn' / ['strip'] / None -> frozenset of switched-off parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise ValueError(f"unknown OpenOBB6 part(s) {bad}; parts: {', '.join(PARTS)}")
    return frozenset(names)


def _ch(c, w, div=16):
    return max(div, int(round(c * w / div)) * div)


class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, zero=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        if zero:                                    # residual branch starts at zero
            nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StripMix(nn.Module):
    """x + MLP(dw3x3(x) + dw(k x 1)(dw(1 x k)(x))): local detail + a separable k x k field, then an inverted
    bottleneck (expand 2x). The residual branch starts at zero."""

    def __init__(self, c, k=11, strip=True):
        super().__init__()
        self.local = ConvBN(c, c, 3, g=c, act=False)
        self.strip = strip
        if strip:
            self.h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
            self.v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
            self.bn = nn.BatchNorm2d(c)
        self.pw1 = ConvBN(c, 2 * c, 1)
        self.pw2 = ConvBN(2 * c, c, 1, act=False, zero=True)

    def forward(self, x):
        t = self.local(x) + self.bn(self.v(self.h(x))) if self.strip else self.local(x)
        return x + self.pw2(self.pw1(t))


class CSPStage(nn.Module):
    """1x1 -> split in halves; one half passes through n StripMix units; every intermediate output is concatenated
    and fused by a 1x1 (cross-stage partial + one-shot aggregation: many gradient paths, half the channels pass
    untouched)."""

    def __init__(self, c1, c2, n=1, k=11, strip=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = ConvBN(c1, 2 * self.c, 1)
        self.m = nn.ModuleList(StripMix(self.c, k, strip) for _ in range(n))
        self.cv2 = ConvBN((2 + n) * self.c, c2, 1)

    def forward(self, x):
        kept, path = torch.tensor_split(self.cv1(x), 2, dim=1)     # one half bypasses, the other goes through the units
        outs = [kept, path]
        for unit in self.m:
            path = unit(path)
            outs.append(path)                                      # every stage output reaches the fusing 1x1
        return self.cv2(torch.cat(outs, dim=1))


class AxisContext(nn.Module):
    """Context along the whole row and the whole column of every location (mean pooling), added back through a
    zero-initialised projection (bounded, additive)."""

    def __init__(self, c):
        super().__init__()
        self.row = ConvBN(c, c, 1, act=False)
        self.col = ConvBN(c, c, 1, act=False)
        self.out = ConvBN(c, c, 1, act=False, zero=True)

    def forward(self, x):
        ctx = self.row(x.mean(3, keepdim=True)) + self.col(x.mean(2, keepdim=True))
        return x + self.out(F.silu(ctx))


class GlobalAttention(nn.Module):
    """One pre-norm transformer layer over the stride-32 map; position from a depthwise 3x3 (works at any size)."""

    def __init__(self, c, heads=8):
        super().__init__()
        self.pos = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.g1 = nn.Parameter(torch.zeros(c))
        self.g2 = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        h = self.n1((x + self.pos(x)).flatten(2).transpose(1, 2))     # position enters the attention only
        t = t + self.g1 * self.attn(h, h, h, need_weights=False)[0]
        t = t + self.g2 * self.ffn(self.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class Down(nn.Module):
    """Stride 2 with a max-pool path (keeps thin, bright structures) beside a depthwise strided convolution."""

    def __init__(self, c1, c2, maxpool=True):
        super().__init__()
        self.maxpool = maxpool
        self.dw = ConvBN(c1, c1, 3, 2, g=c1, act=False)
        self.cv = ConvBN((2 if maxpool else 1) * c1, c2, 1)

    def forward(self, x):
        if not self.maxpool:
            return self.cv(self.dw(x))
        return self.cv(torch.cat([F.max_pool2d(x, 2, 2, ceil_mode=True), self.dw(x)], 1))


class Stem(nn.Module):
    def __init__(self, c, maxpool=True):
        super().__init__()
        self.maxpool = maxpool
        self.s1 = ConvBN(3, c // 2, 3, 2)
        self.s2 = ConvBN(c // 2, c, 3, 2)
        self.fuse = ConvBN(c + (c // 2 if maxpool else 0), c, 1)

    def forward(self, x):
        x = self.s1(x)
        if not self.maxpool:
            return self.fuse(self.s2(x))
        return self.fuse(torch.cat([self.s2(x), F.max_pool2d(x, 2, 2, ceil_mode=True)], 1))


class Backbone(nn.Module):
    """-> feature maps at stride 4, 8, 16, 32."""

    def __init__(self, w, d, off=frozenset()):
        super().__init__()
        c = [_ch(v, w) for v in WIDTHS]
        n = [max(1, round(v * d)) for v in DEPTHS]
        st, mp = "strip" not in off, "maxpool" not in off
        self.channels = c
        self.stem = Stem(c[0], mp)
        self.s4 = CSPStage(c[0], c[0], n[0], KERNELS[0], st)
        self.d8, self.s8 = Down(c[0], c[1], mp), CSPStage(c[1], c[1], n[1], KERNELS[1], st)
        self.d16, self.s16 = Down(c[1], c[2], mp), CSPStage(c[2], c[2], n[2], KERNELS[2], st)
        self.ctx16 = nn.Identity() if "axis" in off else AxisContext(c[2])
        self.d32, self.s32 = Down(c[2], c[3], mp), CSPStage(c[3], c[3], n[3], KERNELS[3], st)
        self.ctx32 = nn.Identity() if "axis" in off else AxisContext(c[3])
        self.attn = nn.Identity() if "attn" in off else GlobalAttention(c[3], heads=8)

    def forward(self, x):
        f4 = self.s4(self.stem(x))
        f8 = self.s8(self.d8(f4))
        f16 = self.ctx16(self.s16(self.d16(f8)))
        f32 = self.attn(self.ctx32(self.s32(self.d32(f16))))
        return f4, f8, f16, f32


class Neck(nn.Module):
    """Top-down then bottom-up fusion (P3-P5); stride-4 detail joins P3 by space-to-depth (lossless)."""

    def __init__(self, cb, w, d, off=frozenset()):
        super().__init__()
        p3, p4, p5 = (_ch(v, w) for v in NECK)
        n = max(1, round(2 * d))
        st, mp = "strip" not in off, "maxpool" not in off
        self.channels = (p3, p4, p5)
        self.lat5 = ConvBN(cb[3], p4, 1)
        self.td4 = CSPStage(cb[2] + p4, p4, n, 11, st)
        self.lat4 = ConvBN(p4, p3, 1)
        self.spd = None if "spd" in off else ConvBN(4 * cb[0], p3 // 2, 1)
        self.td3 = CSPStage(cb[1] + p3 + (0 if self.spd is None else p3 // 2), p3, n, 11, st)
        self.dn3 = Down(p3, p3, mp)
        self.bu4 = CSPStage(p3 + p4, p4, n, 11, st)
        self.dn4 = Down(p4, p4, mp)
        self.bu5 = CSPStage(p4 + cb[3], p5, n, 9, st)

    def forward(self, f):
        f4, f8, f16, f32 = f
        t5 = self.lat5(f32)
        n4 = self.td4(torch.cat([f16, F.interpolate(t5, size=f16.shape[-2:], mode="nearest")], 1))
        t4 = self.lat4(n4)
        parts = [f8, F.interpolate(t4, size=f8.shape[-2:], mode="nearest")]
        if self.spd is not None:
            parts.append(self.spd(F.pixel_unshuffle(f4[..., :2 * f8.shape[-2], :2 * f8.shape[-1]], 2)))
        n3 = self.td3(torch.cat(parts, 1))
        o4 = self.bu4(torch.cat([self.dn3(n3), n4], 1))
        o5 = self.bu5(torch.cat([self.dn4(o4), f32], 1))
        return [n3, o4, o5]


class Head(nn.Module):
    """Per level: oriented regression (OpenOBB3 layout: angle, across offset, two end distributions, thickness
    distribution) and a context-aware class branch (3x3, a large-kernel unit, 3x3)."""

    def __init__(self, ch, nc, w, prior=0.01, off=frozenset()):
        super().__init__()
        cr, cc = _ch(128, w, 32), _ch(160, w, 32)
        mid = (lambda: nn.Identity()) if "cls_ctx" in off else (lambda: StripMix(cc, 11, "strip" not in off))  # noqa: E731
        self.reg = nn.ModuleList(nn.Sequential(ConvBN(c, cr, 3), ConvBN(cr, cr, 3)) for c in ch)
        self.reg_out = nn.ModuleList(nn.Conv2d(cr, REG_CH, 1) for _ in ch)
        self.cls = nn.ModuleList(nn.Sequential(ConvBN(c, cc, 3), mid(), ConvBN(cc, cc, 3)) for c in ch)
        self.cls_out = nn.ModuleList(nn.Conv2d(cc, nc, 1) for _ in ch)
        for m in self.cls_out:
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, -math.log((1 - prior) / prior))
        for m in self.reg_out:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feats):
        cls = [o(m(f)).flatten(2) for f, m, o in zip(feats, self.cls, self.cls_out)]
        reg = [o(m(f)).flatten(2) for f, m, o in zip(feats, self.reg, self.reg_out)]
        return torch.cat(cls, 2).transpose(1, 2).float(), torch.cat(reg, 2).transpose(1, 2).float()


class OpenOBB6(nn.Module):
    arch = "v6"

    def __init__(self, size="x", num_classes=15, img_size=1024, max_det=1000, nms_iou=0.7, off=None, **_):
        super().__init__()
        w, d = SCALES[size]
        self.off = parse_off(off)
        self.size, self.nc, self.img_size = size, num_classes, img_size
        self.max_det, self.nms_iou = max_det, nms_iou
        self.backbone = Backbone(w, d, self.off)
        self.neck = Neck(self.backbone.channels, w, d, self.off)
        self.head = Head(self.neck.channels, num_classes, w, off=self.off)
        self.strides = (8, 16, 32)
        self._grid = {}

    def anchors(self, feats):
        key = tuple(f.shape[-2:] for f in feats) + (feats[0].device,)
        if key not in self._grid:
            pts, st = [], []
            for f, s in zip(feats, self.strides):
                h, w = f.shape[-2:]
                ys, xs = torch.meshgrid(torch.arange(h, device=f.device, dtype=torch.float32),
                                        torch.arange(w, device=f.device, dtype=torch.float32), indexing="ij")
                pts.append(torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], -1).reshape(-1, 2))
                st.append(torch.full((h * w,), float(s), device=f.device))
            self._grid = {key: (torch.cat(pts), torch.cat(st))}
        return self._grid[key]

    def forward(self, x, targets=None, ctx=None):
        feats = self.neck(self.backbone(x))
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        if self.training:
            return {"cls": cls, "reg": reg, "points": pts, "strides": st}
        boxes = decode(reg, pts, st)
        logits, boxes = fast_nms(cls, boxes, self.max_det, conf=getattr(self, "nms_conf", 0.001), iou=self.nms_iou,
                                 min_side=getattr(self, "nms_min_side", 0.0))
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}

    @torch.no_grad()
    def fuse(self):
        """Fold every BatchNorm into the convolution before it (inference only; outputs unchanged)."""
        def fold(conv, bn):
            std = (bn.running_var + bn.eps).sqrt()
            w = conv.weight * (bn.weight / std).view(-1, 1, 1, 1)
            b = bn.bias - bn.running_mean * bn.weight / std
            if conv.bias is not None:
                b = b + conv.bias * bn.weight / std
            new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                            conv.dilation, conv.groups, bias=True).to(conv.weight.device)
            new.weight.copy_(w)
            new.bias.copy_(b)
            return new
        for m in self.modules():
            if isinstance(m, ConvBN) and isinstance(m.bn, nn.BatchNorm2d):
                m.conv, m.bn = fold(m.conv, m.bn), nn.Identity()
            elif isinstance(m, StripMix) and isinstance(getattr(m, "bn", None), nn.BatchNorm2d):
                m.v, m.bn = fold(m.v, m.bn), nn.Identity()
        return self
