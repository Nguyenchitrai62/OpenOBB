"""OpenOBB7: OpenOBB3 (pretrained HGNetv2 + hybrid encoder + dense oriented head) with the defects measured in
research/ARCH_ANALYSIS.md (F37) repaired. Every repair starts as the identity, so a model initialised from D-FINE COCO
weights or from a fresh OpenOBB3 gives OpenOBB3's output at step 0, and each one can be switched off (`off`).

Repairs (docs/OPENOBB7.md):
  guard      parameter-free per-sample scale guard on the four backbone outputs: identity while the RMS is <= tau,
             else rescaled to RMS tau. F37: stage-4 output mean |x| 3566 (P3 / P4 ~0.5) with frozen BN, then the
             encoder squashed it to a constant (zeroing it changed no detection).
  lsk_bound  LSK adapters modulate with tanh: x + x * tanh(m) instead of x + x * m (F37: LSK at P5 amplified the
             exploded map to 1.3e10). Same parameters, identical at init (m = 0).
  spd        stride-4 detail into P3: space-to-depth of the stride-4 map, 1x1 conv + BN (zero gamma), added to the
             P3 backbone map before LSK (Wall_Color: junction 9.5 px, walls 3.7-5.6 px thick, all read at stride 8).
  cls_ctx    a residual large-kernel unit (OpenOBB6 StripMix, k = 11) inside the class tower of the head (F37: 75-93%
             of the class evidence lies outside the object's box; wall_300 vs junction is the main error).
`fuse()` folds every BatchNorm into its convolution for inference.

Stable interface (used by OpenOBB8, do not change):
    OpenOBB7(size, num_classes, img_size, lsk=True, max_det=1000, nms_iou=0.7, off=None)
    .encode(x) -> (f4, [P3, P4, P5]): f4 = stride-4 backbone map (guarded), P3-P5 = encoder outputs (hidden_dim)
    .head(feats) -> (cls (B, A, C), reg (B, A, REG_CH));  .anchors(feats) -> (points px, strides)
    forward(x, targets=None, ctx=None): training -> {"cls", "reg", "points", "strides"}; eval -> OpenOBB3 format
Parameter names of backbone / lsk / encoder / head = OpenOBB3's, so OpenOBB3 and D-FINE COCO weights load; the new
modules are `spd` and `head.ctx` (the guard and tanh have no parameters).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d
from .openobb1 import SelectiveKernel
from .openobb2 import Conv
from .openobb3 import OpenOBB3, OrientedHead, decode, fast_nms
from .openobb6 import ConvBN, StripMix

PARTS7 = ("guard", "lsk_bound", "spd", "cls_ctx")
GUARD_TAU = 4.0          # healthy maps have RMS ~0.5-1 (F37: P3 / P4 mean |x| 0.4-0.5); the dead P5 had 3566


def parse_off7(off):
    """'guard,spd' / ['spd'] / None -> frozenset of switched-off OpenOBB7 parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS7]
    if bad:
        raise ValueError(f"unknown OpenOBB7 part(s) {bad}; parts: {', '.join(PARTS7)}")
    return frozenset(names)


class ScaleGuard(nn.Module):
    """x * min(1, tau / rms(x)) per sample. Exactly the identity while rms <= tau; above it the map is rescaled to
    RMS tau (scale invariant, so the gradient cannot feed a growing magnitude). Statistics in float32 with a
    detached peak normalisation: safe for huge inputs and under bf16 / fp16 autocast."""

    def __init__(self, tau=GUARD_TAU):
        super().__init__()
        self.tau = float(tau)

    def forward(self, x):
        xf = x.float()
        dims = tuple(range(1, x.dim()))
        peak = xf.detach().abs().amax(dims, keepdim=True).clamp(min=1e-12)
        rms = peak * (xf / peak).square().mean(dims, keepdim=True).sqrt()
        return (xf / (rms / self.tau).clamp(min=1.0)).to(x.dtype)

    def extra_repr(self):
        return f"tau={self.tau}"


class BoundedSelectiveKernel(SelectiveKernel):
    """OpenOBB1's LSK adapter (same parameter names) with a bounded modulation: x + x * tanh(m). Identical at init
    (the output projection is zero) and ~identical while |m| is small; the gain can never exceed 2."""

    def forward(self, x):
        a = self.near(x)
        b = self.far(a)
        a, b = self.pn(a), self.pf(b)
        cat = torch.cat([a, b], 1)
        g = self.gate(torch.cat([cat.mean(1, keepdim=True), cat.amax(1, keepdim=True)], 1)).sigmoid()
        return x + x * torch.tanh(self.out(a * g[:, :1] + b * g[:, 1:]))


class ContextOrientedHead(OrientedHead):
    """OpenOBB3's head (same parameter names) + a residual large-kernel unit in each class tower, after its first two
    layers (depthwise 3x3, 1x1 to cc channels). The unit's residual branch starts at zero."""

    def __init__(self, ch, nc, prior=0.01, k=11):
        super().__init__(ch, nc, prior)
        cc = self.cls[0][1].conv.out_channels
        self.ctx = nn.ModuleList(StripMix(cc, k) for _ in ch)

    def forward(self, feats):
        cls, reg = [], []
        for f, mc, mr, cx in zip(feats, self.cls, self.reg, self.ctx):
            h = cx(mc[1](mc[0](f)))
            for layer in list(mc)[2:]:
                h = layer(h)
            cls.append(h.flatten(2))
            reg.append(mr(f).flatten(2))
        return torch.cat(cls, 2).transpose(1, 2).float(), torch.cat(reg, 2).transpose(1, 2).float()


def space_to_depth(f4, size):
    """(B, C, H4, W4) -> (B, 4C, H8, W8) for the stride-8 grid `size` (zero-pads an odd edge, lossless)."""
    h, w = 2 * size[0], 2 * size[1]
    if f4.shape[-2] != h or f4.shape[-1] != w:
        f4 = F.pad(f4[..., :h, :w], (0, max(0, w - f4.shape[-1]), 0, max(0, h - f4.shape[-2])))
    return F.pixel_unshuffle(f4, 2)


@torch.no_grad()
def fold_bn(conv, bn):
    """Conv2d + BatchNorm2d / FrozenBatchNorm2d (eval statistics) -> one Conv2d with bias."""
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    b = bn.bias - bn.running_mean * t
    if conv.bias is not None:
        b = b + conv.bias * t
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                    conv.groups, bias=True, padding_mode=conv.padding_mode)
    new = new.to(device=conv.weight.device, dtype=conv.weight.dtype)
    new.weight.copy_(conv.weight * t.view(-1, 1, 1, 1))
    new.bias.copy_(b)
    return new


def _is_bn(m):
    return isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d))


class OpenOBB7(OpenOBB3):
    arch = "v7"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None, **_):
        super().__init__(size, num_classes, img_size, lsk, max_det, nms_iou, backbone="hgnet")
        self.off = parse_off7(off)
        cin = self.encoder.in_channels
        self.guard = None if "guard" in self.off else ScaleGuard(GUARD_TAU)
        if self.lsk is not None and "lsk_bound" not in self.off:
            self.lsk = nn.ModuleList(BoundedSelectiveKernel(c) for c in cin)
        c4 = self.backbone._out_channels[0]
        self.spd = None if "spd" in self.off else ConvBN(4 * c4, cin[0], 1, act=False, zero=True)
        if "cls_ctx" not in self.off:
            self.head = ContextOrientedHead([self.encoder.hidden_dim] * 3, num_classes)

    def encode(self, x):
        """-> (stride-4 backbone map, [P3, P4, P5] encoder outputs)."""
        bb = self.backbone
        h = bb.stem(x)
        outs = []
        for stage in bb.stages:
            h = stage(h)                                    # the backbone's own chain is untouched
            outs.append(h if self.guard is None else self.guard(h))
        f4, feats = outs[0], outs[1:]
        if self.spd is not None:
            feats[0] = feats[0] + self.spd(space_to_depth(f4, feats[0].shape[-2:]))
        if self.lsk is not None:
            feats = [m(f) for m, f in zip(self.lsk, feats)]
        return f4, self.encoder(feats)

    def forward(self, x, targets=None, ctx=None):
        _, feats = self.encode(x)
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        if self.training:
            return {"cls": cls, "reg": reg, "points": pts, "strides": st}
        boxes = decode(reg, pts, st)
        logits, boxes = fast_nms(cls, boxes, self.max_det, iou=self.nms_iou)
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}

    @torch.no_grad()
    def fuse(self):
        """Inference only: fold every BatchNorm (trainable or frozen) into the convolution before it and merge the
        encoder's re-parameterisable VGG blocks. Outputs are unchanged (up to float rounding)."""
        from .hgnetv2 import ConvBNAct
        from .hybrid_encoder import ConvNormLayer, ConvNormLayer_fuse, VGGBlock
        for m in list(self.modules()):                      # first: VGG blocks read their branches' BN
            if isinstance(m, VGGBlock) and hasattr(m, "conv1"):
                m.convert_to_deploy()
        for m in list(self.modules()):
            if isinstance(m, ConvBNAct) and _is_bn(m.bn):  # act(bn(conv(x))) then lab; 'same' padding = (pad, conv)
                if isinstance(m.conv, nn.Sequential):
                    m.conv[1] = fold_bn(m.conv[1], m.bn)
                else:
                    m.conv = fold_bn(m.conv, m.bn)
                m.bn = nn.Identity()
            elif isinstance(m, (ConvNormLayer, ConvNormLayer_fuse)) and hasattr(m, "conv") and _is_bn(m.norm):
                m.conv, m.norm = fold_bn(m.conv, m.norm), nn.Identity()     # act(norm(conv(x)))
            elif isinstance(m, (Conv, ConvBN)) and _is_bn(m.bn):
                m.conv, m.bn = fold_bn(m.conv, m.bn), nn.Identity()
            elif isinstance(m, StripMix) and _is_bn(getattr(m, "bn", None)):
                m.v, m.bn = fold_bn(m.v, m.bn), nn.Identity()               # bn(v(h(x)))
        for p in self.encoder.input_proj:                   # Sequential(conv, norm)
            if _is_bn(p.norm):
                p.conv, p.norm = fold_bn(p.conv, p.norm), nn.Identity()
        return self
