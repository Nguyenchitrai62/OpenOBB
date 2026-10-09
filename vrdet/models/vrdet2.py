"""VRDet2: dense, NMS-free oriented detector for drawings (thin / long / small objects, small datasets).

VRDet's own implementation; the ideas follow published work, re-implemented from scratch:
  * dense anchor-free prediction at every feature location with task-aligned assignment (TOOD / YOLO family),
    one-to-many head for training signal + one-to-one head for NMS-free output (YOLOv10 / YOLO26);
  * long strip depthwise convolutions for elongated context (Strip R-CNN, LSKNet idea);
  * a P2 (stride 4) level through a two-way PAN, so 2-3 px lines and tiny junctions get their own cells;
  * one global self-attention block on P5 for page-level context (AIFI / C2PSA idea).
New in VRDet2 (segment-aware head): every oriented box is read as a segment along its long axis plus a thickness.
Each location predicts the axis angle, its own position along the segment (relative to the half length), the
segment length in log space (no length cap: a long wall is measured from any point on it), a signed offset across
the axis, and the thickness as a fine distribution (sub-stride precision: 1 px of thickness decides IoU for thin
objects). Thickness is learned independently of where the point sits, so points beside a 2 px line still work.

Sizes follow the usual depth / width multipliers; vrdet2x is the accuracy-first model.
Output (eval): {"pred_logits": (B, A, C), "pred_boxes": (B, A, 5) normalised (cx, cy, w, h, theta)}: one-to-one
classes + the shared segment regression, read by vrdet.models.vrdet.postprocess like the DETR outputs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# depth multiplier, width multiplier, max channels (accuracy-first: x is the reference)
SCALES = {"n": (0.33, 0.25, 1024), "s": (0.33, 0.50, 1024), "m": (0.67, 0.75, 768),
          "l": (1.00, 1.00, 512), "x": (1.00, 1.25, 512)}
REG_BINS = 32            # thickness distribution bins (bin width = stride / 2)


def _c(ch, w, cap):
    return int(math.ceil(min(ch, cap) * w / 16) * 16)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, k2=None):
        super().__init__()
        kk = (k, k2) if k2 is not None else (k, k)
        self.conv = nn.Conv2d(c1, c2, kk, s, (kk[0] // 2, kk[1] // 2), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, strip=0):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        # strip: long thin receptive field along both axes (elongated objects); else a plain 3x3
        self.cv2 = _StripMix(c, strip) if strip else Conv(c, c, 3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class _StripMix(nn.Module):
    """Depthwise 1xk and kx1 strips (both orientations) + 3x3 local, mixed by a pointwise conv."""

    def __init__(self, c, k):
        super().__init__()
        self.local = Conv(c, c, 3, g=c, act=False)
        self.h = Conv(c, c, 1, g=c, act=False, k2=k)
        self.v = Conv(c, c, k, g=c, act=False, k2=1)
        self.pw = Conv(c, c, 1)

    def forward(self, x):
        return self.pw(self.local(x) + self.h(x) + self.v(x))


class CSP(nn.Module):
    """Cross-stage partial block: half the channels pass through n bottlenecks, every stage output is kept."""

    def __init__(self, c1, c2, n=1, strip=0):
        super().__init__()
        h = c2 // 2
        self.cv1 = Conv(c1, 2 * h, 1)
        self.m = nn.ModuleList(Bottleneck(h, strip) for _ in range(n))
        self.cv2 = Conv((2 + n) * h, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.cv1 = Conv(c, c // 2, 1)
        self.cv2 = Conv(c // 2 * 4, c, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.pool(y[-1]))
        return self.cv2(torch.cat(y, 1))


class GlobalAttention(nn.Module):
    """One pre-norm transformer layer over the P5 tokens (2-D sin-cos positions), residual into the map."""

    def __init__(self, c, d=256, heads=8, ffn=1024):
        super().__init__()
        self.inp = nn.Conv2d(c, d, 1)
        self.out = nn.Conv2d(d, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.d = d

    def pos(self, h, w, device, dtype):
        y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                              torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        q = self.d // 4
        om = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
        ox, oy = x.flatten()[:, None] * om[None], y.flatten()[:, None] * om[None]
        return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None].to(dtype)

    def forward(self, x):
        b, _, h, w = x.shape
        t = self.inp(x).flatten(2).transpose(1, 2)
        p = self.pos(h, w, x.device, t.dtype)
        q = self.n1(t)
        t = t + self.attn(q + p, q + p, q, need_weights=False)[0]
        t = t + self.ffn(self.n2(t))
        return x + self.out(t.transpose(1, 2).reshape(b, self.d, h, w))


class Backbone(nn.Module):
    def __init__(self, d, w, cap, strip=(0, 13, 13, 9)):
        super().__init__()
        ch = [_c(c, w, cap) for c in (64, 128, 256, 512, 1024)]
        n = [max(1, round(k * d)) for k in (3, 6, 6, 3)]
        self.stem = Conv(3, ch[0], 3, 2)                                               # /2
        self.s2 = nn.Sequential(Conv(ch[0], ch[1], 3, 2), CSP(ch[1], ch[1], n[0], strip[0]))   # /4  -> C2
        self.s3 = nn.Sequential(Conv(ch[1], ch[2], 3, 2), CSP(ch[2], ch[2], n[1], strip[1]))   # /8  -> C3
        self.s4 = nn.Sequential(Conv(ch[2], ch[3], 3, 2), CSP(ch[3], ch[3], n[2], strip[2]))   # /16 -> C4
        self.s5 = nn.Sequential(Conv(ch[3], ch[4], 3, 2), CSP(ch[4], ch[4], n[3], strip[3]),
                                SPPF(ch[4]), GlobalAttention(ch[4]))                         # /32 -> C5
        self.channels = ch[1:]

    def forward(self, x):
        c2 = self.s2(self.stem(x))
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        return [c2, c3, c4, self.s5(c4)]


class PAN(nn.Module):
    """Top-down then bottom-up fusion over strides 4/8/16/32 (P2 is a real fused level, not a raw stage map)."""

    def __init__(self, ch, d, strip=13):
        super().__init__()
        c2, c3, c4, c5 = ch
        n = max(1, round(3 * d))
        self.td4 = CSP(c5 + c4, c4, n)
        self.td3 = CSP(c4 + c3, c3, n, strip)
        self.td2 = CSP(c3 + c2, c2, n, strip)
        self.dn3, self.bu3 = Conv(c2, c2, 3, 2), CSP(c2 + c3, c3, n, strip)
        self.dn4, self.bu4 = Conv(c3, c3, 3, 2), CSP(c3 + c4, c4, n)
        self.dn5, self.bu5 = Conv(c4, c4, 3, 2), CSP(c4 + c5, c5, n)

    def forward(self, f):
        c2, c3, c4, c5 = f
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="nearest")  # noqa: E731
        n4 = self.td4(torch.cat([up(c5, c4), c4], 1))
        n3 = self.td3(torch.cat([up(n4, c3), c3], 1))
        p2 = self.td2(torch.cat([up(n3, c2), c2], 1))
        p3 = self.bu3(torch.cat([self.dn3(p2), n3], 1))
        p4 = self.bu4(torch.cat([self.dn4(p3), n4], 1))
        p5 = self.bu5(torch.cat([self.dn5(p4), c5], 1))
        return [p2, p3, p4, p5]


class SegmentHead(nn.Module):
    """Per level: class logits + (angle, along position, log length, across offset, thickness distribution)."""

    def __init__(self, ch, nc, prior=0.01, reg=True):
        super().__init__()
        self.nc = nc
        cr = max(64, ch[0] // 2, REG_BINS)
        cc = max(ch[0], min(nc * 2, 128))
        self.reg = nn.ModuleList(nn.Sequential(Conv(c, cr, 3), Conv(cr, cr, 3), nn.Conv2d(cr, 4 + REG_BINS, 1))
                                 for c in ch) if reg else nn.ModuleList()
        self.cls = nn.ModuleList(nn.Sequential(Conv(c, c, 3, g=c), Conv(c, cc, 1), Conv(cc, cc, 3, g=cc),
                                               Conv(cc, cc, 1), nn.Conv2d(cc, nc, 1)) for c in ch)
        for m in self.cls:
            nn.init.constant_(m[-1].bias, -math.log((1 - prior) / prior))
        for m in self.reg:
            nn.init.zeros_(m[-1].bias)

    @staticmethod
    def _run(mods, feats):
        return torch.cat([m(f).flatten(2) for m, f in zip(mods, feats)], 2).transpose(1, 2).float()

    def forward(self, feats, cls=True, reg=True):
        return self._run(self.cls, feats) if cls else None, self._run(self.reg, feats) if reg else None


def decode(reg, pts, strides):
    """reg (B, A, 4 + R) -> boxes (B, A, 5) in pixels (cx, cy, w, h, theta); w = length along theta, h = thickness."""
    th = reg[..., 0]
    s = strides[None, :]
    length = reg[..., 2].clamp(-4.0, 7.0).exp() * s
    du = reg[..., 1] * length / 2                                     # this point's position along the segment
    dv = reg[..., 3] * s                                              # signed offset across the axis
    bins = torch.arange(REG_BINS, device=reg.device, dtype=reg.dtype)
    thick = (reg[..., 4:].softmax(-1) * bins).sum(-1) * (s / 2)       # thickness, bin width = stride / 2
    c, sn = th.cos(), th.sin()
    cx = pts[None, :, 0] - du * c - dv * sn
    cy = pts[None, :, 1] - du * sn + dv * c
    return torch.stack([cx, cy, length, thick, th], -1)


class VRDet2(nn.Module):
    arch = "v2"

    def __init__(self, size="x", num_classes=15, img_size=1024, **_):
        super().__init__()
        d, w, cap = SCALES[size]
        self.size, self.nc, self.img_size = size, num_classes, img_size
        self.strides = (4, 8, 16, 32)
        self.backbone = Backbone(d, w, cap)
        self.neck = PAN(self.backbone.channels, d)
        # one-to-many head: classes (training signal, many positives) + the segment regression used by both heads;
        # one-to-one head: classes only, on detached features (NMS-free output, one point per object). Every point
        # of a segment reads the same box, so the box branch is shared and learns from all positives.
        self.head_o2m = SegmentHead(self.backbone.channels, num_classes)
        self.head_o2o = SegmentHead(self.backbone.channels, num_classes, reg=False)
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
        cls2, _ = self.head_o2o([f.detach() for f in feats] if self.training else feats, reg=False)
        cls1, reg = self.head_o2m(feats, cls=self.training)
        if self.training:
            return {"o2m": (cls1, reg), "o2o": (cls2, reg.detach()), "points": pts, "strides": st}
        boxes = decode(reg, pts, st)
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2   # [-pi/2, pi/2)
        return {"pred_logits": cls2, "pred_boxes": boxes}
