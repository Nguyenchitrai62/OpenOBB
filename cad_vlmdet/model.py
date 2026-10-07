"""Hybrid DEEP VLM-style encoder + PAN neck + decoupled OBB head. MIT license.

Deep variant per user request: must be deep/complex enough for floorplan
wall-window-door-junction topology. Depths scale with variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(8, c // r), 1), nn.SiLU(inplace=True),
            nn.Conv2d(max(8, c // r), c, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class StripPoolGCAM(nn.Module):
    """Global Context Attention for anisotropic CAD structures (walls/corridors)."""

    def __init__(self, c):
        super().__init__()
        self.conv_h = nn.Conv2d(c, c, (1, 11), padding=(0, 5), groups=c, bias=False)
        self.conv_w = nn.Conv2d(c, c, (11, 1), padding=(5, 0), groups=c, bias=False)
        self.conv_d1 = nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False)
        self.fuse = nn.Conv2d(c * 3, c, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.conv_h(x)
        w = self.conv_w(x)
        d = self.conv_d1(x)
        f = self.act(self.bn(self.fuse(torch.cat([h, w, d], dim=1))))
        return x + f * self.gate(f)


class ConvNeXtDeepBlock(nn.Module):
    def __init__(self, c, drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.n = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv2d(c, c * 3, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(c * 3, c, 1)
        self.se = SEBlock(c)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        # layer scale (ConvNeXtV2 trick for very deep nets)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.5)

    def forward(self, x):
        h = self.pw2(self.act(self.pw1(self.n(self.dw(x)))))
        return x + self.drop(self.se(h) * self.gamma)


class GlobalReasonBlock(nn.Module):
    """Deep MHSA stack at P5 -> VLM-like long-range reasoning (4-6 layers for base/large)."""

    def __init__(self, c, heads=8, layers=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=c, nhead=heads, dim_feedforward=c * 3,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        out = self.norm(self.enc(seq))
        return out.transpose(1, 2).reshape(B, C, H, W) + x


class MaskedRefineModule(nn.Module):
    """MFRM-lite: refine low-level detail with high-level outer-contour context."""

    def __init__(self, c_low, c_high):
        super().__init__()
        self.proj_high = ConvBNAct(c_high, c_low, 1)
        self.dir_h = nn.Conv2d(c_low, c_low, (1, 5), padding=(0, 2), groups=c_low, bias=False)
        self.dir_v = nn.Conv2d(c_low, c_low, (5, 1), padding=(2, 0), groups=c_low, bias=False)
        self.fuse = nn.Sequential(ConvBNAct(c_low * 2, c_low, 1), ConvBNAct(c_low, c_low, 3))
        self.gate = nn.Sequential(nn.Conv2d(c_low, c_low, 1), nn.Sigmoid())

    def forward(self, low, high):
        h = self.proj_high(high)
        h = F.interpolate(h, size=low.shape[2:], mode="nearest")
        d = self.dir_h(low) + self.dir_v(low)
        f = self.fuse(torch.cat([low + d, h], dim=1))
        return low + f * self.gate(f)


class HybridVLMEencoder(nn.Module):
    """Deep hierarchical encoder -> [P2(s4), P3(s8), P4(s16), P5(s32)].
    depths=(d3,d4,d5), tr_layers = transformer depth at P5.
      nano  (2,2,2)+2L  : smoke test CPU
      small (3,4,3)+3L  : quick Kaggle test 800 imgs
      base  (4,8,6)+4L  : RECOMMENDED floorplan 16k (deep)
      large (6,12,8)+6L : max accuracy
    """

    def __init__(self, w=1.0, heads=8, depths=(4, 8, 6), tr_layers=4, drop=0.0,
                 use_gcam=True, use_reason=True):
        super().__init__()
        c0, c1, c2, c3, c4 = [int(c * w) for c in (32, 64, 128, 256, 512)]

        def _adj(c, h=8):
            return max(h, (c // h) * h)
        c1, c2, c3, c4 = _adj(c1), _adj(c2), _adj(c3), _adj(c4)
        self.c = (c1, c2, c3, c4)
        self.stem = nn.Sequential(
            ConvBNAct(3, c0, 3, 2),
            ConvBNAct(c0, c0, 3, 1),
            ConvBNAct(c0, c1, 3, 2),
        )
        g3 = StripPoolGCAM(c2) if use_gcam else nn.Identity()
        g4 = nn.Sequential(StripPoolGCAM(c3), StripPoolGCAM(c3)) if use_gcam else nn.Identity()
        g5 = StripPoolGCAM(c4) if use_gcam else nn.Identity()
        rs = GlobalReasonBlock(c4, heads=heads, layers=tr_layers) if use_reason else nn.Identity()
        self.down3 = ConvBNAct(c1, c2, 3, 2)
        self.blk3 = nn.Sequential(*[ConvNeXtDeepBlock(c2, drop) for _ in range(depths[0])], g3)
        self.down4 = ConvBNAct(c2, c3, 3, 2)
        self.blk4 = nn.Sequential(*[ConvNeXtDeepBlock(c3, drop) for _ in range(depths[1])], g4)
        self.down5 = ConvBNAct(c3, c4, 3, 2)
        self.blk5 = nn.Sequential(*[ConvNeXtDeepBlock(c4, drop) for _ in range(depths[2])], g5, rs)
        # cross-scale refinement (outer->inner like FloorPlanFormer)
        self.refine34 = MaskedRefineModule(c2, c3)
        self.refine45 = MaskedRefineModule(c3, c4)

    def forward(self, x):
        p2 = self.stem(x)
        p3 = self.blk3(self.down3(p2))
        p4 = self.blk4(self.down4(p3))
        p5 = self.blk5(self.down5(p4))
        # refine mid levels with deeper semantics (residual)
        p3 = self.refine34(p3, p4)
        p4 = self.refine45(p4, p5)
        return [p2, p3, p4, p5]


class RepCSPDeep(nn.Module):
    def __init__(self, c1, c2, n=2, e=0.5):
        super().__init__()
        ch = max(32, int(c2 * e))
        self.cv1 = ConvBNAct(c1, ch, 1)
        self.cv2 = ConvBNAct(c1, ch, 1)
        self.blocks = nn.Sequential(*[ConvBNAct(ch, ch, 3) for _ in range(n)])
        self.cv3 = ConvBNAct(ch * 2, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class PAN(nn.Module):
    def __init__(self, chs, repeat=2):
        super().__init__()
        c2, c3, c4, c5 = chs
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral54 = ConvBNAct(c5, c4, 1)
        self.csp54 = RepCSPDeep(c4 * 2, c4, n=repeat)
        self.lateral43 = ConvBNAct(c4, c3, 1)
        self.csp43 = RepCSPDeep(c3 * 2, c3, n=repeat)
        self.lateral32 = ConvBNAct(c3, c2, 1)
        self.csp32 = RepCSPDeep(c2 * 2, c2, n=repeat)
        self.down23 = ConvBNAct(c2, c3, 3, 2)
        self.csp32b = RepCSPDeep(c3 * 2, c3, n=repeat)
        self.down34 = ConvBNAct(c3, c4, 3, 2)
        self.csp34b = RepCSPDeep(c4 * 2, c4, n=repeat)
        self.down45 = ConvBNAct(c4, c5, 3, 2)
        self.csp45b = RepCSPDeep(c5 * 2, c5, n=repeat + 1)
        self.out_ch = (c2, c3, c4, c5)

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        t4 = self.csp54(torch.cat([self.up(self.lateral54(p5)), p4], 1))
        t3 = self.csp43(torch.cat([self.up(self.lateral43(t4)), p3], 1))
        t2 = self.csp32(torch.cat([self.up(self.lateral32(t3)), p2], 1))
        b3 = self.csp32b(torch.cat([self.down23(t2), t3], 1))
        b4 = self.csp34b(torch.cat([self.down34(b3), t4], 1))
        b5 = self.csp45b(torch.cat([self.down45(b4), p5], 1))
        return [t2, b3, b4, b5]


class OBBHead(nn.Module):
    def __init__(self, ch, nc=7, reg_max=16, emb_dim=96):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        # deeper decoupled branches (3 convs each) to learn complex floorplan features
        self.cls_conv = nn.Sequential(ConvBNAct(ch, ch, 3), ConvBNAct(ch, ch, 3), ConvBNAct(ch, ch, 3))
        self.box_conv = nn.Sequential(ConvBNAct(ch, ch, 3), ConvBNAct(ch, ch, 3), ConvBNAct(ch, ch, 3))
        self.ang_conv = nn.Sequential(ConvBNAct(ch, max(32, ch // 4), 3), ConvBNAct(max(32, ch // 4), max(32, ch // 4), 3),
                                      nn.Conv2d(max(32, ch // 4), 1, 3, 1, 1))
        self.cls_proj = nn.Conv2d(ch, emb_dim, 1)
        self.class_emb = nn.Embedding(nc, emb_dim)
        nn.init.normal_(self.class_emb.weight, std=0.02)
        self.box_pred = nn.Conv2d(ch, 4 * reg_max, 1)

    def forward(self, x):
        cf = self.cls_conv(x)
        bf = self.box_conv(x)
        emb = self.cls_proj(cf)
        cls = torch.einsum("behw,ne->bnhw", emb, self.class_emb.weight)
        box = self.box_pred(bf)
        ang = self.ang_conv(x)
        return cls, box, ang


class CADVLMDet(nn.Module):
    def __init__(self, width=1.0, nc=7, reg_max=16, depths=(4, 8, 6), tr_layers=4, repeat=2,
                 use_gcam=True, use_reason=True):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = [4, 8, 16, 32]
        self.encoder = HybridVLMEencoder(w=width, depths=depths, tr_layers=tr_layers,
                                         use_gcam=use_gcam, use_reason=use_reason)
        self.neck = PAN(self.encoder.c, repeat=repeat)
        och = self.neck.out_ch
        self.heads = nn.ModuleList([OBBHead(c, nc, reg_max) for c in och])
        self.register_buffer("topo_bias", torch.zeros(nc))

    def forward(self, x):
        feats = self.neck(self.encoder(x))
        return [h(f) for h, f in zip(self.heads, feats)]

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def build_model(variant="base", nc=7, ablate=None):
    """ablate: None | 'noreason' (tat transformer P5, test H2) | 'nogcam' (tat GCAM)."""
    cfg = {
        "nano": dict(width=0.5, depths=(2, 2, 2), tr_layers=2, repeat=1),
        "small": dict(width=0.75, depths=(3, 4, 3), tr_layers=3, repeat=2),
        "base": dict(width=1.0, depths=(4, 8, 6), tr_layers=4, repeat=2),
        "large": dict(width=1.25, depths=(6, 12, 8), tr_layers=6, repeat=3),
    }[variant]
    if ablate == "noreason":
        cfg = dict(cfg, use_reason=False)
    elif ablate == "nogcam":
        cfg = dict(cfg, use_gcam=False)
    return CADVLMDet(nc=nc, **cfg)
