"""H7 vector branch: CAD primitives as tokens, fused into the raster detector.

Every primitive of the drawing (line / arc / circle / ellipse ..., from SVG or PDF vectors) is one token: type,
8 points resampled along it (image-normalised), stroke colour, line width. A light transformer runs over all
tokens of the image (drawing-level reasoning: which strokes form a symbol, what is next to what), then each
token's feature is splatted onto the cells its stroke passes through at the backbone strides and added to the
backbone maps through zero-initialised 1x1 convs. The model therefore starts exactly as the raster-only detector,
and the encoder / decoder learn to use vector evidence (stroke type, layer colour, exact geometry) that is
ambiguous in pixels. Idea family: CADTransformer / SymPoint / VecFormer (papers); implementation is VRDet's own.
"""
import math

import torch
import torch.nn as nn

VEC_DIM = 21        # type, 8 x (x, y) in [0, 1] of the image, r, g, b in [0, 1], line width / image size
NUM_TYPES = 8       # 0 line, 1 arc, 2 circle, 3 ellipse (FloorPlanCAD); 4.. reserved (bezier, rect, text, ...)


class VectorBranch(nn.Module):
    def __init__(self, out_channels, d=128, layers=2, heads=4, nfreq=8, densify=2):
        super().__init__()
        self.densify = densify
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi, persistent=False)
        self.type_emb = nn.Embedding(NUM_TYPES, d)
        self.geo = nn.Sequential(nn.Linear(16 * 2 * nfreq + 16 + 2, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.attr = nn.Linear(4, d)
        self.reg = nn.Parameter(torch.zeros(1, 1, d))       # always-valid drawing token: no fully masked rows
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.lvl = nn.ModuleList([nn.Linear(d, d) for _ in out_channels])
        self.out = nn.ModuleList([nn.Conv2d(d + 1, c, 1) for c in out_channels])     # + log stroke density
        for o in self.out:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def tokens(self, vec, mask):
        B, M, _ = vec.shape
        pts = vec[..., 1:17].float()
        ang = pts.unsqueeze(-1) * self.freqs                             # absolute position (B, M, 16, F)
        p = pts.view(B, M, 8, 2)
        rel = (p - p.mean(2, keepdim=True)) * 8.0                        # local shape, position-free
        ext = (p.amax(2) - p.amin(2)) * 8.0
        g = torch.cat([ang.sin().flatten(2), ang.cos().flatten(2), rel.flatten(2), ext], -1)
        attr = torch.cat([vec[..., 17:20].float(), torch.log1p(vec[..., 20:21].float() * 1024.0)], -1)
        typ = vec[..., 0].long().clamp(0, NUM_TYPES - 1)
        x = self.geo(g) + self.attr(attr) + self.type_emb(typ)
        x = torch.cat([self.reg.expand(B, -1, -1).to(x.dtype), x], 1)
        keep = torch.cat([mask.new_ones(B, 1), mask], 1)
        return self.blocks(self.norm(x), src_key_padding_mask=~keep)[:, 1:]

    def forward(self, feats, vec, mask):
        tok = self.tokens(vec, mask)
        B, M, d = tok.shape
        p = vec[..., 1:17].float().view(B, M, 8, 2)
        if self.densify > 1:            # extra points between samples so long strokes cover every cell they cross
            t = torch.linspace(0, 1, self.densify + 1, device=p.device)[:-1, None]
            mid = p[:, :, :-1, None] * (1 - t) + p[:, :, 1:, None] * t      # (B, M, 7, k, 2)
            p = torch.cat([mid.flatten(2, 3), p[:, :, -1:]], 2)
        bi, mi = mask.nonzero(as_tuple=True)
        pv, tv = p[bi, mi], tok[bi, mi]                                     # valid tokens only: (T, P, 2), (T, d)
        out = []
        for f, lin, conv in zip(feats, self.lvl, self.out):
            H, W = f.shape[-2:]
            ix, iy = (pv[..., 0] * W).floor().long(), (pv[..., 1] * H).floor().long()
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            rows = ok.nonzero(as_tuple=True)[0]
            cell = ((bi[:, None] * H + iy) * W + ix)[ok]
            src = lin(tv).float()[rows]
            acc = src.new_zeros(B * H * W, d).index_add_(0, cell, src)
            cnt = src.new_zeros(B * H * W, 1).index_add_(0, cell, torch.ones_like(src[:, :1]))
            m = torch.cat([acc / cnt.clamp(min=1), torch.log1p(cnt)], 1).view(B, H, W, d + 1).permute(0, 3, 1, 2)
            y = conv(m.to(f.dtype))
            if f.is_contiguous(memory_format=torch.channels_last):
                y = y.contiguous(memory_format=torch.channels_last)
            out.append(f + y)
        return out
