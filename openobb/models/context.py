"""Global image context for tiled inference (OpenOBB H6).

Large inputs (DOTA images up to ~4000 px, CAD sheets 2000x3000+) are processed as high-resolution tiles so
small objects and thin lines stay visible, but a tile alone loses the scene: where the water is (ship vs
harbour, bridge), the stadium layout (soccer field vs track field), the legend / title block of a drawing.

Each image gets ONE extra cheap pass: its thumbnail (long side 512) goes through the shared backbone, the
stride-32 map (16x16 = 256 tokens) is projected to the encoder width, and every tile's P5 tokens
cross-attend to these context tokens. Context token positions are expressed in the tile's own P5 grid
(the same sin-cos code the encoder's AIFI layer uses), so a tile knows where each piece of context lies
relative to itself, including outside its borders. The output projection starts at zero: at
initialisation the model is exactly the tile-only detector.

Cost: one thumbnail pass per image, amortised over its tiles (~25 for a 4000x4000 image), plus one
1024x256 attention per tile. All code is OpenOBB's own (Apache-2.0).
"""
import torch
import torch.nn as nn


def sincos_rc(row, col, dim, temperature=10000.0):
    """AIFI-compatible 2-D sin-cos code for continuous (row, col) grid coordinates; (...,) -> (..., dim)."""
    pos_dim = dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=row.device) / pos_dim
    omega = 1.0 / (temperature ** omega)
    r = row.float()[..., None] * omega
    c = col.float()[..., None] * omega
    return torch.cat([r.sin(), r.cos(), c.sin(), c.cos()], -1)


class GlobalContext(nn.Module):
    def __init__(self, backbone, in_ch, hidden=256, nhead=8, grid=16, p5_size=32):
        super().__init__()
        self.backbone = [backbone]          # shared, not registered twice (list hides it from nn.Module)
        self.grid, self.p5_size, self.hidden = grid, p5_size, hidden
        self.proj = nn.Sequential(nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden))
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.out = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def tokens(self, thumb, tile):
        """thumb (B, 3, T, T) in [0, 1]; tile (B, 4) = tile box on the thumbnail canvas, in canvas pixels."""
        B, _, T, _ = thumb.shape
        # The backbone is shared with the tile path: run the thumbnail with BatchNorm in eval mode and no
        # gradient, otherwise its batch statistics (different scale) leak into the running stats used by
        # the tiles at test time (observed: e2-h6-ctx-s trailed the baseline by ~3 points).
        bb = self.backbone[0]
        was_training = bb.training
        bb.eval()
        with torch.no_grad():
            feat = bb(thumb)[-1]
        bb.train(was_training)
        f = self.pool(self.proj(feat))                                # (B, C, g, g)
        g = self.grid
        tok = f.flatten(2).transpose(1, 2)                            # (B, g*g, C), row-major
        idx = torch.arange(g, device=thumb.device, dtype=torch.float32)
        cy = ((idx + 0.5) * T / g)[:, None].expand(g, g).reshape(-1)  # canvas pixel centres of the cells
        cx = ((idx + 0.5) * T / g)[None, :].expand(g, g).reshape(-1)
        x0, y0, x1, y1 = tile.float().unbind(-1)
        col = (cx[None] - x0[:, None]) / (x1 - x0).clamp(min=1e-6)[:, None] * self.p5_size - 0.5
        row = (cy[None] - y0[:, None]) / (y1 - y0).clamp(min=1e-6)[:, None] * self.p5_size - 0.5
        return tok, sincos_rc(row, col, self.hidden).to(tok.dtype)

    def forward(self, p5, pos, ctx):
        """p5 (B, C, H, W) after AIFI; pos (1, H*W, C) its sin-cos code; ctx dict(thumb, tile, valid)."""
        B, C, H, W = p5.shape
        tok, tpe = self.tokens(ctx["thumb"], ctx["tile"])
        q = self.norm_q(p5.flatten(2).transpose(1, 2)) + pos.to(p5.dtype)
        kv = self.norm_kv(tok)
        y, _ = self.attn(q, kv + tpe, kv, need_weights=False)
        y = self.out(y) * ctx["valid"].to(y.dtype)[:, None, None]     # samples without context: no change
        return p5 + y.transpose(1, 2).reshape(B, C, H, W)
