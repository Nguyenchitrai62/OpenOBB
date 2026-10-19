"""OpenOBB17 losses: OpenOBB14's (dual dense heads + edge refiner, the snap stage trained like a cascade stage on the
refined boxes) plus the boundary field's dense loss.

Field target: every labelled box draws a Gaussian ridge (sigma FIELD_SIGMA px) on its two ends (channel 0) and its
two long sides (channel 1), each segment limited to the box's extent (+ sigma); near-square boxes (L / T < SQUARE)
draw their four sides in both channels; overlapping ridges take the maximum. Loss: quality focal loss (BCE against the
soft target x |target - p|^2), normalised by the number of ridge pixels (target > 0.5).
"""
import torch
import torch.nn.functional as F

from .openobb14_loss import OpenOBB14Loss
from .openobb17 import FIELD_SIGMA, FIELD_STRIDE, SQUARE

CHUNK = 32               # boxes rasterised at once (memory: CHUNK x field pixels floats)


WINDOWED = True          # False: every box over the whole field grid (pre-F112; same targets to < 1e-7)
MARGIN = 6.0             # sigmas around a box's bounding rectangle: beyond it the ridge is < exp(-17)
WINDOW_ELEMS = 1 << 24   # field cells of one block of windows


def _ridges(gx, gy, g, sigma):
    """Cell centres gx, gy (n, ...) px, boxes g (n, 5) px -> (end, side) ridge values (n, ...) of each box, and the
    near-square mask (n,): OpenOBB17's target drawing."""
    swap = g[:, 2] < g[:, 3]                                                          # canonical: L along the axis
    L = torch.where(swap, g[:, 3], g[:, 2])
    T = torch.where(swap, g[:, 2], g[:, 3])
    th = g[:, 4] + swap.float() * (torch.pi / 2)
    shape = (-1,) + (1,) * (gx.dim() - 1)
    dx, dy = gx - g[:, 0].view(shape), gy - g[:, 1].view(shape)
    ca, sa = th.cos().view(shape), th.sin().view(shape)
    u = dx * ca + dy * sa
    v = dx * -sa + dy * ca
    hl, ht = (L / 2).view(shape), (T / 2).view(shape)
    end = torch.exp(-((u.abs() - hl) ** 2) / (2 * sigma ** 2)) * (v.abs() <= ht + sigma)
    side = torch.exp(-((v.abs() - ht) ** 2) / (2 * sigma ** 2)) * (u.abs() <= hl + sigma)
    return end, side, (L < SQUARE * T)


@torch.no_grad()
def field_targets(targets, shape, img_size, device, sigma=FIELD_SIGMA, stride=FIELD_STRIDE):
    """targets (list of {"boxes": (n, 5) normalised (cx, cy, w, h, theta)}), field (h, w) -> (B, 2, h, w) in [0, 1].
    Each box is drawn on a window of cells around it (power-of-two sizes, one block per size) instead of the whole
    grid: the ridge is below exp(-17) outside the window, so the target is the full-grid one (F112)."""
    if not WINDOWED:
        return _field_targets_full(targets, shape, img_size, device, sigma, stride)
    h, w = shape
    B = len(targets)
    out = torch.zeros(B * 2 * h * w, device=device)
    n = [len(t["boxes"]) for t in targets]
    if sum(n) == 0:
        return out.view(B, 2, h, w)
    g = torch.cat([t["boxes"].float().to(device) for t in targets]).clone()
    g[:, :4] *= img_size
    bi = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(n, device=device))
    c, s_ = g[:, 4].cos().abs(), g[:, 4].sin().abs()
    ex = c * g[:, 2] / 2 + s_ * g[:, 3] / 2 + MARGIN * sigma                         # half extents (px) + margin
    ey = s_ * g[:, 2] / 2 + c * g[:, 3] / 2 + MARGIN * sigma
    x0 = torch.floor((g[:, 0] - ex) / stride - 0.5).long()
    y0 = torch.floor((g[:, 1] - ey) / stride - 0.5).long()
    x1 = torch.ceil((g[:, 0] + ex) / stride - 0.5).long()
    y1 = torch.ceil((g[:, 1] + ey) / stride - 0.5).long()
    span = torch.maximum(x1 - x0, y1 - y0) + 1
    size = torch.clamp(2 ** torch.ceil(torch.log2(span.float().clamp(min=8))), max=float(max(h, w))).long()
    sizes = size.tolist()                                                            # the one sync
    for S in sorted(set(sizes)):
        idx = torch.tensor([i for i, v in enumerate(sizes) if v == S], device=device)
        Sy, Sx = min(S, h), min(S, w)
        oy, ox = torch.arange(Sy, device=device), torch.arange(Sx, device=device)
        step = max(1, WINDOW_ELEMS // (Sy * Sx))
        for j in range(0, len(idx), step):
            k = idx[j:j + step]
            # window origin inside the grid; the box's cells [x0, x1] always fit (span <= S)
            ax0 = torch.minimum(x0[k].clamp(min=0), torch.full_like(x0[k], w - Sx))
            ay0 = torch.minimum(y0[k].clamp(min=0), torch.full_like(y0[k], h - Sy))
            iy = ay0[:, None, None] + oy[None, :, None]                               # (n, Sy, 1)
            ix = ax0[:, None, None] + ox[None, None, :]                               # (n, 1, Sx)
            gy = (iy.float() + 0.5) * stride
            gx = (ix.float() + 0.5) * stride
            end, side, sq = _ridges(gx.expand(-1, Sy, Sx), gy.expand(-1, Sy, Sx), g[k], sigma)
            both = torch.maximum(end, side)
            sq = sq[:, None, None]
            c0, c1 = torch.where(sq, both, end), torch.where(sq, both, side)
            cell = (iy * w + ix).expand(-1, Sy, Sx)
            base = (bi[k] * 2 * h * w)[:, None, None]
            flat = torch.cat([(base + cell).reshape(-1), (base + h * w + cell).reshape(-1)])
            out.scatter_reduce_(0, flat, torch.cat([c0.reshape(-1), c1.reshape(-1)]), "amax", include_self=True)
    return out.view(B, 2, h, w)


@torch.no_grad()
def _field_targets_full(targets, shape, img_size, device, sigma=FIELD_SIGMA, stride=FIELD_STRIDE):
    """Pre-F112 drawing: every box over the whole grid (reference of the windowed one)."""
    h, w = shape
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)                          # (h w, 2) px
    out = torch.zeros((len(targets), 2, h * w), device=device)
    for b, t in enumerate(targets):
        g = t["boxes"].float().to(device)
        if not len(g):
            continue
        g = g.clone()
        g[:, :4] *= img_size
        swap = g[:, 2] < g[:, 3]                                                      # canonical: L along the axis
        L = torch.where(swap, g[:, 3], g[:, 2])
        T = torch.where(swap, g[:, 2], g[:, 3])
        th = g[:, 4] + swap.float() * (torch.pi / 2)
        square = L < SQUARE * T
        for i in range(0, len(g), CHUNK):
            sl = slice(i, i + CHUNK)
            d = grid[None] - g[sl, None, :2]                                          # (n, hw, 2)
            ax = torch.stack([th[sl].cos(), th[sl].sin()], -1)[:, None]
            u = (d * ax).sum(-1)
            v = d[..., 0] * -ax[..., 1] + d[..., 1] * ax[..., 0]
            hl, ht = L[sl, None] / 2, T[sl, None] / 2
            end = torch.exp(-((u.abs() - hl) ** 2) / (2 * sigma ** 2)) * (v.abs() <= ht + sigma)
            side = torch.exp(-((v.abs() - ht) ** 2) / (2 * sigma ** 2)) * (u.abs() <= hl + sigma)
            sq = square[sl, None]
            both = torch.maximum(end, side)
            c0 = torch.where(sq, both, end).amax(0)
            c1 = torch.where(sq, both, side).amax(0)
            out[b, 0] = torch.maximum(out[b, 0], c0)
            out[b, 1] = torch.maximum(out[b, 1], c1)
    return out.reshape(len(targets), 2, h, w)


def field_loss(logits, target, gamma=2.0):
    """Quality focal loss on soft targets, normalised by the number of ridge pixels."""
    p = logits.sigmoid()
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (target - p).abs().pow(gamma)
    return l.sum() / (target > 0.5).sum().clamp(min=1.0)


class OpenOBB17Loss(OpenOBB14Loss):
    def __init__(self, w_field=1.0, **kw):
        super().__init__(**kw)
        self.w_field = w_field

    def forward(self, outputs, targets):
        losses = super().forward(outputs, targets)
        if "field" in outputs:
            f = outputs["field"]
            tgt = field_targets(targets, f.shape[-2:], self.img_size, f.device)
            losses["loss_field"] = self.w_field * field_loss(f, tgt)
        return losses
