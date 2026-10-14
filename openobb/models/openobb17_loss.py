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


@torch.no_grad()
def field_targets(targets, shape, img_size, device, sigma=FIELD_SIGMA, stride=FIELD_STRIDE):
    """targets (list of {"boxes": (n, 5) normalised (cx, cy, w, h, theta)}), field (h, w) -> (B, 2, h, w) in [0, 1]."""
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
