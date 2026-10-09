"""DINOv2 ViT backbone for VRDet4 (VRDet's own implementation; parameter names follow the released DINOv2
checkpoints, Apache-2.0, so they load directly).

Why: on small datasets from an unusual domain, self-supervised DINOv2 features transfer better than COCO-supervised
CNN features (RF-DETR, RF100-VL fine-tuning results). A plain ViT has one stride-14 map, so a light adapter builds
the stride 8 / 16 / 32 maps the hybrid encoder expects (simple feature pyramid, ViTDet idea) from four intermediate
layers, and a small convolutional branch at stride 8 restores the pixel detail that 14 px patches blur (thin lines;
ViT-Adapter's spatial-prior idea).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .vrdet2 import Conv

# embed dim, heads, depth, checkpoint tag (registers variant: cleaner dense features)
DINOV2 = {"dinov2_s": (384, 6, 12, "vits14"), "dinov2_b": (768, 12, 12, "vitb14"), "dinov2_l": (1024, 16, 24, "vitl14")}
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_{0}/dinov2_{0}_reg4_pretrain.pth"
PATCH = 14


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, d, init=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d,), init))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.attn, self.mlp = Attention(d, heads), Mlp(d, 4 * d)
        self.ls1, self.ls2 = LayerScale(d), LayerScale(d)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Conv2d(3, d, PATCH, PATCH)

    def forward(self, x):
        return self.proj(x)


class ViT(nn.Module):
    """DINOv2-style ViT with 4 register tokens; positional embeddings interpolated to any grid."""

    def __init__(self, d=768, depth=12, heads=12, n_reg=4, pos_grid=37):
        super().__init__()
        self.d, self.n_reg, self.pos_grid = d, n_reg, pos_grid
        self.patch_embed = PatchEmbed(d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + pos_grid * pos_grid, d))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_reg, d))
        self.mask_token = nn.Parameter(torch.zeros(1, d))              # unused, present in the checkpoints
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        self.grad_ckpt = False

    def _pos(self, h, w, dtype):
        p = self.pos_embed.float()
        cls, grid = p[:, :1], p[:, 1:].reshape(1, self.pos_grid, self.pos_grid, self.d).permute(0, 3, 1, 2)
        if (h, w) != (self.pos_grid, self.pos_grid):
            grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        return torch.cat([cls, grid.flatten(2).transpose(1, 2)], 1).to(dtype)

    def forward(self, x, layers):
        """x (B, 3, H, W) with H, W multiples of 14 -> list of normalised (B, d, H/14, W/14) maps from `layers`."""
        t = self.patch_embed(x)
        b, _, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)
        t = torch.cat([self.cls_token.expand(b, -1, -1).to(t.dtype), t], 1) + self._pos(h, w, t.dtype)
        t = torch.cat([t[:, :1], self.register_tokens.expand(b, -1, -1).to(t.dtype), t[:, 1:]], 1)
        outs, want = [], set(layers)
        for i, blk in enumerate(self.blocks):
            if self.grad_ckpt and self.training:
                t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
            if i in want:
                outs.append(self.norm(t)[:, 1 + self.n_reg:].transpose(1, 2).reshape(b, self.d, h, w))
        return outs


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT + adapter -> [P3, P4, P5] (strides 8 / 16 / 32) with `out_channels` (the encoder's in_channels)."""

    def __init__(self, name="dinov2_b", out_channels=(512, 1024, 2048)):
        super().__init__()
        d, heads, depth, self.tag = DINOV2[name]
        self.name = name
        self.vit = ViT(d, depth, heads)
        self.layers = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        h = d // 2
        self.adapter = nn.ModuleDict({
            "fuse": Conv(4 * d, d, 1),                                            # four depths -> one map
            "up": nn.Sequential(nn.ConvTranspose2d(d, h, 2, 2), nn.BatchNorm2d(h), nn.SiLU()),
            "spm": nn.Sequential(Conv(3, 64, 3, 2), Conv(64, 128, 3, 2), Conv(128, h, 3, 2)),   # pixel detail /8
            "p3": Conv(h, out_channels[0], 3), "p4": Conv(d, out_channels[1], 1), "p5": Conv(d, out_channels[2], 1),
        })
        self._out_channels = list(out_channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        xn = (x - self.mean) / self.std
        ph, pw = math.ceil(H / PATCH) * PATCH - H, math.ceil(W / PATCH) * PATCH - W
        f = self.adapter["fuse"](torch.cat(self.vit(F.pad(xn, (0, pw, 0, ph)), self.layers), 1))
        s8, s16, s32 = (H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)
        p3 = F.interpolate(self.adapter["up"](f), size=s8, mode="bilinear", align_corners=False)
        p3 = self.adapter["p3"](p3 + self.adapter["spm"](x))
        p4 = self.adapter["p4"](F.interpolate(f, size=s16, mode="bilinear", align_corners=False))
        p5 = self.adapter["p5"](F.adaptive_avg_pool2d(f, s32))
        return [p3, p4, p5]

    def load_pretrained(self, log=print):
        """Download the DINOv2 (registers) weights; fail loudly if they do not fit (never train a random ViT)."""
        url = DINOV2_URL.format(self.tag)
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("head")}
        own = self.vit.state_dict()
        bad = [k for k, v in sd.items() if k not in own or own[k].shape != v.shape]
        if bad:
            raise RuntimeError(f"DINOv2 checkpoint does not fit the ViT ({len(bad)} tensors): {bad[:5]}")
        missing = [k for k in own if k not in sd]
        self.vit.load_state_dict(sd, strict=False)
        log(f"[init] DINOv2 {self.name} (Apache-2.0): {len(sd)} tensors loaded, {len(missing)} missing {missing[:3]}")
        if missing:
            raise RuntimeError(f"DINOv2 checkpoint is missing {missing[:5]}")
