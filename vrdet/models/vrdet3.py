"""VRDet3: pretrained hybrid-encoder features + dense anisotropic oriented head (thin / long / small objects).

Lessons it is built on (research/LEDGER.md F23-F25, Wall_Color, 250 images):
  * VRDet1 (DETR decoder, COCO init) 0.63 / 0.35; VRDet2 (dense, from scratch) 0.27 / 0.10 after 50 epochs:
    on small data the pretrained features decide. VRDet3 keeps the COCO-pretrained HGNetv2 backbone + hybrid
    encoder (D-FINE, Apache-2.0) and the selective-kernel adapters measured on CAD (c6: +1.3 mAP50).
  * The DETR decoder (top-K query cap, one positive per object, 6 refinement layers with grid sampling) is replaced
    by a dense head on every location of P3-P5: many positives per object, no query cap, fast.
  * VRDet2's segment regression (relative position x log length) left the along-axis error flat (end_loss ~0.93).
    VRDet3 regresses the two along-axis distances as distributions (DFL, as the YOLO family does for box sides),
    with an anisotropic grid: along bins = 1 stride (31 strides of reach, long walls), thickness bins = stride / 2
    (sub-stride precision, thin objects), and a signed across offset so points beside a thin object can predict it.
  * Candidates: every level gets a row of points along every object (across band >= 1 stride), so coarse points
    that see both ends of a long wall can be assigned to it; task-aligned assignment picks the best level.
  * Output: class-wise Fast-NMS on ProbIoU inside the model (YOLO-style dense output), padded to `max_det`.

Box convention: (cx, cy, w, h, theta) with w measured along theta. Eval output matches VRDet1:
{"pred_logits": (B, K, C), "pred_boxes": (B, K, 5) normalised}, read by vrdet.models.vrdet.postprocess.
"""
import copy
import math

import torch
import torch.nn as nn

from vrdet.ops.obb_torch import probiou

from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .vrdet import CONFIGS, SelectiveKernel
from .vrdet2 import Conv

ALONG_BINS = 32          # distance from the point to each end of the object, bin = 1 stride
THICK_BINS = 32          # thickness (short side), bin = stride / 2
REG_CH = 2 + 2 * ALONG_BINS + THICK_BINS     # angle, across offset, left, right, thickness


def _expect(logits, width):
    bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (logits.softmax(-1) * bins).sum(-1) * width


def decode(reg, pts, strides):
    """reg (B, A, REG_CH) -> boxes (B, A, 5) px. The point sits `l` behind and `r` ahead along theta and `a` across."""
    s = strides[None, :]
    th = reg[..., 0]
    a = reg[..., 1] * s
    left = _expect(reg[..., 2:2 + ALONG_BINS], 1.0) * s
    right = _expect(reg[..., 2 + ALONG_BINS:2 + 2 * ALONG_BINS], 1.0) * s
    thick = _expect(reg[..., 2 + 2 * ALONG_BINS:], 1.0) * (s / 2)
    c, sn = th.cos(), th.sin()
    du = (right - left) / 2
    cx = pts[None, :, 0] + du * c - a * sn
    cy = pts[None, :, 1] + du * sn + a * c
    return torch.stack([cx, cy, left + right, thick, th], -1)


class OrientedHead(nn.Module):
    """Per level: class logits (light depthwise tower) + oriented regression (angle, across offset, two along-axis
    distance distributions, thickness distribution)."""

    def __init__(self, ch, nc, prior=0.01):
        super().__init__()
        cr = max(128, ch[0] // 3)
        cc = max(ch[0] // 2, min(nc * 2, 128))
        self.reg = nn.ModuleList(nn.Sequential(Conv(c, cr, 3), Conv(cr, cr, 3), nn.Conv2d(cr, REG_CH, 1)) for c in ch)
        self.cls = nn.ModuleList(nn.Sequential(Conv(c, c, 3, g=c), Conv(c, cc, 1), Conv(cc, cc, 3, g=cc),
                                               Conv(cc, cc, 1), nn.Conv2d(cc, nc, 1)) for c in ch)
        for m in self.cls:
            nn.init.constant_(m[-1].bias, -math.log((1 - prior) / prior))
        for m in self.reg:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def forward(self, feats):
        cat = lambda mods: torch.cat([m(f).flatten(2) for m, f in zip(mods, feats)], 2).transpose(1, 2).float()  # noqa: E731
        return cat(self.cls), cat(self.reg)


@torch.no_grad()
def fast_nms(logits, boxes, max_det=1000, conf=0.001, iou=0.7, pre=4000):
    """Per image class-wise Fast-NMS on ProbIoU -> padded (B, max_det, C) logits and (B, max_det, 5) boxes px."""
    B, A, C = logits.shape
    out_l = logits.new_full((B, max_det, C), -30.0)
    out_b = boxes.new_zeros((B, max_det, 5))
    out_b[..., 2:4] = 1.0
    for i in range(B):
        sc = logits[i].sigmoid().flatten()
        s, idx = sc.topk(min(pre, sc.numel()))
        keep = s > conf
        s, idx = s[keep], idx[keep]
        if not len(s):
            continue
        lab, b = idx % C, boxes[i][idx // C]
        ov = probiou(b[:, None, :], b[None, :, :]) * (lab[:, None] == lab[None, :])
        keep = torch.triu(ov, diagonal=1).amax(0) <= iou
        s, lab, b = s[keep][:max_det], lab[keep][:max_det], b[keep][:max_det]
        n = len(s)
        out_l[i, torch.arange(n, device=s.device), lab] = torch.logit(s.clamp(1e-6, 1 - 1e-6))
        out_b[i, :n] = b
    return out_l, out_b


class VRDet3(nn.Module):
    arch = "v3"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, backbone="hgnet",
                 **_):
        super().__init__()
        cfg = copy.deepcopy(CONFIGS[size])
        self.size, self.nc, self.img_size = size, num_classes, img_size
        self.max_det, self.nms_iou = max_det, nms_iou
        bb = dict(freeze_at=-1, freeze_norm=False)
        bb.update(cfg["backbone"])
        # same module names as VRDet1 / D-FINE: COCO weights (and a VRDet1 checkpoint) load into backbone/encoder/lsk
        if backbone == "hgnet":
            self.backbone = HGNetv2(**bb, pretrained=False)
        else:                                   # DINOv2 ViT + adapter (vrdet/models/vit.py)
            from .vit import DinoV2Backbone
            self.backbone = DinoV2Backbone(backbone, cfg["encoder"]["in_channels"])
        self.backbone_name = backbone
        self.lsk = nn.ModuleList([SelectiveKernel(c) for c in cfg["encoder"]["in_channels"]]) if lsk else None
        self.encoder = HybridEncoder(**cfg["encoder"], eval_spatial_size=[img_size, img_size])
        hid = cfg["encoder"]["hidden_dim"]
        self.strides = (8, 16, 32)
        self.head = OrientedHead([hid] * 3, num_classes)
        self._grid = {}

    def reset_input_proj(self):
        """Re-initialise the encoder's input projections (their COCO weights were fitted to HGNetv2 maps)."""
        for m in self.encoder.input_proj.modules():
            if hasattr(m, "reset_parameters") and m is not self.encoder.input_proj:
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

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
        feats = self.backbone(x)
        if self.lsk is not None:
            feats = [m(f) for m, f in zip(self.lsk, feats)]
        feats = self.encoder(feats)
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        if self.training:
            return {"cls": cls, "reg": reg, "points": pts, "strides": st}
        boxes = decode(reg, pts, st)
        logits, boxes = fast_nms(cls, boxes, self.max_det, iou=self.nms_iou)
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}


class DenseOriented(nn.Module):
    """VRDet3's oriented head as the dense branch of the DETR model (VRDet4): one-to-many auxiliary supervision of
    the encoder (Co-DETR / RT-DETRv3 idea) and distinct top-k proposals for the decoder queries (DDQ idea).
    Output keys follow the VRDet1 dense interface (normalised boxes) plus what VRDet3Loss needs (pixels)."""

    def __init__(self, ch, nc, img_size, strides=(8, 16, 32)):
        super().__init__()
        self.head = OrientedHead([ch] * len(strides), nc)
        self.strides, self.img_size = strides, img_size
        self._grid = {}

    def forward(self, feats):
        cls, reg = self.head(feats)
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
        pts, st = self._grid[key]
        boxes = decode(reg, pts, st)
        nb = torch.cat([boxes[..., :4] / self.img_size, boxes[..., 4:]], -1)
        return {"dense_logits": cls, "dense_boxes": nb, "dense_points": pts / self.img_size,
                "dense_strides": st / self.img_size, "dense_reg": reg, "dense_pts_px": pts, "dense_st_px": st}
