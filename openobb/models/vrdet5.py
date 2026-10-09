"""VRDet5: VRDet3's dense oriented detector + DINOv2 features + geometry-aware classes + relation re-scoring.

Built from the Wall_Color results (research/LEDGER.md F31-F33, Ultralytics-style mAP):
  * VRDet3 (HGNetv2 COCO + dense oriented head) 0.712 / 0.467: best on rare classes (many positives per object),
    fast (23 ms); weak on wall_300 (0.599) - the class that makes ~95% of the gap to YOLO26x (0.746 / 0.499);
  * VRDet4 (DINOv2 + DETR decoder) 0.663 / 0.457 in one run: better wall_300 (+0.08) and wall fit, but weak rare
    classes (one positive per object in the decoder) and 38 ms.
VRDet5 keeps the dense head as the output (rare classes, speed) and adds what VRDet4 suggested helps:
  1. DINOv2 ViT-B backbone (openobb/models/vit.py) - self-supervised features (default; backbone=hgnet for speed);
  2. geometry-aware classification: the class branch reads the geometry the box branch measured on the same point
     (thickness, length, angle, across offset, thickness uncertainty; detached). wall vs wall_300 differ mostly by
     thickness, which the class branch of VRDet3 could not see;
  3. relation re-scoring: the ~600 distinct best candidates of an image attend to each other in a small transformer
     (2 layers) and correct their class logits (residual, zero-initialised) - context such as "this wall is thicker
     than the walls it meets" or "this door has no swing arc", at a fraction of a DETR decoder's cost
     (relation-network / DETR self-attention idea on dense candidates).
Parameter names of the head match VRDet3, so a VRDet3 checkpoint initialises VRDet5's head (weights=).
"""
import math

import torch
import torch.nn as nn

from openobb.ops.obb_torch import probiou

from .obb_decoder import distinct_topk
from .vrdet3 import ALONG_BINS, THICK_BINS, OrientedHead, VRDet3, decode, fast_nms

GEO_CH = 7


def geometry_maps(r, stride):
    """r (B, REG_CH, H, W) raw regression (detached) -> (B, GEO_CH, H, W) scale-free geometry descriptors."""
    def expect(p):
        bins = torch.arange(p.shape[1], device=p.device, dtype=p.dtype)[None, :, None, None]
        return (p * bins).sum(1), (p * bins * bins).sum(1)
    th, a = r[:, 0], r[:, 1]
    el, _ = expect(r[:, 2:2 + ALONG_BINS].softmax(1))
    er, _ = expect(r[:, 2 + ALONG_BINS:2 + 2 * ALONG_BINS].softmax(1))
    et, et2 = expect(r[:, 2 + 2 * ALONG_BINS:].softmax(1))
    length = (el + er) * stride
    thick = et * stride / 2
    std = (et2 - et * et).clamp(min=0).sqrt()
    return torch.stack([torch.log1p(thick) / 4, torch.log1p(length) / 7, torch.sin(2 * th), torch.cos(2 * th),
                        a.clamp(-4, 4) / 4, std / 8, torch.full_like(th, math.log2(stride) / 5)], 1)


class GeoOrientedHead(OrientedHead):
    """VRDet3's head; the class tower gets a zero-initialised embedding of the predicted geometry after its first
    two layers (same parameter names as OrientedHead + `geo`)."""

    def __init__(self, ch, nc, strides=(8, 16, 32), prior=0.01):
        super().__init__(ch, nc, prior)
        self.strides = strides
        cc = self.cls[0][1].conv.out_channels
        self.geo = nn.ModuleList(nn.Conv2d(GEO_CH, cc, 1) for _ in ch)
        for g in self.geo:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, feats):
        cls, reg = [], []
        for f, mc, mr, g, s in zip(feats, self.cls, self.reg, self.geo, self.strides):
            r = mr(f)
            h = mc[1](mc[0](f)) + g(geometry_maps(r.detach().float(), s).to(f.dtype))
            for layer in list(mc)[2:]:
                h = layer(h)
            cls.append(h.flatten(2))
            reg.append(r.flatten(2))
        return torch.cat(cls, 2).transpose(1, 2).float(), torch.cat(reg, 2).transpose(1, 2).float()


class RelationRescorer(nn.Module):
    """Self-attention over the k distinct best candidates of each image; residual correction of their class logits."""

    def __init__(self, ch, nc, d=256, layers=2, heads=8, k=600):
        super().__init__()
        self.k = k
        self.feat = nn.Linear(ch, d)
        self.geo = nn.Sequential(nn.Linear(6 + nc, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                                               batch_first=True, norm_first=True) for _ in range(layers))
        self.out = nn.Linear(d, nc)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, flat, cls, boxes, img_size):
        """flat (B, A, ch) features, cls (B, A, C) logits, boxes (B, A, 5) px -> idx (B, k), logits (B, k, C)."""
        sc = cls.detach().sigmoid().amax(-1)
        k = min(self.k, sc.shape[1])
        idx = distinct_topk(sc, boxes.detach(), k, 3 * k)
        gather = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
        f, base, b = gather(flat), gather(cls).detach(), gather(boxes).detach()
        geo = torch.cat([b[..., :2] / img_size, torch.log(b[..., 2:4].clamp(min=1) / img_size),
                         torch.sin(2 * b[..., 4:5]), torch.cos(2 * b[..., 4:5]), base.sigmoid()], -1)
        x = self.norm(self.feat(f.float()) + self.geo(geo))
        for blk in self.blocks:
            x = blk(x)
        return idx, base + self.out(x)


class VRDet5(VRDet3):
    arch = "v5"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7,
                 backbone="dinov2_b", geo_cls=True, relate=True, rel_k=600, **_):
        super().__init__(size, num_classes, img_size, lsk, max_det, nms_iou, backbone=backbone)
        hid = self.encoder.hidden_dim
        if geo_cls:
            self.head = GeoOrientedHead([hid] * 3, num_classes, self.strides)
        self.rescorer = RelationRescorer(hid, num_classes, k=rel_k) if relate else None

    def forward(self, x, targets=None, ctx=None):
        feats = self.backbone(x)
        if self.lsk is not None:
            feats = [m(f) for m, f in zip(self.lsk, feats)]
        feats = self.encoder(feats)
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        boxes = decode(reg, pts, st)
        rel = None
        if self.rescorer is not None:
            flat = torch.cat([f.flatten(2) for f in feats], 2).transpose(1, 2)
            rel = self.rescorer(flat, cls, boxes, self.img_size)
        if self.training:
            out = {"cls": cls, "reg": reg, "points": pts, "strides": st}
            if rel is not None:
                idx, logits = rel
                out.update(rel_logits=logits, rel_boxes=boxes.detach().gather(1, idx[..., None].expand(-1, -1, 5)))
            return out
        if rel is not None:                     # output = the re-scored distinct candidates
            idx, cls = rel
            boxes = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
        logits, boxes = fast_nms(cls, boxes, self.max_det, iou=self.nms_iou)
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}


def relation_targets(boxes, targets, img_size, nc, thr=0.5):
    """Soft class targets for the re-scored candidates: ProbIoU with the best-overlapping GT if >= thr."""
    tsc = boxes.new_zeros(boxes.shape[0], boxes.shape[1], nc)
    for b, t in enumerate(targets):
        if not len(t["labels"]):
            continue
        g = t["boxes"].float().clone()
        g[:, :4] *= img_size
        iou = probiou(boxes[b][:, None, :].float(), g[None, :, :]).clamp(0, 1)
        best, gi = iou.max(1)
        pos = best >= thr
        tsc[b, pos.nonzero(as_tuple=True)[0], t["labels"][gi[pos]]] = best[pos]
    return tsc
