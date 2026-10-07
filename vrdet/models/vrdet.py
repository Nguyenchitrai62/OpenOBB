"""VRDet model assembly: HGNetv2 backbone + hybrid encoder (D-FINE, Apache-2.0) + oriented D-FINE decoder.

Sizes mirror D-FINE so its COCO checkpoints (Apache-2.0, COCO-only: no Objects365 terms) initialise
everything that has a matching shape; oriented-specific rows/columns start from neutral values.
"""
import copy
import logging

import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .obb_criterion import OBBCriterion, OBBHungarianMatcher
from .dense_head import DenseRotatedHead
from .obb_decoder import OBBDFINETransformer

DFINE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"

CONFIGS = {
    "s": dict(backbone=dict(name='B0', use_lab=True, return_idx=[1, 2, 3]),
              encoder=dict(in_channels=[256, 512, 1024], hidden_dim=256, depth_mult=0.34, expansion=0.5),
              decoder=dict(num_layers=3, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="s"),
    "m": dict(backbone=dict(name='B2', use_lab=True, return_idx=[1, 2, 3]),
              encoder=dict(in_channels=[384, 768, 1536], hidden_dim=256, depth_mult=0.67, expansion=1.0),
              decoder=dict(num_layers=4, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="m"),
    "l": dict(backbone=dict(name='B4', use_lab=False, return_idx=[1, 2, 3]),
              encoder=dict(in_channels=[512, 1024, 2048], hidden_dim=256, depth_mult=1.0, expansion=1.0),
              decoder=dict(num_layers=6, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="l"),
}

# COCO (contiguous 80-class index) -> DOTA-v1.0 class, used only to initialise classifier rows.
COCO_TO_DOTA = {"plane": 4, "helicopter": 4, "small-vehicle": 2, "large-vehicle": 7, "ship": 8}


class VRDet(nn.Module):
    def __init__(self, size="s", num_classes=15, num_queries=300, img_size=1024, rotate_sampling=True,
                 num_denoising=100, dense=False, dense_width=128, **overrides):
        super().__init__()
        cfg = copy.deepcopy(CONFIGS[size])
        for k, v in overrides.items():          # e.g. decoder=dict(num_layers=4)
            cfg[k].update(v)
        self.cfg = cfg
        self.backbone = HGNetv2(**cfg["backbone"], freeze_at=-1, freeze_norm=False, pretrained=False)
        self.encoder = HybridEncoder(**cfg["encoder"], eval_spatial_size=[img_size, img_size])
        self.decoder = OBBDFINETransformer(num_classes=num_classes, num_queries=num_queries,
                                           eval_spatial_size=[img_size, img_size], rotate_sampling=rotate_sampling,
                                           num_denoising=num_denoising, **cfg["decoder"])
        self.dense_head = DenseRotatedHead(cfg["encoder"]["hidden_dim"], num_classes, (8, 16, 32), dense_width,
                                           img_size) if dense else None

    def forward(self, x, targets=None):
        feats = self.encoder(self.backbone(x))
        out = self.decoder(feats, targets)
        if self.dense_head is not None:
            out.update(self.dense_head(feats))
        return out


def build_criterion(num_classes=15, reg_max=32, box_loss="kld", weights=None, cost=None):
    cost = cost or dict(cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0)
    weights = weights or {'loss_mal': 1, 'loss_bbox': 5, 'loss_kld': 2, 'loss_fgl': 0.15, 'loss_ddf': 1.5}
    return OBBCriterion(OBBHungarianMatcher(**cost), weights, num_classes=num_classes, reg_max=reg_max)


def load_dfine_coco(model, ckpt_or_size, class_names=None, log=print):
    """Initialise from a D-FINE COCO checkpoint; shape-mismatched tensors are copied on their overlap."""
    if isinstance(ckpt_or_size, str) and len(ckpt_or_size) == 1:
        url = DFINE_URL.format(ckpt_or_size)
        ck = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
    else:
        ck = torch.load(ckpt_or_size, map_location="cpu")
    if "ema" in ck and isinstance(ck["ema"], dict) and "module" in ck["ema"]:
        src = ck["ema"]["module"]
    else:
        src = ck.get("model", ck)
    dst = model.state_dict()
    full, partial, skipped = 0, 0, []
    cls_rows = None
    if class_names is not None:
        cls_rows = {i: COCO_TO_DOTA[c] for i, c in enumerate(class_names) if c in COCO_TO_DOTA}
    with torch.no_grad():
        for k, v in dst.items():
            if k not in src:
                skipped.append(k)
                continue
            s = src[k]
            if s.shape == v.shape:
                v.copy_(s)
                full += 1
                continue
            if s.dim() != v.dim():
                skipped.append(k)
                continue
            is_cls = ("score_head" in k) or ("denoising_class_embed" in k)
            if is_cls:
                if cls_rows:
                    for i, j in cls_rows.items():
                        v[i].copy_(s[j])
                    if "denoising_class_embed" in k:
                        v[-1].copy_(s[-1])
                    partial += 1
                else:
                    skipped.append(k)
                continue
            sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, s.shape))
            v[sl].copy_(s[sl])
            partial += 1
    model.load_state_dict(dst)
    msg = f"[init] D-FINE COCO: {full} full, {partial} partial, {len(skipped)} new tensors"
    log(msg)
    logging.info(msg)
    return skipped


@torch.no_grad()
def postprocess(outputs, num_top=300, img_size=1024):
    """-> scores (B, K), labels (B, K), boxes (B, K, 5) in pixels (cx, cy, w, h, theta)."""
    logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
    B, Q, C = logits.shape
    scores = logits.sigmoid().flatten(1)
    k = min(num_top, scores.shape[1])
    s, idx = scores.topk(k, dim=1)
    labels = idx % C
    q = idx // C
    b = boxes.gather(1, q.unsqueeze(-1).expand(-1, -1, 5)).clone()
    b[..., :4] *= img_size
    return s, labels, b
