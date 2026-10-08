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
from .context import GlobalContext
from .dense_head import DenseRotatedHead
from .obb_decoder import OBBDFINETransformer
from .vector import VectorBranch

DFINE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"
# Objects365 -> COCO checkpoints (what YOLO26's yolo26*.pt also starts from: Objects365 then COCO). D-FINE notes
# these may fall under Objects365 terms: benchmarking / research init only (user decision 2026-10-08).
DFINE_O365_FILES = {"s": "dfine_s_obj2coco.pth", "m": "dfine_m_obj2coco.pth", "l": "dfine_l_obj2coco_e25.pth",
                    "x": "dfine_x_obj2coco.pth"}

CONFIGS = {
    "s": dict(backbone=dict(name='B0', use_lab=True, return_idx=[1, 2, 3]),
              encoder=dict(in_channels=[256, 512, 1024], hidden_dim=256, depth_mult=0.34, expansion=0.5),
              decoder=dict(num_layers=3, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="s"),
    "m": dict(backbone=dict(name='B2', use_lab=True, return_idx=[1, 2, 3]),
              encoder=dict(in_channels=[384, 768, 1536], hidden_dim=256, depth_mult=0.67, expansion=1.0),
              decoder=dict(num_layers=4, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="m"),
    # L / X follow D-FINE: stem frozen and BatchNorm frozen in the backbone (also ~halves BN cost)
    "l": dict(backbone=dict(name='B4', use_lab=False, return_idx=[1, 2, 3], freeze_at=0, freeze_norm=True),
              encoder=dict(in_channels=[512, 1024, 2048], hidden_dim=256, depth_mult=1.0, expansion=1.0),
              decoder=dict(num_layers=6, hidden_dim=256, feat_channels=[256, 256, 256], num_points=[3, 6, 3]),
              pretrained="l"),
    "x": dict(backbone=dict(name='B5', use_lab=False, return_idx=[1, 2, 3], freeze_at=0, freeze_norm=True),
              encoder=dict(in_channels=[512, 1024, 2048], hidden_dim=384, depth_mult=1.0, expansion=1.0,
                           dim_feedforward=2048),
              decoder=dict(num_layers=6, hidden_dim=256, feat_channels=[384, 384, 384], num_points=[3, 6, 3],
                           reg_scale=8.0),
              pretrained="x"),
}

# COCO (contiguous 80-class index) -> DOTA-v1.0 class, used only to initialise classifier rows.
COCO_TO_DOTA = {"plane": 4, "helicopter": 4, "small-vehicle": 2, "large-vehicle": 7, "ship": 8}


class StripContext(nn.Module):
    """Large anisotropic receptive field for long / large objects (bridge, harbor, fields): depthwise 1xk and kx1
    strips + 3x3 + pointwise mix, residual, zero-initialised output so it starts as identity. Idea from strip /
    large selective kernels (Strip R-CNN, LSKNet); implementation is VRDet's own."""

    def __init__(self, c, k=11):
        super().__init__()
        self.h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.sq = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        y = self.bn(self.h(x) + self.v(x) + self.sq(x))
        return x + self.pw(nn.functional.silu(y))


class VRDet(nn.Module):
    def __init__(self, size="s", num_classes=15, num_queries=300, img_size=1024, rotate_sampling=True,
                 num_denoising=100, dense=False, dense_width=128, strip_k=0, ortho_heads=False, context=False,
                 dense_queries=False, vectors=False, vec_dim=128, vec_layers=2, o2m_queries=0, **overrides):
        super().__init__()
        cfg = copy.deepcopy(CONFIGS[size])
        for k, v in overrides.items():          # e.g. decoder=dict(num_layers=4)
            cfg[k].update(v)
        self.cfg = cfg
        bb = dict(freeze_at=-1, freeze_norm=False)
        bb.update(cfg["backbone"])
        self.backbone = HGNetv2(**bb, pretrained=False)
        self.encoder = HybridEncoder(**cfg["encoder"], eval_spatial_size=[img_size, img_size])
        self.decoder = OBBDFINETransformer(num_classes=num_classes, num_queries=num_queries,
                                           eval_spatial_size=[img_size, img_size], rotate_sampling=rotate_sampling,
                                           num_denoising=num_denoising, ortho_heads=ortho_heads,
                                           o2m_queries=o2m_queries, **cfg["decoder"])
        hid = cfg["encoder"]["hidden_dim"]
        self.context = nn.ModuleList([StripContext(hid, strip_k) for _ in range(3)]) if strip_k else None
        self.global_ctx = GlobalContext(self.backbone, cfg["encoder"]["in_channels"][-1], hid,
                                        p5_size=img_size // 32) if context else None
        self.dense_head = DenseRotatedHead(cfg["encoder"]["hidden_dim"], num_classes, (8, 16, 32), dense_width,
                                           img_size) if (dense or dense_queries) else None
        self.dense_queries = dense_queries
        self.vector = VectorBranch(cfg["encoder"]["in_channels"], d=vec_dim, layers=vec_layers) if vectors else None

    def forward(self, x, targets=None, ctx=None):
        fn = None
        if ctx is not None and self.global_ctx is not None and "thumb" in ctx:
            def fn(p5, pos):
                return self.global_ctx(p5, pos, ctx)
        feats = self.backbone(x)
        if ctx is not None and self.vector is not None and "vec" in ctx:
            feats = self.vector(feats, ctx["vec"], ctx["vec_mask"])
        feats = self.encoder(feats, ctx_fn=fn)
        if self.context is not None:
            feats = [m(f) for m, f in zip(self.context, feats)]
        dense = self.dense_head(feats) if self.dense_head is not None else None
        out = self.decoder(feats, targets, dense=dense if self.dense_queries else None)
        if dense is not None:
            out.update(dense)
        return out


def build_criterion(num_classes=15, reg_max=32, box_loss="kld", weights=None, cost=None, o2m_k=6):
    """box_loss: 'kld' (O2-DETR / RiO-DETR) or 'probiou' (PP-YOLOE-R, YOLO26); used in both cost and loss."""
    cost = cost or dict(cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0)
    weights = weights or {'loss_mal': 1, 'loss_bbox': 5, 'loss_kld': 2, 'loss_fgl': 0.15, 'loss_ddf': 1.5}
    return OBBCriterion(OBBHungarianMatcher(**cost, gauss=box_loss), weights, num_classes=num_classes,
                        reg_max=reg_max, gauss=box_loss, o2m_k=o2m_k)


def load_dfine_coco(model, ckpt_or_size, class_names=None, log=print, init="coco"):
    """Initialise from a D-FINE COCO (or Objects365->COCO) checkpoint; shape-mismatched tensors are copied on
    their overlap."""
    if isinstance(ckpt_or_size, str) and len(ckpt_or_size) == 1:
        url = DFINE_URL.format(ckpt_or_size) if init == "coco" else             DFINE_URL.rsplit("/", 1)[0] + "/" + DFINE_O365_FILES[ckpt_or_size]
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
    msg = f"[init] D-FINE {init}: {full} full, {partial} partial, {len(skipped)} new tensors"
    log(msg)
    logging.info(msg)
    return skipped


@torch.no_grad()
def postprocess(outputs, num_top=300, img_size=1024, mode="flat"):
    """-> scores (B, K), labels (B, K), boxes (B, K, 5) in pixels (cx, cy, w, h, theta).
    mode "flat": top-K over (query x class) pairs (DETR default; one query may emit several classes);
    "argmax": one class per query (removes cross-class duplicates such as plane queries scoring helicopter)."""
    logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
    B, Q, C = logits.shape
    if mode == "argmax":
        sc, lab = logits.sigmoid().max(-1)
        k = min(num_top, Q)
        s, q = sc.topk(k, dim=1)
        labels = lab.gather(1, q)
    else:
        scores = logits.sigmoid().flatten(1)
        k = min(num_top, scores.shape[1])
        s, idx = scores.topk(k, dim=1)
        labels = idx % C
        q = idx // C
    b = boxes.gather(1, q.unsqueeze(-1).expand(-1, -1, 5)).clone()
    b[..., :4] *= img_size
    return s, labels, b
