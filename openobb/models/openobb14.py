"""OpenOBB14: pretrained trunk + OpenOBB's own high-resolution re-parameterised neck + heavy dual dense oriented head
with direct box regression + an edge-aligned box refiner that reads the stride-4 trunk map. docs/OPENOBB14.md.

Why (research/LEDGER.md F37, F45-F57; research/analysis/roofline.py):
  * Every accurate OpenOBB stood on the D-FINE COCO weights; every core trained from scratch or distilled from a COCO
    teacher failed (v2, v6, v9; F25, F40, F50). The HGNetv2 trunk is therefore kept exactly (its weights load as
    they are: parameter names unchanged, "backbone.*").
  * The borrowed hybrid encoder costs 38% of v12's FLOPs (FPN block at P3 alone 28%), the LSK adapters 7%; both are
    1x1-heavy or depthwise, i.e. memory-bound on a GPU (roofline, F58), and the encoder's attention at 1/32 is flat
    on drawings (F37). They are replaced by a neck of dense re-parameterised 3x3 convolutions (OpenOBB9's RepConv /
    RepCSP: compute-bound, folded into single convolutions at inference), with dilations 1 / 2 / 1 / 3 at P3-P4 for
    the 40-170 px context the class logits actually use (F37), stride-4 detail by space-to-depth, max-pool paths
    when downsampling, and one zero-initialised attention layer at 1/32.
  * The mAP50-95 gap to YOLO26x is box fit of thin objects (F54: thickness +0.66 px bias and 1.0 px error on 5.6 px
    walls, end error 2.4 vs 1.3 px, angle 0.5-2 deg vs ~0): the DFL bins (stride / 2) are coarse for sub-cell
    thickness and the head reads one stride-8 cell. Two changes: (a) direct regression of the two end distances and
    the thickness (YOLO26 style, `box="direct"` default; DFL kept as an option for a one-factor comparison);
    (b) an edge-aligned refiner: for the best K candidates, the stride-4 trunk map (pretrained, 2x the head's
    resolution) is sampled bilinearly ON the predicted long edges, at the two ends and along the centre line, in the
    box's own frame, and a small MLP predicts residuals of the ends, the log thickness, the across offset and the
    angle (zero-initialised: the identity at the start). Box only, no class, no relations: OpenOBB8's relational
    stage never raised accuracy (F43-F50) and cost 35% of the time; this one costs ~0.3% of the FLOPs.
  * The head is heavy and compute-bound like YOLO26's (20% of FLOPs vs 5% in v3; F37 N6): 1x1 + two dense 3x3
    RepConv per tower, the second class conv dilated for context. Dual head as OpenOBB12 (one-to-many for learning,
    one-to-one for inference; +0.03-0.05 measured, F54), class balancing of OpenOBB10, calibrated head / NMS.

Stable interface (trainer, predict, pretrain-style tools):
    OpenOBB14(size, num_classes, img_size, max_det=1000, nms_iou=0.7, off=None, box="direct", ref_k=1500)
    .encode(x) -> (f4, [P3, P4, P5]); .features(x) -> [P3, P4, P5]; .head / .head_o2o(feats) -> (cls, reg)
    .anchors(feats); .refine(feats_or_f4, ...); forward: training -> dict for OpenOBB14Loss, eval -> OpenOBB3 format
    .infer_head ("o2m" / "o2o"), .final_nms, .copy_head_to_o2o(), .fuse()
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder
from .openobb1 import CONFIGS
from .openobb3 import DIRECT_CH, REG_CH, OpenOBB3, decode, fast_nms
from .openobb6 import ConvBN, GlobalAttention
from .openobb7 import GUARD_TAU, ScaleGuard, fuse_model, space_to_depth
from .openobb8 import _sample, jitter_targets, refine_boxes
from .openobb9 import Down, RepConv, RepCSP

PARTS14 = ("attn", "spd", "edge", "dilate", "maxpool", "guard", "dual")
# neck width, head tower width, RepConv per neck block
SCALES14 = {"s": dict(hid=192, head=128, depth=1), "m": dict(hid=256, head=160, depth=2),
            "l": dict(hid=256, head=192, depth=3), "x": dict(hid=384, head=256, depth=3)}
DILATIONS14 = (1, 2, 1, 3)
N_ALONG = 6              # samples along each long edge / the centre line
N_END = 3                # samples across each end
REF_K_TRAIN = 256        # dense candidates refined per image in training (plus as many jittered labelled boxes)
BOX_MODES = ("direct", "dfl")
NECKS = ("own", "dfine")   # own: Neck14 (random init); dfine: the COCO-pretrained hybrid encoder of D-FINE (F60 control)


def parse_off14(off):
    """'attn,edge' / ['attn'] / None -> frozenset of switched-off OpenOBB14 parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS14]
    if bad:
        raise ValueError(f"unknown OpenOBB14 part(s) {bad}; parts: {', '.join(PARTS14)}")
    return frozenset(names)


class Neck14(nn.Module):
    """1x1 projections of the trunk maps -> [attention at /32] -> top-down (RepCSP) -> bottom-up (Down + RepCSP).
    Stride-4 detail joins P3 by space-to-depth. Every block is dense 3x3 (re-parameterised) or 1x1: compute-bound."""

    def __init__(self, cin, c4, hid, n, off=frozenset()):
        super().__init__()
        dil = (1,) if "dilate" in off else DILATIONS14
        mp = "maxpool" not in off
        self.proj = nn.ModuleList(ConvBN(c, hid, 1, act=False) for c in cin)
        self.attn = nn.Identity() if "attn" in off else GlobalAttention(hid, heads=8)
        self.spd = None if "spd" in off else ConvBN(4 * c4, hid // 2, 1)
        self.lat5 = ConvBN(hid, hid, 1)
        self.td4 = RepCSP(2 * hid, hid, n, dil)
        self.lat4 = ConvBN(hid, hid, 1)
        self.td3 = RepCSP(2 * hid + (0 if self.spd is None else hid // 2), hid, n, dil)
        self.dn3, self.bu4 = Down(hid, hid, mp), RepCSP(2 * hid, hid, n, dil)
        self.dn4, self.bu5 = Down(hid, hid, mp), RepCSP(2 * hid, hid, n, (1,))

    def forward(self, f4, c3, c4, c5):
        t5 = self.lat5(self.attn(self.proj[2](c5)))
        n4 = self.td4(torch.cat([self.proj[1](c4), F.interpolate(t5, size=c4.shape[-2:], mode="nearest")], 1))
        t4 = self.lat4(n4)
        parts = [self.proj[0](c3), F.interpolate(t4, size=c3.shape[-2:], mode="nearest")]
        if self.spd is not None:
            parts.append(self.spd(space_to_depth(f4, c3.shape[-2:])))
        p3 = self.td3(torch.cat(parts, 1))
        p4 = self.bu4(torch.cat([self.dn3(p3), t4], 1))
        p5 = self.bu5(torch.cat([self.dn4(p4), t5], 1))
        return [p3, p4, p5]


class Tower(nn.Module):
    """1x1 (cin -> ch) -> RepConv 3x3 -> RepConv 3x3 (dilation d) -> 1x1 out."""

    def __init__(self, cin, ch, cout, d=1):
        super().__init__()
        self.inp = ConvBN(cin, ch, 1)
        self.c1 = RepConv(ch, ch, 1)
        self.c2 = RepConv(ch, ch, d)
        self.out = nn.Conv2d(ch, cout, 1)

    def forward(self, x):
        return self.out(self.c2(self.c1(self.inp(x))))


class Head14(nn.Module):
    """Per level: a class tower (second conv dilated: context) and a box tower (angle, across offset, two end
    distances, thickness: values with box="direct" (DIRECT_CH), distributions with "dfl" (REG_CH))."""

    def __init__(self, ch_in, nc, ch, reg_ch=DIRECT_CH, prior=0.01, dil=2):
        super().__init__()
        self.cls = nn.ModuleList(Tower(c, ch, nc, dil) for c in ch_in)
        self.box = nn.ModuleList(Tower(c, ch, reg_ch, 1) for c in ch_in)
        for m in self.cls:
            nn.init.constant_(m.out.bias, -math.log((1 - prior) / prior))
        for m in self.box:
            nn.init.zeros_(m.out.weight)
            nn.init.zeros_(m.out.bias)

    def forward(self, feats):
        cat = lambda mods: torch.cat([m(f).flatten(2) for m, f in zip(mods, feats)], 2).transpose(1, 2).float()  # noqa: E731
        return cat(self.cls), cat(self.box)


class EdgeRefiner(nn.Module):
    """Box residuals from features sampled in the box's own frame: the stride-4 trunk map (projected to `ce`
    channels) on the two long edges (N_ALONG points each), at the two ends (N_END points across each) and along the
    centre line; P3 (projected) along the centre line. A MLP maps the samples + the box geometry to residuals applied
    by openobb8.refine_boxes (ends in units of clamp(L / 10, 2, 32) px, log thickness, across offset in units of
    max(T / 2, 1) px, angle 0.2 tanh). The last layer starts at zero: the refiner is the identity at the start."""

    def __init__(self, c4, c3, ce=64, hidden=256, s4=4, s3=8):
        super().__init__()
        self.s4, self.s3 = s4, s3                      # strides (input px) of the detail map and of P3
        self.p4 = ConvBN(c4, ce, 1, act=False)
        self.p3 = ConvBN(c3, ce, 1, act=False)
        u = (torch.arange(N_ALONG) + 0.5) / N_ALONG - 0.5
        v_end = (torch.arange(N_END) + 0.5) / N_END / 2 - 0.25
        pts = [torch.stack([u, torch.full_like(u, -0.5)], -1), torch.stack([u, torch.full_like(u, 0.5)], -1),
               torch.stack([torch.full_like(v_end, -0.5), v_end], -1),
               torch.stack([torch.full_like(v_end, 0.5), v_end], -1),
               torch.stack([u, torch.zeros_like(u)], -1)]
        self.register_buffer("offsets", torch.cat(pts), persistent=False)       # (P, 2): (u in L units, v in T units)
        n_in = len(self.offsets) * ce + N_ALONG * ce + 4
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, hidden),
                                 nn.SiLU(inplace=True), nn.Linear(hidden, 5))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def points(self, b):
        """b (B, K, 5) px -> (B, K, P, 2) sample points (image px)."""
        c, L, T, th = b[..., None, :2], b[..., None, 2:3], b[..., None, 3:4], b[..., 4]
        ax = torch.stack([th.cos(), th.sin()], -1)[..., None, :]
        nx = torch.stack([-th.sin(), th.cos()], -1)[..., None, :]
        o = self.offsets
        return c + o[:, 0:1] * L * ax + o[:, 1:2] * T * nx

    def forward(self, f4, p3, b):
        """f4 (B, c4, H/4, W/4), p3 (B, c3, H/8, W/8), candidate boxes b (B, K, 5) px -> refined boxes (B, K, 5)."""
        b = b.detach().float()
        B, K, _ = b.shape
        pts = self.points(b)
        s4 = _sample(self.p4(f4), pts.reshape(B, -1, 2), self.s4).reshape(B, K, -1)
        s3 = _sample(self.p3(p3), pts[:, :, -N_ALONG:].reshape(B, -1, 2), self.s3).reshape(B, K, -1)
        geo = torch.stack([torch.log1p(b[..., 2]) / 7, torch.log1p(b[..., 3]) / 4, torch.sin(2 * b[..., 4]),
                           torch.cos(2 * b[..., 4])], -1)
        d = self.mlp(torch.cat([s4, s3, geo], -1).to(self.mlp[0].weight.dtype)).float()
        return refine_boxes(b, d)


class OpenOBB14(nn.Module):
    arch = "v14"
    anchors = OpenOBB3.anchors

    def __init__(self, size="x", num_classes=15, img_size=1024, max_det=1000, nms_iou=0.7, off=None, box="direct",
                 ref_k=1500, final_nms=None, neck="own", **_):
        super().__init__()
        if box not in BOX_MODES:
            raise ValueError(f"box must be one of {BOX_MODES}, got {box!r}")
        if neck not in NECKS:
            raise ValueError(f"neck must be one of {NECKS}, got {neck!r}")
        self.neck_kind = neck
        cfg = copy.deepcopy(CONFIGS[size])
        self.size, self.nc, self.img_size, self.box = size, num_classes, img_size, box
        self.max_det, self.nms_iou = max_det, nms_iou
        self.off = parse_off14(off)
        bb = dict(freeze_at=-1, freeze_norm=False)
        bb.update(cfg["backbone"])
        self.backbone = HGNetv2(**bb, pretrained=False)          # D-FINE COCO weights load as they are ("backbone.*")
        self.guard = None if "guard" in self.off else ScaleGuard(GUARD_TAU)
        cin = cfg["encoder"]["in_channels"]
        c4 = self.backbone._out_channels[0]
        sc = SCALES14[size]
        if neck == "dfine":
            # F60 control: the same trunk, head and refiner on D-FINE's COCO-pretrained hybrid encoder ("encoder.*"
            # parameter names, so its weights load as they are) instead of the randomly initialised Neck14
            self.hid = cfg["encoder"]["hidden_dim"]
            self.neck = None
            self.encoder = HybridEncoder(**cfg["encoder"], eval_spatial_size=[img_size, img_size])
        else:
            self.hid = sc["hid"]
            self.neck = Neck14(cin, c4, self.hid, sc["depth"], self.off)
        reg_ch = DIRECT_CH if box == "direct" else REG_CH
        self.head = Head14([self.hid] * 3, num_classes, sc["head"], reg_ch)
        self.head_o2o = None if "dual" in self.off else copy.deepcopy(self.head)
        self.edge = None if "edge" in self.off else EdgeRefiner(c4, self.hid)
        self.ref_k, self.ref_k_train = int(ref_k), REF_K_TRAIN
        self.strides = (8, 16, 32)
        self._grid = {}
        # inference head, calibrated on val at the end of training (args["infer_head"]): "o2m" (+ NMS) or "o2o"
        self.infer_head = "o2m"
        self.final_nms = final_nms                 # one-to-one head: None = NMS-free, else a ProbIoU threshold

    @torch.no_grad()
    def copy_head_to_o2o(self):
        """Warm start of the one-to-one head from the one-to-many head (after loading a single-head checkpoint)."""
        if self.head_o2o is not None:
            self.head_o2o.load_state_dict(self.head.state_dict())

    def encode(self, x):
        """-> (stride-4 trunk map (guarded), [P3, P4, P5] neck outputs with `hid` channels)."""
        bb = self.backbone
        h = bb.stem(x)
        outs = []
        for stage in bb.stages:
            h = stage(h)
            outs.append(h if self.guard is None else self.guard(h))
        f4, c3, c4, c5 = outs
        if self.neck is None:
            return f4, list(self.encoder([c3, c4, c5]))
        return f4, self.neck(f4, c3, c4, c5)

    def features(self, x):
        return self.encode(x)[1]

    def run_head(self, feats, o2o):
        return (self.head_o2o if o2o and self.head_o2o is not None else self.head)(feats)

    def refine(self, f4, p3, cls, boxes, k):
        """Refine the k best-scoring candidates in place of their dense boxes -> (boxes (B, A, 5), idx (B, k))."""
        k = min(int(k), cls.shape[1])
        idx = cls.detach().sigmoid().amax(-1).topk(k, dim=1).indices
        cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
        ref = self.edge(f4, p3, cand)
        return boxes.scatter(1, idx[..., None].expand(-1, -1, 5), ref.to(boxes.dtype)), idx

    def forward(self, x, targets=None, ctx=None):
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        if self.training:
            cls, reg = self.run_head(feats, False)
            out = {"cls": cls, "reg": reg, "points": pts, "strides": st}
            if self.head_o2o is not None:
                out["o2o_cls"], out["o2o_reg"] = self.run_head(feats, True)
            if self.edge is not None:
                boxes = decode(reg.detach(), pts, st)
                k = min(self.ref_k_train, boxes.shape[1])
                idx = cls.detach().sigmoid().amax(-1).topk(k, dim=1).indices
                cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
                cand, valid = self._with_jitter(cand, targets)
                out.update(ref_boxes=self.edge(f4, feats[0], cand), ref_cand=cand, ref_valid=valid)
            return out
        o2o = self.infer_head != "o2m" and self.head_o2o is not None
        cls, reg = self.run_head(feats, o2o)
        boxes = decode(reg, pts, st)
        conf = getattr(self, "nms_conf", 0.001)
        if self.edge is not None and self.ref_k > 0:
            boxes, _ = self.refine(f4, feats[0], cls, boxes, self.ref_k)
        nms = self.final_nms if o2o else self.nms_iou           # one-to-many: always NMS
        if nms is not None:
            logits, boxes = fast_nms(cls, boxes, self.max_det, conf=conf, iou=float(nms))
        else:
            k = min(self.max_det, cls.shape[1])
            order = cls.amax(-1).topk(k, dim=1).indices
            logits = cls.gather(1, order[..., None].expand(-1, -1, cls.shape[-1])).float()
            boxes = boxes.gather(1, order[..., None].expand(-1, -1, 5)).float()
            logits = logits.masked_fill(logits.sigmoid().amax(-1, keepdim=True) < conf, -30.0)
        boxes = boxes.clone()
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}

    def _with_jitter(self, cand, targets):
        """Append jittered labelled boxes (<= ref_k_train per image) to the candidates; padding rows are invalid.
        Rare classes the dense head has not found yet still teach the refiner (OpenOBB8 idea)."""
        B, K, _ = cand.shape
        valid = torch.ones((B, K), dtype=torch.bool, device=cand.device)
        if targets is None:
            return cand, valid
        jit = jitter_targets(targets, self.img_size)
        jit = [j[torch.randperm(len(j), device=j.device)[:self.ref_k_train]] if len(j) > self.ref_k_train else j
               for j in jit]
        n = max(len(j) for j in jit)
        if n == 0:
            return cand, valid
        pad = cand.new_zeros((B, n, 5))
        pad[..., 2:4] = 1.0
        ok = torch.zeros((B, n), dtype=torch.bool, device=cand.device)
        for i, j in enumerate(jit):
            pad[i, :len(j)] = j.to(cand.dtype)
            ok[i, :len(j)] = True
        return torch.cat([cand, pad], 1), torch.cat([valid, ok], 1)

    @torch.no_grad()
    def fuse(self):
        """Inference only: merge every RepConv into one 3x3 conv, then fold every BatchNorm into its convolution."""
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse()
        return fuse_model(self)
