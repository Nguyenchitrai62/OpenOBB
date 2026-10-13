"""OpenOBB8: the pretrained OpenOBB7 core + an oriented relational refiner (a second reasoning stage of OpenOBB's own).

Why (research/LEDGER.md F31-F40, Wall_Color, Ultralytics-style mAP):
  * OpenOBB3 (COCO-pretrained HGNetv2 + hybrid encoder + dense oriented head) 0.720 / 0.474, YOLO26x 0.746 / 0.499.
    A new network trained from scratch (OpenOBB6) stalls near 0.55 / 0.32 (F40): on small data the pretrained prior
    decides. OpenOBB8 therefore keeps the pretrained core (OpenOBB7) and puts the new design into a second stage
    that attacks the two measured failure modes:
      1. tiny look-alikes that only their neighbourhood separates: wall_300 (stubs ~10 x 6 px beside door frames)
         predicted as junction 32-49% (F35, F39); median distance wall_300 -> nearest junction 3.9 px, -> door 16 px;
         75-93% of the evidence for a class logit lies outside the box (F37);
      2. box fit of thin objects (mAP50-95 gap; walls 4-6 px thick regressed from stride-8 features, F32).
  * OpenOBB5's relation re-scorer (single-point features, absolute positions) did not help (F35). OpenOBB8 differs:
    each candidate is re-read in its own rotated frame (stride-4 detail across its edges + a ring of context one box
    length away), and candidates attend to each other with an attention bias computed from their RELATIVE geometry
    (j's box expressed in i's frame), so an arrangement ("a stub next to a door swing", "a block where walls cross")
    is recognised at any orientation. Nothing assumes axis-aligned objects.

Stages:
  1. candidates: the K distinct best anchors (distinct_topk on the max class score, pre = 2K); K = rel_k at inference,
     K / 2 in training plus jittered copies of the labelled boxes (rare classes the dense head misses still teach);
  2. oriented sampling (bilinear): inner grid 8 along x 3 across (across rows just outside the two long edges) from
     the stride-4 backbone map + P3; context ring of 18 points about one box length away from P3 + P4;
  3. per-candidate local transformer (2 layers, pre-norm) over [geometry token + 42 samples] -> one token;
  4. relation layer(s) between candidates with the relative-geometry attention bias (pair_geometry -> small MLP ->
     one additive bias per head, scaled_dot_product_attention with a float mask);
  5. zero-initialised outputs: residual class logits (refined = detached base logits + delta) and box residuals in
     the candidate frame (two end points, log thickness, across offset, small angle).
At initialisation OpenOBB8's output equals OpenOBB7's exactly (tests/test_openobb8.py); `off=` switches parts off.

Stable interface of the core used here (openobb/models/openobb7.py): encode(x) -> (f4, [P3, P4, P5]); head(feats);
anchors(feats); decode / fast_nms of OpenOBB3; attributes img_size, max_det, nms_iou.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openobb.ops.obb_torch import probiou

from .obb_decoder import distinct_topk
from .openobb3 import decode, fast_nms
from .openobb7 import OpenOBB7

PARTS8 = ("relate", "geo_bias", "ring", "p2", "jitter", "refine")
IN_U, IN_V, N_RING = 8, 3, 18
PAIR_CH = 7
SPREAD_IOU = 0.7         # anchors overlapping a candidate this much (class-agnostic) get its logit correction
SPREAD_TOP = 4000        # ... among the best SPREAD_TOP anchors (the ones that can reach the output)
NEG = -1e4               # blocked attention pairs (finite: safe in fp16, no NaN when a row is fully blocked)


def parse_off8(off):
    """'ring,p2' / ['ring'] / None -> frozenset of switched-off OpenOBB8 parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS8]
    if bad:
        raise ValueError(f"unknown OpenOBB8 part(s) {bad}; parts: {', '.join(PARTS8)}")
    return frozenset(names)


def _slog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def pair_geometry(b):
    """b (B, K, 5) px boxes -> (B, K, K, PAIR_CH): box j expressed in the frame of box i (row i, column j).

    Offsets along / across i's axis normalised by i's length / thickness, kept only up to a joint sign (the axis of a
    box is defined up to pi, which flips both); relative angle as sin / cos of 2 dtheta; log length / thickness ratios.
    Invariant to rotating (and translating) the whole candidate set and to theta -> theta + pi of any box."""
    b = b.float()
    c, L, T, th = b[..., :2], b[..., 2].clamp(min=1.0), b[..., 3].clamp(min=1.0), b[..., 4]
    ax = torch.stack([th.cos(), th.sin()], -1)
    nx = torch.stack([-ax[..., 1], ax[..., 0]], -1)
    d = c[:, None, :, :] - c[:, :, None, :]                       # (B, i, j, 2) = c_j - c_i
    u = (d * ax[:, :, None, :]).sum(-1) / L[:, :, None]
    v = (d * nx[:, :, None, :]).sum(-1) / T[:, :, None]
    dth = 2 * (th[:, None, :] - th[:, :, None])
    return torch.stack([torch.log1p(u.abs()), torch.log1p(v.abs()), _slog(u * v) / 4, dth.sin(), dth.cos(),
                        torch.log(L[:, None, :] / L[:, :, None]), torch.log(T[:, None, :] / T[:, :, None])], -1)


def _sample(fmap, pts, stride):
    """fmap (B, C, h, w) at `stride`, pts (B, N, 2) image px -> (B, N, C) bilinear samples (fp32).
    Pixel p lies at feature index p / stride - 0.5 (cell centres at (i + 0.5) * stride, as the anchors)."""
    h, w = fmap.shape[-2:]
    g = torch.stack([pts[..., 0] / (stride * w), pts[..., 1] / (stride * h)], -1) * 2 - 1
    out = F.grid_sample(fmap.float(), g.float()[:, :, None, :], mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return out[..., 0].transpose(1, 2)


class RelationLayer(nn.Module):
    """Pre-norm self-attention between candidates + FFN; attention logits get an additive bias per head."""

    def __init__(self, d, heads, ffn=2):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Linear(ffn * d, d))

    def forward(self, x, bias):
        B, K, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, K, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(B, K, D))
        return x + self.ffn(self.n2(x))


class OrientedRelationalRefiner(nn.Module):
    def __init__(self, ch, ch_p2, nc, d=192, heads=6, local_layers=2, rel_layers=1, pair_hidden=32, k=600,
                 off=frozenset()):
        super().__init__()
        self.k, self.nc, self.heads = k, nc, heads
        self.use_p2, self.use_ring = "p2" not in off, "ring" not in off
        self.relate, self.geo_bias = "relate" not in off, "geo_bias" not in off
        self.in3 = nn.Linear(ch, d)
        self.in2 = nn.Linear(ch_p2, d) if self.use_p2 else None
        self.ring3 = nn.Linear(ch, d) if self.use_ring else None
        self.ring4 = nn.Linear(ch, d) if self.use_ring else None
        u = torch.linspace(-0.55, 0.55, IN_U)
        v = torch.tensor([-1.0, 0.0, 1.0])
        inner = torch.stack(torch.meshgrid(u, v, indexing="ij"), -1).reshape(-1, 2)          # (24, 2)
        ur = torch.linspace(-1, 1, 8)
        ring = torch.cat([torch.stack([ur, torch.full_like(ur, s)], -1) for s in (-1.0, 1.0)]
                         + [torch.tensor([[-1.0, 0.0], [1.0, 0.0]])])                         # (18, 2)
        self.register_buffer("inner", inner, persistent=False)
        self.register_buffer("ring", ring, persistent=False)
        self.pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.geo = nn.Sequential(nn.Linear(4 + nc, d), nn.GELU(), nn.Linear(d, d))
        self.token = nn.Parameter(torch.zeros(1, 1, d))
        self.local = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu",
                                                              batch_first=True, norm_first=True)
                                   for _ in range(local_layers))
        if self.relate:
            self.rel = nn.ModuleList(RelationLayer(d, heads) for _ in range(rel_layers))
            if self.geo_bias:      # shared pair encoder (hidden kept small: K^2 x hidden floats), one head per layer
                self.pair = nn.Sequential(nn.Linear(PAIR_CH, pair_hidden), nn.GELU())
                self.pair_out = nn.ModuleList(nn.Linear(pair_hidden, heads) for _ in range(rel_layers))
                for m in self.pair_out:                # bias starts at 0: plain attention at step 0
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            else:                  # ablation: absolute position embedding instead of the relative-geometry bias
                self.abs_pos = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.cls_out = nn.Linear(d, nc)
        self.box_out = nn.Linear(d, 5)
        for m in (self.cls_out, self.box_out):          # residual outputs start at zero: identity at step 0
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def points(self, b):
        """b (B, K, 5) px -> inner (B, K, 24, 2) and ring (B, K, 18, 2) sample positions in px (rotated frames)."""
        c, L, T, th = b[..., None, :2], b[..., 2:3, None], b[..., 3:4, None], b[..., 4]
        ax = torch.stack([th.cos(), th.sin()], -1)[..., None, :]
        nx = torch.stack([-th.sin(), th.cos()], -1)[..., None, :]
        inner = c + self.inner[:, 0:1] * L * ax + self.inner[:, 1:2] * (T / 2 + 2.0) * nx
        m = L.clamp(32.0, 192.0)                        # context margin: about one box length
        ring = c + self.ring[:, 0:1] * (L / 2 + m) * ax + self.ring[:, 1:2] * (T / 2 + m / 2) * nx
        return inner, ring

    def forward(self, f4, p3, p4, base_logits, boxes, img_size, block=None):
        """Candidates boxes (B, K, 5) px with their base logits (B, K, C) -> refined logits, refined boxes px.
        block (B, K, K) bool: pairs that may not attend (training: padding / jittered keys)."""
        B, K, _ = boxes.shape
        b = boxes.detach().float()
        base = base_logits.detach().float()
        inner, ring = self.points(b)
        ip = inner.reshape(B, -1, 2)
        f_in = self.in3(_sample(p3, ip, 8))
        if self.in2 is not None:
            f_in = f_in + self.in2(_sample(f4, ip, 4))
        toks = [f_in.view(B, K, IN_U * IN_V, -1)]
        pos = [F.pad(self.inner, (0, 1))]
        if self.use_ring:
            rp = ring.reshape(B, -1, 2)
            toks.append((self.ring3(_sample(p3, rp, 8)) + self.ring4(_sample(p4, rp, 16))).view(B, K, N_RING, -1))
            pos.append(F.pad(self.ring, (0, 1), value=1.0))
        tok = torch.cat(toks, 2) + self.pos(torch.cat(pos))
        geo = torch.cat([torch.log1p(b[..., 2:3]) / 7, torch.log1p(b[..., 3:4]) / 4, torch.sin(2 * b[..., 4:5]),
                         torch.cos(2 * b[..., 4:5]), base.sigmoid()], -1)
        x = torch.cat([(self.token + self.geo(geo))[..., None, :].expand(B, K, 1, -1), tok], 2).flatten(0, 1)
        for blk in self.local:
            x = blk(x)
        x = x[:, 0].reshape(B, K, -1)
        if self.relate:
            mask = x.new_zeros((B, 1, K, K), dtype=torch.float32)
            if block is not None:
                mask = mask.masked_fill(block[:, None], NEG)
            if self.geo_bias:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    ph = self.pair(pair_geometry(b))                                     # (B, K, K, hidden) fp32
                    biases = [mask + m(ph).permute(0, 3, 1, 2) for m in self.pair_out]   # (B, heads, K, K)
            else:
                rp = torch.cat([b[..., :2] / img_size, torch.sin(2 * b[..., 4:5]), torch.cos(2 * b[..., 4:5])], -1)
                x = x + self.abs_pos(rp)
                biases = [mask.expand(B, self.heads, K, K)] * len(self.rel)
            for layer, bias in zip(self.rel, biases):
                x = layer(x, bias)
        x = self.norm(x)
        return base + self.cls_out(x).float(), refine_boxes(b, self.box_out(x).float())


def refine_boxes(b, d):
    """Apply residuals d (B, K, 5) = (back end, front end, log thickness, across offset, angle) to boxes b px, in each
    box's own frame. End moves in units of clamp(L / 10, 2, 32) px, across offset in units of max(T / 2, 1) px.
    d = 0 returns b exactly (the identity at initialisation)."""
    L, T, th = b[..., 2], b[..., 3], b[..., 4]
    s_al = (L / 10).clamp(2.0, 32.0)
    length = torch.maximum(L + (d[..., 0] + d[..., 1]) * s_al, torch.minimum(L, torch.ones_like(L)))
    du = (d[..., 1] - d[..., 0]) * s_al / 2
    thick = T * torch.exp(d[..., 2].clamp(-1.5, 1.5))
    across = d[..., 3] * (T / 2).clamp(min=1.0)
    c, s = th.cos(), th.sin()
    cx = b[..., 0] + du * c - across * s
    cy = b[..., 1] + du * s + across * c
    return torch.stack([cx, cy, length, thick, th + 0.2 * torch.tanh(d[..., 4])], -1)


@torch.no_grad()
def apply_refinement(cls, boxes, idx, cand, delta, ref_logits, ref_boxes, spread_iou=SPREAD_IOU, top=SPREAD_TOP):
    """Write the refiner's output into the dense outputs before NMS.
    cls (B, A, C), boxes (B, A, 5) px: dense outputs; idx (B, K) candidate anchors, cand (B, K, 5) their boxes;
    delta (B, K, C) = refined - base logits; ref_logits / ref_boxes (B, K, .) the refined candidates.
    The candidates were chosen distinct (distinct_topk), so each stands for a cluster of near-duplicate anchors.
    Correcting only the candidate would let a duplicate keep the uncorrected score and win class-wise NMS (a
    stub re-classified from junction to wall_300 would come back as a junction from its neighbour anchor). So the
    candidate's logit correction is also added to every anchor among the `top` best that overlaps it with ProbIoU
    > spread_iou (its best-overlapping candidate). With delta = 0 the output is unchanged (identity at init)."""
    B, A, C = cls.shape
    cls = cls.clone()
    if spread_iou is not None and A > 0:
        m = min(top, A)
        ti = cls.amax(-1).topk(m, dim=1).indices                                          # (B, m)
        tb = boxes.gather(1, ti[..., None].expand(-1, -1, 5)).float()
        ov = probiou(tb[:, :, None, :], cand[:, None, :, :].float())                      # (B, m, K)
        best, bi = ov.max(-1)
        d = delta.gather(1, bi[..., None].expand(-1, -1, C)) * (best > spread_iou)[..., None]
        cls.scatter_add_(1, ti[..., None].expand(-1, -1, C), d.to(cls.dtype))
    cls = cls.scatter(1, idx[..., None].expand(-1, -1, C), ref_logits.to(cls.dtype))   # candidates: exact values
    boxes = boxes.scatter(1, idx[..., None].expand(-1, -1, 5), ref_boxes.to(boxes.dtype))
    return cls, boxes


def jitter_targets(targets, img_size, gen=None):
    """Labelled boxes (normalised) -> jittered candidate boxes px per image (list of (N, 5)) for refiner training:
    +-15% of the length along the axis, +-half the thickness across, x[0.8, 1.2] length, x[e^-0.4, e^0.4] thickness,
    +-0.05 rad."""
    out = []
    for t in targets:
        g = t["boxes"].float().clone()
        if not len(g):
            out.append(g.new_zeros((0, 5)))
            continue
        g[:, :4] *= img_size
        r = torch.rand(len(g), 5, device=g.device, generator=gen) * 2 - 1
        c, s = g[:, 4].cos(), g[:, 4].sin()
        du = r[:, 0] * 0.15 * g[:, 2]
        dv = r[:, 1] * 0.5 * g[:, 3].clamp(min=2.0)
        j = g.clone()
        j[:, 0] = g[:, 0] + du * c - dv * s
        j[:, 1] = g[:, 1] + du * s + dv * c
        j[:, 2] = g[:, 2] * (1 + 0.2 * r[:, 2])
        j[:, 3] = g[:, 3] * torch.exp(0.4 * r[:, 3])
        j[:, 4] = g[:, 4] + 0.05 * r[:, 4]
        out.append(j)
    return out


class OpenOBB8(OpenOBB7):
    arch = "v8"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 off7=None, rel_k=600, **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off7)
        self.off8 = parse_off8(off)
        self.rel_k = rel_k
        self.spread_iou = SPREAD_IOU                   # None: refined logits reach the candidates only
        self.refiner = None
        if "refine" not in self.off8:
            self.refiner = OrientedRelationalRefiner(self.encoder.hidden_dim, self.backbone._out_channels[0],
                                                     num_classes, k=rel_k, off=self.off8)

    def forward(self, x, targets=None, ctx=None):
        if self.refiner is None:
            return super().forward(x, targets, ctx)
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        cls, reg = self.head(feats)
        boxes = decode(reg, pts, st)
        sc = cls.detach().sigmoid().amax(-1)
        k = min(self.refiner.k // 2 if self.training else self.refiner.k, sc.shape[1])
        idx = distinct_topk(sc, boxes.detach(), k, 2 * k)
        gather = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
        cand, base = gather(boxes).detach(), gather(cls)
        block = valid = None
        if self.training and targets is not None and "jitter" not in self.off8:
            cand, base, block, valid = self._with_jitter(cand, base, cls, feats, targets)
        ref_logits, ref_boxes = self.refiner(f4, feats[0], feats[1], base, cand, self.img_size, block)
        if self.training:
            out = {"cls": cls, "reg": reg, "points": pts, "strides": st,
                   "ref_logits": ref_logits, "ref_boxes": ref_boxes, "ref_cand": cand.detach()}
            if valid is not None:
                out["ref_valid"] = valid
            return out
        cls, boxes = apply_refinement(cls, boxes, idx, cand, ref_logits - base.detach().float(), ref_logits,
                                      ref_boxes, self.spread_iou)
        logits, boxes = fast_nms(cls, boxes, self.max_det, conf=getattr(self, "nms_conf", 0.001), iou=self.nms_iou,
                                 min_side=getattr(self, "nms_min_side", 0.0))
        boxes[..., :4] = boxes[..., :4] / self.img_size
        boxes[..., 4] = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
        return {"pred_logits": logits, "pred_boxes": boxes}

    def _with_jitter(self, cand, base, cls, feats, targets, return_idx=False):
        """Append jittered labelled boxes (at most K / 2 per image, random subset) as extra candidates (base logits
        read at the nearest P3 anchor).
        Attention mask so the relation layer sees at training what it sees at inference: no candidate attends to a
        jittered key (other than itself), and a jittered query does not attend to a dense candidate it overlaps
        with ProbIoU > 0.7 (inference candidates are distinct at that threshold). Padding rows are invalid.
        return_idx: also return (B, n) the labelled object of every jittered row (-1 on padding)."""
        jit = jitter_targets(targets, self.img_size)
        cap = max(self.refiner.k // 2, 1)              # at most K / 2 jittered boxes per image (memory)
        sel = [torch.randperm(len(j), device=j.device)[:cap] if len(j) > cap else
               torch.arange(len(j), device=j.device) for j in jit]
        jit = [j[s] for j, s in zip(jit, sel)]
        n = max(len(j) for j in jit)
        B, K, C = base.shape
        if n == 0:
            return (cand, base, None, None, None) if return_idx else (cand, base, None, None)
        h3, w3 = feats[0].shape[-2:]
        jc = cand.new_zeros((B, n, 5))
        jc[..., 2:4] = 1.0
        jb = base.new_full((B, n, C), -30.0)            # padding rows: score ~0, never a target
        valid = torch.ones((B, K + n), dtype=torch.bool, device=cand.device)
        for i, j in enumerate(jit):
            valid[i, K + len(j):] = False
            if not len(j):
                continue
            jc[i, :len(j)] = j
            gx = (j[:, 0] / 8).long().clamp(0, w3 - 1)          # P3 anchors come first, stride 8
            gy = (j[:, 1] / 8).long().clamp(0, h3 - 1)
            jb[i, :len(j)] = cls[i, gy * w3 + gx]
        cand, base = torch.cat([cand, jc], 1), torch.cat([base, jb], 1)
        N = K + n
        eye = torch.eye(N, dtype=torch.bool, device=cand.device)[None]
        is_jit = torch.zeros((B, N), dtype=torch.bool, device=cand.device)
        is_jit[:, K:] = True
        with torch.no_grad():
            dup = probiou(cand[:, :, None, :].float(), cand[:, None, :, :].float()) > 0.7
        block = (is_jit[:, None, :] | ~valid[:, None, :]) & ~eye
        block = block | (is_jit[:, :, None] & dup & ~is_jit[:, None, :] & ~eye)
        if return_idx:
            jit_gt = torch.full((B, n), -1, dtype=torch.long, device=cand.device)
            for i, s in enumerate(sel):
                jit_gt[i, :len(s)] = s
            return cand, base, block, valid, jit_gt
        return cand, base, block, valid
