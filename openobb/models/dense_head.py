"""Dense rotated head (OpenOBB's one-to-many branch) + rotated task-aligned assigner + loss.

Why: DETR-OBB decoders win the large / context classes on DOTA but lose small dense objects to
YOLO-style dense heads (RiO-DETR vs YOLO26 per-class: PL, SP, HC, SV). OpenOBB keeps both: a cheap
anchor-free dense head on the encoder's P3-P5 (trained one-to-many, giving many positives per object
and recall on tiny objects) next to the relation decoder (one-to-one). The head can be used as an
auxiliary loss only, as a second output fused at inference, or as the query source of the decoder.

All code here is OpenOBB's own (Apache-2.0); the ideas follow published work: TAL (TOOD), rotated
TAL + ProbIoU (PP-YOLOE-R), small-target-aware candidate expansion (YOLO26 STAL), near-square
angle penalty sin^2(2*dtheta) * exp(-ln^2(w/h)/lambda^2) (YOLO26 paper).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from openobb.ops.obb_torch import probiou, rotated_iou


class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def dw_block(c1, c2):
    return nn.Sequential(ConvBN(c1, c1, 3, g=c1), ConvBN(c1, c2, 1))


class DenseRotatedHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=15, strides=(8, 16, 32), width=128, img_size=1024, prior=0.01):
        super().__init__()
        self.nc, self.strides, self.img_size = num_classes, list(strides), img_size
        self.cls_towers = nn.ModuleList([nn.Sequential(dw_block(in_ch, width), dw_block(width, width)) for _ in strides])
        self.reg_towers = nn.ModuleList([nn.Sequential(dw_block(in_ch, width), dw_block(width, width)) for _ in strides])
        self.cls_out = nn.ModuleList([nn.Conv2d(width, num_classes, 1) for _ in strides])
        # 4 log-distances + raw angle (no squash, as YOLO26: every angle loss / assignment here is periodic)
        self.reg_out = nn.ModuleList([nn.Conv2d(width, 5, 1) for _ in strides])
        for c, r in zip(self.cls_out, self.reg_out):
            nn.init.constant_(c.bias, -math.log((1 - prior) / prior))
            nn.init.normal_(c.weight, std=0.01)
            nn.init.normal_(r.weight, std=0.01)
            nn.init.constant_(r.bias, 0.0)
        self._grid = {}

    def anchors(self, shapes, device, dtype):
        key = (tuple(shapes), device, dtype)
        if key not in self._grid:
            pts, st = [], []
            for (h, w), s in zip(shapes, self.strides):
                ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                        torch.arange(w, device=device, dtype=dtype), indexing="ij")
                pts.append(torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], -1).reshape(-1, 2) / self.img_size)
                st.append(torch.full((h * w,), s / self.img_size, device=device, dtype=dtype))
            self._grid = {key: (torch.cat(pts), torch.cat(st))}
        return self._grid[key]

    def forward(self, feats):
        cls, reg, shapes = [], [], []
        for i, f in enumerate(feats):
            b, _, h, w = f.shape
            shapes.append((h, w))
            cls.append(self.cls_out[i](self.cls_towers[i](f)).flatten(2).transpose(1, 2))
            reg.append(self.reg_out[i](self.reg_towers[i](f)).flatten(2).transpose(1, 2))
        cls = torch.cat(cls, 1).float()
        reg = torch.cat(reg, 1).float()
        pts, st = self.anchors(shapes, reg.device, reg.dtype)
        return {"dense_logits": cls, "dense_boxes": decode(reg, pts, st), "dense_points": pts, "dense_strides": st}


def decode(reg, pts, strides):
    d = reg[..., :4].clamp(min=-6.0, max=8.0).exp() * strides[None, :, None]   # l, t, r, b (normalised)
    t = reg[..., 4]
    c, s = torch.cos(t), torch.sin(t)
    du, dv = (d[..., 2] - d[..., 0]) / 2, (d[..., 3] - d[..., 1]) / 2
    cx = pts[None, :, 0] + du * c - dv * s
    cy = pts[None, :, 1] + du * s + dv * c
    return torch.stack([cx, cy, d[..., 0] + d[..., 2], d[..., 1] + d[..., 3], t], -1)


@torch.no_grad()
def rotated_tal(scores, boxes, pts, gt_labels, gt_boxes, topk=13, alpha=1.0, beta=6.0, min_side=16 / 1024,
                exact_iou=True):
    """Task-aligned assignment for one image.

    scores (A, C) sigmoid, boxes (A, 5), pts (A, 2); gt_labels (M,), gt_boxes (M, 5).
    Returns fg (A,) bool, gt_idx (A,), target_scores (A, C)."""
    A, C = scores.shape
    M = gt_boxes.shape[0]
    dev = scores.device
    if M == 0:
        return torch.zeros(A, dtype=torch.bool, device=dev), torch.zeros(A, dtype=torch.long, device=dev), \
            torch.zeros_like(scores)
    gx, gy, gw, gh, gt = gt_boxes.unbind(-1)
    c, s = torch.cos(gt), torch.sin(gt)
    dx = pts[None, :, 0] - gx[:, None]
    dy = pts[None, :, 1] - gy[:, None]
    u = dx * c[:, None] + dy * s[:, None]
    v = -dx * s[:, None] + dy * c[:, None]
    hw = gw.clamp(min=min_side)[:, None] / 2
    hh = gh.clamp(min=min_side)[:, None] / 2
    inside = (u.abs() <= hw) & (v.abs() <= hh)                               # (M, A)
    mi, ai = inside.nonzero(as_tuple=True)
    ov = torch.zeros((M, A), device=dev, dtype=boxes.dtype)
    if len(mi):
        if exact_iou:
            ov[mi, ai] = rotated_iou(boxes[ai].float(), gt_boxes[mi].float()).to(boxes.dtype)
        else:
            ov[mi, ai] = probiou(boxes[ai], gt_boxes[mi]).clamp(0, 1)
    sc = scores[:, gt_labels].T                                              # (M, A)
    # tiny centre prior breaks ties while predictions are still random
    centre = torch.exp(-(u ** 2 / hw ** 2 + v ** 2 / hh ** 2)) * 1e-6
    metric = (sc.pow(alpha) * ov.pow(beta) + centre) * inside
    k = min(topk, A)
    top = metric.topk(k, dim=1).indices
    mask = torch.zeros_like(inside)
    mask.scatter_(1, top, True)
    mask &= inside
    # an anchor claimed by several objects goes to the one it overlaps most
    multi = mask.sum(0) > 1
    if multi.any():
        best = (ov + centre).argmax(0)
        keep = F.one_hot(best, M).T.bool()
        mask = torch.where(multi[None, :], mask & keep, mask)
    fg = mask.any(0)
    gt_idx = mask.float().argmax(0)
    m = metric * mask
    pos_metric_max = m.amax(1, keepdim=True)
    pos_ov_max = (ov * mask).amax(1, keepdim=True)
    norm = (m * pos_ov_max / (pos_metric_max + 1e-9)).amax(0)                # (A,)
    target_scores = torch.zeros_like(scores)
    if fg.any():
        target_scores[fg, gt_labels[gt_idx[fg]]] = norm[fg].to(scores.dtype)
    return fg, gt_idx, target_scores


class DenseCriterion(nn.Module):
    def __init__(self, w_cls=1.0, w_box=2.5, w_angle=0.3, topk=13, alpha=1.0, beta=6.0, lam=3.0, exact_iou=True):
        super().__init__()
        self.w_cls, self.w_box, self.w_angle = w_cls, w_box, w_angle
        self.topk, self.alpha, self.beta, self.lam, self.exact = topk, alpha, beta, lam, exact_iou

    def forward(self, out, targets):
        logits, boxes, pts = out["dense_logits"], out["dense_boxes"], out["dense_points"]
        scores = logits.detach().sigmoid()
        tsc, fgs, gts = [], [], []
        for b, t in enumerate(targets):
            fg, gi, ts = rotated_tal(scores[b], boxes[b].detach(), pts, t["labels"], t["boxes"].float(),
                                     self.topk, self.alpha, self.beta, exact_iou=self.exact)
            tsc.append(ts)
            fgs.append(fg)
            gts.append(t["boxes"][gi].float() if len(t["boxes"]) else boxes.new_zeros((len(fg), 5)))
        tsc = torch.stack(tsc)
        fg = torch.stack(fgs)
        tgt = torch.stack(gts)
        norm = tsc.sum().clamp(min=1.0)
        # Quality focal loss (GFL / RTMDet): BCE to the soft target, modulated by |sigmoid - target|^2 so the
        # ~45k easy negatives per image do not swamp the loss (plain BCE starts at ~70 vs ~35 for the decoder)
        mod = (logits.detach().sigmoid() - tsc).abs().pow(2)
        l_cls = (F.binary_cross_entropy_with_logits(logits, tsc, reduction="none") * mod).sum() / norm
        if fg.any():
            w = tsc.sum(-1)[fg]
            p, g = boxes[fg], tgt[fg]
            l_box = ((1 - probiou(p, g)) * w).sum() / norm
            ratio = torch.log(g[:, 2].clamp(min=1e-6) / g[:, 3].clamp(min=1e-6))
            omega = torch.exp(-ratio.pow(2) / self.lam ** 2)
            l_ang = (torch.sin(2 * (p[:, 4] - g[:, 4])).pow(2) * omega * w).sum() / norm
        else:
            l_box = l_ang = boxes.sum() * 0
        return {"loss_dense_cls": self.w_cls * l_cls, "loss_dense_box": self.w_box * l_box,
                "loss_dense_angle": self.w_angle * l_ang}


@torch.no_grad()
def dense_predict(out, num_top=1000, score_thr=0.01, nms_thr=0.6, img_size=1024):
    """Per image: top-k (score, label, box px) after class-wise Fast-NMS on ProbIoU (GPU friendly)."""
    logits, boxes = out["dense_logits"], out["dense_boxes"]
    res = []
    for lg, bx in zip(logits, boxes):
        sc = lg.sigmoid()
        s, idx = sc.flatten().topk(min(num_top * 3, sc.numel()))
        keep = s > score_thr
        s, idx = s[keep], idx[keep]
        lab = idx % sc.shape[1]
        b = bx[idx // sc.shape[1]]
        if len(s):
            # Fast-NMS: drop a box if a higher-scored box of the same class overlaps it above nms_thr
            iou = probiou(b[:, None, :], b[None, :, :])
            same = lab[:, None] == lab[None, :]
            iou = torch.triu(iou * same, diagonal=1)
            keep = iou.amax(0) <= nms_thr
            s, lab, b = s[keep][:num_top], lab[keep][:num_top], b[keep][:num_top]
        b = b.clone()
        b[:, :4] *= img_size
        res.append((s, lab, b))
    return res
