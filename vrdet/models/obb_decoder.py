"""Oriented-box D-FINE decoder for VRDet.

Derived from DEIM / D-FINE (Apache-2.0):
  DEIM: Copyright (c) 2024 The DEIM Authors. D-FINE: Copyright (c) 2024 The D-FINE Authors.
VRDet changes (Apache-2.0): 5-d oriented references (cx, cy, w, h, theta); rotated FDR (4 edge
distributions in the box frame + 1 angle-residual distribution); rotated deformable sampling;
oriented query selection head; oriented contrastive denoising; representation-invariant
query positional features.
"""
import copy
import functools
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .deim_utils import bias_init_with_prob, deformable_attention_core_func_v2, get_activation
from vrdet.ops.obb_torch import probiou
from .obb_utils import (angle_project, box_to_unact, distance2obb, inverse_sigmoid, pos_features,
                        unact_to_box, weighting_function)


@torch.no_grad()
def distinct_topk(scores, boxes, k, pre, thr=0.7):
    """Top-k anchors after class-agnostic Fast-NMS on ProbIoU among the `pre` best (distinct queries, DDQ idea):
    a one-to-many dense head fires many near-duplicates per large object; without this the query budget is
    spent on duplicates and small objects starve."""
    pre = min(pre, scores.shape[1])
    s, idx = scores.topk(pre, dim=1)
    b = boxes.gather(1, idx[..., None].expand(-1, -1, 5)).float()
    iou = torch.triu(probiou(b[:, :, None, :], b[:, None, :, :]), diagonal=1)
    s = s.masked_fill(iou.amax(1) > thr, -1.0)
    return idx.gather(1, s.topk(min(k, pre), dim=1).indices)

N_DIST = 5      # 4 rotated edges + 1 angle residual
_DEBUG = bool(int(__import__('os').environ.get('VRDET_DEBUG', '0')))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RotatedMSDeformableAttention(nn.Module):
    """Multi-scale deformable attention whose sampling offsets live in the (rotated) box frame."""

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method='default',
                 offset_scale=0.5, rotate=True, ortho_heads=False):
        super().__init__()
        # ortho_heads: half of the heads sample in the frame turned by +90 deg (RiO-DETR's RROA idea)
        turn = [0.0] * (num_heads - num_heads // 2) + [math.pi / 2 * float(ortho_heads)] * (num_heads // 2)
        self.register_buffer('head_turn', torch.tensor(turn), persistent=False)
        self.embed_dim, self.num_heads, self.num_levels = embed_dim, num_heads, num_levels
        self.offset_scale, self.rotate = offset_scale, rotate
        num_points_list = num_points if isinstance(num_points, list) else [num_points] * num_levels
        assert len(num_points_list) == num_levels
        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))
        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        """reference_points: [bs, Lq, 1, 5] oriented boxes (normalised, theta in radians)."""
        bs, Len_q = query.shape[:2]
        off = self.sampling_offsets(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attw = F.softmax(self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list)), -1)
        nps = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        ref = reference_points[:, :, None, :, :]                       # bs Lq 1 1 5
        off = off * nps * self.offset_scale
        ou = off[..., 0] * ref[..., 2]
        ov = off[..., 1] * ref[..., 3]
        if self.rotate:
            th = ref[..., 4] + self.head_turn.to(ref.dtype)[None, None, :, None]
            c, s = torch.cos(th), torch.sin(th)
            dx, dy = ou * c - ov * s, ou * s + ov * c
        else:
            dx, dy = ou, ov
        loc = torch.stack([ref[..., 0] + dx, ref[..., 1] + dy], -1)
        if _DEBUG and not torch.isfinite(loc).all():
            raise FloatingPointError(f'non-finite sampling locations; ref finite={torch.isfinite(reference_points).all().item()}'
                                     f' off finite={torch.isfinite(off).all().item()}')
        return self.ms_deformable_attn_core(value, value_spatial_shapes, loc, attw, self.num_points_list)


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        init.constant_(self.gate.bias, bias_init_with_prob(0.5))
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gates = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0., activation='relu',
                 n_levels=4, n_points=4, cross_attn_method='default', rotate_sampling=True, ortho_heads=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = RotatedMSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method,
                                                       rotate=rotate_sampling, ortho_heads=ortho_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        q = k = target if query_pos_embed is None else target + query_pos_embed
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))
        target2 = self.cross_attn(target if query_pos_embed is None else target + query_pos_embed,
                                  reference_points, value, spatial_shapes)
        target = self.gateway(target, self.dropout2(target2))
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        target = target + self.dropout4(target2)
        return self.norm3(target.clamp(min=-65504, max=65504))


class LQE(nn.Module):
    """Localization quality estimator over the 5 distributions (D-FINE, extended)."""

    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super().__init__()
        self.k, self.reg_max = k, reg_max
        self.reg_conf = MLP(N_DIST * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, N_DIST, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(B, L, -1))


def integral(logits, project_edges, project_angle, reg_max):
    """(..., 5*(reg_max+1)) -> edges (..., 4), angle residual (...)."""
    shape = logits.shape[:-1]
    p = F.softmax(logits.reshape(*shape, N_DIST, reg_max + 1).float(), dim=-1)
    edges = (p[..., :4, :] * project_edges.to(p)).sum(-1)
    ang = (p[..., 4, :] * project_angle.to(p)).sum(-1)
    return edges, ang


class OBBTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, num_head, reg_max, reg_scale, up, eval_idx=-1, act='relu'):
        super().__init__()
        self.hidden_dim, self.num_layers, self.num_head = hidden_dim, num_layers, num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.lqe_layers = nn.ModuleList([LQE(4, 64, 2, reg_max, act=act) for _ in range(num_layers)])

    def value_op(self, memory, spatial_shapes):
        value = memory.reshape(memory.shape[0], memory.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def forward(self, target, ref_unact, memory, spatial_shapes, bbox_head, score_head, query_pos_head,
                pre_bbox_head, up, reg_scale, attn_mask=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, spatial_shapes)
        proj_e = weighting_function(self.reg_max, up, reg_scale)
        proj_a = angle_project(self.reg_max, device=memory.device)
        dec_boxes, dec_logits, dec_corners, dec_refs = [], [], [], []
        ref_detach = unact_to_box(ref_unact.float())          # box geometry always in fp32
        for i, layer in enumerate(self.layers):
            query_pos_embed = query_pos_head(pos_features(ref_detach)).clamp(min=-10, max=10)
            output = layer(output, ref_detach.unsqueeze(2), value, spatial_shapes, attn_mask, query_pos_embed)
            if i == 0:
                pre = pre_bbox_head(output).float()                       # 4 logit deltas + (cos, sin) of 2*dtheta
                pre_xywh = F.sigmoid(pre[..., :4] + inverse_sigmoid(ref_detach[..., :4]))
                pre_t = ref_detach[..., 4] + 0.5 * torch.atan2(pre[..., 5], pre[..., 4])
                pre_boxes = torch.cat([pre_xywh, pre_t.unsqueeze(-1)], -1)
                pre_scores = score_head[0](output).float()
                ref_init = pre_boxes.detach()
            pred_corners = bbox_head[i](output + output_detach).float() + pred_corners_undetach
            edges, dang = integral(pred_corners, proj_e, proj_a, self.reg_max)
            box = distance2obb(ref_init, edges, ref_init[..., 4] + dang, reg_scale)
            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output).float(), pred_corners)
                dec_logits.append(scores)
                dec_boxes.append(box)
                dec_corners.append(pred_corners)
                dec_refs.append(ref_init)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            ref_detach = box.detach()
            output_detach = output.detach()
        return (torch.stack(dec_boxes), torch.stack(dec_logits), torch.stack(dec_corners), torch.stack(dec_refs),
                pre_boxes, pre_scores)


def obb_denoising_group(targets, num_classes, num_queries, class_embed, num_denoising=100,
                        label_noise_ratio=0.5, box_noise_scale=1.0):
    """Contrastive denoising (RT-DETR/D-FINE) for oriented boxes: edge noise in the box frame, no angle noise."""
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None
    num_group = max(num_denoising // max_gt_num, 1)
    bs = len(num_gts)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 5], device=device)
    input_query_bbox[..., 2:4] = 0.01
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    for i in range(bs):
        n = num_gts[i]
        if n > 0:
            input_query_class[i, :n] = targets[i]['labels']
            input_query_bbox[i, :n] = targets[i]['boxes']
            pad_gt_mask[i, :n] = 1
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = (1 - negative_gt_mask).squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    num_denoising = int(max_gt_num * 2 * num_group)
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
    if box_noise_scale > 0:
        b = input_query_bbox
        w, h = b[..., 2], b[..., 3]
        edges = torch.stack([-w / 2, -h / 2, w / 2, h / 2], -1)          # box-frame xyxy around the centre
        diff = torch.stack([w, h, w, h], -1) * 0.5 * box_noise_scale
        rand_sign = torch.randint_like(edges, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(edges)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        edges = edges + rand_sign * rand_part * diff
        x1, y1 = torch.minimum(edges[..., 0], edges[..., 2]), torch.minimum(edges[..., 1], edges[..., 3])
        x2, y2 = torch.maximum(edges[..., 0], edges[..., 2]), torch.maximum(edges[..., 1], edges[..., 3])
        du, dv = (x1 + x2) / 2, (y1 + y2) / 2
        c, s = torch.cos(b[..., 4]), torch.sin(b[..., 4])
        cx = (b[..., 0] + du * c - dv * s).clamp(0, 1)
        cy = (b[..., 1] + du * s + dv * c).clamp(0, 1)
        nw = (x2 - x1).clamp(min=1e-4, max=1)
        nh = (y2 - y1).clamp(min=1e-4, max=1)
        input_query_bbox = torch.stack([cx, cy, nw, nh, b[..., 4]], -1)
    input_query_bbox_unact = box_to_unact(input_query_bbox)
    input_query_logits = class_embed(input_query_class)
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    dn_meta = {"dn_positive_idx": dn_positive_idx, "dn_num_group": num_group,
               "dn_num_split": [num_denoising, num_queries]}
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


class OBBDFINETransformer(nn.Module):
    def __init__(self, num_classes=15, hidden_dim=256, num_queries=300, feat_channels=(256, 256, 256),
                 feat_strides=(8, 16, 32), num_levels=3, num_points=(3, 6, 3), nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0., activation="relu", num_denoising=100, label_noise_ratio=0.5,
                 box_noise_scale=1.0, eval_spatial_size=None, eval_idx=-1, eps=1e-2, aux_loss=True,
                 cross_attn_method='default', reg_max=32, reg_scale=4., mlp_act='relu', rotate_sampling=True,
                 ortho_heads=False, o2m_queries=0):
        super().__init__()
        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim, self.nhead, self.feat_strides = hidden_dim, nhead, feat_strides
        self.num_levels, self.num_classes, self.num_queries = num_levels, num_classes, num_queries
        self.eps, self.num_layers, self.eval_spatial_size = eps, num_layers, eval_spatial_size
        self.aux_loss, self.reg_max = aux_loss, reg_max
        self.o2m_queries = o2m_queries          # H10: extra training-only query group with one-to-many targets
        self._build_input_proj_layer(list(feat_channels))
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
                                        list(num_points), cross_attn_method=cross_attn_method,
                                        rotate_sampling=rotate_sampling, ortho_heads=ortho_heads)
        self.decoder = OBBTransformerDecoder(hidden_dim, layer, num_layers, nhead, reg_max, self.reg_scale, self.up,
                                             eval_idx, act=activation)
        self.num_denoising, self.label_noise_ratio, self.box_noise_scale = num_denoising, label_noise_ratio, box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)
        self.enc_output = nn.Sequential(OrderedDict([('proj', nn.Linear(hidden_dim, hidden_dim)),
                                                     ('norm', nn.LayerNorm(hidden_dim))]))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 6, 3, act=mlp_act)      # 4 deltas + (cos, sin) of 2*theta
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 6, 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, N_DIST * (reg_max + 1), 3, act=mlp_act)
                                            for _ in range(num_layers)])
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors, persistent=False)
            self.register_buffer('valid_mask', valid_mask, persistent=False)
        self._reset_parameters(list(feat_channels))

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        for head in (self.enc_bbox_head, self.pre_bbox_head):
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)
            head.layers[-1].bias.data[4] = 1.0                     # (cos, sin) = (1, 0): zero angle / no rotation
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])))
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])))
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                proj_feats.append(self.input_proj[i](feats[-1]) if i == len_srcs else self.input_proj[i](proj_feats[-1]))
        feat_flatten, spatial_shapes = [], []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device='cpu'):
        if spatial_shapes is None:
            eval_h, eval_w = self.eval_spatial_size
            spatial_shapes = [[int(eval_h / s), int(eval_w / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_unact=None, dense=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        if dense is not None:      # VRDet H4c: queries come from the dense one-to-many head (distinct top-k)
            sc = dense["dense_logits"].detach().float().sigmoid().amax(-1) * valid_mask[..., 0].to(torch.float32)
            topk_ind = distinct_topk(sc, dense["dense_boxes"].detach(), self.num_queries, 3 * self.num_queries)
            topk_memory = output_memory.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            b = dense["dense_boxes"].detach().float().gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, 5))
            b = torch.cat([b[..., :2].clamp(0.001, 0.999), b[..., 2:4].clamp(1e-4, 0.999), b[..., 4:]], -1)
            enc_unact = box_to_unact(b)
            content = topk_memory.detach()
            if denoising_unact is not None:
                enc_unact = torch.concat([denoising_unact, enc_unact], dim=1)
                content = torch.concat([denoising_logits, content], dim=1)
            return content, enc_unact, [], [], None
        enc_logits = self.enc_score_head(output_memory).float()
        n_o2m = self.o2m_queries if self.training else 0
        _, topk_all = torch.topk(enc_logits.max(-1).values, max(self.num_queries, n_o2m), dim=-1)

        def select(ind):
            mem = output_memory.gather(1, ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            anc = anchors.expand(memory.shape[0], -1, -1).gather(1, ind.unsqueeze(-1).repeat(1, 1, 4))
            e = self.enc_bbox_head(mem).float()
            theta = 0.5 * torch.atan2(e[..., 5], e[..., 4])
            return mem, torch.cat([e[..., :4] + anc, theta.unsqueeze(-1)], -1)

        topk_ind = topk_all[:, :self.num_queries]
        topk_memory, enc_unact = select(topk_ind)
        o2m = None
        if n_o2m:                  # same ranked proposals, independent group (overlaps the main top-k)
            m2, u2 = select(topk_all[:, :n_o2m])
            o2m = (m2.detach(), u2.detach())
        enc_boxes_list, enc_logits_list = [], []
        if self.training:
            enc_boxes_list.append(unact_to_box(enc_unact))
            enc_logits_list.append(enc_logits.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, enc_logits.shape[-1])))
        content = topk_memory.detach()
        enc_unact = enc_unact.detach()
        if denoising_unact is not None:
            enc_unact = torch.concat([denoising_unact, enc_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return content, enc_unact, enc_boxes_list, enc_logits_list, o2m

    def forward(self, feats, targets=None, dense=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        if self.training and self.num_denoising > 0:
            dn_logits, dn_unact, attn_mask, dn_meta = obb_denoising_group(
                targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale)
        else:
            dn_logits = dn_unact = attn_mask = dn_meta = None
        content, ref_unact, enc_boxes_list, enc_logits_list, o2m = self._get_decoder_input(
            memory, spatial_shapes, dn_logits, dn_unact, dense=dense)
        n_o2m = 0
        if o2m is not None:        # H10: append the one-to-many group; it neither sees nor is seen by dn/main
            n_main = content.shape[1]
            n_o2m = o2m[0].shape[1]
            content = torch.concat([content, o2m[0]], dim=1)
            ref_unact = torch.concat([ref_unact, o2m[1]], dim=1)
            full = torch.zeros(n_main + n_o2m, n_main + n_o2m, dtype=torch.bool, device=content.device)
            if attn_mask is not None:
                full[:n_main, :n_main] = attn_mask
            full[:n_main, n_main:] = True
            full[n_main:, :n_main] = True
            attn_mask = full
        out_boxes, out_logits, out_corners, out_refs, pre_boxes, pre_logits = self.decoder(
            content, ref_unact, memory, spatial_shapes, self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, self.pre_bbox_head, self.up, self.reg_scale, attn_mask=attn_mask)
        if n_o2m:
            sp = [out_logits.shape[2] - n_o2m, n_o2m]
            pre_logits, o2m_pre_logits = torch.split(pre_logits, sp, dim=1)
            pre_boxes, o2m_pre_boxes = torch.split(pre_boxes, sp, dim=1)
            out_logits, o2m_logits = torch.split(out_logits, sp, dim=2)
            out_boxes, o2m_boxes = torch.split(out_boxes, sp, dim=2)
            out_corners, o2m_corners = torch.split(out_corners, sp, dim=2)
            out_refs, o2m_refs = torch.split(out_refs, sp, dim=2)
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_boxes, pre_boxes = torch.split(pre_boxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_boxes, out_boxes = torch.split(out_boxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
        if not self.training:
            return {'pred_logits': out_logits[-1], 'pred_boxes': out_boxes[-1]}
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_boxes[-1], 'pred_corners': out_corners[-1],
               'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        if self.aux_loss:
            out['aux_outputs'] = self._aux2(out_logits[:-1], out_boxes[:-1], out_corners[:-1], out_refs[:-1],
                                            out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_logits_list, enc_boxes_list)]
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_boxes}
            if dn_meta is not None:
                out['dn_outputs'] = self._aux2(dn_out_logits, dn_out_boxes, dn_out_corners, dn_out_refs,
                                               dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_boxes}
                out['dn_meta'] = dn_meta
            if n_o2m:
                out['o2m_outputs'] = self._aux2(o2m_logits, o2m_boxes, o2m_corners, o2m_refs)
                out['o2m_pre_outputs'] = {'pred_logits': o2m_pre_logits, 'pred_boxes': o2m_pre_boxes}
        return out

    @staticmethod
    def _aux2(outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                 'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
