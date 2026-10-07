"""Matcher and losses for the oriented D-FINE/DEIM head.

Structure follows DEIMCriterion / HungarianMatcher (Apache-2.0; Copyright (c) 2024 The DEIM Authors,
D-FINE Authors, Facebook DETR). Oriented-box parts are VRDet's own:
  * matching cost  = focal class cost + Chamfer corner distance + KLD (weights from O2-DETR's ablation);
  * MAL quality target = exact rotated IoU (no gradient);
  * box loss       = L1 on (cx, cy, w, h, theta/pi) with the target aligned to the predicted angle
                     (no angle-boundary discontinuity) + KLD;
  * FGL / DDF      = D-FINE's distribution losses over 4 rotated edges + 1 angle-residual distribution.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from vrdet.ops.obb_torch import probiou, rotated_iou
from .obb_utils import align_to, chamfer_matrix, kld, obb2distance

N_DIST = 5


class OBBHungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0, alpha=0.25, gamma=2.0, gauss="kld"):
        super().__init__()
        self.cost_class, self.cost_chamfer, self.cost_kld = cost_class, cost_chamfer, cost_kld
        self.alpha, self.gamma, self.gauss = alpha, gamma, gauss

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        if sum(sizes) == 0:
            e = torch.zeros(0, dtype=torch.int64)
            return {'indices': [(e, e) for _ in range(bs)]}
        prob = outputs["pred_logits"].flatten(0, 1).float().sigmoid()
        boxes = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_box = torch.cat([t["boxes"] for t in targets]).float()
        p = prob[:, tgt_ids]
        neg = (1 - self.alpha) * (p ** self.gamma) * (-(1 - p + 1e-8).log())
        pos = self.alpha * ((1 - p) ** self.gamma) * (-(p + 1e-8).log())
        C = self.cost_class * (pos - neg)
        C = C + self.cost_chamfer * chamfer_matrix(boxes, tgt_box)
        g = kld(boxes[:, None, :], tgt_box[None, :, :]) if self.gauss == "kld" else \
            1 - probiou(boxes[:, None, :], tgt_box[None, :, :])
        C = C + self.cost_kld * g
        C = torch.nan_to_num(C.view(bs, nq, -1), nan=1.0).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return {'indices': [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                            for i, j in indices]}


class OBBCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses=('mal', 'boxes', 'local'), num_classes=15, reg_max=32,
                 gamma=1.5, mal_alpha=None, use_uni_set=True, gauss="kld"):
        super().__init__()
        self.gauss = gauss
        self.matcher, self.weight_dict, self.losses = matcher, weight_dict, list(losses)
        self.num_classes, self.reg_max, self.gamma, self.mal_alpha = num_classes, reg_max, gamma, mal_alpha
        self.use_uni_set = use_uni_set
        self._clear_cache()

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _matched(self, outputs, targets, indices):
        idx = self._src_idx(indices)
        src = outputs['pred_boxes'][idx].float()
        tgt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0).float()
        return idx, src, tgt

    # ---------------------------------------------------------------- classification (DEIM MAL)
    def loss_mal(self, outputs, targets, indices, num_boxes):
        idx, src, tgt = self._matched(outputs, targets, indices)
        ious = rotated_iou(src.detach(), tgt).to(src.dtype)
        logits = outputs['pred_logits'].float()
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        score_o = torch.zeros_like(target_classes, dtype=logits.dtype)
        score_o[idx] = ious
        target_score = (score_o.unsqueeze(-1) * target).pow(self.gamma)
        pred_score = logits.sigmoid().detach()
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target
        loss = F.binary_cross_entropy_with_logits(logits, target_score, weight=weight, reduction='none')
        return {'loss_mal': loss.mean(1).sum() * logits.shape[1] / num_boxes}

    # ---------------------------------------------------------------- boxes
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx, src, tgt = self._matched(outputs, targets, indices)
        if len(src) == 0:
            z = outputs['pred_boxes'].sum() * 0
            return {'loss_bbox': z, 'loss_kld': z}
        tgt_al = align_to(tgt, src[:, 4].detach())
        l1 = (src[:, :4] - tgt_al[:, :4]).abs().sum(-1) + (src[:, 4] - tgt_al[:, 4]).abs() / math.pi
        g = kld(src, tgt) if self.gauss == "kld" else 1 - probiou(src, tgt)
        return {'loss_bbox': l1.sum() / num_boxes, 'loss_kld': g.sum() / num_boxes}

    # ---------------------------------------------------------------- FGL + DDF (D-FINE)
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        losses = {}
        if 'pred_corners' not in outputs:
            return losses
        idx, src, tgt = self._matched(outputs, targets, indices)
        R = self.reg_max + 1
        pred_corners = outputs['pred_corners'][idx].reshape(-1, R)
        ref = outputs['ref_points'][idx].detach().float()
        with torch.no_grad():
            key = 'fgl_targets_dn' if 'is_dn' in outputs else 'fgl_targets'
            if getattr(self, key) is None:
                setattr(self, key, obb2distance(ref, align_to(tgt, ref[:, 4]), self.reg_max,
                                                outputs['reg_scale'], outputs['up']))
        target_corners, weight_right, weight_left = getattr(self, key)
        ious = rotated_iou(src.detach(), tgt)
        weight_targets = ious.unsqueeze(-1).repeat(1, N_DIST).reshape(-1).detach()
        losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)
        if 'teacher_corners' in outputs and outputs['teacher_corners'] is not None:
            pc = outputs['pred_corners'].reshape(-1, R)
            tc = outputs['teacher_corners'].reshape(-1, R)
            if not torch.equal(pc, tc):
                wtl = outputs['teacher_logits'].float().sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(wtl, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, N_DIST).reshape(-1)
                wtl[idx] = ious.reshape_as(wtl[idx]).to(wtl.dtype)
                wtl = wtl.unsqueeze(-1).repeat(1, 1, N_DIST).reshape(-1).detach()
                kl = (T ** 2) * nn.KLDivLoss(reduction='none')(F.log_softmax(pc / T, dim=1),
                                                                F.softmax(tc.detach() / T, dim=1)).sum(-1)
                lm = wtl * kl
                if 'is_dn' not in outputs:
                    batch_scale = 8 / outputs['pred_boxes'].shape[0]
                    self.num_pos = (mask.sum() * batch_scale) ** 0.5
                    self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
                l1 = lm[mask].mean() if mask.any() else lm.sum() * 0
                l2 = lm[~mask].mean() if (~mask).any() else lm.sum() * 0
                losses['loss_ddf'] = (l1 * self.num_pos + l2 * self.num_neg) / (self.num_pos + self.num_neg)
        return losses

    @staticmethod
    def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left, weight=None, avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
            + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss.sum() / avg_factor if avg_factor is not None else loss.sum()

    # ---------------------------------------------------------------- driver
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        return {'boxes': self.loss_boxes, 'mal': self.loss_mal, 'local': self.loss_local}[loss](
            outputs, targets, indices, num_boxes)

    @staticmethod
    def _go_indices(indices, indices_aux_list):
        """Union of the matchings over all layers (DEIM): most frequent target per query."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]])) for a, b in zip(indices, indices_aux)]
        for ind in [torch.cat([i[0][:, None], i[1][:, None]], 1) for i in indices]:
            if len(ind) == 0:
                e = torch.zeros(0, dtype=torch.int64)
                results.append((e, e))
                continue
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            unique_sorted = unique[torch.argsort(counts, descending=True)]
            column_to_row = {}
            for row, col in unique_sorted.tolist():
                if row not in column_to_row:
                    column_to_row[row] = col
            results.append((torch.tensor(list(column_to_row.keys()), dtype=torch.int64),
                            torch.tensor(list(column_to_row.values()), dtype=torch.int64)))
        return results

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        out = []
        for i, t in enumerate(targets):
            n = len(t['labels'])
            if n > 0:
                gt_idx = torch.arange(n, dtype=torch.int64).tile(dn_num_group)
                out.append((dn_positive_idx[i].cpu(), gt_idx))
            else:
                e = torch.zeros(0, dtype=torch.int64)
                out.append((e, e))
        return out

    def _weighted(self, l_dict, suffix=''):
        return {k + suffix: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict}

    def forward(self, outputs, targets):
        device = outputs['pred_logits'].device
        outputs_wo_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_wo_aux, targets)['indices']
        self._clear_cache()
        aux_list = outputs.get('aux_outputs', []) + ([outputs['pre_outputs']] if 'pre_outputs' in outputs else [])
        cached = [self.matcher(a, targets)['indices'] for a in aux_list]
        cached_enc = [self.matcher(a, targets)['indices'] for a in outputs.get('enc_aux_outputs', [])]
        indices_go = self._go_indices(indices, cached + cached_enc)
        num_boxes_go = max(float(sum(len(x[0]) for x in indices_go)), 1.0)
        num_boxes = max(float(sum(len(t["labels"]) for t in targets)), 1.0)

        def dev(ind):
            return [(i.to(device), j.to(device)) for i, j in ind]

        indices, indices_go = dev(indices), dev(indices_go)
        cached, cached_enc = [dev(c) for c in cached], [dev(c) for c in cached_enc]
        losses = {}
        for loss in self.losses:
            uni = self.use_uni_set and loss in ('boxes', 'local')
            losses.update(self._weighted(self.get_loss(loss, outputs, targets, indices_go if uni else indices,
                                                       num_boxes_go if uni else num_boxes)))
        for i, aux in enumerate(outputs.get('aux_outputs', [])):
            aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
            for loss in self.losses:
                uni = self.use_uni_set and loss in ('boxes', 'local')
                losses.update(self._weighted(self.get_loss(loss, aux, targets, indices_go if uni else cached[i],
                                                           num_boxes_go if uni else num_boxes), f'_aux_{i}'))
        if 'pre_outputs' in outputs:
            for loss in self.losses:
                uni = self.use_uni_set and loss in ('boxes', 'local')
                losses.update(self._weighted(self.get_loss(loss, outputs['pre_outputs'], targets,
                                                           indices_go if uni else cached[-1],
                                                           num_boxes_go if uni else num_boxes), '_pre'))
        for i, aux in enumerate(outputs.get('enc_aux_outputs', [])):
            for loss in self.losses:
                uni = self.use_uni_set and loss == 'boxes'
                losses.update(self._weighted(self.get_loss(loss, aux, targets, indices_go if uni else cached_enc[i],
                                                           num_boxes_go if uni else num_boxes), f'_enc_{i}'))
        if 'dn_outputs' in outputs:
            indices_dn = dev(self.get_cdn_matched_indices(outputs['dn_meta'], targets))
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux in enumerate(outputs['dn_outputs']):
                aux['is_dn'] = True
                aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    losses.update(self._weighted(self.get_loss(loss, aux, targets, indices_dn, dn_num_boxes), f'_dn_{i}'))
            if 'dn_pre_outputs' in outputs:
                for loss in self.losses:
                    losses.update(self._weighted(self.get_loss(loss, outputs['dn_pre_outputs'], targets, indices_dn,
                                                               dn_num_boxes), '_dn_pre'))
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
