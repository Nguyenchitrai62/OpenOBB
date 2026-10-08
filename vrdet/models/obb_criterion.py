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
    def __init__(self, cost_class=2.0, cost_chamfer=5.0, cost_kld=2.0, alpha=0.25, gamma=2.0, gauss="kld",
                 cost_iou=0.0):
        super().__init__()
        self.cost_class, self.cost_chamfer, self.cost_kld = cost_class, cost_chamfer, cost_kld
        # H15 (Stable-DINO / Rank-DETR idea): class cost on p^(1-g) * IoU^g so the one-to-one positive is the
        # best-localised query and scores learn to rank by localisation quality (ProbIoU as the IoU proxy)
        self.cost_iou = cost_iou
        self.alpha, self.gamma, self.gauss = alpha, gamma, gauss

    def pair_cost(self, logits, boxes, tgt_ids, tgt_box):
        """(N, C) logits, (N, 5) boxes vs (M,) labels, (M, 5) boxes -> (N, M) matching cost."""
        p = logits.float().sigmoid()[:, tgt_ids]
        boxes = boxes.float()
        if self.cost_iou > 0:
            p = p.pow(1 - self.cost_iou) * probiou(boxes[:, None, :], tgt_box[None, :, :].float()).clamp(0, 1).pow(self.cost_iou)
        neg = (1 - self.alpha) * (p ** self.gamma) * (-(1 - p + 1e-8).log())
        pos = self.alpha * ((1 - p) ** self.gamma) * (-(p + 1e-8).log())
        C = self.cost_class * (pos - neg) + self.cost_chamfer * chamfer_matrix(boxes, tgt_box.float())
        g = kld(boxes[:, None, :], tgt_box[None, :, :].float()) if self.gauss == "kld" else \
            1 - probiou(boxes[:, None, :], tgt_box[None, :, :].float())
        return torch.nan_to_num(C + self.cost_kld * g, nan=1e4, posinf=1e4)

    @torch.no_grad()
    def one_to_many(self, outputs, targets, k=6):
        """H10 assignment on the GPU: every target takes its k lowest-cost queries; a query claimed by several
        targets keeps the cheapest one. Per image, so the cost matrix stays (queries x targets of that image)."""
        out = []
        for i, t in enumerate(targets):
            n = len(t["labels"])
            if n == 0:
                e = torch.zeros(0, dtype=torch.int64)
                out.append((e, e))
                continue
            C = self.pair_cost(outputs["pred_logits"][i], outputs["pred_boxes"][i], t["labels"], t["boxes"])
            kk = min(k, C.shape[0])
            cand = C.topk(kk, dim=0, largest=False).indices                # (kk, n) query ids per target
            q = cand.reshape(-1)
            g = torch.arange(n, device=C.device).repeat(kk)
            c = C[q, g]
            order = torch.argsort(c)                                        # cheapest claim wins
            q, g = q[order], g[order]
            uq, inv = torch.unique(q, return_inverse=True)
            seen = torch.full((len(uq),), len(q), device=q.device, dtype=torch.int64)
            seen.scatter_reduce_(0, inv, torch.arange(len(q), device=q.device), reduce="amin")
            first = seen[inv] == torch.arange(len(q), device=q.device)
            out.append((q[first].cpu(), g[first].cpu()))
        return out

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
        if self.cost_iou > 0:
            p = p.pow(1 - self.cost_iou) * probiou(boxes[:, None, :], tgt_box[None, :, :]).clamp(0, 1).pow(self.cost_iou)
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
                 gamma=1.5, mal_alpha=None, use_uni_set=True, gauss="kld", o2m_k=6, o2m_weight=1.0, aqd=False,
                 angle_lam=3.0):
        super().__init__()
        self.o2m_k, self.o2m_weight = o2m_k, o2m_weight
        self.aqd = aqd                  # H12: denoising queries that lose their source GT get a background class
        self.angle_lam = angle_lam      # H13: square-aware angle loss (weight 'loss_angle' in weight_dict)
        self.gauss = gauss
        self.matcher, self.weight_dict, self.losses = matcher, weight_dict, list(losses)
        self.num_classes, self.reg_max, self.gamma, self.mal_alpha = num_classes, reg_max, gamma, mal_alpha
        self.use_uni_set = use_uni_set
        self._clear_cache()

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn, self.fgl_targets_o2m = None, None, None
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
        out = {'loss_bbox': l1.sum() / num_boxes, 'loss_kld': g.sum() / num_boxes}
        if 'loss_angle' in self.weight_dict:
            # YOLO26-OBB: sin^2(2 dtheta), down-weighted for near-square targets whose angle is ill-defined
            ratio = torch.log(tgt[:, 2].clamp(min=1e-6) / tgt[:, 3].clamp(min=1e-6))
            omega = torch.exp(-ratio.pow(2) / self.angle_lam ** 2)
            out['loss_angle'] = (torch.sin(2 * (src[:, 4] - tgt[:, 4])).pow(2) * omega).sum() / num_boxes
        return out

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
            key = 'fgl_targets_dn' if 'is_dn' in outputs else 'fgl_targets_o2m' if 'is_o2m' in outputs \
                else 'fgl_targets'
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

    # ---------------------------------------------------------------- H17 primitive membership
    def loss_members(self, outputs, targets, indices):
        from .vector import members_from_boxes
        logits, pts, vmask = outputs['pred_members'], outputs['vec_pts'], outputs['vec_mask']
        bce, dice, n = logits.sum() * 0, logits.sum() * 0, 0
        for b, (qi, gi) in enumerate(indices):
            valid = vmask[b]
            if len(qi) == 0 or not bool(valid.any()):
                continue
            p = pts[b][valid].view(-1, 8, 2)
            tgt = members_from_boxes(p, targets[b]['boxes'][gi].float()).float()       # (k, M_valid)
            lg = logits[b][qi][:, valid]
            bce = bce + F.binary_cross_entropy_with_logits(lg, tgt, reduction='none').mean(1).sum()
            pr = lg.sigmoid()
            dice = dice + (1 - (2 * (pr * tgt).sum(1) + 1) / (pr.sum(1) + tgt.sum(1) + 1)).sum()
            n += len(qi)
        n = max(n, 1)
        return {'loss_member_bce': bce / n, 'loss_member_dice': dice / n}

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

    @torch.no_grad()
    def aqd_indices(self, dn_out, dn_meta, targets):
        """H12 (RHINO's adaptive query denoising): inside every denoising group, match the positive noised queries
        to the targets with the Hungarian cost on their current predictions; a query whose match is not its own
        source target keeps the box loss but gets a background class target (it duplicates a better query)."""
        out = []
        G = dn_meta["dn_num_group"]
        for i, t in enumerate(targets):
            n = len(t['labels'])
            if n == 0:
                e = torch.zeros(0, dtype=torch.int64)
                out.append((e, e))
                continue
            q = dn_meta["dn_positive_idx"][i].reshape(G, n)
            keep_q, keep_g = [], []
            for g in range(G):
                C = self.matcher.pair_cost(dn_out['pred_logits'][i, q[g]], dn_out['pred_boxes'][i, q[g]],
                                           t['labels'], t['boxes'])
                r, c = linear_sum_assignment(C.cpu().numpy())
                own = torch.as_tensor(r[r == c], dtype=torch.int64)
                keep_q.append(q[g].cpu()[own])
                keep_g.append(own)
            out.append((torch.cat(keep_q), torch.cat(keep_g)))
        return out

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
            indices_dn_cls = indices_dn
            if self.aqd:                        # one group-wise matching on the last dn layer, used for all layers
                indices_dn_cls = dev(self.aqd_indices(outputs['dn_outputs'][-1], outputs['dn_meta'], targets))
            for i, aux in enumerate(outputs['dn_outputs']):
                aux['is_dn'] = True
                aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    ind = indices_dn_cls if loss == 'mal' else indices_dn
                    losses.update(self._weighted(self.get_loss(loss, aux, targets, ind, dn_num_boxes), f'_dn_{i}'))
            if 'dn_pre_outputs' in outputs:
                for loss in self.losses:
                    ind = indices_dn_cls if loss == 'mal' else indices_dn
                    losses.update(self._weighted(self.get_loss(loss, outputs['dn_pre_outputs'], targets, ind,
                                                               dn_num_boxes), '_dn_pre'))
        if 'pred_members' in outputs and 'loss_member_bce' in self.weight_dict:
            losses.update(self._weighted(self.loss_members(outputs, targets, indices)))
        if 'o2m_outputs' in outputs:           # H10: one-to-many group, one assignment (last layer) for all layers
            ind_o2m = dev(self.matcher.one_to_many(outputs['o2m_outputs'][-1], targets, self.o2m_k))
            n_o2m = max(float(sum(len(x[0]) for x in ind_o2m)), 1.0)
            heads = list(enumerate(outputs['o2m_outputs'])) + [('pre', outputs['o2m_pre_outputs'])]
            for i, aux in heads:
                aux['is_o2m'] = True
                aux['up'], aux['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    l_dict = self._weighted(self.get_loss(loss, aux, targets, ind_o2m, n_o2m), f'_o2m_{i}')
                    losses.update({k: v * self.o2m_weight for k, v in l_dict.items()})
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
