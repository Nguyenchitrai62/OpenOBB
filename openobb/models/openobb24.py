"""OpenOBB24: OpenOBB23 + a learnt LOCALISATION QUALITY of every refined box, used where the box is chosen.

Evidence (docs/OPENOBB24.md, research/LEDGER.md F109, checkpoint of F107, Wall_Color val):
  * recall at ProbIoU 0.5 is level with YOLO26x at every length; the gap is box tightness on wall / junction / wall_300
    (recall at 0.75: 0.50 / 0.38 / 0.17 vs YOLO 0.56 / 0.47 / 0.24);
  * the one-to-many head leaves several candidates on each object and NMS keeps the one with the best CLASS score; in
    each NMS cluster a tighter candidate exists (+0.052 mAP50-95 if it were kept, rank_vote_oracle.py), and a perfect
    localisation factor on the score adds +0.044 for this model and +0.048 for YOLO26x (score_iou_corr.py): the class
    score barely knows how tight its box is (Spearman 0.19-0.22 on walls / junctions);
  * the score is learnt on the head's box, but the box that is kept is the one after the edge refiner and the boundary
    snapping (+0.011 together, the snapping alone -0.015 on junctions, F109);
  * without any training, letting the boundary field's agreement with the final box's four sides choose the kept box
    of each cluster already gives +0.015 mAP50-95 (junction +0.019, wall +0.012), mAP50 unchanged (quality_probe.py).

Mechanism: a small MLP (LocQuality) predicts q = ProbIoU(final box, its object) for every refined candidate from what
the model already computes for that box, all DETACHED (the new loss cannot disturb the parts of v23): the edge
refiner's last hidden layer (stride-4 trunk + P3 samples on the box's edges), the boundary field sampled on three lines
across each of the four FINAL sides (stride 2, learnt from the labelled sides), the sharpness / entropy of the head's
three distance distributions (GFLv2's cue), the moves of the refiner and of the snapping, and the box geometry. It is
learnt with the existing fit loss (BCE to ProbIoU >= 0.5, else 0) on the dense candidates only. Inference:
  * the NMS visits candidates by score x q^q_beta: in each cluster the box kept is the tight one, its score stays the
    class score unless q_alpha > 0 (score^(1 - q_alpha) x q^q_alpha);
  * q_beta, q_alpha and the class groups whose snapping is skipped (snap_skip) are chosen on val at the end of
    training, like the NMS (0, 0, none = OpenOBB23 exactly).
The last layer starts at zero (q constant), so OpenOBB24 starts as OpenOBB23 whatever q_beta / q_alpha. Cost: one MLP on
the refined candidates (top 1500), ~0.1 ms.

Switch v24_off=quality: the model is OpenOBB23 (+ the snap_skip option).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb17 import FIELD_STRIDE, _sample, peak_mask
from .openobb18 import OpenOBB23
from .openobb3 import ALONG_BINS, REG_CH

Q_PRIOR = 0.6                    # q at the start (constant): sigmoid of the last layer's bias
N_Q = 5                          # points along each side where the field is read
Q_LINES = (-1.0, 0.0, 1.0)       # px across each side (inside, on, outside)
N_DFL = 6                        # distribution cues: top-2 mass and normalised entropy of left / right / thickness
N_GEO = 4                        # log length, log thickness, sin 2 theta, cos 2 theta
N_MOVE = 5                       # move of a stage, in the input box's frame
OFF24 = ("quality",)


def parse_off24(off):
    names = {s.strip() for s in (off.split(",") if isinstance(off, str) else (off or [])) if str(s).strip()}
    bad = names - set(OFF24)
    if bad:
        raise ValueError(f"v24_off: unknown {sorted(bad)}; known {list(OFF24)}")
    return names


def dfl_cues(rows):
    """Head rows (B, K, R) -> (B, K, N_DFL): for the left end / right end / thickness distributions, the mass of the top
    two bins and the normalised entropy; zeros for a head without distributions (box=direct)."""
    if rows.shape[-1] != REG_CH:
        return rows.new_zeros((*rows.shape[:-1], N_DFL))
    out = []
    for lo, hi in ((2, 2 + ALONG_BINS), (2 + ALONG_BINS, 2 + 2 * ALONG_BINS), (2 + 2 * ALONG_BINS, REG_CH)):
        p = rows[..., lo:hi].float().softmax(-1)
        out += [p.topk(2, -1).values.sum(-1), -(p * p.clamp(min=1e-9).log()).sum(-1) / math.log(hi - lo)]
    return torch.stack(out, -1)


def box_move(a, b):
    """Move from box a to box b (B, K, 5) px, in a's frame: along / across shift in units of a's thickness, log length
    and log thickness ratios, angle change x 5 (wrapped to +-pi/2)."""
    th = a[..., 4]
    d = b[..., :2] - a[..., :2]
    T = a[..., 3].abs().clamp(min=1.0)
    du = (d[..., 0] * th.cos() + d[..., 1] * th.sin()) / T
    dv = (-d[..., 0] * th.sin() + d[..., 1] * th.cos()) / T
    dl = torch.log(b[..., 2].abs().clamp(min=0.5) / a[..., 2].abs().clamp(min=0.5))
    dt = torch.log(b[..., 3].abs().clamp(min=0.5) / a[..., 3].abs().clamp(min=0.5))
    da = torch.remainder(b[..., 4] - th + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack([du.clamp(-8, 8), dv.clamp(-8, 8), dl, dt, 5 * da], -1)


def side_profiles(field, b):
    """Boundary field logits (B, 2, H, W) at stride FIELD_STRIDE on the four sides of boxes b (B, K, 5) px: for each side,
    len(Q_LINES) lines parallel to it (px across it), N_Q points along it, both channels -> (B, K, 4 x lines x N_Q x 2),
    scaled by 1/4."""
    B, K = b.shape[:2]
    c, L, T, th = b[..., :2], b[..., 2].abs(), b[..., 3].abs(), b[..., 4]
    ax = torch.stack([th.cos(), th.sin()], -1)
    nx = torch.stack([-th.sin(), th.cos()], -1)
    q = torch.linspace(-0.4, 0.4, N_Q, device=b.device)
    lines = torch.tensor(Q_LINES, device=b.device)
    pos = []
    # (distance of the side from the centre, outward normal, length of the side, direction along the side)
    for off, normal, run_len, run in ((L / 2, ax, T, nx), (L / 2, -ax, T, nx), (T / 2, nx, L, ax), (T / 2, -nx, L, ax)):
        base = c + off[..., None] * normal                                # (B, K, 2) centre of the side
        p = (base[..., None, None, :] + lines[None, None, :, None, None] * normal[..., None, None, :]
             + (q[None, None, None, :] * run_len[..., None, None])[..., None] * run[..., None, None, :])
        pos.append(p)                                                     # (B, K, lines, N_Q, 2)
    pos = torch.stack(pos, 2)                                             # (B, K, 4, lines, N_Q, 2)
    s = _sample(field, pos.reshape(B, -1, 2), FIELD_STRIDE)               # (B, K * 4 * lines * N_Q, 2)
    return (s.reshape(B, K, -1) / 4).clamp(-5, 5)


class LocQuality(nn.Module):
    """q logit = predicted ProbIoU(final box, its object) of refined candidates (all inputs detached by the caller).
    Starts constant (last layer zero, bias logit(Q_PRIOR))."""

    def __init__(self, hidden_ref, ch=128, field=True):
        super().__init__()
        self.field = field
        n_field = 4 * len(Q_LINES) * N_Q * 2 if field else 0
        n_in = hidden_ref + n_field + N_DFL + N_GEO + 2 * N_MOVE
        self.norm = nn.LayerNorm(n_in)
        self.mlp = nn.Sequential(nn.Linear(n_in, ch), nn.SiLU(inplace=True), nn.Linear(ch, ch), nn.SiLU(inplace=True),
                                 nn.Linear(ch, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, math.log(Q_PRIOR / (1 - Q_PRIOR)))

    def forward(self, h, field, cand, refined, final, dfl):
        geo = torch.stack([torch.log1p(final[..., 2].abs()) / 7, torch.log1p(final[..., 3].abs()) / 4,
                           torch.sin(2 * final[..., 4]), torch.cos(2 * final[..., 4])], -1)
        parts = [h.float(), dfl.float(), geo, box_move(cand, refined), box_move(refined, final)]
        if self.field:
            parts.insert(1, side_profiles(field.float(), final))
        x = self.norm(torch.cat(parts, -1).to(self.mlp[0].weight.dtype))
        return self.mlp(x).float()[..., 0]


class OpenOBB24(OpenOBB23):
    """OpenOBB23 + LocQuality on the refined candidates: quality-guided NMS (+ optional quality-aware score) and the
    per-class-group snapping skip, both calibrated on val. docs/OPENOBB24.md."""
    arch = "v24"

    def __init__(self, *args, off24=None, **kw):
        super().__init__(*args, **kw)
        self.off24 = parse_off24(off24)
        has_edge = getattr(self, "edge", None) is not None
        self.quality = (LocQuality(self.edge.mlp[-1].in_features, field=self.field is not None)
                        if has_edge and "quality" not in self.off24 else None)
        # inference settings, calibrated on val at the end of training (0, 0, () = OpenOBB23)
        self.q_beta, self.q_alpha, self.snap_skip = 0.0, 0.0, ()
        # q of the candidates outside the refined top ref_k (not refined, not snapped): the lowest q of the image's
        # refined candidates ("min"). The constant Q_PRIOR ("prior", 0.25.x) let unrefined boxes win NMS clusters over
        # refined ones whose learnt q was below it (F111); "drop": candidates outside the refined top ref_k are removed
        self.q_outside = "min"

    # ---- refinement with the hidden layer, optional snapping skip per class group
    def edge_snap(self, f4, p3, field, cand, grp=None):
        """-> ([(input, output) per stage], refiner hidden (B, K, hidden)). grp (B, K): class group of each candidate,
        the groups in self.snap_skip keep the refiner's box (no snapping)."""
        r, _, h = self.edge.refine_with_fit(f4, p3, cand, hidden=True)
        if field is None or self.snap is None:
            return [(cand, r)], h
        s = self.snap(field, r)
        if self.snap_skip and grp is not None:
            skip = torch.zeros(self.n_groups, dtype=torch.bool, device=grp.device)
            skip[[int(g) for g in self.snap_skip if int(g) < self.n_groups]] = True
            s = torch.where(skip[grp][..., None], r.to(s.dtype), s)
        return [(cand, r), (r, s)], h

    def refine_stages(self, f4, p3, field, cand):
        return self.edge_snap(f4, p3, field, cand)[0]

    def head_rows(self, reg, idx):
        """reg (B, A, G R) head outputs, idx (B, K) candidate indices (g A + a with class groups) -> (B, K, R)."""
        B, A, GR = reg.shape
        G = self.n_groups
        R = GR // G
        g = idx // A
        flat = (idx - g * A) * G + g             # (idx % A) G + idx // A; no % by a symbolic size (ONNX export)
        return reg.reshape(B, A * G, R).gather(1, flat[..., None].expand(-1, -1, R))

    def quality_logits(self, h, field, stages, rows):
        cand, refined, final = stages[0][0], stages[0][1], stages[-1][1]
        f = None if field is None else field.detach()
        return self.quality(h.detach(), f, cand.detach().float(), refined.detach().float(), final.detach().float(),
                            dfl_cues(rows.detach()))

    # ---- forward
    def forward(self, x, targets=None, ctx=None):
        if self.training:
            return self._forward_train(x, targets)
        return self.select(*self.scored_candidates(x))

    def scored_candidates(self, x):
        """Inference up to the NMS (see OpenOBB17.scored_candidates): the ranking logits carry q^q_beta, the output
        logits score^(1 - q_alpha) x q^q_alpha."""
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        field = None if self.field is None else self.field(f4, feats[0])
        o2o, nms = self.nms_setting()
        cls, reg = self.run_head(feats, o2o)
        cls, boxes = self.candidates(cls, reg, pts, st)
        q, idx = None, None
        if self.ref_k > 0:
            k = min(int(self.ref_k), cls.shape[1])
            idx = cls.detach().sigmoid().amax(-1).topk(k, dim=1).indices
            cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
            grp = idx // reg.shape[1] if self.n_groups > 1 else torch.zeros_like(idx)
            stages, h = self.edge_snap(f4, feats[0], field, cand, grp)
            boxes = boxes.scatter(1, idx[..., None].expand(-1, -1, 5), stages[-1][1].to(boxes.dtype))
            if self.quality is not None and (self.q_beta or self.q_alpha):
                q = self.quality_logits(h, field, stages, self.head_rows(reg, idx)).sigmoid()
        if idx is not None and self.q_outside == "drop":       # only the refined candidates can become detections
            inside = torch.zeros(cls.shape[:2], dtype=torch.bool, device=cls.device).scatter(1, idx, True)
            cls = cls.masked_fill(~inside[..., None], -30.0)
        rank, out_l = cls, None
        if q is not None:
            q = q.float()
            fill = q.amin(1, keepdim=True) if self.q_outside == "min" else torch.full_like(q[:, :1], Q_PRIOR)
            qa = fill.expand(-1, cls.shape[1]).clone().scatter(1, idx, q).clamp(1e-4, 1.0).log()[..., None]
            ls = F.logsigmoid(cls.float())
            rank = torch.logit((ls + self.q_beta * qa).exp().clamp(1e-6, 1 - 1e-6))
            out_l = torch.logit(((1 - self.q_alpha) * ls + self.q_alpha * qa).exp().clamp(1e-6, 1 - 1e-6))
        if isinstance(nms, str) and nms.startswith("peak"):
            keep = peak_mask(cls, [f.shape[-2:] for f in feats], self.n_groups)
            rank = rank.masked_fill(~keep[..., None], -30.0)
            nms = float(nms.split("+")[1]) if "+" in nms else None
        return rank, out_l, boxes, nms

    def _forward_train(self, x, targets):
        f4, feats = self.encode(x)
        pts, st = self.anchors(feats)
        field = None if self.field is None else self.field(f4, feats[0])
        cls, reg = self.run_head(feats, False)
        c1, r1 = self.run_head(feats, True)
        out = {"cls": cls, "reg": reg, "points": pts, "strides": st, "o2o_cls": c1, "o2o_reg": r1,
               "feats": feats, "input": x}
        if self.n_groups > 1:
            out["class_group"] = self.class_group
        if field is not None:
            out["field"] = field
        cc, boxes = self.candidates(cls.detach(), reg.detach(), pts, st)
        k = min(self.ref_k_train, boxes.shape[1])
        idx = cc.sigmoid().amax(-1).topk(k, dim=1).indices
        cand = boxes.gather(1, idx[..., None].expand(-1, -1, 5))
        cand, valid = self._with_jitter(cand, targets)
        stages, h = self.edge_snap(f4, feats[0], field, cand)
        out.update(ref_boxes=stages[0][1], ref_cand=stages[0][0], ref_valid=valid)
        if len(stages) > 1:
            out["ref_stages"] = [(c, r, 0.5) for c, r in stages[1:]]
        if self.quality is not None:
            rows = self.head_rows(reg.detach(), idx)
            pad = cand.shape[1] - rows.shape[1]                           # jittered labelled boxes: no head rows
            rows = torch.cat([rows, rows.new_zeros((rows.shape[0], pad, rows.shape[2]))], 1)
            head = torch.zeros_like(valid)
            head[:, :idx.shape[1]] = True
            out.update(ref_fit=self.quality_logits(h, field, stages, rows), ref_fit_boxes=stages[-1][1],
                       ref_fit_valid=head & valid)
        return out
