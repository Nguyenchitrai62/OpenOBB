"""OpenOBB10: OpenOBB8 (pretrained OpenOBB7 core + oriented relational refiner) with a multi-slot dense head.

Why (docs/OPENOBB10.md; Wall_Color train labels at 1280 px, research/LEDGER.md F45):
  * Every anchor point of OpenOBB3-8 predicts ONE box, and the assigner gives each point to ONE object. Small objects
    that physically overlap a neighbour therefore compete for the same few points: wall_300 (9.9 x 6 px) sits 3.9 px
    (median) from a junction; at stride 8 it covers 1.5 candidate points, 75% of them shared, and 72% of wall_300
    have no point of their own (junction 60%, wall 15%). The measured error is exactly that: wall_300 -> junction
    28-40% of the time (F35, F43, F44). A finer grid does not solve it (stride 4: still 60% without a point of their
    own) because the boxes overlap. With S prediction slots per point (CrowdDet's multiple-instance prediction) an
    object needs a point shared by at most S objects: S = 3 leaves 1% of wall_300 / junction / wall without one.
  * Slot matching per point: the up-to-S objects selected for a point are matched to the S slots by the
    minimum-cost permutation (EMD, CrowdDet); cost = -(score^alpha * IoU^beta) of each slot's own prediction.
  * Identity at initialisation: slot 0 is OpenOBB8's head (same parameter names, an OpenOBB8 / OpenOBB3 checkpoint
    loads into it); extra slots start with zero class weights and a 1e-4 prior (below the default NMS conf 0.001),
    so OpenOBB10's output equals OpenOBB8's at step 0. Their box layers copy slot 0's (warm start).

Layout: the dense outputs are the S slots concatenated along the anchor axis (slot-major, each slot P3, P4, P5);
`anchors()` repeats the points S times, so decode / refiner / fast_nms work unchanged. Training outputs carry
"slots" for the loss (openobb10_loss.py). `off=` switches parts off ("slots"; "reach", "balance" act in the loss).
"""
import math

import torch
import torch.nn as nn

from .openobb3 import REG_CH, fp32_out
from .openobb8 import OpenOBB8

PARTS10 = ("slots", "reach", "balance")
SLOTS = 3
SLOT_PRIOR = 1e-4        # extra slots start below the default NMS confidence 0.001: invisible until trained


def parse_off10(off):
    """'reach,balance' / ['slots'] / None -> frozenset of switched-off OpenOBB10 parts (unknown names are an error)."""
    if not off:
        return frozenset()
    names = [s.strip() for s in (off.split(",") if isinstance(off, str) else off) if str(s).strip()]
    bad = [n for n in names if n not in PARTS10]
    if bad:
        raise ValueError(f"unknown OpenOBB10 part(s) {bad}; parts: {', '.join(PARTS10)}")
    return frozenset(names)


class SlotHead(nn.Module):
    """OpenOBB3 / OpenOBB7 dense oriented head (its modules, same parameter names) + S - 1 extra prediction slots per
    point: one more 1x1 class layer and one more 1x1 box layer per slot on the towers' last features."""

    def __init__(self, base, nc, slots=SLOTS, prior=SLOT_PRIOR):
        super().__init__()
        self.cls, self.reg = base.cls, base.reg
        self.ctx = getattr(base, "ctx", None)            # OpenOBB7's large-kernel unit (None when cls_ctx is off)
        self.slots = slots
        cc, cr = self.cls[0][-1].in_channels, self.reg[0][-1].in_channels
        self.cls_slot = nn.ModuleList(nn.ModuleList(nn.Conv2d(cc, nc, 1) for _ in range(slots - 1)) for _ in self.cls)
        self.reg_slot = nn.ModuleList(nn.ModuleList(nn.Conv2d(cr, REG_CH, 1) for _ in range(slots - 1))
                                      for _ in self.reg)
        for lvl in self.cls_slot:
            for m in lvl:
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, -math.log((1 - prior) / prior))
        self.copy_primary_box()

    @torch.no_grad()
    def copy_primary_box(self):
        """Extra slots' box layers := slot 0's (warm start after loading a single-slot checkpoint)."""
        for mr, lvl in zip(self.reg, self.reg_slot):
            for m in lvl:
                m.weight.copy_(mr[-1].weight)
                m.bias.copy_(mr[-1].bias)

    def forward(self, feats):
        cls = [[] for _ in range(self.slots)]
        reg = [[] for _ in range(self.slots)]
        for i, f in enumerate(feats):
            mc, mr = self.cls[i], self.reg[i]
            h = mc[1](mc[0](f))
            if self.ctx is not None:
                h = self.ctx[i](h)
            h = mc[3](mc[2](h))
            r = mr[1](mr[0](f))
            cls[0].append(mc[4](h).flatten(2))
            reg[0].append(fp32_out(mr[2], r).flatten(2))
            for j in range(self.slots - 1):
                cls[j + 1].append(self.cls_slot[i][j](h).flatten(2))
                reg[j + 1].append(fp32_out(self.reg_slot[i][j], r).flatten(2))
        cat = lambda lst: torch.cat([torch.cat(s, 2) for s in lst], 2).transpose(1, 2).float()  # noqa: E731
        return cat(cls), cat(reg)


class OpenOBB10(OpenOBB8):
    arch = "v10"

    def __init__(self, size="x", num_classes=15, img_size=1024, lsk=True, max_det=1000, nms_iou=0.7, off=None,
                 off8=None, off7=None, rel_k=600, slots=SLOTS, **_):
        super().__init__(size, num_classes, img_size, lsk=lsk, max_det=max_det, nms_iou=nms_iou, off=off8, off7=off7,
                         rel_k=rel_k)
        self.off10 = parse_off10(off)
        self.slots = 1 if "slots" in self.off10 else int(slots)
        if self.slots > 1:
            self.head = SlotHead(self.head, num_classes, self.slots)

    def anchors(self, feats):
        pts, st = super().anchors(feats)
        if self.slots == 1:
            return pts, st
        return pts.repeat(self.slots, 1), st.repeat(self.slots)

    def forward(self, x, targets=None, ctx=None):
        out = super().forward(x, targets, ctx)
        if self.training:
            out["slots"] = self.slots
        return out
