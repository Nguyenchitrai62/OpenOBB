"""OpenOBB15: OpenOBB14 + a full-resolution detail stream, a P2 grid and an edge refiner that reads the full-resolution
map. The pretrained trunk (+ neck) runs on the input halved, as OpenOBB14 at imgsz; the new stream runs on the full
input (2 x imgsz). docs/OPENOBB15.md.

Why (research/LEDGER.md F52, F59, F60): on Wall_Color the whole mAP50-95 gap to YOLO26x sits on tiny / thin objects
(junction 9.5 px, walls 5.6 px thick, wall_300 10 x 6 px at 1280). At the drawing's own resolution they are twice as
large (18 px, 11 px, 19 x 11 px); the same pixel error then costs half the IoU, and 53% of the wall_300 that share a
P3 cell with a junction at 1280 get their own cell (6%, F52). YOLO at 1280 never sees those pixels; to see them it
must run the whole network at 2560 (4x the cost). Here only a light stream runs at full resolution:
  hires   conv s2, conv s2, 2 residual blocks, conv s2, 2 residual blocks -> map at stride 8 of the input (= stride 4
          of the core's scale) carrying pixels the core never sees (OpenOBB13's stream, own code);
  to_p3   its stride-16 summary added to the trunk's stride-8 map before the neck (zero-init: the core starts as v14);
  P2      lateral of the stream + the neck's P3 upsampled -> RepConv -> its own dense head pair (direct boxes);
  edge    OpenOBB14's refiner sampling the full-resolution stream (stride 8 input px) instead of the trunk's stride-4
          map, and P3 at stride 16: the box edges of an 11 px wall are read at ~2 px spacing in the core's units.
Strides in input px: P3 16, P4 32, P5 64, P2 8 (= 8 / 16 / 32 / 4 at the core's scale). Every new module starts as a
no-op for P3-P5 (tests), so a v14 checkpoint loads and gives v14's P3-P5 output at step 0.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .openobb12 import HiResStream
from .openobb14 import SCALES14, EdgeRefiner, Head14, OpenOBB14, RepConv
from .openobb3 import DIRECT_CH, REG_CH
from .openobb6 import ConvBN


class OpenOBB15(OpenOBB14):
    arch = "v15"
    PAD = 64                                   # the core halves the input, then needs multiples of 32

    def __init__(self, size="x", num_classes=15, img_size=2048, max_det=1000, nms_iou=0.7, off=None, box="direct",
                 ref_k=1500, final_nms=None, neck="own", **_):
        super().__init__(size, num_classes, img_size, max_det, nms_iou, off, box, ref_k, final_nms, neck)
        hid = self.hid
        cin = self.backbone._out_channels[1]       # trunk stride-8 channels (where to_p3 is added)
        c2 = max(64, hid // 2)
        self.c2 = c2
        self.hires = HiResStream(c2)
        self.to_p3 = ConvBN(c2, cin, 3, 2, act=False, zero=True)
        self.p2_lat = ConvBN(c2, c2, 1, act=False)
        self.p2_down = ConvBN(hid, c2, 1, act=False)
        self.p2_fuse = RepConv(c2, c2)
        reg_ch = DIRECT_CH if box == "direct" else REG_CH
        self.p2_head = Head14([c2], num_classes, max(64, SCALES14[size]["head"] // 3 // 8 * 8), reg_ch)   # 76.8k cells at x
        self.p2_head_o2o = None if self.head_o2o is None else copy.deepcopy(self.p2_head)
        if self.edge is not None:                  # the refiner reads the full-resolution stream (stride 8 input px)
            self.edge = EdgeRefiner(c2, hid, s4=8, s3=16)
        self.strides = (16, 32, 64, 8)

    @torch.no_grad()
    def copy_head_to_o2o(self):
        super().copy_head_to_o2o()
        if self.p2_head_o2o is not None:
            self.p2_head_o2o.load_state_dict(self.p2_head.state_dict())

    def encode(self, x):
        """-> (full-resolution stream map at stride 8 input px, [P3, P4, P5, P2])."""
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.PAD, (-W) % self.PAD
        if ph or pw:                               # bottom / right only: anchor coordinates are unchanged
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        hr = self.hires(x)
        xc = F.avg_pool2d(x, 2)                    # the core's input: what OpenOBB14 sees at imgsz / 2
        bb = self.backbone
        h = bb.stem(xc)
        outs = []
        for stage in bb.stages:
            h = stage(h)
            outs.append(h if self.guard is None else self.guard(h))
        f4, c3, c4, c5 = outs
        c3 = c3 + self.to_p3(hr)
        feats = list(self.encoder([c3, c4, c5])) if self.neck is None else self.neck(f4, c3, c4, c5)
        p2 = self.p2_lat(hr) + F.interpolate(self.p2_down(feats[0]), size=hr.shape[-2:], mode="nearest")
        return hr, feats + [self.p2_fuse(p2)]

    def run_head(self, feats, o2o):
        o2o = o2o and self.head_o2o is not None
        c, r = (self.head_o2o if o2o else self.head)(feats[:3])
        c2, r2 = (self.p2_head_o2o if o2o else self.p2_head)(feats[3:])
        return torch.cat([c, c2], 1), torch.cat([r, r2], 1)
