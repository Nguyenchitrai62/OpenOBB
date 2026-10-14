"""OpenOBB16: OpenOBB15 + a three-stage cascade of edge refiners and fit-aware scoring. docs/OPENOBB16.md.

Why (research/LEDGER.md F59-F61): what separates OpenOBB from YOLO26x on Wall_Color is mAP50-95, i.e. precision at
IoU 0.75-0.95 on tiny / thin objects. OpenOBB14's single refiner is trained on candidates with ProbIoU >= 0.5 and
reads the edges of the *dense* box; its own output is more precise than its input, so a second refiner that reads the
edges of the refined box (and is trained on better candidates, threshold 0.6, then 0.7) can correct what the first one
could not see: the Cascade R-CNN argument, here at ~4 GFLOPs per stage because a stage is 30 bilinear samples and a
small MLP. mAP50-95 also ranks boxes: a detection with a loose box but a high class score costs more than one with a
tight box. The last stage therefore predicts the ProbIoU of its box (fit); at inference the class score of every
refined candidate becomes score x fit^0.5 before NMS (the quality-aware ranking of GFL / VarifocalNet, applied to the
refined box). At the start the cascade is the identity and fit ~ 1: OpenOBB16 begins exactly as OpenOBB15.
"""
from .openobb15 import OpenOBB15


class OpenOBB16(OpenOBB15):
    arch = "v16"
    CASCADE, QUALITY = 3, True
