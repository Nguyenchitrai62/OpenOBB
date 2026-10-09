# Third-party notices

VRDet (this repository) is proprietary software of the repository owner (see `LICENSE`).
It incorporates a small amount of code from the following permissively licensed projects. Their licenses
allow commercial use, modification and closed-source distribution, provided the copyright notices and
the license text are kept and modified files say they were changed.

| Component | Upstream | License | Files in this repo | Changes |
|---|---|---|---|---|
| HGNetv2 backbone | D-FINE (github.com/Peterande/D-FINE), ported from PaddleDetection | Apache-2.0 | `openobb/models/hgnetv2.py` | config registry and distributed coupling removed; pretrained loading simplified |
| Hybrid encoder (AIFI + CCFM) | DEIM (github.com/ShihuaHuang95/DEIM), from D-FINE / RT-DETR | Apache-2.0 | `openobb/models/hybrid_encoder.py` | registry removed; optional `ctx_fn` hook for VRDet global context |
| Deformable-attention core, utilities | DEIM / D-FINE / RT-DETR | Apache-2.0 | `openobb/models/deim_utils.py` | registry removed |
| FrozenBatchNorm2d | RT-DETR / DETR | Apache-2.0 | `openobb/models/common.py` | none |
| Decoder layer structure, FDR weighting function, contrastive denoising, Hungarian matcher, MAL / FGL / DDF loss structure | D-FINE, DEIM | Apache-2.0 | `openobb/models/obb_decoder.py`, `obb_criterion.py`, `obb_utils.py` | rewritten for oriented boxes (rotated FDR with angle distribution, rotated sampling, Chamfer/KLD matching, rotated-IoU MAL targets, oriented denoising) |

Apache-2.0 license texts: `LICENSES/Apache-2.0-D-FINE.txt`, `LICENSES/Apache-2.0-DEIM.txt`.

All other code in `openobb/`, `colab/data/`, `tools/`, `tests/` is original VRDet code.

## Pretrained weights

- Initialisation uses D-FINE **COCO-only** checkpoints (`dfine_{s,m,l,x}_coco.pth`, Apache-2.0 repository).
  The Objects365 checkpoints are deliberately NOT used: D-FINE notes they may be subject to Objects365 terms.
- VRDet4 (`model=vrdet4x`) downloads the **DINOv2 ViT-B/14 (registers)** weights from Meta
  (github.com/facebookresearch/dinov2, code and weights Apache-2.0) at training start. The ViT code in
  `openobb/models/vit.py` is VRDet's own; only the parameter names follow the released checkpoints. Keep the DINOv2
  copyright notice and the Apache-2.0 text when distributing weights derived from it.
- Weights trained on research datasets (DOTA: academic use only; FloorPlanCAD: CC BY-NC 4.0) are for
  benchmarking. Commercial weights should be fine-tuned from the COCO initialisation on data the owner has
  rights to.

## Not part of VRDet

- **Ultralytics (AGPL-3.0)** is only used by `colab/baselines/yolo_obb.py` to train and measure YOLO baselines
  on Colab. It is never imported by `openobb/` (enforced by `tests/test_license_guard.py`) and is not distributed.
- **PyMuPDF (AGPL-3.0, or a paid Artifex licence)** is what AI_Takeoff uses to read PDF vectors. The VRDet vector
  branch only consumes a neutral token array (`openobb/models/vector.py`, 21 floats per primitive), so the PDF reader
  is replaceable: for a fully permissive product use pypdfium2 (Apache-2.0 / BSD-3, PDFium path objects) or
  pdfminer.six (MIT) to produce the tokens, or keep PyMuPDF under its commercial licence.
- Ideas taken from published papers (O2-DETR, RiO-DETR, YOLO26, PP-YOLOE-R, RTMDet-R, DDQ, LSKNet, Strip R-CNN,
  RF-DETR, ViTDet, ViT-Adapter, Co-DETR, RT-DETRv3, ...) were re-implemented from the papers; no code was copied from non-commercial repositories.
