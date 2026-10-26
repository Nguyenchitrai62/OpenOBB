# Third-party notices

OpenOBB (this repository) is proprietary software of the repository owner (see `LICENSE`).
It incorporates a small amount of code from the following permissively licensed projects. Their licenses
allow commercial use, modification and closed-source distribution, provided the copyright notices and
the license text are kept and modified files say they were changed.

| Component | Upstream | License | Files in this repo | Changes |
|---|---|---|---|---|
| HGNetv2 backbone | D-FINE (github.com/Peterande/D-FINE), ported from PaddleDetection | Apache-2.0 | `openobb/models/hgnetv2.py` | config registry and distributed coupling removed; pretrained loading simplified |
| Hybrid encoder (AIFI + CCFM) | DEIM (github.com/ShihuaHuang95/DEIM), from D-FINE / RT-DETR | Apache-2.0 | `openobb/models/hybrid_encoder.py` | registry removed; optional `ctx_fn` hook for OpenOBB global context; the CSP-ELAN fusion block (`CSPELAN`) and the RepVGG kernel fusion (`VGGBlock.get_equivalent_kernel_bias`) rewritten by OpenOBB on 2026-10-22 (the upstream text of those two matched YOLOv9 / Ultralytics code, which D-FINE could not relicense; licence audit of 2026-10-22) |
| Deformable-attention core, utilities | DEIM / D-FINE / RT-DETR | Apache-2.0 | `openobb/models/deim_utils.py` | registry removed |
| FrozenBatchNorm2d | RT-DETR / DETR | Apache-2.0 | `openobb/models/common.py` | none |
| Decoder layer structure, FDR weighting function, contrastive denoising, Hungarian matcher, MAL / FGL / DDF loss structure | D-FINE, DEIM | Apache-2.0 | `openobb/models/obb_decoder.py`, `obb_criterion.py`, `obb_utils.py` | rewritten for oriented boxes (rotated FDR with angle distribution, rotated sampling, Chamfer/KLD matching, rotated-IoU MAL targets, oriented denoising) |

| RepVGG structural re-parameterisation (3x3 + 1x1 + BN folded into one 3x3 conv) | idea of Ding et al. 2021 (reference code github.com/DingXiaoH/RepVGG, MIT) | MIT (idea; code is OpenOBB's) | `openobb/models/hybrid_encoder.py` `VGGBlock`, `openobb/models/openobb9.py`, `openobb14.py` | own implementation |

Apache-2.0 license texts: `LICENSES/Apache-2.0-D-FINE.txt`, `LICENSES/Apache-2.0-DEIM.txt`. This file and `LICENSES/` are shipped
inside every wheel (`dist-info/licenses/`), including the ONNX handover wheel, because the exported weights derive from the
D-FINE COCO initialisation.

All other code in `openobb/`, `colab/data/`, `tools/`, `tests/` is original OpenOBB code.

## Pretrained weights

- Initialisation uses D-FINE **COCO-only** checkpoints (`dfine_{s,m,l,x}_coco.pth`, Apache-2.0 repository).
  The Objects365 checkpoints are deliberately NOT used: D-FINE notes they may be subject to Objects365 terms.
- OpenOBB4 (`model=openobb4x`) downloads the **DINOv2 ViT-B/14 (registers)** weights from Meta
  (github.com/facebookresearch/dinov2, code and weights Apache-2.0) at training start. The ViT code in
  `openobb/models/vit.py` is OpenOBB's own; only the parameter names follow the released checkpoints. Keep the DINOv2
  copyright notice and the Apache-2.0 text when distributing weights derived from it.
- Weights trained on research datasets (DOTA: academic use only; FloorPlanCAD: CC BY-NC 4.0) are for
  benchmarking. Commercial weights should be fine-tuned from the COCO initialisation on data the owner has
  rights to.

## Not part of OpenOBB

- **Ultralytics (AGPL-3.0)** is only used by `colab/baselines/yolo_obb.py` to train and measure YOLO baselines
  on Colab. It is never imported by `openobb/` (enforced by `tests/test_license_guard.py`) and is not distributed.
  Where OpenOBB reproduces Ultralytics *behaviour* (the `Results` / `OpenOBB` interfaces, the `metric=yolo` validator
  conventions in `openobb/eval/ultra.py`, the training recipe and augmentation defaults, the `Speed:` line), the code was
  written from the documented behaviour. A token-level comparison against Ultralytics 8.4 on 2026-10-22
  (licence audit; token-shingle comparison of every function) found four short functions that
  matched its text; they were rewritten the same day with tests that fix their numbers (`tests/test_license_rewrites.py`).
- **SAHI (MIT)** is an optional dependency (`pip install sahi`, extra `[sahi]`) for sliced inference (`openobb/sahi.py`,
  `openobb_onnx/sahi.py`). It is not distributed with OpenOBB. OpenOBB's SAHI model subclasses SAHI's
  `UltralyticsDetectionModel` (SAHI code, MIT) so that SAHI applies its OBB rules; it does not import or need Ultralytics.
- **PyMuPDF (AGPL-3.0, or a paid Artifex licence)** is what AI_Takeoff uses to read PDF vectors. The OpenOBB vector
  branch only consumes a neutral token array (`openobb/models/vector.py`, 21 floats per primitive), so the PDF reader
  is replaceable: for a fully permissive product use pypdfium2 (Apache-2.0 / BSD-3, PDFium path objects) or
  pdfminer.six (MIT) to produce the tokens, or keep PyMuPDF under its commercial licence.
- Ideas taken from published papers (O2-DETR, RiO-DETR, YOLO26, PP-YOLOE-R, RTMDet-R, DDQ, LSKNet, Strip R-CNN,
  RF-DETR, ViTDet, ViT-Adapter, Co-DETR, RT-DETRv3, ...) were re-implemented from the papers; no code was copied from non-commercial repositories.
