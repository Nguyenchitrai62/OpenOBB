"""
CAD-VLMDet-OBB — Hybrid Vision-Language Encoder + YOLO-style OBB Decoder
for floorplan / CAD / BIM large-image detection.

License: MIT (commercial-friendly, no Ultralytics AGPL code).
Idea synthesis (OpenResearch-style literature review):
 - FloorPlanFormer (AAAI'26): Swin + pixel decoder + GCAM outer/inner refinement
 - FloorplanVLM (2026): pixels-to-sequence + GRPO geometric constraints
 - MMFE (2026): frozen DINOv3 + DPT dense head, per-cell InfoNCE
 - MuraNet (2023): unified attention encoder + YOLOX decoupled head + seg branch
 - EDANet (2026): strip pooling for anisotropic walls + gated boundary fusion

Design choices for this repo:
 - Hierarchical hybrid encoder (ConvNeXt-lite + 2x global self-attention at P5)
   -> "hiểu ảnh như VLM": long-range wall/window/door relations.
 - Strip-pooling Global Context Block (walls are long thin structures).
 - Learnable text/class embeddings, classifier = dot-product (open-vocab, YOLO-World style).
 - Dense anchor-free decoupled OBB head (cls / box-DFL / angle), FCOS-style assignment
   -> "sinh bbox nhanh như YOLO", handles 150-400 boxes/image (DETR queries would fail).
 - Aux heads: junction heatmap + wall mask (multi-task, like MuraNet).
 - Large-image: SAHI tiling + rotated NMS merge, AMP, torch.compile, ONNX export.
"""

CLASS_NAMES = ["wall", "window", "door", "slide_door", "double_door", "opening", "junction"]
NC = 7
# Topology prior mined from floorplan literature + dataset:
#  - window/door/slide_door/double_door/opening MUST lie on/overlap a wall
#  - junction = wall endpoints / corners (tiny, stride-4 features needed)
TOPOLOGY_RULES = {
    "window_on_wall": True,
    "door_on_wall": True,
    "junction_is_endpoint": True,
}
