"""ONNX export: `openobb export model=best.pt` -> best.onnx, run without PyTorch by the openobb_onnx package.

The ONNX graph holds the whole network (trunk, neck, heads, edge refiner, boundary snapping, localisation quality, the
calibrated inference head / quality / peak settings) up to the class-wise NMS, which runs in numpy (openobb_onnx/ops.py;
its result is that of fast_nms). Graph:
    images  uint8 (B, H, W, 3) BGR, H and W multiples of 32 (any size: dynamic axes), padded as in training
 -> rank    (B, K) float  ranking score of the K best (candidate, class) pairs, descending
    score   (B, K) float  their output score
    label   (B, K) int64  their class
    box     (B, K, 5)     their box (cx, cy, w, h, theta) in input px
K = PRE (4000, the pre-selection of fast_nms) with NMS; without NMS (one-to-one head, NMS-free calibration) the pairs of
the max_det best candidates. The settings the runtime needs (class names, image size, per-class thresholds, NMS) are
in the model's metadata (METADATA_KEY, JSON). Names of layers, weights and intermediate tensors are replaced by
neutral ones (n12, w34, t56) and the exporter's source annotations are removed: the file carries the computation and
the weights, not the code's structure or names.
Only dense models with an NMS-free inference path before the NMS are exported (v17-v24).
"""
import json
import math
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

PRE = 4000                     # (candidate, class) pairs kept before the NMS (fast_nms default)
METADATA_KEY = "obb"           # JSON metadata entry read by the runtime
FORMAT = 1                     # version of the graph interface above
MIN_SIDE = 64                  # smallest input side of the graph (2 cells at stride 32); the top-k sizes follow the input
INPUT, OUTPUTS = "images", ("rank", "score", "label", "box")


def _anchors(self, feats):
    """OpenOBB3.anchors without its per-shape cache (the cache keys on concrete sizes; the export traces symbolic
    ones). Same points: cell centres (i + 0.5) x stride."""
    pts, st = [], []
    for f, s in zip(feats, self.strides):
        h, w = f.shape[-2:]
        ys = (torch.arange(h, device=f.device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=f.device, dtype=torch.float32) + 0.5) * s
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        p = torch.stack([gx, gy], -1).reshape(-1, 2)
        pts.append(p)
        st.append(torch.full_like(p[:, 0], float(s)))
    return torch.cat(pts), torch.cat(st)


class ExportNet(nn.Module):
    """uint8 BGR NHWC images -> the PRE best (candidate, class) pairs of a dense OpenOBB model (see module doc)."""

    def __init__(self, model, nms, max_det, pre=PRE):
        super().__init__()
        self.m, self.nms, self.max_det, self.pre = model, nms, int(max_det), int(pre)
        self.m.anchors = types.MethodType(_anchors, self.m)

    def forward(self, images):
        x = images.flip(-1).permute(0, 3, 1, 2).float() / 255.0          # BGR NHWC uint8 -> RGB NCHW [0, 1]
        rank, out_l, boxes, _ = self.m.scored_candidates(x)
        out = rank if out_l is None else out_l
        C = rank.shape[-1]
        if self.nms is None:                                               # NMS-free: the max_det best candidates
            k = min(self.max_det, rank.shape[1])
            order = rank.amax(-1).topk(k, dim=1).indices
            rank = out = out.gather(1, order[..., None].expand(-1, -1, C))
            boxes = boxes.gather(1, order[..., None].expand(-1, -1, 5))
        s = rank.float().sigmoid().flatten(1)
        top, idx = s.topk(min(self.pre, s.shape[1]), dim=1)
        score = out.float().sigmoid().flatten(1).gather(1, idx)
        box = boxes.float().gather(1, (idx // C)[..., None].expand(-1, -1, 5))
        return top, score, idx % C, box


def _nms_value(model):
    """The model's calibrated NMS after the peak filter (applied inside the graph) -> threshold or None."""
    _, nms = model.nms_setting()
    if isinstance(nms, str) and nms.startswith("peak"):
        return float(nms.split("+")[1]) if "+" in nms else None
    return None if nms is None else float(nms)


def anonymize(proto):
    """Neutral names for every node, weight and intermediate tensor (graph inputs / outputs keep theirs); drops doc
    strings, value_info and the exporter's per-node metadata (source lines, module paths)."""
    g = proto.graph
    keep = {v.name for v in g.input} | {v.name for v in g.output}
    ren = {}
    for i, t in enumerate(g.initializer):
        ren[t.name] = f"w{i}"
    k = 0

    def walk(graph):
        nonlocal k
        for node in graph.node:
            for o in node.output:
                if o and o not in keep and o not in ren:
                    ren[o] = f"t{k}"
                    k += 1
            for a in node.attribute:
                if a.g is not None and a.g.node:
                    walk(a.g)
                for sg in a.graphs:
                    walk(sg)

    walk(g)

    def apply(graph):
        for t in graph.initializer:
            t.name = ren.get(t.name, t.name)
            t.doc_string = ""
            del t.metadata_props[:]
        for j, node in enumerate(graph.node):
            node.name = f"n{j}"
            node.doc_string = ""
            del node.metadata_props[:]
            for lst in (node.input, node.output):
                for q, nm in enumerate(lst):
                    lst[q] = ren.get(nm, nm)
            for a in node.attribute:
                a.doc_string = ""
                if a.g is not None and a.g.node:
                    apply(a.g)
                for sg in a.graphs:
                    apply(sg)
        for vi in list(graph.input) + list(graph.output):
            vi.name = ren.get(vi.name, vi.name)
            vi.doc_string = ""
            del vi.metadata_props[:]
        del graph.value_info[:]
        graph.doc_string = ""
        del graph.metadata_props[:]

    apply(g)
    # readable symbolic sizes: batch, height, width of the input; batch, K of the outputs
    for vi in g.input:
        for d, nm in zip(vi.type.tensor_type.shape.dim, ("batch", "height", "width", None)):
            if nm and d.dim_param:
                d.dim_param = nm
    for vi in g.output:
        for d, nm in zip(vi.type.tensor_type.shape.dim, ("batch", "K", None)):
            if nm and (d.dim_param or not d.HasField("dim_value")):
                d.dim_param = nm
    g.name = "graph"
    proto.doc_string = ""
    proto.producer_name, proto.producer_version = "obb", ""
    del proto.metadata_props[:]
    return proto


def metadata(args, names, nms, max_det, pre, size):
    """Runtime settings (JSON in the ONNX metadata): nms = the network's NMS threshold (None: NMS-free output),
    merge_iou = polygon NMS of the detection merge for a whole-image model (predict.merge_iou_for)."""
    from openobb import __version__
    from openobb.predict import merge_iou_for
    thr = args.get("conf_thr") or {}
    return {"format": FORMAT, "version": __version__, "names": list(names), "imgsz": int(size),
            "fit": bool(args.get("fit", False)), "scale": float(args.get("scale", 1.0) or 1.0), "stride": 32,
            "min_side": MIN_SIDE, "nms": nms, "merge_iou": merge_iou_for(args.get("nms_iou"), True),
            "max_det": int(max_det), "pre": int(pre), "conf_thr": {n: float(thr[n]) for n in names if n in thr}}


def export(model, out=None, imgsz=None, opset=18, batch=1, verbose=True):
    """model: OpenOBB checkpoint (.pt) -> path of the .onnx file (default next to it). imgsz: input size the
    runtime resizes to (default: the trained one). The graph accepts any size / batch (dynamic axes)."""
    import onnx

    from openobb.predict import input_size, load_model
    src = Path(model)
    t0 = time.perf_counter()
    m, names, a = load_model(str(src), torch.device("cpu"), 900)       # as OpenOBB(.pt): max_det >= 900
    if not hasattr(m, "scored_candidates"):
        raise SystemExit(f"{src}: architecture {a.get('arch')} cannot be exported to ONNX (OpenOBB17 - 24 only)")
    if not a.get("fit", False):
        print("[export] note: model trained with tiles; the runtime tiles the image as `openobb predict` does")
    nms = _nms_value(m)
    size = input_size(a, imgsz)
    net = ExportNet(m, nms, m.max_det).eval()
    side = max(256, math.ceil(min(size, 640) / 32) * 32)               # example input of the export trace
    # example batch 2 and h != w: a size of 1 or two equal sizes would be specialised (fixed) by the exporter
    x = torch.full((max(2, int(batch)), side, side + 32, 3), 255, dtype=torch.uint8)
    hb, wb = torch.export.Dim("hb", min=MIN_SIDE // 32, max=512), torch.export.Dim("wb", min=MIN_SIDE // 32, max=512)
    dyn = {"images": {0: torch.export.Dim("batch", min=1, max=256), 1: 32 * hb, 2: 32 * wb}}
    out = Path(out) if out else src.with_suffix(".onnx")
    with torch.no_grad():
        prog = torch.onnx.export(net, (x,), input_names=[INPUT], output_names=list(OUTPUTS), opset_version=int(opset),
                                 dynamic_shapes=dyn, dynamo=True, optimize=True, external_data=False, verbose=False)
    proto = prog.model_proto
    try:
        from onnx.inliner import inline_local_functions
        proto = inline_local_functions(proto)
    except ImportError:
        pass
    proto = anonymize(proto)
    meta = metadata(a, names, nms, m.max_det, PRE, size)
    e = proto.metadata_props.add()
    e.key, e.value = METADATA_KEY, json.dumps(meta)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))
    if verbose:
        print(f"[export] {src.name} -> {out} ({out.stat().st_size / 2**20:.0f} MB, opset {opset}, "
              f"{len(proto.graph.node)} nodes, NMS {nms}, imgsz {size}) in {time.perf_counter() - t0:.0f} s")
    return str(out)
