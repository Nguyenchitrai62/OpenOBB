"""Per-part FLOPs, memory traffic and a roofline time estimate of a network forward, measured on the CPU with random
weights (no GPU needed; FLOPs and bytes do not depend on the weights).

Roofline model (docs/OPENOBB9.md section 4): a layer costs max(FLOPs / peak compute, bytes / memory bandwidth).
G4 (RTX PRO 6000 Blackwell): 195 TFLOPS bf16 measured (docs/COLAB.md), ~1.7 TB/s (published; the ratio ~115 FLOP/byte
is the balance point). Bytes of a leaf module = (inputs + outputs + parameters) x 2 (bf16). Functional ops between
modules (F.interpolate, torch.cat, residual adds) are not counted: the estimate is a lower bound on memory traffic,
so a model with many of those (depthwise / add-heavy designs) runs slower than this estimate says, not faster.

usage:  OMP_NUM_THREADS=1 py -3.12 -I research/analysis/roofline.py openobb12x yolo26x-obb [--hw 960x1280] [--nc 7]
Models: openobb12x | openobb14x | yolo26x-obb | yolo11x-obb (Ultralytics, research baseline only; built from its yaml).
Output: one table per model (part, GFLOPs, GB moved, FLOP/byte, roofline ms, share) + a table by operator kind.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
torch.set_num_threads(1)

PEAK_FLOPS = 195e12          # G4 bf16 (measured)
BANDWIDTH = 1.7e12           # G4 (published)


ACTS = (nn.SiLU, nn.ReLU, nn.GELU, nn.Sigmoid, nn.Hardsigmoid, nn.Hardswish, nn.LeakyReLU)


def kind_of(m):
    if isinstance(m, ACTS):
        return "activation"
    if isinstance(m, nn.Conv2d):
        if m.groups > 1 and m.groups == m.in_channels:
            return "conv depthwise"
        if m.kernel_size == (1, 1):
            return "conv 1x1"
        return "conv dense"
    if isinstance(m, (nn.Linear, nn.MultiheadAttention)):
        return "linear / attention"
    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)) or type(m).__name__.endswith("BatchNorm2d"):
        return "norm"
    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Upsample)):
        return "pool / resize"
    return "elementwise / other"


def numel_of(t):
    if torch.is_tensor(t):
        return t.numel()
    if isinstance(t, (list, tuple)):
        return sum(numel_of(u) for u in t)
    return 0


def build(name, nc, hw):
    """-> (module, forward fn(x), part_of(leaf_name) -> part label)."""
    if name.startswith("openobb"):
        from openobb.cli import MODELS
        name, _, opts = name.partition(":")           # openobb15x:neck=dfine,off=stage4 -> constructor kwargs
        arch, size = MODELS[name]
        if arch == "v12":
            from openobb.models.openobb12 import OpenOBB12
            m = OpenOBB12(size, num_classes=nc, img_size=max(hw)).eval().fuse()
            parts = [("backbone.stem", "trunk stem"), ("backbone.stages.0", "trunk stage1 /4"),
                     ("backbone.stages.1", "trunk stage2 /8"), ("backbone.stages.2", "trunk stage3 /16"),
                     ("backbone.stages.3", "trunk stage4 /32"), ("spd", "spd (stride-4 -> P3)"), ("lsk.", "LSK adapters"),
                     ("encoder.input_proj", "encoder proj 1x1"), ("encoder.encoder", "encoder AIFI"),
                     ("encoder.lateral", "encoder FPN"), ("encoder.fpn", "encoder FPN"),
                     ("encoder.downsample", "encoder PAN"), ("encoder.pan", "encoder PAN"),
                     ("head.cls", "head cls (o2m)"), ("head.ctx", "head cls (o2m)"), ("head.reg", "head box (o2m)")]

            def fn(x):
                feats = m.features(x)
                return m.head(feats)
        elif arch in ("v14", "v15", "v16"):
            from openobb.models.openobb14 import OpenOBB14
            from openobb.models.openobb15 import OpenOBB15
            kw = dict(kv.split("=") for kv in opts.split(",") if kv)
            if arch == "v16":
                from openobb.models.openobb16 import OpenOBB16
                cls = OpenOBB16
            else:
                cls = OpenOBB15 if arch == "v15" else OpenOBB14
            m = cls(size, num_classes=nc, img_size=max(hw), **kw).eval().fuse()
            parts = [("backbone.stem", "trunk stem"), ("backbone.stages.0", "trunk stage1 /4"),
                     ("backbone.stages.1", "trunk stage2 /8"), ("backbone.stages.2", "trunk stage3 /16"),
                     ("backbone.stages.3", "trunk stage4 /32"), ("neck.proj", "neck proj 1x1"), ("neck.spd", "neck spd"),
                     ("neck.attn", "neck attention /32"), ("neck.td", "neck top-down"), ("neck.lat", "neck top-down"),
                     ("neck.bu", "neck bottom-up"), ("neck.dn", "neck bottom-up"),
                     ("neck.down5", "neck bottom-up"), ("encoder.", "neck (D-FINE encoder)"),
                     ("hires", "hires stream"), ("to_p3", "hires stream"), ("p2_lat", "P2 level"), ("p2_down", "P2 level"),
                     ("p2_fuse", "P2 level"), ("p2_head.", "P2 head (o2m)"),
                     ("head.cls", "head cls (o2m)"), ("head.box", "head box (o2m)"), ("edge", "edge refiner")]

            def fn(x):
                f4, feats = m.encode(x)
                cls, reg = m.run_head(feats, False)
                pts, st = m.anchors(feats)
                from openobb.models.openobb3 import decode
                return m.refine(f4, feats[0], cls, decode(reg, pts, st), 1500)
        elif arch in ("v17", "v18", "v19", "v20", "v21", "v22"):
            from openobb.models.openobb17 import OpenOBB17
            from openobb.models import openobb18 as m18
            cls = {"v17": OpenOBB17, "v18": m18.OpenOBB18, "v19": m18.OpenOBB19, "v20": m18.OpenOBB20,
                   "v21": m18.OpenOBB21, "v22": m18.OpenOBB22}[arch]
            kw = dict(kv.split("=") for kv in opts.split(",") if kv)
            groups = None if arch in ("v17", "v18") else "1,0,2,2,2,1,0"
            m = cls(size, num_classes=nc, img_size=max(hw), groups=groups, **kw).eval().fuse()
            parts = [("backbone.stem", "trunk stem"), ("backbone.stages.0", "trunk stage1 /4"),
                     ("backbone.stages.1", "trunk stage2 /8"), ("backbone.stages.2", "trunk stage3 /16"),
                     ("backbone.stages.3", "trunk stage4 /32"), ("spd", "spd (stride-4 -> P3)"), ("lsk.", "LSK adapters"),
                     ("encoder.", "encoder (D-FINE hybrid)"), ("neck.", "own neck"),
                     ("head.cls", "head cls (o2m)"), ("head.ctx", "head cls (o2m)"), ("head.reg", "head box (o2m)"),
                     ("ang.", "head angle branch"), ("field", "boundary field"), ("snap", "boundary snapping"),
                     ("edge", "edge refiner")]

            def fn(x):
                return m(x)                                  # inference forward: o2m head, refiner, snapping, NMS
        else:
            raise SystemExit(f"{name}: not supported by this script")
    else:
        from ultralytics.nn.tasks import OBBModel  # research baseline only
        cfg = {"yolo26x-obb": "yolo26x-obb.yaml", "yolo11x-obb": "yolo11x-obb.yaml"}[name]
        m = OBBModel(cfg, nc=nc, verbose=False).eval()
        m.fuse(verbose=False)
        strides = {}

        def tag(i):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, (list, tuple)) and torch.is_tensor(out[0]) else out
                if torch.is_tensor(o) and o.dim() == 4:
                    strides[i] = max(hw) // o.shape[-1] if o.shape[-1] else 0
            return hook
        for i, layer in enumerate(m.model):
            layer.register_forward_hook(tag(i))
        with torch.no_grad():
            m(torch.rand(1, 3, *hw))
        parts = []
        names = [type(layer).__name__ for layer in m.model]
        last_bb = max(i for i, t in enumerate(names) if t in ("SPPF", "C2PSA", "PSA"))
        for i, t in enumerate(names):
            if i == len(names) - 1:
                lab = "head"
            else:
                lab = f"{'trunk' if i <= last_bb else 'neck'} /{strides.get(i, '?')}"
            parts.append((f"model.{i}.", lab))

        def fn(x):
            return m(x)

    def part_of(leaf):
        for p, lab in parts:
            if leaf.startswith(p):
                return lab
        return "other"
    return m, fn, part_of


def measure(name, nc, hw):
    m, fn, part_of = build(name, nc, hw)
    leaves = {n: mod for n, mod in m.named_modules()
              if not list(mod.children()) and not isinstance(mod, (nn.Identity, nn.Dropout))}
    traffic, calls = {}, {}

    def hook(n):
        def h(mod, inp, out):
            b = numel_of(inp) + numel_of(out) + sum(p.numel() for p in mod.parameters(recurse=False))
            traffic[n] = traffic.get(n, 0) + 2 * b
            calls[n] = calls.get(n, 0) + 1
        return h
    hs = [mod.register_forward_hook(hook(n)) for n, mod in leaves.items()]
    x = torch.rand(1, 3, *hw)

    class _Run(nn.Module):                  # the counter names modules from the first module called: make it a root
        def __init__(self):
            super().__init__()
            self.net = m

        def forward(self, t):
            return fn(t)
    runner = _Run()
    t0 = time.perf_counter()
    with torch.no_grad(), FlopCounterMode(display=False, depth=None) as fc:
        runner(x)
    wall = time.perf_counter() - t0
    for h in hs:
        h.remove()
    counts = fc.get_flop_counts()
    flops = {}
    for k, v in counts.items():
        if k.startswith("_Run.net."):
            leaf = k[len("_Run.net."):]
            if leaf in leaves:
                flops[leaf] = sum(v.values())
    total_f = sum(counts.get("Global", {}).values()) or sum(flops.values())
    rows = {}
    kinds = {}
    for n, mod in leaves.items():
        f, b = flops.get(n, 0), traffic.get(n, 0)
        if f == 0 and b == 0:
            continue
        t = max(f / PEAK_FLOPS, b / BANDWIDTH)
        for key, table in ((part_of(n), rows), (kind_of(mod), kinds)):
            r = table.setdefault(key, [0, 0, 0.0])
            r[0] += f
            r[1] += b
            r[2] += t
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"\n== {name}: {params:.1f}M params, input {hw[0]}x{hw[1]}, total {total_f / 1e9:.0f} GFLOPs "
          f"(FlopCounterMode, 2 FLOP per MAC); CPU 1-thread wall {wall:.1f} s (with counting overhead)")
    for title, table in (("part", rows), ("operator kind", kinds)):
        tf = sum(r[0] for r in table.values())
        tb = sum(r[1] for r in table.values())
        tt = sum(r[2] for r in table.values())
        print(f"\n{title:28s} {'GFLOPs':>8s} {'%F':>5s} {'GB':>7s} {'%B':>5s} {'F/B':>6s} {'roof ms':>8s} {'%t':>5s}")
        order = sorted(table.items(), key=lambda kv: -kv[1][2]) if title != "part" else table.items()
        for k, (f, b, t) in order:
            print(f"{k:28s} {f / 1e9:8.1f} {100 * f / tf:5.1f} {b / 1e9:7.3f} {100 * b / tb:5.1f} "
                  f"{(f / b if b else 0):6.0f} {1e3 * t:8.2f} {100 * t / tt:5.1f}")
        print(f"{'TOTAL':28s} {tf / 1e9:8.1f} {'':5s} {tb / 1e9:7.3f} {'':5s} {(tf / tb if tb else 0):6.0f} "
              f"{1e3 * tt:8.2f}")
    return rows, kinds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--hw", default="960x1280")
    ap.add_argument("--nc", type=int, default=7)
    a = ap.parse_args()
    hw = tuple(int(v) for v in a.hw.split("x"))
    for name in a.models:
        measure(name, a.nc, hw)
