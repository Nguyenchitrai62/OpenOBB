"""The same roofline measurement (research/analysis/roofline.py) applied to SOTA backbones side by side, built as
architectures only (random weights; FLOPs and bytes do not depend on the weights). Research only: timm and
Ultralytics are measurement tools here, nothing of them enters openobb/.

For every backbone: parameters, GFLOPs, GB moved (bf16, leaf modules), FLOP / byte, roofline ms on a G4
(195 TFLOPS, 1.7 TB/s), the share of the roofline time per operator kind (dense conv / 1x1 / depthwise / attention /
activation) and the FLOPs per output stride (where the compute sits). Backbone = everything before the neck
(YOLO: layers up to SPPF / C2PSA; timm: features_only).

usage: OMP_NUM_THREADS=1 py -3.12 research/analysis/roofline_backbones.py [--hw 960x1280] [--only name,name]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from roofline import BANDWIDTH, PEAK_FLOPS, kind_of, numel_of  # noqa: E402

torch.set_num_threads(1)


def yolo_backbone(yaml, nc=7):
    from ultralytics.nn.tasks import DetectionModel, OBBModel
    cls = OBBModel if "obb" in yaml else DetectionModel
    m = cls(yaml, nc=nc, verbose=False).eval()
    m.fuse(verbose=False)
    names = [type(l).__name__ for l in m.model]
    last = max(i for i, t in enumerate(names) if t in ("SPPF", "C2PSA", "PSA", "A2C2f"))
    layers = nn.Sequential(*[m.model[i] for i in range(last + 1)])
    return layers


def hgnet(name):
    from openobb.models.hgnetv2 import HGNetv2
    from openobb.models.openobb7 import fuse_model
    m = HGNetv2(name=name, use_lab=name in ("B0", "B1", "B2"), return_idx=[0, 1, 2, 3], freeze_norm=False,
                pretrained=False).eval()
    return fuse_model(m)


def timm_model(name, hw, **kw):
    import timm
    m = timm.create_model(name, pretrained=False, features_only=True, **kw).eval()
    return m


def build(name, hw):
    if name.startswith("hgnetv2_"):
        return hgnet(name.split("_")[1].upper())
    if name.startswith("yolo"):
        return yolo_backbone(name + ".yaml")
    if name == "rtdetr_r50":
        return timm_model("resnet50", hw)
    if name.startswith("dinov2"):
        return timm_model("vit_base_patch14_dinov2", hw, dynamic_img_size=True)
    if name.startswith("swin"):
        return timm_model("swin_base_patch4_window7_224", hw, img_size=hw, strict_img_size=False)
    return timm_model(name, hw)


def measure(name, hw):
    m = build(name, hw)
    leaves = {n: mod for n, mod in m.named_modules()
              if not list(mod.children()) and not isinstance(mod, (nn.Identity, nn.Dropout))}
    traffic, strides = {}, {}

    def hook(n):
        def h(mod, inp, out):
            b = numel_of(inp) + numel_of(out) + sum(p.numel() for p in mod.parameters(recurse=False))
            traffic[n] = traffic.get(n, 0) + 2 * b
            o = out[0] if isinstance(out, (list, tuple)) and len(out) and torch.is_tensor(out[0]) else out
            if torch.is_tensor(o) and o.dim() == 4 and o.shape[-1]:
                strides[n] = round(hw[1] / o.shape[-1])
            elif torch.is_tensor(o) and o.dim() == 3:
                strides[n] = "tok"
        return h
    hs = [mod.register_forward_hook(hook(n)) for n, mod in leaves.items()]
    x = torch.rand(1, 3, *hw)
    if name.startswith("dinov2"):
        x = torch.rand(1, 3, hw[0] // 14 * 14, hw[1] // 14 * 14)

    class _Run(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = m

        def forward(self, t):
            return self.net(t)
    runner = _Run()
    t0 = time.perf_counter()
    with torch.no_grad(), FlopCounterMode(display=False, depth=None) as fc:
        runner(x)
    wall = time.perf_counter() - t0
    for h in hs:
        h.remove()
    flops = {}
    for k, v in fc.get_flop_counts().items():
        if k.startswith("_Run.net."):
            leaf = k[len("_Run.net."):]
            if leaf in leaves:
                flops[leaf] = sum(v.values())
    kinds, by_stride = {}, {}
    tf = tb = tt = 0.0
    for n, mod in leaves.items():
        f, b = flops.get(n, 0), traffic.get(n, 0)
        if f == 0 and b == 0:
            continue
        t = max(f / PEAK_FLOPS, b / BANDWIDTH)
        k = kinds.setdefault(kind_of(mod), [0, 0, 0.0])
        k[0] += f
        k[1] += b
        k[2] += t
        s = by_stride.setdefault(str(strides.get(n, "?")), 0)
        by_stride[str(strides.get(n, "?"))] = s + f
        tf, tb, tt = tf + f, tb + b, tt + t
    params = sum(p.numel() for p in m.parameters()) / 1e6
    row = {"params_M": round(params, 1), "GFLOPs": round(tf / 1e9, 1), "GB": round(tb / 1e9, 3),
           "flop_per_byte": round(tf / tb, 1) if tb else 0, "roof_ms": round(1e3 * tt, 2),
           "time_share": {k: round(100 * v[2] / tt, 1) for k, v in kinds.items()},
           "flops_share_by_stride": {k: round(100 * v / tf, 1) for k, v in sorted(by_stride.items(), key=lambda kv: str(kv[0]))},
           "cpu_wall_s": round(wall, 1)}
    return row


DEFAULT = ["hgnetv2_b5", "hgnetv2_b4", "yolo26x-obb", "yolo11x-obb", "yolo12x", "yolov8x", "convnext_base",
           "convnextv2_base", "resnet50", "resnet101", "swin_base", "dinov2_vitb14", "efficientnet_b5",
           "repvgg_b2", "regnety_032", "cspdarknet53"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="960x1280")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "roofline_backbones.json"))
    a = ap.parse_args()
    hw = tuple(int(v) for v in a.hw.split("x"))
    names = [s for s in a.only.split(",") if s] or DEFAULT
    res = {}
    print(f"{'backbone':18s} {'M':>6s} {'GFLOPs':>7s} {'GB':>6s} {'F/B':>5s} {'roof':>6s} | time share: dense / 1x1 / dw / "
          f"attn / act | FLOPs by stride")
    for n in names:
        try:
            r = measure(n, hw)
        except Exception as e:  # noqa: BLE001
            print(f"{n:18s} failed: {str(e)[:90]}")
            continue
        res[n] = r
        ts = r["time_share"]
        g = lambda k: ts.get(k, 0.0)  # noqa: E731
        bs = " ".join(f"/{k}:{v:.0f}" for k, v in r["flops_share_by_stride"].items() if v >= 1)
        print(f"{n:18s} {r['params_M']:6.1f} {r['GFLOPs']:7.0f} {r['GB']:6.2f} {r['flop_per_byte']:5.0f} "
              f"{r['roof_ms']:6.2f} | {g('conv dense'):4.0f} / {g('conv 1x1'):4.0f} / {g('conv depthwise'):4.0f} / "
              f"{g('linear / attention'):4.0f} / {g('activation'):4.0f} | {bs}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"hw": hw, "results": res}, f, indent=1)
    print(f"saved {a.out}")
