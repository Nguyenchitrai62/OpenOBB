"""Training cost of one image (forward + backward FLOPs) of a dense OpenOBB vs YOLO26x-obb, split by block (F112:
is the training-speed gap to YOLO the architecture?). CPU, FLOP counter only (no timing). Ultralytics is imported here
for measuring only (research script, never in openobb/).

usage: OMP_NUM_THREADS=1 py -3.12 research/analysis/train_flops.py [checkpoint.pt] [imgsz] [objects]
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)

ck_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Downloads\openobb24x.pt"
img = int(sys.argv[2]) if len(sys.argv) > 2 else 960
objects = int(sys.argv[3]) if len(sys.argv) > 3 else 150


def count(fn, model):
    fc = FlopCounterMode(display=False, depth=None)
    with fc:
        fn()
    per = fc.get_flop_counts()
    total = sum(per["Global"].values()) if "Global" in per else 0
    return total, per


def top_level(per, prefixes):
    out = defaultdict(float)
    for name, d in per.items():
        parts = name.split(".")
        if len(parts) == 2 and parts[1] in prefixes:
            out[parts[1]] += sum(d.values())
    return out


# ---- OpenOBB (training forward + loss + backward)
from openobb.models.openobb18_loss import OpenOBB18Loss  # noqa: E402
from openobb.profile import _synthetic_targets  # noqa: E402

ck = torch.load(ck_path, map_location="cpu", weights_only=False)
a, names = ck["args"], ck["classes"]
from openobb.models.openobb24 import OpenOBB24  # noqa: E402

m = OpenOBB24(a.get("size", "x"), num_classes=len(names), img_size=img, groups=a.get("groups"),
              ref_k=int(a.get("ref_k", 1500)), off24=a.get("v24_off") or None).train()
m.load_state_dict(ck["model"], strict=False)
crit = OpenOBB18Loss(img_size=img)
x = torch.rand(1, 3, img, img)
tg = _synthetic_targets(1, objects, len(names), img, "cpu")


def step_obb():
    out = m(x, tg)
    sum(crit(out, tg).values()).backward()


tot, per = count(step_obb, m)
blocks = top_level(per, {"backbone", "neck", "head", "head_o2o", "ang", "ang_o2o", "field", "edge", "snap", "quality"})
print(f"OpenOBB24-{a.get('size', 'x')} train step, 1 x 3 x {img} x {img}, {objects} labels: "
      f"{tot / 1e9:.0f} GFLOPs (fwd + bwd)")
for k, v in sorted(blocks.items(), key=lambda t: -t[1]):
    print(f"  {k:10s} {v / 1e9:8.0f}  {100 * v / tot:5.1f}%")
print(f"  {'rest':10s} {(tot - sum(blocks.values())) / 1e9:8.0f}  (loss, decode, assignment)")
tr = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"  parameters {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M, trainable {tr:.1f}M")

# ---- YOLO26x-obb (training forward of both heads + backward of the sum of the outputs: same graph as its loss)
try:
    from ultralytics.nn.tasks import OBBModel
    y = OBBModel("yolo26x-obb.yaml", nc=len(names), verbose=False).train()
    xy = torch.rand(1, 3, img, img)

    def step_yolo():
        out = y(xy)
        flat = []

        def walk(o):
            if torch.is_tensor(o):
                flat.append(o.float().sum())
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
        walk(out)
        sum(flat).backward()

    ty, py = count(step_yolo, y)
    print(f"YOLO26x-obb train step, same input: {ty / 1e9:.0f} GFLOPs (fwd + bwd); OpenOBB / YOLO = {tot / ty:.2f}x")
    yb = defaultdict(float)
    for name, d in py.items():
        parts = name.split(".")
        if len(parts) == 4 and parts[1] == "model":
            yb[int(parts[2])] += sum(d.values())
    head = max(yb) if yb else None
    if head is not None:
        print(f"  YOLO head (layer {head}) {yb[head] / 1e9:.0f} GFLOPs ({100 * yb[head] / ty:.1f}%)")
except Exception as e:  # noqa: BLE001
    print(f"YOLO26x-obb not measured: {e}")
