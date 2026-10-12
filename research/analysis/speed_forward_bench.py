"""Raw network forward, 1 thread, same input shapes for every model. argv[1] = yolo | v3 | v8."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ul"))
sys.path.insert(0, r"F:\Source_code\NEW_architecture")
torch.set_num_threads(1)
which = sys.argv[1]
if which == "yolo":
    from ultralytics import YOLO
    net = YOLO(r"F:\Downloads\wall_color.pt").model.fuse().eval()
    name = f"YOLO {net.yaml.get('yaml_file', '')}"
else:
    from openobb import OpenOBB
    m = OpenOBB(rf"F:\Downloads\openobb{which[1]}x.pt")
    net = m.model.eval()
    net.nms_conf = 0.25
    name = f"OpenOBB{which[1]}-x"
params = sum(p.numel() for p in net.parameters()) / 1e6
with torch.inference_mode():
    for shape in [tuple(int(v) for v in s.split("x")) for s in sys.argv[2:]]:
        x = torch.rand(1, 3, *shape)
        net(x)
        ts = []
        for _ in range(4):
            t = time.perf_counter()
            net(x)
            ts.append(time.perf_counter() - t)
        print(f"{name:28s} {params:6.1f}M  input {shape[0]}x{shape[1]}: {min(ts) * 1e3:8.0f} ms", flush=True)
