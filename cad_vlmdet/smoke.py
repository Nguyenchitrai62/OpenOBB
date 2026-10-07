"""Smoke test: forward + loss + infer on random tensor (CPU). MIT."""
import torch
from model import build_model
from loss import DetectionLoss

for variant in ["nano", "small"]:
    m = build_model(variant)
    print(variant, f"{m.param_count()/1e6:.2f}M params")
    x = torch.randn(1, 3, 256, 256)
    outs = m(x)
    for i, (c, b, a) in enumerate(outs):
        print(f"  P{i} stride={m.strides[i]} cls={tuple(c.shape)} box={tuple(b.shape)} ang={tuple(a.shape)}")
    tgt = [torch.tensor([[0.5, 0.5, 0.2, 0.1, 30.0, 0], [0.3, 0.3, 0.02, 0.02, 10.0, 6]])]
    loss, d = DetectionLoss()(outs, tgt, 256)
    print(f"  loss={float(loss):.4f}", d)
print("OK")
