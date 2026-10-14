"""How noisy is Wall_Color val (53 images) as a ruler? Bootstrap the val images of one trained model's detections and
report the spread of mAP50 / mAP50-95 overall and per class (CPU, YOLO-style evaluator).

usage: python val_noise_bootstrap.py <dup_raw_<model>_53.npz> [nms_iou=0.7] [n_boot=100]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)
from openobb.eval.ultra import evaluate_ultra  # noqa: E402
from openobb.models.openobb3 import fast_nms  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402

NAMES = ["wall", "note", "door", "slide_door", "double_door", "wall_300", "junction"]
raw_path = sys.argv[1]
NMS_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 100

raw = dict(np.load(raw_path, allow_pickle=True))
gts, dets = {}, {}
for stem, rec in raw.items():
    rec = rec.item()
    gts[stem] = [(np.asarray(p, np.float64), n, 0) for p, n, _ in rec["gt"]]
    sc, cl, b = rec["raw"]
    dets[stem] = []
    if len(sc):
        logits = torch.zeros(1, len(sc), len(NAMES)) - 30
        logits[0, torch.arange(len(sc)), torch.as_tensor(cl, dtype=torch.long)] = torch.logit(
            torch.as_tensor(sc).clamp(1e-6, 1 - 1e-6))
        L, B = fast_nms(logits, torch.as_tensor(b)[None], max_det=3000, conf=0.05, iou=NMS_IOU)
        s2, c2 = L[0].sigmoid().max(-1)
        dets[stem] = [(int(c), float(p), obb2poly(bb[None].astype(np.float64))[0])
                      for c, p, bb in zip(c2.numpy(), s2.numpy(), B[0].numpy())]


def score(stems):
    g, d = {}, {c: ([], [], []) for c in NAMES}
    for j, st in enumerate(stems):
        key = f"{st}#{j}"                               # a resampled image counts as a separate image
        g[key] = gts[st]
        for c, p, pl in dets[st]:
            d[NAMES[c]][0].append(key), d[NAMES[c]][1].append(p), d[NAMES[c]][2].append(pl)
    d = {c: (i, np.asarray(s), np.asarray(p).reshape(-1, 8)) for c, (i, s, p) in d.items()}
    return evaluate_ultra(d, g, NAMES, max_det=3000)


stems = sorted(gts)
full = score(stems)
rng = np.random.default_rng(0)
boots = []
for k in range(NB):
    boots.append(score(list(rng.choice(stems, len(stems), replace=True))))
    print(f"{k + 1}/{NB}", end="\r", flush=True)
print(f"\n{Path(raw_path).name}: {len(stems)} val images, {NB} bootstrap resamples (std = 1-sigma ruler noise)")
row = lambda n, f, a, b: print(f"{n:12s} {f[a]:.3f} ± {np.std(b):.3f}")  # noqa: E731
for key in ("mAP50", "mAP50_95"):
    print(f"{'all ' + key:16s} {full[key]:.3f} ± {np.std([b[key] for b in boots]):.3f}")
for c in NAMES:
    a50 = np.std([b["classes"][c]["AP50"] for b in boots])
    a95 = np.std([b["classes"][c]["AP50_95"] for b in boots])
    print(f"{c:12s} AP50 {full['classes'][c]['AP50']:.3f} ± {a50:.3f} | AP50-95 {full['classes'][c]['AP50_95']:.3f} ± {a95:.3f}")
