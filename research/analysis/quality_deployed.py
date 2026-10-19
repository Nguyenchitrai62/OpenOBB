"""OpenOBB24 through its DEPLOYED forward (load_model: fused, the calibrated NMS / head / snap_skip), CPU, Wall_Color val:
mAP when the learnt localisation quality q is used in the NMS (q_beta) with the two fills for candidates outside the
refined top ref_k: "prior" (constant 0.6, 0.25.1) and "min" (lowest refined q of the image). quality_probe.py only saw the
refined candidates and measured +0.020 for q^2 in NMS, while the end-of-training calibration found every q setting
below q^0 (F110): this tells which part of the deployed path differs.

usage: python quality_deployed.py <openobb24 .pt> [n_images]
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
from openobb.data.imgio import imread  # noqa: E402
from openobb.eval.ultra import evaluate_ultra  # noqa: E402
from openobb.ops.obb import obb2poly  # noqa: E402
from openobb.predict import PAD_BGR, load_model, rect_shape  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
ck, n_img = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
net, names, args = load_model(ck, "cpu")
print(f"{Path(ck).name}: nms {args.get('nms_iou')} head {args.get('infer_head')} q_beta {args.get('q_beta')} "
      f"snap_skip [{args.get('snap_skip')}]", flush=True)
SETTINGS = ([("q^0", 0.0, "min"), ("q^2 min", 2.0, "min")] if os.environ.get("QD_SHORT") == "1" else
            [("q^0 drop", 0.0, "drop"), ("q^2 drop", 2.0, "drop")] if os.environ.get("QD_SHORT") == "drop" else
            [("q^0", 0.0, "min"), ("q^1 prior", 1.0, "prior"), ("q^2 prior", 2.0, "prior"),
             ("q^1 min", 1.0, "min"), ("q^2 min", 2.0, "min"), ("q^4 min", 4.0, "min")])
TOP1 = bool(os.environ.get("QD_TOP1"))      # also score "one class per box": rows of the same box keep their best class
imgs = sorted((ROOT / "images" / "val").iterdir())[:n_img]
dets = {n: {c: ([], [], []) for c in names} for s in SETTINGS for n in ((s[0], s[0] + " top1") if TOP1 else (s[0],))}
gts = {}
t0 = time.time()
for k, p in enumerate(imgs):
    img = imread(str(p))
    h0, w0 = img.shape[:2]
    s = 1280 / max(h0, w0)
    img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = rect_shape(*img.shape[:2], 1280)
    x = np.full((H, W, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    g = []
    for line in (ROOT / "labels" / "val" / f"{p.stem}.txt").read_text().splitlines():
        v = line.split()
        if len(v) >= 9:
            g.append((np.array(v[1:9], float) * np.tile([w0 * s, h0 * s], 4), names[int(v[0])], 0))
    gts[p.stem] = g
    for name, beta, fill in SETTINGS:
        net.q_beta, net.q_alpha, net.q_outside = beta, 0.0, fill
        with torch.inference_mode():
            out = net(xt)
        sc, cl = out["pred_logits"][0].float().sigmoid().max(-1)
        b = out["pred_boxes"][0].float().clone()
        b[:, :4] *= net.img_size
        keep = sc > 0.001
        rows = list(zip(cl[keep].numpy(), sc[keep].numpy(), b[keep].numpy().astype(np.float64)))
        for c, sv, bb in rows:
            ids, scs, pls = dets[name][names[c]]
            ids.append(p.stem), scs.append(float(sv)), pls.append(obb2poly(bb[None])[0])
        if TOP1:
            best = {}
            for c, sv, bb in rows:                     # same box (to 0.01 px) -> keep the highest-scoring class
                key = tuple(np.round(bb, 2))
                if key not in best or sv > best[key][1]:
                    best[key] = (c, sv, bb)
            for c, sv, bb in best.values():
                ids, scs, pls = dets[name + " top1"][names[c]]
                ids.append(p.stem), scs.append(float(sv)), pls.append(obb2poly(bb[None])[0])
    print(f"{k + 1}/{len(imgs)} {time.time() - t0:.0f}s", end="\r", flush=True)

res = {n: evaluate_ultra({c: (i, np.asarray(s_), np.asarray(p_).reshape(-1, 8)) for c, (i, s_, p_) in d.items()},
                         gts, names, max_det=3000) for n, d in dets.items()}
print(f"\n{'setting':12s} {'mAP50':>7s} {'mAP50-95':>9s} {'fitness':>8s} " + " ".join(f"{c[:9]:>9s}" for c in names))
for n, r in res.items():
    f = 0.1 * r["mAP50"] + 0.9 * r["mAP50_95"]
    print(f"{n:12s} {r['mAP50']:7.4f} {r['mAP50_95']:9.4f} {f:8.4f} "
          + " ".join(f"{r['classes'].get(c, {}).get('AP50_95', float('nan')):9.3f}" for c in names))
