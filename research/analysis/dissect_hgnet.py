"""Dissect the pretrained HGNetv2 trunk of a trained OpenOBB checkpoint on real drawings: what each inner mechanism
does to the final detections (F58). Everything is measured by intervention on the trained network, on the CPU.

  ese      gate statistics of every ESE module (channel attention from the global mean): mean gate, spread across
           images, saturated channels; then the gates are replaced by their mean over the images ("fold": no global
           pooling left) -> change of the detections. A constant gate is a per-channel scale that folds into the
           aggregation conv: ESE can be removed exactly.
  blocks   each residual HG_Block in turn has its branch zeroed (output = its input) -> change of the detections,
           with the branch / identity RMS ratio. A block whose removal changes nothing is dead weight.
  dw       the depthwise 5x5 kernels of the LightConv layers (stages 3-4) are cut to their centre 3x3 and to their
           centre tap, one stage at a time -> change of the detections: does the 5x5 spatial extent matter?
  dead     channels of every block output whose activation is (nearly) constant over positions and images.
  stage4   stage 4 output zeroed (F46 check on this checkpoint).

usage: OMP_NUM_THREADS=2 py -3.12 research/analysis/dissect_hgnet.py F:/Downloads/openobb12x.pt --images 3 --conf 0.25
"""
import argparse
import contextlib
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from openobb.models.hgnetv2 import EseModule, HG_Block, LightConvBNAct  # noqa: E402
from openobb.ops.obb_torch import probiou  # noqa: E402
from openobb.predict import load_model, rect_shape  # noqa: E402

VAL = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "dataset_wall_color", "images", "val")


def load_images(n, size):
    out = []
    for p in sorted(glob.glob(os.path.join(VAL, "*.png")))[:n]:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        r = size / max(h, w)
        img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
        th, tw = rect_shape(img.shape[0], img.shape[1], size)
        canvas = np.full((th, tw, 3), 255, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        x = torch.from_numpy(canvas[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
        out.append((os.path.basename(p), x))
    return out


@torch.no_grad()
def detect(m, x, conf, size):
    r = m(x)
    s = r["pred_logits"][0].sigmoid()
    sc, lab = s.max(-1)
    keep = sc >= conf
    b = r["pred_boxes"][0][keep].clone()
    b[:, :4] *= size
    return b, lab[keep], sc[keep]


def compare(base, other):
    """-> (fraction of baseline boxes still found: same class, ProbIoU >= 0.5; count baseline; count other)."""
    b, lb, _ = base
    o, lo, _ = other
    if len(b) == 0:
        return 1.0, 0, len(o)
    if len(o) == 0:
        return 0.0, len(b), 0
    ov = probiou(b[:, None, :], o[None, :, :]) * (lb[:, None] == lo[None, :])
    return float((ov.amax(1) >= 0.5).float().mean()), len(b), len(o)


@contextlib.contextmanager
def patched(mod, fn):
    old = mod.forward
    mod.forward = fn
    try:
        yield
    finally:
        mod.forward = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dissect_hgnet.json"))
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    m, names, args = load_model(a.ckpt, "cpu")
    size = int(args.get("img", 1280))
    m.nms_conf = a.conf
    bb = m.backbone
    images = load_images(a.images, size)
    print(f"{a.ckpt}: arch {args.get('arch')} size {args.get('size')} img {size}; head {args.get('infer_head')} "
          f"nms {args.get('nms_iou')}; {len(images)} val images")
    res = {"ckpt": a.ckpt, "images": [n for n, _ in images], "conf": a.conf}

    # ---- baseline + ESE gates + block branch ratios + dead channels in one pass
    gates = {}          # ese name -> list of (C,) gates per image
    outs = {}           # block name -> list of (C,) per-channel std over positions per image, and mean
    ratios = {}         # block name -> list of branch / identity RMS
    ese_mods = {n: mod for n, mod in bb.named_modules() if isinstance(mod, EseModule)}
    blocks = {n: mod for n, mod in bb.named_modules() if isinstance(mod, HG_Block)}
    hs = []
    for n, mod in ese_mods.items():
        def h(mod, inp, out, n=n):
            g = (out / inp[0].clamp(min=1e-6)).mean((2, 3))[0]        # the gate = out / in (per channel)
            x = inp[0]
            g = torch.sigmoid(mod.conv(x.mean((2, 3), keepdim=True)))[0, :, 0, 0]
            gates.setdefault(n, []).append(g.float())
        hs.append(mod.register_forward_hook(h))
    for n, mod in blocks.items():
        def h(mod, inp, out, n=n):
            o = out[0].float()
            outs.setdefault(n, []).append((o.std((1, 2)), o.abs().mean()))
            if mod.residual:
                branch = (out - inp[0])[0].float()
                ratios.setdefault(n, []).append(float(branch.pow(2).mean().sqrt() / inp[0][0].float().pow(2).mean().sqrt()))
        hs.append(mod.register_forward_hook(h))
    base = []
    t0 = time.perf_counter()
    for name, x in images:
        base.append(detect(m, x, a.conf, size))
    print(f"baseline: {[len(b[0]) for b in base]} boxes; {time.perf_counter() - t0:.1f} s")
    for h in hs:
        h.remove()

    ese_stats = {}
    for n, gs in gates.items():
        G = torch.stack(gs)                                   # (N, C)
        ese_stats[n] = {"channels": G.shape[1], "mean_gate": float(G.mean()),
                        "spread_across_images": float(G.std(0).mean()) if G.shape[0] > 1 else 0.0,
                        "frac_gate_below_0.1": float((G.mean(0) < 0.1).float().mean()),
                        "frac_gate_above_0.9": float((G.mean(0) > 0.9).float().mean())}
    res["ese"] = ese_stats
    print("\nESE gates (channel attention from the global mean):")
    print(f"{'module':34s} {'C':>5s} {'mean':>6s} {'spread/img':>10s} {'<0.1':>6s} {'>0.9':>6s}")
    for n, s in ese_stats.items():
        print(f"{n:34s} {s['channels']:5d} {s['mean_gate']:6.3f} {s['spread_across_images']:10.4f} "
              f"{s['frac_gate_below_0.1']:6.2f} {s['frac_gate_above_0.9']:6.2f}")

    dead = {}
    for n, lst in outs.items():
        std = torch.stack([s for s, _ in lst]).mean(0)
        scale = torch.stack([v for _, v in lst]).mean()
        dead[n] = {"channels": len(std), "frac_dead": float((std < 1e-3 * scale).float().mean()),
                   "frac_weak": float((std < 1e-2 * scale).float().mean()),
                   "branch_over_identity_rms": float(np.mean(ratios[n])) if n in ratios else None}
    res["blocks_stats"] = dead
    print("\nHG_Block outputs: dead channels (std < 1e-3 x mean |x|), weak (< 1e-2), residual branch / identity RMS:")
    for n, s in dead.items():
        r = s["branch_over_identity_rms"]
        print(f"{n:28s} C={s['channels']:5d} dead {s['frac_dead']:5.2f} weak {s['frac_weak']:5.2f} "
              f"branch/id {'-' if r is None else f'{r:.3f}'}")

    def run_variant(label):
        fr, nb, no = [], 0, 0
        for (name, x), b0 in zip(images, base):
            f, n0, n1 = compare(b0, detect(m, x, a.conf, size))
            fr.append(f)
            nb += n0
            no += n1
        row = {"kept": float(np.mean(fr)), "boxes_base": nb, "boxes": no}
        print(f"{label:44s} kept {row['kept']:5.1%}  boxes {nb} -> {no}")
        return row

    # ---- fold ESE: constant (mean) gates instead of image-dependent ones
    print("\nInterventions (fraction of baseline detections still found, same class, ProbIoU >= 0.5):")
    variants = {}
    means = {n: torch.stack(gs).mean(0)[None, :, None, None] for n, gs in gates.items()}
    with contextlib.ExitStack() as st:
        for n, mod in ese_mods.items():
            st.enter_context(patched(mod, lambda x, g=means[n]: x * g.to(x.dtype)))
        variants["ese_folded_all"] = run_variant("ESE -> constant mean gate (all stages)")
    with contextlib.ExitStack() as st:
        for n, mod in ese_mods.items():
            st.enter_context(patched(mod, lambda x: x))
        variants["ese_removed_all"] = run_variant("ESE removed (gate = 1)")

    # ---- each residual block zeroed
    for n, mod in blocks.items():
        if not mod.residual:
            continue
        with patched(mod, lambda x: x):
            variants[f"zero_{n}"] = run_variant(f"branch of {n} zeroed")

    # ---- depthwise 5x5 of the LightConv layers cut to 3x3 / 1x1
    lights = {n: mod for n, mod in bb.named_modules() if isinstance(mod, LightConvBNAct)}
    stages = sorted({n.split(".")[1] for n in lights})
    for st_i in stages:
        convs = [mod.conv2.conv for n, mod in lights.items() if n.split(".")[1] == st_i]
        convs = [c[1] if isinstance(c, torch.nn.Sequential) else c for c in convs]
        saved = [c.weight.detach().clone() for c in convs]
        for cut, k in (("3x3", 1), ("1x1", 2)):
            with torch.no_grad():
                for c, w in zip(convs, saved):
                    mask = torch.zeros_like(w)
                    mask[..., k:w.shape[-2] - k, k:w.shape[-1] - k] = 1
                    c.weight.copy_(w * mask)
            variants[f"dw_stage{st_i}_{cut}"] = run_variant(f"LightConv dw 5x5 -> centre {cut}, stage index {st_i}")
        with torch.no_grad():
            for c, w in zip(convs, saved):
                c.weight.copy_(w)

    # ---- stage 4 zeroed
    with patched(bb.stages[3], lambda x: torch.zeros_like(bb.stages[3].__class__.forward(bb.stages[3], x))):
        variants["stage4_zero"] = run_variant("stage 4 output = 0")
    res["variants"] = variants
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
