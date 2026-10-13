"""Deployment checkpoint: `openobb strip model=best.pt` -> best_deploy.pt with only what inference needs.

A training checkpoint also carries the training run: dataset / cache paths, the run folder, ~100 training arguments,
validation scores, the epoch. A deploy checkpoint keeps the weights, the architecture fields that rebuild the network
(arch, size, image size and switches), the class names and the per-class confidence thresholds. Nothing else.
It loads with the same `OpenOBB(path)` / load_model, under any file name.
"""
import json
from pathlib import Path

import torch

# fields that rebuild the network or set inference behaviour (openobb/predict.py load_model, openobb/model.py)
DEPLOY_KEYS = ("arch", "size", "img", "scale", "fit", "num_top", "queries", "post", "lsk", "p2", "strip_k",
               "ortho_heads", "dense", "dense_queries", "denoising", "no_rotate_sampling", "context", "backbone",
               "dense_v3", "primary", "geo_cls", "relate", "rel_k", "vectors", "vec_dim", "vec_layers", "vec_lfe",
               "vec_ground", "v6_off", "v7_off", "v8_off", "v9_off", "v10_off")
LEAKS = ("openobb", "vrdet", "/content", "/root", "\\users\\", "runs/", "drive/")


def _state(ck):
    if "ema" in ck and isinstance(ck["ema"], dict) and "module" in ck["ema"]:
        return ck["ema"]["module"]
    return ck["model"]


def strip(model, out=None, half=False):
    """model: training checkpoint (.pt) -> path of the deploy checkpoint (default <name>_deploy.pt next to it).
    half=True stores the weights in float16 (half the size; loaded back as float32)."""
    src = Path(model)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    if "arch" not in a:
        raise SystemExit(f"{src} has no args['arch']: not an OpenOBB training checkpoint")
    sd = {k: (v.half() if half and v.is_floating_point() else v).clone() for k, v in _state(ck).items()}
    args = {k: a[k] for k in DEPLOY_KEYS if k in a and a[k] not in (None, "")}
    dep = {"model": sd, "args": args, "classes": list(ck.get("classes") or [])}
    if ck.get("conf_thr") is not None:
        dep["conf_thr"] = ck["conf_thr"]
    text = json.dumps({"args": args, "classes": dep["classes"]}, default=str).lower()
    found = [w for w in LEAKS if w in text]
    if found:
        raise SystemExit(f"deploy fields still contain {found}: {args}")
    out = Path(out) if out else src.with_name(f"{src.stem}_deploy.pt")
    torch.save(dep, out)
    mb = lambda p: p.stat().st_size / 2**20  # noqa: E731
    print(f"[strip] {src.name} ({mb(src):.0f} MB, {len(a)} args) -> {out} ({mb(out):.0f} MB, {len(args)} args: "
          f"{', '.join(args)})")
    return str(out)
