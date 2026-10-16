"""Where does the inference time of a dense OpenOBB (v12-v22) go? GPU timing of each stage of the forward with CUDA
events, per image at a given batch, after warm-up (research tool; FLOPs alone do not explain the gap to YOLO:
OpenOBB22-x 523 GFLOPs vs YOLO26x 605, F92).

Stages: trunk + neck (encode), head (run_head), boundary field, candidate decode + top-k, edge refiner + boundary
snapping (refine_stages on the top ref_k candidates), NMS, rest of forward.

usage:  openobb profile model=best.pt [imgsz=1280] [batch=16] [runs=20] [detail=True] [channels_last=True] [compile=True]
"""
import time

import torch


def profile_parts(model_path, imgsz=1280, batch=16, runs=20, warmup=5, h_ratio=0.75, device=None, detail=False,
                  channels_last=False, compile=False):
    from openobb.predict import load_model
    from openobb.models import openobb3, openobb17
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"
    net = load_model(model_path, dev)[0].eval()
    H = int(round(imgsz * h_ratio / 32) * 32)
    x = torch.rand(batch, 3, H, imgsz, device=dev)
    if cuda:
        torch.backends.cudnn.benchmark = True        # fixed input shape: let cuDNN pick its fastest kernels
    if channels_last:                                # NHWC: what tensor cores read natively (no NCHW <-> NHWC per conv)
        net = net.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    if compile:                                      # fuse the elementwise ops of the static-shape parts (trunk, neck, heads)
        for name in ("backbone", "neck", "encoder", "field"):
            if getattr(net, name, None) is not None:
                setattr(net, name, torch.compile(getattr(net, name)))
        # the heads stay eager: OpenOBB21+ index their towers layer by layer (run_head)
        detail = False                               # compiled modules cannot be timed one by one
    times = {}
    events = []

    def timed(name, fn):
        def wrap(*a, **k):
            if not cuda:                                 # CPU: synchronous, wall clock
                t = time.perf_counter()
                r = fn(*a, **k)
                events.append((name, (time.perf_counter() - t) * 1000, None))
                return r
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            r = fn(*a, **k)
            e.record()
            events.append((name, s, e))
            return r
        return wrap

    patches = []
    for obj, attr, name in ((net, "encode", "trunk + neck"), (net, "run_head", "head"), (net, "candidates", "decode + groups"),
                            (net, "snap_topk", "top-k + refiner + snapping")):
        if hasattr(obj, attr):
            patches.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, timed(name, getattr(obj, attr)))
    if getattr(net, "field", None) is not None:
        f = net.field.forward
        net.field.forward = timed("boundary field", f)
    if detail:                                       # inside trunk + neck: every stage / neck block
        subs = []
        bb = getattr(net, "backbone", None)
        if bb is not None:
            subs += [("  trunk stem", bb.stem)] if hasattr(bb, "stem") else []
            subs += [(f"  trunk stage {i + 1}", s) for i, s in enumerate(getattr(bb, "stages", []))]
        for owner in ("lsk", "encoder", "neck"):
            mod = getattr(net, owner, None)
            if mod is not None:
                subs += [(f"  {owner}.{n}", c) for n, c in mod.named_children()]
        for name, mod in subs:
            f = mod.forward
            mod.forward = timed(name, f)
            patches.append((mod, "forward", f))
    nms_mods = [m for m in (openobb3, openobb17) if hasattr(m, "fast_nms")]
    saved = [(m, m.fast_nms) for m in nms_mods]
    for m, fn in saved:
        m.fast_nms = timed("NMS", fn)
    try:
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
            for _ in range(warmup):
                net(x)
            if cuda:
                torch.cuda.synchronize()
            events.clear()
            t0 = time.perf_counter()
            for _ in range(runs):
                net(x)
            if cuda:
                torch.cuda.synchronize()
            total = (time.perf_counter() - t0) * 1000 / runs / batch
        for name, s, e in events:
            ms = s.elapsed_time(e) if e is not None else s
            times[name] = times.get(name, 0.0) + ms / runs / batch
    finally:
        for m, fn in saved:
            m.fast_nms = fn
        for obj, attr, fn in patches:
            setattr(obj, attr, fn)
    nested = times.get("top-k + refiner + snapping", 0.0)
    mode = ", ".join(n for n, on in (("channels_last", channels_last), ("compile", compile)) if on) or "default (channels_last on a GPU since 0.22.2)"
    print(f"{model_path}: input {batch} x 3 x {H} x {imgsz}, bf16, {mode}, per image (ms), {runs} runs after {warmup} warm-up")
    for k, v in times.items():
        print(f"  {k:30s} {v:7.2f}")
    print(f"  {'TOTAL forward (wall)':30s} {total:7.2f}")
    print(f"  (NMS runs after the refiner; 'rest' = total - listed: {total - sum(times.values()):.2f}; refiner block {nested:.2f})")
    return times, total
