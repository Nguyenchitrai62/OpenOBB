"""Where does the inference time of a dense OpenOBB (v12-v22) go? GPU timing of each stage of the forward with CUDA
events, per image at a given batch, after warm-up (research tool; FLOPs alone do not explain the gap to YOLO:
OpenOBB22-x 523 GFLOPs vs YOLO26x 605, F92).

Stages: trunk + neck (encode), head (run_head), boundary field, candidate decode + top-k, edge refiner + boundary
snapping (refine_stages on the top ref_k candidates), NMS, rest of forward.

usage:  openobb profile model=best.pt [imgsz=1280] [batch=16] [runs=20] [detail=True] [channels_last=True] [compile=True]
Training step (v17-v24): openobb profile model=best.pt train=True [imgsz=960] [batch=32] [objects=150] [assign=legacy]
"""
import math
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


def _synthetic_targets(batch, objects, nc, img, device, seed=0):
    """Drawing-like labels: 70% thin long boxes (walls / pipes, axis-aligned), 20% small squares (junctions),
    10% medium squares (doors / symbols); normalised (cx, cy, w, h, theta)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(batch):
        n = objects
        kind = torch.rand(n, generator=g)
        L = torch.where(kind < 0.7, 20 + torch.rand(n, generator=g) * 380,
                        torch.where(kind < 0.9, 8 + torch.rand(n, generator=g) * 22, 40 + torch.rand(n, generator=g) * 80))
        T = torch.where(kind < 0.7, 3 + torch.rand(n, generator=g) * 9, L * (0.8 + 0.2 * torch.rand(n, generator=g)))
        th = torch.randint(0, 2, (n,), generator=g).float() * math.pi / 2
        cx, cy = torch.rand(n, generator=g) * img, torch.rand(n, generator=g) * img
        b = torch.stack([cx / img, cy / img, L / img, T / img, th], -1)
        out.append({"boxes": b.to(device), "labels": torch.randint(0, nc, (n,), generator=g).to(device)})
    return out


def profile_train(model_path, imgsz=960, batch=32, runs=10, warmup=3, objects=150, assign="batched", device=None,
                  channels_last=True):
    """Time of one training step of a dense OpenOBB (v17-v24) on random images with drawing-like labels, split into
    forward parts, loss parts (assignment, refiner, boundary field), backward and optimizer + EMA. Wall clock with
    a device sync around every part (a Python-bound part shows its real cost; GPU work cannot overlap across parts,
    so the parts add up to slightly more than an unsynchronised step). assign=legacy: the pre-F112 per-image loops.

    usage:  openobb profile model=best.pt train=True [imgsz=960] [batch=32] [objects=150] [runs=10] [assign=legacy]"""
    import collections

    from openobb.engine import ModelEMA
    from openobb.models import openobb10_loss as L10
    from openobb.models import openobb17_loss as L17
    from openobb.models.openobb18 import OpenOBB17, OpenOBB18, OpenOBB19, OpenOBB20, OpenOBB21, OpenOBB22, OpenOBB23
    from openobb.models.openobb18_loss import OpenOBB18Loss
    from openobb.models.openobb24 import OpenOBB24
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    a, names = ck.get("args", {}), ck.get("classes") or ck.get("names")
    arch = a.get("arch")
    classes = {"v17": OpenOBB17, "v18": OpenOBB18, "v19": OpenOBB19, "v20": OpenOBB20, "v21": OpenOBB21,
               "v22": OpenOBB22, "v23": OpenOBB23, "v24": OpenOBB24}
    if arch not in classes:
        raise SystemExit(f"profile train=True supports the dense models v17-v24, got arch {arch!r}")
    kw = dict(num_classes=len(names), img_size=imgsz, lsk=a.get("lsk", True), off=a.get("v7_off") or None,
              box=a.get("box") or "dfl", ref_k=int(a.get("ref_k", 1500)), groups=a.get("groups") or None)
    if arch == "v24":
        kw["off24"] = a.get("v24_off") or None
    model = classes[arch](a.get("size", "x"), **kw)
    sd = ck["ema"]["module"] if isinstance(ck.get("ema"), dict) and "module" in ck["ema"] else ck["model"]
    model.load_state_dict(sd, strict=False)
    model.to(dev).train()
    if channels_last and cuda:
        model.to(memory_format=torch.channels_last)
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    crit = OpenOBB18Loss(img_size=imgsz)
    try:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=cuda)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    ema = ModelEMA(model, 0.9999, 2000)
    x = torch.rand(batch, 3, imgsz, imgsz, device=dev)
    if channels_last and cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    targets = _synthetic_targets(batch, objects, len(names), imgsz, dev)
    acc = collections.defaultdict(float)
    on = [False]

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def timed(name, fn):
        def wrap(*args, **kwargs):
            if not on[0]:
                return fn(*args, **kwargs)
            sync()
            t = time.perf_counter()
            r = fn(*args, **kwargs)
            sync()
            acc[name] += time.perf_counter() - t
            return r
        return wrap

    patches = []

    def patch(obj, attr, name):
        if obj is not None and hasattr(obj, attr):
            patches.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, timed(name, getattr(obj, attr)))
    patch(model, "encode", "fwd  trunk + neck")
    patch(model, "run_head", "fwd  heads (o2m + o2o)")
    patch(getattr(model, "field", None), "forward", "fwd  boundary field")
    patch(model, "edge_snap" if hasattr(model, "edge_snap") else "refine_stages", "fwd  edge refiner + snapping")
    patch(model, "quality_logits", "fwd  quality head")
    patch(crit, "dense_losses", "loss dense o2m (incl. assignment)")
    patch(crit, "group_assign", "loss   assignment o2m")
    patch(crit.o2o, "dense_losses", "loss dense o2o (incl. assignment)")
    patch(crit.o2o, "group_assign", "loss   assignment o2o")
    patch(crit, "refine_loss", "loss refiner stages")
    patch(crit, "fit_loss", "loss quality")
    patch(L17, "field_targets", "loss field targets")
    patch(L17, "field_loss", "loss field")
    old_assign, old_win = L10.ASSIGN, L17.WINDOWED
    L10.ASSIGN, L17.WINDOWED = ("legacy", False) if assign == "legacy" else ("batched", True)
    amp_dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float16
    steps = collections.defaultdict(float)

    def step(timing=True):
        sync()
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=cuda):
            out = model(x, targets)
        if timing:
            sync()
        t1 = time.perf_counter()
        loss = sum(crit(out, targets).values())
        if timing:
            sync()
        t2 = time.perf_counter()
        loss.backward()
        if timing:
            sync()
        t3 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)
        sync()
        t4 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, t4 - t3

    try:
        for _ in range(warmup):
            step(False)
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        on[0] = True
        for _ in range(runs):
            for k, v in zip(("forward", "loss", "backward", "clip + AdamW + EMA"), step()):
                steps[k] += v
        on[0] = False
        t0 = time.perf_counter()                      # unsynchronised steps: the real step time
        for _ in range(runs):
            step(False)
        real = (time.perf_counter() - t0) / runs
    finally:
        for obj, attr, fn in reversed(patches):
            setattr(obj, attr, fn)
        L10.ASSIGN, L17.WINDOWED = old_assign, old_win
    gpu = torch.cuda.get_device_name(dev) if cuda else "CPU"
    print(f"{model_path} ({arch}, {len(names)} classes, groups {a.get('groups')}): train step, batch {batch} x 3 x "
          f"{imgsz} x {imgsz}, {objects} labels / image, {amp_dtype}, assign={assign}, {gpu}, {runs} steps after "
          f"{warmup} warm-up; ms per step")
    for k, v in steps.items():
        print(f"  {k:38s} {1000 * v / runs:8.1f}")
        for name, t in acc.items():
            if (name.startswith("fwd") and k == "forward") or (name.startswith("loss") and k == "loss"):
                print(f"    {name:36s} {1000 * t / runs:8.1f}")
    print(f"  {'step without syncs (real)':38s} {1000 * real:8.1f}  -> {batch / real:.1f} img/s")
    if cuda:
        print(f"  peak memory {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB")
    return dict(steps), dict(acc), real
