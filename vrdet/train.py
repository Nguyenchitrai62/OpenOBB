"""VRDet trainer (single GPU). Resumes from {out}/last.pt, appends one JSON line per epoch to
{out}/metrics.jsonl, ends with a full DOTA-protocol evaluation of the EMA weights on val.

python -u -m vrdet.train --data /content/datasets/dota1_1024 --out OUT --size s --epochs 24 --batch 32
"""
import argparse
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vrdet.data.dota import DotaPatches, collate, dataset_classes
from vrdet.engine import ModelEMA, eval_dota, lr_factor, param_groups, to_device
from vrdet.console import EpochBar, val_rows
from vrdet.report import append_results, plot_labels, plot_results, plot_train_batch, plot_val_predictions
from vrdet.eval.dota import DOTA1_CLASSES, format_table, summary_table, write_task1
from vrdet.models.dense_head import DenseCriterion
from vrdet.models.vrdet import VRDet, build_criterion, load_dfine_coco


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False)          # a typo must fail, not match another flag
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--img", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="image scale the data was tiled at (recorded for inference; set by vrdet.data.prepare)")
    ap.add_argument("--fit", action="store_true",
                    help="data was prepared with each image resized to long side = img (one tile per image); "
                         "recorded so inference does the same")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-mult", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--flat", type=float, default=0.5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1, help="final LR / initial LR (lrf)")
    ap.add_argument("--schedule", default="flatcos", choices=["flatcos", "linear", "cos"],
                    help="flatcos: flat then cosine (D-FINE); linear / cos: decay over the whole run to min-lr-ratio")
    ap.add_argument("--cache", action="store_true", help="decode training images once into RAM")
    ap.add_argument("--fliplr", type=float, default=0.5, help="horizontal flip probability")
    ap.add_argument("--flipud", type=float, default=0.5, help="vertical flip probability")
    ap.add_argument("--no-rot90", action="store_true", help="no random 90-degree rotations")
    ap.add_argument("--freeze", default="", help="fine-tuning: N (stem + first N backbone stages), 'backbone', or "
                                                 "'encoder' (backbone + adapters + encoder: train the decoder only)")
    ap.add_argument("--time", type=float, default=0.0,
                    help="training budget in hours: epochs are re-planned after each epoch so the LR still decays")
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.9998)
    ap.add_argument("--ema-warmups", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--num-top", type=int, default=300)
    ap.add_argument("--weights", default=None,
                    help="fine-tune: initialise from a VRDet checkpoint (last.pt); class heads are re-used when the "
                         "class list matches, otherwise re-initialised")
    ap.add_argument("--init", default="coco", choices=["coco", "obj2coco"],
                    help="D-FINE init: COCO-only (commercially clean) or Objects365->COCO (benchmark parity with YOLO26)")
    ap.add_argument("--post", default="flat", choices=["flat", "argmax"], help="eval post-processing")
    ap.add_argument("--o2m-queries", type=int, default=0,
                    help="H10: training-only one-to-many query group (e.g. 1500); dropped at inference")
    ap.add_argument("--o2m-k", type=int, default=6, help="H10: queries assigned per target in the o2m group")
    ap.add_argument("--aqd", action="store_true", help="H12: adaptive query denoising (RHINO idea)")
    ap.add_argument("--lsk", action="store_true", help="H14: selective large-kernel adapters on backbone maps")
    ap.add_argument("--p2", action="store_true", help="H18: stride-4 decoder level (thin lines, small symbols)")
    ap.add_argument("--cost-iou", type=float, default=0.0, help="H15: IoU exponent in the matching class cost")
    ap.add_argument("--eval-layer", type=int, default=-1, help="eval-only diagnostic: use decoder layer k's output")
    ap.add_argument("--angle-weight", type=float, default=0.0, help="H13: square-aware angle loss on the decoder")
    ap.add_argument("--eval-queries", type=int, default=0,
                    help="H9: also evaluate the final EMA model with N queries / top-N (queries have no parameters)")
    ap.add_argument("--denoising", type=int, default=100)
    ap.add_argument("--no-rotate-sampling", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--eval-every", type=int, default=6)
    ap.add_argument("--eval-images", type=int, default=100, help="val images for the periodic (cheap) eval")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None, help="debug: stop each epoch early")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--hsv", type=float, nargs=3, default=[0.015, 0.5, 0.3])
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--dense", action="store_true", help="add the dense one-to-many rotated head (H4)")
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--dense-queries", action="store_true", help="H4c: decoder queries from the dense head")
    ap.add_argument("--box-loss", default="kld", choices=["kld", "probiou"], help="H1")
    ap.add_argument("--ortho-heads", action="store_true", help="H2: half the heads sample at theta+90")
    ap.add_argument("--strip-k", type=int, default=0, help="H3: strip-context kernel (0 = off)")
    ap.add_argument("--vectors", action="store_true", help="H7: CAD primitive tokens ({data}/vectors) fused in")
    ap.add_argument("--vec-dim", type=int, default=128)
    ap.add_argument("--vec-layers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--vec-lfe", action="store_true", help="H16: per-CAD-layer pooling in the vector branch")
    ap.add_argument("--vec-ground", type=float, default=0.0,
                    help="H17: weight of the query->primitive membership loss (vector-grounded boxes); 0 = off")
    ap.add_argument("--layer-drop", type=float, default=0.0, help="H16: chance to merge all layers of a sample")
    ap.add_argument("--rotate-p", type=float, default=0.0, help="H8: arbitrary-angle rotation prob")
    ap.add_argument("--mosaic-p", type=float, default=0.0, help="H8: oriented mosaic prob")
    ap.add_argument("--mosaic-mode", default="half", choices=["half", "yolo"],
                    help="half: 4 half-scale patches; yolo: 4 full-res patches around a random centre + warp")
    ap.add_argument("--scale-jitter", type=float, default=0.0, help="H11: random scale 1 +- x on every sample")
    ap.add_argument("--translate", type=float, default=0.0, help="H11: random translation +- x * size")
    ap.add_argument("--mosaic-off", type=int, default=4, help="H8: no mosaic in the last N epochs (YOLO/DEIM practice)")
    ap.add_argument("--context", action="store_true", help="H6: whole-image context tokens for each tile")
    ap.add_argument("--profile", type=int, default=0, help="profile N iterations after 10 warm-up ones, then exit")
    ap.add_argument("--channels-last", action="store_true", help="NHWC convs/BN (profiling: BN was ~42% of GPU time)")
    ap.add_argument("--compile", action="store_true", help="torch.compile backbone + encoder (falls back to eager)")
    ap.add_argument("--rfs", type=float, default=0.0, help="repeat-factor sampling threshold t (0 = off)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate EMA weights of {out}/last.pt, no training")
    ap.add_argument("--ctx-dropout", type=float, default=0.0, help="H6: drop the context with this prob")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--aug-iof", type=float, default=0.7,
                    help="visible fraction a crop-cut object needs to keep its label after mosaic / zoom")
    ap.add_argument("--merge-iou", type=float, default=0.1,
                    help="class-wise polygon NMS IoU when merging detections (0.1 = DOTA tile merge)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when val mAP50-95 has not improved for N epochs (needs full-val each epoch; 0 = off)")
    ap.add_argument("--verbose", action="store_true",
                    help="print the detailed per-iteration log instead of the compact progress table")
    return ap


def get_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_done").exists() and "--eval-only" not in (argv or __import__("sys").argv):
        print("train_done exists: nothing to do", flush=True)
        return
    logf = open(out / "train_progress.log", "a")

    def log(msg, console=False):
        """Detailed log -> train_progress.log; also to the console with --verbose (or console=True)."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if a.verbose or console:
            print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def say(msg):
        """Compact console output (also kept in the log file)."""
        if not a.verbose:
            print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def jlog(rec):
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.threads:
        torch.set_num_threads(a.threads)   # CPU debug runs: torch 2.12 CPU kernels race with many threads
    amp = (not a.no_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16
    scaler = None
    if amp and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16       # T4 / V100: fp16 autocast with loss scaling
        scaler = torch.amp.GradScaler("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    classes = dataset_classes(a.data)
    model = VRDet(a.size, num_classes=len(classes), num_queries=a.queries, img_size=a.img,
                  rotate_sampling=not a.no_rotate_sampling, num_denoising=a.denoising, dense=a.dense,
                  strip_k=a.strip_k, ortho_heads=a.ortho_heads, context=a.context,
                  dense_queries=a.dense_queries, vectors=a.vectors, vec_dim=a.vec_dim, vec_layers=a.vec_layers,
                  o2m_queries=a.o2m_queries, lsk=a.lsk, vec_lfe=a.vec_lfe, vec_ground=a.vec_ground > 0, p2=a.p2)
    last = out / "last.pt"
    if not last.exists() and a.weights:
        # shape-tolerant copy of a VRDet checkpoint; class rows are matched by name (new classes start fresh)
        src_cls = list(torch.load(a.weights, map_location="cpu", weights_only=False).get("classes") or [])
        cmap = {i: src_cls.index(c) for i, c in enumerate(classes) if c in src_cls}
        load_dfine_coco(model, a.weights, log=lambda m: log(m, console=True), class_map=cmap)
        log(f"[init] fine-tuning from {a.weights} ({len(cmap)}/{len(classes)} classes carried over)")
        init_desc = f"fine-tune from {a.weights} ({len(cmap)}/{len(classes)} classes carried over)"
    elif not last.exists() and not a.no_pretrained:
        load_dfine_coco(model, a.size, class_names=classes, log=lambda m: log(m, console=True), init=a.init)
        init_desc = "D-FINE COCO weights (Apache-2.0)"
    else:
        init_desc = "resume" if last.exists() else "random"
    model.to(device)
    if a.channels_last:
        model.to(memory_format=torch.channels_last)
    crit = build_criterion(num_classes=len(classes), box_loss=a.box_loss, o2m_k=a.o2m_k, aqd=a.aqd,
                           angle_weight=a.angle_weight, cost_iou=a.cost_iou, member_weight=a.vec_ground)
    a.dense = a.dense or a.dense_queries
    dense_crit = DenseCriterion() if a.dense else None
    ds = DotaPatches(a.data, "train", size=a.img, augment=True, hsv=tuple(a.hsv), limit=a.limit_train,
                     rotate_p=a.rotate_p, mosaic_p=a.mosaic_p, context=a.context, ctx_dropout=a.ctx_dropout,
                     vectors=a.vectors, max_tokens=a.max_tokens, scale_jitter=a.scale_jitter,
                     translate=a.translate, mosaic_mode=a.mosaic_mode, layer_drop=a.layer_drop, aug_iof=a.aug_iof,
                     cache=a.cache, fliplr=a.fliplr, flipud=a.flipud, rot90=not a.no_rot90)
    frozen = []
    if a.freeze:                                # fine-tuning on small data: keep the pretrained early layers fixed
        f = str(a.freeze).lower()
        if f.isdigit():
            frozen = [model.backbone.stem] + list(model.backbone.stages[:int(f)])
        elif f == "backbone":
            frozen = [model.backbone]
        elif f == "encoder":
            frozen = [model.backbone, model.encoder] + [m for m in (model.lsk, getattr(model, "p2_proj", None))
                                                        if m is not None]
        else:
            raise SystemExit(f"--freeze must be an integer, 'backbone' or 'encoder', got {a.freeze!r}")
        for m in frozen:
            for prm in m.parameters():
                prm.requires_grad_(False)
        n_fr = sum(prm.numel() for m in frozen for prm in m.parameters()) / 1e6
        if not any(prm.requires_grad for prm in model.parameters()):
            raise SystemExit("--freeze left nothing to train")
        log(f"freeze={a.freeze}: {n_fr:.2f}M parameters frozen", console=True)
    if a.batch > len(ds):
        log(f"batch {a.batch} > {len(ds)} training samples: using batch {len(ds)}", console=True)
        a.batch = len(ds)
    sampler = None
    if a.rfs > 0:
        from vrdet.data.dota import repeat_factors
        w, f, r = repeat_factors(ds.items, len(classes), a.rfs)
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)
        log("RFS repeat factors: " + " ".join(f"{c[:6]}={x:.1f}" for c, x in zip(classes, r)))
    def make_loader():
        return DataLoader(ds, batch_size=a.batch, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                          collate_fn=collate,
                          pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
                          prefetch_factor=4 if a.workers > 0 else None)
    dl = make_loader()
    iters_per_epoch = len(dl) if a.max_iters is None else min(len(dl), a.max_iters)
    total_iters = iters_per_epoch * a.epochs
    try:                                # fused AdamW: one kernel per step instead of one per tensor
        opt = torch.optim.AdamW(param_groups(model, a.lr, a.backbone_mult, a.wd), lr=a.lr, betas=(0.9, 0.999),
                                fused=device.type == "cuda")
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(param_groups(model, a.lr, a.backbone_mult, a.wd), lr=a.lr, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = ModelEMA(model, a.ema, a.ema_warmups)
    start_epoch, it, best_score, best_epoch = 0, 0, -1.0, -1
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if "model" in ck:                       # eval-only checkpoints may carry the EMA weights alone
            model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch, it = ck["epoch"] + 1, ck["iter"]
        best_score, best_epoch = ck.get("best_score", -1.0), ck.get("best_epoch", -1)
        log(f"resumed from epoch {ck['epoch']} (iter {it})")
        if not a.eval_only:
            say(f"Resuming {out} from epoch {ck['epoch'] + 1}/{a.epochs}")
    # Compile AFTER the EMA deep copy: compiled forwards are bound methods of the live model, and a copy made
    # afterwards kept calling the live (training-mode) backbone/encoder at eval time (bug seen 2026-10-07:
    # identical train losses, ~7 points lower EMA eval).
    if a.compile and device.type == "cuda" and not a.eval_only:
        try:
            model.backbone.forward = torch.compile(model.backbone.forward)
            model.encoder.forward = torch.compile(model.encoder.forward, dynamic=False)
            log("torch.compile enabled for backbone + encoder (training model only)")
        except Exception as e:  # noqa: BLE001
            log(f"torch.compile unavailable, eager mode: {e}", console=True)
    if a.eval_only:
        start_epoch = a.epochs                  # skip training: evaluate the EMA weights of last.pt
        if a.eval_layer >= 0:                   # layer-wise diagnostic (SQR indicator): stop the decoder at layer k
            ema.module.decoder.decoder.eval_idx = a.eval_layer
            log(f"eval-only on decoder layer {a.eval_layer}")
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"VRDet-{a.size} {n_par:.2f}M params | train patches {len(ds)} | {iters_per_epoch} it/epoch x {a.epochs} "
        f"| batch {a.batch} | device {device} amp={amp}")
    val_subset, full_val = None, False
    if a.eval_images:
        gts = sorted(p.stem for p in (Path(a.data) / "gt" / "val").glob("*.txt"))
        val_subset = sorted(random.Random(0).sample(gts, min(a.eval_images, len(gts))))
        full_val = len(val_subset) == len(gts)      # periodic scores comparable with the final one
    if not a.eval_only and start_epoch < a.epochs:
        gpu = (f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)"
               if device.type == "cuda" else "CPU")
        extras = [n for n, on in (("LSK", a.lsk), ("P2", a.p2), ("dense", a.dense)) if on]
        n_val = len(list((Path(a.data) / "gt" / "val").glob("*.txt")))
        say(f"VRDet {__import__('vrdet').__version__} | torch {torch.__version__} | {gpu} | "
            f"AMP {('bf16' if scaler is None else 'fp16') if amp else 'off'} | "
            f"compile {'on' if a.compile and device.type == 'cuda' else 'off'}")
        say(f"model   VRDet-{a.size}{' + ' + ' + '.join(extras) if extras else ''}, {n_par:.2f}M params, "
            f"{a.queries} queries | init: {init_desc}")
        mode = f"whole image, long side {a.img}" if a.fit else f"tiles {a.img}, scale {a.scale}"
        say(f"data    {len(ds)} train tiles, {n_val} val images, {len(classes)} classes | {mode}")
        say(f"train   {a.epochs} epochs x {iters_per_epoch} it, batch {a.batch}, AdamW lr {a.lr:g} -> "
            f"{a.lr * a.min_lr_ratio:g} ({a.schedule}), warmup {a.warmup} it | "
            f"val every {a.eval_every} epoch(s) on {'all' if full_val else len(val_subset or [])} val images")
        say(f"save    {out}  (best.pt, last.pt, results.csv/png, labels.jpg, val_pred.jpg, train_progress.log)")
        if start_epoch == 0:
            plot_labels(out, ds.items, classes, a.img)

    def save_best(score, epoch):
        # compact deployable checkpoint: EMA weights + args + classes (no optimiser state)
        torch.save({"model": ema.module.state_dict(), "epoch": epoch, "args": vars(a), "classes": list(classes),
                    "val_mAP50_95": score}, out / "best.tmp")
        os.replace(out / "best.tmp", out / "best.pt")
        log(f"best.pt <- epoch {epoch} (val mAP50:95 {score:.4f})")

    t_train = time.time()
    for epoch in itertools.count(start_epoch):
        if epoch >= a.epochs:                  # a.epochs may be re-planned by --time
            break
        if ds.mosaic_p and epoch >= a.epochs - a.mosaic_off:
            ds.mosaic_p = 0.0                   # workers hold dataset copies: rebuild the loader to apply it
            dl = make_loader()
            log(f"epoch {epoch}: mosaic off for the last {a.mosaic_off} epochs")
        model.train()
        for m in frozen:                        # frozen parts keep their BatchNorm statistics
            m.eval()
        t_ep, t_data, t_loss = time.time(), 0.0, 0.0
        sums, n = {}, 0
        tick = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prof = None
        bar = EpochBar(epoch, a.epochs, iters_per_epoch, a.img) if not a.verbose else None
        for bi, (imgs, targets) in enumerate(dl):
            if bi >= iters_per_epoch:
                break
            if a.profile and bi == 10:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
                prof.__enter__()
                t_prof = time.time()
            if prof is not None and bi == 10 + a.profile:
                torch.cuda.synchronize()
                dtp = (time.time() - t_prof) / a.profile
                prof.__exit__(None, None, None)
                log(f"[profile] {dtp:.3f} s/it over {a.profile} its (batch {a.batch}, {a.batch / dtp:.1f} img/s)")
                for key in ("self_cuda_time_total", "cpu_time_total"):
                    log(f"[profile] top ops by {key}\n" + prof.key_averages().table(sort_by=key, row_limit=30))
                return
            t_data += time.time() - tick
            f = lr_factor(it, total_iters, a.warmup, a.flat, a.min_lr_ratio, a.schedule)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            if bi == 0 and (epoch == start_epoch or epoch == a.epochs - a.mosaic_off):
                try:
                    plot_train_batch(out / f"train_batch{'_nomosaic' if epoch and epoch == a.epochs - a.mosaic_off else '0'}.jpg",
                                     imgs, targets, classes)
                except Exception as e:  # noqa: BLE001
                    log(f"train batch preview skipped: {e}")
            x, tg, ctx = to_device(imgs, targets, device)
            if a.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                outputs = model(x, tg, ctx=ctx)
            t0 = time.time()
            losses = crit(outputs, tg)
            if dense_crit is not None:
                losses.update({k: v * a.dense_weight for k, v in dense_crit(outputs, tg).items()})
            loss = sum(losses.values())
            t_loss += time.time() - t0
            if not torch.isfinite(loss):
                log(f"non-finite loss at iter {it}: {[(k, float(v)) for k, v in losses.items() if not torch.isfinite(v)]}",
                    console=True)
                opt.zero_grad(set_to_none=True)
                it += 1
                tick = time.time()
                continue
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else 1e9)
            if not torch.isfinite(gn):          # never let one bad batch poison the weights
                if scaler is not None:
                    scaler.update()             # fp16 overflow: lower the loss scale, skip the step
                else:
                    log(f"non-finite grad norm at iter {it}; step skipped", console=True)
                opt.zero_grad(set_to_none=True)
                it += 1
                tick = time.time()
                continue
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            ema.update(model)
            # keep running sums on the GPU: a float() per loss term per step forced ~40 device syncs
            main = {k: v.detach() for k, v in losses.items() if "_aux" not in k and "_dn" not in k and "_enc" not in k
                    and "_pre" not in k}
            main["total"] = loss.detach()
            for k, v in main.items():
                sums[k] = sums[k] + v if k in sums else v.clone()
            n += 1
            it += 1
            if bar is not None:
                bar.update(bi + 1, {k: float(sums[k]) / n for k in sums if k in ("loss_mal", "loss_bbox", "loss_kld",
                                                                             "loss_angle", "loss_fgl")},
                           sum(len(t["labels"]) for t in targets),
                           torch.cuda.memory_reserved() / 2**30 if device.type == "cuda" else None,
                           force=bi + 1 == iters_per_epoch)
            if bi % a.log_every == 0:
                el = time.time() - t_ep
                log(f"ep {epoch} it {bi}/{iters_per_epoch} loss {float(loss.detach()):.3f} "
                    + " ".join(f"{k[5:]}={float(v):.3f}" for k, v in main.items() if k.startswith("loss_"))
                    + f" lr {opt.param_groups[-1]['lr']:.2e} | {(bi + 1) * a.batch / max(el, 1e-9):.1f} img/s"
                    f" data {t_data / (bi + 1):.3f}s crit {t_loss / (bi + 1):.3f}s/it")
            tick = time.time()
        if bar is not None:
            bar.close()
        dt = time.time() - t_ep
        rec = {"epoch": epoch, "iter": it, "lr": opt.param_groups[-1]["lr"], "epoch_s": round(dt, 1),
               "img_s": round(n * a.batch / max(dt, 1e-9), 1), "crit_s_per_it": round(t_loss / max(n, 1), 3),
               "data_s_per_it": round(t_data / max(n, 1), 3),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else None}
        rec.update({k: round(float(v) / max(n, 1), 4) for k, v in sums.items()})
        res = None
        if val_subset and ((epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs) and (epoch + 1 < a.epochs or full_val):
            t_val = time.time()
            res, _ = eval_dota(ema.module, a.data, device, val_subset, batch=a.batch, workers=a.workers,
                               num_top=a.num_top, img_size=a.img, log=log, context=a.context, vectors=a.vectors, post=a.post, merge_iou=a.merge_iou)
            if not a.verbose:
                val_rows(res, time.time() - t_val)
            rec["sub_mAP50"] = round(res["mAP50"], 4)
            rec["sub_mAP50_95"] = round(res["mAP50_95"], 4)
            if full_val and res["mAP50_95"] > best_score:
                best_score, best_epoch = res["mAP50_95"], epoch
                save_best(best_score, epoch)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "iter": it, "args": vars(a), "classes": list(classes), "best_score": best_score,
                    "best_epoch": best_epoch}, out / "last.tmp")
        os.replace(out / "last.tmp", last)
        jlog(rec)
        append_results(out, rec, res)
        plot_results(out)
        if a.time:                              # re-plan the run so it ends inside the time budget
            done = epoch + 1 - start_epoch
            per_ep = (time.time() - t_train) / done
            new_epochs = max(epoch + 1, start_epoch + int(a.time * 3600 / per_ep))
            if new_epochs != a.epochs:
                a.epochs = new_epochs
                total_iters = iters_per_epoch * a.epochs
                log(f"time={a.time}h: {per_ep:.0f}s/epoch -> {a.epochs} epochs", console=True)
            if time.time() - t_train + per_ep > a.time * 3600 * 1.02 and epoch + 1 < a.epochs:
                say(f"Stopping: time budget of {a.time} h reached.")
                break
        if a.patience and full_val and best_epoch >= 0 and epoch - best_epoch >= a.patience:
            say(f"Stopping early: no improvement in val mAP50-95 for {a.patience} epochs "
                f"(best {best_score:.4f} at epoch {best_epoch + 1}).")
            break
        log(f"epoch {epoch} done in {dt / 60:.1f} min: " + json.dumps(rec))

    if a.skip_final_eval:
        if not (out / "best.pt").exists():
            save_best(-1.0, a.epochs - 1)
        return
    weights = "ema_last"
    if full_val and best_score >= 0 and (out / "best.pt").exists() and not a.eval_only:
        ema.module.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"])
        weights = "best"
        say(f"\nTraining finished. Validating best.pt (epoch {best_epoch + 1}) on all val images...")
    else:
        say(f"\nTraining finished. Validating the final EMA model on all val images...")
    res, dets = eval_dota(ema.module, a.data, device, None, batch=a.batch, workers=a.workers, num_top=a.num_top,
                          img_size=a.img, log=log, fusion=a.dense or a.vec_ground > 0, variants=["dec", "dense", "union", "snap", "snapg"],
                          save_preds_to=(out / "val_preds.npz") if a.dense else None, context=a.context,
                          vectors=a.vectors, post=a.post, merge_iou=a.merge_iou)
    if not a.eval_only and not full_val:
        save_best(res["mAP50_95"], a.epochs - 1)
    table = summary_table(res, classes)
    log("\n" + table)
    (out / "eval_val.txt").write_text(table + "\n")
    (out / "eval_val.json").write_text(json.dumps(res, indent=1))
    write_task1(dets, out / "val_task1", classes)
    if a.eval_queries and a.eval_queries != a.queries:       # extra eval, main numbers stay at --queries
        ema.module.decoder.num_queries = a.eval_queries
        rq, _ = eval_dota(ema.module, a.data, device, None, batch=a.batch, workers=a.workers, num_top=a.eval_queries,
                          img_size=a.img, log=log, context=a.context, vectors=a.vectors, post=a.post, merge_iou=a.merge_iou)
        (out / f"eval_val_q{a.eval_queries}.txt").write_text(summary_table(rq, classes) + "\n")
        (out / f"eval_val_q{a.eval_queries}.json").write_text(json.dumps(rq, indent=1))
        jlog({"final_q": a.eval_queries, "mAP50": rq["mAP50"], "mAP50_95": rq["mAP50_95"]})
        ema.module.decoder.num_queries = a.queries
    final = rq if a.eval_queries and a.eval_queries != a.queries else res
    say(summary_table(final, classes))
    if not a.eval_only and (out / "best.pt").exists():  # per-class confidence (best F2) for predict conf=auto
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        ck["conf_thr"] = {c: round(final["classes"][c].get("conf_f2", 0.25), 3) for c in classes}
        ck["conf_f1"] = final.get("conf_f1")
        torch.save(ck, out / "best.tmp")
        os.replace(out / "best.tmp", out / "best.pt")
    try:
        plot_val_predictions(out, ema.module, a.data, device, a.img, classes, num_top=a.num_top)
    except Exception as e:  # noqa: BLE001 - a report must never fail the run
        log(f"val_pred.jpg skipped: {e}")
    lat = None
    if device.type == "cuda":
        m = ema.module.eval()
        x = torch.rand(1, 3, a.img, a.img, device=device)
        side = None
        if a.vectors:                           # latency includes the vector branch (1000 primitives)
            v = torch.rand(1, 1000, 21, device=device)
            v[..., 0] = 0
            side = {"vec": v, "vec_mask": torch.ones(1, 1000, dtype=torch.bool, device=device)}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(20):
                m(x, ctx=side)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(100):
                m(x, ctx=side)
            torch.cuda.synchronize()
        lat = (time.time() - t) / 100 * 1000
    if lat is not None:
        say(f"Speed: {lat:.1f} ms per {a.img}px tile (batch 1, fp16, PyTorch)")
    say(f"Results saved to {out}\n  best.pt (deploy / fine-tune), last.pt (resume), results.csv/png, val_pred.jpg")
    jlog({"final": True, "split": "val", "weights": weights, "mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
          "per_class_AP50": {c: round(r["AP50"], 4) for c, r in res["classes"].items()},
          "latency_ms_pt_fp16_bs1": lat, "fusion": {k: round(v["mAP50"], 4) for k, v in res.get("fusion", {}).items()}})
    if not a.eval_only and last.exists():   # finished: drop the optimiser state from last.pt (like best.pt)
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if "opt" in ck:
            torch.save({"model": ck["ema"]["module"], "epoch": ck["epoch"], "args": ck["args"],
                        "classes": ck["classes"], "best_score": ck.get("best_score")}, out / "last.tmp")
            os.replace(out / "last.tmp", last)
    (out / "train_done").write_text("ok")


if __name__ == "__main__":
    main()
