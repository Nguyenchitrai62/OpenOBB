"""VRDet trainer (single GPU). Resumes from {out}/last.pt, appends one JSON line per epoch to
{out}/metrics.jsonl, ends with a full DOTA-protocol evaluation of the EMA weights on val.

python -u -m vrdet.train --data /content/datasets/dota1_1024 --out OUT --size s --epochs 24 --batch 32
"""
import argparse
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
from vrdet.eval.dota import DOTA1_CLASSES, format_table, write_task1
from vrdet.models.dense_head import DenseCriterion
from vrdet.models.vrdet import VRDet, build_criterion, load_dfine_coco


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--img", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-mult", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--flat", type=float, default=0.5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.9998)
    ap.add_argument("--ema-warmups", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--num-top", type=int, default=300)
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
    ap.add_argument("--rotate-p", type=float, default=0.0, help="H8: arbitrary-angle rotation prob")
    ap.add_argument("--mosaic-p", type=float, default=0.0, help="H8: oriented mosaic prob")
    ap.add_argument("--context", action="store_true", help="H6: whole-image context tokens for each tile")
    ap.add_argument("--profile", type=int, default=0, help="profile N iterations after 10 warm-up ones, then exit")
    ap.add_argument("--channels-last", action="store_true", help="NHWC convs/BN (profiling: BN was ~42% of GPU time)")
    ap.add_argument("--compile", action="store_true", help="torch.compile backbone + encoder (falls back to eager)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    return ap.parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_done").exists():
        print("train_done exists: nothing to do", flush=True)
        return
    logf = open(out / "train_progress.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    classes = dataset_classes(a.data)
    model = VRDet(a.size, num_classes=len(classes), num_queries=a.queries, img_size=a.img,
                  rotate_sampling=not a.no_rotate_sampling, num_denoising=a.denoising, dense=a.dense,
                  strip_k=a.strip_k, ortho_heads=a.ortho_heads, context=a.context,
                  dense_queries=a.dense_queries)
    last = out / "last.pt"
    if not last.exists() and not a.no_pretrained:
        load_dfine_coco(model, a.size, class_names=classes, log=log)
    model.to(device)
    if a.channels_last:
        model.to(memory_format=torch.channels_last)
    if a.compile and device.type == "cuda":
        try:
            model.backbone.forward = torch.compile(model.backbone.forward)
            model.encoder.forward = torch.compile(model.encoder.forward, dynamic=False)
            log("torch.compile enabled for backbone + encoder")
        except Exception as e:  # noqa: BLE001
            log(f"torch.compile unavailable, eager mode: {e}")
    crit = build_criterion(num_classes=len(classes), box_loss=a.box_loss)
    a.dense = a.dense or a.dense_queries
    dense_crit = DenseCriterion() if a.dense else None
    ds = DotaPatches(a.data, "train", size=a.img, augment=True, hsv=tuple(a.hsv), limit=a.limit_train,
                     rotate_p=a.rotate_p, mosaic_p=a.mosaic_p, context=a.context)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)
    iters_per_epoch = len(dl) if a.max_iters is None else min(len(dl), a.max_iters)
    total_iters = iters_per_epoch * a.epochs
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.backbone_mult, a.wd), lr=a.lr, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = ModelEMA(model, a.ema, a.ema_warmups)
    start_epoch, it = 0, 0
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, it = ck["epoch"] + 1, ck["iter"]
        log(f"resumed from epoch {ck['epoch']} (iter {it})")
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"VRDet-{a.size} {n_par:.2f}M params | train patches {len(ds)} | {iters_per_epoch} it/epoch x {a.epochs} "
        f"| batch {a.batch} | device {device} amp={amp}")
    val_subset = None
    if a.eval_images:
        gts = sorted(p.stem for p in (Path(a.data) / "gt" / "val").glob("*.txt"))
        val_subset = sorted(random.Random(0).sample(gts, min(a.eval_images, len(gts))))

    for epoch in range(start_epoch, a.epochs):
        model.train()
        t_ep, t_data, t_loss = time.time(), 0.0, 0.0
        sums, n = {}, 0
        tick = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prof = None
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
            f = lr_factor(it, total_iters, a.warmup, a.flat, a.min_lr_ratio)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            x, tg, ctx = to_device(imgs, targets, device)
            if a.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                outputs = model(x, tg, ctx=ctx)
            t0 = time.time()
            losses = crit(outputs, tg)
            if dense_crit is not None:
                losses.update({k: v * a.dense_weight for k, v in dense_crit(outputs, tg).items()})
            loss = sum(losses.values())
            t_loss += time.time() - t0
            if not torch.isfinite(loss):
                log(f"non-finite loss at iter {it}: {[(k, float(v)) for k, v in losses.items() if not torch.isfinite(v)]}")
                opt.zero_grad(set_to_none=True)
                it += 1
                tick = time.time()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else 1e9)
            if not torch.isfinite(gn):          # never let one bad batch poison the weights
                log(f"non-finite grad norm at iter {it}; step skipped")
                opt.zero_grad(set_to_none=True)
                it += 1
                tick = time.time()
                continue
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
            if bi % a.log_every == 0:
                el = time.time() - t_ep
                log(f"ep {epoch} it {bi}/{iters_per_epoch} loss {float(loss):.3f} "
                    + " ".join(f"{k[5:]}={float(v):.3f}" for k, v in main.items() if k.startswith("loss_"))
                    + f" lr {opt.param_groups[-1]['lr']:.2e} | {(bi + 1) * a.batch / max(el, 1e-9):.1f} img/s"
                    f" data {t_data / (bi + 1):.3f}s crit {t_loss / (bi + 1):.3f}s/it")
            tick = time.time()
        dt = time.time() - t_ep
        rec = {"epoch": epoch, "iter": it, "lr": opt.param_groups[-1]["lr"], "epoch_s": round(dt, 1),
               "img_s": round(n * a.batch / max(dt, 1e-9), 1), "crit_s_per_it": round(t_loss / max(n, 1), 3),
               "data_s_per_it": round(t_data / max(n, 1), 3),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else None}
        rec.update({k: round(float(v) / max(n, 1), 4) for k, v in sums.items()})
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "iter": it, "args": vars(a)}, out / "last.tmp")
        os.replace(out / "last.tmp", last)
        if val_subset and ((epoch + 1) % a.eval_every == 0) and epoch + 1 < a.epochs:
            res, _ = eval_dota(ema.module, a.data, device, val_subset, batch=a.batch, workers=a.workers,
                               num_top=a.num_top, img_size=a.img, log=log, context=a.context)
            rec["sub_mAP50"] = round(res["mAP50"], 4)
            rec["sub_mAP50_95"] = round(res["mAP50_95"], 4)
        jlog(rec)
        log(f"epoch {epoch} done in {dt / 60:.1f} min: " + json.dumps(rec))

    if a.skip_final_eval:
        return
    res, dets = eval_dota(ema.module, a.data, device, None, batch=a.batch, workers=a.workers, num_top=a.num_top,
                          img_size=a.img, log=log, fusion=a.dense, variants=["dec", "dense"],
                          save_preds_to=(out / "val_preds.npz") if a.dense else None, context=a.context)
    table = format_table(res, classes)
    log("\n" + table)
    (out / "eval_val.txt").write_text(table + "\n")
    (out / "eval_val.json").write_text(json.dumps(res, indent=1))
    write_task1(dets, out / "val_task1", classes)
    lat = None
    if device.type == "cuda":
        m = ema.module.eval()
        x = torch.rand(1, 3, a.img, a.img, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(20):
                m(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(100):
                m(x)
            torch.cuda.synchronize()
        lat = (time.time() - t) / 100 * 1000
    jlog({"final": True, "split": "val", "weights": "ema_last", "mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
          "per_class_AP50": {c: round(r["AP50"], 4) for c, r in res["classes"].items()},
          "latency_ms_pt_fp16_bs1": lat, "fusion": {k: round(v["mAP50"], 4) for k, v in res.get("fusion", {}).items()}})
    (out / "train_done").write_text("ok")


if __name__ == "__main__":
    main()
