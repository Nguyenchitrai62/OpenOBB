"""Training/eval utilities: EMA, LR schedule, parameter groups, DOTA prediction + evaluation."""
import copy
import math
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader

from vrdet.data.dota import DotaPatches, collate, dataset_classes
from vrdet.eval.dota import DOTA1_CLASSES, evaluate, load_gt_dir, merge_patches, parse_patch_name
from vrdet.models.vrdet import postprocess
from vrdet.ops.obb import obb2poly


class ModelEMA:
    def __init__(self, model, decay=0.9998, warmups=1000):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, 0

    @torch.no_grad()
    def update(self, model):
        # e*d + m*(1-d) with mul/add, not lerp: lerp turns inf BatchNorm statistics into NaN (inf - inf), which made
        # every validation box NaN when fine-tuning from a checkpoint with an exploded running_var (2026-10-08)
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        msd = model.state_dict()
        fl_e, fl_m = [], []
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                fl_e.append(v)
                fl_m.append(msd[k].detach())
            else:
                v.copy_(msd[k])
        torch._foreach_mul_(fl_e, d)
        torch._foreach_add_(fl_e, fl_m, alpha=1 - d)

    @torch.no_grad()
    def repair(self, model, log=print):
        """Non-finite EMA tensors are re-synced from the live model (reported, never silently evaluated)."""
        msd = model.state_dict()
        bad = [k for k, v in self.module.state_dict().items()
               if v.dtype.is_floating_point and not torch.isfinite(v).all()]
        if bad:
            live_bad = [k for k in bad if not torch.isfinite(msd[k]).all()]
            log(f"WARNING: {len(bad)} EMA tensors are NaN/inf (e.g. {bad[:3]}); re-synced from the training model"
                + (f"; also non-finite in the training model (e.g. {live_bad[:3]}): its activations have exploded, "
                   f"usually from a learning rate far above the default (further repeats go to train_progress.log)"
                   if live_bad else ""))
            sd = self.module.state_dict()
            for k in bad:
                sd[k].copy_(torch.nan_to_num(msd[k], nan=0.0, posinf=0.0, neginf=0.0)
                            if "running_var" not in k else torch.nan_to_num(msd[k], nan=1.0, posinf=1.0, neginf=1.0))
        return bad

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]


def batchnorm_finite(model):
    """False when any BatchNorm running statistic is NaN/inf (a diverging run)."""
    return all(torch.isfinite(m.running_var).all() and torch.isfinite(m.running_mean).all()
               for m in model.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.running_var is not None)


@torch.no_grad()
def sanitize_batchnorm(model, limit=1e10, log=print):
    """BatchNorm running statistics that are NaN/inf or absurdly large (an earlier run diverged; training-mode BN
    hides it because it uses batch statistics) are reset and re-estimated by the next training steps."""
    reset = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats and m.running_var is not None:
            rv, rm = m.running_var, m.running_mean
            if not (torch.isfinite(rv).all() and torch.isfinite(rm).all()) or rv.abs().max() > limit \
                    or rm.abs().max() > limit:
                m.reset_running_stats()
                reset.append(name)
    if reset:
        log(f"WARNING: reset {len(reset)} BatchNorm layers with exploded running statistics (e.g. {reset[:3]}); the "
            f"source checkpoint probably came from a diverging run (learning rate too high)")
    return reset


def lr_factor(it, total, warmup, flat=0.5, min_ratio=0.1, schedule="flatcos"):
    """LR multiplier at iteration `it`. flatcos (D-FINE): linear warmup, flat until `flat` of the run, cosine to
    min_ratio. linear / cos (one-stage trainer recipe): decay from 1 to min_ratio (lrf) over the whole run, the
    warmup ramping up to the decayed value."""
    p = min(it / max(total, 1), 1.0)
    if schedule == "linear":
        base = (1 - p) * (1 - min_ratio) + min_ratio
    elif schedule == "cos":
        base = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    else:
        if it < warmup:
            return (it + 1) / warmup
        if p < flat:
            return 1.0
        q = min((p - flat) / max(1 - flat, 1e-9), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * q))
    return base * min(1.0, (it + 1) / max(warmup, 1))


def param_groups(model, lr, backbone_mult=0.5, wd=1e-4):
    groups = {"bb": [], "bb_norm": [], "nodecay": [], "rest": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        normish = ("norm" in n) or ("bn" in n) or p.ndim <= 1      # biases, norm weights, scales: no decay
        if n.startswith("backbone.") and not n.startswith("backbone.adapter."):   # ViT adapter: new layers
            groups["bb_norm" if normish else "bb"].append(p)
        elif normish or n.endswith(".bias"):
            groups["nodecay"].append(p)
        else:
            groups["rest"].append(p)
    return [dict(params=groups["bb"], lr=lr * backbone_mult, weight_decay=wd, name="bb"),
            dict(params=groups["bb_norm"], lr=lr * backbone_mult, weight_decay=0.0, name="bb_norm"),
            dict(params=groups["nodecay"], lr=lr, weight_decay=0.0, name="nodecay"),
            dict(params=groups["rest"], lr=lr, weight_decay=wd, name="rest")]


def ctx_batch(targets, device):
    """Side inputs gathered from the targets: global context (H6: thumb/tile/valid) and vector tokens (H7:
    vec (B, M, 21) zero-padded + vec_mask). None when the dataset provides neither."""
    out = {}
    if targets and "thumb" in targets[0]:
        out.update(thumb=torch.stack([d["thumb"] for d in targets]).to(device, non_blocking=True).float().div_(255.0),
                   tile=torch.stack([d["tile"] for d in targets]).to(device, non_blocking=True),
                   valid=torch.tensor([d["ctx_valid"] for d in targets], device=device))
    if targets and "vec" in targets[0]:
        M = max(len(d["vec"]) for d in targets)
        vec = torch.zeros(len(targets), M, targets[0]["vec"].shape[1])
        mask = torch.zeros(len(targets), M, dtype=torch.bool)
        for i, d in enumerate(targets):
            vec[i, :len(d["vec"])] = d["vec"]
            mask[i, :len(d["vec"])] = True
        out.update(vec=vec.to(device, non_blocking=True), vec_mask=mask.to(device, non_blocking=True))
    return out or None


def to_device(imgs, targets, device):
    x = imgs.to(device, non_blocking=True).float().div_(255.0)
    t = [{"labels": d["labels"].to(device, non_blocking=True), "boxes": d["boxes"].to(device, non_blocking=True)}
         for d in targets]
    return x, t, ctx_batch(targets, device)


@torch.no_grad()
def predict_patches(model, ds, device, batch=32, workers=8, num_top=300, img_size=1024, amp=True, post="flat"):
    """-> {"dec": [(patch, cls, score, poly8)], "dense": [...] (only if the model has a dense head)}."""
    from vrdet.models.dense_head import dense_predict
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True)
    out = {"dec": []}
    has_dense = getattr(model, "dense_head", None) is not None
    if has_dense:
        out["dense"] = []
    for imgs, tg in dl:
        x = imgs.to(device, non_blocking=True).float().div_(255.0)
        ctx = ctx_batch(tg, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            o = model(x, ctx=ctx) if ctx is not None else model(x)
        s, l, b, qi = postprocess(o, num_top, img_size, mode=post, return_query=True)
        s, l, b, qi = s.cpu().numpy(), l.cpu().numpy(), b.cpu().numpy(), qi.cpu().numpy()
        grounded = "pred_members" in o
        if grounded:                           # H17: snap boxes to the selected CAD primitives
            from vrdet.models.vrdet import snap_boxes
            out.setdefault("snap", [])
            out.setdefault("snapg", [])
            mem = o["pred_members"].float().cpu().numpy()
            vp, vm = o["vec_pts"].cpu().numpy(), o["vec_mask"].cpu().numpy()
        for i, t in enumerate(tg):
            polys = obb2poly(b[i])
            for sc, lab, p in zip(s[i], l[i], polys):
                out["dec"].append((t["name"], int(lab), float(sc), p))
            if grounded:
                sp = snap_boxes(mem[i], vp[i], vm[i], qi[i], polys, img_size)
                for sc, lab, p in zip(s[i], l[i], sp):
                    out["snap"].append((t["name"], int(lab), float(sc), p))
                sg = snap_boxes(mem[i], vp[i], vm[i], qi[i], polys, img_size, box_gate=0.2)
                for sc, lab, p in zip(s[i], l[i], sg):
                    out["snapg"].append((t["name"], int(lab), float(sc), p))
        if has_dense:
            for t, (ds_, dl_, db_) in zip(tg, dense_predict(o, img_size=img_size)):
                for sc, lab, p in zip(ds_.cpu().numpy(), dl_.cpu().numpy(), obb2poly(db_.cpu().numpy())):
                    out["dense"].append((t["name"], int(lab), float(sc), p))
    return out


def _max_side(p):
    return max(np.hypot(p[2] - p[0], p[3] - p[1]), np.hypot(p[4] - p[2], p[5] - p[3]))


def fusion_sets(preds, route=(32, 64)):
    """Inference-time ways to combine the decoder (sparse) and dense outputs; all go through the same merge NMS."""
    sets = {"dec": preds["dec"]}
    for k in ("snap", "snapg"):
        if k in preds:
            sets[k] = preds[k]
    if "dense" in preds:
        sets["dense"] = preds["dense"]
        sets["union"] = preds["dec"] + preds["dense"]
        for r in route:
            sets[f"route{r}"] = [d for d in preds["dense"] if _max_side(d[3]) < r] +                                 [d for d in preds["dec"] if _max_side(d[3]) >= r]
    return sets


def _merge_chunk(args):
    return merge_patches(*args)


def merge_parallel(patch_dets, iou_thr=0.1, workers=16):
    """merge_patches split by source image across processes."""
    by_img = {}
    for d in patch_dets:
        by_img.setdefault(parse_patch_name(d[0])[0], []).append(d)
    imgs = sorted(by_img)
    if workers <= 1 or len(imgs) < 2:
        return merge_patches(patch_dets, iou_thr)
    chunks = [[d for im in imgs[i::workers] for d in by_img[im]] for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_merge_chunk, [(c, iou_thr) for c in chunks if c])
    merged = {}
    for part in parts:
        for c, (ids, sc, pl) in part.items():
            if c in merged:
                i0, s0, p0 = merged[c]
                merged[c] = (i0 + ids, np.concatenate([s0, sc]), np.concatenate([p0, pl]))
            else:
                merged[c] = (list(ids), sc, pl)
    return merged


def save_preds(preds, path):
    """Compact npz of raw patch predictions (for offline fusion studies)."""
    names = sorted({d[0] for v in preds.values() for d in v})
    nid = {n: i for i, n in enumerate(names)}
    arrs = {"names": np.array(names)}
    for k, v in preds.items():
        arrs[f"{k}_patch"] = np.array([nid[d[0]] for d in v], np.int32)
        arrs[f"{k}_cls"] = np.array([d[1] for d in v], np.int8)
        arrs[f"{k}_score"] = np.array([d[2] for d in v], np.float16)
        arrs[f"{k}_poly"] = np.array([d[3] for d in v], np.float32).reshape(-1, 8).astype(np.float16)
    np.savez_compressed(path, **arrs)


def load_preds(path):
    z = np.load(path)
    names = z["names"]
    out = {}
    for k in ("dec", "dense"):
        if f"{k}_patch" in z:
            out[k] = [(str(names[i]), int(c), float(s), p.astype(np.float64)) for i, c, s, p in
                      zip(z[f"{k}_patch"], z[f"{k}_cls"], z[f"{k}_score"], z[f"{k}_poly"])]
    return out


def eval_dota(model, data_root, device, image_ids=None, batch=32, workers=8, num_top=300, img_size=1024,
              merge_workers=16, log=print, fusion=False, variants=None, save_preds_to=None, context=False,
              vectors=False, post="flat", merge_iou=0.1, primary="dec", ultra=False, metric="dota"):
    """DOTA-protocol eval of the decoder output (primary). With fusion=True and a dense head, also scores
    the dense-only / union / size-routed variants (res["fusion"])."""
    t0 = time.time()
    ds = DotaPatches(data_root, "val", augment=False, context=context, vectors=vectors)
    if image_ids is not None:
        keep = set(image_ids)
        ds.items = [m for m in ds.items if m["src"] in keep]
    preds = predict_patches(model, ds, device, batch, workers, num_top, img_size, post=post)
    for k, v in preds.items():                  # drop non-finite boxes (diverging training) before merge / eval
        good = [d for d in v if np.isfinite(d[2]) and np.isfinite(np.asarray(d[3], dtype=np.float64)).all()]
        if len(good) < len(v):
            print(f"WARNING: {len(v) - len(good)} of {len(v)} {k} predictions are NaN/inf (training is diverging: "
                  f"lower the learning rate)", flush=True)
            preds[k] = good
    t1 = time.time()
    classes = dataset_classes(data_root)
    gts = load_gt_dir(f"{data_root}/gt/val", set(image_ids) if image_ids is not None else None)
    if save_preds_to:
        save_preds(preds, save_preds_to)
    sets = fusion_sets(preds) if fusion else {"dec": preds["dec"]}
    if variants:
        sets = {k: v for k, v in sets.items() if k in variants}
    results, primary_dets = {}, None
    for name, pd in sets.items():
        merged = merge_parallel(pd, merge_iou, merge_workers)
        dets = {classes[c]: v for c, v in merged.items()}
        results[name] = evaluate(dets, gts, classes)
        if name == (primary if primary in sets else "dec"):
            primary_dets = dets
    res = results[primary if primary in results else "dec"]
    if ultra or metric == "yolo":               # Ultralytics-validator conventions, comparable with YOLO's mAP
        from vrdet.eval.ultra import evaluate_ultra
        res["ultra"] = evaluate_ultra(primary_dets, gts, classes)
    if metric == "yolo":                        # report (and select best.pt) on the YOLO-comparable numbers
        u = res["ultra"]
        res["dota"] = {"mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
                       "classes": {c: {"AP50": r["AP50"], "AP50_95": r["AP50_95"]} for c, r in res["classes"].items()}}
        res["mAP50"], res["mAP50_95"] = u["mAP50"], u["mAP50_95"]
        for c, r in u["classes"].items():
            res["classes"][c]["AP50"], res["classes"][c]["AP50_95"] = r["AP50"], r["AP50_95"]
        res["metric"] = "yolo"
    if len(results) > 1:
        res["fusion"] = {k: {"mAP50": v["mAP50"], "mAP50_95": v["mAP50_95"],
                             "per_class_AP50": {c: round(r["AP50"], 4) for c, r in v["classes"].items()}}
                         for k, v in results.items()}
    log(f"[eval] {len(ds)} patches / {len(gts)} images: predict {t1 - t0:.0f}s, merge+eval {time.time() - t1:.0f}s, "
        + " | ".join(f"{k} mAP50 {100 * v['mAP50']:.2f}" for k, v in results.items()))
    return res, primary_dets
