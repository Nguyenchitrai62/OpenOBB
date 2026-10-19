"""OpenOBB trainer (single GPU). Resumes from {out}/last.pt, appends one JSON line per epoch to
{out}/metrics.jsonl, ends with a full DOTA-protocol evaluation of the EMA weights on val.

python -u -m openobb.train --data /content/datasets/dota1_1024 --out OUT --size s --epochs 24 --batch 32
"""
import argparse
import copy
import itertools
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openobb import __version__
from openobb.data.dota import DotaPatches, collate, dataset_classes
from openobb.engine import (ModelEMA, batchnorm_finite, eval_dota, lr_factor, param_groups, sanitize_batchnorm,
                          to_device, yolo_lf)
from openobb.console import EpochBar, val_rows
from openobb.report import append_results, plot_labels, plot_results, plot_train_batch, plot_val_predictions
from openobb.eval.dota import DOTA1_CLASSES, format_table, summary_table, write_task1
from openobb.models.dense_head import DenseCriterion
from openobb.models.openobb1 import OpenOBB1, build_criterion, load_dfine_coco


def fused_copy(module, device):
    """Copy of the EMA model for validation with BatchNorm / RepConv folded and channels_last, as the final validation
    and inference run it (same outputs to float rounding; per-epoch validation used the unfolded training graph, F115).
    Models without fuse() are returned as they are."""
    if not hasattr(module, "fuse") or not hasattr(module, "run_head"):
        return module
    vm = copy.deepcopy(module).eval()
    vm.fuse()
    from openobb.predict import fast_layout
    return fast_layout(vm, device)


def compile_training(model):
    """torch.compile of what a dense OpenOBB (v18-v24) runs in training: encode (trunk stages + neck; v18+ call the
    trunk's stages directly, so compiling backbone.forward compiled nothing, F112), the boundary field and the heads.
    Static shapes (square training inputs). A graph that fails to compile falls back to eager instead of stopping
    the run. Call after the EMA copy (the EMA / validation model stays eager)."""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    done = []
    model.encode = torch.compile(model.encode, dynamic=False)
    done.append("trunk + neck")
    if getattr(model, "field", None) is not None:
        model.field.forward = torch.compile(model.field.forward, dynamic=False)
        done.append("boundary field")
    model.run_head = torch.compile(model.run_head, dynamic=False)
    done.append("heads")
    return ", ".join(done) + " (training model only; first steps compile, a few minutes)"


def _to_cpu(obj):
    """Snapshot of a checkpoint: every tensor copied to the CPU (the live ones keep changing while it is written)."""
    if torch.is_tensor(obj):
        t = obj.detach()
        return t.clone() if t.device.type == "cpu" else t.to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class AsyncSaver:
    """torch.save in a background thread (F112): the training loop only waits for the copy to the CPU, not for the
    write (last.pt of size x is ~0.8 GB: model + EMA + optimiser; tens of seconds on a Drive folder). One saver per
    file; wait() before reading the file back."""

    def __init__(self):
        self.thread, self.error = None, None

    def save(self, obj, tmp, dst):
        self.wait()
        snap = _to_cpu(obj)

        def run():
            try:
                torch.save(snap, tmp)
                os.replace(tmp, dst)
            except Exception as e:  # noqa: BLE001 - re-raised by wait()
                self.error = e
        self.thread = threading.Thread(target=run, name=f"save-{Path(dst).name}")
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            e, self.error = self.error, None
            raise e


def _has_keys(path, prefix):
    """True if the checkpoint at `path` holds a parameter whose name starts with `prefix`."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("ema") or ck.get("model") or ck if isinstance(ck, dict) else {}
    if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
        sd = sd["module"]
    return any(isinstance(k, str) and k.startswith(prefix) for k in sd)


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False)          # a typo must fail, not match another flag
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--arch", default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24"],
                    help="v1: OpenOBB (DETR-style decoder); v2: OpenOBB2 (dense segment-aware, NMS-free); "
                         "v3: OpenOBB3 (pretrained encoder + dense anisotropic oriented head); "
                         "v4: OpenOBB4 (DINOv2 backbone + DETR decoder + dense oriented branch); "
                         "v6: OpenOBB6 (own backbone / neck / head, docs/OPENOBB6.md); "
                         "v7: OpenOBB7 (OpenOBB3 repaired, docs/OPENOBB7.md); "
                         "v8: OpenOBB8 (OpenOBB7 + oriented relational refiner, docs/OPENOBB8.md); "
                         "v9: OpenOBB9 (new re-parameterised core + OpenOBB8's head, docs/OPENOBB9.md); "
                         "v10: OpenOBB10 (OpenOBB8 + multi-slot head, reach-aware assignment, class balance, "
                         "docs/OPENOBB10.md); "
                         "v11: OpenOBB11 (dense proposals + one-to-one relational set decoder, NMS-free, docs/OPENOBB11.md); "
                         "v12 / v13: OpenOBB7 / OpenOBB9 core + dual dense head, NMS-free (docs/OPENOBB12.md)")
    ap.add_argument("--backbone", default="hgnet", choices=["hgnet", "dinov2_s", "dinov2_b", "dinov2_l"])
    ap.add_argument("--dense-v3", action="store_true", help="dense branch = OpenOBB3's oriented head (OpenOBB4)")
    ap.add_argument("--geo-cls", action="store_true", help="OpenOBB5: class branch reads the predicted geometry")
    ap.add_argument("--relate", action="store_true", help="OpenOBB5: relation re-scoring of the best candidates")
    ap.add_argument("--rel-k", type=int, default=600, help="OpenOBB5: candidates per image for relation re-scoring")
    ap.add_argument("--v7-off", default="", help="OpenOBB7 ablation: parts switched off, comma-separated "
                                                  "(guard, lsk_bound, spd, cls_ctx)")
    ap.add_argument("--v8-off", default="", help="OpenOBB8 ablation: refiner parts switched off, comma-separated "
                                                  "(relate, geo_bias, ring, p2, jitter, refine)")
    ap.add_argument("--v9-off", default="", help="OpenOBB9 ablation: core parts switched off, comma-separated")
    ap.add_argument("--v14-off", default="", help="OpenOBB14 ablation: parts switched off, comma-separated "
                                                  "(attn, spd, edge, dilate, maxpool, guard, dual, stage4, widehead)")
    ap.add_argument("--v24-off", default="", help="OpenOBB24 ablation: 'quality' (no localisation-quality head)")
    ap.add_argument("--ref-k", type=int, default=1500, help="OpenOBB14: candidates refined by the edge refiner at "
                                                            "inference (0: off)")
    ap.add_argument("--teacher", default=None,
                    help="OpenOBB18: trained OpenOBB12 / 17 checkpoint whose encoder output (P3-P5) the own neck is "
                         "distilled from during training (not part of the saved model)")
    ap.add_argument("--w-distill", type=float, default=2.0, help="OpenOBB18: weight of the neck distillation loss")
    ap.add_argument("--w-ld", type=float, default=0.25,
                    help="OpenOBB18+: weight of the teacher's box-distribution distillation (localization distillation; 0: off)")
    ap.add_argument("--w-kd-cls", type=float, default=0.5,
                    help="OpenOBB18+: weight of the teacher's class-probability distillation (0: off)")
    ap.add_argument("--groups", default="",
                    help="OpenOBB17 / 18 / 19: contention-aware class groups (one box per group at every point): 'auto' "
                         "(from the training labels, OpenOBB19's default), a group id per class '0,0,1,...', or '' (one)")
    ap.add_argument("--n-groups", type=int, default=3, help="number of class groups for --groups auto")
    ap.add_argument("--neck", default="own", choices=["own", "dfine"],
                    help="OpenOBB14: own re-parameterised neck, or D-FINE's COCO-pretrained hybrid encoder (control)")
    ap.add_argument("--nms-iou", default="auto",
                    help="duplicate suppression: auto (calibrated on val at the end and saved in best.pt as the "
                         "inference default), none (no NMS; OpenOBB11), or a ProbIoU threshold")
    ap.add_argument("--box", default="dfl", choices=["dfl", "direct"],
                    help="OpenOBB12 / 13 box regression: dfl (distributions) or direct (values, YOLO26-style, F54)")
    ap.add_argument("--v10-off", default="", help="OpenOBB10 ablation: parts switched off, comma-separated "
                                                   "(slots, reach, balance)")
    ap.add_argument("--v6-off", default="", help="OpenOBB6 ablation: parts switched off, comma-separated "
                                                  "(strip, axis, attn, spd, maxpool, cls_ctx)")
    ap.add_argument("--img", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="image scale the data was tiled at (recorded for inference; set by openobb.data.prepare)")
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
    ap.add_argument("--epoch-lr", action="store_true",
                    help="usual one-stage trainer: the LR factor lf(epoch) (linear / cos to min-lr-ratio) changes once per "
                         "epoch; during the --warmup iterations every LR moves linearly to lr * lf(epoch)")
    ap.add_argument("--warmup-bias-lr", type=float, default=0.0,
                    help="> 0: biases get their own groups whose LR falls from this value to lr * lf(epoch) during warmup "
                         "(usual one-stage trainer: 0.1); the other groups rise from 0")
    ap.add_argument("--nbs", type=int, default=0,
                    help="nominal batch size (usual one-stage trainer: 64): gradients are accumulated over "
                         "round(nbs / batch) batches (ramped from 1 during warmup) and weight decay is scaled by "
                         "batch * accumulate / nbs; 0 = one optimizer step per batch")
    ap.add_argument("--cache", action="store_true", help="decode training images once into RAM")
    ap.add_argument("--fliplr", type=float, default=0.5, help="horizontal flip probability")
    ap.add_argument("--flipud", type=float, default=0.5, help="vertical flip probability")
    ap.add_argument("--no-rot90", action="store_true", help="no random 90-degree rotations")
    ap.add_argument("--unlock-all", action="store_true",
                    help="freeze nothing: every frozen BatchNorm (the pretrained trunk of sizes l / x) becomes a trainable "
                         "BatchNorm2d starting from its frozen statistics, and every frozen parameter (the trunk stem) "
                         "trains (default: frozen as in D-FINE)")
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
                    help="fine-tune: initialise from a OpenOBB checkpoint (last.pt); class heads are re-used when the "
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
    ap.add_argument("--metric", default="yolo", choices=["yolo", "dota"],
                    help="mAP shown each epoch and used for best.pt: 'yolo' = Ultralytics validator conventions "
                         "(ProbIoU, 101-point AP; comparable with YOLO), 'dota' = DOTA devkit (polygon IoU, VOC07)")
    ap.add_argument("--primary", default="dec", choices=["dec", "union"],
                    help="detections scored as the model output: decoder only, or decoder + dense head (NMS)")
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
    ap.add_argument("--angle-loss", default="vec", choices=["vec", "cos"],
                    help="dense heads: vec = L1 on (cos 2t, sin 2t) (F84 fix: vertical objects were tilted), cos = former 1 - cos 2dt")
    ap.add_argument("--angle-balance", default="uniform", choices=["uniform", "tal"],
                    help="dense heads: angle loss weight per positive; uniform (F86 fix: vertical objects got ~0 weight), tal = former")
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
    ap.add_argument("--wdir", default="", help="sub-folder of --out for last.pt / best.pt (the CLI uses 'weights')")
    ap.add_argument("--verbose", action="store_true",
                    help="print the detailed per-iteration log instead of the compact progress table")
    return ap


def get_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wdir = out / a.wdir if a.wdir else out
    wdir.mkdir(parents=True, exist_ok=True)
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
    v2, v4, v5 = a.arch == "v2", a.arch == "v4", a.arch == "v5"
    v3 = a.arch in ("v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24")
    v6, v7, v8, v10 = a.arch == "v6", a.arch == "v7", a.arch == "v8", a.arch == "v10"
    v11, v12, v13, v14 = a.arch == "v11", a.arch in ("v12", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24"), a.arch == "v13", a.arch in ("v14", "v15", "v16")
    v17 = a.arch in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24")   # OpenOBB12 + boundary field / snapping: every v12 path applies
    v18 = a.arch in ("v18", "v19", "v20", "v21", "v22", "v23", "v24")       # OpenOBB17 with the own neck distilled from a trained encoder
    v19 = a.arch in ("v19", "v20", "v21", "v22", "v23", "v24")              # OpenOBB18 + contention-aware class groups
    v20 = a.arch in ("v20", "v21", "v22", "v23", "v24")             # OpenOBB19 + edge refiner on the trunk's stride-4 map
    v21 = a.arch in ("v21", "v22", "v23", "v24")             # OpenOBB20 + double-angle residual angle head
    v22 = a.arch == "v22"                       # OpenOBB21 + bounded angle from 45 deg + direct boxes
    v23 = a.arch in ("v23", "v24")              # OpenOBB21 + rotational boundary snapping + peak filter
    v24 = a.arch == "v24"                       # OpenOBB23 + learnt localisation quality (quality-guided NMS)
    if v22:
        a.box = "direct"
    if v17:
        from openobb.models.groups import parse_groups
        if v19 and not a.groups:
            a.groups = "auto"
        if a.groups and a.groups != "auto":
            try:
                parse_groups(a.groups, len(classes))
            except ValueError:                  # inherited from a checkpoint with other classes: recompute
                a.groups = "auto"
        if a.groups == "auto":
            from openobb.models.groups import contention_groups
            meta = Path(a.data) / "meta" / "train.jsonl"
            items = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
            g = contention_groups(items, len(classes), a.n_groups)
            a.groups = ",".join(map(str, g))
            log("[groups] contention-aware class groups (from the training labels): "
                + " | ".join(", ".join(c for c, k in zip(classes, g) if k == j) for j in range(max(g) + 1)),
                console=True)
    v15, v16 = a.arch in ("v15", "v16"), a.arch == "v16"
    v9 = a.arch == "v9"
    dense_arch = v2 or v3                       # dense heads: no queries / decoder
    if v4:                                      # DETR decoder fed and supervised by the dense oriented branch
        a.dense = a.dense_queries = a.dense_v3 = True
    if v2:
        from openobb.models.openobb2 import OpenOBB2
        from openobb.models.openobb2_loss import OpenOBB2Loss
        model = OpenOBB2(a.size, num_classes=len(classes), img_size=a.img)
    elif v3:
        from openobb.models.openobb3 import OpenOBB3
        from openobb.models.openobb3_loss import OpenOBB3Loss
        if v14:
            from openobb.models.openobb14 import OpenOBB14
            from openobb.models.openobb15 import OpenOBB15
            from openobb.models.openobb16 import OpenOBB16
            cls14 = OpenOBB16 if v16 else (OpenOBB15 if v15 else OpenOBB14)
            model = cls14(a.size, num_classes=len(classes), img_size=a.img, max_det=a.num_top,
                          off=a.v14_off or None, box=a.box, ref_k=a.ref_k, neck=a.neck)
        elif v17:
            from openobb.models.openobb17 import OpenOBB17
            from openobb.models.openobb18 import OpenOBB18
            from openobb.models.openobb18 import OpenOBB19, OpenOBB20, OpenOBB21, OpenOBB22, OpenOBB23
            from openobb.models.openobb24 import OpenOBB24
            v17_kw = {"off24": a.v24_off or None} if v24 else {}
            model = (OpenOBB24 if v24 else OpenOBB23 if v23 else OpenOBB22 if v22 else OpenOBB21 if v21 else OpenOBB20 if v20 else OpenOBB19 if v19 else OpenOBB18 if v18 else OpenOBB17)(
                a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top, off=a.v7_off or None,
                box=a.box, ref_k=a.ref_k, groups=a.groups or None, **v17_kw)
        elif v12 or v13:
            from openobb.models.openobb12 import OpenOBB12, OpenOBB13
            model = (OpenOBB13 if v13 else OpenOBB12)(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk,
                                                      max_det=a.num_top, off=a.v7_off or None, box=a.box)
        elif v11:
            from openobb.models.openobb11 import OpenOBB11
            model = OpenOBB11(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                              off=a.v8_off or None, off7=a.v7_off or None, rel_k=a.rel_k)
        elif v10:
            from openobb.models.openobb10 import OpenOBB10
            model = OpenOBB10(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                              off=a.v10_off or None, off8=a.v8_off or None, off7=a.v7_off or None, rel_k=a.rel_k)
        elif v9:
            from openobb.models.openobb9 import OpenOBB9
            model = OpenOBB9(a.size, num_classes=len(classes), img_size=a.img, max_det=a.num_top,
                             off=a.v9_off or None, off8=a.v8_off or None, rel_k=a.rel_k)
        elif v8:
            from openobb.models.openobb8 import OpenOBB8
            model = OpenOBB8(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                             off=a.v8_off or None, off7=a.v7_off or None, rel_k=a.rel_k)
        elif v7:
            from openobb.models.openobb7 import OpenOBB7
            model = OpenOBB7(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                             off=a.v7_off or None)
        elif v6:
            from openobb.models.openobb6 import OpenOBB6
            model = OpenOBB6(a.size, num_classes=len(classes), img_size=a.img, max_det=a.num_top, off=a.v6_off)
        elif v5:
            from openobb.models.openobb5 import OpenOBB5
            from openobb.models.openobb5_loss import OpenOBB5Loss
            model = OpenOBB5(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                           backbone=a.backbone, geo_cls=a.geo_cls, relate=a.relate, rel_k=a.rel_k)
        else:
            model = OpenOBB3(a.size, num_classes=len(classes), img_size=a.img, lsk=a.lsk, max_det=a.num_top,
                           backbone=a.backbone)
    else:
        model = OpenOBB1(a.size, num_classes=len(classes), num_queries=a.queries, img_size=a.img,
                      rotate_sampling=not a.no_rotate_sampling, num_denoising=a.denoising, dense=a.dense,
                      strip_k=a.strip_k, ortho_heads=a.ortho_heads, context=a.context,
                      dense_queries=a.dense_queries, vectors=a.vectors, vec_dim=a.vec_dim, vec_layers=a.vec_layers,
                      o2m_queries=a.o2m_queries, lsk=a.lsk, vec_lfe=a.vec_lfe, vec_ground=a.vec_ground > 0, p2=a.p2,
                      backbone=a.backbone, dense_v3=a.dense_v3)
    if a.unlock_all:                            # before any loading: the loaders fill the new BatchNorm layers by name
        if a.freeze:
            raise SystemExit("unlock_all=True and freeze= contradict each other: pass one of them")
        from openobb.models.common import unlock_all
        n_bn, n_thaw = unlock_all(model)
        log(f"[init] unlock_all: {n_bn} frozen BatchNorm layers -> trainable BatchNorm2d (start = the frozen statistics, "
            f"momentum 0.03), {n_thaw} frozen tensors made trainable", console=True)
    last = wdir / "last.pt"
    if not last.exists() and a.weights:
        # shape-tolerant copy of a OpenOBB checkpoint; class rows are matched by name (new classes start fresh)
        src_ck = torch.load(a.weights, map_location="cpu", weights_only=False)
        src_cls = list(src_ck.get("classes") or [])
        src_box = (src_ck.get("args") or {}).get("box") or "dfl"
        src_groups = (src_ck.get("args") or {}).get("groups") or ""
        del src_ck
        cmap = {i: src_cls.index(c) for i, c in enumerate(classes) if c in src_cls}
        load_dfine_coco(model, a.weights, log=lambda m: log(m, console=True), class_map=cmap)
        log(f"[init] fine-tuning from {a.weights} ({len(cmap)}/{len(classes)} classes carried over)")
        init_desc = f"fine-tune from {a.weights} ({len(cmap)}/{len(classes)} classes carried over)"
        if v10 and getattr(model, "slots", 1) > 1 and not _has_keys(a.weights, "head.reg_slot."):
            model.head.copy_primary_box()       # single-slot checkpoint: extra slots start from slot 0's boxes
        if (v12 or v13 or v14) and not _has_keys(a.weights, "head_o2o."):
            model.copy_head_to_o2o()            # single-head checkpoint: the one-to-one head starts as its copy
        if v17 and getattr(model, "n_groups", 1) > 1 and not src_groups:
            model.replicate_group_boxes()       # one-box checkpoint: every class group starts as its box predictor
        if hasattr(model, "after_weights") and model.after_weights(a.weights):
            log("[init] OpenOBB22: box outputs rebuilt per class group (angle from 45 deg, ends / thickness restart, "
                "across offset kept)", console=True)
        elif (v12 or v13) and a.box == "direct" and src_box != "direct":
            from openobb.models.openobb12 import restart_direct_distances
            restart_direct_distances(model)     # DFL checkpoint -> direct boxes: distances restart, angle / across kept
            log("[init] box=direct from a DFL checkpoint: distance / thickness outputs restart at zero", console=True)
    elif not last.exists() and not a.no_pretrained and not v2 and not v6 and not v9:
        load_dfine_coco(model, a.size, class_names=classes, log=lambda m: log(m, console=True), init=a.init)
        init_desc = "D-FINE COCO weights (Apache-2.0)"
        if a.backbone != "hgnet":               # ViT backbone: its own weights; encoder input projections restart
            model.backbone.load_pretrained(log=lambda m: log(m, console=True))
            model.reset_input_proj()
            init_desc = f"DINOv2 {a.backbone} backbone + D-FINE COCO encoder/decoder (Apache-2.0)"
    else:
        init_desc = "resume" if last.exists() else "random (trained from scratch)"
    sanitize_batchnorm(model, log=lambda m: log(m, console=True))
    from openobb.predict import (DUP_TOL, apply_head, apply_nms, choose_postprocess, dup_rate, merge_iou_for, parse_nms,
                                 postprocess_options)
    apply_nms(model, a.nms_iou)                 # fixed value: used for every val; auto: architecture default
    model.to(device)
    if a.channels_last:
        model.to(memory_format=torch.channels_last)
    from openobb.models import openobb3_loss as _l3
    _l3.ANGLE_LOSS, _l3.ANGLE_WEIGHT = a.angle_loss, a.angle_balance
    if v2:
        crit = OpenOBB2Loss(img_size=a.img)
    elif v3:
        if v14:
            from openobb.models.openobb14_loss import OpenOBB14Loss
            crit = OpenOBB14Loss(img_size=a.img)
        elif v18:
            from openobb.models.openobb18_loss import OpenOBB18Loss
            teacher = None
            if a.teacher:
                from openobb.predict import load_model
                teacher = load_model(a.teacher, device)[0].eval()
                for p_ in teacher.parameters():
                    p_.requires_grad_(False)
                log(f"[distill] neck distilled from the encoder of {a.teacher} (weight {a.w_distill:g})", console=True)
            crit = OpenOBB18Loss(img_size=a.img, teacher=teacher, w_distill=a.w_distill, w_ld=a.w_ld, w_kd_cls=a.w_kd_cls)
        elif v17:
            from openobb.models.openobb17_loss import OpenOBB17Loss
            crit = OpenOBB17Loss(img_size=a.img)
        elif v12 or v13:
            from openobb.models.openobb12_loss import OpenOBB12Loss
            crit = OpenOBB12Loss(img_size=a.img, off="slots" if v13 else "slots,reach")   # v13: P2 reach is short
        elif v11:
            from openobb.models.openobb11_loss import OpenOBB11Loss
            crit = OpenOBB11Loss(img_size=a.img)
        elif v10:
            from openobb.models.openobb10_loss import OpenOBB10Loss
            crit = OpenOBB10Loss(img_size=a.img, off=a.v10_off or None)
        elif v8 or v9:
            from openobb.models.openobb8_loss import OpenOBB8Loss
            crit = OpenOBB8Loss(img_size=a.img)
        else:
            crit = OpenOBB5Loss(img_size=a.img) if v5 else OpenOBB3Loss(img_size=a.img)
    else:
        crit = build_criterion(num_classes=len(classes), box_loss=a.box_loss, o2m_k=a.o2m_k, aqd=a.aqd,
                               angle_weight=a.angle_weight, cost_iou=a.cost_iou, member_weight=a.vec_ground)
    a.dense = (a.dense or a.dense_queries) and not dense_arch
    if a.dense and a.dense_v3:
        from openobb.models.openobb3_loss import DenseOrientedCriterion
        dense_crit = DenseOrientedCriterion(img_size=a.img)
    else:
        dense_crit = DenseCriterion() if a.dense else None
    ds = DotaPatches(a.data, "train", size=a.img, augment=True, hsv=tuple(a.hsv), limit=a.limit_train,
                     rotate_p=a.rotate_p, mosaic_p=a.mosaic_p, context=a.context, ctx_dropout=a.ctx_dropout,
                     vectors=a.vectors, max_tokens=a.max_tokens, scale_jitter=a.scale_jitter,
                     translate=a.translate, mosaic_mode=a.mosaic_mode, layer_drop=a.layer_drop, aug_iof=a.aug_iof,
                     cache=a.cache, fliplr=a.fliplr, flipud=a.flipud, rot90=not a.no_rot90)
    frozen = []
    if a.freeze:                                # fine-tuning on small data: keep the pretrained early layers fixed
        f = str(a.freeze).lower()
        if v2 or v6 or v9:
            frozen = {"backbone": [model.backbone], "encoder": [model.backbone, model.neck]}.get(f)
            if frozen is None:
                raise SystemExit("OpenOBB2 / OpenOBB6 --freeze must be 'backbone' or 'encoder' (backbone + neck)")
        elif f.isdigit():
            if not hasattr(model.backbone, "stages"):
                raise SystemExit("--freeze N needs the HGNetv2 backbone; use 'backbone' or 'encoder'")
            frozen = [model.backbone.stem] + list(model.backbone.stages[:int(f)])
        elif f == "backbone":
            frozen = [model.backbone]
        elif f == "encoder":
            frozen = [model.backbone] + [m for m in (getattr(model, "encoder", None), getattr(model, "neck", None),
                                                     model.lsk, getattr(model, "p2_proj", None)) if m is not None]
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
        from openobb.data.dota import repeat_factors
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
    # nominal batch size (usual one-stage trainer): accumulate gradients over round(nbs / batch) batches, weight decay
    # scaled by batch * accumulate / nbs so the decay per image seen stays that of nbs
    accumulate = max(round(a.nbs / a.batch), 1) if a.nbs > 0 else 1
    wd = a.wd * a.batch * accumulate / a.nbs if a.nbs > 0 else a.wd
    groups = param_groups(model, a.lr, a.backbone_mult, wd, split_bias=a.warmup_bias_lr > 0)
    try:                                # fused AdamW: one kernel per step instead of one per tensor
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999), fused=device.type == "cuda")
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema = ModelEMA(model, a.ema, a.ema_warmups)
    start_epoch, it, best_score, best_epoch = 0, 0, -1.0, -1
    recoveries, lr_scale = 0, 1.0               # divergence recovery: reload last.pt, halve the LR (<= 3 times)
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if "model" in ck:                       # eval-only checkpoints may carry the EMA weights alone
            model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch, it = ck["epoch"] + 1, ck["iter"]
        best_score, best_epoch = ck.get("best_score", -1.0), ck.get("best_epoch", -1)
        recoveries, lr_scale = ck.get("recoveries", 0), ck.get("lr_scale", 1.0)
        base_lrs = [b * lr_scale for b in base_lrs]
        sanitize_batchnorm(model, log=lambda m: log(m, console=True))       # exploded stats saved in last.pt
        sanitize_batchnorm(ema.module, log=lambda m: log(m))
        log(f"resumed from epoch {ck['epoch']} (iter {it})")
        if not a.eval_only:
            say(f"Resuming {out} from epoch {ck['epoch'] + 1}/{a.epochs}")
    # Compile AFTER the EMA deep copy: compiled forwards are bound methods of the live model, and a copy made
    # afterwards kept calling the live (training-mode) backbone/encoder at eval time (bug seen 2026-10-07:
    # identical train losses, ~7 points lower EMA eval).
    if a.compile and device.type == "cuda" and not a.eval_only:
        try:
            if v18:
                log("torch.compile: " + compile_training(model), console=True)
            else:
                model.backbone.forward = torch.compile(model.backbone.forward)
                if v2 or v6 or v9:
                    model.neck.forward = torch.compile(model.neck.forward, dynamic=False)
                else:
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
    log(f"OpenOBB1-{a.size} {n_par:.2f}M params | train patches {len(ds)} | {iters_per_epoch} it/epoch x {a.epochs} "
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
        say(f"OpenOBB {__import__('openobb').__version__} | torch {torch.__version__} | {gpu} | "
            f"AMP {('bf16' if scaler is None else 'fp16') if amp else 'off'} | "
            f"compile {'on' if a.compile and device.type == 'cuda' else 'off'}")
        if v2:
            say(f"model   OpenOBB2-{a.size} (dense segment-aware, P2-P5, NMS-free), {n_par:.2f}M params, "
                f"top {a.num_top} detections | init: {init_desc}")
        elif v4:
            say(f"model   OpenOBB4-{a.size} ({a.backbone} + {'LSK + ' if a.lsk else ''}hybrid encoder + DETR decoder "
                f"+ dense oriented branch, output {a.primary}), {n_par:.2f}M params, {a.queries} queries | "
                f"init: {init_desc}")
        elif v14:
            neck_desc = "own re-parameterised neck" if a.neck == "own" else "D-FINE COCO hybrid encoder (control)"
            if v15:
                neck_desc = (f"{neck_desc} at {a.img // 2} px + full-resolution {a.img} px detail stream + P2 grid")
            ref_desc = "3-stage cascade of edge refiners + fit-aware scores" if v16 else "edge refiner"
            say(f"model   OpenOBB{16 if v16 else (15 if v15 else 14)}-{a.size} (pretrained HGNetv2 trunk + {neck_desc} + heavy dual dense "
                f"oriented head, box={a.box}, {ref_desc} on {a.ref_k} candidates"
                f"{', off: ' + a.v14_off if a.v14_off else ''}), {n_par:.2f}M params, max {a.num_top} detections | "
                f"init: {init_desc}")
        elif v12 or v13:
            core = f"pretrained HGNetv2 + LSK + hybrid encoder{', off: ' + a.v7_off if a.v7_off else ''}"
            if v18:
                core = (f"pretrained HGNetv2 + own re-parameterised neck"
                        f"{' distilled from ' + a.teacher if a.teacher else ' (no teacher)'}"
                        f"{', off: ' + a.v7_off if a.v7_off else ''}")
            if v13:
                core = (f"{core} at {a.img // 2} px + full-resolution {a.img} px detail stream; dual dense oriented "
                        f"head P2-P5")
            else:
                core += "; dual dense oriented head P3-P5"
            if v17 and a.groups:
                core += f"; {max(map(int, a.groups.split(','))) + 1} contention-aware class groups ({a.groups})"
            if v20:
                core += "; edge refiner on the trunk's stride-4 map"
            if v21:
                core += "; angle corrected in the double-angle space"
            if v22:
                core += "; angle in (-45, 135) deg from 45 deg"
            if v23:
                core += "; rotational boundary snapping (long sides, line fit)"
            if v24 and "quality" not in (a.v24_off or ""):
                core += "; learnt localisation quality of the refined boxes (quality-guided NMS, calibrated)"
            if v17 and "field" not in (a.v7_off or ""):
                core += f"; boundary field (stride 2) + boundary snapping of {a.ref_k} candidates"
            if a.box == "direct":
                core += ", direct box regression (no DFL)"
            say(f"model   OpenOBB{24 if v24 else 23 if v23 else 22 if v22 else 21 if v21 else 20 if v20 else 19 if v19 else (18 if v18 else (17 if v17 else (13 if v13 else 12)))}-{a.size} ({core}: one-to-many for learning + one-to-one for "
                f"inference, NMS-free), {n_par:.2f}M params, max {a.num_top} detections | init: {init_desc}")
            if v13 and a.img < 1536:
                say(f"WARNING  OpenOBB13's core runs at {a.img // 2} px: drawings need the core at >= 1024 "
                    f"(openobb train imgsz=1280 -> input {2 * 1280})")
        elif v9:
            head9 = (f"OpenOBB8 head + relational refiner of {a.rel_k} candidates" if model.refiner is not None
                     else "OpenOBB7 dense head, no relational refiner")
            say(f"model   OpenOBB9-{a.size} (new re-parameterised core + {head9}), {n_par:.2f}M params, "
                f"max {a.num_top} detections | init: {init_desc}")
            if not a.weights:
                say("WARNING  OpenOBB9 starts from random weights: run `openobb pretrain model=openobb9x ...` first "
                    "and pass weights=<that file> (docs/OPENOBB9.md)")
        elif v11:
            say(f"model   OpenOBB11-{a.size} (pretrained HGNetv2 + LSK + hybrid encoder; dense oriented proposals "
                f"+ one-to-one relational set decoder over {a.rel_k} candidates, NMS-free), {n_par:.2f}M params | "
                f"init: {init_desc}")
        elif v10:
            off = ", ".join(f"off: {v}" for v in (a.v7_off, a.v8_off, a.v10_off) if v)
            say(f"model   OpenOBB10-{a.size} (pretrained HGNetv2 + LSK + hybrid encoder; dense oriented head P3-P5 "
                f"with {model.slots} slots per point + oriented relational refiner of {a.rel_k} candidates"
                f"{'; ' + off if off else ''}), {n_par:.2f}M params, max {a.num_top} detections | init: {init_desc}")
        elif v7 or v8:
            off = ", ".join(f"off: {v}" for v in (a.v7_off, a.v8_off if v8 else "") if v)
            say(f"model   OpenOBB{8 if v8 else 7}-{a.size} (pretrained HGNetv2 + LSK + hybrid encoder, repaired; dense "
                f"oriented head P3-P5{' + oriented relational refiner of ' + str(a.rel_k) + ' candidates' if v8 else ''}"
                f"{'; ' + off if off else ''}), {n_par:.2f}M params, max {a.num_top} detections | init: {init_desc}")
        elif v6:
            say(f"model   OpenOBB6-{a.size} (own backbone + neck + context-aware dense oriented head P3-P5"
                f"{', off: ' + a.v6_off if a.v6_off else ''}), "
                f"{n_par:.2f}M params, max {a.num_top} detections | init: {init_desc}")
        elif v5:
            parts = [a.backbone] + (["LSK"] if a.lsk else []) + ["hybrid encoder", "dense oriented head P3-P5"] +                 (["geometry-aware classes"] if a.geo_cls else []) +                 ([f"relation re-scoring of {a.rel_k} candidates"] if a.relate else [])
            say(f"model   OpenOBB5-{a.size} ({' + '.join(parts)}), {n_par:.2f}M params, max {a.num_top} detections | "
                f"init: {init_desc}")
        elif v3:
            say(f"model   OpenOBB3-{a.size} ({a.backbone + ' + ' if a.backbone != 'hgnet' else ''}pretrained encoder{' + LSK' if a.lsk else ''}, dense oriented head P3-P5), "
                f"{n_par:.2f}M params, max {a.num_top} detections | init: {init_desc}")
        else:
            say(f"model   OpenOBB1-{a.size}{' + ' + ' + '.join(extras) if extras else ''}, {n_par:.2f}M params, "
                f"{a.queries} queries | init: {init_desc}")
        mode = f"whole image, long side {a.img}" if a.fit else f"tiles {a.img}, scale {a.scale}"
        say(f"data    {len(ds)} train tiles, {n_val} val images, {len(classes)} classes | {mode}")
        say(f"train   {a.epochs} epochs x {iters_per_epoch} it, batch {a.batch}, AdamW lr {a.lr:g} -> "
            f"{a.lr * a.min_lr_ratio:g} ({a.schedule}{', per epoch' if a.epoch_lr else ''}), warmup {max(a.warmup, 0)} it"
            f"{f' (bias lr {a.warmup_bias_lr:g} -> lr)' if a.warmup_bias_lr > 0 else ''}"
            f"{f', nbs {a.nbs} (accumulate {accumulate}, weight decay {wd:.3g})' if a.nbs > 0 else ''} | "
            f"val every {a.eval_every} epoch(s) on {'all' if full_val else len(val_subset or [])} val images")
        aug = []
        if a.mosaic_p:
            aug.append(f"mosaic {a.mosaic_p:g} (off last {a.mosaic_off})")
        if a.scale_jitter:
            aug.append(f"zoom {1 - a.scale_jitter:g}-{1 + a.scale_jitter:g}")
        if a.translate:
            aug.append(f"translate {a.translate:g}")
        aug.append(f"flip lr {a.fliplr:g} / ud {a.flipud:g}")
        if not a.no_rot90:
            aug.append("rot90")
        if a.rotate_p:
            aug.append(f"rotate any angle {a.rotate_p:g}")
        aug.append(f"angle loss {a.angle_loss} ({a.angle_balance} weights)")
        aug.append("hsv " + "/".join(f"{v:g}" for v in a.hsv))
        say("augment " + ", ".join(aug) + (f" | freeze {a.freeze}" if a.freeze else ""))
        say(f"save    {out}  (best.pt, last.pt, results.csv/png, labels.jpg, val_pred.jpg, train_progress.log)")
        if start_epoch == 0:
            plot_labels(out, ds.items, classes, a.img)

    best_saver, last_saver = AsyncSaver(), AsyncSaver()

    def save_best(score, epoch):
        # compact deployable checkpoint: EMA weights + args + classes (no optimiser state); written in the background
        best_saver.save({"model": ema.module.state_dict(), "epoch": epoch, "args": dict(vars(a)),
                         "classes": list(classes), "val_mAP50_95": score, "version": __version__},
                        wdir / "best.tmp", wdir / "best.pt")
        log(f"best.pt <- epoch {epoch} (val mAP50:95 {score:.4f})")

    t_train = time.time()
    last_opt_step = it - 1                      # the first batch of a run / resume may step
    opt.zero_grad(set_to_none=True)
    warned = set()                              # repeated warnings go to the log file after the first one
    for epoch in itertools.count(start_epoch):
        warned_ema = "ema" in warned
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
        sums, n, bad = {}, 0, 0
        tick = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prof = None
        bar = EpochBar(epoch, a.epochs, iters_per_epoch, a.img, v2=v2, v3=v3, v5=v5) if not a.verbose else None
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
            if a.epoch_lr:                      # usual one-stage trainer: lf(epoch), linear warmup over iterations
                lfe = yolo_lf(epoch, a.epochs, a.min_lr_ratio, a.schedule)
                for g, b in zip(opt.param_groups, base_lrs):
                    if it <= a.warmup:          # biases fall from warmup_bias_lr, the rest rises from 0
                        g0 = a.warmup_bias_lr * g.get("lr_mult", 1.0) * lr_scale if g.get("name", "").endswith(
                            "bias") else 0.0
                        g["lr"] = float(np.interp(it, [0, max(a.warmup, 1)], [g0, b * lfe]))
                    else:
                        g["lr"] = b * lfe
            else:
                f = lr_factor(it, total_iters, a.warmup, a.flat, a.min_lr_ratio, a.schedule)
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * f
            if a.nbs > 0:                       # accumulation ramps from 1 to nbs / batch during warmup
                accumulate = (max(1, int(np.interp(it, [0, max(a.warmup, 1)], [1, a.nbs / a.batch]).round()))
                              if it <= a.warmup else max(round(a.nbs / a.batch), 1))
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
                bad += 1
                opt.zero_grad(set_to_none=True)
                last_opt_step = it
                it += 1
                tick = time.time()
                continue
            # gradients of `accumulate` batches are averaged before one optimizer step (accumulate = 1: every batch)
            if scaler is not None:
                scaler.scale(loss / accumulate).backward()
            else:
                (loss / accumulate).backward()
            if it - last_opt_step < accumulate:
                it += 1
                main = {k: v.detach() for k, v in losses.items() if "_aux" not in k and "_dn" not in k
                        and "_enc" not in k and "_pre" not in k}
                main["total"] = loss.detach()
                for k, v in main.items():
                    sums[k] = sums[k] + v if k in sums else v.clone()
                n += 1
                if bar is not None:
                    bar.update(bi + 1, {k: float(sums[k]) / n for k in sums if k in bar.keys},
                               sum(len(t["labels"]) for t in targets),
                               torch.cuda.memory_reserved() / 2**30 if device.type == "cuda" else None,
                               force=bi + 1 == iters_per_epoch)
                tick = time.time()
                continue
            last_opt_step = it
            if scaler is not None:
                scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else 1e9)
            if not torch.isfinite(gn):          # never let one bad batch poison the weights
                if scaler is not None:
                    scaler.update()             # fp16 overflow: lower the loss scale, skip the step
                else:
                    log(f"non-finite grad norm at iter {it}; step skipped", console=True)
                    bad += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                tick = time.time()
                continue
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
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
                bar.update(bi + 1, {k: float(sums[k]) / n for k in sums if k in bar.keys},
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
        if bad > max(2, iters_per_epoch // 10) or not batchnorm_finite(model):
            # diverging (like the usual trainer's NaN recovery): back to the last good epoch with half the LR
            if recoveries >= 3 or not last.exists():
                raise SystemExit(f"training diverged ({bad} non-finite steps in epoch {epoch + 1}) and cannot recover"
                                 f"{' after 3 retries' if recoveries >= 3 else ''}: lower the learning rate (lr0)")
            last_saver.wait()
            ck = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            it = ck["iter"]
            last_opt_step = it - 1
            opt.zero_grad(set_to_none=True)
            recoveries += 1
            lr_scale *= 0.5
            base_lrs = [b * 0.5 for b in base_lrs]
            sanitize_batchnorm(model, log=lambda m: log(m))
            sanitize_batchnorm(ema.module, log=lambda m: log(m))
            say(f"WARNING: training diverged in epoch {epoch + 1} ({bad} non-finite steps); reloaded last.pt "
                f"(epoch {ck['epoch'] + 1}) and halved the learning rate (now x{lr_scale:g}), recovery {recoveries}/3")
            continue
        dt = time.time() - t_ep
        rec = {"epoch": epoch, "iter": it, "lr": opt.param_groups[-1]["lr"], "epoch_s": round(dt, 1),
               "img_s": round(n * a.batch / max(dt, 1e-9), 1), "crit_s_per_it": round(t_loss / max(n, 1), 3),
               "data_s_per_it": round(t_data / max(n, 1), 3),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else None}
        rec.update({k: round(float(v) / max(n, 1), 4) for k, v in sums.items()})
        res = None
        if val_subset and ((epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs) and (epoch + 1 < a.epochs or full_val):
            t_val = time.time()
            ema.repair(model, log=lambda m: log(m, console=not warned_ema) or warned.add("ema"))
            vm_ep = fused_copy(ema.module, device)
            res, _ = eval_dota(vm_ep, a.data, device, val_subset, batch=a.batch, workers=a.workers,
                               num_top=a.num_top, img_size=a.img, log=log, context=a.context, vectors=a.vectors,
                               post=a.post, merge_iou=a.merge_iou, fusion=a.primary != "dec",
                               variants=["dec", a.primary], primary=a.primary, metric=a.metric,
                               dota=a.metric != "yolo", extras=False)
            del vm_ep
            if not a.verbose:
                val_rows(res, time.time() - t_val)
            rec["sub_mAP50"] = round(res["mAP50"], 4)
            rec["sub_mAP50_95"] = round(res["mAP50_95"], 4)
            if full_val and res["mAP50_95"] > best_score:
                best_score, best_epoch = res["mAP50_95"], epoch
                save_best(best_score, epoch)
        last_saver.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "iter": it, "args": vars(a), "classes": list(classes), "best_score": best_score,
                    "best_epoch": best_epoch, "recoveries": recoveries, "lr_scale": lr_scale,
                    "version": __version__}, wdir / "last.tmp", last)
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

    best_saver.wait()                           # the files below are read back / rewritten
    last_saver.wait()
    if a.skip_final_eval:
        if not (wdir / "best.pt").exists():
            save_best(-1.0, a.epochs - 1)
            best_saver.wait()
        return
    weights = "ema_last"
    if full_val and best_score >= 0 and (wdir / "best.pt").exists() and not a.eval_only:
        ema.module.load_state_dict(torch.load(wdir / "best.pt", map_location="cpu", weights_only=False)["model"])
        weights = "best"
        say(f"\nTraining finished. Validating best.pt (epoch {best_epoch + 1}) on all val images...")
    else:
        say(f"\nTraining finished. Validating the final EMA model on all val images...")
    vm = copy.deepcopy(ema.module).eval()            # validated like YOLO's best.pt: BatchNorm folded (same outputs)
    if hasattr(vm, "fuse"):
        vm.fuse()
    else:
        from openobb.models.openobb7 import fuse_model
        fuse_model(vm)
    from openobb.predict import fast_layout
    vm = fast_layout(vm, device)                     # channels_last on a GPU (after folding BN), as at inference (F97)

    def final_eval(nms):
        apply_nms(vm, nms)
        merge = merge_iou_for(nms, True) if a.fit else a.merge_iou
        return eval_dota(vm, a.data, device, None, batch=a.batch, workers=a.workers, num_top=a.num_top,
                         img_size=a.img, log=log, fusion=a.dense or a.vec_ground > 0,
                         variants=["dec", "dense", "union", "snap", "snapg"],
                         save_preds_to=(out / "val_preds.npz") if a.dense else None, context=a.context,
                         vectors=a.vectors, post=a.post, merge_iou=merge, primary=a.primary, ultra=True,
                         metric=a.metric)

    if parse_nms(a.nms_iou) is None and hasattr(vm, "nms_iou"):
        # NMS is a parameter of the trained model, not a constant: try the settings on val, keep the best fitness
        # (0.1 mAP50 + 0.9 mAP50-95, as YOLO's best.pt) and save it as the inference default (args["nms_iou"])
        # dual-head models also choose the inference head (one-to-one / one-to-many + NMS), saved as args["infer_head"]
        tried = []
        for h, c in postprocess_options(vm):
            apply_head(vm, h)
            r, d = final_eval(c)
            u = r.get("ultra") or r
            tried.append((0.1 * u["mAP50"] + 0.9 * u["mAP50_95"], dup_rate(u), (h, c), r, d))
        # best fitness, but within DUP_TOL of it the setting with the fewest duplicate boxes (F69)
        fit_best, _, (a.infer_head, a.nms_iou), res, dets = choose_postprocess(tried)
        apply_head(vm, a.infer_head)
        apply_nms(vm, a.nms_iou)
        lab = lambda hc: f"{hc[0]} {hc[1]}" if hc[0] else f"{hc[1]}"  # noqa: E731
        dl = lambda q: "" if q is None else f", dup {100 * q:.1f}%"  # noqa: E731
        from openobb.predict import _long_tilt
        tl = lambda r: "" if _long_tilt(r) is None else f", tilt {100 * _long_tilt(r):.1f}%"  # noqa: E731
        say("NMS calibrated on val (saved in best.pt, the inference default; fitness, then fewer duplicates within "
            f"{DUP_TOL:g}, then fewer tilted long boxes): " + " | ".join(f"{lab(c)}: fitness {f:.4f}{dl(q)}{tl(r)}"
                                                                      for f, q, c, r, _ in tried)
            + f" -> {lab((a.infer_head, a.nms_iou))}")
    else:
        res, dets = final_eval(a.nms_iou)
        u = res.get("ultra") or res
        fit_best = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
    a.q_beta = a.q_alpha = a.snap_skip = a.q_outside = None
    if hasattr(vm, "q_beta") and not a.eval_only:
        # OpenOBB24: how the learnt localisation quality is used (quality-guided NMS / score) and which class groups skip
        # the boundary snapping, chosen on val after the NMS like it; (0, 0, none) = OpenOBB23 (docs/OPENOBB24.md)
        from openobb.predict import apply_quality, outside_options, quality_options, snap_skip_options
        u0 = res.get("ultra") or res
        best = (0.1 * u0["mAP50"] + 0.9 * u0["mAP50_95"], (0.0, 0.0), (), res, dets)
        rows = []
        for qb, qa in quality_options(vm)[1:]:
            apply_quality(vm, qb, qa, ())
            r, d = final_eval(a.nms_iou)
            u = r.get("ultra") or r
            f = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
            rows.append(f"q^{qb:g} in NMS{f', score^{1 - qa:g} q^{qa:g}' if qa else ''}: {f:.4f}")
            if f > best[0]:
                best = (f, (qb, qa), (), r, d)
        for sk in snap_skip_options(vm)[1:]:
            apply_quality(vm, *best[1], sk)
            r, d = final_eval(a.nms_iou)
            u = r.get("ultra") or r
            f = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
            rows.append(f"no snapping for group {','.join(map(str, sk))}: {f:.4f}")
            if f > best[0]:
                best = (f, best[1], sk, r, d)
        a.q_outside = "min"
        for qo in outside_options(vm)[1:]:
            apply_quality(vm, *best[1], best[2], qo)
            r, d = final_eval(a.nms_iou)
            u = r.get("ultra") or r
            f = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
            rows.append(f"unrefined candidates {qo}: {f:.4f}")
            if f > best[0]:
                best, a.q_outside = (f, best[1], best[2], r, d), qo
        fit_best, (a.q_beta, a.q_alpha), sk, res, dets = best
        a.snap_skip = ",".join(map(str, sk))
        apply_quality(vm, a.q_beta, a.q_alpha, sk, a.q_outside)
        say(f"Localisation quality calibrated on val (saved in best.pt; fitness, base {0.1 * u0['mAP50'] + 0.9 * u0['mAP50_95']:.4f}): "
            + " | ".join(rows) + f" -> q^{a.q_beta:g} in NMS, score^{1 - a.q_alpha:g} q^{a.q_alpha:g}, "
            f"no snapping for groups [{a.snap_skip}], unrefined candidates {a.q_outside}")
    a.infer_k = None
    ref = getattr(vm, "refiner", None)
    if ref is not None and not a.eval_only:
        # candidate budget of the relational stage at inference (its cost grows with K): the smallest K whose val
        # fitness stays within 0.002 of the trained K is saved in best.pt (args["infer_k"]) and used by predict / val
        k0, best = ref.k, (ref.k, res, dets)
        for k in (k0 // 2, k0 // 4):
            if k < 64:
                break
            ref.k = k
            r, d = final_eval(a.nms_iou)
            u = r.get("ultra") or r
            f = 0.1 * u["mAP50"] + 0.9 * u["mAP50_95"]
            say(f"inference candidates K={k}: fitness {f:.4f} (K={k0}: {fit_best:.4f})")
            if f < fit_best - 0.002:
                break
            best = (k, r, d)
        ref.k, res, dets = best
        a.infer_k = ref.k if ref.k != k0 else None
        if a.infer_k:
            say(f"relational stage: {ref.k} candidates at inference instead of {k0} (saved in best.pt)")
    if not a.eval_only and not full_val:
        save_best(res["mAP50_95"], a.epochs - 1)
    table = summary_table(res, classes)
    log("\n" + table)
    (out / "eval_val.txt").write_text(table + "\n")
    (out / "eval_val.json").write_text(json.dumps(res, indent=1))
    write_task1(dets, out / "val_task1", classes)
    if a.eval_queries and a.eval_queries != a.queries and not dense_arch:   # extra eval, main numbers stay at --queries
        ema.module.decoder.num_queries = a.eval_queries
        rq, _ = eval_dota(ema.module, a.data, device, None, batch=a.batch, workers=a.workers, num_top=a.eval_queries,
                          img_size=a.img, log=log, context=a.context, vectors=a.vectors, post=a.post, merge_iou=a.merge_iou,
                          fusion=a.primary != "dec", variants=["dec", a.primary], primary=a.primary, ultra=True,
                          metric=a.metric)
        (out / f"eval_val_q{a.eval_queries}.txt").write_text(summary_table(rq, classes) + "\n")
        (out / f"eval_val_q{a.eval_queries}.json").write_text(json.dumps(rq, indent=1))
        jlog({"final_q": a.eval_queries, "mAP50": rq["mAP50"], "mAP50_95": rq["mAP50_95"]})
        ema.module.decoder.num_queries = a.queries
    final = rq if a.eval_queries and a.eval_queries != a.queries and not dense_arch else res
    say(summary_table(final, classes))
    if final.get("ultra", {}).get("confusion"):
        from openobb.eval.ultra import top_confusions
        (out / "confusion_matrix.json").write_text(json.dumps(final["ultra"]["confusion"]))
        lines = top_confusions(final["ultra"]["confusion"])
        if lines:
            say("Most frequent errors (true -> predicted, conf 0.25, IoU 0.5; background = missed / false box): "
                + " | ".join(lines))
        from openobb.eval.ultra import duplicate_line
        if duplicate_line(final["ultra"]["confusion"]):
            say(duplicate_line(final["ultra"]["confusion"]))
        from openobb.eval.ultra import geometry_line
        if geometry_line(final["ultra"].get("geometry")):
            say(geometry_line(final["ultra"]["geometry"]))
    if final.get("metric") != "yolo" and final.get("ultra"):
        u = final["ultra"]
        say(f"Ultralytics-style metric (ProbIoU matching, 101-point AP; compare with YOLO's printed mAP): "
            f"mAP50 {u['mAP50']:.3f}  mAP50-95 {u['mAP50_95']:.3f}")
    if not a.eval_only and (wdir / "best.pt").exists():  # per-class confidence (best F2) for predict conf=auto
        ck = torch.load(wdir / "best.pt", map_location="cpu", weights_only=False)
        ck["conf_thr"] = {c: round(final["classes"][c].get("conf_f2", 0.25), 3) for c in classes}
        ck["args"]["nms_iou"] = a.nms_iou
        if getattr(a, "infer_head", None):
            ck["args"]["infer_head"] = a.infer_head
        if a.infer_k:
            ck["args"]["infer_k"] = a.infer_k
        if a.q_beta is not None:
            ck["args"].update(q_beta=a.q_beta, q_alpha=a.q_alpha, snap_skip=a.snap_skip, q_outside=a.q_outside)
        ck["conf_f1"] = final.get("conf_f1")
        torch.save(ck, wdir / "best.tmp")
        os.replace(wdir / "best.tmp", wdir / "best.pt")
    try:
        plot_val_predictions(out, ema.module, a.data, device, a.img, classes, num_top=a.num_top)
    except Exception as e:  # noqa: BLE001 - a report must never fail the run
        log(f"val_pred.jpg skipped: {e}")
    lat = None
    sp = res.get("speed")
    if sp and sp.get("images"):
        from openobb.speed import speed_line
        lat = round(sp["inference"] + sp["postprocess"], 2)
        say(speed_line(sp) + f" ({sp['images']} val images after a warm-up batch, fused, bf16, "
            f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    say(f"Results saved to {out}\n  {wdir / 'best.pt'} (deploy / fine-tune), {wdir / 'last.pt'} (resume=True), "
        f"results.csv/png, val_pred.jpg")
    jlog({"final": True, "split": "val", "weights": weights, "mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
          "per_class_AP50": {c: round(r["AP50"], 4) for c, r in res["classes"].items()},
          "latency_ms_pt_fp16_bs1": lat, "fusion": {k: round(v["mAP50"], 4) for k, v in res.get("fusion", {}).items()},
          "ultra": {k: round(final["ultra"][k], 4) for k in ("mAP50", "mAP50_95")} if final.get("ultra") else None})
    if not a.eval_only and last.exists():   # finished: drop the optimiser state from last.pt (like best.pt)
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if "opt" in ck:
            q_args = {} if a.q_beta is None else dict(q_beta=a.q_beta, q_alpha=a.q_alpha, snap_skip=a.snap_skip,
                                                         q_outside=a.q_outside)
            torch.save({"model": ck["ema"]["module"], "epoch": ck["epoch"], "args": dict(ck["args"], nms_iou=a.nms_iou,
                                                                       **({"infer_head": a.infer_head}
                                                                          if getattr(a, "infer_head", None) else {}),
                                                                       **q_args),
                        "classes": ck["classes"], "best_score": ck.get("best_score"),
                        "version": ck.get("version", __version__)}, wdir / "last.tmp")
            os.replace(wdir / "last.tmp", last)
    (out / "train_done").write_text("ok")


if __name__ == "__main__":
    main()
