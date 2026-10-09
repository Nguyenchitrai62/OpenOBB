"""Ultralytics YOLO-OBB baseline (AGPL: run on Colab ONLY to measure it, never imported by openobb/).

Trains on our DOTA patch split, then predicts the val patches, merges them back to full
images and scores them with OUR DOTA-devkit-protocol evaluator, so the number is directly
comparable with VRDet runs. Resumable: re-running continues from {out}/train/weights/last.pt.

python -u colab/baselines/yolo_obb.py --data /content/datasets/dota1_1024 --out OUT \
    --model yolo26s-obb.yaml --weights yolo26s.pt --epochs 24 --imgsz 1024 --batch 32
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.dota import dataset_classes  # noqa: E402
from openobb.eval.dota import evaluate, format_table, load_gt_dir, merge_patches, write_task1  # noqa: E402


def jlog(out, rec):
    with open(Path(out) / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def make_yaml(data, out, n_val_mon):
    data = Path(data)
    vals = sorted(p for p in (data / "images" / "val").iterdir() if p.suffix in (".jpg", ".png"))
    random.Random(0).shuffle(vals)
    mon = Path(out) / "val_monitor.txt"            # small fixed subset: cheap per-epoch curve
    mon.write_text("\n".join(str(p) for p in sorted(vals[:n_val_mon])) + "\n")
    y = Path(out) / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(dataset_classes(data)))
    y.write_text(f"path: {data}\ntrain: images/train\nval: {mon}\nnames:\n{names}\n")
    return y


def train(a):
    from ultralytics import YOLO
    out = Path(a.out)
    last = out / "train" / "weights" / "last.pt"
    done = out / "train_done"
    if done.exists():
        return
    if last.exists():
        print(f"resuming from {last}", flush=True)
        model = YOLO(str(last))
        kw = dict(resume=True)
    else:
        model = YOLO(a.model)
        if a.weights:
            model = model.load(a.weights)
        kw = dict(data=str(make_yaml(a.data, out, a.val_monitor)), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                  workers=a.workers, project=str(out), name="train", exist_ok=True, seed=0, deterministic=False,
                  plots=False, save_period=-1, patience=0, cache=a.cache)
    t0 = time.time()

    def on_epoch_end(trainer):
        rec = {"epoch": int(trainer.epoch), "elapsed_h": round((time.time() - t0) / 3600, 3),
               "lr": float(trainer.optimizer.param_groups[0]["lr"])}
        try:
            items = trainer.label_loss_items(trainer.tloss, prefix="train")
            rec["train"] = {k.split("/")[-1]: round(float(v), 4) for k, v in items.items()}
        except Exception:  # noqa: BLE001
            pass
        for k, v in (trainer.metrics or {}).items():
            try:
                if k.startswith("metrics/") or k.startswith("val/"):
                    rec[k.replace("metrics/", "patch_").replace("/", "_")] = round(float(v), 4)
            except (TypeError, ValueError):
                pass
        jlog(out, rec)

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(**kw)
    done.write_text("ok")


def evaluate_val(a):
    from ultralytics import YOLO
    out = Path(a.out)
    w = out / "train" / "weights" / ("last.pt" if a.use_last else "best.pt")
    model = YOLO(str(w))
    data = Path(a.data)
    vals = sorted(p for p in (data / "images" / "val").iterdir() if p.suffix in (".jpg", ".png"))
    classes = dataset_classes(data)
    t0 = time.time()
    pd = []
    for i in range(0, len(vals), a.pred_batch):
        chunk = [str(p) for p in vals[i:i + a.pred_batch]]
        for r in model.predict(chunk, imgsz=a.imgsz, conf=a.conf, max_det=a.max_det, verbose=False, half=True):
            name = Path(r.path).stem
            if r.obb is None or len(r.obb) == 0:
                continue
            polys = r.obb.xyxyxyxy.reshape(-1, 8).cpu().numpy()
            for p, s, c in zip(polys, r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy().astype(int)):
                pd.append((name, int(c), float(s), p))
    t_pred = time.time() - t0
    merged = merge_patches(pd, iou_thr=0.1)
    dets = {classes[c]: v for c, v in merged.items()}
    write_task1(dets, out / "val_task1", classes)
    res = evaluate(dets, load_gt_dir(data / "gt" / "val"), classes)
    table = format_table(res, classes)
    print(table, flush=True)
    (out / "eval_val.txt").write_text(table + "\n")
    (out / "eval_val.json").write_text(json.dumps(res, indent=1))
    # latency probe: batch 1, fp16, 1024, PyTorch (not TensorRT) on this GPU
    import torch
    lat = None
    try:
        x = [str(vals[0])] * 1
        for _ in range(10):
            model.predict(x, imgsz=a.imgsz, half=True, verbose=False)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            model.predict(x, imgsz=a.imgsz, half=True, verbose=False)
        torch.cuda.synchronize()
        lat = (time.time() - t) / 50 * 1000
    except Exception as e:  # noqa: BLE001
        print("latency probe failed", e)
    jlog(out, {"final": True, "split": "val", "weights": w.name, "mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
               "per_class_AP50": {c: round(r["AP50"], 4) for c, r in res["classes"].items()},
               "n_patch_dets": len(pd), "pred_s": round(t_pred, 1), "latency_ms_e2e_pt": lat})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26s-obb.yaml")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--cache", default=False)
    ap.add_argument("--val-monitor", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--pred-batch", type=int, default=32)
    ap.add_argument("--use-last", action="store_true", default=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    train(a)
    if not (Path(a.out) / "eval_val.json").exists():
        evaluate_val(a)


if __name__ == "__main__":
    main()
