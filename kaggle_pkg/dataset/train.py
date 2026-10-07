"""Training loop (Kaggle + local compatible). MIT. Usage:
local smoke:  py -3.12 train.py --data subset/data.yaml --variant nano --imgsz 512 --epochs 2 --batch 2
kaggle full:  python train.py --data /kaggle/input/.../data.yaml --variant base --imgsz 1024 --epochs 50 --batch 16
"""
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from model import build_model
from dataset import OBBDataset, collate_fn
from loss import DetectionLoss


def parse_yaml(path):
    d = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="subset/data.yaml")
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "base", "large"])
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--ablate", default=None, choices=[None, "noreason", "nogcam"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = parse_yaml(args.data)
    tr_dir = cfg["train"].strip()
    va_dir = cfg["val"].strip()
    # data.yaml stores absolute local paths; on kaggle allow override via env
    tr_img, tr_lbl = tr_dir, os.path.join(os.path.dirname(tr_dir.rstrip("/\\")), "..", "labels") if False else None
    # convention: <split>/images + <split>/labels
    def split_dirs(img_dir):
        base = os.path.dirname(img_dir.rstrip("/\\"))
        return img_dir, os.path.join(base, "labels")
    tri, trl = split_dirs(tr_dir)
    vai, val = split_dirs(va_dir)
    print(f"train {tri} | {trl}\nvalid {vai} | {val}\ndevice {args.device}")
    train_ds = OBBDataset(tri, trl, imgsz=args.imgsz, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_fn)
    print(f"n_train={len(train_ds)} batches={len(train_dl)}")
    model = build_model(args.variant, ablate=args.ablate).to(args.device)
    print(f"variant={args.variant} ablate={args.ablate} params={model.param_count()/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = DetectionLoss(nc=7)
    scaler = torch.amp.GradScaler("cuda") if args.device.startswith("cuda") else None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = {"loss": 0, "cls": 0, "box": 0, "dfl": 0, "ang": 0, "topo": 0}
        n = 0
        for imgs, tgts, _ in train_dl:
            imgs = imgs.to(args.device)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
                    loss, d = crit(preds, tgts, args.imgsz)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                preds = model(imgs)
                loss, d = crit(preds, tgts, args.imgsz)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            for k in run:
                run[k] += d.get(k, 0) if k != "loss" else float(loss)
            n += 1
        sched.step()
        msg = f"epoch {ep}/{args.epochs} " + " ".join(f"{k}={v/max(1,n):.4f}" for k, v in run.items()) + f" lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.1f}s"
        print(msg, flush=True)
        torch.save({"variant": args.variant, "ablate": args.ablate, "state": model.state_dict(), "epoch": ep}, os.path.join(args.out, "last.pt"))
    torch.save({"variant": args.variant, "ablate": args.ablate, "state": model.state_dict()}, os.path.join(args.out, "best.pt"))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
