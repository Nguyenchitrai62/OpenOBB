"""Training reports written next to the checkpoints: results.csv / results.png (per-epoch losses and val metrics),
labels.jpg (class counts and box sizes of the training data) and val_pred.jpg (predictions on a few val tiles).
Plots need matplotlib; without it only the CSV is written."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

COLUMNS = ["epoch", "time_s", "lr", "cls_loss", "box_loss", "kld_loss", "angle_loss", "dfl_loss",
           "P", "R", "mAP50", "mAP50-95"]
LOSS_KEYS = {"cls_loss": "loss_mal", "box_loss": "loss_bbox", "kld_loss": "loss_kld", "angle_loss": "loss_angle",
             "dfl_loss": "loss_fgl"}
PALETTE = [(75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145), (240, 240, 70),
           (230, 50, 240), (60, 245, 210), (212, 190, 250), (128, 128, 0)]         # BGR


def append_results(out, rec, res=None):
    """One row per epoch; val columns stay empty on epochs without validation."""
    row = {"epoch": rec["epoch"] + 1, "time_s": rec.get("epoch_s"), "lr": f"{rec.get('lr', 0):.3g}"}
    row.update({k: round(rec[v], 5) for k, v in LOSS_KEYS.items() if v in rec})
    if res is not None:
        row.update({"P": round(res["P"], 4), "R": round(res["R"], 4), "mAP50": round(res["mAP50"], 4),
                    "mAP50-95": round(res["mAP50_95"], 4)})
    p = Path(out) / "results.csv"
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def plot_results(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = Path(out) / "results.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    keys = ["cls_loss", "box_loss", "kld_loss", "angle_loss", "dfl_loss", "P", "R", "mAP50", "mAP50-95", "lr"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), tight_layout=True)
    for ax, k in zip(axes.ravel(), keys):
        pts = [(float(r["epoch"]), float(r[k])) for r in rows if r.get(k) not in (None, "")]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, marker="." if len(pts) < 40 else None, linewidth=1.5)
        ax.set_title(k)
        ax.grid(alpha=0.3)
    fig.savefig(Path(out) / "results.png", dpi=110)
    plt.close(fig)


def plot_labels(out, items, classes, size):
    """Class histogram + normalised box width/height of the training tiles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from vrdet.data.dota import polys_to_obb
    cls, wh = [], []
    for m in items:
        if m["objs"]:
            o = np.asarray(m["objs"], dtype=np.float32)
            cls.append(o[:, 0].astype(int))
            wh.append(polys_to_obb(o[:, 3:11])[:, 2:4] / size)
    if not cls:
        return
    cls, wh = np.concatenate(cls), np.concatenate(wh)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), tight_layout=True)
    counts = np.bincount(cls, minlength=len(classes))
    ax[0].bar(range(len(classes)), counts, color="#4c78a8")
    ax[0].set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax[0].set_title(f"instances per class ({len(cls)} in {len(items)} tiles)")
    long_, short = wh.max(1), wh.min(1)
    ax[1].scatter(long_, short, s=2, alpha=0.3)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("long side / tile")
    ax[1].set_ylabel("short side / tile")
    ax[1].set_title("box sizes")
    fig.savefig(Path(out) / "labels.jpg", dpi=110)
    plt.close(fig)


def plot_train_batch(path, imgs, targets, classes, n=4):
    """2x2 preview of an augmented training batch with its target boxes (checks mosaic / zoom / clip)."""
    from vrdet.ops.obb import obb2poly
    tiles = []
    for img, t in list(zip(imgs, targets))[:n]:
        S = img.shape[-1]
        vis = np.ascontiguousarray(img.permute(1, 2, 0).numpy()[..., ::-1])
        b = t["boxes"].numpy().copy()
        b[:, :4] *= S
        for c, p in zip(t["labels"].tolist(), obb2poly(b) if len(b) else []):
            pts = p.reshape(4, 2).round().astype(np.int32)
            color = PALETTE[int(c) % len(PALETTE)]
            cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
            cv2.putText(vis, classes[int(c)], tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (640, 640), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    while len(tiles) < 4:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]), [cv2.IMWRITE_JPEG_QUALITY, 90])


@torch.no_grad()
def plot_val_predictions(out, model, data_root, device, img_size, classes, num_top=300, n=4, conf=0.3):
    """2x2 grid of val tiles: ground truth in thin green, predictions (score >= conf) in class colours."""
    from vrdet.data.dota import DotaPatches
    from vrdet.models.vrdet import postprocess
    from vrdet.ops.obb import obb2poly
    ds = DotaPatches(data_root, "val", augment=False)
    idx = [i for i, m in enumerate(ds.items) if m["objs"]][:n]
    if not idx:
        return
    model.eval()
    tiles = []
    for i in idx:
        img, tgt = ds[i]
        x = img.float().div(255)[None].to(device)
        s, l, b = (t[0].cpu().numpy() for t in postprocess(model(x), num_top, img_size))
        vis = np.ascontiguousarray(img.permute(1, 2, 0).numpy()[..., ::-1])
        gt = tgt["boxes"].numpy().copy()
        gt[:, :4] *= img_size
        for p in obb2poly(gt):
            cv2.polylines(vis, [p.reshape(4, 2).round().astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
        for sc, c, p in zip(s, l, obb2poly(b)):
            if sc >= conf:
                pts = p.reshape(4, 2).round().astype(np.int32)
                color = PALETTE[int(c) % len(PALETTE)]
                cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
                cv2.putText(vis, f"{classes[int(c)]} {sc:.2f}", tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (640, 640), interpolation=cv2.INTER_AREA))
    while len(tiles) < 4:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    cv2.imwrite(str(Path(out) / "val_pred.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
