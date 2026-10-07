"""Eval HBB-mAP50 proxy cho CAD-VLMDet (OBB ignored -> HBB, noted limitation). MIT.
Usage: py eval.py --ckpt runs/vlmdet_small/best.pt --imgdir valid/images --lbldir valid/labels --variant small
"""
import argparse, os
import cv2
import numpy as np
import torch
from model import build_model
from infer import predict


def load_gt(lbldir, stem, W=1.0, H=1.0):
    p = os.path.join(lbldir, stem + ".txt")
    boxes, clses = [], []
    if os.path.exists(p):
        for line in open(p):
            s = line.strip().split()
            if len(s) < 9:
                continue
            c = int(float(s[0]))
            pts = np.array(list(map(float, s[1:9]))).reshape(4, 2)
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            boxes.append([x1, y1, x2, y2]); clses.append(c)
    return np.array(boxes), np.array(clses)


def iou(a, B):
    lt = np.maximum(a[:2], B[:, :2]); rb = np.minimum(a[2:], B[:, 2:])
    wh = np.maximum(rb - lt, 0); inter = wh[:, 0] * wh[:, 1]
    aa = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    bb = np.maximum(B[:, 2]-B[:, 0], 0) * np.maximum(B[:, 3]-B[:, 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def ap50(pred_b, pred_s, gt_b):
    if len(gt_b) == 0:
        return 1.0 if len(pred_b) == 0 else 0.0
    if len(pred_b) == 0:
        return 0.0
    order = np.argsort(-pred_s); tp = np.zeros(len(order)); fp = np.zeros(len(order))
    matched = np.zeros(len(gt_b), bool)
    for i, k in enumerate(order):
        ov = iou(pred_b[k], gt_b)
        j = int(np.argmax(ov))
        if ov[j] >= 0.5 and not matched[j]:
            tp[i] = 1; matched[j] = True
        else:
            fp[i] = 1
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / max(1, len(gt_b)); prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--lbldir", required=True)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--hbb_nms", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck.get("variant", args.variant), ablate=ck.get("ablate"))
    model.load_state_dict(ck["state"]); model.eval()
    NAMES = ["wall", "window", "door", "slide_door", "double_door", "opening", "junction"]
    preds = {c: ([], [], []) for c in range(7)}  # per class: list of (img, boxes, scores)
    gts = {c: {} for c in range(7)}
    fs = sorted([f for f in os.listdir(args.imgdir) if f.endswith((".png", ".jpg"))])
    import time; t0 = time.time(); nimg = 0
    with torch.no_grad():
        for f in fs:
            stem = os.path.splitext(f)[0]
            bgr = cv2.imread(os.path.join(args.imgdir, f))
            H0, W0 = bgr.shape[:2]
            inp = cv2.resize(bgr, (args.imgsz, args.imgsz))
            t = torch.from_numpy(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            boxes, scores, clses, _ = predict(model, t, conf_thr=args.conf, hbb_nms=args.hbb_nms)
            nimg += 1
            # rescale to 0..1
            if len(boxes):
                boxes = boxes / args.imgsz
            gb, gc = load_gt(args.lbldir, stem)
            for c in range(7):
                m = clses == c if len(clses) else np.zeros(0, bool)
                preds[c][0].append(stem); preds[c][1].append(boxes[m] if len(boxes) else np.zeros((0, 4))); preds[c][2].append(scores[m] if len(scores) else np.zeros(0))
                gm = gc == c if len(gc) else np.zeros(0, bool)
                gts[c][stem] = gb[gm] if len(gb) else np.zeros((0, 4))
    dt = time.time() - t0
    print(f"infer {nimg} imgs in {dt:.1f}s = {dt/max(1,nimg)*1000:.1f} ms/img")
    aps = []
    for c in range(7):
        # chi tinh tren anh CO GT (anh khong GT thi skip, nhu COCO;
        # truoc day tra 1.0 lam slide/double-door ao len 0.345/0.390 - fixed exp4)
        ap_list = []
        npred = ngt = 0
        for img_i, stem in enumerate(preds[c][0]):
            if len(gts[c][stem]) == 0:
                continue
            ap_list.append(ap50(preds[c][1][img_i], preds[c][2][img_i], gts[c][stem]))
            npred += len(preds[c][1][img_i]); ngt += len(gts[c][stem])
        m = float(np.mean(ap_list)) if ap_list else 0.0
        aps.append(m)
        print(f"{NAMES[c]:12s} AP50(HBB-proxy)={m:.3f} npred={npred} ngt={ngt}")
    print(f"mean AP50(HBB-proxy) over 7 classes = {np.mean(aps):.3f}")


if __name__ == "__main__":
    main()
