"""Infer + visualize. MIT. Usage: py infer_run.py --ckpt runs/micro_nano/best.pt --img valid/images/xxx.png --out runs/vis"""
import argparse
import os
import cv2
import torch
from model import build_model
from infer import predict

NAMES = ["wall", "window", "door", "slide_door", "double_door", "opening", "junction"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]


def run_one(model, img_path, out_path, conf=0.2, imgsz=768):
    bgr = cv2.imread(img_path)
    H0, W0 = bgr.shape[:2]
    inp = cv2.resize(bgr, (imgsz, imgsz))
    t = torch.from_numpy(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = t.to(next(model.parameters()).device)
    boxes, scores, clses, rects = predict(model, t, conf_thr=conf)
    vis = inp.copy()
    for b, s, c in zip(boxes, scores, clses):
        x1, y1, x2, y2 = b.astype(int)
        col = COLORS[int(c) % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f"{NAMES[int(c)]} {s:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.imwrite(out_path, vis)
    return len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/micro_nano/best.pt")
    ap.add_argument("--imgdir", default="micro/valid/images")
    ap.add_argument("--out", default="runs/vis_micro")
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--maxn", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck.get("variant", args.variant), ablate=ck.get("ablate"))
    model.load_state_dict(ck["state"])
    model.eval()
    fs = sorted([f for f in os.listdir(args.imgdir) if f.endswith((".png", ".jpg"))])[: args.maxn]
    for f in fs:
        n = run_one(model, os.path.join(args.imgdir, f), os.path.join(args.out, f), args.conf, args.imgsz)
        print(f, "->", n, "boxes")


if __name__ == "__main__":
    main()
