"""Same-session GPU speed: YOLO (Ultralytics, research baseline only) vs OpenOBB checkpoints on the same input.
Two numbers per model, median over the timed runs after warm-up:
  network   forward of the fused network (bf16 autocast), input 928 x 1280 (an A-page at long side 1280), batch 1 / 16
  end2end   model.predict(image) at batch 1: preprocessing + network + postprocessing (NMS / decode), what a user sees
usage (Colab):  python speed_gpu.py <yolo.pt> <openobb.pt> [<openobb.pt> ...] [--image page.png] [--runs 50]
Each OpenOBB checkpoint runs with its own calibrated inference settings (head / NMS from best.pt)."""
import argparse
import statistics
import time

import numpy as np
import torch


def timed(fn, runs, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yolo")
    ap.add_argument("openobb", nargs="+")
    ap.add_argument("--image", default=None, help="a drawing for the end-to-end timing (default: a synthetic page)")
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    img = None
    if a.image:
        import cv2
        img = cv2.imread(a.image)
    if img is None:
        img = np.full((1856, 2560, 3), 255, np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(400):                                     # thin lines, like a drawing
            x, y = rng.integers(0, 2560), rng.integers(0, 1856)
            img[y:y + 6, x:x + int(rng.integers(20, 300))] = 0

    from ultralytics import YOLO  # baseline only

    from openobb import OpenOBB
    rows = []
    yolo = YOLO(a.yolo)
    ynet = yolo.model.to(dev).fuse().eval()
    models = [("YOLO " + a.yolo.split("/")[-1], ynet, lambda: yolo.predict(img, imgsz=1280, verbose=False, half=True))]
    for p in a.openobb:
        m = OpenOBB(p)
        m.predict(img, verbose=False, save=False)                # loads the network on the GPU with its settings
        net = m.model.eval()
        a_ = m.args
        tag = f"{p.split('/')[-1]} ({a_.get('arch')}, head {a_.get('infer_head', '-')}, nms {a_.get('nms_iou', '-')})"
        models.append((tag, net, lambda m=m: m.predict(img, verbose=False, save=False)))
    for name, net, e2e in models:
        res = {}
        for bs in (1, 16):
            x = torch.rand(bs, 3, 928, 1280, device=dev)

            def fwd(net=net, x=x):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    net(x)
            res[bs] = timed(fwd, a.runs)
        rows.append((name, res[1], res[16] / 16, timed(e2e, max(10, a.runs // 5), warmup=3)))
    print(f"\n{torch.cuda.get_device_name(dev)} | input 928 x 1280 | median ms")
    print(f"{'model':70s} {'net b1':>8s} {'net b16/img':>12s} {'end2end b1':>11s}")
    for name, b1, b16, e2e in rows:
        print(f"{name:70s} {b1:8.2f} {b16:12.2f} {e2e:11.2f}")


if __name__ == "__main__":
    main()
