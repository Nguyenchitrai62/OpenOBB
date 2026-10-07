"""Baseline YOLO11x/YOLO26x-OBB trên cùng subset (chạy trên Kaggle GPU). AGPL - chỉ so sánh.
Usage (kaggle notebook):
  pip install ultralytics
  python yolo_baseline.py --data /kaggle/input/<ds>/data.yaml --model yolo11x-obb.pt --epochs 50
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11x-obb.pt", choices=["yolo11x-obb.pt", "yolo26x-obb.pt", "yolo11n-obb.pt"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/yolo_baseline")
    args = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            project=args.project, patience=10, mosaic=0.0, degrees=90.0,
            fliplr=0.5, flipud=0.5, cos_lr=True, amp=True, workers=8, val=True)


if __name__ == "__main__":
    main()
