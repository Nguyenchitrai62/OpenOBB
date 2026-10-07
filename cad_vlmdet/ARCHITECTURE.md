# CAD-VLMDet-OBB — kiến trúc mới cho floorplan/CAD/BIM (MIT, thương mại được)

Hybrid: **Vision-Language Encoder (hiểu ảnh) + YOLO-style OBB Decoder (sinh bbox nhanh)**.

## Vì sao YOLO thuần chưa đủ (từ dataset của bạn)
- 16.157 train + 851 valid, ảnh 1280², **2.58M boxes, ~150 boxes/ảnh, max 396**, 62% xoay.
- Mất cân bằng: wall 1.19M, junction 959k >> window 192k, door 189k >> slide_door 28k, double_door 25k; class `opening` = 0.
- YOLO CNN-local khó học quan hệ: window/door phải nằm trên wall, junction là đầu mút wall.
- DETR-style (100 queries) chết với 150-400 boxes/ảnh -> phải dense như YOLO.

## Kiến trúc (deep theo yêu cầu)
```
Input 1024/1280
 └─ Stem 3 conv (/4 = P2, giữ chi tiết junction nhỏ)
 └─ Stage P3: ConvNeXtDeep xN + StripPoolGCAM (N=4 base)
 └─ Stage P4: ConvNeXtDeep xN + GCAM x2 (N=8 base, tường dài)
 └─ Stage P5: ConvNeXtDeep xN + GCAM + Transformer xL (N=6, L=4 base)
 │    -> global reasoning: door/window attend vào wall memory
 ├─ MaskedRefine (MFRM-lite): P5->P4, P4->P3 (outer->inner như FloorPlanFormer)
 └─ PAN top-down + bottom-up (RepCSPDeep)
 └─ Decoupled OBB head / scale (P2 s4, P3 s8, P4 s16, P5 s32):
      cls = dot(visual emb, class_emb)  # open-vocab kiểu YOLO-World/SigLIP
      box = DFL(l,t,r,b), ang = sigmoid*90° (long-edge)
```

| variant | depths | TR | width | params | dùng khi nào |
|---|---|---|---|---|---|
| nano | (2,2,2) | 2L | 0.5 | ~9.8M | smoke CPU |
| small | (3,4,3) | 3L | 0.75 | ~25M | test nhanh Kaggle 800 ảnh |
| **base** | **(4,8,6)** | **4L** | **1.0** | **~54M** | **khuyên dùng 16k + thương mại** |
| large | (6,12,8) | 6L | 1.25 | ~102M | max accuracy |

So với YOLO26x-OBB 62.6M / YOLO11x-OBB ~62M: base tương đương size nhưng hiểu topology hơn.

## Cơ chế tối ưu + ảnh lớn CAD/BIM
- P2 stride-4 riêng cho junction; strip-pooling 1x11/11x1 cho tường hành lang dài.
- Topology loss: door/window/conf cao mà wall prob <0.2 bị phạt (ép "nằm trên tường").
- FCOS center-sampling + size-based level assign (junction luôn về P2).
- Infer ảnh lớn: SAHI tiling 1024 overlap 256 + rotated NMS merge (`infer.py:predict_tiled`).
- AMP, torch.compile, ONNX export (thêm `export.py` khi cần TensorRT).

## Kết quả test local (CPU, chứng minh học được)
- subset 800 train / 200 valid đã tạo từ `dataset_obb_train_v3.zip`.
- micro 40 ảnh, nano, 384px, 2 epochs CPU: **loss 5.158 -> 4.765** (~30s/epoch).
- infer valid: ~150-300 boxes/ảnh @conf 0.5 (GT ~150), đúng mật độ.

## Train thật trên Kaggle GPU (800 ảnh so YOLO trước)
```bash
pip install -r requirements.txt
# upload subset/ lên Kaggle Dataset, rồi trong notebook:
python train.py --data /kaggle/input/<ds>/data.yaml --variant small --imgsz 1024 --epochs 50 --batch 16
python yolo_baseline.py --data /kaggle/input/<ds>/data.yaml --model yolo11x-obb.pt --imgsz 1024 --epochs 50 --batch 16
# rồi so mAP50, mAP50-95, đặc biệt window/door hiếm + junction
```
Full 16k: `--variant base --imgsz 1024 --epochs 100 --batch 32` trên A100/H200.

## License
MIT — dùng thương mại thoải mái (không dính AGPL như Ultralytics YOLO).
Baseline YOLO chỉ để so sánh, không ship.
