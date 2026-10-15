# OpenOBB22: head hình học kiểu YOLO26 trên nền v21

Sổ cái: F81, F84–F90 (bằng chứng), F91 (thiết kế + kỳ vọng). Code: `openobb/models/openobb18.py` (`OpenOBB22`), test `tests/test_openobb22.py`.
Model: `model=openobb22s|m|l|x`.

## 1. Bằng chứng

| Đo | v21 (F90) | YOLO26x (F89, cùng G4, cùng thước) | Nguồn |
|---|---|---|---|
| mAP50 / mAP50-95 | **0.769** / 0.492 | 0.743 / **0.499** | F89, F90 |
| Box trùng | 1.8% | 1.6% | F89, F90 |
| Box tường dài lệch trục > 1° | 14.4% | **0.3%** | F89, F90 |
| Vật dài lệch góc > 1° / đầu lệch khỏi object | 11.3% / 1.9% | **0.0% / 0.0%** | F89, F90 |
| mAP50-95 wall / junction / wall_300 | .545 / .538 / .310 | **.624 / .584 / .441** | F89, F90 |
| mAP50-95 slide_door / double_door / note | **.320 / .435** / .487 | .123 / .392 / .517 | F89, F90 |
| Sai số đầu box của wall (trung vị) | 1.92 px (v20) | 1.30 px | F81 |
| Mẫu dương của tường dọc lúc train | góc dự đoán ~5° (nằm ngang), trọng số TAL ~0 | — | F86 |
| Tốc độ G4 batch 16 (máy nóng) | 9.7 + 3.0 ms | 6.6 + 0.7 ms | F90 |

→ v21 thắng hoặc ngang YOLO ở nhận diện (mAP50, lớp hiếm, box trùng). Phần thua là **hình học box** của các lớp dài, mảnh, nhiều
(hướng + hai đầu), đúng chỗ head của mình khác YOLO26.

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế trong v22 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Góc thô khởi tạo 0 = nằm ngang; mẫu dương tường dọc sai 90° → ProbIoU ~0 → trọng số TAL ~0 → không bao giờ học (F86) | góc = (sigmoid(u) − ¼)·π ∈ (−45°, 135°), khởi tạo 45°: 0° và 90° nằm trong khoảng và sai ĐỀU nhau lúc đầu (ý tưởng YOLO-OBB) | box dài lệch trục ≤ 2% | ≥ 8% |
| DFL 32 bin, bin = stride (8 px ở P3), đọc bằng trung bình: đầu box sai 1.5–1.8× YOLO; YOLO26 bỏ DFL | hồi quy thẳng hai đầu + độ dày (softplus × stride), L1 + ProbIoU (`box=direct`, có sẵn từ F55, nay đúng cho nhóm class) | mAP50-95 wall ≥ .60, wall_300 ≥ .40, junction ≥ .57 | wall < .55 |
| Phần thắng YOLO đến từ nhóm class, bộ tinh chỉnh, bám ranh giới (F72–F79) | giữ nguyên | mAP50 ≥ 0.765, box trùng ≤ 2.0%, lớp hiếm giữ | mAP50 < 0.75 |

## 3. Chi tiết từng phần

| Phần | v22 | Nguồn |
|---|---|---|
| Thân | HGNetv2-x pretrained (D-FINE COCO, Apache-2.0), 33.2M | mượn |
| Neck | RepConv tự viết (v18) | tự viết |
| Head | dense kép (một-nhiều + một-một), 3 nhóm class; mỗi nhóm: góc (sigmoid, 45° lúc đầu), lệch ngang, 2 khoảng cách đầu, độ dày (giá trị trực tiếp) + nhánh sửa góc góc kép (v21) | tự viết |
| Sau head | bộ tinh chỉnh cạnh (v20) → bám ranh giới (v17) → NMS hiệu chỉnh | tự viết |
| Loss | TAL + quality focal (cls), ProbIoU + L1 (box, theo stride), loss góc vectơ (cos 2θ, sin 2θ) trọng số đều (0.20.1) | tự viết |

Tham số 51.06M (v21: 51.28M; head nhẹ hơn vì bỏ 3 × 32 bin mỗi nhóm). Tỉ lệ mượn: thân ~66% tham số; code 100% tự viết.

**Khởi tạo khi fine-tune từ v20 / v21** (`after_weights`): thân, neck, tháp phân loại, bộ tinh chỉnh, trường ranh giới giữ nguyên; hàng ra box
dựng lại theo từng nhóm: góc về 45°, hai đầu + độ dày bắt đầu lại, lệch ngang chép từ checkpoint. Không bắt đầu đúng bằng v21 (đổi nghĩa đầu ra),
nên cần nhiều lần cập nhật hơn: 150 epoch như YOLO.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Wall_Color, fine-tune 150 ep từ best.pt của v21 (F90), mặc định (không `rotate_p`), thước YOLO, NMS hiệu chỉnh. Cặp so sánh: A = v21 cùng số epoch
(run user đang chạy, tắt `rotate_p`), B = v22.

| | v21 (F90) | YOLO26x | **Kỳ vọng v22** |
|---|---|---|---|
| Box dài lệch trục > 1° (`Box geometry`) | 14.4% | 0.3% | ≤ 2% |
| Vật dài lệch góc > 1° | 11.3% | 0.0% | ≤ 2% |
| mAP50-95 | 0.492 | 0.499 | ≥ 0.51 |
| wall / junction / wall_300 (50-95) | .545 / .538 / .310 | .624 / .584 / .441 | ≥ .60 / ≥ .57 / ≥ .40 |
| mAP50 | 0.769 | 0.743 | ≥ 0.765 |
| Box trùng | 1.8% | 1.6% | ≤ 2.0% |
| Tốc độ G4 batch 16 | 12.7 ms | 7.3 ms | ≤ 12.7 ms |

**Bác bỏ** góc sigmoid nếu box dài lệch trục ≥ 8%; bác bỏ hồi quy thẳng nếu wall 50-95 < .55. Nếu run A (v21, không xoay) đã ≤ 2% lệch thì phần góc
của v22 không còn là yếu tố quyết định; khi đó v22 chỉ được chấm ở độ khít (đầu box, mAP50-95 của wall / junction / wall_300).

## 5. Lệnh

```bash
openobb train data=/content/data.yaml model=openobb22x epochs=150 imgsz=1280 batch=16 weights=<best.pt của v21>
```

Log cần thấy: `OpenOBB 0.21.0`; `[init] OpenOBB22: box outputs rebuilt per class group`; dòng `model` có "angle in (-45, 135) deg from 45 deg"
và "direct box regression (no DFL)"; dòng `augment` không có `rotate any angle`. mAP vài epoch đầu sẽ thấp (box bắt đầu lại), rồi tăng.

## 6. Kết quả thật

(chưa train)
