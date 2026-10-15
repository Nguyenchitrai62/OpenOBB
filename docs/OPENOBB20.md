# Vòng nhắm mAP50-95: v19 + chưng cất box (LD), và OpenOBB20 (bộ tinh chỉnh cạnh đọc thân)

User (10-15): v19 đã đạt mAP50 (0.755), tốc độ (12.4 ms) và box trùng (2.8%); chỉ còn **mAP50-95** (độ khít box). Hai bản train song song:
một bản cải tiến v19, một bản do agent chọn. Sổ cái: F77.

Code: `openobb/models/openobb18_loss.py` (chưng cất box, `--w-ld`, `--w-kd-cls`), `openobb18.py` (`OpenOBB20`),
`openobb17.py` (`refine_stages`). Test: `tests/test_openobb20.py`.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| v19 so với v17 + nhóm (khác nhau đúng ở neck) | mAP50-95 0.462 so với 0.487; junction 50-95 .496 so với .531, note .441 / .493, door .793 / .808; mAP50 ngang (0.755 / 0.757) | F74, F75 |
| Neck nhanh: tìm object tốt, đặt box kém khít hơn encoder | P 0.672 so với 0.711; recall .764 so với .717 | F75 |
| Xếp hạng không phải vấn đề, ứng viên chưa đủ khít | box điểm cao nhất gần box tốt nhất (chênh 0.000–0.008) | F69 |
| Bộ tinh chỉnh đọc nét stride 4 trên cạnh box (v14) | wall mAP50-95 +0.045 so với v12 | F60 |
| Vòng 2 của v19 chưa hội tụ | không thầy, `nbs 64` (khoảng 112 lần cập nhật), mAP50-95 còn tăng ở epoch cuối | F75 |

## 2. Hai bản

| | Bản 1: v19 + chưng cất box | Bản 2: OpenOBB20 |
|---|---|---|
| Kiến trúc | v19 (không đổi) | v19 + `EdgeRefiner` (v14) đọc **thân stride 4** + P3 trên cạnh box, trước bước bám ranh giới |
| Lập luận | thầy v17 + nhóm có box khít hơn và **cùng cấu trúc head nhóm** → học thẳng phân phối DFL (hai đầu, độ dày), góc, lệch ngang và xác suất class của thầy (Localization Distillation; tăng mạnh ở IoU cao) | neck mới làm mất độ khít; bộ tinh chỉnh đi vòng qua neck, lấy chi tiết từ thân pretrained (không đổi), sửa đầu / độ dày / lệch ngang |
| Chi phí suy luận | 0 | khoảng 0.3% FLOPs (30 điểm lấy mẫu + MLP nhỏ trên 1500 ứng viên) |
| Khởi tạo | từ `best.pt` của v19 | từ `best.pt` của v19; lớp cuối bộ tinh chỉnh = 0 → bắt đầu đúng bằng v19 (test `test_v20_starts_as_v19`) |
| Train | thầy = v17 + nhóm (run A), chưng cất neck + box + class; 30 ep, công thức cũ (`nbs=0`) | y hệt bản 1; **khác đúng một yếu tố**: bộ tinh chỉnh |

Chưng cất box (cả hai bản, tự bật khi có `teacher=`): trên các điểm thầy chấm ≥ 0.05, KL với nhiệt độ T = 10 cho ba phân phối DFL
(trọng số 0.25, `w_ld`), L1 cho góc và lệch ngang; xác suất class: BCE kiểu quality focal tới xác suất của thầy (0.5, `w_kd_cls`).
Head nhóm của học trò học từ box cùng nhóm của thầy (thầy một nhóm thì dùng box duy nhất của thầy), trọng số = điểm cao nhất của thầy trong nhóm.

## 3. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Wall_Color, thước YOLO, khởi tạo `best.pt` v19 (F75: 0.755 / 0.462, 12.4 ms), thầy = run A (F74 / F76), 30 ep, `nbs=0 warmup_bias_lr=0 epoch_lr=False`.

| | v19 (F75) | YOLO26x | **Bản 1 kỳ vọng** | **Bản 2 kỳ vọng** |
|---|---|---|---|---|
| mAP50-95 | 0.462 | 0.499 | ≥ 0.475 | ≥ bản 1 + 0.01 |
| mAP50 | 0.755 | 0.746 | ≥ 0.75 | ≥ 0.75 |
| junction 50-95 | .496 | .584 | ≥ .52 | ≥ .53 |
| wall 50-95 | .521 | .624 | ≥ .53 | ≥ .55 |
| tốc độ G4 batch 16 | 12.4 ms | — | như v19 (± 0.3) | ≤ 13.0 ms |
| box trùng | 2.8% | — | ≤ 3% | ≤ 3% |

**Mục tiêu đột phá:** mAP50-95 ≥ 0.499 (bằng YOLO26x) với ≤ 13 ms.
**Bác bỏ** bộ tinh chỉnh nếu bản 2 không hơn bản 1 ≥ 0.005 mAP50-95. Chưng cất box chưa tách riêng được khỏi "train thêm có thầy"
(không có đối chứng `w_ld=0`); nếu bản 1 < 0.468 thì coi là không có tác dụng.

## 4. Lệnh

```bash
# Bản 1: v19 + chưng cất box từ v17 + nhóm
openobb train data=/content/data.yaml model=openobb19x epochs=30 imgsz=1280 batch=16 lr=0.001 nbs=0 warmup_bias_lr=0 epoch_lr=False weights=<best.pt của v19> teacher=<best.pt của v17 + nhóm>
# Bản 2: OpenOBB20 (v19 + bộ tinh chỉnh cạnh), cùng mọi thứ còn lại
openobb train data=/content/data.yaml model=openobb20x epochs=30 imgsz=1280 batch=16 lr=0.001 nbs=0 warmup_bias_lr=0 epoch_lr=False weights=<best.pt của v19> teacher=<best.pt của v17 + nhóm>
```

Log phải có `[distill] neck distilled from ...` và các loss `ld`, `kd_cls` trong `results.csv`; bản 2 có "edge refiner on the trunk's
stride-4 map" trong dòng `model`.

## 5. Kết quả thật

**Lần 1 của bản 1 (F78): chạy lệch thiết kế, không dùng để chấm.** File `openobb19x.pt` trên Drive là v17 + nhóm (run A), còn
`openobb17x_v2.pt` là v17 một nhóm. Vậy neck của v19 học lại từ đầu (328 tensor mới) và thầy không có head nhóm.
Kết quả: 0.752 / 0.464, 12.4 ms, box trùng 2.9%; wall .826 / .498 (wall → nền 13%), junction .895 / .518.

| Bản 1 | Kỳ vọng | Lần 1 (lệch) |
|---|---|---|
| mAP50-95 | ≥ 0.475 | 0.464, không đạt |
| mAP50 | ≥ 0.75 | 0.752, đạt |
| junction 50-95 | ≥ .52 | .518, không đạt |
| wall 50-95 | ≥ .53 | .498, không đạt |
| tốc độ | 12.4 ± 0.3 ms | 12.4 ms, đạt |
| box trùng | ≤ 3% | 2.9%, đạt |

Chạy lại đúng thiết kế: `weights=` `best.pt` của lần 1 (v19, neck đã học), `teacher=` file A (đổi tên thành `openobb17x_groups.pt` cho rõ).
Bản 2 dùng cùng `weights=` / `teacher=`. Kiểm tra trước khi train: dòng `[init]` phải có 0 hoặc rất ít `new tensors` (bản 2 chỉ có bộ tinh chỉnh mới).
