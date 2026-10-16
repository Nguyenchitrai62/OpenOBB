# OpenOBB23: v21 + bám ranh giới có xoay + lọc đỉnh thay NMS

Sổ cái: F90 (v21), F93 (v22 thất bại), F94 (profile GPU), F95 (thiết kế + kỳ vọng). Code: `openobb/models/openobb18.py`
(`BoundarySnapRot`, `OpenOBB23`), `openobb/models/openobb17.py` (`peak_mask`), `openobb/predict.py` (lựa chọn `peak`).
Test: `tests/test_openobb23.py`. Model: `model=openobb23s|m|l|x`.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| v21: mAP50 / 50-95 / trùng | 0.769 / 0.492 / 1.8% (YOLO26x 0.743 / 0.499 / 1.6%) | F89, F90 |
| v22 (hồi quy box trực tiếp + góc sigmoid) | wall sụp .466, lệch 16.2% → bỏ hướng này, quay về head DFL của v21 | F93 |
| Box dài lệch trục > 1° | v21 14.4%, YOLO 0.3%; lỗi tăng theo độ dài box | F83, F90 |
| Sau bộ tinh chỉnh, bước bám ranh giới chỉ DỊCH cạnh, không xoay | góc không đổi qua bước bám | F84 |
| Thời gian G4 (v21-x, ms / ảnh) | thân + neck 7.5, **NMS 2.9–3.4**, head 1.0, trường 0.57, tinh chỉnh + bám 0.24; YOLO 6.6 + 0.7 | F94 |
| Box trùng của head một-một khi không NMS (v20, 10 ảnh val) | 28.7% số ứng viên tự tin; **94.5% là hai điểm neo cách ≤ 1.5 stride**, 91.8% cùng tầng, 100% cùng nhóm, 88.5% điểm số ≥ 0.5 × điểm cao nhất | F95 |

## 2. Bài học → cơ chế → kỳ vọng → điều chứng minh sai

| Bài học | Cơ chế v23 | Kỳ vọng | Chứng minh sai |
|---|---|---|---|
| Lệch góc tập trung ở vật dài; trường ranh giới (stride 2) đã học mép theo nhãn | **Bám ranh giới có xoay:** trên 2 cạnh dài của box dài (L ≥ 4T), 7 điểm u ∈ [−0.4, 0.4]·L, mỗi điểm tìm lệch ngang của mép (soft-argmax trên trường); độ dốc bình phương tối thiểu có trọng số = góc lệch, xoay quanh tâm. Box càng dài càng nhiều thông tin → càng chính xác. Hệ số khởi tạo 0 | box dài lệch trục ≤ 3% | ≥ 10% sau train, hoặc hệ số xoay học được < 0.2 |
| Trùng của head một-một = điểm neo kề nhau | **Lọc đỉnh 3×3** theo tầng + nhóm class (một max-pool) thay NMS so cặp; lựa chọn trong bước hiệu chỉnh | — | **đã đo trước khi train (F95): mất 0.035 mAP50 → không làm mặc định** |
| Head một-một cho box ít nghiêng hơn (5.5% so với 15.4%, F95) ở cùng độ chính xác | hiệu chỉnh cuối train chọn cấu hình nghiêng ít hơn khi fitness và trùng ngang nhau | box dài lệch giảm thêm nhờ chọn head | — |
| Head DFL của v21 tốt nhất cho tường (F90 so F93) | giữ nguyên v21 (DFL, nhóm class, nhánh góc góc kép, tinh chỉnh, trường) | mAP50 ≥ 0.765, mAP50-95 ≥ 0.495 | mAP50-95 < 0.485 |

## 3. Chi tiết

- Thân HGNetv2-x (Apache, mượn), neck RepConv, head dense kép DFL + 3 nhóm class + nhánh góc góc kép, bộ tinh chỉnh cạnh, trường ranh giới,
  **bám ranh giới có xoay** (mới, +1 tham số), **lọc đỉnh** (mới, không tham số).
- Tham số: như v21 (51.28M) + 1. Bắt đầu đúng bằng v21 (`test_v23_starts_as_v21`).
- Thời gian dự kiến (G4, batch 16): như v21 (khoảng 12.7 ms; lọc đỉnh không dùng được làm mặc định). Tốc độ để bản sau: thân + neck 7.5 ms,
  NMS ~3 ms; cần `openobb profile ... detail=True` để chọn khối cần làm nhẹ.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Fine-tune từ best.pt v21 (F90), 100 epoch, mặc định (không `rotate_p`), Wall_Color val, thước YOLO.

| | v21 (F90) | YOLO26x | **Kỳ vọng v23** |
|---|---|---|---|
| Box dài lệch trục > 1° (`Box geometry`) | 14.4% | 0.3% | ≤ 3% |
| Vật dài lệch góc > 1° | 11.3% | 0.0% | ≤ 3% |
| mAP50 / mAP50-95 | 0.769 / 0.492 | 0.743 / 0.499 | ≥ 0.765 / ≥ 0.495 |
| Box trùng | 1.8% | 1.6% | ≤ 2.5% |
| Speed (inference + postprocess) | 9.7 + 3.0 ms | 6.6 + 0.7 ms | ≤ 13.0 ms tổng (như v21) |

## 5. Lệnh

```bash
openobb train data=/content/data.yaml model=openobb23x weights=<best.pt của v21> epochs=100 imgsz=1280 batch=16
```

Log cần thấy: dòng `model` có "rotational boundary snapping"; `[init]` báo 1 tensor mới (`snap.rot_gain`); dòng `NMS calibrated` có thêm
`o2o peak` và `tilt x%`; dòng `augment` không có `rotate any angle` (nếu muốn bản tổng quát cho bản vẽ xoay thì chạy thêm một bản có `rotate_p`).

## 6. Kết quả thật

(chưa train)
