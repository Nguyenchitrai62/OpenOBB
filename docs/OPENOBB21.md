# OpenOBB21: sửa lệch góc (tường dọc) + nhánh góc trong không gian góc kép

Sổ cái: F83 (đánh giá lại có thước hướng box), F84 (nguyên nhân lệch góc). Code: `openobb/models/openobb18.py` (`OpenOBB21`),
`openobb/models/openobb3.py` (`reg_out`, `fp32_out`), `openobb/models/openobb3_loss.py` (`angle_loss`, `ANGLE_LOSS`),
`openobb/eval/ultra.py` (`geometry`, dòng `Box geometry`). Test: `tests/test_openobb21.py`, `tests/test_geometry.py`.

## 1. Bằng chứng

| Đo | Số | Nguồn |
|---|---|---|
| Box tường dài (≥ 100 px) lệch > 1° so với trục, val | v17 + nhóm 32.6%, v19 43.2%, v20 26.3%, v20 không xoay ảnh 28.4%, **YOLO26x 0%** | F83 |
| Cùng thước, bản vẽ thật (9F / 三越 / きらぼし) | v20 không xoay 31% / 44% / 0%; YOLO 0% / 0% / 0% | F83, F84 |
| Lệch góc theo hướng trong ảnh (head, 20 ảnh val) | **tường dọc 88.7% lệch > 1°, tường ngang 0.3%** | F84 |
| Xoay ảnh vào 90° | lỗi chuyển sang tường vừa thành dọc (87.5% / 0.9%) → do biểu diễn góc, không do nội dung | F84 |
| Góc của tường dọc | vượt ±90° với trung vị +3.9° (v20 không xoay); v17 + nhóm tản hai phía | F84 |
| Loss góc của head | 1 − cos 2Δθ: ở 1° chỉ 6e-4, gradient gần như bằng 0 dưới vài độ; bộ tinh chỉnh (L1 × 5) có gradient và đã giảm lệch (v19 → v20) | code, F83 |
| Đã loại trừ | nhãn (tường dọc chuẩn hoá đúng −π/2), tăng cường xoay (tắt vẫn lệch), bộ tinh chỉnh (tắt còn tệ hơn), bám ranh giới (không đổi góc), `fuse()` (giống hệt) | F82–F84 |

## 2. Bản sửa chung (mọi model dense v3–v21, bật mặc định từ 0.20.0 / 0.20.1)

**Nguyên nhân gốc (F86, 0.20.1):** mẫu dương của tường dọc vẫn dự đoán box nằm ngang (góc khởi tạo 0) nên trọng số TAL (∝ ProbIoU^6) ≈ 0 và
góc của chúng không bao giờ được dạy. Sửa: loss góc dùng trọng số đều trên mẫu dương (`--angle-balance uniform`, mặc định; `tal` = cũ).


| Bài học | Cơ chế | Kỳ vọng | Điều chứng minh sai |
|---|---|---|---|
| Gradient góc tắt gần tối ưu | loss góc = L1 giữa vectơ góc kép (cos 2θ, sin 2θ) của dự đoán và nhãn: gradient khoảng 2 / rad ở mọi mức sai, tuần hoàn π, không có mép | tường dọc hết vượt ±90° | tường dọc vẫn lệch > 1° ≥ 10% |
| Góc tường dọc ~1.57 rad ra từ conv bf16 (bước 0.45°), góc tường ngang ~0 | lớp ra hồi quy của head chạy fp32 (`reg_out`) | bớt nhiễu góc tường dọc | — (rẻ, đi kèm) |

`angle_loss=cos` trả lại cách cũ (để tái lập). Log in `angle loss vec` ở dòng `augment` và dòng `Box geometry` sau `Duplicate boxes`.

## 3. OpenOBB21 = OpenOBB20 + nhánh sửa góc trong không gian góc kép

- Mỗi tầng của mỗi head (một-nhiều và một-một) thêm một conv 1×1 trên tháp hồi quy, ra 2 kênh cho mỗi nhóm class: (dc, ds).
- Góc mới: θ' = θ + ½·atan2(v × v', v · v'), với v = (cos 2θ, sin 2θ) là vectơ góc kép của góc head cũ và v' = v + (dc, ds).
  Phần sửa nằm trong không gian 0° và 90° đối xứng (đều là vectơ đơn vị, không có mép), bị chặn, tính ở fp32.
- Conv mới khởi tạo 0 → bắt đầu đúng bằng v20 (`test_v21_starts_as_v20`); `fuse()` giữ đầu ra.
- Chi phí: 2 × 3 kênh / tầng / head (G = 3 nhóm), dưới 0.01% FLOPs. Tham số: 51.27M + khoảng 4.6K.
- Tận dụng: 100% của v20; phần mới tự viết.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Wall_Color, fine-tune 30 ep từ `openobb20x_v2.pt` (F84 mốc: 0.760 / 0.486 cache CPU, Colab 0.756 / 0.487; trùng 2.3%; 12.6 ms),
thước YOLO, NMS hiệu chỉnh. Hai hạt giống cùng phiên, khác đúng một yếu tố (nhánh góc của v21):

| | Mốc v20 không xoay (cũ) | YOLO26x | **A: v20 + bản sửa chung** | **B: v21** |
|---|---|---|---|---|
| Box tường dài lệch > 1° (val, dòng `Box geometry`: "long detections > 1 deg off the axes") | 28.4% | 0% | ≤ 8% | ≤ 3% |
| Vật dài (≥ 10 × độ dày) góc > 1° | (đo lại trên checkpoint) | — | giảm ≥ một nửa | ≤ A |
| mAP50-95 | 0.487 | 0.499 | ≥ 0.490 | ≥ 0.495 |
| mAP50 | 0.756 | 0.746 | ≥ 0.750 | ≥ 0.750 |
| Box trùng | 2.3% | 1.5% | ≤ 2.6% | ≤ 2.6% |
| Tốc độ G4 batch 16 | 12.6 ms | — | ≤ 12.9 ms | ≤ 12.9 ms |

**Bác bỏ bản sửa chung** nếu A vẫn có ≥ 20% box tường dài lệch > 1°. **Bác bỏ nhánh góc của v21** nếu B không thấp hơn A ≥ 2 điểm % ở dòng lệch
góc hoặc mAP50-95 thấp hơn A quá 0.01. Bản vẽ thật: chạy `research/analysis/tilt_real.py` trên best.pt (9F, 三越, きらぼし), kỳ vọng ≤ 5%.

## 5. Lệnh

```bash
# A: v20 + bản sửa góc chung (mặc định từ 0.20.0)
openobb train data=/content/data.yaml model=openobb20x epochs=30 imgsz=1280 batch=16 nbs=0 weights=<openobb20x_v2.pt>
# B: OpenOBB21, cùng mọi thứ
openobb train data=/content/data.yaml model=openobb21x epochs=30 imgsz=1280 batch=16 nbs=0 weights=<openobb20x_v2.pt>
```

Cần **0.20.1** trở lên. Bỏ `rotate_p` khỏi notebook và dùng `nbs=0` (450 lần cập nhật thay vì 112).
Kiểm tra trong log: dòng `augment` có `angle loss vec (uniform weights)` và không có `rotate any angle`; `[init]` của A báo 0 tensor mới, của B chỉ `ang.*` / `ang_o2o.*` (6 tensor × 2);
dòng `model` của B có "angle corrected in the double-angle space"; cuối log có dòng `Box geometry`.

## 6. Kết quả thật

(chưa train)
