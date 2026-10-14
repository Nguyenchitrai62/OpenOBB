# OpenOBB15: OpenOBB14 + luồng độ phân giải gốc + lưới P2, bộ tinh chỉnh đọc nét ở độ phân giải gốc (2026-10-14)

`model=openobb15s | m | l | x`. Code: [openobb/models/openobb15.py](../openobb/models/openobb15.py) (kế thừa OpenOBB14).
`imgsz` là cỡ của lõi (như mọi model, ví dụ 1280); khung tự chuẩn bị ảnh ở 2 × imgsz (2560) cho luồng chi tiết. Chưa train.
Mốc: YOLO26x 0.746 / 0.499; v12 0.743 / 0.472 (15.3 ms); v14 0.713 / 0.469 (10.9 ms) (F59, F60).

## 1. Bằng chứng: phần thua nằm ở vật tí hon, và pixel là thứ YOLO không có

| Class (số object val) | v14 mAP50-95 | YOLO26x | Kích thước ở 1280 | ở ảnh gốc 2381 px |
|---|---|---|---|---|
| junction (1773) | .542 | .584 | 9.5 × 9 px | 18 × 17 px |
| wall (1044) | .565 | .624 | dày 5.6 px | dày 10.8 px |
| wall_300 (141) | .291 (mAP50 .677) | .441 (.824) | 10 × 6 px | 19 × 11 px |
| note (60) | .450 | .517 | | |

- mAP50-95 phạt sai số **tương đối**: 1 px lệch trên nút 9 px làm IoU rơi dưới 0.75; cùng 1 px trên nút 18 px thì không.
- 53% wall_300 chung ô P3 với một junction ở 1280; ở lưới stride 4 trên ảnh 2560 chỉ còn 6% (F52) → wall_300 có điểm lưới và đặc trưng riêng.
- YOLO26x đọc ảnh ở 1280 nên cũng chỉ thấy 9 px. Muốn thấy 18 px, YOLO phải chạy cả mạng ở 2560: gấp 4 lần chi phí (905 → 3620 GFLOPs).
  v15 chỉ chạy một luồng nhẹ ở độ phân giải gốc (36 GFLOPs) và một head P2 (40 GFLOPs).
- v13 (0.11.0) là ý này trên lõi v12, nhưng chưa bao giờ có run hợp lệ (F54: đặt sai `imgsz`). v15 ghép ý đó với bộ box của v14 đã chứng minh tác dụng trên tường (+.045, F60).

## 2. Lập luận

| Mảnh | Cơ chế | Kỳ vọng | Bác bỏ nếu |
|---|---|---|---|
| **Luồng độ phân giải gốc** (`hires`) | CNN nhẹ trên ảnh 2560 → bản đồ stride 8 (ảnh gốc) = stride 4 theo thang lõi, 192 kênh; lõi (thân + neck) chạy trên ảnh thu một nửa, đúng như v14 ở 1280 | đặc trưng thấy gấp 4 lần số pixel ở vật tí hon | — |
| **Lưới P2** (`p2_lat`, `p2_down`, `p2_fuse`, `p2_head`, `p2_head_o2o`) | lateral của luồng gốc + P3 phóng lên → RepConv → head kép riêng, hồi quy trực tiếp; stride 8 ảnh gốc | junction / wall_300 có ô riêng: wall_300 mAP50 ≥ .78, ô nhầm wall_300 → junction < 10% | wall_300 mAP50 không tăng ≥ 0.05 so với v14 |
| **Bộ tinh chỉnh đọc luồng gốc** (`edge`, s4 = 8, s3 = 16 px ảnh gốc) | 24 điểm trên cạnh / đầu / tim box lấy mẫu trên bản đồ mang pixel gốc; tường dày 11 px được đọc ở bước ~2 px theo thang lõi | wall mAP50-95 ≥ .60, junction ≥ .58 | junction 50-95 không hơn v14 (.542) ≥ 0.02 |
| **`to_p3`** | tóm tắt stride 16 (ảnh gốc) của luồng gốc cộng vào bản đồ stride 8 của thân, khởi tạo 0 | lõi bắt đầu đúng bằng v14 (test `test_v15_core_levels_start_as_v14_at_half_resolution`) | — |
| **Neck** | `neck=dfine` (encoder pretrained) mặc định cho vòng này, vì F60 cho thấy neck tự thiết kế mất tiên nghiệm trên 250 ảnh; `neck=own` vẫn có | | |

**Vì sao không lặp lại c5 (P2 cho decoder, −9.1 trên FPC):** c5 lấy P2 từ chính ảnh 1024 của lõi (không thêm pixel) và đưa vào decoder DETR 300 query; ở đây P2 mang pixel mới và có head dense riêng.

**Giới hạn nói trước:** chi phí tăng (mục 3) nên v15 nhắm độ chính xác, v14 nhắm tốc độ. Ảnh Wall_Color gốc 2000–3000 px; data nào có ảnh nhỏ hơn 2 × imgsz thì luồng gốc chỉ thấy ảnh phóng, không có pixel mới, lợi ích giảm.

## 3. Kích thước (bản x, đường suy luận, một head)

| | Tham số | GFLOPs | Đầu vào |
|---|---|---|---|
| v12-x | 68.0M | 682 | 960 × 1280 |
| v14-x `neck=dfine` | 69.0M | 713 | 960 × 1280 |
| **v15-x `neck=dfine`** | **71.1M** | **882** (luồng gốc 36, head P2 40) | 1920 × 2560 |
| v15-x `neck=own` | 62.7M | 752 | 1920 × 2560 |

Tận dụng có sẵn (bản `neck=dfine`): thân + encoder D-FINE 53.9M = 76%; tự viết 24% (head kép, head P2, luồng gốc, bộ tinh chỉnh). Bản `neck=own`: 53% / 47%.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Wall_Color, val 53 ảnh, thước YOLO, COCO init, 100 epoch, `imgsz=1280` (ảnh vào 2560), cùng hiệu chỉnh head / NMS cuối train.

| | v14 (F60) | YOLO26x | **v15 `neck=dfine` kỳ vọng** |
|---|---|---|---|
| mAP50 / 50-95 | 0.713 / 0.469 | 0.746 / 0.499 | **≥ 0.76 / ≥ 0.51** |
| junction 50-95 | .542 | .584 | ≥ .58 |
| wall 50-95 | .565 | .624 | ≥ .60 |
| wall_300 mAP50 / 50-95 | .677 / .291 | .824 / .441 | ≥ .78 / ≥ .38 |
| note mAP50 | .657 | .764 | ≥ .72 |
| tốc độ G4, batch 16 | 10.9 ms | — | ≤ 20 ms (khoảng 1.3 × v12) |

**Bác bỏ** nếu mAP50-95 < 0.49, hoặc junction 50-95 không hơn v14 ≥ 0.02, hoặc wall_300 mAP50 tăng < 0.05.

## 5. Vòng hai hạt giống

Khác nhau đúng một yếu tố (luồng gốc + P2 + bộ tinh chỉnh đọc luồng gốc); cả hai dùng encoder pretrained:

```bash
openobb train model=openobb15x data=/content/data.yaml imgsz=1280 epochs=100 neck=dfine
openobb train model=openobb14x data=/content/data.yaml imgsz=1280 epochs=100 neck=dfine
```

Log in `OpenOBB15: imgsz 1280 is the core's size; images are prepared at 2560 px`. Batch tự chọn theo VRAM và tự giảm khi thiếu.
Khi gửi log: bảng theo class, confusion matrix (ô wall_300 → junction), `Duplicate boxes`, `NMS calibrated`, dòng `Speed`.
Fine-tune từ checkpoint v14 (`weights=`): thân, neck, head P3–P5 nạp thẳng; luồng gốc, P2 và bộ tinh chỉnh khởi tạo mới.
