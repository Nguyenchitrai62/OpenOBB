# OpenOBB13: luồng chi tiết độ phân giải gốc + lưới P2 cho vật tí hon (2026-10-13)

`model=openobb13s | m | l | x`. Nền là OpenOBB12 (lõi OpenOBB7 pretrained COCO + head dense kép, không NMS).
**Từ 0.12.0, `imgsz` mang đúng nghĩa như mọi model khác: cỡ ảnh của lõi** (ví dụ 1280). Khung tự chuẩn bị ảnh ở 2 × imgsz
(2560) cho luồng chi tiết; lõi pretrained thấy đúng ảnh như v12 ở 1280.
(Ở 0.11.0, `imgsz` là độ phân giải đầy đủ. Run v13 đầu tiên của user đặt 1280 nên lõi chỉ chạy ở 640: run đó không hợp lệ, F54.)

Bản này thay cho v13 cũ của 0.10.0 (lõi nhanh v9 + head kép). Bản cũ chỉ nhắm tốc độ, nên user bỏ.
Checkpoint v13 cũ không nạp được bằng 0.11.0.

## 1. Bằng chứng: độ chính xác đang mất ở đâu

- So với YOLO26x, **khoảng 95% khoảng cách mAP50 nằm ở wall_300** (0.60–0.65 so với 0.824, F31 / F47).
  Phần còn lại là độ khít của vật mảnh (mAP50-95 của wall, note, junction).
- wall_300 bị nhầm sang junction 27–32%. Ô nhầm này đứng yên qua mọi thay đổi head / ngữ cảnh / chi tiết / slot (v3 → v10).
- Đo trực tiếp trên nhãn (`research/analysis/wall300_*.py`, 540 wall_300):

| Đo | Kết quả |
|---|---|
| Kích thước ở ảnh gốc / ở 1280 | wall_300 19 × 11 px → **10 × 6 px**; junction 19 × 18 → 9.5 × 9 px; tường dày 10.8 → 5.6 px |
| Khoảng cách wall_300 → junction gần nhất | trung vị 0.4 lần chiều dài của nó (wall_300 nằm sát junction, cạnh cửa) |
| **wall_300 chung ô lưới tâm với một junction** | **53% ở P3 (stride 8, ảnh 1280)**; 20% ở lưới stride 4; 6% ở stride 4 trên ảnh 2560 |
| Màu (hồng) có mất khi thu nhỏ không | **không**: tỉ lệ pixel hồng 0.063 (gốc) so với 0.057 (1280); 42% junction cũng có hồng → màu không phải nguyên nhân |

Kết luận:
- Ở lưới P3, một nửa số wall_300 tranh cùng một điểm với junction.
- Head dense chỉ gán một object cho mỗi điểm, và head một-một chỉ ra một box mỗi điểm. Vì vậy wall_300 thua (bỏ sót) hoặc bị gọi là junction.
- v10 đã thử thêm slot (nhiều box mỗi điểm): recall wall_300 tăng 0.41 → 0.50, nhưng ô nhầm không đổi. Lý do là các slot dùng chung một vector đặc trưng đã trộn hai vật.
- Cần **điểm lưới riêng với đặc trưng riêng**: lưới mịn hơn, và đặc trưng nhìn thấy nhiều pixel hơn.

## 2. Lập luận

| Mảnh | Cơ chế | Kỳ vọng | Điều sẽ chứng minh sai |
|---|---|---|---|
| **Luồng độ phân giải gốc** (`hires`, mới) | CNN nhẹ (conv s2 ×3 + khối depthwise dư) chạy trên ảnh đầy đủ (2560). Lõi chỉ thấy ảnh thu nửa | Đặc trưng nhìn thấy 4 lần số pixel ở vật tí hon: wall_300 19 × 11 px thay vì 10 × 6; tường dày 11 px thay vì 5.6 | — |
| **Tầng lưới P2** (stride 4 theo thang của lõi = 8 px ảnh gốc) | P2 = lateral của luồng gốc + P3 của encoder hạ kênh, phóng ×2; head dense kép riêng (`p2_head`, `p2_head_o2o`) | Ô chung với junction 53% → khoảng 6–20%: wall_300 có điểm riêng | Ô nhầm wall_300 → junction không xuống dưới 15% |
| **Chi tiết vào P3** (`to_p3`) | Tóm tắt stride 16 (ảnh gốc) của luồng gốc cộng vào P3 backbone trước LSK, khởi tạo 0 | Tường / junction đọc ở P3 khít hơn (mAP50-95) | mAP50-95 của wall và junction không tăng |
| **Gán theo tầm với** (bật lại `reach`) | P2 chỉ với được khoảng 124 px mỗi phía (31 bin × 4); vật dài phải về P3–P5, không bị cắt cụt | Không sinh box cụt ở P2 | — |
| **Lõi giữ nguyên** | Lõi pretrained thấy đúng ảnh như v12 ở 1280. Mọi mảnh mới khởi tạo sao cho P3–P5 lúc đầu = v12 (test `test_v13_core_levels_start_as_v12_at_half_resolution`) | Không làm kém class đang tốt | junction hoặc door giảm quá 0.02 so với v12 |

**Vì sao không phải lặp lại c5 (P2 bị bác bỏ ở FloorPlanCAD, 10-08):**
- c5 lấy P2 từ chính ảnh của lõi (không thêm pixel) và đưa vào decoder DETR.
- Ở đây P2 nhận **pixel mới** từ ảnh gấp đôi, đi vào head dense có điểm riêng.
- Lỗi nhắm tới được đo trực tiếp: 53% wall_300 chung ô với junction.

**Vì sao vượt được YOLO.** YOLO26x đọc ảnh ở 1280 nên cũng thấy wall_300 chỉ 10 × 6 px. Muốn thấy chi tiết gốc, YOLO phải chạy cả mạng ở 2560: gấp khoảng 4 lần chi phí. v13 chỉ chạy một luồng nhẹ ở độ phân giải gốc.

## 3. Chi tiết từng phần và tỉ lệ tận dụng (bản x)

| Phần | Tham số | Nguồn |
|---|---|---|
| HGNetv2 + LSK + hybrid encoder + `spd` | khoảng 62.7M | D-FINE (Apache-2.0) init COCO + phần sửa của v7 (tự viết) |
| Head kép P3–P5 (một-nhiều + một-một) | 5.3M | tự viết (v12) |
| **Luồng gốc + `to_p3` + P2 + head kép P2** | **2.1M** | tự viết (mới) |
| **Tổng** | **70.1M** | Tận dụng có sẵn khoảng 77% (backbone + encoder D-FINE) |

Chi phí: 853 GFLOPs ở 2560 × 1920 (đường suy luận), so với v12-x 699 GFLOPs ở 1280 × 960 (+22%).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Điều kiện: Wall_Color, val 53 ảnh, thước YOLO, 100 epoch, khởi tạo COCO.
- v13: `imgsz=1280` (lõi 1280, luồng chi tiết 2560).
- v12: `imgsz=1280`.

| | v10 | v12 (kỳ vọng) | **v13 kỳ vọng** | YOLO26x |
|---|---|---|---|---|
| mAP50 / 50-95 | 0.693 / 0.456 | 0.68–0.71 / 0.45–0.47 | **0.73–0.77 / 0.48–0.51** | 0.746 / 0.499 |
| wall_300 mAP50 | 0.649 | | **≥ 0.75** | 0.824 |
| ô wall_300 → junction | 27% | | **< 15%** | |
| wall / junction mAP50-95 | .548 / .517 | | +0.02 trở lên | .624 / .584 |
| box trùng, không NMS | | < 3% | < 3% | |
| suy luận so với v12 | | | ≤ 1.3× | |

**Bác bỏ** nếu xảy ra một trong các điều sau:
- wall_300 mAP50 tăng dưới 0.05 so với v12;
- ô wall_300 → junction ≥ 15%;
- junction hoặc door giảm quá 0.02 so với v12.

## 5. Train (Colab)

```bash
openobb train model=openobb13x data=/content/data.yaml imgsz=1280 epochs=100
```

- Ảnh Wall_Color gốc 2000–3000 px; với `imgsz=1280`, luồng chi tiết đọc ảnh 2560, gần đúng độ phân giải gốc.
  Log in dòng `OpenOBB13: imgsz 1280 is the core's size; images are prepared at 2560 px`.
- Batch tự chọn theo VRAM, tự giảm nếu thiếu bộ nhớ.
- Đi tiếp từ checkpoint v12 / v7 / v8 (`weights=`): lõi và head P3–P5 nạp thẳng. Luồng gốc và P2 khởi tạo mới; head một-một chép từ head một-nhiều.
- Khi gửi log, nhờ kèm: bảng theo class, confusion matrix (ô wall_300 → junction), dòng `Duplicate boxes`, dòng `NMS calibrated`, tốc độ val.
