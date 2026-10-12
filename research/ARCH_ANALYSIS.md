# Phân tích các mảnh kiến trúc: thiết kế gốc, số đo trên bản vẽ CAD, kết luận (2026-10-12)

Mục đích: trước khi tạo kiến trúc mới, hiểu từng mảnh đang dùng hoặc đang so sánh. Với mỗi mảnh, trả lời ba câu:
1. Người thiết kế làm như vậy vì sao (cho bài toán nào)?
2. Trên ảnh bản vẽ CAD nó thực sự làm gì (đo trên model đã train)?
3. Giữ, bỏ, hay thay bằng gì?

Đây là phân tích, chưa phải kiến trúc mới.

**Cách đo.**
- Checkpoint: `vrdet3x_1.pt`, bản tốt nhất của OpenOBB3-x trên Wall_Color (F31: 0.720 / 0.474 thước YOLO).
- Ảnh: một trang Wall_Color thật (`wall_train_0151.png`, 2381×1683, chạy ở 1280).
- Phần cứng: CPU.
- Script: `research/analysis/analyze_v3*.py`; số thô ở `research/analysis/*.json`.
- YOLO11x-obb / YOLO26x-obb dựng từ file yaml của Ultralytics, chỉ để đếm FLOPs và đọc mã nguồn (nghiên cứu; không vào `openobb/`).

**Giới hạn.**
- Một trang ảnh và một checkpoint. Các số đo cơ chế (FLOPs, chuỗi P5 bị nổ, độ phân giải, vùng nhìn) không phụ thuộc nhiều vào ảnh.
- Tỉ lệ nhầm class thì cần cả tập val. Confusion matrix của F35 đã làm việc đó.

---

## 1. Bài toán CAD qua số đo

| Class | Số phát hiện (conf 0.25) | Kích thước điển hình ở 1280 px | Tầng ra kết quả P3/P4/P5 |
|---|---|---|---|
| junction | 105 | 11 × 11 px | 86% / 14% / 0 |
| wall | 59 | dài ~72, dày ~7 px | 73% / 10% / 17% |
| door | 19 | 33 × 34 px | 100% / 0 / 0 |
| **wall_300** | 14 | **13 × 7.5 px** | 86% / 14% / 0 |
| note, double_door | 2 + 2 | 300 × 100, 50 × 40 px | |

- **Vật thể tí hon chiếm đa số.** wall_300 và junction cỡ 1–2 ô của lưới stride 8. Gần như mọi thứ được phát hiện ở P3.
- **Phân loại dựa chủ yếu vào ngữ cảnh.** Gradient của logit class theo ảnh vào (vùng nhìn thực tế) cho thấy:
  - nửa bằng chứng nằm trong bán kính khoảng 36–55 px quanh điểm dự đoán, 90% trong bán kính 120–186 px;
  - **75–93% bằng chứng nằm ngoài chính box của vật**.

  Với vật 11 px, model "nhìn hàng xóm" để quyết định. Vùng nhìn đã đủ rộng (một cánh cửa cỡ 33 px nằm gọn trong đó).
- **Phân loại lưỡng lự đúng chỗ thua YOLO.**
  - 43% wall_300 có khoảng cách xác suất top1−top2 < 0.2; class thứ hai là wall (11/14) hoặc junction (3/14).
  - Junction: class thứ hai là wall (69) hoặc wall_300 (36).
  - Box thì khít và ổn định: dịch ảnh 1–7 px làm tâm box lệch chỉ **0.2–0.5 px**, độ dày 0.3–0.8 px. Phân phối DFL độ dày có một đỉnh, độ bất định khoảng 2.7 px.

  → **Định vị không phải nút thắt. Nút thắt là phân biệt các vật nhỏ trông giống nhau** (wall_300 / junction / wall, slide_door / wall).

---

## 2. HGNetv2-B5 (backbone, 33.2M tham số, 427 GFLOPs = 47% model)

### Thiết kế gốc và lý do (PP-HGNetV2 của Baidu, cho phân loại ImageNet và detection ảnh tự nhiên, tối ưu tốc độ GPU)

| Thành phần | Làm gì | Vì sao |
|---|---|---|
| **Stem** | conv 3×3 s2 → hai nhánh (2 conv 2×2 / max-pool 2×2) nối lại → conv 3×3 s2 → 1×1. Xuống stride 4 sau khoảng 4 lớp | Hạ độ phân giải sớm cho rẻ (ảnh tự nhiên 224 px dư thừa điểm ảnh). Nhánh max-pool giữ đỉnh tín hiệu (cạnh mạnh) mà conv stride làm mờ |
| **HG_Block** | 6 conv nối tiếp; đầu ra của mọi lớp được nối một lần rồi gộp bằng 1×1 (one-shot aggregation, từ VoVNet) | Như DenseNet là giữ đặc trưng nhiều độ sâu, nhưng nối một lần nên ít truy cập bộ nhớ: GPU chạy nhanh |
| **ESE** (effective squeeze-excitation) | Lấy trung bình toàn ảnh, 1×1, sigmoid, nhân lại theo kênh | Chọn kênh theo nội dung cả ảnh. Không có thông tin vị trí |
| **LightConv** ở stage 3–4 | 1×1 rồi depthwise 5×5 | Mở rộng vùng nhìn rẻ ở tầng sâu |
| **Downsample giữa stage** | depthwise 3×3 stride 2, không activation | Rẻ |
| **Phân bổ kênh** | 128 → 512 → 1024 → **2048** kênh ở stride 4 / 8 / 16 / 32 | Kiểu mạng phân loại: ngữ nghĩa nằm ở tầng sâu, độ phân giải thấp. Vật thể ảnh tự nhiên lớn |
| **BN đóng băng** (`freeze_norm`), stem đóng băng | Giữ thống kê ImageNet | Detection batch nhỏ không ước lượng BN tốt |

### Số đo trên CAD

| Đo | Kết quả |
|---|---|
| FLOPs theo stage | stem 8 (1%), stage1 stride-4: 53 (6%), stage2 stride-8: 155 (17%), stage3 stride-16: 153 (17%), stage4 stride-32: 58 (6%) |
| **Đầu ra stage 4 (P5)** | **Độ lớn trung bình 3566** (P3/P4: 0.4–0.5). Có ở **cả 3 checkpoint v3** (3566 / 53 / 25) |
| **Đặt P5 của backbone = 0** | **Detection không đổi** (59/59 tường, 105/105 junction, 14/14 wall_300); điểm số lệch tối đa 0.02 |
| Độ ổn định khi dịch ảnh | Tâm lệch 0.2–0.5 px, dài 0.3–1.2 px, dày 0.3–0.8 px. Stem stride-4 **không** làm hỏng nét mảnh |

**Kết luận.**
1. **Stage 4 đã chết.** BN đóng băng không chặn được độ lớn tăng khi train fine-tune với lr cao. Encoder chuẩn hoá lại (running var 3×10²⁶) thành gần hằng số.
   Model thực chất chỉ dùng stride 4–16 của backbone. Phần ngữ nghĩa "đắt" nhất của ImageNet (2048 kênh) không đóng góp gì.
2. Cách phân bổ kênh của mạng phân loại **ngược** với nhu cầu CAD: vật nhỏ, cần độ phân giải cao và ngữ cảnh rộng, nhưng không cần 2048 kênh ở 1/32.
3. Stem và max-pool giữ nét tốt (dịch ảnh không gây răng cưa). Ý tưởng **nhánh max-pool** nên giữ.
4. ESE chỉ có thông tin toàn ảnh nên không giúp phân biệt vật nhỏ theo hàng xóm.
5. BN đóng băng + fine-tune lr cao là nguồn mất ổn định (lần 2 sau F22 / F31).

---

## 3. LSK (Large Selective Kernel; LSKNet, ICCV 2023; OpenOBB tự viết lại; 8.5M tham số, 64 GFLOPs = 7%)

### Thiết kế gốc và lý do
- **Bài toán gốc:** ảnh viễn thám. Vật cần ngữ cảnh ở tầm khác nhau (tàu cần thấy cảng, xe thì không).
- **Cách làm:**
  - Kernel lớn tách thành chuỗi depthwise: 5×5, rồi 7×7 giãn 3 → vùng nhìn khoảng 23 ô.
  - Hai nhánh (gần / xa) được trộn bằng một **cổng theo vị trí**, tính từ trung bình và max theo kênh.
  - Mỗi điểm ảnh tự chọn tầm ngữ cảnh.
- **Bản của OpenOBB:** đặt sau backbone, dạng `x + x·W_out(...)`, W_out khởi tạo 0.

### Số đo trên CAD
| Đo | P3 | P4 | P5 |
|---|---|---|---|
| Cổng nhánh xa: trên nét / trên nền trắng | 0.40 / 0.38 | 0.18 / 0.18 | **0.00 / 0.00 (bão hoà)** |
| Thay đổi tương đối của feature: trên nét / trên nền | 12.5% / 12.7% | 36% / 45% | **×1.5·10⁶** |
| FLOPs | Phần lớn ở ba conv 1×1 (60 G); kernel lớn thật chỉ khoảng 3 G | | |

**Kết luận.**
1. **Cổng không chọn theo nội dung.** Trên nét và trên nền gần như bằng nhau. LSK hoạt động như một bộ lọc cố định, không "chọn kernel".
2. Dạng nhân `x·(1+m)` không bị chặn. Ở P5 nó **khuếch đại** chuỗi nổ của backbone lên 10¹⁰.
3. Gần 95% chi phí là chiếu kênh 1×1, không phải kernel lớn.
4. Mức +1.3 mAP50 đo trên FloorPlanCAD (c6, v1) có thể đến từ việc thêm tham số và lọc cố định chứ không phải cơ chế chọn.
   → **Ý tưởng "ngữ cảnh rộng rẻ bằng depthwise lớn" đáng giữ; cổng chọn và phép nhân không chặn thì không.**

---

## 4. Hybrid encoder (RT-DETR / D-FINE: AIFI + CCFF; 20.7M tham số, 371 GFLOPs = 41%)

### Thiết kế gốc và lý do (RT-DETR, CVPR 2024, COCO)
- **AIFI:** self-attention chỉ trên S5 (1/32).
  - Lý do của tác giả: S5 mang ngữ nghĩa cấp cao. Attention trên S3 / S4 đắt (bình phương số token) mà ít lợi trên COCO.
  - Tách tương tác trong cùng tầng (attention, đắt) khỏi trộn giữa các tầng (CNN, rẻ).
- **CCFF:** FPN từ trên xuống + PAN từ dưới lên, bằng khối RepNCSPELAN (CSP + ELAN, nhiều đường gradient), 384 kênh.

### Số đo trên CAD
| Đo | Kết quả |
|---|---|
| Đầu vào của AIFI | Gần hằng số (P5 chết, mục 2) |
| Khoảng cách attention trung bình | **682 px**, so với 667 px nếu attention đều tuyệt đối; 95% trọng số rơi xa hơn 160 px; entropy 0.76 |
| FLOPs | AIFI chỉ 11 G (1%). **Khối FPN ở P3 (fpn_blocks.1): 211 G = 23% cả model**, khối nặng nhất. 384 kênh trên lưới 160×160 |

**Kết luận.**
1. Mảnh "toàn cục" duy nhất (AIFI) không làm gì: đầu vào hằng số, attention đều. Ngữ cảnh mà class thật sự dùng (mục 1, bán kính 40–170 px) đến từ chồng conv ở P3 / P4, không phải từ attention.
2. Phần đắt là trộn đa tầng bằng CSP rộng ở độ phân giải cao. Đắt vì 384 kênh ở 1/8, không vì nhu cầu bài toán.
3. Ý tưởng tách "attention ở tầng thấp, CNN ở tầng cao" dựa trên giả định COCO (vật lớn, ngữ nghĩa ở 1/32). Với CAD (vật 11 px, ngữ cảnh 40–170 px), **ngữ cảnh cần có ở 1/8–1/16, không phải 1/32**.

---

## 5. Head dense oriented của OpenOBB3 (của mình; 2.2M tham số, 48 GFLOPs = 5%)

### Thiết kế
- Mỗi điểm P3–P5 dự đoán class (tháp depthwise nhẹ) và hộp xoay. Hộp xoay gồm: góc, lệch ngang, hai khoảng cách dọc trục dạng DFL (bin = stride), độ dày DFL (bin = stride/2).
- Gán nhãn TAL xoay, NMS trên ProbIoU.

### Số đo trên CAD
| Đo | Kết quả |
|---|---|
| DFL dọc trục bão hoà (khối lượng ở 2 bin cuối) | 0–0.001: tầm với đủ |
| Độ dày | Một đỉnh; bất định khoảng 2.6–2.7 px (junction, wall_300) |
| Biên phân loại | wall_300: 43% có top1 − top2 < 0.2 |

**Kết luận.**
- Phần hộp làm tốt.
- Phần class quá nhẹ so với độ khó. YOLO26x dành **20% FLOPs** cho head, OpenOBB3 chỉ 5%, trong khi lỗi chính là phân loại vật nhỏ.

---

## 6. YOLO11x / YOLO26x (đối thủ; đọc mã nguồn Ultralytics 8.4.174)

### Thiết kế và lý do
| Thành phần | Làm gì | Vì sao |
|---|---|---|
| Stem | Hai conv 3×3 s2 (→ 1/4) | Đơn giản, rẻ |
| **C3k2** (CSP + ELAN) | Chia kênh làm đôi; một nửa đi qua chuỗi bottleneck, mọi đầu ra trung gian được nối rồi gộp 1×1. Ở tầng sâu bottleneck là C3k (khối CSP lồng). Tuỳ chọn thêm PSA attention | Nhiều đường gradient với chi phí thấp; nửa kênh đi thẳng giữ thông tin gốc |
| **SPPF** | 1×1 rồi max-pool 5×5 lặp 3 lần (tương đương pool 5/9/13), nối lại. YOLO26 có thêm nối tắt | Vùng nhìn đa tầm gần như miễn phí. **Max-pool giữ nét mảnh** (đỉnh tín hiệu) |
| **C2PSA** | Self-attention trên nửa số kênh ở 1/32. Vị trí mã hoá bằng depthwise 3×3 trên V | Ngữ cảnh toàn cục rẻ ở tầng thấp nhất |
| Neck | PAN-FPN với C3k2 | |
| **Head OBB26** | Ba nhánh conv riêng cho hộp / class / góc trên P3–P5. **YOLO26 bỏ DFL** (`reg_max=1`), góc ra thẳng | Head nặng (20% FLOPs). Bỏ DFL cho dễ export và end-to-end |
| **End-to-end (YOLO26)** | Hai head: một-nhiều (TAL top-10) và một-một (top-7 → 1). Trọng số loss một-nhiều giảm tuyến tính 0.8 → 0.1 theo epoch (ProgLoss); suy luận dùng head một-một, **không NMS** | Ổn định lúc đầu nhờ nhiều mẫu dương, cuối cùng ra một box cho mỗi vật |
| **STAL** | Khi chọn ứng viên, box có cạnh nhỏ hơn `stride_val` được phóng lên `stride_val` quanh tâm | **Vật tí hon luôn có đủ điểm dương.** Đúng với wall_300 / junction (11–13 px) |
| Pretrain | `yolo26x-obb.pt` đã train trên **DOTA** (ảnh viễn thám, vật nhỏ xoay) | Có sẵn "tiên nghiệm" về vật nhỏ xoay. Đây là yếu tố gây nhiễu khi so với v3 (init COCO) |

### Phân bổ tính toán theo stride ở 1280 px (GFLOPs; đo được)
| Stride | YOLO26x-obb (905 G) | OpenOBB3-x (909 G) |
|---|---|---|
| 1/2–1/4 | 115 (13%) | 61 (7%) |
| 1/8 | 234 (26%) | khoảng 425 (47%; riêng khối FPN P3 là 211) |
| 1/16 | 289 (32%) | khoảng 283 (31%) |
| 1/32 | 88 (10%), rộng tối đa 768 kênh | khoảng 89 (10%), rộng 2048 kênh; trong đó khoảng 75 G (stage 4 + LSK P5 + chiếu P5 + AIFI) **không đóng góp** |
| Head | **179 (20%)** | 48 (5%) |

**Kết luận.**
- Cùng ngân sách (khoảng 900 G), YOLO dồn tính toán vào **độ phân giải cao và head**. OpenOBB3 dồn vào một khối trộn P3 rất rộng và một backbone kiểu phân loại có tầng sâu chết.
- Các ý tưởng của YOLO hợp với CAD: max-pool giữ nét (SPPF), CSP giữ một nửa kênh đi thẳng, head nặng, STAL cho vật tí hon, ProgLoss + head một-một.
- Ý tưởng không cần cho CAD: tầng 1/32 lớn.

---

## 7. Nguyên lý rút ra (cơ sở cho kiến trúc mới)

| # | Nguyên lý | Bằng chứng |
|---|---|---|
| N1 | **Tính toán đặt ở 1/4–1/16**; tầng 1/32 chỉ cần mỏng (hoặc bỏ) | 86–100% detection ở P3; stage 4 + AIFI chết mà detection không đổi; YOLO chỉ 10% ở 1/32 |
| N2 | **Ngữ cảnh 40–170 px phải có ở 1/8–1/16**, không phải attention ở 1/32 | 75–93% bằng chứng class nằm ngoài box; r50 khoảng 40 px; AIFI ở 1/32 bằng phẳng |
| N3 | **Ngữ cảnh rẻ:** bản vẽ phần lớn theo trục ngang / dọc. Trộn theo hàng / cột (strip, axial) hoặc depthwise kernel lớn phù hợp hơn attention toàn ảnh | Tường H / V chiếm đa số; LSK cho thấy depthwise lớn rẻ (3 G) |
| N4 | **Ổn định số học là yêu cầu thiết kế:** chuẩn hoá được train (không đóng băng), cổng bị chặn, residual có layer-scale. Không nhân `x·(1+m)` không chặn | P5 nổ ở cả 3 checkpoint; LSK khuếch đại tới 10¹⁰ |
| N5 | **Giữ nét:** nhánh max-pool khi hạ độ phân giải là đủ; không cần cơ chế chống răng cưa phức tạp | Dịch ảnh lệch < 1 px |
| N6 | **Head class mạnh hơn và có quan hệ giữa các vật:** wall_300 / junction / wall phân biệt bằng vật bên cạnh (khung cửa, giao tường) | Biên < 0.2 ở 43% wall_300; nhầm sang junction 32% (F35); YOLO head 20% FLOPs |
| N7 | **Gán nhãn cho vật tí hon** (kiểu STAL) | wall_300 13 × 7.5 px; YOLO26 làm STAL và recall wall_300 0.87 so với 0.33 |
| N8 | **Pretrain vẫn cần, nhưng nên đúng miền:** tầng sâu nhất của ImageNet không được dùng trong cả 3 checkpoint (do nổ số, chưa chứng minh là vô dụng về nguyên tắc); giá trị chắc chắn của pretrain là các tầng đầu (nét, góc) | Stage 4 chết; v2 train từ đầu trên 250 ảnh thì hỏng (F25) |

## 8. Hệ quả cho kiến trúc mới (để thảo luận trước khi viết code)

Một backbone + neck **thiết kế lại cho CAD**, không ghép từ mạng ImageNet:
- Rộng ở 1/4–1/8, vừa ở 1/16, mỏng ở 1/32 (N1).
- Khối cơ bản kiểu CSP nửa kênh đi thẳng. Hạ độ phân giải có nhánh max-pool (N5).
- Ngữ cảnh dài 40–200 px ở 1/8–1/16 bằng trộn theo hàng / cột và depthwise lớn (N2, N3).
- Chuẩn hoá được train, cổng bị chặn, layer-scale (N4).
- Head class nặng hơn, có lớp quan hệ giữa ứng viên. Gán nhãn kiểu STAL (N6, N7).

Đánh đổi cần quyết:
- Backbone mới **không có pretrain ImageNet**. Theo N8 và F25, cần pretrain riêng trên ảnh bản vẽ.
  Cách khả thi: tự giám sát (masked image modeling cho mạng conv, kiểu SparK) trên nhiều trang bản vẽ chưa gán nhãn.
  Có thể khởi tạo stem và hai stage đầu từ phần HGNetv2 còn hữu ích (các tầng nét / góc), phần còn lại học từ pretrain CAD.
- Chi phí: một đợt pretrain tự giám sát trên Colab (ước lượng vài chục CU, tuỳ số trang).

Việc ngay được, không cần kiến trúc mới:
- **v3 hiện tại đang mang khoảng 8% FLOPs chết** (stage 4 + LSK P5 + chiếu P5 + AIFI). PAN P5 vẫn sống nhờ thông tin đi lên từ P3 / P4 (17% detection tường).
- **Lỗi mất ổn định cần sửa trong recipe:** lr của backbone / encoder và BN đóng băng ở P5.
