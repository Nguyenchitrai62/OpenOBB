# VRDet5-x: dense oriented + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ (bản 5, 2026-10-09)

Dùng: CLI giữ nguyên, chỉ đổi `model=`.

```bash
openobb train data=/content/data.yaml model=vrdet5x epochs=100 imgsz=1280
```

- Mặc định: lr 5e-4, phần ViT × 0.2, batch 8.
- Init: DINOv2 ViT-B (Apache) + encoder D-FINE-X COCO. Head và bộ chấm lại khởi tạo mới.
- Tuỳ chọn:

| Tuỳ chọn | Tác dụng |
|---|---|
| `weights=<best.pt của v3>` | Nạp head của v3 (cùng tên tham số). Nên dùng thêm `backbone=hgnet` để nạp cả backbone/encoder v3 |
| `backbone=hgnet` | Bỏ DINOv2, nhanh hơn |
| `geo_cls=False`, `relate=False` | Tắt từng phần mới để đo tác dụng riêng |

## 1. Bằng chứng dẫn tới v5 (Wall_Color, thước YOLO)

| | all | wall_300 | Class hiếm (slide / double / note) | Tốc độ | Bài học |
|---|---|---|---|---|---|
| YOLO26x | 0.746 / 0.499 | 0.824 / 0.441 | .345 / .494 / .764 | – | Mốc |
| v3 (head dense) | 0.712 / 0.467 | **0.599** / 0.256 | **.273 / .589 / .744** | 23 ms | wall_300 chiếm khoảng 95% khoảng cách. Class hiếm tốt |
| v4 (DINOv2 + decoder) | 0.663 / 0.457 | **0.683** / 0.366 | .146 / .471 / .650 | 38 ms | wall_300 tốt hơn v3, độ khít của wall tốt hơn. Class hiếm kém (decoder mỗi object 1 mẫu dương) |

**v5 = đầu ra của v3** (nhiều mẫu dương: tốt cho class hiếm, nhanh) **+ thứ đã giúp v4** (feature DINOv2, ngữ cảnh) **+ 2 phần mới nhắm vào wall_300.**

## 2. Các phần của VRDet5-x (127.9M tham số)

| # | Phần | Làm gì | Tham số | Nguồn |
|---|---|---|---|---|
| 1 | **DINOv2 ViT-B/14** (registers) + adapter (P3/P4/P5, nhánh chi tiết stride 8) | Feature tự giám sát (giống v4) | 86.6 + 8.2M | Code VRDet; weights DINOv2 Apache |
| 2 | LSK ×3 | Trường nhìn thích nghi | 8.5M | VRDet |
| 3 | Hybrid encoder (AIFI + CCFF) | Ngữ cảnh toàn cục + trộn đa tầng | 20.7M | Code D-FINE Apache, weights COCO |
| 4 | **Head dense oriented của v3** (DFL 2 đầu mút + độ dày, lệch ngang, mục tiêu theo hệ trục dự đoán, TAL đa tầng) | Đầu ra chính; nhiều mẫu dương, tốt cho class hiếm | 2.2M | VRDet (v3) |
| 5 | **Phân loại theo hình học (mới)** | Nhánh class nhận 7 số đo mà nhánh box vừa dự đoán ở cùng điểm (đã detach); xem bảng dưới | khoảng 0.01M | VRDet |
| 6 | **Chấm lại theo quan hệ (mới)** | 600 ứng viên khác nhau tốt nhất mỗi ảnh "nhìn nhau" qua 2 lớp transformer (d=256) rồi **sửa logit class** (cộng dư, khởi tạo 0); xem ví dụ dưới | 1.75M | VRDet (ý tưởng Relation Networks / self-attention của DETR, áp lên ứng viên dense) |
| 7 | Đầu ra = các ứng viên đã chấm lại → Fast-NMS ProbIoU 0.7 | | 0 | VRDet |

**Phân loại theo hình học (phần 5).** 7 số đo đưa vào nhánh class:

| Số đo | Ghi chú |
|---|---|
| log độ dày | |
| log chiều dài | |
| sin 2θ, cos 2θ | Góc |
| lệch ngang | |
| độ bất định của độ dày | |
| log stride | Tầng đang xét |

Các số này đi qua conv 1×1 khởi tạo 0, rồi cộng vào nhánh class. wall và wall_300 khác nhau chủ yếu ở độ dày; ở v3, nhánh class không nhìn thấy số đo này.

**Chấm lại theo quan hệ (phần 6).** Mỗi ứng viên mang feature encoder ở vị trí của nó, cộng box và xác suất class. Nhờ "nhìn nhau", model có thể học các quy tắc kiểu:
- "bức tường này dày hơn các bức tường nó giao" → wall_300;
- "cửa này không có cung mở" → slide_door.

Chi phí nhỏ hơn nhiều so với decoder 6 lớp của v4.

**Khởi tạo 0 có tác dụng gì:** lúc bắt đầu, phần 5 và 6 cho ra đúng kết quả của v3 (đã kiểm bằng test). Mô hình chỉ học thêm phần sửa, nên không tệ hơn v3 lúc đầu.

**Loss:** loss của v3 (QFL, 1 − ProbIoU, DFL, lệch ngang, góc) cộng **`loss_rel`**: QFL trên logit đã sửa. Đích là ProbIoU với GT trùng nhiều nhất nếu ≥ 0.5, ngược lại là nền.

## 3. Tận dụng bao nhiêu cái có sẵn

| | v3 | v4 | **v5** |
|---|---|---|---|
| Code tự viết (tỉ lệ tham số) | 17% | khoảng 78% | **khoảng 84%** (ViT + adapter + LSK + head + phần mới) |
| Code D-FINE (Apache) | 83% | khoảng 22% | **khoảng 16%** (encoder) |
| Weights pretrained | COCO | DINOv2 + COCO | **DINOv2 + COCO encoder**; head có thể lấy từ v3 |
| Phần mới | head dị hướng | lai dense + DETR | **phân loại theo hình học, chấm lại theo quan hệ** |
| License | Apache notice | Apache notice | Apache notice. **Thương mại được** |

## 4. Đọc log

```
Epoch  GPU_mem  cls_loss  box_loss  dfl_loss  angle_loss  rel_loss  Instances  Size
```

- `rel_loss` là loss của bộ chấm lại. Phải giảm, nếu không thì phần quan hệ không học được gì.
- **Mới ở mọi run:** cuối run in dòng `Most frequent errors (true -> predicted ...)`, và lưu `confusion_matrix.json`.
  Đọc ra được, ví dụ, `wall_300 -> wall: 37 (26% of wall_300)`, tức wall_300 bị gọi nhầm thành wall. `openobb val` cũng in phần này.

## 5. Rủi ro

- **Nhiều thay đổi cùng lúc:** backbone, phân loại theo hình học, chấm lại theo quan hệ. Để tách tác dụng:
  - `backbone=hgnet` = v3 + 2 phần mới;
  - `relate=False` hoặc `geo_cls=False` để tắt từng phần mới.
- Đầu ra giới hạn ở 600 ứng viên khác nhau mỗi ảnh; tăng bằng `rel_k=`. Data hiện tại trung bình 65 object/ảnh.
- DINOv2 làm chậm hơn v3. Dự kiến giữa v3 (23 ms) và v4 (38 ms); chưa đo.
- LR: v5 là kiểu dense như v3. Mặc định 5e-4; lr 1e-3 từng làm nổ BN ở encoder (F31).

## 6. Đã kiểm chứng tới đâu

- CPU, `tests/test_vrdet5.py` (90 test toàn repo pass):
  - **head v5 lúc khởi tạo cho đúng kết quả của head v3** (nạp được checkpoint v3);
  - đích chấm lại;
  - train/eval với HGNet và DINOv2, gradient tới phần mới;
  - chỉ xuất ứng viên đã chấm lại;
  - CLI `model=vrdet5x`, nạp `best.pt`;
  - luồng train → val → predict (2 vòng lặp);
  - confusion matrix bắt đúng lỗi "wall_300 → wall".
- **Chưa chạy GPU / data thật.**
