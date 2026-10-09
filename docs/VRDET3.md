# VRDet3: encoder pretrained + head dense oriented dị hướng (bản 3, 2026-10-09)

Dùng: CLI giữ nguyên, chỉ đổi `model=`.

```bash
openobb train data=/content/data.yaml model=vrdet3x epochs=100 imgsz=1280
```

- Mặc định: backbone + encoder khởi tạo từ D-FINE-X COCO (Apache-2.0), tự tải về.
- **Tuỳ chọn:** khởi tạo từ checkpoint VRDet1 đã fine-tune trên data của bạn (bản đạt 0.63/0.35). Thêm
  `weights=/content/drive/MyDrive/Train/tu_ph/Wall_Color/model/vrdet_x.pt`. Backbone, encoder và LSK được chép
  nguyên (702 tensor), head mới khởi tạo ngẫu nhiên, class ghép theo tên.
- Fine-tune tiếp từ `best.pt` của VRDet3: `model=.../best.pt`, tự nhận kiến trúc v3.

## 1. Vì sao có v3: rút từ kết quả thật trên Wall_Color

| Bản | Thiết kế | mAP50 / mAP50-95 | Bài học |
|---|---|---|---|
| YOLO11x | CNN dense + NMS, pretrained COCO | 0.74 / 0.50 | Mốc cần vượt |
| VRDet1-X | D-FINE (DETR) + LSK, pretrained COCO | 0.63 / 0.35 | Feature pretrained tốt. Decoder là nút thắt: trần query, 1 mẫu dương/object, chậm |
| VRDet2-x | Dense segment-aware, **train từ đầu** | 0.27 / 0.10 (50 ep) | Trên 250 ảnh, pretrained quyết định. Cách hồi quy segment không làm khít chiều dọc (`end_loss` đứng ở khoảng 0.93) |

**VRDet3 = phần tốt nhất của mỗi bản:**
- Từ v1: feature pretrained và adapter LSK.
- Từ YOLO và v2: head dense nhiều mẫu dương, NMS, gán nhãn TAL theo độ dài.
- Mới: cách hồi quy dọc trục được viết lại.

## 2. Các phần của kiến trúc (bản x)

Tổng **64.6M tham số**.

| # | Phần | Làm gì | Tham số | Nguồn |
|---|---|---|---|---|
| 1 | **Backbone HGNetv2-B5**: stem và BN đóng băng như D-FINE-X | Feature đa tầng, **pretrained COCO** | 33.2M | Code D-FINE (Apache-2.0) đã port từ v1. Weights COCO Apache-2.0 |
| 2 | **LSK adapter** (selective large kernel) ở 3 tầng | Trường nhìn thích nghi theo vị trí: gần (5×5) hoặc xa (7×7 dilation 3). Khởi tạo bằng identity | 8.5M | Ý tưởng LSKNet; code VRDet tự viết (đã xác nhận c6: +1.3 mAP50 trên FloorPlanCAD) |
| 3 | **Hybrid encoder**: AIFI (attention toàn cục trên P5) + CCFF (PAN 2 chiều) | Ngữ cảnh toàn trang + trộn đa tầng, **pretrained COCO** | 20.7M | Code D-FINE (Apache-2.0) |
| 4 | **Head oriented dị hướng** P3–P5 (**mới**). Mỗi điểm dự đoán 5 nhóm số, xem bảng dưới | Đoán box ở mọi điểm, mỗi object nhiều mẫu dương, không có trần query | 2.2M | Ý tưởng DFL (GFL/YOLO); thiết kế dị hướng + lệch ngang là của VRDet |
| 5 | **Gán nhãn TAL xoay** | Top-k theo chiều dài (vật dài nhiều mẫu dương). **Mỗi tầng có một hàng ứng viên dọc mỗi object** (dải ngang ≥ 1 stride), để điểm P4/P5, vốn nhìn thấy cả hai đầu bức tường, được học object dài | 0 | TAL (TOOD/YOLO). Dải đa tầng là của VRDet |
| 6 | **Mục tiêu theo hệ trục gần góc dự đoán** (**mới**) | Hướng đoán ngược thì đổi chỗ trái/phải. Vật gần vuông được dùng cạnh ngắn làm trục. **Không nhảy gián đoạn ở tường dọc** (le90 ±90°) | 0 | VRDet |
| 7 | **Loss**: QFL (cls), 1 − ProbIoU, DFL ×3, lệch ngang tương đối độ dày, 1 − cos 2Δθ | | 0 | GFL, ProbIoU; lệch ngang lấy từ v2 (đã chạy tốt) |
| 8 | **Đầu ra**: Fast-NMS theo class trên ProbIoU (IoU 0.7), tối đa 1000 box | Giống YOLO; khung eval/predict dùng nguyên như cũ | 0 | Ý tưởng Fast-NMS (YOLACT/YOLO-OBB), tự viết |

**Head oriented (phần 4).** Mỗi điểm dự đoán:

| Đại lượng | Cách dự đoán | Ghi chú |
|---|---|---|
| Góc θ | hồi quy trực tiếp | |
| Lệch ngang tới tâm | `raw · stride`, có dấu | Điểm nằm cạnh line mảnh vẫn đoán được line |
| Khoảng cách tới **đầu sau** dọc trục | phân phối 32 bin, mỗi bin = 1 stride | Tầm với 31 stride: P3 248 px, P5 992 px, đủ cho tường dài |
| Khoảng cách tới **đầu trước** dọc trục | phân phối 32 bin, mỗi bin = 1 stride | Như trên |
| Độ dày | phân phối 32 bin, mỗi bin = stride / 2 | Mịn hơn 2 lần, cho vật mảnh |

**Khác v2 ở chỗ quan trọng:** v2 đoán "vị trí tương đối × log(chiều dài)". Hai đại lượng đó nhân với nhau, nên sai số đầu mút không giảm được. v3 đoán **hai khoảng cách độc lập tới hai đầu mút, dưới dạng phân phối**, đúng cách YOLO làm cho 4 cạnh box. Khác YOLO ở chỗ lưới bin dị hướng: dọc thô và xa, ngang mịn. YOLO dùng cùng 16 bin cho cả 4 cạnh.

## 3. Tận dụng bao nhiêu cái có sẵn

| | VRDet1 | VRDet2 | **VRDet3** |
|---|---|---|---|
| Tham số từ code D-FINE (Apache) | khoảng 86% | 0% | **83%** (backbone 33.2M + encoder 20.7M) |
| Tham số code VRDet tự viết | khoảng 14% (decoder OBB) | 100% | **17%** (LSK 8.5M + head 2.2M) |
| Weights pretrained | COCO | không | **COCO** (hoặc VRDet1 của bạn) |
| Phần mới | decoder xoay, LSK | segment head, o2o dùng chung box | head dị hướng, mục tiêu theo hệ trục dự đoán, dải ứng viên đa tầng |
| License | Apache notice | không phụ thuộc | Apache notice, như v1. **Thương mại được** |

**Nói thẳng:** v3 không phải "viết lại toàn bộ". Bằng chứng từ v2 cho thấy trên 250 ảnh, pretrained quan trọng hơn mọi thay đổi kiến trúc. Phần mới của v3 nằm đúng ở chỗ v1 yếu: decoder và độ khít của object dài.

## 4. Công thức train mặc định của `vrdet3x`

CLI tự đặt:
- **AdamW, lr 5e-4.** Backbone lr × 0.1 để giữ feature pretrained; encoder và head chạy LR đầy đủ.
- Weight decay 1e-4, clip grad 10, batch 8.
- LSK, RFS 0.1, tối đa 1000 detection.
- Augment như cũ.

**Về LR:** với backbone pretrained, LR quá cao sẽ phá feature COCO. VRDet1 từng ra NaN ở 1e-3, F22. Nên giữ mặc định, hoặc tối đa `lr0=0.001`.

## 5. Đọc log

```
Epoch  GPU_mem  cls_loss  box_loss  dfl_loss  acr_loss  angle_loss  Instances  Size
```

| Cột | Ý nghĩa |
|---|---|
| `dfl_loss` | Trung bình DFL của 2 khoảng cách dọc trục và độ dày. **Chỉ số chính về độ khít:** phải giảm đều. Đây là chỗ `end_loss` của v2 bị kẹt |
| `acr_loss` | Lệch ngang tương đối độ dày |

## 6. Đã kiểm chứng tới đâu

- **CPU** (`tests/test_vrdet3.py`, 73 test toàn repo pass):
  - decode;
  - mục tiêu theo hệ trục: tường dọc ở biên ±90°, lật hướng, vật gần vuông;
  - Fast-NMS: xoá box trùng nhưng giữ tường song song cách 40 px;
  - tên và shape của backbone/encoder/LSK khớp v1, nên weights COCO và VRDet1 nạp được;
  - CLI `model=vrdet3x`;
  - nạp `best.pt`;
  - luồng train → val → predict (2 vòng lặp).
- **Chưa chạy GPU / data thật.** Chưa có mAP và tốc độ.

## 7. Kỳ vọng, cần log kiểm chứng

- **Kỳ vọng:**
  - mAP50 ≥ v1 (0.63), vì cùng feature pretrained và có nhiều mẫu dương hơn.
  - **mAP50-95 tăng rõ ở wall/wall_300**, nhờ hồi quy dọc trục mới và ứng viên đa tầng.
  - Nhanh hơn v1, vì không còn decoder 6 lớp.
- **Rủi ro:**
  - Head dense mới khởi tạo ngẫu nhiên, cần vài epoch đầu để bắt kịp.
  - Trên class hiếm, NMS kiểu YOLO có thể thua decoder DETR về ngữ cảnh (door/slide_door/double_door).
- Cần gửi: bảng mAP theo class và dòng `Speed`.
