# Kế hoạch nghiên cứu: kiến trúc OBB mới (VRDet)

Cập nhật: 2026-10-07. Trạng thái: **bản thảo v1**. Các mốc số liệu tham chiếu đều đã ghi nguồn. Mục nào chưa có nguồn thì là giả thuyết.

## 1. Bài toán đặt ra

Tạo một **kiến trúc detector OBB mới** theo đúng quy trình của YOLO:
thiết kế kiến trúc → pretrain trên dataset public → công bố benchmark → người dùng finetune trên data riêng.

- Ưu tiên 1: **độ chính xác**. Phải vượt YOLO-OBB cùng cỡ trên benchmark chuẩn.
- Ưu tiên 2: **tốc độ** gần YOLO (chấp nhận chậm hơn đôi chút).
- Ứng dụng đích: bản vẽ CAD vector PDF (AI_Takeoff), nơi ngữ nghĩa và quan hệ giữa các object quan trọng hơn hình dạng cục bộ.
- Data public chỉ dùng cho nghiên cứu/benchmark. Triển khai thật sẽ finetune trên data nội bộ có nhãn.
- Code của kiến trúc: tự viết hoặc lấy từ nguồn MIT/Apache-2.0. Không dùng code Ultralytics (AGPL), chỉ chạy nó làm baseline đo đạc.

## 2. Giao thức benchmark

### 2.1 Benchmark chính: DOTA-v1.0 (giống YOLO-OBB)

- 15 class, ảnh vệ tinh 800–4000 px, rất nhiều object nhỏ, dày đặc, xoay tuỳ ý.
- Tiền xử lý chuẩn: cắt tile 1024×1024 overlap 200. Multi-scale thì resize 0.5/1.0/1.5 rồi mới cắt.
- **Ablation**: train trên `train`, đánh giá `val` tại local bằng rotated mAP50 (metric của DOTA devkit). Có thêm mAP50-95.
- **Số công bố cuối cùng**: train trên `train+val`, nộp kết quả `test` lên server đánh giá chính thức của DOTA (user cần tạo tài khoản).

### 2.2 Mốc so sánh (phải vượt)

YOLO26-OBB, DOTA-v1 test, 1024px. Nguồn: [docs.ultralytics.com/tasks/obb](https://docs.ultralytics.com/tasks/obb/)

| Model | Params | FLOPs | mAP50 | mAP50-95 | T4 TensorRT10 |
|---|---|---|---|---|---|
| YOLO26n-obb | 2.4M | 14.8G | 78.9 | 52.4 | 2.8 ms |
| YOLO26s-obb | 9.8M | 56.7G | 80.9 | 54.8 | 4.9 ms |
| YOLO26m-obb | 21.2M | 184.9G | 81.0 | 55.3 | 10.2 ms |
| YOLO26l-obb | 25.6M | 232.4G | 81.6 | 56.2 | 13.0 ms |
| YOLO26x-obb | 57.6M | 520.1G | 81.7 | 56.7 | 30.5 ms |

Real-time DETR-OBB (Apache-2.0), DOTA-v1 test, 72 epoch, 2×2080Ti. Nguồn: [arXiv 2603.15497](https://arxiv.org/abs/2603.15497), code [ai4rs](https://github.com/wokaikaixinxin/ai4rs)

| Model | Params | FLOPs | FPS (2080Ti) | AP50 | Ghi chú |
|---|---|---|---|---|---|
| O2-DFINE-s | 10M | 60G | 297 | 76.14 | single-scale |
| O2-RTDETR-R50 | 42M | 339G | 119 | 78.45 | single-scale |
| O2-DEIM-R18 | 20M | 147G | 233 | 79.49 | multi-scale |
| O2-DEIM-R50 | 42M | 339G | 119 | 80.15 | multi-scale |

Nhận xét: họ DETR-OBB real-time hiện vẫn **kém YOLO26 khoảng 1.5–2.5 điểm mAP50** ở cùng cỡ. Đây là khoảng trống để kiến trúc mới lấp và vượt.

### 2.3 Benchmark phụ

| Dataset | Vai trò | Ghi chú |
|---|---|---|
| DIOR-R | lặp nhanh (ảnh 800², 20 class) | rẻ hơn DOTA nhiều lần, dùng cho ablation sớm |
| HRSC2016 | object rất dài, tỉ lệ cạnh lớn | kiểm tra góc/aspect ratio, gần với line/wall trong CAD |
| FloorPlanCAD | bản vẽ CAD thật, có vector SVG | benchmark domain CAD + nhánh vector |
| `dataset_obb_train_v3` | floorplan 16k ảnh, 7 class | data có sẵn trong repo |

Tất cả chỉ dùng cho nghiên cứu (DOTA: học thuật; FloorPlanCAD: CC BY-NC).

### 2.4 Quy tắc so sánh công bằng

- Cùng imgsz, cùng split, cùng số epoch (hoặc ghi rõ khác ở đâu), cùng cách đo FPS (TensorRT FP16, batch 1, cùng GPU).
- Mỗi thay đổi kiến trúc = một ablation riêng, có ≥1 seed lặp lại nếu Δ < 0.5 điểm.
- Không chọn checkpoint trên tập test.

## 3. Bài học từ vòng trước (CAD-VLMDet)

Kiến trúc tự chế, train from scratch, assigner/loss/evaluator tự viết, nên thua YOLO11x rất xa (mean AP50 0.144 vs mAP50-95 0.585) và door collapse về 0. Từ đó:

1. **Bước 0 luôn là reproduce**: dựng baseline DETR-OBB và tái hiện số công bố (±1 điểm) trước khi thêm bất kỳ module mới nào.
2. Dùng metric chuẩn (DOTA devkit), không tự chế metric.
3. Mỗi lần chỉ thay một thứ.

## 4. Hướng kiến trúc: VRDet

Tên làm việc: **VRDet**. Lõi là detector OBB raster tổng quát (benchmark được trên DOTA). Nhánh vector là **plug-in** riêng cho CAD.

### 4.1 Khung nền

Real-time DETR-OBB kiểu D-FINE/DEIM (Apache-2.0):
- backbone CNN pretrained,
- hybrid encoder (AIFI + CCFM),
- decoder ~6 lớp, query selection,
- box (cx, cy, w, h, θ) refine bằng phân phối,
- dense O2O matching, NMS-free.

Chọn khung này vì nó có **suy luận quan hệ giữa object** (self-attention giữa các query), đúng thứ YOLO thiếu cho dữ liệu ngữ nghĩa cao.

### 4.2 Giả thuyết đóng góp mới (mỗi cái là một ablation)

| Mã | Giả thuyết | Lý do / bằng chứng gợi ý | Kiểm chứng |
|---|---|---|---|
| H1 | **Angle-aware matching & loss bằng phân phối Gaussian** (ProbIoU/KLD trong cả cost Hungarian lẫn loss) | tránh discontinuity của góc ở biên 0/90/180°; YOLO-OBB đã dùng ProbIoU cho loss | Δ mAP50-95 trên DOTA val, đặc biệt class dài (bridge, harbor) |
| H2 | **Oriented deformable attention**: điểm sampling của decoder xoay theo θ dự đoán và co giãn theo (w, h) | deformable attention gốc lấy mẫu theo hộp thẳng, lệch với object xoay mảnh | Δ trên HRSC2016 và DOTA class dài |
| H3 | **Strip / large-kernel trong backbone hoặc encoder** cho object dài mảnh | object dài (bridge, ship, wall, pipe line trong CAD) cần receptive field dị hướng | Δ trên HRSC + class dài; đo chi phí FPS |
| H4 | **Dense proposal head kiểu YOLO + relation decoder**: head dense một-nhiều vừa làm giám sát phụ vừa sinh proposal cho decoder | ảnh dày (DOTA 100+ object/tile, CAD 150–400) làm DETR thiếu recall; Co-DETR/DEIM cho thấy giám sát dense giúp hội tụ | recall@1000 + mAP; số query cần thiết |
| H5 | **Geometric relation bias** trong self-attention của query (vị trí tương đối, góc tương đối, khoảng cách theo cạnh) | quan hệ "cửa nằm trên tường", "xe xếp hàng trong bãi" là hình học tương đối | Δ mAP ở class phụ thuộc ngữ cảnh |
| H6 | **Global context token xuyên tile**: thumbnail cả ảnh → vài token toàn cục, đưa vào mỗi tile | cắt tile 1024 làm mất ngữ cảnh toàn ảnh (DOTA 4000 px; bản vẽ CAD cả sheet có legend/title block) | Δ mAP trên ảnh lớn; chi phí FPS nhỏ |
| H7 | **Plug-in vector cho CAD**: primitive tokens từ PDF (nét, layer, text) cross-attend vào feature raster | AI_Takeoff: FP và TP giống hệt trên ảnh, chỉ phân biệt được qua layer/text/topology | FloorPlanCAD + data nội bộ |

Thứ tự ưu tiên ban đầu: H1 → H4 → H2 → H3 → H5 → H6 (lõi raster, benchmark DOTA) rồi mới H7 (domain CAD).

### 4.3 Họ model

Theo YOLO, kiến trúc phải scale được thành n/s/m/l/x bằng depth/width multiplier. Mục tiêu sau cùng: **mỗi cỡ VRDet có mAP50 ≥ YOLO26 cùng cỡ + ≥1 điểm**, latency ≤ 1.3× YOLO26 cùng cỡ.

## 5. Lộ trình thí nghiệm

| Bước | Nội dung | Tiêu chí qua | Ước tính GPU |
|---|---|---|---|
| E0 | Hạ tầng: Colab CLI, data DOTA/DIOR-R trên Drive, evaluator DOTA, checkpoint + resume | eval GT=pred cho mAP=1.0; resume đúng sau khi kill | nhỏ |
| E1 | Baseline: YOLO26s-obb (Ultralytics, chỉ đo) + DETR-OBB baseline (code Apache) trên DIOR-R/DOTA val, schedule ngắn | DETR-OBB tái hiện số công bố ±1 điểm | trung bình |
| E2 | Ablation H1, H4 (cỡ s, schedule ngắn) | mỗi H: Δ ≥ +0.5 mAP50 hoặc bỏ | trung bình |
| E3 | Ablation H2, H3, H5, H6 | như trên, kèm FPS | trung bình |
| E4 | Gộp các H đã qua, train đủ schedule trên DOTA train+val, nộp test server | ≥ YOLO26 cùng cỡ | lớn |
| E5 | Scale n→x, đo TensorRT FP16 | bảng benchmark đầy đủ | lớn |
| E6 | Plug-in vector (H7) trên FloorPlanCAD/data nội bộ | Δ rõ ở class ngữ nghĩa | trung bình |

Ngân sách: Colab Pro+, ~500 compute units (2026-10-07). Đo tốc độ tiêu CU thực tế bằng `colab usage` trong E0 rồi chia ngân sách cho từng bước. Chạy ablation ngắn trên L4 hoặc A100, chỉ dùng A100/H100 cho E4/E5.

## 6. Tài liệu tham khảo chính

- YOLO26-OBB benchmark: https://docs.ultralytics.com/tasks/obb/
- O2-DFINE / O2-RTDETR / O2-DEIM (real-time DETR-OBB): https://arxiv.org/abs/2603.15497. Code Apache-2.0: https://github.com/wokaikaixinxin/ai4rs (có 30+ detector OBB trên MMRotate)
- D²Q-DETR (DETR-OBB, 81.24 mAP DOTA multi-scale): https://arxiv.org/abs/2303.00542
- AO2-DETR: https://arxiv.org/abs/2205.12785
- Symbol spotting CAD vector: SymPoint-V2, CADSpotting, VecFormer (https://arxiv.org/abs/2505.23395)
