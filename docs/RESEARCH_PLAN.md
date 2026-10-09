# Kế hoạch nghiên cứu: kiến trúc OBB mới (OpenOBB)

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

Ghi chú license: code O2-RTDETR trong ai4rs giống 82–96% code RHINO (CC BY-NC). OpenOBB chỉ lấy ý tưởng từ paper, tự viết lại.

**Mốc mới (cập nhật 2026-10-07): RiO-DETR** ([arXiv 2603.09411](https://arxiv.org/abs/2603.09411), ECCV 2026, repo Apache-2.0, chỉ công bố baseline). DOTA-v1.0 test, single-scale (SS), train+val:

| Cỡ | RiO-DETR SS | YOLO26 SS (bảng của RiO) | YOLO26 MS (Ultralytics) | RiO params / FLOPs / T4 FP16 |
|---|---|---|---|---|
| n | 78.4 | 77.7 | 78.9 | 4.0M / 17G / 2.7 ms |
| s | 80.3 | 79.7 | 80.9 | 8.2M / 53G / 5.2 ms |
| m | 80.9 (MS 81.49) | 80.0 (MS 81.00) | 81.0 | 18.6M / 158G / 8.8 ms |
| l | 81.7 | 80.2 | 81.6 | 27.5M / 230G / 13.4 ms |
| x | 81.8 (MS 81.76) | 80.4 (MS 81.70) | 81.7 | 62.5M / 527G / 29.9 ms |

Nhận xét:
- Khi tinh chỉnh tốt, DETR-OBB đã ngang YOLO26 ở SS. Ở MS khoảng cách gần như bằng 0: +0.49 với m, +0.06 với x.
- Theo class (x, SS), RiO và YOLO26 mạnh ở những chỗ khác nhau:
  - RiO thắng các class lớn cần ngữ cảnh: BR +4.6, GTF +5.4, LV +7.3, SBF +4.5, RA +4.5.
  - YOLO26 thắng vật nhỏ: PL +1.4, HC +3.6, SP +1.1.
  - Lấy max theo từng class của hai model được **82.3** (RiO-x 81.8, YOLO26x 80.4).
- Kết luận: kiến trúc lai, gồm **head dense** (mạnh vật nhỏ) và **decoder quan hệ** (mạnh vật lớn/ngữ cảnh), là đòn bẩy rõ nhất để vượt cả hai.
- Các mốc khác:
  - Strip R-CNN-S MS 82.28 (license NC, chậm).
  - Nơi mất điểm là BR, SBF, RA, GTF, HA, HC, chiếm 54–59% phần còn thiếu.
  - Công thức train đáng giá: xoay ngẫu nhiên +4, multi-scale +2–4.5, ProbIoU hơn KLD 2.1 (PP-YOLOE-R).

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

## 4. Hướng kiến trúc: OpenOBB

Tên làm việc: **OpenOBB**. Lõi là detector OBB raster tổng quát (benchmark được trên DOTA). Nhánh vector là **plug-in** riêng cho CAD.

### 4.1 Khung nền (đã code, `openobb/models/`)

Dựa trên D-FINE (Apache-2.0), init từ checkpoint **COCO** của D-FINE: chỉ COCO, tránh điều khoản Objects365.
- Backbone HGNetv2 cộng hybrid encoder (AIFI + CCFM), giữ nguyên D-FINE.
- Decoder OBB do OpenOBB tự viết:
  - **rotated FDR**: 4 phân phối cho cạnh trong hệ trục box, cộng 1 phân phối cho góc dư ±45°;
  - GT được "căn" về biểu diễn tương đương gần góc tham chiếu nhất, nên không có bài toán biên góc;
  - deformable sampling xoay theo θ;
  - positional query không chứa θ (RiO-DETR GDQE: +0.7);
  - oriented CDN.
- Matching: focal 2 + Chamfer 5 + KLD 2 (O2: Chamfer là cost tốt nhất).
- Loss: MAL (DEIM) với target là IoU xoay chính xác, L1 (căn góc) 5 + KLD 2 + FGL 0.15 + DDF 1.5.

### 4.2 Lõi đóng góp: dense–sparse hybrid (H4) và các giả thuyết

**Head dense xoay** (`openobb/models/dense_head.py`) gắn lên P3–P5 của encoder:
- Tower depthwise rẻ, train one-to-many bằng rotated TAL (IoU xoay chính xác, mở rộng ứng viên cho vật tí hon kiểu STAL).
- Loss: ProbIoU cộng angle penalty cho box gần vuông (YOLO26).

Head dense được dùng theo ba cách, đo từ cùng một lần train:
- (a) chỉ làm giám sát phụ;
- (b) đầu ra thứ hai, ghép với decoder lúc inference: chỉ decoder / chỉ dense / hợp / định tuyến theo kích thước. Pipeline DOTA vốn đã có NMS khi ghép patch, nên ghép đầu ra không tốn thêm gì;
- (c) nguồn query cho decoder, kèm lọc trùng (DDQ).

| Mã | Giả thuyết | Bằng chứng / lý do | Đo |
|---|---|---|---|
| H4 | Dense one-to-many cộng decoder quan hệ thắng cả hai đơn lẻ | Max theo class của RiO-x và YOLO26x = 82.3 (xem §2.2) | mAP50 val; per-class PL/SV/SP/HC vs BR/GTF/SBF/RA |
| H1 | ProbIoU thay KLD (cost và loss) | PP-YOLOE-R: 78.14 vs 76.03 | mAP50, mAP50-95 |
| H2 | Sampling trực giao (nửa số head xoay θ, nửa xoay θ+90°, RROA) | RiO: +0.56 DIOR-R | mAP50 |
| H3 | Strip / large-kernel depthwise trong CCFM cho BR, GTF, SBF, RA, HA | LSKNet/Strip R-CNN: BR +1.6–2.9, SBF +3 | per-class, FPS |
| H5 | Bias quan hệ hình học trong self-attention của query | quan hệ tương đối (xe xếp hàng, cửa trên tường) | class ngữ cảnh |
| H6 | Token ngữ cảnh toàn ảnh xuyên tile | tile 1024 cắt mất ngữ cảnh (ảnh DOTA 4000 px) | ảnh lớn |
| H7 | Plug-in vector cho CAD | AI_Takeoff: FP và TP giống hệt nhau trên ảnh | FloorPlanCAD + data nội bộ |
| H8 | Công thức train: xoay ngẫu nhiên, mosaic có rot90 mỗi ô (oriented dense O2O), multi-scale | xoay +4.0; RiO dense O2O +0.27; MS +2–4.5 | chỉ khi gộp cuối |

Thứ tự (cập nhật theo mục tiêu sản phẩm, "thông minh như VLM, nhanh như YOLO"): E1 baseline → H4 → H1 / H2 / **H3 line mảnh** → **H6 ngữ cảnh toàn ảnh lớn** → H8 khi gộp cuối → H5 → H7 CAD.

Kiểm chứng đa dataset, sau khi thắng trên DOTA:
- DIOR-R: 20 class, ảnh 800².
- HRSC2016: tàu dài mảnh, sát với kiểu line.
- FloorPlanCAD: CAD thật, có vector.
- Sau cùng là data nội bộ của user (finetune).

### 4.3 Họ model

Theo YOLO, kiến trúc phải scale được thành n/s/m/l/x bằng depth/width multiplier. Mục tiêu sau cùng: **mỗi cỡ OpenOBB có mAP50 ≥ YOLO26 cùng cỡ + ≥1 điểm**, latency ≤ 1.3× YOLO26 cùng cỡ.

## 5. Lộ trình thí nghiệm

| Bước | Nội dung | Tiêu chí qua | Ước tính GPU |
|---|---|---|---|
| E0 | Hạ tầng: Colab CLI, data DOTA tải trực tiếp trên VM, evaluator DOTA, checkpoint + resume | eval GT=pred cho mAP=1.0; resume đúng sau khi kill (**xong**) | nhỏ |
| E1 | Baseline 24 epoch SS train→val: YOLO26s-obb (chỉ đo) + OpenOBB1-S (D-FINE OBB) | OpenOBB1-S ≥ YOLO26s − 2 | ~6 + ~10 CU |
| E2 | Ablation H1, H4 (cỡ s, schedule ngắn) | mỗi H: Δ ≥ +0.5 mAP50 hoặc bỏ | trung bình |
| E3 | Ablation H2, H3, H5, H6 | như trên, kèm FPS | trung bình |
| E4 | Gộp các H đã qua, train đủ schedule trên DOTA train+val, nộp test server | ≥ YOLO26 cùng cỡ | lớn |
| E5 | Scale n→x, đo TensorRT FP16 | bảng benchmark đầy đủ | lớn |
| E6 | Plug-in vector (H7) trên FloorPlanCAD/data nội bộ | Δ rõ ở class ngữ nghĩa | trung bình |

Ngân sách: Colab Pro+, ~500 compute units (2026-10-07). G4 ≈ 8.9 CU/h. YOLO26s 24 epoch trên G4 mất khoảng 35 phút (185 ảnh/s, 42 GB). Các ablation đều chạy schedule ngắn này; chỉ cấu hình thắng mới được train dài và multi-scale.

**Vì sao vẫn tự chạy một baseline YOLO dù đã có số công bố:** số công bố dùng test server, train+val, MS và 100+ epoch. Ablation của ta dùng val, 24 epoch, SS. Cần một mốc cùng điều kiện để biết mỗi thay đổi kiến trúc thắng hay thua, và để kiểm chứng pipeline (split, evaluator). Claim cuối cùng sẽ so với số công bố bằng cách nộp test server.

### 5.0 Mở rộng benchmark theo giai đoạn (user 2026-10-07: thử vài bộ rồi mới mở rộng)

| Giai đoạn | Dataset | Trạng thái |
|---|---|---|
| 1 | DOTA-v1.0 (chính, so với YOLO26x), FloorPlanCAD (CAD) | đang chạy |
| 2 | **DIOR-R**: 20 class, train+val 11,725 / test 11,738 ảnh 800², ship 62k + vehicle 40k object. Script `colab/data/get_dior.py`. **HRSC2016**: tàu dài mảnh, nguồn Kaggle `guofeng/hrsc2016`, cần unrar | script DIOR-R sẵn sàng |
| 3 | DOTA-v1.5/v2.0, thêm bộ CAD khác | khi kiến trúc đã ổn |

### 5.1 Benchmark CAD (E6): FloorPlanCAD (khảo sát 2026-10-07)

- **Nguồn:** bộ SVG GT `gdown 1wsOQxIXjsqYzMlUpPNRjyQiMnwgVbtJG`, 132 MB, khoảng 11.6k bản vẽ, split train/val/test. Bộ gốc kèm PNG nằm trên Google Drive, khoảng 5 GB. License CC BY-NC (chỉ benchmark).
- **Định dạng:**
  - viewBox 140×140 (khối 10 m × 10 m).
  - `<path>` gồm line/arc, cộng `<circle>`/`<ellipse>`.
  - Mỗi phần tử có `semanticId` (1–35) và `instanceId`. Class 1–30 là "thing", 31–35 là "stuff" (tường, kính, lan can...).
  - **Không có text**; layer chỉ còn màu nét.
  - Trung bình khoảng 900 primitive, khoảng 15 object mỗi bản vẽ, tối đa 181.
- **Nhãn OBB:** minAreaRect trên điểm lấy mẫu của mỗi (semanticId, instanceId), chỉ class 1–30. Raster tự vẽ bằng OpenCV, nét ≥ 1 px; không dùng renderer có license lạ.
- **Mốc tham khảo:** paper gốc dùng HBB trên V1. AP50 0.60 (Faster R-CNN), 0.62 (FCOS), 0.64 (YOLOv3). Có nhiều symbol rất mảnh (cửa trượt ~20:1, cửa sổ 5:1), hợp để thử OBB và H3.
- **SOTA symbol spotting** (metric PQ trên primitive, chỉ vector):
  - VecFormer PQ 88.4 (Apache);
  - CADSpotting 87.4;
  - SymPoint 83.3 (NC);
  - CADTransformer 68.9 (raster + vector, MIT).
- **Thí nghiệm E6:** OpenOBB raster-only vs OpenOBB + vector tokens (H7), cùng điều kiện, đo mAP50 / mAP50-95 OBB theo class.

## 6. Tài liệu tham khảo chính

- YOLO26-OBB benchmark: https://docs.ultralytics.com/tasks/obb/
- O2-DFINE / O2-RTDETR / O2-DEIM (real-time DETR-OBB): https://arxiv.org/abs/2603.15497. Code Apache-2.0: https://github.com/wokaikaixinxin/ai4rs (có 30+ detector OBB trên MMRotate)
- D²Q-DETR (DETR-OBB, 81.24 mAP DOTA multi-scale): https://arxiv.org/abs/2303.00542
- AO2-DETR: https://arxiv.org/abs/2205.12785
- Symbol spotting CAD vector: SymPoint-V2, CADSpotting, VecFormer (https://arxiv.org/abs/2505.23395)
