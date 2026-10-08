# Thẻ kiến trúc VRDet (bản 2026-10-08)

VRDet là detector box xoay (OBB) kiểu DETR thời gian thực, có thêm nhánh đọc nét vector CAD. Thẻ này trả lời bốn câu hỏi:
chỉ số đạt được, các lớp của kiến trúc, phần dùng lại lấy từ đâu, và rủi ro bản quyền. Mục 6 là cấu hình đề xuất cho
dataset riêng; sẽ cập nhật khi các thí nghiệm đang chạy (`research/LEDGER.md` §2) có kết quả.

## 1. Sơ đồ

```
Ảnh trang PDF (render) ─► cắt tile 1024×1024, chồng 200 px
                           │
                           ▼
              Backbone HGNetv2 (C3 s8, C4 s16, C5 s32)
                           │                        ┌──────────── Nhánh vector CAD (tuỳ chọn, --vectors) ─────────────┐
                           │◄── cộng (khởi tạo 0) ──┤ nét PDF/SVG → token → gộp theo layer → transformer 2 lớp      │
                           ▼                        │ → gộp theo layer → rải lên lưới s8/16/32 → conv 1×1              │
              Hybrid encoder: AIFI (attention trên C5)   └───────────────────────────────────────────────────────────────┘
                              + CCFM (FPN + PAN)
                           │
                           ▼
              Chọn query: top-K đề xuất từ encoder (300 khi train, 900 khi suy luận)
                           │
                           ▼
              Decoder L lớp: self-attention giữa query → deformable attention xoay theo góc → FFN
                             → phân phối 4 cạnh + góc, tinh chỉnh dần qua từng lớp
                           │
                           ▼
              Box xoay (cx, cy, w, h, θ) + điểm từng class; không NMS trong tile;
              ghép tile về trang bằng NMS đa giác IoU 0.1 theo class
```

## 2. Chỉ số (val, 24 epoch, ảnh 1024, cùng điều kiện với YOLO)

| Benchmark | Model | mAP50 | mAP50:95 | Latency (bs1, fp16, RTX PRO 6000) |
|---|---|---|---|---|
| DOTA-v1.0 | YOLO26x | **78.42** | **53.10** | 12.8 ms |
| DOTA-v1.0 | **VRDet-X** (900 query) | 76.29 | 52.84 | 20.0 ms (64% FPS) |
| DOTA-v1.0, không tính helicopter | YOLO26x / VRDet-X | 78.66 / 77.71 | | |
| DOTA-v1.0 | YOLO26s | 74.77 | 49.49 | 8.9 ms |
| DOTA-v1.0 | **VRDet-S** (900 query) | 72.95 | 48.90 | 9.4 ms |
| FloorPlanCAD (CAD, 30 class) | YOLO26s | **78.51** | **70.54** | 7.8 ms |
| FloorPlanCAD | **VRDet-S + vector** | 76.80 | 67.67 | 11.5 ms |

- VRDet có recall bằng hoặc cao hơn YOLO26x ở 13/15 class DOTA; phần còn kém là xếp hạng điểm và object gần vuông
  (helicopter). Độ khít box (mAP50:95) gần ngang.
- VRDet thắng ở class lớn cần ngữ cảnh: harbor +4.7, baseball-diamond +1.6 so với YOLO26x.
- Nhánh vector: +0.84 mAP50 / +2.51 mAP50:95 trên FloorPlanCAD (cửa đơn +3.6, bay-window +5.7, thang cuốn +6.3).
- Đang đo (kết quả vào sổ cái): layer CAD (e24), head dense đã sửa lỗi (e16), init Objects365 (e15), nhóm query
  một-nhiều (e12b), AQD (e17), loss góc (e18), augmentation YOLO (e13), IoU trong matching (e23).

## 3. Các lớp

### 3.1 Theo cỡ (tham số đếm trực tiếp từ code)

| Khối | S | M | L | X |
|---|---|---|---|---|
| Backbone HGNetv2 | B0, 1.85M | B2, 6.03M | B4, 13.51M | B5, 33.23M |
| Hybrid encoder (rộng) | 4.02M (256) | 7.80M (256) | 9.34M (256) | 20.70M (384) |
| Decoder (số lớp) | 4.39M (3) | 5.70M (4) | 8.31M (6) | 8.61M (6) |
| Nhánh vector | 0.65M | 0.77M | 0.89M | 0.89M |
| **Tổng** | **10.9M** | **20.3M** | **32.1M** | **63.4M** |

### 3.2 Chi tiết từng lớp (bản S; X ghi trong ngoặc)

| # | Lớp | Cấu hình |
|---|---|---|
| 1 | Stem | conv 3×3 s2: 3→16→16 (3→32→64) |
| 2 | Stage 1 (s4) | 1 HG block, 3 conv 3×3, ra 64 kênh (1 block, 6 conv, 128) |
| 3 | Stage 2 (s8) → **C3** | 1 HG block, ra 256 (2 block, 512) |
| 4 | Stage 3 (s16) → **C4** | 2 HG block nhẹ (depthwise 5×5), ra 512 (5 block, 1024) |
| 5 | Stage 4 (s32) → **C5** | 1 HG block nhẹ, ra 1024 (2 block, 2048) |
| 6 | Nhánh vector (tuỳ chọn) | token = Fourier 8 tần số của 8 điểm + hình dạng tương đối + loại nét (8 loại) + màu + log độ dày → MLP (d 128); gộp theo layer (mean+max → MLP); 2 lớp transformer pre-norm (4 head, FFN 256) + 1 register token; gộp theo layer; rải theo đường nét (×2 điểm) lên s8/16/32 kèm log mật độ; conv 1×1 khởi tạo 0 cộng vào C3/C4/C5 |
| 7 | Input proj | conv 1×1 + BN: C3/C4/C5 → 256 (384) |
| 8 | AIFI | 1 lớp transformer encoder trên C5 (8 head, FFN 1024 (2048)) |
| 9 | CCFM | FPN từ trên xuống + PAN từ dưới lên, khối RepNCSPELAN4 (độ sâu 0.34 (1.0)), SCDown |
| 10 | Chọn query | linear + LN trên memory; head class + head box (4 delta + cos/sin 2θ); lấy top-K; vị trí query không chứa góc (GDQE) |
| 11 | Decoder × 3 (× 6) | self-attention (8 head, d 256) → deformable cross-attention xoay theo góc box (8 head, 3 mức, điểm 3/6/3) → FFN 1024 |
| 12 | Head mỗi lớp | class (linear); phân phối 5 × 33 bin (4 cạnh trong khung xoay + góc dư ±π/4); LQE (chất lượng định vị từ phân phối) |
| 13 | Đầu ra | điểm sigmoid theo class + box (cx, cy, w, h, θ); top-K trên (query × class) |

### 3.3 Chỉ dùng khi train (không tốn thời gian suy luận)

- Denoising có hướng (100 query nhiễu); matching Hungarian (focal + Chamfer góc + KLD).
- Loss: MAL với nhãn mềm = IoU xoay chính xác, L1 (căn theo góc), KLD, FGL/DDF trên phân phối cạnh và góc.
- Lấy mẫu RFS cho class hiếm; EMA; LR phẳng rồi cosine.
- Tuỳ chọn đang thử: nhóm query một-nhiều, AQD, loss góc vuông, head dense phụ.

## 4. Phần dùng lại: bao nhiêu và lấy từ đâu

| Thước đo | Dùng lại gần nguyên | Viết lại trên khung có sẵn | Tự viết |
|---|---|---|---|
| Tham số (X) | 86% (backbone 53% + encoder 33%) | 14% (decoder OBB) | 1.4% (nhánh vector) |
| Dòng code (6.3k trong `vrdet/`, `colab/data/`, `tools/`) | 19% (1.2k) | 17% (1.1k) | 64% (4.0k) |

| Thành phần | Nguồn | License | Cách dùng |
|---|---|---|---|
| HGNetv2 backbone | D-FINE (port từ PaddleDetection, Baidu) | Apache-2.0 | chép, đã sửa; giữ notice |
| Hybrid encoder (AIFI + CCFM) | DEIM / D-FINE / RT-DETR | Apache-2.0 | chép, thêm hook |
| Deformable attention core, tiện ích, FrozenBN | DEIM / D-FINE / RT-DETR / DETR | Apache-2.0 | chép |
| Cấu trúc decoder, hàm FDR, LQE, denoising, matcher, khung loss MAL/FGL/DDF | D-FINE, DEIM | Apache-2.0 | viết lại cho box xoay |
| Phần box xoay (FDR + góc, sampling xoay, cost Chamfer/KLD, IoU xoay, denoising có hướng), nhánh vector, gộp layer, head dense, AQD, nhóm một-nhiều, loss góc, IoU-cost, evaluator DOTA, data, trainer, công cụ | tự viết; ý tưởng từ paper: RiO-DETR, O2-DETR, RHINO, YOLO26, SymPoint-V2, VecFormer, H-DETR, Stable-DINO, Rank-DETR, LSKNet | — | chỉ lấy ý tưởng, không chép code |
| Weights khởi tạo | D-FINE COCO (`dfine_{s,m,l,x}_coco.pth`) | repo Apache-2.0 | khởi tạo |

Danh sách đầy đủ: `THIRD_PARTY_NOTICES.md`, license gốc: `LICENSES/`.

## 5. Rủi ro bản quyền (đánh giá kỹ thuật, không phải tư vấn pháp lý)

| Hạng mục | Rủi ro | Lý do / việc cần làm |
|---|---|---|
| Code chép từ D-FINE/DEIM | **Thấp** | Apache-2.0 cho phép bán, sửa, đóng mã. Phải giữ file LICENSE + NOTICE và ghi rõ file đã sửa (đã làm). |
| Ý tưởng lấy từ paper/repo NC hoặc AGPL (YOLO26, RHINO, LSKNet...) | **Thấp** | Bản quyền bảo vệ code, không bảo vệ ý tưởng; VRDet không chép code. Rủi ro còn lại là bằng sáng chế (hiếm với các kỹ thuật học thuật này); nếu thương mại lớn nên cho luật sư rà. |
| Ultralytics (AGPL) | **Không** (nếu giữ quy tắc) | Chỉ dùng ở `colab/baselines/` để đo; `tests/test_license_guard.py` chặn mọi import vào `vrdet/`. Không được đóng gói vào sản phẩm. |
| PyMuPDF (AGPL) | **Trung bình nếu ship** | Chỉ ở `tools/pdf_vectors.py` (tuỳ chọn). Sản phẩm nên dùng pypdfium2 (Apache/BSD) hoặc pdfminer.six (MIT), hoặc mua license Artifex (AI_Takeoff đã dùng PyMuPDF). |
| Weights init D-FINE COCO | **Thấp** | Repo Apache-2.0; COCO là chuẩn ngành cho init. |
| Weights init Objects365→COCO | **Trung bình** | D-FINE lưu ý có thể chịu điều khoản Objects365: chỉ dùng để benchmark, không dùng cho model thương mại. |
| Weights đã train trên DOTA / FloorPlanCAD | **Cao nếu ship** | Dataset chỉ cho học thuật / CC BY-NC: chỉ để nghiên cứu. Model thương mại: train từ init COCO trên data riêng của bạn. |
| Thư viện nền (PyTorch, OpenCV, numpy, scipy, shapely) | **Thấp** | BSD / Apache. |

## 6. Cấu hình đề xuất cho dataset riêng (YOLO-OBB)

Tạm thời, sẽ cập nhật sau khi các thí nghiệm đang chạy xong:

```bash
python -m vrdet.train --data <dataset_vrdet> --out runs/<tên> --size s --epochs 24 --batch 16 --lr 1e-4 \
  --rfs 0.1 --channels-last --compile --eval-queries 900
```

- `--size x --batch 8 --lr 6e-5 --backbone-mult 0.1 --wd 1.25e-4` khi cần chính xác hơn (khoảng 2× thời gian suy luận).
- `--vectors` (và sau e24: `--vec-lfe --layer-drop 0.1`) khi có vector PDF (`tools/pdf_vectors.py`).
- Suy luận: `python -m vrdet.predict --ckpt runs/<tên>/last.pt --src <ảnh trang> --out <thư mục> --queries 900`.
- Hướng dẫn đầy đủ: `docs/FINETUNE.md`.
