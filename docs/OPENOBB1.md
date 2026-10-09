# Thẻ kiến trúc OpenOBB: bản chốt raster (2026-10-08)

Chỉ nói về kiến trúc model: chỉ số, các lớp, phần dùng lại và nguồn, rủi ro bản quyền.

- Mục tiêu: OBB trên **ảnh** bản vẽ CAD (object nhỏ, ống mảnh, khối), không dùng vector khi suy luận.
- Bản chốt: **OpenOBB + LSK** (run `c6-fpc-raster-lsk-s`). Đây là mặc định của `openobb train`.
- Lịch sử thí nghiệm: [research/LEDGER.md](../research/LEDGER.md).

## 1. Sơ đồ

```
Ảnh trang (render) ─► (scale) ─► cắt tile 1024×1024, chồng 200 px
                           │
                           ▼
              Backbone HGNetv2 (C3 s8, C4 s16, C5 s32)
                           │
                           ▼
              Adapter LSK trên C3/C4/C5: kernel gần (dw 5×5) + kernel xa (dw 7×7 giãn 3, vùng nhìn khoảng 23 px)
              trộn theo cổng không gian, nhân vào feature; khởi tạo = identity
                           │
                           ▼
              Hybrid encoder: AIFI (attention trên C5) + CCFM (FPN + PAN)
                           │
                           ▼
              Chọn query: top-K đề xuất từ encoder (300 khi train, 600–900 nếu tile dày object; 900 khi suy luận)
                           │
                           ▼
              Decoder L lớp: self-attention giữa query → deformable attention xoay theo góc → FFN
                             → phân phối 4 cạnh + góc, tinh chỉnh dần qua từng lớp
                           │
                           ▼
              Box xoay (cx, cy, w, h, θ) + điểm từng class; không NMS trong tile;
              ghép tile về trang bằng NMS đa giác IoU 0.1 theo class
```

## 2. Chỉ số (val, 24 epoch, tile 1024, cùng điều kiện; mốc duy nhất: YOLO26x)

| Benchmark | Model | Tham số | mAP50 | mAP50:95 | Latency (bs1, fp16, RTX PRO 6000) |
|---|---|---|---|---|---|
| FloorPlanCAD (ảnh CAD, 30 class) | YOLO26x (mốc) | 57.6M, 202 GFLOPs | **80.16** | **74.96** | 11.6 ms |
| FloorPlanCAD | **OpenOBB1-S + LSK (bản chốt)** | **12.5M** | **78.16** | **68.81** | 11.4 ms |
| FloorPlanCAD | OpenOBB1-S, không LSK (c4) | 10.3M | 76.83 | 66.57 | 10.0 ms |
| DOTA-v1.0 (tham khảo lõi OBB) | YOLO26x | 57.6M | 78.42 | 53.10 | 12.8 ms |
| DOTA-v1.0 | OpenOBB1-X (không LSK) | 63.4M | 76.29 | 52.84 | 20.0 ms |

- Số OpenOBB đo với 900 query khi suy luận. Latency YOLO đo end-to-end, latency OpenOBB đo forward model; hai số gần tương đương.
- **Bản S nhỏ hơn YOLO26x 4.6 lần, cùng tốc độ.**
  - Recall bằng YOLO26x: 92.0 so với 91.9.
  - Thắng ở object lớn cần hình dạng toàn cục: wardrobe +7.2, sofa +6.6 AP50.
  - Thua chủ yếu ở độ khít box của object mảnh, dài: window −12.9, sliding-door −16.0 AP50:95 (phân tích F18 trong sổ cái).
- **LSK so với không LSK:** +1.33 / +2.24, mAP50:95 tăng ở 26/27 class. Tăng mạnh nhất ở class mảnh/nhỏ: bay-window +7.6, airconditioner +5.1, blind-window +4.8, sliding-door +3.3.
- **Chưa đo:** cỡ M/X trên FloorPlanCAD. Trên DOTA, bản X đã gần ngang độ khít box của YOLO26x.

## 3. Các lớp

### 3.1 Theo cỡ (tham số đếm trực tiếp từ code, 30 class, có LSK)

| Khối | S | M | L | X |
|---|---|---|---|---|
| Backbone HGNetv2 | B0, 1.85M | B2, 6.03M | B4, 13.51M | B5, 33.23M |
| Adapter LSK (C3/C4/C5) | 2.20M | 4.86M | 8.54M | 8.54M |
| Hybrid encoder (rộng) | 4.02M (256) | 7.80M (256) | 9.34M (256) | 20.70M (384) |
| Decoder (số lớp) | 4.41M (3) | 5.72M (4) | 8.34M (6) | 8.64M (6) |
| **Tổng** | **12.5M** | **24.4M** | **39.7M** | **71.1M** |

### 3.2 Chi tiết từng lớp (bản S; X ghi trong ngoặc)

| # | Lớp | Cấu hình |
|---|---|---|
| 1 | Stem | conv 3×3 s2: 3→16→16 (3→32→64) |
| 2 | Stage 1 (s4) | 1 HG block, 3 conv 3×3, ra 64 kênh (1 block, 6 conv, 128) |
| 3 | Stage 2 (s8) → **C3** | 1 HG block, ra 256 (2 block, 512) |
| 4 | Stage 3 (s16) → **C4** | 2 HG block nhẹ (depthwise 5×5), ra 512 (5 block, 1024) |
| 5 | Stage 4 (s32) → **C5** | 1 HG block nhẹ, ra 1024 (2 block, 2048) |
| 6 | **LSK** trên C3/C4/C5 | dw 5×5 → dw 7×7 giãn 3; mỗi nhánh conv 1×1 về c/2; cổng = conv 7×7 trên [mean, max] theo kênh → sigmoid → trộn 2 nhánh; conv 1×1 về c (khởi tạo 0); ra x + x·(…) |
| 7 | Input proj | conv 1×1 + BN: C3/C4/C5 → 256 (384) |
| 8 | AIFI | 1 lớp transformer encoder trên C5 (8 head, FFN 1024 (2048)) |
| 9 | CCFM | FPN từ trên xuống + PAN từ dưới lên, khối RepNCSPELAN4 (độ sâu 0.34 (1.0)), SCDown |
| 10 | Chọn query | linear + LN trên memory; head class + head box (4 delta + cos/sin 2θ); lấy top-K; vị trí query không chứa góc (GDQE) |
| 11 | Decoder × 3 (× 6) | self-attention (8 head, d 256) → deformable cross-attention xoay theo góc box (8 head, 3 mức, điểm 3/6/3) → FFN 1024 |
| 12 | Head mỗi lớp | class (linear); phân phối 5 × 33 bin (4 cạnh trong khung xoay + góc dư ±π/4); LQE (chất lượng định vị từ phân phối) |
| 13 | Đầu ra | điểm sigmoid theo class + box (cx, cy, w, h, θ); top-K trên (query × class) |

### 3.3 Chỉ dùng khi train (không tốn thời gian suy luận)

- **Matching:** Hungarian (focal × IoU^0.5 + Chamfer góc + KLD).
- **Nhóm query một-nhiều:** 900 query, mỗi object 6 query, bỏ khi suy luận.
- **Denoising có hướng:** AQD (số nhóm thích ứng theo số object).
- **Loss:**
  - MAL với nhãn mềm là IoU xoay chính xác.
  - L1 căn theo góc, KLD.
  - Loss góc nhạy với box gần vuông.
  - FGL/DDF trên phân phối cạnh và góc.
- **Lấy mẫu:** repeat-factor sampling (t = 0.1) cho class hiếm.

## 4. Phần dùng lại: bao nhiêu và lấy từ đâu

| Thước đo | Dùng lại gần nguyên | Viết lại trên khung có sẵn | Tự viết |
|---|---|---|---|
| Tham số, bản S | 47% (backbone 15% + encoder 32%) | 35% (decoder OBB) | 18% (LSK) |
| Tham số, bản X | 76% (backbone 47% + encoder 29%) | 12% (decoder OBB) | 12% (LSK) |

| Thành phần | Nguồn | License | Cách dùng |
|---|---|---|---|
| HGNetv2 backbone | D-FINE (port từ PaddleDetection, Baidu) | Apache-2.0 | chép, đã sửa; giữ notice |
| Hybrid encoder (AIFI + CCFM) | DEIM / D-FINE / RT-DETR | Apache-2.0 | chép, thêm hook |
| Deformable attention core, tiện ích, FrozenBN | DEIM / D-FINE / RT-DETR / DETR | Apache-2.0 | chép |
| Cấu trúc decoder, hàm FDR, LQE, denoising, matcher, khung loss MAL/FGL/DDF | D-FINE, DEIM | Apache-2.0 | viết lại cho box xoay |
| Phần box xoay (phân phối cạnh + góc, sampling xoay, cost Chamfer/KLD/IoU, IoU xoay, denoising có hướng, AQD, nhóm một-nhiều, loss góc), adapter LSK | tự viết; ý tưởng từ paper: RiO-DETR, O2-DETR, RHINO, H-DETR, YOLO26, LSKNet | — | chỉ lấy ý tưởng, không chép code |

- License gốc của phần chép nằm trong `LICENSES/`.
- Weights khởi tạo: D-FINE COCO (Apache-2.0).

## 5. Rủi ro bản quyền của kiến trúc (đánh giá kỹ thuật, không phải tư vấn pháp lý)

| Phần | Rủi ro | Lý do |
|---|---|---|
| Backbone, encoder, lõi attention chép từ D-FINE/DEIM | **Thấp** | Apache-2.0 cho phép bán, sửa, đóng mã; chỉ cần giữ LICENSE + NOTICE và ghi file đã sửa (đã làm). |
| Decoder OBB viết lại trên khung D-FINE | **Thấp** | Khung gốc Apache-2.0; phần box xoay là code riêng. |
| Phần tự viết theo ý tưởng paper (kể cả paper có code NC/AGPL như RHINO, YOLO26, LSKNet) | **Thấp** | Bản quyền bảo vệ code, không bảo vệ ý tưởng; không chép code. Rủi ro còn lại là bằng sáng chế (hiếm với các kỹ thuật học thuật này). |
| Weights benchmark train trên FloorPlanCAD / DOTA (CC BY-NC / học thuật) | **Trung bình nếu bán chính weights này** | Chỉ dùng để đo. Bản thương mại: fine-tune hoặc train lại trên dataset tự gắn nhãn, khởi tạo từ D-FINE COCO (Apache). |
