# Sổ cái nghiên cứu VRDet (LEDGER)

**Luật (bắt buộc với mọi agent):**
1. Mỗi job xong: thêm 1 dòng vào §2, kèm số liệu và kết luận.
2. Mỗi phát hiện mới, lỗi mới hay hướng bị bác bỏ: thêm vào §3, §4 hoặc §5 kèm bằng chứng. Không xoá dòng cũ; nếu sai thì gạch hoặc ghi "VÔ HIỆU" và lý do.
3. **So sánh chỉ với YOLO26x** (YOLO mạnh nhất; user, 10-08). VRDet có thể nhỏ hơn. Không trình bày so sánh với YOLO-s
   (các dòng YOLO26s cũ chỉ còn là lịch sử).
4. Chọn thí nghiệm kế tiếp từ §6 (hướng mở, xếp theo bằng chứng). Cập nhật §6 sau mỗi kết quả.

Nguồn chi tiết: `research/decisions.log` (nhật ký theo thời gian), `research/HYPOTHESES.md` (giả thuyết và tiêu chí pass), `python tools/summarize.py` (bảng số tự sinh từ `runs/`).

**Tiết kiệm CU (user, 2026-10-08):** sàng lọc ý tưởng ở **12 epoch**, so với mốc 12 epoch `e20-dota-base12-s`. Chỉ ý tưởng thắng mới chạy 24 epoch.
G4 vẫn rẻ nhất trên mỗi ảnh: A100 khoảng 5.3 CU/h nhưng chậm khoảng 2 lần vì máy ít CPU, nên đắt hơn khoảng 1.6 lần mỗi ảnh.

Quy ước: toàn bộ là val, 24 epoch (trừ khi ghi 12ep), ảnh 1024. DOTA cắt SS 1024/200, init COCO trừ khi ghi khác. Số dạng "mAP50 / mAP50:95". Latency đo ở batch 1, fp16, G4. Δ so với mốc ghi trong cột "So với".

## 1. Mốc so sánh (YOLO26, Ultralytics, chỉ để đo)

| Run | Benchmark | mAP50 / 50:95 | Latency | Ghi chú công thức |
|---|---|---|---|---|
| e1-yolo26s-dota-24e | DOTA | 74.77 / 49.49 | 8.9 ms | AdamW (auto), init Objects365→COCO, mosaic 1.0 (tắt 10 ep cuối), scale 0.5, translate 0.1 |
| e5-yolo26x-dota-24e | DOTA | **78.42 / 53.10** | 12.8 ms | như trên, batch 16 |
| e4-fpc-yolo26s-24e | FloorPlanCAD | 78.51 / 70.54 | 7.8 ms | như trên (chỉ còn là lịch sử) |
| e8-fpc-yolo26x-24e | FloorPlanCAD | **80.16 / 74.96** | 11.6 ms | **mốc CAD**, batch 16 |

## 2. Thí nghiệm VRDet

| Ngày | Run | Bench | Cỡ | Thay đổi | So với | Kết quả | Δ | Kết luận | CU |
|---|---|---|---|---|---|---|---|---|---|
| 10-07 | e1-vrdet-s-dota-24e | DOTA | S | baseline (D-FINE-OBB) | – | 69.89 / 46.13 | – | mốc VRDet | 13 |
| 10-07 | e1-vrdet-s-seed1 | DOTA | S | seed khác | e1 | 70.10 / 46.27 | +0.2 | nhiễu seed khoảng 0.2 | 9 |
| 10-07 | e2-h3-strip-s | DOTA | S | strip context (H3) | e1 | 69.76 / 46.63 | −0.13 | bỏ | 9 |
| 10-07 | e2-h6-ctx-s | DOTA | S | thumbnail context (H6) | e1 | 67.97 / 44.74 | −1.93 | **VÔ HIỆU**: lỗi BN (đã sửa) | 9 |
| 10-07 | e2-h4-dense-s | DOTA | S | head dense (H4) | e1 | dừng ở ep 15 | – | **VÔ HIỆU**: lỗi ProbIoU (F10) | 20 |
| 10-07 | e2-h4c-dq-s | DOTA | S | query từ head dense (H4c) | e1 | dừng | – | **VÔ HIỆU**: lỗi ProbIoU | 15 |
| 10-07 | e3-ctx-fix-s | DOTA | S | context, BN đã sửa | e1 | 69.71 / 46.43 | −0.18 | không lợi; HC −19 | 10 |
| 10-07 | e3-rfs-s | DOTA | S | RFS t = 0.1 | e1 | **72.35 / 48.18** | **+2.45** | **XÁC NHẬN**, mặc định | 10 |
| 10-07 | e4-fpc-vrdet-s-24e | FPC | S | RFS | YOLO26s | 75.96 / 65.16 | −2.55 | mốc VRDet trên CAD | 7 |
| 10-07 | e5-rfs-ctx-s | DOTA | S | RFS + context + ctx-dropout | e3 | 68.47 / 44.22 | −3.88 | **BÁC BỎ H6** | 12 |
| 10-08 | e5-rfs-dense-s | DOTA | S | RFS + dense | e3 | dec 71.84; dense 49.8 | −0.51 | **VÔ HIỆU**: lỗi ProbIoU | 14 |
| 10-08 | e7-fpc-vec-s-24e | FPC | S | + nhánh vector (H7) | e4 | **76.80 / 67.67** | +0.84 / **+2.51** | xác nhận một phần; giữ cho CAD | 7 |
| 10-08 | e9-dota-qinfer-s | DOTA | S | 600 / 900 query khi suy luận | e3 | 72.99 / 48.81 (q600) | +0.72 | **XÁC NHẬN**; SV +6.3 | 2 |
| 10-08 | e6-vrdet-x-dota-24e | DOTA | X | VRDet-X + RFS | YOLO26x | 75.60 / 52.31 | −2.81 | dung tích không tự đóng khoảng cách | 36 |
| 10-08 | e11-dota-x-qinfer | DOTA | X | 900 query khi suy luận | e6 | **76.29 / 52.84** | +0.69 | so YOLO26x: −2.13 / −0.26 | 3 |
| 10-08 | e10-fpc-vec-mosaic-s | FPC | S | mosaic thu nhỏ 0.5×, p 0.5 | e7 | 72.92 / 61.89 | −3.88 | **BÁC BỎ** cho CAD | 7 |
| 10-08 | e14-dota-post-eval | DOTA | S+X | argmax post, 900 query | flat q900 | S 72.79 / 48.66; X 75.91 / 52.64 | −0.16 / −0.38 | **BÁC BỎ**: giữ flat; HC cũng giảm (54.1 so với 56.4) | 3 |
| 10-08 | e10-dota-mosaic-s | DOTA | S | mosaic thu nhỏ 0.5×, p 0.5 | e3 | 70.93 / 47.63 | −1.42 (không tính HC: **+0.33**) | HC sụp 53.3 → 27.4; 14 class còn lại gần trung tính (BC +4.8, SP +4.3, BR −3.9) | 10 |

Đang xếp hàng (2026-10-08 09:30):
- e14: argmax post (eval S/X).
- e16: dense đã sửa lỗi ProbIoU.
- e15: init Objects365 → COCO.
- e12: nhóm query một-nhiều (H10).
- e13: công thức augmentation YOLO (H11).

| 10-08 | e16-dota-dense-fix-s | DOTA | S | RFS + dense (sau sửa ProbIoU) | e3 | dec 72.29 / 47.90; **dense 69.08**; hợp 72.93 | dense: +19.3 so với bản lỗi | dense ngang YOLO26s ở class nhỏ, kém ở class lớn → bù cho decoder; ủng hộ kiến trúc lai | 13 |
| 10-08 | e20-dota-base12-s | DOTA | S | mốc 12 epoch | – | 68.21 / 44.17 (q900 69.36) | – | mốc cho sàng lọc 12ep | 5 |
| 10-08 | e15-dota-o365-s | DOTA | S | init Objects365→COCO | e3 | 70.29 / 46.86 | −2.05 (không tính HC −0.98) | **BÁC BỎ**: giữ init COCO | 10 |
| 10-08 | c1-fpc-combo-s | FPC | S | gộp SOTA: vector + layer pooling + o2m 900 + AQD + loss góc + IoU-cost (+900 query) | e7 | **78.21 / 69.31** (q900) | +1.41 / +1.64 | gộp có lợi; so YOLO26x: −1.95 / −5.65 (khoảng cách chính là độ khít box) | 9 |
| 10-08 | c2-fpc-hybrid-s | FPC | S | c1 + kiến trúc lai dense–sparse (query từ head dense) | c1 | 78.16 / 68.80 (q900); hợp 78.14 | −0.05 / −0.51 | **không chọn cho CAD**: bằng c1, chậm hơn (13.6 ms) | 10 |

## 3. Phát hiện (có bằng chứng)

- **F1. Khoảng cách cùng điều kiện:** lõi DETR kém YOLO26 khoảng 2 mAP50 ở cả S lẫn X (24 epoch). Định vị gần ngang: ở X với 900 query, mAP50:95 chỉ kém 0.26 (e11 so với e5).
- **F2. Lỗi nằm ở xếp hạng điểm (precision), không phải recall.** VRDet-X có R50 ≥ YOLO26x ở 13/15 class. HC: R50 93.1 so với 88.9 nhưng AP 56.4 so với 75.0. SP là class duy nhất thiếu recall (80.1 so với 86.6). Nghi phạm "phát nhiều class" đã bị e14 bác bỏ (argmax kém hơn). HC thua chủ yếu vì định vị: AP50:95 là 25 so với 37.
  Hướng xử lý: loss góc (e18), AQD (e17), nhánh dense (e16).
- **F3. Class hiếm:** lấy mẫu RFS cho +2.45 (HC +15.8, RA +7.7, SBF +7.6). One-to-one matching cho quá ít mẫu dương ở lịch 24 epoch.
- **F4. Trần 300 query:** 68/5297 patch DOTA val bão hoà, chứa 28.5% số SV, 46% số SH và 17% số LV được phát hiện. Suy luận 600–900 query cho +0.7 mà không chậm hơn.
- **F5. Dung tích không tự đóng khoảng cách:** S kém −2.4, X kém −2.8 (300 query).
- **F6. Ngữ cảnh thumbnail toàn ảnh làm hại:** học scene prior (sân bay → máy bay), HC giảm 20–27 điểm.
- **F7. Nhánh vector giúp định vị trên CAD:** +2.5 mAP50:95, cửa đơn +3.6, bay-window +5.7, escalator +6.3. Latency +1.6 ms với 1000 token, train chậm hơn 14%.
- **F8. Mosaic thu nhỏ 0.5× làm hại CAD** (−3.9): symbol bị thu nhỏ.
- **F9. Yếu tố gây nhiễu trong so sánh** (đọc `args.yaml` và source YOLO26):
  - YOLO có mosaic giữ tỉ lệ cùng scale jitter 0.5–1.5 và translate 0.1, VRDet thì không.
  - YOLO khởi tạo từ Objects365 → COCO, VRDet chỉ từ COCO.
  - YOLO26 dùng AdamW (MuSGD chỉ bật khi trên 10k iteration).
- **F16. Dense và decoder bù nhau** (e16): head dense kiểu YOLO mạnh ở vật nhỏ (PL, SV, SH, TC, ST ngang YOLO26s), decoder DETR
  mạnh ở vật lớn/ngữ cảnh (BD, BR, SBF, HA, dense kém 9–13). Hợp hai đầu ra: +0.66 mAP50 → cơ sở cho kiến trúc lai c2.
  Chia theo kích thước (route 24–64 px) đều kém hơn chỉ dùng decoder (69.9–72.0) → quy tắc suy luận của kiến trúc lai là **hợp**.
- **F13. Helicopter (HC) cực nhiễu và chi phối mAP DOTA:** chỉ 72 mẫu val; AP dao động 17–56 giữa các run VRDet; một class
  = 6.7% trọng số mAP. Không tính HC: VRDet-X q900 77.71 so với YOLO26x 78.66 (**−0.95**, so với −2.13 khi tính HC).
  Context (e3-ctx) +1.2 và mosaic (e10) +0.3 trên 14 class còn lại, chỉ HC sụp.
  **Từ nay báo cả mAP50 và mAP50 không tính HC**; quyết định ablation xem cả hai. HC gần vuông nên góc mơ hồ: thử e18.
- **F14. Góc của object gần vuông** (phân tích, chưa đo):
  - Loss Gaussian (KLD, ProbIoU) gần như mù góc với box vuông, vì covariance đẳng hướng.
  - Loss sin²(2Δθ)·ω của YOLO26 (ω = 1 cho box vuông, nhỏ dần khi box dài) có chu kỳ 90°, nên chỉ đúng cho box vuông.
    Nó **bổ sung** tín hiệu góc ở chỗ Gaussian không thấy.
  - L1 góc của VRDet (target căn theo biểu diễn tương đương ±90°) cũng cho tín hiệu này. Không được giảm trọng số L1 góc cho box vuông
    (suýt làm nhầm ngày 10-08, đã hoàn tác).
  - Liên quan HC (AP50:95 25 so với 37 của YOLO26x): thử e18.
- **F10. Lỗi ProbIoU** (xem §4).
- **F11. Tốc độ:** VRDet-X 20.0 ms so với YOLO26x 12.8 ms (64% FPS). VRDet-S 9.4–10.1 ms so với 8.9. Số query không ảnh hưởng latency (300 → 900 vẫn khoảng 20 ms). A100 chậm hơn G4 khoảng 2 lần với workload này.
- **F12. YOLO26-OBB, đọc từ source/paper:**
  - Bỏ DFL (L1 trên khoảng cách).
  - Head một-một là bản sao của head một-nhiều, chạy trên feature đã detach; ProgLoss đổi trọng số 0.8/0.2 → 0.1/0.9.
  - STAL: cạnh GT nhỏ hơn 16 px được nới lên 16 px khi chọn ứng viên.
  - Góc là đầu ra thô trong [−45°, 135°). Loss góc ω·sin²(2Δθ) với ω = exp(−ln²(w/h)/9).
  - TAL: top-k 10, α = 1, β = 6.

- **F15. Báo cáo 3 sub-agent (10-08)** — kiến trúc SOTA, ngữ nghĩa CAD, dataset:
  - **FloorPlanCAD có layer:** mỗi nhóm `<g>` của SVG là một layer CAD (có id, không tên). Parser của ta đang bỏ mất.
    SymPoint-V2 gộp đặc trưng theo layer (LFE): **+6.5 PQ**; VecFormer: +2.7 PQ.
  - FloorPlanCAD bản 15.6k (11/2021) có thể có text (TextCAD dùng nó; TextCAD: text gated cross-attention 88.1 → 91.1 PQ).
  - Phần VRDet đã có sẵn: GDQE, LQE 5 phân phối, cost Chamfer (tương đương Hausdorff của RHINO), decoupled angle.
    MAL tương đương IA-BCE/GCL/position-supervised loss.
  - Matcher chưa có IoU (Rank/Stable-DINO +0.4) → e23. Ortho heads (RiO +0.56 AP50) đã code nhưng chưa chạy → e22.
  - Dataset pretrain OBB (HF đã kiểm): DOTA-v2 `nilsleh/dotaV2_patched`, FAIR1M `LittleCollections/FAIR1M1|2`,
    SODA-A `satellite-image-deep-learning/SODA-A`, DroneVehicle `McCheng/DroneVehicle`, ShipRSImageNet `insomnia7/ShipRSImageNet`,
    STAR `Zhuzi24/STAR`, CODrone `huseyincavus/codrone`; HRSC2016 trên Kaggle `guofeng/hrsc2016`.
    Pretrain thêm dữ liệu viễn thám giúp DIOR-R/HRSC nhiều (+2–4) nhưng DOTA-v1 ít (+0.2–1.5) (MTP, RTMDet-R).
  - Dataset CAD: ArchCAD-400K (HF `jackluoluo/ArchCAD`, cần user chấp nhận điều khoản), SESYD (HTTP),
    PID2Graph (Zenodo 14803338), ResPlan (CC BY), CubiCasa5K (Zenodo). Không có dataset MEP mở → synthetic từ thư viện symbol.
  - Foundation backbone viễn thám (MTP, RVSA, SkySense) không đáng ở quy mô này (DOTA-v1 gần như không tăng, quá chậm).

## 4. Lỗi đã gặp (và kết quả bị vô hiệu)

| Ngày | Lỗi | Ảnh hưởng | Sửa |
|---|---|---|---|
| 10-07 | Thumbnail đi qua backbone ở chế độ train, làm hỏng running stats của BN | e2-h6 vô hiệu | thumbnail chạy BN eval + no_grad |
| 10-07 | `torch.compile` gắn trước khi copy EMA, nên eval EMA gọi nhầm model đang train | sub-mAP trước 20:35 của seed1 / ctx-fix / rfs thấp khoảng 7 điểm | compile sau khi tạo EMA |
| 10-07 | Dense: NaN do sqrt(det) và exp không chặn | — | floor kích thước, chặn decode, bỏ qua bước không hữu hạn |
| 10-08 | **`probiou` không bất biến theo tỉ lệ**: = 1.0 với box chuẩn hoá nhỏ hơn khoảng 100 px | Loss box dense ≈ 0, NMS dense xoá object kề nhau, `distinct_topk` hỏng. **H4 / H4c vô hiệu** | Viết lại trong miền log, có test hồi quy |

## 5. Hướng đã bác bỏ

- Thumbnail global context (H6): F6.
- Strip context (H3): −0.13.
- Mosaic thu nhỏ 0.5× trên CAD: F8. Mosaic giữ tỉ lệ kiểu YOLO chưa thử, xem e13.
- Thay backbone tự thiết kế: cần pretrain ImageNet/COCO, hàng nghìn GPU-giờ, ngoài ngân sách.

## 6. Hướng mở, xếp theo bằng chứng

**ĐỔI MỤC TIÊU (user, 10-08 13:15): suy luận trên ẢNH CAD, không vector.** Ưu tiên raster-only trên FloorPlanCAD so với YOLO26x:
- c4 (bản gộp không vector), đang chạy.
- **P2 stride 4** (nét mảnh, vật nhỏ).
- Kernel dải / LSK thử trên CAD (trước đây chỉ thử DOTA).
- Render giữ nét mảnh.
- Nhánh vector, c3 chỉ là tuỳ chọn khi có PDF.


**Ưu tiên của user (10-08, 11:15): tập trung vào KIẾN TRÚC MỚI.** Tạm dừng các thí nghiệm đơn lẻ chỉ là tinh chỉnh nhỏ trên nền D-FINE
(e12b, e17, e18, e13, e23, e24; spec giữ lại).
- **c1 = gộp các phần tốt nhất SOTA:** RFS, nhóm query một-nhiều H-DETR (900), AQD (RHINO), loss góc (YOLO26),
  IoU-cost (Stable-DINO), 900 query khi suy luận; bản FPC có thêm vector + layer pooling (SymPoint-V2).
- **c3 = box neo theo nét vector (đột phá chính, 10-08):** mỗi query chọn các nét CAD thuộc object (membership query × token);
  box xoay tính chính xác từ toạ độ nét được chọn; phân loại dựa trên nét/layer/text. Nhắm F7 (định vị CAD) và ngữ nghĩa.
  Chỉ benchmark FloorPlanCAD trước (user).
- **c2 = kiến trúc lai dense–sparse (bước đột phá):** head dense kiểu YOLO26 (TAL một-nhiều) sinh đề xuất recall cao →
  decoder DETR suy luận quan hệ trên các đề xuất đó (DDQ/Co-DETR) + nhánh vector/text CAD. Phụ thuộc e16.


| # | Hướng | Bằng chứng | Chi phí | Trạng thái |
|---|---|---|---|---|
| 1 | Argmax post (một class mỗi query) | F2 | 3 CU | **xong: bác bỏ** |
| 2 | Head dense kiểu YOLO26 làm giám sát một-nhiều cho encoder | Co-DETR +1.6–2.4, RT-DETRv3 +1.6 ở lịch ngắn; lần thử trước bị lỗi F10 | 13 CU | e16 xếp hàng |
| 3 | Init Objects365 → COCO (ngang YOLO) | F9; D-FINE X +3.5 COCO AP | 10 CU | **xong: bác bỏ** (−2.05; không tính HC −0.98) |
| 4 | Nhóm query một-nhiều (H-DETR) | H-DETR, MS-DETR; LW-DETR Group-DETR +2.9 | 10 CU | e12 (24ep) dừng vì tốn (40 ảnh/s); e12b 12ep xếp hàng |
| 5 | Công thức augmentation YOLO (mosaic giữ tỉ lệ, scale, translate) | F9 | 11 CU | e13 xếp hàng |
| 6 | Adaptive query denoising (RHINO) | DOTA val +1.9 AP50 (DINO); nhắm vào F2 | trung bình | đã code (`--aqd`); e17 xếp hàng |
| 7 | ProgLoss: giảm dần trọng số nhánh một-nhiều | YOLO26 +0.3 AP | thấp | chưa làm |
| 8 | Loss góc cho object gần vuông trên decoder | YOLO26 +0.6 AP50 / +1.2 50:95; HC định vị kém | thấp | đã code (`--angle-weight`); e18 xếp hàng |
| 9 | Train với 900 query | F4; RHINO train 900 | thấp | chưa làm |
| 10 | Tắt mosaic ở 50% số epoch (DEIM) | DEIM | thấp | sau #5 |
| 11 | Copy-paste class hiếm (HC, SBF, BC, SP) | F2, F3 | trung bình | chưa làm |
| 12 | Gộp các thành phần có lợi lên X: DOTA và FPC (so YOLO26x) | — | khoảng 45 + 45 CU | sau #1–#5 |
| 14 | Adapter chọn vùng nhìn kiểu LSKNet (trước encoder) | LSKNet / PKINet / Strip R-CNN mạnh trên DOTA (backbone pretrain ImageNet, two-stage); H3 strip sau encoder −0.13 | 11 CU | hoãn (cắt khỏi hàng đợi 10-08 vì ngân sách; e19 đã soạn sẵn) |
| 15 | IoU trong chi phí matching (p^(1−g)·IoU^g) | Rank-DETR, Stable-DINO +0.4 AP; nhắm F2 | 5 CU | đã code (`--cost-iou`); e23 xếp hàng |
| 16 | Ortho attention heads (RiO-DETR) | +0.56 AP50 DIOR-R; hướng object vuông | 5 CU | hoãn (cắt khỏi hàng đợi 10-08 vì ngân sách; e22 đã soạn sẵn) |
| 17 | Chẩn đoán từng lớp decoder → SQR nếu lớp giữa tốt hơn | SQR +1.4–2.8 AP (Deformable-DETR) | 3 CU | hoãn (cắt khỏi hàng đợi 10-08 vì ngân sách; e21 đã soạn sẵn) |
| 18 | **Layer token + gộp theo layer (LFE) cho nhánh vector** | SymPoint-V2 +6.5 PQ, VecFormer +2.7 | thấp | đã code (`--vec-lfe --layer-drop`); e24 xếp hàng |
| 19 | Token hình học kiểu VecFormer (tâm, độ dài, hướng) + bias quan hệ hình học (GAT-CADNet) | VecFormer, GAT-CADNet +5 PQ | thấp | chưa làm |
| 20 | Fusion 2 chiều + decoder cross-attend token vector (Grounding DINO) | Grounding DINO: decoder cross-attn +0.6 COCO / +1.8 LVIS | trung bình | chưa làm |
| 21 | Co-DETR positive queries từ head dense (nếu e16 dương) | Co-DETR Deformable 12ep +3.4 | trung bình | chờ e16 |
| 22 | DDQ distinct queries (H4c chạy lại sau sửa ProbIoU) | DDQ +1.5 (aux loss), CrowdHuman | trung bình | chờ e16 |
| 23 | Pretrain đa dataset OBB (DOTA-v2, FAIR1M, SODA-A, DIOR-R, ...) cho nền sản phẩm | MTP/RTMDet-R; DIOR-R/HRSC +2–4 | khoảng 75 CU | sau khi chốt kiến trúc |
| 24 | Pretrain nhánh vector/text (ArchCAD-400K, SESYD, PID2Graph, synthetic MEP) | DPSS: gấp đôi dữ liệu ArchCAD +6.4 PQ | cao | sau #18–#20 |
| 13 | Token text và layer cho PDF thật (sản phẩm) | AI_Takeoff: FP/TP chỉ phân biệt được bằng text | code local | sau khi kiến trúc thắng |
