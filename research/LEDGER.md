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
| 10-08 | c4-fpc-raster-s | FPC | S | raster-only: RFS + o2m 900 + AQD + loss góc + IoU-cost (+900 query) | e4 | 76.17 / 66.19; **q900 76.83 / 66.57** | +0.87 / +1.41 (cùng 300 q: +0.21 / +1.03) | mốc raster mới; gói gộp lợi ít khi không có vector (phần lớn lợi của c1 là nhờ vector). So YOLO26x: **−3.33 / −8.39**. 10.0 ms (YOLO26x 11.6) | 9 |
| 10-08 | c6-fpc-raster-lsk-s | FPC | S | c4 + adapter LSK (kernel chọn lọc 5×5 + 7×7 giãn 3) trên C3/C4/C5 | c4 | 77.71 / 68.39; **q900 78.16 / 68.81** | **+1.33 / +2.24** (q900) | **XÁC NHẬN, raster tốt nhất, chốt làm mặc định.** mAP50:95 tăng ở 26/27 class (bay-window +7.6, airconditioner +5.1, blind-window +4.8, sliding-door +3.3, window +2.7); recall 92.0 bằng YOLO26x. So YOLO26x: **−2.00 / −6.15**. 12.5M tham số, 11.4 ms (c4 10.0), train 43.6 ảnh/s (c4 52.4) | 10 |
| 10-08 | c5-fpc-raster-p2-s | FPC | S | c4 + level stride-4 (P2, H18) cho decoder | c4 | 67.04 / 56.49; q900 74.04 / 63.32 | **−9.1 / −2.8** (q900) | **BÁC BỎ** (cách gắn này): mọi class tụt (bay-window 14, opening 21, table 45); train chậm hơn 16% (44 so với 52 ảnh/s), 28 GB so với 21 GB | 10 |

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
- **F17. Raster-only trên CAD (c4): khoảng cách tới YOLO26x là độ khít box, không phải phát hiện.**
  - mAP50 −3.3 nhưng mAP50:95 −8.4.
  - Class tụt mạnh nhất ở AP50:95: sliding-door 53.8 so với 73.1, window 67.9 so với 83.4, airconditioner 52.2 so với 73.8.
  - Bỏ vector thì cửa đơn −6.8 và bay-window −11.4 (so với c1): đúng các object dựa vào nét mảnh.
  - Criterion chiếm 46% thời gian mỗi iteration (0.141 / 0.307 s).
  - Class không có GT trong val (rolling/revolving/folding-door) vẫn nhận hàng nghìn box: FP khi dùng thật, nên đặt ngưỡng conf theo class.
  - → Đòn bẩy raster cần đo: level stride-4 (c5, P2), receptive field cho nét mảnh (c6, LSK).
- **F18. Phân rã khoảng cách c4 (q900) so với YOLO26x trên FloorPlanCAD** (27 class có ≥ 20 GT, trung bình không trọng số):
  - **Recall gần ngang:** R50 90.9 so với 91.9. VRDet tìm thấy object gần bằng YOLO; có class còn cao hơn (opening-symbol +7.7, sofa +6.9, chair +4.2).
  - **Độ khít box là phần thua chính:**
    - mAP50 −3.4, mAP50:95 −8.7. Tỉ lệ AP50:95/AP50 là 0.86 so với 0.93.
    - Tệ nhất ở object mảnh, dài: sliding-door 0.63 so với 0.83, window 0.80 so với 0.96, blind-window 0.79 so với 0.96, airconditioner 0.81 so với 0.93.
  - **Xếp hạng điểm kém hơn chút:** AP50/R50 0.87 so với 0.90.
    - Symbol nhỏ dễ lẫn có recall ngang nhưng AP50 thấp: airconditioner −15, bay-window −13, bath −10, sink −9.
    - VRDet xuất nhiều hơn YOLO 3–10 box mỗi class (top-900 trên query × class, YOLO cắt 300 mỗi tile).
  - **VRDet thắng ở object lớn, cần hình dạng toàn cục:** wardrobe +7.7, sofa +6.3 AP50, giống DOTA (harbor, BD).
  - **Yếu tố gây nhiễu:**
    - Dung tích: VRDet-S 10.3M so với YOLO26x 57.6M tham số (202 GFLOPs).
    - Trên DOTA, VRDet-X gần ngang mAP50:95 của YOLO26x (−0.26), nên phần thua độ khít trên CAD có thể chủ yếu do cỡ S. Chưa đo X trên FloorPlanCAD.
    - Chưa hội tụ: c4 sub-mAP50 còn tăng mạnh từ ep 11 (75.2) tới ep 17 (79.9).
    - YOLO26x nạp 1164/1176 tensor pretrained, aug mạnh (mosaic 1.0, scale ±0.5, translate 0.1, randaugment, erasing).
  - **Sau c6 (LSK):** recall 92.0 (YOLO 91.9), AP50 81.1, AP50:95 71.4 (YOLO 77.7), tỉ lệ khít 0.874 (YOLO 0.930).
    - Vùng nhìn rộng, thích ứng giúp đúng các class mảnh/nhỏ.
    - Phần còn lại vẫn chủ yếu là độ khít box: window −12.9, sliding-door −16.0, blind-window −13.2 AP50:95.
  - Hướng kế tiếp: loss IoU xoay trực tiếp + lấy mẫu dọc trục dài, train dài hơn, cỡ M/X trên FPC, mosaic giữ độ phân giải (e13).

- **F19. Bộ khung fine-tune trên data riêng (Wall_Color: 250 trang khoảng 2381 px, khoảng 54 object/trang).** Đối chiếu với source Ultralytics:
  - **Cắt tile ở độ phân giải gốc:** khoảng 7 tile/trang, nên mỗi epoch chậm hơn YOLO khoảng 3 lần. Tính trên mỗi ảnh, VRDet-S vẫn nhanh hơn YOLO26x. Mặc định đổi sang resize cả ảnh.
  - **Quá ít bước:** 250 ảnh, batch 32, 30 epoch chỉ có 210 bước, warmup chiếm một nửa; mAP = 0 tới epoch 10.
    - Kiểu DETR cần khoảng 2000 bước trở lên.
    - Sửa: batch tự giữ ≥ 25 bước/epoch, warmup = 3 epoch, EMA bán rã khoảng 5% run (0.9998 cũ bán rã khoảng 3500 bước).
  - **Pipeline làm hại nét mảnh (chưa đo Δ):**
    - Box có cạnh ngắn < 2 px bị xoá khỏi target. Sửa: kẹp lên 2 px.
    - Sau crop, mảnh object cần iof ≥ 0.7 mới giữ nhãn. Sửa: 0.25 khi augment.
    - Tile lưu JPEG q95. Sửa: PNG.
    - NMS 0.1 cả khi 1 tile/ảnh. Sửa: 0.7.
  - **Chưa làm:** tích luỹ gradient tới batch danh nghĩa; P/R tại một ngưỡng conf chung; fallback fp16 cho T4; cache ảnh trong RAM.

- **F20. Wall_Color (user tự chạy, G4, cả trang resize 1280, batch 8, 400 query): chững ở khoảng 0.43 / 0.20.**
  - **Đợt 1:** 40 epoch.
  - **Đợt 2:** train tiếp từ best.pt.
    - 11 epoch: mAP50 0.399 → 0.431, mAP50-95 0.183 → 0.197, P ≈ R ≈ 0.5.
    - Loss đi ngang. 10.5 s/epoch, 62 GB.
  - **Tham chiếu YOLO26x** (từ `wall_300kaggle.pt` = đã học tường, cách chấm của Ultralytics): 0.66–0.72 / 0.42–0.46. Chưa so công bằng.
  - **Giả thuyết, chưa đo:**
    - (a) Thiếu query: R 0.5 trong khi trang tới khoảng 400 object → thử 900.
    - (b) Box kém khít với vật mảnh (F18): tỉ lệ mAP50-95/mAP50 = 0.46.
    - (c) Train hai đợt ngắn: mỗi đợt lại warmup.
    - (d) Mosaic/zoom kiểu YOLO chưa đo trên CAD.
  - **Cần:** bảng AP theo class; mốc YOLO26x từ COCO chấm bằng evaluator của mình.

- **F21. Wall_Color, VRDet-X khởi đầu từ D-FINE-X COCO** (cả trang 1280, batch 8, 400 query, 50 epoch = 1550 bước):
  - mAP 0 ở epoch 1. Đến epoch 16: 0.273 / 0.123, vẫn đang tăng. 23 s/epoch, 86.5 GB.
  - Bản S khởi đầu từ checkpoint FloorPlanCAD (F20) đã ở khoảng 0.40 / 0.18 ngay khi bắt đầu đợt 2.
  - → Pretrain cùng miền (CAD) quan trọng hơn dung tích ở dữ liệu 250 trang. Chưa có checkpoint X trên CAD.
  - User hỏi khởi tạo ngẫu nhiên có tốt hơn không: không. Vòng cũ train từ đầu chỉ đạt 0.144 AP50.

- **F22. VRDet-X Wall_Color với `lr0=1e-3`** (user tự chạy, gấp 17 lần mặc định):
  - Đến epoch 35: val mAP50-95 0.327, cao hơn run 6e-5 ở epoch 16 (0.123). Chưa so cùng epoch.
  - Đổi lại: feature tầng stride-32 phình khoảng 1e16, running_var của `encoder.input_proj.2` thành inf.
  - Train tiếp ở 1e-3 thì gradient không hữu hạn, 179 tensor BatchNorm hỏng, val NaN, VRAM vọt 65→91 GB.
  - → LR cao hội tụ nhanh hơn nhưng không ổn định với VRDet. Đã thêm `optimizer=auto` (bỏ qua lr0 như YOLO)
    và cơ chế tự phục hồi (nạp last.pt, giảm LR một nửa).
  - **Hướng đáng đo:** tìm LR tối ưu giữa 6e-5 và 1e-3 (ví dụ 2e-4, 4e-4), cùng số epoch.

- **F23. Wall_Color, kết quả tốt nhất user báo (10-09, cách chấm của từng bên):**
  - VRDet-X: **0.63 / 0.35**. VRDet-S: mAP50 khoảng 0.30–0.35. YOLO11x: **0.74 / 0.50**.
  - Tỉ lệ khít (mAP50-95/mAP50): VRDet-X 0.56, YOLO 0.68.
  - VRDet-X suy luận chậm hơn YOLO11x rõ (user đánh giá, chưa đo bằng số).
  - → Thua cả phát hiện lẫn độ khít trên data dày object mảnh, ít ảnh. Phân tích kiến trúc: xem tin trả lời ngày 10-09 (dense-first hybrid).
- **F24. VRDet2 (10-09): kiến trúc mới, chưa có số GPU.** Chi tiết: [docs/VRDET2.md](../docs/VRDET2.md).
  - Thiết kế: dense P2–P5, segment head (vị trí dọc, log-length, độ dày phân phối), o2o chỉ cls và dùng chung box, không NMS.
  - Ngân sách: 55.6M tham số, 0% code mượn, không pretrained. Dùng: `model=vrdet2x`.
  - Overfit CPU 1 ảnh (vrdet2n, 300 bước): block học nhanh; line dài mảnh chậm hơn. Nguyên nhân:
    - top-k chỉ chọn vài điểm, các điểm giống hệt còn lại bị ép thành nền;
    - nhánh box o2o chỉ có 1 mẫu dương.
  - → Đã sửa bằng vùng bỏ qua (o2m) + ưu tiên điểm giữa (o2o) + o2o dùng chung box. Line từ lệch góc 0.4 rad xuống box đúng (154.5/156 px, dày 3.4/3).
  - Chờ log Wall_Color của user để so YOLO11x (0.74 / 0.50).
- **F25. VRDet2-x Wall_Color, train từ đầu (user, 10-09):**
  - Điều kiện: 50 epoch, imgsz 1280, AdamW lr 0.01 → 1e-4, batch 8, 1550 bước. User coi đây là vòng "pretrain", sẽ fine-tune tiếp.
  - Kết quả: **0.272 / 0.095**, vẫn đang tăng ở epoch cuối. So sánh: VRDet1-X (init COCO) 0.63 / 0.35; YOLO11x (pretrained) 0.74 / 0.50.
  - Theo class (mAP50 / mAP50-95):
    | Class | mAP50 | mAP50-95 |
    |---|---|---|
    | door | 0.65 | 0.31 |
    | wall | 0.52 | 0.17 |
    | junction | 0.42 | 0.12 |
    | wall_300 | 0.15 | 0.03 |
    | double_door | 0.15 | 0.04 |
    | slide_door | 0.02 | – |
    | note | 0.01 | – |
  - Loss: `box_loss` 1.08 → 0.42 và `thick` 1.35 → 0.69, nhưng `end_loss` 1.20 → 0.93 gần như đứng yên. Khớp với overfit CPU: tường 450 px đúng dài/dày, tâm lệch khoảng 160 px dọc tường.
  - Tốc độ: 29.3 ms / tile 1280 (PyTorch fp16, batch 1).
  - Kết luận chưa công bằng về kiến trúc vì bị gây nhiễu bởi pretrained (YOLO có COCO). Cần đối chứng: YOLO11x train từ đầu, cùng 50 epoch.
  - Hai lỗi rõ:
    1. Độ khít dọc trục của object dài (tỉ lệ AP50-95/AP50 của wall = 0.32, door = 0.48).
    2. Class hiếm (≤ 60 mẫu val) gần như chết khi train từ đầu.
- **F26. VRDet3 (10-09): thiết kế từ F23–F25, chưa có số GPU.** Chi tiết: [docs/VRDET3.md](../docs/VRDET3.md).
  - Thành phần: backbone HGNetv2-B5 + hybrid encoder pretrained COCO (như v1) + LSK + head dense P3–P5, Fast-NMS ProbIoU 0.7.
  - Head dense: 2 khoảng cách dọc trục dạng DFL (bin = 1 stride) + độ dày DFL (bin = stride / 2) + lệch ngang có dấu.
  - Mục tiêu tính theo hệ trục gần góc dự đoán. Dải ứng viên đa tầng (≥ 1 stride).
  - Tham số: 64.6M, 83% từ code D-FINE. Fine-tune v2 của user (lr 1e-3, batch 16) ở epoch 7 mới 0.280 / 0.098.
  - Có thể khởi tạo từ VRDet1 của user bằng `weights=`.
  - Tiêu chí: ≥ v1 (0.63 / 0.35); mAP50-95 của wall tăng; nhanh hơn v1.
- **F27. VRDet2-x fine-tune vòng 2 trên Wall_Color (user, 10-09): chạm trần, BÁC BỎ hướng v2 cho data nhỏ.**
  - Điều kiện: từ best.pt của F25, lr 1e-3 → 1e-5, batch 16, 100 epoch.
  - mAP50 / mAP50-95 theo epoch: ep1 0.252 / 0.088 → ep30 0.315 / 0.110 → ep50 0.351 / 0.127 → ep66 0.356 / 0.133. Từ ep50 chỉ tăng khoảng +0.005 mỗi 16 epoch.
  - **Loss train gần như phẳng suốt 66 epoch:**
    | Loss | ep1 → ep66 |
    |---|---|
    | box | 0.449 → 0.435 |
    | end | 0.93 → khoảng 0.91 |
    | thick | 0.66 → 0.62 |
    | cls | 0.29 → 0.24 |
    → Model **underfit**: không khớp nổi cả tập train, không phải overfit. Trần do thiết kế, không do thiếu epoch.
  - Tỉ lệ khít mAP50-95/mAP50 đứng yên ở 0.35–0.37 từ đầu tới cuối (v1 0.56, YOLO11x 0.68). mAP tăng là nhờ phát hiện / calibration, box không khít hơn.
  - Nguyên nhân (khớp F24–F25):
    1. Hồi quy dọc trục "vị trí × log(chiều dài)" + L1 đầu mút không học được (end_loss kẹt).
    2. Không có pretrained.
  - → Không đầu tư thêm cho v2. Thay bằng VRDet3 (F26): DFL hai khoảng cách đầu mút, feature COCO.
- **F28. VRDet3-x Wall_Color, đang chạy (user, 10-09).**
  - Điều kiện: init D-FINE COCO, lr 1e-3, batch 16, 100 epoch.
  - Mốc giữa chừng: ep10 0.283 / 0.085; ep20 0.346 / 0.136; ep26 **0.367 / 0.148**. Sau 26 epoch đã vượt mức cuối của v2 (0.356 / 0.133 sau 116 epoch).
  - Loss train giảm rõ, khác hẳn v2:
    | Loss | ep1 → ep26 |
    |---|---|
    | box | 1.53 → 0.51 |
    | dfl | 1.67 → 0.67 |
    | across | 0.23 → 0.11 |
    | angle | 0.13 → 0.06 |
  - Tỉ lệ khít tăng dần: 0.30 (ep10) → 0.40 (ep26). Vẫn dưới v1 0.56 / YOLO 0.68.
  - Train nhanh hơn v2 khoảng 1.5 lần (13 s/epoch ở batch 16, v2 20 s).
  - Chờ kết quả cuối + bảng theo class để so v1 (0.63 / 0.35).
  - User báo ep55: **0.502 / 0.250** (tỉ lệ khít 0.50), gần hội tụ.
  - → Cùng feature COCO, head dense (v3) < decoder DETR (v1, 0.63). Decoder có ích trên data này.
- **F29. Hai thước đo không cùng chuẩn (10-09).**
  - Số YOLO 0.74 / 0.50 do Ultralytics chấm:
    - ProbIoU thay vì IoU đa giác;
    - khớp mỗi detection với GT có IoU cao nhất, rồi mỗi GT lấy detection có điểm cao nhất;
    - AP nội suy 101 điểm (tối đa 0.995).
  - Ví dụ: tường 3 px lệch 1.2 px có IoU đa giác 0.43 nhưng ProbIoU 0.54. DOTA chấm trượt, Ultralytics chấm trúng.
  - Từ commit `2ba3b81`, val in thêm "Ultralytics-style metric" (`vrdet/eval/ultra.py`). Mọi so sánh với YOLO phải dùng dòng này.
- **F30. VRDet4-x (10-09): thiết kế, chưa có số GPU.** Chi tiết: [docs/VRDET4.md](../docs/VRDET4.md).
  - Thành phần:
    - DINOv2 ViT-B/14 reg4 (Apache) + adapter (ViTDet + spatial prior);
    - LSK + encoder/decoder D-FINE-X COCO (v1);
    - nhánh dense = head v3 (Co-DETR aux + đề xuất query DDQ);
    - đầu ra union.
  - 134.8M tham số. Đã kiểm: checkpoint DINOv2 thật nạp khớp 176/176 tensor.
  - Tiêu chí: vượt v1 (0.63 / 0.35) cùng thước. Nếu không vượt, chạy `backbone=hgnet` để tách tác dụng của DINOv2.
- **F31. VRDet3-x sau fine-tune vòng 2 (user, 10-09): ngang YOLO11x khi chấm CÙNG THƯỚC.**
  - Điều kiện: từ best của F28, lr 1e-3, batch 16, 50 epoch. best = ep31, 23.1 ms / tile 1280 (PyTorch fp16), 64.6M tham số.
  - Kết quả:
    | Thước | mAP50 | mAP50-95 |
    |---|---|---|
    | DOTA devkit | 0.593 | 0.312 |
    | **Ultralytics** | **0.720** | **0.474** |
    | YOLO11x (Ultralytics) | 0.74 | 0.50 |
    → Khoảng cách thật với YOLO11x chỉ −0.02 / −0.026. Thước DOTA thấp hơn thước Ultralytics +0.13 / +0.16 trên data tường mảnh.
  - Theo class (thước DOTA, mAP50 / mAP50-95):
    | Class | mAP50 | mAP50-95 |
    |---|---|---|
    | door | 0.90 | 0.64 |
    | wall | 0.76 | 0.34 |
    | junction | 0.69 | 0.28 |
    | double_door | 0.65 | 0.45 |
    | note | 0.62 | 0.34 |
    | **wall_300** | **0.32** | **0.08** |
    | **slide_door** | **0.21** | **0.06** |
  - Init: checkpoint vòng 1 có `encoder.input_proj.2.norm` (BN tầng P5) bị nổ do lr 1e-3. Cùng lớp đã nổ ở v1 (F22).
  - Bài học:
    1. **Mọi so sánh với YOLO phải cùng thước.** Từ giờ val mỗi epoch và chọn best.pt theo thước Ultralytics (`metric=yolo`, mặc định). Kết luận F23 "thua xa YOLO" phần lớn là do thước.
    2. Pretrained + head dense DFL (v3) học nhanh: plateau sau khoảng 30 epoch fine-tune, nhanh hơn v1 (23 ms).
    3. Điểm yếu còn lại là **ngữ nghĩa / class hiếm** (wall_300 vs wall, slide_door vs door, 31–141 mẫu), không phải định vị.
       Độ khít của vật mảnh (wall, junction: tỉ lệ 0.41–0.45) là điểm yếu thứ hai.
    4. BN của `encoder.input_proj` ở P5 dễ nổ với LR cao → giữ lr ≤ 5e-4 cho encoder pretrained.
  - → Việc cần làm: đo lại v1 bằng thước Ultralytics để xếp hạng v1 / v3 / v4 cho đúng.
- **F32. VRDet3-x so với YOLO26x-obb, CÙNG THƯỚC Ultralytics (user, 10-09).**
  - Điều kiện:
    - Cả hai: Wall_Color, val 53 ảnh, imgsz 1280.
    - YOLO26x-obb: 57.6M tham số, 150 epoch, A100.
    - VRDet3-x: 64.6M tham số, best ep40 của một vòng fine-tune 50 epoch.
  - mAP50 / mAP50-95:
    | Class | YOLO26x | VRDet3-x | Chênh |
    |---|---|---|---|
    | all | 0.746 / 0.499 | 0.712 / 0.467 | −0.034 / −0.032 |
    | wall | 0.919 / 0.624 | 0.894 / 0.575 | −0.025 / −0.049 |
    | note | 0.764 / 0.517 | 0.744 / 0.470 | −0.020 / −0.047 |
    | door | 0.965 / 0.811 | **0.975 / 0.820** | +0.010 / +0.009 |
    | slide_door | 0.345 / 0.123 | 0.273 / 0.119 | −0.072 / −0.004 |
    | double_door | 0.494 / 0.392 | **0.589 / 0.476** | +0.095 / +0.084 |
    | **wall_300** | 0.824 / 0.441 | **0.599 / 0.256** | **−0.225 / −0.185** (R 0.33 so với 0.87) |
    | junction | 0.910 / 0.584 | 0.907 / 0.556 | −0.003 / −0.028 |
  - **Riêng wall_300 chiếm khoảng 95% khoảng cách mAP50** (−0.225 / 7 = −0.032 trên −0.034) và khoảng 80% khoảng cách mAP50-95.
    Bỏ wall_300 ra, v3 ngang YOLO26x ở mAP50 (lệch trung bình 6 class còn lại −0.003).
  - Phần còn lại là độ khít (mAP50-95): wall −0.049, note −0.047, junction −0.028.
  - P / R của VRDet trong bảng vẫn tính bằng ghép IoU đa giác nên không so được với P / R của YOLO. Chỉ so mAP.
  - Tốc độ chưa so được:
    - YOLO in 96.9 ms / ảnh lúc val (A100, batch val, gồm cả warmup).
    - VRDet3 23.1 ms (RTX PRO 6000, batch 1, fp16, sau warmup).
  - Việc tiếp:
    1. Tìm vì sao v3 bỏ sót wall_300: nhầm sang wall? (cần confusion matrix).
    2. Hướng kiến trúc: phân loại có điều kiện theo hình học (độ dày dự đoán đưa vào nhánh class).
    3. Xem wall_300 ở v4.
- **F33. VRDet4-x Wall_Color, 1 run 100 epoch (user, 10-09).**
  - Điều kiện: init DINOv2 + COCO; lr 1e-3 (user đặt, gấp 10 lần mặc định); batch 8 (OOM ở 16); 400 query.
  - Kết quả: best ep96 **0.663 / 0.457** (thước YOLO) | 0.587 / 0.306 (DOTA) | **38.1 ms**, 134.8M tham số.
  - Theo class, thước YOLO (mAP50 / mAP50-95):
    | Class | VRDet4 | VRDet3 (sau 3 vòng) | YOLO26x |
    |---|---|---|---|
    | wall | .886 / **.602** | .894 / .575 | .919 / .624 |
    | wall_300 | **.683 / .366** | .599 / .256 | .824 / .441 |
    | note | .650 / .466 | .744 / .470 | .764 / .517 |
    | door | .934 / .784 | .975 / .820 | .965 / .811 |
    | slide_door | .146 / .072 | .273 / .119 | .345 / .123 |
    | double_door | .471 / .363 | .589 / .476 | .494 / .392 |
    | junction | .874 / .544 | .907 / .556 | .910 / .584 |
  - So sánh không cùng ngân sách:
    - v3 0.712 / 0.467 là kết quả sau 3 vòng train (100 + 50 + 50 epoch).
    - **Cùng ngân sách 1 run 100 epoch từ init pretrained** (thước DOTA): v4 0.587 / 0.306, v3 khoảng 0.55 / 0.28 (best vòng 1, đọc từ ep1 của vòng 2) → **v4 +0.035 / +0.028**.
  - Kết luận:
    - DINOv2 + decoder giúp đúng chỗ v3 yếu: wall_300 +0.08 / +0.11, độ khít của wall +0.027.
    - Kém hơn ở class hiếm (slide_door, double_door, note) và door. Decoder one-to-one ít mẫu dương với class hiếm (giống E1b).
    - Chậm hơn 1.65 lần.
  - Ghi nhận phụ: lúc tắt mosaic (ep91), box_loss tăng từ 0.08 lên 0.28, angle từ 0.004 lên 0.12 (đổi phân phối ảnh); mAP vẫn tăng nhẹ.
  - Việc tiếp:
    - Thí nghiệm một yếu tố: **head v3 + backbone DINOv2** (tách tác dụng DINOv2 khỏi decoder).
    - Xem `eval_val.json` → `fusion` (dec / dense / union) để biết class hiếm nhờ nhánh nào.

## 4. Lỗi đã gặp (và kết quả bị vô hiệu)

| Ngày | Lỗi | Ảnh hưởng | Sửa |
|---|---|---|---|
| 10-08 | Fine-tune Wall_Color VRDet-X, `torch.compile` bật (100 epoch ≥ 3000 bước) + EMA kiểu cache tham chiếu: loss train bình thường nhưng 100% prediction của EMA là NaN, kể cả LR mặc định 6e-5. Run v1 (compile tắt) không bị. Nguyên nhân chính xác chưa xác minh (compile × EMA cache, hoặc init từ best.pt) | val = 0 suốt run | EMA quay lại `state_dict` mỗi bước; `repair()` trước mỗi lần val (báo + đồng bộ lại); CLI tắt compile mặc định |
| 10-08 | Fine-tune Wall_Color VRDet-X với `lr0=0.001` (gấp 17 lần mặc định 6e-5): sinh box NaN/inf, bước ghép val crash trong shapely sau epoch 1 | run của user dừng | bỏ box không hữu hạn ở eval/merge/plot (có cảnh báo); CLI cảnh báo khi lr0 > 3 lần mặc định. Kiểu DETR không dùng được LR 1e-3 của YOLO |
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

**User 10-08 13:35: BỎ nhánh vector, chỉ OBB raster.** c3 dừng giữa chừng. Mọi mục vector bên dưới (#13, #18–#20, #24, c3) ngoài phạm vi.
Raster đang chạy: c4 (gộp SOTA), c5 (+P2). Hàng đợi: c6 (+LSK), c7 (+strip), c8 (cỡ M).


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
