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
| yolo26x-wall-color (user, 10-14) | Wall_Color (thước YOLO) | **0.746 / 0.499** | A100, không cùng phiên | 150 ep, init từ checkpoint Wall_Color trước; F59 |

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
- **F34. VRDet5-x (10-09): thiết kế, chưa có số GPU.** Chi tiết: [docs/VRDET5.md](../docs/VRDET5.md).
  - Thành phần: đầu ra = head dense v3 (giữ ưu thế class hiếm), DINOv2 ViT-B (thứ đã giúp v4), cộng 2 phần mới nhắm wall_300:
    1. **Phân loại theo hình học:** 7 số đo dự đoán, detach, conv 1×1 khởi tạo 0, đưa vào nhánh class.
    2. **Chấm lại theo quan hệ:** 2 lớp self-attention trên 600 ứng viên khác nhau, sửa logit class (cộng dư, khởi tạo 0, `loss_rel`).
  - Kiểm chứng: head v5 lúc khởi tạo = head v3 (test).
  - Tham số: 127.9M.
  - Thêm confusion matrix (thước YOLO, conf 0.25, IoU 0.5) vào mọi lần val, để xác nhận wall_300 bị nhầm thành gì.
  - Tiêu chí: vượt v3 (0.712 / 0.467) và YOLO26x (0.746 / 0.499). wall_300 phải > 0.683 (mức của v4).
  - Nếu không đạt, tách tác dụng bằng `backbone=hgnet`, `relate=False`, `geo_cls=False`.
- **F35. VRDet5-x với backbone=hgnet, fine-tune 30 epoch từ `vrdet3x.pt` (user, 10-09). KHÔNG cải thiện; confusion matrix đổi giả thuyết.**
  - Điều kiện:
    - Checkpoint là bản vòng 1 của v3 (BN `encoder.input_proj.2` đã nổ, phải reset), không phải best 0.712.
    - lr 2e-4, batch 16, 450 bước.
    - rotate_p 0.5: có vẻ code cũ, chỉ xoay trong 10 epoch tắt mosaic. angle_loss từ 0.045 lên 0.098 ở ep21.
  - Kết quả:
    - mAP: ep1 **0.672 / 0.433** → best ep10 **0.671 / 0.448** → ep30 0.659 / 0.429 (thước YOLO). DOTA 0.556 / 0.293.
    - 24.9 ms (v3: 23.1).
    - `rel_loss` 0.30 → 0.19 (có học), nhưng mAP đứng yên. Ở 10 epoch cuối (có xoay) mAP giảm nhẹ.
    - Theo class: thấp hơn v3 best (3 vòng) ở mọi class. slide_door .140 so với .273.
  - **Confusion matrix (lần đầu có):**
    | true → predicted | Số lượng | Tỉ lệ |
    |---|---|---|
    | **wall_300 → junction** | **45** | **32% của wall_300** |
    | wall_300 → background | 26 | 18% |
    | slide_door → wall | 14 | 45% |
    | wall → background | 87 | 8% |
    | junction → background | 93 | 5% |
    | wall → junction | 25 | 2% |
    → wall_300 KHÔNG bị nhầm thành wall như giả thuyết F32 (độ dày). Nó bị nhầm thành **junction** (khối nhỏ ở giao tường?). Cần user cho biết wall_300 nghĩa là gì.
  - Kết luận:
    - "v3 + 2 phần mới" không cải thiện trong 30 epoch.
    - Phần chính của v5 (DINOv2, thứ đã giúp wall_300 ở v4) chưa được thử.
    - Chưa so sạch được vì checkpoint khởi đầu, xoay và LR đều khác.

- **F36. Tốc độ infer openobb3x so với YOLO11x-obb (10-09): khác biệt chủ yếu do imgsz, không do kiến trúc.**
  - User đo trên server: openobb3x khoảng 375 ms/ảnh, YOLO11x khoảng 200–230 ms. Nhưng YOLO là `yolo11x_obb_960` (imgsz 960),
    còn openobb3x train và infer ở 1280.
  - FLOPs (thước Ultralytics = 2×MAC, đo cùng máy; YOLO dựng từ yaml, chỉ để đo):
    | imgsz | YOLO11x-obb | openobb3x | Tỉ lệ |
    |---|---|---|---|
    | 960 | 458 G | 511 G | 1.12× |
    | 1280 | 817 G | 909 G | 1.11× |
  - 909 G (openobb3x ở 1280) / 458 G (YOLO ở 960) = **1.98×**, khớp tỉ lệ 375/210 ≈ 1.8 user đo. Cùng imgsz thì openobb3x chỉ nặng hơn khoảng 11%.
  - Phân bổ của openobb3x: backbone HGNetv2-B5 47%, encoder lai 41%, LSK 7%, head 5%.
  - Ghi chú nếu cần tăng tốc sau này (user nói chưa cần): `fast_nms` tính ma trận ProbIoU 4000×4000. Trên CPU tốn khoảng 0.3 s (GPU thì không đáng kể).
    Thời gian CPU local nhiễu, nên chỉ dùng FLOPs để kết luận.

- **F37. Phân tích từng mảnh của OpenOBB3-x đã train (10-12). Chi tiết: [ARCH_ANALYSIS.md](ARCH_ANALYSIS.md).**
  - Đo trên `vrdet3x_1.pt` (bản tốt nhất F31) với một trang Wall_Color thật. Script: `research/analysis/`.
  - **Đường P5 của backbone đã chết ở cả 3 checkpoint v3.**
    - Đầu ra stage 4 có độ lớn trung bình 3566 / 53 / 25 (P3 / P4 chỉ 0.4–0.5).
    - LSK P5 khuếch đại thêm tới 1.3·10¹⁰; BN của `input_proj.2` có running var 3·10²⁶.
    - Đặt P5 của backbone = 0 thì detection không đổi (59/59 tường, 105/105 junction), điểm số lệch tối đa 0.02.
    - Hệ quả: khoảng 75 G (8% FLOPs: stage 4 + LSK P5 + chiếu P5 + AIFI) không đóng góp.
      AIFI nhận đầu vào hằng số nên attention gần đều (khoảng cách 682 px so với 667 px nếu đều tuyệt đối).
    - Nguyên nhân: BN đóng băng ở backbone + fine-tune lr 1e-3 + phép nhân không chặn `x·(1+m)` của LSK.
  - **LSK:** cổng nhánh xa không chọn theo nội dung (trên nét 0.40, trên nền 0.38). Khoảng 95% FLOPs của LSK là conv 1×1, kernel lớn thật chỉ khoảng 3 G.
  - **Định vị không phải nút thắt.**
    - Dịch ảnh 1–7 px: tâm lệch 0.2–0.5 px, độ dày lệch 0.3–0.8 px.
    - DFL không bão hoà; độ dày một đỉnh, bất định khoảng 2.7 px.
  - **Nút thắt là phân loại vật tí hon.**
    - Ở 1280 px: wall_300 cỡ 13 × 7.5 px, junction 11 × 11 px; 86% được phát hiện ở P3.
    - 43% wall_300 có biên top1 − top2 < 0.2.
    - 75–93% bằng chứng của logit class nằm ngoài box. Bán kính nửa bằng chứng khoảng 40 px, 90% khoảng 120–186 px.
  - **Phân bổ FLOPs ở 1280 px** (YOLO26x-obb 905 G so với OpenOBB3-x 909 G):
    | Phần | YOLO26x-obb | OpenOBB3-x |
    |---|---|---|
    | Head | 20% | 5% |
    | Stride 1/4 | 13% | 7% |
    | Khối nặng nhất | | FPN P3 của encoder, 23% |
  - Ghi chú:
    - Trước đây ghi YOLO26x là 578.9 GFLOPs: con số đó lấy từ comment trong file yaml. Đo lại: 225 G ở 640 px, 898–905 G ở 1280 px.
    - YOLO26:
      - bỏ DFL;
      - STAL: phóng box nhỏ hơn stride khi chọn ứng viên;
      - hai head một-nhiều / một-một, trọng số một-nhiều giảm 0.8 → 0.1;
      - pretrain DOTA.
  - Tám nguyên lý thiết kế rút ra (N1–N8) ở mục 7 của ARCH_ANALYSIS. Đó là cơ sở cho kiến trúc mới (chưa viết code, chờ user duyệt hướng).

- **F38. OpenOBB6 (10-12): kiến trúc tổng quát tự thiết kế từ F37 + phân tích các kiến trúc mạnh. Chưa có số độ chính xác.** Chi tiết: [docs/OPENOBB6.md](../docs/OPENOBB6.md).
  - Không thừa kế mạng phân loại. Backbone + neck + head của riêng OpenOBB:
    - StripMix: depthwise 3×3 + dải 1×k nối tiếp k×1 + MLP, residual khởi tạo 0;
    - CSP nửa kênh đi thẳng;
    - hạ độ phân giải có nhánh max-pool;
    - ngữ cảnh theo hàng / cột + một lớp attention ở 1/32;
    - chi tiết stride 4 vào P3 bằng space-to-depth;
    - head class có đơn vị ngữ cảnh;
    - head hộp dị hướng của v3.
  - Bản x: 37.4M tham số, **504 GFLOPs ở 1280** (OpenOBB3-x 909, YOLO26x 905).
    CPU 4 luồng cùng máy: 2.7 s so với 10.3 s (v3) và 7.9 s (YOLO26x). Tốc độ GPU chưa đo.
  - Pretrain: `openobb pretrain` chưng cất DINOv2 (thầy, Apache-2.0) vào backbone trên ảnh chưa gán nhãn (ước tính 30–45 CU với COCO, 20 epoch).
  - Đo tác dụng từng mảnh: `v6_off=strip|axis|attn|spd|maxpool|cls_ctx`. Mỗi lần chỉ đổi một yếu tố, chạy 2 seed để biết mức nhiễu.
  - Data Wall_Color (thống kê ở 1280 px): 96–100% object thẳng trục; junction 9.5 px; wall_300 9.9 × 6 px, cách junction 3.9 px; tường dày 5.6 px.
    Kiến trúc cố ý **không** dựa vào thẳng trục (theo user: phải tổng quát).
  - Tiêu chí:
    - OpenOBB6-x (pretrain) ≥ OpenOBB3-x cùng thước, cùng ngân sách epoch trên Wall_Color;
    - sau đó so YOLO26x trên DOTA / FloorPlanCAD.

- **F39. OpenOBB6-x Wall_Color, train từ đầu, 50 epoch (user, 10-12). Chưa hội tụ.**
  - Điều kiện: không pretrain, batch 16, 750 bước, lr 1e-3 giảm tuyến tính về 1e-5, mosaic, 1280 px.
  - Kết quả cuối (best = ep50):
    - Thước YOLO: **0.379 / 0.196**. Thước DOTA: 0.274 / 0.088.
    - mAP50 vẫn tăng khoảng +0.006 / epoch ở cuối, dù lr đã gần 1e-5.
  - Theo class (thước YOLO, mAP50 / mAP50-95):
    | Class | mAP50 / mAP50-95 |
    |---|---|
    | junction | .715 / .370 |
    | door | .643 / .429 |
    | wall | .641 / .320 |
    | note | .439 / .170 |
    | wall_300 | .145 / .053 |
    | double_door | .059 / .026 |
    | slide_door | .013 / .006 |
  - So cùng số epoch (thước DOTA):
    | Model | Kết quả |
    |---|---|
    | v2 train từ đầu, ep50 (F25; 1550 bước, gấp đôi) | 0.272 / 0.095 |
    | **v6 train từ đầu, ep50 (750 bước)** | **0.274 / 0.088** |
    | v3 init COCO, ep26 / ep55 (F28) | 0.367 / 0.148 và 0.502 / 0.250 |
    → Ở 50 epoch, pretrain COCO cho khoảng gấp đôi mAP50-95. v6 ngang v2 dù chỉ có một nửa số bước.
    Câu hỏi thật là v6 có vượt mức trần 0.356 (thước DOTA) của v2 khi train lâu không.
  - Loss vẫn giảm đều (box 1.45 → 0.54, dfl 1.60 → 0.69): đây là underfit vì thiếu bước, không phải trần.
  - Tỉ lệ khít mAP50-95 / mAP50 = 0.52 (v3 0.66, YOLO 0.67).
  - Lỗi chính (conf 0.25):
    - bỏ sót junction 20%, wall 24%, door 27%: model còn thiếu tự tin;
    - **wall_300 → junction 49%** (v3: 32%), cùng kiểu nhầm của v3;
    - class hiếm gần như bằng 0, đúng như dự đoán khi train từ đầu.
  - Tốc độ lúc val **25.7 ms / tile 1280** (fp16, batch 1, **chưa gộp BN**). v3 là 23.1 ms.
    → FLOPs bằng 55% nhưng GPU không nhanh hơn. Depthwise + nhiều op nhỏ là loại việc GPU bị giới hạn bởi băng thông bộ nhớ.
    Cần đo bản đã gộp BN (`OpenOBB(best.pt)` tự gộp) và tìm chỗ chậm trước khi kết luận về tốc độ.
  - Việc tiếp:
    - User train tiếp từ best.pt. Khuyến nghị: batch 8 (31 bước / epoch), 300 epoch, lr0 1e-3.
    - Nếu chững dưới khoảng 0.5 (thước YOLO) sau 150 epoch → pretrain chưng cất là bắt buộc.

- **F40. OpenOBB6-x train từ đầu, vòng 2 (user, 10-12): chững ở khoảng 0.55 / 0.32, thấp hơn v3 có pretrain rất xa.**
  - Điều kiện: từ best của F39 (0.379 / 0.196), 100 epoch, batch 16, lr 1e-3 → 1e-5. Tổng cộng 150 epoch, khoảng 2250 bước.
  - mAP50 / mAP50-95 (thước YOLO):
    | Epoch | Kết quả |
    |---|---|
    | ep1 | 0.416 / 0.213 |
    | ep30 | 0.492 / 0.283 |
    | ep61 | 0.545 / 0.310 |
    | ep81 | **0.554 / 0.322** |
    - 20 epoch cuối chỉ tăng +0.009 mAP50.
    - Dự kiến kết thúc khoảng 0.56–0.58 / 0.33 (còn 10 epoch tắt mosaic).
  - Loss train vẫn giảm (box 0.51 → 0.43, dfl 0.60 → 0.53, cls 0.28 → 0.21) trong khi val gần đứng.
    → Giới hạn nằm ở **khả năng tổng quát hoá từ 250 ảnh khi không có tiên nghiệm**, không phải thiếu dung lượng hay thiếu bước.
  - So sánh:
    - v3 có pretrain 0.720 / 0.474: v6 kém khoảng 0.16 / 0.15.
    - Ước tính trước đó cho train từ đầu là 0.55–0.65 / 0.33–0.42: kết quả nằm ở cận dưới.
  - Chưa tách được kiến trúc và khởi tạo. Cần đối chứng v3 train từ đầu, cùng điều kiện, với BN / stem của HGNet được mở (bản x đang đóng băng).
  - Log in `[init] D-FINE coco: 997 full...` khi fine-tune từ best.pt: chỉ là nhãn log sai (hàm nạp dùng chung), weights nạp đúng.

- **F41. OpenOBB8 = lõi OpenOBB7 (v3 đã sửa lỗi) + tầng suy luận quan hệ theo hướng (10-12). Gộp thành một bản, theo user. Chưa có số GPU.**
  Chi tiết: [docs/OPENOBB8.md](../docs/OPENOBB8.md), lập luận ở mục 2; lõi ở [docs/OPENOBB7.md](../docs/OPENOBB7.md).
  - Lõi (v7) sửa 4 lỗi F37, mỗi chỗ bắt đầu bằng phép đồng nhất:
    1. chặn độ lớn ở 4 đầu ra backbone (đồng nhất khi RMS ≤ 4);
    2. LSK `x + x·tanh(m)`;
    3. space-to-depth stride 4 vào P3 (BN gamma 0);
    4. StripMix k=11 trong tháp class.

    Thêm `fuse()` gộp mọi BN khi suy luận.
  - Tầng mới (v8): K = 600 ứng viên khác nhau.
    - Mỗi ứng viên được đọc trong khung xoay của nó: lưới 8×3 ở f4 + P3, vòng 18 điểm ở P3 + P4.
    - Transformer cục bộ 2 lớp, rồi 1 lớp quan hệ có bias attention từ hình học tương đối (bất biến khi xoay và khi θ → θ + π).
    - Đầu ra dư khởi tạo 0 cho class và box.
    - Lúc train có bản rung của box nhãn, kèm mask để chúng không lộ đáp án.
  - Kiểm chứng trên CPU:
    - lõi nạp COCO hoặc một v3 chưa train cho ra đúng đầu ra v3;
    - nạp v3 đã train (`vrdet3x_1.pt`) giữ 219/224 detection, điểm lệch trung vị 0.004 (do tanh nén LSK P4 đã học);
    - v8 nạp v7 cho ra `torch.equal` ở bước 0.
    - 146 test pass; train → predict end-to-end chạy được.
  - Chi phí bản x ở 1280:

    | Model | Tham số | GFLOPs |
    |---|---|---|
    | v3 | 64.6M | 909 |
    | v7 | 65.4M | 933 (915 sau `fuse`) |
    | v8 | 66.6M | khoảng 969 (tầng mới +35 G, +3.8%) |
  - Không train v7 riêng: v8 chứa trọn v7, và bắt đầu y hệt (user: không thử mù, gộp bản lý thuyết tốt hơn).
  - Tiêu chí: hơn v3 (0.720 / 0.474) cùng thước; wall_300 (v3 0.599) và mAP50-95 của wall / junction tăng; mục tiêu cuối YOLO26x 0.746 / 0.499.
  - Điều sẽ chứng minh sai: mục 2 của OPENOBB8.md (mỗi mảnh có một phép thử).

- **F42. OpenOBB8-x Wall_Color, đang chạy (user, 10-12): hội tụ nhanh nhất từ trước tới nay.**
  - Điều kiện: init D-FINE COCO, cả trang 1280, batch 16 (15 bước / epoch), 100 epoch, AdamW lr 1e-3 → 1e-5, warmup 45 bước, mosaic (tắt 10 ep cuối).
    Cùng công thức với v3 vòng 1 (F28: lr 1e-3, batch 16, 100 epoch, COCO).
  - mAP50 / mAP50-95 (thước YOLO):
    | Epoch | Bước | Kết quả | P / R |
    |---|---|---|---|
    | 5 | 75 | 0.295 / 0.131 | 0.42 / 0.31 |
    | 10 | 150 | 0.411 / 0.217 | 0.56 / 0.42 |
    | 20 | 300 | 0.493 / 0.279 | 0.61 / 0.48 |
    | 30 | 450 | 0.561 / 0.339 | 0.67 / 0.54 |
    | 37 | 555 | **0.607 / 0.375** | 0.73 / 0.58 |
    - Ep27 → ep37 vẫn tăng khoảng +0.006 mAP50 / epoch.
  - So sánh:
    - v6 từ đầu (F39, cùng thước) đạt 0.379 ở ep50; v8 đạt mức đó ở **ep8**.
    - v3 vòng 1 (F28) chỉ có số thước DOTA: ep26 0.367 / 0.148. Quy đổi thô bằng tỉ lệ DOTA / YOLO của v3 (F31: 0.82 / 0.66),
      v8 ep26 khoảng 0.44 / 0.21 thước DOTA → **nhanh hơn v3 cùng công thức, ước tính**, chưa phải số đo.
    - v3 0.720 / 0.474 là sau 3 vòng (khoảng 200 epoch); chưa so được với 1 vòng của v8.
  - Tỉ lệ khít mAP50-95 / mAP50 = 0.62 và đang tăng (v6 0.52, v3 0.66, YOLO26x 0.67).
  - Recall là phần chậm: P 0.43 → 0.73, R 0.34 → 0.58 (ep6 → ep37). Có thể do class hiếm chưa bắt được; cần bảng theo class.
  - lr 1e-3 không làm nổ BN ở ep37 (v1 / v3 vòng 1 đều nổ ở mức lr này, F22 / F31). Phù hợp với giả thuyết sửa lỗi chặn độ lớn của v7,
    nhưng chưa phải bằng chứng: cần phép thử "đặt P5 = 0 sau train" (OPENOBB8 §2).
  - Chi phí: 17.2 s / epoch so với v3 13 s cùng batch (**+32%**, OPENOBB8 dự kiến +5–10%). GPU 89.2 / 95 GB.
  - Nguyên nhân hội tụ nhanh (giả thuyết, chưa tách): lõi pretrained + head dense một-nhiều; tầng hai học từ bước đầu nhờ bản rung box nhãn;
    lr cao chịu được nhờ chặn độ lớn.
  - Việc tiếp khi xong: bảng theo class + confusion matrix (ô wall_300 → junction); phép thử P5 = 0;
    đối chứng `v8_off=refine` cùng công thức để tách tầng quan hệ khỏi 4 sửa lỗi lõi.
  - Tổng hợp bài học theo lớp model từ v1 tới v8: [LESSONS_BY_LAYER.md](LESSONS_BY_LAYER.md).
    Phát hiện treo trong đó: v1 (0.63 / 0.35 thước DOTA) cao hơn v3 (0.593 / 0.312 thước DOTA) nhưng chưa từng đo thước YOLO,
    nên thứ hạng v3 > v1 trên Wall_Color chưa được chứng minh.

- **F42. Tốc độ suy luận họ v3 + OpenOBB9 (10-12). Chưa có số GPU.**
  - **Tốc độ v3:** user đo v3 chậm hơn YOLO11x khoảng 60% ở cùng 1280, dù FLOPs chỉ hơn 12% (909 so với 812).
    - Đo phân rã trên CPU: phần ngoài mạng (`merge_dets` 9.7 ms, `polys_to_obb` 0.7 ms) không đáng kể. `fast_nms` 4000×4000 mất 432 ms trên CPU.
    - Hai lãng phí:
      1. Không gộp BN: hàng trăm FrozenBN chạy riêng, VGGBlock của encoder chạy 2 nhánh.
      2. NMS không lọc theo `conf` của người dùng.
    - Đã sửa:
      - `fuse_model` (openobb7.py) cho v3 / v5 / v7 / v8 khi nạp;
      - `nms_conf` = min(ngưỡng conf) trước NMS. Kết quả không đổi, vì box điểm thấp không loại được box điểm cao.
    - Checkpoint v3 thật trên CPU: 9019 → 8059 → 7405 ms (−18%). Test bảo toàn đầu ra: `tests/test_infer_speed.py`.
  - **OpenOBB9** (thiết kế: docs/OPENOBB9.md, đã có code, chưa train):
    - lõi conv 3×3 dày tái tham số hoá + RepCSP, giãn 1,2,1,3, SPPF + 1 lớp attention, max-pool khi hạ độ phân giải, chi tiết stride 4 vào P3;
    - head + tầng suy luận quan hệ của v8 giữ nguyên.
    - x: 29.45M tham số, **543 GFLOPs** (v8-x 969). 95.6% FLOPs của lõi + head nằm ở conv dày ≥ 64 kênh.
    - CPU (đã gộp): 0.53–0.57× v8.
    - Pretrain: `openobb pretrain model=openobb9x` chưng cất từ D-FINE-X COCO (P3 / P4 / P5) + DINOv2 (P4 / P5). Có cổng T2 (cosine ≥ 0.8 trên ảnh giữ lại) và in ảnh/s để tính CU thật.
  - Việc tiếp:
    - T1: đo độ trễ GPU của v9-x so với v8-x (weights ngẫu nhiên là đủ), khoảng 0.2 CU;
    - rồi T2 (pretrain 1 epoch để đo tốc độ, sau đó pretrain đầy đủ);
    - rồi T3.

- **F43. OpenOBB8-x Wall_Color, 1 vòng 100 epoch từ COCO (user, 10-12): bằng v3 cùng ngân sách, chưa thấy lợi ích; chậm hơn 35%.**
  - Điều kiện: lr 1e-3, batch 16 (GPU 89 GB), 100 epoch. Cùng công thức với v3 vòng 1 (F28).
  - Kết quả: best ep84 **0.671 / 0.444** (thước YOLO) | DOTA 0.550 / 0.291.
  - Cùng ngân sách (1 vòng 100 epoch, thước DOTA): v3 khoảng 0.55 / 0.28 (F33) → **chênh khoảng 0 / +0.01, trong mức nhiễu**.
    v3 0.720 / 0.474 là sau 3 vòng (khoảng 200 epoch), không so trực tiếp được.
  - Theo class, v8 vòng 1 so với v3 sau 3 vòng (thước YOLO):
    | Class | v8 vòng 1 | v3 sau 3 vòng |
    |---|---|---|
    | wall | .873 / .571 | .894 / .575 |
    | junction | .900 / .550 | .907 / .556 |
    | door | .966 / .802 | .975 / .820 |
    | note | .735 / .454 | .744 / .470 |
    | double_door | .543 / .455 | .589 / .476 |
    | wall_300 | .571 / .223 | .599 / .256 |
    | slide_door | .111 | .273 |
  - **wall_300 → junction 40%** (v3: 32%), slide_door → wall 71%: tầng suy luận quan hệ và ngữ cảnh trong head class **chưa sửa được lỗi chính** sau 1 vòng.
    Theo tiêu chí bác bỏ ở OPENOBB8 mục 2, mảnh "ngữ cảnh head class" và "tầng quan hệ" **chưa được xác nhận**.
  - **Tốc độ:**
    - 31.2 ms / tile (val, chưa gộp BN), v3 là 23.1 ms → **+35%**, trong khi FLOPs chỉ +7%.
    - Thủ phạm khả dĩ là tầng quan hệ: 600 × 43 token + `grid_sample`, `distinct_topk` 1200², lan delta 4000 × 600 ProbIoU. Toàn op nhỏ, nghẽn bộ nhớ.
    - Bộ nhớ train 89 GB ở batch 16.
  - → **Bài học:** mảnh thêm vào phải được đo cả tốc độ GPU (T1) trước khi train.
    v9 đang mang nguyên head + tầng quan hệ của v8, nên T1 của v9 phải đo cả **có / không tầng quan hệ** (`v8_off=refine`).

- **F44. Vòng 2 của v8, kết quả v9, và giả thuyết "tiên nghiệm DOTA" (user, 10-12).**
  - **v8 vòng 2** (50 epoch từ best vòng 1, cùng lịch với v3 F31): best ep36 **0.699 / 0.460** | DOTA 0.576 / 0.299.
    - v3 cùng ngân sách: 0.720 / 0.474 (DOTA 0.593 / 0.312) → **v8 kém −0.021 / −0.014** và chậm hơn 35% (F43).
    - Phần thêm của v8 **không được xác nhận**. **v3 vẫn là bản tốt nhất.**
    - Theo class, v8 so với v3:
      | Hơn | Ngang | Kém |
      |---|---|---|
      | note .796 / .490 so với .744 / .470; junction .916 / .566 so với .907 / .556 | wall, door | slide_door .209 so với .273, double_door .545 so với .589, wall_300 .582 so với .599 |

      Class hiếm chỉ có 25–31 mẫu val nên nhiễu lớn.
    - wall_300 → junction 28% (v3 32%).
  - **Độ khít box không còn là khoảng cách với YOLO.**
    - Tỉ lệ mAP50-95 / mAP50: v8 0.66, YOLO26x 0.67, v3 0.66.
    - Khoảng cách với YOLO26x (0.746 / 0.499) **dồn vào class tí hon / hiếm**: wall_300 .58–.60 so với .824, slide_door .21–.27 so với .345.
      Các class khác ngang hoặc hơn: door .974 so với .965, junction .916 so với .910, note .796 so với .764, double .545 so với .494; chỉ wall .875 so với .919.
  - **v9** (`results.csv` user gửi, vòng 1 100 epoch + vòng 2 50 epoch): **0.541 / 0.331**, ngang v6 train từ đầu.
    Chưa rõ user có chạy `openobb pretrain` không. Nếu không, đây đúng bài học L1. Nếu có, cần log T2 để biết chưng cất có đạt không.
  - Lỗi khung đã sửa: `results.csv` của các bản dense chỉ ghi `angle_loss`, vì cột loss dùng tên của v1. Nay đã map đúng tên, thêm cột `acr_loss` và `ref_loss`.
  - **Giả thuyết H-DOTA (thí nghiệm tiếp theo, có lập luận):**
    - Lợi thế của YOLO26x nằm ở tiên nghiệm **pretrain DOTA** (`yolo26x-obb.pt`): ảnh viễn thám với vật xoay rất nhỏ (xe 10–20 px), đúng loại vật wall_300 (9.9 × 6 px).
    - Bằng chứng: khoảng cách chỉ nằm ở vật tí hon, các class khác ngang (ở trên).
    - Phép thử một yếu tố: v3 khởi tạo từ **lõi HGNet + encoder đã train DOTA** (`runs/e6-vrdet-x-dota-24e`, tách thành `openobb_dota_core_x.pt` 207 MB, nạp 1196/1196 tensor) thay vì COCO. Cùng lịch 100 + 50 epoch.
    - Dự đoán: wall_300 tăng rõ (về phía .8). Nếu không tăng → bác bỏ, khoảng cách do thứ khác (công thức train của YOLO, 150 epoch, gán nhãn).
    - License: weights train trên DOTA chỉ dùng nghiên cứu.

- **F45. Tốc độ v3 / v8 so với YOLO26x trên một trang thật, và nguyên nhân gốc của lỗi wall_300 / note / class hiếm đo trên nhãn (10-12). → OpenOBB10.**
  Script: `research/analysis/speed_*.py`, `research/analysis/v10_*.py`. Chi tiết: [docs/OPENOBB10.md](../docs/OPENOBB10.md).
  - **Tốc độ** (CPU 1 luồng, máy local, chỉ để so tương đối; số tuyệt đối nhiễu do máy thiếu bộ nhớ):
    - Cùng đầu vào 1280 × 1280: YOLO26x (`wall_color.pt`) 7.9 s, v3-x 10.4 s (1.32×), v8-x 14.0 s (1.77×). FLOPs v3 ≈ YOLO (909 so với 905 G).
    - Trang A 1683 × 2381: YOLO chỉ chạy **928 × 1280** (đệm lên bội số 32), còn OpenOBB đệm vuông 1280 × 1280, tức nhiều hơn 1.38× điểm ảnh.
      Khi đó khoảng cách thực tế là khoảng 1.8×.
    - v3-x chạy ảnh chữ nhật 928 × 1280 (position embedding sincos dựng theo lưới thật): 7.8 s so với 10.4 s; 214 / 215 box, 98% trùng khớp.
      → Suy luận chữ nhật thành mặc định cho mọi model, không cần train lại.
    - Phần còn lại của khoảng cách: cùng FLOPs nhưng nhiều phép nghẽn bộ nhớ (depthwise trong HGNet / LSK, tháp class). Nằm trong phần pretrained,
      nên user quyết định không đụng (rủi ro độ chính xác); cũng không cắt nhánh P5 chết.
  - **wall_300 → junction là tranh điểm, không phải thiếu ngữ cảnh** (nhãn train, luật ứng viên của v3):
    - Ở stride 8, mỗi wall_300 có 1.5 điểm, 75% bị tranh (chủ yếu với junction). **72% wall_300 không có điểm riêng** (junction 60%, wall 15%).
    - Tâm wall_300 cách junction 3.9 px và hai box chồng nhau thật: stride 4 vẫn còn 60%. Giải thích vì sao v7 (ngữ cảnh) và v8 (chi tiết stride 4 + quan hệ) không sửa được.
    - Cho mỗi điểm 2 / 3 slot: 17% / **1%** wall_300 không có chỗ (junction 15% / 1%, wall 5% / 1%).
  - **note học box cụt:** DFL với tới 31 bin (P3: 248 px dọc trục). 78% điểm P3 của note có mục tiêu bị kẹp; 39% note không có điểm P3 nào vẽ đủ (P4: 30% điểm).
    Wall 7%, slide_door 2%. Box cụt có điểm cao sống qua NMS thành box trùng.
  - **Class hiếm không được cân bằng:** RFS (t = 0.1) cho hệ số 1.0 mọi class (slide_door ở 22% ảnh, double_door 26%, wall_300 35%),
    trong khi mất cân bằng theo object là 54 : 1 (junction 6986 / slide_door 130).
  - **OpenOBB10** = v8 + (1) head 3 slot / điểm, ghép slot bằng EMD (CrowdDet); (2) ứng viên theo tầm với DFL; (3) trọng số điểm dương
    clamp(√(n_max / n_c), 1, 4). 66.67M tham số (+84k), +0.2% FLOPs. Đầu vào NMS lúc khởi tạo y hệt v8; nạp được checkpoint v8.
    Chưa có số GPU. Tiêu chí bác bỏ ở OPENOBB10 mục 3: ô wall_300 → junction, mAP50-95 note + `Duplicate boxes`, AP slide_door / double_door.
  - Khuyến nghị train: `weights=<best v8>`, 50 epoch; đối chứng (tuỳ ngân sách) `v10_off=slots,reach,balance` cùng checkpoint.
  - **Dự đoán ghi trước khi train** (bảng đầy đủ: OPENOBB10 mục 6): từ v8 best 0.699 / 0.460 → v10 **0.70–0.77 / 0.46–0.51** (giữa khoảng 0.73 / 0.48); wall_300 .582 → .65–.78; note mAP50-95 .490 → .50–.54; wall / door / junction giảm không quá 0.02.
    Slot bị bác bỏ nếu wall_300 < .62 hoặc ô wall_300 → junction (khoảng 28%) không giảm.

- **F46. Phép thử "đặt P5 = 0 sau train" (OPENOBB8 §2): P5 chết ở v3 nhưng SỐNG ở v8 → không cắt P5 ở v10 (10-12).**
  - Trang thật, `research/analysis/p5_zero_test.py` (đặt đầu ra stage 4 của backbone = 0, conf 0.25, CPU):
    | Checkpoint | Độ lớn P5 | Box giữ nguyên khi P5 = 0 | Thay đổi |
    |---|---|---|---|
    | `openobb3x.pt` | 102 | **100%** (214 / 214), điểm lệch 0.000 | không |
    | `openobb8x.pt` | 4.3·10⁴ (vẫn nổ ở backbone) | **80.7%** (264 → 256 box) | note 4 → 6, slide_door 9 → 7, double_door 6 → 4, wall 71 → 67; điểm lệch trung vị 0.02 |
  - Kết luận:
    - Chặn độ lớn (ScaleGuard, v7) **làm P5 sống lại**: backbone vẫn nổ, nhưng đưa về RMS 4 thì đặc trưng có nghĩa, và đóng góp vào class box lớn / hiếm.
      Tiêu chí bác bỏ của mảnh "chặn độ lớn" (OPENOBB8 §2) **không xảy ra** → mảnh này được xác nhận.
    - User đồng ý cắt P5 với giả định nó vô dụng; giả định đó chỉ đúng với v3. Ở v8 / v10 cắt P5 sẽ đổi kết quả → **giữ P5**.

- **F47. OpenOBB10-x Wall_Color, 1 run 100 epoch (user, 10-12): tổng không hơn v8; giả thuyết "tranh điểm" BỊ BÁC BỎ theo tiêu chí ghi trước.**
  - best ep87 **0.693 / 0.456** (thước YOLO). Ep84–100 dao động 0.684–0.697, đã chững.
    So sánh: v8 best (2 vòng, 150 epoch) 0.699 / 0.460; v8 vòng 1 (100 epoch từ COCO, F43) 0.671 / 0.444.
  - Theo class, v10 so với v8 best (v8 vòng 1 trong ngoặc):
    | Class | v10 | v8 best | Đổi |
    |---|---|---|---|
    | wall | .875 / .548 | .875 / .580 (.873 / .571) | 50-95 −.03 |
    | note | .682 / .453 | .796 / .490 (.735 / .454) | **−.11** |
    | door | .968 / .817 | .974 / .816 | 0 |
    | slide_door | .326 / .140 | .209 / .090 (.111) | **+.12** |
    | double_door | .493 / .427 | .545 / .444 | −.05 (25 mẫu) |
    | wall_300 | .649 / .292, R .504 | .582 / .231, R .411 | **+.07 / +.06** |
    | junction | .855 / .517, P .794 R .837 | .916 / .566, P .838 R .896 | **−.06 / −.05** |
  - Đối chiếu dự đoán (OPENOBB10 mục 6):
    - wall_300 .649: sát cận dưới .65. Nhưng **ô wall_300 → junction vẫn 27%** (v8 khoảng 28%) → theo tiêu chí ghi trước, **slot bị bác bỏ**: phần tăng của wall_300 không đến từ việc gỡ tranh điểm.
    - note mAP50-95 .453 < .50 → mảnh "theo tầm với" không được xác nhận.
    - junction giảm .06 (> 0.02) → v10 **làm hại class đang tốt**, vi phạm tiêu chí.
    - Class hiếm tăng (slide_door +.12, wall_300 R +.09): khớp với tác dụng của **cân bằng class**.
  - **Box trùng: 1367 (16.7% số detection)**: junction 761 (43% số junction thật), wall 390, wall_300 105.
    v8 chưa có số đếm này (dòng in ra từ 0.6.2) → cần val lại `openobb8x.pt` bằng code mới để so.
    Giả thuyết: slot phụ cũng bắn vào junction, và NMS IoU 0.7 không gộp được box 9.5 px lệch 1–2 px → P và R của junction đều giảm.
  - Tốc độ val: 45.8 ms / ảnh (batch 16, chữ nhật trung bình 976 × 1280, fused, bf16). Chưa so được với v8 (31.2 ms / tile, batch 1, chưa fuse);
    lần val đầu với nhiều cỡ ảnh chữ nhật còn bị cudnn.benchmark tune lại theo từng cỡ.
  - Bài học:
    1. Thêm slot làm tăng số ứng viên trùng ở vật tí hon, mà NMS ProbIoU không xử lý được. Nếu cần nhiều object mỗi điểm, phải có cơ chế khử trùng một-một (kiểu YOLO26 / DETR), không phải NMS.
    2. Ô wall_300 → junction đứng yên qua mọi thay đổi ở head / ngữ cảnh / chi tiết / slot (v3 → v10). Nguyên nhân nằm chỗ khác: tiên nghiệm (H-DOTA), nhãn, hoặc công thức train.
    3. Cân bằng class là mảnh duy nhất có dấu hiệu tốt; cần thử riêng (`v10_off=slots,reach`) để xác nhận.
  - **Tốc độ `openobb10x.pt` so với `openobb8x.pt`** (trang thật, ảnh chữ nhật 928 × 1280, CPU 1 luồng, xen kẽ trong cùng tiến trình,
    `research/analysis/speed_v8_v10_parts.py`):
    | Phần | v8 | v10 |
    |---|---|---|
    | lõi (backbone + LSK + encoder) | 21.5 s | 12.7 s (cùng kiến trúc: chênh lệch là nhiễu do máy thiếu bộ nhớ) |
    | head | 0.52 s | 1.32 s |
    | tầng quan hệ + decode + phần còn lại | 0.90 s | 2.01 s |
    | NMS | 0.05 s | 0.27 s |
    | box (conf 0.25) | 264 | 297 |
    → v10 **chậm hơn** v8 khoảng 2.1 s CPU ngoài lõi (+15% so với lõi): 3 lần số anchor qua decode / chọn ứng viên / lan delta / NMS, và nhiều box hơn (box trùng).
    Phần tăng tốc của đợt này đến từ suy luận ảnh chữ nhật trong code 0.7.x, áp dụng như nhau cho mọi model, không đến từ kiến trúc v10.
  - **Cảnh báo: CPU local không đo tốc độ được.** Cùng `openobb10x.pt`, cùng đầu vào 928 × 1280, 1 luồng: 9.1 s ở một tiến trình,
    30.5 s ở tiến trình sau. Thứ tự v8 / v10 đảo giữa các lần chạy; YOLO lúc 1.5 s, lúc 7.0 s. Máy thiếu bộ nhớ (commit còn khoảng 4–6 GB).
    - Kiểm tra denormal (`research/analysis/denormal_test.py`): 0% ở mọi stage backbone của v8 / v10 → không phải nguyên nhân.
      Bật flush-denormal nhanh hơn khoảng 30% nhưng lượt có bật luôn chạy sau, nên chưa loại được thứ tự → chưa đổi gì.
    - Kết luận "v10 chậm gấp 10 lần YOLO" (68.7 s so với 7.0 s, các model chạy nối tiếp) **không có cơ sở**.
      Từ nay so tốc độ chỉ trên GPU (Colab, cùng GPU, cùng ảnh); script đưa user ở hội thoại 10-12.

- **F48. Kế hoạch v11 theo cổng (user chọn, 10-13) + lõi DOTA dùng làm thầy chưng cất.**
  - User xác nhận: **v9 (0.541) chưa từng được pretrain** → kết quả đó là bài học L1 (thiếu tiên nghiệm), không phải lõi nhanh hỏng.
  - Lõi DOTA `runs/e6-vrdet-x-dota-24e/openobb_dota_core_x.pt` (HGNetv2-X + encoder, COCO → DOTA 24 epoch) là thầy khoẻ:
    trang thật 640 px, backbone mean|x| 0.58 / 0.85 / 1.21 (không nổ); đặt P5 backbone = 0 đổi P3 62%, P4 78%, P5 96% → **P5 sống**,
    khác hẳn v3 đã fine-tune (P5 chết).
  - `openobb pretrain ... teacher_dfine=<checkpoint .pt>` (0.7.2): thầy `dfine` là lõi của checkpoint đó thay cho D-FINE COCO.
  - Cổng: T1 tốc độ GPU (v9-x không tầng quan hệ so với v8-x và YOLO26x, khoảng 0.2 CU) → H-DOTA (v8-x khởi tạo từ lõi DOTA,
    100 epoch, so F43 v8 vòng 1 từ COCO 0.671 / 0.444, wall_300 .571; khoảng 5 CU) → chưng cất v9 / v11 từ thầy DOTA + DINOv2 → fine-tune.
  - User hỏi pretrain v9 ngay: hướng dẫn = chưng cất (thầy DOTA + DINOv2) rồi fine-tune với `v8_off=refine` (bỏ tầng quan hệ:
    chưa chứng minh lợi ích, tốn 35% thời gian).
  - Agent đã dựng `openobb11` = lõi v9 + head dense + cân bằng class, không tầng quan hệ. **User bác bỏ ngay (10-13), đã gỡ, không commit.**
    Không gói lại các mảnh cũ thành version mới; làm tiếp trên v9 theo hướng dẫn pretrain.

- **F49. Box trùng của v10 trên 53 ảnh val, và OpenOBB11 (10-13).** Script `research/analysis/dup_analysis.py`, kết quả `dup_analysis_openobb10x.json`.
  - Box thô trước NMS của `openobb10x.pt`, thước YOLO trong pipeline phân tích (mốc 0.675 / 0.438; khác số val chính thức vì max_det 300 / conf 0.05):
    | Lọc | mAP50 | mAP50-95 | Box trùng |
    |---|---|---|---|
    | NMS 0.7 | 0.675 | 0.438 | 1191 / 7355 |
    | NMS 0.6 / 0.5 / 0.4 / 0.3 | 0.689 / 0.693 / 0.692 / 0.694 | 0.442 / 0.444 / 0.444 / 0.446 | 354 / 115 / 85 / 66 |
    | NMS 0.7 + sàn độ dày 12 px | 0.682 | 0.440 | 600 |
    | NMS 0.7 + bỏ box nằm ≥ 80% trong box khác | 0.672 | 0.437 | 1063 |
  - Cặp cùng class còn chồng nhau sau NMS 0.7: 2587 chồng lệch, 738 box ngắn trong box dài, 510 gần trùng khít.
  - Phép "trần khử trùng hoàn hảo" (khớp IoU đa giác ≥ 0.5) không phản ánh được box trùng theo ProbIoU của bộ chấm → không dùng.
  - Kết luận:
    - Phần lớn box trùng là bản sao lệch đầu / lệch ngang của vật mảnh, IoU 0.5–0.7.
    - NMS 0.5 bỏ 90% và tăng nhẹ mAP (+0.018); khuyến nghị dùng ngay `iou=0.5` khi predict v10.
    - Nguyên nhân gốc ở kiến trúc: head dense một-nhiều (tới 48 dương / object); NMS IoU mù với vật mảnh; tầng quan hệ của v8 / v10 train một-nhiều (bản sao là dương).
  - **OpenOBB11** ([docs/OPENOBB11.md](../docs/OPENOBB11.md)):
    - tầng quan hệ train một-một (Hungarian, chi phí −p · ProbIoU⁶), mọi bản sao là âm; bản rung box nhãn là nhóm khử nhiễu riêng;
    - suy luận không NMS, không lan delta;
    - head dense như v10 nhưng chỉ giữ cân bằng class.
    - Dự đoán ghi trước: từ v10 best → 0.70–0.73 / 0.46–0.49, box trùng < 5%. Bác bỏ nếu trùng ≥ 5% hoặc mAP50 < 0.673.
  - User (10-13): NMS không được cố định trong code, phải là tham số của lần train → `nms_iou=auto`:
    - train xong thử trên val (v11: không NMS / 0.7 / 0.5; model khác: 0.7 / 0.6 / 0.5), lưu ngưỡng fitness cao nhất vào `best.pt`;
    - cùng cách đó với `infer_k`: K nhỏ nhất của tầng quan hệ mà fitness không giảm quá 0.002 (tốc độ).
    Áp dụng cho mọi model (v3–v11), giá trị đã lưu là mặc định khi predict / val.

- **F50. OpenOBB11 và OpenOBB9 + `pretrain=coco` (user, 10-13): cả hai không đạt; lõi nhanh v9 rất nhanh nhưng thiếu tiên nghiệm.**
  - Cùng G4, val 53 ảnh, thước YOLO, NMS / K tự hiệu chỉnh khi train xong (0.9.1):
    | | v9 + pretrain coco, `v8_off=refine`, 100 ep | v11 từ COCO, 100 ep |
    |---|---|---|
    | mAP50 / 50-95 | 0.539 / 0.311 | 0.641 / 0.413 |
    | Box trùng (conf 0.25) | 3.8% | 1.3% |
    | NMS chọn | 0.6 (0.7: .3333, 0.6: .3338, 0.5: .3311) | 0.5 (không NMS: .355, 0.7: .427, 0.5: .429) |
    | K chọn | — | 150 (fitness .435; K=300 .438; K=600 .429) |
    | Suy luận (batch 16, bf16) | **5.4 ms** + 3.0 ms hậu xử lý | 13.6 ms + 0 |
    | s / epoch | 7.8 | 17.4 |
    Mốc cùng ngân sách từ COCO: v8 0.671 / 0.444 (F43); v10 0.693 / 0.456 (F47). v9 không pretrain: 0.541 / 0.331 (F44).
  - **v11 bị bác bỏ theo tiêu chí ghi trước** (mAP50 < 0.673).
    - Box trùng thấp nhưng bước hiệu chỉnh vẫn chọn NMS 0.5: tầng quan hệ một-một chưa học khử trùng đủ để bỏ NMS.
    - Bỏ sót tăng: wall → background 21% (v10 14%), note 38%, wall mAP50 .769 so với .875.
      Một dương mỗi object học chậm trên 250 ảnh (bài học DETR v1 / v4 lặp lại).
    - (Chưa thử khởi tạo từ v10 như khuyến nghị; khoảng cách −0.03 với v8 cùng ngân sách cho thấy khó đảo ngược.)
    - K = 150 tốt hơn K = 600: bớt ứng viên thì bớt box sai.
  - **Pretrain chưng cất từ COCO không giúp v9** (0.539 so với 0.541).
    - T2 không đạt: cosine giữ lại P3 0.813, P4 0.787 (ngưỡng 0.8), cosine tăng chậm sau epoch 10.
    - 5.25k ảnh (COCO val + 250 bản vẽ), 20 epoch, 1024 px, 41.5 ảnh/s, khoảng 45 phút G4.
  - **Tốc độ**: lõi v9 không tầng quan hệ 5.4 ms, so với v11 13.6 ms (khoảng 2.5 lần).
    Tầng quan hệ (v8 / v10 / v11) chưa lần nào tăng độ chính xác (F43, F44, F47, F50) nhưng chiếm phần lớn thời gian suy luận.
  - Kết luận:
    - cấu hình chính xác nhất vẫn là head dense + NMS (v10 0.693; NMS hiệu chỉnh +0.018, F49);
    - lõi nhanh v9 cần tiên nghiệm đúng miền.
  - Hướng đề xuất: chưng cất lõi v9 từ chính model v10 đã train trên bản vẽ (thầy cùng miền), thay cho thầy COCO.
    Rẻ, không cần nhãn; T2 sẽ cho biết học trò bắt chước được không. `pretrain=dota` (25–40 CU) để sau.
  - Lỗi log: v9 với `v8_off=refine` vẫn in "relational refiner" (đã sửa).

- **F51. OpenOBB12 / OpenOBB13: head dense kép, không NMS, không tầng quan hệ (10-13, chưa train).** [docs/OPENOBB12.md](../docs/OPENOBB12.md).
  - Lập luận từ F43–F50:
    - độ chính xác đến từ lõi pretrained + head dense một-nhiều;
    - tầng quan hệ không tăng độ chính xác nhưng tốn thời gian;
    - một-một kiểu Hungarian (v11) học chậm.
  - Cơ chế: head thứ hai cùng thiết kế với head dense, gán top-1 mỗi object theo cùng thước đo (điểm × ProbIoU^6), kiểu gán kép của YOLOv10 / YOLO26.
    Suy luận chỉ dùng head một-một, không NMS; NMS là dự phòng do hiệu chỉnh cuối train chọn.
  - v12 = lõi OpenOBB7 (68.0M). v13 = lõi OpenOBB9 (30.9M), chưng cất với thầy là OpenOBB10 đã train (`pretrain_teacher=`, mới: `_Teachers` nhận cả detector đã train).
  - Kỳ vọng ghi trước (100 epoch, Wall_Color):
    | | mAP50 / 50-95 | Box trùng, không NMS | Bác bỏ nếu |
    |---|---|---|---|
    | v12 | 0.68–0.71 / 0.45–0.47 | < 3% | mAP50 < 0.673, hoặc trùng ≥ 3% |
    | v13 | 0.64–0.68 / 0.41–0.45; T2 ≥ 0.8; ≤ 6 ms | < 3% | T2 không đạt, hoặc mAP50 < 0.62 |
  - **User 10-13: ưu tiên độ chính xác, v13 (bản tốc độ) tạm hoãn.**
    Chẩn đoán lại: khoảng 95% khoảng cách mAP50 với YOLO26x nằm ở wall_300 (bị nhầm sang junction 27–32%).
    Ô nhầm này đứng yên qua mọi thay đổi head / ngữ cảnh / slot (v3 → v10).
    Khác biệt lớn chưa từng thử: YOLO26x-obb khởi tạo từ weights DOTA, còn OpenOBB từ COCO.
    → Thí nghiệm kế tiếp là **H-DOTA trên v12**: `weights=openobb_dota_core_x.pt`.
    - Lõi DOTA khớp 100% backbone (650 / 650) và encoder (546 / 546) của v12-x; LSK và head khởi tạo mới.
    - Không tốn CU pretrain. So với v12 khởi tạo COCO, cùng 100 epoch.
    - Kỳ vọng ghi trước: wall_300 mAP50 ≥ 0.72 (v10 .649), ô wall_300 → junction < 20%, tổng ≥ 0.71.
      Bác bỏ nếu wall_300 tăng < 0.03 so với v12 khởi tạo COCO.

- **F52. wall_300 tranh điểm lưới với junction (đo trên nhãn, 10-13) → OpenOBB13 mới: luồng độ phân giải gốc + lưới P2.** [docs/OPENOBB13.md](../docs/OPENOBB13.md).
  - Script: `research/analysis/wall300_inspect.py`, `wall300_colour_scale.py`, `wall300_grid.py`. Crop mẫu: `wall300_crops.png`.
  - Ảnh Wall_Color gốc 2000–3000 px. wall_300: 19 × 11 px ở ảnh gốc, chỉ còn 10 × 6 px ở 1280. Nó là đoạn tường ngắn cạnh cửa, nằm sát junction (khoảng cách trung vị 0.4 lần chiều dài).
  - **Màu không phải nguyên nhân:** tỉ lệ pixel hồng 0.063 (gốc) so với 0.057 (1280); 42% junction cũng có hồng.
  - **wall_300 chung ô lưới tâm với junction: 53% ở P3 (stride 8, 1280).** Ở lưới stride 4 còn 20%; stride 4 trên ảnh 2560 còn 6%.
    Khớp với ô nhầm 27–32% và với việc slot (cùng vector đặc trưng) không gỡ được.
  - User (10-13): v13 cũ (lõi nhanh v9) chỉ nhắm tốc độ, nên bỏ và thay bằng bản nhắm độ chính xác.
  - **OpenOBB13 (0.11.0)** = v12 với 3 thay đổi; mọi mảnh mới khởi tạo sao cho P3–P5 lúc đầu = v12:
    - lõi chạy ở `imgsz / 2`;
    - thêm luồng CNN nhẹ trên ảnh đầy đủ, dẫn vào lưới P2 có head kép riêng, và vào P3 (khởi tạo 0);
    - bật `reach`.

    Bản x: 70.1M tham số (+2.1M), 853 GFLOPs ở 2560 × 1920 so với v12 699 ở 1280 × 960.
  - Kỳ vọng ghi trước (100 epoch, COCO, `imgsz=2560`):
    - tổng 0.73–0.77 / 0.48–0.51;
    - wall_300 ≥ 0.75;
    - ô wall_300 → junction < 15%;
    - mAP50-95 của wall và junction tăng ≥ 0.02.

    Bác bỏ nếu wall_300 tăng < 0.05 so với v12, ô nhầm ≥ 15%, hoặc junction / door giảm quá 0.02.

- **F53. v13 cũ (lõi v9 + head kép, 0.10.0), log của user (10-13): 0.560 / 0.321, đã bỏ.**
  - Điều kiện: fine-tune 50 epoch từ `best.pt` của một run v13 trước. **lr 0.005**, gấp 10 lần mặc định 5e-4 của lõi v9 (do user đặt `lr0`).
    mAP sập ở epoch 2–3 (0.368 → 0.044) rồi mới hồi lại: LR quá cao phá điểm khởi đầu. Kết quả bị nhiễu bởi LR, không chỉ do kiến trúc.
  - Theo class: wall .706, slide_door .086, double_door .232, wall_300 .609. Ô wall_300 → junction 17%.
  - **Tín hiệu về head kép (cần v12 xác nhận):**
    - không NMS: 8.4% box trùng (junction 260, wall 112);
    - hiệu chỉnh chọn NMS 0.7 (fitness 0.345 so với "không NMS" 0.319).

    Head một-một chưa đủ để bỏ NMS ở run này.
  - Tốc độ: 8.4 ms / ảnh (batch 16, bf16, G4), so với v11 13.6 ms.
  - Bài học cho v13 mới: dùng LR mặc định (không đặt `lr0`); `imgsz=2560`; khởi tạo COCO.

- **F54. OpenOBB12-x Wall_Color (user, 10-13): 0.722 / 0.468, bản tốt nhất tới nay; head một-nhiều của chính nó còn tốt hơn: 0.741 / 0.481.**
  - Điều kiện: COCO, 100 epoch, 1280, lr 1e-3, batch 16. Best ep96. Hiệu chỉnh chọn head một-một + NMS 0.7
    (fitness: không NMS .4727, NMS 0.7 .4937, NMS 0.5 .4912).
  - Theo class (YOLO26x trong ngoặc):
    | Class | v12 (YOLO26x) |
    |---|---|
    | wall | .835 / .502 (.919 / .624) |
    | note | .697 / .433 |
    | door | .965 / .797 |
    | slide_door | .324 / .171 |
    | double_door | .622 / .518 (.494 / .392) |
    | wall_300 | .708 / .327 (.824 / .441) |
    | junction | .903 / .531 (.910 / .584) |
  - Ô wall_300 → junction còn 11% (v10 27%).
  - **Đối chiếu kỳ vọng F51:**
    - mAP50 ≥ 0.673: đạt (vượt dải 0.68–0.71).
    - Bỏ NMS: **không đạt**. Box trùng 7.1% (có NMS 0.7), hiệu chỉnh vẫn chọn NMS.
    - Nhanh hơn v11 30%: **không đạt**. 15.3 ms so với 13.6 ms, nhưng hai số đo ở hai phiên khác nhau.
  - **Cùng checkpoint, đo trên máy** (`research/analysis/dup_analysis.py`, `DUP_HEAD=o2m`; thước YOLO, 53 ảnh val):
    | Head, hậu xử lý | mAP50 / 50-95 | Box trùng (conf 0.25) |
    |---|---|---|
    | một-một, NMS 0.7 | 0.718 / 0.470 | 263 / 4466 |
    | một-một, NMS 0.4 | 0.726 / 0.471 | 38 |
    | một-nhiều, NMS 0.7 | 0.735 / 0.481 | 354 / 4730 |
    | **một-nhiều, NMS 0.5** | **0.741 / 0.481** | 73 |
    - Phần tăng của v12 so với v10 (0.693) đến từ việc **train kèm head một-một**: gradient một-một làm lõi chung tốt hơn.
      Khi suy luận, head một-nhiều + NMS vẫn tốt hơn head một-một (cách YOLOv10 / YOLO26 bỏ NMS chưa đạt ở 250 ảnh).
    - → 0.12.0: head dùng khi suy luận (`args["infer_head"]`) được hiệu chỉnh trên val cuối train, cùng với NMS.
      Lệnh `openobb calibrate` hiệu chỉnh lại checkpoint đã train.
  - **Độ khít box so với YOLO26x** (`research/analysis/box_quality.py`; px ở thang 1280; trung vị trên các object được khớp; head một-một NMS 0.7):
    | Class | R@.75 v12 / YOLO | Sai số đầu box | Lệch độ dày (bias) | Lệch ngang | Góc (°) |
    |---|---|---|---|---|---|
    | wall | .31 / .56 | 2.42 / 1.30 | **+0.66 / −0.04** | 0.41 / 0.28 | 0.48 / 0.01 |
    | wall_300 | .09 / .24 | 1.42 / 1.05 | +0.93 / +0.28 | 0.40 / 0.33 | 0.81 / 0.04 |
    | junction | .33 / .47 | 0.90 / 0.63 | +0.26 / −0.34 | 0.55 / 0.38 | **2.21 / 0.06** |
    | door | .81 / .84 | 0.94 / 0.85 | +0.47 / +0.28 | 0.59 / 0.50 | 1.26 / 0.17 |
    - Recall ở IoU 0.5 gần bằng YOLO. **Khoảng cách mAP50-95 nằm ở độ khít**:
      - box dày hơn thật một cách hệ thống: +0.66 px trên tường 5.6 px, tức +12% → IoU khoảng −0.1;
      - đầu box sai gần gấp đôi YOLO;
      - góc lệch 0.5–2° (YOLO gần 0).
    - YOLO26 hồi quy trực tiếp, không DFL. OpenOBB đọc độ dày bằng **trung bình** của phân phối DFL, bin = stride / 2:
      - tường 5.6 px ở P4 chỉ là 0.7 bin, nằm sát biên 0 của phân phối;
      - phân phối trải rộng thì trung bình bị kéo lên.

      Giả thuyết đang kiểm tra bằng cách giải mã quanh đỉnh phân phối (`DUP_DECODE=peak`).
  - **v13 mới (log user, 10-13): run không hợp lệ.** Đặt `imgsz=1280`, theo định nghĩa của 0.11.0 thì lõi chỉ chạy ở 640.
    - Kết quả 0.524 / 0.319 (ep87) không kiểm tra được giả thuyết F52.
    - Lỗi giao diện là do agent: `imgsz` của v13 khác nghĩa với mọi model khác.
    - 0.12.0 sửa: `imgsz` luôn là cỡ của lõi; v13 tự chuẩn bị ảnh ở 2 × imgsz (`predict.input_size`).
  - **Giải mã DFL quanh đỉnh** (`DUP_DECODE=peak`, cùng checkpoint, không train lại):
    - lệch độ dày của wall hết (+0.66 → +0.01 px), nhưng sai số tuyệt đối vẫn 1.00 px (YOLO 0.54);
    - recall wall ở IoU 0.5 tụt (.78 → .66), tổng còn 0.678 / 0.443.

    → Nguyên nhân là độ chính xác của hồi quy bằng phân phối theo bin thô, không phải cách đọc. Sửa sau mạng không đủ.

- **F55. `box=direct` (0.12.0): hồi quy box kiểu YOLO26, không DFL. Kỳ vọng ghi trước, chưa train.**
  - Cơ chế: hai khoảng cách tới hai đầu và độ dày là giá trị liên tục (softplus × stride, × stride / 2), loss L1 theo đơn vị stride + ProbIoU.
    Góc, lệch ngang, gán nhãn, head kép giữ nguyên. Một yếu tố duy nhất so với v12 (F54).
  - Điều kiện: v12-x, COCO, 100 epoch, 1280; so với v12 (`box=dfl`) cùng bước hiệu chỉnh head / NMS.
  - Kỳ vọng:
    - wall: sai số độ dày ≤ 0.7 px (v12 1.00, YOLO 0.54); |lệch| ≤ 0.2 px; sai số đầu box ≤ 1.8 px (v12 2.42);
    - R@.75 của wall ≥ 0.40 (v12 .31, YOLO .56);
    - mAP50-95 ≥ 0.49 (v12 cùng cách hiệu chỉnh 0.481).
  - Bác bỏ nếu một trong hai điều sau xảy ra:
    - mAP50-95 không hơn v12 ít nhất 0.01;
    - sai số độ dày của wall không xuống dưới 0.8 px.
  - **Kết quả sơ bộ (user, 10-13): fine-tune 30 epoch từ `openobb12x.pt` với `box=direct`, KHÔNG phải điều kiện ghi trước.**
    - ep30: 0.703 / 0.436, vẫn đang tăng chậm.
    - Mốc cùng cách đo: `openobb12x.pt` với head một-nhiều + NMS 0.7 = 0.740 / 0.479 (`openobb calibrate` trên máy).
    - Lớp ra của box nạp lệch nghĩa: `24 partial`, 3 / 5 kênh nhận logit bin DFL. box_loss ep1 = 1.96, mAP ep1 = 0.25.
    - Chỉ 450 bước, so với 1500 bước của v12.
    - Nghiêng về bác bỏ nhưng chưa kết luận. Cần `best.pt` để đo độ khít box (`box_quality.py`):
      nếu sai số độ dày wall < 0.7 px → chạy đúng điều kiện (COCO, 100 epoch); nếu không → bác bỏ, không tốn thêm CU.

- **F56. Hiệu chỉnh `openobb12x.pt` trên máy (val chuẩn) + tốc độ CPU (10-13).**
  - `openobb calibrate` chọn head một-nhiều + NMS 0.7. File mới: `F:\Downloads\openobb12x_calibrated.pt`.
    | Head / NMS | mAP50 / 50-95 | fitness |
    |---|---|---|
    | một-một, không NMS | .675 / .450 | .4724 |
    | một-một, 0.7 | .720 / .468 | .4934 |
    | một-một, 0.5 | .724 / .464 | .4903 |
    | **một-nhiều, 0.7** | **.740 / .479** | **.5053** |
    | một-nhiều, 0.5 | .745 / .476 | .5026 |
    YOLO26x: .746 / .499.
  - **Tốc độ CPU** (`research/analysis/speed_cpu_v12.py`, 4 luồng, 928 × 1280, xen kẽ trong một tiến trình):
    - với đầu vào nhiễu ngẫu nhiên: YOLO26x 5.4 s; v12 head một-một 5.0 s; v12 head một-nhiều + NMS 17.4 s; v10 18.1 s.
    - Phần chênh 12 s là NMS trên hàng nghìn box rác do nhiễu sinh ra. **Đầu vào nhiễu không dùng được để đo model có NMS.**
    - Trên trang thật (3307 × 2338 → 1280): `predict` của v12 hết **5.7 s với 4 luồng, 15.0 s với 1 luồng**;
      lần gọi đầu (nạp model) 9.4 s. cProfile: 89% là conv của lõi; NMS và ghép box gần 0.
    - → Trên CPU, v12 ngang YOLO26x. Chưa có số GPU cùng phiên (`colab/speed_gpu.py`).

- **F57. Suy luận chậm trên CPU / GPU của user (10-13): hai lỗi triển khai, không phải kiến trúc.**
  - **GPU (RTX 3050, 3 s / ảnh):** máy cài torch bản CPU (`pip install torch` trên Windows mặc định là bản CPU). User sửa xong thì chạy nhanh.
    → 0.12.2 `openobb/torch_setup.py`: tự cài torch bản CUDA khi có GPU NVIDIA, bản CPU khi không có (`OPENOBB_AUTO_TORCH=0` để tắt).
  - **CPU (app Streamlit 25 s / ảnh):** số denormal làm conv chạy theo đường chậm của x86.
    - Cùng tiến trình, 8 luồng, trang 1683 × 2381: 19.2 s → **9.0 s** khi bật `torch.set_flush_denormal(True)`.
    - Đầu ra giống hệt: 378 / 378 box, cùng class và điểm, lệch góc box 0.000 px.
    - F47 kết luận "không phải denormal" chỉ dựa trên đầu ra của backbone; denormal nằm ở activation bên trong.
    → 0.12.3: `load_model` bật flush khi chạy CPU.
    - **ĐÍNH CHÍNH (10-14): hiệu quả tốc độ của flush denormal KHÔNG lặp lại được.**
      - Hook trên 830 module lá: không có đầu vào hay đầu ra nào là denormal; weights cũng không có (0 / 67 triệu).
      - Chạy xen kẽ 4 vòng trong thread mới: có flush 10.0 / 9.1 / 9.4 / 3.6 s, không flush 10.3 / 9.4 / 7.0 / 3.4 s.
      - → Chênh 19 s so với 9 s trước đó là do máy dao động (CPU lai nhân P + E, Chrome và các tiến trình khác chạy cùng),
        không phải do denormal. Flush được giữ vì vô hại (đầu ra giống hệt), nhưng không phải bản sửa tốc độ.
      - Trên máy dev, chỉ nên so tốc độ CPU khi chạy xen kẽ và nhìn trung vị; một lần đo đơn lẻ có thể lệch tới 3 lần.
      - Mốc từ user: `OpenOBB.predict` (test.py, CPU máy user) = **2.9 s** / ảnh 928 × 1280.
  - Thời gian CPU giữa các tiến trình dao động 3.7–19 s cho cùng ảnh. Máy có 10 nhân / 16 luồng, 2.5 GHz, có thể là kiểu lai nhân P + E.
    App giờ dùng số luồng bằng số nhân vật lý (8 luồng nhanh gấp 2 lần 4 luồng).

- **F58. Mổ bên trong các backbone (HGNetv2 đã train, 16 backbone SOTA) → OpenOBB14 (10-14, chưa train).** [docs/OPENOBB14.md](../docs/OPENOBB14.md).
  - User (10-14): phải đào sâu vào bên trong MỌI backbone SOTA, không chỉ lắp thêm lên HGNetv2; mốc so sánh là YOLO26x và
    v12 tốt nhất = head một-nhiều + NMS hiệu chỉnh (0.741 / 0.481, F56), **không phải `box=direct`**.
  - **(a) Can thiệp từng cơ chế trong thân HGNetv2-B5 của `openobb12x.pt`** (`research/analysis/dissect_hgnet.py`, 3 trang val,
    conf 0.25, số = phần detection gốc còn giữ được, cùng class, ProbIoU ≥ 0.5; kết quả `dissect_hgnet.json`):
    | Can thiệp | Giữ |
    |---|---|
    | nhánh dư `stages.1.blocks.1` (/8) = 0 | 86.8% |
    | nhánh dư từng block /16 (4 block) = 0 | 92.2 / 97.4 / 96.8 / 92.1% |
    | nhánh dư `stages.3.blocks.1` (/32) = 0 | 100% (+10 box); nhánh có RMS gấp 16 lần identity (nổ, guard chặn) |
    | stage 4 = 0 | 96.3% |
    | **depthwise 5×5 LightConv stage 3 cắt còn 3×3 / 1×1** | **61.7% / 86.1%** (1×1: box 142 → 216) |
    | depthwise 5×5 stage 4 cắt còn 3×3 / 1×1 | 99.0 / 97.8% |
    | kênh chết (std < 10⁻³ mức trung bình) | stage 1–3 ≤ 3%; **stage 4: 23% chết, 25% yếu** |
    | ESE | **B4 / B5 không có ESE** (`agg='se'`); chỉ B0–B2 có → nhận xét ESE ở ARCH_ANALYSIS chỉ đúng cỡ s / m |
    Kết luận: thân pretrained gần như toàn bộ sống trên bản vẽ; vùng nhìn 5×5 ở stride 16 thiết yếu; chỗ cắt được (stage 4) chỉ 7% FLOPs.
    → **Không đụng thân.** Đột phá phải nằm ở phần sau thân (45% FLOPs của v12) và ở cơ chế định vị.
  - **(b) Roofline từng mảnh** (`research/analysis/roofline.py`, 960×1280, bf16, G4 195 TFLOPS / 1.7 TB/s; byte = vào + ra + tham số của
    từng module lá, chưa tính cat / cộng / nội suy nên là cận dưới):
    | | v12-x | v14-x | YOLO26x |
    |---|---|---|---|
    | thân | 320 G / 2.40 ms | 320 / 2.40 | 332 / 2.73 |
    | LSK | 48 / 0.31 | — | — |
    | encoder / neck | 260 / 1.61 (FPN P3 189 G) | **133 / 0.80** | 206 / 1.07 |
    | head (một head) | 44 / 0.39 | **129 / 0.76** | 67 / 0.41 |
    | bộ tinh chỉnh | — | 4 / 0.04 | — |
    | **tổng** | **682 G / 4.84 ms** | **586 / 4.08** | **605 / 4.34** |
    | FLOP / byte; % thời gian ở 1×1 / depthwise | 156; 39 / 6 | 168; 36 / 4 | 159; 29 / 1 |
    Thân HGNetv2-B5 và thân YOLO26x ngang nhau theo roofline; khoảng cách tốc độ v12 – YOLO nằm hoàn toàn sau thân.
  - **(c) 16 backbone SOTA cùng thước** (`research/analysis/roofline_backbones.py`, kiến trúc không weights, timm + Ultralytics chỉ để đo;
    960×1280; `roofline_backbones.json`):
    | Backbone | M | GFLOPs | GB | F/B | roof ms | % thời gian dày / 1×1 / dw / attn-linear / act |
    |---|---|---|---|---|---|---|
    | HGNetv2-B5 | 33.3 | 320 | 2.38 | 135 | 2.41 | 28 / 43 / 6 / 0 / 22 |
    | HGNetv2-B4 | 13.5 | 134 | 1.43 | 94 | 1.20 | 23 / 41 / 7 / 0 / 27 |
    | YOLO26x / YOLO11x (thân) | 28.8 | 332 | 2.14 | 155 | 2.39 | 50 / 26 / 0 / 0 / 24 |
    | YOLOv8x (thân) | 30.9 | 401 | 1.67 | 241 | 2.53 | 70 / 12 / 0 / 0 / 17 |
    | CSPDarknet53 | 26.6 | 245 | 2.58 | 95 | 2.27 | 46 / 21 / 0 / 0 / 33 |
    | ResNet-50 / 101 | 23.5 / 42.5 | 200 / 382 | 3.19 / 4.74 | 63 / 81 | 2.38 / 3.91 | 21 / 28 / 0 / 0 / 23 |
    | RegNetY-3.2 | 17.9 | 155 | 2.20 | 70 | 1.59 | 15 / 47 / 0 / 0 / 39 |
    | EfficientNet-B5 | 27.3 | **113** | 4.39 | **26** | **2.72** | 1 / 32 / 23 / 0 / 44 |
    | RepVGG-B2 | 86.5 | 1000 | 2.07 | **483** | 5.62 | 83 / 11 / 0 / 0 / 7 |
    | ConvNeXt-B / V2-B (MLP dạng Linear) | 87.6 | 752 | 5.39 / 7.22 | 140 / 104 | 5.53 / 6.60 | 2 / 0 / 5 / 68 / 0 |
    | Swin-B | 86.7 | 766 | 5.74 | 134 | 5.75 | 0 / 0 / 0 / 69 / 12 |
    | DINOv2 ViT-B/14 | 86.6 | 1057 | 3.84 | 275 | 6.49 | 0 / 0 / 0 / 83 / 8 (không có bản đồ /4, /8) |
    | YOLO12x | — | không đo được (khối A2C2f nhận list) | | | | |
    Rút ra: (1) EfficientNet-B5 ít FLOPs nhất (113 G) mà roofline chậm hơn HGNet-B5 (320 G): depthwise + activation nghẽn băng thông,
    cùng bài học F39. (2) RepVGG (conv dày thuần) có cường độ 483 FLOP/byte, gấp 3.5 HGNet: hướng cho một thân tự thiết kế sau này,
    nhưng cần pretrain (F50). (3) Thân YOLO26 và HGNet-B5 ngang nhau; phân bổ FLOPs theo stride cũng gần nhau (/4 26 so với 13%, /8 33 so
    với 36%, /16 30 so với 36%). (4) ViT / Swin / ConvNeXt cỡ B: 750–1050 G và 5.5–6.5 ms, gấp 2.3–2.7 lần, không có /4–/8 → không dùng khi suy luận.
  - **OpenOBB14** (`openobb/models/openobb14.py`, `openobb14_loss.py`; `model=openobb14s|m|l|x`): thân HGNetv2 giữ nguyên (weights COCO nạp
    thẳng) + neck RepCSP conv dày (giãn 1,2,1,3 ở P3 / P4, spd stride 4 → P3, Down có max-pool, 1 attention zero-init ở /32) thay LSK +
    encoder lai + head kép nặng (1×1 + hai conv dày 3×3 mỗi tháp, class giãn 2), `box=direct` mặc định + **bộ tinh chỉnh box theo cạnh**:
    lấy mẫu bản đồ stride 4 (64 kênh) tại 24 điểm trên cạnh / đầu / tim box + P3 6 điểm → MLP → residual (`openobb8.refine_boxes`),
    zero-init, train trên top-256 ứng viên + box nhãn rung, suy luận top-1500 trước NMS. 62.7M (thân 33.2M = 53%; tự viết 47%).
    Test `tests/test_openobb14.py` (train, identity lúc đầu, fuse giữ đầu ra, nạp thân COCO, ánh xạ class theo tên, suy luận).
  - **Kỳ vọng ghi trước** (Wall_Color, 100 epoch, COCO, 1280, cùng hiệu chỉnh head / NMS): mAP50 / 50-95 **0.73–0.77 / 0.50–0.53**
    (v12 0.741 / 0.481; YOLO26x 0.746 / 0.499); wall R@.75 ≥ .45, sai số dày ≤ 0.7 px, đầu ≤ 1.8 px; junction / door không giảm quá 0.02;
    T1 (G4, cùng phiên) ≤ 0.85× v12 và ≤ 1.1× YOLO26x. **Bác bỏ** nếu mAP50-95 không hơn v12 ≥ 0.01, hoặc sai số dày ≥ 0.8 px, hoặc T1 không đạt.
  - Điều chưa biết (v14 là phép đo): encoder pretrained của D-FINE đáng giá bao nhiêu so với neck mới ngẫu nhiên.
    Đối chứng rẻ đề xuất: v12 với encoder khởi tạo lại. Tách tác dụng sau khi có số: `v14_off=edge`, `box=dfl`.

- **F59. Mốc chính thức trên Wall_Color (log user, 10-14): YOLO26x-obb 0.746 / 0.499 so với OpenOBB12-x (head một-nhiều + NMS 0.7) 0.743 / 0.472.**
  - **YOLO26x-obb** (Ultralytics 8.4.153, A100 80 GB): 150 epoch (patience 30), batch 16 (OOM ở 32), imgsz 1280, `optimizer=auto` → AdamW lr 9.09e-4,
    mosaic 1.0 tắt 10 epoch cuối, scale 0.5, translate 0.1, fliplr 0.5, max_det tự nâng lên 397. **Khởi tạo từ `wall_300kaggle.pt`** (một checkpoint đã train
    trước trên cùng data, không phải weights DOTA gốc): epoch 1 đã 0.662 / 0.426. Best quanh ep 125–129 (0.750 / 0.497); best.pt val lại **0.746 / 0.499**,
    P 0.722 R 0.792. 57.6M tham số (fused), 202 GFLOPs ở 640. Speed log: 96.9 ms inference + 18.6 ms postprocess / ảnh (A100, không cùng phiên với OpenOBB).
    0.653 giờ train.
  - **OpenOBB12-x** (0.12.1, G4 95 GB): fine-tune 30 epoch (450 bước) từ `openobb12x.pt` (bản F54), lr 1e-3 cos, batch 16, 1280, mosaic 1 tắt 10 cuối.
    Hiệu chỉnh cuối train: o2o none .4684 | o2o 0.7 .4867 | o2o 0.5 .4839 | **o2m 0.7 .4994** | o2m 0.5 .4948 → **head một-nhiều + NMS 0.7: 0.743 / 0.472**,
    P 0.712 R 0.720, F2 0.669. Box trùng 8.2% (junction 195, wall 139, wall_300 48). **Tốc độ 15.3 ms / ảnh** (batch 16, bf16, fused, ảnh trung bình 976×1280, G4).
    Cả hai mốc đều là vòng train thứ hai trên cùng data (YOLO từ checkpoint Wall_Color, v12 từ checkpoint Wall_Color).
  - **Theo class (v12 so với YOLO26x), mAP50 / mAP50-95:**
    | Class | v12 | YOLO26x | Δ50 | Δ50-95 |
    |---|---|---|---|---|
    | wall (1044) | .877 / .520 | .919 / .624 | −.042 | **−.104** |
    | note (60) | .741 / .459 | .764 / .517 | −.023 | −.058 |
    | door (391) | .976 / .808 | .965 / .811 | +.011 | −.003 |
    | slide_door (31) | .410 / .217 | .345 / .123 | **+.065** | **+.094** |
    | double_door (25) | .572 / .462 | .494 / .392 | **+.078** | **+.070** |
    | wall_300 (141) | .721 / .310 | .824 / .441 | **−.103** | **−.131** |
    | junction (1773) | .900 / .531 | .910 / .584 | −.010 | −.053 |
  - Lỗi v12 thường gặp (conf 0.25): junction → nền 101 (6%), wall → nền 82 (8%), wall_300 → junction 20 (14%), wall_300 → nền 15 (11%).
  - **Phân rã khoảng cách:** mAP50 tổng chỉ −0.003 (class hiếm v12 thắng bù cho wall_300). mAP50-95 −0.027 nằm ở **độ khít box của vật mảnh /
    dày đặc**: wall −.104, wall_300 −.131, junction −.053, note −.058 (cả bốn class đều là tường / nút / khối chữ nhật), trong khi recall và mAP50 của
    wall / junction gần ngang. Đây đúng là lỗi F54 đã đo bằng `box_quality.py` (dày thừa, đầu box, góc). Phần còn lại là **phân loại wall_300** (mAP50 −.103,
    14% nhầm sang junction) → không phải vấn đề box, v14 không nhắm vào đó.
  - Dùng hai mốc này cho OpenOBB14 (thay số F56 trên máy); kỳ vọng v14 ở [docs/OPENOBB14.md](../docs/OPENOBB14.md) mục 4 đã cập nhật theo.

- **F60. OpenOBB14-x trên Wall_Color (user, 10-14, nhiều vòng): tốc độ ĐẠT (10.9 ms so với 15.3 ms của v12, −29%), độ chính xác KHÔNG ĐẠT (0.713 / 0.469 so với 0.743 / 0.472).**
  - Log cuối: fine-tune 20 epoch (300 bước) từ `train5/best.pt` (một vòng v14 trước), lr 1e-4 cos, batch 16, 1280, G4. Best ở epoch 2; cả 20 epoch
    dao động 0.700–0.712 / 0.456–0.469 → đã chững, fine-tune thêm không đổi. Hiệu chỉnh: o2o none .4327 | o2o 0.7 .4693 | o2m 0.7 .4934 | **o2m 0.5 .4938**.
    Log các vòng trước (từ COCO) user không gửi; chỉ biết là không vòng nào hơn v12.
  - Theo class, v14 so với v12 (F59), mAP50 / mAP50-95:
    | Class | v14 | v12 | Δ50 | Δ50-95 |
    |---|---|---|---|---|
    | wall | .878 / **.565** | .877 / .520 | 0 | **+.045** |
    | junction | .899 / .542 | .900 / .531 | 0 | +.011 |
    | door | .970 / .813 | .976 / .808 | −.006 | +.005 |
    | double_door | .634 / .510 | .572 / .462 | +.062 | +.048 |
    | note | .657 / .450 | .741 / .459 | **−.084** | −.009 |
    | slide_door | .276 / .116 | .410 / .217 | **−.134** | **−.101** |
    | wall_300 | .677 / .291 | .721 / .310 | **−.044** | −.019 |
    Box trùng 3.5% (v12 8.2%). Lỗi mới: door → double_door 15 (4%).
  - **Đối chiếu kỳ vọng ghi trước (OPENOBB14 mục 4):**
    - T1 tốc độ ≤ 13 ms cùng điều kiện: **đạt** (10.9 ms; roofline dự đoán −16%, thực tế −29%: phần mượn sau thân nghẽn bộ nhớ hơn ước lượng).
    - mAP50-95 ≥ 0.482: **không đạt** (0.469) → theo tiêu chí, v14 bị bác bỏ ở vai trò "bản chính xác hơn".
    - wall mAP50-95 +0.03: **đạt** (+.045). Cơ chế hồi quy trực tiếp + tinh chỉnh theo cạnh làm đúng việc trên class nó nhắm tới (tường dài, mảnh).
    - junction / door không giảm quá 0.02: đạt.
  - **Chẩn đoán:** phần mất nằm trọn ở note (−.084), slide_door (−.134), wall_300 (−.044): class cần ngữ nghĩa / ngữ cảnh và class hiếm (31, 60, 141 mẫu).
    Đây đúng là điều đã nêu trước là chưa đo: neck tự thiết kế khởi tạo ngẫu nhiên **mất tiên nghiệm COCO của encoder D-FINE** (20.7M tham số), và
    250 ảnh không đủ để học lại ngữ nghĩa cho class hiếm. Tường / nút (nhiều mẫu, hình học thuần) không cần tiên nghiệm đó nên tăng.
    Giả thuyết thay thế (chưa loại): head nặng gấp 3 overfit class hiếm; `box=direct` kém DFL ở vật nhỏ.
  - **Phép thử tách một yếu tố (đã code, 0.13.1): `neck=dfine`** = cùng thân, cùng head kép, cùng bộ tinh chỉnh, nhưng neck là encoder lai pretrained
    của D-FINE (weights COCO nạp thẳng) thay cho Neck14. Nếu note / slide_door / wall_300 quay về mức v12 mà wall giữ +.045 → nguyên nhân là tiên nghiệm
    neck, và hướng đúng là **giữ encoder pretrained + head v14**, hoặc chưng cất Neck14 từ encoder (thầy cùng kiến trúc thân, dễ hơn F50).
    Nếu không → nguyên nhân ở head / box=direct; thử tiếp `box=dfl` và `v14_off=edge`.
    Kỳ vọng ghi trước cho `neck=dfine` (COCO, 100 epoch): mAP50 ≥ 0.74, mAP50-95 ≥ 0.49, wall 50-95 ≥ .55, note ≥ .72, slide_door ≥ .38; tốc độ khoảng 13–14 ms.
  - Ngoài ra `openobb val model=best.pt ref_k=0` (0.13.1) đo đóng góp của bộ tinh chỉnh trên checkpoint đã có, không cần train lại.
  - **Vòng hai hạt giống (user đề nghị train song song, 10-14):** A = `openobb14x neck=dfine`, B = `openobb14x`, mỗi bản trên Wall_Color (100 ep)
    và FloorPlanCAD (24 ep, `colab/data/floorplancad.py`). Lý do thêm FPC: Wall_Color gần trần 250 ảnh (ba model khác nhau trong 0.03 mAP50;
    class hiếm 25–31 object val là nhiễu). Kỳ vọng ghi trước: OPENOBB14 mục 4c.

- **F61. OpenOBB15 (10-14, chưa train): v14 + luồng độ phân giải gốc + lưới P2 + bộ tinh chỉnh đọc luồng gốc.** [docs/OPENOBB15.md](../docs/OPENOBB15.md).
  - User: cần kiến trúc mạnh hơn YOLO **trên chính Wall_Color**, không đi vòng qua DOTA. Phần thua (F59 / F60) nằm ở mAP50-95 của junction 9.5 px,
    tường dày 5.6 px, wall_300 10 × 6 px ở 1280. Cơ chế duy nhất có lập luận vật lý rõ cho lỗi này là **nhiều pixel hơn trên mỗi vật**: ở ảnh gốc
    2381 px chúng to gấp đôi; cùng sai số 1 px thì IoU ở 0.75 giữ được. 53% wall_300 chung ô với junction ở 1280 → 6% ở stride 4 trên 2560 (F52).
    YOLO ở 1280 không thấy pixel này; muốn thấy phải chạy cả mạng ở 2560 (gấp 4 chi phí). v15 chỉ thêm luồng nhẹ 36 G + head P2 40 G.
  - v15 = OpenOBB14 (kế thừa) + luồng gốc (`HiResStream` của v13, 192 kênh ở stride 8 ảnh gốc) + `to_p3` zero-init + P2 (lateral + P3 phóng → RepConv →
    head kép riêng, rộng 80) + `EdgeRefiner(s4=8, s3=16)` đọc luồng gốc thay bản đồ stride 4 của thân. Strides ảnh gốc 16 / 32 / 64 / 8.
    Lõi bắt đầu đúng bằng v14 ở nửa độ phân giải (test). `imgsz` = cỡ lõi, ảnh vào 2× (như v13 từ 0.12.0).
  - Cỡ x: `neck=dfine` 71.1M, 882 GFLOPs ở 1920 × 2560 (v12 682 ở 960 × 1280, +29%); `neck=own` 62.7M, 752 G. v15 nhắm độ chính xác; v14 nhắm tốc độ.
  - Kỳ vọng ghi trước (Wall_Color, COCO, 100 ep, `neck=dfine`): **≥ 0.76 / ≥ 0.51**; junction 50-95 ≥ .58 (v14 .542), wall ≥ .60 (.565),
    wall_300 mAP50 ≥ .78 (.677), note ≥ .72; tốc độ ≤ 20 ms (batch 16, G4). Bác bỏ nếu mAP50-95 < 0.49, hoặc junction 50-95 không hơn v14 ≥ 0.02,
    hoặc wall_300 mAP50 tăng < 0.05.
  - Vòng hai hạt giống: A = `openobb15x neck=dfine`, B = `openobb14x neck=dfine` (khác đúng một yếu tố: luồng gốc + P2 + tinh chỉnh ở độ phân giải gốc).

- **F62. OpenOBB16 (10-14, chưa train): v15 + cascade 3 bước tinh chỉnh box + chấm điểm theo độ khít.** [docs/OPENOBB16.md](../docs/OPENOBB16.md).
  - User: v15 đang train; làm v16. Yêu cầu: chính xác hơn YOLO, chậm hơn được nhưng không quá chậm.
  - Lập luận: mAP50-95 đo độ khít **và** thứ tự xếp hạng theo độ khít. v16 thêm hai bước tinh chỉnh nữa, mỗi bước đọc lại nét trên box bước trước
    (ngưỡng dương 0.5 → 0.6 → 0.7, Cascade R-CNN) và một head `fit` dự đoán ProbIoU của box cuối; điểm class của ứng viên = điểm × fit^0.5 trước NMS
    (GFL / VarifocalNet, áp lên box đã sửa). Zero-init + fit bias +4 → v16 bắt đầu đúng bằng v15 (test). +8 GFLOPs, +1.2M tham số.
  - Cơ chế nằm trong `OpenOBB14` (`CASCADE`, `QUALITY`, `run_cascade`, `fit_scores`), nên v14 / v15 không đổi (CASCADE = 1, QUALITY = False).
  - Cùng đợt (0.15.0): công tắc `v14_off=stage4` (bỏ stage 4 của thân: 45 GFLOPs nghẽn bộ nhớ, 23% kênh chết, 4% detection; neck tự thiết kế lấy
    P5 từ P4 bằng Down, encoder D-FINE nhận P5 = 0) và `v14_off=widehead` (head rộng 2/3), cho bản nhanh của v15 / v16 khi cần.
    `research/analysis/roofline.py` nhận biến thể `openobb15x:neck=dfine,off=stage4`.
  - Kỳ vọng ghi trước (Wall_Color, COCO, 100 ep, `neck=dfine`): **≥ 0.76 / ≥ 0.53**, junction 50-95 ≥ .60, wall ≥ .63, R@.75 wall ≥ .55, ≤ 21 ms.
    Bác bỏ nếu không hơn v15 ≥ 0.01 mAP50-95, hoặc `ref_k=0` trên checkpoint v16 làm mAP50-95 giảm < 0.01.

- **F63. OpenOBB15-x neck mặc định (`neck=own`) trên Wall_Color (user, 10-14): 0.653 / 0.416. Luồng gốc + P2 không giúp class nhắm tới; note sụp.**
  [docs/OPENOBB15.md](../docs/OPENOBB15.md) mục 6a.
  - Không phải hạt giống đăng ký (`neck=dfine`); user xác nhận chạy bản mặc định. Dòng `model` của log ghi "own re-parameterised neck" (cách kiểm
    tra: bản `neck=dfine` ghi "D-FINE COCO hybrid encoder (control)").
  - Điều kiện: 0.14.0, COCO init, 100 ep, ảnh vào 2560 (lõi 1280), batch 16 OOM → 8 (3100 bước, 53 GB), lr 1e-3 cos, G4. Best epoch 97.
    Hiệu chỉnh o2m 0.5 (.4398; o2o none .3903).
  - So v14 own (F60, 0.713 / 0.469): wall +.001, junction −.006, wall_300 +.008 (mAP50-95) → **đứng yên**; note .439 / .205 (−.218 / −.245),
    double_door −.071, slide_door .149 / .067. Lỗi: slide_door → wall 58%, note → nền 27%, junction → nền 5%. Box trùng 4.2%. 14.5 ms / ảnh (batch 8).
  - Bảng kỳ vọng OPENOBB15 mục 4 (viết cho `neck=dfine`): mAP50-95, junction, wall_300 đều **không đạt**; tốc độ đạt.
  - Đọc: với neck tự thiết kế, thêm pixel (luồng 2560 + P2) không cải thiện độ khít của junction / tường / wall_300 → bằng chứng **chống** lại cơ chế
    "nhiều pixel hơn" của F61, nhưng chưa đủ để bác bỏ vì (1) neck không có tiên nghiệm, (2) số v14 own có thêm 20 epoch fine-tune.
    Phần mất lặp lại kiểu F60 (class ngữ nghĩa / hiếm), nặng hơn → càng củng cố giả thuyết "neck tự thiết kế mất tiên nghiệm COCO".
  - Việc tiếp: đúng cặp đã đăng ký `openobb15x neck=dfine` vs `openobb14x neck=dfine`. Nếu có, xin log v14 own từ COCO 100 ep (cùng lịch) để so
    sạch v15 own với v14 own.

- **F64. Chẩn đoán gốc trên Wall_Color (10-14, CPU, checkpoint v12 head một-nhiều + NMS 0.7, 53 ảnh val): thước nhiễu ±0.024; khoảng hụt nằm ở ĐẦU box;
  nhãn là quy ước, không nằm trên nét ảnh; đầu box trùng cạnh object kề bên.** Script: `research/analysis/val_noise_bootstrap.py`,
  `box_error_oracle.py`, `edge_snap_oracle.py`; log `val_noise_v12.log`, `box_error_v12.log`, `edge_snap_v12_r2.log`.
  - **Độ nhiễu của thước** (bootstrap 100 lần trên ảnh val): mAP50 0.735 ± 0.025, mAP50-95 0.478 ± 0.024 (1σ). Theo class (AP50-95): wall ± .017,
    door ± .018, junction ± .032, wall_300 ± .056, note ± .058, slide_door ± .077, double_door ± .091.
    → Khoảng cách v12 với YOLO26x (−0.027 mAP50-95, −0.003 mAP50) chỉ khoảng 1σ; phần lớn chênh lệch giữa v3–v16 (F43, F44, F47, F60, F63) nằm trong 1–2σ.
    Wall_Color 53 ảnh **không xếp hạng được kiến trúc** ở độ phân giải cần thiết. Mọi kết luận một run trên Wall_Color cần chênh ≥ 0.05 (2σ) hoặc hai hạt giống.
  - **Phân rã sai số box** (thay một thành phần của mỗi detection đã khớp bằng giá trị nhãn, chấm lại; Δ mAP50-95):
    | Class | gốc | độ dày | lệch ngang | **hai đầu** | góc | box hoàn hảo |
    |---|---|---|---|---|---|---|
    | wall | .516 | +.070 | +.073 | +.066 | +.013 | +.378 |
    | note | .493 | +.046 | +.053 | **+.160** | 0 | +.387 |
    | door | .808 | +.014 | +.039 | +.060 | +.006 | +.172 |
    | slide_door | .203 | +.014 | +.040 | **+.123** | 0 | +.322 |
    | double_door | .507 | +.015 | +.025 | +.037 | 0 | +.100 |
    | wall_300 | .290 | +.058 | +.023 | **+.235** | 0 | +.487 |
    | junction | .530 | +.035 | +.088 | **+.132** | 0 | +.400 |
    | **all** | .478 | +.036 | +.049 | **+.116** | +.003 | +.321 (mAP50 +.064) |
    → Hình học box chiếm gần hết phần còn thiếu (box hoàn hảo trên cùng detection / điểm: 0.799); phân loại + xếp hạng chỉ còn +0.064 mAP50.
    Thành phần lớn nhất là **vị trí hai đầu dọc trục**; góc không đáng kể.
  - **Nhãn không nằm trên mép nét ảnh:** kéo mỗi cạnh của box NHÃN về mép mạnh nhất trong ±2 px (ảnh gốc) → IoU trung vị với chính nó wall .611,
    junction .598, wall_300 .511, slide_door .658, door .853; chỉ note .989. Làm cùng việc trên detection của v12: −0.090 mAP50-95 (wall −.204, junction −.230).
    → Box của tường / nút / wall_300 là **quy ước của người gán nhãn** (dải quanh nét màu 2–3 px), không phải đường biên thấy được. Cơ chế "nhìn kỹ
    pixel hơn" (v15: luồng 2560, P2) không tự sửa được → khớp F63.
  - **Đầu box dùng chung với object kề bên:** tâm cạnh đầu nằm trên cạnh của object khác trong ≤ 2 px: wall_300 85%, slide_door 77%, wall 66%,
    junction 56% (≤ 0.5 px chỉ 11–32% → nhãn rung khoảng 1 px). Đối tác chính: wall ↔ wall / door / junction, wall_300 ↔ wall / door.
    → Tường là các nét liên tục được **chia đoạn** thành object; đầu của một object là ranh giới với object kế tiếp.
  - **Hệ quả cho hướng nghiên cứu:** mọi version (v1–v16) và YOLO cùng một khuôn: mỗi object tự hồi quy độ dài từ một điểm neo, độc lập với hàng xóm.
    Sai số lớn nhất đo được (đầu box) lại là quan hệ ranh giới giữa các object kề nhau trên cùng nét. Hướng đột phá đề xuất (chưa code, chờ user):
    **detector theo ranh giới / chia đoạn**: trường ranh giới dày đặc (stride 4, giám sát bằng mọi cạnh nhãn) + định vị đầu box bằng tìm kiếm 1D dọc trục
    trên trường đó, hàng xóm dùng chung một ranh giới. Trần đo được: đầu +0.116, ngang + dày khoảng +0.07. Benchmark chính chuyển sang FloorPlanCAD.

- **F65. Log user 10-15 (đang chạy, val trong lúc train): họ v14–v16 kém v12 bất kể neck → giả thuyết F60 "neck mất tiên nghiệm" BỊ BÁC BỎ ở vai trò
  nguyên nhân chính.**
  - `openobb14x neck=dfine` (0.15.0, COCO, 50 ep, batch 12, 1000 bước, 1280): ep43 **0.622 / 0.380**; init nạp 1196 tensor COCO (encoder có tiên nghiệm).
  - `openobb15x neck=dfine` (0.14.0, fine-tune 30 ep từ một `best.pt` v15 trước, 2560): ep29 **0.666 / 0.415**.
  - `openobb16x` neck mặc định (0.15.0, COCO, 100 ep, batch 12, 2560): ep69 **0.604 / 0.390**, còn tăng chậm.
  - So mốc v12 0.743 / 0.472 (F59, vòng hai) và v12 vòng một 0.722 / 0.468 (F54): cả ba thấp hơn 0.06–0.12 mAP50, ngoài 2σ (F64). Có encoder COCO
    (v14 `neck=dfine`) vẫn tệ → phần làm hỏng nằm ở thứ chung của họ v14: head mới (`box=direct`, tower RepConv nặng, khởi tạo box 0) và bộ tinh chỉnh
    cạnh, hoặc thiếu LSK. Chưa tách được; không đầu tư tiếp vào họ v14 (user cũng đánh giá "quá tệ").
  - Hệ quả: **OpenOBB17 xây trên v12**, không trên v14 (F66).

- **F66. OpenOBB17 (10-15, chưa train): v12 + trường ranh giới + bám ranh giới.** [docs/OPENOBB17.md](../docs/OPENOBB17.md).
  - Từ F64: thành phần "hai đầu" +0.116 là phần lớn nhất; nhãn là quy ước (bám mép ảnh −0.09); 56–85% đầu object nằm trên cạnh object kề.
  - Trường 2 kênh (đầu / cạnh dài) ở stride 2, từ stride-4 thân + P3, giám sát bằng dải Gaussian σ 1 px trên mọi cạnh nhãn (quality focal loss).
    Bước bám: 4 cạnh × 3 đường × 33 mẫu trong bán kính tìm, soft-argmax, nhân độ tin cậy và hệ số bám (khởi tạo 0, × 20). Top-1500 ứng viên trước NMS.
    Train: top-256 + box nhãn rung, loss box như bộ tinh chỉnh v14.
  - 68.1M tham số (phần mới 0.125M = 0.18%, tự viết), khoảng +16 GFLOPs (+2.4%). Bắt đầu đúng bằng v12 (test). `v7_off=field` = v12.
  - Kỳ vọng ghi trước (fine-tune 30 ep từ `openobb12x.pt`, cặp v17 / v12 cùng phiên, công thức F59): mAP50-95 ≥ v12 + 0.025; wall_300 ≥ .34,
    junction ≥ .56, wall ≥ .55; thành phần "hai đầu" ≤ +0.08; hệ số bám đầu ≥ 0.3; ≤ 17 ms. Bác bỏ: chênh < 0.01, "hai đầu" giảm < 15%, hoặc hệ số < 0.1.
  - Cùng đợt (0.16.0): **trainer khớp YOLO** theo yêu cầu user: warmup `max(round(warmup_epochs × số batch), 100)` iteration, nhóm bias riêng với LR
    giảm từ `warmup_bias_lr` 0.1, các nhóm khác tăng từ 0 lên `lr0 × lf(epoch)`; `lf` đổi theo epoch; `nbs=64` cộng dồn gradient (tăng dần trong warmup)
    và weight decay nhân `batch × accumulate / nbs`; EMA cập nhật theo bước optimizer. `nbs=0 warmup_bias_lr=0 epoch_lr=False` = công thức cũ.
    Mọi run từ 0.16.0 trở đi không so trực tiếp được với run cũ trừ khi dùng ba tham số đó.

- **F67. OpenOBB18 (10-15, chưa train): lõi CAD nhanh, bản đột phá của vòng hai version (user: mỗi vòng một bản tinh chỉnh + một bản đột phá).**
  [docs/OPENOBB18.md](../docs/OPENOBB18.md).
  - v18 = v17 (thân HGNetv2, head kép DFL của v12, trường ranh giới) với encoder lai + LSK thay bằng `Neck14` (RepConv, rộng 384 = encoder), neck
    được **chưng cất từ encoder của v12 đã train** (thầy cùng miền, cùng thân; sai số bình phương tương đối P3–P5, trọng số 2) và học cùng loss
    detection. Thân + head nạp nguyên từ `openobb12x.pt`. Thầy chỉ dùng khi train (`teacher=`), không lưu trong model.
  - Đo (CPU, `torch.utils.flop_counter`, 960 × 1280, sau fuse): v12 686 GFLOPs (thân 320, LSK + encoder 322, head 44), v17 702, **v18 520 (−24%;
    neck 140)**. Tham số 67.1M → 50.6M; phần mượn chỉ còn thân (33.2M, 66%).
  - Trả lời user "vì sao lõi tự viết luôn tệ": ý tưởng SOTA đã dùng, phần thiếu là trọng số (F39: COCO gấp đôi từ đầu); v14 hỏng ở head (F65).
    v18 không có phần nào học từ đầu.
  - Kỳ vọng ghi trước (60 ep, công thức cũ, khởi tạo + thầy `openobb12x.pt`): ≥ 0.73 / ≥ 0.48, note ≥ .68, slide_door ≥ .33, wall 50-95 ≥ .53,
    **≤ 11.5 ms** G4 batch 16, `loss_distill` < 0.1. Đột phá = tốc độ ≤ 0.75 × v12 và mAP50-95 ≥ v12 − 0.01. Bác bỏ: > 0.85 × v12, < 0.45, hoặc
    note / slide_door tụt > 0.08.
  - Ghi chú vận hành: run v17 đầu tiên của user (0.16.0) chạy công thức YOLO mới (`nbs 64`, accumulate 4: 30 epoch chỉ khoảng 112 bước cập nhật)
    → đã đề nghị chạy lại với `nbs=0 warmup_bias_lr=0 epoch_lr=False`.

- **F68. OpenOBB17-x trên Wall_Color (user, 10-15): 0.754 / 0.485 (head một-nhiều + NMS 0.7), 15.9 ms. Tổng vùng xám; cơ chế trường ranh giới CHƯA
  được xác nhận (class nhắm tới không tăng).**
  - Điều kiện: 0.16.0, công thức YOLO mới (`nbs 64`, accumulate 4, warmup 100 it, bias lr 0.1, lf theo epoch), 30 ep, batch 16, 1280, khởi tạo từ
    `train2/best.pt` = v17 vòng 1 (30 ep từ `openobb12x.pt`, cùng công thức). Tổng: v12 + 2 vòng fine-tune → **không cùng điều kiện F59**.
    Best epoch 20; ep 15–30 dao động 0.743–0.756 / 0.476–0.486 (đã chững). Hiệu chỉnh: o2o none .4889 | o2o 0.7 .5069 | **o2m 0.7 .5121**.
  - Theo class so với v12 F59 (mAP50 / 50-95): wall .844 / .521 (−.033 / 0), junction .896 / .515 (0 / −.016), wall_300 .712 / .270 (−.009 / −.040),
    note .777 / .508 (+.036 / +.049), slide_door .464 / .254 (+.054 / +.037), double_door .613 / .525 (+.041 / +.063), door .973 / .802.
    Lỗi: wall → nền 10%, junction → nền 6%, wall_300 → junction 22%, slide_door → wall 39%. Box trùng 7.5%.
  - So YOLO26x (0.746 / 0.499): mAP50 +0.008 (lần đầu OpenOBB vượt), mAP50-95 −0.014.
  - Đối chiếu OPENOBB17 mục 4: tổng +0.013 (vùng xám +0.01..+0.025); wall ≥ .55 không đạt (.521), junction ≥ .56 không đạt (.515), wall_300 ≥ .34
    không đạt (.270, còn tụt); tốc độ ≤ 17 ms đạt. Phần tăng ở class hiếm (nhiễu ±0.06–0.09, F64) và có thể do 2 vòng train thêm.
  - Việc tiếp: đọc `snap.gain` + chạy `box_error_oracle.py` trên `best.pt` (CPU); nếu cần, v12 qua đúng 2 vòng cùng công thức làm đối chứng.

- **F69. Box trùng là chuyện chọn ngưỡng NMS; độ khít của vật mảnh phải sửa ở hồi quy (10-15, CPU, ứng viên thô của v12 một-nhiều, 53 ảnh val).**
  Script `research/analysis/postproc_oracle.py`, log `postproc_v12.log`.
  - | Hậu xử lý | mAP50 | 50-95 | box trùng |
    |---|---|---|---|
    | NMS 0.7 (đang dùng) | 0.733 | 0.479 | 9.7% |
    | **NMS 0.5** | **0.740** | **0.478** | **1.6%** |
    | NMS nới vật mảnh ≥ 12 px | 0.735 | 0.479 | 5.6% |
    | bỏ phiếu box (v 0.5–0.6) | 0.730–0.732 | 0.472–0.474 | 9–12% |
  - Xếp hạng không phải vấn đề: ProbIoU của ứng viên điểm cao nhất so với ứng viên tốt nhất (trung vị) wall .805 / .837, junction .818 / .850,
    wall_300 .717 / .723 → chênh trung vị 0.000–0.008. Bỏ phiếu box làm hại. → Các ứng viên tự chúng chưa đủ khít; phải sửa bước hồi quy.
  - Sửa khung (0.17.1): hiệu chỉnh cuối train và `openobb calibrate` chọn fitness cao nhất, nhưng trong khoảng 0.005 thì chọn cài đặt ít box trùng hơn
    (`predict.choose_postprocess`, `DUP_TOL`). v12 F59 (.4994 / .4948) và v17 F68 (.5121 / .5087) đều sẽ chọn NMS 0.5. Cơ chế một-nhiều + NMS của F54
    giữ nguyên; chỉ đổi ngưỡng.
  - Vòng 2 của v17 (bản tinh chỉnh): `box=direct` (head v12 hồi quy trực tiếp kiểu YOLO26; bằng chứng duy nhất: v14 wall 50-95 +0.045, F60),
    fine-tune từ `best.pt` F68, cùng công thức với F68. Khi nạp checkpoint DFL, hàng khoảng cách / độ dày của lớp ra được đưa về 0 (góc / lệch ngang giữ).
    Kỳ vọng ghi trước: wall 50-95 ≥ .56, wall_300 ≥ .30, junction ≥ .54, tổng ≥ 0.495 / mAP50 ≥ 0.745, box trùng ≤ 3%. Bác bỏ: wall 50-95 < .535.

- **F70. OpenOBB18-x chạy SAI thiết kế (user, 10-15): không `weights=`, không `teacher=` → 0.623 / 0.369; tốc độ ĐẠT 12.2 ms.**
  - Log: "own re-parameterised neck (no teacher)", init "D-FINE COCO weights": neck học từ đầu trên thân COCO, head khởi tạo mới = đúng kịch bản v18
    được thiết kế để tránh (giống v14 own, F60). 0.17.0, công thức YOLO mới (`nbs 64`, 100 ep, khoảng 375 bước cập nhật), batch 16, G4. Best ep 97, còn tăng.
  - Theo class: wall .809 / .435, junction .845 / .450, wall_300 .534 / .183, note .707 / .407, door .959 / .754, slide_door .126 / .045,
    double_door .382 / .311. Lỗi: wall_300 → junction 37%, slide_door → wall 68%. Box trùng 10.3%.
  - **Tốc độ: 12.2 ms / ảnh so với v17 15.9 ms (−23%) và v12 15.3 ms (−20%)**, cùng G4 batch 16 → phần tốc độ của v18 được xác nhận
    (kỳ vọng ≤ 11.5 ms: gần đạt; FLOPs −24%).
  - Không phải phép thử giả thuyết độ chính xác của v18 (chưng cất từ v12 đã train). Sửa CLI (0.17.2): `model=openobb18x weights=X` tự dùng X làm
    `teacher=`; thiếu `weights=` thì cảnh báo. Chạy lại đúng thiết kế.

- **F71. Kế hoạch vượt YOLO cả ba mặt (user 10-15: chính xác hơn ở mAP50 và 50-95, nhanh hơn, ít box trùng). Ghi trước.**
  - Box trùng: đã giải bằng hiệu chỉnh (F69). Tốc độ: v18 −20% (F70). Độ chính xác: mọi kiến trúc + YOLO dừng quanh 0.74–0.75 trên 250 ảnh → giới hạn
    là lượng bản vẽ đã học (F39, F21), không phải sơ đồ.
  - Bước 1 (cặp song song): A = v18 đúng thiết kế trên Wall_Color (`weights=openobb12x.pt`, thầy tự bật; 60 ep, công thức cũ). B = **v18 pretrain
    detection trên FloorPlanCAD** (khoảng 7k bản vẽ có nhãn, 30 class, 24 ep, 1024, từ COCO; neck có đủ data để học).
  - Bước 2: finetune Wall_Color từ `best.pt` của B (CLI không tự lấy checkpoint v18 làm thầy, 0.17.3), thử cả `imgsz=1440` (tốc độ dư của v18 trả cho
    pixel: 1440² × 0.8 ≈ thời gian của v17 ở 1280).
  - Kỳ vọng ghi trước: B trên FloorPlanCAD val ≥ 0.78 / ≥ 0.70 (YOLO26x 0.802 / 0.750; VRDet-S c6 0.782 / 0.688). Bước 2 trên Wall_Color
    ≥ 0.76 / ≥ 0.50 (vượt YOLO26x 0.746 / 0.499 cả hai), ≤ 12.5 ms ở 1280, box trùng ≤ 3%. Bác bỏ giả thuyết "data bản vẽ": bước 2 không hơn A ≥ 0.01
    mAP50-95.

- **F72. Khuôn "một điểm lưới, một object" là nút thắt trên bản vẽ → OpenOBB19 (head theo nhóm tranh chấp). (10-15, CPU)**
  [docs/OPENOBB19.md](../docs/OPENOBB19.md). Script `research/analysis/miss_by_length.py`, `openobb/models/groups.py`.
  - Recall (ProbIoU 0.5, conf 0.25) theo độ dài, v12 / YOLO26x: wall < 12 px **0.33 / 0.69** (72 object), 12–24 px 0.72 / 0.85 (151),
    ≥ 24 px 0.94–0.97 / 0.89–0.96; wall_300 < 12 px 0.63 / 0.75. → khoảng cách wall mAP50 (−0.075) nằm ở tường ngắn.
  - Điểm lưới riêng (stride 8, vùng ứng viên = box nới ≥ 8 px như `assign_slots`): không có điểm riêng junction 60%, wall < 12 px 86%,
    wall_300 < 12 px 86%. Stride 4: 72%; ảnh 1920: 76%; nới 16 px (STAL): 93%. Đối thủ: junction 753, wall 245, door 99.
    Ma trận tranh chấp (nghìn): junction ↔ wall 7.4, junction ↔ door 1.6, wall ↔ door 1.3, wall ↔ wall 2.4.
    Tách nhóm junction / wall_300 / còn lại: junction 2%, wall_300 1%, wall ngắn 39%.
  - Hiệu chỉnh `openobb17x.pt` theo luật F69 (CPU): o2m NMS 0.5 → 0.754 / 0.482, box trùng 7.3% → **2.2%** (file `F:\Downloads\openobb17x_calibrated.pt`).
    `snap.gain × 20` học được: đầu 1.53, cạnh 1.05 → model có dùng bước bám ranh giới.
  - OpenOBB19 = v18 + head nhóm (G = 3 tự chia: junction | wall, wall_300 | các loại cửa, note). Công tắc `groups=auto` cho v17 / v18.
    Chi phí < 0.2% FLOPs. Bắt đầu đúng bằng model một nhóm (test). Kỳ vọng ghi trước: OPENOBB19 mục 4.

- **F73. v17 train thêm 30 ep (user, 10-15; định chạy A = `groups=auto` nhưng thiếu cờ, log không có `[groups]`): 0.749 / 0.483, box trùng 2.0%.**
  - 0.18.0, khởi tạo `openobb17x.pt` (F68), công thức YOLO mới (`nbs 64`), 30 ep, batch 16, G4, 16.0 ms. Best ep 26; ep 14–30 dao động 0.746–0.761 / 0.482–0.487.
  - Hiệu chỉnh mới (F69) chọn **o2o 0.5** (fitness .5099, dup 2.0%) thay o2m 0.7 (.5137, dup 6.8%): box trùng 2.0%, mAP giữ trong nhiễu.
  - Theo class: wall .840 / .523, junction .900 / .505, wall_300 .763 / .308, note .703 / .463, door .972 / .800, slide_door .411 / .230,
    double_door .653 / .554. Lỗi: wall → nền 11%, junction → nền 7%, wall_300 → junction 13%.
  - So F68 (0.754 / 0.485): ngang → **v17 đã chạm trần trên Wall_Color**; train thêm không tăng. Phép thử A (`groups=auto`) cần chạy lại đúng cờ.

- **F74. Phép thử A: `openobb17x groups=auto` (user, 10-15): 0.757 / 0.487 (o2m 0.7), box trùng 7.4%. Nhóm giúp wall và junction, wall_300 tụt.**
  - Điều kiện y hệt F73 (khởi tạo `openobb17x.pt`, công thức YOLO mới, 30 ep, batch 16, 1280, G4), khác đúng một yếu tố: 3 nhóm tự chia
    (note, junction | wall, wall_300 | door, slide_door, double_door). 68.27M tham số, 16.1 ms (F73: 16.0 ms). Best ep 29, còn nhích.
  - So F73: wall .872 / .534 (+.032 / +.011; wall → nền 118 → 79, −33%), junction .904 / .531 (0 / +.026; junction → nền 122 → 100),
    note .755 / .493 (+.052 / +.030), slide_door .513 / .256 (+.10 / +.03), door .976 / .808; wall_300 .691 / .273 (−.072 / −.035;
    wall_300 → junction 13% → 21%), double_door .591 / .518 (−.06 / −.04). Tổng +.008 / +.004 (trong nhiễu).
    wall AP50 nhiễu ±0.010 (F64) → +.032 là thay đổi thật; junction 50-95 ±0.032 → +.026 chưa chắc; wall_300 ±0.07 → −.072 khoảng 1σ.
  - So YOLO26x (0.746 / 0.499): mAP50 +0.011, mAP50-95 −0.012.
  - Đối chiếu OPENOBB19 mục 4 (A): mAP50 ≥ 0.775 không đạt; wall ≥ .88 không đạt (.872); wall_300 ≥ .76 không đạt; junction 50-95 ≥ .53 đạt;
    box trùng ≤ 3% không đạt (hiệu chỉnh giữ 0.7: o2m 0.5 .5077 so với .5144, chênh .0067 > DUP_TOL 0.005). Không bị bác bỏ (wall mAP50 +0.032 ≥ 0.02).
    Cần `best.pt` để đo recall tường < 12 px (`miss_by_length.py`).
  - Chẩn đoán wall_300: nhóm cho phép junction và wall_300 cùng có box ở một điểm → phân loại phải tự phân biệt; nhầm wall_300 → junction tăng.
  - Sửa khung (0.18.2): `DUP_TOL` 0.005 → 0.01 (dưới một nửa nhiễu thước ±0.024; user ưu tiên ít box trùng). Run này sẽ chọn o2m 0.5 (2.2%).

- **F75. OpenOBB19-x vòng 2 (user, 10-15): 0.755 / 0.462, 12.4 ms, box trùng 2.8%. Nhanh + ít trùng + mAP50 cao nhất; mAP50-95 hụt (độ khít).**
  - Điều kiện: 0.18.0, khởi tạo `best.pt` của v19 vòng 1 (log vòng 1 chưa có: 60 ep từ `openobb12x.pt`, thầy v12; công thức chưa rõ), 30 ep,
    công thức YOLO mới (`nbs 64`, khoảng 112 lần cập nhật), **không thầy** (weights là v19), batch 16, 1280, G4, 70.6 GB. Best ep 29, còn nhích.
    Hiệu chỉnh: o2m 0.5 (.4909, dup 2.8%) so với o2m 0.7 (.4874, 9.6%), o2o none .4270.
  - Theo class (so A = v17 + nhóm F74): wall .879 / .521 (+.007 / −.013; recall .818, cao nhất từ trước), junction .887 / .496 (−.017 / −.035),
    wall_300 .763 / .287 (+.072 / +.014), note .718 / .441 (−.037 / −.052), door .970 / .793, slide_door .547 / .279, double_door .522 / .413
    (−.07 / −.105; double_door → door 52%). Lỗi: junction → nền 8%, wall → nền 8%, wall_300 → junction 23%. P 0.672 (A 0.711).
  - So v12 (F59): mAP50 +0.012, mAP50-95 −0.010, tốc độ −19% (12.4 so với 15.3 ms), box trùng 2.8% so với 8.2%. So YOLO26x: +0.009 / −0.037.
  - Đối chiếu OPENOBB19 mục 4 (B): ≤ 12.5 ms đạt; box trùng ≤ 3% đạt; ≥ 0.77 / ≥ 0.50 không đạt. OPENOBB18 "đột phá" (≤ 0.75 × v12 tốc độ và
    50-95 ≥ v12 − 0.01): tốc độ 0.81 × không đạt sát; 50-95 −0.010 vừa chạm ngưỡng.
  - Chẩn đoán: neck chưng cất tìm object tốt (recall, mAP50) nhưng box kém khít hơn encoder (junction / door / note 50-95 tụt), độ chính xác (P) thấp hơn.
    Vòng này neck không có thầy và chỉ khoảng 112 lần cập nhật. Phép thử rẻ: train tiếp với `nbs=0` + `teacher=` = A (v17 + nhóm).
    Cần `best.pt` của v19 và A để chạy `box_quality.py` / `box_error_oracle.py`.

- **F76. Luật hiệu chỉnh NMS cuối cùng (0.18.4) trên v17 + nhóm (F74): chọn o2m 0.5 → 0.751 / 0.481, box trùng 2.2%** (so o2m 0.7: 0.759 / 0.488, 7.5%).
  0.18.2 (chỉ "trong 0.01 thì ít trùng nhất") đã chọn o2o 0.5 (0.742 / 0.479, 2.1%): bớt 0.1 điểm trùng mà mất 0.009 mAP50 → 0.18.4 coi các mức trùng
  chênh < 0.5 điểm là ngang nhau rồi chọn bản chính xác hơn (`DUP_EQ`). Calibrate in phiên bản + ngưỡng ở dòng đầu (0.18.3).

- **F77. Vòng nhắm mAP50-95 (10-15, chưa train): v19 + chưng cất box (LD) và OpenOBB20 (bộ tinh chỉnh cạnh đọc thân stride 4).**
  [docs/OPENOBB20.md](../docs/OPENOBB20.md).
  - User: v19 đạt mAP50 / tốc độ / box trùng; chỉ còn mAP50-95. Hai bản song song: cải tiến v19 + bản agent chọn.
  - Chưng cất box (`openobb18_loss.py`, `--w-ld 0.25`, `--w-kd-cls 0.5`, tự bật khi có `teacher=` và head khớp): KL T = 10 trên phân phối DFL
    (hai đầu, độ dày), L1 góc / lệch ngang, BCE quality-focal tới xác suất class của thầy; theo nhóm (thầy một nhóm cũng dùng được).
    Thầy = v17 + nhóm (F74 / F76: box khít hơn v19, cùng head nhóm).
  - OpenOBB20 = v19 + `EdgeRefiner` (v14) đọc thân stride 4 + P3 trên cạnh box, trước bước bám ranh giới; zero-init (bắt đầu đúng bằng v19).
    Lý do: neck nhanh làm mất độ khít (F75); bộ tinh chỉnh đi vòng qua neck; bằng chứng F60 (wall 50-95 +0.045).
  - Kỳ vọng ghi trước: bản 1 ≥ 0.475 mAP50-95, junction ≥ .52; bản 2 ≥ bản 1 + 0.01, wall ≥ .55, ≤ 13.0 ms; cả hai box trùng ≤ 3%, mAP50 ≥ 0.75.
    Mục tiêu đột phá ≥ 0.499 ở ≤ 13 ms. Bác bỏ bộ tinh chỉnh nếu bản 2 không hơn bản 1 ≥ 0.005.

- **F78. Bản 1 của F77 chạy LỆCH thiết kế (user, 10-15): v19 với neck mới, thầy một nhóm → 0.752 / 0.464, 12.4 ms, box trùng 2.9%.**
  - Điều kiện: 0.19.0, `openobb19x`, 30 ep, batch 16, 1280, G4, `nbs=0` (warmup 45 it), cos. `weights=.../openobb19x.pt`, `teacher=.../openobb17x_v2.pt`.
  - **Kiểm tra file (CPU, bản tải về `F:\Downloads`):** `openobb19x.pt` là **v17 + nhóm** (run A, F74: `arch v17`, `groups 1,0,2,2,2,1,0`, 68.5M,
    0.18.0, ep 28), không phải v19. Nạp vào v19-x cho đúng `1084 full, 12 partial, 328 new`, 328 tensor mới đều là `neck` → neck học lại từ đầu
    (ep 1: 0.131 / 0.060). `openobb17x_v2.pt` là **v17 một nhóm** (F73: `groups` rỗng, ep 25). Vậy thực tế: head + field + snap từ A, neck mới chưng cất
    từ encoder của v17 một nhóm, LD + KD class từ thầy một nhóm (không có dòng cảnh báo "head layout differs" → LD bật; loss `ld` / `kd_cls` không in
    trong log / `results.csv`, `report.LOSS_KEYS` chưa có cột này).
  - Theo class (run này / v19 F75 / A F74 o2m 0.7 / thầy F73): wall .826 / .498 | .879 / .521 | .872 / .534 | .840 / .523;
    junction .895 / .518 | .887 / .496 | .904 / .531 | .900 / .505; wall_300 .727 / .252 | .763 / .287 | .691 / .273 | .763 / .308;
    note .700 / .477 | .718 / .441 | .755 / .493; door .974 / .797; slide_door .632 / .299 (cao nhất từ trước); double_door .509 / .408.
    Lỗi: **wall → nền 133 (13%)** (F75 8%, A 79, thầy 118 = 11%), junction → nền 7%, wall_300 → junction 22%, slide_door → wall 52%.
  - So mốc: v19 F75 −.003 / +.002; A đã hiệu chỉnh o2m 0.5 (F76 0.751 / 0.481) +.001 / −.017; YOLO26x +.006 / −.035; v12 +.009 / −.008, nhanh hơn 19%.
  - Đối chiếu OPENOBB20 mục 3 (bản 1): mAP50-95 ≥ 0.475 không đạt; mAP50 ≥ 0.75 đạt; junction ≥ .52 không đạt (.518); wall ≥ .53 không đạt;
    tốc độ đạt; box trùng ≤ 3% đạt. Tiêu chí "< 0.468 → LD không tác dụng" **không áp dụng**: xuất phát là neck mới, không phải `best.pt` v19.
  - Đọc được: (1) neck chưng cất từ encoder đã train đạt mức v19 F75 (60 + 30 ep) chỉ sau 30 ep, đường cong còn lên (ep 23 → 30: .457 → .465).
    (2) Giả thuyết cho wall tụt: thầy một nhóm chỉ có một box mỗi điểm, ở điểm tranh chấp junction thắng → xác suất wall của thầy thấp →
    KD class kéo nhóm wall của học trò xuống, xoá phần lợi của nhóm (F74: wall → nền 118 → 79). Junction (bên thắng) không bị ảnh hưởng.
    Phép thử: chạy lại đúng thiết kế với thầy có nhóm; dự đoán wall → nền ≤ 9%, wall mAP50 ≥ .86.
  - Vòng sau (đúng F77): `weights=` `best.pt` của run này, `teacher=` file A (đang tên `openobb19x.pt`, nên đổi thành `openobb17x_groups.pt`); bản 2 =
    `openobb20x` cùng mọi thứ. Kỳ vọng OPENOBB20 mục 3 giữ nguyên, mốc bản 1 so với run này (0.752 / 0.464).

- **F79. Bản 2 của F77 = OpenOBB20-x, lệch thiết kế giống hệt F78 (user, 10-15): 0.764 / 0.482 (o2m 0.5), 12.6 ms, box trùng 2.6%.**
  - Cùng `weights=openobb19x.pt` (v17 + nhóm) và `teacher=openobb17x_v2.pt` (một nhóm), cùng công thức với F78. `[init]` 1084 / 12 / **346** new
    = 328 neck + 18 của bộ tinh chỉnh → **so với F78 khác đúng một yếu tố (bộ tinh chỉnh cạnh)**: cặp này hợp lệ cho phép thử bộ tinh chỉnh,
    dù cả hai lệch khỏi xuất phát đã định. 51.27M tham số (+0.59M). Best ep 27 (val trong train 0.763 / 0.488); ep 23–30 đi ngang 0.479–0.488.
  - Hiệu chỉnh: o2m 0.7 fitness .5130 (trùng 6.0%) so với o2m 0.5 .5102 (2.6%) → chọn o2m 0.5. o2o không NMS: 33.3% trùng.
  - Theo class (v20 / v19 F78, Δ): wall .835 / .526 (+.009 / **+.028**), junction .897 / .529 (+.002 / +.011), note .820 / .488 (+.120 / +.011),
    door .974 / .807 (0 / +.010), double_door .561 / .464 (+.052 / +.056), wall_300 .695 / .279 (−.032 / +.027), slide_door .568 / .280 (−.064 / −.019).
    P .733 (F78 .689), R .723 (.749). Lỗi: wall → nền 114 (11%; F78 13%, A 7.6%), junction → nền 7%, wall_300 → junction 16% (F78 22%).
  - Tổng so F78: **+0.012 / +0.018**, tốc độ +0.2 ms. Sáu class / bảy tăng 50-95; class tăng nhiều nhất là các class có đầu box dài (wall, double_door,
    wall_300), đúng chỗ F64 chỉ ra (sai số hai đầu). Độ lớn: +0.018 khoảng 0.75σ của thước (±0.024, F64), wall +.028 khoảng 1.6σ (±.017), một hạt giống.
  - So mốc: YOLO26x (0.746 / 0.499) **+0.018 / −0.017**; v12 (0.743 / 0.472) +0.021 / +0.010, nhanh hơn 18% (12.6 so với 15.3 ms);
    v17 + nhóm cùng hiệu chỉnh (F76 0.751 / 0.481) +0.013 / +0.001 ở 12.6 so với 16.1 ms. **Bản tốt nhất hiện tại ở cả ba mặt trừ mAP50-95 so với YOLO.**
  - Đối chiếu OPENOBB20 mục 3 (bản 2): ≥ bản 1 + 0.01 đạt (+0.018); mAP50 ≥ 0.75 đạt; junction ≥ .53 không đạt sát (.529); wall ≥ .55 không đạt (.526);
    ≤ 13.0 ms đạt; box trùng ≤ 3% đạt. Tiêu chí bác bỏ (< bản 1 + 0.005) không chạm → **bộ tinh chỉnh cạnh giữ lại**. Mục tiêu đột phá ≥ 0.499 chưa đạt.
  - Recall wall vẫn thấp (wall → nền 11% so với A 7.6%) dù box khít hơn: bộ tinh chỉnh không sửa recall.
  - **Bỏ giả thuyết "thầy một nhóm kéo wall xuống" (user phản biện, 10-15).** v19 đã qua ba kiểu thầy và tổng gần như không đổi:
    thầy v12 (vòng 1), không thầy (F75: 0.755 / 0.462), thầy v17 một nhóm (F78: 0.752 / 0.464).
    Chênh lệch wall → nền giữa F75 (8%) và F78 (13%) bị lẫn với độ dài train của neck: F75 neck đã học 60 + 30 ep, F78 neck mới học 30 ep.
    Đổi thầy là tinh chỉnh nhỏ, bằng chứng yếu, nên không làm hạt giống. Đường cong v20 đi ngang từ ep 23 → train thêm cùng điều kiện cũng ít lợi.

- **F80. Oracle "đầu box dùng chung với box kề bên": BÁC BỎ hướng A (OpenOBB21) trước khi code. (10-15, CPU, 53 ảnh val)**
  Script `research/analysis/shared_end_oracle.py`, log `shared_end_v17groups.log`, `shared_end_v12.log`. Model: v17 + nhóm (run A, file
  `openobb19x.pt`), head một-nhiều + NMS 0.5: 0.750 / 0.482 (khớp F76).
  - Cách đo, không dùng nhãn: sau NMS, mỗi đầu box được kéo về giao điểm gần nhất giữa trục box với một cạnh của box kề bên (conf ≥ 0.25, góc
    cắt ≥ 30°) nếu cách ≤ R px. Trần tham chiếu: thay hai đầu bằng đầu nhãn (`ends(gt)`, như F64). Thêm trần của một cổng hoàn hảo: chỉ kéo khi
    cạnh kề bên gần đầu nhãn hơn (dùng nhãn để chọn).
  - Δ mAP50-95 (v17 + nhóm): ends(gt) **+0.126**; kéo hẳn R 2 / 3 / 4 / 6 px: −0.014 / −0.029 / −0.041 / −0.054; kéo nửa đường R 3 / R 4: −0.009 / −0.013;
    **cổng hoàn hảo R 4: −0.002** (wall +.003, wall_300 +.019, junction −.036). v12 cho cùng kết luận (cổng hoàn hảo +0.001).
  - Chẩn đoán (detection khớp ProbIoU ≥ 0.5, có cạnh kề bên trong 4 px): sai số dọc trục tới đầu nhãn, trung vị đầu của chính box / cạnh kề bên:
    wall 1.83 / 2.93 px, junction 2.38 / 3.10, wall_300 1.75 / 1.97, door 3.58 / 4.54. Cạnh kề bên gần đầu nhãn hơn chỉ trong 30–34% trường hợp
    (wall, junction, wall_300), double_door 44%.
  - Kết luận: nhãn có đầu trùng cạnh hàng xóm (F64), nhưng **cạnh của detection hàng xóm không mang thêm thông tin về vị trí đầu**: nó sai nhiều hơn
    chính đầu của box, và kể cả biết trước lúc nào nên kéo cũng không tăng. Phần +0.126 của đầu box phải lấy từ đặc trưng ảnh của chính object, không
    từ quan hệ với box kề bên. Không làm OpenOBB21 theo hướng này. Hạn chế: junction gần vuông nên trục dài / ngắn dễ đảo, con số junction kém tin cậy.
  - Bước tiếp: đo `best.pt` v20 (`box_quality.py`, `box_error_oracle.py`) để biết sau bộ tinh chỉnh phần còn lại nằm ở đầu box hay độ dày.

- **F81. Sai số box của OpenOBB20 (F79) so với YOLO26x: phần còn lại gần hết là ĐẦU box; độ dày đã gần bằng YOLO. (10-15, CPU, 53 ảnh val)**
  `F:\Downloads\openobb20x.pt`, head một-nhiều + NMS 0.5: 0.767 / 0.485 trên máy (Colab 0.764 / 0.482). Log `box_error_v20.log`, `box_quality_v20.log`.
  - Trần từng thành phần (Δ mAP50-95; v12 F64 trong ngoặc): độ dày +0.032 (+0.036), lệch ngang +0.052 (+0.049), **hai đầu +0.132** (+0.116),
    góc +0.004, box hoàn hảo +0.348. Theo class, hai đầu: wall +.100, wall_300 +.314, junction +.125, note +.186, slide_door +.111.
  - Độ khít (trung vị, px ở 1280; v20 / YOLO26x): wall đầu box 1.92 / 1.30, |độ dày| 0.72 / 0.54, **lệch độ dày +0.18 / −0.04** (v12 +0.66),
    ngang 0.36 / 0.28, góc 0.24° / 0.01°, R@.75 .42 / .56; wall_300 đầu 1.88 / 1.05, R@.75 .11 / .24; junction đầu 0.85 / 0.63, ngang 0.53 / 0.38,
    góc 0.91° / 0.06°, R@.75 .32 / .47. v20 hơn YOLO ở slide_door (R@.5 .61 / .16) và double_door (.80 / .40).
  - Kết luận: bộ tinh chỉnh đã sửa lệch độ dày (+0.66 → +0.18 px) → hướng B (bin độ dày mịn) chỉ còn trần khoảng +0.03, **bỏ**. Khoảng hụt với YOLO
    nằm ở vị trí hai đầu dọc trục (sai gấp 1.5–1.8 lần YOLO), sau đó là lệch ngang. Góc lệch lớn ở junction nhưng trần gần 0.
  - User (10-15): thử thật thấy v20 tệ hơn v19, hướng box và vị trí box lệch nhiều. Đo `research/analysis/refiner_shift.py` (log
    `refiner_shift_v20.log`, 1500 ứng viên / ảnh): bộ tinh chỉnh xoay rất ít (trung vị 0.15–0.23°, p99 0.8–3.2°, val và 3 bản vẽ ngoài val)
    → hướng sai không do bộ tinh chỉnh; nó dịch ngang nhiều hơn (p90 5–9 px, p99 17–37 px; ngoài val lớn hơn val). Chờ ảnh / lệnh test của user.

- **F82. v20 trên bản vẽ thật của user (`9F作図資料（新規作図） - trang 1 @2x.png`): tường dài bị nghiêng và lệch ngang; bộ tinh chỉnh xoay
  box dài nhiều hơn hẳn so với val. (10-15, CPU)** Script `compare_two.py`, `refiner_shift.py`; log `refiner_shift_v20_bylen.log`,
  `box_quality_v17groups.log`; ảnh `research/analysis/compare_two_out/`.
  - User so `openobb20x.pt` với `openobb19x.pt`, nhưng file sau là **v17 + nhóm** (F78), không phải v19.
  - Trên ảnh 9F (2381 × 1683): v20 233 box, v17 + nhóm 212. Cặp khớp cùng class: góc lệch trung vị wall 0.25°, door 1.39° (p90 22°).
    Nhìn ảnh: ba tường dọc dài, box v20 nghiêng khoảng 3–5° và một box lệch hẳn sang bên; v17 + nhóm cũng nghiêng ở hai tường, nhưng tường giữa đúng.
  - Bộ tinh chỉnh theo độ dài box (góc xoay p90, val / 9F): 40–100 px 0.64° / 1.11°, **100–200 px 0.77° / 3.95°**, ≥ 200 px 0.63° / 1.35°;
    ứng viên bị xoay > 3°: 1.0% / 3.6%. Bộ tinh chỉnh xoay tới ±0.2 rad (11.5°) không phụ thuộc độ dài: với box dài, vài độ làm đầu box văng nhiều px.
    Học trên 250 ảnh, nó khái quát kém sang bản vẽ khác kiểu (chữ viết tay, ghi chú).
  - Trên val, v20 vẫn khít hơn v17 + nhóm (box_quality, NMS 0.5): góc wall 0.24° / 0.40°, junction 0.91° / 3.08°, door 0.64° / 1.37°;
    đầu box wall 1.92 / 2.19 px; R@.75 wall .42 / .36. → Thước val không bắt được lỗi này.
  - **ĐÍNH CHÍNH (cùng ngày):** tắt bộ tinh chỉnh của v20 (lớp cuối = 0, tức hành vi v19) trên ảnh 9F → box còn nghiêng hơn (hình chữ X trên
    tường dọc dài), 248 box so với 233; tường chéo khoảng 25° vẫn sai, junction vẫn nằm thẳng trục thay vì xoay theo tường, junction lạc vẫn còn.
    → Nguyên nhân KHÔNG phải bộ tinh chỉnh (nó còn sửa bớt) mà là **neck nhanh chưng cất của họ v18 / v19** so với encoder lai + LSK của v17
    (ảnh `compare_two_out/openobb20x_norefiner_*.png`). Giả thuyết cơ chế: neck RepConv tầm nhìn cục bộ; góc của object dài / chéo cần ngữ cảnh
    xa mà encoder (attention trên P5) + LSK (kernel lớn) cung cấp. Bỏ hướng "giới hạn góc của bộ tinh chỉnh".
  - Thước val (mAP, trung vị sai số) không bắt được lỗi này → `research/analysis/infer_box_eval.py`: đuôi sai số góc / lệch đầu box theo độ dài,
    đường predict thật, conf 0.25; cộng thước không nhãn "lệch trục" cho tường dài (nhãn val: 100% tường ≥ 100 px nằm trong 0.5° của trục).
  - User nhắc lại log v17 + nhóm (F74): 0.757 / 0.487 (o2m 0.7, trùng 7.4%) hoặc 0.751 / 0.481 (o2m 0.5, trùng 2.2%), 16.1 ms. Cùng NMS 0.5 với v20
    (0.764 / 0.482): mAP50 −0.013, mAP50-95 ngang (trong nhiễu ±0.024), nhưng box đúng hướng trên bản vẽ thật. → **Bản tốt nhất thực tế: v17 + nhóm.**
    Họ v18 / v19 đổi 22% tốc độ lấy mất hướng box của object dài / chéo; thước val không thấy.

- **F83. Đánh giá lại v17 + nhóm / v19 / v20 / YOLO26x với thước hướng box: LỆCH GÓC là điểm yếu hệ thống của cả họ OpenOBB, YOLO gần như không có.
  (10-15, CPU)** Script `research/analysis/eval_versions.py`, `tilt_real.py`; log `eval_versions.log`, `tilt_real.log`.
  v19 = `openobb20x.pt` với lớp cuối bộ tinh chỉnh = 0 (cùng trọng số, không phải checkpoint F78). Ứng viên thô, NMS 0.5, box ở conf 0.25.
  | | mAP50 / 50-95 | trùng | G4 ms | nhãn tìm được: góc > 1° / > 2° | đầu box lệch > 4 px | tường ≥ 100 px: góc > 1° / lệch > 4 px | box tường ≥ 100 px nghiêng > 1° / > 3° (không cần nhãn) |
  |---|---|---|---|---|---|---|---|
  | v17 + nhóm | .753 / .479 | 2.1% | 16.1 | 33.7% / 10.9% | 2.5% | 20.7% / 9.2% | 32.6% / 17.0% |
  | v19 | .774 / .480 | 3.0% | 12.4 | 10.8% / 2.8% | 2.0% | 20.5% / 10.3% | 43.2% / 31.9% |
  | v20 | .769 / .480 | 2.5% | 12.6 | 10.0% / 2.5% | 1.9% | 14.1% / 4.7% | 26.3% / 16.2% |
  | YOLO26x | .746 / .499 (F59) | 1.5% | — | 1.3% / 0.9% | 0.8% | **0.0% / 0.0%** | **0.0% / 0.0%** |
  - Nhãn val: 100% tường / wall_300 ≥ 100 px nằm trong 0.5° của trục → box tường dài nghiêng > 1° là sai, đo được không cần nhãn.
  - Theo độ dài (wall + wall_300, góc > 1°): v17 + nhóm 33% / 12% / 20% / 23% (< 40, 40–100, 100–200, ≥ 200 px); v19 11% / 3% / 20% / 23%;
    v20 10% / 3% / 12% / 22%; YOLO 3% / 0.2% / 0% / 0%. Lỗi của mình tăng theo độ dài; YOLO không.
  - Bản vẽ thật (box tường ≥ 100 px nghiêng > 1°): 9F v17 + nhóm 20% / v19 67% / v20 47% / YOLO 0%; 三越 43% / 67% / 44% / 0%; きらぼし 20% / 40% / 25% / 0%.
    (250805 có tường chéo thật: 100% ở cả bốn model, bỏ.) → Khớp nhận xét của user: v19 / v20 nghiêng nặng hơn v17 trên bản vẽ thật.
  - Theo class (góc > 2°): v17 + nhóm kém nhất ở note 26%, double_door 24%, door 17%, junction 12% (v19 / v20 ≤ 2%) → encoder v17 không phải lời giải;
    YOLO ≤ 1.6% mọi class.
  - Giả thuyết cơ chế (từ code): loss góc của head dense là 1 − cos 2Δθ ≈ 2Δθ²: ở 1° chỉ 6e-4, gần như không có gradient để kéo góc về đúng dưới vài độ;
    sai số góc nhân với độ dài thành lệch đầu box, nên box dài hỏng nhất. Góc của box dài lại suy từ một điểm neo có tầm nhìn cục bộ.
  - mAP và trung vị sai số đều không thấy lỗi này (v19 có mAP50 cao nhất nhưng nghiêng nhiều nhất). → Thước mới bắt buộc cho mọi version.

- **F84. Nguyên nhân lệch góc: tường DỌC trong ảnh (góc ở mép ±π/2 của le90), loss góc không có gradient gần tối ưu. Sửa chung cho mọi
  version dense + OpenOBB21. (10-15, CPU)** Script `research/analysis/tilt_probe.py` (log `tilt_probe_v20v2*.log`, `tilt_probe_v17groups.log`).
  - Run user (10-15): `openobb20x` fine-tune từ `openobb20x.pt`, tắt xoay góc bất kỳ, lr 0.0005, `nbs 64`, 30 ep → **0.756 / 0.487**
    (o2m 0.5), trùng 2.3%, 12.6 ms (file `openobb20x_v2.pt`). Thước F83: box tường dài nghiêng > 1° **28.4%** (val), 9F 31% / 三越 44% (YOLO 0%).
  - Checkpoint cũ (args lưu trong file): v12, v17, v17 + nhóm train với `rotate_p=0.5` (xoay góc bất kỳ, notebook truyền vào, log không in);
    v20 F79 và v20_v2 không xoay nhưng vẫn lệch → tăng cường xoay không phải nguyên nhân chính.
  - Dò bên trong head (20 ảnh val, ứng viên tự tin dài ≥ 100 px): **tường dọc 88.7% lệch > 1°, tường ngang 0.3%**; theo tầng: gần như toàn
    stride 8; theo vị trí neo: |u| < 0.5 lệch 25–27%, |u| 0.5–0.75 lệch 0.7%. Xoay ảnh vào 90°: tường vừa thành dọc lệch 87.5%, tường
    vừa thành ngang 0.9% → lỗi theo HƯỚNG TRONG ẢNH, không theo nội dung. Góc tường dọc vượt ±90° (trung vị +3.9°, 76% ở ngoài); v17 + nhóm:
    tường dọc 67.6% lệch, tản hai phía. Tường dọc cũng có ít ứng viên tự tin hơn (ảnh 0: 21 tường ngang → 73 ứng viên khi nằm ngang, 10 khi dọc).
  - Đã loại trừ: nhãn (polys_to_obb: tường dọc θ = −π/2 đúng), `fuse()` (giống hệt), bám ranh giới (không đổi góc), bộ tinh chỉnh (giảm lệch:
    loss góc của nó là L1 × 5). Còn lại: loss góc head 1 − cos 2Δθ (gradient 4Δθ → 0.07 ở 1°) + lớp ra góc chạy bf16 (bước 0.45° ở 1.57 rad).
    Phép thử suy luận bf16 trên CPU (sau đó): tường dọc vẫn 88.4% lệch > 1°, ngang 0.3% → độ chính xác bf16 KHÔNG phải nguyên nhân (log `tilt_probe_v20v2_bf16.log`).
  - **Sửa chung (0.20.0, mọi model dense v3–v21):** loss góc = L1 trên (cos 2θ, sin 2θ) (`--angle-loss vec`, mặc định; `cos` = cũ); lớp ra
    hồi quy chạy fp32 (`openobb3.reg_out`). **OpenOBB21** = v20 + nhánh sửa góc trong không gian góc kép (zero-init). Thước `Box geometry`
    thêm vào `openobb val` và cuối log train. Kỳ vọng + lệnh: [docs/OPENOBB21.md](../docs/OPENOBB21.md). Checkpoint cũ không tự hết lệch:
    phải fine-tune lại bằng code mới.

- **F85. v17 (một nhóm) fine-tune với bản sửa góc 0.20.0 (user, 10-15): 0.761 / 0.486, trùng 2.2%, 16.0 ms; vật dài lệch > 1°: 24.1% → chưa hết lệch.**
  - Điều kiện: `openobb17x`, khởi tạo `openobb17x.pt` (F68, một nhóm), 30 ep, lr 0.001, **`nbs 64` (112 lần cập nhật)**, **vẫn `rotate any angle 0.5`**
    (notebook truyền `rotate_p=0.5`; nay dòng `augment` in ra), `angle loss vec`, G4. Best ep 17; ep 13–30 đi ngang 0.484–0.488.
  - Theo class: wall .856 / .530, junction .899 / .514, wall_300 .699 / .256, note .840 / .520, door .974 / .806, slide_door .456 / .263, double_door .604 / .513.
  - `Box geometry`: góc > 1° 23.9%, > 2° 6.1%; vật dài (≥ 10 × độ dày) góc > 1° 14.0%, đầu lệch 6.4%; **box dài lệch trục > 1° 24.1%** của 610 (nhãn 0.2%).
    Cùng thước trên cache CPU (NMS 0.5): v17 + nhóm 24.8%, v19 25.0%, v20 F79 15.8%, v20 không xoay 15.3%.
  - Không phải phép thử sạch cho bản sửa: còn xoay góc bất kỳ 50% (tạo tường nghiêng giả trong data) và chỉ 112 lần cập nhật. Run v20 / v21 phải
    bỏ `rotate_p` và dùng `nbs=0` (450 lần cập nhật).

- **F86. NGUYÊN NHÂN GỐC của lệch góc: mẫu dương của tường DỌC không bao giờ được dạy góc (trọng số TAL ~0). Sửa: trọng số loss góc đều trên
  mẫu dương (0.20.1, mặc định mọi model dense). (10-15, CPU)** Script `research/analysis/grad_probe.py` (một forward có nhãn trên ảnh train,
  BN đóng băng, không cập nhật; ghi các mẫu dương mà loss dùng).
  - `openobb20x_v2.pt` trên ảnh train đầu tiên: 50 mẫu dương thuộc tường DÀI DỌC → góc dự đoán |θ| trung vị **5.6°** (10% / 90%: 0.2° / 177.6°,
    tức gần như NẰM NGANG), lệch so với nhãn trung vị −77.8°, 96% sai > 1°; 94 mẫu dương của tường dài ngang: lệch 0.00°, 12.8% > 1°.
  - Cơ chế: lớp ra góc khởi tạo 0 = mọi box nằm ngang. Gán mẫu TAL đặt trọng số = điểm × ProbIoU^6 (chuẩn hoá theo object): box sai 90° có
    ProbIoU ~0 → trọng số loss góc / box ~0 → tường dọc không được dạy, vòng lặp tự duy trì qua mọi fine-tune. Box tường dọc lúc suy luận chỉ đến từ
    vài điểm lẻ → ít ứng viên tự tin (F84), lệch đều +2–4°, chồng chéo hình chữ X. YOLO khởi tạo góc 45° nên dọc và ngang sai đều nhau từ đầu.
  - Vì vậy bản sửa 0.20.0 (dạng loss vectơ) không đủ: F85 (v17 mới) vẫn 24.1%. Bộ tinh chỉnh (v20) giảm lệch vì nó train trên ứng viên + nhãn rung
    với trọng số đều (L1 × 5).
  - Sửa (0.20.1): `--angle-balance uniform` (mặc định): mỗi mẫu dương nhận trọng số trung bình của các mẫu dương (tổng trọng số không đổi);
    `tal` = cũ. Test `test_uniform_angle_weights_reach_positives_with_no_alignment`.
  - Kỳ vọng: sau fine-tune, mẫu dương tường dọc lệch trung vị < 1° (đo lại bằng grad_probe), box tường dài lệch trục > 1° ≤ 5% (dòng `Box geometry`).

- **F87. v17 + nhóm fine-tune với 0.20.0 (user, 10-15): 0.752 / 0.482, trùng 2.2%, 16.1 ms; box dài lệch trục > 1° 20.3% (xuất phát 24.8%).**
  Khởi tạo `openobb19x.pt` (v17 + nhóm), vẫn `rotate any angle 0.5`, `nbs 64` (112 cập nhật), `angle loss vec` nhưng CHƯA có trọng số đều (0.20.1).
  wall .874 / .539, junction .909 / .523, wall_300 .711 / .272, note .691 / .451. `Box geometry`: góc > 1° 19.4%, vật dài góc > 1° 13.2%.
  Chỉ bớt 4.5 điểm lệch → khớp F86 (dạng loss một mình không đủ). Chưa phải phép thử của 0.20.1.

- **F88. Hai run 10-15 (user), CẢ HAI vẫn `rotate any angle 0.5` + `nbs 64` (112 cập nhật):**
  | | phiên bản | khởi tạo | mAP50 / 50-95 | trùng | ms | góc > 1° (tất cả / vật dài) | đầu lệch (vật dài) | box dài lệch trục > 1° |
  |---|---|---|---|---|---|---|---|---|
  | v17 + nhóm | 0.20.1 (trọng số góc đều) | `openobb19x.pt` | .748 / .482 | 2.3% | 16.2 | 21.2% / 18.7% | 7.2% | 28.9% |
  | v17 + nhóm (F87) | 0.20.0 | `openobb19x.pt` | .752 / .482 | 2.2% | 16.1 | 19.4% / 13.2% | 3.3% | 20.3% |
  | **v21** | 0.20.0 | v21 vòng trước (`train3/best.pt`, 0 tensor mới) | .751 / .474 | 2.1% | 12.7 | 26.8% / **9.8%** | **0.6%** | **13.9%** |
  - v21: wall **.892 / .560** (cao nhất của OpenOBB, recall .825; YOLO .624 50-95), note .825 / .517; tụt slide_door .420 / .179, double_door .564 / .457
    (lớp hiếm, nhiễu ±0.08–0.09). Hiệu chỉnh: o2o không NMS 36.8% trùng → o2m 0.5.
  - Trọng số góc đều (0.20.1) KHÔNG giảm lệch ở v17 + nhóm trong 112 cập nhật (28.9% so với 20.3%); loss góc in ra tăng (0.27 → 0.15 so với 0.18 → 0.08)
    đúng như dự kiến (các mẫu dương sai giờ có trọng số). Giả thuyết F86 chưa được xác nhận bằng train; chưa có run sạch.
  - Yếu tố gây nhiễu còn lại: xoay góc bất kỳ 50% (tường nghiêng giả, YOLO `degrees=0`) và quá ít lần cập nhật. Phép thử sạch: bỏ `rotate_p`, `nbs=0`.

- **F89. YOLO26x đo cùng thước, cùng G4 (user, 10-15; cell so sánh dùng `openobb.eval.ultra.confusion / geometry`):**
  `train 16_9/best.pt` (150 ep, `degrees 0`, `nbs 64`): **0.743 / 0.499**, Speed 0.5 ms pre + **11.5 ms inference** + 17.3 ms postprocess (batch 16,
  1280, fp16), trùng **1.6%**, Box geometry: góc > 1° **1.3%**, > 2° 0.9%, đầu lệch 0.1%; vật dài góc > 1° **0.0%**, đầu lệch 0.0%;
  **box dài lệch trục > 1° 0.3%** của 577 (nhãn 0.2%).
  - So v21 (F88, cùng thước): .751 / .474, trùng 2.1%, góc > 1° 26.8%, vật dài 9.8%, box dài lệch 13.9%; 12.7 ms. Khoảng cách còn lại: mAP50-95 −0.025,
    hướng box (13.9% so với 0.3%), trùng +0.5 điểm.
  - Tốc độ: `Speed` của OpenOBB v10–v21 đang tính NMS vào "inference" (module của v12+ không có trong `speed._NMS_MODULES`) → sửa 0.20.2 để
    tách như Ultralytics. Mạng thuần: YOLO 11.5 ms; v21 ≤ 12.7 ms (gồm cả NMS + tinh chỉnh). Postprocess 17.3 ms của YOLO là phần val của Ultralytics.

- **F90. v21 từ `openobb20x_v2.pt` với 0.20.2 (user, 10-15): 0.769 / 0.492 (o2m 0.5), trùng 1.8%, 9.7 + 3.0 ms; box dài lệch trục 14.4%.**
  Vẫn `rotate any angle 0.5`, `nbs 64` (112 cập nhật); `[init]` 12 tensor mới (nhánh góc). Best ep 28. wall .863 / .545, junction .900 / .538,
  wall_300 .730 / .310, note .775 / .487, door .975 / .810, slide_door .601 / .320, double_door .537 / .435.
  - So xuất phát (v20_v2, cache CPU cùng thước): mAP +0.013 / +0.005, box dài lệch 15.3% → 14.4% (không đổi). Bản sửa góc + nhánh góc KHÔNG giảm lệch
    khi còn xoay góc bất kỳ 50%.
  - **Speed sau sửa 0.20.2 (NMS tách ra):** v21 9.7 ms mạng + 3.0 ms NMS = 12.7 ms; YOLO26x cùng G4 khi máy đã nóng (lần đo thứ hai) 6.6 + 0.7 = 7.3 ms
    (lần đầu 11.5 + 17.3 là do khởi tạo, bỏ). → v21 chậm 1.7 lần; NMS (fast_nms, vòng lặp từng ảnh, ProbIoU trên 4000 ứng viên) chiếm 3.0 ms so với
    0.7 ms của YOLO.
  - So YOLO26x cùng thước: mAP50 +0.026, mAP50-95 −0.007 (trong nhiễu ±0.024), trùng +0.2 điểm; còn thua: tốc độ (1.7×) và hướng box (14.4% / 0.3%).

- **F91. OpenOBB22 (10-15, chưa train): v21 + head hình học kiểu YOLO26: góc (sigmoid − ¼)·π ∈ (−45°, 135°) khởi tạo 45°, hồi quy thẳng hai đầu +
  độ dày (bỏ DFL).** [docs/OPENOBB22.md](../docs/OPENOBB22.md).
  - Lý do: v21 đã ngang / hơn YOLO26x ở mAP50, mAP50-95 (trong nhiễu), box trùng, lớp hiếm (F89, F90); phần thua là hình học box của wall / junction /
    wall_300 (đầu box F81, hướng F86). Góc khởi tạo 45° làm tường ngang và dọc sai đều lúc đầu (khắc phục F86 từ gốc); hồi quy thẳng bỏ bin 8 px.
  - Sửa kèm: chuyển checkpoint DFL → `box=direct` trước đây sai với nhóm class (chép hàng theo vị trí trộn layout các nhóm); `OpenOBB22.after_weights`
    dựng lại theo từng nhóm (góc 45°, đầu / độ dày từ 0, lệch ngang giữ). 51.06M tham số.
  - Kỳ vọng ghi trước (150 ep từ best.pt v21, không xoay): box dài lệch ≤ 2%, mAP50-95 ≥ 0.51 (wall ≥ .60, junction ≥ .57, wall_300 ≥ .40),
    mAP50 ≥ 0.765, trùng ≤ 2.0%, ≤ 12.7 ms. Bác bỏ: lệch ≥ 8% (góc) hoặc wall 50-95 < .55 (hồi quy thẳng).

- **F92. Tốc độ: NMS theo từng lớp (0.21.1) + v22 KHÔNG nặng hơn YOLO26x về FLOPs; khoảng cách nằm ngoài conv. (10-15, CPU)**
  - `fast_nms` cũ tính ProbIoU cho mọi cặp (tới 4000², rồi mới lọc cùng lớp): 3.0 ms / ảnh trên G4 (F90) so với 0.7 ms của YOLO26 (không NMS, head
    một-một). Bản mới tính theo khối từng lớp, kết quả y hệt (test `tests/test_nms_classwise.py`), CPU 4000 ứng viên: 1557 → 210 ms / ảnh (7.4×).
  - `research/analysis/roofline.py openobb22x yolo26x-obb --hw 960x1280`: v22-x **523 GFLOPs**, 4.23 GB, roofline 4.13 ms; YOLO26x **605 GFLOPs**,
    3.80 GB, 4.34 ms. Phần lớn của v22: neck tự viết 140 GFLOPs (24% thời gian roofline), thân stage 2 / 3 (18% / 21%), trường ranh giới 16 GFLOPs
    (6%, tầng stride 2), head 43 GFLOPs. Đo thật trên G4: 9.7 ms so với 6.6 ms → 3 ms chênh nằm NGOÀI các lớp conv (chọn ứng viên G × A, bộ tinh
    chỉnh + bám mép trên 1500 ứng viên, vòng lặp Python theo nhóm / tầng, nhiều kernel nhỏ).
  - Công cụ `openobb profile model=best.pt` (0.21.1): thời gian GPU từng khối (thân + neck, head, trường ranh giới, decode + nhóm, top-k + tinh chỉnh
    + bám mép, NMS). Cần chạy trên G4 trước khi thiết kế bản nhanh (OpenOBB23).

- **F93. OpenOBB22-x (user, 10-15), LỆCH điều kiện ghi trước: 0.760 / 0.479 (o2m 0.5), trùng 2.4%, 9.5 + 3.0 ms; wall SỤP (.747 / .466, wall → nền 24%).**
  0.21.0, 30 ep, lr 0.0005, `nbs 64` (112 cập nhật), vẫn `rotate any angle 0.5` (kỳ vọng ghi cho 150 ep, không xoay). `[init] best.pt: 1442 full,
  12 partial, 0 new` và KHÔNG có dòng "box outputs rebuilt" → checkpoint nguồn có `arch v22` trong args (mAP ep 1 = 0.471: box không bắt đầu lại);
  12 tensor partial (lớp ra box) chưa giải thích được, cần biết file nguồn.
  - Theo class (v22 / v21 F90): wall .747 / .466 | .863 / .545; junction .905 / .533 | .900 / .538; wall_300 .746 / .288 | .730 / .310;
    note .711 / .470 | .775 / .487; door .975 / .817; slide_door .662 / .314; double_door .574 / .465. Vật dài tìm được 358 (v21 513), box dài lệch 16.2%.
  - Đối chiếu OPENOBB22 mục 4: lệch ≤ 2% không đạt (16.2%); mAP50-95 ≥ 0.51 không đạt; wall ≥ .60 không đạt và **chạm ngưỡng bác bỏ hồi quy thẳng
    (wall < .55)**; mAP50 ≥ 0.765 không đạt (0.760); trùng ≤ 2.0% không đạt; tốc độ đạt. Điều kiện chưa đúng (30 ep, xoay, 112 cập nhật) nên chưa kết
    luận chính thức, nhưng bằng chứng đang chống lại hồi quy thẳng cho tường dài: softplus × stride phải học khoảng cách 12–40 stride từ 0.69.

- **F94. `openobb profile` trên G4 (user, 10-15), v21-x, batch 16, 960 × 1280, bf16, ms / ảnh:** thân + neck **7.67**, trường ranh giới 0.57,
  head 1.00, decode + nhóm 0.13, top-k + tinh chỉnh + bám mép 0.24, **NMS 3.42**, tổng 13.03 (YOLO26x cả mạng 6.6 + 0.7).
  - Thân + neck chạy ở khoảng 44% hiệu suất roofline (3.37 ms ước lượng) so với khoảng 66% của YOLO → chi phí là bộ nhớ / kernel nhỏ (khối HG nhiều
    concat, conv nhẹ, SPD stride 4, neck 384 kênh), không phải FLOPs. Cần `openobb profile ... detail=True` (0.21.2) để tách từng stage / khối neck.
  - NMS theo từng lớp chậm hơn trên GPU (3.42 so với 3.0 ms, F90): vòng lặp lớp + `unique()` thêm đồng bộ và kernel nhỏ. 0.21.2: GPU dùng lại ma
    trận đầy đủ, CPU giữ theo lớp. Cách dứt điểm: bỏ NMS (head một-một sạch như YOLO26; hiện o2o không NMS còn 31% trùng).

- **F95. Box trùng của head một-một = điểm neo kề nhau → lọc đỉnh 3×3 thay NMS; OpenOBB23 = v21 + bám ranh giới có xoay + lọc đỉnh. (10-15, CPU)**
  Script `research/analysis/o2o_dup_probe.py` (log `o2o_dup_v20v2.log`), `openobb20x_v2.pt`, 10 ảnh val, ứng viên tự tin (≥ 0.25) trước tinh chỉnh / NMS:
  - head một-một: 1274 ứng viên, 366 trùng (28.7%); **94.5% cặp trùng có điểm neo cách ≤ 1.5 stride**, 91.8% cùng tầng, 100% cùng nhóm class,
    88.5% điểm số ≥ 0.5 × điểm cao nhất; 75% trên vật ngắn. head một-nhiều: 53.8% trùng, 68.8% kề nhau, 31% xa hơn.
  - → `openobb17.peak_mask`: giữ ứng viên có điểm cao nhất trong ô 3×3 của tầng + nhóm (một max-pool / tầng, không so cặp). Lựa chọn `peak` trong
    hiệu chỉnh cuối train (`o2o peak`, `o2o peak+0.5`, `o2m peak+0.5`) và trong `nms_iou=`.
  - Đo trên val (53 ảnh, `openobb20x_v2.pt`, `research/analysis/postproc_compare.py`, log `postproc_compare*_v20v2.log`):
    | hậu xử lý | mAP50 / 50-95 | trùng | box dài lệch | hậu xử lý CPU |
    |---|---|---|---|---|
    | o2m + NMS 0.5 (hiện tại) | .759 / .483 | 2.2% | 15.4% | 184 ms |
    | o2o + lọc đỉnh | .722 / .461 | 5.3% | 5.8% | 8 ms |
    | o2o không gì | .693 / .452 | 32.3% | 8.7% | 0 |
    | o2m + lọc đỉnh | .687 / .447 | 24.5% | 12.9% | 7 ms |
    | o2m + lọc đỉnh + NMS 0.5 | .725 / .464 | 1.5% | 12.7% | 120 ms |
    | o2o + lọc đỉnh + NMS 0.5 | .724 / .459 | 1.2% | 5.5% | 121 ms |
    → **lọc đỉnh xoá cả object thật kề nhau (−0.035 mAP50): bác bỏ làm mặc định** (giữ làm lựa chọn, hiệu chỉnh tự loại). Phát hiện phụ:
    **head một-một cho box ít nghiêng hơn hẳn** (5.5% so với 15.4%); trong các lần train, o2o + NMS 0.5 có fitness ngang o2m + NMS 0.5
    (chênh < 0.002) → `choose_postprocess` (0.22.0): khi độ chính xác và trùng ngang nhau, chọn cấu hình nghiêng ít hơn ≥ 2 điểm (`TILT_GAIN`).
  - Oracle bám xoay (chưa train, hệ số 1, trường của v20): box dài lệch 15.4% → 13.5%, đầu lệch 3.2% → 1.8%, mAP không đổi (`postproc_rot_v20v2.log`).
  - **OpenOBB23** ([docs/OPENOBB23.md](../docs/OPENOBB23.md)): nền v21 (bỏ hướng v22, F93) + `BoundarySnapRot`: trên 2 cạnh dài của box dài
    (L ≥ 4T), 7 điểm dọc cạnh, lệch ngang của mép (soft-argmax trên trường ranh giới) → độ dốc bình phương tối thiểu = góc lệch; hệ số khởi tạo 0
    (bắt đầu đúng bằng v21). + lọc đỉnh. Kỳ vọng ghi trước (100 ep từ best.pt v21, không xoay): box dài lệch ≤ 3%, mAP50 ≥ 0.765, mAP50-95 ≥ 0.495,
    trùng ≤ 2.5%, tổng ≤ 10.2 ms. Bác bỏ bám xoay nếu lệch ≥ 10% hoặc hệ số học được < 0.2; bác bỏ lọc đỉnh nếu trùng > 4%.

- **F96. OpenOBB23-x (user, 10-16): 0.800 / 0.512 (o2m 0.5), trùng 1.9%, 9.8 + 3.0 ms; box dài lệch trục 10.3%. Bản tốt nhất từ trước tới nay.**
  0.22.0, `weights=openobb21x.pt` (1 tensor mới = `snap.rot_gain`), 50 ep, `nbs 64` (187 cập nhật), lr 0.001, `rotate any angle 0.5`, G4. Best ep 46,
  còn nhích. Hiệu chỉnh: o2m 0.7 .5410 (trùng 4.9%) / **o2m 0.5 .5403 (1.9%, chọn)** / o2o 0.5 .5280; mọi cấu hình lệch 9.5–10.9% (luật độ nghiêng không đổi
  lựa chọn); lọc đỉnh: o2m peak+0.5 .5264 (trùng 1.2%), o2o peak .5186.
  - Theo class (v23 | v21 F90 | YOLO26x F89), mAP50 / 50-95: wall **.911 / .576** | .863 / .545 | .919 / .624; junction .912 / .545 | .900 / .538 |
    .910 / .584; wall_300 .751 / .331 | .730 / .310 | .824 / .441; note .770 / .494 | .775 / .487 | .764 / .517; door .978 / .811 | .975 / .810 |
    .965 / .811; slide_door .646 / .317 | .601 / .320 | .345 / .123; double_door .631 / .505 | .537 / .435 | .494 / .392.
    Lỗi: wall → nền 6% (v21 9%), junction → nền 6%, wall_300 → junction 11% (v21 15%).
  - `Box geometry`: góc > 1° 18.7% (v21 27.1%), > 2° 4.6%; vật dài góc > 1° **6.6%** (v21 11.3%), đầu lệch 0.6%; box dài lệch trục **10.3%** (v21 14.4%).
  - So YOLO26x: mAP50 **+0.057**, mAP50-95 **+0.013**, trùng +0.3 điểm, chậm 1.75× (12.8 / 7.3 ms), lệch 10.3% / 0.3%. Vượt YOLO ở 2 chỉ số mAP lần đầu
    với khoảng cách mAP50 ngoài nhiễu (±0.025); mAP50-95 trong nhiễu. Còn thua: wall / junction / wall_300 50-95, hướng box, tốc độ.
  - Đối chiếu OPENOBB23 mục 4: lệch ≤ 3% không đạt (10.3%), **chạm ngưỡng bác bỏ bám xoay (≥ 10%)**; mAP50 ≥ 0.765 đạt; mAP50-95 ≥ 0.495 đạt; trùng ≤ 2.5%
    đạt; tốc độ ≤ 13.0 ms đạt. Gây nhiễu: 50 ep / 187 cập nhật so với 30 ep / 112 của v21, và còn xoay ảnh → phần tăng mAP chưa tách được khỏi train dài
    hơn. Cần giá trị `snap.rot_gain` học được (tiêu chí thứ hai: < 0.2 → bác bỏ) và một v21 50 ep cùng điều kiện làm đối chứng.

- **F97. Tốc độ suy luận không đổi trọng số (user G4 + CPU, 10-16): channels_last −1.4 ms; NMS chỉ chấm cặp gần (cùng kết quả, 4.8× trên CPU).**
  `openobb profile` v23-x (best.pt F96), batch 16, 960 × 1280, bf16, ms / ảnh:
  | | thân + neck | head | NMS | tổng |
  |---|---|---|---|---|
  | NCHW (cũ) | 7.23 | 0.99 | 2.91 | 12.10 |
  | **channels_last** | **5.65** | 1.16 | 2.90 | **10.67** |
  - Chi tiết NCHW: stem 0.67, stage 1 0.74, stage 2 1.05, **stage 3 1.68**, stage 4 0.39; neck `td3` **0.82**, các khối khác ≤ 0.16; khoảng 1.3 ms nằm giữa các
    module (nội suy, nối, chép) → loại chi phí bộ nhớ channels_last / compile nhắm tới. `compile=True` lỗi ở 0.22.1 (đã sửa: không compile head).
  - 0.22.2: `predict.fast_layout` (channels_last cho model dense trên GPU, `OPENOBB_CHANNELS_LAST=0` để tắt) ở `load_model` và ở bước val cuối train
    (sau khi gộp BN). `openobb3.nms_keep_near`: chỉ chấm ProbIoU cho cặp cùng lớp có tâm cách ≤ 1.5 × tổng nửa đường chéo; giống hệt ma trận đầy đủ
    (test), CPU 4000 box 657 → 137 ms; dùng cho cả CPU và GPU. Chờ số G4.

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
