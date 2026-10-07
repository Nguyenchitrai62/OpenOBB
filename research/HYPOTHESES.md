# VRDet: giả thuyết, tiêu chí và kết quả

Quy tắc:
- Viết giả thuyết, **một** biến thay đổi và tiêu chí pass/fail **trước** khi launch.
- Số đo: DOTA-v1.0, train → val, ghép patch về ảnh gốc, rotated-mAP50 theo devkit (VOC07, bỏ difficult). Evaluator: `vrdet/eval/dota.py`.
- Không claim nếu chưa có số đo cùng điều kiện. Vòng cũ (CAD-VLMDet) lưu ở `research/archive_cad_vlmdet/`.

## Bối cảnh định lượng (nghiên cứu web 2026-10-07, nguồn ở `docs/RESEARCH_PLAN.md` §2)

- Mốc test (trainval → test server):
  - YOLO26 MS: 78.9 / 80.9 / 81.0 / 81.6 / 81.7 (n → x).
  - **RiO-DETR SS 78.4 / 80.3 / 80.9 / 81.7 / 81.8** (ECCV 2026, Apache). Theo bảng của RiO-DETR, YOLO26 SS chỉ đạt 77.7 / 79.7 / 80.4 / 80.2 / 80.4.
  - O2-DFINE-s SS 76.14. O2-DEIM-R50 MS 80.15.
- Mốc val: YOLO26s, schedule đầy đủ, val AP50 76.0 (theo paper YOLO26). Val (train-only SS) thường thấp hơn test khoảng 4–5 điểm.
- Điểm mất tập trung ở BR, SBF, RA, GTF, HA, HC (54–59% phần còn thiếu). PL, SH, TC, ST đã bão hoà.
  - Decoder DETR thắng YOLO26 ở LV +6.8, GTF +4.4, BR +3.9, SBF +2.9.
  - Nhưng thua ở vật nhỏ: SP −4.0, PL −2.8, SV −1.1. Vì vậy cần **nhánh dense stride-8**.
- Công thức train đáng giá ngang kiến trúc: xoay ngẫu nhiên +4.0, multi-scale +2–4.5, mosaic + schedule dài +1.3, SWA +1.
  - Ngưỡng score 0.25 thay vì 0.05 mất 1.25 điểm.
  - ProbIoU hơn KLD 2.1 điểm (PP-YOLOE-R). Chamfer là matching cost tốt nhất (O2: +1.09).

## E1: baseline cùng điều kiện (24 epoch, 1024/200 SS, train → val, init COCO)

### E1a `e1-yolo26s-dota-24e`: YOLO26s-obb (Ultralytics, chỉ đo)
- Mục đích: mốc cùng điều kiện. Không có pass/fail.
- Sanity: mAP50 val phải ở khoảng hợp lý (≥ 65). Nếu thấp bất thường thì kiểm lại split/evaluator, đừng kết luận gì.
- Đo thêm latency PyTorch fp16 bs1 trên G4 (chỉ tham khảo, không phải số TensorRT).
- **Kết quả (2026-10-07 17:22): val mAP50 74.77, mAP50:95 49.49.** Sanity OK: mức hợp lý so với 76.0 của paper YOLO26 (schedule dài).
  - Theo class: PL 90.1, BD 75.2, **BR 52.2**, **GTF 67.3**, SV 74.7, LV 84.0, SH 89.1, TC 90.9, **BC 69.0**, ST 89.0, SBF 74.3, **RA 69.6**, HA 76.4, **SP 66.4**, **HC 53.4**.
  - Train khoảng 0.6 h (185 ảnh/s, 42 GB). Latency PyTorch e2e (gồm pre/post) 8.9 ms/ảnh 1024 trên G4. Tốn khoảng 7 CU.

### E1b `e1-vrdet-s-dota-24e`: VRDet-S baseline
- Cấu hình: D-FINE-S COCO (HGNetv2-B0, Apache) chuyển sang OBB, gồm:
  - rotated FDR (4 cạnh trong hệ trục box + 1 phân phối góc dư ±45°, GT căn về biểu diễn gần nhất);
  - deformable sampling xoay theo θ;
  - matching: focal 2 + Chamfer 5 + KLD 2;
  - loss: MAL (IoU xoay chính xác) + L1 (căn góc) 5 + KLD 2 + FGL 0.15 + DDF 1.5;
  - oriented CDN; 300 query; aug flip + rot90 + HSV.
- Pass (để đi tiếp E2): mAP50 ≥ E1a − 2. Fail: tìm bug trước (matching, góc, merge), không thêm module mới.
- **Kết quả (19:20): val mAP50 69.89, mAP50:95 46.13. FAIL (−4.9 so với YOLO26s).** Latency PyTorch fp16 bs1 9.4 ms (YOLO e2e 8.9).
  - Ngang YOLO: PL 90.2, LV 83.8, TC 90.6, ST 88.6, HA 76.2. Hơn YOLO: GTF 68.7 (+1.4).
  - Thua YOLO:
    - class hiếm: HC −15.9 (72 GT), SBF −13.9 (87), RA −9.6 (164), BD −7.3 (209), BC −6.3 (124);
    - vật nhỏ dày: SV −4.3, SP −9.1, BR −5.8.
  - Không phải bug: class phổ biến ngang YOLO.
  - Ở class hiếm, recall gần bằng YOLO (BD 86/85, GTF 92/85, SBF 90/95, HC 79/82) nhưng AP thấp, nên lỗi nằm ở **xếp hạng điểm / nhầm class**. Matching one-to-one cho 1 mẫu dương/object/layer, còn TAL của YOLO cho khoảng 10. Lịch 24 epoch làm class hiếm thiếu giám sát.
  - SV recall 88 so với 94 là do trần 300 query/patch.
  - → Đúng các điểm H4 (dense one-to-many: nhiều mẫu dương, nhiều slot) nhắm tới. Quyết định: không đổi baseline, chờ E2; thêm 1 seed để đo nhiễu.

## E2: ablation lõi kiến trúc (mỗi run đổi đúng 1 biến so với E1b, cùng 24 epoch / 1024 SS / train→val)

Nhiễu giữa các run trên DOTA val khoảng ±0.3–0.5. Δ < 0.5 coi là chưa kết luận (lặp seed nếu hứa hẹn).

| Job | Biến | Giả thuyết | Pass |
|---|---|---|---|
| `e2-h4-dense-s` | `--dense` | Head dense one-to-many (TAL xoay) làm giám sát phụ giúp encoder. Khi ghép đầu ra (hợp / định tuyến theo kích thước), vật nhỏ tăng | decoder-only ≥ E1b + 0.5, **hoặc** kiểu ghép tốt nhất ≥ max(E1b, YOLO) + 0.5, kèm SV/PL/SP/HC tăng |
| `e2-h4c-dq-s` | `--dense-queries` | Query decoder lấy từ head dense (top-k sau lọc trùng) cho vật nhỏ "chỗ ngồi" trong 300 query | ≥ E1b + 0.5 |
| `e2-h6-ctx-s` | `--context` | Token ngữ cảnh toàn ảnh (thumbnail) giúp các class cần ngữ cảnh | ≥ E1b + 0.5, BR/HA/SBF/GTF/RA tăng |
| `e2-h3-strip-s` | `--strip-k 11` | Strip-context giúp vật dài mảnh | ≥ E1b + 0.5, hoặc BR/HA tăng ≥ 1.5 |

### Tiến độ giữa chừng E2 (19:41, sub-mAP50 trên 100 ảnh val, epoch 6 / 12 / 18)

| Run | e6 | e12 | e18 |
|---|---|---|---|
| E1b baseline | 61.2 | 65.2 | 67.3 |
| H4 dense (`--dense`) | **63.3** | – | – |
| H4c dense-query | 59.6 | – | – |
| H3 strip | 60.9 | 66.0 | 66.2 |
| H6 ngữ cảnh | 60.4 | 63.1 | 64.1 |

- H6 thấp hơn đều. Đã tìm ra lỗi thiết kế: thumbnail đi qua backbone dùng chung ở chế độ train, làm hỏng running stats của BN mà tile dùng khi eval. Sửa bằng cách cho thumbnail đi qua với BN eval, no_grad, rồi kiểm lại ở E3.
- Tốc độ: `--channels-last --compile` cho 86.4 ảnh/s so với 60.3 (×1.43). Các run từ E3 trở đi đều bật.

**Kết quả cuối E2 (full val, so với E1b 69.89):**
- **H3 strip: 69.76 (−0.13), mAP50:95 +0.49. Chưa kết luận** (ngang nhiễu). SBF +3.9, BD +1.7, LV +1.2; HC −6.1, BC −2.5, GTF −2.4.
- **H6 ngữ cảnh (bản lỗi BN): 67.96 (−1.93). Bác bỏ bản này.** HC −20.7, BC −6.6, GTF −5.5; SBF +6.7, RA +1.6. Đã sửa, đo lại ở E3.
- Nhận xét phương pháp: class hiếm (HC 72, SBF 87, BC 124, GTF 131 GT) dao động ±5–20 điểm giữa các run, chiếm phần lớn nhiễu của mAP50. Cần seed replicate, và nên xem thêm mAP trên các class ≥ 200 GT.

## E3 (bật speed opts; so với trung bình E1b và seed1)

| Job | Biến | Giả thuyết | Pass |
|---|---|---|---|
| `e1-vrdet-s-seed1` | seed 1 | đo nhiễu giữa seed (và kiểm chứng speed opts không đổi accuracy) | **70.10 (+0.21)**. Nhiễu khoảng ±0.2 mAP50, class lẻ ±2.5. Speed opts giữ accuracy. Baseline ≈ 70.0 |
| `e3-ctx-fix-s` | `--context` (đã sửa BN) | H6 công bằng: ngữ cảnh toàn ảnh giúp BR/HA/SBF/GTF/RA | ≥ baseline + 0.5 |

**Kết quả `e3-ctx-fix-s`: 69.71 (−0.18).**
- Class cần ngữ cảnh tăng: SBF +5.4, SP +4.5, RA +2.6, GTF +2.6, BD +2.5, BC +2.1.
- Nhưng **HC −19.3**. Bản lỗi trước cũng −20.7, nên đây là hiệu ứng có hệ thống. Nghi prior bối cảnh "sân bay → máy bay".
- Bỏ HC: 73.39 so với 72.4 (+1.0). 10 class phổ biến: 76.31 so với 75.9.
- → **H6 có tác dụng thật cho các class ngữ nghĩa**, nhưng cần chống thiên kiến theo bối cảnh với class hiếm (RFS / context dropout).
- Phân rã khoảng cách tới YOLO26s: 5 class hiếm kém −8.4, 10 class phổ biến kém −2.9.
| `e3-rfs-s` | `--rfs 0.1` | Repeat-factor sampling sửa điểm yếu class hiếm (HC, SBF, RA, BD, BC) | ≥ baseline + 0.5, class hiếm tăng |

## E2 cũ (dự kiến ban đầu)
- H4 dense: thêm head one-to-many dense (TAL xoay) trên P3–P5 của encoder, dùng làm giám sát phụ và nguồn query. Kỳ vọng tăng SV/PL/SP.
- H1 loss: ProbIoU thay KLD (cost và loss).
- Số query: 300 → 600.
- H3 ngữ cảnh: strip/large-kernel depthwise trong CCFM. Kỳ vọng tăng BR/GTF/SBF/RA/HA.
- Công thức train: xoay ngẫu nhiên góc bất kỳ, mosaic.
