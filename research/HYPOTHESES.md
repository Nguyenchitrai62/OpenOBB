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
- Kết quả: (chưa chạy)

## E2 (dự kiến, mỗi cái 1 ablation so với E1b)
- H4 dense: thêm head one-to-many dense (TAL xoay) trên P3–P5 của encoder, dùng làm giám sát phụ và nguồn query. Kỳ vọng tăng SV/PL/SP.
- H1 loss: ProbIoU thay KLD (cost và loss).
- Số query: 300 → 600.
- H3 ngữ cảnh: strip/large-kernel depthwise trong CCFM. Kỳ vọng tăng BR/GTF/SBF/RA/HA.
- Công thức train: xoay ngẫu nhiên góc bất kỳ, mosaic.
