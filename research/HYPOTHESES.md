# Giả thuyết → Đánh giá lý thuyết → Mới train thử tập nhỏ
Quy tắc: KHÔNG train nếu chưa viết được (a) giả thuyết khả kiểm chứng, (b) bằng chứng nào sẽ xác nhận/bác bỏ, (c) thí nghiệm tập nhỏ rẻ nhất để kiểm tra.

## Yêu cầu bài toán (bất biến)
R1: Hiểu như VLM (ngữ nghĩa + hình học: cửa nằm trên tường, junction là đầu mút).
R2: Đẻ box hàng loạt 1 forward như YOLO (150–400 boxes/ảnh), không autoregressive từng box.
R3: Không chậm như VLM (chấp nhận chậm hơn YOLO đôi chút).
R4: Tổng quát (không chỉ floorplan) + thương mại được (MIT, không AGPL).

## Giả thuyết kiến trúc (đánh giá lý thuyết trước)
- **H1 (bulk-decode)**: dense anchor-free multi-scale head đáp ứng R2. Đánh giá: ĐÚNG về nguyên lý (FCOS/YOLO đã chứng minh dense xử lý ảnh dày; DETR 100 queries thua chắc với 396 boxes/ảnh). Không cần train để biết — chỉ cần giữ thiết kế dense.
- **H2 (hiểu topology)**: strip-GCAM + transformer P5 + topo-loss cho quan hệ wall→door/window (R1). Đánh giá: HỢP LÝ nhưng chưa chứng minh — rủi ro lớn nhất là transformer P5 (32× downsample ở 768px còn ~24×24) quá thô để suy luận vị trí cửa chính xác. Kiểm chứng: ablation tắt P5-transformer, nếu mAP cửa hiếm rớt ≥2 điểm → giữ, không thì cắt cho nhẹ (R3).
- **H3 (P2-junction)**: luồng stride-4 riêng + gán junction về P2. Đánh giá: ĐÚNG (evidence v1/v2: junction 0.70→0.77 cao nhất mọi class). Giữ.
- **H4 (level-assign theo input-px + focal cls)**: sửa v6. Đánh giá: ĐÚNG hướng (wall ×3, window ×10) nhưng chưa đủ → còn bệnh khác (nghi decode/NMS).
- **H5 (decode)**: NMS góc xoay + ngưỡng conf hiện tại đang giết box đúng. Đánh giá: NGHI NGỜ MẠNH (angle head loss đứng yên 0.23, NMS xoay với góc rác sẽ diệt true positive). Kiểm chứng rẻ nhất: sweep conf×NMS-mode trên weights cũ, KHÔNG retrain (đang chạy v7).
- **H6 (calibration)**: classifier quá/ thiếu tự tin theo class (window pred 30× GT, door 2 vs 99). Đánh giá: nếu H5 đúng một phần, H6 là phần còn lại. Kiểm chứng: đọc npred/ngt + PR-curve trong v7, rồi mới quyết định sửa loss (class-wise threshold / sampling) thay vì đoán mò.

## Ma trận thí nghiệm tập nhỏ (mỗi thí nghiệm chỉ trả lời 1 giả thuyết)
| Exp | Giả thuyết | Thí nghiệm (800/200, 768px) | Tiêu chí pass/fail | Tốn |
|---|---|---|---|---|
| v7 (running) | H5 decode | sweep conf×NMS, no-retrain | HBB-NMS hoặc conf thấp tăng mean-AP rõ → H5 ĐÚNG | ~30' GPU |
| v8 | H6 calibration | class-wise conf threshold theo npred/ngt | cân bằng P/R, door R>0.6 → H6 ĐÚNG | eval-only |
| v9 | H2 reasoning | ablation: tắt transformer P5 + GCAM, retrain 20ep | mAP cửa hiếm rớt ≥2đ → giữ module | ~3.5h GPU |
| v10 | H-train | angle-CSL + size-balanced sampling (nếu H5/H6 chưa đủ) | wall/window AP tăng | ~3.5h GPU |
| D4 | Tổng | base 54M full 16k, 100ep | chỉ chạy khi H2/H5/H6 đã pass | ~15-20h GPU |

## Kết quả xử lý (ghi chép trung thực, cập nhật liên tục)
- **H5: BÁC BỎ** (v7, eval-only): sweep conf {0.1,0.2,0.3} × NMS {rotated,HBB} gần như không đổi điểm (wall 0.126↔0.150) → decode/NMS không phải bệnh chính. Dừng đụng vào decode.
- **H6: XÁC NHẬN** (v7 npred/ngt): wall pred gấp 5× GT, door pred bằng 1/7 GT, slide/double-door pred = 0 → classifier mất cân bằng nặng.
- **Artifact eval (fixed exp4)**: ảnh không có GT class hiếm trước đây được chấm 1.0 → slide 0.345 = 69/200 và double 0.390 = 78/200 là SỐ ẢO. Đã sửa: skip ảnh không GT. Số thật của 2 class này ≈ 0.
- **H7: ĐÚNG MỘT NỬA, rồi BÁC quy tắc smallest-wins** (v8 retrain): window 0.10→0.19 (cạnh tranh với wall giảm, tốt), junction 0.772→0.790 (tốt), NHƯNG door 0.047→0.000 (npred=0, sập hoàn toàn). Cơ chế: junction tí hon nằm CHỒNG lên door/window/wall — smallest-wins để junction nuốt hết điểm chung → door chết. Kết luận: vấn đề là cấu trúc LỒNG nhau, không giải được bằng thứ tự ghi. Hướng đúng là **D3: tách junction ra head keypoint riêng**, khỏi cạnh tranh dense classifier (H8).
- **H2: XÁC NHẬN** (v9 ablation noreason 20.89M): tắt transformer P5 → window 0.192→0.141 (−5đ), junction 0.790→0.757 (−3.3đ), wall ~đứng yên. Suy luận toàn cục có tác dụng thật trên cửa/junction → GIỮ module (dù tốn +4.4M params). Bonus: bản noreason window npred 2525 ≈ GT 2256 (calibration đẹp hơn bản full 4929) — transformer giúp localization nhưng làm calibration lệch, ghi nợ nghiên cứu sau.
- **H8: BÁC** (v10 retrain): junction-ghi-trước KHÔNG hồi sinh door (vẫn npred=0), window còn tụt 0.192→0.098. Cạnh tranh điểm không phải (chỉ) cơ chế thứ tự → door chết có nguyên nhân khác (mất cân bằng liên-class / positives quá ít).
- **H9: BÁC** (v11 eval-only, quét tới conf 0.05 trên weights v10): door npred vẫn = 0 tuyệt đối, wall/junction bất động → scores bimodal (wall/junction ~1, door ~0). Không phải calibration — model đã học "door luôn vắng mặt".
- **H10 (đang kiểm bằng v12 retrain quyết định)**: "focal ×20" của ta là GIẢ — chỉ tăng positives mà không down-weight easy-negatives (door pos ~30 cells vs door neg ~5000 cells/ảnh, tỉ lệ 1:150). Fix: focal thật (1−p)^γ=2 + α sqrt-inverse-freq (wall 1.0, junction 1.1, window/door 2.5, slide/double ~6.5). Pass tối thiểu: door npred>0; pass mạnh: door AP>0.3. Quota còn ~9.7h — đây là phát retrain quyết định.
## Ngưỡng đột phá (định nghĩa trước, tránh tự dối mình — GIỮ NGUYÊN, không hạ chuẩn)
- Bắt buộc: mAP50-95 tổng ≥ YOLO11x (0.585) CÙNG điều kiện + mAP cửa hiếm/junction hơn ≥2 điểm + FPS ≥60% YOLO.
- Nếu 3 vòng (v8→v10) không đạt → kết luận trung thực: composition hiện tại chưa đủ, chuyển sang hướng mới (deformable attention / DETR-dense lai) thay vì train tiếp.
