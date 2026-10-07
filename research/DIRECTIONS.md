# OpenResearch-style continuous research — CAD-VLMDet breakthrough loop
Chạy liên tục đến khi có kiến trúc đột phá (vượt YOLO baseline rõ rệt) hoặc user dừng thủ công.

## Experiment tree (git-native: mỗi exp = 1 commit snapshot + log + artifacts)
- exp0 (done): prototype nano/small — CPU smoke, loss 5.158→4.765, infer đúng mật độ.
- exp1 (running, kernel v4): small 20ep vs yolo11x-obb 20ep, 800 imgs, 768px, T4. Evidence: kaggle_pkg/REPORT_METRICS.txt (watcher tự tải khi COMPLETE).

## Các hướng song song (mỗi hướng = 1 agent session độc lập, không giẫm chân)
- D1 — Sâu hơn ở reasoning: tr_layers 4→6, thêm deformable cross-scale attention P3-P5. Giả thuyết: quan hệ wall-door xa cần attention deformable, không chỉ MHSA P5.
- D2 — Mạnh tay topology: topo_w 0.2→0.5 + wall-conditioned NMS (door/window không overlap wall bị hạ điểm). Giả thuyết: precision cửa hiếm tăng.
- D3 — Junction như keypoint: thêm heatmap head + loss focal riêng ở P2, NMS ưu tiên endpoint-on-wall. Giả thuyết: recall junction + mAP50-95 tăng.
- D4 — Scale thật: base 54M, full 16k, imgsz 1024, 100ep, A100. Chỉ chạy khi D1-D3 chứng minh hướng đúng trên subset.
- D5 — Tốc độ: distill base→small, TensorRT FP16, đo FPS T4 vs YOLO11x (chấp nhận chậm hơn đôi chút, mục tiêu ≥60% FPS YOLO).

## Luật vòng lặp (autoresearch)
1. Watcher (5 phút) tải evidence mới → ghi REPORT_METRICS.txt.
2. research_loop.py đọc evidence → so với baseline YOLO + exp trước:
   - Vượt rõ (mAP50-95 tổng + cửa hiếm đều hơn ≥2 điểm): CHỐT, viết báo cáo đột phá + ablations xác nhận.
   - Hòa/thua: chọn hướng D1/D2/D3 theo lỗi (phân tích class-wise) → patch code → đẩy kernel v(n+1) → tiếp tục.
3. Không bao giờ claim vượt trội khi chưa có số. Mỗi exp đều có baseline chạy cùng điều kiện.
4. Dừng khi: user nói dừng, hoặc hết quota GPU (30h/tuần), hoặc đã chốt đột phá.

## Dừng thủ công
- Tắt monitor: schtasks /Delete /TN FloorplanVLMDetWatch /F  (và kill tiến trình watch.py)
- Dừng research: xóa/xả task + nói "dừng" là đủ.
