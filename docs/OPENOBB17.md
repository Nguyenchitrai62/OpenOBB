# OpenOBB17: detector theo ranh giới (trường ranh giới + bám ranh giới)

`model=openobb17s|m|l|x`. Code: `openobb/models/openobb17.py`, `openobb17_loss.py`. Test: `tests/test_openobb17.py`.
Sổ cái: F64 (chẩn đoán), F65 (thiết kế, kỳ vọng ghi trước).

## 1. Bằng chứng (F64, đo trên CPU với checkpoint v12, 53 ảnh val Wall_Color)

| Đo | Kết quả | Hệ quả |
|---|---|---|
| Thay một thành phần box của mỗi detection bằng giá trị nhãn | box hoàn hảo: 0.478 → 0.799 mAP50-95. **Hai đầu dọc trục +0.116** (wall_300 +.235, note +.160, junction +.132, slide_door +.123); lệch ngang +.049; độ dày +.036; góc +.003 | Phần còn thiếu là hình học box, lớn nhất là chỗ object kết thúc |
| Kéo cạnh box NHÃN về mép ảnh mạnh nhất trong ±2 px | IoU trung vị với chính nó: wall .61, junction .60, wall_300 .51; làm trên detection: −0.090 | Box là quy ước gán nhãn, không phải mép nét thấy được; "nhìn kỹ pixel hơn" (v15) không đủ |
| Tâm cạnh đầu của object nằm trên cạnh object khác (≤ 2 px) | wall_300 85%, slide_door 77%, wall 66%, junction 56% | Nét tường được chia đoạn; đầu object = ranh giới với object kế tiếp |
| Bootstrap thước | ±0.024 mAP50-95 (1σ) | Chỉ tính kết quả khi chênh ≥ 2σ hoặc có cặp hạt giống cùng phiên |
| Họ v14–v16 (head mới, `box=direct`) | v14 `neck=dfine` 50 ep: khoảng 0.62 / 0.38; v15 / v16: 0.60–0.67 / 0.39–0.42 | Không làm gốc; v17 xây trên **v12** (0.743 / 0.472, F59) |

## 2. Lập luận

| Bài học | Cơ chế | Kỳ vọng | Điều chứng minh sai |
|---|---|---|---|
| Mọi detector (v1–v16, YOLO) cho mỗi object tự hồi quy độ dài từ một điểm neo, độc lập với hàng xóm; sai số lớn nhất lại ở đầu box | **Trường ranh giới** stride 2, hai kênh: đầu (cạnh ngắn) và cạnh dài; giám sát bằng **mọi cạnh của mọi box nhãn** (dải Gaussian σ = 1 px, quality focal loss) | Đầu box định vị bằng bằng chứng cục bộ, sai số không tăng theo độ dài box | Thành phần "hai đầu" của `box_error_oracle.py` trên checkpoint v17 giảm < 15% |
| Hai object kề nhau có chung ranh giới | Cả hai đọc cùng một dải trên trường → ranh giới khớp nhau | wall / wall_300 / junction cùng tăng mAP50-95 | Tăng chỉ ở một class, class kề bên giảm |
| Box là quy ước, không phải mép ảnh (bám mép ảnh làm hại −0.09) | Trường **học** quy ước từ nhãn, không dùng gradient ảnh | Bám ranh giới học được không làm hại như bám mép ảnh | Hệ số bám (`snap.gain × 20`) cuối train ≈ 0: model tự tắt cơ chế |
| 250 ảnh, mỗi object chỉ một đích box | Giám sát dày đặc: khoảng 13 nghìn object × 4 cạnh mỗi epoch, hàng trăm nghìn pixel | Học nhanh, ít overfit hơn head box | `loss_field` không giảm sau 10 epoch |
| Phần mới phải bắt đầu đúng bằng bản trước | Hệ số bám khởi tạo 0, nhân với GAIN_SCALE = 20 để học nhanh | v17 lúc đầu = v12 (test `test_v17_starts_as_v12`) | — |

## 3. Chi tiết từng phần, tỉ lệ tận dụng

| Phần | Nội dung | Nguồn | Tham số (x) |
|---|---|---|---|
| Thân HGNetv2 + LSK có chặn + encoder lai + head kép DFL | nguyên v12 | D-FINE COCO (Apache-2.0) + phần OpenOBB tự viết | 68.0M |
| `BoundaryField` | 1x1 chiếu stride-4 thân (128 kênh) + P3 (384) → 1x1 trộn → RepConv 3×3 → RepConv 3×3 dilation 2 (64 kênh) ở stride 4 → 1x1 ra 8 kênh → pixel shuffle ×2: 2 bản đồ logit ở stride 2. Bias khởi tạo ở xác suất 1% | tự viết | 0.12M |
| `BoundarySnap` | Với mỗi box: 4 cạnh, mỗi cạnh 3 đường song song × 33 mẫu song tuyến trong bán kính tìm (đầu: clamp(0.15 L, 3, 16) px; cạnh: clamp(0.5 T, 2, 6) px), lấy trung bình, softmax(logit / τ) → độ dời kỳ vọng t*, nhân độ tin cậy (xác suất đỉnh) và hệ số bám (đầu, cạnh). Box mới dựng lại từ 4 cạnh | tự viết | 3 số |
| Loss | v12 (một-nhiều + một-một, cân bằng class) + loss box của bước bám (như bộ tinh chỉnh v14, ngưỡng ProbIoU 0.5, ứng viên top-256 + box nhãn rung) + quality focal loss của trường | v12 / v14 + tự viết | — |
| Suy luận | top-1500 ứng viên của head đang dùng (o2m hoặc o2o) được bám ranh giới trước NMS; hiệu chỉnh head / NMS cuối train như v12 | tự viết | — |

- Phần mới chiếm **0.18% tham số**, khoảng **16 GFLOPs** ở 960 × 1280 (+2.4% so với 682 của v12), 100% tự viết.
- Tắt phần mới: `v7_off=field` (model khi đó là v12).
- Fine-tune từ checkpoint v12 (`weights=openobb12x.pt`): mọi tensor của v12 nạp thẳng, chỉ trường và bước bám là mới.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Phép thử chính, rẻ nhất, khác đúng một yếu tố: fine-tune 30 epoch từ `openobb12x.pt`, cùng công thức với mốc F59
(lr 1e-3 cos theo iteration, batch 16, 1280, không nbs). A = v17; B = mốc F59 (chỉ chạy lại khi kết quả nằm vùng xám, mục 5).

| | B: v12 (F59: 0.743 / 0.472) | **A: v17 kỳ vọng** |
|---|---|---|
| mAP50-95 | khoảng 0.47–0.48 | **≥ B + 0.025** (khoảng 0.50) |
| mAP50 | khoảng 0.74 | không thấp hơn B quá 0.01 |
| wall_300 mAP50-95 | .29–.31 | ≥ .34 |
| junction mAP50-95 | .53 | ≥ .56 |
| wall mAP50-95 | .52 | ≥ .55 |
| Thành phần "hai đầu" (`box_error_oracle.py`) | +0.116 | ≤ +0.08 (giảm ≥ 30%) |
| `snap.gain × 20` cuối train | — | đầu ≥ 0.3 |
| Tốc độ G4 | 15.3 ms | ≤ 17 ms |

**Bác bỏ** nếu: A − B < 0.01 mAP50-95, hoặc thành phần "hai đầu" giảm < 15%, hoặc hệ số bám đầu < 0.1.
Nếu đạt: chạy FloorPlanCAD (24 ep, cặp v17 / v12) để xác nhận trên data lớn, rồi mới train dài từ COCO.

## 5. Lệnh

```bash
openobb train model=openobb17x weights=/content/openobb12x.pt data=/content/data.yaml imgsz=1280 epochs=30 batch=16 lr0=0.001 nbs=0 warmup_bias_lr=0 epoch_lr=False name=v17_ft
```

Mốc so sánh là F59 (v12 fine-tune 30 epoch từ cùng `openobb12x.pt`, cùng công thức: 0.743 / 0.472), không cần train lại v12.
Chỉ khi v17 rơi vào vùng xám (+0.01 đến +0.025 mAP50-95) mới chạy lại v12 cùng phiên để gỡ nhiễu giữa hai lần train:

```bash
openobb train model=openobb12x weights=/content/openobb12x.pt data=/content/data.yaml imgsz=1280 epochs=30 batch=16 lr0=0.001 nbs=0 warmup_bias_lr=0 epoch_lr=False name=v12_ft
```

`nbs=0 warmup_bias_lr=0 epoch_lr=False` giữ đúng công thức train cũ (cùng điều kiện với F59); bỏ ba tham số này là dùng công thức kiểu YOLO mới
(0.16.0). Dòng `model` trong log phải có "boundary field (stride 2) + boundary snapping".
Gửi lại: bảng theo class, confusion matrix, `Duplicate boxes`, `NMS calibrated`, `Speed`, và hai file `best.pt` (để chạy
`research/analysis/box_error_oracle.py` và đọc `snap.gain`).

## 6. Kết quả thật

**Run 1 (user, 10-15, F68):** 0.754 / 0.485, 15.9 ms. Không cùng điều kiện mục 4: công thức YOLO mới (`nbs 64`) và hai vòng fine-tune
(vòng 2 từ `best.pt` vòng 1). Tổng +0.013 so với F59 (vùng xám). wall .521, junction .515, wall_300 .270: **không đạt** cả ba mục tiêu class;
tốc độ đạt. Phần tăng nằm ở note / slide_door / double_door. Chưa kết luận cơ chế: chờ `snap.gain` và `box_error_oracle.py` trên `best.pt`.
