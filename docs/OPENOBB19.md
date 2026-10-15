# OpenOBB19: head theo nhóm tranh chấp (mỗi điểm lưới một box cho mỗi nhóm class)

`model=openobb19x` (= v18 + nhóm), hoặc công tắc `groups=auto` cho v17 / v18. Code: `openobb/models/groups.py`,
`openobb18.py` (`OpenOBB19`), `openobb17.py` (head nhóm), `openobb10_loss.py` (`group_assign`, `grouped_dense_losses`).
Test: `tests/test_openobb19.py`. Sổ cái: F72.

## 1. Bằng chứng (F72, đo trên nhãn và trên ứng viên của v12; CPU)

| Đo | Kết quả |
|---|---|
| Recall wall theo độ dài (ProbIoU 0.5, conf 0.25), v12 / YOLO26x | < 12 px: **0.33 / 0.69**; 12–24 px: 0.72 / 0.85; ≥ 24 px: 0.94–0.97 / 0.89–0.96 |
| Recall wall_300 < 12 px, v12 / YOLO26x | 0.63 / 0.75 |
| Object không có điểm lưới riêng (stride 8, một box mỗi điểm) | **junction 60%**, wall ngắn 86%, wall_300 ngắn 86% |
| Cùng chỉ số ở stride 4 / ảnh 1920 | wall ngắn vẫn 72% / 76%: lưới mịn hơn không giải quyết |
| Nới box nhỏ lên 16 px (kiểu STAL của YOLO26) | tranh chấp tăng 86% → 93% |
| Ai giành điểm của mẩu tường | junction 753, wall 245, door 99 |
| Ma trận tranh chấp (nghìn lần) | junction ↔ wall 7.4, junction ↔ door 1.6, wall ↔ door 1.3, wall ↔ wall 2.4 |
| Tách 3 nhóm (junction / wall_300 / còn lại) | không có điểm riêng: junction 60% → **2%**, wall_300 ngắn 86% → **1%**, wall ngắn 86% → 39% |

Khoảng cách mAP50 lớn nhất với YOLO là wall (−0.075): đúng bằng phần tường ngắn bị sót.
Nguyên nhân là **khuôn "một điểm lưới, một object"** (chung cho YOLO và mọi OpenOBB). Trên bản vẽ, các object chạm và chồng
nhau ở đầu (mẩu tường nằm trong box junction), nên phần lớn object không có điểm của riêng mình.

## 2. Cơ chế

- Các class được chia thành G nhóm (mặc định 3) **tự động từ nhãn train**: tô màu tham lam ma trận tranh chấp; class hay
  giành điểm của nhau nằm ở nhóm khác nhau. Wall_Color: junction | wall + wall_300 | door + slide_door + double_door + note.
- Head kép của v12 giữ nguyên; lớp ra của nhánh box có G × kênh: mỗi điểm đoán **một box cho mỗi nhóm**. Logit class vẫn
  một bộ cho mọi class.
- Gán nhãn: mỗi nhóm gán riêng (object của nhóm, box của nhóm). Hai object khác nhóm chồng nhau đều có điểm dương ở cùng vị trí.
  Mục tiêu class của các nhóm cộng lại (cột rời nhau) thành một loss phân loại.
- Suy luận: mỗi (điểm, nhóm) là một ứng viên, chỉ mang các class của nhóm đó; NMS theo class như cũ (không có box trùng chéo
  nhóm vì class rời nhau). Bám ranh giới (v17) chạy trên các ứng viên này.
- Khởi tạo từ checkpoint một nhóm (v12 / v17 / v18): box của nhóm 0 được chép sang mọi nhóm → lúc đầu cho đúng kết quả của
  model gốc (test `test_grouped_model_starts_as_one_group_model`).
- Chi phí: lớp ra nhánh box × 3 (khoảng +0.7 GFLOPs, < 0.2%), NMS trên 3 × ứng viên (vẫn cắt top 4000).

## 3. Ba mặt so với YOLO

| Mặt | Cơ chế | Nguồn |
|---|---|---|
| Chính xác | head nhóm (tường ngắn, wall_300, junction có điểm riêng) + trường ranh giới | F72, F64 |
| Nhanh | neck RepConv chưng cất (v18): 12.2 ms so với 15.3–15.9 ms | F70 |
| Ít box trùng | hiệu chỉnh ưu tiên NMS ít trùng (7.3% → 2.2% trên v17) | F69 |

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

**A. `openobb17x groups=auto`** từ `openobb17x.pt` (F68), 30 ep, cùng công thức với F68 (một yếu tố: nhóm):
mAP50 ≥ 0.775 (F68 0.754), wall mAP50 ≥ .88 (.844), wall_300 mAP50 ≥ .76 (.712), junction 50-95 ≥ .53 (.515), box trùng ≤ 3%.

**B. `openobb19x`** từ `openobb12x.pt` (thân + head, thầy tự bật), 60 ep, công thức cũ:
≥ 0.77 / ≥ 0.50 (YOLO26x 0.746 / 0.499), ≤ 12.5 ms G4 batch 16, box trùng ≤ 3%.

Chẩn đoán đi kèm: `research/analysis/miss_by_length.py` trên checkpoint mới: recall wall < 12 px ≥ 0.6 (v12 0.33, YOLO 0.69).
**Bác bỏ** nếu recall wall < 12 px < 0.5, hoặc wall mAP50 tăng < 0.02 so với model gốc cùng điều kiện.

## 5. Lệnh

```bash
# A: cơ chế nhóm trên bản tốt nhất (v17)
openobb train data=/content/data.yaml model=openobb17x groups=auto epochs=30 imgsz=1280 batch=16 lr=0.001 weights=/content/openobb17x.pt
# B: bản đột phá đủ ba mặt
openobb train data=/content/data.yaml model=openobb19x epochs=60 imgsz=1280 batch=16 lr=0.001 nbs=0 warmup_bias_lr=0 epoch_lr=False weights=/content/openobb12x.pt
```

Log phải có dòng `[groups] contention-aware class groups (from the training labels): ...`.

## 6. Kết quả thật

**A. `openobb17x groups=auto` (F74):** 0.757 / 0.487 so với cùng điều kiện không nhóm (F73) 0.749 / 0.483. wall mAP50 +0.032 (tường bị sót −33%),
junction 50-95 +0.026, wall_300 −0.072 (nhầm sang junction 13% → 21%). Tiêu chí: junction đạt; mAP50 tổng, wall ≥ .88, wall_300, box trùng không đạt;
không bị bác bỏ. Box trùng 7.4% do hiệu chỉnh giữ NMS 0.7; từ 0.18.2 (DUP_TOL 0.01) sẽ chọn 0.5 (2.2%).
