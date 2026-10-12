# Pretrain lõi OpenOBB9 bằng chưng cất từ hai thầy

Tài liệu này hướng dẫn chạy `openobb pretrain` cho OpenOBB9. Lập luận thiết kế nằm ở [OPENOBB9.md](OPENOBB9.md).

## 1. Pretrain làm gì

- Lõi của OpenOBB9 (backbone + neck) là kiến trúc mới. Không có weights COCO hay ImageNet cho nó.
- Train từ đầu trên data nhỏ thì chững lại (v2, v6: LEDGER F25, F40). Vì vậy lõi phải học trước, trên ảnh **không cần nhãn**.
- Cách học: chưng cất đặc trưng từ hai "thầy" đã đóng băng.
  - **Thầy `dfine`:** lõi D-FINE-X COCO (HGNetv2 + encoder lai). Đây chính là lõi của v3 / v8.
    Thầy nhận cùng ảnh, cùng độ phân giải với học trò (RGB trong [0, 1]).
    P3 / P4 / P5 của thầy (384 kênh với cỡ x) là đích cho P3 / P4 / P5 của học trò.
  - **Thầy `dinov2`:** DINOv2 (mặc định `dinov2_s`). Đặc trưng patch ở lớp cuối là đích cho P4 và P5 của học trò.
- Mỗi cặp (tầng học trò, map của thầy) có một phép chiếu nhỏ: conv 1×1 → BN → GELU → conv 1×1.
  Loss là 1 − cosine tại từng vị trí. Tổng loss là tổng có trọng số. Các phép chiếu bị bỏ sau pretrain.
- **Tầng nào cũng có loss chưng cất** (P3, P4, P5). Đây là cách chặn lỗi "tầng chết" đã gặp ở v3 (OPENOBB9.md mục 5, F37).
  Nếu bỏ thầy `dfine`, P3 tự nhận đích DINOv2 để không tầng nào bị bỏ trống.
- Vì sao làm vậy: Mệnh đề 1 ở [OPENOBB9.md mục 3](OPENOBB9.md#3-lập-luận-a-vì-sao-độ-chính-xác-không-kém-v8).
  Nếu học trò khớp được thầy D-FINE (cosine cao), head của v8 đặt trên học trò làm được gần như những gì nó làm trên lõi v3.
  Điều này chỉ đúng khi các giả định G1–G3 đúng. Phép thử T2 bên dưới kiểm G1 và G2.
- File kết quả `weights/last.pt` chỉ chứa khoá `backbone.*` và `neck.*`. Không có head, không có thầy, không có phép chiếu.

## 2. Chạy trên Colab

### 2.1 Cài đặt

```bash
!pip install -q --force-reinstall --no-deps git+https://github.com/Nguyenchitrai62/OpenOBB.git
!git clone -q https://github.com/Nguyenchitrai62/OpenOBB.git /content/OpenOBB   # chỉ cần cho script tải DOTA
```

### 2.2 Chuẩn bị ảnh (không cần nhãn)

Nên trộn ba nguồn. **Ảnh bản vẽ là bắt buộc** (giả định G2: ảnh pretrain phải giống ảnh dùng thật).

```bash
# a) Ảnh tổng quát: một phần COCO (val2017, 5k ảnh, khoảng 1 GB)
!mkdir -p /content/pre/coco && cd /content/pre/coco && wget -q http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip

# b) Ảnh OBB: DOTA cắt sẵn 1024 px (khoảng 15.8k patch, khoảng 6 phút)
!python /content/OpenOBB/colab/data/get_dota.py --root /content/datasets --splits train
#    -> ảnh ở /content/datasets/dota1_1024/images/train

# c) Ảnh bản vẽ của user (Drive hoặc upload), chia sẵn hai thư mục KHÔNG trùng nhau:
#    /content/drawings/pretrain   dùng để pretrain (có thể gồm ảnh train của Wall_Color, không cần nhãn)
#    /content/drawings/heldout    100–300 bản vẽ CHƯA BAO GIỜ dùng để pretrain (cho phép thử T2)
```

### 2.3 Pretrain

Nên chạy **1 epoch trước** để đo tốc độ, rồi mới quyết định số epoch (bài học L8).

```bash
!openobb pretrain model=openobb9x \
    data=/content/pre/coco/val2017,/content/datasets/dota1_1024/images/train,/content/drawings/pretrain \
    val_data=/content/drawings/heldout \
    teachers=dfine,dinov2 teacher=dinov2_s \
    epochs=1 imgsz=1024 batch=16 workers=8 name=speed
```

Sau đó chạy đầy đủ (ví dụ 10 epoch):

```bash
!openobb pretrain model=openobb9x \
    data=/content/pre/coco/val2017,/content/datasets/dota1_1024/images/train,/content/drawings/pretrain \
    val_data=/content/drawings/heldout teachers=dfine,dinov2 epochs=10 imgsz=1024 batch=16 workers=8
```

Kết quả ở `runs/pretrain/train/`: `weights/last.pt` (lưu mỗi epoch) và `pretrain_log.json`.

### 2.4 Train detection từ lõi đã pretrain

```bash
!openobb train model=openobb9x weights=runs/pretrain/train/weights/last.pt data=/content/Wall_Color/data.yaml epochs=100 imgsz=1024
```

Trainer nạp `backbone.*` và `neck.*`. Head được khởi tạo mới.

### 2.5 Tham số chính

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `teachers` | `dfine,dinov2` | Thầy dùng. Có thể chỉ `dfine` hoặc chỉ `dinov2` (để ablation) |
| `teacher` | `dinov2_s` | Cỡ DINOv2: `dinov2_s`, `dinov2_b`, `dinov2_l` |
| `teacher_dfine` | `x` | Cỡ lõi D-FINE làm thầy. `x` là lõi của v3-x / v8-x |
| `teacher_imgsz` | khoảng 0.7 × imgsz | Cỡ ảnh vào DINOv2 (bội số của 14) |
| `dino_layers` | `last` | `mid,last` để thêm lớp giữa của DINOv2 làm đích cho P4 / P5 |
| `w_dfine` | `1.0` | Trọng số đích dfine ở P3, P4, P5 (ví dụ `w_dfine=1,1,0.5`) |
| `w_dino` | `0.5` | Trọng số đích DINOv2 ở P4, P5 |
| `val_data` | không có | Thư mục ảnh giữ lại cho phép thử T2 |
| `val_limit` | `300` | Số ảnh val tối đa mỗi lần đo |
| `t2_min` | `0.8` | Ngưỡng của T2 |
| `lr`, `wd`, `warmup_epochs` | `2e-3`, `0.05`, `1` | AdamW, warmup rồi cosine |
| `limit` | không có | Chỉ lấy N ảnh đầu (để thử nhanh) |

## 3. Đọc kết quả

### 3.1 Dòng tốc độ và chi phí CU

Mỗi epoch in một dòng như sau:

```
[pretrain] epoch 1/10  loss 1.2345  dfine/P3 0.31 ... | 42.0 img/s  train 950s  val 20s  ETA 146 min
```

- `img/s` là tốc độ thật (gồm cả hai thầy). Đây là số dùng để tính chi phí.
- Thời gian một epoch (giờ) = số ảnh ÷ img/s ÷ 3600.
- **CU một epoch = thời gian (giờ) × 8.9** trên G4 (khoảng 8.9 CU/giờ).
- Ví dụ (số giả định, không phải số đo): 40 000 ảnh, 42 img/s → 952 s = 0.26 giờ → khoảng 2.4 CU mỗi epoch.
  10 epoch là khoảng 24 CU. Kế hoạch ở OPENOBB9.md mục 6 dự kiến 15–25 CU cho T2.
- Epoch đầu chậm hơn một chút (khởi động cuDNN). Nên tính theo epoch thứ hai nếu có.
- `ETA` là thời gian còn lại ước theo epoch vừa xong.

### 3.2 Phép thử T2

Khi có `val_data`, sau mỗi epoch in thêm:

```
[pretrain]   held-out cosine  dfine/P3 0.82  dfine/P4 0.85  dfine/P5 0.79  dinov2/P4 0.74  dinov2/P5 0.71  | T2 (dfine P3, P4 >= 0.8): pass
```

- Mỗi số là cosine trung bình giữa học trò (qua phép chiếu) và thầy, trên ảnh giữ lại.
  Ảnh val không tăng cường: lấy hình vuông ở giữa ảnh rồi resize về `imgsz`.
- Cuối run in kết luận: `gate T2 ... -> PASSED` hoặc `FAILED`.
- **Đạt** khi cosine của thầy `dfine` ở **P3 và P4 ≥ 0.8** (ngưỡng đặt trước, OPENOBB9.md mục 6).
- **Không đạt** thì: pretrain thêm hoặc tăng độ rộng lõi. Nếu vẫn thấp thì giả định G1 sai, dừng v9 (không tốn CU cho T3).
- `val_data` phải là ảnh bản vẽ **chưa dùng để pretrain** (giả định G2). Nếu có ảnh trùng, lệnh in cảnh báo.
- Mọi số được ghi vào `pretrain_log.json`: `loss_pairs`, `img_per_s`, `val_cos`, `t2_pass` theo từng epoch.
  `last.pt` cũng lưu `pretrain_val_cos` của epoch cuối.

## 4. Giấy phép

- Hai thầy đều **Apache-2.0**: D-FINE (weights COCO) và DINOv2. Chúng chỉ chạy **trong lúc pretrain**.
- Model OpenOBB9 xuất ra **không chứa** thầy hay phép chiếu. `last.pt` chỉ có `backbone.*` và `neck.*` của lõi mới.
- Weights tải tự động lúc chạy (`torch.hub`), không lưu trong repo.
- Dataset: COCO (ảnh có giấy phép riêng từng ảnh, annotation CC BY 4.0), DOTA (chỉ học thuật).
  **Data NC / học thuật chỉ dùng cho nghiên cứu và benchmark.** Bản thương mại nên pretrain lại trên ảnh sạch:
  bản vẽ nội bộ và ảnh tự sinh (AGENTS.md mục 4, 7).
