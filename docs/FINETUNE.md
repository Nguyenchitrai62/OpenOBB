# Train / fine-tune VRDet trên dataset OBB tự gắn nhãn

Dataset theo chuẩn OBB 8 điểm (cùng định dạng nhãn bản export "training dataset" của AI_Takeoff), dùng trực tiếp, không cần chuyển đổi.

**Cách nhanh nhất là chạy notebook Colab [`colab/VRDet_train.ipynb`](../colab/VRDet_train.ipynb):** giải nén zip từ Drive, train, xem kết quả, dự đoán.

## 0. Cài đặt

```bash
pip install git+https://github.com/Nguyenchitrai62/vrdet.git     # hoặc trong repo: pip install -e .
```

- Cần torch (CUDA), opencv-python, shapely ≥ 2, scipy, numpy, pyyaml (Colab có sẵn).
- Lần train đầu tự tải weights khởi tạo D-FINE COCO (Apache-2.0).

## 1. Dữ liệu

```
<root>/data.yaml                 names: [...] hoặc {0: ..., 1: ...}; train:/val: thư mục ảnh
<root>/train/images, train/labels     (hoặc images/train + labels/train)
<root>/valid/images, valid/labels     (tuỳ chọn: không có thì tự tách 15% trang train)
```

- Mỗi dòng nhãn: `class_id x1 y1 x2 y2 x3 y3 x4 y4`, toạ độ chuẩn hoá [0, 1] theo ảnh.
- `train: ../train/images` (kiểu Roboflow) cũng đọc được.
- **Trang được cắt thành tile `imgsz` px**, chồng nhau 200 px.
  - `scale` thu nhỏ trang trước khi cắt. Ví dụ trang 1280 px với `scale=0.8` ra đúng 1 tile 1024.
  - Khi eval và predict, kết quả được ghép lại theo trang gốc.
- Tile chỉ cắt một lần, lưu cache ở `~/.cache/vrdet` (đổi chỗ bằng `cache=` hoặc biến `VRDET_CACHE`).

## 2. Train

```bash
vrdet train data=path/data.yaml model=s epochs=24 imgsz=1024 scale=0.8 project=runs name=exp
```

```python
from vrdet import Detector
r = Detector("s").train(data="path/data.yaml", epochs=24, imgsz=1024, scale=0.8, project="runs", name="exp")
```

- `model`:
  - `s` (10.3M tham số), `m`, `l`, `x` (63M) để train từ đầu.
  - Đường dẫn `best.pt` để **fine-tune**: kiến trúc lấy theo checkpoint, class trùng tên giữ lại trọng số.
- `batch` tự chọn theo VRAM. Bản S, batch 16, tile 1024 cần khoảng 21 GB.
- Số query tự tăng (600–900) khi tile dày object (p99 trên 240 object/tile).
- Công thức mặc định là công thức tốt nhất đã đo: RFS 0.1, nhóm query một-nhiều 900, AQD, loss góc, IoU-cost, 900 query khi suy luận.
  - `recipe=False` để tắt.
  - Mọi flag của `python -m vrdet.train` đều truyền được dạng `key=value`, ví dụ `lsk=True`, `rfs=0`, `compile=False`.
- **Resume:** chạy lại đúng lệnh, run chưa xong sẽ train tiếp từ `last.pt`. `resume=False` hoặc tên mới để train lại từ đầu.

Kết quả nằm trong `runs/<name>/`:

| File | Nội dung |
|---|---|
| `best.pt` | EMA weights tốt nhất trên val (gọn), dùng để deploy hoặc fine-tune |
| `last.pt` | checkpoint đầy đủ, dùng để resume |
| `eval_val.txt`, `eval_val_q900.txt` | AP50, AP50:95, recall theo class, tính trên trang gốc (protocol DOTA) |
| `metrics.jsonl` | mỗi epoch một dòng (loss, mAP val định kỳ) |
| `vrdet_args.json` | toàn bộ tham số đã dùng |

## 3. Đánh giá và suy luận

```bash
vrdet val     model=runs/exp/best.pt data=path/data.yaml
vrdet predict model=runs/exp/best.pt source=pages/ conf=0.3 save_dir=preds
```

Mỗi trang ra 3 file:
- `<tên>.txt`: `class x1 y1 x2 y2 x3 y3 x4 y4 score`, chuẩn hoá theo trang.
- `<tên>.json`: toạ độ pixel và tên class.
- `<tên>_vis.jpg`: ảnh có vẽ box.

Dùng trong Python: `Detector("best.pt").predict("page.png", conf=0.3, save_dir=None)` trả về
`{đường dẫn: [{"class", "class_id", "score", "poly"}]}`.
