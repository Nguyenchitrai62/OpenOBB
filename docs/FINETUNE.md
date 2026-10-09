# Train / fine-tune VRDet trên dataset OBB tự gắn nhãn

Dataset theo chuẩn OBB 8 điểm (cùng định dạng nhãn bản export "training dataset" của AI_Takeoff), dùng trực tiếp, không cần chuyển đổi.

- Kiến trúc model là của VRDet.
- Bộ khung train/infer lấy các ý tưởng quen thuộc của YOLO, tự viết lại: resize cả ảnh, mosaic/scale/translate, warmup, dừng sớm, tự giảm batch, `results.csv`/`results.png`.

**Cách nhanh nhất là chạy notebook Colab:**
- [`colab/VRDet_wall_color.ipynb`](../colab/VRDet_wall_color.ipynb): bản gọn, chỉ cài, giải nén, `openobb train`.
- [`colab/VRDet_train.ipynb`](../colab/VRDet_train.ipynb): bản đầy đủ, có chọn dataset, biểu đồ và ảnh dự đoán.

## 0. Cài đặt

```bash
pip install git+https://github.com/Nguyenchitrai62/OpenOBB.git     # hoặc trong repo: pip install -e .
```

- Cần torch (CUDA), opencv-python, shapely ≥ 2, scipy, numpy, pyyaml. Colab có sẵn.
- Lần train đầu tự tải weights khởi tạo D-FINE COCO (Apache-2.0).

## 1. Dữ liệu

```
<root>/data.yaml                 names: [...] hoặc {0: ..., 1: ...}; train:/val: thư mục ảnh
<root>/train/images, train/labels     (hoặc images/train + labels/train)
<root>/valid/images, valid/labels     (tuỳ chọn: không có thì tự tách 15% ảnh train)
```

- Mỗi dòng nhãn: `class_id x1 y1 x2 y2 x3 y3 x4 y4`, toạ độ chuẩn hoá [0, 1] theo ảnh.
  - `train: ../train/images` (kiểu Roboflow) cũng đọc được.
  - Dòng nhãn sai bị bỏ qua và có cảnh báo; log in số ảnh nền và số dòng hỏng.
- **Ảnh cỡ nào cũng được.** Mỗi ảnh được resize riêng về cạnh dài `imgsz`, giữ tỉ lệ, rồi pad thành ô vuông. Train, val và predict đều làm giống nhau, kết quả đổi về toạ độ ảnh gốc.
- **`tile=True`:** cho trang cực lớn có object rất nhỏ.
  - Cắt tile `imgsz` ở độ phân giải gốc, chồng nhau 200 px, giống SAHI.
  - `tile_scale=` để thu nhỏ trước khi cắt; không đặt thì tự chọn theo cỡ object.
- Dữ liệu đã chuẩn bị được cache ở `~/.cache/vrdet` (đổi chỗ bằng `cache_dir=` hoặc biến `VRDET_CACHE`).

> **Deploy (thay YOLO trong code sản phẩm):** xem [DEPLOY.md](DEPLOY.md): `from openobb import OpenOBB; OpenOBB("best.pt").predict(...)`.

## 2. Train

```bash
openobb train data=path/data.yaml model=s epochs=100 imgsz=1024        # -> runs/obb/train
```

```python
from openobb import Detector
r = Detector("s").train(data="path/data.yaml", epochs=100, imgsz=1024)   # r.best = .../weights/best.pt
```

**`model`:**
- **`vrdet5x`** (mới nhất, 127.9M): head dense v3 + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ.
  Chi tiết: [VRDET5.md](VRDET5.md). Cuối mỗi run in thêm các lỗi nhầm class thường gặp (confusion matrix).
- **`vrdet4x`** (mới nhất, 134.8M, ưu tiên độ chính xác): DINOv2 ViT-B + encoder/decoder DETR pretrained + nhánh dense
  oriented (giám sát phụ + đề xuất query), đầu ra decoder ∪ dense. Chi tiết: [VRDET4.md](VRDET4.md).
- **`vrdet3x`** (mới nhất, 64.6M): backbone + encoder pretrained COCO, head dense oriented dị hướng, NMS.
  Chi tiết: [VRDET3.md](VRDET3.md). Có thể khởi tạo từ checkpoint VRDet1 bằng `weights=<v1 .pt>`.
- **`vrdet2x`** (mới, 55.6M, train từ đầu, nên dùng `epochs=300`): kiến trúc dense segment-aware, không NMS.
  Chi tiết: [VRDET2.md](VRDET2.md). Có thêm `vrdet2n/s/m/l`. Chỉ cần đổi `model=`, mọi tham số khác giữ nguyên.
- `vrdet1s/m/l/x` = `s/m/l/x`: kiến trúc cũ (D-FINE-OBB).
- `s` (12.5M tham số), `m`, `l`, `x` (71M) để train từ đầu.
- Đường dẫn `best.pt` để **fine-tune**: kiến trúc lấy theo checkpoint, class trùng tên giữ lại trọng số.

**Mặc định, đổi bằng `key=value`:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `imgsz` | 1024 | cạnh dài ảnh đưa vào model (trang lớn nên 1280) |
| `batch` | tự chọn (`-1`) | theo VRAM và cỡ dataset (≥ 25 bước/epoch); `0.6` = dùng 60% VRAM; hết VRAM ở epoch đầu thì tự giảm một nửa |
| `mosaic` | 1.0 | ghép 4 ảnh, tắt ở `close_mosaic`=10 epoch cuối |
| `scale` | 0.5 | zoom ngẫu nhiên 0.5–1.5 |
| `translate` | 0.1 | dịch ngẫu nhiên ±10% |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.015, 0.5, 0.3 | đổi màu |
| `warmup_epochs` | 3 | khởi động learning rate |
| `lrf` | 0.01 | learning rate giảm tuyến tính từ `lr0` về `lr0 × lrf` ở cuối run (`cos_lr=True`: giảm theo cosine) |
| `cache` | auto | giải mã ảnh train một lần vào RAM nếu chiếm dưới 25% RAM (`cache=False` để tắt) |
| `patience` | 100 | dừng nếu mAP50-95 val không tăng sau N epoch |
| `optimizer` | auto | AdamW; không truyền `lr0` thì dùng LR đã đo cho từng cỡ (S 1e-4, X 6e-5) |
| `lr0` | theo cỡ model | truyền vào là dùng đúng giá trị đó (có cảnh báo nếu > 3 lần mặc định); kiểu DETR thường diverge ở mức 1e-3 của YOLO |
| `fliplr`, `flipud` | 0.5, 0.5 | xác suất lật ngang / dọc; `rot90=False` để tắt xoay 90° (symbol có chiều) |
| `freeze` | – | `backbone`, `encoder` (chỉ train decoder) hoặc số stage backbone; hợp với data rất ít |
| `time` | – | ngân sách giờ train; số epoch tự tính lại sau mỗi epoch để LR vẫn giảm hết |

- Val chạy mỗi epoch khi tập val nhỏ (≤ 2000 ảnh/tile).
- Số query tự tăng (600–900) khi ảnh dày object.
- Kiến trúc mặc định là bản chốt (c6): adapter LSK, RFS, nhóm query một-nhiều, AQD, loss góc, IoU-cost.
  - `recipe=False` để tắt.
  - Mọi flag của `python -m openobb.train` truyền được dạng `key=value`.
- **Thư mục kết quả như YOLO:** mặc định `runs/obb/train`, lần sau `train2`, `train3`... (`project=`, `name=` để đổi; `exist_ok=True` ghi đè thư mục cũ). Weights ở `weights/best.pt` và `weights/last.pt`.
- **Resume chỉ khi `resume=True`** (như YOLO): train tiếp run chưa xong mới nhất (hoặc `name=` / `resume=<last.pt>`) với **đúng tham số đã lưu**.
  - Chỉ đổi được `workers`, `cache`, `patience`, `time`; tham số khác bị bỏ qua và có thông báo.
  - Muốn train lại từ đầu: `resume=False` (sang thư mục mới), hoặc `resume=False exist_ok=True` (ghi đè thư mục cũ).
- **Tự phục hồi khi diverge:** epoch có nhiều bước NaN hoặc thống kê BatchNorm hỏng thì tự nạp lại `last.pt` của epoch trước và giảm LR một nửa (tối đa 3 lần, như YOLO).
- **Cache dữ liệu theo nội dung:** sửa ảnh hay nhãn thì lần sau tự chuẩn bị lại.
- **Gõ sai tên tham số:** báo lỗi kèm gợi ý, ví dụ `'epoch' is not a valid VRDet argument. Similar: epochs`.

**Màn hình** giống YOLO:

```
      Epoch    GPU_mem   cls_loss   box_loss   kld_loss angle_loss   dfl_loss  Instances       Size
     12/100      22.8G     1.0344     0.1456     0.4498     0.0092     1.1246        231       1024: 100% ━━━━━━━━━━━━ 16/16 2.7it/s 6.0s
                 Class     Images  Instances          P          R      mAP50   mAP50-95  (3.1s)
                   all         53       3465      0.747      0.626      0.662      0.426
```

Khi xong, màn hình in bảng theo từng class của `best.pt`. Kết quả nằm trong `runs/<name>/`:

| File | Nội dung |
|---|---|
| `weights/best.pt` | weights tốt nhất trên val (gọn), dùng để deploy hoặc fine-tune |
| `weights/last.pt` | checkpoint đầy đủ, dùng cho `resume=True` |
| `results.csv`, `results.png` | loss, P, R, mAP50, mAP50-95 theo epoch |
| `labels.jpg` | số object theo class và phân bố cỡ box |
| `val_pred.jpg` | dự đoán trên 4 ảnh val (xanh lá: nhãn, màu: dự đoán) |
| `eval_val.txt`, `eval_val_q900.txt` | bảng P / R (một ngưỡng conf chung, như YOLO) / mAP / F2 theo class |
| `train_batch0.jpg` | một batch sau augmentation kèm nhãn, để soát mosaic / zoom |
| `train_progress.log` | log chi tiết từng vòng lặp |

## 3. Đánh giá và suy luận

```bash
openobb val     model=runs/obb/train/weights/best.pt data=path/data.yaml
openobb predict model=runs/obb/train/weights/best.pt source=pages/ conf=0.3   # -> runs/obb/predict
```

- Ảnh được xử lý đúng như lúc train. `tile=True` để cắt tile ảnh rất lớn.
- **`conf`:**
  - `conf=auto` dùng ngưỡng riêng cho từng class, là ngưỡng cho F2 cao nhất trên val, được lưu trong `best.pt`.
  - Bảng val có cột F2 theo class.
- **`classes=`:** chỉ giữ các class này, theo id hoặc tên (ví dụ `classes=wall,door`). `names=` dùng cho checkpoint không có tên class.
- `openobb val model=best.pt` không cần `data=` nếu chạy trên cùng máy lúc train. Lệch thứ tự class giữa data và model thì báo lỗi.
- Mỗi ảnh ra:
  - `<tên>.txt`: `class x1 y1 ... x4 y4 score`, chuẩn hoá theo ảnh;
  - `<tên>.json`: toạ độ pixel và tên class;
  - `<tên>_vis.jpg`: ảnh có vẽ box.
- Dùng trong Python: `Detector("best.pt").predict("page.png", conf=0.3, save_dir=None)`.

## 4. Test nhanh bằng giao diện kéo thả (Streamlit)

Trên máy này đã có sẵn môi trường `.venv` (torch CPU, streamlit, VRDet): **bấm đúp `run_app.bat`** ở thư mục repo.

Máy khác thì tạo lại môi trường:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[app]"
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

- Đặt `best.pt` tải từ Drive vào `models/best.pt`. Hoặc sửa `MODEL_PATH` đầu file [app/streamlit_app.py](../app/streamlit_app.py), hoặc gõ đường dẫn ở thanh bên.
- **Thanh bên:** kéo thả ảnh; ngưỡng confidence và NMS IoU; chọn class; `imgsz` và tuỳ chọn cắt tile (mặc định giống lúc train).
- **Kết quả:** ảnh gốc và ảnh có box đặt cạnh nhau; bảng đếm theo class; tải nhãn TXT (đúng định dạng nhãn train) và JSON.
