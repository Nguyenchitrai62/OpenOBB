# Train / fine-tune OpenOBB trên dataset OBB tự gắn nhãn

Dataset theo chuẩn OBB 8 điểm (cùng định dạng nhãn bản export "training dataset" của AI_Takeoff), dùng trực tiếp, không cần chuyển đổi.

- Kiến trúc model là của OpenOBB.
- Bộ khung train/infer lấy các ý tưởng quen thuộc của YOLO, tự viết lại: resize cả ảnh, mosaic/scale/translate, warmup, dừng sớm, tự giảm batch, `results.csv`/`results.png`.

**Cách nhanh nhất là chạy notebook Colab:**
- [`colab/OpenOBB_wall_color.ipynb`](../colab/OpenOBB_wall_color.ipynb): bản gọn, chỉ cài, giải nén, `openobb train`.
- [`colab/OpenOBB_train.ipynb`](../colab/OpenOBB_train.ipynb): bản đầy đủ, có chọn dataset, biểu đồ và ảnh dự đoán.

## 0. Cài đặt

```bash
pip install git+https://github.com/Nguyenchitrai62/OpenOBB.git     # hoặc trong repo: pip install -e .
```

- Cần torch (CUDA), opencv-python, shapely ≥ 2, scipy, numpy, pyyaml. Colab có sẵn.
- Lần train đầu tự tải weights khởi tạo D-FINE COCO (Apache-2.0).

## 1. Dữ liệu

```
<root>/data.yaml                 names: [...] hoặc {0: ..., 1: ...}; train:/val: thư mục ảnh, file .txt liệt kê ảnh, hoặc list
<root>/train/images, train/labels     (hoặc images/train + labels/train)
<root>/valid/images, valid/labels     (tuỳ chọn: không có thì tự tách 15% ảnh train)
```

- Mỗi dòng nhãn: `class_id x1 y1 x2 y2 x3 y3 x4 y4`, toạ độ chuẩn hoá [0, 1] theo ảnh.
  - Đọc giống Ultralytics: thư mục ảnh được quét cả thư mục con; nhãn của mỗi ảnh là đường dẫn ảnh với `/images/` cuối cùng
    đổi thành `/labels/` và đuôi `.txt` (ảnh và `.txt` để chung thư mục cũng được). Ảnh không có file nhãn là ảnh nền.
  - `train: ../train/images` (kiểu Roboflow) cũng đọc được.
  - Dòng nhãn sai bị bỏ qua và có cảnh báo; log in số ảnh nền và số dòng hỏng.
- **Ảnh cỡ nào cũng được.** Mỗi ảnh được resize riêng về cạnh dài `imgsz`, giữ tỉ lệ, rồi pad thành ô vuông. Train, val và predict đều làm giống nhau, kết quả đổi về toạ độ ảnh gốc.
- **`tile=True`:** cho trang cực lớn có object rất nhỏ.
  - Cắt tile `imgsz` ở độ phân giải gốc, chồng nhau 200 px, giống SAHI.
  - `tile_scale=` để thu nhỏ trước khi cắt; không đặt thì tự chọn theo cỡ object.
- Dữ liệu đã chuẩn bị được cache ở `~/.cache/openobb` (đổi chỗ bằng `cache_dir=` hoặc biến `OPENOBB_CACHE`).

> **Deploy (thay YOLO trong code sản phẩm):** xem [DEPLOY.md](DEPLOY.md): `from openobb import OpenOBB; OpenOBB("best.pt").predict(...)`.

## 2. Train

```bash
openobb train data=path/data.yaml model=openobb3x epochs=100 imgsz=1280   # -> runs/obb/train
```

```python
from openobb import OpenOBB
r = OpenOBB("openobb3x").train(data="path/data.yaml", epochs=100, imgsz=1280)   # r.best = .../weights/best.pt
```

**`model`:**

| `model=` | Kiến trúc | Tham số | Ghi chú |
|---|---|---|---|
| `openobb11s/m/l/x` | **mới, chưa train:** OpenOBB8 + tầng quan hệ train một-một: mỗi object đúng một box, không NMS (bỏ box trùng / mảnh vụn của tường dài). Đi tiếp được từ `best.pt` của v8 / v10 (`weights=`) | x: 66.6M | [OPENOBB11.md](OPENOBB11.md) |
| `openobb12s/m/l/x` | **mới, chưa train:** lõi OpenOBB7 + head dense kép (một-nhiều để học, một-một để suy luận): mỗi object một box, không NMS, không tầng quan hệ | x: 68.0M | [OPENOBB12.md](OPENOBB12.md) |
| `openobb13s/m/l/x` | **mới, chưa train, nhắm độ chính xác:** OpenOBB12 + luồng độ phân giải gốc + lưới P2 (wall_300 và junction có điểm lưới riêng). `imgsz` là độ phân giải đầy đủ (Wall_Color: `imgsz=2560`), lõi chạy ở một nửa | x: 70.1M | [OPENOBB13.md](OPENOBB13.md) |
| `openobb14s/m/l/x` | **mới, chưa train, nhắm cả độ chính xác lẫn tốc độ:** thân HGNetv2 pretrained (COCO) + neck conv dày tái tham số hoá tự thiết kế (thay LSK + encoder lai, 45% FLOPs của v12) + head kép nặng, hồi quy box trực tiếp (không DFL) + bộ tinh chỉnh box đọc nét ở stride 4 ngay trên cạnh box dự đoán. `box=direct` mặc định, `ref_k=1500` | x: 62.7M | [OPENOBB14.md](OPENOBB14.md) |
| `openobb15s/m/l/x` | **mới, chưa train, nhắm độ chính xác:** OpenOBB14 + luồng độ phân giải gốc + lưới P2 + bộ tinh chỉnh box đọc nét ở độ phân giải gốc (junction 18 px thay vì 9 px). `imgsz` = cỡ lõi (1280), ảnh vào 2560 | x: 71.1M (`neck=dfine`) | [OPENOBB15.md](OPENOBB15.md) |
| `openobb20s/m/l/x` | **mới, chưa train:** OpenOBB19 + bộ tinh chỉnh cạnh đọc thân stride 4 (zero-init). Với `teacher=` (từ 0.19.0, mọi v18+): chưng cất thêm phân phối box và xác suất class của thầy (`w_ld`, `w_kd_cls`) | x: khoảng 51M | [OPENOBB20.md](OPENOBB20.md) |
| `openobb21s/m/l/x` | **mới, chưa train:** OpenOBB20 + nhánh sửa góc (zero-init, bắt đầu đúng bằng v20). Từ 0.20.0 mọi model dense train với loss góc vectơ + lớp ra fp32 (`angle_loss=cos` = cách cũ) | x: khoảng 51M | [OPENOBB21.md](OPENOBB21.md) |
| `openobb22s/m/l/x` | **mới, chưa train:** OpenOBB21 + góc trong (−45°, 135°) khởi tạo 45° + hồi quy box thẳng (bỏ DFL); fine-tune từ v20 / v21 dựng lại hàng box (cần ~150 epoch) | x: khoảng 51M | [OPENOBB22.md](OPENOBB22.md) |
| `openobb23s/m/l/x` | **mới, chưa train:** OpenOBB21 + bám ranh giới có xoay (zero-init, bắt đầu đúng bằng v21); fine-tune từ best.pt v21 | x: khoảng 51M | [OPENOBB23.md](OPENOBB23.md) |
| `openobb19s/m/l/x` | **mới, chưa train:** OpenOBB18 + head theo nhóm tranh chấp: mỗi điểm lưới đoán một box cho mỗi nhóm class (nhóm tự chia từ nhãn train, `n_groups=3`), nên object chồng nhau ở đầu (mẩu tường trong box junction) không còn giành điểm. `groups=auto` dùng được cho v17 / v18 | x: khoảng 50.6M | [OPENOBB19.md](OPENOBB19.md) |
| `openobb18s/m/l/x` | **mới, chưa train, nhắm tốc độ + độ chính xác:** OpenOBB17 với encoder lai + LSK thay bằng neck RepConv tự viết; neck được chưng cất từ encoder của một v12 / v17 đã train (`teacher=<.pt>`), thân + head nạp từ `weights=` cùng file. −24% FLOPs | x: 50.6M | [OPENOBB18.md](OPENOBB18.md) |
| `openobb17s/m/l/x` | **mới, chưa train, nhắm mAP50-95:** OpenOBB12 + trường ranh giới stride 2 (đầu / cạnh, giám sát bằng mọi cạnh nhãn) + bám ranh giới: đầu và cạnh của top-1500 box dời về đỉnh của trường bằng soft-argmax 1-D trước NMS. Bắt đầu đúng bằng v12; `v7_off=field` tắt phần mới | x: 68.1M | [OPENOBB17.md](OPENOBB17.md) |
| `openobb16s/m/l/x` | **mới, chưa train, nhắm mAP50-95:** OpenOBB15 + cascade 3 bước tinh chỉnh box theo cạnh (0.5 / 0.6 / 0.7) + điểm class × độ khít dự đoán trước NMS. Bắt đầu đúng bằng v15 | x: 72.3M (`neck=dfine`) | [OPENOBB16.md](OPENOBB16.md) |
| `openobb10s/m/l/x` | **mới, chưa train:** OpenOBB8 + head 3 slot mỗi điểm (vật nhỏ chồng nhau không còn tranh điểm: 72% wall_300 không có điểm riêng ở v8), gán nhãn theo tầm với DFL (note không còn học box cụt), cân bằng class hiếm. Đi tiếp được từ `best.pt` của v8 (`weights=`) | x: 66.7M | [OPENOBB10.md](OPENOBB10.md) |
| `openobb9s/m/l/x` | **mới, chưa train:** lõi tự thiết kế (conv 3×3 dày tái tham số hoá, nhắm nhanh gấp khoảng 1.8 lần v8) + head của v8; cần `openobb pretrain` (chưng cất từ D-FINE COCO + DINOv2) trước | x: 29.5M | [OPENOBB9.md](OPENOBB9.md), [PRETRAIN.md](PRETRAIN.md) |
| `openobb8s/m/l/x` | **bản tốt nhất hiện tại:** lõi pretrained của v3 đã sửa các lỗi đo được (chặn nổ số P5, LSK có chặn, chi tiết stride 4, head class có ngữ cảnh) + tầng suy luận quan hệ theo hướng. Bắt đầu y hệt v3 | x: 66.6M | [OPENOBB8.md](OPENOBB8.md) |
| `openobb6n/s/m/l/x` | thử nghiệm: kiến trúc tự thiết kế, không pretrain. Wall_Color chỉ đạt khoảng 0.55 / 0.32 (v3: 0.72 / 0.47), không dùng cho sản phẩm | x: 37.4M | [OPENOBB6.md](OPENOBB6.md) |
| `openobb5x` | head dense + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ | 127.9M | [OPENOBB5.md](OPENOBB5.md) |
| `openobb4x` | DINOv2 ViT-B + encoder/decoder DETR + nhánh dense, đầu ra decoder ∪ dense | 134.8M | [OPENOBB4.md](OPENOBB4.md) |
| `openobb3s/m/l/x` | encoder pretrained COCO + head dense oriented dị hướng, NMS | x: 64.6M | Tốt nhất đã đo trên Wall_Color, nhanh. [OPENOBB3.md](OPENOBB3.md) |
| `openobb2n/s/m/l/x` | dense segment-aware, train từ đầu, không NMS | x: 55.6M | Đã dừng phát triển. [OPENOBB2.md](OPENOBB2.md) |
| `openobb1s/m/l/x` | D-FINE-OBB (DETR) | s: 12.5M, x: 71M | [OPENOBB1.md](OPENOBB1.md) |
| đường dẫn `best.pt` | **fine-tune**: kiến trúc lấy theo checkpoint, class trùng tên giữ lại trọng số | | |

Có thể khởi tạo `openobb3x` từ một checkpoint `openobb1` bằng `weights=<best.pt>`.

**Mặc định, đổi bằng `key=value`:**

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `imgsz` | 1024 | cạnh dài ảnh đưa vào model (trang lớn nên 1280) |
| `batch` | tự chọn (`-1`) | theo VRAM và cỡ dataset (≥ 25 bước/epoch); `0.6` = dùng 60% VRAM; hết VRAM ở epoch đầu thì tự giảm một nửa |
| `mosaic` | 1.0 | ghép 4 ảnh, tắt ở `close_mosaic`=10 epoch cuối |
| `scale` | 0.5 | zoom ngẫu nhiên 0.5–1.5 |
| `translate` | 0.1 | dịch ngẫu nhiên ±10% |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.015, 0.5, 0.3 | đổi màu |
| `warmup_epochs` | 3 | khởi động learning rate, như YOLO (từ 0.16.0): `max(round(warmup_epochs × số batch mỗi epoch), 100)` iteration; mọi nhóm LR tăng từ 0 lên `lr0 × lf(epoch)` |
| `warmup_bias_lr` | 0 | LR ban đầu của nhóm bias trong warmup. Để 0 với AdamW (YOLO cũng đặt 0 khi tự chọn AdamW); 0.1 chỉ hợp với SGD |
| `nbs` | 64 | batch danh nghĩa như YOLO: cộng dồn gradient `round(nbs / batch)` batch mỗi bước (tăng dần từ 1 trong warmup), weight decay nhân `batch × accumulate / nbs`; `nbs=0` = một bước mỗi batch (công thức cũ). Với `optimizer=auto`, data nhỏ (< 2000 bước tối ưu) tự chuyển `nbs=0` |
| `epoch_lr` | True | hệ số LR `lf(epoch)` đổi theo từng epoch như YOLO; `False` = cosine theo iteration (công thức trước 0.16.0) |
| `iou` / `nms_iou` | `auto` | lọc trùng: `auto` = train xong tự thử trên val và lưu ngưỡng tốt nhất (v11 thử cả không NMS) vào `best.pt`, làm mặc định khi predict; hoặc một số cố định / `none`. Với v12 / v13, bước này chọn luôn head dùng khi suy luận (một-một không NMS, hoặc một-nhiều + NMS). Checkpoint đã train: `openobb calibrate model=best.pt data=data.yaml` |
| `box` | `dfl` | v12 / v13: cách hồi quy box. `direct` = hồi quy trực tiếp độ dài, độ dày, lệch ngang (kiểu YOLO26, không DFL); box khít hơn với vật mảnh (F54) |
| `pretrain` | không | chỉ `openobb9*` / `openobb6*`: `coco`, `dota` hoặc thư mục ảnh. Tự tải data, pretrain lõi (và train trên DOTA với `dota`), lưu cache rồi mới train. Xem [PRETRAIN.md](PRETRAIN.md) mục 0 |
| `lrf` | 0.01 | learning rate giảm theo **cosine** (mặc định từ 0.7.1) từ `lr0` về `lr0 × lrf` ở cuối run (`cos_lr=False`: giảm tuyến tính như mặc định của YOLO) |
| `cache` | auto | giải mã ảnh train một lần vào RAM nếu chiếm dưới 25% RAM (`cache=False` để tắt) |
| `patience` | 100 | dừng nếu mAP50-95 val không tăng sau N epoch |
| `optimizer` | auto | AdamW. `auto` (từ 0.22.5, model dense v2+) tự chọn khi bạn không truyền: `nbs=0` nếu data nhỏ; `lr0 = max(mặc định, 0.002 × 5 / (4 + số class))` khi data có class mà weights khởi đầu chưa biết (không có weights, hoặc checkpoint của class khác), như YOLO chọn LR cho AdamW. Train tiếp cùng class: LR mặc định. Log in dòng `[openobb] auto: ...`. `optimizer=AdamW` để tắt |
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
- **Gõ sai tên tham số:** báo lỗi kèm gợi ý, ví dụ `'epoch' is not a valid OpenOBB argument. Similar: epochs`.

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
- Đầu ra giống `yolo obb predict`, trong `runs/obb/predict`, `predict2`...:
  - `<tên gốc>` (ví dụ `page.png`): ảnh có vẽ box (`save=True`, mặc định của CLI);
  - `labels/<tên>.txt` khi `save_txt=True`: `class x1 y1 ... x4 y4` chuẩn hoá, thêm `conf` khi `save_conf=True`;
  - `crops/<class>/<tên>.jpg` khi `save_crop=True`.
- Dùng trong Python: `OpenOBB("best.pt").predict("page.png", conf=0.3)`, chi tiết ở [DEPLOY.md](DEPLOY.md).

## 4. Test nhanh bằng giao diện kéo thả (Streamlit)

Trên máy này đã có sẵn môi trường `.venv` (torch CPU, streamlit, OpenOBB): **bấm đúp `run_app.bat`** ở thư mục repo.

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
