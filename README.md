# OpenOBB

Phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD: tường, cửa, nút giao, ghi chú... Train, predict và đọc kết quả giống YOLO-OBB.
Không phụ thuộc Ultralytics (AGPL): chỉ dùng thành phần Apache-2.0, BSD hoặc MIT. Kiến trúc mới nhất: **OpenOBB23**.

## Kết quả

Wall_Color (250 ảnh train, 53 ảnh val, 7 class), ảnh cạnh dài 1280, mAP chấm theo cách của Ultralytics.
Tốc độ đo cùng một GPU (RTX PRO 6000), batch 16, trang 960 × 1280, ms / ảnh (mạng + NMS).

| Model | Tham số | mAP50 | mAP50-95 | Box trùng | Tốc độ | Tốc độ + `compile` |
|---|---|---|---|---|---|---|
| **OpenOBB23-x** | 51.3M | **0.798** | **0.526** | 1.8% | 9.1 ms | 8.0 ms |
| YOLO26x-obb | 57.6M | 0.743 | 0.499 | 1.6% | 7.3 ms | — |

YOLO26x-obb train 150 epoch từ trọng số chính thức; OpenOBB23-x fine-tune từ các checkpoint OpenOBB trước trên cùng data
(lần cuối: 30 epoch với khung training YOLO, thân đóng băng).

mAP50-95 theo class (OpenOBB23-x / YOLO26x): wall .599 / .624, junction .542 / .584, wall_300 .362 / .441, note .532 / .517,
door .819 / .811, slide_door .265 / .123, double_door .564 / .392.

## Cài đặt

**Máy có GPU NVIDIA:**

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu126
pip install -U "openobb @ git+https://github.com/Nguyenchitrai62/OpenOBB.git"
```

**Máy chỉ có CPU:**

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
pip install -U "openobb @ git+https://github.com/Nguyenchitrai62/OpenOBB.git"
```

- GPU RTX 50xx: đổi `cu126` thành `cu130`. Chạy lại đúng lệnh để cập nhật.
- Model tự chạy trên GPU nếu có. Nếu máy có GPU NVIDIA mà PyTorch là bản CPU, lần dùng đầu sẽ tự cài bản GPU rồi báo chạy lại
  (tắt bằng `OPENOBB_AUTO_TORCH=0`).

## Train

Dataset định dạng YOLO-OBB (có `data.yaml`). Fine-tune từ một checkpoint OpenOBB23 đã train:

```bash
openobb train model=openobb23x weights=openobb23x.pt data=data.yaml epochs=100 imgsz=1280 batch=16   # -> runs/obb/train/weights/best.pt
openobb val model=runs/obb/train/weights/best.pt data=data.yaml
```

- Cỡ model: `openobb23s`, `openobb23m`, `openobb23l`, `openobb23x`.
- Khung training giống hệt YOLO (Ultralytics): AdamW với LR theo số class, warmup 3 epoch, EMA, nbs 64, mosaic / zoom / translate / lật ngang / HSV. Chỉ kiến trúc model là của OpenOBB; khác YOLO một điểm: LR giảm theo cosine (`cos_lr=False` để giảm tuyến tính như YOLO). Thêm `rotate_p=0.5` để xoay góc bất kỳ khi bản vẽ có đối tượng nghiêng.
- `unlock_all=True`: không đóng băng phần nào của model (thân pretrained). Nên thử khi có data lớn.
- Cuối train, head và NMS được hiệu chỉnh trên val và lưu vào `best.pt`.
- Notebook Colab: [colab/OpenOBB_wall_color.ipynb](colab/OpenOBB_wall_color.ipynb). Hướng dẫn chi tiết: [docs/FINETUNE.md](docs/FINETUNE.md).

## Suy luận

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
for r in model.predict("page.png", conf=0.3):
    print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)
```

```bash
openobb predict model=best.pt source=pages/ conf=0.3
```

`best.pt` dùng thẳng được, không cần export: khi nạp, model tự gộp Conv + BatchNorm, chạy bf16 và `channels_last` trên GPU.
Đối chiếu API với Ultralytics: [docs/DEPLOY.md](docs/DEPLOY.md). Giao diện kéo-thả để thử model: `run_app.bat` (Streamlit).

**Nhanh hơn với `compile`** (như `compile=` của YOLO; mặc định tắt): hợp với tiến trình chạy lâu như server. Lần gọi đầu mất 1–3 phút
biên dịch, các lần sau nhanh hơn khoảng 12% (G4). Nhận `True` hoặc tên mode của `torch.compile` (`reduce-overhead`, `max-autotune`...).
Không biên dịch được thì báo cảnh báo và chạy bản thường.

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
model.predict("pages/", compile=True)
```

```bash
openobb predict model=best.pt source=pages/ compile=True
openobb val model=best.pt data=data.yaml compile=True
```

Biến môi trường `OPENOBB_COMPILE=1` vẫn dùng được, có tác dụng khi không truyền `compile=`.

- File giao đi gọn hơn (fp16, bỏ tham số train; tốc độ không đổi): `openobb strip model=best.pt half=True`.
- Đo thời gian từng khối trên GPU: `openobb profile model=best.pt detail=True`.
- Đo thời gian một bước train (forward / loss / backward / optimizer): `openobb profile model=best.pt train=True imgsz=960 batch=32`.
- Train v18–v24 bật `compile` mặc định (epoch đầu lâu hơn để biên dịch, các epoch sau nhanh hơn khoảng 25%); `compile=False` để tắt.

## Export ONNX và infer ONNX

File `.onnx` chứa toàn bộ mạng tới trước NMS, cùng các thiết lập đã hiệu chỉnh lúc train (tên class, imgsz, NMS, ngưỡng conf theo class).
Tên layer, module và đường dẫn code bị xoá. Infer bằng `openobb_onnx`: dùng ONNX Runtime + numpy, không cần PyTorch, kết quả giống
`openobb predict` trên `.pt`. Chi tiết và số đo: [docs/ONNX.md](docs/ONNX.md).

**Export** (model OpenOBB17 – 24; nên dùng OpenOBB23):

```bash
pip install onnx onnxscript onnxruntime            # hoặc pip install "openobb[export]"
openobb export model=best.pt                       # -> best.onnx cạnh best.pt
openobb export model=best.pt out=giao/best.onnx imgsz=1280
```

```python
from openobb import OpenOBB
OpenOBB("best.pt").export(format="onnx")           # -> best.onnx
```

**Infer** (GPU NVIDIA: cài `onnxruntime-gpu[cuda,cudnn]` thay cho `onnxruntime`, driver hỗ trợ CUDA 13; không cài chung hai bản):

```bash
openobb-onnx info model=best.onnx                                   # class, imgsz, NMS, thiết bị (cpu / cuda)
openobb-onnx predict model=best.onnx source=pages/ conf=auto        # ảnh có box -> runs/obb/predict, predict2...
openobb-onnx predict model=best.onnx source=page.png save_txt=True save_conf=True device=cpu classes=wall,door
```

```python
from openobb_onnx import OpenOBB
model = OpenOBB("best.onnx")                       # device=None: GPU nếu có, không thì CPU; "cpu", 0, "cuda:1", "trt"
for r in model.predict("page.png", conf=0.3):      # ảnh, thư mục, glob, list, ảnh numpy (BGR) hoặc PIL
    r.obb.xywhr, r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names
    r.plot(); r.save_txt("page.txt"); r.to_json()
```

Tham số giống `predict` của bản `.pt`: `conf` (số, hoặc `"auto"` = ngưỡng theo class lưu lúc train), `iou` (mặc định là NMS đã hiệu
chỉnh; `"none"` = không NMS), `imgsz`, `max_det`, `classes`, `device`, `save`, `save_txt`, `save_conf`. App kéo thả ảnh:
`run_app.bat` nhận cả `.pt` lẫn `.onnx` ở dòng `MODEL=`.

## Các kiến trúc

Kết quả tốt nhất của mỗi kiến trúc trên Wall_Color val (cỡ x, ảnh 1280, thước Ultralytics), xếp theo mAP50-95.

| `model=` | Kiến trúc | Tham số | mAP50 | mAP50-95 | Tài liệu |
|---|---|---|---|---|---|
| **`openobb23x`** | v21 + bám ranh giới có xoay (dịch và xoay box theo mép) | 51.3M | **0.798** | **0.526** | [OPENOBB23](docs/OPENOBB23.md) |
| `openobb21x` | v20 + sửa góc trong không gian góc kép | 51.3M | 0.769 | 0.492 | [OPENOBB21](docs/OPENOBB21.md) |
| `openobb20x` | v19 + bộ tinh chỉnh cạnh đọc chi tiết stride 4 của thân | 51.3M | 0.764 | 0.482 | [OPENOBB20](docs/OPENOBB20.md) |
| `openobb17x` | v12 + trường ranh giới + bám ranh giới, nhóm class | 68.3M | 0.751 | 0.481 | [OPENOBB17](docs/OPENOBB17.md) |
| `openobb22x` | v21 + góc sigmoid + hồi quy box trực tiếp | 51.1M | 0.760 | 0.479 | [OPENOBB22](docs/OPENOBB22.md) |
| `openobb12x` | lõi encoder lai + head dense kép (một-nhiều, một-một) | 67.1M | 0.743 | 0.472 | [OPENOBB12](docs/OPENOBB12.md) |
| `openobb14x` | neck tái tham số hoá + head tự viết, hồi quy box trực tiếp | 62.7M | 0.713 | 0.469 | [OPENOBB14](docs/OPENOBB14.md) |
| `openobb3x` | encoder COCO + head dense oriented | 64.6M | 0.712 | 0.467 | [OPENOBB3](docs/OPENOBB3.md) |
| `openobb19x` | neck nhanh tự viết + một box mỗi nhóm class | 50.7M | 0.755 | 0.462 | [OPENOBB19](docs/OPENOBB19.md) |
| `openobb4x` | DINOv2 + DETR + nhánh dense | 134.8M | 0.663 | 0.457 | [OPENOBB4](docs/OPENOBB4.md) |
| YOLO26x-obb (tham chiếu) | Ultralytics | 57.6M | 0.743 | 0.499 | |

Số epoch và điểm khởi tạo khác nhau giữa các lần train. Cỡ s / m / l có cho mọi kiến trúc trên (`openobb23s`...). Các kiến trúc thử nghiệm
khác (v1, v2, v5–v11, v13, v15, v16, v18) chưa có kết quả đủ điều kiện trên Wall_Color; tài liệu ở `docs/`. Mới, chưa train:
`openobb24x` (v23 + độ khít học được, dùng để chọn box trong NMS, [OPENOBB24](docs/OPENOBB24.md)).

## Kiến trúc OpenOBB23

Thân HGNetv2 pretrained (Apache-2.0) → neck tái tham số hoá tự viết → head dense kép (một-nhiều để học, một-một), mỗi điểm lưới
một box cho mỗi nhóm class → bộ tinh chỉnh cạnh đọc chi tiết stride 4 → bám ranh giới (dịch và xoay box theo mép trên trường ranh giới
stride 2) → NMS hiệu chỉnh. Chi tiết: [docs/OPENOBB23.md](docs/OPENOBB23.md).

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `openobb/` | Thư viện: models, train, predict, eval, CLI |
| `tests/` | `pytest -q tests/` (chạy CPU) |
| `colab/` | Notebook Colab, script dữ liệu |
| `docs/` | Tài liệu kiến trúc, deploy, fine-tune |
| `research/` | Sổ cái nghiên cứu và script đo |
| `tools/` | Script tự động, đồng bộ repo |
| `app/` | App Streamlit để thử model |

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
