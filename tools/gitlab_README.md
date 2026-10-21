# OpenOBB

Phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD: tường, cửa, nút giao, ghi chú... Các kiến trúc **OpenOBB** là tài sản riêng, dùng
thương mại được: chỉ dùng thành phần Apache-2.0 / BSD / MIT, không phụ thuộc Ultralytics (AGPL). Giao diện predict / kết quả giống YOLO-OBB.
Kiến trúc mới nhất: **OpenOBB23**.

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

## Bàn giao (ONNX): bảo mật, chỉ infer

Bên nhận chỉ nhận **hai file**: model `best.onnx` và wheel `openobb_onnx-X.Y.Z-py3-none-any.whl` (thư mục `wheels/`, khoảng 30 KB).
Wheel chỉ chứa code infer ONNX (ONNX Runtime + numpy / OpenCV / shapely): **không PyTorch, không code kiến trúc, không code train**.
File ONNX không còn tên layer / module / đường dẫn code. Kết quả giống hệt `openobb predict` trên `.pt`. Chi tiết: [docs/ONNX.md](docs/ONNX.md).

### Bước 1. Export (nội bộ, máy có gói đầy đủ và `best.pt`)

```bash
pip install onnx onnxscript                       # một lần
openobb export model=best.pt                      # -> best.onnx cạnh best.pt (model OpenOBB17 - 24; nên giao OpenOBB23)
openobb export model=best.pt out=giao/best.onnx      # đặt tên / thư mục file ra
```

```python
from openobb import OpenOBB
OpenOBB("best.pt").export(format="onnx")          # -> best.onnx
```

File ONNX mang theo mọi thiết lập đã hiệu chỉnh lúc train (tên class, imgsz, NMS, ngưỡng conf theo class): bên nhận không cần cấu hình gì.

### Bước 2. Cài (bên nhận) — chọn MỘT trong hai

**Máy có GPU NVIDIA** (driver hỗ trợ CUDA 13: `nvidia-smi` hiện "CUDA Version: 13.x"; CUDA / cuDNN tự cài qua pip, không cần cài CUDA riêng):

```bash
pip uninstall -y onnxruntime
pip install "./openobb_onnx-X.Y.Z-py3-none-any.whl[gpu,app]"
openobb-onnx info model=best.onnx                 # phải in "device cuda"
```

**Máy chỉ có CPU:**

```bash
pip uninstall -y onnxruntime-gpu
pip install "./openobb_onnx-X.Y.Z-py3-none-any.whl[cpu,app]"
openobb-onnx info model=best.onnx                 # in "device cpu"
```

Bỏ `,app` nếu không cần app Streamlit. Không cài chung `onnxruntime` và `onnxruntime-gpu`. Máy có GPU mà vẫn in "device cpu": xem cảnh
báo khi nạp model (thường do driver cũ).

### Bước 3. Infer

Dòng lệnh (giống `yolo obb predict`):

```bash
openobb-onnx predict model=best.onnx source=pages/                  # ảnh có box -> runs/obb/predict, predict2...
openobb-onnx predict model=best.onnx source=page.png conf=auto save_txt=True save_conf=True
openobb-onnx predict model=best.onnx source="pages/*.png" device=cpu classes=wall,door
```

Python:

```python
from openobb_onnx import OpenOBB
model = OpenOBB("best.onnx")                      # device=None: GPU nếu có, không thì CPU; "cpu", 0, "cuda:1"
for r in model.predict("page.png", conf=0.3):     # ảnh, thư mục, glob, list, ảnh numpy (BGR) hoặc PIL
    r.obb.xywhr        # (N, 5) cx, cy, w, h, góc (rad) theo pixel ảnh gốc
    r.obb.xyxyxyxy     # (N, 4, 2) bốn góc
    r.obb.conf, r.obb.cls, r.names
    r.plot()           # ảnh BGR có vẽ box;  r.save("out.png"), r.save_txt("out.txt"), r.to_json()
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `conf` | 0.25 | ngưỡng điểm; `"auto"` = ngưỡng tốt nhất theo từng class, lưu lúc train |
| `iou` | theo model | ngưỡng NMS (đã hiệu chỉnh lúc train); `"none"` = không NMS |
| `imgsz` | imgsz lúc train (lưu trong file) | cạnh dài ảnh đưa vào mạng; không cần truyền |
| `max_det` | 300 | số box tối đa mỗi ảnh |
| `classes` | tất cả | id hoặc tên class cần giữ |
| `device` | tự chọn | `cpu`, `0`, `cuda:1`, `trt` (TensorRT nếu có) |
| `save`, `save_txt`, `save_conf` | CLI: `save=True` | lưu ảnh có box / nhãn YOLO-OBB (tọa độ chuẩn hoá) / kèm điểm |

App kéo thả ảnh để thử model: `openobb-onnx app model=best.onnx`. Hoặc dùng file `run_app.bat` đặt cạnh thư mục `.venv`; muốn đổi model thì
sửa dòng `MODEL=`:

```bat
@echo off
set "MODEL=D:\models\best.onnx"
cd /d "%~dp0"
set "OPENOBB_MODEL=%MODEL%"
if not exist "%MODEL%" echo Canh bao: khong thay model %MODEL%
".venv\Scripts\python.exe" -m openobb_onnx.cli app
pause
```

## Cài đặt (nội bộ: train, val, export)

Gói đầy đủ (PyTorch, kiến trúc, train) chỉ dành cho nhóm phát triển, cài từ repo. Thư mục `wheels/` chỉ có wheel infer ONNX để bàn giao (mục trên).

**Máy có GPU NVIDIA:**

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu126
pip install -U "openobb @ git+https://git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git"
```

**Máy chỉ có CPU:**

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
pip install -U "openobb @ git+https://git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git"
```

- Luôn cài bản mới nhất; chạy lại đúng lệnh để cập nhật. GPU RTX 50xx: đổi `cu126` thành `cu130`.
- Khi suy luận, model tự chạy trên GPU nếu có. Nếu máy có GPU NVIDIA mà PyTorch là bản CPU, lần dùng đầu tiên sẽ tự cài
  bản GPU rồi báo chạy lại (tắt bằng biến môi trường `OPENOBB_AUTO_TORCH=0`).
- Kiểm tra phiên bản: `python -c "import openobb; print(openobb.__version__)"`.

## Suy luận (thay YOLO)

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
for r in model.predict("page.png", conf=0.3):
    print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)
```

`best.pt` dùng thẳng được, không cần export: khi nạp, model tự gộp Conv + BatchNorm, chạy bf16 và `channels_last` trên GPU.
Bảng đối chiếu đầy đủ với Ultralytics: [docs/DEPLOY.md](docs/DEPLOY.md).

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

## Train / fine-tune

```bash
openobb train model=openobb23x weights=openobb23x.pt data=data.yaml epochs=100 imgsz=1280 batch=16   # runs/obb/train/weights/best.pt
openobb val model=runs/obb/train/weights/best.pt data=data.yaml
openobb predict model=runs/obb/train/weights/best.pt source=pages/ conf=0.3
```

`data.yaml` theo định dạng YOLO-OBB. mAP in ra theo cách chấm của Ultralytics để so trực tiếp với YOLO; kèm box trùng, độ lệch hướng box
và các lỗi nhầm class thường gặp. Khung training giống hệt YOLO (AdamW với LR theo số class, warmup, LR giảm theo cosine, EMA, nbs 64,
augment); chỉ kiến trúc model là riêng. Thêm `rotate_p=0.5` để xoay ảnh góc bất kỳ khi bản vẽ có đối tượng nghiêng; `unlock_all=True`
để không đóng băng phần nào của model (nên thử khi có data lớn).
Hướng dẫn: [docs/FINETUNE.md](docs/FINETUNE.md).

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
`openobb24x` (v23 + độ khít học được, dùng để chọn box trong NMS, [OPENOBB24](docs/OPENOBB24.md)). Mỗi thẻ có lập luận
thiết kế, kỳ vọng ghi trước khi train, kết quả thật và phụ lục kiến trúc theo lớp.

## Kiến trúc OpenOBB23

Thân HGNetv2 pretrained (Apache-2.0) → neck tái tham số hoá tự viết → head dense kép (một-nhiều để học, một-một), mỗi điểm lưới
một box cho mỗi nhóm class → bộ tinh chỉnh cạnh đọc chi tiết stride 4 → bám ranh giới (dịch và xoay box theo mép trên trường ranh giới
stride 2) → NMS hiệu chỉnh. Chi tiết: [docs/OPENOBB23.md](docs/OPENOBB23.md).

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Weights khởi tạo được tải lúc train (D-FINE COCO, DINOv2; Apache-2.0); repo không chứa weights.

## Test

```bash
pip install -e ".[dev]"
pytest -q tests/
```
