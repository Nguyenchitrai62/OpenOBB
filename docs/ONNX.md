# Bàn giao model bằng ONNX

Bên nhận chỉ cần hai thứ: file `best.onnx` và wheel `openobb-X.Y.Z-py3-none-any.whl` (khoảng 30 KB). Wheel chỉ chứa ONNX
Runtime + numpy / OpenCV / shapely: **không có PyTorch, không có code kiến trúc, không có code train**. Kết quả giống `openobb predict`
trên `best.pt`.

**Code deploy không phải sửa.** Wheel cài vào đúng tên gói của gói đầy đủ (`openobb`), cùng class `OpenOBB`, cùng lệnh `openobb`,
cùng tham số `predict` và cùng giao diện kết quả (mảng có `.cpu()`, `.numpy()`, `.to()` như tensor). Code infer viết cho `.pt` chạy
nguyên văn, chỉ đổi đường dẫn model sang `.onnx`. Gói đầy đủ cũng nhận `.onnx` (`OpenOBB("best.onnx")`, `openobb predict model=best.onnx`),
nên máy dev test cả hai loại bằng cùng một đoạn code.

## 1. Bên giao: export

```bash
pip install "openobb[export]"          # onnx + onnxscript + onnxruntime, cài một lần
openobb export model=best.pt           # -> best.onnx cạnh best.pt; out= để đổi tên / thư mục
```

Python: `OpenOBB("best.pt").export(format="onnx")`. Export được các model dense từ OpenOBB17 tới OpenOBB24; bản mặc định để
giao là **OpenOBB23-x**.

File ONNX chứa toàn bộ mạng tới trước NMS: thân, neck, head, bộ tinh chỉnh cạnh, bám ranh giới, và các thiết lập đã hiệu chỉnh
lúc train (head suy luận, NMS, lọc đỉnh). NMS và gộp box chạy bằng numpy trong wheel, theo đúng luật của bản PyTorch. Kích thước
ảnh và batch là động: ảnh chữ nhật cỡ nào cũng chạy được, cạnh là bội số của 32.

**File ONNX không còn gì?**
- Tên lớp, tên module, tên weights và đường dẫn file code: mọi node / tensor đổi thành `n12`, `w34`, `t56`; chú thích nguồn của
  exporter bị xoá.
- Cách train: loss, gán nhãn, lịch LR, đóng băng thân, augment, chưng cất, cách hiệu chỉnh.

**File ONNX vẫn còn gì?** Đồ thị phép tính và weights. Xem được bằng Netron, và người có kinh nghiệm có thể đổi ngược sang
PyTorch (ví dụ `onnx2torch`). Nhưng muốn fine-tune ra kết quả tốt thì vẫn phải tự làm lại cách gán nhãn, loss và lịch train.
Bảo vệ mạnh hơn nữa là cho model chạy trên server của mình (API) và ràng buộc bằng điều khoản license.

## 2. Bên nhận: cài đặt

Wheel bàn giao cùng tên với gói đầy đủ (`openobb`): trên máy đang cài wheel đầy đủ (bản cũ), `pip install -U` thay nó như một lần
nâng cấp, không phải gỡ tay. Nếu máy đã có đúng số phiên bản đó thì thêm `--force-reinstall`.
Chỉ cài **một** trong hai bản ONNX Runtime (GPU hoặc CPU); hai bản cài cùng lúc sẽ ghi đè lẫn nhau.

**Máy có GPU NVIDIA.** Cần driver hỗ trợ CUDA 13 (`nvidia-smi` hiện "CUDA Version: 13.x"). CUDA / cuDNN được kéo về qua pip
(`onnxruntime-gpu[cuda,cudnn]`), không phải cài CUDA riêng.

```bash
pip uninstall -y onnxruntime
pip install -U "./openobb-X.Y.Z-py3-none-any.whl[gpu,app]"
openobb info model=best.onnx           # phải in "device cuda:0"
```

Khi máy không dùng được GPU (driver cũ, thiếu DLL), model in cảnh báo và chạy CPU. Lúc đó sẽ chậm hơn `.pt` chạy GPU nhiều lần.

**Máy chỉ có CPU:**

```bash
pip uninstall -y onnxruntime-gpu
pip install -U "./openobb-X.Y.Z-py3-none-any.whl[cpu,app]"
```

Bỏ `,app` nếu không cần app Streamlit.

## 3. Dùng (giống hệt code cho `.pt`)

```bash
openobb predict model=best.onnx source=pages/ conf=auto          # -> runs/obb/predict, predict2...
openobb predict model=best.onnx source=page.png save_txt=True save_conf=True device=cpu
openobb app model=best.onnx                                      # app kéo thả ảnh (Streamlit)
```

```python
from openobb import OpenOBB
model = OpenOBB("best.onnx")                    # device=None: GPU nếu có; "cpu", 0, "cuda:1", "trt" (TensorRT)
for r in model.predict("page.png", conf=0.3):   # conf="auto": ngưỡng theo class đã lưu lúc train; iou=, classes=, stream=...
    r.obb.xywhr.cpu().numpy(), r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names
    r.plot(); r.save_txt("page.txt"); r.to_json()
```

File `.bat` mở app, đặt cạnh thư mục `.venv` (giống `run_app.bat` của OpenOBB; đổi model ở dòng `MODEL=`):

```bat
@echo off
set "MODEL=D:\models\best.onnx"
cd /d "%~dp0"
set "OPENOBB_MODEL=%MODEL%"
if not exist "%MODEL%" echo Canh bao: khong thay model %MODEL%
".venv\Scripts\python.exe" -m openobb.cli app
pause
```

## 3b. Thay YOLO trong service có SAHI (cắt slice)

Cài thêm SAHI (MIT): `pip install "<wheel>[gpu,sahi]"` (hoặc `[cpu,sahi]`). Module `openobb.sahi` cho `AutoDetectionModel` giống hệt của SAHI.
Gói đầy đủ (PyTorch) nhận cả `.pt` lẫn `.onnx`, như YOLO; wheel bàn giao chỉ nhận `.onnx` (không có PyTorch). Cùng một trang, qua SAHI,
`.pt` và `.onnx` cho cùng 29 box, đỉnh lệch tối đa 0.09 px.
Model trả về là một `UltralyticsDetectionModel` loại OBB của SAHI nhưng không cần cài ultralytics hay PyTorch, nên `get_sliced_prediction`
gốc của SAHI xử lý y như với YOLO OBB: gộp các slice bằng NMS, mỗi object giữ đúng 4 đỉnh trong `mask.segmentation`.

Một service đang chạy YOLO + SAHI (ví dụ `ai-torch-service`) chỉ đổi các dòng import và đường dẫn model:

| File của service | Trước | Sau |
|---|---|---|
| `services/load_model.py`, `services/model_tools.py`, `services/yolo_batch_runner.py` | `from ultralytics import YOLO` | `from openobb import OpenOBB as YOLO` |
| `services/load_model.py`, `services/model_tools.py` | `from sahi import AutoDetectionModel` | `from openobb.sahi import AutoDetectionModel` |
| `services/yolo_batch_runner.py` | `from ultralytics.engine.results import Results` | `from openobb import Results` (chỉ dùng làm type hint) |
| `core/config.py` | `.../wall_color.pt`, `.../wall_color_sahi.pt` | `.../wall_color.onnx`, `.../wall_color_sahi.onnx` (wheel bàn giao); gói đầy đủ giữ được `.pt` OpenOBB |

Phần còn lại giữ nguyên: `AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=..., confidence_threshold=0.1,
device=...)`, `get_sliced_prediction(...)`, `model.predict(list_images, conf=, iou=)`, `model.names`, `result.obb.xywhr / .xyxy /
.xyxyxyxy / .xyxyxyxyn / .data / .shape`, `result.boxes` (None), `prediction.mask.segmentation`.

- `from openobb.sahi import AutoDetectionModel` nhận ra model OpenOBB từ file, không nạp mạng: metadata của `.onnx` (vài ms), hoặc
  pickle bên trong `.pt` (checkpoint OpenOBB chỉ chứa dữ liệu, checkpoint YOLO chứa class của ultralytics). File khác chuyển nguyên cho
  SAHI, nên các route còn dùng YOLO vẫn chạy khi đổi import này cho cả service.
- `from openobb import OpenOBB as YOLO` chỉ nạp được model OpenOBB. Nếu service còn route chạy YOLO, giữ `ultralytics` cho các route
  đó và chọn theo file trong `yolo_loader`, ví dụ `OpenOBB(path) if openobb.sahi.is_model_file(path) else YOLO(path)`.
- Client gửi `iou=0.7` (mặc định của YOLO) thì model dùng NMS 0.7 thay cho ngưỡng đã hiệu chỉnh lúc train. Muốn dùng ngưỡng hiệu chỉnh:
  không truyền `iou`.
- SAHI chỉ có lợi khi model được train ở đúng tỉ lệ của slice. Model train trên cả trang (mặc định) thấy vật to hơn lúc train khi chạy
  trên slice 1024 px; số đo ở mục 4.

## 4. Đã kiểm chứng (2026-10-21, OpenOBB23-x `openobb23x_new.pt`, CPU)

| Kiểm tra | Kết quả |
|---|---|
| Đầu ra mạng ONNX so với PyTorch, 3 cỡ ảnh | điểm lệch ≤ 9e-6, box lệch ≤ 0.001 px |
| Detection cuối, 8 ảnh val Wall_Color × 2 ngưỡng conf | 1158 / 1158 box khớp; góc box lệch ≤ 0.14 px, điểm lệch ≤ 5e-5 (chỉ do bản PyTorch làm tròn đầu ra) |
| Tốc độ CPU (16 luồng, ảnh 3307×2338 → 928×1280) | ONNX 1.16 s, PyTorch 2.85 s |
| File ONNX | 185 MB (fp32), không còn tên module / package |
| App mở bằng `.bat`, cửa sổ console nằm nền | trước khi sửa 8.8 s / trang (Windows 11 EcoQoS dồn tiến trình nền sang lõi E); giờ model tự tắt việc này khi nạp: 1.1 s |

**SAHI (2026-10-26, cùng model, 53 trang val Wall_Color, CPU, ngưỡng conf 0.1 cho cả hai):**

| Cách chạy | mAP50 | mAP50-95 | P | R | wall_300 (mAP50 / 50-95) | Thời gian / trang |
|---|---|---|---|---|---|---|
| Cả trang (mặc định) | 0.802 | 0.537 | 0.774 | 0.799 | .777 / .370 | 1.3 s |
| SAHI: slice 1024, overlap 0.5, kèm lượt cả trang (như service) | 0.772 | 0.494 | 0.691 | 0.815 | .838 / .421 | 26 s |

SAHI tăng recall và class nét mảnh ngắn nhất (`wall_300`), nhưng các class khác tụt: model này train trên cả trang thu về 1280, trong
slice 1024 vật to hơn lúc train 1.6–3 lần, và SAHI lọc trùng giữa các slice bằng NMS trên box thẳng trục. Đường service với SAHI chạy
được trong venv chỉ có wheel bàn giao + SAHI (không torch, không ultralytics), mọi box đọc được bằng `polygon_to_xywhr` của service.
`tests/test_sahi.py`.

GPU chưa đo trên máy thật. Lệnh đo khi có GPU: chạy `openobb predict model=best.onnx ... device=0` hai lần (lần đầu còn khởi tạo) và so với
`openobb predict` trên `.pt`.

Test: `tests/test_onnx.py` (đầu ra mạng, NMS, end-to-end trên checkpoint thật nếu có). Build wheel:
`python tools/build_onnx_wheel.py` (-> `dist_onnx/`). `tools/push_all.py` build bản sản phẩm `hicasobb_onnx` vào `wheels/` của GitLab;
đó là wheel duy nhất ở đó. Wheel gói đầy đủ (PyTorch, kiến trúc, train) chỉ lưu ở `dist_hicasobb/` trên máy, không đẩy lên.
