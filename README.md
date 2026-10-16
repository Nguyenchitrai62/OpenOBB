# OpenOBB

Phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD: tường, cửa, nút giao, ghi chú... Train, predict và đọc kết quả giống YOLO-OBB.
Không phụ thuộc Ultralytics (AGPL): chỉ dùng thành phần Apache-2.0, BSD hoặc MIT. Kiến trúc mới nhất: **OpenOBB23**.

## Kết quả

Wall_Color (250 ảnh train, 53 ảnh val, 7 class), ảnh cạnh dài 1280, mAP chấm theo cách của Ultralytics.
Tốc độ đo cùng một GPU (RTX PRO 6000), batch 16, trang 960 × 1280, ms / ảnh (mạng + NMS).

| Model | Tham số | mAP50 | mAP50-95 | Box trùng | Tốc độ | Tốc độ + `compile` |
|---|---|---|---|---|---|---|
| **OpenOBB23-x** | 51.3M | **0.800** | **0.512** | 1.9% | 9.1 ms | 8.0 ms |
| YOLO26x-obb | 57.6M | 0.743 | 0.499 | 1.6% | 7.3 ms | — |

YOLO26x-obb train 150 epoch từ trọng số chính thức; OpenOBB23-x fine-tune từ các checkpoint OpenOBB trước trên cùng data.

mAP50-95 theo class (OpenOBB23-x / YOLO26x): wall .576 / .624, junction .545 / .584, wall_300 .331 / .441, note .494 / .517,
door .811 / .811, slide_door .317 / .123, double_door .505 / .392.

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
- Tăng cường dữ liệu như YOLO (mosaic, zoom, translate, lật, xoay 90°); thêm `rotate_p=0.5` để xoay góc bất kỳ khi bản vẽ có đối tượng nghiêng.
- Cuối train, head và NMS được hiệu chỉnh trên val và lưu vào `best.pt`.
- Notebook Colab: [colab/OpenOBB_wall_color.ipynb](colab/OpenOBB_wall_color.ipynb). Hướng dẫn chi tiết: [docs/FINETUNE.md](docs/FINETUNE.md).

## Suy luận

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
for r in model.predict("page.png", conf=0.25):
    print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)
```

```bash
openobb predict model=best.pt source=pages/ conf=0.25
```

`best.pt` dùng thẳng được, không cần export: khi nạp, model tự gộp Conv + BatchNorm, chạy bf16 và `channels_last` trên GPU.
Đối chiếu API với Ultralytics: [docs/DEPLOY.md](docs/DEPLOY.md). Giao diện kéo-thả để thử model: `run_app.bat` (Streamlit).

**Nhanh hơn với `compile`** (GPU, tiến trình chạy lâu như server): đặt biến môi trường **trước** khi nạp model. Lần gọi đầu mất 1–3 phút
biên dịch, các lần sau nhanh hơn khoảng 12%.

```python
import os
os.environ["OPENOBB_COMPILE"] = "1"
from openobb import OpenOBB
model = OpenOBB("best.pt")
```

```bash
OPENOBB_COMPILE=1 openobb predict model=best.pt source=pages/
```

- File giao đi gọn hơn (fp16, bỏ tham số train; tốc độ không đổi): `openobb strip model=best.pt half=True`.
- Đo thời gian từng khối trên GPU: `openobb profile model=best.pt detail=True`.

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
