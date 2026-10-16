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

## Cài đặt

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

**Cài từ file wheel** (không cần quyền truy cập repo; file mới nhất nằm trong thư mục `wheels/` của repo sản phẩm,
hoặc là file được bàn giao). Đặt file `.whl` vào thư mục đang đứng rồi chạy; `X.Y.Z` là phiên bản ghi trong tên file.

Máy có GPU NVIDIA:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu126
pip install -U ./openobb-X.Y.Z-py3-none-any.whl
```

Máy chỉ có CPU:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
pip install -U ./openobb-X.Y.Z-py3-none-any.whl
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

**Nhanh hơn với `compile`** (GPU, tiến trình chạy lâu như server): đặt biến môi trường **trước** khi nạp model. Lần gọi đầu mất 1–3 phút
biên dịch, các lần sau nhanh hơn khoảng 12%.

```python
import os
os.environ["OPENOBB_COMPILE"] = "1"
from openobb import OpenOBB
model = OpenOBB("best.pt")
```

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
