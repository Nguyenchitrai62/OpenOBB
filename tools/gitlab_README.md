# OpenOBB

Thư viện phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD và ảnh chụp từ trên cao. Các kiến trúc **OpenOBB** là tài sản
riêng, dùng thương mại được: chỉ dùng thành phần Apache-2.0 / BSD / MIT, không phụ thuộc Ultralytics (AGPL).
Giao diện predict / kết quả giống YOLO-OBB.

## Cài đặt

**Máy có GPU NVIDIA:**

```bash
pip install -U torch "openobb @ git+https://git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

**Máy chỉ có CPU:**

```bash
pip install -U torch "openobb @ git+https://git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

**Cài từ file wheel** (không cần quyền truy cập repo; file mới nhất nằm trong thư mục `wheels/` của repo sản phẩm,
hoặc là file được bàn giao). Đặt file `.whl` vào thư mục đang đứng rồi chạy; `X.Y.Z` là phiên bản ghi trong tên file.

Máy có GPU NVIDIA:

```bash
pip install -U torch ./openobb-X.Y.Z-py3-none-any.whl --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

Máy chỉ có CPU:

```bash
pip install -U torch ./openobb-X.Y.Z-py3-none-any.whl --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

- Luôn cài bản mới nhất; chạy lại đúng lệnh để cập nhật. GPU RTX 50xx: đổi `cu126` thành `cu130`.
- Khi suy luận, model tự chạy trên GPU nếu có. Nếu máy có GPU NVIDIA mà PyTorch là bản CPU, lần dùng đầu tiên sẽ tự cài
  bản GPU rồi báo chạy lại (tắt bằng biến môi trường `OPENOBB_AUTO_TORCH=0`).
- Kiểm tra phiên bản: `python -c "import openobb; print(openobb.__version__)"`.

## Suy luận (thay YOLO)

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
for r in model.predict("page.png", conf=0.25):
    print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)
```

Bảng đối chiếu đầy đủ với Ultralytics: [docs/DEPLOY.md](docs/DEPLOY.md).

## Train / fine-tune

```bash
openobb train data=data.yaml model=openobb5x epochs=100 imgsz=1280      # runs/obb/train, weights/best.pt
openobb val model=runs/obb/train/weights/best.pt data=data.yaml
openobb predict model=runs/obb/train/weights/best.pt source=pages/ conf=0.3
```

`data.yaml` theo định dạng YOLO-OBB. mAP in ra theo cách chấm của Ultralytics để so trực tiếp với YOLO; kèm thêm thước
DOTA và các lỗi nhầm class thường gặp. Hướng dẫn: [docs/FINETUNE.md](docs/FINETUNE.md).

## Kiến trúc

| `model=` | Mô tả | Tài liệu |
|---|---|---|
| `openobb11s/m/l/x` | mới: đề xuất dense + tầng quan hệ train một-một (khử trùng tự học, không NMS); nạp được checkpoint v8 / v10 | [docs/OPENOBB11.md](docs/OPENOBB11.md) |
| `openobb12s/m/l/x` | mới: lõi OpenOBB7 + head dense kép (một-nhiều để học, một-một để suy luận), không NMS, không tầng quan hệ | [docs/OPENOBB12.md](docs/OPENOBB12.md) |
| `openobb13s/m/l/x` | mới, nhắm độ chính xác: v12 + luồng độ phân giải gốc + lưới P2 cho vật tí hon (`imgsz` = độ phân giải đầy đủ, ví dụ 2560; lõi chạy ở một nửa) | [docs/OPENOBB13.md](docs/OPENOBB13.md) |
| `openobb10s/m/l/x` | mới, chưa train: v8 + head 3 slot / điểm (vật nhỏ chồng nhau như wall_300 cạnh junction), gán nhãn theo tầm với (box lớn), cân bằng class hiếm; nạp được checkpoint v8 | [docs/OPENOBB10.md](docs/OPENOBB10.md) |
| `openobb9s/m/l/x` | mới, chưa train: lõi tự thiết kế (conv 3×3 dày tái tham số hoá) + head v8; pretrain bằng chưng cất 2 thầy | [docs/OPENOBB9.md](docs/OPENOBB9.md) |
| `openobb8s/m/l/x` | bản tốt nhất: v3 sửa lỗi + tầng suy luận quan hệ theo hướng (bắt đầu y hệt v3) | [docs/OPENOBB8.md](docs/OPENOBB8.md) |
| `openobb6n/s/m/l/x` | kiến trúc tự thiết kế từ phân tích (backbone + neck + head riêng), pretrain bằng chưng cất DINOv2 (`openobb pretrain`) | [docs/OPENOBB6.md](docs/OPENOBB6.md) |
| `openobb5x` | head dense + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ | [docs/OPENOBB5.md](docs/OPENOBB5.md) |
| `openobb4x` | DINOv2 + DETR + nhánh dense | [docs/OPENOBB4.md](docs/OPENOBB4.md) |
| `openobb3x` | encoder COCO + head dense oriented dị hướng (nhanh) | [docs/OPENOBB3.md](docs/OPENOBB3.md) |
| `openobb1x` | D-FINE-OBB (DETR) | [docs/OPENOBB1.md](docs/OPENOBB1.md) |

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Weights khởi tạo được tải lúc train (D-FINE COCO, DINOv2; Apache-2.0); repo không chứa weights.

## Test

```bash
pip install -e ".[dev]"
pytest -q tests/
```
