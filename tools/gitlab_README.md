# OpenOBB

Thư viện phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD và ảnh chụp từ trên cao. Các kiến trúc **OpenOBB** (v1–v5) là tài sản
riêng, dùng thương mại được: chỉ dùng thành phần Apache-2.0 / BSD / MIT, không phụ thuộc Ultralytics (AGPL).
Giao diện train / predict / kết quả giống YOLO-OBB.

## Cài đặt

```bash
pip install "openobb @ git+http://git.anybim.vn/CxDP/service/hicasai/openanything/openobb.git"           # mới nhất
pip install "openobb @ git+http://git.anybim.vn/CxDP/service/hicasai/openanything/openobb.git@v0.2.0"    # đúng một phiên bản
```

Phiên bản: `openobb.__version__`; mỗi phiên bản có tag `vX.Y.Z`.

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
