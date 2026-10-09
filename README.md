# OpenOBB

Thư viện phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD, gồm các kiến trúc **OpenOBB** (v1–v5).
Train, predict và đọc kết quả giống YOLO-OBB. Không phụ thuộc Ultralytics (AGPL): chỉ dùng thành phần Apache-2.0, BSD hoặc MIT.

> Đây là repo **nghiên cứu và phát triển**: có code thư viện, notebook Colab, script thí nghiệm và sổ cái.
> Bản sản phẩm sạch (chỉ phần thương mại được) nằm ở GitLab công ty `openanything/openobb`.
> Hai repo được đồng bộ bằng `python tools/push_all.py`.

## Cài đặt

```bash
pip install "openobb @ git+https://github.com/Nguyenchitrai62/OpenOBB.git"           # bản mới nhất
pip install "openobb @ git+https://github.com/Nguyenchitrai62/OpenOBB.git@v0.2.0"    # đúng một phiên bản
```

Kiểm tra phiên bản đã cài: `python -c "import openobb; print(openobb.__version__)"`.
Mỗi lần code thư viện thay đổi, `tools/push_all.py` tự tăng phiên bản và gắn tag `vX.Y.Z` trên cả GitHub lẫn GitLab.

## Train trên Colab

Mở [colab/OpenOBB_wall_color.ipynb](colab/OpenOBB_wall_color.ipynb)
([chạy trên Colab](https://colab.research.google.com/github/Nguyenchitrai62/OpenOBB/blob/main/colab/OpenOBB_wall_color.ipynb)).
Notebook giải nén dataset YOLO-OBB (có `data.yaml`) rồi chạy:

```bash
openobb train model=openobb5x data=/content/data.yaml epochs=100 imgsz=1280 batch=16   # -> runs/obb/train/weights/best.pt
openobb val model=runs/obb/train/weights/best.pt data=/content/data.yaml
openobb predict model=runs/obb/train/weights/best.pt source=pages/ conf=0.3
```

- mAP được chấm theo cách của Ultralytics, để so trực tiếp với YOLO.
- Kèm thêm thước DOTA và danh sách lỗi nhầm class thường gặp.
- Hướng dẫn chi tiết: [docs/FINETUNE.md](docs/FINETUNE.md).

## Suy luận (thay YOLO trong code sản phẩm)

```python
from openobb import OpenOBB
model = OpenOBB("best.pt")
for r in model.predict("page.png", conf=0.25):
    print(r.obb.xyxyxyxy, r.obb.conf, r.obb.cls, r.names)
```

Bảng đối chiếu với Ultralytics: [docs/DEPLOY.md](docs/DEPLOY.md).
Test model bằng giao diện kéo-thả: `run_app.bat` (Streamlit).

## Kiến trúc

| `model=` | Mô tả | Tài liệu |
|---|---|---|
| `openobb5x` | head dense + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ | [docs/OPENOBB5.md](docs/OPENOBB5.md) |
| `openobb4x` | DINOv2 + DETR + nhánh dense | [docs/OPENOBB4.md](docs/OPENOBB4.md) |
| `openobb3x` | encoder COCO + head dense oriented dị hướng (nhanh) | [docs/OPENOBB3.md](docs/OPENOBB3.md) |
| `openobb2x` | dense segment-aware, train từ đầu (đã dừng phát triển) | [docs/OPENOBB2.md](docs/OPENOBB2.md) |
| `openobb1x` | D-FINE-OBB (DETR) | [docs/OPENOBB1.md](docs/OPENOBB1.md) |

Kết quả trên Wall_Color (thước Ultralytics): YOLO26x 0.746 / 0.499; OpenOBB3-x 0.712 / 0.467.
Chi tiết ở [research/LEDGER.md](research/LEDGER.md).

## Cấu trúc repo

| Thư mục | Nội dung |
|---|---|
| `openobb/` | Thư viện: models, train, predict, eval, CLI |
| `tests/` | `pytest -q tests/` (chạy CPU) |
| `colab/` | Notebook Colab, script dữ liệu, baseline YOLO (chỉ dùng để đo) |
| `docs/` | Tài liệu kiến trúc, deploy, fine-tune |
| `research/` | Sổ cái nghiên cứu, giả thuyết |
| `tools/` | Script tự động (Colab jobs, đồng bộ repo) |
| `app/` | App Streamlit để test model |

## License

Proprietary, xem [LICENSE](LICENSE). Thành phần bên thứ ba (Apache-2.0): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
