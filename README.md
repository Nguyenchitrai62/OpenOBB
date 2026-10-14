# OpenOBB

Thư viện phát hiện đối tượng xoay (OBB) cho ảnh bản vẽ CAD, gồm các kiến trúc **OpenOBB** (v1–v5).
Train, predict và đọc kết quả giống YOLO-OBB. Không phụ thuộc Ultralytics (AGPL): chỉ dùng thành phần Apache-2.0, BSD hoặc MIT.

> Đây là repo **nghiên cứu và phát triển**: có code thư viện, notebook Colab, script thí nghiệm và sổ cái.
> Bản sản phẩm sạch (chỉ phần thương mại được) nằm ở GitLab công ty `openanything/openobb`.
> Hai repo được đồng bộ bằng `python tools/push_all.py`.

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

- Luôn cài bản mới nhất; chạy lại đúng lệnh để cập nhật. GPU RTX 50xx: đổi `cu126` thành `cu130`.
- Khi suy luận, model tự chạy trên GPU nếu có. Nếu máy có GPU NVIDIA mà PyTorch là bản CPU, lần dùng đầu tiên sẽ tự cài
  bản GPU rồi báo chạy lại (tắt bằng biến môi trường `OPENOBB_AUTO_TORCH=0`).
- Kiểm tra phiên bản: `python -c "import openobb; print(openobb.__version__)"`.

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
| `openobb11s/m/l/x` | mới: đề xuất dense + tầng quan hệ train một-một (khử trùng tự học, không NMS); nạp được checkpoint v8 / v10 | [docs/OPENOBB11.md](docs/OPENOBB11.md) |
| `openobb12s/m/l/x` | mới: lõi OpenOBB7 + head dense kép (một-nhiều để học, một-một để suy luận), không NMS, không tầng quan hệ | [docs/OPENOBB12.md](docs/OPENOBB12.md) |
| `openobb13s/m/l/x` | mới, nhắm độ chính xác: v12 + luồng độ phân giải gốc + lưới P2 cho vật tí hon (`imgsz` = độ phân giải đầy đủ, ví dụ 2560; lõi chạy ở một nửa) | [docs/OPENOBB13.md](docs/OPENOBB13.md) |
| `openobb14s/m/l/x` | mới, nhắm cả độ chính xác lẫn tốc độ: thân HGNetv2 pretrained + neck conv dày tái tham số hoá tự thiết kế (thay LSK + encoder lai) + head kép nặng hồi quy box trực tiếp + bộ tinh chỉnh box đọc nét ở stride 4 (`box=direct` mặc định, `ref_k`) | [docs/OPENOBB14.md](docs/OPENOBB14.md) |
| `openobb15s/m/l/x` | mới, nhắm độ chính xác: v14 + luồng độ phân giải gốc + lưới P2, bộ tinh chỉnh box đọc nét ở độ phân giải gốc (`imgsz` = cỡ lõi, ảnh vào 2×; `neck=dfine` cho vòng này) | [docs/OPENOBB15.md](docs/OPENOBB15.md) |
| `openobb10s/m/l/x` | mới, chưa train: v8 + head 3 slot / điểm (vật nhỏ chồng nhau như wall_300 cạnh junction), gán nhãn theo tầm với (box lớn), cân bằng class hiếm; nạp được checkpoint v8 | [docs/OPENOBB10.md](docs/OPENOBB10.md) |
| `openobb9s/m/l/x` | mới, chưa train: lõi tự thiết kế (conv 3×3 dày tái tham số hoá) + head v8; pretrain bằng chưng cất 2 thầy | [docs/OPENOBB9.md](docs/OPENOBB9.md) |
| `openobb8s/m/l/x` | bản tốt nhất: v3 sửa lỗi + tầng suy luận quan hệ theo hướng (bắt đầu y hệt v3) | [docs/OPENOBB8.md](docs/OPENOBB8.md) |
| `openobb6n/s/m/l/x` | kiến trúc tự thiết kế từ phân tích (backbone + neck + head riêng), pretrain bằng chưng cất DINOv2 (`openobb pretrain`) | [docs/OPENOBB6.md](docs/OPENOBB6.md) |
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
