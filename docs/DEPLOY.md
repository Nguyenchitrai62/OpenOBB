# Deploy VRDet thay cho YOLO-OBB

Giao diện suy luận của VRDet giống model OBB của Ultralytics. Code đang đọc kết quả YOLO chỉ cần đổi 2 dòng import và tạo model.
VRDet tự viết phần giao diện này; package không phụ thuộc Ultralytics.

## Cài đặt

```bash
pip install "openobb @ git+https://github.com/Nguyenchitrai62/OpenOBB.git"
```

- Phụ thuộc: torch, numpy, opencv-python, shapely, scipy, pyyaml. Pillow là tuỳ chọn, chỉ cần khi đưa ảnh PIL vào.
- Nếu repo chuyển sang private: `pip install "openobb @ git+https://<token>@github.com/Nguyenchitrai62/OpenOBB.git"`.
- Chỉ cần file `best.pt`. Mọi thông tin kiến trúc (v1–v5), class và imgsz đều nằm trong checkpoint.

## Đổi code

```python
# trước
from ultralytics import YOLO
model = YOLO("best.pt")

# sau
from openobb import VRDet
model = VRDet("best.pt")
```

Phần còn lại giữ nguyên, ví dụ:

```python
results = model.predict(img_bgr, conf=0.25, iou=0.7, verbose=False)   # hoặc model(img_bgr)
for r in results:
    for pts, c, s in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
        name = r.names[int(c)]          # pts: (4, 2) toạ độ pixel 4 góc
```

## Bảng đối chiếu

| Ultralytics | VRDet | Ghi chú |
|---|---|---|
| `YOLO("best.pt")` | `VRDet("best.pt")` | Mọi checkpoint VRDet v1–v5 |
| `model.predict(source, conf, iou, imgsz, device, max_det, classes, agnostic_nms, save, save_txt, save_conf, stream, verbose, project, name, exist_ok, line_width, show, show_labels, show_conf)` | giống | Tham số khác bị bỏ qua |
| `model(source)` | giống | |
| `source`: path, thư mục, glob, URL, list, numpy BGR, PIL, tensor (B,3,H,W) RGB 0–1 | giống | **Chưa hỗ trợ video/webcam/stream** |
| `r.obb.xywhr` | giống | cx, cy, w, h, góc (rad); **w ≥ h, góc trong [0, π)** như Ultralytics |
| `r.obb.xyxyxyxy`, `.xyxyxyxyn`, `.xyxy`, `.conf`, `.cls`, `.data` (N,7), `.id` (= None) | giống | Tensor torch trên CPU; `.cpu()`, `.numpy()`, `.to()` dùng được |
| `r.boxes` | `None` | Như model OBB của YOLO |
| `r.names`, `r.orig_img` (BGR), `r.orig_shape`, `r.path`, `r.speed`, `r.save_dir` | giống | |
| `r.plot()`, `r.save()`, `r.show()`, `r.save_txt(f, save_conf)`, `r.summary()`, `r.to_json()`, `r.to_df()`, `r.verbose()` | giống | `save_txt` theo định dạng nhãn YOLO-OBB (toạ độ chuẩn hoá) |
| `len(r)`, `r[i]` | giống | |
| `model.names`, `model.task` (= "obb") | giống | |
| `model.val(data=...)` → `metrics.box.map`, `.map50`, `.maps`, `.mp`, `.mr`, `metrics.results_dict` | giống | mAP tính theo cách chấm của YOLO |
| `model.train(...)` | `VRDet("vrdet5x").train(data=..., epochs=...)` | Hoặc CLI `openobb train ...` |
| `model.export(...)`, `model.track(...)` | chưa có | |

## Khác biệt cần biết

| Tham số | Ultralytics | VRDet |
|---|---|---|
| `imgsz` mặc định | 640 | **kích thước lúc train** (ví dụ 1280). Nên để mặc định |
| `iou` mặc định | 0.7 | 0.7 với model resize cả ảnh; 0.1 với model cắt tile (`tile=True`), vì ghép tile cần NMS chặt |
| `conf` | số | Thêm `conf="auto"`: dùng ngưỡng riêng từng class (F2 tốt nhất trên val) lưu trong `best.pt` |
| `max_det` | 300 | 300; bản vẽ dày có thể tăng |
| `half` | tham số | GPU tự chạy bf16 |
| `tile` | không có | `tile=True` cắt trang rất lớn thành tile ở độ phân giải gốc |

Kết quả của `VRDet.predict` giống hệt `openobb predict` (CLI) và app Streamlit: cùng tiền xử lý như lúc train và cùng bước ghép NMS.
