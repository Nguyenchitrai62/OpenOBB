# Deploy OpenOBB thay cho YOLO-OBB

Giao diện suy luận của OpenOBB giống model OBB của Ultralytics. Code đang đọc kết quả YOLO chỉ cần đổi 2 dòng import và tạo model.
OpenOBB tự viết phần giao diện này; package không phụ thuộc Ultralytics.

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
from openobb import OpenOBB
model = OpenOBB("best.pt")
```

Phần còn lại giữ nguyên, ví dụ:

```python
results = model.predict(img_bgr, conf=0.25, iou=0.7, verbose=False)   # hoặc model(img_bgr)
for r in results:
    for pts, c, s in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
        name = r.names[int(c)]          # pts: (4, 2) toạ độ pixel 4 góc
```

## Bảng đối chiếu

| Ultralytics | OpenOBB | Ghi chú |
|---|---|---|
| `YOLO("best.pt")` | `OpenOBB("best.pt")` | Mọi checkpoint OpenOBB v1–v5 |
| `model.predict(source, conf, iou, imgsz, device, batch, max_det, classes, agnostic_nms, save, save_txt, save_conf, save_crop, stream, verbose, project, name, exist_ok, line_width, show, show_labels, show_conf, show_boxes)` | giống | Tên tham số sai thì báo lỗi như YOLO; tham số chỉ dành cho video (`vid_stride`, `stream_buffer`, `visualize`, `augment`, `embed`...) chỉ cảnh báo rồi bỏ qua |
| `model(source)` | giống | |
| `source`: path / `Path`, thư mục, glob, URL, file `.txt` liệt kê ảnh, list, numpy BGR, PIL, tensor (B,3,H,W) RGB 0–1 | giống | Ảnh đọc giống loader của YOLO (`IMREAD_COLOR`: BGR, áp xoay EXIF, bỏ alpha, đường dẫn Unicode được), đọc lần lượt từng ảnh. **Không hỗ trợ video/webcam/stream** (bản vẽ là ảnh tĩnh) |
| `r.obb.xywhr` | giống | cx, cy, w, h, góc (rad); **góc trong [0, π/2)**, đổi chỗ w/h khi cần, đúng quy ước `regularize_rboxes` của Ultralytics |
| `r.obb.xyxyxyxy`, `.xyxyxyxyn`, `.xyxy`, `.conf`, `.cls`, `.data` (N,7), `.id` (= None) | giống | Tensor torch nằm trên device của model (GPU nếu có), thứ tự 4 góc như YOLO; `.cpu()`, `.numpy()`, `.to()` dùng được |
| `r.boxes` | `None` | Như model OBB của YOLO |
| `r.names`, `r.orig_img` (BGR), `r.orig_shape`, `r.path`, `r.speed`, `r.save_dir` | giống | |
| `r.plot()`, `r.save()`, `r.show()`, `r.save_txt(f, save_conf)`, `r.save_crop(dir, name)`, `r.summary()`, `r.to_json()`, `r.to_df()`, `r.to_csv()`, `r.verbose()`, `r.new()`, `r.update(obb=...)` | giống | `save_txt` theo định dạng nhãn YOLO-OBB (toạ độ chuẩn hoá, `%g`), không tạo file khi ảnh không có object; `save_crop` cắt xoay theo box |
| `len(r)`, `r[i]` | giống | |
| `model.names`, `model.task` (= "obb") | giống | |
| `model.val(data=...)` → `metrics.box.map`, `.map50`, `.map75`, `.maps`, `.mp`, `.mr`, `.p`, `.r`, `.ap50`, `.ap`, `.ap_class_index`, `metrics.results_dict`, `metrics.fitness` | giống | mAP và P/R tính theo cách chấm của YOLO (P/R tại conf có F1 trung bình cao nhất) |
| `model.train(...)` | `OpenOBB("openobb5x").train(data=..., epochs=...)` | Hoặc CLI `openobb train ...` |
| `model.export(...)`, `model.track(...)` | chưa có | Gọi sẽ báo `NotImplementedError` |
| CLI `yolo obb predict model=... source=...` | `openobb predict model=... source=...` | Cùng tham số và mặc định (`save=True`); ảnh vẽ box lưu `runs/obb/predict/<tên gốc>`, nhãn ở `labels/<tên>.txt` khi `save_txt=True`; log `image i/n ...`, `Speed: ...`, `Results saved to ...` |

## Khác biệt cần biết

| Tham số | Ultralytics | OpenOBB |
|---|---|---|
| `imgsz` mặc định | 640 | **kích thước lúc train** (ví dụ 1280). Nên để mặc định |
| `iou` mặc định | 0.7 | 0.7 với model resize cả ảnh; 0.1 với model cắt tile (`tile=True`), vì ghép tile cần NMS chặt |
| `conf` | số | Thêm `conf="auto"`: dùng ngưỡng riêng từng class (F2 tốt nhất trên val) lưu trong `best.pt` |
| `max_det` | 300 | 300; bản vẽ dày có thể tăng |
| `half` | tham số | GPU tự chạy bf16 |
| `tile` | không có | `tile=True` cắt trang rất lớn thành tile ở độ phân giải gốc |

Kết quả của `OpenOBB.predict` giống hệt `openobb predict` (CLI) và app Streamlit: cùng tiền xử lý như lúc train và cùng bước ghép NMS.

Tiền xử lý so với YOLO: YOLO letterbox ảnh về `imgsz` (pad màu xám 114); OpenOBB resize cạnh dài về `imgsz` rồi pad thành ô vuông bằng màu nền lúc train. Cả hai đều đổi kết quả về toạ độ ảnh gốc, nên đầu ra dùng y như nhau.
