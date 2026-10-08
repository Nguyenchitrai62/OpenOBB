# Train VRDet trên dataset YOLO-OBB tự gắn nhãn

Áp dụng cho dataset OBB chuẩn YOLO (Ultralytics), gồm cả bản export "training dataset" của AI_Takeoff.
Mọi lệnh chạy từ thư mục gốc repo.

## 0. Môi trường

```bash
pip install -r requirements.txt          # torch (CUDA), opencv-python, shapely, scipy, numpy, pyyaml
```

- GPU: 1 GPU từ 24 GB (bản S) hoặc 40 GB (bản X). Colab G4/A100 đều được; G4 rẻ nhất trên mỗi ảnh.
- Lần chạy đầu tự tải weights khởi tạo D-FINE COCO (Apache-2.0).

## 1. Chuẩn bị dữ liệu

Bố cục đầu vào được chấp nhận:

```
<src>/images/train/*.jpg|png   <src>/labels/train/*.txt     (Ultralytics)
<src>/train/images/*            <src>/train/labels/*         (AI_Takeoff export: <zip>/data)
<src>/data.yaml                 names: [...] hoặc {0: ..., 1: ...}
```

Mỗi dòng nhãn: `class_id x1 y1 x2 y2 x3 y3 x4 y4`, toạ độ chuẩn hoá [0, 1] theo ảnh.

```bash
python tools/import_yolo_obb.py --src <src> --out datasets/mydata
```

- Trang lớn được cắt thành tile 1024×1024 chồng 200 px. Khi eval, kết quả được gộp lại theo trang gốc, nên mAP tính trên từng trang.
- Thiếu tập val (`valid/` trống như export AI_Takeoff) thì tự tách 15% số trang (`--val-frac`).
- Ra thư mục `datasets/mydata`: `images/`, `labels/`, `meta/`, `gt/`, `classes.json`.

### 1b. (Tuỳ chọn) Nhánh vector: nét vẽ và layer từ PDF

Bật khi có PDF gốc của từng trang. Đây là phần "hiểu ngữ nghĩa" mà YOLO không có.

```bash
python tools/pdf_vectors.py --export <thư mục export đã giải nén> --pdf-dir <thư mục chứa PDF> --out <export>/vectors
python tools/import_yolo_obb.py --src <export>/data --out datasets/mydata --vectors-dir <export>/vectors
```

`pdf_vectors.py` đọc `dataset.json` (tên PDF, số trang, tỉ lệ render) và quy đổi nét vẽ cùng layer (OCG) về đúng toạ độ ảnh export.

## 2. Train

Bản S (nhanh, 10.9M tham số):

```bash
python -m vrdet.train --data datasets/mydata --out runs/my_s --size s --epochs 36 --batch 16 --lr 1e-4 \
  --rfs 0.1 --channels-last --compile --eval-every 6 --eval-images 100 --eval-queries 900
```

- **Có vector:** thêm `--vectors --vec-lfe --layer-drop 0.1`.
- **Bản X** (chính xác hơn, khoảng 2× thời gian suy luận): `--size x --batch 8 --lr 6e-5 --backbone-mult 0.1 --wd 1.25e-4`.
- **Số epoch:** dataset nhỏ (dưới 2k trang) thì 36–50 epoch; lớn thì 24.
- **Thành phần kiến trúc tốt nhất** (nhóm một-nhiều, AQD, loss góc, IoU-cost, box neo theo nét vector, kiến trúc lai): xem
  `docs/ARCHITECTURE_CARD.md` §6 sau khi các thí nghiệm c1/c2/c3 xong.
- **Resume:** chạy lại đúng lệnh; trainer tự nạp `runs/<tên>/last.pt`.

Kết quả trong `runs/<tên>/`:
- `metrics.jsonl`: mỗi epoch một dòng.
- `eval_val.txt`: AP50, AP50:95 và recall theo class, tính trên trang gốc.
- `last.pt`: checkpoint, đã chứa danh sách class.

## 3. Suy luận

```bash
python -m vrdet.predict --ckpt runs/my_s/last.pt --src <ảnh trang hoặc thư mục> --out preds/ --vis
# model train với --vectors: thêm --vectors-dir <thư mục npz từ pdf_vectors.py>
```

Mỗi ảnh ra 3 file:
- `<tên>.txt`: `class x1 y1 x2 y2 x3 y3 x4 y4 score`, chuẩn hoá theo ảnh.
- `<tên>.json`: toạ độ pixel và tên class.
- `<tên>_vis.jpg`: ảnh vẽ kết quả (khi có `--vis`).

## 4. Train trên Colab

1. Nén repo (bỏ `runs/`, `datasets/`) và `datasets/mydata` thành zip, upload lên Google Drive.
2. Mở `colab/finetune_vrdet.ipynb` trên Colab, chọn GPU, chạy từng ô: mount Drive, giải nén, cài đặt, train, suy luận.
3. Checkpoint ghi thẳng vào Drive nên mất phiên vẫn resume được.
