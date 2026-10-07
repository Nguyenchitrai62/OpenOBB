# Kaggle CLI training (dataset 800 ảnh + full 16k)
# Chạy các lệnh này trên máy có kaggle.json (Settings -> API -> Create New Token),
# hoặc chạy trực tiếp trong Kaggle Notebook (khuyên dùng, khỏi CLI).

# 0) Cài CLI
py -3.12 -m pip install kaggle

# 1) Cấu hình (copy kaggle.json vào %USERPROFILE%/.kaggle/kaggle.json)
# Kaggle -> avatar -> Settings -> API -> Create New Token -> tải kaggle.json

# 2) Upload subset 800/200 thành dataset (để notebook dùng lại, khỏi unzip 1.6GB mỗi lần)
# Nén subset:
#   Compress-Archive -Path cad_vlmdet/subset/* -DestinationPath subset_800.zip
#   kaggle datasets create -p kaggle_upload/  (cần dataset-metadata.json, xem kaggle_docs.txt)
# Đơn giản hơn: trong Kaggle Notebook -> Add Input -> Upload -> kéo subset_800.zip lên.

# 3) Notebook Kaggle (GPU T4/P100/A100): chọn GPU, chạy từng cell:
#   !pip install -q torch opencv-python-headless pillow
#   !cp -r /kaggle/input/<subset-ds>/* /kaggle/working/data/
#   !cp -r /kaggle/input/<code-ds>/cad_vlmdet /kaggle/working/  (hoặc git clone repo code)
#   !cd /kaggle/working/cad_vlmdet && python train.py --data /kaggle/working/data/data.yaml --variant small --imgsz 1024 --epochs 50 --batch 16 --out /kaggle/working/runs/vlmdet_small
#   !cd /kaggle/working/cad_vlmdet && pip install -q ultralytics && python yolo_baseline.py --data /kaggle/working/data/data.yaml --model yolo11x-obb.pt --epochs 50 --batch 16
#   -> so sánh runs/vlmdet_small vs runs/yolo_baseline (mAP50, mAP50-95, P/R từng class hiếm)

# 4) Full 16k (upload dataset_obb_train_v3.zip thành Kaggle dataset một lần):
#   kaggle datasets create -p kaggle_full/ --dir-mode zip
#   train: python train.py --data .../data.yaml --variant base --imgsz 1024 --epochs 100 --batch 32 --lr 2e-4

# Lưu ý .env của bạn (kaggle_key=KGAT_...) KHÔNG phải kaggle.json chuẩn.
# Hãy tạo token mới như bước 1, không commit kaggle.json lên git.
