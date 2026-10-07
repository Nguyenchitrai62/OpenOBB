import json, os
k = r"F:\Source_code\NEW_architecture\kaggle_pkg\kernel"
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name,memory.total --format=csv",
    "import glob, os\nprint(glob.glob('/kaggle/input/*'))\nIN = glob.glob('/kaggle/input/*floorplan*')[0]\nprint('IN=', IN)\nprint(os.listdir(IN)[:20])",
    "import glob\nIN = glob.glob('/kaggle/input/*floorplan*')[0]\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py\n!ls $IN/data/train/images | head -3\n!ls $IN/data/train/images | wc -l",
    "!cd /kaggle/working && python train.py --data /kaggle/input/floorplan-vlmdet-800/data_kaggle.yaml --variant small --imgsz 768 --epochs 20 --batch 8 --lr 2e-4 --out /kaggle/working/runs/vlmdet_small 2>&1 | tail -30",
    "!pip install -q ultralytics && cd /kaggle/working && python yolo_baseline.py --data /kaggle/input/floorplan-vlmdet-800/data_kaggle.yaml --model yolo11x-obb.pt --imgsz 768 --epochs 20 --batch 8 --project /kaggle/working/runs/yolo11x 2>&1 | tail -15",
    "!ls /kaggle/working/runs/ && ls /kaggle/working/runs/vlmdet_small && cd /kaggle/working && python infer_run.py --ckpt /kaggle/working/runs/vlmdet_small/best.pt --imgdir /kaggle/input/floorplan-vlmdet-800/data/valid/images --out /kaggle/working/runs/vis --variant small --conf 0.3 --imgsz 768 --maxn 12 2>&1 | tail -15",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v2 staged")
