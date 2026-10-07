import json, os
k = r"F:\Source_code\NEW_architecture\kaggle_pkg\kernel"
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name,memory.total --format=csv",
    "import os\nroots=[]\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs:\n        roots.append(r)\nprint(roots)\nIN=roots[0]\nopen('/tmp/kin.txt','w').write(IN)\nprint(IN)\nprint(os.listdir(IN))\n# sinh data.yaml DONG voi mount thuc te\nL=open(os.path.join(IN,'data_kaggle.yaml')).read()\nprint('OLD yaml:'); print(L)\nnew='train: '+IN+'/data/train/images\\nval: '+IN+'/data/valid/images\\nnc: 7\\nnames: [wall, window, door, slide_door, double_door, opening, junction]\\n'\nopen('/tmp/data.yaml','w').write(new)\nprint('NEW yaml:'); print(new)\n!ls $IN/data/train/images | wc -l\n!ls $IN/data/valid/images | wc -l",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "!cd /kaggle/working && python train.py --data /tmp/data.yaml --variant small --imgsz 768 --epochs 20 --batch 8 --lr 2e-4 --out /kaggle/working/runs/vlmdet_small 2>&1 | tail -30",
    "!pip install -q ultralytics && cd /kaggle/working && python yolo_baseline.py --data /tmp/data.yaml --model yolo11x-obb.pt --imgsz 768 --epochs 20 --batch 8 --project /kaggle/working/runs/yolo11x 2>&1 | tail -15",
    "!ls /kaggle/working/runs/ && ls /kaggle/working/runs/vlmdet_small && cd /kaggle/working && IN=$(cat /tmp/kin.txt) && python infer_run.py --ckpt /kaggle/working/runs/vlmdet_small/best.pt --imgdir $IN/data/valid/images --out /kaggle/working/runs/vis --variant small --conf 0.3 --imgsz 768 --maxn 12 2>&1 | tail -15",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v4 staged")
