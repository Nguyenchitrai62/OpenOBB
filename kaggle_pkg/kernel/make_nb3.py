import json, os
k = r"F:\Source_code\NEW_architecture\kaggle_pkg\kernel"
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name,memory.total --format=csv",
    "import os\nroots=[]\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs:\n        roots.append(r)\nprint('roots=',roots)\nIN=roots[0]\nprint('IN=',IN)\nprint(os.listdir(IN))\nprint(os.listdir(os.path.join(IN,'data')))",
    "import os\nIN=open('/tmp/kin.txt').read().strip() if os.path.exists('/tmp/kin.txt') else None\nprint(IN)",
    "print('skip')",
]
# real robust single-flow: do everything with discovery in each cell (no /tmp dependency)
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name,memory.total --format=csv",
    "import os\nroots=[]\nfor r,ds,fs in os.walk('/kaggle/input'):\n    SSA=[x for x in fs if x=='data_kaggle.yaml']\n    if SSA:\n        roots.append(r)\nprint(roots)\nIN=roots[0]\nopen('/tmp/kin.txt','w').write(IN)\nprint(IN)\nprint(os.listdir(IN))\n!ls $IN/data/train/images | wc -l",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cd /kaggle/working && python train.py --data $IN/data_kaggle.yaml --variant small --imgsz 768 --epochs 20 --batch 8 --lr 2e-4 --out /kaggle/working/runs/vlmdet_small 2>&1 | tail -30",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!pip install -q ultralytics && cd /kaggle/working && python yolo_baseline.py --data $IN/data_kaggle.yaml --model yolo11x-obb.pt --imgsz 768 --epochs 20 --batch 8 --project /kaggle/working/runs/yolo11x 2>&1 | tail -15",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!ls /kaggle/working/runs/ && ls /kaggle/working/runs/vlmdet_small && cd /kaggle/working && python infer_run.py --ckpt /kaggle/working/runs/vlmdet_small/best.pt --imgdir $IN/data/valid/images --out /kaggle/working/runs/vis --variant small --conf 0.3 --imgsz 768 --maxn 12 2>&1 | tail -15",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v3 staged")
