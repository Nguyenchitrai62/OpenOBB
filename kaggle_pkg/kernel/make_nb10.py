import json, os
k = r"F:\Source_code\NEW_architecture\kaggle_pkg\kernel"
meta = {"id": "nguynchtrai/floorplan-vlmdet-train", "title": "floorplan-vlmdet-train",
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["nguynchtrai/floorplan-vlmdet-800"],
        "competition_sources": [], "kernel_sources": [],
        "model_sources": []}
open(os.path.join(k, "kernel-metadata.json"), "w").write(json.dumps(meta, indent=2))
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name --format=csv,noheader",
    "import os\nroots=[]\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs:\n        roots.append(r)\nprint(roots)\nIN=roots[0]\nopen('/tmp/kin.txt','w').write(IN)\nnew='train: '+IN+'/data/train/images\\nval: '+IN+'/data/valid/images\\nnc: 7\\nnames: [wall, window, door, slide_door, double_door, opening, junction]\\n'\nopen('/tmp/data.yaml','w').write(new)\n!ls $IN/data/train/images | wc -l",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "# EXP6/H8: junction ghi truoc, box-that ghi de. Retrain small 20ep.\n!cd /kaggle/working && python train.py --data /tmp/data.yaml --variant small --imgsz 768 --epochs 20 --batch 8 --lr 2e-4 --out /kaggle/working/runs/vlmdet_small4 2>&1 | tail -25",
    "# EVAL exp6\n!cd /kaggle/working && python eval.py --ckpt /kaggle/working/runs/vlmdet_small4/best.pt --imgdir $(cat /tmp/kin.txt)/data/valid/images --lbldir $(cat /tmp/kin.txt)/data/valid/labels --variant small --conf 0.2 --imgsz 768 2>&1 | tail -12",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v10 staged")
