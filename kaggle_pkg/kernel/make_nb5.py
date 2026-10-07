import json, os
k = r"F:\Source_code\NEW_architecture\kaggle_pkg\kernel"
meta = {"id": "nguynchtrai/floorplan-vlmdet-train", "title": "floorplan-vlmdet-train",
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["nguynchtrai/floorplan-vlmdet-800"],
        "competition_sources": [], "kernel_sources": ["nguynchtrai/floorplan-vlmdet-train"],
        "model_sources": []}
open(os.path.join(k, "kernel-metadata.json"), "w").write(json.dumps(meta, indent=2))
cells = [
    "import torch\nprint(torch.__version__, torch.cuda.is_available())\n!nvidia-smi --query-gpu=name --format=csv,noheader",
    "import os\nIN=None; PREV=None\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs: IN=r\n    if 'vlmdet_small' in ds or 'vlmdet_small' in r: PREV=r if 'runs' in r or 'vlmdet_small' in r else PREV\nprint('IN=',IN)\n# tim weights v4\nimport glob\ncands=glob.glob('/kaggle/input/**/vlmdet_small/best.pt',recursive=True)+glob.glob('/kaggle/input/**/yolo11x/train/weights/best.pt',recursive=True)\nprint(cands)\nopen('/tmp/paths.txt','w').write(IN+'\\n'+'/kaggle/input')",
    "import os,glob\nIN=open('/tmp/paths.txt').read().splitlines()[0]\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "# EVAL model moi (HBB-proxy mAP50) + do FPS\nimport glob\nW=glob.glob('/kaggle/input/**/vlmdet_small/best.pt',recursive=True)[0]\nprint('W=',W)\n!cd /kaggle/working && python eval.py --ckpt $W --imgdir $IN/data/valid/images --lbldir $IN/data/valid/labels --variant small --conf 0.3 --imgsz 768 2>&1 | tail -15",
    "# FPS YOLO11x tren cung valid (predict, khong retrain)\nimport glob,os\nIN=open('/tmp/paths.txt').read().splitlines()[0]\nYW=glob.glob('/kaggle/input/**/yolo11x/train/weights/best.pt',recursive=True)[0]\nprint('YW=',YW)\n!pip install -q ultralytics && cd /kaggle/working && python -c \"from ultralytics import YOLO; import time,glob; m=YOLO('$YW'); fs=sorted(glob.glob('$IN/data/valid/images/*'))[:50]; t=time.time(); m.predict(fs,imgsz=768,conf=0.3,verbose=False); dt=time.time()-t; print(f'yolo11x 50 imgs {dt:.1f}s = {dt/50*1000:.1f} ms/img')\" 2>&1 | tail -3",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v5 staged")
