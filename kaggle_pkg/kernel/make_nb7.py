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
    "import torch\nprint(torch.__version__, torch.cuda.is_available())",
    "import os,glob\nIN=None\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs: IN=r\nopen('/tmp/kin.txt','w').write(IN)\nW=glob.glob('/kaggle/input/**/vlmdet_small2/best.pt',recursive=True)\nprint('IN=',IN)\nprint('W=',W)\nopen('/tmp/w.txt','w').write(W[0] if W else '')",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "# EXP3: sweep conf x NMS-mode tren weights v6 (khong retrain)\n!cd /kaggle/working && for C in 0.1 0.2 0.3; do for N in \"\" \"--hbb_nms\"; do echo \"=== conf=$C $N ===\"; python eval.py --ckpt $(cat /tmp/w.txt) --imgdir $(cat /tmp/kin.txt)/data/valid/images --lbldir $(cat /tmp/kin.txt)/data/valid/labels --variant small --conf $C --imgsz 768 $N 2>&1 | grep -E \"AP50|mean|infer\"; done; done",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v7 staged")
