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
    "import os,glob\nIN=None\nfor r,ds,fs in os.walk('/kaggle/input'):\n    if 'data_kaggle.yaml' in fs and 'datasets' in r: IN=r\nopen('/tmp/kin.txt','w').write(IN)\nWs=glob.glob('/kaggle/input/**/vlmdet_small3/best.pt',recursive=True)+glob.glob('/kaggle/input/**/vlmdet_small4/best.pt',recursive=True)\nprint('IN=',IN)\nprint('W=',Ws)\nopen('/tmp/ws.txt','w').write('\\n'.join(Ws))",
    "import os\nIN=open('/tmp/kin.txt').read().strip()\n!cp $IN/*.py /kaggle/working/ && ls /kaggle/working/*.py",
    "# EXP7: quet nguong thap tren weights v8/v10 - door co song o score thap khong?\n!cd /kaggle/working && for W in $(cat /tmp/ws.txt); do echo \"### $W\"; for C in 0.05 0.1 0.15; do echo \"--- conf=$C\"; python eval.py --ckpt $W --imgdir $(cat /tmp/kin.txt)/data/valid/images --lbldir $(cat /tmp/kin.txt)/data/valid/labels --variant small --conf $C --imgsz 768 2>&1 | grep -E \"AP50|mean|infer\"; done; done",
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [{"cell_type": "code", "metadata": {}, "source": [c], "outputs": [], "execution_count": None} for c in cells]}
open(os.path.join(k, "notebook.ipynb"), "w").write(json.dumps(nb))
print("v11 staged")
