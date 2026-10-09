import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from openobb.data.dota import DotaPatches, dataset_classes

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_roundtrip(tmp_path):
    src = tmp_path / "export"
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for k in range(2):
            W, H = 1400, 900
            img = np.full((H, W, 3), 255, np.uint8)
            cv2.rectangle(img, (200, 300), (400, 360), (0, 0, 0), 2)
            cv2.imwrite(str(src / "images" / split / f"{split}{k}.jpg"), img)
            poly = (np.array([200, 300, 400, 300, 400, 360, 200, 360], float).reshape(4, 2) / [W, H]).reshape(-1)
            (src / "labels" / split / f"{split}{k}.txt").write_text("1 " + " ".join(f"{v:.6f}" for v in poly) + "\n")
    (src / "data.yaml").write_text("names:\n  0: junction\n  1: fire pipe\n")
    out = tmp_path / "vrdet"
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_dataset.py"), "--src", str(src), "--out", str(out),
                    "--workers", "1"], check=True)
    assert dataset_classes(out) == ("junction", "fire_pipe")
    metas = [json.loads(l) for l in (out / "meta" / "val.jsonl").read_text().splitlines()]
    assert len(metas) == 2 * 2            # 1400 px wide -> two 1024 tiles with 200 px overlap
    objs = [o for m in metas for o in m["objs"]]
    assert all(o[0] == 1 for o in objs) and len(objs) == 2
    assert (out / "gt" / "val" / "val0.txt").read_text().split()[8] == "fire_pipe"
    img, t = DotaPatches(out, "val")[0]
    assert img.shape == (3, 1024, 1024)


def test_import_holds_out_val_when_missing(tmp_path):
    src = tmp_path / "export" / "data"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    (src / "valid" / "images").mkdir(parents=True)                   # AI_Takeoff exports an empty valid/
    for k in range(10):
        img = np.full((500, 600, 3), 255, np.uint8)
        cv2.imwrite(str(src / "train" / "images" / f"p{k}.jpg"), img)
        (src / "train" / "labels" / f"p{k}.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2\n")
    (src / "data.yaml").write_text("names: [junction]\n")
    out = tmp_path / "vrdet"
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_dataset.py"), "--src", str(src), "--out", str(out),
                    "--workers", "1", "--val-frac", "0.2"], check=True)
    n_val = len((out / "meta" / "val.jsonl").read_text().splitlines())
    n_tr = len((out / "meta" / "train.jsonl").read_text().splitlines())
    assert n_val == 2 and n_tr == 8
