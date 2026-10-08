import json
import os

import cv2
import numpy as np
import pytest

from vrdet.cli import _run_dir, parse_kv


def test_parse_kv():
    kv = parse_kv(["data=a/data.yaml", "epochs=50", "lr0=0.0001", "compile=False", "name=my-run", "p2=true"])
    assert kv == {"data": "a/data.yaml", "epochs": 50, "lr0": 1e-4, "compile": False, "name": "my-run", "p2": True}


def test_run_dir_resume_and_increment(tmp_path):
    d, resuming = _run_dir(tmp_path, "exp", exist_ok=False, resume=True)
    assert d == tmp_path / "exp" and not resuming
    d.mkdir()
    (d / "last.pt").write_text("x")                       # unfinished run -> resumed
    assert _run_dir(tmp_path, "exp", False, True) == (d, True)
    assert _run_dir(tmp_path, "exp", False, False)[0] == tmp_path / "exp2"
    (d / "train_done").write_text("ok")                   # finished run -> new dir
    assert _run_dir(tmp_path, "exp", False, True) == (tmp_path / "exp2", False)


def _dataset(root, n_train=3, n_val=1, size=640):
    for split, n in (("train", n_train), ("valid", n_val)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for k in range(n):
            img = np.full((size, size, 3), 255, np.uint8)
            cv2.rectangle(img, (100, 100), (260, 140), (0, 0, 0), 2)
            cv2.imwrite(str(root / split / "images" / f"{split}{k}.png"), img)
            poly = np.array([100, 100, 260, 100, 260, 140, 100, 140], float) / size
            (root / split / "labels" / f"{split}{k}.txt").write_text("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")
    (root / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnames: [door, window]\n")
    return root / "data.yaml"


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU smoke train (set VRDET_SLOW=1)")
def test_train_val_predict_smoke(tmp_path):
    from vrdet.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("s")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=512, scale=0.8, project=str(tmp_path / "runs"),
                name="t", cache=str(tmp_path / "cache"), workers=0, recipe=False, no_pretrained=True,
                max_iters=1, threads=1)
    assert os.path.exists(r.best) and m.model == r.best
    args = json.loads((tmp_path / "runs" / "t" / "vrdet_args.json").read_text())
    assert args["scale"] == 0.8 and args["img"] == 512
    res = m.val(str(data), cache=str(tmp_path / "cache"), workers=0, batch=2)
    assert "mAP50" in res
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1 and (tmp_path / "pred" / "valid0.json").exists()
    # fine-tune from the result inherits the architecture and maps classes by name
    r2 = Detector(r.best).train(data=str(data), epochs=1, batch=2, project=str(tmp_path / "runs"), name="ft",
                                cache=str(tmp_path / "cache"), workers=0, recipe=False, max_iters=1, threads=1)
    assert os.path.exists(r2.best)
