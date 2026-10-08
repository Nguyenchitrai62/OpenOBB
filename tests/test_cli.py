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


def test_auto_scale(tmp_path):
    from vrdet.cli import auto_scale
    data = _dataset(tmp_path / "ds", size=640)
    assert auto_scale(str(data), 512) == 0.8              # 640 px pages -> one 512 tile
    assert auto_scale(str(data), 1024) == 1.0             # page fits in a tile: native resolution
    assert auto_scale(str(data), 256) == 1.0              # much larger than a tile: tiled at native resolution


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU smoke train (set VRDET_SLOW=1)")
def test_train_val_predict_smoke(tmp_path):
    from vrdet.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("s")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=512, tile_scale=0.8, project=str(tmp_path / "runs"),
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


def test_console_rows():
    import io

    from vrdet.console import EpochBar, val_rows
    from vrdet.eval.dota import evaluate, summary_table
    sq = np.array([0, 0, 10, 0, 10, 10, 0, 10], float)
    gts = {"a": [(sq.tolist(), "door", False)], "b": [((sq + 50).tolist(), "door", False)]}
    dets = {"door": (["a", "b", "b"], np.array([0.9, 0.8, 0.3]), np.stack([sq, sq + 50, sq + 20]))}
    res = evaluate(dets, gts, classes=("door", "window"))
    assert res["P"] == 1.0 and res["R"] == 1.0 and res["classes"]["door"]["images"] == 2
    table = summary_table(res, ("door", "window")).splitlines()
    assert len(table) == 3 and table[1].split()[:3] == ["all", "2", "2"]          # window has no GT: not listed
    buf = io.StringIO()
    bar = EpochBar(0, 10, 4, 1024, stream=buf)
    bar.update(4, {"loss_mal": 0.5, "loss_angle": 1e-12}, 37, 21.5, force=True)
    bar.close()
    val_rows(res, 3.0, stream=buf)
    out = buf.getvalue()
    assert "1/10" in out and "21.5G" in out and "100%" in out and "e-12" not in out and "mAP50-95" in out


def test_fit_mode_one_tile_per_image(tmp_path):
    import json as _json

    from vrdet.data.prepare import prepare
    from vrdet.eval.dota import merge_patches
    data = _dataset(tmp_path / "ds", size=640)
    out = prepare(str(data), tmp_path / "prep", size=512, fit=True, workers=1)
    metas = [_json.loads(l) for l in (out / "meta" / "val.jsonl").read_text().splitlines()]
    assert len(metas) == 1 and abs(metas[0]["rate"] - 0.8) < 1e-9             # 640 px page -> long side 512
    poly = np.array(metas[0]["objs"][0][3:])
    assert np.allclose(poly, [80, 80, 208, 80, 208, 112, 80, 112], atol=0.5)
    merged = merge_patches([(metas[0]["name"], 0, 0.9, poly)])               # back to original pixels
    assert np.allclose(merged[0][2][0], [100, 100, 260, 100, 260, 140, 100, 140], atol=0.7)


def test_train_flags_and_oom_retry(tmp_path, monkeypatch):
    import vrdet.train as trainer
    from vrdet.cli import train
    calls = []

    def fake_main(argv):
        calls.append(argv)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(trainer, "main", fake_main)
    data = _dataset(tmp_path / "ds")
    train(str(data), epochs=10, batch=8, imgsz=512, project=str(tmp_path / "runs"), name="t",
          cache=str(tmp_path / "cache"), workers=1)
    assert len(calls) == 2
    first, second = (" ".join(c) for c in calls)
    for flag in ("--fit", "--lsk", "--mosaic-p 1.0", "--mosaic-mode yolo", "--scale-jitter 0.5", "--translate 0.1",
                 "--mosaic-off 10", "--patience 100", "--warmup 100", "--eval-every 1", "--batch 8"):
        assert flag in first, flag
    assert "--batch 4" in second                                             # halved after the OOM
