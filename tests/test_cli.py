import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from openobb.cli import parse_kv


def test_parse_kv():
    kv = parse_kv(["data=a/data.yaml", "epochs=50", "lr0=0.0001", "compile=False", "name=my-run", "p2=true"])
    assert kv == {"data": "a/data.yaml", "epochs": 50, "lr0": 1e-4, "compile": False, "name": "my-run", "p2": True}


def test_run_folders_and_resume_lookup(tmp_path):
    from openobb.cli import _find_last, increment_path
    assert increment_path(tmp_path, "train") == tmp_path / "train"
    (tmp_path / "train" / "weights").mkdir(parents=True)
    assert increment_path(tmp_path, "train") == tmp_path / "train2"            # never reuses a folder by default
    assert increment_path(tmp_path, "train", exist_ok=True) == tmp_path / "train"
    (tmp_path / "train" / "weights" / "last.pt").write_text("x")
    (tmp_path / "train2" / "weights").mkdir(parents=True)
    (tmp_path / "train2" / "weights" / "last.pt").write_text("x")
    newer = (tmp_path / "train" / "weights" / "last.pt").stat().st_mtime + 10
    os.utime(tmp_path / "train2" / "weights" / "last.pt", (newer, newer))
    assert _find_last(True, "s", tmp_path, None) == tmp_path / "train2" / "weights" / "last.pt"   # newest
    assert _find_last(True, "s", tmp_path, "train") == tmp_path / "train2" / "weights" / "last.pt"   # newest of train*
    assert _find_last(True, "pretrained.pt", tmp_path, None) == tmp_path / "train2" / "weights" / "last.pt"
    (tmp_path / "train2" / "train_done").write_text("ok")
    assert _find_last(True, "s", tmp_path, None) == tmp_path / "train" / "weights" / "last.pt"


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
    from openobb.cli import auto_scale
    data = _dataset(tmp_path / "ds", size=640)
    assert auto_scale(str(data), 512) == 0.8              # 640 px pages -> one 512 tile
    assert auto_scale(str(data), 1024) == 1.0             # page fits in a tile: native resolution
    assert auto_scale(str(data), 256) == 1.0              # much larger than a tile: tiled at native resolution


@pytest.mark.skipif(os.environ.get("VRDET_SLOW") != "1", reason="CPU smoke train (set VRDET_SLOW=1)")
def test_train_val_predict_smoke(tmp_path):
    from openobb.cli import Detector
    data = _dataset(tmp_path / "ds")
    m = Detector("s")
    r = m.train(data=str(data), epochs=1, batch=2, imgsz=512, tile_scale=0.8, project=str(tmp_path / "runs"),
                name="t", cache_dir=str(tmp_path / "cache"), workers=0, recipe=False, no_pretrained=True,
                max_iters=1, threads=1)
    assert os.path.exists(r.best) and m.model == r.best
    args = json.loads((tmp_path / "runs" / "t" / "vrdet_args.json").read_text())
    assert args["scale"] == 0.8 and args["img"] == 512
    res = m.val(str(data), cache_dir=str(tmp_path / "cache"), workers=0, batch=2)
    assert "mAP50" in res
    out = m.predict(str(tmp_path / "ds" / "valid" / "images"), conf=0.0, save_dir=str(tmp_path / "pred"))
    assert len(out) == 1 and (tmp_path / "pred" / "valid0.json").exists()
    # fine-tune from the result inherits the architecture and maps classes by name
    r2 = Detector(r.best).train(data=str(data), epochs=1, batch=2, project=str(tmp_path / "runs"), name="ft",
                                cache_dir=str(tmp_path / "cache"), workers=0, recipe=False, max_iters=1, threads=1)
    assert os.path.exists(r2.best)


def test_console_rows():
    import io

    from openobb.console import EpochBar, val_rows
    from openobb.eval.dota import evaluate, summary_table
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

    from openobb.data.prepare import prepare
    from openobb.eval.dota import merge_patches
    data = _dataset(tmp_path / "ds", size=640)
    out = prepare(str(data), tmp_path / "prep", size=512, fit=True, workers=1)
    metas = [_json.loads(l) for l in (out / "meta" / "val.jsonl").read_text().splitlines()]
    assert len(metas) == 1 and abs(metas[0]["rate"] - 0.8) < 1e-9             # 640 px page -> long side 512
    poly = np.array(metas[0]["objs"][0][3:])
    assert np.allclose(poly, [80, 80, 208, 80, 208, 112, 80, 112], atol=0.5)
    merged = merge_patches([(metas[0]["name"], 0, 0.9, poly)])               # back to original pixels
    assert np.allclose(merged[0][2][0], [100, 100, 260, 100, 260, 140, 100, 140], atol=0.7)


def test_train_flags_and_oom_retry(tmp_path, monkeypatch):
    import openobb.train as trainer
    from openobb.cli import train
    calls = []

    def fake_main(argv):
        calls.append(argv)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(trainer, "main", fake_main)
    data = _dataset(tmp_path / "ds")
    train(str(data), epochs=10, batch=8, imgsz=512, project=str(tmp_path / "runs"), name="t",
          cache_dir=str(tmp_path / "cache"), workers=1)
    assert len(calls) == 2
    first, second = (" ".join(c) for c in calls)
    for flag in ("--fit", "--lsk", "--mosaic-p 1.0", "--mosaic-mode yolo", "--scale-jitter 0.5", "--translate 0.1",
                 "--mosaic-off 10", "--patience 100", "--warmup 3", "--eval-every 1", "--batch 8",
                 "--aug-iof 0.25", "--merge-iou 0.7", "--ema 0.986", "--schedule linear", "--min-lr-ratio 0.01"):
        assert flag in first, flag
    assert "--batch 4" in second                                             # halved after the OOM


def test_lr_schedules_and_ema():
    import math as _m

    import torch

    from openobb.engine import ModelEMA, lr_factor
    assert lr_factor(0, 1000, 100, schedule="linear", min_ratio=0.01) < 0.02           # warmup ramps up
    assert abs(lr_factor(500, 1000, 100, schedule="linear", min_ratio=0.01) - 0.505) < 1e-6
    assert abs(lr_factor(1000, 1000, 100, schedule="linear", min_ratio=0.01) - 0.01) < 1e-9
    assert abs(lr_factor(1000, 1000, 100, schedule="cos", min_ratio=0.01) - 0.01) < 1e-9
    assert lr_factor(400, 1000, 100) == 1.0                                              # flatcos unchanged
    net = torch.nn.Linear(3, 2)
    ema = ModelEMA(net, decay=0.9, warmups=1)
    w0 = ema.module.weight.clone()
    with torch.no_grad():
        net.weight.add_(1.0)
    ema.update(net)
    d = 0.9 * (1 - _m.exp(-1))
    assert torch.allclose(ema.module.weight, d * w0 + (1 - d) * net.weight)


def test_dataset_ram_cache(tmp_path):
    from openobb.data.dota import DotaPatches
    from openobb.data.prepare import prepare
    out = prepare(str(_dataset(tmp_path / "ds")), tmp_path / "prep", size=512, fit=True, workers=1)
    a, b = DotaPatches(out, "train"), DotaPatches(out, "train", cache=True)
    assert len(b.cache) == len(b.items) and (a[0][0] == b[0][0]).all()


def test_typo_and_resume_rules(tmp_path, monkeypatch):
    import json as _json

    import openobb.train as trainer
    from openobb.cli import check_keys, train
    with pytest.raises(SystemExit, match="Similar: epochs"):
        check_keys({"epoch": 10})
    check_keys({"freeze": "backbone", "time": 1.0, "fliplr": 0.0, "rot90": False, "lr0": 1e-4})
    calls = []

    def fake_main(argv):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        (out / "weights").mkdir(parents=True, exist_ok=True)
        (out / "weights" / "last.pt").write_text("x")   # a run that stopped mid-way

    monkeypatch.setattr(trainer, "main", fake_main)
    data = _dataset(tmp_path / "ds")
    kw = dict(project=str(tmp_path / "runs"), name="r", cache_dir=str(tmp_path / "cache"), workers=1)
    train(str(data), epochs=10, batch=4, imgsz=512, **kw)
    saved = _json.loads((tmp_path / "runs" / "r" / "vrdet_args.json").read_text())
    assert saved["batch"] == 4 and saved["epochs"] == 10
    train(str(data), epochs=20, batch=8, imgsz=512, **kw)                     # default: a new run, r2
    assert "r2" in " ".join(calls[-1]) and "--epochs 20" in " ".join(calls[-1])
    train(epochs=50, batch=16, imgsz=512, patience=5, resume=True, **kw)      # newest unfinished r*: r2
    second = " ".join(calls[-1])
    assert "r2" in second and "--batch 8" in second and "--epochs 20" in second   # its saved settings win
    train(str(data), epochs=20, batch=8, imgsz=512, exist_ok=True, **kw)     # start over in the same folder
    assert "--epochs 20" in " ".join(calls[-1]) and "--batch 8" in " ".join(calls[-1])


def test_conf_thresholds():
    from openobb.predict import conf_thresholds
    assert conf_thresholds(["a", "b"], {}, 0.4) == [0.4, 0.4]
    assert conf_thresholds(["a", "b"], {"conf_thr": {"a": 0.1}}, "auto") == [0.1, 0.25]


def test_nan_predictions_do_not_crash_merge_or_eval():
    from openobb.eval.dota import evaluate, merge_patches
    sq = np.array([0, 0, 10, 0, 10, 10, 0, 10], float)
    bad = np.full(8, np.nan)
    merged = merge_patches([("a__1.0__0___0", 0, 0.9, sq), ("a__1.0__0___0", 0, 0.8, bad),
                            ("a__1.0__0___0", 0, float("nan"), sq + 30)], iou_thr=0.1)
    assert len(merged[0][1]) == 1
    gts = {"a": [(sq.tolist(), "door", False)]}
    res = evaluate({"door": (["a", "a"], np.array([0.9, 0.5]), np.stack([sq, np.full(8, np.inf)]))}, gts,
                   classes=("door",))
    assert abs(res["classes"]["door"]["AP50"] - 1.0) < 1e-9


def test_ema_repair_resyncs_non_finite():
    import torch

    from openobb.engine import ModelEMA
    net = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    ema = ModelEMA(net, decay=0.9, warmups=1)
    with torch.no_grad():
        ema.module[0].weight.fill_(float("nan"))
        ema.module[1].running_var.fill_(float("inf"))
    bad = ema.repair(net, log=lambda m: None)
    assert len(bad) == 2 and torch.isfinite(ema.module[0].weight).all()
    assert torch.equal(ema.module[0].weight, net[0].weight) and torch.isfinite(ema.module[1].running_var).all()


def test_inf_batchnorm_stats_never_become_nan():
    import torch

    from openobb.engine import ModelEMA, sanitize_batchnorm
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    with torch.no_grad():
        net[1].running_var.fill_(float("inf"))
    ema = ModelEMA(net, decay=0.9, warmups=1)
    ema.update(net)                                    # inf with inf: stays inf (lerp gave inf - inf = NaN)
    assert not torch.isnan(ema.module[1].running_var).any()
    assert sanitize_batchnorm(net, log=lambda m: None) == ["1"]
    assert torch.isfinite(net[1].running_var).all() and float(net[1].running_var.max()) == 1.0


def test_optimizer_auto_ignores_lr0(tmp_path, monkeypatch):
    import openobb.train as trainer
    from openobb.cli import train
    calls = []
    monkeypatch.setattr(trainer, "main", lambda argv: calls.append(" ".join(argv)))
    data = _dataset(tmp_path / "ds")
    kw = dict(epochs=10, batch=4, imgsz=512, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "c"), workers=1)
    train(str(data), name="a", **kw)
    assert "--lr 0.0001 " in calls[-1]                      # no lr0: measured default for VRDet-s
    train(str(data), name="c", lr0=0.001, **kw)
    assert "--lr 0.001 " in calls[-1]                       # lr0 given: used as is
    train(str(data), name="b", lr0=0.0002, optimizer="AdamW", **kw)
    assert "--lr 0.0002 " in calls[-1]
