"""`openobb train ... pretrain=coco|dota|<folders>` (openobb/auto_pretrain.py, openobb/data/download.py).
CPU, tiny, no downloads and no real training: the download / distillation / training stages are injected."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from openobb.auto_pretrain import auto_pretrain, parse_sources

torch.set_num_threads(1)


def _imgs(d, n=2):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{i}.jpg"), np.full((64, 64, 3), 255, np.uint8))
    return d


def test_parse_sources(tmp_path):
    assert parse_sources("coco") == (["coco"], [])
    assert parse_sources("DOTA, coco") == (["coco", "dota"], [])
    f = _imgs(tmp_path / "mine")
    assert parse_sources(f"coco,{f}") == (["coco"], [str(f)])
    with pytest.raises(SystemExit):
        parse_sources("cocoo")
    with pytest.raises(SystemExit):
        parse_sources("")


def _fake_distill(calls):
    def distill(model, data, val_data=None, project=None, name=None, **kw):
        calls.append(dict(model=model, data=data, val_data=val_data, **kw))
        w = Path(project) / name / "weights" / "last.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}, "args": {"arch": "v9"}}, w)
        return str(w)
    return distill


def test_folders_distill_once_then_cached(tmp_path):
    user_tr, user_va, extra = _imgs(tmp_path / "tr"), _imgs(tmp_path / "va"), _imgs(tmp_path / "pages")
    calls = []
    kw = dict(root=tmp_path / "cache", log=lambda *_: None, _distill=_fake_distill(calls))
    w1 = auto_pretrain("openobb9s", str(extra), user_tr, user_va, imgsz=1280, **kw)
    assert Path(w1).exists() and len(calls) == 1
    c = calls[0]
    assert c["teachers"] == "dfine,dinov2" and c["teacher_dfine"] == "x" and c["imgsz"] == 1024
    assert c["data"].split(",") == [str(extra), str(user_tr)] and c["val_data"] == str(user_va)
    assert c["epochs"] == 20
    w2 = auto_pretrain("openobb9s", str(extra), user_tr, user_va, imgsz=1280, **kw)
    assert w2 == w1 and len(calls) == 1                       # same settings: cached, no second distillation
    auto_pretrain("openobb9s", str(extra), user_tr, user_va, imgsz=1280, epochs=3, **kw)
    assert len(calls) == 2 and calls[1]["epochs"] == 3        # other settings: new key


def test_v6_uses_the_dinov2_teacher_and_refuses_dota(tmp_path):
    calls = []
    tr = _imgs(tmp_path / "tr")
    auto_pretrain("openobb6s", str(_imgs(tmp_path / "p")), tr, root=tmp_path / "c", log=lambda *_: None,
                  _distill=_fake_distill(calls))
    assert calls[0]["teachers"] == "dinov2" and calls[0]["val_data"] is None
    with pytest.raises(SystemExit):
        auto_pretrain("openobb6s", "dota", tr, root=tmp_path / "c", log=lambda *_: None, _distill=_fake_distill(calls))


def test_dota_distills_then_trains_on_dota(tmp_path):
    dota_dir = tmp_path / "dota1_1024"
    _imgs(dota_dir / "images" / "train")
    fake_dl = SimpleNamespace(coco_images=lambda root, log=print: _imgs(Path(root) / "coco"),
                              dota=lambda root, log=print: dota_dir)
    calls, trains = [], []

    def fake_train(**kw):
        trains.append(kw)
        best = Path(kw["project"]) / kw["name"] / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}, "args": {"arch": "v9"}}, best)
        return SimpleNamespace(best=str(best), last=str(best))

    w = auto_pretrain("openobb9x", "coco,dota", _imgs(tmp_path / "tr"), root=tmp_path / "c", log=lambda *_: None,
                      _download=fake_dl, _distill=_fake_distill(calls), _train=fake_train,
                      switches={"v8_off": "refine"})
    assert Path(w).exists() and "coco+dota" in Path(w).name
    assert calls[0]["epochs"] == 10 and str(dota_dir / "images" / "train") in calls[0]["data"]
    t = trains[0]
    assert t["data"] == str(dota_dir) and t["model"] == "openobb9x" and t["tile"] and t["epochs"] == 12
    assert t["v8_off"] == "refine" and t["weights"].endswith(".pt")
    w0 = auto_pretrain("openobb9x", "dota", _imgs(tmp_path / "tr2"), root=tmp_path / "c", det_epochs=0,
                       log=lambda *_: None, _download=fake_dl, _distill=_fake_distill(calls), _train=fake_train)
    assert len(trains) == 1 and Path(w0).exists()             # det_epochs=0: distilled core only


def test_dota_download_prepares_patches(tmp_path):
    from openobb.cli import _is_prepared
    from openobb.data.download import dota

    def fetch(raw, splits):
        src = Path(raw) / "src"
        for s in splits:
            (src / s / "images").mkdir(parents=True)
            (src / s / "labelTxt").mkdir(parents=True)
            cv2.imwrite(str(src / s / "images" / "P0001.png"), np.full((300, 400, 3), 255, np.uint8))
            (src / s / "labelTxt" / "P0001.txt").write_text("10 10 60 10 60 30 10 30 small-vehicle 0\n")
        return src
    out = dota(tmp_path, splits=("train", "val"), size=256, gap=64, workers=1, log=lambda *_: None, fetch=fetch)
    assert _is_prepared(out) and (out / ".done_train").exists() and (out / ".done_val").exists()
    assert len(list((out / "images" / "train").iterdir())) > 0
    assert dota(tmp_path, splits=("train", "val"), size=256, workers=1, log=lambda *_: None, fetch=None) == out


def test_cli_train_pretrain_rules(tmp_path, monkeypatch):
    import openobb.auto_pretrain as ap
    import openobb.train as trainer
    from openobb.cli import train
    from tests.test_cli import _dataset
    data = _dataset(tmp_path / "ds")
    common = dict(epochs=1, batch=2, imgsz=256, project=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"),
                  workers=1)
    with pytest.raises(SystemExit):                            # COCO-initialised models do not take pretrain=
        train(str(data), model="openobb8s", pretrain="coco", name="a", **common)
    seen, argv = {}, []
    monkeypatch.setattr(trainer, "main", lambda a: argv.append(a))

    def fake(model, spec, user_train, user_val=None, **kw):
        seen.update(model=model, spec=spec, train=user_train, val=user_val, **kw)
        return str(tmp_path / "w.pt")
    monkeypatch.setattr(ap, "auto_pretrain", fake)
    train(str(data), model="openobb9s", pretrain="coco", pretrain_epochs=5, v8_off="refine", name="b", **common)
    assert seen["model"] == "openobb9s" and seen["spec"] == "coco" and seen["epochs"] == 5
    assert Path(seen["train"]).name == "train" and seen["switches"]["v8_off"] == "refine"
    joined = " ".join(argv[-1])
    assert f"--weights {tmp_path / 'w.pt'}" in joined
    with pytest.raises(SystemExit):
        train(str(data), model="openobb9s", pretrain="coco", weights=str(tmp_path / "w.pt"), name="c", **common)
