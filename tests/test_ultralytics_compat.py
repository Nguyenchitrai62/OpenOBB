"""Inference and data reading follow the Ultralytics conventions (sources, outputs, angle range, dataset layout)."""
import math

import cv2
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from openobb.models.openobb3 import OpenOBB3
    d = tmp_path_factory.mktemp("m")
    torch.manual_seed(0)
    m = OpenOBB3("s", num_classes=2, img_size=256)
    p = d / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v3", "size": "s", "img": 256, "lsk": True, "fit": True},
                "classes": ["wall", "door"]}, p)
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 60), (300, 70), (0, 0, 0), -1)
    return p, img


def test_regularize_like_ultralytics():
    from openobb.results import regularize_xywhr, xywhr_to_corners
    b = torch.tensor([[10.0, 20.0, 40.0, 10.0, 2.0], [0, 0, 30, 5, -0.3], [0, 0, 30, 5, 3.5]])
    r = regularize_xywhr(b)
    assert ((r[:, 4] >= 0) & (r[:, 4] < math.pi / 2)).all()
    assert torch.allclose(r[0], torch.tensor([10.0, 20.0, 10.0, 40.0, 2.0 - math.pi / 2]), atol=1e-5)  # w, h swapped
    for i in range(len(b)):                                        # same rectangle: same set of corners
        p, q = xywhr_to_corners(b[i:i + 1])[0], xywhr_to_corners(r[i:i + 1])[0]
        assert torch.cdist(p, q).min(1).values.max() < 1e-3


def test_image_decoding_matches_training(tmp_path):
    from openobb.data.imgio import imread, imwrite
    from openobb.model import load_sources
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), np.uint8)
    p = tmp_path / "bản vẽ.png"                                   # non-ASCII path
    assert imwrite(p, img) and np.array_equal(imread(p), img)
    rgba = np.dstack([img, np.full((40, 60), 255, np.uint8)])
    cv2.imencode(".png", rgba)[1].tofile(str(tmp_path / "a.png"))
    grey = img[..., 0]
    cv2.imencode(".png", grey)[1].tofile(str(tmp_path / "g.png"))
    srcs = dict(load_sources(str(tmp_path)))
    assert all(im.shape[2] == 3 and im.dtype == np.uint8 for im in srcs.values()) and len(srcs) == 3
    (tmp_path / "list.txt").write_text(f"{tmp_path / 'a.png'}\n./g.png\n")
    assert len(load_sources(str(tmp_path / "list.txt"))) == 2
    with pytest.raises(NotImplementedError):
        load_sources(str(tmp_path / "clip.mp4"))
    with pytest.raises(NotImplementedError):
        load_sources(0)


def test_predict_outputs_like_yolo(ckpt, tmp_path):
    from openobb import OpenOBB
    p, img = ckpt
    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "page.png"), img)
    cv2.imwrite(str(src / "blank.png"), np.full((300, 400, 3), 255, np.uint8))
    model = OpenOBB(str(p))
    with pytest.raises(SyntaxError):
        model.predict(img, confidence=0.5)                         # typo: refused like YOLO
    with pytest.warns(UserWarning):
        model.predict(img, augment=True, verbose=False)
    rs = model.predict(str(src), conf=0.0, max_det=5, save=True, save_txt=True, save_crop=True, imgsz=[256, 200],
                       project=str(tmp_path / "runs"), verbose=False)
    out = tmp_path / "runs" / "predict"
    assert (out / "page.png").exists() and (out / "blank.png").exists()          # original name and suffix
    assert (out / "labels" / "page.txt").exists()
    assert any((out / "crops").rglob("page*.jpg"))
    assert rs[0].obb.data.device == model.device
    r = model.predict(img, conf=1.01, verbose=False)[0]
    r.save_txt(tmp_path / "empty.txt")
    assert not (tmp_path / "empty.txt").exists()                                  # no detections: no file
    r.update(obb=torch.tensor([[100.0, 50.0, 40.0, 10.0, 0.3, 0.9, 1.0]]))
    assert len(r) == 1 and len(r.new()) == 0 and "door" in r.verbose()
    model.predict(img, device="cpu", verbose=False)
    assert model.device == torch.device("cpu")


def test_cli_predict_like_yolo(ckpt, tmp_path, monkeypatch):
    from openobb.cli import main
    p, img = ckpt
    cv2.imwrite(str(tmp_path / "page.jpg"), img)
    monkeypatch.chdir(tmp_path)
    assert main(["predict", f"model={p}", f"source={tmp_path / 'page.jpg'}", "conf=0.0", "save_txt=True",
                 "classes=[0,1]", "verbose=False"]) == 0
    assert (tmp_path / "runs" / "obb" / "predict" / "page.jpg").exists()
    assert (tmp_path / "runs" / "obb" / "predict" / "labels" / "page.txt").exists()
    with pytest.raises(SystemExit):
        main(["predict", f"model={p}", f"source={tmp_path / 'page.jpg'}", "confi=0.1"])


def _write(img_path, lbl_path, cls=0):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), np.full((64, 64, 3), 200, np.uint8))
    lbl_path.write_text(f"{cls} 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3\n")


def test_dataset_reading_like_ultralytics(tmp_path):
    from openobb.data.prepare import find_splits, label_path
    root = tmp_path / "ds"
    # nested folders under images/train (searched recursively), duplicate file names in two folders
    _write(root / "images" / "train" / "a" / "p1.png", root / "labels" / "train" / "a" / "p1.txt")
    _write(root / "images" / "train" / "b" / "p1.png", root / "labels" / "train" / "b" / "p1.txt")
    _write(root / "images" / "extra" / "p2.png", root / "labels" / "extra" / "p2.txt")
    _write(root / "images" / "val" / "v1.png", root / "labels" / "val" / "v1.txt")
    (root / "val.txt").write_text("./images/val/v1.png\n")
    (root / "data.yaml").write_text("path: .\ntrain: [images/train, images/extra]\nval: val.txt\nnames: {0: wall}\n")
    names, splits = find_splits(root / "data.yaml")
    assert names == ["wall"] and len(splits["train"]) == 3 and len(splits["val"]) == 1
    assert label_path(splits["train"][0]).parts[-4] == "labels"
    from openobb.data.prepare import prepare
    out = prepare(root / "data.yaml", tmp_path / "prep", size=64, gap=0, workers=1, fit=True)
    assert len(list((out / "gt" / "train").glob("*.txt"))) == 3                    # p1, p1_2, p2: none overwritten
