"""The deployment API mirrors the Ultralytics OBB results interface (model.predict -> r.obb.xyxyxyxy, ...)."""
import json
import math

import cv2
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from openobb.models.vrdet3 import VRDet3
    d = tmp_path_factory.mktemp("m")
    torch.manual_seed(0)
    m = VRDet3("s", num_classes=2, img_size=256)
    p = d / "best.pt"
    torch.save({"model": m.state_dict(), "args": {"arch": "v3", "size": "s", "img": 256, "lsk": True, "fit": True},
                "classes": ["wall", "door"]}, p)
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 60), (300, 70), (0, 0, 0), -1)
    cv2.imwrite(str(d / "a.png"), img)
    cv2.imwrite(str(d / "b.jpg"), img)
    return p, d, img


def test_obb_geometry_follows_yolo_conventions():
    from openobb.results import OBB
    o = OBB(torch.tensor([[100.0, 50.0, 40.0, 10.0, math.pi / 2, 0.9, 1.0]]), (200, 300))
    p = o.xyxyxyxy[0]
    assert torch.allclose(p[:, 0].sort().values, torch.tensor([95.0, 95.0, 105.0, 105.0]), atol=1e-4)
    assert torch.allclose(p[:, 1].sort().values, torch.tensor([30.0, 30.0, 70.0, 70.0]), atol=1e-4)
    assert torch.allclose(o.xyxy[0], torch.tensor([95.0, 30.0, 105.0, 70.0]), atol=1e-4)
    assert torch.allclose(o.xyxyxyxyn[0, :, 0].max(), torch.tensor(105 / 300), atol=1e-6)
    assert o.conf.item() == pytest.approx(0.9) and int(o.cls.item()) == 1 and o.id is None
    assert isinstance(o.numpy().xyxyxyxy, np.ndarray)


def test_predict_like_yolo(ckpt, tmp_path):
    from openobb import VRDet
    p, d, img = ckpt
    model = VRDet(str(p))
    assert model.names == {0: "wall", 1: "door"} and model.task == "obb"
    res = model.predict(img, conf=0.0, max_det=20, verbose=False)
    assert isinstance(res, list) and len(res) == 1
    r = res[0]
    assert r.orig_shape == (300, 400) and r.boxes is None and len(r) <= 20
    o = r.obb
    assert o.data.shape[1] == 7 and o.xywhr.shape[1] == 5 and o.xyxyxyxy.shape[1:] == (4, 2)
    assert (o.xywhr[:, 2] >= o.xywhr[:, 3] - 1e-3).all()                       # w >= h
    assert ((o.xywhr[:, 4] >= 0) & (o.xywhr[:, 4] < math.pi)).all()            # rotation in [0, pi)
    assert (o.conf[:-1] >= o.conf[1:]).all()                                    # sorted by confidence
    xs = o.cpu().numpy().xyxyxyxy
    assert isinstance(xs, np.ndarray)
    # same detections from every source type
    n = len(r)
    from PIL import Image
    pil = Image.fromarray(img[..., ::-1])
    ten = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255
    for src in (str(d / "a.png"), pil, ten, [img, str(d / "b.jpg")], str(d), str(d / "*.png")):
        out = model(src, conf=0.0, max_det=20, verbose=False)
        assert len(out[0]) == n, type(src)
    assert len(model.predict(str(d), conf=0.0, verbose=False)) == 2
    gen = model.predict(img, stream=True, verbose=False)
    assert next(iter(gen)).orig_shape == (300, 400)
    # filters
    assert all(int(c) == 1 for c in model(img, conf=0.0, classes=[1], verbose=False)[0].obb.cls)
    assert len(model(img, conf=1.01, verbose=False)[0]) == 0


def test_result_outputs(ckpt, tmp_path):
    from openobb import VRDet
    p, d, img = ckpt
    r = VRDet(str(p))(img, conf=0.0, max_det=5, verbose=False)[0]
    s = r.summary()
    assert len(s) == len(r) and {"name", "class", "confidence", "box"} <= set(s[0]) and "x4" in s[0]["box"]
    assert math.isfinite(json.loads(r.to_json(normalize=True))[0]["box"]["x1"])
    t = tmp_path / "l.txt"
    r.save_txt(t, save_conf=True)
    rows = t.read_text().splitlines()
    assert len(rows) == len(r) and len(rows[0].split()) == 10
    assert r.plot().shape == img.shape and (tmp_path / "x.jpg").exists() is False
    r.save(str(tmp_path / "x.jpg"))
    assert (tmp_path / "x.jpg").exists() and r.verbose().endswith(", ")
    out = VRDet(str(p)).predict(img, conf=0.0, max_det=3, save=True, save_txt=True, project=str(tmp_path / "runs"),
                                name="predict", verbose=False)
    assert (tmp_path / "runs" / "predict" / "image0.jpg").exists()
    assert (tmp_path / "runs" / "predict" / "labels" / "image0.txt").exists() and out[0].save_dir is not None
