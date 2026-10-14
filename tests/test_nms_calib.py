"""NMS / candidate budget are parameters of the trained model (calibrated on val, saved in args, inference default)."""
import torch

from openobb.models.openobb3 import OpenOBB3
from openobb.models.openobb11 import OpenOBB11
from openobb.predict import apply_nms, load_model, merge_iou_for, parse_nms

torch.set_num_threads(1)


def test_parse_apply_merge():
    assert parse_nms("auto") is None and parse_nms(None) is None and parse_nms("none") == "none"
    assert parse_nms("0.5") == 0.5
    v3 = OpenOBB3("s", num_classes=2, img_size=128)
    assert apply_nms(v3, 0.5).nms_iou == 0.5 and apply_nms(v3, "none").nms_iou > 1
    apply_nms(v3, "auto")
    assert v3.nms_iou > 1                                     # auto leaves the current setting
    v11 = OpenOBB11("s", num_classes=2, img_size=128, rel_k=40)
    assert v11.final_nms is None
    assert apply_nms(v11, 0.6).final_nms == 0.6 and apply_nms(v11, "none").final_nms is None
    assert merge_iou_for(None, True) == 0.7 and merge_iou_for("none", True) is None
    assert merge_iou_for(0.5, True) == 0.5 and merge_iou_for(0.5, False) == 0.1


def test_load_model_uses_the_calibrated_values(tmp_path):
    torch.manual_seed(0)
    m = OpenOBB11("s", num_classes=2, img_size=128, rel_k=40)
    p = tmp_path / "v11.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b"],
                "args": {"arch": "v11", "size": "s", "img": 128, "rel_k": 40, "nms_iou": 0.6, "infer_k": 20}}, p)
    net, _, a = load_model(str(p), torch.device("cpu"))
    assert net.final_nms == 0.6 and net.refiner.k == 20
    torch.save({"model": m.state_dict(), "classes": ["a", "b"],
                "args": {"arch": "v11", "size": "s", "img": 128, "rel_k": 40, "nms_iou": "none"}}, p)
    net, _, _ = load_model(str(p), torch.device("cpu"))
    assert net.final_nms is None and net.refiner.k == 40
    from openobb.strip import strip
    torch.save({"model": m.state_dict(), "classes": ["a", "b"],
                "args": {"arch": "v11", "size": "s", "img": 128, "rel_k": 40, "nms_iou": 0.6, "infer_k": 20,
                         "data": "/root/x"}}, p)
    ck = torch.load(strip(str(p)), map_location="cpu", weights_only=True)
    assert ck["args"]["nms_iou"] == 0.6 and ck["args"]["infer_k"] == 20 and "data" not in ck["args"]


def test_train_iou_alias_reaches_the_trainer(tmp_path, monkeypatch):
    import openobb.train as trainer
    from openobb.cli import train
    from tests.test_cli import _dataset
    argv = []
    monkeypatch.setattr(trainer, "main", lambda a: argv.append(a))
    data = _dataset(tmp_path / "ds")
    train(str(data), model="openobb11s", epochs=1, batch=2, imgsz=256, iou=0.5, project=str(tmp_path / "runs"),
          name="t", cache_dir=str(tmp_path / "cache"), workers=1)
    assert "--nms-iou 0.5" in " ".join(argv[-1])
    assert "nms_iou" in trainer.build_parser().parse_args(["--data", "x", "--out", "y"]).__dict__


def test_calibration_prefers_fewer_duplicates_within_tolerance():
    from openobb.predict import choose_postprocess, dup_rate
    # OpenOBB17 (F68): o2m 0.7 fitness .5121, o2m 0.5 .5087 -> 0.5 (duplicates ~6x fewer, F69)
    tried = [(0.4889, 0.01, "o2o", "none"), (0.5121, 0.075, "o2m", 0.7), (0.5087, 0.016, "o2m", 0.5)]
    assert choose_postprocess(tried)[2:] == ("o2m", 0.5)
    # a clearly better setting still wins even with more duplicates
    tried = [(0.5200, 0.075, "o2m", 0.7), (0.5087, 0.016, "o2m", 0.5)]
    assert choose_postprocess(tried)[2:] == ("o2m", 0.7)
    # no duplicate counts (older evaluators): plain best fitness
    assert choose_postprocess([(0.50, None, None, 0.7), (0.499, None, None, 0.5)])[3] == 0.7
    u = {"confusion": {"duplicates": {"wall": 3, "junction": 7}, "detections": 200}}
    assert abs(dup_rate(u) - 0.05) < 1e-12 and dup_rate({}) is None
