"""SAHI integration (openobb_onnx.sahi, openobb.sahi): the calls of a YOLO + SAHI service work unchanged on OpenOBB
models, results match the model's own predict, other model files still go to SAHI."""
import sys
import zipfile

import numpy as np
import pytest
import torch

pytest.importorskip("sahi")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from sahi.models.ultralytics import UltralyticsDetectionModel  # noqa: E402

from openobb.export import export  # noqa: E402
from test_onnx import _ckpt, _image  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("sahi")
    pt = _ckpt(d)
    onnx_path = export(str(pt), out=str(d / "m.onnx"), verbose=False)
    return str(pt), str(onnx_path)


def _rgb(seed=0, h=300, w=420):
    return np.ascontiguousarray(_image(h, w, seed)[..., ::-1])


def _rows(preds):
    out = []
    for p in preds:
        assert p.mask is not None and len(p.mask.segmentation) == 1 and len(p.mask.segmentation[0]) == 8
        out.append((p.category.id, round(p.score.value, 4), *np.round(np.asarray(p.mask.segmentation[0], np.float64), 1)))
    return sorted(out)


def _model_rows(model, rgb, conf):
    r = model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=conf, verbose=False)[0]
    o = r.obb
    def np_(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return sorted((int(c), round(float(s), 4), *np.round(p.reshape(-1), 1))
                  for c, s, p in zip(np_(o.cls), np_(o.conf), np_(o.xyxyxyxy).astype(np.float64)))


def test_metadata_read_without_loading_the_network(models, tmp_path):
    import onnx
    from openobb_onnx.sahi import is_model_file, onnx_metadata_keys
    _, onnx_path = models
    assert onnx_metadata_keys(onnx_path) == [p.key for p in onnx.load(onnx_path, load_external_data=False).metadata_props]
    assert is_model_file(onnx_path)
    other = tmp_path / "other.onnx"
    other.write_bytes(b"\x08\x07\x12\x03abc")                   # a tiny protobuf without metadata
    assert not is_model_file(other) and not is_model_file(tmp_path / "missing.onnx")


def test_onnx_drop_in_matches_predict(models):
    from openobb_onnx import OpenOBB
    from openobb_onnx.sahi import AutoDetectionModel, OpenOBBDetectionModel, get_prediction
    _, onnx_path = models
    had_ultralytics = "ultralytics" in sys.modules
    m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=onnx_path, confidence_threshold=0.1,
                                           device="cpu")
    assert isinstance(m, OpenOBBDetectionModel) and isinstance(m, UltralyticsDetectionModel) and m.is_obb
    if not had_ultralytics:                                    # nothing of ultralytics is imported
        assert "ultralytics" not in sys.modules
    assert m.category_mapping == {"0": "wall", "1": "door", "2": "junction"}
    img = _rgb()
    got = _rows(get_prediction(img, m).object_prediction_list)
    assert got and got == _model_rows(OpenOBB(onnx_path, device="cpu"), img, 0.1)


def test_sliced_prediction_shifts_slices_into_the_page(models):
    """Two 256 x 256 halves, two slices without overlap: SAHI's result = the model on each half, the right half shifted
    by 256 px; every object keeps its 4 corners (SAHI combines OBB slices with NMS, never polygon unions)."""
    from openobb_onnx import OpenOBB
    from openobb_onnx.sahi import AutoDetectionModel, get_sliced_prediction
    _, onnx_path = models
    m = AutoDetectionModel.from_pretrained(model_type="openobb", model_path=onnx_path, confidence_threshold=0.1)
    left, right = _rgb(1, 256, 256), _rgb(2, 256, 256)
    page = np.ascontiguousarray(np.concatenate([left, right], 1))
    res = get_sliced_prediction(page, m, slice_height=256, slice_width=256, overlap_height_ratio=0.0,
                                overlap_width_ratio=0.0, perform_standard_pred=False,
                                postprocess_match_threshold=1.01, verbose=0)    # 1.01: nothing suppressed
    got = _rows(res.object_prediction_list)
    ref_model = OpenOBB(onnx_path, device="cpu")
    want = []
    for img, dx in ((left, 0.0), (right, 256.0)):
        for c, sc, *pts in _model_rows(ref_model, img, 0.1):
            q = np.array(pts, np.float64).reshape(4, 2) + [dx, 0.0]
            x1, y1 = max(q[:, 0].min(), 0), max(q[:, 1].min(), 0)               # SAHI drops boxes with no area
            x2, y2 = min(q[:, 0].max(), 512), min(q[:, 1].max(), 256)           # inside the page
            if x1 < x2 and y1 < y2:
                q = np.minimum(q, [512.0, 256.0])                               # SAHI caps shifted corners at the
                want.append((c, sc, *np.round(q.reshape(-1), 1)))              # page's right / bottom edge
    assert got and got == sorted(want)


def test_other_models_go_to_sahi(monkeypatch, tmp_path):
    import openobb_onnx.sahi as s
    seen = {}
    monkeypatch.setattr(s._SahiAutoDetectionModel, "from_pretrained",
                        staticmethod(lambda **kw: seen.update(kw) or "sahi-model"))
    yolo = tmp_path / "yolo.onnx"
    yolo.write_bytes(b"\x08\x07")
    assert s.AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(yolo)) == "sahi-model"
    assert seen["model_type"] == "ultralytics" and seen["model_path"] == str(yolo)


def test_full_package_pt_and_onnx(models, tmp_path):
    from openobb.model import OpenOBB
    from openobb.sahi import AutoDetectionModel, OpenOBBDetectionModel, get_prediction, is_checkpoint
    pt, onnx_path = models
    assert is_checkpoint(pt)
    fake = tmp_path / "yolo.pt"                                # an Ultralytics checkpoint pickles its classes
    torch.save({"model": None, "train_args": {"arch": 1, "classes": 2}, "ultralytics": "8.4"}, fake)
    assert zipfile.is_zipfile(fake) and not is_checkpoint(fake)
    img = _rgb(3)
    for path in (pt, onnx_path):
        m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=path, confidence_threshold=0.1,
                                               device="cpu")
        assert isinstance(m, OpenOBBDetectionModel)
        got = _rows(get_prediction(img, m).object_prediction_list)
        ref = OpenOBB(path)
        if path.endswith(".pt"):
            ref = ref.to("cpu")
        assert got == _model_rows(ref, img, 0.1)
    loaded = OpenOBB(pt)                                       # a model object (model=) is recognised too
    m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model=loaded, confidence_threshold=0.1)
    assert isinstance(m, OpenOBBDetectionModel)
