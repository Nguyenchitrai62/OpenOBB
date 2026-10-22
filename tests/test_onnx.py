"""ONNX export (openobb/export.py) + the torch-free runtime (openobb_onnx): same detections as the PyTorch package."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from openobb.export import ExportNet, _nms_value, export  # noqa: E402
from openobb.models.openobb18 import OpenOBB23  # noqa: E402
from openobb.models.openobb24 import OpenOBB24  # noqa: E402
from openobb.predict import load_model  # noqa: E402

torch.set_num_threads(1)
NAMES = ["wall", "door", "junction"]
G = "0,1,1"


def _ckpt(tmp_path, arch="v23", nms="0.5", head="o2m", **extra):
    """Small random model whose zero-initialised parts (refiner, snapping gains, quality head) are made active."""
    torch.manual_seed(0)
    cls = {"v23": OpenOBB23, "v24": OpenOBB24}[arch]
    m = cls("s", num_classes=3, img_size=256, max_det=300, groups=[0, 1, 1])
    with torch.no_grad():
        torch.nn.init.normal_(m.edge.mlp[-1].weight, std=0.02)
        m.snap.gain.fill_(0.03)
        m.snap.rot_gain.fill_(0.02)
        for head_ in (m.head, m.head_o2o):
            for seq in head_.cls:                                 # spread-out scores: few exact ties (whose order
                torch.nn.init.normal_(seq[-1].weight, std=0.3)    # differs by float noise between the two runtimes)
                seq[-1].bias.fill_(-1.5)
        if arch == "v24":
            torch.nn.init.normal_(m.quality.mlp[-1].weight, std=0.5)
    # ref_k above the candidate count: every candidate is refined, no cut among near-equal scores of a random model
    args = {"arch": arch, "size": "s", "img": 256, "fit": True, "groups": G, "nms_iou": nms, "infer_head": head,
            "num_top": 300, "ref_k": 100000, **extra}
    p = tmp_path / f"{arch}.pt"
    torch.save({"model": m.state_dict(), "args": args, "classes": NAMES,
                "conf_thr": {"wall": 0.3, "door": 0.25, "junction": 0.35}}, p)
    return p


def _image(h=300, w=420, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 255, np.uint8)
    for _ in range(25):
        p = rng.integers(0, [w, h, w, h])
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 0), int(rng.integers(1, 5)))
    return img


CASES = [("v23", "0.5", "o2m", {}), ("v24", "0.5", "o2m", {"q_beta": 1.0, "q_alpha": 0.5}),
         ("v23", "peak+0.5", "o2o", {}), ("v23", "none", "o2o", {})]


@pytest.mark.parametrize("arch,nms,head,extra", CASES)
def test_export_matches_torch(tmp_path, arch, nms, head, extra):
    import onnxruntime as ort
    from openobb import OpenOBB as Torch
    from openobb_onnx import OpenOBB as Onnx
    p = _ckpt(tmp_path, arch, nms, head, **extra)
    out = Path(export(str(p), verbose=False))
    raw = out.read_bytes()
    for word in (b"backbone", b"edge", b"snap", b"quality", b"field", b"neck", b"head", b"openobb", b"torch"):
        assert word not in raw, word                              # no module / package names in the file
    import onnx
    assert not [n for n in onnx.load(str(out)).graph.node if n.op_type == "BatchNormalization"]   # all folded
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = json.loads(sess.get_modelmeta().custom_metadata_map["obb"])
    assert meta["names"] == NAMES and meta["imgsz"] == 256 and meta["conf_thr"]["door"] == 0.25
    m, _, _ = load_model(str(p), torch.device("cpu"), 900)
    net = ExportNet(m, _nms_value(m), m.max_det).eval()
    assert meta["nms"] == _nms_value(m)
    for shape in ((2, 256, 288, 3), (1, 320, 256, 3)):            # dynamic batch / height / width
        x = torch.from_numpy(np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8))
        with torch.no_grad():
            ref = [v.numpy() for v in net(x)]
        got = sess.run(None, {"images": x.numpy()})
        assert got[0].shape == ref[0].shape
        np.testing.assert_allclose(got[0], ref[0], atol=1e-4)                     # sorted ranking scores
        # a random model has many near-equal scores, whose order differs by float noise: output scores are compared
        # sorted, labels / boxes where the ranking score is clearly apart from its neighbours
        np.testing.assert_allclose(np.sort(got[1], 1), np.sort(ref[1], 1), atol=1e-4)
        for k in range(shape[0]):
            gap = np.abs(np.diff(ref[0][k]))
            alone = np.concatenate([[True], gap > 1e-4]) & np.concatenate([gap > 1e-4, [True]])
            assert (got[2][k][alone] == ref[2][k][alone]).all()
            np.testing.assert_allclose(got[3][k][alone], ref[3][k][alone], atol=0.05)
    # end to end, the runtime runs and keeps the same score range (a random model scores every box almost equally,
    # so which of the tied boxes the NMS keeps is float noise; exact detections: test_real_checkpoint_end_to_end)
    img = _image()
    a = Torch(str(p)).predict(img, conf=0.1, verbose=False)[0]
    b = Onnx(str(out)).predict(img, conf=0.1, verbose=False)[0]
    assert len(b) > 0 and abs(float(a.obb.conf.max()) - float(b.obb.conf.max())) < 1e-3


def _same_detections(a, b):
    """Same boxes (corner sets, class) and scores; a box may be detected with two classes."""
    assert len(a) == len(b) > 0
    ca, cb = a.obb.xyxyxyxy.cpu().numpy(), b.obb.xyxyxyxy
    for i in range(len(a)):
        d = [np.abs(np.sort(ca[i], 0) - np.sort(cb[j], 0)).max() if int(b.obb.cls[j]) == int(a.obb.cls[i])
             else np.inf for j in range(len(b))]
        j = int(np.argmin(d))
        assert d[j] < 0.25 and abs(float(a.obb.conf[i]) - float(b.obb.conf[j])) < 1e-3


REAL = Path(__import__("os").environ.get("OPENOBB_TEST_CKPT", r"F:/Downloads/openobb23x_new.pt"))
REAL_IMAGES = sorted((Path(__file__).resolve().parents[1] / "datasets/dataset_wall_color/images/val").glob("*.png"))[:2]


@pytest.mark.skipif(not (REAL.exists() and REAL_IMAGES), reason="trained checkpoint / its val images not available")
def test_real_checkpoint_end_to_end(tmp_path):
    """A trained model: the runtime's detections are the PyTorch package's (count, class, corners, scores)."""
    from openobb import OpenOBB as Torch
    from openobb_onnx import OpenOBB as Onnx
    out = export(str(REAL), out=str(tmp_path / "m.onnx"), verbose=False)
    pt, ox = Torch(str(REAL)), Onnx(out, threads=1)
    from openobb_onnx.results import imread
    for f in REAL_IMAGES:
        img = imread(f)
        for conf in (0.25, "auto"):
            _same_detections(pt.predict(img, conf=conf, verbose=False)[0], ox.predict(img, conf=conf, verbose=False)[0])


def test_runtime_has_no_torch_or_training_imports():
    for f in (Path(__file__).resolve().parents[1] / "openobb_onnx").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
            for mod in mods:
                assert mod.split(".")[0] not in ("torch", "torchvision", "openobb", "ultralytics"), (f.name, mod)


def test_runtime_nms_matches_fast_nms():
    from openobb.models.openobb3 import fast_nms as torch_nms
    from openobb_onnx.ops import fast_nms
    rng = np.random.default_rng(0)
    n, C = 400, 3
    box = np.concatenate([rng.uniform(0, 200, (n, 2)), rng.uniform(4, 60, (n, 2)), rng.uniform(-1.5, 1.5, (n, 1))],
                         1).astype(np.float32)
    logits = rng.normal(-2, 2, (n, C)).astype(np.float32)
    lt, bt = torch_nms(torch.from_numpy(logits)[None], torch.from_numpy(box)[None], max_det=300, conf=0.05, iou=0.5)
    keep_t = (lt[0].amax(-1) > -20).sum().item()
    s = 1 / (1 + np.exp(-logits.reshape(-1)))
    order = np.argsort(-s, kind="stable")
    sc, lab, bx = fast_nms(s[order], s[order], order % C, box[order // C], conf=0.05, iou=0.5, max_det=300)
    assert len(sc) == keep_t
    np.testing.assert_allclose(np.sort(bx[:, 0]), np.sort(bt[0, :keep_t, 0].numpy()), atol=1e-4)


def _pt_style_inference(model_path, img, tmp_path):
    """Inference code as written for a .pt model (PyTorch package API): only the path changes for an .onnx."""
    from openobb import OpenOBB
    model = OpenOBB(model_path)
    names = model.names
    out = []
    for r in model.predict(img, conf=0.1, iou=None, imgsz=None, device="cpu", max_det=300, stream=True, verbose=False,
                           save_crop=True, project=str(tmp_path), name="pred", show_boxes=True, compile=False):
        xywhr = r.obb.xywhr.cpu().numpy()
        conf, cls = r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy().astype(int)
        corners = r.obb.xyxyxyxy.cpu().numpy()
        assert xywhr.shape == (len(r), 5) and corners.shape == (len(r), 4, 2) and r.boxes is None
        assert [names[int(c)] for c in cls] == [d["name"] for d in r.summary()]
        out.append((len(r), r.plot().shape, r.to_json()[:1]))
    return model, out


def test_pt_code_runs_unchanged_on_onnx(tmp_path):
    """OpenOBB("x.onnx") gives the ONNX runtime's model; code written for the .pt results runs as it is."""
    from openobb_onnx import OpenOBB as Onnx
    p = _ckpt(tmp_path)
    out = export(str(p), verbose=False)
    img = _image()
    model, res = _pt_style_inference(out, img, tmp_path)
    assert isinstance(model, Onnx) and model.device.type == "cpu" and model.args["imgsz"] == 256
    _, ref = _pt_style_inference(str(p), img, tmp_path)
    assert res[0][1] == ref[0][1] and res[0][2] == ref[0][2] == "["
    with pytest.raises(ValueError, match="exported .onnx"):
        Onnx(str(p))                                              # the runtime alone: a .pt gets a clear message
    with pytest.raises(NotImplementedError):
        model.train(data="x.yaml")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "tools" / "build_onnx_wheel.py").exists(),
                    reason="the build script is not in this copy of the repository")
def test_handover_wheel_installs_under_the_package_name(tmp_path):
    """The handover wheel installs the runtime as `openobb` (`hicasobb` in the product), class and command included."""
    import sys
    import zipfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from build_onnx_wheel import build
    whl = build(out=tmp_path)
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
        ep = z.read([n for n in names if n.endswith("entry_points.txt")][0]).decode()
        app = z.read("openobb/app.py").decode()
        init = z.read("openobb/__init__.py").decode()
    assert whl.name.startswith("openobb-") and all(n.startswith(("openobb/", "openobb-")) for n in names)
    assert "openobb = openobb.cli:main" in ep and "onnx = " not in ep               # one name, one command
    assert "from openobb import OpenOBB" in app and "openobb_onnx" not in app + init
    assert not any(n.startswith("openobb/models") or n.endswith(("train.py", "export.py")) for n in names)


def test_fuse_merges_every_repconv_and_batchnorm():
    """fuse() of the v17+ models merges the RepConv of the boundary field and of the own neck (missed before 0.25.13:
    14 three-branch blocks stayed in the deployed v23 and in its ONNX) and folds every BatchNorm; outputs unchanged."""
    from openobb.models.openobb9 import RepConv
    torch.manual_seed(0)
    m = OpenOBB23("s", num_classes=3, img_size=256, max_det=300, groups=[0, 1, 1]).eval()
    with torch.no_grad():
        for mod in m.modules():                                   # non-trivial BN statistics in every branch
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
        m.snap.gain.fill_(0.03)
        x = torch.rand(1, 3, 256, 288)
        ref = m.scored_candidates(x)
        assert any(isinstance(mod, RepConv) and mod.conv1 is not None for mod in m.modules())
        m.fuse()
        out = m.scored_candidates(x)
    assert not [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    assert all(mod.conv1 is None for mod in m.modules() if isinstance(mod, RepConv))
    torch.testing.assert_close(out[0], ref[0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out[2], ref[2], atol=0.05, rtol=1e-3)
