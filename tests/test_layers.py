import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from openobb.data.dota import DotaPatches
from openobb.models.vector import LayerPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab" / "data"))
from floorplancad import _elements_with_layer  # noqa: E402

torch.set_num_threads(1)


def test_svg_groups_become_layers():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><g id="a"><path d="M 0 0 L 1 1"/><path d="M 1 1 L 2 2"/></g>'
           '<g id="b"><circle cx="5" cy="5" r="1"/></g><path d="M 3 3 L 4 4"/></svg>')
    got = [(el.tag.split("}")[-1], layer) for el, layer in _elements_with_layer(ET.fromstring(svg))
           if el.tag.split("}")[-1] in ("path", "circle")]
    assert got == [("path", 0), ("path", 0), ("circle", 1), ("path", -1)]


def test_layer_pool_identity_then_shared_context():
    torch.manual_seed(0)
    lp = LayerPool(8)
    x = torch.randn(2, 6, 8)
    layer = torch.tensor([[1, 1, 2, 0, 2, 1], [1, 1, 1, 1, 0, 0]])
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
    assert torch.equal(lp(x, layer, mask), x)                       # zero-initialised: identity
    torch.nn.init.normal_(lp.mlp[-1].weight)
    d = lp(x, layer, mask) - x
    assert torch.allclose(d[0, 0], d[0, 1]) and torch.allclose(d[0, 0], d[0, 5])     # same layer, same context
    assert not torch.allclose(d[0, 0], d[0, 2])                     # other layer, other context
    assert torch.equal(d[0, 3], torch.zeros(8)) and torch.equal(d[1, 3], torch.zeros(8))   # unknown / padded


def test_dataset_keeps_layers_through_mosaic(tmp_path):
    S = 256
    for d in ("images/val", "vectors/val", "meta"):
        (tmp_path / d).mkdir(parents=True)
    import cv2
    cv2.imwrite(str(tmp_path / "images/val/d0.png"), np.full((S, S, 3), 255, np.uint8))
    rows = [np.r_[0, (np.linspace([20, 20], [60, 20], 8) / S).ravel(), 0, 0, 0, 0.5, 0, 0] for _ in range(3)]
    np.savez_compressed(tmp_path / "vectors/val/d0.npz", tokens=np.array(rows, np.float32), view=np.float32(S),
                        layer=np.array([0, 1, -1], np.int32))
    meta = {"name": "d0", "file": "d0.png", "src": "d0", "rate": 1.0, "x0": 0, "y0": 0, "w": S, "h": S,
            "img_w": S, "img_h": S, "objs": []}
    (tmp_path / "meta/val.jsonl").write_text(json.dumps(meta) + "\n")
    _, t = DotaPatches(tmp_path, "val", size=S, vectors=True)[0]
    assert t["vec"][:, 21].tolist() == [1.0, 2.0, 0.0]
    _, t = DotaPatches(tmp_path, "val", size=S, vectors=True, augment=True, mosaic_p=1.0, hsv=(0, 0, 0))[0]
    ids = t["vec"][:, 21]
    known = ids[ids > 0]
    assert len(set(known.tolist())) == 8                            # 4 copies x 2 layers, never merged


def test_snap_boxes_recovers_exact_primitive_rectangle():
    import cv2
    from openobb.models.openobb1 import snap_boxes
    from openobb.ops.obb import poly_iou
    S = 1024
    true = cv2.boxPoints(((500.0, 400.0), (120.0, 40.0), 25.0))            # the CAD object's exact rectangle
    pts = np.zeros((6, 16), np.float32)
    for k in range(4):                                                     # its 4 strokes
        a, b = true[k], true[(k + 1) % 4]
        pts[k] = (np.linspace(a, b, 8) / S).reshape(-1)
    pts[4] = (np.linspace([100, 100], [200, 120], 8) / S).reshape(-1)     # unrelated strokes
    pts[5] = (np.linspace([480, 380], [700, 380], 8) / S).reshape(-1)
    logits = np.full((3, 6), -5.0, np.float32)
    logits[1, :4] = 5.0                                                    # query 1 selected the 4 strokes
    regressed = cv2.boxPoints(((505.0, 398.0), (110.0, 46.0), 20.0)).reshape(1, 8)   # imprecise prediction
    q = np.array([1])
    out = snap_boxes(logits, pts, np.ones(6, bool), q, regressed.astype(np.float64), S)
    assert poly_iou(out.astype(np.float64), true.reshape(1, 8).astype(np.float64))[0, 0] > 0.98
    far = cv2.boxPoints(((900.0, 900.0), (50.0, 50.0), 0.0)).reshape(1, 8)  # disagreeing box is left untouched
    out2 = snap_boxes(logits, pts, np.ones(6, bool), q, far.astype(np.float64), S)
    assert np.allclose(out2, far)


def test_snap_box_gate_ignores_far_primitives():
    import cv2
    from openobb.models.openobb1 import snap_boxes
    from openobb.ops.obb import poly_iou
    S = 1024
    true = cv2.boxPoints(((500.0, 400.0), (120.0, 40.0), 25.0))
    pts = np.zeros((5, 16), np.float32)
    for k in range(4):
        a, b = true[k], true[(k + 1) % 4]
        pts[k] = (np.linspace(a, b, 8) / S).reshape(-1)
    pts[4] = (np.linspace([800, 800], [900, 820], 8) / S).reshape(-1)     # far stroke wrongly selected
    logits = np.full((2, 5), 5.0, np.float32)
    regressed = cv2.boxPoints(((505.0, 398.0), (110.0, 46.0), 20.0)).reshape(1, 8).astype(np.float64)
    q = np.array([0])
    loose = snap_boxes(logits, pts, np.ones(5, bool), q, regressed, S)
    gated = snap_boxes(logits, pts, np.ones(5, bool), q, regressed, S, box_gate=0.2)
    assert np.allclose(loose, regressed)                                 # far stroke ruins the snap -> IoU gate keeps it
    assert poly_iou(gated, true.reshape(1, 8).astype(np.float64))[0, 0] > 0.98
