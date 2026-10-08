import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from vrdet.data.dota import DotaPatches
from vrdet.models.vector import LayerPool

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
