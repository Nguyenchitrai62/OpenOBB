import math

import torch

from openobb.models.groups import contention_groups, grouped_candidates, parse_groups
from openobb.models.openobb3 import decode
from openobb.models.openobb12_loss import OpenOBB12Loss
from openobb.models.openobb17 import OpenOBB17
from openobb.models.openobb18 import OpenOBB19
from openobb.models.openobb18_loss import OpenOBB18Loss

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    # a wall stub (class 0) lying inside a junction box (class 2): they claim the same grid points
    return [{"labels": torch.tensor([0, 2, 1]),
             "boxes": torch.tensor([[100 / IMG, 100 / IMG, 10 / IMG, 6 / IMG, 0.0],
                                    [104 / IMG, 100 / IMG, 10 / IMG, 10 / IMG, 0.0],
                                    [0.7, 0.6, 40 / IMG, 6 / IMG, 1.2]])}]


def _items():
    """Prepared-items format: class 0 (stubs) always inside class 2 (junctions); class 1 elsewhere."""
    objs = []
    for k in range(6):
        x = 40 + 30 * k
        objs.append([0, 0, 0, x - 5, 97, x + 5, 97, x + 5, 103, x - 5, 103])
        objs.append([2, 0, 0, x - 1, 95, x + 9, 95, x + 9, 105, x - 1, 105])
        objs.append([1, 0, 0, x, 180, x + 20, 180, x + 20, 186, x, 186])
    return [{"w": 256, "h": 256, "objs": objs}]


def test_contention_groups_separate_touching_classes():
    g = contention_groups(_items(), NC, n_groups=2)
    assert g[0] != g[2]                                   # stubs and junctions go to different groups
    assert parse_groups("0,0,0", NC) is None and parse_groups([0, 1, 1], NC) == [0, 1, 1]


def test_grouped_candidates_mask_other_groups():
    cls = torch.zeros(1, 4, NC)
    boxes = torch.rand(1, 4, 2, 5)
    c, b = grouped_candidates(cls, boxes, torch.tensor([0, 1, 1]))
    assert c.shape == (1, 8, NC) and b.shape == (1, 8, 5)
    assert (c[0, :4, 1:] == -30).all() and (c[0, :4, 0] == 0).all()          # group 0 rows: class 0 only
    assert (c[0, 4:, 0] == -30).all() and torch.equal(b[0, 4:], boxes[0, :, 1])


def test_overlapping_objects_of_two_groups_both_get_positives():
    torch.manual_seed(0)
    m = OpenOBB17("s", num_classes=NC, img_size=IMG, groups=[0, 1, 1]).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    crit = OpenOBB12Loss(img_size=IMG)
    G = 2
    regs = out["reg"].view(*out["reg"].shape[:-1], G, -1)
    boxes = [decode(regs[..., g, :], out["points"], out["strides"]) for g in range(G)]
    tsc, fgs, _ = crit.group_assign(out["cls"], boxes, out["points"], out["strides"], _targets(), out["class_group"])
    both = (tsc[0, :, 0] > 0) & (tsc[0, :, 2] > 0)                        # one point, stub AND junction targets
    assert both.any() and fgs[0][0].any() and fgs[1][0].any()


def test_v19_trains_grouped_heads():
    torch.manual_seed(0)
    m = OpenOBB19("s", num_classes=NC, img_size=IMG, groups=[0, 1, 1]).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    losses = OpenOBB18Loss(img_size=IMG)(out, _targets())
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    for head in (m.head, m.head_o2o):
        g = head.reg[0][-1].weight.grad
        r = g.shape[0] // 2
        assert g[:r].abs().sum() > 0 and g[r:].abs().sum() > 0               # both groups learn


def test_grouped_model_starts_as_one_group_model():
    """A one-box checkpoint loaded into the grouped head (replicated boxes) gives the same detections."""
    torch.manual_seed(0)
    a = OpenOBB17("s", num_classes=NC, img_size=128, max_det=50).eval()
    b = OpenOBB17("s", num_classes=NC, img_size=128, max_det=50, groups=[0, 1, 1]).eval()
    sd = a.state_dict()
    with torch.no_grad():
        for k, v in b.state_dict().items():
            src = sd[k]
            if src.shape == v.shape:
                v.copy_(src)
            else:                                                            # box output rows -> group 0
                v[:src.shape[0]].copy_(src)
    b.replicate_group_boxes()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    sa = oa["pred_logits"].sigmoid().amax(-1)
    sb = ob["pred_logits"].sigmoid().amax(-1)
    ka, kb = sa > 0.05, sb > 0.05
    assert ka.sum() == kb.sum()
    assert torch.allclose(oa["pred_boxes"][ka], ob["pred_boxes"][kb], atol=1e-5)


def test_v19_registry():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb19x"] == ("v19", "x") and "v19" in KNOWN_ARCHS


import os  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.skipif(os.environ.get("OPENOBB_SLOW") != "1", reason="CPU train steps (set OPENOBB_SLOW=1)")
def test_v19_auto_groups_train_smoke(tmp_path):
    import json

    import openobb.train as trainer
    from openobb.data.prepare import prepare
    from tests.test_cli import _dataset
    out = prepare(str(_dataset(tmp_path / "ds", n_train=2)), tmp_path / "prep", size=256, fit=True, workers=1)
    trainer.main(["--data", str(out), "--out", str(tmp_path / "run"), "--arch", "v19", "--size", "s", "--img", "256",
                  "--fit", "--epochs", "1", "--batch", "1", "--max-iters", "2", "--workers", "0", "--no-pretrained",
                  "--threads", "1", "--eval-every", "100", "--eval-images", "0", "--groups", "auto", "--n-groups", "2"])
    args = json.loads((tmp_path / "run" / "args.json").read_text()) if (tmp_path / "run" / "args.json").exists() else None
    ck = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert "," in ck["args"]["groups"] and ck["args"]["arch"] == "v19"
    assert args is None or "groups" in args
