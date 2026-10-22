"""The functions rewritten in the 2026-10-22 licence audit keep their numbers.
Each is checked against an independent, deliberately naive reference written from the definition, not against the
old code: the YOLO-ruler AP and the F1 smoothing (eval/ultra.py), the HSV jitter (data/dota.py) and the CSP-ELAN block
+ RepVGG fusion (models/hybrid_encoder.py)."""
import types

import cv2
import numpy as np
import torch

from openobb.data.dota import DotaPatches
from openobb.eval.ultra import _ap, _smooth
from openobb.models.hybrid_encoder import CSPELAN, RepNCSPELAN4, VGGBlock

torch.set_num_threads(1)


def _ap_reference(rec, prec):
    """Definition: envelope by explicit loop, 101 recall levels, trapezoid rule."""
    r = [0.0] + list(rec) + [1.0]
    p = [1.0] + list(prec) + [0.0]
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    xs = [k / 100 for k in range(101)]
    ys = [float(np.interp(x, r, p)) for x in xs]
    return sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(100))


def test_ap_matches_definition():
    rng = np.random.RandomState(3)
    for _ in range(30):
        n = rng.randint(1, 300)
        tp = rng.rand(n) < rng.rand()
        tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
        npos = tpc[-1] + rng.randint(0, 40)
        rec, prec = tpc / max(npos, 1e-16), tpc / np.maximum(tpc + fpc, 1e-16)
        assert abs(_ap(rec, prec) - _ap_reference(rec, prec)) < 1e-12
    # the ruler closes the curve at (recall 1, precision 0): a perfect detector scores 0.995, not 1 (the last 1 % of
    # recall is a triangle); with no true positive the recall never leaves 0 and the AP is exactly 0
    assert abs(_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) - 0.995) < 1e-12
    assert _ap(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0


def test_smooth_is_edge_padded_moving_average():
    rng = np.random.RandomState(1)
    for n, f in [(1000, 0.1), (1000, 0.05), (37, 0.1), (2, 0.5)]:
        y = rng.rand(n)
        out = _smooth(y, f)
        width = round(n * f * 2) // 2 + 1
        half = width // 2
        ext = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
        ref = np.array([ext[i:i + width].mean() for i in range(len(ext) - width + 1)])
        assert out.shape == ref.shape
        assert np.allclose(out, ref, atol=1e-12)
    # constant curve stays constant, argmax of a peaked curve stays near the peak
    assert np.allclose(_smooth(np.full(100, 0.3), 0.1), 0.3)
    peak = np.exp(-((np.arange(200) - 120) / 10.0) ** 2)
    assert abs(int(_smooth(peak, 0.1).argmax()) - 120) <= 1


def test_hsv_jitter_matches_pixelwise_definition():
    ds = types.SimpleNamespace(hsv=(0.015, 0.5, 0.3))
    img = np.random.RandomState(0).randint(0, 256, (12, 9, 3), np.uint8)
    for seed in range(4):
        np.random.seed(seed)
        out = DotaPatches._hsv(ds, img)
        np.random.seed(seed)
        gh, gs, gv = 1 + np.random.uniform(-1, 1, 3) * np.array(ds.hsv)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ref = np.zeros_like(hsv)
        for i in range(hsv.shape[0]):
            for j in range(hsv.shape[1]):
                h, s, v = (float(c) for c in hsv[i, j])
                ref[i, j] = (int((h + (gh - 1) * 180) % 180), int(min(s * gs, 255)), int(min(v * gv, 255)))
        assert np.array_equal(out, cv2.cvtColor(ref, cv2.COLOR_HSV2BGR))
    # gains of zero: the image is returned untouched
    assert DotaPatches._hsv(types.SimpleNamespace(hsv=(0, 0, 0)), img) is img


def _randomise(m):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.normal_()
                mod.running_var.uniform_(0.5, 2.0)


def test_cspelan_forward_matches_block_definition():
    torch.manual_seed(0)
    m = CSPELAN(32, 40, 24, 16, n=2).eval()
    _randomise(m)
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out = m(x)
        stem = m.cv1(x)
        a, b = stem[:, :12], stem[:, 12:]
        s1 = m.cv2(b)
        s2 = m.cv3(s1)
        ref = m.cv4(torch.cat([a, b, s1, s2], 1))
    assert out.shape == (2, 40, 16, 16)
    assert torch.equal(out, ref)
    assert RepNCSPELAN4 is CSPELAN                      # old name still resolves
    # state-dict keys are the D-FINE checkpoint names
    assert {k.split(".")[0] for k in m.state_dict()} == {"cv1", "cv2", "cv3", "cv4"}


def test_vggblock_fusion_keeps_output():
    torch.manual_seed(1)
    blk = VGGBlock(8, 8).eval()
    _randomise(blk)
    x = torch.randn(2, 8, 10, 10)
    with torch.no_grad():
        before = blk(x)
        k, b = blk.get_equivalent_kernel_bias()
        assert k.shape == (8, 8, 3, 3) and b.shape == (8,)
        blk.convert_to_deploy()
        after = blk(x)
    assert not hasattr(blk, "conv1") and hasattr(blk, "conv")
    assert torch.allclose(before, after, atol=1e-5)


def test_csp_blocks_of_v2_v6_v9_match_definition():
    """CSP / CSPStage / RepCSP: 1x1 stem split in halves, one half through the units, every output concatenated."""
    from openobb.models.openobb2 import CSP
    from openobb.models.openobb6 import CSPStage
    from openobb.models.openobb9 import RepCSP
    torch.manual_seed(2)
    x = torch.randn(2, 16, 12, 12)
    for blk in (CSP(16, 24, n=2), CSPStage(16, 24, n=2), RepCSP(16, 24, n=2)):
        blk.eval()
        _randomise(blk)
        with torch.no_grad():
            out = blk(x)
            stem = blk.cv1(x)
            a, b = stem[:, :12], stem[:, 12:]
            parts = [a, b]
            for unit in blk.m:
                b = unit(b)
                parts.append(b)
            ref = blk.cv2(torch.cat(parts, 1))
        assert out.shape == (2, 24, 12, 12) and torch.equal(out, ref), type(blk).__name__
