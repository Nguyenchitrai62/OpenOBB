"""Inference speed-ups must not change results: BatchNorm folding (fuse_model) and the pre-NMS score filter."""
import torch

from openobb.models.openobb3 import OpenOBB3
from openobb.models.openobb7 import fuse_model

torch.set_num_threads(1)


def _model():
    torch.manual_seed(0)
    m = OpenOBB3("s", num_classes=3, img_size=128)
    for mod in m.modules():                                 # non-trivial BN statistics
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.3, 0.3)
            mod.running_var.uniform_(0.5, 1.5)
    with torch.no_grad():                                   # a head that scores something
        for o in m.head.cls:
            o[-1].bias.fill_(0.0)
    return m.eval()


def test_fuse_model_keeps_openobb3_outputs():
    m = _model()
    with torch.no_grad():
        for o in m.head.reg:                                # non-zero regression so boxes differ
            torch.nn.init.normal_(o[-1].weight, std=0.05)
    x = torch.rand(1, 3, 128, 128)

    def raw():                                          # network outputs before NMS (NMS order breaks score ties)
        return m.head(m.encoder([l(t) for l, t in zip(m.lsk, m.backbone(x))]))
    with torch.no_grad():
        a = raw()
        fuse_model(m)
        b = raw()
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())
    assert all(torch.allclose(u, v, atol=1e-4) for u, v in zip(a, b))


def test_pre_nms_score_filter_keeps_the_boxes_above_it():
    m = _model()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        lo = m(x)
        m.nms_conf = 0.3
        hi = m(x)
    def above(o, c=0.3):
        s = o["pred_logits"][0].sigmoid()
        k = s.amax(-1) >= c
        return sorted(zip(s[k].amax(-1).tolist(), [tuple(round(v, 4) for v in b) for b in o["pred_boxes"][0][k].tolist()]))
    assert above(lo) == above(hi) and len(above(lo)) > 0
