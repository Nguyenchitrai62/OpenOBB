import math

import torch

from openobb.models.openobb3 import decode
from openobb.models.openobb7 import OpenOBB7
from openobb.models.openobb10_loss import assign_slots
from openobb.models.openobb12 import OpenOBB12, OpenOBB13
from openobb.models.openobb12_loss import OpenOBB12Loss

torch.set_num_threads(1)

IMG, NC = 256, 3


def _targets():
    return [{"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[0.5, 0.3, 0.7, 4 / IMG, 0.0], [0.25, 0.7, 12 / IMG, 10 / IMG, 0.3],
                                    [0.7, 0.6, 30 / IMG, 6 / IMG, 1.2]])}]


def test_v12_trains_both_heads():
    torch.manual_seed(0)
    m = OpenOBB12("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    assert {"cls", "reg", "o2o_cls", "o2o_reg", "points", "strides"} <= set(out)
    losses = OpenOBB12Loss(img_size=IMG)(out, _targets())
    assert {"loss_cls", "loss_box", "loss_o2o_cls", "loss_o2o_box"} <= set(losses)
    total = sum(losses.values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    grads = [p.grad for n, p in m.named_parameters() if n.startswith("head_o2o.") and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_one_to_one_assignment_gives_one_positive_per_object():
    torch.manual_seed(0)
    m = OpenOBB12("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    crit = OpenOBB12Loss(img_size=IMG)
    o = crit.o2o
    t = _targets()[0]
    gb = t["boxes"].float().clone()
    gb[:, :4] *= IMG
    k = (o.topk + gb[:, 2] / o.topk_len).round().clamp(max=o.topk_max).long()
    assert k.tolist() == [1, 1, 1]
    boxes = decode(out["o2o_reg"], out["points"], out["strides"])
    fg, gi, _ = assign_slots(out["o2o_cls"].detach().sigmoid()[0], boxes[0].detach(), out["points"], out["strides"],
                             t["labels"], gb, k, 1, o.alpha, o.beta, o.min_side, o.across, o.reach)
    per_obj = torch.bincount(gi[fg], minlength=3)
    assert per_obj.max() <= 1 and fg.sum() >= 1
    k_o2m = (crit.topk + gb[:, 2] / crit.topk_len).round().clamp(max=crit.topk_max).long()
    assert (k_o2m > 1).all()                                # the one-to-many head keeps its many positives


def test_inference_is_one_to_one_head_without_nms():
    torch.manual_seed(0)
    m = OpenOBB12("s", num_classes=NC, img_size=128, max_det=50).eval()
    with torch.no_grad():
        r = m(torch.rand(2, 3, 128, 128))
    assert r["pred_logits"].shape == (2, 50, NC) and r["pred_boxes"].shape == (2, 50, 5)
    s = r["pred_logits"].sigmoid().amax(-1)
    assert torch.all(s[:, :-1] >= s[:, 1:])                 # best first, no NMS
    assert r["pred_boxes"][..., 4].abs().max() <= math.pi / 2 + 1e-6
    m.final_nms = 0.7                                        # calibrated fallback
    with torch.no_grad():
        r2 = m(torch.rand(1, 3, 128, 128))
    assert r2["pred_logits"].shape[1] == 50


def test_v12_loads_single_head_weights_and_copies_the_head():
    torch.manual_seed(0)
    v7 = OpenOBB7("s", num_classes=NC, img_size=IMG)
    m = OpenOBB12("s", num_classes=NC, img_size=IMG)
    missing, unexpected = m.load_state_dict(v7.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("head_o2o.") for k in missing)
    m.copy_head_to_o2o()
    for (n1, p1), (n2, p2) in zip(m.head.state_dict().items(), m.head_o2o.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_v13_trains_with_p2_level_and_reach():
    torch.manual_seed(0)
    m = OpenOBB13("s", num_classes=NC, img_size=IMG).train()
    out = m(torch.rand(1, 3, IMG, IMG), _targets())
    n3 = sum((IMG // s) ** 2 for s in (16, 32, 64))
    assert out["cls"].shape[1] == n3 + (IMG // 8) ** 2 == out["o2o_cls"].shape[1]
    assert sorted(out["strides"].unique().tolist()) == [8.0, 16.0, 32.0, 64.0]
    crit = OpenOBB12Loss(img_size=IMG, off="slots")
    assert crit.reach and crit.o2o.reach
    total = sum(crit(out, _targets()).values())
    assert math.isfinite(float(total.detach()))
    total.backward()
    assert m.hires.net[0].conv.weight.grad is not None and m.p2_head.cls[0][-1].weight.grad is not None


def test_v13_core_levels_start_as_v12_at_half_resolution():
    """New modules start as no-ops on P3-P5: v13 on a 2x input == v12 on the input halved (same core weights)."""
    torch.manual_seed(0)
    v12 = OpenOBB12("s", num_classes=NC, img_size=128).eval()
    m = OpenOBB13("s", num_classes=NC, img_size=256).eval()
    missing, unexpected = m.load_state_dict(v12.state_dict(), strict=False)
    assert not unexpected and all(k.split(".")[0] in ("hires", "to_p3", "p2_lat", "p2_down", "p2_fuse", "p2_head",
                                                      "p2_head_o2o") for k in missing)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        f13 = m.features(x)
        f12 = v12.encode(torch.nn.functional.avg_pool2d(x, 2))[1]
    for a, b in zip(f13[:3], f12):
        assert torch.allclose(a, b, atol=1e-5)


def test_v13_inference_any_size_and_fused():
    torch.manual_seed(0)
    m = OpenOBB13("s", num_classes=NC, img_size=256, max_det=40).eval()
    with torch.no_grad():
        r = m(torch.rand(1, 3, 200, 136))                    # not a multiple of 64: padded inside
        m.fuse()
        r2 = m(torch.rand(1, 3, 200, 136))
    assert r["pred_logits"].shape == (1, 40, NC) and r2["pred_boxes"].shape == (1, 40, 5)


def test_trained_detector_as_pretrain_teacher(tmp_path):
    from openobb.pretrain import _Teachers
    torch.manual_seed(0)
    v12 = OpenOBB12("s", num_classes=NC, img_size=128)
    path = tmp_path / "v12.pt"
    torch.save({"model": v12.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v12", "size": "s", "img": 128}}, path)
    tch = _Teachers(["dfine"], 128, str(path), "dinov2_s", ["last"], 112, log=lambda *_: None)
    assert tch.dfine_full is not None and tch.channels["dfine"] == v12.encoder.hidden_dim
    x = torch.rand(1, 3, 128, 128)
    feats = tch(x)["dfine"]
    v12.eval()
    with torch.no_grad():
        ref = v12.encode(x)[1]
    assert len(feats) == 3
    for a, b in zip(feats, ref):
        assert a.shape == b.shape


def test_cli_names_and_known_archs():
    from openobb.cli import MODELS
    from openobb.predict import KNOWN_ARCHS
    assert MODELS["openobb12x"] == ("v12", "x") and MODELS["openobb13s"] == ("v13", "s")
    assert {"v12", "v13"} <= set(KNOWN_ARCHS)


def test_results_csv_has_o2o_loss_and_keeps_old_header(tmp_path):
    import csv
    from openobb.report import append_results
    rec = {"epoch": 1, "loss_cls": 1.0, "loss_o2o_cls": 0.5, "loss_o2o_box": 0.25}
    append_results(tmp_path, rec)
    rows = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert float(rows[0]["o2o_loss"]) == 0.75
    old = tmp_path / "old"
    old.mkdir()
    (old / "results.csv").write_text("epoch,cls_loss\n1,2.0\n")      # run started before the column existed
    append_results(old, dict(rec, epoch=2))
    last = (old / "results.csv").read_text().splitlines()[-1].split(",")
    assert len(last) == 2 and last[1] == "1.0"                       # same columns as the old header


def test_v13_imgsz_means_the_core_size():
    from openobb.predict import input_size
    assert input_size({"arch": "v13", "img": 2560}) == 2560          # trained size kept
    assert input_size({"arch": "v13", "img": 2560}, 1280) == 2560    # user imgsz = core size -> 2x input
    assert input_size({"arch": "v12", "img": 1280}, 1024) == 1024


def test_inference_head_is_a_calibrated_setting(tmp_path):
    from openobb.predict import apply_head, load_model, postprocess_options
    torch.manual_seed(0)
    m = OpenOBB12("s", num_classes=NC, img_size=128, max_det=30).eval()
    assert m.infer_head == "o2o" and ("o2m", 0.5) in postprocess_options(m)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        r1 = m(x)
        apply_head(m, "o2m")
        r2 = m(x)                                           # one-to-many head, always through NMS
    assert m.infer_head == "o2m" and not torch.equal(r1["pred_logits"], r2["pred_logits"])
    path = tmp_path / "v12.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v12", "size": "s", "img": 128, "nms_iou": 0.5, "infer_head": "o2m"}}, path)
    net, _, _ = load_model(str(path), torch.device("cpu"))
    assert net.infer_head == "o2m" and net.final_nms == 0.5


def test_direct_box_regression_trains_and_decodes():
    from openobb.models.openobb3 import DIRECT_CH
    torch.manual_seed(0)
    for C in (OpenOBB12, OpenOBB13):
        m = C("s", num_classes=NC, img_size=IMG, box="direct").train()
        assert m.head.reg[0][-1].out_channels == DIRECT_CH == m.head_o2o.reg[0][-1].out_channels
        out = m(torch.rand(1, 3, IMG, IMG), _targets())
        assert out["reg"].shape[-1] == DIRECT_CH
        losses = OpenOBB12Loss(img_size=IMG)(out, _targets())
        assert {"loss_dfl", "loss_o2o_dfl"} <= set(losses)       # L1 on the regressed values, same report column
        total = sum(losses.values())
        assert math.isfinite(float(total.detach()))
        total.backward()
        m.eval()
        with torch.no_grad():
            r = m(torch.rand(1, 3, IMG, IMG))
        assert (r["pred_boxes"][..., 2:4] > 0).all()


def test_direct_box_checkpoint_loads(tmp_path):
    from openobb.predict import load_model
    m = OpenOBB12("s", num_classes=NC, img_size=128, box="direct")
    path = tmp_path / "v12d.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c"],
                "args": {"arch": "v12", "size": "s", "img": 128, "box": "direct"}}, path)
    net, _, _ = load_model(str(path), torch.device("cpu"))
    assert net.box == "direct"
