import numpy as np

from openobb.eval.dota import evaluate
from openobb.eval.ultra import evaluate_ultra, probiou


def _poly(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2])


def test_probiou_identity_and_thin_leniency():
    b = np.array([[100.0, 100.0, 200.0, 3.0, 0.0]])
    assert abs(probiou(b, b)[0, 0] - 1.0) < 1e-3
    shifted = np.array([[100.0, 101.0, 200.0, 3.0, 0.0]])      # 1 px across a 3 px wall: polygon IoU 0.5
    v = probiou(b, shifted)[0, 0]
    assert 0.55 < v < 0.7


def test_perfect_and_thin_offset():
    gts = {"a": [(_poly(100, 100, 200, 3).tolist(), "wall", False), (_poly(50, 50, 20, 20).tolist(), "door", False)]}
    perfect = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 100, 200, 3)])),
               "door": (["a"], np.array([0.8]), np.stack([_poly(50, 50, 20, 20)]))}
    u = evaluate_ultra(perfect, gts, ["wall", "door"])
    assert abs(u["mAP50"] - 0.995) < 1e-6 and abs(u["mAP50_95"] - 0.995) < 1e-6   # 101-point AP tops at 0.995
    off = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 101.2, 200, 3)])),     # polygon IoU ~0.43
           "door": (["a"], np.array([0.8]), np.stack([_poly(50, 50, 20, 20)]))}
    d = evaluate(off, gts, ["wall", "door"])
    u = evaluate_ultra(off, gts, ["wall", "door"])
    assert d["classes"]["wall"]["AP50"] == 0.0 and u["classes"]["wall"]["AP50"] > 0.99    # same box, two verdicts


def test_eval_metric_switch_reports_yolo_numbers():
    """metric='yolo' puts the Ultralytics-style numbers in the main fields and keeps the DOTA ones aside."""
    import copy

    from openobb.eval.ultra import evaluate_ultra
    gts = {"a": [(_poly(100, 100, 200, 3).tolist(), "wall", False)]}
    off = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 101.2, 200, 3)]))}
    res = evaluate(off, gts, ["wall"])
    dota = copy.deepcopy(res)
    u = evaluate_ultra(off, gts, ["wall"])
    # same swap as openobb.engine.eval_dota(metric="yolo")
    res["dota"] = {"mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"]}
    res["mAP50"], res["mAP50_95"] = u["mAP50"], u["mAP50_95"]
    assert res["dota"]["mAP50"] == dota["mAP50"] == 0.0 and res["mAP50"] > 0.99


def test_confusion_matrix_counts_class_mixups():
    from openobb.eval.ultra import top_confusions
    gts = {"a": [(_poly(100, 100, 200, 6).tolist(), "wall_300", False), (_poly(50, 50, 20, 20).tolist(), "door", False)]}
    dets = {"wall": (["a"], np.array([0.9]), np.stack([_poly(100, 100, 200, 6)])),     # wall_300 called wall
            "door": (["a"], np.array([0.1]), np.stack([_poly(50, 50, 20, 20)]))}        # below conf 0.25: missed
    u = evaluate_ultra(dets, gts, ["wall", "wall_300", "door"])
    lines = top_confusions(u["confusion"])
    assert any(l.startswith("wall_300 -> wall: 1") for l in lines) and any(l.startswith("door -> background: 1")
                                                                            for l in lines)


def test_yolo_only_validation_matches_full(monkeypatch):
    """Per-epoch validation (metric='yolo', dota=False, extras=False) reports the same YOLO-ruler numbers as the full
    pass (F115), with the image / instance counts of the table and no F2 column."""
    from openobb import engine
    from openobb.eval.dota import summary_table
    rng = np.random.default_rng(0)
    classes = ["wall", "door"]
    gts, preds = {}, {"dec": []}
    for k in range(6):
        name = f"im{k}"
        g = [(_poly(rng.uniform(50, 400), rng.uniform(50, 400), rng.uniform(20, 200), 3), classes[i % 2], False)
             for i in range(8)]
        gts[name] = [(p.tolist(), c, d) for p, c, d in g]
        for p, c, _ in g:
            q = p + rng.normal(0, 0.7, 8)
            preds["dec"].append((f"{name}__1.0__0___0", classes.index(c), float(rng.uniform(0.1, 1)), q))
    class NoPatches(list):
        items = []
    monkeypatch.setattr(engine, "DotaPatches", lambda *a, **k: NoPatches())
    monkeypatch.setattr(engine, "predict_patches", lambda *a, **k: {"dec": list(preds["dec"]), "_speed": None})
    monkeypatch.setattr(engine, "prepared_fit", lambda root: False)
    monkeypatch.setattr(engine, "dataset_classes", lambda root: classes)
    monkeypatch.setattr(engine, "load_gt_dir", lambda d, keep=None: gts)
    full, _ = engine.eval_dota(None, "x", "cpu", metric="yolo", merge_workers=1, log=lambda *a: None)
    fast, _ = engine.eval_dota(None, "x", "cpu", metric="yolo", dota=False, extras=False, merge_workers=1,
                               log=lambda *a: None)
    for k in ("mAP50", "mAP50_95", "P", "R"):
        assert abs(full[k] - fast[k]) < 1e-12, k
    assert fast["n_instances"] == 48 and fast["n_images"] == 6
    assert "F2" not in summary_table(fast, classes) and "F2" in summary_table(full, classes)
    assert "confusion" not in fast["ultra"] and "confusion" in full["ultra"]
