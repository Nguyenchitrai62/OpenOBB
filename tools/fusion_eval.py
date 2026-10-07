"""Offline study of how to combine VRDet's decoder (sparse) and dense outputs, from a saved val_preds.npz.

python tools/fusion_eval.py runs/<id>/val_preds.npz --gt <dota1_1024>/gt/val   (gt: original labelTxt dir)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrdet.engine import fusion_sets, load_preds, merge_parallel  # noqa: E402
from vrdet.eval.dota import DOTA1_CLASSES, evaluate, load_gt_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--route", default="16,24,32,48,64,96")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    preds = load_preds(a.preds)
    gts = load_gt_dir(a.gt)
    sets = fusion_sets(preds, route=[int(r) for r in a.route.split(",")])
    rows = {}
    for name, pd in sets.items():
        merged = merge_parallel(pd, 0.1, a.workers)
        res = evaluate({DOTA1_CLASSES[c]: v for c, v in merged.items()}, gts)
        rows[name] = {"mAP50": res["mAP50"], "mAP50_95": res["mAP50_95"],
                      "per_class": {c: round(r["AP50"], 4) for c, r in res["classes"].items()}}
        print(f"{name:10s} mAP50 {100 * res['mAP50']:.2f}  mAP50:95 {100 * res['mAP50_95']:.2f}", flush=True)
    out = Path(a.preds).with_name("fusion_eval.json")
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
