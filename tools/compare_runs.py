"""Per-class AP50 comparison of finished runs (reads runs/<id>/eval_val.json).

python tools/compare_runs.py e1-yolo26s-dota-24e e1-vrdet-s-dota-24e e2-h4-dense-s ...   [--ref e1-vrdet-s-dota-24e]
Also prints fusion variants (decoder / dense / ...) when a run stored them.
"""
import argparse
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
SHORT = {"plane": "PL", "baseball-diamond": "BD", "bridge": "BR", "ground-track-field": "GTF", "small-vehicle": "SV",
         "large-vehicle": "LV", "ship": "SH", "tennis-court": "TC", "basketball-court": "BC", "storage-tank": "ST",
         "soccer-ball-field": "SBF", "roundabout": "RA", "harbor": "HA", "swimming-pool": "SP", "helicopter": "HC"}


def load(run):
    p = RUNS / run / "eval_val.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    rows = {run: ({c: v["AP50"] for c, v in r["classes"].items()}, r["mAP50"], r["mAP50_95"])}
    for k, v in r.get("fusion", {}).items():
        if k != "dec":
            rows[f"{run}:{k}"] = (v["per_class_AP50"], v["mAP50"], v["mAP50_95"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    rows = {}
    for r in a.runs:
        got = load(r)
        if got is None:
            print(f"(no eval_val.json for {r})")
            continue
        rows.update(got)
    if not rows:
        return
    classes = list(next(iter(rows.values()))[0])
    ref = rows.get(a.ref) if a.ref else None
    head = f"{'run':34s} {'mAP50':>6s} {'50:95':>6s} " + " ".join(f"{SHORT.get(c, c[:4]):>5s}" for c in classes)
    print(head)
    for name, (pc, m50, m5095) in rows.items():
        cells = []
        for c in classes:
            v = 100 * pc.get(c, 0)
            if ref is not None and name != a.ref:
                v -= 100 * ref[0].get(c, 0)
                cells.append(f"{v:+5.1f}")
            else:
                cells.append(f"{v:5.1f}")
        d = f"{100 * (m50 - ref[1]):+6.2f}" if ref is not None and name != a.ref else f"{100 * m50:6.2f}"
        d2 = f"{100 * (m5095 - ref[2]):+6.2f}" if ref is not None and name != a.ref else f"{100 * m5095:6.2f}"
        print(f"{name[:34]:34s} {d} {d2} " + " ".join(cells))


if __name__ == "__main__":
    main()
