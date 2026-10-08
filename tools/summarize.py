"""One markdown table of every finished run (val mAP50 / mAP50:95 / latency), grouped by benchmark.

python tools/summarize.py [--md out.md]
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def info(run):
    r = json.loads((run / "eval_val.json").read_text())
    spec = ROOT / "research" / "jobs" / f"{run.name}.json"
    cmd = json.loads(spec.read_text())["cmd"] if spec.exists() else ""
    data = re.search(r"--data \S*/(\w+?)(?:_1024)?\b", cmd)
    data = {"dota1": "DOTA-v1.0", "floorplancad": "FloorPlanCAD", "dior_r": "DIOR-R"}.get(
        data.group(1) if data else "", "DOTA-v1.0" if "dota" in run.name else "?")
    if "yolo_obb" in cmd:
        m = re.search(r"--model (\S+)", cmd)
        model = (m.group(1) if m else "yolo").replace("-obb.yaml", "").replace(".yaml", "").upper()
    else:
        size = re.search(r"--size (\w)", cmd)
        flags = [f for f in ("rfs", "context", "dense", "vectors", "strip-k", "dense-queries") if f"--{f}" in cmd]
        model = f"VRDet-{(size.group(1) if size else 's').upper()}" + (" +" + "+".join(flags) if flags else "")
    lat = None
    mf = run / "metrics.jsonl"
    if mf.exists():
        for line in mf.read_text().splitlines():
            d = json.loads(line)
            lat = d.get("latency_ms_pt_fp16_bs1") or d.get("latency_ms_e2e_pt") or lat     # VRDet / YOLO probes
    fusion = {k: v["mAP50"] for k, v in r.get("fusion", {}).items() if k != "dec"}
    c = r["classes"]
    if "helicopter" in c:            # F13: helicopter (72 val instances) swings +-20 AP between runs
        ex = [v["AP50"] for k, v in c.items() if k != "helicopter"]
        fusion = dict(fusion, **{"w/o HC": sum(ex) / len(ex)})
    return data, model, r["mAP50"], r["mAP50_95"], lat, fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = {}
    for run in sorted((ROOT / "runs").iterdir()):
        if (run / "eval_val.json").exists() and "smoke" not in run.name:
            d, m, m50, m95, lat, fus = info(run)
            rows.setdefault(d, []).append((run.name, m, m50, m95, lat, fus))
    out = []
    for d, rs in rows.items():
        out += [f"### {d} (val, 24 epoch, cùng điều kiện)", "", "| Run | Model | mAP50 | mAP50:95 | Latency ms (bs1, fp16) |",
                "|---|---|---|---|---|"]
        for name, m, m50, m95, lat, fus in sorted(rs, key=lambda x: -x[2]):
            extra = "".join(f"<br>{k}: {100 * v:.2f}" for k, v in fus.items())
            out.append(f"| `{name}` | {m} | {100 * m50:.2f}{extra} | {100 * m95:.2f} | {lat:.1f} |" if lat else
                       f"| `{name}` | {m} | {100 * m50:.2f}{extra} | {100 * m95:.2f} | – |")
        out.append("")
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
