"""Prepare a labelled OBB dataset (data.yaml + 'class x1 y1 x2 y2 x3 y3 x4 y4' normalised txt labels) as VRDet tiles.
Thin wrapper of openobb.data.prepare; `openobb train data=<data.yaml>` does the same automatically.

python tools/prepare_dataset.py --src <data.yaml or dataset folder> --out datasets/mydata [--size 1024] [--val-frac 0.15]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openobb.data.prepare import prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data.yaml or dataset folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None, help="comma-separated class names (else data.yaml)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    names = [n.strip() for n in a.names.split(",")] if a.names else None
    prepare(a.src, a.out, a.size, a.gap, a.val_frac, a.seed, names, a.workers)
    print(f"[prepare] done: {a.out}")


if __name__ == "__main__":
    main()
