"""Fetch DOTA-v1.0 (train+val, original PNG + original labelTxt with difficult flags) and cut the
1024/200 single-scale patch split. Idempotent: every stage leaves a marker file.

Source: HF dataset Last-Bullet/DOTAv1.0 (pinned revision; research/benchmark use only, DOTA terms).
Fallback: Ultralytics' DOTAv1.zip (JPEG re-encoded images + labels/*_original labelTxt).

python colab/data/get_dota.py --root /content/datasets [--rates 1.0] [--splits train,val]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "Last-Bullet/DOTAv1.0"
REV = "251502d4b016f95f2688471c32cd79d3764db058"
ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"


def fetch_hf(raw, splits):
    from huggingface_hub import snapshot_download
    pats = [f"DOTA_V1.0/{s}/*" for s in splits]
    for attempt in range(5):
        try:
            snapshot_download(REPO, repo_type="dataset", revision=REV, allow_patterns=pats, local_dir=str(raw),
                              max_workers=16)
            return raw / "DOTA_V1.0"
        except Exception as e:  # noqa: BLE001  (429 rate limits, transient network)
            print(f"[get_dota] HF attempt {attempt} failed: {str(e)[:300]}", flush=True)
            time.sleep(60 * (attempt + 1))
    raise RuntimeError("HF download failed")


def fetch_zip(raw, splits):
    z = raw / "DOTAv1.zip"
    if not z.exists():
        subprocess.run(["wget", "-q", "-O", str(z), ZIP_URL], check=True)
    subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(raw)], check=True)
    out = raw / "DOTA_from_zip"
    for s in splits:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        # the splitter reads *.png; the zip holds JPEGs: convert names via symlink-free copy of decoded images
        import cv2
        for p in (raw / "DOTAv1" / "images" / s).glob("*.jpg"):
            cv2.imwrite(str(out / s / "images" / f"{p.stem}.png"), cv2.imread(str(p)))
        shutil.copytree(raw / "DOTAv1" / "labels" / f"{s}_original", out / s / "labelTxt", dirs_exist_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/datasets")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--rates", default="1.0")
    ap.add_argument("--name", default="dota1_1024")
    ap.add_argument("--gap", type=int, default=200)
    a = ap.parse_args()
    root = Path(a.root)
    raw = root / "dota_raw"
    raw.mkdir(parents=True, exist_ok=True)
    splits = a.splits.split(",")
    t0 = time.time()
    src_marker = raw / ".fetched"
    if src_marker.exists():
        src = Path(src_marker.read_text().strip())
    else:
        try:
            src = fetch_hf(raw, splits)
        except Exception as e:  # noqa: BLE001
            print(f"[get_dota] falling back to the Ultralytics zip: {e}", flush=True)
            src = fetch_zip(raw, splits)
        for s in splits:
            n = len(list((src / s / "images").glob("*.png")))
            m = len(list((src / s / "labelTxt").glob("*.txt")))
            print(f"[get_dota] {s}: {n} images, {m} label files", flush=True)
            assert n > 0 and m > 0
        src_marker.write_text(str(src))
    print(f"[get_dota] source ready in {time.time() - t0:.0f}s: {src}", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from colab.data.split_dota import split_set  # noqa: E402
    out = root / a.name
    for s in splits:
        mk = out / f".done_{s}"
        if mk.exists():
            continue
        t1 = time.time()
        shutil.rmtree(out / "images" / s, ignore_errors=True)
        shutil.rmtree(out / "labels" / s, ignore_errors=True)
        split_set(src, out, s, size=1024, gap=a.gap, rates=[float(r) for r in a.rates.split(",")],
                  workers=os.cpu_count())
        mk.write_text("ok")
        print(f"[get_dota] split {s} in {time.time() - t1:.0f}s", flush=True)
    print(f"[get_dota] all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
