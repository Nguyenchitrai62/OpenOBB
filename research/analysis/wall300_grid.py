"""Grid competition wall_300 vs junction at imgsz 1280 (long side): for each wall_300, is the grid point nearest to its
centre also the nearest point of a junction (shared cell), at stride 8 (P3) and stride 4 (P2), at 1280 and 2560 input."""
import sys
from pathlib import Path

import numpy as np

root = Path("datasets/dataset_wall_color")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openobb.data.imgio import imread  # noqa: E402

out = {}
for split in ("train", "val"):
    for p in sorted((root / "images" / split).glob("*.*")):
        im = imread(str(p)); h, w = im.shape[:2]
        rows = [l.split() for l in (root / "labels" / split / (p.stem + ".txt")).read_text().splitlines() if l.strip()]
        objs = [(int(r[0]), np.array(r[1:9], float).reshape(4, 2).mean(0) * [w, h]) for r in rows]
        for size in (1280, 2560):
            r = size / max(h, w)
            for s in (8, 4):
                cell = lambda c: tuple(np.floor(c * r / s).astype(int))
                jc = {cell(c) for k, c in objs if k == 6}
                key = (size, s)
                tot, shared = out.get(key, (0, 0))
                w3 = [cell(c) for k, c in objs if k == 5]
                out[key] = (tot + len(w3), shared + sum(c in jc for c in w3))
for (size, s), (t, sh) in sorted(out.items()):
    print(f"input {size}, stride {s}: wall_300 sharing its centre cell with a junction {sh}/{t} = {sh / t:.0%}")
