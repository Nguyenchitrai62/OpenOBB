"""Synthetic DOTA-format data: coloured rotated rectangles on a noisy background (functional test only)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrdet.eval.dota import DOTA1_CLASSES  # noqa: E402
from vrdet.ops.obb import obb2poly  # noqa: E402

COLORS = {0: (40, 40, 230), 6: (40, 220, 40), 9: (230, 60, 40)}   # plane, ship, storage-tank (BGR)


def make(root, n_train=64, n_val=16, S=384, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        (Path(root) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(root) / split / "labelTxt").mkdir(parents=True, exist_ok=True)
        for k in range(n):
            img = rng.integers(90, 140, (S, S, 3)).astype(np.uint8)
            lines = []
            boxes = []
            for _ in range(rng.integers(3, 9)):
                c = list(COLORS)[rng.integers(0, 3)]
                w, h = rng.uniform(40, 120), rng.uniform(14, 40)
                ob = np.array([rng.uniform(60, S - 60), rng.uniform(60, S - 60), w, h, rng.uniform(-np.pi / 2, np.pi / 2)])
                p = obb2poly(ob)[0]
                if any(np.hypot(*(ob[:2] - b[:2])) < 70 for b in boxes):
                    continue
                boxes.append(ob)
                cv2.fillPoly(img, [p.reshape(4, 2).astype(np.int32)], COLORS[c])
                lines.append(" ".join(f"{v:.1f}" for v in p) + f" {DOTA1_CLASSES[c]} 0")
            name = f"S{split[0]}{k:04d}"
            cv2.imwrite(str(Path(root) / split / "images" / f"{name}.png"), img)
            (Path(root) / split / "labelTxt" / f"{name}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    make(sys.argv[1], *(int(v) for v in sys.argv[2:4]))
