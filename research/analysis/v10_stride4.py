import sys, numpy as np
sys.argv = ["x"]
exec(open(__file__.replace("v10_stride4.py", "v10_label_stats.py"), encoding="utf-8").read().split("for split in")[0])
data = load("train")
for stride, ms in ((8, 8.0), (4, 8.0), (4, 4.0)):
    res = {k: [0, 0, 0] for k in range(7)}   # objects, objects with 0 free points, free points
    for _, r, (H, W) in data:
        xs = (np.arange(int(np.ceil(W / stride))) + 0.5) * stride
        ys = (np.arange(int(np.ceil(H / stride))) + 0.5) * stride
        PX, PY = np.meshgrid(xs, ys)
        masks = [inside(PX, PY, g, min_side=ms, band=stride) for g in r]
        for i, g in enumerate(r):
            k = int(g[0]); other = np.zeros_like(masks[i])
            for j, g2 in enumerate(r):
                if j != i and int(g2[0]) != k:
                    other |= masks[j]
            free = (masks[i] & ~other).sum()
            res[k][0] += 1; res[k][1] += free == 0; res[k][2] += free
    print(f"stride {stride} min_side {ms}: " + "  ".join(
        f"{NAMES[k]} free/obj {v[2]/v[0]:.1f} none {v[1]/v[0]*100:.0f}%" for k, v in res.items() if k in (0, 3, 4, 5, 6)))
