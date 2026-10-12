import sys, numpy as np
exec(open(__file__.replace("v10_slots.py", "v10_label_stats.py"), encoding="utf-8").read().split("for split in")[0])
data = load("train")
for stride in (8, 16):
    for slots in (1, 2, 3):
        res = {k: [0, 0, 0.0] for k in range(7)}
        for _, r, (H, W) in data:
            xs = (np.arange(int(np.ceil(W / stride))) + 0.5) * stride
            ys = (np.arange(int(np.ceil(H / stride))) + 0.5) * stride
            PX, PY = np.meshgrid(xs, ys)
            masks = np.array([inside(PX, PY, g, band=stride) for g in r])
            cover = masks.sum(0)
            for i, g in enumerate(r):
                k = int(g[0])
                ok = (masks[i] & (cover <= slots)).sum()
                res[k][0] += 1; res[k][1] += ok == 0; res[k][2] += ok
        print(f"stride {stride} slots {slots}: " + "  ".join(
            f"{NAMES[k]} {v[2]/v[0]:.1f}pts none {v[1]/v[0]*100:.0f}%" for k, v in res.items() if k in (0, 2, 3, 4, 5, 6)), flush=True)
