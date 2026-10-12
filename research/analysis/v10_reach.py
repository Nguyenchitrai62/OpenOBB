import numpy as np
exec(open(__file__.replace("v10_reach.py", "v10_label_stats.py"), encoding="utf-8").read().split("for split in")[0])
data = load("train")
# share of candidate points (assign rule) whose along distances exceed the DFL reach (31 bins x stride) or whose
# thickness exceeds 31 x stride/2: their training target is truncated (clamped)
for stride in (8, 16, 32):
    out = {}
    for _, r, (H, W) in data:
        xs = (np.arange(int(np.ceil(W / stride))) + 0.5) * stride
        ys = (np.arange(int(np.ceil(H / stride))) + 0.5) * stride
        PX, PY = np.meshgrid(xs, ys)
        for g in r:
            m = inside(PX, PY, g, band=stride)
            c, s = np.cos(g[5]), np.sin(g[5])
            u = (PX[m] - g[1]) * c + (PY[m] - g[2]) * s
            reach = 31 * stride
            bad = (g[3] / 2 + np.abs(u) > reach) | (g[4] > 31 * stride / 2)
            k = NAMES[int(g[0])]
            o = out.setdefault(k, [0, 0, 0, 0])
            o[0] += m.sum(); o[1] += bad.sum(); o[2] += 1; o[3] += (m.sum() > 0 and bad.all())
    print(f"stride {stride}: " + "  ".join(f"{k} trunc {v[1]/max(v[0],1)*100:.0f}% of pts, all-trunc objs {v[3]/v[2]*100:.0f}%" for k, v in out.items() if k in ("wall", "note", "slide_door")))
