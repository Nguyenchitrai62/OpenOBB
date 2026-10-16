"""Layer inventory of an OpenOBB architecture (default size x): every module down to a depth, its class, parameter count
and the shape of its first output for a 1280 x 1280 input (run at 640 on CPU, spatial sizes reported x2).
Used to write the "kiến trúc theo lớp" appendix of docs/OPENOBBN.md.

    OMP_NUM_THREADS=1 py -3.12 research/analysis/arch_layers.py v23 --depth 3 > out.txt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(1)

NC = 7                     # Wall_Color classes
GROUPS = "1,0,2,2,2,1,0"   # its contention-aware groups (v19+)


def build(arch, size):
    kw = dict(num_classes=NC, img_size=640)
    if arch == "v1":
        from openobb.models.openobb1 import OpenOBB1
        return OpenOBB1(size, num_queries=300, lsk=True, **kw)
    if arch == "v4":
        from openobb.models.openobb1 import OpenOBB1
        return OpenOBB1(size, num_queries=300, lsk=True, dense=True, dense_queries=True, dense_v3=True,
                        backbone="dinov2_b", **kw)
    if arch == "v2":
        from openobb.models.openobb2 import OpenOBB2
        return OpenOBB2(size, **kw)
    if arch == "v3":
        from openobb.models.openobb3 import OpenOBB3
        return OpenOBB3(size, lsk=True, **kw)
    if arch == "v5":
        from openobb.models.openobb5 import OpenOBB5
        return OpenOBB5(size, lsk=True, backbone="dinov2_b", geo_cls=True, relate=True, rel_k=600, **kw)
    if arch == "v6":
        from openobb.models.openobb6 import OpenOBB6
        return OpenOBB6(size, **kw)
    if arch == "v7":
        from openobb.models.openobb7 import OpenOBB7
        return OpenOBB7(size, lsk=True, **kw)
    if arch == "v8":
        from openobb.models.openobb8 import OpenOBB8
        return OpenOBB8(size, lsk=True, rel_k=600, **kw)
    if arch == "v9":
        from openobb.models.openobb9 import OpenOBB9
        return OpenOBB9(size, rel_k=600, **kw)
    if arch == "v10":
        from openobb.models.openobb10 import OpenOBB10
        return OpenOBB10(size, lsk=True, rel_k=600, **kw)
    if arch == "v11":
        from openobb.models.openobb11 import OpenOBB11
        return OpenOBB11(size, lsk=True, rel_k=600, **kw)
    if arch in ("v12", "v13"):
        from openobb.models.openobb12 import OpenOBB12, OpenOBB13
        return (OpenOBB13 if arch == "v13" else OpenOBB12)(size, lsk=True, box="dfl", **kw)
    if arch in ("v14", "v15", "v16"):
        from openobb.models.openobb14 import OpenOBB14
        from openobb.models.openobb15 import OpenOBB15
        from openobb.models.openobb16 import OpenOBB16
        cls = {"v14": OpenOBB14, "v15": OpenOBB15, "v16": OpenOBB16}[arch]
        return cls(size, box="direct", ref_k=1500, neck="own", **kw)
    from openobb.models.openobb17 import OpenOBB17
    from openobb.models import openobb18 as m18
    cls = {"v17": OpenOBB17, "v18": m18.OpenOBB18, "v19": m18.OpenOBB19, "v20": m18.OpenOBB20, "v21": m18.OpenOBB21,
           "v22": m18.OpenOBB22, "v23": m18.OpenOBB23}.get(arch)
    if arch == "v24":
        from openobb.models.openobb24 import OpenOBB24
        cls = OpenOBB24
    groups = None if arch in ("v17", "v18") else GROUPS
    return cls(size, lsk=True, box="direct" if arch == "v22" else "dfl", ref_k=1500, groups=groups, **kw)


def shape_of(o):
    if isinstance(o, dict):
        o = next((v for v in o.values() if torch.is_tensor(v)), None)
    if isinstance(o, (list, tuple)):
        ts = [t for t in o if torch.is_tensor(t)]
        return " · ".join(shape_of(t) for t in ts[:4]) if ts else "-"
    if not torch.is_tensor(o):
        return "-"
    s = list(o.shape[1:])
    if o.dim() == 4:
        s = [s[0], s[1] * 2, s[2] * 2]
    return "×".join(map(str, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch")
    ap.add_argument("--size", default="x")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--repr", default="", help="comma list of module names to print in full")
    a = ap.parse_args()
    m = build(a.arch, a.size).eval()
    shapes = {}

    def hook(name):
        def h(mod, inp, out):
            shapes.setdefault(name, shape_of(out))
        return h
    for n, mod in m.named_modules():
        if n and n.count(".") < a.depth:
            mod.register_forward_hook(hook(n))
    with torch.no_grad():
        try:
            m(torch.zeros(1, 3, 640, 640))
        except Exception as e:                       # shapes stay partial; parameters are still exact
            print("forward failed:", type(e).__name__, e)
    tot = sum(p.numel() for p in m.parameters())
    print(f"{a.arch}-{a.size}: {tot / 1e6:.2f}M parameters ({tot})")
    for n, mod in m.named_modules():
        if not n or n.count(".") >= a.depth:
            continue
        p = sum(q.numel() for q in mod.parameters())
        print(f"{'  ' * n.count('.')}{n.split('.')[-1]:<16} {type(mod).__name__:<26} {p / 1e6:8.3f}M  {shapes.get(n, '-')}")
    for name in filter(None, a.repr.split(",")):
        print(f"\n== {name}\n{m.get_submodule(name)}")


if __name__ == "__main__":
    main()
