"""Which loss term holds the angle of vertical objects away from 90 deg? One forward of a trained dense model on a
training image with its labels (BatchNorm frozen, no update), then the gradient of EACH loss term w.r.t. the angle
channel of the one-to-many head, averaged over the points that the angle loss supervises, split by the predicted
angle (horizontal ~0 / vertical ~+-pi/2). At an equilibrium the terms cancel; the one pushing against the angle loss
is the cause of the bias. Analysis only (no optimiser step).

usage: python grad_probe.py <checkpoint.pt> [n_images]
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.set_num_threads(2)
from openobb import OpenOBB  # noqa: E402
from openobb.data.dota import polys_to_obb  # noqa: E402
from openobb.data.imgio import imread  # noqa: E402
from openobb.models.openobb18_loss import OpenOBB18Loss  # noqa: E402
from openobb.predict import PAD_BGR  # noqa: E402

ROOT = Path(r"F:\Source_code\NEW_architecture\datasets\dataset_wall_color")
ck, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
net = OpenOBB(ck).model
S = 1280
net.img_size = S
crit = OpenOBB18Loss(img_size=S)
import openobb.models.openobb10_loss as _l10
_ft = _l10.frame_targets
POS = []                                         # (gt long side / thickness, gt angle, predicted angle) per positive


def _rec(pts, theta_p, g, *a, **k):
    POS.append(torch.stack([g[:, 2] / g[:, 3].clamp(min=1e-3), g[:, 4], theta_p.detach()], -1).float())
    return _ft(pts, theta_p, g, *a, **k)


_l10.frame_targets = _rec
acc = {}
for p in sorted((ROOT / "images" / "train").iterdir())[:n]:
    img = imread(str(p))
    h0, w0 = img.shape[:2]
    s = S / max(h0, w0)
    img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    x = np.full((S, S, 3), PAD_BGR, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    lab = ROOT / "labels" / "train" / f"{p.stem}.txt"
    rows = [l.split() for l in lab.read_text().splitlines() if len(l.split()) >= 9]
    polys = np.array([[float(v) for v in r[1:9]] for r in rows], np.float32) * np.tile([w0 * s, h0 * s], 4)
    obb = polys_to_obb(polys)
    obb[:, :4] /= S
    tgt = [{"labels": torch.tensor([int(r[0]) for r in rows]), "boxes": torch.from_numpy(obb).float()}]
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    xt = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255)
    out = net(xt, tgt)
    reg = out["reg"]
    losses = crit(out, tgt)
    G = int(out["class_group"].max()) + 1 if out.get("class_group") is not None else 1
    R = reg.shape[-1] // G
    th = reg.detach()[0].view(-1, G, R)[..., 0]                          # (A, G) predicted angles
    ga = torch.autograd.grad(losses["loss_angle"], reg, retain_graph=True)[0][0].view(-1, G, R)[..., 0]
    sup = ga.abs() > 0                                                      # points the angle loss supervises
    for k, v in losses.items():
        if not v.requires_grad:
            continue
        g = torch.autograd.grad(v, reg, retain_graph=True, allow_unused=True)[0]
        if g is None:
            continue
        g = g[0].view(-1, G, R)[..., 0]
        for name, sel in (("vertical", sup & (th.abs() > 1.2) & (th.abs() < 1.95)), ("horizontal", sup & (th.abs() < 0.35))):
            if sel.any():
                # sign convention: + = the term wants the angle to grow (gradient descent moves along -grad)
                push = (-g[sel] * torch.sign(th[sel] if name == "vertical" else torch.ones_like(th[sel]))).float()
                a = acc.setdefault((name, k), [])
                a.append(push)
    print(p.stem, {k: round(float(v), 4) for k, v in losses.items()}, flush=True)

P = torch.cat(POS) if POS else torch.zeros(0, 3)
lng = P[:, 0] >= 10
gv = (P[:, 1].abs() - math.pi / 2).abs() < math.radians(5)
gh = P[:, 1].abs() < math.radians(5)
for nm, k in (("long VERTICAL walls", lng & gv), ("long HORIZONTAL walls", lng & gh)):
    d = torch.rad2deg(torch.remainder(P[k, 2] - P[k, 1] + math.pi / 2, math.pi) - math.pi / 2)
    tp = torch.rad2deg(P[k, 2]).abs()
    print(f"  {nm}: |predicted angle| deg quantiles 10/50/90: "
          f"{[round(float(tp.quantile(q)), 1) for q in (0.1, 0.5, 0.9)] if len(tp) else []}")
    print(f"positives on {nm}: {int(k.sum())}; predicted - label angle median "
          f"{float(d.median()) if len(d) else float('nan'):+.2f} deg, |.| > 1 deg {float((d.abs() > 1).float().mean()) if len(d) else 0:.1%}")
print(f"\n{Path(ck).name}: mean push on |angle| per supervised point (+ = away from 0, i.e. vertical objects tilt past 90 deg)")
for name in ("vertical", "horizontal"):
    for (nm, k), v in sorted(acc.items()):
        if nm == name:
            t = torch.cat(v)
            print(f"  {name:10s} {k:22s} n {len(t):5d} mean {float(t.mean()):+.5f}  sum {float(t.sum()):+.4f}")
