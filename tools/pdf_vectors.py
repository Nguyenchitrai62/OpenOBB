"""Extract vector tokens (and CAD layer ids) from the PDF pages behind an AI_Takeoff training export, in the pixel
frame of the exported page images, for training / inference with `--vectors`.

AI_Takeoff export (unzipped): dataset.json lists samples {page_number, image, ...}, the document filename and the
render `scale`; images are full pages rendered in display space (top-left origin) at that scale.

Output: {out}/{image_stem}.npz  with  tokens (N, 21) = type, 8 x (x, y) px, r, g, b, width px;  layer (N,) int
(-1 = no optional-content layer);  layer_names (L,).  See vrdet/data/vectors.py.

PDF reading uses PyMuPDF (AGPL-3.0 / commercial licence; AI_Takeoff already depends on it). VRDet itself only
consumes the token arrays, so a permissive reader (pypdfium2, pdfminer.six) can replace this script later.

python tools/pdf_vectors.py --export D:/exports/fire_doc12 --pdf-dir D:/pdfs --out D:/exports/fire_doc12/vectors
"""
import argparse
import json
from pathlib import Path

import numpy as np

def _resample(pts, n=8):
    pts = np.asarray(pts, np.float64)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if seg[-1] <= 0:
        return np.repeat(pts[:1], n, 0)
    tq = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(tq, seg, pts[:, 0]), np.interp(tq, seg, pts[:, 1])], 1)


def _bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def page_tokens(page, scale):
    """PyMuPDF page -> (tokens (N, 21), layer (N,), layer_names)."""
    m = page.rotation_matrix                     # drawings are in unrotated page space; images are display space
    A = np.array([[m.a, m.c], [m.b, m.d]], np.float64)
    t = np.array([m.e, m.f], np.float64)
    rows, layers, names = [], [], {}

    def to_px(pts):
        return (np.asarray(pts, np.float64) @ A.T + t) * scale

    for d in page.get_drawings():
        col = d.get("color") or d.get("fill") or (0, 0, 0)
        width = float(d.get("width") or 1.0) * scale
        lname = d.get("layer") or ""
        lid = names.setdefault(lname, len(names)) if lname else -1
        segs = []                                  # (type, points in page space)
        for it in d["items"]:
            op = it[0]
            if op == "l":
                segs.append((0, [(it[1].x, it[1].y), (it[2].x, it[2].y)]))
            elif op == "c":
                segs.append((4, _bezier(*[np.array([v.x, v.y]) for v in it[1:5]])))
            elif op == "re":
                r = it[1]
                c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                segs += [(0, [c[k], c[(k + 1) % 4]]) for k in range(4)]
            elif op == "qu":
                q = it[1]
                c = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                segs += [(0, [c[k], c[(k + 1) % 4]]) for k in range(4)]
        for kind, pts in segs:
            rows.append(np.r_[kind, _resample(to_px(pts)).reshape(-1), col[:3], width])
            layers.append(lid)
    tokens = np.array(rows, np.float32).reshape(-1, 21)
    return tokens, np.array(layers, np.int32), sorted(names, key=names.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="unzipped AI_Takeoff export (contains dataset.json)")
    ap.add_argument("--pdf-dir", required=True, help="folder holding the source PDF(s), matched by filename")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import fitz                                   # PyMuPDF
    exp = Path(a.export)
    man = json.loads((exp / "dataset.json").read_text(encoding="utf-8"))
    pdf = Path(a.pdf_dir) / man["document"]["filename"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        for s in man["samples"]:
            page = doc[int(s["page_number"]) - 1]
            tokens, layer, lnames = page_tokens(page, float(man["scale"]))
            stem = Path(s["image"]).stem
            np.savez_compressed(out / f"{stem}.npz", tokens=tokens, layer=layer, layer_names=np.array(lnames))
            print(f"{stem}: {len(tokens)} primitives, {len(lnames)} layers")


if __name__ == "__main__":
    main()
