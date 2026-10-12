"""Rectangular inference (YOLO letterbox auto / rect val): a whole image is padded only to multiples of 32, not to
the full square, and the networks give the same outputs as on the square-padded input on the shared area."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

torch.set_num_threads(1)

S = 320                     # model / square input size
H = 160                     # rectangular input: H x S (long side = S, as in fit mode)
PAD = torch.tensor([124, 116, 104]).view(1, 3, 1, 1) / 255.0      # PAD_BGR as RGB in [0, 1]


def _drawing(h=H, w=S, seed=1):
    g = torch.Generator().manual_seed(seed)
    img = PAD.expand(1, 3, h, w).clone()
    for _ in range(40):                         # thin dark strokes, like a CAD page
        if torch.rand(1, generator=g) < 0.5:
            y = int(torch.randint(0, h - 2, (1,), generator=g))
            x0, x1 = sorted(torch.randint(0, w, (2,), generator=g).tolist())
            img[:, :, y:y + 2, x0:x1] = 0.0
        else:
            x = int(torch.randint(0, w - 2, (1,), generator=g))
            y0, y1 = sorted(torch.randint(0, h, (2,), generator=g).tolist())
            img[:, :, y0:y1, x:x + 2] = 0.0
    sq = PAD.expand(1, 3, S, S).clone()
    sq[:, :, :h, :w] = img
    return img, sq


def _raw(m, x):
    """Pre-NMS dense outputs: class logits (A, C), regression (A, R), anchor points (A, 2) px, map shapes."""
    if hasattr(m, "encode"):
        _, feats = m.encode(x)
    else:
        feats = m.backbone(x)
        if m.lsk is not None:
            feats = [lk(f) for lk, f in zip(m.lsk, feats)]
        feats = m.encoder(feats)
    pts, _ = m.anchors(feats)
    cls, reg = m.head(feats)
    return cls[0], reg[0], pts, [tuple(f.shape[-2:]) for f in feats]


def _shared(shapes_sq, shapes_rect, strides, margin):
    """Indices (square, rect) of the anchors both inputs have, keeping rows at least margin(stride) px above the
    rect's bottom edge (the convolutions' zero padding there replaces the square input's padded content)."""
    isq, irc, osq, orc = [], [], 0, 0
    for (hs, ws), (hr, wr), s in zip(shapes_sq, shapes_rect, strides):
        for y in range(hr):
            if (y + 0.5) * s <= hr * s - margin(s):
                for x in range(wr):
                    isq.append(osq + y * ws + x)
                    irc.append(orc + y * wr + x)
        osq += hs * ws
        orc += hr * wr
    return torch.tensor(isq), torch.tensor(irc)


def _spread_head(m):
    """Random-init heads output ~constant logits / zero regression; random weights make the outputs depend on the
    features so the comparison means something."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for mod in m.head.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.copy_(torch.randn(mod.weight.shape, generator=g) / math.sqrt(mod.weight[0].numel()))


@pytest.mark.parametrize("arch", ["v3", "v7", "v8"])
def test_dense_rect_matches_square(arch):
    from openobb.models.openobb3 import OpenOBB3
    from openobb.models.openobb7 import OpenOBB7, fuse_model
    from openobb.models.openobb8 import OpenOBB8
    torch.manual_seed(0)
    m = {"v3": OpenOBB3, "v7": OpenOBB7, "v8": OpenOBB8}[arch]("s", num_classes=3, img_size=S).eval()
    fuse_model(m)
    _spread_head(m)
    rect, sq = _drawing()
    with torch.no_grad():
        cs, rs, ps, ss = _raw(m, sq)
        cr, rr, pr, sr = _raw(m, rect)
        assert sr == [(H // 8, S // 8), (H // 16, S // 16), (H // 32, S // 32)]
        # v7 / v8 class towers hold an 11-cell strip conv (+-5 cells): 7 cells off the edge (no P5 row here)
        cells = 4 if arch == "v3" else 7
        a, b = _shared(ss, sr, m.strides, margin=lambda s: max(96, cells * s))
        assert len(a) > 0.25 * len(pr)
        assert torch.equal(ps[a], pr[b])                     # same anchor pixels: boxes stay in input pixels
        for t_sq, t_r in ((cs, cr), (rs, rr)):
            err = (t_sq[a] - t_r[b]).abs().max() / t_r.std()
            assert err < 0.02, float(err)                    # global attention (AIFI) sees the padding: ~0.2%
        out = m(rect)                                        # full forward (NMS, v8 relation stage) on the rect
    ok = out["pred_logits"][0].sigmoid().amax(-1) > 0.001
    assert ok.any() and torch.isfinite(out["pred_boxes"][0][ok]).all()


def test_encoder_pos_embed_rect_equals_square_on_shared_cells():
    from openobb.models.hybrid_encoder import HybridEncoder
    enc = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=64, eval_spatial_size=[S, S])
    sq = enc.pos_embed2                                      # cached square embedding (S / 32) ** 2
    n = S // 32
    h, w = 3, n
    rect = enc.grid_pos_embed(h, w, torch.device("cpu"))
    assert rect.shape == (1, h * w, 64)
    assert torch.equal(sq.view(n, n, 64)[:h, :w], rect.view(h, w, 64))
    tall = enc.grid_pos_embed(n, 4, torch.device("cpu"))     # portrait page: crop of the columns
    assert torch.equal(sq.view(n, n, 64)[:, :4], tall.view(n, 4, 64))
    assert enc.grid_pos_embed(h, w, torch.device("cpu")) is rect          # cached per (h, w, device)
    # the square grid is unchanged (the trained weights saw build(w, h) on squares)
    assert torch.equal(enc.build_2d_sincos_position_embedding(n, n, 64, enc.pe_temperature), sq)


def test_encoder_rect_forward_uses_rect_embedding():
    from openobb.models.hybrid_encoder import HybridEncoder
    torch.manual_seed(0)
    enc = HybridEncoder(in_channels=[32, 64, 128], hidden_dim=64, eval_spatial_size=[S, S]).eval()
    feats = [torch.rand(1, c, S // s // 2, S // s) for c, s in ((32, 8), (64, 16), (128, 32))]
    with torch.no_grad():
        outs = enc(feats)                                    # used to fail: cached square embedding, rect grid
    assert [tuple(o.shape[-2:]) for o in outs] == [tuple(f.shape[-2:]) for f in feats]


def test_detr_decoder_rect_frame():
    """OpenOBB1 / 4 decoder: on a rect input, anchors are normalised by the long side N (isotropic) and equal the
    square ones on the shared cells; deformable sampling rescales to the rect maps, so sampling a rect map equals
    sampling the square map on the same cells."""
    from openobb.models.obb_decoder import OBBDFINETransformer, RotatedMSDeformableAttention
    dec = OBBDFINETransformer(num_classes=3, hidden_dim=64, num_queries=10, feat_channels=(64, 64, 64), nhead=4,
                              num_layers=1, dim_feedforward=64, num_denoising=0, eval_spatial_size=[S, S]).eval()
    sq_shapes = dec._eval_shapes()
    rect_shapes = [[H // s, S // s] for s in dec.feat_strides]
    norm, loc_scale = dec._frame(rect_shapes)
    assert norm == S and loc_scale == (1.0, S / H)
    assert dec._frame(sq_shapes) == (None, None)
    a_sq, v_sq = dec._generate_anchors(sq_shapes)
    a_r, v_r = dec._generate_anchors(rect_shapes, norm=norm)
    off_s = off_r = 0
    for (hs, ws), (hr, wr) in zip(sq_shapes, rect_shapes):
        sq_cells = a_sq[0, off_s:off_s + hs * ws].view(hs, ws, 4)[:hr, :wr]
        assert torch.allclose(sq_cells, a_r[0, off_r:off_r + hr * wr].view(hr, wr, 4), atol=1e-5)
        assert torch.equal(v_sq[0, off_s:off_s + hs * ws].view(hs, ws, 1)[:hr, :wr],
                           v_r[0, off_r:off_r + hr * wr].view(hr, wr, 1))
        off_s, off_r = off_s + hs * ws, off_r + hr * wr

    torch.manual_seed(0)
    att = RotatedMSDeformableAttention(64, 4, 3, [3, 6, 3]).eval()
    with torch.no_grad():
        att.sampling_offsets.weight.normal_(0, 0.02)
        att.attention_weights.weight.normal_(0, 0.02)
        maps_sq = [torch.rand(1, 4, 16, h, w) for h, w in sq_shapes]           # bs, heads, c, h, w
        maps_r = [v[..., :h, :w] for v, (h, w) in zip(maps_sq, rect_shapes)]

        def value(maps):                                                       # decoder layout: bs, heads, c, h*w
            return [v.reshape(1, 4, 16, -1) for v in maps]

        q = torch.rand(1, 6, 64)
        ref = torch.tensor([[0.2, 0.1, 0.05, 0.03, 0.3], [0.7, 0.15, 0.04, 0.04, -0.6], [0.5, 0.2, 0.08, 0.02, 1.2],
                            [0.1, 0.25, 0.03, 0.05, 0.0], [0.9, 0.12, 0.05, 0.05, 0.8], [0.4, 0.05, 0.02, 0.02, 0.1]])
        ref = ref.view(1, 6, 1, 5)
        o_sq = att(q, ref, value(maps_sq), sq_shapes)
        o_r = att(q, ref, value(maps_r), rect_shapes, loc_scale=loc_scale)
    assert torch.allclose(o_sq, o_r, atol=1e-5)


def test_detr_model_runs_rect():
    from openobb.models.openobb1 import OpenOBB1, postprocess
    torch.manual_seed(0)
    m = OpenOBB1("s", num_classes=3, img_size=256, num_denoising=0).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 128, 256))
        sq = m(torch.rand(1, 3, 256, 256))
    assert out["box_norm"] == 256 and "box_norm" not in sq
    s, lab, b = postprocess(out, 20, 256)
    assert torch.isfinite(b).all() and b[..., 1].max() <= 128 * 1.01 + 64      # sigmoid boxes, N = 256 px units


def test_vit_rect_pos_and_levels():
    from openobb.models.vit import DinoV2Backbone
    torch.manual_seed(0)
    bb = DinoV2Backbone("dinov2_s", (32, 64, 128)).eval()
    with torch.no_grad():
        n = math.ceil(S / 14)
        full = bb.vit._pos(n, n, torch.float32)
        part = bb.vit._pos(math.ceil(H / 14), n, torch.float32)
        d = full.shape[-1]
        assert torch.equal(full[:, :1], part[:, :1])                                  # cls token
        h = math.ceil(H / 14)
        assert torch.equal(full[0, 1:].view(n, n, d)[:h], part[0, 1:].view(h, n, d))  # top rows of the square grid
        p3, p4, p5 = bb(torch.rand(1, 3, H, S))
    assert p3.shape[-2:] == (H // 8, S // 8) and p4.shape[-2:] == (H // 16, S // 16)
    assert p5.shape[-2:] == (H // 32, S // 32)


class _Stub(nn.Module):
    """Records the network input shapes; emits one box at the image centre (normalised by img_size)."""

    def __init__(self, img_size):
        super().__init__()
        self.img_size, self.shapes = img_size, []

    def forward(self, x, ctx=None):
        self.shapes.append(tuple(x.shape))
        B = x.shape[0]
        logits = torch.tensor([[[5.0, -20.0]]]).expand(B, 1, 2)
        boxes = torch.tensor([[[0.5, 0.25, 0.1, 0.05, 0.0]]]).expand(B, 1, 5)
        return {"pred_logits": logits, "pred_boxes": boxes}


def test_predict_image_fit_pads_to_32():
    from openobb.predict import predict_image, rect_shape
    assert rect_shape(700, 1024, 1024) == (704, 1024) and rect_shape(1024, 1024, 1024) == (1024, 1024)
    assert rect_shape(999, 500, 1000) == (1000, 512)                          # capped at a non-32 size
    cpu = torch.device("cpu")
    img = np.full((1683, 2381, 3), 255, np.uint8)                              # A-size page, landscape
    m = _Stub(512)
    info = {}
    dets = predict_image(m, img, 512, 200, 8, cpu, 1, scale=512 / 2381, info=info)
    assert m.shapes == [(1, 3, 384, 512)] and info["shape"] == (1, 3, 384, 512)      # 1683 * 0.215 = 362 -> 384
    assert len(dets) == 1 and np.allclose(dets[0][3].reshape(4, 2).mean(0), [256, 128])  # pixels of the input
    m.shapes.clear()
    predict_image(m, img, 512, 200, 8, cpu, 1, scale=512 / 2381, rect=False)
    assert m.shapes == [(1, 3, 512, 512)]
    m.shapes.clear()
    predict_image(m, img[:600, :900], 256, 64, 8, cpu, 1)                      # tiled: square tiles as before
    assert len(m.shapes) == 2 and all(s[2:] == (256, 256) for s in m.shapes)


def test_rect_validation_batches(tmp_path):
    """Prepared fit data: val tiles are cropped to their valid area rounded to 32, batched by shape, and a mixed
    batch is padded (top-left) to its largest H and W."""
    import cv2

    from openobb.data.dota import PAD_BGR, DotaPatches, collate
    from openobb.data.prepare import prepare
    from openobb.engine import predict_patches, prepared_fit
    root = tmp_path / "ds"
    sizes = {"a": (400, 640), "b": (640, 300), "c": (640, 640), "d": (300, 640)}
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for k, (h, w) in sizes.items():
            img = np.full((h, w, 3), 255, np.uint8)
            cv2.rectangle(img, (40, 40), (200, 80), (0, 0, 0), 2)
            cv2.imwrite(str(root / split / "images" / f"{split}{k}.png"), img)
            poly = np.array([40 / w, 40 / h, 200 / w, 40 / h, 200 / w, 80 / h, 40 / w, 80 / h])
            (root / split / "labels" / f"{split}{k}.txt").write_text("0 " + " ".join(f"{v:.6f}" for v in poly))
    (root / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnames: [door, window]\n")
    out = prepare(str(root / "data.yaml"), tmp_path / "prep", size=256, fit=True, workers=1)
    assert prepared_fit(out) and not prepared_fit(tmp_path)
    ds = DotaPatches(out, "val", size=256, rect=True)
    shapes = {ds.items[i]["name"]: tuple(ds[i][0].shape[-2:]) for i in range(len(ds))}
    assert sorted(shapes.values()) == [(128, 256), (160, 256), (256, 128), (256, 256)]   # 400x640 -> 160x256 ...
    assert sorted(len(b) for b in ds.rect_batches(2)) == [2, 2]
    assert DotaPatches(out, "val", size=256)[0][0].shape[-2:] == (256, 256)              # square unless rect
    items = [ds[i] for i in range(len(ds))]
    imgs, _ = collate(items)
    assert imgs.shape == (4, 3, 256, 256)
    k = [i for i, it in enumerate(items) if it[0].shape[-2] == 128][0]
    assert (imgs[k, :, 128:] == torch.tensor(PAD_BGR[::-1], dtype=torch.uint8).view(3, 1, 1)).all()
    assert torch.equal(imgs[k, :, :128], items[k][0])
    m = _Stub(256)
    preds = predict_patches(m, ds, torch.device("cpu"), batch=1, workers=0, num_top=1, img_size=256)
    assert sorted(s[-2:] for s in m.shapes) == sorted(shapes.values())
    assert preds["_speed"]["rect"] and len(preds["dec"]) == 4
    assert json.loads((out / ".prepared.json").read_text())["fit"] is True
