"""Inference: decode + rotated NMS + SAHI tiling. MIT."""
import cv2
import numpy as np
import torch


def decode_level(cls, box, ang, stride, reg_max=16, conf_thr=0.25):
    B, nc, H, W = cls.shape
    prob = torch.sigmoid(cls)
    d = box.reshape(B, 4, reg_max, H, W).softmax(2)
    proj = torch.arange(reg_max, device=box.device, dtype=d.dtype)
    dist = (d * proj.view(1, 1, -1, 1, 1)).sum(2) * stride  # pixels in input
    gy, gx = torch.meshgrid(
        torch.arange(H, device=box.device) * stride + stride / 2,
        torch.arange(W, device=box.device) * stride + stride / 2,
        indexing="ij",
    )
    ang_deg = torch.sigmoid(ang) * 90.0
    out = []
    for b in range(B):
        pb = prob[b]  # nc,H,W
        conf, cl = pb.max(0)  # H,W
        m = conf > conf_thr
        if not m.any():
            out.append((np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))))
            continue
        ys, xs = torch.where(m)
        l = dist[b, 0, ys, xs]
        t = dist[b, 1, ys, xs]
        r = dist[b, 2, ys, xs]
        bb = dist[b, 3, ys, xs]
        cx = gx[ys, xs]
        cy = gy[ys, xs]
        x1 = (cx - l).cpu().numpy()
        y1 = (cy - t).cpu().numpy()
        x2 = (cx + r).cpu().numpy()
        y2 = (cy + bb).cpu().numpy()
        # to rotated rects: ((cx,cy),(w,h),angle)
        w = x2 - x1
        h = y2 - y1
        a = ang_deg[b, 0, ys, xs].cpu().numpy()
        # OpenCV angle convention for NMS: use ((cx+w/2...)) keep 0..90
        rects = [((float((x1[i] + x2[i]) / 2), float((y1[i] + y2[i]) / 2)), (float(w[i]), float(h[i])), float(-a[i])) for i in range(len(xs))]
        out.append((np.stack([x1, y1, x2, y2], 1), conf[ys, xs].cpu().numpy(), cl[ys, xs].cpu().numpy(), rects))
    return out


@torch.no_grad()
def predict(model, img_t, conf_thr=0.25, iou_thr=0.5):
    """img_t: 1,3,H,W float 0..1. Returns boxes xyxy, scores, classes, rects."""
    model.eval()
    preds = model(img_t)
    H0, W0 = img_t.shape[2:]
    all_b, all_s, all_c, all_r = [], [], [], []
    for (cls, box, ang), s in zip(preds, model.strides):
        dec = decode_level(cls, box, ang, s, model.reg_max, conf_thr)[0]
        b, sc, cl, rc = dec
        if len(b):
            all_b.append(b)
            all_s.append(sc)
            all_c.append(cl)
            all_r.extend(rc)
    if not all_b:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), []
    B = np.concatenate(all_b)
    S = np.concatenate(all_s)
    C = np.concatenate(all_c)
    # rotated NMS per class via cv2
    keep = []
    for c in np.unique(C):
        idx = np.where(C == c)[0]
        rects = [all_r[i] for i in idx]
        scores = S[idx].tolist()
        try:
            k = cv2.dnn.NMSBoxesRotated(rects, scores, conf_thr, iou_thr)
            k = np.array(k).flatten() if len(k) else []
        except Exception:
            k = np.argsort(-scores)[:300]
        keep.extend(idx[k].tolist() if len(k) else [])
    keep = sorted(keep, key=lambda i: -S[i])[:500]
    return B[keep], S[keep], C[keep], [all_r[i] for i in keep]


@torch.no_grad()
def predict_tiled(model, img_bgr, tile=1024, overlap=256, conf_thr=0.25, iou_thr=0.5):
    """SAHI-style tiling for large CAD/BIM images (e.g. 3000-8000px)."""
    H, W = img_bgr.shape[:2]
    step = tile - overlap
    boxes, scores, clses, rects = [], [], [], []
    import torch as _t

    for y in range(0, max(1, H - overlap), step):
        for x in range(0, max(1, W - overlap), step):
            x2, y2 = min(x + tile, W), min(y + tile, H)
            x1, y1 = x2 - min(tile, W), y2 - min(tile, H)
            crop = img_bgr[y1:y2, x1:x2]
            th, tw = crop.shape[:2]
            inp = cv2.resize(crop, (1024, 1024))
            t = _t.from_numpy(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            # scale boxes back
            b, s, c, r = predict(model, t.to(next(model.parameters()).device), conf_thr, iou_thr)
            if len(b):
                sx, sy = tw / 1024.0, th / 1024.0
                b[:, [0, 2]] = b[:, [0, 2]] * sx + x1
                b[:, [1, 3]] = b[:, [1, 3]] * sy + y1
                boxes.append(b)
                scores.append(s)
                clses.append(c)
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    B = np.concatenate(boxes)
    S = np.concatenate(scores)
    C = np.concatenate(clses)
    # global HBB NMS merge
    idx = cv2.dnn.NMSBoxes(B.tolist(), S.tolist(), conf_thr, iou_thr)
    idx = np.array(idx).flatten() if len(idx) else np.argsort(-S)[:500]
    return B[idx], S[idx], C[idx]
