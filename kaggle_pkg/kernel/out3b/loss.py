"""Losses: cls BCE + box IoU + DFL + angle + topology constraint. MIT."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_dfl(distance, anchor_points, reg_max=16):
    """distance B,4*reg_max,H,W -> xyxy in feature pixels."""
    B, _, H, W = distance.shape
    d = distance.reshape(B, 4, reg_max, H, W).softmax(2)
    proj = torch.arange(reg_max, device=distance.device, dtype=d.dtype)
    dist = (d * proj.view(1, 1, -1, 1, 1)).sum(2)
    ax, ay = anchor_points  # H,W
    x1 = ax - dist[:, 0]
    y1 = ay - dist[:, 1]
    x2 = ax + dist[:, 2]
    y2 = ay + dist[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywha_to_xyxy(xc, yc, w, h):
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)


class DetectionLoss(nn.Module):
    def __init__(self, nc=7, reg_max=16, strides=(4, 8, 16, 32), topo_w=0.2):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.topo_w = topo_w
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets, imgsz):
        """preds: list[(cls B,nc,H,W),(box B,4R,H,W),(ang B,1,H,W)]
        targets: list[N,6] per image (xc,yc,w,h,ang_deg,cls) normalized 0..1.
        Returns total loss + dict.
        """
        device = preds[0][0].device
        total_cls, total_box, total_dfl, total_ang, total_topo = 0, 0, 0, 0, 0
        n_pos = 0
        for si, (cls, box, ang) in enumerate(preds):
            s = self.strides[si]
            B, _, H, W = cls.shape
            # grids in feature pixels
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32) + 0.5,
                torch.arange(W, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            # per image assignment (FCOS center sampling, size-based level)
            for bi in range(B):
                tgt = targets[bi].to(device) if len(targets[bi]) else torch.zeros((0, 6), device=device)
                # target boxes in feature pixels
                if len(tgt):
                    tx = tgt[:, 0] * (imgsz / s)
                    ty = tgt[:, 1] * (imgsz / s)
                    tw = tgt[:, 2] * (imgsz / s)
                    th = tgt[:, 3] * (imgsz / s)
                    ta = tgt[:, 4]  # 0..90
                    tc = tgt[:, 5].long()
                    # level suitability: assign by max(w,h) range
                    maxwh = torch.max(tw, th)
                    if si == 0:
                        keep = maxwh < 48
                    elif si == 1:
                        keep = (maxwh >= 32) & (maxwh < 96)
                    elif si == 2:
                        keep = (maxwh >= 64) & (maxwh < 192)
                    else:
                        keep = maxwh >= 128
                    # always keep junctions (cls 6) on P2 for tiny objects
                    keep = keep | ((tc == 6) & (si == 0))
                    idx = torch.where(keep)[0]
                else:
                    idx = torch.zeros((0,), dtype=torch.long, device=device)
                # build dense target maps
                cls_t = torch.zeros((self.nc, H, W), device=device)
                box_t = torch.zeros((4, H, W), device=device)
                ang_t = torch.zeros((1, H, W), device=device)
                pos_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
                if len(idx):
                    for j in idx:
                        cx, cy = tx[j], ty[j]
                        gx0, gy0 = int(cx.item()), int(cy.item())
                        # center 3x3 sampling
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                xx, yy = gx0 + dx, gy0 + dy
                                if 0 <= xx < W and 0 <= yy < H:
                                    # inside box check (axis approx)
                                    if abs(xx + 0.5 - cx.item()) < tw[j].item() / 2 + 1 and abs(yy + 0.5 - cy.item()) < th[j].item() / 2 + 1:
                                        pos_mask[yy, xx] = True
                                        cls_t[tc[j], yy, xx] = 1.0
                                        box_t[0, yy, xx] = (xx + 0.5) - (cx.item() - tw[j].item() / 2)
                                        box_t[1, yy, xx] = (yy + 0.5) - (cy.item() - th[j].item() / 2)
                                        box_t[2, yy, xx] = (cx.item() + tw[j].item() / 2) - (xx + 0.5)
                                        box_t[3, yy, xx] = (cy.item() + th[j].item() / 2) - (yy + 0.5)
                                        box_t[:, yy, xx].clamp_(0, self.reg_max - 0.01)
                                        ang_t[0, yy, xx] = ta[j].item() / 90.0
                # cls loss (all locations, with pos weighting)
                cls_loss = self.bce(cls[bi].unsqueeze(0), cls_t.unsqueeze(0)).mean()
                total_cls += cls_loss
                if pos_mask.sum() > 0:
                    n_pos += 1
                    # box DFL loss: regress distribution
                    pred_d = box[bi]  # 4R,H,W
                    # target distribution (two-hot)
                    tgt_d = box_t  # 4,H,W
                    tl = tgt_d.floor().long().clamp(0, self.reg_max - 1)
                    tr = (tl + 1).clamp(0, self.reg_max - 1)
                    wl = (tr.float() - tgt_d)
                    wr = 1 - wl
                    B4, R, Hp, Wp = pred_d.reshape(4, self.reg_max, H, W).shape[0], self.reg_max, H, W
                    logp = pred_d.reshape(4, self.reg_max, H, W).log_softmax(1)
                    # gather at pos
                    pm = pos_mask
                    loss_dfl = 0
                    for k in range(4):
                        lp = logp[k][:, pm].transpose(0, 1)  # Npos,R
                        l = tl[k][pm]
                        r = tr[k][pm]
                        loss_dfl = loss_dfl + (F.nll_loss(lp, l, reduction="none") * wl[k][pm] + F.nll_loss(lp, r, reduction="none") * wr[k][pm]).mean()
                    total_dfl += loss_dfl / 4
                    # IoU loss on decoded boxes (HBB approx)
                    with torch.no_grad():
                        pass
                    dec = decode_dfl(box[bi : bi + 1], (gx, gy), self.reg_max)[0]  # H,W,4
                    # build pred xyxy at pos vs gt xyxy
                    px1y1x2y2 = dec[pm]  # N,4
                    # gt xyxy in feat pixels
                    gxc = (box_t[2] - box_t[0]) / 2 + (torch.arange(W, device=device).float().unsqueeze(0) + 0.5)
                    # simpler: reconstruct from box_t distances
                    # gt corners:
                    gridx = torch.arange(W, device=device).float() + 0.5
                    gridy = torch.arange(H, device=device).float() + 0.5
                    gy2, gx2 = torch.meshgrid(gridy, gridx, indexing="ij")
                    gx1 = gx2 - box_t[0] + box_t[2] * 0  # placeholder
                    # direct: x1 = cx-w/2 etc using stored? recompute from tgt list is complex for multi-assign;
                    # use distance-IoU surrogate: L1 on distances
                    l1 = (dec - torch.stack([(gx2 - box_t[0]), (gy2 - box_t[1]), (gx2 + box_t[2]), (gy2 + box_t[3])], dim=-1))[pm].abs().mean()
                    total_box += l1 / max(1, s)
                    # angle loss
                    pa = torch.sigmoid(ang[bi])  # 1,H,W 0..1
                    total_ang += F.l1_loss(pa[:, pm], ang_t[:, pm])
                    # topology: door/window (1,2,3,4) should overlap wall (0) — soft constraint:
                    # penalize high door/window score where wall score is very low (on this level)
                    with torch.no_grad():
                        wall_prob = torch.sigmoid(cls[bi, 0]).detach()
                    for c in (1, 2, 3, 4):
                        pc = torch.sigmoid(cls[bi, c])[pm]
                        wv = wall_prob[pm]
                        # if model is confident it's door but wall prob <0.2 -> penalty
                        total_topo += ((pc * (0.2 - wv).clamp(min=0)).mean() * self.topo_w)
        n = max(1, B * len(preds))
        denom = max(1, n_pos)
        loss = total_cls / n + (total_box + total_dfl + total_ang) / denom + total_topo / denom
        return loss, {
            "cls": float(total_cls / n),
            "box": float(total_box / denom),
            "dfl": float(total_dfl / denom),
            "ang": float(total_ang / denom),
            "topo": float(total_topo / denom),
            "npos": n_pos,
        }
