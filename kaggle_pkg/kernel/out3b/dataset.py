"""Dataset loader for YOLO-OBB format (cls + 4 normalized points). MIT."""
import os
import random
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def poly_to_obb_xywha(pts):
    """pts: (4,2) normalized xy -> (xc,yc,w,h,angle_deg 0..90 long-edge)."""
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (xc, yc), (w, h), ang = rect  # ang in [-90,0)
    # convert to long-edge 0..90: ensure w>=h
    if w < h:
        w, h = h, w
        ang = ang + 90.0
    # ang now in [-90,0) with w>=h? normalize to [0,90)
    ang = -ang  # 0..90
    ang = float(np.clip(ang, 0, 90))
    return xc, yc, w, h, ang


class OBBDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, imgsz=768, augment=False, max_boxes=300):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.imgsz = imgsz
        self.augment = augment
        self.max_boxes = max_boxes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        fn = self.imgs[i]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        W0, H0 = img.size
        img = img.resize((self.imgsz, self.imgsz))
        arr = np.array(img)  # HWC RGB
        # flip augment
        flip_h = self.augment and random.random() < 0.5
        flip_v = self.augment and random.random() < 0.5
        rot90 = self.augment and random.random() < 0.25
        if flip_h:
            arr = arr[:, ::-1, :].copy()
        if flip_v:
            arr = arr[::-1, :, :].copy()
        if rot90:
            arr = np.transpose(arr, (1, 0, 2))[:, ::-1, :].copy()  # 90deg cw

        stem = os.path.splitext(fn)[0]
        lp = os.path.join(self.lbl_dir, stem + ".txt")
        boxes = []  # xc,yc,w,h,angle(0-90),cls  (normalized)
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    s = line.strip().split()
                    if len(s) < 9:
                        continue
                    cls = int(float(s[0]))
                    pts = np.array(list(map(float, s[1:9])), dtype=np.float32).reshape(4, 2)
                    # augment coords consistently
                    if flip_h:
                        pts[:, 0] = 1.0 - pts[:, 0]
                    if flip_v:
                        pts[:, 1] = 1.0 - pts[:, 1]
                    if rot90:
                        pts = pts[:, [1, 0]]
                        pts[:, 0] = 1.0 - pts[:, 0]
                        # note: flip order may break polygon order; minAreaRect is order-invariant-ish
                    xc, yc, w, h, ang = poly_to_obb_xywha(pts)
                    # clip tiny
                    if w < 1e-4 or h < 1e-4:
                        continue
                    boxes.append([xc, yc, w, h, ang, cls])
        boxes = boxes[: self.max_boxes]
        img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return img_t, torch.tensor(boxes, dtype=torch.float32) if len(boxes) else torch.zeros((0, 6)), fn


def collate_fn(batch):
    imgs, boxes, fns = zip(*batch)
    return torch.stack(imgs), list(boxes), list(fns)
