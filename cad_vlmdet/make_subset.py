"""Build 800/200 subset from dataset_obb_train_v3.zip for arch testing. MIT."""
import argparse
import os
import random
import shutil
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=r"F:\Source_code\NEW_architecture\dataset_obb_train_v3.zip")
    ap.add_argument("--out", default=r"F:\Source_code\NEW_architecture\cad_vlmdet\subset")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_valid", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    z = zipfile.ZipFile(args.zip)
    train_imgs = sorted([n for n in z.namelist() if "/train/images/" in n and n.endswith(".png")])
    valid_imgs = sorted([n for n in z.namelist() if "/valid/images/" in n and n.endswith((".png", ".jpg"))])
    print(f"pool train={len(train_imgs)} valid={len(valid_imgs)}")
    sel_train = random.sample(train_imgs, min(args.n_train, len(train_imgs)))
    sel_valid = random.sample(valid_imgs, min(args.n_valid, len(valid_imgs)))

    for split, sel in (("train", sel_train), ("valid", sel_valid)):
        idir = os.path.join(args.out, split, "images")
        ldir = os.path.join(args.out, split, "labels")
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        # clean old
        for d in (idir, ldir):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        for n in sel:
            base = os.path.basename(n)
            stem = os.path.splitext(base)[0]
            # image
            with z.open(n) as src, open(os.path.join(idir, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            # label: train pool uses /train/labels/, valid uses /valid/labels/
            for cand in (n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",):
                try:
                    data = z.read(cand)
                    with open(os.path.join(ldir, stem + ".txt"), "wb") as dst:
                        dst.write(data)
                    break
                except KeyError:
                    open(os.path.join(ldir, stem + ".txt"), "wb").close()
    # data.yaml
    names = ["wall", "window", "door", "slide_door", "double_door", "opening", "junction"]
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"train: {os.path.join(args.out,'train','images')}\n")
        f.write(f"val: {os.path.join(args.out,'valid','images')}\n")
        f.write(f"nc: 7\nnames: {names}\n")
    print("wrote subset to", args.out)


if __name__ == "__main__":
    main()
