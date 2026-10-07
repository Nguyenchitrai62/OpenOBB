"""ONNX export for CAD-VLMDet (TensorRT-ready). MIT.
Usage: py export.py --ckpt runs/micro_nano/best.pt --imgsz 1024 --out vlmdet_nano.onnx
"""
import argparse
import torch
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="vlmdet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    m = build_model(ck.get("variant", "nano"))
    m.load_state_dict(ck["state"])
    m.eval()
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(m, x, args.out, opset_version=args.opset,
                      input_names=["images"], output_names=["p2_cls", "p2_box", "p2_ang", "p3_cls", "p3_box", "p3_ang",
                                                             "p4_cls", "p4_box", "p4_ang", "p5_cls", "p5_box", "p5_ang"],
                      dynamic_axes={"images": {0: "batch"}})
    print("exported", args.out)


if __name__ == "__main__":
    main()
