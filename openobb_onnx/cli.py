"""Command line of the ONNX runtime (YOLO-style key=value arguments). The handover wheel installs it as `openobb`.

    openobb predict model=best.onnx source=pages/ [conf=0.25|auto] [iou=0.5] [device=cpu|0|trt]
                         [save=True] [save_txt=True] [save_conf=True] [classes=wall,door] [max_det=300]
    openobb info model=best.onnx
    openobb app [model=best.onnx] [port=8501]   the Streamlit test viewer (pip install streamlit)
"""
import sys


def _value(v):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "none":
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"argument '{it}' must be key=value")
        k, v = it.split("=", 1)
        out[k.strip()] = _value(v.strip())
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help") or argv[0] not in ("predict", "info", "app"):
        print(__doc__)
        return 0 if not argv or argv[0] in ("-h", "--help", "help") else 2
    mode, kv = argv[0], parse_kv(argv[1:])
    if mode == "app":
        import os
        import subprocess
        from pathlib import Path
        env = dict(os.environ)
        if kv.get("model"):
            env["OPENOBB_MODEL"] = str(Path(str(kv["model"])).resolve())
        app = Path(__file__).with_name("app.py")
        extra = ["--server.port", str(kv["port"])] if kv.get("port") else []
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *extra], env=env)
    from .model import OpenOBB
    if "model" not in kv:
        raise SystemExit(f"{mode} needs model=<file.onnx>")
    model = OpenOBB(kv.pop("model"), device=kv.pop("device", None))
    if mode == "info":
        model.info()
        return 0
    if "source" not in kv:
        raise SystemExit("predict needs source=<image, folder, glob or .txt list>")
    if isinstance(kv.get("classes"), str):
        kv["classes"] = [_value(c.strip()) for c in kv["classes"].split(",") if c.strip()]
    kv.setdefault("save", True)
    for _ in model.predict(stream=True, **kv):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
