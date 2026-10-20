"""OpenOBB quick test: a full-window viewer for the oriented boxes of one drawing image.

    pip install streamlit
    streamlit run app/streamlit_app.py

The whole page is the viewer (app/viewer/index.html, a two-way Streamlit component): drop or pick an image, choose the
model and the NMS threshold in its left panel. The model runs here exactly as in deployment (settings saved in best.pt:
image size, head, NMS inside the network); a new NMS value re-runs it (cached per image and value). Everything else runs
in the browser without re-running: zoom / pan, click a box for its full details, confidence threshold, show / hide
classes or single boxes, labels, TXT export of the visible boxes. The confidence filter gives the same boxes as passing
conf= to predict (a lower-scored box never removes a higher-scored one).
"""
import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openobb.torch_setup import ensure_torch  # noqa: E402

TORCH_SETUP = ensure_torch()     # before torch is imported: CUDA PyTorch on a machine with an NVIDIA GPU, else CPU
from openobb import OpenOBB  # noqa: E402
from openobb.predict import parse_nms  # noqa: E402

MODEL_PATH = r"F:\Downloads\openobb23x_new.pt"      # đường dẫn model: tuyệt đối, hoặc tương đối so với thư mục repo
MODEL_PATH = os.environ.get("OPENOBB_MODEL") or MODEL_PATH   # run_app.bat sets it (MODEL=...)

viewer = components.declare_component("obb_viewer", path=str(Path(__file__).with_name("viewer")))
BROWSER_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
HIDE_PAGE = """<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important}
[data-testid="stMainBlockContainer"], .block-container {padding: 0 !important; max-width: none !important}
</style>"""


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


@st.cache_resource(show_spinner=False)
def get_model(path):
    return OpenOBB(path)


@st.cache_data(show_spinner=False, max_entries=16)
def detect(path, data, iou):
    """predict() as deployed; iou=None: the NMS calibrated in best.pt, else that threshold ("none": no NMS)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None, 0.0
    t = time.perf_counter()
    r = get_model(path).predict(img, conf=0.05, iou=iou, max_det=1000, verbose=False)[0]
    return img, r, (time.perf_counter() - t) * 1000


def default_nms(args):
    """The NMS calibrated at the end of training (saved in best.pt) as a slider value; 1.0 = no NMS."""
    v = parse_nms(args.get("nms_iou"))
    if v is None:
        return 0.7
    if v == "none":
        return 1.0
    if isinstance(v, str):                       # "peak+0.5" -> 0.5 (the local-peak filter is not reproduced here)
        return float(v.split("+")[1]) if "+" in v else 1.0
    return float(v)


def result(name, img, r, ms):
    """Every box with conf >= 0.05 for the viewer (it filters by confidence itself)."""
    pts = _np(r.obb.xyxyxyxy).reshape(len(r), 8) if len(r) else np.zeros((0, 8))
    objs = [{"c": int(c), "s": round(float(s), 6), "p": [round(float(v), 2) for v in p],
             "r": [round(float(v), 4) for v in rb]}
            for c, s, p, rb in zip(_np(r.obb.cls), _np(r.obb.conf), pts, _np(r.obb.xywhr))]
    out = {"file": name, "w": img.shape[1], "h": img.shape[0], "ms": ms, "objs": objs,
           "names": [r.names.get(i, str(i)) for i in range(max(r.names) + 1)] if r.names else []}
    if not name.lower().endswith(BROWSER_IMAGES):   # BMP / TIFF: the browser cannot show them, send a PNG back
        out["img"] = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    return out


def main():
    st.set_page_config(page_title="OpenOBB test", page_icon="📐", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(HIDE_PAGE, unsafe_allow_html=True)
    req = st.session_state.get("viewer") or {}          # what the viewer asked for last: image, model, NMS
    local = sorted((ROOT / "models").glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
    path = Path(req.get("model") or (MODEL_PATH if (ROOT / MODEL_PATH).exists() or not local else str(local[0])))
    path = path if path.is_absolute() else ROOT / path
    args = {"id": req.get("id", 0), "model": str(path)}
    try:
        if not path.exists():
            raise FileNotFoundError(f"Không thấy model: {path}")
        model = get_model(str(path))
        nms0 = default_nms(model.args)
        args.update(nms0=nms0, info=f"{model.device}" + (" · cài xong PyTorch GPU, mở lại app để dùng GPU"
                                                          if TORCH_SETUP == "restart" else ""))
        if req.get("data"):
            iou = req.get("nms")
            run_iou = None if iou is None or abs(iou - nms0) < 1e-9 else ("none" if iou >= 1.0 else float(iou))
            img, r, ms = detect(str(path), base64.b64decode(req["data"]), run_iou)
            if img is None:
                raise ValueError(f"Không đọc được ảnh {req.get('file')}")
            args["result"] = result(req.get("file") or "image", img, r, ms)
    except Exception as e:                               # shown in the viewer's panel
        args["error"] = str(e)
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
