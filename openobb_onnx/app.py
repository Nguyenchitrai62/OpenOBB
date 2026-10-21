"""Streamlit test viewer for an exported .onnx model: `openobb-onnx app model=best.onnx`
(or `streamlit run app.py` with OPENOBB_MODEL=<file.onnx>).

The whole page is the viewer (viewer/index.html, a two-way Streamlit component): drop or pick an image, type the model
path and choose the NMS threshold in its left panel. The model runs as in deployment (settings saved in the .onnx);
every new image, NMS value or model runs the model again (only the last result is kept). Zoom / pan, box details, confidence filter, class toggles and
TXT export run in the browser.
"""
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from openobb_onnx import OpenOBB

HERE = Path(__file__).resolve().parent
# the viewer ships inside the wheel (viewer/); in the source tree it is the PyTorch package's app/viewer
VIEWER = next(p for p in (HERE / "viewer", HERE.parent / "app" / "viewer") if (p / "index.html").exists())
MODEL_PATH = os.environ.get("OPENOBB_MODEL") or "best.onnx"

viewer = components.declare_component("obb_viewer", path=str(VIEWER))
BROWSER_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
HIDE_PAGE = """<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important}
[data-testid="stMainBlockContainer"], .block-container {padding: 0 !important; max-width: none !important}
</style>"""


@st.cache_resource(show_spinner=False)
def get_model(path):
    return OpenOBB(path).warmup()                 # the first image is then as fast as the next ones


def detect(path, data, iou):
    """predict() as deployed; iou=None: the NMS calibrated at training, else that threshold ("none": no NMS)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None, 0.0
    t = time.perf_counter()
    r = get_model(path).predict(img, conf=0.05, iou=iou, max_det=1000, verbose=False)[0]
    return img, r, (time.perf_counter() - t) * 1000


def result(name, img, r, ms):
    """Every box with conf >= 0.05 for the viewer (it filters by confidence itself)."""
    pts = r.obb.xyxyxyxy.reshape(len(r), 8) if len(r) else np.zeros((0, 8))
    objs = [{"c": int(c), "s": round(float(s), 6), "p": [round(float(v), 2) for v in p],
             "r": [round(float(v), 4) for v in rb]}
            for c, s, p, rb in zip(r.obb.cls, r.obb.conf, pts, r.obb.xywhr)]
    out = {"file": name, "w": img.shape[1], "h": img.shape[0], "ms": ms, "objs": objs,
           "names": [r.names.get(i, str(i)) for i in range(max(r.names) + 1)] if r.names else []}
    if not name.lower().endswith(BROWSER_IMAGES):   # BMP / TIFF: the browser cannot show them, send a PNG back
        out["img"] = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    return out


def main():
    st.set_page_config(page_title="OBB test (ONNX)", page_icon="📐", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(HIDE_PAGE, unsafe_allow_html=True)
    req = st.session_state.get("viewer") or {}          # what the viewer asked for last: image, model, NMS
    path = Path(req.get("model") or MODEL_PATH).expanduser()
    args = {"id": req.get("id", 0), "model": str(path)}
    try:
        if not path.exists():
            raise FileNotFoundError(f"Không thấy model: {path}")
        model = get_model(str(path))
        nms = model.meta.get("nms")
        nms0 = 1.0 if nms is None else float(nms)
        args.update(nms0=nms0, info=f"ONNX Runtime · {model.device}")
        if req.get("data"):
            iou = req.get("nms")
            run_iou = None if iou is None or abs(iou - nms0) < 1e-9 else ("none" if iou >= 1.0 else float(iou))
            # every request of the viewer (new image, NMS or model: a new id) runs the model; a rerun of the page
            # without a new request reuses that one result (no cache of past images: they would hold the decoded pages)
            key = (req.get("id"), str(path), run_iou)
            last = st.session_state.get("last_run")
            if last is None or last[0] != key:
                last = (key, detect(str(path), base64.b64decode(req["data"]), run_iou))
                st.session_state["last_run"] = last
            img, r, ms = last[1]
            if img is None:
                raise ValueError(f"Không đọc được ảnh {req.get('file')}")
            args["result"] = result(req.get("file") or "image", img, r, ms)
    except Exception as e:                               # shown in the viewer's panel
        args["error"] = str(e)
    viewer(**args, key="viewer", default=None)


if __name__ == "__main__":
    main()
