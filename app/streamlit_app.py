"""VRDet quick test: drag and drop a drawing image, see the oriented boxes.

    pip install streamlit
    streamlit run app/streamlit_app.py

Put the downloaded best.pt at MODEL_PATH (or type another path in the sidebar). Works on CPU or GPU; pages of any
size are tiled exactly as in training.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrdet.predict import load_model, merge_dets, predict_image, to_label_lines  # noqa: E402

MODEL_PATH = "models/best.pt"       # đường dẫn model: tuyệt đối, hoặc tương đối so với thư mục repo

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (128, 0, 0),
           (128, 128, 0), (0, 0, 128), (255, 225, 25), (128, 128, 128)]          # RGB, one per class


@st.cache_resource(show_spinner="Loading model...")
def get_model(path, queries):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model, names, targs = load_model(path, device, queries)
    return model, list(names), targs, device


def draw(img_rgb, dets, names, thickness, show_text):
    out = img_rgb.copy()
    for d in dets:
        color = PALETTE[d["class_id"] % len(PALETTE)]
        pts = np.array(d["poly"]).reshape(4, 2).round().astype(np.int32)
        cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
        if show_text:
            x, y = pts[np.argmin(pts[:, 1])]
            cv2.putText(out, f"{d['class']} {d['score']:.2f}", (int(x), int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * thickness, color, max(1, thickness // 2), cv2.LINE_AA)
    return out


def main():
    st.set_page_config(page_title="VRDet OBB test", page_icon="📐", layout="wide")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Model (.pt)", MODEL_PATH)
        p = Path(model_path)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            st.error(f"Không thấy model: {p}")
            st.stop()
        queries = st.select_slider("Queries", [300, 600, 900], value=900)
        model, names, targs, device = get_model(str(p), queries)
        st.caption(f"VRDet-{targs.get('size', 's')} · {len(names)} class · tile {targs.get('img', 1024)} · {device}")

        st.header("Ảnh")
        up = st.file_uploader("Kéo thả ảnh", type=("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"))
        c1, c2 = st.columns(2)
        conf = c1.slider("Confidence", 0.05, 1.0, 0.3, 0.05)
        iou = c2.slider("NMS IoU", 0.0, 1.0, 0.1, 0.05, help="gộp box trùng (giữa các tile)")
        with st.expander("Kích thước ảnh"):
            size = st.number_input("imgsz", 256, 2048, int(targs.get("img", 1024)), 32,
                                   help="cạnh dài ảnh đưa vào model (mặc định = lúc train)")
            tile = st.checkbox("Cắt tile ở độ phân giải gốc", value=not targs.get("fit", False),
                               help="cho trang rất lớn có object rất nhỏ; mặc định resize cả ảnh như lúc train")
            scale = st.number_input("tile_scale", 0.1, 4.0, float(targs.get("scale", 1.0)), 0.05, disabled=not tile)
            gap = st.number_input("Tile overlap (px)", 0, 512, 200, 8, disabled=not tile)
        st.subheader("Class")
        selected = [i for i, n in enumerate(names) if st.checkbox(n, value=True, key=f"cls_{i}")]
        thickness = st.slider("Độ dày nét", 1, 8, 2)
        show_text = st.checkbox("Hiện tên + điểm", value=False)
        detect = st.button("Detect", type="primary")

    if up is None:
        st.info("Kéo thả một ảnh bản vẽ vào thanh bên trái.")
        return
    data = up.getvalue()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Không đọc được ảnh.")
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    run_scale = float(scale) if tile else int(size) / max(img.shape[:2])     # whole image: long side = imgsz
    key = (up.name, len(data), str(p), queries, int(size), run_scale, int(gap))
    if detect:
        t = time.time()
        with st.spinner("Detecting..."):
            raw = predict_image(model, img, int(size), int(gap), 8, device, queries, scale=run_scale)
        st.session_state.update(raw=raw, key=key, ms=(time.time() - t) * 1000)

    left, right = st.columns(2)
    left.image(rgb, caption=f"{up.name} ({img.shape[1]}×{img.shape[0]})")
    if st.session_state.get("key") != key:
        right.info("Bấm **Detect**.")
        return

    dets = [d for d in merge_dets(st.session_state["raw"], names, conf, iou) if d["class_id"] in selected]
    right.image(draw(rgb, dets, names, thickness, show_text), caption=f"{len(dets)} objects · "
                f"{st.session_state['ms']:.0f} ms ({device})")

    counts = pd.Series([d["class"] for d in dets], dtype=str).value_counts()
    table = pd.DataFrame({"class": names, "count": [int(counts.get(n, 0)) for n in names]})
    c1, c2 = st.columns([1, 2])
    c1.dataframe(table[table["class"].isin([names[i] for i in selected])], hide_index=True)
    with c2:
        stem = Path(up.name).stem
        lines = to_label_lines(dets, img.shape)
        st.download_button("Tải nhãn TXT (class x1 y1 ... x4 y4)", "\n".join(lines) + ("\n" if lines else ""),
                           file_name=f"{stem}.txt", mime="text/plain")
        st.download_button("Tải JSON (toạ độ pixel + điểm)", json.dumps(dets, ensure_ascii=False, indent=1),
                           file_name=f"{stem}.json", mime="application/json")
        if lines:
            st.code("\n".join(lines[:10]), language=None)


if __name__ == "__main__":
    main()
