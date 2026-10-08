import sys
from pathlib import Path

import numpy as np
import pytest

fitz = pytest.importorskip("fitz")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pdf_vectors import page_tokens  # noqa: E402

from vrdet.data.vectors import cut_tile  # noqa: E402


@pytest.mark.parametrize("rotation", [0, 90])
def test_tokens_land_on_rendered_strokes(rotation):
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.draw_line((20, 30), (250, 30), color=(0, 0, 0), width=3)
    page.draw_rect(fitz.Rect(60, 80, 140, 150), color=(0, 0, 0), width=3)
    page.draw_bezier((160, 170), (190, 60), (230, 60), (280, 170), color=(0, 0, 0), width=3)
    page.set_rotation(rotation)
    scale = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    tokens, layer, names = page_tokens(page, scale)
    assert len(tokens) == 1 + 4 + 1 and set(tokens[:, 0].tolist()) == {0.0, 4.0}
    pts = tokens[:, 1:17].reshape(-1, 2)
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, pix.width - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, pix.height - 1)
    dark = np.array([img[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].min() for x, y in zip(xi, yi)])
    assert (dark < 100).mean() > 0.95, (rotation, dark)
    assert np.allclose(tokens[:, 20], 3 * scale)
    t = cut_tile({"tokens": tokens, "layer": layer}, 0, 0, max(pix.width, pix.height))
    assert t.shape == (6, 22) and (t[:, 21] == 0).all()        # no optional-content layers -> layer 0
