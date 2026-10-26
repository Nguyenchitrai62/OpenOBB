"""Build the ONNX runtime wheel (handover): only the inference code for exported .onnx models (no PyTorch, no
architecture, no training code). python tools/build_onnx_wheel.py [--root DIR] [--pkg NAME] [--out DIR]

The runtime's source package (<pkg> = openobb_onnx, or hicasobb_onnx in the renamed product tree) is installed under
the NAME of the full package (<pkg> without "_onnx": openobb / hicasobb): same distribution name (wheel
<name>-X.Y.Z-py3-none-any.whl), import name, class name and command. Inference code written for a .pt
(`from hicasobb import HicasOBB; HicasOBB("best.pt").predict(...)`, `hicasobb predict model=...`) runs unchanged with the
.onnx path, and `pip install -U <wheel>` replaces an installed full package like any upgrade. The wheel holds <name>/ (model.py, ops.py,
results.py, cli.py, app.py) + the Streamlit viewer (app/viewer/index.html -> <name>/viewer/) + LICENSE,
THIRD_PARTY_NOTICES.md and LICENSES/ (dist-info/licenses/). Its version is
the full package's (openobb/__init__.py, or <product>/__init__.py). tools/push_all.py calls build() on the product tree.

    pip install <wheel>[gpu]       ONNX Runtime with CUDA (onnxruntime-gpu + the CUDA / cuDNN pip packages)
    pip install <wheel>[cpu]       ONNX Runtime on the CPU
    pip install <wheel>[app]       + Streamlit for `<name> app`
    pip install <wheel>[sahi]      + SAHI for sliced inference with a YOLO service's calls (`<name>.sahi`)
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(torch|torchvision|ultralytics|openobb\b|hicasobb\b)", re.M)

PYPROJECT = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "{dist}"
version = "{version}"
description = "Oriented-box (OBB) detection runtime for exported ONNX models (ONNX Runtime, no PyTorch)"
readme = "README.md"
requires-python = ">=3.9"
license = "LicenseRef-Proprietary"
license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/*"]
dependencies = ["numpy", "opencv-python>=4.6", "shapely>=2.0"]

[project.optional-dependencies]
gpu = ["onnxruntime-gpu[cuda,cudnn]>=1.21"]
cpu = ["onnxruntime>=1.21"]
app = ["streamlit>=1.30"]
sahi = ["sahi>=0.12"]

[project.scripts]
{name} = "{name}.cli:main"

[tool.setuptools]
packages = ["{name}"]

[tool.setuptools.package-data]
{name} = ["viewer/*.html"]
"""

README = """# {dist}

Runtime for exported oriented-box (OBB) models (`.onnx`): ONNX Runtime + numpy, no PyTorch. Same package name, class
`{cls}` and command `{name}` as the full package: code written for a `.pt` runs unchanged, only the model path becomes the
`.onnx`; installing it over the full package replaces it. Install ONE of the two ONNX Runtime builds below.

## GPU (NVIDIA)

Needs an NVIDIA driver that supports CUDA 13 (`nvidia-smi` shows "CUDA Version: 13.x"). CUDA / cuDNN come from pip,
no separate CUDA install:

```bash
pip uninstall -y onnxruntime             # the CPU build must not be installed next to the GPU one
pip install -U "{wheel}[gpu]"          # + [app] for the Streamlit viewer: "{wheel}[gpu,app]"
{name} info model=best.onnx             # must print "device cuda:0"
```

## CPU only

```bash
pip uninstall -y onnxruntime-gpu
pip install -U "{wheel}[cpu]"          # or "{wheel}[cpu,app]"
```

## Use

```bash
{name} predict model=best.onnx source=pages/ conf=auto        # -> runs/obb/predict (save_txt=True for labels)
{name} app model=best.onnx                                     # Streamlit viewer (needs [app])
```

```python
from {name} import {cls}
model = {cls}("best.onnx")                 # device=None: GPU when available, else CPU; "cpu", 0, "trt"
for r in model.predict("page.png", conf=0.25):
    print(r.obb.xywhr.cpu().numpy(), r.obb.conf, r.obb.cls, r.names)
```

Sliced inference (SAHI, needs `[sahi]`): the calls of a YOLO + SAHI service, one import changed.

```python
from {name}.sahi import AutoDetectionModel       # instead of: from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction   # unchanged
model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="best.onnx",
                                           confidence_threshold=0.1, device="cuda")
result = get_sliced_prediction(image, model, slice_height=1024, slice_width=1024,
                               overlap_height_ratio=0.5, overlap_width_ratio=0.5)
```
"""


def version_of(root, pkg):
    """Version of the main package next to the runtime (openobb_onnx -> openobb/__init__.py)."""
    init = root / pkg.removesuffix("_onnx") / "__init__.py"
    m = re.search(r'__version__ = "(\d+\.\d+\.\d+)"', init.read_text(encoding="utf-8")) if init.exists() else None
    if not m:
        sys.exit(f"no __version__ in {init}")
    return m.group(1)


def build(root=ROOT, pkg="openobb_onnx", out=None, cls=None):
    """-> path of the built wheel in `out` (default <root>/dist_onnx)."""
    root = Path(root)
    src = root / pkg
    bad = [p.name for p in src.glob("*.py") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    if bad:
        sys.exit(f"refusing to build: {pkg} imports PyTorch / the training package in {bad}")
    version = version_of(root, pkg)
    name = dist = pkg.removesuffix("_onnx")                # package / distribution name = the full package's
    cls = cls or ("HicasOBB" if pkg.startswith("hicas") else "OpenOBB")
    out = Path(out or root / "dist_onnx")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copytree(src, t / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in (t / name).glob("*.py"):                  # absolute imports / messages: the installed name
            f.write_text(f.read_text(encoding="utf-8").replace(pkg, name), encoding="utf-8")
        (t / name / "_version.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
        (t / name / "viewer").mkdir(exist_ok=True)
        shutil.copy(root / "app" / "viewer" / "index.html", t / name / "viewer" / "index.html")
        shutil.copy(root / "LICENSE", t / "LICENSE")
        shutil.copy(root / "THIRD_PARTY_NOTICES.md", t / "THIRD_PARTY_NOTICES.md")     # Apache-2.0 s.4: notices +
        shutil.copytree(root / "LICENSES", t / "LICENSES")                             # licence texts ship with the weights
        wheel = f"{name}-{version}-py3-none-any.whl"
        (t / "README.md").write_text(README.format(dist=dist, wheel=wheel, name=name, cls=cls), encoding="utf-8")
        (t / "pyproject.toml").write_text(PYPROJECT.format(dist=dist, version=version, name=name), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", str(t), "--no-deps", "--no-build-isolation", "-q",
                            "-w", str(t / "dist")], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"pip wheel failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        whl = next((t / "dist").glob("*.whl"))
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
        extra = [n for n in names if not (n.startswith(f"{name}/") or ".dist-info/" in n)]
        if extra:
            sys.exit(f"unexpected files in the wheel: {extra}")
        dst = out / whl.name
        shutil.copy(whl, dst)
    print(f"[onnx wheel] {dst} ({dst.stat().st_size / 1024:.0f} KB, {len(names)} files, import name {name})")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--pkg", default="openobb_onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.root, a.pkg, a.out)


if __name__ == "__main__":
    main()
