"""Make sure the installed PyTorch matches the machine: a CUDA build when an NVIDIA GPU is present, else the CPU build.

pip's default torch on Windows is CPU-only, so a page that takes ~0.1-0.3 s on an RTX GPU ran at CPU speed (seconds)
without any error. On first use (`from openobb import OpenOBB`, the CLI, the Streamlit app) this module checks:

  NVIDIA GPU (nvidia-smi) + torch missing or a CPU build -> pip install torch from the PyTorch CUDA index matching the
                                                            driver (newest cuXXX the driver supports)
  no NVIDIA GPU + torch missing                           -> pip install torch from the PyTorch CPU index

A torch already imported in this process cannot be swapped: the CUDA build is installed for the next start and this
run continues on CPU (a message says so). Set OPENOBB_AUTO_TORCH=0 to turn the check off.
This module must not import torch at module level (it runs before torch exists).
"""
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys

INDEX = "https://download.pytorch.org/whl/{}"
# PyTorch CUDA wheel indexes, newest first, with the CUDA version the NVIDIA driver must support
CUDA_INDEXES = (("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1)),
                ("cu118", (11, 8)))
_DONE = None


def nvidia_gpu():
    """-> (gpu name, driver CUDA version (major, minor)) from nvidia-smi, or None without an NVIDIA GPU / driver."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
        name = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not name or m is None:
        return None
    return name[0].strip(), (int(m.group(1)), int(m.group(2)))


def cuda_index(driver_cuda):
    """Newest PyTorch CUDA index the driver supports, or None (driver too old: CPU build)."""
    for tag, need in CUDA_INDEXES:
        if tuple(driver_cuda) >= need:
            return tag
    return None


def plan(gpu, torch_installed, torch_cuda_build):
    """Decision only (no side effects): -> (index tag or None, reason). index None = nothing to install."""
    if gpu is not None:
        if torch_installed and torch_cuda_build:
            return None, "ok"
        tag = cuda_index(gpu[1])
        if tag is None:
            return (None, "ok") if torch_installed else ("cpu", f"driver CUDA {gpu[1]} too old for PyTorch: CPU build")
        return tag, f"{gpu[0]} found but PyTorch is {'a CPU build' if torch_installed else 'missing'}"
    return (None, "ok") if torch_installed else ("cpu", "no NVIDIA GPU: CPU build")


def _pip_install(tag, log):
    pkgs = ["torch"] + (["torchvision"] if importlib.util.find_spec("torchvision") is not None else [])
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-cache-dir", *pkgs,
           "--index-url", INDEX.format(tag)]
    log(f"[setup] installing PyTorch ({tag}): {' '.join(cmd)}")
    r = subprocess.run(cmd)
    return r.returncode == 0


def ensure_torch(log=print):
    """-> "ok" | "installed" (usable now) | "restart" (installed; a new process uses it) | "failed" | "off"."""
    global _DONE
    if _DONE is not None:
        return _DONE
    if os.environ.get("OPENOBB_AUTO_TORCH", "1").lower() in ("0", "false", "no", "off"):
        _DONE = "off"
        return _DONE
    installed = importlib.util.find_spec("torch") is not None
    loaded = "torch" in sys.modules
    cuda_build = False
    if installed:
        import torch
        cuda_build = torch.version.cuda is not None
        if cuda_build or shutil.which("nvidia-smi") is None:      # fast path: nothing to decide
            _DONE = "ok"
            if cuda_build and not torch.cuda.is_available() and shutil.which("nvidia-smi") is not None:
                log("[setup] WARNING: PyTorch has CUDA but sees no GPU (driver / CUDA mismatch?): running on CPU")
            return _DONE
        loaded = True
    gpu = nvidia_gpu()
    tag, why = plan(gpu, installed, cuda_build)
    if tag is None:
        _DONE = "ok"
        return _DONE
    log(f"[setup] {why}")
    if not _pip_install(tag, log):
        log("[setup] PyTorch install failed: see the pip output above (or install it yourself from pytorch.org)")
        _DONE = "failed"
        return _DONE
    if loaded:
        log("[setup] PyTorch with CUDA installed. Restart Python / the app to use the GPU; this run stays on CPU.")
        _DONE = "restart"
    else:
        importlib.invalidate_caches()
        _DONE = "installed"
    return _DONE
