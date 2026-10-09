"""Commercial-use guard: the VRDet package must not depend on AGPL / non-commercial code."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "openobb"
FORBIDDEN = ("ultralytics", "mmcv", "mmdet", "mmrotate", "ai4rs")     # AGPL, or carrying NC-derived code


def test_vrdet_has_no_forbidden_imports():
    bad = []
    for f in PKG.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [f"{f.relative_to(PKG)}: {n}" for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad
