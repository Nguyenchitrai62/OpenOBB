"""Push this research repo to GitHub, then sync the clean product subset to the company GitLab, in one command.

    python tools/push_all.py              # git push (GitHub) -> sync -> test -> push GitLab
    python tools/push_all.py --dry-run    # show what would change on GitLab, push nothing
    python tools/push_all.py --no-test    # skip the test run on the GitLab copy (not recommended)

GitHub (github.com/Nguyenchitrai62/OpenOBB) holds everything: research, Colab notebooks (Colab installs from it).
GitLab (openanything/openobb) holds only the commercial-clean product: ALLOW below, minus EXCLUDE (tests that need
research scripts), with TRANSFORMS applied (notices / docs that mention research files, install URLs). Its own
README.md and .gitignore are never overwritten. Only committed files are synced (`git archive HEAD`), commits go on
top of GitLab main (never a force-push), and nothing is pushed unless the GitLab copy passes its tests and contains
every source file of the package.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITLAB = "https://git.anybim.vn/CxDP/service/hicasai/openanything/openobb.git"
GITLAB_PIP = "git.anybim.vn/CxDP/service/hicasai/openanything/openobb.git"
CHECKOUT = ROOT / ".gitlab_export"                     # persistent working clone of GitLab (gitignored)
ALLOW = ["openobb", "tests", "docs/DEPLOY.md", "docs/FINETUNE.md", "docs/VRDET2.md", "docs/VRDET3.md",
         "docs/VRDET4.md", "docs/VRDET5.md", "docs/ARCHITECTURE_CARD.md", "app/streamlit_app.py", "run_app.bat",
         "LICENSE", "LICENSES", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.txt"]
EXCLUDE = {"tests/test_context.py", "tests/test_import.py", "tests/test_layers.py", "tests/test_pdf_vectors.py"}
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(ultralytics|fitz|pymupdf|mmcv|mmdet|mmrotate|ai4rs)\b", re.M)
# (file, pattern, replacement, is_regex): product-repo wording for text that mentions research-only files
TRANSFORMS = [
    ("THIRD_PARTY_NOTICES.md", "All other code in `openobb/`, `colab/data/`, `tools/`, `tests/` is original VRDet code.",
     "All other code in `openobb/`, `tests/`, `app/` is original OpenOBB / VRDet code.", False),
    ("THIRD_PARTY_NOTICES.md",
     r"- \*\*Ultralytics \(AGPL-3\.0\)\*\* is only used by `colab/baselines/yolo_obb\.py`.*?is not distributed\.",
     "- **Ultralytics (AGPL-3.0)** is not used or distributed by this repository. `tests/test_license_guard.py` "
     "fails if\n  `openobb/` imports it (or mmcv / mmdet / mmrotate / ai4rs). YOLO models were only trained, in a "
     "separate research\n  repository, as accuracy baselines.", True),
    ("docs/ARCHITECTURE_CARD.md", "- Lịch sử thí nghiệm: [research/LEDGER.md](../research/LEDGER.md).",
     "- Lịch sử thí nghiệm: sổ cái của repo nghiên cứu (không nằm trong repo này).", False),
    ("docs/FINETUNE.md", r"- \[`colab/VRDet_wall_color\.ipynb`\].*\n- \[`colab/VRDet_train\.ipynb`\].*",
     "- Notebook Colab nằm ở repo nghiên cứu. Trong repo này dùng thẳng CLI: `pip install` rồi `openobb train ...` "
     "(mục 2).", True),
    ("requirements.txt", r"(?m)^.*(pdf_vectors|pymupdf).*\n", "", True),
    ("requirements.txt", "# VRDet runtime", "# OpenOBB runtime", False),
    ("pyproject.toml", 'readme = "docs/FINETUNE.md"', 'readme = "README.md"', False),
    ("docs/DEPLOY.md", r"https://<token>@github\.com/Nguyenchitrai62/OpenOBB\.git",
     f"https://<user>:<token>@{GITLAB_PIP}", True),
    ("docs/DEPLOY.md", "Nếu repo chuyển sang private", "Repo GitLab công ty cần đăng nhập", False),
    ("docs/DEPLOY.md", r"https://github\.com/Nguyenchitrai62/OpenOBB\.git", f"https://{GITLAB_PIP}", True),
    ("docs/FINETUNE.md", r"https://github\.com/Nguyenchitrai62/OpenOBB\.git", f"https://{GITLAB_PIP}", True),
]


def run(cmd, cwd=ROOT, check=True, capture=False):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}\n{(r.stdout or '')[-2000:]}{(r.stderr or '')[-2000:]}")
    return r


def build_tree(dst):
    """Committed ALLOW files of HEAD -> dst, minus EXCLUDE, with TRANSFORMS applied."""
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "a.tar"
        run(["git", "archive", "-o", str(tar), "HEAD", *ALLOW])
        with tarfile.open(tar) as t:
            t.extractall(dst, filter="data")
    for rel in EXCLUDE:
        (dst / rel).unlink(missing_ok=True)
    for rel, pat, rep, is_re in TRANSFORMS:
        p = dst / rel
        s = p.read_text(encoding="utf-8")
        t = re.sub(pat, rep, s, flags=re.S if ".*?" in pat else 0) if is_re else s.replace(pat, rep)
        if t == s and not (rep and rep in s):
            print(f"  note: transform did not apply to {rel}: {pat[:60]!r} (text changed? check the GitLab wording)")
        p.write_text(t, encoding="utf-8")
    bad = [str(p.relative_to(dst)) for p in (dst / "openobb").rglob("*.py") if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    if bad:
        sys.exit(f"refusing to sync: forbidden imports in {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    head = run(["git", "rev-parse", "--short", "HEAD"], capture=True).stdout.strip()
    subject = run(["git", "log", "-1", "--format=%s"], capture=True).stdout.strip()
    dirty = run(["git", "status", "--porcelain", "--", *ALLOW], capture=True).stdout.strip()
    if dirty:
        print("note: uncommitted changes are NOT synced:\n" + dirty)

    if not a.dry_run:
        print("== GitHub: git push")
        run(["git", "push", "origin", "HEAD:main"])

    print("== GitLab: update the working clone")
    if (CHECKOUT / ".git").exists():
        run(["git", "fetch", "-q", "origin"], cwd=CHECKOUT)
        run(["git", "reset", "-q", "--hard", "origin/main"], cwd=CHECKOUT)
        run(["git", "clean", "-qfdx"], cwd=CHECKOUT)
    else:
        run(["git", "clone", "-q", GITLAB, str(CHECKOUT)])
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "tree"
        new.mkdir()
        build_tree(new)
        for rel in ALLOW:                                 # replace the managed paths, keep README.md / .gitignore
            p = CHECKOUT / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        shutil.copytree(new, CHECKOUT, dirs_exist_ok=True)
    run(["git", "add", "-A"], cwd=CHECKOUT)
    committed = set(run(["git", "ls-files", "openobb"], capture=True).stdout.split())
    tracked = set(run(["git", "ls-files", "openobb"], cwd=CHECKOUT, capture=True).stdout.split())
    missing = sorted(committed - tracked)
    if missing:
        sys.exit(f"refusing to push: package files not tracked on GitLab (ignore rule?): {missing[:5]}")
    diff = run(["git", "diff", "--cached", "--stat"], cwd=CHECKOUT, capture=True).stdout.strip()
    if not diff:
        print("GitLab already up to date.")
        return
    print(diff)
    if a.dry_run:
        print("(dry run: nothing pushed)")
        return
    if not a.no_test:
        print("== GitLab copy: pytest")
        r = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"], cwd=CHECKOUT,
                check=False, capture=True)
        print((r.stdout or "").strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
        if r.returncode:
            sys.exit("tests failed on the GitLab copy: nothing pushed to GitLab (GitHub already pushed)")
        run(["git", "clean", "-qfdX"], cwd=CHECKOUT)       # __pycache__ etc. created by the test run
    msg = (f"Sync from GitHub {head}: {subject}\n\n"
           "Product subset of github.com/Nguyenchitrai62/OpenOBB (tools/push_all.py).\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=CHECKOUT, input=msg, text=True, check=True)
    run(["git", "push", "-q", "origin", "HEAD:main"], cwd=CHECKOUT)
    print("GitLab:", run(["git", "log", "--oneline", "-1"], cwd=CHECKOUT, capture=True).stdout.strip())


if __name__ == "__main__":
    main()
