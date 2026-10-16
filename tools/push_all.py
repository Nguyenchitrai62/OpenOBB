"""Push this research repo to GitHub, then sync the clean product subset to the company GitLab, in one command.

    python tools/push_all.py              # git push (GitHub) -> sync -> test -> push GitLab
    python tools/push_all.py --dry-run    # show what would change on GitLab, push nothing
    python tools/push_all.py --no-test    # skip the test run on the GitLab copy (not recommended)
    python tools/push_all.py --minor      # bump 0.X.0 instead of the patch number (breaking / larger changes)

Versioning: openobb/__init__.py `__version__` is the only version (pyproject reads it). When the package
(openobb/, pyproject.toml, requirements.txt) changed since the commit that last set the version, the patch number is
bumped and committed first ("Version x.y.z"); every released version is tagged vX.Y.Z on GitHub and on GitLab.

GitHub (github.com/Nguyenchitrai62/OpenOBB) holds everything: research, Colab notebooks (Colab installs from it).
GitLab (openanything/openobb) holds only the commercial-clean product: ALLOW below, minus EXCLUDE (tests that need
research scripts), with TRANSFORMS applied (notices / docs that mention research files, install URLs). Its own
README.md comes from tools/gitlab_README.md; its .gitignore is never overwritten. Only committed files are synced (`git archive HEAD`), commits go on
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
GITLAB = "https://git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git"     # product repo (since 0.8.1)
GITLAB_PIP = "git.anybim.vn/CxDP/service/hicasai/openanything/hicasobb.git"
CHECKOUT = ROOT / ".gitlab_export"                     # persistent working clone of GitLab (gitignored)
ALLOW = ["openobb", "tests", "docs/DEPLOY.md", "docs/FINETUNE.md", "docs/OPENOBB2.md", "docs/OPENOBB3.md",
         "docs/OPENOBB4.md", "docs/OPENOBB5.md", "docs/OPENOBB6.md", "docs/OPENOBB7.md",
         "docs/OPENOBB8.md", "docs/OPENOBB9.md", "docs/OPENOBB10.md", "docs/OPENOBB11.md", "docs/OPENOBB12.md", "docs/OPENOBB13.md", "docs/PRETRAIN.md", "docs/OPENOBB1.md", "app/streamlit_app.py", "app/viewer/index.html",
         "run_app.bat",
         "LICENSE", "LICENSES", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.txt"]
# files from older layouts removed from GitLab when present (renamed in the research repo)
RETIRED = ["hicas_detection_obb", "docs/ARCHITECTURE_CARD.md", "docs/VRDET2.md", "docs/VRDET3.md", "docs/VRDET4.md", "docs/VRDET5.md",
           "tests/test_vrdet2.py", "tests/test_vrdet3.py", "tests/test_vrdet4.py", "tests/test_vrdet5.py"]
README_SRC = "tools/gitlab_README.md"                 # GitLab README (product wording), copied as README.md
EXCLUDE = {"tests/test_context.py", "tests/test_import.py", "tests/test_layers.py", "tests/test_pdf_vectors.py"}
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(ultralytics|fitz|pymupdf|mmcv|mmdet|mmrotate|ai4rs)\b", re.M)
# (file, pattern, replacement, is_regex): product-repo wording for text that mentions research-only files
TRANSFORMS = [
    ("THIRD_PARTY_NOTICES.md", "All other code in `openobb/`, `colab/data/`, `tools/`, `tests/` is original OpenOBB code.",
     "All other code in `openobb/`, `tests/`, `app/` is original OpenOBB code.", False),
    ("THIRD_PARTY_NOTICES.md",
     r"- \*\*Ultralytics \(AGPL-3\.0\)\*\* is only used by `colab/baselines/yolo_obb\.py`.*?is not distributed\.",
     "- **Ultralytics (AGPL-3.0)** is not used or distributed by this repository. `tests/test_license_guard.py` "
     "fails if\n  `openobb/` imports it (or mmcv / mmdet / mmrotate / ai4rs). YOLO models were only trained "
     "separately,\n  as accuracy baselines.", True),
    ("docs/OPENOBB1.md", "- Lịch sử thí nghiệm: [research/LEDGER.md](../research/LEDGER.md).",
     "- Lịch sử thí nghiệm: tài liệu nội bộ (không nằm trong repo này).", False),
    ("docs/FINETUNE.md", r"- \[`colab/OpenOBB_wall_color\.ipynb`\].*\n- \[`colab/OpenOBB_train\.ipynb`\].*",
     "- Repo này không kèm notebook Colab. Dùng thẳng CLI: `pip install` rồi `openobb train ...` "
     "(mục 2).", True),
    ("requirements.txt", r"(?m)^.*(pdf_vectors|pymupdf).*\n", "", True),
    ("docs/DEPLOY.md", r"https://<token>@github\.com/Nguyenchitrai62/OpenOBB\.git",
     f"https://<user>:<token>@{GITLAB_PIP}", True),
    ("docs/DEPLOY.md", "Nếu repo chuyển sang private", "Repo GitLab công ty cần đăng nhập", False),
    ("docs/DEPLOY.md", r"https://github\.com/Nguyenchitrai62/OpenOBB\.git", f"https://{GITLAB_PIP}", True),
    ("docs/FINETUNE.md", r"https://github\.com/Nguyenchitrai62/OpenOBB\.git", f"https://{GITLAB_PIP}", True),
    ("docs/OPENOBB7.md", r", \[research/ARCH_ANALYSIS\.md\]\(\.\./research/ARCH_ANALYSIS\.md\)", "", True),
    ("docs/OPENOBB9.md", r"\[research/LEDGER\.md\]\(\.\./research/LEDGER\.md\)",
     "tài liệu nội bộ", True),
    ("docs/OPENOBB10.md", "; script trong `research/analysis/`", "", False),
    ("docs/PRETRAIN.md", r"git\+https://github\.com/Nguyenchitrai62/OpenOBB\.git", f"git+https://{GITLAB_PIP}", True),
    ("docs/PRETRAIN.md", r"(?m)^!git clone -q https://github\.com/Nguyenchitrai62/OpenOBB\.git.*\n", "", True),
    ("docs/PRETRAIN.md", "!python /content/OpenOBB/colab/data/get_dota.py --root /content/datasets --splits train",
     "!python -c \"from openobb.data.download import dota; dota('/content/datasets', splits=('train',))\"", False),
    ("docs/OPENOBB6.md", r"\[research/ARCH_ANALYSIS\.md\]\(\.\./research/ARCH_ANALYSIS\.md\)",
     "báo cáo phân tích nội bộ (không nằm trong repo này)", True),
]


# GitLab branding (user, 2026-10-13): the product copy is renamed, GitHub keeps "openobb".
#   package openobb -> hicasobb, class OpenOBB* -> HicasOBB*, models / modules openobbN -> hicasobbN,
#   CLI command / run and cache file names openobb -> hicasobb. Repo URLs are kept.
# Checkpoints hold only tensors / dicts / strings (torch.load(weights_only=True) works; the model is rebuilt from
# args["arch"]), so .pt files trained with either name load in both.
PKG_NEW, CLI_NEW, CLASS_NEW = "hicasobb", "hicasobb", "HicasOBB"
KEEP = (GITLAB_PIP,)
TEXT_EXT = {".py", ".md", ".toml", ".txt", ".bat", ".cfg", ".ini", ".json", ".yaml", ".yml"}
TEXT_NAMES = {"LICENSE", "NOTICE", "COPYING"}                      # text files without a suffix


def is_text(f):
    return f.suffix.lower() in TEXT_EXT or f.name in TEXT_NAMES


# the product repo never names the research repo / package (customer-facing): checked before every push
LEAK = re.compile(r"openobb|vrdet|nguyenchitrai|github\.com/nguyen|(?<!\w)research/|repo nghiên cứu|research repo", re.I)
SCRUB = (("research/LEDGER.md", "internal benchmark notes"), ("research/ARCH_ANALYSIS.md", "internal analysis"),
         ("research/LESSONS_BY_LAYER.md", "internal notes"), ("repo nghiên cứu", "nhóm phát triển"),
         ("research repository", "internal repository"), ("research repo", "internal repository"))


def rebrand_text(s):
    masks = {}
    for i, k in enumerate(KEEP):
        tok = f"\0KEEP{i}\0"
        s = s.replace(k, tok)
        masks[tok] = k
    s = s.replace("OpenOBB", CLASS_NEW).replace("OPENOBB", CLASS_NEW.upper())
    s = re.sub(r"openobb(?=\d)", CLI_NEW, s)                               # openobb8x, openobb8.py, .openobb8
    s = re.sub(r"(?<=\bimport )openobb\b|(?<=\bfrom )openobb\b", PKG_NEW, s)
    s = re.sub(r"(?<![\w-])openobb(?=[./\*])|(?<=[\"'])openobb(?=[\"'])", PKG_NEW, s)   # openobb.x, openobb/, "openobb"
    s = s.replace("openobb", CLI_NEW)                                       # command, openobb_args.json, ...
    s = s.replace("VRDet", CLASS_NEW).replace("VRDET", CLASS_NEW.upper()).replace("vrdet", CLI_NEW)
    s = re.sub(r"\[([^\]]*)\]\((?:\.\./)*research/[^)]*\)", r"\1", s)          # [text](../research/x) -> text
    for old, new in SCRUB:
        s = s.replace(old, new)
    s = re.sub(r"(?<!\w)research/[\w./-]*", "internal notes", s)              # not github.com/facebookresearch/
    for tok, k in masks.items():
        s = s.replace(tok, k)
    return s


def rebrand_path(rel):
    """Posix path relative to the repo -> its GitLab name."""
    parts = rel.split("/")
    head = [PKG_NEW] if parts[0] == "openobb" else [rebrand_text(parts[0])]
    return "/".join(head + [rebrand_text(x) for x in parts[1:]])


def leaks(dst):
    """Text files of the exported tree that still name the research repo / package -> [(file, match)]."""
    out = []
    for f in dst.rglob("*"):
        if f.is_file() and is_text(f):
            m = LEAK.search(f.read_text(encoding="utf-8"))
            if m:
                out.append((f.relative_to(dst).as_posix(), m.group(0)))
    return out


LOCAL_WHEELS = ROOT / "dist_hicasobb"                 # every released product wheel, kept locally (gitignored)


def build_wheel(src):
    """Product wheel (pure Python, ~200 KB) of the exported tree -> src/wheels/<wheel> (only the latest is kept
    there; git history has the older ones) + a copy in LOCAL_WHEELS. Customers install it with
    `pip install hicasobb-X.Y.Z-py3-none-any.whl` (dependencies from PyPI), no access to the repo needed."""
    import zipfile
    with tempfile.TemporaryDirectory() as tmp:
        run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "--no-build-isolation", "-q", "-w", tmp])
        run(["git", "clean", "-qfdX"], cwd=src)            # build/, *.egg-info left by the build
        whl = next(Path(tmp).glob("*.whl"))
        with zipfile.ZipFile(whl) as z:
            bad = [n for n in z.namelist() if LEAK.search(n + z.read(n).decode("utf-8", "ignore"))]
        if bad:
            sys.exit(f"refusing to ship: research names inside the wheel: {bad[:5]}")
        dst = src / "wheels"
        dst.mkdir(exist_ok=True)
        for old in dst.glob("*.whl"):
            old.unlink()
        shutil.copy(whl, dst / whl.name)
        LOCAL_WHEELS.mkdir(exist_ok=True)
        shutil.copy(whl, LOCAL_WHEELS / whl.name)
    return dst / whl.name


def rebrand(dst):
    """Rename files / folders and rewrite text files of the exported tree in place."""
    files = [f for f in dst.rglob("*") if f.is_file()]
    for f in files:
        rel = f.relative_to(dst).as_posix()
        if is_text(f):
            f.write_text(rebrand_text(f.read_text(encoding="utf-8")), encoding="utf-8")
        new = dst / rebrand_path(rel)
        if new != f:
            new.parent.mkdir(parents=True, exist_ok=True)
            f.replace(new)
    for d in sorted((d for d in dst.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        if not any(d.iterdir()):
            d.rmdir()


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
        run(["git", "archive", "-o", str(tar), "HEAD", *ALLOW, README_SRC])
        with tarfile.open(tar) as t:
            t.extractall(dst, filter="data")
    for rel in EXCLUDE:
        (dst / rel).unlink(missing_ok=True)
    (dst / README_SRC).replace(dst / "README.md")
    shutil.rmtree(dst / "tools")
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
    rebrand(dst)
    left = leaks(dst)
    if left:
        sys.exit(f"refusing to sync: research names left in the product copy: {left[:8]}")


VERSION_FILE = ROOT / "openobb" / "__init__.py"
CODE_PATHS = ["openobb", "pyproject.toml", "requirements.txt"]
VERSION_RE = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')


def read_version():
    m = VERSION_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        sys.exit("no __version__ = \"x.y.z\" in openobb/__init__.py")
    return tuple(int(v) for v in m.groups())


def next_version(kind, dry_run):
    """Bump when package code changed since the commit that last set the version; returns the version to release."""
    cur = read_version()
    base = run(["git", "log", "-1", "--format=%H", "-G", '__version__ = "', "--", "openobb/__init__.py"],
               capture=True).stdout.strip()
    changed = run(["git", "diff", "--name-only", f"{base}..HEAD", "--", *CODE_PATHS], capture=True).stdout.split() \
        if base else ["(no version commit yet)"]
    if not changed:
        return cur, False
    major, minor, patch = cur
    new = (major, minor + 1, 0) if kind == "minor" else (major, minor, patch + 1)
    tag = ".".join(map(str, new))
    print(f"== version {'.'.join(map(str, cur))} -> {tag} ({len(changed)} package files changed since the last release)")
    if not dry_run:
        text = VERSION_FILE.read_text(encoding="utf-8")
        VERSION_FILE.write_text(VERSION_RE.sub(f'__version__ = "{tag}"', text, count=1), encoding="utf-8")
        run(["git", "add", str(VERSION_FILE)])
        subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=ROOT, check=True, text=True,
                       input=f"Version {tag}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    return new, True


def ensure_tag(tag, cwd):
    """Tag HEAD of `cwd` as `tag` and push it, unless the remote already has that tag."""
    if run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], cwd=cwd, capture=True).stdout.strip():
        return
    run(["git", "tag", "-f", tag], cwd=cwd, capture=True)
    run(["git", "push", "-q", "origin", f"refs/tags/{tag}"], cwd=cwd)
    print(f"   tagged {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--minor", action="store_true", help="bump the minor version instead of the patch number")
    a = ap.parse_args()
    dirty = run(["git", "status", "--porcelain", "--", *ALLOW], capture=True).stdout.strip()
    if dirty:
        print("note: uncommitted changes are NOT synced:\n" + dirty)
    version, _ = next_version("minor" if a.minor else "patch", a.dry_run)
    tag = "v" + ".".join(map(str, version))

    if not a.dry_run:
        print(f"== GitHub: git push ({tag})")
        run(["git", "push", "origin", "HEAD:main"])
        ensure_tag(tag, ROOT)

    print("== GitLab: update the working clone")
    if (CHECKOUT / ".git").exists():
        url = run(["git", "remote", "get-url", "origin"], cwd=CHECKOUT, capture=True).stdout.strip()
        if url != GITLAB:                                  # product repo moved: start from a fresh clone
            print(f"   GitLab project changed ({url} -> {GITLAB}): re-cloning")
            for child in CHECKOUT.iterdir():               # keep the folder itself (Windows: may be held open)
                if child.is_dir():
                    shutil.rmtree(child, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
                else:
                    child.unlink()
    if (CHECKOUT / ".git").exists():
        run(["git", "fetch", "-q", "--tags", "origin"], cwd=CHECKOUT)
        if run(["git", "rev-parse", "--verify", "-q", "origin/main"], cwd=CHECKOUT, check=False,
               capture=True).returncode == 0:
            run(["git", "reset", "-q", "--hard", "origin/main"], cwd=CHECKOUT)
        run(["git", "clean", "-qfdx"], cwd=CHECKOUT)
    else:
        run(["git", "clone", "-q", GITLAB, str(CHECKOUT)])
    if not (CHECKOUT / ".gitignore").exists():            # new project: anchored rules only (lesson 10-09)
        (CHECKOUT / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n/runs/\n/build/\n/dist/\n"
                                             "*.egg-info/\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / "tree"
        new.mkdir()
        build_tree(new)
        managed = ALLOW + RETIRED + ["README.md"]
        for rel in managed + [rebrand_path(r) for r in managed]:     # replace the managed paths, keep .gitignore
            p = CHECKOUT / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        shutil.copytree(new, CHECKOUT, dirs_exist_ok=True)
    run(["git", "add", "-A"], cwd=CHECKOUT)
    committed = {rebrand_path(f) for f in run(["git", "ls-files", "openobb"], capture=True).stdout.split()}
    tracked = set(run(["git", "ls-files", PKG_NEW], cwd=CHECKOUT, capture=True).stdout.split())
    missing = sorted(committed - tracked)
    if missing:
        sys.exit(f"refusing to push: package files not tracked on GitLab (ignore rule?): {missing[:5]}")
    diff = run(["git", "diff", "--cached", "--stat"], cwd=CHECKOUT, capture=True).stdout.strip()
    if not diff:
        print("GitLab already up to date.")
        if not a.dry_run:
            ensure_tag(tag, CHECKOUT)
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
    whl = build_wheel(CHECKOUT)
    run(["git", "add", "-A", "wheels"], cwd=CHECKOUT)
    print(f"   wheel {whl.name} ({whl.stat().st_size / 1024:.0f} KB): GitLab wheels/ and {LOCAL_WHEELS}")
    msg = f"Release {tag}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"    # no repo / package names
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=CHECKOUT, input=msg, text=True, check=True)
    run(["git", "push", "-q", "origin", "HEAD:main"], cwd=CHECKOUT)
    ensure_tag(tag, CHECKOUT)
    print("GitLab:", run(["git", "log", "--oneline", "-1"], cwd=CHECKOUT, capture=True).stdout.strip())


if __name__ == "__main__":
    main()
