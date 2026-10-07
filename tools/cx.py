"""Thin, retrying wrapper around the `colab` CLI for unattended use on Windows.

Everything an autonomous agent needs: create/stop VMs, run Python on them,
move files both ways (uploads are chunked: the CLI drops single uploads above
~20-80 MB), and read the compute-unit balance. No step needs a human.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

CHUNK = 16 * 2**20
MARK = "@@CX@@"

ENV = dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1")
COLAB = shutil.which("colab") or str(Path.home() / ".local/bin/colab.exe")


class ColabError(RuntimeError):
    pass


def run(args, stdin=None, timeout=900, check=True):
    p = subprocess.run([COLAB, *args], input=stdin, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ENV, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    # the CLI sometimes reports transfer errors with rc=0 ("[colab] Upload failed: ...")
    if check and (p.returncode != 0 or re.search(r"\[colab\] \w+ failed", out)):
        raise ColabError(f"colab {' '.join(args)} -> rc={p.returncode}\n{out[-2000:]}")
    return p.returncode, out


def retry(fn, tries=4, wait=10):
    for i in range(tries):
        try:
            return fn()
        except (ColabError, subprocess.TimeoutExpired) as e:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


# ---------- sessions ----------

def sessions():
    """{name: hardware} for sessions known to this CLI state (excludes '[?]' foreign ones)."""
    _, out = run(["sessions"], check=False, timeout=120)
    res = {}
    for m in re.finditer(r"^\[([^\]]+)\]\s+\S+\s+\|\s+Hardware:\s+(\S+)", out, re.M):
        if m.group(1) != "?":
            res[m.group(1)] = m.group(2)
    return res


def n_sessions():
    """Number of live sessions on the account (including foreign '[?]' ones); None if the listing failed."""
    rc, out = run(["sessions"], check=False, timeout=120)
    if rc != 0:
        return None
    return len(re.findall(r"^\[[^\]]+\]\s+\S+\s+\|\s+Hardware:", out, re.M))


def new(name, gpu=None, high_mem=False):
    args = ["new", "-s", name] + (["--gpu", gpu] if gpu else []) + (["--high-mem"] if high_mem else [])
    rc, out = run(args, check=False, timeout=1200)
    if rc != 0 or "READY" not in out:
        raise ColabError(f"new {name} {gpu} failed:\n{out[-1500:]}")


def stop(name):
    run(["stop", "-s", name], check=False, timeout=300)


def balance():
    _, out = run(["usage"], check=False, timeout=120)
    m = re.search(r"balance:\s*([\d.]+)", out)
    r = re.search(r"rate:\s*([\d.]+)", out)
    return (float(m.group(1)) if m else None, float(r.group(1)) if r else None)


# ---------- execution ----------

def exec_py(name, code, timeout=900):
    """Run code in the session kernel; returns (rc, text). Use emit() to pass back JSON."""
    return run(["exec", "-s", name], stdin=code, timeout=timeout, check=False)


def exec_retry(name, code, timeout=180, tries=4):
    """exec for short IDEMPOTENT snippets: the CLI sometimes hangs on exec, so use a short timeout and retry."""
    def once():
        rc, out = run(["exec", "-s", name], stdin=code, timeout=timeout, check=False)
        if MARK + "ok" not in out:
            raise ColabError(f"exec on {name} gave no ok marker (rc={rc}): {out[-500:]}")
        return out
    return retry(once, tries=tries, wait=5)


def exec_json(name, code, timeout=900):
    """Run code that ends by printing MARK + json; return the decoded object."""
    rc, out = exec_py(name, code, timeout)
    for line in reversed(out.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise ColabError(f"no result from exec on {name} (rc={rc}):\n{out[-2000:]}")


def alive(name):
    try:
        return exec_json(name, f"print({MARK!r}+'1')", timeout=180) == 1
    except Exception:
        return False


# ---------- files ----------

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def remote_sha(name, remote):
    code = f"""import hashlib,os
p={remote!r}
h=hashlib.sha256()
if os.path.exists(p):
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
print({MARK!r}+repr(h.hexdigest() if os.path.exists(p) else None).replace("'",'"'))"""
    return exec_json(name, code, timeout=300)


def put(name, local, remote):
    """Chunked upload with sha256 verification."""
    local = Path(local)
    size = local.stat().st_size
    if size <= CHUNK:
        retry(lambda: run(["upload", "-s", name, str(local), remote], timeout=600))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="cxput_"))
        parts = []
        with open(local, "rb") as f:
            i = 0
            while chunk := f.read(CHUNK):
                part = tmp / f"p{i:05d}"
                part.write_bytes(chunk)
                parts.append(part)
                i += 1
        rdir = f"/content/.cxup/{local.name}"
        exec_retry(name, f"import os;os.makedirs({rdir!r},exist_ok=True);print({MARK!r}+'ok')")
        for part in parts:
            retry(lambda part=part: run(["upload", "-s", name, str(part), f"{rdir}/{part.name}"], timeout=600))
        shutil.rmtree(tmp, ignore_errors=True)
        # idempotent re-assembly (safe to retry after a hung exec): build .part, rename, then drop the chunks
        exec_retry(name, f"""import os,shutil,glob
os.makedirs(os.path.dirname({remote!r}) or '.',exist_ok=True)
ps=sorted(glob.glob({rdir!r}+'/p*'))
if len(ps)=={len(parts)}:
    with open({remote!r}+'.part','wb') as o:
        for p in ps:
            with open(p,'rb') as i: shutil.copyfileobj(i,o)
    os.replace({remote!r}+'.part',{remote!r})
    shutil.rmtree({rdir!r},ignore_errors=True)
print({MARK!r}+'ok')""", timeout=300)
    if retry(lambda: remote_sha(name, remote), tries=3, wait=5) != _sha(local):
        raise ColabError(f"sha mismatch after upload {local} -> {remote}")


def get(name, remote, local, size=None):
    """Download; files above CHUNK are split on the VM and fetched piecewise (large single downloads fail),
    then re-assembled locally and verified with sha256. A snapshot that was only partly fetched is resumed
    on the next call (remote split dir + local chunk dir both carry the snapshot sha), so every poll makes
    progress even when individual chunk downloads keep failing."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if size is None or size <= CHUNK:
        retry(lambda: run(["download", "-s", name, remote, str(local)], timeout=1800))
        if not local.exists():
            raise ColabError(f"download produced no file: {remote}")
        return
    rdir = f"/content/.cxdl/{Path(remote).name}"
    info = retry(lambda: exec_json(name, f"""import os,shutil,hashlib,json
m={rdir!r}+'/manifest.json'
if os.path.exists(m):
    info=json.load(open(m))
else:
    shutil.rmtree({rdir!r},ignore_errors=True);os.makedirs({rdir!r})
    h=hashlib.sha256();n=0;sz=0
    with open({remote!r},'rb') as f:
        while True:
            b=f.read({CHUNK})
            if not b: break
            h.update(b);open({rdir!r}+'/p%05d'%n,'wb').write(b);n+=1;sz+=len(b)
    info={{"n":n,"sha":h.hexdigest(),"size":sz}}
    json.dump(info,open(m,'w'))
print({MARK!r}+json.dumps(info))""", timeout=600), tries=3, wait=5)
    ldir = Path(str(local) + ".chunks")
    man = ldir / "manifest.json"
    if not man.exists() or json.loads(man.read_text()).get("sha") != info["sha"]:
        shutil.rmtree(ldir, ignore_errors=True)
        ldir.mkdir(parents=True)
        man.write_text(json.dumps(info))
    n = info["n"]
    for i in range(n):
        part = ldir / f"p{i:05d}"
        want = CHUNK if i < n - 1 else info["size"] - CHUNK * (n - 1)
        if part.exists() and part.stat().st_size == want:
            continue
        retry(lambda part=part, i=i: run(["download", "-s", name, f"{rdir}/p{i:05d}", str(part)], timeout=600),
              tries=5, wait=10)
        if not part.exists() or part.stat().st_size != want:
            raise ColabError(f"chunk {i} of {remote} incomplete")
    with open(str(local) + ".part", "wb") as o:
        for i in range(n):
            o.write((ldir / f"p{i:05d}").read_bytes())
    if _sha(str(local) + ".part") != info["sha"]:
        shutil.rmtree(ldir, ignore_errors=True)
        raise ColabError(f"sha mismatch after chunked download {remote}")
    os.replace(str(local) + ".part", local)
    shutil.rmtree(ldir, ignore_errors=True)
    exec_py(name, f"import shutil;shutil.rmtree({rdir!r},ignore_errors=True)", timeout=120)
