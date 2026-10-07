"""Unattended Colab job runner: launch, babysit, resume after VM death, sync, stop.

A job is research/jobs/<id>.json:
{
  "id": "e1-dior-s",            # also the Colab session name prefix
  "gpu": "A100",                # T4 | L4 | A100 | H100
  "bundle": ["vrdet", "colab"], # local dirs shipped to /content/work/code
  "setup": ["pip install -q -r colab/requirements.txt", "bash colab/data/dior_r.sh"],
  "cmd": "python -u -m vrdet.train --cfg cfg/x.yaml --out {out}",
  "sync": ["last.pt", "best.pt", "*.json", "*.jsonl", "*.log", "*.png"],
  "max_hours": 10, "max_retries": 3
}
Conventions the training code MUST follow:
  * write everything under {out} (= /content/work/runs/<id>);
  * resume automatically when {out}/last.pt exists;
  * setup steps must be idempotent (they rerun on every new VM).

CLI:
  python tools/job.py launch <id>   # start (or restart) the job on a fresh VM
  python tools/job.py poll <id>     # one babysit step: sync, detect end/death
  python tools/job.py watch [ids]   # loop poll over active jobs; exits when one ends
  python tools/job.py stop <id>     # final sync + release the VM
  python tools/job.py status
"""
import fnmatch
import io
import json
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "research" / "jobs"
RUNS = ROOT / "runs"
WORK = "/content/work"
RESERVE_CU = 40          # never let the balance drop below this
POLL_MIN = float(__import__("os").environ.get("JOB_POLL_MIN", 10))            # minutes between polls in watch mode
LOG = JOBS / "watch.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def spec(jid):
    return json.loads((JOBS / f"{jid}.json").read_text(encoding="utf-8"))


def state(jid):
    p = JOBS / f"{jid}.state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(jid, st):
    st["updated"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    (JOBS / f"{jid}.state.json").write_text(json.dumps(st, indent=2), encoding="utf-8")


def _bundle(dirs):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in dirs:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pt", ".zip"}:
                    tar.add(p, arcname=str(p.relative_to(ROOT)).replace("\\", "/"))
    path = JOBS / ".bundle.tar.gz"
    path.write_bytes(buf.getvalue())
    return path


def _start_remote(sess, sp, out):
    """Ship code, run setup, push last.pt if we have one, start cmd detached."""
    cx.exec_py(sess, f"import os,shutil;shutil.rmtree('{WORK}/code',ignore_errors=True);"
                     f"os.makedirs('{WORK}/code',exist_ok=True);os.makedirs('{out}',exist_ok=True)")
    cx.put(sess, _bundle(sp["bundle"]), f"{WORK}/bundle.tar.gz")
    cx.exec_py(sess, f"import tarfile;tarfile.open('{WORK}/bundle.tar.gz').extractall('{WORK}/code')")
    # restore everything already synced (last.pt, metrics.jsonl, logs...) so the
    # resumed run appends to its history instead of starting new files
    local_dir = RUNS / sp["id"]
    pats = sp.get("sync", ["*"]) + ["train.log"]
    restore = [p for p in local_dir.rglob("*") if p.is_file()
               and any(fnmatch.fnmatch(p.name, pat) for pat in pats)] if local_dir.exists() else []
    if restore:
        mb = sum(p.stat().st_size for p in restore) >> 20
        log(f"{sp['id']}: restoring {len(restore)} synced files ({mb} MB) for resume")
        for p in restore:
            cx.put(sess, p, f"{out}/{p.relative_to(local_dir).as_posix()}")
    for step in sp.get("setup", []):
        res = cx.exec_json(sess, f"""import subprocess
r=subprocess.run({step!r},shell=True,cwd='{WORK}/code',capture_output=True,text=True)
open('{out}/setup.log','a').write('$ '+{step!r}+'\\n'+r.stdout[-20000:]+r.stderr[-20000:]+'\\n')
print({cx.MARK!r}+str(r.returncode))""", timeout=7200)
        if res != 0:
            raise cx.ColabError(f"setup step failed (rc={res}): {step}")
    cmd = sp["cmd"].format(out=out)
    cx.exec_py(sess, f"""import subprocess,os
os.makedirs('{out}',exist_ok=True)
for f in ('exitcode',):
    p=os.path.join('{out}',f)
    if os.path.exists(p): os.remove(p)
sh="cd {WORK}/code && ( " + {cmd!r} + " ) >> {out}/train.log 2>&1; echo $? > {out}/exitcode"
p=subprocess.Popen(['bash','-c',sh],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
open('{out}/pid','w').write(str(p.pid))""")


def launch(jid):
    sp, st = spec(jid), state(jid)
    bal, _ = cx.balance()
    if bal is not None and bal < RESERVE_CU:
        raise SystemExit(f"balance {bal} CU < reserve {RESERVE_CU}; refusing to launch")
    st.setdefault("attempts", 0)
    st["attempts"] += 1
    st.setdefault("started", f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    sess = f"{jid}-{st['attempts']}"
    out = f"{WORK}/runs/{jid}"
    log(f"{jid}: launching attempt {st['attempts']} on {sp['gpu']} as session {sess}")
    cx.new(sess, sp.get("gpu"), sp.get("high_mem", False))
    st.update(session=sess, out=out, status="running", launched=time.time())
    save_state(jid, st)
    try:
        _start_remote(sess, sp, out)
    except Exception as e:
        log(f"{jid}: start failed: {e}")
        st["status"] = "start_failed"
        save_state(jid, st)
        cx.stop(sess)
        raise
    log(f"{jid}: running")


def _remote_status(sess, out, patterns):
    code = f"""import os,json,glob,fnmatch
o='{out}'
files={{}}
for p in glob.glob(o+'/**',recursive=True):
    if os.path.isfile(p):
        r=os.path.relpath(p,o)
        if any(fnmatch.fnmatch(os.path.basename(r),pat) for pat in {patterns!r}):
            st=os.stat(p); files[r]=[st.st_size,int(st.st_mtime)]
ec=open(o+'/exitcode').read().strip() if os.path.exists(o+'/exitcode') else None
pid=open(o+'/pid').read().strip() if os.path.exists(o+'/pid') else None
alive=bool(pid) and os.path.exists('/proc/'+pid)
tail=open(o+'/train.log',errors='replace').read()[-1500:] if os.path.exists(o+'/train.log') else ''
print({cx.MARK!r}+json.dumps(dict(files=files,exitcode=ec,alive=alive,tail=tail)))"""
    return cx.exec_json(sess, code, timeout=150)


def _sync(jid, sess, out, files, st):
    seen = st.setdefault("synced", {})
    for rel, meta in files.items():
        if seen.get(rel) != meta:
            dst = RUNS / jid / rel
            cx.get(sess, f"{out}/{rel}", dst)
            seen[rel] = meta


def poll(jid):
    """Returns the job status after one babysit step."""
    sp, st = spec(jid), state(jid)
    if st.get("status") != "running":
        return st.get("status")
    sess, out = st["session"], st["out"]
    try:
        rs = _remote_status(sess, out, sp.get("sync", ["*"]) + ["train.log", "setup.log", "exitcode"])
        st["fails"] = 0
    except Exception as e:
        rs = None
        st["fails"] = st.get("fails", 0) + 1
        gone = sess not in cx.sessions()
        log(f"{jid}: status failed #{st['fails']} (session listed={not gone}): {str(e)[:200]}")
        if not gone and st["fails"] < 3:
            save_state(jid, st)   # transient: exec can hang while the VM is fine
            return "running"
    if rs is None:
        st["fails"] = 0
        if st["attempts"] > sp.get("max_retries", 3):
            st["status"] = "failed_vm"
            save_state(jid, st)
            return st["status"]
        cx.stop(sess)
        st["status"] = "resuming"
        save_state(jid, st)
        launch(jid)
        return "running"
    _sync(jid, sess, out, rs["files"], st)
    st["tail"] = rs["tail"][-600:]
    hours = (time.time() - st.get("launched", time.time())) / 3600
    if rs["exitcode"] is not None:
        st["status"] = "done" if rs["exitcode"] == "0" else f"failed_rc{rs['exitcode']}"
    elif not rs["alive"]:
        st["status"] = "failed_died"
    elif hours > sp.get("max_hours", 12):
        st["status"] = "timeout"
    save_state(jid, st)
    if st["status"] != "running":
        log(f"{jid}: finished with status {st['status']}; stopping VM")
        cx.stop(sess)
    return st["status"]


def stop(jid):
    st = state(jid)
    if st.get("session"):
        try:
            sp = spec(jid)
            rs = _remote_status(st["session"], st["out"], sp.get("sync", ["*"]) + ["train.log"])
            _sync(jid, st["session"], st["out"], rs["files"], st)
        except Exception as e:
            log(f"{jid}: final sync failed: {e}")
        cx.stop(st["session"])
    st["status"] = "stopped"
    save_state(jid, st)


def active():
    return [p.name[:-11] for p in JOBS.glob("*.state.json")
            if json.loads(p.read_text(encoding="utf-8")).get("status") == "running"]


def watch(ids=None, max_hours=8):
    """Babysit until any job leaves 'running' (or max_hours), then exit so the agent wakes up."""
    t0 = time.time()
    while True:
        ids_now = ids or active()
        if not ids_now:
            log("watch: no active jobs")
            return
        bal, rate = cx.balance()
        if bal is not None and bal < RESERVE_CU:
            log(f"watch: balance {bal} < reserve {RESERVE_CU}; stopping all jobs")
            for j in ids_now:
                stop(j)
            return
        ended = []
        for j in ids_now:
            try:
                s = poll(j)
            except Exception as e:
                log(f"{j}: poll error {e}")
                continue
            if s != "running":
                ended.append((j, s))
        log(f"watch: balance={bal} rate={rate}/h active={ids_now} ended={ended}")
        if ended or time.time() - t0 > max_hours * 3600:
            return
        time.sleep(POLL_MIN * 60)


def status():
    for p in sorted(JOBS.glob("*.state.json")):
        st = json.loads(p.read_text(encoding="utf-8"))
        print(f"{p.name[:-11]:30s} {st.get('status'):15s} attempts={st.get('attempts')} updated={st.get('updated')}")
    print("balance/rate:", cx.balance())


def _lock():
    """One babysitter at a time: concurrent execs on a kernel can hang."""
    import os
    lk = JOBS / ".lock"
    JOBS.mkdir(parents=True, exist_ok=True)
    if lk.exists() and time.time() - lk.stat().st_mtime < 3 * 3600:
        pid = lk.read_text().strip()
        # os.kill(pid, 0) would TERMINATE the process on Windows, so ask tasklist
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        if pid.isdigit() and pid in out:
            raise SystemExit(f"another job.py (pid {pid}) holds {lk}")
    lk.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: lk.unlink(missing_ok=True))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd != "status":
        _lock()
    {"launch": lambda: launch(args[0]), "poll": lambda: print(poll(args[0])),
     "watch": lambda: watch(args or None), "stop": lambda: stop(args[0]),
     "status": status}[cmd]()
