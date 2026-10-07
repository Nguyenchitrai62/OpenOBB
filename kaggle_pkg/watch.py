"""Watch Kaggle kernel every 5 min; on COMPLETE/ERROR fetch outputs + notify. UTF-8 safe."""
import os, sys, json, time, subprocess, datetime

KERNEL = "nguynchtrai/floorplan-vlmdet-train"
BASE = os.path.dirname(os.path.abspath(__file__))  # kaggle_pkg/
KDIR = os.path.join(BASE, "kernel")
STATE = os.path.join(BASE, "watch_state.json")
INTERVAL = 300


def run(*args):
    e = dict(os.environ); e["PYTHONUTF8"] = "1"; e["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-m", "kaggle"] + list(args),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=e)
    return r


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(BASE, "watch.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)


def trigger_research(reason):
    try:
        r = subprocess.run([sys.executable, r"F:\Source_code\NEW_architecture\research\research_loop.py"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log(f"research_loop({reason}): {(r.stdout or '')[-500:]}")
    except Exception as ex:
        log(f"research trigger error: {ex}")


def parse_log(path):
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
        evts = json.loads(raw)
        full = "".join(e.get("data", "") for e in evts if e.get("stream_name") == "stdout")
        return full
    except Exception as ex:
        return f"<parse error {ex}>"


def notify(title, msg):
    """Toast Windows + ghi CHAT_FEED (nhat ky de agent bao cao trong chat)."""
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(BASE, "CHAT_FEED.md"), "a", encoding="utf-8") as f:
            f.write(f"- [{ts}] {title}: {msg}\n")
    except Exception:
        pass
    if os.environ.get("SCHED_ONCE") != "1":
        return  # chi toast khi chay tu Task Scheduler (tranh double)
    try:
        import subprocess as sp
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
            "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(1);"
            f"$t.GetElementsByTagName('text')[0].AppendChild($t.CreateTextNode({title!r})) | Out-Null;"
            f"$t.GetElementsByTagName('text')[1].AppendChild($t.CreateTextNode({msg!r})) | Out-Null;"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('CAD-VLMDet').Show($t)"
        )
        sp.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps],
               capture_output=True, timeout=30)
    except Exception:
        pass


def heartbeat(status, extra=""):
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        short = status.split(chr(34))[-2] if chr(34) in status else status
        open(os.path.join(BASE, "LIVE_STATUS.md"), "w", encoding="utf-8").write(
            f"# LIVE - auto-check Kaggle (5 phut/lan)\n\n"
            f"- Lan check cuoi: **{ts}**\n- Kernel `floorplan-vlmdet-train`: **{short}**\n"
            f"- Tien trinh nen + Task Scheduler `FloorplanVLMDetWatch` deu dang chay.\n"
            f"- {extra}\n"
            f"- Log chi tiet: `kaggle_pkg/watch.log` | Metrics: `kaggle_pkg/REPORT_METRICS.txt`\n")
    except Exception:
        pass


def main_once():
    r = run("kernels", "status", KERNEL)
    out = (r.stdout + r.stderr).strip().splitlines()
    status = out[-1] if out else "UNKNOWN"
    log(f"status: {status}")
    heartbeat(status, "v6 (fix level-assign + focal cls) dang train tren T4, uoc ~3.5h.")
    short = status.split(chr(34))[-2] if chr(34) in status else status
    notify("Kaggle check", f"{short} (kernel floorplan-vlmdet-train)")
    st = state()
    if "COMPLETE" in status and not st.get("done"):
        log("COMPLETE -> downloading outputs")
        d = os.path.join(KDIR, "final")
        os.makedirs(d, exist_ok=True)
        r2 = run("kernels", "output", KERNEL, "-p", d, "--force")
        log(f"download rc={r2.returncode}")
        lp = os.path.join(d, "floorplan-vlmdet-train.log")
        if os.path.exists(lp):
            full = parse_log(lp)
            open(os.path.join(BASE, "REPORT_GPU.txt"), "w", encoding="utf-8").write(full)
            # trich so quan trong
            keys = [l for l in full.splitlines() if any(k in l for k in
                    ["epoch", "mAP", "Precision", "Recall", "all", "wall", "window", "door", "junction", "loss"])]
            open(os.path.join(BASE, "REPORT_METRICS.txt"), "w", encoding="utf-8").write("\n".join(keys[-60:]))
            log(f"metrics lines: {len(keys)} -> see REPORT_METRICS.txt")
        st["done"] = True; st["status"] = "COMPLETE"; save_state(st)
        trigger_research("COMPLETE")
        return "DONE"
    if "ERROR" in status and not st.get("done"):
        log("ERROR -> downloading log for diagnosis")
        d = os.path.join(KDIR, "err")
        os.makedirs(d, exist_ok=True)
        run("kernels", "output", KERNEL, "-p", d, "--force")
        st["done"] = True; st["status"] = "ERROR"; save_state(st)
        trigger_research("ERROR")
        return "ERROR"
    return "WAIT"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    if mode == "once":
        print(main_once())
    else:
        log("watcher started, interval 300s")
        while True:
            try:
                res = main_once()
                if res in ("DONE", "ERROR"):
                    log(f"{res}: giu vong lap, cho version kernel moi (reset state khi day kernel).")
            except Exception as ex:
                log(f"watch error: {ex}")
            time.sleep(INTERVAL)
