import subprocess, os, signal, sys
out = subprocess.run(["wmic", "process", "where", 'name="python.exe"', "get", "ProcessId,CommandLine"],
                     capture_output=True, text=True).stdout
me = os.getpid()
killed = []
for line in out.splitlines():
    if "watch.py" in line and "kill_watch" not in line:
        parts = line.strip().split()
        try:
            pid = int(parts[-1])
        except Exception:
            continue
        if pid != me:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except Exception as ex:
                print("fail", pid, ex)
print("killed:", killed)
