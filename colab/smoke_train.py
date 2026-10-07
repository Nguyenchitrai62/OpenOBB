"""Fake training loop used to test tools/job.py (resume, sync, VM death)."""
import argparse, json, os, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--sec", type=int, default=60)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
last = os.path.join(a.out, "last.pt")
start = 0
if os.path.exists(last):
    start = torch.load(last)["epoch"] + 1
    print(f"resumed from epoch {start}", flush=True)
if start == 0:  # quick bf16 matmul throughput probe, logged once
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50): y = x @ x
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory>>30}GB "
          f"bf16={2*8192**3*50/dt/1e12:.1f} TFLOPS", flush=True)
for ep in range(start, a.epochs):
    time.sleep(a.sec)
    torch.save({"epoch": ep, "w": torch.randn(10 * 2**20 // 4 * 4)}, last)  # ~40 MB -> chunked upload on resume
    with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": ep, "loss": 1 / (ep + 1), "gpu": torch.cuda.get_device_name(0)}) + "\n")
    print(f"epoch {ep} done", flush=True)
