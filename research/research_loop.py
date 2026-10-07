"""Research loop: doc evidence moi tu watcher -> quyet dinh huong tiep theo.
Chay moi khi co REPORT_METRICS.txt moi (watcher goi) hoac chay tay.
Khong tu day kernel khi chua co evidence (tranh dot quota GPU).
"""
import os, json, datetime

BASE = r"F:\Source_code\NEW_architecture"
PKG = os.path.join(BASE, "kaggle_pkg")
METRICS = os.path.join(PKG, "REPORT_METRICS.txt")
DECISIONS = os.path.join(BASE, "research", "decisions.log")


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DECISIONS, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def main():
    if not os.path.exists(METRICS):
        log("chua co evidence GPU (kernel v4 RUNNING). Cho watcher 5-phut. Khong dot quota.")
        return "WAIT"
    txt = open(METRICS, encoding="utf-8", errors="replace").read()
    if not txt.strip():
        log("REPORT_METRICS rong -> doi evidence day du.")
        return "WAIT"
    # phan tich so bo: tim dong mAP
    maps = [l for l in txt.splitlines() if "mAP" in l or "all" in l]
    log(f"evidence co {len(txt.splitlines())} dong, {len(maps)} dong mAP.")
    for l in maps[-10:]:
        log("EVID: " + l[:200])
    log("De nghi: so sanh VLMDet-small vs yolo11x cung dieu kien; neu cua hiem/junction hon >=2 diem -> CHOT D4 (base full 16k); nguoc lai -> D1/D2/D3 theo class loi nhat.")
    return "ANALYZED"


if __name__ == "__main__":
    print(main())
