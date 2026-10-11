#!/usr/bin/env python3
r"""Does the GPU run at full clocks during the hybrid decode? (goal G-TG8: "better GPU utilization")

Runs one arm of a tools/run-tg-arms.py spec while a 50 ms nvidia-smi logger records SM / memory clocks, power, temperature and the throttle-reason bitmask for the
whole run, then summarises the samples taken while the GPU was busy (utilization.gpu > 0 and memory.used above the idle floor) against the card's maximum clocks.
Inside one mutex hold:

  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g8-clocks -Command \
    "python tools\gpu-clock-probe.py --spec build\llama-cuda\specs\g8-clk.json --attest build\llama-cuda\bin\g8\llama-layer-streaming.exe"

It records; it claims nothing.
"""
import argparse
import csv
import io
import os
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = "timestamp,clocks.sm,clocks.mem,clocks.gr,clocks.max.sm,clocks.max.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--attest", nargs="*", default=[])
    a = ap.parse_args()
    rows = []
    stop = threading.Event()
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=" + FIELDS, "--format=csv,noheader,nounits", "--loop-ms=50"], stdout=subprocess.PIPE, text=True)

    def reader():
        for ln in p.stdout:
            rows.append(ln.strip())
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.time()
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "run-tg-arms.py"), "--spec", a.spec], cwd=ROOT).returncode
    p.kill()
    th.join(timeout=5)
    print("run rc=%s wall %.1fs, %d clock samples" % (rc, time.time() - t0, len(rows)))
    recs = list(csv.reader(io.StringIO("\n".join(rows))))
    cols = FIELDS.split(",")
    data = []
    for r in recs:
        if len(r) == len(cols):
            d = dict(zip(cols, [x.strip() for x in r]))
            data.append(d)
    if not data:
        print("no samples")
        return 1
    def num(d, k):
        try:
            return float(d[k])
        except Exception:
            return None
    idle_mem = min(num(d, "memory.used") or 1e9 for d in data)
    busy = [d for d in data if (num(d, "utilization.gpu") or 0) > 0 and (num(d, "memory.used") or 0) > idle_mem + 3000]
    print("busy samples: %d of %d (idle floor %.0f MiB)" % (len(busy), len(data), idle_mem))
    if busy:
        for k in ("clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu"):
            v = [num(d, k) for d in busy if num(d, k) is not None]
            print("  %-16s median %8.1f  min %8.1f  max %8.1f   (card max sm %s mem %s)" % (k, statistics.median(v), min(v), max(v), busy[0]["clocks.max.sm"], busy[0]["clocks.max.mem"]))
        from collections import Counter
        print("  throttle reasons active:", Counter(d["clocks_throttle_reasons.active"] for d in busy).most_common(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
