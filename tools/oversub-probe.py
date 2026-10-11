#!/usr/bin/env python3
r"""Why does the hybrid engine's CPU miss graph take ~78 ms per call when every logical CPU is busy? (CPU-only; no GPU, no model)

Starts N busy-loop python processes (normal priority) and runs the engine-shaped microbenchmark (moe-cpu-bench) under the same threadpool settings as the
engine, for one or more configurations, printing microseconds per call. It records; it claims nothing.

  python tools/oversub-probe.py --dir build/llama-cuda/bin/g7 --loads 0,8,16,24 [--threads 8 --first 0 --stride 2] [--ms 4]
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench(exe, cwd, threads, first, stride, m, iters):
    out = subprocess.run([exe, "--threads", str(threads), "--first", str(first), "--stride", str(stride), "--pfs", "0", "--ms", str(m),
                          "--layers", "3", "--iters", str(iters)], cwd=cwd, capture_output=True, text=True, timeout=900).stdout
    res = {}
    for kind in ("hot", "cold"):
        mm = re.search(r"%s\s+m=%d\s+mean\s+([0-9.]+) us\s+p50\s+([0-9.]+)\s+p10\s+([0-9.]+)\s+p90\s+([0-9.]+)" % (kind, m), out)
        if mm:
            res[kind] = tuple(float(x) for x in mm.groups())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("build", "llama-cuda", "bin", "g7"))
    ap.add_argument("--loads", default="0,8,16,24")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ms", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    d = os.path.join(ROOT, a.dir)
    exe = os.path.join(d, "moe-cpu-bench.exe")
    for n in [int(x) for x in a.loads.split(",")]:
        procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]
        time.sleep(1.5)
        t0 = time.time()
        r = bench(exe, d, a.threads, a.first, a.stride, a.ms, a.iters)
        dt = time.time() - t0
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        print("busy-loop processes %2d | threads %d first %d stride %d m=%d | hot mean/p50/p10/p90 us %s | cold %s | wall %.1fs" % (
            n, a.threads, a.first, a.stride, a.ms, r.get("hot"), r.get("cold"), dt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
