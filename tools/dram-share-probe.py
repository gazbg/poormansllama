#!/usr/bin/env python3
r"""Does PCIe DMA from host memory take DRAM bandwidth away from the CPU miss path? (goal G-TG8, ledger row on GPU-assisted miss service)

Runs the engine-shaped CPU miss microbenchmark (moe-cpu-bench) alone, then again while a background host->device copy loop (h2d-load) streams pinned
memory to the GPU, and prints the per-expert cold cost of both. If the DRAM ceiling is shared, the cost rises by about the copy bandwidth's share.
It records; it claims nothing. GPU work: run it inside one mutex hold, naming a binary of the build on the wrapped command line:

  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag dram-share -Command \
    "python tools\dram-share-probe.py --dir build\llama-cuda\bin\g6 --attest build\llama-cuda\bin\g6\llama-layer-streaming.exe"
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench(exe, cwd, threads, first, stride):
    out = subprocess.run([exe, "--threads", str(threads), "--first", str(first), "--stride", str(stride), "--pfs", "0", "--ms", "2,4,8",
                          "--layers", "6", "--iters", "300"], cwd=cwd, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"fit cold: ([0-9.]+) \+ ([0-9.]+) \* m us", out)
    h = re.search(r"fit hot : ([0-9.]+) \+ ([0-9.]+) \* m us", out)
    return (float(m.group(1)), float(m.group(2))) if m else None, (float(h.group(1)), float(h.group(2))) if h else None, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("build", "llama-cuda", "bin", "g6"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--attest", nargs="*", default=[])
    a = ap.parse_args()
    d = os.path.join(ROOT, a.dir)
    bexe = os.path.join(d, "moe-cpu-bench.exe")
    lexe = os.path.join(d, "h2d-load.exe")
    for p in (bexe, lexe):
        if not os.path.exists(p):
            print("MISSING", p)
            return 2
    rows = []
    for rep in range(a.reps):
        c0, h0, _ = bench(bexe, d, a.threads, a.first, a.stride)
        load = subprocess.Popen([lexe, "25", "256"], cwd=d, stdout=subprocess.PIPE, text=True)
        time.sleep(3.0)                      # the copy loop is at full rate before the bench starts
        c1, h1, _ = bench(bexe, d, a.threads, a.first, a.stride)
        txt = load.communicate(timeout=120)[0]
        gbs = re.search(r"= ([0-9.]+) GB/s", txt)
        rows.append((c0, h0, c1, h1, float(gbs.group(1)) if gbs else None))
        print("rep %d: cold alone %s | cold with h2d load %s | hot alone %s | hot with load %s | h2d %s GB/s" % (
            rep + 1, c0, c1, h0, h1, rows[-1][4]), flush=True)
    ok = [r for r in rows if r[0] and r[2]]
    if ok:
        b0 = sum(r[0][1] for r in ok) / len(ok)
        b1 = sum(r[2][1] for r in ok) / len(ok)
        g = [r[4] for r in ok if r[4]]
        print("\nmean per-expert cold cost: alone %.2f us, with the copy loop %.2f us (%+.1f%%); copy loop %.2f GB/s" % (
            b0, b1, 100.0 * (b1 / b0 - 1.0), sum(g) / len(g) if g else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
