#!/usr/bin/env python3
r"""Does a VRAM ballast really limit what another process can allocate under WDDM?  (goal G8)

Runs tools/vram-probe.cu (allocate 256 MiB chunks until cudaMalloc fails) against a vram-ballast process holding N MiB, once idle and once pinned by a persistent kernel.
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag ballast-selftest -Command \
    "python tools\ballast-selftest.py --mib 16000 --attest build\llama-cuda\bin\g15\llama-layer-streaming.exe"
It records; it claims nothing.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "llama-cuda", "bin", "tools")


def run_case(mib, pin):
    args = [os.path.join(BIN, "vram-ballast.exe"), str(mib), "512"] + (["pin"] if pin else [])
    b = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    line = b.stdout.readline().strip()
    print("ballast:", line, flush=True)
    time.sleep(2.0)
    out = subprocess.run([os.path.join(BIN, "vram-probe.exe"), "30000"], capture_output=True, text=True, timeout=300).stdout.strip()
    print("probe  :", out, flush=True)
    b.stdin.close()
    try:
        b.wait(timeout=60)
    except Exception:
        b.kill()
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=16000)
    ap.add_argument("--attest", nargs="*", default=[])
    a = ap.parse_args()
    print("== idle ballast")
    run_case(a.mib, False)
    print("== pinned ballast (persistent kernel)")
    run_case(a.mib, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
