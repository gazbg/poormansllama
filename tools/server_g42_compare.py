#!/usr/bin/env python3
"""G42: compare the per-request wall time (launch_slot_ -> stop processing, from server.log) of several arms, each arm = one or more replicate sessions.

  python tools/server_g42_compare.py base=k5201-base-def,k5208-base-def fuse=k5202-fuse-def,k5207-fuse-def [--base base] [--dir build/llama-cuda]

Prints, per request index, the mean wall of every arm and its ratio to --base (default: the first arm), then the mean over the requests. Host-only, stdlib only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_phase_timeline import events  # noqa: E402


def walls(path):
    out = []
    t0 = None
    for t, s in events(path):
        if "launch_slot_" in s and "processing task" in s:
            t0 = t
        elif "stop processing" in s and t0 is not None:
            out.append(t - t0)
            t0 = None
    return out


def main():
    root = "build/llama-cuda"
    base = None
    arms = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--base":
            base = args[i + 1]
            i += 2
        elif args[i] == "--dir":
            root = args[i + 1]
            i += 2
        else:
            name, dirs = args[i].split("=", 1)
            arms.append((name, dirs.split(",")))
            i += 1
    if base is None:
        base = arms[0][0]
    data = {}
    for name, dirs in arms:
        runs = [walls(os.path.join(root, d, "server.log")) for d in dirs]
        n = min(len(r) for r in runs)
        data[name] = [sum(r[k] for r in runs) / len(runs) for k in range(n)]
        spread = max(abs(sum(r) / len(r) - sum(data[name]) / n) for r in [rr[:n] for rr in runs]) if len(runs) > 1 else 0.0
        print("arm %-12s %d replicate(s), %d requests, max deviation of a replicate's mean from the arm mean %.3f s" % (name, len(runs), n, spread))
    n = min(len(v) for v in data.values())
    names = [a[0] for a in arms]
    print("%-4s" % "req" + "".join("%14s" % nm for nm in names) + "".join("%12s" % (nm + "/b") for nm in names if nm != base))
    for k in range(n):
        row = "%-4d" % k + "".join("%14.2f" % data[nm][k] for nm in names)
        row += "".join("%12.3f" % (data[nm][k] / data[base][k]) for nm in names if nm != base)
        print(row)
    mean = {nm: sum(data[nm][:n]) / n for nm in names}
    print("%-4s" % "mean" + "".join("%14.3f" % mean[nm] for nm in names) + "".join("%12.3f" % (mean[nm] / mean[base]) for nm in names if nm != base))


if __name__ == "__main__":
    main()
