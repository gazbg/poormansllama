#!/usr/bin/env python3
"""Tabulate the rare single-step decode stalls of a stall-capture session (host-only; reads a tools/run-tg-arms.py session directory).

For every run it lists the decode steps longer than --min-ms (default 60) from the runner telemetry, and the engine's own diagnostics from the run log:
"slow host request" lines (stage times of a request that took > 5 ms), "slow DMA submission" lines, watchdog reports (phase, heartbeat age). It groups by arm
and prints events per 1000 steps, so arms that differ only in an environment variable can be compared.

  python tools/analyze-stalls.py build/llama-cuda/g7-stall [--min-ms 60]

It measures; it claims nothing.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def rtext(p):
    with open(p, "rb") as fh:
        return fh.read().replace(b"\x00", b"").decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--min-ms", type=float, default=60.0)
    a = ap.parse_args()
    session = os.path.basename(a.session_dir.rstrip("/\\"))
    runs = defaultdict(list)
    for jp in sorted(glob.glob(os.path.join(a.session_dir, session + "-*.jsonl"))):
        stem = os.path.basename(jp)[len(session) + 1:-6]          # ARM_rN
        if "hybrid" in stem:
            continue
        arm = stem.rsplit("_r", 1)[0]
        steps = []
        for l in rtext(jp).splitlines():
            if '"event":"tg_logits"' in l:
                j = json.loads(l)
                steps.append((j["step"], j["t_us"]))
        steps.sort()
        spikes = [(steps[i][0], (steps[i][1] - steps[i - 1][1]) / 1e3) for i in range(1, len(steps))
                  if (steps[i][1] - steps[i - 1][1]) / 1e3 > a.min_ms and steps[i][0] != 1]
        logp = jp[:-6] + ".log"
        log = rtext(logp) if os.path.exists(logp) else ""
        slow = re.findall(r"slow host request #(\d+): layer (\d+) nmiss (\d+) took ([0-9.]+) ms \(copy-in ([0-9.]+), graph compute ([0-9.]+), rest ([0-9.]+)\), dma_state (\d+)", log)
        dma = re.findall(r"slow DMA submission #(\d+) took ([0-9.]+) ms", log)
        wd = re.findall(r"watchdog: request (\d+) unanswered for ([0-9.]+) ms \(res_seq (\d+), layer (-?\d+), nmiss (-?\d+)\); host engine phase (\d+)\.(\d+) "
                        r"for ([0-9.]+) ms \(last request seen (\d+)\), heartbeat age ([0-9.]+) ms", log)
        runs[arm].append({"run": stem, "steps": len(steps), "spikes": spikes, "slow": slow, "dma": dma, "wd": wd})
    for arm, rs in sorted(runs.items()):
        n_steps = sum(r["steps"] for r in rs)
        n_sp = sum(len(r["spikes"]) for r in rs)
        n_runs_sp = sum(1 for r in rs if r["spikes"])
        print("== %s: %d runs, %d steps, %d runs with a step > %.0f ms, %d spikes = %.2f per 1000 steps" % (
            arm, len(rs), n_steps, n_runs_sp, a.min_ms, n_sp, 1000.0 * n_sp / max(n_steps, 1)))
        for r in rs:
            if r["spikes"] or r["slow"] or r["dma"] or r["wd"]:
                print("   %-10s spikes %s" % (r["run"], [(s, int(ms)) for s, ms in r["spikes"]]))
                for s in r["slow"]:
                    print("      slow request #%s layer %s nmiss %s: total %s ms (copy-in %s, graph compute %s, rest %s) dma_state %s" % s)
                for d in r["dma"]:
                    print("      slow DMA submission #%s: %s ms" % d)
                for w in r["wd"]:
                    print("      watchdog: req %s pending %s ms layer %s nmiss %s | engine phase %s.%s for %s ms | heartbeat age %s ms" % (
                        w[0], w[1], w[3], w[4], w[5], w[6], w[7], w[9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
