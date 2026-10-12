#!/usr/bin/env python3
"""One table for a multi-arm, counterbalanced session: per arm the mean TG tok/s, the paired gain over the control (percent, mean and per pair), the positive pairs, the engine's hit rate,
promotions per token and serve time per request, the stall and gap of the token, and the poller's VRAM peak.

    python tools/pol-table.py <session verdict.json> [control arm]

Everything is read from the verdict and the arm logs (the engine's shutdown lines); nothing is estimated.
"""
import glob
import json
import os
import re
import statistics
import sys


def text(path):
    try:
        return open(path, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    except OSError:
        return ""


def engine(log):
    out = {}
    m = re.search(r"hit rate ([\d.]+) \((\d+)/(\d+)\), promotions/token ([\d.]+).*?host requests (\d+), serve avg ([\d.]+) max ([\d.]+) us", log)
    if m:
        out.update(hit=float(m.group(1)), prom=float(m.group(4)), serve=float(m.group(6)))
    m = re.search(r"timeline per token over (\d+) tokens \(us\): span ([\d.]+) = attn ([\d.]+) \+ ovl ([\d.]+) \+ stall ([\d.]+), gap ([\d.]+)", log)
    if m:
        out.update(span=float(m.group(2)), stall=float(m.group(5)), gap=float(m.group(6)))
    m = re.search(r"non-uniform slot split from \S+ \(alpha ([\d.]+)\): (\d+) \.\. (\d+) slots", log)
    if m:
        out["split"] = "%s..%s" % (m.group(2), m.group(3))
    out["slow"] = len(re.findall(r"slow host request", log))
    return out


def main():
    d = json.load(open(sys.argv[1]))
    ctl = sys.argv[2] if len(sys.argv) > 2 else d["spec"]["control"]
    arms = {}
    for k, v in d["arms"].items():
        arms.setdefault(v["base"], {})[v["rep"]] = v
    names = [n for n in arms]
    print("session %s | %s | control %s" % (d["session"], d["power"]["ac_line"], ctl))
    print("%-7s %9s %9s %7s %-30s %6s %6s %7s %7s %7s %5s %s" % ("arm", "tok/s", "min-max", "gain%", "paired gains", "pos", "hit", "prom/t", "serve", "stall", "slow", "peakMiB"))
    base = arms[ctl]
    for n in names:
        reps = sorted(arms[n])
        tok = [arms[n][r]["summary"]["tg_tokens_per_s"] for r in reps]
        gains = []
        for r in reps:
            if r in base:
                gains.append(100.0 * (arms[n][r]["summary"]["tg_tokens_per_s"] / base[r]["summary"]["tg_tokens_per_s"] - 1.0))
        es = [engine(text(arms[n][r]["log"])) for r in reps]
        mean = lambda key: statistics.mean(e[key] for e in es if key in e) if any(key in e for e in es) else float("nan")
        split = next((e["split"] for e in es if "split" in e), "")
        print("%-7s %9.2f %4.1f-%4.1f %+7.2f %-30s %3d/%d %6.4f %7.1f %7.1f %7.0f %5d %s%s" % (
            n, statistics.mean(tok), min(tok), max(tok), statistics.mean(gains) if gains else 0.0,
            " ".join("%+.1f" % g for g in gains) if n != ctl else "-", sum(1 for g in gains if g > 0) if n != ctl else 0, len(gains) if n != ctl else 0,
            mean("hit"), mean("prom"), mean("serve"), mean("stall"), sum(e["slow"] for e in es), max(arms[n][r].get("vram_peak_mib", 0) for r in reps),
            ("  split " + split) if split else ""))


if __name__ == "__main__":
    main()
