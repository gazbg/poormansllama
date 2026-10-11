#!/usr/bin/env python3
"""Tabulate host-engine stalls per run from the hybrid engine's shutdown summary (host-only, stdlib-only).

Each LM/plain run that used the hybrid engine ends with a log line
  moe_hybrid: tokens N, hit rate ..., on_token avg A max M us | host requests R, serve avg S max X us, nmiss hist ...
`serve max` is the worst time the host engine took to answer ONE request. A healthy run has 2-4 ms; a stalled run has tens or hundreds of ms.
A session spec's arms are grouped by base name, so stall frequency can be compared between arms.

  python tools/hybrid-stalls.py build/llama-cuda/g3-soak [--threshold-ms 20]
  python tools/hybrid-stalls.py --selftest
"""
import argparse
import glob
import os
import re
import statistics
import sys

PAT = re.compile(r"host requests (\d+), serve avg ([\d.]+) max ([\d.]+) us")
WD = re.compile(r"moe_hybrid watchdog")


def run_rows(session_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(session_dir, "*-*.log")), key=os.path.getmtime):
        base = os.path.basename(f)
        m_arm = re.match(r".*?-(?:[^-]+-)?([A-Za-z0-9]+)_r(\d+)\.log$", base)
        text = open(f, encoding="utf-8", errors="replace").read().replace("\x00", "")
        m = PAT.search(text)
        if not m:
            continue
        arm = m_arm.group(1) if m_arm else base
        rows.append({"file": base, "arm": arm, "reqs": int(m.group(1)), "avg_us": float(m.group(2)), "max_ms": float(m.group(3)) / 1e3,
                     "watchdog": len(WD.findall(text))})
    return rows


def summarize(rows, thr):
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    out = []
    for arm, rs in by.items():
        stalled = [r for r in rs if r["max_ms"] > thr]
        out.append((arm, len(rs), len(stalled), [round(r["max_ms"], 1) for r in stalled],
                    statistics.median([r["max_ms"] for r in rs])))
    return out


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    for i, (arm, mx) in enumerate([("A", 3000.0), ("A", 103500.0), ("B", 2800.0), ("B", 3100.0)]):
        with open(os.path.join(d, "s-%s_r%d.log" % (arm, i)), "w") as fh:
            fh.write("noise\nmoe_hybrid: tokens 510 | host requests 20319, serve avg 226.6 max %.1f us, nmiss hist\n" % mx)
    res = {a: (n, k) for a, n, k, _, _ in summarize(run_rows(d), 20.0)}
    ok = res.get("A") == (2, 1) and res.get("B") == (2, 0)
    print("[%s] stall counting per arm: %s" % ("ok" if ok else "FAIL", res))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?")
    ap.add_argument("--threshold-ms", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.session_dir:
        ap.error("session_dir required")
    rows = run_rows(a.session_dir)
    print("%-10s %5s %9s  %s" % ("arm", "runs", "stalled", "stalled serve-max values (ms)  [threshold %.0f ms; median over all runs]" % a.threshold_ms))
    for arm, n, k, vals, med in summarize(rows, a.threshold_ms):
        print("%-10s %5d %5d/%-3d %s   (median %.1f ms)" % (arm, n, k, n, vals, med))
    return 0


if __name__ == "__main__":
    sys.exit(main())
