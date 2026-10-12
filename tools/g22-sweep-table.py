#!/usr/bin/env python3
"""G22: the short-prompt prefill sweep (sessions g46-<N>): stock llama-completion against the layer-major prefill, per prompt length.

    python tools/g22-sweep-table.py [N ...]       default N = 3 8 16 32 64 128 256 512 1024 2048

Reads build/llama-cuda/g46-<N>/*.log (stock: `prompt eval time`; layer-major: `PP: N tokens in X ms`) and the layer-major trace (phase_summary of the prefill: summed copy,
wait and compute time over the 40 layers). Prints a table and a JSON line per N for the schedule model (tools/g22-model.py). Host-only.
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda")


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def stock_prefill(text):
    m = None
    for m in re.finditer(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text):
        pass
    return (float(m.group(1)), int(m.group(2))) if m else None


def lm_prefill(text):
    m = None
    for m in re.finditer(r"PP: (\d+) tokens in ([\d.]+) ms", text):
        pass
    return (float(m.group(2)), int(m.group(1))) if m else None


def lm_trace(path):
    out = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if '"phase_summary"' in ln and '"phase": "pp"' in ln:
            d = json.loads(ln)
            out = {k: d[k] for k in ("wall_us", "total_transfer_us", "total_wait_us", "total_compute_us") if k in d}
    return out


def main(argv):
    ns = [int(x) for x in argv] or [3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    print("%6s %7s | %10s %9s | %10s %9s %9s %9s | %8s" % ("N", "tokens", "stock ms", "tok/s", "LM2 ms", "tok/s", "wait ms", "compute", "LM/stock"))
    rows = []
    for n in ns:
        d = os.path.join(ROOT, "g46-%d" % n)
        st, lm, tr = [], [], []
        for p in sorted(glob.glob(os.path.join(d, "g46-%d-STK_r*.log" % n))):
            r = stock_prefill(read(p))
            if r:
                st.append(r)
        for p in sorted(glob.glob(os.path.join(d, "g46-%d-LM2_r*.log" % n))):
            r = lm_prefill(read(p))
            if r:
                lm.append(r)
            t = lm_trace(p.replace(".log", ".jsonl"))
            if t:
                tr.append(t)
        if not st or not lm:
            print("%6d  (no data yet)" % n)
            continue
        tokens = lm[0][1]
        st_ms = statistics.median(x[0] for x in st)
        lm_ms = statistics.median(x[0] for x in lm)
        wait = statistics.median(t["total_wait_us"] for t in tr) / 1000.0 if tr else float("nan")
        comp = statistics.median(t["total_compute_us"] for t in tr) / 1000.0 if tr else float("nan")
        print("%6d %7d | %10.1f %9.0f | %10.1f %9.0f %9.1f %9.1f | %8.2f" % (n, tokens, st_ms, tokens * 1000.0 / st_ms, lm_ms, tokens * 1000.0 / lm_ms, wait, comp, lm_ms / st_ms))
        rows.append({"n": n, "tokens": tokens, "stock_ms": st_ms, "lm_ms": lm_ms, "lm_wait_ms": wait, "lm_compute_ms": comp})
    print()
    for r in rows:
        print("ROW " + json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
