#!/usr/bin/env python3
"""Compare two tools/server_mt.py reports (STOCK vs OPT) against the G31 contract (docs/plans/llama-server-integration.md).

  python tools/server_mt_compare.py --stock build/llama-cuda/g230-stock/report.json --opt build/llama-cuda/g231-opt/report.json [--min-free-mib 16000] [--opt-log SERVER.LOG]

SRV-2: TG ratio per request with >= 16 generated tokens, banded by the request's context (prompt_n + cache_n): <= 6k, 6k-30k, 30k-100k, > 100k.
SRV-4 (relative): OPT prompt_n <= STOCK prompt_n + 8 for every request of the cache scenarios.
SRV-3: min free VRAM (nvidia-smi view, +1,086 MiB above the CUDA-runtime view) and the in-process watcher minimum from the server log.
Exit 1 when an SRV-4 relative check or a hard failure (http error, dead server) is found; the ratios are reported and judged by the contract text, not here.
"""
import argparse
import json
import re
import statistics
import sys

BANDS = [(0, 6000, "<=6k"), (6000, 30000, "6k-30k"), (30000, 100000, "30k-100k"), (100000, 10**9, ">100k")]
CACHE_SCEN = {"prefix", "agentic", "branch", "alternate", "slots", "growth"}


def load(p):
    return json.load(open(p, "r", encoding="utf-8"))


def rows_by_key(d):
    out = {}
    for r in d["rows"]:
        if r.get("check"):
            continue
        out[(r["scenario"], r["name"])] = r
    return out


def watcher_min(log):
    if not log:
        return None
    lo = None
    with open(log, "rb") as f:
        txt = f.read().replace(b"\x00", b"").decode("utf-8", "replace")
    for m in re.finditer(r"free-watch: (?:new minimum of free device memory|the engine is gone; lowest free device memory) (\d+) MiB", txt):
        v = int(m.group(1))
        lo = v if lo is None else min(lo, v)
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--min-free-mib", type=int, default=16000)
    ap.add_argument("--opt-log")
    a = ap.parse_args()
    s, o = load(a.stock), load(a.opt)
    sr, orr = rows_by_key(s), rows_by_key(o)
    fails = []
    print("%-9s %-26s %8s %8s %6s | %8s %8s | %7s %7s %6s" % ("scenario", "name", "S pr_n", "O pr_n", "dlt", "S cache", "O cache", "S tps", "O tps", "ratio"))
    ratios = {b[2]: [] for b in BANDS}
    for k, ro in orr.items():
        rs = sr.get(k)
        if rs is None:
            print("%-9s %-26s (no stock row)" % k)
            continue
        po, ps = ro.get("prompt_n"), rs.get("prompt_n")
        ratio = None
        if ro.get("tps") and rs.get("tps") and (ro.get("predicted_n") or 0) >= 16 and (rs.get("predicted_n") or 0) >= 16:
            ratio = ro["tps"] / rs["tps"]
            ctx = (po or 0) + (ro.get("cache_n") or 0)
            for lo, hi, name in BANDS:
                if lo <= ctx < hi:
                    ratios[name].append(ratio)
        flag = ""
        if k[0] in CACHE_SCEN and po is not None and ps is not None and po > ps + 8:
            flag = "  <-- SRV-4 relative FAIL"
            fails.append("%s/%s: OPT processed %d prompt tokens, STOCK %d" % (k[0], k[1], po, ps))
        print("%-9s %-26s %8s %8s %6s | %8s %8s | %7.2f %7.2f %6s%s" % (k[0], k[1], ps, po, (po - ps) if (po is not None and ps is not None) else "", rs.get("cache_n"), ro.get("cache_n"),
              rs.get("tps") or 0, ro.get("tps") or 0, ("%.2fx" % ratio) if ratio else "", flag))
    print()
    print("SRV-2 TG ratio OPT/STOCK per context band (requests with >= 16 generated tokens):")
    for lo, hi, name in BANDS:
        v = ratios[name]
        if v:
            print("  %-9s n=%2d  min %.2fx  median %.2fx  max %.2fx" % (name, len(v), min(v), statistics.median(v), max(v)))
        else:
            print("  %-9s no samples" % name)
    for name, rep in (("STOCK", s), ("OPT", o)):
        print("%s: server alive at the end: %s; nvidia-smi free min %s MiB; failed expectations: %d" % (name, rep.get("alive"), rep.get("free_min_smi_mib"), len(rep.get("fails", []))))
        for f in rep.get("fails", []):
            print("    ", f)
    w = watcher_min(a.opt_log)
    if w is not None:
        print("OPT in-process watcher (CUDA runtime view): lowest free %d MiB (target %d): %s" % (w, a.min_free_mib, "OK" if w >= a.min_free_mib else "BELOW"))
        if w < a.min_free_mib:
            fails.append("SRV-3: watcher minimum %d MiB < %d" % (w, a.min_free_mib))
    if not o.get("alive", True):
        fails.append("OPT server died")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("\nno SRV-4 relative failure, server alive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
