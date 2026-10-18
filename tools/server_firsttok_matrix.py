#!/usr/bin/env python3
r"""G32 LM-3: first-token distances over many prompts, pair class by pair class (the calibration view).

  python tools/server_firsttok_matrix.py LABEL=firsttok.json [LABEL=firsttok.json ...] --pair A,B [--pair A,B ...]

For every pair (runs of A against runs of B of the same prompt, tools/server_firsttok.py output) it prints the distribution over prompts of the per-prompt MAX top-1 |dlogprob| and KL(top-10): median, 90th
percentile, max, and how many prompts exceed 0.10 / 0.25 nats. The point is the comparison BETWEEN pair classes: a pair that involves no layer-major pass (stock -ub 512 against stock -ub 2048) shows how far two
ordinary paths drift apart on the same prompts; the layer-major pairs are then read against it. stdlib only."""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_firsttok_cmp import stats  # noqa: E402


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    R = {}
    for r in a.runs:
        label, _, path = r.partition("=")
        R[label] = json.load(open(path))
    for pr in a.pair:
        A, B = pr.split(",")
        t1s, kls, same3 = [], [], []
        rows = []
        for k in R[A]:
            if k not in R[B]:
                continue
            ss = [stats(x["top"], y["top"]) for x in R[A][k] for y in R[B][k]]
            ss = [s for s in ss if s]
            if not ss:
                continue
            t1 = max((s[0] if s[0] is not None else 9.99) for s in ss)     # 9.99: the top-1 token of A is not in B's top-10 at all
            kl = max(s[2] for s in ss)
            t1s.append(t1)
            kls.append(kl)
            rows.append((k, t1, kl))
        if not t1s:
            print("%s -> %s: no common prompts" % (A, B))
            continue
        print("%-24s %2d prompts: top-1 |dlp| median %.4f p90 %.4f max %.4f | >0.10: %2d, >0.25: %2d | KL median %.5f max %.5f" % (
            "%s -> %s" % (A, B), len(t1s), statistics.median(t1s), pct(t1s, 0.9), max(t1s), sum(1 for x in t1s if x > 0.10), sum(1 for x in t1s if x > 0.25), statistics.median(kls), max(kls)))
        if a.v:
            for k, t1, kl in rows:
                print("    %-24s top-1 %.4f  KL %.5f" % (k, t1, kl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
