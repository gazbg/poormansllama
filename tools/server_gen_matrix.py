#!/usr/bin/env python3
r"""G32 LM-3: summarise 64-token greedy continuations (tools/server_gen.py gen.json files) of the same prompts from several servers, pair by pair.

  python tools/server_gen_matrix.py LABEL=gen.json [LABEL=gen.json ...] --pair REF,CAND [--pair REF,CAND ...] [--near-tie 0.25] [-v]

For every pair and prompt: identical over the common length, or the first differing position with the margin = the reference's logprob of its own token minus its logprob of the candidate's token (looked up in
the reference's top list; ">top-n" when it is not there). The summary line per pair: how many prompts are identical, how many differ by a near tie (< --near-tie), how many by more, the median / max margin over
the differing prompts and the median first-difference position. The point of the matrix is the CALIBRATION: pairs that involve no layer-major pass (e.g. stock -ub 512 -> stock -ub 2048) show how often two ordinary
paths miss the rule on the same prompts. stdlib only."""
import argparse
import json
import statistics
import sys


def first_diff(R, C):
    rs, cs = R["steps"], C["steps"]
    n = min(len(rs), len(cs))
    first = next((i for i in range(n) if rs[i]["token"] != cs[i]["token"]), None)
    if first is None:
        return n, None, None
    lp = dict(rs[first]["top"])
    own, other = rs[first]["logprob"], lp.get(cs[first]["token"])
    margin = float("inf") if (own is None or other is None) else abs(own - other)
    return n, first + 1, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--near-tie", type=float, default=0.25)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        label, _, path = r.partition("=")
        runs[label] = json.load(open(path))
    for pr in a.pair:
        ref, cand = pr.split(",")
        R, C = runs[ref], runs[cand]
        same = near = far = 0
        margins, poss = [], []
        lines = []
        for k in R:
            if k not in C or not R[k]["steps"] or not C[k]["steps"]:
                continue
            n, pos, m = first_diff(R[k], C[k])
            if pos is None:
                same += 1
                lines.append("  %-14s identical over %d tokens" % (k, n))
            else:
                margins.append(m)
                poss.append(pos)
                if m < a.near_tie:
                    near += 1
                else:
                    far += 1
                lines.append("  %-14s first difference at token %2d, margin %s%s" % (k, pos, "%.3f" % m if m != float("inf") else ">top-n", "" if m < a.near_tie else "   <-- above the rule"))
        tot = same + near + far
        mm = ("median %.3f / max %.3f" % (statistics.median(margins), max(margins))) if margins else "-"
        pp = ("%d" % statistics.median(poss)) if poss else "-"
        print("%-22s %2d prompts: identical %2d, near-tie flip %2d, ABOVE %2.2f nats %2d | margins %s | median first-difference token %s" % ("%s -> %s" % (ref, cand), tot, same, near, a.near_tie, far, mm, pp))
        if a.v:
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
