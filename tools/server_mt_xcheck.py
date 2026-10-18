#!/usr/bin/env python3
"""Compare the greedy continuations recorded by the `xcheck` scenario of two tools/server_mt.py reports (G31 SRV-5 / SRV-7).

  python tools/server_mt_xcheck.py --ref build/llama-cuda/<stock>/report.json --cand build/llama-cuda/<opt>/report.json [--near-tie 0.25]

For every prompt: identical, or the first differing token of CAND must be one of REF's top-2 with a logprob margin < near-tie (REF's own top-2 at that position). Anything else is a hard failure.
Exit 1 on a hard failure or a missing xcheck blob.
"""
import argparse
import json
import sys


def compare(ref, cand, tie):
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i][0] != cand[i][0]:
            lp_ref = ref[i][1]
            lp_c = None
            for tok, lp in ref[i][2]:
                if tok == cand[i][0]:
                    lp_c = lp
            if lp_c is None or lp_ref is None:
                return "DIFF token %d: %r vs %r, candidate not in the reference top-2" % (i, ref[i][0], cand[i][0]), False
            m = lp_ref - lp_c
            return "near-tie flip at token %d (margin %.3f nats)" % (i, m), m < tie
    if len(ref) != len(cand):
        return "same prefix, lengths %d vs %d" % (len(ref), len(cand)), True
    return "identical over %d tokens" % len(ref), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--near-tie", type=float, default=0.25)
    a = ap.parse_args()
    r = json.load(open(a.ref, "r", encoding="utf-8")).get("blobs", {}).get("xcheck")
    c = json.load(open(a.cand, "r", encoding="utf-8")).get("blobs", {}).get("xcheck")
    if not r or not c:
        print("missing xcheck blob in one of the reports")
        return 1
    bad = 0
    cm = {x["k"]: x for x in c}
    for x in r:
        y = cm.get(x["k"])
        if y is None:
            print("turn %d: missing in the candidate" % x["k"])
            bad += 1
            continue
        msg, ok = compare([tuple(t) for t in x["tokens"]], [tuple(t) for t in y["tokens"]], a.near_tie)
        print("turn %2d: %s %s" % (x["k"], "ok  " if ok else "FAIL", msg))
        bad += 0 if ok else 1
    print("xcheck: %s" % ("all ok" if not bad else "%d failure(s)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
