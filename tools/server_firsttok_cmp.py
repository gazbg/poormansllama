#!/usr/bin/env python3
"""Compare two firsttok.json files (tools/server_firsttok.py): for every prompt, over all pairs (run of A, run of B):
  top-1 dlp : |logprob difference| of A's top-1 token (the token that wins in both),
  mass dlp  : max |logprob difference| over the tokens that have probability > 2% (logprob > -3.9) in either run (the tail of a top-10 list moves a lot in log space and means nothing),
  KL        : KL(A || B) in nats between the two top-10 lists, each renormalised to its own top-10 mass (tokens missing from the other list get its smallest listed probability),
  same top-3: whether the top-3 tokens come in the same order.

  python tools/server_firsttok_cmp.py A.json B.json
"""
import json
import math
import sys


def prep(top):
    d = [(t, lp) for t, lp in top if lp is not None]
    return d


def stats(a, b):
    A, B = prep(a), prep(b)
    if not A or not B:
        return None
    Ad, Bd = dict(A), dict(B)
    top1 = A[0][0]
    t1 = abs(Ad[top1] - Bd[top1]) if top1 in Bd else None
    big = [t for t in Ad if Ad[t] > -3.9 or Bd.get(t, -99) > -3.9]
    mass = max([abs(Ad[t] - Bd[t]) for t in big if t in Bd and t in Ad] or [0.0])
    pa = {t: math.exp(lp) for t, lp in A}
    pb = {t: math.exp(lp) for t, lp in B}
    sa, sb = sum(pa.values()), sum(pb.values())
    floor = min(pb.values()) / sb
    kl = 0.0
    for t, p in pa.items():
        q = pb.get(t, None)
        q = (q / sb) if q is not None else floor
        kl += (p / sa) * math.log((p / sa) / q)
    same3 = [t for t, _ in A[:3]] == [t for t, _ in B[:3]]
    return t1, mass, kl, same3


def main():
    A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    for k in A:
        if k not in B:
            continue
        rows = []
        for x in A[k]:
            for y in B[k]:
                s = stats(x["top"], y["top"])
                if s:
                    rows.append(s)
        if not rows:
            continue
        t1 = [r[0] for r in rows if r[0] is not None]
        print("%-26s %d pairs: top-1 dlp max %.4f | mass dlp max %.4f | KL max %.5f | same top-3 order in %d of %d" % (
            k, len(rows), max(t1) if t1 else float("nan"), max(r[1] for r in rows), max(r[2] for r in rows), sum(1 for r in rows if r[3]), len(rows)))


if __name__ == "__main__":
    main()
