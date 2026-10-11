#!/usr/bin/env python3
"""Where does the GPU's real work go inside one hybrid decode token? (host-only, reads an `nsys export --type sqlite` file)

Aggregates every main-stream kernel of the analysed tokens by name: calls per token, total time per token, mean per call, and the share of the
token's non-wait kernel time, so the fat kernels (and swarms of tiny ones) stand out. The wait_add kernels are reported separately (GPU spin on the host).

  python tools/nsys-kernels.py prof.sqlite [--layers 40] [--skip 4] [--top 25]

It measures; it claims nothing.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = [(s, e, names.get(n, str(n)), st) for s, e, n, st in
            con.execute("SELECT start, end, COALESCE(demangledName, shortName), streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    pub = [(s, e, st) for s, e, n, st in rows if "moe_hybrid_publish" in n]
    if len(pub) < 2 * a.layers:
        print("too few publish kernels:", len(pub))
        return 2
    stream = max(set(st for _, _, st in pub), key=[st for _, _, st in pub].count)
    ntok = len(pub) // a.layers
    t_lo = pub[a.skip * a.layers][0]
    t_hi = pub[(ntok - 1) * a.layers][0]
    n_tok = ntok - 1 - a.skip
    agg = defaultdict(lambda: [0, 0.0])
    for s, e, n, st in rows:
        if st != stream or s < t_lo or s >= t_hi:
            continue
        k = n.replace("void ", "")[:78]
        agg[k][0] += 1
        agg[k][1] += (e - s)
    wait = sum(v[1] for k, v in agg.items() if "moe_hybrid_wait_add" in k)
    real = sum(v[1] for k, v in agg.items() if "moe_hybrid_wait_add" not in k)
    print("tokens %d, main stream %s: wait_add %.0f us/token, other kernels %.0f us/token (%.0f kernels/token)" % (
        n_tok, stream, wait / n_tok / 1e3, real / n_tok / 1e3, sum(v[0] for v in agg.values()) / n_tok))
    print("%-80s %9s %10s %9s %6s" % ("kernel", "calls/tok", "us/token", "us/call", "share"))
    for k, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        if "moe_hybrid_wait_add" in k:
            continue
        print("%-80s %9.1f %10.1f %9.2f %5.1f%%" % (k, c / n_tok, t / n_tok / 1e3, t / c / 1e3, 100.0 * t / real))
    return 0


if __name__ == "__main__":
    sys.exit(main())
