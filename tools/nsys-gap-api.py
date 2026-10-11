#!/usr/bin/env python3
"""List the CUDA runtime API calls of one inter-token gap of the hybrid decode (host-only; reads an `nsys export --type sqlite` file).

The window starts at the last moe_hybrid_wait_add end of token t and ends at the first moe_hybrid_publish start of token t+1. Calls are printed with
their start offset from the window start and their duration, and summarised by API name (count, total time inside the window), so the host-side
composition of the gap (synchronous input uploads, graph launch, waits) is visible.

  python tools/nsys-gap-api.py prof.sqlite [--layers 40] [--token 20]

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
    ap.add_argument("--token", type=int, default=20)
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    k = con.execute("SELECT start, end, COALESCE(demangledName, shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    pub = [(s, e) for s, e, n in k if "moe_hybrid_publish" in names.get(n, "")]
    wai = [(s, e) for s, e, n in k if "moe_hybrid_wait_add" in names.get(n, "")]
    L = a.layers
    t = a.token
    if len(pub) < (t + 2) * L:
        print("not enough tokens:", len(pub) // L)
        return 2
    w_end = wai[t * L + L - 1][1]
    p_start = pub[(t + 1) * L][0]
    print("gap window %.1f us (token %d -> %d)" % ((p_start - w_end) / 1e3, t, t + 1))
    rt = con.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE end > ? AND start < ? ORDER BY start",
                     (w_end - 100000, p_start + 20000)).fetchall()
    summ = defaultdict(lambda: [0, 0.0])
    for s, e, nid in rt:
        nm = names.get(nid, str(nid)).replace("_v3020", "").replace("_v10000", "")
        if not a.quiet and s >= w_end - 100000:
            print("%9.1f %9.1f  %s" % ((s - w_end) / 1e3, (e - s) / 1e3, nm))
        if w_end <= s < p_start:
            summ[nm][0] += 1
            summ[nm][1] += (e - s) / 1e3
    print("\nruntime API calls that START inside the window:")
    for nm, (c, tt) in sorted(summ.items(), key=lambda kv: -kv[1][1]):
        print("  %-28s %3d calls  %8.1f us" % (nm, c, tt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
