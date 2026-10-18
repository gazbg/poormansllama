#!/usr/bin/env python3
"""Anatomy of an MTP round from an Nsight Systems export (host-only; reads an `nsys export --type sqlite` file of a --mtp run, see tools/run-nsys-arm.py).

  python tools/g30-nsys-rounds.py prof.sqlite [--layers 40] [--skip 8] [--top 14]

A verify round is the group of `moe_hybrid_publish` kernels (one per layer, `--layers` of them) of the target decode on the main stream. Everything the GPU does between the last kernel of a verify
round and the first publish kernel of the next is the DRAFT window (the draft head's decode, the logits fetch, the host sampling, the launch of the next verify). Per round (means over the steady-state
rounds after --skip): the wall of the verify window and of the draft window, the GPU-busy time and the number of kernels in each, the idle share, and the fattest kernels of each window.
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
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()
    con = sqlite3.connect(a.sqlite)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = [(s, e, names.get(n, str(n)).replace("void ", ""), st) for s, e, n, st in
            con.execute("SELECT start, end, COALESCE(demangledName, shortName), streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    pub = [(s, e, st) for s, e, n, st in rows if "moe_hybrid_publish" in n]
    if len(pub) < 4 * a.layers:
        print("too few publish kernels:", len(pub))
        return 2
    stream = max(set(st for _, _, st in pub), key=[st for _, _, st in pub].count)
    main_rows = [(s, e, n) for s, e, n, st in rows if st == stream]
    n_rounds = len(pub) // a.layers
    starts = [pub[r * a.layers][0] for r in range(n_rounds)]
    ends = [pub[(r + 1) * a.layers - 1][1] for r in range(n_rounds)]
    # a verify window runs from its first publish to the first kernel after the round's last publish that starts a new group: approximate by the last publish end + the tail until the next draft kernel;
    # the boundary between verify tail and draft is the logits copy: use the gap structure: the verify window = [first publish, last publish end + tail], tail = kernels within 150 us of the last publish end
    def window(lo, hi):
        ks = [(s, e, n) for s, e, n in main_rows if s >= lo and s < hi]
        busy = sum(e - s for s, e, n in ks)
        return ks, busy

    agg_v = defaultdict(lambda: [0, 0.0])
    agg_d = defaultdict(lambda: [0, 0.0])
    tot = defaultdict(float)
    cnt = 0
    for r in range(a.skip, n_rounds - 1):
        v_lo, v_hi = starts[r], ends[r]
        # tail of the verify graph after its last publish (the last layer's experts, the output norm and the lm head), up to the first kernel followed by a gap of > 120 us
        tail_end = v_hi
        for s, e, n in main_rows:
            if s < v_hi:
                continue
            if s - tail_end > 120_000:
                break
            tail_end = max(tail_end, e)
        d_lo, d_hi = tail_end, starts[r + 1]
        vk, vbusy = window(v_lo, tail_end)
        dk, dbusy = window(d_lo, d_hi)
        tot["verify_wall"] += tail_end - v_lo
        tot["verify_busy"] += vbusy
        tot["verify_kernels"] += len(vk)
        tot["draft_wall"] += d_hi - d_lo
        tot["draft_busy"] += dbusy
        tot["draft_kernels"] += len(dk)
        for s, e, n in vk:
            k = n[:76]
            agg_v[k][0] += 1
            agg_v[k][1] += e - s
        for s, e, n in dk:
            k = n[:76]
            agg_d[k][0] += 1
            agg_d[k][1] += e - s
        cnt += 1
    if cnt == 0:
        print("no rounds to analyse")
        return 2
    us = lambda x: x / cnt / 1e3
    print("rounds analysed %d (of %d), main stream %s" % (cnt, n_rounds, stream))
    print("verify window: wall %.0f us, GPU busy %.0f us (%.0f%%), %.0f kernels" % (us(tot["verify_wall"]), us(tot["verify_busy"]), 100.0 * tot["verify_busy"] / tot["verify_wall"], tot["verify_kernels"] / cnt))
    print("draft window : wall %.0f us, GPU busy %.0f us (%.0f%%), %.0f kernels" % (us(tot["draft_wall"]), us(tot["draft_busy"]), 100.0 * tot["draft_busy"] / max(1.0, tot["draft_wall"]), tot["draft_kernels"] / cnt))
    for title, agg in (("verify", agg_v), ("draft", agg_d)):
        print("-- fattest kernels of the %s window (us per round, calls per round, us per call)" % title)
        for k, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
            if "moe_hybrid_wait_add" in k:
                print("   %-76s %9.1f %6.1f %8.1f   (GPU spin on the host leg)" % (k, us(t), c / cnt, t / c / 1e3))
            else:
                print("   %-76s %9.1f %6.1f %8.1f" % (k, us(t), c / cnt, t / c / 1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
