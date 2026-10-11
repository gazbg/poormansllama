#!/usr/bin/env python3
"""Attribute the inter-token GAP of the hybrid decode (host-only; reads an `nsys export --type sqlite` file).

The hybrid engine launches, in every one of the n_layer layers of a token, a `moe_hybrid_publish` and a `moe_hybrid_wait_add` kernel.
The GAP of a token boundary is the window from the LAST wait_add end of token t to the FIRST publish start of token t+1
(final norm, lm_head, logits copy, sampling on the host, next embedding, launch, layer-0 front). For every boundary this tool
reports the window, the busy time of the GPU (union of kernel and copy intervals inside it), the idle time (the host is on the
critical path), and the top kernels/copies by time. Times in the file are nanoseconds.

  python tools/nsys-gap.py prof.sqlite [--layers 40] [--skip 4]
  python tools/nsys-gap.py --selftest

It measures; it claims nothing.
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def union_len(intervals):
    """Total length of the union of [a, b) intervals."""
    tot, cur_a, cur_b = 0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot


def load(con):
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    k = [(s, e, names.get(n, str(n)), "kernel") for s, e, n in
         con.execute("SELECT start, end, COALESCE(demangledName, shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    m = []
    try:
        for s, e, kind, nb in con.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            m.append((s, e, "memcpy kind %s (%d B)" % (kind, nb), "memcpy"))
    except sqlite3.Error:
        pass
    return k, m


def analyse(con, n_layer, skip):
    kern, mem = load(con)
    pub = sorted((s, e) for s, e, n, _ in kern if "moe_hybrid_publish" in n)
    wai = sorted((s, e) for s, e, n, _ in kern if "moe_hybrid_wait_add" in n)
    if len(pub) < 2 * n_layer or len(wai) < 2 * n_layer:
        return None, "too few hybrid kernels: publish %d, wait_add %d" % (len(pub), len(wai))
    ntok = min(len(pub), len(wai)) // n_layer
    allk = sorted(kern + mem)
    rows = []
    agg = defaultdict(float)
    for t in range(skip, ntok - 1):
        w_end = wai[t * n_layer + n_layer - 1][1]        # last wait_add of token t
        p_start = pub[(t + 1) * n_layer][0]              # first publish of token t+1
        if p_start <= w_end:
            continue
        win = p_start - w_end
        inside, per = [], defaultdict(float)
        for s, e, n, kind in allk:
            if e <= w_end or s >= p_start:
                continue
            a, b = max(s, w_end), min(e, p_start)
            inside.append((a, b))
            per[n if kind == "memcpy" else n[:70]] += b - a
        busy = union_len(inside)
        rows.append((win, busy, win - busy))
        for n, v in per.items():
            agg[n] += v
    if not rows:
        return None, "no token boundary found"
    n = len(rows)
    out = {"boundaries": n,
           "window_us": sum(r[0] for r in rows) / n / 1e3,
           "busy_us": sum(r[1] for r in rows) / n / 1e3,
           "idle_us": sum(r[2] for r in rows) / n / 1e3,
           "top": sorted(((v / n / 1e3, k) for k, v in agg.items()), reverse=True)[:10]}
    return out, ""


def selftest():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER, shortName INTEGER)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER)")
    con.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "moe_hybrid_publish_f32"), (2, "moe_hybrid_wait_add_f32"), (3, "lm_head_gemv")])
    # 2 layers, 4 tokens; token period 1000 ns: layer0 publish [0,10] wait [10,30]; layer1 publish [40,50] wait [50,100]
    # gap: 100 -> next publish at 1000 : window 900 ns, lm_head kernel [200,500] and memcpy [500,600] => busy 400, idle 500
    for t in range(4):
        b = t * 1000
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (b + 0, b + 10, 1, 1))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (b + 10, b + 30, 2, 2))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (b + 40, b + 50, 1, 1))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (b + 50, b + 100, 2, 2))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)", (b + 200, b + 500, 3, 3))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)", (b + 500, b + 600, 2, 1048576))
    res, err = analyse(con, 2, 0)
    fails = []

    def check(name, ok):
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            fails.append(name)

    check("found boundaries (3 of 4 tokens have a successor): %s" % (res and res["boundaries"]), res is not None and res["boundaries"] == 3)
    if res:
        check("window = 900 ns = 0.9 us (%.3f)" % res["window_us"], abs(res["window_us"] - 0.9) < 1e-9)
        check("busy = 400 ns (kernel 300 + copy 100) (%.3f us)" % res["busy_us"], abs(res["busy_us"] - 0.4) < 1e-9)
        check("idle = 500 ns (%.3f us)" % res["idle_us"], abs(res["idle_us"] - 0.5) < 1e-9)
        check("top item is lm_head_gemv at 0.3 us", res["top"][0][1].startswith("lm_head_gemv") and abs(res["top"][0][0] - 0.3) < 1e-9)
    check("union of overlapping intervals", union_len([(0, 10), (5, 20), (30, 40)]) == 30)
    print("selftest: %s (%d checks)" % ("PASS" if not fails else "FAIL", 6))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="?")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--skip", type=int, default=4, help="warm-up tokens to skip")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.sqlite:
        ap.error("sqlite path required")
    con = sqlite3.connect(a.sqlite)
    res, err = analyse(con, a.layers, a.skip)
    if res is None:
        print("ERROR:", err)
        return 2
    print("token boundaries: %d" % res["boundaries"])
    print("gap window  %8.1f us   (last wait_add end -> next token's first publish start)" % res["window_us"])
    print("  GPU busy  %8.1f us   (union of kernels and copies inside the window)" % res["busy_us"])
    print("  GPU idle  %8.1f us   (the host / launch latency is on the critical path)" % res["idle_us"])
    print("top items by mean time inside the window (us):")
    for v, k in res["top"]:
        print("  %8.1f  %s" % (v, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
