#!/usr/bin/env python3
"""Decompose one decode token of the hybrid engine from an `nsys export --type sqlite` file (host-only; times in the file are nanoseconds).

Per token the hybrid launches, in each of the n_layer layers, a `moe_hybrid_publish` kernel (hand the router result to the host engine) and later a
`moe_hybrid_wait_add` kernel (spin until the host engine's weighted sum arrives, then add it). On the MAIN stream (the stream the publish kernels run on):

  period   first publish of token t -> first publish of token t+1
  wait     sum over layers of the wait_add kernel duration: the GPU is spinning for the CPU engine (cache misses on the critical path)
  compute  union of all OTHER main-stream kernels inside the period: real GPU work (attention, DeltaNet, shared expert, resident experts, lm_head...)
  idle     period - union(all main-stream kernels): no kernel running at all (launch latency, host work, synchronisation)
  overlap  the part of wait that other main-stream kernels share (0 on one stream; reported to show the decomposition is exact)

The decomposition answers where a token's time goes: the CPU miss path (wait), real GPU work (compute) or launch/host latency (idle).

  python tools/nsys-token.py prof.sqlite [--layers 40] [--skip 4]
  python tools/nsys-token.py --selftest

It measures; it claims nothing.
"""
import argparse
import sqlite3
import statistics
import sys


def union_len(intervals):
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


def clip(intervals, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def load(con):
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    sid = "streamId" if "streamId" in cols else "0"
    rows = con.execute("SELECT start, end, COALESCE(demangledName, shortName), %s FROM CUPTI_ACTIVITY_KIND_KERNEL" % sid)
    return [(s, e, names.get(n, str(n)), st) for s, e, n, st in rows]


def analyse(kern, n_layer, skip):
    pub = sorted((s, e, st) for s, e, n, st in kern if "moe_hybrid_publish" in n)
    wai = sorted((s, e, st) for s, e, n, st in kern if "moe_hybrid_wait_add" in n)
    if len(pub) < 2 * n_layer or len(wai) < 2 * n_layer:
        return None, "too few hybrid kernels: publish %d, wait_add %d" % (len(pub), len(wai))
    main_stream = max(set(st for _, _, st in pub), key=[st for _, _, st in pub].count)
    main = [(s, e, n) for s, e, n, st in kern if st == main_stream]
    ntok = min(len(pub), len(wai)) // n_layer
    per = []
    for t in range(skip, ntok - 1):
        lo = pub[t * n_layer][0]
        hi = pub[(t + 1) * n_layer][0]
        if hi <= lo:
            continue
        period = hi - lo
        inside = [(max(s, lo), min(e, hi), n) for s, e, n in main if e > lo and s < hi]
        w_ints = [(s, e) for s, e, n in inside if "moe_hybrid_wait_add" in n]
        o_ints = [(s, e) for s, e, n in inside if "moe_hybrid_wait_add" not in n]
        wait = sum(e - s for s, e in w_ints)
        compute = union_len(o_ints)
        busy = union_len([(s, e) for s, e, _ in inside])
        per.append({"period": period, "wait": wait, "compute": compute, "idle": period - busy,
                    "overlap": union_len(w_ints) + compute - busy,
                    "n_kernels": len(inside)})
    if not per:
        return None, "no complete token found"
    return {"tokens": len(per), "per": per, "main_stream": main_stream}, ""


def summarize(res):
    per = res["per"]
    mean = lambda k: statistics.mean(p[k] for p in per) / 1e3
    out = {k: mean(k) for k in ("period", "wait", "compute", "idle", "overlap")}
    out["kernels"] = statistics.mean(p["n_kernels"] for p in per)
    ps = sorted(p["period"] for p in per)
    ws = sorted(p["wait"] for p in per)
    out["period_p50"], out["period_p90"] = ps[len(ps) // 2] / 1e3, ps[int(len(ps) * 0.9)] / 1e3
    out["wait_p50"], out["wait_p90"] = ws[len(ws) // 2] / 1e3, ws[int(len(ws) * 0.9)] / 1e3
    return out


def selftest():
    kern = []
    names = {}

    def k(s, e, name, st=7):
        kern.append((s, e, name, st))

    # 2 layers, 5 tokens, period 1000 ns: publish [0,10], compute [10,300], wait [300,500], compute [500,700], publish [700,710],
    # wait [710,800] ... idle [800,1000) except an aux-stream kernel [850,900) that must NOT count
    for t in range(5):
        b = t * 1000
        k(b + 0, b + 10, "moe_hybrid_publish_f32")
        k(b + 10, b + 300, "attn")
        k(b + 300, b + 500, "moe_hybrid_wait_add_f32")
        k(b + 500, b + 700, "ffn")
        k(b + 700, b + 710, "moe_hybrid_publish_f32")
        k(b + 710, b + 800, "moe_hybrid_wait_add_f32")
        k(b + 850, b + 900, "aux_dma", st=9)
    res, err = analyse(kern, 2, 0)
    fails = []

    def check(name, ok):
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            fails.append(name)

    check("tokens found (4 of 5 have a successor)", res is not None and res["tokens"] == 4)
    s = summarize(res)
    check("period = 1.0 us (%.3f)" % s["period"], abs(s["period"] - 1.0) < 1e-9)
    check("wait = 200 + 90 = 290 ns (%.3f us)" % s["wait"], abs(s["wait"] - 0.29) < 1e-9)
    check("compute = 290 + 200 + 10 + 10 = 510 ns (publish counts as compute) (%.3f us)" % s["compute"], abs(s["compute"] - 0.51) < 1e-9)
    check("idle = 1000 - 800 = 200 ns; the aux kernel is not main-stream (%.3f us)" % s["idle"], abs(s["idle"] - 0.2) < 1e-9)
    check("overlap = 0 on one stream (%.3f us)" % s["overlap"], abs(s["overlap"]) < 1e-9)
    check("union of overlapping intervals", union_len([(0, 10), (5, 20), (30, 40)]) == 30)
    print("selftest: %s (7 checks)" % ("PASS" if not fails else "FAIL"))
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
    res, err = analyse(load(sqlite3.connect(a.sqlite)), a.layers, a.skip)
    if res is None:
        print("ERROR:", err)
        return 2
    s = summarize(res)
    print("tokens analysed: %d  (main stream id %s, %.0f kernels per token)" % (res["tokens"], res["main_stream"], s["kernels"]))
    print("token period        %8.1f us   (p50 %.1f, p90 %.1f)" % (s["period"], s["period_p50"], s["period_p90"]))
    print("  GPU spin-wait CPU %8.1f us  %5.1f%%   (wait_add kernels; p50 %.1f, p90 %.1f)" % (
        s["wait"], 100 * s["wait"] / s["period"], s["wait_p50"], s["wait_p90"]))
    print("  GPU real work     %8.1f us  %5.1f%%   (union of every other main-stream kernel)" % (s["compute"], 100 * s["compute"] / s["period"]))
    print("  GPU idle          %8.1f us  %5.1f%%   (no kernel running: launch / host latency)" % (s["idle"], 100 * s["idle"] / s["period"]))
    print("  (overlap of wait with other kernels: %.1f us)" % s["overlap"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
