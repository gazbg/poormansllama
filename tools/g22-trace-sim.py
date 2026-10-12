#!/usr/bin/env python3
"""G22: read a layer-major PP trace (one chunk per layer, i.e. a prompt of at most one ubatch) and replay it with a different number of layer slots.

    python tools/g22-trace-sim.py <trace.jsonl> [--c MS] [--slots 2,3,4,6,8]

From the trace: the measured PP wall, the per-layer compute (compute_us + setup_us of the layer_compute event), the copy observations. The replay is the controller's own protocol
(tools: llama-layer-streaming.cpp layer_major_pp): the first S layers are submitted at t = 0, in order; layer i starts at max(end of layer i-1, ready of layer i) and ends after its measured
compute; the successor of layer i (layer i + S, the same slot) is submitted when layer i ends; ONE copy engine serves the submissions in order and takes c ms per bank. `--c` fixes c;
without it c is fitted so that the replay of the traced S reproduces the measured sweep wall. The replay of S = the traced slot count against the measured wall is the check of the method;
the other rows are what the same compute sequence would do with more slots. Host-only.
"""
import argparse
import json
import statistics
import sys


def load(path):
    ev = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        try:
            ev.append(json.loads(ln))
        except ValueError:
            pass
    return ev


def simulate(g, c, slots, t_start=0.0):
    n = len(g)
    ready = [None] * n
    copy_free = t_start
    # preload: the first `slots` layers, in order
    for i in range(min(slots, n)):
        ready[i] = copy_free + c
        copy_free = ready[i]
    t = t_start
    end = [0.0] * n
    for i in range(n):
        start = max(t, ready[i])
        end[i] = start + g[i]
        t = end[i]
        j = i + slots
        if j < n:
            sub = end[i]
            begin = max(copy_free, sub)
            ready[j] = begin + c
            copy_free = ready[j]
    return t - t_start


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--slots", default="2,3,4,6,8")
    a = ap.parse_args(argv)
    ev = load(a.trace)
    rs = next(e for e in ev if e.get("event") == "run_start")
    n_layer = rs["n_layer"]
    s_traced = rs["n_slots"]
    ps = next(e for e in ev if e.get("event") == "phase_summary" and e.get("phase") == "pp")
    g = [0.0] * n_layer
    seen = [0] * n_layer
    for e in ev:
        if e.get("event") == "layer_compute" and e.get("phase") == 0:
            il = e["layer"]
            g[il] += (e["compute_us"] + e["setup_us"]) / 1000.0
            seen[il] += 1
    if min(seen) == 0:
        print("trace has no layer_compute for some layers; need one chunk per layer")
        return 2
    wall = ps["wall_us"] / 1000.0
    # the sweep wall = the measured wall minus the non-sweep parts we cannot see (embed + output graphs): use the layer_compute span instead
    first = min(e["t_us"] for e in ev if e.get("event") == "copy_submit" and e.get("phase") == 0)
    last = max(e["t_us"] for e in ev if e.get("event") == "layer_compute" and e.get("phase") == 0)
    sweep = (last - first) / 1000.0
    rec = [g[i] for i in range(n_layer) if i % 4 != 3]
    att = [g[i] for i in range(n_layer) if i % 4 == 3]
    print("trace %s: pp wall %.1f ms, sweep (first submit .. last layer end) %.1f ms, traced slots %d" % (a.trace, wall, sweep, s_traced))
    print("per-layer compute (ms): total %.1f | recurrent layers (30) mean %.1f median %.1f max %.1f | attention layers (10) mean %.1f median %.1f max %.1f | layer 0 %.1f" % (
        sum(g), statistics.mean(rec), statistics.median(rec), max(rec), statistics.mean(att), statistics.median(att), max(att), g[0]))
    if a.c is None:
        lo, hi = 5.0, 80.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if simulate(g, mid, s_traced) < sweep:
                lo = mid
            else:
                hi = mid
        c = (lo + hi) / 2
        print("fitted copy time c = %.2f ms per bank (replay of S=%d reproduces the sweep wall)" % (c, s_traced))
    else:
        c = a.c
        print("copy time c = %.2f ms per bank (given)" % c)
    print("replay with the SAME per-layer compute:")
    for s in [int(x) for x in a.slots.split(",")]:
        t = simulate(g, c, s)
        print("  S=%d: sweep %.1f ms%s" % (s, t, "   (traced)" if s == s_traced else "   (%+.1f%% against the traced sweep)" % (100.0 * (t - sweep) / sweep)))
    print("lower bound with unlimited slots: copy-bound %.1f ms (40 x c) or compute-bound %.1f ms (sum of compute): %.1f ms" % (n_layer * c, sum(g), max(n_layer * c + g[-1], sum(g))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
