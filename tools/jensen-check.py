#!/usr/bin/env python3
"""Does evaluating the cost model at the MEAN hit rate overstate throughput?

`tools/placement-model.py` reports one throughput per policy, computed from the
mean of the two prompts' measured hit rates.  Under overlap the per-expert cost
is `max(h*d, (1-h)*c)`, which is convex in h with a minimum at h* -- so
throughput is CONCAVE in h, and by Jensen's inequality

    tok_s(mean(h))  >=  mean(tok_s(h))

i.e. averaging the hit rates first is an upward-biased estimate of what the
policy delivers on average.  The bias is zero only if both prompts land on the
same side of h* and the segment is linear there; it is largest when the two
prompts straddle h*, which is exactly what warm-then-pin does.

This prints both, per policy, so the bias can be read off.  Constants and
measured hit rates are identical to placement-model.py's.
"""

C_HOST_US = 35.7
D_DEV_US = 14.8
TOP_K = 8
N_LAYER = 40
ANCHOR = 40.96
OTHER_MS = 14.26  # solved from the anchor at the MEASURED h_stock = 0.188;
                  # see placement-model.py and tools/hstock-from-log.py

# (label, capacity rows/layer, h on prompt B "honest", h on prompt A)
# Every figure re-derived from the raw traces 2026-09-30; see audit note below.
POLICIES = [
    ("static pin (held out)", 48, 0.3340, 0.3884),
    ("warm-then-pin W=128",   48, 0.5087, 0.7270),
    ("LRU-K K=6 steady",      48, 0.6345, 0.7156),
    ("static pin (held out)", 64, 0.4117, 0.4839),
    ("warm-then-pin W=128",   64, 0.5979, 0.8143),
    ("LRU-K K=8 steady",      64, 0.7198, 0.7860),
    ("in-sample oracle",      64, 0.7572, 0.8727),
]


def overlapped_us(h):
    return max(h * TOP_K * D_DEV_US, (1.0 - h) * TOP_K * C_HOST_US)


def tok_s(h):
    return 1000.0 / (OTHER_MS + N_LAYER * overlapped_us(h) / 1000.0)


def main():
    h_star = C_HOST_US / (C_HOST_US + D_DEV_US)
    print("h* = %.4f, ceiling %.2f tok/s = %.2fx\n"
          % (h_star, tok_s(h_star), tok_s(h_star) / ANCHOR))
    print("%-24s %4s %15s %16s %16s %9s %14s"
          % ("policy", "rows", "h (B / A)", "at mean h", "mean of tok/s",
             "bias", "per-prompt"))
    for name, rows, hb, ha in POLICIES:
        at_mean = tok_s((hb + ha) / 2.0)
        mean_of = (tok_s(hb) + tok_s(ha)) / 2.0
        lo, hi = sorted((tok_s(hb), tok_s(ha)))
        straddle = " <- straddles h*" if (hb < h_star) != (ha < h_star) else ""
        print("%-24s %4d %15s %16s %16s %9s %14s%s"
              % (name, rows, "%.4f / %.4f" % (hb, ha),
                 "%.2f  %.2fx" % (at_mean, at_mean / ANCHOR),
                 "%.2f  %.2fx" % (mean_of, mean_of / ANCHOR),
                 "%+.2fx" % ((at_mean - mean_of) / ANCHOR),
                 "%.2f-%.2fx" % (lo / ANCHOR, hi / ANCHOR), straddle))

    print("\nThe 'at mean h' column is what placement-model.py reports.")
    print("The 'mean of tok/s' column is what the policy actually averages.")
    print("Worst case for a deployed policy is the 'per-prompt' low end, not")
    print("either mean: the placement is frozen per session, so a session gets")
    print("one prompt's hit rate, not the average of two.")


if __name__ == "__main__":
    main()
