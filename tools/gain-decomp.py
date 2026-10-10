#!/usr/bin/env python3
"""Decompose the 56-row hit-rate gain on prompt B into its three sources.

Re-derived 2026-09-30 from build/llama-cuda/gplr19-honest-k6-ab/arm-B1.jsonl
(scored), ranked against gplr19-k6-ab/arm-B1.jsonl for the held-out leg.  The
placement draft's paragraph quoted 0.337 / 0.565 / 0.647 for these three; the
first two do not reproduce at 56 rows.
"""

C_HOST_US, D_DEV_US, TOP_K, N_LAYER = 35.7, 14.8, 8, 40
OTHER_MS, ANCHOR = 14.26, 40.96

STAGES = [
    ("stock whole-bank",                 0.1879),
    ("+ granularity (static, held out)", 0.3753),
    ("+ per-session warm-then-pin",      0.5559),
    ("+ per-token cache (LRU-K K=6)",    0.6461),
]


def ov(h):
    return 1000.0 / (OTHER_MS + N_LAYER * max(h * TOP_K * D_DEV_US,
                                              (1 - h) * TOP_K * C_HOST_US) / 1000.0)


def ser(h):
    return 1000.0 / (OTHER_MS + N_LAYER * TOP_K
                     * (h * D_DEV_US + (1 - h) * C_HOST_US) / 1000.0)


def main():
    h_star = C_HOST_US / (C_HOST_US + D_DEV_US)
    total = STAGES[-1][1] - STAGES[0][1]
    print("56 rows/layer = 3,780 MiB, prompt B (honest).  h* = %.4f\n" % h_star)
    print("%-34s %8s %10s %14s %14s" % ("stage", "h", "d(h)", "serial", "overlapped"))
    prev = None
    for name, h in STAGES:
        step = "" if prev is None else "%+.4f (%.0f%%)" % (h - prev, 100 * (h - prev) / total)
        print("%-34s %8.4f %10s %14s %14s"
              % (name, h, step, "%.2f %.2fx" % (ser(h), ser(h) / ANCHOR),
                 "%.2f %.2fx" % (ov(h), ov(h) / ANCHOR)))
        prev = h
    print("\ntotal h gain %.4f; every stage stays BELOW h* = %.4f on this prompt,"
          % (total, h_star))
    print("so on prompt B more hit rate is still strictly better -- the")
    print("overshoot regime that reorders policies at cap 64 never engages here.")


if __name__ == "__main__":
    main()
