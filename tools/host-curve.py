#!/usr/bin/env python3
"""Replace the constant host cost `c` with the MEASURED per-expert curve.

The cost model in tools/placement-model.py charges the host leg as

    (1-h) * top_k * c        with c = 35.7 us, a constant

but sec 12.1a measured that the per-expert host cost is a *function of how many
experts the host runs*, and that count is itself (1-h)*top_k.  Charging a
constant is therefore an approximation the document already flagged:

    experts   median us   us/expert
        1        33.10      33.10     1.77 MB, sits in L3
        2        66.95      33.48
        4       144.40      36.10     14.2 MB, does not
        8       288.65      36.08

Section 12.1a checked what this does to the overlapped CEILING and correctly
found it negligible -- above h* the device leg is binding, so the host's exact
cost is off the critical path.  But it did not check the other channel.  `other`
is solved by making the model reproduce the 40.96 tok/s anchor, and the anchor
sits at h_stock = 0.188, where the host runs 6.5 experts per layer -- squarely
in the region where the curve charges MORE than the constant does.  `other` is
common to every policy, so that channel reaches every number in sec 13.4.

This script swaps the constant for piecewise-linear interpolation of the
measured totals and re-solves everything.

The four points are all from the SAME sweep (audit-C) so the curve is
internally consistent.  The separate threads-k8 invocation put 8 experts at
285.6 us against audit-C's 288.65 -- 1.0% apart -- and that spread is the
honest uncertainty on any figure below.  c = 35.7 in the other scripts is
285.6/8, i.e. it comes from threads-k8, not from this curve.
"""

N_LAYER, TOP_K = 40, 8
D_DEV_US = 14.8
C_CONST_US = 35.7
ANCHOR_TOK_S = 40.96
ANCHOR_MS = 1000.0 / ANCHOR_TOK_S
H_STOCK = (5192.44 - 1921.0) / (N_LAYER * (432.0 + 3.1875))

# sec 12.1a, audit-C, 8 threads, --pool 256 --id-stride 8
CURVE = [(1, 33.10), (2, 66.95), (4, 144.40), (8, 288.65)]

MEASURED = [
    (48, "static build-time pin (held out)", 0.3340, 0.3884),
    (48, "warm-then-pin, W=128, frozen",     0.5087, 0.7270),
    (48, "LRU-K K=6 (steady state)",         0.6345, 0.7156),
    (64, "static build-time pin (held out)", 0.4117, 0.4839),
    (64, "warm-then-pin, W=128, frozen",     0.5979, 0.8143),
    (64, "LRU-K K=8 (steady state)",         0.7198, 0.7860),
    (64, "in-sample oracle (ceiling)",       0.7572, 0.8727),
]


def host_us(n):
    """Total host us for n experts in one layer, n fractional."""
    if n <= 0:
        return 0.0
    if n <= CURVE[0][0]:
        # below one expert, charge the 1-expert per-expert rate
        return n * CURVE[0][1] / CURVE[0][0]
    for (n0, t0), (n1, t1) in zip(CURVE, CURVE[1:]):
        if n <= n1:
            return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
    # above 8, extrapolate on the last segment's slope
    (n0, t0), (n1, t1) = CURVE[-2], CURVE[-1]
    return t1 + (t1 - t0) / (n1 - n0) * (n - n1)


def layer_us(h, overlapped, curve):
    dev = h * TOP_K * D_DEV_US
    host = host_us((1.0 - h) * TOP_K) if curve else (1.0 - h) * TOP_K * C_CONST_US
    return max(dev, host) if overlapped else dev + host


def tok_s(h, overlapped, other_ms, curve):
    return 1000.0 / (other_ms + N_LAYER * layer_us(h, overlapped, curve) / 1000.0)


def solve_other(curve):
    return ANCHOR_MS - N_LAYER * layer_us(H_STOCK, False, curve) / 1000.0


def h_star(curve):
    """Where the device and host legs cross, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid * TOP_K * D_DEV_US < (host_us((1.0 - mid) * TOP_K) if curve
                                     else (1.0 - mid) * TOP_K * C_CONST_US):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    print("measured host curve (sec 12.1a, audit-C, 8 threads):")
    for n, t in CURVE:
        print("   %d expert(s)  %7.2f us total  %6.2f us/expert" % (n, t, t / n))
    print("   the constant the model uses, c = 35.7 us, is 285.6/8 from the")
    print("   separate threads-k8 run; audit-C's own 8-expert cell is 36.08.\n")

    n_stock = (1.0 - H_STOCK) * TOP_K
    print("at the STOCK anchor h = %.3f the host runs %.2f experts per layer:"
          % (H_STOCK, n_stock))
    print("   constant c charges %7.2f us/layer" % ((1 - H_STOCK) * TOP_K * C_CONST_US))
    print("   measured curve     %7.2f us/layer   (%+.2f us)"
          % (host_us(n_stock), host_us(n_stock) - (1 - H_STOCK) * TOP_K * C_CONST_US))
    print("   -> this is the channel sec 12.1a did not check: it reaches `other`,")
    print("      and `other` is common to every policy.\n")

    for curve in (False, True):
        o = solve_other(curve)
        hs = h_star(curve)
        print("%-16s other = %.3f ms   h* = %.4f   floor %.1f us/layer"
              "   ceiling %.2f tok/s = %.3fx"
              % ("MEASURED curve" if curve else "constant c", o, hs,
                 layer_us(hs, True, curve),
                 tok_s(hs, True, o, curve),
                 tok_s(hs, True, o, curve) / ANCHOR_TOK_S))

    o_c, o_m = solve_other(False), solve_other(True)
    print("\n%-6s %-34s %15s %14s %14s %8s"
          % ("rows", "policy", "h (B / A)", "const c", "curve", "delta"))
    flips = []
    for rows, name, hb, ha in MEASURED:
        vc = (tok_s(hb, True, o_c, False) + tok_s(ha, True, o_c, False)) / 2.0
        vm = (tok_s(hb, True, o_m, True) + tok_s(ha, True, o_m, True)) / 2.0
        flips.append((name + " @%d" % rows, vc, vm))
        print("%-6d %-34s %15s %14s %14s %+7.3fx"
              % (rows, name, "%.3f / %.3f" % (hb, ha),
                 "%.2f  %.3fx" % (vc, vc / ANCHOR_TOK_S),
                 "%.2f  %.3fx" % (vm, vm / ANCHOR_TOK_S),
                 (vm - vc) / ANCHOR_TOK_S))

    rank_c = [n for n, _, _ in sorted(flips, key=lambda r: -r[1])]
    rank_m = [n for n, _, _ in sorted(flips, key=lambda r: -r[2])]
    print("\nranking under constant c : %s" % " > ".join(rank_c))
    print("ranking under curve      : %s" % " > ".join(rank_m))
    print("RANKING %s" % ("UNCHANGED" if rank_c == rank_m else "*** CHANGED ***"))


if __name__ == "__main__":
    main()
