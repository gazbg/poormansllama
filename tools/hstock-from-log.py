#!/usr/bin/env python3
"""Derive h_stock from the anchor run's own fitter log instead of guessing it.

docs/tg-expert-path-reconciliation.md solved `other` from h_stock = 9/40 =
0.225, inferred as "CUDA0 model buffer 5192.44 MiB less ~1302 MiB of dense
weights, / 432 MiB per bank".  The 1302 was a guess.  The log does not require
it: `-lv 4` prints a memory breakdown for the probe configuration with ALL MoE
tensors moved to system memory, whose model column IS the dense-only size.

All inputs below are quoted from "docs/llamacpp 8gb vram base.md", build 10892,
--fit-target 16000, the same run that produced the 40.96 tok/s anchor.
"""

# line 39: everything on device      -> CUDA0 (21217 = 19399 + 1390 + 427)
MODEL_ALL_ON_DEV = 19399.0
# line 46: all MoE moved to host     -> CUDA0 ( 3879 =  1921 + 1390 + 567)
MODEL_DENSE_ONLY = 1921.0
# line 270: the configuration actually served
MODEL_CHOSEN = 5192.44

N_LAYER = 40                       # trunk layers with loaded experts
# line 255-257: blk.40.ffn_{down,gate,up}_exps.weight, 150994944 B each
TENSOR_MIB = 150994944 / (1024 * 1024)
BANK_MIB = 3 * TENSOR_MIB
# line 259-261: ffn_{gate,up,down}_shexp.weight, 1114112 B each
SHEXP_MIB = 3 * 1114112 / (1024 * 1024)

ANCHOR, C_HOST_US, D_DEV_US, TOP_K = 40.96, 35.7, 14.8, 8
ANCHOR_MS = 1000.0 / ANCHOR


def ceiling(h_stock):
    other = ANCHOR_MS - N_LAYER * TOP_K * (
        h_stock * D_DEV_US + (1 - h_stock) * C_HOST_US) / 1000.0
    floor_us = TOP_K * C_HOST_US * D_DEV_US / (C_HOST_US + D_DEV_US)
    return other, 1000.0 / (other + N_LAYER * floor_us / 1000.0) / ANCHOR


def main():
    print("bank = 3 x %.2f MiB = %.2f MiB; shared-expert trio = %.4f MiB\n"
          % (TENSOR_MIB, BANK_MIB, SHEXP_MIB))

    # Cross-check that MODEL_DENSE_ONLY is a real dense-only figure: the
    # difference against the all-on-device model column must be the whole
    # expert mass, which we know independently from the tensor sizes.
    expert_mass = MODEL_ALL_ON_DEV - MODEL_DENSE_ONLY
    predicted = N_LAYER * (BANK_MIB + SHEXP_MIB)
    print("CROSS-CHECK of the dense-only figure")
    print("  19399 - 1921            = %8.2f MiB of expert mass (from the log)"
          % expert_mass)
    print("  40 x (bank + shexp)     = %8.2f MiB              (from tensor sizes)"
          % predicted)
    print("  agreement               = %8.2f%%  (%.0f MiB apart: gate_inp, norms,"
          % (100.0 * expert_mass / predicted, expert_mass - predicted))
    print("                                       MTP dense, MiB rounding)\n")

    dev_expert = MODEL_CHOSEN - MODEL_DENSE_ONLY
    print("EXPERT WEIGHTS RESIDENT AT STOCK = %.2f - %.0f = %.2f MiB"
          % (MODEL_CHOSEN, MODEL_DENSE_ONLY, dev_expert))

    routes = [
        ("bank-only  (/ %.0f MiB)" % BANK_MIB, dev_expert / BANK_MIB),
        ("bank+shexp (/ %.2f MiB)" % (BANK_MIB + SHEXP_MIB),
         dev_expert / (BANK_MIB + SHEXP_MIB)),
        # line 93-95: (n_layer, n_part, overflow_type) = (41, 33, GATE).
        # 41 - 33 = 8 non-overflowing layers, one of which is the MTP layer
        # whose expert tensors the loader ignored, so 7 real full banks.  The
        # GATE/UP/none ladder cost 5192 / 5045 / 4905 MiB of model, i.e. steps
        # of ~144 MiB = one expert tensor, so GATE keeps 2 of 3 tensors of the
        # boundary layer on top of those 7.
        ("fitter structure (7 full + 2/3)", 7 + 2.0 / 3.0),
    ]
    print("\n%-34s %10s %10s %10s" % ("route", "banks", "h_stock", "ceiling"))
    for name, banks in routes:
        h = banks / N_LAYER
        print("%-34s %10.3f %10.4f %9.4fx" % (name, banks, h, ceiling(h)[1]))

    lo = min(b for _, b in routes)
    hi = max(b for _, b in routes)
    mid = sum(b for _, b in routes) / len(routes)
    print("\nthree independent routes span %.2f-%.2f banks, mean %.2f" % (lo, hi, mid))
    o_new, c_new = ceiling(mid / N_LAYER)
    o_old, c_old = ceiling(9.0 / N_LAYER)
    print("  assumed  9.00 banks -> h_stock %.4f, other %.4f ms, ceiling %.4fx"
          % (9.0 / N_LAYER, o_old, c_old))
    print("  measured %.2f banks -> h_stock %.4f, other %.4f ms, ceiling %.4fx"
          % (mid, mid / N_LAYER, o_new, c_new))
    print("  the assumption was HIGH by %.2f banks; correcting it RAISES every"
          % (9.0 - mid))
    print("  multiplier in the document by %.1f%%." % (100 * (c_new / c_old - 1)))


if __name__ == "__main__":
    main()
