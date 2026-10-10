#!/usr/bin/env python3
"""How much does the ceiling move if the host leg uses the low-expert-count c?

audit-C measures 33.1 us/expert at 1 expert and 36.1 at 8.  The split runs the
host at 1-3 experts/layer, so the relevant c may be the lower figure.  `other`
is still solved at c = 35.7 here.

CORRECTED 2026-09-30: an earlier docstring justified that by saying "at
h_stock = 0.188 the host runs 6.5 experts/layer, where 35.7 is the right
constant".  That is wrong.  35.7 is 285.6/8, the EIGHT-expert rate from the
separate threads-k8 run; within audit-C itself 6.5 experts interpolate to
36.07 us/expert.  The two differ by the 1.0% cross-invocation spread, so 35.7
is defensible as a number but not for that reason.  Solving `other` with the
full measured curve instead gives 14.147 ms and a 1.399x ceiling against the
1.388x below -- see tools/host-curve.py, which also confirms the sec 13.4
ranking is unchanged.  The constant is kept here so this document has ONE cost
model; the curve is the standing check that it stays conservative.

h_stock itself is measured from the anchor run's fitter log; see
tools/hstock-from-log.py.  The second sweep below brackets it, and includes the
9/40 = 0.225 an earlier revision assumed so the size of that error is visible.
"""

D, TK, NL, ANCHOR = 14.8, 8, 40, 40.96
ANCHOR_MS = 1000.0 / ANCHOR
H_STOCK = (5192.44 - 1921.0) / (NL * (432.0 + 3.1875))
C_ANCHOR = 35.7

other = ANCHOR_MS - NL * TK * (H_STOCK * D + (1 - H_STOCK) * C_ANCHOR) / 1000.0
print("other = %.4f ms (solved at c = %.1f, h_stock = %.3f)\n" % (other, C_ANCHOR, H_STOCK))

print("%-8s %8s %12s %10s" % ("c (us)", "h*", "ceiling tok/s", "vs anchor"))
for c in (33.1, 33.48, 35.7, 36.08, 36.1):
    h_star = c / (c + D)
    floor_us = TK * c * D / (c + D)
    ceil = 1000.0 / (other + NL * floor_us / 1000.0)
    print("%-8.2f %8.4f %12.2f %10.4fx" % (c, h_star, ceil, ceil / ANCHOR))

print("\nsensitivity of the ceiling to h_stock (which sets `other`), at c = 35.7:")
for hs in (0.150, 0.175, H_STOCK, 0.200, 0.225):
    o = ANCHOR_MS - NL * TK * (hs * D + (1 - hs) * C_ANCHOR) / 1000.0
    ceil = 1000.0 / (o + NL * TK * C_ANCHOR * D / (C_ANCHOR + D) / 1000.0)
    tag = "  <- MEASURED" if hs == H_STOCK else ("  <- earlier assumption"
                                                 if hs == 0.225 else "")
    print("   h_stock = %.3f (%4.1f of 40 banks) -> other %5.2f ms, ceiling %.4fx%s"
          % (hs, hs * NL, o, ceil / ANCHOR, tag))
