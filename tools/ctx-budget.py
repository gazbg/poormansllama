#!/usr/bin/env python3
"""Which row-cache capacity still FITS as session context grows?

Every throughput figure in docs/tg-expert-path-reconciliation.md sec 13.4 was
computed at the anchor's context.  KV grows with context and competes with the
row cache for the same VRAM, so a policy that wins on throughput-per-MiB can
still be the wrong choice if it does not fit.

The accounting is not modelled, it is the anchor's own log, and it reconciles
exactly:

    model 5192.44 + KV 1328.12 + RS 60.72 + compute 427.53 = 7008.81 MiB
    fitter: "41 layers (33 overflowing), 7008 MiB used"

so the same five terms can be re-evaluated at another context with nothing
added.  All figures from "docs/llamacpp 8gb vram base.md", build 10892.

Budget is reported at two values on purpose.  `--fit-target 16000` on a 24,462
MiB card looks like an 8,462 MiB budget, and several sections of the
reconciliation document say "8,416 MiB" on that reasoning.  The fitter itself
disagrees, in writing: line 49 prints "id=0, target=7055 MiB", because it
measures free memory AFTER CUDA init, when 1,407 MiB of the card is already
spoken for.  Its own arithmetic agrees -- line 95, "7008 MiB used, 16046 MiB
free", is 7054 available against a 16000 target, one MiB of rounding away.
Section 10.1 established this; it is quoted here, not re-derived.

Which budget applies depends on whether the 8 GB target is a bare card or a
card also driving a desktop.  This is not a rounding difference: it decides
whether any row cache reaches the settled 236,000-token session context.
"""

DENSE_MIB = 1921.0           # dense-only probe, sec 12.1b (reproduces in a 2nd run)
# The fitter's memory column lumps KV and recurrent state: line 90 prints
# "7008 = 5192 + 1388 + 427", and 1328.12 + 60.72 = 1388.84 exactly, so these
# two constants are that middle column split by the loader's own two lines.
RS_MIB = 60.72               # CUDA0 recurrent-state buffer
COMPUTE_MIB = 427.53         # CUDA0 compute buffer (set by ubatch 512, not by ctx)
# llama_kv_cache: 1328.12 MiB (128000 cells, 10 layers), K/V both q8_0.
# Only 10 of 40 layers carry a KV cache; the other 30 are recurrent, which is
# why context costs a quarter of what a dense transformer would charge.
# Confirmed against the unrelated 512k run: "5312.50 MiB (256000 cells,
# 10 layers, 2/2 seqs)" = 0.020752 MiB/cell = exactly 2x this, i.e. the same
# per-token-per-sequence cost.  SHIP_CTX below is therefore one slot.
KV_MIB_PER_TOKEN = 1328.12 / 128000.0

ROW_MIB = 1769472 / (1024.0 * 1024.0)
N_LAYER = 40
SHIP_CTX = 236000            # settled per-session context
BUDGETS = [("realised: fitter's own \"id=0, target=7055 MiB\", sec 10.1", 7055.0),
           ("nominal: bare card, 24462 - 16000", 8462.0)]
CAPS = [(48, "LRU-K K=6"), (56, "56-row draft"), (64, "LRU-K K=8")]


def cache_mib(rows):
    return N_LAYER * rows * ROW_MIB


def main():
    print("fixed VRAM: dense %.0f + RS %.1f + compute %.1f = %.1f MiB"
          % (DENSE_MIB, RS_MIB, COMPUTE_MIB, DENSE_MIB + RS_MIB + COMPUTE_MIB))
    print("KV costs %.5f MiB per context token (10 of 40 layers, q8_0)"
          % KV_MIB_PER_TOKEN)
    print("anchor reconciles: 5192.44 + 1328.12 + 60.72 + 427.53 = %.2f"
          " vs fitter's 7008 MiB\n" % (5192.44 + 1328.12 + 60.72 + 427.53))

    fixed = DENSE_MIB + RS_MIB + COMPUTE_MIB
    for label, budget in BUDGETS:
        print("budget %.0f MiB -- %s" % (budget, label))
        print("   %-22s %10s %12s %10s %s"
              % ("policy", "cache MiB", "max ctx", "at 236k", "spare MiB"))
        for rows, name in CAPS:
            c = cache_mib(rows)
            mc = (budget - fixed - c) / KV_MIB_PER_TOKEN
            spare = budget - fixed - c - SHIP_CTX * KV_MIB_PER_TOKEN
            print("   %-22s %10.0f %12.0f %10s %9.0f"
                  % ("%d rows, %s" % (rows, name), c, max(mc, 0.0),
                     "YES" if spare >= 0 else "no", spare))
        print("   %-22s %10.0f %12.0f %10s %9.0f"
              % ("(no cache, KV only)", 0.0, (budget - fixed) / KV_MIB_PER_TOKEN,
                 "YES", budget - fixed - SHIP_CTX * KV_MIB_PER_TOKEN))
        print()

    print("Stock is not on the table above because its expert residency is not")
    print("a fixed cache -- it is whatever the fitter has left over, so it")
    print("SHRINKS as context grows while a row cache's does not.  That is an")
    print("asymmetry, not a detail: stock DEGRADES at high context, a fixed row")
    print("cache FAILS TO FIT.\n")
    print("Read this 2x2 DOWN the columns, not across: budget is a property of")
    print("the machine, context is the thing being scaled.  Comparing 0.188 at")
    print("one budget with a figure at the other budget conflates the two.\n")
    print("   %-22s %12s %12s" % ("stock, leftover ->", "128,000 ctx", "%d ctx" % SHIP_CTX))
    for label, budget in BUDGETS:
        cells = []
        for ctx in (128000, SHIP_CTX):
            left = budget - fixed - ctx * KV_MIB_PER_TOKEN
            cells.append("%.0f MiB %.3f" % (left, left / (N_LAYER * (432.0 + 3.1875))))
        print("   %-22s %12s %12s" % ("budget %.0f MiB" % budget, cells[0], cells[1]))
    anchor_left = 5192.44 - DENSE_MIB
    modelled = BUDGETS[0][1] - fixed - 128000 * KV_MIB_PER_TOKEN
    print("\n   (the %.0f/128000 cell is the anchor itself: %.0f MiB, h %.3f --"
          % (BUDGETS[0][1], anchor_left,
             anchor_left / (N_LAYER * (432.0 + 3.1875))))
    print("    the model reproduces it to %.0f MiB, which is the %.0f MiB of"
          % (abs(anchor_left - modelled), BUDGETS[0][1] - 7008.81))
    print("    headroom the fitter left unspent.  No free parameters.)")
    print("\nAt the anchor's OWN budget, going 128k -> %dk costs stock %.1f banks"
          % (SHIP_CTX // 1000,
             (modelled - (BUDGETS[0][1] - fixed - SHIP_CTX * KV_MIB_PER_TOKEN))
             / 432.0))
    print("and drops h_stock 0.188 -> 0.126.  So 0.188 is a best case for that")
    print("machine, not a typical one, and the sec 13.4 multipliers are")
    print("understated at the shipping context rather than overstated.  On a")
    print("bare card the extra %.0f MiB more than pays for the extra KV, so"
          % (BUDGETS[1][1] - BUDGETS[0][1]))
    print("stock instead RISES to 0.207 -- better hardware, not better scaling.")
    print("The 512,000 run on disk shows where the trend ends: there the fitter")
    print("got only 1311.95 MiB of model onto the device, BELOW the 1921 MiB of")
    print("dense weight, i.e. zero expert banks and dense itself spilling.")


if __name__ == "__main__":
    main()
