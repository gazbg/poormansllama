#!/usr/bin/env python3
"""Throughput of each expert-placement policy under the validated cost model.

Cost model (validated 2026-09-30 to 13% against the measured streaming path):

    token_wall = other + n_layer * top_k * per_expert(h)

    per_expert(h) = h*d + (1-h)*c                serial
                  = max(h*d, (1-h)*c)            host and device overlapped

  c = 35.7 us   host expert FFN, per expert, DDR-bandwidth-bound  (MEASURED)
  d = 14.8 us   device expert FFN, per expert                     (NOT MEASURED)
  other         everything else in the token, solved from the anchor run

`other` is solved so the model reproduces the stock anchor exactly, which means
it absorbs whatever error is in d at the baseline h.

All hit rates below are MEASURED, not interpolated: LRU-K from the run's own
`n_rows_resident` telemetry (second half, matching the "steady state" label),
the placement policies from `tools/expert-skew.py` replaying the same traces'
routes.  Two prompts, reported per prompt, because they disagree materially and
averaging them first is a modelling error (see the note at the table).

Every figure in MEASURED was re-derived from the raw traces on 2026-09-30 and
all fourteen reproduce exactly:
    prompt A  = build/llama-cuda/gplr19-k8-ab/arm-B1.jsonl
    prompt B  = build/llama-cuda/gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl
"""

N_LAYER = 40
TOP_K = 8
C_HOST_US = 35.7
D_DEV_US = 14.8
ROW_BYTES = 1769472

# stock anchor: docs/llamacpp 8gb vram base.md, build 10892, --fit-target 16000
ANCHOR_TOK_S = 40.96
ANCHOR_MS = 1000.0 / ANCHOR_TOK_S
# h_stock MEASURED from the anchor run's own fitter log, not inferred.
# "docs/llamacpp 8gb vram base.md" prints a memory breakdown for the probe
# configuration with all MoE tensors moved to system memory; its model column
# is the dense-only device size, 1921 MiB.  The configuration actually served
# used 5192.44 MiB, so 3271.44 MiB of expert weight was resident, out of
# 40 x (432 + 3.19) MiB.  Three independent routes (bank-only, bank+shexp, and
# the fitter's own "(41, 33, GATE)" structure) agree on 7.52-7.67 banks.
# An earlier revision assumed 9 banks from a guessed 1302 MiB of dense weight;
# that was high by 1.4 banks and understated every multiplier here by 1.3%.
# See tools/hstock-from-log.py.
H_STOCK = (5192.44 - 1921.0) / (N_LAYER * (432.0 + 3.1875))

# (capacity rows/layer, policy, h on prompt B "honest", h on prompt A)
MEASURED = [
    (48, "static build-time pin (held out)", 0.3340, 0.3884),
    (48, "warm-then-pin, W=128, frozen",     0.5087, 0.7270),
    (48, "LRU-K K=6 (steady state)",         0.6345, 0.7156),
    (64, "static build-time pin (held out)", 0.4117, 0.4839),
    (64, "warm-then-pin, W=128, frozen",     0.5979, 0.8143),
    (64, "LRU-K K=8 (steady state)",         0.7198, 0.7860),
    (64, "in-sample oracle (ceiling)",       0.7572, 0.8727),
]


def expert_us(h, overlapped, d=None):
    d = D_DEV_US if d is None else d
    dev = h * TOP_K * d
    host = (1.0 - h) * TOP_K * C_HOST_US
    return max(dev, host) if overlapped else dev + host


def tok_s(h, overlapped, other_ms, d=None):
    return 1000.0 / (other_ms + N_LAYER * expert_us(h, overlapped, d) / 1000.0)


def main():
    other_ms = ANCHOR_MS - N_LAYER * expert_us(H_STOCK, False) / 1000.0
    if other_ms <= 0:
        raise SystemExit("anchor cannot support this cost model: other <= 0")

    print("c = %.1f us host (measured), d = %.1f us device (NOT measured)"
          % (C_HOST_US, D_DEV_US))
    print("anchor %.2f tok/s = %.2f ms, h_stock = %.3f -> other = %.2f ms"
          % (ANCHOR_TOK_S, ANCHOR_MS, H_STOCK, other_ms))

    # Under overlap the two legs cross at h* = c/(c+d).  Below it the host leg
    # is the bottleneck, above it the device leg is -- so hit rate beyond h* is
    # not merely useless, it is actively worse.  This is the ceiling of the
    # whole mechanism, independent of any placement policy.
    h_star = C_HOST_US / (C_HOST_US + D_DEV_US)
    floor_us = TOP_K * C_HOST_US * D_DEV_US / (C_HOST_US + D_DEV_US)
    print("\noverlapped optimum: h* = c/(c+d) = %.4f, cost %.1f us/layer, "
          "%.2f tok/s = %.2fx"
          % (h_star, floor_us, tok_s(h_star, True, other_ms),
             tok_s(h_star, True, other_ms) / ANCHOR_TOK_S))
    print("hit rate above h* moves the bottleneck onto the GPU and costs more.")

    print("\n%-6s %-34s %15s %16s %16s"
          % ("rows", "policy", "h (B / A)", "serial tok/s", "overlap tok/s"))
    print("%-6s %-34s %15s %16s %16s"
          % ("", "stock whole-bank, 7.5 of 40 L", "%.3f" % H_STOCK,
             "%.2f  1.00x" % ANCHOR_TOK_S, "%.2f  1.00x" % ANCHOR_TOK_S))
    for rows, name, hb, ha in MEASURED:
        # Evaluate each prompt separately and average the THROUGHPUTS.
        # Averaging the hit rates first and evaluating once is Jensen's
        # inequality pointing the wrong way: throughput is concave in h (the
        # overlapped cost is a max of two lines, minimised at h*), so
        # tok_s(mean h) >= mean tok_s.  The gap reaches +0.06x for
        # warm-then-pin at 64 rows, whose two prompts straddle h* almost
        # symmetrically -- their mean lands 0.0008 from h*, so the old code
        # reported the ceiling and attributed it to the policy.  See
        # tools/jensen-check.py.
        s = (tok_s(hb, False, other_ms) + tok_s(ha, False, other_ms)) / 2.0
        o = (tok_s(hb, True, other_ms) + tok_s(ha, True, other_ms)) / 2.0
        olo, ohi = sorted((tok_s(hb, True, other_ms), tok_s(ha, True, other_ms)))
        mib = N_LAYER * rows * ROW_BYTES / (1024 * 1024)
        print("%-6s %-34s %15s %16s %16s  %s"
              % ("%d/%.0fM" % (rows, mib), name,
                 "%.3f / %.3f" % (hb, ha),
                 "%.2f  %.2fx" % (s, s / ANCHOR_TOK_S),
                 "%.2f  %.2fx" % (o, o / ANCHOR_TOK_S),
                 "per-prompt %.2f-%.2fx" % (olo / ANCHOR_TOK_S,
                                            ohi / ANCHOR_TOK_S)))

    print("\nThe overlap column is the mean of the two prompts' throughputs, and")
    print("the per-prompt range is what a single session actually gets: the")
    print("placement is frozen per session, so a session sees one prompt's hit")
    print("rate, never the average of two.  Judge policies on the range.")

    print("\nsensitivity of the overlapped ceiling to the unmeasured d:")
    print("(the h=0.706 column is a hypothetical policy that lands exactly on")
    print(" h* at d=14.8, not warm-then-pin -- warm-then-pin achieves 0.598 or")
    print(" 0.814 depending on the prompt, and neither is 0.706.")
    print(" the h=0.646 column is the 56-row LRU-K K=6 figure the placement")
    print(" draft quotes, kept here so that draft cites a script.)")
    print("   %-12s %8s %9s %11s %11s %11s"
          % ("d", "h*", "ceiling", "h=0.706", "h=0.646 ov", "h=0.646 ser"))
    for d in (8.0, 10.0, 14.8, 20.0, 25.0, 30.0):
        hs = C_HOST_US / (C_HOST_US + d)
        o = ANCHOR_MS - N_LAYER * expert_us(H_STOCK, False, d) / 1000.0
        print("   d = %4.1f us %8.3f %8.2fx %10.2fx %10.2fx %10.2fx"
              % (d, hs,
                 tok_s(hs, True, o, d) / ANCHOR_TOK_S,
                 tok_s(0.706, True, o, d) / ANCHOR_TOK_S,
                 tok_s(0.646, True, o, d) / ANCHOR_TOK_S,
                 tok_s(0.646, False, o, d) / ANCHOR_TOK_S))


if __name__ == "__main__":
    main()
