#!/usr/bin/env python3
"""Bound the unmeasured device expert-FFN cost `d` from the two measured paths.

Both the stock path and the streaming path pay the same non-expert remainder
`O`, so differencing them eliminates it and leaves `d` and the streaming path's
rowset machinery `M`:

    stock:      O + dA + 0.320*(h*d + (1-h)*c) = 24.41 ms  (anchor, 40.96 tok/s, ctx 52,819)
    streaming:  O + 0.320*d + M                = I        (fitted intercept, no bytes staged)

Two equations, three unknowns; `M >= 0` closes it from one side. `dA` is the
attention the anchor pays at 52,819 tokens of context and the short-context
streaming runs do not (ctx 768 / 1280): expected ~0.6-1.0 ms from reading
574.7 MB of device-resident q8 KV per token, unmeasured, so it is reported
as a sensitivity rather than subtracted.

Recomputed 2026-09-30 after h_stock was measured off the anchor's fitter log
(0.188, tools/hstock-from-log.py) rather than guessed at 9/40 = 0.225.

Recomputed again 2026-09-30 (eighteenth finding): the 28.92 ms intercept came
from pooling cache-off and cache-on arms, which run different execution shapes
(src/llama-layer-streaming.cpp load_row_set, retention branch), and is biased
up. The cache-on-only fits are the path's own intercepts.
"""

C_HOST_US, TOP_K, N_LAYER = 35.7, 8, 40
STOCK_MS = 1000.0 / 40.96
H_STOCK = (5192.44 - 1921.0) / (N_LAYER * (432.0 + 3.1875))
K = N_LAYER * TOP_K / 1000.0          # ms per us of per-expert cost
D_WORKING_US = 14.8
INTERCEPTS = (                        # (label, ms/token)
    ("A/B pooled k6+k8 (withdrawn)", 28.92),
    ("cache-on prompt A, ctx 768", 26.47),
    ("cache-on kprobe off, ctx 1280", 23.69),
)


def report(h, label, stream_ms, d_a=0.0):
    coef = K * (1 - h)
    # 0.320*(1-h)*d + M = (I - stock) + 0.320*(1-h)*c + dA
    rhs = (stream_ms - STOCK_MS) + coef * C_HOST_US + d_a
    d_max = rhs / coef
    m_at = rhs - coef * D_WORKING_US
    print("%-30s I %.2f dA %.1f: %.5f*d + M = %.3f ms -> d <= %.1f us, M(d=%.1f) = %.2f ms (%.0f%% of a stock token)"
          % (label, stream_ms, d_a, coef, rhs, d_max, D_WORKING_US, m_at, 100 * m_at / STOCK_MS))
    return d_max, m_at


def main():
    print("stock %.2f ms/token, c = %.1f us, h_stock %.4f (measured) vs %.4f (earlier assumption)\n"
          % (STOCK_MS, C_HOST_US, H_STOCK, 9.0 / N_LAYER))
    new = report(H_STOCK, "withdrawn A/B, measured h", 28.92)
    old = report(9.0 / N_LAYER, "withdrawn A/B, assumed h", 28.92)
    print("  measuring h_stock moved the withdrawn figures: d bound %.1f -> %.1f us, M %.2f -> %.2f ms\n"
          % (old[0], new[0], old[1], new[1]))
    for label, ms in INTERCEPTS[1:]:
        for d_a in (0.0, 1.0):
            report(H_STOCK, label, ms, d_a)


if __name__ == "__main__":
    main()
