#!/usr/bin/env python3
"""Thread optimum as a function of expert count -- and what it does to the model.

MEASURED 2026-09-30, host-only, no GPU, no mutex (same footing as sec 12).
  build/expert-cpu-bench/threads-x-experts.json
  --pool 256 --id-stride 8 --iters 1000 --threads 1,2,4,6,8,12,16,32
                                        --experts 1,2,3,4,8

The gap this closes.  Section 12.1 swept THREADS at k=8 and concluded "past 8
it gets worse".  Section 12.1a swept EXPERT COUNT at 8 threads.  Nobody swept
the cross-product -- yet the overlapped split runs the host at (1-h)*8 = 2-3
experts, a 4 MB working set, not k=8's 14.2 MB.  A saturation point measured at
14.2 MB does not transfer to 4 MB, and it does not: the optimum moves to 12
threads and 8 threads leaves 13-14% on the table there.

Two separate things fall out, and they must not be confused:

  1. A CORRECTION.  `other` is solved by making the model reproduce the anchor.
     The anchor ran at n_threads = 16 ("docs/llamacpp 8gb vram base.md" line 24,
     "n_threads = 16 (n_threads_batch = 16) / 32").  The model solves `other`
     with c = 35.7 us, which is 285.6/8 -- an EIGHT-thread figure.  That is a
     unit mismatch, not a tuning choice.

  2. An OPTIMIZATION.  Running the host expert leg at its per-regime optimum
     instead of the anchor's 16 threads is free throughput, and it lands on the
     binding leg below h*, so it also moves h* itself.

CAVEAT, stated rather than buried: the bench gives the CPU backend the whole
box to run expert FFN alone.  In the real split the device leg runs
concurrently and the CPU is also doing attention, the router and the shared
expert.  Thread counts that win in isolation may contend there.  Every figure
below is therefore an upper bound on the thread-tuning gain, not a promise.

PREFILL, corrected 2026-09-30 (sec 12.5).  An earlier version priced stock
prefill as the host expert FFN on the CPU and projected --threads-batch onto
the anchor's 152.13 s.  Stock does not take that path: at a batch of 32
tokens or more the CUDA backend's op offload (ggml-cuda.cu:5645) copies every
host weight a ubatch uses to the GPU and runs it there, so --threads-batch
moves no host expert and prefill is a copy schedule.  pp() keeps the host-FFN cells,
which price --no-op-offload only; copy_model() prices the levers from the
anchor's logged cadence.  "The thread lever strictly dominates the ubatch
lever" is retracted: under op offload ubatch is the lever.

TG, corrected 2026-09-30 (twelfth finding).  An earlier version priced the
stock-only thread gain as 40 layers at the interpolated (1-h_stock)*8 = 6.5
experts and published +1.36% at 12 threads.  Stock keeps each bank wholly on
the host or wholly on the device, so each of its 32.79 host bank-equivalents
runs all 8 experts on the CPU and each device layer runs none: the k=8 cell
prices it.  The same cells then give +0.92%, and tg_threads() reads every
invocation that has the k=8 pair to show why even that is unproven.
"""

import json
import os
import re
import sys

D_DEV_US, TOP_K, N_LAYER = 14.8, 8, 40
ANCHOR_TOK_S = 40.96
ANCHOR_MS = 1000.0 / ANCHOR_TOK_S
ANCHOR_THREADS = 16          # log line 24, not a choice
H_STOCK = (5192.44 - 1921.0) / (N_LAYER * (432.0 + 3.1875))
JSON = os.path.join(os.path.dirname(__file__), "..", "build",
                    "expert-cpu-bench", "threads-x-experts.json")

# sec 13.4's measured policies: rows, name, h on arm B (cache on), h on arm A.
MEASURED = [
    (48, "static build-time pin (held out)", 0.3340, 0.3884),
    (48, "warm-then-pin, W=128, frozen",     0.5087, 0.7270),
    (48, "LRU-K K=6 (steady state)",         0.6345, 0.7156),
    (64, "static build-time pin (held out)", 0.4117, 0.4839),
    (64, "warm-then-pin, W=128, frozen",     0.5979, 0.8143),
    (64, "LRU-K K=8 (steady state)",         0.7198, 0.7860),
    (64, "in-sample oracle (ceiling)",       0.7572, 0.8727),
]


def load():
    with open(JSON) as f:
        cells = json.load(f)["cells"]
    # TG only: one token per pass.
    grid = {}
    for c in cells:
        if c["n_tokens"] != 1:
            continue
        grid.setdefault(c["threads"], {})[c["experts_used"]] = c["median_us"]
    return grid


def load_pp():
    """k=8, 512 tokens/call -- the PP cell, merged over every invocation."""
    here = os.path.dirname(JSON)
    runs = {}
    for tag in ("pp-threads", "pp-threads-fine", "pp-threads-top"):
        p = os.path.join(here, tag + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for c in json.load(f)["cells"]:
                if c["n_tokens"] == 512 and c["experts_used"] == 8:
                    runs.setdefault(c["threads"], []).append(c["median_us"])
    return runs


def host_us(grid, threads, n):
    """Total host us for n experts at a thread count, piecewise-linear in n."""
    row = grid[threads]
    ns = sorted(row)
    if n <= 0:
        return 0.0
    if n <= ns[0]:
        return n * row[ns[0]] / ns[0]
    for a, b in zip(ns, ns[1:]):
        if n <= b:
            return row[a] + (row[b] - row[a]) * (n - a) / (b - a)
    return row[ns[-1]]


def best_threads(grid, n):
    return min(grid, key=lambda t: host_us(grid, t, n))


def h_star(grid, threads):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo + hi) / 2.0
        if m * TOP_K * D_DEV_US < host_us(grid, threads, (1.0 - m) * TOP_K):
            lo = m
        else:
            hi = m
    return (lo + hi) / 2.0


def main():
    grid = load()
    threads = sorted(grid)
    experts = sorted(grid[threads[0]])

    print("median us, 1 token, by thread count x expert count:")
    print("   %-9s %s" % ("threads", "".join("%10d" % e for e in experts)))
    for t in threads:
        row = "".join("%10.1f" % grid[t][e] for e in experts)
        print("   %-9d %s" % (t, row))

    print("\nbest thread count per expert count (and what 8 threads costs):")
    print("   %-9s %8s %10s %10s %8s" % ("experts", "best t", "best us",
                                         "8 thr us", "8thr pays"))
    for e in experts:
        bt = min(threads, key=lambda t: grid[t][e])
        pen = 100.0 * (grid[8][e] - grid[bt][e]) / grid[bt][e]
        print("   %-9d %8d %10.1f %10.1f %7.1f%%" % (e, bt, grid[bt][e],
                                                     grid[8][e], pen))
    print("   sec 12.1 concluded 'past 8 threads it gets worse'.  True at k=8,")
    print("   which is where it measured.  False at 2-4 experts, which is where")
    print("   the split runs.")

    n_stock = (1.0 - H_STOCK) * TOP_K
    print("\n-- CORRECTION: `other` was solved with an 8-thread constant --")
    print("   anchor ran at n_threads = %d; at h_stock = %.3f the host runs"
          % (ANCHOR_THREADS, H_STOCK))
    print("   %.2f experts/layer." % n_stock)
    const = (1.0 - H_STOCK) * TOP_K * 35.7
    meas = host_us(grid, ANCHOR_THREADS, n_stock)
    print("   model's constant c = 35.7 (8-thread)  -> %7.2f us/layer" % const)
    print("   measured at the anchor's %2d threads   -> %7.2f us/layer (%+.2f)"
          % (ANCHOR_THREADS, meas, meas - const))
    o_pub = ANCHOR_MS - N_LAYER * (H_STOCK * TOP_K * D_DEV_US + const) / 1000.0
    o_fix = ANCHOR_MS - N_LAYER * (H_STOCK * TOP_K * D_DEV_US + meas) / 1000.0
    print("   other: %.3f ms published -> %.3f ms corrected (%+.3f)"
          % (o_pub, o_fix, o_fix - o_pub))

    print("\n-- OPTIMIZATION: run the host leg at its per-regime optimum --")
    print("   %-26s %9s %9s %9s" % ("", "h*", "floor us", "ceiling"))
    for t in (ANCHOR_THREADS, 8, 12):
        hs = h_star(grid, t)
        floor = host_us(grid, t, (1.0 - hs) * TOP_K)
        ceil = 1000.0 / (o_fix + N_LAYER * floor / 1000.0)
        print("   host leg at %2d threads     %9.4f %9.1f %8.3fx"
              % (t, hs, floor, ceil / ANCHOR_TOK_S))
    hs = h_star(grid, 12)
    print("   (12 threads is the optimum at %.2f experts, which is what the"
          % ((1 - hs) * TOP_K))
    print("    split runs at h*.  The published ceiling is 1.388x.)")

    print("\n-- and what stock alone would gain, no split, just threads --")
    old = N_LAYER * (meas - host_us(grid, 12, n_stock)) / 1000.0
    print("   RETRACTED: %+.2f%% at 12 threads priced %d layers at an interpolated"
          % (100.0 * old / ANCHOR_MS, N_LAYER))
    print("   %.2f experts.  Stock runs all %d experts on each of its %.2f host"
          % (n_stock, TOP_K, L_HOST))
    print("   bank-equivalents and none on a device layer, so the k=%d cell"
          % TOP_K)
    share = L_HOST * grid[ANCHOR_THREADS][TOP_K] / 1000.0
    print("   prices it: %.2f x %.1f us = %.2f ms, %.1f%% of the token."
          % (L_HOST, grid[ANCHOR_THREADS][TOP_K], share,
             100.0 * share / ANCHOR_MS))
    for t in (8, 12):
        gain = L_HOST * (grid[ANCHOR_THREADS][TOP_K] - grid[t][TOP_K]) / 1000.0
        print("   stock host leg %2d -> %2d threads: %+.3f ms/token = %+.2f%% "
              "-> %.2f tok/s"
              % (ANCHOR_THREADS, t, -gain, 100.0 * gain / ANCHOR_MS,
                 1000.0 / (ANCHOR_MS - gain)))
    o_stock = ANCHOR_MS - (L_HOST * grid[ANCHOR_THREADS][TOP_K]
                           + (N_LAYER - L_HOST) * TOP_K * D_DEV_US) / 1000.0
    hs = h_star(grid, ANCHOR_THREADS)
    floor = host_us(grid, ANCHOR_THREADS, (1.0 - hs) * TOP_K)
    print("   the same error moves `other`: %.3f ms in this regime, not %.3f"
          % (o_stock, o_fix))
    print("   (%+.1f%%), so the %d-thread ceiling is %.3fx, not %.3fx.  Inside the"
          % (100.0 * (o_stock / o_fix - 1), ANCHOR_THREADS,
             1000.0 / (o_stock + N_LAYER * floor / 1000.0) / ANCHOR_TOK_S,
             1000.0 / (o_fix + N_LAYER * floor / 1000.0) / ANCHOR_TOK_S))
    print("   0.014x cross-invocation spread, so the tables keep %.3f." % o_fix)
    tg_threads(grid, 100.0 * gain / ANCHOR_MS)

    # The ceiling barely moves (above h* the device leg binds and host cost is
    # off the critical path -- sec 12.1a's point, and it holds).  But every
    # MEASURED policy except the two best sits BELOW h*, host-bound, where the
    # thread gain lands in full.  That is the table worth printing.
    print("\n-- per-policy, at the anchor's 16 threads vs the optimum 12 --")
    print("   %-6s %-34s %13s %9s %9s %7s"
          % ("rows", "policy", "h (B / A)", "16 thr", "12 thr", "gain"))
    rank16, rank12 = [], []
    for rows, name, hb, ha in MEASURED:
        v = []
        for t in (ANCHOR_THREADS, 12):
            f = lambda h: 1000.0 / (o_fix + N_LAYER * max(
                h * TOP_K * D_DEV_US,
                host_us(grid, t, (1.0 - h) * TOP_K)) / 1000.0)
            v.append((f(hb) + f(ha)) / 2.0)
        rank16.append((name + " @%d" % rows, v[0]))
        rank12.append((name + " @%d" % rows, v[1]))
        bound = "host" if max(hb, ha) < h_star(grid, 12) else "dev"
        print("   %-6d %-34s %13s %8.3fx %8.3fx %+6.3fx  %s"
              % (rows, name, "%.3f / %.3f" % (hb, ha), v[0] / ANCHOR_TOK_S,
                 v[1] / ANCHOR_TOK_S, (v[1] - v[0]) / ANCHOR_TOK_S, bound))
    r16 = [n for n, _ in sorted(rank16, key=lambda r: -r[1])]
    r12 = [n for n, _ in sorted(rank12, key=lambda r: -r[1])]
    print("   RANKING %s" % ("UNCHANGED" if r16 == r12 else "*** CHANGED ***"))
    print("   the deployable policy (LRU-K K=6 @48, the only one that reaches")
    print("   236k on this budget, sec 13.6) is host-bound, so it takes the")
    print("   thread gain in full.")

    if r16 != r12:
        d16, d12 = dict(rank16), dict(rank12)
        print("\n   at 16 threads: %s" % " > ".join(r16[:2]))
        print("   at 12 threads: %s" % " > ".join(r12[:2]))
        m16 = (d16[r16[0]] - d16[r16[1]]) / ANCHOR_TOK_S
        m12 = (d12[r12[0]] - d12[r12[1]]) / ANCHOR_TOK_S
        print("   margin %+.4fx -> %+.4fx.  BOTH margins are inside the 1.0%%"
              % (m16, m12))
        print("   cross-invocation spread on the host measurements (sec 12.1),")
        print("   which on a ~1.38x figure is about 0.014x.  So the honest")
        print("   reading is not '48 rows wins on throughput' -- it is that the")
        print("   two are TIED, and the tie-break is capacity: 48 rows is the")
        print("   only one that fits 236k at all (sec 13.6).  Thread tuning")
        print("   removes the one argument that pointed the other way.")

    pp(grid)


VERIFY_JSON = os.path.join(os.path.dirname(JSON), "verify-shape.json")
# Every invocation that ran the k=8 one-token cell at 16 threads and below.
PAIR_RUNS = ("pp-shape", "pp-threads-fine", "threads-k8", "threads-x-experts",
             "verify-shape")
CANDIDATE_MIN = 8                    # the lowest --threads sec 12.1c/12.2 proposed


def tg_threads(grid, pct12):
    """The k=8 one-token cell at 16 threads against each lower count, in every
    invocation that ran both.  The bench runs threads outermost, in --threads
    order (expert-cpu-bench.cpp:581), and writes its cells in run order, so
    each file records the order its thread counts ran in."""
    print("   RETRACTED as a point estimate: the %+.2f%% rests on the k=%d cell."
          % (pct12, TOP_K))
    print("   every invocation that has it, priced in the stock regime:")
    print("   %-18s %6s %-24s %7s %8s %8s %7s"
          % ("invocation", "stride", "thread order", "threads", "16 thr",
             "fewer", "token"))
    votes = {True: 0, False: 0}
    for tag in PAIR_RUNS:
        p = os.path.join(os.path.dirname(JSON), tag + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = json.load(f)
        order, k8 = [], {}
        for c in data["cells"]:
            if not order or order[-1] != c["threads"]:
                order.append(c["threads"])
            if c["n_tokens"] == 1 and c["experts_used"] == TOP_K:
                k8[c["threads"]] = c["median_us"]
        lower = [t for t in sorted(k8) if CANDIDATE_MIN <= t < ANCHOR_THREADS]
        if ANCHOR_THREADS not in k8 or not lower:
            continue
        votes[min(k8[t] for t in lower) < k8[ANCHOR_THREADS]] += 1
        for t in lower:
            d = k8[ANCHOR_THREADS] - k8[t]
            print("   %-18s %6s %-24s %7d %8.1f %8.1f %+6.2f%%"
                  % (tag, data.get("id_stride", "-"),
                     ",".join(str(x) for x in order), t, k8[ANCHOR_THREADS],
                     k8[t], 100.0 * L_HOST * d / 1000.0 / ANCHOR_MS))
    print("   fewer threads win in %d invocations, %d threads in %d.  Each ran its"
          % (votes[True], ANCHOR_THREADS, votes[False]))
    print("   thread counts in the order shown, so thread count is confounded")
    print("   with run order.")

    with open(VERIFY_JSON) as f:
        data = json.load(f)
    cells = {(c["threads"], c["experts_used"]): c["median_us"]
             for c in data["cells"] if c["n_tokens"] == 1}
    ks = sorted(k for t, k in cells if t == 12 and (16, k) in cells)
    if not ks:
        sys.exit("%s: no one-token cells at both 12 and 16 threads" % VERIFY_JSON)
    print("   verify-shape.json (id-stride %d), in run order, per expert us:"
          % data["id_stride"])
    print("   %-8s %9s %9s %8s %8s %9s" % ("experts", "12 thr", "16 thr",
                                           "12/exp", "16/exp", "12 vs 16"))
    for k in ks:
        a, b = cells[(12, k)], cells[(16, k)]
        print("   %-8d %9.1f %9.1f %8.1f %8.1f %+8.1f%%"
              % (k, a, b, a / k, b / k, 100.0 * (a / b - 1)))
    per12 = [cells[(12, k)] / k for k in ks]
    wins = [k for k in ks[3:] if cells[(12, k)] < cells[(16, k)]]
    print("   the 12-thread sweep ran first and its per-expert cost climbs from")
    print("   %.1f to %.1f us across it; 12 threads wins its first three cells,"
          % (per12[0], max(per12)))
    print("   then %d of the %d after them (k=%s).  One invocation cannot tell box"
          % (len(wins), len(ks) - 3, ",".join(str(k) for k in wins) or "none"))
    print("   drift from a real k-dependence.  Neither size nor sign is settled:")
    print("   llama-bench -p 0 -n 128 -r 5 at -t 16,12,8 and then -t 8,12,16")
    print("   decides (sec 12.5 A/B vehicle, item 3).")


# sec 12.5: stock prefill under op offload.  Every input is logged, measured
# or bounded, and says which.
ANCHOR_LOG = os.path.join(os.path.dirname(__file__), "..", "docs",
                          "llamacpp 8gb vram base.md")
N_BATCH, UBATCH = 8192, 512          # anchor log lines 280-281
# tools/bank-bytes.py --ngl 41 --n-part 33 --overflow-type 3 reproduces the
# logged CUDA0 model buffer (5192.44 MiB) exactly: block 0 sits on the CPU whole.
HOST_EXPS_B = 14854127616            # routed banks of layers 0, 9..39 + down of 8
HOST_WHOLE_B = 41771776 + 1114112    # block 0 dense + blk.8.ffn_down_shexp, copied whole
BANK_B, N_EXPERT = 452984832, 256    # one Q4_K bank, the unit the host bench runs
L_HOST = HOST_EXPS_B / BANK_B        # stock's host bank-equivalents (tenth finding)
# Mean route-union width per layer over one 512-token chunk, sec 15.3
# (build/llama-cuda/gplr14-route-overlap.txt, gplr12-qa-512: another prompt).
USED_U512 = 254
OFFLOAD_MIN_BATCH = 32               # ggml-cuda.cu:5645, GGML_OP_OFFLOAD_MIN_BATCH default
WIRE_GBPS = BANK_B / 31.5e-3 / 1e9   # copy_gpu_ms p50, AGENT_ENVIRONMENT.md:1214 ("~14.6")
LM_TOK_S = 2532.5                    # research engine steady u512 (sec 11.7.2)
G_HI = UBATCH / LM_TOK_S             # device compute per 512 tokens cannot exceed this
G_LO = 0.05                          # ASSUMED optimistic end, not measured
PROG = re.compile(r"prompt processing, n_tokens =\s*(\d+), progress = [\d.]+, "
                  r"t =\s*([\d.]+) s")
PEVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def load_shape():
    """{threads: {tokens: us}} for k=8 from ubatch-shape.json, or {}."""
    p = os.path.join(os.path.dirname(JSON), "ubatch-shape.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        cells = [c for c in json.load(f)["cells"] if c["experts_used"] == 8]
    g = {}
    for c in cells:
        g.setdefault(c["threads"], {})[c["n_tokens"]] = c["median_us"]
    return g


def pp(grid):
    runs = load_pp()
    if not runs:
        return
    med = {t: sum(v) / len(v) for t, v in runs.items()}
    print("\n== PREFILL host expert FFN: k=8 over one 512-token ubatch ==")
    print("   prices --no-op-offload ONLY: stock never runs this cell at prefill")
    print("   (copy model below).")
    print("   %-9s %6s %26s %8s %10s"
          % ("threads", "runs", "median us per run", "spread", "vs 16 thr"))
    for t in sorted(runs):
        v = runs[t]
        sp = "%.1f%%" % (100.0 * (max(v) - min(v)) / min(v)) if len(v) > 1 else "-"
        print("   %-9d %6d %26s %8s %10s"
              % (t, len(v), " ".join("%.0f" % x for x in v), sp,
                 "--" if t == 16 else "%+.1f%%" % (100.0 * (med[t] - med[16])
                                                   / med[16])))
    best = min(med, key=med.get)
    print("   on the CPU, 16 threads is a LOCAL PESSIMUM: %d threads is %.1f%% faster."
          % (best, 100.0 * (med[16] - med[best]) / med[16]))
    plateau = [t for t in med if med[t] <= med[best] * 1.03]
    print("   within 3%% of best: %s threads -- a PLATEAU, not a knife edge."
          % ",".join(str(t) for t in sorted(plateau)))
    print("   but 32 is %+.1f%% vs %d: 'use every hardware thread' is WRONG."
          % (100.0 * (med[32] - med[best]) / med[best], best))
    print("   RETRACTED: sec 12.2a and 12.3 projected this cell onto the anchor's")
    print("   prefill and credited --threads-batch with it.  No host expert runs")
    print("   on the CPU there.")
    shape = load_shape()
    if shape:
        ubatch(shape)
    copy_model(grid, med, best, shape)


def ubatch(g):
    """Threads vs ubatch for the host expert FFN: the --no-op-offload path."""
    toks = sorted(g[sorted(g)[0]])
    print("\n   per-token-per-layer host expert FFN cost, us (ubatch-shape.json):")
    print("   %-9s %s" % ("threads", "".join("%9d" % t for t in toks)))
    for th in sorted(g):
        print("   %-9d %s" % (th, "".join("%9.1f" % (g[th][t] / t)
                                          for t in toks)))
    f16, f28 = g[16][1024] / 1024.0, g[28][512] / 512.0
    print("   on the CPU, threads and ubatch reach the SAME compute floor near %.0f us"
          % min(f16, f28))
    print("   per token per layer (16 threads at ubatch 1024: %.1f; 28 at 512: %.1f;"
          % (f16, f28))
    print("   28 at 1024: %.1f), and they do not stack.  That holds for"
          % (g[28][1024] / 1024.0))
    print("   --no-op-offload only.  RETRACTED: 'the thread lever STRICTLY DOMINATES")
    print("   the ubatch lever'.  Stock copies where this computes, pays the copy once")
    print("   per ubatch, and so ubatch is the stock lever.")


def chunks(n_prompt, first, u):
    """Prompt chunks the server decodes at ubatch u: the logged first call,
    then n_batch chunks that break n_ubatch + 4 and 4 tokens before the end
    (server-context.cpp:3465-3484)."""
    out, pos = [first], first
    for stop in (n_prompt - 4 - u, n_prompt - 4, n_prompt):
        while pos < stop:
            out.append(min(N_BATCH, stop - pos))
            pos += out[-1]
    return out


def offloaded(chunk, u):
    """ubatches of one chunk that reach op offload (batch >= 32)."""
    sizes = [u] * (chunk // u) + ([chunk % u] if chunk % u else [])
    return sum(1 for s in sizes if s >= OFFLOAD_MIN_BATCH)


def ubatch_bytes(u):
    """Host bytes a full ubatch copies: the used experts' slices of every host
    bank, plus the host tensors copied whole."""
    used = USED_U512 if u == UBATCH else N_EXPERT
    return HOST_EXPS_B * used / N_EXPERT + HOST_WHOLE_B


def copy_model(grid, med, best, shape):
    with open(ANCHOR_LOG, encoding="utf-8", errors="replace") as f:
        text = f.read()
    prog = [(int(n), float(t)) for n, t in PROG.findall(text)]
    m = PEVAL.search(text)
    if len(prog) < 3 or m is None:
        sys.exit("%s: no prompt-processing cadence" % ANCHOR_LOG)
    total_s, n_prompt = float(m.group(1)) / 1000.0, int(m.group(2))
    first, first_s = prog[0]
    cum = [0] + [n for n, _ in prog] + [n_prompt]
    logged = [b - a for a, b in zip(cum, cum[1:])]
    if chunks(n_prompt, first, UBATCH) != logged:
        sys.exit("the server's chunk rule does not reproduce the logged chunks %s"
                 % logged)
    win_tok, win_s = prog[-1][0] - first, prog[-1][1] - first_s
    tail_s = total_s - prog[-1][1]

    def n_win(u):
        return sum(offloaded(c, u) for c in chunks(n_prompt, first, u)[1:-1])

    k = n_win(UBATCH)
    print("\n== STOCK PREFILL IS A COPY SCHEDULE (sec 12.5) ==")
    print("   at a batch of %d tokens or more, CUDA op offload (ggml-cuda.cu:5645)"
          % OFFLOAD_MIN_BATCH)
    print("   copies every host weight a ubatch uses to the GPU and runs it there.")
    print("   anchor cadence, %d-token prompt, n_batch %d, n_ubatch %d:"
          % (n_prompt, N_BATCH, UBATCH))
    print("   %8s %9s %9s %10s" % ("tokens", "seconds", "ubatches", "s/ubatch"))
    t_prev = 0.0
    for (_, t), c in zip(prog, logged):
        nu = offloaded(c, UBATCH)
        print("   %8d %9.2f %9d %10.3f%s" % (c, t - t_prev, nu, (t - t_prev) / nu,
                                           "  page-in, kept as logged"
                                           if t_prev == 0.0 else ""))
        t_prev = t
    print("   %8d %9.2f %9d %10s  below %d tokens: the CPU, kept as logged"
          % (logged[-1], tail_s, 0, "-", OFFLOAD_MIN_BATCH))
    print("   offloaded ubatches %d; after the first call %d (%d tokens, %.2f s)"
          % (sum(offloaded(c, UBATCH) for c in logged), k, win_tok, win_s))

    def C(g):
        return (win_s - win_tok / UBATCH * g) / k

    b512 = ubatch_bytes(UBATCH)
    w512 = b512 / (WIRE_GBPS * 1e9)
    print("\n   window = %d C + (%d/%d) g: C the copy per ubatch, g the device"
          % (k, win_tok, UBATCH))
    print("   compute per 512 tokens.  A full ubatch costs T = C + g:")
    for g, why in ((0.0, "no compute"), (G_LO, "ASSUMED low"),
                   (G_HI, "bound %d/%.1f" % (UBATCH, LM_TOK_S))):
        print("      g %.4f s %-18s C %.4f s  T %.4f s  copy rate %.2f GB/s"
              % (g, "(%s)" % why, C(g), C(g) + g, b512 / C(g) / 1e9))
    print("   bytes per ubatch: u512 %.2f GB (routed x %d/%d + %d B copied whole);"
          % (b512 / 1e9, USED_U512, N_EXPERT, HOST_WHOLE_B))
    print("   u1024 and up %.2f GB (all %d experts used)."
          % (ubatch_bytes(2 * UBATCH) / 1e9, N_EXPERT))
    print("   wire %.2f GB/s -> W = %.3f s per u512 ubatch; C >= W: %s"
          % (WIRE_GBPS, w512, "holds" if C(G_HI) >= w512 else "VIOLATED"))
    print("   the prompt copies ~%.2f TB; a layer-major pass copies the host banks"
          % (sum(offloaded(c, UBATCH) for c in logged) * b512 / 1e12))
    print("   once, %.2f GB." % (HOST_EXPS_B / 1e9))

    bank_eq = L_HOST                   # host bench cost scaled by bank bytes

    def copy_leg(u, g, pinned):
        c = (ubatch_bytes(u) / (WIRE_GBPS * 1e9) if pinned
             else C(g) * ubatch_bytes(u) / b512)
        return c + u / UBATCH * g, n_win(u) * c + win_tok / UBATCH * g

    def cpu_leg(u, us, g):
        return (bank_eq * us / 1e6 + u / UBATCH * g,
                win_tok / u * bank_eq * us / 1e6 + win_tok / UBATCH * g)

    levers = [(label, u, status, lambda g, u=u, p=pinned: copy_leg(u, g, p))
              for label, u, status, pinned in (
                  ("as recorded: -ub 512, mmap", 512, "logged", False),
                  ("(a) -ub 1024", 1024, "expected", False),
                  ("(a) -ub 2048", 2048, "expected", False),
                  ("(a) -ub 4096", 4096, "expected", False),
                  ("(b) --load-mode none", 512, "bound", True),
                  ("(a)+(b) -ub 2048 --load-mode none", 2048, "bound", True))]
    levers.append(("(c) --no-op-offload -tb %d" % best, 512, "optimistic",
                   lambda g: cpu_leg(512, med[best], g)))
    if 2048 in shape.get(28, {}):
        levers.append(("(c) --no-op-offload -tb 28 -ub 2048", 2048, "optimistic",
                       lambda g: cpu_leg(2048, shape[28][2048], g)))
    print("\n   levers, speedup over the recorded prefill, g from %.2f to %.4f s:"
          % (G_LO, G_HI))
    print("   %-37s %-10s %13s %13s" % ("", "status", "steady", "this prompt"))
    for label, u, status, leg in levers:
        steady, whole = [], []
        for g in (G_LO, G_HI):
            x, w = leg(g)
            steady.append((C(g) + g) * u / UBATCH / x)
            whole.append(total_s / (first_s + w + tail_s))
        print("   %-37s %-10s %5.2f-%-5.2fx %5.2f-%-5.2fx"
              % (label, status, min(steady), max(steady), min(whole), max(whole)))
    print("   steady is per full ubatch; this prompt keeps the first call's page-in")
    print("   (%.2f s) and the %d-token tail as logged, so it reads lower."
          % (first_s, logged[-1]))

    bank_s = [BANK_B / (b512 / C(g)) for g in (0.0, G_HI)]
    tg_ms = (grid[ANCHOR_THREADS][TOP_K] - TOP_K * D_DEV_US) / 1000.0
    print("   (a) the compute buffer grows with ubatch, and its VRAM at u1024/u2048")
    print("   under this fit is UNKNOWN (llama-fit-params, on the GPU).  Every bank it")
    print("   pushes to the host adds %.1f-%.1f ms (%.1f-%.1f%%) to a u512 ubatch and"
          % (1000 * bank_s[1], 1000 * bank_s[0], 100 * bank_s[1] / (C(G_HI) + G_HI),
             100 * bank_s[0] / C(0.0)))
    print("   %.3f ms (%.2f%%) to a TG token at %d threads; a bank moved to the GPU"
          % (tg_ms, 100 * tg_ms / ANCHOR_MS, ANCHOR_THREADS))
    print("   saves the same.")
    print("   (b) C >= W makes it a ceiling, and it page-locks the host model buffer.")
    print("   (c) the bench gives the expert FFN the whole box: block 0's dense ops,")
    print("   the activation traffic and the Q6_K heavy downs are left out, and cost")
    print("   is scaled by bank bytes (%.2f banks).  Dominated by (a) above u512."
          % bank_eq)
    print("   --threads-batch moves no host expert under op offload: at a batch of")
    print("   %d or more none runs on the CPU.  It keeps the embedding lookup"
          % OFFLOAD_MIN_BATCH)
    print("   (GET_ROWS never offloads, ggml-cuda.cu:5458) and block 0's state-bound")
    print("   ops (expected small, unmeasured; a -tb arm checks).")

    stock = [UBATCH / (C(g) + g) for g in (G_LO, G_HI)]
    at2048 = [2048 / copy_leg(2048, g, False)[0] for g in (G_LO, G_HI)]
    print("\n   layer-major margin: %.1f tok/s against stock's steady %.1f = %.2fx."
          % (LM_TOK_S, min(stock), LM_TOK_S / min(stock)))
    print("   against stock at -ub 2048 it would be %.2f-%.2fx (unproven): sec 9.1's"
          % (LM_TOK_S / max(at2048), LM_TOK_S / min(at2048)))
    print("   7x was measured against untuned stock.")


if __name__ == "__main__":
    main()
