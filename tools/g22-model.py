#!/usr/bin/env python3
"""G22: schedule model for the prefill of a prompt on one fast unit (the GPU, whose expert banks must be copied over a slow link) and one slow unit (the CPU, whose expert banks are
already in RAM). Host-only. Version 3 (2026-10-12): per-layer-type GPU times from the traces of the layer-major sweep (`tools/g22-trace-sim.py`), the copy protocol of the controller.

Measured inputs (build g40; sweep sessions g46-<N>; tools/moe-cpu-batch-bench.cpp):
  g_rec(N), g_att(N)  the GPU time of one layer WITH a resident bank, median over the 30 recurrent layers and the 10 attention layers (every fourth layer, il % 4 == 3) of the traces.
                      An attention layer costs about 32 ms at ANY prompt length (its fixed cost, cause not established) and 39 ms at 2,047 tokens; a recurrent layer 0.7 ms at 3 tokens
                      and 9.3 ms at 2,047. (The first layer of a cold process also pays ~190 ms of one-time kernel loading; it is a harness warm-up and is not in the model.)
  p(N)                the CPU time of ONE layer's MoE FFN (generic ggml-cpu mul_mat_id over Q4_K banks, 16 threads; bench table)
  c                   the copy time of one 432 MiB bank over PCIe Gen4 x8 from the registered mmap, as fitted from the traces: 33 ms at 3 tokens, 35-37 ms at 63-1,023, 41 ms at 2,047
                      (14.2 GB/s from a pinned buffer = 32 ms)

A layer is executed either as STREAMED (its bank is copied into a slot, the GPU runs the whole layer: g_type) or on the CPU LEG (the GPU runs the dense part, the CPU the expert FFN from the
host bank: no copy): dense_type + p(N) + the hidden-state round trip. The dense part is g_type minus `--efrac` x g_rec (the share of a recurrent layer's GPU time that is its MoE FFN).
The controller's protocol is modelled exactly: the slot of layer il is il % S; the first streamed layer of each slot is submitted at t = 0 (in layer order); the successor of a slot (the next
streamed layer with the same il % S) is submitted when the layer that holds the slot has ended; one copy engine serves the submissions in order and takes c ms per bank (x `--copy-slow`);
a CPU layer runs `--dram` times slower for the part that overlaps a copy (1.163 measured by `dram-share`). The assignment is SEARCHED (local search from 25 starts).

    python tools/g22-model.py [--c 37] [--slots 2,3,4,6,8] [--pre 0,2] [--dram 1.163] [--efrac 0.5] [--warmup-ms 0] [--show-plan] [--ns 63,256,...]
"""
import argparse
import math
import random

N_TOK = [3, 10, 14, 29, 63, 125, 256, 511, 1023, 2047]
STOCK_MS = [92.0, 159.1, 209.5, 343.1, 727.1, 904.9, 1111.0, 1248.2, 2310.1, 4483.1]       # llama-completion `prompt eval time` (it warms up before the timer)
LM_MS = [1400.5, 1551.2, 1434.6, 1755.9, 1694.3, 1552.5, 1659.2, 1836.0, 1689.5, 1900.8]     # g46 layer-major, ubatch 2048, 2 slots, COLD process (includes ~190 ms of one-time kernel loading)
G_REC_MS = [0.72, 0.88, 0.90, 1.15, 1.38, 2.03, 2.70, 3.77, 5.40, 9.30]                       # median GPU time of a recurrent layer with a resident bank
G_ATT_MS = [32.5, 32.65, 32.65, 32.85, 33.1, 33.65, 34.1, 34.9, 35.9, 38.9]                  # median GPU time of an attention layer (il % 4 == 3)
C_MS = [33.0, 36.0, 33.5, 37.5, 35.0, 34.0, 36.0, 39.0, 37.0, 41.0]                          # copy time of one bank, fitted from the traces (noisy: 33-44 ms between reps)
P_N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]                               # moe-cpu-batch-bench, median ms of ONE layer's MoE FFN, uniform routing
P_MS = [0.241, 0.540, 1.030, 1.935, 3.721, 6.534, 10.512, 17.490, 32.244, 50.100, 97.932, 189.687, 379.4]
SETUP_MS = 0.15            # per-layer launch and bookkeeping next to the compute of a resident layer
SWITCH_MS = 0.3            # synchronisation between the GPU and the CPU leg of a layer
EFRAC = 0.5


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1] * x / xs[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (math.log(x) - math.log(xs[i - 1])) / (math.log(xs[i]) - math.log(xs[i - 1]))
            return ys[i - 1] + f * (ys[i] - ys[i - 1])


def layer_costs(n, efrac=EFRAC, layers=40):
    """Per-layer GPU time of a STREAMED layer (g) and of a CPU-LEG layer (tc), in ms, for a prompt of n tokens."""
    g_rec = interp(N_TOK, G_REC_MS, n) + SETUP_MS
    g_att = interp(N_TOK, G_ATT_MS, n) + SETUP_MS
    e = efrac * interp(N_TOK, G_REC_MS, n)                          # the MoE FFN of the GPU layer: what the CPU leg takes over
    p = interp(P_N, P_MS, n)
    xfer = n * 2048 * 4 * 2 / 14.0e9 * 1000.0                      # f32 hidden state to the host and back over the link, ms
    g = [g_att if il % 4 == 3 else g_rec for il in range(layers)]
    tc = [gi - e + p + SWITCH_MS + xfer for gi in g]
    return g, tc


def evaluate(seq, g, tc, c, slots, pre, dram):
    """Time of one assignment `seq` (a string of 'C' and 'G', one letter per layer). `c` is the copy time already scaled by the copy slowdown; `dram` slows the CPU layers while a copy runs.
    Returns 1e30 for a plan that is not executable (two of the `pre` resident banks would share a slot)."""
    gpu = [il for il, k in enumerate(seq) if k == 'G']
    n_gpu = len(gpu)
    occ = []                # occupant: the previous streamed layer of the same slot, as an index into `gpu`, or None
    last = {}
    for j, il in enumerate(gpu):
        occ.append(last.get(il % slots))
        last[il % slots] = j
    for j in range(min(pre, n_gpu)):
        if occ[j] is not None:
            return 1e30
    t = 0.0
    loads = []              # (start, end) of the copies, in order
    ready = {}
    for j in range(min(pre, n_gpu)):
        ready[j] = 0.0
    next_k = min(pre, n_gpu)
    copy_free = 0.0
    consume_end = {}
    gpu_idx = 0
    for il, kind in enumerate(seq):
        while next_k < n_gpu:                       # submit every copy whose slot is free
            o = occ[next_k]
            if o is not None and o not in consume_end:
                break
            start = max(copy_free, consume_end[o] if o is not None else 0.0)
            end = start + c
            loads.append((start, end))
            ready[next_k] = end
            copy_free = end
            next_k += 1
        if kind == 'C':
            rem, cur = tc[il], t
            for (s, e) in loads:
                if e <= cur:
                    continue
                if s > cur:
                    run = min(rem, s - cur)
                    cur += run
                    rem -= run
                    if rem <= 1e-12:
                        break
                a = max(s, cur)
                cap = (e - a) / dram
                if cap >= rem:
                    cur = a + rem * dram
                    rem = 0.0
                    break
                rem -= cap
                cur = e
            if rem > 1e-12:
                cur += rem
            t = cur
        else:
            t = max(t, ready[gpu_idx]) + g[il]
            consume_end[gpu_idx] = t
            gpu_idx += 1
    return t


def optimize(n, c, slots, pre, dram, copy_slow=1.0, efrac=EFRAC, layers=40, seed=1):
    g, tc = layer_costs(n, efrac, layers)
    rng = random.Random(seed)

    def ev(s):
        return evaluate(s, g, tc, c * copy_slow, slots, pre, dram)

    starts = ['C' * layers, 'G' * layers, ('CG' * layers)[:layers], ('GC' * layers)[:layers]]
    for k in (2, 4, 8, 12, 16, 20, 24, 28, 32):
        starts.append('G' * k + 'C' * (layers - k))
        starts.append('C' * (layers - k) + 'G' * k)
        s = ['C'] * layers
        for i in range(k):
            s[int((i + 0.5) * layers / k)] = 'G'
        starts.append(''.join(s))
    best, best_t = None, 1e30
    for st in starts:
        cur, cur_t = list(st), ev(st)
        improved = True
        while improved:
            improved = False
            for i in range(layers):
                cand = cur[:]
                cand[i] = 'G' if cand[i] == 'C' else 'C'
                tt = ev(''.join(cand))
                if tt < cur_t - 1e-9:
                    cur, cur_t, improved = cand, tt, True
            for _ in range(80):
                i, j = rng.sample(range(layers), 2)
                cand = cur[:]
                cand[i], cand[j] = cand[j], cand[i]
                tt = ev(''.join(cand))
                if tt < cur_t - 1e-9:
                    cur, cur_t, improved = cand, tt, True
        if cur_t < best_t:
            best, best_t = ''.join(cur), cur_t
    return best_t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=None, help="copy time per bank in ms (default: the traced value for each length)")
    ap.add_argument("--slots", default="2,3,4,6,8")
    ap.add_argument("--pre", default="0,2")
    ap.add_argument("--dram", type=float, default=1.163)
    ap.add_argument("--copy-slow", type=float, default=1.0)
    ap.add_argument("--efrac", type=float, default=EFRAC)
    ap.add_argument("--warmup-ms", type=float, default=0.0, help="added to every modelled time (use ~190 to compare with the cold-process layer-major column)")
    ap.add_argument("--show-plan", action="store_true")
    ap.add_argument("--ns", default="3,8,16,32,64,128,256,512,1024,2048")
    a = ap.parse_args()
    slots_list = [int(x) for x in a.slots.split(",")]
    pre_list = [int(x) for x in a.pre.split(",")]
    print("v3 model: per-type GPU times, copy time %s, CPU layers x%.3f while a copy runs, MoE share of a recurrent layer %.2f, +%.0f ms warm-up; ms; (n) = layers streamed" % (
        ("%.0f ms" % a.c) if a.c else "per length (33-41 ms)", a.dram, a.efrac, a.warmup_ms))
    combos = [(s, p) for s in slots_list for p in pre_list if p <= s]
    head = "%6s | %7s %8s %6s %6s |" % ("N", "stock", "LM(meas)", "all-GPU", "all-CPU")
    for s, p in combos:
        head += "  S%d/K%d    " % (s, p)
    head += "| best hybrid vs min(stock, all-GPU model)"
    print(head)
    for n in [int(x) for x in a.ns.split(",")]:
        stock = interp(N_TOK, STOCK_MS, n)
        lm_meas = interp(N_TOK, LM_MS, n)
        c = a.c if a.c else interp(N_TOK, C_MS, n)
        g, tc = layer_costs(n, a.efrac)
        allg = min(evaluate('G' * 40, g, tc, c * a.copy_slow, s, 0, a.dram) for s in slots_list) + a.warmup_ms
        allc = evaluate('C' * 40, g, tc, c * a.copy_slow, 2, 0, a.dram) + a.warmup_ms
        line = "%6d | %7.0f %8.0f %6.0f %6.0f |" % (n, stock, lm_meas, allg, allc)
        best, best_cfg = 1e30, None
        plans = {}
        for s, p in combos:
            t, seq = optimize(n, c, s, p, a.dram, a.copy_slow, a.efrac)
            t += a.warmup_ms
            plans[(s, p)] = seq
            if t < best:
                best, best_cfg = t, (s, p)
            line += " %6.0f(%2d) " % (t, seq.count('G'))
        line += "| %.2fx at S%d/K%d" % (min(stock, allg) / best, best_cfg[0], best_cfg[1])
        print(line)
        if a.show_plan:
            pl = plans[best_cfg]
            print("        plan " + pl + "   cpu layers: " + ",".join(str(i) for i, k in enumerate(pl) if k == 'C'))


if __name__ == "__main__":
    main()
