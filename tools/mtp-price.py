#!/usr/bin/env python3
"""Price stock MTP speculative decoding against the no-MTP TG anchor.

Host-only: reads two server logs, a host expert bench and recorded TG routes.

sec 2 sets 32.44 tok/s (2026-09-10, MTP draft 5) beside 40.96 tok/s
(2026-09-29, no MTP) and concludes MTP "was likely costing" the baseline.
The two runs differ in four ways at once:

  context    the MTP run decoded after a 372-token prompt, the anchor after
             52,819 tokens;
  binary     build 10876 against one 16 builds newer;
  placement  with MTP loaded the fitter counts the nextn block
             (common/fit.cpp:139-142) and reserves the draft context (f16 KV
             at 262k) before fitting the target (server-context.cpp:1030-1076):
             it logged n_layer=34, so blocks 0..7 ran wholly on the CPU;
             the anchor has one whole block there;
  draft      n_max 5, and every round drafted all five.

So that comparison cannot price the lever.  The MTP log does measure the
per-token draft cost, the round cost and the acceptance at every draft
position; this script re-derives all three from it.

What no log measures is the VERIFY batch on the anchor's placement.  A verify
of n = d+1 tokens is below the op-offload threshold of 32 (ggml-cuda.cu:5645),
so the host experts run on the CPU over the UNION of the batch's routes.  Two
measurements price that:

  routes  rowset_fetch union_ids of recorded TG runs, one event per layer per
          generated token.  U(n) is the mean number of distinct experts per
          host layer over n consecutive tokens.  Accepted drafts ARE the next
          tokens, so their routes are these; rejected ones are unrecorded.
  bench   build/expert-cpu-bench/verify-shape.json, the real op chain on the
          real Q4_K slabs with iteration-disjoint expert sets.  A cell of k
          experts over n tokens touches U = k+n-1 experts in k*n pairs (ids
          are (base + slot + token) % pool, expert-cpu-bench.cpp:651-657).
          One-token cells fit h1(U) = a + b*U; every n-token cell's excess
          over h1(U) fits F(n) + p(n) * (k*n - U).  Both are median-of-slopes
          lines, so one outlier cell cannot move them.

  host(n)     = h1(U(n)) + F(n) + p(n) * (8n - U(n))
  T_verify(n) = T1 + L_host * (host(n) - h1(8)) + (n - 1) * delta
  T_round(d)  = d * c_draft + T_verify(d + 1)
  N(d)        = 1 + sum(acc_pos[:d])           tokens delivered per round

delta is everything else an extra verify token costs: attention, GPU dense
and expert reads, the CPU block, sampling.  Nothing recorded measures it on
this placement, so the script prints the delta at which each draft length
breaks even.  The GPU A/B decides.
"""

import argparse
import glob
import itertools
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ANCHOR_LOG = os.path.join(ROOT, "docs", "llamacpp 8gb vram base.md")
MTP_LOG = os.path.join(ROOT, "out", "llamacpp-baseline.log")
BENCH = os.path.join(ROOT, "build", "expert-cpu-bench", "verify-shape.json")
TRACES = os.path.join(ROOT, "build", "llama-cuda", "**", "*.jsonl")
TOP_K = 8
THREADS = 16                  # anchor log line 24: n_threads = 16
N_LAYER_ALL = 41              # block_count; the loader counts nextn (llama-model.cpp:1354)
# Host routed-expert bytes at the anchor fit over one Q4_K bank:
# tools/bank-bytes.py --ngl 41 --n-part 33 --overflow-type 3 (loader actual)
# puts the banks of layers 0 and 9..39 and blk.8's down on the host.
L_HOST = 14854127616 / 452984832
HOST_LAYERS = [0] + list(range(9, 40))   # whole host banks; blk.8's down is left out of U
OFFLOAD_MIN_BATCH = 32        # ggml-cuda.cu:5645
PREFIX = 32                   # generated ids that identify a stream

EVAL = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
STATS = re.compile(r"#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+).*?"
                   r"#gen tokens =\s*(\d+), #acc tokens =\s*(\d+).*?"
                   r"#acc rate/pos = \(([^)]*)\), "
                   r"dur\(b,g,a\) =\s*([\d.]+),\s*([\d.]+),\s*([\d.]+) ms")
FIT_TRY = re.compile(r"id=0, n_layer=\s*(\d+), n_part=\s*(\d+), overflow_type=(\d+)")
FIT_SET = re.compile(r"set ngl_per_device\[0\]")
OVERFLOW = {1: "ATTN", 2: "UP", 3: "GATE", 4: "MOE"}   # common/fit.cpp:18-22


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def grep(path, rx, what):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                return m
    sys.exit("%s: no %s line" % (rel(path), what))


def fitted(path):
    """The fitter's final (n_layer, n_part, overflow_type): the candidate
    tried last before each 'set ngl_per_device[0]' line, last one wins."""
    tried = chosen = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = FIT_TRY.search(line)
            if m:
                tried = tuple(int(x) for x in m.groups())
            elif FIT_SET.search(line) and tried:
                chosen = tried
    if chosen is None:
        sys.exit("%s: no fitter decision" % rel(path))
    return chosen


def theil_sen(pts):
    """(intercept, slope) of the median-of-slopes line through (x, y) points."""
    b = statistics.median((y2 - y1) / (x2 - x1)
                          for (x1, y1), (x2, y2) in itertools.combinations(sorted(pts), 2)
                          if x2 != x1)
    return statistics.median(y - b * x for x, y in pts), b


def bench():
    """Fitted host expert cost at THREADS: h1 = (a, b) for one token over U
    experts, fp = {n: (F, p)} for the excess of n tokens, and the cells the
    fits leave more than 15% off."""
    with open(BENCH) as f:
        data = json.load(f)
    cells = {(c["experts_used"], c["n_tokens"]): c["median_us"]
             for c in data["cells"] if c["threads"] == THREADS}
    widest = max(k + n - 1 for k, n in cells)
    if data["id_stride"] < widest or data["pool_experts"] < 256:
        sys.exit("%s: id-stride %d < widest union %d (or pool %d) lets an iteration "
                 "reuse the last one's experts" % (rel(BENCH), data["id_stride"], widest,
                                                   data["pool_experts"]))
    ns = sorted({n for _, n in cells if n > 1})
    if len([k for k, n in cells if n == 1]) < 3 or TOP_K not in {k for k, _ in cells} or not ns:
        sys.exit("%s: need %d-thread cells over several k at 1 token and at n > 1"
                 % (rel(BENCH), THREADS))
    a, b = theil_sen([(k, us) for (k, n), us in cells.items() if n == 1])
    fp = {1: (0.0, 0.0)}
    for n in ns:
        fp[n] = theil_sen([(k * n - (k + n - 1), us - a - b * (k + n - 1))
                           for (k, n2), us in cells.items() if n2 == n])
    model = lambda k, n: a + b * (k + n - 1) + fp[n][0] + fp[n][1] * (k * n - (k + n - 1))
    off = sorted((k, n, us, model(k, n)) for (k, n), us in cells.items()
                 if abs(us / model(k, n) - 1) > 0.15)
    return (a, b), fp, off, widest, cells[(TOP_K, 1)]


def trace_routes(p):
    """(generated ids, {layer: [(t_us, expert set)]}), or a reason the trace
    is not one route per layer per generated token."""
    seq, ids = defaultdict(list), None
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"event":"rowset_fetch"' in line:
                e = json.loads(line)
                if e.get("phase") != 1:
                    continue
                s = frozenset(int(x) for x in e["union_ids"].split(","))
                if e["n_tokens"] != 1 or e["top_k"] != TOP_K or len(s) != TOP_K:
                    return None, "a TG fetch is not one token's %d experts" % TOP_K
                seq[e["layer"]].append((e["t_us"], s))
            elif '"event":"tg_tokens"' in line:
                ids = json.loads(line)["ids"]
    if not seq:
        return None, "no TG rowset_fetch"
    if ids is None or len(ids) < PREFIX:
        return None, "fewer than %d generated ids" % PREFIX
    if any(len(seq.get(il, ())) != len(ids) for il in HOST_LAYERS):
        return None, "fetches per layer do not match generated ids"
    return ids, seq


def streams(paths):
    """{first PREFIX ids: (path, tokens, {layer: [expert sets, token order]})},
    keeping the longest trace of each distinct generated stream, and the
    skipped traces counted by reason."""
    best, skipped = {}, defaultdict(int)
    for p in paths:
        ids, seq = trace_routes(p)
        if ids is None:
            skipped[seq] += 1
            continue
        key = tuple(ids[:PREFIX])
        if key not in best or len(ids) > best[key][1]:
            best[key] = (p, len(ids), {il: [s for _, s in sorted(seq[il], key=lambda r: r[0])]
                                       for il in HOST_LAYERS})
    return best, skipped


def union_curve(layers, n_max):
    out = {}
    for n in range(1, n_max + 1):
        tot = cnt = 0
        for sets in layers.values():
            for t in range(len(sets) - n + 1):
                tot += len(frozenset().union(*sets[t:t + n]))
                cnt += 1
        out[n] = tot / cnt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*",
                    help="TG route traces (default: every *.jsonl under build/llama-cuda)")
    ap.add_argument("--extra-banks", type=float, default=2.0,
                    help="routed banks the MTP VRAM reserve pushes to the host")
    a = ap.parse_args()

    m = grep(ANCHOR_LOG, EVAL, "eval time")
    t1 = float(m.group(1)) / int(m.group(2))
    m = grep(MTP_LOG, EVAL, "eval time")
    mtp_ms, mtp_tok = float(m.group(1)), int(m.group(2))
    s = grep(MTP_LOG, STATS, "draft-mtp statistics")
    rounds, gen, acc = int(s.group(2)), int(s.group(4)), int(s.group(5))
    acc_pos = [float(x) for x in s.group(6).split(",")]
    dur_g, dur_a = float(s.group(8)), float(s.group(9))
    d_max = len(acc_pos)
    if gen != d_max * rounds:
        sys.exit("%s: drafts were cut short (%d tokens over %d rounds); the "
                 "per-position rates are then not a d-truncation" % (rel(MTP_LOG), gen, rounds))
    c_draft = dur_g / gen
    verify = (mtp_ms - dur_g - dur_a) / rounds

    print("MEASURED, 2026-09-10 MTP run (%s)" % rel(MTP_LOG))
    print("   rounds %d, drafted %d (%d each), accepted %d, delivered %d"
          % (rounds, gen, d_max, acc, mtp_tok))
    print("   acceptance by draft position: %s" % ", ".join("%.3f" % x for x in acc_pos))
    print("   draft  %.3f ms per drafted token (dur g %.1f ms / %d)" % (c_draft, dur_g, gen))
    print("   round  %.2f ms, of which verify of %d tokens + bookkeeping %.2f ms (%.2f x T1)"
          % (mtp_ms / rounds, d_max + 1, verify, verify / t1))
    for label, path, mtp in (("MTP run", MTP_LOG, True), ("anchor ", ANCHOR_LOG, False)):
        ngl, n_part, ot = fitted(path)
        cpu = N_LAYER_ALL + 1 - ngl          # loader i_gpu_start
        print("   %s fitter: n_layer %d, n_part %d, overflow %s -> %s whole on the CPU%s"
              % (label, ngl, n_part, OVERFLOW[ot],
                 "no block" if cpu <= 0 else "block 0" if cpu == 1 else "blocks 0..%d" % (cpu - 1),
                 "" if mtp else " (fitter meant none: tenth finding)"))
    print("   anchor (no MTP, 52.8k context, newer build): T1 = %.2f ms/token" % t1)
    print("   the MTP round is therefore not a verify cost at the anchor placement.")

    (h_a, h_b), fp, off, widest, cell8 = bench()
    h1 = lambda u: h_a + h_b * u
    h8 = h1(TOP_K)
    print("\nMEASURED host expert cost, %d threads (%s), median-of-slopes fits:"
          % (THREADS, rel(BENCH)))
    print("   1 token over U experts: %.1f + %.2f*U us  (U=8: %.1f us, the k=8 cell %.1f)"
          % (h_a, h_b, h8, cell8))
    for n in sorted(fp)[1:]:
        print("   %d tokens: + %.1f us + %.2f us per extra token-expert pair" % (n, *fp[n]))
    for k, n, us, mod in off:
        print("   cell k=%d n=%d measured %.1f us, fit %.1f: left out as an outlier" % (k, n, us, mod))
    print("   at U=8 the host leg is %.2f ms of T1's %.2f (x %.2f bank-equivalents)"
          % (L_HOST * h8 / 1000, t1, L_HOST))

    paths = a.traces or sorted(glob.glob(TRACES, recursive=True))
    found, skipped = streams(paths)
    if not found:
        sys.exit("no TG route trace with at least %d generated tokens" % PREFIX)
    print("\nMEASURED route unions: mean distinct experts per host layer over n "
          "consecutive generated tokens")
    print("   %d distinct streams in %d traces; skipped: %s"
          % (len(found), len(paths), "; ".join("%d %s" % (c, r) for r, c in sorted(skipped.items()))))
    print("   %-52s %6s  %s" % ("longest trace of the stream", "tokens",
                                "".join("%7s" % ("n=%d" % n) for n in range(1, d_max + 2))))
    curves = []
    for p, n_tok, layers in sorted(found.values(), key=lambda v: v[0]):
        u = union_curve(layers, d_max + 1)
        curves.append(u)
        print("   %-52s %6d  %s" % (rel(p), n_tok, "".join("%7.2f" % u[n] for n in sorted(u))))
    rng = lambda f: (min(f(u) for u in curves), max(f(u) for u in curves))

    def pair_fit(n):
        """(F, p) at n tokens, linear between the bench's token counts."""
        ks = sorted(fp)
        if n > ks[-1]:
            sys.exit("verify of %d tokens: the bench stops at %d; rerun it wider" % (n, ks[-1]))
        lo = max(k for k in ks if k <= n)
        hi = min(k for k in ks if k >= n)
        w = 0.0 if hi == lo else (n - lo) / (hi - lo)
        return tuple(fp[lo][i] + (fp[hi][i] - fp[lo][i]) * w for i in (0, 1))

    def host(n, u):
        f, p = pair_fit(n)
        return h1(u) + f + p * (TOP_K * n - u)

    xb = a.extra_banks
    print("\nEXPECTED at the anchor placement, delta = 0 (ranges span the streams):")
    print("   %-5s %6s %11s %13s %13s %13s %14s %15s"
          % ("draft", "N(d)", "U(d+1)", "host adds ms", "ms/token", "speedup",
             "+%g host banks" % xb, "break-even delta"))
    for d in range(1, d_max + 1):
        n, tok = d + 1, 1 + sum(acc_pos[:d])
        add = lambda u: L_HOST * (host(n, u[n]) - h8) / 1000
        ms = lambda u: (d * c_draft + t1 + add(u)) / tok
        # Each bank the reserve evicts costs the verify round host(n), not h1(8).
        ms_x = lambda u: ms(u) + xb * host(n, u[n]) / 1000 / tok
        be = lambda u: (tok * t1 - d * c_draft - t1 - add(u)) / d
        r = [rng(lambda u: u[n]), rng(add), rng(ms), rng(lambda u: t1 / ms(u)),
             rng(lambda u: t1 / ms_x(u)), rng(be)]
        print("   %-5d %6.3f %5.2f-%-5.2f %6.2f-%-6.2f %6.2f-%-6.2f %5.2f-%-5.2fx "
              "%5.2f-%-5.2fx %6.2f-%-6.2f ms"
              % (d, tok, r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1],
                 r[3][0], r[3][1], r[4][0], r[4][1], r[5][0], r[5][1]))
    u_top = max(u[d_max + 1] for u in curves)
    if u_top > widest:
        print("   U up to %.1f extends the bench's widest union %d on its fitted line"
              % (u_top, widest))
    print("   N(d) takes the 2026-09-10 acceptance (a 372-token prompt); at 52.8k and on")
    print("   another prompt it is unmeasured.  c_draft is that run's, with the nextn")
    print("   block's experts on the host.  Every verify of up to %d tokens stays below"
          % (OFFLOAD_MIN_BATCH - 1))
    print("   the offload threshold, so no copy term enters TG.  Loading MTP also")
    print("   realigns fitter and loader (common.cpp:1689 sets load_mtp), which the A/B")
    print("   must control for with the block-0 fix arm.")


if __name__ == "__main__":
    main()
