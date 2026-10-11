#!/usr/bin/env python3
"""Can the experts that MISS the VRAM cache at layer L+1 be predicted from what is known before layer L+1's router runs? (host-only, from a routing trace)

Replays the engine's cache policy (EMA + promotion budget, see moe-policy-sim.py) to find, per token and layer, which selected experts missed. Then scores
predictors that see only information available earlier in the same token or in earlier tokens:
  prev     the experts layer L+1 missed at the previous token
  coact    learned co-activation: score(e) = sum over a in S(L) of counts[L][a][e], counts learned online from earlier tokens (decayed); only uncached e qualify
  both     union of the best of each (n/2 + n/2)
For n = 1..4 predicted experts it reports precision (predicted & missed) / predicted and coverage (predicted & missed) / missed. A predictor that cannot beat
a few tens of percent precision is useless for prefetching, whichever resource (PCIe DMA into spare slots, or the CPU's cache) would consume the prediction.

  python tools/moe-copred-sim.py routing.bin --prior build/llama-cuda/aer-freq-train.txt --slots 42 [--lambda 0.97] [--budget 16]

It measures; it claims nothing.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np


def load_policy_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("moe_policy_sim", os.path.join(here, "moe-policy-sim.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def miss_masks(ids, n_layer, n_expert, C, prior, lam, budget):
    """Replay the EMA policy; return bool array [tokens, layers, n_used] = selected expert was NOT in the published cache at that token."""
    n_tok, _, n_used = ids.shape
    fmax = float(prior.max())
    tie = 1e-6 * prior / fmax
    ema = np.zeros((n_layer, n_expert))
    cached = np.zeros((n_layer, n_expert), dtype=bool)
    pub = np.zeros((n_layer, n_expert), dtype=bool)
    for l in range(n_layer):
        order = np.argsort(-prior[l], kind="stable")[:C]
        cached[l, order] = True
        pub[l, order] = True
    miss = np.zeros(ids.shape, dtype=bool)
    pend = []
    for t in range(n_tok):
        sel = ids[t]
        miss[t] = ~pub[np.arange(n_layer)[:, None], sel]
        ema *= lam
        for l in range(n_layer):
            ema[l, sel[l]] += (1.0 - lam)
        for l, e in pend:
            pub[l, e] = True
        pend = []
        if budget == 0:
            continue
        pairs = []
        for l in range(n_layer):
            sc = ema[l] + tie[l]
            unc = np.where(~cached[l])[0]
            if len(unc) == 0:
                continue
            cs = unc[np.argsort(-sc[unc], kind="stable")[:8]]
            cac = np.where(cached[l])[0]
            vs = cac[np.argsort(sc[cac], kind="stable")[:8]]
            for k in range(min(len(cs), len(vs))):
                gain = sc[cs[k]] - sc[vs[k]]
                if not gain > 0.0:
                    break
                pairs.append((gain, l, int(cs[k]), int(vs[k])))
        if budget > 0 and len(pairs) > budget:
            pairs.sort(key=lambda p: -p[0])
            pairs = pairs[:budget]
        for gain, l, c, v in pairs:
            pub[l, v] = False
            cached[l, v] = False
            cached[l, c] = True
            pend.append((l, c))
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--slots", type=int, default=42)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.97)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--decay", type=float, default=0.999, help="decay of the co-activation counts per token")
    a = ap.parse_args()
    pol = load_policy_module()
    ids, n_layer, n_used, n_expert = pol.read_trace(a.trace)
    prior = np.array([[int(x) for x in line.split()] for line in open(a.prior) if line.strip()], dtype=np.int64)
    miss = miss_masks(ids, n_layer, n_expert, a.slots, prior, a.lam, a.budget)
    n_tok = ids.shape[0]
    print("trace %s: %d tokens, C=%d, mean misses per token-layer %.2f (hit rate %.4f)" % (
        a.trace, n_tok, a.slots, miss.sum() / (n_tok * n_layer), 1.0 - miss.mean()))

    # the cache content seen by the predictors: approximate "uncached" by the policy-independent rule "not among the C highest prior counts"
    # is wrong for a dynamic cache, so predictors score every expert and the evaluation counts only truly missed ones as hits of the prediction
    co = np.zeros((n_layer - 1, n_expert, n_expert), dtype=np.float32)   # co[l][a][b]: a selected at l, b selected at l+1
    nmax = 4
    stats = {pn: np.zeros((nmax + 1, 3)) for pn in ("prev", "coact", "both")}   # per n: [predicted, correct, tokens]
    total_miss = 0
    for t in range(1, n_tok):
        for l in range(1, n_layer):
            missed = set(int(e) for e, m in zip(ids[t, l], miss[t, l]) if m)
            total_miss += len(missed)   # layers without a miss still cost a (wasted) prediction below
            prev_pred = [int(e) for e, m in zip(ids[t - 1, l], miss[t - 1, l]) if m]
            score = co[l - 1][ids[t, l - 1]].sum(axis=0)
            order = np.argsort(-score, kind="stable")
            co_pred = [int(e) for e in order[:nmax] if score[e] > 0]
            for n in range(1, nmax + 1):
                p1 = prev_pred[:n]
                p2 = co_pred[:n]
                p3 = list(dict.fromkeys(co_pred[: (n + 1) // 2] + prev_pred[: n // 2] + co_pred[(n + 1) // 2:] + prev_pred[n // 2:]))[:n]
                for name, pr in (("prev", p1), ("coact", p2), ("both", p3)):
                    stats[name][n, 0] += len(pr)
                    stats[name][n, 1] += len(set(pr) & missed)
        # learn from token t after predicting it
        co *= a.decay
        for l in range(n_layer - 1):
            for x in ids[t, l]:
                co[l, int(x), ids[t, l + 1]] += 1.0
    print("misses scored: %d" % total_miss)
    print("%-6s %3s %10s %10s" % ("pred", "n", "precision", "coverage"))
    for name in ("prev", "coact", "both"):
        for n in range(1, nmax + 1):
            pr, ok = stats[name][n, 0], stats[name][n, 1]
            print("%-6s %3d %9.1f%% %9.1f%%" % (name, n, 100.0 * ok / max(pr, 1.0), 100.0 * ok / max(total_miss, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
