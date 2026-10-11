#!/usr/bin/env python3
"""How much more expert traffic does a T-token verification step need than a single-token step? (host-only, from a routing trace)

For windows of T consecutive tokens the cache state is the same for every token of the window (they run in one graph), so the host serves the UNION of the
tokens' missed experts once: u(T) = distinct missed experts per window / (T x mean misses of one token). u = 1 means no sharing at all (traffic grows
linearly with T), u = 1/T means the tokens miss the same experts. Also reports the same ratio for distinct selected experts, cache or not.

  python tools/moe-mt-union.py routing.bin --prior build/llama-cuda/aer-freq-train.txt --slots 42

It measures; it claims nothing.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np


def load(name, fname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--slots", type=int, default=42)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.97)
    ap.add_argument("--budget", type=int, default=16)
    a = ap.parse_args()
    pol = load("moe_policy_sim", "moe-policy-sim.py")
    cop = load("moe_copred_sim", "moe-copred-sim.py")
    ids, n_layer, n_used, n_expert = pol.read_trace(a.trace)
    prior = np.array([[int(x) for x in line.split()] for line in open(a.prior) if line.strip()], dtype=np.int64)
    miss = cop.miss_masks(ids, n_layer, n_expert, a.slots, prior, a.lam, a.budget)
    n_tok = ids.shape[0]
    print("trace %s: %d tokens, C=%d, mean misses per token-layer %.3f" % (a.trace, n_tok, a.slots, miss.sum() / (n_tok * n_layer)))
    for T in (2, 3, 4):
        uniq_miss = uniq_sel = base_miss = base_sel = 0
        for t0 in range(0, n_tok - T + 1):
            for l in range(n_layer):
                ms, ss = set(), set()
                for t in range(t0, t0 + T):
                    ms |= set(int(e) for e, m in zip(ids[t, l], miss[t, l]) if m)
                    ss |= set(int(e) for e in ids[t, l])
                    base_miss += int(miss[t, l].sum())
                    base_sel += n_used
                uniq_miss += len(ms)
                uniq_sel += len(ss)
        print("T=%d  miss union ratio u = %.3f (distinct misses / sum of per-token misses), selected-experts union ratio %.3f -> traffic x%.2f for %d tokens" % (
            T, uniq_miss / max(base_miss, 1), uniq_sel / max(base_sel, 1), T * uniq_miss / max(base_miss, 1), T))
    return 0


if __name__ == "__main__":
    sys.exit(main())
