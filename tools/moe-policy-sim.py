#!/usr/bin/env python3
"""Replay a routing trace (LLAMA_MOE_HYBRID_TRACE) against cache policies and bounds, host-only.

Trace format (written by src/llama-moe-hybrid.cpp): "MOETRC01", int32 n_layer, n_used, n_expert, max_used; then per token an int32 token index and
int32[n_layer * max_used] expert ids (-1 = layer had no data). Per layer the cache holds C experts; the 8 experts a token selects are hits when cached.

Policies:
  ema     the engine's policy: per-layer routing EMA (lambda) + 1e-6 * prior-frequency tie-break; before each token the best uncached (up to 8) are paired
          with the worst cached (up to 8); pairs with positive gain are promotion candidates and the B largest gains over all layers are promoted. A promoted
          expert takes its victim's slot: the victim leaves at once, the newcomer is published for the NEXT token (its DMA is in flight).
  lru     demand paging: a missed expert is inserted (evicting the least recently used), lookups refresh recency
  lfu     demand paging with aged frequency counts
  static  the C most frequent experts of the prior, never changing
  belady  clairvoyant demand paging (evict the cached expert used farthest in the future); insertion only on a miss
  bound   clairvoyant with free prefetch: the cache before each token holds the C experts used soonest (an upper bound for ANY policy, promotion budget ignored)

  python tools/moe-policy-sim.py routing.bin --prior build/llama-cuda/aer-freq-train.txt --slots 14,28,42 [--lambda 0.97] [--budget 16]

It measures; it claims nothing.
"""
import argparse
import struct
import sys

import numpy as np


def read_trace(path):
    with open(path, "rb") as fh:
        assert fh.read(8) == b"MOETRC01"
        n_layer, n_used, n_expert, max_used = struct.unpack("<4i", fh.read(16))
        rec = 1 + n_layer * max_used
        data = np.frombuffer(fh.read(), dtype="<i4")
    n_tok = len(data) // rec
    data = data[: n_tok * rec].reshape(n_tok, rec)
    ids = data[:, 1:].reshape(n_tok, n_layer, max_used)[:, :, :n_used]
    return ids.astype(np.int64), n_layer, n_used, n_expert


def sim_ema(ids, n_layer, n_expert, C, prior, lam, budget):
    n_tok = ids.shape[0]
    fmax = float(prior.max())
    tie = 1e-6 * prior / fmax
    ema = np.zeros((n_layer, n_expert))
    cached = np.zeros((n_layer, n_expert), dtype=bool)   # slot_of >= 0 (incl. pending)
    pub = np.zeros((n_layer, n_expert), dtype=bool)      # published map
    for l in range(n_layer):
        order = np.argsort(-prior[l], kind="stable")[:C]
        cached[l, order] = True
        pub[l, order] = True
    pending = []
    hits = lookups = 0
    for t in range(n_tok):
        sel = ids[t]
        h = pub[np.arange(n_layer)[:, None], sel]
        hits += int(h.sum())
        lookups += sel.size
        ema *= lam
        for l in range(n_layer):
            ema[l, sel[l]] += (1.0 - lam)
        for l, e in pending:
            pub[l, e] = True
        pending = []
        if budget == 0:
            continue
        pairs = []
        K = 8
        for l in range(n_layer):
            sc = ema[l] + tie[l]
            unc = np.where(~cached[l])[0]
            if len(unc) == 0:
                continue
            cs = unc[np.argsort(-sc[unc], kind="stable")[:K]]
            cac = np.where(cached[l])[0]
            vs = cac[np.argsort(sc[cac], kind="stable")[:K]]
            n = min(len(cs), len(vs))
            for k in range(n):
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
            pending.append((l, c))
    return hits / lookups


def sim_demand(ids, n_layer, n_expert, C, prior, policy):
    n_tok = ids.shape[0]
    hits = lookups = 0
    for l in range(n_layer):
        order = list(np.argsort(-prior[l], kind="stable")[:C])
        cache = {int(e): 0 for e in order}      # expert -> last use / score
        cnt = {int(e): 0.0 for e in order}
        # next-use tables for belady
        if policy == "belady":
            nxt = {}
            seq = [set(int(x) for x in ids[t, l]) for t in range(n_tok)]
            use_pos = {}
            for t in range(n_tok):
                for e in seq[t]:
                    use_pos.setdefault(e, []).append(t)
            ptr = {e: 0 for e in use_pos}
        for t in range(n_tok):
            sel = [int(x) for x in ids[t, l]]
            for e in sel:
                lookups += 1
                if e in cache:
                    hits += 1
            if policy == "belady":
                for e in set(sel):
                    while ptr[e] < len(use_pos[e]) and use_pos[e][ptr[e]] <= t:
                        ptr[e] += 1
            for e in set(sel):
                if e in cache:
                    cache[e] = t
                    cnt[e] = cnt.get(e, 0.0) + 1.0
                    continue
                if len(cache) >= C:
                    if policy == "lru":
                        v = min(cache, key=lambda k: cache[k])
                    elif policy == "lfu":
                        v = min(cache, key=lambda k: cnt.get(k, 0.0))
                    else:  # belady: farthest next use
                        def nu(k):
                            p = use_pos.get(k)
                            return p[ptr[k]] if p and ptr[k] < len(p) else 1 << 30
                        v = max(cache, key=nu)
                    del cache[v]
                cache[e] = t
                cnt[e] = cnt.get(e, 0.0) + 1.0
            if policy == "lfu" and t % 8 == 0:
                for k in cnt:
                    cnt[k] *= 0.97
    return hits / lookups


def sim_static(ids, n_layer, n_expert, C, prior):
    hits = lookups = 0
    for l in range(n_layer):
        top = set(int(x) for x in np.argsort(-prior[l], kind="stable")[:C])
        for t in range(ids.shape[0]):
            for e in ids[t, l]:
                lookups += 1
                hits += int(e) in top
    return hits / lookups


def sim_bound(ids, n_layer, n_expert, C):
    n_tok = ids.shape[0]
    hits = lookups = 0
    for l in range(n_layer):
        use_pos = {}
        for t in range(n_tok):
            for e in set(int(x) for x in ids[t, l]):
                use_pos.setdefault(e, []).append(t)
        ptr = {e: 0 for e in use_pos}
        for t in range(n_tok):
            sel = set(int(x) for x in ids[t, l])
            # rank the experts by their next use at or after t
            nu = []
            for e, p in use_pos.items():
                while ptr[e] < len(p) and p[ptr[e]] < t:
                    ptr[e] += 1
                nu.append((p[ptr[e]] if ptr[e] < len(p) else 1 << 30, e))
            nu.sort()
            top = set(e for _, e in nu[:C])
            for e in ids[t, l]:
                lookups += 1
                hits += int(e) in top
    return hits / lookups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--slots", default="14,28,42")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.97)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--policies", default="ema,static,lru,lfu,belady,bound")
    a = ap.parse_args()
    ids, n_layer, n_used, n_expert = read_trace(a.trace)
    prior = np.array([[int(x) for x in line.split()] for line in open(a.prior) if line.strip()], dtype=np.int64)
    assert prior.shape == (n_layer, n_expert), (prior.shape, n_layer, n_expert)
    print("trace: %d tokens x %d layers x %d experts used (%s)" % (ids.shape[0], n_layer, n_used, a.trace))
    for C in [int(x) for x in a.slots.split(",")]:
        out = []
        for pol in a.policies.split(","):
            if pol == "ema":
                r = sim_ema(ids, n_layer, n_expert, C, prior, a.lam, a.budget)
            elif pol == "static":
                r = sim_static(ids, n_layer, n_expert, C, prior)
            elif pol == "bound":
                r = sim_bound(ids, n_layer, n_expert, C)
            else:
                r = sim_demand(ids, n_layer, n_expert, C, prior, pol)
            out.append("%s %.4f" % (pol, r))
        print("C=%-3d  %s" % (C, "   ".join(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
