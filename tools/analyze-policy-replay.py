#!/usr/bin/env python3
"""G17 offline policy study (host-only, deterministic): the engine's cache policy replayed EXACTLY (analyze-pregate.replay_policy reproduces the
engine's hit rate and per-step hits, G13-0), with the variants the idea ledger left open:

  row 22  a non-uniform per-layer slot split at EQUAL total slots (3,240 = 81 x 40): miss-proportional (from the other prompt), a split from the
          training frequency prior only, and the same-prompt oracle (an upper bound);
  rows 17, 25  the promotion rule: a gain threshold, a larger budget B, "second touch" admission, implemented as a recency filter (a candidate must have been routed in the latest step too), and a layer-type dependent EMA decay (gated DeltaNet layers against full-attention layers).

For every variant it reports the hit rate (steps >= warm), promotions per token, and the modelled change of the host time per token:
    delta_us = - (extra hits per token) * miss_us  +  (extra promotions per token) * promo_us
with miss_us = 42 us (the in-situ cost of one cold missed expert on the lean leg) and promo_us = 7 us (the DRAM time a promotion's DMA takes from the miss
path: 16 promotions slow the CPU leg by about 16% while the link runs, ledger row 23). It is a screen: it does not simulate the timeline.

Inputs: the g28 capture traces (trace_*.bin) and the frequency prior, as in analyze-pregate.py. No GPU, no model.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("apg", os.path.join(HERE, "analyze-pregate.py"))
apg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apg)

K_CAND = 8        # candidate pairs per layer and token (the engine's MOE_HYBRID_K)


def replay(ids, prior, slots_l, lam_l, budget=16, theta=0.0, second_touch=False, scan_mode=2):
    """The engine policy with per-layer slot counts `slots_l` [nl] and EMA decays `lam_l` [nl]; `theta` = minimum gain of a promotion,
    `second_touch` = a candidate must have been selected in the previous step as well. Same order of operations as analyze-pregate.replay_policy."""
    steps, nl, nu = ids.shape
    ne = prior.shape[1]
    fmax = float(prior.max())
    tie = (1e-6 * prior.astype(np.float64)) / fmax
    ema = np.zeros((nl, ne))
    pub = np.zeros((nl, ne), dtype=bool)
    cached = np.zeros((nl, ne), dtype=bool)
    for l in range(nl):
        order = np.argsort(-prior[l], kind="stable")[:int(slots_l[l])]
        pub[l, order] = True
        cached[l, order] = True
    lam = np.asarray(lam_l, dtype=np.float64)[:, None]
    pending = []
    scan_out, scan_have = [], False
    hits = np.zeros(steps, dtype=np.int64)
    prom = np.zeros(steps, dtype=np.int64)
    rows = np.arange(nl)[:, None]
    prev_sel = np.zeros((nl, ne), dtype=bool)

    def pairs_now():
        score = ema + tie
        out = []
        for l in range(nl):
            sc = score[l]
            unc = np.flatnonzero(~cached[l])
            if second_touch:
                unc = unc[prev_sel[l, unc]]
            cac = np.flatnonzero(cached[l])
            cs = unc[np.argsort(-sc[unc], kind="stable")[:K_CAND]]
            vs = cac[np.argsort(sc[cac], kind="stable")[:K_CAND]]
            for k in range(min(len(cs), len(vs))):
                gain = float(sc[cs[k]] - sc[vs[k]])
                if not gain > theta:
                    break
                out.append((gain, l, int(cs[k]), int(vs[k])))
        cap = budget
        if cap > 0 and len(out) > cap:
            out = sorted(out, key=lambda p: -p[0])[:cap]
        return out

    for s in range(steps):
        sel = ids[s]
        hits[s] = int(pub[rows, sel].sum())
        ema *= lam
        ema[rows, sel] += (1.0 - lam)
        cur_sel = np.zeros((nl, ne), dtype=bool)
        cur_sel[rows, sel] = True
        if pending:
            for (l, e) in pending:
                pub[l, e] = True
            pending = []
        if budget != 0:
            if scan_mode == 0:
                pairs = pairs_now()
            else:
                pairs = scan_out if scan_have else []
                scan_have = False
            for (_g, l, c, v) in pairs:
                pub[l, v] = False
                cached[l, v] = False
                cached[l, c] = True
                pending.append((l, c))
            prom[s] = len(pairs)
        prev_sel = cur_sel
        if scan_mode != 0 and budget != 0:
            scan_out = pairs_now()
            scan_have = True
    return hits, prom


def split_from_weights(w, total, lo, hi):
    """integer slot counts proportional to w, summing to total, each in [lo, hi] (largest remainder after clipping)"""
    w = np.asarray(w, dtype=np.float64)
    nl = len(w)
    s = np.full(nl, float(lo))
    free = total - lo * nl
    cap = np.full(nl, float(hi - lo))
    share = w / w.sum()
    alloc = np.zeros(nl)
    active = np.ones(nl, dtype=bool)
    for _ in range(20):
        want = np.where(active, share * free, 0.0)
        over = want > (cap - alloc)
        if not over.any():
            alloc = alloc + want
            break
        alloc = np.where(over & active, cap, alloc)
        free = total - lo * nl - alloc.sum()
        active = active & ~over
        share = np.where(active, w, 0.0)
        share = share / share.sum() if share.sum() > 0 else share
    s = lo + alloc
    out = np.floor(s).astype(int)
    rem = total - out.sum()
    order = np.argsort(-(s - out))
    for i in range(int(rem)):
        out[order[i % nl]] += 1
    return out


def greedy_prior_split(prior, total, lo, hi):
    """static greedy: give the next slot to the layer whose next most frequent expert has the largest training frequency share"""
    nl, ne = prior.shape
    freq = np.sort(prior.astype(np.float64), axis=1)[:, ::-1]
    freq = freq / freq.sum(axis=1, keepdims=True)
    s = np.full(nl, lo)
    left = total - lo * nl
    while left > 0:
        gain = np.where(s < hi, freq[np.arange(nl), np.minimum(s, ne - 1)], -1.0)
        l = int(np.argmax(gain))
        s[l] += 1
        left -= 1
    return s


def layer_miss_counts(ids_o, prior, ne, slots, warm):
    steps, nl_, nu = ids_o.shape
    ema = np.zeros((nl_, ne))
    tie = (1e-6 * prior.astype(np.float64)) / float(prior.max())
    pub = np.zeros((nl_, ne), dtype=bool)
    cached = np.zeros((nl_, ne), dtype=bool)
    for l in range(nl_):
        order = np.argsort(-prior[l], kind="stable")[:slots]
        pub[l, order] = True
        cached[l, order] = True
    pending, scan_out, scan_have = [], [], False
    rows = np.arange(nl_)[:, None]
    miss = np.zeros(nl_)
    for s in range(steps):
        sel = ids_o[s]
        if s >= warm:
            miss += nu - pub[rows, sel].sum(axis=1)
        ema *= 0.97
        ema[rows, sel] += 0.03
        for (l, e) in pending:
            pub[l, e] = True
        pending = []
        pairs = scan_out if scan_have else []
        scan_have = False
        for (_g, l, c, v) in pairs:
            pub[l, v] = False; cached[l, v] = False; cached[l, c] = True
            pending.append((l, c))
        score = ema + tie
        out = []
        for l in range(nl_):
            sc = score[l]
            unc = np.flatnonzero(~cached[l]); cac = np.flatnonzero(cached[l])
            cs = unc[np.argsort(-sc[unc], kind="stable")[:K_CAND]]
            vs = cac[np.argsort(sc[cac], kind="stable")[:K_CAND]]
            for k in range(min(len(cs), len(vs))):
                g = float(sc[cs[k]] - sc[vs[k]])
                if not g > 0.0:
                    break
                out.append((g, l, int(cs[k]), int(vs[k])))
        if len(out) > 16:
            out = sorted(out, key=lambda q: -q[0])[:16]
        scan_out, scan_have = out, True
    return miss



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "build", "llama-cuda"))
    ap.add_argument("--prompts", default="code,prose")
    ap.add_argument("--session-prefix", default="g28-cap-")
    ap.add_argument("--slots", type=int, default=81)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--miss-us", type=float, default=42.0)
    ap.add_argument("--promo-us", type=float, default=7.0)
    ap.add_argument("--token-us", type=float, default=8000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer variants")
    ap.add_argument("--emit-split", default=None, help="write the per-layer misses of the baseline replay of each prompt, and pooled, as <prefix>-<prompt>.txt / <prefix>-pooled.txt (the engine's LLAMA_MOE_HYBRID_SPLIT files) and exit")
    ap.add_argument("--combos", action="store_true", help="after the table, also score the combinations of the cross-prompt split with a larger budget and the recency filter (G17b)")
    a = ap.parse_args(argv)
    prompts = a.prompts.split(",")
    out_dir = a.out or os.path.join(a.root, "g17-policy")
    os.makedirs(out_dir, exist_ok=True)
    prior = None
    tr = {}
    for p in prompts:
        tr[p] = apg.read_trace(os.path.join(a.root, "%s%s" % (a.session_prefix, p), "trace_%s.bin" % p))
    nl, ne = tr[prompts[0]]["n_layer"], tr[prompts[0]]["n_expert"]
    prior = apg.read_prior(os.path.join(a.root, "aer-freq-train.txt"), nl, ne)
    total = a.slots * nl
    if a.emit_split:
        pooled = np.zeros(nl)
        for p in prompts:
            m = layer_miss_counts(tr[p]["ids"], prior, ne, a.slots, a.warm)
            pooled += m
            open("%s-%s.txt" % (a.emit_split, p), "w").write("\n".join("%.1f" % v for v in m) + "\n")
        open("%s-pooled.txt" % a.emit_split, "w").write("\n".join("%.1f" % v for v in pooled) + "\n")
        print("wrote", a.emit_split + "-{" + ",".join(prompts) + ",pooled}.txt", "per-layer baseline misses (steps >= %d, %d slots)" % (a.warm, a.slots))
        return 0
    lam0 = np.full(nl, 0.97)
    uniform = np.full(nl, a.slots)

    # full-attention layers: every 4th, layers 3, 7, 11, ... (qwen35moe full_attn_interval 4)
    fa = np.array([(l + 1) % 4 == 0 for l in range(nl)])

    def run_variant(name, ids, slots_l, lam_l, **kw):
        t0 = time.time()
        hits, prom = replay(ids, prior, slots_l, lam_l, **kw)
        S = slice(a.warm, None)
        steps = ids.shape[0] - a.warm
        lookups = float(steps * nl * ids.shape[2])
        hit_rate = hits[S].sum() / lookups
        hit_all = hits.sum() / float(ids.shape[0] * nl * ids.shape[2])
        return dict(name=name, hit_rate=float(hit_rate), hit_all=float(hit_all), prom_per_token=float(prom[S].mean()), miss_per_token=float(ids.shape[1] * ids.shape[2] * (1 - hit_rate)), secs=time.time() - t0)

    results = {}
    for p in prompts:
        ids = tr[p]["ids"]
        res = []
        res.append(run_variant("baseline uniform %d, B16, lam .97" % a.slots, ids, uniform, lam0))
        base_hits = res[0]["hit_rate"]
        print("[%s] baseline hit rate %.4f over steps >= %d, %.4f over all steps (the engine logged 0.8213 code / 0.8371 prose over all steps) (%.0f s)" % (p, base_hits, a.warm, res[0]["hit_all"], res[0]["secs"]), flush=True)

        # ---- row 22: non-uniform splits at equal total slots
        other = [q for q in prompts if q != p]
        # per-layer miss rate of the baseline replay of the OTHER prompt (cross-prompt: no test-on-train)
        def layer_miss(ids_o):
            steps, nl_, nu = ids_o.shape
            h, _ = replay(ids_o, prior, uniform, lam0)
            # per-layer misses need the per-layer hits: recompute with a light variant
            return None
        # per-layer miss counts come from a replay that keeps them: re-run baseline with layer resolution
        if a.combos and other:
            mo_c = layer_miss_counts(tr[other[0]]["ids"], prior, ne, a.slots, a.warm)
            for alpha in (0.5, 1.0):
                sp_c = split_from_weights(mo_c ** alpha + 1e-9, total, 24, 160)
                for b_c in (16, 24, 32):
                    for rec_c in (False, True):
                        if b_c == 16 and not rec_c:
                            continue
                        res.append(run_variant("COMBO split(%s misses^%.1f) + B=%d%s" % (other[0], alpha, b_c, " + recency" if rec_c else ""), ids, sp_c, lam0, budget=b_c, second_touch=rec_c))
        if other:
            mo = layer_miss_counts(tr[other[0]]["ids"], prior, ne, a.slots, a.warm)
            for alpha in (0.5, 1.0):
                sp = split_from_weights(mo ** alpha + 1e-9, total, 24, 160)
                res.append(run_variant("split by %s misses^%.1f (cross-prompt) [min %d max %d]" % (other[0], alpha, sp.min(), sp.max()), ids, sp, lam0))
        mself = layer_miss_counts(ids, prior, ne, a.slots, a.warm)
        for alpha in (0.5, 1.0):
            sp = split_from_weights(mself ** alpha + 1e-9, total, 24, 160)
            res.append(run_variant("split by OWN misses^%.1f (oracle, upper bound) [min %d max %d]" % (alpha, sp.min(), sp.max()), ids, sp, lam0))
        sp = greedy_prior_split(prior, total, 24, 160)
        res.append(run_variant("split by the training prior (greedy static) [min %d max %d]" % (sp.min(), sp.max()), ids, sp, lam0))

        # ---- rows 17, 25: promotion rule variants
        for b in (8, 12, 24, 32):
            res.append(run_variant("B=%d" % b, ids, uniform, lam0, budget=b))
        for th in (0.002, 0.005, 0.01, 0.02):
            res.append(run_variant("gain threshold %.3f" % th, ids, uniform, lam0, theta=th))
        res.append(run_variant("recency filter (candidate was routed in the latest step)", ids, uniform, lam0, second_touch=True))
        res.append(run_variant("recency filter, B=24", ids, uniform, lam0, second_touch=True, budget=24))
        res.append(run_variant("recency filter, B=32", ids, uniform, lam0, second_touch=True, budget=32))
        if not a.quick:
            for ld, lf in ((0.95, 0.97), (0.98, 0.97), (0.97, 0.90), (0.97, 0.99), (0.95, 0.90), (0.98, 0.99)):
                lam = np.where(fa, lf, ld)
                res.append(run_variant("lam DeltaNet %.2f / full-attention %.2f" % (ld, lf), ids, uniform, lam))
        results[p] = res

    lines = ["## G17 offline policy study (analyze-policy-replay.py): exact engine policy replay, slots %d x %d layers, scan mode 2, steps >= %d" % (a.slots, nl, a.warm),
             "modelled change of the host time per token = -(extra hits) x %.0f us + (extra promotions) x %.0f us (a model); share of an %.0f us token" % (a.miss_us, a.promo_us, a.token_us), ""]
    for p in prompts:
        base = results[p][0]
        lines.append("### %s" % p)
        lines.append("| variant | hit rate | d hit (pp) | promotions/token | misses/token | modelled d host us/token | share of token |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in results[p]:
            dh = r["hit_rate"] - base["hit_rate"]
            dmiss = -(r["miss_per_token"] - base["miss_per_token"])
            dprom = r["prom_per_token"] - base["prom_per_token"]
            d_us = -dmiss * a.miss_us + dprom * a.promo_us
            lines.append("| %s | %.4f | %+.2f | %.1f | %.1f | %+.0f | %+.2f%% |" % (r["name"], r["hit_rate"], 100 * dh, r["prom_per_token"], r["miss_per_token"], d_us, 100 * d_us / a.token_us))
        lines.append("")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(out_dir, "tables.md"), "w").write(md + "\n")
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
