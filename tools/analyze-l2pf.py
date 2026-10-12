#!/usr/bin/env python3
"""G15 offline screen (host-only, deterministic): how many of the host leg's cold expert reads would a hidden-state predictor warm,
if the engine's own worker threads prefetch the predicted experts into their L2 caches between two requests?

The mechanism being screened (see docs/plans/tg-beat-stock-8gb-262k.md, "G15 CONTRACT"):
  * when the engine serves the request of layer l-1 it holds the layer's FFN input x_{l-1} (the mailbox's x);
  * after it has replied it computes router_l(x_{l-1}) with a host copy of layer l's router weights (lead 1, no re-normalisation: "P-A"),
    takes the first K experts of that ranking that are NOT resident in the VRAM cache (limited to the ranking's top-POOL), and every worker
    thread issues software prefetches for its own row slices of those experts' up / gate / down banks;
  * layer l's request then finds a predicted expert's rows in the workers' caches and reads them at the "hot" cost instead of the DRAM-bound "cold" one.
Constraint of the v1 mechanism: a prediction exists only for layers whose predecessor had a request (>= 1 miss) in the same decode step.

Inputs are the g28 capture sessions (see analyze-pregate.py): trace_*.bin, xdump_*.bin, the router weights cache g13-pregate/weights.npz, the
frequency prior. The cache state of every (step, layer) is the exact replay of the engine's policy (G13-0 validated it to the step).

Reported per prompt and pooled, for K = 1, 2, 3 and router weights in float32 / bfloat16 / int8 (per-row scale): prefetch triggers and
prefetches issued per token, USEFUL prefetches per token (the expert was a true miss of layer l), precision, the fraction of all misses (and of the
misses of triggered layers) that were prefetched, and the modelled host time saved per token = useful * (cold - hot) in us and as a share of the token.

It is a screen, not a result: the saving is a model (cold and hot per-expert costs are inputs) and the arrival of the data in the cache is assumed.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("apg", os.path.join(HERE, "analyze-pregate.py"))
apg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apg)


def to_bf16(a):
    """round-to-nearest-even float32 -> bfloat16 -> float32"""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def to_int8_rowscale(a):
    """per-row symmetric int8 quantization -> dequantized float32"""
    a = np.asarray(a, dtype=np.float32)
    s = np.abs(a).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return (np.clip(np.round(a / s), -127, 127) * s).astype(np.float32)


def quantize_router(R, kind):
    if kind == "f32":
        return R.astype(np.float32)
    if kind == "bf16":
        return np.stack([to_bf16(R[l]) for l in range(R.shape[0])])
    if kind == "int8":
        return np.stack([to_int8_rowscale(R[l]) for l in range(R.shape[0])])
    raise ValueError(kind)


def screen_prompt(name, paths, R, prior, slots, ks, pool, warm, kinds):
    tr = apg.read_trace(paths["trace"])
    X = apg.read_xdump(paths["xdump"])
    ids = tr["ids"]
    steps, nl, nu = ids.shape
    ne = tr["n_expert"]
    rp = apg.replay_policy(ids, prior, slots, lam=0.97, budget=16, scan_mode=2)
    M = rp["M"]
    Tind = np.zeros((steps, nl, ne), dtype=bool)
    np.put_along_axis(Tind, ids, True, axis=2)
    miss = Tind & ~M                                       # true misses [steps, nl, ne]
    nmiss = miss.sum(axis=2)                               # [steps, nl]
    S = np.arange(warm, steps)
    res = {"name": name, "steps": int(len(S)), "misses_per_token": float(nmiss[S].sum(axis=1).mean()),
           "miss_layers_per_token": float((nmiss[S] > 0).sum(axis=1).mean())}
    for kind in kinds:
        Rq = quantize_router(R, kind)
        # per target layer l = 1 .. nl-1: scores from the FFN input of layer l-1
        tot = {k: dict(trig=0.0, issued=0.0, useful=0.0, useful_trig_layers=0.0, miss_trig_layers=0.0) for k in ks}
        for l in range(1, nl):
            Z = X[S, l - 1].astype(np.float32) @ Rq[l].T                    # [S, ne]
            order = np.argsort(-Z, axis=1, kind="stable")[:, :pool]          # the ranking's top-`pool`
            nonres = ~np.take_along_axis(M[S, l], order, axis=1)             # non-resident among the pool, in rank order
            csum = nonres.cumsum(axis=1)
            trig = nmiss[S, l - 1] > 0                                       # the predecessor posted a request
            missl = np.take_along_axis(miss[S, l], order, axis=1)            # true miss, in rank order
            for k in ks:
                sel = nonres & (csum <= k) & trig[:, None]
                tot[k]["trig"] += float(trig.sum())
                tot[k]["issued"] += float(sel.sum())
                tot[k]["useful"] += float((sel & missl).sum())
                # misses of layers whose predecessor posted a request: the part the v1 mechanism can address at all
                tot[k]["miss_trig_layers"] += float(nmiss[S, l][trig].sum())
        nst = float(len(S))
        res[kind] = {}
        for k in ks:
            t = tot[k]
            res[kind][k] = dict(trig_per_token=t["trig"] / nst, issued_per_token=t["issued"] / nst, useful_per_token=t["useful"] / nst,
                                precision=(t["useful"] / t["issued"]) if t["issued"] else 0.0,
                                cover_all=t["useful"] / float(nmiss[S].sum()),
                                cover_addressable=(t["useful"] / t["miss_trig_layers"]) if t["miss_trig_layers"] else 0.0,
                                addressable_misses_per_token=t["miss_trig_layers"] / nst)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "build", "llama-cuda"))
    ap.add_argument("--prompts", default="code,prose")
    ap.add_argument("--session-prefix", default="g28-cap-")
    ap.add_argument("--weights-cache", default=None)
    ap.add_argument("--slots", type=int, default=81)
    ap.add_argument("--pool", type=int, default=12, help="candidates are taken from the predictor's own top-POOL ranks")
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--kinds", default="f32,bf16,int8")
    ap.add_argument("--cold-us", type=float, default=42.0, help="in-situ cold cost of one missed expert on the lean leg (us)")
    ap.add_argument("--hot-us", default="13,20", help="comma list: cost of an expert whose rows are in the workers' caches (us)")
    ap.add_argument("--token-us", type=float, default=8360.0, help="token time of the lean leg (us) for the share column")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    kinds = a.kinds.split(",")
    hots = [float(x) for x in a.hot_us.split(",")]
    prompts = a.prompts.split(",")
    out_dir = a.out or os.path.join(a.root, "g15-l2pf")
    os.makedirs(out_dir, exist_ok=True)

    prior_path = os.path.join(a.root, "aer-freq-train.txt")
    paths = {}
    for p in prompts:
        d = os.path.join(a.root, "%s%s" % (a.session_prefix, p))
        paths[p] = dict(xdump=os.path.join(d, "xdump_%s.bin" % p), trace=os.path.join(d, "trace_%s.bin" % p))
        for k in paths[p].values():
            if not os.path.exists(k):
                print("missing input:", k)
                return 2
    first = apg.read_trace(paths[prompts[0]]["trace"])
    nl, ne = first["n_layer"], first["n_expert"]
    prior = apg.read_prior(prior_path, nl, ne)
    cache = a.weights_cache or os.path.join(a.root, "g13-pregate", "weights.npz")
    R, W, eps, fa, src = apg.load_weights(cache, "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf", nl)
    print("weights:", R.shape, src)

    results = {}
    for p in prompts:
        print("prompt", p, flush=True)
        results[p] = screen_prompt(p, paths[p], R, prior, a.slots, ks, a.pool, a.warm, kinds)

    lines = ["## G15 offline screen (analyze-l2pf.py): pool top-%d, lead 1, trigger = the predecessor layer posted a request, slots %d" % (a.pool, a.slots), ""]
    for p in prompts + ["pooled"]:
        if p == "pooled":
            lines.append("### pooled (mean of the two prompts, equal weight)")
        else:
            r = results[p]
            lines.append("### %s: %d steps, %.1f misses per token, %.1f layers with misses per token" % (p, r["steps"], r["misses_per_token"], r["miss_layers_per_token"]))
        hdr = "| router | K | triggers/token | issued/token | useful/token | precision | share of all misses | share of addressable misses | addressable misses/token |" + "".join(
            " saved us (hot %g) | share of token |" % h for h in hots)
        lines.append(hdr)
        lines.append("|" + "---|" * (9 + 2 * len(hots)))
        for kind in kinds:
            for k in ks:
                if p == "pooled":
                    d = {key: float(np.mean([results[q][kind][k][key] for q in prompts])) for key in results[prompts[0]][kind][k]}
                else:
                    d = results[p][kind][k]
                row = "| %s | %d | %.1f | %.1f | %.1f | %.3f | %.3f | %.3f | %.1f |" % (
                    kind, k, d["trig_per_token"], d["issued_per_token"], d["useful_per_token"], d["precision"], d["cover_all"], d["cover_addressable"], d["addressable_misses_per_token"])
                for h in hots:
                    sv = d["useful_per_token"] * (a.cold_us - h)
                    row += " %.0f | %.1f%% |" % (sv, 100.0 * sv / a.token_us)
                lines.append(row)
        lines.append("")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(out_dir, "tables.md"), "w").write(md + "\n")
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
