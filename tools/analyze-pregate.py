#!/usr/bin/env python3
"""G13 offline analysis (host-only, deterministic): can the router of layer l, fed with the FFN input of layer l-1
(lead 1) or l-2 (lead 2), name the experts that layer l will miss in the per-layer VRAM expert cache?

Pre-registered in docs/plans/tg-beat-stock-8gb-262k.md, section "G13 CONTRACT" (hypothesis H13, criteria G13-0, G13-1, G13-2).
The criteria are constants below and are NOT tunable from the command line.

Inputs (all read-only; captured by the g28 sessions):
  xdump_<p>.bin   "MOEXD001", int32[4] = (steps, n_layer, n_embd, 0), float32 [steps][n_layer][n_embd]: the router input (the
                  FFN input after the layer's post-attention norm) of every layer for every decode step
  trace_<p>.bin   "MOETRC01", int32[4] = (n_layer, n_used, n_expert, MAX_USED), then per decode step: int32 token index and
                  int32[n_layer][MAX_USED] selected experts (router top-k order)
  hybrid_PLAIN.jsonl, *-PLAIN.log   the engine's own per-step hits / promotions / waits and its final "hit rate" line
  aer-freq-train.txt                the frequency prior (initial fill and EMA tie-break)
  weights.npz     blk.N.ffn_gate_inp.weight and blk.N.post_attention_norm.weight for N = 0..39, cached from the GGUF on first use

Steps:
  1. G13-0 validity. (a) the router recomputed from the captured input must reproduce the trace's selected set in >= 99.5% of
     (step, layer) cells; (b) the cache policy replay (EMA lambda, tie-break from the prior, 8 best-uncached against 8 worst-cached
     candidates per layer, B largest gains per token, initial fill by the prior, scan mode 2 = "lag 1", promotions published one step
     after they are decided) must reproduce the engine's logged hit rate within 0.2 pp (it is also compared step by step).
  2. Predictors at lead 1 and lead 2 (P-A router_l(x) as is; P-B x un-normalised by the source layer's norm weight, RMS-normalised,
     re-weighted by the target layer's norm weight; P-C = P-B whose candidate pool is limited to its own top-12) and the baselines
     (history, ema, oracle, chance): recall of the true set, and for b = 1..4 the b highest-ranked predicted experts that are NOT
     resident in the replayed cache at that step: precision (fraction that are true misses), true misses covered per step, and
     fraction of ALL misses covered.
  3. Verdict against G13-1 and G13-2 exactly as pre-registered.

Exit status: 0 = G13-0 valid and G13-1 + G13-2 hold for every predictor variant (P-A, P-B, P-C); 1 = G13-0 valid and they fail for every
variant; 3 = G13-0 valid but the verdict depends on the variant (the pre-registration does not say whether the candidate pool is the
predictor's whole ranking or its predicted top-12); 2 = G13-0 invalid or an input is missing (nothing may be concluded).
--selftest: 0 pass, 1 fail.

It measures; it claims nothing beyond the pre-registered criteria.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---------------------------------------------------------------------------------------------------------------------------
# Pre-registered criteria (docs/plans/tg-beat-stock-8gb-262k.md, "G13 CONTRACT"). Not to be moved after the numbers.
# ---------------------------------------------------------------------------------------------------------------------------
G13_0_HIT_TOL_PP = 0.2        # replayed hit rate within 0.2 percentage points of the engine's logged one
G13_0_ROUTER_MIN_PCT = 99.5   # router recomputed from the captured input reproduces the trace in >= 99.5% of (step, layer) cells
G13_1_RECALL12_MIN = 0.70     # recall of the true 8 within the top-12 predicted
G13_1_PREC2_MIN = 0.40        # top-2 predicted NON-RESIDENT experts are true misses at least 40% of the time
G13_1_PREC2_EACH_MIN = 0.35   # ... and each prompt at least 35%
G13_2_COVER2_MIN = 0.25       # prefetching those 2 per layer turns >= 25% of the cache misses into hits
HEADLINE_PREDICTOR = "P-B"    # the hypothesis' own form ("(re-normalised) FFN input"); P-A and P-C are evaluated and reported with the same weight

KS = (8, 12, 16, 24)                  # recall depths
BS = (1, 2, 3, 4)                     # prefetch candidates per layer
BS_EXT = (1, 2, 3, 4, 6, 8)           # extended coverage curve (reported, not gated)
MKS = (8, 12, 16, 24, 32, 64)         # depth of the predicted ranking at which the true MISSES are found (residency-agnostic)
POOLS = (8, 12, 16, 24, 256)          # candidate pool = the predictor's own top-pool ranks (256 = no restriction)
P_C_POOL = 12
MOE_HYBRID_K = 8                      # engine constant: candidate pairs per layer and token


# ---------------------------------------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_trace(path):
    """MOETRC01 routing trace -> dict(ids int64 [steps, n_layer, n_used], tok, n_layer, n_used, n_expert)."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != b"MOETRC01":
            raise ValueError("%s: bad magic %r (expected MOETRC01)" % (path, magic))
        n_layer, n_used, n_expert, max_used = struct.unpack("<4i", fh.read(16))
        raw = fh.read()
    rec = 1 + n_layer * max_used
    arr = np.frombuffer(raw, dtype="<i4")
    if arr.size % rec != 0:
        raise ValueError("%s: %d int32 are not a whole number of %d-int records" % (path, arr.size, rec))
    arr = arr.reshape(-1, rec)
    ids = arr[:, 1:].reshape(arr.shape[0], n_layer, max_used)[:, :, :n_used].astype(np.int64)
    return dict(ids=ids, tok=arr[:, 0].astype(np.int64), n_layer=n_layer, n_used=n_used, n_expert=n_expert)


def read_xdump(path):
    """MOEXD001 router-input dump -> float32 [steps, n_layer, n_embd]."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != b"MOEXD001":
            raise ValueError("%s: bad magic %r (expected MOEXD001)" % (path, magic))
        steps, n_layer, n_embd, _ = struct.unpack("<4i", fh.read(16))
        x = np.fromfile(fh, dtype="<f4")
    if x.size != steps * n_layer * n_embd:
        raise ValueError("%s: %d floats, header says %d x %d x %d" % (path, x.size, steps, n_layer, n_embd))
    return x.reshape(steps, n_layer, n_embd)


def write_trace(path, ids, n_expert, max_used=8):
    steps, n_layer, n_used = ids.shape
    with open(path, "wb") as fh:
        fh.write(b"MOETRC01")
        fh.write(struct.pack("<4i", n_layer, n_used, n_expert, max_used))
        for s in range(steps):
            blk = np.full((n_layer, max_used), -1, dtype="<i4")
            blk[:, :n_used] = ids[s]
            fh.write(struct.pack("<i", s))
            fh.write(blk.tobytes())


def write_xdump(path, x):
    steps, n_layer, n_embd = x.shape
    with open(path, "wb") as fh:
        fh.write(b"MOEXD001")
        fh.write(struct.pack("<4i", steps, n_layer, n_embd, 0))
        fh.write(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_prior(path, n_layer, n_expert):
    rows = [[int(v) for v in line.split()] for line in open(path) if line.strip()]
    prior = np.array(rows, dtype=np.int64)
    if prior.shape != (n_layer, n_expert):
        raise ValueError("%s: prior has shape %s, expected (%d, %d)" % (path, prior.shape, n_layer, n_expert))
    return prior


def read_engine_jsonl(path):
    """The engine's per-step records ({"t":..,"hits":..,"lookups":..,"prom":..,"wait":..}), config and summary events."""
    steps, config, summary, timeline = [], None, None, None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if "t" in j:
                steps.append(j)
            elif j.get("event") == "config":
                config = j
            elif j.get("event") == "summary":
                summary = j
            elif j.get("event") == "timeline":
                timeline = j
    steps.sort(key=lambda r: r["t"])
    return dict(steps=steps, config=config, summary=summary, timeline=timeline)


def read_engine_log(path):
    """The runner log (may contain NULs / UTF-16 remnants): the final 'hit rate' line and the resized slot count."""
    with open(path, "rb") as fh:
        text = fh.read().replace(b"\x00", b"").decode("utf-8", "replace")
    out = dict(hit_rate=None, hits=None, lookups=None, tokens=None, slots=None, scan_mode=None)
    m = re.search(r"moe_hybrid: tokens (\d+), hit rate ([0-9.]+) \((\d+)/(\d+)\)", text)
    if m:
        out.update(tokens=int(m.group(1)), hit_rate=float(m.group(2)), hits=int(m.group(3)), lookups=int(m.group(4)))
    m = re.search(r"cache resized to (\d+) slots", text)
    if m:
        out["slots"] = int(m.group(1))
    m = re.search(r"on_token breakdown \(scan mode (\d)\)", text)
    if m:
        out["scan_mode"] = int(m.group(1))
    return out


def _tensor_f32(t):
    from gguf import GGMLQuantizationType as Q
    if t.tensor_type == Q.F32:
        return np.asarray(t.data, dtype=np.float32)
    if t.tensor_type == Q.F16:
        return np.asarray(t.data).astype(np.float32)
    from gguf.quants import dequantize
    return np.asarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32)


def load_weights(cache_path, model_path, n_layer, rebuild=False):
    """router [n_layer, n_expert, n_embd] f32, norm [n_layer, n_embd] f32, eps, fa_interval; cached as an npz."""
    if os.path.exists(cache_path) and not rebuild:
        z = np.load(cache_path)
        fa = int(z["fa_interval"]) if "fa_interval" in z.files else 4
        return z["router"], z["norm"], float(z["eps"]), fa, "cache " + cache_path
    sys.path.insert(0, os.path.join(REPO, "third_party", "llama.cpp", "gguf-py"))
    from gguf import GGUFReader
    rd = GGUFReader(model_path)
    by = {t.name: t for t in rd.tensors}
    arch = rd.fields["general.architecture"].contents()
    eps = float(rd.fields["%s.attention.layer_norm_rms_epsilon" % arch].contents())
    fa = int(rd.fields["%s.full_attention_interval" % arch].contents()) if "%s.full_attention_interval" % arch in rd.fields else 0
    routers, norms = [], []
    for l in range(n_layer):
        tr = by["blk.%d.ffn_gate_inp.weight" % l]
        tn = by["blk.%d.post_attention_norm.weight" % l]
        n_embd, n_expert = (int(v) for v in tr.shape.tolist())   # ggml order: ne[0] = n_embd, ne[1] = n_expert
        routers.append(_tensor_f32(tr).reshape(n_expert, n_embd))
        norms.append(_tensor_f32(tn).reshape(n_embd))
    router = np.stack(routers)
    norm = np.stack(norms)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, router=router, norm=norm, eps=np.float32(eps), fa_interval=np.int32(fa), n_layer=np.int32(n_layer),
             src=np.array("%s blk.N.ffn_gate_inp.weight [n_embd, n_expert], blk.N.post_attention_norm.weight [n_embd], N = 0..%d"
                          % (os.path.basename(model_path), n_layer - 1)))
    return router, norm, eps, fa, "gguf " + model_path


# ---------------------------------------------------------------------------------------------------------------------------
# The cache policy (llama-moe-hybrid.cpp: init / resize / on_token / compute_pairs), replayed on the host
# ---------------------------------------------------------------------------------------------------------------------------
def compute_pairs(ema, tie, cached, K=MOE_HYBRID_K, budget=16, scan_mult=1):
    """Per layer the K best uncached experts (EMA + tie-break, descending) against the K worst cached ones (ascending), paired while
    the gain is positive; then the `budget` largest gains over all layers (stable). Ties keep the lower expert id first, exactly like
    the engine's insertion lists."""
    nl = ema.shape[0]
    out = []
    score = ema + tie
    for l in range(nl):
        sc = score[l]
        unc = np.flatnonzero(~cached[l])
        cac = np.flatnonzero(cached[l])
        cs = unc[np.argsort(-sc[unc], kind="stable")[:K]]
        vs = cac[np.argsort(sc[cac], kind="stable")[:K]]
        for k in range(min(len(cs), len(vs))):
            gain = float(sc[cs[k]] - sc[vs[k]])
            if not gain > 0.0:
                break
            out.append((gain, l, int(cs[k]), int(vs[k])))
    cap = budget * max(1, scan_mult) if budget > 0 else budget
    if cap > 0 and len(out) > cap:
        out = sorted(out, key=lambda p: -p[0])[:cap]    # python's sort is stable, like std::stable_sort with a.gain > b.gain
    return out


def replay_policy(ids, prior, slots, lam=0.97, budget=16, scan_mode=2, waits=None):
    """Replay the engine for one-row decode steps. State seen by the graph of step s (all recorded before step s is ingested):
      M[s]  published map (what the router lookup of step s saw), E[s] EMA after steps < s, PD[s] promoted-but-not-yet-published.
    Order inside the engine's on_token call that ingests step s (call #s+1): ingest the trace (hits are counted against the map the
    graph used, then EMA = lam*EMA + (1-lam)*[selected]) -> publish pending promotions (unless the DMA event was not ready: `waits`)
    -> apply the pairs (mode 0: scanned now; modes 1/2: the pairs scanned at the END of the previous call) -> scan for the next call.
    Applying a pair removes the victim at once; the newcomer is published by the next call."""
    steps, nl, nu = ids.shape
    ne = prior.shape[1]
    fmax = float(prior.max())
    if fmax <= 0:
        raise ValueError("invalid frequency prior")
    tie = (1e-6 * prior.astype(np.float64)) / fmax
    decay = 1.0 - lam
    ema = np.zeros((nl, ne))
    pub = np.zeros((nl, ne), dtype=bool)
    cached = np.zeros((nl, ne), dtype=bool)
    for l in range(nl):
        order = np.argsort(-prior[l], kind="stable")[:slots]       # most frequent first, ties -> lower id
        pub[l, order] = True
        cached[l, order] = True
    pending = []
    scan_out, scan_have = [], False
    hits = np.zeros(steps, dtype=np.int64)
    prom = np.zeros(steps, dtype=np.int64)
    M = np.zeros((steps, nl, ne), dtype=bool)
    PD = np.zeros((steps, nl, ne), dtype=bool)
    E = np.zeros((steps, nl, ne))
    rows = np.arange(nl)[:, None]
    for s in range(steps):
        M[s] = pub
        PD[s] = cached & ~pub
        E[s] = ema
        sel = ids[s]
        hits[s] = int(pub[rows, sel].sum())
        ema *= lam
        ema[rows, sel] += decay
        waited = bool(waits[s]) if waits is not None else False
        if pending and not waited:
            for (l, e) in pending:
                pub[l, e] = True
            pending = []
        if not waited and budget != 0:
            if scan_mode == 0:
                pairs = compute_pairs(ema, tie, cached, budget=budget)
            else:
                pairs = scan_out if scan_have else []
                scan_have = False
            for (_gain, l, c, v) in pairs:
                if not (pub[l, v] and cached[l, v] and not cached[l, c]):
                    raise AssertionError("replay diverged from the engine's invariants at step %d layer %d (cand %d victim %d)" % (s, l, c, v))
                pub[l, v] = False
                cached[l, v] = False
                cached[l, c] = True
                pending.append((l, c))
            prom[s] = len(pairs)
        if scan_mode != 0 and budget != 0:
            scan_out = compute_pairs(ema, tie, cached, budget=budget)
            scan_have = True
    return dict(hits=hits, prom=prom, M=M, PD=PD, E=E, tie=tie)


def replay_reference(ids, prior, slots, lam=0.97, budget=16, scan_mode=2, waits=None):
    """Literal pure-python port of the engine code (insertion lists, slot_of / map / pending), for the selftest only."""
    steps, nl, nu = ids.shape
    ne = prior.shape[1]
    freq = [[int(v) for v in row] for row in prior]
    fmax = max(max(r) for r in freq)
    tie = [[(1e-6 * float(freq[l][e])) / float(fmax) for e in range(ne)] for l in range(nl)]
    ema = [[0.0] * ne for _ in range(nl)]
    mp = [[-1] * ne for _ in range(nl)]          # published slot
    slot_of = [[-1] * ne for _ in range(nl)]     # slot incl. pending
    for l in range(nl):
        order = sorted(range(ne), key=lambda e: -freq[l][e])        # python sort is stable
        for s in range(slots):
            mp[l][order[s]] = s
            slot_of[l][order[s]] = s
    pending = []
    scan_out, scan_have = [], False
    decay = 1.0 - lam

    def pairs_now():
        out = []
        for l in range(nl):
            cs, ce, vs, ve = [], [], [], []
            for e in range(ne):
                sc = ema[l][e] + tie[l][e]
                if slot_of[l][e] < 0:
                    if len(cs) < MOE_HYBRID_K:
                        cs.append(0.0)
                        ce.append(0)
                        p = len(cs) - 1
                    elif sc > cs[MOE_HYBRID_K - 1]:
                        p = MOE_HYBRID_K - 1
                    else:
                        continue
                    while p > 0 and cs[p - 1] < sc:
                        cs[p], ce[p] = cs[p - 1], ce[p - 1]
                        p -= 1
                    cs[p], ce[p] = sc, e
                else:
                    if len(vs) < MOE_HYBRID_K:
                        vs.append(0.0)
                        ve.append(0)
                        p = len(vs) - 1
                    elif sc < vs[MOE_HYBRID_K - 1]:
                        p = MOE_HYBRID_K - 1
                    else:
                        continue
                    while p > 0 and vs[p - 1] > sc:
                        vs[p], ve[p] = vs[p - 1], ve[p - 1]
                        p -= 1
                    vs[p], ve[p] = sc, e
            for k in range(min(len(cs), len(vs))):
                gain = cs[k] - vs[k]
                if not gain > 0.0:
                    break
                out.append((gain, l, ce[k], ve[k]))
        cap = budget if budget > 0 else budget
        if cap > 0 and len(out) > cap:
            out.sort(key=lambda p: -p[0])
            out = out[:cap]
        return out

    hits = []
    prom = []
    Ms = []
    for s in range(steps):
        Ms.append(np.array([[mp[l][e] >= 0 for e in range(ne)] for l in range(nl)]))
        h = 0
        for l in range(nl):
            for e in ids[s][l]:
                h += 1 if mp[l][int(e)] >= 0 else 0
        hits.append(h)
        for l in range(nl):
            for e in range(ne):
                ema[l][e] *= lam
            for e in ids[s][l]:
                ema[l][int(e)] += decay
        waited = bool(waits[s]) if waits is not None else False
        if pending and not waited:
            for (l, e, sl) in pending:
                mp[l][e] = sl
            pending = []
        n_prom = 0
        if not waited and budget != 0:
            if scan_mode == 0:
                pairs = pairs_now()
            else:
                pairs = scan_out if scan_have else []
                scan_have = False
            for (_g, l, c, v) in pairs:
                sl = slot_of[l][v]
                assert sl >= 0 and mp[l][v] == sl and slot_of[l][c] < 0
                mp[l][v] = -1
                slot_of[l][v] = -1
                slot_of[l][c] = sl
                pending.append((l, c, sl))
            n_prom = len(pairs)
        prom.append(n_prom)
        if scan_mode != 0 and budget != 0:
            scan_out = pairs_now()
            scan_have = True
    return dict(hits=np.array(hits), prom=np.array(prom), M=np.array(Ms))


# ---------------------------------------------------------------------------------------------------------------------------
# Router and predictors
# ---------------------------------------------------------------------------------------------------------------------------
def router_logits(x, r, dtype=np.float64):
    """x [S, n_embd], r [n_expert, n_embd] -> [S, n_expert]. float64 by default (run-to-run determinism); the engine's GPU router
    accumulates in float32, so validate_router can also be run in float32."""
    return np.asarray(x, dtype=dtype) @ np.asarray(r, dtype=dtype).T


def renorm_input(y_src, w_src, w_dst, eps, tiny=1e-6):
    """P-B input. y_src = rmsnorm(h) * w_src (what the xdump holds). Undo the source norm weight (components with |w| <= tiny carry no
    recoverable information and are zeroed), RMS-normalise, apply the target layer's norm weight."""
    y = np.asarray(y_src, dtype=np.float64)
    ws = np.asarray(w_src, dtype=np.float64)
    wd = np.asarray(w_dst, dtype=np.float64)
    ok = np.abs(ws) > tiny
    h = np.where(ok, y / np.where(ok, ws, 1.0), 0.0)
    h = h / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
    return h * wd


def validate_router(X, T, R, n_used, dtype=np.float64):
    """G13-0 part 2: top-n_used set of router(x) against the trace, per (step, layer). Returns match [steps, n_layer] and the
    logit gap between the n_used-th and the (n_used+1)-th expert of every cell."""
    steps, nl, _ = X.shape
    match = np.zeros((steps, nl), dtype=bool)
    gap = np.zeros((steps, nl))
    for l in range(nl):
        lg = router_logits(X[:, l], R[l], dtype=dtype)
        order = np.argsort(-lg, axis=1, kind="stable")
        top = np.sort(order[:, :n_used], axis=1)
        match[:, l] = np.all(top == np.sort(T[:, l], axis=1), axis=1)
        srt = np.take_along_axis(lg, order, axis=1)
        gap[:, l] = srt[:, n_used - 1] - srt[:, n_used]
    return match, gap


def shifted_match_pct(X, T, R, n_used, shift):
    """Fraction of cells that match when the xdump step s is paired with the trace step s+shift (alignment check)."""
    steps, nl, _ = X.shape
    a0, a1 = (0, steps - shift) if shift >= 0 else (-shift, steps)
    ok = tot = 0
    for l in range(nl):
        lg = router_logits(X[a0:a1, l], R[l])
        top = np.sort(np.argsort(-lg, axis=1, kind="stable")[:, :n_used], axis=1)
        tr = np.sort(T[a0 + shift:a1 + shift, l], axis=1)
        ok += int(np.all(top == tr, axis=1).sum())
        tot += top.shape[0]
    return 100.0 * ok / tot


def make_scorers(X, W, R, eps, E, tie, Tind_all, pred):
    """Return Z(l, lead) -> [steps, n_expert] scores for the named ranking predictor."""
    if pred == "P-A":
        return lambda l, lead: router_logits(X[:, l - lead], R[l])
    if pred in ("P-B", "P-C"):
        return lambda l, lead: router_logits(renorm_input(X[:, l - lead], W[l - lead], W[l], eps), R[l])
    if pred == "history":
        prev = np.zeros_like(Tind_all)
        prev[1:] = Tind_all[:-1]
        return lambda l, lead: 1e3 * prev[:, l].astype(np.float64) + E[:, l] + tie[l]
    if pred == "ema":
        return lambda l, lead: E[:, l] + tie[l]
    if pred == "oracle":
        return lambda l, lead: Tind_all[:, l].astype(np.float64)
    raise ValueError(pred)


# ---------------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------------
def cell_metrics(Z, Tids, Mres, PDl, n_used, pools=POOLS, bs=BS_EXT, ks=KS, mks=MKS):
    """Per-(step) metrics of one target layer. Z [S, NE] scores, Tids [S, n_used] true experts, Mres [S, NE] resident (published),
    PDl [S, NE] pending. Returns a dict name -> array [S]:
      miss, nonres; rec{k} = # of the true set inside the top-k; mrec{k} = # of the true MISSES inside the top-k (residency-agnostic);
      iss{b}@{pool}, cov{b}@{pool}, pcov{b}@{pool}: the b highest-ranked predicted experts that are not resident (limited to the
      predictor's own top-`pool` ranks): how many were issued, how many are true misses, how many of those were already pending."""
    S, NE = Z.shape
    order = np.argsort(-Z, axis=1, kind="stable")
    Tind = np.zeros((S, NE), dtype=bool)
    np.put_along_axis(Tind, Tids, True, axis=1)
    miss = Tind & ~Mres
    Tord = np.take_along_axis(Tind, order, axis=1)
    missord = np.take_along_axis(miss, order, axis=1)
    nr = ~np.take_along_axis(Mres, order, axis=1)
    pendord = np.take_along_axis(PDl, order, axis=1)
    pos = np.arange(NE)[None, :]
    out = {"miss": miss.sum(axis=1), "nonres": (~Mres).sum(axis=1)}
    for k in ks:
        out["rec%d" % k] = Tord[:, :k].sum(axis=1)
        out["nr%d" % k] = nr[:, :k].sum(axis=1)               # non-resident experts inside the predictor's top-k
    for k in mks:
        out["mrec%d" % k] = missord[:, :k].sum(axis=1)
    for pool in pools:
        nrp = nr & (pos < pool)
        csum = nrp.cumsum(axis=1)
        for b in bs:
            sel = nrp & (csum <= b)
            out["iss%d@%d" % (b, pool)] = sel.sum(axis=1)
            out["cov%d@%d" % (b, pool)] = (missord & sel).sum(axis=1)
            out["pcov%d@%d" % (b, pool)] = (missord & pendord & sel).sum(axis=1)
    return out


def chance_cell_metrics(Tids, Mres, n_used, pools=(256,), bs=BS_EXT, ks=KS, mks=MKS):
    """Analytic expectation of a uniformly random ranking (no sampling noise)."""
    S, NE = Mres.shape
    Tind = np.zeros((S, NE), dtype=bool)
    np.put_along_axis(Tind, Tids, True, axis=1)
    miss = (Tind & ~Mres).sum(axis=1).astype(np.float64)
    nonres = (~Mres).sum(axis=1).astype(np.float64)
    out = {"miss": miss, "nonres": nonres}
    for k in ks:
        out["rec%d" % k] = np.full(S, n_used * k / NE)
        out["nr%d" % k] = nonres * k / NE
    for k in mks:
        out["mrec%d" % k] = miss * k / NE
    for b in bs:
        out["iss%d@256" % b] = np.full(S, float(b))
        out["cov%d@256" % b] = b * miss / nonres
        out["pcov%d@256" % b] = np.zeros(S)
    return out


def evaluate_ranker(Zfun, layers, lead, T, M, PD, warm, n_used, chance=False, cg=None):
    """Evaluate a ranking predictor over target `layers` and steps >= warm. Returns per-layer [L, S] arrays.
    cg (optional, length n_used+1, cg[0] = 0): the host engine's CPU-graph time (us) of a layer-step with m misses; when given, also
    sav<b>@<pool> = cg[misses] - cg[misses - covered], the host serve time that covering those misses would remove (a model, see the
    report; the copies are taken as free, exactly like the G13-2 bound)."""
    steps = T.shape[0]
    S = np.arange(warm, steps)
    arr = {}
    for li, l in enumerate(layers):
        if chance:
            cm = chance_cell_metrics(T[S, l], M[S, l], n_used)
        else:
            cm = cell_metrics(Zfun(l, lead)[S], T[S, l], M[S, l], PD[S, l], n_used)
            if cg is not None:
                n_miss = cm["miss"].astype(np.int64)
                for key in [k for k in cm if k.startswith("cov")]:
                    c = cm[key].astype(np.int64)
                    cm["sav" + key[3:]] = cg[n_miss] - cg[n_miss - c]
        for name, v in cm.items():
            if name not in arr:
                arr[name] = np.zeros((len(layers), len(S)), dtype=np.float64)
            arr[name][li] = v
    return arr


def reduce_cells(arr):
    """Per-layer sums [L], per-step sums [S] and totals of the [L, S] arrays."""
    return dict(per_layer={k: v.sum(axis=1) for k, v in arr.items()},
                per_step={k: v.sum(axis=0) for k, v in arr.items()},
                total={k: float(v.sum()) for k, v in arr.items()},
                n_layers=next(iter(arr.values())).shape[0], n_steps=next(iter(arr.values())).shape[1])


def derive(sums, ncell, nsteps, n_used, extra_miss=0.0):
    """Ratios from sums (works on totals, on one layer's sums or on a bootstrap resample's sums)."""
    d = {"ncell": ncell, "miss_per_cell": sums["miss"] / ncell, "miss_per_step": sums["miss"] / nsteps,
         "resident_share_true": 1.0 - sums["miss"] / (n_used * ncell)}
    for k in KS:
        d["rec%d" % k] = sums["rec%d" % k] / (n_used * ncell)
        d["nonres_in_top%d" % k] = sums["nr%d" % k] / ncell
    for k in MKS:
        d["mrec%d" % k] = sums["mrec%d" % k] / sums["miss"] if sums["miss"] > 0 else float("nan")
    for pool in POOLS:
        for b in BS_EXT:
            key = "%d@%d" % (b, pool)
            if "iss" + key not in sums:
                continue
            iss, cov = sums["iss" + key], sums["cov" + key]
            d["prec" + key] = cov / iss if iss > 0 else float("nan")                 # among the prefetches actually issued
            d["sprec" + key] = cov / (b * ncell)                                     # per prefetch slot (unissued slots count as wasted)
            d["covfrac" + key] = cov / sums["miss"] if sums["miss"] > 0 else float("nan")
            d["covfrac_wt" + key] = cov / (sums["miss"] + extra_miss) if (sums["miss"] + extra_miss) > 0 else float("nan")
            d["covstep" + key] = cov / nsteps
            d["issstep" + key] = iss / nsteps
            d["isscell" + key] = iss / ncell
            d["pendshare" + key] = sums["pcov" + key] / cov if cov > 0 else float("nan")
            if "sav" + key in sums:
                d["savstep" + key] = sums["sav" + key] / nsteps          # modelled host serve time removed per step (us)
    return d


def block_bootstrap_idx(n_steps, block, n_boot, seed):
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n_steps))
    nb = -(-n_steps // block)
    starts = rng.integers(0, n_steps - block + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, nb * block)[:, :n_steps]


def bootstrap_ci(per_step_list, idx_list, fn):
    """per_step_list: [dict name -> [S]] per prompt, idx_list: [B, S] per prompt (same B). fn(sums) -> value, where sums are summed over
    the prompts of one resample. Returns (lo, hi) of the 95% percentile interval."""
    B = idx_list[0].shape[0]
    vals = np.zeros(B)
    names = per_step_list[0].keys()
    for b in range(B):
        sums = {k: 0.0 for k in names}
        for ps, idx in zip(per_step_list, idx_list):
            for k in names:
                sums[k] += float(ps[k][idx[b]].sum())
        vals[b] = fn(sums)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def headline_ci(results, prompts, pred, n_used):
    """Block-bootstrap 95% intervals of the verdict-relevant lead-1 metrics, per prompt and pooled over both."""
    key = "%s|lead1" % pred
    pool = P_C_POOL if pred == "P-C" else 256
    need = ["rec12", "miss"] + ["%s%d@%d" % (a, b, pool) for a in ("cov", "iss") for b in (1, 2)]
    out = {}
    for scope in prompts + ["pooled"]:
        sel = prompts if scope == "pooled" else [scope]
        psl = [{k: results["prompts"][p]["preds"][key]["per_step"][k] for k in need} for p in sel]
        idl = [results["prompts"][p]["_boot_idx"] for p in sel]
        ncell = sum(results["prompts"][p]["preds"][key]["ncell"] for p in sel)

        def ratio(num, den):
            return num / den if den > 0 else float("nan")

        fns = {"rec12": lambda s: s["rec12"] / (n_used * ncell),
               "prec1": lambda s: ratio(s["cov1@%d" % pool], s["iss1@%d" % pool]),
               "prec2": lambda s: ratio(s["cov2@%d" % pool], s["iss2@%d" % pool]),
               "covfrac1": lambda s: ratio(s["cov1@%d" % pool], s["miss"]),
               "covfrac2": lambda s: ratio(s["cov2@%d" % pool], s["miss"])}
        out[scope] = {m: list(bootstrap_ci(psl, idl, fn)) for m, fn in fns.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Verdict (pre-registered logic)
# ---------------------------------------------------------------------------------------------------------------------------
def g13_verdict(per_prompt, pooled, key_prec="prec2@256", key_cov="covfrac2@256", key_rec="rec12"):
    """per_prompt: {prompt: derived-metrics dict}, pooled: derived-metrics dict over both prompts (cell-weighted).
    G13-1: recall of the true 8 within the top-12 >= 0.70 and the top-2 non-resident predicted are true misses >= 40% of the time
    (each prompt >= 35%). G13-2: the 2 prefetches per layer cover >= 25% of the misses. Recall and coverage are required of the pooled
    value AND of each prompt (the stricter reading); precision as written (pooled 0.40, each prompt 0.35)."""
    v = {}
    v["recall12_pooled"] = (pooled[key_rec], pooled[key_rec] >= G13_1_RECALL12_MIN)
    v["recall12_each"] = ({p: d[key_rec] for p, d in per_prompt.items()}, all(d[key_rec] >= G13_1_RECALL12_MIN for d in per_prompt.values()))
    v["prec2_pooled"] = (pooled[key_prec], pooled[key_prec] >= G13_1_PREC2_MIN)
    v["prec2_each"] = ({p: d[key_prec] for p, d in per_prompt.items()}, all(d[key_prec] >= G13_1_PREC2_EACH_MIN for d in per_prompt.values()))
    v["cover2_pooled"] = (pooled[key_cov], pooled[key_cov] >= G13_2_COVER2_MIN)
    v["cover2_each"] = ({p: d[key_cov] for p, d in per_prompt.items()}, all(d[key_cov] >= G13_2_COVER2_MIN for d in per_prompt.values()))
    v["G13-1"] = all(v[k][1] for k in ("recall12_pooled", "recall12_each", "prec2_pooled", "prec2_each"))
    v["G13-2"] = all(v[k][1] for k in ("cover2_pooled", "cover2_each"))
    return v


# ---------------------------------------------------------------------------------------------------------------------------
# One prompt
# ---------------------------------------------------------------------------------------------------------------------------
PREDICTORS = ("P-A", "P-B", "P-C", "history", "ema", "oracle", "chance")


def analyze_prompt(name, paths, W, R, eps, prior, args, log):
    t0 = time.time()
    tr = read_trace(paths["trace"])
    X = read_xdump(paths["xdump"])
    ids, nl, nu, ne = tr["ids"], tr["n_layer"], tr["n_used"], tr["n_expert"]
    steps = ids.shape[0]
    if X.shape[0] != steps or X.shape[1] != nl:
        raise ValueError("%s: xdump %s and trace %s disagree" % (name, X.shape, ids.shape))
    if not np.array_equal(tr["tok"], np.arange(steps)):
        raise ValueError("%s: trace token indices are not 0..%d" % (name, steps - 1))
    res = {"name": name, "steps": steps, "n_layer": nl, "n_used": nu, "n_expert": ne}

    # ---- engine logs
    ej = read_engine_jsonl(paths["jsonl"]) if os.path.exists(paths["jsonl"]) else None
    el = read_engine_log(paths["log"]) if os.path.exists(paths["log"]) else None
    waits = None
    if ej and len(ej["steps"]) == steps:
        waits = [r["wait"] for r in ej["steps"]]
    lam = float(ej["config"]["lambda"]) if ej and ej["config"] else 0.97
    budget = int(ej["config"]["B"]) if ej and ej["config"] else 16
    slots = args.slots or (el["slots"] if el and el["slots"] else 81)
    scan_mode = args.scan_mode if args.scan_mode is not None else (el["scan_mode"] if el and el["scan_mode"] is not None else 2)
    res["engine"] = dict(log=el, summary=(ej["summary"] if ej else None), waits_used=waits is not None, n_waits=int(sum(waits)) if waits else None,
                         slots=slots, lam=lam, budget=budget, scan_mode=scan_mode)

    # ---- G13-0 (a): router reproduction and alignment
    match, gap = validate_router(X, ids, R, nu)
    mism = np.argwhere(~match)
    res["router"] = dict(cells=int(match.size), match=int(match.sum()), pct=100.0 * float(match.mean()),
                         mismatches=[dict(step=int(s), layer=int(l), gap_8th_9th=float(gap[s, l])) for s, l in mism[:20]],
                         gap_median_all=float(np.median(gap)), gap_p01_all=float(np.percentile(gap, 1)),
                         shift_minus1_pct=shifted_match_pct(X, ids, R, nu, -1), shift_plus1_pct=shifted_match_pct(X, ids, R, nu, 1))
    match32, gap32 = validate_router(X, ids, R, nu, dtype=np.float32)       # the GPU router accumulates in float32: report that too
    mism32 = np.argwhere(~match32)
    res["router_float32"] = dict(cells=int(match32.size), match=int(match32.sum()), pct=100.0 * float(match32.mean()),
                                 mismatches=[dict(step=int(s), layer=int(l), gap_8th_9th=float(gap32[s, l]), gap_8th_9th_float64=float(gap[s, l]))
                                             for s, l in mism32[:20]])
    log("  [%s] router: %d/%d cells match in float64 (%.4f%%), %d/%d in float32 (%.4f%%), shift-1 %.3f%%, shift+1 %.3f%%" % (
        name, match.sum(), match.size, res["router"]["pct"], match32.sum(), match32.size, res["router_float32"]["pct"],
        res["router"]["shift_minus1_pct"], res["router"]["shift_plus1_pct"]))

    # ---- G13-0 (b): policy replay
    rp = replay_policy(ids, prior, slots, lam=lam, budget=budget, scan_mode=scan_mode, waits=waits)
    lookups = steps * nl * nu
    replay_rate = float(rp["hits"].sum()) / lookups
    eng_hits = eng_lookups = None
    if ej and ej["summary"]:
        eng_hits, eng_lookups = int(ej["summary"]["hits"]), int(ej["summary"]["lookups"])
    elif el and el["hits"]:
        eng_hits, eng_lookups = el["hits"], el["lookups"]
    eng_rate = eng_hits / eng_lookups if eng_hits else None
    rep = dict(replay_hits=int(rp["hits"].sum()), replay_lookups=lookups, replay_rate=replay_rate, engine_hits=eng_hits, engine_lookups=eng_lookups,
               engine_rate=eng_rate, engine_rate_log4=(el["hit_rate"] if el else None),
               diff_pp=(100.0 * (replay_rate - eng_rate) if eng_rate is not None else None))
    if ej and len(ej["steps"]) == steps:
        eh = np.array([r["hits"] for r in ej["steps"]])
        ep = np.array([r["prom"] for r in ej["steps"]])
        rep.update(steps_hits_equal=int((eh == rp["hits"]).sum()), steps_prom_equal=int((ep == rp["prom"]).sum()), max_abs_step_hit_diff=int(np.abs(eh - rp["hits"]).max()))
    M, PD, E, tie = rp["M"], rp["PD"], rp["E"], rp["tie"]
    Tind_all = np.zeros((steps, nl, ne), dtype=bool)
    np.put_along_axis(Tind_all, ids, True, axis=2)
    miss_all = Tind_all & ~M
    # The engine also logs, per decoded step, the number of layers that had misses (`ml`, from the GPU timeline), and per host request the number
    # of experts that missed (summary nmiss_hist). The replay must reproduce `ml` step by step; the request histogram can only differ by a step the trace
    # never ingested: the engine serves the requests of the LAST decode graph, but no on_token call follows it to write its trace row.
    nmiss_replay = np.bincount(miss_all.sum(axis=2).ravel(), minlength=nu + 1)[:nu + 1]
    ml_replay = (miss_all.sum(axis=2) > 0).sum(axis=1)
    if ej and len(ej["steps"]) == steps and all("ml" in r for r in ej["steps"]):
        rep["steps_ml_equal"] = int((np.array([r["ml"] for r in ej["steps"]]) == ml_replay).sum())
    if ej and ej["summary"] and "nmiss_hist" in ej["summary"]:
        eng_h = [int(v) for v in ej["summary"]["nmiss_hist"][1:]]                           # logged as [err, 1..n_used]
        rep_h = [int(v) for v in nmiss_replay[1:]]
        sur = [e - a for e, a in zip(eng_h, rep_h)]
        rep.update(nmiss_hist_engine=eng_h, nmiss_hist_replay=rep_h, nmiss_hist_surplus=sur, nmiss_hist_replay_within_engine=all(v >= 0 for v in sur),
                   requests_surplus=int(sum(sur)), misses_surplus=int(sum((m + 1) * v for m, v in enumerate(sur))))
    res["replay"] = rep
    log("  [%s] replay: hit rate %.4f (%d/%d) vs engine %s (%s) | per-step hits equal %s | per-step miss-layers equal %s | engine requests exceed the ingested steps' by %s | "
        "slots %d, scan mode %d, B %d, lambda %g" % (
            name, replay_rate, rp["hits"].sum(), lookups, ("%.4f" % eng_rate) if eng_rate is not None else "n/a",
            ("%d/%d" % (eng_hits, eng_lookups)) if eng_hits else "n/a",
            ("%d/%d" % (rep["steps_hits_equal"], steps)) if "steps_hits_equal" in rep else "n/a",
            ("%d/%d" % (rep["steps_ml_equal"], steps)) if "steps_ml_equal" in rep else "n/a",
            ("%d (%d misses)" % (rep["requests_surplus"], rep["misses_surplus"])) if "requests_surplus" in rep else "n/a", slots, scan_mode, budget, lam))

    # ---- diagnostics shared by the predictor runs
    S = np.arange(args.warm, steps)
    res["warm"] = args.warm
    res["resident_count_published_mean"] = float(M[S].sum(axis=2).mean())
    res["misses_per_layer_step"] = [float(miss_all[S, l].sum(axis=1).mean()) for l in range(nl)]
    nm = miss_all[S].sum(axis=2)                                         # [S, nl] true misses per cell
    res["miss_count_hist_layers1plus"] = [float((nm[:, 1:] == m).mean()) for m in range(0, 9)]
    # oracle ceiling at b: E[min(b, N)] / b
    res["oracle_prec_ceiling"] = {str(b): float(np.minimum(nm[:, 1:], b).mean() / b) for b in BS_EXT}

    # host serve model (a model, not a measurement): the engine's own CPU-graph time by miss count, from its timeline line
    tl = ej["timeline"] if ej else None
    cg = None
    if tl and tl.get("cpu_graph_us") and len(tl["cpu_graph_us"]) == nu:
        cg = np.concatenate([[0.0], np.asarray(tl["cpu_graph_us"], dtype=np.float64)])
        model_hserve = float(cg[nm.reshape(len(S), nl)].sum(axis=1).mean())            # layers 0..nl-1, steps >= warm
        res["host_model"] = dict(cpu_graph_us=cg.tolist(), model_hserve_us_per_step=model_hserve, engine_hserve_us=tl["hserve_us"],
                                 engine_stall_us=tl["stall_us"], engine_span_us=tl["span_us"], engine_gap_us=tl["gap_us"],
                                 engine_ovl_us=tl["ovl_us"], engine_attn_us=tl["attn_us"],
                                 model_hserve_layers1plus_us=float(cg[nm[:, 1:]].sum(axis=1).mean()),
                                 layer_period_us=(tl["attn_us"] + tl["ovl_us"] + tl["stall_us"]) / nl,
                                 token_us=tl["span_us"] + tl["gap_us"], miss_layers=tl.get("miss_layers"))
        log("  [%s] host-serve model: sum of cg[misses] = %.0f us/step (layers 0..%d, steps >= %d) vs engine hserve %.0f us/token" % (
            name, model_hserve, nl - 1, args.warm, tl["hserve_us"]))

    # ---- predictors
    preds = {}
    for lead in args.leads:
        layers = list(range(lead, nl))
        extra_miss = float(miss_all[S][:, :, :].sum(axis=2)[:, :lead].sum())      # misses of layers < lead (unpredictable at this lead)
        for pred in PREDICTORS:
            if pred == "chance":
                arr = evaluate_ranker(None, layers, lead, ids, M, PD, args.warm, nu, chance=True)
            else:
                Zf = make_scorers(X, W, R, eps, E, tie, Tind_all, pred)
                arr = evaluate_ranker(Zf, layers, lead, ids, M, PD, args.warm, nu, cg=cg)
            red = reduce_cells(arr)
            ncell = len(layers) * len(S)
            d = derive(red["total"], ncell, len(S), nu, extra_miss)
            per_layer = {}
            for li, l in enumerate(layers):
                sums_l = {k: float(v[li]) for k, v in red["per_layer"].items()}
                per_layer[l] = derive(sums_l, len(S), len(S), nu)
            preds["%s|lead%d" % (pred, lead)] = dict(pred=pred, lead=lead, layers=layers, ncell=ncell, derived=d, per_layer=per_layer,
                                                      sums=red["total"], per_step=red["per_step"], extra_miss=extra_miss)
        log("  [%s] lead %d predictors done (%.1fs)" % (name, lead, time.time() - t0))
    res["preds"] = preds

    # ---- why the lead-1 prediction is off: similarity of consecutive layers' router inputs
    cos = {}
    for lead in args.leads:
        rows = []
        for l in range(lead, nl):
            ya = X[S, l - lead].astype(np.float64)
            yb = X[S, l].astype(np.float64)
            pb = renorm_input(X[S, l - lead], W[l - lead], W[l], eps)
            hA = ya / W[l - lead].astype(np.float64)
            hB = yb / W[l].astype(np.float64)
            def cosm(u, v):
                return float(((u * v).sum(1) / np.sqrt((u * u).sum(1) * (v * v).sum(1))).mean())
            lt = router_logits(yb, R[l])
            lp = router_logits(pb, R[l])
            lt = lt - lt.mean(1, keepdims=True)
            lp = lp - lp.mean(1, keepdims=True)
            corr = float(((lt * lp).sum(1) / np.sqrt((lt * lt).sum(1) * (lp * lp).sum(1))).mean())
            rows.append(dict(layer=l, cos_asis=cosm(ya, yb), cos_renorm=cosm(pb, yb), cos_resid=cosm(hA, hB), logit_corr=corr))
        cos["lead%d" % lead] = rows
    res["similarity"] = cos

    # ---- bootstrap indices (block bootstrap over steps; per prompt, seeded)
    res["_boot_idx"] = block_bootstrap_idx(len(S), args.boot_block, args.boot, args.seed + sum(ord(c) for c in name)) if args.boot > 0 else None
    res["_elapsed_s"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------------------------------------------------------
def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.3f" % v


def pct1(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.1f%%" % (100.0 * v)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


PRED_LABEL = {"P-A": "P-A (as is)", "P-B": "P-B (re-normalised)", "P-C": "P-C (P-B, pool top-12)", "history": "history (prev. step's set)",
              "ema": "ema (engine score)", "oracle": "oracle (true set)", "chance": "chance (analytic)"}


def render_pred_tables(pdata, prompts, title):
    """pdata[prompt] = derived dicts by predictor for this lead; also 'pooled'."""
    out = ["### %s" % title]
    for pname in prompts + ["pooled"]:
        D = pdata[pname]
        key = "%s" % pname
        out.append("")
        out.append("**%s** (%d cells per predictor; mean true misses per cell %.3f)" % (key, D["P-B"]["ncell"], D["P-B"]["miss_per_cell"]))
        rows = []
        for pr in PREDICTORS:
            if pr not in D:
                continue
            d = D[pr]
            pool = P_C_POOL if pr == "P-C" else 256
            rows.append([PRED_LABEL[pr], f3(d["rec8"]), f3(d["rec12"]), f3(d["rec16"]), f3(d["rec24"]),
                         " / ".join(f3(d["prec%d@%d" % (b, pool)]) for b in BS),
                         " / ".join(f3(d["covfrac%d@%d" % (b, pool)]) for b in BS),
                         f3(d["covstep2@%d" % pool]), f3(d["isscell2@%d" % pool])])
        out.append(md_table(["predictor", "rec@8", "rec@12", "rec@16", "rec@24", "precision b=1/2/3/4", "fraction of ALL misses covered b=1/2/3/4",
                             "misses covered per step (b=2)", "issued per layer-step (b=2)"], rows))
    return "\n".join(out)


def _ci(R_, pred, scope, metric):
    ci = R_.get("ci", {}).get(pred, {}).get(scope, {}).get(metric)
    return "" if not ci else " [%.3f, %.3f]" % (ci[0], ci[1])


def render_verdict(R_, prompts, args):
    out = ["### Verdict against the pre-registered criteria (lead 1; 39 target layers; steps >= %d)" % args.warm,
           "",
           "Thresholds exactly as pre-registered: G13-1 recall of the true 8 within the top-12 predicted >= %.2f, and the top-2 predicted experts that are NOT "
           "resident are true misses >= %.0f%% of the time (each prompt >= %.0f%%); G13-2 the 2 prefetches per layer cover >= %.0f%% of the misses. "
           "Recall and coverage are required of the pooled value AND of each prompt (the stricter reading). Brackets: 95%% block-bootstrap interval over steps "
           "(block %d, %d resamples)." % (G13_1_RECALL12_MIN, 100 * G13_1_PREC2_MIN, 100 * G13_1_PREC2_EACH_MIN, 100 * G13_2_COVER2_MIN, args.boot_block, args.boot)]
    summary = []
    for pred in ("P-A", "P-B", "P-C"):
        pool = P_C_POOL if pred == "P-C" else 256
        v = R_["verdict"][pred]
        cols = prompts + ["pooled"]
        dd = {s: R_["derived"][s]["lead1"][pred] for s in cols}
        out.append("")
        out.append("**%s** (candidate pool: %s)" % (PRED_LABEL[pred], "the predictor's top-%d ranks" % pool if pool < 256 else "its whole ranking"))
        rows = []
        for label, key, thr_pool, thr_each, res_keys in (
                ("G13-1 recall of the true 8 within the top-12", "rec12", G13_1_RECALL12_MIN, G13_1_RECALL12_MIN, ("recall12_pooled", "recall12_each")),
                ("G13-1 precision of the top-2 non-resident (among those issued)", "prec2@%d" % pool, G13_1_PREC2_MIN, G13_1_PREC2_EACH_MIN, ("prec2_pooled", "prec2_each")),
                ("G13-2 fraction of ALL misses covered by the 2 prefetches", "covfrac2@%d" % pool, G13_2_COVER2_MIN, G13_2_COVER2_MIN, ("cover2_pooled", "cover2_each"))):
            metric = key.split("@")[0]
            cells = []
            for s in cols:
                cells.append("%s%s" % (f3(dd[s][key]), _ci(R_, pred, s, metric)))
            ok = v[res_keys[0]][1] and v[res_keys[1]][1]
            rows.append([label, cells[0], cells[1], cells[2], ">= %.2f pooled, each >= %.2f" % (thr_pool, thr_each), "PASS" if ok else "FAIL"])
        out.append(md_table(["criterion", prompts[0]] + prompts[1:] + ["pooled", "threshold", "result"], rows))
        summary.append([PRED_LABEL[pred], "PASS" if v["G13-1"] else "FAIL", "PASS" if v["G13-2"] else "FAIL",
                        f3(dd["pooled"]["isscell2@%d" % pool])])
    out.append("")
    out.append("**Summary by predictor variant**")
    out.append(md_table(["variant", "G13-1", "G13-2", "prefetches actually issued per layer-step (b=2)"], summary))
    return "\n".join(out)


def render_cause(R_, prompts):
    cols = prompts + ["pooled"]
    D = {s: R_["derived"][s]["lead1"] for s in cols}
    D2 = {s: R_["derived"][s]["lead2"] for s in cols} if "lead2" in R_["derived"][cols[0]] else None
    hist = {p: R_["prompts"][p]["miss_count_hist_layers1plus"] for p in prompts}
    hist["pooled"] = [float(np.mean([hist[p][m] for p in prompts])) for m in range(9)]

    def row(label, fn):
        return [label] + [fn(s) for s in cols]

    rows = [
        row("true misses per layer-step (mean N, layers 1-39)", lambda s: f3(D[s]["P-B"]["miss_per_cell"])),
        row("resident share of the true 8 (1 - N/8)", lambda s: f3(D[s]["P-B"]["resident_share_true"])),
        row("layer-steps with 0 / 1 / >=2 true misses", lambda s: "%s / %s / %s" % (pct1(hist[s][0]), pct1(hist[s][1]), pct1(sum(hist[s][2:])))),
        row("oracle precision b=1 / b=2 (ceiling: mean min(b,N)/b)", lambda s: "%s / %s" % (f3(D[s]["oracle"]["prec1@256"]), f3(D[s]["oracle"]["prec2@256"]))),
        row("P-B precision b=1 / b=2", lambda s: "%s / %s" % (f3(D[s]["P-B"]["prec1@256"]), f3(D[s]["P-B"]["prec2@256"]))),
        row("P-B precision as a fraction of the oracle ceiling, b=1 / b=2",
            lambda s: "%s / %s" % (pct1(D[s]["P-B"]["prec1@256"] / D[s]["oracle"]["prec1@256"]), pct1(D[s]["P-B"]["prec2@256"] / D[s]["oracle"]["prec2@256"]))),
        row("precision b=2 needed for G13-1 as a fraction of the oracle ceiling", lambda s: pct1(G13_1_PREC2_MIN / D[s]["oracle"]["prec2@256"])),
        row("P-B recall of the true 8 within the top-8 / 12 / 24", lambda s: "%s / %s / %s" % (f3(D[s]["P-B"]["rec8"]), f3(D[s]["P-B"]["rec12"]), f3(D[s]["P-B"]["rec24"]))),
        row("P-B: true MISSES found inside the predicted top-8 / 12 / 16 / 24 / 32 / 64 (any residency)",
            lambda s: " / ".join(f3(D[s]["P-B"]["mrec%d" % k]) for k in (8, 12, 16, 24, 32, 64))),
        row("P-B: non-resident experts inside the predicted top-8 / top-12", lambda s: "%s / %s" % (f3(D[s]["P-B"]["nonres_in_top8"]), f3(D[s]["P-B"]["nonres_in_top12"]))),
        row("P-B: true misses inside the predicted top-8 / top-12 (per layer-step)",
            lambda s: "%s / %s" % (f3(D[s]["P-B"]["mrec8"] * D[s]["P-B"]["miss_per_cell"]), f3(D[s]["P-B"]["mrec12"] * D[s]["P-B"]["miss_per_cell"]))),
        row("P-C: prefetches issued per layer-step / precision / coverage (b=2)",
            lambda s: "%s / %s / %s" % (f3(D[s]["P-C"]["isscell2@12"]), f3(D[s]["P-C"]["prec2@12"]), f3(D[s]["P-C"]["covfrac2@12"]))),
        row("P-C: share of the covered misses that were already pending promotions (b=2)", lambda s: pct1(D[s]["P-C"]["pendshare2@12"])),
        row("lead 0 (the layer's own input) = the router itself", lambda s: "recall 1.000, precision = oracle (router reproduced in G13-0)"),
        row("lead 1 -> lead 2, P-B recall@12", lambda s: "%s -> %s" % (f3(D[s]["P-B"]["rec12"]), f3(D2[s]["P-B"]["rec12"])) if D2 else "n/a"),
        row("lead 1 -> lead 2, P-B precision b=2", lambda s: "%s -> %s" % (f3(D[s]["P-B"]["prec2@256"]), f3(D2[s]["P-B"]["prec2@256"])) if D2 else "n/a"),
        row("history baseline (previous step's set) precision b=2 / recall@12", lambda s: "%s / %s" % (f3(D[s]["history"]["prec2@256"]), f3(D[s]["history"]["rec12"]))),
        row("ema baseline (engine score) precision b=2 / recall@12", lambda s: "%s / %s" % (f3(D[s]["ema"]["prec2@256"]), f3(D[s]["ema"]["rec12"]))),
        row("chance precision b=2", lambda s: f3(D[s]["chance"]["prec2@256"])),
    ]
    sim = {}
    for p in prompts:
        sl = R_["prompts"][p]["similarity"]["lead1"]
        sim[p] = (float(np.mean([r["cos_resid"] for r in sl])), float(np.mean([r["logit_corr"] for r in sl])),
                  float(np.mean([r["cos_asis"] for r in sl])), float(np.mean([r["cos_renorm"] for r in sl])))
    sim["pooled"] = tuple(float(np.mean([sim[p][i] for p in prompts])) for i in range(4))
    rows.append(row("mean over layers 1-39: cosine of consecutive layers' unit-RMS residual / P-B logit correlation with the true logits",
                    lambda s: "%s / %s" % (f3(sim[s][0]), f3(sim[s][1]))))
    rows.append(row("mean cosine of P-A input vs true input / of P-B input vs true input", lambda s: "%s / %s" % (f3(sim[s][2]), f3(sim[s][3]))))
    return "### Cause analysis, lead 1 (all numbers from the replayed cache of each prompt)\n\n" + md_table(["quantity"] + cols, rows)


def render_link(R_, prompts, args):
    """Link-budget arithmetic: copies per token a predictor configuration would issue against what the PCIe link and the layer period allow."""
    out = ["### Lead time and PCIe link budget (arithmetic on measured/given numbers; EXPECTED, not measured)", ""]
    hm = {p: R_["prompts"][p].get("host_model") for p in prompts}
    exp_us = args.expert_us
    nt = R_["n_layer"] - 1
    out.append("Given: one expert = 1.69 MiB = %.0f us on the link (14.4 GB/s pinned DMA); the engine promotes up to B = 16 experts per token over the same link." % exp_us)
    for p in prompts:
        if hm[p]:
            out.append("- %s run: %.0f us per token (span %.0f + inter-token gap %.0f) = %.0f us per layer counting the gap; %.0f us between two consecutive layer publishes "
                       "(attn %.0f + overlap %.0f + stall %.0f over %d layers); the gap is not between layers." % (
                           p, hm[p]["token_us"], hm[p]["engine_span_us"], hm[p]["engine_gap_us"], hm[p]["token_us"] / R_["n_layer"], hm[p]["layer_period_us"],
                           hm[p]["engine_attn_us"], hm[p]["engine_ovl_us"], hm[p]["engine_stall_us"], R_["n_layer"]))
    first = next((p for p in prompts if hm[p]), None)
    if first:
        w1 = hm[first]["layer_period_us"]
        out.append("- Window of a lead-1 prefetch = one layer period: %.0f us holds %.2f experts (lead 2: %.0f us holds %.2f); two serialized copies take %.0f us, "
                   "so the second copy of a lead-1 pair lands after layer l has already read the map." % (w1, w1 / exp_us, 2 * w1, 2 * w1 / exp_us, 2 * exp_us))
        for p in prompts:
            h = hm[p]
            if h and h.get("miss_layers"):
                ovl_l = h["engine_ovl_us"] / R_["n_layer"]
                attn_l = h["engine_attn_us"] / (R_["n_layer"] - 1)
                stall_m = h["engine_stall_us"] / h["miss_layers"]
                short, long_ = ovl_l + attn_l, ovl_l + attn_l + stall_m
                frac = h["miss_layers"] / R_["n_layer"]
                out.append("- %s run, the period split by whether layer l-1 itself had misses (the host stall only exists then): overlap %.1f + attention/router %.1f = %.0f us "
                           "without (%.0f%% of layers), + stall %.1f = %.0f us with (%.0f%% of layers); mean %.0f us. One expert (%.0f us) fits both; two (%.0f us) fit neither." % (
                               p, ovl_l, attn_l, short, 100 * (1 - frac), stall_m, long_, 100 * frac, (1 - frac) * short + frac * long_, exp_us, 2 * exp_us))
    out.append("")
    rows = []
    cfgs = (("P-B", 256, 1), ("P-B", 256, 2), ("P-B", 256, 3), ("P-C", 12, 1), ("P-C", 12, 2), ("P-C", 12, 3), ("P-B", 8, 2), ("P-B", 16, 2))
    for (pred, pool, b) in cfgs:
        row = ["%s pool %s b=%d" % (pred if pred != "P-C" else "P-B", ("top-%d" % pool) if pool < 256 else "all", b)]
        src = "P-C" if pool == 12 else "P-B"
        for s in prompts + ["pooled"]:
            d = R_["derived"][s]["lead1"][src]
            copies = d["isscell%d@%d" % (b, pool)] * nt
            if s == "pooled":
                tok_us = float(np.mean([hm[p]["token_us"] for p in prompts if hm[p]])) if first else float("nan")
            else:
                tok_us = hm[s]["token_us"] if hm[s] else float("nan")
            link_us = copies * exp_us + args.promos_per_token * exp_us
            row.append("%.1f copies, %.1f ms (%s of the token with the 16 promotions); precision %s, coverage %s" % (
                copies, copies * exp_us / 1000.0, pct1(link_us / tok_us), f3(d["prec%d@%d" % (b, pool)]), f3(d["covfrac%d@%d" % (b, pool)])))
        rows.append(row)
    out.append(md_table(["configuration"] + [("%s per token" % s) for s in prompts + ["pooled"]], rows))
    return "\n".join(out)


def render_value(R_, prompts, args):
    out = ["### Modelled value of the covered misses (MODEL, expected, unproven; copies taken as free and instantaneous, as in G13-2)", ""]
    hm = {p: R_["prompts"][p].get("host_model") for p in prompts}
    if not all(hm.values()):
        return "\n".join(out + ["engine timeline missing: model not computed"])
    out.append("Model: a layer-step with m misses costs the host engine cg[m] us of CPU-graph time (the engine's own timeline, by miss count); covering c of the m misses "
               "removes cg[m] - cg[m-c]. The part of the host serve time that shows up as GPU stall is taken as the measured ratio stall/hserve of the engine's timeline; "
               "the result is divided by the engine's measured token time. Check of the model: summing cg over the replayed misses of all 40 layers gives the engine's "
               "own host-serve time per token:")
    for p in prompts:
        out.append("- %s: model %.0f us per step against the engine's measured %.0f us per token (ratio %.3f)" % (
            p, hm[p]["model_hserve_us_per_step"], hm[p]["engine_hserve_us"], hm[p]["model_hserve_us_per_step"] / hm[p]["engine_hserve_us"]))
    out.append("")
    rows = []
    for label, src, key in (("all misses of layers 1-39 covered (ceiling)", None, None),
                            ("oracle, 2 prefetches per layer", "oracle", "savstep2@256"),
                            ("P-B, 1 prefetch per layer", "P-B", "savstep1@256"),
                            ("P-B, 2 prefetches per layer", "P-B", "savstep2@256"),
                            ("P-C (top-12 pool), up to 1 per layer", "P-C", "savstep1@12"),
                            ("P-C (top-12 pool), up to 2 per layer", "P-C", "savstep2@12"),
                            ("P-B top-8 pool, up to 2 per layer", "P-B", "savstep2@8"),
                            ("P-B lead 2, 2 prefetches per layer", "P-B|lead2", "savstep2@256")):
        if src is not None and src.endswith("|lead2") and "lead2" not in R_["derived"][prompts[0]]:
            continue
        row = [label]
        for p in prompts:
            h = hm[p]
            if src is None:
                sav = h["model_hserve_layers1plus_us"]
            elif src.endswith("|lead2"):
                sav = R_["derived"][p]["lead2"]["P-B"][key]
            else:
                sav = R_["derived"][p]["lead1"][src][key]
            stall = sav * h["engine_stall_us"] / h["engine_hserve_us"]
            row.append("%.0f us host serve, %.0f us stall = %.1f%% of the %.0f us token" % (sav, stall, 100.0 * stall / h["token_us"], h["token_us"]))
        rows.append(row)
    out.append(md_table(["configuration"] + prompts, rows))
    return "\n".join(out)


def window_metrics(R_, sel, pred, a, b, warm, lead=1):
    """derive() over the steps [a, b) (absolute step numbers; only steps >= warm exist) of the prompts in `sel`, from the stored per-step sums."""
    nu = R_["n_used"]
    sums, ncell, nst = {}, 0, 0
    for p in sel:
        pr = R_["prompts"][p]["preds"]["%s|lead%d" % (pred, lead)]
        i0, i1 = max(a, warm) - warm, b - warm
        i1 = min(i1, pr["per_step"]["miss"].shape[0])
        if i1 <= i0:
            continue
        for k, v in pr["per_step"].items():
            sums[k] = sums.get(k, 0.0) + float(v[i0:i1].sum())
        ncell += len(pr["layers"]) * (i1 - i0)
        nst += i1 - i0
    return derive(sums, ncell, nst, nu) if nst else None


def render_warm_sensitivity(R_, prompts, args):
    """The verdict as a function of the warm-up cut (steps >= cut). Recomputed from the stored per-step sums, so it needs no extra run."""
    cuts = [c for c in (8, 16, 32, 48, 64, 96, 128, 192, 256) if c >= args.warm]
    out = ["### Sensitivity of the verdict to the warm-up cut (steps >= cut, lead 1; the pre-registered cut is steps >= 8)", ""]
    rows = []
    end = 10 ** 9
    for c in cuts:
        row = [c]
        flags = []
        for pred in ("P-A", "P-B", "P-C"):
            pool = P_C_POOL if pred == "P-C" else 256
            per_prompt = {p: window_metrics(R_, [p], pred, c, end, args.warm) for p in prompts}
            pooled = window_metrics(R_, prompts, pred, c, end, args.warm)
            if pooled is None or any(v is None for v in per_prompt.values()):
                flags.append("n/a")
                continue
            v = g13_verdict(per_prompt, pooled, key_prec="prec2@%d" % pool, key_cov="covfrac2@%d" % pool)
            flags.append("%s/%s" % ("PASS" if v["G13-1"] else "FAIL", "PASS" if v["G13-2"] else "FAIL"))
            if pred in ("P-B", "P-C"):
                row.append("%s (%s)" % (f3(pooled["prec2@%d" % pool]), ", ".join(f3(per_prompt[p]["prec2@%d" % pool]) for p in prompts)))
        pc = window_metrics(R_, prompts, "P-C", c, end, args.warm)
        row.append(f3(pc["covfrac2@12"]) if pc else "n/a")
        row.append(f3(pc["rec12"]) if pc else "n/a")
        row += flags
        rows.append(row)
    out.append(md_table(["cut", "P-B precision b=2: pooled (%s)" % ", ".join(prompts), "P-C precision b=2: pooled (%s)" % ", ".join(prompts), "P-C coverage b=2 pooled",
                         "recall@12 pooled", "P-A G13-1/G13-2", "P-B G13-1/G13-2", "P-C G13-1/G13-2"], rows))
    return "\n".join(out)


def render_windows(R_, prompts, args):
    """The lead-1 metrics by step window. The cache is still adapting during the 511 steps (the true-miss density falls), and precision at a fixed
    number of prefetches depends on that density, so a verdict taken at one warm-up cut is not stationary."""
    nu = R_["n_used"]
    wins = [(args.warm, 64), (64, 128), (128, 256), (256, 10 ** 9)]
    wins = [(a, b) for (a, b) in wins if b > a]
    out = ["### Stationarity: lead-1 metrics by step window (the cache keeps adapting; layers 1-%d)" % (R_["n_layer"] - 1), ""]

    def window_derived(sel, pred, a, b):
        return window_metrics(R_, sel, pred, a, b, args.warm)

    rows = []
    for scope in prompts + ["pooled"]:
        sel = prompts if scope == "pooled" else [scope]
        for (a, b) in wins:
            B = window_derived(sel, "P-B", a, b)
            if B is None:
                continue
            Cc = window_derived(sel, "P-C", a, b)
            O = window_derived(sel, "oracle", a, b)
            last = R_["prompts"][sel[0]]["steps"]
            rows.append([scope, "%d-%d" % (max(a, args.warm), min(b, last) - 1), f3(1.0 - B["miss_per_cell"] / nu), f3(B["miss_per_cell"]), f3(B["rec12"]),
                         f3(B["prec2@256"]), f3(B["covfrac2@256"]), f3(O["prec2@256"]), f3(Cc["prec2@12"]), f3(Cc["isscell2@12"]), f3(Cc["covfrac2@12"])])
    out.append(md_table(["scope", "steps", "hit rate (layers 1-39)", "true misses / cell", "P-B rec@12", "P-B precision b=2", "P-B coverage b=2", "oracle precision b=2",
                         "P-C precision b=2", "P-C issued / layer-step", "P-C coverage b=2"], rows))
    return "\n".join(out)


def render_tables(R_, prompts, args):
    """R_ is the results dict assembled in main()."""
    out = []
    out.append("## Tables (generated by tools/analyze-pregate.py; steps >= %d, layers = target layers of the lead)" % args.warm)
    # G13-0
    out.append("\n### G13-0 validity")
    rows = []
    for p in prompts:
        r = R_["prompts"][p]
        rep, ro, r32 = r["replay"], r["router"], r["router_float32"]
        rows.append([p, "%d / %d (%.4f%%)" % (ro["match"], ro["cells"], ro["pct"]), "%d / %d (%.4f%%)" % (r32["match"], r32["cells"], r32["pct"]),
                     "%.3f%% / %.3f%%" % (ro["shift_minus1_pct"], ro["shift_plus1_pct"]),
                     "%.4f" % rep["replay_rate"] if rep["replay_rate"] is not None else "n/a",
                     ("%.4f (%d/%d)" % (rep["engine_rate"], rep["engine_hits"], rep["engine_lookups"])) if rep["engine_rate"] is not None else "n/a",
                     ("%+.4f" % rep["diff_pp"]) if rep["diff_pp"] is not None else "n/a",
                     ("%d/%d" % (rep["steps_hits_equal"], r["steps"])) if "steps_hits_equal" in rep else "n/a",
                     ("%d/%d" % (rep["steps_prom_equal"], r["steps"])) if "steps_prom_equal" in rep else "n/a",
                     ("%d/%d" % (rep["steps_ml_equal"], r["steps"])) if "steps_ml_equal" in rep else "n/a",
                     ("replay %d requests, engine %d: surplus %d requests / %d misses (%s) = the final graph run, never ingested" % (
                         sum(rep["nmiss_hist_replay"]), sum(rep["nmiss_hist_engine"]), rep["requests_surplus"], rep["misses_surplus"],
                         ",".join(str(v) for v in rep["nmiss_hist_surplus"]))) if "requests_surplus" in rep else "n/a"])
    out.append(md_table(["prompt", "router top-8 set reproduced, float64", "same, float32", "shifted alignment (-1 / +1)", "replay hit rate", "engine hit rate",
                         "diff (pp)", "per-step hits equal", "per-step promotions equal", "per-step layers-with-misses equal", "host requests (engine vs replay)"], rows))
    for p in prompts:
        r32 = R_["prompts"][p]["router_float32"]
        if r32["mismatches"]:
            out.append("")
            out.append("%s float32 router mismatches (step, layer, logit gap between 8th and 9th expert in float32 / float64): %s; median gap of all cells %.3f" % (
                p, "; ".join("(%d, %d, %.2e / %.2e)" % (m["step"], m["layer"], m["gap_8th_9th"], m["gap_8th_9th_float64"]) for m in r32["mismatches"]),
                R_["prompts"][p]["router"]["gap_median_all"]))
    out.append("")
    out.append(render_verdict(R_, prompts, args))
    for lead in args.leads:
        pdata = {}
        for p in prompts + ["pooled"]:
            pdata[p] = R_["derived"][p]["lead%d" % lead]
        out.append("")
        out.append(render_pred_tables(pdata, prompts, "Lead %d (layer l predicted from the input of layer l-%d; target layers %d..%d)" % (lead, lead, lead, R_["n_layer"] - 1)))
    out.append("")
    out.append(render_cause(R_, prompts))
    if 1 in args.leads:
        out.append("")
        out.append(render_windows(R_, prompts, args))
        out.append("")
        out.append(render_warm_sensitivity(R_, prompts, args))
    # pool sweep
    out.append("\n### Candidate-pool sweep, P-B lead 1, b = 2 (reported, not gated)")
    rows = []
    for pool in POOLS:
        row = ["top-%d" % pool if pool < 256 else "no limit"]
        for p in prompts + ["pooled"]:
            d = R_["derived"][p]["lead1"]["P-B"]
            row.append("%s / %s / %s" % (f3(d["prec2@%d" % pool]), f3(d["covfrac2@%d" % pool]), f3(d["isscell2@%d" % pool])))
        rows.append(row)
    out.append(md_table(["pool"] + ["%s: precision / coverage / issued per layer-step" % p for p in prompts + ["pooled"]], rows))
    # coverage curve
    out.append("\n### Coverage curve, P-B lead 1, no pool limit (reported, not gated)")
    rows = []
    for b in BS_EXT:
        row = [b]
        for p in prompts + ["pooled"]:
            d = R_["derived"][p]["lead1"]["P-B"]
            o = R_["derived"][p]["lead1"]["oracle"]
            row.append("%s / %s / %s" % (f3(d["prec%d@256" % b]), f3(d["covfrac%d@256" % b]), f3(o["covfrac%d@256" % b])))
        rows.append(row)
    out.append(md_table(["b"] + ["%s: precision / coverage / oracle coverage" % p for p in prompts + ["pooled"]], rows))
    out.append("")
    out.append(render_link(R_, prompts, args))
    out.append("")
    out.append(render_value(R_, prompts, args))
    # per layer
    fa = args.fa_interval
    for p in prompts:
        out.append("\n### Per layer, %s prompt, lead 1 (DN = gated DeltaNet layer, FA = full attention layer; oracle = mean min(2, N)/2, the precision a perfect predictor reaches)" % p)
        rows = []
        pr = R_["prompts"][p]["preds"]
        for l in range(1, R_["n_layer"]):
            d = pr["P-B|lead1"]["per_layer"][l]
            o = pr["oracle|lead1"]["per_layer"][l]
            c = pr["P-C|lead1"]["per_layer"][l]
            sm = R_["prompts"][p]["similarity"]["lead1"][l - 1]
            rows.append([l, ("FA" if fa and (l + 1) % fa == 0 else "DN"), f3(d["miss_per_cell"]), f3(d["rec12"]), f3(d["prec1@256"]), f3(d["prec2@256"]),
                         f3(o["prec2@256"]), pct1(d["prec2@256"] / o["prec2@256"]) if o["prec2@256"] > 0 else "n/a", f3(d["covfrac2@256"]),
                         f3(c["prec2@12"]), f3(c["isscell2@12"]), f3(sm["cos_resid"]), f3(sm["logit_corr"])])
        out.append(md_table(["layer", "type", "misses/cell", "rec@12", "P-B prec b=1", "P-B prec b=2", "oracle prec b=2", "P-B / oracle", "P-B cover b=2",
                             "P-C prec b=2", "P-C issued b=2", "cos(resid l-1, l)", "logit corr"], rows))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------------------------------------------------------
def selftest():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print("  [%s] %s" % ("ok" if cond else "FAIL", msg))
        ok = ok and bool(cond)

    rng = np.random.default_rng(1234)
    print("selftest:")

    # 1. binary round trip
    with tempfile.TemporaryDirectory() as td:
        ids = rng.integers(0, 16, size=(5, 3, 4))
        for s in range(5):
            for l in range(3):
                ids[s, l] = rng.permutation(16)[:4]
        X = rng.standard_normal((5, 3, 8)).astype(np.float32)
        write_trace(os.path.join(td, "t.bin"), ids, 16)
        write_xdump(os.path.join(td, "x.bin"), X)
        tr = read_trace(os.path.join(td, "t.bin"))
        check(np.array_equal(tr["ids"], ids) and tr["n_layer"] == 3 and tr["n_used"] == 4 and tr["n_expert"] == 16 and np.array_equal(tr["tok"], np.arange(5)),
              "trace write/read round trip")
        check(np.array_equal(read_xdump(os.path.join(td, "x.bin")), X), "xdump write/read round trip")
        with open(os.path.join(td, "bad.bin"), "wb") as fh:
            fh.write(b"NOTMAGIC" + b"\0" * 16)
        try:
            read_trace(os.path.join(td, "bad.bin"))
            check(False, "bad magic is refused")
        except ValueError:
            check(True, "bad magic is refused")

    # 2. replay against the literal port of the engine code
    nl, ne, nu, steps = 3, 24, 3, 90
    prior = rng.integers(1, 500, size=(nl, ne))
    prior[0, :5] = prior[0, 0]                                  # exact ties in the prior
    p_e = 1.0 / np.arange(1, ne + 1) ** 1.2
    all_equal = True
    for scan_mode in (0, 1, 2):
        for budget in (0, 3, 100):
            for use_waits in (False, True):
                ids = np.zeros((steps, nl, nu), dtype=np.int64)
                for s in range(steps):
                    for l in range(nl):
                        pp = np.roll(p_e, (s // 20) * 3 + l)
                        ids[s, l] = rng.choice(ne, size=nu, replace=False, p=pp / pp.sum())
                waits = (rng.random(steps) < 0.2).astype(int) if use_waits else None
                a = replay_policy(ids, prior, 7, lam=0.9, budget=budget, scan_mode=scan_mode, waits=waits)
                b = replay_reference(ids, prior, 7, lam=0.9, budget=budget, scan_mode=scan_mode, waits=waits)
                same = np.array_equal(a["hits"], b["hits"]) and np.array_equal(a["prom"], b["prom"]) and np.array_equal(a["M"], b["M"])
                if not same:
                    all_equal = False
                    print("    mismatch: scan_mode %d budget %d waits %s" % (scan_mode, budget, use_waits))
    check(all_equal, "vectorised replay == literal engine port (hits, promotions, published map) for scan modes 0/1/2, B 0/3/100, with and without waits")
    ids = np.zeros((steps, nl, nu), dtype=np.int64)
    for s in range(steps):
        for l in range(nl):
            ids[s, l] = rng.choice(ne, size=nu, replace=False, p=p_e / p_e.sum())
    a = replay_policy(ids, prior, 7, lam=0.9, budget=3, scan_mode=2)
    check(np.all((a["M"].sum(axis=2) + a["PD"].sum(axis=2)) == 7), "published + pending residents == slots at every step and layer")
    check(a["prom"][0] == 0 and a["prom"][1:].max() <= 3, "lag-1: no promotion at the first call, never more than B afterwards")
    # lag-1 semantics on a hand-made case: the promotion decided at call #s+1 from the EMA through step s-1
    prior2 = np.array([[10, 9, 8, 1, 1, 1]])
    ids2 = np.array([[[5]]] * 6)                                 # expert 5 (uncached, 2 slots) is selected every step
    r2 = replay_policy(ids2, prior2, 2, lam=0.5, budget=1, scan_mode=2)
    check(list(r2["hits"]) == [0, 0, 0, 1, 1, 1] and not r2["M"][1, 0, 5] and not r2["M"][2, 0, 5] and r2["M"][3, 0, 5] and r2["PD"][2, 0, 5] and list(r2["prom"][:3]) == [0, 1, 0],
          "lag-1 timing: expert 5 scanned after step 0, applied at call #2 (victim leaves at once), in flight for step 2, published for step 3")
    r2b = replay_policy(ids2, prior2, 2, lam=0.5, budget=1, scan_mode=0)
    check(list(r2b["hits"]) == [0, 0, 1, 1, 1, 1], "mode 0 (inline scan) is one step earlier than lag 1")

    # 3. metrics against a brute-force loop
    S, NE, nu = 40, 30, 4
    Z = rng.standard_normal((S, NE))
    Z[:, :3] = 0.0                                               # ties
    Tids = np.stack([rng.permutation(NE)[:nu] for _ in range(S)])
    Mres = rng.random((S, NE)) < 0.35
    PDl = (~Mres) & (rng.random((S, NE)) < 0.1)
    cm = cell_metrics(Z, Tids, Mres, PDl, nu, pools=(6, 12, 30), bs=(1, 2, 3, 4), ks=(4, 8), mks=(4, 8))
    good = True
    for s in range(S):
        order = sorted(range(NE), key=lambda e: (-Z[s, e], e))
        tset = set(int(e) for e in Tids[s])
        missset = set(e for e in tset if not Mres[s, e])
        good &= cm["miss"][s] == len(missset) and cm["nonres"][s] == int((~Mres[s]).sum())
        for k in (4, 8):
            good &= cm["rec%d" % k][s] == len(tset & set(order[:k]))
            good &= cm["mrec%d" % k][s] == len(missset & set(order[:k]))
        for pool in (6, 12, 30):
            for b in (1, 2, 3, 4):
                cand = [e for e in order[:pool] if not Mres[s, e]][:b]
                good &= cm["iss%d@%d" % (b, pool)][s] == len(cand)
                good &= cm["cov%d@%d" % (b, pool)][s] == len([e for e in cand if e in missset])
                good &= cm["pcov%d@%d" % (b, pool)][s] == len([e for e in cand if e in missset and PDl[s, e]])
    check(good, "cell_metrics == brute force (ties, pools 6/12/30, b 1..4, recall, miss recall, pending share)")
    # oracle / reversed oracle / chance
    Tind = np.zeros((S, NE))
    np.put_along_axis(Tind, Tids, 1.0, axis=1)
    mo = cell_metrics(Tind, Tids, Mres, PDl, nu, pools=(NE,), bs=(1, 2, 3, 4), ks=(4,), mks=(4,))
    nmiss = mo["miss"]
    check(all(np.array_equal(mo["cov%d@%d" % (b, NE)], np.minimum(nmiss, b)) for b in (1, 2, 3, 4)) and np.all(mo["rec4"] == nu),
          "oracle: coverage = min(b, true misses), recall = 1")
    mr = cell_metrics(-Tind, Tids, Mres, PDl, nu, pools=(NE,), bs=(2,), ks=(4,), mks=(4,))
    check(mr["cov2@%d" % NE].sum() <= mo["cov2@%d" % NE].sum() and mr["rec4"].sum() < mo["rec4"].sum(), "reversed oracle is worse than the oracle")
    mc = chance_cell_metrics(Tids, Mres, nu, bs=(2,), ks=(4,), mks=(4,))
    sim = np.zeros(S)
    trials = 2000
    for _ in range(trials):
        sim += cell_metrics(rng.standard_normal((S, NE)), Tids, Mres, PDl, nu, pools=(NE,), bs=(2,), ks=(4,), mks=(4,))["cov2@%d" % NE]
    check(abs(sim.sum() / trials - mc["cov2@256"].sum()) < 0.05 * max(1.0, mc["cov2@256"].sum()), "chance(analytic) agrees with a Monte-Carlo random ranking")
    # derive() on the totals
    arr = evaluate_ranker(lambda l, lead: Z, [0, 1], 1, np.stack([Tids] * 2, axis=1), np.stack([Mres] * 2, axis=1), np.stack([PDl] * 2, axis=1), 0, nu)
    red = reduce_cells(arr)
    d = derive(red["total"], 2 * S, S, nu)
    check(abs(d["prec2@256"] - red["total"]["cov2@256"] / red["total"]["iss2@256"]) < 1e-12 and abs(d["covfrac2@256"] - red["total"]["cov2@256"] / red["total"]["miss"]) < 1e-12,
          "derive(): precision = covered / issued, coverage = covered / misses")

    # 4. P-B algebra: with an unchanged residual stream, P-B on layer l-1's dump equals the router on layer l's dump
    n_embd, eps = 64, 1e-6
    h = rng.standard_normal((10, n_embd)) * 3.0
    w1 = rng.uniform(0.2, 2.0, n_embd)
    w2 = rng.uniform(0.2, 2.0, n_embd)
    nrm = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + eps)
    y1, y2 = nrm * w1, nrm * w2
    Rr = rng.standard_normal((16, n_embd))
    lg_true = router_logits(y2, Rr)
    lg_b = router_logits(renorm_input(y1, w1, w2, eps), Rr)
    lg_a = router_logits(y1, Rr)
    check(np.allclose(lg_b, lg_true, rtol=1e-5, atol=1e-5) and not np.allclose(lg_a, lg_true, rtol=1e-3),
          "P-B recovers the next layer's input (to the eps of the second RMS-normalisation) when the residual is unchanged; P-A does not")
    wz = w1.copy()
    wz[:3] = 0.0
    y_z = nrm * wz
    check(np.all(np.isfinite(renorm_input(y_z, wz, w2, eps))), "P-B survives zero norm weights (guarded)")

    # 5. router validation
    Rr3 = rng.standard_normal((3, 16, n_embd))
    Xr = rng.standard_normal((12, 3, n_embd)).astype(np.float32)
    Tr = np.zeros((12, 3, 4), dtype=np.int64)
    for l in range(3):
        Tr[:, l] = np.argsort(-router_logits(Xr[:, l], Rr3[l]), axis=1, kind="stable")[:, :4]
    m, g = validate_router(Xr, Tr, Rr3, 4)
    check(m.all() and (g >= 0).all(), "router validation: exact on self-generated trace")
    Tr2 = Tr.copy()
    Tr2[4, 1, 0] = [e for e in range(16) if e not in Tr2[4, 1]][0]
    m2, _ = validate_router(Xr, Tr2, Rr3, 4)
    check((~m2).sum() == 1 and not m2[4, 1], "router validation: one corrupted cell is found")
    check(shifted_match_pct(Xr, Tr, Rr3, 4, 1) < 50.0, "shifted alignment does not match")

    # 5b. engine log parsers (NULs in the runner log, per-step jsonl)
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "e.log")
        with open(lp, "wb") as fh:
            fh.write(b"junk\x00\x00 resize: moe_hybrid: cache resized to 81 slots x 40 layers (5598.4 MiB VRAM) in 574.0 ms\n"
                     b"moe_hybrid: on_token breakdown (scan mode 2), avg us\n"
                     b"moe_hybrid: tokens 511, hit rate 0.8213 (134295/163520), promotions/token 15.97\n")
        el = read_engine_log(lp)
        check(el["slots"] == 81 and el["scan_mode"] == 2 and el["hits"] == 134295 and el["lookups"] == 163520 and abs(el["hit_rate"] - 0.8213) < 1e-12,
              "engine log parser (slots, scan mode, hit rate)")
        jp = os.path.join(td, "h.jsonl")
        with open(jp, "w") as fh:
            fh.write('{"event":"config","slots":14,"B":16,"lambda":0.96999999999999997}\n{"t":1,"hits":5,"lookups":8,"prom":2,"wait":0}\n'
                     '{"t":0,"hits":4,"lookups":8,"prom":0,"wait":0}\n{"event":"summary","hits":9,"lookups":16}\n')
        ej = read_engine_jsonl(jp)
        check([r["t"] for r in ej["steps"]] == [0, 1] and ej["config"]["lambda"] == 0.97 and ej["summary"]["hits"] == 9, "engine jsonl parser (sorted steps, config, summary)")

    # 6. bootstrap
    i1 = block_bootstrap_idx(100, 8, 50, 7)
    i2 = block_bootstrap_idx(100, 8, 50, 7)
    check(np.array_equal(i1, i2) and i1.shape == (50, 100) and i1.min() >= 0 and i1.max() < 100, "block bootstrap is deterministic and in range")
    ps = {"num": rng.random(100), "den": np.ones(100)}
    lo, hi = bootstrap_ci([ps], [i1], lambda s: s["num"] / s["den"])
    check(lo <= ps["num"].mean() <= hi, "bootstrap interval contains the point estimate")

    # 7. verdict logic
    def dd(rec, prec, cov):
        return {"rec12": rec, "prec2@256": prec, "covfrac2@256": cov}
    v = g13_verdict({"code": dd(0.8, 0.45, 0.5), "prose": dd(0.9, 0.45, 0.5)}, dd(0.85, 0.45, 0.5))
    check(v["G13-1"] and v["G13-2"], "verdict: all thresholds met -> PASS")
    v = g13_verdict({"code": dd(0.8, 0.34, 0.5), "prose": dd(0.9, 0.50, 0.5)}, dd(0.85, 0.42, 0.5))
    check(not v["G13-1"] and v["G13-2"], "verdict: pooled 0.42 but code 0.34 < 0.35 -> G13-1 FAIL (each-prompt floor)")
    v = g13_verdict({"code": dd(0.8, 0.30, 0.2), "prose": dd(0.9, 0.35, 0.3)}, dd(0.85, 0.33, 0.25))
    check(not v["G13-1"] and not v["G13-2"], "verdict: precision 0.33 -> G13-1 FAIL; code coverage 0.20 -> G13-2 FAIL (each prompt)")
    v = g13_verdict({"code": dd(0.69, 0.5, 0.5), "prose": dd(0.9, 0.5, 0.5)}, dd(0.8, 0.5, 0.5))
    check(not v["G13-1"], "verdict: recall@12 0.69 on one prompt -> G13-1 FAIL")
    # 8. end to end on a synthetic session laid out like the real one (exercises every parser, the replay, the predictors, the bootstrap, the verdict and the report)
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as td:
        nl_, ne_, nu_, nemb_, steps_, slots_ = 6, 32, 4, 64, 120, 8
        Rr_ = (rng.standard_normal((nl_, ne_, nemb_)) * 0.3).astype(np.float32)
        Wn_ = rng.uniform(0.5, 1.5, (nl_, nemb_)).astype(np.float32)
        prior_ = rng.integers(1, 800, size=(nl_, ne_))
        np.savez(os.path.join(td, "w.npz"), router=Rr_, norm=Wn_, eps=np.float32(1e-6), fa_interval=np.int32(4))
        with open(os.path.join(td, "aer-freq-train.txt"), "w") as fh:
            for l in range(nl_):
                fh.write(" ".join(str(int(v)) for v in prior_[l]) + "\n")
        for pi, pname in enumerate(("code", "prose")):
            d = os.path.join(td, "g28-cap-" + pname)
            os.makedirs(d)
            h = rng.standard_normal((steps_, nemb_)) * 2.0
            X_ = np.zeros((steps_, nl_, nemb_), dtype=np.float32)
            for l in range(nl_):
                h = h + 0.4 * rng.standard_normal((steps_, nemb_))
                X_[:, l] = (h / np.sqrt((h * h).mean(axis=1, keepdims=True) + 1e-6)) * Wn_[l]
            T_ = np.zeros((steps_, nl_, nu_), dtype=np.int64)
            for l in range(nl_):
                T_[:, l] = np.argsort(-router_logits(X_[:, l], Rr_[l]), axis=1, kind="stable")[:, :nu_]
            write_xdump(os.path.join(d, "xdump_%s.bin" % pname), X_)
            write_trace(os.path.join(d, "trace_%s.bin" % pname), T_, ne_)
            ref = replay_reference(T_, prior_, slots_, lam=0.9, budget=5, scan_mode=2)
            miss_ = np.zeros((steps_, nl_, ne_), dtype=bool)
            np.put_along_axis(miss_, T_, True, axis=2)
            miss_ &= ~ref["M"]
            hist_ = np.bincount(miss_.sum(axis=2).ravel(), minlength=nu_ + 1)
            with open(os.path.join(d, "hybrid_PLAIN.jsonl"), "w") as fh:
                fh.write(json.dumps({"event": "config", "slots": 14, "B": 5, "lambda": 0.9}) + "\n")
                for s_ in range(steps_):
                    fh.write(json.dumps({"t": s_, "hits": int(ref["hits"][s_]), "lookups": nl_ * nu_, "prom": int(ref["prom"][s_]), "wait": 0,
                                         "ml": int((miss_[s_].sum(axis=1) > 0).sum())}) + "\n")
                fh.write(json.dumps({"event": "summary", "tokens": steps_, "hits": int(ref["hits"].sum()), "lookups": steps_ * nl_ * nu_,
                                     "nmiss_hist": [0] + [int(v) for v in hist_[1:]]}) + "\n")
                fh.write(json.dumps({"event": "timeline", "span_us": 7000.0, "attn_us": 3500.0, "ovl_us": 1500.0, "stall_us": 2000.0, "gap_us": 1400.0, "hserve_us": 3000.0,
                                     "cpu_graph_us": [75.0, 117.0, 167.0, 206.0]}) + "\n")
            with open(os.path.join(d, "g28-cap-%s-PLAIN.log" % pname), "wb") as fh:
                fh.write(("resize: moe_hybrid: cache resized to %d slots x %d layers (1.0 MiB VRAM) in 1.0 ms\n"
                          "moe_hybrid: on_token breakdown (scan mode 2), avg us per token: x\n"
                          "moe_hybrid: tokens %d, hit rate %.4f (%d/%d), promotions/token 1.0\n" % (
                              slots_, nl_, steps_, ref["hits"].sum() / (steps_ * nl_ * nu_), ref["hits"].sum(), steps_ * nl_ * nu_)).encode())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["--root", td, "--weights-cache", os.path.join(td, "w.npz"), "--out", os.path.join(td, "out"), "--boot", "30", "--boot-block", "8", "--warm", "4"])
        res_ = json.load(open(os.path.join(td, "out", "results.json")))
        tabs_ = open(os.path.join(td, "out", "tables.md")).read()
        check(rc in (0, 1, 3), "end to end: the analysis ran and G13-0 was valid (exit status %d)" % rc)
        check(all(v["router_ok"] and v["replay_ok"] for v in res_["g13_0"].values()), "end to end: router reproduced and the replay equals the synthetic engine log")
        check(all(res_["prompts"][p]["replay"]["requests_surplus"] == 0 and res_["prompts"][p]["replay"]["steps_hits_equal"] == steps_ and
                  res_["prompts"][p]["replay"]["steps_ml_equal"] == steps_ for p in ("code", "prose")),
              "end to end: per-step hits, per-step miss layers and the request histogram equal the engine log")
        pooled_ = res_["derived"]["pooled"]["lead1"]
        check(pooled_["oracle"]["rec12"] == 1.0 and pooled_["P-B"]["prec2@256"] <= pooled_["oracle"]["prec2@256"] + 1e-12 and pooled_["chance"]["prec2@256"] < pooled_["P-B"]["prec2@256"],
              "end to end: chance < P-B <= oracle on correlated synthetic layers")
        check("### Verdict against the pre-registered criteria" in tabs_ and "### Cause analysis" in tabs_ and "### Lead time and PCIe link budget" in tabs_ and "### Modelled value" in tabs_ and "### Stationarity" in tabs_,
              "end to end: the report sections are rendered")
        check(res_["ci"]["P-B"]["pooled"]["prec2"][0] <= pooled_["P-B"]["prec2@256"] <= res_["ci"]["P-B"]["pooled"]["prec2"][1] + 0.05, "end to end: bootstrap interval brackets the pooled precision")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc2 = main(["--root", td, "--weights-cache", os.path.join(td, "w.npz"), "--out", os.path.join(td, "out2"), "--boot", "0", "--slots", "9"])
        check(rc2 == 2, "end to end: a wrong slot count breaks the replay check and is reported as invalid (exit 2), not silently accepted")
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    ap.add_argument("--root", default=os.path.join(REPO, "build", "llama-cuda"), help="build/llama-cuda directory")
    ap.add_argument("--prompts", default="code,prose")
    ap.add_argument("--session-prefix", default="g28-cap-")
    ap.add_argument("--arm", default="PLAIN")
    ap.add_argument("--prior", default=None, help="frequency prior (default <root>/aer-freq-train.txt)")
    ap.add_argument("--model", default="C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf")
    ap.add_argument("--weights-cache", default=None, help="npz with the router and norm weights (default <out>/weights.npz)")
    ap.add_argument("--rebuild-cache", action="store_true", help="re-read the weights from the GGUF")
    ap.add_argument("--out", default=None, help="output directory (default <root>/g13-pregate)")
    ap.add_argument("--leads", default="1,2")
    ap.add_argument("--warm", type=int, default=8, help="steps skipped while the cache warms up")
    ap.add_argument("--slots", type=int, default=0, help="cache slots per layer (default: from the engine log, else 81)")
    ap.add_argument("--scan-mode", type=int, default=None, help="engine scan mode (default: from the engine log, else 2)")
    ap.add_argument("--fa-interval", type=int, default=0, help="full-attention interval (default: from the GGUF / cache)")
    ap.add_argument("--boot", type=int, default=500, help="block-bootstrap resamples (0 = off)")
    ap.add_argument("--boot-block", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--expert-us", type=float, default=117.0, help="link time of one expert (1.69 MiB at the measured 14.4 GB/s), for the link-budget table")
    ap.add_argument("--promos-per-token", type=int, default=16, help="engine promotions per token over the same link, for the link-budget table")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    args.leads = [int(x) for x in args.leads.split(",") if x]
    prompts = [p for p in args.prompts.split(",") if p]
    out_dir = args.out or os.path.join(args.root, "g13-pregate")
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t_start = time.time()
    prior_path = args.prior or os.path.join(args.root, "aer-freq-train.txt")
    paths = {}
    for p in prompts:
        d = os.path.join(args.root, "%s%s" % (args.session_prefix, p))
        paths[p] = dict(xdump=os.path.join(d, "xdump_%s.bin" % p), trace=os.path.join(d, "trace_%s.bin" % p),
                        jsonl=os.path.join(d, "hybrid_%s.jsonl" % args.arm), log=os.path.join(d, "%s%s-%s.log" % (args.session_prefix, p, args.arm)))
        for k in ("xdump", "trace"):
            if not os.path.exists(paths[p][k]):
                log("missing input: %s" % paths[p][k])
                return 2
    if not os.path.exists(prior_path):
        log("missing input: %s" % prior_path)
        return 2

    first = read_trace(paths[prompts[0]]["trace"])
    nl, ne = first["n_layer"], first["n_expert"]
    prior = read_prior(prior_path, nl, ne)
    cache = args.weights_cache or os.path.join(out_dir, "weights.npz")
    R, W, eps, fa, wsrc = load_weights(cache, args.model, nl, rebuild=args.rebuild_cache)
    args.fa_interval = args.fa_interval or fa
    log("weights: router %s, norm %s, eps %g, full-attention interval %d (%s)" % (R.shape, W.shape, eps, args.fa_interval, wsrc))
    if R.shape[0] != nl or R.shape[1] != ne:
        log("weights do not match the trace shape")
        return 2

    results = {"prompts": {}, "n_layer": nl, "n_expert": ne, "n_used": first["n_used"]}
    for p in prompts:
        log("prompt %s" % p)
        results["prompts"][p] = analyze_prompt(p, paths[p], W, R, eps, prior, args, log)

    # ---- pooled and per-prompt derived metrics, CIs, verdict
    results["derived"] = {}
    pooled_pred = {}
    for lead in args.leads:
        for pred in PREDICTORS:
            key = "%s|lead%d" % (pred, lead)
            sums = {}
            ncell = nsteps = 0
            extra = 0.0
            for p in prompts:
                pr = results["prompts"][p]["preds"][key]
                for k, v in pr["sums"].items():
                    sums[k] = sums.get(k, 0.0) + v
                ncell += pr["ncell"]
                nsteps += pr["per_step"]["miss"].shape[0]
                extra += pr["extra_miss"]
            pooled_pred[key] = derive(sums, ncell, nsteps, first["n_used"], extra)
    for p in prompts + ["pooled"]:
        results["derived"][p] = {}
        for lead in args.leads:
            results["derived"][p]["lead%d" % lead] = {}
            for pred in PREDICTORS:
                key = "%s|lead%d" % (pred, lead)
                results["derived"][p]["lead%d" % lead][pred] = pooled_pred[key] if p == "pooled" else results["prompts"][p]["preds"][key]["derived"]

    # 95% block-bootstrap intervals (resampling steps) for the headline metrics (lead 1)
    results["ci"] = {}
    if args.boot > 0 and 1 in args.leads:
        for pred in ("P-A", "P-B", "P-C"):
            results["ci"][pred] = headline_ci(results, prompts, pred, first["n_used"])

    # verdict for each pre-registered predictor variant (lead 1)
    results["verdict"] = {}
    if 1 in args.leads:
        for pred in ("P-A", "P-B", "P-C"):
            pool = P_C_POOL if pred == "P-C" else 256
            per_prompt = {p: results["derived"][p]["lead1"][pred] for p in prompts}
            v = g13_verdict(per_prompt, results["derived"]["pooled"]["lead1"][pred], key_prec="prec2@%d" % pool, key_cov="covfrac2@%d" % pool)
            results["verdict"][pred] = v

    # G13-0 verdict
    g0 = {}
    for p in prompts:
        r = results["prompts"][p]
        hit_ok = r["replay"]["diff_pp"] is not None and abs(r["replay"]["diff_pp"]) <= G13_0_HIT_TOL_PP
        g0[p] = dict(router_ok=r["router"]["pct"] >= G13_0_ROUTER_MIN_PCT, replay_ok=bool(hit_ok))
    results["g13_0"] = g0
    g13_0_ok = all(v["router_ok"] and v["replay_ok"] for v in g0.values())

    # ---- provenance and outputs
    prov = {"script": os.path.relpath(os.path.abspath(__file__), REPO), "argv": (argv if argv is not None else sys.argv[1:]), "numpy": np.__version__, "python": sys.version.split()[0],
            "elapsed_s": time.time() - t_start, "weights_source": wsrc, "inputs": {}}
    for p in prompts:
        prov["inputs"][p] = {k: dict(path=v, bytes=os.path.getsize(v), sha256=sha256_file(v)) for k, v in paths[p].items() if os.path.exists(v)}
    prov["inputs"]["prior"] = dict(path=prior_path, sha256=sha256_file(prior_path))
    prov["inputs"]["weights_cache"] = dict(path=cache, sha256=sha256_file(cache))
    results["provenance"] = prov
    results["criteria"] = dict(G13_0_HIT_TOL_PP=G13_0_HIT_TOL_PP, G13_0_ROUTER_MIN_PCT=G13_0_ROUTER_MIN_PCT, G13_1_RECALL12_MIN=G13_1_RECALL12_MIN,
                               G13_1_PREC2_MIN=G13_1_PREC2_MIN, G13_1_PREC2_EACH_MIN=G13_1_PREC2_EACH_MIN, G13_2_COVER2_MIN=G13_2_COVER2_MIN,
                               headline_predictor=HEADLINE_PREDICTOR, warm=args.warm, leads=args.leads)

    def jsonable(o):
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_") and str(k) not in ("per_step",)}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(jsonable(results), fh, indent=1)
    tables = render_tables(results, prompts, args)
    with open(os.path.join(out_dir, "tables.md"), "w") as fh:
        fh.write(tables + "\n")

    # ---- console summary
    log("")
    log("G13-0: " + ", ".join("%s router %s replay %s" % (p, "OK" if v["router_ok"] else "FAIL", "OK" if v["replay_ok"] else "FAIL") for p, v in g0.items()))
    if 1 in args.leads:
        for pred in ("P-A", "P-B", "P-C"):
            pool = P_C_POOL if pred == "P-C" else 256
            v = results["verdict"][pred]
            parts = []
            for p in prompts + ["pooled"]:
                d = results["derived"][p]["lead1"][pred]
                parts.append("%s rec12 %.3f prec2 %.3f cover2 %.3f" % (p, d["rec12"], d["prec2@%d" % pool], d["covfrac2@%d" % pool]))
            log("lead 1 %s%s: %s => G13-1 %s, G13-2 %s" % (pred, " (headline)" if pred == HEADLINE_PREDICTOR else "", " | ".join(parts),
                                                          "PASS" if v["G13-1"] else "FAIL", "PASS" if v["G13-2"] else "FAIL"))
    log("outputs: %s, %s (%.1f s)" % (os.path.join(out_dir, "results.json"), os.path.join(out_dir, "tables.md"), time.time() - t_start))
    if not g13_0_ok:
        return 2
    if 1 in args.leads:
        passes = [results["verdict"][pred]["G13-1"] and results["verdict"][pred]["G13-2"] for pred in ("P-A", "P-B", "P-C")]
        if all(passes):
            return 0
        if not any(passes):
            return 1
        log("NOTE: the verdict depends on the predictor variant / candidate-pool reading (exit 3): %s" % ", ".join(
            "%s %s" % (pred, "PASS" if ok else "FAIL") for pred, ok in zip(("P-A", "P-B", "P-C"), passes)))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
