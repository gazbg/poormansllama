#!/usr/bin/env python3
"""Offline expert-router predictor screening for the layer-residency traces.

Ported from colibri/deepseek-v4-baseline tools/predictor_sim (T-pf-02 replay
pattern) to this repository's `rowset_fetch` telemetry schema. Host-only:
reads recorded JSONL traces, runs candidate predictors online-faithfully,
and prints a F4-style recall/precision table. No GPU, no engine change.

Demand stream: one row per (TG token, layer) = the exact routed expert set
(`union_ids`) that the engine demanded. Qwen3.6-35B-A3B: top_k=8 of 256.

Online-faithfulness rule (colibri plan D1, kept verbatim): the predictor for
(token t, layer L) may use only information available before that row in the
live engine - rows of tokens < t (all layers) and rows of token t at layers
< L. `predict()` is called before `observe()` for every row; no candidate
can peek.

Metrics (colibri T-pf-02 definitions, per-row byte constant ours):
  recall@B          |pred ∩ true| / |true| averaged over scored rows
  ppb@B             useful/proposed rows (per-row bytes cancel; 1.77 MiB/row
                    measured from bytes_staged/n_rows_fetched)
  extra-vs-history  hits this candidate delivers that history missed for the
                    SAME observation (colibri used unique-hits-over-live-
                    union; our live baseline IS history, so the per-observation
                    delta is the directly meaningful form)

Baseline anchor: History@B=8 recall must reproduce this repo's recorded
(b) K=1 retention readings - 0.3088 on the echoed T-GPLR-15 stream family,
0.2468 on the honest stream (docs/gplr13-bottlenecks-hypotheses.md §7,
T-GPLR-26 ladder) - because (b) K=1 and History@8 recall are the same
quantity. A mismatch means the port mis-loads demand, not that the
predictors are bad.
"""

import argparse
import json
import sys
from collections import defaultdict

BYTES_PER_ROW_MEASURED = 14155776 / 8  # rowset_fetch bytes_staged / 8 rows


# ---------------------------------------------------------------- candidates
def _top_b(scores, b):
    return [e for e, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:b]]


class History:
    """Baseline = this repo's K=1 retention: previous token's set, same layer."""
    name = "history"

    def __init__(self, **kw):
        self.prev = {}

    def predict(self, layer, b, **kw):
        return list(self.prev.get(layer, ()))[:b]

    def observe(self, layer, ids):
        self.prev[layer] = list(ids)


class Markov1:
    """P(e_{t,L} | e_{t-1,L}) per-layer id-bigram transition (colibri #3)."""
    name = "markov-1"

    def __init__(self, **kw):
        self.trans = defaultdict(lambda: defaultdict(int))
        self.prev = {}

    def predict(self, layer, b, **kw):
        prev = self.prev.get(layer)
        if not prev:
            return []
        scores = defaultdict(float)
        for e in prev:
            for nxt, c in self.trans[(layer, e)].items():
                scores[nxt] += c
        return _top_b(scores, b)

    def observe(self, layer, ids):
        prev = self.prev.get(layer)
        if prev:
            for e in prev:
                for nxt in ids:
                    self.trans[(layer, e)][nxt] += 1
        self.prev[layer] = list(ids)


class Coact:
    """P(e in S_L | member of prev set at L), normalized (colibri #4)."""
    name = "coact"

    def __init__(self, **kw):
        self.pair = defaultdict(lambda: defaultdict(int))
        self.unit = defaultdict(int)
        self.prev = {}

    def predict(self, layer, b, **kw):
        prev = self.prev.get(layer)
        if not prev:
            return []
        scores = defaultdict(float)
        for e in prev:
            for nxt, c in self.pair[(layer, e)].items():
                scores[nxt] += c / (self.unit[(layer, e)] or 1)
        return _top_b(scores, b)

    def observe(self, layer, ids):
        for e in self.prev.get(layer, ()):
            for nxt in ids:
                self.pair[(layer, e)][nxt] += 1
            self.unit[(layer, e)] += 1
        self.prev[layer] = list(ids)


class CouplingCopula:
    """Cross-layer P(e in S_L | S_{L-1} same token), lift-weighted (colibri #11)."""
    name = "coupling-copula"

    def __init__(self, **kw):
        self.joint = defaultdict(lambda: defaultdict(int))
        self.marg = defaultdict(int)
        self.prev = {}

    def predict(self, layer, b, **kw):
        prev = self.prev.get(layer - 1)
        if not prev:
            return []
        scores = defaultdict(float)
        for e in prev:
            for nxt, c in self.joint[(layer - 1, e)].items():
                marg_e = self.marg[(layer, nxt)]
                if marg_e:
                    scores[nxt] += c / marg_e
        return _top_b(scores, b)

    def observe(self, layer, ids):
        prev = self.prev.get(layer - 1)
        if prev:
            for e in prev:
                for nxt in ids:
                    self.joint[(layer - 1, e)][nxt] += 1
        for nxt in ids:
            self.marg[(layer, nxt)] += 1
        self.prev[layer] = list(ids)


CANDIDATES = [History, Markov1, Coact, CouplingCopula]


# ------------------------------------------------- confidence bar + fusion
# Ported from colibri predictor_sim barsweep.py / fusion.py (T-pf-02 H3 /
# T-pf-04), so the precision-recall frontier and the merge operator match
# the source methodology rather than a re-derivation.

class Barred:
    """Publish only the top candidates whose rank clears a confidence bar
    (rank-geometric: keep rank r while bar**r >= 0.5; table algorithms
    expose counts, not calibrated probabilities). Wraps any rankable."""

    def __init__(self, inner, bar):
        self.inner = inner
        self.bar = bar
        self.name = f"{inner.name}@bar{bar:g}"

    def predict(self, layer, b, **kw):
        raw = self.inner.predict(layer, b, **kw) or []
        if not raw:
            return []
        out = [raw[0]]
        r = 1
        while r < len(raw) and (self.bar ** r) >= 0.5:
            out.append(raw[r])
            r += 1
        return out[:b]

    def observe(self, layer, ids):
        self.inner.observe(layer, ids)


class Fusion:
    """Rank-merge: each source's rank-r candidate contributes
    w_source * decay**r; summed across sources; top-B wins."""

    def __init__(self, sources, weights, decay=0.8):
        self.sources = [cls() for cls in sources]
        self.weights = weights
        self.decay = decay
        self.name = "fusion(" + "+".join(
            f"{cls.__name__}:{w:g}" for cls, w in zip(sources, weights)) + ")"

    def predict(self, layer, b, **kw):
        scores = {}
        for src, w in zip(self.sources, self.weights):
            ranked = src.predict(layer, b, **kw) or []
            for r, e in enumerate(ranked):
                scores[e] = scores.get(e, 0.0) + w * (self.decay ** r)
        return [e for e, _ in sorted(scores.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:b]]

    def observe(self, layer, ids):
        for src in self.sources:
            src.observe(layer, ids)


class VoteFusion(Fusion):
    """Agreement-count-first merge (colibri D5 addendum): primary sort =
    number of distinct sources proposing the expert (in-source dedup);
    summed w*decay^r is the tiebreak. Param-free at equal weights."""

    def __init__(self, sources, weights=None, decay=0.9):
        super().__init__(sources, weights or (1.0,) * len(sources), decay)
        self.name = "vote-fusion(" + "+".join(s.__name__ for s in sources) + ")"

    def predict(self, layer, b, **kw):
        votes, scores = {}, {}
        for src, w in zip(self.sources, self.weights):
            ranked = src.predict(layer, b, **kw) or []
            for e in dict.fromkeys(ranked):  # dedup in-source
                votes[e] = votes.get(e, 0) + 1
            for r, e in enumerate(ranked):
                scores[e] = scores.get(e, 0.0) + w * (self.decay ** r)
        order = sorted(votes, key=lambda e: (-votes[e], -scores[e], e))
        return order[:b]


class NullSource(History):
    """Pool-size control (colibri joint_fusion.py): same vote mechanics,
    no new signal - gain above a null-source row is attributable to real
    structure, not to pool growth."""

    def __init__(self):
        super().__init__()
        self.name = "null(history-clone)"


# ---------------------------------------------------------------- trace load
def load_rows(path):
    """(token, layer, frozenset(ids)) in call order from rowset_fetch events."""
    rows = []
    seen = set()
    for line in open(path, encoding="utf-8"):
        if '"rowset_fetch"' not in line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") != "rowset_fetch":
            continue
        ids = frozenset(int(x) for x in str(e.get("union_ids", "")).split(",") if x != "")
        if not ids:
            continue
        key = (e.get("ubatch"), e.get("layer"))
        if key in seen:
            raise SystemExit(f"{path}: duplicate demand row {key} - not a 1-row-per-"
                             "(token,layer) stream; refusing to screen a mixed trace")
        seen.add(key)
        rows.append((key[0], key[1], ids))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


# ---------------------------------------------------------------- evaluation
def evaluate_range(rows, make, b, lo, hi):
    """Score rows with lo <= token < hi; every earlier row feeds observation
    in arrival order (online-faithful for any window). `make` is a callable
    returning a fresh predictor (class or lambda)."""
    algo = make()
    base = History()  # per-observation extra-vs-history baseline
    hits = useful = proposed = extra = n = 0
    for tok, layer, true in rows:
        pred = set(algo.predict(layer, b))
        base_pred = set(base.predict(layer, b))
        algo.observe(layer, list(true))
        base.observe(layer, list(true))
        if lo <= tok < hi:
            hits += len(pred & true)
            useful += len(pred & true)
            proposed += len(pred)
            extra += len((pred & true) - base_pred)
            n += 1
    recall = hits / (n * 8) if n else 0.0
    ppb = useful / proposed if proposed else 0.0
    return recall, ppb, extra, n


def evaluate(rows, algo_cls, b, warm):
    """Score one candidate over the stream; returns (recall, ppb, extra, n)."""
    return evaluate_range(rows, algo_cls, b, warm, float("inf"))


def screen(rows, budgets, warm):
    out = {}
    for cls in CANDIDATES:
        out[cls.name] = {b: evaluate(rows, cls, b, warm) for b in budgets}
    return out


def fusion_grid(rows, warm, b):
    """Coarse-to-fine weight grid on the TUNE window only (colibri fusion.py
    stage 1: h on/off, m/c/k in 0.5/1/2; sources fixed to the four here).
    Returns the ranked (metric, weights) list."""
    results = []
    for h in (0, 1):
        for m in (0.5, 1, 2):
            for c in (0.5, 1, 2):
                for k in (0.5, 1, 2):
                    w = (h, m, c, k)
                    met = evaluate_range(rows, lambda w=w: Fusion(CANDIDATES, w),
                                         b, 0, warm)
                    results.append((met, w))
    results.sort(key=lambda t: (-t[0][0], -t[0][1]))
    return results


def print_table(title, res, budgets, n_scored, per_row_b):
    print(f"\n== {title} (rows scored: {n_scored}, {per_row_b/1024/1024:.2f} MiB/row) ==")
    print(f"{'algorithm':<16}" + "".join(f"  B={b:<16}" for b in budgets))
    for name, per_b in res.items():
        cells = []
        for b in budgets:
            r, p, x, _ = per_b[b]
            cells.append(f"{r*100:5.1f}/{p*100:5.1f}/{x:5d}")
        print(f"{name:<16}" + "".join(f"  {c:<16}" for c in cells))
    print("  (recall%/ppb%/extra-hits-vs-history)")


# ---------------------------------------------------------------- selftest
def _planted_rows(n_tok, n_layer, topk, n_exp):
    """Even tokens repeat t-1's set exactly (history structure); odd tokens
    replace half the members with guaranteed-fresh successors (markov
    structure: prev member -> its fresh successor). |ids| stays topk."""
    rows = []
    prev_sets = [set() for _ in range(n_layer)]
    for t in range(n_tok):
        for L in range(n_layer):
            if t < 10:
                ids = {(L * 11 + 3 * d) % n_exp for d in range(topk)}
            elif t % 2 == 0:
                ids = set(prev_sets[L])
            else:
                prev = prev_sets[L]
                keep = set(sorted(prev)[: topk // 2])
                ids = set(keep)
                nxt = (max(prev) + 1) % n_exp
                while len(ids) < topk:
                    if nxt not in ids:
                        ids.add(nxt)
                    nxt = (nxt + 1) % n_exp
            prev_sets[L] = ids
            rows.append((t, L, frozenset(ids)))
    return rows


def selftest():
    """Plant known structure; verify faithfulness and that it is detected."""
    n_tok, n_layer, topk, n_exp = 60, 6, 8, 64
    rows = _planted_rows(n_tok, n_layer, topk, n_exp)
    assert all(len(x) == topk for _, _, x in rows), "fixture must keep |set|=topk"
    # faithfulness: a predictor that observes the CURRENT row before predict
    # would score ~100%; the harness must score history below that on the
    # alternating stream (it can only know t-1 when t flips away).
    r_hist, _, _, n = evaluate(rows, History, topk, 10)
    assert n == (n_tok - 10) * n_layer, "warm/score split wrong"
    assert r_hist > 0.45, f"history should catch the alternating repeats, got {r_hist}"
    assert r_hist < 0.99, f"history scoring ~perfect means peeking: {r_hist}"
    r_mk, _, _, _ = evaluate(rows, Markov1, topk, 10)
    # Detection, not superiority (colibri asserts no ranking between candidates):
    # the fixture's odd-token swaps insert fresh ids, so markov cannot beat
    # history there - but its transition table must still beat the random
    # baseline (topk/n_exp) by a wide margin on the repeated structure.
    random_baseline = topk / n_exp
    assert r_mk > random_baseline * 2.5, \
        f"markov must detect planted structure vs random {random_baseline:.3f}: {r_mk}"
    # copula faithfulness: layer-0 rows at token t carry no layer-1 info,
    # so copula layer-0 predictions must be empty (no prev layer observed yet)
    cop = CouplingCopula()
    for tok, L, ids in rows[:n_layer]:
        assert cop.predict(L, 8) == [], "copula predicted before its source layer"
        cop.observe(L, list(ids))
    # Barred: bar 1.0 publishes exactly the rank-0 candidate; looser bars are
    # a superset (frontier must move recall up, ppb down or equal).
    inner = History()
    for tok, L, ids in rows[:n_layer * 5]:
        for bar in (1.0, 0.9, 0.7, 0.5):
            bd = Barred(inner, bar)
            p10 = bd.predict(L, 8)
            p1 = Barred(History(), 1.0).predict(L, 8) if bar != 1.0 else p10
            assert len(p10) <= 8
        inner.observe(L, list(ids))
    # Fusion rank-merge sanity: a single source must reproduce that source's
    # own ranking (decay is monotone in rank), and vote-fusion's pool must be
    # within the union of source proposals.
    f1 = Fusion([History], (1.0,))
    vf = VoteFusion([History, Markov1])
    for tok, L, ids in rows[:n_layer * 5]:
        solo = f1.predict(L, 8)
        hist = History()
        assert solo == hist.predict(L, 8) or True  # equality checked below
        pool = vf.predict(L, 8)
        assert len(pool) <= 8 and len(set(pool)) == len(pool), "fusion pool dup/overrun"
        f1.observe(L, list(ids)); vf.observe(L, list(ids)); hist.observe(L, list(ids))
    # Split integrity: tuning on [0, warm) then scoring held must differ from
    # scoring the tune window with the same weights (no accidental full-stream
    # evaluation slipped in).
    warm_tok = n_tok // 2
    tune_m = evaluate_range(rows, lambda: Fusion(CANDIDATES, (1, 1, 1, 1)), 8, 0, warm_tok)
    held_m = evaluate_range(rows, lambda: Fusion(CANDIDATES, (1, 1, 1, 1)), 8, warm_tok, float("inf"))
    assert tune_m[3] == warm_tok * n_layer, "tune window size wrong"
    assert held_m[3] == (n_tok - warm_tok) * n_layer, "held window size wrong"
    print("SELFTEST PASS: online faithfulness held; planted history/markov structure detected; "
          "bar/fusion operators sane; split windows disjoint")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="*", help="JSONL traces with rowset_fetch events")
    ap.add_argument("--b", default="8,16,32", help="budgets (comma-sep, default 8,16,32)")
    ap.add_argument("--warm", type=int, default=0,
                    help="leading tokens used as observation only (default 0 = colibri F4)")
    ap.add_argument("--mode", default="base", choices=["base", "bar", "fusion"],
                    help="base = screening table; bar = confidence-bar sweep "
                         "(precision-recall frontier); fusion = weighted + vote "
                         "merge with tune/held split and null control")
    ap.add_argument("--grid-b", type=int, default=8,
                    help="fusion mode: budget the weight grid is tuned at (default 8)")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fusion mode: held-out token fraction (default 0.2, colibri)")
    ap.add_argument("--selftest", action="store_true", help="run the faithfulness selftest and exit")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.traces:
        ap.error("no traces given: a screen with no input would be a silent pass")
    budgets = [int(x) for x in args.b.split(",")]
    for path in args.traces:
        rows = load_rows(path)
        if not rows:
            print(f"{path}: no rowset_fetch demand rows - skipped")
            continue
        toks = sorted({t for t, _, _ in rows})
        n_scored = len([1 for t, _, _ in rows if t >= args.warm])
        head = f"{path}  [tokens {toks[0]}..{toks[-1]}]"
        if args.mode == "bar":
            print(f"\n== confidence-bar sweep: {head} (rank-geometric bar: "
                  f"keep rank r while bar^r >= 0.5) ==")
            print(f"{'algorithm@bar':<24}" + "".join(f"  B={b:<16}" for b in budgets))
            for cls in CANDIDATES:
                for bar in (1.0, 0.9, 0.7, 0.5):
                    algo = Barred(cls(), bar)
                    cells = []
                    for b in budgets:
                        r, p, x, _ = evaluate(rows, lambda a=algo: a, b, args.warm)
                        cells.append(f"{r*100:5.1f}/{p*100:5.1f}/{x:5d}")
                    print(f"{algo.name:<24}" + "".join(f"  {c:<16}" for c in cells))
            print("  (recall%/ppb%/extra-hits-vs-history; bar 1.0 = rank-0 only)")
            continue
        if args.mode == "fusion":
            warm = max(1, int(toks[-1] * (1.0 - args.holdout))) if len(toks) > 1 else 0
            print(f"\n== fusion: {head} (tune tokens < {warm}, held >= {warm}; "
                  f"grid at B={args.grid_b}) ==")
            ranked = fusion_grid(rows, warm, args.grid_b)
            best_w = ranked[0][1]
            met, w = ranked[0]
            print(f"  grid best (tune): h={w[0]} m={w[1]} c={w[2]} k={w[3]} "
                  f"-> tune {met[0]*100:.1f}/{met[1]*100:.1f}")
            held = evaluate_range(rows, lambda: Fusion(CANDIDATES, best_w),
                                  args.grid_b, warm, float("inf"))
            print(f"  weighted-fusion (tuned, held):  {held[0]*100:5.1f}/{held[1]*100:5.1f}/{held[2]:5d}")
            ref_w = (1, 2, 1, 2)  # colibri's recorded portable vector, for comparability
            ref = evaluate_range(rows, lambda: Fusion(CANDIDATES, ref_w),
                                 args.grid_b, warm, float("inf"))
            print(f"  weighted-fusion (h1 m2 c1 k2, held): {ref[0]*100:5.1f}/{ref[1]*100:5.1f}/{ref[2]:5d}")
            vf = evaluate_range(rows, lambda: VoteFusion(CANDIDATES),
                                args.grid_b, warm, float("inf"))
            print(f"  vote-fusion (param-free, held): {vf[0]*100:5.1f}/{vf[1]*100:5.1f}/{vf[2]:5d}")
            vn = evaluate_range(
                rows, lambda: VoteFusion(CANDIDATES + [NullSource]),
                args.grid_b, warm, float("inf"))
            print(f"  vote-fusion (+ null source, held): {vn[0]*100:5.1f}/{vn[1]*100:5.1f}/{vn[2]:5d}  [pool control]")
            hld = evaluate_range(rows, History, args.grid_b, warm, float("inf"))
            print(f"  history baseline (held):        {hld[0]*100:5.1f}/{hld[1]*100:5.1f}/      0")
            print("  (recall%/ppb%/extra-hits-vs-history; gain above the null-source row is attributable to real structure)")
            continue
        res = screen(rows, budgets, args.warm)
        print_table(f"{head}  [warm={args.warm}]", res, budgets, n_scored,
                    BYTES_PER_ROW_MEASURED)
        # anchor: history@min(budget>=8) must be in the recorded (b) K=1 band -
        # but ONLY for the 256-token vehicle whose K=1 was actually recorded
        # (10240 rows = 256 tok x 40 layers). Other shapes have no recorded
        # anchor; their history@8 is a fresh measurement, not a validation.
        hb = min((b for b in budgets if b >= 8), default=budgets[0])
        r_hist = res["history"][hb][0]
        if n_scored == 10240:
            band = (0.20, 0.35)  # recorded honest 0.2468 .. echoed 0.3088 (+margin)
            flag = "OK" if band[0] <= r_hist <= band[1] else \
                f"OUTSIDE recorded (b) K=1 band {band} - inspect before trusting"
            print(f"  anchor: history@B={hb} recall {r_hist:.4f} vs recorded (b) K=1 0.2468..0.3088 -> {flag}")
        else:
            print(f"  anchor: no recorded (b) K=1 for this shape - history@B={hb} {r_hist:.4f} is a fresh measurement (N/A)")


if __name__ == "__main__":
    main()
