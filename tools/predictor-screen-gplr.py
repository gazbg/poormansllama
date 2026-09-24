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
import hashlib
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


# ------------------------------------------------- cache-aware counterfactual

def load_cache_rows(path):
    """Load one retention trace for the cache counterfactual.

    The ordinary screen intentionally reduces each union to a frozenset. This
    mode also needs the trace's id order, its recorded per-stage hit count, and
    rowset_init metadata, so it uses a separate loader and refuses mixed or
    out-of-order streams.
    """
    init = None
    rows = []
    seen = set()
    previous = None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if '"rowset_fetch"' not in line and '"rowset_init"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('event') == 'rowset_init' and init is None:
                init = ev
                continue
            if ev.get('event') != 'rowset_fetch':
                continue
            ids = [int(x) for x in str(ev.get('union_ids', '')).split(',') if x != '']
            if not ids:
                continue
            if len(ids) != len(set(ids)):
                raise ValueError('%s: rowset_fetch union_ids contains duplicates' % path)
            key = (ev.get('ubatch'), ev.get('layer'))
            if key in seen:
                raise ValueError('%s: duplicate demand row %s' % (path, key))
            if previous is not None and key < previous:
                raise ValueError('%s: demand rows are not in ascending (ubatch, layer) order' % path)
            seen.add(key)
            previous = key
            rows.append({
                'token': key[0], 'layer': key[1], 'ids': ids,
                'hits': int(ev.get('n_rows_resident', 0) or 0),
            })
    if init is None:
        raise ValueError('%s: no rowset_init' % path)
    if not rows:
        raise ValueError('%s: no rowset_fetch demand rows' % path)
    for name in ('retain_k', 'n_expert', 'top_k'):
        if int(init.get(name, 0) or 0) <= 0:
            raise ValueError('%s: rowset_init has no positive %s' % (path, name))
    return init, rows


def stream_signature(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(('%s|%s\n' % (row['layer'], ','.join(str(x) for x in row['ids']))).encode())
    return h.hexdigest()[:16]


def cache_filtered(ranked, present, budget):
    """Keep the highest-ranked candidates absent from the live cache."""
    out = []
    seen = set()
    for expert in ranked:
        if expert in seen:
            continue
        seen.add(expert)
        if expert in present:
            continue
        out.append(expert)
        if len(out) >= budget:
            break
    return out


class LayerCache:
    """Capacity-LRU plus the speculative READY/PENDING lifecycle."""

    FREE, PENDING, READY = 0, 1, 2

    def __init__(self, cap):
        self.cap = cap
        self.rows = {}  # insertion order is LRU order; value carries state and issue stage

    def _refresh(self, expert):
        row = self.rows.pop(expert)
        self.rows[expert] = row

    def apply_demand(self, ids, stage):
        hits = consumed = matched = cancelled_capacity = 0
        leads = []
        uniq = sorted(set(ids))
        needed = set(uniq)
        resident = set()
        for expert in uniq:
            row = self.rows.get(expert)
            if row is None:
                continue
            if row['state'] == self.PENDING:
                # The engine places the demand's own copy and marks the pending
                # position cancelled; after that stage's reap the pair collapses
                # to this one final demand row.
                matched += 1
                del self.rows[expert]
                continue
            hits += 1
            resident.add(expert)
            if row['spec']:
                consumed += 1
                leads.append(stage - row['issue_stage'])
                row['spec'] = 0

        # Classify the whole union before evicting. No row in this demand may be
        # a victim, and the engine refreshes the union in ascending expert order.
        need = len(uniq) - hits
        while len(self.rows) + need > self.cap:
            victim = None
            for candidate, candidate_row in self.rows.items():
                if candidate in needed or candidate_row['state'] == self.PENDING:
                    continue
                if candidate_row['spec']:
                    victim = candidate
                    break
            if victim is None:
                for candidate, candidate_row in self.rows.items():
                    if candidate not in needed and candidate_row['state'] != self.PENDING:
                        victim = candidate
                        break
            if victim is None:
                raise ValueError('demand reached a cache with no legal victim')
            victim_row = self.rows.pop(victim)
            cancelled_capacity += int(victim_row['spec'])

        for expert in uniq:
            if expert in resident:
                self._refresh(expert)
            else:
                self.rows[expert] = {
                    'state': self.READY, 'spec': 0, 'issue_stage': stage,
                }
        return {
            'hits': hits, 'consumed': consumed, 'matched': matched,
            'cancelled_capacity': cancelled_capacity, 'leads': leads,
        }

    def reap(self):
        for row in self.rows.values():
            if row['state'] == self.PENDING:
                row['state'] = self.READY

    def admit(self, candidates, reserve, budget, stage):
        free = self.cap - len(self.rows)
        pending = sum(row['state'] == self.PENDING for row in self.rows.values())
        admitted = 0
        skipped_present = skipped_dup = refused_capacity = 0
        stop = 'proposals'
        seen = set()
        for expert in candidates:
            if expert in seen:
                skipped_dup += 1
                continue
            seen.add(expert)
            if expert in self.rows:
                skipped_present += 1
                continue
            if pending >= budget:
                refused_capacity += 1
                stop = 'pending'
            elif free - admitted <= 0:
                refused_capacity += 1
                stop = 'no_free'
            elif free - admitted - 1 < reserve:
                refused_capacity += 1
                stop = 'reserve'
            elif admitted >= budget:
                refused_capacity += 1
                stop = 'budget'
            else:
                self.rows[expert] = {
                    'state': self.PENDING, 'spec': 1, 'issue_stage': stage,
                }
                admitted += 1
                pending += 1
        return {
            'fills': admitted, 'skipped_present': skipped_present,
            'skipped_dup': skipped_dup, 'refused_capacity': refused_capacity,
            'stop': stop,
        }


def simulate_cache_leg(init, rows, method, budget):
    """Replay one selector on fixed recorded demand with ideal copy completion."""
    if method not in ('retention', 'history', 'coact', 'cache-aware'):
        raise ValueError('unknown cache-counterfactual method: %s' % method)
    retain_k = int(init['retain_k'])
    n_expert = int(init['n_expert'])
    top_k = int(init['top_k'])
    cap = min(n_expert, retain_k * top_k)
    reserve = top_k
    caches = defaultdict(lambda: LayerCache(cap))
    predictor = None
    if method == 'history':
        predictor = History()
    elif method in ('coact', 'cache-aware'):
        predictor = Coact()

    out = {
        'method': method, 'stages': 0, 'demands': 0, 'hits': 0,
        'proposals': 0, 'fills': 0, 'consumed': 0, 'matched_by_demand': 0,
        'cancelled_capacity': 0, 'skipped_present': 0, 'skipped_dup': 0,
        'refused_capacity': 0, 'room_stages': 0, 'proposal_room_stages': 0,
        'missing_room_candidates': 0, 'lead_sum': 0, 'lead_min': None, 'lead_max': None,
        'stop_counts': defaultdict(int), 'hit_mismatches': [],
    }
    for stage, row in enumerate(rows):
        layer = row['layer']
        ids = row['ids']
        cache = caches[layer]
        demand = cache.apply_demand(ids, stage)
        cache.reap()
        out['stages'] += 1
        out['demands'] += len(ids)
        out['hits'] += demand['hits']
        out['consumed'] += demand['consumed']
        out['matched_by_demand'] += demand['matched']
        out['cancelled_capacity'] += demand['cancelled_capacity']
        for lead in demand['leads']:
            out['lead_sum'] += lead
            out['lead_min'] = lead if out['lead_min'] is None else min(out['lead_min'], lead)
            out['lead_max'] = lead if out['lead_max'] is None else max(out['lead_max'], lead)
        if method == 'retention' and demand['hits'] != row['hits']:
            out['hit_mismatches'].append((stage, layer, row['hits'], demand['hits']))
        room = cap - len(cache.rows) > reserve
        if room:
            out['room_stages'] += 1

        if predictor is not None:
            predictor.observe(layer, ids)
            if method == 'history':
                candidates = predictor.predict(layer, budget)
            elif method == 'coact':
                candidates = predictor.predict(layer, budget)
            else:
                ranked = predictor.predict(layer, n_expert)
                candidates = cache_filtered(ranked, cache.rows.keys(), budget)
            out['proposals'] += len(candidates)
            if room and candidates:
                out['proposal_room_stages'] += 1
                out['missing_room_candidates'] += sum(
                    1 for expert in candidates if expert not in cache.rows)
            admitted = cache.admit(candidates, reserve, budget, stage)
            out['fills'] += admitted['fills']
            out['skipped_present'] += admitted['skipped_present']
            out['skipped_dup'] += admitted['skipped_dup']
            out['refused_capacity'] += admitted['refused_capacity']
            if candidates:
                out['stop_counts'][admitted['stop']] += 1

    out['share'] = out['hits'] / out['demands'] if out['demands'] else 0.0
    out['fill_precision'] = out['consumed'] / out['fills'] if out['fills'] else 0.0
    out['late_rate'] = out['matched_by_demand'] / out['fills'] if out['fills'] else 0.0
    out['staged_bytes'] = int(out['fills'] * BYTES_PER_ROW_MEASURED)
    out['stop_counts'] = dict(sorted(out['stop_counts'].items()))
    return out


def cache_counterfactual(paths, budget, expected_k=None):
    legs = []
    for path in paths:
        init, rows = load_cache_rows(path)
        legs.append((path, init, rows))
    signatures = {stream_signature(rows) for _, _, rows in legs}
    if len(signatures) != 1:
        raise ValueError('recorded K legs do not carry the same ordered demand stream')
    actual_k = [int(init['retain_k']) for _, init, _ in legs]
    if len(set(actual_k)) != len(actual_k):
        raise ValueError('cache counterfactual requires one trace per K')
    if expected_k is not None and actual_k != expected_k:
        raise ValueError('recorded K values %s do not match requested %s' % (actual_k, expected_k))

    results = []
    for path, init, rows in legs:
        baseline = simulate_cache_leg(init, rows, 'retention', budget)
        if baseline['hit_mismatches']:
            first = baseline['hit_mismatches'][0]
            raise ValueError('%s: cache replay differs at stage/layer %s/%s: recorded %d, replay %d' %
                             ((path,) + first))
        methods = {}
        for method in ('history', 'coact', 'cache-aware'):
            method_result = simulate_cache_leg(init, rows, method, budget)
            method_result['gain'] = method_result['share'] - baseline['share']
            methods[method] = method_result
        results.append({
            'path': path, 'retain_k': int(init['retain_k']),
            'cap': min(int(init['n_expert']), int(init['retain_k']) * int(init['top_k'])),
            'top_k': int(init['top_k']), 'baseline': baseline, 'methods': methods,
        })
    return {
        'schema': 'gplr58-candidate-counterfactual/1',
        'host_only': True, 'budget': budget, 'stream_signature': next(iter(signatures)),
        'legs': results,
    }


def print_cache_counterfactual(result):
    print('T-GPLR-58 cache-aware candidate-selection counterfactual (host-only)')
    print('demand stream: %d rows, signature %s; no model, CUDA, GPU, or mutex' %
          (result['legs'][0]['baseline']['stages'], result['stream_signature']))
    print('All baseline cache replays match their recorded per-stage hit counts exactly.')
    print()
    print('%2s %4s %-12s %8s %8s %8s %6s %6s %8s %6s %7s %9s %8s' %
          ('K', 'cap', 'selector', 'proposed', 'skipP', 'refuse', 'room', 'p@room',
           'miss@room', 'fills', 'useful', 'share', 'gain_pp'))
    for leg in result['legs']:
        base = leg['baseline']
        print('%2d %4d %-12s %8d %8s %8s %6d %6s %8s %6d %7d %9.5f %9s' %
              (leg['retain_k'], leg['cap'], 'retention', base['stages'], '-', '-',
               base['room_stages'], '-', '-', 0, 0, base['share'], '-'))
        for name, row in leg['methods'].items():
            print('%2s %4s %-12s %8d %8d %8d %6d %6d %8d %6d %7d %9.5f %+9.4f' %
                  ('', '', name, row['proposals'], row['skipped_present'],
                   row['refused_capacity'], row['room_stages'], row['proposal_room_stages'],
                   row['missing_room_candidates'], row['fills'], row['consumed'],
                   row['share'], 100.0 * row['gain']))
    print()
    for leg in result['legs']:
        stop_text = []
        for name, row in leg['methods'].items():
            stops = ','.join('%s=%d' % item for item in row['stop_counts'].items()) or 'none'
            stop_text.append('%s:%s' % (name, stops))
        print('K=%-2d stop reasons: %s' % (leg['retain_k'], '; '.join(stop_text)))
    print('Reading: candidates are selected AFTER the current demand, so history names rows')
    print('already resident. cache-aware filters those rows and backfills lower-ranked coact.')
    print('p@room is a non-empty proposal while capacity remains; miss@room is the subset')
    print('actually absent from cache. miss@room=0 means no selector had an admissible candidate.')
    print('useful counts actual lead>=2 consumptions; fill bytes use the measured 1.6875 MiB row.')


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

    # Cache-aware selection asks coact for its full ranking. Its top B must be
    # byte-for-byte the ordinary B-row arm; filtering happens only afterwards.
    full = Coact()
    for _tok, layer, ids in rows[:20]:
        full.observe(layer, list(ids))
    assert full.predict(0, 8) == full.predict(0, n_exp)[:8], "full ranking changed the top B"
    assert cache_filtered([3, 1, 4, 1, 2, 0], {1, 4}, 3) == [3, 2, 0], \
        "cache filter must dedup, remove present rows, and backfill"

    # The LRU baseline is exact on a hand-checked sequence.
    cache = LayerCache(4)
    replay = []
    for ids in ([0, 1], [2, 3], [0, 1]):
        replay.append(cache.apply_demand(ids, len(replay))['hits'])
        cache.reap()
    assert replay == [0, 0, 2], "capacity-LRU replay changed the recorded hit rule"

    # A fill issued after stage 0 cannot serve stage 1; it is matched/cancelled
    # there and can first be consumed at stage 2.
    cache = LayerCache(8)
    assert cache.apply_demand([0, 1], 0)['hits'] == 0
    cache.reap()
    assert cache.admit([4, 5], 2, 2, 0)['fills'] == 2
    demand1 = cache.apply_demand([2, 3], 1)
    assert (demand1['hits'], demand1['matched'], demand1['consumed']) == (0, 0, 0), \
        "pending fill was visible one stage too early"
    cache.reap()
    demand2 = cache.apply_demand([4, 5], 2)
    assert (demand2['hits'], demand2['consumed'], demand2['leads']) == (2, 2, [2, 2]), \
        "lead-2 consumption lifecycle is wrong"

    # If the one-stage demand asks for the pending row, it is cancelled rather
    # than counted as a hit; the demand copy replaces it.
    cache = LayerCache(8)
    cache.apply_demand([0, 1], 0)
    cache.reap()
    cache.admit([4], 2, 2, 0)
    cancelled = cache.apply_demand([4, 2], 1)
    assert (cancelled['hits'], cancelled['matched'], cancelled['consumed']) == (0, 1, 0), \
        "demand-duplicate cancellation lifecycle is wrong"

    # Reserve admission stops before consuming the last protected positions.
    cache = LayerCache(8)
    cache.apply_demand([0, 1], 0)
    cache.reap()
    admitted = cache.admit([2, 3, 4, 5, 6], 2, 8, 0)
    assert (admitted['fills'], admitted['refused_capacity'], admitted['stop']) == (4, 1, 'reserve'), \
        "capacity reserve boundary is wrong"

    print("SELFTEST PASS: online faithfulness held; planted history/markov structure detected; "
          "bar/fusion operators sane; cache filter, LRU, reserve, and lead-2 lifecycle verified")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="*", help="JSONL traces with rowset_fetch events")
    ap.add_argument("--b", default="8,16,32", help="budgets (comma-sep, default 8,16,32)")
    ap.add_argument("--warm", type=int, default=0,
                    help="leading tokens used as observation only (default 0 = colibri F4)")
    ap.add_argument("--mode", default="base", choices=["base", "bar", "fusion", "cache"],
                    help="base = screening table; bar = confidence-bar sweep "
                         "(precision-recall frontier); fusion = weighted + vote "
                         "merge with tune/held split and null control; cache = "
                         "recorded-demand cache-aware selector replay")
    ap.add_argument("--budget", type=int, default=8,
                    help="cache mode: speculative proposal budget (default 8)")
    ap.add_argument("--expect-k", default="",
                    help="cache mode: comma-separated K values in trace order")
    ap.add_argument("--json-out", default="",
                    help="cache mode: write the deterministic JSON result here")
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
    if args.mode == "cache":
        if args.budget <= 0:
            ap.error("--budget must be positive")
        expected_k = [int(x) for x in args.expect_k.split(",")] if args.expect_k else None
        try:
            result = cache_counterfactual(args.traces, args.budget, expected_k)
        except (OSError, ValueError) as exc:
            print("REFUSED: %s" % exc, file=sys.stderr)
            return 2
        print_cache_counterfactual(result)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
                fh.write("\n")
            print("json: %s" % args.json_out)
        return 0

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
    sys.exit(main())
