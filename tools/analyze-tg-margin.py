#!/usr/bin/env python3
"""The incumbent row cache's UNCOVERED MARGIN, and the ceiling any id-stream predictor faces.

Host-only: reads recorded traces, needs no GPU, no mutex and no model.  The question it answers
is the one that bounds every predictor arm before a hold is spent (T-GPLR-58 / T-GPLR-59): the
row cache at depth K already serves `share` of the demanded rows, the rest is the MARGIN a
predictor could try to win, and only part of that margin is reachable by a predictor built from
observed expert ids.

RETENTION SEMANTICS, IDENTIFIED FROM THE RECORDED TRACES (this matters - the obvious reading is
wrong).  The SDD describes the row cache as "the last K windows retained".  Rebuilding the
engine's OWN `n_rows_resident` from `union_ids` under three candidate models gives:

  model                                   gplr19 honest K=6   qa-depth K=4   gplr19 echoed K=2
  last K windows, per layer                    -16.92%           -17.15%           -7.53%
  last K windows, per SLOT                     -61.90%           -69.44%          -86.60%
  capacity-LRU, per layer, K*top_k rows         -0.18%            -0.16%           -1.01%

  (deviation of the rebuilt hit count from the engine's own booking).  So the cache is PER LAYER
  (slot sharing is decisively excluded, even though `slot == layer % n_slots` in those runs) and
  its semantics are a capacity-LRU: each layer holds K*top_k ROWS with recency, which means the
  effective retention horizon EXCEEDS K tokens whenever ids repeat inside the window.  The
  incumbent is therefore stronger than the "last K windows" wording implies, and the predictor's
  room is smaller than a naive reading of K would suggest.  A residual of 0.16-1.01% remains
  unexplained (eviction tie-breaking, or a first-window special case); the tool prints it, and a
  deviation beyond `--max-dev` (default 2%) refuses the trace instead of scoring it.

Per trace it reports, in engine units:
  demanded  = sum(top_k) over rowset_fetch rows            (the TG demand stream, W=1)
  hits      = sum(n_rows_resident)   <- the engine's own booking
  hits_lru  = the same rebuilt by this tool under capacity-LRU; the deviation is printed
  share     = hits / demanded          margin = 1 - share

and decomposes the margin by what a PREDICTOR could reach (each id counted once):
  C1 immediate  - the id is demanded at the SAME token in an EARLIER layer (available for free,
                  layers execute in order; this is the cross-layer/coupling region)
  C2 same-layer - the id was demanded at THIS layer more than K but at most W tokens ago
                  (the temporal families' reach: markov-1/coact approximate exactly this)
  C3 elsewhere  - the id was seen earlier somewhere (any layer, any earlier token) but not C1/C2
  C4 first-visit- the id was never demanded earlier anywhere: NOT reachable by ANY id-stream
                  predictor, so C4 is the hard floor of the margin.
  ceiling share = 1 - (C4 / demanded)  -- the best any id-stream predictor can do at this K.

usage:
  python tools/analyze-tg-margin.py --selftest
  python tools/analyze-tg-margin.py build/llama-cuda/gplr19-honest-k6-ab/arm-B1.jsonl ...
  python tools/analyze-tg-margin.py --W 8,16,32,64 --json out.json <traces...>

Refusals (exit 2, NO numbers printed): no rowset_init, no rowset_fetch, a route_union that is
not top_k (the W=1 model is broken), an accounting trace whose own hits+fetched do not close
against its demand, a row cache that is OFF (retain_k = 0 is the REACTIVE baseline - it has no
incumbent and no margin to bound), and a trace where the rebuilt hit count deviates from the
engine's booking by more than --max-dev.
"""

import argparse
import math
import json
import os
import statistics
import subprocess
import sys
import tempfile

EXIT_OK, EXIT_REFUSED = 0, 2


def fail(msg):
    sys.stderr.write('REFUSED: %s\n' % msg)
    sys.exit(EXIT_REFUSED)


def load(path):
    """-> (init, rows) in file order (`t_us` is monotonic in the recorded runs)."""
    init, rows = None, []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ev = d.get('event')
            if ev == 'rowset_init':
                init = d
            elif ev == 'rowset_fetch':
                rows.append(d)
    return init, rows


def ids_of(row):
    s = row.get('union_ids')
    if isinstance(s, list):
        return [int(x) for x in s]
    if not isinstance(s, str):
        return None
    out = []
    for part in s.split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def rebuid_hits_lru(seq_by_layer, keep_k, top_k, n_expert):
    """capacity-LRU hits, and the margin set per (token index, layer) at the moment of demand.

    -> (hits, margins) where margins[(L, i)] = set of ids demanded that are NOT resident.
    """
    cap = min(n_expert, keep_k * top_k) if n_expert > 0 else keep_k * top_k
    hits = 0
    margins = {}
    for L, lst in seq_by_layer.items():
        lst = sorted(lst, key=lambda x: x[0])
        cache = {}          # id -> last-demand tick
        for tick, (t_ix, cur) in enumerate(lst):
            resident = cur & set(cache.keys())
            hits += len(resident)
            margins[(L, t_ix)] = set(cur) - set(cache.keys())
            for e in cur:
                cache[e] = tick
            while len(cache) > cap:
                victim = min(cache.items(), key=lambda kv: kv[1])[0]
                del cache[victim]
    return hits, margins


def _window_hits(groups, keep_k):
    """hits if each group resident set were the union of the previous keep_k entries."""
    hits = 0
    for key, lst in groups.items():
        for n, (i, cur) in enumerate(lst):
            hist = set()
            for j in range(1, keep_k + 1):
                if n - j >= 0:
                    hist |= lst[n - j][1]
            hits += len(cur & hist)
    return hits


def _lru_hits(groups, keep_k, top_k, n_expert=0):
    """hits if each group held a capacity-LRU cache of keep_k*top_k rows."""
    cap = min(n_expert, keep_k * top_k) if n_expert else keep_k * top_k
    hits = 0
    for key, lst in groups.items():
        cache = {}
        for tick, (i, cur) in enumerate(lst):
            hits += len(cur & set(cache))
            for e in cur:
                cache[e] = tick
            while len(cache) > cap:
                victim = min(cache.items(), key=lambda kv: kv[1])[0]
                del cache[victim]
    return hits


def identify_models(path):
    """Which retention model reproduces this trace's OWN booking? -> dict, or None.

    This exists because the tool cannot score a margin it does not understand the cache of:
    the model it is built on (per-layer capacity-LRU) was IDENTIFIED this way, against the two
    plausible alternatives, before any margin was reported. Run it whenever a trace's deviation
    is larger than the tool allows, or before modelling the cache host-side for any other
    purpose.
    """
    init, rows = load(path)
    if init is None or not rows:
        return None
    keep_k = int(init.get('retain_k', 0))
    top_k = int(init.get('top_k', 0))
    n_slots = int(init.get('n_slots', 1)) or 1
    n_expert = int(init.get('n_expert', 0))
    toks = sorted({int(r['ubatch']) for r in rows})
    t_ix = {t: i for i, t in enumerate(toks)}
    by_layer, by_slot = {}, {}
    slots_of = {}
    for r in rows:
        L = int(r['layer'])
        s = int(r.get('slot', L % n_slots))
        slots_of[L] = s
        ent = (t_ix[int(r['ubatch'])], set(ids_of(r)))
        by_layer.setdefault(L, []).append(ent)
        by_slot.setdefault(s, []).append((r, ent))
    # the per-slot view reconstructs a slot's demand timeline from its member layers
    slot_groups = {}
    for s, lst in by_slot.items():
        lst.sort(key=lambda x: x[1][0])
        slot_groups[s] = [(ent[0], ent[1]) for _, ent in lst]
    for g in list(by_layer.values()) + list(slot_groups.values()):
        g.sort(key=lambda x: x[0])
    engine = sum(int(r.get('n_rows_resident', 0)) for r in rows)
    demanded = sum(len(ids_of(r)) for r in rows)
    if not demanded:
        return None
    slots_shared = len({s for s in slots_of.values()}) < len(slots_of)
    dev = {}
    for tag, groups, kk, tk in (('window_layer', by_layer, keep_k, top_k),
                                ('window_slot', slot_groups, keep_k, top_k),
                                ('lru_layer', by_layer, keep_k, top_k),
                                ('lru_slot', slot_groups, keep_k, top_k)):
        if tag.startswith('window'):
            h = _window_hits(groups, keep_k)
        else:
            h = _lru_hits(groups, kk, tk, n_expert)
        # A trace with no resident hits at all cannot have a percentage deviation: a model that
        # predicts any hit is then unambiguously wrong, not "0% off", so it is reported as +inf
        # rather than silently identical to the exact model.
        if engine:
            dev[tag] = 100.0 * (h - engine) / engine
        else:
            dev[tag] = 0.0 if h == 0 else float('inf')
    return {'file': path, 'retain_k': keep_k, 'top_k': top_k, 'n_slots': n_slots,
            'layers': len(slots_of), 'engine_hits': engine, 'demanded': demanded,
            'share': engine / demanded, 'dev': dev,
            'slot_is_layer_mod_n_slots': all(L % n_slots == s for L, s in slots_of.items()),
            'slots_shared_by_more_than_one_layer': slots_shared}


def analyze(path, W_list, max_dev=2.0):
    init, rows = load(path)
    if init is None:
        fail('%s: no rowset_init (not a row-set trace)' % path)
    if not rows:
        fail('%s: no rowset_fetch rows (nothing to measure)' % path)
    keep_k = int(init.get('retain_k', 0))
    if keep_k == 0:
        fail('%s: retain_k = 0 - this is the REACTIVE baseline (row cache OFF), so it has no '
             'incumbent and no margin to bound. Point the tool at a retention leg (arm-B*) and '
             'use this trace only as the no-incumbent reference.' % path)
    top_k = int(init.get('top_k', 0))
    if top_k <= 0:
        fail('%s: rowset_init has no usable top_k' % path)
    n_expert = int(init.get('n_expert', 0))
    phases = sorted({int(r.get('phase', -1)) for r in rows})
    for r in rows:
        if int(r.get('route_union', top_k)) != top_k:
            fail('%s: route_union %s != top_k %d (the W=1 assumption this tool is built on is '
                 'broken for this trace)' % (path, r.get('route_union'), top_k))

    toks = sorted({int(r['ubatch']) for r in rows})
    t_ix_of = {t: i for i, t in enumerate(toks)}
    layers = sorted({int(r['layer']) for r in rows})
    cur_by_layer = {}
    for r in rows:
        cur_by_layer.setdefault(int(r['layer']), []).append((t_ix_of[int(r['ubatch'])], set(ids_of(r))))

    demanded = sum(len(v[1]) for lst in cur_by_layer.values() for v in lst)
    engine_hits = sum(int(r.get('n_rows_resident', 0)) for r in rows)
    engine_fetched = sum(int(r.get('n_rows_fetched', 0)) for r in rows)
    if engine_hits + engine_fetched != demanded:
        fail('%s: the trace does not close - hits %d + fetched %d != demanded %d, so nothing '
             'below can be trusted' % (path, engine_hits, engine_fetched, demanded))

    recon_hits, margins = rebuid_hits_lru(cur_by_layer, keep_k, top_k, n_expert)
    dev = 100.0 * (recon_hits - engine_hits) / engine_hits if engine_hits else 0.0
    if abs(dev) > max_dev:
        fail('%s: the capacity-LRU rebuild is off by %.2f%% (%.2f%% allowed): the retention model '
             'does not reproduce this trace, so its margin is not a measurement. Re-identify the '
             'semantics before scoring it.' % (path, dev, max_dev))

    # ids demanded at token t' (any layer), for the "seen earlier anywhere" test
    tok_ids = [set() for _ in toks]
    for L, lst in cur_by_layer.items():
        for i, s in lst:
            if s:
                tok_ids[i] |= s
    seen_before, acc = [], set()
    for i in range(len(toks)):
        seen_before.append(set(acc))
        acc |= tok_ids[i]

    per_layer = {}
    for L in layers:
        per_layer[L] = {i: s for i, s in cur_by_layer.get(L, [])}

    c1 = c2 = c3 = c4 = 0
    per_layer_margin, per_token_margin = {}, {}
    w_hits = {w: 0 for w in W_list}
    for i, t in enumerate(toks):
        for L in layers:
            margin = margins.get((L, i))
            if not margin:
                continue
            per_layer_margin[L] = per_layer_margin.get(L, 0) + len(margin)
            per_token_margin[t] = per_token_margin.get(t, 0) + len(margin)
            far_by_w = {}
            for w in sorted(W_list, reverse=True):
                far = set()
                for j in range(keep_k + 1, w + 1):
                    if i - j >= 0:
                        far |= per_layer[L].get(i - j, set())
                far_by_w[w] = far
            for e in margin:
                if any(e in per_layer[Lp].get(i, set()) for Lp in layers if Lp < L):
                    c1 += 1
                    continue
                matched = None
                for w in sorted(W_list):
                    if e in far_by_w.get(w, set()):
                        matched = w
                        break
                if matched is not None:
                    c2 += 1
                    w_hits[matched] += 1
                    continue
                if e in seen_before[i]:
                    c3 += 1
                    continue
                c4 += 1

    share = engine_hits / demanded if demanded else 0.0
    classified = c1 + c2 + c3 + c4
    return {
        'file': path,
        'retain_k': keep_k,
        'phases': phases,
        'top_k': top_k,
        'n_expert': n_expert,
        'cache_rows_per_layer': (min(n_expert, keep_k * top_k) if n_expert else keep_k * top_k),
        'tokens': len(toks),
        'layers': len(layers),
        'demanded': demanded,
        'engine_hits': engine_hits,
        'engine_fetched': engine_fetched,
        'recon_hits_lru': recon_hits,
        'model_deviation_pct': dev,
        'share': share,
        'margin': demanded - engine_hits,
        'margin_share': 1.0 - share,
        'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
        'classified': classified,
        'ceiling_share': 1.0 - (c4 / demanded) if demanded else 0.0,
        'w_hits': w_hits,
        'margin_rows_per_row': (demanded - engine_hits) / len(rows) if rows else 0.0,
        'per_layer_margin': per_layer_margin,
        'per_token_margin': per_token_margin,
        'tokens_list': toks,
    }


def report(a, W_list):
    print('=' * 100)
    print('%s' % a['file'])
    print('  retain_k=%d  phases=%s  top_k=%d  cache %d rows/layer  %d tokens x %d layers' %
          (a['retain_k'], a['phases'], a['top_k'], a['cache_rows_per_layer'], a['tokens'], a['layers']))
    print('  demanded %d | engine hits %d (fetched %d) | share %.4f | margin %d (%.4f)' %
          (a['demanded'], a['engine_hits'], a['engine_fetched'], a['share'], a['margin'], a['margin_share']))
    print('  capacity-LRU rebuild %d rows vs engine %d -> deviation %+.2f%%' %
          (a['recon_hits_lru'], a['engine_hits'], a['model_deviation_pct']))
    print('  margin per (token, layer): %.3f rows (of top_k=%d)' % (a['margin_rows_per_row'], a['top_k']))
    tot = a['classified'] or 1
    print('  MARGIN DECOMPOSITION (each row counted once, priority order):')
    print('    C1 immediate (same token, earlier layer) : %7d  %5.1f%% of margin' % (a['c1'], 100.0 * a['c1'] / tot))
    print('    C2 same-layer, seen >K .. <=%-2d tokens  : %7d  %5.1f%% of margin  %s' %
          (max(W_list), a['c2'], 100.0 * a['c2'] / tot,
           ' '.join('W%d=%d' % (w, a['w_hits'][w]) for w in sorted(W_list))))
    print('    C3 seen earlier anywhere (not C1/C2)    : %7d  %5.1f%% of margin' % (a['c3'], 100.0 * a['c3'] / tot))
    print('    C4 FIRST-VISIT (unreachable by ids)     : %7d  %5.1f%% of margin' % (a['c4'], 100.0 * a['c4'] / tot))
    print('  CEILING for any id-stream predictor at this K: share <= %.4f (vs incumbent %.4f)' %
          (a['ceiling_share'], a['share']))
    pl = a['per_layer_margin']
    if pl:
        vals = sorted(pl.items(), key=lambda kv: -kv[1])
        print('  per-layer margin rows: min %d, median %.0f, max %d | worst: %s' %
              (min(pl.values()), statistics.median(pl.values()), max(pl.values()),
               ', '.join('L%d=%d' % (L, v) for L, v in vals[:3])))
    pt, toks = a['per_token_margin'], a['tokens_list']
    warm = [pt[t] for t in toks[:a['retain_k']] if t in pt]
    steady = [pt[t] for t in toks[a['retain_k']:] if t in pt]
    if warm and steady:
        print('  first K tokens (warm-up) mean margin %.1f rows vs steady-state %.1f rows' %
              (sum(warm) / len(warm), sum(steady) / len(steady)))


# --------------------------------------------------------------------------- selftest
def _fr(t, L, ids, resident, top_k=2):
    n = len(ids)
    return {'event': 'rowset_fetch', 'ubatch': t + 3, 'layer': L, 'generation': 1, 'chunk': t + 2,
            'phase': 1, 'top_k': top_k, 'route_union': top_k,
            'union_ids': ','.join(str(x) for x in ids),
            'n_rows_resident': resident, 'n_rows_fetched': n - resident}


def _write(tmp, tag, rows, retain_k=2, top_k=2, n_expert=64, n_slots=4):
    p = os.path.join(tmp, tag + '.jsonl')
    nl = chr(10)
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'event': 'rowset_init', 'retain_k': retain_k, 'top_k': top_k,
                             'n_slots': n_slots, 'n_expert': n_expert, 'window_tokens': 1}) + nl)
        for r in rows:
            fh.write(json.dumps(r) + nl)
    return p


def selftest():
    cases, fails = 0, []

    def check(name, got, want):
        nonlocal cases
        cases += 1
        if got != want:
            fails.append('%s: got %r want %r' % (name, got, want))

    tmp = tempfile.mkdtemp(prefix='tgm-')

    # Case 1 (identity + ceiling): K=2, top_k=2 -> capacity 4 rows/layer, one layer, 3 tokens.
    # t0 cache {} -> hits 0; t1 {} -> 0; t2 {1,2} still resident (cache {1,2,3,4} == cap) -> 2.
    rows = [_fr(0, 0, [1, 2], 0), _fr(1, 0, [3, 4], 0), _fr(2, 0, [1, 2], 2)]
    a = analyze(_write(tmp, 'repeat', rows, retain_k=2), [4, 8])
    check('repeat.recon == engine', a['recon_hits_lru'], a['engine_hits'])
    check('repeat.hits', a['engine_hits'], 2)
    check('repeat.demanded', a['demanded'], 6)
    check('repeat.c4', a['c4'], 4)
    check('repeat.ceiling', round(a['ceiling_share'], 4), round(1 - 4 / 6.0, 4))

    # Case 2 (degenerate end): every token brings ids never seen before -> ceiling 0.0.
    # K=1 -> capacity 2. t1's union reuses 2, so exactly one id is resident, as the trace says.
    rows = [_fr(0, 0, [1, 2], 0), _fr(1, 0, [2, 6], 1), _fr(2, 0, [7, 8], 0)]
    a = analyze(_write(tmp, 'firstvisit', rows, retain_k=1), [2, 4])
    check('firstvisit.recon == engine', a['recon_hits_lru'], a['engine_hits'])
    check('firstvisit.c4', a['c4'], 5)
    check('firstvisit.ceiling', round(a['ceiling_share'], 4), round(1 - 5 / 6.0, 4))

    # Case 3 (C1): the same token's earlier layer already demanded the id, and that id is
    # NOT resident in this layer's cache (so it is genuinely in the margin and C1 is
    # the reason it is reachable). Distinct ids per token keep retention out of it: no
    # id of t1 appears at t0, so every resident count here is 0 for both models.
    rows = [_fr(0, 0, [10, 11], 0), _fr(0, 1, [12, 13], 0),
            _fr(1, 0, [30, 31], 0), _fr(1, 1, [30, 32], 0)]
    a = analyze(_write(tmp, 'crosslayer', rows, retain_k=1), [4, 8])
    check('crosslayer.recon == engine', a['recon_hits_lru'], a['engine_hits'])
    check('crosslayer.C1 fires', a['c1'] >= 1, True)

    # Case 4 (C2): an id demanded at this layer beyond K but inside W -> the temporal reach.
    rows = [_fr(0, 0, [20, 21], 0), _fr(1, 0, [22, 23], 0), _fr(2, 0, [20, 24], 0)]
    a = analyze(_write(tmp, 'C2', rows, retain_k=1), [2, 4])
    check('C2.recon == engine', a['recon_hits_lru'], a['engine_hits'])
    check('C2 fires for the beyond-K id', a['c2'] >= 1, True)

    # Case 5 (C3): seen earlier SOMEWHERE, but neither in-token nor at this layer within W.
    rows = [_fr(0, 0, [70, 71], 0), _fr(1, 0, [80, 81], 0), _fr(2, 1, [70, 82], 0),
            _fr(3, 0, [70, 99], 0)]
    a = analyze(_write(tmp, 'C3', rows, retain_k=1), [2])
    check('C3.recon == engine', a['recon_hits_lru'], a['engine_hits'])
    check('C3 fires for the elsewhere-seen id', a['c3'] >= 1, True)

    # Case 6: refusal - the reactive baseline has no incumbent, so it has no margin.
    p = _write(tmp, 'reactive', [_fr(0, 0, [1, 2], 0)], retain_k=0)
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), p],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    check('reactive exit code', rc, EXIT_REFUSED)

    # Case 7: refusal - the trace's own accounting must close.
    p = _write(tmp, 'unclosed', [_fr(0, 0, [1, 2], 0)], retain_k=1)
    txt = open(p, encoding='utf-8').read().replace('"n_rows_fetched": 2', '"n_rows_fetched": 0')
    open(p, 'w', encoding='utf-8').write(txt)
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), p],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    check('unclosed accounting exit code', rc, EXIT_REFUSED)

    # Case 8: refusal - a trace whose resident counts this model cannot reproduce (deliberately
    # inflated), i.e. the semantics do not hold for it and it must not be scored.
    rows = [_fr(0, 0, [1, 2], 2), _fr(1, 0, [3, 4], 2), _fr(2, 0, [1, 2], 2)]
    p = _write(tmp, 'unmodelable', rows, retain_k=1)
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), p],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    check('unmodelable deviation exit code', rc, EXIT_REFUSED)

    # Case 9 (identification): a trace whose booking is per-layer LRU, on a shape where
    # `slot == layer % n_slots` holds AND slots are shared (layers 0 and 2 both map to slot 0),
    # so the per-slot reading is available and is the WRONG one. This pins the red-herring: the
    # identity fact does not make per-slot sharing true.
    rows = [_fr(0, 0, [1, 2], 0), _fr(0, 1, [3, 4], 0),
            _fr(1, 2, [1, 5], 0), _fr(1, 3, [6, 7], 0)]
    p = _write(tmp, 'identify', rows, retain_k=1, n_slots=2)
    idn = identify_models(p)
    check('identify.slot == layer % n_slots', idn['slot_is_layer_mod_n_slots'], True)
    check('identify.slots shared', idn['slots_shared_by_more_than_one_layer'], True)
    check('identify.per-layer LRU is exact', round(idn['dev']['lru_layer'], 6), 0.0)
    check('identify.per-slot LRU over-predicts', idn['dev']['lru_slot'] > 0.0, True)
    check('identify.per-slot window over-predicts too', idn['dev']['window_slot'] > 0.0, True)

    # Case 10: refusal - route_union != top_k means W>1 and the W=1 model does not hold.
    p = os.path.join(tmp, 'w2.jsonl')
    nl = chr(10)
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'event': 'rowset_init', 'retain_k': 1, 'top_k': 2, 'n_slots': 4,
                             'n_expert': 64, 'window_tokens': 1}) + nl)
        bad = _fr(0, 0, [1, 2], 0)
        bad['route_union'] = 3
        fh.write(json.dumps(bad) + nl)
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), p],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    check('route_union mismatch exit code', rc, EXIT_REFUSED)

    if fails:
        sys.stderr.write('SELFTEST FAILED (%d cases):\n' % cases)
        for f in fails:
            sys.stderr.write('  - %s\n' % f)
        return 1
    print('SELFTEST PASS: %d cases - capacity-LRU identity at three shapes, the ceiling at both '
          'ends, the C1/C2/C3 regions, the retention-model identification (including the shared-'
          'slot red herring), and four refusals (reactive baseline, unclosed accounting, '
          'unmodelable deviation, W>1 shape)' % cases)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='*')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--W', default='8,16,32,64', help='same-layer look-back horizons beyond K')
    ap.add_argument('--max-dev', type=float, default=2.0,
                    help='refuse a trace whose capacity-LRU rebuild deviates more than this %%')
    ap.add_argument('--json', help='write the table as JSON to this path')
    ap.add_argument('--identify', action='store_true',
                    help='print which retention model reproduces each trace (per-layer/per-slot x '
                         'last-K-windows/capacity-LRU) and exit; no margin is scored')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.identify:
        if not args.traces:
            ap.error('--identify needs at least one trace')
        for t in args.traces:
            idn = identify_models(t)
            if idn is None:
                fail('%s: no rowset_init/rowset_fetch rows - nothing to identify' % t)
            print('=' * 100)
            print(idn['file'])
            print('  retain_k=%d top_k=%d n_slots=%d layers=%d | slot == layer %% n_slots: %s | '
                  'a slot carries >1 layer: %s'
                  % (idn['retain_k'], idn['top_k'], idn['n_slots'], idn['layers'],
                     idn['slot_is_layer_mod_n_slots'],
                     idn['slots_shared_by_more_than_one_layer']))
            print('  engine hits %d of %d (share %.4f) - deviation of each candidate model:'
                  % (idn['engine_hits'], idn['demanded'], idn['share']))
            for tag in ('window_layer', 'window_slot', 'lru_layer', 'lru_slot'):
                v = idn['dev'][tag]
                print('    %-13s %s' % (tag, ('%+8.2f%%' % v) if math.isfinite(v) else '    +inf'))
        return EXIT_OK
    if not args.traces:
        ap.error('give at least one trace, or --selftest')
    W_list = sorted({int(x) for x in args.W.split(',') if x.strip()})
    out = []
    for t in args.traces:
        a = analyze(t, W_list, max_dev=args.max_dev)
        report(a, W_list)
        out.append({k: v for k, v in a.items()
                    if k not in ('per_layer_margin', 'per_token_margin', 'tokens_list')})
    if out:
        print('=' * 100)
        print('%-40s %3s %8s %7s %8s %6s %6s %6s %6s %8s %7s' %
              ('trace', 'K', 'demanded', 'share', 'margin', 'C1%', 'C2%', 'C3%', 'C4%',
               'ceiling', 'dev%'))
        for a in out:
            tot = a['classified'] or 1
            print('%-40s %3d %8d %7.4f %8d %5.1f%% %5.1f%% %5.1f%% %5.1f%% %8.4f %+6.2f%%' %
                  (a['file'][-40:], a['retain_k'], a['demanded'], a['share'], a['margin'],
                   100.0 * a['c1'] / tot, 100.0 * a['c2'] / tot, 100.0 * a['c3'] / tot,
                   100.0 * a['c4'] / tot, a['ceiling_share'], a['model_deviation_pct']))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print('table written: %s' % args.json)
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
