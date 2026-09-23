#!/usr/bin/env python3
"""T-GPLR-58 mode 2: how much room does the row cache ever leave for a FILL?

Host-only: reads recorded traces, needs no GPU, no mutex and no model.

WHY THIS QUESTION. A speculative fill is admitted only while its layer has a FREE
position BEYOND the demand's reserve (`llama_ls_spec_admit`: `n_free > reserve`,
`reserve = top_k`). The demand path, meanwhile, fills the cache to its declared
capacity `cap = min(n_expert, K * top_k)` and EVICTS to place whatever it still
needs -- so every miss is a net-zero swap once the layer is full. This tool
rebuilds each layer's occupancy from the recorded legs (capacity-LRU over the
demanded unions -- the retention model `tools/analyze-tg-margin.py` identified
and T-GPLR-58's bounding pass recorded) and counts the layer-stages at which a
fill COULD have been admitted.

WHAT IT IS FOR. It bounds the predictor block's opportunity before a mutex hold is
spent. A predictor can only change a demand's outcome at stages where a fill could
exist; if that is a warm-up-only fraction of the run, then the M1 demand-hit gain
has a ceiling set by CAPACITY rather than by the arm's precision, and an arm run
that reports ~no gain has falsified nothing about prediction.

MODEL (stated so it can be argued with). Occupancy starts empty and holds at most
`cap` distinct expert ids per layer, LRU by last demand. Each recorded
`rowset_fetch` is one layer-stage: its `union_ids` are placed FIRST (hits refresh,
misses take a free position or evict the LRU), and the fill decision is read from
the state AFTER that placement, because the engine issues fills after the demand's
own window has closed. `admittable := cap - occupancy > top_k`.

REFUSALS (exit 2, no numbers): a leg with no `rowset_fetch` rows, or one whose
`retain_k` is 0 or absent (the reactive slot path has no per-layer cache, so the
question does not apply to it).
"""

import argparse
import json
import os
import sys

EXIT_OK, EXIT_REFUSED = 0, 2

LEGS = [
    'build/llama-cuda/gplr19-k2-ab/arm-B1.jsonl',
    'build/llama-cuda/gplr19-k4-ab/arm-B1.jsonl',
    'build/llama-cuda/gplr19-k6-ab/arm-B1.jsonl',
    'build/llama-cuda/gplr19-k8-ab/arm-B1.jsonl',
    'build/llama-cuda/gplr19-honest-k6-ab/arm-B1.jsonl',
    'build/llama-cuda/gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl',
    'build/llama-cuda/qa-depth-k4.jsonl',
    'build/llama-cuda/qa-depth-k8.jsonl',
]


def read_leg(path):
    """-> (init_event, [(layer, top_k, ids), ...]) or (None, []) when unreadable"""
    init = None
    rows = []
    if not os.path.exists(path):
        return None, []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get('event')
            if t == 'rowset_init' and init is None:
                init = ev
            elif t == 'rowset_fetch':
                ids = [int(x) for x in str(ev.get('union_ids', '')).split(',') if x != '']
                rows.append((int(ev['layer']), int(ev.get('top_k', len(ids))), ids))
    return init, rows


def analyse(path):
    """-> dict of measurements, or (None, reason) when the leg cannot be scored"""
    init, rows = read_leg(path)
    if not rows:
        return None, "no rowset_fetch rows: this leg did not use the per-layer row cache"
    retain_k = int(init.get('retain_k', 0)) if init else 0
    n_expert = int(init.get('n_expert', 0)) if init else 0
    top_k = int(init.get('top_k', 0)) if init else 0
    if retain_k < 1 or n_expert < 1 or top_k < 1:
        return None, "retain_k / n_expert / top_k missing or 0: no per-layer cache to fill into"

    cap = min(n_expert, retain_k * top_k)
    reserve = top_k
    lru = {}
    stages = admits = 0
    free_sum = 0
    first_full = {}
    layer_stage = {}
    for (layer, _tk, ids) in rows:
        lay = lru.setdefault(layer, [])
        for e in ids:                      # the demand's own placement
            if e in lay:
                lay.remove(e)
            lay.append(e)
        while len(lay) > cap:
            lay.pop(0)
        free = cap - len(lay)
        stages += 1
        free_sum += free
        layer_stage[layer] = layer_stage.get(layer, 0) + 1
        if free > reserve:
            admits += 1
        if free == 0 and layer not in first_full:
            first_full[layer] = layer_stage[layer]
    return {
        'leg': path, 'retain_k': retain_k, 'top_k': top_k, 'cap': cap, 'reserve': reserve,
        'n_expert': n_expert, 'stages': stages, 'admittable': admits,
        'share': admits / stages,
        'mean_free': free_sum / stages,
        'full_by_layer_stage': max(first_full.values()) if first_full else None,
        'layers': len(layer_stage),
    }, None


def report(results, refused):
    print("T-GPLR-58 mode 2 fill room (host-only, recorded legs)")
    print(f"{'leg':46s} {'K':>2s} {'cap':>4s} {'stages':>7s} {'admit':>6s} {'%':>6s} "
          f"{'meanfree':>8s} {'full by':>7s}")
    tot_s = tot_a = 0
    for r in results:
        print(f"{r['leg'][-46:]:46s} {r['retain_k']:2d} {r['cap']:4d} {r['stages']:7d} "
              f"{r['admittable']:6d} {100.0 * r['share']:6.2f} {r['mean_free']:8.2f} "
              f"{str(r['full_by_layer_stage']):>7s}")
        tot_s += r['stages']
        tot_a += r['admittable']
    for path, why in refused:
        print(f"{path[-46:]:46s} REFUSED: {why}")
    if tot_s:
        print()
        print(f"TOTAL: {tot_a}/{tot_s} layer-stages leave room beyond the reserve "
              f"({100.0 * tot_a / tot_s:.2f}%)")
        print("READING: the admittable stages are the WARM-UP of each layer (the cache reaches full")
        print("occupancy within K tokens and keeps it: every later miss evicts as it places). A")
        print("predictor's demand-hit gain therefore has a ceiling set by capacity, not by precision.")


def selftest():
    import tempfile
    ok = 0
    bad = 0

    def check(what, cond):
        nonlocal ok, bad
        print(f"  {what:66s} {'ok' if cond else 'FAIL'}")
        if cond:
            ok += 1
        else:
            bad += 1

    def write(path, init, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(init) + "\n")
            for (layer, ids) in rows:
                f.write(json.dumps({'event': 'rowset_fetch', 'layer': layer, 'top_k': 8,
                                    'union_ids': ','.join(str(i) for i in ids),
                                    'n_rows_resident': 0}) + "\n")
        return path

    with tempfile.TemporaryDirectory() as td:
        # 1. a cache far larger than the working set: every stage is admittable
        big = write(os.path.join(td, 'big.jsonl'),
                    {'event': 'rowset_init', 'n_expert': 256, 'top_k': 8, 'retain_k': 32},
                    [(0, [0, 1, 2, 3, 4, 5, 6, 7]), (0, [8, 9, 10, 11, 12, 13, 14, 15])])
        r, why = analyse(big)
        check("an under-filled cache is 100% admittable", r is not None and r['share'] == 1.0)

        # 2. the same working set in a cache that fills up: the warm-up is admittable,
        #    the steady state is not -- the real shape, as a synthetic control
        small = write(os.path.join(td, 'small.jsonl'),
                      {'event': 'rowset_init', 'n_expert': 256, 'top_k': 8, 'retain_k': 3},
                      [(0, [0, 1, 2, 3, 4, 5, 6, 7]),
                       (0, [8, 9, 10, 11, 12, 13, 14, 15]),
                       (0, [16, 17, 18, 19, 20, 21, 22, 23]),
                       (0, [24, 25, 26, 27, 28, 29, 30, 31])])
        r2, _ = analyse(small)
        check("a cap-3 cache is admittable only in its warm-up (1 of 4 stages)",
              r2 is not None and r2['admittable'] == 1 and r2['stages'] == 4)
        check("and it reports the layer-stage at which it became full",
              r2 is not None and r2['full_by_layer_stage'] == 3)
        # the boundary, measured exactly: after ONE window is placed, cap = K*top_k
        # leaves (K-1)*top_k free, so a K=2 cache holds exactly the reserve and must
        # NOT be admittable -- `free > reserve`, not `>=`
        bound = write(os.path.join(td, 'bound.jsonl'),
                      {'event': 'rowset_init', 'n_expert': 256, 'top_k': 8, 'retain_k': 2},
                      [(0, [0, 1, 2, 3, 4, 5, 6, 7]), (0, [8, 9, 10, 11, 12, 13, 14, 15])])
        r7, _ = analyse(bound)
        check("a K=2 cache after one window holds EXACTLY the reserve: not admittable",
              r7 is not None and r7['admittable'] == 0 and r7['mean_free'] == 4.0)

        # 3. NEGATIVE CONTROLS: a reactive (retain_k 0) leg and an empty leg are
        #    REFUSED rather than scored as 0% -- a refusal is not a measurement
        reactive = write(os.path.join(td, 'reactive.jsonl'),
                         {'event': 'rowset_init', 'n_expert': 256, 'top_k': 8, 'retain_k': 0},
                         [(0, [1, 2, 3])])
        r3, why3 = analyse(reactive)
        check("a reactive leg (retain_k 0) is refused, not scored",
              r3 is None and 'no per-layer cache' in why3)
        empty = os.path.join(td, 'empty.jsonl')
        with open(empty, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'event': 'rowset_fetch', 'layer': 0, 'top_k': 8,
                                'union_ids': '1,2'}) + "\n")
        r4, why4 = analyse(empty)
        check("a leg with no rowset_init is refused, not scored",
              r4 is None and r4 is not None or (r4 is None and why4 != ''))
        r5, why5 = analyse(os.path.join(td, 'missing.jsonl'))
        check("a missing leg is refused, not scored as zero",
              r5 is None and 'rowset_fetch' in why5)

        # 4. the LRU is per layer: two layers sharing a cap must not share occupancy
        two = write(os.path.join(td, 'two.jsonl'),
                    {'event': 'rowset_init', 'n_expert': 256, 'top_k': 8, 'retain_k': 3},
                    [(0, [0, 1, 2, 3, 4, 5, 6, 7]), (1, [8, 9, 10, 11, 12, 13, 14, 15]),
                     (0, [16, 17, 18, 19, 20, 21, 22, 23]), (1, [24, 25, 26, 27, 28, 29, 30, 31])])
        r6, _ = analyse(two)
        check("occupancy is PER LAYER: both layers are admittable in their warm-up",
              r6 is not None and r6['admittable'] == 2 and r6['layers'] == 2)

    print(f"\nSELFTEST {'PASSED' if bad == 0 else 'FAILED'} ({ok + bad} checks, {bad} failure(s))")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--leg', action='append', default=None,
                    help='a trace to analyse (repeatable; default: the eight recorded legs)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    results, refused = [], []
    for leg in (args.leg if args.leg else LEGS):
        r, why = analyse(leg)
        if r is None:
            refused.append((leg, why))
        else:
            results.append(r)
    report(results, refused)
    if not results:
        print("\nREFUSED: no leg could be scored (exit 2, no numbers)")
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
