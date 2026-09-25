#!/usr/bin/env python3
"""CLI and self-tests for the shared T-GPLR-59 host pool replay."""

import argparse
import json
import os
import sys
import tempfile

from tg_pool_core import (
    POOL_BUDGET,
    SCHEMA,
    PoolLayer,
    RealCacheLayer,
    Refusal,
    _fnv,
    analyze,
    load_trace,
    replay_pool,
    replay_real,
)


def _pct(value):
    return 'n/a' if value is None else '%.2f%%' % (100.0 * value)


def _mib(value):
    return '%.2f MiB' % (value / (1024.0 * 1024.0))


def print_report(report, stream):
    write = stream.write
    write('T-GPLR-59 pool counterfactual replay (%s)\n' % report['schema'])
    write('mode: %s | completion: %s\n' % (report['mode'], report['completion_assumption']))
    write('budgets: pool positions %d | per-stage admission bound %d\n'
          % (report['pool_budget'], report['admission_bound']))
    for leg in report['legs']:
        write('\n[%s] %s\n' % (leg['role'], leg['path']))
        write('  model %s | shape n_expert=%d top_k=%d retain_k=%d (real cap %d) | '
              'row_bytes %s | source budget %s\n'
              % (leg['model'], leg['n_expert'], leg['top_k'], leg['retain_k'],
                 leg['real_capacity'], leg['row_bytes_distinct'], leg['source_label']))
        write('  demand: %d rows, %d ids | stream %s | rows %s\n'
              % (leg['demand_rows'], leg['demand_ids'], leg['stream_signature'],
                 leg['demand_rows_signature']))
        write('  real replay: %s (%d parity stages, %d divergent, %d fill rows applied)\n'
              % (leg['real_replay_status'], leg['real_parity_stages'],
                 leg['real_divergent_stages'], leg['real_fill_rows_applied']))
        write('  real cache: hits %d fetched %d evicted %d share %s\n'
              % (leg['real_hits'], leg['real_fetched'], leg['real_evicted'],
                 _pct(leg['real_share'])))
        pool = leg['pool']
        write('  pool lifecycle: proposals %d admitted %d | skipped real %d pool %d dup %d '
              'invalid %d | refused budget %d capacity %d | evictions %d\n'
              % (pool['proposals'], pool['admitted'], pool['skipped_real'],
                 pool['skipped_pool'], pool['skipped_dup'], pool['skipped_invalid'],
                 pool['refused_budget'], pool['refused_capacity'], pool['evictions']))
        write('  pool fates: hits %d pending-fallbacks %d gen-cancellations %d '
              '(stale-retirements %d) evictions %d live-at-end %d\n'
              % (pool['hits'], pool['pending_fallbacks'], pool['generation_cancellations'],
                 pool['stale_retirements'], pool['evictions'], pool['live_at_end']))
        write('  pool lifecycle: modeled completions %d, modeled completion WAITS %d, '
              'unused final proposals %d (%d ids), max occupancy %d/12\n'
              % (pool['modeled_completions'], pool['modeled_completion_waits'],
                 pool['unused_final_proposals'], pool['unused_final_candidates'],
                 pool['max_occupancy']))
        write('  closures: candidate identity %s | fate identity %s | occupancy %s | '
              'no in-flight overwrite %s\n'
              % (leg['closures']['per_stage_candidate_identity'],
                 leg['closures']['fate_identity'], leg['closures']['occupancy_bound_le_12'],
                 leg['closures']['no_inflight_overwrite']))
        write('  pool hits: %d | hit rate over admitted %s | over resolved %s | '
              'extra share %s | total share %s\n'
              % (pool['hits'], _pct(leg['hit_rate_admitted']), _pct(leg['hit_rate_resolved']),
                 _pct(leg['extra_share']), _pct(leg['total_share'])))
        write('  outcomes: next-demand fallbacks %d | later-carryover hits %d '
              '(lead 2: %d, lead >= 3: %d) | lead hist %s\n'
              % (leg['next_demand_fallbacks'], leg['carryover_hits'], leg['hits_lead_2'],
                 leg['hits_lead_ge_3'], pool['lead_hist']))
        write('  bytes: H2D %s | D2D %s | net_h2d_delta %s\n'
              % (_mib(leg['h2d_bytes']), _mib(leg['d2d_bytes']),
                 _mib(leg['net_h2d_bytes'])))
    paired = report.get('paired_deltas')
    if paired:
        write('\npaired arm-null (deltas are arm minus null):\n')
        for name, row in paired['metrics'].items():
            write('  %-18s arm %s | null %s | delta %s\n'
                  % (name, row['arm'], row['null'], row['delta_arm_minus_null']))
    write('\nevidence limitations:\n')
    for line in report['evidence_limitations']:
        write('  - %s\n' % line)


# ------------------------------------------------------------------- selftest
class Selftest(object):
    def __init__(self):
        self.ok = 0
        self.failures = []

    def check(self, condition, name):
        if condition:
            self.ok += 1
            print('[ok]   %s' % name)
        else:
            self.failures.append(name)
            print('[FAIL] %s' % name)

    def equal(self, got, want, name):
        self.check(got == want, '%s (got %r, want %r)' % (name, got, want))

    def refusal(self, fn, name, fragment=None):
        try:
            fn()
        except Refusal as exc:
            if fragment is not None and fragment not in str(exc):
                self.failures.append(name)
                print('[FAIL] %s (wrong cause: %s)' % (name, exc))
                return
            self.ok += 1
            print('[ok]   %s' % name)
            return
        self.failures.append(name)
        print('[FAIL] %s (no refusal)' % name)


def _fnv_json(text):
    return str(_fnv(text))


def _synth(path, stages, n_expert=32, top_k=2, retain_k=2, budget=2,
            model='synthetic', mode='overlay', sabotage=None, omit_candidates=(),
            omit_fills=(), extra_run_start=False):
    """Write a hermetic mode-2 trace. Stage dicts carry layer/generation/phase/
    union/candidates/fills; the recorded retention counts are produced by the
    very real-cache model the replay mirrors (the five recorded legs are the
    external anchor for that model)."""
    row_bytes = 256
    cap = min(n_expert, retain_k * top_k)
    layers = {}
    computed = []
    for index, stage in enumerate(stages):
        layer = layers.setdefault(stage.get('layer', 0), RealCacheLayer(cap))
        layer.boundary(stage.get('generation', 1), stage.get('phase', 1))
        fills = list(stage.get('fills', [])) if mode == 'overlay' else []
        hits, evicted, _ = layer.stage(list(stage['union']), fills)
        computed.append({'hits': len(hits), 'fetched': len(stage['union']) - len(hits),
                         'retired': evicted})
    with open(path, 'w', encoding='utf-8') as fh:
        run_start = {'event': 'run_start', 'run_id': 1, 't0_us': 0, 'model': model,
                     'rowset': 'on', 'row_spec_mode': 2, 'row_spec_budget': budget}
        fh.write(json.dumps(run_start) + '\n')
        if extra_run_start:
            fh.write(json.dumps(run_start) + '\n')
        fh.write(json.dumps({'event': 'rowset_init', 'run_id': 1, 'policy': 'moe-row-set',
                             'n_slots': 1, 'slot_bytes': n_expert * row_bytes,
                             'window_tokens': 1, 'n_expert': n_expert, 'top_k': top_k,
                             'retain_k': retain_k}) + '\n')
        for index, stage in enumerate(stages):
            ubatch = stage.get('ubatch', index)
            layer_id = stage.get('layer', 0)
            candidates = list(stage.get('candidates', []))
            fills = list(stage.get('fills', []))
            spec = {'event': 'spec', 'run_id': 1, 'ubatch': ubatch, 'layer': layer_id,
                    'mode': 2, 'budget': budget, 'cap': cap, 'prop': len(candidates),
                    'arm_count': len(candidates), 'submitted': len(fills),
                    'skipped_present': 0, 'skipped_dup': 0, 'plan_refused_capacity': 0,
                    'consumed': 0, 'matched_by_demand': 0, 'cancelled_demand': 0,
                    'cancelled_generation': 0, 'cancelled_capacity': 0, 'dup_refused': 0,
                    'stop': 0, 'stop_reason': 'proposals'}
            if index not in omit_candidates:
                spec['candidate_ids'] = candidates
            if index not in omit_fills:
                spec['fill_ids'] = fills
            # the emitter writes spec BEFORE its rowset_fetch: opposite line order
            fh.write(json.dumps(spec) + '\n')
            union_text = ','.join(str(x) for x in stage['union'])
            counts = computed[index]
            fetch = {'event': 'rowset_fetch', 'run_id': 1, 'ubatch': ubatch,
                     'layer': layer_id, 'slot': 0,
                     'generation': stage.get('generation', 1), 'chunk': 0,
                     'phase': stage.get('phase', 1), 'n_tokens': 1, 'top_k': top_k,
                     'route_union': len(stage['union']), 'hash': _fnv_json(union_text),
                     'n_rows_fetched': counts['fetched'],
                     'n_rows_resident': counts['hits'],
                     'n_rows_retained': counts['hits'],
                     'n_rows_retired': counts['retired'],
                     'bytes_staged': counts['fetched'] * row_bytes,
                     'bytes_full_bank': n_expert * row_bytes,
                     'union_ids': union_text}
            if sabotage is not None:
                sabotage(index, fetch)
            fh.write(json.dumps(fetch) + '\n')
    return path


def selftest():
    t = Selftest()
    tmp = tempfile.mkdtemp(prefix='gplr59-pool-')

    def p(name):
        return os.path.join(tmp, name)

    def run(pairs, mode='overlay'):
        return analyze(pairs, mode)

    def one(name, stages, **kw):
        path = _synth(p(name), stages, **kw)
        return path

    # --- A: 12-position capacity, 12-item admission bound, duplicate
    #     suppression, READY eviction, identity closure, occupancy, unused final.
    stage_a = [
        {'layer': 0, 'union': [0, 1], 'candidates': [2, 2, 3, 4]},
        {'layer': 0, 'union': [0, 1], 'candidates': [0, 2, 5]},
        {'layer': 0, 'union': [0, 1], 'candidates': list(range(6, 21))},
        {'layer': 0, 'union': [0, 1], 'candidates': [21, 22]},
    ]
    report = run([('arm', one('a.jsonl', stage_a))])
    pool = report['legs'][0]['pool']
    t.equal(pool['max_occupancy'], 12, '12-position capacity: max occupancy is exactly 12')
    t.equal(pool['admitted'], 16, '12-item admission bound: 3 + 1 + 12 admitted')
    t.equal(pool['refused_budget'], 3, '12-item admission bound: tail refused_budget')
    t.equal(pool['evictions'], 4, 'READY-only LRU eviction: four READY victims')
    t.equal(pool['skipped_dup'], 1, 'publication duplicate suppression')
    t.equal(pool['skipped_real'], 1, 'real duplicate suppression')
    t.equal(pool['skipped_pool'], 1, 'pool duplicate suppression')
    t.equal(pool['proposals'], 22, 'candidate identity closure: every proposal classified')
    t.equal(pool['unused_final_proposals'], 1, 'unused final proposal: one per layer')
    t.equal(pool['unused_final_candidates'], 2, 'unused final proposal: ids reported')
    t.equal(report['legs'][0]['closures']['fate_identity'], 'ok',
            'candidate/admitted identity closure: fate closure ok')
    t.equal(report['legs'][0]['closures']['occupancy_bound_le_12'], 'ok',
            'occupancy bounds: never above 12')
    t.equal(pool['inflight_overwrites'], 0, 'occupancy bounds: no PENDING/CANCELLED reuse')

    # --- B: READY hit first at lead 2 vs PENDING next-demand fallback without wait.
    stage_b = [
        {'layer': 0, 'union': [0, 1], 'candidates': [2, 3]},
        {'layer': 0, 'union': [2, 3], 'candidates': [4, 5]},
        {'layer': 0, 'union': [0, 1], 'candidates': [6, 7]},
        {'layer': 0, 'union': [4, 5], 'candidates': [8, 9]},
        {'layer': 0, 'union': [0, 1], 'candidates': [10, 11]},
    ]
    report = run([('arm', one('b.jsonl', stage_b))])
    pool = report['legs'][0]['pool']
    t.equal(pool['hits'], 2, 'READY hit first at lead 2: carryover hits land')
    t.equal(report['legs'][0]['hits_lead_2'], 2, 'READY hit first at lead 2: all leads are 2')
    t.equal(pool['lead_hist'], {2: 2}, 'READY hit first at lead 2: lead histogram')
    t.equal(pool['pending_fallbacks'], 2,
            'PENDING next-demand fallback without wait: fallbacks counted, not hits')
    t.equal(pool['modeled_completion_waits'], 0,
            'PENDING next-demand fallback without wait: zero waits')
    t.equal(pool['hits'] + pool['pending_fallbacks'], 4,
            'PENDING fallback is never folded into hits')

    # --- C: generation cancellation (stale READY retires, stale PENDING cancels).
    stage_c = [
        {'layer': 0, 'generation': 1, 'union': [0, 1], 'candidates': [2, 3, 4]},
        {'layer': 0, 'generation': 1, 'union': [0, 1], 'candidates': [5]},
        {'layer': 0, 'generation': 2, 'union': [0, 1], 'candidates': [6]},
    ]
    report = run([('arm', one('c.jsonl', stage_c))])
    pool = report['legs'][0]['pool']
    t.equal(pool['stale_retirements'], 3, 'generation cancellation: stale READY retires')
    t.equal(pool['generation_cancelled_pending'], 1,
            'generation cancellation: stale PENDING cancels')
    t.equal(pool['generation_cancellations'], 4,
            'generation cancellation: closure bucket covers both')

    # --- D: no real-cache promotion (pool replay must not mutate frozen real state).
    path = one('d.jsonl', stage_b)
    leg = load_trace(path, 'overlay')
    real = replay_real(leg)
    frozen = [(set(s['real_hit_ids']), list(s['miss_ids']), set(s['real_rows_after']))
              for s in leg['stages']]
    replay_pool(leg)
    after = [(set(s['real_hit_ids']), list(s['miss_ids']), set(s['real_rows_after']))
             for s in leg['stages']]
    t.equal(after, frozen, 'no real-cache promotion: frozen real facts unchanged')
    t.equal(leg['stages'][3]['real_hit_ids'], [],
            'no real-cache promotion: the pool-hit demand is still a real miss')
    t.check(set([4, 5]) <= set(leg['stages'][3]['miss_ids']),
            'no real-cache promotion: pool hits stay out of the real cache')
    t.equal(real['real_hits'] + real['real_fetched'], 10,
            'no real-cache promotion: real identity route_union == hits + fetched')

    # --- E: overlay hit parity (fills lifecycle) and the parity refusal.
    stage_e = [
        {'layer': 0, 'union': [0, 1], 'fills': [4, 5], 'candidates': []},
        {'layer': 0, 'union': [4, 0], 'candidates': []},
        {'layer': 0, 'union': [5, 0], 'candidates': []},
    ]
    report = run([('arm', one('e.jsonl', stage_e))])
    leg = report['legs'][0]
    t.equal(leg['real_replay_status'], 'parity-ok', 'overlay hit parity: status is parity-ok')
    t.equal(leg['real_parity_stages'], 3, 'overlay hit parity: every stage matched')
    t.equal(leg['real_hits'], 3, 'overlay hit parity: fill hit at lead 2, demand dup at 1')
    t.equal(leg['real_divergent_stages'], 0, 'overlay hit parity: zero divergent stages')
    t.refusal(lambda: run([('arm', one('e-bad.jsonl', stage_e,
                                       sabotage=lambda i, f: f.update(
                                           n_rows_retired=f['n_rows_retired'] + 1)
                                       if i == 2 else None))]),
              'overlay hit parity: a divergent stage refuses the pool result',
              'overlay replay parity failed')

    # --- F: refusals (missing candidates / fill ids, mixed streams, bad bytes).
    t.refusal(lambda: run([('arm', one('f1.jsonl', stage_b, omit_candidates=(0,)))]),
              'refusal: missing raw candidates', 'missing raw candidates')
    t.check(run([('arm', one('f2.jsonl', stage_b))])['legs'][0]['demand_rows'] == 5,
            'baseline fixture loads through the full pipeline')
    t.refusal(lambda: run([('arm', one('f3.jsonl', stage_b, omit_fills=(0,)))]),
              'refusal: missing fill_ids in overlay mode', 'needs fill_ids')
    report = run([('off', one('f4.jsonl', stage_b, omit_fills=(0, 1, 2, 3, 4)))],
                 mode='demand-only')
    t.equal(report['legs'][0]['real_replay_status'], 'diagnostic (demand-only)',
            'refusal: demand-only accepts old traces as a diagnostic')
    t.refusal(lambda: run([('arm', one('f5.jsonl', stage_b, extra_run_start=True))]),
              'refusal: mixed traces (two run_start)', 'mixed traces')
    t.refusal(lambda: run([('arm', one('f6.jsonl', stage_b,
                                       sabotage=lambda i, f: f.update(
                                           bytes_staged=f['bytes_staged'] + 1)
                                       if i == 1 else None))]),
              'refusal: inconsistent byte arithmetic', 'inconsistent byte arithmetic')
    other = [
        {'layer': 0, 'union': [3, 4], 'candidates': [0, 1]},
        {'layer': 0, 'union': [0, 1], 'candidates': [2, 3]},
        {'layer': 0, 'union': [2, 3], 'candidates': [4, 5]},
    ]
    t.refusal(lambda: run([('arm', one('f7.jsonl', stage_b)),
                           ('null', one('f8.jsonl', other))]),
              'refusal: mismatched arm/null demand streams',
              'mismatched arm/null demand streams')
    t.refusal(lambda: run([('arm', one('f9.jsonl', stage_b)),
                           ('arm', one('f10.jsonl', stage_b))]),
              'refusal: a role declared twice', 'declared twice')

    # --- G: arm/null paired deltas on a matched pair.
    null_stages = [
        {'layer': 0, 'union': [0, 1], 'candidates': [7, 8]},
        {'layer': 0, 'union': [2, 3], 'candidates': [20, 21]},
        {'layer': 0, 'union': [0, 1], 'candidates': [9, 10]},
        {'layer': 0, 'union': [4, 5], 'candidates': [11, 12]},
        {'layer': 0, 'union': [0, 1], 'candidates': [13, 14]},
    ]
    report = run([('arm', one('g-arm.jsonl', stage_b)),
                  ('null', one('g-null.jsonl', null_stages))])
    paired = report['paired_deltas']
    t.check(paired is not None, 'paired deltas: produced for one arm and one null')
    t.equal(paired['metrics']['pool_hits']['arm'], 2,
            'paired deltas: per-leg pool hits reported')
    t.equal(paired['metrics']['pool_hits']['delta_arm_minus_null'], 2,
            'paired deltas: arm minus null pool hits')

    # --- H: unit cases unreachable through the stage pipeline.
    pool_u = PoolLayer(12)
    ok_all = all(pool_u.admit(e, 1)[0] for e in range(12))
    ok, evicted_ord = pool_u.admit(12, 1)
    t.check(ok_all and not ok and evicted_ord == -1,
            'all-PENDING refusal: full pool with no READY victim refuses')
    t.check(all(pool_u.pos[e]['state'] == PoolLayer.PENDING for e in range(12)),
            'all-PENDING refusal: no PENDING bytes overwritten')
    pool_v = PoolLayer(2)
    pool_v.admit(5, 1)          # issued first
    pool_v.reap()               # completes -> READY
    pool_v.admit(6, 1)          # PENDING, never a victim
    ok, evicted_ord = pool_v.admit(7, 1)
    t.check(ok and evicted_ord == 0 and pool_v.pos[0]['expert'] == 7 and
            pool_v.pos[1]['state'] == PoolLayer.PENDING,
            'READY-only LRU eviction: least-recently-issued READY loses, PENDING kept')

    print('\nselftest: %d checks, %d failures' % (t.ok, len(t.failures)))
    return 0 if not t.failures else 1


# ------------------------------------------------------------------------ CLI
def _main(argv):
    parser = argparse.ArgumentParser(
        description='T-GPLR-59 host-only counterfactual replay of the 12-position '
                    'prediction pool over recorded mode-2 traces')
    parser.add_argument('--leg', action='append', default=[], metavar='ROLE=PATH',
                        help='explicit session role declaration (arm=null=off=PATH); '
                             'repeatable, at most one arm and one null')
    parser.add_argument('--mode', choices=('overlay', 'demand-only'), default='overlay')
    parser.add_argument('--json', metavar='PATH', default=None,
                        help='also write the %s report to PATH' % SCHEMA)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.leg:
        parser.error('at least one --leg ROLE=PATH is required')
    pairs = []
    for item in args.leg:
        if '=' not in item:
            raise Refusal('malformed --leg %r (expected ROLE=PATH)' % item)
        role, path = item.split('=', 1)
        pairs.append((role, path))
    report = analyze(pairs, args.mode)
    print_report(report, sys.stdout)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, sort_keys=False)
            fh.write('\n')
    return 0


def main(argv):
    try:
        return _main(argv)
    except Refusal as exc:
        print('REFUSED (no measurement): %s' % exc, file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - unexpected internal error, loud and traced
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
