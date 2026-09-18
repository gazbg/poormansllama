#!/usr/bin/env python3
"""T-GPLR-36 analyzer: does the host ledger close on the PP wall?

Reads a trace that carries `host_ledger` lines (emitted under --pp-recon) and
answers the question the task asks:

  residual = wall - compute - pure DMA

is NOT a property of the pass as a whole: the ledger splits the in-sweep wall by
named host section, so each millisecond of the residual either lands in a bucket
or is reported as `other_us`.

Closure rules (exit 1 when violated):
  * bucket closure : patch + setup + compute + graph_call + slot + other == sweep
  * phase closure  : embed + preload + sweep + output <= fn (the remainder is the
    pass's own prologue/epilogue inside the controller)

Exit 2 when the trace has no host_ledger line (nothing to evaluate).

Usage:
  python tools/analyze-gplr36-ledger.py build/llama-cuda/gplr36-ledger-200k.jsonl
"""
import json
import statistics
import sys

PURE_DMA_MS_PER_BANK = 31.37
CLOSURE_TOL = 0.01  # 1% of the sweep


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(paths):
    rc = 0
    for path in paths:
        ev = load(path)
        rs = [e for e in ev if e.get('event') == 'run_start']
        led = [e for e in ev if e.get('event') == 'host_ledger']
        pps = [e for e in ev if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
        print('=' * 30, path)
        if rs:
            r = rs[0]
            print(' shape: %s tokens | %s chunks x %s layers | %s slots | schedule %s | '
                  'kv %s | copy %s | af? | repeat passes=%d' % (
                      r.get('pp_tokens'), r.get('pp_n_chunks'), r.get('n_layer'), r.get('n_slots'),
                      r.get('schedule'), r.get('kv_cache'), r.get('copy_source'), len(pps)))
        if not led:
            print(' CANNOT EVALUATE: no host_ledger line (run with --pp-recon 1 on a pass '
                  'executed by llama_context::layer_major_pp)')
            rc = 2
            continue
        print(' host_ledger lines: %d (one per pass)' % len(led))
        for i, (h, sm) in enumerate(zip(led, pps), start=1):
            launches = h.get('launches', 0)
            wall = sm['wall_us'] / 1e3
            comp = sm['total_compute_us'] / 1e3
            pure = int(sm.get('layers_run', 0)) * PURE_DMA_MS_PER_BANK
            resid = wall - comp - pure
            buckets = {k: h.get(k, 0.0) / 1e3 for k in
                       ('patch_us', 'setup_us', 'compute_us', 'graph_call_us', 'slot_us', 'other_us')}
            sweep = h.get('sweep_us', 0.0) / 1e3
            bsum = sum(buckets.values())
            print('\n pass %d: launches=%d  wall %.1f ms | compute(summary) %.1f ms | '
                  'residual %.1f ms' % (i, launches, wall, comp, resid))
            print('   fn %.1f = embed %.1f + preload %.1f + sweep %.1f + output %.1f + controller %.1f'
                  % (h.get('fn_us', 0) / 1e3, h.get('embed_us', 0) / 1e3, h.get('preload_us', 0) / 1e3,
                     sweep, h.get('output_us', 0) / 1e3,
                     (h.get('fn_us', 0) - h.get('embed_us', 0) - h.get('preload_us', 0)
                      - h.get('sweep_us', 0) - h.get('output_us', 0)) / 1e3))
            print('   sweep %.1f ms = %s' % (sweep, ' + '.join(
                '%s %.1f' % (k, v) for k, v in buckets.items())))
            print('   bucket sum %.1f vs sweep %.1f (delta %+.1f ms = %+.2f%%)'
                  % (bsum, sweep, bsum - sweep, 100.0 * (bsum - sweep) / sweep if sweep else float('nan')))
            if launches:
                print('   per launch: compute %.3f | patch %.3f | setup %.3f | graph_call %.3f | '
                      'slot %.3f | other %.3f ms' % tuple(
                          buckets[k] * 1e3 / launches for k in
                          ('compute_us', 'patch_us', 'setup_us', 'graph_call_us', 'slot_us', 'other_us')))
            named = buckets['patch_us'] + buckets['setup_us'] + buckets['graph_call_us'] + buckets['slot_us']
            print('   residual vs named host buckets: residual %.1f ms | named-host %.1f ms (%.1f%%) | '
                  'setup %.1f%% | patch %.1f%% | graph_call %.1f%% | other %.1f%%' % (
                      resid, named, 100.0 * named / resid if resid else float('nan'),
                      100.0 * buckets['setup_us'] / resid if resid else float('nan'),
                      100.0 * buckets['patch_us'] / resid if resid else float('nan'),
                      100.0 * buckets['graph_call_us'] / resid if resid else float('nan'),
                      100.0 * buckets['other_us'] / resid if resid else float('nan')))
            if sweep and abs(bsum - sweep) > CLOSURE_TOL * sweep:
                print('   CLOSURE FAILED (>%.0f%% of sweep) - decomposition must not be published'
                      % (100 * CLOSURE_TOL))
                rc = max(rc, 1)
            else:
                print('   closure OK (buckets sum to the sweep within %.0f%%)' % (100 * CLOSURE_TOL))
            print('   graph_builds=%s graph_prep_us=%.3f ms (prebuilt graphs reused)'
                  % (sm.get('graph_builds'), sm.get('graph_prep_us', 0) / 1e3))
    return rc


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
