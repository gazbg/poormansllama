#!/usr/bin/env python3
"""T-GPLR-53 arm A1: the pass-1 gap attribution table (host-only, no GPU).

WHY THIS EXISTS
  The 2026-09-21 sweep left one item open: `phase_summary` unaccounted
  (wall - total_compute_us - total_wait_us) is 2.3-2.9 s on pass 1 at 50k/200k
  and ~0.2-0.6 s later, and the sweep could only say it was "neither staging nor
  compute" (both are flat across passes). The T-GPLR-53 drafting recon showed
  the gap IS the embed step - `host_ledger.embed_us`, step 1 of layer-major PP,
  which `phase_summary` never sums into layer stats - and that the T-GPLR-36
  ledger double-books it (its per-launch buckets accumulate embed+output
  launches while `sweep_us` starts after the embed step). This tool makes that
  attribution reproducible instead of a drafting-session hand calculation.

WHAT IT PRINTS, PER TRACE
  For every PP pass (a `host_ledger` paired by order with the `phase_summary`
  of the pass it closes):

      wall  gap  embed  preload  output  rest

  where gap = wall - total_compute_us - total_wait_us, and
  rest = gap - embed - preload - output (the sweep's named host buckets, which
  are pass-invariant in the mmap-registered traces).

CHECKED EXPECTATIONS (falsifiable; the pre-registered bands are in the plan
block T-GPLR-53 under "Established facts" F1/F2 and F6):
  * CHK-gap   (structural) gap == embed+preload+output+rest, |residual| <= 1 us
  * CHK-led   (F6)         |loop_us + embed_us + output_us| <= 5 ms
  * CHK-embed (F1, mmap)   embed share of the pass1->pass2 gap excess >= 0.95
  * CHK-pin   (F3, pinned) embed share of the same excess <= 0.05 (the pinned
                           path pays its first-pass cost in slot/preload, not
                           in the embed step)
  * CHK-rest  (F1b, mmap)  max |rest_i - rest_2| / rest_2 <= 0.10
  A ledger that predates `loop_us` (the field landed with T-GPLR-39) SKIPS CHK-led
  with a printed note - it did not exist then, so F6 is not evaluable there - and
  never fails it by treating the missing field as zero.
  One-sided on purpose: the captured failure mode is "the gap is NOT the embed
  step". Shares slightly ABOVE 1.0 are pre-T-45-era bucket noise (measured
  0.97-1.03 across trace generations), not a falsification, and are printed.

KNOWN-CONFOUNDED TRACES (table only, expectation checks skipped, reason printed)
  gplr36-remeasure-200k : pre-T-47 binary, its pass 2 carries the T-46/T-47
                          pass-2 compute regression (wall 206 s), so the pass
                          pair is not a clean warm reference.

PAIRING RULE (measured): a trace emitted under the per-pass rebuild carries one
`host_ledger` per PP pass, in file order before that pass's `phase_summary`;
the tool requires equal counts and refuses loudly otherwise (it never guesses).

USAGE
  python tools/analyze-gplr53-gap.py TAG1 TAG2 ...   # build/llama-cuda/<TAG>.jsonl
  python tools/analyze-gplr53-gap.py --placement     # re-derive F7 from the code
  python tools/analyze-gplr53-gap.py --selftest      # synthetic fixtures + controls

  Trailing `--json` prints the rows as JSON for later joins.

EXIT CODES
  0 = PASS  every evaluated trace satisfied its expectations
  1 = FAIL  at least one checked expectation failed (named in the output)
  2 = cannot evaluate: no arguments, a missing trace, no `host_ledger` in a
      trace, a ledger/summary count mismatch, or no `run_start` (a run that
      died before resolution is NOT a pass)
"""
import io
import json
import os
import sys

LAYERS = 40  # Qwen3.6-35B-A3B decoder layers (shape census context for the table)

CONFOUNDED = {
    'gplr36-remeasure-200k':
        'pre-T-47 binary: pass 2 carries the T-46/T-47 compute regression (wall 206 s)',
}

MMAP_SHARE_MIN = 0.95   # F1
PINNED_SHARE_MAX = 0.05  # F3
REST_DRIFT_MAX = 0.10    # F1b
LEDGER_RESID_MAX_US = 5000.0  # F6, |loop + embed + output|
GAP_RESID_MAX_US = 1.0   # structural


def load_events(path):
    events = []
    with io.open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '"event"' not in line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def us(v, default=0.0):
    return float(v) if v is not None else default


def rows_for(path):
    """Return (rows, reason). reason is set when the trace cannot be evaluated."""
    events = load_events(path)
    run_start = next((e for e in events if e.get('event') == 'run_start'), None)
    if run_start is None:
        return None, 'no run_start (trace died before resolution)'
    summaries = [e for e in events if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    ledgers = [e for e in events if e.get('event') == 'host_ledger']
    if not ledgers:
        return None, ('no host_ledger in trace (rebuild-off run? nothing to evaluate - '
                      'this is exit 2, NOT a pass)')
    if not summaries:
        return None, 'no pp phase_summary in trace'
    if len(ledgers) != len(summaries):
        return None, ('ledger/summary count mismatch: %d host_ledger vs %d pp phase_summary '
                      '- pairing is refused, never guessed' % (len(ledgers), len(summaries)))

    rows = []
    for i, (led, sm) in enumerate(zip(ledgers, summaries)):
        wall = us(sm.get('wall_us')) / 1000.0
        comp = us(sm.get('total_compute_us')) / 1000.0
        wait = us(sm.get('total_wait_us')) / 1000.0
        gap = wall - comp - wait
        embed = us(led.get('embed_us')) / 1000.0
        preload = us(led.get('preload_us')) / 1000.0
        output = us(led.get('output_us')) / 1000.0
        rest = gap - embed - preload - output
        loop = us(led.get('loop_us')) / 1000.0
        rows.append({
            'pass': i + 1,
            'wall_ms': wall, 'gap_ms': gap, 'embed_ms': embed,
            'preload_ms': preload, 'output_ms': output, 'rest_ms': rest,
            'loop_ms': loop,
            'has_loop': 'loop_us' in led,
            'ledger_resid_ms': (loop + embed + output) if 'loop_us' in led else float('nan'),
            'slot_ms': us(led.get('slot_us')) / 1000.0,
            'patch_ms': us(led.get('patch_us')) / 1000.0,
            'other_ms': us(led.get('other_us')) / 1000.0,
            'route_ms': us(led.get('route_us')) / 1000.0,
            'tokens': sm.get('tokens'), 'ubatches': sm.get('ubatches'),
            'tokens_per_s': sm.get('tokens_per_s'),
        })
    meta = {
        'trace': os.path.basename(path).rsplit('.', 1)[0],
        'pp_tokens': run_start.get('pp_tokens'),
        'copy_source': run_start.get('copy_source'),
        'copy_source_source': run_start.get('copy_source_source'),
        'kv_cache': run_start.get('kv_cache'),
        'slots': run_start.get('n_slots'),
        'prebuild_per_pass': run_start.get('pp_prebuild_per_pass'),
    }
    return {'meta': meta, 'rows': rows}, None


def check_trace(info):
    """Return (verdict, findings) with verdict in {EVAL, CONFOUNDED}."""
    meta, rows = info['meta'], info['rows']
    tag = meta['trace']
    findings = []
    if tag in CONFOUNDED:
        return 'CONFOUNDED', [('SKIP', 'expectation checks skipped: ' + CONFOUNDED[tag])]

    # structural: gap decomposes exactly by construction
    worst = max(abs(r['gap_ms'] - (r['embed_ms'] + r['preload_ms'] + r['output_ms'] + r['rest_ms']))
                for r in rows) * 1000.0
    findings.append(('PASS' if worst <= GAP_RESID_MAX_US else 'FAIL',
                     'CHK-gap structural residual %.3f us (<= %.0f)' % (worst, GAP_RESID_MAX_US)))

    # F6 ledger identity (only evaluable on ledgers that carry loop_us; the
    # field landed with T-GPLR-39, so pre-T-39 traces are skipped, not failed)
    if not all(r['has_loop'] for r in rows):
        findings.append(('SKIP', 'CHK-led: ledger predates loop_us (pre-T-GPLR-39) - '
                                 'F6 is not evaluable on this trace'))
    else:
        worst_led = max(abs(r['ledger_resid_ms']) for r in rows) * 1000.0
        findings.append(('PASS' if worst_led <= LEDGER_RESID_MAX_US else 'FAIL',
                         'CHK-led |loop+embed+output| %.2f ms (<= %.0f) - F6' %
                         (worst_led / 1000.0, LEDGER_RESID_MAX_US / 1000.0)))

    if len(rows) < 2:
        findings.append(('INFO', 'single pass: no pass-2 reference, excess checks skipped'))
        return 'EVAL', findings

    p1, p2 = rows[0], rows[1]
    gap_ex = p1['gap_ms'] - p2['gap_ms']
    embed_ex = p1['embed_ms'] - p2['embed_ms']
    if abs(gap_ex) < 1.0:
        findings.append(('INFO', 'pass1-pass2 gap excess is ~0: no attribution claim to check'))
        return 'EVAL', findings
    share = embed_ex / gap_ex
    if meta['copy_source'] == 'mmap-registered':
        findings.append(('PASS' if share >= MMAP_SHARE_MIN else 'FAIL',
                         'CHK-embed mmap embed share of the +%.1f ms gap excess: %.4f '
                         '(>= %.2f) - F1' % (gap_ex, share, MMAP_SHARE_MIN)))
        drift = max(abs(r['rest_ms'] - p2['rest_ms']) / p2['rest_ms'] for r in rows) \
            if p2['rest_ms'] > 1.0 else float('nan')
        ok = (drift != drift) or drift <= REST_DRIFT_MAX
        findings.append(('PASS' if ok else 'FAIL',
                         'CHK-rest max rest drift %.1f%% (<= %.0f%%) - F1b' %
                         (100.0 * drift, 100.0 * REST_DRIFT_MAX)))
    else:
        ok = share <= PINNED_SHARE_MAX
        findings.append(('PASS' if ok else 'FAIL',
                         'CHK-pin pinned embed share of the +%.1f ms gap excess: %.4f '
                         '(<= %.2f); slot+preload excess %+.1f ms - F3' %
                         (gap_ex, share, PINNED_SHARE_MAX,
                          (p1['slot_ms'] - p2['slot_ms']) + (p1['preload_ms'] - p2['preload_ms']))))
    return 'EVAL', findings


def print_trace(info, findings, verdict):
    meta, rows = info['meta'], info['rows']
    print('== %s ==' % meta['trace'])
    print('  pp_tokens=%s source=%s (%s) kv=%s slots=%s prebuild_per_pass=%s passes=%d' % (
        meta['pp_tokens'], meta['copy_source'], meta['copy_source_source'],
        meta['kv_cache'], meta['slots'], meta['prebuild_per_pass'], len(rows)))
    for r in rows:
        print('   p%d wall %9.1f | gap %8.1f = embed %8.1f + preload %6.2f + output %6.2f '
              '+ rest %7.1f | loop %9.2f | tok/s %7.1f' % (
                  r['pass'], r['wall_ms'], r['gap_ms'], r['embed_ms'], r['preload_ms'],
                  r['output_ms'], r['rest_ms'], r['loop_ms'], r['tokens_per_s'] or 0.0))
    if len(rows) >= 2:
        p1, p2 = rows[0], rows[1]
        print('   pass1-pass2 excess: gap %+8.1f | embed %+8.1f | slot %+8.1f | preload %+7.1f '
              '| rest %+8.1f' % (
                  p1['gap_ms'] - p2['gap_ms'], p1['embed_ms'] - p2['embed_ms'],
                  p1['slot_ms'] - p2['slot_ms'], p1['preload_ms'] - p2['preload_ms'],
                  p1['rest_ms'] - p2['rest_ms']))
    for level, text in findings:
        print('   [%s] %s' % (level, text))
    if verdict == 'EVAL' and any(f[0] == 'FAIL' for f in findings):
        print('   -> FAIL')
    else:
        print('   -> %s' % ('PASS (confounded: table only)' if verdict == 'CONFOUNDED' else 'PASS'))


def print_contrast(infos):
    """Source-contrast row set: same pp_tokens, different copy_source."""
    by_shape = {}
    for info in infos:
        by_shape.setdefault(info['meta']['pp_tokens'], []).append(info)
    printed = False
    for pp_tokens, group in sorted(by_shape.items(), key=lambda kv: kv[0] or 0):
        sources = set(i['meta']['copy_source'] for i in group)
        if len(sources) < 2:
            continue
        if not printed:
            print()
            print('SOURCE CONTRAST (same shape, different --copy-source):')
            printed = True
        print('  pp_tokens=%s' % pp_tokens)
        for info in sorted(group, key=lambda i: i['meta']['copy_source']):
            rows = info['rows']
            if len(rows) < 2:
                continue
            p1, p2 = rows[0], rows[1]
            print('   %-16s %-14s gap %+8.1f | embed %+8.1f | slot %+8.1f | preload %+7.1f '
                  '| rest %+8.1f' % (
                      info['meta']['trace'], info['meta']['copy_source'],
                      p1['gap_ms'] - p2['gap_ms'], p1['embed_ms'] - p2['embed_ms'],
                      p1['slot_ms'] - p2['slot_ms'], p1['preload_ms'] - p2['preload_ms'],
                      p1['rest_ms'] - p2['rest_ms']))


PLACEMENT_ANCHORS = [
    ('third_party/llama.cpp/tools/layer-streaming/main.cpp',
     ['mparams.n_gpu_layers = -1;',
      'mparams.tensor_buft_overrides = forced_off ? nullptr : overrides;',
      'ffn_(gate|up|down|gate_up)_exps']),
    ('third_party/llama.cpp/src/llama-layer-streaming.cpp',
     ['LS_CLS_STATIC = 0,',
      'n == "token_embd.weight" || n == "output_norm.weight" || n == "output.weight"']),
]


def placement(root):
    """F7 re-derived from the code: the embed step reads no host-mapped model page."""
    ok = True
    print('F7 placement check (the embed step\'s weights are CUDA-resident):')
    for rel, anchors in PLACEMENT_ANCHORS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print('  MISSING %s' % rel)
            ok = False
            continue
        lines = io.open(path, 'r', encoding='utf-8', errors='replace').read().splitlines()
        for a in anchors:
            hit = next((i + 1 for i, ln in enumerate(lines) if a in ln), None)
            if hit is None:
                print('  FAIL %s:%s anchor NOT FOUND: %s' % (rel, '?', a))
                ok = False
            else:
                print('  ok   %s:%d  %s' % (rel, hit, a))
    print('RESULT:', 'PASS - runner loads n_gpu_layers=-1 with only the pageable class overridden'
          if ok else 'FAIL - anchors moved; re-derive F7 before citing it')
    return 0 if ok else 1


# ---------------------------------------------------------------- self-test
def _fixture(path, meta, passes, legacy=False):
    """passes: list of dicts {wall, comp, wait, embed, preload, output, slot, loop_resid}.
    legacy=True omits the post-T-GPLR-39 loop_us/route_us/tel_us fields."""
    t = 1000
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict({'event': 'run_start', 'n_layer': LAYERS}, **meta)) + '\n')
        for i, p in enumerate(passes):
            t += 1000
            gap = p['wall'] - p['comp'] - p['wait']
            led = {
                'event': 'host_ledger', 'fn_us': p['wall'] * 1000.0,
                'embed_us': p['embed'] * 1000.0, 'preload_us': p['preload'] * 1000.0,
                'sweep_us': 0.0, 'output_us': p['output'] * 1000.0,
                'patch_us': 0.0, 'setup_us': 0.0, 'compute_us': 0.0,
                'graph_call_us': 0.0, 'slot_us': p['slot'] * 1000.0,
                'other_us': 0.0,
            }
            if not legacy:
                led.update({'route_us': 0.0, 'tel_us': 0.0,
                            'loop_us': -(p['embed'] + p['output'] + p['loop_resid']) * 1000.0})
            f.write(json.dumps(led) + '\n')
            f.write(json.dumps({
                'event': 'phase_summary', 'phase': 'pp', 'tokens': meta['pp_tokens'],
                'ubatches': 98, 'wall_us': p['wall'] * 1000.0, 'tokens_per_s': 1000.0,
                'total_compute_us': p['comp'] * 1000.0, 'total_wait_us': p['wait'] * 1000.0,
                'bytes_transferred': 0, 'evictions': 0, 'layers_run': LAYERS,
            }) + '\n')
    return path


def selftest():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'llama-cuda'))
    os.makedirs(base, exist_ok=True)
    ok = True

    def run(path, expect_verdict, expect_levels):
        nonlocal ok
        info, reason = rows_for(path)
        if info is None:
            got, findings = 'REFUSED(%s)' % reason.split('(')[0].strip(), []
        else:
            got, findings = check_trace(info)
        levels = [(f[0], f[1].split()[0].rstrip(':')) for f in findings]
        good = got.startswith(expect_verdict) and all(
            any(lv == want and name == key for lv, name in levels) for want, key in expect_levels)
        print('  %-46s -> %-10s %s' % (os.path.basename(path), got,
                                       'OK' if good else 'MISMATCH (expected %s %s; got %s)' %
                                       (expect_verdict, expect_levels, levels)))
        ok = ok and good

    mmap_meta = {'pp_tokens': 200000, 'copy_source': 'mmap-registered',
                 'copy_source_source': 'explicit', 'kv_cache': 'f16', 'n_slots': 3,
                 'pp_prebuild_per_pass': 1}
    pinned_meta = dict(mmap_meta, copy_source='pinned')

    # 1. clean mmap pair: the gap excess IS the embed excess (F1), rest invariant (F1b)
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-mmap.jsonl'), mmap_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 1800, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
    ])
    run(p, 'EVAL', [('PASS', 'CHK-embed'), ('PASS', 'CHK-rest'), ('PASS', 'CHK-led')])

    # 2. NEGATIVE CONTROL: mmap pair whose excess is in slot, embed flat -> CHK-embed FAIL
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-mmap-broken.jsonl'), mmap_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 1930, 'loop_resid': 0.2},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 150, 'loop_resid': 0.2},
    ])
    run(p, 'EVAL', [('FAIL', 'CHK-embed')])

    # 3. clean pinned pair: embed flat, the copy path carries the first-pass cost (F3)
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-pinned.jsonl'), pinned_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 330,
         'output': 10, 'slot': 1830, 'loop_resid': 0.3},
        {'wall': 8200, 'comp': 8000, 'wait': 50, 'embed': 19, 'preload': 123,
         'output': 10, 'slot': 320, 'loop_resid': 0.3},
    ])
    run(p, 'EVAL', [('PASS', 'CHK-pin')])

    # 4. NEGATIVE CONTROL: pinned pair with an embed-carried excess -> CHK-pin FAIL
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-pinned-broken.jsonl'), pinned_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 1800, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.3},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.3},
    ])
    run(p, 'EVAL', [('FAIL', 'CHK-pin')])

    # 5. NEGATIVE CONTROL: ledger identity violated by 50 ms -> CHK-led FAIL
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-ledger-broken.jsonl'), mmap_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 1800, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 50.0},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 50.0},
    ])
    run(p, 'EVAL', [('FAIL', 'CHK-led')])

    # 6. no host_ledger -> refuse (exit 2 in main; here: refuse, not crash)
    p = os.path.join(base, 'gplr53-a1-selftest-noledger.jsonl')
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(dict({'event': 'run_start', 'n_layer': LAYERS}, **mmap_meta)) + '\n')
        f.write(json.dumps({'event': 'phase_summary', 'phase': 'pp', 'tokens': 200000,
                            'wall_us': 10000000.0, 'total_compute_us': 8000000.0,
                            'total_wait_us': 50000.0}) + '\n')
    run(p, 'REFUSED', [])

    # 7. confounded trace: table printed, expectation checks skipped
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-confounded.jsonl'), mmap_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
    ])
    info, reason = rows_for(p)
    info['meta']['trace'] = 'gplr36-remeasure-200k'
    verdict, findings = check_trace(info)
    good = verdict == 'CONFOUNDED' and findings and findings[0][0] == 'SKIP'
    print('  %-46s -> %-10s %s' % ('selftest-confounded (renamed)', verdict,
                                   'OK' if good else 'MISMATCH'))
    ok = ok and good

    # 8. legacy ledger (pre-T-39, no loop_us): F6 SKIPPED with a note, not failed
    p = _fixture(os.path.join(base, 'gplr53-a1-selftest-legacy.jsonl'), mmap_meta, [
        {'wall': 10000, 'comp': 8000, 'wait': 50, 'embed': 1800, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
        {'wall': 8222, 'comp': 8000, 'wait': 50, 'embed': 20, 'preload': 2,
         'output': 10, 'slot': 0, 'loop_resid': 0.2},
    ], legacy=True)
    run(p, 'EVAL', [('SKIP', 'CHK-led'), ('PASS', 'CHK-embed')])

    for t in ('mmap', 'mmap-broken', 'pinned', 'pinned-broken', 'ledger-broken',
              'noledger', 'confounded', 'legacy'):
        q = os.path.join(base, 'gplr53-a1-selftest-%s.jsonl' % t)
        if os.path.exists(q):
            os.remove(q)
    print('SELFTEST:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if '--selftest' in argv:
        return selftest()
    if '--placement' in argv:
        return placement(root)
    as_json = '--json' in argv
    tags = [a for a in argv if not a.startswith('--')]
    if not tags:
        print(__doc__)
        return 2
    base = os.path.join(root, 'build', 'llama-cuda')
    infos, rc_fail, rc_uneval = [], False, False
    payload = []
    for tag in tags:
        path = tag if os.path.isabs(tag) else os.path.abspath(
            os.path.join(base, tag if tag.endswith('.jsonl') else tag + '.jsonl'))
        if not os.path.exists(path):
            print('%s: MISSING (cannot evaluate)' % tag)
            rc_uneval = True
            continue
        info, reason = rows_for(path)
        if info is None:
            print('%s: REFUSED - %s' % (tag, reason))
            rc_uneval = True
            continue
        verdict, findings = check_trace(info)
        if not as_json:
            print_trace(info, findings, verdict)
        if any(f[0] == 'FAIL' for f in findings):
            rc_fail = True
        infos.append(info)
        payload.append({'meta': info['meta'], 'rows': info['rows'],
                        'verdict': verdict, 'findings': findings})
    if infos and not as_json:
        print_contrast(infos)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not infos:
        print('RESULT: cannot evaluate - no trace produced a table (exit 2)')
        return 2
    if rc_fail:
        print('RESULT: FAIL - at least one checked expectation failed (exit 1)')
        return 1
    if rc_uneval:
        print('RESULT: PASS on the traces evaluated, but at least one trace was '
              'unevaluable (exit 2)')
        return 2
    print('RESULT: PASS - every evaluated trace satisfies its expectations')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
