"""T-GPLR-52 arm A2 analysis: pass-index vs median-copy table + H2 verdict.

Host-only. Reads layer-streaming traces, splits phase-0 (PP) copy_complete
events into per-pass segments, and emits:

  - per pass: copy p50/min/max, effective staging rate (bytes_transferred /
    total_transfer_us from the matching pp phase_summary), wall, tok/s,
    evictions
  - the H2 verdict per the pre-registered bands (identical to the arm
    runner's in-trace evaluation; this tool re-derives it from the trace)

Pass segmentation rules (measured on the baseline traces):
  * With --pp-repeat N there are N phase_summary events with phase == "pp",
    in time order, and N * 40 phase-0 copy_complete events (one whole-bank
    copy per layer per pass; T-52 F1).
  * Copies are sorted by t_us and dealt 40 to each pass index. The count
    must be exactly N * 40 or the tool refuses (loud, structured).

Usage:
  python tools/analyze-gplr52-a2.py TAG1 TAG2 ...      # analyze build/llama-cuda/<TAG>.jsonl
  python tools/analyze-gplr52-a2.py --selftest         # synthetic-trace self-test

The verdict bands (pre-registered in the T-GPLR-52 block BEFORE the arms ran):
  200k: pass-3 ~ pass-2 (|p3-p2|/p2 <= 10%)  -> CONFIRMED (evicted regime)
        pass-3 <= 0.80 * pass-2              -> FALSIFIED (banks survive)
  50k : pass-3 <= 0.85 * pass-2              -> CONFIRMED (survivor regime)
        pass-3 ~ pass-2                      -> survivor prediction FALSIFIED
"""
import io
import json
import sys
import os

PASS_LAYERS = 40  # Qwen3.6-35B-A3B decoder layers; copy_count assertions key on this


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


def median(v):
    v = sorted(v)
    n = len(v)
    if n == 0:
        return float('nan')
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def analyze_tag(path):
    events = load_events(path)
    run_start = next((e for e in events if e.get('event') == 'run_start'), None)
    pp_summaries = [e for e in events if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    copies = sorted(
        (e for e in events if e.get('event') == 'copy_complete' and e.get('phase') == 0),
        key=lambda e: e.get('t_us', 0))

    if run_start is None:
        print('%s: NO run_start (trace dead before resolution) -> FAIL' % os.path.basename(path))
        return None
    n_pass = len(pp_summaries)
    want = n_pass * PASS_LAYERS
    if n_pass == 0 or len(copies) != want:
        print('%s: copy count %d vs expected %d (passes=%d) -> refuse (loud)' %
              (os.path.basename(path), len(copies), want, n_pass))
        return None

    tag = os.path.basename(path).rsplit('.', 1)[0]
    print('== %s ==' % tag)
    print('  copy_source=%s (%s)  prebuild_per_pass=%s (%s)  kv=%s  slots=%s  repeat=%d' % (
        run_start.get('copy_source'), run_start.get('copy_source_source'),
        run_start.get('pp_prebuild_per_pass'), run_start.get('pp_prebuild_per_pass_source'),
        run_start.get('kv_cache'), run_start.get('n_slots'), n_pass))

    per = []
    for i in range(n_pass):
        seg = copies[i * PASS_LAYERS:(i + 1) * PASS_LAYERS]
        ms = [c.get('transfer_us', 0) / 1000.0 for c in seg]
        sm = pp_summaries[i]
        total_ms = sm.get('total_transfer_us', 0) / 1000.0
        bytes_ = sm.get('bytes_transferred', 0)
        rate = (bytes_ / (total_ms / 1000.0) / (1024 * 1024)) if total_ms > 0 else float('nan')
        per.append(median(ms))
        print('  pass %d: p50=%7.1f ms  min=%7.1f  max=%7.1f | wall=%8.2f s  tok/s=%7.1f  '
              'eff_rate=%5.0f MB/s (bucket)  evictions=%d' % (
                  i + 1, median(ms), min(ms), max(ms),
                  sm.get('wall_us', 0) / 1e6, sm.get('tokens_per_s', 0.0), rate,
                  sm.get('evictions', -1)))

    if n_pass < 3:
        print('  (fewer than 3 passes; no H2 band applies)')
        return {'passes': per, 'h2_verdict': 'N/A (%d passes)' % n_pass}

    rel23 = abs(per[2] - per[1]) / per[1] if per[1] else float('nan')
    if '200k' in tag:
        if per[2] <= 0.80 * per[1]:
            verdict = 'FALSIFIED: banks survive at 200k'
        elif rel23 <= 0.10:
            verdict = 'CONFIRMED: evicted regime, pass-3 ~ pass-2'
        else:
            verdict = 'ANOMALOUS: pass-3 moved 10-20% without a prediction'
    else:
        if per[2] <= 0.85 * per[1]:
            verdict = 'CONFIRMED: survivor regime, pass-3 drops'
        elif rel23 <= 0.10:
            verdict = 'survivor prediction FALSIFIED: no surviving regime at this shape'
        else:
            verdict = 'ANOMALOUS: pass-3 rose >10%'
    print('  pass1->2 drop %+.1f%%, pass2->3 drop %+.1f%% | H2: %s' % (
        100.0 * (per[0] - per[1]) / per[0] if per[0] else float('nan'),
        100.0 * (per[1] - per[2]) / per[1] if per[1] else float('nan'),
        verdict))
    return {'passes': per, 'h2_verdict': verdict}


# ---------------------------------------------------------------- self-test
def selftest():
    # Synthetic trace: 3 pp passes, 40 layers each, with a pass-3 drop at 50k
    # and a flat pass-3 at 200k. Validates segmentation, medians, and bands.
    def write_synthetic(tag, medians, copy_source, shape_200k):
        path = os.path.join(os.path.dirname(__file__), '..', 'build',
                            'llama-cuda', 'a2-selftest-%s.jsonl' % tag)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t = 1000
        with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps({'event': 'run_start', 'copy_source': copy_source,
                                'copy_source_source': 'explicit',
                                'pp_prebuild_per_pass': 1,
                                'pp_prebuild_per_pass_source': 'default',
                                'kv_cache': 'f16', 'n_slots': 3}) + '\n')
            for p_i, med in enumerate(medians):
                for layer in range(PASS_LAYERS):
                    t += int(med * 1000)
                    f.write(json.dumps({'event': 'copy_complete', 'phase': 0, 'layer': layer,
                                        'slot': layer % 3, 'transfer_us': int(med * 1000),
                                        't_us': t}) + '\n')
                f.write(json.dumps({'event': 'phase_summary', 'phase': 'pp',
                                    'wall_us': 1000000, 'total_transfer_us': med * 1000 * PASS_LAYERS,
                                    'bytes_transferred': 18327011328, 'tokens_per_s': 1475.0,
                                    'evictions': 40}) + '\n')
        return path

    ok = True
    # 200k synthetic: flat 3000/3000/3010 -> CONFIRMED (evicted regime)
    p = write_synthetic('200k', [3000.0, 3000.0, 3010.0], 'mmap-registered', True)
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = analyze_tag(p)
    if r is None or 'CONFIRMED' not in r['h2_verdict']:
        print('SELFTEST FAIL (200k flat): %r' % (r and r['h2_verdict']))
        ok = False
    else:
        print('selftest 200k flat -> CONFIRMED: OK')
    # 200k synthetic: pass-3 collapses -> FALSIFIED (survivor)
    p = write_synthetic('200ksurv', [3000.0, 3000.0, 600.0], 'mmap-registered', True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = analyze_tag(p)
    if r is None or 'FALSIFIED' not in r['h2_verdict']:
        print('SELFTEST FAIL (200k survivor): %r' % (r and r['h2_verdict']))
        ok = False
    else:
        print('selftest 200k survivor -> FALSIFIED: OK')
    # 50k synthetic: pass-3 drops 20% -> CONFIRMED (survivor regime)
    p = write_synthetic('50k', [950.0, 800.0, 640.0], 'mmap-registered', False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = analyze_tag(p)
    if r is None or 'CONFIRMED' not in r['h2_verdict']:
        print('SELFTEST FAIL (50k survivor): %r' % (r and r['h2_verdict']))
        ok = False
    else:
        print('selftest 50k survivor -> CONFIRMED: OK')
    # 50k synthetic: flat -> FALSIFIED (no surviving regime)
    p = write_synthetic('50kflat', [950.0, 800.0, 795.0], 'mmap-registered', False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = analyze_tag(p)
    if r is None or 'FALSIFIED' not in r['h2_verdict']:
        print('SELFTEST FAIL (50k flat): %r' % (r and r['h2_verdict']))
        ok = False
    else:
        print('selftest 50k flat -> FALSIFIED: OK')
    # count refusal: drop one copy -> refuse, not crash
    p = write_synthetic('short', [100.0, 100.0, 100.0], 'mmap-registered', False)
    lines = io.open(p, encoding='utf-8').readlines()
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines[:-3])  # drop last copy + its phase summary
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = analyze_tag(p)
    if r is not None:
        print('SELFTEST FAIL (short trace must refuse): %r' % (r,))
        ok = False
    else:
        print('selftest short trace -> refused loudly: OK')
    for t in ('200k', '200ksurv', '50k', '50kflat', 'short'):
        q = os.path.join(os.path.dirname(__file__), '..', 'build', 'llama-cuda',
                         'a2-selftest-%s.jsonl' % t)
        if os.path.exists(q):
            os.remove(q)
    print('SELFTEST:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main(argv):
    if '--selftest' in argv:
        return selftest()
    if not argv:
        print(__doc__)
        return 2
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'llama-cuda')
    rc = 0
    for tag in argv:
        path = tag if os.path.isabs(tag) else os.path.abspath(
            os.path.join(base, tag if tag.endswith('.jsonl') else tag + '.jsonl'))
        if not os.path.exists(path):
            print('%s: MISSING' % tag)
            rc = 1
            continue
        r = analyze_tag(path)
        if r is None:
            rc = 1
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
