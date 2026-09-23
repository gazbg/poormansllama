#!/usr/bin/env python3
"""T-GPLR-54 A2: validate the default-off `tg_logits` readout (host-only).

WHY THIS EXISTS
  A2 adds one runner flag, `--tg-logit-margin N` (<= 0 = off, the default), and
  one trace event: for the first N greedy TG steps the runner emits
  `tg_logits {step, top1_id, top1_logit, top2_id, top2_logit, margin, t_us}` —
  the SAME penalized top-1/top-2 values the pick consumed (T-GPLR-06's stderr
  line, now in the trace). The exe's two validators accept the event by
  forward compatibility; this checker judges the EVENT SEMANTICS, and it is
  the gate A2's acceptance names.

WHAT IT CHECKS (per trace, all falsifiable)
  * CHK-gate   the default-off invariant, on traces: a run_start WITHOUT a
               positive `tg_logit_margin` must carry NO tg_logits rows (the
               T-GPLR-49 lesson: a default run's trace is byte-identical);
               a run_start WITH `tg_logit_margin: N` carries exactly
               min(N, n_generated) rows.
  * CHK-ids    the rows are steps 0..k-1 with no gap and no duplicate, and
               every row's top1_id EQUALS that step's greedy id from
               `tg_tokens` — the readout annotates the pick that was made.
  * CHK-shape  required fields present; margin > 0 (top-1 IS the argmax);
               margin inside the recorded sanity band [1e-4, 60.0] (the
               measured stderr margins span 0.29..19.28); and the printed
               margin equals top1_logit - top2_logit within 1e-3.

PAIRING RULE (never guessed): rows pair with tg_tokens by step index; a trace
without `tg_tokens` is "cannot evaluate" (exit 2), NOT a pass.

USAGE
  python tools/check-gplr54-logits.py TAG1 TAG2 ...  # build/llama-cuda/<TAG>.jsonl
  python tools/check-gplr54-logits.py PATH           # an existing path is used as-is
  python tools/check-gplr54-logits.py --selftest     # inline fixtures + controls
  python tools/check-gplr54-logits.py TAG --json     # rows as JSON

EXIT CODES
  0 = PASS     every evaluated trace satisfied every check
  1 = FAIL     at least one named check failed (the check name is printed)
  2 = cannot evaluate: no arguments, a missing trace, no run_start, or no
      `tg_tokens` to pair against (a run that produced no decode is NOT a pass)
"""
import io
import json
import os
import sys

BUILD_DIR = os.path.join('build', 'llama-cuda')
SELFTEST_DIR = os.path.join(BUILD_DIR, 'gplr54-a2-selftest')

MARGIN_MIN = 1e-4    # recorded stderr margins span 0.29..19.28; a positive
MARGIN_MAX = 60.0    # argmax gap is never below ~1e-4 or above ~60 in practice
MARGIN_EPS = 1e-3    # printed with %.6f; 1e-3 absorbs any float formatting


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


def evaluate(events):
    """Return (checks, reason). reason set => cannot evaluate (exit 2)."""
    checks = []
    run_start = next((e for e in events if e.get('event') == 'run_start'), None)
    if run_start is None:
        return None, 'no run_start (a run that died before resolution is not a pass)'
    tg = next((e for e in events if e.get('event') == 'tg_tokens'), None)
    if tg is None:
        return None, 'no tg_tokens in trace: the readout pairs with the greedy ids it annotates (exit 2, NOT a pass)'
    ids = tg.get('ids') or []
    rows = [e for e in events if e.get('event') == 'tg_logits']
    declared = run_start.get('tg_logit_margin')

    # CHK-gate: default-off invariant + count = min(N, n_generated)
    if declared is None or declared <= 0:
        checks.append(('CHK-gate', not rows,
                       'run_start declares NO logit readout (tg_logit_margin %s) but the '
                       'trace carries %d tg_logits row(s) - a default run must emit none'
                       % (declared, len(rows))))
        if not rows:
            checks.append(('CHK-ids', True, 'no rows to pair (readout off) - default-off invariant holds'))
            checks.append(('CHK-shape', True, 'no rows to shape-check'))
            return checks, None
        # rows present but undeclared: gate failed; ids/shape still evaluable
        want = len(rows)
    else:
        want = min(int(declared), len(ids))
        checks.append(('CHK-gate', len(rows) == want,
                       'run_start tg_logit_margin=%s, tg_tokens carries %d id(s): expected %d row(s), found %d'
                       % (declared, len(ids), want, len(rows))))

    # CHK-ids: steps 0..k-1 contiguous, top1 == the greedy pick
    steps = [r.get('step') for r in rows]
    checks.append(('CHK-ids', steps == list(range(len(rows))),
                   'steps are %s, expected 0..%d' % (steps[:8], len(rows) - 1)))
    if steps == list(range(len(rows))):
        bad = [(r['step'], r.get('top1_id'), ids[r['step']])
               for r in rows
               if not isinstance(r.get('top1_id'), int) or r['step'] >= len(ids)
               or r['top1_id'] != ids[r['step']]]
        checks.append(('CHK-ids', not bad,
                       'top1_id == greedy pick per step%s'
                       % ('' if not bad else '; first offender: step %s top1 %s vs pick %s'
                          % bad[0])))

    # CHK-shape: fields, positive margin, sanity band, printed difference
    shape_bad = []
    for i, r in enumerate(rows):
        need = ('step', 'top1_id', 'top1_logit', 'top2_id', 'top2_logit', 'margin')
        if any(r.get(k) is None for k in need):
            shape_bad.append('row %d: missing field(s)' % i)
            continue
        m = r['margin']
        diff = r['top1_logit'] - r['top2_logit']
        if not (MARGIN_MIN <= m <= MARGIN_MAX):
            shape_bad.append('row %d: margin %.6f outside [%g, %g]' % (i, m, MARGIN_MIN, MARGIN_MAX))
        if abs(m - diff) > MARGIN_EPS:
            shape_bad.append('row %d: margin %.6f != top1-top2 %.6f' % (i, m, diff))
        if not isinstance(r.get('top1_id'), int) or not isinstance(r.get('top2_id'), int):
            shape_bad.append('row %d: ids must be integers' % i)
    checks.append(('CHK-shape', not shape_bad,
                   'field/positive/band/difference checks%s'
                   % ('' if not shape_bad else '; ' + '; '.join(shape_bad[:3]))))
    return checks, None


def resolve(arg):
    if os.path.isfile(arg):
        return arg
    p = os.path.join(BUILD_DIR, arg + '.jsonl')
    return p if os.path.isfile(p) else None


def run_one(path):
    events = load_events(path)
    checks, reason = evaluate(events)
    if reason is not None:
        print('REFUSED  %s' % path)
        print('         %s' % reason)
        return 2
    failed = [c for c in checks if not c[1]]
    print('%-8s %s' % ('FAIL' if failed else 'PASS', path))
    for name, good, detail in checks:
        print('         %-10s %s%s' % (name, 'ok' if good else 'FAILED',
                                       '' if good else ' - ' + detail))
    return 1 if failed else 0


# --- selftest fixtures (transient, under build/) ------------------------------

def base_fixture(margin_rows, declare=None, with_tg=True, with_run_start=True):
    """8-step margin shape on a 96-id stream; margins are the recorded
    T-GPLR-06 stderr values (bare-default-k.log steps 0..7)."""
    rec = [(0.438701, 248046, 48), (16.412857, 198, 271), (16.118988, 248045, 248068),
           (0.970795, 248068, 74455), (0.297215, 271, 198), (19.276882, 248069, 510),
           (11.994511, 271, 198), (6.535427, 48, 45)]
    ids = [m[1] for m in rec] + [900000 + i for i in range(88)]
    lines = []
    if with_run_start:
        rs = {'event': 'run_start', 'run_id': 7, 'residency': 'streaming'}
        if declare is not None:
            rs['tg_logit_margin'] = declare
        lines.append(rs)
    for t, (m, top1, top2) in enumerate(margin_rows):
        top1_logit = 12.0
        lines.append({'event': 'tg_logits', 'run_id': 7, 'step': t,
                      'top1_id': top1, 'top1_logit': top1_logit,
                      'top2_id': top2, 'top2_logit': round(top1_logit - m, 6),
                      'margin': m, 't_us': 42100000 + 130000 * t})
    if with_tg:
        lines.append({'event': 'tg_tokens', 'run_id': 7,
                      'n_tokens_generated': len(ids), 'ids': ids})
    return lines


def selftest():
    rec = [(0.438701, 248046, 48), (16.412857, 198, 271),
           (16.118988, 248045, 248068), (0.970795, 248068, 74455),
           (0.297215, 271, 198), (19.276882, 248069, 510),
           (11.994511, 271, 198), (6.535427, 48, 45)]
    clean = base_fixture(rec, declare=8)
    rows = lambda lines: [e for e in lines if e.get('event') == 'tg_logits']
    off_default = base_fixture([], declare=None)          # no key, no rows
    declared_zero = base_fixture([], declare=0)           # explicit 0, no rows
    bad_count = base_fixture(rec[:-1], declare=8)         # declared 8, 7 rows
    id_mismatch = [dict(e) for e in clean]
    rows(id_mismatch)[2]['top1_id'] = 42                  # step 2 picks a foreign id
    step_gap = [dict(e) for e in clean]
    for r in rows(step_gap)[4:]:
        r['step'] += 1                                    # steps 0,1,2,4,5,6,7,8
    torn_margin = [dict(e) for e in clean]
    rows(torn_margin)[5]['margin'] = 3.0                  # != top1-top2
    event_when_off = base_fixture(
        [(0.438701, 248046, 48)], declare=None)           # rows without the key
    off_band = [dict(e) for e in clean]
    rows(off_band)[4]['margin'] = 90.0                    # outside the sanity band

    cases = [
        ('clean', 0, clean),
        ('off-default', 0, off_default),
        ('declared-zero', 0, declared_zero),
        ('bad-count', 1, bad_count),
        ('id-mismatch', 1, id_mismatch),
        ('step-gap', 1, step_gap),
        ('torn-margin', 1, torn_margin),
        ('event-when-off', 1, event_when_off),
        ('margin-band', 1, off_band),
        ('no-run-start', 2, [e for e in clean if e['event'] != 'run_start']),
        ('no-tg-tokens', 2, [e for e in clean if e['event'] != 'tg_tokens']),
    ]
    os.makedirs(SELFTEST_DIR, exist_ok=True)
    bad = 0
    for name, expect, lines in cases:
        path = os.path.join(SELFTEST_DIR, name + '.jsonl')
        with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
            for e in lines:
                f.write(json.dumps(e) + '\n')
        got = run_one(path)
        print('[selftest] %-16s expect exit %d, got %d %s'
              % (name, expect, got, 'ok' if got == expect else 'BAD'))
        if got != expect:
            bad += 1
    print('[selftest] %d/%d as expected' % (len(cases) - bad, len(cases)))
    return 1 if bad else 0


def main():
    if '--selftest' in sys.argv:
        return selftest()
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__)
        return 2
    worst = 0
    for arg in args:
        path = resolve(arg)
        if path is None:
            print('REFUSED  %s' % arg)
            print('         cannot evaluate: trace not found (looked at %s and %s)'
                  % (arg, os.path.join(BUILD_DIR, arg + '.jsonl')))
            worst = max(worst, 2)
            continue
        worst = max(worst, run_one(path))
    return worst


if __name__ == '__main__':
    sys.exit(main())
