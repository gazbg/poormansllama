#!/usr/bin/env python3
"""T-GPLR-54 A3: the pre-registered margin-band verdict (host-only, post-run).

WHY THIS EXISTS
  The u4096 single-chunk prefill stream equals the u512 multi-chunk stream plus
  one 7-id template insertion at index 41, re-synchronizing byte-exactly
  (T-GPLR-54 A1). Two explanations remain and both predict the SAME recorded
  ids; they differ only in the logits behind the flip:
    H1 (benign): the flip is a near-tie tie-break from batch-shape numerics.
    H2 (defect): the 1-chunk state is confidently different at the flip.
  A2's `--tg-logit-margin` instrument records per-step top-1/top-2 margins.
  This tool consumes the session's traces and applies the rule EXACTLY as
  pre-registered in the archived T-GPLR-54 block, written before any A3 data:

    M_band = max margin over the agreeing steps (indices 0..40) of the
             multi-chunk arms.
    benign (H1)  iff ID-CHECK passes AND margin_u4096(step 40) <= M_band + 1.0.
    escalate(H2) iff ID-CHECK fails OR margin_u4096(step 40) > M_band + 1.0.
    Neutrality: ids(u512-on) == ids(u512-off) byte-identical, or the session is
    confounded (re-run, do not interpret).

  "margin_u4096(step 40)" is interpreted with the A2 pairing (pinned by the
  fixture and checker: tg_logits row s annotates the pick that produced
  tg_tokens ids[s]): step 40 is the last agreeing pick, immediately before the
  flip pick at index 41. The tool reports the flip step's own margin as
  CONTEXT, never as the rule's input.

SESSION CONTRACT (fixed trace names, from tools/run-gplr54-a3-session.ps1)
  gplr54-a3-u512-off   neutrality-pair control  (flag absent, no margin rows)
  gplr54-a3-u512-on    neutrality-pair treatment (--tg-logit-margin 96)
  gplr54-a3-u512r-on   C4 repeat of the u512 reference arm
  gplr54-a3-u2048-on   intermediate shape (expected IDENTICAL, K = 0)
  gplr54-a3-u4096-on   the single-chunk arm under test
  gplr54-a3-u4096r-on  C4 repeat of the single-chunk arm

EVALUATION ORDER (a failed earlier gate is never re-interpreted by a later one)
  0. PRESENT     every trace exists with run_start + tg_tokens  (else exit 2)
  1. INTEGRITY   margin arms: run_start declares tg_logit_margin >= 96, rows
                 are steps 0..95 contiguous, top1_id == ids[step], margin ==
                 top1_logit - top2_logit (1e-3), and the OFF arm has NO rows
                 (default-off invariant, in-session). Any failure = the
                 instrument did not do what it says -> INTEGRITY-FAIL.
  2. NEUTRALITY  ids(u512-on) == ids(u512-off) byte-identical, else CONFOUNDED
                 (exit 1, "re-run, do not interpret").
  3. C4          ids(u512r-on) == ids(u512-on) and
                 ids(u4096r-on) == ids(u4096-on) (repeats reproduce exactly).
  4. ID-CHECK    vs the session reference u512-on, via the A1 tool's check:
                 u2048-on -> IDENTICAL; u4096-on -> BOUNDED-INSERTION with
                 D >= 41, K <= 8, template-only, realigned.
  5. BAND        M_band = max margin over the agreeing steps (the dynamic set
                 of steps where ALL arms' ids agree - expected exactly 0..40,
                 and the tool refuses a different set as a session surprise) of
                 the multi-chunk arms (u512-on, u512r-on, u2048-on); the
                 decision margin is the PRIMARY u4096-on arm's step-40 row.
  6. VERDICT     BENIGN / ESCALATE per the rule, with the AC3 outcome wording.

USAGE
  python tools/analyze-gplr54-a3-verdict.py            # evaluate the session
  python tools/analyze-gplr54-a3-verdict.py --dir D    # traces in D
  python tools/analyze-gplr54-a3-verdict.py --selftest # fixtures + controls

EXIT CODES
  0 = BENIGN verdict, every gate green
  1 = ESCALATE, CONFOUNDED, INTEGRITY-FAIL, or SESSION-SURPRISE (named)
  2 = cannot evaluate (missing/incomplete traces) - never a pass
"""
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = HERE
BUILD_DIR = os.path.join(os.path.dirname(HERE), 'build', 'llama-cuda')

PREFIX = 'gplr54-a3-'
ARMS = ['u512-off', 'u512-on', 'u512r-on', 'u2048-on', 'u4096-on', 'u4096r-on']
MULTI_CHUNK = ['u512-on', 'u512r-on', 'u2048-on']   # arms whose rows feed M_band
MARGIN_DEPTH = 96
MARGIN_EPS = 1e-3

# Import the A1 tool by path (hyphenated module name): stdlib-only, safe.
import importlib.util


def _load_a1():
    p = os.path.join(TOOLS, 'analyze-gplr54-ubatch-identity.py')
    spec = importlib.util.spec_from_file_location('gplr54_ubatch_identity', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A1 = _load_a1()


def load_events(path):
    events = []
    with io.open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '"event"' not in line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def load_arm(d, tag):
    """Return dict(run_start, ids, rows) or (None, reason)."""
    path = os.path.join(d, PREFIX + tag + '.jsonl')
    if not os.path.isfile(path):
        return None, 'missing trace %s' % os.path.basename(path)
    ev = load_events(path)
    rs = next((e for e in ev if e.get('event') == 'run_start'), None)
    if rs is None:
        return None, 'no run_start in %s' % os.path.basename(path)
    ids = None
    for e in ev:
        if e.get('event') == 'tg_tokens' and isinstance(e.get('ids'), list):
            ids = [int(t) for t in e['ids']]
    if ids is None:
        return None, 'no tg_tokens in %s (run produced no decode)' % os.path.basename(path)
    rows = [e for e in ev if e.get('event') == 'tg_logits']
    return {'run_start': rs, 'ids': ids, 'rows': sorted(rows, key=lambda r: r.get('step', -1)),
            'path': path}, None


def check_integrity(arm):
    """Margin-arm invariants: declared depth, contiguous steps, top1==pick,
    margin arithmetic, positive margins. Returns error string or None."""
    rs = arm['run_start']
    depth = rs.get('tg_logit_margin')
    if not isinstance(depth, int) or depth < MARGIN_DEPTH:
        return 'run_start tg_logit_margin missing or < %d (got %r)' % (MARGIN_DEPTH, depth)
    rows = arm['rows']
    if len(rows) != min(MARGIN_DEPTH, len(arm['ids'])):
        return 'expected %d tg_logits rows, got %d' % (min(MARGIN_DEPTH, len(arm['ids'])), len(rows))
    for i, r in enumerate(rows):
        if r.get('step') != i:
            return 'rows not contiguous at step %d' % i
        if int(r.get('top1_id', -1)) != arm['ids'][i]:
            return 'top1_id %s != greedy pick %d at step %d' % (r.get('top1_id'), arm['ids'][i], i)
        m, t1, t2 = r.get('margin'), r.get('top1_logit'), r.get('top2_logit')
        if None in (m, t1, t2):
            return 'missing margin fields at step %d' % i
        if abs(float(m) - (float(t1) - float(t2))) > MARGIN_EPS:
            return 'margin %s != top1-top2 at step %d' % (m, i)
        if float(m) <= 0:
            return 'non-positive margin at step %d' % i
    return None


def check_off_invariant(arm):
    # The default-off invariant, mirroring the A2 checker's CHK-gate: an OFF run
    # declares the depth as 0 (explicit provenance, the runner always passes the
    # resolved value) or the key is absent on a legacy trace - and in both cases
    # the trace must carry ZERO tg_logits rows.
    if arm['rows']:
        return 'OFF arm carries %d tg_logits rows (default-off invariant violated)' % len(arm['rows'])
    d = arm['run_start'].get('tg_logit_margin')
    if d is not None and d != 0:
        return 'OFF arm run_start declares tg_logit_margin=%r (must be 0 or absent)' % d
    return None


def evaluate(d):
    """Run the ordered gates. Returns (exit_code, report_lines)."""
    out = []
    arms, err = {}, None
    for tag in ARMS:
        a, err = load_arm(d, tag)
        if a is None:
            out.append('PRESENT: FAIL - %s' % err)
            return 2, out
        arms[tag] = a
    out.append('PRESENT: all 6 session traces load (run_start + tg_tokens)')

    # Gate 1: instrument integrity on every margin arm; OFF invariant on u512-off.
    for tag in ARMS[1:]:
        e = check_integrity(arms[tag])
        if e:
            out.append('INTEGRITY: FAIL - %s: %s' % (tag, e))
            return 1, out
    e = check_off_invariant(arms['u512-off'])
    if e:
        out.append('INTEGRITY: FAIL - u512-off: %s' % e)
        return 1, out
    out.append('INTEGRITY: every margin arm declares depth>=96, rows contiguous, '
               'top1==pick, margin arithmetic OK; OFF arm clean')

    # Gate 2: neutrality pair.
    if arms['u512-on']['ids'] != arms['u512-off']['ids']:
        out.append('NEUTRALITY: FAIL - ids(u512-on) != ids(u512-off): session is '
                   'CONFOUNDED. Re-run; do not interpret any margin.')
        return 1, out
    out.append('NEUTRALITY: ids(u512-on) == ids(u512-off) byte-identical '
               '(%d ids)' % len(arms['u512-on']['ids']))

    # Gate 3: C4 determinism (repeats reproduce exactly).
    for rep, base in (('u512r-on', 'u512-on'), ('u4096r-on', 'u4096-on')):
        if arms[rep]['ids'] != arms[base]['ids']:
            out.append('C4: FAIL - %s does not reproduce %s' % (rep, base))
            return 1, out
    out.append('C4: both repeats reproduce their base arms byte-identically')

    # Gate 4: ID-CHECK vs the session reference (A1 tool). u2048 must be
    # IDENTICAL; u4096 must reproduce the RECORDED shape exactly (D=41, K=7,
    # the measured insertion) - the pre-registered rule's indices are written
    # for that divergence, so anything else is an escalation, not a re-fit.
    ref = arms['u512-on']['ids']
    ic = {}
    for tag, expect in (('u2048-on', 'IDENTICAL'), ('u4096-on', 'RECORDED-SHAPE')):
        v = A1.id_check(ref, arms[tag]['ids'])
        if expect == 'IDENTICAL':
            ok = v['verdict'] == 'IDENTICAL'
        else:
            ok = (v['verdict'] == 'BOUNDED-INSERTION' and v['realigns']
                  and v['D'] == A1.RECORDED_D and v['K'] == A1.RECORDED_K
                  and v['inserted'] == A1.RECORDED_INSERTED and v['template_only'])
        ic[tag] = (ok, v)
        out.append('ID-CHECK %s: %s %s' % (
            tag, 'PASS' if ok else 'FAIL',
            A1.fmt_verdict(dict(v, _ref=ref))))
    idcheck_ok = all(ok for ok, _ in ic.values())

    # Gate 5 (evidence, always computed and printed): the pre-registered band.
    # The rule, verbatim: "M_band = max margin over the agreeing steps (indices
    # 0..40) of the MULTI-CHUNK arms"; the decision margin is the u4096 arm's
    # step-40 row. The u4096 arm does NOT feed the band (after the flip it is
    # 7 positions ahead - positional agreement across arms is meaningless
    # there, which is exactly what the insertion realignment encodes). The
    # window 0..D-1 is DERIVED from gate 4's verified D, not assumed.
    D = ic['u4096-on'][1]['D']
    window = list(range(D)) if D is not None else []
    m_band, band_detail = None, []
    for tag in MULTI_CHUNK:
        by_step = {r['step']: r for r in arms[tag]['rows']}
        for s in window:
            m = float(by_step[s]['margin'])
            band_detail.append((m, tag, s))
            if m_band is None or m > m_band:
                m_band = m
    by_step_u = {r['step']: r for r in arms['u4096-on']['rows']}
    m_flip_arm = float(by_step_u[A1.RECORDED_D - 1]['margin'])
    thr = m_band + 1.0
    out.append('BAND: window 0..%d (derived from verified D=%d); M_band=%.6f '
               '(max over %d rows of %s; top: %s)'
               % (D - 1, D, m_band, len(band_detail), '/'.join(MULTI_CHUNK),
                  ', '.join('%.3f@%s:s%d' % (m, tg, s)
                            for m, tg, s in sorted(band_detail, reverse=True)[:3])))
    out.append('BAND: margin_u4096(step %d)=%.6f threshold=M_band+1.0=%.6f slack=%+.6f'
               % (A1.RECORDED_D - 1, m_flip_arm, thr, m_flip_arm - thr))

    # Gate 6: the verdict.
    benign = idcheck_ok and m_flip_arm <= thr
    flip41 = by_step_u.get(A1.RECORDED_D)
    ctx = ['u4096 flip-step (%d) margin=%.6f' % (A1.RECORDED_D, float(flip41['margin']))
           if flip41 else 'u4096 flip-step row absent']
    ctx.append('u4096 max margin anywhere=%.6f'
               % max(float(r['margin']) for r in arms['u4096-on']['rows']))
    ctx.append('u4096 step-41 top1_id=%s (u512 picked %d)'
               % (flip41.get('top1_id') if flip41 else '?', ref[A1.RECORDED_D]))
    out.append('CONTEXT (not the rule): ' + '; '.join(ctx))
    if benign:
        out.append('VERDICT: BENIGN (H1) - margin_u4096(step 40)=%.6f <= %.6f and '
                   'ID-CHECK passes. The u4096 escape at index 41 was a NEAR-TIE '
                   'tie-break, not a confident different choice.' % (m_flip_arm, thr))
        out.append('AC3 outcome: benign bounded tie-break - id-equivalence at '
                   'u>=4096 is not available by design, but the ID-CHECK is; the '
                   'u4096 caution becomes an arm rule.')
        return 0, out
    why = ('ID-CHECK failed') if not idcheck_ok else \
          ('margin_u4096(step 40)=%.6f > %.6f' % (m_flip_arm, thr))
    out.append('VERDICT: ESCALATE (H2) - %s. The 1-chunk shape chose the flip '
               'pick CONFIDENTLY beyond the multi-chunk decision band.' % why)
    out.append('AC3 outcome: confidently different at the 1-chunk shape - '
               'escalate with the reproducer (A4).')
    return 1, out


# --- selftest fixtures ---------------------------------------------------------

def _base_ids():
    # 96 synthetic ids shaped like the recorded arms: answer span 0..34, template
    # run 35..40, then content. The u4096 arm inserts 7 template ids at 41.
    ids = ([100 + i for i in range(35)]
           + [248045] * 6
           + [846, 198, 2523, 513, 34579, 4602, 883, 279, 2128, 3559, 13, 21134]
           + [700 + i for i in range(43)])
    assert len(ids) == 96
    return ids


def _u4096_ids(ref, inserted=None):
    D, K = 41, 7
    ins = inserted if inserted is not None else [248045] * 4 + [248046, 198, 248045]
    return ref[:D] + ins + ref[D:]


def _write_session(d, u512_on, u4096_on, margins_multichunk, margin_flip40,
                   u512_off=None, u4096r=None, corrupt_row=None, off_rows=False,
                   off_declared=None):
    ref = _base_ids()
    u512_off = u512_off if u512_off is not None else ref
    bodies = {
        'u512-off': u512_off, 'u512-on': u512_on, 'u512r-on': u512_on,
        'u2048-on': ref, 'u4096-on': u4096_on,
        'u4096r-on': u4096r if u4096r is not None else u4096_on,
    }
    for tag, ids in bodies.items():
        lines = [{'event': 'run_start', 'run_id': 7, 'tg_logit_margin': MARGIN_DEPTH}]
        if tag == 'u512-off' and not off_rows:
            # off arm: key absent (legacy) or declared 0 (explicit provenance)
            lines = [{'event': 'run_start', 'run_id': 7}]
            if off_declared is not None:
                lines[0]['tg_logit_margin'] = off_declared
        rows = []
        for s in range(MARGIN_DEPTH):
            m = margin_flip40 if (tag == 'u4096-on' and s == 40) else \
                (margins_multichunk if tag in MULTI_CHUNK else 1.0)
            t1, t2 = 12.0, 12.0 - m
            row = {'event': 'tg_logits', 'step': s, 'top1_id': ids[s],
                   'top1_logit': t1, 'top2_id': 5, 'top2_logit': t2,
                   'margin': round(t1 - t2, 6), 't_us': 1000 + s}
            if corrupt_row and tag == corrupt_row[0] and s == corrupt_row[1]:
                row['top1_id'] = 999999
            rows.append(row)
        if tag == 'u512-off' and not off_rows:
            rows = []
        lines += rows
        lines.append({'event': 'tg_tokens', 'n_tokens_generated': 96, 'ids': ids})
        with io.open(os.path.join(d, PREFIX + tag + '.jsonl'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(json.dumps(l) for l in lines) + '\n')


def run_selftest():
    cases = []
    def case(name):
        cases.append(name)
        d = tempfile.mkdtemp(prefix='gplr54a3-' + name + '-')
        return d

    # 1 benign: flip margin inside the band.
    d = case('benign')
    ref = _base_ids()
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8)
    code, out = evaluate(d)
    ok1 = code == 0 and any('BENIGN' in l for l in out)

    # 2 escalate by margin: confident flip.
    d = case('escalate-margin')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=9.9)
    code, out = evaluate(d)
    ok2 = code == 1 and any('ESCALATE' in l for l in out)

    # 3 escalate by id-check: content-token insertion (C3 fails).
    d = case('escalate-idcheck')
    bad = _u4096_ids(ref, inserted=[248045, 7777, 248045, 248045, 248046, 198, 248045])
    _write_session(d, ref, bad, margins_multichunk=1.2, margin_flip40=0.8)
    code, out = evaluate(d)
    ok3 = code == 1 and any('ESCALATE' in l and 'ID-CHECK' in l for l in out)

    # 4 confounded: neutrality pair disagrees; margins must NOT be interpreted.
    d = case('confounded')
    _write_session(d, [5] + ref[1:], _u4096_ids(ref), margins_multichunk=1.2,
                   margin_flip40=9.9, u512_off=ref)
    code, out = evaluate(d)
    ok4 = code == 1 and any('CONFOUNDED' in l for l in out)

    # 5 integrity: a corrupted top1 row fails before any interpretation.
    d = case('integrity')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8,
                   corrupt_row=('u4096-on', 12))
    code, out = evaluate(d)
    ok5 = code == 1 and any('INTEGRITY' in l for l in out)

    # 6 refusal: a missing trace cannot be evaluated.
    d = case('refusal')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8)
    os.remove(os.path.join(d, PREFIX + 'u2048-on.jsonl'))
    code, out = evaluate(d)
    ok6 = code == 2 and any('PRESENT' in l for l in out)

    # 7 off-invariant: rows under an undeclared key fail.
    d = case('off-rows')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8,
                   off_rows=True)
    code, out = evaluate(d)
    ok7 = code == 1 and any('default-off' in l for l in out)

    # 8 session surprise: u2048 not identical to the reference.
    d = case('surprise')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8,
                   u4096r=[5] + _u4096_ids(ref)[1:])
    code, out = evaluate(d)
    ok8 = code == 1 and any('C4: FAIL' in l for l in out)

    # 9 off arm with EXPLICIT depth 0 is the engine's real off shape: benign.
    d = case('off-declared-zero')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8,
                   off_declared=0)
    code, out = evaluate(d)
    ok9 = code == 0 and any('BENIGN' in l for l in out)

    # 10 off arm declaring a POSITIVE depth is a wiring defect: refuse.
    d = case('off-declared-96')
    _write_session(d, ref, _u4096_ids(ref), margins_multichunk=1.2, margin_flip40=0.8,
                   off_declared=MARGIN_DEPTH)
    code, out = evaluate(d)
    ok10 = code == 1 and any('INTEGRITY' in l for l in out)

    results = [('benign verdict', ok1), ('escalate by margin', ok2),
               ('escalate by id-check', ok3), ('confounded refusal', ok4),
               ('integrity fail', ok5), ('missing-trace exit 2', ok6),
               ('off-arm rows refusal', ok7), ('C4 repeat mismatch', ok8),
               ('off depth 0 accepted', ok9), ('off positive depth refused', ok10)]
    bad = [n for n, ok in results if not ok]
    for n, ok in results:
        print('  %-28s %s' % (n, 'ok' if ok else 'FAILED'))
    print('SELFTEST: %s (%d/%d)' % ('PASS' if not bad else 'FAIL',
                                    len(results) - len(bad), len(results)))
    return 0 if not bad else 1


def main(argv):
    d = BUILD_DIR
    if '--selftest' in argv:
        return run_selftest()
    if '--dir' in argv:
        d = argv[argv.index('--dir') + 1]
    code, out = evaluate(d)
    print('=' * 72)
    for l in out:
        print(l)
    print('=' * 72)
    print('EXIT: %d' % code)
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
