#!/usr/bin/env python3
"""T-GPLR-82 AC1, run as written: 512-token anchor, --experts-host 2 vs 0, >=512
generated tokens, graded by semantic QA and top-1 agreement.

WHY THIS SESSION EXISTS. T-GPLR-82's AC1 is the task's real gate:

    "Semantic QA aggregate PASS at 50 and 512 tokens on the 512 anchor,
     check-telemetry OK, negative control FAILing as designed. Additionally
     top-1 agreement >= 99% against the recorded control over >= 512 tokens,
     and every divergence explained as a near-tie (top-2 logit gap below the
     measured drift). Unexplained divergence fails."

It has NO recorded evidence: tools/run-tests.ps1 never passes --experts-host, so
the 2026-10-05 ladder that "passed for the host-leg fix" exercised the fix's
regression test, not this mechanism. Only ad-hoc launchers (run-gplr82-hostarm.ps1,
gplr61-ac2/ac3, run-gplr62-kdepth.py) ever turn the leg on.

WHAT THE ARMS ARE, AND WHY FOUR.

    A0   --experts-host 0                        control (flag present, leg off)
    A2   --experts-host 2 --experts-host-budget inf   split, host-first
    A2d  --experts-host 2                        split, DEFAULT budget
    A0b  --experts-host 0                        control repeat (drift bracket)

A2d exists because `experts_host_budget` DEFAULTS TO 0.0 (main.cpp:3311). If 0.0
means "no host work allowed", a bare `--experts-host 2` arm is INERT, and its
clean QA result would be read as a PASS for a mechanism that never ran -- the
worst possible outcome for this gate. A2d settles that by measurement, and turns
out to separate a second defect on its own (see the report).

A0b exists because AC1's near-tie rule needs a MEASURED drift floor. If the two
control arms disagree anywhere, the maximum top-2 gap at those steps is the
drift; if they agree exactly, the floor is 0.0 and every divergence is
unexplained, which is what AC1 says it is.

---- GRADING RULE THAT IS EASY TO GET WRONG (learned the hard way, 2026-10-10) ----

A non-finite logit is written by the engine as `nan` / `-inf`, which is NOT valid
JSON. A grader that does `except Exception: continue` therefore DROPS those steps
and reports agreement over only the surviving finite ones -- which is how a run
whose logits go NaN at step 6 can look like "5 divergent of 6" instead of
"506 non-finite plus 5 wrong of 512". This grader counts unparseable lines, maps
them back to their step, and treats a missing or non-finite readout as a
DIVERGENCE that can never be a near-tie. Unparseable lines are reported, never
swallowed.

This runner claims nothing of its own. It records and grades.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build', 'llama-cuda')
EXE = os.path.join(BUILD, 'bin', 'Release', 'llama-layer-streaming.exe')
MODEL = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
ANCHOR = os.path.join(BUILD, 'gplr07-qa-prompts', 'anchor-512.txt')
ANCHOR_DEPTH_COPY = os.path.join(BUILD, 'gplr62-kdepth-anchor-512.txt')
CHK = os.path.join(ROOT, 'tools', 'semantic-qa', 'check-answer.py')
SESSION = 'gplr82-ac1'
for _i, _a in enumerate(sys.argv):
    if _a == '--session' and _i + 1 < len(sys.argv):
        SESSION = sys.argv[_i + 1]
S = os.path.join(BUILD, SESSION)

TG_TOKENS = 512
MARGIN_STEPS = 512          # --tg-logit-margin: records steps [0, N)
ITEMS = 'N1,N2,N3,N4,K1'    # the item list this 512 anchor was calibrated with
RECORDED_CONTROL_TOK_S = 19.31   # AC7's tg82 control (round-to-round median)
GRADE_ONLY = '--grade-only' in sys.argv

COMMON = [
    '--model', MODEL, '--prompt-file', ANCHOR,
    '--ubatch', '512', '--tg-tokens', str(TG_TOKENS),
    '--tg-logit-margin', str(MARGIN_STEPS),
    '--pp-no-logits', '--pp-schedule', 'batch-major',
    '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
    '--allow-full-residency', '--no-handoff', '--residency-tg',
    '--residency-tg-retain', '2',
]
ARMS = [
    ('A0',  ['--experts-host', '0'], 'control: flag present, host leg off'),
    ('A2',  ['--experts-host', '2', '--experts-host-budget', 'inf'],
            'split, host-first (the T-GPLR-62 shape that realised 2.00 rows/layer)'),
    ('A2d', ['--experts-host', '2'],
            'split, DEFAULT budget 0.0 -- does the default engage the leg at all?'),
    ('A0b', ['--experts-host', '0'], 'control repeat: drift bracket'),
]

verdict = {
    'session': SESSION,
    'task': 'T-GPLR-82',
    'ac': 'AC1',
    'grade_only': GRADE_ONLY,
    'anchor': ANCHOR,
    'items': ITEMS,
    'tg_tokens': TG_TOKENS,
    'recorded_control_tok_s': RECORDED_CONTROL_TOK_S,
    'arms': {}, 'checks': {}, 'notes': [
        'AC1 as written: semantic QA aggregate PASS + check-telemetry OK + negative '
        'control FAILing as designed, top-1 agreement >= 99% over >= 512 tokens, and '
        'every divergence explained as a near-tie (top-2 gap below measured drift).',
        'A2d is present because experts_host_budget defaults to 0.0; a bare '
        '--experts-host 2 arm may be inert, and an inert clean run must not be read '
        'as a PASS for the mechanism.',
        'The shape matches the T-GPLR-62 reference session, so A0 is comparable to '
        'its recorded retain-2 no-split arm (which was id-identical to K0).',
        'Non-finite logits are written as bare nan / -inf (invalid JSON) and are '
        'graded as divergences, never dropped; unparseable lines are counted.',
        'The runner claims nothing: it grades the recorded traces.',
    ],
}
t0 = time.time()


def log(*a):
    print(*a, flush=True)


def run(cmd, timeout=1800):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors='replace', timeout=timeout)
        return p.returncode, time.time() - t, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t, str(out) + ' TIMEOUT'


def sha16(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
    except OSError:
        return None
    return h.hexdigest()[:16]


# ---- 0. preconditions -------------------------------------------------------
for p in (EXE, ANCHOR, CHK):
    if not os.path.exists(p):
        log('MISSING REQUIRED FILE:', p)
        sys.exit(2)
verdict['provenance'] = {
    'exe_sha256_16': sha16(EXE),
    'llama_dll_sha256_16': sha16(os.path.join(BUILD, 'bin', 'Release', 'llama.dll')),
    'anchor_sha256_16': sha16(ANCHOR),
}
same = None
if os.path.exists(ANCHOR_DEPTH_COPY):
    same = sha16(ANCHOR_DEPTH_COPY) == verdict['provenance']['anchor_sha256_16']
verdict['provenance']['anchor_matches_calibrated_copy'] = same
if same is False:
    verdict['notes'].append(
        'WARNING: the anchor differs from the byte-identical copy whose item list '
        '(N1,N2,N3,N4,K1) was validated by the T-GPLR-62 session.')
    log('WARNING: anchor differs from the calibrated copy -- item list unverified')
log('provenance: exe=%s dll=%s anchor=%s (matches calibrated copy: %s)'
    % (verdict['provenance']['exe_sha256_16'],
       verdict['provenance']['llama_dll_sha256_16'],
       verdict['provenance']['anchor_sha256_16'], same))

# ---- 1. the arms ------------------------------------------------------------
for name, flags, why in ARMS:
    tr, tx = '%s-arm-%s.jsonl' % (S, name), '%s-arm-%s.txt' % (S, name)
    lg = '%s-arm-%s.log' % (S, name)
    if GRADE_ONLY:
        rc = 0 if os.path.exists(tr) else None
        verdict['arms'][name] = {'rc': rc, 'wall_s': None, 'trace': tr,
                                 'tgtext': tx, 'log': lg, 'why': why, 'flags': flags,
                                 'cmd': '(not executed: --grade-only)'}
        log('ARM %-4s not executed (--grade-only); trace %s'
            % (name, 'present' if rc == 0 else 'MISSING'))
        continue
    cmd = [EXE] + COMMON + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name] = {'rc': rc, 'wall_s': round(dur, 1), 'trace': tr,
                             'tgtext': tx, 'log': lg, 'why': why, 'flags': flags,
                             'cmd': ' '.join(cmd)}
    log('ARM %-4s rc=%s wall=%.1fs  (%s)' % (name, rc, dur, why))
    if rc != 0:
        log('  ARM %s tail: %s' % (name, out[-800:].replace('\n', ' | ')))

# ---- 2. check-telemetry per arm --------------------------------------------
for name, a in verdict['arms'].items():
    if a['rc'] != 0:
        verdict['checks']['checktel_' + name] = 'skipped (arm failed)'
        continue
    rc, _, out = run([EXE, '--check-telemetry', a['trace']], 600)
    verdict['checks']['checktel_' + name] = rc
    tail = out.strip().splitlines()[-1] if out.strip() else ''
    log('check-telemetry %-4s rc=%d %s' % (name, rc, tail[-150:]))

# ---- 3. trace readers -------------------------------------------------------


def read_trace(name):
    """Per-arm: greedy top-1/top-2 readout, ids, ledger, host share, phases.

    A step whose readout is non-finite (nan / -inf, written as invalid JSON) is
    recorded as a step with NO finite readout, not omitted.
    """
    a = verdict['arms'][name]
    r = {'steps': {}, 'ids': None, 'lc': 0, 'host_sum': 0, 'total_sum': 0,
         'violations': 0, 'nonfinite_steps': [],
         'host_scan_records': 0, 'host_nonfinite_records': 0, 'host_finite_records': 0,
         'unparsed': 0, 'unparsed_tg_logits': 0,
         'phases': {}, 'run_start': None,
         'fetches_with_host_rows': 0, 'ready_at_consumption_missing': 0}
    if a['rc'] != 0:
        return r
    try:
        fh = open(a['trace'], encoding='utf-8', errors='replace')
    except OSError:
        return r
    with fh:
        for raw in fh:
            if '"event"' not in raw:
                continue
            try:
                e = json.loads(raw)
            except Exception:
                # NOT swallowed: count it, and if it is a per-step logit readout
                # map it back to its step so it counts as a non-finite step.
                r['unparsed'] += 1
                if '"tg_logits"' in raw:
                    r['unparsed_tg_logits'] += 1
                    m = re.search(r'"step":\s*(\d+)', raw)
                    if m:
                        s = int(m.group(1))
                        r['steps'][s] = (None, None, None)
                        r['nonfinite_steps'].append(s)
                continue
            ev = e.get('event')
            if ev == 'tg_logits':
                st = e.get('step')
                r['steps'][st] = (e.get('top1_id'), e.get('margin'), e.get('top2_id'))
                mm = e.get('margin')
                if mm is None or mm != mm:
                    r['nonfinite_steps'].append(st)
            elif ev == 'tg_tokens':
                r['ids'] = e.get('ids')
            elif ev == 'layer_compute' and 'experts_total' in e:
                r['lc'] += 1
                h = e.get('experts_host') or 0
                r['host_sum'] += h
                r['total_sum'] += e.get('experts_total') or 0
                if (e.get('experts_device') or 0) + h != (e.get('experts_total') or 0):
                    r['violations'] += 1
                if h > 0:
                    # scan_host_expert_outputs sets host_finite = bad.empty() ? 0 : 1
                    # (llama-layer-streaming.cpp:4626): 0 = every host per-row output
                    # was FINITE, 1 = the scan found a non-finite one (and the run
                    # then aborts via ls_shape_failure). Reading 1 as "finite" is
                    # backwards, so both counts are kept explicitly.
                    hf = e.get('host_finite', -1)
                    if hf >= 0:
                        r['host_scan_records'] += 1
                        if hf == 1:
                            r['host_nonfinite_records'] += 1   # guard FIRED, run must abort
                        else:
                            r['host_finite_records'] += 1      # all host outputs finite
            elif ev == 'phase_summary':
                r['phases'][e.get('phase')] = e
            elif ev == 'run_start':
                r['run_start'] = e
            elif ev == 'rowset_fetch':
                if (e.get('experts_host') or 0) > 0:
                    r['fetches_with_host_rows'] += 1
                if 'ready_at_consumption' not in e:
                    r['ready_at_consumption_missing'] += 1
    return r


traces = {n: read_trace(n) for n in verdict['arms']}


def agree(a, b):
    """Top-1 agreement over every step EITHER arm recorded.

    A step one arm did not record, or recorded without a finite readout, counts
    as a divergence: absence of a finite readout is not agreement.
    """
    sa, sb = traces[a]['steps'], traces[b]['steps']
    if not sa or not sb:
        return None, None, None
    shared = sorted(set(sa) | set(sb))
    div = []
    for s in shared:
        va, vb = sa.get(s), sb.get(s)
        if va is None or vb is None or va[0] is None or vb[0] is None:
            div.append(s)
        elif va[0] != vb[0]:
            div.append(s)
    return 1.0 - len(div) / len(shared), len(shared), div


def is_near_tie(a, b, s, floor_val):
    """AC1's rule: near-tie iff the control's top-2 gap is below the drift floor."""
    va, vb = traces[a]['steps'].get(s), traces[b]['steps'].get(s)
    if va is None or vb is None or va[0] is None or vb[0] is None:
        return False, None            # non-finite / missing is never a near-tie
    m = va[1]
    if m is None or m != m:
        return False, None
    if floor_val is None:
        return None, m
    return (m < floor_val), m


# ---- 4. the AC1 numbers -----------------------------------------------------
ctrl, treat = 'A0', 'A2'
agr_floor, floor_steps, floor_div = agree('A0', 'A0b')
if agr_floor is None:
    floor_note, floor_val = 'control repeat missing -- drift floor UNMEASURED', None
else:
    floor_val = 0.0
    margins = [abs(traces['A0']['steps'][s][1] or 0.0) for s in floor_div
               if traces['A0']['steps'].get(s) and traces['A0']['steps'][s][1] is not None]
    if margins:
        floor_val = max(margins)
    floor_note = ('controls agree exactly at every aligned step -> drift floor 0.0'
                  if not floor_div else
                  'controls diverge at %d step(s); drift floor = max top-2 gap = %.6f'
                  % (len(floor_div), floor_val))
verdict['checks']['drift_floor'] = {
    'control_repeat_top1_agreement': agr_floor, 'steps_compared': floor_steps,
    'divergent_steps': floor_div, 'drift_floor': floor_val, 'note': floor_note,
}
log('')
log('drift floor: %s' % floor_note)

agr, shared_n, div = agree(ctrl, treat)
detail, unexplained = [], 0
if div is not None:
    for s in div:
        tie, m = is_near_tie(ctrl, treat, s, floor_val)
        if tie is not True:
            unexplained += 1
        detail.append({'step': s, 'control_top1': traces[ctrl]['steps'].get(s, (None,))[0],
                       'treatment_top1': traces[treat]['steps'].get(s, (None,))[0],
                       'control_margin': m, 'near_tie': tie,
                       'treatment_nonfinite': s in traces[treat]['nonfinite_steps']})
verdict['checks']['top1_agreement'] = {
    'arm': treat, 'control': ctrl, 'agreement': agr, 'steps_compared': shared_n,
    'n_divergent': None if div is None else len(div),
    'first_divergence': None if not div else div[0],
}
verdict['checks']['divergences'] = detail[:64]
verdict['checks']['n_divergent'] = None if div is None else len(div)
verdict['checks']['n_unexplained'] = unexplained if div is not None else None

log('')
if agr is None:
    log('top-1 agreement: UNMEASURABLE (a trace is missing its readout)')
else:
    log('top-1 agreement %s vs %s: %.6f over %d steps, %d divergent, %d unexplained'
        % (treat, ctrl, agr, shared_n, len(div), unexplained))
    log('  %s non-finite steps: %d (first: %s)'
        % (treat, len(traces[treat]['nonfinite_steps']),
           traces[treat]['nonfinite_steps'][:1] or '-'))
    for d in detail[:6]:
        log('  step %-4d %s -> %s   control top-2 gap %s   near-tie=%s%s'
            % (d['step'], d['control_top1'], d['treatment_top1'],
               d['control_margin'], d['near_tie'],
               '  [treatment NON-FINITE]' if d['treatment_nonfinite'] else ''))

# pairwise vs the control, so a second defect cannot hide behind the first
pairwise = {}
for name in verdict['arms']:
    if name == ctrl:
        continue
    a2, n2, d2 = agree(ctrl, name)
    pairwise[name] = {'agreement': a2, 'steps_compared': n2,
                      'first_divergence': None if not d2 else d2[0],
                      'n_divergent': None if d2 is None else len(d2),
                      'nonfinite_steps': len(traces[name]['nonfinite_steps']),
                      'host_share': (traces[name]['host_sum'] / traces[name]['total_sum'])
                      if traces[name]['total_sum'] else None}
verdict['checks']['pairwise_vs_control'] = pairwise

# ---- 5. semantic QA, positive + negative control ---------------------------
qa = {}
for name, a in verdict['arms'].items():
    if a['rc'] != 0:
        qa[name] = 'skipped (arm failed)'
        continue
    rp, _, op = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                     '--items', ITEMS], 600)
    rn, _, on = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                     '--items', ITEMS, '--negative'], 600)
    qa[name] = {'positive_rc': rp, 'positive_pass': (rp == 0) and ('aggregate=PASS' in op),
                'positive_tail': op.strip().splitlines()[-1][-140:] if op.strip() else '',
                'negative_rc': rn,
                'negative_fails_as_designed': (rn != 0) and ('aggregate=FAIL' in on),
                'negative_tail': on.strip().splitlines()[-1][-140:] if on.strip() else ''}
    log('QA %-4s positive rc=%d pass=%s | negative rc=%d fails_as_designed=%s'
        % (name, rp, qa[name]['positive_pass'], rn,
           qa[name]['negative_fails_as_designed']))
verdict['checks']['qa'] = qa

# ---- 6. AC2 / AC3 / AC4 / AC7 side-evidence from the same runs -------------
side = {}
for name, a in verdict['arms'].items():
    t = traces[name]
    tg = t['phases'].get('tg') or {}
    side[name] = {
        'rc': a['rc'], 'wall_s': a['wall_s'],
        'experts_host_sum': t['host_sum'], 'experts_total_sum': t['total_sum'],
        'host_share': (t['host_sum'] / t['total_sum']) if t['total_sum'] else None,
        'ledger_violations_AC2': t['violations'],
        'host_scan_records': t['host_scan_records'],
        'host_nonfinite_records': t['host_nonfinite_records'],
        'host_finite_records': t['host_finite_records'],
        'nonfinite_logit_steps': len(t['nonfinite_steps']),
        'unparsed_lines': t['unparsed'], 'unparsed_tg_logits': t['unparsed_tg_logits'],
        'tg_tokens_per_s': tg.get('tokens_per_s'),
        'tg_wall_us': tg.get('wall_us'),
        'bytes_transferred': tg.get('bytes_transferred'),
        'logits_steps_recorded': len(t['steps']),
        'ids_len': len(t['ids']) if t['ids'] else 0,
        'fetches_with_host_rows': t['fetches_with_host_rows'],
        'ready_at_consumption_missing': t['ready_at_consumption_missing'],
    }
verdict['checks']['side_evidence'] = side
log('')
log('%-4s %-4s %7s %10s %8s %9s %6s %6s %6s  %s'
    % ('arm', 'rc', 'tok/s', 'host_shr', 'ledger', 'hostScan', 'nanst', 'ids', 'unpar', 'note'))
for name in verdict['arms']:
    s = side[name]
    log('%-4s %-4s %7s %10s %8d %8d %6d %6d %6d  %s' % (
        name, s['rc'],
        ('%.2f' % s['tg_tokens_per_s']) if s['tg_tokens_per_s'] else 'n/a',
        ('%.5f' % s['host_share']) if s['host_share'] is not None else 'n/a',
        s['ledger_violations_AC2'], s['host_scan_records'],
        s['nonfinite_logit_steps'], s['ids_len'], s['unparsed_lines'],
        verdict['arms'][name]['why'][:44]))
log('')
for name, pw in pairwise.items():
    log('  vs %-4s agreement=%s over %s steps, first divergence=%s, '
        'non-finite steps=%d, host share=%s'
        % (name, pw['agreement'], pw['steps_compared'], pw['first_divergence'],
           pw['nonfinite_steps'],
           ('%.5f' % pw['host_share']) if pw['host_share'] is not None else 'n/a'))

needles = []
for name, a in verdict['arms'].items():
    try:
        txt = open(a['log'], encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    for n in ('cudaErrorLaunchTimeout', 'TDR', 'out of memory', 'CUDA error'):
        if n in txt:
            needles.append([name, n])
verdict['checks']['ac3_needles'] = needles

# ---- 7. the AC1 verdict, stated as the contract states it ------------------
ac1_semantic = {n: (isinstance(qa[n], dict) and qa[n]['positive_pass']
                    and qa[n]['negative_fails_as_designed']) for n in qa}
checktel_ok = all(v == 0 for k, v in verdict['checks'].items()
                  if k.startswith('checktel_')) and \
    any(k.startswith('checktel_') for k in verdict['checks'])
treat_nonfinite = len(traces[treat]['nonfinite_steps'])
ac1 = {
    'semantic_qa_pass_controls': all(v for k, v in ac1_semantic.items() if k in ('A0', 'A0b')),
    'split_arm_semantic_qa_pass': ac1_semantic.get(treat),
    'check_telemetry_ok': checktel_ok,
    'treatment_nonfinite_logit_steps': treat_nonfinite,
    'top1_agreement': agr,
    'top1_requires': 0.99,
    'top1_met': (agr is not None and agr >= 0.99),
    'n_unexplained_divergences': unexplained if div is not None else None,
    'near_tie_rule_met': (div is not None and unexplained == 0),
    'VERDICT': None,
}
if not ac1['check_telemetry_ok']:
    ac1['VERDICT'] = 'INVALID (a trace failed check-telemetry)'
elif agr is None:
    ac1['VERDICT'] = 'INVALID (top-1 readout missing)'
elif not ac1['split_arm_semantic_qa_pass']:
    ac1['VERDICT'] = ('FAIL (split arm fails semantic QA; %d non-finite logit steps)'
                      % treat_nonfinite)
elif not ac1['top1_met']:
    ac1['VERDICT'] = 'FAIL (top-1 agreement %.4f < 0.99)' % agr
elif not ac1['near_tie_rule_met']:
    ac1['VERDICT'] = 'FAIL (unexplained divergence: %d)' % ac1['n_unexplained_divergences']
else:
    ac1['VERDICT'] = 'PASS (as this gate defines PASS)'
verdict['ac1'] = ac1

verdict['session_wall_s'] = round(time.time() - t0, 1)
out = S + '-verdict.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(verdict, f, indent=1)
log('')
log('=' * 72)
log('AC1 VERDICT: %s' % ac1['VERDICT'])
log('  check-telemetry OK          : %s' % ac1['check_telemetry_ok'])
log('  control (A0/A0b) QA pass    : %s' % ac1['semantic_qa_pass_controls'])
log('  split (A2) QA pass          : %s' % ac1['split_arm_semantic_qa_pass'])
log('  split (A2d) QA pass         : %s' % ac1_semantic.get('A2d'))
log('  top-1 agreement A2 vs A0    : %s (needs >= 0.99, over %s steps)'
    % (agr, shared_n))
log('  divergences / unexplained   : %s / %s'
    % (verdict['checks']['n_divergent'], verdict['checks']['n_unexplained']))
log('  A2 non-finite logit steps   : %d' % treat_nonfinite)
log('  drift floor                 : %s' % floor_note)
log('  A2d host share              : %s (does the DEFAULT budget engage the leg?)'
    % side['A2d']['host_share'])
log('=' * 72)
log('VERDICT WRITTEN ' + out)
log('session wall %.1fs' % verdict['session_wall_s'])
