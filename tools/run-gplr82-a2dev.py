#!/usr/bin/env python3
"""T-GPLR-82 FALSIFIER: the same 6/2 routed-row partition, entirely on the device.

THE QUESTION THIS ANSWERS. Arm A2 (the shipped split: 6 rows on the device, the
last 2 per token on the CPU) reads step-0 top1_logit 16.167877 where the control
reads 16.194481 -- a 0.026604 (0.164%) shift, with a MEASURED drift floor of 0.0
(A0 vs A0b agree bit-exactly). The host leg is CPU-placed BY CONSTRUCTION: it
binds the model LOADER's host-placed expert banks, so no flag moves it. CPU
accumulation and the partition itself are therefore confounded: a 1.6e-3
relative shift is far too large for F32 regrouping of eight terms (~1e-7) and
far too small for a wrong row or a wrong id (O(10%)).

WHAT IS NEW. `--experts-host-leg device` builds the SAME tail leg -- same
host_rows width, same positional suffix, same weights from the ONE w_full, the
same build_moe_ffn reduction into the same add chain -- against the SLOT layer's
staged (device) tensors, so it executes on the same backend as the head. It is a
DIAGNOSTIC partition that exists only to break the confound; it is never a
placement policy (it gives away the mechanism's point) and it builds no host
sub-op and no hsel input.

HOW TO READ THE RESULT (the criterion is stated before the run):

  A2dev step 0 == A0 step 0 EXACTLY (top1_id, top1_logit, margin)
      -> the CPU accumulation is the cause: the partition is numerically
         transparent on the device, so the 0.164% shift is the CPU leg's.
  A2dev step 0 != A0 exactly, but much closer to it than A2
      -> BOTH effects are present, with the partition's own regrouping now
         measured; the CPU term is the larger one.
  A2dev step 0 == A2 (within rounding)
      -> the CPU is NOT the source; the partition's own arithmetic is, and the
         host leg is exonerated. That is a finding about the split's shape.

ARMS (all on the 512-token anchor, the T-GPLR-62/T-GPLR-82 shape):

    A0     --experts-host 0                                     control
    A2     --experts-host 2 --experts-host-budget inf            CPU tail (shipped)
    A2dev  --experts-host 2 --experts-host-budget inf \
             --experts-host-leg device                           DEVICE tail (new)

A0 also re-measures the recorded control on the NEW binary: if A0 does not
reproduce 16.194481 / 248046 / 0.524198 exactly, the change was not inert and
the whole comparison is void -- that is checked first and reported as such.

This runner claims nothing of its own: it records and grades.
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
CHK = os.path.join(ROOT, 'tools', 'semantic-qa', 'check-answer.py')
SESSION = 'gplr82-a2dev'
for _i, _a in enumerate(sys.argv):
    if _a == '--session' and _i + 1 < len(sys.argv):
        SESSION = sys.argv[_i + 1]
S = os.path.join(BUILD, SESSION)
GRADE_ONLY = '--grade-only' in sys.argv

TG_TOKENS = 512
ITEMS = 'N1,N2,N3,N4,K1'
# The recorded control this falsifier is measured against (session gplr82-ac1-post,
# exe 4412cc4a60309fd3, arm A0): bit-identical to A0b and to A2d.
RECORDED_STEP0 = {'top1_id': 248046, 'top1_logit': 16.194481, 'margin': 0.524198}
RECORDED_A2_STEP0 = {'top1_id': 248046, 'top1_logit': 16.167877, 'margin': 0.511299}

COMMON = [
    '--model', MODEL, '--prompt-file', ANCHOR,
    '--ubatch', '512', '--tg-tokens', str(TG_TOKENS),
    '--tg-logit-margin', str(TG_TOKENS),
    '--pp-no-logits', '--pp-schedule', 'batch-major',
    '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
    '--allow-full-residency', '--no-handoff', '--residency-tg',
    '--residency-tg-retain', '2',
]
ARMS = [
    ('A0', ['--experts-host', '0'],
     'control: flag present, host leg off'),
    ('A2', ['--experts-host', '2', '--experts-host-budget', 'inf'],
     'shipped split: tail (2 of 8 rows) on the CPU'),
    ('A2dev', ['--experts-host', '2', '--experts-host-budget', 'inf',
               '--experts-host-leg', 'device'],
     'FALSIFIER: same partition, tail on the DEVICE'),
]

verdict = {
    'session': SESSION, 'task': 'T-GPLR-82', 'kind': 'falsifier',
    'ac': 'A2-corruption attribution (diagnostic, not an AC)',
    'grade_only': GRADE_ONLY, 'anchor': ANCHOR, 'items': ITEMS,
    'tg_tokens': TG_TOKENS,
    'recorded_control_step0': RECORDED_STEP0,
    'recorded_split_step0': RECORDED_A2_STEP0,
    'arms': {}, 'checks': {}, 'notes': [
        'Falsifier: the same constant-width partition with the tail leg on the device.',
        'Criterion stated before the run: A2dev step 0 == A0 exactly -> the CPU '
        'accumulation is the cause; A2dev == A2 -> the partition arithmetic is, and '
        'the CPU leg is exonerated.',
        'A0 on the new binary is checked against the recorded control first; if it '
        'does not reproduce it, the comparison is void and is reported as such.',
        'Non-finite logits are written as bare nan / -inf (invalid JSON): counted, and '
        'never dropped.',
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
    'recorded_control_binary': 'exe 4412cc4a60309fd3 / dll 2e96ab55278a9c9c '
                               '(session gplr82-ac1-post); side-copied to '
                               'build/llama-cuda/bin/a2dev-control-4412cc4a/ before this rebuild',
}
log('provenance: exe=%s dll=%s anchor=%s'
    % (verdict['provenance']['exe_sha256_16'],
       verdict['provenance']['llama_dll_sha256_16'],
       verdict['provenance']['anchor_sha256_16']))

# ---- 1. the arms ------------------------------------------------------------
for name, flags, why in ARMS:
    tr, tx = '%s-arm-%s.jsonl' % (S, name), '%s-arm-%s.txt' % (S, name)
    lg = '%s-arm-%s.log' % (S, name)
    if GRADE_ONLY:
        rc = 0 if os.path.exists(tr) else None
        verdict['arms'][name] = {'rc': rc, 'wall_s': None, 'trace': tr, 'tgtext': tx,
                                 'log': lg, 'why': why, 'flags': flags,
                                 'cmd': '(not executed: --grade-only)'}
        log('ARM %-6s not executed (--grade-only); trace %s'
            % (name, 'present' if rc == 0 else 'MISSING'))
        continue
    cmd = [EXE] + COMMON + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name] = {'rc': rc, 'wall_s': round(dur, 1), 'trace': tr, 'tgtext': tx,
                             'log': lg, 'why': why, 'flags': flags, 'cmd': ' '.join(cmd)}
    log('ARM %-6s rc=%s wall=%.1fs  (%s)' % (name, rc, dur, why))
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
    log('check-telemetry %-6s rc=%d %s' % (name, rc, tail[-150:]))

# ---- 3. trace readers -------------------------------------------------------


def read_trace(name):
    """Per-arm: the step-0 readout, every readout, the ledger, backend census."""
    a = verdict['arms'][name]
    r = {'steps': {}, 'ids': None, 'host_sum': 0, 'total_sum': 0, 'dev_sum': 0,
         'violations': 0, 'nonfinite_steps': [], 'unparsed': 0, 'unparsed_tg_logits': 0,
         'ffn_dev_hist': {}, 'graph_ops_hist': {}, 'run_start': None,
         'host_finite_hist': {}, 'lc': 0, 'phases': {}, 'tg_tokens_per_s': None}
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
                r['unparsed'] += 1
                if '"tg_logits"' in raw:
                    r['unparsed_tg_logits'] += 1
                    m = re.search(r'"step":\s*(\d+)', raw)
                    if m:
                        r['nonfinite_steps'].append(int(m.group(1)))
                continue
            ev = e.get('event')
            if ev == 'tg_logits':
                st = e.get('step')
                r['steps'][st] = (e.get('top1_id'), e.get('top1_logit'), e.get('margin'))
                mm = e.get('margin')
                if mm is None or mm != mm:
                    r['nonfinite_steps'].append(st)
            elif ev == 'tg_tokens':
                r['ids'] = e.get('ids')
            elif ev == 'layer_compute' and 'experts_total' in e:
                r['lc'] += 1
                h = e.get('experts_host') or 0
                d = e.get('experts_device') or 0
                tot = e.get('experts_total') or 0
                r['host_sum'] += h
                r['dev_sum'] += d
                r['total_sum'] += tot
                if d + h != tot:
                    r['violations'] += 1
                key = str(e.get('ffn_dev'))
                r['ffn_dev_hist'][key] = r['ffn_dev_hist'].get(key, 0) + 1
                if h > 0:
                    key = str(e.get('host_finite'))
                    r['host_finite_hist'][key] = r['host_finite_hist'].get(key, 0) + 1
            elif ev == 'graph_backend_ops':
                key = 'part_b_host=%d part_b_device=%d' % (
                    len(e.get('part_b_host') or {}), len(e.get('part_b_device') or {}))
                r['graph_ops_hist'][key] = r['graph_ops_hist'].get(key, 0) + 1
            elif ev == 'phase_summary':
                r['phases'][e.get('phase')] = e
            elif ev == 'run_start':
                r['run_start'] = e
    tg = r['phases'].get('tg') or {}
    r['tg_tokens_per_s'] = tg.get('tokens_per_s')
    return r


traces = {n: read_trace(n) for n in verdict['arms']}


def step0(name):
    t = traces[name]['steps']
    if not t:
        return None
    k = min(t.keys())
    return {'step': k, 'top1_id': t[k][0], 'top1_logit': t[k][1], 'margin': t[k][2]}


# ---- 4. THE FALSIFIER: step 0, exactly -------------------------------------
s0 = {n: step0(n) for n in verdict['arms']}
verdict['checks']['step0'] = s0
log('')
log('=== step 0 (the falsifier\'s measurement) ' + '=' * 40)
ctrl = s0.get('A0')
for n in verdict['arms']:
    v = s0.get(n)
    if v is None:
        log('  %-6s step0: MISSING (arm did not record a readout)' % n)
        continue
    d_id = '' if ctrl is None else ('id %s' % ('SAME' if v['top1_id'] == ctrl['top1_id'] else 'DIFF'))
    d_lg = '' if ctrl is None or v['top1_logit'] is None or ctrl['top1_logit'] is None \
        else ('logit delta %+.6f' % (v['top1_logit'] - ctrl['top1_logit']))
    log('  %-6s id=%-8s logit=%-12s margin=%-10s  %s %s'
        % (n, v['top1_id'], v['top1_logit'], v['margin'], d_id, d_lg))
log('  recorded control (old binary): id=%s logit=%s margin=%s'
    % (RECORDED_STEP0['top1_id'], RECORDED_STEP0['top1_logit'], RECORDED_STEP0['margin']))
log('  recorded split   (old binary): id=%s logit=%s margin=%s'
    % (RECORDED_A2_STEP0['top1_id'], RECORDED_A2_STEP0['top1_logit'], RECORDED_A2_STEP0['margin']))


def exactly(a, b):
    va, vb = s0.get(a), s0.get(b)
    if va is None or vb is None:
        return None
    return (va['top1_id'] == vb['top1_id'] and va['top1_logit'] == vb['top1_logit']
            and va['margin'] == vb['margin'])


def reproduces_recorded(name):
    v = s0.get(name)
    if v is None:
        return None
    return (v['top1_id'] == RECORDED_STEP0['top1_id']
            and v['top1_logit'] == RECORDED_STEP0['top1_logit']
            and v['margin'] == RECORDED_STEP0['margin'])


verdict['checks']['falsifier'] = {
    'A0_reproduces_recorded_control': reproduces_recorded('A0'),
    'A2dev_step0_equals_A0_exactly': exactly('A2dev', 'A0'),
    'A2_step0_equals_A0_exactly': exactly('A2', 'A0'),
    'A2dev_step0_equals_A2_exactly': exactly('A2dev', 'A2'),
}
f = verdict['checks']['falsifier']
log('')
log('=== verdict inputs ' + '=' * 55)
log('  A0 reproduces the recorded control exactly : %s' % f['A0_reproduces_recorded_control'])
log('  A2  step0 == A0 (the known shift, re-measured): %s' % f['A2_step0_equals_A0_exactly'])
log('  A2dev step0 == A0 EXACTLY (the falsifier)    : %s' % f['A2dev_step0_equals_A0_exactly'])
log('  A2dev step0 == A2 exactly                    : %s' % f['A2dev_step0_equals_A2_exactly'])
if f['A0_reproduces_recorded_control'] is False:
    log('  NOTE: A0 does NOT reproduce the recorded control -- the comparison below '
        'is VOID as a binary-to-binary statement; the new build is not inert.')

# ---- 5. backend and ledger side-evidence -----------------------------------
side = {}
for name, a in verdict['arms'].items():
    t = traces[name]
    side[name] = {
        'rc': a['rc'], 'wall_s': a['wall_s'],
        'experts_host_sum': t['host_sum'], 'experts_device_sum': t['dev_sum'],
        'experts_total_sum': t['total_sum'],
        'host_share': (t['host_sum'] / t['total_sum']) if t['total_sum'] else None,
        'ledger_violations': t['violations'],
        'ffn_dev_hist': t['ffn_dev_hist'],
        'graph_ops_hist': t['graph_ops_hist'],
        'host_finite_hist': t['host_finite_hist'],
        'layer_compute_records': t['lc'],
        'nonfinite_logit_steps': len(t['nonfinite_steps']),
        'first_nonfinite_step': (t['nonfinite_steps'][0] if t['nonfinite_steps'] else None),
        'unparsed_lines': t['unparsed'], 'unparsed_tg_logits': t['unparsed_tg_logits'],
        'steps_recorded': len(t['steps']),
        'tg_tokens_per_s': t['tg_tokens_per_s'],
        'ids_len': len(t['ids']) if t['ids'] else 0,
    }
verdict['checks']['side_evidence'] = side
log('')
log('%-6s %-4s %8s %10s %8s %-26s %8s %6s' % (
    'arm', 'rc', 'tok/s', 'host_shr', 'ledger', 'ffn_dev histogram', 'nansteps', 'unparsed'))
for name in verdict['arms']:
    s = side[name]
    log('%-6s %-4s %8s %10s %8d %-26s %8d %6d' % (
        name, s['rc'],
        ('%.2f' % s['tg_tokens_per_s']) if s['tg_tokens_per_s'] else 'n/a',
        ('%.5f' % s['host_share']) if s['host_share'] is not None else 'n/a',
        s['ledger_violations'],
        str(s['ffn_dev_hist'])[:26], s['nonfinite_logit_steps'], s['unparsed_lines']))
log('')
for name in verdict['arms']:
    log('  %-6s graph_backend_ops: %s' % (name, str(side[name]['graph_ops_hist'])[:150]))
log('')
log('  ffn_dev semantics: 1 = the whole FFN ran on the device, 2 = MIXED (some op on '
    'the CPU), -1/absent = unclassified.')

# ---- 6. semantic QA, positive + negative control ---------------------------
qa = {}
if not GRADE_ONLY:
    for name, a in verdict['arms'].items():
        if a['rc'] != 0:
            qa[name] = 'skipped (arm failed)'
            continue
        rp, _, op = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                         '--items', ITEMS], 600)
        rn, _, on = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                         '--items', ITEMS, '--negative'], 600)
        qa[name] = {'positive_rc': rp, 'positive_pass': (rp == 0) and ('aggregate=PASS' in op),
                    'negative_rc': rn,
                    'negative_fails_as_designed': (rn != 0) and ('aggregate=FAIL' in on)}
        log('QA %-6s positive rc=%d pass=%s | negative rc=%d fails_as_designed=%s'
            % (name, rp, qa[name]['positive_pass'], rn,
               qa[name]['negative_fails_as_designed']))
verdict['checks']['qa'] = qa

out = S + '-verdict.json'
with open(out, 'w', encoding='utf-8') as fh:
    json.dump(verdict, fh, indent=1)
log('')
log('=' * 72)
log('FALSIFIER INPUTS: A2dev==A0 exact: %s | A2==A0 exact: %s | A0 reproduces recorded: %s'
    % (f['A2dev_step0_equals_A0_exactly'], f['A2_step0_equals_A0_exactly'],
       f['A0_reproduces_recorded_control']))
log('session wall %.1fs | verdict %s' % (time.time() - t0, out))
