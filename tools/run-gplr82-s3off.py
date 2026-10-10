#!/usr/bin/env python3
"""T-GPLR-82(b): is the NON-FINITE CASCADE caused by the CPU tail leg, or by the
S3 admission (the copy-wait decoupling) that the CPU tail leg enables?

THE CONFOUND THIS BREAKS. Every arm that has ever driven the run non-finite has
had TWO properties at once: (i) a host (CPU) leg computing 2 of the 8 routed
rows, and (ii) the S3 admission active (--experts-host-budget > 0 and s3_cap > 0),
which is what lets a missed row be computed on the host while its copy is in
flight. Every arm with the admission OFF (budget 0 -> the split is not even
built) was bit-identical to the control, so nothing yet separates (i) from (ii).

THE SEPARATOR. s3_cap = floor(budget_us / gate_host_us) with gate_host_us = 36.5,
clamped to experts_host_n = 2. So:

    --experts-host 2 --experts-host-budget 0.001
        -> ls_split_in_play TRUE (budget > 0): the graph IS the 6/2 split and the
           tail 2 rows DO execute on the CPU (ffn_dev = 2, census MUL_MAT_ID 120)
        -> s3 = TRUE but s3_cap = floor(0.001/36.5) = 0
        -> s3_compute_split early-returns, s3_split_run = false,
           s3_need_wait = true
        => THE CPU TAIL LEG RUNS WITH THE DECOUPLING OFF: every copy awaited,
           every union row refreshed, nothing ever PENDING for the plan.

    --experts-host 2 --experts-host-budget 100
        -> s3_cap = floor(100/36.5) = 2 = experts_host_n: the SAME admission
           behaviour as 'inf' (the cap saturates), i.e. the shipped A2 arm.

So B1 vs B2 differ in ONE thing: whether the admission/decoupling is active. The
CPU tail leg is identical, the arithmetic is identical, the weights are
identical.

HOW TO READ IT (criterion stated before the run):
    B1 finite over 512 steps AND B2 non-finite -> the CPU arithmetic is
        EXONERATED; the cascade is the S3 decoupling's bookkeeping.
    B1 non-finite too -> the CPU tail leg's own arithmetic is the cause, and the
        admission is not needed to produce it.
    B1 finite and B2 finite -> the cascade needs the unbounded budget too.

ARMS (512 anchor, --ubatch 512, --residency-tg-retain 2, 512 TG tokens):
    C0  --experts-host 0                                   control
    B1  --experts-host 2 --experts-host-budget 0.001        CPU tail, S3 OFF
    B2  --experts-host 2 --experts-host-budget 100          CPU tail, S3 ON (cap=2)

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
SESSION = 'gplr82-s3off'
for _i, _a in enumerate(sys.argv):
    if _a == '--session' and _i + 1 < len(sys.argv):
        SESSION = sys.argv[_i + 1]
S = os.path.join(BUILD, SESSION)
GRADE_ONLY = '--grade-only' in sys.argv

TG_TOKENS = 512
ITEMS = 'N1,N2,N3,N4,K1'
# The recorded control this run is measured against (session gplr82-a2dev3, exe
# a9ddf1699801b235 / dll c11ac969d739f56d, arm A0).
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
    ('C0', ['--experts-host', '0'],
     'control: flag present, host leg off'),
    ('B1', ['--experts-host', '2', '--experts-host-budget', '0.001'],
     'CPU tail leg, S3 DECOUPLING OFF (s3_cap = floor(0.001/36.5) = 0)'),
    ('B2', ['--experts-host', '2', '--experts-host-budget', '100'],
     'CPU tail leg, S3 decoupling ON (s3_cap = floor(100/36.5) = 2, == inf)'),
]

verdict = {
    'session': SESSION, 'task': 'T-GPLR-82', 'kind': 'attribution',
    'ac': 'CPU-tail-vs-admission attribution (diagnostic, not an AC)',
    'grade_only': GRADE_ONLY, 'anchor': ANCHOR, 'items': ITEMS,
    'tg_tokens': TG_TOKENS,
    'recorded_control_step0': RECORDED_STEP0,
    'recorded_split_step0': RECORDED_A2_STEP0,
    'arms': {}, 'checks': {}, 'notes': [
        'B1 keeps the CPU tail leg and removes ONLY the S3 admission.',
        'ready_latency_us == 0 on a rowset_fetch whose bytes were staged is the '
        'direct evidence that a wait was skipped by the decoupling; it is counted '
        'per arm so the mechanism is measured, not inferred.',
        'Non-finite logits are written as bare nan / -inf (invalid JSON): counted, '
        'and never dropped.',
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
log('provenance: exe=%s dll=%s anchor=%s'
    % (verdict['provenance']['exe_sha256_16'],
       verdict['provenance']['llama_dll_sha256_16'],
       verdict['provenance']['anchor_sha256_16']))

# ---- 1. the arms ------------------------------------------------------------
if not GRADE_ONLY:
    os.makedirs(S, exist_ok=True)
for name, flags, why in ARMS:
    tr = os.path.join(S, '%s-arm-%s.jsonl' % (SESSION, name))
    tx = os.path.join(S, '%s-arm-%s.txt' % (SESSION, name))
    lg = os.path.join(S, '%s-arm-%s.log' % (SESSION, name))
    if GRADE_ONLY:
        rc = 0 if os.path.exists(tr) else None
        verdict['arms'][name] = {'rc': rc, 'wall_s': None, 'trace': tr, 'tgtext': tx,
                                 'log': lg, 'why': why, 'flags': flags,
                                 'cmd': '(not executed: --grade-only)'}
        log('ARM %-4s not executed (--grade-only); trace %s'
            % (name, 'present' if rc == 0 else 'MISSING'))
        continue
    cmd = [EXE] + COMMON + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name] = {'rc': rc, 'wall_s': round(dur, 1), 'trace': tr, 'tgtext': tx,
                             'log': lg, 'why': why, 'flags': flags, 'cmd': ' '.join(cmd)}
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
    a = verdict['arms'][name]
    r = {'steps': {}, 'ids': None, 'host_sum': 0, 'total_sum': 0, 'dev_sum': 0,
         'violations': 0, 'nonfinite_steps': [], 'unparsed': 0, 'unparsed_tg_logits': 0,
         'ffn_dev_hist': {}, 'graph_ops_hist': {}, 'host_finite_hist': {},
         'lc': 0, 'phases': {}, 'tg_tokens_per_s': None,
         'fetch_total': 0, 'fetch_staged': 0, 'fetch_wait_skipped': 0,
         'fetch_wait_taken': 0, 'ready_at_consumption_true': 0,
         'ready_at_consumption_false': 0}
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
            elif ev == 'rowset_fetch':
                r['fetch_total'] += 1
                if e.get('n_rows_fetched'):
                    r['fetch_staged'] += 1
                if e.get('ready_at_consumption') is True:
                    r['ready_at_consumption_true'] += 1
                elif e.get('ready_at_consumption') is False:
                    r['ready_at_consumption_false'] += 1
                if e.get('n_rows_fetched') and (e.get('ready_latency_us') or 0) == 0.0:
                    r['fetch_wait_skipped'] += 1
                elif e.get('n_rows_fetched'):
                    r['fetch_wait_taken'] += 1
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


# ---- 4. step 0 --------------------------------------------------------------
s0 = {n: step0(n) for n in verdict['arms']}
verdict['checks']['step0'] = s0
log('')
log('=== step 0 ' + '=' * 60)
ctrl = s0.get('C0')
for n in verdict['arms']:
    v = s0.get(n)
    if v is None:
        log('  %-4s step0: MISSING (no finite readout recorded)' % n)
        continue
    d_id = '' if ctrl is None else ('id %s' % ('SAME' if v['top1_id'] == ctrl['top1_id'] else 'DIFF'))
    d_lg = '' if ctrl is None or v['top1_logit'] is None or ctrl['top1_logit'] is None \
        else ('logit delta %+.6f' % (v['top1_logit'] - ctrl['top1_logit']))
    log('  %-4s id=%-8s logit=%-12s margin=%-10s  %s %s'
        % (n, v['top1_id'], v['top1_logit'], v['margin'], d_id, d_lg))
log('  recorded control (session gplr82-a2dev3, same binary family): id=%s logit=%s margin=%s'
    % (RECORDED_STEP0['top1_id'], RECORDED_STEP0['top1_logit'], RECORDED_STEP0['margin']))

# ---- 5. side evidence -------------------------------------------------------
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
        's3_fetch_total': t['fetch_total'], 's3_fetch_staged': t['fetch_staged'],
        's3_wait_skipped': t['fetch_wait_skipped'], 's3_wait_taken': t['fetch_wait_taken'],
        'ready_at_consumption_true': t['ready_at_consumption_true'],
        'ready_at_consumption_false': t['ready_at_consumption_false'],
    }
verdict['checks']['side_evidence'] = side
log('')
log('%-4s %-4s %8s %10s %8s %-22s %8s %6s' % (
    'arm', 'rc', 'tok/s', 'host_shr', 'ledger', 'ffn_dev histogram', 'nansteps', 'unparsed'))
for name in verdict['arms']:
    s = side[name]
    log('%-4s %-4s %8s %10s %8d %-22s %8d %6d' % (
        name, s['rc'],
        ('%.2f' % s['tg_tokens_per_s']) if s['tg_tokens_per_s'] else 'n/a',
        ('%.5f' % s['host_share']) if s['host_share'] is not None else 'n/a',
        s['ledger_violations'],
        str(s['ffn_dev_hist'])[:22], s['nonfinite_logit_steps'], s['unparsed_lines']))
log('')
log('=== S3 decoupling activity (the mechanism, measured) ' + '=' * 22)
log('%-4s %10s %12s %12s %16s %14s' % (
    'arm', 'fetch_evts', 'staged_evts', 'wait_skipped', 'ready_at_cons=T', 'host_finite'))
for name in verdict['arms']:
    s = side[name]
    log('%-4s %10d %12d %12d %16d %14s' % (
        name, s['s3_fetch_total'], s['s3_fetch_staged'], s['s3_wait_skipped'],
        s['ready_at_consumption_true'], str(s['host_finite_hist'])[:14]))
log('  wait_skipped = a rowset_fetch that staged bytes yet blocked for 0 us: the')
log('  decoupling ran. host_finite: the host-leg non-finite scan (only when host rows > 0).')
log('')
for name in verdict['arms']:
    log('  %-4s graph_backend_ops: %s' % (name, str(side[name]['graph_ops_hist'])[:120]))

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
        log('QA %-4s positive rc=%d pass=%s | negative rc=%d fails_as_designed=%s'
            % (name, rp, qa[name]['positive_pass'], rn,
               qa[name]['negative_fails_as_designed']))
verdict['checks']['qa'] = qa

out = os.path.join(S, SESSION + '-verdict.json')
with open(out, 'w', encoding='utf-8') as fh:
    json.dump(verdict, fh, indent=1)
log('')
log('=' * 72)
log('ATTRIBUTION INPUTS:')
log('  C0 finite         : %s' % (side['C0']['nonfinite_logit_steps'] == 0 if 'C0' in side else None))
log('  B1 (S3 OFF) finite: %s   nansteps=%s' % (
    (side['B1']['nonfinite_logit_steps'] == 0) if 'B1' in side else None,
    side['B1']['nonfinite_logit_steps'] if 'B1' in side else None))
log('  B2 (S3 ON ) finite: %s   nansteps=%s' % (
    (side['B2']['nonfinite_logit_steps'] == 0) if 'B2' in side else None,
    side['B2']['nonfinite_logit_steps'] if 'B2' in side else None))
log('  B1 wait_skipped: %s | B2 wait_skipped: %s' % (
    side['B1']['s3_wait_skipped'] if 'B1' in side else None,
    side['B2']['s3_wait_skipped'] if 'B2' in side else None))
log('session wall %.1fs | verdict %s' % (time.time() - t0, out))
