#!/usr/bin/env python3
"""T-GPLR-61 AC2 paired session -- run INSIDE ONE with-gpu-lock hold.

Session name gplr61-ac2: every artifact it writes is prefixed gplr61-ac2,
so no recorded evidence is ever overwritten. Three arms share one fresh
calibrated 512-token anchor prompt:

  A  recorded streaming shape (all-GPU, no row-set)
  B  row-set reactive, all rows device (the paired control)
  C  row-set reactive + --experts-host 2 + --experts-host-budget inf (candidate)

Primary AC2 check: C ids == B ids, byte-identical. Secondary context:
B == A and A == the recorded gplr07-qa-512 control ids. QA: check-answer
PASS 5/5 on C plus must-fail negative control. AC4 parsed from C's
layer_compute records; AC7 = host share; AC3 = walls + crash grep.
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build', 'llama-cuda')
EXE = os.path.join(BUILD, 'bin', 'Release', 'llama-layer-streaming.exe')
if len(sys.argv) > 1:
    EXE = sys.argv[1]  # exe path doubles as the wrapper's provenance token
TOK = os.path.join(BUILD, 'bin', 'Release', 'llama-tokenize.exe')
MODEL = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
GEN = os.path.join(ROOT, 'tools', 'semantic-qa', 'gen-needle-prompt.py')
CHK = os.path.join(ROOT, 'tools', 'semantic-qa', 'check-answer.py')
S = os.path.join(BUILD, 'gplr61-ac2')
ANCHOR = S + '-anchor-512.txt'
RECORDED = os.path.join(BUILD, 'gplr07-qa-512.jsonl')
ROW_BYTES = 1769472

verdict = {'session': 'gplr61-ac2', 'arms': {}, 'checks': {}, 'notes': []}
t_session0 = time.time()

def log(*a):
    print(*a, flush=True)

def run(cmd, timeout=900):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors='replace', timeout=timeout)
        out = (p.stdout or '') + (p.stderr or '')
        return p.returncode, time.time() - t0, out
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b'') if isinstance(e.stdout, bytes) else (e.stdout or ''))
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t0, out + ' TIMEOUT'

# ---- 1. calibrate a FRESH anchor (never touches the recorded prompt) ----
rc, dur, out = run([sys.executable, GEN, '--target-tokens', '512',
                    '--calibrate', TOK, '--model', MODEL, '--out', ANCHOR], 300)
m1 = re.search(r'achieved=(\d+)', out)
m2 = re.search(r'items=(\S+)', out)
if rc != 0 or not m1 or not m2:
    log('CALIBRATION FAILED rc=', rc)
    log(out[-3000:])
    sys.exit(1)
ACH = int(m1.group(1))
ITEMS = m2.group(1)
verdict['achieved'] = ACH
verdict['items'] = ITEMS
log('calibrated achieved=%d items=%s' % (ACH, ITEMS))

base = ['--model', MODEL, '--prompt-file', ANCHOR, '--ubatch', str(ACH),
        '--tg-tokens', '96', '--pp-no-logits', '--pp-schedule', 'batch-major',
        '--residency', 'moe-expert-bank', '--kv-cache', 'q8', '--allow-full-residency']
arms = [
    ('A', '-streaming', ['--no-handoff']),
    ('B', '-reactive', ['--no-handoff', '--residency-tg']),
    ('C', '-split', ['--no-handoff', '--residency-tg',
                       '--experts-host', '2', '--experts-host-budget', 'inf']),
]
for name, tag, flags in arms:
    tr = S + '-arm' + name + tag + '.jsonl'
    tx = S + '-arm' + name + tag + '.txt'
    lg = S + '-arm' + name + tag + '.log'
    cmd = [EXE] + base + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd, 900)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name] = {'rc': rc, 'wall_s': round(dur, 1), 'trace': tr,
                             'tgtext': tx, 'log': lg, 'cmd': ' '.join(cmd)}
    log('ARM %s rc=%s wall=%.1fs' % (name, rc, dur))
    if rc != 0:
        log('ARM %s tail:' % name, out[-1500:])

# ---- 2. strict check-telemetry per arm ----
for name in ('A', 'B', 'C'):
    a = verdict['arms'].get(name)
    if not a or a['rc'] != 0:
        verdict['checks']['checktel_' + name] = 'skipped (arm failed)'
        continue
    rc, dur, out = run([EXE, '--check-telemetry', a['trace']], 300)
    verdict['checks']['checktel_' + name] = rc
    tail = out.strip().splitlines()[-1] if out.strip() else ''
    log('check-telemetry %s rc=%d %s' % (name, rc, tail))

# ---- 3. greedy-ID extraction and comparison ----
def ids_of(path):
    try:
        fh = open(path, encoding='utf-8')
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('event') == 'tg_tokens':
                return ev.get('ids')
    return None

ids = {}
for name, a in verdict['arms'].items():
    ids[name] = ids_of(a['trace']) if a['rc'] == 0 else None
ids['RECORDED'] = ids_of(RECORDED)

def eq(x, y):
    return x is not None and y is not None and x == y

verdict['checks']['ac2_ids_C_eq_B'] = eq(ids['C'], ids['B'])
verdict['checks']['ids_B_eq_A'] = eq(ids['B'], ids['A'])
verdict['checks']['ids_A_eq_recorded'] = eq(ids['A'], ids['RECORDED'])
for k in ('A', 'B', 'C', 'RECORDED'):
    v = ids.get(k)
    log('ids %-8s %s' % (k, ('len=%d' % len(v)) if v else 'MISSING'))
first_diff = None
if ids.get('B') is not None and ids.get('C') is not None and ids['B'] != ids['C']:
    for i in range(max(len(ids['B']), len(ids['C']))):
        b = ids['B'][i] if i < len(ids['B']) else None
        c = ids['C'][i] if i < len(ids['C']) else None
        if b != c:
            first_diff = {'pos': i, 'B': b, 'C': c}
            break
verdict['checks']['ac2_first_diff'] = first_diff
if first_diff:
    log('AC2 FIRST DIFF', first_diff)

# ---- 4. QA grading on every arm ----
qa = {}
for name in ('A', 'B', 'C'):
    a = verdict['arms'].get(name)
    if not a or a['rc'] != 0:
        qa[name] = 'skipped'
        continue
    rcp, _, outp = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                        '--items', ITEMS], 300)
    rcn, _, outn = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                        '--items', ITEMS, '--negative'], 300)
    pos_ok = (rcp == 0) and ('aggregate=PASS' in outp)
    neg_ok = (rcn != 0) and ('aggregate=FAIL' in outn)
    qa[name] = {'positive_rc': rcp, 'positive_pass': pos_ok,
                'negative_rc': rcn, 'negative_fails_as_designed': neg_ok}
    log('QA %s positive rc=%d pass=%s negative rc=%d fails_as_designed=%s'
        % (name, rcp, pos_ok, rcn, neg_ok))
verdict['checks']['qa'] = qa

# ---- 5. AC4 / AC5 / AC6 / AC7 from the traces ----
ROWB = ROW_BYTES
def scan_trace(path, want_fields):
    st = {'records': 0, 'violations': 0, 'staged_mismatch_unclamped': 0,
          'staged_not_row_multiple': 0, 'host_finite_bad': 0,
          'gate_records': 0, 'host_sum': 0, 'total_sum': 0, 'dev_sum': 0,
          'gate_cpu': 0, 'gate_gpu': 0, 'examples': []}
    try:
        fh = open(path, encoding='utf-8')
    except OSError:
        return st
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('event') != 'layer_compute' or 'experts_total' not in ev:
                continue
            st['records'] += 1
            et, eh, ed = ev['experts_total'], ev['experts_host'], ev['experts_device']
            if ed + eh != et:
                st['violations'] += 1
                if len(st['examples']) < 3:
                    st['examples'].append(['sum', ev.get('layer'), et, eh, ed])
            sb = ev.get('staged_bytes_avoided', -1)
            if sb is not None and sb >= 0:
                if int(sb) % ROWB != 0:
                    st['staged_not_row_multiple'] += 1
                if int(sb) != eh * ROWB and eh != et:
                    st['staged_mismatch_unclamped'] += 1
                    if len(st['examples']) < 6:
                        st['examples'].append(['staged', ev.get('layer'),
                                               int(sb), eh * ROWB, eh])
            if ev.get('host_finite', -1) == 1:
                st['host_finite_bad'] += 1
            rg = ev.get('rows_gate_cpu', -1)
            if rg >= 0:
                st['gate_records'] += 1
                st['gate_cpu'] += rg
                st['gate_gpu'] += ev.get('rows_gate_gpu', 0)
            st['host_sum'] += eh
            st['total_sum'] += et
            st['dev_sum'] += ed
    return st

ac4c = scan_trace(verdict['arms'].get('C', {}).get('trace', ''), True)     if verdict['arms'].get('C', {}).get('rc') == 0 else {}
ac4b = scan_trace(verdict['arms'].get('B', {}).get('trace', ''), True)     if verdict['arms'].get('B', {}).get('rc') == 0 else {}
verdict['checks']['ac4_armC'] = ac4c
share = None
if ac4c.get('total_sum'):
    share = ac4c['host_sum'] / ac4c['total_sum']
verdict['checks']['ac7_host_share'] = share
verdict['checks']['ac5_did_not_fire'] = bool(ac4c) and ac4c.get('host_sum', 0) == 0
# AC6 inertness on arm B: flag absent -> host 0 everywhere, gate fields -1
inert_ok = bool(ac4b) and ac4b.get('host_sum', 0) == 0
verdict['checks']['ac6_B_inert'] = inert_ok
log('AC4 C records=%s viol=%s staged_mismatch_unclamped=%s host_finite_bad=%s'
    % (ac4c.get('records'), ac4c.get('violations'),
       ac4c.get('staged_mismatch_unclamped'), ac4c.get('host_finite_bad')))
log('AC4 B records=%s (host_sum=%s) inert=%s'
    % (ac4b.get('records'), ac4b.get('host_sum'), inert_ok))
log('AC7 host share = %s (host=%s total=%s gate_cpu=%s gate_gpu=%s)'
    % (share, ac4c.get('host_sum'), ac4c.get('total_sum'),
       ac4c.get('gate_cpu'), ac4c.get('gate_gpu')))

# ---- 6. AC3: walls + crash grep ----
ac3 = {'needles_found': [], 'walls': {}}
for name, a in verdict['arms'].items():
    ac3['walls'][name] = a['wall_s']
    try:
        txt = open(a['log'], encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    for needle in ('cudaErrorLaunchTimeout', 'TDR', 'out of memory', 'CUDA error'):
        if needle in txt:
            ac3['needles_found'].append([name, needle])
verdict['checks']['ac3'] = ac3
log('AC3 walls', ac3['walls'], 'needles', ac3['needles_found'])

# ---- 7. provenance + summary ----
import hashlib
def sha16(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()[:16]
verdict['provenance'] = {
    'exe_sha256_16': sha16(EXE),
    'llama_dll_sha256_16': sha16(os.path.join(BUILD, 'bin', 'Release', 'llama.dll')),
}
ct = verdict['checks']
qaC = qa.get('C') or {}
summary = {
    'ac2_identity_C_eq_B': ct.get('ac2_ids_C_eq_B'),
    'checktelemetry_all_ok': all(ct.get('checktel_' + n) == 0 for n in ('A', 'B', 'C')),
    'qa_C_pass_5_of_5': bool(qaC.get('positive_pass')),
    'qa_C_negative_fails': bool(qaC.get('negative_fails_as_designed')),
    'ac4_ok': bool(ac4c) and ac4c.get('violations') == 0
              and ac4c.get('host_finite_bad') == 0,
    'ac2_nonzero_host': bool(ac4c) and ac4c.get('host_sum', 0) > 0,
    'ac5_did_not_fire': verdict['checks']['ac5_did_not_fire'],
    'ac6_B_inert': verdict['checks']['ac6_B_inert'],
    'ac7_host_share': share,
    'ac3_clean': not ac3['needles_found'],
    'ids_A_eq_recorded': ct.get('ids_A_eq_recorded'),
    'session_wall_s': round(time.time() - t_session0, 1),
}
verdict['summary'] = summary
outp = S + '-verdict.json'
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(verdict, f, indent=1)
log('VERDICT WRITTEN ' + outp)
log('SUMMARY ' + json.dumps(summary, sort_keys=True))
