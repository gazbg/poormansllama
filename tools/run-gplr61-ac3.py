#!/usr/bin/env python3
"""T-GPLR-61 AC3 follow-up session -- ONE with-gpu-lock hold, name gplr61-ac3.

Closes the two audit gaps without touching gplr61-ac2 evidence:
  (1) route telemetry + --pp-repeat 2 on every TIMED arm so
      tools/check-route-parity.py can issue a measured pass, and PP route
      tuples are comparable across arms;
  (2) comparable timings: cold ID-reference runs go first (they pay the
      cold file cache), then one warm-up of the A-mode, then two TIMED
      passes in opposite orders (A,B,C then C,B,A).

Arms:
  Cref/Bref  exact gplr61-ac2 flags (no repeat/route) -- ID anchors only
  A-warm     streaming mode + repeat/route -- first-touch, excluded from timing
  *-p1/-p2   timed passes; A=streaming, B=row-set all-device, C=split
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build', 'llama-cuda')
EXE = os.path.join(BUILD, 'bin', 'Release', 'llama-layer-streaming.exe')
if len(sys.argv) > 1:
    EXE = sys.argv[1]
TOK = os.path.join(BUILD, 'bin', 'Release', 'llama-tokenize.exe')
MODEL = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
GEN = os.path.join(ROOT, 'tools', 'semantic-qa', 'gen-needle-prompt.py')
S = os.path.join(BUILD, 'gplr61-ac3')
ANCHOR = S + '-anchor-512.txt'
AC2 = os.path.join(BUILD, 'gplr61-ac2')
RECORDED = os.path.join(BUILD, 'gplr07-qa-512.jsonl')

def log(*a):
    print(*a, flush=True)

def run(cmd, timeout=900):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors='replace', timeout=timeout)
        return p.returncode, time.time() - t0, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t0, out + ' TIMEOUT'

verdict = {'session': 'gplr61-ac3', 'arms': {}, 'checks': {}}
t0 = time.time()

# 1. fresh calibration (never touches recorded prompts)
rc, dur, out = run([sys.executable, GEN, '--target-tokens', '512',
                    '--calibrate', TOK, '--model', MODEL, '--out', ANCHOR], 300)
import re
m1 = re.search(r'achieved=(\d+)', out)
m2 = re.search(r'items=(\S+)', out)
if rc != 0 or not m1 or not m2:
    log('CALIBRATION FAILED rc=', rc, out[-2000:])
    sys.exit(1)
ACH, ITEMS = int(m1.group(1)), m2.group(1)
verdict['achieved'], verdict['items'] = ACH, ITEMS
log('calibrated achieved=%d items=%s' % (ACH, ITEMS))

# anchor identity vs ac2 and the recorded prompt
def sha_file(p):
    import hashlib
    try:
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    except OSError:
        return None
verdict['anchors'] = {
    'ac3': sha_file(ANCHOR),
    'ac2': sha_file(AC2 + '-anchor-512.txt'),
    'recorded': sha_file(os.path.join(BUILD, 'gplr07-qa-prompts', 'anchor-512.txt')),
}
log('anchor sha16:', verdict['anchors'])

base = ['--model', MODEL, '--prompt-file', ANCHOR, '--ubatch', str(ACH),
        '--tg-tokens', '96', '--pp-no-logits', '--pp-schedule', 'batch-major',
        '--residency', 'moe-expert-bank', '--kv-cache', 'q8', '--allow-full-residency']
AF = ['--no-handoff']
BF = ['--no-handoff', '--residency-tg']
CF = BF + ['--experts-host', '2', '--experts-host-budget', 'inf']
RR = ['--pp-repeat', '2', '--route-telemetry', '1']

# (label, tag, flags, timed)
arms = [
    ('Cref', '-ref',  CF, False),
    ('Bref', '-ref',  BF, False),
    ('A',    '-warm', AF + RR, False),
    ('A',    '-p1',   AF + RR, True),
    ('B',    '-p1',   BF + RR, True),
    ('C',    '-p1',   CF + RR, True),
    ('C',    '-p2',   CF + RR, True),
    ('B',    '-p2',   BF + RR, True),
    ('A',    '-p2',   AF + RR, True),
]
for name, tag, flags, timed in arms:
    stem = S + '-arm' + name + tag
    tr, tx, lg = stem + '.jsonl', stem + '.txt', stem + '.log'
    cmd = [EXE] + base + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd, 900)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name + tag] = {
        'rc': rc, 'wall_s': round(dur, 1), 'timed': timed,
        'trace': tr, 'tgtext': tx, 'log': lg, 'cmd': ' '.join(cmd)}
    log('ARM %s%s rc=%s wall=%.1fs timed=%s' % (name, tag, rc, dur, timed))
    if rc != 0:
        log('ARM %s%s tail:' % (name, tag), out[-1200:])

# 2. strict check-telemetry on every trace
for k, a in verdict['arms'].items():
    rc, dur, out = run([EXE, '--check-telemetry', a['trace']], 300)
    verdict['checks']['checktel_' + k] = rc
    tail = out.strip().splitlines()[-1] if out.strip() else ''
    log('check-telemetry %s rc=%d %s' % (k, rc, tail))

# 3. ID extraction and comparison
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
for k, a in verdict['arms'].items():
    ids[k] = ids_of(a['trace']) if a['rc'] == 0 else None
ids['ac2A'] = ids_of(AC2 + '-armA-streaming.jsonl')
ids['ac2B'] = ids_of(AC2 + '-armB-reactive.jsonl')
ids['ac2C'] = ids_of(AC2 + '-armC-split.jsonl')
ids['REC'] = ids_of(RECORDED)

def eq(x, y):
    return x is not None and y is not None and x == y

cmp_ = {
    'Cref_eq_ac2C': eq(ids.get('Cref-ref'), ids.get('ac2C')),
    'Bref_eq_ac2B': eq(ids.get('Bref-ref'), ids.get('ac2B')),
    'Ap1_eq_ac2A': eq(ids.get('A-p1'), ids.get('ac2A')),
    'Bp1_eq_ac2B': eq(ids.get('B-p1'), ids.get('ac2B')),
    'Cp1_eq_ac2C': eq(ids.get('C-p1'), ids.get('ac2C')),
    'Ap1_eq_Ap2': eq(ids.get('A-p1'), ids.get('A-p2')),
    'Bp1_eq_Bp2': eq(ids.get('B-p1'), ids.get('B-p2')),
    'Cp1_eq_Cp2': eq(ids.get('C-p1'), ids.get('C-p2')),
    'AC2_repro_Bp1_ne_Cp1': (eq(ids.get('B-p1'), ids.get('C-p1')) is False
                             and ids.get('B-p1') is not None
                             and ids.get('C-p1') is not None),
    'Bp1_eq_Ap1': eq(ids.get('B-p1'), ids.get('A-p1')),
}
verdict['checks']['ids'] = cmp_
for k, v in sorted(cmp_.items()):
    log('IDS %-26s %s' % (k, v))
# ids is keyed by name+tag: the two cold refs are 'Cref-ref'/'Bref-ref',
# not 'Cref'/'Bref' (the old names printed MISSING for every run).
for k in ('Cref-ref', 'Bref-ref', 'A-p1', 'B-p1', 'C-p1', 'A-p2', 'B-p2', 'C-p2',
          'ac2A', 'ac2B', 'ac2C', 'REC'):
    v = ids.get(k)
    log('ids %-6s %s' % (k, ('len=%d first=%s' % (len(v), v[:3])) if v else 'MISSING'))

# 4. timing table (timed arms only, warm pass and refs excluded)
timing = {}
for arm in ('A', 'B', 'C'):
    p1 = verdict['arms'].get(arm + '-p1', {}).get('wall_s')
    p2 = verdict['arms'].get(arm + '-p2', {}).get('wall_s')
    timing[arm] = {'p1_s': p1, 'p2_s': p2,
                   'mean_s': round((p1 + p2) / 2.0, 1) if p1 and p2 else None,
                   'order_p1': 'A,B,C', 'order_p2': 'C,B,A'}
verdict['timing'] = timing
verdict['excluded_from_timing'] = {k: a['wall_s'] for k, a in verdict['arms'].items()
                                   if not a['timed']}
log('TIMING ' + json.dumps(timing))
log('EXCLUDED ' + json.dumps(verdict['excluded_from_timing']))

verdict['summary'] = {
    'ids': cmp_,
    'timing': timing,
    'checktelemetry_all_ok': all(v == 0 for k, v in verdict['checks'].items()
                                 if k.startswith('checktel_')),
    'session_wall_s': round(time.time() - t0, 1),
}
outp = S + '-verdict.json'
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(verdict, f, indent=1)
log('VERDICT WRITTEN ' + outp)
log('SUMMARY ' + json.dumps(verdict['summary'], sort_keys=True))
