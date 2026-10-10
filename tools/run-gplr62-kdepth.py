#!/usr/bin/env python3
"""T-GPLR-62 follow-up: the row-cache DEPTH arms the T-GPLR-61 session missed.

T-GPLR-61 AC2 ran both row-set arms at the row cache's DEFAULT (off): its
launcher passed --residency-tg but never --residency-tg-retain, so
rowset_init.retain_k == 0 and every round re-staged the full 320-row union
(n_rows_reused == 0 in all 96 rounds). The contract reasons from K=4 (12.5%
of a bank resident) and the 8 GB shape's K=2, so the recorded split verdict
is a K=0 verdict.

This session runs the depth arms that were missing, on the CURRENT binary,
in ONE with-gpu-lock hold, plus a same-binary K=0 reference:

  K0   row-set reactive, cache OFF, split OFF   (same-binary identity ref)
  K2   cache K=2, split OFF
  K4   cache K=4, split OFF
  K2S  cache K=2 + --experts-host 2             (the contract's shape)
  K4S  cache K=4 + --experts-host 2             (the contract's shape)
  K0b  repeat of K0                             (drift bracket, same binary)

Why the split picture can change with depth: ls_plan_split() sends a row to
the host ONLY when its state is PENDING (issued, copy not complete);
READY rows always execute on the device. With the cache ON a large share of
demanded rows are already READY, so --experts-host N caps a smaller
host-eligible pool and the realised host share is expected to FALL below the
K=0 value. That interaction is the point of the S arms, not an anomaly.

Session name gplr62-kdepth: every artifact carries that prefix, so no
recorded T-GPLR-61 / T-GPLR-19 evidence is overwritten. No claim is made
here; the runner only records what happened.
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
if len(sys.argv) > 1:
    EXE = sys.argv[1]  # exe path doubles as the wrapper's provenance token
TOK = os.path.join(BUILD, 'bin', 'Release', 'llama-tokenize.exe')
MODEL = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
GEN = os.path.join(ROOT, 'tools', 'semantic-qa', 'gen-needle-prompt.py')
CHK = os.path.join(ROOT, 'tools', 'semantic-qa', 'check-answer.py')
S = os.path.join(BUILD, 'gplr62-kdepth')
ANCHOR = S + '-anchor-512.txt'
ROW_BYTES = 1769472

verdict = {'session': 'gplr62-kdepth', 'arms': {}, 'checks': {}, 'notes': [
    'T-GPLR-61 AC2 ran its row-set arms with the row cache at its default '
    '(off): retain_k=0, no reuse. This session supplies the missing K arms.',
    'Split interaction: only PENDING rows are host-eligible, so a deeper '
    'cache is expected to lower the realised host share.',
]}
t_session0 = time.time()


def log(*a):
    print(*a, flush=True)


def run(cmd, timeout=1800):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors='replace', timeout=timeout)
        out = (p.stdout or '') + (p.stderr or '')
        return p.returncode, time.time() - t0, out
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t0, str(out) + ' TIMEOUT'


def sha16(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()[:16]


# ---- 1. calibrate a FRESH anchor (never touches the recorded prompt) ----
rc, dur, out = run([sys.executable, GEN, '--target-tokens', '512',
                    '--calibrate', TOK, '--model', MODEL, '--out', ANCHOR], 600)
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
        '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
        '--allow-full-residency', '--no-handoff', '--residency-tg']
SPLIT = ['--experts-host', '2', '--experts-host-budget', 'inf']
arms = [
    ('K0', '-k0-ref', []),
    ('K2', '-k2', ['--residency-tg-retain', '2']),
    ('K4', '-k4', ['--residency-tg-retain', '4']),
    ('K2S', '-k2-split', ['--residency-tg-retain', '2'] + SPLIT),
    ('K4S', '-k4-split', ['--residency-tg-retain', '4'] + SPLIT),
    ('K0b', '-k0-ref2', []),
]
for name, tag, flags in arms:
    tr = S + '-arm' + tag + '.jsonl'
    tx = S + '-arm' + tag + '.txt'
    lg = S + '-arm' + tag + '.log'
    cmd = [EXE] + base + flags + ['--telemetry', tr, '--tg-text', tx]
    rc, dur, out = run(cmd, 1800)
    with open(lg, 'w', encoding='utf-8') as f:
        f.write(out)
    verdict['arms'][name] = {'rc': rc, 'wall_s': round(dur, 1), 'trace': tr,
                             'tgtext': tx, 'log': lg, 'flags': flags,
                             'cmd': ' '.join(cmd)}
    log('ARM %-4s rc=%s wall=%.1fs' % (name, rc, dur))
    if rc != 0:
        log('ARM %s tail:' % name, out[-1800:])

# ---- 2. strict check-telemetry per arm ----
for name, a in verdict['arms'].items():
    if a['rc'] != 0:
        verdict['checks']['checktel_' + name] = 'skipped (arm failed)'
        continue
    rc, dur, out = run([EXE, '--check-telemetry', a['trace']], 600)
    verdict['checks']['checktel_' + name] = rc
    tail = out.strip().splitlines()[-1] if out.strip() else ''
    log('check-telemetry %-4s rc=%d %s' % (name, rc, tail[-160:]))

# ---- 3. greedy ids + first diff vs the same-binary K0 reference ----


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
ref = ids.get('K0')

for name, a in verdict['arms'].items():
    v = ids.get(name)
    log('ids %-4s %s' % (name, ('len=%d' % len(v)) if v else 'MISSING'))
    if v is None or ref is None:
        verdict['checks']['ids_%s_eq_K0' % name] = None
        continue
    verdict['checks']['ids_%s_eq_K0' % name] = (v == ref)
    if v != ref:
        for i in range(max(len(ref), len(v))):
            b = ref[i] if i < len(ref) else None
            c = v[i] if i < len(v) else None
            if b != c:
                verdict['checks']['first_diff_' + name] = {'pos': i, 'K0': b,
                                                           name: c}
                log('FIRST DIFF %s at %d: K0=%s %s=%s' % (name, i, b, name, c))
                break

# ---- 4. phase summaries: PP/TG tok/s, wait/transfer/compute ----


def phases_of(path):
    out = {}
    try:
        fh = open(path, encoding='utf-8', errors='replace')
    except OSError:
        return out
    with fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('event') == 'phase_summary':
                out[ev.get('phase')] = ev
    return out


for name, a in verdict['arms'].items():
    a['phases'] = phases_of(a['trace']) if a['rc'] == 0 else {}
    tg = a['phases'].get('tg') or {}
    pp = a['phases'].get('pp') or {}
    if tg:
        log('%-4s  PP %6.2f tok/s   TG %6.3f tok/s   TG wall %7.2fs   '
            'wait %6.2fs   transfer %6.2fs   compute %6.2fs   bytes %s'
            % (name, pp.get('tokens_per_s', 0), tg.get('tokens_per_s', 0),
               (tg.get('wall_us') or 0) / 1e6, (tg.get('total_wait_us') or 0) / 1e6,
               (tg.get('total_transfer_us') or 0) / 1e6,
               (tg.get('total_compute_us') or 0) / 1e6, tg.get('bytes_transferred')))

# ---- 5. row-cache + split telemetry per arm ----
for name, a in verdict['arms'].items():
    if a['rc'] != 0:
        continue
    st = {'retain_k': None, 'n_round': 0, 'rows_fetched': 0, 'rows_reused': 0,
          'rows_resident': 0, 'rows_retained': 0, 'staged_bytes': 0,
          'fetch_records': 0, 'lc_records': 0, 'host_sum': 0, 'total_sum': 0,
          'gate_cpu': 0, 'gate_gpu': 0, 'host_zero_all': None,
          'staged_not_row_multiple': 0, 'violations': 0}
    try:
        fh = open(a['trace'], encoding='utf-8', errors='replace')
    except OSError:
        continue
    with fh:
        for line in fh:
            if '"event"' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            e = ev.get('event')
            if e == 'rowset_init':
                st['retain_k'] = ev.get('retain_k')
            elif e == 'rowset_round':
                st['n_round'] += 1
                st['rows_fetched'] += ev.get('n_rows_fetched') or 0
                st['rows_reused'] += ev.get('n_rows_reused') or 0
                st['staged_bytes'] += ev.get('bytes_staged') or 0
            elif e == 'rowset_fetch':
                st['fetch_records'] += 1
                st['rows_resident'] += ev.get('n_rows_resident') or 0
                st['rows_retained'] += ev.get('n_rows_retained') or 0
            elif e == 'layer_compute' and 'experts_total' in ev:
                st['lc_records'] += 1
                st['host_sum'] += ev.get('experts_host') or 0
                st['total_sum'] += ev.get('experts_total') or 0
                if ev.get('experts_device', 0) + (ev.get('experts_host') or 0) != \
                        ev.get('experts_total'):
                    st['violations'] += 1
                sb = ev.get('staged_bytes_avoided')
                if sb is not None and int(sb) >= 0 and int(sb) % ROW_BYTES != 0:
                    st['staged_not_row_multiple'] += 1
                rg = ev.get('rows_gate_cpu')
                if rg is not None and rg >= 0:
                    st['gate_cpu'] += rg
                    st['gate_gpu'] += ev.get('rows_gate_gpu') or 0
    if st['lc_records']:
        st['host_zero_all'] = (st['host_sum'] == 0)
        st['host_share'] = st['host_sum'] / st['total_sum'] if st['total_sum'] else None
    else:
        st['host_share'] = None
    a['rowset'] = st
    log('%-4s  retain_k=%s rounds=%d fetched=%d reused=%d resident_rows=%d '
        'staged=%.3f GiB  host_share=%s (lc=%d)'
        % (name, st['retain_k'], st['n_round'], st['rows_fetched'],
           st['rows_reused'], st['rows_resident'], st['staged_bytes'] / 2**30,
           ('%.4f' % st['host_share']) if st['host_share'] is not None else 'n/a',
           st['lc_records']))

# ---- 6. QA grading on every arm ----
qa = {}
for name, a in verdict['arms'].items():
    if a['rc'] != 0:
        qa[name] = 'skipped'
        continue
    rcp, _, outp = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                        '--items', ITEMS], 600)
    rcn, _, outn = run([sys.executable, CHK, 'check', '--tg-text', a['tgtext'],
                        '--items', ITEMS, '--negative'], 600)
    qa[name] = {'positive_rc': rcp, 'positive_pass': (rcp == 0) and ('aggregate=PASS' in outp),
                'negative_rc': rcn, 'negative_fails_as_designed': (rcn != 0) and ('aggregate=FAIL' in outn)}
    log('QA %-4s positive rc=%d pass=%s   negative rc=%d fails_as_designed=%s'
        % (name, rcp, qa[name]['positive_pass'], rcn,
           qa[name]['negative_fails_as_designed']))
verdict['checks']['qa'] = qa

# ---- 7. crash/refusal grep + provenance ----
needles = []
for name, a in verdict['arms'].items():
    try:
        txt = open(a['log'], encoding='utf-8', errors='replace').read()
    except OSError:
        continue
    for needle in ('cudaErrorLaunchTimeout', 'TDR', 'out of memory',
                   'CUDA error', 'error:', 'refus'):
        if needle in txt:
            needles.append([name, needle])
verdict['checks']['needles_found'] = needles
verdict['provenance'] = {
    'exe_sha256_16': sha16(EXE),
    'llama_dll_sha256_16': sha16(os.path.join(BUILD, 'bin', 'Release', 'llama.dll')),
}
verdict['session_wall_s'] = round(time.time() - t_session0, 1)

outp = S + '-verdict.json'
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(verdict, f, indent=1)
log('VERDICT WRITTEN ' + outp)
log('session wall %.1fs' % verdict['session_wall_s'])
