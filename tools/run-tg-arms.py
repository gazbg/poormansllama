#!/usr/bin/env python3
"""Spec-driven TG arm runner for the layer-streaming engine (goal: beat stock TG at 8 GB / 262k).

Generalizes tools/run-gplr82-s3off.py: arms come from a JSON spec, each arm may set environment
variables, binaries may come from a side-copied directory (so a rebuild during a session cannot
change what runs), guard messages are captured from the log, the semantic-QA grader runs with
UTF-8 output (the s3off runner recorded cp1252 crashes as grades), and every arm is compared
with the spec's control arm (first token divergence and same-context logit drift).

Run it inside ONE mutex hold:
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag <tag> -AllowUnprovenanced \
      -Command "python tools/run-tg-arms.py --spec <spec.json>"

Spec (JSON):
  {"session": "e1-guard", "exe_dir": "build/llama-cuda/bin/e1-guard", "tg_tokens": 512,
   "items": "N1,N2,N3,N4,K1", "prompt": "build/llama-cuda/gplr07-qa-prompts/anchor-512.txt",
   "common": [...optional, replaces DEFAULT_COMMON...], "extra_common": [...appended...],
   "control": "C0", "qa": true,
   "arms": [{"name": "C0", "flags": ["--experts-host", "0"], "env": {}, "why": "control"}, ...]}
The runner records and grades. It claims nothing.
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, 'build', 'llama-cuda')
MODEL = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
CHK = os.path.join(ROOT, 'tools', 'semantic-qa', 'check-answer.py')
GUARD_PATTERNS = {
    'stale_slab': r'slot slab was not written this round',
    'nonfinite_split_tensor': r'non-finite split FFN tensor',
    'nonfinite_host_row': r'non-finite host-expert output',
    'step_c_prefix': r'is not READY but would land on the device prefix',
    'ggml_assert': r'GGML_ASSERT',
}


def log(*a):
    print(*a, flush=True)


def sha16(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
    except OSError:
        return None
    return h.hexdigest()[:16]


def run(cmd, env=None, timeout=3600):
    t = time.time()
    e = dict(os.environ)
    e['PYTHONIOENCODING'] = 'utf-8'
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout, env=e)
        return p.returncode, time.time() - t, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t, str(out) + ' TIMEOUT'


def read_trace(path):
    r = {'steps': {}, 'ids': None, 'host_sum': 0, 'total_sum': 0, 'dev_sum': 0, 'violations': 0,
         'nonfinite_steps': [], 'unparsed': 0, 'ffn_dev_hist': {}, 'host_finite_hist': {},
         'phases': {}, 'round_wall_us': [], 'rows_fetched': [], 'bytes_staged': [],
         'fetch_wait_skipped': 0, 'fetch_staged': 0, 'fetch_total': 0, 'run_start': None,
         'host_us': [], 'ffn_compute_us': [], 'attn_compute_us': []}
    try:
        fh = open(path, encoding='utf-8', errors='replace')
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
                    m = re.search(r'"step":\s*(\d+)', raw)
                    if m:
                        r['nonfinite_steps'].append(int(m.group(1)))
                continue
            ev = e.get('event')
            if ev == 'tg_logits':
                st = e.get('step')
                r['steps'][st] = e
                mm = e.get('margin')
                if mm is None or mm != mm:
                    r['nonfinite_steps'].append(st)
            elif ev == 'tg_tokens':
                r['ids'] = e.get('ids')
            elif ev == 'layer_compute' and 'experts_total' in e:
                h = e.get('experts_host') or 0
                d = e.get('experts_device') or 0
                tot = e.get('experts_total') or 0
                r['host_sum'] += h
                r['dev_sum'] += d
                r['total_sum'] += tot
                if d + h != tot:
                    r['violations'] += 1
                k = str(e.get('ffn_dev'))
                r['ffn_dev_hist'][k] = r['ffn_dev_hist'].get(k, 0) + 1
                k = str(e.get('host_finite'))
                r['host_finite_hist'][k] = r['host_finite_hist'].get(k, 0) + 1
                for key, dst in (('host_us', 'host_us'), ('ffn_compute_us', 'ffn_compute_us'),
                                 ('attn_compute_us', 'attn_compute_us')):
                    if isinstance(e.get(key), (int, float)):
                        r[dst].append(e[key])
            elif ev == 'rowset_round':
                r['round_wall_us'].append(e.get('round_wall_us', 0))
                r['rows_fetched'].append(e.get('n_rows_fetched', 0))
                r['bytes_staged'].append(e.get('bytes_staged', 0))
            elif ev == 'rowset_fetch':
                r['fetch_total'] += 1
                if e.get('n_rows_fetched'):
                    r['fetch_staged'] += 1
                    if (e.get('ready_latency_us') or 0) == 0.0:
                        r['fetch_wait_skipped'] += 1
            elif ev == 'phase_summary':
                r['phases'][e.get('phase')] = e
            elif ev == 'run_start':
                r['run_start'] = e
    return r


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--grade-only', action='store_true')
    a = ap.parse_args()
    with open(a.spec, encoding='utf-8') as f:
        spec = json.load(f)
    session = spec['session']
    sdir = os.path.join(BUILD, session)
    os.makedirs(sdir, exist_ok=True)
    exe_dir = os.path.join(ROOT, spec.get('exe_dir', 'build/llama-cuda/bin/Release'))
    exe = os.path.join(exe_dir, 'llama-layer-streaming.exe')
    prompt = os.path.join(ROOT, spec.get('prompt', 'build/llama-cuda/gplr07-qa-prompts/anchor-512.txt'))
    tg = int(spec.get('tg_tokens', 512))
    items = spec.get('items', 'N1,N2,N3,N4,K1')
    common = spec.get('common') or [
        '--model', MODEL, '--prompt-file', prompt, '--ubatch', '512', '--tg-tokens', str(tg),
        '--tg-logit-margin', str(tg), '--pp-no-logits', '--pp-schedule', 'batch-major',
        '--residency', 'moe-expert-bank', '--kv-cache', 'q8', '--allow-full-residency', '--no-handoff',
        '--residency-tg', '--residency-tg-retain', '2']
    common = common + spec.get('extra_common', [])
    for p in (exe, prompt):
        if not os.path.exists(p):
            log('MISSING REQUIRED FILE:', p)
            return 2
    t0 = time.time()
    verdict = {'session': session, 'spec': spec, 'provenance': {
        'exe_sha256_16': sha16(exe), 'llama_dll_sha256_16': sha16(os.path.join(exe_dir, 'llama.dll')),
        'prompt_sha256_16': sha16(prompt), 'exe_dir': exe_dir}, 'arms': {}}
    log('provenance:', json.dumps(verdict['provenance']))

    for arm in spec['arms']:
        name = arm['name']
        tr = os.path.join(sdir, '%s-%s.jsonl' % (session, name))
        tx = os.path.join(sdir, '%s-%s.txt' % (session, name))
        lg = os.path.join(sdir, '%s-%s.log' % (session, name))
        cmd = [exe] + common + arm.get('flags', []) + ['--telemetry', tr, '--tg-text', tx]
        rec = {'flags': arm.get('flags', []), 'env': arm.get('env', {}), 'why': arm.get('why', ''),
               'trace': tr, 'tgtext': tx, 'log': lg, 'cmd': ' '.join(cmd)}
        if a.grade_only:
            rec['rc'] = 0 if os.path.exists(tr) else None
            out = open(lg, encoding='utf-8', errors='replace').read() if os.path.exists(lg) else ''
        else:
            rc, dur, out = run(cmd, env=arm.get('env'))
            with open(lg, 'w', encoding='utf-8') as f:
                f.write(out)
            rec['rc'] = rc
            rec['wall_s'] = round(dur, 1)
            log('ARM %-8s rc=%s wall=%.1fs  %s' % (name, rc, dur, rec['why']))
        rec['guards'] = {k: len(re.findall(p, out)) for k, p in GUARD_PATTERNS.items()}
        hits = [ln for ln in out.splitlines() if any(re.search(p, ln) for p in GUARD_PATTERNS.values())]
        rec['guard_lines'] = hits[:5]
        if rec['rc'] not in (0, None):
            rec['log_tail'] = out[-1200:]
            log('  tail: ' + out[-600:].replace('\n', ' | '))
        verdict['arms'][name] = rec

    traces = {}
    for name, rec in verdict['arms'].items():
        if rec['rc'] == 0 and os.path.exists(rec['trace']):
            rc, _, out = run([exe, '--check-telemetry', rec['trace']], timeout=900)
            rec['check_telemetry_rc'] = rc
            rec['check_telemetry_tail'] = (out.strip().splitlines() or [''])[-1][-200:]
        traces[name] = read_trace(rec['trace'])

    ctrl_name = spec.get('control', spec['arms'][0]['name'])
    ctrl = traces.get(ctrl_name)
    for name, rec in verdict['arms'].items():
        t = traces[name]
        tgp = t['phases'].get('tg') or {}
        s = {
            'tg_tokens_per_s': tgp.get('tokens_per_s'), 'tg_wall_us': tgp.get('wall_us'),
            'tg_total_compute_us': tgp.get('total_compute_us'), 'tg_total_transfer_us': tgp.get('total_transfer_us'),
            'tg_total_wait_us': tgp.get('total_wait_us'),
            'round_wall_us_median': med(t['round_wall_us']),
            'tok_s_from_median_round': (1e6 / med(t['round_wall_us'])) if t['round_wall_us'] else None,
            'rows_fetched_median': med(t['rows_fetched']), 'bytes_staged_median': med(t['bytes_staged']),
            'experts_host_sum': t['host_sum'], 'experts_total_sum': t['total_sum'],
            'host_share': (t['host_sum'] / t['total_sum']) if t['total_sum'] else None,
            'ledger_violations': t['violations'], 'ffn_dev_hist': t['ffn_dev_hist'],
            'host_finite_hist': t['host_finite_hist'], 'nonfinite_steps': len(t['nonfinite_steps']),
            'first_nonfinite_step': min(t['nonfinite_steps']) if t['nonfinite_steps'] else None,
            'steps_recorded': len(t['steps']), 'fetch_wait_skipped': t['fetch_wait_skipped'],
            'host_us_median': med(t['host_us']), 'ffn_compute_us_median': med(t['ffn_compute_us']),
        }
        if t['steps']:
            k0 = min(t['steps'])
            s['step0'] = {x: t['steps'][k0].get(x) for x in ('top1_id', 'top1_logit', 'margin')}
        if ctrl and ctrl['ids'] and t['ids'] and name != ctrl_name:
            n = min(len(ctrl['ids']), len(t['ids']))
            first = next((k for k in range(n) if ctrl['ids'][k] != t['ids'][k]), None)
            kend = first if first is not None else n
            ds = []
            for k in range(kend):
                c, x = ctrl['steps'].get(k), t['steps'].get(k)
                if c and x and isinstance(c.get('top1_logit'), (int, float)) and isinstance(x.get('top1_logit'), (int, float)) \
                        and math.isfinite(x['top1_logit']):
                    ds.append(abs(c['top1_logit'] - x['top1_logit']))
            s['vs_control'] = {'first_divergence_step': first, 'same_context_steps': len(ds),
                               'dlogit_max': max(ds) if ds else None, 'dlogit_median': med(ds),
                               'ctrl_margin_at_divergence': (ctrl['steps'].get(first) or {}).get('margin') if first is not None else None}
        rec['summary'] = s

    if spec.get('qa', True) and not a.grade_only:
        for name, rec in verdict['arms'].items():
            if rec['rc'] != 0:
                rec['qa'] = 'skipped (arm failed)'
                continue
            rp, _, op = run([sys.executable, CHK, 'check', '--tg-text', rec['tgtext'], '--items', items], timeout=600)
            rn, _, on = run([sys.executable, CHK, 'check', '--tg-text', rec['tgtext'], '--items', items, '--negative'], timeout=600)
            rec['qa'] = {'positive_rc': rp, 'positive_pass': rp == 0 and 'aggregate=PASS' in op,
                         'negative_rc': rn, 'negative_fails_as_designed': rn != 0 and 'aggregate=FAIL' in on,
                         'positive_tail': op.strip().splitlines()[-1:] if op.strip() else []}

    log('')
    log('%-8s %-5s %7s %7s %8s %8s %6s %9s %10s %s' % ('arm', 'rc', 'tok/s', 'medrnd', 'rowsF', 'hostshr',
                                                        'nan', 'firstdiv', 'dlogitmed', 'qa/guards'))
    for name, rec in verdict['arms'].items():
        s = rec.get('summary', {})
        vc = s.get('vs_control') or {}
        qa = rec.get('qa')
        qa_s = ('P' if qa.get('positive_pass') else 'F') + ('/n-ok' if qa.get('negative_fails_as_designed') else '/n-BAD') if isinstance(qa, dict) else str(qa)[:10]
        g = {k: v for k, v in rec['guards'].items() if v}
        log('%-8s %-5s %7s %7s %8s %8s %6s %9s %10s %s %s' % (
            name, rec['rc'],
            '%.2f' % s['tg_tokens_per_s'] if s.get('tg_tokens_per_s') else '-',
            '%.2f' % s['tok_s_from_median_round'] if s.get('tok_s_from_median_round') else '-',
            s.get('rows_fetched_median'), '%.4f' % s['host_share'] if s.get('host_share') is not None else '-',
            s.get('nonfinite_steps'), vc.get('first_divergence_step'),
            '%.4f' % vc['dlogit_median'] if vc.get('dlogit_median') is not None else '-', qa_s, g))
    out = os.path.join(sdir, session + '-verdict.json')
    verdict['session_wall_s'] = round(time.time() - t0, 1)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(verdict, fh, indent=1, default=str)
    log('verdict:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
