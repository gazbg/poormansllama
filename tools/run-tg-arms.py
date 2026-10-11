#!/usr/bin/env python3
r"""Spec-driven TG arm runner for the layer-streaming engine (goal: beat stock TG at 8 GB / 262k).

Generalizes tools/run-gplr82-s3off.py: arms come from a JSON spec, each arm may set environment
variables, binaries may come from a side-copied directory (so a rebuild during a session cannot
change what runs), guard messages are captured from the log, the semantic-QA grader runs with
UTF-8 output (the s3off runner recorded cp1252 crashes as grades), and every arm is compared
with the spec's control arm (first token divergence and same-context logit drift).

Run it inside ONE mutex hold, and NAME the exe on the wrapped command line (via --attest) so the
wrapper's provenance gate can attest the binary without -AllowUnprovenanced:
  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag <tag> \
      -Command "python tools/run-tg-arms.py --spec <spec.json> --attest build\llama-cuda\bin\g1\llama-layer-streaming.exe"

Spec (JSON):
  {"session": "e1-guard", "exe_dir": "build/llama-cuda/bin/e1-guard", "tg_tokens": 512,
   "items": "N1,N2,N3,N4,K1", "prompt": "build/llama-cuda/gplr07-qa-prompts/anchor-512.txt",
   "common": [...optional, replaces DEFAULT_COMMON...], "extra_common": [...appended...],
   "control": "C0", "qa": true,
   "reps": 3, "interleave": true,
   "stock_exe": "llama-completion.exe", "stock_common": [...llama-completion flags...],
   "arms": [{"name": "C0", "flags": ["--experts-host", "0"], "env": {}, "why": "control"},
            {"name": "STK", "engine": "stock", "flags": ["-ncmoe", "37"], "env": {}, "why": "stock"}, ...]}
  reps / interleave : each arm runs `reps` times; interleave orders rep-major (A1 B1 C1 A2 B2 C2 ...) so
                      every arm sees the same thermal/background state. Record names become NAME#r when reps > 1.
  engine "stock"    : the arm runs stock_exe (llama-completion) with stock_common + arm flags; its stdout is the
                      generated text (pass --no-display-prompt) and its `eval time` line is the TG rate.
  env values may use {name} (record file stem) and {sdir} (session directory).
The runner records and grades. It claims nothing.
"""
import argparse
import threading
import hashlib
import importlib.util
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
REP = os.path.join(ROOT, 'tools', 'check-repetition.py')
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


def _env(env):
    e = dict(os.environ)
    e['PYTHONIOENCODING'] = 'utf-8'
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return e


def run(cmd, env=None, timeout=3600):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=timeout, env=_env(env))
        return p.returncode, time.time() - t, (p.stdout or '') + (p.stderr or '')
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout or ''
        if isinstance(out, bytes):
            out = out.decode('utf-8', 'replace')
        return -999, time.time() - t, str(out) + ' TIMEOUT'


def run_split(cmd, env=None, timeout=3600):
    """Like run(), but keeps stdout (the generated text) apart from stderr (the log)."""
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=_env(env))
        return (p.returncode, time.time() - t, p.stdout.decode('utf-8', 'replace'),
                p.stderr.decode('utf-8', 'replace'))
    except subprocess.TimeoutExpired as ex:
        so = ex.stdout.decode('utf-8', 'replace') if isinstance(ex.stdout, bytes) else str(ex.stdout or '')
        se = ex.stderr.decode('utf-8', 'replace') if isinstance(ex.stderr, bytes) else str(ex.stderr or '')
        return -999, time.time() - t, so, se + ' TIMEOUT'


def stock_tok_s(stderr):
    """TG rate from llama-completion's `eval time` line (not `prompt eval time`)."""
    for ln in stderr.splitlines():
        if 'common_perf_print' in ln and 'eval time' in ln and 'prompt' not in ln:
            m = re.search(r'/\s*(\d+)\s+runs\s*\(\s*([\d.]+)\s+ms per token,\s*([\d.]+)\s+tokens per second', ln)
            if m:
                return float(m.group(3)), int(m.group(1))
    return None, None


def poll_vram(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
            peak = max(peak, int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        stop.wait(0.4)
    out['peak_mib'] = peak


def poll_vram_fast(stop, out, ms):
    """Whole-GPU memory.used at `ms` resolution from ONE long-lived nvidia-smi (spec key poll_ms): the per-call
    poller above cannot see a spike shorter than a process launch (~0.4 s)."""
    peak = 0
    n = 0
    p = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                          '--loop-ms=%d' % ms], stdout=subprocess.PIPE, text=True)

    def reader():
        nonlocal peak, n
        for ln in p.stdout:
            try:
                peak = max(peak, int(ln.strip()))
                n += 1
            except ValueError:
                pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    stop.wait()
    p.kill()
    th.join(timeout=5)
    out['peak_mib'] = peak
    out['samples'] = n


def idle_vram():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def start_load(n):
    """n busy-loop python processes at normal priority (arm key `load`): a controlled stand-in for other heavy host work."""
    n = int(n or 0)
    return [subprocess.Popen([sys.executable, '-c', 'while True: pass'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(n)]


def stop_load(procs):
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            pass


def load_rep():
    sp = importlib.util.spec_from_file_location('check_repetition', REP)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def read_trace(path):
    r = {'steps': {}, 'ids': None, 'host_sum': 0, 'total_sum': 0, 'dev_sum': 0, 'violations': 0,
         'nonfinite_steps': [], 'unparsed': 0, 'ffn_dev_hist': {}, 'host_finite_hist': {},
         'phases': {}, 'round_wall_us': [], 'rows_fetched': [], 'bytes_staged': [],
         'fetch_wait_skipped': 0, 'fetch_staged': 0, 'fetch_total': 0, 'run_start': None,
         'host_us': [], 'ffn_compute_us': [], 'attn_compute_us': [], 'pp_last': None}
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
            elif ev == 'pp_last_logits':
                r['pp_last'] = e.get('top')
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
    ap.add_argument('--attest', nargs='*', default=[],
                    help='exe paths named only so the wrapper provenance gate can attest them (ignored here)')
    a = ap.parse_args()
    with open(a.spec, encoding='utf-8') as f:
        spec = json.load(f)
    session = spec['session']
    sdir = os.path.join(BUILD, session)
    os.makedirs(sdir, exist_ok=True)
    exe_dir = os.path.join(ROOT, spec.get('exe_dir', 'build/llama-cuda/bin/Release'))
    exe = os.path.join(exe_dir, 'llama-layer-streaming.exe')
    stock_exe = os.path.join(exe_dir, spec.get('stock_exe', 'llama-completion.exe'))
    prompt = os.path.join(ROOT, spec.get('prompt', 'build/llama-cuda/gplr07-qa-prompts/anchor-512.txt'))
    tg = int(spec.get('tg_tokens', 512))
    items = spec.get('items', 'N1,N2,N3,N4,K1')
    common = spec.get('common') or [
        '--model', MODEL, '--prompt-file', prompt, '--ubatch', '512', '--tg-tokens', str(tg),
        '--tg-logit-margin', str(tg), '--pp-no-logits', '--pp-schedule', 'batch-major',
        '--residency', 'moe-expert-bank', '--kv-cache', 'q8', '--allow-full-residency', '--no-handoff',
        '--residency-tg', '--residency-tg-retain', '2']
    common = common + spec.get('extra_common', [])
    scommon = spec.get('stock_common') or []
    reps = int(spec.get('reps', 1))
    has_stock = any(arm.get('engine') == 'stock' for arm in spec['arms'])
    has_lm = any(arm.get('engine') != 'stock' for arm in spec['arms'])
    arm_exes = [os.path.join(ROOT, arm['exe_dir'], 'llama-layer-streaming.exe') for arm in spec['arms'] if arm.get('exe_dir') and arm.get('engine') != 'stock']
    need = [prompt] + ([exe] if has_lm else []) + arm_exes + ([stock_exe] if has_stock else [])
    for p in need:
        if not os.path.exists(p):
            log('MISSING REQUIRED FILE:', p)
            return 2
    if has_stock and not scommon:
        log('stock arm without stock_common')
        return 2
    plan = []
    if spec.get('interleave'):
        for r in range(1, reps + 1):
            for arm in spec['arms']:
                plan.append((arm, r))
    else:
        for arm in spec['arms']:
            for r in range(1, reps + 1):
                plan.append((arm, r))
    t0 = time.time()
    base_vram = idle_vram()
    verdict = {'session': session, 'spec': spec, 'idle_vram_mib': base_vram, 'attest': a.attest,
               'provenance': {
                   'exe_sha256_16': sha16(exe) if has_lm else None,
                   'llama_dll_sha256_16': sha16(os.path.join(exe_dir, 'llama.dll')),
                   'stock_exe_sha256_16': sha16(stock_exe) if has_stock else None,
                   'prompt_sha256_16': sha16(prompt), 'exe_dir': exe_dir}, 'arms': {}}
    log('provenance:', json.dumps(verdict['provenance']))

    for arm, rep in plan:
        base = arm['name']
        name = base if reps == 1 else '%s#%d' % (base, rep)
        stem = base if reps == 1 else '%s_r%d' % (base, rep)
        stock = arm.get('engine') == 'stock'
        tr = os.path.join(sdir, '%s-%s.jsonl' % (session, stem))
        tx = os.path.join(sdir, '%s-%s.txt' % (session, stem))
        lg = os.path.join(sdir, '%s-%s.log' % (session, stem))
        env = {k: str(v).replace('{name}', stem).replace('{sdir}', sdir) for k, v in arm.get('env', {}).items()}
        if stock:
            cmd = [stock_exe] + scommon + arm.get('flags', [])
        else:
            arm_exe = os.path.join(ROOT, arm['exe_dir'], 'llama-layer-streaming.exe') if arm.get('exe_dir') else exe  # per-arm side-copied build for A/B across builds
            cmd = [arm_exe] + common + arm.get('flags', []) + ['--telemetry', tr, '--tg-text', tx]
        rec = {'base': base, 'rep': rep, 'engine': 'stock' if stock else 'lm', 'flags': arm.get('flags', []),
               'env': env, 'why': arm.get('why', ''), 'trace': tr, 'tgtext': tx, 'log': lg,
               'cmd': ' '.join(cmd)}
        if arm.get('exe_dir') and not stock:
            rec['exe_dir'] = arm['exe_dir']
            rec['exe_sha256_16'] = sha16(arm_exe)
            rec['llama_dll_sha256_16'] = sha16(os.path.join(ROOT, arm['exe_dir'], 'llama.dll'))
        if a.grade_only:
            rec['rc'] = 0 if (os.path.exists(tr) or os.path.exists(tx)) else None
            out = open(lg, encoding='utf-8', errors='replace').read() if os.path.exists(lg) else ''
            if stock:
                rec['stock_tok_s'], rec['stock_runs'] = stock_tok_s(out)
        else:
            stop = threading.Event(); vr = {}
            loadp = start_load(arm.get('load'))
            if loadp:
                time.sleep(1.0)  # let the load reach every core before the arm starts
                rec['load'] = len(loadp)
            # per-arm override: 'none' (no VRAM poller at all), 'call' (one nvidia-smi spawn per sample), 'loop' (one long-lived
            # nvidia-smi --loop-ms). NVML queries take driver locks, so the poller itself can stall CUDA calls of the process under test.
            mode = arm.get('poll') or ('loop' if spec.get('poll_ms') else 'call')
            if mode == 'none':
                th = None
            elif mode == 'loop':
                th = threading.Thread(target=poll_vram_fast, args=(stop, vr, int(spec.get('poll_ms') or 50)), daemon=True)
            else:
                th = threading.Thread(target=poll_vram, args=(stop, vr), daemon=True)
            if th is not None:
                th.start()
            rec['poll_mode'] = mode
            if stock:
                rc, dur, so, se = run_split(cmd, env=env)
                with open(tx, 'w', encoding='utf-8') as f:
                    f.write(so)
                out = se
                rec['stock_tok_s'], rec['stock_runs'] = stock_tok_s(se)
            else:
                rc, dur, out = run(cmd, env=env)
            stop.set()
            if th is not None:
                th.join()
            stop_load(loadp)
            rec['vram_peak_mib'] = vr.get('peak_mib')
            rec['vram_samples'] = vr.get('samples')
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
        if rec['rc'] == 0 and rec['engine'] == 'lm' and os.path.exists(rec['trace']):
            rc, _, out = run([exe, '--check-telemetry', rec['trace']], timeout=900)
            rec['check_telemetry_rc'] = rc
            rec['check_telemetry_tail'] = (out.strip().splitlines() or [''])[-1][-200:]
        traces[name] = read_trace(rec['trace'])

    ctrl_name = spec.get('control', spec['arms'][0]['name'])
    if ctrl_name not in verdict['arms'] and ('%s#1' % ctrl_name) in verdict['arms']:
        ctrl_name = '%s#1' % ctrl_name
    ctrl = traces.get(ctrl_name)
    repm = load_rep()
    for name, rec in verdict['arms'].items():
        t = traces[name]
        tgp = t['phases'].get('tg') or {}
        s = {
            'tg_tokens_per_s': tgp.get('tokens_per_s') if rec['engine'] == 'lm' else rec.get('stock_tok_s'),
            'tg_wall_us': tgp.get('wall_us'),
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
        if t['pp_last']:
            s['pp_last_top'] = t['pp_last']
        if os.path.exists(rec['tgtext']):
            s['rep6'] = repm.metrics(repm.words_of(rec['tgtext']), 6)
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

    # per-base-arm statistics over the reps (idle baseline subtracted from the VRAM peak)
    agg = {}
    for name, rec in verdict['arms'].items():
        agg.setdefault(rec['base'], []).append(rec)
    verdict['aggregate'] = {}
    for base, recs in agg.items():
        rates = [r.get('summary', {}).get('tg_tokens_per_s') for r in recs if r.get('rc') == 0]
        rates = [x for x in rates if isinstance(x, (int, float))]
        peaks = [r.get('vram_peak_mib') for r in recs if isinstance(r.get('vram_peak_mib'), (int, float))]
        net = [p - (base_vram or 0) for p in peaks]
        verdict['aggregate'][base] = {
            'n': len(recs), 'n_ok': sum(1 for r in recs if r.get('rc') == 0), 'tok_s': rates,
            'mean': statistics.mean(rates) if rates else None, 'min': min(rates) if rates else None,
            'max': max(rates) if rates else None, 'vram_peak_net_mib_max': max(net) if net else None,
            'qa_pass': [bool((r.get('qa') or {}).get('positive_pass')) if isinstance(r.get('qa'), dict) else None
                        for r in recs],
            'qa_neg_ok': [bool((r.get('qa') or {}).get('negative_fails_as_designed')) if isinstance(r.get('qa'), dict) else None
                          for r in recs],
            'rep6_ratio': [r.get('summary', {}).get('rep6', {}).get('ratio') for r in recs]}

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
    if reps > 1 or len(verdict['aggregate']) > 1:
        log('')
        log('%-8s %3s %9s %7s %7s %11s %s' % ('arm', 'n', 'mean', 'min', 'max', 'vramNetMax', 'rep6'))
        for base, ag in verdict['aggregate'].items():
            log('%-8s %3s %9s %7s %7s %11s %s' % (
                base, ag['n_ok'], '%.2f' % ag['mean'] if ag['mean'] else '-',
                '%.2f' % ag['min'] if ag['min'] else '-', '%.2f' % ag['max'] if ag['max'] else '-',
                ag['vram_peak_net_mib_max'], ['%.3f' % x if x is not None else '-' for x in ag['rep6_ratio']]))
    out = os.path.join(sdir, session + '-verdict.json')
    verdict['session_wall_s'] = round(time.time() - t0, 1)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(verdict, fh, indent=1, default=str)
    log('verdict:', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
