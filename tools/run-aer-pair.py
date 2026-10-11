#!/usr/bin/env python3
"""Paired TG measurement of the AER hybrid engine (side-copied binaries) against stock at an 8 GB / 262k-context shape.

Arms are interleaved (stock, hybrid C..., stock, ...) inside ONE mutex hold. VRAM is polled with nvidia-smi
(whole-GPU used memory minus the idle baseline), not taken from engine logs. Records and grades; claims nothing.
Run: powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag <tag> -AllowUnprovenanced -Command "python tools/run-aer-pair.py --spec <spec.json>"
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def poll_vram(stop, out):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
            v = int(r.stdout.strip().splitlines()[0])
            peak = max(peak, v)
        except Exception:
            pass
        stop.wait(0.4)
    out['peak_mib'] = peak


def idle_vram():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=10)
    return int(r.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding='utf-8'))
    exe = os.path.join(ROOT, spec['exe_dir'], 'llama-completion.exe')
    outdir = os.path.join(ROOT, 'build', 'llama-cuda', spec['session'])
    os.makedirs(outdir, exist_ok=True)
    prompt = spec.get('prompt', '')
    results = []
    base = idle_vram()
    print('idle VRAM used (MiB):', base, flush=True)
    for i, arm in enumerate(spec['arms']):
        name = '%02d-%s' % (i, arm['name'])
        pargs = ['-f', spec['prompt_file']] if spec.get('prompt_file') else ['-p', prompt]
        cmd = [exe] + spec['common'] + arm['flags'] + pargs
        env = dict(os.environ)
        env.update({k: str(v) for k, v in arm.get('env', {}).items()})
        stop = threading.Event()
        vr = {}
        th = threading.Thread(target=poll_vram, args=(stop, vr), daemon=True)
        th.start()
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=1800)
        dur = time.time() - t0
        stop.set(); th.join()
        log = (p.stderr or '') + '\n' + (p.stdout or '')
        open(os.path.join(outdir, name + '.log'), 'w', encoding='utf-8').write(p.stderr or '')
        open(os.path.join(outdir, name + '.txt'), 'w', encoding='utf-8').write(p.stdout or '')
        m = re.search(r'common_perf_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second', log)
        pm = re.search(r'common_perf_print:\s+prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens', log)
        rec = {'arm': arm['name'], 'rc': p.returncode, 'wall_s': round(dur, 1), 'peak_vram_mib': vr.get('peak_mib'),
               'vram_over_idle_mib': (vr.get('peak_mib') or 0) - base,
               'tg_tok_s': float(m.group(3)) if m else None, 'tg_tokens': int(m.group(2)) if m else None,
               'pp_tokens': int(pm.group(2)) if pm else None, 'text_chars': len(p.stdout or ''), 'cmd': ' '.join(cmd)[:400]}
        results.append(rec)
        print('%-12s rc=%s tg=%s tok/s  vram_peak=%s MiB (+%s)  wall=%.0fs' % (
            arm['name'], p.returncode, rec['tg_tok_s'], rec['peak_vram_mib'], rec['vram_over_idle_mib'], dur), flush=True)
    by = {}
    for r in results:
        if r['tg_tok_s']:
            by.setdefault(r['arm'], []).append(r['tg_tok_s'])
    print('\nSUMMARY (mean of reps)')
    for k, v in by.items():
        print('  %-12s n=%d  mean %.2f  min %.2f  max %.2f' % (k, len(v), statistics.mean(v), min(v), max(v)))
    json.dump({'spec': spec, 'results': results, 'idle_vram_mib': base}, open(os.path.join(outdir, spec['session'] + '-verdict.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
