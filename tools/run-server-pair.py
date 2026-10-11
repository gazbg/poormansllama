#!/usr/bin/env python3
r"""Paired llama-server runs for goal G-TG8: the owner's baseline came from llama-server, so measure it in that tool.

For every (arm, rep), interleaved: start llama-server with the arm's flags/env on 127.0.0.1, wait for /health, send ONE greedy
/completion request (temperature 0, ignore_eos, no prompt cache), read the server's own `timings`, poll whole-GPU memory.used
(50 ms, one long-lived nvidia-smi) for the server's whole life, then kill the server tree. It records; it claims nothing.

  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag <tag> -Command \
    "python tools/run-server-pair.py --spec <spec.json> --attest build\llama-cuda\bin\g3\llama-server.exe"

Spec (JSON): {"session": "g3-server", "exe": "build/llama-cuda/bin/g3/llama-server.exe", "model": "...gguf",
  "prompt_file": "...", "n_predict": 512, "reps": 2, "port": 8091, "common": [server flags shared by all arms],
  "arms": [{"name": "SRV", "flags": [...], "env": {...}, "why": "..."}, ...]}
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def poll_fast(stop, out, ms=50):
    peak, n = 0, 0
    p = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '--loop-ms=%d' % ms],
                         stdout=subprocess.PIPE, text=True)

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
    out['peak_mib'], out['samples'] = peak, n


def http(url, data=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8', 'replace')


def one_run(spec, arm, rep, sdir):
    port = int(spec.get('port', 8091))
    exe = os.path.join(ROOT, spec['exe'])
    stem = '%s_r%d' % (arm['name'], rep)
    cmd = [exe, '-m', spec['model'], '--host', '127.0.0.1', '--port', str(port), '--no-webui'] + spec.get('common', []) + arm.get('flags', [])
    env = dict(os.environ)
    env.update({k: str(v).replace('{sdir}', sdir).replace('{name}', stem) for k, v in arm.get('env', {}).items()})
    logf = open(os.path.join(sdir, stem + '.log'), 'w', encoding='utf-8', errors='replace')
    stop = threading.Event()
    vr = {}
    th = threading.Thread(target=poll_fast, args=(stop, vr), daemon=True)
    th.start()
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    rec = {'arm': arm['name'], 'rep': rep, 'cmd': ' '.join(cmd), 'why': arm.get('why', '')}
    try:
        ready = False
        while time.time() - t0 < 240:
            if srv.poll() is not None:
                rec['error'] = 'server exited early rc=%s' % srv.returncode
                break
            try:
                st, _ = http('http://127.0.0.1:%d/health' % port, timeout=2)
                if st == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        rec['startup_s'] = round(time.time() - t0, 1)
        if ready:
            prompt = open(os.path.join(ROOT, spec['prompt_file']), encoding='utf-8').read()
            body = {'prompt': prompt, 'n_predict': int(spec.get('n_predict', 512)), 'temperature': 0.0, 'top_k': 1,
                    'ignore_eos': True, 'cache_prompt': False, 'stream': False}
            rec['requests'] = []
            for qi in range(int(spec.get('requests', 1))):
                st, txt = http('http://127.0.0.1:%d/completion' % port, body, timeout=900)
                rec['http'] = st
                js = json.loads(txt)
                t = js.get('timings', {})
                one = {k: t.get(k) for k in ('prompt_n', 'prompt_per_second', 'predicted_n', 'predicted_ms', 'predicted_per_second', 'draft_n', 'draft_n_accepted')}
                rec['requests'].append(one)
                if qi == 0:
                    rec['content_head'] = (js.get('content') or '')[:120]
                with open(os.path.join(sdir, '%s_q%d.txt' % (stem, qi + 1)), 'w', encoding='utf-8') as f:
                    f.write(js.get('content') or '')
            rec['timings'] = rec['requests'][-1]      # the last request is the warm one
            rec['timings_first'] = rec['requests'][0]  # the first request pays the cold start
    except Exception as ex:
        rec['error'] = repr(ex)
    finally:
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass
        time.sleep(1.0)
        stop.set()
        th.join(timeout=10)
        logf.close()
    rec['vram_peak_mib'] = vr.get('peak_mib')
    rec['vram_samples'] = vr.get('samples')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--attest', nargs='*', default=[], help='exe paths named only so the wrapper can attest them')
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding='utf-8'))
    sdir = os.path.join(ROOT, 'build', 'llama-cuda', spec['session'])
    os.makedirs(sdir, exist_ok=True)
    runs = []
    for r in range(1, int(spec.get('reps', 2)) + 1):
        for arm in spec['arms']:
            rec = one_run(spec, arm, r, sdir)
            runs.append(rec)
            tm = rec.get('timings') or {}
            tf = rec.get('timings_first') or {}
            print('RUN %-8s rep %d  TG first %s -> last %s tok/s  (PP last %s)  peak %s MiB  startup %ss  %s' % (
                arm['name'], r, tf.get('predicted_per_second'), tm.get('predicted_per_second'), tm.get('prompt_per_second'),
                rec.get('vram_peak_mib'), rec.get('startup_s'), rec.get('error', '')), flush=True)
    agg, agg_first = {}, {}
    for rec in runs:
        v = (rec.get('timings') or {}).get('predicted_per_second')
        if isinstance(v, (int, float)):
            agg.setdefault(rec['arm'], []).append(v)
        v = (rec.get('timings_first') or {}).get('predicted_per_second')
        if isinstance(v, (int, float)):
            agg_first.setdefault(rec['arm'], []).append(v)
    summary = {k: {'n': len(v), 'mean': statistics.mean(v), 'min': min(v), 'max': max(v),
                   'first_mean': statistics.mean(agg_first[k]) if k in agg_first else None} for k, v in agg.items()}
    out = {'session': spec['session'], 'spec': spec, 'attest': a.attest, 'runs': runs, 'summary': summary}
    with open(os.path.join(sdir, spec['session'] + '-verdict.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print()
    for k, v in summary.items():
        print('%-8s n=%d  LAST request mean %.2f  min %.2f  max %.2f tok/s   (FIRST request mean %s)' % (
            k, v['n'], v['mean'], v['min'], v['max'], '%.2f' % v['first_mean'] if v['first_mean'] else '-'))
    print('verdict:', os.path.join(sdir, spec['session'] + '-verdict.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
