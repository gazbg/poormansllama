#!/usr/bin/env python3
"""T-GPLR-37 analyzer: pinned hop vs direct mmap-sourced staging (both variants).

Per arm (last pp summary = warm pass): wall / compute / wait / tok/s, the PP-only
per-bank host-submit window (the hop), the recon's pure bank DMA window, the host
ledger buckets, and the registration evidence:
  * `mmap-register` events + total bytes + their `mode` field
  * `mmap-register-init`: the pre-timed registration cost and total bytes
  * churn check: how many registrations happen BEFORE the first layer_compute —
    the churn-free variant must register everything up front (init-once), while
    the churn variant registers during the pass.

Pairing is derived from the file names: for every candidate ('reg'/'ron') the
control containing 'pin' with the same trailing slot token (sN) is used.

Usage:
  python tools/analyze-gplr37.py build/llama-cuda/gplr37b-{pin,ron}-s{2,3}.jsonl
"""
import json
import re
import statistics
import sys

PURE_DMA_MS = 31.37


def load(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def arm(path):
    ev = load(path)
    rs = [e for e in ev if e.get('event') == 'run_start']
    pps = [e for e in ev if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    cc = [e for e in ev if e.get('event') == 'copy_complete' and e.get('phase') == 0]
    ra = [e for e in ev if e.get('event') == 'recon_attr']
    regs = [e for e in ev if e.get('event') == 'mmap_register']
    reginit = [e for e in ev if e.get('event') == 'mmap_register_init']
    lc = [e for e in ev if e.get('event') == 'layer_compute']
    leds = [e for e in ev if e.get('event') == 'host_ledger']
    print('=' * 30, path.split('/')[-1])
    if not pps:
        print('  NO pp summary (arm failed?)')
        return None
    r = rs[0] if rs else {}
    print('  copy-source=%s slots=%s schedule=%s chunks=%s' % (
        r.get('copy_source'), r.get('n_slots'), r.get('schedule'), r.get('pp_n_chunks')))
    out = {'copy': r.get('copy_source'), 'slots': r.get('n_slots'), 'path': path}
    for i, s in enumerate(pps, start=1):
        tag = 'warm' if i == len(pps) else 'cold'
        print('  pass %d (%s): wall %.1f ms | compute %.1f | wait %.1f | transfer %.1f | %.1f tok/s'
              % (i, tag, s['wall_us'] / 1e3, s['total_compute_us'] / 1e3, s['total_wait_us'] / 1e3,
                 s['total_transfer_us'] / 1e3, s['tokens_per_s']))
        if tag == 'warm':
            out['wall_ms'] = s['wall_us'] / 1e3
            out['toks'] = s['tokens_per_s']
            out['compute_ms'] = s['total_compute_us'] / 1e3
            out['wait_ms'] = s['total_wait_us'] / 1e3
    nbank = len(cc) // max(1, len(pps))
    for i, sm in enumerate(pps, start=1):
        chunk = cc[(i - 1) * nbank: i * nbank]
        if not chunk:
            continue
        hs = [e['host_submit_us'] for e in chunk]
        print('   pp bank hop pass %d: median %.2f ms sum %.1f ms (%d banks)'
              % (i, statistics.median(hs) / 1e3, sum(hs) / 1e3, len(chunk)))
        if i == len(pps):
            out['hop_sum_ms'] = sum(hs) / 1e3
            out['hop_med_ms'] = statistics.median(hs) / 1e3
    gpu = [e['copy_gpu_ms'] for e in ra if 'copy_gpu_ms' in e]
    if gpu:
        print('  recon pure bank DMA window: median %.2f ms (%d samples, ceiling %.2f)'
              % (statistics.median(gpu), len(gpu), PURE_DMA_MS))
        out['dma_med_ms'] = statistics.median(gpu)
    if regs:
        modes = sorted({e.get('mode', 'per-bank') for e in regs})
        if lc:
            first_lc = min(e['t_us'] for e in lc)
            before = sum(1 for e in regs if e['t_us'] < first_lc)
        else:
            before = 0
        print('  mmap_register: %d events, %s, %.2f GiB cumulative; %d/%d fire BEFORE the first '
              'layer_compute (init-once = all, churn = ~1)' % (
                  len(regs), modes, sum(e['bytes'] for e in regs) / 2**30, before, len(regs)))
        out['regs'] = len(regs)
        out['reg_mode'] = ','.join(modes)
        out['reg_before'] = before
    else:
        out['regs'] = 0
        out['reg_mode'] = 'none'
    if reginit:
        ri = reginit[-1]
        print('  mmap_register_init: %.2f GiB pinned in %.1f ms (pre-timed setup)'
              % (ri['bytes'] / 2**30, ri['us'] / 1e3))
        out['reg_gib'] = ri['bytes'] / 2**30
        out['reg_setup_ms'] = ri['us'] / 1e3
    if leds:
        h = leds[-1]
        print('  ledger (warm): sweep %.1f = compute %.1f + patch %.1f + other %.1f + slot %.1f '
              '+ setup %.1f + graph_call %.1f ms'
              % (h['sweep_us'] / 1e3, h['compute_us'] / 1e3, h['patch_us'] / 1e3, h['other_us'] / 1e3,
                 h['slot_us'] / 1e3, h['setup_us'] / 1e3, h['graph_call_us'] / 1e3))
        out['setup_ms'] = h['setup_us'] / 1e3
        out['slot_ms'] = h['slot_us'] / 1e3
        out['compute_led_ms'] = h['compute_us'] / 1e3
    return out


def main(paths):
    res = {}
    for p in paths:
        res[p.split('/')[-1]] = arm(p)
    print('\n--- paired verdicts (warm pass) ---')
    for name, b in res.items():
        if not b or not re.search(r'(reg|ron)', name):
            continue
        tok = re.search(r'(s\d+)', name)
        ctrl_name = next((n for n in res if 'pin' in n and tok and tok.group(1) in n), None)
        a = res.get(ctrl_name)
        if not a:
            print('  %s: no pinned control found' % name)
            continue
        dw = b['wall_ms'] - a['wall_ms']
        print('  %s vs %s: wall %.1f -> %.1f ms (%+.1f ms, %+.2f%%), tok/s %.1f -> %.1f (%+.2f%%)'
              % (name, ctrl_name, a['wall_ms'], b['wall_ms'], dw, 100.0 * dw / a['wall_ms'],
                 a['toks'], b['toks'], 100.0 * (b['toks'] - a['toks']) / a['toks']))
        print('     hop %.2f -> %.2f ms/bank | setup %.1f -> %.1f ms | slot %.1f -> %.1f ms | '
              'DMA %.2f -> %.2f ms | regs %d (%s), %s GiB, pre-timed %s ms'
              % (a.get('hop_med_ms', float('nan')), b.get('hop_med_ms', float('nan')),
                 a.get('setup_ms', float('nan')), b.get('setup_ms', float('nan')),
                 a.get('slot_ms', float('nan')), b.get('slot_ms', float('nan')),
                 a.get('dma_med_ms', float('nan')), b.get('dma_med_ms', float('nan')),
                 b.get('regs', 0), b.get('reg_mode', 'none'),
                 ('%.2f' % b['reg_gib']) if 'reg_gib' in b else '-',
                 ('%.1f' % b['reg_setup_ms']) if 'reg_setup_ms' in b else '-'))


if __name__ == '__main__':
    main(sys.argv[1:])
