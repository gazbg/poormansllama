#!/usr/bin/env python3
"""T-GPLR-36 follow-up analyzer: where does the process's host memory go?

Reads a trace that carries `host_mem` lines (emitted under --pp-recon, twice per
pass: pass_begin / pass_end) and answers the question the instrument was built
for -- the owner's observed host-RSS climb across a run (~19 GB -> ~30 GB) had no
instrument at all, while every byte figure in the trace was VRAM-side.

The attribution axis is the SAMPLE ITSELF, not a model:

  ws   = working set (resident pages)
  priv = this process's own anonymous resident pages
  file = ws - priv  = file-backed/shared share (the mapped model, DLLs)

so growth in `priv` is the runner's own allocations and growth in `file` is the
model mapping being paged in or pinned. `eng` (the engine's own accounted host
holdings: per-slot pinned staging + persistent pool + registered mmap spans) is
then compared against the anonymous growth: whatever `priv` grew by beyond `eng`
is NOT the residency machinery.

Exit 2 when the trace has no host_mem line (nothing to evaluate).
Exit 1 when a sample declares reader_ok false (the counters did not read: the
numbers are not evidence).

Usage:
  python tools/analyze-gplr36-hostmem.py build/llama-cuda/gplr36-hostmem-200k.jsonl
"""
import json
import sys

MIB = 1024.0 * 1024.0


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def mib(v):
    return (v or 0) / MIB


def main(paths):
    rc = 0
    for path in paths:
        ev = load(path)
        rs = [e for e in ev if e.get('event') == 'run_start']
        ba = [e for e in ev if e.get('event') == 'byte_account']
        hm = [e for e in ev if e.get('event') == 'host_mem']
        pp = [e for e in ev if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
        print('=' * 30, path)
        if rs:
            r = rs[0]
            print(' shape: %s tokens | %s chunks x %s layers | %s slots | schedule %s | '
                  'kv %s | copy %s | repeat passes=%d' % (
                      r.get('pp_tokens'), r.get('pp_n_chunks'), r.get('n_layer'), r.get('n_slots'),
                      r.get('schedule'), r.get('kv_cache'), r.get('copy_source'), len(pp)))
            print(' vram at run_start: static %s MiB | slot window %s MiB | free %s MiB of %s MiB' % (
                round(mib(r.get('static_vram_bytes')), 1), round(mib(r.get('slot_vram_bytes')), 1),
                round(mib(r.get('free_vram_bytes')), 1), round(mib(r.get('total_vram_bytes')), 1)))
        if ba:
            b = ba[0]
            print(' byte_account: kv %s MiB | act_slab %s MiB | slot_window %s MiB | static %s MiB | '
                  'free_vram %s MiB (bounded+row_cache+slab+reserve within free: %s)' % (
                      round(mib(b.get('kv')), 1), round(mib(b.get('act_slab')), 1),
                      round(mib(b.get('slot_window')), 1), round(mib(b.get('static')), 1),
                      round(mib(b.get('free_vram')), 1), b.get('bounded_within_free')))
        if not hm:
            print(' CANNOT EVALUATE: no host_mem line (needs --pp-recon 1 on a pass executed by '
                  'llama_context::layer_major_pp)')
            rc = max(rc, 2)
            continue

        print('\n host_mem samples: %d (pass_begin + pass_end per pass)' % len(hm))
        print('   %-4s %-11s %9s %9s %9s %9s %9s %9s %9s' % (
            'pass', 'label', 'ws', 'priv', 'file', 'commit', 'd_ws', 'g_ws', 'eng'))
        for e in hm:
            if not e.get('reader_ok'):
                print('   sample pass %s (%s): reader_ok=false -- the process counters did not read; '
                      'treating the trace as unevaluable' % (e.get('pass'), e.get('label')))
                rc = max(rc, 1)
            print('   %-4s %-11s %9s %9s %9s %9s %9s %9s %9s' % (
                e.get('pass'), e.get('label'),
                round(mib(e.get('ws_bytes')), 1), round(mib(e.get('ws_private_bytes')), 1),
                round(mib(e.get('ws_file_bytes')), 1), round(mib(e.get('commit_bytes')), 1),
                round(mib(e.get('d_ws_bytes')), 1), round(mib(e.get('g_ws_bytes')), 1),
                round(mib(e.get('eng_bytes')), 1)))
            if not e.get('ws_private_known'):
                print('      note: ws_private_known=false -- the private/file split is NOT available '
                      'in this build/OS, so `priv`/`file` above are unusable')

        first, last = hm[0], hm[-1]
        d_ws = last.get('ws_bytes', 0) - first.get('ws_bytes', 0)
        d_priv = last.get('ws_private_bytes', 0) - first.get('ws_private_bytes', 0)
        d_file = last.get('ws_file_bytes', 0) - first.get('ws_file_bytes', 0)
        d_commit = last.get('commit_bytes', 0) - first.get('commit_bytes', 0)
        d_eng = last.get('eng_bytes', 0) - first.get('eng_bytes', 0)
        known = all(e.get('ws_private_known') for e in hm)

        print('\n CLIMB: first sample (pass %s %s) -> last sample (pass %s %s)' % (
            first.get('pass'), first.get('label'), last.get('pass'), last.get('label')))
        print('   ws      %+9.1f MiB | private (anonymous) %+9.1f MiB | file-backed/shared %+9.1f MiB'
              % (mib(d_ws), mib(d_priv), mib(d_file)))
        print('   commit  %+9.1f MiB | engine-accounted host %+9.1f MiB' % (mib(d_commit), mib(d_eng)))
        print('   peak seen by this run %.1f MiB | OS lifetime peak %.1f MiB | sys free now %.1f MiB '
              'of %.1f MiB' % (mib(last.get('ws_peak_seen_bytes')), mib(last.get('ws_peak_os_bytes')),
                               mib(last.get('sys_free_bytes')), mib(last.get('sys_total_bytes'))))
        if known and d_ws:
            print('   attribution: private %.1f%% of the climb, file-backed/shared %.1f%%' % (
                100.0 * d_priv / d_ws, 100.0 * d_file / d_ws))
            resid = d_priv - d_eng
            share = 100.0 * resid / d_priv if d_priv else float('nan')
            if resid > 0:
                print('   the engine ACCOUNTS for %.1f MiB of the %+.1f MiB anonymous growth; '
                      '%+.1f MiB (%.1f%%) is allocated outside its own bookkeeping' % (
                          mib(d_eng), mib(d_priv), mib(resid), share))
            else:
                print('   the engine accounts for ALL of the anonymous growth (%.1f MiB measured '
                      'against %+.1f MiB accounted)' % (mib(d_eng), mib(d_priv)))
        elif not known:
            print('   attribution: NOT AVAILABLE -- no private/file split on this platform')
        else:
            print('   attribution: no measurable climb between the first and last sample')
        print('   per-pass deltas (d_ws) are the pass-to-pass signal; a climb that is flat in '
              'd_ws but large in g_ws happened before the timed passes began')
    return rc


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
