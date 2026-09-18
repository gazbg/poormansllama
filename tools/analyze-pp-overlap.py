# Relocated from build/analyze_pp_overlap.py (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
import json, statistics, sys

def per_layer(path, label):
    events = []
    run_start = None
    for line in open(path):
        if '"copy_submit"' in line:
            e = json.loads(line)
            if e.get('phase') == 0:
                events.append(('sub', e['layer'], e['t_us'], e.get('bytes', 0)))
        elif '"copy_complete"' in line:
            e = json.loads(line)
            if e.get('phase') == 0:
                events.append(('cpy', e['layer'], e['t_us'], e['host_submit_us'], e['ready_latency_us']))
        elif '"layer_compute"' in line:
            e = json.loads(line)
            if e.get('phase') == 0 and 'compute_us' in e:
                events.append(('cmp', e['layer'], e['t_us'], e['compute_us']))
        elif '"run_start"' in line and run_start is None:
            run_start = json.loads(line)
    events.sort(key=lambda x: x[2])
    ns = run_start.get('n_slots') if run_start else '?'
    sch = run_start.get('schedule') if run_start else '?'
    src = run_start.get('copy_source') if run_start else '?'
    first = {}
    for ev in events:
        if ev[0] == 'sub' and ev[1] not in first:
            first[ev[1]] = ev[2]
    overlap = 0
    cmps = [ev for ev in events if ev[0] == 'cmp']
    for ev in cmps:
        nxt = ev[1] + 1
        if nxt in first and first[nxt] <= ev[2]:
            overlap += 1
    subs = [ev for ev in events if ev[0] == 'sub']
    gaps = [subs[i+1][2]-subs[i][2] for i in range(len(subs)-1)] if len(subs) > 1 else []
    cmp_us = [ev[3] for ev in cmps]
    print('=== %s | slots=%s sched=%s src=%s' % (label, ns, sch, src))
    print('  copy events %d | next-copy-submitted-at-compute-start %d/%d = %.0f%%' %
          (len(subs), overlap, len(cmps), 100.0*overlap/max(len(cmps),1)))
    if cmp_us:
        print('  compute mean %.2f ms median %.2f ms sum %.2f s' %
              (statistics.mean(cmp_us)/1e3, statistics.median(cmp_us)/1e3, sum(cmp_us)/1e6))
    if gaps:
        print('  submit inter-arrival mean %.1f ms median %.1f ms occupancy %.2f s' %
              (statistics.mean(gaps)/1e3, statistics.median(gaps)/1e3, sum(gaps)/1e6))
    hs = [ev[3] for ev in events if ev[0] == 'cpy']
    rl = [ev[4] for ev in events if ev[0] == 'cpy']
    if hs:
        print('  host_submit mean %.1f ms | ready_lat mean %.2f ms | sums %.2f/%.2f s' %
              (statistics.mean(hs)/1e3, statistics.mean(rl)/1e3, sum(hs)/1e6, sum(rl)/1e6))

per_layer(sys.argv[1], sys.argv[1] if len(sys.argv) < 3 else sys.argv[2])
