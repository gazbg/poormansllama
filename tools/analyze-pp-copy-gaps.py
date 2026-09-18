# Relocated from build/analyze_pp_copy_gaps.py (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
import json, statistics, sys

def summarize(path, label):
    submits, completes = [], []
    run_start = None
    for line in open(path):
        if '"copy_submit"' in line:
            e = json.loads(line)
            if e.get('phase') == 0:
                submits.append((e['layer'], e['bytes'], e['t_us']))
        elif '"copy_complete"' in line:
            e = json.loads(line)
            if e.get('phase') == 0:
                completes.append((e['layer'], e['host_submit_us'], e['ready_latency_us'], e['t_us']))
        elif '"run_start"' in line and run_start is None:
            run_start = json.loads(line)
    src = run_start.get('copy_source') if run_start else '?'
    print('===', label, '| copy_source:', src, '| copies:', len(submits))
    if not submits or not completes:
        print('  (no PP copy events)')
        return
    print('  bytes/layer:', submits[0][1])
    gaps = [submits[i+1][2] - submits[i][2] for i in range(len(submits)-1)]
    if gaps:
        print('  submit inter-arrival: mean %.1f median %.1f max %.1f us | occupancy %.2f s' %
              (statistics.mean(gaps), statistics.median(gaps), max(gaps), sum(gaps)/1e6))
    hs = [r[1] for r in completes]
    rl = [r[2] for r in completes]
    print('  host_submit: mean %.1f median %.1f max %.1f us | sum %.2f s' %
          (statistics.mean(hs), statistics.median(hs), max(hs), sum(hs)/1e6))
    print('  ready_lat:   mean %.1f median %.1f max %.1f us | sum %.2f s' %
          (statistics.mean(rl), statistics.median(rl), max(rl), sum(rl)/1e6))
    # staging bandwidth implied by host_submit: bytes / host_submit
    b = submits[0][1]
    eff = b / statistics.median(hs) / 1e3  # MB/s (median host_submit us)
    print('  implied host-feed bandwidth from median host_submit: %.1f MB/s' % eff)

for p, l in [a.split('::') for a in sys.argv[1:]]:
    try:
        summarize(p, l)
    except FileNotFoundError:
        print('===', l, 'MISSING', p)
