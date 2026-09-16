#!/usr/bin/env python3
"""T-GPLR-35 analyzer: does staging on the compute context's aux stream
overlap the bank DMA with layer compute?

Reads the two A/B traces (gplr35-overlap0/overlap1.jsonl) and reports, per arm:
- matched submit->consumption spans per (layer, generation): the effective copy
  window as observed by the host (embeds WDDM submission latency and any
  waiting). Baseline median ~75 ms; an overlapped arm should show ~35-40 ms;
- how many windows exceed the 45 ms "stretched" threshold;
- per-pass sum(transfer_us) as a second lens (pure DMA rate is ~31.5 ms/bank);
- the phase_summary wall/tok fields textually, so the verdict numbers come from
  the trace itself rather than from a re-typed metric name.

Usage: python tools/analyze-gplr35.py overlap0 overlap1
Prediction/falsifier: plan T-GPLR-35 (warm wall 2.75-2.8 -> ~2.0-2.1 s;
falsifier: wall >= ~2.6 s with windows unchanged).
Limitation (stated, not hidden): windows are host-observed spans; the trace
carries no GPU-side timestamps, so an arm that overlaps but still waits on
consumption cannot be distinguished from one that does not overlap at all
except through the wall and the pass-level transfer sums.
"""
import json, statistics, sys
from collections import defaultdict, Counter

BASE = 'build/llama-cuda/'

def analyze(path, label):
    sub = {}          # (layer, gen) -> (submit_us, submit_batch)
    comp = {}         # (layer, gen) -> (complete_us, transfer_us)
    sb_counter = Counter()
    transfer_sum_by_pass = defaultdict(float)
    pass_bounds = [json.loads(l)['t_us'] for l in open(path) if '"pp_pass"' in l]
    phase_lines = [l.rstrip() for l in open(path) if '"phase_summary"' in l]
    for line in open(path):
        if '"copy_submit"' not in line and '"copy_complete"' not in line:
            continue
        e = json.loads(line)
        ev = e.get('event')
        if e.get('phase') != 0:
            continue
        if ev == 'copy_submit' and e.get('bytes', 0) > 0:
            sb = e.get('submit_batch', 0)
            sb_counter[sb] += 1
            sub[(e['layer'], e.get('generation', 0))] = (e['t_us'], sb)
        elif ev == 'copy_complete':
            comp[(e['layer'], e.get('generation', 0))] = (e['t_us'], e.get('transfer_us', 0.0))

    spans = []
    for k, (ts, sb) in sub.items():
        if k in comp:
            spans.append((ts, comp[k][0], sb))
            pid = sum(1 for b in pass_bounds if ts >= b)
            transfer_sum_by_pass[pid] += comp[k][1] / 1e3
    spans.sort()

    windows = [(e - s) / 1e3 for (s, e, _) in spans]
    print(f"== {label} ==")
    print(f"  nonzero phase-0 submits: {len(sub)}, matched spans: {len(spans)}")
    print(f"  submit_batch path counts: {dict(sorted(sb_counter.items()))}")
    if windows:
        ws = sorted(windows)
        print(f"  effective window ms: mean {statistics.mean(windows):.1f}, median {statistics.median(windows):.1f}, p90 {ws[int(len(ws)*0.9)]:.1f}, max {ws[-1]:.1f}")
        n_stretch = sum(1 for w in windows if w > 45.0)
        print(f"  windows > 45 ms (stretched): {n_stretch}/{len(windows)}")
    for pid in sorted(transfer_sum_by_pass):
        print(f"  pass {pid+1} sum(transfer_us): {transfer_sum_by_pass[pid]:.0f} ms")
    print("  phase summaries (from the trace):")
    for pl in phase_lines:
        e = json.loads(pl)
        keys = [k for k in e if any(t in k.lower() for t in ('wall', 'tok', 'pp_', 'us', 'ms'))]
        print("    " + ", ".join(f"{k}={e[k]}" for k in keys))
    print()

for name in sys.argv[1:]:
    analyze(BASE + f'gplr35-{name}.jsonl', name)
