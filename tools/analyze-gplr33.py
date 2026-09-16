#!/usr/bin/env python3
"""T-GPLR-33 analyzer: does member-batched staging shrink the window?

Reads the two A/B traces (gplr33-batch0/batch1.jsonl) and reports, per arm:
- submit_batch path distribution from copy_submit (0 = legacy per-member,
  1 = single-span V1 copy, 2 = mixed span+per-member degradation) — since
  the flag-vs-branch fix this reports the ACTUALLY-TAKEN branch, so the
  path distribution is self-verifying evidence, not flag echo;
- matched submit->consumption spans per (layer, generation) — the effective
  window as observed by the host (embeds WDDM submission latency);
- how many windows exceed the 45 ms "stretched" threshold;
- per-pass transfer sums (from copy_complete transfer_us) as a second lens.

Usage: python tools/analyze-gplr33.py batch0 batch1
Prediction context: plan T-GPLR-33 (window 91 -> ~44-58 ms; falsifier
window >= ~80 ms). Trace-level verdict lands in docs/currentstate.md.
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

    spans = []   # (submit_us, complete_us, submit_batch)
    for k, (ts, sb) in sub.items():
        if k in comp:
            spans.append((ts, comp[k][0], sb))
            # assign this bank's transfer to the pass containing its submit
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
    print()

for name in sys.argv[1:]:
    analyze(BASE + f'gplr33-{name}.jsonl', name)
