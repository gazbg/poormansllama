#!/usr/bin/env python3
"""T-GPLR-32 analyzer: per-copy effective windows and DMA concurrency.

Reads the two A/B traces (gplr32-ser0/ser1.jsonl) and reports, per arm:
- matched submit->consumption spans per (layer, generation) — the effective
  window as observed by the host (the trace carries no GPU-side timestamps,
  so the span embeds WDDM submission latency by construction);
- peak concurrent in-flight DMAs (any source) by span overlap;
- the af_inflight telemetry maximum (worker DMAs submitted but unconsumed);
- how many windows exceed 45 ms (the "stretched" threshold; pure-rate is
  ~31 ms for the 453 MiB bank at the measured 14.4 GB/s link).

Usage: python tools/analyze-gplr32.py ser0 ser1
Verdict context: docs/currentstate.md 2026-09-16 T-GPLR-32 entry and the
plan task — walls identical, stretch intrinsic to the per-copy window.
"""
import json, statistics, sys
from collections import defaultdict

BASE = 'build/llama-cuda/'

def analyze(path, label):
    sub = {}          # (layer, gen) -> (submit_us, af_inflight) phase-0 nonzero copies
    comp = {}         # (layer, gen) -> complete_us
    max_inflight = 0
    for line in open(path):
        if '"copy_submit"' not in line and '"copy_complete"' not in line:
            continue
        e = json.loads(line)
        ev = e.get('event')
        if e.get('phase') != 0:
            continue
        if ev == 'copy_submit' and e.get('bytes', 0) > 0:
            max_inflight = max(max_inflight, e.get('af_inflight', 0))
            sub[(e['layer'], e.get('generation', 0))] = (e['t_us'], e.get('af_inflight', 0))
        elif ev == 'copy_complete':
            comp[(e['layer'], e.get('generation', 0))] = e['t_us']

    spans = []   # (submit_us, complete_us, af_inflight)
    for k, (ts, infl) in sub.items():
        if k in comp:
            spans.append((ts, comp[k], infl))
    spans.sort()

    windows = [(e - s) / 1e3 for (s, e, _) in spans]
    events = []
    for (s, e, _) in spans:
        events.append((s, 1)); events.append((e, -1))
    events.sort()
    cur = peak = 0
    for (_, d) in events:
        cur += d
        peak = max(peak, cur)

    print(f"== {label} ==")
    print(f"  nonzero phase-0 submits: {len(sub)}, matched spans: {len(spans)}")
    print(f"  max af_inflight field: {max_inflight}")
    print(f"  peak concurrent in-flight DMAs (any source): {peak}")
    if windows:
        ws = sorted(windows)
        print(f"  effective window ms: mean {statistics.mean(windows):.1f}, median {statistics.median(windows):.1f}, p90 {ws[int(len(ws)*0.9)]:.1f}, max {ws[-1]:.1f}")
        n_stretch = sum(1 for w in windows if w > 45.0)
        print(f"  windows > 45 ms (stretched): {n_stretch}/{len(windows)}")
    print()

for name in sys.argv[1:]:
    analyze(BASE + f'gplr32-{name}.jsonl', name)
