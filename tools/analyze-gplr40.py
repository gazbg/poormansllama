#!/usr/bin/env python3
"""T-GPLR-40: summarize a bisect arm's trace — passes completed, last chunk/layer
seen, crash position. Failure isolation only.

Usage: python tools/analyze-gplr40.py <trace.jsonl> [more ...]
"""
import json
import sys

for path in sys.argv[1:]:
    try:
        evs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        print('  (no trace)', path)
        continue
    pps = [e for e in evs if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    lc = [e for e in evs if e.get('event') == 'layer_compute']
    last = lc[-1] if lc else None
    if last:
        pos = f"last launch: chunk {last.get('chunk')} layer {last.get('layer')} slot {last.get('slot')} gen {last.get('generation')}"
    else:
        pos = 'no layer launches'
    print(f"  pp summaries: {len(pps)} ({'survived both passes' if len(pps) >= 2 else 'crashed/incomplete'}) | {pos}")
    for e in pps:
        print(f"    pass wall {e['wall_us']/1e3:.1f} ms  {e.get('tokens_per_s', 0):.1f} tok/s")
