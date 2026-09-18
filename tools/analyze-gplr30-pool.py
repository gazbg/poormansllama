# Relocated from build/analyze_gplr30_pool.py (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
import json
from collections import Counter

base = 'build/llama-cuda/'
arms = ['gplr30-lru4', 'gplr30-fixed4', 'gplr30-fixed10', 'gplr30-fixed20', 'gplr30-fixed40']

for arm in arms:
    pool_init = None
    passes = []          # pp_pass boundaries (t_us order)
    ev_by_pass = []      # per-pass list of pool events
    cur = 0
    ev_by_pass.append([])
    # first read: collect boundaries + pool events in time order
    events = []
    with open(base + arm + '.jsonl') as f:
        for line in f:
            if '"pool_' not in line and '"pp_pass"' not in line:
                continue
            e = json.loads(line)
            events.append(e)
    # pp_pass fires at the START of each repeat pass (first is pass 1)
    pass_starts = [e for e in events if e.get('event') == 'pp_pass']
    # build pass windows from phase_summary phase-0 t_us? simpler: pool events
    # before the first pp_pass belong to pass 1's preload; between boundary k
    # and k+1 belong to pass k+1. Boundaries in t order:
    boundaries = [e['t_us'] for e in pass_starts]
    per_pass = [[] for _ in range(len(boundaries) + 1)]
    pi = 0
    for e in events:
        if e.get('event') == 'pp_pass':
            pi += 1
            continue
        per_pass[pi].append(e)
    pool_init = next((e for e in events if e.get('event') == 'pool_init'), None)
    hits = sum(1 for e in events if e.get('event') == 'pool_event' and e.get('hit'))
    misses = sum(1 for e in events if e.get('event') == 'pool_event' and not e.get('hit'))
    hit_bytes = sum(e.get('bytes', 0) for e in events if e.get('event') == 'pool_event' and e.get('hit'))
    print(f"== {arm}")
    if pool_init:
        print(f"  pool_init: n={pool_init['n']} policy={pool_init['policy']} bytes={pool_init['bytes']/2**20:.0f} MiB populated={pool_init.get('populated_bytes',0)/2**20:.0f} MiB populate_us={pool_init['populate_us']:.0f}")
    print(f"  totals: hits={hits} misses={misses} hit_bytes={hit_bytes/2**20:.0f} MiB")
    for i, evs in enumerate(per_pass):
        h = sum(1 for e in evs if e.get('hit'))
        m = sum(1 for e in evs if not e.get('hit'))
        print(f"  pass {i+1}: hits={h} misses={m}")
    print()
