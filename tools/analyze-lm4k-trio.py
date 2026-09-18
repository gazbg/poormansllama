# Relocated from build/analyze-lm4k-trio.py (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
import json
from collections import Counter

def load(path):
    evs = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or '"event"' not in line:
                continue
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return evs

for label, path in [('bm1', 'build/llama-cuda/gplr25-lm4k-bm1.jsonl'),
                    ('lm',  'build/llama-cuda/gplr25-lm4k-lm.jsonl'),
                    ('bm2', 'build/llama-cuda/gplr25-lm4k-bm2.jsonl')]:
    evs = load(path)
    print(f"=== {label} ===")
    copies = [e for e in evs if e.get('event') == 'copy_submit']
    # bytes distribution of copy_submit
    bytes_hist = Counter(e.get('bytes') for e in copies)
    print("copy_submit bytes histogram (top):", bytes_hist.most_common(5))
    nonzero = [e for e in copies if e.get('bytes', 0) > 0]
    print(f"copy_submit nonzero-bytes: {len(nonzero)}, sum={sum(e['bytes'] for e in nonzero)/1e9:.2f} GB")
    # sample copy_submit keys
    if copies:
        print("copy_submit keys:", sorted(copies[0].keys()))
    # slot_evict
    se = [e for e in evs if e.get('event') == 'slot_evict']
    print(f"slot_evict: {len(se)}")
    # phase_summary events
    ps = [e for e in evs if e.get('event') == 'phase_summary']
    for p in ps:
        print("phase_summary:", {k: v for k, v in p.items() if k != 'event'})
    # layer_compute: count + sum if duration present
    lc = [e for e in evs if e.get('event') == 'layer_compute']
    if lc:
        print("layer_compute keys:", sorted(lc[0].keys()))
        for durk in ('compute_us', 't_us', 'dur_us', 'dt_us'):
            if durk in lc[0]:
                tot = sum(e[durk] for e in lc)
                print(f"layer_compute n={len(lc)} sum({durk})={tot/1e6:.2f} s")
                break
    print()
