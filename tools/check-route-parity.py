#!/usr/bin/env python3
"""Route PASS-PARITY gate (T-GPLR-48 evidence; also the cheap pass-2 gate
T-GPLR-47 can reuse).

Under `--pp-repeat 2` the same prompt is processed twice, so every sampled
(chunk, layer) is routed once per pass and a CLEAN run emits each sampled route
tuple EXACTLY TWICE. Route telemetry is emitted by the sampler
(`telemetry_chunk_stride`, default every 10th chunk), so the property is
checkable from a trace alone: no decode, no answer text, no reference stream.

Why this exists: the 4k TG answer->echo defect (T-GPLR-46/49b) needs the pw
schedule AND the per-pass rebuild together, and at 50k the answer cannot be
graded at all (only 3 TG ids are legal there). Parity is measurable at every
length, and it localizes the damage (which chunk, which layer) instead of
reporting a stream class.

MEASURED (2026-09-19, T-GPLR-49b, 4k/96, one binary, pw schedule):
  rebuild ON  -> parity BROKEN for every sampled chunk except the first
  rebuild OFF -> parity CLEAN
So parity tracks the rebuild, which is what T-GPLR-48's default change acts on.

Usage:
  python tools/check-route-parity.py TRACE [TRACE ...] [--json OUT] [--quiet]

Exit codes: 0 = every trace parity-clean; 1 = at least one diverged (the
diverging (chunk, layer) pairs are printed); 2 = a trace has no route events
(nothing to evaluate -- NOT a pass).

T-GPLR-39 (owner decision 2026-09-20): route telemetry DEFAULTS OFF, so a
defaulted run's trace exits 2 by design. The parity gate requires an explicit
--route-telemetry 1 run; exit 2 on such a trace still means the run died
before routing (see the AGENTS.md route-parity note).
"""
import argparse
import collections
import json
import os
import sys

# Identity fields of a route sample. `t_us` and `run_id` are deliberately
# excluded: they differ between the two passes by construction.
KEY_FIELDS = ('chunk', 'layer', 'top_k', 'n_tokens', 'route_union', 'hash', 'sampled_ids')


def route_keys(path):
    """Every route tuple in the trace, in order."""
    keys = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if '"event":"route"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get('event') != 'route':
                continue
            try:
                keys.append(tuple(e.get(f) for f in KEY_FIELDS))
            except TypeError:
                continue
    return keys


def check(path):
    keys = route_keys(path)
    counts = collections.Counter(keys)
    clean = [k for k, n in counts.items() if n == 2]
    once = [k for k, n in counts.items() if n == 1]
    over = [(k, n) for k, n in counts.items() if n > 2]
    return {
        'trace': path,
        'route_events': len(keys),
        'distinct': len(counts),
        'paired_exactly_twice': len(clean),
        'singletons': len(once),
        'over_counted': len(over),
        'parity': 'clean' if (counts and not once and not over) else 'DIVERGED',
        'diverging_pairs': [
            {'chunk': k[0], 'layer': k[1], 'count': 1} for k in sorted(once, key=lambda k: (k[0] or 0, k[1] or 0))
        ] + [
            {'chunk': k[0], 'layer': k[1], 'count': n} for k, n in sorted(over, key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0))
        ],
    }


def print_report(r, quiet):
    if quiet:
        return
    print(f"{r['trace']}")
    print(f"  route events {r['route_events']}  distinct {r['distinct']}  "
          f"paired-x2 {r['paired_exactly_twice']}  singletons {r['singletons']}  "
          f"over-counted {r['over_counted']}  -> {r['parity']}")
    if r['diverging_pairs']:
        # Group by chunk, then by layer: a layer whose two passes disagreed
        # contributes TWO singleton tuples (one per pass value), so dedupe the
        # layers or the count reads 2x the number of damaged (chunk, layer)
        # samples.
        by_chunk = collections.defaultdict(set)
        counts = collections.Counter()
        for d in r['diverging_pairs']:
            by_chunk[d['chunk']].add(d['layer'])
            counts[d['chunk']] += 1
        for chunk in sorted(c for c in by_chunk if c is not None):
            layers = sorted(l for l in by_chunk[chunk] if l is not None)
            shown = ','.join(str(l) for l in layers[:12]) + (',...' if len(layers) > 12 else '')
            print(f"    chunk {chunk}: {len(layers)} diverging layer(s) [{shown}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--json', dest='json_out', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    missing = [p for p in args.traces if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"MISSING TRACE: {p}", file=sys.stderr)
        return 2

    reports = [check(p) for p in args.traces]
    for r in reports:
        print_report(r, args.quiet)

    empty = [r for r in reports if r['route_events'] == 0]
    diverged = [r for r in reports if r['parity'] != 'clean']

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8', newline='\n') as fh:
            json.dump({'reports': reports,
                       'verdict': 'PASS' if not diverged and not empty else 'FAIL'}, fh, indent=2)
            fh.write('\n')

    if empty:
        for r in empty:
            print(f"NO ROUTE EVENTS in {r['trace']} -- nothing was evaluated (not a pass)", file=sys.stderr)
        return 2
    if diverged:
        for r in diverged:
            print(f"ROUTE PARITY DIVERGED: {r['trace']} ({r['singletons']} singleton, "
                  f"{r['over_counted']} over-counted)", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"ROUTE PARITY OK: {len(reports)} trace(s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
