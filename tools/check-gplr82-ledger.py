#!/usr/bin/env python3
"""T-GPLR-82 N1 checker: does the routed-expert demand ledger close?

WHAT THIS CHECKS. `llama_layer_streaming::tel_layer_compute`
(src/llama-layer-streaming.cpp) emits an OPTIONAL row-set block on every
`layer_compute` record, gated on `attn_compute_us >= 0.0` so prefill records
keep their byte-identical short form. The block carries three integers:

    "experts_total":%d,"experts_host":%d,"experts_device":%d

`experts_total` is what the router demanded for that (layer, ubatch, chunk);
the other two are where those rows were computed. The contract's AC2 is the
identity

    experts_device + experts_host == experts_total

on EVERY record. A row the engine computed but never demanded, or dropped, or
double-counted breaks it. This tool walks the trace and reports the first
offending record by name rather than a count.

WHAT THIS DOES NOT CHECK, and the distinction is not cosmetic. This is
BOOKKEEPING, not arithmetic. A trace can balance perfectly while the CPU half
computed the wrong numbers, because the router weight that multiplies each
expert output is never written to the trace. Numerical correctness is the
existing quality benchmark's job, not this file's.

THE SHARE IS THE FALSIFIER (AC7). `experts_host / experts_total` is the
measurement the owner reads first: it is the fraction of routed work that left
the GPU. A run with the flag ON that reports a share of exactly 0.0 has a
closing ledger AND a dormant mechanism. Those are different findings and this
tool reports them differently on purpose.

Exit codes:
    0  identity holds on every record (and the share met --min-host-share,
       when given). A share of 0.0 is NOT a failure here; it prints DORMANT.
    1  at least one skewed record; the first one is printed in full
    2  a trace carried zero row-set records -- NOTHING TO EVALUATE, NOT A PASS

The exit-2 case is load-bearing, not defensive: the row-set block is absent
from any run that did not take the row-set path, so a defaulted or mis-pointed
trace would otherwise read as a clean pass. Modelled on
tools/check-route-parity.py, which exits 2 for the same reason.

Usage:
    python tools/check-gplr82-ledger.py TRACE [TRACE ...]
        [--min-host-share F] [--json OUT] [--quiet]
"""
import json
import sys

# the three integers the identity is made of, and the presence key that marks a
# record as carrying the row-set block at all
K_TOTAL = 'experts_total'
K_HOST = 'experts_host'
K_DEVICE = 'experts_device'


def load(path):
    """Return (events, malformed) for a JSONL trace.

    A killed run leaves a truncated final line; that is a fact about the run,
    not a reason to refuse to read the rest, so malformed lines are counted and
    reported rather than raised.
    """
    events = []
    malformed = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                malformed.append((lineno, str(exc)))
    return events, malformed


def rowset_records(events):
    """The layer_compute records that carry the optional row-set block."""
    return [e for e in events
            if e.get('event') == 'layer_compute' and K_TOTAL in e]


def describe(rec, lineno=None):
    """Name a record precisely enough to find it in the raw trace."""
    where = ''
    if lineno is not None:
        where = ' line=%d' % lineno
    return ('layer=%s ubatch=%s chunk=%s phase=%s slot=%s generation=%s'
            '%s experts_total=%s experts_host=%s experts_device=%s'
            % (rec.get('layer'), rec.get('ubatch'), rec.get('chunk'),
               rec.get('phase'), rec.get('slot'), rec.get('generation'),
               where, rec.get(K_TOTAL), rec.get(K_HOST), rec.get(K_DEVICE)))


def check_one(path, quiet):
    """Return (exit_code, summary_dict)."""
    events, malformed = load(path)
    rows = rowset_records(events)

    print('=' * 66)
    print(path)

    if malformed:
        print('  NOTE: %d unparseable line(s); first at %s' %
              (len(malformed), malformed[0][0]))

    layer_computes = sum(1 for e in events if e.get('event') == 'layer_compute')
    run_start = next((e for e in events if e.get('event') == 'run_start'), None)

    if not rows:
        # The whole point: a trace with no row-set block evaluates nothing.
        print('  row-set records: 0 of %d layer_compute records' % layer_computes)
        print('  RESULT: NOTHING TO EVALUATE (exit 2, NOT a pass)')
        print('  The row-set block only appears when the row-set executor ran.')
        print('  If this run was meant to exercise it, the trace is the wrong')
        print('  one, or --residency-tg never took effect.')
        return 2, {'path': path, 'rowset_records': 0, 'nothing_to_evaluate': True}

    total_rows = 0
    host_rows = 0
    device_rows = 0
    skew = None
    host_us = []
    per_record_share = []

    for rec in rows:
        t = rec[K_TOTAL]
        h = rec[K_HOST]
        d = rec[K_DEVICE]
        total_rows += t
        host_rows += h
        device_rows += d
        if rec.get('host_us') is not None:
            host_us.append(rec['host_us'])
        if t > 0:
            per_record_share.append(h / float(t))
        if skew is None and h + d != t:
            skew = rec

    share = (host_rows / float(total_rows)) if total_rows else 0.0

    print('  row-set records: %d of %d layer_compute records' % (len(rows), layer_computes))
    print('  routed rows: total=%d host=%d device=%d' % (total_rows, host_rows, device_rows))
    print('  share experts_host/experts_total: %.6f' % share)
    if per_record_share:
        print('  per-record share: min=%.6f max=%.6f' %
              (min(per_record_share), max(per_record_share)))
    if host_us:
        print('  host_us over records: mean=%.1f max=%.1f (S1 span, not host arithmetic)'
              % (sum(host_us) / len(host_us), max(host_us)))
    if run_start:
        # only the flags that change what this trace MEANS, not a full dump
        for k in ('pp_prebuild_per_pass', 'pp_prebuild_per_pass_source',
                  'pp_schedule', 'kv_cache_type', 'n_slots'):
            if k in run_start:
                print('  run_start %s: %s' % (k, run_start[k]))

    if skew is not None:
        print('  RESULT: LEDGER SKEW on the first offending record')
        print('    %s' % describe(skew))
        print('    experts_device + experts_host != experts_total')
        return 1, {'path': path, 'rowset_records': len(rows),
                    'total': total_rows, 'host': host_rows,
                    'device': device_rows, 'share': share, 'skew': describe(skew)}

    if total_rows == 0:
        # identity holds trivially but there was no demand to measure
        print('  RESULT: ledger closed, but every record demanded 0 rows')
        print('  Nothing was computed, so the share is not meaningful.')
        return 2, {'path': path, 'rowset_records': len(rows),
                    'total': 0, 'host': 0, 'device': 0,
                    'share': 0.0, 'nothing_to_evaluate': True}

    if share == 0.0:
        print('  RESULT: ledger CLOSED and mechanism DORMANT')
        print('  The identity holds, but experts_host is 0 on every record:')
        print('  no routed row was computed on the host. The flag may not')
        print('  have reached the graph builder, or the row-set path did not')
        print('  run. This is a real finding, not a pass with a benefit.')
    else:
        print('  RESULT: ledger CLOSED, %.2f%% of routed rows computed on the host'
              % (100.0 * share))

    if not quiet and per_record_share:
        hist = {}
        for s in per_record_share:
            hist[round(s, 4)] = hist.get(round(s, 4), 0) + 1
        print('  share histogram: %s' % ', '.join(
            '%s x%d' % (k, v) for k, v in sorted(hist.items())))

    return 0, {'path': path, 'rowset_records': len(rows), 'total': total_rows,
               'host': host_rows, 'device': device_rows, 'share': share}


def main(argv):
    paths = []
    min_share = None
    json_out = None
    quiet = False

    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--min-host-share':
            i += 1
            min_share = float(argv[i])
        elif a == '--json':
            i += 1
            json_out = argv[i]
        elif a == '--quiet':
            quiet = True
        elif a.startswith('--'):
            print('unknown option: %s' % a, file=sys.stderr)
            return 1
        else:
            paths.append(a)
        i += 1

    if not paths:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        print('usage: check-gplr82-ledger.py TRACE [TRACE ...] '
              '[--min-host-share F] [--json OUT] [--quiet]', file=sys.stderr)
        return 1

    worst = 0
    fail_reason = None
    summaries = []
    for path in paths:
        rc, summary = check_one(path, quiet)
        summaries.append(summary)
        # 2 (nothing to evaluate) must not be masked by an earlier 0
        if rc == 2:
            worst = 2
            fail_reason = fail_reason or 'a trace carried no row-set records'
        elif rc == 1 and worst != 2:
            worst = 1
            fail_reason = fail_reason or 'a record broke experts_device + experts_host == experts_total'

    print('=' * 66)
    measured = [s for s in summaries if 'share' in s and not s.get('nothing_to_evaluate')]
    if measured:
        agg_host = sum(s['host'] for s in measured)
        agg_total = sum(s['total'] for s in measured)
        print('AGGREGATE over %d trace(s): host %d / total %d = %.6f' %
              (len(measured), agg_host, agg_total,
               (agg_host / float(agg_total)) if agg_total else 0.0))
        if min_share is not None:
            share = (agg_host / float(agg_total)) if agg_total else 0.0
            if share < min_share:
                print('SHARE GATE FAILED: %.6f < required %.6f' % (share, min_share))
                if worst == 0:
                    worst = 1
                    fail_reason = 'measured share %.6f is below the required %.6f' % (share, min_share)

    verdict = {0: 'PASS (ledger closed)', 1: 'FAIL',
               2: 'NOTHING TO EVALUATE'}[worst]
    print('VERDICT: %s' % verdict)
    if fail_reason:
        # name WHICH failure, so a share-gate miss is never read as a ledger skew
        print('  reason: %s' % fail_reason)

    if json_out:
        with open(json_out, 'w', encoding='utf-8') as fh:
            json.dump({'verdict': verdict, 'traces': summaries}, fh, indent=2)
        print('wrote %s' % json_out)

    return worst


if __name__ == '__main__':
    sys.exit(main(sys.argv))
