#!/usr/bin/env python3
"""T-GPLR-58 A1: the PORT COMPARATOR -- does the engine-side predictor policy
produce exactly what the offline screen produces, row by row?  Host-only: no
GPU, no mutex, no model.

What it compares, and against WHAT
----------------------------------
The reference is `tools/predictor-screen-gplr.py` AT A RECORDED REVISION (its
file hash is printed in every run; `--expect-sha` pins it and refuses if the
tool moved).  This tool imports that module and drives its own candidate
classes, in the trace's demand order, with predict()-then-observe() per row --
so the reference is the screen's own code and not a transcription of it.

It then runs the nested A1 test in its `--dump` mode over the same trace and
requires the module's per-row proposal SEQUENCE to equal the reference's, for
every family and for each of the three pre-registered arms:

    coact   -> the screen's Coact
    bar09   -> Barred(Markov1, 0.9)   (the row T-GPLR-58 quotes as the safety arm)
    fusion  -> VoteFusion(history, markov-1, coact, coupling)  (param-free)

A mismatch BLOCKS A1 and is enumerated, never tolerated as a tie-break.  The
count-matched structure-free null has no Python counterpart (the screen has no
null arm); it is checked here for the properties it must have instead -- count
exactness against the arm it is matched to, distinct ids in range, and the same
sequence on a re-run (its demand-blindness is asserted in the nested test,
where a reordered stream is fed to it directly).

A measured caveat, recorded rather than hidden: the screen observes
`list(frozenset(ids))`, so ITS History family carries CPython's set-iteration
order, while the engine sees `union_ids` in the trace's order.  Measured
2026-09-23 (build/tmp-tg-a1/orderexp.py, three streams, seven candidates): the
order changes ONLY the fusion arm, by at most 0.09pp, and every number the
findings doc records for the families and the bar arms reproduces exactly under
either order.  The reference here is therefore driven in the TRACE's order --
the order the engine actually has -- and the fusion equivalence is exact by
construction on both sides.

usage:
  python tools/check-tg-predict-port.py --selftest
  python tools/check-tg-predict-port.py                      # the eight recorded legs
  python tools/check-tg-predict-port.py --leg echo=path.jsonl --tables
  python tools/check-tg-predict-port.py --json out.json --expect-sha <hex>
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

EXIT_OK, EXIT_MISMATCH, EXIT_REFUSED = 0, 1, 2

SCREEN = os.path.join('tools', 'predictor-screen-gplr.py')
SCREEN_REVISION = 'pinned by --expect-sha in the A1 result'  # the recorded hash lives there
FINDINGS = os.path.join('docs', 'findings', 'predictor-screen-gplr.md')

DEFAULT_DUMP_BINS = [
    os.path.join('build', 'llama-cuda', 'bin', 'Release', 'test-layer-streaming-tg-predict.exe'),
    os.path.join('build', 'llama-cuda', 'bin', 'Release', 'test-layer-streaming-tg-predict'),
]

# The eight recorded retention legs, each with the demand stream it belongs to.
# The stream label is CHECKED (below): the four echo legs must share one stream,
# the two honest legs another, the two qa legs a third - which is the same
# "same-stream files reproduce identical tables" claim the findings doc makes.
DEFAULT_LEGS = [
    ('echo',   'build/llama-cuda/gplr19-k2-ab/arm-B1.jsonl'),
    ('echo',   'build/llama-cuda/gplr19-k4-ab/arm-B1.jsonl'),
    ('echo',   'build/llama-cuda/gplr19-k6-ab/arm-B1.jsonl'),
    ('echo',   'build/llama-cuda/gplr19-k8-ab/arm-B1.jsonl'),
    ('honest', 'build/llama-cuda/gplr19-honest-k6-ab/arm-B1.jsonl'),
    ('honest', 'build/llama-cuda/gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl'),
    ('qa',     'build/llama-cuda/qa-depth-k4.jsonl'),
    ('qa',     'build/llama-cuda/qa-depth-k8.jsonl'),
]

# The findings doc's recorded values for those streams (B=8, full stream): recall% /
# ppb%.  Only the rows the doc actually states are listed - the qa stream has no
# recorded bar row, so none is asserted, and the doc's fusion row is a held-out
# tune/held quantity that is not comparable to a full-stream number here.
DOC_TABLE = {
    'echo':   {'history': (30.8, 30.9), 'markov-1': (39.5, 39.9), 'coact': (42.6, 43.0),
               'coupling': (31.9, 32.9), 'bar09': (36.1, 41.7)},
    'honest': {'history': (24.6, 24.7), 'markov-1': (27.4, 27.7), 'coact': (28.1, 28.4),
               'coupling': (20.6, 21.2), 'bar09': (25.1, 29.0)},
    'qa':     {'history': (33.3, 33.7), 'markov-1': (35.5, 36.4), 'coact': (36.8, 37.7),
               'coupling': (30.2, 31.4)},
}

DOC_PRECISION = 0.05  # the doc quotes one decimal, so half a tenth is the tolerance
# The float-scored families divide integer counts, so the REFERENCE's id order can
# move the last bits of a score across the doc's rounding boundary: measured
# 2026-09-23, coupling +0.02pp (echo) and +0.01pp (qa) - exactly one rounded tenth.
# Those rows are held to the doc's rounding plus that measured artifact, and no more.
ORDER_SENSITIVE = {'coupling'}
ORDER_PRECISION = 0.15


class Refused(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- reference
def load_screen():
    if not os.path.exists(SCREEN):
        raise Refused('screen-missing', 'the reference tool %s is absent: A1 cannot be '
                                        'validated without it' % SCREEN)
    spec = importlib.util.spec_from_file_location('gplr_screen', SCREEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def screen_sha256():
    return hashlib.sha256(open(SCREEN, 'rb').read()).hexdigest()


def load_rows_ordered(screen, path):
    """The trace's demand rows in FILE order, keeping the trace's id order.

    The screen's own loader (`screen.load_rows`) sorts by (ubatch, layer) and
    keeps ids in a frozenset; this mirrors its refusal semantics while keeping
    the order the engine has, which is the order the port is defined over.
    """
    if not os.path.exists(path):
        raise Refused('leg-missing', 'no such leg: %s' % path)
    rows, seen = [], set()
    for line in open(path, encoding='utf-8'):
        if '"rowset_fetch"' not in line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('event') != 'rowset_fetch':
            continue
        ids = [int(x) for x in str(e.get('union_ids', '')).split(',') if x != '']
        if not ids:
            continue
        key = (e.get('ubatch'), e.get('layer'))
        if key in seen:
            raise Refused('leg-not-single-row', '%s repeats the demand row %s: not a one-row-'
                                                 'per-(token, layer) stream (the reference refuses '
                                                 'these too)' % (path, key))
        if seen and key < max(seen):
            raise Refused('leg-not-sorted', '%s is not in ascending (ubatch, layer) order, so the '
                                            'port and the reference would evolve over different '
                                            'prefixes of the stream' % path)
        seen.add(key)
        rows.append((key[0], key[1], ids))
    if not rows:
        raise Refused('leg-no-demands', '%s has no rowset_fetch rows' % path)
    return rows


def ref_sequences(screen, rows, b):
    """Every family's and arm's proposal, per row, from the SCREEN'S OWN classes."""
    fams = {'history': screen.History(), 'markov-1': screen.Markov1(),
            'coact':   screen.Coact(),   'coupling': screen.CouplingCopula()}
    arms = {'coact':  screen.Coact(),
            'bar09':  screen.Barred(screen.Markov1(), 0.9),
            'fusion': screen.VoteFusion(screen.CANDIDATES)}
    out = {'family': {k: [] for k in fams}, 'arm': {k: [] for k in arms}}
    for tok, layer, ids in rows:
        for name, obj in fams.items():
            out['family'][name].append(list(obj.predict(layer, b)))
        for name, obj in arms.items():
            out['arm'][name].append(list(obj.predict(layer, b)))
        # observe AFTER predicting, for every candidate (the screen's own order)
        for obj in list(fams.values()) + list(arms.values()):
            obj.observe(layer, list(ids))
    return out


def ref_metrics(screen, rows, b):
    """recall% and ppb% per candidate, the doc's two numbers."""
    res = {}
    for name, make in (('history', screen.History), ('markov-1', screen.Markov1),
                       ('coact', screen.Coact), ('coupling', screen.CouplingCopula),
                       ('bar09', lambda: screen.Barred(screen.Markov1(), 0.9))):
        algo, hits, proposed, n = make(), 0, 0, 0
        for tok, layer, ids in rows:
            true = set(ids)
            pred = set(algo.predict(layer, b))
            algo.observe(layer, list(ids))
            hits += len(pred & true)
            proposed += len(pred)
            n += 1
        res[name] = (100.0 * hits / (n * 8), 100.0 * hits / proposed if proposed else 0.0)
    return res


def stream_digest(rows):
    h = hashlib.sha256()
    for tok, layer, ids in rows:
        h.update(('%s|%s|%s\n' % (tok, layer, ','.join(str(i) for i in ids))).encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- the port
def find_dump_bin(explicit):
    if explicit:
        if not os.path.exists(explicit):
            raise Refused('dump-bin-missing', '--dump-bin %s does not exist' % explicit)
        return explicit
    for cand in DEFAULT_DUMP_BINS:
        if os.path.exists(cand):
            return cand
    raise Refused('dump-bin-missing',
                  'the A1 test binary is not built: expected one of %s. Build it with '
                  '`cmake --build build/llama-cuda --config Release --target '
                  'test-layer-streaming-tg-predict` - a guard whose subject was never built '
                  'has not passed.' % ', '.join(DEFAULT_DUMP_BINS))


def run_dump(binary, path, b, null_matched_to='coact'):
    cmd = [binary, '--dump', path, '--b', str(b), '--null-matched-to', null_matched_to]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise Refused('dump-refused', 'the dump mode refused %s (exit %d): %s'
                                      % (path, p.returncode, (p.stderr or '').strip()))
    return parse_dump(p.stdout)


def parse_dump(text):
    header, proposals = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            raise Refused('dump-unparseable', 'the dump emitted a line that is not JSON: %r' % line[:120])
        if e.get('event') == 'pred_dump':
            header = e
        elif e.get('event') == 'proposal':
            proposals.append(e)
    if header is None:
        raise Refused('dump-unparseable', 'the dump emitted no pred_dump header')
    if not proposals:
        raise Refused('dump-unparseable', 'the dump emitted no proposal rows')
    return header, proposals


# --------------------------------------------------------------------------- compare
def compare(dump_header, proposals, ref):
    """Per-row, per-arm equality.  Returns (comparisons, mismatches)."""
    n_rows = dump_header.get('rows')
    keyed = {}
    for e in proposals:
        keyed.setdefault((e.get('kind'), e.get('arm')), {})[(e['ubatch'], e['layer'])] = e
    comps, bad = 0, []
    for kind in ('family', 'arm'):
        for name, want in ref[kind].items():
            got_rows = keyed.get((kind, name))
            if not got_rows:
                bad.append(('%s:%s' % (kind, name), 'the dump has no rows for it'))
                continue
            if len(got_rows) != n_rows:
                bad.append(('%s:%s' % (kind, name),
                            'the dump has %d rows, the stream has %d' % (len(got_rows), n_rows)))
                continue
            for i, key in enumerate(sorted(got_rows.keys())):
                tok, layer = key
                got = got_rows[key]
                exp = want[i]   # the reference is driven in the same sorted row order
                comps += 1
                if list(got['ids']) != list(exp):
                    bad.append(('%s:%s' % (kind, name),
                                'row %d (ubatch %s, layer %s): port %s != reference %s'
                                % (i, tok, layer, got['ids'], exp)))
    return comps, bad


def check_null(proposals, dump_header):
    """The null's own properties: count-exact, distinct, in range."""
    bad = []
    matched_to = dump_header.get('null_matched_to')
    arm_rows = {}
    for e in proposals:
        if e.get('kind') == 'arm' and e.get('arm') == matched_to:
            arm_rows[(e['ubatch'], e['layer'])] = e
    n_expert = dump_header.get('n_expert', 0)
    seen_rows = 0
    for e in proposals:
        if e.get('kind') != 'null':
            continue
        seen_rows += 1
        ref_row = arm_rows.get((e['ubatch'], e['layer']))
        if ref_row is None:
            bad.append('null row (%s, %s) has no matched %s arm row'
                       % (e['ubatch'], e['layer'], matched_to))
            continue
        if e['count'] != ref_row['n']:
            bad.append('null row (%s, %s) matched count %s but the %s arm published %s'
                       % (e['ubatch'], e['layer'], e['count'], matched_to, ref_row['n']))
        if e['n'] != e['count']:
            bad.append('null row (%s, %s) published %s ids for a count of %s'
                       % (e['ubatch'], e['layer'], e['n'], e['count']))
        ids = e['ids']
        if len(set(ids)) != len(ids) or any(not (0 <= i < n_expert) for i in ids):
            bad.append('null row (%s, %s) has duplicate or out-of-range ids' % (e['ubatch'], e['layer']))
    if seen_rows != dump_header.get('rows'):
        bad.append('the dump has %d null rows for %d stream rows' % (seen_rows, dump_header.get('rows')))
    return bad


# --------------------------------------------------------------------------- tables
def check_tables(screen, legs, b, doc_text):
    """The reference against the findings doc's recorded rows (one check per stream)."""
    lines, bad, seen = [], [], set()
    for role, path in legs:
        want = DOC_TABLE.get(role)
        if want is None or role in seen:
            continue
        seen.add(role)
        rows = load_rows_ordered(screen, path)
        got = ref_metrics(screen, rows, b)
        for name, (wr, wp) in want.items():
            gr, gp = got[name]
            tol = ORDER_PRECISION if name in ORDER_SENSITIVE else DOC_PRECISION
            ok = abs(gr - wr) <= tol and abs(gp - wp) <= tol
            if not ok:
                bad.append('%s %s: reference %.1f/%.1f vs doc %.1f/%.1f (tolerance %.2f)'
                           % (role, name, gr, gp, wr, wp, tol))
            lines.append('    %-6s %-9s reference %5.1f/%5.1f  doc %5.1f/%5.1f  %s%s'
                         % (role, name, gr, gp, wr, wp, 'ok' if ok else 'MISMATCH',
                            '  (order-sensitive row)' if name in ORDER_SENSITIVE else ''))
            # the doc must actually state the row it is being held to
            pair = '%.1f / %.1f' % (wr, wp)
            if pair not in doc_text:
                bad.append('the findings doc no longer states %r for %s' % (pair, name))
    return lines, bad


# --------------------------------------------------------------------------- selftest
def _planted_trace(path, n_tok=3, n_layer=3, n_expert=24, per_row=4):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'event': 'run_start', 'run_id': 1, 'n_layer': n_layer}) + '\n')
        fh.write(json.dumps({'event': 'rowset_init', 'run_id': 1, 'retain_k': 4, 'top_k': per_row,
                             'n_expert': n_expert}) + '\n')
        for t in range(n_tok):
            for L in range(n_layer):
                ids = [str((t * 3 + L * 5 + k) % n_expert) for k in range(per_row)]
                fh.write(json.dumps({'event': 'rowset_fetch', 'run_id': 1, 'ubatch': t, 'layer': L,
                                     'union_ids': ','.join(ids), 'n_rows_resident': 0,
                                     'n_rows_fetched': per_row}) + '\n')
        fh.write(json.dumps({'event': 'run_end', 'run_id': 1, 'exit_status': 0}) + '\n')
    return path


def selftest():
    cases, fails = 0, []

    def check(name, got, want):
        nonlocal cases
        cases += 1
        if got != want:
            fails.append('%s: got %r want %r' % (name, got, want))

    import tempfile
    tmp = tempfile.mkdtemp(prefix='tgport-')
    screen = load_screen()

    # 1. a synthetic leg: the port must equal the reference, row by row
    leg = _planted_trace(os.path.join(tmp, 'leg.jsonl'))
    binary = find_dump_bin(None)
    header, props = run_dump(binary, leg, 8)
    rows = load_rows_ordered(screen, leg)
    ref = ref_sequences(screen, rows, 8)
    comps, bad = compare(header, props, ref)
    check('the planted leg agrees on every family and arm', bad, [])
    check('the comparison actually compared rows', comps > 0, True)
    check('the null is count-exact and in range', check_null(props, header), [])

    # 2. the NEGATIVE control: a tampered dump must be CAUGHT, not tolerated
    tampered = json.loads(json.dumps(props))
    for e in tampered:
        if e.get('kind') == 'arm' and e.get('arm') == 'fusion' and e['n'] > 0:
            e['ids'][0] = (e['ids'][0] + 1) % header['n_expert']
            break
    comps2, bad2 = compare(header, tampered, ref)
    check('a single flipped id is detected', len(bad2) >= 1, True)
    check('the mismatch names the row and both sequences', 'port' in bad2[0][1], True)

    dropped = [e for e in props if not (e.get('kind') == 'family' and e.get('arm') == 'coact')]
    _c3, bad3 = compare(header, dropped, ref)
    check('a dropped proposal row is detected', len(bad3) >= 1, True)

    bad4 = check_null([json.loads(json.dumps(e)) for e in props]
                      + [{'event': 'proposal', 'kind': 'null', 'arm': 'null', 'ubatch': 0, 'layer': 0,
                          'count': 2, 'n': 1, 'ids': [1]}], header)
    check('a short null row is detected', len(bad4) >= 1, True)

    # 3. REFUSALS: a leg the dump cannot read, an unsorted leg, a missing binary
    noinit = os.path.join(tmp, 'noinit.jsonl')
    with open(noinit, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'event': 'rowset_fetch', 'ubatch': 0, 'layer': 0, 'union_ids': '1,2'}) + '\n')
    try:
        run_dump(binary, noinit, 8)
        check('a trace without rowset_init is refused', False, True)
    except Refused as r:
        check('a trace without rowset_init is refused', r.code, 'dump-refused')

    unsorted = os.path.join(tmp, 'unsorted.jsonl')
    _planted_trace(unsorted)
    lines = open(unsorted, encoding='utf-8').read().splitlines()
    fetches = [i for i, l in enumerate(lines) if 'rowset_fetch' in l]
    lines[fetches[0]], lines[fetches[1]] = lines[fetches[1]], lines[fetches[0]]
    open(unsorted, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    try:
        load_rows_ordered(screen, unsorted)
        check('an out-of-order leg is refused', False, True)
    except Refused as r:
        check('an out-of-order leg is refused', r.code, 'leg-not-sorted')

    try:
        find_dump_bin(os.path.join(tmp, 'nope.exe'))
        check('a missing dump binary is refused', False, True)
    except Refused as r:
        check('a missing dump binary is refused', r.code, 'dump-bin-missing')

    # 4. determinism ACROSS PROCESSES: a second dump of the same leg is identical
    _h2, props2 = run_dump(binary, leg, 8)
    check('two independent dumps of one leg are identical',
          [(e['kind'], e['arm'], e['ids']) for e in props] ==
          [(e['kind'], e['arm'], e['ids']) for e in props2], True)

    # 5. the doc table checker really compares, and really fails on a wrong value
    doc = open(FINDINGS, encoding='utf-8').read()
    lines5, bad5 = check_tables(screen, [('other', leg)], 8, doc)
    check('the table checker tolerates an unlisted stream role', (lines5, bad5), ([], []))
    saved = DOC_TABLE['qa']
    DOC_TABLE['qa'] = {'history': (99.9, 99.9)}
    _l6, bad6 = check_tables(screen, [('qa', 'build/llama-cuda/qa-depth-k4.jsonl')], 8, doc)
    DOC_TABLE['qa'] = saved
    check('a wrong doc row is reported as a mismatch', len(bad6) >= 1, True)

    if fails:
        sys.stderr.write('SELFTEST FAILED (%d/%d):\n' % (len(fails), cases))
        for f in fails:
            sys.stderr.write('  %s\n' % f)
        return 1
    sys.stdout.write('SELFTEST PASSED (%d checks)\n' % cases)
    return 0


# --------------------------------------------------------------------------- main
def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--leg', action='append', default=[], metavar='ROLE=PATH',
                    help='a leg to compare (role: echo|honest|qa|other); repeatable')
    ap.add_argument('--b', type=int, default=8, help='proposal budget (default 8)')
    ap.add_argument('--dump-bin', default=None, help='path to the nested A1 test binary')
    ap.add_argument('--tables', action='store_true',
                    help='also re-derive the findings doc\'s recorded rows from the reference')
    ap.add_argument('--expect-sha', default=None,
                    help='required sha256 of the reference tool (pins the revision)')
    ap.add_argument('--json', dest='json_out', default=None)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    legs = []
    for spec in args.leg:
        if '=' not in spec:
            sys.stderr.write('REFUSED: --leg wants ROLE=PATH, got %r\n' % spec)
            return EXIT_REFUSED
        role, path = spec.split('=', 1)
        legs.append((role.strip().lower(), path))
    if not legs:
        legs = list(DEFAULT_LEGS)

    try:
        sha = screen_sha256()
        if args.expect_sha and args.expect_sha != sha:
            raise Refused('reference-moved',
                          'the reference tool is %s but --expect-sha pins %s: A1 was validated '
                          'against a DIFFERENT revision, so this comparison would move the target'
                          % (sha, args.expect_sha))
        screen = load_screen()
        binary = find_dump_bin(args.dump_bin)
        legs_loaded = [(role, path, load_rows_ordered(screen, path)) for role, path in legs]
    except Refused as r:
        sys.stderr.write('REFUSED (exit 2): [%s] %s\n' % (r.code, r.detail))
        return EXIT_REFUSED

    print('T-GPLR-58 A1 - port comparator (host-only)')
    print('  reference : %s  sha256 %s' % (SCREEN, sha))
    print('  port      : %s (--dump mode)' % binary)
    print('  budget    : B=%d' % args.b)
    print('')

    # the stream identity the doc's "same stream" claim rests on
    digests = {}
    for role, path, rows in legs_loaded:
        digests.setdefault(role, {}).setdefault(stream_digest(rows), []).append(path)
    print('  demand streams')
    for role, paths in digests.items():
        for d, members in paths.items():
            print('    %-6s %s  %d leg(s)  %s' % (role, d, len(members),
                                                  ', '.join(os.path.basename(p) for p in members)))
    print('')

    total_rows, failures, refused = 0, [], []
    for role, path, rows in legs_loaded:
        try:
            header, props = run_dump(binary, path, args.b)
            ref = ref_sequences(screen, rows, args.b)
            comps, bad = compare(header, props, ref)
            null_bad = check_null(props, header)
        except Refused as r:
            refused.append((path, r))
            print('  %-5s %-52s REFUSED [%s] %s' % (role, os.path.basename(path), r.code, r.detail))
            continue
        total_rows += comps
        ok = not bad and not null_bad
        print('  %-5s %-52s %s  rows compared: %d' % (role, os.path.basename(path),
                                                      'PASS' if ok else 'MISMATCH', comps))
        for name, detail in bad[:6]:
            print('        %-14s %s' % (name, detail))
            failures.append((path, name, detail))
        for detail in null_bad[:6]:
            print('        %-14s %s' % ('null', detail))
            failures.append((path, 'null', detail))

    tables_lines, table_bad = [], []
    if args.tables:
        doc = open(FINDINGS, encoding='utf-8').read()
        tables_lines, table_bad = check_tables(screen, legs, args.b, doc)
        print('')
        print('  the findings doc\'s recorded rows re-derived from the reference')
        for l in tables_lines:
            print(l)
        for b in table_bad:
            failures.append((FINDINGS, 'tables', b))
            print('    MISMATCH %s' % b)

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump({'reference_sha256': sha, 'budget': args.b, 'rows_compared': total_rows,
                       'failures': failures,
                       'refused': [{'leg': p, 'code': r.code, 'detail': r.detail} for p, r in refused],
                       'tables': table_bad}, fh, indent=2, sort_keys=True)

    print('')
    if refused:
        print('VERDICT: REFUSED (%d leg(s) could not be compared). A refused leg is not a pass.'
              % len(refused))
        return EXIT_REFUSED
    if failures:
        print('VERDICT: MISMATCH - the port and the offline screen disagree. This BLOCKS A1 and is '
              'enumerated above, never tolerated as a tie-break.')
        return EXIT_MISMATCH
    print('VERDICT: EQUAL - the port\'s per-row proposal sequence equals the offline screen\'s on '
          '%d leg(s), %d row comparisons, with the null count-exact. This is a gate RESULT, not an '
          'authorization: every gate in this repository is soft and final acceptance is the '
          'owner\'s decision.' % (len(legs_loaded), total_rows))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
