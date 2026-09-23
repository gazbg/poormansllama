#!/usr/bin/env python3
"""T-GPLR-53 A2: validate the recon-gated `embed_chunk` timeline (host-only).

WHY THIS EXISTS
  A2 adds one event to the layer-major PP trace, `embed_chunk`, carrying the
  step-1 (embed) timeline per chunk: {chunk, n_tokens, setup_us, compute_us,
  span_us, t_us}. The exe's two validators ignore unknown events by design
  (forward compatibility), so they prove the event does NOT break a trace but
  cannot judge whether the timeline is complete or whether it reconciles with
  the ledger. This checker does that, and it is the gate A2's acceptance names:
  "the fixture's per-chunk sums reconcile to `embed_us` within 1 ms".

WHAT IT CHECKS (per trace, all falsifiable)
  * CHK-shape  every embed_chunk carries chunk/n_tokens/setup_us/compute_us/
               span_us/t_us; n_tokens > 0; the three times are >= 0;
               setup_us + compute_us <= span_us (+1 us of float slack)
  * CHK-order  inside a pass the chunk ids are exactly 0..k-1 in file order
               (a chunk 0 starts a new pass, which is how --pp-repeat re-emits
               them), and t_us never decreases across the trace
  * CHK-cover  every pass carries the same number of chunks, and - when
               run_start declares pp_n_chunks - that number
  * CHK-sum    per pass: |sum(span_us) - that pass's host_ledger.embed_us|
               <= 1000 us (the A2 tolerance; span_us is the chunk's whole wall,
               so the sum reproduces embed_us instead of estimating it)
  * CHK-gate   a trace whose run_start declares the streaming path off
               (policy/residency off) must carry NO embed_chunk events: the
               event is emitted only under --pp-recon, so one there is loud

  --join additionally checks the second half of A2, the recon generation join:
  every `recon_attr` sample must carry the generation of a REAL staging copy of
  that same layer. Measured before the fix (gplr52-a3-mmap, 82 paired samples):
  every drift was -1, because recon_begin_bank was called with slot.generation
  before the increment -- under this keying that stamp identifies NO copy of the
  layer, so the pre-fix defect fails 100% (gplr51-a3 matched 0/89 samples,
  gplr52-a2-50k 0/125; the post-fix A3 traces match 124/124 at both shapes).

  WHY NOT (layer, ubatch) -- RE-KEYED 2026-09-22. `ubatch` on recon_attr is
  stamped by recon_flush_all from the LIVE n_ubatches at flush time, not from the
  ring entry that describes the measured copy (llama-layer-streaming.cpp:1323-1325
  writes ubatch=n_ubatches while e.il/e.gen come from the entry). A sample
  therefore names the chunk current when it was FLUSHED while its generation names
  the copy it MEASURED. The old key reported a phantom one-pass-stride drift on
  exactly the samples flushed across a pass boundary: 6 per trace at layers 38/39
  (the last PP ubatch of passes 2 and 3, plus the first TG step), -13/-14 after A2
  and -14/-15 before, byte-identical at 50k and 200k, and equal to each slot's
  per-pass generation stride (13 for layer 38 in slot 2, 14 for layer 39 in slot
  0). That the magnitude equalled the stride, and that copies-per-key was exactly
  1 for all 240 keys, is what identified it as a keying artifact rather than a
  stale stamp. The emitter site is recorded, not silently tolerated: the matched
  copy's own ubatch and the count of samples flushed after it are reported.

  The join is NEVER positional-within-layer either: submits span both PP passes
  and the TG re-staging burst (98 per layer on a 2-pass 4k run), so the k-th
  sample of a layer whose sampled banks skip a pass lands on a foreign copy and
  the phantom drift equals the inter-pass stride (measured +12 x2 on
  gplr50-ctl-4k-f16).

  Two ordering conditions keep the corrected key non-vacuous: a sample may not
  name a chunk BEFORE the copy it identifies, and when both carry t_us the copy
  must precede the sample. Both hold on every A3 sample (124/124, zero
  violations), and neither is satisfied by construction.

PAIRING RULE (never guessed): a pass's chunk group is paired by file order with
that pass's `host_ledger`, and the counts must be equal. Traces recorded before
A2 carry no embed_chunk at all: that is "cannot evaluate" (exit 2), NOT a pass.

USAGE
  python tools/check-gplr53-embed-chunks.py TAG1 TAG2 ...  # build/llama-cuda/<TAG>.jsonl
  python tools/check-gplr53-embed-chunks.py PATH           # an existing path is used as-is
  python tools/check-gplr53-embed-chunks.py --selftest     # inline fixtures + controls
  python tools/check-gplr53-embed-chunks.py TAG --json     # rows as JSON
  python tools/check-gplr53-embed-chunks.py TAG --join     # + the recon generation join

EXIT CODES
  0 = PASS     every evaluated trace satisfied every check
  1 = FAIL     at least one named check failed (the check name is printed); with
               --join, any nonzero recon/copy generation drift fails too
  2 = cannot evaluate: no arguments, a missing trace, no embed_chunk in the
      trace (pre-A2 recording), no host_ledger, a chunk-group/ledger count
      mismatch, or no run_start (a run that died before resolution is NOT a pass)
"""
import io
import json
import os
import sys

BUILD_DIR = os.path.join('build', 'llama-cuda')
SELFTEST_DIR = os.path.join(BUILD_DIR, 'gplr53-a2-selftest')
SUM_TOLERANCE_US = 1000.0   # the A2 acceptance tolerance
SPAN_SLACK_US = 1.0         # float noise between span and setup+compute


def load_events(path):
    events = []
    with io.open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '"event"' not in line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def num(e, key):
    v = e.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def resolve(arg):
    if os.path.exists(arg):
        return arg
    cand = os.path.join(BUILD_DIR, arg + '.jsonl')
    return cand if os.path.exists(cand) else None


def evaluate(events, checks):
    """Return (passes, reason). reason is set when the trace cannot be evaluated."""
    def record(name, good, detail):
        checks.append((name, good, detail))

    run_start = next((e for e in events if e.get('event') == 'run_start'), None)
    if run_start is None:
        return None, 'no run_start (a run that died before resolution is not a pass)'
    chunks = [e for e in events if e.get('event') == 'embed_chunk']
    ledgers = [e for e in events if e.get('event') == 'host_ledger']
    if not chunks:
        return None, ('no embed_chunk in trace: the event is recon-gated, so this is '
                      'either a pre-A2 recording or a run without --pp-recon - nothing '
                      'to evaluate (exit 2, NOT a pass)')
    if not ledgers:
        return None, ('no host_ledger in trace: nothing to reconcile the chunk sums '
                      'against (exit 2, NOT a pass)')

    # CHK-gate: the event belongs to the opt-in path only.
    off_declared = (str(run_start.get('policy', '')).lower() == 'off' or
                    str(run_start.get('residency', '')).lower() == 'off')
    if off_declared:
        record('CHK-gate', False, 'embed_chunk in a run whose run_start declares the streaming path off')
    else:
        record('CHK-gate', True, 'run_start does not declare the streaming path off')

    # CHK-shape
    shape_bad = []
    for i, e in enumerate(chunks):
        bad = []
        if not isinstance(e.get('chunk'), int) or e.get('chunk') < 0:
            bad.append('chunk')
        if num(e, 'n_tokens') is None or num(e, 'n_tokens') <= 0:
            bad.append('n_tokens')
        for k in ('setup_us', 'compute_us', 'span_us', 't_us'):
            if num(e, k) is None or num(e, k) < 0:
                bad.append(k)
        if not bad and num(e, 'setup_us') + num(e, 'compute_us') > num(e, 'span_us') + SPAN_SLACK_US:
            bad.append('setup+compute > span')
        if bad:
            shape_bad.append('event %d: %s' % (i, ', '.join(bad)))
    record('CHK-shape', not shape_bad,
           '; '.join(shape_bad[:3]) if shape_bad else
           '%d event(s) well formed (setup+compute <= span)' % len(chunks))

    # pass groups: a chunk id of 0 starts a new pass
    groups = []
    for e in chunks:
        if not groups or e.get('chunk') == 0:
            groups.append([])
        groups[-1].append(e)

    # CHK-order
    order_bad = []
    for gi, g in enumerate(groups):
        ids = [e.get('chunk') for e in g]
        if ids != list(range(len(g))):
            order_bad.append('pass %d chunk ids %r are not 0..%d in order'
                             % (gi + 1, ids, len(g) - 1))
    ts = [num(e, 't_us') for e in chunks]
    if any(b < a for a, b in zip(ts, ts[1:])):
        order_bad.append('t_us decreases between events')
    record('CHK-order', not order_bad,
           '; '.join(order_bad[:3]) if order_bad else
           '%d pass group(s), ids contiguous, t_us monotonic' % len(groups))

    # CHK-cover
    sizes = [len(g) for g in groups]
    declared = run_start.get('pp_n_chunks')
    cover_bad = []
    if len(set(sizes)) != 1:
        cover_bad.append('pass chunk counts differ: %r' % (sizes,))
    if isinstance(declared, int) and declared > 0:
        for gi, n in enumerate(sizes):
            if n != declared:
                cover_bad.append('pass %d has %d chunk(s), run_start.pp_n_chunks = %d'
                                 % (gi + 1, n, declared))
    record('CHK-cover', not cover_bad,
           '; '.join(cover_bad[:3]) if cover_bad else
           'every pass carries %d chunk(s)%s' % (
               sizes[0], ' (= pp_n_chunks)' if isinstance(declared, int) and declared > 0 else ''))

    # CHK-sum (group/ledger pairing by order; a mismatch is a refusal, not a guess)
    if len(groups) != len(ledgers):
        return None, ('chunk-group/host_ledger count mismatch: %d group(s) vs %d ledger(s) '
                      '- pairing is refused, never guessed' % (len(groups), len(ledgers)))
    passes = []
    sum_bad = []
    for gi, (g, led) in enumerate(zip(groups, ledgers)):
        span_sum = round(sum(num(e, 'span_us') for e in g), 1)
        embed = num(led, 'embed_us')
        if embed is None:
            sum_bad.append('pass %d ledger has no embed_us' % (gi + 1))
            continue
        diff = abs(span_sum - embed)
        passes.append({'pass': gi + 1, 'chunks': len(g), 'span_sum_us': span_sum,
                       'embed_us': embed, 'diff_us': round(diff, 1)})
        if diff > SUM_TOLERANCE_US:
            sum_bad.append('pass %d: sum(span_us) %.1f vs embed_us %.1f (off by %.1f us > %.0f us)'
                           % (gi + 1, span_sum, embed, diff, SUM_TOLERANCE_US))
    record('CHK-sum', not sum_bad,
           '; '.join(sum_bad[:3]) if sum_bad else
           'per-pass sum(span_us) == embed_us within %.0f us' % SUM_TOLERANCE_US)
    return passes, None


def join_check(events, checks):
    """CHK-join: every recon_attr sample's generation must identify a real copy of
    that layer, and that copy must not postdate the sample's flush.

    T-GPLR-53 A1 follow-up (2026-09-21): the pairing is keyed, never positional
    within a layer -- the positional form mispairs across passes and across the
    PP->TG re-staging burst (measured +12 x2 on gplr50-ctl-4k-f16).

    T-GPLR-53 (2026-09-22) re-keyed the KEY from (layer, ubatch) to
    (layer, generation). See the module docstring for the measurement that forced
    it: recon_attr's ubatch is the LIVE counter at flush time, not the copy's own
    chunk, so the old key produced a phantom pass-stride drift on 6 samples per
    trace. Generations are unique per layer (each copy increments the slot
    counter), so (layer, generation) pins the exact copy while staying insensitive
    to the flush stamp. The rider keeps its teeth: the pre-fix -1 stamp matches no
    copy of that layer, and every pre-A2 trace fails 100% (gplr51-a3 0/89,
    gplr52-a2-50k 0/125, gplr52-a3-mmap 0/85 matched).
    """
    recons = []
    submits = {}
    for e in events:
        if e.get('event') == 'recon_attr':
            recons.append(e)
        elif e.get('event') == 'copy_submit':
            submits.setdefault((e.get('layer'), e.get('generation')), []).append(e)
    if not recons:
        return 'no recon_attr in trace: it needs --pp-recon, so nothing to join (exit 2, NOT a pass)'
    unkeyed = [r for r in recons if r.get('layer') is None or r.get('generation') is None]
    unkeyed += [e for k, v in submits.items() for e in v if k[0] is None or k[1] is None]
    if unkeyed:
        return ('event(s) missing layer/generation fields: cannot key the join (exit 2); '
                'first offender: layer=%s generation=%s'
                % (unkeyed[0].get('layer'), unkeyed[0].get('generation')))
    if any(r.get('ubatch') is None for r in recons) or \
            any(e.get('ubatch') is None for v in submits.values() for e in v):
        return ('recon_attr/copy_submit without ubatch: the flush-order condition cannot be '
                'checked (exit 2, NOT a pass)')
    paired = 0
    unmatched = []
    live_stamped = 0
    early = []
    for r in recons:
        key = (r['layer'], r['generation'])
        slist = submits.get(key, [])
        if not slist:
            unmatched.append(key)
            continue
        paired += 1
        s = slist[0]
        if r['ubatch'] > s['ubatch']:
            live_stamped += 1
        elif r['ubatch'] < s['ubatch']:
            early.append((key, r['ubatch'], s['ubatch']))
        if 't_us' in r and 't_us' in s and s['t_us'] > r['t_us']:
            early.append((key, 'copy t_us', s['t_us']))
    ok = (not unmatched) and (not early)
    if unmatched:
        detail = ('%d sample(s) carry a generation NO copy of that layer ever had - the '
                  'pre-T-GPLR-53 stamp offset (first: layer %s generation %s)'
                  % (len(unmatched), unmatched[0][0], unmatched[0][1]))
    elif early:
        detail = ('%d sample(s) name a flush EARLIER than the copy they identify '
                  '(ubatch/timestamp order violated), first: %s' % (len(early), early[0]))
    else:
        detail = ('%d sample(s) identify a real copy by (layer, generation); %d were '
                  'flushed after the live ubatch counter advanced past it (tolerated: the '
                  'generation is the identity, the ubatch is the flush stamp)'
                  % (paired, live_stamped))
    checks.append(('CHK-join', ok, detail))
    return None


def run_one(path, rows, want_join=False):
    checks = []
    events = load_events(path)
    if want_join:
        reason = join_check(events, checks)
        if reason is not None:
            print('REFUSED  %s' % path)
            print('         %s' % reason)
            rows.append({'status': 'REFUSED', 'path': path, 'detail': reason, 'passes': []})
            return 2
        failed = [c for c in checks if not c[1]]
        print('%-8s %s' % ('FAIL' if failed else 'PASS', path))
        for name, good, detail in checks:
            print('         %-10s %s%s' % (name, 'ok' if good else 'FAILED', '' if good else ' - ' + detail))
        rows.append({'status': 'FAIL' if failed else 'PASS', 'path': path, 'detail': '', 'passes': []})
        return 1 if failed else 0
    passes, reason = evaluate(events, checks)
    if reason is not None:
        print('REFUSED  %s' % path)
        print('         %s' % reason)
        rows.append({'status': 'REFUSED', 'path': path, 'detail': reason, 'passes': []})
        return 2
    failed = [c for c in checks if not c[1]]
    print('%-8s %s' % ('FAIL' if failed else 'PASS', path))
    for p in passes or []:
        print('         pass %d: %d chunk(s), sum(span_us) %.1f, embed_us %.1f, diff %.1f us'
              % (p['pass'], p['chunks'], p['span_sum_us'], p['embed_us'], p['diff_us']))
    for name, good, detail in checks:
        print('         %-10s %s%s' % (name, 'ok' if good else 'FAILED',
                                       '' if good else ' - ' + detail))
    rows.append({'status': 'FAIL' if failed else 'PASS', 'path': path, 'detail': '',
                 'passes': passes or []})
    return 1 if failed else 0


# --- selftest fixtures (transient, under build/) -----------------------------

def join_fixture(sample_drift, sample_ubatch=None):
    """One layer, two staging copies in successive passes, a recon sample each
    (the lazy recon flush samples a subset, as recorded), offset by
    `sample_drift`. The copy generation is the join identity; `sample_ubatch`
    overrides the sample's flush chunk so the ordering condition can be exercised.
    """
    lines = [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'}]
    for gen, ub in ((1, 8), (15, 16)):
        lines.append({'event': 'copy_submit', 'run_id': 1, 'layer': 0, 'slot': 0,
                      'generation': gen, 'ubatch': ub})
        lines.append({'event': 'copy_complete', 'run_id': 1, 'layer': 0, 'slot': 0,
                      'generation': gen, 'ubatch': ub})
        lines.append({'event': 'recon_attr', 'run_id': 1, 'layer': 0,
                      'ubatch': ub if sample_ubatch is None else sample_ubatch,
                      'generation': gen + sample_drift, 'copy_gpu_ms': 31.5, 'lag_ms': 0.0})
    return lines


def tg_burst_fixture(sample_drift):
    """The measured gplr50 phantom, as a negative control: a layer sampled only
    in pass 2 (ub 16) whose layer-level positional index lands on its pass-1
    copy (ub 8) because the TG burst fills the layer's submit list. Under the
    keyed pairing the sample sits at `sample_drift` of its TRUE copy; the old
    positional pairing would report stride - stamp = 13. """
    lines = [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'}]
    # pass 1 (ub 8) staging, unsampled for this layer
    lines.append({'event': 'copy_submit', 'run_id': 1, 'layer': 17, 'slot': 0,
                  'generation': 6, 'ubatch': 8})
    # pass 2 (ub 16) staging: the pass stride puts generation 14 ahead
    lines.append({'event': 'copy_submit', 'run_id': 1, 'layer': 17, 'slot': 0,
                  'generation': 20, 'ubatch': 16})
    # ...followed by the TG re-staging burst for the same layer
    for k in range(96):
        lines.append({'event': 'copy_submit', 'run_id': 1, 'layer': 17, 'slot': 0,
                      'generation': 22 + 14 * k, 'ubatch': 17 + k})
    # the one recon sample: the sampled bank was stamped drift below its copy
    lines.append({'event': 'recon_attr', 'run_id': 1, 'layer': 17, 'ubatch': 16,
                  'generation': 20 + sample_drift, 'copy_gpu_ms': 31.5, 'lag_ms': 0.0})
    return lines


def chunk_event(chunk, span, t_us, n_tokens=512):
    return {'event': 'embed_chunk', 'run_id': 1, 'chunk': chunk, 'n_tokens': n_tokens,
            'setup_us': 20.0, 'compute_us': round(span - 45.0, 1), 'span_us': span, 't_us': t_us}


def fixture(runs, policy='streaming', pp_n_chunks=None):
    """`runs` = [(spans, embed_us), ...]: one host_ledger per run, in order."""
    lines = [{'event': 'run_start', 'run_id': 1, 'policy': policy, 'residency': policy,
              'n_slots': 3, 'n_layer': 2, 'pp_tokens': 512 * len(runs[0][0])}]
    if pp_n_chunks is not None:
        lines[0]['pp_n_chunks'] = pp_n_chunks
    t = 1000
    for spans, embed in runs:
        for i, span in enumerate(spans):
            lines.append(chunk_event(i, span, t))
            t += 10
        lines.append({'event': 'host_ledger', 'run_id': 1, 'embed_us': embed,
                      'output_us': 5.0, 'loop_us': round(-(embed + 5.0), 1), 't_us': t})
        t += 10
    return lines


def selftest():
    cases = [
        ('one-pass', 0, fixture([([1000.0, 400.0, 300.0, 146.0], 1846.0)])),
        ('two-pass', 0, fixture([([1000.0, 400.0, 300.0, 146.0], 1846.0),
                                 ([62.5, 62.5, 62.5, 62.5], 250.0)])),
        ('bad-sum', 1, fixture([([100.0, 100.0, 100.0, 100.0], 1846.0)])),
        ('bad-index', 1, fixture([([100.0, 100.0, 100.0, 100.0], 400.0)])),
        ('dup-index', 1, fixture([([100.0, 100.0, 100.0, 100.0], 400.0)])),
        ('bad-span', 1, fixture([([100.0, 100.0, 100.0, 100.0], 400.0)])),
        ('count-vs-manifest', 1, fixture([([100.0, 100.0], 200.0)], pp_n_chunks=4)),
        ('off-run', 1, fixture([([100.0, 100.0], 200.0)], policy='off')),
        ('no-event', 2, [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'},
                         {'event': 'host_ledger', 'run_id': 1, 'embed_us': 1.0}]),
        ('no-ledger', 2, [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'},
                          chunk_event(0, 100.0, 1000)]),
        ('no-run-start', 2, [chunk_event(0, 100.0, 1000),
                             {'event': 'host_ledger', 'run_id': 1, 'embed_us': 100.0}]),
        ('group-mismatch', 2, [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'},
                               chunk_event(0, 100.0, 1000), chunk_event(1, 100.0, 1010),
                               chunk_event(0, 100.0, 1020), chunk_event(1, 100.0, 1030),
                               {'event': 'host_ledger', 'run_id': 1, 'embed_us': 200.0}]),
    ]
    # join cases run in --join mode: the recon sample must carry the copy's generation
    join_cases = [
        ('join-clean', 0, join_fixture(0)),
        ('join-offbyone', 1, join_fixture(-1)),
        ('join-no-events', 2, [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'}]),
        # the gplr50 phantom under the OLD positional pairing: the keyed join holds
        # it to its true stamp, and a pre-fix -1 identifies no copy of layer 17
        ('join-tg-burst-phantom', 1, tg_burst_fixture(-1)),
        # a stamp that identifies NO copy at all fails on the same shape
        ('join-tg-burst-nocopy', 1, tg_burst_fixture(-13)),
        # RE-SPECIFIED 2026-09-22 (was 'join-tg-burst-bad', expect 1): -14 lands
        # exactly on this layer's pass-1 copy (generation 6), so under the
        # corrected identity keying it is byte-for-byte the flush-later shape the
        # check must ACCEPT. It only failed before because the key was
        # (layer, ubatch) and the sample named a later chunk -- a requirement the
        # emitter never met (see the module docstring). The acceptance PATH is now
        # what is asserted; the failure path is covered by the -1/-13 controls
        # above and by the four pre-A2 traces, which fail 100%.
        ('join-tg-burst-stale-legal', 0, tg_burst_fixture(-14)),
        # a sample that names a flush BEFORE the copy it identifies is a real
        # inconsistency and must fail
        ('join-flush-before-copy', 1, join_fixture(0, sample_ubatch=4)),
        # missing ubatch: the join cannot be keyed - refused, not guessed
        ('join-no-ubatch', 2, [{'event': 'run_start', 'run_id': 1, 'residency': 'streaming'},
                               {'event': 'copy_submit', 'run_id': 1, 'layer': 0, 'slot': 0,
                                'generation': 1},
                               {'event': 'recon_attr', 'run_id': 1, 'layer': 0, 'ubatch': 8,
                                'generation': 0, 'copy_gpu_ms': 1.0, 'lag_ms': 0.0}]),
    ]
    # single-defect variants: each mutated line carries exactly one fault
    cases[3][2][4]['chunk'] = 4                                   # ids 0,1,2,4: a gap
    cases[4][2][4]['chunk'] = 2                                   # ids 0,1,2,2: a duplicate
    cases[5][2][3]['setup_us'] = cases[5][2][3]['compute_us'] = 90.0  # span 100 < 180

    os.makedirs(SELFTEST_DIR, exist_ok=True)
    bad = 0
    for name, expect, lines in cases:
        path = os.path.join(SELFTEST_DIR, name + '.jsonl')
        with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
            for e in lines:
                f.write(json.dumps(e) + '\n')
        rows = []
        got = run_one(path, rows)
        print('[selftest] %-18s expect exit %d, got %d (%s) %s'
              % (name, expect, got, rows[0]['status'], 'ok' if got == expect else 'BAD'))
        if got != expect:
            bad += 1
    join_bad = 0
    for name, expect, lines in join_cases:
        path = os.path.join(SELFTEST_DIR, name + '.jsonl')
        with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
            for e in lines:
                f.write(json.dumps(e) + '\n')
        rows = []
        got = run_one(path, rows, want_join=True)
        print('[selftest] %-18s expect exit %d, got %d (%s) %s'
              % (name, expect, got, rows[0]['status'], 'ok' if got == expect else 'BAD'))
        if got != expect:
            join_bad += 1
    total = len(cases) + len(join_cases)
    bad += join_bad
    print('[selftest] %d/%d as expected' % (total - bad, total))
    return 1 if bad else 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--selftest' in sys.argv:
        return selftest()
    if not args:
        print(__doc__)
        return 2
    rows = []
    worst = 0
    for arg in args:
        path = resolve(arg)
        if path is None:
            print('REFUSED  %s' % arg)
            print('         cannot evaluate: trace not found (looked at %s and %s)'
                  % (arg, os.path.join(BUILD_DIR, arg + '.jsonl')))
            rows.append({'status': 'REFUSED', 'path': arg, 'detail': 'missing trace',
                         'passes': []})
            worst = max(worst, 2)
            continue
        worst = max(worst, run_one(path, rows, want_join='--join' in sys.argv))
    if '--json' in sys.argv:
        print(json.dumps(rows, indent=2))
    return worst


if __name__ == '__main__':
    sys.exit(main())
