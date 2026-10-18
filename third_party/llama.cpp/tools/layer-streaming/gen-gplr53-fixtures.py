#!/usr/bin/env python3
"""T-GPLR-53 A2: emit the `embed_chunk` telemetry fixture (host-only, no GPU).

WHY
  A2 adds one recon-gated event, `embed_chunk`, to the layer-major PP trace. Its
  acceptance requires a trace that carries the event through BOTH host
  validators (`--check-telemetry`, `--validate-telemetry`), and a pass-1/pass-2
  pair whose per-chunk spans reconcile to `host_ledger.embed_us`. This generator
  builds that trace from a RECORDED run: the run_start / host_ledger /
  phase_summary SHAPES are copied from the source trace (field names and order
  come from the producer, not from this script), while the numbers that the
  fixture is about are synthesized so that

      sum(span_us) over a pass == host_ledger.embed_us of that pass   (exact)

  Pass 1 carries the front-loaded profile the A3 discriminator looks for
  (0.62 / 0.18 / 0.13 / 0.07 of the measured pass-1 embed); pass 2 sits flat on
  the measured warm band. The ledger's own identity is kept too
  (loop_us = -(embed_us + output_us)), so the fixture stays checkable by the
  T-GPLR-53 A1 attribution tool as well.

FIXTURE SHAPE (one PP pass x 4 chunks, 512 tokens each)
  run_start, embed_chunk x4, host_ledger, phase_summary(pp), run_end
  Order follows the producer: the embed step runs BEFORE the layer sweep, so the
  chunk timeline precedes the ledger. One pass, not two, because
  --validate-telemetry allows one phase_summary per phase and a --pp-repeat run
  emits one per pass (measured: a second pp summary is rejected as a duplicate),
  so a two-pass fixture could not satisfy both validators.

NEGATIVE COUNTERPARTS live in tools/check-gplr53-embed-chunks.py --selftest
  (bad sum, chunk gap, duplicate chunk, span below setup+compute, off-run event,
  missing ledger): the exe validators cannot judge those, the checker does.

USAGE
  python tools/layer-streaming/gen-gplr53-fixtures.py
  python tools/layer-streaming/gen-gplr53-fixtures.py --trace <recorded.jsonl> \
      --out tools/layer-streaming/tests/gplr53-embed-chunk.jsonl

  The output is deterministic: same source trace in, same bytes out. The
  checked-in fixture was generated from the T-GPLR-52 A2 arm
  `build/llama-cuda/gplr52-a2-50k.jsonl` (see the A2 result block in
  docs/plans/generic-prefill-layer-residency-plan.md).

EXIT CODES
  0 = fixture written
  2 = cannot generate (source trace missing, or it lacks run_start /
      host_ledger / pp phase_summary)
"""
import argparse
import io
import json
import os
import sys

# repo root = third_party/llama.cpp/tools/layer-streaming/../../.. ; --trace paths
# are resolved against the CWD first, then against the root (build/ lives there)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..', '..', '..'))

RUN_ID = 5353001          # fixture run id, distinct from every recorded run
FIXTURE_CHUNKS = 4
CHUNK_TOKENS = 512
PASS1_WEIGHTS = (0.62, 0.18, 0.13, 0.07)   # front-loaded (the A3 discriminator's shape)
# pass-invariant terms (the fixture's own non-embed accounting); preload/output
# mirror the measured magnitudes, `rest` keeps the A1 tool's F1b drift band at 0.
PRELOAD_US = 663.0
OUTPUT_US = 1234.0
REST_US = 372000.0
SWEEP_COMPUTE_US = 2400000.0
SWEEP_WAIT_US = 12000.0
SETUP_US = 20.0
GRAPH_CALL_US = 25.0


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


def pick(events, event, phase=None, need=1):
    out = [e for e in events if e.get('event') == event and (phase is None or e.get('phase') == phase)]
    if len(out) < need:
        return None
    return out


def split_exact(total, weights):
    """Split `total` (us, float) by weights; the sum is exact, not rounded-sum."""
    vals = [round(total * w, 1) for w in weights[:-1]]
    vals.append(round(total - sum(vals), 1))
    return vals


def emit(out_path, run_start, ledger1, summary1):
    spans = split_exact(float(ledger1['embed_us']), PASS1_WEIGHTS)
    pass_embed = round(sum(spans), 1)

    lines = []
    rs = dict(run_start)
    # The pp_chunks manifest is dropped on purpose: with it declared, the strict
    # checker requires a per-chunk act_bind pair per slab, which this fixture
    # (about the embed step, not the slab layout) does not model.
    rs.pop('pp_chunks', None)
    rs.pop('pp_n_chunks', None)
    rs.update({
        'run_id': RUN_ID,
        'pp_tokens': FIXTURE_CHUNKS * CHUNK_TOKENS,
    })
    lines.append(rs)

    t_us = int(run_start.get('t0_us', 1)) + 1000
    for chunk, span in enumerate(spans):
        lines.append({
            'event': 'embed_chunk',
            'run_id': RUN_ID,
            'chunk': chunk,
            'n_tokens': CHUNK_TOKENS,
            'setup_us': SETUP_US,
            'compute_us': round(span - SETUP_US - GRAPH_CALL_US, 1),
            'span_us': span,
            't_us': t_us,
        })
        t_us += int(span) + 500

    led = dict(ledger1)
    led['run_id'] = RUN_ID
    led['t_us'] = t_us
    led['embed_us'] = pass_embed
    led['preload_us'] = PRELOAD_US
    led['output_us'] = OUTPUT_US
    led['loop_us'] = round(-(pass_embed + OUTPUT_US), 1)
    led['route_us'] = 0.0
    led['launches'] = FIXTURE_CHUNKS
    lines.append(led)
    t_us += 100

    sm = dict(summary1)
    sm['run_id'] = RUN_ID
    sm['t_us'] = t_us
    sm['tokens'] = FIXTURE_CHUNKS * CHUNK_TOKENS
    sm['ubatches'] = FIXTURE_CHUNKS
    sm['wall_us'] = int(SWEEP_COMPUTE_US + SWEEP_WAIT_US +
                        pass_embed + PRELOAD_US + OUTPUT_US + REST_US)
    sm['total_compute_us'] = SWEEP_COMPUTE_US
    sm['total_wait_us'] = SWEEP_WAIT_US
    lines.append(sm)
    t_us += 100

    lines.append({
        'event': 'run_end',
        'run_id': RUN_ID,
        'exit_status': 0,
        'peak_vram_bytes': 0,
        'n_tokens_generated': 0,
        'failed': False,
        'error': 'none',
        'total_patch_us': 0.0,
    })

    with io.open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for e in lines:
            f.write(json.dumps(e) + '\n')
    return lines


def main():
    ap = argparse.ArgumentParser(description='emit the T-GPLR-53 A2 embed_chunk fixture')
    ap.add_argument('--trace', default='build/llama-cuda/gplr52-a2-50k.jsonl',
                    help='recorded trace whose metadata and ledger numbers the fixture reuses')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'tests', 'gplr53-embed-chunk.jsonl'))
    args = ap.parse_args()

    trace = next((c for c in (args.trace, os.path.join(REPO_ROOT, args.trace))
                  if os.path.exists(c)), None)
    if trace is None:
        print('cannot generate: source trace not found: %s' % args.trace)
        return 2
    events = load_events(trace)
    run_start = next((e for e in events if e.get('event') == 'run_start'), None)
    ledgers = pick(events, 'host_ledger', need=1)
    summaries = pick(events, 'phase_summary', phase='pp', need=1)
    if run_start is None or ledgers is None or summaries is None:
        print('cannot generate: %s lacks run_start / host_ledger / pp phase_summary' % args.trace)
        return 2

    lines = emit(args.out, run_start, ledgers[0], summaries[0])
    print('wrote %s (%d lines) from %s' % (args.out, len(lines), trace))
    for e in lines:
        if e.get('event') in ('embed_chunk', 'host_ledger'):
            print('  %-11s %s' % (e['event'], json.dumps({k: v for k, v in e.items()
                                                          if k not in ('run_id',)})))
    return 0


if __name__ == '__main__':
    sys.exit(main())
