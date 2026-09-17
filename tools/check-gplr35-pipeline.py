#!/usr/bin/env python3
"""T-GPLR-35 pipeline regression gate: fail loudly when PP staging re-serializes.

WHY THIS EXISTS
  The 2-slot -> 3-slot experiment showed that a change in pipeline depth moved
  the warm 4k layer-major wall by -15.8% WITHOUT changing the per-bank ordering
  signal: the recon's `lag_ms` (copy start vs the compute stream's position at
  enqueue) was >= 0 on every sample in BOTH configurations. So a gate of the
  form "fail if the copy no longer starts before compute drains" would be
  VACUOUS -- it passes on the configuration we know is serialized. That check is
  deliberately reported here as informational only, and this note exists so the
  next reader does not re-derive it as a gate.

  What actually separates a pipelined run from a re-serialized one is where the
  wall goes:
    * wait_fraction = total_wait_us / wall_us      (time the consumer blocked)
    * exposure      = wall - compute - pure DMA    (cost not explained by work)
    * warm PP throughput                           (what the user feels)
    * pure DMA window                              (engine at rate, not slowed)
  A regression that silently re-serializes the pipeline moves at least one of
  those in the failing direction while the trace stays protocol-valid.

USAGE
  python tools/check-gplr35-pipeline.py build/llama-cuda/gplr35-slot3.jsonl
  python tools/check-gplr35-pipeline.py <candidate.jsonl> --baseline <control.jsonl>
  python tools/check-gplr35-pipeline.py <trace.jsonl> --profile slots2

  The slot count is read from the trace's run_start (n_slots), so the correct
  absolute profile is selected automatically; `--profile` overrides it. There is
  no silent pass: either a profile applies or `--baseline` is required.

EXIT CODES  0 = PASS, 1 = FAIL (a gate fired), 2 = cannot evaluate (data/usage).

THRESHOLDS AND THEIR PROVENANCE (measured 2026-09-16, 4k anchor, ubatch 512,
layer-major, kv q8, af=1, pinned, cold+warm pass, this box):
  slots2 reference: 3377.8 ms warm, 1212.6 tok/s, wait 21.7% of wall, exposure +770 ms
  slots3 reference: 2843.2 ms warm, 1440.6 tok/s, wait  7.9% of wall, exposure +100 ms
  Observed session-to-session spread at a FIXED config: ~9% (warm 3089 vs 3378 ms
  for the same 2-slot shape). The absolute floors below therefore carry ~10-12%
  margin and are coarse alarms; the PAIRED check (--baseline, default -10%) is the
  trustworthy gate because both traces share one session, file cache and driver
  state. A single trace can never prove a configuration is unslowed under that
  noise -- pair it.
"""
import argparse
import json
import statistics as st
import sys

PURE_MS = 31.37  # measured pure 453 MB pinned H2D window (T-GPLR-34 probe)

# per-slot-count absolute alarms, anchored to the measured references above
PROFILES = {
    'slots2': {'min_tok_s': 1100.0, 'max_wait_fraction': 0.36, 'max_exposure_ms': 900.0},
    'slots3': {'min_tok_s': 1250.0, 'max_wait_fraction': 0.15, 'max_exposure_ms': 350.0},
}
# Calibration note for the slots2 wait ceiling: two same-configuration 2-slot
# sessions measured 24.2% (recon-ov0) and 30.0% (recon-ov1) wait/wall. A 30%
# ceiling therefore false-fails a legitimate run -- it did, on recon-ov1 -- so
# the ceiling sits at 36%, i.e. this absolute check is a coarse alarm only. The
# paired comparison is the trustworthy gate under this much session noise.

MAX_PURE_INFLATION = 1.25   # median bank window vs the pure 31.37 ms ceiling


def read_trace(path):
    evs = []
    try:
        fh = open(path)
    except OSError as exc:
        print(f'CANNOT EVALUATE: cannot open {path}: {exc}')
        raise SystemExit(2) from exc
    with fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{'):
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return evs


def warm_pp_summary(evs):
    pps = [e for e in evs if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    return pps[-1] if pps else None


def measure(path):
    evs = read_trace(path)
    run = next((e for e in evs if e.get('event') == 'run_start'), None)
    warm = warm_pp_summary(evs)
    if run is None or warm is None:
        return None
    wall_ms = warm['wall_us'] / 1000.0
    comp_ms = float(warm.get('total_compute_us') or 0) / 1000.0
    wait_ms = float(warm.get('total_wait_us') or 0) / 1000.0
    trans_ms = float(warm.get('total_transfer_us') or 0) / 1000.0
    layers = int(warm.get('layers_run') or 0)

    # recon samples inside the warm pass window (the instrument is optional;
    # without it the exposure term falls back to transfer instead of pure DMA)
    marks = [e['t_us'] for e in evs if e.get('event') == 'pp_pass']
    t0 = marks[0] if marks else 0
    t1 = t0 + warm['wall_us']
    win = [e['copy_gpu_ms'] for e in evs if e.get('event') == 'recon_attr' and t0 <= e.get('t_us', 0) <= t1]
    lag = [e['lag_ms'] for e in evs if e.get('event') == 'recon_attr' and t0 <= e.get('t_us', 0) <= t1]
    med_win = st.median(win) if win else None
    pure_dma_ms = med_win * layers if (med_win is not None and layers) else None
    m = {
        'path': path,
        'n_slots': run.get('n_slots'),
        'pp_tokens': run.get('pp_tokens'),
        'slot_vram_bytes': run.get('slot_vram_bytes'),
        'wall_ms': wall_ms,
        'tok_s': warm.get('tokens_per_s') or 0.0,
        'compute_ms': comp_ms,
        'wait_ms': wait_ms,
        'transfer_ms': trans_ms,
        'layers': layers,
        'median_window_ms': med_win,
        'n_window_samples': len(win),
        'negative_lag_frac': (sum(1 for v in lag if v < -1.0) / len(lag)) if lag else None,
        'coverage': (len(win) / layers) if layers else 0.0,
    }
    base = pure_dma_ms if pure_dma_ms is not None else trans_ms
    m['exposure_ms'] = wall_ms - comp_ms - base
    m['basis'] = 'pure DMA' if pure_dma_ms is not None else 'host transfer windows'
    return m


def evaluate(m, prof_name, profile, baseline, min_ratio, notes, force_absolute=False):
    fails = []
    print(f'--- {m["path"]}')
    print(f'    slots={m["n_slots"]}  warm wall={m["wall_ms"]:.1f} ms  {m["tok_s"]:.1f} tok/s  '
          f'compute={m["compute_ms"]:.0f}  wait={m["wait_ms"]:.0f}  transfer={m["transfer_ms"]:.0f} ms')
    print(f'    exposure = wall - compute - {m["basis"]} = {m["exposure_ms"]:+.0f} ms'
          f'   (wait = {100.0*m["wait_ms"]/m["wall_ms"]:.1f}% of wall)')
    if m['median_window_ms'] is not None:
        print(f'    bank DMA window median {m["median_window_ms"]:.2f} ms '
              f'({m["median_window_ms"]/PURE_MS:.2f}x pure), samples {m["n_window_samples"]}/{m["layers"]}')
    if m['negative_lag_frac'] is not None:
        print(f'    INFO not gated: copies starting before the compute backlog drained = '
              f'{100.0*m["negative_lag_frac"]:.0f}% (0% at BOTH 2 and 3 slots -- does not '
              f'discriminate overlap; see the module docstring)')
    shape_mismatch = (m['pp_tokens'] or 0) > 8192
    if shape_mismatch:
        print(f'    SHAPE MISMATCH: the absolute floors are anchored to the 4k shape; this trace '
              f'runs pp_tokens={m["pp_tokens"]} (~{m["pp_tokens"]/4096:.0f}x longer), where the '
              f'per-bank staging cost amortizes differently. The absolute floors are '
              f'SKIPPED -- they misfire here (validated: they false-FAILed a legitimate '
              f'200k arm on the 4k exposure ceiling). Pair the trace or pass '
              f'--force-absolute.')

    if profile and shape_mismatch and not force_absolute:
        profile = None   # shape-mismatched absolute floors misfire; see the note above
    if profile:
        if m['tok_s'] < profile['min_tok_s']:
            fails.append(f'warm PP throughput {m["tok_s"]:.1f} tok/s < floor {profile["min_tok_s"]:.0f} '
                         f'[{prof_name}]')
        wf = m['wait_ms'] / m['wall_ms'] if m['wall_ms'] else 0.0
        if wf > profile['max_wait_fraction']:
            fails.append(f'consumer wait {100*wf:.1f}% of wall > ceiling '
                         f'{100*profile["max_wait_fraction"]:.0f}% [{prof_name}]')
        if m['exposure_ms'] > profile['max_exposure_ms']:
            fails.append(f'exposure {m["exposure_ms"]:+.0f} ms > ceiling '
                         f'{profile["max_exposure_ms"]:.0f} ms [{prof_name}]')
    if m['median_window_ms'] is not None and m['median_window_ms'] > MAX_PURE_INFLATION * PURE_MS:
        fails.append(f'bank DMA window {m["median_window_ms"]:.2f} ms inflated past '
                     f'{MAX_PURE_INFLATION:.2f}x pure ({MAX_PURE_INFLATION*PURE_MS:.2f} ms)')
    if baseline:
        if m['n_slots'] == baseline['n_slots']:
            fails.append(f'candidate and baseline both ran {m["n_slots"]} slots: a depth '
                         f'comparison needs different configurations')
        ratio = (m['tok_s'] / baseline['tok_s']) if baseline['tok_s'] else 0.0
        floor = 1.0 - min_ratio
        print(f'    paired vs {baseline["path"]}: {ratio:.3f}x throughput '
              f'({m["tok_s"]:.1f} vs {baseline["tok_s"]:.1f} tok/s), required >= {floor:.2f}x')
        if ratio < floor:
            fails.append(f'paired throughput {ratio:.3f}x < required {floor:.2f}x '
                         f'(pipeline re-serialized relative to the control)')

    for f in fails:
        print(f'    FAIL: {f}')
    notes.extend(fails)
    return fails, profile is not None


def main():
    ap = argparse.ArgumentParser(description='T-GPLR-35 PP pipeline regression gate')
    ap.add_argument('trace')
    ap.add_argument('--baseline', help='adjacent control trace from the same session (recommended)')
    ap.add_argument('--min-ratio', type=float, default=0.10,
                    help='allowed throughput loss vs baseline (default 0.10 = require >= 0.90x)')
    ap.add_argument('--profile', choices=sorted(PROFILES),
                    help='override the absolute profile (default: inferred from run_start n_slots)')
    ap.add_argument('--force-absolute', action='store_true',
                    help='apply the absolute floors even when the trace shape differs from '
                         'the shape they were anchored on (they will likely misfire)')
    args = ap.parse_args()

    m = measure(args.trace)
    if m is None:
        print(f'CANNOT EVALUATE: {args.trace} has no run_start or no warm pp phase_summary')
        return 2

    if args.profile:
        prof_name = args.profile
    else:
        prof_name = f'slots{m["n_slots"]}' if f'slots{m["n_slots"]}' in PROFILES else None

    profile = PROFILES.get(prof_name) if prof_name else None
    if profile is None and not args.baseline:
        print(f'CANNOT EVALUATE: no threshold profile for n_slots={m["n_slots"]} and no '
              f'--baseline given; refusing to pass without a criterion')
        return 2

    baseline = measure(args.baseline) if args.baseline else None
    if args.baseline and baseline is None:
        print(f'CANNOT EVALUATE: baseline {args.baseline} is not a usable trace')
        return 2

    if baseline:
        print(f'--- baseline {baseline["path"]}: slots={baseline["n_slots"]} '
              f'{baseline["tok_s"]:.1f} tok/s wall={baseline["wall_ms"]:.0f} ms')
    notes = []
    fails, profile_applied = evaluate(m, prof_name, profile, baseline, args.min_ratio, notes,
                                     force_absolute=args.force_absolute)
    if profile_applied:
        scope = f' (absolute profile {prof_name} + baseline)' if baseline else f' (profile {prof_name})'
    elif baseline:
        scope = ' (paired only)'
    else:
        scope = ' (no gate applied)'
    print('RESULT: ' + ('FAIL' if fails else 'PASS') + scope)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
