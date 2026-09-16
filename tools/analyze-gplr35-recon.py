#!/usr/bin/env python3
"""T-GPLR-35 recon analyzer: attribute the per-bank copy/compute coupling.

Reads the recon telemetry produced by --pp-recon 1 (see tools/run-gplr35-recon.ps1):

  recon_identity  one per run: the stream handles actually used, so the numbers
                  below are tied to a verified topology.
  recon_attr      one per staged bank:
                    copy_gpu_ms  the copy's own GPU window (start -> end)
                    lag_ms       copy_start - compute_ref, i.e. how long AFTER
                                 the compute stream's backlog at enqueue time
                                 the DMA actually started. NEGATIVE = the DMA
                                 began while that backlog was still draining
                                 (overlap). >= 0 = the DMA waited for compute.

Verdict classes (pre-registered in the plan, T-GPLR-35 design contract):
  ORDERING      ov0 lag >= ~compute time with a pure copy window; ov1 lag < 0
                -> the coupling is the staging/compute stream topology.
  RATE          copy_gpu_ms ~2x pure (~65-75 ms) in either arm
                -> the coupling is DMA rate sharing, not ordering.
  BOTH          lag >= 0 AND a stretched window
                -> ordering plus rate contention.
  NONE          ov1 shows no change in either metric -> coupling is elsewhere.
Pure-window reference for this box (measured, T-GPLR-34 probe): 453 MB pinned
H2D = 31.37 ms min (~14.4 GB/s).
"""
import json
import statistics as st
import sys
from collections import defaultdict

PURE_MS = 31.37  # measured pure 453 MB pinned H2D window (T-GPLR-34 probe)


def load(path):
    ident, attrs = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = e.get('event')
            if ev == 'recon_identity':
                ident.append(e)
            elif ev == 'recon_attr':
                attrs.append(e)
    return ident, attrs


def pct(vals, q):
    if not vals:
        return float('nan')
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def ledger(arm, path):
    """Warm-pass cost ledger: where the wall actually goes, in named buckets.

    transfer_us (host window per bank) = host_submit_us (pinned hop + submit)
    + ready_latency_us (submit -> completion). The recon's copy_gpu_ms is the
    DMA's own GPU window, which is what separates "the engine is slow" from
    "the engine was busy / we submitted late".
    """
    import json as _json
    evs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{'):
                try:
                    evs.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
    marks = [e['t_us'] for e in evs if e.get('event') == 'pp_pass']
    pps = [e for e in evs if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    if not marks or not pps:
        return
    t0 = marks[0]
    warm = pps[-1]
    wall_ms = warm['wall_us'] / 1000.0
    t1 = t0 + warm['wall_us']
    host = ready = trans = comp = wait = 0.0
    banks = 0
    cg = []
    for e in evs:
        ev, t = e.get('event'), e.get('t_us', 0)
        if e.get('phase') == 0 and t0 <= t <= t1:
            if ev == 'copy_complete':
                trans += float(e.get('transfer_us') or 0) / 1000.0
                host += float(e.get('host_submit_us') or 0) / 1000.0
                ready += float(e.get('ready_latency_us') or 0) / 1000.0
                banks += 1
            elif ev == 'layer_compute':
                comp += float(e.get('compute_us') or 0) / 1000.0
            elif ev == 'layer_ready_wait':
                wait += float(e.get('wait_us') or 0) / 1000.0
        if ev == 'recon_attr' and t0 <= t <= t1:
            cg.append(e['copy_gpu_ms'])
    # summary totals are authoritative for compute/transfer/wait (the per-event
    # sums below are scoped to the pass window and undercount at its edges)
    s_comp = float(warm.get('total_compute_us') or 0) / 1000.0
    s_trans = float(warm.get('total_transfer_us') or 0) / 1000.0
    s_wait = float(warm.get('total_wait_us') or 0) / 1000.0
    print(f'  -- warm-pass ledger (wall {wall_ms:.0f} ms, {banks} banks, '
          f'{warm["tokens"]/(wall_ms/1000):.0f} tok/s, evictions {warm["evictions"]})')
    print(f'     compute (summary) = {s_comp:6.0f} ms | transfer (summary) = {s_trans:6.0f} ms'
          f' | consumer wait (summary) = {s_wait:6.0f} ms')
    print(f'     of the transfer: host hop (pinned hop + submit) = {host:6.0f} ms,'
          f' ready latency = {ready:6.0f} ms  [{banks} banks]')
    if cg:
        print(f'     pure DMA window sum = {sum(cg):6.0f} ms (median {st.median(cg):.1f} ms, n={len(cg)})'
              f' | engine queue wait (ready - pure DMA) = {ready - sum(cg):6.0f} ms')
        print(f'     wall - (compute + pure DMA) = {wall_ms - s_comp - sum(cg):+.0f} ms'
              f'  => the transfer/queue/CPU cost that is NOT the DMA engine itself')


def summarize(arm, path):
    ident, attrs = load(path)
    print(f'=== {arm} ===')
    if not ident:
        print('  NO recon_identity event: the arm did not run with --pp-recon 1')
        return None
    for e in ident:
        print(f"  identity: overlap={e.get('overlap')} compute_s0={e.get('compute_s0')} "
              f"compute_s1={e.get('compute_s1')} copy_s0={e.get('copy_s0')} "
              f"same_ctx_stream={e.get('same_ctx_stream')}")
    if not attrs:
        print('  NO recon_attr samples (events never completed? run too short?)')
        return None
    cg = [e['copy_gpu_ms'] for e in attrs]
    lag = [e['lag_ms'] for e in attrs]
    print(f'  samples={len(attrs)}  banners/unique layers={len(set(e["layer"] for e in attrs))}')
    print(f'  copy_gpu_ms  min={min(cg):7.2f} p10={pct(cg,0.10):7.2f} '
          f'median={st.median(cg):7.2f} p90={pct(cg,0.90):7.2f} max={max(cg):7.2f} '
          f'(pure={PURE_MS})')
    print(f'  lag_ms       min={min(lag):8.2f} p10={pct(lag,0.10):8.2f} '
          f'median={st.median(lag):8.2f} p90={pct(lag,0.90):8.2f} max={max(lag):8.2f}')
    neg = sum(1 for v in lag if v < -1.0)
    print(f'  lag < -1 ms (DMA started while compute backlog was draining): {neg}/{len(lag)} '
          f'({100.0*neg/len(lag):.0f}%)')
    # per-layer attach: median lag by layer, to expose a systematic pattern
    by_layer = defaultdict(list)
    for e in attrs:
        by_layer[e['layer']].append(e['lag_ms'])
    med = {k: st.median(v) for k, v in by_layer.items()}
    worst = sorted(med.items(), key=lambda kv: -kv[1])[:5]
    print('  layers with the largest median lag: ' +
          ', '.join(f'L{k}={v:.1f}' for k, v in worst))
    print(f'  copy window inflation vs pure: {st.median(cg)/PURE_MS:.2f}x')
    ledger(arm, path)
    return {'cg_med': st.median(cg), 'lag_med': st.median(lag),
            'frac_overlap': neg / len(lag), 'n': len(attrs)}


def verdict(ov0, ov1):
    print('=== attribution ===')
    if ov0 is None or ov1 is None:
        print('  inconclusive: an arm produced no usable samples')
        return 'INCONCLUSIVE'
    ordered0 = ov0['lag_med'] >= 5.0
    rate0 = ov0['cg_med'] >= 1.5 * PURE_MS
    rate1 = ov1['cg_med'] >= 1.5 * PURE_MS
    decoupled1 = ov1['lag_med'] < -1.0 and ov1['frac_overlap'] > 0.5
    print(f"  ov0: median lag {ov0['lag_med']:+.1f} ms, window {ov0['cg_med']:.1f} ms "
          f"({ov0['cg_med']/PURE_MS:.2f}x pure), overlap {100*ov0['frac_overlap']:.0f}%")
    print(f"  ov1: median lag {ov1['lag_med']:+.1f} ms, window {ov1['cg_med']:.1f} ms "
          f"({ov1['cg_med']/PURE_MS:.2f}x pure), overlap {100*ov1['frac_overlap']:.0f}%")
    if ordered0 and not rate0 and decoupled1 and not rate1:
        v = 'ORDERING (stream/context topology)'
    elif rate0 and ordered0:
        v = 'BOTH (ordering + rate contention)'
    elif rate0:
        v = 'RATE (DMA rate sharing, not ordering)'
    elif decoupled1:
        v = 'ORDERING-PARTIAL (aux stream decouples the start, rate unaffected)'
    else:
        v = 'NONE (neither arm explains the chaining; coupling is elsewhere)'
    print(f'  verdict: {v}')
    return v


if __name__ == '__main__':
    base = 'build/llama-cuda/'
    arms = sys.argv[1:] or ['recon-ov0', 'recon-ov1']
    res = {}
    for arm in arms:
        res[arm] = summarize(arm, f'{base}gplr35-{arm}.jsonl')
    if len(arms) == 2:
        verdict(res[arms[0]], res[arms[1]])
