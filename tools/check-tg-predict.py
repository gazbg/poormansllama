#!/usr/bin/env python3
"""T-GPLR-58 A2: the TG expert-prediction VERDICT SURFACE (host-only).

Reads recorded layer-streaming traces and reports the quantities T-GPLR-58
pre-registered, or REFUSES the leg.  No GPU, no mutex, no model.

What a leg must be
------------------
A leg is one recorded run's JSONL trace that actually exercised the per-layer row
cache.  The checker reads the RUN'S OWN RESOLVED STATE from the trace, never from a
filename or a command line:

  rowset_init   retain_k / top_k / n_expert  -- the declared retention and router shape
  run_start     row_spec_mode / _budget / _reserve  -- present iff speculation ran
  rowset_fetch  one per (ubatch, layer) demand stage: the TG demand stream
  spec          one per layer per stage: the P1-P6 counters and M7's lead histogram
  byte_account  the VRAM decomposition
  run_end       peak_vram_bytes, exit_status

Roles are declared with --leg (or inferred from the filename stem, and the
inference is echoed).  They decide which bands can be computed:

  off      no prediction: the baseline every band is measured against
  arm      a predictor arm to be judged
  null     the count-matched, structure-free control (pool growth)
  retref   a retention reference leg (the VRAM band's denominator)

A role is a SESSION DECLARATION, not a filename property: `off`/`null`/`retref` are
inferred from a path only when the name says so unambiguously, and `arm` is NEVER
inferred -- this repository names every A/B leg `arm-A1.jsonl` / `arm-B1.jsonl`, so
inferring a predictor arm from a name would turn an ordinary recorded run into a
claim.  `--leg ROLE=PATH` pins a role; a bare `--trace PATH` scores the leg's own
evidence and leaves the role bands NOT COMPUTABLE rather than guessing them.

Bands (as pre-registered in the block; NOT re-derived here)
-----------------------------------------------------------
  share            hits / demands, over `rowset_fetch` rows: engine booking only
  margin           1 - share (the room a predictor could win)
  M2               share(null) - share(off): the pool-growth share of a merged set
  M4               margin_share(arm) = (share(arm) - share(off)) / (1 - share(off))
  M4n              share(arm) - share(null): the NULL-CORRECTED gain (structure only)
  M3               the same session's arm@K2 share against its off@K4 share
  M5               staged-volume growth: bytes_staged(arm) / bytes_staged(off)
  M7a              lead_min >= 2 over CONSUMED fills: the reap-ordering floor
  M7b              the late-fill rate matched/(consumed+matched) <= 0.5 on an ARM
                   leg with >= 8 resolved fills (the 2026-09-23 restatement)
  M7c              lead_rows min/median/max over CONSUMED fills, plus the
                   8-bucket histogram: REPORTED, never gated

M7's decision rule (RESTATED 2026-09-23, because the original band was vacuous).
The engine's only hot-path reap runs at the END of a demand stage, so a fill issued
at the end of stage t first counts as READY at the end of stage t+1 and cannot serve
a demand before stage t+2: a lead below 2 is IMPOSSIBLE, and this checker treats it
as an integrity failure (M7a) rather than a lucky schedule.  A median criterion is
vacuous in BOTH forms -- `> 0` and `>= 2` -- because the floor already forces every
consumed fill's lead to 2 or more; the old wording was also SILENT on the worst case,
a leg that consumed nothing at all.  The signal M7 actually carries is the balance
between `consumed` (fills that landed in time) and `matched` (fills whose own row was
demanded while they were still in flight -- the guess arrived after its demand), and
that balance is the M7b band, scored on ARM legs only.

Invariants (falsifiers).  A FAIL here blocks REPORTING a task complete; it never
authorizes anything.  Every gate in this repository is soft.
  P1  spec_waited_by_demand == 0        a demand never waited on a speculative copy
  P2  the fill accounting closes        submitted >= consumed + cancelled + live
  P4  cancelled == gen + demand + capacity   every cancellation names its firing point
  P6  an off leg carries no speculation  the default path is unpolluted
  M7a lead_min >= 2 on a consumed leg    the reap ordering cannot be beaten

Refusals (exit 2, NO numbers printed -- a refused leg is not a scored leg)
------------------------------------------------------------------------
  file-missing | not-a-trace | no-retention (no rowset_init) | no-retain-k |
  no-rowset-fetch | arm-without-spec (a leg declared arm/null emitted no `spec`) |
  spec-missing-keys (incl. a trace whose `spec` lacks `cancelled`) |
  cancel-hook-never-fired (submitted > 0 with cancelled == 0: P4 is UNTESTED, so the
  leg is refused rather than scored -- this is the A2 acceptance fixture) |
  off-leg-with-speculation (role/label conflict) |
  accounting-negative (submitted < consumed + cancelled + live: double counting) |
  lead-inconsistent (lead_hist and `consumed` disagree, or the sum is unreachable)

usage:
  python tools/check-tg-predict.py --selftest
  python tools/check-tg-predict.py --leg off=a.jsonl --leg arm=b.jsonl --leg null=c.jsonl
  python tools/check-tg-predict.py --json out.json --leg arm=b.jsonl --leg off=a.jsonl
  python tools/check-tg-predict.py --trace some-leg.jsonl      # role inferred, bands need --leg
"""

import argparse
import contextlib
import io
import json
import os
import re
import statistics
import sys
import tempfile

EXIT_OK, EXIT_INVARIANT, EXIT_REFUSED = 0, 1, 2

# M7b -- the late-fill rate band (2026-09-23 M7 restatement).  A median criterion is
# vacuous in both forms (`> 0` and `>= 2`) because the reap ordering forces lead >= 2,
# so the band reads the one datum that can fail: how often a guess was still in flight
# when its own row was demanded (the fill is cancelled and the demand fetches its own
# copy, so that guess is paid for twice and returns nothing).
M7B_MAX_RATE = 0.5        # matched/(consumed+matched): more land in time than arrive late
M7B_MIN_RESOLVED = 8      # resolved fills required before the rate is SCORED at all

# `spec` keys the checker requires on an arm/null leg.  `cancelled` is the one the
# block names; the rest are what makes the counters interpretable at all.
SPEC_REQUIRED = ('submitted', 'ready', 'cancelled', 'cancelled_capacity',
                 'cancelled_demand', 'cancelled_generation', 'refused_capacity',
                 'dup_refused', 'matched_by_demand', 'waited_by_demand',
                 'consumed', 'lead_sum', 'lead_min', 'lead_max', 'lead_hist')


class Refused(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- read

def read_leg(path, role):
    """Parse one trace into a leg dict, or raise Refused.  Never returns a number
    it could not justify: every refusal below is a reason NOT to score the leg."""
    if not os.path.isfile(path):
        raise Refused('file-missing', 'no such trace file: %s' % path)

    leg = {
        'path': path, 'role': role, 'events': 0,
        'init': None, 'start': None, 'end': None, 'account': None,
        'hits': 0, 'fetched': 0, 'stages': 0, 'phases': {},
        'staged_bytes': 0, 'bank_bytes': 0, 'retired': 0,
        'xfer': [], 'region_union': 0,
        'spec_events': 0, 'spec': {}, 'lead_hist': None, 'spec_layers': {},
        'refused_events': [],
        'phase_summary': {},
    }

    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ev = d.get('event')
            if ev is None:
                continue
            leg['events'] += 1
            if ev == 'rowset_init':
                leg['init'] = d
            elif ev == 'run_start':
                leg['start'] = d
            elif ev == 'run_end':
                leg['end'] = d
            elif ev == 'byte_account':
                leg['account'] = d
            elif ev == 'rowset_refused':
                leg['refused_events'].append(d)
            elif ev == 'rowset_fetch':
                leg['stages'] += 1
                leg['hits'] += int(d.get('n_rows_resident', 0) or 0)
                leg['fetched'] += int(d.get('n_rows_fetched', 0) or 0)
                leg['staged_bytes'] += int(d.get('bytes_staged', 0) or 0)
                leg['bank_bytes'] += int(d.get('bytes_full_bank', 0) or 0)
                leg['retired'] += int(d.get('n_rows_retired', 0) or 0)
                ph = d.get('phase')
                leg['phases'][str(ph)] = leg['phases'].get(str(ph), 0) + 1
                if d.get('route_union') is not None:
                    leg['region_union'] = max(leg['region_union'], int(d['route_union']))
                if d.get('transfer_us') is not None:
                    leg['xfer'].append(float(d['transfer_us']))
            elif ev == 'spec':
                leg['spec_events'] += 1
                for k in SPEC_REQUIRED:
                    if k in d:
                        if k == 'lead_hist':
                            cur = leg['spec'].get(k)
                            if cur is None or sum(d[k]) >= sum(cur):
                                leg['spec'][k] = list(d[k])
                        elif dfn(leg['spec'].get(k), d[k]):
                            leg['spec'][k] = d[k]
                    elif k not in leg['spec']:
                        leg['spec'][k] = None
                lay = d.get('layer')
                if lay is not None:
                    leg['spec_layers'][int(lay)] = d
                for k in ('mode', 'budget', 'reserve', 'cap'):
                    if k in d:
                        leg['spec'][k] = d[k]
            elif ev == 'phase_summary':
                leg['phase_summary'][str(d.get('phase'))] = d

    if leg['events'] == 0:
        raise Refused('not-a-trace', 'no JSON telemetry events in %s' % path)

    # -- R: instrumented at all? ------------------------------------------------
    if leg['init'] is None:
        raise Refused('no-retention',
                      'no rowset_init: this trace never armed the per-layer row cache, '
                      'so it has no TG row set to score (%s)' % path)
    rk = leg['init'].get('retain_k')
    if rk is None:
        raise Refused('no-retain-k',
                      'rowset_init carries no retain_k: the retention depth is unknown, so '
                      'no band can be attributed to a K (%s)' % path)
    rk = int(rk)
    if rk <= 0:
        raise Refused('no-retain-k',
                      'rowset_init retain_k = %d: the REACTIVE baseline has no per-layer '
                      'residency and therefore no margin to score (%s)' % (rk, path))
    if leg['stages'] == 0:
        raise Refused('no-rowset-fetch',
                      'no rowset_fetch rows: the run never executed a TG demand stage (%s)' % path)

    leg['retain_k'] = rk
    leg['top_k'] = int(leg['init'].get('top_k', 0) or 0)
    leg['n_expert'] = int(leg['init'].get('n_expert', 0) or 0)
    leg['demands'] = leg['hits'] + leg['fetched']
    leg['share'] = (leg['hits'] / leg['demands']) if leg['demands'] else 0.0
    leg['margin'] = 1.0 - leg['share']

    # -- R: what the RUN says it resolved (not what a filename implies) ---------
    start = leg['start'] or {}
    spec_mode = int(start.get('row_spec_mode', 0) or 0)
    leg['run_spec_mode'] = spec_mode
    claims_prediction = bool(spec_mode > 0 or leg['spec_events'] > 0)

    if role == 'off' and claims_prediction:
        raise Refused('off-leg-with-speculation',
                      'declared off but the run resolved row_spec_mode=%d with %d spec '
                      'event(s): the baseline leg carries speculation, so it cannot serve as '
                      'the no-prediction control (%s)' % (spec_mode, leg['spec_events'], path))

    if role in ('arm', 'null'):
        if leg['spec_events'] == 0:
            raise Refused('arm-without-spec',
                          'declared %s but the trace contains no `spec` events: the runner did '
                          'not run the instrument (or the mode was refused before the model '
                          'load), so there is nothing to score (%s)' % (role, path))
        missing = [k for k in SPEC_REQUIRED if leg['spec'].get(k) is None]
        if missing:
            raise Refused('spec-missing-keys',
                          'instrumented leg is missing required spec keys: %s. A leg without '
                          'them cannot be scored (a trace whose `spec` lacks `cancelled` is '
                          'exactly this case: P4 is unverifiable) (%s)' % (', '.join(missing), path))

    if leg['spec_events']:
        missing = [k for k in SPEC_REQUIRED if leg['spec'].get(k) is None]
        if missing:
            raise Refused('spec-missing-keys',
                          'instrumented leg is missing required spec keys: %s (%s)'
                          % (', '.join(missing), path))
        s = leg['spec']
        leg['lead_hist'] = [int(x) for x in s['lead_hist']]
        if len(leg['lead_hist']) != 8:
            raise Refused('lead-inconsistent',
                          'lead_hist has %d buckets, expected 8 (%s)' % (len(leg['lead_hist']), path))
        consumed = int(s['consumed'])
        if sum(leg['lead_hist']) != consumed:
            raise Refused('lead-inconsistent',
                          'lead_hist sums to %d but consumed = %d: the histogram and the counter '
                          'disagree, so neither can be scored (%s)'
                          % (sum(leg['lead_hist']), consumed, path))
        lo, hi, tot = int(s['lead_min']), int(s['lead_max']), int(s['lead_sum'])
        if consumed == 0 and (lo or hi or tot or sum(leg['lead_hist'])):
            raise Refused('lead-inconsistent',
                          'no fill was consumed, yet lead_min/max/sum are %d/%d/%d (%s)'
                          % (lo, hi, tot, path))
        if consumed > 0:
            if not (consumed * lo <= tot <= consumed * hi):
                raise Refused('lead-inconsistent',
                              'lead_sum %d is outside [consumed*lead_min, consumed*lead_max] = '
                              '[%d, %d] (%s)' % (tot, consumed * lo, consumed * hi, path))
            if leg['lead_hist'][min(hi, 8) - 1] <= 0:
                raise Refused('lead-inconsistent',
                              'lead_max = %d but its histogram bucket is empty (%s)' % (hi, path))
            if leg['lead_hist'][min(lo, 8) - 1] <= 0:
                raise Refused('lead-inconsistent',
                              'lead_min = %d but its histogram bucket is empty (%s)' % (lo, path))

        submitted = int(s['submitted'])
        cancelled = int(s['cancelled'])
        live = sum(int(d.get('pending', 0) or 0) for d in leg['spec_layers'].values()) + \
               sum(int(d.get('ready_spec', 0) or 0) for d in leg['spec_layers'].values())
        if submitted < consumed + cancelled + live:
            raise Refused('accounting-negative',
                          'submitted %d < consumed %d + cancelled %d + live %d: a fill is being '
                          'counted twice, so the leg is not scoreable (%s)'
                          % (submitted, consumed, cancelled, live, path))
        leg['live_at_end'] = live
        leg['residual'] = submitted - consumed - cancelled - live

        if submitted > 0 and cancelled == 0:
            raise Refused('cancel-hook-never-fired',
                          'submitted %d fill(s) with cancelled = 0: the CANCELLATION hook never '
                          'fired in this leg, so P4 is UNTESTED -- the leg is refused rather '
                          'than scored. A leg whose cancel path never runs cannot testify that '
                          'it works (%s)' % (submitted, path))
    else:
        if role in ('arm', 'null'):
            raise Refused('arm-without-spec', 'no speculation telemetry in %s' % path)

    leg['lead_median'] = None
    if leg['spec_events']:
        consumed = int(leg['spec']['consumed'])
        if consumed > 0:
            half = (consumed + 1) // 2
            cum = 0
            for b, n in enumerate(leg['lead_hist']):
                cum += n
                if cum >= half:
                    leg['lead_median'] = b + 1
                    break

    if leg['xfer']:
        leg['xfer_p50'] = statistics.median(leg['xfer'])
        srt = sorted(leg['xfer'])
        leg['xfer_p95'] = srt[min(len(srt) - 1, int(round(0.95 * (len(srt) - 1))))]
    else:
        leg['xfer_p50'] = leg['xfer_p95'] = None
    return leg


def dfn(a, b):
    """True when b is 'more' than a (counters in the trace are cumulative, so the
    maximum seen is the run's total; a snapshot field that appears once is kept)."""
    if a is None:
        return True
    try:
        return float(b) > float(a)
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------ report

def mib(n):
    return '%.0f' % (float(n) / (1024.0 * 1024.0)) if n is not None else '-'


def leg_line(leg):
    return ('  %-6s %-44s mode=%d K=%-2d stages=%-5d share=%.4f margin=%.4f'
            % (leg['role'], os.path.basename(os.path.dirname(leg['path'])) + '/' +
               os.path.basename(leg['path']), leg['run_spec_mode'], leg['retain_k'],
               leg['stages'], leg['share'], leg['margin']))


def report(legs, session):
    out = []
    out.append('T-GPLR-58 A2 - TG expert-prediction verdict surface (host-only)')
    if session:
        out.append('session: %s' % session)
    out.append('')
    out.append('LEGS (mode and K read from each trace, never from its name)')
    for leg in legs:
        out.append(leg_line(leg))

    out.append('')
    out.append('PER-LEG EVIDENCE')
    out.append('  %-6s %-13s %-11s %-9s %-9s %-11s %-11s'
               % ('role', 'staged_GiB', 'bank_GiB', 'xfer_p50', 'xfer_p95', 'peak_VRAM_MiB',
                  'rowcache_MiB'))
    for leg in legs:
        acct = leg['account'] or {}
        end = leg['end'] or {}
        out.append('  %-6s %-13s %-11s %-9s %-9s %-11s %-11s'
                   % (leg['role'],
                      '%.3f' % (leg['staged_bytes'] / 2 ** 30),
                      '%.3f' % (leg['bank_bytes'] / 2 ** 30),
                      '%.0f' % leg['xfer_p50'] if leg['xfer_p50'] is not None else '-',
                      '%.0f' % leg['xfer_p95'] if leg['xfer_p95'] is not None else '-',
                      mib(end.get('peak_vram_bytes')),
                      mib(acct.get('row_cache'))))
    for leg in legs:
        if leg['refused_events']:
            out.append('  note: %s recorded %d loud rowset_refused event(s): %s'
                       % (os.path.basename(leg['path']), len(leg['refused_events']),
                          ', '.join(sorted({str(e.get('reason')) for e in leg['refused_events']}))))

    spec_legs = [l for l in legs if l['spec_events']]
    if spec_legs:
        out.append('')
        out.append('SPECULATION (P1-P6 evidence; `matched` is the fallback, not a lead)')
        out.append('  %-6s %-10s %-9s %-8s %-9s %-5s %-7s %-9s %-8s %-7s %-7s %-9s %-5s %-5s %-5s'
                   % ('role', 'submitted', 'consumed', 'matched', 'cancelled', 'gen', 'demand',
                      'capacity', 'refused', 'dup_ref', 'waited', 'lead_min', 'med', 'max', 'resid'))
        for leg in spec_legs:
            s = leg['spec']
            out.append('  %-6s %-10d %-9d %-8d %-9d %-5d %-7d %-9d %-8d %-7d %-7d %-9d %-5s %-5s %-5d'
                       % (leg['role'], int(s['submitted']), int(s['consumed']),
                          int(s['matched_by_demand']), int(s['cancelled']),
                          int(s['cancelled_generation']), int(s['cancelled_demand']),
                          int(s['cancelled_capacity']), int(s['refused_capacity']),
                          int(s['dup_refused']), int(s['waited_by_demand']),
                          int(s['lead_min']),
                          leg['lead_median'] if leg['lead_median'] is not None else '-',
                          int(s['lead_max']), leg['residual']))
        out.append('  lead_hist (bucket i = lead i+1 stages, bucket 7 = lead >= 8):')
        for leg in spec_legs:
            out.append('    %-6s %s' % (leg['role'], leg['lead_hist']))

    offs = [l for l in legs if l['role'] == 'off']
    arms = [l for l in legs if l['role'] == 'arm']
    nulls = [l for l in legs if l['role'] == 'null']
    rets = [l for l in legs if l['role'] == 'retref']

    out.append('')
    out.append('BANDS (pre-registered in T-GPLR-58; computed, never re-derived here)')
    if offs:
        base = sum(l['share'] for l in offs) / len(offs)
        out.append('  off baseline share (n=%d)                       : %.4f' % (len(offs), base))
    else:
        base = None
        out.append('  off baseline share                             : NOT COMPUTABLE '
                   '(no off leg supplied)')

    if nulls and base is not None:
        for nl in nulls:
            out.append('  M2  null-vs-off pool-growth share (null K=%d)  : %.4f - %.4f = %+.4f'
                       % (nl['retain_k'], nl['share'], base, nl['share'] - base))
    for al in arms:
        if base is not None:
            gain = al['share'] - base
            ms = gain / (1.0 - base) if base < 1.0 else float('nan')
            out.append('  M4  margin_share(arm K=%d vs off)             : (%0.4f-%0.4f)/(1-%0.4f) = %+.4f'
                       % (al['retain_k'], al['share'], base, base, ms))
            if len(nulls) == 1:
                out.append('  M4n null-corrected gain (structure only)       : %.4f - %.4f = %+.4f'
                           % (al['share'], nulls[0]['share'], al['share'] - nulls[0]['share']))
            elif len(nulls) > 1:
                out.append('  M4n null-corrected gain                        : ambiguous '
                           '(%d null legs supplied; pair the null explicitly)' % len(nulls))
            else:
                out.append('  M4n null-corrected gain                        : NOT COMPUTABLE '
                           '(no null leg: a raw gain cannot be separated from pool growth)')

    k2_arms = [l for l in arms if l['retain_k'] == 2]
    k4_offs = [l for l in offs if l['retain_k'] == 4]
    if k2_arms and k4_offs:
        out.append('  M3  same-session arm@K2 vs off@K4               : %.4f vs %.4f = %+.4f'
                   % (k2_arms[0]['share'], k4_offs[0]['share'],
                      k2_arms[0]['share'] - k4_offs[0]['share']))
    else:
        out.append('  M3  same-session arm@K2 vs off@K4               : NOT COMPUTABLE from these '
                   'legs (needs an arm at K=2 and THE SAME session\'s off at K=4)')

    if arms and offs:
        a_staged = sum(l['staged_bytes'] for l in arms)
        o_staged = sum(l['staged_bytes'] for l in offs)
        if o_staged:
            out.append('  M5  staged-volume growth (arm/off)              : %.3fx (%+.1f%%)  '
                       '[soft band <= +5%%]' % (a_staged / o_staged, 100.0 * (a_staged / o_staged - 1.0)))

    lead_legs = [l for l in spec_legs if l['lead_median'] is not None]
    if lead_legs:
        for lg in lead_legs:
            s = lg['spec']
            cons, matched = int(s['consumed']), int(s['matched_by_demand'])
            out.append('  M7c lead_rows min/median/max (%s)        : %d / %d / %d  '
                       '(consumed %d, matched-in-flight %d = %.1f%% of guesses)'
                       % (lg['role'], int(s['lead_min']), lg['lead_median'], int(s['lead_max']),
                          cons, matched, 100.0 * matched / max(1, cons + matched)))
        out.append('      note: the only hot-path reap runs at the END of a stage, so the '
                   'structural floor is lead 2 -- a guess aimed at the very next stage of its '
                   'layer arrives after that stage has already been classified.')
    elif spec_legs:
        out.append('  M7c lead_rows min/median/max                    : UNMEASURED '
                   '(no fill was consumed in any supplied leg)')

    # --- M7b: the late-fill rate (an ARM band; the 2026-09-23 M7 restatement) -----
    # Scored on `arm` legs only: the count-matched `null` control draws its ids with no
    # signal, so gating its targeting would fail the control BY CONSTRUCTION.
    arm_rates, band_fails = [], []
    for lg in spec_legs:
        s = lg['spec']
        cons, matched = int(s['consumed']), int(s['matched_by_demand'])
        resolved = cons + matched
        if resolved == 0:
            continue
        rate = matched / float(resolved)
        if lg['role'] != 'arm':
            out.append('  %-46s: %.3f (matched %d / resolved %d)  [reported, not gated]'
                       % ('M7b late-fill rate (%s)' % lg['role'], rate, matched, resolved))
            continue
        arm_rates.append((lg, rate, resolved))
        if resolved < M7B_MIN_RESOLVED:
            out.append('  %-46s: UNSCORED - matched %d / resolved %d is below the n >= %d floor'
                       % ('M7b late-fill rate (arm)', matched, resolved, M7B_MIN_RESOLVED))
            continue
        verdict = 'FAIL' if rate > M7B_MAX_RATE else 'PASS'
        if verdict == 'FAIL':
            band_fails.append((lg, rate, resolved))
        out.append('  %-46s: %.3f (matched %d / resolved %d)  [arm band <= %.3f] %s'
                   % ('M7b late-fill rate (arm)', rate, matched, resolved,
                      M7B_MAX_RATE, verdict))
    if not arm_rates:
        out.append('  %-46s: UNMEASURED (no arm leg resolved a fill)'
                   % 'M7b late-fill rate (arm)')
    for lg, rate, resolved in band_fails:
        out.append('      BAND FAIL (not an invariant, does not halt the session): M7b on %s - '
                   'late-fill rate %.3f over %d resolved fills, above %.3f. This fails the leg\'s '
                   'M7 band exactly as M1/M5 fail theirs.'
                   % (os.path.basename(lg['path']), rate, resolved, M7B_MAX_RATE))

    if rets:
        for rl in rets:
            out.append('  retref K=%d share                               : %.4f'
                       % (rl['retain_k'], rl['share']))

    # ------------------------------------------------------------ invariants
    out.append('')
    out.append('INVARIANTS (falsifiers; a FAIL blocks reporting, it does not authorize)')
    failures = []

    p1_bad = [l for l in spec_legs if int(l['spec']['waited_by_demand']) != 0]
    if not spec_legs:
        out.append('  P1  demand never waits on a speculative copy   : no evidence (no arm leg)')
    elif p1_bad:
        failures.append('P1')
        out.append('  P1  demand never waits on a speculative copy   : FAIL %s'
                   % ', '.join(os.path.basename(l['path']) for l in p1_bad))
    else:
        out.append('  P1  demand never waits on a speculative copy   : PASS (all legs 0)')

    if spec_legs:
        res = [(l['role'], l['residual']) for l in spec_legs]
        out.append('  P2  fill accounting closes                     : PASS (residual %s; a positive '
                   'residual is a READY fill evicted before any demand used it - uninstrumented)'
                   % ', '.join('%s=%d' % r for r in res))
    else:
        out.append('  P2  fill accounting closes                     : no evidence (no arm leg)')

    p4_bad = []
    for l in spec_legs:
        s = l['spec']
        ssum = int(s['cancelled_generation']) + int(s['cancelled_demand']) + int(s['cancelled_capacity'])
        if ssum != int(s['cancelled']):
            p4_bad.append(l)
    if not spec_legs:
        out.append('  P4  cancellation names its firing point        : no evidence (no arm leg)')
    elif p4_bad:
        failures.append('P4')
        out.append('  P4  cancellation names its firing point        : FAIL %s (named points do not '
                   'sum to `cancelled`)' % ', '.join(os.path.basename(l['path']) for l in p4_bad))
    else:
        out.append('  P4  cancellation names its firing point        : PASS (fire count and named '
                   'points agree; `cancelled` > 0 was required to score at all)')

    off_polluted = [l for l in offs if l['run_spec_mode'] != 0]
    if off_polluted:
        failures.append('P6')
        out.append('  P6  an off leg carries no speculation          : FAIL')
    else:
        out.append('  P6  an off leg carries no speculation          : PASS')

    below_floor = [l for l in spec_legs if l['lead_median'] is not None and int(l['spec']['lead_min']) < 2]
    if below_floor:
        failures.append('M7a')
        out.append('  M7a lead_min >= 2 (the reap-ordering floor)    : FAIL %s (a consumption '
                   'earlier than the reap ordering permits means the ordinal bookkeeping is '
                   'wrong)' % ', '.join(os.path.basename(l['path']) for l in below_floor))
    elif lead_legs:
        out.append('  M7a lead_min >= 2 (the reap-ordering floor)    : PASS (no illegal lead)')
    else:
        out.append('  M7a lead_min >= 2 (the reap-ordering floor)    : no evidence (no '
                   'consumption)')

    out.append('')
    if failures:
        out.append('VERDICT: FAIL-INVARIANT (%s). Numbers above are reported for the record; the '
                   'legs are not usable as evidence until the invariant is repaired.'
                   % ', '.join(failures))
    else:
        out.append('VERDICT: scored. Every pre-registered invariant held on the supplied legs. '
                   'This is a gate RESULT, not an authorization: M-bands are advisory evidence '
                   'and final acceptance is the owner\'s decision.')
    return '\n'.join(out), failures


def infer_role(path):
    """A role is a SESSION DECLARATION, not a filename property, so only the
    unambiguous control names are inferred.  In particular `arm` is NEVER inferred:
    this repository names every A/B leg `arm-A1.jsonl` / `arm-B1.jsonl`, which has
    nothing to do with a predictor arm -- inferring one would silently turn an
    ordinary recorded run into an 'arm' claim and refuse it (or worse, judge it).
    Predictor arms and nulls are declared with --leg.
    """
    hay = os.path.dirname(path).lower() + '/' + os.path.basename(path).lower()
    if 'null' in hay:
        return 'null'
    if 'retref' in hay or 'ret-ref' in hay:
        return 'retref'
    if 'off' in hay or 'ctl' in hay or 'ctrl' in hay or 'baseline' in hay:
        return 'off'
    return None


# -------------------------------------------------------------------------- cli

def run(argv):
    ap = argparse.ArgumentParser(add_help=True, description=__doc__.split('\n')[0])
    ap.add_argument('legs', nargs='*', help='traces; use --leg role=path to declare roles')
    ap.add_argument('--leg', action='append', default=[], metavar='ROLE=PATH',
                    help='declare a leg and its role (off|arm|null|retref); repeatable')
    ap.add_argument('--trace', action='append', default=[], metavar='PATH',
                    help='score one trace with its role inferred from the name; use --leg to pin '
                         'the role when the bands depend on it (a bare --trace cannot compute M2/M4)')
    ap.add_argument('--session', default=None, help='label for the leg set (recorded in the report)')
    ap.add_argument('--json', dest='json_out', default=None, help='write the computed quantities as JSON')
    ap.add_argument('--selftest', action='store_true', help='run the host fixture suite (no GPU)')
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    pairs = []
    for spec in args.leg:
        if '=' not in spec:
            sys.stderr.write('REFUSED: --leg wants ROLE=PATH, got %r\n' % spec)
            return EXIT_REFUSED
        role, path = spec.split('=', 1)
        role = role.strip().lower()
        if role not in ('off', 'arm', 'null', 'retref'):
            sys.stderr.write('REFUSED: unknown role %r (off|arm|null|retref)\n' % role)
            return EXIT_REFUSED
        pairs.append((role, path))
    for path in args.legs:
        pairs.append((infer_role(path), path))
    for path in args.trace:
        pairs.append((infer_role(path) or 'unknown', path))
    if not pairs:
        ap.error('give at least one leg, or --selftest')

    legs, refusals, inferred = [], [], []
    for role, path in pairs:
        if role is None:
            refusals.append(('role-unknown', 'cannot infer a role for %s: declare it with '
                                             '--leg off=... | arm=... | null=... | retref=...' % path))
            continue
        if not any(role == r and path == p for r, p in args_leg_pairs(args)):
            inferred.append('%s=%s' % (role, path))
        try:
            legs.append(read_leg(path, role))
        except Refused as r:
            refusals.append((r.code, r.detail))

    if refusals:
        sys.stderr.write('REFUSED (exit 2): %d leg(s) cannot be scored. NO numbers are printed for a '
                         'refused leg: a leg that cannot be justified is not evidence.\n' % len(refusals))
        for code, detail in refusals:
            sys.stderr.write('  [%s] %s\n' % (code, detail))
        return EXIT_REFUSED

    if inferred:
        sys.stderr.write('note: roles inferred from path names (declare with --leg to pin them): %s\n'
                         % ', '.join(inferred))

    text, failures = report(legs, args.session)
    sys.stdout.write(text + '\n')

    if args.json_out:
        payload = {
            'session': args.session,
            'legs': [{k: v for k, v in l.items() if k not in ('events',)} for l in legs],
            'failures': failures,
        }
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        sys.stdout.write('json: %s\n' % args.json_out)

    return EXIT_INVARIANT if failures else EXIT_OK


def args_leg_pairs(args):
    out = []
    for spec in args.leg:
        if '=' in spec:
            role, path = spec.split('=', 1)
            out.append((role.strip().lower(), path))
    return out


# --------------------------------------------------------------------- selftest

def _ev(**kw):
    return json.dumps(kw, sort_keys=True)


def write_trace(tmp, name, events):
    path = os.path.join(tmp, name)
    with open(path, 'w', encoding='utf-8') as fh:
        for e in events:
            fh.write(e + '\n')
    return path


def _init(retain_k=4, top_k=8, n_expert=256):
    return _ev(event='rowset_init', run_id=1, policy='moe-row-set', n_slots=4, slot_bytes=1,
               window_tokens=1, n_expert=n_expert, top_k=top_k,
               **({} if retain_k is None else {'retain_k': retain_k}), t_us=1)


def _fetch(layer, resident, fetched, phase=1, xfer=100.0, staged=1000):
    return _ev(event='rowset_fetch', run_id=1, ubatch=1, layer=layer, slot=0, generation=1,
               chunk=0, phase=phase, n_tokens=1, top_k=8, route_union=8, union_ids='1,2',
               hash=5, n_rows_resident=resident, n_rows_fetched=fetched, n_rows_retained=0,
               n_rows_retired=0, bytes_staged=staged * fetched, bytes_full_bank=staged * 8,
               host_submit_us=10.0, transfer_us=xfer, ready_latency_us=5.0, t_us=2)


def _spec(layer, submitted, consumed, matched, cancelled, gen, dem, cap, refused=0, dup=0,
          waited=0, lead_min=2, lead_max=8, lead_sum=20, hist=(1, 2, 1, 0, 0, 0, 0, 1),
          pending=1, ready_spec=0, mode=1):
    return _ev(event='spec', run_id=1, ubatch=1, layer=layer, mode=mode, budget=8, reserve=8,
               cap=16, pending=pending, ready_spec=ready_spec, submitted=submitted, ready=1,
               cancelled=cancelled, cancelled_capacity=cap, cancelled_demand=dem,
               cancelled_generation=gen, refused_capacity=refused, dup_refused=dup,
               matched_by_demand=matched, waited_by_demand=waited, reap_wait_us=0,
               issued_bytes=100, consumed=consumed, lead_sum=lead_sum, lead_min=lead_min,
               lead_max=lead_max, lead_hist=list(hist), t_us=3)


def _start(role_mode=0):
    d = {'event': 'run_start', 'run_id': 1, 'arch': 'qwen35moe', 'n_layer': 40, 'rowset': 'on',
         'kv_cache': 'q8', 'schedule': 'layer-major'}
    if role_mode:
        d.update({'row_spec_mode': role_mode, 'row_spec_budget': 8, 'row_spec_reserve': 8})
    return _ev(**d)


def _end(peak=9_000_000_000):
    return _ev(event='run_end', run_id=1, exit_status=0, peak_vram_bytes=peak,
               n_tokens_generated=256, failed=False, error='none')


def _acct(row_cache=3_000_000_000):
    return _ev(event='byte_account', run_id=1, row_cache=row_cache, bounded_window=4_000_000_000,
               per_layer=1_000_000, kv=1_000_000, static=1_000_000, slot_window=2_000_000,
               free_vram=20_000_000_000, total_vram=25_000_000_000)


def _run(args):
    """Run the checker in-process; -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = EXIT_REFUSED
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = run(args)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else EXIT_REFUSED
        except Refused as r:
            err.write('REFUSED: %s\n' % r.detail)
            code = EXIT_REFUSED
    return code, out.getvalue(), err.getvalue()


def selftest():
    cases, fails = 0, []

    def check(name, got, want):
        nonlocal cases
        cases += 1
        if got != want:
            fails.append('%s: got %r want %r' % (name, got, want))

    tmp = tempfile.mkdtemp(prefix='tgp-')

    # ---- scored legs -------------------------------------------------------
    # off: 4096 demands, 2048 hits -> share 0.5
    off = write_trace(tmp, 'off-K4.jsonl',
                      [_start(0), _init(4), _fetch(0, 2048, 2048), _acct(), _end()])
    # arm: 4096 demands, 3072 hits -> share 0.75; invariant-consistent spec block
    # submitted 10 = consumed 5 + cancelled 3 + live 2 (... + residual 0)
    arm = write_trace(tmp, 'arm-K2.jsonl',
                      [_start(1), _init(2), _fetch(0, 3072, 1024, xfer=250.0, staged=2000),
                       _spec(0, submitted=10, consumed=5, matched=2, cancelled=3, gen=1, dem=1,
                             cap=1, hist=(0, 3, 1, 0, 0, 0, 0, 1), lead_min=2, lead_max=8,
                             lead_sum=19, pending=1, ready_spec=1),
                       _acct(), _end()])
    null = write_trace(tmp, 'null-K4.jsonl',
                       [_start(1), _init(4), _fetch(0, 2560, 1536),
                        _spec(0, submitted=4, consumed=2, matched=1, cancelled=1, gen=1, dem=0,
                              cap=0, hist=(0, 2, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2,
                              lead_sum=4, pending=0, ready_spec=1),
                        _acct(), _end()])

    code, out, err = _run(['--leg', 'off=' + off, '--leg', 'arm=' + arm, '--leg', 'null=' + null,
                           '--session', 'selftest'])
    check('scored exit code', code, EXIT_OK)
    check('off share 0.5000', 'share=0.5000' in out, True)
    check('arm share 0.7500', 'share=0.7500' in out, True)
    check('M4 margin_share = +0.5000', 'M4  margin_share(arm K=2 vs off)             : '
                                       '(0.7500-0.5000)/(1-0.5000) = +0.5000' in out, True)
    check('M2 pool-growth = +0.1250', 'M2  null-vs-off pool-growth share (null K=4)  : '
                                      '0.6250 - 0.5000 = +0.1250' in out, True)
    check('M4n null-corrected = +0.1250', 'M4n null-corrected gain (structure only)       : '
                                          '0.7500 - 0.6250 = +0.1250' in out, True)
    check('M7c median is 2', 'M7c lead_rows min/median/max (arm)        : 2 / 2 / 8' in out, True)
    check('P1 PASS', 'P1  demand never waits on a speculative copy   : PASS' in out, True)
    check('P2 PASS', 'P2  fill accounting closes                     : PASS' in out, True)
    check('P4 PASS', 'P4  cancellation names its firing point        : PASS' in out, True)
    check('P6 PASS', 'P6  an off leg carries no speculation          : PASS' in out, True)
    check('M7a floor PASS', 'M7a lead_min >= 2 (the reap-ordering floor)    : PASS' in out, True)
    check('verdict scored', 'VERDICT: scored.' in out, True)
    # M7b (the 2026-09-23 restatement): this fixture's arm resolves 7 fills, one below the
    # sample floor, so its rate is DECLARED but not scored -- and the `null` leg's rate is
    # reported without being gated (gating the structure-free control would fail it by
    # construction).
    check('M7b thin sample is UNSCORED',
          'UNSCORED - matched 2 / resolved 7 is below the n >= 8 floor' in out, True)
    check('M7b null leg is reported, not gated',
          'M7b late-fill rate (null)' in out and '[reported, not gated]' in out, True)

    # ---- M7b: the late-fill rate band (the 2026-09-23 M7 restatement) --------
    # A band, not an invariant: every case below must still verdict `scored`.  The three
    # fixtures are the arm that is mostly too late (0.750 -> FAIL), the boundary arm that
    # splits exactly evenly (0.500 -> PASS, the threshold is inclusive), and the arm whose
    # guesses NEVER landed (rate 1.000 -> FAIL) -- the case the pre-restatement wording
    # could not see at all, because it consumed nothing and published no median.
    late = write_trace(tmp, 'arm-late-K2.jsonl',
                       [_start(1), _init(2), _fetch(0, 2800, 1296),
                        _spec(0, submitted=12, consumed=3, matched=9, cancelled=9, gen=2, dem=6,
                              cap=1, hist=(0, 2, 1, 0, 0, 0, 0, 0), lead_min=2, lead_max=3,
                              lead_sum=8, pending=0, ready_spec=0),
                        _acct(), _end()])
    even = write_trace(tmp, 'arm-even-K2.jsonl',
                       [_start(1), _init(2), _fetch(0, 2800, 1296),
                        _spec(0, submitted=8, consumed=4, matched=4, cancelled=4, gen=1, dem=3,
                              cap=0, hist=(0, 2, 1, 1, 0, 0, 0, 0), lead_min=2, lead_max=4,
                              lead_sum=11, pending=0, ready_spec=0),
                        _acct(), _end()])
    never = write_trace(tmp, 'arm-never-K2.jsonl',
                        [_start(1), _init(2), _fetch(0, 2900, 1196),
                         _spec(0, submitted=9, consumed=0, matched=9, cancelled=9, gen=0, dem=9,
                               cap=0, hist=(0, 0, 0, 0, 0, 0, 0, 0), lead_min=0, lead_max=0,
                               lead_sum=0, pending=0, ready_spec=0),
                         _acct(), _end()])

    code, out, err = _run(['--leg', 'arm=' + late, '--leg', 'off=' + off])
    check('M7b FAIL keeps the exit code at 0', code, EXIT_OK)
    check('M7b late arm 0.750 FAIL',
          '0.750 (matched 9 / resolved 12)' in out and '[arm band <= 0.500] FAIL' in out, True)
    check('M7b FAIL is printed as a band, not an invariant',
          'BAND FAIL (not an invariant, does not halt the session)' in out
          and 'VERDICT: scored.' in out, True)

    code, out, err = _run(['--leg', 'arm=' + even, '--leg', 'off=' + off])
    check('M7b boundary arm 0.500 PASS (inclusive)',
          '0.500 (matched 4 / resolved 8)' in out and '[arm band <= 0.500] PASS' in out, True)

    code, out, err = _run(['--leg', 'arm=' + never, '--leg', 'off=' + off])
    check('M7b never-landed arm 1.000 FAIL',
          '1.000 (matched 9 / resolved 9)' in out and '[arm band <= 0.500] FAIL' in out, True)
    check('M7a is silent when nothing was consumed',
          'no evidence (no consumption)' in out, True)

    nofill = write_trace(tmp, 'arm-nofill-K2.jsonl',
                         [_start(1), _init(2), _fetch(0, 2900, 1196),
                          _spec(0, submitted=5, consumed=0, matched=0, cancelled=5, gen=5, dem=0,
                                cap=0, hist=(0, 0, 0, 0, 0, 0, 0, 0), lead_min=0, lead_max=0,
                                lead_sum=0, pending=0, ready_spec=0),
                          _acct(), _end()])
    code, out, err = _run(['--leg', 'arm=' + nofill, '--leg', 'off=' + off])
    check('M7b UNMEASURED when no arm leg resolved a fill',
          'UNMEASURED (no arm leg resolved a fill)' in out, True)

    # ---- refusals: exit 2, no numbers --------------------------------------
    def refusal(name, argv_extra, code_want, marker, trace=None):
        code, out, err = _run(argv_extra)
        check('%s exit code' % name, code, EXIT_REFUSED)
        check('%s names the rule' % name, marker in err, True)
        check('%s prints no band numbers' % name,
              ('M4 ' not in out and 'M7 ' not in out and 'share=' not in out), True)

    # 1. non-instrumented: a trace with events but no rowset_init
    p = write_trace(tmp, 'uninstrumented.jsonl', [_start(0), _fetch(0, 1, 1), _end()])
    refusal('no rowset_init', ['--leg', 'off=' + p], EXIT_REFUSED, 'no-retention', p)

    # 2. a leg with no retain_k
    p = write_trace(tmp, 'no-retain-k.jsonl', [_start(0), _init(None), _fetch(0, 1, 1), _end()])
    refusal('no retain_k', ['--leg', 'off=' + p], EXIT_REFUSED, 'no-retain-k', p)

    # 3. retention OFF (reactive baseline: no per-layer residency at all)
    p = write_trace(tmp, 'reactive.jsonl', [_start(0), _init(0), _fetch(0, 1, 1), _end()])
    refusal('retain_k = 0', ['--leg', 'off=' + p], EXIT_REFUSED, 'no-retain-k', p)

    # 4. an arm leg whose `spec` lacks `cancelled` (the schema the block names)
    bad = _json_drop(_spec(0, submitted=4, consumed=2, matched=1, cancelled=1, gen=1, dem=0, cap=0,
                           hist=(1, 1, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2, lead_sum=4,
                           pending=0, ready_spec=1),
                     ('cancelled', 'cancelled_capacity', 'cancelled_demand',
                      'cancelled_generation'))
    p = write_trace(tmp, 'arm-nocancel.jsonl', [_start(1), _init(2), _fetch(0, 1, 1), bad, _end()])
    refusal('spec missing cancelled', ['--leg', 'arm=' + p], EXIT_REFUSED, 'spec-missing-keys', p)

    # 5. the A2 acceptance fixture: the cancel hook NEVER fires -> refused, not scored
    p = write_trace(tmp, 'arm-nocancel-fired.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=4, consumed=2, matched=2, cancelled=0, gen=0, dem=0, cap=0,
                           hist=(0, 2, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2, lead_sum=4,
                           pending=0, ready_spec=0),
                     _end()])
    refusal('cancel hook never fired', ['--leg', 'arm=' + p], EXIT_REFUSED,
            'cancel-hook-never-fired', p)

    # 6. an arm leg with no spec events at all
    p = write_trace(tmp, 'arm-silent.jsonl', [_start(1), _init(2), _fetch(0, 1, 1), _end()])
    refusal('arm without spec', ['--leg', 'arm=' + p], EXIT_REFUSED, 'arm-without-spec', p)

    # 7. role/label conflict: an off leg that carries speculation
    p = write_trace(tmp, 'off-but-spec.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=2, consumed=1, matched=0, cancelled=1, gen=1, dem=0, cap=0,
                           hist=(1, 0, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2, lead_sum=2,
                           pending=0, ready_spec=1),
                     _end()])
    refusal('off leg with speculation', ['--leg', 'off=' + p], EXIT_REFUSED,
            'off-leg-with-speculation', p)

    # 8. double counting: submitted below consumed + cancelled + live
    p = write_trace(tmp, 'arm-double.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=3, consumed=5, matched=0, cancelled=1, gen=1, dem=0, cap=0,
                           hist=(0, 5, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2, lead_sum=10,
                           pending=0, ready_spec=0),
                     _end()])
    refusal('negative residual', ['--leg', 'arm=' + p], EXIT_REFUSED, 'accounting-negative', p)

    # 9. lead_hist and consumed disagree
    p = write_trace(tmp, 'arm-leadbad.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=4, consumed=5, matched=0, cancelled=1, gen=1, dem=0, cap=0,
                           hist=(0, 2, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2, lead_sum=10,
                           pending=0, ready_spec=0),
                     _end()])
    refusal('lead histogram disagreement', ['--leg', 'arm=' + p], EXIT_REFUSED,
            'lead-inconsistent', p)

    # 10. an undeclarable role (no --leg and an unrecognizable name)
    p = write_trace(tmp, 'mystery.jsonl', [_start(0), _init(2), _fetch(0, 1, 1), _end()])
    code, out, err = _run([p])
    check('unknown role exit code', code, EXIT_REFUSED)
    check('unknown role names the fix', 'declare it with --leg' in err, True)

    # 11. not a trace at all
    code, out, err = _run(['--leg', 'off=' + os.path.join(tmp, 'nope.jsonl')])
    check('missing file exit code', code, EXIT_REFUSED)
    check('missing file names the rule', 'file-missing' in err, True)

    # ---- invariant failures are reported, not refused ----------------------
    # P1 fires: a reap inside a demand window (waited_by_demand = 1)
    p = write_trace(tmp, 'arm-p1.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=2, consumed=1, matched=0, cancelled=1, gen=1, dem=0, cap=0,
                           waited=1, hist=(0, 1, 0, 0, 0, 0, 0, 0), lead_min=2, lead_max=2,
                           lead_sum=2, pending=0, ready_spec=0),
                     _end()])
    code, out, err = _run(['--leg', 'arm=' + p, '--leg', 'off=' + off])
    check('P1 violation exit code', code, EXIT_INVARIANT)
    check('P1 violation named', 'VERDICT: FAIL-INVARIANT (P1)' in out, True)

    # M7 floor: a consumption with lead 1 (impossible on the reap ordering)
    p = write_trace(tmp, 'arm-lead1.jsonl',
                    [_start(1), _init(2), _fetch(0, 1, 1),
                     _spec(0, submitted=2, consumed=1, matched=0, cancelled=1, gen=1, dem=0, cap=0,
                           hist=(1, 0, 0, 0, 0, 0, 0, 0), lead_min=1, lead_max=1, lead_sum=1,
                           pending=0, ready_spec=0),
                     _end()])
    code, out, err = _run(['--leg', 'arm=' + p, '--leg', 'off=' + off])
    check('lead below floor exit code', code, EXIT_INVARIANT)
    check('lead below floor named', 'M7a' in out, True)

    # ---- emitter <-> checker schema (host-only, no GPU) --------------------
    keys = _emitter_keys()
    if keys is None:
        sys.stdout.write('note: the engine source was not found; the emitter<->checker key '
                         'cross-check was skipped\n')
    else:
        missing = [k for k in SPEC_REQUIRED if k not in keys]
        check('every key the checker requires is emitted by tel_spec', missing, [])
        check('the emitter emits `cancelled` (the key the block names)', 'cancelled' in keys, True)
        check('the emitter emits the lead histogram', 'lead_hist' in keys, True)

    # ---- real recorded traces: the off baseline must score -----------------
    # (host-only; absent in a clean checkout, in which case this case is skipped)
    real = os.path.join('build', 'llama-cuda', 'gplr19-honest-k6-ab', 'arm-B1.jsonl')
    if os.path.isfile(real):
        code, out, err = _run(['--leg', 'off=' + real])
        check('recorded legacy off leg scores', code, EXIT_OK)
        check('recorded legacy off leg reports a share', 'share=0.' in out, True)
        # the shares this checker computes must equal the margin analyzer's own
        # booking on the same trace: two independent readers of one recording
        code, out, err = _run(['--trace', real])
        check('--trace scores a repo A/B-named leg (no arm inference)', code, EXIT_OK)
        check('--trace reports the same share as --leg off', 'share=0.6461' in out, True)
        check('--trace does not claim a role it cannot know', 'role  ' in out and 'unknown' in out, True)

    if fails:
        sys.stderr.write('SELFTEST FAILED (%d/%d)\n' % (len(fails), cases))
        for f in fails:
            sys.stderr.write('  %s\n' % f)
        return 1
    sys.stdout.write('SELFTEST PASSED (%d checks)\n' % cases)
    return 0


def _emitter_keys(rel=os.path.join('third_party', 'llama.cpp', 'src', 'llama-layer-streaming.cpp')):
    """The JSON keys `tel_spec()` actually emits, read out of the engine source.

    This is the host-only half of the EMITTER <-> CHECKER contract: the checker
    refuses a leg whose `spec` is missing a key it needs, so a key the engine never
    emits would make every arm leg refuse -- a failure worth catching with no GPU.
    Returns None when the vendored source is not present.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as fh:
        src = fh.read()
    i = src.find('void llama_layer_streaming::tel_spec(')
    if i < 0:
        return None
    body = src[i:i + 4000]
    return set(re.findall(r'\\"([A-Za-z_0-9]+)\\":', body))


def _json_drop(line, keys):
    d = json.loads(line)
    for k in keys:
        d.pop(k, None)
    return json.dumps(d, sort_keys=True)


if __name__ == '__main__':
    sys.exit(run(sys.argv[1:]))
