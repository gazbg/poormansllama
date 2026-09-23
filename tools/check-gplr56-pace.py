#!/usr/bin/env python3
"""T-GPLR-56 A3 arm `p1-pace`: paced-registration trace analyzer - host-only, no GPU.

WHAT IT READS. Only the events the paced mode itself emits, plus the existing copy
events a contention reading needs:

  * `span_pace_init`  - one per paced run: mode, ahead, groups, bytes_total, and what
                        the pre-pass head start pinned (bytes_issued).
  * `span_pace`       - one per registration GROUP issued (group index, first layer,
                        bytes, and bytes_unissued after the issue).
  * `span_pace_stage` - one per layer at its FIRST staging copy: whether the span was
                        already locked (`covered`) and how many groups had been issued.
  * `span_pace_end`   - the last group was issued: the commit is complete.
  * `copy_submit` / `copy_complete` - the existing staged-copy events, for the first
                        stall and the transfer medians (the contention instrument).

WHY THOSE. The mode's claim is that the ~1.9 s deferred commit is issued in group
order a bounded distance ahead of the copy that needs it, so pending unlocked bytes
stay near ONE group instead of 17.23 GiB. That claim is readable only from these
events: how many groups were issued before a layer's first touch (the pacing
distance), the pending-bytes trajectory, and whether a copy was ever staged with its
span not yet locked (which the engine's own refusal owns, and which is a DEFECT here,
never a result).

WHAT IS NOT OBSERVABLE HERE, stated so it is not implied: how many REGISTERED-BUT-NOT-YET-
COMMITTED bytes are outstanding at a moment. The driver's deferred commit is not
instrumented, so `bytes_unissued` (bytes whose cudaHostRegister has not been CALLED
YET) is not that quantity - it starts near the whole span and only falls as the sweep
issues. The pacing claim is read from the ORDER (issue distance) and the STALL (the
first staged copy), not from an outstanding-bytes counter.

REFUSALS (exit 2, no numbers printed) - this tool refuses rather than reporting a
defaulted reading:
  * a trace with no `span_pace_init`: it is not a paced run, so nothing here applies
    (this is exactly how the `p1-off` / `p1-early` arms are rejected);
  * a paced trace whose commit never finished: `span_pace_end` absent, or groups
    issued < groups total. The mode's contract is that every executed layer's span is
    registered before its copy; an unfinished commit leaves that unmet, and reading it
    as "paced" would score a partial mode as a whole one;
  * a `span_pace` record with no `bytes`, or a `span_pace_init` with groups <= 0.

EXIT CODES: 0 = evaluated (findings printed), 1 = the coverage integrity check FAILED
(a layer's first staging copy ran with `covered = 0` - name the layer), 2 = refused.

    python tools/check-gplr56-pace.py --trace build/llama-cuda/gplr56-a3-pace-4k.jsonl \
                                      [--control build/llama-cuda/gplr56-a3-off-4k.jsonl]
    python tools/check-gplr56-pace.py --selftest
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile

MS = 1000.0


def load_events(path):
    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    return events


def of(events, name):
    return [e for e in events if e.get("event") == name]


def pass_windows(events):
    """[(pass_no, begin_us, end_us)] from the recon-gated host_mem brackets."""
    out = []
    cur = {}
    for e in events:
        if e.get("event") != "host_mem":
            continue
        lbl = e.get("label")
        if lbl == "pass_begin":
            cur[e.get("pass")] = [e.get("t_us"), None]
        elif lbl == "pass_end" and e.get("pass") in cur:
            cur[e.get("pass")][1] = e.get("t_us")
    for p in sorted(cur, key=lambda x: (x is None, x)):
        b, en = cur[p]
        if b is not None and en is not None:
            out.append((p, b, en))
    return out


def transfers(events, lo=None, hi=None):
    out = []
    for e in of(events, "copy_complete"):
        t = e.get("transfer_us")
        if t is None:
            continue
        if lo is not None and (e.get("t_us") is None or e["t_us"] < lo):
            continue
        if hi is not None and (e.get("t_us") is None or e["t_us"] > hi):
            continue
        out.append(float(t))
    return out


def med(xs):
    return statistics.median(xs) if xs else None


def pacing_lead(issues, stages):
    """How many groups had been issued BEYOND the group covering the layer, at the
    moment that layer's first staging copy was reached. The covering group of layer il
    is the LAST group whose first layer is <= il (groups are coalesced by address, so a
    merged group covers every layer its first layer precedes)."""
    order = sorted(issues, key=lambda g: g.get("gi", 0))
    if not order:
        return []
    firsts = [(g.get("gi"), g.get("layer")) for g in order]
    out = []
    for s in stages:
        il = s.get("layer")
        n_issued = s.get("groups_issued")
        if il is None or n_issued is None:
            continue
        cover_idx = 0
        for i, (_, fl) in enumerate(firsts):
            if fl is not None and fl <= il:
                cover_idx = i
        out.append(n_issued - (cover_idx + 1))
    return out


def analyze(events, control_events=None):
    """Returns (exit_code, lines)."""
    lines = []
    init = of(events, "span_pace_init")
    if not init:
        return 2, [f"REFUSED: no `span_pace_init` in this trace - it is not a paced "
                   f"registration run, so the paced reading does not apply.",
                   "  (a p1-off / p1-early arm is refused here on purpose: only a run "
                   "that named the mode can be scored on it)"]
    init = init[0]
    groups_total = init.get("groups")
    if not groups_total or groups_total <= 0:
        return 2, ["REFUSED: `span_pace_init` carries no group count (groups <= 0) - "
                   "the mode's own accounting is missing, so no paced reading is possible"]
    issues = of(events, "span_pace")
    for g in issues:
        if g.get("bytes") is None:
            return 2, [f"REFUSED: `span_pace` group {g.get('gi')} carries no `bytes` - "
                       f"a malformed paced record, not a measurement"]
    stages = of(events, "span_pace_stage")
    end = of(events, "span_pace_end")

    if not end or len(issues) < groups_total:
        return 2, [f"REFUSED: the paced commit is UNFINISHED - {len(issues)} of {groups_total} "
                   f"groups were issued and `span_pace_end` is {'absent' if not end else 'present'}.",
                   "  An unfinished commit leaves executed layers whose span was never locked, "
                   "which the mode's contract forbids: this is a refusal, not a pacing result."]

    lines.append(f"PACE INIT   mode {init.get('mode')}  ahead {init.get('ahead')} group(s)  "
                 f"groups {groups_total}  total {init.get('bytes_total')} B  "
                 f"head start pinned {init.get('bytes_issued')} B in {(init.get('us') or 0)/MS:.1f} ms")
    # The groups are issued from stage_members, so their stamps tell when each group
    # entered the commit queue; span_pace_end closes it.
    first_issue = min(g["t_us"] for g in issues if g.get("t_us") is not None)
    last_issue = max(g["t_us"] for g in issues if g.get("t_us") is not None)
    commit_us = (end[0]["t_us"] - first_issue) if end[0].get("t_us") is not None else None
    unissued = [g["bytes_unissued"] for g in issues if g.get("bytes_unissued") is not None]
    peak_unissued = max(unissued) if unissued else None
    windows = pass_windows(events)
    pass1 = windows[0] if windows else None
    overlap = None
    if pass1 is not None and commit_us is not None:
        cb, ce = first_issue, end[0]["t_us"]
        overlap_us = max(0, min(ce, pass1[2]) - max(cb, pass1[1]))
        overlap = 100.0 * overlap_us / max(1, (ce - cb))
    tail = (f", closing {(end[0]['t_us'] - first_issue)/MS:.1f} ms after the first"
            if commit_us is not None else "")
    if peak_unissued is not None:
        tail += f"; bytes whose register call had not been made yet: {peak_unissued} B at the start"
    lines.append(f"PACE ISSUE  {len(issues)}/{groups_total} groups issued; "
                 f"{(last_issue - first_issue)/MS:.1f} ms of issue calls{tail}")

    # The pre-registered discriminator: how many groups were already issued when each
    # layer's FIRST staging copy was reached.
    if stages:
        issued_at_touch = [s.get("groups_issued") for s in stages if s.get("groups_issued") is not None]
        covered = [s for s in stages if s.get("covered") == 1]
        uncovered = [s for s in stages if s.get("covered") != 1]
        lines.append(f"COVERAGE    {len(stages)} layer(s) seen at a first staging copy; "
                     f"{len(covered)} covered, {len(uncovered)} NOT covered")
        if issued_at_touch:
            lines.append(f"            groups already issued when a layer was first staged: "
                         f"min {min(issued_at_touch)}, median {med(issued_at_touch)}, "
                         f"max {max(issued_at_touch)} of {groups_total}")
            lead = pacing_lead(issues, stages)
            if lead:
                # The tail cannot be paced: once the last group is issued there is
                # nothing left to run ahead of, so the final layers' lead falls to 0
                # by construction. That is a property of the mode, not a defect, and
                # the max is the configured distance.
                lines.append(f"            pacing LEAD at a layer's first touch (groups issued "
                             f"beyond the group that covers it): min {min(lead)}, "
                             f"median {med(lead)}, max {max(lead)} "
                             f"(0 on the final layers = nothing left to issue ahead of)")
    else:
        lines.append("COVERAGE    no `span_pace_stage` records: the run never reached a staging "
                     "copy, so the pacing distance is UNMEASURED here")
        uncovered = []

    # Contention instrument: transfer medians, per pass, against an optional control.
    t_all = transfers(events)
    lines.append("CONTENTION  copy_complete transfer medians"
                 + (f" (n={len(t_all)})" if t_all else " (no copy_complete events)"))
    if t_all:
        lines.append(f"            all passes : median {med(t_all)/MS:.2f} ms  "
                     f"min {min(t_all)/MS:.2f}  max {max(t_all)/MS:.2f}")
        for (p, b, en) in windows:
            tp = transfers(events, b, en)
            if tp:
                lines.append(f"            pass {p}     : median {med(tp)/MS:.2f} ms (n={len(tp)})")
        first_complete = min((e["t_us"] for e in of(events, "copy_complete") if e.get("t_us") is not None),
                             default=None)
        if pass1 is not None and first_complete is not None:
            # The stall proxy: from the pass-1 bracket opening to the first staged copy
            # completing. Comparable across arms in one session (all carry the markers).
            lines.append(f"            pass-1 stall proxy (pass_begin -> first copy_complete): "
                         f"{(first_complete - pass1[1])/MS:.1f} ms")
    if control_events is not None:
        c_all = transfers(control_events)
        if c_all:
            lines.append(f"            control    : median {med(c_all)/MS:.2f} ms (n={len(c_all)})"
                         f"   -> this arm {'' if not t_all else f'{med(t_all)/MS:.2f} ms'}; "
                         f"delta {'' if not t_all else f'{(med(t_all)-med(c_all))/MS:+.2f} ms'}")
        else:
            lines.append("            control    : no copy_complete events - no comparison")
    if overlap is not None:
        lines.append(f"OVERLAP     {overlap:.1f}% of the commit window lies inside the pass-1 "
                     f"window; the head start sits before it")

    if uncovered:
        names = ", ".join(str(s.get("layer")) for s in uncovered[:12])
        lines.append(f"INTEGRITY   FAIL: {len(uncovered)} layer(s) had their first staging copy "
                     f"staged with the span NOT locked (layers {names}"
                     f"{', ...' if len(uncovered) > 12 else ''})")
        lines.append("            The mode's contract is that registration is issued BEFORE the "
                     "copy that needs it; this is a defect to report, not a pacing result.")
        return 1, lines
    lines.append("INTEGRITY   OK: every layer's span was locked before its first staging copy")
    return 0, lines


def write_fixture(d, name, lines):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return p


def paced_trace(groups=4, ahead=1, uncovered_layer=None, finish=True, bytes_per=1000):
    """A synthetic paced run: `groups` groups, one layer each, issued `ahead` early."""
    out = [{"event": "run_start", "t0_us": 100, "n_layer": groups},
           {"event": "host_mem", "label": "pass_begin", "pass": 1, "t_us": 200},
           {"event": "span_pace_init", "run_id": 1, "mode": 1, "ahead": ahead, "groups": groups,
            "bytes_total": groups * bytes_per, "bytes_issued": bytes_per, "us": 1600.0, "t_us": 200}]
    for gi in range(groups):
        out.append({"event": "span_pace", "run_id": 1, "gi": gi, "layer": gi,
                    "bytes": bytes_per, "members": 3, "merged": 0, "base": 4096 * (gi + 1),
                    "issued": gi + 1, "total": groups,
                    "bytes_unissued": (groups - gi - 1) * bytes_per, "t_us": 300 + 100 * gi})
        # the layer's first staging copy, `ahead` groups after its own group was issued
        n_issued = min(groups, gi + 1 + ahead)
        cov = 1 if gi != uncovered_layer else 0
        out.append({"event": "span_pace_stage", "run_id": 1, "layer": gi, "covered": cov,
                    "groups_issued": n_issued, "groups_total": groups,
                    "bytes_unissued": max(0, (groups - n_issued)) * bytes_per,
                    "t_us": 320 + 100 * gi})
        out.append({"event": "copy_complete", "run_id": 1, "ubatch": 8, "layer": gi, "slot": gi % 3,
                    "generation": 1, "transfer_us": 31000.0 + gi, "host_submit_us": 40.0,
                    "ready_latency_us": 30960.0, "phase": 0, "t_us": 330 + 100 * gi})
    if finish:
        out.append({"event": "span_pace_end", "run_id": 1, "groups": groups,
                    "bytes": groups * bytes_per, "t_us": 300 + 100 * groups + 5000})
    out.append({"event": "host_mem", "label": "pass_end", "pass": 1, "t_us": 300 + 100 * groups + 6000})
    return out


def selftest():
    d = tempfile.mkdtemp(prefix="gplr56pace-")
    cases = []
    try:
        def run_case(name, want, trace_path, control=None):
            got, lines = analyze(load_events(trace_path),
                                 load_events(control) if control else None)
            ok = got == want
            print(f"  [{'ok ' if ok else 'BAD'}] {name}: exit {got} (want {want})")
            cases.append(ok)

        good = write_fixture(d, "good.jsonl", paced_trace())
        uncovered = write_fixture(d, "uncovered.jsonl", paced_trace(uncovered_layer=2))
        unfinished = write_fixture(d, "unfinished.jsonl", paced_trace(finish=False))
        short = write_fixture(d, "short.jsonl", paced_trace(groups=4, finish=True)[:-2])
        not_paced = write_fixture(d, "notpaced.jsonl",
                                  [{"event": "run_start", "t0_us": 1},
                                   {"event": "copy_complete", "transfer_us": 31000.0, "t_us": 2}])
        malformed = write_fixture(d, "malformed.jsonl",
                                  [dict(e, bytes=None) if e.get("event") == "span_pace" and e.get("gi") == 1 else e
                                   for e in paced_trace()])
        nogroups = write_fixture(d, "nogroups.jsonl",
                                 [dict(e, groups=0) if e.get("event") == "span_pace_init" else e
                                  for e in paced_trace()])
        control = write_fixture(d, "control.jsonl",
                                [{"event": "copy_complete", "transfer_us": 31500.0, "t_us": 10},
                                 {"event": "copy_complete", "transfer_us": 31510.0, "t_us": 11}])

        run_case("clean paced run -> evaluated", 0, good)
        run_case("clean paced run vs control -> evaluated", 0, good, control)
        run_case("a layer staged with covered=0 -> INTEGRITY FAIL", 1, uncovered)
        run_case("no span_pace_end (commit unfinished) -> REFUSED", 2, unfinished)
        run_case("some groups never issued -> REFUSED", 2, short)
        run_case("non-paced trace (p1-off / p1-early) -> REFUSED", 2, not_paced)
        run_case("span_pace record with no bytes -> REFUSED", 2, malformed)
        run_case("span_pace_init with groups=0 -> REFUSED", 2, nogroups)

        # The refusal must PRINT no reading, and the integrity failure must NAME the layer.
        _, lines = analyze(load_events(not_paced))
        ok = not any(l.startswith(("PACE INIT", "COVERAGE")) for l in lines)
        print(f"  [{'ok ' if ok else 'BAD'}] refusal prints no reading")
        cases.append(ok)
        _, lines = analyze(load_events(uncovered))
        ok = any("layers 2" in l for l in lines)
        print(f"  [{'ok ' if ok else 'BAD'}] integrity failure names the layer")
        cases.append(ok)

        # The pacing lead is the number the block pre-registered, so it must be read
        # back from the fixture exactly (ahead=1 -> every layer's group was issued one
        # group beyond the one covering it) and must FAIL the discriminator when the
        # fixture is issued with no head start (ahead=0 -> lead 0, i.e. no pacing).
        for want_ahead in (1, 3):
            ev = paced_trace(groups=8, ahead=want_ahead)
            lead = pacing_lead(of(ev, "span_pace"), of(ev, "span_pace_stage"))
            # The last `ahead` layers cannot be paced (nothing left to issue), so the
            # read-back is: max == the configured distance, and every interior layer
            # exactly that. A lead ABOVE the configured distance would mean the tool is
            # manufacturing pacing that the run did not have.
            interior = lead[:len(lead) - want_ahead] if lead and want_ahead < len(lead) else []
            ok = (lead and max(lead) == want_ahead and min(lead) >= 0
                  and interior and set(interior) == {want_ahead})
            print(f"  [{'ok ' if ok else 'BAD'}] lead reads back as ahead={want_ahead} (got "
                  f"{sorted(set(lead)) if lead else None}, interior {sorted(set(interior)) if interior else None})")
            cases.append(bool(ok))
        ev = paced_trace(groups=8, ahead=0)
        lead = pacing_lead(of(ev, "span_pace"), of(ev, "span_pace_stage"))
        ok = lead and set(lead) == {0}
        print(f"  [{'ok ' if ok else 'BAD'}] ahead=0 reads as lead 0 (no pacing; got "
              f"{sorted(set(lead)) if lead else None})")
        cases.append(bool(ok))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    bad = cases.count(False)
    print(f"SELFTEST: {'PASS' if bad == 0 else 'FAIL'} {len(cases) - bad}/{len(cases)}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="T-GPLR-56 A3 paced-registration trace analyzer (host-only)")
    ap.add_argument("--trace", help="paced arm trace JSONL")
    ap.add_argument("--control", help="same-session control trace, for the transfer medians")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.trace:
        ap.error("--trace or --selftest is required")
    if not os.path.exists(a.trace):
        print(f"REFUSED: {a.trace} does not exist")
        return 2
    events = load_events(a.trace)
    if not events:
        print(f"REFUSED: {a.trace} has no JSONL events")
        return 2
    control = None
    if a.control:
        if not os.path.exists(a.control):
            print(f"REFUSED: control {a.control} does not exist")
            return 2
        control = load_events(a.control)
    code, lines = analyze(events, control)
    print(f"paced registration: {os.path.basename(a.trace)}")
    for l in lines:
        print("  " + l)
    print(f"PACE RESULT: {'EVALUATED' if code == 0 else ('INTEGRITY FAIL' if code == 1 else 'REFUSED')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
