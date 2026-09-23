#!/usr/bin/env python3
"""T-GPLR-56 acceptance quantity Q - host-only, no GPU, no model.

Q is the SUM from process start to the end of the first prefill pass, chosen so a
relocation cannot satisfy it: pre-registered PASS requires BOTH that Q falls by more
than the margin against the same-session control AND that the pass-1 wall falls.

Reads exactly two things, and nothing else:
  * the pre-sink marker `proc_start`  (t_us)   - emitted before the sink exists
  * `host_mem` with label == "pass_end" and pass == 1  (t_us)
`host_mem` is self-gated on --pp-recon, so a Q arm must run with recon on; that is a
precondition of the quantity, not a fallback.

A trace missing either piece is REFUSED (exit 2) with the missing piece named. This is
deliberate: the T-GPLR-53 decay rule once refused its own input rather than deciding from
it, and Q must not repeat that class of defect by silently reading an absent field as 0.

    python tools/check-gplr56-q.py --trace <arm.jsonl> [--control <control.jsonl>]
    python tools/check-gplr56-q.py --selftest

Exit: 0 = computed (and, with --control, the rule's verdict printed), 2 = refused,
1 = the rule's verdict is FAIL (only with --control).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

MARGIN_MS = 250.0     # T-GPLR-56 pre-registered margin (A4 conserved-sum spreads: 0.19 s, 0.06 s)
RULE = "PASS iff (control_Q - cand_Q) > MARGIN_MS AND (control_pass1_wall - cand_pass1_wall) > 0"


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


def quantity(events):
    """Returns (q_dict, missing_list, order_errors_list)."""
    missing = []
    order_errors = []
    seen = {}

    def first_at(event, pred=lambda d: True):
        for e in events:
            if e.get("event") == event and pred(e):
                return e
        return None

    proc = first_at("proc_start")
    load_b = first_at("load_begin")
    load_e = first_at("load_end")
    pass_b = first_at("host_mem", lambda d: d.get("label") == "pass_begin" and d.get("pass") == 1)
    pass_e = first_at("host_mem", lambda d: d.get("label") == "pass_end" and d.get("pass") == 1)

    if proc is None or proc.get("t_us") is None:
        missing.append("proc_start (pre-sink process-start marker)")
    if pass_e is None or pass_e.get("t_us") is None:
        missing.append('host_mem label="pass_end" pass=1 (run the arm with --pp-recon on)')
    if missing:
        return None, missing, order_errors

    q = {
        "proc_start_us": proc["t_us"],
        "pass1_end_us": pass_e["t_us"],
        "q_ms": (pass_e["t_us"] - proc["t_us"]) / 1000.0,
        # Reported, never gating:
        "pass1_wall_ms": ((pass_e["t_us"] - pass_b["t_us"]) / 1000.0) if pass_b else None,
        "pre_load_ms": ((load_b["t_us"] - proc["t_us"]) / 1000.0) if load_b else None,
        "load_ms": ((load_e["t_us"] - load_b["t_us"]) / 1000.0) if load_b and load_e else None,
        "load_result": (load_e or {}).get("detail"),
        "seq": proc.get("seq"),
    }
    # Ordering: the markers must appear in call order and before run_start's t0.
    stamps = [(e["event"], e.get("t_us")) for e in (proc, load_b, load_e, pass_b, pass_e) if e]
    for (na, ta), (nb, tb) in zip(stamps, stamps[1:]):
        if ta is not None and tb is not None and tb < ta:
            order_errors.append(f"{nb} ({tb}) precedes {na} ({ta})")
    run_start = first_at("run_start")
    if run_start and run_start.get("t0_us") is not None and pass_e["t_us"] < run_start["t0_us"]:
        order_errors.append("pass1_end precedes run_start")
    return q, missing, order_errors


def describe(q, label):
    print(f"  {label:22} Q = {q['q_ms']:9.1f} ms   (proc_start {q['proc_start_us']} -> "
          f"pass1_end {q['pass1_end_us']})")
    if q["pass1_wall_ms"] is not None:
        print(f"  {'':22} pass-1 wall = {q['pass1_wall_ms']:8.1f} ms   (never gating)")
    if q["load_ms"] is not None:
        print(f"  {'':22} pre-load = {q['pre_load_ms']:.1f} ms, load = {q['load_ms']:.1f} ms "
              f"({q['load_result']})")


def run_one(trace, control, quiet=False):
    events = load_events(trace)
    if not events:
        print(f"REFUSED: {trace} has no JSONL events")
        return 2
    q, missing, order_errors = quantity(events)
    if missing:
        print(f"REFUSED: {os.path.basename(trace)} cannot yield Q - missing {', '.join(missing)}")
        print("  (no value is printed: an absent marker is a refusal, never a zero)")
        return 2
    if order_errors:
        print(f"REFUSED: {os.path.basename(trace)} marker order is broken - {'; '.join(order_errors)}")
        return 2
    if not quiet:
        print(f"Q for {os.path.basename(trace)}:")
        describe(q, "candidate" if control else "arm")
    if control is None:
        return 0

    c_events = load_events(control)
    cq, c_missing, c_order = quantity(c_events)
    if c_missing or c_order:
        print(f"REFUSED: control {os.path.basename(control)} cannot yield Q - "
              f"{', '.join(c_missing or c_order)}")
        return 2
    print(f"control {os.path.basename(control)}:")
    describe(cq, "control")
    d_q = cq["q_ms"] - q["q_ms"]
    d_wall = (cq["pass1_wall_ms"] or 0.0) - (q["pass1_wall_ms"] or 0.0)
    print(f"  deltas: Q {d_q:+.1f} ms, pass-1 wall {d_wall:+.1f} ms  (margin {MARGIN_MS:.0f} ms)")
    if d_q > MARGIN_MS and d_wall > 0:
        print("  VERDICT: PASS (Q fell by more than the margin AND the wall fell)")
        return 0
    if d_wall > 0:
        print("  VERDICT: FAIL - RELOCATION (the wall fell while Q did not): the "
              "pre-registered rule is not satisfied")
        return 1
    print("  VERDICT: FAIL (Q did not fall by more than the margin)")
    return 1


def write_fixture(d, name, lines):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return p


def selftest():
    d = tempfile.mkdtemp(prefix="gplr56q-")
    cases = []
    try:
        def trace(proc_us, load_us, pass1_end_us, pass1_begin_us):
            out = []
            if proc_us is not None:
                out.append({"event": "proc_start", "seq": 0, "detail": "", "pre_run": 1, "t_us": proc_us})
            out.append({"event": "load_begin", "seq": 1, "detail": "m.gguf", "pre_run": 1, "t_us": load_us[0]})
            out.append({"event": "load_end", "seq": 2, "detail": "ok", "pre_run": 1, "t_us": load_us[1]})
            out.append({"event": "run_start", "t0_us": pass1_begin_us - 100, "n_layer": 40})
            out.append({"event": "host_mem", "label": "pass_begin", "pass": 1, "t_us": pass1_begin_us})
            out.append({"event": "host_mem", "label": "pass_end", "pass": 1, "t_us": pass1_end_us})
            return out

        # Realistic microsecond scale: the recorded 4k control is proc_start ~0,
        # load ~2.9 s, pass-1 bracket 3.56 s, pass1_end ~6.6 s.
        LOAD = (100_000, 3_000_000)
        control = write_fixture(d, "control.jsonl", trace(0, LOAD, 6_660_000, 3_100_000))
        good = write_fixture(d, "moved.jsonl", trace(0, LOAD, 6_100_000, 3_040_000))
        reloc = write_fixture(d, "reloc.jsonl", trace(0, LOAD, 6_660_000, 3_240_000))
        legacy = write_fixture(d, "legacy.jsonl", trace(None, LOAD, 6_660_000, 3_100_000))
        nopass = write_fixture(d, "nopar.jsonl", [ln for ln in trace(0, LOAD, 6_660_000, 3_100_000)
                                                 if not (ln.get("label") == "pass_end")])
        swapped = write_fixture(d, "swapped.jsonl", trace(0, (3_200_000, 3_000_000), 6_660_000, 3_100_000))

        def expect(name, want, argv):
            got = run_one(argv[0], argv[1], quiet=True)
            ok = (got == want)
            print(f"  [{'ok ' if ok else 'BAD'}] {name}: exit {got} (want {want})")
            cases.append(ok)

        expect("full pair, Q down 560 ms and wall down 500 ms -> PASS", 0, (good, control))
        expect("wall down 140 ms but Q flat -> FAIL (relocation)", 1, (reloc, control))
        expect("legacy trace: no proc_start -> REFUSED", 2, (legacy, control))
        expect("no pass_end marker -> REFUSED", 2, (nopass, control))
        expect("marker order broken -> REFUSED", 2, (swapped, None))
        expect("control is itself a legacy trace -> REFUSED", 2, (good, legacy))
        expect("standalone arm (no control) -> exit 0", 0, (good, None))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    bad = cases.count(False)
    print(f"SELFTEST: {'PASS' if bad == 0 else 'FAIL'} {len(cases) - bad}/{len(cases)}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="T-GPLR-56 acceptance quantity Q (host-only)")
    ap.add_argument("--trace", help="arm trace JSONL")
    ap.add_argument("--control", help="same-session control trace JSONL (applies the rule)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.trace:
        ap.error("--trace or --selftest is required")
    return run_one(a.trace, a.control)


if __name__ == "__main__":
    print(f"rule: {RULE}")
    sys.exit(main())
