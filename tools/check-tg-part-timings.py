#!/usr/bin/env python3
"""Per-part TG compute timings: can this trace attribute TG transport against compute?

T-GPLR-13's ladder recorded 0/120 nonzero TG `layer_compute` timings and that number was
for years indistinguishable from "TG compute costs nothing". It was not a zero cost: the
row-set TG executor at that time emitted `compute_us=0` for a timing it never took. The
F3/H8 fix (2026-09-14, docs/gplr13-bottlenecks-hypotheses.md section 3.6) now times both
graph parts and emits `attn_compute_us`/`attn_setup_us` (part A = attention + residual +
post-norm + router) and `ffn_compute_us`/`ffn_setup_us` (part B = FFN + shared experts on
the staged rows) alongside the unchanged `compute_us`/`setup_us` (the A+B sums).

So a zero in a TG timing has TWO meanings and a reader must never confuse them:

    NOT-TAKEN          the part fields are ABSENT -> the path never measured them. A
                       consumer that does `e.get("attn_compute_us", 0.0)` reads this as
                       a zero-cost part. That is the defect class this checker removes.
    PARTS-ALL-ZERO     the fields are present and all zero -> the F3 defect pattern (a
                       path that claims to have timed the split and reports nothing), or
                       an arm whose timing is structurally broken. Never a measurement.
    MEASURED           present, and at least one part carries real time. Individual zero
                       values inside a MEASURED group are REAL measured zeros (a part
                       that took under a microsecond) and must not be flagged.

The path is derived from the trace's own signals, never from its filename: a trace
carrying `rowset_fetch` events ran the two-part row-set TG executor, so its phase-1
`layer_compute` group is REQUIRED to be MEASURED. A batch-major TG trace (no
`rowset_fetch`) fuses the layer into one graph and has no part split by construction --
NOT-TAKEN there is a true statement about the path, not a defect, and it can never supply
a transport-vs-compute attribution.

    python tools/check-tg-part-timings.py --trace <arm.jsonl> [--require-parts]
    python tools/check-tg-part-timings.py --census [--root build/llama-cuda]
    python tools/check-tg-part-timings.py --selftest

Exit, one meaning per code: 0 = classified (MEASURED, or NOT-TAKEN on a path with no split
by construction); 1 = PARTS-ALL-ZERO -- a group that CLAIMS a split and reports all zero,
which is a defect finding on any path and outranks every other reading; 2 = REFUSED, the
required attribution input is NOT-TAKEN or MIXED -- NOT A PASS, because a gate that cannot
evaluate its input must never report success (the T-GPLR-53 decay rule refused its own
input for exactly this reason).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

PART_KEYS = ("attn_compute_us", "attn_setup_us", "ffn_compute_us", "ffn_setup_us")
COMPUTE_PARTS = ("attn_compute_us", "ffn_compute_us")

NOT_TAKEN = "NOT-TAKEN"
ALL_ZERO = "PARTS-ALL-ZERO"
MEASURED = "MEASURED"
MIXED = "MIXED"
EMPTY = "NO-EVENTS"


def load_events(path):
    out = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def classify_group(events):
    """Classify one (trace, phase) group of layer_compute events."""
    with_parts = [e for e in events if any(k in e for k in PART_KEYS)]
    if not events:
        return {"verdict": EMPTY, "n": 0, "n_with_parts": 0, "n_nonzero_part": 0}
    if not with_parts:
        # never taken: the fields are ABSENT, so no reader may read them as 0
        return {"verdict": NOT_TAKEN, "n": len(events), "n_with_parts": 0, "n_nonzero_part": 0}
    nonzero = sum(1 for e in with_parts
                  if any(float(e.get(k, 0.0) or 0.0) > 0.0 for k in COMPUTE_PARTS))
    if len(with_parts) != len(events):
        verdict = MIXED
    elif nonzero == 0:
        verdict = ALL_ZERO
    else:
        verdict = MEASURED
    return {"verdict": verdict, "n": len(events), "n_with_parts": len(with_parts),
            "n_nonzero_part": nonzero}


def classify_trace(path):
    events = load_events(path)
    groups = {}
    for e in events:
        if e.get("event") != "layer_compute":
            continue
        groups.setdefault(e.get("phase"), []).append(e)
    rowset_groups = [g for g in events if g.get("event") == "rowset_fetch"]
    run_id = next((e.get("run_id") for e in events if e.get("run_id") is not None), None)
    return {
        "path": path,
        "run_id": run_id,
        "rowset_path": bool(rowset_groups),
        "n_rowset_fetch": len(rowset_groups),
        "groups": {ph: classify_group(ev) for ph, ev in sorted(groups.items(), key=lambda kv: str(kv[0]))},
    }


def describe(info):
    r = info["groups"]
    lines = []
    for ph, g in r.items():
        extra = ""
        if g["verdict"] == MEASURED:
            extra = f" ({g['n_nonzero_part']}/{g['n_with_parts']} events carry a nonzero part)"
        elif g["verdict"] == NOT_TAKEN:
            extra = " (fields ABSENT: never measured -- a reader must NOT read this as 0)"
        elif g["verdict"] == ALL_ZERO:
            extra = " (fields present, every part zero: the F3 defect pattern)"
        elif g["verdict"] == MIXED:
            extra = f" (only {g['n_with_parts']}/{g['n']} events carry parts)"
        lines.append(f"      phase {ph}: n={g['n']:<6} {g['verdict']}{extra}")
    return lines


def evaluate(info, require_parts):
    """Return (exit_code, headline). One meaning per code, in this order."""
    # 1 = defect, on any path: a group that carries the part fields and reports all zero
    zeros = [ph for ph, g in info["groups"].items() if g["verdict"] == ALL_ZERO]
    if zeros:
        return 1, (f"DEFECT: phase {zeros} carries part fields and reports every part zero "
                   f"(the F3 defect pattern) -- not a measurement")
    tg = info["groups"].get(1)
    if info["rowset_path"]:
        # the two-part row-set TG executor: phase 1 MUST carry measured parts
        if tg is None or tg["verdict"] in (NOT_TAKEN, MIXED, EMPTY):
            got = tg["verdict"] if tg else EMPTY
            return 2, (f"REFUSED: row-set TG phase-1 group is {got} -- NOT A PASS "
                       f"(no attribution input; the timing was never taken, so it is not a zero)")
        return 0, "MEASURED: row-set TG phase 1 carries per-part timings"
    if require_parts:
        return 2, "REFUSED: --require-parts but this trace ran no row-set TG (parts are not decomposable here)"
    if tg is not None and tg["verdict"] == NOT_TAKEN:
        return 0, "NEVER MEASURED: batch-major TG has no part split by construction (not a zero cost)"
    return 0, "classified"


def check_trace(path, require_parts):
    if not os.path.exists(path):
        print(f"REFUSED: no such trace: {path}")
        return 2
    info = classify_trace(path)
    code, headline = evaluate(info, require_parts)
    print(f"trace: {path}")
    print(f"  run_id={info['run_id']} rowset_tg={'yes' if info['rowset_path'] else 'no'} "
          f"rowset_fetch={info['n_rowset_fetch']}")
    for line in describe(info):
        print(line)
    print(f"  => {headline}")
    return code


def census(root, quiet):
    traces = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
    counts = {}
    rows = []
    for t in traces:
        try:
            info = classify_trace(t)
        except OSError:
            continue
        if not info["groups"]:
            continue
        tg = info["groups"].get(1)
        tgv = tg["verdict"] if tg else EMPTY
        code, _ = evaluate(info, require_parts=False)
        counts[tgv] = counts.get(tgv, 0) + 1
        rows.append((t, info, tgv, code))
    if not quiet:
        print("phase-1 layer_compute classification per trace:")
        for t, info, tgv, code in rows:
            tag = "rowset" if info["rowset_path"] else "batch "
            print(f"  {os.path.relpath(t):<72} {tag} run_id={info['run_id']} phase1={tgv}")
    defects = [t for t, _, tgv, code in rows if code == 1]
    print()
    print(f"census root: {root}")
    print(f"  traces with layer_compute: {len(rows)}")
    for k in (MEASURED, NOT_TAKEN, ALL_ZERO, MIXED, EMPTY):
        if counts.get(k):
            print(f"    {k:<16} {counts[k]}")
    print(f"  attribution-usable (phase-1 MEASURED): {counts.get(MEASURED, 0)}")
    print(f"  PARTS-ALL-ZERO defects: {len(defects)}")
    if defects:
        for t in defects:
            print(f"    DEFECT {os.path.relpath(t)}")
        return 1
    return 0


def _lc(phase, **kw):
    e = {"event": "layer_compute", "run_id": 4242, "layer": 0, "phase": phase,
         "compute_us": 100.0, "setup_us": 5.0, "t_us": 1}
    e.update(kw)
    return e


def _write(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def selftest():
    """Hermetic fixtures. The negative controls are the recorded defect shapes."""
    d = tempfile.mkdtemp(prefix="tgparts-")
    fails = []

    def case(name, events, expect_verdict, expect_code, require_parts=False):
        p = os.path.join(d, name + ".jsonl")
        _write(p, events)
        info = classify_trace(p)
        code, _ = evaluate(info, require_parts)
        got = info["groups"].get(1, {}).get("verdict", EMPTY)
        ok = (got == expect_verdict and code == expect_code)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: phase-1={got} exit={code} "
              f"(expected {expect_verdict}/{expect_code})")
        if not ok:
            fails.append(name)

    # 1. the recorded ladder shape: 120 phase-1 events, compute_us=0, NO part fields.
    #    A naive reader says "TG compute = 0"; the check must say NEVER TAKEN.
    ladder = ([{"event": "rowset_fetch", "run_id": 9, "layer": i % 40, "chunk": c} for i in range(120) for c in [1]]
              + [_lc(1, compute_us=0.0, setup_us=0.0, chunk=c, layer=i) for i in range(120) for c in [1]]
              + [_lc(0, compute_us=800.0, layer=0)])
    case("prefix-ladder-0of120", ladder, NOT_TAKEN, 2, require_parts=True)

    # 2. fields present, every part zero: the F3 defect pattern, refused either way
    #    A defect outranks requirement: exit 1 either way, and it is reported as a defect
    #    rather than as "cannot certify".
    allzero = ([{"event": "rowset_fetch", "run_id": 9}] * 3
               + [_lc(1, layer=i, **{k: 0.0 for k in PART_KEYS}) for i in range(120)]
               + [_lc(0, layer=0)])
    case("parts-all-zero", allzero, ALL_ZERO, 1, require_parts=True)
    case("parts-all-zero-norequire", allzero, ALL_ZERO, 1, require_parts=False)
    #    and the same defect on a batch-major trace: still exit 1, never a silent 0
    allzero_batch = ([_lc(1, layer=i, **{k: 0.0 for k in PART_KEYS}) for i in range(120)]
                     + [_lc(0, layer=0)])
    case("parts-all-zero-batch-major", allzero_batch, ALL_ZERO, 1, require_parts=False)

    # 3. MEASURED, including a REAL measured zero in one part (must not be flagged)
    meas = ([{"event": "rowset_fetch", "run_id": 9}] * 3
            + [_lc(1, layer=i, attn_compute_us=(0.0 if i == 7 else 12.0), attn_setup_us=1.0,
                   ffn_compute_us=4.0, ffn_setup_us=0.5) for i in range(120)]
            + [_lc(0, layer=0)])
    case("measured-with-real-zero", meas, MEASURED, 0, require_parts=True)
    case("measured-norequire", meas, MEASURED, 0, require_parts=False)

    # 4. batch-major TG: real compute, no part split by construction. Not a defect, and
    #    --require-parts correctly refuses it as an attribution input.
    batch = [_lc(1, layer=i, compute_us=250.0) for i in range(120)] + [_lc(0, layer=0)]
    case("batch-major-tg", batch, NOT_TAKEN, 0, require_parts=False)
    case("batch-major-required", batch, NOT_TAKEN, 2, require_parts=True)

    # 5. half the group carries parts: MIXED, cannot be read as either
    mixed = ([{"event": "rowset_fetch", "run_id": 9}] * 3
             + [_lc(1, layer=i, **({"attn_compute_us": 9.0, "attn_setup_us": 1.0,
                                    "ffn_compute_us": 3.0, "ffn_setup_us": 0.5} if i < 60 else {}))
                for i in range(120)]
             + [_lc(0, layer=0)])
    case("mixed", mixed, MIXED, 2, require_parts=True)

    # 6. the distinction itself: the same 120 all-zero phase-1 events must classify
    #    NOT-TAKEN without part fields and ALL-ZERO with them. This is the whole point.
    without = classify_trace(os.path.join(d, "prefix-ladder-0of120.jsonl"))["groups"][1]["verdict"]
    withparts = classify_trace(os.path.join(d, "parts-all-zero.jsonl"))["groups"][1]["verdict"]
    ok = without != withparts
    print(f"  [{'PASS' if ok else 'FAIL'}] never-taken ({without}) is distinguishable from "
          f"all-zero ({withparts})")
    if not ok:
        fails.append("distinction")

    shutil.rmtree(d, ignore_errors=True)
    print(f"\nselftest: {'OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="trace to classify")
    ap.add_argument("--require-parts", action="store_true",
                    help="refuse (exit 2) unless the trace's TG layer_compute carries measured part timings")
    ap.add_argument("--census", action="store_true", help="classify every trace under --root")
    ap.add_argument("--root", default="build/llama-cuda", help="census root (default: build/llama-cuda)")
    ap.add_argument("--quiet", action="store_true", help="census: summary only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.trace:
        return check_trace(a.trace, a.require_parts)
    if a.census:
        return census(a.root, a.quiet)
    ap.error("--trace, --census or --selftest is required")


if __name__ == "__main__":
    sys.exit(main())
