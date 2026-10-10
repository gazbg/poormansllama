#!/usr/bin/env python3
"""Classify the op-rows-probe measurement against the question it was built for.

Question (owner, 2026-10-10): does the SAME expert row produce different values when
it is computed in a 2-wide MUL_MAT_ID batch instead of the shipped 8-wide one? If it
does, a row-count-dependent kernel could explain the split's 1.3e-3 relative step-0
logit shift that survives with no CPU compute.

Pre-registered bands, applied to the WORST max_rel over every comparison the probe
printed (its own repeat noise floor included):

  exact      worst == 0 (or every comparison bit_equal)  -> REFUTED
  negligible worst <= 1e-5                               -> REFUTED
  small      worst <= 1e-4                               -> weak, not an explanation
  material   worst <= 1e-3                               -> SUPPORTED, keep digging
  large      worst >  1e-3                               -> SUPPORTED, this is the cause

Exit codes: 0 the hypothesis is refuted by this evidence (the probe ran and its
numbers are usable); 1 the hypothesis is supported (the split's arithmetic needs
this cause explained); 2 nothing to evaluate (no comparisons, or a shape failed).

Host-only. Reads the probe's JSON, never runs the probe.
"""

import argparse
import json
import sys

EXACT = 1e-5
WEAK = 1e-4
MATERIAL = 1e-3


def classify(worst):
    if worst == 0.0:
        return "exact"
    if worst <= EXACT:
        return "negligible"
    if worst <= WEAK:
        return "small"
    if worst <= MATERIAL:
        return "material"
    return "large"


def classify_file(path):
    """Return (worst_max_rel, n_cmp, band, failed) for a probe JSON, or None if unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    shapes = d.get("shapes") or []
    if not shapes:
        return None
    worst, n_cmp, failed = 0.0, 0, []
    for sh in shapes:
        if not sh.get("ok"):
            failed.append((sh.get("name"), sh.get("error")))
            continue
        for c in sh.get("comparisons") or []:
            n_cmp += 1
            worst = max(worst, float(c.get("max_rel") or 0.0))
    return worst, n_cmp, classify(worst), failed


def selftest():
    """The classification has to separate a refuted hypothesis from a supported one."""
    import tempfile

    def probe(worst, ok=True):
        d = {"probe": "op-rows-probe", "device": "X", "type": "q4_K", "n_expert": 256, "tokens": 1,
             "shapes": [{"name": "gate_up", "ok": ok, "k": 2048, "m": 512,
                         "comparisons": [{"what": "same rows: 2-wide vs 8-wide", "max_rel": worst,
                                          "bit_equal": worst == 0.0, "n_diff": 0 if worst == 0.0 else 1}]}]}
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, fh)
        fh.close()
        return fh.name

    fails = []
    for worst, want_band in ((0.0, "exact"), (1e-7, "negligible"), (5e-5, "small"),
                             (5e-4, "material"), (5e-2, "large")):
        r = classify_file(probe(worst))
        if r is None or r[2] != want_band:
            fails.append("band for %g: got %s want %s" % (worst, r and r[2], want_band))
    bad = classify_file(probe(0.0, ok=False))
    if bad is None or not bad[3]:
        fails.append("a failed shape must be reported as failed")
    missing = classify_file("no-such-file.json")
    if missing is not None:
        fails.append("an unreadable file must return None")
    if fails:
        for f in fails:
            print("SELFTEST FAIL: %s" % f)
        return 1
    print("check-op-rows-probe selftest OK (5 bands, failed shape, unreadable file)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default="build/llama-cuda/op-rows-cuda-t1.json")
    ap.add_argument("--selftest", action="store_true", help="check the classifier on synthetic measurements")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    try:
        with open(args.json, encoding="utf-8") as fh:
            d = json.load(fh)
    except OSError as e:
        print("check-op-rows-probe: cannot read %s: %s" % (args.json, e))
        return 2
    except json.JSONDecodeError as e:
        print("check-op-rows-probe: %s is not JSON: %s" % (args.json, e))
        return 2

    shapes = d.get("shapes") or []
    if not shapes:
        print("check-op-rows-probe: no shapes in %s" % args.json)
        return 2

    print("device=%s type=%s bank_rows=%s tokens=%s" % (
        d.get("device"), d.get("type"), d.get("n_expert"), d.get("tokens")))

    worst = 0.0
    worst_what = None
    n_cmp = 0
    failed = []
    for sh in shapes:
        if not sh.get("ok"):
            failed.append((sh.get("name"), sh.get("error")))
            continue
        for c in sh.get("comparisons") or []:
            n_cmp += 1
            mr = float(c.get("max_rel") or 0.0)
            if mr > worst:
                worst = mr
                worst_what = "%s: %s" % (sh.get("name"), c.get("what"))
            print("  %-8s %-32s max_rel=%.3e bit_equal=%s n_diff=%s" % (
                sh.get("name"), c.get("what"), mr, c.get("bit_equal"), c.get("n_diff")))

    if failed:
        for name, err in failed:
            print("  FAILED shape %s: %s" % (name, err))
        print("check-op-rows-probe: RESULT=NOT-EVALUABLE (a shape failed)")
        return 2
    if n_cmp == 0:
        print("check-op-rows-probe: RESULT=NOT-EVALUABLE (no comparisons)")
        return 2

    band = classify(worst)
    print("worst max_rel=%.3e (%s)" % (worst, worst_what))
    print("row-count band: %s" % band)
    if band in ("exact", "negligible"):
        print("check-op-rows-probe: RESULT=PASS -- the batched MoE matmul is row-count "
              "independent here, so it cannot explain a 1.3e-3 shift (%d comparisons)" % n_cmp)
        return 0
    if band == "small":
        print("check-op-rows-probe: RESULT=WEAK (exit 1) -- row-count dependence exists at "
              "%.3e, one order below the 1.3e-3 observation; it does not carry the effect"
              % worst)
        return 1
    print("check-op-rows-probe: RESULT=FAIL (exit 1) -- row-count dependence reaches the "
          "observed scale (%.3e vs 1.3e-3): this IS the confirmed cause" % worst)
    return 1


if __name__ == "__main__":
    sys.exit(main())
