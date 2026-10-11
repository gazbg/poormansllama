#!/usr/bin/env python3
"""Mechanical evaluation of contract G2 (decode-phase cache resize) in docs/plans/tg-beat-stock-8gb-262k.md.

Thresholds are constants copied from the contract, never read from results. Host-only, stdlib-only.
It also tallies every resized arm (flag --tg-hybrid-slots) across all g2-* sessions, failures included, so a crash cannot be left out.

  python tools/check-g2-contract.py [--root .]      exit 0 iff G2-1..G2-7 hold
"""
import argparse
import glob
import json
import os
import re
import sys

R_MIN = 1.10      # HYB28 / HYB14
STK_MIN = 1.80    # HYB28 / STK
CAP_MIB = 8192
DLOGIT_MAX = 0.30
INERT = 0.03      # g2 HYB14 vs g1 HYB14
CORE = ["g2-code", "g2-prose", "g2-anchor"]


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    B = os.path.join(a.root, "build", "llama-cuda")
    rows = []

    def row(cid, ok, text):
        rows.append((cid, ok, text))

    ver = {s: jload(os.path.join(B, s, "%s-verdict.json" % s)) for s in CORE}
    ag = {s: ver[s]["aggregate"] for s in CORE}

    # G2-1: speed on code and prose
    parts, ok1 = [], True
    for s in ("g2-code", "g2-prose"):
        g = ag[s]
        r_h = g["HYB28"]["mean"] / g["HYB"]["mean"]
        r_s = g["HYB28"]["mean"] / g["STK"]["mean"]
        sep = g["HYB28"]["min"] > g["STK"]["max"]
        ok1 = ok1 and r_h >= R_MIN and r_s >= STK_MIN and sep and g["HYB28"]["n_ok"] == g["HYB28"]["n"]
        parts.append("%s: HYB28 %.2f / HYB14 %.2f = %.3fx, / STK %.2f = %.3fx, min>maxSTK %s" % (
            s[3:], g["HYB28"]["mean"], g["HYB"]["mean"], r_h, g["STK"]["mean"], r_s, sep))
    row("G2-1", ok1, "; ".join(parts) + " (need >= %.2fx and >= %.2fx)" % (R_MIN, STK_MIN))

    # G2-2: whole-GPU peak of HYB28 (50 ms poller)
    peaks = {s: [r.get("vram_peak_mib") for n, r in ver[s]["arms"].items() if r["base"] == "HYB28"] for s in CORE}
    ok2 = all(p and all(isinstance(x, (int, float)) and x <= CAP_MIB for x in p) for p in peaks.values())
    row("G2-2", ok2, "HYB28 whole-GPU peaks (50 ms poller) %s <= %d MiB" % (
        {s[3:]: max(p) for s, p in peaks.items()}, CAP_MIB))

    # G2-3: needle QA on the anchor
    arms3 = [r for n, r in ver["g2-anchor"]["arms"].items() if r["base"] == "HYB28"]
    good3 = [r for r in arms3 if isinstance(r.get("qa"), dict) and r["qa"]["positive_pass"] and r["qa"]["negative_fails_as_designed"]]
    row("G2-3", bool(arms3) and len(good3) == len(arms3),
        "anchor (511 tokens) HYB28 needle QA positive pass and negative control fails as designed in %d/%d reps" % (len(good3), len(arms3)))

    # G2-4: logit drift vs the unresized control
    d, ok4 = {}, True
    for s in CORE:
        vals = [r["summary"].get("vs_control", {}).get("dlogit_median") for n, r in ver[s]["arms"].items() if r["base"] == "HYB28"]
        vals = [v for v in vals if v is not None]
        d[s[3:]] = max(vals) if vals else None
        ok4 = ok4 and bool(vals) and max(vals) <= DLOGIT_MAX
    row("G2-4", ok4, "HYB28 vs HYB14 same-context dlogit median up to the first divergence %s <= %.2f" % (d, DLOGIT_MAX))

    # G2-5: default inertness against the g1 build
    g1 = {
        "anchor": jload(os.path.join(B, "g1-speedn", "g1-speedn-verdict.json"))["aggregate"]["HYB"]["mean"],
        "prose": jload(os.path.join(B, "g1-aer-prose", "g1-aer-prose-verdict.json"))["aggregate"]["HYB"]["mean"],
        "code": jload(os.path.join(B, "g1-aer-code", "g1-aer-code-verdict.json"))["aggregate"]["HYB"]["mean"],
    }
    g2 = {"anchor": ag["g2-anchor"]["HYB"]["mean"], "prose": ag["g2-prose"]["HYB"]["mean"], "code": ag["g2-code"]["HYB"]["mean"]}
    dev = {k: g2[k] / g1[k] - 1.0 for k in g1}
    ct = os.path.join(B, "g3-ctest.log")
    ctok, cttxt = None, "no ctest log"
    if os.path.exists(ct):
        m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", open(ct, encoding="utf-8", errors="replace").read())
        if m:
            ctok = m.group(1) == "100" and m.group(2) == "0"
            cttxt = "ctest %s/%s" % (int(m.group(3)) - int(m.group(2)), m.group(3))
    row("G2-5", all(abs(v) <= INERT for v in dev.values()) and bool(ctok),
        "HYB14 g2 vs g1: %s (need within +-%.0f%%); host engine %s" % (
            {k: "%+.1f%%" % (100 * v) for k, v in dev.items()}, 100 * INERT, cttxt))

    # G2-6: provenance
    waivers = ""
    wp = os.path.join(B, "verdicts", "provenance-waivers.log")
    if os.path.exists(wp):
        waivers = open(wp, encoding="utf-8", errors="replace").read()
    bad = []
    for tag in CORE:
        v = jload(os.path.join(B, "verdicts", tag + ".verdict.json"))
        prov = (v.get("fields") or {}).get("provenance") or {}
        ok = (v.get("status") == "PASS" and prov.get("binary_state") == "hashed" and prov.get("sources_newer_than_binary") is False
              and not (v.get("fields") or {}).get("refused_unprovenanced") and not re.search(r"\b%s\b" % re.escape(tag), waivers))
        if not ok:
            bad.append(tag)
    row("G2-6", not bad, "%d core sessions attested without waiver (%s)%s" % (len(CORE) - len(bad), ", ".join(CORE),
                                                                         "; BAD: %s" % bad if bad else ""))

    # G2-7: records
    plan = open(os.path.join(a.root, "docs", "plans", "tg-beat-stock-8gb-262k.md"), encoding="utf-8").read()
    cur = open(os.path.join(a.root, "docs", "currentstate.md"), encoding="utf-8", errors="replace").read()
    row("G2-7", "G2 ACCEPTANCE RESULT" in plan and "IDEA LEDGER" in plan and "G2 decode-phase" in cur,
        "plan has G2 result + idea ledger: %s / %s; currentstate entry: %s" % (
            "G2 ACCEPTANCE RESULT" in plan, "IDEA LEDGER" in plan, "G2 decode-phase" in cur))

    for cid, ok, text in rows:
        print("%-5s %-4s %s" % (cid, "PASS" if ok else "FAIL", text))

    # reliability tally over every resized arm in every g2-* session (reported, never filtered)
    tot = fail = 0
    fails = []
    for vp in sorted(glob.glob(os.path.join(B, "g2-*", "g2-*-verdict.json"))):
        v = jload(vp)
        for n, r in v["arms"].items():
            if "--tg-hybrid-slots" in r.get("flags", []):
                tot += 1
                if r.get("rc") != 0:
                    fail += 1
                    fails.append("%s/%s rc=%s" % (v["session"], n, r.get("rc")))
    print("\nRELIABILITY (reported, not gated): resized arms %d, non-zero exits %d %s" % (tot, fail, fails or ""))
    allok = all(r[1] for r in rows)
    print("CONTRACT G2: %s (mechanical evaluation; acceptance is the agent's own decision, not owner approval)" % (
        "ALL G2-1..G2-7 HOLD" if allok else "NOT ALL CRITERIA HOLD"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
