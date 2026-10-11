#!/usr/bin/env python3
"""Mechanical evaluation of contract G3 (release the prefill-sized scheduler buffers) in docs/plans/tg-beat-stock-8gb-262k.md.

Thresholds are constants copied from the contract (with amendment A3), never read from results. Host-only, stdlib-only.
It also tallies every --tg-release-compute arm across all g4-* sessions, failures included, and prints the contention test
(g4-load) as a REPORT: its predictions are not gated.

  python tools/check-g3-contract.py [--root .] [--tag g4]      exit 0 iff G3-1..G3-6 hold

--tag names the build under test: sessions are <tag>-code, <tag>-prose, <tag>-anchor (speed, QA, VRAM), <tag>-rel (reliability),
<tag>-ab (same-session A/B against the g2 build; reported with the reliability tally), <tag>-nsys (profile), <tag>-load (contention,
reported, never gated).
"""
import argparse
import glob
import json
import os
import re
import sys

REL_MIN_MIB = 700.0   # G3-1: the re-reserve must release at least this much
CAP_MIB = 8192        # G3-1: whole-GPU peak of the anchor HYB42R arm
R_H28 = 1.06          # G3-2: HYB42R / HYB28
R_STK = 2.00          # G3-2: HYB42R / STK
DLOGIT_MAX = 0.30     # G3-3
N_REL = 12            # G3-4: HYB42R runs without a failure
INERT = 0.03          # G3-5 (amendment A3 baselines)
CORE_SUFFIX = ["code", "prose", "anchor"]
GATED_SUFFIX = ["code", "prose", "anchor", "rel"]       # sessions whose attested verdict G3-6 requires
TALLY_SUFFIX = ["code", "prose", "anchor", "rel", "ab", "load"]   # every session whose hybrid arms enter the reliability tally


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def rtext(p):
    # runner logs may be UTF-16: drop NULs instead of decoding wrongly
    with open(p, "rb") as fh:
        return fh.read().replace(b"\x00", b"").decode("utf-8", "replace")


def arms_of(ver, base):
    return [r for n, r in ver["arms"].items() if r["base"] == base]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="g4")
    a = ap.parse_args()
    B = os.path.join(a.root, "build", "llama-cuda")
    T = a.tag
    CORE = ["%s-%s" % (T, x) for x in CORE_SUFFIX]
    ALL_SESSIONS = ["%s-%s" % (T, x) for x in GATED_SUFFIX]
    rows = []

    def row(cid, ok, text):
        rows.append((cid, ok, text))

    ver = {s: jload(os.path.join(B, s, "%s-verdict.json" % s)) for s in CORE}
    ag = {s: ver[s]["aggregate"] for s in CORE}

    # G3-1: release size and the whole-GPU peak on the anchor
    rel = []
    for r in arms_of(ver[T + "-anchor"], "HYB42R"):
        m = re.search(r"re-reserved for decode, ([0-9.]+) MiB released", rtext(r["log"]))
        rel.append(float(m.group(1)) if m else None)
    peaks = [r.get("vram_peak_mib") for r in arms_of(ver[T + "-anchor"], "HYB42R")]
    ok1 = (bool(rel) and all(x is not None and x >= REL_MIN_MIB for x in rel)
           and bool(peaks) and all(isinstance(x, (int, float)) and x <= CAP_MIB for x in peaks))
    row("G3-1", ok1, "anchor HYB42R released %s MiB (need >= %.0f); whole-GPU peak (50 ms poller) %s MiB (need <= %d)" % (
        rel, REL_MIN_MIB, peaks, CAP_MIB))

    # G3-2: speed on code and prose
    parts, ok2 = [], True
    for s in (T + "-code", T + "-prose"):
        g = ag[s]
        r_h = g["HYB42R"]["mean"] / g["HYB28"]["mean"]
        r_s = g["HYB42R"]["mean"] / g["STK"]["mean"]
        sep = g["HYB42R"]["min"] > g["HYB28"]["max"]
        ok2 = ok2 and r_h >= R_H28 and r_s >= R_STK and sep and g["HYB42R"]["n_ok"] == g["HYB42R"]["n"]
        parts.append("%s: HYB42R %.2f / HYB28 %.2f = %.3fx, / STK %.2f = %.3fx, min>maxHYB28 %s" % (
            s[len(T) + 1:], g["HYB42R"]["mean"], g["HYB28"]["mean"], r_h, g["STK"]["mean"], r_s, sep))
    row("G3-2", ok2, "; ".join(parts) + " (need >= %.2fx and >= %.2fx)" % (R_H28, R_STK))

    # G3-3: needle QA and drift on the anchor
    arms3 = arms_of(ver[T + "-anchor"], "HYB42R")
    good3 = [r for r in arms3 if isinstance(r.get("qa"), dict) and r["qa"]["positive_pass"] and r["qa"]["negative_fails_as_designed"]]
    dl = [r["summary"].get("vs_control", {}).get("dlogit_median") for r in arms3]
    dl = [v for v in dl if v is not None]
    ok3 = bool(arms3) and len(good3) == len(arms3) and bool(dl) and max(dl) <= DLOGIT_MAX
    row("G3-3", ok3, "anchor HYB42R needle QA positive pass and negative control fails as designed in %d/%d reps; "
                     "dlogit median vs HYB28 up to the first divergence %s <= %.2f" % (len(good3), len(arms3), dl, DLOGIT_MAX))

    # G3-4: reliability over every --tg-release-compute arm in every g4 session
    tot = fail = 0
    fails = []
    for s in ALL_SESSIONS:
        vp = os.path.join(B, s, "%s-verdict.json" % s)
        if not os.path.exists(vp):
            continue
        v = jload(vp)
        for n, r in v["arms"].items():
            if "--tg-release-compute" in r.get("flags", []):
                tot += 1
                if r.get("rc") != 0:
                    fail += 1
                    fails.append("%s/%s rc=%s" % (s, n, r.get("rc")))
    row("G3-4", tot >= N_REL and fail == 0, "HYB42R runs %d (need >= %d), non-zero exits %d %s" % (tot, N_REL, fail, fails or ""))
    # every hybrid (LM) arm of every tally session, failures included, reported beside the gated line
    all_tot = all_fail = 0
    all_fails = []
    for suf in TALLY_SUFFIX:
        vp = os.path.join(B, "%s-%s" % (T, suf), "%s-%s-verdict.json" % (T, suf))
        if not os.path.exists(vp):
            continue
        for n, r in jload(vp)["arms"].items():
            if r.get("engine", "lm") == "lm" and r.get("env", {}).get("LLAMA_MOE_HYBRID"):
                all_tot += 1
                if r.get("rc") != 0:
                    all_fail += 1
                    all_fails.append("%s-%s/%s rc=%s" % (T, suf, n, r.get("rc")))
    print("RELIABILITY (reported, all hybrid arms of the tally sessions): runs %d, non-zero exits %d %s" % (all_tot, all_fail, all_fails or ""))

    # G3-5: default inertness against the baselines of amendment A3
    base14 = {"code": [jload(os.path.join(B, "g3-soak", "g3-soak-verdict.json"))["aggregate"]["HYB"]["mean"],
                       jload(os.path.join(B, "g2-code", "g2-code-verdict.json"))["aggregate"]["HYB"]["mean"]],
              "prose": [jload(os.path.join(B, "g2-prose", "g2-prose-verdict.json"))["aggregate"]["HYB"]["mean"]],
              "anchor": [jload(os.path.join(B, "g2-anchor", "g2-anchor-verdict.json"))["aggregate"]["HYB"]["mean"]]}
    base28 = {k: [jload(os.path.join(B, "g2-" + k, "g2-%s-verdict.json" % k))["aggregate"]["HYB28"]["mean"]] for k in ("code", "prose", "anchor")}
    cur = {k: ag[T + "-" + k] for k in ("code", "prose", "anchor")}
    dev14 = {k: max(abs(cur[k]["HYB"]["mean"] / b - 1.0) for b in base14[k]) for k in cur}
    dev28 = {k: max(abs(cur[k]["HYB28"]["mean"] / b - 1.0) for b in base28[k]) for k in cur}
    ct = os.path.join(B, T + "-ctest.log")
    ctok, cttxt = None, "no ctest log"
    if os.path.exists(ct):
        m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", rtext(ct))
        if m:
            ctok = m.group(1) == "100" and m.group(2) == "0"
            cttxt = "ctest %s/%s" % (int(m.group(3)) - int(m.group(2)), m.group(3))
    row("G3-5", all(v <= INERT for v in dev14.values()) and all(v <= INERT for v in dev28.values()) and bool(ctok),
        "g4 vs g2/g3 baselines, worst deviation: HYB14 %s, HYB28 %s (need within +-%.0f%%); host engine %s" % (
            {k: "%.1f%%" % (100 * v) for k, v in dev14.items()}, {k: "%.1f%%" % (100 * v) for k, v in dev28.items()}, 100 * INERT, cttxt))

    # G3-6: provenance and records
    waivers = ""
    wp = os.path.join(B, "verdicts", "provenance-waivers.log")
    if os.path.exists(wp):
        waivers = rtext(wp)
    bad = []
    for tag in ALL_SESSIONS:
        vpath = os.path.join(B, "verdicts", tag + ".verdict.json")
        if not os.path.exists(vpath):
            bad.append(tag + " (no verdict)")
            continue
        v = jload(vpath)
        prov = (v.get("fields") or {}).get("provenance") or {}
        ok = (v.get("status") == "PASS" and prov.get("binary_state") == "hashed" and prov.get("sources_newer_than_binary") is False
              and not (v.get("fields") or {}).get("refused_unprovenanced") and not re.search(r"\b%s\b" % re.escape(tag), waivers))
        if not ok:
            bad.append(tag)
    plan = rtext(os.path.join(a.root, "docs", "plans", "tg-beat-stock-8gb-262k.md"))
    cs = rtext(os.path.join(a.root, "docs", "currentstate.md"))
    recs = ("G3 ACCEPTANCE RESULT" in plan, "G3 release the prefill-sized scheduler buffers" in cs)
    row("G3-6", not bad and all(recs), "%d/%d g4 sessions attested without waiver%s; plan G3 result %s, currentstate G3 entry %s" % (
        len(ALL_SESSIONS) - len(bad), len(ALL_SESSIONS), "; BAD: %s" % bad if bad else "", recs[0], recs[1]))

    for cid, ok, text in rows:
        print("%-5s %-4s %s" % (cid, "PASS" if ok else "FAIL", text))

    # contention test: reported, never gated
    lp = os.path.join(B, T + "-load", T + "-load-verdict.json")
    if os.path.exists(lp):
        v = jload(lp)
        print("\nCONTENTION TEST load (reported, not gated):")
        means = {}
        for base in ("L0", "L8", "L24"):
            ar = arms_of(v, base)
            vals = [r["summary"].get("tg_tok_s") for r in ar if r.get("rc") == 0]
            vals = [x for x in vals if isinstance(x, (int, float))]
            if not vals:
                vals = [x for x in v["aggregate"].get(base, {}).get("tok_s", []) if isinstance(x, (int, float))]
            means[base] = sum(vals) / len(vals) if vals else None
            n_stall = n_worst_runs = 0
            worst = 0.0
            traps = 0
            hb_stag = hb_run = 0
            for r in ar:
                txt = rtext(r["log"])
                if r.get("rc") != 0 or re.search(r"timed out|__trap|0xC0000409|unspecified launch failure", txt):
                    traps += 1
                m = re.search(r"watchdog summary: (\d+) requests observed pending for 5 ms or more, worst observed pending ([0-9.]+) ms", txt)
                if m:
                    n_stall += int(m.group(1))
                    if float(m.group(2)) > 20.0:
                        n_worst_runs += 1
                    worst = max(worst, float(m.group(2)))
                for mm in re.finditer(r"unanswered for ([0-9.]+) ms .*? phase (\d+)\.(\d+) for ([0-9.]+) ms .*?heartbeat age ([0-9.]+) ms", txt):
                    hb_run += 1
                    pend, hb_age = float(mm.group(1)), float(mm.group(5))
                    if hb_age >= 0.5 * pend:
                        hb_stag += 1
            print("  %-4s runs %d  mean %s tok/s  runs with a >20 ms request %d  total pending>=5ms %d  worst %.1f ms  "
                  "watchdog reports %d (stagnant heartbeat %d)  failures/traps %d" % (
                      base, len(ar), "%.2f" % means[base] if means[base] else "n/a", n_worst_runs, n_stall, worst, hb_run, hb_stag, traps))
        if means.get("L0"):
            for base, lim in (("L8", 0.10), ("L24", 0.35)):
                if means.get(base):
                    fall = 1.0 - means[base] / means["L0"]
                    print("  prediction (i) %s: tok/s change %+.1f%% (predicted fall < %.0f%%) -> %s" % (
                        base, -100 * fall, 100 * lim, "holds" if fall < lim else "REFUTED"))
    else:
        print("\nCONTENTION TEST load: not run")

    allok = all(r[1] for r in rows)
    print("CONTRACT G3: %s (mechanical evaluation; acceptance is the agent's own decision, not owner approval)" % (
        "ALL G3-1..G3-6 HOLD" if allok else "NOT ALL CRITERIA HOLD"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
