#!/usr/bin/env python3
"""Mechanical evaluation of acceptance contract G-TG8 (docs/plans/tg-beat-stock-8gb-262k.md).

Reads the session artifacts under build/llama-cuda and applies the thresholds FIXED in the contract
(they are constants here, copied from the plan, never read from results). Host-only. C9 (records) is
checked only for the presence of the result marker; whether the prose is right is a human/agent job.

  python tools/check-g1-contract.py [--root .]      exit 0 iff C1-C8 hold and the C9 marker exists
"""
import argparse
import importlib.util
import json
import os
import re
import sys

BAR = 30.68            # owner doc, tok/s
C1_MIN = 1.10 * BAR
C2_RATIO = 1.20
C3_MAX_MIB = 8192
C5_SLACK = 0.05
C6_LIMIT = 2.42
SESSIONS_CERT = ["g1-smoke", "g1-speedn", "g1-gapn", "g1-qa50", "g1-qa4096", "g1-qa24000", "g1-loop2048"]
SESSIONS_INFO = ["g1-speed", "g1-gap", "g1-bias100"]   # mismatched-prompt / diagnostic runs: reported only
RUNGS = [("g1-qa50", 50), ("g1-qa4096", 4096), ("g1-qa24000", 24000)]


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pp_tokens(trace):
    with open(trace, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if '"event":"run_start"' in ln:
                return json.loads(ln).get("pp_tokens")
    return None


def count_events(trace, name):
    n = 0
    with open(trace, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if '"event":"%s"' % name in ln:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    B = os.path.join(a.root, "build", "llama-cuda")
    rows = []

    def row(cid, ok, text):
        rows.append((cid, ok, text))

    sp = jload(os.path.join(B, "g1-speedn", "g1-speedn-verdict.json"))
    ag = sp["aggregate"]
    hyb, nh, stk = ag["HYB"], ag["NH"], ag["STK"]
    row("C1", hyb["mean"] >= C1_MIN and hyb["n_ok"] == 3,
        "HYB mean %.2f over %d reps (min %.2f) >= %.2f" % (hyb["mean"], hyb["n_ok"], hyb["min"], C1_MIN))
    row("C2", hyb["mean"] >= C2_RATIO * stk["mean"] and hyb["min"] > stk["max"],
        "HYB %.2f vs STK %.2f = %.3fx (need >= %.2fx); min(HYB) %.2f > max(STK) %.2f; HYB/NH %.3fx" % (
            hyb["mean"], stk["mean"], hyb["mean"] / stk["mean"], C2_RATIO, hyb["min"], stk["max"], hyb["mean"] / nh["mean"]))
    c3_ok = hyb["vram_peak_net_mib_max"] <= C3_MAX_MIB
    c3_txt = "HYB whole-GPU peak %s MiB <= %d (STK %s, NH %s, HYBU %s)" % (
        hyb["vram_peak_net_mib_max"], C3_MAX_MIB, stk["vram_peak_net_mib_max"], nh["vram_peak_net_mib_max"],
        ag["HYBU"]["vram_peak_net_mib_max"])
    fv = os.path.join(B, "g1-fastvram", "g1-fastvram-verdict.json")
    if os.path.exists(fv):   # extra evidence: the same peaks re-read from ONE nvidia-smi at 50 ms (~400-500 samples/run)
        fa = jload(fv)["arms"]
        fast = [r.get("vram_peak_mib") for n, r in fa.items() if n.startswith("HYB")]
        c3_ok = c3_ok and bool(fast) and max(fast) <= C3_MAX_MIB
        c3_txt += "; 50 ms poller HYB peaks %s over %s samples" % (
            fast, [r.get("vram_samples") for n, r in fa.items() if n.startswith("HYB")])
    row("C3", c3_ok, c3_txt)

    qa_ok = all(hyb["qa_pass"]) and all(hyb["qa_neg_ok"])
    qa_txt = ["512: %s" % ("pass" if qa_ok else "FAIL")]
    for sess, want in RUNGS:
        v = jload(os.path.join(B, sess, "%s-verdict.json" % sess))
        r = v["arms"]["HYB"]
        n = pp_tokens(r["trace"])
        ok = bool(r.get("rc") == 0 and r["qa"]["positive_pass"] and r["qa"]["negative_fails_as_designed"] and n == want)
        qa_ok = qa_ok and ok
        qa_txt.append("%d: %s (pp_tokens %s, VRAM %s MiB)" % (want, "pass" if ok else "FAIL", n, r.get("vram_peak_mib")))
    row("C4", qa_ok, "needle QA positive + negative control: " + "; ".join(qa_txt))

    lp = jload(os.path.join(B, "g1-loop2048", "g1-loop2048-verdict.json"))
    ra = lp["arms"]["HYB"]["summary"]["rep6"]["ratio"]
    rb = lp["arms"]["NH"]["summary"]["rep6"]["ratio"]
    rs = lp["arms"]["STK"]["summary"]["rep6"]["ratio"]
    row("C5", ra >= rb - C5_SLACK,
        "2048 tokens, distinct-6-word-gram ratio HYB %.4f >= NH %.4f - %.2f (STK %.4f)" % (ra, rb, C5_SLACK, rs))

    gp = jload(os.path.join(B, "g1-gap", "g1-gap-nonl.json"))
    base = next(c for c in gp["configs"] if c["label"] == "base")
    others = ["%s: delta %.3f" % (c["label"], c["delta"]) for c in gp["configs"] if c["label"] != "base"]
    row("C6", base.get("delta") is not None and base["delta"] <= C6_LIMIT,
        "tokens %s: stock gap %.3f %s vs LM gap %.3f -> Delta %.3f <= %.2f (%s)" % (
            gp["tokens"], base["gap_stock_A_over_B"], base["gap_stock_bounds"], base["gap_lm_A_over_B"],
            base["delta"], C6_LIMIT, "; ".join(others)))

    waivers = ""
    wp = os.path.join(B, "verdicts", "provenance-waivers.log")
    if os.path.exists(wp):
        waivers = open(wp, encoding="utf-8", errors="replace").read()
    bad = []
    for tag in SESSIONS_CERT:
        v = jload(os.path.join(B, "verdicts", tag + ".verdict.json"))
        prov = (v.get("fields") or {}).get("provenance") or {}
        ok = (v.get("status") == "PASS" and prov.get("binary_state") == "hashed"
              and prov.get("sources_newer_than_binary") is False
              and not (v.get("fields") or {}).get("refused_unprovenanced")
              and not re.search(r"\b%s\b" % re.escape(tag), waivers))
        if not ok:
            bad.append(tag)
    info_waived = [t for t in SESSIONS_INFO if re.search(r"\b%s\b" % re.escape(t), waivers)]
    row("C7", not bad, "%d sessions attested, no waiver (%s); reported-only sessions waived: %s" % (
        len(SESSIONS_CERT) - len(bad), ", ".join(SESSIONS_CERT), info_waived or "none") + ("; BAD: %s" % bad if bad else ""))

    ct = open(os.path.join(B, "g1-ctest.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", ct)
    on = count_events(os.path.join(B, "g1-smoke", "g1-smoke-HYB.jsonl"), "pp_last_logits")
    off = count_events(os.path.join(B, "g1-smoke", "g1-smoke-DEF.jsonl"), "pp_last_logits")
    row("C8", bool(m) and m.group(1) == "100" and m.group(2) == "0" and on >= 1 and off == 0,
        "host engine ctest %s; pp_last_logits events: margin-on %d (>=1), default %d (==0)  [amendment A1: doc-size guard excluded]" % (
            ("%s/%s passed" % (int(m.group(3)) - int(m.group(2)), m.group(3))) if m else "NO RESULT", on, off))

    plan = open(os.path.join(a.root, "docs", "plans", "tg-beat-stock-8gb-262k.md"), encoding="utf-8").read()
    cur = open(os.path.join(a.root, "docs", "currentstate.md"), encoding="utf-8", errors="replace").read()
    row("C9", "ACCEPTANCE RESULT G-TG8" in plan and "G-TG8" in cur,
        "result marker in plan: %s; G-TG8 entry in currentstate.md: %s" % (
            "ACCEPTANCE RESULT G-TG8" in plan, "G-TG8" in cur))

    w = max(len(r[2]) for r in rows)
    for cid, ok, text in rows:
        print("%-3s %-4s %s" % (cid, "PASS" if ok else "FAIL", text))
    allok = all(r[1] for r in rows)
    print("\nCONTRACT G-TG8: %s (mechanical evaluation; acceptance is the agent's own decision, not owner approval)" % (
        "ALL C1-C9 HOLD" if allok else "NOT ALL CRITERIA HOLD"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
