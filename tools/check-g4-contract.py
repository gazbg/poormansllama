#!/usr/bin/env python3
"""Mechanical evaluation of contract G4 (input-upload batching, `LLAMA_CUDA_INPUT_BATCH=1`) in docs/plans/tg-beat-stock-8gb-262k.md.

Thresholds are constants copied from the contract, never read from results. Host-only, stdlib-only.

  python tools/check-g4-contract.py [--root .] [--tag g6]      exit 0 iff G4-1..G4-5 hold

Sessions: <tag>-code, <tag>-prose (4 reps, arms H28_g2 HYB HYB28 HYB42R HYB42RB STK), <tag>-anchor (2 reps), <tag>-rel (HYB42R and HYB42RB x12),
<tag>-nsys (profiles of HYB42R#1 and HYB42RB#1 taken from <tag>-code).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

R_MIN = 1.03       # G4-1: HYB42RB / HYB42R
IDLE_DROP_US = 400.0   # G4-3
N_RUNS = 12        # G4-4
INERT = 0.02       # G4-5


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def load_gap(root):
    spec = importlib.util.spec_from_file_location("nsys_gap", os.path.join(root, "tools", "nsys-gap.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tag", default="g6")
    a = ap.parse_args()
    B = os.path.join(a.root, "build", "llama-cuda")
    T = a.tag
    rows = []

    def row(cid, ok, text):
        rows.append((cid, ok, text))

    ver = {s: jload(os.path.join(B, "%s-%s" % (T, s), "%s-%s-verdict.json" % (T, s))) for s in ("code", "prose", "anchor")}
    ag = {s: ver[s]["aggregate"] for s in ver}

    # G4-1
    parts, ok1 = [], True
    for s in ("code", "prose"):
        g = ag[s]
        r = g["HYB42RB"]["mean"] / g["HYB42R"]["mean"]
        sep = g["HYB42RB"]["min"] > g["HYB42R"]["max"]
        ok1 = ok1 and r >= R_MIN and sep and g["HYB42RB"]["n_ok"] == g["HYB42RB"]["n"]
        parts.append("%s: with %.2f / without %.2f = %.3fx, min(with) %.2f %s max(without) %.2f" % (
            s, g["HYB42RB"]["mean"], g["HYB42R"]["mean"], r, g["HYB42RB"]["min"], ">" if sep else "<=", g["HYB42R"]["max"]))
    row("G4-1", ok1, "; ".join(parts) + " (need >= %.2fx and separated)" % R_MIN)

    # G4-2: byte-identical generated text, per rep
    same, total, diff = 0, 0, []
    for s in ("code", "prose", "anchor"):
        arms = ver[s]["arms"]
        for name, r in arms.items():
            if r["base"] != "HYB42RB":
                continue
            other = arms.get("HYB42R#%d" % r["rep"])
            total += 1
            if other and os.path.exists(r["tgtext"]) and os.path.exists(other["tgtext"]) and sha(r["tgtext"]) == sha(other["tgtext"]):
                same += 1
            else:
                diff.append("%s/%s" % (s, name))
    row("G4-2", total > 0 and same == total, "generated text identical with/without the env in %d/%d paired reps %s" % (same, total, diff or ""))

    # G4-3: nsys gap idle
    gap = load_gap(a.root)
    idle = {}
    for arm in ("HYB42R_1", "HYB42RB_1"):
        dbp = os.path.join(B, "%s-nsys" % T, "prof-%s.sqlite" % arm)
        if not os.path.exists(dbp):
            idle[arm] = None
            continue
        import sqlite3
        res, err = gap.analyse(sqlite3.connect(dbp), 40, 4)
        idle[arm] = res["idle_us"] if res else None
    ok3 = all(v is not None for v in idle.values()) and (idle["HYB42R_1"] - idle["HYB42RB_1"]) >= IDLE_DROP_US
    row("G4-3", ok3, "gap idle per token: without %s us, with %s us (need a drop >= %.0f us)" % (
        "%.0f" % idle["HYB42R_1"] if idle["HYB42R_1"] is not None else "n/a", "%.0f" % idle["HYB42RB_1"] if idle["HYB42RB_1"] is not None else "n/a", IDLE_DROP_US))

    # G4-4: reliability of the batched arms across every session
    tot = fail = 0
    fails = []
    for s in ("code", "prose", "anchor", "rel"):
        vp = os.path.join(B, "%s-%s" % (T, s), "%s-%s-verdict.json" % (T, s))
        if not os.path.exists(vp):
            continue
        for n, r in jload(vp)["arms"].items():
            if r.get("base") == "HYB42RB":
                tot += 1
                if r.get("rc") != 0:
                    fail += 1
                    fails.append("%s-%s/%s rc=%s" % (T, s, n, r.get("rc")))
    row("G4-4", tot >= N_RUNS and fail == 0, "runs with the env %d (need >= %d), non-zero exits %d %s" % (tot, N_RUNS, fail, fails or ""))

    # G4-5: default inertness in the same session (code and prose have an H28_g2 control arm)
    dev, ok5 = {}, True
    for s in ("code", "prose"):
        g = ag[s]
        d = g["HYB28"]["mean"] / g["H28_g2"]["mean"] - 1.0
        dev[s] = "%+.1f%%" % (100 * d)
        ok5 = ok5 and abs(d) <= INERT
    row("G4-5", ok5, "HYB28 on %s vs the g2 build in the same session: %s (need within +-%.0f%%)" % (T, dev, 100 * INERT))

    for cid, ok, text in rows:
        print("%-5s %-4s %s" % (cid, "PASS" if ok else "FAIL", text))
    allok = all(r[1] for r in rows)
    print("CONTRACT G4: %s (mechanical evaluation; acceptance is the agent's own decision, not owner approval)" % (
        "ALL G4-1..G4-5 HOLD" if allok else "NOT ALL CRITERIA HOLD"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
