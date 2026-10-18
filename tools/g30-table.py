#!/usr/bin/env python3
"""G30 markdown rows: one row per (session, MTP arm) against the session's B0 arm (host-only).

    python tools/g30-table.py g99-sc24 g99-sp24 g98-lc4k ...

Columns: cell, MTP arm, R = mean of the counterbalanced pair ratios TG(arm)/TG(B0) with the pairs, the contract verdict (WIN: R >= 1.03 and every pair > 1.00; NEUTRAL: 0.98 <= R < 1.03, or R >= 1.03
with a pair <= 1.00 marked '(R ok, a pair <= 1.00)'; LOSS: R < 0.98), acceptance, cache slots B0 / MTP, PP B0 -> MTP (ms) and the ratio B0 / MTP (G30-2: >= 0.95), the catch-up time, and the strict bars
(poller peak MiB / whole-run minimum free MiB, both readings: whole run | decode phase). Reads build/llama-cuda/<session>/ (logs, hybrid traces, verdict).
"""
import glob
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("g30anatomy", os.path.join(HERE, "g30-anatomy.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
ROOT = an.ROOT


def bars(session):
    vp = os.path.join(ROOT, session, session + "-verdict.json")
    out = {}
    if not os.path.exists(vp):
        return out
    v = json.load(open(vp))
    for k, a in v["arms"].items():
        name = k.split("#")[0]
        d = out.setdefault(name, {"peak": [], "whole": [], "dec": []})
        d["peak"].append(a.get("vram_peak_raw_mib"))
        rep = k.split("#")[1] if "#" in k else "1"
        for lp in glob.glob(os.path.join(ROOT, session, "%s-%s_r%s.log" % (session, name, rep))):
            t = an.read(lp)
            w = re.findall(r"free-watch: minimum free device memory ([\d.]+) MiB", t)
            c = re.findall(r"free-watch \(decode phase[^)]*\): minimum free device memory ([\d.]+) MiB", t)
            if w:
                d["whole"].append(float(w[-1]))
            if c:
                d["dec"].append(float(c[-1]))
    return out


def main():
    for s in sys.argv[1:]:
        runs = an.session_runs(s)
        if not runs:
            print("| %s | no runs |" % s)
            continue
        arms = sorted({a for a, _, _ in runs})
        plain = "B0"
        b = bars(s)
        per_rep = {}
        for a, rep, d in runs:
            per_rep.setdefault(rep, {})[a] = d
        for mtp in [a for a in arms if a != plain and a not in ("B0S", "B0old", "B0new")]:
            ratios = [per_rep[r][mtp]["tg"] / per_rep[r][plain]["tg"] for r in sorted(per_rep)
                      if plain in per_rep[r] and mtp in per_rep[r] and per_rep[r][mtp].get("tg") and per_rep[r][plain].get("tg")]
            if not ratios:
                continue
            R = statistics.mean(ratios)
            allpos = all(x > 1.0 for x in ratios)
            if R >= 1.03 and allpos:
                verdict = "WIN"
            elif R >= 1.03:
                verdict = "R ok, a pair <= 1.00"
            elif R >= 0.98:
                verdict = "NEUTRAL"
            else:
                verdict = "LOSS"
            acc = an.mean_of(runs, mtp, "acc")
            sl0, sl1 = an.mean_of(runs, plain, "slots"), an.mean_of(runs, mtp, "slots")
            p0, p1 = an.mean_of(runs, plain, "pp_s"), an.mean_of(runs, mtp, "pp_s")
            cu = an.mean_of(runs, mtp, "catchup_ms")

            def bar(name):
                d = b.get(name)
                if not d or not d["peak"]:
                    return "-"
                pk = max(x for x in d["peak"] if x is not None)
                wh = min(d["whole"]) if d["whole"] else None
                de = min(d["dec"]) if d["dec"] else None
                ok = pk <= 7370 and wh is not None and wh >= 16000
                return "%d / %s | %s %s" % (pk, "-" if wh is None else "%.0f" % wh, "-" if de is None else "%.0f" % de, "held" if ok else "NOT held")
            print("| %s | %s | %.3f (%s) | %s | %s | %d / %d | %.0f -> %.0f ms (%.3f) | %s | B0 %s; %s %s |" % (
                s, mtp, R, " ".join("%.3f" % x for x in ratios), verdict, "-" if acc is None else "%.3f" % acc, round(sl0 or 0), round(sl1 or 0),
                p0 or 0, p1 or 0, (p0 / p1) if p0 and p1 else 0, "-" if cu is None else "%.0f ms" % cu, bar(plain), mtp, bar(mtp)))


if __name__ == "__main__":
    main()
