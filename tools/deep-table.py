#!/usr/bin/env python3
"""One markdown row for a headline session of two arms (a stock control and a hybrid test arm): the numbers the G19/G21 results tables quote.

    python tools/deep-table.py <session> <control arm> <test arm>      (reads build/llama-cuda/<session>/*.log, built from tools/deep-summary.py's parser)

Columns: prompt tokens | control TG mean (min-max) | test TG mean (min-max) | ratio of means | positive pairs | slots | decode-phase min free | whole-run min free | PP control | PP test
Pairs are matched by repetition index (the sessions run counterbalanced pairs). Host-only.
"""
import glob
import importlib.util
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("deep_summary", os.path.join(HERE, "deep-summary.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def runs(session, arm):
    out = {}
    for p in sorted(glob.glob(os.path.join(ds.ROOT, session, "%s-%s_r*.log" % (session, arm)))):
        m = re.search(r"_r(\d+)\.log$", p)
        out[int(m.group(1))] = ds.summarize(p)
    return out


def tg(d):
    if "tg" in d:
        return float(d["tg"][2])
    if "stock_tg" in d:
        return float(d["stock_tg"][2])
    return None


def pp(d):
    if "pp_tok" in d:
        return int(d["pp_tok"][0]), float(d["pp_tok"][1]) / 1000.0, float(d["pp_tok"][2])
    if "stock_pp" in d:
        return int(d["stock_pp"][1]), float(d["stock_pp"][0]) / 1000.0, float(d["stock_pp"][2])
    return None


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    session, ctl, tst = argv
    c, t = runs(session, ctl), runs(session, tst)
    reps = sorted(set(c) & set(t))
    cv = [tg(c[r]) for r in reps]
    tv = [tg(t[r]) for r in reps]
    pos = sum(1 for a, b in zip(cv, tv) if a is not None and b is not None and b > a)
    ratio = statistics.mean(tv) / statistics.mean(cv)
    slots = sorted({t[r]["slots"][0] for r in reps if "slots" in t[r]})
    dfree = sorted({float(t[r]["freetg"][0]) for r in reps if "freetg" in t[r]})
    wfree = sorted({float(t[r]["free"][0]) for r in reps if "free" in t[r]})
    pc, pt = pp(c[reps[0]]), pp(t[reps[0]])
    row = "| %s | %.2f (%.2f-%.2f) | %.2f (%.2f-%.2f) | **%.2fx** | %d of %d | %s | %s | %s | %s | %s |" % (
        pt[0] if pt else "?", statistics.mean(cv), min(cv), max(cv), statistics.mean(tv), min(tv), max(tv), ratio, pos, len(reps),
        "/".join(slots), "/".join("%.0f" % x for x in dfree), "/".join("%.0f" % x for x in wfree),
        ("%.0f tok/s (%.1f s)" % (pc[2], pc[1])) if pc else "-", ("%.0f tok/s (%.1f s)" % (pt[2], pt[1])) if pt else "-")
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
