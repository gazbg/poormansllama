#!/usr/bin/env python3
"""G41 tables of --lm-prefill-canonical (used by tools/server_lm_review_tables.py, sections `canonical` and `canoncost`). Host-only; reads build/llama-cuda/<tag>/ and build/llama-cuda/k<NNNN>.out."""
import json
import os
import re
import subprocess
import sys

B = "build/llama-cuda/"
HERE = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [
    ("stock decode", "swapped np1", "k4201-canon-lmstock-mt-np1-swap"), ("stock decode", "original np1", "k4202-canon-lmstock-mt-np1"),
    ("stock decode", "swapped np2", "k4607-canon-lmstock-mt-np2-swap"), ("stock decode", "original np2", "k4608-canon-lmstock-mt-np2"),
    ("frozen cache", "swapped np1", "k4203-canon-frozen-mt-np1-swap"), ("frozen cache", "original np1", "k4204-canon-frozen-mt-np1"),
    ("frozen cache", "swapped np2", "k4609-canon-frozen-mt-np2-swap"), ("frozen cache", "original np2", "k4610-canon-frozen-mt-np2"),
    ("dynamic cache", "swapped np1", "k4211-canon-dyn-mt-np1-swap"), ("dynamic cache", "original np1", "k4212-canon-dyn-mt-np1"),
    ("dynamic cache", "swapped np2", "k4613-canon-dyn-mt-np2-swap"), ("dynamic cache", "original np2", "k4614-canon-dyn-mt-np2"),
]

CHECK_RE = re.compile(r"^(branch|alternate|equiv) +(rollback-vs-fresh|restore-vs-first|same-turn\d+) +(ok|FAIL): (.*)$", re.M)
SHORT = {"rollback-vs-fresh": "rollback", "restore-vs-first": "restore"}


def read_out(tag):
    num = re.match(r"k(\d+)", tag).group(1)
    p = B + "k%s.out" % num
    if not os.path.exists(p):
        return None
    return open(p, "rb").read().decode("utf-8", "replace").replace(chr(0), "")


def report(tag):
    p = B + tag + "/report.json"
    return json.load(open(p)) if os.path.exists(p) else None


def sec_canonical():
    print("### --lm-prefill-canonical (G41: np 1 sessions on build s45, np 2 sessions on build s46 = s45 + G41b): the five hit-versus-fresh checks of the suite, session by session ('identical' = every generated token equal; dN mX = first difference at token N, margin X nats)")
    n_ok = n_all = n_ident = 0
    for dec, sess, tag in SESSIONS:
        txt = read_out(tag)
        if txt is None:
            print("- %-13s %-13s %s: not run" % (dec, sess, tag))
            continue
        cells = []
        for m in CHECK_RE.finditer(txt):
            name = SHORT.get(m.group(2), m.group(2))
            msg = m.group(4)
            n_all += 1
            n_ok += 1 if m.group(3) == "ok" else 0
            if "identical over" in msg:
                n_ident += 1
                cells.append("%s identical" % name)
            else:
                d = re.search(r"first difference at token (\d+), margin ([\d.]+)", msg)
                cells.append("%s %s%s" % (name, ("d%s m%s" % d.groups()) if d else msg[:30], "" if m.group(3) == "ok" else " FAIL"))
        ft = re.findall(r"1st token \(from the prefill\): max\|dlogprob\| ([\d.]+)", txt)
        print("- %-13s %-13s %-34s %s | first token max|dlogprob| %s" % (dec, sess, tag, "; ".join(cells), max(map(float, ft)) if ft else "-"))
    print("  %d of %d checks pass, %d of them bit-identical" % (n_ok, n_all, n_ident))
    print("### replicated pairs (17 prompts, 3 fresh + 2 hit runs): default mode against canonical mode")
    runs = [("DEF-LMSTOCK", "k2706-hf-lmstock"), ("CANON-LMSTOCK", "k4205-hf-canon-lmstock"), ("DEF-FROZEN", "k2705-hf-frozen-flag-lm"), ("CANON-FROZEN", "k4206-hf-canon-frozen"),
            ("DEF-DYN", "k2709-hf-lm-dyn"), ("CANON-DYN", "k4215-hf-canon-dyn")]
    args = [sys.executable, os.path.join(HERE, "server_hitfresh.py"), "analyze"] + ["%s=%s%s/hitfresh.json" % (lab, B, t) for lab, t in runs if os.path.exists(B + t + "/hitfresh.json")]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    keep = [ln for ln in out.split(chr(10)) if ln.startswith("==") or ln.startswith("   FF") or ln.startswith("   HF") or ln.startswith("   HH") or "hit requests" in ln]
    print(chr(10).join(keep))


def sec_canoncost():
    print("### what the canonical mode costs: agentic transcript (fixture A, np 1, dynamic cache) and the long growth transcript (fixture G); default, canonical, canonical with --lm-prefill-min 513")
    sets = (("agentic", (("default", "k4301-cost-def-agentic"), ("canonical", "k4302-cost-canon-agentic"), ("canonical min 513", "k4303-cost-canon513-agentic"))),
            ("growth", (("default", "k4304-cost-def-growth"), ("canonical", "k4305-cost-canon-growth"), ("canonical min 513", "k4306-cost-canon513-growth"))))
    for scen, tags in sets:
        res = {}
        for lab, tag in tags:
            r = report(tag)
            if r:
                res[lab] = [x for x in r["rows"] if x["scenario"] == scen and x.get("prompt_n") is not None]
        if not res:
            continue
        print("- %s: " % scen + " | ".join("%s: %d turns, wall %.1f s, prompt tokens %d, cached %d" % (lab, len(rows), sum(x["wall_s"] for x in rows), sum(x["prompt_n"] for x in rows), sum(x.get("cache_n") or 0 for x in rows)) for lab, rows in res.items()))
        if "default" in res and "canonical" in res:
            for a, c in zip(res["default"], res["canonical"]):
                print("    %-8s default prompt_n %6s cache_n %6s wall %5.2f | canonical prompt_n %6s cache_n %6s wall %5.2f" % (a["name"], a["prompt_n"], a.get("cache_n"), a["wall_s"], c["prompt_n"], c.get("cache_n"), c["wall_s"]))


if __name__ == "__main__":
    sec_canonical()
    sec_canoncost()
