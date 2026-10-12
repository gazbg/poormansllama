#!/usr/bin/env python3
"""G22 stage C: the prefill time of every arm of the g50-<N> sessions (STK stock, LM2 all-GPU layer-major, HYC all-CPU layers, HY2 / HY4 model plans), per prompt length.

    python tools/g22-hybrid-table.py [N ...]       default N = 32 64 128 256 512 1024 2048

Per length: the median prefill time of each arm over its reps, the gain of each hybrid arm against the better of STK and LM2 of the SAME rep (the contract G22-3 counts the arm as a PASS when
every rep gains at least 10%), and the model's predicted time next to the measured one (read from the arm's `why` in build/llama-cuda/specs/g50-<N>.json). Host-only.
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "llama-cuda")
ARMS = ("STK", "LM2", "HYC", "HY2", "HY4")


def read(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def prefill_ms(text):
    m = None
    for m in re.finditer(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", text):
        pass
    if m:
        return float(m.group(1)), int(m.group(2))
    for m in re.finditer(r"PP: (\d+) tokens in ([\d.]+) ms", text):
        pass
    return (float(m.group(2)), int(m.group(1))) if m else None


def main(argv):
    ns = [int(x) for x in argv] or [32, 64, 128, 256, 512, 1024, 2048]
    print("%6s %6s | %s | %s" % ("N", "tokens", " ".join("%8s" % a for a in ARMS), "gain vs min(STK, LM2): HYC HY2 HY4   (predicted ms: HY2 HY4)"))
    for n in ns:
        session = "g50-%d" % n
        d = os.path.join(ROOT, session)
        per = {}
        tokens = 0
        for a in ARMS:
            per[a] = {}
            for p in sorted(glob.glob(os.path.join(d, "%s-%s_r*.log" % (session, a)))):
                r = int(re.search(r"_r(\d+)\.log$", p).group(1))
                v = prefill_ms(read(p))
                if v:
                    per[a][r] = v[0]
                    tokens = v[1]
        if not per["STK"] or not per["LM2"]:
            print("%6d  (no data yet)" % n)
            continue
        med = {a: (statistics.median(per[a].values()) if per[a] else float("nan")) for a in ARMS}
        gains = []
        for a in ("HYC", "HY2", "HY4"):
            g = []
            for r in sorted(per[a]):
                if r in per["STK"] and r in per["LM2"]:
                    g.append(min(per["STK"][r], per["LM2"][r]) / per[a][r])
            gains.append(("%.2fx" % statistics.mean(g)) + ("*" if g and min(g) >= 1.1 else " ") if g else "   -  ")
        pred = []
        try:
            spec = json.load(open(os.path.join(ROOT, "specs", session + ".json")))
            for a in spec["arms"]:
                if a["name"] in ("HY2", "HY4"):
                    m = re.search(r"predicted (\d+) ms", a.get("why", ""))
                    pred.append(m.group(1) if m else "?")
        except OSError:
            pred = ["?", "?"]
        print("%6d %6d | %s | %s   (%s)" % (n, tokens, " ".join("%8.0f" % med[a] for a in ARMS), " ".join("%7s" % g for g in gains), " ".join(pred)))
    print("(* = every rep gained at least 10% against the better of STK and LM2 of the same rep)")


if __name__ == "__main__":
    main(sys.argv[1:])
