#!/usr/bin/env python3
"""SRV-1 helper: compare the steps.jsonl of two tools/server-probe.py sessions (same scenario, the same flags): texts, token counts, PP and TG.

  python tools/server_probe_compare.py --a build/llama-cuda/<s0 run>/steps.jsonl --b build/llama-cuda/<new run>/steps.jsonl [--tol 0.03] [--logs A.log B.log]

SRV-1 (default-off identity): per step the greedy text is identical, prompt_n / cache_n / predicted_n are identical, PP and TG are within --tol (default 3 %) of each other.
With --logs, the load logs are compared line by line after removing the timestamps and the numbers (a changed line is printed; the server prints the same lines when the flags are the same).
Exit 1 when a text or count differs or a rate is outside the tolerance.
"""
import argparse
import json
import re
import sys


def load(p):
    return [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]


def norm_log(p):
    txt = open(p, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    out = []
    for l in txt.splitlines():
        l = re.sub(r"^\d+\.\d+\.\d+\.\d+ ", "", l)
        l = re.sub(r"\d+(\.\d+)?", "#", l)
        if l.startswith("ENV ") or "exe" in l[:0]:
            continue
        out.append(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--logs", nargs=2)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    bad = 0
    for x, y in zip(A, B):
        tx, ty = x["timings"], y["timings"]
        probs = []
        if x.get("text") != y.get("text"):
            probs.append("TEXT DIFFERS")
        for k in ("prompt_n", "cache_n", "predicted_n"):
            if tx.get(k) != ty.get(k):
                probs.append("%s %s vs %s" % (k, tx.get(k), ty.get(k)))
        rr = []
        for k, nm in (("prompt_per_second", "PP"), ("predicted_per_second", "TG")):
            if tx.get(k) and ty.get(k) and tx.get("prompt_n", 0) > 64 or (k == "predicted_per_second" and tx.get(k) and ty.get(k)):
                r = ty[k] / tx[k]
                rr.append("%s %.1f -> %.1f (%+.1f%%)" % (nm, tx[k], ty[k], 100 * (r - 1)))
                if abs(r - 1) > a.tol:
                    probs.append("%s off by %.1f%%" % (nm, 100 * (r - 1)))
        print("%-12s %s %s" % (x["name"], "; ".join(rr), ("  <-- " + "; ".join(probs)) if probs else ""))
        bad += 1 if probs else 0
    if len(A) != len(B):
        print("different number of steps: %d vs %d" % (len(A), len(B)))
        bad += 1
    if a.logs:
        la, lb = norm_log(a.logs[0]), norm_log(a.logs[1])
        sa, sb = set(la), set(lb)
        only_a = [l for l in la if l not in sb]
        only_b = [l for l in lb if l not in sa]
        print("load log: %d lines vs %d lines; only in A: %d, only in B: %d" % (len(la), len(lb), len(only_a), len(only_b)))
        for l in only_a[:12]:
            print("  A> " + l[:200])
        for l in only_b[:12]:
            print("  B> " + l[:200])
    print("SRV-1 %s" % ("FAILED" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
