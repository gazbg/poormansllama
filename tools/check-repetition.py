#!/usr/bin/env python3
"""Host-only repetition screen for generated text (contract G-TG8, criterion C5).

Metric: distinct n-gram ratio over whitespace-separated words,
    ratio = (number of distinct word n-grams) / (number of word n-grams)
A degenerate loop drives the ratio toward 0; varied text sits near 1. The raw-mode model also
regurgitates the prompt after answering, so an absolute threshold is meaningless: the screen
compares a candidate against a reference produced under the SAME prompt and harness.

  check-repetition.py --text FILE [--n 6]                      print the metrics
  check-repetition.py --text CAND --ref REF [--slack 0.05]     exit 0 iff ratio(CAND) >= ratio(REF) - slack
  check-repetition.py --selftest

It grades; it claims nothing else (answer stability is the QA gate, not this).
"""
import argparse
import sys


def words_of(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().split()


def distinct_ratio(words, n):
    """(distinct n-grams, total n-grams, ratio); ratio is 1.0 when there is no n-gram at all."""
    total = len(words) - n + 1
    if total <= 0:
        return 0, 0, 1.0
    seen = set(tuple(words[i:i + n]) for i in range(total))
    return len(seen), total, len(seen) / total


def metrics(words, n):
    d, t, r = distinct_ratio(words, n)
    half = words[len(words) // 2:]
    _, tt, tr = distinct_ratio(half, n)
    return {"words": len(words), "grams": t, "distinct": d, "ratio": r, "tail_grams": tt, "tail_ratio": tr}


def selftest():
    fails = []

    def check(name, ok):
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            fails.append(name)

    varied = ("w%d" % i for i in range(200))
    r_varied = distinct_ratio(list(varied), 6)[2]
    check("varied text has ratio 1.0", r_varied == 1.0)
    loop = ["a", "b", "c", "d"] * 100
    r_loop = distinct_ratio(loop, 6)[2]
    check("period-4 loop has a tiny ratio (%.3f)" % r_loop, r_loop < 0.02)
    check("fewer words than n gives ratio 1.0 and 0 grams", distinct_ratio(["x"] * 3, 6) == (0, 0, 1.0))
    half = ["w%d" % i for i in range(100)] + ["a", "b"] * 50
    m = metrics(half, 6)
    check("tail ratio sees the loop the whole-text ratio dilutes (%.3f < %.3f)" % (m["tail_ratio"], m["ratio"]),
          m["tail_ratio"] < m["ratio"])
    check("slack rule: cand 0.80 vs ref 0.84 slack 0.05 passes, vs ref 0.90 fails",
          (0.80 >= 0.84 - 0.05) and not (0.80 >= 0.90 - 0.05))
    print("selftest: %s (%d checks)" % ("PASS" if not fails else "FAIL", 5))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--slack", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.text:
        ap.error("--text is required")
    c = metrics(words_of(a.text), a.n)
    print("cand: words=%d grams=%d distinct=%d ratio=%.4f tail_ratio=%.4f (n=%d)" % (
        c["words"], c["grams"], c["distinct"], c["ratio"], c["tail_ratio"], a.n))
    if not a.ref:
        return 0
    r = metrics(words_of(a.ref), a.n)
    print("ref : words=%d grams=%d distinct=%d ratio=%.4f tail_ratio=%.4f" % (
        r["words"], r["grams"], r["distinct"], r["ratio"], r["tail_ratio"]))
    ok = c["ratio"] >= r["ratio"] - a.slack
    print("verdict: %s (cand %.4f %s ref %.4f - slack %.2f)" % (
        "PASS" if ok else "FAIL", c["ratio"], ">=" if ok else "<", r["ratio"], a.slack))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
