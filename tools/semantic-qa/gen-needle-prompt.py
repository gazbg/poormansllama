#!/usr/bin/env python3
"""T-GPLR-07 semantic-QA ladder: deterministic needle-prompt generator.

Host-only, stdlib-only (dependency policy: no new external deps).

Design contract (docs/plans/generic-prefill-layer-residency-plan.md, T-GPLR-07):
  - Primary anchors are EXTRACTION needles planted in the prompt at
    ~25/50/75/95% depth. A failure at length implicates the context path
    (KV, positions, recurrent state), not model quality.
  - One knowledge flavor item (capital of France) validates short-answer
    format compliance; it is enabled at short lengths and in the full set.
  - The prompt ends with an explicit answer contract naming every expected
    answer so one short TG burst can be checked programmatically.
  - Same target -> byte-identical prompt across runs (deterministic).

Length modes (calibrated against the real tokenizer):
  target <  15 : refused (the minimal meaningful scaffold is ~15 tokens).
  15 <= t < 30 : K1 only (knowledge question, no document).
  30 <= t < 54 : N1 only (one needle statement + question).
  54 <= t < 90 : N1 + K1.
  target >= 90 : FULL document mode: four needles at ~25/50/75/95% depth in
                 a filler document, the knowledge item, the answer contract.
  Gaps to the exact target are closed with single-token pad words
  ("Pad: x x ..."), so every emitted prompt hits its target exactly
  (verified with llama-tokenize --ids, vocab-only, host-only).

The printed `items=` list must be passed to check-answer.py via --items.
"""

import argparse
import os
import re
import subprocess
import sys

NEEDLES = [
    {"id": "N1", "answer": "QX-7741"},
    {"id": "N2", "answer": "TR-2918"},
    {"id": "N3", "answer": "MK-5501"},
    {"id": "N4", "answer": "LD-8834"},
]

DEPTH = {0: 0.25, 1: 0.50, 2: 0.75, 3: 0.95}

FILLER_SENTENCE = (
    "The maintenance log for row {r:05d} records a routine inspection "
    "of the coolant manifold, no anomalies were observed, and the next "
    "scheduled review is in fourteen days."
)

INTRO = (
    "You are answering questions about the document below. "
    "Answer with the requested values only. "
    "Do not explain. Do not be polite. Do not add any other words. "
)

CONTRACT = (
    "From the document above, list the four archive access codes in order, "
    "then state the capital of France. "
    "Answer with the values only, separated by spaces, nothing else."
)

N1_STATEMENT = "Note N1: The archive access code N1 is QX-7741."
N1_QUESTION = "State the archive access code. Values only."
K1_QUESTION = "What is the capital of France? Answer with the city name only."


def short_scaffold(items, pads=0):
    # Pads NEVER go after the question: the model attends to whatever sits
    # last and will echo pad tokens instead of answering (observed 2026-09-06,
    # 50-token rung). Pads go between the statement and the question.
    lines = []
    if "N1" in items:
        lines.append(N1_STATEMENT)
    if pads > 0:
        lines.append(pad_text(pads).rstrip("\n"))
    if "K1" in items:
        lines.append(K1_QUESTION)
    if "N1" in items:
        lines.append(N1_QUESTION)
    return "\n".join(lines) + "\n"


def pad_text(n):
    return "" if n <= 0 else "Pad: " + " ".join(["x"] * n) + "\n"


def short_items_for(target):
    if target < 30:
        return ["K1"]
    if target < 54:
        return ["N1"]
    return ["N1", "K1"]


def emit_full(rows, pads=0):
    """Document scaffold: needles at depth inside filler rows + contract.
    Pads go between the filler document and the contract, never after it."""
    needle_row = {i: max(0, min(rows - 1, int(round(DEPTH[i] * rows)) - 1))
                  for i in range(len(NEEDLES))}
    needle_lines = {}
    for i, n in enumerate(NEEDLES):
        needle_lines[needle_row[i]] = (
            f"Note {n['id']}: The archive access code {n['id']} "
            f"is {n['answer']}."
        )
    lines = [INTRO.rstrip()]
    for r in range(rows):
        lines.append(needle_lines.get(r, FILLER_SENTENCE.format(r=r)))
    if pads > 0:
        lines.append(pad_text(pads).rstrip("\n"))
    lines.append(CONTRACT)
    items = [n["id"] for n in NEEDLES] + ["K1"]
    return "\n".join(lines) + "\n", items, needle_row


_TC_CACHE = {}


def token_count(text, exe, model):
    """Exact token count via llama-tokenize --ids (vocab-only load, host-only).
    Memoized: calibration makes many near-duplicate queries."""
    key = text
    if key in _TC_CACHE:
        return _TC_CACHE[key]
    exe = os.path.abspath(exe)  # CreateProcess rejects forward-slash paths
    res = subprocess.run(
        [exe, "-m", model, "--stdin", "--ids", "--no-bos"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        sys.exit(f"error: llama-tokenize failed: {res.stderr.decode(errors='replace').strip()}")
    m = re.search(rb"\[([0-9,\s]*)\]", res.stdout)
    if not m:
        sys.exit(f"error: unexpected llama-tokenize output: {res.stdout[:200]!r}")
    body = m.group(1).strip()
    tc = 0 if not body else len(body.split(b","))
    _TC_CACHE[key] = tc
    return tc


def calibrate_short(target, exe, model):
    """Hit the target exactly (pad granularity ~1 token) or die loudly."""
    items = short_items_for(target)
    base = short_scaffold(items)
    tc0 = token_count(base, exe, model)
    if tc0 > target:
        sys.exit(f"error: target {target} is below the {items}-scaffold floor "
                 f"({tc0} tokens); use a target >= {tc0}")
    need = target - tc0
    best_p, best_tc, best_diff = 0, tc0, abs(tc0 - target)
    for p in range(0, need + 8):
        tc = token_count(short_scaffold(items, pads=p), exe, model)
        if abs(tc - target) < best_diff:
            best_p, best_tc, best_diff = p, tc, abs(tc - target)
        if tc == target:
            break
        if tc > target + 2:
            break
    if best_diff > 2:
        sys.exit(f"error: could not hit {target} exactly (closest {best_tc}); "
                 f"adjust the scaffold")
    return short_scaffold(items, pads=best_p), items, best_tc


def calibrate_full(target, exe, model):
    """Largest filler-row count that stays <= target, then pad to exact.
    Row count vs token count is monotonic, so bisect (O(log) spawns).
    Pads are inserted before the contract, never after it."""
    lo, hi = 1, max(2, target)
    best_rows = None
    while lo <= hi:
        mid = (lo + hi) // 2
        tc = token_count(emit_full(mid)[0], exe, model)
        if tc <= target:
            best_rows = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best_rows is None:
        sys.exit(f"error: even one filler row exceeds target {target}")
    text, items, needle_row = emit_full(best_rows)
    tc = token_count(text, exe, model)
    if tc < target:
        need = target - tc
        best_p, best_diff = 0, need
        for p in range(0, need + 8):
            cand, _, _ = emit_full(best_rows, pads=p)
            ctc = token_count(cand, exe, model)
            if abs(ctc - target) < best_diff:
                best_p, best_diff = p, abs(ctc - target)
            if ctc == target:
                break
            if ctc > target + 2:
                break
        text, items, needle_row = emit_full(best_rows, pads=best_p)
        tc = token_count(text, exe, model)
    if abs(tc - target) > 2:
        sys.exit(f"error: could not hit {target} exactly (closest {tc})")
    return text, items, needle_row, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate", default="", help="path to llama-tokenize.exe")
    ap.add_argument("--model", default="", help="GGUF path (vocab-only read)")
    args = ap.parse_args()

    if bool(args.calibrate) != bool(args.model):
        sys.exit("error: --calibrate and --model must be given together")

    if args.calibrate:
        if args.target_tokens < 15:
            sys.exit("error: targets below 15 tokens are refused: the minimal "
                     "meaningful scaffold (capital-of-France question) is "
                     "~15 tokens; no shorter run exists in the ladder")
        if args.target_tokens < 90:
            # best_tc is the REAL token count of the emitted scaffold (the
            # pad calibration tolerates a +-2 gap), so achieved= is honest;
            # callers pick --ubatch from it.
            text, items, tc = calibrate_short(args.target_tokens,
                                              args.calibrate, args.model)
            needle_row = None
        else:
            text, items, needle_row, tc = calibrate_full(
                args.target_tokens, args.calibrate, args.model)
    else:
        if args.target_tokens < 90:
            items = short_items_for(max(15, args.target_tokens))
            text, needle_row, tc = short_scaffold(items), None, None
        else:
            text, items, needle_row = emit_full(max(1, args.target_tokens // 25))
            tc = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print(f"target={args.target_tokens} achieved={tc if tc is not None else 'unverified'} "
          f"items={','.join(items)}")
    if needle_row is not None:
        print("needle_rows=" + ",".join(f"{NEEDLES[i]['id']}:{needle_row[i]}"
                                        for i in range(len(NEEDLES))))
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
