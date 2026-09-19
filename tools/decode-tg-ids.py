#!/usr/bin/env python3
"""decode-tg-ids.py -- detokenize a run's TG ids and grade them as an answer.

Owned home of the vocab-decode probe used by T-GPLR-46/T-GPLR-47 evidence
(previously a conversation-only heredoc). Given a telemetry trace (or mixed
runner log) it:

  1. parses the single `tg_tokens` event (loud failure if missing/ambiguous),
  2. loads the model's `tokenizer.ggml.tokens` from a JSON cache, building the
     cache from GGUF metadata on first use (metadata-only read, no tensors),
  3. detokenizes the ids with the gpt2 byte-level-BPE rules,
  4. grades the decoded text with check-answer.py's needle grader.

Reuses check-answer.py's reader/detokenizer/grader verbatim -- this tool owns
only the trace->text plumbing and the vocab cache.

Usage:
  python tools/decode-tg-ids.py --trace TRACE [--model GGUF | --vocab-cache JSON]
      [--items N1,N2] [--negative] [--answer ID=TEXT]... [--json OUT] [--quiet]

Exit codes: 0 grade PASS, 1 grade FAIL, 2 loud error (bad trace, no ids, vocab
mismatch). Grading is the soft semantic gate per AGENTS.md -- a PASS proves
needle presence only, never fitness for use.
"""

import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VOCAB_CACHE = os.path.join(_HERE, "..", "build", "llama-cuda",
                                    "vocab-cache.json")


def _load_check_answer():
    """Import the sibling checker module (hyphenated filename, so importlib)."""
    path = os.path.join(_HERE, "semantic-qa", "check-answer.py")
    spec = importlib.util.spec_from_file_location("check_answer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load_check_answer()


def _fail(msg):
    """Loud error: message to stderr, exit code 2 (distinct from grade FAIL=1)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def load_vocab(model, vocab_cache):
    """Return the token list, via cache when possible, else GGUF metadata.

    Cache JSON is a plain list of token strings, one per id. Built once from
    read_gpt2_vocab (GGUF metadata only); every later run is host-only and
    needs no model. A cache/model length mismatch fails loudly.
    """
    cache = os.path.abspath(vocab_cache)
    tokens = None
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            _fail(f"{cache}: vocab cache must be a JSON list of strings")
    if model is not None:
        mt, tok_model = ca.read_gpt2_vocab(model)
        if tok_model not in (None, "gpt2"):
            _fail(f"{model}: unsupported tokenizer model {tok_model!r}")
        if tokens is not None and len(tokens) != len(mt):
            _fail(f"vocab cache has {len(tokens)} tokens but {model} "
                  f"has {len(mt)}; refusing to mix vocabularies")
        if tokens is None:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(mt, f, ensure_ascii=False)
            print(f"vocab cache built: {cache} ({len(mt)} tokens)")
        return tokens if tokens is not None else mt
    if tokens is None:
        _fail("no vocab: give --model (first run builds the cache) "
              f"or an existing --vocab-cache")
    return tokens


def _display(text):
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="backslashreplace").decode(enc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True,
                    help="JSONL telemetry trace or mixed runner log")
    ap.add_argument("--model", default=None,
                    help="GGUF path (only needed when the vocab cache is cold)")
    ap.add_argument("--vocab-cache", default=os.path.normpath(_DEFAULT_VOCAB_CACHE))
    ap.add_argument("--items", default=None,
                    help="comma-separated needle ids to grade (default: all)")
    ap.add_argument("--negative", action="store_true",
                    help="PASS iff no expected answer appears")
    ap.add_argument("--answer", action="append", default=[], metavar="ID=TEXT",
                    help="override expected answer for one item id")
    ap.add_argument("--json", dest="out_json", default=None,
                    help="write the full record (ids, text, verdict) as JSON")
    ap.add_argument("--quiet", action="store_true", help="verdict line only")
    a = ap.parse_args()

    items = [s.strip() for s in (a.items or "").split(",") if s.strip()] or None
    overrides = {}
    for kv in a.answer:
        k, _, v = kv.partition("=")
        if not _:
            _fail(f"--answer expects ID=TEXT, got {kv!r}")
        overrides[k] = v

    try:
        _, _, ids, _mixed = ca.parse_trace(a.trace)
    except (ValueError, OSError) as e:
        _fail(f"{a.trace}: {e}")

    tokens = load_vocab(a.model, a.vocab_cache)
    try:
        text = ca.detokenize_ids(tokens, ids)
    except ValueError as e:
        _fail(f"{a.trace}: {e}")

    verdict = ca.grade(text, negative=a.negative, items=items,
                       answer_overrides=overrides)

    record = {
        "trace": os.path.abspath(a.trace),
        "vocab_cache": os.path.abspath(a.vocab_cache),
        "n_ids": len(ids),
        "first_ids": ids[:16],
        "text": text,
        "verdict": verdict,
    }
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)

    if a.quiet:
        print(f"{verdict['aggregate']} n={len(ids)} {os.path.basename(a.trace)}")
    else:
        print(f"trace: {a.trace}")
        print(f"ids: n={len(ids)} first16={record['first_ids']}")
        print("--- decoded text ---")
        print(_display(text))
        print("--- end ---")
        ca.print_verdict(verdict)

    sys.exit(0 if verdict["aggregate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
