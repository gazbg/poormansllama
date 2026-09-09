#!/usr/bin/env python3
"""T-GPLR-07 semantic-QA ladder: deterministic programmatic answer checker.

Host-only, stdlib-only (dependency policy: no new external deps).

Grading contract (docs/plans/generic-prefill-layer-residency-plan.md, T-GPLR-07):
  - Answers are graded by normalized substring checks on the DETOKENIZED
    generated text, never by bit/byte/hash identity. Greedy token IDs may
    drift at length (CUDA reduction order, Q8 KV); answer stability is the
    observable, not token-ID stability.
  - The generated text comes either from the runner's `--tg-text` file
    (already detokenized by the runner with the model's own vocab via
    `llama_token_to_piece`) or from the trace's `tg_tokens` event (token IDs)
    detokenized here with the model's own vocab (GGUF metadata only: the
    gpt2-style byte-level-BPE `tokenizer.ggml.tokens` array; no tensor data
    is read).
  - Normalization strips <think> blocks, punctuation, and whitespace, then
    checks each needle's expected answer. Case-insensitive. QX-7741 and
    qx7741 are the same answer.
  - Negative control (--negative): aggregate PASS iff NONE of the expected
    answers appear. A grader that cannot fail is not a grader.

Subcommands:
  self-test     run the internal unit checks (normalizer, negative logic,
                detokenizer round-trip on a synthetic vocab) and exit
  verify-vocab  host-only end-to-end vocab check: tokenize a text file with
                llama-tokenize --ids, detokenize here, compare normalized
                forms. Must be run once against the real model before any
                GPU run.
  check         grade one run: --tg-text PATH (runner-generated text file,
                no model needed) or --trace PATH (parse + detokenize, needs
                --model); [--negative] [--out-verdict F]
"""

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# The runner keeps generating after EOS when its token budget exceeds the
# stop point; everything from the first special-token boundary on is loop
# restart, not answer (observed 2026-09-07 on the 30-token rung).
_EOS_RE = re.compile(r"<\|endoftext\|>.*$", re.DOTALL)


def normalize(text):
    """Lowercase, strip <think> blocks and every non-alphanumeric char."""
    text = _EOS_RE.sub(" ", text)
    text = _THINK_RE.sub(" ", text)
    # unterminated think block: everything from <think> on is chatter
    text = re.sub(r"<think>.*$", " ", text, flags=re.DOTALL)
    return _NON_ALNUM_RE.sub("", text.lower())


# ---------------------------------------------------------------------------
# GGUF vocab reader (metadata only) + gpt2 byte-level-BPE detokenizer
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
_T_U8, _T_I8, _T_U16, _T_I16, _T_U32, _T_I32, _T_F32 = 0, 1, 2, 3, 4, 5, 6
_T_BOOL, _T_STR, _T_ARRAY, _T_U64, _T_I64, _T_F64 = 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {_T_U8: "<B", _T_I8: "<b", _T_U16: "<H", _T_I16: "<h",
               _T_U32: "<I", _T_I32: "<i", _T_F32: "<f", _T_BOOL: "<B",
               _T_U64: "<Q", _T_I64: "<q", _T_F64: "<d"}
_SCALAR_SIZE = {_T_U8: 1, _T_I8: 1, _T_U16: 2, _T_I16: 2, _T_U32: 4,
                _T_I32: 4, _T_F32: 4, _T_BOOL: 1, _T_U64: 8, _T_I64: 8,
                _T_F64: 8}


def _bytes_to_unicode():
    """GPT-2 byte-to-unicode table (printable bytes map to themselves)."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\u00a1"), ord("\u00ac") + 1)) +
          list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_CHAR_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}


def _piece_to_bytes(piece):
    out = bytearray()
    for ch in piece:
        if ch in _CHAR_TO_BYTE:
            out.append(_CHAR_TO_BYTE[ch])
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def _read_scalar(f, t):
    import struct
    fmt = _SCALAR_FMT.get(t)
    if fmt is None:
        raise ValueError(f"unsupported scalar type {t}")
    size = _SCALAR_SIZE[t]
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF")
    return struct.unpack(fmt, data)[0]


def _read_str(f):
    n = _read_scalar(f, _T_U64)
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF string")
    return data.decode("utf-8", errors="replace")


def _skip_value(f, t):
    if t == _T_STR:
        _read_str(f)
        return
    if t == _T_ARRAY:
        et = _read_scalar(f, _T_U32)
        n = _read_scalar(f, _T_U64)
        for _ in range(n):
            _skip_value(f, et)
        return
    f.seek(_SCALAR_SIZE[t], 1)


def read_gpt2_vocab(path):
    """Read tokenizer.ggml.tokens (+ model key) from GGUF metadata only.

    Returns (tokens_list, tokenizer_model). Fails loudly on anything it
    cannot parse; never reads tensor data (stops after the KV section).
    """
    with open(path, "rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version = _read_scalar(f, _T_U32)
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} not supported")
        _tensor_count = _read_scalar(f, _T_U64)
        kv_count = _read_scalar(f, _T_U64)
        tokens = None
        tok_model = None
        for _ in range(kv_count):
            key = _read_str(f)
            t = _read_scalar(f, _T_U32)
            if key == "tokenizer.ggml.tokens" and t == _T_ARRAY:
                et = _read_scalar(f, _T_U32)
                n = _read_scalar(f, _T_U64)
                if et != _T_STR:
                    raise ValueError(f"{path}: tokenizer.ggml.tokens is not a string array")
                tokens = [_read_str(f) for _ in range(n)]
            elif key == "tokenizer.ggml.model" and t == _T_STR:
                tok_model = _read_str(f)
            else:
                _skip_value(f, t)
            if tokens is not None and tok_model is not None:
                break
    if tokens is None:
        raise ValueError(f"{path}: tokenizer.ggml.tokens not found in metadata")
    return tokens, tok_model


def detokenize_ids(tokens_list, ids):
    """Concatenate vocab pieces for the ids and restore real bytes (gpt2 BPE)."""
    chunks = []
    for i in ids:
        if i < 0 or i >= len(tokens_list):
            raise ValueError(f"token id {i} out of vocab range [0, {len(tokens_list)})")
        chunks.append(_piece_to_bytes(tokens_list[i]))
    return b"".join(chunks).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

def read_tg_text(path):
    """Read the runner's --tg-text generated-text file.

    The runner writes the detokenized generated text (model vocab, rendered
    with llama_token_to_piece, special=false) only when TG produced tokens,
    so a missing or empty file is a broken pipeline, not a legitimate empty
    answer -- fail loudly (an empty file would vacuously PASS the negative
    control and silently mask a runner failure).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        raise ValueError(f"{path}: cannot read tg-text file: {e}")
    if not text.strip():
        raise ValueError(f"{path}: tg-text file is empty or whitespace-only "
                         f"(runner writes it only when TG produced tokens)")
    return text


def parse_trace(path):
    """Extract (run_start_ok, exit_status, ids) with loud failures.

    Accepts either a pure JSONL telemetry file or a mixed runner log that
    interleaves llama.cpp stderr with JSONL telemetry events (as produced by
    redirecting the runner's combined stdout/stderr to one file). Non-JSON
    lines are skipped; a malformed JSON line that STARTS like an event is a
    loud error, not a silent skip.
    """
    run_start = None
    run_end = None
    tg_events = []
    mixed_text = None
    in_text_section = False
    text_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            raw = line.rstrip("\r\n")
            if raw == "--- generated text ---":
                in_text_section = True
                text_lines = []
                continue
            if in_text_section:
                if raw == "--- end ---":
                    mixed_text = "\n".join(text_lines)
                    in_text_section = False
                else:
                    text_lines.append(raw)
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                if line.lstrip().startswith("{"):
                    raise ValueError(f"{path}: line {ln}: bad JSON: {e}")
                if "generated ids:" in line:
                    # Mixed-log fallback: the runner prints the first 16
                    # generated IDs in human-readable form ("first 16
                    # generated ids: 271 248068 ..."). Wrap as a synthetic
                    # tg_tokens event so grading can proceed without a
                    # dedicated telemetry file (adequate for <=16-token
                    # answers; dedicated JSONL remains the strict gate).
                    body = line.split(":", 1)[1]
                    ids = [int(t) for t in body.split()]
                    tg_events.append({"event": "tg_tokens", "mixed_log": True,
                                      "n_tokens_generated": len(ids), "ids": ids})
                continue  # llama.cpp log noise interleaved with telemetry
            name = ev.get("event")
            if name == "run_start" and run_start is None:
                run_start = ev
            elif name == "run_end" and run_end is None:
                run_end = ev
            elif name == "tg_tokens":
                tg_events.append(ev)
            elif isinstance(name, str) and "generated" in name:
                # (also accepted if the runner ever emits it as JSON)
                tg_events.append(ev)
    if run_start is None and (run_end is not None or tg_events):
        # Mixed-log mode: the runner prints human-readable phase summaries
        # instead of JSONL run_start/run_end events on combined stdout/stderr.
        # If the generated-IDs event is present, grading may proceed; the
        # scheduler-invariant gate (--check-telemetry on a dedicated JSONL
        # telemetry file) remains a separate required check.
        pass
    elif run_start is None:
        raise ValueError(f"{path}: no run_start event")
    if run_end is not None and run_end.get("exit_status") != 0:
        raise ValueError(f"{path}: run_end.exit_status = "
                         f"{run_end.get('exit_status')} (run failed; no grading)")
    if len(tg_events) == 0:
        raise ValueError(f"{path}: no tg_tokens event (TG not run?)")
    if len(tg_events) > 1:
        raise ValueError(f"{path}: {len(tg_events)} tg_tokens events; ambiguous")
    ids = tg_events[0].get("ids")
    if not isinstance(ids, list):
        raise ValueError(f"{path}: tg_tokens.ids missing")
    return run_start, run_end, [int(i) for i in ids], mixed_text


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

NEEDLES = [
    {"id": "N1", "answer": "QX-7741"},
    {"id": "N2", "answer": "TR-2918"},
    {"id": "N3", "answer": "MK-5501"},
    {"id": "N4", "answer": "LD-8834"},
    {"id": "K1", "answer": "Paris"},  # knowledge flavor item
]


def grade(answer_text, negative=False, items=None, answer_overrides=None):
    """Return the verdict dict. Aggregate PASS = every enabled needle found
    (or, in negative mode, NONE found). items: subset of ids to grade (the
    generator prints the per-length list); default = all. answer_overrides:
    id -> expected-text overrides for the negative control (the negative
    prompt must ask something different, so the expected answer changes)."""
    norm = normalize(answer_text)
    wanted = [n for n in NEEDLES if items is None or n["id"] in items]
    if not wanted:
        raise ValueError("no grading items selected (empty --items?)")
    overrides = answer_overrides or {}
    graded = []
    for n in wanted:
        exp = normalize(overrides.get(n["id"], n["answer"]))
        hit = exp in norm
        graded.append({
            "id": n["id"], "expected": overrides.get(n["id"], n["answer"]),
            "found": hit, "pass": hit if not negative else not hit,
        })
    overridden = set(overrides)
    for k, v in overrides.items():
        if k in overridden and any(n["id"] == k for n in NEEDLES):
            continue  # already graded above with the override applied
        exp = normalize(v)
        hit = exp in norm
        graded.append({
            "id": k, "expected": v,
            "found": hit, "pass": hit if not negative else not hit,
        })
    aggregate = all(i["pass"] for i in graded)
    return {
        "mode": "negative-control" if negative else "needle",
        "aggregate": "PASS" if aggregate else "FAIL",
        "items": graded,
        "answer_text": answer_text,
        "answer_normalized": norm[:400],
    }


def print_verdict(v):
    print(f"mode={v['mode']} aggregate={v['aggregate']}")
    for i in v["items"]:
        mark = "PASS" if i["pass"] else "FAIL"
        print(f"  [{mark}] {i['id']}: expected '{i['expected']}' "
              f"({'found' if i['found'] else 'not found'} in answer)")
    shown = v["answer_text"] if len(v["answer_text"]) <= 300 else v["answer_text"][:300] + "..."
    print(f"answer_text: {shown!r}")


# ---------------------------------------------------------------------------
# Self-test (host-only, no model needed)
# ---------------------------------------------------------------------------

def self_test():
    failures = []

    def expect(name, cond):
        if not cond:
            failures.append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("normalizer:")
    expect("plain hit", normalize("QX-7741 is here") in normalize("The code QX-7741 is here."))
    expect("case/punct invariant", "qx7741" == normalize("qx-7741"))
    expect("think block stripped",
           "paris" in normalize("<think>hmm let me reason</think> Paris."))
    expect("unterminated think stripped",
           normalize("<think>never ends Paris") == "")
    expect("terminated-but-empty think keeps tail",
           "paris" in normalize("<think></think> Paris"))
    expect("no false hit", "qx7741" not in normalize("The code is TR-2918."))

    print("grading logic:")
    v = grade("QX-7741 TR-2918 MK-5501 LD-8834 Paris")
    expect("positive aggregate PASS", v["aggregate"] == "PASS")
    v = grade("QX-7741 TR-2918 MK-5501 LD-8834")  # Paris missing
    expect("missing item -> FAIL", v["aggregate"] == "FAIL")
    v_neg = grade("The codes are 1111 2222 3333 4444 and Rome", negative=True)
    expect("negative-control PASS on wrong answer", v_neg["aggregate"] == "PASS")
    v_neg2 = grade("QX-7741 TR-2918 MK-5501 LD-8834 Paris", negative=True)
    expect("negative-control FAIL when answers present", v_neg2["aggregate"] == "FAIL")

    print("gpt2 detokenizer round-trip (synthetic vocab):")
    table = _bytes_to_unicode()
    def enc(s):  # string -> gpt2-style piece
        return "".join(table[b] for b in s.encode("utf-8"))
    vocab = ["Note", "\u0120N1", "\u0120says", "\u0120the", "\u0120QX", "-", "7741", "\u0120Paris", "."]
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    text = detokenize_ids(vocab, ids)
    expect("detok text", text == "Note N1 says the QX-7741 Paris.")
    expect("detok -> needle N1 found", grade(text)["items"][0]["pass"] is True)
    expect("piece bytes restore", _piece_to_bytes("\u0120abc") == b" abc")

    print("tg-text file mode:")
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("QX-7741 TR-2918 MK-5501 LD-8834 Paris")
        v = grade(read_tg_text(tmp))
        expect("tg-text read + grading aggregate PASS", v["aggregate"] == "PASS")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("   \n  ")
        try:
            read_tg_text(tmp)
            expect("empty tg-text fails loudly", False)
        except ValueError:
            expect("empty tg-text fails loudly", True)
    finally:
        os.unlink(tmp)
    try:
        read_tg_text(tmp + ".does-not-exist")
        expect("missing tg-text fails loudly", False)
    except ValueError:
        expect("missing tg-text fails loudly", True)

    print(f"self-test: {'PASS' if not failures else 'FAIL ' + str(failures)}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# verify-vocab: host-only end-to-end vocab validation against the real model
# ---------------------------------------------------------------------------

def verify_vocab(args):
    """Tokenize a text file with llama-tokenize, detokenize here, compare."""
    import subprocess
    import os
    exe = os.path.abspath(args.tokenize_exe)  # CreateProcess rejects forward-slash paths
    with open(args.text, "rb") as f:
        src = f.read().decode("utf-8")
    res = subprocess.run(
        [exe, "-m", args.model, "--stdin", "--ids", "--no-bos"],
        input=src.encode("utf-8"), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        sys.exit(f"error: llama-tokenize failed: {res.stderr.decode(errors='replace')}")
    m = re.search(rb"\[([0-9,\s]*)\]", res.stdout)
    if not m:
        sys.exit(f"error: unexpected llama-tokenize output: {res.stdout[:200]!r}")
    body = m.group(1).strip()
    ids = [int(x) for x in body.split(b",")] if body else []

    tokens_list, tok_model = read_gpt2_vocab(args.model)
    print(f"vocab: {len(tokens_list)} tokens, tokenizer.ggml.model={tok_model!r}")
    if tok_model != "gpt2":
        sys.exit(f"error: this checker supports gpt2-style vocabs only "
                 f"(got {tok_model!r}); fail loudly rather than mis-decode")
    round_trip = detokenize_ids(tokens_list, ids)
    ok = normalize(round_trip) == normalize(src)
    print(f"ids={len(ids)} round-trip normalized match: {ok}")
    if not ok:
        a, b = normalize(src), normalize(round_trip)
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                print(f"first normalized divergence at char {i}: "
                      f"src ...{a[max(0,i-40):i+40]!r} vs detok ...{b[max(0,i-40):i+40]!r}")
                break
        sys.exit(1)
    print("verify-vocab: PASS")
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--self-test", dest="selftest", action="store_true")

    p_st = sub.add_parser("self-test")
    p_st.set_defaults(func=lambda a: self_test())

    p_vv = sub.add_parser("verify-vocab")
    p_vv.add_argument("--model", required=True)
    p_vv.add_argument("--text", required=True)
    p_vv.add_argument("--tokenize-exe", required=True)
    p_vv.set_defaults(func=verify_vocab)

    p_ck = sub.add_parser("check")
    p_ck.add_argument("--trace", default="",
                      help="JSONL telemetry trace to grade (parse tg_tokens ids; "
                           "requires --model to detokenize). Exactly one of "
                           "--trace / --tg-text must be given.")
    p_ck.add_argument("--tg-text", default="",
                      help="the runner's --tg-text generated-text file to grade "
                           "directly (already detokenized by the runner; no "
                           "--model needed). Exactly one of --trace / --tg-text "
                           "must be given.")
    p_ck.add_argument("--model", default="",
                      help="GGUF path (vocab-only read); required only with --trace")
    p_ck.add_argument("--items", default="",
                      help="comma-separated item ids to grade (default: all); "
                           "use the items= list printed by gen-needle-prompt.py")
    p_ck.add_argument("--answer", action="append", default=[],
                      metavar="ID=TEXT",
                      help="override the expected answer for one item id "
                           "(negative control: --answer K1=Rome)")
    p_ck.add_argument("--negative", action="store_true",
                      help="negative control: PASS iff NO expected answer appears")
    p_ck.add_argument("--out-verdict", default="")
    def grade_and_report(text, a):
        items = [s.strip() for s in a.items.split(",") if s.strip()] or None
        overrides = {}
        for kv in a.answer:
            k, _, v = kv.partition("=")
            if not k or not v:
                sys.exit(f"error: --answer expects ID=TEXT, got {kv!r}")
            overrides[k] = v
        v = grade(text, negative=a.negative, items=items,
                  answer_overrides=overrides or None)
        print_verdict(v)
        if a.out_verdict:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(a.out_verdict)) or ".", exist_ok=True)
            with open(a.out_verdict, "w", encoding="utf-8") as f:
                json.dump(v, f, indent=2)
        return 0 if v["aggregate"] == "PASS" else 1
    def do_check(a):
        if bool(a.trace) == bool(a.tg_text):
            sys.exit("error: exactly one of --trace or --tg-text is required")
        if a.tg_text:
            text = read_tg_text(a.tg_text)
        else:
            if not a.model:
                sys.exit("error: --model is required with --trace "
                         "(needed to detokenize tg_tokens ids)")
            _, _, ids, mixed_text = parse_trace(a.trace)
            tokens_list, tok_model = read_gpt2_vocab(a.model)
            if tok_model != "gpt2":
                sys.exit(f"error: unsupported tokenizer {tok_model!r}")
            text = detokenize_ids(tokens_list, ids)
            if mixed_text and len(normalize(mixed_text)) > len(normalize(text)):
                # Mixed-log mode: the first-16-ids fallback truncates long answers;
                # the runner's own detokenized generated-text section is the same
                # ids passed through llama_token_to_piece, so it is equivalent
                # evidence and strictly more complete.
                text = mixed_text
        return grade_and_report(text, a)
    p_ck.set_defaults(func=do_check)

    args = ap.parse_args()
    if getattr(args, "selftest", False):
        sys.exit(self_test())
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
