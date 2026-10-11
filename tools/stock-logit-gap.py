#!/usr/bin/env python3
"""Stock logit-gap probe (contract G-TG8, criterion C6, general form after amendment A2).

llama-completion cannot print logits, but it takes --logit-bias. For a token x let
    b*_x = smallest bias b such that `--logit-bias x+b` makes the stock greedy pick x first
         = logit(stock top) - logit(x)        (0 when x already IS the stock top)
Boosting x changes the first token only when x overtakes the top, so "the output changed" is a
sufficient flip test and no token text is needed. For two tokens A, B:
    gap_STK(A over B) = logit(A) - logit(B) = b*_B - b*_A
which is compared with the other engine's own gap (LM `pp_last_logits`).

Per token the probe makes: one +hi run (flip?), one -hi run when +hi did not flip (is x the stock
top?), then a bisection to --res. Each probe is one `llama-completion -n 1` run (model load
~10-25 s), so the whole script must run inside ONE mutex hold; name the exe on the wrapped command
line so the wrapper can attest it:

  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g1-gap -Command \\
    "python tools/stock-logit-gap.py --exe build\\llama-cuda\\bin\\g1\\llama-completion.exe ..."

Several --config LABEL=FLAGS entries repeat the measurement under different valid stock
configurations (e.g. -ub 128, f16 KV): their spread is the stock's own numeric noise on the gap.
It measures; it claims nothing.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

MODEL = "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
BASE = ["-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "none", "-fit", "off",
        "-c", "262000", "-n", "1", "--ignore-eos", "--temp", "0", "-no-cnv", "--no-display-prompt",
        "-ncmoe", "37", "-t", "16", "--no-mmap"]


def run_once(args, extra, token, bias):
    cmd = [args.exe, "-m", args.model, "-f", args.prompt_file] + BASE + extra
    if bias:
        cmd += ["-l", "%d%+g" % (token, bias)]  # TOKEN_ID(+/-)BIAS, e.g. 198+3.5 or 198-8
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, timeout=args.timeout)
    return p.returncode, p.stdout.decode("utf-8", errors="replace"), time.time() - t, \
        p.stderr.decode("utf-8", errors="replace")


def probe(args, label, extra, runs):
    """-> (text0, {token: {...}})"""
    def go(token, bias):
        rc, text, dur, err = run_once(args, extra, token, bias)
        runs.append({"token": token, "bias": bias, "rc": rc, "text": text[:40], "wall_s": round(dur, 1)})
        print("  [%s] tok %-7s bias %-6g rc=%d text=%r (%.1fs)" % (label, token, bias, rc, text[:16], dur), flush=True)
        if rc != 0:
            print("  stderr tail: " + err[-400:].replace("\n", " | "), flush=True)
            raise RuntimeError("llama-completion failed (rc=%d)" % rc)
        return text

    text0 = go(args.tokens[0], 0)
    out = {}
    for tok in args.tokens:
        r = {"token": tok}
        if go(tok, args.hi) == text0:
            r["is_top"] = go(tok, -args.hi) != text0
            if r["is_top"]:
                r["lo"], r["hi"], r["est"] = 0.0, 0.0, 0.0
            else:
                r["note"] = "no flip either way up to %g: gap > hi" % args.hi
                r["lo"], r["hi"], r["est"] = args.hi, None, None
        else:
            r["is_top"] = False
            lo, hi = 0.0, args.hi
            while hi - lo > args.res:
                mid = round((lo + hi) / 2, 6)
                if go(tok, mid) != text0:
                    hi = mid
                else:
                    lo = mid
            r["lo"], r["hi"], r["est"] = lo, hi, (lo + hi) / 2
        out[tok] = r
    return text0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--tokens", required=True, help="A,B: token ids; the gap measured is logit(A) - logit(B)")
    ap.add_argument("--lm-logits", default="", help="A:logitA,B:logitB from the other engine, to print Delta")
    ap.add_argument("--limit", type=float, default=2.42)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--config", action="append", default=[],
                    help="LABEL=FLAGS (extra llama-completion flags, shell-quoted); repeatable; default: base=")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.tokens = [int(x) for x in args.tokens.split(",")]
    if len(args.tokens) != 2:
        ap.error("--tokens needs exactly two ids")
    lm = {}
    if args.lm_logits:
        for part in args.lm_logits.split(","):
            k, _, v = part.partition(":")
            lm[int(k)] = float(v)
    a_tok, b_tok = args.tokens
    results = []
    for c in (args.config or ["base="]):
        label, _, flags = c.partition("=")
        print("config %s: %s" % (label, flags or "(base flags)"), flush=True)
        runs = []
        text0, per = probe(args, label, shlex.split(flags, posix=False) if flags else [], runs)
        ra, rb = per[a_tok], per[b_tok]
        res = {"label": label, "extra": flags, "baseline_text": text0[:40], "tokens": per, "runs": runs}
        if ra["est"] is not None and rb["est"] is not None:
            res["gap_stock_A_over_B"] = rb["est"] - ra["est"]
            res["gap_stock_bounds"] = [rb["lo"] - ra["hi"], rb["hi"] - ra["lo"]]
        if lm and a_tok in lm and b_tok in lm and "gap_stock_A_over_B" in res:
            res["gap_lm_A_over_B"] = lm[a_tok] - lm[b_tok]
            res["delta"] = abs(res["gap_stock_A_over_B"] - res["gap_lm_A_over_B"])
            res["delta_bounds"] = [max(0.0, abs(res["gap_lm_A_over_B"] - x)) for x in res["gap_stock_bounds"]]
            res["within_limit"] = res["delta"] <= args.limit
        results.append(res)
    rec = {"exe": args.exe, "exe_size": os.path.getsize(args.exe), "model": args.model,
           "prompt_file": args.prompt_file, "tokens": args.tokens, "res": args.res, "limit": args.limit,
           "lm_logits": lm, "configs": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("")
    for r in results:
        print("config %-7s stock gap(%d over %d) = %s  bounds %s | LM gap = %s | Delta = %s  within %.2f: %s" % (
            r["label"], a_tok, b_tok, r.get("gap_stock_A_over_B"), r.get("gap_stock_bounds"),
            r.get("gap_lm_A_over_B"), r.get("delta"), args.limit, r.get("within_limit")))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
