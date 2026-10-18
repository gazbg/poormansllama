#!/usr/bin/env python3
r"""G32 LM-3: the logits of the FIRST generated token of a prompt that the prefill alone produces (run the server with --ctx-checkpoints 0 so that the prompt is not cut into the checkpoint chunks, and the last batch
-- the one whose last token carries the logits -- goes through the prefill path under test).

  python tools/server_firsttok.py --config CFG.json --out DIR --prompt FILE[:LEN] [--prompt FILE[:LEN] ...] [--reps 3] [--n-probs 10] [--extra-arg ...]

For every prompt (a prefix of LEN tokens of FILE, as token ids) it sends --reps fresh requests (cache_prompt false, n_predict 1, greedy) and records the top --n-probs (token, logprob) of the first token.
Writes DIR/firsttok.json and prints, per prompt, the max |delta logprob| over the tokens that two runs share (rep 1 against every later rep) and whether the top-1 token agrees.
tools/server_firsttok_cmp.py compares two such files (two servers). stdlib only; imports tools/server_mt.py (Server)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, load_cfg  # noqa: E402


def top(js):
    cp = js.get("completion_probabilities") or []
    if not cp:
        return []
    c0 = cp[0]
    tops = c0.get("top_logprobs") or c0.get("top_probs") or []
    out = []
    for t in tops:
        out.append((t.get("token"), t.get("logprob") if "logprob" in t else (None if t.get("prob") is None else __import__("math").log(max(t["prob"], 1e-30)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    srv = Server(load_cfg(a.config), a.out, a.port, a.extra_arg)
    res = {}
    rc = 0
    try:
        srv.start()
        for spec in a.prompt:
            path, ln = spec, ""
            head, sep, tail = spec.rpartition(":")
            if sep and tail.isdigit():     # FILE:LEN (a drive letter's colon is not a length)
                path, ln = head, tail
            text = open(path, "r", encoding="utf-8", errors="replace").read()
            st, js = srv.api("/tokenize", {"content": text})
            toks = js["tokens"]
            if ln:
                toks = toks[:int(ln)]
            runs = []
            for r in range(a.reps):
                t0 = time.time()
                st, js = srv.api("/completion", {"prompt": toks, "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": a.n_probs, "seed": 1})
                tm = js.get("timings", {}) if isinstance(js, dict) else {}
                runs.append({"status": st, "top": top(js), "text": js.get("content") if isinstance(js, dict) else None, "prompt_n": tm.get("prompt_n"), "pp_tps": tm.get("prompt_per_second"), "wall": round(time.time() - t0, 3)})
            key = "%s:%d" % (os.path.basename(path), len(toks))
            res[key] = runs
            base = dict((t, lp) for t, lp in runs[0]["top"])
            deltas = []
            for r in runs[1:]:
                other = dict((t, lp) for t, lp in r["top"])
                shared = [t for t in base if t in other and base[t] is not None and other[t] is not None]
                deltas.append((max([abs(base[t] - other[t]) for t in shared]) if shared else None, (runs[0]["top"][0][0] if runs[0]["top"] else None) == (r["top"][0][0] if r["top"] else None)))
            print("%-30s prompt_n %s reps %d: top1 %r; run 1 vs later: max|dlogprob| %s; top-1 agrees %s" % (
                key, runs[0]["prompt_n"], len(runs), runs[0]["top"][0] if runs[0]["top"] else None,
                ["%.4f" % d[0] if d[0] is not None else "-" for d in deltas], [d[1] for d in deltas]))
        json.dump(res, open(os.path.join(a.out, "firsttok.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        alive = srv.alive()
        srv.stop()
        if not alive:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
