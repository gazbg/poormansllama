#!/usr/bin/env python3
r"""G32 LM-6(iii): the layer-major pass with TWO slots (-np 2, unified KV): both slots, in turn and at once, with the answer of every step recorded the way tools/server_gen.py records it (so `server_gen.py compare`
can compare two servers step by step).

  python tools/server_np2.py --config CFG.json --out DIR --x FILE_X --y FILE_Y --ext FILE_EXT [--n-predict 48] [--ext-tokens 3000] [--extra-arg ...]

Steps (greedy, n_probs 5, cache_prompt true; the slot of a request is the server's choice unless --pin-slots; the "slot" column is the conversation, X = 0, Y = 1):
  s0-fresh   slot 0: X                              (a fresh sequence, the other slot is empty)
  s1-fresh   slot 1: Y                              (a fresh sequence while slot 0 holds cells: the layout that -np 1 never has)
  s0-turn2   slot 0: X + an extension of N tokens   (a continuing sequence, slot 1 holds cells)
  s1-turn2   slot 1: Y + an extension
  s0-turn3   slot 0: X + extension + a second extension
  both       slots 0 and 1 at the same time: X + 2 extensions and Y + 2 extensions (the server batches both: the pass must decline, the answers must be right)
Writes DIR/gen.json (steps with token / logprob / top lists, prompt_n, cache_n, pp t/s); exit 1 on a failed request."""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, load_cfg  # noqa: E402
from server_gen import top_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--ext-tokens", type=int, default=3000)
    ap.add_argument("--len", type=int, default=0, help="use only the first LEN tokens of X and Y (0 = whole files)")
    ap.add_argument("--pin-slots", action="store_true", help="send id_slot with every request. DEFAULT OFF: with a unified KV cache the server clears an idle slot (its prompt goes to the host prompt cache) and only consults that cache when it picks the slot itself, so a pinned request always reprocesses its prompt (stock -kvu does the same)")
    ap.add_argument("--single", action="store_true", help="slot 0 only: s0-fresh, s0-turn2, s0-turn3 (works with -np 1)")
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    srv = Server(load_cfg(a.config), a.out, a.port, a.extra_arg)
    res = {}
    rc = 0
    try:
        srv.start()

        def toks(path, n=0):
            text = open(path, "r", encoding="utf-8", errors="replace").read()
            st, js = srv.api("/tokenize", {"content": text})
            t = js["tokens"]
            return t[:n] if n else t

        X, Y = toks(a.x, a.len), toks(a.y, a.len)
        E = toks(a.ext)
        E1, E2 = E[:a.ext_tokens], E[a.ext_tokens:2 * a.ext_tokens]
        # the extensions are cut from different places of the file so that the two slots never share a prefix beyond the base prompt
        E3, E4 = E[2 * a.ext_tokens:3 * a.ext_tokens], E[3 * a.ext_tokens:4 * a.ext_tokens]
        assert len(E4) > 0, "the extension file is too short for 4 x %d tokens" % a.ext_tokens

        def req(name, slot, prompt, out):
            t0 = time.time()
            body = {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": True, "n_probs": 5, "seed": 1}
            if a.pin_slots:
                body["id_slot"] = slot
            st, js = srv.api("/completion", body)
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            out[name] = {"status": st, "slot": slot, "prompt_len": len(prompt), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "pp_tps": tm.get("prompt_per_second"),
                         "tg_tps": tm.get("predicted_per_second"), "text": js.get("content") if isinstance(js, dict) else None,
                         "steps": top_lists(js) if isinstance(js, dict) else [], "wall": round(time.time() - t0, 2)}

        seq = [("s0-fresh", 0, X), ("s1-fresh", 1, Y), ("s0-turn2", 0, X + E1), ("s1-turn2", 1, Y + E3), ("s0-turn3", 0, X + E1 + E2), ("s1-turn3", 1, Y + E3 + E4)]
        if a.single:
            seq = [s for s in seq if s[1] == 0]
        for name, slot, prompt in seq:
            req(name, slot, prompt, res)
            r = res[name]
            print("%-10s slot %d: prompt %6d tokens, processed %6s, cached %6s, pp %6.0f t/s, tg %5.1f t/s, status %s" % (name, slot, len(prompt), r["prompt_n"], r["cache_n"], r["pp_tps"] or 0, r["tg_tps"] or 0, r["status"]))
            if r["status"] != 200 or not r["steps"]:
                rc = 1
        # both slots at once: new extensions on top of the cached prompts
        ths = []
        for name, slot, prompt in () if a.single else (("both-s0", 0, X + E1 + E2 + E4[:1000]), ("both-s1", 1, Y + E3 + E4 + E2[:1000])):
            th = threading.Thread(target=req, args=(name, slot, prompt, res))
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        for name in () if a.single else ("both-s0", "both-s1"):
            r = res[name]
            print("%-10s slot %d: prompt %6d tokens, processed %6s, cached %6s, pp %6.0f t/s, tg %5.1f t/s, status %s" % (name, r["slot"], r["prompt_len"], r["prompt_n"], r["cache_n"], r["pp_tps"] or 0, r["tg_tps"] or 0, r["status"]))
            if r["status"] != 200 or not r["steps"]:
                rc = 1
        json.dump(res, open(os.path.join(a.out, "gen.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        alive = srv.alive()
        srv.stop()
        if not alive:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
