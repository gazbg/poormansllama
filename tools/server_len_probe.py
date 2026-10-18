#!/usr/bin/env python3
r"""One fresh server per prompt length: does a prompt of N tokens (cut from a prompt file, sent as token ids to /completion) crash the server?

  python tools/server_len_probe.py --config CFG.json --out DIR --prompt FILE --lengths 889,2048,4985 [--mode completion|chat] [--extra-arg ...]

Prints one line per length: status, prompt_n, PP t/s, server alive / exit code. stdlib only; imports tools/server_mt.py (Server).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, load_cfg, conv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--chat-fixture", help="fixture JSON: the tokens are those of conv(fixture, --turn) rendered by the server's chat template")
    ap.add_argument("--turn", type=int, default=2)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--cache-prompt", action="store_true")
    ap.add_argument("--mode", default="completion", choices=["completion", "chat"])
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    text = open(a.prompt, "r", encoding="utf-8", errors="replace").read() if a.prompt else None
    fx = json.load(open(a.chat_fixture, "r", encoding="utf-8")) if a.chat_fixture else None
    rc = 0
    for n in [int(x) for x in a.lengths.split(",")]:
        srv = Server(load_cfg(a.config), os.path.join(a.out, "len%d" % n), a.port, a.extra_arg)
        try:
            srv.start()
            if fx is not None:
                toks = srv.render_tokens(conv(fx, a.turn))[:n]
            else:
                st, js = srv.api("/tokenize", {"content": text})
                toks = js["tokens"][:n]
            t0 = time.time()
            try:
                if a.mode == "chat":
                    r = srv.chat(conv(fx, a.turn), a.n_predict, cache_prompt=a.cache_prompt)
                    st, js = r["status"], {"timings": r["timings"]}
                else:
                    st, js = srv.api("/completion", {"prompt": toks, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": a.cache_prompt})
            except Exception as e:  # noqa: BLE001
                st, js = None, {"error": repr(e)}
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            time.sleep(1.5)
            alive = srv.alive()
            print("len %6d: status %s prompt_n %s pp %.1f t/s wall %.1fs server %s" % (n, st, tm.get("prompt_n"), tm.get("prompt_per_second") or 0, time.time() - t0,
                  "alive" if alive else "DIED (exit %s)" % (srv.proc.returncode)))
            if not alive:
                rc = 1
        finally:
            srv.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
