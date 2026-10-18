#!/usr/bin/env python3
r"""G32 LM-3: needle questions through the server at exact context lengths (the semantic-QA prompts of T-GPLR-07: four archive codes planted at ~25/50/75/95% depth, the capital of France).

  python tools/server_needle.py --config CFG.json --out DIR --prompt FILE [--prompt FILE ...] [--n-predict 48] [--items N1,N2,N3,N4,K1] [--extra-arg ...]

For every prompt file the text goes to /completion (cache_prompt false, greedy, n_predict tokens). The generated text is graded with tools/semantic-qa/check-answer.py (normalised substring checks: answer
stability, never token identity) and, as the must-fail negative control, once with the answers reversed (--negative semantics). Writes DIR/needle.json and DIR/needle-<name>.txt; exit 1 when a
positive grade fails, a negative control passes, or the server died. stdlib only; imports tools/server_mt.py (Server)."""
import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from server_mt import Server, load_cfg  # noqa: E402


def load_checker():
    spec = importlib.util.spec_from_file_location("check_answer", os.path.join(HERE, "semantic-qa", "check-answer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--items", default="N1,N2,N3,N4,K1")
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    chk = load_checker()
    items = [s.strip() for s in a.items.split(",") if s.strip()]
    srv = Server(load_cfg(a.config), a.out, a.port, a.extra_arg)
    res = {}
    rc = 0
    try:
        srv.start()
        for path in a.prompt:
            text = open(path, "r", encoding="utf-8", errors="replace").read()
            name = os.path.splitext(os.path.basename(path))[0]
            t0 = time.time()
            st, js = srv.api("/completion", {"prompt": text, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": False, "seed": 1})
            tm = js.get("timings", {}) if isinstance(js, dict) else {}
            out = js.get("content", "") if isinstance(js, dict) else ""
            with open(os.path.join(a.out, "needle-%s.txt" % name), "w", encoding="utf-8") as f:
                f.write(out)
            pos = chk.grade(out, negative=False, items=items)
            neg = chk.grade(out, negative=True, items=items)
            ok = st == 200 and pos["aggregate"] == "PASS" and neg["aggregate"] == "FAIL"
            if not ok:
                rc = 1
            res[name] = {"status": st, "prompt_n": tm.get("prompt_n"), "pp_tps": tm.get("prompt_per_second"), "wall": round(time.time() - t0, 2), "text": out,
                         "positive": pos["aggregate"], "negative_control": neg["aggregate"], "items": {k: v for k, v in (pos.get("items") or {}).items()} if isinstance(pos.get("items"), dict) else pos.get("items")}
            print("%-18s prompt_n %s pp %.0f t/s: answer %r -> positive %s, negative control %s%s" % (
                name, tm.get("prompt_n"), tm.get("prompt_per_second") or 0, out[:70], pos["aggregate"], neg["aggregate"], "" if ok else "   <-- FAILED"))
        json.dump(res, open(os.path.join(a.out, "needle.json"), "w"), indent=1)
    finally:
        time.sleep(1.0)
        alive = srv.alive()
        srv.stop()
        if not alive:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
