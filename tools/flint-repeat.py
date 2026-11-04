#!/usr/bin/env python3
"""Send one prompt of tools/dense_equiv.py to a fresh flint server several times and print the first token's top-20 logprobs of each reply (run-to-run determinism probe; run UNDER tools/with-gpu-lock.ps1).

  python tools/flint-repeat.py --label rep-l512 --flags "-ngl 99 --cpu-moe -ub 512 -b 4096 --ctx-checkpoints 0 --lm-prefill --lm-prefill-ubatch 512 --lm-prefill-segment 4096 --lm-prefill-min 256" --set big --index 18 --reps 4

The server environment is the caller's (GGML_OP_OFFLOAD_MIN_BATCH etc.).  Prints one line per rep: the first token id and logprob, a short hash of the top-20 list, and the maximum |dlogprob| to rep 0.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
EXE = os.environ.get("DENSE_EXE") or os.path.join(ROOT, "build", "llama-cuda", "bin", "Release", "llama-server.exe")
MODEL = r"C:\Models\ALL\other\flint\flint-UD-Q5_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--set", choices=["std", "big"], default="big")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()

    import dense_equiv
    prompts = dense_equiv.big_prompts() if a.set == "big" else None
    text = dense_equiv.build(a.index, prompts)

    out_dir = os.path.join(ROOT, "build", "flint")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "srv-" + a.label + ".log")
    cmd = [EXE, "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port), "-fa", "on", "-np", "1", "-c", str(a.ctx), "-ctk", "q8_0", "-ctv", "q8_0", "-t", "16",
           "--no-warmup", "-lm", "none"] + a.flags.split()
    print("CMD", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["LLAMA_LM_LOG"] = "1"
    with open(log, "wb") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 600:
            if srv.poll() is not None:
                print("SERVER DIED rc", srv.returncode, flush=True)
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if ok:
            ref = None
            for r in range(a.reps):
                d = dense_equiv.post(a.port, {"prompt": text, "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 20})
                top = [(x.get("id"), x.get("logprob")) for x in ((d.get("completion_probabilities") or [{}])[0].get("top_logprobs") or [])]
                h = hashlib.sha256(repr(top).encode()).hexdigest()[:10]
                lp0 = top[0][1] if top else float("nan")
                if ref is None:
                    ref = {i: lp for i, lp in top}
                dmax = max((abs(lp - ref[i]) for i, lp in top if i in ref), default=float("nan"))
                print(f"rep {r}: prompt_n {d['timings']['prompt_n']} top1 id {top[0][0] if top else None} lp {lp0:.6f} hash {h} max|d| to rep0 {dmax:.6f}", flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
