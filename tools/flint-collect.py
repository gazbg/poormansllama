#!/usr/bin/env python3
"""Start a llama-server on a flint (or any) model, run tools/dense_equiv.py collect against it, stop it.  Run it UNDER tools/with-gpu-lock.ps1.

  python tools/flint-collect.py --label ub512 --flags "-ub 512" --sets std,big

Writes build/flint/eq-<label>-<set>.json (+ sha256 lines on stdout) and build/flint/srv-<label>.log.  Shared server flags (-fa on, -np 1,
q8_0 KV, -t 16, -lm none) are set here; --flags adds the arm's own.  The prompts and the rule are those of tools/dense_equiv.py
(equal chunk shape only; AGENTS.md).
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE = os.environ.get("DENSE_EXE") or os.path.join(ROOT, "build", "llama-cuda", "bin", "Release", "llama-server.exe")
MODEL = r"C:\Models\ALL\other\flint\flint-UD-Q5_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--sets", default="std,big")
    ap.add_argument("--n-big", type=int, default=24, help="prompts of the big (KL) set; the std set always takes its 8")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--model-path", default=MODEL)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--health-timeout", type=int, default=600)
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "build", "flint")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "srv-" + a.label + ".log")
    cmd = [EXE, "--model", a.model_path, "--host", "127.0.0.1", "--port", str(a.port), "-fa", "on", "-np", "1", "-c", str(a.ctx),
           "-ctk", "q8_0", "-ctv", "q8_0", "-t", "16", "--no-warmup", "-lm", "none"] + a.flags.split()
    print("CMD", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["LLAMA_LM_LOG"] = "1"
    with open(log, "wb") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    rc = 1
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < a.health_timeout:
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
        print("load_s", round(time.time() - t0, 1), "ok", ok, flush=True)
        if ok:
            rc = 0
            for s in a.sets.split(","):
                out = os.path.join(out_dir, f"eq-{a.label}-{s}.json")
                r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "dense_equiv.py"), "collect", "--port", str(a.port), "--out", out,
                                    "--n-predict", str(a.n_predict), "--set", s, "--n-prompts", str(a.n_big if s == "big" else 8)], cwd=ROOT)
                rc = rc or r.returncode
                if os.path.exists(out):
                    print("SHA256", os.path.basename(out), hashlib.sha256(open(out, "rb").read()).hexdigest(), flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
