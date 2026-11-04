#!/usr/bin/env python3
"""FL-1 needle answers through a flint server (run UNDER tools/with-gpu-lock.ps1): starts the server with the arm's flags, sends cold needle prompts of the given sizes,
prints prompt size, prefill rate, the answer and a PASS / FAIL on the substring 7391, and how many layer-major passes the server log shows.

  python tools/flint-needle.py --label fl1 --flags "-ngl 99 --cpu-moe -ub 2048 -b 4096 --lm-prefill" --chars 13500,81000 --ctx 65536

Exit status 0 only if every size answered.  The prompt is tools/dense_probe.py build_prompt (repo text read from the pinned commit, the needle at --depth, then the question).
"""
import argparse
import os
import re
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
    ap.add_argument("--chars", default="13500", help="comma list of prompt sizes in characters (about 3.4 characters per flint token of repo text)")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--expect-refusal", default="", help="expect the server to exit at start-up with this text in its log (the loud refusal paths of FL-1)")
    a = ap.parse_args()

    import dense_probe

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
    rc = 1
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 600:
            if srv.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if a.expect_refusal:
            time.sleep(1)
            text = open(log, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
            dead = srv.poll() is not None
            hit = a.expect_refusal in text
            print(f"refusal check: server exited {dead}, text {a.expect_refusal!r} in log {hit}, healthy {ok}")
            for l in text.splitlines():
                if a.expect_refusal in l:
                    print("   ", l.strip()[:400])
                    break
            return 0 if (dead and hit and not ok) else 1
        if not ok:
            print("SERVER NOT HEALTHY rc", srv.poll())
            return 1
        rc = 0
        for chars in [int(x) for x in a.chars.split(",")]:
            prompt = dense_probe.build_prompt(chars, a.depth)
            d = dense_probe.post(a.port, {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "cache_prompt": False})
            t = d["timings"]
            good = "7391" in d["content"]
            rc = rc or (0 if good else 1)
            print(f"{'PASS' if good else 'FAIL'} chars {chars:7d} prompt_n {t['prompt_n']:6d} prefill {t['prompt_ms'] / 1000:7.2f}s ({t['prompt_per_second']:7.1f} tok/s) answer {d['content'].strip()[:60]!r}", flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass
        text = open(log, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
        passes = len(re.findall(r"lm-prefill: pass \d+:", text))
        print(f"passes in the log: {passes}; enabled line: {[l.strip()[:200] for l in text.splitlines() if 'lm_prefill_enable' in l][:1]}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
