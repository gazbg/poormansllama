#!/usr/bin/env python3
"""MP-1b: the default of --lm-prefill-prearm. Starts llama-server several times (GPU: run under tools/with-gpu-lock.ps1) and checks the start-up lines.
  python tools/check-mp-flags.py --exe build/llama-cuda/bin/mp1/llama-server.exe [--only NAME ...]
Each case: (name, model, extra args, expectation). 'ready' = /health answers 200 and the line is in the log; 'refuse' = the process exits non-zero with the line in its log."""
import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request

QWEN = "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
FLINT = "C:/Models/ALL/other/flint/flint-UD-Q5_K_XL.gguf"
COMMON = ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-c", "32768", "-np", "1"]     # the server default is -np auto (4): the cases that need one slot say so

CASES = [
    # name, model, args, mode, must-contain, must-not-contain
    ("default-np1-hybrid", QWEN, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000"], "ready", ["idle preparation of the pass: always", "(start-up, history 0): armed"], ["idle preparation of the pass is off"]),
    ("default-np2", QWEN, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000", "-np", "2"], "ready", ["idle preparation of the pass is off (-np is not 1)"], ["need -np 1"]),
    ("default-no-hybrid", QWEN, ["--lm-prefill", "-ngl", "99", "-ncmoe", "37", "-fit", "off"], "ready", ["idle preparation of the pass is off (no --hybrid-moe)"], ["need -np 1"]),
    ("explicit-np2", QWEN, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000", "-np", "2", "--lm-prefill-prearm", "always"], "refuse", ["need -np 1"], []),
    ("explicit-off", QWEN, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000", "--lm-prefill-fuse", "--lm-prefill-prearm", "off"], "ready", ["idle preparation of the pass: off"], ["start-up, history"]),
    ("flint-default", FLINT, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000", "-ub", "2048"], "ready", ["idle preparation of the pass: always", "(start-up, history 0): armed"], []),
    ("flint-explicit", FLINT, ["--hybrid-moe", "--lm-prefill", "-fitt", "16000", "-ub", "2048", "--lm-prefill-prearm", "off"], "ready", [], ["start-up, history"]),
]


def run_case(exe, case, port, ready_s, refuse_s):
    name, model, args, mode, must, must_not = case
    cmd = [os.path.abspath(exe), "--model", model] + COMMON + args + ["--port", str(port), "--no-webui"]
    lines = []
    env = dict(os.environ, LLAMA_LM_LOG="1")     # the start-up arm line is printed only with this
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    def pump():
        for raw in p.stdout:
            lines.append(raw.decode("utf-8", "replace").replace("\x00", ""))

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    t0 = time.time()
    verdict = "timeout"
    deadline = ready_s if mode == "ready" else refuse_s
    while time.time() - t0 < deadline:
        rc = p.poll()
        if rc is not None:
            verdict = "exited %d" % rc
            break
        if mode == "ready":
            try:
                if urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).status == 200:
                    time.sleep(3.0)     # the start-up preparation runs right after the server is up
                    verdict = "ready"
                    break
            except Exception:
                pass
        time.sleep(1.0)
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
    th.join(timeout=5)
    text = "".join(lines)
    ok = True
    notes = []
    if mode == "ready" and verdict != "ready":
        ok = False
        notes.append("not ready (%s)" % verdict)
    if mode == "refuse" and not verdict.startswith("exited") or (mode == "refuse" and verdict == "exited 0"):
        ok = False
        notes.append("did not refuse (%s)" % verdict)
    for s in must:
        if s not in text:
            ok = False
            notes.append("missing: " + s)
    for s in must_not:
        if s in text:
            ok = False
            notes.append("unexpected: " + s)
    print("%-20s %s  %s  (%s, %.0f s)" % (name, "OK  " if ok else "FAIL", "; ".join(notes) if notes else "all expectations", verdict, time.time() - t0))
    if not ok:
        for l in lines[-25:]:
            print("    | " + l.rstrip()[:220])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--ready-s", type=int, default=240)
    ap.add_argument("--refuse-s", type=int, default=150)
    a = ap.parse_args()
    ok = True
    for c in CASES:
        if a.only and c[0] not in a.only:
            continue
        ok = run_case(a.exe, c, a.port, a.ready_s, a.refuse_s) and ok
    sys.exit(0 if ok else 1)


main()
