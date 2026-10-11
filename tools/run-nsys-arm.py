#!/usr/bin/env python3
r"""Re-run ONE recorded LM arm under Nsight Systems, export the SQLite and attribute the inter-token gap (goal G-TG8 / G3).

The command line and environment come from the arm record in a tools/run-tg-arms.py verdict, so the profiled run is the measured configuration
(only --tg-tokens is shortened, and the hybrid log path is redirected). It records; it claims nothing. nsys perturbs timing, so the tok/s of a
profiled run is NOT a result: only the structure (kernel order, gaps, idle) is.

  powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag <tag> -Command \
    "python tools\run-nsys-arm.py --verdict build\llama-cuda\g4-code\g4-code-verdict.json --arm HYB42R#1 --tg-tokens 64 \
     --out build\llama-cuda\g4-nsys\hyb42r --attest build\llama-cuda\bin\g4\llama-layer-streaming.exe"

Host-only follow-up on the exported file:  python tools/nsys-gap.py <out>.sqlite --layers 40 --skip 4
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

NSYS_DEFAULT = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--arm", required=True, nargs="+", help="arm record name(s) in the verdict, e.g. HYB28#1 HYB42R#1")
    ap.add_argument("--tg-tokens", type=int, default=64)
    ap.add_argument("--out", required=True, help="output base path; the arm name is appended (no extension)")
    ap.add_argument("--nsys", default=NSYS_DEFAULT)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--attest", nargs="*", default=[], help="exe paths named only so the wrapper can attest them")
    a = ap.parse_args()

    v = json.load(open(os.path.join(ROOT, a.verdict) if not os.path.isabs(a.verdict) else a.verdict, encoding="utf-8"))
    worst = 0
    for arm in a.arm:
        worst = max(worst, one_arm(a, v, arm))
    return worst


def one_arm(a, v, arm):
    rec = v["arms"][arm]
    cmd = rec["cmd"].split(" ")
    out_base = (a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)) + "-" + re.sub(r"[^A-Za-z0-9]+", "_", arm)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    # shorten the generation (--tg-tokens N; everything else stays as recorded) and redirect every output path the recorded run wrote,
    # so profiling never overwrites the measured arm's telemetry or text
    for i, tok in enumerate(cmd):
        if tok == "--tg-tokens":
            cmd[i + 1] = str(a.tg_tokens)
        elif tok == "--telemetry":
            cmd[i + 1] = out_base + ".telemetry.jsonl"
        elif tok == "--tg-text":
            cmd[i + 1] = out_base + ".tgtext.txt"
    env = dict(os.environ)
    env.update({k: str(x) for k, x in rec.get("env", {}).items()})
    env["LLAMA_MOE_HYBRID_LOG"] = out_base + ".hybrid.jsonl"
    prof = [a.nsys, "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
            "--force-overwrite=true", "--output=" + out_base] + cmd
    print("nsys profile of arm %s: %s" % (arm, " ".join(prof)), flush=True)
    t0 = time.time()
    with open(out_base + ".run.log", "wb") as lf:
        rc = subprocess.run(prof, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print("profile rc=%s wall=%.1fs" % (rc, time.time() - t0), flush=True)
    rep = out_base + ".nsys-rep"
    db = out_base + ".sqlite"
    if rc != 0 or not os.path.exists(rep):
        tail = open(out_base + ".run.log", "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")[-1500:]
        print("PROFILE FAILED; log tail:\n" + tail)
        return 1
    ex = subprocess.run([a.nsys, "export", "--type=sqlite", "--force-overwrite=true", "--output=" + db, rep], capture_output=True, text=True)
    print("export rc=%s %s" % (ex.returncode, (ex.stdout + ex.stderr).strip()[-300:]), flush=True)
    if ex.returncode != 0:
        return 1
    for tool in ("nsys-token.py", "nsys-gap.py"):
        res = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), db, "--layers", str(a.layers), "--skip", "4"],
                             capture_output=True, text=True)
        print("--- %s (%s)" % (tool, arm))
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr[-800:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
