#!/usr/bin/env python3
"""Per-layer output hashes of one prompt through a flint server (debug; run UNDER tools/with-gpu-lock.ps1), and the comparison of two such runs.

  python tools/flint-layerhash.py run --label stock2048 --prompt 3 --flags "-ngl 99 --cpu-moe -ub 2048 -b 4096"
  python tools/flint-layerhash.py run --label lm2048    --prompt 3 --flags "-ngl 99 --cpu-moe -ub 2048 -b 4096 --lm-prefill"
  python tools/flint-layerhash.py diff build/flint/lh-stock2048.log build/flint/lh-lm2048.log

The server runs with LLAMA_DEBUG_LAYER_HASH=1 (llama-context.cpp): every "l_out-N" tensor of the ordinary decode and of the layer-major pass's pieces is hashed on the host
and logged as `LDBG l_out-N rows=R fnv=H first=a b c`.  The prompt is prompt i of tools/dense_equiv.py (the needle set); one token is generated.
`diff` lines the two logs up by (rows, layer) in order of appearance and prints the first layers whose hashes differ.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE = os.environ.get("DENSE_EXE") or os.path.join(ROOT, "build", "llama-cuda", "bin", "Release", "llama-server.exe")
MODEL = r"C:\Models\ALL\other\flint\flint-UD-Q5_K_XL.gguf"
LINE = re.compile(r"LDBG l_out-(\d+) rows=(\d+) fnv=([0-9a-f]+) first=(\S+) (\S+) (\S+)")


def run(a):
    out_dir = os.path.join(ROOT, "build", "flint")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "lh-" + a.label + ".log")
    cmd = [EXE, "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port), "-fa", "on", "-np", "1", "-c", str(a.ctx), "-ctk", "q8_0", "-ctv", "q8_0", "-t", "16",
           "--no-warmup", "-lm", "none"] + a.flags.split()
    env = dict(os.environ)
    env["LLAMA_DEBUG_LAYER_HASH"] = "1"
    env["LLAMA_LM_LOG"] = "1"
    print("CMD", " ".join(cmd), flush=True)
    with open(log, "wb") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    try:
        t0 = time.time()
        ok = False
        while time.time() - t0 < 600:
            if srv.poll() is not None:
                print("SERVER DIED rc", srv.returncode)
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
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            import dense_equiv
            big = dense_equiv.big_prompts() if a.set == "big" else None
            for pi in [int(x) for x in str(a.prompt).split(",")]:      # several prompts in sequence: the last one's hashes are the ones of interest when the history matters
                d = dense_equiv.post(a.port, {"prompt": dense_equiv.build(pi, big), "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 2})
                t = d["timings"]
                print("prompt", pi, "prompt_n", t["prompt_n"], "content", repr(d["content"]), flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass
    n = sum(1 for l in open(log, "rb").read().decode("utf-8", "replace").splitlines() if "LDBG" in l)
    print("LDBG lines:", n, "->", log)


def parse(path):
    seq = {}
    for l in open(path, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace").splitlines():
        m = LINE.search(l)
        if m:
            il, rows, h = int(m.group(1)), int(m.group(2)), m.group(3)
            seq.setdefault((rows, il), []).append((h, m.group(4), m.group(5), m.group(6)))
    return seq


def diff(a):
    """For every row count in both logs: the occurrences of each layer are compared aligned at the END of the logs (the last request's, when the logs hold several), and the first difference in execution order (layer, then chunk) is printed."""
    s1, s2 = parse(a.log1), parse(a.log2)
    keys = sorted(set(s1) & set(s2), key=lambda k: (-k[0], k[1]))
    print("rows values in both logs:", sorted({k[0] for k in keys}, reverse=True))
    shown = 0
    for rows in sorted({k[0] for k in keys}, reverse=True):
        first_bad = None
        n_same = 0
        for il in range(0, 49):
            k = (rows, il)
            if k not in s1 or k not in s2:
                continue
            n = min(len(s1[k]), len(s2[k]), a.last)
            for j in range(n):
                h1, h2 = s1[k][len(s1[k]) - n + j], s2[k][len(s2[k]) - n + j]
                if h1[0] == h2[0]:
                    n_same += 1
                elif first_bad is None:
                    first_bad = (il, j, n, h1, h2)
        if first_bad is None:
            print(f"rows={rows}: all {n_same} (layer, occurrence) pairs identical")
        else:
            il, j, n, h1, h2 = first_bad
            print(f"rows={rows}: {n_same} pairs identical before/around, FIRST DIFFERENCE at layer {il}, occurrence {j} of the last {n}: {h1[1:]} vs {h2[1:]}")
        shown += 1
        if shown >= a.max_groups:
            break


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--prompt", default="3", help="prompt index, or a comma list sent in sequence")
    r.add_argument("--set", choices=["std", "big"], default="std")
    r.add_argument("--flags", default="")
    r.add_argument("--ctx", type=int, default=16384)
    r.add_argument("--port", type=int, default=8099)
    d = sub.add_parser("diff")
    d.add_argument("log1")
    d.add_argument("log2")
    d.add_argument("--max-groups", type=int, default=3)
    d.add_argument("--last", type=int, default=64, help="compare at most this many trailing occurrences of a (rows, layer)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        diff(a)


if __name__ == "__main__":
    main()
