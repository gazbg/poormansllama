#!/usr/bin/env python3
"""KL divergence of KV cache type pairs with llama.cpp's own tool (llama-perplexity --kl-divergence-base / --kl-divergence), dense Qwen3.8-27B-UD-Q6_K_XL on the RTX 5090 Laptop.
Run it UNDER tools/with-gpu-lock.ps1.

  python tools/dense/kl-ppl.py [--arms f16/f16,f16/f16,q8_0/q8_0,q8_0/q4_0,q4_0/q4_0] [--ctx 16384] [--text tests/WebPageTest 10kcontext.txt]

The first arm writes the base logits (build/dense/kl-base.bin: n_vocab x 2 bytes per scored token, the second half of every chunk); every other arm is compared with them token by token over the whole
vocabulary: mean KL divergence, perplexity ratio, probability change, share of tokens whose top-1 is the same.  A second f16/f16 arm gives the run-to-run noise floor.  Same model placement in every arm
(the first 24 layers on the host through a tensor override), only -ctk / -ctv change.
"""
import argparse, os, re, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # llama-perplexity prints a Greek delta; a redirected stdout is cp1252 on Windows

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=os.path.join(ROOT, "build", "llama-cuda-faq", "bin", "Release", "llama-perplexity.exe"))
ap.add_argument("--model", default=r"C:\Models\ALL\other\qwen-3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf")
ap.add_argument("--arms", default="f16/f16,f16/f16,q8_0/q8_0,q8_0/q4_0,q4_0/q4_0")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--ub", type=int, default=512, help="batch / ubatch size (a different value is a benign arithmetic change, for a noise-floor control)")
ap.add_argument("--text", default=os.path.join(ROOT, "tests", "WebPageTest 10kcontext.txt"))
ap.add_argument("--host-layers", default=r"blk\.(?:[0-9]|1[0-9]|2[0-3])\.=CPU")
ap.add_argument("--base", default=os.path.join(ROOT, "build", "dense", "kl-base.bin"), help="the file the first arm writes and the others are compared with")
ap.add_argument("--reuse-base", action="store_true", help="the base file already exists (an earlier run): every arm, the first included, is compared with it")
a, _ = ap.parse_known_args()      # the GPU wrapper passes a stray "*" (from the "*>" redirect): ignored

base = a.base
os.makedirs(os.path.join(ROOT, "build", "dense", "win"), exist_ok=True)
KEYS = ("Mean PPL(Q)  ", "Mean PPL(base)", "Mean PPL(Q)/PPL(base)", "Mean    KLD", "Maximum KLD", "99.9% KLD", "Median KLD", "Same top p", "RMS Δp", "Mean    Δp")
rows = []
for n, arm in enumerate(a.arms.split(",")):
    arm, _, opt = arm.partition("+")          # "q8_0/q8_0+tile": same arm with LLAMA_FATTN_HEAD_TILE=1 (the head-tiled f16 conversion of the flash attention)
    ctk, ctv = arm.split("/")
    env = dict(os.environ)
    if opt == "tile":
        env["LLAMA_FATTN_HEAD_TILE"] = "1"
    else:
        env.pop("LLAMA_FATTN_HEAD_TILE", None)
    log = os.path.join(ROOT, "build", "dense", "win", f"klppl-{n}-{ctk}-{ctv}{'-' + opt if opt else ''}.log")
    cmd = [a.exe, "-m", a.model, "-f", a.text, "-c", str(a.ctx), "--chunks", str(a.chunks), "-b", str(a.ub), "-ub", str(a.ub), "-fa", "on", "-ngl", "99", "-ot", a.host_layers,
           "-ctk", ctk, "-ctv", ctv, "-t", "16", "--kl-divergence-base", base]
    if n > 0 or a.reuse_base:
        cmd.append("--kl-divergence")
    print("ARM", n, arm, opt, " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    with open(log, "wb") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    txt = open(log, "rb").read().decode("utf-8", "replace").replace("\x00", "")
    print(f"  exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    got = {}
    for line in txt.splitlines():
        for k in KEYS:
            if k in line and k not in got:
                got[k] = line.strip()
    for k in KEYS:
        if k in got:
            print("   ", got[k], flush=True)
    rows.append((arm, r.returncode, got))
    time.sleep(3)
print("KL-PPL DONE", flush=True)
