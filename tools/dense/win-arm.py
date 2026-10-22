#!/usr/bin/env python3
"""One server arm on Windows (RTX 5090 Laptop, dense Qwen3.8-27B): start llama-server, wait for /health, run warm-up + decode probes,
poll the VRAM, print one JSON line per probe and a summary, stop the server.  Run it UNDER tools/with-gpu-lock.ps1 (it takes no lock itself).

  python tools/dense/win-arm.py --label a0 --ctx 65536 --flags "--lm-prefill-host-layers 0,2,4" --reps 2 --tg 64

Server flags that every arm shares (q8_0 KV, fa, -np 1, -t 16, greedy probes) are set here; --flags adds the arm's own.
Logs: build/dense/win/<label>.server.log, <label>.vram.csv, <label>.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXE = os.environ.get("DENSE_EXE") or os.path.join(ROOT, "build", "llama-cuda", "bin", "Release", "llama-server.exe")      # DENSE_EXE: another build directory
MODEL = {"q6": r"C:\Models\ALL\other\qwen-3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf", "q4": r"C:\Models\ALL\other\qwen-3.8\Qwen3.8-27B-Q4_K_M.gguf"}
sys.path.insert(0, os.path.join(ROOT, "tools"))
import dense_probe  # noqa: E402


def smi_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def gpu_proc(pid):
    """WDDM per-process GPU memory of the server (MiB): dedicated (on the card), shared (system memory the GPU can use: page-locked host layers AND anything the driver evicted from the card),
    total committed. One CIM query, ~0.7 s."""
    ps = ("Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object { $_.Name -like 'pid_%d_*' } | "
          "ForEach-Object { '{0},{1},{2},{3}' -f $_.DedicatedUsage, $_.SharedUsage, $_.NonLocalUsage, $_.TotalCommitted }" % pid)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20).stdout.split()
        rows = [[int(x) / 1048576.0 for x in o.split(",")] for o in out]
        return [sum(r[i] for r in rows) for i in range(4)] if rows else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", choices=["q6", "q4"], default="q6")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--model-path", default="", help="a GGUF file instead of --model (the MoE model, for example)")
    ap.add_argument("--prompt-file", default="", help="a text file used as the prompt of the prefill and decode probes (instead of the synthetic filler)")
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--flags", default="", help="the arm's own server flags")
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--tg", type=int, default=50, help="tokens per decode probe")
    ap.add_argument("--reps", type=int, default=2, help="decode probes after the warm-up")
    ap.add_argument("--chars", type=int, default=29000, help="prompt size of the decode probes in characters (29000 = about 10k tokens)")
    ap.add_argument("--pp-chars", type=int, default=0, help="also one cold prompt of this many characters (prefill rate)")
    ap.add_argument("--pp-tokens", type=int, default=4, help="tokens generated after the prefill probe")
    ap.add_argument("--health-timeout", type=int, default=900)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--env", action="append", default=[], help="NAME=VALUE for the server process")
    ap.add_argument("--exe", default="", help="llama-server.exe of this arm (default: DENSE_EXE or build/llama-cuda)")
    ap.add_argument("--gpuproc", action="store_true", help="also sample the server's WDDM dedicated / shared GPU memory every ~2 s (<label>.gpuproc.csv)")
    ap.add_argument("--nsys", default="", help="output base path: run the server under Nsight Systems and capture ONLY the first cached decode probe")
    a = ap.parse_args()
    nsys_exe = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"

    out_dir = os.path.join(ROOT, "build", "dense", "win")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, a.label + ".server.log")
    cmd = [a.exe or EXE, "--model", (a.model_path or MODEL[a.model]), "--host", "127.0.0.1", "--port", str(a.port), "-fa", "on", "-np", "1", "-c", str(a.ctx),
           "-ctk", a.kv, "-ctv", a.kv, "-t", str(a.threads), "--no-warmup"] + a.flags.split()
    env = dict(os.environ)
    env["LLAMA_LM_LOG"] = "1"
    for e in a.env:
        k, v = e.split("=", 1)
        env[k] = v
    if a.nsys:
        subprocess.run([nsys_exe, "shutdown", "--session=dsr", "--kill=sigkill"], capture_output=True)
        cmd = [nsys_exe, "launch", "--session-new=dsr", "--trace=cuda"] + cmd
    print("CMD", " ".join(cmd), flush=True)
    result = {"label": a.label, "cmd": " ".join(cmd), "probes": []}
    base_used = smi_used()
    with open(log, "wb") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    stop = threading.Event()
    samples = []

    def poll():
        while not stop.is_set():
            samples.append((time.time(), smi_used()))
            stop.wait(0.5)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    gp_samples = []

    def poll_proc():
        while not stop.is_set():
            g = gpu_proc(srv.pid)
            if g:
                gp_samples.append((time.time(), *g))
            stop.wait(1.0)
    th2 = threading.Thread(target=poll_proc, daemon=True) if a.gpuproc else None
    if th2:
        th2.start()
    ok = False
    try:
        t0 = time.time()
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
        result["load_s"] = round(time.time() - t0, 1)
        if ok:
            def probe(label, chars, n, ignore_eos=True, cache=False):
                prompt = open(a.prompt_file, encoding="utf-8", errors="replace").read() if (a.prompt_file and label != "warm") else dense_probe.build_prompt(chars, 0.5)
                body = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": cache, "ignore_eos": ignore_eos}
                d = dense_probe.post(a.port, body, timeout=3000)
                t = d["timings"]
                row = {"label": label, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n", 0), "pp_tps": round(t["prompt_per_second"], 2), "tg_n": t["predicted_n"],
                       "tg_tps": round(t["predicted_per_second"], 3), "tg_ms_per_tok": round(t["predicted_ms"] / max(1, t["predicted_n"]), 1),
                       "needle": "7391" in d["content"], "sha": hashlib.sha1(d["content"].encode()).hexdigest()[:10]}
                if "draft_n" in t:
                    row["draft_n"] = t.get("draft_n")
                    row["draft_accepted"] = t.get("draft_n_accepted")
                result["probes"].append(row)
                print(json.dumps(row), flush=True)
            probe("warm", 400, 2)
            probe("prefill", a.chars, 2)                      # cold: the decode probes below reuse this prompt from the cache
            for i in range(a.reps):
                cap = bool(a.nsys) and i == 0
                if cap:
                    r = subprocess.run([nsys_exe, "start", "--session=dsr", "--output=" + os.path.join(ROOT, a.nsys), "--force-overwrite=true"], capture_output=True, text=True)
                    print("nsys start:", r.returncode, r.stdout.strip()[:200], r.stderr.strip()[:200], flush=True)
                probe(f"tg{i + 1}", a.chars, a.tg, cache=True)
                if cap:
                    r = subprocess.run([nsys_exe, "stop", "--session=dsr"], capture_output=True, text=True)
                    print("nsys stop:", r.returncode, r.stdout.strip()[-300:], r.stderr.strip()[:200], flush=True)
            if a.pp_chars:
                probe("pp", a.pp_chars, a.pp_tokens)
                probe("tg_after_pp", a.chars, a.tg, cache=True)
    except Exception as e:
        print("ERROR", repr(e), flush=True)
        result["error"] = repr(e)
    finally:
        stop.set()
        th.join(timeout=3)
        if th2:
            th2.join(timeout=5)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=30)
        except Exception:
            pass
    used = [u for _, u in samples if u >= 0]
    if gp_samples:
        t0g = gp_samples[0][0]
        with open(os.path.join(out_dir, a.label + ".gpuproc.csv"), "w") as f:
            f.write("t_s,dedicated_mib,shared_mib,nonlocal_mib,committed_mib\n")
            for r in gp_samples:
                f.write("%.1f,%.0f,%.0f,%.0f,%.0f\n" % (r[0] - t0g, r[1], r[2], r[3], r[4]))
        result["gpuproc_max_mib"] = {"dedicated": round(max(r[1] for r in gp_samples)), "shared": round(max(r[2] for r in gp_samples)),
                                     "nonlocal": round(max(r[3] for r in gp_samples)), "committed": round(max(r[4] for r in gp_samples))}
    result["vram_base_mib"] = base_used
    result["vram_peak_mib"] = max(used) if used else -1
    with open(os.path.join(out_dir, a.label + ".vram.csv"), "w") as f:
        for t, u in samples:
            f.write(f"{t:.1f},{u}\n")
    with open(os.path.join(out_dir, a.label + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"SUMMARY {a.label}: load {result['load_s']} s, peak VRAM {result['vram_peak_mib']} MiB (base {base_used}), ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
