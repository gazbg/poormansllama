#!/usr/bin/env python3
"""One server arm on the GTX 1070 box: the owner's Qwen3.6-35B-A3B baseline command, cold prefill at exact token
counts, greedy decode on short chats and after a long prompt. Prints one line per measurement and writes results.jsonl.

  python3 -I tools/pascal/moe_bench.py --label base [--bin DIR] [--env K=V ...] [--add "--flag v"] [--drop "--flag"]
         [--pp 4096,16384,32768] [--tg 256] [--deep 16384] [--hlog]

Prompt text: build/ctx100k/base.txt of the main checkout (or --text). Prefill prompts are token arrays cut from it,
so a prompt is exactly N tokens. The decode prompts are fixed chats (thinking left at the model default).
GPU 0 only. Refuses to start while another compute process is on the GPUs."""
import argparse, hashlib, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BASE = ("-dev CUDA0 -sm none -mg 0 -fa on -np 1 -c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill "
        "--lm-prefill-ubatch 2048 --lm-prefill-segment 4096 -b 4096 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 "
        "--fattn-head-tile --lm-prefill-attn-int8 --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 "
        "--spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-mtp-defer -lv 4")
CHATS = [
    "Write a Python function that parses an ISO 8601 duration string such as P3DT4H5M6S into a number of seconds, with tests.",
    "Explain how a CPU cache hierarchy works, and why a matrix multiplication is usually tiled for it.",
    "A train leaves at 09:40 and travels 312 km at an average of 78 km/h, then waits 25 minutes and covers 96 km at 64 km/h. When does it arrive? Show the steps.",
]
DEEP_Q = "\n\nQuestion: summarise the text above in five bullet points."

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
ap.add_argument("--model", default=MODEL)
ap.add_argument("--text", default=os.path.join(REPO, "build/ctx100k/base.txt") if os.path.exists(os.path.join(REPO, "build/ctx100k/base.txt"))
                else os.path.expanduser("~/Development/poormansllama/build/ctx100k/base.txt"))
ap.add_argument("--env", action="append", default=[], help="K=V for the server process")
ap.add_argument("--add", action="append", default=[], help="extra server flags")
ap.add_argument("--drop", action="append", default=[], help="a flag of the baseline to remove, with its value if it has one")
ap.add_argument("--pp", default="4096,16384,32768")
ap.add_argument("--tg", type=int, default=256)
ap.add_argument("--deep", type=int, default=16384, help="tokens of context before the deep decode (0 = skip)")
ap.add_argument("--reps", type=int, default=1, help="repeat the short decode chats")
ap.add_argument("--port", type=int, default=8191)
ap.add_argument("--out", default=os.path.join(REPO, "build/pascal"))
ap.add_argument("--hlog", action="store_true", help="LLAMA_MOE_HYBRID_LOG per-token engine log")
ap.add_argument("--allow-busy", action="store_true")
ap.add_argument("--wrap", default="", help="a command prefix for the server, e.g. an nsys profile line")
ap.add_argument("--chats", type=int, default=len(CHATS), help="how many of the short chats to run")
a = ap.parse_args()

out = os.path.join(a.out, a.label)
os.makedirs(out, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}"

def gpu_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()]

def gpu0_used():
    r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])

def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def text_hash(r):
    msg = r["choices"][0]["message"]
    s = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    with open(os.path.join(out, "texts.jsonl"), "a") as f:
        f.write(json.dumps({"reasoning": msg.get("reasoning_content"), "content": msg.get("content")}) + "\n")
    return hashlib.sha256(s.encode()).hexdigest()[:12]

def drop_flags(args, drops):
    toks = shlex.split(args)
    for d in drops:
        dt = shlex.split(d)
        for i in range(len(toks) - len(dt) + 1):
            if toks[i:i + len(dt)] == dt:
                del toks[i:i + len(dt)]
                break
        else:
            sys.exit(f"--drop {d!r}: not in the server command")
    return toks

busy = gpu_apps()
if busy and not a.allow_busy:
    sys.exit(f"GPU busy, refusing: {busy}")

cmd = shlex.split(a.wrap) + [os.path.join(a.bin, "llama-server"), "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port)]
cmd += drop_flags(BASE, a.drop)
for x in a.add:
    cmd += shlex.split(x)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH=os.path.abspath(a.bin))  # the RUNPATH names the build dir: a side copy needs this
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
if a.hlog:
    env["LLAMA_MOE_HYBRID_LOG"] = os.path.join(out, "hybrid.jsonl")
with open(os.path.join(out, "cmd.txt"), "w") as f:
    f.write(" ".join(f"{k}={env[k]}" for k in sorted(set(env) - set(os.environ)) ) + " " + shlex.join(cmd) + "\n")

log = open(os.path.join(out, "server.log"), "w")
t_start = time.time()
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
peak = [0]
stop = threading.Event()
def poll():
    while not stop.is_set():
        try:
            peak[0] = max(peak[0], gpu0_used())
        except Exception:
            pass
        stop.wait(0.5)
threading.Thread(target=poll, daemon=True).start()

results = []
def emit(rec):
    rec["label"] = a.label
    results.append(rec)
    print(json.dumps(rec), flush=True)

try:
    for _ in range(600):
        if srv.poll() is not None:
            sys.exit(f"server exited with {srv.returncode}, see {out}/server.log")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(1)
    libs = ""
    if not a.wrap:
        with open(f"/proc/{srv.pid}/maps") as f:
            libs = sorted({l.split()[-1] for l in f if "libllama" in l or "libggml-cuda.so" in l})
    emit({"what": "load", "s": round(time.time() - t_start, 1), "vram_mib": gpu0_used(), "libs": libs})

    text = open(a.text, encoding="utf-8").read()
    toks = post("/tokenize", {"content": text})["tokens"]
    # warm-up (kernel loading, first pass), not recorded
    post("/completion", {"prompt": toks[-3000:], "n_predict": 1, "temperature": 0, "cache_prompt": False})

    for n in [int(x) for x in a.pp.split(",") if x]:
        if n > len(toks):
            continue
        r = post("/completion", {"prompt": toks[:n], "n_predict": 1, "temperature": 0, "cache_prompt": False})
        t = r["timings"]
        emit({"what": "pp", "n": t["prompt_n"], "s": round(t["prompt_ms"] / 1000, 2), "tok_s": round(t["prompt_per_second"], 1)})

    for rep in range(a.reps):
        for i, c in enumerate(CHATS[:a.chats]):
            r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": c}], "max_tokens": a.tg, "temperature": 0, "seed": 1})
            t = r["timings"]
            emit({"what": "tg", "chat": i, "rep": rep, "prompt_n": t["prompt_n"], "n": t["predicted_n"], "tok_s": round(t["predicted_per_second"], 2),
                  "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "text": text_hash(r)})

    if a.deep:
        ctx = post("/detokenize", {"tokens": toks[:a.deep]})["content"]
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": ctx + DEEP_Q}], "max_tokens": a.tg, "temperature": 0, "seed": 1})
        t = r["timings"]
        emit({"what": "deep", "prompt_n": t["prompt_n"], "pp_tok_s": round(t["prompt_per_second"], 1), "n": t["predicted_n"],
              "tok_s": round(t["predicted_per_second"], 2), "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "text": text_hash(r)})
finally:
    stop.set()
    if srv.poll() is None:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(300 if a.wrap else 60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
    log.close()
    for _ in range(60):
        if gpu0_used() < 300:
            break
        time.sleep(1)
    emit({"what": "end", "peak_vram_mib": peak[0], "rc": srv.returncode, "gpu_after": gpu_apps()})
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
