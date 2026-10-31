#!/usr/bin/env python3
"""Smoke test of the full two-GPU stack with two server slots (-np 2): concurrent chat requests, memory of both GPUs while they run, errors in the server log.

  python3 -I tools/pascal/g1_np2_smoke.py [--rounds 3] [--tokens 192] [--bin DIR] [--out build/pascal/g1np2]

Starts the server like tools/pascal/moe_bench.py does (owner's command without --spec-mtp-defer, which needs -np 1, with --hybrid-moe-gpu1 auto, --mtp-draft-vocab 131072 and the MTP experts in GPU 0), sends --rounds rounds of
two simultaneous chats, prints tokens per second per request and the GPU memory sampled during the rounds. It proves that nothing crashes or hangs; it measures no speed-up.
Refuses to start while another compute process is on the GPUs."""
import argparse, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BASE = ("-dev CUDA0 -sm none -mg 0 -fa on -np 2 -c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill "
        "--lm-prefill-ubatch 2048 --lm-prefill-segment 4096 -b 4096 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 "
        "--fattn-head-tile --lm-prefill-attn-int8 --spec-type draft-mtp --spec-draft-p-min 0.7 --spec-draft-n-min 0 "
        "--spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 -fitt 512 "   # --spec-mtp-defer needs -np 1
        "--hybrid-moe-gpu1 auto --mtp-draft-vocab 131072 -ot blk\\.40\\.ffn_(up|gate|down)_exps\\.weight=CUDA0")
CHATS = [
    "Write a Python function that parses an ISO 8601 duration string such as P3DT4H5M6S into a number of seconds, with tests.",
    "Explain how a CPU cache hierarchy works, and why a matrix multiplication is usually tiled for it.",
    "A train leaves at 09:40 and travels 312 km at an average of 78 km/h, then waits 25 minutes and covers 96 km at 64 km/h. When does it arrive? Show the steps.",
    "List ten practical tips for keeping a long-running Linux server secure, with one sentence of explanation for each.",
    "Explain the difference between a mutex and a semaphore, then give a short C++ example that uses std::mutex to protect a counter incremented by four threads.",
    "Translate into French and explain two grammar choices: 'The committee postponed the vote because several members had not received the final draft of the report.'",
]

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--tokens", type=int, default=192)
ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
ap.add_argument("--out", default=os.path.join(REPO, "build/pascal/g1np2"))
ap.add_argument("--port", type=int, default=8193)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split("\n")


if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip():
    sys.exit("GPU busy, refusing")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", LD_LIBRARY_PATH=os.path.abspath(a.bin), LLAMA_MOE_HYBRID_THREADS="4", LLAMA_MOE_HYBRID_G1_B="40")
cmd = [os.path.join(a.bin, "llama-server"), "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(BASE)
log = open(os.path.join(a.out, "server.log"), "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
peak = [0, 0]
stop = threading.Event()


def poll():
    while not stop.is_set():
        try:
            v = [int(x) for x in smi("memory.used")[:2]]
            peak[0], peak[1] = max(peak[0], v[0]), max(peak[1], v[1])
        except Exception:
            pass
        stop.wait(0.5)


threading.Thread(target=poll, daemon=True).start()
ok = True
try:
    for _ in range(900):
        if srv.poll() is not None:
            sys.exit(f"server exited with {srv.returncode}, see {a.out}/server.log")
        try:
            if urllib.request.urlopen(URL + "/health", timeout=2).status == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    for rnd in range(a.rounds):
        res = [None, None]

        def one(k):
            q = CHATS[(2 * rnd + k) % len(CHATS)]
            t0 = time.time()
            js = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": a.tokens, "temperature": 0, "seed": 1, "id_slot": -1})
            n = js["usage"]["completion_tokens"]
            res[k] = (n, n / (time.time() - t0), (js["choices"][0]["message"].get("content") or js["choices"][0]["message"].get("reasoning_content") or "")[:60].replace("\n", " "))

        th = [threading.Thread(target=one, args=(k,)) for k in range(2)]
        [t.start() for t in th]
        [t.join(600) for t in th]
        for k in range(2):
            if res[k] is None or res[k][0] < 8:
                ok = False
        print(f"round {rnd}: " + " | ".join("failed" if r is None else f"{r[0]} tokens, {r[1]:.1f} tok/s, '{r[2]}'" for r in res), flush=True)
finally:
    stop.set()
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(60)
    except subprocess.TimeoutExpired:
        srv.kill()
txt = open(os.path.join(a.out, "server.log"), errors="replace").read()
bad = [l for l in txt.splitlines() if any(w in l.lower() for w in ("cuda error", "timed out", "unanswered", "abort", "segmentation"))]
print(f"peak memory: GPU 0 {peak[0]} MiB, GPU 1 {peak[1]} MiB; error lines in the server log: {len(bad)}")
for l in bad[:5]:
    print("   ", l[:200])
print("RESULT:", "PASS" if ok and not bad else "FAIL")
