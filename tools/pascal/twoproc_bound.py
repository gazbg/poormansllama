#!/usr/bin/env python3
"""BOUND only (not a deliverable): two llama-server processes, one per GPU, sharing the mmap'd model; one request to each at the
same time. Shows what 'one slot pinned per card' could give inside one process. python3 -I tools/pascal/twoproc_bound.py"""
import json, os, shlex, signal, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN = os.path.join(REPO, "build/pascal/base-bin")
MODEL = os.path.expanduser("~/models/qwen-3.6/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf")
THREADS = os.environ.get("TP_THREADS", "")
R7 = ("-n 16000 -cram 0 --ctx-checkpoints 16 -c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 "
      "-b 32768 -ub 512 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 --hybrid-moe-prefill-free-all "
      "--lm-prefill-segment 16384 --lm-prefill-attn-int8 --device CUDA0")
CH = ["Write a Python function that parses an ISO 8601 duration string such as P3DT4H5M6S into a number of seconds, with tests.",
      "Explain how a CPU cache hierarchy works, and why a matrix multiplication is usually tiled for it.",
      "List ten practical tips for keeping a long-running Linux server secure, with one sentence of explanation for each.",
      "Explain the difference between a mutex and a semaphore, then give a short C++ example that uses std::mutex to protect a counter incremented by four threads."]
if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip():
    sys.exit("GPU busy")
out = os.path.join(REPO, "build/pascal/par/twoproc" + (("-t" + THREADS) if THREADS else "")); os.makedirs(out, exist_ok=True)
srv = []
for g in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LD_LIBRARY_PATH=BIN, LLAMA_MOE_HYBRID_LEAN_STALL_US="10000", LLAMA_FATTN_GQA8_MAXROWS="4")
    if THREADS:
        env["LLAMA_MOE_HYBRID_THREADS"] = THREADS
        if g == 1:
            env["LLAMA_MOE_HYBRID_LEAN_CPUS"] = os.environ.get("TP_CPUS1", "")
    srv.append(subprocess.Popen([BIN + "/llama-server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(8197 + g)] + shlex.split(R7),
                                stdout=open(f"{out}/server{g}.log", "w"), stderr=subprocess.STDOUT, env=env))
def post(g, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{8197+g}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())
try:
    for g in (0, 1):
        for _ in range(1800):
            if srv[g].poll() is not None: sys.exit(f"server {g} died")
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{8197+g}/health", timeout=2).status == 200: break
            except Exception: pass
            time.sleep(1)
    text = subprocess.run(["git", "-C", REPO, "show", "HEAD:docs/plans/generic-prefill-layer-residency-archive.md"], capture_output=True, text=True).stdout.replace("\\", "/")
    toks = post(0, "/tokenize", {"content": text[:400000]})["tokens"]
    chat = lambda i: {"messages": [{"role": "user", "content": CH[i]}], "max_tokens": 192, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    prm = lambda off: {"prompt": toks[off:off + 8192], "n_predict": 96, "temperature": 0, "cache_prompt": False}
    def both(bodies, path, who=(0, 1)):
        ans = {}
        th = [threading.Thread(target=lambda g=g: ans.__setitem__(g, post(g, path, bodies[g]))) for g in who]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
        tm = [ans[g]["timings"] for g in who]
        print(f"who={who} wall={time.time()-t0:7.2f}s PP/req={[round(t['prompt_per_second'],1) for t in tm]} TG/req={[round(t['predicted_per_second'],2) for t in tm]} TGsum={sum(t['predicted_per_second'] for t in tm):.2f}", flush=True)
    for g in (0, 1): post(g, "/v1/chat/completions", dict(chat(3), max_tokens=16))
    print("solo chat on each (other idle)"); both({0: chat(0)}, "/v1/chat/completions", (0,)); both({1: chat(1)}, "/v1/chat/completions", (1,))
    print("two chats at once"); both({0: chat(0), 1: chat(1)}, "/v1/chat/completions"); both({0: chat(2), 1: chat(3)}, "/v1/chat/completions")
    print("two cold 8192-token prompts at once"); both({0: prm(0), 1: prm(8192)}, "/completion")
    print("mem MiB:", subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split())
    print("host:", subprocess.run("free -m | sed -n 2p; grep -E 'VmRSS|RssAnon' /proc/%d/status /proc/%d/status" % (srv[0].pid, srv[1].pid), shell=True, capture_output=True, text=True).stdout)
finally:
    for s in srv: s.send_signal(signal.SIGINT)
    for s in srv:
        try: s.wait(timeout=60)
        except Exception: s.kill()
