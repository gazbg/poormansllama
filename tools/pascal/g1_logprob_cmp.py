#!/usr/bin/env python3
"""Decode-path drift of the second-GPU expert tier (plan H79): greedy generation of the same prompts on a one-GPU server and on a server with --hybrid-moe-gpu1,
then, over the stretch where the two token sequences are still identical, the difference of the chosen token's logprob (same prefix, only the expert placement differs).

  python3 -I tools/pascal/g1_logprob_cmp.py [--n 96] [--bin DIR] [--out build/pascal/g1lp]

Arms: ctl (tuned one-GPU command), g1 (same + --hybrid-moe-gpu1 auto, 4 team threads, 40 swaps), ctl2 (ctl again: the run-to-run floor of the one-GPU path), ctlB8 (ctl with 8 promotions per token:
a different expert placement on the same one GPU). Arms already saved under --out are reused; --arms names the ones to run.
Prompts are raw text (no chat template). GPU 0 (and 1 for the g1 arm); refuses to start while another compute process is on the GPUs."""
import argparse, json, os, shlex, signal, subprocess, sys, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BASE = ("-dev CUDA0 -sm none -mg 0 -fa on -np 1 -c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill "
        "--lm-prefill-ubatch 2048 --lm-prefill-segment 4096 -b 4096 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 "
        "--fattn-head-tile --lm-prefill-attn-int8 --spec-type draft-mtp --spec-draft-p-min 0.7 --spec-draft-n-min 0 "
        "--spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-mtp-defer -fitt 512")
PROMPTS = [
    "The following is a short Python module that parses ISO 8601 durations into seconds.\n\nimport re\n\ndef parse_duration(s):\n",
    "A CPU cache hierarchy works by keeping recently used data close to the core. The three levels are",
    "Question: A train leaves at 09:40 and travels 312 km at an average of 78 km/h, then waits 25 minutes and covers 96 km at 64 km/h. When does it arrive?\nAnswer: Let us compute step by step.",
    "Translate into French: 'The committee postponed the vote because several members had not received the final draft of the report.'\nFrench:",
    "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.created_at >= '2024-01-01' GROUP BY",
    "Ten practical tips for keeping a long-running Linux server secure:\n1. Keep the system updated.\n2.",
    "Once upon a time, in a lighthouse on the edge of a stormy sea, there lived a keeper named",
    "The difference between a mutex and a semaphore is that a mutex",
]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
ap.add_argument("--out", default=os.path.join(REPO, "build/pascal/g1lp"))
ap.add_argument("--port", type=int, default=8192)
ap.add_argument("--arms", default="ctl,g1,ctl2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def busy():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def run_arm(name):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH=os.path.abspath(a.bin), LLAMA_MOE_HYBRID_THREADS="6")
    extra = ""
    if name == "ctlB8":
        env.update(LLAMA_MOE_HYBRID_B="8")
    if name == "g1":
        env.update(CUDA_VISIBLE_DEVICES="0,1", LLAMA_MOE_HYBRID_THREADS="4", LLAMA_MOE_HYBRID_G1_B="40")
        extra = " --hybrid-moe-gpu1 auto"
    cmd = [os.path.join(a.bin, "llama-server"), "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(BASE + extra)
    log = open(os.path.join(a.out, name + ".log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    res = []
    try:
        for _ in range(900):
            if srv.poll() is not None:
                sys.exit(f"{name}: server exited with {srv.returncode}")
            try:
                if urllib.request.urlopen(URL + "/health", timeout=2).status == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        post("/completion", {"prompt": "Hello", "n_predict": 8, "temperature": 0, "cache_prompt": False})   # warm-up
        for p in PROMPTS:
            js = post("/completion", {"prompt": p, "n_predict": a.n, "temperature": 0, "cache_prompt": False, "n_probs": 3, "seed": 1})
            toks = []
            for c in js.get("completion_probabilities", []):
                top = c.get("top_logprobs") or []
                toks.append({"t": c.get("token"), "lp": c.get("logprob"), "top": [(x.get("token"), x.get("logprob")) for x in top]})
            res.append(toks)
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
    json.dump(res, open(os.path.join(a.out, name + ".json"), "w"))
    return res


if busy():
    sys.exit(f"GPU busy, refusing: {busy()}")
arms = {}
for n in ("ctl", "g1", "ctl2", "ctlB8"):
    f = os.path.join(a.out, n + ".json")
    if n in a.arms.split(","):
        arms[n] = run_arm(n)
    elif os.path.exists(f):
        arms[n] = json.load(open(f))


def compare(x, y, label):
    print(f"-- {label}")
    all_d = []
    for i, (tx, ty) in enumerate(zip(x, y)):
        k = 0
        while k < min(len(tx), len(ty)) and tx[k]["t"] == ty[k]["t"]:
            k += 1
        d = [abs(tx[j]["lp"] - ty[j]["lp"]) for j in range(k) if tx[j]["lp"] is not None and ty[j]["lp"] is not None]
        all_d += d
        mean = sum(d) / len(d) if d else 0.0
        mx = max(d) if d else 0.0
        note = ""
        if k < min(len(tx), len(ty)):
            top = sorted(tx[k]["top"], key=lambda z: -z[1])
            margin = (top[0][1] - top[1][1]) if len(top) > 1 else float("nan")
            note = f"diverges at token {k} (margin of the first arm's top-2 there: {margin:.3f} nats)"
        else:
            note = f"identical over all {k} tokens"
        print(f"   prompt {i}: {k} identical tokens, mean |dlogprob| {mean:.4f}, max {mx:.4f}; {note}")
    if all_d:
        s = sorted(all_d)
        print(f"   all identical-prefix tokens: {len(s)}, mean {sum(s)/len(s):.4f}, median {s[len(s)//2]:.4f}, p95 {s[int(len(s)*0.95)]:.4f}, max {s[-1]:.4f}")


names = list(arms)
if "ctl" in arms and "ctl2" in arms:
    compare(arms["ctl"], arms["ctl2"], "ctl against ctl2 (run-to-run floor of the one-GPU path)")
if "ctl" in arms and "g1" in arms:
    compare(arms["ctl"], arms["g1"], "ctl against g1 (second-GPU tier)")
if "ctl" in arms and "ctlB8" in arms:
    compare(arms["ctl"], arms["ctlB8"], "ctl against ctlB8 (a different residency on one GPU)")
