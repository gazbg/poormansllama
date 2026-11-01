#!/usr/bin/env python3
"""Plan H94, milestones M2.0 / M2.1 / M2.2: the two-GPU prefill (sequence split) against the same server without it.

  python3 -I tools/pascal/split_check.py --arm ref|peer [--sizes 4096,8192,16384] [--env K=V ...] [--add "--flag v"] [--frac 0.5]
  python3 -I tools/pascal/split_check.py --compare ref peer

An arm starts the server (the owner's command with --lm-prefill, MTP and the hybrid cache, no second-GPU expert tier; the peer arm adds --lm-prefill-peer-gpu 1), sends one cold prompt per size
(token ids cut from build/ctx100k/base.txt, cache_prompt false, greedy, 32 tokens, the top 5 logprobs of every token) and writes build/pascal/split/ARM.json with the tokens, the logprobs, the
prefill time and the server's pass lines. --compare checks that the generated tokens and every logprob are bit-identical (the acceptance of H94 (i)) and prints the speed-up of the prefill.
Refuses to start while another compute process is on the GPUs."""
import argparse, json, os, shlex, signal, subprocess, sys, threading, time, urllib.error, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
OUT = os.path.join(REPO, "build/pascal/split")
BASE = ("-dev CUDA0 -sm none -mg 0 -fa on -np 1 -c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill "
        "--lm-prefill-ubatch 2048 --lm-prefill-segment 16384 -b 16384 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 "
        "--fattn-head-tile --lm-prefill-attn-int8 --spec-type draft-mtp --spec-draft-p-min 0.7 --spec-draft-n-min 0 "
        "--spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-mtp-defer -fitt 512 -lv 3")
TEXT = os.path.join(REPO, "build/ctx100k/base.txt")
if not os.path.exists(TEXT):
    TEXT = os.path.expanduser("~/Development/poormansllama/build/ctx100k/base.txt")


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def busy():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def run_arm(a):
    os.makedirs(OUT, exist_ok=True)
    if busy():
        sys.exit(f"GPU busy, refusing: {busy()}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", LD_LIBRARY_PATH=os.path.abspath(a.bin), LLAMA_MOE_HYBRID_THREADS="6", LLAMA_LM_LOG="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    if a.frac:
        env["LLAMA_LM_PEER_FRAC"] = a.frac
    extra = " ".join(a.add)
    base = BASE
    if a.nospec:   # no MTP draft: for the experiments that --lm-prefill-fuse refuses to run with
        for frag in ("--spec-type draft-mtp ", "--spec-draft-p-min 0.7 ", "--spec-draft-n-min 0 ", "--spec-draft-n-max 3 ", "-ctkd q8_0 ", "-ctvd q8_0 ", "--spec-mtp-defer "):
            base = base.replace(frag, "")
    if a.arm == "peer":
        extra += f" --lm-prefill-peer-gpu 1 --lm-prefill-peer-ctx {a.peer_ctx}" + (f" --lm-prefill-peer-min {a.peer_min}" if a.peer_min else "")
    cmd = [os.path.join(a.bin, "llama-server"), "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(base + " " + extra)
    logp = os.path.join(OUT, a.arm + ".server.log")
    log = open(logp, "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    peak = [0, 0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
                peak[0], peak[1] = max(peak[0], int(v[0])), max(peak[1], int(v[1]))
            except Exception:
                pass
            stop.wait(0.5)

    threading.Thread(target=poll, daemon=True).start()
    res = {"arm": a.arm, "cmd": " ".join(cmd), "runs": []}
    try:
        t0 = time.time()
        for _ in range(1800):
            if srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}, see {logp}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        res["load_s"] = round(time.time() - t0, 1)
        toks = post(a.port, "/tokenize", {"content": open(TEXT, encoding="utf-8").read()})["tokens"]
        post(a.port, "/completion", {"prompt": toks[-3000:], "n_predict": 1, "temperature": 0, "cache_prompt": False})   # warm-up, below the split threshold
        for n in [int(x) for x in a.sizes.split(",") if x]:
            for rep in range(a.reps):
                t_req = time.time()
                try:
                    r = post(a.port, "/completion", {"prompt": toks[:n], "n_predict": 32, "temperature": 0, "cache_prompt": a.cache, "n_probs": 5, "seed": 1})
                except urllib.error.HTTPError as e:
                    if len(res["runs"]) + 0 >= 0 and a.tolerate > 0:
                        a.tolerate -= 1
                        res["runs"].append({"n": n, "rep": rep, "error": e.code})
                        print(f"{a.arm} {n}: the request failed with HTTP {e.code} (expected)", flush=True)
                        continue
                    raise
                wall = time.time() - t_req
                cp = r.get("completion_probabilities", [])
                res["runs"].append({
                    "n": r["timings"]["prompt_n"], "rep": rep, "prompt_ms": r["timings"]["prompt_ms"], "wall_s": round(wall, 2),
                    "tokens": [c.get("token") for c in cp],
                    "logprobs": [[(t.get("token"), t.get("logprob")) for t in (c.get("top_logprobs") or [])] for c in cp],
                })
                mem = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()]
                res["runs"][-1]["mem_mib"] = mem
                print(f"{a.arm} {n}: prefill {r['timings']['prompt_ms']/1000:.2f} s ({r['timings']['prompt_per_second']:.0f} tok/s), GPU memory {mem[0]} / {mem[1]} MiB", flush=True)
    finally:
        stop.set()
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(90)
        except subprocess.TimeoutExpired:
            srv.kill()
    res["peak_mib"] = peak
    txt = open(logp, errors="replace").read()
    res["pass_lines"] = [l.split(" ", 2)[-1][:300] for l in txt.splitlines() if "lm-prefill: pass" in l]
    res["error_lines"] = [l[:300] for l in txt.splitlines() if any(w in l.lower() for w in ("cuda error", "failed", "abort", "the split pass", "differs"))][:10]
    json.dump(res, open(os.path.join(OUT, a.arm + ".json"), "w"))
    print(f"peak memory GPU0 {peak[0]} MiB, GPU1 {peak[1]} MiB; split passes in the log: {sum('split' in l for l in res['pass_lines'])}; error lines: {len(res['error_lines'])}")
    for l in res["error_lines"][:5]:
        print("   ", l)


def compare(x, y):
    A = json.load(open(os.path.join(OUT, x + ".json")))
    B = json.load(open(os.path.join(OUT, y + ".json")))
    ok = True
    for ra, rb in zip(A["runs"], B["runs"]):
        if "error" in ra or "error" in rb:
            print(f"n {ra['n']:6d} rep {ra['rep']}: request failed on one side ({ra.get('error')} / {rb.get('error')}), skipped")
            continue
        same_tok = ra["tokens"] == rb["tokens"]
        same_lp = ra["logprobs"] == rb["logprobs"]
        first_div = next((i for i, (p, q) in enumerate(zip(ra["tokens"], rb["tokens"])) if p != q), None)
        sp = ra["prompt_ms"] / rb["prompt_ms"]
        ok = ok and same_tok and same_lp
        print(f"n {ra['n']:6d} rep {ra['rep']}: tokens {'identical' if same_tok else 'DIFFER at ' + str(first_div)}, top-5 logprobs {'bit-identical' if same_lp else 'DIFFER'}; "
              f"prefill {ra['prompt_ms']/1000:.2f} s -> {rb['prompt_ms']/1000:.2f} s  (x{sp:.2f})")
    print("RESULT:", "IDENTICAL" if ok else "DIFFERENT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ref", "peer"])
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--sizes", default="4096,8192,16384")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--frac", default="")
    ap.add_argument("--peer-ctx", dest="peer_ctx", type=int, default=33024)
    ap.add_argument("--peer-min", dest="peer_min", type=int, default=0, help="--lm-prefill-peer-min; 0 = the server's default")
    ap.add_argument("--port", type=int, default=8194)
    ap.add_argument("--cache", action="store_true", help="send cache_prompt true: a longer prompt that extends the one before continues from the host cache")
    ap.add_argument("--nospec", action="store_true", help="drop the MTP draft flags of the baseline")
    ap.add_argument("--tolerate", type=int, default=0, help="how many failed requests are expected (fault injection)")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    elif a.arm:
        run_arm(a)
    else:
        ap.error("--arm or --compare")
