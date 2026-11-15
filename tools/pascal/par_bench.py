#!/usr/bin/env python3
"""Parallel-request bench for the two GTX 1070 box: one llama-server arm, solo and N-at-a-time measurements.

  python3 -I tools/pascal/par_bench.py --label NAME --flags "<server flags>" [--env K=V ...] [--bin DIR]
        [--np-list 1,2,4] [--tg 192] [--pp 4096,16384] [--par-pp 8192] [--deep 0] [--gpus 0,1]

For every n in --np-list (n must be <= the server's -np): n simultaneous chats (decode rate per request and
summed), n simultaneous cold prompts of --par-pp tokens each (distinct text; prefill rate per request, wall
time, decode rate after the prompt). With n = 1 also the cold prefill sizes of --pp. Records the peak memory
of both GPUs. Writes build/pascal/par/NAME/results.jsonl and prints a table. Refuses a busy GPU.
"""
import argparse, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen-3.6/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf")
TEXT_SRC = "docs/plans/generic-prefill-layer-residency-archive.md"   # read from HEAD, never the work tree
CHATS = [
    "Write a Python function that parses an ISO 8601 duration string such as P3DT4H5M6S into a number of seconds, with tests.",
    "Explain how a CPU cache hierarchy works, and why a matrix multiplication is usually tiled for it.",
    "A train leaves at 09:40 and travels 312 km at an average of 78 km/h, then waits 25 minutes and covers 96 km at 64 km/h. When does it arrive? Show the steps.",
    "List ten practical tips for keeping a long-running Linux server secure, with one sentence of explanation for each.",
    "Explain the difference between a mutex and a semaphore, then give a short C++ example that uses std::mutex to protect a counter incremented by four threads.",
    "Write a SQL query that returns, for each customer, the total amount of their three most recent orders, given tables customers(id, name) and orders(id, customer_id, amount, created_at). Explain the query.",
    "Describe how TCP congestion control works, from slow start to fast recovery, in about three hundred words.",
    "Write a Rust function that merges two sorted vectors of integers into one sorted vector, with a unit test.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--flags", required=True, help="server flags after --model/--host/--port")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
ap.add_argument("--model", default=MODEL)
ap.add_argument("--np-list", default="1,2")
ap.add_argument("--tg", type=int, default=192)
ap.add_argument("--pp", default="4096,16384")
ap.add_argument("--par-pp", type=int, default=8192)
ap.add_argument("--deep", type=int, default=0, help="solo: decode after a prompt of this many tokens")
ap.add_argument("--fill", default="", help="capacity test: N1,N2,.. tokens; request i holds a prompt of Ni tokens, all at once, then 64 tokens each")
ap.add_argument("--gpus", default="0,1")
ap.add_argument("--skip-chat", action="store_true", help="no simultaneous-chat rows")
ap.add_argument("--check-solo", action="store_true", help="after n simultaneous prompts send each one alone and compare the continuation text")
ap.add_argument("--stagger", type=float, default=0.0, help="seconds between the starts of the simultaneous prompts")
ap.add_argument("--mixed", type=int, default=0, help="one chat of this many tokens decoding while one cold --par-pp prompt arrives 3 s later")
ap.add_argument("--port", type=int, default=8195)
ap.add_argument("--out", default=os.path.join(REPO, "build/pascal/par"))
ap.add_argument("--timeout", type=int, default=3600)
a = ap.parse_args()
out = os.path.join(a.out, a.label)
os.makedirs(out, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout or a.timeout) as r:
        return json.loads(r.read())


def smi(q):
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip():
    sys.exit("GPU busy, refusing")
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, LD_LIBRARY_PATH=os.path.abspath(a.bin))
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = [os.path.join(a.bin, "llama-server"), "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(a.flags)
json.dump({"cmd": cmd, "env": a.env, "gpus": a.gpus}, open(os.path.join(out, "cmd.json"), "w"), indent=1)
log = open(os.path.join(out, "server.log"), "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
peak = [0, 0]
stop = threading.Event()


def poll():
    while not stop.is_set():
        try:
            v = [int(x) for x in smi("memory.used")[:2]]
            for i, x in enumerate(v):
                peak[i] = max(peak[i], x)
        except Exception:
            pass
        stop.wait(0.5)


threading.Thread(target=poll, daemon=True).start()
res = open(os.path.join(out, "results.jsonl"), "w")
rows = []


def rec(**kw):
    kw["gpu_peak_mib"] = list(peak)
    res.write(json.dumps(kw) + "\n")
    res.flush()
    rows.append(kw)


def together(bodies, path, delays=None):
    """send all bodies at once (or body i after delays[i] seconds); returns (answers, wall seconds). A body may be (path, body)."""
    ans = [None] * len(bodies)

    def one(i):
        try:
            if delays:
                time.sleep(delays[i])
            b = bodies[i]
            ans[i] = post(b[0], b[1]) if isinstance(b, tuple) else post(path, b)
        except Exception as e:
            ans[i] = {"error": repr(e)}

    th = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    return ans, time.time() - t0


def summarize(kind, n, ans, wall, extra=None):
    ok = [x for x in ans if x and "timings" in x]
    tm = [x["timings"] for x in ok]
    d = dict(kind=kind, n=n, ok=len(ok), wall_s=round(wall, 2),
             pp_n=[t.get("prompt_n") for t in tm], pp_tps=[round(t.get("prompt_per_second") or 0, 1) for t in tm],
             tg_n=[t.get("predicted_n") for t in tm], tg_tps=[round(t.get("predicted_per_second") or 0, 2) for t in tm],
             errors=[x.get("error") for x in ans if x and "error" in x])
    d["tg_sum"] = round(sum(d["tg_tps"]), 2)
    d["pp_wall_tps"] = round(sum(x or 0 for x in d["pp_n"]) / max(1e-9, max([(t.get("prompt_ms") or 0) for t in tm] or [0]) / 1000), 1) if tm else 0
    if extra:
        d.update(extra)
    rec(**d)
    print(f"{kind:10s} n={n} ok={d['ok']}/{n} wall={d['wall_s']:8.2f}s  PP/req={d['pp_tps']} PPsum(wall)={d['pp_wall_tps']}  TG/req={d['tg_tps']} TGsum={d['tg_sum']}  gpu_peak={peak}", flush=True)
    for e in d["errors"]:
        print("   ERROR", e, flush=True)


try:
    for _ in range(1800):
        if srv.poll() is not None:
            sys.exit(f"server exited with {srv.returncode}, see {out}/server.log")
        try:
            if urllib.request.urlopen(URL + "/health", timeout=2).status == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    text = subprocess.run(["git", "-C", REPO, "show", "HEAD:" + TEXT_SRC], capture_output=True, text=True).stdout.replace("\\", "/")
    need = max([a.par_pp * 4] + [int(x) for x in a.pp.split(",") if x] + [a.deep] + ([sum(int(x) for x in a.fill.split(","))] if a.fill else []))
    toks = []
    pos = 0
    while len(toks) < need + 64:
        if pos >= len(text):
            pos = 0
        toks += post("/tokenize", {"content": text[pos:pos + 400000]})["tokens"]
        pos += 400000
    q = post("/tokenize", {"content": "\n\nQuestion: summarise the text above in five bullet points.\n\nAnswer:"})["tokens"]

    def chat(i, n_tok):
        return {"messages": [{"role": "user", "content": CHATS[i % len(CHATS)]}], "max_tokens": n_tok, "temperature": 0,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}

    def prompt(off, n, n_pred):
        return {"prompt": toks[off:off + n - len(q)] + q, "n_predict": n_pred, "temperature": 0, "cache_prompt": False}

    post("/v1/chat/completions", chat(7, 16))   # warm-up
    for n in [int(x) for x in a.np_list.split(",") if x]:
        for rep in range(0 if a.skip_chat else 2):
            ans, wall = together([chat(i + rep * n, a.tg) for i in range(n)], "/v1/chat/completions")
            summarize("chat", n, ans, wall, {"rep": rep})
        if n == 1:
            for s in [int(x) for x in a.pp.split(",") if x]:
                ans, wall = together([prompt(0, s, 32)], "/completion")
                summarize(f"pp{s}", 1, ans, wall)
            if a.deep:
                ans, wall = together([prompt(1000, a.deep, a.tg)], "/completion")
                summarize(f"deep{a.deep}", 1, ans, wall)
        else:
            bodies = [prompt(i * a.par_pp, a.par_pp, 96) for i in range(n)]
            ans, wall = together(bodies, "/completion", [i * a.stagger for i in range(n)] if a.stagger else None)
            summarize(f"parpp{a.par_pp}", n, ans, wall)
            if a.check_solo:
                same = []
                for i in range(n):
                    solo, w1 = together([bodies[i]], "/completion")
                    summarize(f"solo{i}", 1, solo, w1)
                    ta, tb = (ans[i] or {}).get("content"), (solo[0] or {}).get("content")
                    k = 0
                    while ta and tb and k < min(len(ta), len(tb)) and ta[k] == tb[k]:
                        k += 1
                    same.append(ta is not None and ta == tb)
                    print(f"   check-solo n={n} req {i}: {'IDENTICAL' if same[-1] else 'DIFFERENT'} (common prefix {k} of {len(ta or '')} chars)", flush=True)
                    if not same[-1]:
                        print("      par :", repr((ta or "")[max(0, k - 40):k + 80]), "\n      solo:", repr((tb or "")[max(0, k - 40):k + 80]), flush=True)
                rec(kind="check_solo", n=n, identical=same)
    if a.mixed:
        ans, wall = together([("/v1/chat/completions", chat(0, a.mixed)), ("/completion", prompt(5 * a.par_pp % 1000, a.par_pp, 96))], None, [0, 3.0])
        summarize(f"mixed{a.par_pp}", 2, ans, wall)
    if a.fill:
        sizes = [int(x) for x in a.fill.split(",")]
        offs = [sum(sizes[:i]) % max(1, len(toks) - max(sizes)) for i in range(len(sizes))]
        ans, wall = together([prompt(offs[i], sizes[i], 64) for i in range(len(sizes))], "/completion")
        summarize("fill" + "+".join(str(s) for s in sizes), len(sizes), ans, wall)
        for x in ans:
            if x and "content" in x:
                print("   text:", x["content"][:160].replace("\n", " | "))
finally:
    stop.set()
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(timeout=60)
    except Exception:
        srv.kill()
    json.dump({"rows": rows, "gpu_peak_mib": peak}, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("gpu peak MiB:", peak, " out:", out)
