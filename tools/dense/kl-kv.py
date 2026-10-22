#!/usr/bin/env python3
"""First-token KL of KV cache type pairs against an f16/f16 reference (dense Qwen3.8-27B-UD-Q6_K_XL, RTX 5090 Laptop, Windows).  Run it UNDER tools/with-gpu-lock.ps1.

  python tools/dense/kl-kv.py [--exe build/llama-cuda-faq/bin/Release/llama-server.exe] [--arms f16/f16,f16/f16,q8_0/q8_0,q8_0/q4_0,q4_0/q4_0]

Every arm is a fresh server with the same placement (the first 24 layers on the host, the ordinary prefill path, -ub 512, -c 24576), only -ctk / -ctv differ.  Each arm answers the 24 prompts of
tools/dense_equiv.py --set big (1.5-4.3k tokens, greedy, n_probs 20) and then the owner's 20.7k-token web page (tests/WebPageTest 10kcontext.txt), one generated token each.  KL(reference || test) is taken
over the reference's top-20 tokens of the FIRST generated token, as in tools/dense/kl-study.sh.  The first arm is the reference; a second f16/f16 arm measures the run-to-run noise floor.
Results: build/dense/kv-<n>-<ctk>-<ctv>.json, server logs build/dense/win/kv-<n>.server.log.
"""
import argparse, json, math, os, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=os.path.join(ROOT, "build", "llama-cuda-faq", "bin", "Release", "llama-server.exe"))
ap.add_argument("--model", default=r"C:\Models\ALL\other\qwen-3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf")
ap.add_argument("--arms", default="f16/f16,f16/f16,q8_0/q8_0,q8_0/q4_0,q4_0/q4_0")
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--n-prompts", type=int, default=24)
ap.add_argument("--host-layers", default="0-23")
ap.add_argument("--page", default=os.path.join(ROOT, "tests", "WebPageTest 10kcontext.txt"))
a, _ = ap.parse_known_args()      # the GPU wrapper passes a stray "*" (from the "*>" redirect): ignored

out_dir = os.path.join(ROOT, "build", "dense")
os.makedirs(os.path.join(out_dir, "win"), exist_ok=True)


def post(body, timeout=3000):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_entry(d, i, prompt_n):
    cp = (d.get("completion_probabilities") or [{}])[0]
    top = cp.get("top_logprobs") or []
    return {"i": i, "prompt_n": prompt_n, "first_top": [(x.get("id"), x.get("logprob")) for x in top], "tokens": [{"id": cp.get("id"), "lp": cp.get("logprob")}], "text": d.get("content", "")}


def kl_one(r, t):
    pr = {i: math.exp(lp) for i, lp in r["first_top"]}
    pt = {i: math.exp(lp) for i, lp in t["first_top"]}
    floor = min(pt.values()) if pt else 1e-9
    z = sum(pr.values())
    return max(sum((pr[i] / z) * math.log((pr[i] / z) / (pt.get(i, floor) / z)) for i in pr), 0.0)


results = {}
arms = a.arms.split(",")
for n, arm in enumerate(arms):
    ctk, ctv = arm.split("/")
    label = f"{n}-{ctk}-{ctv}"
    log = os.path.join(out_dir, "win", f"kv-{n}.server.log")
    cmd = [a.exe, "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port), "-fa", "on", "-np", "1", "-c", "24576", "-ctk", ctk, "-ctv", ctv, "-ub", "512", "-t", "16", "--no-warmup",
           "--ctx-checkpoints", "0", "--lm-prefill-host-layers", a.host_layers]
    print("ARM", label, " ".join(cmd[1:]), flush=True)
    env = dict(os.environ); env["LLAMA_LM_LOG"] = "1"
    with open(log, "wb") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    ok = False
    t0 = time.time()
    while time.time() - t0 < 600:
        if srv.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(2)
    if not ok:
        print("ARM", label, "FAILED to start", flush=True)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        continue
    out_json = os.path.join(out_dir, f"kv-{label}.json")
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "dense_equiv.py"), "collect", "--port", str(a.port), "--out", out_json, "--n-predict", "1",
                            "--n-prompts", str(a.n_prompts), "--set", "big"], cwd=ROOT)
        data = json.load(open(out_json))
        page = open(a.page, encoding="utf-8", errors="replace").read()
        t1 = time.time()
        d = post({"prompt": page, "n_predict": 1, "temperature": 0, "cache_prompt": False, "n_probs": 20})
        data.append(first_entry(d, len(data), d["timings"]["prompt_n"]))
        print(f"web page: {d['timings']['prompt_n']} tokens in {time.time() - t1:.0f} s, first token {d.get('content', '')!r}", flush=True)
        json.dump(data, open(out_json, "w"))
        results[label] = data
    except Exception as e:
        print("ARM", label, "ERROR", repr(e), flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
        try:
            srv.wait(timeout=60)
        except Exception:
            pass
        time.sleep(5)

labels = list(results)
if labels:
    ref = results[labels[0]]
    print("\n=== first-token KL(reference || arm), reference = " + labels[0])
    for lb in labels[1:]:
        t = results[lb]
        short = [kl_one(r, x) for r, x in zip(ref[:-1], t[:-1])]
        long_kl = kl_one(ref[-1], t[-1])
        agree = sum(r["tokens"][0]["id"] == x["tokens"][0]["id"] for r, x in zip(ref, t))
        s = sorted(short)
        print(f"{lb:18s} 24 short prompts: mean KL {sum(short) / len(short):.5f}  median {s[len(s) // 2]:.5f}  max {max(short):.5f} | web page ({ref[-1]['prompt_n']} tokens): KL {long_kl:.5f} | top-1 agrees {agree}/{len(ref)}")
print("KL-KV DONE", flush=True)
