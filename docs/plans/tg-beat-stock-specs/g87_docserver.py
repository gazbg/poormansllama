"""G24 baseline reproduction: the OWNER'S llama-server, started with the exact flags of docs/llamacpp 8gb vram 262k base.md, driven with a chat request of the document's shape.

    powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g87-doc -Command "python docs/plans/tg-beat-stock-specs/g87_docserver.py nat-B"

What the document ran (2026-10-10): C:\\Development\\AI\\llamacppCUDA\\llama-server.exe (build 10892, e5a8d439c), model Qwen3.6-35B-A3B-MTP-UD-Q4_K_S, q8_0 KV, flash attention,
--parallel 1, --batch-size 8192, --ubatch-size 512, -c 262000, --jinja, -lv 4, --fit-target 16000; its 839-token chat request generated 918 tokens at 30.68 tok/s
(eval time 29,891 ms; 29.56 tok/s and 28.05 tok/s for the two shorter requests before it).
This script changes exactly one thing: --host 127.0.0.1 instead of 0.0.0.0 (no network exposure; no effect on the decode). It records the fit decision lines of the server log
(layers on the GPU, MoE layers overflowing to the CPU, projected free memory), the `timings` object of every response (predicted_per_second is the server's own TG rate,
prompt processing excluded), and the free device memory every 100 ms from nvidia-smi (poller unit; the server's CUDA-runtime-view free memory is printed in its log).
Requests: (1) a short warm-up, (2) and (3) the prompt <name> as ONE user message with the model's default (sampled) settings = the document's conditions,
(4) the same message with temperature 0 / top_k 1 (greedy, the setting of the other G24 arms). Output: build/llama-cuda/g87-doc/{server.log,smi.csv,summary.json}.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mk_g21  # noqa: E402

ROOT = mk_g21.root
SERVER = r"C:\Development\AI\llamacppCUDA\llama-server.exe"
MODEL = r"c:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
PROMPTS = os.path.join(ROOT, "build", "llama-cuda", "gplr07-qa-prompts")
OUT = os.path.join(ROOT, "build", "llama-cuda", "g87-doc")
PORT = 8033
N_GEN = 918


def post(path, obj, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(obj).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, time.time() - t0


def poll_smi(stop, path):
    with open(path, "w") as f:
        f.write("t_s,used_mib,free_mib\n")
        t0 = time.time()
        while not stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try:
                used, free = [x.strip() for x in r.stdout.strip().split(",")]
                f.write("%.3f,%s,%s\n" % (time.time() - t0, used, free))
                f.flush()
            except Exception:
                pass
            stop.wait(0.1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nat-B"
    text = open(os.path.join(PROMPTS, name + ".txt"), "rb").read().decode("utf-8")
    m = re.match(r"<\|im_start\|>user\n(.*)<\|im_end\|>\n<\|im_start\|>assistant\n<think>$", text, re.S)
    if not m:
        raise SystemExit("prompt %s is not a chat-templated prompt" % name)
    user = m.group(1)
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "server.log"), "wb")
    cmd = [SERVER, "--host", "127.0.0.1", "--port", str(PORT), "--model", MODEL, "-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on", "--parallel", "1",
           "--batch-size", "8192", "--ubatch-size", "512", "--alias", "Qwen3.6-35B-A3B-MTP-UD-Q4_K_S", "--jinja", "-c", "262000", "--log-prefix", "--log-timestamps",
           "-lv", "4", "--fit-target", "16000"]
    stop = threading.Event()
    th = threading.Thread(target=poll_smi, args=(stop, os.path.join(OUT, "smi.csv")), daemon=True)
    th.start()
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    import hashlib
    sdir = os.path.dirname(SERVER)
    summary = {"cmd": cmd, "requests": [], "server_sha256_16": hashlib.sha256(open(SERVER, "rb").read()).hexdigest()[:16],
               "server_dir_files": {n: os.path.getsize(os.path.join(sdir, n)) for n in sorted(os.listdir(sdir)) if n.lower().endswith((".exe", ".dll"))}}
    try:
        ready = False
        while time.time() - t_start < 600:
            if proc.poll() is not None:
                raise SystemExit("server exited early rc=%s" % proc.returncode)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(1.0)
        if not ready:
            raise SystemExit("server not ready in 600 s")
        summary["ready_s"] = round(time.time() - t_start, 1)
        plan = [("warmup", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}),
                ("sampled-1", {"messages": [{"role": "user", "content": user}], "max_tokens": N_GEN}),
                ("sampled-2", {"messages": [{"role": "user", "content": user}], "max_tokens": N_GEN}),
                ("greedy", {"messages": [{"role": "user", "content": user}], "max_tokens": N_GEN, "temperature": 0, "top_k": 1})]
        for label, body in plan:
            resp, wall = post("/v1/chat/completions", body)
            t = resp.get("timings", {})
            rec = {"label": label, "wall_s": round(wall, 1), "usage": resp.get("usage"), "timings": {k: t.get(k) for k in
                   ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second")}}
            summary["requests"].append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        stop.set()
        th.join(timeout=5)
        log.close()
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
