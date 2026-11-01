#!/usr/bin/env python3
"""Plan H96: does the server exit when it is asked to stop in the middle of a generation?

  python3 -I tools/pascal/sigint_check.py --cfg one|tier|peer [--delay 3] [--reps 2]

Starts the server (the owner's command; cfg tier adds the second-GPU tier and its recipe, peer adds the two-GPU prefill with the shrunk tier), sends a chat request with a ~32k-token
context and 512 tokens to generate in a thread, waits until the first token has been streamed plus --delay seconds, sends SIGINT to the server and measures the time until the process
exits (limit 60 s; then SIGKILL and the run counts as a hang). Refuses to start while another compute process is on the GPUs."""
import argparse, http.client, json, os, shlex, signal, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.expanduser("~/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BASE = ("-dev CUDA0 -sm none -mg 0 -fa on -np 1 -c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill "
        "--lm-prefill-ubatch 2048 --lm-prefill-segment 4096 -b 4096 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 "
        "--fattn-head-tile --lm-prefill-attn-int8 --spec-type draft-mtp --spec-draft-p-min 0.7 --spec-draft-n-min 0 "
        "--spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-mtp-defer -fitt 512 -lv 3")
CFG = {
    "one": ("", {"LLAMA_MOE_HYBRID_THREADS": "6", "CUDA_VISIBLE_DEVICES": "0"}),
    "tier": ("--hybrid-moe-gpu1 auto --mtp-draft-vocab 131072 -ot blk\\.40\\.ffn_(up|gate|down)_exps\\.weight=CUDA0", {"LLAMA_MOE_HYBRID_THREADS": "4", "LLAMA_MOE_HYBRID_G1_B": "40", "CUDA_VISIBLE_DEVICES": "0,1"}),
    "peer": ("--hybrid-moe-gpu1 auto --mtp-draft-vocab 131072 -ot blk\\.40\\.ffn_(up|gate|down)_exps\\.weight=CUDA0 --lm-prefill-peer-gpu 1 --lm-prefill-peer-ctx 33024 -b 16384 --lm-prefill-segment 16384",
             {"LLAMA_MOE_HYBRID_THREADS": "4", "LLAMA_MOE_HYBRID_G1_B": "40", "LLAMA_MOE_HYBRID_G1_RESERVE_MIB": "4900", "CUDA_VISIBLE_DEVICES": "0,1"}),
}
TEXT = os.path.join(REPO, "build/ctx100k/base.txt")
if not os.path.exists(TEXT):
    TEXT = os.path.expanduser("~/Development/poormansllama/build/ctx100k/base.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", choices=list(CFG), required=True)
ap.add_argument("--delay", type=float, default=3.0)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--mode", choices=["stream", "hold"], default="stream", help="stream: a streaming client; hold: a non-streaming client that keeps its connection open and is closed 40 s after the signal if the server is still up")
ap.add_argument("--bin", default=os.path.join(REPO, "build/llama-cuda/bin"))
ap.add_argument("--port", type=int, default=8195)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip():
    sys.exit("GPU busy, refusing")
extra, envx = CFG[a.cfg]
env = dict(os.environ, LD_LIBRARY_PATH=os.path.abspath(a.bin), **envx)
os.makedirs(os.path.join(REPO, "build/pascal/sigint"), exist_ok=True)
for rep in range(a.reps):
    delay = a.delay if rep % 2 == 0 else a.delay + 5
    logp = os.path.join(REPO, f"build/pascal/sigint/{a.cfg}-{rep}.log")
    srv = subprocess.Popen([os.path.join(a.bin, "llama-server"), "--model", MODEL, "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(BASE + " " + extra),
                           stdout=open(logp, "w"), stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(1800):
            if srv.poll() is not None:
                sys.exit(f"server exited with {srv.returncode}")
            try:
                if urllib.request.urlopen(URL + "/health", timeout=2).status == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        toks = post("/tokenize", {"content": open(TEXT, encoding="utf-8").read()})["tokens"]
        ctx = post("/detokenize", {"tokens": toks[:32768]})["content"]
        first = threading.Event()
        err = []

        def gen():
            try:
                req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": ctx + "\n\nQuestion: summarise the text above in many bullet points."}],
                                                                                          "max_tokens": 512, "temperature": 0, "stream": True}).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for line in r:
                        if line.startswith(b"data:") and b"content" in line:
                            first.set()
            except Exception as e:
                err.append(str(e)[:80])

        conn = None
        if a.mode == "hold":
            conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=3600)
            body = json.dumps({"messages": [{"role": "user", "content": ctx + "\n\nQuestion: summarise the text above in many bullet points."}], "max_tokens": 512, "temperature": 0})

            def hold():
                try:
                    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                    conn.getresponse().read()
                except Exception as e:
                    err.append(str(e)[:60])

            th = threading.Thread(target=hold, daemon=True)
            th.start()
            time.sleep(60 + delay)       # the prompt takes ~50 s, then the generation runs
        else:
            th = threading.Thread(target=gen, daemon=True)
            th.start()
            if not first.wait(900):
                print(f"{a.cfg} rep {rep}: no token within 900 s")
            time.sleep(delay)
        t0 = time.time()
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(40 if conn is not None else 60)
            print(f"{a.cfg} rep {rep} ({a.mode}): SIGINT -> exited in {time.time()-t0:.1f} s (rc {srv.returncode}); client: {err[:1]}", flush=True)
        except subprocess.TimeoutExpired:
            if conn is None:
                print(f"{a.cfg} rep {rep}: HANG - no exit within 60 s of SIGINT", flush=True)
                srv.kill()
                srv.wait()
            else:
                print(f"{a.cfg} rep {rep} (hold): the server is still up 40 s after SIGINT with the client's connection open; closing the connection ...", flush=True)
                t1 = time.time()
                try:
                    conn.sock.shutdown(2)
                except Exception:
                    pass
                conn.close()
                try:
                    srv.wait(60)
                    print(f"{a.cfg} rep {rep} (hold): exited {time.time()-t1:.1f} s after the client hung up (rc {srv.returncode})", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"{a.cfg} rep {rep} (hold): STILL UP 60 s after the client hung up", flush=True)
                    srv.kill()
                    srv.wait()
    finally:
        if srv.poll() is None:
            srv.kill()
            srv.wait()
    time.sleep(3)
