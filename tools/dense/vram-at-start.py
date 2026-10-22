#!/usr/bin/env python3
"""Start llama-server with the given flags, wait until it says it listens, read the device memory in use and the sizes it printed, stop it. Run it UNDER tools/with-gpu-lock.ps1.
  python tools/dense/vram-at-start.py LABEL EXE -- <server flags>
Prints one JSON line: label, idle VRAM (nvidia-smi), model / KV / recurrent / compute buffer MiB from the server's own log lines."""
import json, os, re, subprocess, sys, time

label, exe = sys.argv[1], sys.argv[2]
flags = sys.argv[sys.argv.index("--") + 1:]
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
log = os.path.join(root, "build", "dense", "win", label + ".server.log")
env = dict(os.environ); env["LLAMA_LM_LOG"] = "1"
with open(log, "wb") as lf:
    # VRAM_WARMUP=1: no --no-warmup, i.e. the scheduler reserve and the warm-up decode run before "listening" (their compute buffers are part of the figure)
    srv = subprocess.Popen([exe, "--host", "127.0.0.1", "--port", "8072"] + ([] if os.environ.get("VRAM_WARMUP") == "1" else ["--no-warmup"]) + flags, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=root)
t0 = time.time(); ok = False
while time.time() - t0 < 300:
    if srv.poll() is not None:
        break
    txt = open(log, "rb").read().decode("utf-8", "replace").replace("\x00", "")
    if "all slots are idle" in txt or "listening on" in txt:
        ok = True
        break
    time.sleep(1)
time.sleep(3)
used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True).stdout.split()[0])
txt = open(log, "rb").read().decode("utf-8", "replace").replace("\x00", "")
subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
def grab(pat, allv=False):
    m = re.findall(pat, txt)
    return [float(x) for x in m] if allv else (float(m[0]) if m else None)
print(json.dumps({"label": label, "ok": ok, "vram_used_mib": used,
                  "model_cuda_mib": grab(r"CUDA0 model buffer size =\s+([\d.]+) MiB"),
                  "kv_cuda_mib": grab(r"CUDA0 KV buffer size =\s+([\d.]+) MiB", True),
                  "rs_mib": grab(r"CUDA0 RS buffer size =\s+([\d.]+) MiB"),
                  "compute_mib": grab(r"CUDA0 compute buffer size =\s+([\d.]+) MiB", True)}))
