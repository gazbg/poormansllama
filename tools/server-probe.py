#!/usr/bin/env python3
r"""Start a llama-server, run a scenario file against it, stop it (host + GPU; run it under tools/with-gpu-lock.ps1).

  python tools/server-probe.py --exe build\llama-cuda\bin\g60\llama-server.exe --out build\llama-cuda\g200-x --scenario tools\server-scenarios\smoke.json
         [--env KEY=VALUE ...] [--ready-timeout 600] -- <llama-server arguments>

The scenario file is JSON: {"steps": [ {"name": ..., "endpoint": "/completion" | "/v1/chat/completions", "body": {...}, "repeat": 1, "expect": {...}} ... ]}.
Every response's `timings` (prompt_n, prompt_ms, cache_n, predicted_n, predicted_per_second ...) and text are written to <out>/steps.jsonl; the server's stdout / stderr go to <out>/server.log.
"expect" may hold: "min_cache_n" (the request must reuse at least that many prompt tokens), "max_prompt_n" (at most that many tokens may be processed), "contains" (a substring of the text),
"same_text_as" (the name of an earlier step whose text must be identical, greedy requests), "min_tps" (predicted_per_second at least). A failed expectation makes the exit status 1.
stdlib only. The server is stopped with a terminate (then kill after 30 s); the GPU wrapper checks that no compute process is left.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def http(method, url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


class FreeSampler(threading.Thread):
    """samples the free device memory (MiB) with nvidia-smi every `period` seconds; min/last are read at the end"""
    def __init__(self, path, period=0.1):
        super().__init__(daemon=True)
        self.path, self.period, self.stop_flag = path, period, False
        self.min_free = None
        self.max_free = None
        self.samples = 0

    def run(self):
        t0 = time.time()
        with open(self.path, "w") as f:
            while not self.stop_flag:
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                    v = int(out.splitlines()[0])
                    self.samples += 1
                    self.min_free = v if self.min_free is None else min(self.min_free, v)
                    self.max_free = v if self.max_free is None else max(self.max_free, v)
                    f.write("%.2f %d\n" % (time.time() - t0, v))
                except Exception:
                    pass
                time.sleep(self.period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe")
    ap.add_argument("--config", help="JSON file {\"exe\": ..., \"args\": [...], \"env\": {...}}: replaces --exe / --env / the argument tail (no shell quoting)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--ready-timeout", type=int, default=900)
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    sargs = a.args[1:] if a.args and a.args[0] == "--" else a.args
    if a.config:
        cfg = json.load(open(a.config, "r", encoding="utf-8"))
        a.exe = cfg["exe"]
        sargs = [str(x) for x in cfg["args"]]
        a.env = ["%s=%s" % (k, v) for k, v in cfg.get("env", {}).items()] + a.env
    if not a.exe:
        print("need --exe or --config")
        return 2
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    cmd = [a.exe, "--host", "127.0.0.1", "--port", str(a.port)] + sargs
    log = open(os.path.join(a.out, "server.log"), "wb")
    log.write((" ".join(cmd) + "\n").encode("utf-8"))
    log.write(("ENV " + " ".join("%s=%s" % (kv.split("=", 1)[0], kv.split("=", 1)[1]) for kv in a.env) + "\n").encode("utf-8"))
    log.flush()
    t_start = time.time()
    sampler = FreeSampler(os.path.join(a.out, "free_mib.txt"))
    sampler.start()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    base = "http://127.0.0.1:%d" % a.port
    rc = 0
    try:
        ready = False
        while time.time() - t_start < a.ready_timeout:
            if proc.poll() is not None:
                print("server exited early with", proc.returncode)
                return 2
            try:
                st, js = http("GET", base + "/health", timeout=5)
                if st == 200 and js.get("status") == "ok":
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        if not ready:
            print("server not ready in", a.ready_timeout, "s")
            return 2
        print("server ready after %.1f s" % (time.time() - t_start))
        scen = json.load(open(a.scenario, "r", encoding="utf-8"))
        texts = {}
        out = open(os.path.join(a.out, "steps.jsonl"), "w", encoding="utf-8")
        for step in scen["steps"]:
            for rep in range(int(step.get("repeat", 1))):
                name = step["name"] + ("" if int(step.get("repeat", 1)) == 1 else "#%d" % (rep + 1))
                body = step["body"]
                if "prompt_file" in step:
                    body = dict(body)
                    body["prompt"] = open(step["prompt_file"], "r", encoding="utf-8").read() + step.get("prompt_suffix", "")
                t0 = time.time()
                try:
                    st, js = http("POST", base + step.get("endpoint", "/completion"), body)
                except urllib.error.HTTPError as e:
                    st, js = e.code, {"error": e.read().decode("utf-8", "replace")}
                wall = time.time() - t0
                tm = js.get("timings", {}) if isinstance(js, dict) else {}
                if "content" in js:
                    text = js["content"]
                elif "choices" in js and js["choices"]:
                    ch = js["choices"][0]
                    text = (ch.get("message") or {}).get("content") or ch.get("text") or ""
                else:
                    text = ""
                texts[name] = text
                rec = {"name": name, "status": st, "wall_s": round(wall, 3), "timings": tm, "text": text, "usage": js.get("usage") if isinstance(js, dict) else None}
                if st != 200:
                    rec["error"] = js.get("error")
                fails = []
                ex = step.get("expect", {})
                if st != 200:
                    fails.append("http %d" % st)
                if "min_cache_n" in ex and tm.get("cache_n", 0) < ex["min_cache_n"]:
                    fails.append("cache_n %s < %s" % (tm.get("cache_n"), ex["min_cache_n"]))
                if "max_prompt_n" in ex and tm.get("prompt_n", 10**9) > ex["max_prompt_n"]:
                    fails.append("prompt_n %s > %s" % (tm.get("prompt_n"), ex["max_prompt_n"]))
                if "contains" in ex and ex["contains"] not in text:
                    fails.append("text lacks %r" % ex["contains"])
                if "same_text_as" in ex and texts.get(ex["same_text_as"]) != text:
                    fails.append("text differs from %s" % ex["same_text_as"])
                if "min_tps" in ex and tm.get("predicted_per_second", 0) < ex["min_tps"]:
                    fails.append("tps %.2f < %s" % (tm.get("predicted_per_second", 0), ex["min_tps"]))
                rec["fails"] = fails
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print("%-28s http %d wall %6.2fs prompt_n %5s cache_n %5s prompt %8.1f ms (%7.1f t/s)  gen %4s %7.2f t/s%s" % (
                    name, st, wall, tm.get("prompt_n"), tm.get("cache_n"), tm.get("prompt_ms", 0), tm.get("prompt_per_second", 0),
                    tm.get("predicted_n"), tm.get("predicted_per_second", 0), ("   FAIL: " + "; ".join(fails)) if fails else ""))
                if fails:
                    rc = 1
        out.close()
    finally:
        sampler.stop_flag = True
        print("free VRAM during the session: min %s MiB, max %s MiB (%d samples)" % (sampler.min_free, sampler.max_free, sampler.samples))
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
