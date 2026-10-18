#!/usr/bin/env python3
r"""Soak test for llama-server (contract G31 SRV-6): >= N mixed requests, optional concurrency, cancelled streams; checks the server stays up and the log stays clean.

  python tools/server_soak.py --config CFG.json --out DIR --fixtures tools/server-fixtures/agentic-A.json,tools/server-fixtures/agentic-B.json,tools/server-fixtures/agentic-G.json [--requests 160] [--workers 2] [--seed 7]

Request mix (seeded, identical for every server): conversation continuation (cache hit), exact repeat, branch (last tool result replaced), switch of conversation, tiny cold chat, stream cancelled after a few
chunks, sampled requests. Context is capped by --max-ctx (default 40k tokens by fixture prefix length). After the run: HTTP status census, server alive, log scan (CUDA error / GGML_ASSERT / abort / 'failed'),
working set of the server process at start / end, nvidia-smi free VRAM at start / end (idle), and a final probe request (the same prompt as the first one) to prove the server still answers identically-shaped.
Output: DIR/soak.json. Exit 1 on a hard failure.
stdlib only; imports tools/server_mt.py (Server).
"""
import argparse
import http.client
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_mt import Server, conv, load_cfg  # noqa: E402


def rss_mib(pid):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-Process -Id %d).WorkingSet64" % pid], capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out) // (1024 * 1024)
    except Exception:
        return None


def smi_free():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def stream_cancel(base_port, messages, n_chunks):
    """a streamed chat request that is closed after n_chunks data lines; returns (status, chunks_read)"""
    body = json.dumps({"messages": messages, "max_tokens": 200, "temperature": 0.0, "stream": True, "cache_prompt": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    c = http.client.HTTPConnection("127.0.0.1", base_port, timeout=3600)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    got = 0
    try:
        while got < n_chunks:
            line = r.fp.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                got += 1
    finally:
        c.close()
    return r.status, got


def build_plan(fxs, n, seed, max_ctx_tokens):
    rnd = random.Random(seed)
    # usable prefix length (turns) per fixture: cumulative tool-message tokens within the cap
    caps = []
    for fx in fxs:
        tot, kmax = 1500, 0
        for k, t in enumerate(fx["turns"]):
            tot += t["tokens"] + 60
            if tot > max_ctx_tokens:
                break
            kmax = k + 1
        caps.append(kmax)
    plan = []
    state = [0] * len(fxs)    # next continuation turn per fixture
    last = None
    for i in range(n):
        r = rnd.random()
        fi = rnd.randrange(len(fxs))
        fx = fxs[fi]
        if last is not None and r < 0.12:
            plan.append(dict(kind="repeat", **last))
        elif r < 0.55:
            k = state[fi]
            plan.append(dict(kind="continue", fi=fi, k=k))
            last = dict(fi=fi, k=k)
            state[fi] = (state[fi] + 1) % (caps[fi] + 1)
        elif r < 0.68:
            k = rnd.randrange(1, caps[fi] + 1) if caps[fi] >= 1 else 0
            plan.append(dict(kind="branch", fi=fi, k=k, tag=rnd.randrange(1000)))
        elif r < 0.78:
            plan.append(dict(kind="tiny", q=rnd.choice(["Say hello.", "What is 17 * 23?", "Name three colors.", "Write one line of Python that reverses a string.", "What does GPU stand for?"]),
                             temp=rnd.choice([0.0, 0.7])))
        elif r < 0.90:
            k = rnd.randrange(0, caps[fi] + 1)
            plan.append(dict(kind="cancel", fi=fi, k=k, chunks=rnd.choice([1, 3, 8])))
        else:
            k = rnd.randrange(0, caps[fi] + 1)
            plan.append(dict(kind="sampled", fi=fi, k=k, seed=rnd.randrange(10**6)))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--requests", type=int, default=160)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-ctx", type=int, default=40000)
    ap.add_argument("--rss-allow-mib", type=int, default=1024, help="allowed growth of the server's working set over the run (with the default --cache-ram 8192 the host prompt cache alone may grow by that much: pass 9216, or run with --cache-ram 0 and keep 1024)")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--extra-arg", action="append", default=[])
    a = ap.parse_args()
    fxs = [json.load(open(p, "r", encoding="utf-8")) for p in a.fixtures.split(",")]
    plan = build_plan(fxs, a.requests, a.seed, a.max_ctx)
    srv = Server(load_cfg(a.config), a.out, a.port, a.extra_arg)
    results = []
    lock = threading.Lock()
    rc = 0
    try:
        t_ready = srv.start()
        print("server ready after %.1f s; %d requests, %d workers" % (t_ready, len(plan), a.workers))
        pid = srv.proc.pid
        rss0, free0 = rss_mib(pid), smi_free()
        probe_msgs = conv(fxs[0], 1)
        p0 = srv.chat(probe_msgs, 24)
        rss_after_probe0, free_after_probe0 = rss_mib(pid), smi_free()
        idx = [0]
        t0 = time.time()

        def run_one(i, item):
            kind = item["kind"]
            t1 = time.time()
            rec = {"i": i, "kind": kind, "status": None, "ok": False, "wall_s": None}
            try:
                if kind in ("continue", "repeat"):
                    m = conv(fxs[item["fi"]], item["k"])
                    r = srv.chat(m, 24)
                elif kind == "branch":
                    k = item["k"]
                    m = conv(fxs[item["fi"]], k - 1) + [{"role": "assistant", "content": fxs[item["fi"]]["turns"][k - 1]["assistant"]},
                                                         {"role": "user", "content": "Result of the last tool call:\nbranch %d: the command printed nothing." % item["tag"]}]
                    r = srv.chat(m, 24)
                elif kind == "tiny":
                    r = srv.chat([{"role": "user", "content": item["q"]}], 32, temperature=item["temp"])
                elif kind == "sampled":
                    r = srv.chat(conv(fxs[item["fi"]], item["k"]), 40, temperature=0.7, extra={"top_p": 0.95, "top_k": 20, "seed": item["seed"]})
                elif kind == "cancel":
                    st, got = stream_cancel(a.port, conv(fxs[item["fi"]], item["k"]), item["chunks"])
                    rec.update(status=st, ok=(st == 200), chunks=got)
                    r = None
                if r is not None:
                    rec.update(status=r["status"], ok=(r["status"] == 200 and bool((r.get("content") or "").strip()) or (r["status"] == 200 and r.get("tool_calls"))),
                               prompt_n=r["timings"].get("prompt_n"), cache_n=r["timings"].get("cache_n"), predicted_n=r["timings"].get("predicted_n"))
                    if not rec["ok"]:
                        rec["error"] = r.get("error") or "empty content"
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)
            rec["wall_s"] = round(time.time() - t1, 2)
            with lock:
                results.append(rec)
                n = len(results)
            if n % 10 == 0 or not rec["ok"]:
                print("[%4d/%d] %-8s status %s ok %s wall %.1fs prompt_n %s cache_n %s %s" % (n, len(plan), kind, rec["status"], rec["ok"], rec["wall_s"], rec.get("prompt_n"), rec.get("cache_n"), rec.get("error", "")))
                sys.stdout.flush()

        def worker():
            while True:
                with lock:
                    i = idx[0]
                    idx[0] += 1
                if i >= len(plan):
                    return
                run_one(i, plan[i])
                if not srv.alive():
                    return

        ths = [threading.Thread(target=worker) for _ in range(a.workers)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        total = time.time() - t0
        alive = srv.alive()
        time.sleep(3)
        p1 = srv.chat(probe_msgs, 24) if alive else None
        rss1, free1 = (rss_mib(pid) if alive else None), smi_free()
        bad = [r for r in results if not r["ok"]]
        census = {}
        for r in results:
            census[str(r["status"])] = census.get(str(r["status"]), 0) + 1
        summary = {"requests": len(results), "planned": len(plan), "census": census, "failed": len(bad), "alive": alive, "wall_s": round(total, 1), "rss_mib": [rss0, rss_after_probe0, rss1],
                   "smi_free_mib": [free0, free_after_probe0, free1], "free_min_smi_mib": srv.sampler.min_free, "probe_before": p0.get("content"), "probe_after": p1.get("content") if p1 else None,
                   "probe_status": [p0.get("status"), p1.get("status") if p1 else None], "workers": a.workers}
        # log scan
        srv.stop()
        txt = open(os.path.join(a.out, "server.log"), "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
        pats = [r"CUDA error", r"GGML_ASSERT", r"ggml_abort", r"could not make room", r"failed to", r"slot.*error", r"exception", r"out of memory"]
        hits = {}
        for p in pats:
            m = re.findall(r"^.*(?:%s).*$" % p, txt, flags=re.M | re.I)
            if m:
                hits[p] = [x[:200] for x in m[:3]] + (["... %d lines" % len(m)] if len(m) > 3 else [])
        summary["log_hits"] = hits
        w = [int(x) for x in re.findall(r"free-watch: (?:new minimum of free device memory|the engine is gone; lowest free device memory) (\d+) MiB", txt)]
        summary["watcher_min_mib"] = min(w) if w else None
        shrinks = re.findall(r"expert cache (\d+) -> (\d+) slots", txt)
        summary["resizes"] = shrinks
        json.dump({"summary": summary, "results": results, "plan": plan}, open(os.path.join(a.out, "soak.json"), "w"), indent=1)
        print(json.dumps(summary, indent=1))
        fails = []
        if len(results) < a.requests:
            fails.append("only %d of %d requests ran" % (len(results), a.requests))
        if bad:
            fails.append("%d requests failed: %s" % (len(bad), [(r["i"], r["kind"], r["status"], r.get("error")) for r in bad[:5]]))
        if not alive:
            fails.append("server died")
        if p1 is None or p1.get("status") != 200:
            fails.append("final probe failed")
        if summary["rss_mib"][0] is not None and summary["rss_mib"][2] is not None and summary["rss_mib"][2] - summary["rss_mib"][1] > a.rss_allow_mib:
            fails.append("host RSS grew by %d MiB (allowed %d)" % (summary["rss_mib"][2] - summary["rss_mib"][1], a.rss_allow_mib))
        for p in ("CUDA error", "GGML_ASSERT", "ggml_abort", "out of memory"):
            if p in hits:
                fails.append("log: " + p)
        if fails:
            print("FAILED:")
            for f in fails:
                print("  ", f)
            rc = 1
        else:
            print("soak ok")
    finally:
        try:
            srv.stop()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
