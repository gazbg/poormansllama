#!/usr/bin/env python3
r"""Multi-turn test harness for llama-server (contract G31, docs/plans/llama-server-integration.md). Host + GPU: run it under tools/with-gpu-lock.ps1.

  python tools/server_mt.py build-fixture --config CFG.json --out DIR --fixture tools/server-fixtures/agentic-1.json [--turns 10]
  python tools/server_mt.py run --config CFG.json --out DIR --fixture tools/server-fixtures/agentic-1.json --scenarios prefix,agentic,branch,alternate,sampling,equiv

CFG.json = {"exe": ..., "args": [...], "env": {...}} (see tools/server-config-gen.py). The fixture holds a fixed agentic transcript (system prompt + tool-result messages of 0.5-20k tokens +
the assistant replies recorded once), so that STOCK and OPT servers see the SAME prompts. Every request is greedy (temperature 0) unless a scenario says otherwise.
Output: DIR/report.json (every request: timings, cache reuse, expectations) and a table on stdout. Exit status 1 when an expectation fails.

Equivalence (SRV-5): a cache-hit continuation and the same prompt processed from scratch (cache_prompt false) are compared token by token; they may differ only at a near tie: the first differing
token must be the reference's runner-up with a logprob margin < NEAR_TIE (0.25 nats).
stdlib only.
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

NEAR_TIE = 0.25


# ----------------------------------------------------------------------------- server

class FreeSampler(threading.Thread):
    """nvidia-smi free device memory every 100 ms (the CUDA-runtime view is ~1.1 GiB lower: see the server log's hybrid-moe free-watch lines)"""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop_flag, self.min_free, self.max_free, self.n = path, False, None, None, 0

    def run(self):
        t0 = time.time()
        with open(self.path, "w") as f:
            while not self.stop_flag:
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                    v = int(out.splitlines()[0])
                    self.n += 1
                    self.min_free = v if self.min_free is None else min(self.min_free, v)
                    self.max_free = v if self.max_free is None else max(self.max_free, v)
                    f.write("%.2f %d\n" % (time.time() - t0, v))
                except Exception:
                    pass
                time.sleep(0.1)


class Server:
    def __init__(self, cfg, out, port=18080, extra_args=None, extra_env=None):
        self.cfg, self.out, self.port = cfg, out, port
        self.args = [str(x) for x in cfg["args"]] + (extra_args or [])
        self.env = dict(os.environ)
        for k, v in dict(cfg.get("env", {}), **(extra_env or {})).items():
            self.env[k] = str(v)
        self.base = "http://127.0.0.1:%d" % port
        self.proc = None
        self.sampler = None

    def start(self, ready_timeout=900):
        os.makedirs(self.out, exist_ok=True)
        self.logf = open(os.path.join(self.out, "server.log"), "ab")
        cmd = list(self.cfg.get("wrap", [])) + [self.cfg["exe"], "--host", "127.0.0.1", "--port", str(self.port)] + self.args
        self.logf.write((" ".join(cmd) + "\n").encode("utf-8"))
        self.logf.flush()
        self.sampler = FreeSampler(os.path.join(self.out, "free_mib.txt"))
        self.sampler.start()
        t0 = time.time()
        # SERVER_GRACEFUL_STOP=1: the server owns a console so that stop() can end it with Ctrl-C (tools/win_ctrlc.py); compute-sanitizer then prints its ERROR SUMMARY
        self.graceful = os.environ.get("SERVER_GRACEFUL_STOP") == "1" and os.name == "nt"
        if self.graceful:
            # Ctrl-C is inherited as DISABLED from a non-interactive parent (the wrapper shells): re-enable it here so that the server's console handler can receive the event of stop()
            import ctypes
            ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=self.env, creationflags=(subprocess.CREATE_NEW_CONSOLE if self.graceful else 0))
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early with %s (see %s/server.log)" % (self.proc.returncode, self.out))
            try:
                st, js = self.api("/health", timeout=5)
                if st == 200 and js.get("status") == "ok":
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("server not ready")

    def stop(self):
        if self.sampler:
            self.sampler.stop_flag = True
        if getattr(self, "stopped", False):
            return
        self.stopped = True
        if self.proc is not None and self.proc.poll() is not None:
            print("server exit code: %s (0x%X)" % (self.proc.returncode, self.proc.returncode & 0xFFFFFFFF))
        if self.proc and self.proc.poll() is None and getattr(self, "graceful", False):
            try:
                if os.environ.get("SERVER_STOP_MODE", "child") == "child":
                    # the process we started is a wrapper (compute-sanitizer): end its CHILD, the target, and let the wrapper finish its report on its own (a Ctrl-C ends the wrapper too, without the report)
                    out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process -Filter 'ParentProcessId=%d').ProcessId" % self.proc.pid], capture_output=True, text=True, timeout=60).stdout.split()
                    for kid in out:
                        subprocess.run(["taskkill", "/PID", kid, "/F"], capture_output=True, timeout=60)
                    print("server stop: ended the target process(es) %s of the wrapper" % out)
                else:
                    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "win_ctrlc.py"), str(self.proc.pid)], timeout=60)
                self.proc.wait(timeout=int(os.environ.get("SERVER_GRACEFUL_WAIT", "600")))
                print("wrapper exited with code %s" % self.proc.returncode)
            except Exception as e:   # noqa: BLE001
                print("graceful stop failed: %r" % (e,))
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def api(self, path, body=None, timeout=3600):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(txt)
            except Exception:
                return e.code, {"error": txt}
        except (ConnectionError, OSError) as e:
            rc = None
            for k in range(8):
                time.sleep(2.0)
                rc = self.proc.poll() if self.proc is not None else None
                state = "ALIVE" if (self.proc is not None and rc is None) else ("exited with %s (0x%X)" % (rc, (rc or 0) & 0xFFFFFFFF))
                health = None
                if rc is None and path != "/health":
                    try:
                        health = urllib.request.urlopen(self.base + "/health", timeout=3).status
                    except Exception as e2:   # noqa: BLE001
                        health = repr(e2)[:80]
                print("connection error on %s: %r; +%ds server process %s; /health %s" % (path, e, 2 * (k + 1), state, health))
                if rc is not None:
                    break
            raise

    # ---- helpers
    def chat(self, messages, max_tokens=48, temperature=0.0, logprobs=False, cache_prompt=True, extra=None, tools=None):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 1, "cache_prompt": cache_prompt,
                "chat_template_kwargs": {"enable_thinking": False}}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if tools:
            body["tools"] = tools
        if extra:
            body.update(extra)
        if THINK_S > 0:
            time.sleep(THINK_S)
        t0 = time.time()
        st, js = self.api("/v1/chat/completions", body)
        wall = time.time() - t0
        rec = {"status": st, "wall_s": round(wall, 3), "timings": js.get("timings", {}) if isinstance(js, dict) else {}, "usage": js.get("usage") if isinstance(js, dict) else None}
        if st == 200 and js.get("choices"):
            ch = js["choices"][0]
            msg = ch.get("message") or {}
            rec["content"] = msg.get("content") or ""
            rec["finish"] = ch.get("finish_reason")
            rec["tool_calls"] = msg.get("tool_calls")
            lp = (ch.get("logprobs") or {}).get("content") or []
            rec["tokens"] = [(e.get("token"), e.get("logprob"), [(t.get("token"), t.get("logprob")) for t in (e.get("top_logprobs") or [])]) for e in lp]
        else:
            rec["content"] = ""
            rec["error"] = js.get("error") if isinstance(js, dict) else str(js)
        return rec

    def render_tokens(self, messages, tools=None):
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            body["tools"] = tools
        st, js = self.api("/apply-template", body)
        if st != 200:
            raise RuntimeError("apply-template failed: %s" % js)
        st, js2 = self.api("/tokenize", {"content": js["prompt"], "add_special": False, "parse_special": True})
        if st != 200:
            raise RuntimeError("tokenize failed: %s" % js2)
        return js2["tokens"]

    def n_tokens(self, text):
        st, js = self.api("/tokenize", {"content": text, "add_special": False})
        return len(js["tokens"])


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compare_greedy(ref, cand):
    """ref / cand: token lists of chat(): (token, logprob, [(tok, lp) top-2]). Returns (identical, first_diff, margin, ok)."""
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i][0] != cand[i][0]:
            top = ref[i][2]
            lp_ref = ref[i][1]
            lp_c = None
            for tok, lp in top:
                if tok == cand[i][0]:
                    lp_c = lp
            if lp_c is None or lp_ref is None:
                return False, i, None, False
            margin = lp_ref - lp_c
            return False, i, margin, margin < NEAR_TIE
    return len(ref) == len(cand), None, None, True


# ----------------------------------------------------------------------------- report

class Report:
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.fails = []
        self.blobs = {}

    def add(self, scenario, name, rec, expect_fails=None, note=None):
        t = rec.get("timings", {})
        row = {"scenario": scenario, "name": name, "status": rec.get("status"), "wall_s": rec.get("wall_s"), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
               "prompt_ms": t.get("prompt_ms"), "prompt_per_second": t.get("prompt_per_second"), "predicted_n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
               "finish": rec.get("finish"), "fails": expect_fails or [], "note": note}
        if rec.get("status") != 200:
            row["fails"] = row["fails"] + ["http %s: %s" % (rec.get("status"), str(rec.get("error"))[:200])]
        self.rows.append(row)
        for f in row["fails"]:
            self.fails.append("%s/%s: %s" % (scenario, name, f))
        print("%-9s %-26s http %s wall %7.2fs prompt_n %6s cache_n %6s %8.1f t/s pp | gen %4s %7.2f t/s%s%s" % (
            scenario, name, row["status"], row["wall_s"] or 0, row["prompt_n"], row["cache_n"], row["prompt_per_second"] or 0, row["predicted_n"], row["tps"] or 0,
            ("  [%s]" % note) if note else "", ("  FAIL: " + "; ".join(row["fails"])) if row["fails"] else ""))
        sys.stdout.flush()

    def check(self, scenario, name, ok, msg):
        row = {"scenario": scenario, "name": name, "check": True, "fails": [] if ok else [msg]}
        self.rows.append(row)
        if not ok:
            self.fails.append("%s/%s: %s" % (scenario, name, msg))
        print("%-9s %-26s %s" % (scenario, name, "ok: " + msg if ok else "FAIL: " + msg))

    def save(self, extra=None):
        d = {"rows": self.rows, "fails": self.fails, "blobs": self.blobs}
        if extra:
            d.update(extra)
        json.dump(d, open(self.path, "w"), indent=1)


# ----------------------------------------------------------------------------- fixture

SYSTEM_TOOLS = """You are a careful coding agent working in a repository through tools. Answer in at most two short sentences unless asked for code.
Available tools (call them by writing the tool name and a JSON object):
- read_file {"path": string, "start": integer, "end": integer}: returns the lines of a file with line numbers.
- search {"pattern": string, "path": string}: returns matching lines as path:line:text.
- run_tests {"target": string}: runs the tests of a target and returns their output.
- edit_file {"path": string, "old": string, "new": string}: replaces a text in a file and returns a diff.
Rules: read before you edit, keep edits minimal, never invent file contents, report the result of every tool call before the next step, and stop when the failing test passes.
""" + "\n".join("Guideline %d: prefer small, verifiable steps and explain what you checked in one sentence." % i for i in range(1, 60))


def code_pool(root):
    paths = []
    for base in ("tools", "docs/plans/tg-beat-stock-specs"):
        for fn in sorted(os.listdir(os.path.join(root, base))):
            if fn.endswith(".py") or fn.endswith(".sh") or fn.endswith(".ps1"):
                paths.append(os.path.join(base, fn))
    pool = []
    for p in paths:
        try:
            lines = open(os.path.join(root, p), "r", encoding="utf-8", errors="replace").read().replace("\\", "/").splitlines()
        except Exception:
            continue
        pool.extend("%s:%d: %s" % (p, i + 1, l) for i, l in enumerate(lines))
    return pool


def slice_to_tokens(server, pool, start, n_tokens):
    """a slice of the pool of about n_tokens tokens (ratio calibrated once per call; backslash-free text)"""
    chars = int(n_tokens * 3.1)
    out, i = [], start % len(pool)
    size = 0
    while size < chars:
        l = pool[i % len(pool)]
        out.append(l)
        size += len(l) + 1
        i += 1
    text = "\n".join(out)
    n = server.n_tokens(text)
    # one correction step
    if n > 0 and abs(n - n_tokens) > 0.05 * n_tokens:
        text = text[: int(len(text) * n_tokens / n)]
    return text


FIRST_USER = {"A": "The test test_streaming_layout fails after the last change. Investigate step by step with the tools. Start by searching for the failing target.",
              "B": "Please add a --dry-run option to the benchmark runner and make the unit test for the option pass. Work step by step with the tools and start by reading the option parser.",
              "G": "The nightly build is red. Find out why with the tools, step by step. Keep the build key of this repository in mind: you will be asked for it."}
NEEDLE = "QKFM-7391"
MAX_TURNS = None
THINK_S = 0.0         # --think SEC: idle seconds before every chat request (a user reading, a tool running: the time the server has to prepare the next pass)
REPROCESS_ALLOW = 0   # --reprocess-allow N: tokens a cache hit may re-prefill beyond its new tokens (--lm-prefill-canonical resumes from the newest chunk-boundary checkpoint: up to one chunk)
OFFSET0 = {"A": 0, "B": 4321, "G": 9000}


def build_fixture(server, path, sizes, root, variant="A"):
    pool = code_pool(root)
    msgs = [{"role": "system", "content": SYSTEM_TOOLS}, {"role": "user", "content": FIRST_USER[variant]}]
    turns = []
    offset = OFFSET0[variant]
    for k, sz in enumerate(sizes):
        rec = server.chat(msgs, max_tokens=48)
        reply = rec["content"].strip() or "I will continue with the next step."
        text = slice_to_tokens(server, pool, offset, sz)
        offset += 997 * (k + 3)
        tool_msg = "Result of the last tool call:\n" + (("NOTE: the build key of this repository is %s.\n" % NEEDLE) if (variant == "G" and k == 0) else "") + text
        turns.append({"assistant": reply, "user": tool_msg, "tokens": server.n_tokens(tool_msg)})
        msgs = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": tool_msg}]
        print("fixture turn %2d: reply %3d chars, tool message %6d tokens" % (k, len(reply), turns[-1]["tokens"]))
    fx = {"system": SYSTEM_TOOLS, "first_user": msgs[1]["content"], "turns": turns}
    if variant == "G":
        fx["needle"] = NEEDLE
    json.dump(fx, open(path, "w"), indent=1)


def conv(fx, n_turns, upto_user=True):
    """messages of the fixed conversation after n_turns tool results: system, user, (assistant, tool-result) x n_turns"""
    msgs = [{"role": "system", "content": fx["system"]}, {"role": "user", "content": fx["first_user"]}]
    for k in range(n_turns):
        msgs.append({"role": "assistant", "content": fx["turns"][k]["assistant"]})
        msgs.append({"role": "user", "content": fx["turns"][k]["user"]})
    return msgs


# ----------------------------------------------------------------------------- scenarios

def sc_prefix(srv, rep, fx):
    """T1: identical repeat, extension"""
    m = conv(fx, 2)
    a = srv.chat(m, 32)
    rep.add("prefix", "cold", a)
    n = (a["timings"].get("prompt_n") or 0) + (a["timings"].get("cache_n") or 0)
    b = srv.chat(m, 32)
    f = []
    if (b["timings"].get("prompt_n") or 999) > 8 + REPROCESS_ALLOW:
        f.append("repeat processed %s prompt tokens (> %d)" % (b["timings"].get("prompt_n"), 8 + REPROCESS_ALLOW))
    if (b["timings"].get("cache_n") or 0) < n - 8 - REPROCESS_ALLOW:
        f.append("repeat reused only %s of %s tokens" % (b["timings"].get("cache_n"), n))
    rep.add("prefix", "repeat", b, f)
    m2 = m + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "Now say done."}]
    c = srv.chat(m2, 16)
    f = []
    if (c["timings"].get("cache_n") or 0) < n - 600 - REPROCESS_ALLOW:
        f.append("extension reused %s of the %s earlier tokens" % (c["timings"].get("cache_n"), n))
    rep.add("prefix", "extension", c, f)


def sc_agentic(srv, rep, fx, max_turns=None):
    """T2/T5: the fixed agentic transcript, turn by turn"""
    prev_tokens, prev_gen = None, []
    total = 0
    if max_turns is None:
        max_turns = MAX_TURNS
    n = len(fx["turns"]) if max_turns is None else min(max_turns, len(fx["turns"]))
    for k in range(n + 1):
        m = conv(fx, k)
        toks = srv.render_tokens(m)
        r = srv.chat(m, 48)
        t = r["timings"]
        f = []
        note = "ctx %d" % len(toks)
        if prev_tokens is not None:
            l = lcp(toks, prev_tokens)
            new = len(toks) - l
            note += ", lcp(prev prompt) %d, new %d" % (l, new)
            # SRV-4 (absolute form): at most the new tokens + 64 are processed; the relative form (OPT vs STOCK) is checked by tools/server_mt_compare.py
            if (t.get("prompt_n") or 0) > new + 64 + REPROCESS_ALLOW:
                f.append("SRV-4: processed %s prompt tokens, new part is %d (+%d allowed)" % (t.get("prompt_n"), new, 64 + REPROCESS_ALLOW))
        rep.add("agentic", "turn%02d" % k, r, f, note)
        prev_tokens = toks
    return


def sc_branch(srv, rep, fx):
    """T3: after the long transcript, a request that keeps the first j turns and changes the next tool message"""
    n = len(fx["turns"])
    full = conv(fx, n)
    r = srv.chat(full, 24)
    rep.add("branch", "full", r)
    j = n // 2
    m = conv(fx, j)
    m = m + [{"role": "assistant", "content": fx["turns"][j]["assistant"]}, {"role": "user", "content": "Result of the last tool call:\nTimeout: the command did not finish."}]
    r2 = srv.chat(m, 24, logprobs=True)
    toks = srv.render_tokens(m)
    full_toks = srv.render_tokens(full)
    d = lcp(toks, full_toks)
    rep.add("branch", "rollback", r2, [], "diverges at token %d of %d" % (d, len(toks)))
    r3 = srv.chat(m, 24)
    f = []
    if (r3["timings"].get("prompt_n") or 999) > 8 + REPROCESS_ALLOW:
        f.append("repeat of the branch processed %s tokens" % r3["timings"].get("prompt_n"))
    rep.add("branch", "branch-repeat", r3, f)
    ref = srv.chat(m, 24, logprobs=True, cache_prompt=False)
    rep.add("branch", "branch-fresh", ref)
    ident, first, margin, ok = compare_greedy(ref["tokens"], r2["tokens"])
    rep.check("branch", "rollback-vs-fresh", ok, "identical over %d tokens" % len(ref["tokens"]) if ident else "first difference at token %s, margin %s nats" % (first, None if margin is None else round(margin, 3)))


def sc_alternate(srv, rep, fx, fx2):
    """T4: conversation A, then B on the same slot, then A again (host prompt cache / slot similarity)"""
    ka = min(4, len(fx["turns"]))
    ma = conv(fx, ka)
    a1 = srv.chat(ma, 16, logprobs=True)
    rep.add("alternate", "A-first", a1)
    n_a = (a1["timings"].get("prompt_n") or 0) + (a1["timings"].get("cache_n") or 0)
    mb = conv(fx2, min(4, len(fx2["turns"])))
    b1 = srv.chat(mb, 16)
    rep.add("alternate", "B-first", b1)
    a2 = srv.chat(ma, 16, logprobs=True)
    rep.add("alternate", "A-again", a2, [], "A had %d tokens" % n_a)
    ident, first, margin, ok = compare_greedy(a1["tokens"], a2["tokens"])
    rep.check("alternate", "restore-vs-first", ok, "identical over %d tokens" % len(a1["tokens"]) if ident else "first difference at token %s, margin %s nats" % (first, None if margin is None else round(margin, 3)))


def sc_sampling(srv, rep, fx):
    """T7: the sampling parameters of a typical client"""
    m = conv(fx, 3)
    for i in range(3):
        r = srv.chat(m, 64, temperature=0.6, extra={"top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 100 + i})
        f = []
        if not r["content"].strip():
            f.append("empty response")
        rep.add("sampling", "sampled#%d" % i, r, f)


def sc_equiv(srv, rep, fx):
    """T8 / SRV-5: a cache-hit continuation against the same prompt processed from scratch, at a few depths"""
    for k in (1, 3, len(fx["turns"])):
        k = min(k, len(fx["turns"]))
        m = conv(fx, k)
        srv.chat(conv(fx, max(k - 1, 0)), 8)                      # populate the cache with the previous turn's prompt
        hit = srv.chat(m, 64, logprobs=True)                       # cache hit: only the new part is processed
        ref = srv.chat(m, 64, logprobs=True, cache_prompt=False)   # from scratch
        ident, first, margin, ok = compare_greedy(ref["tokens"], hit["tokens"])
        fd, top1 = first_token_delta(ref["tokens"], hit["tokens"])
        msg = "turn %d: identical over %d tokens" % (k, len(ref["tokens"])) if ident else "turn %d: first difference at token %s, margin %s nats" % (k, first, None if margin is None else round(margin, 3))
        msg += " | 1st token (from the prefill): max|dlogprob| %s, top-1 %s" % (None if fd is None else round(fd, 4), "same" if top1 else "DIFFERENT")
        rep.add("equiv", "hit-turn%02d" % k, hit, [], "cache_n %s" % hit["timings"].get("cache_n"))
        rep.add("equiv", "fresh-turn%02d" % k, ref, [], "prompt_n %s" % ref["timings"].get("prompt_n"))
        rep.check("equiv", "same-turn%02d" % k, ok, msg)


def sc_growth(srv, rep, fx):
    """T5: the conversation grows turn by turn (to >= 100k tokens with fixture G); every third turn and the last one a quality probe asks for the needle of the first tool result"""
    n = len(fx["turns"]) if MAX_TURNS is None else min(MAX_TURNS, len(fx["turns"]))
    prev_tokens = None
    for k in range(n + 1):
        m = conv(fx, k)
        toks = srv.render_tokens(m)
        r = srv.chat(m, 24)
        t = r["timings"]
        f = []
        note = "ctx %d" % len(toks)
        if prev_tokens is not None:
            l = lcp(toks, prev_tokens)
            new = len(toks) - l
            note += ", new %d" % new
            if (t.get("prompt_n") or 0) > new + 64 + REPROCESS_ALLOW:
                f.append("SRV-4: processed %s prompt tokens, new part is %d (+%d allowed)" % (t.get("prompt_n"), new, 64 + REPROCESS_ALLOW))
        rep.add("growth", "turn%02d" % k, r, f, note)
        prev_tokens = toks
        if fx.get("needle") and k >= 1 and (k % 3 == 0 or k == n):
            q = m + [{"role": "assistant", "content": fx["turns"][k]["assistant"] if k < n else "Understood."},
                     {"role": "user", "content": "What is the build key of this repository, as stated in the first tool result? Answer with the key only."}]
            p = srv.chat(q, 16)
            ok = fx["needle"] in (p.get("content") or "")
            rep.add("growth", "probe%02d" % k, p, [] if ok else ["quality probe: the build key %s is not in the answer %r" % (fx["needle"], (p.get("content") or "")[:80])], "answer %r" % (p.get("content") or "")[:40])


def sc_xcheck(srv, rep, fx):
    """cross-server check material: the greedy continuation (tokens + top-2 logprobs) of fixed prompts processed from scratch; tools/server_mt_xcheck.py compares two reports (SRV-5/SRV-7: ubatch 512 vs 2048, STOCK vs OPT)"""
    out = []
    for k in (1, 3, 6, len(fx["turns"])):
        k = min(k, len(fx["turns"]))
        m = conv(fx, k)
        r = srv.chat(m, 64, logprobs=True, cache_prompt=False)
        rep.add("xcheck", "fresh-turn%02d" % k, r)
        out.append({"k": k, "tokens": r.get("tokens", [])})
    rep.blobs["xcheck"] = out


def pair_note(ref, cand):
    """(identical, first_diff, margin_to_top1_of_ref or None, rank_in_ref_top5 or None)"""
    n = min(len(ref), len(cand))
    for i in range(n):
        if ref[i][0] != cand[i][0]:
            top = ref[i][2]
            rank = None
            m = None
            for r, (tok, lp) in enumerate(top):
                if tok == cand[i][0]:
                    rank, m = r + 1, (ref[i][1] - lp) if ref[i][1] is not None else None
            return False, i, m, rank
    return len(ref) == len(cand), None, None, None


def first_token_delta(ref, cand):
    """the first generated token comes out of the PREFILL (no hybrid decode involved): max |delta logprob| over the tokens both runs list in their top-5, and whether the top-1 agrees"""
    if not ref or not cand:
        return None, None
    a = dict((t, lp) for t, lp in ref[0][2])
    b = dict((t, lp) for t, lp in cand[0][2])
    shared = [t for t in a if t in b]
    if not shared:
        return None, ref[0][0] == cand[0][0]
    return max(abs(a[t] - b[t]) for t in shared), ref[0][0] == cand[0][0]


def sc_noise(srv, rep, fx):
    """the noise floor of the engine: the SAME fresh prompt three times (cache_prompt false) and one cache-hit continuation; every pair's first difference with its margin. Informative: it is what SRV-5's 0.25 nats has to be read against"""
    summary = {}
    for k in (1, 3):
        m = conv(fx, k)
        fr = [srv.chat(m, 64, logprobs=True, cache_prompt=False) for _ in range(3)]
        srv.chat(conv(fx, k - 1), 8)
        hit = srv.chat(m, 64, logprobs=True)
        runs = {"f1": fr[0], "f2": fr[1], "f3": fr[2], "hit": hit}
        lines = []
        for a_, b_ in (("f1", "f2"), ("f1", "f3"), ("f2", "f3"), ("f1", "hit"), ("f2", "hit"), ("f3", "hit")):
            ident, first, marg, rank = pair_note(runs[a_]["tokens"], runs[b_]["tokens"])
            lines.append("%s~%s: %s" % (a_, b_, "identical" if ident else "diff@%s margin %s rank %s" % (first, None if marg is None else round(marg, 3), rank)))
            fd, top1 = first_token_delta(runs[a_]["tokens"], runs[b_]["tokens"])
            lines[-1] += " (1st token: max|dlogprob| %s, top1 %s)" % (None if fd is None else round(fd, 4), "same" if top1 else "DIFFERENT")
            summary.setdefault("k%d" % k, []).append({"pair": a_ + "~" + b_, "identical": ident, "first": first, "margin": marg, "rank": rank, "first_token_dlogprob": fd, "first_token_top1_same": top1})
        rep.add("noise", "turn%02d" % k, hit, [], "; ".join(lines))
    rep.blobs["noise"] = summary


def sc_slots(srv, rep, fx):
    """slot save / erase / restore (needs --slot-save-path): after the restore the same prompt is served from the restored state (prompt_n small) and continues like the original"""
    m = conv(fx, 3)
    r1 = srv.chat(m, 32, logprobs=True)
    rep.add("slots", "original", r1)
    n_all = (r1["timings"].get("prompt_n") or 0) + (r1["timings"].get("cache_n") or 0)
    st, js = srv.api("/slots/0?action=save", {"filename": "g31-slot0.bin"})
    rep.check("slots", "save", st == 200, "save -> http %s %s" % (st, str(js)[:120]))
    st, js = srv.api("/slots/0?action=erase", {})
    rep.check("slots", "erase", st == 200, "erase -> http %s %s" % (st, str(js)[:120]))
    r2 = srv.chat(m, 32, logprobs=True)
    f = []
    if (r2["timings"].get("cache_n") or 0) > 8:
        f.append("after the erase %s tokens came from a cache" % r2["timings"].get("cache_n"))
    rep.add("slots", "after-erase", r2, f, "prompt_n %s" % r2["timings"].get("prompt_n"))
    st, js = srv.api("/slots/0?action=erase", {})
    st, js = srv.api("/slots/0?action=restore", {"filename": "g31-slot0.bin"})
    rep.check("slots", "restore", st == 200, "restore -> http %s %s" % (st, str(js)[:160]))
    r3 = srv.chat(m, 32, logprobs=True)
    # a hybrid / recurrent model cannot drop the generated tokens from the restored state (no checkpoint is saved in the file), so the same prompt is reprocessed in full on STOCK as well:
    # the rule is the relative one (tools/server_mt_compare.py: OPT prompt_n <= STOCK prompt_n + 8), here only a note
    rep.add("slots", "after-restore", r3, [], "the saved state held %d tokens (prompt + generated)" % n_all)
    ident, first, margin, ok = compare_greedy(r2["tokens"], r3["tokens"])
    rep.check("slots", "restored-vs-fresh", ok, "identical over %d tokens" % len(r2["tokens"]) if ident else "first difference at token %s, margin %s nats" % (first, None if margin is None else round(margin, 3)))


def sc_stream(srv, rep, fx):
    """a streamed chat completion read to the end: data lines, a finish reason, [DONE], usage; the text equals the non-streamed one up to a near tie"""
    import http.client
    m = conv(fx, 2)
    plain = srv.chat(m, 40)
    body = json.dumps({"messages": m, "max_tokens": 40, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    text, done, finish, usage, n = "", False, None, None, 0
    for raw in r.fp:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        ev = json.loads(payload)
        n += 1
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            text += (ch.get("delta") or {}).get("content") or ""
            finish = ch.get("finish_reason") or finish
    c.close()
    f = []
    if not done:
        f.append("no [DONE]")
    if not text.strip():
        f.append("empty streamed text")
    if finish is None:
        f.append("no finish_reason")
    rep.add("stream", "streamed", {"status": r.status, "wall_s": round(time.time() - t0, 3), "timings": {}, "finish": finish}, f, "%d events, %d chars, plain %d chars, usage %s" % (n, len(text), len(plain["content"]), usage))


SCENARIOS = {"prefix": sc_prefix, "agentic": sc_agentic, "branch": sc_branch, "sampling": sc_sampling, "equiv": sc_equiv, "growth": sc_growth, "xcheck": sc_xcheck, "noise": sc_noise, "slots": sc_slots, "stream": sc_stream}


def load_cfg(path):
    return json.load(open(path, "r", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-fixture", "run"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--fixture2")
    ap.add_argument("--scenarios", default="prefix,agentic,branch,alternate,sampling,equiv")
    ap.add_argument("--sizes", default="700,2600,900,6200,1300,500,12500,1000,3200,20000")
    ap.add_argument("--variant", default="A")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--reprocess-allow", type=int, default=0)
    ap.add_argument("--think", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--extra-env", action="append", default=[])
    a = ap.parse_args()
    global MAX_TURNS, REPROCESS_ALLOW, THINK_S
    MAX_TURNS = a.max_turns
    THINK_S = a.think
    REPROCESS_ALLOW = a.reprocess_allow
    cfg = load_cfg(a.config)
    srv = Server(cfg, a.out, a.port, a.extra_arg, dict(kv.split("=", 1) for kv in a.extra_env))
    rc = 0
    try:
        t = srv.start()
        print("server ready after %.1f s" % t)
        root = os.getcwd()
        if a.cmd == "build-fixture":
            build_fixture(srv, a.fixture, [int(x) for x in a.sizes.split(",")], root, a.variant)
        else:
            fx = json.load(open(a.fixture, "r", encoding="utf-8"))
            fx2 = json.load(open(a.fixture2, "r", encoding="utf-8")) if a.fixture2 else None
            rep = Report(os.path.join(a.out, "report.json"))
            for name in a.scenarios.split(","):
                print("--- scenario", name)
                if name == "alternate":
                    sc_alternate(srv, rep, fx, fx2 or fx)
                else:
                    SCENARIOS[name](srv, rep, fx)
                if not srv.alive():
                    rep.fails.append("%s: the server died" % name)
                    break
            extra = {"free_min_smi_mib": srv.sampler.min_free, "free_max_smi_mib": srv.sampler.max_free, "alive": srv.alive()}
            rep.save(extra)
            print("free VRAM (nvidia-smi view) min %s max %s MiB; server alive: %s" % (srv.sampler.min_free, srv.sampler.max_free, srv.alive()))
            if rep.fails:
                print("FAILED expectations:")
                for f in rep.fails:
                    print("  ", f)
                rc = 1
            else:
                print("all expectations met")
    finally:
        srv.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
