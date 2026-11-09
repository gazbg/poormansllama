#!/usr/bin/env python3
r"""Vision harness for llama-server (plan docs/plans/vision-on-demand-plan.md, tasks VOD-0 .. VOD-6). stdlib only (no Pillow: the fixtures are PNGs written here).

  python tools/server_vision.py selftest                               # host only, no GPU, no server
  python tools/server_vision.py fixtures --out DIR [--sides 724,1024,1448]
  python tools/server_vision.py mkcfg --arm base|novis|cpuv|od --exe EXE --out CFG.json [--free-all] [--np 1] [--extra-arg A ...] [--env K=V ...]
  python tools/server_vision.py run --config CFG.json --out DIR [--phase p0] [--sides 724,1024,1448] [--reps 2] [--port 18080]   # GPU: run under tools/with-gpu-lock.ps1
  python tools/server_vision.py table DIR [DIR ...]

Arms (definitions of the plan's acceptance contract):
  base  the R8 recipe + --mmproj, projector resident          novis  base without --mmproj
  cpuv  base + --no-mmproj-offload (CPU vision)               od     base + --mmproj-on-demand (VOD-2 onward)
Fixtures: a coloured square on a grey field ("what colour") and a big seven-segment-style digit string ("what number"), at several side lengths (one image token covers about
32x32 px on Qwen3-VL-style projectors; the real token count of every fixture is MEASURED as usage.prompt_tokens minus a text-only control), plus a 3-image request.
Grading is programmatic only (normalised substring), with a must-fail negative control: the same answer graded against a wrong expectation has to FAIL.
Output: DIR/report.json (every request, log scrapes, VRAM readings), DIR/server.log, DIR/free_mib.txt, DIR/fixtures/*.png.
"""
import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf"
MMPROJ = "C:/Models/ALL/other/qwen-3.6/mmproj-qwen-35b-tq3-BF16.gguf"
MMPROJ_SHA256 = "da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa"

# the R8 recipe of the setup wizard (setup.ps1 $script:RECIPES R8), without the model path
R8_ARGS = ["-c", "262144", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-np", "1", "--hybrid-moe", "--lm-prefill", "--lm-prefill-canonical", "--lm-prefill-fuse",
           "--lm-prefill-prearm", "auto", "--lm-prefill-ubatch", "2048", "--lm-prefill-segment", "32768", "-b", "32768", "--hybrid-moe-prefill-ubatch", "0",
           "-fitt", "16000", "--ctx-checkpoints", "32", "-lv", "4"]

# ----------------------------------------------------------------------------- PNG + fixtures

FONT = {   # 5x7 glyphs
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
COLORS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 30), "purple": (140, 40, 170)}


def png_bytes(w, h, rgb):
    """rgb: bytes/bytearray of w*h*3"""
    rows = b"".join(b"\x00" + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b"")


def png_decode(data):
    """minimal decoder for the PNGs written above (8-bit RGB, filter 0): returns (w, h, rgb bytes)"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + body) & 0xFFFFFFFF, "crc"
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    out = bytearray()
    stride = w * 3 + 1
    for y in range(h):
        assert raw[y * stride] == 0
        out += raw[y * stride + 1:(y + 1) * stride]
    return w, h, bytes(out)


class Canvas:
    def __init__(self, w, h, bg):
        self.w, self.h = w, h
        self.px = bytearray(bytes(bg) * (w * h))

    def rect(self, x0, y0, x1, y1, col):
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.w, x1), min(self.h, y1)
        row = bytes(col) * max(0, x1 - x0)
        for y in range(y0, y1):
            self.px[(y * self.w + x0) * 3:(y * self.w + x1) * 3] = row

    def png(self):
        return png_bytes(self.w, self.h, self.px)


def colour_image(side, name):
    c = Canvas(side, side, (225, 225, 225))
    q = side // 5
    c.rect(side // 2 - q, side // 2 - q, side // 2 + q, side // 2 + q, COLORS[name])
    return c.png()


def digit_image(side, digits):
    c = Canvas(side, side, (255, 255, 255))
    cols = len(digits) * 6 - 1
    cell = max(2, side // (cols + 4))
    x0 = (side - cols * cell) // 2
    y0 = (side - 7 * cell) // 2
    for i, d in enumerate(digits):
        for r, line in enumerate(FONT[d]):
            for k, ch in enumerate(line):
                if ch == "1":
                    c.rect(x0 + (i * 6 + k) * cell, y0 + r * cell, x0 + (i * 6 + k + 1) * cell, y0 + (r + 1) * cell, (0, 0, 0))
    return c.png()


def make_fixtures(sides):
    """list of dicts {name, kind, side, images:[png bytes], question, expect}"""
    fx = []
    cols = list(COLORS)
    for i, side in enumerate(sides):
        col = cols[i % len(cols)]
        digs = str(4821 + 1111 * i)[:4]
        fx.append({"name": "colour-%d" % side, "kind": "colour", "side": side, "images": [colour_image(side, col)],
                   "question": "What is the colour of the square in the image? Answer with one word.", "expect": col})
        fx.append({"name": "digits-%d" % side, "kind": "digits", "side": side, "images": [digit_image(side, digs)],
                   "question": "What number is written in the image? Answer with the digits only.", "expect": digs})
    side = sides[0]
    trio = ["red", "green", "blue"]   # DL-6: a digit trio was misread by the CONTROL itself (5206 -> 5246 at 724 px, 3917 -> 291 at 1024 px); three colour squares are read reliably
    fx.append({"name": "trio-%d" % side, "kind": "trio", "side": side, "images": [colour_image(side, c) for c in trio],
               "question": "The three images each show a coloured square. List the three colours in order, separated by spaces.", "expect": " ".join(trio)})
    return fx


def make_eq_fixtures(sides):
    """make_fixtures() plus (VOD-4, DL-16) the prompts that put a batch of >= 1,024 rows BEHIND an image: a long text after one image (the text batch follows an M-RoPE position jump)"""
    fx = make_fixtures(sides)
    side = sides[0]
    filler = " ".join("Note %d: the apple is on the table." % i for i in range(1, 421))     # ~4,000 tokens of text that has nothing to do with the question
    col = list(COLORS)[0]
    fx.append({"name": "image-then-text-%d" % side, "kind": "longtext", "side": side, "images": [colour_image(side, col)],
               "question": "Read the following notes and ignore them.\n" + filler + "\nNow answer: what is the colour of the square in the image? Answer with one word.", "expect": col})
    # DL-17 control: the same kind of long text WITHOUT an image, and a question the model is unsure about (a spread-out first-token distribution shows arithmetic differences the most). It tells whether a
    # pass-vs-ordinary difference belongs to image rows or to the pass in general (the text pass was bit-near-equal to the ordinary path in G32).
    fx.append({"name": "text-4k", "kind": "textonly", "side": 0, "images": [],
               "question": "Read the following notes and ignore them.\n" + filler + "\nNow pick a number between 1 and 10 and answer with the number only.", "expect": ""})
    return fx


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def grade(answer, expect):
    a, e = norm(answer), norm(expect)
    if re.fullmatch(r"[0-9 ]+", e or ""):
        return all(tok in re.sub(r"[^0-9 ]+", "", a).replace(",", " ").split() or tok in a.replace(" ", "") for tok in e.split())
    return e in a.split() or e in a


def message_for(fx):
    content = []
    for png in fx["images"]:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}})
    content.append({"type": "text", "text": fx["question"]})
    return [{"role": "user", "content": content}]


def text_control(fx):
    return [{"role": "user", "content": fx["question"]}]


# ----------------------------------------------------------------------------- configs

STOCK_ARGS = ["-c", "32768", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-np", "1", "-ngl", "99", "-ncmoe", "37", "-fit", "off", "-lv", "4"]   # the stock placement: no hybrid engine, no layer-major pass


def mkcfg(arm, exe, free_all=False, np_=1, extra_args=(), env=(), wrap=(), stock=False):
    args = ["--model", MODEL]
    if arm != "novis":
        args += ["--mmproj", MMPROJ]
    args += list(STOCK_ARGS if stock else R8_ARGS)
    if np_ != 1:
        args[args.index("-np") + 1] = str(np_)
    if free_all and not stock:
        args.append("--hybrid-moe-prefill-free-all")
    if arm == "cpuv":
        args.append("--no-mmproj-offload")
    if arm == "od":
        args.append("--mmproj-on-demand")
    args += list(extra_args)
    e = {}
    for kv in env:
        k, v = kv.split("=", 1)
        e[k] = v
    root = os.getcwd().replace("\\", "/")
    cfg = {"exe": exe if os.path.isabs(exe) else root + "/" + exe, "args": args, "env": e}
    if wrap:
        cfg["wrap"] = list(wrap)   # a command prefix (e.g. compute-sanitizer and its options), one word per entry
    return cfg


# ----------------------------------------------------------------------------- log scrapes

RE_SIZING = re.compile(r"hybrid-moe: expert cache (\d+) -> (\d+) slots per layer \((.*?)\): free VRAM was (-?\d+) MiB, margin (\d+) MiB \+ (\d+) MiB \(\+ (\d+) MiB slack\)")
RE_COMPUTE = re.compile(r"compute buffer size\s*=\s*([\d.]+) MiB")
RE_VOD = re.compile(r"vod: (.*)")
RE_FREEWATCH = re.compile(r"free-watch[^\n]*?(?:lowest|minimum|min)[^\d]*(\d+) MiB")


def scrape_log(path):
    txt = open(path, "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    out = {"sizing": [], "compute_buffers_mib": [], "vod": [], "free_watch": [], "loaded_mmproj": "loaded multimodal model" in txt, "cuda_errors": []}
    for line in txt.splitlines():
        m = RE_SIZING.search(line)
        if m:
            out["sizing"].append({"before": int(m.group(1)), "after": int(m.group(2)), "why": m.group(3), "free_mib": int(m.group(4)), "margin_mib": int(m.group(5)),
                                  "safety_mib": int(m.group(6)), "slack_mib": int(m.group(7))})
        m = RE_COMPUTE.search(line)
        if m:
            out["compute_buffers_mib"].append({"mib": float(m.group(1)), "line": line.strip()[:200]})
        m = RE_VOD.search(line)
        if m:
            out["vod"].append(m.group(1).strip())
        m = RE_FREEWATCH.search(line)
        if m:
            out["free_watch"].append({"mib": int(m.group(1)), "line": line.strip()[:200]})
        if "CUDA error" in line or "GGML_ASSERT" in line:
            out["cuda_errors"].append(line.strip()[:200])
    return out


def smi_used_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return None


# ----------------------------------------------------------------------------- run

def sha16(path):
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def provenance(exe):
    """binary identity of the run (the wrapper's own gate cannot see it on the server path): exe + the engine DLLs beside it by SHA-256 prefix, and the PROVENANCE.txt of the binary directory"""
    d = os.path.dirname(exe)
    out = {"exe": exe, "files": {n: sha16(os.path.join(d, n)) for n in ("llama-server.exe", "llama.dll", "llama-server-impl.dll", "mtmd.dll", "ggml-cuda.dll", "llama-common.dll")}}
    try:
        out["note"] = open(os.path.join(d, "PROVENANCE.txt"), "rb").read().decode("utf-8", "replace")
    except OSError:
        out["note"] = None
    return out


def cmd_run(a):
    from server_mt import Server, load_cfg
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    sides = [int(x) for x in a.sides.split(",")]
    fixtures = make_fixtures(sides)
    fdir = os.path.join(a.out, "fixtures")
    os.makedirs(fdir, exist_ok=True)
    for f in fixtures:
        for i, png in enumerate(f["images"]):
            open(os.path.join(fdir, "%s-%d.png" % (f["name"], i)), "wb").write(png)
    srv = Server(cfg, a.out, port=a.port)
    rep = {"phase": a.phase, "config": cfg, "mmproj_sha256_expected": MMPROJ_SHA256, "requests": [], "t_ready_s": None, "provenance": provenance(cfg["exe"])}
    ok = True
    try:
        rep["t_ready_s"] = round(srv.start(), 2)
        time.sleep(3.0)
        rep["smi_used_mib_after_ready"] = smi_used_mib()
        has_vision = "--mmproj" in cfg["args"]
        for f in fixtures:
            if not has_vision:
                break
            ctl = srv.chat(text_control(f), max_tokens=4, cache_prompt=False)
            for r in range(a.reps):
                rec = srv.chat(message_for(f), max_tokens=a.max_tokens, cache_prompt=False)
                tk = (rec.get("usage") or {}).get("prompt_tokens")
                ck = (ctl.get("usage") or {}).get("prompt_tokens")
                rec.update({"fixture": f["name"], "kind": f["kind"], "side": f["side"], "n_images": len(f["images"]), "rep": r, "expect": f["expect"],
                            "image_tokens": (tk - ck) if (tk and ck) else None, "graded_ok": grade(rec.get("content"), f["expect"]),
                            "negative_control_fails": not grade(rec.get("content"), "zzqxj 000000"), "tokens": None})
                rec.pop("tokens", None)
                rep["requests"].append(rec)
                ok = ok and rec["status"] == 200
                print("%-14s rep %d  status %s  image_tokens %s  prompt_ms %s  wall %.2fs  answer %r  ok=%s" % (f["name"], r, rec["status"], rec["image_tokens"],
                      (rec.get("timings") or {}).get("prompt_ms"), rec["wall_s"], rec.get("content", "")[:40], rec["graded_ok"]))
        time.sleep(3.0)
        rep["smi_used_mib_idle_end"] = smi_used_mib()
    finally:
        srv.stop()
    rep["log"] = scrape_log(os.path.join(a.out, "server.log"))
    mn = srv.sampler.min_free if srv.sampler else None
    rep["smi_free_mib_min"], rep["smi_free_mib_max"] = mn, (srv.sampler.max_free if srv.sampler else None)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("report:", os.path.join(a.out, "report.json"))
    sys.exit(0 if ok else 1)


def cmd_table(a):
    for d in a.dirs:
        p = os.path.join(d, "report.json")
        if not os.path.exists(p):
            print(d, "no report.json")
            continue
        r = json.load(open(p))
        lg = r.get("log", {})
        print("== %s  ready %ss  smi used after ready %s / idle end %s MiB  mmproj loaded %s" % (d, r.get("t_ready_s"), r.get("smi_used_mib_after_ready"), r.get("smi_used_mib_idle_end"), lg.get("loaded_mmproj")))
        for s in lg.get("sizing", []):
            print("   sizing: cache %d -> %d slots (%s) free %d MiB margin %d + %d (+%d slack)" % (s["before"], s["after"], s["why"], s["free_mib"], s["margin_mib"], s["safety_mib"], s["slack_mib"]))
        for c in lg.get("compute_buffers_mib", [])[:6]:
            print("   compute buffer %.2f MiB  %s" % (c["mib"], c["line"][:90]))
        for q in r.get("requests", []):
            print("   %-14s rep %d image_tokens %-5s prompt_ms %-9s wall %-6s ok=%s ctl_fails=%s" % (q["fixture"], q["rep"], q.get("image_tokens"), (q.get("timings") or {}).get("prompt_ms"), q["wall_s"], q["graded_ok"], q["negative_control_fails"]))
        for v in lg.get("vod", [])[:40]:
            print("   vod:", v)


# ----------------------------------------------------------------------------- vod: statistics

RE_V_ENC = re.compile(r"encode slot=\d+ chunks=(\d+) tokens=(\d+) res=(-?\d+) t_encode_ms=([\d.]+)")
RE_V_DEC = re.compile(r"decode_chunk slot=\d+ idx=\d+ tokens=(\d+) res=(-?\d+) t_decode_ms=([\d.]+)")
RE_V_INIT = re.compile(r"projector init t_init_ms=([\d.]+) device_free_before_mib=([\d.]+) device_free_after_mib=([\d.]+) delta_mib=(-?[\d.]+) estimate_mib=([\d.]+)")


def vod_stats(vod_lines):
    """per image-token count: n, first / median / min of the encode and of the ordinary-path decode (ms), and the decode rate in tokens/s of the median"""
    enc, dec, init = {}, {}, None
    for v in vod_lines:
        m = RE_V_ENC.search(v)
        if m:
            enc.setdefault(int(m.group(2)), []).append(float(m.group(4)))
        m = RE_V_DEC.search(v)
        if m:
            dec.setdefault(int(m.group(1)), []).append(float(m.group(3)))
        m = RE_V_INIT.search(v)
        if m:
            init = {"t_init_ms": float(m.group(1)), "free_before_mib": float(m.group(2)), "free_after_mib": float(m.group(3)), "delta_mib": float(m.group(4)), "estimate_mib": float(m.group(5))}

    def med(x):
        y = sorted(x)
        return y[len(y) // 2]
    rows = []
    for tok in sorted(set(enc) | set(dec)):
        e, d = enc.get(tok, []), dec.get(tok, [])
        rows.append({"tokens": tok, "n": max(len(e), len(d)), "enc_first": e[0] if e else None, "enc_med": med(e) if e else None, "enc_min": min(e) if e else None,
                     "dec_first": d[0] if d else None, "dec_med": med(d) if d else None, "dec_min": min(d) if d else None,
                     "dec_tok_s": (tok / (med(d) / 1000.0)) if d else None})
    return init, rows


def cmd_compare(a):
    for d in a.dirs:
        p = os.path.join(d, "report.json")
        if not os.path.exists(p):
            print(d, "no report.json")
            continue
        r = json.load(open(p))
        init, rows = vod_stats(r["log"].get("vod", []))
        print("== %s" % d)
        if init:
            print("   projector init: %.1f ms, device free %.1f -> %.1f MiB (delta %.1f MiB; estimate %.1f MiB incl. host)" % (init["t_init_ms"], init["free_before_mib"], init["free_after_mib"], init["delta_mib"], init["estimate_mib"]))
        print("   %-7s %-3s %-26s %-26s %s" % ("tokens", "n", "encode ms first/med/min", "decode ms first/med/min", "decode tok/s (median)"))
        f = lambda x: "-" if x is None else "%.1f" % x
        for w in rows:
            print("   %-7d %-3d %-26s %-26s %s" % (w["tokens"], w["n"], "%s / %s / %s" % (f(w["enc_first"]), f(w["enc_med"]), f(w["enc_min"])),
                                                   "%s / %s / %s" % (f(w["dec_first"]), f(w["dec_med"]), f(w["dec_min"])), f(w["dec_tok_s"])))
        by = {}
        for q in r.get("requests", []):
            by.setdefault(q["fixture"], []).append(q)
        print("   request prompt_ms (median) / graded ok:")
        for k, v in by.items():
            pm = sorted((x.get("timings") or {}).get("prompt_ms") or 0 for x in v)
            print("     %-14s image_tokens %-5s prompt_ms %-9.1f ok %s" % (k, v[0].get("image_tokens"), pm[len(pm) // 2], all(x["graded_ok"] for x in v)))



# ----------------------------------------------------------------------------- acceptance session (VOD-2): one fixed request sequence per arm

def filler(n, start=0):
    """deterministic filler text, ~25 tokens per sentence"""
    return " ".join("Entry %d: the measured value of channel %d was %d units at station %d." % (i, i % 17, (i * 7919) % 1000, i % 5) for i in range(start, start + n))


def logtext(out):
    try:
        return open(os.path.join(out, "server.log"), "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
    except OSError:
        return ""


def log_counts(out):
    t = logtext(out)
    images = sum(int(m) for m in re.findall(r"vod: encode slot=\d+ chunks=(\d+)", t)) + sum(int(m) for m in re.findall(r"vod: window: (\d+) image chunks", t))
    return {"images_encoded": images, "windows": len(re.findall(r"vod: window:", t)), "windows_failed": len(re.findall(r"vod: window:.*FAILED", t)), "no_room": len(re.findall(r"lm-prefill: no room", t)),
            "pass_armed": len(re.findall(r"lm-prefill: armed for", t)), "pass_declined_ordinary": len(re.findall(r"this batch takes the ordinary path", t)),
            "vod_waiting": len(re.findall(r"vod: waiting for", t)), "cache_resizes": len(re.findall(r"cache resized to", t))}


def rec_short(rec):
    tm = rec.get("timings") or {}
    return {"status": rec.get("status"), "content": rec.get("content"), "finish": rec.get("finish"), "wall_s": rec.get("wall_s"), "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "predicted_n": tm.get("predicted_n"),
            "prompt_ms": tm.get("prompt_ms"), "prompt_per_second": tm.get("prompt_per_second"), "predicted_per_second": tm.get("predicted_per_second"), "error": rec.get("error")}


def cmd_acc(a):
    from server_mt import Server, load_cfg
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    fixtures = make_fixtures([724, 1024, 1448])
    srv = Server(cfg, a.out, port=a.port)
    rep = {"phase": "acc", "config": cfg, "provenance": provenance(cfg["exe"]), "steps": [], "idle": {}}
    ok = True
    has_vision = "--mmproj" in cfg["args"]

    def snap(name):
        time.sleep(3.0)
        rep["idle"][name] = {"smi_used_mib": smi_used_mib(), "counts": log_counts(a.out)}

    last_images = [0]

    def step(name, rec, **extra):
        r = rec_short(rec)
        r.update(extra)
        r["name"] = name
        r["counts_after"] = log_counts(a.out)
        r["images_in_step"] = r["counts_after"]["images_encoded"] - last_images[0]   # image chunks encoded since the previous step (BASE: lazy encodes, OD: windows), needs LLAMA_VOD_LOG=1
        last_images[0] = r["counts_after"]["images_encoded"]
        rep["steps"].append(r)
        print("%-24s status %s  prompt_n %-5s cache_n %-5s pred_n %-4s imgs %-2s pp %-8s tg %-7s  %r" % (name, r["status"], r["prompt_n"], r["cache_n"], r["predicted_n"], r["images_in_step"],
              ("%.0f" % r["prompt_per_second"]) if r["prompt_per_second"] else "-", ("%.1f" % r["predicted_per_second"]) if r["predicted_per_second"] else "-", (r["content"] or "")[:40]))
        return r

    try:
        rep["t_ready_s"] = round(srv.start(), 2)
        snap("ready")
        # 1. text smoke: a short, a mid and a long prompt (the long one engages the layer-major pass; its PP / TG are the text-turn numbers)
        texts = [("text-short", "Reply with the single word: ready", 0, 16), ("text-mid", "Reply with the single word: done", 60, 16),
                 ("text-long", "Count from 1 to 60 separated by commas, nothing else.", 240, 64)]
        for name, q, n_sent, mt in texts:
            content = (filler(n_sent) + "\n\n" if n_sent else "") + q
            r = step(name, srv.chat([{"role": "user", "content": content}], max_tokens=mt, cache_prompt=False))
            ok = ok and r["status"] == 200
        snap("after_text")
        # 2. the fixtures (cold prefill each)
        if has_vision:
            for f in fixtures:
                rec = srv.chat(message_for(f), max_tokens=a.max_tokens, cache_prompt=False)
                r = step("fix-" + f["name"], rec, expect=f["expect"], graded_ok=grade(rec.get("content"), f["expect"]))
                ok = ok and r["status"] == 200 and r["graded_ok"]
            snap("after_fixtures")
            # 3. a three-turn image conversation with the prompt cache on: turn 2 must hit the cache with ZERO encodes, turn 3 (a new image) must open exactly one window
            f_a, f_b = fixtures[1], fixtures[4]          # digits-724 then colour-1448
            hist = message_for(f_a)
            rec1 = srv.chat(hist, max_tokens=16, cache_prompt=True)
            step("mt-turn1-image", rec1, expect=f_a["expect"], graded_ok=grade(rec1.get("content"), f_a["expect"]))
            hist = hist + [{"role": "assistant", "content": rec1.get("content") or ""}, {"role": "user", "content": "Add 1 to that number. Answer with the digits only."}]
            n_before = log_counts(a.out)["windows"]
            rec2 = srv.chat(hist, max_tokens=16, cache_prompt=True)
            step("mt-turn2-text", rec2, windows_in_turn=log_counts(a.out)["windows"] - n_before, expect=str(int(f_a["expect"]) + 1), graded_ok=grade(rec2.get("content"), str(int(f_a["expect"]) + 1)))
            hist = hist + [{"role": "assistant", "content": rec2.get("content") or ""}, {"role": "user", "content": message_for(f_b)[0]["content"]}]
            n_before = log_counts(a.out)["windows"]
            rec3 = srv.chat(hist, max_tokens=16, cache_prompt=True)
            step("mt-turn3-new-image", rec3, windows_in_turn=log_counts(a.out)["windows"] - n_before, expect=f_b["expect"], graded_ok=grade(rec3.get("content"), f_b["expect"]))
            snap("after_mt")
    finally:
        srv.stop()
    rep["log"] = scrape_log(os.path.join(a.out, "server.log"))
    rep["log_counts_final"] = log_counts(a.out)
    rep["smi_free_mib_min"] = srv.sampler.min_free if srv.sampler else None
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("report:", os.path.join(a.out, "report.json"))
    sys.exit(0 if ok else 1)


def cmd_acc_compare(a):
    """identity table of acceptance reports: the first dir is the reference"""
    reps = [(d, json.load(open(os.path.join(d, "report.json")))) for d in a.dirs]
    ref_d, ref = reps[0]
    print("reference: %s" % ref_d)
    for d, r in reps[1:]:
        print("== %s against %s" % (d, ref_d))
        by_ref = {s["name"]: s for s in ref["steps"]}
        n_same = 0
        n_all = 0
        for s in r["steps"]:
            q = by_ref.get(s["name"])
            if q is None:
                continue
            n_all += 1
            same_txt = s["content"] == q["content"]
            same_cnt = (s["prompt_n"], s["cache_n"], s["predicted_n"]) == (q["prompt_n"], q["cache_n"], q["predicted_n"])
            # the image-encode counts come from the LLAMA_VOD_LOG lines: compared only when both runs have them (a pristine binary has none)
            same_img = (s.get("images_in_step") == q.get("images_in_step")) if (r["log_counts_final"]["images_encoded"] > 0 and ref["log_counts_final"]["images_encoded"] > 0) else True
            same_cnt = same_cnt and same_img
            n_same += 1 if (same_txt and same_cnt) else 0
            pp = (s["prompt_per_second"] / q["prompt_per_second"]) if s.get("prompt_per_second") and q.get("prompt_per_second") else None
            tg = (s["predicted_per_second"] / q["predicted_per_second"]) if s.get("predicted_per_second") and q.get("predicted_per_second") else None
            print("  %-24s text %-5s counts %-5s (%s | %s)  pp ratio %s  tg ratio %s" % (s["name"], "same" if same_txt else "DIFF", "same" if same_cnt else "DIFF",
                  (s["prompt_n"], s["cache_n"], s["predicted_n"], s.get("images_in_step")), (q["prompt_n"], q["cache_n"], q["predicted_n"], q.get("images_in_step")), ("%.3f" % pp) if pp else "-", ("%.3f" % tg) if tg else "-"))
        print("  identical (text and counts): %d of %d steps" % (n_same, n_all))
        for k in ("ready", "after_text", "after_fixtures", "after_mt"):
            if k in r["idle"] and k in ref["idle"]:
                print("  idle %-15s smi used %s vs %s MiB   counts %s" % (k, r["idle"][k]["smi_used_mib"], ref["idle"][k]["smi_used_mib"], r["idle"][k]["counts"]))
        print("  free-watch lines:", [w["mib"] for w in r["log"].get("free_watch", [])][-4:], "reference:", [w["mib"] for w in ref["log"].get("free_watch", [])][-4:])


# ----------------------------------------------------------------------------- fault injection and soak (VOD-2: VOD-5 / VOD-6 criteria)

def health_poller(srv):
    """polls /health every 0.2 s on its own thread; stop with .stop = True. Records how many polls failed and the slowest answer."""
    import threading
    import urllib.request

    class Poller(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.stop, self.n, self.bad, self.max_ms = False, 0, 0, 0.0

        def run(self):
            while not self.stop:
                t0 = time.time()
                try:
                    st = urllib.request.urlopen(srv.base + "/health", timeout=5).status
                except Exception:
                    st = None
                self.max_ms = max(self.max_ms, (time.time() - t0) * 1000.0)
                self.n += 1
                self.bad += 0 if st == 200 else 1
                time.sleep(0.2)
    p = Poller()
    p.start()
    return p


def cmd_fault(a):
    """a.mode: room (LLAMA_VOD_FAIL_ROOM=N in the config env: the request waits and then succeeds), encode (LLAMA_VOD_FAIL_ENCODE=1: that request is HTTP 500, the next one succeeds),
    timeout (LLAMA_VOD_FAIL_ROOM=1000 and --mmproj-wait-timeout 3: HTTP 500 after ~3 s, the server lives)"""
    from server_mt import Server, load_cfg
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    f = make_fixtures([724])[1]      # digits-724
    srv = Server(cfg, a.out, port=a.port)
    rep = {"phase": "fault", "mode": a.mode, "config": cfg, "provenance": provenance(cfg["exe"]), "steps": [], "checks": []}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        rep["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        print(("ok   " if cond else "FAIL ") + name + ((" (" + detail + ")") if detail else ""))
    try:
        rep["t_ready_s"] = round(srv.start(), 2)
        time.sleep(3.0)
        rep["smi_used_ready"] = smi_used_mib()
        poll = health_poller(srv)
        r1 = srv.chat(message_for(f), max_tokens=a.max_tokens, cache_prompt=False)
        r2 = srv.chat(message_for(f), max_tokens=a.max_tokens, cache_prompt=False)
        r3 = srv.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=8, cache_prompt=False)
        poll.stop = True
        time.sleep(0.4)
        time.sleep(3.0)
        rep["smi_used_end"] = smi_used_mib()
        rep["health"] = {"polls": poll.n, "bad": poll.bad, "max_ms": round(poll.max_ms, 1)}
        for name, r in (("img-1", r1), ("img-2", r2), ("text", r3)):
            d = rec_short(r)
            d["name"] = name
            rep["steps"].append(d)
            print("%-8s status %s wall %.2fs %r %s" % (name, d["status"], d["wall_s"] or 0, (d["content"] or "")[:30], (d["error"] or "") if d["status"] != 200 else ""))
        lc = log_counts(a.out)
        rep["log_counts"] = lc
        right = grade(r1.get("content"), f["expect"])
        check("server alive after the scenario", srv.alive())
        check("text request after the scenario answered", r3.get("status") == 200)
        if a.mode == "room":
            check("request 1 succeeded after waiting and the answer is right", r1.get("status") == 200 and right, "wall %.2f s" % r1["wall_s"])
            check("request 1 waited (WARN lines + slow)", lc["vod_waiting"] >= 1 and r1["wall_s"] >= 2.0, "waiting lines %d" % lc["vod_waiting"])
            check("request 2 did not wait", r2.get("status") == 200 and r2["wall_s"] < 0.8 * r1["wall_s"] + 0.5, "wall %.2f s" % r2["wall_s"])
            check("/health answered during the wait", poll.bad == 0 and poll.n >= 5, "polls %d bad %d max %.0f ms" % (poll.n, poll.bad, poll.max_ms))
        elif a.mode == "encode":
            check("request 1 failed with an HTTP 500", r1.get("status") == 500, "status %s" % r1.get("status"))
            check("request 2 succeeded and the answer is right", r2.get("status") == 200 and grade(r2.get("content"), f["expect"]))
            check("the failed window is logged and the projector was parked again", lc["windows_failed"] >= 1 and lc["windows"] >= 2, "windows %d failed %d" % (lc["windows"], lc["windows_failed"]))
        elif a.mode == "timeout":
            check("request 1 failed with an HTTP 500 after about the timeout", r1.get("status") == 500 and 2.5 <= r1["wall_s"] <= 9.0, "status %s wall %.2f s" % (r1.get("status"), r1["wall_s"]))
            check("request 2 failed too (the injection counter is not used up)", r2.get("status") == 500)
            check("/health answered during the waits", poll.bad == 0, "polls %d bad %d" % (poll.n, poll.bad))
    finally:
        srv.stop()
    rep["log"] = scrape_log(os.path.join(a.out, "server.log"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("report:", os.path.join(a.out, "report.json"))
    sys.exit(0 if ok else 1)


def cmd_soak(a):
    """a.n mixed text / image requests (seeded), a share of them cancelled by closing the connection mid-request; then the same short requests as at the start: the idle memory must be the same"""
    import http.client
    import random
    from server_mt import Server, load_cfg
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    fixtures = make_fixtures([724, 1024])
    singles = [f for f in fixtures if f["kind"] != "trio"]
    trio = [f for f in fixtures if f["kind"] == "trio"][0]
    srv = Server(cfg, a.out, port=a.port)
    rng = random.Random(a.seed)
    rep = {"phase": "soak", "n": a.n, "seed": a.seed, "config": cfg, "provenance": provenance(cfg["exe"]), "kinds": {}, "idle": {}}
    ok = True
    n_ok = n_bad = n_cancel = n_graded = n_graded_ok = 0
    errors = []

    def body_for(messages, max_tokens):
        return {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": 1, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}

    def cancel(body, delay):
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        time.sleep(delay)
        conn.close()

    def settle():
        for _ in range(2):
            srv.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=8, cache_prompt=False)
        time.sleep(3.0)
        return smi_used_mib()
    try:
        rep["t_ready_s"] = round(srv.start(), 2)
        rep["idle"]["start"] = settle()
        t_begin = time.time()
        for i in range(a.n):
            r = rng.random()
            do_cancel = rng.random() < a.cancel_rate
            if r < 0.40:
                n_sent = rng.choice([0, 0, 20, 60, 120])
                content = (filler(n_sent, start=i * 7) + "\n\n" if n_sent else "") + "Reply with the single word: ok"
                msgs, kind, fx = [{"role": "user", "content": content}], "text", None
            elif r < 0.92:
                fx = rng.choice(singles)
                msgs, kind = message_for(fx), "image"
            else:
                fx = trio
                msgs, kind = message_for(fx), "image3"
            body = body_for(msgs, 16)
            rep["kinds"][kind] = rep["kinds"].get(kind, 0) + 1
            if do_cancel:
                cancel(body, rng.uniform(0.05, 0.9))
                n_cancel += 1
                continue
            rec = srv.chat(msgs, max_tokens=16, cache_prompt=False)
            if rec["status"] == 200:
                n_ok += 1
                if fx is not None:
                    n_graded += 1
                    n_graded_ok += 1 if grade(rec.get("content"), fx["expect"]) else 0
            else:
                n_bad += 1
                errors.append({"i": i, "kind": kind, "status": rec["status"], "error": str(rec.get("error"))[:200]})
            if (i + 1) % 20 == 0:
                print("  %d / %d requests, %d cancelled, %d failed, %.0f s" % (i + 1, a.n, n_cancel, n_bad, time.time() - t_begin))
        rep["idle"]["end"] = settle()
        rep["soak_s"] = round(time.time() - t_begin, 1)
    finally:
        srv.stop()
    lc = log_counts(a.out)
    rep["log_counts"] = lc
    rep["log"] = scrape_log(os.path.join(a.out, "server.log"))
    rep["n_ok"], rep["n_failed"], rep["n_cancelled"], rep["graded"], rep["graded_ok"], rep["errors"] = n_ok, n_bad, n_cancel, n_graded, n_graded_ok, errors[:20]

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("ok   " if cond else "FAIL ") + name + ((" (" + detail + ")") if detail else ""))
    check("no failed request", n_bad == 0, "%d ok, %d failed, %d cancelled of %d" % (n_ok, n_bad, n_cancel, a.n))
    check("every graded image answer is right", n_graded_ok == n_graded, "%d of %d" % (n_graded_ok, n_graded))
    check("no CUDA error / assert in the log", not rep["log"]["cuda_errors"], str(rep["log"]["cuda_errors"][:2]))
    check("idle memory at the end equals the start (within 64 MiB)", rep["idle"]["start"] is not None and rep["idle"]["end"] is not None and abs(rep["idle"]["end"] - rep["idle"]["start"]) < 64,
          "%s -> %s MiB (nvidia-smi)" % (rep["idle"]["start"], rep["idle"]["end"]))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("report:", os.path.join(a.out, "report.json"), " windows:", lc["windows"], " failed windows:", lc["windows_failed"], " waits:", lc["vod_waiting"])
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------- equivalence of two prefill paths (VOD-4 / VOD-5): per-token logprobs of fixed image prompts

EQ_QUESTIONS = [("answer", None, 16), ("describe", "Describe the image in one sentence.", 64)]   # 64 tokens: the length of the contract (VOD-7)


def cmd_eq(a):
    """greedy 16 / 48 token continuations with top-5 logprobs for every fixture (cache_prompt false), saved per token: report.json['eq'][name] = [[token, logprob, [[tok, lp] ...]] ...]"""
    from server_mt import Server, load_cfg
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    fixtures = make_eq_fixtures([int(x) for x in a.sides.split(",")])
    srv = Server(cfg, a.out, port=a.port)
    rep = {"phase": "eq", "config": cfg, "provenance": provenance(cfg["exe"]), "eq": {}, "status": {}}
    ok = True
    try:
        rep["t_ready_s"] = round(srv.start(), 2)
        time.sleep(2.0)
        for f in fixtures:
            for qname, qtext, mt in EQ_QUESTIONS:
                msgs = message_for(f)
                if qtext is not None:
                    msgs[0]["content"][-1]["text"] = qtext
                rec = srv.chat(msgs, max_tokens=mt, cache_prompt=False, logprobs=True)
                key = "%s/%s" % (f["name"], qname)
                rep["eq"][key] = [[t[0], t[1], [[x[0], x[1]] for x in t[2]]] for t in (rec.get("tokens") or [])]
                rep["status"][key] = {"status": rec.get("status"), "content": rec.get("content"), "prompt_n": (rec.get("timings") or {}).get("prompt_n")}
                ok = ok and rec.get("status") == 200
                print("%-24s status %s  %d tokens  %r" % (key, rec.get("status"), len(rep["eq"][key]), (rec.get("content") or "")[:50]))
        time.sleep(1.0)
    finally:
        srv.stop()
    rep["log"] = scrape_log(os.path.join(a.out, "server.log"))
    rep["log_counts"] = log_counts(a.out)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("report:", os.path.join(a.out, "report.json"), " passes:", len(re.findall(r"lm-prefill: pass \d+:", logtext(a.out))), " ordinary-path embedding batches:", len(re.findall(r"carries embeddings", logtext(a.out))))
    sys.exit(0 if ok else 1)


def eq_compare_pair(ref, cand, near_tie=0.25):
    """per prompt: first-token top-1 logprob delta, identical prefix length, and for the first differing token the reference's logprob margin between its pick and the candidate's pick"""
    rows = []
    for key in sorted(ref):
        r, c = ref[key], cand.get(key)
        if not c or not r:
            rows.append((key, None, None, None, "missing"))
            continue
        d_first = abs(r[0][1] - c[0][1])
        n = min(len(r), len(c))
        k = 0
        while k < n and r[k][0] == c[k][0]:
            k += 1
        if k == n and len(r) == len(c):
            rows.append((key, d_first, len(r), None, "identical"))
        else:
            if k < n:
                top = {x[0]: x[1] for x in r[k][2]}
                margin = abs(r[k][1] - top[c[k][0]]) if c[k][0] in top else None
            else:
                margin = None
            rows.append((key, d_first, k, margin, "near-tie flip" if (margin is not None and margin < near_tie) else "FLIP"))
    return rows


def cmd_eq_compare(a):
    reps = [(d, json.load(open(os.path.join(d, "report.json")))["eq"]) for d in a.dirs]
    ref_d, ref = reps[0]
    print("reference: %s" % ref_d)
    ok = True
    for d, cand in reps[1:]:
        rows = eq_compare_pair(ref, cand)
        n_id = sum(1 for r in rows if r[4] == "identical")
        n_nt = sum(1 for r in rows if r[4] == "near-tie flip")
        n_fl = sum(1 for r in rows if r[4] not in ("identical", "near-tie flip"))
        firsts = sorted(r[1] for r in rows if r[1] is not None)
        print("== %s: %d prompts: %d identical, %d near-tie flips (< 0.25 nats at the first difference), %d other; first-token |dlogprob| median %.4f max %.4f" % (
            d, len(rows), n_id, n_nt, n_fl, firsts[len(firsts) // 2] if firsts else float("nan"), firsts[-1] if firsts else float("nan")))
        for key, d1, k, margin, verdict in rows:
            if verdict != "identical" or a.verbose:
                print("   %-26s first-token delta %s  identical prefix %s tokens  margin %s  %s" % (key, "%.4f" % d1 if d1 is not None else "-", k, ("%.3f" % margin) if margin is not None else "-", verdict))
        ok = ok and n_fl == 0 and (not firsts or firsts[-1] <= 0.25)
    sys.exit(0 if ok else 1)

# ----------------------------------------------------------------------------- selftest

def cmd_selftest(_a):
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + name)
        fails += 0 if cond else 1
    png = colour_image(64, "red")
    w, h, rgb = png_decode(png)
    check("png round trip size", (w, h) == (64, 64))
    check("png centre is red", rgb[(32 * 64 + 32) * 3:(32 * 64 + 32) * 3 + 3] == bytes(COLORS["red"]))
    check("png corner is grey", rgb[0:3] == bytes((225, 225, 225)))
    d = digit_image(200, "4821")
    w, h, rgb = png_decode(d)
    black = sum(1 for i in range(0, len(rgb), 3) if rgb[i] == 0)
    check("digit image has ink and background", 500 < black < w * h // 2)
    check("digit glyph table is 5x7", all(len(g) == 7 and all(len(r) == 5 for r in g) for g in FONT.values()) and len(FONT) == 10)
    check("grade colour right", grade("Red.", "red"))
    check("grade colour in sentence", grade("The square is blue", "blue"))
    check("grade colour wrong", not grade("green", "red"))
    check("grade digits exact", grade("4821", "4821"))
    check("grade digits in text", grade("The number is 4821.", "4821"))
    check("grade digits wrong", not grade("4812", "4821"))
    check("grade trio", grade("Red, green, blue.", "red green blue"))
    check("grade trio wrong order", not grade("green red blue", "red green blue"))
    check("negative control fails as designed", not grade("red", "zzqxj 000000") and not grade("4821", "zzqxj 000000"))
    fx = make_fixtures([128, 256])
    check("fixture set", [f["name"] for f in fx] == ["colour-128", "digits-128", "colour-256", "digits-256", "trio-128"] and len(fx[-1]["images"]) == 3)
    m = message_for(fx[0])
    check("message has image then question", m[0]["content"][0]["type"] == "image_url" and m[0]["content"][-1]["type"] == "text" and m[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
    c = mkcfg("od", "build/x.exe", free_all=True)
    check("mkcfg od", "--mmproj-on-demand" in c["args"] and "--mmproj" in c["args"] and "--hybrid-moe-prefill-free-all" in c["args"] and c["args"][:2] == ["--model", MODEL])
    check("mkcfg novis has no mmproj", "--mmproj" not in mkcfg("novis", "x.exe")["args"])
    check("mkcfg cpuv", "--no-mmproj-offload" in mkcfg("cpuv", "x.exe")["args"])
    line = "srv  hybrid-moe: expert cache 0 -> 61 slots per layer (start): free VRAM was 21000 MiB, margin 16000 MiB + 160 MiB (+ 128 MiB slack), 12.0 ms"
    mm = RE_SIZING.search(line)
    check("sizing line parses", bool(mm) and mm.group(2) == "61" and mm.group(5) == "16000")
    check("compute buffer line parses", RE_COMPUTE.search("reserve_compute_meta:     CUDA0 compute buffer size =   123.45 MiB").group(1) == "123.45")
    init, rows = vod_stats(["projector init t_init_ms=609.6 device_free_before_mib=17429.0 device_free_after_mib=16317.0 delta_mib=1112.0 estimate_mib=1134.0 (unit: cudaMemGetInfo of the first GPU)",
                            "encode slot=0 chunks=1 tokens=529 res=0 t_encode_ms=271.95", "decode_chunk slot=0 idx=4 tokens=529 res=0 t_decode_ms=1452.73",
                            "encode slot=0 chunks=1 tokens=529 res=0 t_encode_ms=88.43", "decode_chunk slot=0 idx=4 tokens=529 res=0 t_decode_ms=1440.64",
                            "encode slot=0 chunks=1 tokens=529 res=0 t_encode_ms=86.87", "decode_chunk slot=0 idx=4 tokens=529 res=0 t_decode_ms=1289.95"])
    check("vod stats parse", init is not None and init["delta_mib"] == 1112.0 and len(rows) == 1 and rows[0]["n"] == 3 and rows[0]["enc_min"] == 86.87 and rows[0]["enc_med"] == 88.43 and abs(rows[0]["dec_tok_s"] - 529 / 1.44064) < 0.01)
    t = filler(3)
    check("filler is deterministic", t == filler(3) and t.startswith("Entry 0:") and filler(3, 2).startswith("Entry 2:"))
    check("rec_short picks the timing fields", rec_short({"status": 200, "content": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 1}, "timings": {"prompt_n": 5, "cache_n": 0, "predicted_n": 1}})["prompt_n"] == 5)
    ra = {"p/a": [["x", -0.1, [["x", -0.1], ["y", -1.5]]], ["a", -0.2, [["a", -0.2], ["b", -0.4]]]]}
    check("eq compare identical", eq_compare_pair(ra, {"p/a": [["x", -0.1, []], ["a", -0.2, []]]})[0][4] == "identical")
    check("eq compare near-tie flip", eq_compare_pair(ra, {"p/a": [["x", -0.12, []], ["b", -0.4, []]]})[0][4] == "near-tie flip")
    check("eq compare real flip", eq_compare_pair(ra, {"p/a": [["y", -1.5, []], ["b", -0.4, []]]})[0][4] == "FLIP")
    print("selftest:", "PASS" if fails == 0 else "FAIL (%d)" % fails)
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    p = sp.add_parser("fixtures")
    p.add_argument("--out", required=True)
    p.add_argument("--sides", default="724,1024,1448")
    p = sp.add_parser("mkcfg")
    p.add_argument("--arm", required=True, choices=["base", "novis", "cpuv", "od"])
    p.add_argument("--exe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--free-all", action="store_true")
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--extra-arg", action="append", default=[])
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--wrap", action="append", default=[])
    p.add_argument("--stock", action="store_true", help="the stock placement (-ngl 99 -ncmoe 37 -fit off) instead of the R8 hybrid recipe")
    p = sp.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--phase", default="p0")
    p.add_argument("--sides", default="724,1024,1448")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--port", type=int, default=18080)
    p = sp.add_parser("table")
    p.add_argument("dirs", nargs="+")
    p = sp.add_parser("compare")
    p.add_argument("dirs", nargs="+")
    p = sp.add_parser("acc")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--port", type=int, default=18080)
    p = sp.add_parser("acc-compare")
    p.add_argument("dirs", nargs="+")
    p = sp.add_parser("eq")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sides", default="724,1024,1448")
    p.add_argument("--port", type=int, default=18080)
    p = sp.add_parser("eq-compare")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--verbose", action="store_true")
    p = sp.add_parser("fault")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True, choices=["room", "encode", "timeout"])
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--port", type=int, default=18080)
    p = sp.add_parser("soak")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--cancel-rate", type=float, default=0.10)
    p.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "fixtures":
        os.makedirs(a.out, exist_ok=True)
        for f in make_fixtures([int(x) for x in a.sides.split(",")]):
            for i, png in enumerate(f["images"]):
                open(os.path.join(a.out, "%s-%d.png" % (f["name"], i)), "wb").write(png)
            print(f["name"], "expect", f["expect"])
    elif a.cmd == "mkcfg":
        cfg = mkcfg(a.arm, a.exe, a.free_all, a.np, a.extra_arg, a.env, a.wrap, a.stock)
        json.dump(cfg, open(a.out, "w"), indent=1)
        print(a.out, a.arm, "args:", len(cfg["args"]))
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "table":
        cmd_table(a)
    elif a.cmd == "compare":
        cmd_compare(a)
    elif a.cmd == "acc":
        cmd_acc(a)
    elif a.cmd == "acc-compare":
        cmd_acc_compare(a)
    elif a.cmd == "eq":
        cmd_eq(a)
    elif a.cmd == "eq-compare":
        cmd_eq_compare(a)
    elif a.cmd == "fault":
        cmd_fault(a)
    elif a.cmd == "soak":
        cmd_soak(a)


if __name__ == "__main__":
    main()
