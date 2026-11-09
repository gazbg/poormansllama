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

def mkcfg(arm, exe, free_all=False, np_=1, extra_args=(), env=()):
    args = ["--model", MODEL]
    if arm != "novis":
        args += ["--mmproj", MMPROJ]
    args += list(R8_ARGS)
    if np_ != 1:
        args[args.index("-np") + 1] = str(np_)
    if free_all:
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
    return {"exe": exe if os.path.isabs(exe) else root + "/" + exe, "args": args, "env": e}


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
        cfg = mkcfg(a.arm, a.exe, a.free_all, a.np, a.extra_arg, a.env)
        json.dump(cfg, open(a.out, "w"), indent=1)
        print(a.out, a.arm, "args:", len(cfg["args"]))
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "table":
        cmd_table(a)
    elif a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()
