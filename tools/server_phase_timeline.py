#!/usr/bin/env python3
"""G42: the timeline of every request in a server.log recorded with LLAMA_SRV_PHASE_LOG=1 and LLAMA_LM_LOG=1.

  python tools/server_phase_timeline.py DIR/server.log [DIR2/server.log ...] [--brief]

For every request (from `launch_slot_` to `stop processing`) it prints, relative to the launch: each decode of the server (tokens, duration), the cache shrink / room line of the pass, the pass line, the resources line, the
expert-cache regrow, and a summary row: launch -> first decode, decode time of the prompt batches, time after the last prompt decode until the release (first token + regrow), total.
Host-only, stdlib only."""
import re
import sys

LINE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) (\w) (.*)$")


def events(path):
    out = []
    for raw in open(path, "rb").read().decode("utf-8", "replace").replace(chr(0), "").split("\n"):
        m = LINE.match(raw)
        if not m:
            continue
        mm, ss, ms, us = map(int, m.groups()[:4])
        out.append((mm * 60 + ss + ms / 1000.0 + us / 1e6, m.group(6)))
    return out


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    brief = "--brief" in sys.argv
    for path in paths:
        ev = events(path)
        print("=====", path)
        reqs = []
        cur = None
        for t, s in ev:
            if "launch_slot_" in s and "processing task" in s:
                cur = {"t0": t, "rows": [], "decodes": []}
                reqs.append(cur)
                continue
            if cur is None:
                continue
            m = re.search(r"phase: decode of (\d+) tokens took ([\d.]+) ms", s)
            if m:
                cur["decodes"].append((t, int(m.group(1)), float(m.group(2))))
                cur["rows"].append((t, "decode %6d tokens %8.1f ms" % (int(m.group(1)), float(m.group(2)))))
            elif "lm-prefill: pass" in s:
                cur["rows"].append((t, "pass: " + re.sub(r"; embed.*", "", s.split("lm-prefill: ")[1])[:110]))
            elif "lm-prefill: resources" in s:
                cur["rows"].append((t, s.split("lm-prefill: ")[1][:100]))
            elif "room for" in s or "make_room_impl" in s:
                cur["rows"].append((t, s.split("hybrid-moe: ")[-1][:110]))
            elif "prefill window: expert cache" in s:
                cur["rows"].append((t, "regrow: " + s.split("hybrid-moe: ")[-1][:80]))
            elif "stop processing" in s:
                cur["t1"] = t
                cur["rows"].append((t, "release " + s.split("stop processing: ")[-1][:40]))
                cur = None
        tot = {"pre": 0.0, "dec": 0.0, "post": 0.0, "all": 0.0, "n": 0}
        for i, r in enumerate(reqs):
            if "t1" not in r or not r["decodes"]:
                continue
            first = r["decodes"][0]
            t_first_start = first[0] - first[2] / 1000.0
            prompt = [d for d in r["decodes"] if d[1] > 1]
            last_prompt_end = prompt[-1][0] if prompt else first[0]
            pre = t_first_start - r["t0"]
            dec = sum(d[2] for d in prompt) / 1000.0
            post = r["t1"] - last_prompt_end
            allt = r["t1"] - r["t0"]
            tot["pre"] += pre
            tot["dec"] += dec
            tot["post"] += post
            tot["all"] += allt
            tot["n"] += 1
            print("request %2d: launch -> first decode %.2f s | prompt decodes %.2f s (%s) | last prompt decode -> release %.2f s | total %.2f s" % (
                i, pre, dec, ", ".join("%d tok %.2f s" % (d[1], d[2] / 1000.0) for d in prompt), post, allt))
            if not brief:
                for t, text in r["rows"]:
                    print("    %+7.2f  %s" % (t - r["t0"], text))
        if tot["n"]:
            print("  mean over %d requests: set-up before the first decode %.2f s, prompt decodes %.2f s, after the last prompt decode %.2f s, total %.2f s" % (
                tot["n"], tot["pre"] / tot["n"], tot["dec"] / tot["n"], tot["post"] / tot["n"], tot["all"] / tot["n"]))


if __name__ == "__main__":
    main()
