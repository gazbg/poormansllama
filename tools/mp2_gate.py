#!/usr/bin/env python3
"""MP-2 gate (host-only): split expert-cache shrinks by cause and time the resizes, from one server.log.
usage: python mp2_gate.py <server.log>
A shrink line 'device memory is short ... (a KV commit)' is caused by the pass when a
'lm-prefill: room for' line follows within 1.0 s (the pass made its room); otherwise it is a KV-pressure shrink."""
import re
import sys

TS = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s")
SHRINK = re.compile(r"device memory is short by ([\d.]+) MiB \(a KV commit\): expert cache (\d+) -> (\d+)")
ROOM = re.compile(r"lm-prefill: (?:room for [\d.]+ MiB made|no room for)")
PASS = re.compile(r"lm-prefill: pass (\d+): (\d+) tokens at position (\d+) in .*?: ([\d.]+) s ")
RESIZE = re.compile(r"cache resized to (\d+) slots .*? in ([\d.]+) ms")
REFUSED = re.compile(r"could not make room for a KV commit")


def t_of(line):
    m = TS.match(line)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6


def main():
    raw = open(sys.argv[1], "rb").read().decode("utf-8", "replace").replace("\x00", "")
    lines = [l for l in raw.split("\n") if l.strip()]
    stamped = [(t_of(l), l) for l in lines]
    rooms = [t for t, l in stamped if t is not None and ROOM.search(l)]
    passes = []
    for t, l in stamped:
        m = PASS.search(l) if t is not None else None
        if m:
            passes.append((t, int(m.group(2)), float(m.group(4))))      # end time, tokens, duration
    shrinks = []
    refused = 0
    resizes = []
    for t, l in stamped:
        if t is None:
            continue
        m = SHRINK.search(l)
        if m:
            caused = any(0.0 <= rt - t <= 1.0 for rt in rooms)
            shrinks.append((t, int(m.group(2)), int(m.group(3)), caused))
        if REFUSED.search(l):
            refused += 1
        r = RESIZE.search(l)
        if r:
            resizes.append((t, int(r.group(1)), float(r.group(2))))

    kv_only = [s for s in shrinks if not s[3]]
    print(f"shrinks total {len(shrinks)}: pass-room {len(shrinks) - len(kv_only)}, KV-pressure only {len(kv_only)}; refused KV commits {refused}")
    print(f"passes {len(passes)}; resizes {len(resizes)}, total resize time {sum(r[2] for r in resizes):.0f} ms")
    print("per pass (end s, tokens, duration s, KV-only shrinks inside, resize ms after):")
    for i, (end, tok, dur) in enumerate(passes):
        start = end - dur
        nkv = sum(1 for s in kv_only if start <= s[0] <= end)
        nxt = passes[i + 1][0] if i + 1 < len(passes) else float("inf")
        rms = sum(r[2] for r in resizes if end <= r[0] < nxt)
        print(f"  {end:9.2f} {tok:7d} {dur:6.2f} {nkv:3d} {rms:8.0f}")
    long_passes = [p for p in passes if p[1] >= 8000]
    long_with_kv = sum(1 for i, p in enumerate(passes) if p[1] >= 8000 and any(p[0] - p[2] <= s[0] <= p[0] for s in kv_only))
    print(f"long passes (>= 8000 tokens): {len(long_passes)}; with >= 1 KV-pressure shrink inside: {long_with_kv}")
    print("KV-pressure shrink events:")
    for s in kv_only:
        print(f"  t={s[0]:9.2f}  expert cache {s[1]} -> {s[2]} slots")


if __name__ == "__main__":
    main()
