#!/usr/bin/env python3
"""MP-2b summary per arm (host-only): request walls, passes, declines, floor shrinks, KV-pressure shrinks inside passes,
start cache size, free-VRAM minimum. usage: python tools/mp2b_summary.py build/llama-cuda/<tag> [more tags]"""
import json
import os
import re
import sys

TS = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s")
SHRINK = re.compile(r"device memory is short by ([\d.]+) MiB \(a KV commit\): expert cache (\d+) -> (\d+)")
ROOM = re.compile(r"lm-prefill: (?:room for [\d.]+ MiB made|no room for)")
PASS = re.compile(r"lm-prefill: pass (\d+): (\d+) tokens at position (\d+) in .*?: ([\d.]+) s ")
RESIZE = re.compile(r"cache resized to (\d+) slots .*? in ([\d.]+) ms")
START = re.compile(r"expert cache (\d+) -> (\d+) slots per layer \(start\)")
MINFREE = re.compile(r"free-watch: new minimum of free device memory (\d+) MiB")


def t_of(line):
    m = TS.match(line)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6


def summarize(tag_dir):
    rep = json.load(open(os.path.join(tag_dir, "report.json"), encoding="utf-8"))
    rows = rep["rows"]
    wall = sum(r.get("wall_s") or 0.0 for r in rows)
    http_ok = sum(1 for r in rows if r.get("status") == 200)
    raw = open(os.path.join(tag_dir, "server.log"), "rb").read().decode("utf-8", "replace").replace("\x00", "")
    lines = [l for l in raw.split("\n") if l.strip()]
    stamped = [(t_of(l), l) for l in lines]
    rooms = [t for t, l in stamped if t is not None and ROOM.search(l)]
    passes = []
    declines = 0
    start_slots = None
    minfree = None
    shrinks = []
    resizes = []
    for t, l in stamped:
        if t is None:
            continue
        m = PASS.search(l)
        if m:
            passes.append((t, int(m.group(2)), float(m.group(4))))
        if "no room for" in l:
            declines += 1
        m = START.search(l)
        if m and start_slots is None:
            start_slots = int(m.group(2))
        m = MINFREE.search(l)
        if m:
            minfree = int(m.group(1))
        m = SHRINK.search(l)
        if m:
            caused = any(0.0 <= rt - t <= 1.0 for rt in rooms)
            shrinks.append((t, int(m.group(2)), int(m.group(3)), caused))
        m = RESIZE.search(l)
        if m:
            resizes.append((t, int(m.group(1)), float(m.group(2))))
    kv_only_inside = 0
    for end, tok, dur in passes:
        start = end - dur
        kv_only_inside += sum(1 for s in shrinks if not s[3] and start <= s[0] <= end)
    floor_resizes = [r for r in resizes if r[1] == 2]
    other_resizes = [r for r in resizes if r[1] != 2]
    tail = None
    wrap = os.path.join(tag_dir, "wrapper.txt")
    if os.path.exists(wrap):
        for l in open(wrap, encoding="utf-8", errors="replace"):
            if "free VRAM (nvidia-smi view) min" in l:
                tail = l.strip()
    return {
        "tag": os.path.basename(tag_dir.rstrip("/\\")),
        "wall_s": round(wall, 2),
        "http_200": f"{http_ok}/{len(rows)}",
        "passes": len(passes),
        "declines": declines,
        "kv_only_inside_passes": kv_only_inside,
        "start_slots": start_slots,
        "floor_resizes": len(floor_resizes),
        "floor_resize_ms": round(sum(r[2] for r in floor_resizes)),
        "other_resize_ms": round(sum(r[2] for r in other_resizes)),
        "free_min_watch_mib": minfree,
        "free_min_harness": tail,
    }


def main():
    print(f"{'arm':22s} {'wall_s':>8s} {'http':>6s} {'passes':>6s} {'decl':>4s} {'kvIn':>4s} {'start':>5s} {'floorR':>6s} {'floor_ms':>8s} {'other_ms':>8s} {'free_min'}")
    for d in sys.argv[1:]:
        try:
            s = summarize(d)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"{os.path.basename(d):22s} not ready: {e}")
            continue
        print(f"{s['tag']:22s} {s['wall_s']:8.2f} {s['http_200']:>6s} {s['passes']:6d} {s['declines']:4d} {s['kv_only_inside_passes']:4d} "
              f"{str(s['start_slots']):>5s} {s['floor_resizes']:6d} {s['floor_resize_ms']:8d} {s['other_resize_ms']:8d} {s['free_min_watch_mib']}")
        if s["free_min_harness"]:
            print(f"{'':22s} {s['free_min_harness']}")


if __name__ == "__main__":
    main()
