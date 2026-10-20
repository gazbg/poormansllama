#!/usr/bin/env python3
"""Cut one decode token out of an nsys capture of the dense decode-stream server and write the lane data of the timeline diagram.
  python3 -I tools/dense/timeline-data.py build/dense/tg-diag.sqlite OUT.json [TOKENS_BACK]
Times are ms from the start of the token. A token ends at the logits copy to the host (the only device->host copy over 0.5 MiB).
Lanes:
  packages  host->GPU weight copies. One streamed layer = six consecutive big copies (11.5, 15.9, 32.5, 36.5, 36.5, 21.5 MiB); the package that
            lands right before the n-th layer burst feeds layer 2n (burst 0 uses the package that landed during the previous token).
  gpu       kernel clusters (gap < 0.5 ms merged): 'layer' = a streamed layer (>= 20 kernels), 'small' = the few small ops around the CPU layers
  cpu       the window of an odd (host) layer: from the end of the activation copy to the host (20,480 B) to the start of the copy back
  sync      main thread blocked in cudaStreamSynchronize / cudaEventSynchronize for more than 0.2 ms
  hops      the 20,480 B activation copies (down = GPU->host, up = host->GPU)
"""
import json, sqlite3, sys, bisect

path, out = sys.argv[1], sys.argv[2]
back = int(sys.argv[3]) if len(sys.argv) > 3 else 4
db = sqlite3.connect(path)
q = lambda s, a=(): db.execute(s, a).fetchall()

ends = [r[1] for r in q("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and bytes>500000 order by start")]
t1, t0 = ends[-1 - back], ends[-2 - back]
T = (t1 - t0) / 1e6
ms = lambda t: (t - t0) / 1e6

# --- GPU clusters
ker = q("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<? order by start", (t0, t1))
cl = []
for s, e in ker:
    s, e = max(s, t0), min(e, t1)
    if cl and s - cl[-1][1] < 0.5e6:
        cl[-1][1] = max(cl[-1][1], e); cl[-1][2] += 1; cl[-1][3] += e - s
    else:
        cl.append([s, e, 1, e - s])
layer_cl = [c for c in cl if c[2] >= 20]
gpu = []
li = 0
for c in cl:
    if c[2] >= 20:
        gpu.append({"s": ms(c[0]), "e": ms(c[1]), "k": c[2], "kind": "layer", "layer": 2 * li}); li += 1
    else:
        gpu.append({"s": ms(c[0]), "e": ms(c[1]), "k": c[2], "kind": "small", "layer": None})

# --- weight packages (six big copies each, starting with the 11.5 MiB tensor)
big = [(s, e, b) for s, e, b in q("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>1000000 and end>? and start<? order by start", (t0, t1))]
pk, cur = [], []
for s, e, b in big:
    if abs(b / 1048576 - 11.5) < 0.2 and cur:
        pk.append(cur); cur = []
    cur.append((max(s, t0), min(e, t1), b))
if cur: pk.append(cur)
packages = []
for p in pk:
    packages.append({"s": ms(p[0][0]), "e": ms(p[-1][1]), "mib": round(sum(x[2] for x in p) / 1048576, 1), "n": len(p), "parts": [[ms(x[0]), ms(x[1]), round(x[2] / 1048576, 1)] for x in p]})
# package -> layer: the package that ENDS within 3 ms of a layer burst's end feeds that burst
lend = [(ms(c[1]), 2 * i) for i, c in enumerate(layer_cl)]
for pkg in packages:
    best = min(lend, key=lambda x: abs(x[0] - pkg["e"]))
    pkg["layer"] = best[1] if abs(best[0] - pkg["e"]) < 3.0 else None

# --- hand-offs and CPU windows
sm = q("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where bytes=20480 and end>? and start<? order by start", (t0, t1))
hops = [{"s": ms(s), "e": ms(e), "dir": "up" if k == 1 else "down"} for s, e, b, k in sm]
down = [h for h in hops if h["dir"] == "down"]; up = [h for h in hops if h["dir"] == "up"]
cpu = []
for i, c in enumerate(layer_cl):
    tE = ms(c[1])
    nxt = ms(layer_cl[i + 1][0]) if i + 1 < len(layer_cl) else T
    d = [h for h in down if h["s"] >= tE - 0.1 and h["s"] < nxt]
    u = [h for h in up if h["e"] <= nxt + 0.1 and h["s"] > tE]
    if d and u:
        cpu.append({"s": d[0]["e"], "e": u[-1]["s"], "layer": 2 * i + 1})

# --- main thread blocked in a CUDA synchronisation
tid = q("select globalTid, count(*) c from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by c desc limit 1")[0][0]
names = {r[0]: r[1] for r in q("select id,value from StringIds")}
sync = [{"s": ms(max(s, t0)), "e": ms(min(e, t1))} for s, e, n in q(
    "select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and end>? and start<? order by start", (tid, t0, t1))
    if "Synchronize" in names[n] and (e - s) > 0.2e6]

stats = {
    "token_ms": round(T, 1),
    "layer_bursts": len(layer_cl), "packages": len(packages), "cpu_windows": len(cpu),
    "copy_busy_ms": round(sum(p["e"] - p["s"] for p in packages), 1),
    "gpu_layer_ms": round(sum(c[3] for c in layer_cl) / 1e6, 1),
    "gpu_all_ms": round(sum(c[3] for c in cl) / 1e6, 1),
    "cpu_layer_ms": round(sum(w["e"] - w["s"] for w in cpu), 1),
    "sync_ms": round(sum(x["e"] - x["s"] for x in sync), 1),
    "package_ms_mean": round(sum(p["e"] - p["s"] for p in packages) / len(packages), 2),
    "package_period_ms": round((packages[-1]["e"] - packages[0]["e"]) / (len(packages) - 1), 2),
    "cpu_window_ms_mean": round(sum(w["e"] - w["s"] for w in cpu) / max(1, len(cpu)), 2),
    "layer_burst_ms_mean": round(sum(c[3] for c in layer_cl) / 1e6 / len(layer_cl), 2),
}
json.dump({"stats": stats, "packages": packages, "gpu": gpu, "cpu": cpu, "sync": sync, "hops": hops}, open(out, "w"), separators=(",", ":"))
print(stats)
print("packages: layer, start, end:", [(p["layer"], round(p["s"], 1), round(p["e"], 1)) for p in packages[:4]], "...", [(p["layer"], round(p["s"], 1), round(p["e"], 1)) for p in packages[-2:]])
print("layer bursts:", [(g["layer"], round(g["s"], 1), round(g["e"], 1)) for g in gpu if g["kind"] == "layer"][:4], "...", [(g["layer"], round(g["s"], 1), round(g["e"], 1)) for g in gpu if g["kind"] == "layer"][-2:])
print("cpu windows:", [(w["layer"], round(w["s"], 1), round(w["e"], 1)) for w in cpu[:4]])
print("small clusters:", sum(1 for g in gpu if g["kind"] == "small"))
