#!/usr/bin/env python3
"""Decode-window analysis of an nsys capture of the dense decode-stream server.
  python3 -I tools/dense/analyze-decode.py build/dense/tg-diag.sqlite N_TOKENS TOK_PER_S [LAYER_MIB]
The decode window is the last N_TOKENS / TOK_PER_S seconds before the last GPU kernel. Prints copy-engine / GPU / host-wait shares per token."""
import sqlite3, sys, collections
path, n_tok, rate = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
layer_mib = float(sys.argv[4]) if len(sys.argv) > 4 else 155.0
db = sqlite3.connect(path)
end = db.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
first = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
dur = n_tok / rate
W0, W1 = end - dur * 1e9, end

def union(iv):
    iv = sorted(iv); tot = 0; cur = None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: tot += cur[1] - cur[0]
    return tot

def clipped(rows):
    out = []
    for r in rows:
        s, e = max(r[0], W0), min(r[1], W1)
        if e > s: out.append((s, e) + tuple(r[2:]))
    return out

h2d = clipped(db.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and end>? and start<?", (W0, W1)).fetchall())
d2h = clipped(db.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and end>? and start<?", (W0, W1)).fetchall())
ker = clipped(db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?", (W0, W1)).fetchall())
big = [r for r in h2d if r[2] > 1 << 20]
Wd = (W1 - W0) / 1e9
bc = union([(s, e) for s, e, b in h2d]) / 1e9
bk = union([(s, e) for s, e in ker]) / 1e9
ba = union([(s, e) for s, e, b in h2d] + [(s, e) for s, e in ker]) / 1e9
byt = sum(b for s, e, b in h2d)
print(f"capture {(end-first)/1e9:.1f} s; decode window {Wd:.2f} s = {n_tok} tokens at {rate} tok/s ({1000/rate:.0f} ms per token)")
print(f"weights copied host->GPU: {byt/1048576:.0f} MiB = {byt/1048576/layer_mib:.0f} layers = {byt/1048576/layer_mib/n_tok:.1f} layers per token = {byt/1e9/n_tok:.2f} GB per token")
print(f"copy engine busy {bc/n_tok*1000:6.0f} ms/token ({100*bc/Wd:5.1f} %), effective {byt/1e9/bc:.2f} GB/s")
print(f"GPU kernels busy {bk/n_tok*1000:6.0f} ms/token ({100*bk/Wd:5.1f} %), {len(ker)/n_tok:.0f} kernels per token")
print(f"GPU idle (no copy, no kernel) {(Wd-ba)/n_tok*1000:5.1f} ms/token ({100*(Wd-ba)/Wd:.1f} %)")
print(f"device->host copies {len(d2h)/n_tok:.0f} per token ({sum(b for s,e,b in d2h)/1048576/n_tok:.2f} MiB per token)")
# copy sizes per layer: group of tensor sizes
sizes = collections.Counter(round(b / 1048576, 1) for s, e, b in big)
print("large copies per token by tensor size (MiB):", {k: round(v / n_tok, 1) for k, v in sorted(sizes.items(), reverse=True)[:7]})
# gaps in the copy stream between consecutive big copies
bs = sorted(big)
gaps = [(bs[i + 1][0] - bs[i][1]) / 1e6 for i in range(len(bs) - 1) if bs[i + 1][0] > bs[i][1]]
print(f"gaps between consecutive large copies: {len(gaps)} gaps, total {sum(gaps)/n_tok:.1f} ms/token, max {max(gaps) if gaps else 0:.1f} ms")
rows = db.execute("""select s.value, count(*), sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
  where r.start >= ? and r.end <= ? group by s.value order by sum(r.end-r.start) desc limit 6""", (W0, W1)).fetchall()
print("host CUDA API calls in the window (per token):")
host = {}
for name, n, ms in rows:
    short = name.split('_v')[0]
    host[short] = ms
    print(f"   {short:26s} {n/n_tok:7.0f} calls, {ms/n_tok:7.1f} ms blocked/spent")
sync_ms = host.get('cudaStreamSynchronize', 0) + host.get('cudaEventSynchronize', 0)
other_api = sum(v for k, v in host.items() if k not in ('cudaStreamSynchronize', 'cudaEventSynchronize'))
cpu_work = 1000 / rate - sync_ms / n_tok - other_api / n_tok
print(f"=> per token {1000/rate:.0f} ms = host blocked in syncs {sync_ms/n_tok:.0f} ms + CUDA API calls {other_api/n_tok:.0f} ms + CPU compute and the rest {cpu_work:.0f} ms")
