#!/usr/bin/env python3
"""Anatomy of a decode capture of the dense ring (win-arm.py --nsys): copy engine, kernels, host CUDA calls.  Host-only.
  python tools/dense/nsys-ring.py build/dense/win/n4-s8.sqlite [--tokens N]
Times in the sqlite are NANOSECONDS (AGENTS.md)."""
import argparse, sqlite3, collections

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--tokens", type=int, default=20)
ap.add_argument("--timeline", type=int, default=60, help="events of the timeline to print")
a = ap.parse_args()
c = sqlite3.connect(a.db)
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
def q(sql, *p):
    return c.execute(sql, p).fetchall()
t0 = min(q("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL")[0][0], q("select min(start) from CUPTI_ACTIVITY_KIND_MEMCPY")[0][0])
t1 = max(q("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL")[0][0], q("select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY")[0][0])
wall = (t1 - t0) / 1e6
print(f"capture window {wall:.1f} ms, {wall / a.tokens:.1f} ms per token ({a.tokens} tokens)")
print("--- memcpy by kind/stream (copyKind 1 = HtoD, 2 = DtoH, 8 = DtoD)")
for r in q("select copyKind, streamId, count(*), sum(bytes), sum(end-start), avg(end-start), max(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind, streamId"):
    k, s, n, b, d, av, mx = r
    print(f"kind {k} stream {s}: {n:5d} copies {b / 1048576:9.1f} MiB  busy {d / 1e6:8.1f} ms ({100 * d / 1e6 / wall:5.1f}% of window)  mean {av / 1e3:8.1f} us  max {mx / 1e3:8.1f} us  rate {b / max(d, 1):6.2f} GB/s")
print("--- kernels by stream")
for r in q("select streamId, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by streamId"):
    print(f"stream {r[0]}: {r[1]:6d} kernels, busy {r[2] / 1e6:8.1f} ms ({100 * r[2] / 1e6 / wall:5.1f}%)")
print("--- host CUDA runtime calls (total ms over the window, top 12)")
rt = "CUPTI_ACTIVITY_KIND_RUNTIME"
if rt in tabs:
    for r in q(f"select s.value, count(*), sum(r.end-r.start), max(r.end-r.start) from {rt} r join StringIds s on s.id=r.nameId group by s.value order by 3 desc limit 12"):
        print(f"{r[0][:44]:44s} n={r[1]:6d} total {r[2] / 1e6:9.1f} ms  max {r[3] / 1e6:8.2f} ms")
print("--- HtoD copies in time order (first 24): start offset ms, MiB, duration ms, stream")
rows = q("select start, end, bytes, streamId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start")
big = [r for r in rows if r[2] > 1 << 20]
for r in big[:24]:
    print(f"  +{(r[0] - t0) / 1e6:9.2f} ms  {r[2] / 1048576:7.2f} MiB  {(r[1] - r[0]) / 1e6:7.2f} ms  stream {r[3]}")
if big:
    gaps = [(big[i + 1][0] - big[i][1]) / 1e6 for i in range(len(big) - 1)]
    print(f"big-copy count {len(big)}; idle gap between consecutive big copies: median {sorted(gaps)[len(gaps) // 2]:.2f} ms, mean {sum(gaps) / len(gaps):.2f} ms, max {max(gaps):.2f} ms")

# --- hop latency: small HtoD copies, split by whether a ring copy (the stream with the most HtoD bytes) was in flight when the API call returned
import bisect, statistics
ring = q("select streamId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 group by streamId order by sum(bytes) desc limit 1")[0][0]
bulk = q("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and streamId=? order by start", ring)
rows = q("""select m.start, m.end, m.bytes, r.end from CUPTI_ACTIVITY_KIND_MEMCPY m join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = m.correlationId
            where m.copyKind=1 and m.streamId<>? and m.bytes < 1048576 order by m.start""", ring)
bs = [b[0] for b in bulk]
def busy(t, slack=0):
    i = bisect.bisect_right(bs, t) - 1
    return i >= 0 and bulk[i][1] + slack >= t
lat_in, lat_out = [], []
for ms, me, b, rend in rows:
    (lat_in if busy(rend, 2000) else lat_out).append((ms - rend) / 1e3)
def summ(n, v):
    v = sorted(v)
    if not v:
        print(f"{n:42s} none"); return
    print(f"{n:42s} n={len(v):5d} ({len(v) / a.tokens:6.1f}/token) median {statistics.median(v):8.1f} us  p90 {v[int(.9 * len(v))]:9.1f}  max {v[-1]:10.1f}  sum {sum(v) / 1e3:8.1f} ms ({sum(v) / 1e3 / a.tokens:6.1f} ms/token)")
print(f"--- hop latency (API return -> transfer start), ring stream = {ring}")
summ("small HtoD issued while ring copy active", lat_in)
summ("small HtoD issued with ring copy idle", lat_out)
# GPU activity: fraction of the window the copy engine (ring stream) is busy, and the longest idle stretch between pieces
if bulk:
    gaps = sorted((bulk[i + 1][0] - bulk[i][1]) / 1e3 for i in range(len(bulk) - 1))
    print(f"ring-stream pieces {len(bulk)}; idle gap between pieces: p50 {gaps[len(gaps)//2]:.1f} us, p99 {gaps[int(.99*len(gaps))]:.1f} us, max {gaps[-1]/1e3:.2f} ms")
