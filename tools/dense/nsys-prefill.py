#!/usr/bin/env python3
"""Anatomy of a prefill capture (nsys export of one prompt): how busy the GPU is, what the kernels are, what the copies do. Host-only.
  python tools/dense/nsys-prefill.py capture.sqlite --tokens 4001 [--params 27.3e9] [--top 25] [--gaps 10]
Times in the sqlite are NANOSECONDS (AGENTS.md). The window is first kernel start .. last kernel end of the capture, so a capture that
holds only the prompt gives the prompt's own anatomy; --tokens turns the matmul time into an effective rate."""
import argparse, re, sqlite3, collections

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--tokens", type=int, default=0, help="prompt tokens in the window (for the effective TFLOPS)")
ap.add_argument("--params", type=float, default=27.3e9, help="weights touched per token (dense: all parameters)")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--gaps", type=int, default=10)
a = ap.parse_args()
c = sqlite3.connect(a.db)
q = lambda sql, *p: c.execute(sql, p).fetchall()

GGML_TYPES = {2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
              16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16"}

def category(name):
    n = name.lower()
    if "mul_mat_q" in n:
        m = re.search(r"\(ggml_type\)(\d+)", name)
        return "mmq " + (GGML_TYPES.get(int(m.group(1)), "type" + m.group(1)) if m else "?")
    if "flash_attn" in n or "fattn" in n:
        return "flash attention (" + ("tile" if "tile" in n else "vec" if "vec" in n else "mma" if "mma" in n else "other") + ")"
    if "gated_delta" in n or "delta_net" in n:
        return "gated delta net"
    if "ssm" in n or "conv1d" in n:
        return "ssm / conv"
    if "mul_mat_vec" in n or "mmvq" in n or "mmvf" in n:
        return "mat-vec"
    if "quantize" in n:
        return "quantize q8_1"
    if "norm" in n:
        return "norm"
    if "rope" in n:
        return "rope"
    if "silu" in n or "swiglu" in n or "glu" in n or "unary" in n:
        return "activation"
    if "get_rows" in n or "set_rows" in n:
        return "get/set rows"
    if "cpy" in n or "dup" in n or "cont" in n or "convert" in n or "dequant" in n:
        return "copy / convert"
    if "bin_bcast" in n or "scale" in n or "soft_max" in n or "softmax" in n:
        return "elementwise"
    return "other"

tabs = {r[0] for r in q("select name from sqlite_master where type='table'")}
K = "CUPTI_ACTIVITY_KIND_KERNEL"
rows = q(f"select k.start, k.end, k.streamId, s.value from {K} k join StringIds s on s.id = k.demangledName order by k.start")
if not rows:
    raise SystemExit("no kernels in the capture")
t0, t1 = rows[0][0], max(r[1] for r in rows)
wall = (t1 - t0) / 1e6
# union of the kernel intervals = time the GPU was running something
busy = 0; cur_s, cur_e = rows[0][0], rows[0][1]
for s, e, _, _ in rows[1:]:
    if s > cur_e:
        busy += cur_e - cur_s; cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
ksum = sum(r[1] - r[0] for r in rows)
print(f"window {wall / 1000:.2f} s, {len(rows)} kernels; GPU running a kernel {busy / 1e6 / wall * 100:.1f}% of the window ({busy / 1e9:.2f} s), idle {100 - busy / 1e6 / wall * 100:.1f}%")
print(f"sum of kernel durations {ksum / 1e9:.2f} s (overlap across streams {100 * (ksum - busy) / max(ksum, 1):.1f}%)")

cat = collections.defaultdict(lambda: [0, 0])
byname = collections.defaultdict(lambda: [0, 0])
for s, e, _, name in rows:
    d = e - s
    cat[category(name)][0] += d; cat[category(name)][1] += 1
    byname[name][0] += d; byname[name][1] += 1
print("\n--- kernel time by category (share of kernel time, share of the window)")
for k, (d, n) in sorted(cat.items(), key=lambda x: -x[1][0]):
    print(f"{k:34s} {d / 1e6:10.1f} ms  {100 * d / ksum:5.1f}% of kernels  {100 * d / 1e6 / wall:5.1f}% of window  n={n:7d}  mean {d / n / 1e3:9.1f} us")
mm = sum(d for k, (d, n) in cat.items() if k.startswith("mmq"))
if a.tokens and mm:
    fl = 2 * a.params * a.tokens
    print(f"\nmatmul (mmq) kernels: {mm / 1e9:.2f} s for {a.tokens} tokens x {2 * a.params / 1e9:.1f} GFLOP = {fl / 1e12:.0f} TFLOP -> {fl / (mm / 1e9) / 1e12:.1f} TFLOPS while a matmul runs")
    print(f"whole window effective: {fl / (wall / 1e3) / 1e12:.1f} TFLOPS, {a.tokens / (wall / 1e3):.1f} tokens/s")

print(f"\n--- top {a.top} kernels by total time")
for name, (d, n) in sorted(byname.items(), key=lambda x: -x[1][0])[:a.top]:
    print(f"{d / 1e6:9.1f} ms {100 * d / ksum:5.1f}%  n={n:6d} mean {d / n / 1e3:9.1f} us  {name[:110]}")

print("\n--- kernels by stream")
for r in q(f"select streamId, count(*), sum(end - start) from {K} group by streamId order by 3 desc"):
    print(f"stream {r[0]}: {r[1]:7d} kernels, busy {r[2] / 1e6:9.1f} ms ({100 * r[2] / 1e6 / wall:5.1f}% of window)")

M = "CUPTI_ACTIVITY_KIND_MEMCPY"
if M in tabs:
    print("\n--- memcpy by kind/stream (copyKind 1 = HtoD, 2 = DtoH, 8 = DtoD)")
    for r in q(f"select copyKind, streamId, count(*), sum(bytes), sum(end - start), max(end - start) from {M} group by copyKind, streamId order by 4 desc"):
        kd, st, n, b, d, mx = r
        print(f"kind {kd} stream {st}: {n:6d} copies {b / 1048576:10.1f} MiB  busy {d / 1e6:9.1f} ms ({100 * d / 1e6 / wall:5.1f}% of window)  rate {b / max(d, 1):6.2f} GB/s  max {mx / 1e3:9.1f} us")
    hb = q(f"select sum(bytes), sum(end - start) from {M} where copyKind = 1 and bytes > 1048576")
    if hb and hb[0][0]:
        print(f"big HtoD copies (>1 MiB): {hb[0][0] / 1073741824:.2f} GiB in {hb[0][1] / 1e9:.2f} s of copy-engine time = {hb[0][0] / hb[0][1]:.2f} GB/s; the window is {wall / 1000:.2f} s, so a pure copy lower bound is {hb[0][1] / 1e9:.2f} s")

print(f"\n--- the {a.gaps} longest idle gaps (no kernel running on any stream)")
gaps = []; end = rows[0][1]
for s, e, _, name in rows[1:]:
    if s > end:
        gaps.append((s - end, end, name))
    end = max(end, e)
gaps.sort(reverse=True)
tot = sum(g[0] for g in gaps)
print(f"{len(gaps)} gaps, {tot / 1e6:.1f} ms in total; gaps > 1 ms: {sum(1 for g in gaps if g[0] > 1e6)} ({sum(g[0] for g in gaps if g[0] > 1e6) / 1e6:.1f} ms)")
for g, e, name in gaps[:a.gaps]:
    print(f"  +{(e - t0) / 1e6:9.1f} ms  idle {g / 1e6:8.2f} ms  before {name[:80]}")
