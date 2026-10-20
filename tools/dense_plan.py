#!/usr/bin/env python3
"""Plan the layer placement of the dense Qwen3.8-27B (UD-IQ3_XXS) on a small card: which layers stay resident on the GPU (R), which stay
in host memory and are computed by the CPU (C), and which of the host layers are streamed through GPU slots during decode (S).

  python3 -I tools/dense_plan.py --ctx 16384                        # best plan for the card
  python3 -I tools/dense_plan.py --ctx 16384 --pass-mib 1130        # leave room for the layer-major prefill pass
  python3 -I tools/dense_plan.py --ctx 16384 --table                # every candidate, not just the best ones
  python3 -I tools/dense_plan.py --ctx 16384 --plan attn            # the owner's proposal: only the full-attention layers on the GPU

The model of a decode token (host-only arithmetic, constants measured on the GTX 1070 / i7-4770K box, 2026-10-19; every one is an argument):
  T = max( base + C * c * (1 + sigma) + (R + S) * g ,  S * d )       with  d = layer bytes / PCIe rate
  c  ms of CPU per host layer, g ms of GPU per device layer, sigma the slowdown of the CPU while the copy engine reads the same DRAM,
  base the per-token cost that does not scale with the plan (embedding, head, sampling, graph launch, activation hops).
The copy of a streamed layer overlaps the compute of everything else of the token; the ring of slots must close (S a multiple of the slots).
VRAM = resident layers + slots + output head + KV + recurrent state + compute buffer + CUDA context (+ the room the prefill pass wants when --pass-mib is given).
It is a planning aid, not a measurement: tools/dense_plan.py prints the prediction and the measured value belongs next to it in the results.
"""
import argparse
import sys

N_LAYER = 64
LAYER_MIB = {"rec": 154.86, "att": 155.71}      # UD-IQ3_XXS file
OUT_MIB = 834.0                                # output.weight Q5_K (device resident)
STATE_MIB = 150.0                              # recurrent state of the 48 Gated DeltaNet layers (one sequence)


def is_att(il):
    return (il + 1) % 4 == 0


def layer_mib(il):
    return LAYER_MIB["att" if is_att(il) else "rec"]


def spread(host, k):
    """k layers of the host list, evenly spread (so that the copies overlap the CPU layers between them)."""
    if k <= 0:
        return []
    n = len(host)
    return sorted({host[min(n - 1, int((i + 0.5) * n / k))] for i in range(k)})


def fmt_list(v):
    """3,7,11 -> compact ranges"""
    out, i = [], 0
    v = sorted(v)
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[j] + 1:
            j += 1
        out.append(f"{v[i]}-{v[j]}" if j > i else f"{v[i]}")
        i = j + 1
    return ",".join(out)


def evaluate(host, stream, slots, a):
    res = [il for il in range(N_LAYER) if il not in set(host)]
    s_n, c_n = len(stream), len(host) - len(stream)
    t_cpu = c_n * a.c_ms * (1.0 + (a.sigma if s_n else 0.0))
    t_gpu = (len(res) + s_n) * a.g_ms
    d_ms = (sum(layer_mib(il) for il in stream) / 1024.0 * 1.073741824) / a.pcie_gbs * 1000.0 if s_n else 0.0
    t = max(a.base_ms + t_cpu + t_gpu, d_ms)
    vram = (sum(layer_mib(il) for il in res) + (slots * LAYER_MIB["att"] if s_n else 0) + OUT_MIB + STATE_MIB
            + a.ctx * a.kv_bytes / 2**20 + a.fixed_mib + a.pass_mib)
    return t, vram, len(res), c_n, s_n, d_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384, help="context size (tokens) of the server")
    ap.add_argument("--kv-bytes", type=float, default=34816.0, help="KV bytes per token (q8_0: 34816, f16: 69632)")
    ap.add_argument("--vram-mib", type=float, default=8107.0, help="device memory the process can use")
    ap.add_argument("--margin-mib", type=float, default=250.0, help="kept free on top (allocator slack, the guard of the prefill pass wants 96 MiB beyond its need)")
    ap.add_argument("--fixed-mib", type=float, default=370.0, help="CUDA context + decode compute buffer + slack (calibrated: 6,393 and 6,405 MiB measured against 6,401 and 6,402 predicted)")
    ap.add_argument("--pass-mib", type=float, default=0.0, help="room the layer-major prefill pass needs on top (beyond the ring, whose slots the pass takes over: ~450 at chunk 1024, segment 4096, bf16 slab; 1130 when the ring is absent, chunk 2048)")
    ap.add_argument("--c-ms", type=float, default=16.5, help="CPU ms per host layer")
    ap.add_argument("--g-ms", type=float, default=1.3, help="GPU ms per device layer")
    ap.add_argument("--base-ms", type=float, default=90.0, help="per-token cost independent of the plan")
    ap.add_argument("--pcie-gbs", type=float, default=6.45, help="host-to-device GB/s")
    ap.add_argument("--sigma", type=float, default=0.15, help="CPU slowdown while the copy engine streams")
    ap.add_argument("--slots", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--plan", choices=["best", "attn", "stock"], default="best")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    print(f"model: {N_LAYER} layers, {sum(layer_mib(i) for i in range(N_LAYER)) / 1024:.2f} GiB; budget {a.vram_mib:.0f} MiB minus a margin of {a.margin_mib:.0f}, ctx {a.ctx} ({a.ctx * a.kv_bytes / 2**20:.0f} MiB KV), pass room {a.pass_mib:.0f} MiB")
    rows = []
    for H in range(0, N_LAYER + 1):
        host = list(range(H))                         # the first H layers stay on the host
        for slots in a.slots:
            ks = [0] + [k for k in range(slots, H + 1) if k % slots == 0 or k <= slots]
            for k in sorted(set(ks)):
                if k > H or (k > slots and k % slots):
                    continue
                stream = spread(host, k)
                if len(stream) != k:
                    continue
                t, vram, r, c, s, d = evaluate(host, stream, slots, a)
                if vram <= a.vram_mib - a.margin_mib:
                    rows.append((t, vram, H, slots, k, host, stream, r, c, s, d))
    if not rows:
        print("nothing fits")
        return 1
    rows.sort(key=lambda x: x[0])

    def show(row, label):
        t, vram, H, slots, k, host, stream, r, c, s, d = row
        print(f"{label:10s} predicted {1000.0 / t:5.2f} tok/s ({t:5.0f} ms)  VRAM {vram:5.0f} MiB | resident {r:2d}, CPU {c:2d}, streamed {s:2d} through {slots if s else 0} slots (copy {d:4.0f} ms/token)")
        print(f"           --lm-prefill-host-layers {fmt_list(host)}" + (f" --lm-decode-stream {fmt_list(stream)} --lm-decode-slots {slots}" if s else ""))

    best_nostream = next(r for r in rows if r[4] == 0)
    if a.plan == "attn":
        # the owner's proposal: the 16 full-attention layers on the GPU through 2 slots, every other layer on the CPU, nothing resident
        host = list(range(N_LAYER))
        stream = [il for il in host if is_att(il)]
        t, vram, r, c, s, d = evaluate(host, stream, 2, a)
        print(f"owner plan  predicted {1000.0 / t:5.2f} tok/s ({t:5.0f} ms)  VRAM {vram:5.0f} MiB | resident {r}, CPU {c}, streamed {s} through 2 slots (copy {d:.0f} ms/token)")
        print(f"            --lm-prefill-host-layers 0-63 --lm-decode-stream {fmt_list(stream)} --lm-decode-slots 2")
        return 0
    show(best_nostream, "no stream")
    show(rows[0], "best")
    if a.table:
        for row in rows[:40]:
            show(row, "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
