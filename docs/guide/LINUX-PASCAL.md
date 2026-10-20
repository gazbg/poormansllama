# Linux and Pascal (GTX 1070) notes

The rest of this guide was written and measured on Windows with an RTX 5090. This
file records what it took to build and run the project on Linux with an 8 GiB
GTX 1070 (Pascal, compute capability 6.1, no tensor cores), and what was measured
there. Unless a line says otherwise, a number is one run on one machine.

Machine, 2026-10-19 and 2026-10-20: Ubuntu 22.04, gcc 12 and g++ 11, CMake 4.4,
CUDA 12.8, GTX 1070 8 GiB (two cards installed, only one used), Intel i7-4770K (4
cores, 8 threads), 30 GB RAM. Model `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.7 GB). The
GPU held 1,898 MHz under load without throttling.

## 1. Build

```bash
cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
cmake --build build/llama-cuda --target llama-server --parallel 8
```

Set `CMAKE_CUDA_ARCHITECTURES` to the card. The default list targets Blackwell.

The fork did not compile with GCC and CUDA 12.8. The patches are in the commit
"third_party: build the CUDA backend and layer streaming with GCC and CUDA 12.8":
an export macro instead of bare `__declspec`, the `cudaMemcpyBatchAsync`
argument list below CUDA 13, the Win32-only prefetch worker guarded by
`LS_PF_ENABLED`, and an inline overload instead of a `= {}` default argument.
`--pp-prefetch-ahead` still refuses on Linux. `tools/layer-streaming/main.cpp`
still has Win32 code and was not built.

## 2. Run on one GPU

```bash
LLAMA_FATTN_TILE_Q8QK=1 LLAMA_MOE_HYBRID_LEAN_CPUS=2,3,1,0 CUDA_VISIBLE_DEVICES=0 \
$REPO/build/llama-cuda/bin/llama-server \
  -m $MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8034 \
  -dev CUDA0 -sm none -mg 0 \
  -c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 \
  --hybrid-moe \
  --lm-prefill --lm-prefill-fuse --lm-prefill-prearm auto \
  --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 \
  --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32
```

- `-fitt` is left out on purpose. The default 1,024 MiB margin suits an 8 GiB card;
  the 16,000 MiB in the main recipe is for a 24 GiB card.
- `LLAMA_MOE_HYBRID_LEAN_CPUS` pins the expert threads, one per physical core
  (section 3.2). Choose the list for your CPU.
- `LLAMA_FATTN_TILE_Q8QK=1` is the int8 attention path (section 3.4).
- `--lm-prefill-canonical` is left out (section 3.6). Add it back for a cache hit
  that matches a fresh run.

Measured with exactly this command (measured):

| What | Result |
|---|---|
| Decode, short prompt | 32 to 34 tok/s |
| Decode with 100k tokens of context | about 26.5 tok/s |
| Cold prefill, 19k tokens | 28.4 s (670 tok/s) |
| Cold prefill, 100k tokens (server at 131,072 context) | 332 s (309 tok/s); the hidden needle was found |
| Follow-up with 35 new tokens after 100k | 1.4 s |
| Peak VRAM, GPU 0 | 6.8 GiB; GPU 1 untouched |

## 3. Findings

### 3.1 Causal attention was wrong without tensor cores (fixed)

`--lm-prefill` gives `FLASH_ATTN_EXT` the causal window as a negative mask
stride. The MMA kernel (Turing and newer) honoured it. The tile kernel, used on
GPUs without tensor cores, did not, so every token of a 2,048-token chunk could
see the later tokens of the same chunk. `test-backend-ops` failed its own causal
cases: normalised error 0.15 to 0.17 against 5e-4 at head size 256, 0.65 at head
size 64. A 6k-token needle prompt returned an empty answer (first-token logprob
-0.61 against -0.09 on the ordinary path). The fix is in `fattn-tile.cuh`. The
`FLASH_ATTN_EXT` suite now passes 2,935 of 2,935 with the int8 path off and on.
It costs about 5% in the FP32 kernel on a full prompt (100k tokens: 471 s before,
497 s after). Any earlier `--lm-prefill` number taken on a GPU that uses the tile
kernel used the wrong attention.

### 3.2 Decode: pin the expert threads

The default thread layout assumes the two hardware threads of a core are
neighbouring CPUs (stride 2). On this CPU they are `i` and `i+4`, so spinning team
threads share cores. `LLAMA_MOE_HYBRID_LEAN_CPUS=2,3,1,0` (one thread per physical
core, the last entry for the engine thread) took decode from 5 to 8 tok/s to 30
to 35 tok/s. The generated text was identical. A profile showed the team threads
spending about 99.7% of their time in barrier spin loops, not in expert math.

### 3.3 Prefill limits on GP104 (measured)

- FP32 peak 7.3 TFLOPS at the sustained clock. FP16 is about 1/64 of that. Int8
  `dp4a` peak is about 29 TOPS (computed from 1,920 lanes, 4 MACs, 1,898 MHz).
- Profile of a 19k-token pass before the changes: flash attention 36.0%, integer
  matmuls (`mul_mat_q`) 43.3%, gated delta net 7.6%.
- Cost of a 2,048-token chunk: about 2.3 s at depth 0 plus 0.14 s for every 1k of
  depth (fit over three passes). At 100k depth about 86% of a chunk is attention.
- The tile attention kernel ran at 2.68 TFLOPS, 37% of peak. It issues about 2.7
  FMAs per shared-memory load, and per-lane loads cost four cycles of the 128
  byte/cycle shared-memory bandwidth, so the QK phase is load bound.
- MoE matmuls reach 12 to 16% of the `dp4a` peak and dense Q8_0 about 30%.
- The pass has a fixed floor of about 3.1 s (the expert banks stream over PCIe).

### 3.4 Int8 QK attention (opt-in)

With `LLAMA_FATTN_TILE_Q8QK=1`, head size 256, a q8_0 K cache and the FP32 tile
kernel, QK^T runs on `dp4a` against the raw q8_0 K blocks, with the query columns
quantized to int8 per 32 values. That is the arithmetic the decode kernel already
uses for q8_0 K. V and PV stay FP32.

- Attention kernel 1.67 to 1.69 times faster at every depth (isolated, measured).
- Cold prefill, 19k tokens: 35.1 s to 28.2 s. 100k tokens: 471 s to 332 s.
- Quality (measured, thin): mean first-token KL against the ordinary path over
  four prompts (4.0k, 5.8k, 7.7k, 12.5k tokens), top 8 tokens renormalised: 0.0048
  for the fixed FP32 pass, 0.0073 for the int8 pass. Every generated answer was
  identical. This is not a perplexity-grade evaluation.
- `LLAMA_FATTN_TILE_Q8CFG=n` selects one of seven block configurations. 0 is the
  fastest of those tried.

### 3.5 Q8_0 matmul tile

`mmq-config-pascal.cuh` gives Q8_0 a 128-row tile at J=64 (default on Pascal).
Dense Q8_0 matmul 3.84 to 3.10 ms (24% faster); model outputs bit-identical.
Q4_K and Q5_K keep the 64-row tile because the MoE matmuls were 14 to 16% slower
with 128 rows. The same tile is now also set for IQ3_XXS, IQ3_S and IQ4_XS (the
quantisations of the dense Qwen3.8-27B IQ3_XXS file): 25 to 30% faster
(8.2 to 9.2 -> 10.4 to 11.6 TFLOPS at 2,048 tokens, section 7); the stock path's
first-token logprobs and greedy text are bit-identical to the old tile.

### 3.6 Layer-major threshold and canonical chunks

- The break-even between the ordinary path and the pass is about 570 tokens here.
  The default `--lm-prefill-fuse-min 700` is about right. Forcing the pass for
  every size (`--lm-prefill-fuse-min 1 --lm-prefill-min 2`) made an 8-token prompt
  14 times slower (0.21 s to 3.07 s) and gained nothing above 570 tokens. Do not set
  `--lm-prefill-min 1`: it would also catch 1-token decode steps.
- `--lm-prefill-canonical` re-prefills up to 2,048 tokens on every follow-up. A
  3-token follow-up to a 4k-token conversation took 3.2 s with it and 0.12 s
  without. A growing seven-turn conversation was 6% faster without it. Cold prompts
  are 4 to 5% slower without it, and a cache hit is no longer bit-identical to a
  fresh run.
- `--lm-prefill-prearm auto`: turns 3 to 7 of a growing conversation averaged
  6.26 s against 6.68 s with it off (6%).

## 4. Tried and dropped (measured)

| Idea | Result |
|---|---|
| f16 instead of q8_0 KV | Same kernel time at every depth |
| `--lm-prefill-ubatch` 1024, 2048, 4096 | 503, 549, 538 tok/s before int8; 673 and 698 after (+3.7% for a larger buffer) |
| 128-row MMQ tile for Q4_K, Q5_K, Q6_K | MoE matmuls 14 to 16% slower |
| Cap on the MoE column tile (J 48 down to 16) | Slower at every value |
| Int8 kernel with 128 threads or 64 K rows per iteration | 1.6 to 7.5 times slower (register spills) |
| Whole 32-row V tile (fewer barriers) | No change |
| `--load-mode none` | No faster; 60 s to load against 4 s |
| Forcing the pass for all sizes | Section 3.6 |

## 5. Limits of this evidence

- One machine, mostly single runs. Treat differences under about 5% as noise.
- Quality: four prompts, first-token statistics and needle questions. No perplexity.
- A full 262,144-token prompt was not run; the 100k run used a 131,072-token
  server, and the 262,144-token server was tested with prompts up to 19k.
- Not tested: other models, f16 KV with the int8 flag (the path needs a q8_0 K
  cache, head size 256 and the FP32 tile kernel), tensor-core GPUs, Windows.
- Remaining gaps: the PV step is FP32 and instruction-issue bound (about 2.7
  instructions per cycle); the MoE matmuls are far from the `dp4a` peak.
- `ncu` needs root for hardware counters here; kernel behaviour was read from the
  machine code and `nsys`. Profiling used `kernel.perf_event_paranoid=1`.

## 6. Reproduce

```bash
T=build/llama-cuda/bin/test-backend-ops
$T test -o FLASH_ATTN_EXT -b CUDA0
LLAMA_FATTN_TILE_Q8QK=1 $T test -o FLASH_ATTN_EXT -b CUDA0
$T perf -o FLASH_ATTN_EXT -b CUDA0 -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=[0-9]+,nb=2048'
$T perf -o MUL_MAT_ID -b CUDA0 -p 'n_mats=256,n_used=8'
```

The perf cases for this model's attention and matmul shapes, and the causal cases
at head size 256, were added to `tests/test-backend-ops.cpp`.

## 7. Dense Qwen3.8-27B (UD-IQ3_XXS) on the 8 GiB card

`Qwen3.8-27B-UD-IQ3_XXS.gguf` (11.9 GB, architecture `qwen35`, 64 layers, 16 of them full attention
and 48 Gated DeltaNet, 155 MiB per layer, no routed experts) is larger than the card. The layer-major
prefill and a streamed decode were extended to it on 2026-10-19 (plan and contract:
`docs/plans/dense-qwen38-iq3-residency.md`; scripts: `tools/dense/`). All numbers are one session on this machine,
`-ctk q8_0 -ctv q8_0 -fa on -np 1 -t 4`, one GPU, greedy, needle prompts.

### 7.1 Run it

```bash
CUDA_VISIBLE_DEVICES=0 $REPO/build/llama-cuda/bin/llama-server \
  -m $MODELS/Qwen3.8-27B-UD-IQ3_XXS.gguf --host 127.0.0.1 --port 8099 \
  -dev CUDA0 -sm none -mg 0 -c 16384 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 4 -ub 512 \
  --lm-prefill-host-layers 0-30 \
  --lm-decode-stream 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 --lm-decode-slots 2 \
  --lm-prefill --lm-prefill-fuse --lm-prefill-ubatch 1024 --lm-prefill-segment 2048 --lm-prefill-attn-int8
```

- `--lm-prefill-host-layers 0-30`: 31 layers stay in host memory (page-locked), 33 are resident. The KV cache and the
  recurrent state of all layers are on the device. The server refuses loudly when a layer is split or the KV is on the host.
- `--lm-decode-stream`: 16 of the host layers are computed on the device in decode steps from a ring of 2 slots that a copy
  stream refills while the CPU computes the other 15 host layers. The count must be a multiple of the slots.
- `python3 -I tools/dense_plan.py --ctx 16384 --pass-mib 450` proposes the three lists for the card, the context and the room
  the prefill pass wants (its cost model matched the measured decode within 2 to 6%).
- The pass declines (and says why) when the free memory does not cover it; the ordinary path then takes the batch.
- `--lm-prefill-attn-int8` is the int8 attention approximation (section 3.4): +9% prefill at 16k tokens (1.19x the best stock
  arm; 1.08x without it). Validated with the f32 slab (continuation rule 8/8, first-token KL at the noise floor of any numerical
  change). Do NOT combine it with `--lm-prefill-bf16`: that pair failed the continuation rule on 1 of 8 prompts. A segment of 2,048
  with the f32 slab uses the same memory as 4,096 with bf16.

### 7.2 What was measured

| What | Result |
|---|---|
| IQ3_XXS / IQ3_S / IQ4_XS matmul, 2,048 tokens | 8.2 to 9.2 -> 10.4 to 11.6 TFLOPS (+25 to 30%), stock-path outputs bit-identical |
| Cold prefill, best stock (`-ngl 36 -ub 2048`) | 145.6 / 140.6 / 128.2 tok/s at 3.7k / 10k / 18k tokens |
| Cold prefill, layer-major pass, exact attention | 170.5 / 159.0 / 138.9 tok/s (1.17x / 1.13x / 1.08x) |
| Same with `--lm-prefill-attn-int8`, f32 slab | 1.18x at 3.7k, **1.19x at 15.7k** (161.1 against 135.6 tok/s, A-B-B-A) |
| Pass with 26, 16, 6 layers resident | 170.5, 170.0, 168.1 tok/s at 3.7k; peak VRAM 7.0, 5.5, 3.9 GB |
| Decode, stock default placement, `-c 16384` (36 of 66 layers on the GPU) | 1.62 tok/s (6.8 GB) |
| Decode, stock default placement, `-c 262144` (the native maximum; 14 of 66 layers on the GPU, 7.1 GB of KV in host memory) | 1.03 tok/s (6.7 GB) |
| Decode, same VRAM, 20 of 37 host layers streamed (`-c 16384`) | 1.94 tok/s (1.20x the default placement) |
| Decode, 16 of 31 host layers streamed, 2 slots (`-c 16384`) | 2.35 tok/s, 7.3 GB (1.45x the default placement) |
| Decode, owner's plan: only the 16 attention layers on the GPU, nothing resident | 1.03 tok/s at 2.2 GB |
| Combined command above | prefill 172 / 165 tok/s at 3.7k / 10k, decode 2.34 / 2.32 tok/s, peak 7.7 GB |

Streamed layers give bit-identical tokens and logprobs to the same layers resident (8 prompts). The pass matches the ordinary
path at an equal chunk shape (first-token logprob within 0.0015, 8 prompts).

### 7.3 Why it is like this

- The pass is GPU-compute bound, not copy bound: the weight copies are hidden (25 ms of slot waits in a 21 s pass). At 18k tokens
  the GPU kernel time is the same for the pass and for the stock path (123 s against 124 s); the pass only removes about 14 s of copy
  stalls, so its advantage over stock shrinks with depth (1.17x -> 1.08x). Attention is 22% of the pass at 18k tokens and grows
  with depth (`--lm-prefill-attn-int8` makes it faster, section 3.4). The bound is (K + C) / K with K the GPU kernel time both paths
  share and C the stock path's copy stalls, so no schedule beats about 1.1x at depth with exact kernels.
- Decode is CPU bound: one CPU layer costs 16.5 ms (IQ3 dequantisation on 4 cores, 9 GB/s), a streamed layer costs 25 ms of PCIe
  Gen3 x8 (6.45 GB/s) but overlaps the CPU layers, so CPU and PCIe work together: `T = max(base + C*c*1.15 + (R+S)*g, S*d)`.
- The owner's idea (only the full-attention layers on the GPU) is not lighter: the FFN is two thirds of every layer, an attention
  layer is as big as a DeltaNet layer. It frees VRAM (about 170k tokens of q8_0 KV fit beside it) but the CPU then computes 48
  layers: 1.03 tok/s.

### 7.4 Limits

- One model and quantisation, `-np 1`, contexts up to 20k tokens in the benchmarks, prompts up to 18k tokens. No perplexity.
- The CPU is the decode ceiling. The same model decodes at 0.85 tok/s on the CPU alone.
- The decode ring uses the markers in the ordinary decode graph; CUDA graphs are not used for those graphs (they are off on Pascal anyway).
- A graph of 32 or more tokens does not use the ring (the scheduler offloads per op there).
