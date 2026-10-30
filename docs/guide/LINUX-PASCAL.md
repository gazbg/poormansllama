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
LLAMA_FATTN_TILE_Q8QK=1 CUDA_VISIBLE_DEVICES=0 \
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
- The expert team defaults to the number of physical cores the process may use, at most 8 (Linux, since
  2026-10-22; section 3.2), so `LLAMA_MOE_HYBRID_LEAN_CPUS` is no longer needed. It never pinned threads on
  Linux (the pinning code is Windows-only); only its number of entries counted, as the team size.
  `LLAMA_MOE_HYBRID_THREADS=n` sets the size explicitly.
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

### 3.2 Decode: size the expert team to the physical cores

The expert team defaults to 8 threads (`LLAMA_MOE_HYBRID_THREADS`), which are busy-spin
workers. On a CPU with 4 cores and 8 hardware threads, with the decoding thread, the
engine thread and the GPU driver threads also needing CPU, they preempt each other for a
full scheduler timeslice (the stall log shows 2.4 to 3.7 ms). `LLAMA_MOE_HYBRID_LEAN_CPUS=2,3,1,0`
(four entries) took decode from 5 to 8 tok/s to 30 to 35 tok/s. The generated text was
identical. A profile showed the team threads spending about 99.7% of their time in
barrier spin loops, not in expert math.

What actually does it, checked 2026-10-22 (Qwen3.6-35B-A3B Q4_K_M, MTP, two 192-token chat
requests per arm, i7-4770K): with no variable the process has 29 threads, the stall log
reported 4,988 slow lean requests and decode was 6.0 and 7.8 tok/s. With the 4-entry list, or
with `LLAMA_MOE_HYBRID_THREADS=4` and no list, the process has 25 threads, there were 0
warnings and decode was 30.0 / 33.1 and 30.2 / 33.4 tok/s. In all three cases every thread of the
process had `Cpus_allowed_list` 0-7: nothing is pinned on Linux (every affinity call in the hybrid
code is `SetThreadAffinityMask`), so the CPU numbers in the list are ignored and only the team
size matters. Team sizes other than 4 were not tested.

Since 2026-10-22 the default on Linux is the number of physical cores the process may use (read from
the sibling lists in sysfs, capped at 8; a failure to read them keeps 8). The same command with no
variable then logged `the expert team defaults to 4 threads`, ran 25 threads with 0 stall warnings and
decoded 30.2 / 33.4 tok/s; an explicit `LLAMA_MOE_HYBRID_THREADS=8` still gives the old 29-thread process.

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

With `LLAMA_FATTN_TILE_Q8QK=1`, head size 256, a q8_0 or q4_0 K cache and the FP32 tile
kernel, QK^T runs on `dp4a` against the raw K blocks, with the query columns
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

A q4_0 K cache (`-ctk q4_0`) takes the same path (added 2026-10-22). A q4_0 block is
unpacked when the K tile is loaded to int8 `(nibble - 8)` with its own scale, which is
exactly the value it stands for, so the shared layout, the scales and the `dp4a` core
are those of q8_0 and the query is quantized the same way. It needs no extra VRAM, and
it works together with `--fattn-head-tile`. Only the layer-major pass reaches it (no
mask tensor, so no query-head grouping).

- Unit test (`FLASH_ATTN_EXT`, head size 256, causal window, q4_0 K and V): NMSE
  against the CPU reference 3.8e-8 with the flag, 1.6e-6 without (limit 5e-4). The q8_0
  case gives 3.6e-8 and 1.5e-6. The kernel trace shows the new variant
  (`flash_attn_tile<256,256,32,1,0,0,1>`) is what runs.
- Qwen3.8-27B Q6_K_XL, q4_0/q4_0 KV, MTP, `-c 64000`, eight fixed prompts of 1.6k to
  4.8k tokens against the exact kernel: flag off, 8 of 8 bit-identical (the default path
  is untouched). Flag on, the first token is the same on 8 of 8 (worst |dlogprob|
  0.0084) and the continuation is identical on 6 of 8. The other two diverge in free text
  after the answer; without MTP, where every token carries logprobs, the exact kernel's
  top-2 gap at the first difference was 0.008 nats (a near-tie) and **0.318 nats, above
  the 0.25-nat rule**. So one prompt of eight does not meet the rule.
- Speed was not benchmarked for q4_0. The equivalence run logged prefill rates 0.6 to
  3.2% higher with the flag at these lengths (single runs, not a controlled A/B);
  attention is about 5% of the kernel time at 4k tokens, so the gain is small there.

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

### 3.7 Decode and prefill with MTP, 2026-10-24 (measured)

Command: the owner's MoE command (`-c 262000 -ctk q8_0 -ctv q8_0 --hybrid-moe --lm-prefill --lm-prefill-ubatch 2048
--lm-prefill-segment 4096 -b 4096 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32 --fattn-head-tile --lm-prefill-attn-int8
--spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-mtp-defer`),
greedy decode of 256 tokens on three chats and after a 16k-token prompt, A-B-B-A arms (`tools/pascal/`).

Where a decode round went before the changes (nsys, one round = one MTP verify step of up to 4 token rows plus up to 3 drafts,
about 105 ms): the GPU waited for the CPU expert leg about 47 ms; the matrix-vector kernels took about 22 ms, 12 ms of it the
output head (Q6_K, 248,320 rows, 398 MiB, read once per verify and once per draft); the cached experts about 10 ms; the GPU
was idle about 11 ms between two verify steps (draft MoE on the CPU, sampling). The CPU leg is near its hardware: one expert
costs 172 us of dot products on one thread (Q4_K row of 2048: 89 ns; Q5_K row of 512: 39 ns, `tools/pascal/bench_vecdot.cpp`),
43 us on four threads, against about 90 us of DRAM reads (1.81 MiB at 21 GB/s); the team measures about 112 us per expert.

Three changes are kept (all on by default):

- **One host job per verify step.** The rows of a verify step post one request each for the same layer; the engine now serves
  the requests already posted as one job of the lean team (`llama-moe-lean.h`), which reads each missed expert once and uses each
  weight row for every row that selected the expert. Each row's sum keeps its own expert order: the result is bit-identical to
  one job per row (`test-moe-lean-expert`, 864 rows with shared experts). Multi-row jobs read 19% fewer experts than their (row,
  expert) pairs; a 4-row job costs 1,384 us against 4 x 394 us. Decode +1.9 / +1.9 / +2.2% on the chats, +1.6% at 16k, the same
  text in all four arms. `LLAMA_MOE_HYBRID_LEAN_BATCH=0` restores one job per request.
- **A small ubatch for a deferred MTP head.** With `--spec-mtp-defer` the draft context only decodes the sampled token and the
  accepted drafts (at most n-max + 1 rows), but it was reserved for the target's 512-token ubatch, which made the scheduler keep a
  VRAM copy of the MTP layer's expert bank for offload: compute buffer 564 MiB. It now takes `max(16, n-max + 1)` (below the
  offload threshold of 32): 266 MiB, and the expert cache grows from 41 to 45 slots per layer (hit rate 59.0% -> 63.0%). A round
  costs 3.2 / 5.4 / 3.2 / 6.0% less. The text changes (a different set of resident experts changes the numerics), so decode in
  tok/s moved -4.7 / +2.3 / +0.6 / +7.3%: one chat drew a text the head predicts worse (acceptance 77% -> 68%).
- **More rows per block in the Pascal matrix-vector kernel.** On Pascal a block of `mul_mat_vec_q` that finishes its row in two
  K iterations costs more to schedule than to run: the output head launched 248,320 blocks and moved 139 GB/s. Rows of at most two
  K iterations now take one row per warp (one column; also 2 to 4 columns for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0), when the matrix
  has at least 4,096 rows that divide evenly. Output head 3.13 -> 2.32 ms (178 GB/s); Q8_0 8192 x 2048: 111 -> 87 us (one
  column), 106 -> 96 us (four columns). Q6_K at four columns was 7% slower with it and keeps the old blocks. Decode +3.2 / +2.5 /
  +2.1 / +1.7%, the same text in all four arms. `test-backend-ops` has eval cases for the new path and its 4,098-row fallback.

All three together, the owner's command, the build before the changes (A) against the build after (B), A-B-B-A:

| Measurement | A (tok/s) | B (tok/s) | B / A | ms per round A -> B | draft acceptance A -> B |
|---|---|---|---|---|---|
| Decode, chat 1 | 30.80 | 32.55 | 1.057 | 97.8 -> 92.5 | 0.695 -> 0.687 |
| Decode, chat 2 | 29.48 | 31.30 | 1.062 | 94.4 -> 88.0 | 0.614 -> 0.604 |
| Decode, chat 3 | 32.38 | 35.28 | 1.090 | 102.7 -> 94.2 | 0.799 -> 0.806 |
| Decode after a 16k-token prompt | 21.95 | 24.84 | 1.131 | 107.0 -> 103.1 | 0.467 -> 0.538 |
| Prefill, 4,096 tokens | 476.7 | 468.2 | 0.982 | | |
| Prefill, 16,384 tokens | 574.6 | 568.4 | 0.989 | | |

Two of the four texts are the same in all arms, two differ (the cache size changes which experts are resident). Prefill is 1
to 2% slower; the likely cause is the cache regrow that the prompt's last 4-token chunk triggers inside the prompt time, which
now uploads 45 slots instead of 41. Hit rate 59.1% -> 60.6% on this mix, peak VRAM 6,803 -> 6,795 MiB.

Prefill, measured with flags only:

| Prompt | baseline | `--lm-prefill-ckpt-chunk 24` | `-b 16384` |
|---|---|---|---|
| 4,096 tokens | 466 tok/s | 601 tok/s (+29%) | 465 |
| 16,384 tokens | 565 | 609 (+7.8%) | 576 (+2%) |
| 32,768 tokens | 492 | 507 (+3%) | 494 |

Without it, the server cuts the last 516 tokens of every prompt into batches of their own for the context checkpoints; the
512-token one takes the ordinary path, which copies all 40 expert banks over PCIe (17 GB at 6.4 GB/s, about 2.3 s per prompt in
the profile). `--lm-prefill-ckpt-chunk 24` makes that chunk 24 tokens, which run on the CPU. The cost: the checkpoint sits 28
tokens before the end of a prompt instead of 516, and a cache hit then differs from a fresh run by a little more (2 of 3 frozen
pairs at 0.261 / 0.281 nats against the 0.25 rule, measured on the RTX 5090 in G32, `AGENTS.md`). Recommended for speed; leave
it out when a cache hit must match a fresh run closely.

The pass itself is compute-bound (16k-token profile): flash attention 20.5% of kernel time, routed-expert matmuls 34.7% (Q4_K
gate / up at 4.4 TFLOPS, Q5_K down at 3.1), dense Q8_0 projections 16.4% (11.4 TFLOPS), gated delta net 9.8%. The expert
matmuls are the lever left (a third of the efficiency of the dense ones); a wider column tile (J 96 / 128) made them 5 to 6%
slower. Attention grows with depth: a 4,096-token pass took 5.4 s at depth 0 and 9.4 s at 24k.

Why the expert matmuls run at half the dense speed (`test-backend-ops perf`, 2048 tokens, 256 experts, 8 used; the same 34.4
GFLOP as one dense matrix):

| Format | dense TFLOPS | expert shape TFLOPS | penalty |
|---|---|---|---|
| Q8_0 | 11.3 | 6.1 (gate / up), 5.4 (down) | 1.8x, 2.1x |
| Q4_0 | 9.7 | 5.1, 4.0 | 1.9x, 2.4x |
| Q4_K | 8.7 | 4.7 (gate / up) | 1.85x |
| Q5_K | 8.0 | 3.55 (down) | 2.25x |

The MoE shape costs about twice on every format; the K-quant format itself only 1.3 to 1.4x. Two suspects were tested. (1)
Empty blocks: the grid is sized from the token count (`ncols_max` = 2048, 32 column tiles of 64 per expert) although an expert
gets about 64 tokens, so about 95% of the blocks exit at once. Cutting them 16x with the work per expert fixed (16 experts and
128 tokens instead of 256 and 2048) moved Q4_K from 4.69 to 4.87 to 5.04 TFLOPS: at most 5%. (2) Tile padding: with about 64
tokens per expert, half of the experts spill a few columns into a second 64-column tile. Raising the tokens per expert (32
experts, 256 to 2048 tokens) raised Q4_K from 4.90 to 5.80, 6.42 and 6.71 TFLOPS (about 64, 128, 256 and 512 tokens each) and Q8_0
from 6.26 to 8.79. A model of the tile counts gives 32% padded columns for uniform routing and 34% with the expert popularity of
the decode routing trace (a prompt's popularity was not measured); a last tile sized to the leftover columns would leave 5%.
That is also what `--lm-prefill-ubatch 4096` buys (more tokens per expert), at the price of a larger compute buffer.

Remainder tiles (`GGML_CUDA_MMQ_REMAINDER`, on by default for GPUs older than Volta since 2026-10-29; `=0` turns them off). The last tile of an expert, the columns left after its full
64-column tiles, is computed in a tile as wide as the leftover needs. Mode 1 uses widths 16 / 32 / 48 / 64 (an expert takes
exactly one remainder tile, so it reads its weights as often as before), mode 2 uses 32 / 64, mode 3 uses 24 / 40 / 64. It
applies to MoE matmuls whose rows are a multiple of 128, more than 64 tokens and tiles without stream-k (Pascal); anything else
runs as before. The flag is read at every launch.

Flat tile list (`GGML_CUDA_MMQ_ID_FLAT`, on by default for GPUs older than Volta since 2026-10-30; `=0` turns it off; H37). The main launch of an expert matmul had a grid of
(row tiles, ntx = tokens / 64, all experts), 65,536 blocks for the Q4_K gate / up matrices and 262,144 for the Q5_K down matrix at a 2,048-token chunk, while an expert gets about 64 tokens,
so 96% of the blocks found nothing to do and exited, after holding an SM slot for the read of the expert bounds. The grid now lists only the tiles that hold columns (a one-block
prefix sum over the experts' tile counts, a binary search per block, at most tokens x experts used / 64 + experts tiles): the op is 4.6% (Q4_K) and 30% (Q5_K down) faster at
2,048 tokens, prefill +3.6% at 4k and +4.1% at 16k (A-B-B-A, no overlap), bit-identical (first-token log-probabilities equal, `check_mmq_remainder` hashes equal, MUL_MAT_ID 887/887, MUL_MAT 1,218/1,218).

- Matmul (`test-backend-ops perf`, 2048 tokens, 256 experts): Q4_K gate / up 7.25 ms -> 6.54 (mode 1), 6.88 (mode 2), 6.72
  (mode 3); Q5_K down 9.46 ms -> 8.99, 9.05, 9.11.
- Prefill, the owner's command, A-B-B-A, mode 1 against off: 4,096 tokens 470.8 -> 488.8 tok/s (+3.8%), 16,384 tokens 573.5 ->
  594.4 (+3.6%). Decode +0.2 / +0.2 / +1.1%, the same text in all four arms, peak VRAM unchanged (6,795 MiB).
- Exactness: `tools/pascal/check_mmq_remainder.cpp` runs the same `MUL_MAT_ID` with the mode off and on: 30 comparisons (Q4_K, Q5_K,
  Q6_K, Q8_0, Q4_0, IQ4_XS; uniform and skewed routing; every leftover class populated) are bit-identical. `test-backend-ops`
  `MUL_MAT_ID` 887 of 887 and `MUL_MAT` 1,218 of 1,218 pass in every mode (new eval cases have rows that are a multiple of 128).
  Registers and stack of the J = 64 kernels are unchanged (113 / 119 / 115 registers, no stack).

Why this is +4% and not the +30% that the column count suggests: a tile has a fixed cost (loading and unpacking its weights,
writing its result back) that does not shrink with its columns, so a 16-column tile costs far more than a quarter of a 64-column
one. One fixed cost per tile reproduces the three modes within 1% (1.6 times the column work of a full tile for Q4_K gate / up,
3.7 times for the Q5_K down matmul, whose K of 512 gives a tile only two K steps). With that cost even a leftover tile for every
multiple of 8 columns would give 10%, and mode 1 gets 9.9%. So the expert matmul is limited by the NUMBER of tiles, not by the
columns they hold; the next step is cheaper fixed work per tile (the unpack of the weight tile), not less padding.

### 3.8 Decode tuning with MTP: draft confidence, expert team, fit margin (2026-10-29)

Three settings, no code change, take the owner's command from 32.6 / 31.4 / 35.3 tok/s to 39.6 / 37.6 / 39.3 on three chats and from 24.9 to 31.7 tok/s after a 16k-token prompt:

- `--spec-draft-p-min 0.7` (the command has 0.1). The draft chain stops when the head's best token has a probability below p-min, so a doubtful draft is not verified.
- `LLAMA_MOE_HYBRID_THREADS=6` (the default is the 4 physical cores).
- `-fitt 512` (the default margin is 1,024 MiB; 256 gives a little more and leaves 540 MiB free).

Measured A-B-B-A, the owner's command against the command with these three settings (the two A arms and the two B arms do not overlap in any cell):

| | before | after | change | ms per round | draft acceptance |
|---|---|---|---|---|---|
| Decode, chat 1 (tok/s) | 32.55 | 39.59 | +21.6% | 92.5 -> 67.4 | 0.687 -> 0.941 |
| Decode, chat 2 | 31.38 | 37.59 | +19.8% | 87.7 -> 64.3 | 0.604 -> 0.829 |
| Decode, chat 3 | 35.27 | 39.30 | +11.4% | 94.3 -> 74.9 | 0.806 -> 0.914 |
| Decode after a 16k-token prompt | 24.90 | 31.70 | +27.3% | 102.8 -> 58.5 | 0.538 -> 0.837 |
| Prefill, 4,096 / 16,384 tokens | 466.3 / 566.9 | 455.5 / 560.0 | -2.3% / -1.2% | | |
| Peak VRAM / expert cache | 6,795 MiB / 45 slots | 7,305 MiB / 52 slots | +510 MiB | | |

The costs: 510 MiB of the card's headroom and 1 to 2% of the prefill rate (a bigger cache is shrunk and re-uploaded around every large prompt). Safety: a 32k-token and a 60k-token prompt and a decode at 32k depth all completed with `-fitt 512` and with `-fitt 256` (no error, no memory-guard decline); the peak was 7,321 and 7,569 MiB of the 8,107 MiB.

Why `--spec-draft-p-min` matters so much here: a draft token and its verify row cost about 20 ms per round on this machine (the draft layer's experts and the 398 MiB output head run on the CPU / are read again for every draft, and the verify row adds expert misses for the CPU leg) and yield about 0.4 tokens at the default threshold. n-max 2 or 1 gave the same direction (n-max 2: +6 to +13%, n-max 1: +37% at 16k depth); a confidence threshold keeps depth where the text is predictable and drops it where it is not. Screening of the threshold (n-max 3, tok/s change on the three chats and at 16k depth): 0.3 -0.5 / +0.2 / +2.2% / +10.3%; 0.5 +7.9 / +5.1 / +5.5% / +20.5%; 0.6 +11.8 / +7.2 / +6.7% / +29.5%; **0.7 +10.7 / +7.9 / +8.0% / +30.6%**; 0.8 +7.2 / +7.0 / +1.0% / +31.5%; 0.9 +6.4 / +5.4 / +3.9% / +30.6%.

An optional fourth setting, `-ot "blk\.40\.ffn_(up|gate|down)_exps\.weight=CUDA0"` (the MTP layer's experts in VRAM, 430 MiB): on top of the three settings, eight chats
and the 16k decode, A-B-B-A, tok/s +3.7 / +4.8 / +8.1 / +0.9 / +4.9 / +5.7 / +13.1 / +3.5% (mean +5.6%) and +4.6% after the 16k prompt, ms per round -3.2% on average
(7 of 9 cells without overlap), prefill +2.0% at 4k, peak VRAM unchanged. The cache ends at 45 slots instead of 52 (hit rate 0.617 against 0.649) and the text differs from
the control in every cell, because a different set of experts is resident. The 32k and 60k prompts of the safety gate were not rerun with it.

Decode at depth (2026-10-30). The three settings above were tuned at 0 to 16k tokens of context. At 32k, on one text, `--spec-draft-p-min 0.9` beat 0.7 (a result later qualified, see below), A-B-B-A on a 32,791-token prompt, 27.84 -> 28.73 tok/s (+3.2%), ms per round 66.2 -> 56.8, acceptance 0.80 -> 0.92 (the text differs). At shallow depth 0.7 is better (section above), so use 0.7 for chats and short contexts; 0.5 is -9% at 32k. Later sessions qualified a rule 'use 0.9 from 32k tokens': the gain of 0.9 alone is not a function of depth (H33, each depth a different cut of one text: -0.6% at 24k, +3.2% at 32k, +6.7% at 49k, -0.4% at 64k; ms per round falls 10 to 16% at every depth from 24k, but tok/s follows the text), an automatic rule was built and rejected, and it belongs to the text: on four texts at 32k tokens (H35) 0.9 against 0.7 is +3.2% / +2.4% / -1.3% / -3.5% (mean +0.2%, no overlap in any of the four) while ms per round falls 14 to 20% every time, because the shorter drafts also carry fewer tokens per round. Keep 0.7 at every depth unless you measure your own texts. `LLAMA_FATTN_GQA8=1` (packed decode attention) does nothing at 49k (26.17 -> 26.05 tok/s, same text): the host leg, not the attention, bounds the decode here. At 32k, A-B-B-A against the stack: the MTP layer's experts on the GPU 27.73 -> 29.05 tok/s (+4.7%), `--spec-draft-p-min 0.8` +1.9% (27.70 -> 28.23, ms per round -12.6%), and 0.9 together with the MTP experts on the GPU +7.2% (27.88 -> 29.89, ms per round -18%, acceptance 0.80 -> 0.93): the best recipe measured on that one text (see below for other texts). At 64k tokens (H34, A-B-B-A, 256 tokens) the same recipe gives 25.45 -> 26.66 tok/s (+4.7%, ms per round 77.4 -> 62.8, acceptance 0.85 -> 0.93, no overlap); p-min 0.9 alone is -0.4% at 64k and the MTP-experts flag alone is 0.0% there (H36, 25.29 -> 25.30), so the +4.7% is an interaction of the two. At 32k the flag alone is +4.7% (base text), +4.3% (guide text) and +5.6% (code text), the two texts on which p-min 0.9 alone loses 3.5% and 1.3% (H36): the flag is the part that is robust across texts at 32k, mean +3.3% over the three cells counting 64k. H38 then put p-min 0.9 on top of the flag (A-B-B-A, 256 tokens): +3.6% at 64k on the default text, -1.5% on the guide text and -2.3% on the code text at 32k (rounds 16 to 19% shorter in every cell, fewer tokens per round), mean -0.1%: rejected. For contexts of 32k tokens and more the advice is `-ot blk\.40\.ffn_(up|gate|down)_exps\.weight=CUDA0` with `--spec-draft-p-min 0.7`; try 0.9 only after measuring your own texts. The flag's gain is solid at 32k (+4.3 to +5.6% on three texts) but text-bound at 64k (H65 / H66: 0.0%, -0.5%, +4.4%, +4.9% and +2.2% on five texts, mean +2.2%, range -0.5% to +4.9%; the first two texts it was repeated on missed a +2% bar by 0.05 points, the pooled five meet it), so at 64k and beyond measure it on your own texts. The memory gate with the recipe passed at 32k and 61k tokens (exit 0, peak VRAM 7,321 MiB, no error or decline in the server log, the cache resized between 21 and 45 slots). One deep text and 256 tokens per arm.

Why a round costs more at depth, and the opt-in fix (H40, H44; 2026-10-30). An nsys profile of the decode at 8k and 49k tokens shows the GPU kernel time growing by 8.6 of the 9.2 ms of wall time per token,
and two thirds of it is attention: a verify batch has 2 to 4 rows (`--spec-draft-n-max 3`); with 1 or 2 rows the quantized-KV rule picks the vector path and the packed GQA kernel (`fattn_gqa8`,
on by default with `--hybrid-moe`), but with 3 or 4 rows (about 40% of the rounds) it picks the tile kernel, which first converts K and V of each KV head to f16 (`--fattn-head-tile`): 5.3 ms
per such round at 8k tokens, 27.4 ms at 49k. `LLAMA_FATTN_GQA8_MAXROWS` (opt-in `=4`: it was the default from H48 to H58 and went back to 2 when H58's four further texts failed the pre-registered KL bar; it needs `LLAMA_FATTN_GQA8=1`, which `--hybrid-moe` sets and plain runs do not) sends the 3-4 row batches to the packed kernel too:
decode A-B-B-A +6.1% / +7.9% / +11.0% / +10.6% at 8k / 32k (guide text) / 49k / 64k tokens, ms per round -5.7% to -11.6%, no overlap in any cell. It is not exact: the packed kernel quantizes Q to
8 bits like the vector path (the tile kernel keeps f32), a KL divergence of 0.0070 against the tile routing on 4,096 tokens decoded 4 rows at a time, same top token 95.2%, perplexity unchanged (ratio 0.9988 +- 0.0034).
That KL is the floor of the test, not a measure of this kernel (H47): scaling every attention score of the exact f32 kernel by 1 + 1e-6 gives 0.0057 and 96% same top token, because the quantized matmuls re-quantize their
activations at every layer and any rounding flip propagates; the floor is 0.0056 on prose and 0.0020 on code. Against that floor the packed kernel is at 0.98 x and 1.01 x on a prose and a code text (H48), the tile route is as far
from the exact kernel as the packed one is, which is why the packed routing is the default. The text still changes at near ties and the draft acceptance moved by -0.06 to +0.03 (mean -0.02) over the four cells.
`LLAMA_FATTN_GQA8_EXACT_ROWS=3` selects an exact f32 Q.K variant of the packed kernel (no Q quantization): as fast per round as the 8-bit one at 49k tokens, 8 points slower at 8k; a reference, not a speed route. A q4_0 K and V cache cannot use it (the packed kernel is q8_0 only), which is why that cache is
28.9% slower in the deep decode (H43). It stacks with the MTP experts flag and p-min 0.7 (H45, A-B-B-A, old routing against 4 rows, both with the flag): 25.34 -> 28.97 tok/s (+14.3%, ms per round -4.5%, the text accepts more drafts) at 64k and 32.16 -> 34.27 (+6.6%, ms per round -7.0%) at 32k on the code text, no overlap in either cell. End to end (H53, A-B-B-A, the four defaults of the last rounds: flat tile list, 3-4 row verify routing, int8 PV and 4-lane delta net kernel, against the build before them, deep advice flags in both): prefill +14.8% at 16k (624.05 -> 716.20 tok/s) and +19.3% at 32k (537.70 -> 641.70), decode +4.0% at 32k (28.98 -> 30.13, ms per round -4.5%, below the +5% bar of its contract) and +13.6% at 64k (25.30 -> 28.73), peak VRAM equal. The 32k decode cell on three more texts (H54) gives +7.8% (guide), +9.1% (code) and +8.0% (docs), so the four 32k ratios are +4.0 to +9.1%, mean +7.2%: quote the range, the text decides. These decode figures include the 3-4 row verify routing, which was the default when they were measured and is the opt-in `LLAMA_FATTN_GQA8_MAXROWS=4` since H58 (section above); without it only the prefill gains of the end-to-end numbers apply.

The team size result is reproducible and the cause is not known: with the default of 4 threads the engine has no gain from 3, 5 or 6 in theory, but the measurement says 3 gives +4.6%, 5 gives +3.9%, 6 gives +6.1 and +7.1% (two runs), 7 and an explicit 4 give 0 (screening; the texts are identical to the control). The scheduler's placement of the spinning threads on the 8 logical CPUs (4 cores) is the likely reason.

Tried and left out (screening, texts identical or compared by ms per round; the control repeats to within 0.6%):

| Setting | Result |
|---|---|
| `LLAMA_MOE_HYBRID_LEAN_PF` 0 / 3 / 4 (prefetch of predicted experts) | within +-0.6% of the default 2 |
| `LLAMA_MOE_HYBRID_B` 8 / 32, `LLAMA_MOE_HYBRID_LAMBDA` 0.95 / 0.98 | hit rate falls (0.594 / 0.630 / 0.614 / 0.620 against 0.630); rounds 1 to 5% slower |
| `LLAMA_MOE_HYBRID_RECENCY=1` | hit rate 0.610, rounds 0 to 6% slower |
| `LLAMA_CUDA_TOPK_MOE_SMALL=1` | -0.8% per round on average (below the 2% bar) |
| `LLAMA_FATTN_GQA8=1`, `LLAMA_CUDA_INPUT_BATCH=1`, `LLAMA_CUDA_HOP_KERNEL=1`, `LLAMA_MOE_HYBRID_SCAN=1` | within +-0.7% |
| `LLAMA_MTP_DRAFT_VOCAB_N=32768` (the draft head reads the first 32,768 token ids only) | mean +2.9% alone; on the stack it loses 6.9% on the code chat (acceptance 0.953 -> 0.839) and gains 3 to 6% elsewhere |
| `--spec-draft-n-max 4` | not built: the engine handles at most 4 token rows, and the arithmetic above says it would lose |

### 3.9 Time to first token for 700 to 4,000-token prompts (2026-10-29)

On the owner's command a cold prompt of 1,500 tokens takes 9.1 s, longer than one of 2,500 tokens (7.4 s). The layer-major pass takes only batches of `--lm-prefill-min` (1,024) tokens or more, and the server first cuts the last 516 tokens off every prompt for a context checkpoint, so a prompt under about 1,540 tokens runs as ordinary 512-token batches, and each of them copies every expert over PCIe (about 2.3 s). Two flags fix the range:

- `--lm-prefill-ckpt-chunk 4`: the checkpoint chunk is 4 tokens (it runs on the CPU, no bank copy) instead of 512, so the checkpoint sits 8 tokens before the end of the prompt.
- `--lm-prefill-min 512`: a batch of 512 tokens or more takes the pass.

And one variable for short prompts: `GGML_OP_OFFLOAD_MIN_BATCH=112` (default 32). A batch under that size is computed by the CPU instead of copying the used experts to the GPU; the break-even on this machine is about 110 tokens (at 124 tokens the CPU path is already 5% slower).

Measured A-B-B-A, the owner's command against the command with the two flags and the variable (prompt rate, higher is better; all A / B pairs separate):

| Prompt tokens | 48 | 96 | 384 | 700 | 1,000 | 2,500 | 4,000 |
|---|---|---|---|---|---|---|---|
| default (tok/s) | 33.0 | 51.9 | 134.7 | 130.0 | 162.8 | 336.0 | 458.5 |
| with the settings (tok/s) | 45.9 | 53.8 | 128.4 | 161.9 | 229.1 | 534.9 | 635.8 |
| change | +38.9% | +3.7% | -4.7% | +24.5% | +40.7% | +59.2% | +38.7% |

A growing conversation (each request extends the previous prompt with the prompt cache on), wall time in seconds, default -> with the settings: cold 1,500 tokens 9.69 -> 5.01 (-48%); +300 tokens 3.21 -> 3.29; +500 tokens 3.73 -> 3.84; +1,200 tokens 9.02 -> 4.76 (-47%); +100 tokens 2.40 -> 2.28; +1,200 tokens 8.86 -> 4.79 (-46%). The prompt cache is used exactly as before (`cache_n` is identical in every request; 4 extra tokens are re-processed, as with the default). The cost is on turns of 300 to 600 tokens (+3%, about 0.1 s): the 4-token checkpoint is one more batch. A chunk of 8 or 16 costs 0.15 to 0.4 s there for no extra gain; 24 costs 0.4 to 0.5 s. The prefill window (`--hybrid-moe-prefill-ubatch` with `--hybrid-moe-prefill-min 513`) does nothing for this range: the checkpoint cut still leaves a 516-token batch, and with a 4- or 24-token chunk it only adds the cost of opening and closing the window (5.55 s at 700 tokens).

One effect to know: the checkpoint 8 tokens before the end changes which tokens a rollback must re-process when a new prompt diverges inside the last tokens of the old one, and the layer-major pass was validated on the RTX 5090 against a checkpoint 516 tokens before the end (G32); hit-versus-fresh equality with the shallower checkpoint was not re-measured here.

### 3.10 Prefill: cache resize, tile default, gated delta net kernel (2026-10-29)

Measured on the stack of section 3.8 (prefill only, A-B-B-A, no A / B overlap), against the previous build.

- **Cache resize that keeps the slots (on by default).** Around every large prompt the server shrinks the expert cache, because the
  layer-major pass needs the VRAM, and regrows it afterwards. Each of the two used to upload the whole cache over PCIe. Now the slots
  that stay are copied inside the card and only the slots that must leave it are uploaded again. 4,096 tokens 477.8 -> 510.4 tok/s
  (+6.8%), 16,384 tokens 587.9 -> 610.7 (+3.9%). The resizes of a 4k prompt took 1.13 s and now take 0.85 s, those of a 16k prompt 1.19 s
  and 0.62 s; what remains is the upload of the slots that leave the card (1,128 slots, 2,096 MiB, in 438 ms = 5.0 GB/s of the 6.4 GB/s
  the link measures). The contents are identical: `LLAMA_MOE_HYBRID_RESIZE_VERIFY=1` reads every slot back after a resize and compares it with
  the host bank, "0 banks differ" in all 10 resizes of a 4k + 16k run (it adds 1.2 s per resize, a test only). Cost: the final cache has
  one slot fewer (51 instead of 52, 74 MiB; peak VRAM 7,299 against 7,303 MiB) and decode shows no loss (tok/s +0.4 / +2.5 / +2.7% on
  two chats and after a 16k prompt). `LLAMA_MOE_HYBRID_RESIZE_KEEP=0` goes back to the full upload.
- **Remainder tiles are now on by default for GPUs older than Volta** (section 3.7): bit-identical, prefill +3.8% / +3.6% at 4k / 16k in an
  A-B-B-A, +4.3 / +3.7 / +3.6% at 4k / 16k / 32k in single arms. `GGML_CUDA_MMQ_REMAINDER=0` turns them off.
- **Gated delta net kernel with 4 lanes per state column (`GGML_CUDA_GDN_QUAD=1`; the default under `--hybrid-moe` since H51, off elsewhere).** The generic kernel gives a warp to a state
  column and reduces over 32 lanes twice per token. The new one gives 4 lanes to a column (32 rows each in registers, 2 shuffles per
  reduction) and stages k and q of 8 tokens in shared memory; it is used for 16 tokens or more with a scalar gate and no snapshots.
  The op is 2.14 times faster (2,048 tokens, 16 heads: 10,149 -> 4,744 us; 2 or 8 lanes per column: 7.1-7.7 and 6.4-6.7 ms), and the prefill
  is +4.9% at 4k and +6.3 / +6.4% after a 16k / 12k prompt, with +64 MiB peak VRAM at 12k. It sums in a different order, so it is not bit-identical:
  against the generic kernel `llama-perplexity --kl-divergence` (4 chunks of 2,048 tokens) gives a mean KL divergence of 0.0115, the same top
  token in 94.0% and PPL 18.72 against 18.78 (-0.3% +- 0.3%). That is the size of any change of the order of the sums in every delta net
  layer here: the 2-lane and 8-lane variants differ from the 4-lane one by 0.0114 and 0.0116, so the new kernel is as far from the generic one
  as the variants are from each other. A change that keeps the order (16 tokens per stage) gives 0.0000 with 99.98% of the top tokens equal,
  and a smaller micro-batch (`-ub 256`) is bit-identical. The generated text changes (every chat and the deep prompt differed), the decode ms per round averages
  +0.4% over six samples (no systematic effect; the kernel does not run in decode). Use it where a different but equally good text is
  acceptable.
- **K-quant expert tiles at 128 threads (default for these GPUs, 2026-10-29).** The thread count of the expert tiles had not been swept. In
  `mmq-config-pascal.cuh` the Q4_K tiles of 16 to 64 columns and the Q5_K tiles of 16 to 48 columns now run 128 threads at occupancy 3
  (the Q5_K 64-column tile stays at 256 threads, 128 was 3% worse): a thread covers 16 columns instead of 8, so (the likely reason, not isolated) the weights it unpacks and
  loads are used twice as often. Op level (2,048 tokens, 256 experts, 8 used): Q4_K gate / up 6.57 -> 5.92 ms (-10%), Q5_K down 9.1 -> 8.58 ms
  (-4 to -7%); occupancy 4 gives the same, 64 threads is 2x slower. Prefill, A-B-B-A against the previous build, 4k 512.4 -> 523.5 tok/s
  (+2.2%) and 16k 613.5 -> 627.9 (+2.3%). Bit-identical: the output bits of the checker's 10 cases are equal between the old and the new
  libraries. The smaller expert batches gain more (op level, old against new: 128 / 256 / 512 / 1,024 tokens, Q4_K gate / up -19 / -18 / -17 / -14%, Q5_K
  down -26 / -26 / -15 / -8%), which matters for the 512-token ubatches of the ordinary path. The inner loop was measured on its own (`tools/pascal/mmq_q4k_inner_bench.cu`, shared memory only): 9.6 TOPS, of which the min
  term costs 12%, the nibble unpack 12% and the whole scale epilogue 34%; unpacking by hand outside the column loop changes nothing because
  the compiler already does it, and 4-row tiles with 128 or 256 threads are slower (167 / 151 registers).
- **Int8 PV in the attention kernel (opt-in `LLAMA_FATTN_TILE_Q8PV=1`, together with `--lm-prefill-attn-int8`; q8_0 K and V, head size 256).** Removing only the
  PV multiply-adds cuts the int8-QK kernel by 37%, so PV is where the time is. The softmax weights P of a 32-key tile are quantized to int8 per column, the V rows
  of the tile are requantized to one scale per block of 32 head-dim values (the q8_0 scale varies per key, which cannot be taken out of an integer sum over
  keys), transposed to words of 4 keys, and the products are `dp4a`. V is read raw from the cache, so the f16 V conversion goes away. The rounding of P matters:
  round-to-nearest drops every weight below 0.4% of the tile maximum, a bias (first-token KL against exact FP32 attention 0.147 on one of two 32k prompts, top-1
  logprob off by 0.244 nats); a second byte for P repairs that and is not faster than FP32 PV; the kept variant rounds with a deterministic hash-driven dither
  (unbiased). With the int8 PV the QK phase is 28% of the kernel (ablation: 16.9 of 60.8 ms at 10,240 keys), 17.6% for the shared-memory reads of Q (a 16-byte
  broadcast per column, block and half, for every key) and 9.3% for those of K; the `dp4a` arithmetic costs nothing. So the int8 PV runs a 64-key tile (configuration 3):
  each lane holds two keys and every Q read serves both (a second key per Q read cost +6.3 ms in a proxy that doubled the K reads and `dp4a` of the existing kernel,
  against 16.9 ms for the first), and the PV is done in two 32-key halves. Configuration 3 had spilled 488 B of registers and been 3.5 times slower; loading Q
  once per (column, block, half) for all of a lane's keys removed the spill (24 B) and left the 32-key kernels bit-identical. Attention, 2,048 queries, q8_0 KV: FP32 PV
  74.2 -> 59.8 ms (32-key tile) -> 55.3 ms (64-key tile) at 10,240 keys (-25.5%), 262.5 -> 215.5 -> 193.5 ms at 34,816 (-26.3%). Prefill, A-B-B-A, one binary with the
  variable off against on (the variable is the default under `--hybrid-moe` since H49: within the first-token KL floor on nine prompts, 16k +3.9% and 32k +9.3% again in A-B-B-A; `=0` turns it off): 16k +4.2% (618.3 -> 644.3 tok/s), 32k +8.8% (530.6 -> 577.4), 49k +12.7% (451.7 -> 509.0), 98k +19.8% (307.3 -> 368.0), 131k +22.2% (254.1 -> 310.6), no A/B overlap, peak VRAM unchanged; the 16k / 32k / 49k session was repeated on the final build (+4.2 / +8.8 / +12.8%, 617.3 -> 643.3, 526.3 -> 572.6, 447.3 -> 504.6 tok/s), the 98k one was repeated later (H34: 307.1 -> 367.6 tok/s, +19.7%, arms of one kind within 0.07%) and the 131k one was repeated as well (H42: 255.95 -> 311.05 tok/s, +21.5%, no overlap, VRAM +8 MiB); at 150k tokens it is +22.7% (H56: 243.50 -> 298.80 tok/s, no overlap, VRAM equal). At 250,000 tokens (H60, a pair with the default 2-slot window) the memory guard declines the layer-major pass for 14 batches per run late in the prompt, because the lazily committed KV leaves 2.3 GB then 1.7 GB free against the 1.6 GB plus 0.7 GB margin the pass asks for; those batches take the ordinary path (loud, no wrong data), and the pair is only +9.9% (116.4 -> 127.9 tok/s). With `--lm-prefill-slots 1` the declines fall to 4 and the 250k prefill runs at 158.2 tok/s against 128.1 for two slots (+23.5%, A-B-B-A, H63; H61 had 158.7 in a single arm): for prompts beyond about 200k tokens use one slot; on top of one slot the int8 PV adds +19.2% at 250k tokens (132.75 -> 158.25 tok/s, A-B-B-A, H64), a few points under its 150k figure because the 4 remaining declined batches run the ordinary path
  (the 32-key tile gave +2.9 / +5.4 / +8.2% in the session before). Quality, first-token KL (top 20) against exact attention with the 64-key tile: 6 prompts of 8k /
  24k tokens 0.0200 (int8 QK alone 0.0186), 2 prompts of 32k 0.0079 (0.0026), one top-1 flip on a prompt where int8 QK alone also flips; with the 32-key tile
  0.0040 at 4k / 12k (4 prompts), 0.0052 at 32k, 0.0040 at 55k. A paired next-token log-likelihood (54 positions, the true next token as the target) shows no
  loss either: 44 windows of 4,096 tokens, exact attention NLL 1.4008, int8 QK -0.021 +- 0.021, int8 QK + PV -0.022 +- 0.021; 10 windows of 16,384 tokens, 1.2440,
  -0.043 +- 0.036 and -0.109 +- 0.068 (negative is noise; the resolution is about 0.02 and 0.07 nats). At 98,304 tokens of context (one window) the first-token KL against exact attention is 0.00007 for int8 QK and 0.00024 with the PV, the same top token, and the plain-decode prefill takes 449 s, 314 s and 262 s. It is not exact (text changes).
  At 131,072 tokens (one window) the first-token KL is 0.0194 (int8 QK) and 0.0191 (with the PV), the true-token NLL -0.025 and -0.014 against exact attention; a wider paired
  log-likelihood over 144 windows of 4,096 tokens gives int8 QK +0.0053 +- 0.0086 and int8 QK + PV +0.0037 +- 0.0079 (about +0.4% perplexity at a resolution of +-0.8%).
  A second set of 146 windows from another stretch of the text is noisier (digits inside number tables have an exact NLL of 5 to 9, and three of them move by 5.6-8.0 nats, all
  towards a better NLL): int8 QK -0.0246 +- 0.0447, int8 QK + PV -0.1266 +- 0.0764, medians 0.0006 / -0.0001. Pooled over the two sets (290 positions): exact NLL 1.6670, int8 QK
  -0.0098 +- 0.0229, int8 QK + PV -0.0619 +- 0.0388, top token equal to exact attention at 270/290 and 274/290 positions. So no loss is detectable at a resolution of about
  +-0.05 / +-0.08 nats (a loss of 1% perplexity or less cannot be excluded); 5.5-6.9% of positions change the top token.
  `tools/pascal/ftk_arm.py` (`--ppl` for the log-likelihood) and `fa_pv8_check.cpp` are the harnesses. With a q4_0 K and V cache the int8 PV also works (the V rows are
  mapped to +-127 of the tile's largest scale; with the q8_0 scale their 4-bit values were squeezed to a few levels, NMSE 1.2e-3): attention 73.3 -> 59.3 ms at 10,240 keys
  (-19%), 255 -> 216 ms at 34,816 (-15%), NMSE 5e-5 against the FP32 PV; there the 32-key tile is faster than the 64-key one (61.8 against 69.8 ms) and is the default. On the hybrid server with a q4_0 cache the
  prefill gains +2.8% at 16k, +7.8% at 49k and +11.7% at 131k (264.3 -> 295.2 tok/s, A-B-B-A, no overlap).
- **Settings that did not help** (all with `--lm-prefill`): `--lm-prefill-ubatch 4096` on top of the tiles and the resize +1.2% at 4k, +0.4% at 16k (again after the flat tile list: +1.1% / +0.1%, H39);
  `GGML_CUDA_MMVQ_MOE_ROWS=4` (rows per block of the cached-expert matvec) saves 0.4 ms of a 65 ms verify round; 8 or 16 warps per block
  in the generic delta net kernel; 2 MiB pages for the host banks (at most 1.4% in a microbenchmark of the CPU leg, `tools/pascal/bench_stream.cpp`).
  The dense Q8_0 tile was also swept: 64 rows or 128 rows with 128 threads are 18-20% slower than the adopted 128 rows with 256 threads (3.6-3.7
  against 3.03-3.13 ms for 4096 x 2048 x 2048), and a standalone model of its inner loop (`tools/pascal/mmq_q8_inner_bench.cu`) shows why: the 4 x 8
  register tile is shared-memory-load bound at 63% of the 26.9 TOPS that register-only `dp4a` reaches on this card (17.1 TOPS with all scale
  arithmetic removed), the real loop reaches 13.5-14.9, 8 x 4 and 8 x 8 tiles run out of registers (151 / 183) and are slower, and the exact
  magic-number int-to-float replacement is 5-7% slower than the hardware conversion.

Where the prefill time goes now (nsys, 16,384-token prompt, the stack, final build; share of kernel time; 26.4 s of kernels, 25.2 s with the int8 PV): flash attention 22.2% (18.9% with
the int8 PV on the 64-key tile, 5,881 -> 4,748 ms; 19.6% on the 32-key tile), `mmq` Q8_0 17.8%, Q4_K 15.5% (4,664 -> 4,087 ms since the expert-tile change), Q5_K 13.6% (3,759 -> 3,598 ms), gated delta net 10.7%; every other
category is under 5%. Decode is unchanged (host wait 34%, Q6_K head 9%, cached-expert matvecs 11%). The pass is compute bound: kernel time is 26.9 s of the 27.3 s request, and the
bank copies (91 GiB at 6.45 GB/s) overlap with the kernels. The dense Q8_0 matmuls reach 39% of the `dp4a` peak and the inner-loop model above says
the adopted tile is within 20% of what that loop can do. Attention was the one place with a large untried lever (PV in FP32); the opt-in int8 PV above takes it,
and what is left there is the QK phase, which is shared-memory-load bound (section 3.4). In decode (3 chats) the GPU spends 34% of its kernel time spinning for the
host's experts (`moe_hybrid_wait_add`), 8.9% in the Q6_K output head (417 MB in 1.75 ms = 238 GB/s of the card's 256) and 10.5% in the
cached-expert matvecs; the host reads an expert in 88 us (22 GB/s) with four threads.

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
| CUDA graphs on Pascal (the `cc < 7.0` check removed, 2026-10-24) | Decode 0.991 to 0.998 of without; the GPU waits on the CPU leg anyway |
| Skipping the all-zero miss slot in the cached-expert matvec | No measurable change (within 0.3% per round): those reads hit the L2 |
| Pinning the lean team on Linux, one thread per core (`1,2,3,0` or `0,1,2,3`) | 1.0 to 2.5% slower per round than the scheduler's own placement |
| MTP layer's experts in VRAM (`-ot blk\.40\.ffn_(up\|gate\|down)_exps\.weight=CUDA0`) | On the owner's command (2026-10-24): 7 cache slots fewer (45 -> 38), round cost -1.8 to +6.8%: mixed. On the tuned stack of section 3.8 (2026-10-29, eight chats): tok/s +5.6% on average (+0.9 to +13.1%, every chat and the 16k decode up), ms per round -3.2%, prefill +2.0% at 4k, cache 45 against 52 slots, the text differs. Worth adding as an optional fourth setting; the 32k / 60k prompts were not rerun with it |
| Int8 PV with round-to-nearest P (2026-10-29) | 20% faster attention, but biased: first-token KL 0.147 on one of two 32k prompts (0.004 for int8 QK alone). Replaced by dithered rounding |
| Int8 PV with a two-byte P (hi + lo) | Quality repaired (KL 0.0035) but 75.8 against 74.2 ms for FP32 PV: the kernel spills 184 B at 128 registers |
| Int8 QK attention with 64 query columns per block instead of 32 (2026-10-29, built and timed) | 74.4 -> 121.9 ms (256 threads, occupancy 2) and 159.3 ms (occupancy 1) at 10,240 keys, 259 -> 461 / 585 ms at 34,816; 16 warps need 54.5 KB of shared memory (limit 48 KB). Reverted |
| Gated delta net with 8 or 16 warps per block (2026-10-29) | No gain at the op level; reverted. A 4-lanes-per-column kernel is faster (section 3.10) |
| `--lm-prefill-ubatch 4096` on top of the remainder tiles and the keep-slots resize | +1.2% at 4k, +0.4% at 16k (A-B-B-A), under the 2% bar |
| `GGML_CUDA_MMVQ_MOE_ROWS=4` (cached-expert matvec, 4 rows per block) | Faster on the 4-row verify (Q5_K 138.8 -> 127.9 us) and slower on one row; 0.4 ms of a 65 ms round |
| `LLAMA_MOE_HYBRID_COPY_MODE` 0 / 1; `-td` 1 / 2 / 3; `-t` 3 / 5 on the stack | +1.4% at best, inside the 2% drift of the control arm |
| 2 MiB pages for the host expert banks (CPU-leg microbenchmark) | 0 to -1.4% per request; the real banks are file-backed |
| `LLAMA_MOE_HYBRID_LEAN_SPIN_US=1500`; `-td 2` | 0.3 to 1.4% slower; within noise |
| MMQ column tile J 96 / 128 for Q4_K and Q5_K | Expert matmuls 5 to 6% slower |

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

Ideas not pursued (2026-10-30). The ledger of `docs/plans/gtx1070-moe-mtp-optimization.md` has no open row, which only records that every listed idea was closed. These ones exist and were
not tried; the sizes are expectations, not measurements.

- (Tried since this list was written: a draft threshold that follows the context depth was built as an env rule and rejected, because the gain of p-min 0.9 is not a function of depth, H33; the MTP layer's experts in VRAM at depth were confirmed at 32k and 64k and the memory gate with the flag passed at 61k, H34.)
- The largest lever left is not a flag: expert-cache slots. Decode is linear in the cache's miss rate (H68: ms per round = 39.2 + 79.8 x miss, R squared 0.997 over 14 to 56 slots at `-c 8192`; 56 slots is the most one card holds), so a cache that never misses would give 69.7 tok/s against 39.6, and a second card used as extra cache (about 155 slots) would be worth an estimated +35 to +47% (extrapolated). It needs engine work and lifting the one-GPU rule; neither was done.
- (Tried since: a q4_0 K and V cache for deep decode, H43: -28.9% tok/s at 64k (25.44 -> 18.09, ms per round 76 -> 110), one cache slot more instead of 7 to 13; rejected.)
- Stream-k scheduling for the K-quant expert tiles (off in the Pascal configuration): the premise (grid tail effects) is bounded below 1% (5,000 or more active tiles per call over 30-45 resident blocks); the real inefficiency of that grid was its empty blocks, removed by the flat tile list (H37, +3.6% / +4.1% prefill); stream-k itself is not pursued: a profile of a 4,096-token prefill (H50) puts the wave-tail loss of the K-quant expert launches at 1.11% of their time, 0.28% of the kernel time.
- (Closed since: the 4-lane gated delta net kernel did not need to be bit-identical to be the default under `--hybrid-moe`. H51 judged it against the noise floor of the first-token KL (a one-millionth change of the layer output, `GGML_CUDA_GDN_EPS=1e-6`): on nine prompts its mean KL is 0.158 against the bar 0.226 (floor 0.082 +- 0.072), the same top token on 9 of 9, medians 0.0036 against 0.0034, and without the one knife-edge prompt 0.0103 against 0.0106; prefill +4.8% at 4k and +6.5% at 16k (A-B-B-A, no overlap). `GGML_CUDA_GDN_QUAD=0` turns it off.)
- The int8 PV beyond 131k tokens, with a q4_0 cache at 131k (a session was planned by the second instance of the work), and on the draft head's attention: not measured.

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
