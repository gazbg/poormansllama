# Parameters

Every option below is defined in `third_party/llama.cpp/common/arg.cpp` (flags),
`third_party/llama.cpp/common/common.h` (defaults) and the engine sources (environment
variables). Help text is paraphrased from the source. The source is authoritative
when this file and the code disagree. Check with:

```bash
grep -n '"--lm-prefill' third_party/llama.cpp/common/arg.cpp
```

Every flag also has an environment variable `LLAMA_ARG_<NAME>`, with the leading
dashes removed and the rest uppercased, for example `LLAMA_ARG_LM_PREFILL_UBATCH`.

## 1. Build option

| Option | Default | Effect |
|---|---|---|
| `-DGGML_CUDA=ON` | ON in this repo | CUDA backend |
| `-DLLAMA_LAYER_STREAMING=ON` | OFF | Builds the layer-streaming path. Required by every `--hybrid-moe` and `--lm-prefill` flag |

## 2. Core flags used with the feature

These are stock llama.cpp flags. The values below are the ones the record uses.

| Flag | Used as | Effect |
|---|---|---|
| `-m` | model path | The GGUF file |
| `-c`, `--ctx-size` | `262144` | Context length in tokens |
| `-ctk`, `-ctv` | `q8_0` | KV cache type. q8_0 is about half the size of f16 |
| `-fa` | `on` | Flash attention |
| `-np`, `--parallel` | `1` | Number of slots. The fused and prearm features need `1` |
| `-b`, `--batch-size` | `32768` | Logical batch. `--lm-prefill` takes `16384` if `-b` is not given |
| `-ub`, `--ubatch-size` | not set | Physical batch. `--hybrid-moe` takes `2048` if not given |
| `-fitt`, `--fit-target` | `16000` | Free VRAM to keep per device, in MiB. The owner's keep-free margin. Single value or comma list |
| `-ctxcp`, `--ctx-checkpoints` | `32` | Maximum prefix checkpoints per slot. Default 32 |
| `-cms`, `--checkpoint-min-step` | stock | Minimum spacing between checkpoints, in tokens. `0` means no minimum |
| `-cram`, `--cache-ram` | stock `8192` | Maximum prompt-cache size in MiB. `-1` no limit, `0` disable |
| `--host`, `--port` | `127.0.0.1`, `8034` | Listen address. Keep `127.0.0.1` unless needed |

## 3. Hybrid MoE decode (`--hybrid-moe`)

| Flag | Default | Effect |
|---|---|---|
| `--hybrid-moe` | off | Expert banks stay on the host. A VRAM cache serves most experts. Sets the engine environment variables you did not set, turns `--fit` off, takes `-ub 2048` unless `-ub` is given, and for `-np > 1` takes the unified KV cache unless `-kvu` or `-no-kvu` is given |
| `--hybrid-moe-slots N` | `-1` | Expert-cache slots per layer. `-1` sizes the cache to the `-fitt` margin. With `--spec-type draft-mtp` (or a draft model) the server first adds the speculative context's estimated memory to that margin, because the first sizing runs before the context exists, then takes it out again and sizes the cache once more when the context is allocated (log: `the speculative context is allocated, its N MiB leave the margin`). Before 2026-10-22 it stayed in the margin, which counted it twice: the MoE cache held 28 instead of 41 slots per layer on an 8 GiB card (5,813 against 6,759 MiB in use at idle). Decode with MTP rose by 10.5% on average (26.5 to 29.2 tok/s; Qwen3.6-35B-A3B Q4_K_M, `-c 262000`, q4_0 KV, 3 chat prompts x 2 rounds of 192 greedy tokens, order old-fixed-fixed-old, repeat-to-repeat spread 0.1 tok/s); all six cells were faster, by 4 to 15%. One machine, short generations, context up to about 4.9k tokens. The G42 start log showed 54 of 256 experts |
| `--hybrid-moe-frozen` | off | Deterministic decode. Fixed cache of `--hybrid-moe-slots` slots (16 if unset), no promotions. A cache-hit continuation equals a fresh run. Decode is slower |
| `--hybrid-moe-ubatch N` | `0` | Physical batch for the whole run when `-ub` is not given. `0` keeps the stock 512. Prefer `--hybrid-moe-prefill-ubatch` |
| `--hybrid-moe-prefill-ubatch N` | `2048` | Physical batch while a large prompt batch runs under `--hybrid-moe`. `0` turns it off. G42 sets `0` because `--lm-prefill` takes the big batches |
| `--hybrid-moe-prefill-min N` | `1024` | Smallest batch, in tokens, that opens the hybrid prefill window |
| `--hybrid-moe-seed N` | `16` | Seed the expert popularity counts with the routing of the last `N` prompt tokens. `0` off |
| `--hybrid-moe-freq FILE` | uniform prior | Expert frequency table (layers by experts, whitespace separated) that fills the cache at start |

## 4. Layer-major prefill (`--lm-prefill`)

Acts on prompt batches of at least `--lm-prefill-min` tokens. Works with and without
`--hybrid-moe`.

| Flag | Default | Effect |
|---|---|---|
| `--lm-prefill` | off | Turn the feature on |
| `--lm-prefill-min N` | `1024` | Smallest prompt batch that takes the pass |
| `--lm-prefill-ubatch N` | `2048` | Tokens per chunk of the pass |
| `--lm-prefill-segment N` | `16384` | Tokens per activation-slab segment. Must be a multiple of the chunk. A longer batch repeats the layer sweep per segment. G42 uses `32768` |
| `--lm-prefill-slots N` | `2` | Layer slots in the VRAM window. Two lets the next bank copy while the current layer computes |
| `--lm-prefill-batch N` | `16384` | Logical batch when `-b` is not given |
| `--lm-prefill-bf16` | off | Activations in bf16, half the slab. Not validated against f32 for the MoE model; for the dense Qwen3.8 see the KL study in `docs/plans/dense-qwen38-iq3-residency.md`. Do not use for claims elsewhere |
| `--lm-prefill-attn-int8` | off | The pass's flash attention computes Q.K^T with int8 dot products against the q8_0 K cache (head size 256, a GPU without fast FP16, such as Pascal). About 1.7x faster attention, 22% of an 18k-token pass of the dense Qwen3.8-27B. An APPROXIMATION (Q quantised to int8 per 32 values, as the decode kernel does). Use with the f32 slab (without `--lm-prefill-bf16`): the pair failed the continuation rule on 1 of 8 prompts. Same as `LLAMA_FATTN_TILE_Q8QK=1` |
| `--fattn-head-tile` | off | With a QUANTIZED K/V cache the CUDA TILE / MMA flash attention (every prefill batch, the pass included) reads K and V as f16, so every context with a flash-attention node reserves and fills an f16 copy of the layer's whole K and V: 4 KiB per context token for the dense Qwen3.8-27B (~1 GiB at 262,144 tokens; the MTP draft context has its own). This flag converts ONE KV head at a time (runs the kernel once per KV head on strided views), so the scratch is 1/n_head_kv of that (4x smaller dense, 2x the MoE). Same arithmetic per head; not applied with ALiBi, sinks, an f16/f16 cache, or the VEC decode kernel (which reads the quantized cache directly and never had the scratch). Measured at `-c 128000`, q4_0/q4_0 target and draft cache: compute buffers 768.77 -> 396.77 MiB (target) and 705.28 -> 333.28 MiB (each draft context), idle VRAM 7,480 -> 6,736 MiB; `test-backend-ops -o FLASH_ATTN_EXT` 711/711 with and without it; KL against the f16 cache inside the noise of the untiled kernel (below). Not tested: the TILE kernel (Pascal), several sequences per batch, speed. Same as `LLAMA_FATTN_HEAD_TILE=1`. See `docs/plans/dense-decode-ring-windows.md` G-TILE |
| `--lm-prefill-ckpt-chunk N` | `0` | The last `N + 4` tokens go in a separate chunk, so a checkpoint exists `N + 4` tokens before the end. `0` uses the stock rule (`4 + ubatch`). A chunk of 24 runs on the CPU in about 0.3 s. A chunk of 512 copies every bank, about 1.3 s (about 2.3 s on a GTX 1070 at PCIe Gen3 x8: 24 gave +29% / +7.8% / +3% prefill at 4k / 16k / 32k tokens, LINUX-PASCAL.md section 3.7). The cost is a shallower rollback |
| `--lm-prefill-canonical` | off | A cache hit equals a fresh run bit for bit. Checkpoints only at multiples of `--lm-prefill-ubatch`. A continuation resumes from the newest checkpoint before the common prefix. Costs up to one chunk more prefill per follow-up. Needs `-b` to be a multiple of the chunk size and context checkpoints enabled |
| `--lm-prefill-fuse` | off | With `-np 1`: the rest of a prompt goes in one batch instead of being cut at checkpoints. Checkpoints are still stored. Saves the second and third bank streams, about 1.3 s each |
| `--lm-prefill-fuse-min N` | `700` | Fewest prompt tokens left after a cache hit that `--lm-prefill-fuse` sends whole |
| `--lm-prefill-prearm MODE` | `off` | With `-np 1`, prepare the next pass while the slots are idle. `off`, `always`, or `auto` (when at least 3 of the last 4 requests used the pass). An unused preparation costs a cache regrow of about 0.4 s |

### 4.1 Dense models (Qwen3.5 / Qwen3.8, no routed experts)

The pass also takes the dense architecture. A streamed package is the WHOLE decoder layer, so
the placement is by layer: a layer is entirely on the device (no slot, no copy) or entirely
in host memory (copied once per pass into a slot). The KV cache and the recurrent state of every
layer stay on the device. The server refuses loudly when that does not hold. See
[LINUX-PASCAL.md](LINUX-PASCAL.md) section 7 and `docs/plans/dense-qwen38-iq3-residency.md`.

| Flag | Default | Effect |
|---|---|---|
| `--lm-prefill-host-layers SPEC` | off | The layers whose weights stay in host memory (page-locked once). `SPEC` is a count `N` (the first `N` layers) or layers and ranges such as `0-30,34`. Sets `-ngl` to all layers unless given and turns the fit step off. Works without `--lm-prefill` too (the ordinary path then computes or offloads them) |
| `--lm-decode-stream SPEC` | off | Layers (a subset of the host layers, given as layers and ranges) that a DECODE step computes on the device from a ring of VRAM slots, which a copy stream refills while the CPU computes the other host layers. Their number must be a multiple of `--lm-decode-slots`. `tools/dense_plan.py` proposes a plan for the card. Spread the layers evenly over the host layers (bunched layers cannot hide their copies); 4 to 8 is the useful range on the 24 GiB Windows laptop (`docs/plans/dense-decode-ring-windows.md`). `dynamic` or `dynamic:P` lets the statistics of the previous decode steps choose the ring layers (any host layer can be one), instead of a fixed spread: the copy time of a layer (percentile P, default 95, measured by the copy worker), the CPU time between two ring layers and the GPU time of a ring layer (CUDA timing events) tell how many CPU layers the copy of the next ring layer must hide behind. The plan changes only between decode steps, with hysteresis; a changed plan rebuilds the decode graph once. Needs `--lm-prefill-host-layers` or `--lm-resident-layers`. Tuning by environment: `LLAMA_DS_MARGIN` (0.25), `LLAMA_DS_OVERHEAD_MS` (2), `LLAMA_DS_WINDOW_TOKENS` (16), `LLAMA_DS_REPLAN` (2). `lookahead:L` / `spaced:L` pick the ring layers by formula from `--lm-prefill-host-layers` (the first slots' worth of host layers, then after L CPU-computed host layers the next ring layer; `spaced` counts from the previous ring layer, `lookahead` from each ring layer that ends). Measured on that laptop: `spaced:5` / `spaced:6` (6 ring layers of 32) match the best spread plan (+7..11%), `spaced:3` is flat, `spaced:2` and `lookahead:2/3` lose 29..45% |
| `--lm-resident-layers N` | off | Dense models: how many decoder layers stay in VRAM. They are spread evenly over the model and end-aligned (layer ((i+1)*n_layer)/N - 1: 8 of 64 = 7,15,23,...,63; 2 of 64 = 31,63; 16 of 64 = the attention layers), so the last layer, which feeds the output head, is always one of them; all the others are host layers, as if `--lm-prefill-host-layers` listed them (an explicit `--lm-prefill-host-layers` wins). The layer count is read from the GGUF header (block_count minus the NextN layers). KV cache and recurrent state of every layer stay on the device. Unless `--lm-decode-stream` is given, it also starts the dynamic decode ring over the host layers (`--lm-decode-stream off` keeps them all on the CPU) |
| `--lm-evict-layers N` | off | With `--lm-resident-layers`: N of the resident layers (spread evenly over them, never the last layer) are loaded in HOST memory and get a VRAM slot of their own for decode steps: they are computed on the device like resident layers. The layer-major pass (`--lm-prefill`) frees those slots, which gives it the room it needs on a full card, and streams the layers from host memory like any host layer; the first decode step fills the slots again (~40 ms per layer). Costs N x 345 MiB of RAM. Choose N from the pass's own refusal message (`... MiB of device memory are free, the pass needs ... MiB`): each evictable layer adds ~345 MiB. Measured on the 24 GiB laptop: 24 resident + 5 evictable, MTP + `--spec-mtp-defer`: 974 tok/s prefill against ~370 on the ordinary path |
| `--lm-decode-slots N` | `2` | Slots of that ring. More than two did not help in any measurement |
| env `LLAMA_DS_PIECE_KIB` / `LLAMA_DS_WINDOW` / `LLAMA_DS_POLL_US` | `1024` / `2` / `20` on Windows, `0` (one piece) elsewhere | Ring copy in pieces of that size with at most WINDOW in flight, issued by a worker thread: on Windows an unpaced layer copy blocks the CPU->GPU activation hops of the CPU layers for its whole duration (22 ms) and made the ring slower than no ring. `LLAMA_DS_PIECE_KIB=0` = old behaviour. `LLAMA_DS_FLUSH=0` turns off the submit flush after each queued copy; `LLAMA_DS_NOCOPY=1` never fills the slots (timing only, wrong output). `LLAMA_DS_HOP_KERNEL` (default on with the ring, `0` off) and `LLAMA_CUDA_HOP_KERNEL=1` (any run): small CPU->GPU buffer copies as a kernel on pinned memory instead of a copy-engine copy |

The ring and the prefill pass share the VRAM in time: the ring frees its slots while a pass runs
and takes them back (reloading its first layers, about 50 ms) when the pass ends.

## 5. Environment variables for the engine

These are read by the engine. `--hybrid-moe` sets the ones it needs. Set them by
hand only when a plan or a run note asks for it. The names below come from a grep of
the fork sources. The groups follow the name prefixes only. Read the line where a
variable is used before relying on what it does.

Names starting with `LLAMA_LM_`, `LLAMA_TRACE`, `LLAMA_DECODE_PROFILE`,
`LLAMA_SRV_`, `LLAMA_SERVER_`, `LLAMA_HYBRID_`, `LLAMA_LS_`: `LLAMA_LM_LOG`,
`LLAMA_LM_HASH`, `LLAMA_LM_FA_EXACT`, `LLAMA_LM_FAIL_AT`, `LLAMA_LM_FAIL_RUN`,
`LLAMA_TRACE`, `LLAMA_DECODE_PROFILE`, `LLAMA_SRV_PHASE_LOG`,
`LLAMA_SERVER_SLOTS_DEBUG`, `LLAMA_HYBRID_FREE_WATCH_MS`, `LLAMA_LS_HOST_SHADOW`,
`LLAMA_LS_TEST_NO_ALLOC_RESET`. The `FAIL` names look like fault-injection switches
for tests. Do not set them outside tests.

Names starting with `LLAMA_MOE_HYBRID_`: `LLAMA_MOE_HYBRID_LOG`,
`LLAMA_MOE_HYBRID_FREQ`, `LLAMA_MOE_HYBRID_TRACE`, `LLAMA_MOE_HYBRID_SPLIT`,
`LLAMA_MOE_HYBRID_XDUMP`, `LLAMA_MOE_HYBRID_LEAN_CPUS`,
`LLAMA_MOE_HYBRID_DEBUG_STALL`, `LLAMA_MOE_HYBRID_COPY_MODE`. `LLAMA_CUDA_INPUT_BATCH`
is listed here for the same reason.

CUDA backend, names starting with `GGML_CUDA_`: `GGML_CUDA_DISABLE_FUSION`,
`GGML_CUDA_NO_PINNED`, `GGML_CUDA_REGISTER_HOST`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`.

Linux and Pascal (see [LINUX-PASCAL.md](LINUX-PASCAL.md)): `LLAMA_FATTN_TILE_Q8QK=1`
turns on int8 QK attention (a q8_0 K cache, head size 256, a GPU without fast FP16;
off by default). `LLAMA_FATTN_TILE_Q8CFG` picks the block configuration of that
kernel, 0 to 6, default 0. `LLAMA_MOE_HYBRID_LEAN_CPUS` lists the logical CPUs of
the expert team, the last one for the engine thread. The CPUs are pinned on Windows only; on
Linux only the number of entries counts, as the team size. The team defaults to the physical
cores the process may use, at most 8, on Linux (a flat 8 elsewhere); `LLAMA_MOE_HYBRID_THREADS`
overrides it. A team larger than the physical cores made decode 4 to 5 times slower on an
i7-4770K (LINUX-PASCAL.md section 3.2). `LLAMA_MOE_HYBRID_LEAN_BATCH` (default 1) serves the
requests that the rows of an MTP verify step post for one layer as one job of the team, so each
missed expert is read once; the results are bit-identical, `0` serves every request on its own
(LINUX-PASCAL.md section 3.7).

Build and tests: `Q38LS_MODEL_PATH` is read by the CTest entries. The model-backed
tests fail without it, and the CMake cache keeps the value from configure time.

## 6. Profiles

| Profile | Flags | Use for | Trade-off |
|---|---|---|---|
| Canonical (G42) | the command in [USAGE.md](USAGE.md#4-run-the-best-configuration) | Interactive and agent work on one slot | Small turns see nothing. Follow-ups prefill up to one chunk more |
| Deterministic | canonical plus `--hybrid-moe-frozen` | Reproducible answers; hit-versus-fresh checks | Decode is slower |
| Stock baseline | unmodified llama-server with `-c`, `-ctk`, `-ctv`, `-fa`, `-np`, `-fitt 16000` | Comparison numbers | Slower; no expert offload |
| Two slots | `-np 2` with the layer-major prefill | Concurrent clients | Big batches decline under concurrency; needs the unified KV cache |

## 7. Multi-token prediction (`--mtp`)

In `arg.cpp`, `--mtp` belongs to the model download tool. It loads the MTP head. The
MTP self-speculation results in the record (G30) come from the research runner, not
from the canonical server command. G30 accepted MTP as opt-in, default off, with a
recipe per band. Draft depth above 1 lost.

### 7.1 The server's own MTP (`--spec-type draft-mtp`) with the layer-major prefill

`--spec-type draft-mtp` (upstream) asks the target for its hidden state at EVERY prompt
position, and keeps recurrent-state snapshots (`n_rs_seq`). The layer-major pass declines both,
so with plain `draft-mtp` the prompt takes the ordinary path (`--lm-prefill-fuse` and
`--lm-prefill-prearm` are refused at start-up).

| Flag | Default | Effect |
|---|---|---|
| `--spec-mtp-defer` | off | MTP: no hidden states while the prompt is processed, so the prompt batches output only their last row and `--lm-prefill` takes them. Drafting starts at the second generated token (the first one has no hidden state of the position before it). The draft cache holds no prompt positions, so drafts are worse. Needs `-np 1` and `--spec-type draft-mtp`; one MTP head with its own cache only. The draft context then gets a ubatch of `max(16, n-max + 1)`: it never decodes more rows, and its compute buffer shrinks (564 -> 266 MiB for Qwen3.6-35B-A3B), which `--hybrid-moe` gives to the expert cache |

Inside the pass a context with snapshots runs as if `n_rs_seq` were 0 (plain delta-net graph,
plane 0 of the state, no keep-tail split) and is restored afterwards; the pass declines when a
rollback is pending (`rs_idx != 0`) or when the context extracts layer inputs (EAGLE3, DFlash).

Measured (Qwen3.8-27B-UD-Q6_K_XL, even layers on the CPU through `-ot`, `-t 16`, q8_0 KV, 128 greedy tokens, RTX 5090 Laptop,
`docs/currentstate.md` 2026-10-20): prefill / decode in tok/s, drafts accepted.

| Prompt | no speculation | `draft-mtp` | `draft-mtp --spec-mtp-defer` |
|---|---|---|---|
| 8k | 1082 / 3.53 | 409 / 8.49, 67% | 1085 / 6.44, 43% |
| 24k | 1163 / 3.47 | 409 / 6.92, 52% | 1166 / 5.79, 40% |
| 50k | 1042 / 3.48 | 389 / 8.10, 65% | 1038 / 5.82, 39% |
| 250k (the pass found no free VRAM) | 276 / 3.19 | 79 / 1.54, 46% | 278 / 4.72, 31% |

Deferred MTP finishes a request with a 128-token answer first at every length measured (at 250k only
by 2% of the wall time over no speculation: 925 s against 944 s, and 3,260 s for full MTP). With longer
answers full MTP wins up to 50k: break-even about 320 / 1,350 / 1,650 generated tokens at 8k / 24k / 50k
(arithmetic from the table, not measured). At 250k full MTP decodes slower than no speculation. At equal chunk size (`--lm-prefill-ubatch 512`) the
first-token log-probabilities of the three paths are identical to the last digit at 8k, 24k and 50k,
cold and after a checkpoint resume.

### 7.2 KV cache types: K and V must have the same type (default build)

The CUDA flash attention of the default build (`GGML_CUDA_FA_ALL_QUANTS=OFF`) has kernels only for a K and a V cache of the SAME type: f16/f16, q4_0/q4_0, q8_0/q8_0, bf16/bf16 (`fattn.cu`, `ggml_cuda_get_best_fattn_kernel`). Any other pair, for example `-ctk q8_0 -ctv q4_0`, or `-ctvd q4_0` next to the default f16 `-ctkd`, has no GPU kernel: the scheduler runs the attention on the CPU and copies the cache to it. Prompt and decode then crawl while the GPU idles (measured: 0% GPU, 7 W, 9 CPU threads busy for minutes, against 95% GPU with `q8_0/q8_0`). Use matching types, or build with `GGML_CUDA_FA_ALL_QUANTS=1 BUILD_DIR=build/llama-cuda-faq tools/build/build-rtx5090.sh` (a much longer compile). The target (`-ctk/-ctv`) and the MTP draft (`-ctkd/-ctvd`) caches are checked separately.

**Quality of the K/V cache types** (llama-perplexity `--kl-divergence`, dense Qwen3.8-27B-UD-Q6_K_XL, f16/f16 reference, 8,190 scored tokens of the 20.7k-token web page text, 24 layers on the host, RTX 5090 Laptop; the standard error of the KLD is 0.0007-0.0008; single-token maxima of 10-12 nats appear in every quantized arm):

| `-ctk` / `-ctv` | KV size at 128k | PPL ratio | mean KLD | same top-1 | 99% KLD | RMS dp |
|---|---|---|---|---|---|---|
| q8_0 / q8_0 | 4,250 MiB | 1.0007 | 0.0016 | 99.29% | 0.0058 | 1.10% |
| q8_0 / q4_0 | 3,250 MiB | 1.0013 | 0.0034 | 98.61% | 0.0200 | 1.72% |
| q4_0 / q4_0 | 2,250 MiB | 1.0040 | 0.0052 | 97.86% | 0.0461 | 2.33% |

Two f16/f16 runs against each other give KLD 0.0000 (the path is deterministic). With `--fattn-head-tile` the same arms give KLD 0.0018 (q8/q8) and 0.0057 (q4/q4) against the f16 reference, 98.03% same top-1 against the untiled q4/q4 logits; a different batch size alone (`-ub 256` against 512, untiled) moves the logits by KLD 0.0035, so this is the floor of any change in arithmetic. Mixed K/V types need `GGML_CUDA_FA_ALL_QUANTS=1` (above).
