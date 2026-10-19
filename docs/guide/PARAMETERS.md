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
| `--hybrid-moe-slots N` | `-1` | Expert-cache slots per layer. `-1` sizes the cache to the `-fitt` margin. The G42 start log showed 54 of 256 experts |
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
| `--lm-prefill-bf16` | off | Activations in bf16, half the slab. Not validated against f32. Do not use for claims |
| `--lm-prefill-ckpt-chunk N` | `0` | The last `N + 4` tokens go in a separate chunk, so a checkpoint exists `N + 4` tokens before the end. `0` uses the stock rule (`4 + ubatch`). A chunk of 24 runs on the CPU in about 0.3 s. A chunk of 512 copies every bank, about 1.3 s. The cost is a shallower rollback |
| `--lm-prefill-canonical` | off | A cache hit equals a fresh run bit for bit. Checkpoints only at multiples of `--lm-prefill-ubatch`. A continuation resumes from the newest checkpoint before the common prefix. Costs up to one chunk more prefill per follow-up. Needs `-b` to be a multiple of the chunk size and context checkpoints enabled |
| `--lm-prefill-fuse` | off | With `-np 1`: the rest of a prompt goes in one batch instead of being cut at checkpoints. Checkpoints are still stored. Saves the second and third bank streams, about 1.3 s each |
| `--lm-prefill-fuse-min N` | `700` | Fewest prompt tokens left after a cache hit that `--lm-prefill-fuse` sends whole |
| `--lm-prefill-prearm MODE` | `off` | With `-np 1`, prepare the next pass while the slots are idle. `off`, `always`, or `auto` (when at least 3 of the last 4 requests used the pass). An unused preparation costs a cache regrow of about 0.4 s |

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
the expert team, the last one for the engine thread; match it to the CPU's SMT
layout.

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
recipe per band. Draft depth above 1 lost. MTP with `--lm-prefill` is untested.
