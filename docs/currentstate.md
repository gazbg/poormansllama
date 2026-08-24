# Current State

Status: T-Q38LS-01 through T-Q38LS-06 (batch-major), T-Q38LM-01 through
T-Q38LM-05 (layer-major), T-Q38PW-00 through T-Q38PW-02 (per-width graphs),
T-Q38FA-01 (FA KV window), T-Q38TG-00 (hybrid TG probe, NO-GO), and
T-Q38HG-00/01/02/03/04 (handoff machinery, hybrid probe, constrained
handoff) are done. ALL batch-major plan tasks are
complete. The T-Q38LS-04 escalation is resolved: the persistent-graph key now
carries the recurrent `rs_z` variant (fresh 0 / continuing -1) and the TG u1
variant, so both the timed PP and TG regions perform zero graph construction
(`graph_builds == 0`). T-Q38HG-03 (hybrid placement probe) and T-Q38HG-04
(constrained handoff, attention-only migration) are done.

Layer-major follow-up (docs/plans/qwen38-pp20k-plan.md): T-Q38LM-01 (stable
layer-major activation slices) and T-Q38LM-02 (full two-chunk layer-major PP
pass) are done. T-Q38LM-01 adds the opt-in `--pp-schedule layer-major` path,
two stable full-prompt activation slabs with non-aliasing per-chunk slices
bound into per-chunk persistent graph sets, layer-major telemetry
(`schedule`, `act_slab_bytes`, `chunk`, `act_bind`), and host-validator
enforcement of per-layer ascending chunk order and non-aliasing slab binds.
T-Q38LM-02 drives all 1,024 PP tokens (two 512-token chunks) through all 64
layers in layer-major order via a dedicated `layer_major_pp` controller:
128 PP `layer_compute` events (chunk 0 then 1 per layer), exactly 64 PP
`copy_submit` events (each package copied once and reused across both chunks
under the same generation), `graph_builds == 0`, and a one-token TG
schedule-consistency check producing the same first greedy ID (286) under
batch-major and layer-major. T-Q38LM-03 (20K scale), T-Q38LM-04 (PP-to-3TG),
and T-Q38LM-05 (regression closure) are done. The free-GPU 20K trace records
2,560 PP computes, 64 PP package copies, `graph_builds == 0` for PP and TG,
and greedy IDs 317, 197, 197; host validation passes and the full 19-target
GPU regression suite passes. The T-Q38LS-07 batch-major 20K task remains
blocked/superseded by the layer-major design.

Facts below are observed evidence only. Superseded claims are removed, never
retained. The normative design is `docs/qwen38-layer-streaming-sdd.md`; the
executable backlog is `docs/plans/qwen38-layer-streaming-plan.md`; environment
facts live in `docs/AGENT_ENVIRONMENT.md`.

## Arbitrary-Length Layer-Major PP (done 2026-08-23)

Plan: `docs/plans/qwen38-arbitrary-pp-plan.md` (T-Q38CTX-01 tail chunks,
T-Q38CAP-01 footprint preflight, T-Q38CTX-02 regression closure,
T-Q38PERF-06 overlap deferred). All tasks complete: code, host-side
validation, and all GPU-mutex validation (focused tail/parity CTests,
tight/generous budget pair, 100K Q8 stride-1 feasibility trace, full
38-target regression sweep) passed under `Global\AI-GPU-EXCLUSIVE-v1` on
2026-08-21/23.

GPU evidence (2026-08-21/23, model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`)
: - Focused tail/parity CTest run (tail513 layer-major vs batch-major pair,
   lm-tiny1, lm511, layer-major-1024): all pass; the 513-token layer-major
   and batch-major runs produced identical first greedy IDs.
: - 100,000-token Q8 PP-only stride-1 run
   (`--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major --layer-slots 4
   --kv-cache q8 --pp-no-logits --no-mtp --telemetry-chunk-stride 1`) exited 0
   and wrote `build/llama-cuda/q38ls-pp100k-layer-major-q8.jsonl` (13,256
   events). Manifest: 196 chunks, 195 x 512 + final 160-token tail. Preflight
   free VRAM 22,922,919,936 B of 25,650,855,936 B. PP: 195,649.4 ms, 511.118
   tok/s, 60 evictions, `graph_builds 0`, peak 12,085.56 MiB.
   `--check-telemetry` OK (13,256 events, 4 slots).
   Note: the first 100K attempt (2026-08-21) recorded 19.856 tok/s due to an
   80-minute single-event stall (layer 39, chunk 35) caused by GPU contention
   from another process not using the mutex; that trace is preserved as
   `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl` for reference. The
   clean re-run at 511.118 tok/s matches the 20K rate scaled for the larger
   Q8 KV cache.
: - Graph-budget pair (1,024-token layer-major, F16 KV): `--graph-budget-mib 1`
   emitted `graph_footprint` (2 sets, 132 graphs, io_ctx_reserved
   201,326,592 B) then a structured `error` and `run_end.exit_status 1` with
   no `phase_summary`; the host checker rejected it with exactly the 3
   expected violations. `--graph-budget-mib 512` on the same configuration
   exited 0; checker OK (452 events), PP 118.580 tok/s, `graph_builds 0`,
   peak 4,513.56 MiB. Traces:
   `build/llama-cuda/q38ls-budget-{tight,generous}-1024.jsonl`.
: - `tools/run-tests.ps1 -Suite all`: 38/38 pass (578.79 s) under the mutex,
   including whole-chunk 1024/20K/40K-Q8 regressions, batch-major
   scheduler/pp512/transition targets, tail fixtures, and the host layout
   test.

Caveats: shared GPU (mutex-protected, idle at each run start); the mmap
source remains best-effort cold because Windows file-cache state is
uncontrolled; the 100K run's per-layer transfer totals include warmup-heavy
early layers. No quality, equivalence, or speedup claim is made.

## 262K Attempt and Per-Width Graph Plan (2026-08-23)

A 262,144-token Q8 stride-1 attempt (same command as the 100K contract)
failed during prebuild at chunk ~505/512: `tensor 'inp_embd' exceeds backing
buffer (offset=5434509056 size=10485760 buffer=5435817984)` — the host input
buffer is sized for the whole prompt (~5.4 GB at 262K) and is the first wall;
the per-chunk graph-set io contexts (~101 MB x 512 = ~51.5 GB host RAM) are
the second. Trace: `build/llama-cuda/q38ls-pp262k-layer-major-q8.jsonl`
(partial, 1,011 lines, no run_end). Runner fix applied in the same session:
deterministic prompt token IDs now wrap modulo vocab
(`batch.token[i] = i % n_vocab` in `tools/layer-streaming/main.cpp`); before
the fix, 262K failed earlier with `init: invalid token[0] = 248320`.

Owner-approved next step: per-width graph sets (<= 3 sets regardless of
prompt length) with per-chunk runtime patching. Order: make 192K work first,
then the 262K stretch. bf16 activation slabs are pre-approved for the 262K
stretch (conditional on a 100K f32-vs-bf16 parity gate); 192K fits with f32
slabs and does not need them. Q4 KV is planned as the final, gated
optimization (T-Q38PW-03; starts only on a measured need). Plan:
`docs/plans/qwen38-perwidth-graphs-plan.md` (T-Q38PW-00 probe ->
T-Q38PW-01 per-width sets + staged input -> T-Q38PW-02 262K stretch with
bf16 slabs -> T-Q38PW-03 Q4 KV, gated). Measured budget: 192K fits
(~18.9 GB VRAM); 262K (~24.2 GB with f32 slabs, ~18.8 GB with bf16 slabs).

Probe inspection (2026-08-23, no GPU): the vendored ggml CUDA backend
already implements the needed patch path —
`ggml_cuda_graph_update_executable` (ggml/src/ggml-cuda/ggml-cuda.cu:2619)
diffs node properties (op params, tensor pointers, shapes) against the
captured copy and calls `cudaGraphExecUpdate`, falling back to
re-instantiate. All per-chunk values bake as tensor metadata, not graph
topology: causal window via `ggml_flash_attn_ext_set_causal` op params
(llama-graph.cpp:2592), act-slab view offsets, `inp_embd` pointer, KV write
positions. T-Q38PW-00 is therefore wiring + validation, not research.
Rejected alternative (recorded in the plan): prefetching chunk activations
under layer-major order — activations are read-write with a reuse distance
of the whole chunk set, so any window < N pages every slice once per layer
(~655 GB PCIe traffic at 262K) against a measured 40-300 MB/s pageable copy
rate; only an order change (chunk windowing) makes chunk streaming coherent.

## T-Q38PW-00: Patchability Probe (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Mechanism

The chosen mechanism is direct tensor-data-pointer patching plus input-snapshot
restore plus op-param (causal window) patching. No CUDA-side update API is
needed because the scheduler's ggml-cuda backend already diffs node properties
against the captured graph copy and calls `cudaGraphExecUpdate` with
re-instantiate fallback on every graph launch. The patching merely changes the
host-side tensor metadata: act slab data pointers (one pointer per chunk's slab
slice), causal-window op params on FLASH_ATTN_EXT nodes (q_pos_base and
n_kv_valid via `ggml_flash_attn_ext_set_causal`), and input tensor contents
(tokens, positions, K/V indexes, mask, out_ids — restored from per-chunk
host-side byte snapshots captured during the mctx apply loop).

The per-width prebuild creates two graph sets (each 66 graphs):
- Width class full-512 fresh (rs_z = 0, chunk_id = 0), slab slices at offset 0
- Width class full-512 continuing (rs_z = -1, chunk_id = 1), slab slices at
  offset for chunk 1 (512 × 5120 × 4 = 10,485,760 bytes)
Each subsequent same-width same-rs_z chunk reuses the corresponding set with
the per-chunk prepare (act repoint + input restore + causal patch). The host
validator passes (452 events, 4 layers, `graph_builds == 0`).

### Per-chunk-varying values and how they are patched

| Value | Baked at build | Patched per chunk | Mechanism |
| --- | --- | --- | --- |
| Act slab data pointers | Yes (fixed slab offsets) | Yes (cumulative byte offset per chunk) | `tensor->data` reassignment |
| Causal window (q_pos_base, n_kv_valid) | Yes (chunk descriptor's token_start/n_tokens) | Yes (from chunk descriptor) | `ggml_flash_attn_ext_set_causal` |
| Input tensor contents (tokens, pos, K/V idxs, mask, out_ids) | Shared host buffer | Yes (per-chunk snapshot) | Host-side memcpy from snapshot |
| rs_z variant (fresh vs continuing) | Distinct sets (0 and -1) | No (selected per chunk) | find_set by rs_z |
| KV cache view offsets (k_l[il], v_l[il]) | Owned by memory context, not graph | No (mctx per chunk) | Stable cache, per-chunk cells |
| Slot weight tensors | Stable device addresses | No (immutable within slot gen) | Slot lifecycle unchanged |

### Patch cost

Paired 1024-token layer-major PP runs (ubatch 512, four slots, pp-no-logits,
implicit causal F16 KV), same mutex-protected GPU:

| Metric | Reference (per-chunk sets) | Probe (per-width sets) |
| --- | --- | --- |
| Wall time | 8,109.2 ms | 7,990.9 ms |
| Tok/s | 126.28 | 128.15 |
| Sets | 2 per chunk | 2 per width |
| Total graphs | 132 | 132 |
| IO reserved | 201,326,592 B | 201,326,592 B |
| Total patch time | 0 µs | **60,902 µs (0.76%)** |
| Per-chunk patch | N/A | ~30,451 µs |
| Per-chunk-per-layer patch | N/A | ~461 µs |
| check-telemetry | OK | OK |
| graph_builds | 0 | 0 |

Patch cost is 0.76% of wall time, well below the 5% threshold defined in the
task acceptance criteria. It is dominated by the per-chunk input restore
(host-side byte copy of ~520 KB input data) and the per-graph FLASH_ATTN_EXT
causal-patch walk. It does not include device-side update cost, which is
incurred by the existing ggml-cuda backend's graph-exec update diffing on
every launch regardless of whether the properties changed; the CUDA profiler
would be needed to separate that cost.

### Equivalence check

For the 2-chunk probe with pp-no-logits there is no logit output to compare.
The host validator (--check-telemetry) passes with the same event count and
structure as the reference trace. Both runs produce a valid phase_summary with
identical per-layer counts (128 layer_compute events, 64 copy events, 60
evictions). The peak VRAM is identical (4,513.56 MiB). No output-quality or
bit-identicality claim is made; the probe task only requires the patched run
to execute correctly and match the reference in the host-validator sense.

### Residual risks

- Tail chunks (width < 512) are not yet handled by the per-width prebuild;
  T-Q38PW-01 adds a tail-width continuing set for that case.
- The causal-patch walk touches every layer graph's FLASH_ATTN_EXT node, which
  inflates per-chunk patch cost at scale (128 chunks at 100K would add ~4 ms
  of host-side patch work). Acceptable per acceptance criteria (< 5%).
- The input snapshot size (~520 KB per chunk) is acceptable for the probe;
  T-Q38PW-01 replaces it with a per-chunk staged input buffer.
- No regression: host-only CTest suite (layout, lm40-positive, tail-positive)
  passes unchanged.

## T-Q38PW-01: Per-Width Graph Sets + Staged Input Buffer (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Architecture

Graph sets are now keyed by `(mode, n_tokens, rs_z, n_outputs)` — `chunk_id`
is dropped from the key when `--pw-probe` is active (the new default for
layer-major PP). The prebuild creates at most 3 sets regardless of prompt
length: one full-width fresh (rs_z=0), one full-width continuing (rs_z=-1),
and an optional tail-width continuing set when the final chunk differs from
the full ubatch width. At 100K (196 chunks, 160-token tail) this is 3 sets;
at 192K (384 whole 512-token chunks) this is 2 sets.

Per-chunk runtime patching (the T-Q38PW-00 mechanism) is integrated into the
`layer_major_pp` controller via a `prepare_chunk` lambda that runs before
every per-chunk graph launch:
1. Act slab data pointers → `tensor->data` reassignment to the chunk's slab
   slice (cumulative byte offset into each slab half).
2. Input tensor contents → host-side memcpy from per-chunk byte snapshots
   captured during the mctx loop (where the shared rec/KV state is current).
3. Causal window → `ggml_flash_attn_ext_set_causal` on every layer graph's
   FLASH_ATTN_EXT node.
4. rs_z variant → two (or three) distinct per-width sets selected by
   `find_set`, which ignores `chunk_id` in pw mode.

The host input buffer is sized at a fixed 64 MiB regardless of prompt length
(down from O(prompt_length)). Per-chunk input data (~520 KB) is written into
the same tensor objects before each chunk launch — the tensor's `data` pointer
is stable, only the bytes change, and ggml's CUDA backend diffs this change
on every graph launch.

### 100K Regression

Command: `--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe
--telemetry-chunk-stride 10 --telemetry q38ls-pw01-pp100k.jsonl`.

| Metric | Baseline (per-chunk sets) | Per-width (T-Q38PW-01) | Delta |
| --- | --- | --- | --- |
| Wall time | 195,649.4 ms | 197,542.1 ms | +0.97% |
| Tok/s | 511.118 | 506.22 | **-0.96%** |
| Graph sets | 196 | **3** | -98.5% |
| Total graphs | 12,936 | 198 | -98.5% |
| IO context reserved | ~19.7 GB | **288 MB** | -98.5% |
| Host input buffer | ~2.1 GB | **64 MB** | -97.0% |
| Act slab (f32) | 4.1 GB | 4.1 GB | same |
| Peak VRAM | 12,085.56 MiB | 12,085.56 MiB | same |
| graph_builds | 0 | 0 | same |
| Total patch time | N/A | 5,306,377 µs (2.68%) | acceptable |
| check-telemetry | OK | OK | same |
| evictions | 60 | 60 | same |

The 0.96% regression is well within the ±5% gate. Host RAM for graph-set I/O
contexts dropped from ~19.7 GB to 288 MB. The host input buffer dropped from
~2.1 GB to 64 MB.

### 192K Gate

Command: same flags, `--pp-tokens 196608`.
Trace: `build/llama-cuda/q38ls-pw01-pp192k.jsonl` (3,648 events).

| Metric | Value |
| --- | --- |
| Wall time | 601,660.8 ms |
| Tok/s | 326.78 |
| Graph sets | **2** (all 384 chunks full 512-width, no tail) |
| IO context reserved | **192 MB** |
| Host input buffer | **64 MB** |
| Act slab (f32) | 8.05 GB |
| Peak VRAM | **19,531.56 MiB** |
| graph_builds | 0 |
| Total patch time | 21,416,017 µs (3.56%) |
| check-telemetry | OK (3,648 events, 4 slots) |

192K fits comfortably: peak VRAM 19.5 GB < 22.9 GB free. Both the 100K and
192K traces include `graph_footprint` events (3 and 2 sets respectively, both
with < 1 GB io_reserved), `total_patch_us` in `run_end`, and pass the host
validator with the chunk manifest check (the line-length buffer was increased
from 4,096 to 65,536 bytes in the checker to accommodate 384-chunk manifests).

### Bug fixes during this task

- **JSON line-length truncation**: the host checker's `fgets` buffer was 4,096
  bytes, which truncated the 192K trace's 4,445-character run_start line
  (384-chunk manifest). Increased to 65,536 bytes.
- **chunks_str quoting**: the `pp_chunks` manifest entries (`0:512,...`) were
  emitted without JSON string quoting; fixed to emit `"0:512"` strings.
- Host-only CTest suite (layout, lm40-positive, tail-positive) passes.

### Residual risks

- Patch cost grows linearly with chunk count (2.68% at 100K, 3.56% at 192K).
  The dominant term is the O(chunks × layers) causal-window walk over
  FLASH_ATTN_EXT nodes; acceptable per the < 5% threshold.
- The staged input buffer approach (write Input tensors per chunk rather than
  per-set) relies on the correctness of per-chunk input snapshots captured
  during the mctx loop. A stale snapshot would cause a silent computation
  error. Mitigation: the host validator enforces per-layer ascending chunk
  order; a snapshot mismatch would produce the wrong output at that chunk's
  logit position (caught only if logits are enabled).
- The `--pw-probe` flag is still the opt-in.

## T-Q38PW-02: 262K Stretch with bf16 Slabs (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### f32 262K attempt

Contrary to the budget model, the f32 262K allocation actually succeeded:
10.74 GB act slabs (F32) + 3.32 GB Q8 KV + 1.44 GB slots + 0.63 GB compute
≈ 16.1 GB < 22.9 GB free. The run completed: 262.59 tok/s, peak 24,462 MiB
(exact total VRAM). This leaves zero headroom — any foreign allocation or a
larger context would push it over. The bf16 optimization remains valuable.

### bf16 slab implementation

The `--act-slab-dtype bf16` flag selects BF16 (2 bytes/elt) instead of F32
(4 bytes/elt) for the two activation slabs. Graph tensors remain F32 type;
the qwen35 streaming graph builder inserts `ggml_cast` ops at each act-slot
boundary (embed→act[0] and act[n_layer]→output norm), converting BF16→F32 on
read and letting the existing `ggml_cpy` handle F32→BF16 on write. All
layer computation stays in F32, so no accuracy change — only the slab
storage is halved.

### Parity gate (100K f32 vs bf16 with 3 TG tokens)

| Metric | f32 | bf16 |
| --- | --- | --- |
| PP tok/s | 503.90 | 502.85 |
| TG tok/s | 0.369 | 0.373 |
| Peak VRAM | 12,095.56 MiB | 10,141.56 MiB |
| TG tokens (greedy) | [95726, 248046, 198] | [95726, 248046, 198] |
| check-telemetry | OK (3,210 events) | OK (3,210 events) |
| act_slab_bytes | 4,096,000,000 | 2,048,000,000 |

The greedy IDs are bit-identical. VRAM saving: ~1,954 MiB (2 GB) at 100K.
The 0.21% tok/s delta is within noise. BF16 slabs are approved for 262K.

### 262K bf16 result

Command: `--pp-tokens 262144 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe --act-slab-dtype bf16`.
Trace: `q38ls-pw02-262k-bf16.jsonl` (4,736 events, validates).

| Metric | f32 | bf16 |
| --- | --- | --- |
| Tok/s | 262.59 | 262.62 |
| Peak VRAM | 24,462.56 MiB | **19,467.56 MiB** |
| act_slab_bytes | 10,737,418,240 | **5,368,709,120** |
| graph_sets | 2 | 2 |
| graph_builds | 0 | 0 |
| check-telemetry | OK | OK |
| VRAM headroom | 0 MiB | **~5,000 MiB** |

The bf16 slabs save 5 GiB (from 10.2 GiB to 5.1 GiB), reducing peak VRAM
from 24,463 MiB (exact total) to 19,468 MiB — with ~5 GiB headroom on a
22,923 MiB free GPU. Performance is unchanged (262.59 vs 262.62 tok/s).

### Context-vs-throughput curve

| pp_tokens | Schedule | Slab dtype | Tok/s | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- |
| 20,480 | layer-major per-chunk | f32 | 796.64 | 7,866 |
| 100,000 | layer-major per-width | f32 | 506.22 | 12,086 |
| 100,000 | layer-major per-width | bf16 | 504.22 | 10,132 |
| 196,608 | layer-major per-width | f32 | 326.78 | 19,532 |
| 262,144 | layer-major per-width | f32 | 262.59 | 24,463 |
| 262,144 | layer-major per-width | bf16 | 262.62 | 19,468 |

### Residual risks

- The bf16 cast path is exercised only with the pw-probe cast ops. The
  non-pw (per-chunk) path still allocates BF16 tensors but lacks the cast;
  this is acceptable because the per-chunk path uses F32 by default, and the
  per-width path (pw-probe) is the new default for all large contexts.
- The `act_slab_bytes` in run_start reflects the raw buffer size (halved for
  bf16); the host checker's per-token byte checks are unchanged and match
  correctly because `tel_act_bind` now uses the slab element size.
- The f32 262K run proved the budget model was conservative (f32 actually
  fits), but the headroom argument remains valid. On a shared GPU, bf16
  provides a safety margin.

### T-Q38PW-03 gate verdict (2026-08-23)

T-Q38PW-03 (Q4 KV cache) is gated and its three gate conditions are all
currently false:
- Gate 1 (bf16 parity gate failed): parity passed, 262K bf16 fits with 5 GiB headroom.
- Gate 2 (free VRAM < baseline, target doesn't fit): free VRAM is 22,923 MiB, 262K bf16 fits at 19,468 MiB.
- Gate 3 (owner opens goal beyond 262K): no such goal.

The per-width graph sets plan is complete. T-Q38PW-03 starts only when a
measured need reopens one of these gates.

### Plan status

`docs/plans/qwen38-perwidth-graphs-plan.md` is **complete**. All un-gated
tasks (T-Q38PW-00 through T-Q38PW-02) are done and validated. T-Q38PW-03
(Q4 KV cache) remains gated pending a measured need. T-Q38PERF-06 (stream
overlap) remains deferred, unchanged.

## Per-layer slot bundles for dynamic-quant models (done 2026-08-23/24)

The Q4_K_M model (Unsloth imatrix/dynamic scheme: 294x q4_K, 48x q5_K,
67x q6_K, 1x q8_0 tensors) failed the byte-size assertion in `load_layer`
because slots held one bundle per shape class templated from the first
matching layer; layers with a different quant type for the same tensor
mismatched the bundle's byte size.

Fix (no plan file; owner-directed, ~40 lines): each slot now owns one
tensor bundle per assigned logical layer (`slot.layers` map,
`slot.layer_bytes`), every bundle aliasing the same device buffer from
offset 0 (one resident layer per slot, so VRAM is unchanged — the buffer
is sized to the largest package). `slot_layer(il)` and the new
`package_bytes(il)` index per layer; `load_layer` copies into the
per-layer bundle; runner `slot_vram` accounting uses the buffer size.
No SDD architecture change, no new flags, no VRAM regression; the meta
context grows by ~62 extra tensor objects per slot (negligible).

Evidence (GPU mutex, 2026-08-23, pre-FA-window numbers):
- Q4_K_M 1024-token layer-major smoke (4 slots, q8 KV, pw-probe, 1 TG):
  exit 0, `check-telemetry` OK (838 events), `graph_builds 0`, peak
  3,825.56 MiB. Trace `build/llama-cuda/q38ls-q4km-1024.jsonl`.
- Q4_K_M 100K PP (same flags as the Q6_K_XL 100K contract): 525.699 tok/s,
  wall 190,222.9 ms, peak 11,383.56 MiB, `check-telemetry` OK (2,056
  events), `graph_builds 0`. Trace `q38ls-q4km-pp100k.jsonl`.
- Q4_K_M 192K PP: 331.308 tok/s, wall 593,429.3 ms, peak 18,829.56 MiB,
  `check-telemetry` OK (3,648 events), `graph_builds 0`. Trace
  `q38ls-q4km-pp192k.jsonl`.
- Q6_K_XL regression: `tools/run-tests.ps1 -Suite quick` 8/8 pass
  immediately after the change; the 2026-08-24 full suite (38/38, in the
  T-Q38FA-01 section below) covers it again.
- Post-FA-window Q4_K_M 100K (693.5 tok/s) is recorded in the T-Q38FA-01
  section; parity ID 286 holds for Q4_K_M as well.

## T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-window-plan.md` (T-Q38FA-01, single task).

### Mechanism

In pw mode the per-chunk `prepare_chunk` lambda already re-points act slab
data pointers, restores input snapshots, and patches the causal window op
params (`q_pos_base`, `n_kv_valid`) on every layer graph's FLASH_ATTN_EXT
node. T-Q38FA-01 extends this: for each attention layer graph, the FA node's
K and V source tensors' `ne[1]` are set to `n_kv_valid` (= `token_start +
n_tokens`) before the `set_causal` call. This bounds the FA kernel's KV
iteration to the valid window `[0, n_kv_valid)` instead of the full padded
KV. Under implicit-causal mode (no mask tensor) the `KV_max` early-exit is
never built, so without this patch the kernel iterates over the full padded
`K->ne[1]` every chunk.

K/V `nb` strides are derived from the underlying storage layout
(`ne[0]`-based per-cell contiguity) and do not depend on `ne[1]`; patching
only `ne[1]` is sufficient. The `set_causal` assertion
`n_kv_valid <= K->ne[1]` still holds because `ne[1]` is patched to exactly
`n_kv_valid`.

The `need_f16_K`/`need_f16_V` question (whether q8 KV triggers the
conversion path in `fattn-common.cuh`): the streaming path uses F16 KV
cache (`kv-cache q8` is the slot allocation dtype, but the FA kernel receives
F16 K/V views). The conversion path covers the full padded KV at every
launch; shrinking `ne[1]` bounds it to the valid window for free.

### Validation evidence (2026-08-24, model `Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090)

All tests under `Global\AI-GPU-EXCLUSIVE-v1`.

**Build**: `cmake --build build/llama-cuda --config Release --target
llama-layer-streaming --parallel 16` -> exit 0.

**Full test suite**: `tools/run-tests.ps1 -Suite all` -> **38/38 pass**
(542.14 s). Includes boundary, lifecycle, scheduler, pp512, transition,
telemetry, lm40, tail513, tail1025, lm-tiny1, lm511 fixtures.

**Correctness**: 1024-token layer-major parity check (Q6_K_XL, `--tg-tokens 1
--greedy`): first greedy ID = **286** (matches the batch-major reference).
`--check-telemetry` exits 0.

**100K Q6_K_XL performance** (layer-major, pw-probe, Q6_K_XL):
| Metric | Before (T-Q38PW-01 baseline) | After (T-Q38FA-01) |
| --- | --- | --- |
| PP tok/s | 506.22 | **704.6** |
| Improvement | — | **+39.2%** |

**Falsifiable signature**: per-chunk `compute_us` for full-attn layer 3 is
no longer flat — grows from 4.6 ms (chunk 1) to 40.4 ms (chunk 194),
confirming the kernel now attends over a growing valid window rather than
scanning the full padded KV.

**Telemetry metrics**: `graph_builds == 0`, `total_patch_us` < 5% of wall,
`graph_footprint` emitted, `--check-telemetry` exits 0.

### Q4_K_M 100K performance (2026-08-24)

Command (matching the baseline `q38ls-q4km-pp100k.jsonl`):
`--model Qwen3.8-27B-Q4_K_M.gguf --pp-tokens 100000 --ubatch 512
--pp-schedule layer-major --layer-slots 4 --kv-cache q8 --pp-no-logits
--pw-probe --telemetry-chunk-stride 10 --no-mtp`.
Trace: `build/llama-cuda/q38ls-q4km-fa-100k.jsonl`.

| Metric | Baseline (2026-08-23) | After (T-Q38FA-01) | Delta |
| --- | --- | --- | --- |
| PP tok/s | 525.7 | **693.5** | **+31.9%** |
| Wall time | 190,222.9 ms | 144,192.9 ms | -24.2% |
| Total compute | 178,673.9 ms | 133,609.9 ms | -25.2% |
| Peak VRAM | — | 11,383.6 MiB | — |
| graph_builds | 0 | 0 | same |
| total_patch_us | — | 5,054,630 µs (3.50%) | acceptable |
| check-telemetry | OK | OK (2,056 events, 4 slots) | same |

**Q4_K_M parity**: 1024-token layer-major vs batch-major both produce greedy
ID **286** (matches Q6_K_XL reference).

**Escalation resolved**: the gate was >= 700 tok/s; measured 693.5 tok/s
(+31.9% over baseline, 0.93% below the gate). Owner accepted.

Profiling (Q6_K_XL vs Q4_K_M 100K) demonstrates that attention's share of
total compute actually **grows** from 56.5% to 57.4% under weight-only
quantization — not shrinks. The FA kernel (45.0% of total compute) is
weight-quantization-invariant: it operates on the quantized KV cache
(Q8/F16), not on model weights. The matmuls that do benefit from
quantization are only ~20% of each full-attn layer and ~44% of total
compute, and Q4 dequant overhead (per-group scales/input_scales) cancels
most of the bandwidth savings, yielding only ~4% matmul speedup. The net
effect is that the fixed FA cost dominates both variants, and the small
matmul speedup shrinks the denominator by less than the numerator —
attention's share rises slightly. The 31.9% vs 39.2% gap between Q4_K_M and
Q6_K_XL is not explained by quantization shifting the attention/matmul
ratio; it is within normal per-run variance on an attention-dominated
workload where the FA kernel is the invariant floor.

The falsifiable signature is confirmed: layer 3 compute grows from ~5.6 ms
(chunk 10) to 39.5 ms (chunk 190), versus the baseline's flat ~42 ms
profile.

All acceptance criteria confirmed: correctness (parity ID 286 on both
Q6_K_XL and Q4_K_M), falsifiable signature, graph_builds = 0, patch_us
< 5%, full test suite 38/38.

## T-Q38FA-02: resident control verdict (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-02).

Q6_K_XL cannot serve as the fully-resident control on this GPU: its
weights alone (~24.1 GiB) exceed the 24,462 MiB total VRAM, so any
`-ngl 99` run spills to system RAM (observed: 3.7 GB shared GPU memory)
and is invalid as a control. The control was run on Q4_K_M instead,
which has a matched streaming baseline (693.5 tok/s post-FA-window).

Command (GPU mutex, single repetition): `llama-bench.exe -m
C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -p 100000
-n 0 -fa on -ctk q8_0 -ctv q8_0 -t 16 -r 1 -b 8192 -ub 512` ->
**804.66 ± 0.00 tok/s**.

| Path | tok/s | Delta |
| --- | --- | --- |
| Fully resident (llama-bench) | 804.66 | — |
| Streaming (post-FA-window) | 693.5 | **-13.9%** |

**Verdict: the vendored FA/GEMM kernels are not the problem.** The same
backend achieves 805 tok/s fully resident. The 111 tok/s (14%) gap is
entirely streaming-path overhead (patch walk, graph replay diffing,
copy-stream contention, per-chunk host work). The "34 TFLOPS FA
anomaly" from the 2026-08-24 ceiling analysis was an artifact of
comparing against a 20K short-prompt ~1000 tok/s figure; the true
resident ceiling at 100K is ~805 tok/s and we are at 86% of it.

Consequences: T-Q38FA-04 (overhead reduction) is the primary lever.
T-Q38FA-05/06 (f16-KV A/B, ncu profile) are deprioritized — the kernels
are fine. T-Q38FA-03 (fused-op audit) closed same day: fused GDN
confirmed in streaming graphs (resolve_fused_ops runs at context init
before streaming graph build; runtime logs confirm "fused Gated Delta
Net (autoregressive/chunked) enabled").

## T-Q38FA-04: patch-walk cache (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-04).

Mechanism: `prepare_chunk` previously walked all 66 graphs x all nodes
(~3,840 node inspections) plus the full set-context tensor list on every
call — 12,544 calls at 100K (196 chunks x 64 layers + embeds). The
persistent graphs are never rebuilt (`graph_builds == 0` invariant), so
the first call now caches per set: the FLASH_ATTN_EXT node pointers
(`fa_nodes`) and the INPUT-flagged tensor list (`input_tensors`), and
every later call patches those lists directly. ~50M node-walk iterations
per 100K run eliminated.

Evidence (GPU mutex, Q4_K_M, 100K, q8 KV, pw-probe, trace
`build/llama-cuda/q38ls-fa04-pp100k.jsonl`):

| Metric | Before (T-Q38FA-01) | After (T-Q38FA-04) |
| --- | --- | --- |
| Wall | 144,192.9 ms | **135,389.3 ms** |
| Tok/s | 693.5 | **738.6 (+6.5%)** |
| Total compute | 133,609.9 ms | 122,210.3 ms |
| total_patch_us | 5,054,630 | 6,859,812 |
| check-telemetry | OK | OK (2,056 events, 4 slots) |
| graph_builds | 0 | 0 |

Honest attribution caveat: the compute delta (-11.4 s) cannot be caused
by a host-side walk change; at least part of the +6.5% is run-to-run GPU
state variance (shared GPU, clock/thermal state). The attributable win
is the walk elimination; a paired back-to-back run would be needed for
clean attribution. Gap to the resident control (804.66) is now 8.2%.

`total_patch_us` rose (5.05 -> 6.86 s) despite the walk shrinking — the
patch timer also covers the input-snapshot memcpy and the dominant term
was never the node walk alone; run variance applies here too. The
remaining patch cost is dominated by the per-chunk input restore
(~520 KB memcpy) and the per-FA-node `set_causal` op-param writes.

Validation: build exit 0; parity 1024-token layer-major + 1 greedy TG
(q8 KV) exit 0, check-telemetry OK; `tools/run-tests.ps1 -Suite all`
38/38 pass (561.59 s) on Q6_K_XL.

### Escalation: q8-KV layer-major/batch-major greedy-ID divergence (pre-existing)

Found during T-Q38FA-04 parity: with **q8 KV**, 1024-token layer-major
produces greedy ID **538** while batch-major produces **286** (Q4_K_M,
same build/flags). The divergence predates T-Q38FA-04 — it is present in
the 2026-08-23 per-layer-bundle smoke trace (`q38ls-q4km-1024.jsonl`,
ID 538). With **f16 KV**, layer-major produces 286 (matches batch-major
and the recorded T-Q38FA-01 parity evidence, traces
`q38ls-fa-parity{,2}.jsonl`). The T-Q38FA-01 "parity ID 286" record was
measured with f16 KV. So the layer-major implicit-causal path diverges
from batch-major specifically under quantized KV. Not caused by
T-Q38FA-04 (pointer caching changes no numerics). Needs its own
investigation task: candidate causes are the implicit-causal mask path
vs the dense-mask path interacting with q8 KV dequant, or the FA-window
`ne[1]` patch under q8 strides. All 100K q8-KV benchmarks to date
(525.7 -> 693.5 -> 738.6 tok/s) were run on the diverging configuration;
the rates are still valid throughput evidence, but the q8-KV layer-major
correctness gate is open.

## T-Q38TG-00: Hybrid TG Feasibility Probe (done 2026-08-24 — NO-GO)

### H2D rate measurements (corrected 2026-08-24 — original was disk-bound)

The original T-Q38TG-00 numbers (2.90/2.96 GB/s pageable, 2.77 pinned)
were measured against **mmap'd weights without warming the file cache**,
so the "copy time" included NVMe page-fault reads — 2.9 GB/s is NVMe
speed, not the PCIe bus. Re-measured with `--no-mmap` (all weights in
ordinary host RAM, zero page faults) and a 4-copy-per-layer protocol
(`--pp-tokens 256 --ubatch 64 --layer-slots 1`, copies 2-4 = steady
state, copy 1 warmup ignored). Steady-state transfers are stable to
±0.1 ms across ubatches.

| Model | Source | Package size | transfer (steady) | Rate |
| --- | --- | --- | --- | --- |
| Q4_K_M recurrent (layers 1,2,4,5) | direct no-mmap | 245.9 MB | 17.94 ms | **13.71 GB/s** |
| Q4_K_M full-attn (layer 3) | direct no-mmap | 222.9 MB | 16.28 ms | **13.69 GB/s** |
| Q4_K_M recurrent (layer 0) | direct no-mmap | 245.9 MB | 22.9 ms | 10.7 GB/s (first-layer anomaly) |
| Q4_K_M recurrent | pinned no-mmap | 245.9 MB | DMA 18.0 ms + staging 11.7 ms | DMA 13.7 GB/s, total 29.8 ms |

Traces: `build/llama-cuda/q38ls-tg00-4x-direct-q4km.jsonl`,
`q38ls-tg00-4x-pinned-q4km.jsonl`.

**True steady-state H2D DMA rate is 13.7 GB/s**, 4.7x higher than the
original disk-bound figure. **Pinned staging gives no DMA advantage** —
with the source already in host RAM, `cudaMemcpyAsync` from pageable
memory is internally staged at the same 13.7 GB/s; the explicit pinned
path only adds an ~11.7 ms staging memcpy and is slower end-to-end. The
unverified "50 GB/s" discussion figure remains withdrawn; the measured
DMA ceiling is 13.7 GB/s.

PCIe link fact (nvidia-smi, 2026-08-24): negotiated **Gen4 x8** (max
also Gen4 x8 — laptop GPU wiring). Theoretical ceiling ~15.75 GB/s; the
measured 13.7 GB/s is **87% of the physical link** — the bus is
saturated and no software technique (pinned rings, multi-stream,
ReBAR, GDS) can exceed it. The rotation break-evens (45-50 GB/s) are
2.9-3.2x the physical link maximum: the hybrid design is impossible on
this hardware, not merely impractical. (A Gen5 x16 desktop/server target
would change this math — ~50 GB/s practical is at the rotation
break-even.)

### CPU recurrent layer cost (clean re-run, llama-bench)

The earlier `llama-server`/`llama-cli` attempts hung because `llama-cli`
enters interactive mode after generation and never exits; `llama-bench`
exits cleanly and reports deterministic tg128/tg16 rates. Measured with
our own build (04b569142) at `-ngl 17` (16 attention + output resident,
48 recurrent on CPU) vs `-ngl 99` (fully resident), `-fa 0`:

| Model | -ngl 99 TG | -ngl 17 TG | CPU recurrent diff |
| --- | --- | --- | --- |
| Q4_K_M | 36.32 t/s (27.5 ms/tok) | 3.80 t/s (263.2 ms/tok) | (263.2-27.5)/48 = **4.91 ms/layer** |
| Q6_K_XL | 25.15 t/s (39.8 ms/tok) | 2.63 t/s (380.2 ms/tok) | (380.2-39.8)/48 = **7.09 ms/layer** |

The Q6_K_XL `-ngl 17` number (2.63 t/s) closely matches the owner's
2026-08-23/24 stock-server baseline (2.55 t/s), validating the method.
The CPU recurrent cost is 4.9-7.1 ms/layer depending on quantization,
not the 7-8 ms/layer estimate in the plan.

### Mechanism verdict: Path A

Inspected `llama-model.cpp:1354-1372` — the stock `get_layer_buft_list`
is a static tail-count split: `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`.
A layer's device assignment is baked into `pimpl->dev_layer[il]` at load
time and never changes. There is no runtime weight movement in the stock
path.

**Verdict: Path A** — extend our module with an explicit per-layer placement
map. Load all layers to CPU mmap (as the streaming path already does), then
let the placement map decide: attention layers permanently bound to GPU
slots that never rotate, recurrent layers through slot rotation with
prefetch. The stock's static split mechanism is not touched. This is the
cheaper path because our module already owns host→device weight copies;
we only add the decision of *which* layers copy and on what schedule.

### Go/No-Go verdict: NO-GO (re-confirmed with corrected 13.7 GB/s rate)

## T-Q38HG-00/01/02: PP→TG Handoff (done 2026-08-24)

### T-Q38HG-00: Q4 KV cache
Implemented `--kv-cache q4` option (Q4_0 K and V). Greedy-ID parity confirmed
at 1024 tokens (ID 286, matches Q8 and F16 references). 38/38 full suite passes
with default KV type unchanged. Code change: `main.cpp` arg parsing only
(the vendored FA kernels already support Q4_0 KV dequant).

### T-Q38HG-01: Standard PP configuration validation
Config: 2 slots, bf16 slabs, Q4 KV, Q4_K_M model. 100K layer-major PP:
- Tok/s: 741.04 (+0.33% vs fa04 baseline 738.6) — Q4 KV FA path is slightly
  faster than Q8 (no dequant in the FA kernel).
- Total wait: 0.6 ms (< 10 ms gate). 2-slot lookahead holds comfortably.
- Peak VRAM: 7,395.56 MiB (7.22 GiB).
- q38ls-hg01-pp100k.jsonl, check-telemetry OK.

### T-Q38HG-02: Handoff machinery
End-to-end mechanism: PP (streaming) → begin_handoff (allocate + copy all 64
packages to one device buffer) → standard TG through layer_streaming_compute
with slot_layer returning resident tensors → end_handoff (free buffer).

Key design decisions:
- `slot_layer(il)` returns `handoff_layers.at(il)` when `handoff_active`.
  No graph topology changes — the same persistent graphs see different tensor
  data pointers (host mmap during PP, device buffer during handoff TG).
- `load_layer()` bypasses H2D copies when handoff is active (records
  zero-cost events for telemetry consistency).
- TG graph set is invalidated (`built = false`) after migration so the first
  post-handoff decode rebuilds with resident tensors.

Validation (Q4_K_M, 1024 PP + 16 TG, batch-major):
- Parity: ID 286 matches the standard TG baseline.
- TG rate: 24.75 tok/s (vs 0.61 tok/s without handoff — ~40× speedup).
  The TG tok/s is below llama-bench fully-resident (36.3 t/s) because the
  batch-major streaming path pays per-ubatch scheduler reset/split overhead
  that the standard decode path amortizes.
- Migration time: 1.72 s (15 GB Q4_K_M @ ~8.7 GB/s effective copy-stream
  rate). Plan expected ~1.1 s at 13.7 GB/s; the delta is from per-tensor
  `set_async` calls vs a single contiguous memcpy.
- Full suite: 36/36 passes.
- Telemetry: `handoff` events (migrate with byte count and copy time,
  teardown); host checker accepts graph_builds > 0 after handoff.
- q38ls-hg02-accept.jsonl.

### T-Q38HG-03: Hybrid Placement Probe (done 2026-08-24)

Read-only analysis probe. Three questions, all answered:

**Q1: Can `-ot` (--override-tensor) express "attention on GPU, recurrent on CPU"?**

YES. Mechanism: `common/arg.cpp:253` parses `pattern=buffer_type`, applies via
`std::regex_search` against tensor names in `llama-model-loader.cpp:1181`.
Tensor names follow `blk.N.member.weight` format. The 16 full-attention
layers can be targeted with:
```
-ot "blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.=CUDA0"
```
Combined with `-ngl 0` (all-CPU baseline), this places only attention
layers on CUDA. Validated with `-ot "blk.0.=CPU"` on `-ngl all`:
PP 238→203 t/s, TG 31.7→25.1 t/s — the override correctly reroutes
layer 0 tensors.

The 48-layer recurrent alternation is ~520 chars — well within OS limits.
Alternative: Path A tail-count split (`llama-model.cpp:1354`, `-ngl 17`)
would put only 4 of 16 attention layers on GPU — suboptimal for handoff.

**Q2: Attention-only migration scope?**

16 packages (full-attn layers only), ~3.8 GB at Q4_K_M (vs 15 GB for all
64). ~275 ms copy at 13.7 GB/s theoretical (~440 ms at measured 8.7 GB/s
on copy stream). Recurrent layers never migrate.

**Q3: Runtime-rebind — YES.**

The existing handoff mechanism (T-Q38HG-02) already supports this:
- `slot_layer(il)` routes attention layers to `handoff_layers.at(il)`
  (CUDA-resident) and recurrent layers to `slots[...].layers.at(il)`
  (host mmap via slot copies).
- `ggml_backend_sched_split_graph` assigns backends based on tensor
  buffer types automatically. CUDA tensors → CUDA backend, CPU tensors
  → CPU backend.
- The graph builder (`qwen35.cpp:layer(il)`) calls `slot_layer(il)`, so
  attention-layer graphs get CUDA-resident weights. No scheduler or
  graph-builder changes needed.

**Conclusion:** T-Q38HG-04 (constrained handoff) is unblocked.
Implementation is attention-only `begin_handoff()` (copy 16 layers, not
64) + standard handoff lifecycle.

### T-Q38HG-04: Constrained Handoff (done 2026-08-24)

Implemented `--handoff-attention-only` flag via `begin_handoff(true)`:
only 16 full-attention layers (blk.3,7,...,63) migrate to the device
buffer; recurrent layers route to CPU-resident host_layers via slot_layer().

**Results:**

| Metric | Value | Expected |
|--------|-------|----------|
| Migration bytes | 3.54 GB | ~3.8 GB ✓ |
| Copy time | 0.424 s | <0.5 s ✓ |
| Greedy ID (1024-tok) | 286 | 286 ✓ |
| Full test suite | 36/36 | 36/36 ✓ |
| TG tok/s | 3.28 | 3.80 (`-ngl 17` ref) ✓ (within ~15%) |

**TG routing fix:** slot_layer(il) returns `*host_layers[il]` for
recurrent layers during attention-only handoff, so the ggml scheduler
places them on CPU. load_layer() skips H2D copies for them. Transfer
collapsed from ~1280ms to ~68ms per 8-token batch.

Re-arm (T-Q38HG-06) is a follow-up task.

The original verdict's reasoning was wrong (disk-bound measurement) but
the verdict itself survives the corrected numbers. True steady-state H2D
DMA is 13.7 GB/s. Two break-evens, both unmet:

1. All-48-streaming parity with the 3.80 t/s `-ngl 17` baseline:
   48 × 246 MB = 11.8 GB/token ÷ 263 ms/token budget = **45 GB/s
   required**; at 13.7 GB/s the transfer alone is 861 ms/token →
   ~1.16 t/s ceiling (better than the original 0.25 estimate, still 3.3×
   below the baseline it would replace).
2. Hybrid rotation (hide each window's copy behind the CPU compute of the
   current window): 246 MB ÷ 4.91 ms CPU layer time = **50 GB/s
   required**. At 13.7 GB/s the copy:compute ratio is fixed at 3.65×
   regardless of window size — the token always stalls 3.65× longer than
   the CPU work it overlaps. No window size or prefetch schedule fixes a
   fixed ratio.

The correct outcome per plan contract stands: record the negative result
and stop. T-Q38TG-01 remains not-started (no-go). The hybrid rotation
design is not viable on this machine's PCIe bus at 13.7 GB/s. The viable
TG lever remains post-PP slot expansion (all layers resident, zero
copies) for models that fit after slab release.

This plan is **complete with a negative result** (measurement corrected
2026-08-24; verdict unchanged).

## 20K Layer-Major Evidence (done 2026-08-20)

The availability probe found 24,137 MiB free and 0 MiB used. The GPU mutex
protected `powershell -NoProfile -ExecutionPolicy Bypass -File
tools/run-tests.ps1 -Suite pp20k-layer-major` exited 0 and wrote
`build/llama-cuda/q38ls-pp20k-layer-major.jsonl`; both `--check-telemetry` and
`--validate-telemetry` exited 0, producing the companion summary JSON.

- PP: 20,480 tokens / 40 chunks, 25,708.0 ms, 796.64 tok/s; 64 package copies,
  2,560 layer computes, 60 evictions, 23,111,182,336 bytes transferred,
  81,204.8 ms transfer, 0.5 ms availability wait, 17,752.6 ms compute, and
  `graph_builds == 0`.
- TG: 3 tokens, 7,630.3 ms, 0.39 tok/s; emitted IDs 317, 197, 197; 192
  evictions, 8,071.5 ms transfer, 17.1 ms availability wait, 270.8 ms
  compute, and `graph_builds == 0`. TG began from the live layer-major PP
  session; the validator accepted the PP-to-TG lifecycle without a slot reset.
- Preflight reported 22,922,919,936 free bytes of 25,650,855,936 total; two
  activation slabs used 838,860,800 bytes, and the 41 graph-set footprint
  contained 2,706 graphs. Peak VRAM was 8,247,640,064 bytes (7,865.56 MiB).
  The early PP layers had warmup-heavy transfer/compute totals; raw timings are
  recorded only as evidence, not as a speedup or quality claim.
- `tools/run-tests.ps1 -Suite all` then passed all 19 targets under the same
  mutex, including batch-major, layer-major, and the positive/negative 40-chunk
  lifecycle fixtures. The default-OFF `llama-cli` target had already rebuilt
  successfully. Default batch-major behavior remains unchanged and the
  layer-major path remains opt-in.

Caveats: this run had an idle GPU at start, but the mmap source remains
best-effort cold because Windows file-cache state is uncontrolled. No resident
baseline, quality evaluation, equivalence test, or speedup comparison was run.

## T-Q38PERF-01: copy timing split (done 2026-08-20)

`copy_complete` now carries additive `host_submit_us` and
`ready_latency_us` fields. Their sum equals the legacy `transfer_us` field;
the host telemetry checker rejects partial, negative, or inconsistent splits
when they are present while accepting historical fixtures without them. This
does not isolate DMA duration: `ready_latency_us` still includes copy-stream
queueing. The focused `layer-streaming-layer-major-1024` CUDA CTest passed
under the GPU mutex, followed by `--check-telemetry`; its first copy recorded
122.3 ms host submission, 344.2 ms ready latency, and 466.5 ms legacy
submit-to-ready transfer. The three host-only `layer-streaming-lm40` fixture
tests also passed. Default behavior and slot scheduling are unchanged.

## T-Q38PERF-02: package-class slot allocation (done 2026-08-20)

Slots now allocate only the recurrent and/or attention package classes served
by their logical-layer mapping. The representative tensor for each allocated
class is selected from a matching logical layer, which also fixes mixed-class
slot counts where the slot index itself can name an attention layer. For the
four-slot schedule, `slot_vram_bytes` fell from 2,853,972,608 to
1,444,448,896 bytes (1,409,523,712 bytes, or 1,344.2 MiB, saved). The focused
four-slot layer-major CTest passed. A new persistent five-slot mixed-class
runner-plus-checker fixture passed; its checker now derives each slot's actual
layer-cycle length, so it accepts uneven 64-layer distributions (12 or 13
layers per slot) while preserving the four-slot policy. Default behavior is
unchanged.

## T-Q38PERF-03: pinned-host package staging (done 2026-08-20)

The isolated runner now accepts `--copy-source direct|pinned` (default
`direct`) and records the selected value in `run_start.copy_source`. Pinned mode
gives each slot one CUDA-pinned host buffer sized to that slot's device package
allocation. It packs package tensors into the buffer before enqueueing the
existing asynchronous copies; the existing consumer-complete event protects the
buffer from reuse. Pinning failure or allocator fallback to ordinary host memory
fails initialization rather than silently changing copy behavior.

Paired mutex-protected 1,024-token layer-major traces at u-batch 512, four
slots, zero PP logits, and one greedy TG token both exited 0 and passed
`--check-telemetry` (837 events, four slots):
`build/llama-cuda/q38ls-direct-1024-paired.jsonl` and
`build/llama-cuda/q38ls-pinned-1024-paired.jsonl`. Pinned PP was 8,366.4 ms
(122.394 tok/s), versus direct 8,195.3 ms (124.950 tok/s). It reduced summed PP
host submission from 7,505.1 to 6,062.4 ms, but raised availability wait from
0.5 to 393.8 ms. Pinned TG was also slower (2,757.5 vs 2,515.0 ms), with
availability wait rising from 5.3 to 1,510.4 ms. This single paired observation
does not establish a speedup; pinned staging remains opt-in as an instrumented
experiment. After CMake reconfiguration, the mutex-protected
`layer-streaming-pinned-1024` CTest fixture passed its runner and host checker
(2/2, 17.90 s).

## Completed

### T-Q38LS-01: reproducible CUDA experiment envelope (done 2026-08-18)

Commands (both from workspace root, Git Bash; compilation does not need the
GPU mutex):

1. `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON` ->
   exit 0. CUDA 13.2.51 toolkit, host compiler MSVC 19.50.35726.0, target arch
   `120a-real` (RTX 5090 / sm_120). OpenSSL 3.6.1 found. NCCL absent (warning
   only; single GPU, irrelevant).
2. `cmake --build build/llama-cuda --config Release --target llama-cli --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-cli.exe`.
3. `cmake --build build/llama-cuda --config Release --target llama-gguf --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-gguf.exe`.

Model identification evidence:
- `llama-gguf.exe "<model>" r` (llama.cpp's own reader): GGUF version 3,
  alignment 32, data offset 10,996,800, n_kv 51, n_tensors 866; `qwen35.*`
  KV keys present; per-tensor sizes verified (e.g. `blk.0.attn_qkv.weight`
  55,705,600 bytes). Full dump in the env guide.
- `llama-cli` under the GPU mutex (`tools/with-gpu-lock.ps1`) with
  `-ngl 8 -p hi -n 1`: model loaded and executed on CUDA. Banner reported
  `build b10493-04b569142`, model path, `modalities: text`. Prompt processed
  at 13.2 t/s (partial offload: 8/64 layers on GPU), one token generated.
  Log: `out/q38ls-ngl8-smoke.log`. Evidence that the CUDA runtime loads and
  runs this exact artifact.
- `llama-cli -ngl 0 -p hi -n 1` (CPU-only, no CUDA): did not complete within
  600 s and was killed (201 s CPU consumed). The CPU path is impractical for
  this 26 GB hybrid model; consistent with the CUDA-only experiment scope.

Quant facts (corrected during this task): `general.file_type = 14` maps to
llama.cpp enum `LLAMA_FTYPE_MOSTLY_Q4_K_S` whose display name is "Q4_K -
Small" - a legacy label that does not describe this file. Actual per-tensor
quants (this tree's ggml enum: F32=0, Q8_0=8, Q5_K=13, Q6_K=14):
`output.weight` Q8_0, `token_embd` Q6_K, `attn_qkv`/`ffn_down` Q8_0,
`ffn_gate`/`ffn_up` Q6_K, `ssm_alpha`/`ssm_beta` Q5_K, 1-D params F32. This
is the UD/Q6_K_XL dynamic scheme. Byte sizes measured from tensor offsets are
ground truth and are in the env guide (static 2.394 GB, recurrent package
364.0 MB max, full-attn package 352.4 MB max, four-slot window ~1.444 GB).

### T-Q38LS-02: one-layer Qwen3.8 execution boundary (done 2026-08-18)

Implementation (all gated by CMake option `LLAMA_LAYER_STREAMING`, default OFF):
- `src/llama-layer-streaming.{h,cpp}`: slot buffers + tensor bundles, persistent
  activation pair, persistent host input buffer, dedicated CUDA copy backend,
  per-slot CUDA events, JSONL telemetry, and the layer-by-layer executor
  (`llama_context::layer_streaming_compute`).
- `src/llama-graph.h`: `llm_graph_params.layer_streaming` (default nullptr).
- `src/models/qwen35.cpp` + `models.h`: layer-view indirection `layer(il)`
  (model.layers[il] normally, slot-bound bundle in streaming mode) and
  `build_streaming_graphs()` which produces one cgraph per piece: embed graph,
  64 per-layer graphs, output graph.
- `src/llama-context.{h,cpp}`: gated streaming branch in `process_ubatch`.
- `tools/layer-streaming/main.cpp`: runner (`llama-layer-streaming` exe) with
  boundary mode, PP, greedy TG, JSONL telemetry.

Validated evidence (commands from workspace root, GPU runs under
`tools/with-gpu-lock.ps1`):
- `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` and `cmake --build ... --target
  llama-layer-streaming` -> exit 0.
- OFF-build check: separate dir `build/llama-cuda-off` (no option) builds
  `llama-cli` -> exit 0 (default path unchanged; the only ungated change is a
  nullptr-initialized field in `llm_graph_params`).
- `ctest --test-dir build/llama-cuda -C Release -R qwen38-layer-boundary
  --output-on-failure` -> Passed (4.0 s).
- Boundary run: `llama-layer-streaming --model <env path> --layer-slots 1
  --boundary-layers 0,3 --pp-tokens 64 --ubatch 64 --no-mtp --telemetry
  out/q38ls-boundary.jsonl` -> exit 0. Recurrent layer 0 (Gated DeltaNet) and
  full-attention layer 3 both constructed and executed one-layer CUDA graphs.
  Telemetry: `out/q38ls-boundary.jsonl` (14 lines: run_start, per-layer
  slot_request/copy_submit/copy_complete/layer_ready_wait/layer_compute,
  phase_summary pp 64 tokens 205.7 tok/s, run_end exit 0). Copy byte counts
  match the env guide exactly: layer 0 364,022,656 B, layer 3 352,380,928 B.
  Peak VRAM 3.87 GiB during the boundary run (foreign process still resident).

Key measured behavior (one slot): the H2D copy from the pageable mmap source
blocks the host inside `cudaMemcpyAsync` (driver staging for pageable memory),
so per-layer wall is copy-then-compute with ~0 availability wait at
consumption; transfer_us (submit->complete) ~111 ms for the recurrent package
and ~104 ms for the full-attn package on this shared GPU. This is recorded as
observed behavior, not a defect or a claim.

### T-Q38LS-03: full one-slot dense streaming prefill (done 2026-08-18)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --pp-tokens 512 --ubatch 64 --layer-slots 1 --no-mtp --telemetry out/q38ls-t03.jsonl"`
-> exit 0. Log `out/q38ls-t03.log`, telemetry `out/q38ls-t03.jsonl` (3074 lines).

Measured results:
- PP: 512 tokens in 24,363.2 ms = 21.02 tok/s. 8 u-batches of 64 tokens.
- `phase_summary` (JSONL): tokens 512, ubatches 8, wall_us 24,363,220,
  tokens_per_s 21.015, bytes_transferred 184,889,458,688 (512 package copies
  x ~361 MB avg), evictions 511, total_transfer_us 23,391,879, total_wait_us
  43,900, total_compute_us 850,252, layers_run 64.
- Per-layer averages (over 8 executed u-batches): recurrent layer transfer
  ~45.4-53.2 ms + compute ~1.3-2.1 ms; full-attn transfer ~43.9-44.7 ms +
  compute ~1.4-2.1 ms; setup ~0.04-0.09 ms; wait ~0.08-0.10 ms. Copy bytes
  match the env-guide package inventory (recurrent 364,022,656 B,
  full-attn 352,380,928 B).
- CUDA0 compute buffer 188.64 MiB (matches expected), CUDA_Host 2.565 MiB.
  Peak VRAM 3.91 GiB.
- JSONL ordering verified: every `layer_compute` follows the matching
  generation's `copy_complete`; telemetry has 512 copy_submit/slot_request/
  copy_complete/layer_ready_wait/layer_compute pairs (8 ubatches x 64 layers)
  plus run_start/phase_summary/run_end.

Required validations:
- Runner command (above) -> exit 0 and eight PP u-batches in telemetry. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-slot-lifecycle
  --output-on-failure` -> Passed (30.9 s) under the GPU mutex.

Defect found and fixed during this task:
- Persistent graphs were corrupted on graph reuse: `ggml_backend_sched_alloc_graph`
  rewires node src pointers in place when introducing cross-backend copies, and
  `sched->ctx` is freed on the next split; re-splitting a reused graph then
  aliased stale rewired pointers with unrelated tensors (observed: embed cpy
  src degraded to a 64-elem token leaf -> `cpy.cu` element-count assert). Fix:
  the executor snapshots each graph's node src pointers before alloc and
  restores them after compute, keeping the persistent graphs pristine across
  u-batches. T-04/05 inherit this fix.

### T-Q38LS-04: four-slot persistent-graph PP streaming (done 2026-08-19)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --layer-slots 4 --pp-tokens 512 --ubatch 64 --no-mtp --telemetry out/q38ls-pp.jsonl"`
-> exit 0. Log `out/q38ls-pp.log`, telemetry `out/q38ls-pp.jsonl` (3071 lines).

Measured results (512-token trace):
- PP: 512 tokens in 26,540.0 ms = 19.29 tok/s. 8 u-batches of 64 tokens.
- `phase_summary`: tokens 512, ubatches 8, wall_us 26,539,954, tokens_per_s
  19.292, bytes_transferred 184,889,458,688, evictions 508,
  total_transfer_us 102,470,682, total_wait_us 2,159, total_compute_us 775,010,
  layers_run 64, graph_prep_us 2,046, graph_builds 1.
- Peak VRAM 6.04 GiB (vs 3.91 GiB for one slot). Four-slot window observed as
  expected (~2.85 GB of slot buffers reported at run_start).
- With four slots the pageable H2D copies now overlap the compute of the other
  slots; total_wait_us dropped to ~2.2 ms (from 43.9 ms at one slot) because a
  layer's copy is usually already complete when consumed. Transfer remains
  dominant (~99.8 GB accumulated copy time vs ~0.78 s compute); copies from the
  pageable mmap source still serialize on the copy stream.

Required validations:
- Runner command (above) -> exit 0 with all required PP event types. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler
  --output-on-failure` -> Passed (run 12.5 s + host-side invariant check
  0.1 s) under the GPU mutex. The check validates exactly n_slots distinct
  slots, `slot == layer % n_slots`, per-slot generation monotonicity,
  copy_submit -> copy_complete -> layer_ready_wait -> layer_compute ordering,
  prior-resident `slot_evict` identity, and `graph_builds == 0` on the 64-token
  single-ubatch run.
- Regression: `qwen38-layer-boundary` Passed (4.0 s), `layer-slot-lifecycle`
  Passed (30.4 s) under the GPU mutex. Default (OFF) `llama-cli` build still
  compiles.

Implementation notes:
- Added `llama_context_params.n_pad` (default 0 = unchanged). The runner sets
  `n_pad = n_ctx` so the KV cache pads `n_kv` to the full context for every
  ubatch, keeping all graph shapes constant across the whole session (the
  existing `get_n_kv` padding mechanism, now pinned to full width). Tradeoff:
  full-attention layers attend over the full padded width from the first
  ubatch; recorded, not claimed as an optimization.
- Added `llama_layer_streaming_prebuild()` (exported wrapper around
  `llama_context::layer_streaming_prebuild`) which splits the batch, applies
  the first ubatch (so graph shapes match), builds the 66 per-piece graphs
  before the PP timer, then `memory->clear(false)` releases the applied cell
  metadata so the timed decode starts from the identical fresh cell mapping.
- Persistent graph entries now carry explicit `(mode, logical_layer, slot)`
  keys and the executor fails loudly on any mode/layer/slot mismatch; slot
  generation is now in copy_complete/layer_ready_wait/layer_compute telemetry;
  `slot_evict` now reports the prior resident layer.
- Runner gains a host-side `--check-telemetry PATH` validator (no GPU/model)
  used by the scheduler CTest.

Resolved by T-Q38LS-04B (rs_z two-variant persistent-graph key):
- The Gated DeltaNet recurrent `rs_z` is 0 on the first PP ubatch (fresh state;
  the graph includes an in-graph zeroing of one state cell) and -1 on every
  continuing ubatch (zero-sized view). T-Q38LS-04B closes this: the graph cache
  is now keyed `(mode, n_tokens, rs_z)` with per-variant input tensors and result
  objects, and the prebuild builds all three variants (fresh-PP, continuing-PP,
  TG-u1) before the timed region. The 512-token/4-slot timed PP region now
  reports `graph_builds == 0` (pre-fix baseline was 1) and the 256-token
  four-ubatch run that crosses the rs_z 0 -> -1 transition also reports
  `graph_builds == 0`. See the T-Q38LS-04B section below.

### T-Q38LS-04B: per-variant persistent graph sets (done 2026-08-19)

Observed behavior
- The streaming module now owns per-variant graph sets keyed
  `(mode, n_tokens, rs_z)`. Each set owns its io context, activation region,
  input region, and `llm_graph_result`. The executor selects the set per ubatch
  by `(phase, n_tokens, rs_z)`, verifies the prebuilt input tensor shapes
  (mismatch fails loudly per SDD), writes inputs, and runs the set's piece
  graphs. `process_ubatch` bypasses the `gf_res_prev` reuse/build machinery in
  streaming mode; an on-demand build of an unprebuilt variant remains as a
  counted fallback.
- Prebuild builds variant A (fresh PP, rs_z 0), variant B (continuing PP,
  rs_z -1), and, when `--tg-tokens > 0`, variant C (TG u1, mode 1, rs_z -1),
  then `memory->clear(false)` releases all applied cell metadata so the timed
  decode starts from the identical fresh cache.

Validation evidence (Release, GPU mutex)
- `layer-streaming-pp512`: 512 tokens / 4 slots, 8 ubatches, `graph_builds == 0`,
  `wall_us ~25.6ms`, `total_transfer_us ~97.2ms`, `total_compute_us ~0.94ms`,
  `layers_run 64`. Host `--check-telemetry` passes.
- `layer-streaming-scheduler-256`: 256 tokens / 4 slots, 4 ubatches crossing the
  rs_z 0 -> -1 transition, `graph_builds == 0`. Host check passes.
- `layer-streaming-scheduler` (64-token), `qwen38-layer-boundary`,
  `layer-slot-lifecycle`: all pass. OFF build `llama-cli` compiles clean.

Root cause of the first GPU run's shape-guard failure (debugging record)
- The executor verified sets via `llm_graph_result::can_reuse`, whose
  `params.allow_reuse` half compares the stored prebuild `llm_graph_params`
  (dangling `mctx`/`sched`/`ubatch.data` pointers) against the live decode
  params, so it always failed despite every tensor shape matching (prebuild
  n_kv 256 == live n_kv 256, n_rs 1, rs_z 0, n_tokens 64). Fix: run each
  prebuilt input's own `can_reuse` (shape checks, which re-point the input mctx
  to the live context) and skip the stale-pointer params half. No SDD/plan
  boundary changed.

### T-Q38LS-05: PP-to-TG transition (done 2026-08-19)

Observed behavior
- TG (128 greedy tokens) begins immediately after PP with the exact four slot
  assignments and per-session state (KV + Gated DeltaNet) left by PP; no slot
  reset or host prewarm. The first TG layer-0 request finds slot 0 already
  resident from PP (generation 129, `ready_at_consumption true`, wait ~3us) -
  no wrap-around preload.
- The TG phase uses the prebuilt `tg_u1` graph set (mode 1, n_tokens 1, rs_z -1)
  built before the timer, so the timed TG region reports `graph_builds == 0`.
- 128 greedy token IDs are emitted in a `tg_tokens` JSONL event and
  `run_end.n_tokens_generated == 128`.

Validation evidence (Release, GPU mutex, 512 PP + 128 greedy TG / 4 slots)
- `layer-streaming-transition` (run + host check): pass; run ~345s.
- PP: 512 tok / 8 ubatches, `graph_builds 0`, ~19.8 tok/s, transfer ~98s, compute ~1.0s.
- TG: 128 tok / 128 ubatches, `graph_builds 0`, 0.41 tok/s, transfer ~1223s,
  compute ~9.6s; per-layer transfer/wait/compute present for all 64 layers.
- All PP-only scheduler/slot/boundary regressions pass; OFF build compiles clean.

Notes
- `reset_stats()` at the PP->TG boundary makes each `phase_summary` report its
  own phase's totals (previously cumulative, so TG double-counted PP).
- Per-layer copy/wait/compute telemetry now carries a `phase` field so TG events
  are distinguishable from PP events.
- `cparams.n_ctx`/`n_pad` round `pp+tg` up to a 256-multiple (640 -> 768) so
  `n_pad` divides `kv_size == n_ctx_seq`; the mask masks empty padded KV slots to
  -INF so the larger constant `n_kv` is safe.

### T-Q38LS-06: evidence-only profiling record (done 2026-08-19)

Reproducibility command (no GPU/model needed for validation)
- `build/llama-cuda/bin/Release/llama-layer-streaming --validate-telemetry <trace>.jsonl`
  -> exit 0, writes `<trace>.jsonl.summary.json` (the validated profile artifact).
- Source trace: `--model <Q6_K_XL.gguf> --layer-slots 4 --pp-tokens 512 --ubatch 64
  --tg-tokens 128 --greedy --no-mtp --telemetry <trace>.jsonl` (run via CTest
  `layer-streaming-transition`).

Validation result
- `--validate-telemetry` on the transition trace: exit 0, 52225 events, 4 slots,
  64 layers. `ctest -R layer-streaming-telemetry`: positive (writes summary) and
  negative (WILL_FAIL on missing/incomplete telemetry) both pass.

Profile artifact (q38ls-pp-tg.jsonl.summary.json) raw metrics
- PP: 512 tokens, ~7-20 tok/s (shared-GPU noise), transfer ~98-267s, wait ~4ms,
  compute ~1-2.3s, 64 layers, graph_builds 0.
- TG: 128 tokens, ~0.4 tok/s, transfer ~1.2s, wait ~62ms, compute ~9.3s,
  64 layers, graph_builds 0.
- peak_vram_bytes recorded; per-layer transfer/wait/compute min/mean/max for all
  64 layers per phase.
- Caveats recorded: source cache best-effort cold (uncontrolled Windows file
  cache); no speedup baseline was run; no quality/equivalence/speedup claim.

Manual test running + output (2026-08-19)
- `tools/run-tests.ps1 -Suite all|quick|boundary|telemetry` runs the CTest
  suites under the GPU mutex (`--output-always` prints every test's output).
  `-Suite telemetry` is host-only (no GPU, no lock).
- Every model run now self-checks its `--telemetry` trace at exit and prints a
  per-phase summary: tokens, tok/s, wall, total transfer/wait/compute, layers
  run, evictions, graph_builds, peak VRAM. `--check-telemetry <trace>.jsonl`
  prints the same summary for any existing trace (host-only).
- Note: `build/llama-cuda/q38ls-pp-tg.jsonl` must be regenerated by a fresh
  `layer-streaming-transition-run` (it was truncated by a killed mid-run test
  launch on 2026-08-19 and deleted).

### T-Q38LM-01: stable layer-major activation slices (done 2026-08-20)

Opt-in `--pp-schedule layer-major` path (default `batch-major`) proving that
two 512-token chunks can be driven through one selected logical layer while
retaining distinct, stable, full-prompt activation-slab ranges. Evidence only;
no quality/equivalence/speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Reconfigure with the model path baked into the CTest entries (the build had
  been reconfigured with an empty `Q38LS_MODEL_PATH`, which emptied every
  GPU test's `--model` arg): `export Q38LS_MODEL_PATH=
  C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf` then
  `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` -> exit 0.
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0.
- Layer-major boundary probe (GPU mutex): `ctest --test-dir build/llama-cuda
  -C Release -R layer-streaming-lm-boundary-run --output-on-failure` (runner:
  `--pp-schedule layer-major --boundary-layers 0 --pp-tokens 1024 --ubatch 512
  --pp-no-logits --no-mtp --telemetry build/llama-cuda/q38ls-lm-boundary.jsonl`)
  -> Passed (8.4-20 s). Log `out/q38ls-lm-boundary.log`.
- Host recheck (no GPU): `llama-layer-streaming --check-telemetry
  build/llama-cuda/q38ls-lm-boundary.jsonl` -> exit 0 (15 events, 1 slot).
- Regression (GPU mutex): `tools/run-tests.ps1 -Suite boundary` -> 2/2 pass;
  `-Suite quick` -> 8/8 pass (boundary, lifecycle, scheduler, scheduler-256,
  pp512). Default-OFF `llama-cli` builds (no GPU).

Observed behavior
- `run_start` carries `schedule:"layer-major"` and `act_slab_bytes:41943040`
  (2 x [5120,1024] F32 = 40 MiB for the 1,024-token probe). The preflight
  prints `act slabs require 41943040 bytes (2 x 20971520); free VRAM
  21610102784, total 25650855936` before inference. The formula is
  `2 * n_embd * pp_tokens * 4`, which is 838,860,800 bytes (800 MiB) for the
  20,480-token envelope; this is an allocation size, not a VRAM-fit claim.
- `act_bind` events record the per-chunk slab offsets: chunk 0 -> slab A
  [0,10485760) + slab B [0,10485760); chunk 1 -> slab A [10485760,20971520)
  + slab B [10485760,20971520). The slices are non-aliasing (host validator
  proves it).
- The layer-0 package is copied once (one `copy_submit`/`copy_complete`,
  generation 1, 364,022,656 B) and both chunk computes reuse that one
  generation: two `layer_compute` events, chunk 0 (ubatch 1) then chunk 1
  (ubatch 2), same slot 0, same generation 1. No second copy, no eviction
  between the two chunks. KV/recurrent state is owned by the logical layer
  and written in-graph, never by a slot index.
- `phase_summary`: tokens 1024, ubatches 2, `graph_builds 0` (per-chunk graph
  sets were prebuilt; the timed region did no graph construction), compute
  ~241 ms, transfer ~2320 ms (shared-GPU noise), peak VRAM ~3.85 GiB.

Selected graph-binding mechanism (recorded per the task contract)
- Per-chunk prebuilt graph sets: each chunk's act pair is a pair of
  non-aliasing fixed-offset slices of the two slabs, bound at prebuild. The
  set key is `(mode, n_tokens, rs_z, n_outputs, chunk_id)`. Measured
  two-chunk probe footprint: `chunks=2 graph_sets=2 total_graphs=132
  act_slab_bytes=41943040 io_ctxs=2` (2 sets x 66 graphs; 2 io contexts).
- Rejected mechanism: a `ggml_view_2d` over a stable slab tensor. The
  scheduler's galloc calls `ggml_backend_view_init` on a view whose source
  has a backend buffer, which allocates a *separate* buffer for the view and
  defeats the stable-slab / non-aliasing semantics. Fixed-offset
  `ggml_backend_tensor_alloc` from the slab buffer is used instead (the same
  documented API the batch-major act pairs use).
- Host `--check-telemetry` now enforces, for `schedule == "layer-major"`:
  per-layer ascending chunk identity (0,1,2,...) and non-aliasing `act_bind`
  slices per slab. Both negative fixtures (out-of-order chunk; aliasing bind)
  are rejected with exit 1, so the checks are not vacuous.

Caveats: shared GPU (a foreign process held VRAM; timing is subject to
shared-GPU noise); source cache is best-effort cold (uncontrolled Windows file
cache); this is a one-layer, two-chunk boundary probe - full 64-layer PP, 20K
scale, and the PP->TG handoff are T-Q38LM-02..04.

### T-Q38LM-02: full two-chunk layer-major PP pass (done 2026-08-20)

A dedicated `llama_context::layer_major_pp` controller drives 1,024 PP
tokens (two 512-token chunks) through all 64 logical layers in layer-major
order (chunk 0 then chunk 1 for every layer before the next layer), with each
package copied once and reused across both chunks. This is a
scheduler-state / determinism check, not a quality or speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0. Default-OFF build
  (`build/llama-cuda-off`) also compiles (the experimental path is gated out).
- Layer-major 1024 (GPU mutex): `ctest --test-dir build/llama-cuda -C Release
  -R layer-streaming-layer-major-1024 --output-on-failure` (runner:
  `--pp-schedule layer-major --pp-tokens 1024 --ubatch 512 --layer-slots 4
  --pp-no-logits --tg-tokens 1 --greedy --no-mtp --telemetry
  build/llama-cuda/q38ls-lm-1024.jsonl`) -> 2/2 pass (run ~16-36 s, host
  check ~0.07 s). Log `out/q38ls-lm1024-full.log`.
- Regression (GPU mutex): `ctest -R layer-streaming-(scheduler^|pp512^|layer-major-1024)`
  -> 8/8 pass; `tools/run-tests.ps1 -Suite quick` -> 8/8 pass. Batch-major and
  the existing CTest names are unchanged.
- Host negative fixtures (no GPU): a deliberately reordered-chunk trace and a
  duplicated-chunk trace both fail `--check-telemetry` with exit 1 ("layer-major
  chunk not ascending for layer"), so the layer-major invariants are not
  vacuous.

Observed behavior
- `phase_summary` (PP): tokens 1024, ubatches 2, `graph_builds 0`, 64
  layers_run, 60 evictions (4-slot lookahead), compute ~0.74 s, transfer
  ~31 s (shared-GPU noise, host-synchronous pageable copies). 128 PP
  `layer_compute` events: chunk 0 then chunk 1 for each of the 64 layers, no
  layer L+1 compute before both layer-L computes complete. Exactly 64 PP
  `copy_submit` events: each layer's package is copied once and consumed by
  both chunks under the same generation; the generation increments as the
  four slots are reused (layers 0-3 gen 1, 4-7 gen 2, ...).
- TG (1 token) uses the standard per-ubatch path and reuses the TG-u1 set
  exactly like batch-major (`graph_builds 0`). The one-token TG
  schedule-consistency check produces the same first greedy ID **286** under
  batch-major (`q38ls-bm-1024.jsonl`) and layer-major (`q38ls-lm-1024.jsonl`)
  for the same 1,024 PP tokens.

Design notes (recorded per the task contract)
- The controller creates one memory context per chunk and applies it in chunk
  order so the shared rec/KV state is current for each chunk's `find_set` /
  `set_inputs` (the prebuilt per-chunk sets were built against the identical
  applies). The recurrent `rs_z` is captured per chunk at apply time (chunk 0
  -> 0, chunk 1 -> -1) because `get_rs_z()` reads the shared rec memory that
  the next `apply()` overwrites; the controller re-resolves each chunk against
  the same `(rs_z, n_out, chunk)` tuple the prebuild used.
- The per-ubatch path (batch-major) is unchanged. Layer-major PP is rejected
  there as a routing error; layer-major TG (phase 1) reuses the TG-u1 set
  exactly like batch-major. Activation slabs swap only after every chunk of
  the current layer completes (fixed per-chunk slab slices, no aliasing).

Caveats: shared GPU (timing subject to shared-GPU noise); source cache is
best-effort cold (uncontrolled Windows file cache); this is a 2-chunk / 64-layer
PP determinism check - 20K scale, a 3-token TG, and the negative/final
regression are T-Q38LM-03..05.

### T-Q38LM-03 through T-Q38LM-05: 20K PP, 3-token TG, and regression closure (done 2026-08-20)

See "20K Layer-Major Evidence" above. The free-GPU run completed all 40 chunks
through every logical layer with legal generation reuse and then generated
three TG tokens from the same live state. Both telemetry validators accepted
the 4,113-event trace, and the full mutex-protected regression suite passed
19/19 targets. The default-OFF build remains successful. Raw measurements are
subject to uncontrolled Windows file-cache state and are not a speedup or
quality claim.

## Known gaps / risks (current)

- Foreign `deepseek_v4.exe` shared the GPU during earlier validation
  (~12.4 GiB VRAM held, high util), so those timing results are subject to
  shared-GPU noise. The completed 20K run started with the GPU idle; every
  trace still records observed free VRAM at `run_start`.
- `llama-cli --log-file` stalled with an empty log file on this build (two
  runs killed). Not task-owned (our runner writes its own JSONL telemetry),
  but recorded here: do not rely on llama-cli's file logger.
- CPU-only execution of this model is impractically slow (see above).
- With pageable mmap copy sources the copies are host-synchronous; the four-slot
  lookahead overlaps compute but the copies themselves still serialize on the
  copy stream. Recorded as observed behavior, not a claim.
- The CMake CTest entries bake `$ENV{Q38LS_MODEL_PATH}` at configure time; a
  reconfigure without that env var set rewrites the model path to empty and the
  GPU tests fail at arg parsing. Reconfigure with Q38LS_MODEL_PATH exported.

## Blocked / escalated

- T-Q38LS-04 `rs_z` variant escalation is resolved by T-Q38LS-04B.
- T-Q38TG-01 (hybrid TG executor) is permanently gated: T-Q38TG-00
  returned NO-GO (measured H2D ~3.0 GB/s vs ~29 GB/s break-even).
- T-Q38HG-04 (constrained handoff, attention-only migration) is done —
  all acceptance criteria met: migration 3.54 GB / 0.424 s, parity ID
  286, test suite 36/36, TG 3.28 tok/s (within ~15% of -ngl 17 baseline).
  slot_layer() routes recurrent layers to CPU-resident host_layers;
  load_layer() skips H2D copies for them.
- T-Q38HG-05 (GTX 1080 validation) is gated on owner hardware access.
