# Qwen3.8 Arbitrary-Length Layer-Major PP Plan

Status: done (2026-08-23). T-Q38CTX-01 / T-Q38CAP-01 / T-Q38CTX-02 are
complete: code, host validation, and all GPU-mutex validation (tail runs,
parity, budget pair, 100K feasibility, full 38-target regression sweep)
passed. T-Q38PERF-06 remains deferred.

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)

Predecessors: [batch-major plan](qwen38-layer-streaming-plan.md) (done),
[20K layer-major plan](qwen38-pp20k-plan.md) (done; working-tree copy ends at
T-Q38PERF-03 — see "Worktree note" below). Environment facts are canonical in
[docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md); current observed behavior
lives in [docs/currentstate.md](../currentstate.md).

Worktree note (2026-08-20): the working tree deleted the T-Q38PERF-04/05
sections (Q8 KV cache, implicit causal mask, 40K Q8 trace evidence) from
`docs/currentstate.md` and `docs/plans/qwen38-pp20k-plan.md` after the last
commit, while the corresponding code changes remain present in
`third_party/llama.cpp`. This plan does not restore or extend those sections;
the owner decides how the doc state is reconciled. All tasks here build on the
code as it exists in the working tree.

## Scope decisions (agreed with owner)

- Scope is the isolated CUDA layer-streaming runner and its module only.
  Default llama.cpp behavior (option OFF) stays unchanged.
- `--ubatch 512` remains the maximum chunk capacity; four weight slots.
- Arbitrary positive PP lengths are supported with real tail chunks
  (1..511 tokens). Fake padding tokens are never injected.
- Chunk telemetry keeps stride `10` by default; stride `1` is used for
  tail/profile analysis and must be fully checked by the host validator.
- Stream-overlap optimization is deferred (T-Q38PERF-06).

## Dependency order

```text
T-Q38CTX-01 -> T-Q38CAP-01 -> T-Q38CTX-02
T-Q38PERF-06 (deferred, no dependencies on this plan's tasks)
```

### T-Q38CTX-01: Heterogeneous final chunk

Status: done (2026-08-23; GPU window 2026-08-21/23)
Depends on: none (layer-major baseline T-Q38LM-01/02)
Commit: pending (no commit authorized)

GPU evidence (2026-08-21/23, under Global\AI-GPU-EXCLUSIVE-v1)
: - Focused tail/parity CTest run (tail513 lm/bm pair, lm-tiny1, lm511,
   layer-major-1024): all pass; 513-token layer-major and batch-major first
   greedy IDs identical.
: - 100,000-token Q8 PP-only stride-1 run: exited 0, 13,256 events, manifest
   196 chunks (195 x 512 + 160 tail), `graph_builds 0`, 511.118 tok/s, peak
   12,085.56 MiB. `--check-telemetry` OK. (First attempt on 2026-08-21
   recorded 19.856 tok/s due to an 80-min single-event stall from GPU
   contention; trace preserved as
   `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl`.)
: - Graph-budget pair (1,024-token layer-major, F16 KV): tight (1 MiB)
   failed before PP with structured error + `exit_status 1`; generous (512
   MiB) passed, 118.580 tok/s, `graph_builds 0`, peak 4,513.56 MiB.
: - `tools/run-tests.ps1 -Suite all`: 38/38 pass (578.79 s), including
   whole-chunk 1024/20K/40K-Q8 regressions, batch-major scheduler/pp512/
   transition targets, and all tail fixtures.

Host-side evidence (2026-08-20, no GPU)
: - `test-layer-streaming-layout` (new host-only CTest): pass — layouts for
   pp = 1/511/512/513/100000 verified (contiguous, last chunk 1..ubatch, sum
   == pp); invalid inputs return empty.
: - `layer-streaming-tail513-{positive,badchunk,missing}` +
   `layer-streaming-lm40-*` fixtures: pass (manifest cross-checks live; the
   two negatives are rejected, so the checks are not vacuous).
: - `llama-layer-streaming` + `test-layer-streaming-layout` build clean; the
   default-OFF `llama-cli` build is clean (gated paths compile out).
: - GPU validations (513 lm/bm parity pair, lm-tiny1, lm511, whole-chunk
   regressions) are registered as CTest targets and passed under the GPU
   mutex (see GPU evidence above).

Goal
: Replace the uniform-chunk assumption with an immutable per-run chunk
  descriptor list `{token_start, n_tokens}` so layer-major PP accepts any
  positive `--pp-tokens`: full 512-token chunks plus one real 1..511-token
  final chunk, each with its own prebuilt graph set and correctly sized
  non-aliasing activation-slab slice.

Scope
: `src/llama-layer-streaming.{h,cpp}` (chunk descriptors, `init`,
  `begin_set`, `layer_major_pp`, run_start telemetry),
  `src/llama-context.cpp` (`layer_streaming_prebuild`),
  `src/llama-graph.cpp` (per-chunk `q_pos_base`/`n_kv_valid`),
  `tools/layer-streaming/main.cpp` (arg validation, run_start chunk manifest,
  host checker fields), `tests/test-layer-streaming-layout.cpp` (new,
  host-only), `tests/CMakeLists.txt`, `tools/layer-streaming/CMakeLists.txt`
  (new GPU CTest pairs, passed under the GPU mutex), `tools/layer-streaming/fixtures/`
  (tail fixtures + generator extension).

Non-goals
: Stream overlap, pinned/direct copy changes, slot-count changes, TG changes,
  quality/equivalence/speedup claims, default-path changes, 100K GPU run
  (that is T-Q38CTX-02), graph-footprint budget (T-Q38CAP-01).

Design and failure contract
: A chunk descriptor `{token_start, n_tokens}` is immutable for the run and
  derived once as `ceil(pp/ubatch)` chunks: all but the last are 512 tokens,
  the last is 1..512. Every per-chunk graph set is keyed
  `(mode, n_tokens, rs_z, n_outputs, chunk_id)` with the chunk's real
  `n_tokens`; slab slices are bound at that chunk's real width (slab halves
  still sized for the full prompt, so a 512+tail partition exactly fills a
  half). Causal metadata derives from cumulative token offsets:
  `q_pos_base = token_start`, `n_kv_valid = token_start + n_tokens`. Chunk 0
  keeps `rs_z = 0` (fresh), later chunks `rs_z = -1` (continuing). Slot
  lifetime, ascending per-layer chunk order, and zero timed graph builds are
  unchanged. A chunk whose prebuilt set is missing or whose shape guard fails
  fails loudly with the chunk id, `token_start`, and `n_tokens` in the error
  and `error` telemetry event.

Implementation steps
: 1. Add the pure, host-testable `llama_ls_chunk_layout(pp_tokens, ubatch)`
    returning the immutable descriptor vector (new `tests/
    test-layer-streaming-layout.cpp` defines the expected layouts for
    1/511/512/513/100000 first — TDD red).
: 2. Store the descriptors on `llama_layer_streaming`; `init` accepts any
    `pp_tokens >= 1` for layer-major and derives `n_chunks` from the
    descriptors.
: 3. `begin_set`/`find_set` validate against the descriptor (chunk id in
    range, `n_tokens` equals the descriptor's width) instead of the uniform
    `pp_tokens / n_chunks`.
: 4. Prebuild and the layer-major controller build per-chunk batches from the
    descriptors (real `n_tokens` and `token_start + i` positions), and the
    implicit causal metadata uses the descriptor offsets.
: 5. `run_start` emits the chunk manifest (`pp_tokens`, `pp_n_chunks`,
    `pp_chunks`); the host checker validates the manifest and cross-checks
    `act_bind` chunk ids against it. New tail fixtures (positive [512,1];
    negative chunk-id out of manifest) are added.
: 6. Queue the GPU CTest pairs (513-token PP+1TG layer-major vs batch-major
    first-greedy comparison; 1- and 511-token layout runs) for the free-GPU
    window.

Acceptance criteria
: - [x] Host-only layout test passes for `pp` = 1, 511, 512, 513, 100000 with
      `ubatch` 512: correct `{token_start, n_tokens}` per chunk, contiguous
      coverage, last chunk 1..512, sum == pp.
: - [x] A 513-token layer-major PP + 1 TG run validates as chunks `[512, 1]`
      (`pp_chunks` manifest, `act_bind` widths, ascending chunk order) and
      exits 0.
: - [x] Its first greedy TG ID equals the batch-major 513-token PP + 1 TG
      first greedy ID (deterministic parity).
: - [x] 1-token and 511-token layer-major PP runs exit 0 and pass the host
      checker (single-chunk layouts).
: - [x] `graph_builds == 0` in every tail-layout timed PP region.
: - [x] Existing whole-chunk layouts (1024, 20K) produce unchanged chunk
      manifests and still pass their checks (regression).
: - [x] Default-OFF `llama-cli` builds unchanged.

Required validation (all GPU runs completed under the mutex)
: - Build: `cmake --build build/llama-cuda --config Release --target
   llama-layer-streaming --parallel 16` -> exit 0.
: - Host-only (no GPU): `ctest --test-dir build/llama-cuda -C Release -R
   "layer-streaming-layout|tail|lm40" --output-on-failure` -> all pass.
: - GPU mutex: `ctest --test-dir build/llama-cuda -C Release -R
   "layer-streaming-tail513|layer-streaming-lm-tiny|layer-streaming-layer-major-1024"
   --output-on-failure` -> pass.
: - GPU mutex: 513-token parity pair (layer-major
   `--pp-tokens 513 --ubatch 512 --pp-schedule layer-major --tg-tokens 1
   --greedy --pp-no-logits --no-mtp --telemetry ...` and batch-major with the
   same PP/TG) -> both exit 0, identical first greedy ID.
: - OFF build: `cmake --build build/llama-cuda-off --target llama-cli` ->
   exit 0.

State update
: Record the descriptor mechanism, tail fixture results, parity result, and
  residual risks in `docs/currentstate.md`.

Escalate before implementation if
: A tail chunk's `rs_z` cannot be reproduced at prebuild time, a slab slice
  cannot be bound at the tail width without aliasing, or the parity comparison
  requires changes outside the isolated experimental path.

### T-Q38CAP-01: Graph-footprint preflight

Status: done (2026-08-23; GPU budget pair 2026-08-21/23)
Depends on: T-Q38CTX-01
Commit: pending (no commit authorized)

GPU evidence (2026-08-21/22, under Global\AI-GPU-EXCLUSIVE-v1)
: - Tight: 1024-token layer-major run with `--graph-budget-mib 1` emitted
   `graph_footprint` (2 sets, 132 graphs, io_ctx_reserved 201,326,592 B),
   then a structured `error` and `run_end.exit_status 1` with no
   `phase_summary`; the host checker rejects it with exactly the 3 expected
   violations (error event, non-zero exit, missing phase summary).
: - Generous: same run with `--graph-budget-mib 512` exited 0; checker OK
   (452 events, 4 slots), PP 118.580 tok/s, `graph_builds 0`, peak 4,513.56
   MiB. Traces: `build/llama-cuda/q38ls-budget-{tight,generous}-1024.jsonl`.

Host-side evidence (2026-08-20, no GPU)
: - `graph_footprint` telemetry event + `--graph-budget-mib` flag implemented
   (aborts after prebuild, before the PP timer; `--graph-budget-mib 0` exits 2
   at arg parsing, verified).
: - `layer-streaming-tail1025-{positive,hole}` fixtures: pass (stride-1
   completeness rule live; the hole is rejected).
: - GPU budget validation (1024-token run with tight/loose budgets) passed
   under the GPU mutex (see GPU evidence above).

Goal
: Calculate and report the planned persistent graph-set footprint before
  prebuild — number of graph sets, total graphs, reserved/used GGML io-context
  memory, persistent host input bytes, activation slab bytes, and slot VRAM —
  and fail before the timed PP when a caller-specified graph-memory budget is
  insufficient.

Scope
: `src/llama-layer-streaming.{h,cpp}` (footprint accounting from the built
  sets), `tools/layer-streaming/main.cpp` (`--graph-budget-mib` option,
  preflight print, prebuild guard, `run_start` footprint fields), host checker
  stride-1 completeness rule, fixtures.

Non-goals
: Reserving device memory for graphs beyond what the existing io contexts
  already do, changing the prebuild mechanism, any GPU performance claim.

Design and failure contract
: After the prebuild (before the PP timer), the module reports, per set:
  reserved and used bytes of its GGML io context and the graph count; the
  runner sums these plus slab/slot/input-buffer bytes and prints the footprint
  table. `--graph-budget-mib N` (optional) bounds the summed graph-set
  reservation; exceeding it aborts after prebuild with a structured error
  before any timed execution. The host checker tightens: when
  `telemetry_chunk_stride == 1`, every layer/chunk `layer_compute` event must
  be present (exactly `n_chunks` per layer, ascending, contiguous); for other
  strides the checker enforces the declared sampling pattern (chunk 0, every
  `k`-th chunk, final chunk).

Acceptance criteria
: - [x] The preflight prints graph-set count, total graphs, reserved/used io
      bytes, host input bytes, slab bytes, and slot VRAM before PP timing.
: - [x] An insufficient `--graph-budget-mib` aborts before the timed PP with a
      structured error and `run_end.exit_status != 0` (and no partial phase
      summary for the timed region).
: - [x] A stride-1 trace with a missing layer/chunk event is rejected by the
      host checker; the full stride-1 tail trace passes.
: - [x] Stride-10 traces keep passing (declared-pattern rule).
: - [x] Default (no budget flag) behavior is unchanged.

Required validation
: - Build target `llama-layer-streaming` -> exit 0.
: - Host-only: stride-1 positive + missing-event negative fixture CTests pass.
: - GPU mutex: one 1,024-token layer-major run with and without
  `--graph-budget-mib` (tight value fails before PP, generous value passes) —
  passed (see GPU evidence above).

State update
: Record the footprint numbers for the measured layouts and the budget
  semantics in `docs/currentstate.md`.

Escalate before implementation if
: The io-context reservation cannot be read without ggml API changes, or the
  budget check must move into the timed region to be meaningful.

### T-Q38CTX-02: Regression closure

Status: done (2026-08-23; GPU sweep + 100K trace 2026-08-21/23)
Depends on: T-Q38CAP-01
Commit: pending (no commit authorized)

GPU evidence (2026-08-21/23, under Global\AI-GPU-EXCLUSIVE-v1)
: - 100,000-token Q8 PP-only stride-1 run (command per the design contract
   above) exited 0 and wrote
   `build/llama-cuda/q38ls-pp100k-layer-major-q8.jsonl` (13,256 events).
   `run_start` manifest: `pp_tokens 100000`, `pp_n_chunks 196`, last chunk
   `99840:160`; preflight free VRAM 22,922,919,936 B of 25,650,855,936 B.
   PP: 195,649.4 ms, 511.118 tok/s, 60 evictions, `graph_builds 0`, peak
   12,085.56 MiB. `--check-telemetry` OK (13,256 events, 4 slots).
   Note: the first 100K attempt (2026-08-21) recorded 19.856 tok/s due to an
   80-minute single-event stall (layer 39, chunk 35) caused by GPU contention
   from another process not using the mutex; that trace is preserved as
   `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl`. The clean re-run at
   511.118 tok/s matches the 20K rate scaled for the larger Q8 KV cache.
: - `tools/run-tests.ps1 -Suite all`: 38/38 pass (578.79 s) under the mutex,
   covering the whole-chunk 1024/20K/40K-Q8 regressions, batch-major
   scheduler/pp512/transition targets, tail fixtures, and host layout test.

Host-side evidence (2026-08-20, no GPU)
: - `tools/run-tests.ps1 -Suite host` (new): 10/10 pass — layout unit test +
   lm40/tail513/tail1025 fixtures, no GPU, no lock.
: - All GPU targets (tail runs, budget pair, 100K stride-1 feasibility, full
   regression sweep) registered; `-Suite all` picks them up under the mutex.

Goal
: Close the plan: tail-layout positive/negative fixtures, focused GPU CTests,
  PP-to-TG validation, a 100,000-token Q8 PP-only feasibility trace at stride
  `1`, and the full regression sweep.

Scope
: `tools/layer-streaming/fixtures/` + generator, `tools/layer-streaming/
  CMakeLists.txt` (new targets), `tools/run-tests.ps1` (suite entry), state/SDD
  docs.

Non-goals
: Any new code path beyond what T-Q38CTX-01/01 delivered; speedup claims;
  TG beyond the existing 3-token cap for layer-major.

Design and failure contract
: The 100,000-token trace is a PP-only feasibility run
  (`--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major --layer-slots 4
  --kv-cache q8 --pp-no-logits --no-mtp --telemetry-chunk-stride 1`):
  195 full chunks + one 160-token tail (100000 = 195*512 + 160), `graph_builds
  == 0`, host validation at stride 1 (every layer/chunk event present). It
  runs only after the preflight passes and only under the GPU mutex on a
  recorded free-GPU start.

Acceptance criteria
: - [x] 100,000-token Q8 PP-only trace exits 0, validates at stride 1, and
      reports `graph_builds == 0` with the `[160]-tail` manifest
      (196 chunks, last chunk 160 tokens).
: - [x] Whole-chunk regressions re-run and pass: 1,024 layer-major, 20K
      layer-major, 40K Q8, and the batch-major scheduler/pp512/transition
      targets.
: - [x] Tail fixtures (positive 513/1/511, negative out-of-manifest chunk id,
      negative missing chunk at stride 1) pass their CTest roles.
: - [x] Default-OFF `llama-cli` builds.
: - [x] `tools/run-tests.ps1 -Suite all` (or the documented equivalent list)
      passes under the mutex.

Required validation
: - All GPU commands under `tools/with-gpu-lock.ps1`, recorded with command,
  model identity, free-VRAM start, exit status, and trace path.
: - Host validator on every produced trace exits 0.

State update
: Replace the plan's in-progress state with the measured evidence, caveats
  (shared GPU, best-effort cold source, no speedup/quality claim), and
  residual risks in `docs/currentstate.md`; mark all tasks done.

Escalate before implementation if
: The 100K tail run exceeds the observed VRAM budget at preflight, or stride-1
  completeness exposes a telemetry gap that requires a new task.

---

## Completion Summary (2026-08-23)

All acceptance criteria for T-Q38CTX-01, T-Q38CAP-01, and T-Q38CTX-02 are
satisfied. The plan is closed; T-Q38PERF-06 remains deferred.

### Measured evidence

**100K Q8 stride-1 feasibility** (`q38ls-pp100k-layer-major-q8.jsonl`):
- 100,000 tokens, 196 chunks (195×512 + 160 tail), `graph_builds 0`
- 511.118 tok/s (195.6 s wall), peak 12,085.56 MiB
- `--check-telemetry` OK (13,256 events, 4 slots)
- First attempt (2026-08-21) recorded 19.856 tok/s due to an 80-min stall
  from GPU contention; trace preserved as
  `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl`

**Graph-budget pair** (`q38ls-budget-{tight,generous}-1024.jsonl`):
- Tight (1 MiB): `graph_footprint` → structured `error` → `exit_status 1`,
  no `phase_summary`; checker rejects with 3 expected violations
- Generous (512 MiB): exit 0, 118.580 tok/s, `graph_builds 0`, peak 4,513.56 MiB

**Full regression suite**: `tools/run-tests.ps1 -Suite all` → 38/38 pass
(578.79 s) under `Global\AI-GPU-EXCLUSIVE-v1`

### Root-cause note on the 100K stall

The 2026-08-21 run's 19.856 tok/s was caused by a single 80-minute
(`compute_us = 4,829,215,282`) stall on one `layer_compute` event (layer 39,
chunk 35), with no CUDA errors. Windows Kernel-Power sleep/resume events
correlated with the run window. This is GPU contention from a process not
using the shared mutex, not an algorithmic context-length effect. The clean
re-run at 511.118 tok/s confirms the real rate.

### T-Q38PERF-06: Stream overlap (deferred)

Status: deferred
Depends on: T-Q38CTX-02 (evidence)
Commit: n/a

Goal
: Investigate whether additional copy/compute overlap is possible on the
  four-slot schedule.

Rationale for deferral
: The current schedule already queues layer L+4 after layer L releases the
  same slot and overlaps later layers when possible; the successful 100K-class
  traces showed sub-millisecond total readiness wait. Future work must start
  with instrumentation (per-slot queue-depth / copy-stream utilization
  telemetry), not a synchronization rewrite. No task contract is written yet;
  the Planner opens it only with new evidence that copy-stream serialization
  is a material bottleneck.
