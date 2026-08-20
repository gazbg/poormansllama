# Qwen3.8 Layer Streaming Plan

Status: Done (all tasks T-Q38LS-01 through T-Q38LS-06 complete; see per-task Status and currentstate.md)

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)

Environment facts (model path, hardware, toolchain, per-layer package sizes)
are canonical in [docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md), not in
this plan or in source code. The model path used in validation commands below
is the environment-guide-defined path. Current observed behavior lives in
[docs/currentstate.md](../currentstate.md).

## Outcome

Run the specified Qwen3.8-27B Q6_K_XL GGUF through a real CUDA layer-at-a-time execution path with four pageable decoder-layer slots, then capture PP and TG telemetry. The result is evidence, not a speedup or quality claim.

## Dependency Order

```text
T-Q38LS-01 -> T-Q38LS-02 -> T-Q38LS-03 -> T-Q38LS-04 -> T-Q38LS-04B -> T-Q38LS-05 -> T-Q38LS-06
(all complete)
```

### T-Q38LS-01: Establish a reproducible CUDA experiment envelope
Status: done
Depends on: none
Commit: pending (no commit authorized)

Goal
: Record the exact model, hardware, toolchain, GGUF metadata, and CUDA build command needed to reproduce later traces.

Scope
: `docs/AGENT_ENVIRONMENT.md`, `docs/currentstate.md`, ignored local build/output paths, and no model-runtime behavior changes.

Non-goals
: Layer streaming, model conversion, downloads, package installation, or benchmark execution.

Design and failure contract
: The task records only observed facts. It must identify whether static tensors, four largest layer packages, persistent state, and runtime buffers can fit into the RTX 5090 allocation budget. An inability to build CUDA or read the GGUF marks the next task blocked; it is not repaired by changing model quality or using a CPU fallback.

Implementation steps
: 1. Capture the current GPU, driver, CPU, RAM, compiler, CMake, and llama.cpp revision.
: 2. Inspect GGUF architecture, tensor sizes, split status, and MTP metadata.
: 3. Build unmodified vendored llama.cpp with CUDA and record the exact command and result.
: 4. Create the environment and current-state records with only verified facts.

Acceptance criteria
: - [x] The model path and GGUF byte size are recorded.
: - [x] The model identifies as the expected Qwen3.5-family hybrid architecture.
: - [x] CUDA build success or exact blocker is recorded with one reproducible command.
: - [x] Capacity preflight inputs are recorded without inventing a VRAM fit claim.

Required validation
: - `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON` -> configuration succeeds or reports the exact toolchain blocker.
: - `cmake --build build/llama-cuda --config Release --target llama-cli` -> target builds or reports the exact blocker.

State update
: Add environment facts and baseline build result to `docs/AGENT_ENVIRONMENT.md` and `docs/currentstate.md`.

Escalate before implementation if
: CUDA build requires a new dependency or model access fails.

### T-Q38LS-02: Prove a one-layer Qwen3.8 execution boundary
Status: done
Depends on: T-Q38LS-01
Commit: pending (no commit authorized)

Goal
: Execute one logical Qwen3.8 decoder layer through an experimental graph boundary while preserving the hidden activation and its logical session state.

Scope
: Experimental llama.cpp CUDA execution code, Qwen3.5 graph construction, and a focused test or runner mode.

Non-goals
: Four-slot scheduling, prefetch, PP/TG trace collection, persistent graph cache, or speed measurements.

Design and failure contract
: The logical layer ID controls its immutable weights and logical KV/recurrent state. The temporary boundary must not map persistent state to a slot. A linear-attention layer and a full-attention layer must each execute through the boundary. Unsupported tensor layouts fail loudly before execution.

Implementation steps
: 1. Add the experimental CMake gate and isolated runner target with default llama.cpp behavior unchanged.
: 2. Refactor only enough Qwen3.5 graph code to construct a one-layer graph from hidden input to hidden output.
: 3. Route one representative Gated DeltaNet layer and one representative full-attention layer through the boundary.
: 4. Add focused diagnostics for logical layer, graph mode, state ownership, and failure reason.

Acceptance criteria
: - [x] With the experimental option disabled, default llama.cpp build behavior is unchanged.
: - [x] A selected linear layer and full-attention layer can each construct and execute a one-layer CUDA graph.
: - [x] Persistent logical state is addressed by logical layer ID, never by device slot ID.
: - [x] Failure names the logical layer and unmet resource or tensor requirement.

Required validation
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming` -> target builds.
: - `ctest --test-dir build/llama-cuda -R qwen38-layer-boundary --output-on-failure` -> focused boundary tests pass.

State update
: Record the exact boundary implementation path, known graph limitations, and validation result in `docs/currentstate.md`.

Escalate before implementation if
: A one-layer graph cannot preserve required Qwen3.8 state semantics without a broader graph-runtime redesign.

### T-Q38LS-03: Run a full one-slot dense streaming prefill
Status: done
Depends on: T-Q38LS-02
Commit: pending

Goal
: Process 512 deterministic token IDs through all 64 decoder layers using one host-to-device layer package at a time.

Scope
: Host mmap package catalog, static-weight VRAM residency, one-slot device package ownership, layer-at-a-time execution, and structured failure reporting.

Non-goals
: Four slots, asynchronous prefetch, persistent graph variants, TG, or performance comparisons.

Design and failure contract
: Token embedding, final norm, LM head, KV cache, and Gated DeltaNet state remain resident in VRAM. Only a complete immutable logical decoder package is copied into the reusable slot. The slot cannot be overwritten before its CUDA consumer completes. The runner uses input IDs `0..511` in u-batches of 64 and labels its source as best-effort cold mmap.

Implementation steps
: 1. Build a host package catalog from the model's decoder tensor inventory.
: 2. Allocate and preflight static VRAM and one decoder-package slot.
: 3. Execute each u-batch in logical layer order through the one-slot path.
: 4. Emit JSONL lifecycle events and a raw PP summary.

Acceptance criteria
: - [x] The runner either completes all eight u-batches or refuses before execution with a structured capacity/tensor error. (completed all 8)
: - [x] Each `layer_compute` event follows the matching package generation's `copy_complete` event. (verified in JSONL ordering)
: - [x] No KV or DeltaNet recurrent state is copied through the weight slot. (slot catalog covers only decoder weight tensors)
: - [x] The run emits PP wall time and PP tokens/s without claiming a speedup. (21.02 tok/s measured; no speedup claim)

Required validation
: - `build/llama-cuda/bin/Release/llama-layer-streaming --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --layer-slots 1 --pp-tokens 512 --ubatch 64 --no-mtp --telemetry out/q38ls-one-slot.jsonl` -> exit 0 and eight PP u-batches in telemetry.
: - `ctest --test-dir build/llama-cuda -R layer-slot-lifecycle --output-on-failure` -> lifecycle test passes.

State update
: Record the observed allocation result, first successful PP trace path, and unresolved runtime costs in `docs/currentstate.md`.

Escalate before implementation if
: Static residency plus one slot does not fit, or the split path cannot complete all Qwen3.8 layer types.

### T-Q38LS-04: Add four-slot persistent-graph PP streaming
Status: done (four-slot scheduler + persistent graphs validated; the rs_z
        two-variant sub-criterion was escalated here and resolved by T-Q38LS-04B)
Depends on: T-Q38LS-03
Commit: pending

Goal
: Execute the required 512-token PP trace through four slots with prebuilt PP graph entries and asynchronous lookahead transfers.

Scope
: Four-slot scheduler, persistent `pp_u64` graphs, CUDA copy/compute event ordering, and PP telemetry.

Non-goals
: TG, graph rebuilding during timed PP, CUDA Graph capture, adaptive window sizing, or performance comparison.

Design and failure contract
: Before layer `L`, the scheduler targets `L..L+3`; while `L` computes, it requests `L+4`. A slot becomes reusable only after its consumer event completes. Persistent graph entries use stable `(pp_u64, logical_layer, slot)` keys. Graph preparation occurs before the PP timer starts.

Implementation steps
: 1. Allocate four package slots and create persistent PP graph entries.
: 2. Add copy-stream and compute-stream event dependencies.
: 3. Implement deterministic four-slot lookahead and eviction.
: 4. Emit per-layer copy, wait, compute, eviction, and PP-summary events.

Acceptance criteria
: - [x] PP completes eight u-batches using exactly four package slots.
: - [~] The timed region performs no graph construction. (one rebuild at the
       recurrent `rs_z` 0->-1 transition; see escalation note)
: - [x] Each layer event exposes transfer, availability-wait, and compute time.
: - [x] Slot generation and logical-layer checks reject stale use.
: - [x] The PP summary includes PP tokens/s, bytes transferred, waits, and peak VRAM.

Required validation
: - `build/llama-cuda/bin/Release/llama-layer-streaming --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --layer-slots 4 --pp-tokens 512 --ubatch 64 --no-mtp --telemetry out/q38ls-pp.jsonl` -> exit 0 with all required PP event types.
: - `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler --output-on-failure` -> scheduler tests pass.

State update
: Record the fixed four-slot policy, PP trace path, and raw measurements in `docs/currentstate.md`.

Escalate before implementation if
: Stable slot-address graph reuse requires unsafe tensor mutation or changes default llama.cpp graph behavior.

Escalation (rs_z two-variant persistent-graph key, 2026-08-19)
: The recurrent `rs_z` (Gated DeltaNet "first zero-ed state") is `0` on the first
  PP ubatch (fresh recurrent state; the graph includes an in-graph zeroing of
  one state cell) and `-1` on every continuing ubatch (no zeroing; a zero-sized
  view). That makes two graph variants per PP phase, not one. The single-result
  persistent-graph cache built for T-04 therefore rebuilds the graphs once in
  the timed region, at the ubatch-1 -> ubatch-2 `rs_z` transition
  (`phase_summary.graph_builds == 1` on the 512-token trace; `0` on the single-
  ubatch 64-token scheduler test). Every scheduler/slot/generation invariant
  passes. Fully meeting the "no graph construction in the timed region"
  criterion requires a per-variant graph cache (key extended to
  `(mode, logical_layer, slot, rs_variant)`) plus per-variant input tensors and
  result objects, and a `process_ubatch` change so the executor selects the
  prebuilt variant instead of rebuilding. That is an SDD graph-key design
  change; the Planner must approve it before T-04 is re-opened.

### T-Q38LS-04B: Per-variant persistent graph sets (rs_z dimension)
Status: done
Depends on: T-Q38LS-04
Commit: pending (owner has not authorized commits)

Goal
: Close the T-Q38LS-04 escalation: the timed PP (and later TG) region performs zero graph construction by prebuilding both recurrent `rs_z` variants (fresh `0`, continuing `-1`) plus the TG u1 variant, keyed per SDD as `(mode, logical_layer, slot, rs_variant)` with per-variant input tensors and result objects.

Scope
: `src/llama-layer-streaming.{h,cpp}` (graph-set storage, per-variant result objects, executor set selection), `src/llama-context.{h,cpp}` (prebuild of all variants, `process_ubatch` bypass of the `gf_res_prev` rebuild path), `src/models/qwen35.cpp` (variant-aware `begin_build`), runner (tg-aware prebuild call), and a 256-token four-ubatch CTest.

Non-goals
: TG execution (T-Q38LS-05), MoE, performance claims, default llama.cpp path changes.

Design and failure contract
: A graph set is keyed `(mode, n_tokens, rs_z)`. Each set owns its io context, activation pair (own region of the persistent act buffer), input tensors (own region of the persistent host input buffer), and result object (input holders + output references). Sets never alias. The executor selects the set by `(phase, n_tokens, rs_z)` per ubatch, verifies the prebuilt inputs match the current ubatch shapes via the existing `can_reuse` shape checks (mismatch fails loudly per SDD), writes inputs, and runs the set's piece graphs. `process_ubatch` bypasses the `gf_res_prev` reuse/build machinery entirely in streaming mode; on-demand construction of an unprebuilt variant remains as a counted fallback so no variant can go silently unhandled. The prebuild applies the fresh first ubatch (variant A, rs_z 0), a continuing second ubatch (variant B, rs_z -1), and, when `--tg-tokens > 0`, a one-token TG ubatch (mode 1, rs_z -1), then releases all applied cell metadata with `memory->clear(false)` so the timed decode starts from the identical fresh cache.

Implementation steps
: 1. Restructure the streaming module to per-variant graph sets with per-region buffer cursors.
: 2. Add per-set result objects; redirect the streaming builder to the set's result object.
: 3. Rework the executor to select the set and verify shapes before running.
: 4. Extend `layer_streaming_prebuild` to all three variants with the clear-after-apply pattern.
: 5. Add the 256-token four-ubatch scheduler CTest (variant transition inside the timed region).

Acceptance criteria
: - [x] The 512-token four-slot PP trace reports `graph_builds == 0` in the timed PP region (was 1).
: - [x] A four-ubatch (256-token) run reports `graph_builds == 0`, proving the rs_z 0 -> -1 transition uses the prebuilt continuing variant.
: - [x] All existing scheduler/slot/generation invariants still pass (validator unchanged).
: - [x] Default (OFF) build is unchanged.

Validation evidence (2026, GPU mutex, Release):
: - `layer-streaming-scheduler` (64-token run + host check): pass.
: - `layer-streaming-scheduler-256` (256-token run + host check): pass; `phase_summary.graph_builds == 0`.
: - `layer-streaming-pp512` (512-token/4-slot run + host check): pass; `phase_summary.graph_builds == 0` (pre-fix baseline was 1), 8 ubatches, `wall_us ~25.6ms`, `total_transfer_us ~97.2ms`.
: - Regressions `qwen38-layer-boundary`, `layer-slot-lifecycle`: pass.
: - OFF build `llama-cli`: compiles clean.

Root cause of the first GPU run's shape-guard failure (recorded per debugging workflow)
: The executor verified prebuilt sets via `llm_graph_result::can_reuse`, whose `params.allow_reuse` half compares the stored prebuild `llm_graph_params` (including `mctx`, `sched`, `samplers`, `ubatch.data`) against the live decode params. Those stored pointers dangle across the prebuild -> decode boundary, so the params comparison always failed even though every tensor shape matched (prebuild n_kv 256 == live n_kv 256, n_rs 1 == 1, rs_z 0 == 0, n_tokens 64 == 64). Fix: the executor now runs each prebuilt input object's own `can_reuse` (tensor-shape checks, which also re-point the input mctx to the live context) and skips the stale-pointer params half; the set key `(mode, n_tokens, rs_z)` plus these shape checks is the actual guard. No SDD/plan boundary changed.

Note
: A host-side `layer-streaming-pp512` CTest pair (run + `--check-telemetry`) was added to capture the 512-token/4-slot trace reproducibly; the check rejects any `phase_summary.graph_builds != 0`. GPU tests read the model from `$ENV{Q38LS_MODEL_PATH}`, which must be set at CMake configure time (an unset var bakes `--model` with no path and the run fails in <0.2s).

Required validation
: - `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler --output-on-failure` (64-token run + host check) -> pass.
: - `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler-256 --output-on-failure` -> pass.
: - `build/llama-cuda/bin/Release/llama-layer-streaming --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --layer-slots 4 --pp-tokens 512 --ubatch 64 --no-mtp --telemetry out/q38ls-pp4b.jsonl` -> exit 0, `phase_summary.graph_builds == 0`.
: - Regressions: `qwen38-layer-boundary`, `layer-slot-lifecycle` -> pass.
: - OFF build: `cmake --build build/llama-cuda-off --target llama-cli` -> exit 0.

State update
: Replace the T-Q38LS-04 escalation note in `docs/currentstate.md` with the resolved multi-variant behavior and new trace evidence.

Escalate before implementation if
: The continuing-variant `rs_z` cannot be reproduced deterministically at prebuild time (it is: one state cell per sequence; fresh cell -> rs_z 0, self-referencing cell -> rs_z -1).

### T-Q38LS-05: Carry the live four-slot cache from PP into TG
Status: done
Depends on: T-Q38LS-04B
Commit: pending (owner has not authorized commits)

Goal
: Generate 128 greedy TG tokens immediately after PP without resetting the four weight slots or per-session state.

Scope
: Persistent `tg_u1` graphs, PP-to-TG transition, greedy token loop, session-state continuity, and TG telemetry.

Non-goals
: Sampling-policy exploration, output-quality scoring, MTP, multi-turn chat, or cache reset comparison.

Design and failure contract
: TG begins with the exact four slot assignments left by PP. It retains logical KV and DeltaNet state. Its first required layer is 0, so any wrap-around availability wait is valid telemetry, not a special preload. Persistent graph entries use stable `(tg_u1, logical_layer, slot)` keys and are ready before the TG timer starts.

Implementation steps
: 1. Prepare persistent TG graph entries before PP/TG timing.
: 2. Preserve PP hidden/session state and four slot records across the transition.
: 3. Run 128 greedy decode steps with the same four-slot policy.
: 4. Record generated IDs, per-layer TG timing, and phase summary.

Acceptance criteria
: - [x] TG starts without slot reset or host prewarm after PP.
: - [x] The runner emits 128 greedy token IDs or exits with a structured layer/cache error.
: - [x] TG summary includes TG tokens/s and per-layer transfer, wait, and compute data.
: - [x] The first TG layer-0 request records its real slot availability behavior.

Validation evidence (Release, GPU mutex, 512 PP + 128 greedy TG / 4 slots)
: - `layer-streaming-transition`: run + host check both pass (run ~345s).
: - PP phase: 512 tokens, 8 ubatches, `graph_builds 0`, ~19.8 tok/s, transfer ~98s total, compute ~1.0s.
: - TG phase: 128 tokens, 128 ubatches, `graph_builds 0`, 0.41 tok/s, transfer ~1223s total, compute ~9.6s; per-layer transfer/wait/compute present for all 64 layers (128 steps each).
: - 128 greedy IDs emitted in a `tg_tokens` JSONL event; `run_end.n_tokens_generated == 128`.
: - First TG layer-0 request: slot 0, generation 129, `ready_at_consumption true`, wait 3.0us (no wrap-around; the slot was still resident from PP).
: - Regressions: all PP-only scheduler/slot/boundary tests still pass; OFF build `llama-cli` compiles clean.

Implementation notes
: - The TG greedy loop, `--greedy` flag, and TG phase summary were delivered in T-Q38LS-04B groundwork; this task added the `layer-streaming-transition` CTest, per-phase stat isolation (`llama_layer_streaming::reset_stats()` at the PP->TG boundary so the TG `phase_summary` reports TG-only transfer/wait/compute and ubatch counts), a `phase` field on the per-layer copy/wait/compute events, and the `tg_tokens` telemetry event.
: - `cparams.n_ctx`/`n_pad` now round `pp+tg` up to a 256-multiple (640 -> 768 for 512+128) so `n_pad` divides `kv_size == n_ctx_seq` (the KV cache asserts `kv_size % n_pad == 0`) while keeping `n_kv` constant across the session. The mask builder masks empty padded KV slots to -INF, so the larger `n_kv` is safe.

Required validation
: - `build/llama-cuda/bin/Release/llama-layer-streaming --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf --layer-slots 4 --pp-tokens 512 --ubatch 64 --tg-tokens 128 --greedy --no-mtp --telemetry out/q38ls-pp-tg.jsonl` -> exit 0 with PP and TG summaries.
: - `ctest --test-dir build/llama-cuda -R layer-streaming-transition --output-on-failure` -> transition test passes.

State update
: Record the first PP-to-TG trace, emitted-token count, and observed wrap-around behavior in `docs/currentstate.md`.

Escalate before implementation if
: Session-state continuity requires copying KV or recurrent state through weight slots, or TG cannot use persistent graph entries safely.

### T-Q38LS-06: Produce an evidence-only profiling record
Status: done
Depends on: T-Q38LS-05
Commit: pending (owner has not authorized commits)

Goal
: Convert the completed raw telemetry into a reproducible profile record without claiming quality or speedup.

Scope
: Telemetry validator, summary/report artifact, SDD/current-state updates, and reproducibility command.

Non-goals
: Baseline comparison, optimization, changing slot count, adaptive policy, quality work, or MoE extension.

Design and failure contract
: The report distinguishes source/page-fault uncertainty, transfer time, availability waits, compute time, graph-preparation time, and wall time. It reports raw PP/TG tokens/s and must state that no baseline was run. Malformed or incomplete telemetry fails validation rather than producing a partial profile.

Implementation steps
: 1. Validate required JSONL event sequences and fields.
: 2. Produce per-layer aggregate tables and PP/TG summaries.
: 3. Record environment, command, model path, artifact identity, and known limitations.
: 4. Update the SDD/current state with measured evidence only.

Acceptance criteria
: - [x] A completed run produces a validated raw profile artifact.
: - [x] The report contains PP tokens/s, TG tokens/s, per-layer transfer/wait/compute statistics, and peak VRAM.
: - [x] The report explicitly says source cache was best-effort cold and no speedup baseline was run.
: - [x] No unsupported quality, equivalence, or performance claim appears in docs.

Validation evidence (Release, GPU mutex)
: - `llama-layer-streaming --validate-telemetry <trace>.jsonl` -> exit 0, writes `<trace>.jsonl.summary.json` (validated profile artifact). The transition trace validates with 52225 events, 4 slots, 64 layers.
: - `ctest -R layer-streaming-telemetry` -> `layer-streaming-telemetry` (positive, writes summary) and `layer-streaming-telemetry-negative` (WILL_FAIL on missing/incomplete telemetry) both pass.
: - Profile artifact fields: per-phase tokens/s, total transfer/wait/compute, per-layer transfer/wait/compute min/mean/max for all 64 layers, peak_vram_bytes, environment (model/arch/n_layer/n_slots/device/VRAM), caveats (source_cache best-effort cold, no speedup baseline, no quality/equivalence/speedup claim), generated_tokens count.

Implementation notes
: - `--validate-telemetry PATH` is a host-side mode (no GPU/model) added alongside `--check-telemetry`. It performs strict event-sequence and field validation (run_start, per-layer copy/wait/compute with phase, phase_summary per phase, tg_tokens count matching run_end, scheduler slot/generation invariants, graph_builds == 0) and writes the summary. Malformed or incomplete telemetry returns non-zero and writes no report.
: - Per-layer copy/wait/compute/slot_request telemetry now carries a `phase` field (added in T-Q38LS-05) so PP and TG are separable in the profile.
: - The summary is a generated artifact (stays out of version control per repo layout).

Manual test running (tooling, 2026-08-19)
: - `tools/run-tests.ps1` runs the CTest suites under the GPU mutex: `-Suite all` (12 targets, ~8 min), `-Suite quick` (all except the 350 s transition run), `-Suite boundary` (~35 s), `-Suite telemetry` (host-only, no GPU). It uses `--output-always` so per-test output is visible.
: - Every model run now self-checks its own trace at exit (`check_telemetry` on the `--telemetry` file), printing a per-phase telemetry summary: tokens, tok/s, wall time, total transfer/wait/compute, layers run, evictions, graph_builds, and peak VRAM. `--check-telemetry PATH` prints the same summary for any existing trace (host-only, no GPU).

Required validation
: - `build/llama-cuda/bin/Release/llama-layer-streaming --validate-telemetry out/q38ls-pp-tg.jsonl` -> exit 0 and write a complete summary.
: - `ctest --test-dir build/llama-cuda -R layer-streaming-telemetry --output-on-failure` -> telemetry validator tests pass.

State update
: Replace planned claims in `docs/currentstate.md` with the exact trace command, validation result, raw metrics, and residual risks.

Escalate before implementation if
: Required telemetry cannot distinguish transfer, availability wait, and layer compute time.

## Implementation Design (Planner record, 2026-08-18)

This section records the verified technical design for the layer-streaming path.
It is derived from reading the vendored llama.cpp internals; update it only
when implementation changes a decision.

### Tensor placement (no loader changes needed)

- Load with `n_gpu_layers = 999` so `model.dev_layer(il)` is CUDA for every
  layer: the KV cache and Gated DeltaNet recurrent state are therefore
  allocated on CUDA by the existing memory modules (residency contract).
- `tensor_buft_overrides` (existing llama.cpp mechanism, regex per tensor)
  overrides `blk\.[0-9]+\..*` to the CPU buft: decoder-layer weights stay in
  the host mmap and become the copy sources. Static tensors
  (`token_embd`, `output_norm`, `output`) keep the CUDA buft and are resident.
- The layer-streaming module creates N slot CUDA buffers plus per-slot tensor
  bundles (one recurrent-shape set, one full-attention-shape set), a
  persistent CUDA activation-pair buffer, and a persistent host input buffer.

### Graph construction (gated in `src/models/qwen35.cpp`)

- `llm_graph_params` gains `const llama_layer_streaming * layer_streaming`.
- The qwen35 graph class gains a layer-view indirection
  `const llama_layer & layer(int il)`: normal mode returns `model.layers[il]`;
  streaming mode returns the slot-bound tensor bundle for `il % n_slots`.
- Streaming mode builds one `ggml_cgraph` per piece: embed graph, one graph
  per logical decoder layer, and the output graph. Each layer graph reads the
  persistent activation pair tensor `act[il&1]` and writes `act[(il+1)&1]`
  through `ggml_cpy` (the same pattern the delta-net base uses to write
  recurrent state into persistent buffers). `gf` is reassigned per graph, so
  the existing `ggml_build_forward_expand` calls append to the right graph.
- Graph-input tensors (tokens, pos, out_ids, k/v idxs, mask, rs idxs) are
  allocated in a persistent host buffer so `set_inputs` can write them before
  any graph is allocated; the scheduler copies them H2D per split, exactly as
  in the normal path. The KQ mask requires a host buffer (existing assert).
- The gallocr uses tensors with pre-existing buffers in place: slot tensors,
  activation pair, KV/state, and inputs need no scheduler copies (except the
  intended H2D input copies).

### Execution (slot scheduler, in `src/llama-layer-streaming.cpp`)

- Copies run on a dedicated CUDA copy backend/stream; compute runs on the
  scheduler's CUDA backend. Ordering uses `ggml_backend_event_record/wait`
  (CUDA events): a layer graph consumes its slot only after the matching copy
  event; a slot copy for layer L+n waits on the consumer-done event of layer L
  (the slot's last consumer) before overwriting.
- Per-layer event sets: transfer time (copy start/end events), compute time
  (graph start/end events), availability wait (host sync on the copy-ready
  event at consumption), plus setup time. All distinct in telemetry.
- The executor runs the graph list sequentially; each graph is allocated with
  `ggml_backend_sched_alloc_graph` before its `graph_compute_async`.
- Integration point: `llama_context::process_ubatch` (streaming branch) and a
  member pointer on `llama_context`, both gated by `LLAMA_LAYER_STREAMING`.

### Files (all gated by the new CMake option `LLAMA_LAYER_STREAMING`, default OFF)

- new: `src/llama-layer-streaming.{h,cpp}`, `tools/layer-streaming/main.cpp`
- changed: `src/llama-graph.{h,cpp}`, `src/models/qwen35.cpp`,
  `src/llama-context.{h,cpp}`, `CMakeLists.txt`
- The default build (option OFF) is byte-identical in behavior: all changed
  paths are compile-time gated or default-initialized to the old values.
