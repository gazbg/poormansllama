# Qwen3.8 Layer-Major 20K Prompt-Processing Plan

Status: Done. T-Q38LS-07 remains blocked by the owner-requested layer-major PP design; T-Q38LM-01 through T-Q38LM-05 are complete.

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)

This is an opt-in follow-up experiment. It does not replace the SDD's normative
512-token PP / 128-token TG batch-major trace. It adds a separately measured
layer-major PP schedule. Environment facts remain canonical in
[docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md), and observed results
belong in [docs/currentstate.md](../currentstate.md).

## Scheduling Decision

The requested PP schedule is layer-major:

```text
embed chunks 0..39 into activation slab A
for logical layer L in 0..63:
	retain/package-load L once under the four-slot policy
	for chunk C in 0..39:
		execute L on A[C] and write B[C]
	swap A and B
```

Each layer must process chunks in ascending token-position order so that its
logical KV cache and Gated DeltaNet recurrent state advance causally. Logical
state remains owned by layer ID; device slots hold only immutable weights. The
existing batch-major mode remains the default and must be unchanged.

The layer-major mode needs two full-prompt F32 activation slabs. For this model
and the requested 20,480-token prompt, they require
`2 * 5120 * 20480 * 4 = 838,860,800` bytes (800 MiB), excluding graph, mask,
KV, recurrent-state, static-weight, and slot allocations. The CUDA preflight
must measure the complete allocation rather than treating this arithmetic as a
VRAM-fit claim.

## Outcome

Run one opt-in CUDA layer-streaming session with 20,480 PP tokens in 40 whole
u-batches of 512, processing all chunks through a logical layer before loading
the next layer, then generate 3 greedy TG tokens from the same live session.
Record raw telemetry for reuse of a decoder package across PP chunks. The result
is evidence, not a quality, equivalence, or speedup claim.

## Dependency Order

```text
T-Q38LS-01..06 (done) -> T-Q38LM-01 -> T-Q38LM-02 -> T-Q38LM-03 -> T-Q38LM-04 -> T-Q38LM-05
```

### T-Q38LS-07: Batch-major 20K PP with zero PP logits (superseded)

Status: blocked - do not run. The owner requested layer-major PP after this task's batch-major implementation work began.

Depends on: T-Q38LS-06

Commit: pending (no commit authorized)

Goal
: Preserved historical task contract for the batch-major 20K path. It is not acceptance evidence for the requested layer-major experiment.

Scope
: The experimental `LLAMA_LAYER_STREAMING` path only: graph-set selection keyed by `(mode, n_tokens, rs_z, n_outputs)`, PP prebuild variants, the isolated layer-streaming runner flag, `tools/run-tests.ps1`'s `pp20k` suite, focused regression validation, and evidence/state documentation.

Non-goals
: Changing the SDD baseline trace, output-quality evaluation, long-context tuning, changing KV/recurrent-state ownership, changing the four-slot policy, optimizing performance, CPU fallback, MTP, MoE, dependencies, public APIs, upstream work, commits, or running the owner's 20K experiment before the owner frees the shared GPU.

Known evidence
: The failed trace reached PP u-batch 2 and isolated the mismatch to `llm_graph_input_out_ids`: prebuilt variant B had 512 output rows while the real intermediate u-batch had 0. `n_kv` remained 20,736 throughout. The graph-set key has already been extended with `n_outputs`. Sets A `(pp, 512, 0, 0)`, B `(pp, 512, -1, 0)`, and C `(tg, 1, -1, 1)` build; the extra final-PP B-prime set with one output row asserts during graph allocation. The fixed 64 MiB persistent host-input buffer is also insufficient: each padded `[20736, 512]` F32 KQ mask is about 40.5 MiB, and the two PP sets must not alias. Estimated peak VRAM is 9-10 GiB on a 24,463 MiB shared RTX 5090, subject to preflight.

Unknowns to resolve by validation
: Whether llama.cpp accepts `n_outputs_max = 1` with every PP output flag false, whether all 40 PP u-batches select the two prebuilt zero-output PP sets without fallback construction, and whether shared-GPU free VRAM is sufficient when the owner starts the run.

Design and failure contract
: Add `--pp-no-logits` as an opt-in runner-only flag. When enabled, every PP token has `batch.logits[i] = false`; TG still requests one logits row per decode step. Set `n_outputs_max` to 1 when TG is requested and to 0 otherwise, unless an existing llama.cpp allocation contract requires clamping it to 1. Prebuild only the fresh PP set `(0, 512, 0, 0)`, continuing PP set `(0, 512, -1, 0)`, and TG set `(1, 1, -1, 1)`. Keep `n_outputs` in the graph-set key and shape guard. A missing set, shape mismatch, fallback graph build, telemetry invariant failure, or insufficient VRAM must fail loudly; it must not silently emit PP logits, rebuild in a timed phase, use another layer's weights, or fall back to CPU. PP-to-TG session state and live slot assignments remain intact.

Local implementation hypothesis
: Removing B-prime and making all PP u-batches use `n_outputs == 0` will leave exactly two PP graph shapes plus one TG shape, so the existing set selector will report `graph_builds == 0` in both phases. The cheapest disproof is a 64-token PP / 3-token TG run with `--pp-no-logits`; any context-creation error, shape mismatch, or nonzero `graph_builds` falsifies it before the 20K run.

Implementation steps
: 1. In `src/llama-context.cpp`, remove the B-prime final-PP prebuild block, retain the running applied-cell count used by TG, and update the variant-B comment for the zero-output PP contract.
: 2. In `tools/layer-streaming/main.cpp`, replace `--pp-logits-last` with `--pp-no-logits`, set every PP output flag false when enabled, and keep at most one output row for TG.
: 3. Size `llama_layer_streaming`'s persistent host-input buffer for two padded KQ masks at the runner's largest u-batch plus the existing 64 MiB input allowance; pass the padded context width from the runner so it does not allocate for the model's 262K metadata limit.
: 4. Keep the existing `n_outputs` graph-set key changes in `src/llama-layer-streaming.{h,cpp}` and `src/models/qwen35.cpp`; verify all declarations, callers, diagnostics, and shape checks agree.
: 5. Keep only diagnostics useful after the defect is understood. Remove the newly added prebuild/input shape debug prints gated by `LLAMA_LAYER_STREAMING_DEBUG`; do not remove older T-Q38LS-04 executor diagnostics unless they obstruct validation.
: 6. Use `tools/run-tests.ps1 -Suite pp20k` as the single owner-facing 20K command. It must invoke `--pp-no-logits`, write `build/llama-cuda/q38ls-pp20k.jsonl`, acquire `Global\AI-GPU-EXCLUSIVE-v1`, and return the runner's exit code.
: 7. Build and run the focused small sanity check before requesting the owner-run experiment. If it fails, repair the same slice and repeat this check before broader validation.
: 8. After the owner supplies a successful 20K trace, run the shared-path regression suite and OFF build, then update the completed plan's T-Q38LS-04B implementation note and `docs/currentstate.md` with observed evidence only.

Acceptance criteria
: - [ ] `--pp-no-logits` is documented by `--help`, accepted by the runner, and causes all PP output flags to be false while preserving one TG logits row when `--tg-tokens > 0`.
: - [ ] The graph-set key and lookup include `n_outputs`; no B-prime/final-PP graph set is built.
: - [ ] A 64-token PP / 3-token TG sanity run exits 0, self-validates its telemetry, and reports `graph_builds == 0` for PP and TG.
: - [ ] The owner-run `pp20k` suite exits 0 with PP `tokens == 20480`, `ubatches == 40`, TG `tokens == 3`, and exactly 3 greedy token IDs.
: - [ ] The 20K trace passes the host telemetry validator; both phase summaries report `graph_builds == 0`, all lifecycle/order invariants pass, and peak VRAM plus observed free VRAM are recorded.
: - [ ] All 12 layer-streaming CTest targets pass and the default-OFF `llama-cli` target builds.
: - [ ] Documentation records raw PP/TG wall time, tokens/s, transfer/wait/compute totals, evictions, peak VRAM, generated IDs, per-layer highlights, and the shared-GPU/file-cache caveats without a speedup or quality claim.

Required validation
: - Build (no GPU): `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - Small sanity (GPU mutex required): run `build/llama-cuda/bin/Release/llama-layer-streaming.exe` with the environment-guide model, `--layer-slots 4 --pp-tokens 64 --ubatch 64 --tg-tokens 3 --greedy --no-mtp --pp-no-logits --telemetry build/llama-cuda/q38ls-pp-no-logits-smoke.jsonl` through `tools/with-gpu-lock.ps1` -> exit 0; both summaries have `graph_builds 0`.
: - Owner-run experiment after freeing the shared GPU: `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite pp20k` -> exit 0 and writes a complete `build/llama-cuda/q38ls-pp20k.jsonl`.
: - Host recheck (no GPU): `build/llama-cuda/bin/Release/llama-layer-streaming.exe --check-telemetry build/llama-cuda/q38ls-pp20k.jsonl` -> exit 0 and prints PP/TG summaries.
: - Shared-path regression (GPU mutex handled by script): `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite all` -> all 12 targets pass.
: - Default path regression (no GPU): `cmake --build build/llama-cuda-off --config Release --target llama-cli --parallel 16` -> exit 0.

Owner-run handoff
: The implementer must not start the 20K command while the owner's other model occupies the shared GPU. Stop after the build and focused sanity evidence, report the exact command, and let the owner run `tools/run-tests.ps1 -Suite pp20k`. Completion remains pending until the resulting trace and summaries satisfy the acceptance criteria.

State updates
: Add a `20K PP experiment` section to `docs/currentstate.md` with the exact command, artifact path, raw measurements, three generated token IDs, observed free/peak VRAM, and caveats. Add an implementation note under T-Q38LS-04B in `docs/plans/qwen38-layer-streaming-plan.md` stating that `n_outputs` was added to the graph-set key; do not reopen completed task status. Mark this task done only after all criteria and validations pass.

Escalate before broadening if
: Zero-output PP requires changing normal non-streaming behavior, stable graph reuse requires unsafe tensor mutation, the small sanity run reports a timed fallback graph build, the 20K run cannot fit after the owner frees the GPU, or any fix would change the SDD's state/slot invariants, dependency policy, model artifact, or public compatibility.

## Layer-Major Implementation Backlog

The existing unvalidated `--pp-no-logits`, B-prime removal, and enlarged
per-chunk input-buffer changes are batch-major pre-work. They may be reused only
after the first layer-major task verifies their lifecycle assumptions. They do
not make `tools/run-tests.ps1 -Suite pp20k` a layer-major test yet.

### T-Q38LM-01: Prove stable layer-major activation slices

Status: done (2026-08-20)

Depends on: T-Q38LS-06

Commit: pending (no commit authorized)

Implementation evidence (observed only; see docs/currentstate.md T-Q38LM-01)
: Opt-in `--pp-schedule layer-major` added (default `batch-major`, invalid
  values rejected before CUDA). Two stable full-prompt F32 activation slabs
  (A/B) allocated per `2 * n_embd * pp_tokens * 4`; each PP chunk owns a
  non-aliasing fixed-offset slice of each slab, bound at prebuild into a
  per-chunk persistent graph set (chunk id is part of the set key). Chunk 0
  owns the fresh-state set (rs_z 0); each later chunk owns its own
  continuing-state set (rs_z -1). The rejected mechanism was a `ggml_view`
  over the slab: the scheduler galloc calls `ggml_backend_view_init` on a
  view whose source has a backend buffer, allocating a separate buffer and
  defeating the stable-slab semantics. `layer_compute` now carries a `chunk`
  field; `run_start` carries `schedule` + `act_slab_bytes`; `act_bind` events
  record the per-chunk slab offsets. The host `--check-telemetry` validator
  enforces, for layer-major, per-layer ascending chunk order and
  non-aliasing slab binds. Two-chunk / layer-0 / 1024-token / 512-ubatch
  probe: exit 0, one package generation, two `layer_compute` events (chunk 0
  then 1, same generation), `graph_builds == 0`, host check exit 0; negative
  fixtures (out-of-order chunk, aliasing bind) rejected. Batch-major CTest
  (boundary + quick suites) unchanged and green; default-OFF `llama-cli`
  builds.

Goal
: Establish an opt-in `--pp-schedule layer-major` path that safely processes two 512-token chunks through one selected logical layer while retaining distinct full-prompt activation-slab ranges.

Scope
: `docs/qwen38-layer-streaming-sdd.md`, `src/llama-layer-streaming.{h,cpp}`, `src/llama-context.{h,cpp}`, `src/models/qwen35.cpp`, the isolated runner, telemetry validator, and one focused CUDA test. Default and batch-major streaming paths are read-only regression surfaces.

Non-goals
: Full 64-layer PP, 20K execution, TG, performance conclusions, CUDA Graph capture, direct tensor-data-pointer mutation, host-spilled activations, or changing the default scheduler.

Design and failure contract
: Add a schedule selector whose default is `batch-major`; reject unknown schedules before model execution. Allocate two stable device activation slabs with one non-aliasing slice per PP chunk. The layer-major graph mechanism must use documented ggml tensor/view or buffer-allocation APIs. It must not retarget a persistent graph by assigning a different `ggml_tensor::data` address. Before implementation selects either prebuilt per-chunk views or explicit fixed-stage copies, measure graph/context and device-memory cost in the two-chunk probe; record the selected mechanism in the SDD. Preflight reports required and available VRAM and aborts before inference when insufficient. A selected layer executes chunk 0 before chunk 1, using its logical state; no state is addressed by device slot.

Implementation steps
: 1. Add the layer-major scheduling and activation-residency contract to the SDD while preserving the baseline batch-major contract.
: 2. Add `--pp-schedule batch-major|layer-major`, defaulting to `batch-major`, and emit the selected schedule plus activation-slab bytes in `run_start` telemetry.
: 3. Implement two stable activation slabs and a bounded two-chunk slice proof for `--boundary-layers 0`; select the graph-binding mechanism only after measuring its graph and allocation footprint.
: 4. Extend host telemetry validation to require monotonically increasing chunk identity for one layer-major logical-layer pass.

Acceptance criteria
: - [ ] The SDD names layer-major PP as a separate opt-in experiment and records the activation-residency, causal-order, slot-ownership, and telemetry contracts.
: - [ ] `--pp-schedule` defaults to batch-major; invalid values fail before CUDA inference; the existing batch-major focused CTest still passes unchanged.
: - [ ] A 1,024-token / 512-ubatch layer-0 probe executes chunks 0 then 1 through different stable activation ranges and passes host telemetry validation.
: - [ ] The probe emits one logical-layer package generation and two `layer_compute` events in the same generation; it never writes KV/recurrent state via a slot index.
: - [ ] Preflight reports the computed 800 MiB two-slab requirement for the 20K configuration and fails loudly if the complete allocation cannot fit.

Required validation
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - GPU mutex: layer-major boundary probe with `--pp-schedule layer-major --boundary-layers 0 --pp-tokens 1024 --ubatch 512 --pp-no-logits` -> exit 0; telemetry orders chunk 0 before chunk 1.
: - `llama-layer-streaming --check-telemetry <probe>.jsonl` -> exit 0.
: - `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite boundary` -> exit 0.

State update
: Record the selected graph-binding mechanism, measured activation/graph allocation, exact probe command, and any rejected mechanism in `docs/currentstate.md`.

Escalate before implementation if
: No documented ggml mechanism can bind non-aliasing chunk slices without timed graph construction, the minimum two-slab allocation does not fit after the owner frees the shared GPU, or the mechanism requires a default-path change.

### T-Q38LM-02: Complete a two-chunk layer-major PP pass

Status: done

Depends on: T-Q38LM-01

Commit: pending (no commit authorized)

Goal
: Process 1,024 deterministic PP tokens as two chunks through all 64 logical layers in layer-major order, with each package copied once and reused across both chunks.

Scope
: The layer-major executor, per-layer/chunk graph selection, four-slot layer lookahead, memory-context traversal, activation-slab swapping, schedule-aware telemetry validation, and one focused CUDA CTest.

Non-goals
: 20K scale, TG, quality claims, batch-major scheduler changes, CPU fallback, or an optimization comparison.

Design and failure contract
: For each logical layer, execute chunk 0 then chunk 1 before any compute of the next layer. A package may be copied once and consumed by both chunks under the same slot generation; its slot cannot be reused until both consumers complete. The controller must use supported `llama_memory_*` context operations to bind each chunk's logical KV/recurrent state. It must not replay a chunk with reset state or pre-apply later chunks such that an earlier attention/recurrent operation observes future state. Activation slabs swap only after every chunk of the current layer completes.

Implementation steps
: 1. Add a layer-major PP controller rather than routing this mode through the batch-major `llama_decode` loop unchanged.
: 2. Build/select layer-specific graphs and memory inputs for fresh chunk 0 and continuing chunk 1 without timed graph construction.
: 3. Keep the four-slot lookahead in logical-layer order: before layer L, resident targets are L through L+3; reuse the L package for both chunks; then request L+4 only after L's final consumer.
: 4. Add schedule-aware JSONL invariants and a CTest that rejects layer-major traces with out-of-order chunks, premature evictions, repeated copies, or timed fallback builds.

Acceptance criteria
: - [x] The trace contains 128 PP `layer_compute` events: chunks 0 and 1 for each logical layer, with no layer L+1 compute before both layer-L computes complete. (verified: 128 PP layer_compute, ascending chunk order per layer, host check green)
: - [x] The trace contains exactly 64 PP `copy_submit` events and each generation has one `copy_complete`, two consumers, and a legal final eviction or run end. (verified: 64 PP copy_submit, each layer copied once and reused across both chunks under the same generation)
: - [x] PP reports `graph_builds == 0`; any unprebuilt graph variant fails the run rather than silently entering the timed phase. (verified: phase_summary graph_builds=0)
: - [x] A one-token TG schedule-consistency check produces the same first greedy ID for 1,024 PP tokens under batch-major and layer-major modes. (verified: first greedy ID 286 under both schedules)
: - [x] Existing 512-token batch-major scheduler and transition tests remain green. (verified: scheduler/scheduler-256/pp512 + quick suite 8/8 green)

Implementation evidence
: Dedicated `llama_context::layer_major_pp` controller (src/llama-layer-streaming.cpp): creates one memory context per chunk, applies it in chunk order (so the shared rec/KV state is current for each chunk's `find_set`/`set_inputs`), runs the prebuilt per-chunk sets in layer-outer/chunk-inner order with the four-slot lookahead, and hands off to the standard per-ubatch TG path. The per-ubatch path (batch-major) is unchanged; layer-major PP is rejected there as a routing error, and layer-major TG reuses the TG-u1 set exactly like batch-major.
: Rec-current rs_z is captured per chunk at apply time (chunk 0 -> 0, chunk 1 -> -1) because `get_rs_z()` reads the shared rec memory that the next `apply()` overwrites; the prebuilt per-chunk sets are already selected against these values, so the controller re-resolves with the same (rs_z, n_out, chunk) tuple.

Required validation
: - GPU mutex: layer-major `--pp-tokens 1024 --ubatch 512 --layer-slots 4 --pp-no-logits --tg-tokens 1 --greedy` -> exit 0 with the stated event counts.
: - Host `--check-telemetry` on the layer-major trace -> exit 0; a deliberately reordered/duplicated fixture -> nonzero exit.
: - `ctest --test-dir build/llama-cuda -C Release -R "layer-streaming-(scheduler|pp512|layer-major-1024)" --output-on-failure` through `tools/with-gpu-lock.ps1` -> pass.

State update
: Record event counts, slot-generation behavior, the batch-major/layer-major first-TG-ID comparison, and residual graph-memory cost in `docs/currentstate.md`.

Escalate before implementation if
: Supported memory-context APIs cannot preserve per-layer causal state across the chunk-major inversion, the schedule requires copying weights per chunk, or layer-major output differs from the batch-major schedule-consistency check.

### T-Q38LM-03: Scale layer-major PP to 20K tokens

Status: done (2026-08-20)

Depends on: T-Q38LM-02

Prep evidence (host-side, no GPU)
: `pp20k-layer-major` suite added to `tools/run-tests.ps1` (Direct suite, `--pp-schedule layer-major --pp-tokens 20480 --ubatch 512 --layer-slots 4 --tg-tokens 0 --no-mtp --pp-no-logits`, writes `q38ls-pp20k-layer-major.jsonl`); `run-tests.ps1` parses clean. Host `--check-telemetry` verified to accept a PP-only layer-major trace (run_end with 0 generated tokens); the `--validate-telemetry` "generated tokens but no TG phase_summary" guard is correct for PP-only. 2-chunk PP reference: 3 sets peaked at 5,873 MiB, so the 40-set / 800 MiB-slab / 20K-KV 20K run must be measured on a free GPU (the owner's `llama-server` was holding ~23.6 GiB during prep).

Commit: pending (no commit authorized)

Goal
: Complete the requested 20,480-token / 512-ubatch layer-major PP pass with 40 chunks and collect valid lifecycle telemetry.

Scope
: Layer-major capacity preflight, 40-chunk PP execution, telemetry/profile validation, and the owner-facing PowerShell suite command. TG remains the next task.

Non-goals
: TG, quality evaluation, speedup claims, changing model/context settings beyond the planned 20K envelope, default scheduler changes, or GPU runs while the owner's other workload is active.

Design and failure contract
: A complete PP pass loads each of the 64 logical layer packages once, consumes it across chunks 0 through 39 in ascending order, then releases/reuses its slot only after chunk 39 completes. PP emits 2,560 `layer_compute` events and 64 package-copy lifecycles. The full preflight allocation, including static tensors, four slots, KV/recurrent state, two activation slabs, graphs, masks, and work buffers, must fit or abort before the PP timer. Telemetry reports distinct PP graph/setup, copy, wait, device-copy-if-any, and compute time; it does not label the result a speedup.

Implementation steps
: 1. Extend the PowerShell runner with a distinct `pp20k-layer-major` suite; leave the blocked batch-major `pp20k` suite untouched until the owner explicitly removes it.
: 2. Add trace validation for 40 chunk identities per layer, 64 PP package copies, 2,560 PP computes, monotonically increasing per-slot generation, and no graph construction in the timed PP region.
: 3. Run the owner-authorized 20K PP-only experiment after the shared GPU is free; retain the complete JSONL trace and validated profile report.

Acceptance criteria
: - [x] The 20K layer-major PP run exits 0 with `tokens == 20480`, `ubatches == 40`, 64 package copies, and 2,560 layer computes.
: - [x] Every layer's 40 compute events are chunk ordered 0 through 39 and share one package generation; no slot overwrites an in-use generation.
: - [x] `phase_summary.graph_builds == 0`; the telemetry checker and profile validator both pass.
: - [x] Run-start records schedule, requested/available VRAM, activation-slab bytes, source cache mode, and slot count; run-end records peak VRAM.

Required validation
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - Owner-run GPU mutex: `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite pp20k-layer-major` -> exit 0 and writes `q38ls-pp20k-layer-major.jsonl`.
: - `llama-layer-streaming --check-telemetry build/llama-cuda/q38ls-pp20k-layer-major.jsonl` -> exit 0.
: - `llama-layer-streaming --validate-telemetry build/llama-cuda/q38ls-pp20k-layer-major.jsonl` -> exit 0 and writes its summary JSON.

State update
: Add PP-only raw timing, per-layer highlights, allocation facts, and shared-GPU/file-cache caveats to `docs/currentstate.md`.

Escalate before implementation if
: Preflight fails after the owner frees the shared GPU, profile validation cannot express legal multi-consumer generations, or measured graph storage exceeds the declared capacity envelope.

### T-Q38LM-04: Continue layer-major PP into three TG tokens

Status: done (2026-08-20)

Depends on: T-Q38LM-03

Commit: pending (no commit authorized)

Goal
: Generate 3 greedy TG tokens immediately after successful 20K layer-major PP without resetting session state or slot assignments.

Scope
: PP-to-TG graph and state handoff, runner/TG telemetry, schedule-aware validation, and the owner-facing 20K-plus-3 command.

Non-goals
: Response-quality scoring, a fully resident baseline, MTP, speculative decoding, or changing TG's established one-token batch execution order.

Design and failure contract
: After layer-major PP, KV cache and Gated DeltaNet state stay associated with their logical layers and the four live slot assignments remain intact. TG switches to the existing one-token layer order; it may copy a package only when its required logical layer is not resident. It must never clear PP state, prefill slot 0, or assume the batch-major final-slot layout. The runner must request one TG logits row without resurrecting PP logits. The trace records exactly three greedy IDs.

Implementation steps
: 1. Prebuild or otherwise prepare the TG-u1 graph set before the TG timer using the state left by the layer-major PP preparation/execution contract.
: 2. Make the executor switch phase without resetting slots or logical memory state; update the validator for a layer-major PP followed by standard TG event sequence.
: 3. Extend `pp20k-layer-major` with `--tg-tokens 3 --greedy`, then capture and validate the owner-run trace.

Acceptance criteria
: - [x] The 20K PP / 3 TG run exits 0 with PP `tokens == 20480`, TG `tokens == 3`, and exactly three emitted IDs.
: - [x] PP has 64 package copies and 2,560 layer computes; TG preserves the live PP state and reports its own transfer/wait/compute totals.
: - [x] Both phase summaries report `graph_builds == 0`; the checker accepts the PP-to-TG logical-state transition without a slot reset.
: - [x] No PP logits buffer proportional to 20,480 rows is allocated.

Required validation
: - Owner-run GPU mutex: `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite pp20k-layer-major` -> exit 0, complete PP/TG JSONL trace.
: - Host `--check-telemetry` and `--validate-telemetry` on the result -> both exit 0.

State update
: Add exact PP/TG command, raw timings, generated IDs, peak/free VRAM, slot-boundary behavior, and caveats to `docs/currentstate.md`.

Escalate before implementation if
: Layer-major PP cannot produce a valid TG-u1 prebuild without replaying PP, a state reset is required, or the handoff requires changing default llama.cpp decode behavior.

### T-Q38LM-05: Close regressions and record evidence

Status: done (2026-08-20)

Depends on: T-Q38LM-04

Prep evidence (host-side, no GPU)
: Static 40-chunk / 64-layer / 4-slot layer-major fixtures generated by `gen-lm40-fixtures.py` and checked in under `tools/layer-streaming/fixtures/`: `lm40-positive.jsonl` (valid), `lm40-badorder.jsonl` (swapped chunks), `lm40-premature.jsonl` (slot re-copied before the prior generation's chunks complete). Three host-only CTest targets (`layer-streaming-lm40-positive`, `-badorder` WILL_FAIL, `-premature` WILL_FAIL) all pass: the positive trace is accepted and both malformed traces are rejected ("chunk not ascending" and "layer_compute before matching copy_complete"). Default-OFF build: `cmake --build build/llama-cuda-off --config Release --target llama-cli --parallel 16` -> exit 0; `llama-cli --version` runs and its help carries no layer-streaming flags (the opt-in path is gated out).

Commit: pending (no commit authorized)

Goal
: Demonstrate that the opt-in layer-major mode is isolated, regression-safe, and documented as evidence only.

Scope
: Layer-major CTests and negative telemetry fixtures, the existing full suite, default-OFF build, the SDD, this plan, and `docs/currentstate.md`.

Non-goals
: New feature work, performance comparison, commits, or upstream-facing artifacts.

Design and failure contract
: The default CMake-off path and default batch-major runner behavior remain unchanged. Validation must cover both schedules, reject malformed layer-major ordering/lifecycle traces, and record measurements without claiming equivalence or speedup.

Implementation steps
: 1. Add focused positive and negative layer-major telemetry fixtures to the existing host validator suite.
: 2. Run all layer-streaming CTests under the GPU mutex, then build default-OFF `llama-cli`.
: 3. Update the SDD, current state, and plan statuses with only observed facts; leave the blocked T-Q38LS-07 history intact.

Acceptance criteria
: - [x] Layer-major positive and negative validation tests pass with the required 40-chunk/layer ordering rules.
: - [x] All 19 layer-streaming CTest targets pass under the GPU mutex.
: - [x] `cmake --build build/llama-cuda-off --config Release --target llama-cli --parallel 16` exits 0.
: - [x] Documentation distinguishes batch-major baseline evidence from layer-major evidence and states the shared-GPU/file-cache limitations.

Required validation
: - `powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite all` -> all existing targets pass.
: - New host-only layer-major telemetry tests -> pass, including malformed-order and premature-eviction negatives.
: - `cmake --build build/llama-cuda-off --config Release --target llama-cli --parallel 16` -> exit 0.

State update
: Mark T-Q38LM-01 through T-Q38LM-05 done only with recorded command results, trace paths, and residual risk. Do not change the completed T-Q38LS-01 through T-Q38LS-06 statuses.

Escalate before implementation if
: Any regression alters default batch-major behavior or the default-OFF build, validation cannot distinguish legal package reuse from stale weight use, or the observed results contradict the layer-major state contract.

### T-Q38PERF-01: Split host submission from ready latency

Status: done (2026-08-20)

Depends on: T-Q38LM-05

Goal
: Make pageable-host submission cost observable without changing the slot scheduler.

Scope
: Experimental layer-streaming copy telemetry, its host checker, and evidence documentation.

Design and failure contract
: `copy_complete.transfer_us` remains the compatibility submit-to-ready interval.
New `host_submit_us` measures the host copy-issue interval and
`ready_latency_us` measures enqueue-complete to ready-event observation; their
sum must equal `transfer_us`. The checker accepts historical traces without the
new fields but rejects partial, negative, or inconsistent fields when present.

Acceptance criteria
: - [x] The isolated runner builds.
: - [x] A CUDA layer-major trace emits the timing split and passes its self-check.
: - [x] Historical 40-chunk host fixtures remain valid.

Validation evidence
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - GPU mutex: `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-layer-major-1024 --output-on-failure` -> 2/2 passed.
: - `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-lm40 --output-on-failure` -> 3/3 passed.

### T-Q38PERF-02: Allocate only the package classes a slot serves

Status: done (2026-08-20)

Depends on: T-Q38PERF-01

Goal
: Reduce experimental slot VRAM without changing logical-layer to slot mapping or copy lifecycle.

Scope
: Slot-bundle allocation in the experimental path and focused four-slot validation.

Design and failure contract
: A slot allocates a recurrent bundle only when at least one logical layer that
maps to it is recurrent, and an attention bundle only when at least one mapped
layer is attention. Representative tensors must come from the matching class,
not from the slot index. Missing class bundles fail before copy or graph
execution; default llama.cpp behavior remains unchanged.

Acceptance criteria
: - [x] Four-slot `slot_vram_bytes` is lower than the current dual-bundle allocation (2,853,972,608 -> 1,444,448,896 bytes).
: - [x] The four-slot layer-major lifecycle test still passes.
: - [x] A mixed-class slot-count smoke test completes without a missing bundle.

Required validation
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - GPU mutex: focused four-slot layer-major CTest -> pass.
: - GPU mutex: `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-slot-mixed-5 --output-on-failure` -> 2/2 passed.

### T-Q38PERF-03: Opt-in pinned-host package staging

Status: done (2026-08-20)

Depends on: T-Q38PERF-02

Commit: pending (no commit authorized)

Goal
: Measure whether slot-owned CUDA-pinned host staging changes observable copy submission, readiness, or phase timing without changing the direct-source default.

Scope
: Experimental slot-owned pinned host buffers, `--copy-source direct|pinned`, trace labeling, one focused pinned CUDA fixture, and evidence documentation.

Non-goals
: Changing the slot scheduler, default direct mmap source, CUDA graph behavior, file-cache policy, output quality, a speedup claim, dependencies, CPU fallback, or a broader multithreaded pack pipeline.

Design and failure contract
: Pinned mode allocates one staging buffer per device slot, packs the selected immutable package into that buffer, and enqueues existing asynchronous tensor copies from the packed storage. The slot's existing consumer-complete event prevents overwriting staging storage while its prior generation may be consumed. Pinned allocation failure or fallback to ordinary host memory is a structured initialization error; it must not silently use direct/pageable storage. `run_start.copy_source` labels every trace. Direct remains the default.

Acceptance criteria
: - [x] The isolated runner accepts only `direct` and `pinned` copy-source values and labels `run_start` accordingly.
: - [x] A pinned layer-major PP/TG trace completes and passes lifecycle validation.
: - [x] A same-workload direct/pinned pair records split timing evidence without a speedup claim.
: - [x] A persistent pinned fixture exercises the opt-in lifecycle path.

Validation evidence
: - `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16` -> exit 0.
: - GPU mutex: direct and pinned runs with `--pp-schedule layer-major --layer-slots 4 --pp-tokens 1024 --ubatch 512 --pp-no-logits --tg-tokens 1 --greedy --no-mtp` -> both exit 0; both `--check-telemetry` checks report 837 events and 4 slots.
: - The paired traces are `build/llama-cuda/q38ls-direct-1024-paired.jsonl` and `build/llama-cuda/q38ls-pinned-1024-paired.jsonl`. Pinned PP was 8,366.4 ms / 122.394 tok/s versus direct 8,195.3 ms / 124.950 tok/s; pinned TG was 2,757.5 ms versus direct 2,515.0 ms. Pinned reduced PP host submission from 7,505.1 to 6,062.4 ms but increased PP availability wait from 0.5 to 393.8 ms. It is retained as an opt-in measurement path, not a performance default.
: - GPU mutex: `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-pinned-1024 --output-on-failure` -> 2/2 passed (17.90 s).

State updates
: Record the source mode, paired trace paths, measured split timings, and the absence of a speedup claim in `docs/currentstate.md` and the SDD.

Escalate before broadening if
: A future change needs staging-buffer reuse before the consumer-complete event, needs a worker thread or new dependency, changes direct-mode behavior, or a controlled repeated study establishes a reason to alter the default.
