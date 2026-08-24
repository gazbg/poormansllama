# Qwen3.8 Per-Width Graph Sets Plan (large-context feasibility)

Status: done (Planner-approved 2026-08-23; T-Q38PW-00 done 2026-08-23; T-Q38PW-01 done 2026-08-23; T-Q38PW-02 done 2026-08-23; T-Q38PW-03 gated)

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
Predecessors: [arbitrary-length plan](qwen38-arbitrary-pp-plan.md) (done),
[20K layer-major plan](qwen38-pp20k-plan.md) (done). Environment facts:
[docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md). Observed state:
[docs/currentstate.md](../currentstate.md).

## Owner decisions (2026-08-23)

1. SDD architecture change approved: the persistent graph cache moves from
   per-chunk sets to per-width sets with per-chunk runtime patching.
2. The zero-timed-build invariant is relaxed: small GPU-side patch/update
   operations inside the timed region are acceptable, provided they are
   telemetered and their cost is measured against per-chunk compute.
3. Target: any context > 192K is a success; 262K (the advertised max) is the
   stretch goal. Performance must be monitored at every step; any tok/s
   regression vs the 100K baseline (511.118 tok/s) must be investigated and
   explained, not just recorded.
4. Order: make 192K work first (T-Q38PW-01 gate); the 262K stretch comes
   after (T-Q38PW-02).
5. bf16 activation slabs are pre-approved for the 262K stretch, conditional
   on the parity gate in T-Q38PW-02 (100K f32 vs bf16 comparison with a
   documented tolerance; no quality claim). 192K itself fits with f32 slabs
   and does not require them.

## Measured walls (evidence, 2026-08-21/23)

| Resource | Formula | 100K (works) | 192K (target) | 262K (stretch) |
| --- | --- | --- | --- | --- |
| Chunks (ubatch 512) | ceil(pp/512) | 196 | 384 | 512 |
| Graph-set io ctx (host RAM) | ~101 MB/set x n_chunks | 19.7 GB | 38.8 GB | 51.5 GB |
| Host input buffer | ~21 KB/token x pp | 2.1 GB | 4.0 GB | 5.4 GB |
| Activation slabs (VRAM, f32) | 2 x pp x 5120 x 4 B | 4.1 GB | 7.9 GB | 10.7 GB |
| Q8 KV cache (VRAM) | 16 layers x 262144 cells scaled | 3.3 GB | 6.4 GB | 9.1 GB |
| Slots + compute + static + RS (VRAM) | ~constant | ~4.6 GB | ~4.6 GB | ~4.6 GB |

- 100K passed at 511.118 tok/s, peak 12,085.56 MiB
  (`q38ls-pp100k-layer-major-q8.jsonl`).
- 262K failed during prebuild at chunk ~505/512:
  `tensor 'inp_embd' exceeds backing buffer (offset=5434509056
  size=10485760 buffer=5435817984)` — the host input buffer (sized for the
  whole prompt) is the first wall; the ~51.5 GB io-context total is the
  second. Trace: `q38ls-pp262k-layer-major-q8.jsonl` (partial, 1011 lines,
  no run_end). Runner fix applied in the same session: prompt token IDs now
  wrap modulo vocab (`batch.token[i] = i % n_vocab`); before the fix, 262K
  failed earlier with `init: invalid token[0] = 248320`.
- VRAM conclusion: with per-width sets, 192K fits (~18.9 GB < 22.9 GB free);
  262K does not (~24.2 GB) unless the f32 activation slabs are reduced
  (bf16 slabs would halve them to 5.4 GB — a numerics change, owner decision
  required, see T-Q38PW-02 escalation).

## Why per-chunk sets exist today

Each chunk bakes `q_pos_base = token_start`, `n_kv_valid = token_start +
n_tokens`, its act-slab slice offsets, its input-buffer region, and its KV
write positions into the captured graphs at prebuild time. That is why there
is one io context per chunk. All of these vary per chunk but none of them
change graph topology — they are parameters, which is what makes per-width
sets possible.

## Deferred levers (decided 2026-08-23, not tasks)

- **Q4 KV cache** (~9.1 -> ~4.5 GB at 262K): not needed for 192K or 262K
  with bf16 slabs. Deferred as the final optimization in the plan
  (T-Q38PW-03) — implemented only against a measured need (parity gate
  failure, reduced free VRAM, or a >262K goal). Rejected for now also
  because a second numerics change would confound the bf16 parity
  attribution (one numerics variable at a time).
- **Chunk-windowed hybrid order** (bounds slabs to a window): rejected for
  now — it re-streams all 64 packages per window (~8x transfer bytes at
  window 64), trading a memory problem for a bandwidth problem on a machine
  where pageable copies are already the slow part. Revisit only if both slab
  and KV levers are exhausted. Note (2026-08-23 discussion): prefetching
  chunk activations while keeping layer-major order was also considered and
  rejected — activations are read-write with a reuse distance of the whole
  chunk set, so any window < N pages every slice once per layer (~655 GB
  PCIe traffic at 262K), which cannot hide behind ~16 ms/chunk compute at
  the measured 40-300 MB/s pageable copy rate. Only an order change
  (windowing) makes chunk streaming coherent.
- **ubatch > 512**: no longer a footprint lever once per-width sets exist
  (chunk count no longer drives host RAM; slabs and KV are per-token). The
  SDD envelope stays unchanged.

## Dependency order

```text
T-Q38PW-00 -> T-Q38PW-01 -> T-Q38PW-02 -> T-Q38PW-03 (Q4 KV, last, gated)
T-Q38PERF-06 (stream overlap) remains deferred, unchanged.
```

### T-Q38PW-00: Patchability probe (feasibility)

Status: done
Depends on: none
Commit: pending (no commit authorized)

Goal
: Determine, with measured evidence, which per-chunk patching mechanism works
  in this vendored ggml and what it costs, before any controller rewrite.
  This is a feasibility/instrumentation task; it does not prove lifecycle
  correctness or performance.

Scope
: `src/llama-layer-streaming.{h,cpp}`, `src/models/qwen35.cpp`,
  `tools/layer-streaming/main.cpp` (probe flag), ggml CUDA backend read-only
  inspection (`ggml/src/ggml-backend/ggml-backend-cuda.cu` graph
  capture/launch path).

Non-goals
: Controller rewrite, slab changes, 192K/262K runs, default-path changes.

Design and failure contract
: Build ONE 512-width PP set (chunk 0) at prebuild. Then attempt to execute
  it correctly against chunk 1 (different `token_start`) via candidate
  mechanisms, cheapest first:
  1. Parameter-table indirection: bake device pointers to a small patch
     buffer (q_pos_base, n_kv_valid, slab offsets, input base, KV write
     position) into the graph once; per chunk, update the buffer with one
     small H2D copy before launch.
  2. CUDA-graph exec update: `cudaGraphExecUpdate` /
     `cudaGraphExecKernelNodeSetParams` on the instantiated exec.
  3. Cheap re-instantiation from the same io context (no ggml rebuild, no
     allocation) if 1 and 2 are impossible.
  A mechanism "works" only if chunk 1's output activations and KV writes are
  bit-identical (or within a documented fp tolerance) to a reference
  per-chunk-set run of the same two chunks. Any mechanism that silently
  produces chunk 0's positions or another chunk's data fails loudly.

Implementation steps
: 1. Inspect the ggml CUDA graph capture/launch path; record what is baked
    (kernel params vs tensor pointers) and what update APIs are reachable.
: 2. Identify every per-chunk-varying value in the qwen35 streaming graph
    build (`q_pos_base`, `n_kv_valid`, slab view offsets, `inp_embd` region,
    KV cache row offsets, rs_z views) and where each enters the graph.
: 3. Implement the cheapest viable mechanism behind an opt-in probe flag
    (`--pw-probe`); run a 2-chunk (1024-token) layer-major PP where chunk 1
    executes through chunk 0's patched set; compare against the existing
    per-chunk-set 1024 run (telemetry invariants + final hidden state or
    first greedy ID).
: 4. Measure patch cost per chunk (new `patch_us` telemetry field) over the
    1024-token run.

Acceptance criteria
: - [ ] A written probe report (in `docs/currentstate.md`) names the chosen
      mechanism, lists every per-chunk-varying value and how it is patched,
      and records per-chunk patch cost in µs.
: - [ ] The patched 2-chunk run matches the reference per-chunk-set run on
      the chosen equivalence check and exits 0.
: - [ ] Patch cost is < 5% of the measured per-chunk compute (~1 s at 100K
      scale; ~5–7 ms at 1024 scale) or the task escalates with the numbers.
: - [ ] No default-path change: existing CTest suite passes unchanged.

Required validation
: - Build target `llama-layer-streaming` -> exit 0.
: - GPU mutex: probe 1024-token run + reference 1024-token run, both under
  `tools/with-gpu-lock.ps1`, both validated by `--check-telemetry`.
: - Host-only: `ctest -R "layer-streaming-(lm40|tail|layout)"` still passes.

State update
: Record mechanism, patch-cost numbers, and residual unknowns in
  `docs/currentstate.md`.

Escalate before implementation if
: No mechanism can patch all per-chunk values without a full ggml rebuild,
  or patch cost exceeds 5% of per-chunk compute, or equivalence requires
  changes outside the isolated experimental path.

### T-Q38PW-01: Per-width graph sets + staged input buffer

Status: done
Depends on: T-Q38PW-00 (done)
Commit: pending (no commit authorized)

Goal
: Replace per-chunk graph sets with per-width sets (full-512, tail, TG-u1:
  at most 3 sets) using the T-Q38PW-00 mechanism, and replace the
  whole-prompt host input buffer with a per-chunk staged buffer (~10 MB).
  Host-RAM footprint becomes O(1) in prompt length.

Scope
: `src/llama-layer-streaming.{h,cpp}` (set keying, patch application, input
  staging), `src/llama-context.cpp` (prebuild), `src/llama-graph.cpp`
  (patchable causal metadata), `src/models/qwen35.cpp` (graph build against
  the patch buffer), `tools/layer-streaming/main.cpp` (footprint fields,
  `patch_us` telemetry, budget accounting), host checker (accept patched-set
  telemetry), SDD section update (per-width cache replaces per-chunk cache).

Non-goals
: Slab dtype or layout changes (T-Q38PW-02), stream overlap (T-Q38PERF-06),
  TG beyond the existing cap, quality claims, default-path changes.

Design and failure contract
: Graph sets are keyed `(mode, width_class, rs_z, n_outputs)` where
  width_class in {full 512, tail width, TG 1}. Prebuild builds each distinct
  width once. Before each chunk launch the executor applies the per-chunk
  patch (T-Q38PW-00 mechanism) and verifies the patch landed (readback or
  generation counter); a failed patch fails loudly with chunk id,
  `token_start`, `n_tokens` and an `error` telemetry event — it must never
  run another chunk's positions. Slot lifetime, ascending per-layer chunk
  order, non-aliasing slab binds, and stride-1 completeness are unchanged.
  The input buffer holds one chunk's embeddings and is refilled per chunk
  (its refill must not serialize against compute — overlap with the existing
  copy pipeline or fail loudly). Telemetry gains `patch_us` per chunk and
  `graph_footprint` reports sets <= 3.

Implementation steps
: 1. Re-key sets and prebuild per width; keep the per-chunk manifest and
    host-checker contract unchanged.
: 2. Apply the probe patch mechanism in `layer_major_pp`; add `patch_us`.
: 3. Stage `inp_embd` per chunk; remove whole-prompt input sizing.
: 4. Update footprint accounting and `--graph-budget-mib` semantics
    (budget now bounds <= 3 sets).
: 5. Update the SDD residency/graph-cache sections to match.

Acceptance criteria
: - [x] 100K regression: Q8 stride-1 run exits 0, validates, `graph_builds
      == 0`, manifest unchanged (196 chunks, [160] tail), and tok/s is
      within ±5% of 511.118 (506.22 tok/s, -0.96%).
: - [x] `graph_footprint` reports <= 3 sets and io_reserved < 1 GB at 100K
      (3 sets, 288 MB; was 19.7 GB).
: - [x] Host input buffer <= 64 MB regardless of pp_tokens (64 MiB = 67,108,864 B).
: - [x] 196,608-token (384 whole chunks) Q8 stride-1 run exits 0,
      validates at stride 1, `graph_builds == 0`, peak VRAM 19,531.56 MiB
      (< 22,923 MiB free).
: - [x] Existing suite (`tools/run-tests.ps1 -Suite all`) passes; tail
      layouts (513/1/511) still validate with the tail-width set.
: - [x] Default-OFF `llama-cli` builds unchanged.

Required validation
: - All GPU commands under `tools/with-gpu-lock.ps1`, recorded with command,
  model identity, free-VRAM start, exit status, and trace path.
: - Host validator on every produced trace exits 0.
: - Performance comparison table (old vs new at 100K: wall, tok/s, patch_us,
  transfer/wait/compute splits) in `docs/currentstate.md`.

State update
: Replace in-progress state with measured evidence, the performance
  comparison, caveats, and residual risks in `docs/currentstate.md`.

Escalate before implementation if
: The 196,608 run's VRAM preflight fails (would contradict the measured
  budget), the ±5% performance gate fails without an explainable cause, or a
  patch verification gap is found that could let one chunk run another's
  positions.

### T-Q38PW-02: 262K stretch with bf16 slabs (pre-approved)

Status: done
Depends on: T-Q38PW-01 (done)
Commit: pending (no commit authorized)

Goal
: Reach the 262,144-token advertised maximum. Per the measured budget, f32
  slabs (10.7 GB) + Q8 KV (9.1 GB) + constants (~4.6 GB) ≈ 24.2 GB exceeds
  the observed 22.9 GB free. The owner has pre-approved bf16 activation
  slabs (halving slabs to 5.4 GB, ~18.8 GB total) for this task, conditional
  on the parity gate below.

Scope
: `src/llama-layer-streaming.{h,cpp}` (slab dtype + bind paths),
  `src/models/qwen35.cpp` (slab view types), runner flag
  (`--act-slab-dtype f32|bf16`, default f32 unchanged), telemetry
  (`act_slab_dtype` in run_start), host checker, SDD residency section.

Non-goals
: bf16 for anything other than the two activation slabs (KV dtype, slot
  weights, and recurrent state stay as-is), stream overlap (T-Q38PERF-06),
  quality claims, default-path changes.

Design and failure contract
: 1. First attempt 262K with f32 slabs — expected to abort at preflight with
     required-vs-available recorded (cheap evidence; confirms the budget
     model). 2. Then implement bf16 slabs behind `--act-slab-dtype bf16`:
     slab allocation, per-chunk view binding, and the patch mechanism's slab
     offset accounting all switch element type; graph topology is unchanged.
     Parity gate before any 262K claim: 100K f32 vs 100K bf16 runs, compare
     final hidden state (max abs / rel diff with a documented tolerance) and
     first greedy IDs if TG is run; record tok/s delta. 3. Then the 262K bf16
     run. A failed preflight, patch verification, or parity violation fails
     loudly — never silently.

Implementation steps
: 1. f32 262K attempt; record preflight shortfall.
: 2. bf16 slab implementation + `--act-slab-dtype` flag + run_start field.
: 3. Parity gate at 100K (f32 vs bf16): hidden-state diff + tok/s delta.
: 4. 262K bf16 stride-1 run; validate; record.

Acceptance criteria
: - [x] f32 262K preflight shortfall recorded (required vs available bytes).
      Unexpected: f32 actually fits at 262K (peak 24,463 MiB = total VRAM —
      zero headroom). Budget model was conservative.
: - [x] bf16 slabs pass the 100K parity gate with a documented tolerance and
      the diff/tok-s numbers recorded in `docs/currentstate.md` (identical
      greedy IDs [95726, 248046, 198], 0.21% tok/s delta).
: - [x] 262K bf16 run exits 0, validates at stride 1, `graph_builds == 0`,
      manifest 512 whole chunks (no tail), 262.62 tok/s, peak VRAM 19,468 MiB
      (~5 GiB under total VRAM).
: - [x] Context-vs-throughput table (20K/100K/192K/262K) in
      `docs/currentstate.md`, with any regression explained.
: - [x] Existing suite passes with the default f32 slab unchanged.

State update
: Record the outcome and the context/throughput curve in
`docs/currentstate.md`; mark the plan done.

Escalate before implementation if
: 262K still does not fit with bf16 slabs (would contradict the measured
  budget — re-check before adding Q4 KV or any second numerics change), or
  the parity gate shows a difference outside any defensible tolerance.

### T-Q38PW-03: Q4 KV cache (last optimization, gated)

Status: not-started — **gate reopened 2026-08-24** (the 8 GB target
machine makes q8 KV not fit the standard PP config; implementation moved
to T-Q38HG-00 in [qwen38-pp-tg-handoff-plan.md](qwen38-pp-tg-handoff-plan.md),
which inherits this task's contract verbatim)
Depends on: T-Q38PW-02 (done)
Commit: pending (no commit authorized)

Goal
: Add opt-in Q4_0 KV cache (`--kv-cache q4`) as the final footprint lever
  (~9.1 -> ~4.5 GB at 262K, i.e. ~4.5 GB saved on the 16 full-attention
  layers' K/V). It exists in the plan because the owner expects to need it
  in the future, but it must not start until a measured need exists.

Gate (any one suffices; record which)
: - The bf16-slab parity gate in T-Q38PW-02 failed or was rejected, and 262K
  still needs ~1.5 GB.
: - A run-time preflight shows free VRAM below the ~22.9 GB measured baseline
  (shared GPU) and a target context no longer fits.
: - The owner opens a goal beyond 262K tokens.

Scope
: `tools/layer-streaming/main.cpp` (`--kv-cache q4` option mapping to
  ggml's Q4_0 K/V types before context creation), runner preflight
  accounting, telemetry (`kv_cache` field already records the choice), SDD
  KV-cache section.

Non-goals
: Quantizing anything else (recurrent state, slot weights, slabs), quality
  claims, changing the f16/q8 defaults, default-path changes.

Design and failure contract
: Q4_0 for both K and V, selected before context creation exactly like the
  existing q8 option; flash attention enablement and KV layout follow
  llama.cpp's existing quantized-KV handling. Everything else (slots,
  slabs, graphs, telemetry invariants) is unchanged. A Q4 run validates
  with the same host checker; no silent dtype fallback — if Q4 allocation
  fails, fail loudly.

Acceptance criteria
: - [ ] A recorded measured need (gate) exists before implementation starts.
: - [ ] `--kv-cache q4` runs the 1024-token layer-major CTest-equivalent
      configuration and validates; KV buffer size in the context log shows
      ~half of the q8 size.
: - [ ] One large-context run (the context that motivated the gate) exits 0,
      validates at stride 1, `graph_builds == 0`; peak VRAM and tok/s
      recorded and compared to the q8 point.
: - [ ] Existing suite passes with the default KV type unchanged.

State update
: Record the gate that triggered the task, the VRAM savings, and the
throughput comparison in `docs/currentstate.md`.

Escalate before implementation if
: No measured need exists (do not implement on speculation), or ggml's Q4_0
KV path is incompatible with the flash-attention configuration this runner
uses.

**Gate check (2026-08-23):**

- Gate 1 (bf16 parity failed, 262K needs ~1.5 GB): parity gate **passed**
  (identical greedy IDs [95726, 248046, 198]); 262K bf16 peaks at 19,468 MiB
  with ~5 GiB headroom. **Not met.**
- Gate 2 (free VRAM < 22.9 GB, target does not fit): free VRAM is 22,923 MiB;
  262K bf16 fits at 19,468 MiB. **Not met.**
- Gate 3 (owner opens goal beyond 262K): no such goal. **Not met.**

T-Q38PW-03 remains gated. The plan is complete — all un-gated tasks
(T-Q38PW-00 through T-Q38PW-02) are done and validated. Start T-Q38PW-03
only when a measured need reopens one of these gates.
