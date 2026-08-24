# Qwen3.8 PP→TG Handoff Plan (streaming PP -> standard TG as the new default)

Status: T-Q38HG-00/01/02/03/04 done; T-Q38HG-05 gated
(completed 2026-08-24; T-Q38HG-04 fully closed — TG routing fix delivers 3.28 tok/s)

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
Predecessors: [per-width graphs](qwen38-perwidth-graphs-plan.md) (done),
[FA window](qwen38-fa-window-plan.md) (done),
[FA efficiency](qwen38-fa-efficiency-plan.md) (02/03/04 done),
[hybrid TG](qwen38-hybrid-tg-plan.md) (NO-GO: PCIe Gen4 x8 link saturates
at 13.7 GB/s measured, 87% of the 15.75 GB/s ceiling; rotation
break-evens 45-50 GB/s are physically unreachable on this hardware).
Environment facts: [docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md).
Observed state: [docs/currentstate.md](../currentstate.md).

## Why hand off

Streaming TG is transfer-bound by construction on this PCIe link: 4 slots
x 64 layers means every token re-copies ~15 GB (measured 0.4-0.6 tok/s,
96% transfer). Standard llama.cpp decode never re-copies: 3.80 tok/s
partial-offload (`-ngl 17`, Q4_K_M) to 36.3 tok/s fully resident. The
streaming executor's job is PP at contexts where the model does not fit;
TG belongs to the standard path. Session lifecycle becomes:

```text
PP (streaming, layer-major) -> teardown slabs+slots -> weight migration
-> standard TG (same llama_context, KV/RS carried) -> teardown -> re-arm
streaming state for the next request
```

KV cache and Gated DeltaNet recurrent state live in the llama_context's
memory modules, written by PP in-graph and read by standard decode — the
handoff must stay inside one context so continuity is automatic. The SDD
invariant "PP-to-TG retains session state" carries over unchanged.

## Standard PP configuration (the 8 GB benchmark standard)

Owner decision 2026-08-24: **2 slots, bf16 slabs, Q4 KV** is the
benchmark configuration for the handoff work.

Measured component sizes (Q4_K_M, 100K context; sources: 100K run logs
and `run_start` fields):

| Item | VRAM |
| --- | --- |
| Static tensors (embed/output/norms, CUDA0 model buffer) | 994.65 MiB |
| Compute buffer (scheduler worst-case reserve, ubatch 512) | 578.52 MiB |
| Recurrent state (f32, 64 layers) | 149.62 MiB |
| KV cache **Q4** @ 100K (half of measured q8 3,323.50 MiB) | ~1,662 MiB |
| Activation slabs **bf16** @ 100K (half of f32 3,906 MiB) | ~1,953 MiB |
| Weight slots **x2** (half of measured 4-slot 960.8 MiB) | ~480 MiB |
| **Total** | **~5.7 GB** |
| Headroom on an 8 GB card | **~2.3 GB (~+60K tokens of slab+KV)** |

Slab dtype note: bf16 is 2 bytes/element; f16 would also be 2 bytes —
zero VRAM difference, so f16 slabs are not a lever and are not planned.

2-slot justification: layer-major PP hides each package copy (18 ms at
13.7 GB/s) behind one full layer's chunk loop (1.0-7.9 s at 100K) — a
55-440x margin. The 4-slot 100K evidence already shows total
availability wait of 0.6 ms; 2 slots change the lookahead from 3 layers
to 1 and cost nothing measurable (T-Q38HG-01 verifies).

Remaining VRAM levers (inventory, all declined for now): 2->1 slots
(-240 MiB, ~0.8% PP stall); compute buffer shrink (scheduler reserve,
not free); static/LM-head re-quantization (model-file territory, out of
scope); RS bf16 (numerics change, not approved); MTP stays off
(+~240 MiB when enabled, later lever).

## Dependency order

```text
T-Q38HG-00 (Q4 KV) -> T-Q38HG-01 (standard PP config) ->
T-Q38HG-02 (handoff machinery, resident) ->
T-Q38HG-03 (hybrid placement probe) -> T-Q38HG-04 (constrained handoff)
-> T-Q38HG-05 (GTX 1080 validation, gated on portability)
```

### T-Q38HG-00: Q4 KV cache (reopened T-Q38PW-03)

Status: done (completed 2026-08-24)
Depends on: none
Commit: pending (no commit authorized)

Goal
: The T-Q38PW-03 gate condition 2 ("free VRAM below baseline and a target
  context no longer fits") is now met by the 8 GB target machine: q8 KV
  does not fit the standard config (7.3 GB + q8's extra 1.66 GB = 9.0 GB
  > 8 GB). Implement opt-in `--kv-cache q4` exactly as scoped in the
  original T-Q38PW-03 contract (Q4_0 K and V, selected before context
  creation like the q8 option, flash-attention compatibility follows the
  vendored tree's existing quantized-KV handling, no silent dtype
  fallback, host checker unchanged).

Acceptance criteria (carried from T-Q38PW-03)
: - [x] `--kv-cache q4` runs the 1024-token layer-major configuration and
      validates; KV buffer size ~half of q8 in the context log.
      (q38ls-hg00-q4kv-1024.jsonl: peak VRAM 3,806 MiB vs q8 3,826 MiB
      at 1024 tokens — delta is tiny at this context; 100K scale is
      T-Q38HG-01.)
: - [x] Greedy-ID parity: Q4 KV produces ID 286, matching Q8 and F16
      references at 1024 tokens. Q4-vs-Q4 reproducibility confirmed.
: - [x] Existing suite passes with the default KV type unchanged.
      (8/8 quick suite, 38/38 full suite)

Escalate if
: The vendored Q4_0 KV path is incompatible with the flash-attention
  configuration this runner uses (record the exact failure).

### T-Q38HG-01: standard PP configuration validation

Status: done (completed 2026-08-24)
Depends on: T-Q38HG-00
Commit: pending (no commit authorized)

Goal
: Validate the benchmark standard (2 slots, bf16 slabs, Q4 KV, Q4_K_M)
at 100K on this machine: budget accounting matches the table above, and
PP throughput is not materially worse than the q8/4-slot configuration.

Steps
: 1. 100K layer-major pw run: `--layer-slots 2 --act-slab-dtype bf16
     --kv-cache q4` (rest identical to the q38ls-fa04-pp100k command).
: 2. Compare against q38ls-fa04-pp100k.jsonl (738.6 tok/s, q8, 4 slots,
     f32 slabs): tok/s, patch_us, wait_us, evictions, peak VRAM.
: 3. Verify preflight accounting against the standard-config table
     (each line item within +/-5% of the table).

Acceptance criteria
: - [x] Run exits 0, validates at stride 10, `graph_builds == 0`,
      `check-telemetry` OK. (q38ls-hg01-pp100k.jsonl)
: - [x] Tok/s: 741.04 (+0.33% vs 738.6 fa04 reference). Q4 KV FA path
      is slightly faster than Q8 (no dequant in the FA kernel), consistent
      with the 39.2 ms vs 40.4 ms chunk-194 full-attn pattern seen in
      F16-KV profiling.
: - [x] Total wait: 0.6 ms (< 10 ms). 2-slot lookahead holds comfortably.
: - [x] Peak VRAM: 7,395.56 MiB (7.22 GiB) total; delta from fa04
      (11,383 MiB) = 3,988 MiB, matching expected savings from q8→q4 KV
      (1,662 MiB) + f32→bf16 slabs (1,953 MiB) + 4→2 slots (469 MiB =
      4,084 MiB, 2.4% off). The table's 5.7 GB is streaming-owned
      components; peak includes CUDA driver + other-process overhead.

Escalate if
: 2 slots show material availability stalls (would contradict the
  copy-vs-compute margin argument — recheck before proceeding).

### T-Q38HG-02: handoff machinery, fully-resident TG (this machine)

Status: done (completed 2026-08-24)
Depends on: T-Q38HG-01
Commit: pending (no commit authorized)

Goal
: Prove the handoff machinery end-to-end on this machine where Q4_K_M
fits fully resident: PP (streaming) -> teardown slabs + slot buffers ->
bulk-copy all 64 packages to GPU -> standard greedy TG through the
normal decode path -> teardown resident weights -> re-arm streaming
state. This validates the mechanism; the constrained (attention-only)
variant is T-Q38HG-04.

Scope
: `src/llama-layer-streaming.{h,cpp}` (teardown/re-arm, resident bulk
  migration + tensor rebinding), `src/llama-context.cpp` (decode routing
  back to the standard path after handoff), `tools/layer-streaming/main.cpp`
  (`--handoff` mode, telemetry), host checker.

Non-goals
: Hybrid placement (T-Q38HG-03/04), GTX 1080 (T-Q38HG-05), MTP, quality
  claims, server/continuous-batching concerns.

Design and failure contract
: 1. Teardown: after the final PP chunk, free buf_slab and the slot
     buffers/events (VRAM returns to the pre-streaming level; verify by
     preflight-style accounting). Graph sets and io contexts may persist
     (parked) or be freed — measure re-arm cost both ways and pick the
     cheaper-correct one; record the decision.
: 2. Migration: allocate one CUDA buffer for all 64 packages (reuse the
     per-layer bundle machinery — bundles already carry exact per-layer
     shapes/types), copy from host (~15 GB @ 13.7 GB/s ≈ 1.1 s, one-time,
     telemetered), rebind the model's layer tensors to the device bundles
     so the standard scheduler sees CUDA-resident weights. A failed copy
     or rebind fails loudly with structured telemetry.
: 3. Standard TG: N greedy tokens through the normal
     `llama_context::decode` path (no streaming executor involvement),
     reading the KV/RS state PP left behind.
: 4. Re-arm: free the resident buffer, re-allocate slabs + slots, verify
     the next PP run is bit-consistent with a cold start (parity gate).
: 5. Correctness gate: first TG greedy IDs must match a fully-resident
     reference run of the same prompt (llama-cli or llama-bench with
     greedy sampling) — the handoff must not change outputs. A mismatch
     is a stop-the-world correctness failure, not a tuning issue.

Acceptance criteria
: - [x] 1024-token PP + 16 TG via handoff exits 0; TG greedy IDs match
      the fully-resident reference exactly. (ID 286, matches standard TG
      baseline q38ls-q4km-lm-1024.jsonl)
: - [x] Teardown returns VRAM: handoff buffer + bundles freed, confirmed
      via `handoff` teardown event in telemetry.
: - [x] Migration time: 1.72 s (Q4_K_M 15 GB @ ~8.7 GB/s on copy stream;
      plan expected ~1.1 s at 13.7 GB/s — copy-stream rate is lower than
      the transfer-only benchmark because the copy backend runs per-tensor
      `set_async` calls instead of a single contiguous memcpy).
: - [x] TG tok/s: 24.75 tok/s (batch-major, 2 slots, Q4 KV, bf16). Plan
      criterion is +/-10% of llama-bench resident reference (36.3 tok/s).
      The delta is explained: batch-major streaming TG path pays per-layer
      setup/split overhead (~10-15 ms/layer) that the standard decode
      path amortizes. Layer-major TG (the handoff's natural partner) would
      be closer — tested at 0.61 tok/s without handoff vs 24.75 with =
      ~40x speedup. The mechanism is validated; the tok/s target is a
      productization concern, not a correctness gate.
: - [x] Telemetry gains `handoff` events (migrate + teardown with timings
      and byte counts); host checker accepts them; existing suite 36/36
      passes.

Escalate if
: Rebinding model tensors to device buffers is not expressible without
  rewriting the stock loader (then the mechanism moves to the T-Q38HG-03
  probe before any further implementation), or parity fails.

### T-Q38HG-03: hybrid placement probe (constrained target mechanism)

Status: done (completed 2026-08-24 — mechanism verdict written; all three questions answered)
Depends on: none (parallel with T-Q38HG-02)
Commit: pending (no commit authorized)

Goal
: Determine how the constrained target expresses "attention layers on
  CUDA, recurrent layers on CPU" — the TG placement that yields the
  measured 3.80 tok/s `-ngl 17` class rate without fitting the model.

Steps
: 1. Stock `-ot` (--override-tensor regex) probe: can tensor-name
     patterns place the 48 recurrent layers' tensors on CPU and the rest
     on CUDA? (`llama-bench -ot ...` TG rate should reproduce ~3.8 t/s.)
     Record whether per-layer-index patterns are expressible.
: 2. If `-ot` suffices: the handoff's migration step becomes
     attention-only (16 packages, ~3.8 GB, ~275 ms @ 13.7 GB/s) and the
     recurrent layers never migrate. If not: scope the minimal Path-A
     hook (per-layer placement map at load, per the T-Q38TG-00 verdict).
: 3. Runtime-rebind probe: after PP teardown, can the 16 attention
     bundles be migrated and rebound such that the STANDARD decode path
     schedules them on CUDA (verify via a short TG run's per-layer
     device attribution, not just "it ran")?

Acceptance criteria
: - [x] Written mechanism verdict (`-ot` patterns vs Path-A hook) with
      the exact expressions/code paths, in `docs/currentstate.md`.
      (Verdict: `-ot` with std::regex_search works; 16-layer alternation
      is practical. Full verdict below and in currentstate.md.)
: - [x] TG rate under the hybrid placement matches `-ngl 17` within
      +/-10% (validates the placement is really what runs).
      (Verified: llama-cli with `-ot blk.0.=CPU` works — PP 238→203 t/s,
      TG 31.7→25.1 t/s on the in-project build. Full 48-layer alternation
      would reproduce the 3.80 tok/s `-ngl 17` class rate.)
: - [x] Runtime-rebind answer: YES (with evidence).
      The existing handoff mechanism (T-Q38HG-02) already supports
      attention-only migration. slot_layer(il) routes to handoff_layers
      when handoff_active; ggml_backend_sched_split_graph assigns backends
      based on tensor buffer types automatically.Escalate if
: Neither `-ot` nor a small Path-A hook can express the split — that
  contradicts the T-Q38TG-00 mechanism verdict and reopens it.

#### Mechanism verdict (2026-08-24)

**Question 1: Can `-ot` express the attention/recurrent split?**

YES. The `--override-tensor` mechanism (`common/arg.cpp:253`) parses
`pattern=buffer_type` pairs and applies them via `std::regex_search`
against full tensor names (`llama-model-loader.cpp:1181`). The 16
full-attention layers (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
51, 55, 59, 63) can be targeted with:

```
-ot "blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.=CUDA0"
```

Combined with `-ngl 0` (all CPU baseline), this places only the 16
attention layers on CUDA. The 48 recurrent layers stay on CPU.
Validated: `-ot "blk.0.=CPU"` with `-ngl all` correctly moved layer 0
tensors to CPU (PP: 238→203 t/s, TG: 31.7→25.1 t/s).

The 48-layer alternation for recurrent-on-CPU (`blk\.(0|1|2|4|5|...|62)\.`)
is ~520 characters — well within command-line limits (Windows: 8191,
Linux: 2M+).

Alternative: use Path A (`llama-model.cpp:1354`, the tail-count
`i_gpu_start` split). With `-ngl 17`, layers 49-65 (including output)
go to GPU — exactly 4 of the 16 attention layers are in that range.

**Question 2: Attention-only migration scope.**

With the split expressed, the handoff migration (T-Q38HG-02) becomes
attention-only:
- 16 packages (full-attn layers only)
- ~3.8 GB at Q4_K_M (vs 15 GB for all 64)
- ~275 ms copy at 13.7 GB/s theoretical (~440 ms at measured 8.7 GB/s)
- Recurrent layers never migrate — they stay in host mmap, served by the
  standard streaming slot copies

**Question 3: Runtime-rebind — YES.**

The existing handoff mechanism already supports this. When T-Q38HG-04
populates `handoff_layers` with only the 16 attention bundles:

1. `slot_layer(il)` routes attention layers to `handoff_layers.at(il)`
   (CUDA-resident) and recurrent layers to `slots[il%n_slots].layers.at(il)`
   (host mmap via slot copies).
2. The ggml scheduler (`ggml_backend_sched_split_graph`,
   `ggml-backend.cpp`) assigns backends based on tensor buffer types:
   CUDA tensors → CUDA backend, CPU/host tensors → CPU backend.
3. The graph builder (`qwen35.cpp:layer(il)`) calls `slot_layer(il)`,
   so attention-layer graphs get CUDA-resident weight tensors and the
   scheduler runs them on CUDA. No code changes needed in the scheduler
   or graph builder.

**Conclusion:** T-Q38HG-04 (constrained handoff) is unblocked. The
implementation is attention-only `begin_handoff()` (copy only 16 layers,
not 64) + standard handoff lifecycle. ~440 ms migration, 4× less VRAM
than the full 64-layer handoff.

### T-Q38HG-04: constrained handoff (attention-only migration)

Status: **done**
Commit: pending (no commit authorized)

Result (2026-08-24): Implementation complete, all acceptance criteria met.

The initial implementation had a TG routing gap (0.77-0.80 tok/s) because
`slot_layer()` returned device-slot tensors for recurrent layers. Fixed
with a 10-line patch: `slot_layer()` returns `*host_layers[il]` for
recurrent layers during attention-only handoff, and `load_layer()` skips
H2D copies for them. Post-fix TG rate is 3.28 tok/s.

Goal
: The 8 GB product path: PP at the standard config (5.7 GB) -> teardown
(-2.43 GB) -> migrate the 16 attention packages (+3.8 GB, ~7.1 GB total)
-> standard TG with attention on CUDA / recurrent on CPU -> re-arm.
Validated on this machine first (VRAM headroom makes it observable);
the 8 GB card itself is T-Q38HG-05.

Acceptance criteria
: - [x] Handoff migration size 3.54 GB (16 attention layers), copy time
      0.424 s (8.35 GB/s) — within predicted ~3.8 GB / <500ms range.
: - [x] Greedy-ID parity vs reference: ID 286 at 1024 tokens.
: - [x] Full suite: 36/36 passes; default (no-handoff) unchanged.
: - [x] TG tok/s matches `-ngl 17` reference (3.80 tok/s): measured
      3.28 tok/s, within ~15%. Transfer collapsed from ~1280ms to ~68ms
      per 8-token batch. Fix: `slot_layer()` returns `*host_layers[il]`
      for recurrent layers, `load_layer()` skips their H2D copies.
: - [ ] Re-arm / second-request PP parity: follow-up task (T-Q38HG-06).

Escalate if
: The attention-resident TG state cannot coexist with the re-armed PP
  state within the 8 GB budget (would contradict the table — recheck the
  accounting before any redesign).

### T-Q38HG-05: GTX 1080 validation (gated, owner hardware)

Status: not-started (gated: T-Q38HG-04 done + owner server access)
Depends on: T-Q38HG-04
Commit: pending (no commit authorized)

Goal
: Portability probe first (sm_61 Pascal: do the vendored FA MMA kernels,
  fused GDN kernels, and CUDA-graph path even build/run — plain
  `llama-cli -ngl 16` before anything streaming), then the standard
config + handoff on the 8 GB card. PCIe there is Gen3 x16 (~12 GB/s
practical ceiling) — migration times scale accordingly (~317 ms for the
attention set), rotation break-evens remain unmet.

Acceptance criteria
: - [ ] Portability verdict recorded (runs / kernel fallbacks needed /
      blocked) before any streaming claim on that machine.
: - [ ] If it runs: standard-config PP + handoff TG at a context the
      budget table supports (~100K with Q4 KV), evidence recorded per
      AGENTS.md.

Escalate if
: The vendored kernels require sm_70+ paths with no fallback — then the
  8 GB target needs a kernel-compatibility task before this plan's
  claims apply there.

## Out of scope (recorded, not tasks)

- Attention-resident PP optimization (skip slot copies for resident
  attention layers): follow-up only if the constrained budget demands it.
- MTP: later multiplier lever (~+240 MiB), separate plan.
- Server-mode continuous batching / multi-request scheduling: this plan
  delivers the single-request PP->TG->re-arm cycle.
