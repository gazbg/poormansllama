# Qwen3.8 FA Efficiency Plan (close the 704.6 -> ceiling gap)

Status: T-Q38FA-02 done, T-Q38FA-03 done, T-Q38FA-04 done (2026-08-24);
T-Q38FA-05/06 deprioritized (kernels proven fine by the resident control).
Open escalation: q8-KV layer-major/batch-major greedy-ID divergence
(pre-existing, recorded in currentstate.md).

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
Predecessor: [FA window plan](qwen38-fa-window-plan.md) (done: 506.2 -> 704.6
tok/s Q6_K_XL 100K, +39.2%). Environment facts:
[docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md). Observed state:
[docs/currentstate.md](../currentstate.md).

## Standing numbers (evidence, 2026-08-24)

Post-FA-window Q6_K_XL 100K ceiling decomposition (from the FA-window trace
`q38ls-fa-100k.jsonl`; wall 141.92 s = 704.6 tok/s):

| Component | Time (s) | % of wall |
| --- | --- | --- |
| Wall clock | 141.92 | 100% |
| Total compute (sum of per-layer compute_us) | 127.38 | 89.8% |
| -- FA (QK^T + softmax + AV), 16 attn layers x 196 chunks | 57.4 | 40.4% |
| -- Attn-layer matmuls (QKV/out proj + FFN) | 14.6 | 10.3% |
| -- Recurrent layers (fused GDN + FFN), 48 layers x 196 chunks | 55.5 | 39.1% |
| Residual (wall - compute): patch 5.37 + other 9.17 | 14.54 | 10.2% |

Asymptotic ceiling if FA cost went to zero: (127.38 - 57.4) + 14.54 =
84.5 s -> 1,183 tok/s. The recurrent floor (55.5 s) is the wall after that.

## Roofline correction (recorded, supersedes the bandwidth claim)

The FA cost is NOT bandwidth-bound at current sizes:
- FA work (avg ~50K valid KV per chunk, 16 layers, 196 chunks) ~2 PFLOP
  -> 57.4 s = **~34 TFLOPS effective (~16% of RTX 5090 peak)**.
- Valid-window K/V reads (Q8) ~320 GB total -> **~5.6 GB/s effective**,
  i.e. <1% of HBM bandwidth.
- Same-run GEMMs for comparison: attn matmuls ~75 TFLOPS effective,
  recurrent floor ~70 TFLOPS effective (~35% of peak, normal for MMQ).

Conclusion: halving K/V bytes (Q4 KV) cannot halve a cost that is not
bandwidth-bound. Q4 KV remains a VRAM lever for the 8 GB target
(T-Q38PW-03), never a PP-speed lever. The anomaly to explain is why the FA
kernel runs at half the efficiency of the GEMMs in the same graph.

## Dependency order

```text
T-Q38FA-02 (resident control) --+
                                +--> T-Q38FA-05 (f16 KV A/B, gated)
T-Q38FA-03 (fused-op audit) ----+--> T-Q38FA-06 (ncu profile, gated)
T-Q38FA-04 (overhead reduction)     independent, run any time
```

### T-Q38FA-02: fully-resident control baseline (the decisive split)

Status: ready
Depends on: none
Commit: pending (no commit authorized)

Goal
: Measure fully-resident batch-major PP at 100K on this machine (no
  streaming, no slots, no graphs-of-ours) for both models. This splits the
  whole problem in half: if the resident run also shows FA at ~34 TFLOPS
  effective, the inefficiency is vendored-kernel-wide and this plan stops
  at a recorded verdict (kernel tuning is out of scope for the streaming
  experiment). If the resident run is materially faster per FA FLOP, the
  captured-graph path is implicated and T-Q38FA-05/06 open.

Scope
: Runs only. `build/llama-cuda/bin/Release/llama-bench.exe` (exists,
  verified 2026-08-24). No code changes.

Non-goals
: Any code change, TG measurement, quality claims.

Steps
: 1. Q6_K_XL: under `tools/with-gpu-lock.ps1`:
     `llama-bench.exe -m <Q6_K_XL> -ngl 99 -p 100000 -n 0 -fa on
      -ctk q8_0 -ctv q8_0 -c 100096 -t 16` (match our KV dtype/context;
      add `-b 8192 -ub 512` if needed to mirror ubatch 512). Record pp
      tok/s. If 100K exceeds llama-bench's practical limits on this
      build, fall back to `-p 65536` and `-p 32768` points and record
      the tok/s-vs-context curve instead.
: 2. Same for Q4_K_M.
: 3. Compute FA-attributable efficiency for the control: same FLOP model
     as the roofline correction above (valid-window causal triangle),
     derive effective TFLOPS for the attention share. Compare against the
     streaming 34 TFLOPS figure.
: 4. Record both raw numbers and the verdict in `docs/currentstate.md`.

Acceptance criteria
: - [ ] Resident pp tok/s recorded for both models (or the context curve),
      with command, free-VRAM start, exit status.
: - [ ] Effective-FA-TFLOPS comparison written: resident vs streaming.
: - [ ] Verdict recorded: "kernel-wide" (stop, document) vs
      "streaming-path-specific" (open T-Q38FA-05/06).

Required validation
: None beyond the measurements; every run records command/model/VRAM/exit
  per AGENTS.md. GPU runs under the mutex.

Escalate if
: llama-bench cannot allocate the 100K context (then use the curve points
  and say so); or the resident number is not reproducible within +/-5%
  across two runs (record both, take the better, note the variance).

### T-Q38FA-03: fused-op audit of captured streaming graphs

Status: ready
Depends on: none
Commit: pending (no commit authorized)

Goal
: Verify the fused Gated DeltaNet / Lightning Indexer / DSv4-HC ops are
  actually present in the captured per-layer streaming graphs, not just in
  the scheduler reserve log. `resolve_fused_ops` runs at scheduler reserve
  time; the streaming graphs are built through a separate path and could
  in principle capture unfused elementwise chains. If unfused, part of the
  55.5 s recurrent floor is ours to fix cheaply.

Scope
: Read-only inspection + one cheap instrumented run. No executor changes.

Steps
: 1. Source inspection: trace `build_streaming_graphs()` (qwen35.cpp) ->
     confirm whether the fused-op resolution applies to the per-piece
     graphs (look for the same hook `llama_context::process_ubatch` uses,
     e.g. `resolve_fused_ops` / `ggml_fuse_*` calls and whether they run
     for streaming builds). Record file:line evidence.
: 2. Cheap runtime check: 1024-token layer-major run with existing
     telemetry; compare per-layer recurrent `compute_us` against the
     fused-op expectation (fused GDN autoregressive at 512 tokens should
     be GEMM-dominated; an unfused chain would show markedly higher time
     per recurrent layer). Cross-check against the 100K trace's recurrent
     per-layer averages (~5.4 ms/chunk).
: 3. Record verdict in `docs/currentstate.md`: fused-confirmed or
     unfused-found (with the op list from the captured graph).

Acceptance criteria
: - [ ] Written source-level verdict with file:line evidence.
: - [ ] If unfused: the missing hook named and a fix task proposed (new
      task, not implemented here). If fused: closed with evidence.

Required validation
: Host-only inspection needs no GPU; the 1024-token run runs under the
  mutex with `--check-telemetry` exit 0.

Escalate if
: The streaming graph build path has no fused-op hook at all and adding
  one requires changes outside the isolated experimental path.

### T-Q38FA-04: timed-region overhead reduction (the free ~10%)

Status: ready
Depends on: none
Commit: pending (no commit authorized)

Goal
: Reduce the 14.54 s residual (wall - compute) at 100K. Three known
  components: `total_patch_us` 5.37 s (3.8%), graph-related 3.82 s
  (attribution unverified, see step 1), sync/copy-tail ~5.35 s. Target:
  residual below 8 s at 100K Q6_K_XL (+4-5% tok/s), no correctness
  change.

Scope
: `src/llama-layer-streaming.cpp` (`prepare_chunk` walk, prebuild timing),
  telemetry only. No kernel changes, no schedule changes.

Steps
: 1. **Verify the 3.82 s attribution first.** Check the `graph_footprint`
     event timestamp against the PP wall window in
     `q38ls-fa-100k.jsonl` (run_start.t_us vs phase_summary wall). If
     prebuild happens before the PP timer (as designed), the 3.82 s is
     NOT in the wall and the pasted ceiling analysis overstates overhead;
     record the corrected decomposition and re-scope this task to the
     real residual. If it IS inside the wall (pw-mode prebuild moved
     after the timer at some point), move it out or amortize it.
: 2. Patch walk: the dominant term is the O(chunks x layers) causal-patch
     walk over FLASH_ATTN_EXT nodes plus the input-snapshot memcpy.
     Measure the split (add per-walk-stage timing to `patch_us` if not
     already separable), then optimize the larger half: cache the FA-node
     pointer list per graph at prebuild (avoid re-walking), and skip the
     causal patch when `q_pos_base`/`n_kv_valid` are unchanged vs the
     previous chunk of the same set (they always change — verify before
     assuming; if they do, keep the walk but drop redundant work).
: 3. Sync/copy tail: measure where the 5.35 s goes (host sync between
     chunk launches? trailing copy completion?). If it is host-side
     launch serialization, batch the per-chunk prepare+launch.
: 4. Rerun the 100K Q6_K_XL configuration; compare residual.

Acceptance criteria
: - [ ] Step-1 attribution verdict recorded (corrected decomposition if
      the 3.82 s is outside the wall).
: - [ ] Residual (wall - total_compute) at 100K Q6_K_XL <= 8 s, or the
      task escalates with the measured split showing the remainder is not
      reducible host-side.
: - [ ] Greedy-ID parity (1024 tokens, ID 286) unchanged; `--check-telemetry`
      OK; `graph_builds == 0`.
: - [ ] `tools/run-tests.ps1 -Suite all` passes.

Required validation
: Build -> exit 0; 100K Q6_K_XL paired against `q38ls-fa-100k.jsonl`
  (already on disk); parity pair; full suite. GPU runs under the mutex.

Escalate if
: The residual turns out to be device-side (kernel gaps visible only in a
  profiler) — hand that finding to T-Q38FA-06 instead of guessing.

### T-Q38FA-05: f16-KV A/B (gated on T-Q38FA-02 verdict)

Status: not-started (gated: opens only if T-Q38FA-02 says
  "streaming-path-specific" AND the q8->f16 conversion path is implicated)
Depends on: T-Q38FA-02
Commit: pending (no commit authorized)

Goal
: Test the hypothesis that q8 KV triggers the `need_f16_K`/`need_f16_V`
  conversion in `fattn-common.cuh` inside every captured graph replay —
  a per-chunk re-conversion of the valid window that would explain FA
  running at half GEMM efficiency. One-flag A/B: 100K with `--kv-cache
  f16` (+3.3 GB VRAM, fits on this machine) vs the q8 baseline.

Steps
: 1. Source check first (cheap): does the streaming graph's FA node take
     q8 K/V sources with `need_f16_K` true? (fattn-common.cuh:1022-1048,
     f16_extra allocation). Record file:line.
: 2. If yes: 100K Q6_K_XL f16-KV run, same flags as the q8 baseline.
     Compare tok/s and per-layer FA compute.
: 3. Verdict: conversion tax quantified or ruled out. Record in
     `docs/currentstate.md`. If the tax is real, the fix is to keep the
     FA inputs f16-native or cache the conversion — new task, not here.

Acceptance criteria
: - [ ] Source-level answer to "does our graph hit need_f16_K" with
      file:line.
: - [ ] If it does: paired 100K numbers (q8 vs f16 KV) with the
      per-layer FA compute delta.
: - [ ] Verdict recorded; Q4-KV-as-speed-lever claim formally closed
      either way (it is a VRAM lever only).

Escalate if
: f16 KV does not fit the 100K context on the current free VRAM (then run
  at 65K and scale the conclusion, saying so).

### T-Q38FA-06: ncu kernel profile (gated, last resort)

Status: not-started (gated: opens only if T-Q38FA-02 says
  "streaming-path-specific" and T-Q38FA-03/05 did not explain it)
Depends on: T-Q38FA-02, T-Q38FA-03, T-Q38FA-05
Commit: pending (no commit authorized)

Goal
: Nsight Compute profile of one representative attention-layer graph
  replay at a mid-context chunk (e.g. chunk 100 of a 100K run): kernel
  occupancy, achieved TFLOPS, conversion-kernel presence, graph-launch
  gaps. Produces the kernel-level explanation or confirms the ceiling is
  architectural. Instrumentation task; no code changes beyond enabling
  profiling hooks.

Steps
: 1. Check ncu availability (`ncu --version`; CUDA 13.2 toolkit per env
     guide). If absent, record and close as environment-blocked.
: 2. Profile a bounded region: 1024-token run is too small to be
     representative; use a 20K run (40 chunks) and profile chunk 20's
     layer-3 graph replay. Record per-kernel durations.
: 3. Write the finding: which kernels dominate, their achieved occupancy
     and TFLOPS, whether conversion kernels appear, whether graph replay
     shows inter-kernel gaps.

Acceptance criteria
: - [ ] Per-kernel table for one representative chunk (kernel name,
      duration, achieved occupancy) recorded.
: - [ ] Written explanation of the 34-vs-70 TFLOPS FA/GEMM split, or an
      explicit "architectural ceiling" verdict.

Escalate if
: Profiling requires disabling CUDA graphs (then the profile measures a
  different code path — record that limitation and interpret accordingly).

## Out of scope (recorded, not tasks)

- Recurrent-floor GEMM tuning (Marlin/Machete/NVFP4): kernel work outside
  the streaming experiment; the 55.5 s floor runs at ~70 TFLOPS effective,
  near the MMQ roofline. Revisit only as a separate owner-approved plan.
- Q4 KV as a speed lever: closed by the roofline correction (VRAM lever
  only, T-Q38PW-03 gate).
- MTP: later lever, separate plan.
