# Qwen3.8 FA Window Plan (kill the padded-KV attention scan)

Status: done (completed 2026-08-24; all acceptance criteria satisfied, Q4_K_M escalation accepted by owner)

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
Predecessor: [per-width graphs plan](qwen38-perwidth-graphs-plan.md) (done).
Environment facts: [docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md).
Observed state: [docs/currentstate.md](../currentstate.md).

## Diagnosis (evidence, 2026-08-23)

The 100K Q4_K_M layer-major run (`q38ls-q4km-pp100k.jsonl`) measures compute
as 178.7 s of the 190.2 s wall (94%); copies are fully hidden (wait ≈ 0).
So the compute path itself — not streaming — is the gap vs the ~1000 tok/s
fully-resident baseline.

Root cause, confirmed two ways:

1. Source: the ggml FA kernel's KV-scan early-exit (`KV_max` -> `kb0_stop`
   bound in `fattn-mma-f16.cuh`) is only computed when a real mask tensor
   exists (`fattn-common.cuh:1094`, `if (mask && ...)`). The streaming path
   uses implicit causal metadata (`use_implicit_causal_mask = true`, no mask
   tensor), so `KV_max` is never built and every chunk's attention iterates
   over the full padded KV length (`iter_k = ceil(K->ne[1] / nbatch_fa)`,
   `K->ne[1]` = padded `n_ctx`). `n_kv_valid` is host-side-assertion-only
   (`fattn-common.cuh:1197`) and never reaches the kernel.
2. Telemetry: per-chunk compute for full-attn layer 3 is FLAT across chunk
   index (42.5 ms at chunk 0 with 512 valid KV, 39.7 ms at chunk 100 with
   ~51K valid KV, 42.0 ms at chunk 190 with ~98K valid KV; tail chunk 195
   scales exactly as 160/512). Attention FLOPs grow linearly with position
   and dominate the layer; a flat profile is the signature of scanning the
   full padded KV every chunk.

Attribution: 16 attention layers x 7.9 s = 126 s of the 179 s compute (71%).
With a correct window the average attention work halves -> ~63 s -> total
compute ~116 s -> ~860 tok/s ceiling at 100K (up from 525.7).

Secondary candidate folded in: if q8 KV takes the `need_f16_K`/`need_f16_V`
conversion path in `fattn-common.cuh`, the conversion currently covers the
full padded KV at every launch; shrinking the K/V views bounds it to the
valid window for free. Verify during implementation and record which case
applies.

### Ruled-out alternatives (2026-08-23 triage)

The ~1000 tok/s resident baseline and the streaming path share the same
vendored ggml backend, kernels, and model, so shared-kernel efficiency
(GEMM kernel family, GDN fusion) is a constant factor, not the gap:

- Quantized GEMM kernel family (Marlin/Machete/NVFP4): identical MMQ both
  paths. Not the gap. (Absolute-speed lever for a future task only.)
- CUDA graphs vs eager: launch overhead single-digit % at 40 ms kernels.
- Layer-major activation reuse: extra slab HBM re-reads ~263 GB total
  (~4 GB per layer transition x 64) ≈ 33 ms at 5090 HBM bandwidth.
  Negligible vs 179 s.
- GDN chunked kernels: same fused kernels both paths; recurrent layers run
  ~5.4 ms/chunk already.
- PCIe copy interference: 15 GB copied over 190 s ≈ 80 MB/s average.
  Negligible vs HBM bandwidth.

## Dependency order

```text
T-Q38FA-01 (this plan, single task) -> record in currentstate.md
T-Q38PERF-06 (stream overlap) remains deferred, unchanged.
```

### T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path

Status: done
Depends on: per-width graphs plan (done)
Commit: 57c3b4c (parent-repo docs only; submodule left dirty per repo convention)

Goal
: Bound each layer-major PP chunk's flash-attention KV scan to the valid
  window `[0, token_start + n_tokens)` instead of the full padded KV, using
  the existing per-chunk patch mechanism. Expected: ~halved attention work,
  ~+30-60% PP tok/s at 100K. This is a performance task with a hard
  correctness gate; it makes no quality claim.

Scope
: `src/llama-layer-streaming.{h,cpp}` (`prepare_chunk` patch walk),
  `src/models/qwen35.cpp` only if K/V view tensor discovery requires it,
  `tools/layer-streaming/main.cpp` only if a new telemetry field is added.
  No vendored ggml kernel changes (escalate if one becomes necessary).

Non-goals
: TG path changes (TG legitimately scans all past KV), batch-major changes,
  dense-mask mode changes, KV dtype changes, slot lifecycle changes, CUDA
  graph capture changes, quality claims.

Design and failure contract
: In pw mode the per-chunk `prepare_chunk` lambda already re-points act slab
  data pointers, restores input snapshots, and patches the causal window op
  params (`q_pos_base`, `n_kv_valid`) on every layer graph's FLASH_ATTN_EXT
  node. Extend it: for each attention layer graph, set the FA node's K and V
  source tensors' `ne[1]` to `chunk.token_start + chunk.n_tokens` (and fix
  `nb` strides only if they depend on `ne[1]` — check; K/V cache rows are
  contiguous per cell so strides are ne[0]-derived and unchanged).
  Cells `[0, n_kv_valid)` are exactly the written ones under layer-major
  order; the causal frontier inside the window is already handled by the
  baked `q_pos_base` op params. The host-side FA assertion
  `n_kv_valid == q_pos_base + Q->ne[1] && n_kv_valid <= K->ne[1]`
  still holds with equality. The captured grid may be oversized for small
  chunks; the in-kernel `kbc` work partitioning tolerates empty tiles —
  verify, don't assume.
  ggml's CUDA backend diffs node shapes on every graph launch
  (`ggml_cuda_graph_update_executable`) and patches kernel params via
  `cudaGraphExecUpdate`, falling back to re-instantiate. A per-chunk
  re-instantiation would show up as a wall/`patch_us` blowup — that is a
  loud failure of this mechanism, not a silent success.
  Correctness gate: first greedy ID parity between batch-major and
  layer-major at 1024 tokens (existing harness, ID 286 reference on
  Q6_K_XL) plus `--check-telemetry`. A wrong window must fail parity or the
  host validator — never silently attend to unwritten cells.

Implementation steps
: 1. Locate the FA node K/V source tensors per attention layer graph (same
     walk the causal patch already performs).
: 2. In `prepare_chunk`, patch `ne[1]` to the chunk's `n_kv_valid` on all
     sets (fresh, continuing, tail) before launch; leave the TG set
     untouched.
: 3. Check the `need_f16_K`/`need_f16_V` question (diagnosis above) and
     record the answer in the task report.
: 4. Add the K/V window to `patch_us` accounting (it is part of the same
     walk; no new field needed unless the walk splits).
: 5. Run the validation battery below.

Acceptance criteria
: - [ ] Correctness: 1024-token layer-major run produces the same first
      greedy ID as batch-major (Q6_K_XL reference: 286) and
      `--check-telemetry` exits 0.
: - [ ] Falsifiable signature: per-chunk `compute_us` for a full-attn layer
      (e.g. layer 3) is NO LONGER flat — chunk 0 must drop materially
      (prediction: 42.5 ms -> ~10-15 ms, GEMM-dominated) and grow toward
      the old flat value by chunk 190. If the profile stays flat, the patch
      did not land and the task fails.
: - [ ] Performance: 100K Q4_K_M run (same command as
      `q38ls-q4km-pp100k.jsonl`) reaches >= 700 tok/s (+33% over 525.7)
      or escalates with the measured numbers and the per-layer compute
      table. No silent acceptance of a miss.
: - [ ] `graph_builds == 0` in the timed region; `patch_us` stays < 5% of
      wall (no per-chunk re-instantiation).
: - [ ] 100K Q6_K_XL regression run within +/-5% of the better of the old
      baseline (506.22 tok/s) — i.e. the fix must not regress the model it
      already worked for; improvement expected.
: - [ ] `tools/run-tests.ps1 -Suite all` passes; default-OFF build compiles.

Required validation
: - Build: `cmake --build build/llama-cuda --config Release --target
    llama-layer-streaming --parallel 16` -> exit 0.
: - GPU runs under `tools/with-gpu-lock.ps1`, each recorded with command,
    model identity, free-VRAM start, exit status, trace path.
: - Parity pair (Q6_K_XL, 1024 tokens, `--tg-tokens 1 --greedy`):
    batch-major vs layer-major first greedy ID.
: - 100K Q4_K_M paired against `q38ls-q4km-pp100k.jsonl` (already on disk).
: - 100K Q6_K_XL paired against `q38ls-pw01-pp100k.jsonl` (already on disk).
: - Host validator on every produced trace exits 0.
: - Full suite: `tools/run-tests.ps1 -Suite all`.

State update
: Record the before/after table (tok/s, per-layer compute profile, patch_us,
  peak VRAM), the `need_f16_K` answer, and residual risks in
  `docs/currentstate.md`; mark this plan done.

Escalate before implementation if
: The K/V view tensors are not patchable per graph (shared across graphs in a
  way that breaks per-set patching), or `cudaGraphExecUpdate` cannot absorb
  the shape change and re-instantiates per chunk (measured), or parity fails
  (would mean the window exposes unwritten KV cells — stop and re-derive).
