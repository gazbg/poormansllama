# Qwen3.8 Dense Layer Streaming SDD

Status: Draft - scope agreed, implementation pending

## 1. Purpose

This experiment measures real Qwen3.8-27B prompt-processing (PP) and token-generation (TG) behavior when decoder-layer weights are streamed from a host-mapped GGUF into a four-slot CUDA VRAM window.

The purpose is to collect evidence before optimizing for speed. It is not a quality, quantization, MoE-routing, vision, MTP, or production-serving project.

## 2. Fixed Experiment Envelope

| Item | Decision |
| --- | --- |
| Model artifact | `C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf` |
| Model family | Dense Qwen3.8 / `qwen3_5` hybrid model |
| GPU backend | CUDA only |
| Test GPU | NVIDIA GeForce RTX 5090, 24,463 MiB VRAM |
| Host | AMD Ryzen 9 9955HX, 64 GiB RAM, Windows |
| CUDA driver evidence | Driver 610.74, CUDA UMD 13.3 |
| Layer window | Four device-resident decoder-layer packages |
| PP trace | 512 deterministic valid token IDs, u-batch 64 |
| TG trace | 128 greedy tokens immediately after PP |
| MTP | Disabled and out of scope |
| MoE expert prediction | Deferred; Qwen3.8 is dense |
| Source policy | Best-effort cold mmap; no deliberate prewarm |
| Output | JSON Lines telemetry plus one text summary per run |

## 3. Terms

- PP: prompt processing or prefill.
- TG: autoregressive decode after PP.
- Layer package: all immutable tensors required by one logical decoder layer.
- Slot: one reusable CUDA allocation that holds one logical layer package.
- Static weights: token embedding, final norm, and LM head.
- Session state: full-attention KV cache and Gated DeltaNet recurrent state.
- Ready: a layer package has completed its host-to-device copy and its CUDA event has been observed by its consumer stream.

## 4. Scope

### In scope

- An experimental, CUDA-only path in the vendored llama.cpp checkout.
- Qwen3.8 Qwen3.5-family graph execution split at decoder-layer boundaries.
- Four-slot weight streaming for the 64 main decoder layers.
- Persistent per-layer graph cache for PP u-batch 64 and TG batch 1.
- Asynchronous copy scheduling, CUDA-event ordering, cache-slot lifecycle, and telemetry.
- A live PP-to-TG session where the four slots are not reset between phases.
- Raw PP/TG tokens-per-second and per-layer transfer, wait, and compute measurements.

### Out of scope

- Output-quality evaluation, quantization comparison, or changing the Q6_K_XL artifact.
- A fully resident baseline or a speedup claim in v1.
- CPU, Vulkan, multi-GPU, disk-cache-control, production server, or public API work.
- Vision, video, tools, chat-template behavior, MTP, speculative decoding, YaRN, or long-context tuning.
- MoE experts, expert prediction, expert prefetch, and routing telemetry.
- An upstream llama.cpp contribution or pull request.

## 5. Architecture Reality

The current llama.cpp Qwen3.5 graph contains all logical layers in one graph and binds model tensor buffers at load time. Existing static `n_gpu_layers` offload cannot provide this experiment.

The experimental path must execute the model one decoder layer at a time. It must carry the hidden activation through the 64-layer sequence while preserving the normal logical KV and recurrent-state ownership. This is an experimental execution path, not a wrapper around the public libllama API.

The four slots cache weights only. They do not reduce or remap the 64 logical layers of session state.

## 6. Residency Contract

### Persistent in VRAM for the active session

- Token embedding, final norm, and LM head.
- Full-attention KV cache for all applicable logical layers.
- Gated DeltaNet recurrent state for all applicable logical layers.
- Hidden-activation work buffers, graph work buffers, CUDA streams, events, and telemetry buffers.

The default full-attention K/V cache type is F16. The isolated runner also
supports opt-in `--kv-cache q8`, which selects llama.cpp's Q8_0 type for both
K and V before context creation. This does not quantize the Gated DeltaNet
recurrent state, decoder weights, or any slot data.

### Pageable from host mmap to four device slots

- Immutable tensors of the 64 main decoder layers only.
- A slot holds the complete package for one logical layer, including the appropriate full-attention or Gated DeltaNet tensors and FFN tensors.

### Host source contract

- The GGUF remains mmap-backed by the host process.
- The runner does not deliberately pre-touch layer pages before a timed run.
- Windows file-cache state is not controllable by this process. A run is labeled `source_cache_mode=best_effort_cold`, never claimed to be provably cold.
- Static-weight and four-slot-capacity preflight occurs before PP. If allocation cannot fit, the runner exits before inference and reports required and available VRAM. There is no CPU fallback.

### Copy-source modes

- `--copy-source direct` is the default: tensor copies read directly from the
  mmap-backed GGUF source.
- `--copy-source pinned` is an opt-in experiment: each device slot owns one
  CUDA-pinned host buffer sized to its device package allocation. Before each
  package copy, tensors are packed into that slot's staging buffer and copied
  from it asynchronously.
- The slot-reuse consumer-complete event guards staging-buffer reuse as well as
  device-slot reuse. A failed pinned allocation, including CUDA allocator
  fallback to ordinary host memory, aborts initialization; pinned mode never
  silently becomes direct or pageable staging.

## 7. Four-Slot Streaming Policy

The policy is deterministic because dense Qwen3.8 uses layers in the known order `0..63` for every PP microbatch and every TG token.

```text
Before layer L: slots target L, L+1, L+2, L+3 when those layers exist.
While layer L computes: schedule host-to-device copy for L+4.
After the consumer CUDA event completes: its slot may be reused.
At layer 63: the next required logical layer is 0 for the next pass.
```

At the PP-to-TG boundary, slots remain live. TG starts with whatever four logical layers PP left resident; it does not reset or prefill slot 0. This intentionally exposes the real wrap-around behavior.

With only four slots, this is dense layer streaming and transfer overlap, not long-lived cache-hit reuse. A logical layer normally needs another host-to-device copy before its next full pass because all 64 layers are needed between uses.

## 8. Persistent Graph Contract

Graphs are prepared before the timed phase and keyed by:

```text
(execution_mode, logical_layer, device_slot)

execution_mode: pp_u64 | tg_u1
logical_layer:  0..63
device_slot:    0..3
```

The initial design therefore permits up to 512 persistent graph entries. Each entry has stable CUDA addresses for its slot and logical-layer state. The implementation may deduplicate equivalent entries only if it preserves stable-address and telemetry semantics.

CUDA Graph capture is not required in v1. Graph-build/setup time is recorded separately and excluded from PP/TG timed sections once persistent graphs are ready.

## 9. Execution Contract

### PP

1. Generate the deterministic valid input IDs `0..511` without chat-template processing.
2. Execute eight u-batches of 64 tokens.
3. For each u-batch, run logical decoder layers `0..63` through the slot scheduler.
4. Retain model session state and the four live slots after the final PP layer.

### TG

1. Start immediately after PP in the same context and sequence.
2. Use greedy sampling for 128 decoded tokens.
3. Execute logical decoder layers `0..63` for each decoded token through the same four-slot policy.
4. Record emitted token IDs but do not score response quality.

### Safety and failure behavior

- A slot has a monotonically increasing generation number and an assigned logical-layer ID.
- A graph may execute only after the matching logical-layer package is ready in its assigned generation.
- A slot cannot be overwritten until all CUDA consumers of that generation have completed.
- A missing package, stale generation, failed CUDA event, or insufficient allocation aborts the run with a structured error event. It must never silently run a different layer's data.
- Debug validation may inspect package metadata and copy boundaries. No fully resident output-comparison baseline is required in v1.

## 10. Telemetry Contract

The runner writes JSON Lines. Each event includes `run_id`, monotonic timestamp, phase, logical layer, slot, slot generation, and sequence or u-batch identity where applicable.

Required event types and fields:

| Event | Required fields |
| --- | --- |
| `run_start` | model path, GPU, build revision, source cache mode, `copy_source`, `kv_cache`, slot count, static VRAM allocation |
| `slot_request` | logical layer, slot, requested bytes, phase |
| `copy_submit` | logical layer, slot, generation, bytes |
| `copy_complete` | logical layer, slot, generation, `transfer_us`, `host_submit_us`, `ready_latency_us` |
| `layer_ready_wait` | logical layer, slot, generation, `availability_wait_us` |
| `layer_compute` | logical layer, slot, generation, `compute_us` |
| `slot_evict` | prior logical layer, slot, generation, reason |
| `phase_summary` | phase wall time, tokens, tokens/s, transfer total, wait total, compute total |
| `run_end` | exit status, peak VRAM, generated token count, error if any |

The text summary reports at minimum:

- PP tokens/s for 512 tokens at u-batch 64.
- TG tokens/s for 128 greedy tokens.
- Per-logical-layer count, total, mean, p50, p95, and maximum for compute, transfer, and availability wait.
- Total bytes transferred, slot evictions, ready-at-consumption count, and wait-at-consumption count.
- VRAM preflight result and peak observed usage.

`transfer_us` remains the legacy submit-to-ready interval for compatibility.
It equals `host_submit_us + ready_latency_us`. `host_submit_us` measures the
host interval spent issuing the package's tensor-copy operations. For the
pageable mmap source it can include driver staging. `ready_latency_us` begins
when host submission finishes and ends when the ready event is observed; it
includes copy-stream queueing and device work, so it is not an isolated DMA
duration. `availability_wait_us` remains the host block at consumption.

No metric is a pass/fail performance threshold in v1. The report must not claim a speedup without a later comparison task.

## 11. Implementation Boundary

The fastest viable implementation is an explicitly experimental CUDA patch in `third_party/llama.cpp`, behind a dedicated CMake option and isolated runner target. Likely ownership points include the Qwen3.5 model graph, model tensor loading/buffer placement, llama context execution, recurrent memory ownership, and CUDA copy/event plumbing.

The implementation must not alter default llama.cpp execution when the experimental option is disabled.

## 12. Risks and Known Limits

- Splitting the all-layer graph changes the execution architecture and is the main feasibility risk.
- PP and TG state semantics must remain logically indexed by all model layers; slot indices apply only to weights.
- Best-effort cold mmap includes uncontrolled Windows file-cache behavior and may include storage faults.
- Four slots give no meaningful full-pass weight reuse for a dense model. The result primarily measures overlap and stalls.
- Persistent graph storage may consume substantial host and VRAM bookkeeping memory; preflight must measure it.
- No fully resident baseline and no quality gate means v1 can expose transfer behavior but cannot establish equivalence or speedup.

## 13. Layer-Major PP Experiment (opt-in, evidence only)

The batch-major schedule of section 9 is the normative default. A separately
measured **layer-major** PP schedule is an additional opt-in experiment,
enabled only with `--pp-schedule layer-major`. The batch-major contract is
unchanged and remains the default; the two schedules never mix within a run.

### Schedule

```text
embed chunks 0..C-1 into activation slab A
for logical layer L in the executed layer set:
    retain/package-load L once under the four-slot policy
    for chunk C in 0..C-1:
        execute L on A[C] and write B[C]
swap A and B
```

### Activation residency

- The layer-major PP mode owns **two stable full-prompt F32 activation
  slabs** `A` and `B`, each `n_embd x pp_tokens`. For the Qwen3.8-27B
  20,480-token envelope they require
  `2 * 5120 * 20480 * 4 = 838,860,800` bytes (800 MiB), excluding graphs,
  masks, KV, recurrent state, static weights, and slot allocations. The
  arithmetic is an allocation size, not a VRAM-fit claim: preflight measures
  the complete allocation and reports required vs available VRAM, aborting
  before inference when it cannot fit.
- Each PP chunk sees **one non-aliasing slice** of `A` and one of `B`, bound
  at graph-build time as a stable tensor allocated from the slab buffer at a
  fixed per-chunk offset (the same `ggml_backend_tensor_alloc` API the
  batch-major act pairs use, only with a fixed offset instead of a running
  cursor). A persistent graph is never retargeted by assigning a different
  `ggml_tensor::data` address at execution time. A `ggml_view` over the slab
  is rejected: the scheduler's galloc calls `ggml_backend_view_init` on a
  view whose source has a backend buffer, which allocates a *separate*
  buffer for the view and defeats the stable-slab semantics.
- The selected binding mechanism is **per-chunk prebuilt graph sets**: the
  per-chunk activation-slice tensors are distinct tensor objects created at
  prebuild and bound to fixed slab offsets, so a graph executes only against
  the slice it was built for. Chunk 0 owns the fresh-state set (rs_z = 0);
  every later chunk owns its own continuing-state set (rs_z = -1), because
  each chunk's act slices must be distinct non-aliasing tensors. The set
  count therefore grows with the chunk count (not the 64-layer fan-out), and
  the per-set footprint (activation slice bytes, io-context bytes, per-set
  input/mask bytes) is measured by the two-chunk probe and recorded in the
  current-state record.
- The rejected alternative is an explicit fixed-stage device-to-device copy
  into one shared act pair per chunk compute: it keeps a single set but adds
  one full-slab D2D copy per layer per chunk and mutates a shared tensor that
  every graph of the set reads. For the 2-chunk probe the prebuilt-view
  mechanism is selected: zero extra D2D copies and no shared mutable act.

### Causal order and slot ownership

- Each executed logical layer processes its chunks in **ascending
  token-position order**, so the logical KV cache and Gated DeltaNet
  recurrent state of that layer advance causally. No later chunk may be
  pre-applied in a way that an earlier attention/recurrent operation of the
  same logical layer observes future state.
- A layer package is copied **once** per layer pass and consumed by all of
  the layer's chunks under one slot generation. The slot may not be reused
  until the layer's final chunk compute completes. Logical state (KV and
  recurrent) is owned by the logical layer, never by a device slot; a slot
  index is never an address for session state.
- The PP-to-TG boundary keeps the same session-state and slot-live rules as
  section 7 once the layer-major PP pass is complete.

### Telemetry contract (additions)

- `run_start` carries `schedule` (`batch-major` or `layer-major`) and
  `act_slab_bytes` (0 for batch-major), plus `copy_source` (`direct` or
  `pinned`).
- `layer_compute` carries a `chunk` field: the ubatch index of the
  full-prompt batch for the batch-major schedule, and the chunk identity
  (0..C-1) for the layer-major schedule.
- In layer-major mode the host validator requires, per executed logical
  layer, monotonically increasing chunk identity (0, 1, 2, ...) across the
  PP pass, and every `layer_compute` for a layer shares the one generation
  its package was copied in.
- Eligible Qwen3.8 layer-major PP attention graphs may set explicit implicit
  causal metadata on `GGML_OP_FLASH_ATTN_EXT` instead of allocating a dense
  F16 mask. Eligibility requires causal FlashAttention, one contiguous
  sequence beginning at scalar position zero, no SWA or ALiBi, and a
  CUDA MMA-compatible shape. The operation applies
  `k < n_kv_valid && k <= q_pos_base + q`; all ineligible graphs and TG retain
  dense masks.

The layer-major result is evidence. It is not a quality, equivalence, or
speedup claim, and it does not replace the batch-major normative trace.

## 14. Deferred MoE Extension

After dense streaming evidence exists, a Qwen3.5 MoE model can reuse the slot manager with keys `(logical_layer, expert_id)`. Dense Qwen3.8 establishes fixed-order transfer scheduling. MoE would add predicted expert selection, prefetch hit/miss metrics, and synchronous actual-expert fallback. That work is explicitly excluded from this SDD.
