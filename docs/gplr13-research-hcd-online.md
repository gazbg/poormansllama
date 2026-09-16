
## 8. Documented online research pass on H-C and H-D (2026-09-15, no GPU runs, no code changes)

Per AGENTS.md "Problems and Online Help": the two mechanisms H-C and H-D hinge
on (host-side event synchronization cost, and pageable H2D staging) are CUDA
platform behavior, not local-source facts, so an authoritative-source pass was
run before any repair is improvised. What was consulted, and what it changed:

### 8.1 Sources consulted

- CUDA Handbook, ch. 6.3 "CUDA Events: CPU/GPU Synchronization"
  (cudahandbook.com/book/ch6/cuda-events-cpugpu-synchronization) -- blocking
  vs. polling events; pageable-memcpy staging pattern with two pinned buffers
  + events; measured cost of moving the sync inside the copy loop.
- CUDA Runtime API "API synchronization behavior" notes as discussed on the
  NVIDIA developer forums: "Synchronization of cudaMemcpyAsync for pageable
  memory" (forums.developer.nvidia.com/t/191014) and "Clarification on
  cudaMemcpy synchronization behavior with pageable memory and non-blocking
  streams" (forums.developer.nvidia.com/t/356640, Jan 2026) -- pageable H2D is
  staged through pinned memory "for DMA transfer to device memory" and the API
  returns once the staging copy completes, not the final DMA.
- NVIDIA forum "Is WDDM causing this?" (forums.developer.nvidia.com/t/29903,
  NVIDIA staff reply) -- the WDDM user-mode software queue batches submissions
  into ~20-command buffers; `cudaEventQuery(0)` is the documented flush.
- llama.cpp upstream precedent: NVIDIA blog "Optimizing llama.cpp AI Inference
  with CUDA Graphs" (Aug 2024) and ggml-org/llama.cpp discussion #17621
  (Nov 2025) -- launch/sync overhead is a recognized TG cost class upstream.

### 8.2 H-C (retention hit-path ready-latency ~198 vs ~101 us) -- what the sources imply

The measured quantity is `ggml_backend_event_synchronize` wall time measured on
the host after enqueue (llama-layer-streaming.cpp:1275 `wait_slot_ready`), i.e.
it includes the API-side sync entry cost, not just device wait. Three
platform mechanisms are consistent with the hit path's larger reading, none of
them the naive "transfer is slower":

1. **WDDM software-queue batching (the RTX 5090 runs under WDDM,
   docs/AGENT_ENVIRONMENT.md).** The CUDA driver on WDDM accumulates submitted
   commands in a user-mode software queue and flushes to the kernel-mode queue
   in ~20-command batches; a synchronize call is one of the things that forces
   a submit. The row-set HIT path enqueues zero copies between the previous
   consumer event and its `wait_slot_ready`, so its synchronize lands on a
   near-empty software queue and pays the flush + completion-notification
   round trip with nothing to overlap it; the MISS path enqueues the whole
   copy burst first, so the synchronize batches behind the just-submitted
   command buffer. This predicts the direction of the measured inversion and
   its roughly-constant ~100 us scale. Falsifiable locally: if H-C's
   bookkeeping story were the sole cause, the hit-path wait should scale with
   K (retained-set verification work grows with K); the WDDM story predicts
   K-independence. The existing data shows ~198 us at 4k and ~185 us at 50k,
   both K=4 -- no K-variation data exists yet; a K=1 vs K=8 A/B on the same
   stream is the cheapest disambiguation and needs no code change.
2. **Sync-mode default.** ggml calls `cudaSetDeviceFlags(cudaDeviceScheduleSpin)`
   only for cc 12.1 iGPUs (ggml-cuda.cu:369); a cc 12.0 dGPU keeps the runtime
   default `Auto`, which resolves to spin when cores are free -- so syncs are
   already spin-waits and the blocking-sync penalty (~100 us class) does NOT
   apply; the residual is submission/flush latency, not sleep/wake latency.
   This NARROWS H-C: the wait is WDDM submission mechanics, not event-mode
   selection, and toggling `cudaEventBlockingSync` would likely make it worse,
   not better.
3. **Per-call event cost is real but small.** The Handbook's measurements put
   pure event synchronize overhead in the low microseconds on TCC/Linux; the
   ~100 us WDDM-class delta is 1-2 orders above that, consistent with
   submission mechanics dominating API mechanics here.

**Consequence for H-C's proposed repair.** The H-C text predicted "per-fetch
slot event synchronization on the K>0 cache"; the WDDM reading keeps the
per-fetch-sync attribution but changes the mechanism from "bookkeeping +
retained-set verification" to "software-queue flush on an empty submission".
The cheapest reversible probe (no vendored change): batch the hit path's wait
-- when a layer-window needs no fetch, defer the `wait_slot_ready` synchronize
until the first consumer op of the slot is enqueued on the compute stream (the
event-wait already orders the data; the host block exists for telemetry
accounting). If the ~100 us delta moves from wait into host_submit/ready on
the same traces, the WDDM story is confirmed and the fix is queue-shape, not
bookkeeping. H-C's sub-linearity-in-K prediction is unchanged and still
testable in the same A/B.

### 8.3 H-D (PP full-bank staging 3.34 GB/s vs TG reactive 9.16 GB/s) -- what the sources imply

The row-set fetch source is the model's mmap'd host buffer (`src_t->data +
slab_bytes` offsets, the load_row_set 4a loop) -- **pageable** memory. Two
sourced facts combine:

1. **Pageable H2D is a staged copy.** The driver copies pageable source into
   an internal pinned staging buffer "for DMA transfer to device memory" and
   the async call returns after the staging copy, not the DMA. So every
   `ggml_backend_tensor_set_async` on a pageable source performs a hidden
   host-side staging copy per invocation, and the observable "transfer" is
   DMA-from-staging at pinned rates while the host-side memcpy + staging
   copy cost lands in `host_submit_us`.
2. **Concurrency comes from decoupling the staging copies from the DMA.**
   The Handbook's two-buffer ping-pong is the canonical shape: CPU memcpy into
   pinned buffer A while the GPU pulls buffer B, ~2x pageable throughput, and
   the author's "synchronous" variant -- sync moved after submit -- loses
   ~30%. Our PP executor is structurally the SYNCHRONOUS variant: per row
   `set_async` (hidden staging copy inside the API) followed by a host-block
   on `ev_copy_ready` before the next layer's fetch. The measured 3.34 GB/s
   is consistent with roughly half of the pinned link rate (the TG reactive
   9.16 GB/s overlaps fetch with decode and batches many small fetches into
   one submission).

**Verified-locally upstream mechanism that applies.** The vendored CUDA backend
exposes `ggml_backend_cuda_register_host_buffer` (ggml-cuda.cu:4641), gated
behind env `GGML_CUDA_REGISTER_HOST`, which `cudaHostRegister`s a host buffer
`Portable | ReadOnly`. **Nothing in `src/` or the runner calls it** (verified
by tree-wide grep; only the SYCL twin exists alongside). So the mmap weight
window is never pinned today, and the documented upstream knob is available
WITHOUT any code change: export `GGML_CUDA_REGISTER_HOST=1` and the CUDA
backend pins registered host buffers. Caveat found while verifying: the hook
is exported through `ggml_backend_reg_get_proc_address` (name
`"ggml_backend_register_host_buffer"`), but no loader/runner call site for it
exists in this vendored tree's `src/` either -- so the env-var route may be
dead code on this build. The honest experiment is: set the env var, run the 4k
arm, and check whether `host_submit_us` per fetch drops at all. If it does
not, the mmap buffer is not being registered, and a minimal registration call
at runner bring-up (register the mmap base window once, read-only) is the
smallest vendored change that would make H-D's overlap real. Probe order:
env var first (zero code), then the registration call, then (only if
staging-copy cost is proven dominant) the explicit two-buffer ping-pong, which
is a much larger change to the fetch loop.

This also refines H-D's framing: the "no overlap window" is not primarily a
scheduling failure of the executor -- it is the pageable-staging path
serializing host memcpy and DMA inside each API call. TG reactive looks 2.7x
faster per byte because its many small fetches batch into the software queue
behind decode work; PP's eviction-driven whole-bank copies submit-and-wait.
The proposed measurement stands (prefetch-ahead depth A/B), with the pinned
source as the prerequisite that decides whether prefetch can overlap at all.

### 8.4 Source-vs-tree verification and residual unknowns

Every online claim above was checked against the vendored tree before being
recorded: events are created with `cudaEventDisableTiming` only
(ggml-cuda.cu:5356), `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` exists only
in the cc 12.1 iGPU branch (ggml-cuda.cu:369), `GGML_CUDA_USE_GRAPHS` is
build-enabled (ggml/src/ggml-cuda/CMakeLists.txt:135), the copy backend is a
second CUDA backend on the same device (`ggml_backend_cuda_init(device)`,
llama-layer-streaming.cpp:660), and `set_tensor_2d_async`/`cudaMemcpy2DAsync`
exist upstream of the row-set loop (ggml-backend.cpp:282, ggml-cuda.cu:2452)
but the row-set fetch loop issues per-row `set_async` calls, not a 2D batch.

Residual unknowns (not resolvable from sources): (a) whether WDDM command-
buffer depth on driver 610.74 / CUDA UMD 13.3 is still ~20 commands as the
2013-era sources describe -- the batch-size constant is not part of the public
API and may have changed; (b) whether `GGML_CUDA_REGISTER_HOST` pinning can
apply to an mmap window large enough for one full bank (18.33 GB) or must be
drilled to the slab region actually read -- `cudaHostRegister` on a 25 GB
region may fail or inflate commit charge, and AGENT_ENVIRONMENT.md already
records no SeLockMemoryPrivilege (that governs VirtualLock, not
cudaHostRegister, but it bounds how much host page-locking this token can
do); (c) the K-dependence A/B (8.2) that would fully separate bookkeeping
from submission mechanics.

No default, no gate verdict, no implementation changed. Nothing committed
(owner's standing instruction for this mission).
