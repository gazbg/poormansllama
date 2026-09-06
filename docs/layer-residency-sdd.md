# Generic Prefill Layer Residency SDD

Status: Draft - scope approved 2026-09-04; implementation pending

Reference implementation: `third_party/llama.cpp/src/llama-layer-streaming.*`

Reference evidence: [Qwen3.8 Dense Layer Streaming SDD](qwen38-layer-streaming-sdd.md)
and [Current State](currentstate.md)

Executable plan: [Generic Prefill Layer Residency Plan](plans/generic-prefill-layer-residency-plan.md)

## 1. Purpose

Generalize the proven Qwen3.8 CUDA layer streamer into an opt-in llama.cpp
mechanism that stages complete immutable layer weight packages through a
bounded VRAM window during prefill.

The mechanism is common. Package membership and one-layer graph construction
remain architecture-declared because llama.cpp models do not share one tensor
layout, mutable-state contract, or graph topology.

"Available to any loaded model" means every model receives an explicit
capability result. A supported adapter supplies a package catalog and one-layer
graph contract. An unsupported model is refused with a structured reason and
can still use normal llama.cpp when the experiment is disabled.

## 2. Fixed Acceptance Targets

### Dense reference

- Model: the verified Qwen3.8-27B GGUF in `docs/AGENT_ENVIRONMENT.md`.
- Package: one complete immutable decoder layer.
- Purpose: prove extraction preserves the existing implementation and evidence.

### MoE target

- Model: `C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`.
- Model class: Qwen3.6-35B-A3B MoE with MTP weights present.
- Context capacity: 262,144 tokens.
- KV cache: Q8_0 for both K and V.
- Prefill chunk width: 512 tokens unless inventory finds a model constraint.
- Package: the complete routed-expert bank for one logical MoE layer.
- Intent: static weights, Q8 KV, state, activations, workspaces, and all model
  weights deliberately do not need to fit together in VRAM. The experiment
  must prove bounded residency, not accidental full-model placement.
- MTP execution: disabled for the first prefill acceptance run. MTP tensors are
  inventoried but are not part of the rolling expert package.

The model path is known to exist. Byte size, GGUF architecture key, layer and
expert counts, tensor types, package sizes, and exact capacity arithmetic must
be recorded by the environment-inventory task before implementation claims
compatibility.

## 3. Existing Evidence and Gap

The Qwen reference already has generation-checked CUDA slots, asynchronous
copy/consumer events, layer-major prefill, persistent per-width graph sets,
activation slabs, tail chunks, capacity telemetry, and runs through 262,144
tokens.

It is not generic today. `llama-layer-streaming.cpp` hard-codes Qwen recurrent,
attention, and MTP `llama_layer` members, while `qwen35.cpp` owns the split
layer graph. No second architecture or MoE full-expert package has passed the
same lifecycle.

## 4. Package Contract

An architecture adapter returns one immutable package descriptor per pageable
logical layer. It records:

- architecture, logical layer, and package policy;
- deterministic source and destination tensor identities;
- exact type, shape, bytes, alignment, and backend requirements;
- package class and slot-capacity requirement; and
- explicit static-weight and mutable-state exclusions.

Package members become ready and retire atomically. A graph cannot observe a
partial copy. Catalog construction rejects missing required members, duplicate
destination ranges, mutable tensors, unsupported types, and undersized slots.

### Dense package

The package contains all immutable weights consumed by one dense decoder
layer. Embeddings, final norm, output head, KV cache, recurrent state, and
workspaces remain outside the slot window.

### MoE prefill package

The package contains every routed-expert weight for one MoE layer. Router,
attention, normalization, shared experts, embeddings, output head, KV cache,
and mutable state remain outside it unless a later measured policy says
otherwise.

For chunk width B and top-k routing, a layer sees B*k expert selections. A
512-token chunk can cover most or all experts, so one complete expert-bank
transfer may be cheaper than sparse reads and uploads. This policy does not
predict routing: the package is ready before the router runs, and the existing
graph evaluates exactly the router-selected IDs.

## 5. Layer-Major Prefill

```text
embed prompt chunks into activation slab A
for logical layer L in model order:
    make package L ready in its assigned slot generation
    for chunk C in ascending token-position order:
        execute layer L on A[C] and write B[C]
    release the final consumer of package L
    swap A and B
run the output graph for requested logits
```

Logical KV, recurrent, convolution, and other mutable state stays indexed by
logical layer. State is never stored in or addressed through a weight slot.
Architecture-specific causal metadata and state variants are prepared by the
adapter and shape-checked before graph launch.

The common executor may prefetch deterministic future packages while layer L
computes. It waits only for L's matching ready event. Slot reuse waits for all
consumers of the previous generation.

## 6. Adapter Boundary

Common code owns package storage, slots, generations, copy/event ordering,
layer-major activation slabs, capacity preflight, graph-set lifecycle, and
telemetry.

Each architecture adapter owns only:

1. capability and unsupported-reason reporting;
2. static, pageable, and mutable-state classification;
3. deterministic package-member enumeration;
4. slot-bound one-layer graph construction;
5. causal and state preparation per chunk; and
6. required graph variants and a focused parity output.

Adapters must reuse existing model tensors and graph helpers. They must not
copy the slot scheduler, telemetry writer, or capacity planner. The existing
Qwen code becomes the first adapter by extraction, not rewrite.

## 7. Residency and Capacity

- Source weights remain backed by the existing GGUF loader lifetime.
- Quantized bytes are copied without conversion or reinterpretation.
- Direct host-backed copies remain the default; pinned staging stays optional.
- Every slot has a logical layer and monotonically increasing generation.
- Copy-ready and consumer-done events belong to that generation.
- A slot is reused only after the prior generation's consumers complete.
- Coalescing adjacent source ranges is allowed only when byte identity and
  destination offsets remain exact.

Preflight accounts for static CUDA weights, slot buffers, Q8 KV at 262,144
tokens, recurrent state, activation slabs, graph/input contexts, workspaces,
and a headroom reserve. It reports required, available, and excluded bytes.

The MoE acceptance run must record that the complete model plus runtime state
does not fit in physical VRAM. If it unexpectedly fits, the task must choose a
smaller explicit residency budget rather than claiming an out-of-core result.

## 8. Policy

Initial private modes are:

- `off`: normal llama.cpp behavior;
- `dense-layer`: complete dense decoder packages;
- `moe-expert-bank`: complete routed-expert packages; and
- `auto`: only a later policy task may select among adapter-supported modes.

Forced modes exist for lifecycle testing. Until a cost model is validated,
`auto` may use an explicit prompt-width threshold but must report all inputs
and must not claim optimality.

### 8.1 Tested support matrix (T-GPLR-05)

The registry currently declares exactly two adapters. Every other
architecture receives a deterministic `off` result with the stable reason
`no layer-residency adapter for this architecture`.

| Architecture | Adapter | Declared policy | Status |
| --- | --- | --- | --- |
| `qwen35` | `qwen35` | `dense-layer` | supported: validated through T-GPLR-00/01/03/04 regressions (Qwen3.8-27B reference) |
| `qwen35moe` | `qwen35moe` | `moe-expert-bank` | supported: validated through T-GPLR-02/03/04 (Qwen3.6-35B-A3B, 262K Q8 out-of-core proof) |
| everything else | none | `off` | unsupported: forced policies refused before CUDA init (exit 2, stable reason); `auto` runs `off` and explains why |

Case classes the documentation must keep distinct:

- **Supported**: an adapter-declared policy whose acceptance gates passed in
  this workspace (the two rows above; nothing else).
- **Unsupported**: any other architecture or a forced policy/arch mismatch —
  refused host-only before `llama_backend_init` (exit 2, stable reason);
  `auto` on an unsupported architecture runs `off` and explains why on stderr (Policy, section 8).
- **Correct-but-slower**: configurations that are validated for correctness
  but not claimed as performance wins (for example dense layer-major with
  4 slots, or any forced policy without a cost model). Correctness parity is
  recorded per task; no throughput claim accompanies them.
- **Unmeasured**: MTP/TG expert prediction, weight rotation, multi-GPU, and
  any other architecture — no correctness or performance evidence exists;
  the probe reports `off` for them by construction.

## 9. Failure Contract

The experimental path fails loudly for a forced unsupported architecture or policy,
invalid package membership, insufficient capacity, stale generation, source
read or copy failure, failed CUDA event, missing graph variant, or graph launch
failure.

It never substitutes a layer, drops an expert, changes routing, continues with
partial weights, or silently falls back. Normal llama.cpp remains available by
running with the experiment disabled.

## 10. Telemetry

Retain the existing JSON Lines lifecycle. Add model architecture, adapter,
package policy, member count, package bytes, slot capacities, context and KV
types, static/slot/KV/activation/workspace byte classes, and support decision.

The MoE path also reports expert count, top-k, route union width per layer and
chunk, expert-bank copies per layer pass, and sampled exact router IDs for the
parity check. Routing telemetry is observational and never influences routing.

Host submission, ready latency, availability wait, compute, graph patch/setup,
and wall time remain distinct. Windows file-cache state remains best-effort and
must not be described as controlled cold-cache evidence.

## 11. Acceptance Gates

1. Extracted Qwen adapter reproduces the reference package bytes, lifecycle,
   focused greedy ID, and telemetry.
2. The Qwen3.6 MoE inventory proves the 262,144-token Q8 configuration's exact
   capacity and package layout before CUDA execution.
3. Every MoE layer copies one complete expert bank per layer-major pass and
   preserves exact router IDs and top-k cardinality against a control.
4. The 262,144-token configuration either completes or fails at the documented
   capacity preflight; no accidental full-model CUDA placement is allowed.
5. Unsupported models and insufficient budgets fail before timed execution.
6. Default-off llama.cpp builds and behavior remain unchanged.
7. Any speed claim requires a paired normal-versus-residency run on the same
   model, prompt, build, and observed GPU state.

## 12. Known Risks

- Qwen3.6 support in the vendored revision and its exact graph ownership are
  not yet verified.
- A 262,144-token activation slab or Q8 KV allocation may dominate VRAM before
  weight slots; preflight must distinguish those classes.
- Some CUDA quantized kernels repack weights after upload. Repacking every
  layer load can erase the benefit.
- Full-expert-bank transfer may lose to sparse admission when routing is
  strongly skewed or expert count is very large.
- Dense token generation remains PCIe-bound and is not the target of this
  prefill design.
