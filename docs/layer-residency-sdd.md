# Generic Prefill Layer Residency SDD

Status: Draft - scope approved 2026-09-04; implementation pending
Amended 2026-09-12: `moe-row-set` TG package granularity approved by the
owner (T-GPLR-13 gate (a)); see sections 1, 4, 8, 9, 10, 11, 12.
Amended 2026-09-13: per-layer row cache (`moe-row-cache`, the F2 class) defined
as a TG-phase residency class under `moe-row-set`, so that measured row reuse
is realizable and T-GPLR-15's cumulative-resident-set ceiling is measurable;
see sections 1, 4, 6, 7, 8, 9, 10, 11, 12.
Amended 2026-09-13: gate 9 decision rule stated by owner direction (T-GPLR-15
metric reconciliation), so the verdict statistic is single-owned here; see
sections 10 and 11 (gate 9).

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

Scope amendment 2026-09-12: the primary workload remains layer-major prefill.
The `moe-row-set` package (section 4) extends the package contract to the
token-generation phase of MoE models as an owner-approved, separately gated
policy. It changes nothing in the prefill policies, package classes, or
acceptance targets above.

Scope amendment 2026-09-13: the per-layer row cache (section 4) is the
storage class that makes measured row reuse realizable and gives T-GPLR-15's
predictive variant a substrate to pre-stage into. It is an opt-in sub-mode of
`moe-row-set`, applies only to the TG phase, changes nothing in prefill, and
changes no default.

Scope amendment 2026-09-13 (owner-directed): the `moe-row-cache` RETENTION
POLICY is resolved to a ROW-CAPACITY LRU. `K` names a row BUDGET of
`min(n_expert, K*W*top_k)` rows per layer, and the cache keeps that budget's
most recently ROUTED rows; it does not hold exactly the rows routed in the
last K windows. Section 4's capacity language and gate 9's statistic (b) had
been read those two ways, and the two readings are not the same rule: they
agree exactly at K=1 and diverge at K>=2, where the row-capacity reading can
only retain more. The footprint is `K*W*top_k` rows per layer under EITHER
reading, so the resolved policy costs nothing and the window-count reading is
retired as strictly weaker. This makes gate 9's (b) — already computed under
the row-capacity reading — the mechanism's measured behaviour instead of an
overstatement of it; measured on the preserved 10,200-pair probe stream, (b)
at K=2/K=4 is 0.4343/0.6284 under the resolved policy against
0.4015/0.5194 under the retired one. The implementation, the C++ checker and
the offline analyzer are pinned together by
`tools/route_overlap/crosscheck-retention.py`, which fails on any divergence.

Scope amendment 2026-09-14 (owner-directed): the per-layer row cache's DEFAULT
retention depth is K=4 — 32 rows per layer, ~2.19 GB across the 40 MoE layers,
the shallowest depth whose offline statistic (b) clears gate 9's 0.6 threshold
(0.6284, first crossing ~K=3.6). This is a PROVISIONAL OPERATING DEFAULT, not a
gate verdict. T-GPLR-15's close stands unchanged: the gate is read at the depth
a claiming task declares, and at the SDD's own reference sizing (K=1-2) it was
measured BELOW 0.6 on the bias-robust half of the measurement. No gate PASS is
ESTABLISHED at any depth, here or since: every reading above 0.6 (K>=4) comes
from the stream that echoes its own anchor, so its absolute (b) is optimistic
and none of them was validated on a non-echoing stream. The non-echoing
re-probe recorded as the precondition for any retention above the reference
band therefore becomes the outstanding VALIDATION of this default and
the first requirement of any task that claims a K=4 gate PASS — it is no longer
a precondition for provisioning the depth. The mechanism itself remains opt-in
and default-off: disabling the experimental option changes no default placement
and no non-experimental behaviour, and the depth applies only when an operator
selects the cache. `--residency-tg-retain` with no value selects K=4 where the
device affords it and otherwise the largest depth that does, so K=4 is a depth
CEILING for the bare flag rather than a fixed allocation: the depth is fitted at
bring-up to free VRAM less the operator's `--reserve-mib` and less the slot
window the run has not allocated yet, and a device that cannot take even K=1 is
refused loudly before any timed work. That arithmetic has exactly one owner: a
single budget model in the runner (`llama_ls_vram_budget`) that holds the flags,
the bounded-window estimate, both device readings and every derived byte
quantity, and that the T-GPLR-04 capacity preflight and this fit both read — so
the preflight's numbers and the fit's numbers cannot disagree about what is
already allocated. A NAMED depth is never reduced — any depth
in 1..64 may still be named explicitly and is allocated in full or fails loudly
— so the depth a task declares for a gate reading is always the depth actually
provisioned, and the fitted result is what the `rowset_init` telemetry reports.

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

### moe-row-set package (TG row windows; amended 2026-09-12)

For one logical MoE layer and one token-generation window W (default: one
token; small token groups are an implementation-review decision for T-GPLR-13),
the package contains the deduplicated union of routed-expert rows — across
`ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` — that the router selects
for the tokens in W, at top-k cardinality. Router, attention, normalization,
shared experts, embeddings, output head, KV cache, and mutable state remain
outside it.

- **Reactive membership.** The union is computed from the router's exact
  executed IDs for W. The policy never predicts routing and never alters
  router arithmetic, expert IDs, or quantization bytes; rows are copied
  byte-exact at their natural offsets. Because the IDs for W exist only
  after the router runs, the layer executes in two phases at TG — the
  attention-plus-router phase first, then row staging by the residency
  machinery, then the row-consuming FFN phase — and the execution-shape
  constraint below binds the row-consuming phase: both phases consume only
  CUDA-resident weights, and staging between them belongs to the residency
  mechanism (copy event, generation, atomic readiness), never to the
  backend scheduler. The exact graph split point is a T-GPLR-13
  implementation-review decision; a predictive variant (T-GPLR-15) may
  pre-stage the next window during the current one but still waits for the
  reactive superset before its FFN phase runs.
- **Immutable generation.** A row-set package is immutable for its
  generation; its members (the three row sets plus row-identity metadata)
  become ready and retire atomically. A graph cannot observe a partial copy.
- **Execution-shape constraint (hard requirement).** Row-set packages are
  CUDA-resident slot buffers staged by the residency mechanism BEFORE the
  layer graph executes, under the same copy-event, generation, and
  consumer-done lifecycle as every other package. `MUL_MAT_ID` in a row-set
  graph must always consume CUDA-resident weights owned by the matching slot
  generation. Cross-backend staging INSIDE graph execution is prohibited on
  this path: the backend scheduler's expert row-copy (`ggml-backend.cpp`
  compute_splits) and any op-offload path that rewires a weight source to a
  compute-usage staging buffer must be disabled for row-set graphs. Evidence:
  the T-GPLR-16 MIN_BATCH=1 route deadlocked the device on the 3rd TG token
  whenever CUDA `MUL_MAT_ID` at token batch consumed scheduler-staged
  host-bank weights — regardless of CUDA graphs or rowcopy
  (docs/currentstate.md, 2026-09-12). This constraint keeps the row-set path
  in the proven full-handoff execution shape.
- **Loud under-coverage.** The staged row set must be a superset of the
  router's exact selected IDs for W. A missing row refuses loudly (exit 2,
  structured telemetry, stable reason) and never substitutes or drops an
  expert.
- **Capacity.** Row-set window bytes are a preflight input like any slot
  occupancy. Union width is volatile (near-saturated for wide windows per
  T-GPLR-14); preflight must account for the worst case, not the mean.

### moe-row-cache package (per-layer persistent rows; amended 2026-09-13)

The F2 storage class (label from docs/gplr13-bottlenecks-hypotheses.md section
6; no task ID assigned yet). It exists so that measured row reuse can be
REALIZED: the `moe-row-set` slot buffers are keyed by slot, and the bundles of
the layers that share a slot alias one device buffer at offset 0, so between
two visits of a logical layer an intervening owner overwrites the same byte
ranges — a resident claim made without per-layer storage is therefore unsound
(analysis: section 6.4 of that file). The same storage is what T-GPLR-15's
predictive variant needs in order to hold pre-staged rows until their window
arrives.

- **Membership.** Per logical MoE layer, the routed-expert rows that layer has
  staged for recent windows, at member granularity (`ffn_gate_exps`,
  `ffn_up_exps`, `ffn_down_exps`), as byte-exact contiguous slabs at their
  natural offsets — row identity is exactly the `moe-row-set` identity.
- **Per-layer keying and provenance (defining property).** Storage and every
  resident claim are keyed by LOGICAL LAYER, never by slot. A row is claimable
  as resident only when stored provenance — (layer, member, expert, written
  window/generation) — proves the bytes belong to that layer and have not been
  invalidated. Provenance is structural, stored with the bytes; it is never
  inferred from slot ownership or from a bare coverage bitmap.
- **Bounded retention (policy resolved 2026-09-13, owner-directed).** Capacity
  is declared per layer and in total, sized for the worst case:
  `min(n_expert, K*W*top_k)` rows x that layer's member slab bytes, for a
  budget of K windows' worth of rows at width W (W=1, top_k=8 on the Qwen3.6
  target: ~14.2 MB per 8-row set; a one-to-two window budget is ~0.57-1.13 GB
  across 40 MoE layers). K is a ROW BUDGET, not a window count: under that
  budget the cache keeps the K*W*top_k most recently ROUTED rows per layer,
  evicting a row only when a row this window needs must be placed. Retention
  DEPTH is therefore data-dependent and unbounded in time; only the row count
  is bounded, which is what the VRAM bound requires. A row needed by the
  current window is never an eviction victim, so a write that would exceed the
  declared capacity is a loud refusal — never silent growth, and never
  eviction of an entry a running consumer may still read. **Default depth
  (owner-directed 2026-09-14): K=4** (32 rows/layer, ~2.19 GB over 40 MoE
  layers), fitted to the device — a caller that selects the cache without naming
  a depth gets K=4 where the device affords it and otherwise the largest depth
  that fits, while a NAMED depth is never reduced. The capacity arithmetic above
  is unchanged and any K in 1..64 may be named explicitly. See the 2026-09-14
  scope amendment for the fit and for why this is a provisional operating
  default rather than a gate verdict.
- **Atomicity per window.** A window's rows become ready together and retire
  together; a row-consuming graph observes only ready entries, and a cached
  row consumed by a layer's row-consuming phase is ordered after the matching
  copy event, exactly like a link-staged row.
- **Execution shape.** Cache entries are CUDA-resident, staged by the
  residency mechanism BEFORE the graph executes, under the same
  copy-event/generation/consumer-done lifecycle; the section-4 execution-shape
  constraint applies unchanged. Two shapes are permitted, and the adopting
  task must declare which it uses: (i) cache to slot-bundle device-side
  refresh before the row-consuming phase, or (ii) graph tensors bound
  directly to the cache buffer, with provenance covering every bound range.
  Either way the FFN consumes only weights owned by the matching generation.
- **Reuse optimizes copies only.** The router's exact IDs still define the
  required coverage; a cache entry removes a row from the copy list and never
  substitutes for the superset check.
- **Not prediction.** Filling the cache reactively is this class; filling it
  ahead of the reactive need is T-GPLR-15's variant, whose fills are still
  residency copies and whose consumption still waits for reactive coverage.
  The class never alters routing, expert IDs, quantization bytes, or
  checkpoint metadata, and never owns KV, recurrent state, or session state.

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

For `moe-row-set` the adapter additionally owns (7) per-window reactive
row-membership enumeration — computing each window's deduplicated expert-row
union from the router's exact executed IDs and emitting the `rowset_fetch`
telemetry — and (8) declaring which scheduler staging paths its row-set graphs
forbid, so the section-4 execution-shape constraint is enforced at the
adapter boundary rather than trusted to global defaults.

For a per-layer row cache the adapter additionally owns (9) declaring the
cache's worst-case row count per layer and the member/expert ranges its
provenance may hold, so cache capacity and soundness are adapter facts rather
than global defaults.

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
- A per-layer row cache (section 4) is a further residency class: its declared
  capacity is a preflight input like slot occupancy, its fills and refreshes
  are ordinary residency copies under the same event/generation/
  consumer-done discipline, and it never changes a slot window's contents or
  lifecycle. Headroom accounting must carry the cache, the slots, KV, state,
  activations, and workspaces together.

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
- `moe-expert-bank`: complete routed-expert packages;
- `moe-row-set`: TG-phase reactive routed-expert row windows (amended
  2026-09-12; owner-gated, implemented by T-GPLR-13, falsifiable per the
  T-GPLR-15 gate); and
- `auto`: only a later policy task may select among adapter-supported modes.

`moe-row-set` is opt-in and CUDA-only like every other experimental mode. It
applies only to the token-generation phase; PP keeps its declared policy
(`moe-expert-bank` unless a later task says otherwise). Enabling it changes no
default: without the explicit flag, runner placement, `-ncmoe` behavior, and
the CPU path are byte-for-byte unchanged.

The per-layer row cache is an opt-in sub-mode of `moe-row-set`
(`moe-row-cache`, amended 2026-09-13), CUDA-only and TG-only like its parent.
Where the parent declares a per-slot row window, the sub-mode adds per-layer
persistent storage for rows that the logical layer has already staged. It
changes no default: without the explicit flag the row-set parent behavior,
runner placement, `-ncmoe` behavior, and the CPU path stay byte-for-byte
unchanged.

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

For `moe-row-set` it additionally fails loudly for a staged row set that does
not cover the router's exact selected IDs for the window (under-coverage), a
row-set generation consumed before its copy event completes, or any attempt to
route `MUL_MAT_ID` on this path through a scheduler cross-backend staging
buffer (the execution-shape constraint of section 4). It never substitutes a
layer, drops an expert, changes routing, continues with partial weights, or
silently falls back.

For the per-layer row cache it fails loudly for allocation beyond the declared
capacity, retention overflow, a missing or invalidated provenance record, a
resident claim made without provenance, or an entry consumed before its copy
event completes. It never silently disables the cache mid-run: a cache-off run
is an explicit, separately recorded run with its own trace. Normal llama.cpp remains available by
running with the experiment disabled.

## 10. Telemetry

Retain the existing JSON Lines lifecycle. Add model architecture, adapter,
package policy, member count, package bytes, slot capacities, context and KV
types, static/slot/KV/activation/workspace byte classes, and support decision.

The MoE path also reports expert count, top-k, route union width per layer and
chunk, expert-bank copies per layer pass, and sampled exact router IDs for the
parity check. Routing telemetry is observational and never influences routing.

For `moe-row-set`, the path additionally emits per-layer `rowset_fetch`
events carrying the FULL per-token union ID list for each window (also the
gate telemetry for T-GPLR-15's temporal hit rate — no separate instrument
appears), staged row-set bytes per window and per TG round, and the standard
distinct timings (host submit, ready latency, availability wait, compute,
graph setup, wall time) per window. Staging bytes must be reported against the
full-bank baseline so the union-width saving is measurable, never assumed.

For a per-layer row cache the path separately reports cache-served rows,
device-side refresh bytes, link-staged bytes, cache high-water bytes, and
invalidation counts, so the link saving and the added copy class are both
measurable and never double-booked. Row identity fields (`route_union`,
`union_ids`, `hash`) keep their exact semantics: union IDs remain the router's
EXECUTED IDs — never predicted IDs and never the cache's contents — so the
reactive superset gate stays checkable. The T-GPLR-15 gate inputs are two
distinct statistics: (a) consecutive-window exact containment from the union
streams, which is a property of the routing stream and therefore measurable
with the cache disabled; and (b) cumulative-resident-set containment, the
realized predictor ceiling — computable offline by simulating a bounded
retention over the recorded union streams, and additionally well-defined at
runtime once per-layer persistence exists (the cache-served-row telemetry
above is that runtime form, and is the positive-reuse check of gate 10).

Gate 9 decision rule (sole owner; owner-directed 2026-09-13): the T-GPLR-15
verdict statistic is (b), the realized predictor ceiling — the fraction of
the NEXT window's routed expert rows that a persisting per-layer cache
(retention budget K windows' worth of rows, W=1) would already hold, with the
next window in the denominator (`|A n B| / |B|`, A = resident set, B = the
next window's required set). The cache the statistic assumes is the section 4
`moe-row-cache` policy resolved 2026-09-13: a ROW-CAPACITY LRU keeping the
`min(n_expert, K*W*top_k)` most recently routed rows per layer, which is
exactly what the runtime implements — so (b) predicts the mechanism's measured
staging reduction rather than bounding it from above. The 0.6 threshold in gate 9 is read against that statistic.
Context statistics may be reported in either direction (`/ |A|` or `/ |B|`),
but the direction must be stated wherever they appear; the verdict statistic
is always next-window-in-denominator. Exact window-hash equality and exact
consecutive-window containment are reported as CONTEXT and are never the
verdict: at top_k=8 over n_expert=256,
exact window identity is structurally near zero — measured 0/80 pairs at
every recorded ladder rung — so gating on it would falsify a live direction
on a sparsity artifact rather than on routing predictability. The T-GPLR-15
task block and `docs/currentstate.md` cite this rule; neither restates a
different one.

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
8. `moe-row-set` (amended 2026-09-12): for every layer and window, the
   executed row set covers the router's exact expert IDs (route-event parity;
   no missing-row path); the checker validates `rowset_fetch` completeness
   (union ID lists sum to per-token route counts); staging bytes per TG round
   are recorded against the full-bank baseline as an instrumentation result;
   the semantic-QA answer gate passes on the same run; and the dense ctest
   plus qa-smoke 512 default-off regressions stay green.
9. `moe-row-set` performance is claimed only if wall TG tok/s improves over
   the reactive row-set baseline at equal shape (T-GPLR-15's gate), and the
   measured token-level temporal hit rate reaches 0.6 before any predictive
   variant is implemented (the statistic the phrase names, its definition,
   and the direction of every containment figure are owned by the gate 9
   decision rule in section 10); below 0.6 the predictive task closes as
   falsified-by-evidence.
10. `moe-row-cache` (amended 2026-09-13): with the cache enabled, executed
    coverage still equals the router's exact IDs for every layer and window
    (no missing-row path; checker-validated), and reuse is demonstrated
    POSITIVELY — cache-served rows non-zero in a real trace, with bytes
    reported against both the full-bank baseline and the reactive cache-off
    baseline. Any performance claim requires a cache-on/cache-off pair at
    equal shape, because the cache changes the byte denominator and a lone
    fast run is not evidence. Cache-off and default-off paths remain
    byte-for-byte unchanged.

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
- `moe-row-set` risks (amended 2026-09-12): the token-level temporal hit rate
  that motivates row windows is UNMEASURED (T-GPLR-14 found no trace with
  per-token TG routes); if it measures below 0.6, the predictive variant is
  falsified and even the reactive policy may not beat full-bank at TG.
  Union width is volatile — near-saturated for wide windows, collapsing in
  late layers on short prompts — so capacity preflight must size for the
  worst case. The execution-shape constraint (section 4) is a measured
  landmine, not a style preference: scheduler cross-backend staging of
  host-resident banks deadlocked the device under T-GPLR-16, and any future
  "optimization" that moves row-set staging inside graph execution walks
  back into that failure mode.
- `moe-row-cache` risks (amended 2026-09-13): the class adds a VRAM footprint
  (~0.57-1.13 GB at the measured Qwen3.6 shape for one-to-two window retention
  across 40 MoE layers) against the bounded-residency intent, so preflight
  must carry it explicitly. Provenance mistakes could execute another layer's
  weights — the exact hazard that falsified the bookkeeping-only variant
  (bottlenecks doc section 6.4) — so provenance must be structural, never
  inferred. The device-side refresh is a second copy class that can be
  mistaken for a link saving, and a cache can mask a link regression, so
  cache-on/cache-off pairs are required. A passing 0.6 stream gate does not by
  itself imply a wall-time win: the measured 13.7 GB/s H2D link (T-Q38TG-00)
  bounds per-byte gains, and per-call overhead (plan task T-GPLR-18) bounds
  the rest.
