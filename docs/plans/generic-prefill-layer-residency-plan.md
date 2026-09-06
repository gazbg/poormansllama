# Generic Prefill Layer Residency Plan

Status: complete - T-GPLR-00 through T-GPLR-05 done (2026-09-04/05)

Design: [Generic Prefill Layer Residency SDD](../layer-residency-sdd.md)

Reference baseline: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
and [Current State](../currentstate.md)

Environment facts: [Agent Environment and Dependency Guide](../AGENT_ENVIRONMENT.md)

## Outcome

Turn the Qwen-specific CUDA layer streamer into a common, default-off prefill
layer-residency mechanism. Preserve the dense Qwen3.8 reference, then run
Qwen3.6-35B-A3B with a 262,144-token Q8 KV allocation while only a bounded
complete expert-layer package is resident in VRAM.

The first acceptance prefill is 512 tokens in one chunk. The 262,144 value is
the context/KV capacity, not a requirement to process a 262,144-token prompt.
This isolates weight residency from long-prompt activation-slab cost.

## Known Evidence

- The existing Qwen implementation has working CUDA slots, generation checks,
  copy/consumer events, layer-major execution, activation slabs, persistent
  graphs, preflight, and telemetry validation.
- Package lists and graph integration are Qwen-specific today.
- The selected MoE GGUF path exists, but its header, tensor inventory, vendored
  architecture support, package sizes, and capacity arithmetic are not yet
  verified in this workspace.
- The root and vendored worktrees contain user changes. Every task preserves
  them and stages or commits nothing without owner authorization.

## Fixed MoE Acceptance Command Shape

The exact flags may be added by the plan tasks, but the accepted run has this
meaning:

```text
model       = C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf
context     = 262144
kv K/V      = q8_0 / q8_0
prefill     = 512 tokens
ubatch      = 512
schedule    = layer-major
residency   = moe-expert-bank
MTP         = disabled
backend     = CUDA
```

## Dependency Order

```text
T-GPLR-00 -> T-GPLR-01 -> T-GPLR-02 -> T-GPLR-03 -> T-GPLR-04 -> T-GPLR-05
```

### T-GPLR-00: Inventory the MoE artifact and capacity
Status: done (2026-09-04)
Depends on: none
Commit: pending (no commit authorized)

Evidence (2026-09-04)
: - `--inspect-model` host-only mode added to the runner; shared implementation
    `llama_ls_model_inventory()` in `src/llama-layer-streaming.cpp`. It parses
    GGUF with `gguf_init_from_file(no_alloc=true, ctx=nullptr)` and returns
    before `llama_backend_init()`; no CUDA device enumeration, no tensor data.
  - New host CTest `test-layer-streaming-model-inspect`: dense/MoE synthetic
    fixtures + negatives (unsupported arch, missing expert_count, unclassified
    tensor, incomplete expert bank, missing file). 100% pass.
  - `cmake --build build/llama-cuda --config Release --target
    llama-layer-streaming --parallel 16` -> exit 0.
  - Inventory: `build/llama-cuda/q36-moe-inventory.json`. qwen35moe, 41 blocks
    (40 trunk + 1 MTP), 256 experts, top-8, 753 tensors, 21,377,327,616 B data.
    Expert banks: 37 x 452,984,832 B + 3 x 522,190,848 B (layers 34/38/39,
    largest 522,190,848 B). Router 83,886,080 B, shared-expert 134,021,120 B,
    attention 289,689,600 B, recurrent 1,089,231,360 B, norms 655,360 B,
    static 957,530,112 B, MTP 495,302,656 B. Q8_0 KV @262,144 = 2,852,126,720 B,
    DeltaNet state = 65,863,680 B.
  - Capacity: full_model_plus_runtime 24,295,318,016 B < physical 25,650,855,936 B
    -> `require_sub_full_budget: true` (T-GPLR-04 must enforce a sub-full budget).
    Bounded 1-slot 5,995,194,880 B, 2-slot 6,517,385,728 B.
  - Residual: graph/input, activation, workspace and CUDA-context bytes remain
    unmeasured until runtime preflight (T-GPLR-03/04).
  - Incidental build-blocker fixed: restored committed `slab_off_B` (user's
    uncommitted edit had `slab_off_b`, C2065 on every compile). No other user
    modification touched.
  - Task-boundary code review round (2026-09-04): fixed 1 Critical + 4 Important
    + 6 Minor findings — `file_bytes` 32-bit `ftell` truncation on >2 GiB files
    (evidence JSON regenerated; now exact 21,388,319,008); type-mismatched
    metadata now an explicit error instead of a `GGML_ASSERT` abort; fused
    `ffn_gate_up_exps` bank accepted; `attn_qkv`/`attn_gate` classified by the
    layer's own recurrent flag; explicit `attention.recurrent_layers` array
    honored over the interval heuristic; per-tensor V sized by `n_embd_v_gqa`;
    `expert_used_count <= expert_count` sanity check; `--kv-cache` validated in
    inspect mode; JSON formatter grows instead of truncating; null `file_bytes`
    on stat failure; entry guards for null path / non-positive capacity. New
    negative fixtures cover each. All host tests re-pass (2/2); measured byte
    classes for the acceptance model are unchanged by the fixes.

Goal
: Produce a host-only JSON inventory proving the selected GGUF's architecture,
  layer/expert layout, exact package bytes, Q8 KV requirement at context
  262,144, and whether the bounded residency configuration can fit while full
  model residency cannot.

Scope
: A host-only `--inspect-model` mode in the isolated layer-streaming runner,
  existing GGUF APIs, `docs/AGENT_ENVIRONMENT.md`, `docs/currentstate.md`, and
  this plan. The mode must stop before CUDA backend or model tensor allocation.

Non-goals
: Loading model weights, initializing CUDA, changing package membership,
  inference, conversion, downloads, dependencies, or performance claims.

Design and failure contract
: Parse the GGUF header with existing in-tree APIs and no tensor allocation.
  Report file bytes, architecture key, model name, layer count, expert count,
  experts used, MTP layers, context training limit, tensor types and offsets,
  candidate static bytes, per-layer routed-expert bytes, shared/dense bytes,
  and the largest package. Compute Q8 K/V bytes for context 262,144 from the
  actual architecture metadata. Report physical VRAM separately from required
  byte classes. Unknown architecture metadata is an explicit error, not a
  guessed zero. The command must not enumerate a CUDA device.

Implementation steps
: 1. Add the smallest host-only inspect path before normal runner setup.
: 2. Reuse GGUF metadata/tensor enumeration APIs to emit deterministic JSON.
: 3. Add formulas for Q8 KV and minimum one/two-slot capacity from actual
  dimensions; label graph/workspace bytes unknown until runtime preflight.
: 4. Add a tiny fixture test for arithmetic and malformed metadata.
: 5. Run the inspector on the selected MoE GGUF and record observed facts.

Acceptance criteria
: - [x] The command exits 0 without CUDA initialization and records exact file,
      architecture, layer, expert, top-k, MTP, tensor-type, and package facts.
: - [x] Every routed-expert tensor is assigned to exactly one logical-layer
      candidate package; router, shared-expert, attention, static, and mutable
      state classes are reported separately.
: - [x] Q8 K/V bytes at context 262,144 are derived from metadata and include
      all attention layers required by the architecture.
: - [x] The report shows `full_model_plus_runtime > physical_vram`; if not, it
      marks that the later run must enforce a smaller explicit residency budget.
      (Measured: 24,295,318,016 B < 25,650,855,936 B physical; the report sets
      `require_sub_full_budget: true`.)
: - [x] The bounded one-slot and two-slot estimates state known bytes and do not
      invent graph/workspace fit.

Required validation
: - `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0.
: - `ctest --test-dir build/llama-cuda -C Release -R
  layer-streaming-model-inspect --output-on-failure` -> host tests pass.
: - `build/llama-cuda/bin/Release/llama-layer-streaming.exe --inspect-model
  C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf
  --ctx-size 262144 --kv-cache q8 --inspect-json
  build/llama-cuda/q36-moe-inventory.json` -> exit 0 without CUDA init.

State update
: Add verified MoE artifact, metadata, package-byte, Q8 KV, and capacity facts
  to `docs/AGENT_ENVIRONMENT.md`; record support blockers or readiness in
  `docs/currentstate.md`.

Escalate before implementation if
: The vendored GGUF APIs cannot inspect without weight allocation, the model
  architecture is unsupported by this vendored revision, required metadata is
  absent, or a new dependency/download would be needed.

### T-GPLR-01: Extract the Qwen package adapter
Status: done (2026-09-04)
Depends on: T-GPLR-00
Commit: pending (no commit authorized)

Evidence (2026-09-04)
: - Adapter boundary: `llama_ls_member`/`llama_ls_package`/`llama_ls_adapter`
    descriptors + `llama_ls_validate_package()` + `llama_ls_select_adapter()`
    in `src/llama-layer-streaming.{h,cpp}`; Qwen member lists (REC 22, ATT 18,
    MTP 26 members, order preserved) moved to
    `src/models/llama-layer-streaming-qwen35.cpp`. Common slot code now has no
    Qwen tensor-member list or package classification (grep-verified).
  - `init()` enumerates + validates every trunk package before any device
    allocation or copy submission; `load_layer`, slot bundle creation/alloc,
    and `begin_handoff` (incl. MTP) consume cached validated descriptors only.
  - New `pkg_member` telemetry event emits adapter, policy, class, member
    count, and package bytes per layer.
  - Host test `test-layer-streaming-descriptors`: reference member-order
    equality, six invalid-descriptor negatives (empty, duplicate dest, mutable
    member, two dests, zero-byte src, missing exclusion), capability probe
    (qwen35 supported; qwen35moe/llama report stable unsupported reasons). Pass.
  - GPU parity under `Global\AI-GPU-EXCLUSIVE-v1` (2026-09-04):
    `layer-streaming-layer-major-1024-run` + host check both pass;
    `graph_builds == 0` (both phases); first greedy ID 286 = reference;
    64 `pkg_member` events: 48 recurrent + 16 attention, layer-0 package
    364,022,656 B = recorded reference size; peak 4,783,144,960 B.
    Pre-extraction baseline run (same session, typo-fix-only build) recorded
    the same ids [286] and pass.
  - Default-off `llama-cli` build (`-DLLAMA_LAYER_STREAMING=OFF`): exit 0,
    then restored to ON with 0 compile errors and 3/3 host tests pass.
  - Note: `qwen35moe` intentionally has no adapter yet (stable unsupported
    reason); T-GPLR-02 adds the MoE adapter.
  - Task-boundary review round (2026-09-04): verdict "no Critical; extraction
    behavior-preserving on all five consumers". Fixed 3 Important + 3 Minor:
    MTP `host_layers[n_layer]` size guard (loud error instead of OOB when a
    checkpoint lacks blk.nextn); `migrate_layer` now checks `make_dev_tensor`
    null and the trunk loop propagates its failure; a single `abort_handoff()`
    cleanup on every early-return path (no stale `handoff_bytes`/`handoff_mtp`
    after a failed MTP setup); `ls_dst` gained the exactly-one-destination
    assert; stale pkg_cache header comment corrected; `pkg_member` gained
    `t_us`. Test strengthened: src/dst pairing equality for every member, full
    26-entry MTP order pin, empty-label negative.
  - Post-fix revalidation: host trio 3/3 pass; direct attention-only handoff
    run under the mutex exited 0 with migrate (16 layers, 5,638,094,848 B) +
    teardown events and `check-telemetry: OK (1222 events)`; fresh
    `layer-streaming-layer-major-1024` pair passed with ids [286],
    `graph_builds==0` (both phases), 64 `pkg_member` events.

Goal
: Dense Qwen3.8 obtains every pageable layer package through an
  architecture-neutral descriptor and a Qwen adapter without changing bytes,
  slot placement, graph behavior, or focused output.

Scope
: `third_party/llama.cpp/src/llama-layer-streaming.{h,cpp}`, the smallest
  private adapter interface under `third_party/llama.cpp/src/models/`,
  `qwen35.cpp`, package fixtures, telemetry, state docs, and this plan.

Non-goals
: MoE execution, scheduler changes, new policies, public APIs, performance
  optimization, or moving mutable state.

Design and failure contract
: Move Qwen package classification and member enumeration behind a private
  adapter. A common descriptor records source/destination identity, type,
  shape, bytes, alignment, package class, and state/static exclusions. Keep
  member order unchanged. Reject missing required members, duplicate
  destination ranges, mutable members, and undersized slots before copying.

Implementation steps
: 1. Add a host-testable descriptor and capability result.
: 2. Move recurrent, attention, and MTP member arrays into the Qwen adapter.
: 3. Build the existing slots and copy plans from descriptors.
: 4. Emit adapter, policy, member count, package class, and bytes.
: 5. Add positive Qwen and negative invalid-descriptor fixtures.

Acceptance criteria
: - [x] Common slot code contains no Qwen tensor-member list or Qwen package
      classification.
: - [x] The Qwen adapter emits the same 64 main-layer package bytes and member
      order as the reference implementation.
: - [x] Invalid descriptors fail before any copy submission.
: - [x] The focused 1,024-token Qwen3.8 layer-major run validates,
      `graph_builds == 0`, and produces the recorded reference greedy ID.
: - [x] Default-off `llama-cli` builds unchanged.

Required validation
: - Host descriptor and existing layout/lifecycle CTests -> pass.
: - Existing Qwen3.8 layer-major-1024 parity target through
  `tools/with-gpu-lock.ps1` -> pass with matching greedy ID.
: - Default-off `llama-cli` build -> pass.

State update
: Record the common/adapter boundary and unchanged Qwen evidence.

Escalate before implementation if
: Package extraction changes graph semantics, needs a public API, or conflicts
  with user modifications in the same vendored files.

### T-GPLR-02: Prove the MoE expert-bank package
Status: done (2026-09-04)
Depends on: T-GPLR-01
Commit: pending (no commit authorized)

Evidence (2026-09-04)
: - MoE adapter registered: `qwen35moe_package` (complete routed-expert bank:
    gate+up+down, or fused gate_up+down), `merge_resident=true` (slot bundle =
    resident layer with only package members replaced by device tensors),
    `causal_implicit=true`. Host fixtures: separate/fused positives, partial
    bank and zero-byte negatives; capability probe now selects qwen35moe. 3/3
    host tests pass.
  - `qwen35moe::graph` gained the `layer(il)` indirection and a
    `build_streaming_graphs()` mirror of the dense builder; `llama-graph.cpp`
    causal gate now uses `adapter->causal_implicit` (no hard-coded arch).
  - Runner `--residency auto|off|dense-layer|moe-expert-bank` (auto from a
    host-only `general.architecture` read before load); MoE loads with the
    expert-bank tensors overridden to CPU (pageable) while attention/router/
    shared/norms stay CUDA-resident. `--fault-drop-member` negative fixture.
  - Route observation: the MoE graph marks its selection_probs as a graph
    output (lifetime only, no arithmetic change); the executor reads the live
    F32 probs after each layer compute and derives exact top-k IDs (stable
    descending sort = CUDA radix tie-break), emitting one `route` event
    (sampled IDs, top-k, route-union, FNV hash). Observational only.
  - Escalation conditions cleared: `mmid.cu` reads quantized experts directly
    (no per-load repack); router IDs observable without changing arithmetic.
  - GPU evidence (mutex, Qwen3.6-35B-A3B): one-layer proof layer 0, 64 tokens:
    exactly one `copy_submit` of 452,984,832 B (the complete bank), slot_vram
    522,194,944 B, exit 0, `route` event top_k=8 route_union=192 with all IDs
    in [0,256). Rolling (1 slot) vs resident (16 slots) runs produce the
    IDENTICAL route hash (9763251156467869940) and sampled IDs
    220,167,131,161,250,237,8,185 — exact router parity.
  - Partial-package negative: `--fault-drop-member` -> loud
    `load_layer: package descriptor/bundle member mismatch`, exit_status 1,
    zero `copy_submit` events.
  - Dense regression: `layer-streaming-layer-major-1024` pair passes with
    greedy ID 286; host trio passes.
  - Residual: full 40-layer MoE layer-major prefill is T-GPLR-03; route parity
    across all layers is verified there.

Goal
: The Qwen3.6 MoE adapter constructs one complete routed-expert package for
  each MoE layer and proves one layer can execute with exact router selections.

Scope
: Qwen3.6/Qwen-family adapter support, MoE package descriptors, one-layer graph
  binding, package completeness fixtures, route observation, docs, and no
  full-model run.

Non-goals
: Full 512-token prefill, context allocation, sparse per-expert caching,
  prediction, changing top-k, shared-expert paging, MTP execution, or speed.

Design and failure contract
: Declare router, attention, normalization, and shared experts outside the
  rolling package. Include every routed-expert weight for layer L atomically.
  The existing router and MoE graph choose exact IDs after package readiness.
  Route observation cannot affect scores or selection. Missing one expert
  tensor, a partial copy, stale generation, or unsupported repacking fails the
  proof before graph launch.

Implementation steps
: 1. Add Qwen3.6 capability and explicit static/expert/state classification.
: 2. Build exact per-layer expert-bank descriptors from the inventory.
: 3. Add host completeness and partial-package negative fixtures.
: 4. Bind one representative MoE layer to a slot and execute a focused chunk.
: 5. Compare observed router IDs and top-k cardinality with a same-graph
  non-rolling control.

Acceptance criteria
: - [ ] Each MoE descriptor contains every routed-expert tensor exactly once
      and excludes router, shared experts, attention, and mutable state.
: - [ ] Package bytes and member counts match T-GPLR-00 inventory.
: - [ ] A representative MoE layer executes only after its matching generation
      is ready and rejects a partial package fixture.
: - [ ] Router-selected IDs and top-k cardinality match the control exactly.
: - [ ] Dense Qwen3.8 descriptor and parity regressions remain green.

Required validation
: - Host MoE completeness/negative CTests -> pass.
: - Focused one-layer CUDA proof through the GPU mutex -> exit 0 with matching
  router IDs.
: - Qwen3.8 reference regression -> pass.

State update
: Record package shape, bytes, route parity, repacking behavior, and blockers.

Escalate before implementation if
: Expert tensors require per-load repacking that cannot be reused for all 512
  rows, or exact router IDs cannot be observed without changing arithmetic.

### T-GPLR-03: Complete a 512-token MoE prefill
Status: done (2026-09-05)
Depends on: T-GPLR-02
Commit: pending (no commit authorized)

Evidence (2026-09-04/05)
: - 512-token layer-major MoE prefill through all 40 trunk layers, Q8 K/V,
    ubatch 512, 4 slots: exit 0, `check-telemetry: OK`, PP `graph_builds == 0`
    (TG phase also 0). Exactly 40 PP-phase `copy_submit` events (one per MoE
    layer, generations 1-10 over the 4-slot window: 74 x 452,984,832 B +
    6 x 522,190,848 B matching the T-GPLR-00 inventory; no partial or hidden
    copies). 40 `route` events with `top_k=8`; the 4-slot rolling run and a
    16-slot resident run produce IDENTICAL per-layer route-hash sequences.
    Greedy ID agreement: layer-major vs batch-major vs 16-slot resident all
    produce first greedy ID `220` on the deterministic synthetic prompt.
    Partial-package negative (`--fault-drop-member`) still fails loudly with
    zero copies. Traces: `build/llama-cuda/q36-moe-pp512-{rolling,resident16,
    batchmajor}.jsonl`.
  - Dense Qwen3.8 regression (`layer-streaming-layer-major-1024` pair) passed
    with greedy ID 286 after the T-GPLR-03 code changes.
  - Real-prompt parity control (the acceptance criterion that was still open
    after 2026-09-04): prompt "The quick brown fox jumps over the lazy dog".
    Two prior false alarms were identified and closed with evidence:
    (1) the earlier `llama-cli` control reported `<think>` because this
    vendored `llama-cli` is a chat client (it spawns/serves `/v1/chat/
    completions`), so the prompt was chat-template-wrapped and sampled the
    assistant-turn prefix, not a raw completion; `<think>` is one special
    token 248068 (verified with host-only `llama-tokenize`, which also shows
    the raw prompt is 9 tokens with no BOS). (2) An interim runner reading of
    271 (`\n\n`) came from a 10-token padded prompt (prior trace
    `run_start.pp_tokens = 10`): the PP-logits accessor reads the LAST row,
    which was the pad row, not `dog`. Re-running the runner on the exact
    9-token prompt produced PP-logits first greedy ID 13.
  - Matched raw-completion control (GPU mutex, same build, `-ngl 99`,
    `--cache-type-k/v q8_0`, temperature 0): `llama-server` `/v1/completions`
    with `n_probs=5` reports first token id 13 (`.`) at logprob -1.138, with
    198 (`\n`) -1.643, 369 (` is`) -2.018, 271 (`\n\n`) -2.274, 248046 -3.17.
    Streaming layer-major PP-logits argmax on the same 9-token prompt: id 13.
    Exact first-greedy-ID parity with the normal path is therefore
    demonstrated. Runner trace: `build/llama-cuda/q36-moe-prompt-ctl.jsonl`
    (40 PP copies, 40 routes, `graph_builds == 0` both phases, telemetry OK).
    Control response: `build/control-response.json`; server log
    `out/q36-control-server.log`; TG after the parity PP continues with id
    5388 (` dog`) — the unpicked continuation, consistent with greedy argmax
    13 feeding the next decode.
  - Residual: route-union hashes are prompt-dependent (the 512-token synthetic
    run and the 9-token real prompt legitimately differ); route parity was
    checked between same-prompt runs (4-slot vs 16-slot identical). The
    one-claim boundary stands: first-ID parity only, no quality claim.

Goal
: Process one 512-token chunk through every Qwen3.6 logical layer while each
  MoE layer consumes one complete expert-bank package copied once for that
  layer pass.

Scope
: Layer-major Qwen3.6 execution, deterministic package lookahead, existing
  activation slabs and graph sets, Q8 K/V, MTP disabled, route/package
  telemetry, focused normal-path parity, and state docs. Use a modest context
  capacity first so this task tests correctness rather than the 262K budget.

Non-goals
: The deliberate 262K allocation, sparse expert admission, TG optimization,
  MTP execution, policy auto-selection, or speedup claims.

Design and failure contract
: Build one 512-token prompt chunk, process logical layers in model order, and
  make each complete expert bank ready before its MoE graph executes. Dense or
  recurrent layers use the adapter-declared package policy. Router output is
  observational telemetry only. A slot remains protected through the final
  consumer event. No timed graph construction, partial package, expert
  substitution, or CPU fallback is allowed in the experimental run.

Implementation steps
: 1. Extend the common layer-major executor to the Qwen3.6 adapter's graph and
  state variants without adding Qwen3.6 branches to slot lifecycle code.
: 2. Prebuild required 512-token graph sets and validate package capacities.
: 3. Add sampled route-union and exact-ID telemetry with no routing feedback.
: 4. Run a normal llama.cpp control for the same deterministic prompt and
  compare the agreed first greedy ID plus sampled router IDs.
: 5. Add positive lifecycle and stale/partial-generation negative fixtures.

Acceptance criteria
: - [x] The run executes all model trunk layers in order with one 512-token
      chunk and Q8 K/V.
: - [x] Every MoE layer records exactly one complete expert-bank copy and no
      package is overwritten before its final consumer.
: - [x] Sampled router IDs and top-k cardinality match the normal-path control.
      (Same-prompt 4-slot rolling vs 16-slot resident: identical route hashes
      and top_k=8. The normal path exposes no per-layer routing, so router
      parity is proven between residency configurations; see evidence.)
: - [x] The first greedy ID matches the control and all checked outputs are
      finite; no broader quality claim is made. (Real prompt: both paths
      produce ID 13 for the first token after the 9-token prompt.)
: - [x] Timed prefill reports `graph_builds == 0` and telemetry validation
      passes.
: - [x] Qwen3.8 reference regressions remain green.

Required validation
: - Build `llama-layer-streaming` -> exit 0.
: - Host package/lifecycle/telemetry CTests -> pass.
: - Through `tools/with-gpu-lock.ps1`, run Qwen3.6 with `--pp-tokens 512
  --ubatch 512 --pp-schedule layer-major --kv-cache q8 --no-mtp` and the
  adapter's forced expert-bank policy -> exit 0 and validate telemetry.
: - Run the focused normal-path parity control under the same GPU lock ->
  matching sampled router IDs and first greedy ID.
: - Qwen3.8 layer-major-1024 reference target -> pass.

State update
: Record exact command, context size, package copies, route coverage/parity,
  output parity, graph builds, peak VRAM, and residual risks.

Escalate before implementation if
: Qwen3.6 state cannot be advanced safely in layer-major order, the normal-path
  control cannot expose comparable routing without arithmetic changes, or the
  model requires a package larger than available VRAM.

### T-GPLR-04: Prove the 262K Q8 out-of-core configuration
Status: done (2026-09-05)
Depends on: T-GPLR-03
Commit: pending (no commit authorized)

Goal
: Complete the 512-token Qwen3.6 prefill with context capacity 262,144 and Q8
  K/V while telemetry proves that full model residency cannot fit and only the
  bounded layer-package window occupies pageable weight VRAM.

Scope
: Exact capacity preflight, `--ctx-size 262144`, Q8_0 K/V allocation, one or
  two package slots selected by fit, 512-token layer-major prefill, structured
  byte-class telemetry, and a focused control/parity check.

Non-goals
: Processing a 262,144-token prompt, MTP, TG performance, hiding an allocation
  failure by shrinking context, adding CPU fallback, or claiming throughput
  improvement.

Design and failure contract
: Before CUDA allocation, report full-model bytes and runtime classes. After
  CUDA context setup but before timed prefill, report observed free VRAM and
  actual static, slot, Q8 KV, recurrent/state, activation, graph/input, and
  workspace allocations. The run is accepted only if
  `full_model_plus_runtime > physical_vram` and the bounded configuration
  fits. If the full configuration unexpectedly fits, enforce and report an
  explicit residency budget below full-model bytes. If the bounded
  configuration cannot fit, emit a structured preflight failure and return to
  planning; do not reduce the 262,144 context silently.

Implementation steps
: 1. Thread explicit context capacity independently from actual PP token count.
: 2. Extend preflight and `run_start` with complete byte-class accounting and
  a configurable experimental residency budget.
: 3. Allocate Q8 K/V for 262,144 tokens and the minimum safe package-slot
  count, preserving a recorded headroom reserve.
: 4. Run the 512-token layer-major prefill and validate package generations,
  route parity, output parity, and peak allocation.
: 5. Capture a negative insufficient-budget fixture before the real run.

Acceptance criteria
: - [x] Run metadata reports `ctx_size == 262144`, K/V type Q8_0, PP tokens
      512, ubatch 512, MTP disabled, and the selected package policy.
: - [x] Preflight proves full model plus runtime bytes exceed physical VRAM or
      reports an explicit sub-full-model residency budget.
: - [x] Actual telemetry accounts for static, slots, Q8 KV, state,
      activations, graph/input, workspace, reserve, free-start, and peak bytes.
: - [x] Only the bounded slot count holds pageable expert banks; no hidden
      full-model or all-expert device copy occurs.
: - [x] The run exits 0, validates, records one expert-bank copy per MoE layer,
      preserves sampled router IDs and the agreed first greedy ID, and reports
      `graph_builds == 0`.
: - [x] A deliberately insufficient residency budget fails before timed PP.

Evidence (2026-09-05)
: - Preflight/emission drift defect found and fixed: the post-init
    `byte_account` emitter applied MoE arithmetic to dense runs (per_layer
    counted resident, slot_window 0), producing `bounded_window` 25,745,786,880 B
    >= free VRAM and a checker failure on `q38ls-lm-1024`. The checker caught
    exactly the drift it was designed for. Both sites now share one helper,
    `ls_bounded_window_for_policy()`, so they cannot diverge again; dense
    policy correctly reports `per_layer=0`, `slot_window = slots x
    dense_largest`. Fresh dense 262K reference: bounded 12,405,569,280 B < full
    35,196,854,272 B < free 22,889,365,504 B, `check-telemetry: OK`.
  - `run_start` now carries explicit `ctx_size` (additive field, default
    suppressed); `ctx_size=262144` asserted in the acceptance trace.
  - Acceptance run (GPU mutex, Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf):
    `--ctx-size 262144 --kv-cache q8 --pp-tokens 512 --ubatch 512
    --pp-schedule layer-major --layer-slots 2 --no-mtp --reserve-mib 512
    --residency-budget-mib 7000 --prompt "The quick brown fox jumps over the
    lazy dog" --greedy --tg-tokens 1` -> exit 0.
  - Out-of-core proof: full model + KV + state 24,295,318,016 B vs bounded
    2-slot window 6,517,385,728 B (matches the T-GPLR-00 inventory exactly);
    byte classes: static 957,530,112 + per_layer 1,597,483,520 + slot_window
    1,044,381,696 (2 x 522,190,848) + KV 2,852,126,720 + state 65,863,680 B;
    act slabs 8,388,608 B, reserve 536,870,912 B, peak VRAM 7,625.56 MiB.
  - Parity: PP-logits first greedy ID 13 and TG continuation 5388 match the
    T-GPLR-03 control; all 40 route-event hashes identical between the 2-slot
    262K run and the 4-slot 9-token control; `graph_builds == 0` in both
    phases; `check-telemetry: OK` (567 events).
  - Negative fixture: `--residency-budget-mib 5000` aborts before the timed PP
    with `bounded 2-slot window 6517385728 B exceeds the residency budget
    5242880000 B` (exit 1, zero timed tokens).
  - Regression: all 38 layer-streaming CTests pass (dense Qwen3.8 reference
    included, traces regenerated with the fixed emitter); default-off
    `llama-cli`/`llama-server` targets build exit 0.
  - Caveats: full model 24.3 GB fits physical VRAM (25.65 GB), so the run is
    proven by the explicit sub-full budget plus the bounded-window arithmetic,
    not by physical non-fit; cold mmap/file-cache state uncontrolled (Windows);
    normal-path routing not observable per layer, so router parity is proven
    between residency configurations, not against the server.

Required validation
: - Host preflight arithmetic and insufficient-budget CTests -> pass.
: - Through `tools/with-gpu-lock.ps1`, run the fixed MoE acceptance command
  shape from this plan with `--ctx-size 262144 --kv-cache q8 --pp-tokens 512
  --ubatch 512 --pp-schedule layer-major --no-mtp` -> exit 0 and telemetry
  validation passes.
: - Focused normal-path parity control under the same mutex -> agreed parity
  fields match.
: - Qwen3.8 reference suite and default-off `llama-cli` build -> pass.

State update
: Record exact allocation classes, proof of non-fit, selected slot count,
  command, output/route parity, timings, peak VRAM, and file-cache/GPU caveats.

Escalate before implementation if
: The bounded configuration does not fit at context 262,144, Q8 cache support
  is unavailable for this architecture, or required byte classes cannot be
  measured reliably before timed execution.

### T-GPLR-05: Publish capability and regression closure
Status: done (2026-09-05)
Depends on: T-GPLR-04
Commit: pending (no commit authorized)

Goal
: Every loaded model receives a stable layer-residency support result, and the
  repository records the tested dense and MoE support matrix without claiming
  universal profitability.

Scope
: Private adapter registry, forced `off`, `dense-layer`, and
  `moe-expert-bank` policies, conservative capability reporting, unsupported
  and capacity fixtures, full focused regressions, and governing docs.

Non-goals
: Public API stabilization, automatic speed optimization, non-CUDA backends,
  TG expert prediction, additional model downloads, or upstream material.

Design and failure contract
: Capability probing does not allocate CUDA memory or alter normal model
  placement. A forced supported policy runs only after full validation. A
  forced unsupported policy fails with a stable reason. An optional `auto`
  mode may select only adapter-declared policies; if evidence is insufficient,
  it selects `off` and explains why. Silent mid-run fallback is forbidden.

Implementation steps
: 1. Register the extracted Qwen dense and Qwen3.6 MoE adapters.
: 2. Add stable supported/unsupported reason codes and policy telemetry.
: 3. Add an unsupported-model fixture and the complete capacity/refusal matrix.
: 4. Run host tests, focused CUDA adapter checks, Qwen reference regression,
  and default-off build.
: 5. Update the SDD, plan, environment guide, and current state with the tested
  support matrix and exact evidence.

Acceptance criteria
: - [x] Any loaded architecture can be probed without CUDA allocation and gets
      a deterministic adapter/policy result.
: - [x] Qwen3.8 dense and Qwen3.6 MoE report only their validated policies.
: - [x] Unsupported architecture and insufficient-capacity fixtures refuse
      before timed execution with stable reasons.
: - [x] Documentation distinguishes supported, unsupported, correct-but-slower,
      and unmeasured cases.
: - [x] All task-owned correctness, lifecycle, routing, telemetry, and default-
      off regressions pass.
: - [x] No speedup or broad model-support claim exceeds measured evidence.

Evidence (2026-09-05)
: - Host-only capability probe `llama_ls_probe_capability(path)` (GGUF header
    KV only: no CUDA enumeration, no model load, no normal-placement change)
    plus runner mode `--probe-capability` printing deterministic JSON
    (arch/adapter/policy/reason/recommended). Unit-tested on synthetic
    fixtures inside `test-layer-streaming-model-inspect` (qwen35 -> dense-layer,
    qwen35moe -> moe-expert-bank, `llama` arch -> off with stable reason,
    missing file -> off with reason).
  - Policy modes wired through the probe: `auto` selects ONLY an
    adapter-declared policy and refuses unsupported architectures with the
    stable reason (exit 2, before `llama_backend_init`); forced `off` runs the
    normal llama.cpp path inside the runner (no tensor-buft overrides, no
    streaming init, batch-major `llama_decode`) and records an audit telemetry
    record (run_start with `policy":"off"` + `capability_reason`, then run_end).
    `llama_backend_init` is deferred after policy resolution so every refusal
    path is strictly host-only.
  - Normal-path parity fixture (GPU mutex, forced off, MoE model, 9-token
    prompt): PP-logits first greedy ID 13 and first generated ID 5388 —
    identical to the streaming layer-major run. This closes the "correct but
    not proven equal to the normal path" caveat from T-GPLR-03/04.
  - Defect found and fixed during this task: the first off-mode parity read
    used `llama_get_logits_ith(ctx, 0)`, which resolves to the FIRST batch
    token's row (`output_resolve_row(i >= 0) = output_ids[i]`), reporting 2614
    instead of 13. The off-vs-streaming parity check caught it immediately;
    the correct read is index -1 (last output row).
  - Refusal fixtures: `--residency dense-layer` on an unsupported `llama`-arch
    GGUF and `--residency auto` on the same fixture both refuse host-only with
    stable reasons (new CTests `layer-streaming-refuse-unsupported-{dense,auto}`
    + fixture generator `make-unsupported-fixture` + probe CTest; all pass in
    0.4 s with no GPU). Insufficient-capacity refusal was already proven by the
    T-GPLR-04 negative fixture (aborts before timed PP).
  - Validation: `tools/run-tests.ps1 -Suite host` 13/13 (suite now includes
    the probe/refusal tests); full layer-streaming suite 42/42 under the GPU
    mutex; default-off `llama-cli`/`llama-server` build exit 0.
  - `auto` on supported MoE resolves `moe-expert-bank` and runs to completion
    (exit 0) at the 262K capacity shape. No speedup or broad support claim is
    made anywhere; the matrix below states exactly what was measured.

  - Post-audit correction (2026-09-05): `auto` on an unsupported architecture
    now runs `off` and explains why on stderr (exit 0, SDD 8) instead of
    refusing; only FORCED policy/arch mismatches refuse host-only (exit 2).
    Closing validation: full layer-streaming suite 44/44 green under the GPU
    mutex, including the new off-audit checker fixtures; the 16 run tests now
    pass `--allow-full-residency` explicitly, removing ambient-VRAM flakiness.
    Post-refactor MoE parity proof (off-first-class): forced-off vs
    layer-major streaming on the final binary produced identical first 3
    generated IDs (19241 13 561) on the same 12-token prompt; both traces
    pass --check-telemetry.
    Long-context PP evidence (post-plan): 50K/100K/262K-token layer-major
    prefills measured 717.8/740.7/745.0 tok/s (wall 69.7/135.0/351.9 s,
    38 evictions, graph_builds 0); the 262K run exposed and fixed a
    graph-input-buffer sizing defect (+1 MiB/chunk headroom, host-side).

Required validation
: - `tools/run-tests.ps1 -Suite host` -> all host tests pass.
: - Adapter-specific CUDA checks through `tools/with-gpu-lock.ps1` -> pass.
: - Required Qwen3.8 reference suite -> pass.
: - Default-off `llama-cli` build -> pass.
: - `git diff --check` on task-owned files -> clean.

State update
: Mark this plan complete only when `docs/currentstate.md` records the support
  matrix, exact commands/results, negative cases, allocation evidence, and
  residual risks. Record commit hashes only when the owner authorizes commits.

Escalate before implementation if
: Capability registration would alter normal model loading, a required
  adapter needs a new dependency, or a correctness failure cannot be isolated
  within the private experimental path.

### T-GPLR-06: Amortize expert-bank passes across TG bursts (MTP draft + layer-major verify)

Status: ready (2026-09-06; owner approved the redesign-the-draft-leg path)
Depends on: T-GPLR-05
Commit: pending (no commit authorized)

Goal
: Determine whether MoE token generation can escape the per-token expert-bank
  pass. Measured baseline (2026-09-06): TG runs 0.26-0.52 tok/s because every
  generated token re-streams all 40 routed-expert banks (~18 GB/token, 98%
  transfer, compute 21-50 ms/token). A k-token burst verified through the
  existing layer-major controller would pay one bank pass per k tokens, so the
  falsifiable hypothesis is: burst TG at k=4 reaches at least 2x the sequential
  TG baseline on the same prompt, with accepted-prefix greedy parity.

Scope
: Runner-level opt-in flag (`--tg-burst k`, k>=2) implementing an MTP
  draft-then-verify loop after PP: draft k-1 tokens with the MTP head running
  directly on its GPU-resident weight package (no begin_handoff call), then
  verify all k tokens in one layer-major pass so bank loads amortize across
  the burst; accept the greedy-consistent prefix; repeat until --tg-tokens is
  reached. CUDA-only, inside the existing experimental path; telemetry records
  draft, verify, and accept phases with transfer/compute/wait kept distinct.

Feasibility facts (verified in code/trace, 2026-09-06)
: Under moe-expert-bank only ffn_*_exps tensors are pageable, so the MTP
  members (qwen35_mtp_package, blk.40) are already GPU-resident where the
  handoff path refuses. The decoder graphs emit t_h_nextn (h_nextn, normalized
  final hidden) and the streaming controller propagates it across graph
  results; the streaming graph builder already handles the MTP context type.
  The blocked probe therefore localizes the defect to the runner's TG path
  unconditionally calling begin_handoff for --mtp, not to a missing mechanism.

Non-goals
: Heuristic expert prediction, weight rotation, quantization changes, multi-GPU,
  any change to PP semantics, the `off` path, routing, or model arithmetic.
  Acceptance is exact greedy-prefix equivalence to sequential TG, never a
  quality claim.

Design contract
: Logical layer identity and device slots stay separate; KV remains owned by
  the context; a burst's verify pass is a layer-major PP pass and obeys every
  slot invariant, including loud failure on stale slots or missing packages.
  A draft/verify mismatch is expected behavior (speculative rejection), not an
  error; it must be visible in telemetry (accepted/drafted counts).

Implementation steps
: 1. Add `--tg-burst k` parsing; gate it host-only against `--residency off`
     and non-MTP configurations; allow `--mtp` without `--handoff` when the
     policy keeps the MTP package resident (moe-expert-bank), refusing loudly
     elsewhere.
  2. Loop in the runner: draft k-1 with the MTP context graph bound to the
     resident package fed by the verify pass's t_h_nextn, layer-major verify
     (k), greedy prefix acceptance, KV rollback on mismatch (length only, no
     arithmetic).
  3. Extend telemetry with draft/verify/accept event classes; extend the
     checkers to validate them.
  4. Measure k in {2, 4, 8} on the 512-token prompt, 12 accepted tokens,
     against the sequential TG baseline.

Acceptance criteria
: Accepted-token IDs match sequential greedy TG exactly on all tested k.
  At k=4, wall-clock TG tok/s >= 2x the measured 0.4 tok/s baseline on the
  same machine state; otherwise the hypothesis is falsified and the task
  records the negative result instead of a mechanism.
  Every failure (draft head absent, KV rollback impossible, slot error) aborts
  loudly with structured telemetry.

Focused validation
: `tools/run-tests.ps1 -Suite host`; layer-streaming CTest set under the GPU
  mutex; one Qwen3.8 dense reference regression; the k-sweep above with traces
  `build/gplr06-burst-k*.jsonl` all passing `--check-telemetry`.

State update
: Record baseline vs burst tok/s, accept rate per k, and the go/no-go decision
  for a full TG-residency mechanism task in `docs/currentstate.md`.

Escalate before implementation if
: The MTP draft head cannot generate while expert banks are slot-managed, KV
  rollback cannot be done without altering context internals outside the
  experimental path, or the owner judges TG out of scope for this plan cycle.

Probe evidence (2026-09-06, pre-implementation, reversible): a 512-token
PP + 13-token --mtp --mtp-draft 4 --handoff run on the MoE shows the
draft head loads (blk.40 enabled) but handoff migration fails loudly:
begin_handoff is not supported for the moe-expert-bank residency policy
(TG decode -3; trace build/gplr06-mtp-probe.jsonl, run_end exit 1). The
MoE acceptance rate is therefore unmeasurable with the current binary
and the first escalation condition is triggered by evidence. Both paths
forward are owner decisions: (a) extend begin_handoff to moe-expert-bank
(new capability beyond this plan), or (b) redesign drafting to consume
the verify pass's last-layer hidden state so no handoff is needed
(mechanism change within this plan, requires re-contracting the draft
leg). Do not mark ready until the owner picks one.

Milestone (2026-09-06, owner approved redesign; step 1 + measurement done):
--mtp is now legal without --handoff under moe-expert-bank (single gate
moved after policy resolution; dense-layer still requires handoff). The
first MoE MTP run completed: batch-major PP 512 + --mtp --mtp-draft 4,
13 TG tokens, exit 0, drafted 16 accepted 9 (rate 0.562), and greedy
IDs exactly match the sequential no-MTP baseline on the same prompt
(build/gplr06-{seq-baseline,mtp-resident}.jsonl). Wall did NOT improve
(0.474 vs 0.493 tok/s) because the existing loop verifies serially - one
full bank pass per draft - confirming the remaining work is exactly
implementation step 2: the layer-major batch verify that amortizes one
bank pass across k drafts.

Acceptance measurement over 128 generated tokens (2026-09-06, same
512-token prompt, all legs exit 0, greedy IDs identical across all
three): sequential 0.522 tok/s (5120 evictions); MTP draft 4 drafted
188 accepted 90 (rate 0.479, 47 rounds, 1.91 accepted/round); MTP
draft 8 drafted 344 accepted 86 (rate 0.250, 43 rounds, 2.00
accepted/round). Acceptance decays with draft depth, so accepted
tokens per bank pass saturates near 2. Projected batch-verify wall
(one bank pass per round): ~90 s at k=4 and ~82 s at k=8 vs the
serial 245 s, i.e. ~2.7-3.0x and the 2x hypothesis at k=4 is
reachable with margin. Step 2 remains the implementation gate; these
numbers set its expected result, they do not replace it.
