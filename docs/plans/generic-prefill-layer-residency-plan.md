# Generic Prefill Layer Residency Plan

Status: T-GPLR-00 through T-GPLR-08 and T-GPLR-10/T-GPLR-11 done
(2026-09-04/05/09; T-GPLR-08's prefetch gate falsified -- see its status
note; T-GPLR-10 MoE handoff closed with parity/speed/answer-gate evidence;
T-GPLR-11 tail-ubatch gap closed); T-GPLR-09
PARKED (owner decision 2026-09-09 -- two-key door: privilege blocker plus
falsified premise; reopen triggers in its status note).
Remaining program items: endurance ladder rungs 50k/100k/200k
(owner-invoked), the T-GPLR-08 attribution question, and a second
architecture passing the acceptance gates (owner-gated product decision).
Commit record (2026-09-09, owner-authorized): vendored-tree changes in
third_party/llama.cpp 099a27a7e (T-GPLR-08/10/11); parent docs/tools in
84be194 (recorded T-GPLR-08/10/11 completion, gitlink bump) and 7bfe83b
(T-GPLR-10 evictfix follow-up). No push authorized; both repos local.

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
- Worktree state (2026-09-09): all task code is committed (vendored tree
  clean at 099a27a7e; parent docs/tools committed through 7bfe83b).
  Generated build/trace/telemetry output stays out of version control;
  every task still stages or commits nothing beyond its task-owned files
  without owner authorization.

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

Implementation serialization (one ready task at a time): T-GPLR-06/07
closed 2026-09-09; T-GPLR-08 (prefetch probe, hypothesis falsified) and
T-GPLR-10 (MoE handoff) closed 2026-09-09; T-GPLR-11 closed 2026-09-09.
T-GPLR-09 (host-RAM residency tier) is PARKED (owner decision 2026-09-09:
a two-key door -- the privilege blocker plus the falsified T-GPLR-08
premise; reopen triggers in its status note). If reopened, it reuses
T-GPLR-08's range walker and telemetry plumbing and makes the rolling
prefetch unnecessary (all banks resident), with T-GPLR-08 remaining the
fallback tier and the probe for the declined case.
Remaining program items are listed in the header status above.
```

### T-GPLR-00: Inventory the MoE artifact and capacity
Status: done (2026-09-04)
Depends on: none
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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

Status: done (2026-09-09 owner review closed: acceptance met -- k=4 wall 2.58x
>= the 2x gate (d8 2.97x; code-text d16 6.91x vs the measured serial);
greedy-prefix parity under the recorded knife-edge standard with top-2
margins; every failure loud with structured telemetry; the two en-route
defects (n_outputs_max MTP clamp, graph-input buffer sizing) fixed and
recorded). Evidence in docs/currentstate.md.
Design note: the burst verify is realized as k prebuilt width-k mode-1 graph
sets driven through the per-ubatch streaming executor (one bank pass per
layer, k sequential per-layer sub-launches), because the recurrent snapshot
ring is written per decode call, so exact length-only rollback requires one
call per verified token position. This satisfies the plan's layer-major
verify intent with exact greedy-prefix parity.
Depends on: T-GPLR-05
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

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

Step 2 IMPLEMENTED (2026-09-06): batch verify decodes the frontier plus
all drafts in ONE width-(1+d) ubatch per round (one expert-bank pass),
commits the matching-gold prefix, and rolls rejected rows back via exact
KV seq_rm plus the recurrent snapshot ring (n_rs_seq = mtp_draft);
persistent prebuilt width-k phase-1 graph sets keep graph_builds == 0.
Measured on the 128-token draft-4 config vs the 245.4 s serial baseline:
95.1 s wall (2.58x, target ~90 s), exit 0, graph_builds 0, 47 rounds
(3 full / 44 partial, avg accepted/round 1.70, 44 rollback rounds).
Greedy-parity verdict (owner decision 2026-09-06: not bit-exact across
ubatch shapes; divergence must be MEASURED, not assumed benign): streams
differ from idx 52 and re-join the serial stream thereafter; diagnosis
instrumented both paths with top-2 logit margins and runner-up tokens.
The flip row is the same logical decode in both paths (identical token,
position, and committed history): serial picks 6365 over 1970 by margin
0.0037 - the smallest margin of all 128 serial steps; the width-5 batch
shape picks 1970 over 6365 by 0.14 (Q4_K_S shape-noise scale). Every
greedy row comparable before the divergence agrees on BOTH gold and
runner-up; after the flip both streams stay locally coherent and re-join
twice, ruling out state/wiring corruption (which cascades). Verdict:
single knife-edge argmax flip from quantized-weight shape noise, correct
behavior on both paths; reruns are bit-identical (deterministic).
Traces: build/gplr06-{seq128-margins,batchverify-d4-v2}.jsonl + logs.

Draft-8 batch verify measured (2026-09-07): 82.6 s wall (1.55 tok/s), exit
0, graph_builds 0, check-telemetry clean - 2.97x over the serial baseline
and ahead of the ~82 s projection. 40 rounds (2 full / 38 partial, avg
accepted/round 2.17 vs d4's 1.70): under batch verify, deeper drafts DO
pay off - rounds shrink faster than per-row compute grows, overturning
the serial-era "k=8 pointless beyond k=4" note. The 2x hypothesis is
exceeded (2.97x measured at k=8). d8 vs serial-d8 diverges at the same
idx-52 knife-edge pair (6365/1970); d8 vs d4 diverges once at idx 68
(margin 0.059) and rejoins - the same measured shape-noise mechanism at
each width boundary, three streams, no structural defect. Trace:
build/gplr06-batchverify-d8.jsonl + log.

Draft-32 boundary probe (2026-09-07): 96.1 s (1.33 tok/s), exit 0,
graph_builds 0, check-telemetry clean, peak VRAM 7.42 GB - the depth
frontier is crossed: d32 LOSES to d8 (82.6 s). Acceptance horizon is 7
(0 accepted at depth >= 8 in all 43 rounds; 84 accepted/1176 drafted =
0.071/draft), so the extra 24 drafted rows are pure serial draft-chain
cost. Rollback machinery is exercised at full depth (6 rounds accepted
0 with a 32-deep ring rewind; all first-16 IDs match every other
config). Two code fixes landed: prebuild variant applies now advance
positions strictly (the width loop reused one position and failed the
KV contiguity check), and the activation slab is sized for the
cumulative 2*sum(w)*n_embd_io TG pair demand over widths 1..(1+n_rs_seq)
(was width-1-only; d8 fit by luck, d32 exhausted the buffer). Trace:
build/gplr06-batchverify-d32.jsonl + log.

Coding-prompt depth sweep (2026-09-07, calculator-spec prompt via
--prompt-env, 1024 PP tokens, ctx 2048, tg 128, batch-major): all three
legs exit 0, check-telemetry clean, graph_builds 0. d8: 35.5 s TG
(3.60 tok/s), 17 rounds, accept 110/135 = 0.815. d16: 25.7 s (4.98
tok/s), 12 rounds, accept 115/177 = 0.650. d32: 18.2 s (7.02 tok/s),
8 rounds, accept 119/240 = 0.496 - tok/s still RISING at d32, so on
structured code text the synthetic-prompt depth horizon (7) dissolves:
long full-accept runs appear at every depth (12x8 at d8, 4x16 at d16,
3x32 at d32). Caveat: d8 and d32 fell into a greedy 2-token repetition
loop (198,12 bullet loop; no repetition penalty in this harness) while
d16 escaped via a round-0 row-0 knife-edge flip (1749 vs 198, margin
0.089) onto real HTML/JS code text - the acceptance profile therefore
mixes loop and code text, and a no-loop rerun (rep penalty or sampled
decode) is required before generalizing. Cfg note: --allow-full-
residency preflight warns "full model WOULD FIT" on this host and the
runs stay on the bank-paging path via the documented override; peak
VRAM d8 5.83 / d16 6.36 / d32 7.42 GB. Traces:
build/gplr06-codeprompt-d{8,16,32}.{jsonl,log}.

Coding-prompt depth sweep, repetition penalty (2026-09-07, same
calculator prompt/1024 PP/ctx 2048/tg 128/batch-major + --rep-penalty
1.3, last-n 64): the loop is dead at every depth - max period-2 run
1/2/1 (vs 25+ unpenalized) and all legs emit real code (d16/d32 open
<!DOCTYPE html> <html> <style>). Runner change: minimal --rep-penalty
F / --rep-last-n N flags replicating llama_sampler_penalties_apply's
penalty_repeat semantics (positive logits /F, non-positive *F, ring of
last-N accepted tokens with count map), applied identically at all four
greedy pick sites - initial frontier, draft-head chain (ring = true
ring + hypothetical drafts), batch-verify rows (ring = true ring +
row-input prefix, including the row's own input token), serial TG -
so acceptance stays meaningful (the draft must predict what the
penalized target picks). Results: d8 76.4 s TG (1.68 tok/s), 38
rounds, accept 89/298 = 0.299. d16 36.1 s (3.54 tok/s), 17 rounds,
accept 110/258 = 0.426 - the measured optimum. d32 76.1 s (1.68
tok/s), 36 rounds, accept 91/1035 = 0.088 - acceptance collapses at
depth 32 because the draft ring fills with its own hypothetical
penalized tokens, so deep drafts predict anti-repetition tokens the
target rejects. The unpenalized sweep's "rising at d32" was loop
contamination as flagged; with the loop removed, d16 is the code-text
optimum at 3.54 tok/s = 6.8x serial. Serial baseline (0.52 tok/s)
unchanged. Exit 0, check-telemetry clean, graph_builds 0 everywhere.
Traces: build/gplr06-codeprompt-p13-d{8,16,32}.{jsonl,log}.
Serial no-MTP leg on the same prompt (2026-09-07, penalty 1.3, natural 885-token PP): 249.69 s TG (0.513 tok/s), real HTML output, loop-free (max period-2 run 1) - the first measured code-text serial baseline; d16 36.12 s => 6.91x true MTP speedup on code (the prior 6.8x inherited the synthetic-prompt baseline). Caveat: batch legs pad PP to 1024 with PAD tokens (mirror requires exact ubatches), serial runs natural 885, so the streams diverge at idx 0 by construction (pad-cell frontier); TG per-token bank geometry is identical, so the timing ratio is clean. Trace: build/gplr06-codeprompt-serial-p13.{jsonl,log}.

### T-GPLR-07: Semantic-QA length ladder (programmatic needle grading)

Status: done (2026-09-09; acceptance closed -- smoke tier 50/512 with the
default gate at 512; 4k and 24k rungs green; 512/4k strict dedicated-JSONL
re-verification complete: gplr07-qa-512.jsonl and gplr07-qa-4k.jsonl both
check-telemetry OK; dense Qwen3.8 regression re-run green
(layer-streaming-layer-major-1024 2/2, greedy ID 286). Owner approved design
2026-09-06: programmatic grading only; smoke tier pinned at the 512-token
anchor; ladder lengths run individually on owner request). Evidence 2026-09-07/09: smoke tier 50/512, 4k ladder rung,
and 24k ladder rung (owner-invoked) all PASS (all 5 items each),
negative control verified live, 24k also green on the strict dedicated-
--telemetry JSONL gate (build/gplr07-qa-24k.jsonl, check-telemetry OK exit 0,
byte-identical first-16 IDs across two runs). See docs/currentstate.md.
Runner flags --prompt-file/--tg-text (default-off) landed 2026-09-07; the
implemented contract (including the interaction decisions the plan left open)
is recorded in docs/currentstate.md.
Smoke wiring done (2026-09-07; ladder series reduced to 50/512/4k/24k/50k/
100k/200k and default gate narrowed to 512 on 2026-09-09 by owner request):
`tools/run-tests.ps1 -Suite qa-smoke` runs the smoke gate under the GPU
mutex with DEFAULT rung 512 on the 512-token anchor; the 50-token rung is
opt-in via `-SmokeLengths 50` (comma list; the quoted-string form silently
binds as one int and must not be used). 4k+ ladder lengths are manual
owner-invoked commands, never part of the suite. The 15/30 rungs were
removed from the ladder entirely; their 2026-09-09 live evidence is
historical (see docs/currentstate.md). Gate shape:
generator calibration -> runner with --prompt-file/--telemetry/--tg-text ->
strict --check-telemetry on the dedicated JSONL -> check-answer.py per-needle
grading + negative control on the runner's --tg-text generated-text file (no
trace re-parsing at grading time; --trace/--model grading remains supported).
Acceptance closed 2026-09-09: dedicated --telemetry JSONL traces now exist
for BOTH previously mixed-log-only rungs (gplr07-qa-512.jsonl and
gplr07-qa-4k.jsonl, both check-telemetry OK; the 4k answer also passes all
5 items with the negative control failing as designed), and the dense
Qwen3.8 regression re-run is green (layer-streaming-layer-major-1024 2/2,
greedy ID 286).
Depends on: T-GPLR-05
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

Goal
: Add an answer-correctness observable that survives to 200,000-token prompts,
  where first-greedy-ID parity cannot exist (--pp-no-logits is mandatory at
  long PP; a full per-position logits buffer at 200k would be hundreds of GB)
  and exact ID agreement is fragile under CUDA reduction order and Q8 KV.
  The test grades generated TEXT semantically with deterministic, host-only,
  falsifiable checks — never bit/byte/hash identity. Extends the T-GPLR-03
  parity evidence from one token to a whole answer, between residency
  configurations and across the length ladder.

Instrument shape (decided, per owner)
: - Grading is programmatic only: normalized substring/regex checks plus a
    must-fail negative control. No LLM judge in the gate; if one is ever
    wanted, it is a separately gated advisory experiment, not acceptance.
  - Smoke tier = {50, 512}-token runs on the 512-token anchor prompt; runs
    on every runner change (default gate: 512 only; 50 opt-in via
    -SmokeLengths). (The generator refuses targets below 15 tokens — the
    minimal meaningful K1 scaffold floor — so no shorter run exists.)
  - Ladder lengths {4k, 24k, 50k, 100k, 200k} are never run as a batch; the
    owner invokes one length at a time as an explicit GPU-mutex acceptance
    run.

Test design (why these items)
: - Primary anchors are EXTRACTION needles, not knowledge questions: facts
    planted in the prompt itself ("The archive access code is QX-7741"),
    duplicated at ~25/50/75/95% prompt depth. A failure at length then
    implicates the context path (KV, positions, recurrent state), not model
    quality. Codes/numbers are checker-robust (no synonym drift).
  - Knowledge items ("What is the capital of France? Answer with the city
    name only. Do not chatter.") are secondary flavor: they exercise short-
    format compliance but a miss says nothing about residency.
  - Multiple needles are planted in ONE prompt and answered in ONE short TG
    burst (one TG budget per run, ~16-32 tokens per needle): long-context
    runs must amortize their one expensive PP across many anchors.
  - Short-to-long with stop-on-first-failure diagnosis: fail at 10-50 ->
    task-owned (prompt format or checker); pass short, fail long ->
    context-path defect. The 10-50 tier runs FIRST and must pass before any
    long run is interpreted.

Checker contract
: - Normalize: strip `<think>` blocks and whitespace, casefold, strip
    punctuation, then case-insensitive substring/regex for each needle's
    expected answer (e.g. `QX-7741`, `Paris`).
  - Checker verdicts are per-item PASS/FAIL, aggregate PASS only when every
    enabled item passes, and are recorded in the run summary next to the
    telemetry verdict (check-telemetry OK remains a separate, required gate).
  - Negative control: one item whose expected answer is deliberately absent
    from the prompt must FAIL; a grader that cannot fail is not a grader.
  - Known-stable greedy IDs (e.g. the 512-token first ID 13) stay recorded as
    advisory cross-checks; they are not the acceptance gate at length.

Non-goals
: Quality claims of any kind, chat templating, sampling beyond greedy,
  LLM-as-judge in the gate, multi-GPU, non-CUDA backends, bit-identity
  claims. The ladder promises ANSWER stability, not token-ID stability; no
  speedup or equivalence claim is made anywhere.

Implementation steps
: 1. Additive runner flags, default-off, inside the experimental path:
     `--prompt-file PATH` (Windows argv/env limits are ~32 KB; a 200k-token
     prompt is ~800 KB of text, so file input is required) and `--tg-text
     PATH` (detokenize `generated` via `llama_token_to_piece` so the answer
     text is inspectable without a new tool).
  2. `tools/semantic-qa/`: needle-prompt generator (host-only, deterministic
     seed) emitting the anchor prompt text per length; checker (normalizer +
     per-item verdicts + aggregate + negative control) reading `--tg-text`
     output and the trace's `tg_tokens` event.
  3. Suite wiring: smoke targets in `tools/run-tests.ps1` (default run 512;
     50 opt-in via `-SmokeLengths 50`); ladder lengths (4k+) documented as
     individual owner-invoked commands.
  4. Measure the ladder once per length as acceptance runs, storing traces
     `build/gplr07-qa-<len>.jsonl` + `out/gplr07-qa-<len>.log`.

Acceptance criteria
: - Smoke tier: both lengths (50/512) produce check-telemetry OK and aggregate
    checker PASS on the 512-token anchor prompt, including the negative
    control failing as designed.
  - Ladder: each requested length runs exit 0 under the GPU mutex, passes
    check-telemetry, and reports per-needle verdicts; a long-length FAIL with
    short-length PASS is a loud, structured finding (never a silent fallback
    or another layer's data).
  - Dense Qwen3.8 reference regression still passes; no default-path build
    or behavior change (flags default-off).
  - Traces and verdicts recorded in `docs/currentstate.md` with model path,
    command, hardware, exit status, and telemetry path.

Focused validation
: `tools/run-tests.ps1 -Suite host`; smoke targets under the GPU mutex; the
  one requested ladder length; `--check-telemetry` on every produced trace.

State update
: Record per-length PASS/FAIL per needle, answer text, and any short-pass/
  long-fail diagnosis in `docs/currentstate.md`; remove superseded parity
  caveats only if the ladder actually closes them.

Escalate before implementation if
: The owner wants chat-template formatting (new runner surface beyond the
  experimental path), an LLM judge in the gate (dependency + nondeterminism
  policy), or ladder lengths re-batched (GPU budget decision).

### T-GPLR-08: One-layer-ahead PrefetchVirtualMemory probe (streaming tier)

Status: done (2026-09-09; implemented and probed). Primary wall/transfer
gate FALSIFIED per the contract's falsification clause -- N=1 vs N=0:
352.9 s vs 350.7 s wall (+0.6%, noise), 328.3 s vs 326.4 s transfer;
39/39 prefetches succeeded with 0 errors yet the fault-driven feed did not
rise. Recorded as an instrumentation result with full per-bank telemetry
(docs/currentstate.md). The flag remains available (default off) for
re-probing under different host memory conditions; parity criterion passed
(512 + 50 rungs on the prefetch binary).
Depends on: T-GPLR-05
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

Goal
: While the GPU computes layer L, a dedicated worker thread issues
  `PrefetchVirtualMemory` over layer L+1's bank member ranges, so the
  page-fault-driven memcpy in `load_layer` reads resident pages. Converts the
  measured 52 MB/s effective feed (262K) into bulk large-I/O prefetch. Device
  evidence (2026-09-07): MSI M461 2TB NVMe, 2,728 MB/s unbuffered qd1 / 7,745
  MB/s buffered sequential reads of the model file — ~50x headroom over the
  fault-path rate, so the falsifiable question is how much of that headroom
  one-ahead prefetch recovers on the real workload.
  Warm-cache leg (2026-09-07, recorded in currentstate.md): with the whole
  21.39 GB file resident in standby before AND during the run, wall/per-layer
  copy times are identical to baseline — standby residency does NOT speed up
  the fault-driven memcpy (~200-215 MB/s per bank). But layer 0's copy runs
  at ~3.5 GB/s (0.13 s) because it rides the loader's one-shot map-time
  PrefetchVirtualMemory into the WORKING SET. So the fast path is working-set
  residency, which is exactly what a rolling per-bank prefetch populates;
  the acceptance gate below should be derived from the layer-0 rate (feed
  ~3.5 GB/s -> wall floor ~66 s, matching the warm-resident boundary traces
  at 3968 tok/s) rather than from the raw 52 MB/s.

Scope
: Windows-only experimental path; new flag `--pp-prefetch-ahead N` (default 0
  = today's behavior, byte-identical); probe runs at N=1; per-member
  `WIN32_MEMORY_RANGE_ENTRY` ranges (3 per MoE bank: ffn_gate/up/down_exps);
  one worker thread with a 1-deep queue; new telemetry events
  `prefetch_submit` / `prefetch_complete` per bank with duration and result;
  parity via the T-GPLR-07 smoke tier.

Non-goals
: TG-loop changes, batch verify (T-GPLR-06 owns TG), host-RAM residency
  (T-GPLR-09), multi-GPU, non-Windows/non-CUDA backends.

Design / failure contract
: - `PrefetchVirtualMemory` is a hint: a partially failed hint degrades to
      today's fault-path behavior and can never produce wrong data, but every
      failed/skipped hint is recorded in structured telemetry (never silent).
    - Init-time guard: every package member's data pointer must lie within its
      owning `llama_mmap` span; a member that does not is skipped with a
      telemetry event (hint semantics: degrade, don't abort).
    - The prefetch thread must never run on the copy/compute thread;
      `PrefetchVirtualMemory` blocks for roughly the transfer duration.
    - Ranges are per member tensor: { src->data, ggml_nbytes(src) }; one call
      with NumberOfEntries = member count (the API supports discontiguous
      ranges).
    - Quantization bytes and the H2D copy path are untouched; only the host
      feed changes.

Implementation steps
: 1. Add `--pp-prefetch-ahead N` (default 0) to the runner flag table with a
       host-safe parse entry (see the off-incompatible flag gate table in
       docs/currentstate.md).
    2. Build per-layer range lists from the T-GPLR-01/02 package adapter's
       member tensors; validate each range against the owning mapping at init.
    3. Worker thread + 1-deep queue; in `load_layer(il)`, after the
       eviction/consumer-done block, enqueue bank il+1's ranges; the worker
       calls `PrefetchVirtualMemory` and emits `prefetch_complete`.
    4. Telemetry: `prefetch_submit` (layer, range count, bytes) /
       `prefetch_complete` (layer, duration, error code); extend the run
       summary with an effective feed-rate column (bytes / prefetch duration).
    5. Probe: 262K layer-major Q8 runs at N=0 vs N=1 under the GPU mutex, same
       conditions as the T-GPLR-04 baseline (351.9 s wall / 327.6 s transfer).

Acceptance criteria (falsifiable)
: - Primary gate: 262K PP wall at N=1 drops >= 20% vs the 351.9 s T-GPLR-04
    baseline AND bank staging transfer time drops >= 2x; otherwise the
    hypothesis is recorded as falsified with the per-bank feed-rate telemetry
    (instrumentation result, not a performance claim).
  - Feed-rate gate: effective per-bank feed >= 2x the 52 MB/s cold baseline
    (>= ~104 MB/s sustained across all 38 evictions); expected value is far
    higher given the 2.7 GB/s device — this gate is deliberately
    conservative.
  - Parity: T-GPLR-07 smoke tier (50/512 on the 512-token anchor) all
    PASS with check-telemetry OK and aggregate checker PASS; negative control
    fails as designed. (This runner change touches the experimental path.)
  - Default path: N=0 runs behave byte-identically to today (same exit codes,
    same telemetry events; no new events when the flag is off).
  - H2D untouched: per-bank copy_us unchanged (~0.26 s layer-0 copies),
    proving the change is in the host feed only.

Focused validation
: `cmake --build build/llama-cuda --config Release --target llama-layer-streaming`;
  `tools/run-tests.ps1 -Suite host` (flag table); under the GPU mutex:
  `tools/with-gpu-lock.ps1` + runner 262K N=0 and N=1 with
  `--check-telemetry` on every trace; smoke tier per T-GPLR-07.

State update
: Record N=0/N=1 wall, transfer, and per-bank feed-rate tables in
  docs/currentstate.md; confirm or falsify with the trace evidence.

Escalate before implementation if
: Wall fails to improve >= 20% while feed rate improves >= 2x (mis-attribution
  — where does the remaining time go?); flag-table conflicts; owner wants the
  probe extended to the TG loop.

### T-GPLR-09: Host-RAM expert-bank cache tier (15% admission rule)

Status: PARKED (owner decision 2026-09-09, recorded in docs/currentstate.md
the same day). Do not plan or implement this task without an explicit
reopen trigger below. History: drafted alongside T-GPLR-08 (planner review
2026-09-09, initially HELD); T-GPLR-08 then ran and FALSIFIED its gate (see
its status note), which supplies the second key below.

Rationale -- a two-key door: the tier opens only if BOTH keys turn.
- Key 1 (policy): the <host>\<user> token has NO SeLockMemoryPrivilege
  and SeIncreaseWorkingSetPrivilege is disabled (probed 2026-09-09), so the
  contract's primary path (SetProcessWorkingSetSize raise + VirtualLock)
  cannot execute on this machine as configured -- the acceptance gates
  (wall <= 110 s) are unreachable without a Local Security Policy change,
  an owner-level environment change outside the implementer lane.
- Key 2 (premise): T-GPLR-08's falsified probe (2026-09-09) showed the
  working-set mechanism this tier relies on does not transfer to the
  rolling stage -- hinted prefetch populated pages 39/39 yet the
  fault-driven feed did not move, and the Sep 7 warm-cache leg already
  showed standby residency adds nothing to PP. A locked private copy
  additionally inverts the design's own tradeoff: dirty private pages trim
  to the pagefile under pressure (up to ~18 GB pagefile traffic), strictly
  worse than today's file-backed standby behavior.

Reopen triggers (any ONE reopens planning; owner-initiated, recorded here):
1. SeLockMemoryPrivilege is granted to the token AND a fault-count
   attribution probe (working-set query before/after prefetch; the open
   T-GPLR-08 attribution question) shows staging is fault-bound with
   trimming -- not memcpy-bandwidth-bound.
2. The primary workload moves to host-memory-tight conditions where the
   file-backed/standby path demonstrably thrashes (recorded hard-fault/
   RSS evidence per run, not inference).
3. Target hardware/OS changes so the privilege arrives by default (fresh
   provisioned box, admin-by-default account).

While parked: the 15% admission rule below is retained as design evidence,
not a commitment. TG batch-verify staging already sits at the RAM-speed
floor (see currentstate), so a resurrected tier could only ever target PP.
Depends on: T-GPLR-08 plumbing (range walker, telemetry) for any
resurrection; its falsified gate is also the attribution evidence the
reopen decision needs.
Commit: none (docs-only decision; no code was ever written)

Goal
: When the pageable bank set fits in host RAM under the admission rule, load
  it once at startup and keep it resident for the whole process, so every
  layer swap stages from RAM and the SSD is not touched during the run.
  Deterministic wall time (no more uncontrolled 263->52 MB/s variance).
  Admission rule (owner): reserve 15% of TOTAL physical RAM for the OS and
  other processes; admit iff avail_phys - bank_bytes >= 0.15 x total_phys.
  This machine today: total 61.69 GiB, reserve 9.25 GiB, banks 18.33 GB
  (17.07 GiB), free 50.50 GiB -> 33.43 GiB >= 9.25 GiB -> admitted; the rule
  rejects larger pageable sets automatically.

Scope
: Windows-only experimental path; opt-in flag `--pp-bank-cache` (default
  off). Admission evaluated host-only at preflight (GlobalMemoryStatusEx)
  before llama_backend_init, mirroring the existing preflight culture.

Design (safe implementation of "copy into RAM")
: - NOT a private committed copy: private pages under memory pressure get
      trimmed to the pagefile (dirty pages -> up to 18 GB pagefile write +
      read-back, strictly worse than today's file-backed standby behavior).
    - The file-backed mapping is already the source; pin it instead: (1) raise
      the process working-set minimum via SetProcessWorkingSetSize to
      bank_bytes + margin; (2) PrefetchVirtualMemory over all bank ranges in
      one bulk call (~7 s expected at 2.7 GB/s, one-time); (3) VirtualLock
      each bank range. Locked file-backed pages are clean (eviction = drop,
      no pagefile write) and cannot be reclaimed while locked -> deterministic
      residency. The staging path (load_layer memcpy) is unchanged.
    - Failure contract: if admission fails or the working-set raise /
      VirtualLock fails, the run ABORTS loudly with structured telemetry
      naming total/avail/reserve/bank bytes (fail-loudly culture). A separate
      opt-in flag `--pp-bank-cache-fallback-streaming` logs the decline and
      proceeds on the T-GPLR-08 streaming tier.
    - Host-only test knobs (mirroring the existing --physical-vram-bytes
      pattern): `--pp-bank-cache-test-avail N` overrides the reported
      avail_phys; `--pp-bank-cache-test-lock-fail` forces the lock-failure
      path. These exercise the decline/abort branches without manipulating
      the machine.

Non-goals
: TG-specific optimization (T-GPLR-06 owns TG; the cache's TG side-effect —
  per-token staging becomes H2D-bound — is a measured secondary observation,
  not a claim), multi-GPU, non-Windows, changing the slot/VRAM mechanism.

Implementation steps
: 1. Add `--pp-bank-cache`, `--pp-bank-cache-fallback-streaming`, and the two
       test knobs to the flag table; host-safe parse.
    2. Preflight admission: GlobalMemoryStatusEx; reserve = 0.15 x
       ullTotalPhys; admit iff ullAvailPhys - sum(bank_bytes) >= reserve;
       record total/avail/reserve/bank bytes + verdict in run_start telemetry.
    3. Working-set raise: SetProcessWorkingSetSize(min = bank_bytes + 256 MiB);
       on failure -> abort (or fallback path per flag).
    4. Bulk prefetch + lock: PrefetchVirtualMemory(all ranges, one call);
       VirtualLock each member range; verify and record results (cache_lock
       event with error code).
    5. Staging path untouched; add `cache_admitted` / `cache_lock` telemetry
       and a per-bank staging feed rate; record total disk-read bytes for
       banks during the run (expect ~0 after the startup load).
    6. GPU runs: 262K layer-major Q8 with --pp-bank-cache, twice, under the
       mutex; compare against the 351.9 s baseline and the warm-resident
       ceiling (~66 s from the 139-3968 tok/s, 0-eviction boundary traces).

Acceptance criteria (falsifiable)
: - Host: admission math table test passes (given total/avail/bank -> admit
    or decline); `--pp-bank-cache-test-avail` produces the loud decline (exit
    code + telemetry, no fallback unless the fallback flag is set);
    `--pp-bank-cache-test-lock-fail` produces the loud abort.
  - GPU 262K with cache: wall <= 110 s (>= 3.2x vs 351.9 s; warm ceiling ~66 s
    leaves headroom) AND bank staging feed >= 500 MB/s sustained (RAM-copy
    scale vs 52 MB/s) AND bank disk reads ~0 during the run (telemetry).
  - Determinism: two consecutive 262K runs within +/- 5% wall (vs today's
    uncontrolled variance).
  - Reserve honored: telemetry shows avail_phys_after >= 0.15 x total_phys
    (>= 9.25 GiB today).
  - Parity: T-GPLR-07 smoke tier all PASS + negative control fails as
    designed; route-event hashes identical to baseline (byte semantics
    preserved; staging still memcpys the identical mapped bytes).

Focused validation
: build + `tools/run-tests.ps1 -Suite host` (admission table, test knobs);
  under the GPU mutex: two 262K runs with the cache, one 262K baseline
  re-check, smoke tier; `--check-telemetry` on every trace.

State update
: Record admission numbers, lock results, wall/feed/determinism tables, and
  process commit peak in docs/currentstate.md; remove the
  uncontrolled-variance caveat only if the determinism gate holds.

Escalate before implementation if
: VirtualLock/working-set quota fails on this machine (privilege question ->
  owner decision); the standard 262K run declines admission (rule too strict
  -> owner decides the reserve fraction); wall stays far above the ~66 s warm
  ceiling despite residency (mis-attribution -> deeper probe); pagefile
  growth observed during runs (pinning hazard realized).

### T-GPLR-10: MoE handoff — streaming PP + stock llama.cpp TG (hybrid TG)

Commit: vendored 099a27a7e (T-GPLR-08/10/11, includes the evictfix); outer
docs recorded in the same-day follow-up commit (no push authorized)
Status: done (2026-09-09). Implemented: begin_handoff moe-expert-bank
relaxation + attention_only refusal under merge_resident, migrate_layer
bundle-fix (non-bank members point at the slot bundle, `dst = src`),
runner gates (--mtp+--handoff+moe refusal, layer-major tg-cap lift with
--handoff, policy-aware message), VRAM preflight handoff-fit branch with
peak-order slot/slab release, free_vram on the migrate event, hybrid-TG
checker rules (backward compatible with dense handoff traces), host
fixtures + run-tests.ps1 entries (host suite 22/22). Validation: host
suite 22/22; parity pair P1/P2 layer-major route-hash sequences 40/40
identical, TG prefix ids [220, 248046, 198] exact; batch-major handoff
TG 64.63 tok/s vs 0.495 streaming reference (~130x, instrumentation
floor >=2x exceeded); migrate 18,327,011,328 B in ~3.08 s, layers=40,
attention_only=false; free VRAM at migrate 3.14 GB (peak-order rule
holds); 262K Q8 refusal exit 1 with byte table before the timed PP;
dense Q4_K_M regression (1024 PP + 16 TG) first ID 286, all 16 ids
identical streaming vs handoff, checker OK both; QA-512 rung with
--handoff aggregate=PASS (5/5 needles), strict checker OK, negative
control FAILs as designed; shared-memory attribution run: adapter
Shared Usage stays at the ~0.3-0.37 GB desktop baseline through load,
PP, and the handoff copy (no 15 GB shared spike attributable to the
mechanism); run gplr10-attrib peaked dedicated 18.63 GB, TG ids
identical to gplr10-ho-bm reference. Same-day follow-up: the TG phantom
evictions (640 MoE / 1024 dense per 16 TG tokens) were traced to
load_layer's handoff branches running the streaming slot-reuse protocol
(consumer wait + eviction count + slot_evict event) although handoff TG
copies nothing and overwrites no slot data; the blocks were removed and
revalidated (gplr10-evictfix-{moe,dense}: post-migrate slot_evict 0,
TG summary evictions 0, PP evictions and transfer bytes unchanged, TG
ids unchanged, checker OK both, host suite 22/22). Traces:
gplr10-ref-bm/-lm, gplr10-ho-bm/-lm, gplr10-refuse-262k,
gplr10-dense-ref/-ho, gplr10-qa512-ho, gplr10-attrib (+
gpu-mem-poll-attrib.csv), gplr10-evictfix-{moe,dense}.
Depends on: T-GPLR-05 (merge-resident adapter machinery); T-GPLR-07 (answer
gate for the parity runs). Independent of T-GPLR-08/09 (different flags and
machinery); serializes in either order with T-GPLR-08 (closed 2026-09-09).

Goal
: After layer-residency PP (expert banks paged through the slot window),
  migrate all 40 MoE expert banks into one device buffer (18,327,011,328 B =
  37 x 452,984,832 + 3 x 522,190,848, per the T-GPLR-00/02 inventory) and run
  the remaining TG as stock llama.cpp llama_decode — no staging, no streaming
  executor — reproducing on MoE the hybrid already proven on dense
  (T-Q38HG-02/04: 24.75 tok/s TG). The MoE TG staging bound (18.33 GB/round
  at ~9 GB/s -> ~0.5 tok/s serial, T-GPLR-06 evidence) is removed by
  construction; the instrumented observable is TG tok/s with a falsifiable
  floor.

Scope
: Windows-only experimental path; --handoff under moe-expert-bank
  (batch-major PP, the dense-proven shape, and layer-major PP with the
  tg-cap lift below).

Migration list (exact)
: - 40 trunk bank packages pkg_cache[0..39], class moe-expert-bank: members
    = routed-expert tensors (ffn_gate/up/down_exps or fused gate_up+down),
    byte-exact as the T-GPLR-02 adapter enumerates them.
  - NOT migrated: router, shared-expert, attention, recurrent, layer-norm,
    static, MTP classes — already GPU-resident under merge_resident (non-bank
    ~3.05 GB total per inventory).
  - KV cache + DeltaNet recurrent state: context-owned, untouched
    (layer-residency invariant).
  - No MTP package: adapter_qwen35moe.mtp_package == nullptr (MTP execution
    is out of scope for the MoE adapter); MoE MTP stays the resident
    streaming path (T-GPLR-06) and --mtp + --handoff under moe-expert-bank is
    a loud gate refusal, not a migration item.

Gate changes (all host-safe, before llama_backend_init)
: 1. begin_handoff (llama-layer-streaming.cpp ~2193): replace the blanket
      merge_resident refusal with a moe-expert-bank path — the existing
      migrate_layer lambda runs verbatim on pkg_cache.at(il) (bank packages);
      dense behavior byte-identical. New refusal: handoff_attention_only
      under merge_resident (partial handoff has no meaning when only banks
      are pageable).
   2. main.cpp (~1842 mtp-policy gate): --mtp + --handoff + residency
      moe-expert-bank -> exit 2 with a stable message (stock TG has no draft
      loop; MoE MTP verify requires the streaming executor). Dense
      --mtp + --handoff stays legal (T-Q38HG-06).
   3. main.cpp layer-major cap (~1679): allow --tg-tokens > 3 when --handoff
      is active (post-handoff TG is stock llama_decode regardless of PP
      schedule); the cap remains for layer-major without handoff.
   4. Handoff progress message "all (64 layers)" -> policy-aware text
      ("all (40 expert banks)" under moe-expert-bank).
   5. off-conflict table: no new entries needed (--handoff already conflicts
      with --residency off).

Design / failure contract
: - Post-handoff TG is stock llama_decode: no per-layer copy/route events.
    Telemetry keeps the existing handoff event (layers=40,
    attention_only=false, mtp=false, total_bytes = 18,327,011,328 +
    alignment); the checker's handoff-aware rules (graph_builds > 0
    acceptable after handoff) extend so the hybrid TG phase validates
    WITHOUT inventing bank copy pairs: bank transfer ~0 in the post-handoff
    TG phase_summary is expected and checked, not an error.
  - VRAM preflight (fail before the timed PP; reuse the T-GPLR-04 preflight
    math at main.cpp ~2077): --handoff under moe-expert-bank requires
    full_model_plus_runtime + act_slab_bytes + reserve_bytes <= free_vram
    (free measured at run start); otherwise a structured error naming the
    byte table (weights/KV/state/slabs/reserve/free) and exit 1 before the
    timed PP.
  - Peak-order rule: release slot buffers and activation slabs before
    allocating buf_handoff (dense teardown precedent) so peak = resident
    classes + KV/state + handoff buffer, never their sum with the slot
    window.
  - Environment facts (not constants): physical 25.65 GB; observed free
    22.92 GB (2026-08-21 run_start); 262K Q8 full+runtime 24.30 GB ->
    long-ctx handoff refuses by design; short-ctx (512-4k) ~21.4-21.5 GB +
    overhead ~ marginal, decided by the measured free VRAM at run start. No
    fit claim is made; the preflight decides.
  - Every failure (allocation, copy, gate) is loud with structured
    telemetry; never silent fallback to paged TG.

Non-goals
: TG sampling/temperature (greedy only); MTP inside handoff TG; the
  layer-major tg-cap lift without handoff; multi-GPU; non-Windows; KV/state
  relocation; changing bank bytes, routing, or model arithmetic; the
  T-GPLR-08/09 host-feed tiers.

Implementation steps
: 1. begin_handoff: relax the merge_resident gate for moe-expert-bank
       (reuse migrate_layer); refuse attention_only under merge_resident;
       keep the dense path and MTP handling untouched.
    2. Runner gates: --mtp + --handoff + moe-expert-bank refusal; layer-major
       cap lift when handoff is active; policy-aware handoff message; all
       exit 2 before llama_backend_init.
    3. Preflight: handoff fit branch in the existing preflight; slot/slab
       release before buf_handoff allocation.
    4. Checker: hybrid-TG rules (bank transfer ~0 in the post-handoff TG
       phase; handoff event fields validated: layers=40, total_bytes
       expected).
    5. Host fixtures for the new gates; run-tests.ps1 host-suite entries.
    6. GPU probe under the mutex: reference streaming run -> handoff run
       (same prompt), then layer-major PP + handoff, then the 512-rung QA
       anchor with --handoff, then the 262K refusal probe.

Acceptance criteria (falsifiable)
: - Host: the new gate behaviors exit 2 with the recorded messages; dense
    combinations unchanged (--mtp --handoff legal, attention-only legal,
    layer-major without handoff still caps tg at 3); host suite green.
  - Parity (GPU mutex, MoE, same prompt/model/quant): handoff run exit 0;
    check-telemetry OK; handoff event total_bytes == 18,327,011,328 +
    alignment and layers == 40; first-16 generated IDs identical to the
    streaming reference; PP route-hash sequence identical to the reference;
    peak VRAM recorded.
  - Answer gate: T-GPLR-07 512 rung with --handoff -> aggregate PASS and
    the negative control fails as designed (hybrid path end-to-end).
  - Speed (instrumented, falsifiable floor): handoff TG tok/s >= 2x the
    streaming serial TG rate on the same prompt (>= ~1.04 tok/s vs the
    measured 0.52); recorded as instrumentation (bank transfer ~0,
    compute-only wall), not a general claim.
  - Dense regression: dense handoff (1024 PP + 16 TG) still exits 0 with
    the recorded reference first ID — the relaxed gate changed nothing for
    dense.
  - VRAM: 262K Q8 + --handoff under moe-expert-bank refuses before the
    timed PP with the byte table (exit 1); short-ctx either runs or refuses
    loudly with the same table (no silent fallback).

Focused validation
: `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming` -> exit 0; `tools/run-tests.ps1 -Suite host`; under
  the GPU mutex the GPU probe list above with `--check-telemetry` on every
  trace.

State update
: Record the migration byte table, gate behaviors, parity verdict, TG tok/s
  table (streaming vs handoff), and the VRAM verdicts in docs/currentstate.md.

Escalate before implementation if
: First-16 parity diverges beyond the recorded knife-edge standard (top-2
  margins, T-GPLR-06 precedent); the preflight cannot be made host-safe; the
  slot/slab release order cannot be verified in telemetry; owner wants
  sampling (non-greedy) TG in scope.

### T-GPLR-11: Close the batch-major tail-ubatch gap (prebuild the tail variant)

Status: done (2026-09-09). Pre-coding probes corrected the contract: the
hypothesized outputs class does NOT exist (with PP logits enabled the runner
marks every position, so variant B already carries the final ubatch's
outputs; probe 512/64 logits exited 0 pre-fix). The only missing variant is
the tail WIDTH (n % ubatch != 0 with > 2 ubatches). Fix as contracted:
prebuild_variant for the real tail tokens (continuing state, consecutive
synthetic positions like variant B, real logits values) + buf_act grows one
act pair (PP term 4 -> 6 halves of max_ubatch). Evidence: previously
failing 511/64 now exits 0 with graph_builds == 0 and check-telemetry OK;
512/64 logits exits 0 with PP first greedy ID 48 identical pre/post fix;
dense layer-major-1024 CTest 2/2; host suite green; semantic-QA 512 rung
aggregate PASS with the negative control failing as designed. Plan header
consolidation deferred to the T-GPLR-10 close-out to avoid concurrent
header edits across sessions.
Depends on: none (defect repair on the existing batch-major prebuild)
Commit: closed 2026-09-09 -- vendored code through third_party/llama.cpp 099a27a7e; outer docs/tools through 84be194 (follow-up 7bfe83b).

Background / defect record
: docs/currentstate.md ("Known gaps"): a batch-major PP whose token count is
  not a multiple of --ubatch fails loudly at the tail boundary
  ("begin_set: activation buffer exhausted", e.g. 511 tokens with ubatch 64):
  the batch-major prebuild creates only the full-ubatch variants A (fresh,
  rs_z=0) and B (continuing, rs_z=-1), so a runtime tail ubatch whose
  (n_tokens, n_outputs) key differs from A/B must begin_set at runtime and
  allocates a fresh act pair from the fixed buf_act, which is sized for
  exactly the prebuilt pairs. The failure is loud, never wrong-data. The
  recorded workaround is --ubatch = pp-tokens (single-ubatch PP), used by
  every run in this workspace. Layer-major is immune: T-Q38CTX-01 already
  prebuilds every chunk at its real width, including the tail.

Goal
: Any positive --pp-tokens runs to completion under batch-major with any
  --ubatch <= pp-tokens, with zero timed graph builds (the prebuild covers
  every reachable PP variant) and byte-identical results for all currently
  working configurations.

Root cause / design contract
: Two reachable mismatch classes at the tail ubatch:
  1. Width: n_tokens % ubatch != 0 -> tail width t < ubatch. The runtime
     ubatch splits at max_ubatch, so find_set(mode 0, t, ...) misses.
  2. Outputs: with logits enabled (no --pp-no-logits), the FINAL ubatch
     carries n_outputs == t rows while every earlier ubatch has 0
     (logits only on the last prompt position), so even a full-width final
     ubatch has a distinct set key from variant B when t == ubatch.
  Fix, minimal and failure-loud:
  - The prebuild additionally builds a tail PP variant at width
    t = n_tokens % ubatch (continuing state, rs_z = -1) whenever t != 0,
    using the real tail tokens at their real positions (batch-major keeps
    cells consecutive; no layer-major-style position patching is needed).
  - The prebuild additionally builds a final-ubatch variant with the real
    output-row count of the last ubatch (n_outputs = t or ubatch), so the
    logits-carrying final ubatch is prebuilt with logits enabled.
  - buf_act grows by at most one extra act pair: sized as today PLUS
    2 x n_embd_io x max(t,1) floats when a distinct tail pair is required
    (the final-outputs variant reuses the tail pair when t != 0 and the
    full-width pair when t == 0 -- it differs only in n_outputs, and act
    pairs are keyed per set, so the DISTINCT case t==0 && outputs>0 needs
    its own pair). Exact rule in code, asserted by a static comment.
  - --pp-no-logits runs keep n_outputs == 0 everywhere: only the width rule
    applies, byte cost = one extra tail pair.
  - Failure contract unchanged: any still-unreachable variant fails loudly
    at begin_set as today; no silent graph builds are added to the timed
    region (builds at runtime remain counted in graph_builds telemetry and
    the checkers already flag PP-phase graph_builds != 0).

Non-goals
: Layer-major (already correct); TG variant coverage (already enumerated);
  MTP paths; chunk-schedule changes; new flags; checker rule changes beyond
  none (the existing PP graph_builds rule already enforces the contract).

Implementation steps
: 1. llama-context.cpp layer_streaming_prebuild, batch-major branch: after
       variant B, build the tail variant (t = n_tokens % ubatch, real tail
       tokens/positions, rs_z = -1, expected n_outputs per the logits rule)
       and, when t == 0 and logits are enabled, a full-width final variant
       with n_outputs = ubatch. Each build_variant call already allocates
       its act pair from buf_act via begin_set.
    2. llama-layer-streaming.cpp init: extend act_bytes by the tail term
       (2 * n_embd_io * t_max floats, t_max = tail width if 0 < t < ubatch
       else 1 when a distinct final-outputs pair is required) and record the
       sizing rule in a comment tied to this task.
    3. Host fixtures: a fixture JSONL for a tail-ubatch run shape is NOT
       needed (no new events); instead add a CTest-free host check only if
       a new event appears (not expected).
    4. GPU validation under the mutex with the reference runner:
       a. 511-token prompt, ubatch 64, batch-major, MoE model -- previously
          failing, must now exit 0 with graph_builds == 0.
       b. 512-token prompt, ubatch 64, WITH logits (no --pp-no-logits) --
          previously failing at the final ubatch, must now exit 0,
          graph_builds == 0, PP first greedy ID unchanged vs the recorded
          512/64 logits reference if one exists (else record it).
       c. Regression: 512/512 single-ubatch (the standard shape) byte
           behavior unchanged; layer-major 1024 CTest regression green.
    5. Strict --check-telemetry on every new trace.

Acceptance criteria (falsifiable)
: - (a) and (b) exit 0 with check-telemetry OK and PP graph_builds == 0;
    the runs complete without the activation-buffer error.
  - (c) the standard 512 full-ubatch shape is unchanged (same events, same
    exit codes) and the dense layer-major CTest regression stays green.
  - The workload previously requiring --ubatch = pp-tokens now runs with a
    practical ubatch; host suite green.

Focused validation
: cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming; tools/run-tests.ps1 -Suite host; then the GPU legs
  above under tools/with-gpu-lock.ps1.

State update
: Record the defect closure, the new buf_act sizing rule, and the (a)/(b)/(c)
  evidence in docs/currentstate.md; remove the stale "workaround --ubatch =
  pp-tokens" guidance.

Escalate before implementation if
: The tail variant's memory application (cells consecutive) cannot be made
  to match runtime find_slot behavior exactly (risks KV layout mismatch);
  buf_act growth exceeds one act pair; the logits final-ubatch variant turns
  out to already be reachable (verify first with a 512/64 logits probe
  before coding -- a cheap disprovable check).
