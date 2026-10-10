# T-Q36-AER: Qwen3.6 Activation Expert-Ranker Feasibility Pipeline

Status: Phase 1 tiny gate complete (2026-09-25); Phase 2 pilot, training, and campaign remain owner-gated and have not started.

Dependencies: `docs/layer-residency-sdd.md`; `docs/currentstate.md`; `docs/AGENT_ENVIRONMENT.md`; Qwen3.6 inventory; existing route telemetry, row-cache replay, retention validation, and host predictor tooling; `third_party/llama.cpp/AGENTS.md`.

Commit: unfilled. Do not stage or commit until the owner explicitly authorizes it.

## Goal

Determine whether a causally early activation can predict the same token's later Qwen3.6 MoE expert routes well enough to improve expert-row cache placement.

This is a feasibility study. It must not claim a production predictor, decode speedup, or quality result. The new predictor family must use materially different information from the existing history-only family: an early activation feature rather than only prior expert IDs.

## Locked decisions

- Primary target: same-token, same-trunk-layer activation ranker.
- Feature point: layer input available before the attention/Gated-DeltaNet block.
- Label: eventual authoritative top-8 routed expert IDs.
- Runtime router remains unchanged; predictions are prefetch hints only.
- Main workload: code-first streamed public prompt mix plus a smaller general-chat slice.
- Storage ceiling: 100 GiB for the first campaign, with resumable shards and measured gates.
- Representation: dual staged campaign: raw FP16 reference plus compact 256/512-dimensional main representation.
- Scope: offline data collection, training, evaluation, and cache simulation first. llama.cpp runtime integration is a later, separately authorized task.

## Current repository facts

The Qwen3.6 vehicle is a 40-trunk-layer, 256-routed-expert, top-8 model with hidden size 2,048. The GGUF has one additional MTP/nextn block; MTP is excluded from the primary dataset and is not silently mixed into ordinary decode labels.

The existing history-only predictor family produced repeatable signal but was stopped because it lost to equal-row retention on the relevant cache-miss objective. This task must therefore be judged on real cache-miss coverage and transfer cost, not ordinary top-k accuracy alone.

Reuse existing repository route telemetry, row-cache replay, retention validation, and host predictor conventions rather than creating an incompatible second cache simulator.

## Scope and non-goals

### Scope

- opt-in llama.cpp instrumentation for early features and actual route labels;
- bounded, resumable trace collection on the existing Qwen3.6 GGUF;
- raw and compact feature representations;
- route-only baselines and per-layer activation rankers;
- held-out cache-miss evaluation and existing replay integration;
- mandatory owner checkpoint after the pilot;
- a separately gated proposal for future runtime integration.

### Non-goals

- no change to the real router or routing arithmetic;
- no predictor-generated replacement for an actual expert selection;
- no prefill predictor in the first campaign;
- no MTP/nextn predictor in the primary dataset;
- no production server integration;
- no full Transformers or Unsloth model download;
- no unbounded dataset download;
- no claim of decode speedup from offline metrics;
- no modification to the active generic SDD or the existing history-only closure record.

## Phase 0: Reference identity and feasibility contract

Pin and record:

- Qwen3.6 GGUF path, size, and SHA-256;
- vendored llama.cpp revision and build flags;
- CUDA, PyTorch, and Python versions;
- trunk layers, expert count, top-k, hidden size, and MTP metadata;
- baseline uninstrumented decode command and throughput;
- prompt-group split manifest;
- 100 GiB campaign budget;
- privacy policy for private prompts.

The campaign must not begin until this manifest identifies the exact feature point, model, source revision, and label source.

## Phase 1: Opt-in trace instrumentation

Add an isolated opt-in llama.cpp trace path that leaves normal inference unchanged when disabled.

For each valid generated token and each trunk layer, capture:

- early layer-input feature, expected dimension 2,048;
- eventual actual top-8 expert IDs;
- layer ID, sequence/request ID, token position, and mode;
- stop reason and valid-row count;
- optional pilot-only router logits or margins if affordable.

The collector must write bounded checksummed shards, record schema/model/source metadata, resume from completed shards, reject incomplete or ambiguous sequences, never repair alignment heuristically, and report instrumented throughput, GPU readback time, bytes/token, peak VRAM, and disk growth.

The existing Qwen graph and route telemetry are the reference for proving that emitted labels match the real router. The tiny trace must verify exact top-8 IDs, ordering/tie behavior, layer alignment, sequence boundaries, and tracing-off output identity.

### Phase 1 acceptance criteria

- tracing-disabled inference retains the known-good behavior and build contract;
- a tiny trace has exact feature dimensions, exactly eight unique valid labels per row, valid layer IDs, and no duplicate sequence/layer rows;
- emitted labels match existing route telemetry for the same deterministic run;
- interrupted or incomplete sequences are rejected rather than repaired;
- a host-only parser/checker reports a clear pass or structured refusal.

### Phase 1 tiny CUDA gate result (2026-09-25)

The owner-approved 512-token prompt plus one generated-token tracing-off baseline, ordinary trace arm, and row-set trace arm ran under `Global\\AI-GPU-EXCLUSIVE-v1` with clean pre/post GPU probes. The tracing-off run, ordinary arm, and row-set arm each completed with exit 0; the two trace manifests contain one token at position 513, 40 ordered layers, 2,048 finite FP16 features, and eight unique in-range labels. Both generated token IDs were `248046`; the tracing-off, ordinary, and row-set text outputs were empty because `--tg-text` records text only when produced by the runner.

A first ordinary-arm attempt failed with CUDA illegal memory access during TG. The failure was caused by a fresh per-layer `ggml_cpy` added only for AER capture; replacing it with `ggml_set_output(inpSA)` plus capture of the existing F32 layer-input tensor removed that failure. A subsequent host top-k experiment exposed a second real issue: deriving ordinary labels from a diagnostic probability tensor produced 40/40 layer mismatches against the row-set arm. The ordinary path now reads the graph's authoritative I32 `ffn_moe_topk` output, and the graph build marks that output live. The rerun passed ordinary and row-set checks and cross-arm parity: 40 rows compared, exact top-8 label parity, identical model/source/build/prompt identity, and 165,484 bytes per trace.

Measured gate evidence: ordinary `check-telemetry` PASS (845 events), 156.877 PP tok/s, 17.296 TG tok/s, peak 3,849.56 MiB; row-set `check-telemetry` PASS (605 events), 156.439 PP tok/s, 6.489 TG tok/s, peak 5,727.56 MiB, with two row-set rounds each staging 572,719,104 bytes (3.12% of the 18,327,011,328-byte full bank). The host CTest targets and Python self-test pass; outer and nested whitespace checks pass. The first failed/mismatched artifacts are preserved under `build/q36-aer-gate/previous-*` for diagnosis. These are advisory gates, not approval for Phase 2.



Collect approximately 10,000 decoded tokens from preassigned code-first and chat prompt groups. Use short bounded completions and record the full manifest.

Measure actual feature bytes/token, instrumented tok/s, GPU readback and trace overhead, peak VRAM, storage growth, route-label correctness, feature/label row counts, sequence boundaries, raw feature validity, and a small raw-versus-compact comparison if the pilot supports it.

### Owner checkpoint: hard stop

After the pilot, stop the campaign and produce a self-contained report containing:

- collected token and sequence counts;
- data-integrity verdict;
- measured throughput and storage cost;
- projected size of the next campaign;
- raw predictor quality on a small held-out set, if available;
- compact 256/512 signal comparison, if available;
- cache-miss recall and equal-row-retention comparison, if available;
- known limitations and unresolved risks;
- recommendation: continue raw, continue compact, revise the feature point, or stop.

No larger collection, package installation, additional dataset source, or runtime integration may begin without an explicit owner decision. Record the decision in the campaign manifest so an agent cannot silently resume.

Continuation branches:

- **Continue raw:** raw activations show useful held-out cache value and pilot overhead/storage are acceptable.
- **Continue compact:** compact features retain acceptable cache value at 256 or 512 dimensions.
- **Revise:** labels align but cache value is weak or the feature point is too late.
- **Stop:** compact/raw rankers fail to beat equal-row retention or collection overhead is impractical.

## Phase 3: Raw reference dataset

Only after owner approval, collect a bounded all-layer raw FP16 reference campaign, initially approximately 50,000-100,000 decoded tokens.

```text
40 * (2048 FP16 values + 8 uint16 expert IDs)
= 164,480 bytes/token, approximately 156.9 KiB/token
```

Approximate totals: 10k tokens = 1.53 GiB; 50k = 7.66 GiB; 100k = 15.3 GiB. Do not collect 500k raw tokens automatically. Expand only when learning curves and cache simulations still improve.

## Phase 4: Compact feature projection experiment

Use the raw reference to compare 256- and 512-dimensional representations.

Candidate projection arms:

1. PCA fitted on training prompt groups only;
2. fixed random Gaussian projection as a control;
3. optional learned linear projection trained with a ranking objective.

Fit every projection on training data only and apply it unchanged to validation/test. Split by prompt group, conversation, task template, and source repository; never randomly split token rows.

Train the same downstream ranker families on each representation: linear `input -> 256 expert scores` and small MLP `input -> hidden -> 256 expert scores`.

Measure recall@M, exact top-8 set match, any-hit@M, raw-versus-compact top-M overlap, score-ranking agreement, cache-conditional recall of real misses, useful speculative rows, wasted rows/bytes, candidate lead/ready timing, simulated demand-load stalls, and results by layer, block type, workload, and decode position.

A compact dimension is acceptable only if it preserves the agreed cache-value margin against the raw model, does not materially worsen wasted transfer, remains stable across seeds/groups, and still beats equal-row retention. A reduction in ordinary accuracy is not itself a failure if cache-conditional value is preserved; a cache-value collapse is a failure.

Projection after full GPU readback saves disk space but not readback bandwidth. A runtime-useful projection must eventually execute on the GPU before readback.

## Phase 5: Offline predictor lab

Train in increasing complexity:

1. route-only popularity;
2. previous-token same-layer routes;
3. existing co-occurrence/cross-layer baselines;
4. raw linear ranker;
5. raw small MLP;
6. compact linear/MLP rankers;
7. optional shared multi-layer model only if per-layer models show a clear limitation.

Use one model per trunk layer. The first target is a multi-hot vector over 256 experts with eight positives. Start with sparse multi-label binary cross-entropy and positive-class weighting. If router logits are captured, add secondary distillation; expert IDs remain the minimum authoritative label.

Split data before flattening. Preserve sequence and token boundaries so later history experiments can use fixed windows without crossing requests. The first MLP is row-independent and requires no causal mask.

## Phase 6: Cache-aware decision

Reuse or minimally extend existing host replay tools. Compare predictors against equal-row retention, the existing history-only family, count-matched nulls, and the recorded Qwen3.6 row-cache behavior.

Primary offline question:

> At equal logical row/cache budget, does the activation ranker cover more real cache misses than equal-row retention while keeping wasted transfer and predictor overhead within an agreed bound?

Report per layer and workload slice, not only a global average. Do not claim decode speedup from offline recall.

## Phase 7: Future runtime task, only if authorized

If offline gates pass, create a separate task for exporting trained weights, adding a tiny opt-in runtime ranker, predicting only absent candidates, respecting the demand reserve and row-pool lifecycle, measuring predictor latency, transfer lead time, cache hits, wasted H2D, demand stalls, p50/p95 token latency, and tok/s, and running constrained-cache A/B comparisons under the shared GPU mutex.

The real router must remain authoritative. The ranker may move weights or propose candidates; it may never replace actual routing or silently substitute experts.

## Local Unsloth Studio access and environment preflight

Unsloth Studio is a local service at `http://127.0.0.1:8888`. The launcher fallback is:

`C:/Users/<user>/AppData/Local/Unsloth Studio/launch-studio.ps1`

Do not start a second Studio instance when the health endpoint is already responding.

### Verified no-key API calls

The following read-only endpoints currently work without a password or API key:

- `GET /api/health` — service health;
- `GET /api/auth/status` — initialization/login state;
- `GET /api/inference/status` — current model/runtime status;
- `GET /api/inference/monitor` — active request and queue status;
- `GET /api/export/status` — export state.

`GET /api/auth/api-keys` is protected and returned `403` without authentication. Do not put the password, API token, or bearer value in the repository, plan, logs, or command history. If a future mutating operation requires authentication, obtain owner direction first and use a session-scoped mechanism rather than persisting credentials in project files.

### Verified training environment

Use the existing Unsloth interpreter for dataset/training tooling:

`C:/Users/<user>/.unsloth/studio/unsloth_studio/Scripts/python.exe`

Verified on 2026-09-25:

- Python 3.13.14;
- PyTorch 2.10.0+cu130;
- CUDA available through the Unsloth environment;
- NumPy 2.5.2;
- datasets 4.3.0;
- PyArrow 23.0.1;
- Transformers present.

The default `C:/Program Files/Python313/python.exe` is not the training environment. The Python 3.10 environment has PyTorch but is CPU-only. Do not install into or modify the Unsloth environment without owner approval.

### Environment preflight before any model run

Run and record these checks before the first real-model workload:

1. `nvidia-smi` before and after the run; reject a busy GPU unless the shared-mutex policy explicitly permits it.
2. Acquire `Global/AI-GPU-EXCLUSIVE-v1` through `tools/with-gpu-lock.ps1` for every CUDA/model run.
3. Confirm the Qwen3.6 GGUF path, byte size, and SHA-256.
4. Run the host-only `--inspect-model` check and verify the 40 trunk layers, 256 experts, top-8 routing, hidden size 2,048, and one separate MTP block.
5. Confirm the existing CUDA/layer-streaming build has `GGML_CUDA=ON` and `LLAMA_LAYER_STREAMING=ON`.
6. Confirm the host-only Unsloth API status calls above return 200 and report no active model before starting.
7. Confirm free disk remains above the campaign budget plus temporary-file headroom; stop at the 100 GiB campaign cap.
8. Run all host-only parser, checksum, split, projection, and metric self-tests.
9. Run only the smallest owner-approved real-model trace first; do not begin the 10k pilot until that trace passes.

No package installation, dataset download, model run, or prompt campaign is required to complete the environment verification. These checks establish readiness without changing the machine.

## Validation and infrastructure

Host-only tests cover shard headers/checksums, fixed-width feature/label decoding, exact top-8 validation, variable-length indexing, prompt-group split and duplicate detection, deterministic projections, ranking/cache fixtures, replay parity, and refusal cases for malformed, incomplete, leaked, stale, or over-budget data.

Compilation checks cover opt-in instrumented and tracing-disabled builds. GPU runs use `tools/with-gpu-lock.ps1` and record GPU pre/post state, exact commands, model identity, flags, exit status, and telemetry paths.

The agent operates the pipeline through bounded wrapper commands that check disk, acquire the GPU lock, resume completed shards, write structured verdicts, and produce machine-readable and human-readable summaries.

The owner configures only Qwen3.6 model identity, approved prompt sources/privacy policy, the 100 GiB cap, package-installation approval, the pilot continuation decision, and any future runtime experiment.

Every completed phase ends with an owner summary containing the original goal, plain-language mechanism, glossary, measured/expected/unproven pros and cons with evidence paths, and a recommendation to adopt, keep experimenting, or discard.

## Required repository updates

The implementation task must update:

- `docs/plans/qwen36-activation-expert-ranker-plan.md` as the authoritative task contract;
- `docs/currentstate.md` with observed behavior, validation evidence, known gaps, blockers, and residual risk;
- nested dependency instructions and relevant task notes if implementation reveals a reusable invariant;
- no generated traces, checkpoints, datasets, or model outputs in version control.

## Escalation condition

Escalate to the owner if:

- the early feature cannot be captured before the required decision point without changing routing arithmetic;
- pilot trace overhead or storage projection exceeds the approved bound;
- the owner checkpoint has no explicit continuation decision;
- compact features lose cache-conditional value;
- a new dependency is required;
- the plan would require modifying normal llama.cpp behavior instead of the opt-in experimental path;
- any invariant, source revision, model hash, or split manifest is ambiguous.

No online answer or implementation improvisation may bypass these gates.
