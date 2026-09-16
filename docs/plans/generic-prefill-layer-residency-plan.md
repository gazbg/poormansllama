### T-GPLR-25: PP eviction prefetch-overlap prototype (device-side copy overlap in layer-major PP)

Status: implemented 2026-09-15 (prototype + owner slot-default decision; evidence
in docs/currentstate.md same day). Paired A/B verdict 2026-09-15: 512 rung
shows NO af delta (af0 11.958 s vs af1 11.989 s, -0.3%, noise; worker engaged,
zero idle headroom at 1 chunk/layer) and the earlier unpaired
"12.29 s vs 17.6 s" prototype wall comparison is WITHDRAWN as confounded
(slot shape 2-vs-3 and cold-vs-warm cache). The 4k pair has now run:
af delta +3.0% (29.123 s vs 30.034 s) sits AT the TG-leg noise floor (+2.3%
between the same two arms) — NO performance claim; direction positive and
mechanism-consistent, magnitude unproven pending replicated pairs. The 4k
traces also revealed that batch-major PP re-stages every bank once per chunk
sweep (320 copies = 8.09x bank bytes at 4k, 318 evictions), making the sweep
schedule — not copy bandwidth — the real H-D cost driver at 4k. The owner
budget-refusal path was verified in the same session (run_refused,
cap_source=budget, no silent downgrade; traces in docs/currentstate.md).
The same session then fixed the task's telemetry residue: the async-feed
worker was never fed under batch-major (both enqueue sites lived in the
layer-major controller) and async_feed_ready fired unconditionally on every
consumption — the false-positive overlap evidence. The worker is now the
single copy writer per slot under both schedules (enqueue-when-af-on,
consumer-done wait in af_submit_copy, claim-on-stage, af_fed gate on the
ready event); the (layer,generation) submit/ready event sets are an exact
bijection on the validation arm, and af=0 behavior is byte-identical. Amended from the draft below during
implementation: the drafted `--pp-lookahead K` flag was NOT added because the
slot window is already the existing `--layer-slots` flag -- the owner decision
made the default 2 (auto-3 on fit), and the async-feed worker
(`--pp-async-feed 1`) supplies the device-side overlap the draft wanted. The
checker defect the 3-slot rung exposed (slot_evict's per-slot count for uneven
n_layer/n_slots) was fixed in the same pass.
Depends on: T-GPLR-13 (bottleneck evidence), T-GPLR-08 (falsified HOST-side
PrefetchVirtualMemory probe — this task is the DEVICE-side analogue and shares
no mechanism with it)
Commit: third_party/llama.cpp 68cd1726d (2026-09-16: af rewire + residue fix portion)

Goal
: Close the H-D hypothesis with a measured prototype. The 2026-09-15 trace
  analysis (build/analyze_pp_copy_gaps.py, build/analyze_pp_overlap.py over
  the RECORDED arms, no new GPU runs) established the serialization structure:

  - direct-200k (`rung-200000-fix.jsonl`): 15,640 PP copy_submit events,
    submit inter-arrival mean 65.3 ms / median 52.5 ms, occupancy 1,020.98 s;
    copy_complete host_submit sum 738.21 s, ready_lat sum 3.67 s. The enqueue
    loop itself (~47 ms mean for 453 MB) is the driver's internal pageable
    staging; DMA completes ~0.2 ms later (effective feed ~9.7 GB/s while
    submitting).
  - The pinned `--copy-source pinned` arm is WORSE and is not the answer:
    host_submit 110 ms + ready_lat 476 ms (gplr18-B1/B2) — the explicit
    staging memcpy serializes against the DMA instead of overlapping as the
    driver's internal staging does (hd-pinned: 221 ms + 32 ms).
  - direct-4k (`gplr07-qa-4096.jsonl`): 0/40 layer_compute events have the
    next layer's copy already submitted at compute start; submit
    inter-arrival median 193.6 ms, compute median 19.6 ms. Copy occupancy
    8.05 s of a ~8.3 s PP wall. This is where the recorded 3.34 GB/s
    "PP staging bandwidth" comes from: bytes/wall with zero overlap — NOT a
    slow copy path. The same copy path at 200k (larger compute windows)
    overlaps naturally and reaches ~8.7 GB/s.
  - Root cause of the 4k shape: the recorded runs use `--layer-slots 1`.
    With one slot, `load_layer(il+1)` cannot issue until
    `record_consumer_done(il)` — the slot-reuse invariant forbids
    overwriting the weights the GPU is computing from. The lookahead call
    already exists (controller line ~2489) and is correct; it is starved by
    the window, not by the copy path.

Scope
: Prototype only, no default change. Runner flag `--pp-lookahead K`
  (default 0 = today's behavior byte-identical) that when K >= 1 raises the
  layer-major PP slot window to n_slots = K+1 for the PP phase by passing the
  deeper slot count at ls init — NO new copy scheduling code. The existing
  preload block (`preloaded < ls->n_slots`) plus the existing
  `load_layer(il + n_slots)` lookahead after `record_consumer_done` then
  naturally keeps n_slots copies in flight while n_slots-1 layers compute.
  VRAM cost: (K+1) x slot_vram (~453 MB per slot for the MoE bank, 522 MB
  allocated today at 1 slot; K=1 -> ~1.04 GB device, K=3 -> ~2.09 GB).
  Allowed iff the T-GPLR-04 capacity preflight admits the deeper window
  (it already measures the slot window; no new budget code).

Non-goals
: No change to copy_source (the pinned variant is measurably worse), no
  row-set/TG changes, no handoff interaction (handoff refuses deeper windows
  is acceptable — document if hit), no host-RAM prefetch (T-GPLR-08 closed),
  no batch-major executor change (it already looks ahead via graph_entries),
  no new telemetry events beyond the existing slot_request/copy_submit/
  copy_complete stream (the checker must accept the traces unchanged).

Design / failure contract
: - The slot-reuse invariant is untouched: a slot may be overwritten only
    after ev_consumer_done (SDD). Deeper windows create MORE overlap while
    respecting it; they never relax it.
  - A layer graph consumes a slot only after the matching copy event
    completes (wait_slot_ready + ev_copy_ready wait) — unchanged.
  - K=0 must be byte-identical: same n_slots path as today, no new events.
  - Refuse loudly (runner exit 2, pre-sink `run_refused` via the existing
    capacity preflight) when (K+1) x slot window + residency budget exceeds
    free VRAM. Never fall back silently to K=0.

Implementation steps
: 1. Runner: add `--pp-lookahead K` (default 0) to the flag table with the
       off-incompatible-flag gate entries (handoff modes reject it; row-set
       TG unaffected — TG slot window is separate).
    2. Runner: when K >= 1, pass n_slots = K+1 into the layer-streaming init
       (the same plumbing `--layer-slots` uses today) and record
       `pp_slots` in run_start (checker ignores unknown run_start fields —
       verify against the checker before relying on that).
    3. Verify the layer-major controller's preload + lookahead path needs no
       change (expected: none; it is parameterized by n_slots already).
    4. Focused validation (below) at K=0 (regression) and K=1/K=3 (probe).

Acceptance criteria (falsifiable)
: - Overlap gate: at 4k layer-major with K=1, >= 80% of layer_compute events
    have the NEXT layer's copy already submitted at compute start
    (analyze_pp_overlap.py on the new trace) — vs 0% today.
  - Bandwidth gate: derived PP staging bandwidth (bytes_staged / PP wall)
    at 4k rises from the recorded 3.34 GB/s toward the >= 8 GB/s the copy
    path demonstrates at 200k, OR the trace shows the residual is compute-
    bound (either outcome closes H-D; both are reportable results).
  - Wall gate (informational, not a PASS claim): 4k PP wall drops vs the
    recorded 6.26 s warm arm when file-cache state is held comparable
    (cold-cache caveat per AGENTS.md applies; compare structure, not raw).
  - Parity: qa-smoke 512 gate PASS (check-telemetry OK, answer 5/5, negative
    control FAILs as designed) on the K=0 default path, and the 512 K=1 arm
    answer-gates PASS too (the change touches the experimental path).
  - Default immunity: K=0 trace event-name multiset equals the recorded
    512 trace; zero non-volatile field differences.
  - VRAM: byte_account reports the deeper window; peak stays within the
    preflight's declared budget (no new peak beyond window growth).

Focused validation
: `cmake --build build/llama-cuda --config Release --target llama-layer-streaming`;
  `tools/run-tests.ps1 -Suite host`; then under the GPU mutex
  (`tools/with-gpu-lock.ps1`):
  4k layer-major direct arms K=0/K=1/K=3 (`--pp-schedule layer-major
  --pp-tokens 4096 --ubatch 512 --pp-lookahead {0,1,3} --copy-source direct`)
  with `--check-telemetry` on every trace; qa-smoke 512 for the parity gate.
  Analysis: build/analyze_pp_overlap.py + build/analyze_pp_copy_gaps.py on
  the new traces.

State update
: Record the K=0/1/3 overlap %, staging-bandwidth, and PP-wall table in
  docs/currentstate.md; annotate H-D in docs/gplr13-bottlenecks-hypotheses.md
  with the measured closure (confirmed/refuted + residual).

Escalate before implementation if
: The checker rejects unknown run_start fields (needs a checker change —
  separate task); handoff interacts with deeper windows unexpectedly;
  K >= 1 regresses the 4k PP wall (overlap without gain means the copy
  contends with compute for bandwidth — measure and report, do not tune).

### T-GPLR-26: Gate-eligibility decision recording — binding vs advisory (item 4 brief)

Status: DRAFT — pending owner pick. Decision material is the 2026-09-14 brief
in docs/currentstate.md ("should gate eligibility BIND provisioning or stay
ADVISORY?"); this draft is the task that records the pick and executes its
consequences. Nothing is implemented and no gate verdict is claimed by this
draft.
Depends on: T-GPLR-22 (the PRE-SINK run_refused plumbing the brief names as
the existing shape for a gate-ineligible provisioning refusal; the missing
part is the oracle, not the plumbing).
Commit: (fill on completion)

Goal
: Close the owner call on whether row-set gate eligibility (the informational
  (b) reading vs the 0.6 line) BINDS provisioning or stays ADVISORY, and make
  the repo state match the pick exactly — no third state.

Scope — branch A (owner picks BIND)
: Prerequisite chain FIRST, as its own tasks, because the brief establishes
  no admissible input exists at bring-up: (1) a tracked (b)-vs-K oracle
  surface for a DECLARED workload class, produced by the existing analyzer
  over recorded rowset_fetch union streams, with per-stream provenance
  recorded (model, prompt family, W, retention rule, stream length); (2) a
  pre-registered verdict rule fixing WHICH stream length, W and retention
  rule count produce a verdict; (3) a stated behaviour for a depth the
  surface has no entry for (refuse, or fall back to the SDD's reference
  band); (4) a new refusal reason + trace fields so a gate-ineligible
  provisioning refusal is recoverable from JSONL (enum + emit site + checker
  parse — one owner each). Only then does provisioning consult the surface.

Scope — branch B (owner picks ADVISORY, the brief's recommendation)
: A documentation-only task: SDD scope note amended to state the gate is
  advisory permanently until an oracle exists; the K=4 default note updated
  to carry the non-echoing caveat (0.5355 at K=4 on the honest stream) next
  to the echoed origin; NO code change; the current measured table becomes
  the recorded evidence.

Non-goals (both branches)
: No TG token-generation behaviour change; no re-measurement of the (b)
  table inside this task (the brief's numbers are the input); no quality or
  speed claim.

Acceptance criteria (falsifiable)
: Branch A: the oracle artifact exists at a recorded path with provenance
  fields populated for every contributing stream; a GPU arm provisions
  against it and a depth whose verdict is below 0.6 produces run_refused
  with the new reason and is accepted by --check-telemetry as a REFUSAL
  TRACE; a depth above 0.6 provisions unchanged; host suite green; default
  llama.cpp paths byte-identical. Branch B: the SDD scope note and the K=4
  annotation exist and name the brief; git diff shows zero source changes;
  host suite green.

Focused validation
: Branch A: `tools/run-tests.ps1 -Suite host`; under the GPU mutex a
  row-set arm at a below-threshold depth (`--layer-slots 4 --rowset-retain
  K` shape per the T-GPLR-19 driver) and the refusal-trace check;
  `--check-telemetry` on every produced trace. Branch B: `git diff --stat`
  (docs only) plus the host suite.

State update
: docs/currentstate.md records the pick, the branch executed, and (branch A)
  the oracle path + provenance or (branch B) the amended doc locations. The
  brief section gains a one-line pointer to this task's outcome.

Escalate before implementation if
: The owner picks A but wants the oracle built from the EXISTING two-stream
  measurement (the brief shows that input is wrong-sized for a policy);
  the refusal-reason enum change collides with an in-flight task; branch B
  would contradict a newer SDD amendment.

### T-GPLR-27: Second-architecture first pass — dense target (item 6 brief)

Status: DRAFT — pending owner pick of the architecture. Decision material is
the 2026-09-14 brief in docs/currentstate.md ("which second architecture
should be the first to pass the generic gates?"); the brief recommends
Gemma-4 12B with the qwen35-arch small models run first as a cheap precursor
that does NOT satisfy the gate. This draft is written against the
recommended target and is re-scoped if the owner picks otherwise. No
adapter, code, default, or gate verdict exists yet.
Depends on: owner pick (activation condition); T-GPLR-04 preflight and the
T-GPLR-22 run_refused path (already arch-independent); T-GPLR-07 semantic-QA
tooling (needs per-model re-calibration); the Qwen reference gates standing
green (any regression there blocks this task).
Commit: (fill on completion)

Goal
: One non-Qwen architecture passes the SDD §11 gates in their generic form —
  gates 5 and 6 unchanged, gate 7 bound only if a speed claim is made, and
  the per-arch FORM of gate 1: package bytes, slot lifecycle, and a focused
  greedy ID matched against a non-residency control on the same build,
  prompt and shape. This closes the SDD's "generic support is not complete
  until another architecture passes the active plan's acceptance gates"
  condition or produces the evidence that a gate cannot generalize.

Scope (recommended target: gemma4 12B, 48 blocks, dense, 6.26 GB Q4_0)
: (1) Adapter: package descriptor + factory + registry entry (~70-90 lines,
  new src/models/llama-layer-streaming-gemma4.cpp; no loader work needed —
  LLM_ARCH_GEMMA4 already loads through the standard pattern). (2) The graph
  seam: split layer-major graph hook in src/models/gemma4.cpp (no streaming
  seam today; ~250-450 lines by analogy with qwen35.cpp's seam). (3)
  Inventory generalization in llama_ls_model_inventory — the Qwen-hybrid
  ssm.* requirement fails gemma4 at the first key today; head_count_kv is an
  array; five arch-string hard-code sites become adapter-declared facts.
  Guarded by the existing Qwen capacity fixtures. (4) Runner arch gates (~30
  lines). (5) Tests/fixtures: host-suite descriptors, ctest lifecycle, and a
  re-calibrated 512-token anchor with the gemma4 tokenizer plus re-derived
  checker answers. (6) MTP checkpoint inventoried but NOT executed (SDD
  first-acceptance precedent).

Explicit non-goals
: No capacity claim (the model fits in VRAM; gate 4's property is
  unobservable here — a green run proves lifecycle/parity, NOT capacity);
  no MoE row-set work (dense target); no ISWA/sliding-window optimization
  beyond what correctness requires (40 SWA + 8 full-attention layers with
  different head widths and rope bases; the iSWA graph input type exists so
  this is wiring — but it is the widest error bar in the estimate); no speed
  claim unless a separate task invokes gate 7; DeepSeek V4 and Laguna remain
  out of scope per AGENTS.md and size.

Acceptance criteria (falsifiable)
: Host suite green with default-off behavior unchanged; the gemma4
  non-residency control arm and the residency arm produce traces accepted by
  --check-telemetry with slot generations and atomic per-package readiness
  validated; focused greedy ID parity holds between control and residency
  arms on identical shape; an insufficient-budget arm run_refuses with
  reason=capacity_preflight before any timed execution (gate 5 on the new
  arch); the re-calibrated qa-smoke 512 rung passes with the negative
  control failing as designed; Qwen reference regression (qa-smoke 512 on
  the Qwen vehicle) still passes on the same build.

Focused validation
: `cmake --build build/llama-cuda --config Release`;
  `tools/run-tests.ps1 -Suite host -ModelPath <gemma4 path>`; under the GPU
  mutex: control arm (no residency flags) vs residency arm
  (`--residency moe-expert-bank --layer-slots 2 --pp-async-feed 0`) on the
  gemma4 anchor, `--check-telemetry` both, the semantic-QA checker on the
  recalibrated anchor, then the Qwen qa-smoke 512 regression.

State update
: docs/currentstate.md records the pass/fail per gate with trace paths;
  docs/layer-residency-sdd.md is updated ONLY per the SDD's own amendment
  rule; the item 6 brief gains a one-line pointer to the outcome.

Escalate before implementation if
: The gemma4 graph refuses to split at the seam without arithmetic changes
  (the brief's unknown #1 — time-box a probe first); ISWA attention breaks
  the one-attention-shape-per-layer residency model; the inventory
  generalization regresses any Qwen capacity fixture; the owner picks a
  different architecture (re-scope this draft against the brief's §2 table).

### T-GPLR-29: Pinned-pool probe — measure the true pinned H2D ceiling vs the direct-copy path

Status: done (2026-09-16)
Depends on: T-GPLR-28 verdict (first-touch = pay-once-per-process page state),
the 2026-09-16 research pass in docs/currentstate.md (pageable cudaMemcpy
bounce-buffer behavior; cudaHostRegister-on-mmap invalid; PrefetchVirtualMemory
not added to the working set).
Commit: third_party/llama.cpp 68cd1726d (2026-09-16: --copy-source pinned default + probe tooling gates)

Goal
: Measure on this box (RTX 5090, WDDM, Gen5 x16, CUDA 13.2) the actual
  per-copy staging rates the research could not supply from public sources:
  (a) pinned→VRAM DMA ceiling, (b) the direct mmap→VRAM pageable path (must
  reproduce the recorded ~4.4 GB/s warm figure as a control), (c) the
  T-GPLR-18 two-hop chain (mmap→pinned memcpy + pinned DMA), and (d) whether
  cudaHostRegister accepts a mapped file view under WDDM (the unrecorded
  Windows data point).

Scope
: Standalone throwaway microbenchmark (build/gplr29-pinprobe.cu, nvcc,
  outside the vendored tree), one bank-sized (453 MB) transfer mapped from the
  real MoE GGUF so the pageable case IS our copy source. No changes to the
  runner, lib, SDD, or default paths. GPU-mutex run.

Falsifiable prediction
: (a) ≥ 20 GB/s (Gen5 ceiling class) and ≥ 4x the direct path; (b) lands
  within ~15% of the recorded 4.4 GB/s warm figure; (c) sum ≥ direct path rate
  only if the CPU memcpy hop (warm pages, ~10 GB/s class) does not dominate —
  if (c) ≤ (b), the pinned staging chain's win comes ONLY from overlap
  potential, not per-copy rate, which redirects the T-GPLR-18 payoff story;
  (d) fails with cudaErrorInvalidValue as on Linux.

Acceptance criteria
: Probe binary compiles clean; all four cases print per-iteration + min/median
  GB/s; case (b) reproduces the recorded direct rate within 15% (else the
  probe itself is invalid and must be fixed before any conclusion); results +
  interpretation recorded in docs/currentstate.md; no vendored-tree changes.

Focused validation
: nvcc compile; GPU-mutex run of the probe; cross-check (b) against
  gplr28-repeat-3pass.jsonl warm per-copy spans.

Results (2026-09-16)
: All criteria met; three of four predictions falsified — each falsification
  redirects a belief. Probe build/gplr29-pinprobe.{cu,exe}, runs
  build/llama-cuda/gplr29-pinprobe.log (x2, reproducible to <2%) and
  link-state samples; end-to-end arm build/llama-cuda/gplr29-repeat-3pass-
  pinned.{jsonl,log} (checker OK, 20,348 events).

  NEW HARDWARE FACT (nvidia-smi under load): this box's 5090 is the LAPTOP
  part and the link is Gen4 x8 (~16 GB/s theoretical), not Gen5 x16. Every
  prior bandwidth framing that assumed Gen5 is void for this machine.

  Probe (453 MB bank-mapped region of the real GGUF, 5-80 iters):
  A pinned cudaHostAlloc->VRAM 14.4 GB/s (= ~90% of the Gen4 x8 link);
  B direct mmap->VRAM warm 10.0 GB/s; B0 cold first-touch 548 ms (0.83 GB/s)
  on the first probe and 139.5 ms (3.25 GB/s) minutes later on the second
  probe — live confirmation of partial cross-process standby inheritance;
  C1 CPU memcpy mmap->pinned 24.2-24.5 GB/s; C full T-GPLR-18 chain serial
  9.03-9.05 GB/s (SLOWER than direct); D cudaHostRegister on the mapped view
  FAILED with cudaErrorOutOfMemory under WDDM (locked-page quota), same
  practical verdict as the Linux InvalidValue: registration is not viable;
  the pinned pool must stay cudaHostAlloc (as T-GPLR-18 already builds it).

  End-to-end (--copy-source pinned, gplr28 shape, per pass wall vs direct):
  pass 1 7.12 s vs 7.15 s (no change); pass 2 4.10 s vs 3.26 s (anomaly:
  wait 635 ms vs 0.4 ms, compute inflated ~+380 ms — needs its own
  explanation before any claim); pass 3 3.26 s vs 3.23 s (no change),
  transfer 4.60 s vs 4.96 s (-7%). TG 76.4 tok/s, 0 evictions (unchanged).
  VERDICT: no end-to-end PP-wall win from pinned at this shape; the +44%
  per-copy DMA gain is consumed by the serialized CPU memcpy hop. The
  production 4.4 GB/s effective per-copy figure is pipeline overhead and
  stream gaps, NOT the pure pageable-copy rate (probe: 10 GB/s) — the
  research-pass bounce-buffer explanation of the 4.4 GB/s figure is
  REVISED: the pageable penalty on this box/driver is only ~1.44x, and the
  dominant remaining cost is first-touch (cold 3-12x), exactly as T-GPLR-28
  concluded.

  Consequence ranking (measured): 1) keep banks warm in-process (done);
  2) hide the copy chain behind compute (T-GPLR-25 af overlap — the chain's
  only justification); 3) pinned buys +44% DMA headroom for that overlap,
  nothing at serial; 4) the mlock/WS privilege path stays parked pending an
  owner LSP change. No vendored-tree changes were made.

Addendum — af overlap A/B with pinned (same day, same shape, one mutex hold;
the question this task left open: does OVERLAP convert the chain's +44% DMA
headroom into wall time?):

| arm (warm passes 2+3, wall) | af0 | af1 |
|---|---|---|
| pinned (this pair, cache-adjacent) | 3.43 / 3.33 s | **2.80 / 2.87 s** |
| direct (af1 control vs gplr28 af0) | 3.26 / 3.23 s | 3.41 / 3.35 s (no gain) |

Cold pass 1: af1-pinned 6.26 s vs af0-pinned 6.96 s (~10%). TG unchanged in
all four arms (~76–77 tok/s, 0 evictions, handoff auto). All traces checker
OK (af pair: 20,348 / 20,577 events).

VERDICT: overlap converts ONLY with pinned staging — af=1 + direct shows no
wall gain, so the gain is attributable to the pinned chain's DMA headroom
being consumed by the worker, not to overlap per se. Measured gain: ~0.5 s
per warm 4k prefill (13.8–15.5%, vs the ~1.1% warm-pass drift and ~6%
run-to-run noise floor — likely real, modest). Effective warm staging rises
from ~4.4 GB/s-class production figures to 18.33 GB / (2.80–2.87 s wall minus
~1.35 s compute) ≈ 12–13 GB/s-class. The af=1 + pinned combination is the
first configuration measured on this box that beats the warm steady state.
Costs/limits: one pair per cell (af0-pinned re-run included as drift control);
wait-telemetry semantics under af grow (768–958 ms vs 606–636 ms) and need a
dedicated interpretation pass before being read as slot-starvation; af=1
remains an opt-in prototype with the T-GPLR-25 contract. Files:
gplr29-afab-{af0,af1,af1direct}.{jsonl,log}.

Addendum 2 — wait-signature replication verdict (same day, no new GPU runs;
uses the three independent pinned traces already on disk):

The original "pass-2 anomaly" decomposes into TWO separate observations with
OPPOSITE verdicts:

1. The 4.10 s pass-2 WALL spike: NOT a pinned-path cost — not replicated (the
   afab af0 arm's pass 2 ran 3.43 s; its wait matched the systematic 635 ms;
   the spike's extra time sat in an inflated compute figure that did not
   reproduce). Classified environmental one-off (shared-GPU interference
   window during that hold).

2. The ~635 ms pass WAIT: SYSTEMATIC pinned-path behavior — per-pass wait
   totals across four warm passes in two independent runs:
   635.1/635.3/635.3/635.9 ms (agreement 0.8 ms = 0.13%), absent in every
   direct arm (0.4-7 ms). Distribution: ~120 layer_ready_wait events per
   3-pass trace (59 > 1 ms, max 36.4 ms) — a distributed per-layer cost, not
   one spike.

Mechanism (arithmetic, consistent with all arms): the pinned chain's CPU
memcpy hop costs ~18.7 ms x 40 banks = 748 ms/pass; the DMA saving is
~45.5->31.4 ms x 40 = 564 ms/pass; the measured serial wall penalty
(af0-pinned vs direct: +70-170 ms) matches 748-564 = ~184 ms with the hop
nearly fully exposed at af=0; at af=1 the worker hides most of the hop (walls
0.45-0.53 s faster than direct, approaching the 564 ms bound) but the worker
service time (50 ms/bank > ~34 ms/layer compute) makes it the pipeline
bottleneck, which is why the wait account GROWS to 927-958 ms while the wall
SHRINKS. The wait figure is therefore accounting-visible chain exposure, not
direct wall cost — walls and waits moved in opposite directions across the
af pair.

Ceiling implication (expected, unproven): eliminating the per-pass hop
(persistent pre-populated pinned pool across passes, or equivalent) is worth
up to ~0.5-0.6 s/pass; the af=1+pinned warm wall could approach ~2.3-2.5 s.
The exact wait-accounting semantics under af remain an instrumentation open
question.

Addendum 3 — wait-signature investigation (same day; event-level timeline
analysis of the af1/af0 traces + a 3-slot counterfactual arm, 2 runs):

WHAT THE WAIT IS (measured, event level): per-layer layer_ready_wait ends
EXACTLY at that layer's own copy_complete (delta 0.0 ms on every event),
quantized at the per-bank effective DMA window (31.45 ms = the pure pinned
DMA time; outliers 36.2 ms). ~10 of 40 layers wait 0. The wait is therefore
NOT worker lag and NOT an accounting artifact: it is the deficit
max(0, effective_DMA_window(L) - compute_window(L)) summed over layers. At
2 slots: avg effective window 69.6 ms vs ~34 ms compute -> ~23 ms deficit on
~30 layers = the 927-958 ms. MODEL VALIDATED: the same arithmetic predicted
~9 ms/layer x 40 = 363 ms at 3 slots; measured 363.0/363.2 ms (two runs,
0.2 ms spread).

COUNTERFACTUAL ("wall without the wait"): wall - wait = 1.85-1.92 s
(measured subtraction, 2.80-2.87 s minus 927-958 ms). The engine-bound floor
is lower still: 40 x 31.4 ms pure DMA = 1.26 s overlapped with ~0.55 s
sampled compute -> ~1.5-1.8 s if every copy ran at the pure pinned rate with
zero bubbles. The 1.26->2.78 s gap is CONCURRENCY INFLATION: copy windows
stretch ~2.2x when two DMAs overlap on the WDDM copy engine (28 concurrent
window pairs), so achieved aggregate DMA is ~6.6 GB/s vs the 14.4 GB/s pure
copy rate.

PRESCRIPTION TEST (3 slots + af=1 + pinned, T-GPLR-10-legal TG=3 shape,
reproduced twice): FALSIFIED and harmful. Warm walls 4.14-4.83 s (worse than
2-slot 2.80-2.87 s); warm transfer ~9.2-9.5 s vs 3.69-3.73 s (windows inflate
to ~237 ms avg); wait only halves (363 ms) because compute windows inflate
too; cold pass 23.2-43.9 s (vs ~7 s at 2 slots). The copy engine is already
saturated at 2 slots under pinned+af; a third slot adds contention, not
runway. MEASURED IMPLICATION for the owner's slots policy: under
af=1+pinned, 2 slots is the optimum on this box; 3 is counterproductive.
Also recorded: handoff auto declines at 3 slots at this shape (96-token TG
refused; TG=3 legal), and the 3-slot TG leg degraded (0.42 tok/s).

BOTTOM LINE: the af=1+pinned wall without the wait is 1.85-1.92 s (empirical
subtraction) with a ~1.5-1.8 s engine-bound floor. The wait cannot be removed
by slot count (proven) — only by reducing DMA bytes or eliminating the WDDM
concurrency inflation (outside current scope). The wait telemetry is
trustworthy: it reads as chain exposure, matches the deficit model to 0.2 ms
across two slot geometries.

### T-GPLR-32: WDDM DMA-inflation serialization — one H2D in flight at a time (af worker DMA gate)

Status: done (2026-09-16) — verdict: DISCARD the serialization default; the
wall falsifier FIRED. A/B ran under the GPU mutex (tools/run-gplr32.ps1;
ser0 vs ser1, 4k 3-pass 2-slot pinned shape; both traces check-telemetry OK,
20,577 events each, no deadlock). RESULTS: warm passes ser0 2777.8/2768.9 ms
(wait 959.3/958.6 ms) vs ser1 2782.9/2751.8 ms (wait 958.8/958.5 ms) —
identical within noise; the predicted wait collapse and 1.9-2.3 s wall did
NOT materialize. Gate verified operating: af_inflight field present (ser1 max
2 = submitted-but-unconsumed worker DMAs), stream gate enqueued, 120/120
copies per arm. MECHANISM RESOLUTION (sharper than the pre-registered
readings): with worker DMAs serialized, ~114/120 windows remain stretched
(>45 ms; mean 82.6 vs 82.3 ms) while the only possible concurrent partner is
a main-thread copy (preload, ~3-6 per pass) — far too few to stretch 114
windows. The stretch is therefore INTRINSIC per-copy cost under WDDM
(submission/window overhead), not DMA-pair contention: the original
"2.2x inflation from concurrent copies" interpretation is falsified, and
with it the entire DMA-concurrency lever family (3-slot falsification,
T-GPLR-32 falsification). Analyzer caveat: windows are submit->consumption
spans (host-observed; the trace carries no GPU-side timestamps), but the
wall and wait verdicts are independent of that limitation.
--pp-af-serialize stays as opt-in diagnostic tooling (harmless: identical
walls, checkers OK). Remaining PP lever family: shrink the per-copy window
itself (member batching / GPU-direct storage) — new task territory.
Commit: nested 3bc3aff71 (2026-09-16, gate + af_inflight telemetry);
outer 7b83828 (2026-09-16, verdicts + scripts tools/run-gplr32{,-arms}.ps1,
analyzer tools/analyze-gplr32.py, charter)
Depends on: T-GPLR-29 addendum 3 (warm wall engine-bound floor ~1.5-1.8 s;
pinned A/B methodology), T-GPLR-13/25 trace arms (the 2.2x window-stretch
evidence), 2026-09-16 research pass in docs/currentstate.md (NVIDIA engine-
model + WDDM batching sources; asyncEngineCount=5 local measurement).
Commit: (fill on completion)

Goal
: The traces show concurrent two-copy stretches inflating effective DMA
  windows 2.2x (69.6 ms vs 31.45 ms pure; ~237 ms at 3 slots) while the
  device reports asyncEngineCount=5. Hypothesis (from the NVIDIA engine
  model: same-direction copies sharing one engine/link stretch each other;
  batching restores the full rate): limiting the af worker to ONE DMA in
  flight removes the stretch; compute overlap is untouched because copies
  and kernels run in different engines.

Scope
: Gate ONLY the af worker's H2D enqueue on the previous copy_complete
  event. The CPU memcpy hop into the pinned staging buffer must stay fully
  overlapped (do not gate the hop). Keep 2 slots, pinned source, layer-
  major default schedule, 4k 3-pass shape for comparability with the
  T-GPLR-25/29 recorded arms.

Non-goals
: No change to batch-major behavior, no multi-stream copy fan-out, no
  VRAM/land layout changes, no upstream-facing material.

Design/failure contract
: If windows return to ~31-36 ms and per-pass wait collapses toward zero
  with wall dropping toward ~1.9-2.3 s, the contention hypothesis is
  CONFIRMED and serialization is a default candidate (gated behind the
  existing af flag family, loud telemetry). If wall is unchanged AND
  windows remain stretched, the stretch is per-copy WDDM submission
  overhead outside the DMA window — record that and DISCARD. If windows
  are pure-rate but wait persists, the wait accounting itself is the next
  suspect. All outcomes are recorded verdicts, not failures.

Implementation steps
: 1. In the af worker (llama-layer-streaming.cpp), add a one-in-flight gate:
  before cudaMemcpyAsync H2D enqueue, cudaStreamWaitEvent(copy_stream,
  last_copy_complete); after enqueue, record last_copy_complete from
  copy_stream. (Events, not host sync — no CPU stall.)
  2. Add telemetry: an af_inflight field on copy_submit, 0/1, so the
  analyzer can verify the invariant (never two concurrent worker DMAs).
  3. Flag-gate: reuse the existing af mechanism; add
  --pp-af-serialize 0|1 (default 0) so the A/B is one flag.
  4. Build, run A/B (serialize 0 vs 1) under the GPU mutex, 4k 3-pass.
  5. Analyze with the existing per-copy window analyzer + wait accounting.

Acceptance criteria
: - A/B traces both checker OK; af_inflight invariant holds (max 1) in the
    serialized arm.
  - Verdict recorded against the three outcome classes above with numbers
    (window ms, wait ms/pass, warm wall, tok/s).
  - docs/currentstate.md updated; plan status set with the verdict.

Validation commands
: powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1
  -TimeoutSeconds 420 -Command "<runner> --pp-af-serialize 1 ..." (exact
  command recorded in docs/currentstate.md at run time); trace checker per
  T-GPLR-07 smoke rules; window analyzer over both jsonl traces.

State/document updates
: docs/currentstate.md (run + verdict), this task (status + verdict),
  T-GPLR-29 addendum 3 cross-reference if the floor is revised.

Escalation condition
: If the serialized arm regresses beyond noise (>10% wall) or the gate
  deadlocks (wait never fires), stop, capture the trace, and escalate to
  the owner before touching the enqueue path again.

### T-GPLR-33: Member-batched submission — one submission per bank (91 -> ~44-58 ms window)

Status: done (2026-09-16) — verdict: DISCARD as a wall lever; the
falsifier FIRED. A/B under the GPU mutex (tools/run-gplr33{,-arms}.ps1,
analyzer tools/analyze-gplr33.py): V1 verified live (submit_batch=1 on
120/120 banks — contiguity check tiled exactly, single span submission
everywhere) and windows did NOT move: median 76.1 ms (batch1) vs 75.1
(batch0), warm walls 3483.2/3055.4 vs 3418.7/2949.0 ms (batch1 marginally
worse, within variance). The per-submission-overhead model is DEAD; with
T-GPLR-32, the whole submission-shaping family (serialization, submission
count) is closed on this box. Surviving arithmetic: warm wall = 40 x
(~34 ms compute + ~31.5 ms DMA) chained — the DMA does not overlap layer
compute on WDDM (or completion signaling defers to the compute boundary).
Discriminating next step pre-registered in docs/currentstate.md: a
standalone mid-kernel H2D probe (event ~65 ms => true serialization;
~31.5 ms => concurrent DMA with deferred completion). --pp-submit-batch
stays as opt-in diagnostic tooling (default 0). Commit: nested fe6fce575
(2026-09-16, V1 code); outer 96ad669 (2026-09-16, verdict + why-analysis
+ scripts tools/run-gplr33{,-arms}.ps1, analyzer tools/analyze-gplr33.py,
gitlink bump)
Depends on: T-GPLR-32 (done — its falsification LOCATED the cost), the
2026-09-16 follow-up research pass (docs/currentstate.md: warm ready
latency 69.7 ms = 3 x (10.5 pure + ~12.7 per-submission); banks are
exactly 3 members x 144 MiB; cudaMemcpyBatchAsync present in the
installed CUDA 13.2 SDK), T-GPLR-29/30 (pinned staging + pool: sources
already contiguous in member order).
Commit: (fill on completion)

Goal
: Convert the per-bank staging from 3 sequential cudaMemcpyAsync calls to
  ONE submission, removing the measured ~12.7 ms per-submission driver
  overhead from two of the three member copies. This is the direct attack
  on the last standing warm-wall cost after T-GPLR-32 closed the
  DMA-concurrency family: warm 4k wall ~2.77 s -> ~1.8-2.0 s (the
  engine-bound floor band from T-GPLR-29 addendum 3).

Scope
: Two implementation variants, decided by a cheap runtime check at slot
  build time:
  - V1 (preferred, zero new API): if the slot bundle's member tensors are
    verified back-to-back in the destination buffer (dst offset of member
    k+1 == dst offset of member k + nbytes(member k), checked for every
    slot at init and asserted per run) AND the staging source (pool mirror
    or slot.buf_staging) holds the bank contiguously in the same member
    order (it already does by construction), issue ONE cudaMemcpyAsync of
    the full bank bytes instead of 3.
  - V2 (fallback if V1's offset check fails): one cudaMemcpyBatchAsync
    (CUDA 13.2, count=3, srcAccessOrder=cudaMemcpySrcAccessOrderAny,
    optionally the cudaMemcpyFlagPreferOverlapWithCompute hint as a
    sub-variant arm M2).
  Behind an explicit flag (--pp-submit-batch 0|1, default 0) so the A/B
  is one flag; the CPU memcpy hop loop and the pool protocol are
  untouched; telemetry keeps per-bank copy_submit/copy_complete shape
  (one event pair per bank, bytes = full bank).

Non-goals
: No change to the af worker's gating/queueing, the eviction protocol, or
  the main-thread preload path beyond reusing the same helper; no
  GPU-direct storage work (M3) in this task; no batch-major changes.

Design/failure contract
: PREDICTION (falsifiable): mean effective window 91 -> ~44-58 ms
  (host-observed span, per the analyzer caveat), transfer phase per warm
  pass 3.65 -> ~2.4 s, warm wall 2.77 -> ~1.8-2.0 s, PP tok/s ~1,480 ->
  ~2,100-2,300. FALSIFIER: window unchanged (>= ~80 ms) — the ~12.7 ms
  toll lives inside the DMA engine per OPERATION-BYTES rather than per
  SUBMISSION, closing this lever family too and confirming ~2.75-2.8 s as
  the pinned-af optimum on this box. V1 legality: the offset check must
  gate EVERY stage (not just init) — a failed check must fall back to the
  3-call path loudly (telemetry field submit_batch=0 on that bank), never
  copy wrong addresses.

Implementation steps
: 1. At slot build: compute and cache per-slot member offsets; verify
  contiguity (V1) or mark the slot V2-eligible.
  2. Add the --pp-submit-batch flag (parse, usage, module setter with
  loud refusal if the runtime lacks the batch API).
  3. In both staging sites (af_submit_copy, load_layer), replace the
  member loop's tensor_set_async calls with the single-submission helper
  when the flag is on and the slot check passed: V1 one cudaMemcpyAsync
  over (staging_src_base, dst_base, bank_bytes); V2 cudaMemcpyBatchAsync
  with per-member arrays built once per slot.
  4. Telemetry: copy_submit gains submit_batch (0=legacy 3-call, 1=V1
  single, 2=V2 batch API) so the analyzer can verify which path ran.
  5. Build; host-validate (flag parse, refusal path); run the A/B
  (batch 0 vs 1) with the T-GPLR-32 script shape (4k 3-pass 2-slot
  pinned af=1) under the GPU mutex; strict check-telemetry both arms.
  6. Analyze with the T-GPLR-32 analyzer extended for submit_batch; the
  qa-smoke 512 gate is NOT required (staging path only, no code that can
  alter generated output — but run it if any doubt arises).

Acceptance criteria
: - Both arms check-telemetry OK; submit_batch field proves the intended
    path ran on >= 118/120 banks per arm (preload/main-thread copies may
    differ).
  - Verdict recorded against the prediction/falsifier with numbers
    (window ms, wait, warm wall, tok/s) in docs/currentstate.md and this
    task; no silent fallback unaccounted for.
  - If adopted as default: a loud note in the usage line and the plan
    status updated with the promotion decision left to the owner.

Validation commands
: powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-gplr33.ps1
  (mutex wrapper, same shape as tools/run-gplr32.ps1, arms batch0/batch1);
  python tools/analyze-gplr33.py batch0 batch1; strict --check-telemetry
  per arm inside the wrapper.

State/document updates
: docs/currentstate.md (run + verdict + mechanism resolution), this task
  (status + Commit field), T-GPLR-32 cross-reference (its "intrinsic
  per-copy cost" wording is superseded by per-submission cost on adoption).

Escalation condition
: If the contiguity check fails on the real model (V1 unavailable), record
  it and continue with V2; if the V2 arm REGRESSES (>10% wall) or the
  batch API returns unexpected errors on this driver, stop and escalate
  to the owner before any further submission-path changes.

### T-GPLR-31: Locked-RAM probe — does the driver grant a full-pool-size pinned allocation on this box?

Status: done (2026-09-16) — verdict: allocation is NOT the blocker; 18 GiB pool grantable with large headroom
Depends on: T-GPLR-29 (probe methodology; cudaHostRegister OOM precedent;
Gen4 x8 laptop link), T-GPLR-30 (the fixed40 arm already held 17,478 MiB of
pinned pool buffers successfully — the direct motivation for this probe).
Commit: (fill on completion)

Goal
: Answer, before any persistent-pool design work, the question T-GPLR-30
  flagged as the first thing to check: does the CUDA driver grant ~18 GiB of
  `cudaHostAlloc` pinned memory on this box, at what N does it stop, what
  does refusal look like, and is the earlier `cudaHostRegister`
  `cudaErrorOutOfMemory` (T-GPLR-29 case D) a file-backed-pages issue or a
  general registration limit.

Scope
: Standalone throwaway probe (build/gplr31-pinpool-probe.cu, nvcc, outside
  the vendored tree), no runner/lib changes, GPU-mutex run, telemetry-free.
  Four measurements: (A) incremental ladder — allocate 432 MiB pinned blocks
  one at a time up to failure, record the largest successful total and the
  exact cudaError; (B) single-shot ladder — allocate one block of the stated
  size (1, 4, 8, 12, 16, 17, 17.5, 18, 20 GiB) with cudaError propagation,
  to detect a large-allocation path or driver-imposed single-block cap;
  (C) register-on-malloc isolation — cudaHostRegister on ordinary malloc'd
  anonymous memory, separating the T-GPLR-29 failure from file-backed pages;
  (D) before/after GlobalMemoryStatusEx so OS-level commit (vs driver quota)
  can be identified as the binding constraint.

Falsifiable predictions
: (P1) the incremental ladder reaches >= 17 GiB (T-GPLR-30's fixed40 already
  granted 17.07 GiB in-process) but FAILS well before the physical 61.7 GiB
  — the WDDM locked-page quota is far below physical RAM; (P2) single-shot
  large allocations fail EARLIER than the incremental total (fragmentation
  or a per-allocation cap in the WDDM path); (P3) cudaHostRegister on
  anonymous malloc memory ALSO fails (the T-GPLR-29 OOM is quota on
  registration generally, not file-backed-page-specific).

Acceptance criteria
: Probe compiles clean; all four cases print per-step status with exact
  cudaError names; the ladder's failure point and error are recorded;
  GlobalMemoryStatusEx before/after shows the OS-visible effect of the
  pinned pool; results + verdict recorded in docs/currentstate.md and this
  task's status note; no vendored-tree changes; no changes to default
  execution paths.

Focused validation
: nvcc compile; GPU-mutex run of the probe binary; cross-check the ladder's
  successful total against T-GPLR-30 fixed40's 17,478 MiB grant.

Escalation
: if the ladder's ceiling lands BELOW what T-GPLR-30 already granted
  (17,478 MiB), the probe methodology itself is wrong and must be fixed
  before any conclusion is drawn from it.

Results (2026-09-16)
: All acceptance criteria met; ALL THREE PREDICTIONS FALSIFIED, each in the
  direction that EXPANDS feasibility. Probe build/gplr31-pinpool-probe.cu,
  run build/llama-cuda/gplr31-pinpool-probe.log (GPU mutex, CUDA 13.2
  toolkit; the 12.3 toolkit on PATH rejects the Blackwell host compiler —
  same quirk the T-GPLR-29 probe already worked around).

  (A) INCREMENTAL LADDER: 106 x 432 MiB = 44.72 GiB of cudaHostAlloc pinned
  memory granted and touched before cudaErrorMemoryAllocation ("out of
  memory"). P1 FALSIFIED — there is no small WDDM locked-page quota; the
  driver grants to near-physical exhaustion. At the failure point
  GlobalMemoryStatusEx showed 10.11 GiB still available (83% load), so the
  binding constraint sits ~10 GiB above the granted total — either a
  locked-memory cap around ~45 GiB or an OS floor that refuses to give up
  the last ~10 GiB; distinguishing those two is unproven and irrelevant at
  pool scale. Clean accounting: physLoad 10% -> 83% -> 10% across
  allocate/free.

  (B) SINGLE-SHOT LADDER: every size up to 20 GiB allocated AND touched OK
  (1/4/8/12/16/17/17.5/18/20). P2 FALSIFIED — no per-allocation cap, no
  large-block fragmentation penalty; a one-shot 20 GiB pool is grantable.

  (C) REGISTER ISOLATION: cudaHostRegister on ANONYMOUS malloc pages
  SUCCEEDS at 64 MiB and 432 MiB. P3 FALSIFIED, and this REVISES the
  T-GPLR-29 case-D interpretation: registration itself is viable under WDDM
  on this driver; the earlier cudaErrorOutOfMemory was specific to the
  FILE-BACKED mmap view (section-object pages), not to registration
  generally. Consequence beyond the pool: pinning a large anonymous staging
  area is a legal design option here (the "registration is not viable"
  shorthand recorded earlier must not be read broader than file-backed
  views).

  VERDICT: allocation is NOT the blocker for the persistent pool — 18 GiB
  (and far beyond) is grantable, confirming and extending T-GPLR-30
  fixed40's 17,478 MiB in-process grant. The pool's remaining obstacles are
  exactly the T-GPLR-30 findings (warm-wall flat under af=1; populate at
  the cold ~162 ms/bank rate), both of which are design/measurement
  problems, not driver refusals. The two residual pool uses (hot-cache
  populate, af=0 serial staging) lose their feasibility contingency.

### T-GPLR-30: Partial persistent pinned pool — warm-wall savings curve vs RAM cost

Status: done (2026-09-16) — verdict: DISCARD for warm-wall purposes; retained as opt-in tooling
Depends on: T-GPLR-18 (pinned staging buffers), T-GPLR-25 (af worker, rewired),
T-GPLR-28 (first-touch pay-once-per-process), T-GPLR-29 addenda 2-3 (the
~635 ms/pass wait = the chain's 18.7 ms/bank CPU hop; af=1+pinned warm wall
2.80-2.87 s at 2 slots; 3 slots harmful; 61.7 GiB RAM, 432 MiB/bank exactly,
40 banks = 18.1 GB full-pool commitment).
Commit: third_party/llama.cpp 68cd1726d (2026-09-16: --pp-pool acquire/release protocol)

Goal
: Prototype the persistent pinned pool at PARTIAL sizes and measure the
  warm-wall savings curve against its locked-RAM cost. This turns T-GPLR-29
  addendum 2's unproven ~0.5-0.6 s/pass ceiling (full 40-bank pool, expected
  wall ~2.3-2.5 s) into a measured curve the owner can price at any RAM point.

Mechanism
: At init (when --copy-source pinned), allocate N extra bank-sized pinned
  buffers (the per-slot staging buffers remain), memcpy banks into them once,
  and make both stage sites (af worker + load_layer) serve a pooled bank's
  members directly from the pool (skipping the mmap->pinned hop) while DMA
  stays identical. Weights are immutable, so a pooled mirror is byte-exact;
  no arithmetic, routing, or placement change. Telemetry: one pool_hit or
  pool_miss event per bank stage with policy, pool size, and bank id; the
  checker ignores unknown events (forward-compat), so no checker change.

Policies
: fixed — banks 0..N-1 are pooled permanently (the savings-curve dial).
  lru — an N-entry LRU refilled on miss (the naive persistent-cache idea).

Falsifiable predictions
: (P1 LRU cliff) under layer-major, banks are visited in cyclic order, so an
  LRU smaller than the 40-bank working set gets ZERO hits per pass — the 4
  survivors of pass 1 are the first 4 evicted in pass 2. lru4 must show ~0
  hits/pass and no warm-wall change; lru40 == fixed40 (full benefit).
  (P2 curve) fixed-N saves ~18.7 ms x N of worker service per warm pass
  (worker is the af=1 pipeline bottleneck), so walls fall roughly linearly:
  fixed4 ~ -75 ms (inside the ~6% run-to-run noise floor — expect to be
  indistinguishable), fixed10 ~ -190 ms, fixed20 ~ -370 ms, fixed40 ~ -750 ms
  of service time, wall landing ~2.3-2.5 s (the addendum-2 ceiling). Hits must
  equal N per pass exactly in warm passes.
  (P3) pool population costs N x ~19 ms once at init and doubles as
  first-touch warming for pooled banks, so pass-1 cold walls drop slightly
  with N (bank-touch cost moves from pass 1 into init).

Acceptance criteria
: Implementation compiles; --pp-pool parse/refusals loud (requires pinned
  source, N in 1..n_layer, lru refused with the structural-cliff reason if
  kept disabled by design decision); checker OK on every GPU trace; every
  arm's hits/pass and walls recorded; P1/P2/P3 verdicts recorded in this plan
  and docs/currentstate.md; RAM cost table (N x 432 MiB) in the verdict; no
  default-path change (pool off unless --pp-pool given).

Focused validation
: cmake --build build/llama-cuda --config Release --target llama-layer-streaming;
host refusal checks (no GPU); one mutex hold running lru4 + fixed4 + fixed10
+ fixed20 + fixed40 (gplr28 3-pass 4k shape, 2 slots, af=1, pinned, tg=3
legal shape); wall-vs-N table + hits/pass from pool events.

Escalation
: if pinned allocation for N=40 fails (driver locked-page quota — the
  cudaHostRegister refusal precedent), record the actual limit, run the
  largest N that allocates, and surface the quota to the owner (it caps the
  whole persistent-pool idea on this box).

Results (2026-09-16)
: All acceptance criteria met. Five arms under one mutex hold, every trace
  checker OK (1990-2098 events). Arms: lru4 (cliff control) + fixed 4/10/20/40,
  shape = gplr28/29 3-pass 4k, 2 slots, af=1, pinned, TG=3.

  RAM commitment (pool_init bytes): fixed4 1728 MiB, fixed10 4320 MiB,
  fixed20 8640 MiB, fixed40 17478 MiB (1.69/4.22/8.44/17.07 GiB).

  HIT MECHANICS (measured, exact): fixed arms record exactly N pool hits per
  warm pass (4/10/20/40; hit bytes = 3 x N x 432 MiB over 3 passes) and ZERO
  misses — preload + lookahead never double-acquire (the active guard held).
  lru4: 0 hits, ~112 misses (3 passes) — P1 VALIDATED DECISIVELY. Under
  layer-major's cyclic visit order an N-entry LRU < the 40-bank working set
  hits nothing (the pass-1 survivors are the first evictions of pass 2), and
  the fills ADD work: lru4 warm walls 4.17/3.64 s vs the 2.80-2.87 s baseline
  (+27-45% harm). The naive persistent-cache idea is structurally dead here.

  P2 FALSIFIED — THE SAVINGS CURVE IS FLAT AT ZERO. Hits are perfect at every
  N, yet warm walls stay in the baseline band at every N:
  fixed4 3.20/2.91, fixed10 3.10/2.83, fixed20 3.07/2.86, fixed40 2.93/3.02 s
  vs baseline af1-pinned 2.80/2.87 s (noise floor ~6%). The predicted
  ~0.5-0.6 s warm saving (addendum-2 ceiling ~2.3-2.5 s) does not exist.
  WHY (mechanism, closing the chain): the addendum-2 model assumed the
  worker's 18.7 ms memcpy hop made it the pipeline bottleneck; the truth is
  the addendum-3 finding — the wall is set by the CONCURRENCY-INFLATED DMA
  WINDOW chain (wait ends at own-DMA-end), and the pool leaves the DMA
  untouched (same bytes, same engine, same windows). At af=1 the hop was
  ALREADY fully hidden; removing it changes nothing. Addendum-2's
  "~2.3-2.5 s ceiling" is void; addendum-3's own mechanism had already
  predicted this outcome. Warm wall at af=1 is now PROVEN DMA-window-bound
  by two independent falsifications (3-slot geometry, pool hop removal).

  P3 REFRAMED BY MEASUREMENT: cold pass 1 collapses with N (fixed40:
  3.24 s vs baseline 6.26 s = -48%; the fault+hop cost moved out of the
  pass), BUT the populate pays it serially at init: 630/1582/3193/6484 ms
  (157-162 ms/bank = the COLD first-touch memcpy rate, ~2.8 GB/s effective —
  NOT the 24 GB/s warm-page rate, because init overlaps nothing). Net for
  the whole 3-pass process: populate 6.48 + passes 9.19 = 15.67 s vs baseline
  11.93 s — a net LOSS for short-lived processes. The pool's only plausible
  audience: a long-lived process whose OS file cache is HOT at startup, where
  populate should approach ~19 ms/bank (~0.75 s total; unproven — OS-cache-
  state lottery per T-GPLR-28) — and af=0 serial staging, where the 0.75
  s/pass hop exposure is real (not measured here).

  VERDICT: DISCARD the persistent pool for warm-wall purposes. At af=1 every
  N from 4 to 40 buys zero warm improvement for 1.7-17.1 GiB of locked RAM.
  Implementation retained as opt-in tooling (correct: byte-exact mirrors,
  checker OK, exact hit accounting; refusals loud) because the two open
  levers above may justify a cheap owner-decided probe later. No default-path
  change; --pp-pool off by default.

### T-GPLR-28: In-process layer-major PP repeat — first-touch mechanism experiment

Status: done (2026-09-16)
Depends on: T-GPLR-12 (layer-major default controller), the 2026-09-16 lm4k
trio verdict in docs/currentstate.md (layer-major 4k wall 13.30 s vs
predicted 7-8 s; every copy first-touch at ~570 ms vs ~103 ms warm; the
second batch-major arm's sweep 1 still cold after layer-major had touched
every bank, ruling out pure file-cache contents as the carrier).
Commit: third_party/llama.cpp 68cd1726d (2026-09-16: --pp-repeat + per-pass batch lifetime)

Goal
: Decide WHICH first-touch mechanism the 5.5x per-copy penalty is by
  re-running the identical 4k prefill N times inside one process. The
  per-pass copy spans separate the schedule's intrinsic cost from the
  first-touch cost with no new measurement machinery.

Scope
: Opt-in runner flag --pp-repeat N (default 1, no behavior change). Passes
  1..N-1 clear KV (positions restart at 0), reset per-pass stats, emit a
  pp_pass marker event, and re-run the same batch. Slots are deliberately
  not reset: banks left resident from the previous pass are the warm
  restart condition. TG runs once after the final pass. A pp_pass marker
  event (run_id, pass, pp_tokens) indexes the per-pass phase_summaries;
  the checker ignores unknown event types and accepts multiple pp
  summaries (verified in the checker source).

Non-goals
: No scheduler, feed-chain, or copy-source change; no prefetch; no MTP
  support under repeat (refused at arg validation); no claim about TG
  behavior beyond the existing single-TG shape.

Design/failure contract
: Each pass emits one phase_summary with graph_builds referenced to the
  same prebuild baseline (builds0 captured once before the loop), so any
  timed graph construction still trips the checker. A pass that fails
  decode sets exit_status and stops the loop; the run reports failed, not
  refused.

Falsifiable prediction (the experiment's reason to exist)
: If pass 1 warm-mmap copies run at the batch-major warm rate (~103 ms),
  the penalty is host page state that dies with the process (disk-in or
  working-set trim on process teardown) and the lever is persistence
  across process lifetime (pinned staging, mapping reuse). If pass 1 is
  still ~570 ms per copy, something invalidates pages WITHIN the process
  (18.33 GB cycling through a bounded working set is the candidate) and
  only pinned memory helps. Partial warmth = a time-constant trimmer.

Acceptance criteria (falsifiable)
: Host build green; --pp-repeat 0 refused with a clear error; --pp-repeat 2
  + --mtp refused with a clear error; a 3-pass 4k MoE-vehicle arm under the
  GPU mutex produces N phase_summaries + N-1 pp_pass markers in one trace,
  accepted by --check-telemetry, with pass 0 reproducing the recorded trio
  numbers within noise (~13.3 s wall, 40 copies) and per-pass copy spans
  reported at ~1 ms granularity.

Focused validation
: cmake --build build/llama-cuda --config Release; host arg-validation
  refusals; under the GPU mutex the 3-pass arm (--pp-repeat 3, default
  schedule, handoff auto, --layer-slots 2) with --check-telemetry on the
  trace, then the per-pass copy-span analysis over the JSONL.

Results (2026-09-16)
: All acceptance criteria met. Host refusals verified (--pp-repeat 0/17
  -> "must be in 1..16"; repeat 2 + --mtp -> clear error). GPU arm
  build/llama-cuda/gplr28-repeat-3pass.{jsonl,log} (--pp-repeat 3, 4k
  anchor, 2 slots, handoff auto-on): 20,348 events, check-telemetry OK
  after extending the checker for multi-pass traces (pp_pass resets the
  per-layer ascending-chunk state; final-chunk/stride-1/act-bind rules
  made per-pass aware; all 14 lm40/tail/byte-account checker fixtures
  still pass). Per-pass summary (same process, same conditions):
  pass 1 wall 7.15 s, transfer 12.36 s (~309 ms/copy, 38 evictions);
  pass 2 wall 3.26 s, transfer 4.99 s (~125 ms/copy, 40 evictions);
  pass 3 wall 3.23 s, transfer 4.96 s (~124 ms/copy, 40 evictions);
  TG (96, handoff auto) 76.9 tok/s, 0 evictions.

  Verdict on the falsifiable prediction: the "active in-process
  invalidator/trimmer" branch is FALSIFIED — passes 2..3 are identical
  within 1.1% and run at ~4.4x the cold per-copy rate, so the warm state,
  once established, persists for the process lifetime with no decay. The
  first-touch penalty is a pay-once-per-bank-per-process cost. Combined
  with the trio's bm2 observation (a fresh process re-pays sweep 1 even
  seconds after another process touched every bank), the carrier is
  process-lifetime page state, not stable OS file cache; but this run's
  fresh-process pass 1 (~309 ms/copy, between the trio's 570 ms cold and
  103 ms warm) shows cross-process inheritance is partial and
  time-varying — the exact carrier model stays partially open (recorded
  as uncontrolled Windows file-cache/working-set behavior).

  Expected (unproven) consequence: a long-lived server process should
  hold the warm rate for repeated prefills (~1,255-1,270 tok/s at 4k
  measured in-pass), and the persistence levers are pinned staging
  (--copy-source pinned, T-GPLR-18) or a server-style resident process —
  both unmeasured.

  Follow-up finding (not in the original contract): the checker's
  multi-pass extension is generic — any future feature that emits more
  than one prefill per trace relies on it.

State update
: docs/currentstate.md records the per-pass table and the mechanism
  verdict with the trace path; this task's Status/Commit fields updated.

Escalate before implementation if
: The layer-major controller refuses a second sequential prefill on the
  same context (residency state machine does not reset) - that would be a
  real lifecycle defect to scope separately, not paper over here.
