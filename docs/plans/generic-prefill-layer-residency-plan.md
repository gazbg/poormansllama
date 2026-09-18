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
  analysis (tools/analyze-pp-copy-gaps.py, tools/analyze-pp-overlap.py over
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
  Analysis: tools/analyze-pp-overlap.py + tools/analyze-pp-copy-gaps.py on
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

Status: CLOSED — branch B (ADVISORY) executed 2026-09-18. **Owner decision:
ADVISORY — the (b)-versus-0.6 reading binds nothing, and the higher-throughput
retention depth is NAMED PER RUN (K=8) instead of being defaulted.** Branch B
is the documentation-only branch, so this closure changes no source and makes
no gate verdict: no gate PASS exists at any depth, before or after this pick.
The 2026-09-14 brief's "should gate eligibility BIND provisioning or stay
ADVISORY?" is answered here; the decision material below is kept as the record
of what was weighed, including the unbuilt branch-A prerequisite chain.
Depends on: T-GPLR-22 (the PRE-SINK run_refused plumbing the brief names as
the existing shape for a gate-ineligible provisioning refusal; the missing
part is the oracle, not the plumbing).
Commit: 4c1f339 (branch B, documents only — SDD sections 4/10 + this plan +
  currentstate; ZERO source changes, so no vendored-tree gitlink bump)

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

#### T-GPLR-26 RESULT (2026-09-18): branch B executed — ADVISORY, zero source changes

Owner pick: **ADVISORY** (branch B), taken together with the depth decision —
the owner names the retention depth per run to take the higher-throughput option
(**K=8**, `--residency-tg-retain 8`, requires `--residency-tg`) rather than have
a default chosen for them. Branch B's acceptance criteria, checked: the SDD
scope note and the K=4 caveat exist and name the brief, `git diff` for this task
shows ZERO source changes — the closure is documents only
(`docs/layer-residency-sdd.md` header amendment + section 4 depth amendment +
section 10 provisioning-status note; section 11 gate 9 wording unchanged) — and
the host suite is unaffected because nothing executable changed.

Why the depth pick makes the gate question moot at this operating point: on the
non-echoing (honest) curve the 0.6 threshold is cleared by K=6 (0.6487) and K=8
(0.7283) and missed by K=4 (0.5355). A binding rule would therefore refuse the
DEPRECATED default, not the depth the owner now names — there is no refusal left
to exercise at K=8, and branch A's prerequisite chain (tracked oracle, verdict
rule, absent-depth behaviour, new refusal reason + trace fields) would buy
nothing the `slot`/`other` ledger and `rows_reused` do not already show. That is
the recorded reason the cheaper branch was chosen: not that binding was found
wrong, but that it has no work to do at the chosen depth.

Recorded WITH the pick so it cannot be read as an endorsement — three limits, all
measured: (1) the +12.609 tok/s at K=8 came from the OPTIMISTIC echo pairing of
the T-GPLR-19 A/B (that table's `(b)off` column is the echoed series).
**MEASURED LATER THE SAME DAY, and this limit is now partly closed:** the same A/B
re-run at K=6 and K=8 on the declared non-echoing prompt (`build/calc-prompt.txt`,
885 tokens; one mutex acquisition per rung, A,B,A,B inside each) PASSED criteria 1
and 2 at both rungs and returned honest gains of **+9.944 tok/s at K=6**
(echoed +10.196) and **+11.414 at K=8** (echoed +12.609). Read that carefully:
the BYTE-level echo inflation is confirmed and large (**+23.98% and +21.94%** more
bytes staged per round on the honest stream), while the tok/s delta sits INSIDE
the within-arm B spread (0.817 and 2.013), so the honest gain is
INDISTINGUISHABLE from the echoed one at this sample size rather than proven
uncompressed. The pre-registered honest (b) prediction was reproduced within
0.840% and 1.157%, with `rows_retained` equal to the offline `hits_total` exactly
(52,930 / 59,432) and reuse on 40/40 layers. Full table, the length confound
(honest and echoed also differ in PP length, 885 vs 512), the artifact-path
defect and the evidence paths: `docs/currentstate.md` 2026-09-18 "ROW-CACHE A/B
ON THE HONEST STREAM"; driver `tools/run-gplr19-rowcache-ab-honest.ps1`;

**Vehicle rule, from the same day:** the non-echoing prompt is now the DEFAULT
vehicle for row-cache rungs (`tools/run-gplr19-rowcache-ab.ps1`,
`-Vehicle honest`), and the echoed prompt is opt-in reproduction that labels
itself as the optimistic pairing in its own log. A depth decision made on an
echoed rung alone is no longer an acceptable reading: every run logs the other
vehicle's gain at the same K, and criterion 1 now checks the reactive arms
against the recorded baseline's route digest and greedy ids rather than only its
byte counts. Note this is a DEFAULT plus a labelled opt-out, not a gate: nothing
mechanically blocks someone from citing an echoed rung. The rule is recorded
here and in SDD section 4 so that citing one is a visible choice. (2) the row cache is a
TG-phase residency class, so NO depth changes prefill — this pick does not move
the program's primary PP objective; (3) K=8 holds 4370.1 MiB of the row-cache
allowance and its marginal K6->K8 step converts at 7.69 GB/s, the worst step in
the ladder, so it is the most tok/s and the least efficient per byte. A NAMED
depth is a contract: if it does not fit the allowance the run refuses loudly
(`row_cache_depth`) instead of silently reducing it.

#### T-GPLR-26 FOLLOW-UP (2026-09-18, owner-directed): the bare flag's default depth is raised from K=4 to K=8

Status: DONE (owner decision taken immediately after the advisory closure above).
**Why:** the advisory closure reasoned that at K=8 a binding gate "has no work to
do", and called K=4 "the DEPRECATED default" — but the DEFAULT itself was still
K=4, i.e. the one measured depth that MISSES the 0.6 threshold on the non-echoing
stream that now decides depth evidence (0.5355, against 0.6487 at K=6 and 0.7283
at K=8), while already costing ~2.19 GB. The owner picked raising the default to
K=8: the depth with the best measured honest (b), the best measured honest TG
gain (+11.414 tok/s) and the owner's own operating depth. The superseded K=4
sentence in the RESULT above is retained as the record of the reasoning then.

Scope: ONE constant and its two prose sites in
`third_party/llama.cpp/tools/layer-streaming/main.cpp` — `kRowSetRetainDefault`
4 -> 8, the comment that justified K=4 (rewritten: its 0.6284 came from the echoed
stream, and the non-echoing re-probe it named as its own validation MISSES at
K=4), and the `--help` text. No other behaviour moves: the depth is still fitted
DOWNWARD to the cache allowance and never upward, and a NAMED depth is still
never reduced.

This is a DEFAULT change, NOT a gate verdict: **no gate PASS is established at
K=8 or at any other depth** — gate 9 is still read at the depth a claiming task
declares, and the honest stream's non-echoing certification remains partial
(16-gram criterion met, compression criterion not).

Acceptance evidence, all measured 2026-09-18 (one mutex hold per run, each leaving
a verdict and a `.done` sentinel):
- Build: the `llama-layer-streaming` target compiles; host-only tests **#49
  (route-topk) and #50 (host-mem) pass 2/2**.
- Bare flag on the MoE vehicle (512-anchor prompt, `--layer-slots 4`) resolves to
  **K=8**: `row-cache (T-GPLR-19): default depth K=8 fitted to the device
  (4582408192 B of a 17149460480 B cache allowance)`, alloc line `40 layers x 64
  rows/layer = 4370.1 MiB private VRAM, K=8`, `rowset_init` carries
  `"retain_k": 8`, `--check-telemetry` OK, peak VRAM 10161.56 MiB, exit 0.
- The ceiling still fits DOWNWARD: the same bare flag with `--reserve-mib 14000`
  -> `provisioning the largest fitting depth K=5 (2864250880 B)`, `retain_k: 5`,
  2731.6 MiB, exit 0. (K=6 would have needed 3436314624 B against a 3006267392 B
  allowance, so K=5 is the correct largest fit — the search takes the largest
  depth at or below the request, not a fixed step.)
- A NAMED depth is still never reduced: explicit `--residency-tg-retain 8` under
  that same allowance refuses loudly — `run_refused` with `reason:
  row_cache_depth`, `bytes_needed 4582408192` vs `bytes_allowed 3006267392`,
  naming K=5 as the largest depth that fits, exit 1.
- Consumers: no CTest target and no driver uses the bare flag; the only script
  that does is `tools/run-tests.ps1`'s `retention` semantic-QA smoke mode, whose
  comment ("bare flag = the device-fitted default depth") stays true while its
  depth changes from K=4 to K=8. Any retention-mode QA rung recorded before
  2026-09-18 therefore ran at a DIFFERENT depth than one recorded after it.
- One probe DISCARDED as mis-parameterized and its artifacts deleted: a first
  squeeze attempt used `--reserve-mib 19600`, which trips the T-GPLR-04 capacity
  preflight (bounded residency + reserve exceeds free VRAM) BEFORE the row cache
  is fitted, so it measures nothing about this default and is not kept.

Escalation: a bare-flag run refused for `row_cache_fit` (not even K=1 fits) means
the ALLOWANCE is too small for any retention — raise `--reserve-mib`/free VRAM or
set `--residency-budget-mib`; it is not a reason to lower this default.
Commit: engine `068ee0c75` (vendored tree); outer `68db1ef` (gitlink bump plus
these records).

### T-GPLR-27 (CLOSED — DEFERRED 2026-09-18 by owner decision): Second-architecture first pass — dense target (item 6 brief)

Status: CLOSED — DEFERRED (2026-09-18, owner decision). **Owner, verbatim: "I do not
need to prove genaralization now, note it and close it."** This is closed as
INTENTIONALLY NOT PURSUED, not as done: no second-architecture adapter exists and
none is planned, and the T-GPLR-04 preflight / T-GPLR-22 refusal path remain
arch-independent in code but exercised against the Qwen MoE vehicle only — so no
generic-architecture claim is made or implied by closing this. The draft below is
kept as the design record and as the starting point if that decision is reversed.
Historical decision material (for such a reversal) is
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
Commit: NO code commit - the task was CLOSED-DEFERRED before any implementation, so there
  is no engine or nested change to point at. The block was drafted in outer 26762d1
  (2026-09-16, the commit that introduced its text - verified: the section does not exist
  in 26762d1^), and the deferral itself, with the owner's verbatim decision, was recorded
  in outer fe9c1cf (2026-09-18).

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
: Standalone throwaway microbenchmark (tools/gplr29-pinprobe.cu, nvcc,
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
Depends on: T-GPLR-29 addendum 3 (warm wall engine-bound floor ~1.5-1.8 s;
pinned A/B methodology), T-GPLR-13/25 trace arms (the 2.2x window-stretch
evidence), 2026-09-16 research pass in docs/currentstate.md (NVIDIA engine-
model + WDDM batching sources; asyncEngineCount=5 local measurement).
Commit: nested 3bc3aff71 (2026-09-16, gate + af_inflight telemetry);
outer 7b83828 (2026-09-16, verdicts + scripts tools/run-gplr32{,-arms}.ps1,
analyzer tools/analyze-gplr32.py, charter)

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
everywhere; CONFIRMED by direct branch telemetry after the flag-vs-branch
fix, see below) and windows did NOT move: median 76.1 ms (batch1) vs 75.1
(batch0), warm walls 3483.2/3055.4 vs 3418.7/2949.0 ms (batch1 marginally
worse, within variance). The per-submission-overhead model is DEAD; with
T-GPLR-32, the whole submission-shaping family (serialization, submission
count) is closed on this box. Surviving arithmetic: warm wall = 40 x
(~34 ms compute + ~31.5 ms DMA) chained — the DMA does not overlap layer
compute on WDDM (or completion signaling defers to the compute boundary).
Discriminating next step pre-registered in docs/currentstate.md: a
standalone mid-kernel H2D probe (event ~65 ms => true serialization;
~31.5 ms => concurrent DMA with deferred completion). PROBE RUN
(T-GPLR-34, 2026-09-16, see docs/currentstate.md): neither arm of the
pre-registered dichotomy — the third mechanism won. Two-stream arm [6]
shows the copy OVERTAKING the kernel (copy start -31.6 ms rel. kernel
start; copy window pure 31.38 ms; kernel window unaffected 30.36 ms):
WDDM cross-engine serialization and deferred signaling are both
falsified. Single-stream arm [5] reproduces the chained 62 ms additive
(k+copy) as the CUDA stream-ordering reference. CORRECTION (same day,
before any fix was attempted): the runner does NOT stage on the compute
stream — `backend_copy` is a dedicated CUDA backend with its own context
and stream (llama-layer-streaming.cpp:1305, common.cuh:1420/1494), and the
compute side is the SCHEDULER's CUDA backend (`backend_compute`, resolved
at :2274), a different context again. The chaining is therefore NOT stream
identity but the runner's own CROSS-CONTEXT ordering gate
(ev_consumer_done: created on the copy device :1324, recorded on
backend_compute :2105, waited on the copy stream :602/:1900) — a shape the
probe (two streams inside ONE context) never tested. NEXT LEVER: DRAFTED
AS T-GPLR-35 (the task block lands with the next doc update) — issue the
bank staging DMAs on a SECOND STREAM INSIDE THE COMPUTE CONTEXT so every
staging wait is intra-context; predicted warm
4k ~2.75-2.8 -> ~2.0-2.1 s (bounded by max(compute, transfer) per layer),
with the falsifier pre-registered in the task. --pp-submit-batch
stays as opt-in diagnostic tooling (default 0). GAP CLOSED (later the same
day): submit_batch now reports the ACTUALLY-TAKEN branch threaded from
stage_members (0 = per-member, 1 = single span, 2 = mixed) instead of the
flag state; verification A/B: batch0 {0:120}, batch1 {1:120} — the span
path demonstrably ran on every bank of the falsified A/B. Telemetry-fix
commits: nested a03fe510e, outer 15615a9 (2026-09-16, branch-true
submit_batch, analyzer docstring, gitlink bump). V1
code commit: nested fe6fce575
(2026-09-16, V1 code); outer 96ad669 (2026-09-16, verdict + why-analysis
+ scripts tools/run-gplr33{,-arms}.ps1, analyzer tools/analyze-gplr33.py,
gitlink bump)
Depends on: T-GPLR-32 (done — its falsification LOCATED the cost), the
2026-09-16 follow-up research pass (docs/currentstate.md: warm ready
latency 69.7 ms = 3 x (10.5 pure + ~12.7 per-submission); banks are
exactly 3 members x 144 MiB; cudaMemcpyBatchAsync present in the
installed CUDA 13.2 SDK), T-GPLR-29/30 (pinned staging + pool: sources
already contiguous in member order).
Commit: nested fe6fce575 (2026-09-16, V1 member-batched staging, `--pp-submit-batch`);
nested a03fe510e (2026-09-16, submit_batch reports the actually-taken staging branch);
outer 96ad669 (2026-09-16, verdict + why-analysis + scripts tools/run-gplr33{,-arms}.ps1
+ analyzer tools/analyze-gplr33.py, gitlink bump); outer 15615a9 (2026-09-16, branch-true
submit_batch + analyzer docstring, gitlink bump). All five verified present in this tree.

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

### T-GPLR-35: PP staging overlap — bank DMAs on a second stream inside the compute context (warm 4k ~2.75-2.8 -> ~2.0-2.1 s)

Status: CLOSED (2026-09-18) — DISCARD the CUDA stream-topology lever; the closure verdict, the acceptance disposition and the default decision are at the END of this block. Closing this task sets NO default.
Historical status (2026-09-16): in-progress — RECON RUN DONE, FALSIFIER CLASS REACHED
(NONE). V1 implemented, built clean, and measured: with the aux stream (ov1)
the warm 4k wall is 3088.4 ms vs the copy-backend baseline's 3089.0 ms —
measurably identical — and in BOTH arms the copy windows are already pure
(median 31.4 ms = 1.00x the 31.37 ms pure-rate ceiling) with lag >= 0 on
76/76 samples (no DMA ever starts while the compute backlog drains). The 2.75
-> 2.0-2.1 s prediction therefore FAILS and its falsifier fires: the coupling
is not stream/context-level. The warm-pass ledger names what it IS instead:
of the 3089 ms wall, compute is 1350 ms and the pure DMA is 1238 ms, while the
host transfer window (3988 ms) splits into a **1559 ms host hop (pinned mirror
+ submit)** and **2429 ms ready latency, of which only 1238 ms is engine time
and 1191 ms is queue wait** — i.e. the wall is dominated by host-side pipeline
and one-bank-deep ordering, not by the engine and not by stream topology. The
old "74 ms chained window" was a host-window artifact: the host-observed
window (~100 ms/bank) is hop + queue + DMA. A real V1 defect was found and
fixed by the recon: `backend_compute` was published ONLY inside
`layer_streaming_compute`, which layer-major PP never calls (it executes graphs
directly at :3274), so on the primary workload the aux mechanism could not
engage at all (ov1 aborted loudly) and the recon silently emitted nothing;
it is now published in `layer_major_pp` with the loud aux-stream refusal, and
the recon emits `recon_unavailable` rather than a silent no-op. Remaining
levers per the goal charter: host-hop removal (direct mmap staging) and
pipeline depth (more than one bank in flight), NOT CUDA stream topology.
Evidence: build/llama-cuda/gplr35-recon-{ov0,ov1}.jsonl (both checkers OK),
tools/run-gplr35-recon.ps1, tools/analyze-gplr35-recon.py, wrapper log
build/llama-cuda/gplr35-recon-wrapper.log; full numbers in docs/currentstate.md.
NOT a performance result: no wall moved, so no default changes are proposed
here. Staging DMAs now
run on the auxiliary stream (No. 1) of the COMPUTE backend's CUDA context
behind `--pp-copy-overlap 0|1` (default 0 = the pre-T-GPLR-35 copy-backend
path, byte-identical). New experimental surface (ggml/include/ggml-cuda.h +
ggml-cuda.cu): ggml_backend_cuda_aux_stream_ok / memcpy_1d_aux /
event_record_aux / event_wait_aux / aux_synchronize. Every staging submission
and every staging-side ordering call in llama-layer-streaming.cpp now routes
through one plumbing layer (ls_stage_aux/record/wait/drain), so the mode
switch is atomic — an event can never be recorded on a stream that carries no
copy work; drains are conservative (both streams) so no site can be missed;
the handoff-migration and TG row-cache paths keep the copy backend and their
own consistent topology; a missing compute backend with the flag on fails
loudly instead of falling back silently. A/B armed: tools/run-gplr35.ps1 +
tools/run-gplr35-arms.ps1 (overlap0 vs overlap1, gplr32/33 shape) +
tools/analyze-gplr35.py. NOT YET RUN (no GPU time spent). Recon limitation
stated, not hidden: stream/context identity is not logged, so the attribution
rests on the A/B's wall, window and pass-transfer evidence.
Depends on: T-GPLR-33 (done — closed the submission-shaping family), T-GPLR-34
(the standalone two-stream probe: cross-engine overlap is proven in hardware;
one-stream work is additive by CUDA semantics), T-GPLR-32 (done — the af gate
is wall-neutral), T-GPLR-29/30 (pinned staging + pool defaults), T-GPLR-13/25
(the trace arms this A/B stays comparable with)
Commit: nested 69799064a (V1 aux-stream staging + `--pp-recon` instrumentation +
  the layer-major `backend_compute` publication fix); outer ed2dd39 (docs,
  recon scripts, analyzer, gitlink bump); outer b7cd387 (pipeline regression
  gate + its runner wiring + the slot-depth addenda 1/2 verdicts)

Goal
: Recover copy/compute overlap in the runner so the warm 4k layer-major pass
  stops paying compute AND transfer serially (~68.7 ms per layer = ~34 ms
  compute + ~31.5 ms copy). T-GPLR-34 arm [6] proved the hardware can do it:
  with the bank copy on a second stream inside the SAME context the copy
  window stayed pure (31.38 ms min) and the kernel window was unaffected
  (30.36 ms), the copy even overtaking the kernel; arm [5] gives the
  one-stream additive reference (62.12 = 30.74 + 31.37). Target: warm 4k
  ~2.0-2.1 s (PP tok/s ~1,480 -> ~1,900-2,050).

Known evidence and a correction to the T-GPLR-34 write-up
: The runner does NOT stage on the compute stream. `backend_copy` is a
  dedicated CUDA backend (llama-layer-streaming.cpp:1305
  `ggml_backend_cuda_init`) whose `ggml_backend_cuda_context` owns its own
  `streams[device][...]` array (common.cuh:1420, `stream()` = :1494), while
  the compute side is the scheduler's CUDA backend (`backend_compute`,
  resolved at :2274) — two contexts, two streams. The "shared stream
  identity" causal sentence recorded during T-GPLR-34 is SUPERSEDED. The
  recon step below must attribute the chaining among the remaining
  candidates: (i) CROSS-context event waits — `ev_consumer_done` is created
  on the copy device (:1324), recorded on `backend_compute` (:2105) and
  waited on the copy stream (:602 worker, :1900 main); the probe used two
  streams in ONE context, so cross-context wait behavior under WDDM is
  untested; (ii) the slot-reuse gate itself (`consumer_pending` ->
  ev_consumer_done wait; with 2 slots the target slot's consumer is two
  layers back, so this should not chain, but the event's recorded position
  must be verified); (iii) the worker's H2D enqueue position relative to the
  compute graph (the 21.5 ms host hop is already overlapped for af=1,
  T-GPLR-29); (iv) graph-launch semantics (`cudaGraphLaunch`,
  ggml-cuda.cu:4222, plus the `ev_copy_ready` dependency).

Scope
: The staging DMA path only: `stage_members` (both call sites, worker and
  main), the slot events that order copies against consumers, and the flag +
  telemetry around them. Keep 2 slots, pinned staging/pool source, one H2D per
  bank (span path when the contiguity check passes), the layer-major default
  schedule, and the 4k 3-pass shape for comparability with the T-GPLR-29/32/33
  arms.

Non-goals
: No change to staging bytes, member order, or member count; no routing/KV/
  session-state change; no TG-side work; no multi-GPU; no new dependency; no
  default flips without an owner decision; no upstream-facing material.

Design/failure contract
: 1. RECON FIRST (cheap; decides the variant from evidence, not assumption):
  log the stream/context identity of the compute and copy paths at init and
  per-bank copy/compute GPU timestamps, so the analyzer can say whether the
  copy STARTS during the layer's compute interval and what the copy's own GPU
  window is. If the copy already starts mid-compute with a pure window while
  the wall stays chained, the coupling is on the consumption side and this
  task STOPS with that verdict.
  2. V1 (the mechanism implemented here): issue the staging DMAs on a SECOND
  STREAM INSIDE THE COMPUTE CONTEXT (`ggml_backend_cuda_context::
  stream(device, 1)`, lazily created by the existing accessor) and move every
  staging ordering event into that context so ALL waits are intra-context —
  the configuration T-GPLR-34 proved overlaps. Reached through the existing
  experimental ggml-cuda surface (precedent: `#include "ggml-cuda.h"` at
  llama-layer-streaming.cpp:30 and `ggml_backend_cuda_host_buffer_type()` at
  :936). One H2D per bank, bytes and order unchanged; the slot-reuse
  invariant (no overwrite before the prior consumer completes) is preserved
  by keeping the consumer-done dependency, expressed as an intra-context wait.
  3. V2 (only if recon exonerates (i)): relax the gate ordering — e.g. record
  the slot's consumer-done at the consumer's enqueue bound rather than graph
  completion, or stage one layer earlier. Must be justified by the recon
  telemetry; no speculative reordering.
  4. Loud failure: with the flag on, if the second stream cannot be created or
  any staging wait would remain cross-context, abort with structured
  telemetry (never silently fall back to the serialized path, which would
  manufacture a false null).
  5. Flag `--pp-copy-overlap 0|1` (default 0) so the A/B is one flag; value 2
  is refused with the contract message.

Implementation steps
: 1. Recon instrumentation (contract 1) plus one short GPU run to read it.
  DONE 2026-09-16: `--pp-recon 0|1` + `recon_identity` / `recon_attr` /
  `recon_unavailable` events (timing-enabled events on the actual staging and
  compute streams), `tools/run-gplr35-recon.ps1` under the mutex, analyzer
  `tools/analyze-gplr35-recon.py` (distribution + warm-pass ledger). Verdict:
  NONE — see the Status block.
  2. Add the experimental CUDA accessor for the compute context's second
  stream and the per-bank copy helper in the isolated path; wire
  `stage_members` and the slot event creation/recording to it behind the flag.
  3. Telemetry: per-bank copy GPU timestamps plus a compute-window field so
  the analyzer can report the overlap fraction; one init line carrying the
  stream/context identity.
  4. Build; host validation (flag parse/usage, refusal on 2, non-CUDA builds
  unaffected, event protocol still ordered).
  5. A/B under the GPU mutex: `tools/run-gplr35.ps1` (T-GPLR-32/33 script
  shape: arms overlap0/overlap1, strict check-telemetry per arm, one mutex
  hold) on the 4k anchor, 3-pass, 2 slots, af=1, pinned.
  6. Analyze with `tools/analyze-gplr35.py`: per-bank copy start vs compute
  interval, copy window, wait, phase walls, total PP wall and tok/s.
  7. Record the verdict and the attributed mechanism; update this task's
  status and Commit fields.

Acceptance criteria (falsifiable)
: - Warm 4k per-pass wall falls from the recorded 2.75-2.8 s into ~2.0-2.1 s
  (ideal ceiling ~1.5-1.6 s if the overlap is perfect); PP tok/s ~1,480 ->
  ~1,900-2,050; the transfer phase does not increase.
  - Mechanism evidence: in the overlap=1 arm the copy STARTS during the
  layer's compute interval for a majority of banks, the copy's own GPU window
  stays ~31.5 ms, and the per-pass wait collapses.
  - FALSIFIER: warm wall >= ~2.6 s with the copy still starting at/after
  compute end (or the copy window unchanged) -> the coupling is not
  stream/context-level; record it, close the stream-family lever, and
  escalate (remaining territory: GPU-direct storage, platform settings).
  - Both arms check-telemetry OK, no deadlock, refusal path exercised.
  - No qa-smoke required (staging-path only, cannot alter generated output)
  unless a run shows any output difference; if it does, stop and escalate.

Validation
: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-gplr35.ps1`
  (wraps `tools/with-gpu-lock.ps1`; ~20 min) then
  `python tools/analyze-gplr35.py overlap0 overlap1`. Expected: the
  acceptance numbers above, with the falsifier reported explicitly either way.

State/document updates
: docs/currentstate.md (recon finding, run, verdict, mechanism attribution),
  this task (status + Commit), and the T-GPLR-33/34 notes (their superseded
  causal wording corrected). If V1 wins and becomes a default candidate that
  is an owner decision — recorded, not applied here.

Escalation condition
: Stop and escalate if the recon cannot attribute the coupling after ~30 min
  of GPU time; if V1 requires changes outside the isolated experimental path
  (public ggml/llama API, allocator, non-CUDA builds); or if the flag-on path
  cannot guarantee the slot-reuse invariant.

Addendum 1 (2026-09-16) — slot-depth probe: does a third slot convert the
recon's queue wait into wall time? RUN, mechanism answered, prediction band
NOT reached.
: Question and pre-registration: the recon measured a 3089 ms warm wall with
  1191 ms of engine queue wait and 747 ms of consumer wait at one bank of
  pipeline depth; the prediction was that 3 slots shrink both and bring the
  warm wall to ~2.2-2.6 s, with the falsifier being a wall >= ~2.9 s at
  unchanged waits. A/B: tools/run-gplr35-slots.ps1 under the mutex, adjacent
  arms in one hold (2-slot control re-run in-session), 4k anchor, ubatch 512,
  layer-major, kv q8, af=1, pinned, --pp-repeat 2, --pp-recon 1, strict checker
  per arm; traces build/llama-cuda/gplr35-slot{2,3}.jsonl.
  Result (measured): warm 3377.8 -> 2843.2 ms (-534.6 ms, -15.8%), PP 1212.6 ->
  1440.6 tok/s (+18.8%), cold 6364 -> 5697 ms. The MECHANISM is not the
  predicted one: engine queue wait GREW (1155 -> 2667 ms) while consumer wait
  COLLAPSED (734.4 -> 225.3 ms) and the residual `wall - (compute + pure DMA)`
  fell 770 -> 100 ms. Deeper lookahead does not remove queueing; it takes the
  queueing off the critical path, which is why the wall gain (-534.6 ms) is
  essentially the consumer-wait drop (-509 ms). Per-bank DMA windows stay pure
  at both depths (median 31.4 ms = 1.00x ceiling) and lag is never negative
  (0/76, 0/66), so this is still not a rate or stream-ordering effect.
  Cost: +522,194,944 B (498 MiB) of slot VRAM per extra slot (run_start:
  1,044,389,888 -> 1,566,584,832 B) for the 15.8% wall gain.
  Open, unattributed: the per-pass compute total rose 1369 -> 1611 ms (+18%)
  with identical bytes and arithmetic — unexplained, not a claimed result.
  Prediction band 2.2-2.6 s not reached (2843 ms); the falsifier did not fire.
  Default unchanged: the slot count remains an owner decision (default 2, the
  ladder may take a third slot when the budget allows). Next lever per the
  ledger: the 1607 ms host hop (direct mmap-sourced staging).
  Regression gate added (tools/check-gplr35-pipeline.py, wired into the runner
  after the strict checkers): gates warm PP throughput, consumer wait share,
  exposure = wall - compute - pure DMA, and the bank DMA window vs the pure
  ceiling, per depth, with a same-session `--baseline` pair as the trustworthy
  form and exit 2 when no criterion applies. It deliberately does NOT gate the
  recon's lag signal: that signal is 0%-negative at BOTH depths, so gating it
  would be vacuous. Validated on the four existing traces (all PASS), two
  negative controls (both FAIL correctly), and the two error paths (exit 2).

Addendum 2 (2026-09-16) — long-context depth probe: do 3/4 slots still fit and
still pay at the crowded-VRAM shape? RUN; depth does NOT pay there.
: Question and pre-registration: the 4k shape rewards depth (-15.8% at 3 slots),
  but at 200k tokens each bank is consumed across ~391 chunks, so staging should
  already be hidden and depth should buy little; predicted <= ~5% wall change,
  falsified only if a slot jump cut the 200k wall by > 10%. A/B (one mutex
  hold, tools/run-gplr35-slots262k.ps1): arms slots3 / slots4 / slots4-tight,
  200,000-token prefill on the 262,144-token F16 KV configuration, layer-major,
  ubatch 512, af=1, pinned, bf16 slabs, --pp-repeat 2, --pp-recon 1, strict
  checker per arm.
  Result (measured): warm 2 slots 1363.2 -> 3 slots 1369.6 -> 4 slots 1374.7
  tok/s (walls 146,713 -> 146,030 -> 145,487 ms): +0.5% then +0.4%, +0.8%
  total. Prediction confirmed, falsifier did not fire. Consumer wait 1,182.5 ->
  36.5 -> 36.5 ms while the wall stays compute-dominated (92%); summed host
  transfer windows grow 137.7 -> 269.5 -> 400.3 s with no wall change, i.e. the
  queueing at this shape is already fully overlapped. Bank windows stay pure
  (1.00x/1.01x) and lag stays 0% negative.
  Fit and refusal: the 4-slot run completes under the derived budget (preflight
  window arithmetic: 8,809,572,864 B for 4 slots vs ~20.7 GiB free); with
  --residency-budget-mib 1500 the same config refuses LOUDLY at capacity
  preflight with both the bounded window and the budget in the message.
  Instrument defect reported: the `peak VRAM` telemetry line reads 3265.56 MiB
  identically at 2/3/4 slots at this shape, so it cannot judge slot residency
  here; use the preflight window arithmetic. No independent VRAM sample (the
  sampling pass was interrupted) — recorded as a gap.
  Gate follow-up: the 4k-anchored absolute floors false-FAILED the legitimate
  3-slot 200k arm, so the gate now skips absolute floors on a shape mismatch
  (labelled `no gate applied` / `paired only`; `--force-absolute` overrides),
  with the negative controls and the four 4k traces re-validated after the
  change. Absolute floors do not transfer across prompt shapes; paired
  comparisons do.
  Charter consequence: at long context the extra slot is not worth its VRAM
  (optimum stays in the 2-3 slot range) and the remaining lever is the compute
  side, which is 92% of that wall.

Closure (2026-09-18) - VERDICT: **DISCARD the CUDA stream-topology lever** (V1 aux-stream staging).
Closing this task sets NO default: `--pp-copy-overlap` stays 0, and the slot count stays 2.

: Acceptance disposition: the wall criterion FAILED and the contract's own falsifier clause
  applies. The overlap=1 arm measured **3088.4 ms** warm 4k against the copy-backend
  baseline's **3089.0 ms** - measurably identical - while in BOTH arms the copy windows were
  already pure (median 31.4 ms = 1.00x the 31.37 ms pure-rate ceiling) with lag >= 0 on 76/76
  samples. The 2.75 -> 2.0-2.1 s target is not reachable from stream/context topology, and the
  prediction's failure IS the evidence: no DMA ever started while the compute backlog drained,
  so there was nothing for a second stream to overlap.

: What the wall is instead, per the recon's own ledger: of the 3089 ms, compute 1350 ms and
  pure DMA 1238 ms, while the 3988 ms host transfer window splits into a 1559 ms host hop
  (pinned mirror + submit) and 2429 ms of ready latency, of which only 1238 ms is engine time
  and 1191 ms is queue wait. The choke is host-side pipeline plus one-bank-deep ordering.

: KEPT, because it is a defect fix rather than a performance result: `backend_compute` was
  published ONLY inside `layer_streaming_compute`, which layer-major PP never calls (it
  executes graphs directly at :3274), so on the PRIMARY workload the aux mechanism could not
  engage at all - the flag-on arm aborted loudly and the recon silently emitted nothing. It is
  now published in `layer_major_pp` with the loud aux-stream refusal, and the recon emits
  `recon_unavailable` rather than a silent no-op. That fix is what made this falsification
  observable instead of invisible, and it stays.

: Neither remaining lever is orphaned by this closure, so the closing verdict does not narrow
  the program: host-hop removal is T-GPLR-37 (V2 mmap-registered staging measured -23.3%/-15.4%
  warm at 4k at 2/3 slots, at 17.23 GiB of pinned pages, with NO transfer to the crowded shape -
  owner decision, still open), and pipeline depth is this task's own addenda (3 slots -15.8% at
  4k; at 200k +0.5%/+0.4% for 3/4 slots, i.e. depth does NOT pay at the shape the charter
  prioritises).

: Admissibility: both arms check-telemetry OK, no deadlock, the refusal path exercised; the
  contract states no qa-smoke is required (staging-path only, cannot alter generated output)
  and no run showed an output difference.

: Disclosed and left unattributed at closure: the 3-slot arm's per-pass compute total rose
  1369 -> 1611 ms (+18%) with identical bytes and arithmetic. It does not bear on this verdict
  (which rests on the 2-slot ov0/ov1 pair) and is recorded as an open anomaly, not explained.

### T-GPLR-36: Long-context PP residual — what is the ~10 s that is neither compute nor engine time?

Status: CLOSED (2026-09-18) — the residual is NAMED and it is host bookkeeping; the closure verdict and the Planner hand-back are at the END of this block. Closing this task sets NO default (`--pp-recon` stays opt-in, default 0).
Historical status (2026-09-17): in-progress — RUN; PREDICTION FALSIFIED IN ITS DETAIL. The
per-graph-setup hypothesis is answered NO (278.6 ms per pass = 0.017 ms/launch =
3.0% of the residual), and the residual is per-launch HOST work outside the
graph window: other 4,516 ms (48.3%, unnamed yet), patch 3,402 ms (36.4%,
redundant per-launch chunk patching), slot 1,065 ms (11.4%), setup 3.0%,
graph_call 0.2%. Ledger closure exact (0.00%).
Depends on: T-GPLR-35 (the recon + slot-depth arms whose traces carry the
residual), T-GPLR-25 (async feed), T-GPLR-29 (pinned staging)
Commit: nested 8e14abd36 (the first commit carrying the `host_ledger` emitter and its
  fields - the ledger was MEASURED 2026-09-17 before it was committed, consistent with that
  day's "Nothing committed" entries); outer a58fdee (ledger runner + analyzer family); outer
  4b13ac3 (gplr36 tooling + the verdict docs); the host-RSS half was instrumented later -
  nested c5d79ad5c, outer fe9c1cf

Goal
: Name the warm 200k-token PP residual. At 3 slots the recorded pass is
  wall 146,030 ms, compute 134,592 ms (92.2%), consumer wait 36.5 ms — and
  wall - compute - pure DMA = 10,183 ms (0.65 ms/launch over 15,640 launches)
  that no recorded instrument attributes. The owner's question: is it
  per-graph setup repeated across those 15,640 launches?

Known evidence (measured from the existing traces, no GPU time)
: * Per-graph setup is ALREADY FALSIFIED as the explanation: `setup_us`
  (sched reset + `ggml_backend_sched_alloc_graph`, the exact quantity the
  question names) is median 13-18 us/launch, summing to ~0.35 s per pass
  (3.4% of the 10,183 ms residual). `graph_builds` is 0 in every pp summary:
  the graphs are prebuilt once and never rebuilt.
* `prepare_chunk` (input-snapshot memcpy + FA causal/KV-bound patching) IS
  timed already, via `total_patch_us`: 7,108.8 ms for the whole 200k run
  (~3.55 s per pass = ~35% of the residual), and only ~0.24 s for the whole 4k
  run. It is called once per launch (15,640 times per pass) even though its
  inputs are per-chunk, not per-layer.
* The sweep window (first layer compute -> last layer compute end) is
  145,010 of the 146,030 ms wall, so the embed / preload / output phases
  together are ~1.0-1.4 s: the residual lives INSIDE the layer sweep.
* The pass is bimodal: the 30 non-attention layers are flat ~3.5 ms/launch at
  every context position, while the 10 full-attention layers (every 4th) ramp
  3.9 -> 40.6 ms/launch as the KV grows. Those 10 layers are ~69% of the
  134.6 s compute: the long-context wall is attention-bound.
* The executor synchronises after every launched graph
  (`ggml_backend_sched_graph_compute_async` + `sched_synchronize`), so any host
  work between launches is exposed on the wall by construction.

Prediction (pre-registered before the ledger run)
: The ledger closes on `sweep_us`: patch + setup + compute + graph_call + slot
  + other = sweep within ~1%. `graph_call_us` — the currently UNTIMED part of
  `run_graph` (the per-launch src-pointer snapshot save/restore, O(n_nodes x
  GGML_MAX_SRC), plus call overhead) — is the largest unattributed host bucket
  and accounts for at least half of the remaining ~6.3 s. `other_us` (loop +
  telemetry) stays under ~1 s per pass.
  FALSIFIER: if `graph_call_us` and `other_us` are both under ~1 s per pass
  while closure holds, then the residual is NOT host bookkeeping at all — the
  remaining witness would be GPU idle inside the compute window's own sync
  wait, and the next lever is a deep-submit schedule, not host-work removal.

Result (measured, 2026-09-17; trace build/llama-cuda/gplr36-ledger-200k.jsonl,
checker OK, pipeline gate EXIT 0, one mutex hold)
: Warm pass: wall 141,847.9 ms (1,409.96 tok/s) vs the recorded 146,030 ms
  reference (a ~2.9% session-to-session spread); wait 36.5 ms. Ledger closes
  exactly (0.00%): sweep 140,807.7 = compute 131,523.3 + other 4,516.3 +
  patch 3,402.3 + slot 1,065.1 + setup 278.6 + graph_call 22.1 ms. Per launch
  (16,422 launches): compute 8.009, other 0.275, patch 0.207, slot 0.065,
  setup 0.017, graph_call 0.0013 ms. Per-graph setup is 3.0% of the 9,343.8 ms
  residual -> the hypothesis the owner asked to test is FALSIFIED; graph_call
  (src-pointer snapshot) 0.2% -> the pre-registered prediction was wrong about
  both graph_call (predicted largest, measured 0.022 s) and other (predicted
  < 1 s, measured 4.52 s). The falsifier did not fire: the residual IS host
  bookkeeping, but its largest piece (`other`) is still unnamed.
  Side finding: the ledger's compute bucket exceeds the summary's
  total_compute_us by 274 ms because the embed and output loops execute graphs
  that no layer stat accumulates — that 0.27 s of 'residual' is real compute.
  Consequence for the wave the owner saw in the GPU chart: the per-launch sync
  exposes 0.564 ms of host prep per launch (duty 93.4%), beating with the
  bimodal layer cost; pre-staging the next layer cannot fix it (staging is
  already hidden: wait 36.5 ms, slot block 11.4% of the residual).
  Next levers, in measured order: (1) name and cut `other` (4.5 s);
  (2) make `prepare_chunk` per-chunk instead of per-launch (3.4 s, ~97%
  redundant); (3) deep-submit chunk graphs (remove the per-launch sync) so host
  work overlaps compute — together worth up to ~9.3 s of a 141.8 s pass (-6.5%)
  at this shape, with no extra VRAM.

Scope
: Instrumentation + one measurement arm + documentation. The buckets are host
  `ggml_time_us()` reads on an existing loop; no execution order, routing,
  arithmetic, quantization, or default path changes.

Non-goals
: No optimization in this task (no removal of the redundant per-launch
  patching, no deep-submit scheduler, no default change). Those are separate
  tasks, each needing its own pre-registration and A/B.

Implementation steps
: 1. Add the ledger to `llama_context::layer_major_pp`: timers around
  `prepare_chunk`, the `run_graph` call (minus its own setup/compute), the
  per-layer slot block, and the four phase loops; accumulate `compute`,
  `setup`, `patch`, `graph_call`, `slot`, `launches`.
  2. Emit one `host_ledger` JSON line per pass, gated on `--pp-recon` (unknown
  events are ignored by the checker, so existing traces and rules are
  untouched).
  3. Build; host-validate (usage + refusal paths unchanged).
  4. Run one arm at the charter shape (200k tokens, 262k F16 KV, 3 slots,
  layer-major, ubatch 512, af=1, pinned, `--pp-repeat 2`, `--pp-recon 1`) under
  the GPU mutex with the strict checker, then read the ledger.

Acceptance criteria
: * `host_ledger` appears once per pp pass with `launches == ubatches x
  layers_run`.
* Closure holds: |patch + setup + compute + graph_call + slot + other -
  sweep| <= 1% of sweep.
* The buckets are reported per pass, and the verdict on the per-graph-setup
  hypothesis (already falsified from `setup_us`) is restated with the ledger's
  independent numbers.
* Both strict checkers (telemetry) pass, and the 200k wall is within the
  expected session-to-session spread of the recorded 146,030 ms.

Validation commands
: * `tools/run-gplr36-ledger-locked.ps1` (one mutex hold; ~10 min: one 200k
  arm, 2 passes at ~146 s each, plus load).
* `python tools/analyze-gplr36-ledger.py build/llama-cuda/gplr36-ledger-200k.jsonl`
* `llama-layer-streaming.exe --check-telemetry <trace>` (already inside the
  runner).

State/document updates
: `docs/currentstate.md` entry with the measured bucket table, the verdict on
  the per-graph-setup question, and the falsifier status; this block's status
  and Commit field.

Escalation condition
: If the ledger fails to close (buckets do not sum to the sweep within 1%),
  stop and record the discrepancy rather than publishing a decomposition.

Closure (2026-09-18) - VERDICT: **CLOSED; the ~10 s residual is NAMED and it is host
bookkeeping.** Closing sets NO default: `--pp-recon` stays opt-in (default 0), so the ledger
and its sub-buckets cost a normal run nothing.

: Answer, with the ledger closing exactly (0.00%): of a 141,847.9 ms warm pass (1,409.96 tok/s)
  over a 140,807.7 ms sweep with wait 36.5 ms, the residual is `other` 4,516.3 ms (48.3%),
  `patch` 3,402.3 ms (36.4%), `slot` 1,065.1 ms (11.4%), `setup` 278.6 ms (3.0%) and
  `graph_call` 22.1 ms (0.2%) across 16,422 launches. The per-graph-setup hypothesis the owner
  asked to test is FALSIFIED (3.0%), and both pre-registered predictions were wrong in the same
  informative direction: `graph_call` was predicted to be the largest unattributed bucket and
  measured 22.1 ms, while `other` was predicted under ~1 s and measured 4,516 ms. The
  pre-registered falsifier therefore did NOT fire - the residual IS host bookkeeping.

: The `other` bucket was subsequently NAMED by T-GPLR-39's recon: `route_us` = 1161.5 of
  1152.8 ms at u2048 (the split closes exactly by construction, so the >100% share is closure,
  not double counting). It is route TELEMETRY - the device->host probs readback through
  `ggml_backend_tensor_get` stalling the compute stream, plus a host sort of
  O(n_tok x 512 log 512) - and it scales with ubatch SIZE (237.9 ms at u512, 484.5 at u1024,
  962.1 at u2048, 959.0 at u4096), not with launch count.

: The owner's separate observation that motivated this family - host RSS growing 19 -> 30 GB
  with NO instrument - got its own instrument and its first real arm the same day: at the 200k
  shape the working set climbs 4.13 -> 25.05 GiB, of which **83.5% is the file-backed model
  mapping** and 16.5% private, while the engine's own accounted host holdings are FLAT at
  1,494.0 MiB. It is not a leak: the first pass memcpy's every package member out of the model
  mapping, so it touches essentially the whole 19.92 GiB GGUF once. Reaching a 30 GB ceiling
  needs more ANONYMOUS demand (deeper KV spill, more slots, a pinned pool), not more mapping.
  Cross-instrument check: an independent external monitor measured 23,458 MiB max working set on
  a 200k arm, and this instrument measured 23,457.1 MiB at pass 1's end.

: Acceptance criteria, all met: `host_ledger` appears once per pp pass (2 lines in the trace,
  one per pass, `launches == ubatches x layers_run`); closure holds exactly (0.00%); the verdict
  on the per-graph-setup hypothesis is restated with the ledger's own numbers; both strict
  checkers pass; and the 200k wall (141,847.9 ms) sits inside the session-to-session spread of
  the recorded 146,030 ms reference (-2.9%).

: Side finding, recorded at closure: the ledger's `compute` bucket exceeds the summary's
  `total_compute_us` by 274 ms because the embed and output loops execute graphs that no layer
  stat accumulates - so ~0.27 s of what looked like "residual" is real compute.

: HANDED BACK TO THE PLANNER, not drafted here (drafting tasks is Planner work): this task's
  own measured lever list is still un-drafted. (2) Make `prepare_chunk` per-chunk instead of
  per-launch - 3,402.3 ms per pass at 200k, ~97% redundant, because its inputs are per-chunk but
  it is called once per launch. (3) Deep-submit chunk graphs to remove the per-launch sync,
  which is what exposes 0.564 ms/launch of host prep on the wall by construction (duty 93.4%).
  Together they are worth up to ~9.3 s of a 141.8 s pass (-6.5%) at this shape with NO extra
  VRAM - the largest measured PP lever this program currently has open, and squarely inside the
  charter's first priority.

### T-GPLR-37: direct mmap-sourced staging — remove the CPU hop from bank staging

Status: in-progress (2026-09-17) — PROBE FEASIBLE; V1 (per-bank churn)
**FALSIFIED AT 4k** (warm regression -15.4% at 2 slots, -23.8% at 3); V2
(churn-free, register once at init — see addendum 2 below) **CONFIRMS the churn
mechanism and WINS at 4k at both depths** (warm -23.3% at 2 slots, -15.4% at 3,
paired against adjacent pinned controls), at a measured cost of 17.23 GiB of
pinned file pages. Not promoted to default; owner decision.

V1 detail (kept as the falsified half of the family) The hop was removed from the accounting (host submit 1587.8 ->
1222.3 ms/pass at 2 slots; `slot` bucket 1315.6 -> 1153.3) but per-launch host
windows grew more (`setup` 11.2 -> 227.8 ms at 2 slots; `compute` 1329.5 ->
1845.1 / 1581.4 -> 2441.7 ms), with bank DMA windows unchanged (31.48 ms pure)
and no starvation (3-slot wait 36.5 ms). Indicated mechanism: per-bank
register/unregister churn over ~111k pages x 80 registrations (34.46 GiB
cumulative). Cold passes look 17-23% better but are CONFOUNDED by arm order
(file cache warmed by the earlier arms) and are not claimed as a result. Next
and last test of the family: register ONCE at init (whole mapping / all exec
spans, ~17 GiB pinned, T-GPLR-31 proved 18 GiB grantable) so there is no churn;
falsifier = the `setup` inflation persists without churn, which closes the
family.
Depends on: T-GPLR-29/30/31 (pinned staging + pool), T-GPLR-33 (shared
`stage_members` submission helper), T-GPLR-35 (the recon that measured the hop),
T-GPLR-36 (the ledger that showed no host bucket contains the hop, because it
runs on the async-feed worker)
Commit: nested 7bcbd7a75 (cudaHostRegister host-range API) + 8e14abd36
(register-once staging + ledger sub-buckets) + 1e72d62c7 (runner wiring); outer
4b13ac3 (evidence/tooling, gitlink bump)



Goal
: Remove the CPU hop from bank staging without giving up pinned transfer speed.
  Today every member is `memcpy`'d into a pinned staging buffer and the DMA reads
  that copy; `--copy-source direct` skips the copy but DMAs from pageable memory
  at 9.59 GB/s. The third option does not exist yet: page-lock the model's own
  mapped source pages and DMA straight from the mapping.

Probe evidence (measured, tools/gplr37-mmapreg-probe.cu, one mutex hold, 452.5
MiB window of the real MoE GGUF at offset 1 GiB; every arm verified byte-for-byte
against the file window)
: * A pageable mmap DMA (= `--copy-source direct`): 49.47 ms, 9.59 GB/s
  * B registered mmap DMA (`cudaHostRegister` on the FILE-BACKED mapping):
    **32.86 ms, 14.44 GB/s = the pinned ceiling**, registration cost 0.16 ms
  * C the current path: CPU hop 20.22 ms + DMA 32.86 ms = 53.08 ms per bank
  So the hop is 20.22 ms/bank (~809 ms per 40-bank pass) and a registered source
  both deletes it and lowers the transfer below the pageable path.

Prediction (pre-registered before the A/B)
: Mechanism: `copy_complete.host_submit_us` per bank collapses from ~39 ms to
  ~0.2 ms, one `mmap_register` event fires per bank fill (~452 MB each), and the
  bank DMA windows stay pure (~31.4 ms). Wall at **3 slots: no change beyond
  noise (+-3%)** — the recorded 3-slot account already closes as wall 2843 ms ~
  compute 1611 + pure DMA 1255 with consumer wait 225 ms, i.e. the hop is hidden
  behind compute at that depth. Wall at 2 slots: staging (hop+DMA) exceeded the
  per-bank compute there (consumer wait 734 ms), so the candidate may win up to
  ~5%.
  FALSIFIER: a >= 0.3 s (>=10%) warm-wall drop at 3 slots means the hop was on
  the critical path after all and the "already hidden" reading is wrong; a >= 5%
  REGRESSION at either depth means register/unregister churn per bank costs more
  than the hop it removes.

Scope
: An experimental copy-source mode and its measurement. `--copy-source
  mmap-registered` registers the page-aligned span covering a package's member
  sources, caches it on the slot, unregisters it before that slot stages a
  different package (safe: a slot is only re-staged after the previous
  generation's consumer completed), and unregisters at slot teardown. The mode
  allocates no staging buffers at all and is mutually exclusive with `--pp-pool`
  (which requires the pinned path and refuses loudly).

Non-goals
: No default change; no removal of the pinned path; no change to routing,
  arithmetic, quantization or KV handling; no attempt to keep whole-run
  registrations alive (that would pin the full ~18 GB bank set permanently —
  measured as grantable by T-GPLR-31 but a much larger RAM trade).

Acceptance criteria
: * Every arm's strict checker passes and the mmap-registered traces carry one
  `mmap_register` event per bank fill with the expected bytes.
* The hop collapses in the accounting (per-bank host submit window ~0.2 ms vs
  ~39 ms) with bank DMA windows unchanged.
* The warm-wall verdict at 2 and 3 slots is recorded against the pre-registered
  prediction and falsifier, with the VRAM/RAM price stated (no staging buffers
  in this mode; pinned file pages held instead, bounded by n_slots x ~452 MB).

Validation commands
: * `tools/run-gplr37-probe-locked.ps1` (stage-1 feasibility, ~10 s of GPU work)
* `tools/run-gplr37.ps1` (four 4k arms, one mutex hold, ~12 min)
* `python tools/analyze-gplr37.py build/llama-cuda/gplr37-{pin,reg}-s{2,3}.jsonl`

State/document updates
: `docs/currentstate.md` entry with the probe table, the A/B verdict against the
  prediction, and the RAM/RAM trade; this block's status and Commit field.

Escalation condition
: If registration fails on the real model (file-backed mapping refused, or the
  span spans far-apart members and pinning explodes the footprint), record the
  failure and stop — the pinned path stays the answer and the mode is withdrawn.

#### T-GPLR-37 addendum 2 RESULT (2026-09-17): churn-free variant — mechanism CONFOUNDS CONFIRMED, large warm win, prediction wrong

Measured verdict (four adjacent arms, one mutex hold, 4k anchor, ubatch 512,
layer-major, kv q8, af=1, `--pp-repeat 2`; all four strict checkers OK; traces
`build/llama-cuda/gplr37b-*.jsonl`; analyzer `tools/analyze-gplr37.py`)
: * Warm pass: **pin-s2 3490.2 ms / 1173.6 tok/s -> ron-s2 2676.9 ms / 1530.1
  tok/s (-813.3 ms, -23.3%, +30.4% tok/s)**; **pin-s3 2706.0 ms / 1513.7 ->
  ron-s3 2290.6 ms / 1788.2 tok/s (-415.5 ms, -15.4%, +18.1%)**.
  * FALSIFIER DID NOT FIRE: `setup` returned to the clean band (13.0 -> 8.2 ms at
  2 slots, 12.1 -> 7.9 at 3, vs the churn arms' 227.8 / 134.7), `mmap_register`
  fired 40x with `mode="init-once"` and **40/40 before the first
  `layer_compute`** (churn: 80 events, 34.46 GiB, 2/80 before), and one
  `mmap_register_init` reports 18,500,517,888 B (17.23 GiB) pinned in 1.6 ms
  pre-timed. Churn was the whole cause of the V1 regression.
  * Mechanism: PP hop median 45.24 -> 0.06 ms/bank (s2) and 23.43 -> 0.11 (s3);
  `slot` bucket 1620.7 -> 991.2 ms (s2) and 582.3 -> 69.4 (s3); bank DMA windows
  pure (31.38-31.48 ms = 1.00x); compute ~unchanged (1363.6 -> 1297.2 at s2); TG
  unaffected (76.6 vs 76.75 tok/s). Greedy 96-token `--tg-text` byte-identical
  between pinned and mmap-registered at both depths (identity sanity check only,
  NOT the semantic-QA gate).
  * **Prediction WRONG:** I pre-registered "within +-3%, gain <=5% at 2 slots";
  measured -23.3% / -15.4%. The pre-registered reading that the hop is "already
  hidden behind compute at 3 slots" was false at this shape.
  * Headline vs history: ron-s3 2290.6 ms is the fastest warm 4k layer-major pass
  recorded (previous best 2706 / 2811.9 / 2843.2), and **ron-s2 2676.9 ms matches
  pin-s3 2706.0 ms with one slot fewer (-498 MiB VRAM)** — depth-3 pinning is not
  needed for this shape under this mode.
  * Price: 17.23 GiB of pinned file pages for the run (vs 0.9-1.4 GiB of pinned
  staging buffers for the pinned path); no staging buffers allocated. The side RAM
  monitor's window missed the arms (started before the ~26 min mutex wait, stopped
  after ~74 s) — its CSV is NOT cited as evidence; the 17.23 GiB is the runner's
  own pre-timed measurement.
  * Cold-pass numbers remain CONFOUNDED in the candidates' favour (they page in
  the whole model before the timed pass) and are not claimed.
  * Gate change: `tools/check-gplr35-pipeline.py` paired mode was depth-only and
  its absolute wait/wall fraction false-FAILed the faster arm; it now accepts a
  same-depth pair with a different copy source, anchors wait (ms, <= control x
  1.25 + 50) and exposure (<= control + 350) to the control, keeps the pure-DMA
  window check, and FAILs an identical-configuration pair. Re-validated on six
  controls (depth pair PASS 1.188x; reversed FAIL 0.842x; churn arms FAIL 0.846x;
  ron pairs PASS 1.304x / 1.181x; identical FAIL).
  * Not a default change and not a fitness claim: the owner decides whether a
  17.23 GiB pinned-RAM trade for -23% warm 4k PP is acceptable, and whether the
  next step is the crowded-VRAM (262k KV) shape where the charter's decisive case
  lives.

#### T-GPLR-37 addendum 2 (2026-09-17): churn-free variant — register every span ONCE at init

Motivation (from the falsified V1)
: V1's regression had exactly one measured candidate cause: 80 register/unregister
  cycles over ~111k pages each (34.46 GiB of churn per run) landing inside the
  per-launch host windows (`setup` 11.2 -> 227.8 ms at 2 slots; `compute` +0.5 to
  +0.9 s). `mmap-registered` therefore becomes the churn-FREE mode: one
  `register_all_pkg_sources` call runs after the prebuild and BEFORE the timed PP
  (pre-timed, so its cost is setup and cannot pollute the pass wall), registering
  the page-aligned source span of every executed layer once; staging then reads
  the cached per-layer span and never touches page tables during the pass. The
  falsified per-bank churn stays reproducible as `--copy-source
  mmap-registered-churn` so the V1 evidence remains re-runnable.

Pre-registered prediction (written before the run)
: * Mechanism: 40 `mmap_register` events with `mode="init-once"`, ALL before the
  first `layer_compute` of pass 1, plus one `mmap_register_init` carrying the
  total bytes and the pre-timed cost; no registration inside a pass.
  * Ledger: `setup` returns to the pinned control's band (~11-12 ms) and
  `compute` returns to the control's band (1329 ms at 2 slots / 1581 at 3), with
  bank DMA windows still pure (~31.4 ms).
  * Wall: within +-3% of the pinned control at BOTH depths; a gain is expected
  only at 2 slots (<= 5%), where the hop was partly exposed rather than hidden.
  * FALSIFIER: if `setup` still reads 100-230 ms without churn, the cost is
  inherent to file-backed registered pages rather than to churn, and the
  mmap-registered family closes. A >= 5% warm regression at either depth
  rejects the variant regardless of mechanism.
  * Price to report: the pre-timed registration phase and the pinned footprint
  (expected ~17 GiB of file pages held for the run — the RAM cost of removing
  churn, and the reason this mode is opt-in).

Known confound, stated up front
: The candidate arms pin (and therefore page in) the whole model before their
  timed pass, so any COLD-pass advantage is confounded in the candidate's favour
  (V1 measured cold 17-23% better and that was not claimed either). The WARM
  comparison in adjacent pin/ron pairs is the verdict target.

Validation commands (addendum 2)
: * `tools/run-gplr37b.ps1` (four adjacent arms + strict checkers + analyzer +
  pipeline gate, one mutex hold, ~15-20 min)
* `python tools/analyze-gplr37.py build/llama-cuda/gplr37b-{pin,ron}-s{2,3}.jsonl`
* `python tools/check-gplr35-pipeline.py <ron trace> --baseline <pin trace>`

#### T-GPLR-37 addendum 3 RESULT (2026-09-17): crowded shape measured — prediction CONFIRMED, win does not transfer

Measured verdict (four adjacent arms, one mutex hold, all strict checkers OK;
traces `build/llama-cuda/gplr37c-*.jsonl`; analyzer `tools/analyze-gplr37.py`;
pipeline gate s2/s3 both PASS paired)
: * Warm pass: **pin-s2 142,165.5 ms / 1406.8 tok/s -> ron-s2 141,853.7 /
  1409.9 (-311.8 ms, -0.22%)**; **pin-s3 141,666.6 / 1411.8 -> ron-s3 141,134.4 /
  1417.1 (-532.2 ms, -0.38%)**. Both within the pre-registered <= ~1% noise band;
  neither falsifier branch fired.
  * Mechanism confirmed where it can be seen: per-bank hop 44.01 -> 0.08 ms (s2)
  and 44.47 -> 0.13 ms (s3); `slot` bucket 1759.0 -> 1280.6 (s2) and
  **637.3 -> 70.3 ms (s3)**; bank DMA windows pure (31.5-31.6 ms = 1.00x);
  compute/patch/other/setup unchanged. The hop is deleted; the wall is 92.5%
  compute and does not care.
  * RAM (measured by the fixed monitor, whole-run window): min free system RAM
  **19,447 MiB** of 61.7 GiB with 17.23 GiB pinned + ~5 GiB spilled KV; max
  process WS 23,458 MiB. No pressure event; no eviction/trim observed.
  * Consequence: the churn-free mmap-registered mode is a SHORT/AMORTIZED-CONTEXT
  lever (staging on the critical path), not a long-context one; at 200k it costs
  17.23 GiB pinned for ~0.3% wall. Default posture unchanged (pinned + depth for
  long context; mmap-registered opt-in); a shape-aware mode switch is a possible
  later owner decision, not made here.

#### T-GPLR-37 addendum 3 (2026-09-17): the crowded-VRAM shape — 200k tokens on 262k F16 KV, pinned vs mmap-registered

Motivation
: The charter's decisive case is PP under VRAM/RAM crowding. Addendum 2 measured
  the churn-free win at the 4k shape (-23.3% warm at 2 slots, -15.4% at 3) for
  17.23 GiB of pinned file pages. This addendum tests whether that win survives
  where staging is amortized differently: at 200k tokens each bank serves ~391
  chunks, the 4k-measured depth effect had already collapsed to <=1% (T-GPLR-35
  addendum 2), and the host-spilled KV (5.0 GiB f16, 10 full-attention layers at
  262144 ctx) plus the 17.23 GiB pinning meet on the host for the first time
  (61.7 GiB total system RAM).

Shape (identical to the recorded 200k references, unchanged)
: `--pp-tokens 262144`, `--ubatch 512`, `--kv-cache f16`, `--act-slab-dtype
  bf16`, layer-major, af=1, `--pp-repeat 2`, `--tg-tokens 0`; control = pinned,
  candidate = mmap-registered (churn-free), at 2 and 3 slots; four adjacent arms
  in one mutex hold; ~10 min/pass x 2 passes x 4 arms ~ 80 min budget.

Pre-registered prediction (written before the run)
: * Mechanism: at this shape the wall is compute-dominated (91%+ measured;
  T-GPLR-35 addendum 2: slots 2->3->4 moved the wall +0.8% total, wait 36.5 ms).
  The hop that ron deletes is ~0.9-1.6 s per pass (T-GPLR-37 V1 ledger), i.e. ~1%
  of the ~147 s wall, and af=1 already hides it behind compute. Expected: warm
  walls within noise (<= ~1%) at BOTH depths; ron's advantage concentrated in
  the `slot` ledger bucket and the per-bank host submit, not the wall.
  * RAM: ron arms pin 17.23 GiB of file pages ON TOP of the ~5.0 GiB spilled KV
  residency; pinned arms hold only n_slots x 452 MiB of staging buffers. Both
  fit 61.7 GiB; the monitor's sys_free column records the actual floor.
  * FALSIFIER: if ron still cuts the 200k warm wall by > 3% at either depth, the
  "staging is already hidden at long context" reading is wrong and the mode is
  a long-context win too. A warm-wall REGRESSION > 3% would instead indicate the
  pinning interacts badly with the spilled-KV working set (page-cache pressure).

Validation commands (addendum 3)
: * `tools/run-gplr37c.ps1` (four adjacent 200k arms + strict checkers + analyzer
  + paired gates, one mutex hold, ~80 min budget)
* `python tools/analyze-gplr37.py build/llama-cuda/gplr37c-{pin,ron}-s{2,3}.jsonl`
* RAM monitor `tools/monitor-mem-gplr36.ps1` (fixed: process-anchored window +
  sys_free column) writes `build/llama-cuda/gplr37c-mem-monitor.csv`

### T-GPLR-38: ubatch sweep on the 4k anchor — does fewer, fatter launches buy wall time under the current best mode?

Status: CLOSED (2026-09-18) — u1024 is the measured 4k sweet spot and u2048 is a regression under the current host loop; the closure verdict, the output-identity observation and the default decision are at the END of this block. Closing sets NO default (`--ubatch` stays 512).
Historical status (2026-09-17): run — prediction INVERTED at u2048; sweet spot u1024
(−4.0% warm wall); the falsifier fired informatively: the `other` ledger bucket
scales with UBITSIZE, not launch count. Verdict recorded in
docs/currentstate.md; follow-up lever = make the other-bucket work
chunk-independent (a ~0.7 s/pass prize at u2048), drafted as T-GPLR-39.
Depends on: T-GPLR-37 addendum 2 (ron = the current best 4k mode; the falsified
churn arm stays reproducible), T-GPLR-36 (the host ledger that named the
per-launch overhead), T-GPLR-35 (the pipeline gate and recon)
Commit: nested 8e14abd36 (ledger sub-buckets); outer 4b13ac3 (sweep evidence +
tooling + gitlink bump)

Goal
: The 200k ledger attributed the long-context residual to per-launch host cost
  (~0.564 ms/launch across patch+other+slot+setup, x15,640 launches) and the 4k
  warm ron-s3 pass has 336 launches. Fewer, fatter ubatches cut that overhead
  proportionally AND let each bank serve fewer, longer consumers — but coarsen
  pipelining granularity (a 1-chunk bank at ubatch 4096 gives the af worker one
  ~33 ms DMA to hide behind a ~46 ms compute instead of eight). Sweep
  `--ubatch 512/1024/2048/4096` at 3 slots, ron mode, 4k anchor, and measure
  where the wall, consumer wait, and peak VRAM actually move.

Scope
: Measurement only. No default change (`--ubatch` already accepts any value > 0;
  `--mtp` requires `pp_tokens %% ubatch == 0` and stays out of these arms). Each
  arm is `--pp-repeat 2` (cold+warm); the warm pass is the verdict target.

Memory expectation (derived from code + measured traces, stated up front)
: ubatch-dependent VRAM is small: `buf_act` = 24/48/96/192 MiB at
  512/1024/2048/4096, plus the transient ggml compute workspace (unmeasured
  term, scales with ubatch). Everything dominant is ubatch-independent: 3 slots
  x 498 MiB, the 32 MiB bf16 act slabs (sized by pp_tokens, not ubatch), KV.
  Baseline peak VRAM 3265.56 MiB; expected sweep range roughly 3.3-4.5 GiB —
  the capacity preflight refuses loudly with exact numbers if a budget cannot
  hold it.

Pre-registered prediction (written before the run)
: * Wall: improves monotonically with ubatch but SUB-LINEARLY — host overhead
  removed is only ~0.19 -> ~0.02 s of a 2.29 s pass, while larger ubatches risk
  worse pipelining (at ubatch 4096 there is ONE chunk per bank; if its DMA is
  not hidden, consumer wait grows). Predicted: warm wall 2290 -> ~2050-2200 ms
  at ubatch 2048; ubatch 4096 flat or worse than 2048 (coarse granularity).
  * Consumer wait: rises with ubatch (fewer, longer chunks to hide staging
  behind), most visibly at 4096.
  * Peak VRAM: rises modestly and monotonically (buf_act + workspace), never
  near a refusal on this card.
  * FALSIFIER: if the wall IMPROVES at 4096 vs 2048, the granularity concern is
  wrong and bigger is strictly better here; if the wall WORSENS at 1024+, the
  per-launch overhead model is wrong (something else grows faster than the
  overhead shrinks).

Validation commands
: * `tools/run-gplr38-ubatch.ps1` (four adjacent ron-s3 arms + 512 control re-run,
  one mutex hold, ~6 min)
* `python tools/analyze-gplr37.py build/llama-cuda/gplr38-u{512,1024,2048,4096}.jsonl`
* paired gates vs the fresh u512 control

State/document updates
: `docs/currentstate.md` entry with the wall/wait/VRAM table against the
  pre-registered prediction; this block's status and Commit field.

Escalation condition
: If an arm refuses on VRAM or the checker rejects a trace, record the shape and
  stop the sweep at the last clean point.

Closure (2026-09-18) - VERDICT: **CLOSED; u1024 is the measured 4k sweet spot, and u2048 is a
regression under the current host loop.** Closing sets NO default: `--ubatch` stays **512**
(the runner's default, main.cpp:2534; help text line 50) for the three reasons below.

: Measured (four adjacent arms, one mutex hold, all four strict checkers OK): warm wall
  2317.5 ms (u512) -> **2224.3 ms (u1024, -4.0%)** -> 2510.6 ms (u2048, +8.3%) -> 2435.2 ms
  (u4096); tok/s 1767.4 -> 1841.4 -> 1631.5 -> 1682.0.

: The pre-registered prediction was WRONG in the informative direction. Consumer wait stayed
  FLAT (67.7-68.0 ms at every ubatch, so the predicted granularity penalty never materialised),
  peak VRAM was identical (3265.56 MiB - the known defect reading; the sweep probes the shared
  3-slot window, so the buf_act deltas of 24 -> 192 MiB hide inside the slack), and the ledger
  `compute` bucket fell monotonically (1862.6 -> 1525.1 -> 1375.0 -> 1282.7 ms) exactly as
  predicted - but `other` EXPLODED (237.9 -> 484.5 -> 962.1 -> 959.0 ms), swallowing the compute
  saving and then some. The falsifier's second clause fired: the per-launch overhead model was
  **launch-COUNT-based** when the residue is **ubatch-SIZE-based**. u1024 is the only arm that
  keeps the best of both (-334 ms compute, +247 ms other, net -93 ms wall).

: WHY THE DEFAULT IS NOT RAISED TO 1024 - a decision, not an oversight: (i) the win is a net
  -93 ms at ONE shape (4k anchor, ron mode, 3 slots) and the gate cannot model a moving ubatch
  axis - the paired runs correctly REFUSE same-config pairs, so the printed ratios
  (1.042x / 0.923x / 0.952x) are recorded but ungated; (ii) the MECHANISM says bigger is not
  generically better, because the bucket that grows is a function of ubatch SIZE, so raising the
  default before the host loop is fixed trades a 4k gain for a long-context loss; (iii) the only
  long-context ubatch arm (144k, u1024 vs u2048) is INVALID - both arms crashed in pass 2 and the
  surviving pass-1 numbers are COLD and arm-order confounded (the crash was later bisected: the
  tail chunk is exonerated and q8 KV + pass 2 is the trigger). This is the same failure mode
  that produced the K=4 row-cache default, so it is not repeated here. The lever that would make
  u2048+ safe is T-GPLR-39's fix class - named and measured, still an open owner decision.

: Output identity, recorded rather than dropped: greedy TG text is byte-identical at u1024 and
  u2048, but **u4096 DIFFERS (262 vs 305 bytes - the answer is cut short)**. That is an identity
  sanity check, NOT the semantic-QA gate. It is unexplained, and no arm at or above u4096 should
  be treated as output-equivalent until it is investigated.

### T-GPLR-39 (draft): make the `other` ledger bucket chunk-independent — the residue that grows with ubatch size

Status: in-progress (2026-09-17) — RECON RUN: attribution complete.
**`route_us` = 1161.5 of 1152.8 ms `other` (100.8%, closure exact) at u2048**;
tel 1.0 ms; loop −9.7 ms. The prediction ("route dominates, ubatch-scaled") is
CONFIRMED; the falsifier did not fire. The growth is route TELEMETRY (device
readback + host sort inside `tel_route`, 80 events/pass), not the launch loop.
Remaining: owner decision on the fix class — (a) sample route heads / F16
readback (diagnostic cost cut ~32-64x), (b) GPU-side sort / drop hash, or (c)
accept the cost as the price of the exact-routing evidence. Note: u1024 stays
the measured sweet spot; the u2048 "regression" is largely the telemetry
measuring itself.
Depends on: T-GPLR-38 (the sweep that measured `other` 237.9 -> 959.0 ms as
ubatch 512 -> 4096 at fixed launch count), T-GPLR-36 (the ledger that names the
bucket)
Commit: (fill on completion)

Goal
: T-GPLR-38 measured that the host loop's `other` bucket (loop body + sampled
telemetry + route lookup per the T-GPLR-36 attribution) GROWS with ubatch size
while the launch count FALLS: 237.9 ms (u512) -> 484.5 (u1024) -> 962.1 (u2048)
-> 959.0 (u4096) per warm 4k pass. If that work is made chunk-size-independent
(precompute once per chunk instead of per (layer, chunk) launch, or hoist out of
the layer loop entirely where inputs are per-chunk), the u2048 wall could drop
by up to ~0.7 s toward the ~1.8 s compute floor.

Pre-registered prediction (recon stage, written before the run)
: * Instrument: the ledger emits `route_us` (tel_route: device->host probs
  readback + host sort/hash, fires 2 chunks x 40 layers per pass), `tel_us`
  (remaining sampled telemetry writes, per-launch), and `loop_us` (residue).
  Closure: other == route + tel + loop.
  * Prediction: `route_us` dominates the growth and scales with ubatch — each
  call reads n_expert x n_tok f32 (512 x 2048 = 4 MiB at u2048 vs 1 MiB at
  u512) and stable-sorts 512 experts x n_tok rows, 80 calls per pass; predicted
  route_us ~ 500-800 ms at u2048 (vs < 100 ms at u512). `tel_us` and `loop_us`
  predicted flat (per-launch, launch count fell).
  * If confirmed: the fix is sampling/decimation of route telemetry (stride is
  already 10; the cost is the READBACK + SORT, so reduce n_tok per route event
  or drop the host sort), predicted warm u2048 wall 2510.6 -> ~1850-1950 ms.
  * FALSIFIER: if `loop_us` (not route) carries the growth, the residue is the
  per-launch loop body itself and the family closes; if route is large but FLAT
  across ubatch, the readback is latency- not size-bound and a different fix
  applies.

Non-goals
: No change to the staging path, no new copy-source mode, no ubatch default
  change (1024 stays opt-in until this lands), no TG-path changes.

Recon result (measured, 2026-09-17)
: * Arm: `tools/run-gplr39-recon.ps1`, u2048 / 3 slots / ron / 4k anchor /
  `--pp-repeat 2`; checker OK; trace `build/llama-cuda/gplr39-u2048-recon.jsonl`.

#### T-GPLR-39 implementation arm (2026-09-18 — COMMITTED, vendored tree c5d79ad5c)

Fix class implemented: (d) exact bounded host selection, alongside the
(a)/(b)/(c) options above. `tel_route` and `read_router_ids` now derive the
top-k ids with `llama_ls_route_topk_row` (O(n_expert*k) bounded insertion and
one reused ids buffer) instead of rebuilding an index array and
`std::stable_sort`ing all n_expert experts per token; the route event's union
moves from `unordered_set` to a byte map. The derivation is unchanged by
construction, so no emitted evidence changes meaning: route ids, union, hash
and `sampled_ids` stay the same values.

Second call site is NOT telemetry: `read_router_ids` produces the row-set
ROUTING decision that the caller memcpys into `sel_in`, so the replacement had
to preserve the id ORDER (ties by ascending expert index). Equivalence
argument: its only live output is `ids_per_token` (`probs_buf` is written but
never read at its single call site), the ids are bit-identical, and the staged
row set derives from those same ids. End-to-end confirmation is unmeasured.

Host validation (measured, no GPU): `tests/test-layer-streaming-route-topk.cpp`
34/34 PASS — 28 row-level cases (k=1..n_expert, heavy ties, all-equal, all-zero
underflow, signed zeros, n_expert 1..5 exhaustive, clamp, degenerate inputs)
plus 6 row-set cases (ubatch rows, duplicate rows, ties, W=1, k extremes) that
also compare the caller's `sel_in` packing, each against the pre-change
derivation kept verbatim as the oracle.

Unmeasured / open: (1) wall-clock effect — the per-call device->host probs
readback is untouched and is the larger term, so expect a modest trim rather
than the 32-64x class (a) targets; (2) end-to-end row-set equality (arm:
`tools/run-gplr13-rowset-probe.ps1`, GPU mutex); (3) proposed GPU acceptance
for this arm: re-run the u2048 recon arm and require all 80 route records
(hash/union/sampled_ids) to match `build/llama-cuda/gplr39-u2048-recon.jsonl`,
measured reproducible (80/80 keys identical across pass 1 and pass 2).

#### T-GPLR-36 host-memory instrument (2026-09-18 — COMMITTED, vendored tree c5d79ad5c)

The open question from the 200k ledger run — the owner's observed host-RSS climb
(~19 -> 30 GB across a run) had NO instrument, every byte figure in the trace
being VRAM-side, and the external `tools/monitor-mem-gplr36.ps1` can only report
a 5 s time series, never attribute — now has an in-tree per-pass instrument.

What it emits: a `host_mem` record at each PP pass boundary, twice per pass
(`pass_begin` / `pass_end`), recon-gated exactly like the T-GPLR-36 ledger.
Fields: process working set, private (anonymous) resident share and the derived
file-backed/shared share (with `ws_private_known` flagging whether that split is
available at all — a silent zero would make the file-backed share read as the
whole working set, so the record says which case it is in), private commit, OS
lifetime peak and this run's peak,
deltas since the previous sample and since the run's first sample, system
free/total RAM and commit, plus `eng_bytes` — the engine's OWN accounted host
holdings at the same instant (per-slot pinned staging + persistent pinned pool +
registered mmap spans), read from the live allocations, never re-derived.

Why that shape: `ws - ws_private` separates the runner's own anonymous
allocations from the mapped model's file-backed pages, and that is the split that
attributes the climb; `eng_bytes` then says whether the anonymous delta is the
residency machinery or something else. The baseline deliberately survives
`reset_stats()` (the runner resets stats at every pass boundary; cross-pass
growth is the point). Sampler `llama_ls_host_mem_sample()` lives in its own TU
(`src/llama-layer-streaming-host-mem.cpp`) because `<windows.h>` defines
min/max/small as macros while the main source uses std::min/max and locals named
`small`; a platform without the counters reports `ok == false`, not a zero.

Validation (measured, host-only, no GPU): `tests/test-layer-streaming-host-mem.cpp`,
CTest #50 PASS — field invariants plus the reactivity property (a touched 256 MiB
allocation moved the working set by exactly +256 MiB and commit by +257 MiB),
which is what a constant or wrong-process query would fail; CTest #49
(`test-layer-streaming-route-topk`) still PASSes and llama.dll builds clean with
the new TU. Checker safety is established, not assumed: unknown non-empty event
types are ignored by both validators (`check_telemetry` has no unknown-event
error path; `validate_telemetry` errors only on an EMPTY event field), the same
property the `host_ledger` record and the T-GPLR-39 sub-buckets already relied on.

Superseded on the same day: the first real arm has since run, so the 19 -> 30 GB
climb is still UNEXPLAINED — it is now measurable and attributable per pass,
which is not the same as measured. Consumption is a `--pp-recon 1` arm on any
crowded shape (the 200k/262k F16 reference is the natural pair to the recorded
ledger run).

#### T-GPLR-36 host-memory instrument — FIRST REAL ARM (measured, 2026-09-18)

Arm `tools/run-gplr36-hostmem.ps1` (byte-identical flags to the ledger arm: 200k
anchor, 262k F16 KV config, layer-major, u512, af=1, pinned, bf16 slabs, 3 slots,
--pp-repeat 2, --pp-recon 1) under the mutex via `tools/run-gplr36-hostmem-locked.ps1`.
Runner exit 0, strict --check-telemetry OK, pipeline gate PASS (shape-mismatch
skip: no absolute floors applied). GPU otherwise IDLE at arm start (0 MiB of
24,463) — the other project under the mutex released just before — so "crowded"
here is the long-context KV shape, not external VRAM pressure. Trace
`build/llama-cuda/gplr36-hostmem-200k.jsonl`, wrapper log
`build/llama-cuda/gplr36-hostmem-wrapper.log`, analyzer
`tools/analyze-gplr36-hostmem.py`.

Measured (MiB):

| pass | label | ws | private | file | commit | d_ws | eng |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | begin | 4,126.9 | 2,129.3 | 1,997.6 | 12,989.5 | 0.0 | 1,494.0 |
| 1 | end | 23,457.1 | 3,974.6 | 19,482.5 | 15,752.7 | +19,330.2 | 1,494.0 |
| 2 | begin | 21,867.5 | 2,385.0 | 19,482.5 | 13,376.9 | (shrank 1,589.6; clamped to 0) | 1,494.0 |
| 2 | end | 25,054.0 | 5,572.2 | 19,481.8 | 16,929.1 | +3,186.5 | 1,494.0 |

**The climb is the model's own mapping, not the residency machinery.** Working
set +20,927.1 MiB over the run = 83.5% file-backed/shared (+17,484.2) + 16.5%
private (+3,442.9), while the engine's accounted host holdings are FLAT at
1,494.0 MiB (d_eng = 0 at all four samples). File-backed pages land almost
entirely in PASS 1 (+17,484.9) and are flat in pass 2 (+0.0 / -0.7): the pinned
staging path memcpy's every package member out of the mapping, so the first pass
touches essentially the whole 19.92 GiB GGUF. Those are reclaimable model pages,
not a leak.

**The residue is the anonymous growth and it is NOT ours:** +3,442.9 MiB — pass 1
+1,845.3, a 1,589.6 MiB SHRINK across the pass boundary, then pass 2 +3,186.5.
That sawtooth is what a buffer freed and re-allocated at the boundary looks like,
and its magnitude is in the same band as the runner's own byte_account figure for
the HOST-SPILLED KV at this shape (3,910 MiB), which the engine deliberately does
not account (KV belongs to the memory module). Leading candidate: the KV host
spill re-touching across the boundary. Not measured yet — the residue is now
quantified and its candidate named.

**Cross-instrument agreement:** the independent external monitor
(`tools/monitor-mem-gplr36.ps1`) measured a 200k arm's max process working set at
23,458 MiB (T-GPLR-37c); this instrument measured 23,457.1 MiB at pass 1's end.
Two instruments, different processes, ~1 MiB apart.

**The 19 -> 30 GB observation:** this shape climbs 4.13 -> 25.05 GiB, so the
mechanism reproduces (a large, overwhelmingly file-backed climb) while the 30 GB
ceiling does not appear here, and the recorded external maximum (23.5 GiB)
agrees with this run rather than with 30. Reaching ~30 GB needs more ANONYMOUS
demand (deeper KV spill, more slots, a pinned pool), not more model mapping, and
the instrument can now price those knobs separately.

**Instrument limitations this arm demonstrated:** (1) `d_ws` clamps negative
deltas, so the 1,589.6 MiB boundary shrink is visible only via `g_ws` — deliberate
(an unsigned wrap would read as 16 EiB), and a signed field is the improvement if
shrink magnitudes matter. (2) Two samples per pass: a climb INSIDE a pass is
bracketed, not resolved. (3) `eng_bytes` covers only this module's buffers, so
"unaccounted" means "outside the residency machinery", not "outside the process".

**Unrelated anomaly in the same run, recorded not explained:** pass 2 (warm) was
SLOWER than pass 1 (cold) — wall 199,426.3 ms / 1,002.877 tok/s vs 139,073.4 ms /
1,438.090 tok/s, compute 194,031.5 vs 132,394.9 ms, transfer 363,524.3 vs
256,427.0 ms — where the recorded ledger reference for this shape is the opposite
(cold ~146 s, warm 141.8 s). Candidate causes, none tested: GPU contention during
pass 2, host page-fault pressure on the 19.5 GiB mapping, or a boundary effect
specific to this shape. Needs the repo's standard replicated adjacent pairs
before any interpretation.

#### T-GPLR-40 (bisect, 2026-09-17): the 144k pass-2 crash — tail chunk vs long-context+repeat

Commit: outer 0a6e54b (bisect scripts + stock-probe attribution + docs)

The 144k A/B (T-GPLR-38 addendum) crashed deterministically in pass 2 of
`--pp-repeat 2` at the same position in both arms (layer 3, slot 0, generation
16; ~37% into the pass). Both shapes shared one new property: a 640-token TAIL
chunk (144000 = 140x1024 + 640 = 70x2048 + 640); every earlier `--pp-repeat`
run divided the prompt exactly. Bisect:
: * Arm B1 `nobreak`: `--pp-tokens 143360` = 140x1024 EXACTLY (no tail), u1024,
  3 slots, ron mode, `--pp-repeat 2`.
  * Arm C1 `tail`: `--pp-tokens 144000` (141 chunks, 640-token tail), identical
  otherwise — re-run for adjacency.
  * Outcome table (pre-registered): B1 survives + C1 crashes -> the TAIL CHUNK
  is the trigger (fix belongs in the tail-chunk graph/slab binding or the reset
  path); B1 crashes + C1 crashes -> tail exonerated, long-context+repeat is the
  trigger (KV clear / act-slab rebind at this ctx); B1 survives + C1 survives ->
  the original crash was order/environment dependent, re-run the u2048 arm.
  No performance claims: this is a failure-isolation run.

Bisect result (measured, 2026-09-17): TAIL EXONERATED — trigger is q8 KV + long ctx + pass 2
: * B1 no-tail 143360 / q8 KV: CRASH in pass 2 at chunk 130, layer 3, slot 0,
  gen 16 — byte-identical signature to the tail-bearing arms.
  * C1 tail 144000 / q8 KV: same crash, same position (adjacency re-run).
  * D1 no-tail 143360 / **f16 KV**: SURVIVES both passes (generation reached 28
  = full pass 2; warm 140,810.8 ms / 1022.7 tok/s — the first clean warm 144k
  reference). Traces `build/llama-cuda/gplr40-*.jsonl`.
  * Conclusion: the trigger is the **quantized (q8) KV cache combined with long
  context and the pass-2 KV clear/reuse path** — not the tail chunk, not the
  ubatch size, not the copy-source mode. Consistent with all prior evidence:
  4k repeat runs (q8, tiny ctx) survived; 200k repeat runs (f16 KV) survived.
  * Defect ownership: the crash is in the q8 KV clear/rebind path under
  `--pp-repeat`, NOT in the layer-residency staging path (f16 staging identical).
  Next step (owner decision): reproduce with stock llama.cpp q8 KV + KV-clear
  at long ctx to decide whether it is upstream or residency-owned before any
  fix is attempted.
  * Process note: two aborted attempts (ctx smaller than the 144,001-token
  file; the generator overshoot again) — the runner now carries the working
  flags; the file-length requirement is ctx >= prompt file tokens regardless
  of --pp-tokens.

#### T-GPLR-40 addendum (2026-09-17): ownership test — stock libllama, no residency path

Commit: outer 0a6e54b (tools/gplr40-stock-probe.cpp + build/run scripts)

Design
: A standalone probe (no layer-streaming, no residency module — plain
  llama_model + llama_context from the vendored tree) that reproduces the crash
  shape with the public API only: fill a q8 KV cache with ~143k tokens in
  1024-token batches (llama_decode), then `llama_memory_clear(mem, true)` and
  fill it AGAIN — mirroring exactly what `--pp-repeat 2` does across the pass
  boundary. Arms: q8 (`--cache-type-k q8_0 --cache-type-v q8_0` equivalent via
  `cache_type_k/v`), f16, and a no-clear control (decode once). u1024, ctx
  144128, CUDA0 only, same model.

Pre-registered interpretation
: * Stock probe CRASHES on the second fill with q8 (and survives f16) -> the
  defect is UPSTREAM (stock KV q8 clear/reuse at long ctx); file an upstream-
  facing issue, do not fix in the vendored tree beyond a workaround, and the
  residency path is exonerated.
  * Stock probe SURVIVES both fills -> the crash is RESIDENCY-OWNED (something
  in the layer-residency pass-boundary reset or its q8 KV interaction); fix
  belongs in src/llama-layer-streaming.cpp / the runner's reset sequence.
  * Stock probe survives q8 but with a WARNING/defrag difference -> treat as
  residency-owned but look at the ubatch/defrag interaction.
  * Failure-isolation run: no performance claims from any arm.

RESULT (2026-09-17, same day)
: Stock probe SURVIVES both fills in both arms -- q8_0 and f16 each completed
  fill 1 (144,001 tokens, u1024, ctx 144128) + `llama_memory_clear(mem, true)`
  + fill 2, exit 0 both (`gplr40-stock-wrapper.log`: `ARM kv=q8_0 EXIT: 0`,
  `ARM kv=f16 EXIT: 0`). -> By the pre-registered matrix the defect is
  **RESIDENCY-OWNED**: it lives in the layer-residency pass-boundary reset or
  its q8-KV interaction (src/llama-layer-streaming.cpp / runner reset
  sequence), NOT in stock q8 KV clear/reuse.
  Placement equivalence verified against the owner's validity challenge: the
  crashing residency runs' own logs show `llama_kv_cache: CUDA0 KV buffer size
  = 1495.47 MiB` (q8 arms) — KV was fully VRAM-resident on CUDA0 there, the
  same placement the probe used (stock defaults; 22.3 GiB total VRAM incl.
  expert banks + q8 KV + compute). The control matches the failing
  configuration on the axis that matters for a KV clear/reuse bug.
  Disclosed differences (what the probe does NOT reproduce): the layer-major
  controller, slot/staging machinery, per-chunk memory contexts held across
  the pass, and the runner's reset sequence — which is exactly where the
  attribution now points. Candidate mechanism for the fix (hypothesis, not
  evidence): the controller keeps `chunk_mctx` alive for the whole pass while
  the runner clears memory between passes; stale q8 cell references in
  pass-2 graph inputs are consistent with the async illegal access at a fixed
  slot generation early in pass 2.
  Deviation from the design: the no-clear control arm was dropped (both
  residency crashes occur after the clear; the f16-vs-q8 pair already
  isolates the axis).
  Build notes: VS 18 BuildTools vcvars (the 2022 install's vswhere dependency
  is broken on this box); probe links `src/Release/llama.lib`, loads DLLs
  from `bin/Release`. Evidence: `tools/gplr40-stock-probe.cpp`,
  `tools/run-gplr40-stock.ps1`, `build/llama-cuda/gplr40-stock-wrapper.log`.
  Next step (owner decision): fix the residency pass-boundary reset, then
  re-run the B1/C1/D1 matrix as the regression.
  * Warm ledger: `other` 1152.8 ms = **route 1161.5 (100.8%)** + tel 1.0 + loop
  −9.7 ms (closure exact by construction; small negative loop = timer jitter).
  80 route events/pass confirmed in the trace.
  * Cross-check: u512's `other` was 237.9 ms with the same 80 events but 4x
  smaller readbacks (1 MiB vs 4 MiB per event, 512 experts x n_tok f32) — the
  per-event cost scales with n_tok (~14.5 ms/event at u2048).
  * Flag-combination trap found and documented: `--tg-tokens 0` skips the
  handoff block entirely while the fit gate can still declare `auto-on`, and the
  checker then rejects the trace (`auto-on but no handoff events`). The 200k
  addendum-3 arms escaped only because their fit gate measured auto-OFF. Recon
  arms must either pass `--tg-tokens > 0` or force `--handoff-mode off`.

#### T-GPLR-41 (recon, 2026-09-17): pass-boundary memory-lifecycle telemetry — prove or refute the stale-reference hypothesis

Status: done (falsifier fired — P2 recorded; escalated per plan)
Depends on: T-GPLR-40 addendum (stock probe => RESIDENCY-OWNED verdict)
Commit: outer 0a6e54b (instrument + analyzer + scripts)

Goal
: Before fixing the q8-KV 144k pass-2 crash, prove its mechanism with
  telemetry. Working hypothesis from the stock-probe attribution: a
  stale-reference between the prebuilt per-chunk graph sets and pass 2's
  cleared-and-refilled memory (baked KV-view shapes ne[2] = build-time n_kv,
  and/or baked KV storage pointers, diverging from pass 2's live layout).

Instrument (implemented, behind --pp-mctx-recon 1)
: In the layer-major mctx loop, AFTER each chunk's apply and BEFORE the sweep:
  (a) one `mctx` event per chunk: pass index, live n_kv (public
  llama_kv_cache_context::get_n_kv()), recurrent rs_z, and the selected set's
  (chunk_id, rs_z, n_outputs); (b) one `mctx_kvview` event per exec layer per
  chunk: the set's baked cache_k_l*/cache_v_l* view shapes (ne[2]) and the KV
  storage device addresses the views point at. Both passes emit; pass-2 events
  all precede the sweep, so the crashing run leaves a complete picture.
  Diagnostics only: host-only reads, no execution-order change, unknown event
  types ignored by both checkers. Refused under --mtp (different controller).

Pre-registered interpretation (falsifiable)
: P1 CONFIRM (stale-reference family): pass-2 vs pass-1 divergence in any
  emitted fact — live_n_kv trajectory differs at the same chunk, set
  selection differs (rs_z or n_out), baked k_ne2/v_ne2 != the pass's live
  n_kv at some chunk, or KV storage addresses differ between passes. Fix
  then targets the diverging fact; regression = B1/C1 re-run surviving.
: P2 REFUTE (falsifier): every emitted fact byte-identical between pass 1
  and pass 2 AND pass 2 still crashes => the stale-reference family as
  instrumented is NOT the mechanism; escalate to the un-instrumented
  suspects one at a time (recurrent zero-view state, scheduler alloc-graph
  reuse across the clear, async-feed/copy-stream interaction with cleared
  cells).
: P3 TIMING: pass 2 completes under the recon (crash not reproduced) =>
  the crash is timing-dependent (the instrument's host work perturbs it);
  re-run without the flag to confirm reproduction before any conclusion.

Non-goals
: No fix in this task. No performance claims. No behavior change when the
  flag is off.

Validation
: Run the exact crashing shape (B1: --pp-tokens 143360, u1024, 3 slots,
  mmap-registered, --pp-repeat 2, q8) with --pp-mctx-recon 1, plus the f16
  control (D1 shape). Expected: q8 crashes in pass 2 with complete pass-2
  mctx streams in the trace; f16 completes and passes --check-telemetry
  (new event types ignored). Analyzer: tools/analyze-gplr41.py.

Escalation condition
: P2 (falsifier fires) or P3 (repro lost) — stop, record, return to the
  owner with the discriminating facts before any code fix.

RESULT (2026-09-17, same day — P2: FALSIFIER FIRED)
: The crash REPRODUCED under instrumentation (q8 arm, exit -1073740791,
  illegal access in pass 2, detected at layer 3 / slot 0 / chunk 130 — the
  exact recorded signature; f16 arm completed both passes, checker OK,
  14878 events, new event types ignored). Pass-2 lifecycle facts were
  emitted for ALL 141 chunks BEFORE the sweep, so the crashing run has a
  complete pass-2 picture. Measured: mctx=282 (141 chunks x 2 passes),
  kvview=11280 (141 x 40 x 2) per arm. **Every emitted fact is identical
  between pass 1 and pass 2:** live_n_kv trajectory (chunk 130 = 144128
  padded in both), rs_z (-1 continuing in both), set selection (chunk_id=1,
  n_out=0 in both), baked cache_k/cache_v view shapes ne[2] (all chunk x
  layer sites), and KV storage + view device pointers (identical — no
  reallocation, no cell relocation, no defrag across the clear).
  => The stale-reference family AS DEFINED (stale shapes, stale pointers,
  changed set selection, changed live n_kv) is NOT the mechanism. The
  crash cannot be explained by anything this instrument reaches; per the
  pre-registered matrix the task stops here and escalates.
  Notes: (a) the instrument is expensive on this shape (~40% wall overhead
  from 11.3k per-layer graph walks per run; f16 walls 149.5/142.0 s vs the
  D1 references 134.5/140.8 s) but preserved the crash — gross timing
  sensitivity is excluded at the applied perturbation level; (b) pass 1's
  KV buffer starts as fresh cudaMalloc garbage while pass 2's starts
  memset-zeroed — a data-dependent mechanism would predict pass 1 to be
  the fragile pass, the opposite of what happens; (c) remaining suspects
  (un-instrumented, for the owner to prioritize): recurrent-side state
  across the boundary (GDN cells/s_copy bindings the hybrid clear resets
  vs what the prebuilt zero-views assume), scheduler CUDA-graph
  capture/exec-update state persisting across alloc_graph after the clear,
  and async-feed/copy-stream interaction with the cleared cells; a cheap
  discriminator is --kv-cache q4 (crash => quantized-KV path generally,
  survive => q8-specific). Evidence: gplr41-q8-mctx.jsonl/.log,
  gplr41-f16-mctx.jsonl/.log, tools/analyze-gplr41.py,
  tools/run-gplr41-mctx-arms.ps1.

#### T-GPLR-42 (discriminator, 2026-09-17): q4-KV at the 144k pass-2 shape — quantized-KV-general vs q8-specific

Status: done (row 1 — QUANTIZED-KV-GENERAL)
Depends on: T-GPLR-41 (mctx recon REFUTED the stale-reference family; this is
its cheapest listed discriminator)
Commit: outer 0a6e54b (scripts)

Goal
: Decide whether the pass-2 crash needs q8 specifically or any quantized KV.
  Arms (B1 shape, u1024, 3 slots, mmap-registered, af=1, --pp-repeat 2, NO
  mctx recon): `q4` (the discriminator) + an adjacent `q8` confirm arm in the
  SAME mutex hold, so a q4 survival cannot be blamed on environment drift.

Pre-registered interpretation (falsifiable)
: q4 CRASHES in pass 2 (with q8 also crashing) => the trigger is the
  quantized-KV path generally; q8-specific hypotheses (block-scale handling)
  drop out; suspicion shifts to shared quantized-KV machinery across the
  clear (view/buffer handling, quantized attention path).
: q4 SURVIVES both passes (with q8 crashing adjacently) => the mechanism is
  Q8-SPECIFIC: inside vendored q8-KV code engaged by the layer-major pass
  across the boundary (Q8_0 32-wide block handling, scale layout,
  dequant-kernel interaction). Fix targets the q8 path.
: q4 survives AND q8 fails to crash => environment changed; re-establish
  the q8 repro before any conclusion (discriminator aborted).
: Failure isolation only; no performance claims from any arm.

Validation
: tools/run-gplr42-q4disc-arms.ps1 under the GPU mutex; analyzer
  tools/analyze-gplr40.py for pass/crash positions; checker on surviving
  traces.

Escalation condition
: Third row of the matrix (q8 repro lost) — stop and report.

RESULT (2026-09-17, same day — row 1: q4 CRASHES, trigger is QUANTIZED-KV-GENERAL)
: Both arms ran under one mutex hold on the free GPU. q4 arm: pass 1
  complete (143,013.2 ms, 1006.9 tok/s), pass 2 CRASHED — illegal memory
  access at chunk 130, layer 3, slot 0, generation 16 (analyzer line: `pp
  summaries: 1 (crashed/incomplete) | last launch: chunk 130 layer 3 slot 0
  gen 16`); q8 confirm arm: identical crash at the identical position
  (144,543.2 ms pass 1). Byte-identical signatures; KV placement CUDA0
  (791.72 MiB q4 / 1495.47 MiB q8).
  => By the pre-registered matrix the trigger is the QUANTIZED-KV path
  generally, NOT q8-specific block handling. Suspects narrow to machinery
  shared by Q4_0 and Q8_0 KV but absent for f16, engaged across the pass
  boundary in the layer-major residency pass: (a) quantized-KV buffer/view
  handling across clear+refill (e.g. the [TAG_V_CACHE_VARIABLE] transposed-V
  path differs between quantized and f16 caches), (b) the quantized
  attention/dequant kernel path interacting with pass-2 graph reuse, (c)
  buffer-clear semantics on quantized KV buffers. The f16-vs-quantized axis
  is now the sharpest known fact about this defect.
  Next discriminator (owner decision): rebuild the per-chunk graph sets
  between passes (bypasses graph reuse entirely; distinguishes
  graph-reuse-across-clear from memory-state mechanisms) or bisect the
  position (why layer 3 / gen 16 / ~37% into pass 2 at every ubatch and
  every quantization). Evidence: gplr42-q4-disc.{jsonl,log},
  gplr42-q8-confirm.{jsonl,log}, tools/run-gplr42-q4disc-arms.ps1.

#### T-GPLR-43 (probe, 2026-09-17): graph-rebuild between passes — graph-reuse-across-clear vs memory-state

Status: done (row 1: graph-reuse-across-clear CONFIRMED as the mechanism)
Depends on: T-GPLR-42 (quantized-KV-general), T-GPLR-41 (lifecycle facts
identical across passes)
Commit: nested 0ffd0ab14, outer 0a6e54b

Goal
: Decide the last open split: does pass 2 crash because the layer-major pass
  REUSES prebuilt graph sets across the clear (graph-reuse-across-clear), or
  because of memory-state the rebuild cannot see (recurrent/GDN cells,
  scheduler internals)? Mechanism probe: re-run the full prebuild between
  passes so pass 2 executes against COMPLETELY FRESH graphs (new cgraphs,
  rebuilt KV views against the post-clear memory), then run the B1 q8 shape.

Implementation (behind --pp-prebuild-per-pass 1, runner flag)
: In the --pp-repeat pass loop, after the pass-2 memory clear + reset_stats +
  tel_pp_pass, re-invoke llama_layer_streaming_prebuild(ctx, batch,
  tg_tokens, st). begin_set's existing rebuild path (find_set match ->
  res->reset + graph lists cleared + rebuild into the same ctx/slab offsets)
  makes this safe; no act_bind re-emission occurs on rebuild (bind-count rule
  preserved); builds0 is advanced to the new build_generation so per-pass
  phase summaries keep graph_builds = 0 (checker-clean). Refused under --mtp.

Pre-registered interpretation (falsifiable)
: REBUILT-ON SURVIVES both passes (adjacent rebuilt-off q8 crashes) =>
  graph-reuse-across-clear IS the mechanism: something the rebuilt graphs
  re-derive (KV views/copy nodes bound to the live cache) fixes pass 2. FIX:
  rebuild (or equivalent safe rebind) between passes in the residency path;
  regression = B1 + C1 with the fix surviving with checker OK, f16 control
  unchanged; performance cost of the rebuild measured and reported.
: REBUILT-ON STILL CRASHES at the same position => memory-state mechanism
  (the rebuild cannot see it); escalate to recurrent-side instrumentation
  (T-GPLR-41 suspect 2); no fix attempted.
: Rebuild itself fails loudly (prebuild error) => record and stop.

Validation
: tools/run-gplr43-rebuild-arms.ps1 (rebuilt-on + rebuilt-off q8, B1 shape,
  one mutex hold); analyzer tools/analyze-gplr40.py; checker on any
  surviving trace.

Escalation condition
: Second row (rebuild cannot fix) — stop and report before any fix.

RESULT (2026-09-17, same day — row 1: REBUILT-ON SURVIVES, mechanism CONFIRMED)
: One mutex hold, adjacent arms, B1 q8 shape. **q8-rebuild (pass 2 against
  FRESH graphs): SURVIVED both passes** (gen 28 = complete pass 2; pass 1
  78,531.3 ms / 1833.7 tok/s, pass 2 111,790.7 ms / 1288.1 tok/s; rebuild
  cost 7.4 ms, build_generation 3 -> 6). **q8-norebuild control: CRASHED at
  the byte-identical signature** (chunk 130, layer 3, slot 0, gen 16).
  => By the pre-registered matrix, graph-reuse-across-clear IS the
  mechanism: pass 2 fails only while executing graphs whose executed state
  was derived at prebuild time against the PRE-clear cache. Combined with
  T-GPLR-41 (view shapes, storage pointers, set selection, live n_kv all
  identical across passes), the stale element is something the rebuild
  re-derives that metadata does not capture (executed/captured graph state
  bound to the pre-clear cache) — on quantized KV only (f16 survives
  reuse). The 7.4 ms rebuild is timing-negligible, and T-GPLR-41 already
  excluded gross-timing effects at ~40% overhead.
  Probe artifact (fix prerequisite): the rebuilt arm exited 1 — its trace
  was rejected by --check-telemetry with 564 act_bind violations: the
  rebuild re-emitted every act_bind (282 pairs x 2). The binds describe
  the slab layout, unchanged by a rebuild; re-emission is redundant.
  FIX (T-GPLR-43 fix, next block): (1) emit act_bind only when a set is
  newly created (rebuild is silent); (2) make the per-pass rebuild the
  DEFAULT for --pp-repeat > 1 (explicit --pp-prebuild-per-pass 0 restores
  the single-prebuild behavior for evidence reproducibility).

  FIX RESULT (2026-09-17): the 564-violation artifact persisted through the
  first fix build; forensics traced it to a SECOND emitter — the pw-mode
  act_bind block in llama-context.cpp (T-Q38PW-00), which fires on every
  prebuild and pw_probe is ON by default in layer-major (main.cpp), so
  begin_set's own guard was inert in these runs and the "per-pass double
  emission" was the pw emitter re-emitting all chunk binds on the pass-2
  prebuild (141 chunks x 2 slabs x 2 passes = 564; the 4k diag showed
  8 x 2 x 2 = 32). Fix applied: the pw emitter now fires once per process
  (pw_bind_emitted latch); the diag [ls-set] stderr probe was removed as
  residue. REGRESSION (one mutex hold, B1 q8 / C1 q8-tail / D1 f16, default
  per-pass rebuild): ALL THREE SURVIVE both passes (last launch chunk 140,
  layer 39, gen 28 in each) with strict checker OK and zero act_bind
  violations. B1 walls 78,102.6 ms / 1843.7 tok/s (pass 1), 111,000.5 ms /
  1297.3 tok/s (pass 2). The q8 pass-2 crash is FIXED; f16 behavior is
  unchanged. The known checker rule "handoff_mode auto-on but no handoff
  events" fires only in TG-less probe shapes (--tg-tokens 0), documented in
  T-GPLR-39; all regression arms ran with TG so it does not apply.

#### T-GPLR-38 addendum (2026-09-17): ubatch A/B at the middle shape — 100k tokens, u1024 vs u2048

Motivation
: The 4k sweep (T-GPLR-38) found u1024 best and u2048 worse, but the u2048 loss
  was largely the route telemetry measuring itself (T-GPLR-39 recon: 1161.5 ms
  of 1152.8 ms `other`). At 100k the per-bank chunk count (~24-25) sits BETWEEN
  the two regimes already measured (8 at 4k, ~391 at 200k), so this shape tests
  where the crossover actually sits — and doubles as a route-cost scaling data
  point (same 2-chunk-stride x 40 layers = 80 events/pass, each now n_tok x 512).

Shape
: `--pp-tokens 100000` (anchor-100000.txt, generated by the semantic-QA needle
  generator), ubatch 1024 vs 2048, 3 slots, ron (churn-free) mode, kv q8,
  layer-major, af=1, `--pp-repeat 2`, `--tg-tokens 96` (keeps the handoff path
  alive; see the T-GPLR-39 trap note).

Pre-registered prediction (written before the run)
: * 100k is compute-dominated but less extremely than 200k: per-bank compute
  ~24 chunks x ~8 ms = ~192 ms vs a 31.4 ms staging window — staging ~hidden.
  Predicted warm walls within ~2% of each other (u1024 ~ 72-75 s, u2048 ~
  72-76 s); the route telemetry overhead (~1.16 s/pass at u2048, ~0.29 s at
  u1024, from the same 80 events with 4x-different readbacks) is ~1.6% and
  ~0.4% of the wall respectively — the same absolute pattern as 4k, diluted.
  * FALSIFIER: if u2048 beats u1024 by > 2%, the 4k ranking was telemetry-only
  and fat ubatches genuinely win from 100k up; if u2048 loses by > 4%, the
  host-side residue still grows super-linearly with ubatch beyond the route
  term and T-GPLR-39's `loop` bucket needs a second look at this shape.

Validation commands
: * `tools/run-gplr38-100k.ps1` (two adjacent arms, one mutex hold, ~10 min)
* `python tools/analyze-gplr37.py build/llama-cuda/gplr38-100k-u{1024,2048}.jsonl`

Run result (2026-09-17): INCOMPLETE — deterministic pass-2 crash in BOTH arms.
: * The generated anchor tokenized to 144,001 tokens (generator overshoot);
  arms ran `--pp-tokens 144000` (ctx 144256, KV q8 1.46 GiB, act slabs 0.75 GiB).
  * Pass 1 completed cold in both arms (u1024 80,631.8 ms / 1785.9 tok/s;
  u2048 76,972.0 / 1870.8) but the delta is arm-order-confounded and NOT a
  verdict. Pass 2 crashed identically in both: CUDA illegal memory access at
  chunk 130/141 (u1024) and 60/71 (u2048), layer 3, slot 0, generation 16 —
  ubatch-independent and deterministic.
  * Both shapes end in a 640-token TAIL CHUNK (144000 = 140x1024 + 640 =
  70x2048 + 640); every earlier `--pp-repeat` run divided the prompt exactly.
  Candidate cause: pass-boundary reset vs tail-chunk graph/act-slab binding.
  * Flag traps hit and documented: the needle generator's token target is
  unverified (overshoot); at this KV size the handoff fit gate is auto-OFF so
  `--tg-tokens 96` refused (arms used `--tg-tokens 0`); strict checkers reject
  the crash traces (no warm pass) — correct behavior.
  * Follow-up (new candidate task): bisect exact-multiple `--pp-tokens
  143360` (140x1024, no tail) vs 144000 to separate tail-chunk from long-ctx
  + repeat, then fix the reset path. Route telemetry measured 4.39 s/4.31 s
  per pass (~3%), extending the T-GPLR-39 scaling line.

### T-GPLR-31: Locked-RAM probe — does the driver grant a full-pool-size pinned allocation on this box?

Status: done (2026-09-16) — verdict: allocation is NOT the blocker; 18 GiB pool grantable with large headroom
Depends on: T-GPLR-29 (probe methodology; cudaHostRegister OOM precedent;
Gen4 x8 laptop link), T-GPLR-30 (the fixed40 arm already held 17,478 MiB of
pinned pool buffers successfully — the direct motivation for this probe).
Commit: NO nested commit - the probe is a standalone .cu under tools/, not part of the
  engine. outer 7b83828 (2026-09-16: introduced this task's block AND its verdict); outer
  a58fdee (2026-09-16: the probe tooling - tools/gplr31-pinpool-probe.cu (116 lines),
  tools/run-gplr31-probe.ps1, tools/run-gplr31-262k.ps1, tools/run-gplr31-262k-f16.ps1).
  No versioned evidence artifact exists for this task by policy: the probe output lives
  under build/ (`gplr31-pinpool-probe.log`, `gplr31-262k-f16-af1pinned.jsonl`), which
  .gitignore excludes, so the verdict text above is the record.

Goal
: Answer, before any persistent-pool design work, the question T-GPLR-30
  flagged as the first thing to check: does the CUDA driver grant ~18 GiB of
  `cudaHostAlloc` pinned memory on this box, at what N does it stop, what
  does refusal look like, and is the earlier `cudaHostRegister`
  `cudaErrorOutOfMemory` (T-GPLR-29 case D) a file-backed-pages issue or a
  general registration limit.

Scope
: Standalone throwaway probe (tools/gplr31-pinpool-probe.cu, nvcc, outside
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
  direction that EXPANDS feasibility. Probe tools/gplr31-pinpool-probe.cu,
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
