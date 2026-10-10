# T-GPLR-61 — per-expert CPU/GPU split at token generation (`moe-split-expert-ffn`)

> **SUPERSEDED AS THE CONTRACT, 2026-10-01. This file is now the DERIVATION, not
> the task.** The live, `ready` block is
> `### T-GPLR-61 (ready)` in
> `docs/plans/generic-prefill-layer-residency-plan.md`. Both blockers that held
> this at DRAFT are discharged: the SDD amendment is written
> (`docs/layer-residency-sdd.md` section 4, the **`moe-row-split` class**), and
> the execution-shape question is resolved **OUTSIDE** the row-set prohibition
> (item 5 below, decided 2026-09-30). The live block is additionally **re-shaped
> onto the 8 GB target card**, which changes what the mechanism is *for*: at
> K=4 device residency caps at 12.5% of a bank, so most demanded experts have no
> device residency available at all and the split stops being a TG tuning knob.
> The cost gate below is unchanged and is not to be re-derived (owner directive
> 2026-10-01). Everything in this file — the withdrawn measurements, the
> reference-project provenance, the three mandatory test cases it took them to
> find, the corrected §8 arithmetic — is retained as the record behind the
> contract. Where this file and the live block disagree, **the live block wins**.

Status: **READY (owner-directed 2026-10-01)** — see the live block. This file's
own status line is retained as written on 2026-09-29. Owner direction: the split
is **per expert, per layer, per token, and not fixed** — experts present on the
GPU compute there, experts not yet present compute on the CPU, and the layer does
not wait for them. The predictor's role is to raise the resident share, not to
choose the split.
Dependencies: T-GPLR-13/T-GPLR-19 (row-set and row-cache execution shape, which
this must not violate), T-GPLR-16 (the closed negative that motivates the shape
and the hard constraint it produced), T-GPLR-51 (the measured fact that the copy
wait is already overlapped, which is what makes a cost gate mandatory),
T-GPLR-59 (the resident-share measurement that sizes the CPU share), T-GPLR-07
(semantic-QA answer gate), SDD section 4 (a new package/placement class).
Commit: unassigned (no commit authorized).

Goal. Let a token-generation layer begin its expert FFN before every demanded
expert row has landed, by computing the rows that are still **PENDING** on the
CPU and the rows that are already resident on the GPU, and summing both into the
layer output. The layer never waits on a row copy that has not completed. The
split is decided at demand time from actual residency, per layer and per token;
nothing about it is fixed at run start.

Scope. Opt-in, CUDA+CPU, DEFAULT OFF, single process, **TG only**. A
residency-driven split of the expert FFN into a device sub-op and a host sub-op;
a trigger and cost gate; a fixed-width padded execution shape so the persistent
graph set stays static; a non-finite guard; telemetry that separates device
experts, host experts, and rows the split skipped. The identity probe is the
blocking gate. The reference project's `V4_CUDA_TG_CPU_DIRECT` is a **read-only
source of the mechanism**, not a dependency.

Non-goals.

- **No placement-fixed CPU banks.** A per-layer static CPU/GPU split is a
  different mechanism and is not implemented by this block.
- **No change to the demand set.** The router's executed IDs for the window are
  authoritative and untouched. The split never drops, substitutes, delays or
  reorders an expert.
- **No change to the predictor.** This block consumes whatever is resident. It
  does not add, retune or re-gate any predictor; T-GPLR-58/59/60 records stand.
- **No `GGML_OP_OFFLOAD_MIN_BATCH` override and no scheduler expert-rowcopy** for
  the affected layers. Both are forbidden; see the hard constraint.
- No prefill change. In layer-major PP the bank arrives whole, so the split has
  nothing to split; this is a TG class by construction.
- No change to routing, router arithmetic, top-k, quantization bytes, KV or
  recurrent state, logits pattern, sampling, the add-chain reduction, or any
  default path.
- No performance claim. The first deliverable is feasibility/correctness.

Design and failure contract.

1. **Structural exactness (why this is not a numerics gamble).** The MoE output
   is a plain additive reduction over experts
   (`src/llama-graph.cpp`: `experts = ggml_mul(build_lora_mm_id(down_exps, ...),
   weights)`, then one `view_2d` per expert and a `ggml_add` chain). Splitting the
   expert index set into `S_gpu` and `S_cpu` and summing the two partial results
   computes the same sum. The only permitted divergence sources are (a) the
   floating-point summation order and (b) the CPU-vs-CUDA quantized kernel
   difference. Corruption is not an available failure mode. Neither permitted
   source is claimed to be zero — see the identity gate.
2. **Trigger — resident vs not, which the measurements make decisive.** The
   split fires for every demanded row that is **not already resident** in the
   layer's CUDA-resident row set. A row that has not been *issued* is not a
   candidate (that is a staging bug and must refuse loudly); a row that is READY
   always runs on the GPU, because then there is no transfer to save. This is
   the reference project's trigger (`demand_storage_pending`) and the measured
   constants below are what make it cheap rather than a judgement call.
3. **Cost gate — measured, not estimated. CORRECTED 2026-09-30; the first
   measurement was contaminated and is withdrawn.** `tools/expert-cpu-bench`
   measures the real op chain on the real Q4_K bytes at the TG shape
   (n_tokens=1, n_expert_used sweep), 300 iterations, medians.

   **The withdrawn run (2026-09-29).** It used `--pool 64` and an expert-id
   stride of 1, which was hard-coded, undocumented and unnoticed. At stride 1,
   iteration *n* reads experts {n..n+7} and iteration *n+1* reads {n+1..n+8}:
   **7 of 8 slabs were touched microseconds earlier and are still L2/L3-warm**,
   so the marginal cost being measured was a cache cost, not a DRAM cost. The
   tell is in the run's own JSON: its `gbps` column rises 41.9 -> 128.6 GB/s as
   k goes 1 -> 8 (+207%). Cold DRAM cannot get faster the more of it you read;
   the k=1 cell (41.9 GB/s) is this box's actual DDR rate. The bug is now fixed
   and parameterised — `--id-stride N` (default 1, which reproduces the old run
   byte-for-byte) — and the value is emitted into the JSON.

   **The clean run (2026-09-30), arm C: `--pool 256 --id-stride 8`**, i.e. a
   full 256-expert bank and iteration-disjoint expert sets. Its `gbps` column is
   flat (53.5 -> 49.0), which is what a genuinely bandwidth-bound read looks
   like:

   | demanded experts per layer | CPU (8 thr, CLEAN) | withdrawn 2026-09-29 | link @ 8.26 GB/s (best measured end-to-end) | link @ ~7.35 GB/s (pageable) |
   | --- | --- | --- | --- | --- |
   | 1 | 33.1 us | 46.6 us | 214.2 us | 240.7 us |
   | 2 | 67.0 us | 56.4 us | 428.4 us | 481.5 us |
   | 4 | 144.4 us | 74.1 us | 856.9 us | 963.0 us |
   | 8 | 288.6 us | 108.9 us | 1,713.8 us | 1,926.0 us |

   The 14.6 GB/s column that appeared here previously is deleted: that figure is
   `copy_gpu_ms` wire time, withdrawn as an end-to-end feed rate. 8.26 GB/s is
   the best end-to-end feed ever achieved on this box.

   **What changed and what did not.** The **marginal** cost of one additional
   expert on the CPU is **~36.5 us**, not ~8.7-9.8 us — understated **~3.9x**.
   The k=8 total is understated **2.62x** against the withdrawn run and **1.73x**
   against a same-session stride-1 control. **The rule survives unchanged**: at
   every k the host is 3.4-6.7x cheaper than the link, so "not resident -> host"
   still holds, full stop. Two things about the rule's *argument* change:
   - The margin is **3.4-6.7x, not 8.9-17.7x**, and it is **constant in k**
     rather than growing. The contaminated data made batching look like it
     improved the CPU's standing; it does not.
   - The direction of the error was *against* the GPU path. Correcting it
     **lowers** the transfer break-even hit rate from ~88.8% to **83.2%** at
     8.26 GB/s, and to **79.9%** at the marginal 10.35 GB/s measured later that
     day. Measured retention reaches **0.7255** at K=8 and **0.8319** at K=12,
     which *does* cross it. **The break-even argument is dead either way** —
     not because the gap is unclosable but because it was never the binding
     term: the streaming path's non-transfer floor alone exceeds the whole stock
     token, so it loses at `h = 1`. See item 8, Q2.

   The estimator stays explicit and host-side so the rule can be *re-derived*
   when the host or the model changes, and its inputs/outputs stay telemetry
   (`rows_gate_cpu`, `rows_gate_gpu`, `gate_estimated_host_us`,
   `gate_estimated_wait_us`). A named gate policy is never silently widened; a
   policy that cannot be evaluated refuses the run rather than defaulting.
   **Bounds of the measurement, carried into any claim:** this is the expert FFN
   alone, on a microbenchmark, with the CPU work serial to everything else. The
   whole-model `-ncmoe 20` differential implies ~430 us per layer. Against the
   withdrawn figure that was **~4x and unexplained** — the draft flagged it as
   such. Against the clean 288.6 us it is **1.49x**, which is exactly the
   scheduler-overhead-and-no-overlap gap the draft already attributed it to.
   **The whole-model differential was right all along; the microbenchmark was
   wrong.** That reconciliation is the strongest evidence that the clean number
   is the true one.
4. **Fixed-width padded execution shape (required for the graph cache).** The
   device sub-op and the host sub-op are each emitted at a **constant width**
   (top_k for both), with absent experts padded by a zero router weight, so
   every (token, layer) produces the same graph shapes and the persistent graph
   set never rebuilds because of a subset. At TG (`n_tokens=1`, top_k=8) the
   extra work is `mul_mat_vec`-shaped and negligible next to a 1.77 MB row
   transfer. A variable-width sub-op is explicitly out of scope and must be
   escalated, not improvised.
5. **Hard execution constraint (inherited, non-negotiable).** The host sub-op
   executes with the CPU backend against **deliberately host-bound weight
   tensors** for the life of the graph set. The scheduler's expert rowcopy
   (`ggml_backend_sched_compute_splits`) and any op-offload rewrite of a weight
   source must remain disabled for these layers, as SDD section 4 already
   requires for row-set graphs. Rationale is measured: the T-GPLR-16 MIN_BATCH=1
   route — CUDA `MUL_MAT_ID` consuming scheduler-staged host rows — deadlocked
   the device on the 3rd TG token, with and without CUDA graphs and with and
   without rowcopy (2026-09-12).

   **DECIDED 2026-09-30: OUTSIDE the prohibition.** Step A is discharged; this
   task is not stopped by it. Reasoning, read directly against the constraint
   text at `docs/layer-residency-sdd.md:337-351`:

   - The constraint's subject is narrow and explicit: *"`MUL_MAT_ID` **in a
     row-set graph** must always consume CUDA-resident weights owned by the
     matching slot generation"*, and what is forbidden is *"the backend
     scheduler's expert row-copy (`ggml-backend.cpp` compute_splits) and any
     op-offload path that **rewires a weight source** to a compute-usage staging
     buffer"*. Both clauses govern a **CUDA** op whose **weight** source the
     scheduler silently redirects.
   - The host sub-op is neither. It is a **CPU-backend** op reading a tensor at
     its own declared host placement. No CUDA op consumes host weights; no
     weight source is rewired; nothing is staged into a compute-usage buffer.
     The weights are where the graph says they are, for the life of the graph
     set.
   - The deadlock evidence is specific to the forbidden shape: *"CUDA
     `MUL_MAT_ID` at token batch consumed scheduler-staged **host-bank
     weights**"*. Deliberate host placement consumed by a host op was never the
     failing configuration, and it is what ordinary `-ngl` partial offload does
     on every run of this repo.

   **Residual hazard, named rather than assumed.** Activations must cross the
   backend boundary: `cur` is CUDA-resident, the host sub-op needs it, and the
   partial must come back for the add chain. That IS scheduler cross-backend
   staging inside graph execution — of **activations**. The constraint does not
   reach it (it says *weight* source, and the deadlock evidence is about
   host-bank *weights*), so it is permitted by both letter and evidence. But it
   is permitted, not proven safe: it is where the per-layer sync cost will
   actually land, and it is the nearest neighbour of a known device-deadlock.
   Therefore step D's guard must assert **specifically** that no *weight* source
   on this path is ever rewired by the scheduler, and step E must measure the
   activation round-trip rather than assume it free. If a measurement ever shows
   the scheduler rewiring a weight source here, that is an immediate stop and an
   escalation, not a tuning problem.
6. **Loud failure contract.** Refuse before model load (exit 2, `run_refused`
   with the byte arithmetic) on: an unknown gate policy; a gate that cannot be
   evaluated; a shape that cannot support the split (handoff active, a
   non-MoE residency policy, a non-CUDA backend, MTP draft loop if the split
   cannot be made draft-safe). At run time: a non-finite value from any host
   expert is a **loud abort** naming layer and expert ID; an expert that is
   demanded, PENDING, gate-approved for the host, and then not executed is a
   loud abort; a demanded row that is neither READY nor PENDING is a loud abort.
7. **Telemetry.** Per (layer, token): `experts_total`, `experts_device`,
   `experts_host`, `rows_gate_cpu`/`rows_gate_gpu`, gate estimates, host
   execution wall, `staged_bytes_avoided`, and the non-finite scan result. The
   split counters are the mechanism's own falsifier instruments and must be
   emitted even when the split never fires. Existing `rowset_fetch` semantics are
   unchanged; a trace that enables the split and omits or contradicts the new
   fields is refused, not downgraded, and a legacy trace stays valid when the
   option is absent.

Implementation steps.

- **A. Verify the execution-shape question first (host-only, docs-level).**
  Read SDD section 4's constraint text and the T-GPLR-16 deadlock evidence
  against the deliberate-host-tensor mechanism; record the decision and its
  reasoning in this block. If the answer is "inside the prohibition", this task
  stops here and the owner is told why.
  **DONE 2026-09-30: OUTSIDE. Decision and reasoning in item 5.** The task
  proceeds; the residual activation-crossing hazard named in item 5 is carried
  into steps D and E.
- **B. Host-only policy and shape.** Pure functions: residency classification at
  demand time (READY / PENDING / absent), the cost gate, the fixed-width padding
  plan, and the host/device sub-op descriptor. No CUDA handles. CTest fixtures
  covering: all-READY (split must not fire), all-PENDING with gate open, gate
  closed by estimate, a mixed set, an absent row (refuse), padding producing
  zero contribution, and a gate policy that cannot be evaluated (refuse).
- **C. Engine.** Extend the row-set path to build the two sub-ops against the
  existing add-chain reduction, with the host sub-op's weights bound to the
  in-flight source rows on the host. Slot, generation, copy-event and
  consumer-done lifecycles are unchanged; the in-flight copies keep running and
  their completion still retires normally — the split must not cancel or consume
  a copy the mechanism still owns.
- **D. Guards.** Extend the host-only T-GPLR-16 guard to assert the MIN_BATCH
  override and the scheduler rowcopy are unreachable on this path; add the
  non-finite abort; add the absent/never-executed aborts.
- **E. Focused host validation**, then the identity probe. No performance arm.

Falsifiable acceptance criteria.

- **AC1 (policy and shape, host-only, blocking).** Fixtures prove: all-READY
  never splits; a PENDING row goes to the host only when the gate opens; the
  gate closing sends it back to the ordinary wait path unchanged; padding
  contributes exactly zero; an absent demanded row refuses; an unevaluable
  policy refuses; the emitted sub-op descriptors are constant-width for every
  input.
- **AC2 (IDENTITY, blocking, the real gate).** On the 512-token anchor with the
  split enabled and the gate forced open: greedy token IDs **byte-identical** to
  the recorded all-GPU streaming arm for the same shape, `check-telemetry` OK,
  aggregate `PASS` 5/5, negative control FAILing as designed. **If AC2 fails the
  task stops there** and records the falsification — no placement tuning, no
  gate tuning, no rescue attempt. Per AC1's structure the failure can only be
  fp-order or kernel-domain divergence, and that distinction must be stated in
  the record.
- **AC3 (no hang).** The arm completes inside the recorded wall band with no TDR
  and no `cudaErrorLaunchTimeout`. A hang is a hard fail even if AC2 passed.
- **AC4 (instrumentation, not performance).** The split counters are true: the
  sum of `experts_device` and `experts_host` equals `experts_total` at every
  (layer, token); in-flight copies complete and retire normally; `staged_bytes_
  avoided` is consistent with `experts_host` and the measured row bytes.
- **AC5 (inertness is a reportable result, not a failure).** If
  `experts_host == 0` for the whole run, the mechanism did not fire. That is a
  valid, recorded outcome — and it must be reported as such, never as a pass
  with an implied benefit.
- **AC6 (no default change).** Flag absent: trace byte-identical to the recorded
  control, host suite green, dense regression unchanged.
- **AC7 (the falsifier the owner should read first).** The recorded
  `experts_host / experts_total` share at the 512 anchor. The T-GPLR-59 frontier
  run measured a demand-hit share of 0.4518, so the *prior* is that the host
  carries a large minority-to-majority of expert work. A large host share is not
  a defect to be tuned down — it is the input to the performance arithmetic, and
  until that arm is authorized no speed claim of any kind may be made.

Focused validation commands.

```powershell
# 0. host-only, no GPU
cmake --build build/llama-cuda --config Release --target test-layer-streaming-layout test-layer-streaming-tg-spec --parallel 16
ctest --test-dir build/llama-cuda -C Release -R "^(test-layer-streaming-layout|test-layer-streaming-tg-spec)$" --output-on-failure
python tools/check-tg-predict.py --selftest
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-tests.ps1 -Suite host

# 1. the blocking identity probe (GPU, mutex, owner-invoked)
powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -TimeoutSeconds 1800 -Command "<recorded arm: canonical 512 anchor, split enabled, gate forced open, --check-telemetry strict, --tg-text, --route-telemetry 1>"
#    expected: exit 0, checker OK, ids identical to the recorded control,
#    aggregate=PASS 5/5, negative control exit 1, and a NON-ZERO experts_host.
#    a refusal (exit 2) is a valid recorded result and discharges AC1 only.

# 2. only after AC2 passes, and only with fresh owner authorization:
python tools/check-route-parity.py <trace>     # decode-free pass-2 integrity
python tools/decode-tg-ids.py <trace>          # answer-shape check
```

Expected results: every host step green; step 1 either the identity result with
a non-zero host share, or a loud refusal — never a silent degradation; step 2
only after step 1 passes.

Required state and document updates.

- A dated `docs/currentstate.md` entry separating **feasibility** (does it run,
  is the output identical), **instrumentation** (are the counters and the
  in-flight lifecycle true) and **performance** (not measured here — say so).
- The step-A execution-shape decision recorded in this block with its reasoning,
  and an SDD amendment adding the class to section 4 with its constraint,
  capacity treatment and failure contract, before this task may reach `ready`.
- The owner-summary: goal, plain-language mechanism, glossary, pros/cons with
  every claim labeled measured / expected / unproven, and a recommendation.
- No build, trace, verdict or telemetry artifact into version control.

Escalate to the owner before proceeding if: step A concludes the mechanism falls
inside the SDD's cross-backend prohibition; AC2 fails; any hang or TDR appears;
the mechanism turns out to need the MIN_BATCH override, the scheduler rowcopy, or
a variable-width graph; a demanded row cannot be classified as READY, PENDING or
absent; a new dependency is needed; a flag would change an accepted trace's
meaning; or a performance arm is proposed before AC2 passes.

---

## Provenance of the shape (read-only reference, no dependency)

`C:/Development/AI/colibri` (DeepSeek-V4 baseline) runs this mechanism as
`V4_CUDA_TG_CPU_DIRECT`, default off. Its recorded properties, cited so the
differences are visible rather than assumed:

- **Trigger**: rows whose staging is storage-PENDING; the upload continues while
  the CPU computes them.
- **Shape**: the route plan is filtered (`expert_to_group` holds -1 for CPU
  experts) and CPU-direct chunks always launch the **host** route-plan variants.
- **Merge**: CPU partials accumulate into the same accumulator the device path
  consumes; a non-finite scan guards the merge.
- **Measured**: three-leg semantic QA, 2026-09-24, 30 and 512 tokens — 1,209 and
  5,125 CPU-direct experts, 0 fallbacks, 0 non-finite, generated text
  **byte-identical** to both all-GPU legs on both rungs, needles and Paris PASS,
  negative control FAIL as designed. `tg_stage_s` -61.7% (45-token) and -50.4%
  (4k) on a pinned binary. Owner disposition 2026-09-24: **NARROW** — opt-in
  lever, adoption re-gated; content risk measured-clear **at smoke tier only**
  (one leg per cell, 100-token answers, not the 24k ladder).
- **Defects it took to get there — all three are mandatory test cases here**: a
  12 KB stack smash from a merge buffer sized `float[4096]` where `d=7168` floats
  were written; the `route_gather` cursor[-1] atomicAdd hazard from launching the
  device route variant with a filtered plan; and merge math that dropped the
  shared-expert contribution. In our graph the shared expert is added to the
  same residual downstream of the add chain, so the split must not touch it.

**What does not transfer.** Their engine and quantization are not ours, their
byte-identical result is one model at two short lengths with one leg per cell,
and our own model already flipped a greedy token under a coarser version of the
same idea (T-GPLR-16: whole-layer flip, margin 16.30 -> 1.75 at TG step 1). The
additive structure rules out corruption on both sides; it does not rule out a
near-tie flipping. AC2 is therefore re-run here rather than inherited.

---

## Later evidence that bears on this block (2026-09-29, corrected 2026-09-30)

Two measurements taken after this draft was written change its cost gate and
leave its mechanism untouched. The first of them was itself wrong and has been
re-measured.

1. **The cost gate is no longer a heuristic — but the first numbers were
   contaminated and are withdrawn.** `tools/expert-cpu-bench` measures the real
   op chain on the real Q4_K bytes at the TG shape, 300 iterations. Two errors
   in the 2026-09-29 reading, found by audit on 2026-09-30:
   - **"453 MB working set (beyond any cache)" is false.** That run had
     `"pool_experts": 64`, so the working set was 64 x 1,769,472 =
     **113 MB**, not 453 MB. 452,984,832 is `bank_bytes` — the size of a *full*
     256-expert bank, of which the run populated a quarter. 113 MB is still
     beyond L3, so this error alone does not invalidate the run.
   - **The expert-id stride did.** Consecutive iterations overlapped in k-1 of
     k experts, so the marginal cost measured was a cache cost. Root cause,
     clean re-measurement and the surviving rule are in item 3 above.

   Corrected: the marginal host cost of one additional expert is **~36.5 us**
   (not ~8.7-9.8 us) against **214-241 us** to fetch the same row across PCIe at
   a real end-to-end rate. There is still no k at which fetching wins, so the
   rule is unchanged: "not resident -> host", not a tunable threshold. Item 3
   carries the corrected table and the withdrawn one side by side.
2. **The row cache this block would feed is not worth building** — *as a
   transfer-avoidance device*. `docs/tg-expert-path-reconciliation.md`
   reconciles the stock CPU-expert baseline against our row-set ceiling and
   finds the ceiling at or below that baseline across every link rate ever
   measured on this box. That conclusion **survives the correction and is now
   unconditional**: fitting the A/B arms' wall time against their own staged
   bytes gives `ms = 28.92 + bytes/10.35 GB/s`, so the streaming path's
   *non-transfer* floor of 28.92 ms/token already exceeds the incumbent's whole
   24.41 ms token, and it loses at `h = 1` (reconciliation §15.1). The
   break-even-retention argument in item 8 below is superseded — it was
   arguing about a term that was not binding. The cache is the optional part of
   this design and it is the part that does not pay *when its job is to avoid
   transfers*. **The mechanism is unaffected:** with the row cache off the split
   IS the stock arrangement; with it on, it is a strict improvement on the stock
   arrangement, because under this block a cache hit avoids **host compute**,
   not a transfer — and that trade is 36.5 us of CPU against ~14.8 us of GPU,
   which wins at every hit rate above zero. What changes is priority: the
   mechanism is now the only part of the CPU-expert line with a measured
   constant behind it.

---

## 8. Where the win actually is (derived 2026-09-30 from the corrected constants)

All three questions in this domain reduce to the same comparison, and the
corrected numbers answer all three. Constants used: host expert FFN
**36.5 us/expert** (clean, arm C), row bytes **1,769,472**, **marginal**
end-to-end feed **10.35 GB/s** = 171 us/expert, 40 layers, top_k=8, measured
retention **h = 0.7255 (K=8) / 0.8319 (K=12)**, stock TG anchor **40.96 tok/s =
24.41 ms/token**.

*(Constants corrected 2026-09-30. This section previously used 8.26 GB/s and
"best measured retention h = 0.6487 (K=8)". The retention figure is rung
**6**'s `honest_b`, mislabelled; the feed figure is an instrumented whole-bank
copy, not the marginal cost of a byte in a real token. See reconciliation §14
and §15.2.)*

**Q1. Is it ever right to transfer a non-resident expert to the GPU?**
Per expert: host 36.5 us vs link 171 us. The link is **4.7x** more expensive.
**No, at any k. CLOSED** — this is item 3's rule, now confirmed on clean data.
*(Was stated as 5.9x against the 8.26 GB/s figure; the faster measured feed
narrows the margin and does not change the answer.)*

**Q2. Can the pure row-set streaming path beat stock?**
**No, at any hit rate including `h = 1`. CLOSED, unconditionally.** Fitting
ms/token against staged bytes/token over the 8 matched T-GPLR-19 A/B arms —
identical routes, identical peak VRAM, 3 byte levels — gives

    ms_per_token = 28.92 + staged_bytes / 10.35 GB/s     (residuals within 1.9%)

The intercept is the path's cost with **zero bytes on the link**: 28.92 ms
against the incumbent's whole 24.41 ms token, i.e. **34.58 tok/s vs 40.96 =
0.84x at a perfect cache**. Measured at K=8 the path reaches 24.34-24.62 tok/s,
1.68x behind.

*(SUPERSEDED 2026-09-30. This answer previously read: "per layer the streaming
path pays `(1-h) x 8 x 214 us` in transfer where the host would have paid
`8 x 36.5 = 288.6 us`; break-even is `h > 0.832`; measured best is 0.6487 and
K=6 -> K=8 moved it by 0.0026 — an 18-point gap that tuning does not close",
followed by a model that reproduced the measured throughput "within 13%". Three
problems: 0.6487 is K=6's `honest_b`, not K=8's; the 0.0026 was that same run
differenced against its own residency rate rather than a step between rungs; and
the break-even framing asked whether `h` could get high enough when no `h` is
high enough. The direct fit replaces the model entirely and agrees to 1.9%.)*

**Q3. Does the per-expert split (this block) pay?**
This block **removes the link from the equation** — non-resident computes on the
host, it is never fetched — so the comparison is host compute against device
compute and any hit rate above zero is profit. Device expert FFN is estimated at
**~14.8 us/expert** by differencing the streaming path's non-transfer remainder
against stock's non-expert budget; that estimate is **loose**, and it sets the
overlapped optimum `h* = c/(c+d)` (reconciliation §13.4), so it is worth
measuring as a **setpoint** — but it is no longer a gate on whether to build.

| arrangement | expert cost / layer | tok/s | vs stock |
| --- | --- | --- | --- |
| all-host (what the host pays alone) | 285.6 us | — | — |
| stock static `-ncmoe` split, h=0.188 | 254.2 us | 40.96 (anchor) | 1.00x |
| this block, h=0.646, **serial** | 177.4 us | 46.8 | **1.14x** |
| this block, h=0.646, **overlapped** | 100.8 us | 54.6 | **1.33x** |

**CORRECTED 2026-09-30.** The first version of this table read `~49.9 / ~1.22x`
serial and `~53.9-59.0 / ~1.32-1.44x` overlapped. Those tok/s were computed
from an **all-host** decomposition (`other = 24.41 - 40x288.6/1000 = 12.87 ms`)
while the row immediately above them was labelled `h~0.25`. The two are
mutually exclusive: if the 40.96 tok/s anchor already has banks resident, the
non-expert remainder is `24.41 - 40x254.2/1000 = 14.25 ms`, not 12.87 ms, and
every derived tok/s in the column was overstated. The serial case was the one
that mattered — it falls from 1.22x to **1.14x**. The overlapped case is barely
affected (1.33x, and the 1.44x upper end was pure artefact). Arithmetic is
reproducible: `tools/placement-model.py`, `tools/gain-decomp.py`.

**CORRECTED AGAIN 2026-09-30** (second pass). The `h=0.225` this table carried
for stock was itself a guess — "nine banks resident", inferred from an assumed
1,302 MiB of dense weights. The anchor ran at `-lv 4` and its fitter log states
the dense-only device size outright (1,921 MiB), giving **7.5 banks and
h=0.188**. See reconciliation §12.1b and `tools/hstock-from-log.py`. Stock is
holding *less* on the device than this plan credited it with, so `other` drops
to 14.25 ms and every multiplier above rises ~1.3%.

**The unmeasured device cost matters far less than this section claimed.** Under
overlap the per-layer cost is `max(h*8*d, (1-h)*8*c)`, and at h=0.646 the host
leg is 100.8 us while the device leg is `0.646*8*d`. The host leg is the
maximum for any `d < 19.6 us`, so **the expert leg of the overlapped result is
exactly independent of d over that whole range** — the small residual movement
below is only `other` being re-solved at each `d`. Sweeping d from 10 to 30 us
moves the overlapped prize **1.25x-1.35x** (peaking at d=20, where h* = 0.641
sits almost exactly on this policy's 0.646), against **1.18x-1.04x** for
serial. Measuring d is therefore a prerequisite for the *serial* variant and a
robustness check, not a gate, for the *overlapped* one — which reverses the
"single number this block should measure first" framing above. The sweep is the
last block of `tools/placement-model.py`.

**The structural argument for a dynamic split, which is the one that matters.**
Stock's `-ncmoe` split is per **layer**: a GPU-resident layer needs its whole
256-expert bank in VRAM (452.98 MB), so the 3,271 MiB the fitter spends on
expert weights buys only **7.5 of 40 layers, h_static = 0.188**. A row cache
stores only **demanded** rows — 8 of 256 = 3.1% of a bank — and at 48
rows/layer it occupies **3,240 MiB, i.e. 31 MiB *less* than stock**, while
reaching h = 0.634. **Fewer bytes, 3.4x the hit rate.** That is why the dynamic
split can beat the static one, and it is an argument about granularity, not
about tuning.

**The byte comparison is anchor-specific; the granularity argument is not.**
Stock's residency is *leftover* VRAM, so it shrinks as KV grows: at the settled
236,000-token context on the same 7,055 MiB budget it falls to 2,197 MiB
(h = 0.126), and the 48-row cache then costs 1,043 MiB **more** than stock, not
less. The hit-rate ratio moves the other way — 0.634 against 0.126 is **5.0x**,
not 3.4x. So "fewer bytes" holds only at the anchor; "far more hit rate per
byte" holds everywhere, and is the claim to lean on. See
`docs/tg-expert-path-reconciliation.md` §13.6 and `tools/ctx-budget.py`, which
also finds that **48 rows is the only capacity that reaches 236k at all**, and
only on a card not also driving a display.

**But granularity is not the same as dynamism, and the difference was measured
2026-09-30.** See `docs/tg-expert-path-reconciliation.md` §13. Pinning the
*statically hottest* 56 experts per layer — same granularity, same bytes, no
runtime machinery at all — reaches only h = 0.375 on held-out prompts, because
the hot set is a property of the prompt and not of the model. Freezing the set
once after a 128-token warm-up in the same session reaches h = 0.556. The full
per-token cache reaches 0.646. So of the 0.188 -> 0.646 gain, **granularity
alone is the largest single share (41%) and per-session adaptation is close
behind (39%)**; the per-token eviction machinery is worth the last 0.090
(20%), which is 1.33x against 1.26x end to end. That is the honest price of the
T-GPLR-62/63/64 apparatus.

> **CORRECTED 2026-09-30.** The three hit rates above previously read 0.337 /
> 0.565 / 0.647. Re-derived at 56 rows from the raw traces
> (`gplr19-honest-k6-ab/arm-B1.jsonl` scored, `gplr19-k6-ab/arm-B1.jsonl`
> ranking the held-out leg) they are **0.3753 / 0.5559 / 0.6461** — the first
> two did not reproduce and look like figures carried over from a 48-row run.
> The 1.25x and 1.32x conclusions are unaffected; the share attributed to
> granularity rises from "about a quarter" to 36%. `tools/gain-decomp.py`.
>
> Note this paragraph scored **one prompt**, which is why it escaped the
> averaging error corrected in §13.4 of the reconciliation doc and reached the
> right conclusion about the per-token cache. On prompt B every stage stays
> below `h*` = 0.7069, so more hit rate is strictly better here and the
> overshoot regime that reorders policies at cap 64 never engages.

**What this means for priority.** The prize is **~1.13x serial / ~1.32x
overlapped**, not a step change, and the overlap is the part that carries it.
Revised order of work: (a) establish the **best achievable stock configuration**
first, since `-ncmoe`/`-ngl`/fit-target tuning costs no engineering and moves
the same h_static this block competes against; (b) build the split op with a
**tunable device share `alpha`** — that is the piece that carries the whole
gain, and it needs no cache, no eviction, no predictor and no per-token
staging; (c) measure the device expert FFN cost `d` from that build, where it is
a direct observable rather than a differenced estimate; (d) choose the placement
policy with `d` in hand. On the corrected numbers the per-token cache is the
leading candidate rather than the optional extra it was called here — see §13.4
of the reconciliation doc, where LRU-K beats warm-then-pin at both capacities —
but that ordering is conditional on `d` and should not be fixed before (c).
Building the full cache first would still be wrong: it would spend the
project's remaining budget before the term that decides the ranking is known.
