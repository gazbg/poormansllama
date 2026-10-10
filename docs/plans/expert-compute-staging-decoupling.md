# T-GPLR-82 (ready): decouple expert compute placement from staging — host-compute a missed expert without cancelling its copy

**Status: READY (owner-directed 2026-10-03).**
**Dependencies:** T-GPLR-61 step A (`src/llama-expert-split-policy.h`, AC1 green) — supplies
`ls_classify_row` / `ls_evaluate_gate` / `ls_plan_split`, which are **compiled into nothing
today**. T-GPLR-19 (`--residency-tg-retain`) supplies the cache this populates.
**Commit:** _(unfilled — no commit authorised)_

This task is deliberately **not** filed inside
`docs/plans/generic-prefill-layer-residency-plan.md`, which is 479,789 B against a
480,000 B FAIL band and has **211 B of headroom**. It is a separate workstream with
its own acceptance model.

---

## Owner directives this task implements

Three, recorded verbatim in intent, 2026-10-03:

1. **No preliminary probes.** Do not run numerics or contention experiments first.
   Implement the mechanism, then test how it behaves. The risk of discovering late
   that the host leg does not pay is **accepted by the owner** and recorded here so
   it reads as a decision, not an oversight.
2. **Byte-identity is not the acceptance model.** The planned
   "flag absent → trace byte-identical to control" check is withdrawn. Acceptance is
   **semantic QA**.
3. **Explore the running CPU/GPU hybrid regardless of what any SDD says.** This is
   exploratory work; SDD conformance is not a gate on it.

## Goal

Make a routed expert row that misses the device row cache **compute on the CPU while its
copy to the GPU continues in the background**, so the cache stays warm for the next token
instead of trading a miss now for a colder cache later.

## Why this is the shape

The staging source is the host **mmap of the model file** — never consumed, always fully
present (`bytes_full_bank` is identical on round 49 and round 176 of the `tg82` run). Weights
are immutable. A CPU reader and a DMA reader on the same bytes is two reads, not a conflict.

| per demanded row that MISSES | cost | source |
|---|---|---|
| stage over link, then compute on GPU | **185.8 µs** | 171 link (10.35 GB/s) + 14.8 device (differenced, not measured) |
| compute on host, in place | **59.4 µs** | measured in situ, ab18b |
| host advantage | **3.13×** | |

The asymmetry is the argument: a *speculative* stage costs PCIe time that can overlap 59.4 µs
of host work; a *deferred* miss costs the same PCIe time **and blocks the token**.

## Scope

Token generation on the row-set path. The change is in
`src/llama-layer-streaming.cpp`, function **`load_row_set(int il, const std::vector<int32_t> & ids)`**
at **`:9977`**, whose two return paths are:

| branch | line | condition | behaviour today |
|---|---|---|---|
| cache-served | `:10516` | `fetch` empty | returns **without staging** |
| staging | `:10656` | rows missing | stages the complement, copy **already not cancelled** |

**The staging branch already does the right thing on staging.** Nothing is added to it.
The entire mechanism is: in that branch, build a host sub-op for the missed rows bound to the
in-flight source rows, and do **not** make the consumer wait for those rows' readiness.

## Non-goals

- Not a prefill change. `load_row_set` is shared; prefill behaviour must remain unchanged and is
  gated by AC5.
- Not a KV, recurrent-state, or slot-lifecycle change.
- Not a redefinition of `K` / `--residency-tg-retain` depth.
- Not a predictor or speculative-guess mechanism. `--residency-tg-spec` stays off; the predictor
  line is closed on arithmetic (reconciliation §13.5).
- **Not** a performance claim. The first deliverable is that it runs and is semantically correct.

## Design and failure contract

**The existing signal, and what it is not.** `ffn_dev` on `layer_compute` is computed at
**`:4942`** by `ls_classify_graph_backend(set->ff_graphs[il])` — *after* the graph is built. It is
an **observation of where the backend landed, not a decision**, which is why it reads `1` on all
5,120 records of the `tg82` run. This task's mechanism is the decision that has to happen
*before* that build. `ffn_dev` therefore remains the independent corroboration that a host sub-op
actually executed, and S1 must keep emitting it — AC2's `experts_host` counter says what was
*intended*; `ffn_dev == 0` is what proves it *happened*.

**Decision rule**, evaluated per demanded row at demand time:

```
READY   (resident, generation matches) -> DEVICE
PENDING (copy in flight)               -> WAIT, then DEVICE   (never cancel)
ABSENT  (nothing in flight)            -> HOST if --experts-host B allows, else WAIT->DEVICE
```

Staging is **unconditional** for anything that misses; there is no `--stage-policy` flag,
because `load_row_set` already stages everything not served from cache.

**Budget.** `--experts-host-budget B` in µs of host expert work per token. `B=0` is today's
behaviour; `B=∞` is host-first. B exists because host-first is not obviously optimal — the host
leg is bandwidth-bound (1,769,472 B ÷ 59.4 µs ≈ 29.8 GB/s implied) and the DMA reads the same
pages, so success can be self-defeating.

**Eviction interaction — a trap, not a note.** `llama_ls_spec_victim_better`
(`:324-333`) makes any entry labelled `spec` the **first eviction victim**, and rows are
stamped so that *"a speculative row loses every tie against a demanded one"* (`:8863-8865`).
Rows this task stages are **demanded now**, so they must be stamped **demanded**. Getting this
wrong silently discards the entire benefit on the next LRU sweep. It is guarded, not commented.

**Failure contract.** A missing package, stale slot generation, failed CUDA event, or a
non-finite host result **aborts the run loudly** with structured telemetry. It must never
silently compute another layer's data, and it must never silently fall back to waiting.

## Implementation steps

**S1 — Instrumentation (part of the mechanism, not a preliminary test).**
Make the decision observable. All default-off:
- `experts_host` / `experts_device` per (layer, token) on `layer_compute`; invariant
  `experts_device + experts_host == experts_total`.
- `ready_at_consumption` on **TG** `rowset_fetch` — the field currently exists only on PP
  `layer_ready_wait`, so TG readiness is unmeasured today.
- per-token `t_us` on `rowset_round` — the 2,662.864 ms outlier token of `tg82` currently
  cannot be placed in the timeline (round gaps cover only 7.249 s of a 10.680 s wall).
- `host_us` **actual**, not estimated — closes the 36.5 (microbench) vs 59.4 (in situ) gap.

**S2 — Host sub-op.** Build the CPU-backend sub-op against the existing add-chain reduction,
with weights bound to the in-flight source rows on the host. **No weight source may ever be
rewired** — SDD section 4 forbids it and the guard must assert it specifically. Extend the
T-GPLR-16 guard to also assert the MIN_BATCH override and the scheduler row-copy are
unreachable. New flag `--experts-host N`, default `0` = off, 0..1 or a per-layer count.

**S3 — The decoupling.** In the `load_row_set` staging branch (`:10620`–`:10656`), when
`--experts-host` is non-zero and the gate is open, mark the missed rows host-bound and let the
consumer build its sub-op against the host copy without awaiting readiness. The in-flight copy
**keeps running and retires normally** — it must not be cancelled or consumed.

**S4 — Budget.** `--experts-host-budget B`. Gate uses `ls_evaluate_gate` with the measured
constants; `ls_plan_split` supplies the constant-width sub-op descriptors.

**S5 — Validate** (the testing the owner asked to do *after* implementation):
semantic QA ladder at 50 and 512; host suite green; the 512-anchor arm with `--experts-host`
forced on; and an inertness arm with the flag absent.

**S6 — 8 GB shape,** when an arm is authorised: **2 slots / K=2** (7,961.1 MiB, fits with
230.9 MiB spare) and **1 slot / K=3**. **Not 1 slot / K=4** — that totals 8,555.7 MiB and
refuses by 363.7. (The existing plan's ladder column is wrong: it measures headroom as
`budget − bounded` and omits the act slab and reserve.)

## Acceptance criteria — semantic, not byte-identity

- **AC1 (correctness, the real gate).** Semantic QA aggregate **PASS** at 50 and 512 tokens on
  the 512 anchor, `check-telemetry` OK, negative control FAILing as designed. Additionally
  top-1 agreement **≥ 99%** against the recorded control over ≥ 512 tokens, and **every**
  divergence explained as a near-tie (top-2 logit gap below the measured drift). Unexplained
  divergence fails.
- **AC2 (lifecycle).** `experts_device + experts_host == experts_total` at every
  (layer, token). In-flight copies complete and retire normally. Slot, generation,
  copy-event and consumer-done lifecycles unchanged — verified against the trace, not asserted.
- **AC3 (no hang).** The arm completes inside the recorded wall band with no TDR and no
  `cudaErrorLaunchTimeout`. A hang fails even if AC1 passed.
- **AC4 (inertness — semantic form, replacing byte-identity).** With the flag absent,
  `experts_host == 0` for the whole run. This is an **observation reported as such**, never a
  pass with an implied benefit. *Note the gap this leaves: a semantic test cannot detect a
  change in inert numerics, only inert execution. If that matters, it needs a separate check.*
- **AC5 (no default change).** Flag absent: dense regression unchanged, host suite green,
  prefill path byte-for-byte unchanged in **phase** behaviour (`load_row_set` is shared, so
  prefill is explicitly re-checked here even though TG is the target).
- **AC6 (guards).** The extended T-GPLR-16 guard passes, including the assertion that **no
  weight source on this path is ever rewired**.
- **AC7 (the falsifier the owner reads first).** The measured `experts_host / experts_total`
  share and end-to-end tok/s against the `tg82` control of **19.31 tok/s** (round-to-round
  median; **not** the 60.99 tok/s compute-only figure, which must never be quoted as decode).
  A host share of 0 is a valid outcome and is reported, not tuned away.

## Focused validation commands

```powershell
# host-only, no GPU, no mutex
cmake --build build/llama-cuda --config Release --target test-layer-streaming-expert-split
ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-expert-split --output-on-failure
powershell -NoProfile -File tools/run-tests.ps1 -Suite host
python tools/check-doc-sizes.py

# GPU arms: mutex required. Never setsid. Dot-source arm-verdict.ps1 and
# Clear-ArmVerdict the tag BEFORE Start-Process.
powershell -NoProfile -File tools/with-gpu-lock.ps1 `
  -Command "powershell -NoProfile -File <launcher>.ps1" `
  -VerdictTag <tag> -AllowUnprovenanced
```

Expected: host suite green with the new test included; doc guard PASS; every GPU arm exits 0
with `gpu_pre`/`gpu_post` clean and a verdict file written.

## Required state and document updates

- `docs/currentstate.md` — observed behaviour, telemetry path, the falsifier result.
- The **eviction-stamp rule** (demanded, never `spec`) belongs in the SDD `moe-row-split`
  section, or in this plan if the SDD is not being edited.
- Record the ladder correction (1 slot → K=3, not K=4) somewhere that is **not** the 480,000 B
  plan file.

## Escalation condition

Stop and return to the owner on: a hang or TDR; a lifecycle violation (stale generation,
cancelled copy, rewired weight source); semantic QA AC1 failure; or `experts_host == 0` for a
whole forced-on arm (the mechanism did not fire — that is AC7's inert outcome, not a tuning
opportunity). **Do not** tune placement, budget, or `K` to rescue a failing arm before the
falsifier is recorded.

## Known risk, accepted by the owner

The numerics drift ceiling and the DDR-contention question are both **unmeasured**. The owner
directed that preliminary probes be skipped (2026-10-03) and the effort be made. If either
turns out badly, that is a finding to report, not a reason to have gated the work.
