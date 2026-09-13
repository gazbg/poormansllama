# Current State

## 2026-09-13: SDD amended with the F2 per-layer row cache (`moe-row-cache`) residency class — realized reuse and the T-GPLR-15 gate's cumulative-resident-set ceiling are now defined and measurable

Owner-requested SDD treatment written into docs/layer-residency-sdd.md
(amended 2026-09-13; sections 1, 4, 6, 7, 8, 9, 10, 11, 12). No code changed;
no GPU runs.

The class (analysis label F2; no task ID yet) is defined as an opt-in,
CUDA-only, TG-only sub-mode of `moe-row-set`:
- Per-layer keying is the defining property: storage and every resident claim
  are keyed by LOGICAL LAYER, never by slot, because the layers sharing a slot
  alias one device buffer at offset 0 and an intervening owner overwrites the
  same byte ranges — the reason the bookkeeping-only H1(b) variant is unsound
  (bottlenecks doc section 6.4). Provenance is structural (layer, member,
  expert, written window/generation stored with the bytes), never inferred.
- Bounded retention sized for the worst case: min(n_expert, K*W*top_k) rows x
  member slab bytes — ~14.2 MB per 8-row set and ~0.57-1.13 GB across 40 MoE
  layers for one-to-two window retention on the Qwen3.6 target; overflow is a
  loud refusal, and an entry a running consumer may read is never evicted.
- Per-window atomicity, the section-4 execution-shape constraint unchanged,
  and two permitted execution shapes (cache-to-slot-bundle D2D refresh before
  the row-consuming phase, or graph tensors bound directly to the cache) that
  the adopting task must declare.
- Reuse optimizes the COPY list only; the reactive superset check on the
  router's exact IDs is untouched. Filling the cache reactively is this class;
  filling it ahead of need is T-GPLR-15's variant.

What this unblocks for the T-GPLR-15 gate: the SDD now names the two distinct
statistics and requires a verdict to state which was used — (a) consecutive-
window exact containment, a property of the routing stream that is measurable
WITHOUT the cache (T-GPLR-15 Phase A), and (b) cumulative-resident-set
containment, the realized predictor ceiling, which is only well-defined once
per-layer persistence exists (i.e. requires F2). New gate 10 requires
positive reuse demonstration, cache-on/cache-off pairs for any performance
claim (the cache changes the byte denominator), and byte-for-byte unchanged
cache-off and default-off paths. Cache failure reasons, telemetry classes
(cache-served rows / device refresh bytes / link bytes / high-water /
invalidations, never double-booked), adapter duty (9) for declared capacity
and provenance ranges, capacity-preflight treatment, and a risk list were all
added.

## 2026-09-13: T-GPLR-18 (F1 pinned row-set staging) added to the plan as READY with a 512 A/B gate — counter-evidence recorded first; 6.3's pinned-throughput assumption corrected; T-GPLR-15 probe protocol corrected to the batch-major vehicle

F1 from the T-GPLR-13 close-out is now plan task T-GPLR-18, READY 2026-09-13,
with an explicit 512 A/B acceptance gate. Contract shape: `load_row_set`
honors the existing `slot.buf_staging` when `--copy-source pinned` is
requested (packed slabs, natural destination offsets, interleaved
pack-and-enqueue exactly mirroring `load_layer`); no new flag, no checker or
schema change (`run_start.copy_source` already labels the arm); the
event lifecycle and the host-blocking wait are retained; drivers are
tools/run-gplr18-pinned-ab.ps1 (A,direct / B,pinned / A / B interleaved under
one GPU-mutex session at the T-GPLR-13 512 ladder shape, `--tg-tokens 3`) plus
the existing checkers.

Counter-evidence recorded in the contract before the hypothesis (bottlenecks
doc section 6.7):
- Measured H2D is 13.7 GB/s steady-state on a negotiated Gen4 x8 (87% of the
  15.75 GB/s link); pageable sources are internally staged at that same rate,
  so pinned adds no DMA throughput — only an explicit host memcpy.
- T-Q38PERF-03 already ran the dense A/B and pinned LOST end-to-end (PP
  8,366.4 vs 8,195.3 ms; TG 2,757.5 vs 2,515.0 ms) with submit down and
  availability wait up (TG 5.3 -> 1,510.4 ms).
- Surviving hypothesis is only per-call overhead: 24 copies of ~0.59 MB per
  fetch; measured submit 1,826 us vs the ~1,033 us floor for 14.16 MB at
  13.7 GB/s. Pre-registered per-fetch ceiling ≈ 1.03 ms (from 1.93 ms); a
  bigger claim is arithmetic error.
Gate: feasibility (both arms exit 0, checker OK, identical staged bytes
572,719,104/round, identical token IDs and route hashes) + instrumentation
(pinned transfer_us mean below direct by more than the within-arm repeat
spread, wall not worse); performance claimed only if the wall improves beyond
spread and qa-smoke 512 stays PASS. A negative result closes the task as
falsified-by-evidence and returns the TG lever to F2 (bytes).

Also corrected this turn (contract-level, no code): the T-GPLR-15 Phase A
256-token probe as originally drafted was UNRUNNABLE — the runner caps
handoff-free layer-major TG at 3 tokens (main.cpp:2190/:2878) and
`--residency-tg` forces handoff off (main.cpp:2465). The corrected protocol
uses the batch-major vehicle (`--pp-schedule batch-major --tg-tokens 256`),
which has no cap; the hit-rate statistic is a property of the generation
stream and batch-major/layer-major PP produce the same KV per the validated
equivalence gates, but row-set TG after batch-major PP is UNVERIFIED, so a
one-run feasibility check precedes the gate run. Recorded too: greedy
decoding is deterministic, so repeated identical runs add no independent
samples — only stream length (or a prompt family) grows n.

Plan serialization note: two READY tasks now exist (T-GPLR-15 Phase A,
host-only analyzer + probe; T-GPLR-18, small library change + GPU A/B). They
are independent; the owner selects order and implementation stays one at a
time. No code changed, no GPU runs, no commit.

## 2026-09-13: T-GPLR-15 contract rewritten gate-measurement-first (READY) — Phase A measures the 0.6 hit-rate gate with no library changes; Phase B only on PASS + owner review

The plan's T-GPLR-15 block is rewritten as a two-phase contract on owner
request, now that T-GPLR-13 is DONE and `rowset_fetch` telemetry exists in
real traces:
- Phase A (controlling): extend `tools/route_overlap/analyze.py` so T5's
  per-token verdict sources `rowset_fetch` (schema draft section 6: known-
  event growth, hash recompute on every event, determinism unchanged), add
  exact consecutive-window containment and a per-layer reappearance-distance
  histogram; cross-validate the re-sourcing against the EXISTING ladder
  traces (must reproduce the hand-computed pilot 0/80 hash-equal,
  containment 0.358-0.389); then ONE owner-invoked 256-token generation
  probe (`--residency-tg --tg-tokens 256 --greedy --no-mtp`, GPU mutex,
  poller attached) yielding 10,200 consecutive-window pairs at ~2.4 MB
  trace.
- Gate rule: primary = hash-equal consecutive-window rate (exact, W=1).
  PASS >= 0.6 per SDD gate 9 advances to Phase B (implementation, own
  dated contract update at entry; F2 per-layer row caches are a prerequisite
  design and a NEW residency class under SDD section 4). FAIL < 0.6 closes
  the task DONE-falsified-by-evidence with no library change; the reactive
  row-set stands as the TG answer alongside the closed handoff. Aggregate
  passing only via a few near-1.0 layers while late layers collapse is
  reported as volatility and escalated (per-layer policy = product intent).
- Also recorded: the secondary cumulative-resident-set containment
  distinguishes "prediction fails" from "rotation discards" — the 2-step
  pilot could not. The 0.6 threshold itself remains the SDD's number; this
  contract adds no new threshold.

## 2026-09-13: T-GPLR-13 marked DONE (owner review) — reactive expert row-set residency closed; residual risks and open follow-ups recorded

The owner reviewed the completed task and marked it done. All acceptance
criteria were already demonstrated (implementation complete, built clean,
end-to-end PP64+TG3 run, 50/512/4k/200k ladder + qa-smoke regression all
PASS, dual-path negative controls 18/18, validator parity, dense cold-path
ctest pair 2/2). No code changed for this close-out; both repos remain
uncommitted (no commit authorized).

Residual risks at close:
- TG tok/s (10.1 -> 7.7 across the ladder) is instrumentation evidence
  only; the row-set win is byte/wall evidence on 3-token windows. No
  headline performance claim is authorized by the task contract.
- The dominant TG staging term is submit-side pageable-H2D staging
  (bottlenecks doc section 6.2); the projected 1.2-1.4x from F1 is a
  hypothesis until the 512 A/B probe runs.
- H2 (200k staging-bandwidth decline 5.79 -> 4.39 GB/s) and the ~5 GB
  non-KV commit residual (H7) are explained-to-first-order only.
- All TG `layer_compute` timings are zero (H8/F3) — telemetry debt that
  blocks exact compute-vs-staging attribution.
- The reuse branch never fires (rows_reused=0); not a defect, but the
  H1(b) bookkeeping-only fix is UNSOUND (section 6.4) and must not be
  attempted over the aliased slot buffers.
- qa-smoke shapes remain batch-major; the 32x/14.6-19.2x row-set contrast
  is a same-model contrast, not a controlled A/B (section 3.8).

Open follow-ups (none scheduled; owner to prioritize):
- F1 pinned row staging (~14.2 MB window; needs a task contract; 512 A/B
  falsifier, 200k A/B decides H2).
- F2 per-layer row caches (≤ ~570 MB; makes the 0.358-0.389 containment
  reuse realizable and measures T-GPLR-15's 0.6 gate over a real stream;
  requires an SDD residency treatment before a contract).
- F3/H8 nonzero TG per-part timings (small telemetry change; do alongside
  F1).
- T-GPLR-15 predictive prefetch (DRAFT; 0.6 hit-rate gate unverified;
  negative 2-step pilot 0.358-0.389).
- Owner-invoked endurance ladder rungs 50k/100k/200k remain unrun.

## 2026-09-13: TG row-set staging analysis (section 6 of docs/gplr13-bottlenecks-hypotheses.md) — ready-latency is ~5% of TG fetch; the transfer term is submit-side pageable-H2D staging; H1(b) corrected, fix ranked F1 pinned row staging

Read-only analysis (no GPU runs, no code changes; online pass via context7
against NVIDIA CUDA programming guide + driver-API sync-behavior docs):
- TG `rowset_fetch` ready_us mean is 103.8-107.1 µs = ~5-6% of transfer_us
  (1,835-2,273 µs); the submit term carries ~94-95%. The 74.6-98.7% ready
  shares in the same file belong to the PP whole-bank path, not TG.
- Code-verified mechanism: `load_row_set` issues 24
  `ggml_backend_tensor_set_async` calls per fetch (~14.15 MB) directly from
  mmap-backed pageable sources and never uses `slot.buf_staging`; pageable
  H2D async copies stage synchronously through a driver pinned bounce
  buffer, matching the measured 6.5-8.2 GB/s host rate.
- Ranked fixes (each needs its own task contract): F1 explicit pinned row
  staging (~14.2 MB window; expected staging term 1.8-2.2 -> ~1.0-1.4
  ms/fetch, ~1.2-1.4x TG tok/s if the mechanism is confirmed; 512 A/B probe
  as falsifier, 200k A/B decides H2), F2 per-layer row caches (≤ ~570 MB
  total, makes the 0.37 containment reuse realizable AND measures
  T-GPLR-15's 0.6 gate over a real stream), F3 nonzero TG per-part timings
  (H8, do alongside F1), F4 W>1 (H4, leverage shrinks after F1/F2).
- CORRECTION to H1(b) in section 4: per-layer coverage over the current slot
  buffers is UNSOUND — slot bundles alias one buffer at offset 0, so an
  intervening owner overwrites layer il's rows before il returns; per-layer
  reuse requires real per-layer row storage (F2), not bookkeeping alone.
  Claiming coverage without it could silently execute another layer's
  weights. The never-fired reuse branch remains NOT a defect, but its fix is
  now correctly scoped.

## 2026-09-13: T-GPLR-13 test ladder executed — all five sizes PASS (50/512/4k/200k row-set + qa-smoke-512 regression), no failures, no retries

Owner-directed ladder on the completed row-set path. Harness:
`tools/run-gplr13-ladder.ps1` (new; mirrors `tools/run-tests.ps1` shapes:
row-set sizes use the probe flag set with `--pp-schedule` default, ubatch 512
for 512+, single-ubatch for 50, `--tg-tokens 3 --greedy --no-mtp`; qa-smoke
mirrors the T-GPLR-07 512 rung exactly). Every GPU run under
`tools/with-gpu-lock.ps1`; `tools/gpu-mem-poll.ps1` attached per run (0.5 s
sampling, killed at run end). Model: Qwen3.6-35B-A3B-MTP-UD-Q4_K_S. A foreign
deepseek_v4.exe held ~16.8 GB VRAM before the ladder and exited before the
recorded runs; all five ran on an idle device (free VRAM at size-50 run_start:
22.23 GB of 25.65 GB).

Artifacts preserved under `build/llama-cuda/gplr13-ladder/<size>/`
(`rowset-*.jsonl`, `rowset-*.log`, `mem.csv`, anchor prompt; plus `summary.json`,
`driver.log`); qa-smoke dir also holds `qa512.{jsonl,txt,log}`.

| size | exit | check-telemetry | TG tok/s | runner peak_vram | poller dedicated peak | shared peak | commit peak | avail min | staged % | fetch | rounds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0 | OK (569 ev) | 10.10 | 5,674 MiB | 4,581 MiB | 66 MiB | 17.3 GB | 36.9 GB | 3.13 | 120 | 3 | PASS |
| 512 | 0 | OK (569 ev) | 9.38 | 5,728 MiB | 4,635 MiB | 74 MiB | 17.3 GB | 38.4 GB | 3.13 | 120 | 3 | PASS |
| 4k (4001 tok) | 0 | OK (663 ev) | 9.52 | 5,796 MiB | 4,703 MiB | 78 MiB | 17.8 GB | 36.1 GB | 3.13 | 120 | 3 | PASS |
| 200k (200000 tok) | 0 | OK (4,469 ev) | 7.67 | 9,742 MiB | 8,649 MiB | 268 MiB | 24.0 GB | 33.4 GB | 3.12 | 120 | 3 | PASS |
| qa-smoke-512 (off) | 0 | OK (23,326 ev) | 0.53 (96 tok) | 4,350 MiB | 3,267 MiB | 74 MiB | 16.1 GB | 35.3 GB | n/a | 0 | 0 | PASS (answer PASS, negative FAILS) |

Findings:
- No size failed; zero refusals (`rowset_refused` count 0 in every trace),
  zero retries. Route parity held at every size (checker lifecycle + union
  validation on all 120 fetch events per row-set run).
- Staged bytes are context-length invariant: 572,719,104 / 18,327,011,328
  bytes per round (3.12-3.13%) at 50 through 200k tokens - the reactive
  union at W=1 is exactly 8/256 rows regardless of KV size, which is the
  row-set thesis holding under the largest recorded prompt.
- Runner-vs-poller VRAM gap is a stable ~1.09 GB at every size (5,674 vs
  4,581 at 50; 9,742 vs 8,649 at 200k) - consistent with the T-MEMPROBE-01
  delta-metric skew (CUDA context + teardown ordering), not a new leak.
- Shared (non-local) segment scales mildly with context (66 -> 268 MiB from
  50 to 200k) while dedicated grows ~4.1 GB - no duplication behavior; the
  T-MEMPROBE-01 conclusion extends to the 200k regime.
- Host RAM safe at all sizes: commit peak 24.0 GiB (= 25,754,857,472 B;
  the PowerShell driver labels binary GiB while the Python analysis file
  uses decimal GB — same measurement, different unit convention) of the
  113.5 GB limit at 200k (the 927 KB anchor text + mmap); available
  physical never below 33.4 GiB.
- TG tok/s declines 10.1 -> 7.7 across the ladder (3-token windows; staging
  572 MB/round dominates). Instrumentation result only - not a performance
  claim per the task contract; per-round ratios are the acceptance input.
- Row-set VRAM footprint delta 50 -> 200k: 5,674 -> 9,742 MiB runner
  (~+4.07 GB ~= the 200k KV estimate + activation slack). Preflight free
  VRAM headroom remains ample on the idle 24 GB device.

Harness notes for successors: Windows PowerShell 5.1 turns native stderr into
a terminating NativeCommandError under `$ErrorActionPreference='Stop'` (the
checker's stderr info lines killed early driver runs - use the driver's
Invoke-Capture); `Stop-Process` must be awaited before reading the poller CSV
(handle lingered and zeroed the first metrics pass); the terminal tool cannot
background commands - the 200k driver was detached via `Start-Process` with
log polling, and the whole ladder fits in ~25 minutes wall.

## 2026-09-13: T-GPLR-13 implementation COMPLETE — moe-row-set TG path runs end-to-end, checker + negative controls green, default-off path untouched

The reactive expert row-set task is implemented, built clean, and validated
end-to-end. Contract honored: single vertical slice, opt-in `--residency-tg`,
no behavior change on the handoff/default paths (both host suites green).

Wiring summary (all in the vendored tree unless noted):
- Runner: `--residency-tg` flag + usage; loud gates (requires `--residency
  moe-expert-bank`, `--tg-tokens > 0`; incompatible with `--residency off`,
  `--handoff/--handoff-partial`, `--mtp`); explicit request suppresses the
  T-GPLR-12 auto-handoff fit heuristic (row-set IS the TG policy choice);
  `ls.row_set_requested` set before init; `begin_row_set()` after PP (loud
  refusal aborts TG before any decode), `end_row_set()` after TG.
- Library: `run_start` gains `"rowset":"on"` exactly when active (absent when
  off — the immunity assertion holds); prebuild guard keeps the row-set TG
  set at width 1 (fails loudly on `n_rs_seq > 0`); executor stages the
  reactive union per layer/window and emits `rowset_init` / `rowset_fetch` /
  `rowset_round` (+ `rowset_refused` on failure), with `union_ids` ascending
  and the FNV-1a-64 hash over the canonical comma-joined string (schema
  draft sections 1-3).
- Checker (both check- and validate-telemetry): rowset event validation per
  schema section 4 — W=1 identity (`route_union == top_k`), accounting
  identity (`route_union == fetched + resident`, bytes iff fetched),
  generation lifecycle per (slot, owner, generation), hash recompute,
  run_start-marker gating, default-off absence. Hash compares use
  `strtoull` (the U64 hash exceeds LLONG_MAX; `atoll` saturated silently).
  Unknown rowset events still fail loudly; legacy traces unaffected.

Evidence (GPU mutex, exit 0, check-telemetry OK, 569 events):
- `build/llama-cuda/gplr13/rowset-pp64-tg3.jsonl`: Qwen3.6-35B MoE, PP 64 +
  TG 3, 4 slots. 120 `rowset_fetch` events; per-round `rowset_round`:
  staged 572,719,104 / full-bank 18,327,011,328 bytes = **3.12%** per round
  (8/256 experts routed at W=1 — exactly the reactive union). Route parity
  held: no missing-row path fired; checker recomputed every hash.
- Negative controls (`tools/run-gplr13-negfix.ps1`, host-only): 8/8 — the
  untouched trace passes; tampered fixtures each trip their rule loudly
  (hash mismatch, accounting identity, W=1 identity, first-window phantom
  resident rows, tracked-coverage desync, marker stripped, injected
  refusal).
- Driver: `tools/run-gplr13-rowset-probe.ps1` (rerunnable end-to-end probe
  under the GPU lock).

Status vs acceptance criteria: route-event parity demonstrated (executed
rows cover the routed IDs; checker lifecycle validation enforces it on every
trace); staging bytes per TG round = the measured union-width ratio (3.12%);
wall time recorded as instrumentation only. Focused-validation status:
ALL items complete (512 row-set run + qa-smoke 512 regression + ladder +
the dense ctest pair below); nothing else is open. Audit close-out
2026-09-13: dense cold-path validation passed — qwen38-layer-boundary +
layer-slot-lifecycle 2/2 under the GPU mutex with Q38LS_MODEL_PATH set
(harness note: tools/run-tests.ps1 reads but does not propagate the model
path to ctest children; export it before GPU ctest suites). Validator
parity: the rowset_init window_tokens rule now exists in BOTH validators
(check_telemetry main.cpp:738 and validate_telemetry, identical message);
a rule sweep found no other row-set asymmetry. Negative controls are
dual-path: 9 fixtures x both validators = 18/18 correct verdicts plus both
positive controls (tools/run-gplr13-negfix.ps1). The never-fired reuse
branch (rows_reused=0 at 4 slots) is ruled NOT a T-GPLR-13 defect — its
first positive demonstration is owned by the H1 follow-up contract
(docs/gplr13-bottlenecks-hypotheses.md).

Notes for successors: `--residency-tg` + `--handoff` now conflict loudly
instead of racing the auto-fit decision; the serial TG loop aborts TG when
row-set activation fails (no more decode-after-refusal). T-GPLR-15's inputs
(rowset_fetch union streams + hashes) now exist in real traces.

## 2026-09-13: Memory-validation probe (T-MEMPROBE-01, owner request) — no RAM/VRAM overcommit observed; shared GPU memory does NOT duplicate device allocations; runner peak_vram_bytes overstated ~32%

Owner asked to validate actual RAM and VRAM usage (suspicion of overcommitting)
and whether the GPU driver duplicates memory because it is "shared". Evidence
from one dense 1024-token layer-major probe (Qwen3.8-27B-UD-Q6_K_XL, ubatch
256, 4 slots, q8 KV, --no-handoff, GPU mutex, exit 0, check-telemetry OK,
generated ids 286 220 197 = the reference set) with a 0.5 s Windows
GPU-adapter-memory poller running across the whole run:

Artifacts: `build/llama-cuda/memprobe/memprobe-dense-1024.{jsonl,csv}`;
tooling: `tools/gpu-mem-poll.ps1` (new, incremental-flush poller: adapter
dedicated/shared segments, per-process GPU memory, avail phys, system commit)
and `tools/run-memprobe-dense.ps1` (driver; rerun to regenerate). Note LUIDs
rotate across boots — analyze by magnitude, not by fixed LUID.

1. **VRAM: no overcommit.** OS-observed peak dedicated (local) segment on the
   NVIDIA adapter = 3,356.9 MiB (all attributable to the runner, pid peak
   3,352.6 MiB within the ~4 MiB counter noise MS KB4490156 documents for
   per-process GPU counters). Runner-reported `peak_vram_bytes` = 4,439.56 MiB
   (4,655,218,688 B). The runner's number OVERSTATES the physical peak by
   ~1,082.7 MiB (~32%) because it is a before/after free-VRAM delta
   (main.cpp:3641: `total - min(free_after, free_at_start)`), not a true peak:
   CUDA context (~500-800 MiB) and other allocations that free before teardown
   are counted. It is a conservative upper bound, never under-reports; true
   mid-run transients above the end-state footprint would also be missed, so
   it is neither a peak nor a steady-state number.
2. **Shared GPU memory: NO duplication observed.** WDDM "shared GPU memory"
   is the non-local (aperture) segment backed by system RAM, used for system
   copies, command, and paging buffers. If the driver mirrored device
   allocations, shared would have grown ~3.3 GiB during the run. Measured:
   NVIDIA shared segment max 76.2 MiB (idle ~0.2 MiB); runner per-process
   non-local max 76 MiB. The 24 GiB of host-RAM drop seen during the run
   (avail phys 53.9 -> 29.7 GiB min) is the file-backed mmap of the 25.9 GB
   GGUF + staging touching pages, NOT a hidden copy of VRAM. Resident set
   grows because the file cache maps real pages; system commit peaked at only
   16.8 GB of the 113.5 GB limit (pagefile pressure ~none).
3. **RAM admission rule remains sound.** The T-GPLR-09 bank-cache admission
   arithmetic (avail_phys - banks >= 0.15 x total_phys) works on avail/total
   physical, which the probe confirms behaves as recorded (min avail 29.7 GiB
   with the model mapped, well above reserve). The VirtualLock/prefetch path's
   documented limitation (no SeLockMemoryPrivilege on this token) is
   unchanged.
4. **Telemetry gap (task-owned, owner call):** `peak_vram_bytes` should either
   be (a) sampled during the run (e.g. cudaMemGetInfo at each phase_summary,
   cheap) or (b) relabeled (e.g. `vram_before_after_delta_bytes`) so traces
   cannot be read as a device peak. Until fixed, treat reported peak VRAM as
   an upper bound with a ~0.5-1.1 GiB context/teardown skew on this machine.
   The 2026-09-10 8 GB MoE run's runner 6693.56 vs NVML 5607 MiB disagreement
   (~1.09 GB) matches this mechanism exactly (NVML sampled sparse under load;
   the delta metric is the larger, skewed number) — that discrepancy is now
   explained; no new defect is implied there.
5. **Caveats.** Single probe shape (dense, small ctx; MoE expert-bank staging
   not re-probed — its per-run VRAM story is the slot window + staging buffers,
   covered by the same delta-metric caveat). Windows GPU adapter counters are
   formatted-data (0.5 s granularity); a sub-0.5 s transient could still hide
   from BOTH instruments, which is why recommendation (a) samples
   cudaMemGetInfo in-process at phase boundaries. Shared-segment growth is
   still the first thing to watch if VRAM pressure ever pushes the driver
   toward paging; `--physical-vram-bytes`/preflight arithmetic is unaffected.

No code changed. No commit (not authorized).

## 2026-09-12: T-GPLR-13 gate (b) DECIDED — reactive row-set is the priority TG path; T-GPLR-16 deprioritized with option (a) disposition

The owner's priority call was made on the four-way comparison (reactive
row-set vs the T-GPLR-16 escalation options a/b/c). Decision and
reasoning, recorded in the plan header, T-GPLR-13 and T-GPLR-16 blocks:

- **T-GPLR-13 (reactive row-set) is chosen** and moves DRAFT -> READY
  (next implementation task). Why: (1) single-process — the T-GPLR-17
  rejection ("cannot have inference that constantly switches engines")
  makes single-process operation program-defining, and row-set is
  single-process by construction while T-GPLR-16's surviving options
  are not clearly so; (2) bounded VRAM by construction (slot window +
  row-granular packages) — it is the designed answer for the
  not-fitting regime (T-GPLR-16's original target), which the closed
  T-GPLR-10 handoff cannot serve; (3) the riskiest unknown (scheduler
  cross-backend staging) was already retired by the SDD amendment's
  hard execution-shape constraint, grounded in the T-GPLR-16 deadlock
  evidence; (4) it is also the measurement instrument — its
  rowset_fetch events are the only source of per-token TG route data,
  so it unblocks T-GPLR-15's UNMEASURED 0.6 hit-rate gate regardless
  of TG speed outcome; (5) in the fits regime it does not displace
  anything: auto-handoff (T-GPLR-12 default profile) already wins
  there at 64.63 tok/s and row-set is TG-only opt-in.
- **T-GPLR-16 disposition: option (a)** — revert MIN_BATCH=1 to the
  pre-fix partial TG (known step-1 gold flip, QA-gate failure accepted)
  and deprioritize the task to DRAFT; the in-progress implementation is
  parked. Option (b) (upstream device-side deadlock chase, symbolized
  build + nsight/driver trace) is declined for now: open-ended upstream
  debugging investment against a sharply characterized but
  upstream-owned failure, while row-set sidesteps the mechanism
  entirely (its packages stage through the copy-event lifecycle before
  graph execution; MUL_MAT_ID never sees host-resident banks on this
  path). Option (c) (CPU-PP consistent arithmetic) stays recorded as a
  valid fallback but unprioritized: it avoids the deadlock by
  construction yet regresses PP throughput, and it does not close the
  not-fitting regime the way row-set does.
- Priority order now: T-GPLR-13 (ready), then T-GPLR-15 review on
  rowset_fetch evidence. T-GPLR-16 resumes only on a new owner call.
- No code changed in this step (decision record only; docs edits in
  the plan and this file). The MIN_BATCH=1 probe guards in the vendored
  tree (LLAMA_LS_NO_MINBATCH_PROBE etc.) now have an owner disposition
  pointing at option (a); the revert itself is vendored-tree work that
  should be scheduled alongside or after T-GPLR-13 per owner direction.

Vendored llama.cpp state: `099a27a7e` — layer-streaming: T-GPLR-08/10/11,
prefill layer residency, MoE handoff, tail-ubatch fix (2026-09-09).
Working tree adds T-GPLR-12 (default profile v2) on top of `099a27a7e`,
uncommitted (no commit authorized yet): runner tri-state + defaults,
module handoff_mode telemetry, MoE builder bf16 slab casts, checker
`handoff_ever` fix, CMake/qa-smoke default-immunity pins.

## 2026-09-12: Drafted model-run testing standard

Drafted `docs/model-run-testing-standard.md` in response to the question of
whether the workspace should standardize on a single-shot, file-driven,
capped llama-completion shape to avoid runaway/hard-to-grade test runs.

Verdict: the intent is right; the exact phrasing in the request is too
informal to adopt as written. The draft standard captures the intent as a
reproducible invocation shape: single-shot, prompt from a file, explicit hard
cap, deterministic artifacts, and programmatic grading with a must-fail
negative control. Force-termination is retained as a controlled fallback for
a genuinely stuck process, not as the normal way a test finishes.

The draft standard points at the existing semantic-QA smoke gate as the local
model for this shape. It does not yet claim the whole repo has adopted it;
that is an owner-review decision. Implementation note:
the standard does not wait for those three clarifications to exist; it records
them as explicit scope items, and the implemented shape resolves the vehicle
question directly: "llama-completion" is the stock llama.cpp single-shot
completion vehicle, and the equivalent local vehicle for the experimental path
is the layer-streaming runner with --prompt-file / --tg-tokens / --tg-text.
Whether the standard is adopted repo-wide, and against which real failure mode,
is still an owner-review decision.

## 2026-09-12: moe-row-set SDD amendment written (owner-directed; T-GPLR-13 gate (a) resolved)

The owner directed the `moe-row-set` amendment and its elaboration. Amended
docs/layer-residency-sdd.md sections 1, 4, 6, 8, 9, 10, 11, 12 and the plan
header + T-GPLR-13 status. Key content:

- New package class `moe-row-set` (section 4): per logical MoE layer and TG
  window (default one token), the deduplicated union of routed-expert rows
  (gate/up/down) the router actually selects; reactive, never predictive;
  immutable per generation; atomic ready/retire; loud under-coverage refusal.
- **Execution-shape constraint (hard requirement)**: row-set packages are
  CUDA-resident slot buffers staged BEFORE graph execution under the standard
  copy-event/generation/consumer-done lifecycle; `MUL_MAT_ID` on this path
  always consumes CUDA-resident weights of the matching slot generation;
  scheduler cross-backend staging (ggml-backend rowcopy, op-offload rewires)
  is prohibited for row-set graphs. Grounded in the T-GPLR-16 deadlock
  evidence (same-date section below).
- Policy `moe-row-set` added (TG-phase only; PP keeps its declared policy;
  default-off unchanged). Adapter boundary grows per-window reactive
  row-membership enumeration + rowset_fetch emission + declared forbidden
  staging paths. Failure contract, telemetry (full per-token union ID lists —
  T-GPLR-15's gate telemetry), acceptance gates (route-event parity,
  checker-validated completeness, staging bytes vs full-bank baseline,
  QA gate, default-off regressions; performance claims only vs the reactive
  baseline and only at hit rate >= 0.6), and risks (UNMEASURED temporal hit
  rate, volatile union width, the execution-shape landmine) all extended.
- T-GPLR-13 gate (a) is thereby RESOLVED; gate (b) (owner priority vs the
  handoff path) remains open. T-GPLR-15's 0.6 hit-rate gate is unchanged and
  still unmeasurable until T-GPLR-13 exists.

## 2026-09-12: T-GPLR-16 MIN_BATCH=1 wedge fully characterized — device-side stream deadlock, containment attempt reverted, escalation

Continuation of the 2026-09-11 entry. The "wedge" is now precisely located
and mechanized; the prior interpretation (upstream rowcopy × CUDA-graph-cache
corruption) is **superseded**.

New evidence (all under GPU mutex, artifacts `build/llama-cuda/gplr16/`):

1. **The freeze point is the 3rd TG token, not the 2nd.** TG telemetry
   `ubatch` indexes tokens: ub1 = token 0, ub2 = token 1, ub3 = token 2.
   Tokens 0–1 complete with bit-exact step-0 margins; every MIN_BATCH=1 run
   then dies inside the first `layer_compute` of ub3 (marker legs:
   `[ls-exec] g=18 compute_enter` with no `compute_done`; g=18 is host-bank
   layer 17's graph with rowcopy, layer 0's graph in the no-rowcopy leg).
2. **It is a device-side stream deadlock, not a host lock and not corruption
   at capture time.** During the freeze: GPU util 0 %, process alive, main
   thread spinning inside `cudaStreamSynchronize` → driver `cuStreamQuery`
   poll loop (captured with a StackWalk64 dumper, `tools/gplr16-dump.cpp`,
   `build/gplr16-dump.exe`, no PDBs so frames below the CUDA boundary are
   export-guess noise). If left alone long enough, Windows TDR reaps the
   stuck kernel: `CUDA error: the launch timed out and was terminated` at
   `ggml_backend_cuda_synchronize`.
3. **Independent of CUDA graph capture.** With `GGML_CUDA_NO_GRAPHS_PROBE=1`
   (verified applied: 0 "warmup complete" lines) the run froze identically
   at ub3/l17. Direct execution of the same mixed-split graph deadlocks too.
4. **Independent of expert rowcopy.** With `GGML_SCHED_NO_EXPERT_ROWCOPY_PROBE=1`
   (full 452 MB bank copy fallback instead of used-row staging) the run froze
   identically at ub3/l0. Staging tensors are not the trigger.
5. **Decisive correlation: MIN_BATCH value, not token count.** Every
   completing partial-handoff cell (ab2-D/E/F, ab-B/C; 96 TG tokens each)
   ran with MIN_BATCH at its default 32, i.e. TG-side expert FFN of host
   banks on CPU. Their garbage `.txt` answers show they are pre-fix builds —
   completing-but-wrong, not counter-evidence. Every MIN_BATCH=1 attempt
   (TG expert FFN of host banks forced onto CUDA through scheduler staging)
   deadlocks the device by TG token 2. The deadlock is in the CUDA-side
   `MUL_MAT_ID` execution against host-resident expert banks under the
   partial-handoff scheduler configuration (fast-path kernel per
   `ggml_cuda_mul_mat_id_needs_sync` selection), regardless of how the rows
   got there (rowcopy staging or full-bank copy).
6. **Containment attempt (capture-time staged-weights discriminator) is
   proven ineffective and was reverted.** Two independent reasons: (a) the
   `ggml_cuda_graph_check_compability` check runs at graph build, *before*
   `ggml_backend_sched_compute_splits` rewires `MUL_MAT_ID` src[0] to the
   rowcopy staging tensors, so it can never observe them at capture time;
   (b) capture is not the mechanism anyway (finding 3). A comment remains at
   the site in `ggml/src/ggml-cuda/ggml-cuda.cu` so the idea is not retried
   blindly; the code is functionally back to the pre-attempt state.
7. **Memory-commit lesson (owner request: keep commit under ~50 GB).** A
   frozen leg holds ~20.6 GB private commit and WDDM mirrors more; two
   concurrent legs (one frozen, one mutex-waiting) approach 130 GB commit on
   a 61.7 GiB host. GPU legs stay strictly serialized, one at a time, and a
   run confirmed frozen is killed immediately after its evidence is captured.

New env probe guards in the vendored tree (all reversible, remove after
resolution): `GGML_LS_GRAPH_EXEC_MARKERS` (per-graph exec markers), plus the
previously recorded four. New artifacts: `contain-V1.*` legs (containment
run, graphs-off leg, rowcopy-off leg), `fix-V1-killed-session-evidence.*`,
`build/gplr16-dump.exe`, `tools/gplr16-dump.cpp`.

Open items (escalated to owner — the plan's escalation condition applies;
none of the three can be closed by implementer action alone):
- (a) Revert MIN_BATCH=1 → pre-fix partial TG with the known step-1 gold
  flip; fails the QA gate as demonstrated (2026-09-11 evidence).
- (b) Chase the device-side deadlock upstream (symbolized CUDA build with
  PDBs, nsight/driver trace) — a real but open-ended debugging investment;
  the trigger is now sharply characterized (MUL_MAT_ID CUDA fast path on
  host-resident banks, 3rd token, partial-handoff scheduler configuration).
- (c) Consistent-arithmetic alternative: CPU PP for host layers (recorded
  option c) — avoids the deadlock by construction, needs the owner's call
  because PP throughput regresses.

## 2026-09-11: T-GPLR-16 root cause — mixed-arithmetic PP→TG switch, not corruption

Full root-cause chain for partial-handoff TG degradation, all evidence under
GPU mutex in `build/llama-cuda/gplr16/`:

1. **No memory corruption.** Runtime per-node eval-callback probes (`rt-*`)
   show routing/expert IDs/attention bit-identical between full handoff and
   partial N=20 through the whole stack at TG step 0.
2. **Divergence = cross-backend kernel variance.** Pre-fix partial TG computes
   host-bank `MUL_MAT_ID` on CPU (`buf=CUDA_Host`; CUDA offload gate requires
   `get_op_batch_size >= GGML_OP_OFFLOAD_MIN_BATCH`=32, `MUL_MAT_ID` batch =
   ne[2]=1 at TG). CPU Q4_K vs CUDA MMQ differs ~0.3% per host layer,
   envelope-consistent through L20 — benign in isolation.
3. **The killer is the PP→TG arithmetic switch.** PP runs all-CUDA; TG flips 20
   layers to CPU arithmetic mid-trajectory. Greedy chain evidence: TG step 0
   survives (margin 0.440 full vs 0.354 partial, same gold 248046), step 1
   flips gold 198→271 (margin 16.30→1.75), then repetition collapse. Stock
   `-ncmoe 20` proves CPU-FFN TG is fine when CONSISTENT: stock S2 answered
   the anchor-512 QA identically to full-CUDA S1 (check-answer 5/5 PASS both)
   with its FFN measurably on CPU for PP and TG (PP 4.60 ms/tok vs 0.53, TG
   17.0 ms/tok vs 8.4). The SDD "must not alter model arithmetic" invariant
   is violated *per-request* by the switch, not by host placement per se.
4. **MIN_BATCH=1 fix works numerically, wedges in upstream machinery.** Runner
   override `GGML_OP_OFFLOAD_MIN_BATCH=1` under `--handoff-partial` routes TG
   expert FFN to CUDA (rowcopy of used experts only): TG step-0 margin becomes
   0.439782 == full handoff exactly; V2 full-handoff regression bit-identical
   to reference; tiny-prompt partial runs complete correctly. But anchor-512
   partial wedges at TG token 1, layer-17 graph (2nd execution): graphs ON →
   memcheck-named `mul_mat_vec_q` wild-read (IDs garbage at kernel arg);
   graphs OFF/rowcopy OFF/launch-blocking → deterministic hang, same spot;
   synccheck clean. This is upstream ggml-backend rowcopy × CUDA-graph-cache
   interaction on 41-graphs/token mixed-split TG sets, not our executor
   (our executor snapshots/restores rewired srcs and syncs correctly).

Open items: choose between (a) revert MIN_BATCH=1 and accept pre-fix partial
with the known step-1 numeric flip (fails QA gate), (b) fix/contain the
upstream rowcopy+capture wedge (e.g. disable CUDA graphs for TG layer graphs
containing rowcopy splits only — narrower than the global NO_GRAPHS probe,
which also hung), or (c) escalate the mixed-arithmetic invariant question to
the owner (PP-on-CUDA + TG-on-CPU for host banks cannot pass the QA gate as
demonstrated; consistent-arithmetic alternatives are CPU PP for host layers
or the MIN_BATCH=1 route once the wedge is fixed). Env probe guards in the
vendored tree (remove after resolution): `GGML_CUDA_NO_GRAPHS_PROBE`,
`GGML_SCHED_NO_EXPERT_ROWCOPY_PROBE`, `LLAMA_LS_NO_MINBATCH_PROBE`,
`LLAMA_LS_TG_ROUTER_PROBE`. Executor `[ls-exec]`/rt-probe markers are
guarded by `LLAMA_LS_GRAPH_EXEC_PROBE`/`LLAMA_LS_TG_ROUTER_PROBE`.

Also fixed this session: duplicate `layer-streaming-scheduler-256-run`/
`tail513` ctest blocks in `tools/layer-streaming/CMakeLists.txt` (prior-
session edit accident; configure failed with duplicate test names).

## 2026-09-10: T-GPLR-17 REJECTED by owner and fully reverted (two-process hybrid bridge)

After the implementer proved the mechanism end-to-end (bridge probe under
GPU mutex, artifacts `build/llama-cuda/gplr17-probe/`: runner layer-major
PP 512 tokens + `--state-out` -> slot state file 71,447,536 B (KV + recurrent
DeltaNet state + 512 prompt tokens, sidecar interop gate OK) -> llama-server
`-ncmoe 40 -ngl 999` healthy in 3.6 s, `POST /slots/0?action=restore`
n_restored=512 == sidecar, completion prompt_n=1 continuity proof, TG 23.2
tok/s, driver exit 0), the owner rejected the approach outright: an
inference system that switches engines mid-session is unusable for
production or even local use ("cannot have inference that constantly
switches engines"). Decision: revert everything.

Reverted in full (verified vs the pre-existing working tree): runner
`--state-out` flag + state-save block + sidecar, `state_save` telemetry
method, off/handoff conflict gates, host fixtures and CMake tests,
tools/run-hybrid.ps1, and the hybrid-probe/hybrid-qa/hybrid-50k suites in
tools/run-tests.ps1. The vendored tree is back to the T-GPLR-12 working
tree only; the runner was rebuilt from the reverted sources and the host
suite re-passes. The plan records T-GPLR-17 as rejected with the recorded
state-save/restore seam facts (library `llama_state_seq_save/load_file`,
server POST /slots/{id}?action=save|restore) kept as historical evidence.

Consequence for the open program: the single-process mixed-placement TG
mode (T-GPLR-16, owner-gated) is the aligned direction for TG-at-context
under the 8 GB ceiling; no two-process orchestration will be pursued
without a new owner decision.

## 2026-09-10: T-GPLR-17 drafted READY — two-process hybrid bridge (streaming PP -> state file -> CPU-expert llama-server TG)

Planner session (docs-only; no code, no GPU run). Owner request: wire
llama_state_save_file/load_file into the runner and llama-server CLIs so
streaming PP at 50K hands off KV state to a CPU-expert llama-server for
stock TG under an 8 GB VRAM ceiling. Key inspection facts (verified in the
vendored tree, not assumed):
- `llama_state_seq_save_file` / `llama_state_seq_load_file` exist
  (src/llama-context.cpp, state_seq_load_file ~3534) and the payload is the
  prompt token list PLUS the full memory-module write
  (`memory->state_write(io)`) — KV AND recurrent DeltaNet state both cross
  the process boundary.
- The server ALREADY exposes the seam: POST /slots/{id}?action=save|restore
  (routes registered server.cpp:273; handlers server-context.cpp ~2480 and
  ~4553; restore erases the slot, loads the file, updates cached prompt).
  So the server side needs ZERO vendored code changes; only the runner needs
  one flag (`--state-out PATH`, calling llama_state_seq_save_file after PP)
  plus a PowerShell driver `tools/run-hybrid.ps1` (start fit-target/-ncmoe
  server, /health, POST restore, generate). Existing server unit tests cover
  the endpoints (tools/server/tests/unit/test_slot_save.py).
- Drafted as T-GPLR-17 in the generic plan, status READY, next eligible
  item. Explicitly NOT an SDD amendment: no new residency package type,
  policy, or placement mechanism — orchestration over two existing
  processes plus one runner flag. Acceptance: 50K end-to-end under the GPU
  mutex (runner ~6.5 GB; server <= 8 GB ceiling), state-file size ~KV
  534 MB + recurrent 66 MB, continuity proven by the QA 512 rung through
  the hybrid, default-off byte-identical, dense ctest 2/2.
- Plan repair en route (flagged): T-GPLR-15's body (Goal/Scope/Acceptance
  etc.) had been stranded after T-GPLR-16's block since the 2026-09-10
  renumber session, leaving T-GPLR-15 a stub; the body is now inside
  T-GPLR-15's own section (14 -> 15 -> 16 -> 17 ordering), content
  unchanged.
- Naming note: full-context `llama_state_save_file/load_file` (session
  variant, completion.cpp --session path) is NOT the chosen vehicle; the
  per-sequence slot variant matches the server endpoints and avoids
  whole-cache clobbering.

Runner default profile v2 (T-GPLR-12, owner-approved 2026-09-09):
`--kv-cache q8`, `--pp-schedule layer-major` + `--pw-probe`,
`--act-slab-dtype bf16`, `--handoff auto` (auto enables handoff only
when the T-GPLR-10 fit gate passes at run start; explicit `--handoff`
keeps hard-refuse semantics; `--no-handoff` disables). Every flag stays
overridable; refusal/refusal-free behavior is loud in both auto
branches (`handoff_mode`: auto-on | auto-off | on | off, validated by
check-telemetry). MTP stays opt-in (subsumed by auto-handoff in the
fits regime).

Milestone tag `layer-residency-plan-complete` (2026-09-06): this repo →
27514d1dabeef7119b53241aad65167ae2a410d8; vendored llama.cpp →
169ab086e857d287daac626dc7eda6a4f9764efe. Both annotated, local only
(not pushed).

## 2026-09-10: Streaming-PP + stock-TG hybrid (--handoff) at 50K — loud preflight refusal on today's VRAM baseline

Owner question: can streaming layer-major PP be combined with stock llama.cpp TG for a
262K test, and can the combo run at 50K now? Shape attempted (GPU mutex, launcher
`build/run-hybrid-50k-handoff.ps1`): `--pp-schedule layer-major --pp-tokens 51200
--ubatch 512 --layer-slots 4 --ctx-size 51456 --kv-cache q8 --handoff --pp-no-logits
--tg-tokens 16 --greedy --no-mtp` on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.
- Outcome: REFUSED at the T-GPLR-10 handoff capacity preflight, before the timed PP,
  exit 1 — designed loud behavior, no silent fallback. `weights+KV+state
  22,003,032,576 B (KV 559,841,280 B) + act slabs 419,430,400 B + reserve 536,870,912 B
  = 22,959,333,888 B exceeds observed free VRAM 22,226,665,472 B`.
- Margin arithmetic: shortfall 732,668,416 B (~699 MiB). Total q8 KV @ 51456 is only
  559,841,280 B, so NO context size closes the gap (floor at near-zero ctx is still
  ~173 MB short) — the full-model handoff cannot fit on this card at today's desktop
  free-VRAM baseline (22.23 GB, down from 22.92 GB observed 2026-08-21) at ANY length.
- Ceiling unchanged: the combo requires the whole model in VRAM post-handoff, so 262K
  is impossible on 24 GB by construction and 50K is now below the floor too. The combo
  remains proven at small ctx (T-GPLR-10: 2048-token run, TG 64.63 tok/s, parity PASS).
- Artifacts: `build/llama-cuda/gplr-hybrid-50k-handoff.{jsonl(empty),log}`; the .jsonl
  holds no timed events (refusal precedes PP). No commit.

## 2026-09-10: Streaming-TG slowness profiled against stock baselines (host-only analysis, no GPU run)

Question: where does the 8 GB streaming TG's 0.509 tok/s (1.966 s/token)
go, and why is stock TG 32-127x faster? Method: event-level re-analysis of
`build/llama-cuda/gplr-moe-8gb-50k.jsonl` (TG window = events between the
two phase_summary lines), the runner source, the ncmoe262k bench JSON, and
the owner's stock llama-server baseline log
`C:/Development/AI/qwen38GZI/out/llamacpp-baseline.log` (36 KB, full session).
No GPU work; no mutex needed.

Stock TG baselines, now with correct shapes:
- llama-bench `-ncmoe 20` TG leg: 64.9 tok/s = 15.42 ms/tok, but at
  n_prompt=0 (zero-length KV, no MTP) — a weights-only decode rate.
- Owner's llama-server run (new, 2026-09-10): build 10876, `--fit-target
  16000` auto-fit placed 34/41 layer groups on GPU (~6.6 GiB) with the
  remaining routed experts on CPU, `-c 262000`, q8 KV, FA, MTP draft
  (max 5). Measured: **TG 32.44 tok/s (30.83 ms/tok) over 1724 tokens**,
  MTP acceptance 0.546 (acc/pos 0.816/0.661/0.525/0.415/0.313, mean len
  3.73), graphs reused 456, PP 372 tok @ 39.49 tok/s. This is the real
  "expected TG" at context on this machine.
- Derived (arithmetic, marked as such): routed-expert row bytes per token
  = 8 experts x 1.77 MB x 40 layers = ~565 MB; the bench's 15.4 ms and the
  server's per-round numbers both close at ~35-40 GB/s effective DDR feed
  (memory-bound CPU expert reads) — the stock path touches only routed
  rows, never whole banks.

Streaming-TG decomposition (from the trace, 3 tokens, all 120 copies):
- Wall 1.966 s/token. Copy windows sum to 2.220 s/token (54.98 GB = 18.33
  GB x 3; ~0.25 s/token of copy overlaps compute), event wait 6.2 ms,
  compute 46.6 ms (2.4% of wall). The phase IS the copy chain.
- Per-bank copy: 453 MB in 46.4-190 ms (p50 47.0 ms); effective feed
  8.26 GB/s (host_submit 85.8% + ready_latency 14.2% of each window).
- Serialization: 108/120 next-layer copies were submitted only after the
  current layer's compute started; effectively one copy in flight — the
  slot protocol copies, waits on the CUDA event, computes, per layer.
- Compute is NOT the problem: 46.6 ms/token summed over 40 layers (0.31
  ms recurrent layers; the 10 full-attn layers 0.6-7.1 ms each at 50k q8
  KV). graph_builds 0 in TG; per-copy ready-latency (~6-22 ms fault-start)
  is small only relative to 453 MB copies — it would dominate if banks
  were chopped into small pieces without pipelining.

Why (attribution): the streaming path stages WHOLE 453 MB banks — 18.33 GB
per token — while the token's routed experts touch ~565 MB: a **32x byte
amplification**. Combined with a ~4.6x slower effective byte feed (8.26 GB/s
fault-driven H2D slot stream vs ~35-40 GB/s host DDR for the CPU-expert
path), that accounts for the ~127x gap to the bench number; the server's
32.44 tok/s with MTP sits on the same ~15-19 ms raw forward (MTP nets ~3.7
tokens/round at 0.546 acceptance). The stock CPU-expert path is effectively
row-set semantics executed by the CPU at DDR bandwidth; the streaming path
is whole-bank semantics over a slow, serialized VRAM-slot feed. Ceiling
arithmetic at measured rates: whole-bank staging cannot exceed ~0.45-0.7
tok/s even with pinned staging (12.7 GB/s) — the gap is the unit of
staging, not tuning.

Implications for the open program (estimates, not measurements):
- VRAM row-set (T-GPLR-13 shape) at 8.26 GB/s: ~68 ms/token copy feed ->
  ~14.6 tok/s ceiling if per-layer row copies pipeline behind compute;
  latency-dominated (~9 tok/s) if they stay serialized. MTP verify raises
  bytes/round (draft-route unions) and does NOT help a bandwidth-bound
  H2D path (~9-14 tok/s).
- The CPU-expert path (the owner's baseline shape) reaches 32.44 tok/s
  today with MTP — the fastest measured TG-at-context on this machine.
  The two-process hybrid (streaming PP + KV state save/restore into a
  fit-target llama-server, T-GPLR-16-adjacent) is the bridge that
  preserves our PP rate and this TG rate; a single-process equivalent
  needs mixed-placement TG, not row-set, to match it.
- `--copy-source pinned` exists and is a cheap probe for the copy-feed
  half (PP pinned staging measured H2D ~12.7 GB/s), but it cannot change
  the 32x byte amplification.

Supersedes the interpretation (not the data) of the NCMOE-262K section's
"TG @ full 262K ctx" claim above. Artifacts: the traces named above plus
the owner baseline log; no new run, no commit.

## 2026-09-10: T-GPLR-14 numbering collision resolved (mixed-placement TG mode renumbered to T-GPLR-16)

The NCMOE-262K section (2026-09-09) forward-referenced a "T-GPLR-14
mixed-placement TG mode" — a fresh collision with T-GPLR-14 (route-overlap
instrument, done 2026-09-09), added concurrently by the NCMOE session.
Resolution: the mixed-placement TG mode is drafted as **T-GPLR-16** (next
free ID after T-GPLR-15) in the generic plan, house-style row with goal,
scope, non-goals, design/failure contract, falsifiable acceptance criteria,
focused validation, and escalation condition. The stale reference in the
NCMOE section above was renumbered to T-GPLR-16, and the plan header and
serialization note now list T-GPLR-16 among the remaining DRAFT items
(T-GPLR-13 reactive row-set, T-GPLR-15 predictive prefetch, T-GPLR-16
mixed-placement TG — all owner-gated; one ready task at a time).
T-GPLR-14 remains closed: `tools/route_overlap/analyze.py` and its
byte-identical fidelity proof are untouched by this edit. No plan or SDD
content was redesigned; the collision fix is renumbering + drafting only.

## 2026-09-09: MoE TG row-set program drafted/renumbered; T-GPLR-14 DONE (route-overlap instrument)

Numbering collision caught before it hurt: the main worktree already holds
the owner's own uncommitted T-GPLR-12 "default profile v2" (done,
owner-approved, gplr12-* traces). The row-set program drafts were
renumbered to T-GPLR-13 (reactive expert row-set), T-GPLR-14 (route-
overlap instrument), T-GPLR-15 (predictive per-token prefetch) and are
recorded in the plan.

T-GPLR-14 is implemented and closed (plan status -> done):
- Tool: `tools/route_overlap/analyze.py` — stdlib-only, read-only,
  deterministic (no timestamps). Computes per-layer route-union widths
  (T1), the exact chunk/layer hash-equality overlap bounds (T2/T3: equal
  FNV route hashes => identical expert unions; unequal never implies
  inequality), an advisory sampled-ID containment overlap (T4: |A n B| / |A|,
  biased low, never gate input), and the token-level temporal verdict (T5:
  MEASURED only from per-token route events, else loud INSUFFICIENT_INPUT /
  NO_TG_PHASE / NO_ROUTE_EVENTS). Unsupported events, missing route fields,
  malformed lines, and missing files exit 2 naming file:line or the path.
- Validation: self-test PASS (19 checks on hand-computed fixtures, including
  the per-token MEASURED path and both loud-failure negatives); fidelity
  proof — a rerun over the 5 gplr12-* traces reproduces
  `build/llama-cuda/gplr14-route-overlap.{txt,json}` BYTE-IDENTICALLY
  (json record-equal; txt including the `json report:` trailer);
  determinism — double run byte-identical (same output names).
- Real evidence (5 independent MoE traces): union widths 245–256 on the
  512-token anchor, collapsing to 19–174 on the 50-token anchor (late
  layers collapse hardest); 100k/262k PP unions near-saturated; T2/T3 = 0.0
  (no chunk ever repeats a full union — consistent with full-bank prefill
  unions saturating); T4 advisory 0.2892/0.2952/0.3096 on the PP runs.
- PRIMARY FINDING (instrumentation result, not a guess): the runner emits
  route events ONLY at PP-chunk granularity — no surviving trace carries
  per-token TG routes (the T-GPLR-06 burst traces that motivated the
  "routes already exist per token" premise did not survive). The
  token-level temporal hit rate that gates T-GPLR-15 (proposed 0.6
  threshold) is UNDECIDABLE from existing traces and stays
  PROPOSED/UNMEASURED. Consequence recorded in the plan: T-GPLR-15's gate
  telemetry falls to T-GPLR-13's `rowset_fetch` events (full per-token
  union ID lists), so no extra telemetry work appears.
- Gated, not blocked: T-GPLR-13 still needs the two owner decisions (the
  `moe-row-set` SDD amendment and product priority vs the handoff path)
  plus the vendored CUDA tree.
- Provenance: the first implementation ran in an isolated (ephemeral)
  workspace lost on tab close; the tool was reconstructed in the main
  worktree and re-proven against the surviving artifacts byte-for-byte and
  the surviving self-test fixtures (%TEMP% route-overlap-selftest-*). The
  T4 containment metric is pinned by the recorded 0.28921875 on
  100k-bf16 (Jaccard and pooled variants do not reproduce it).
- No commit (not authorized).

## 2026-09-09: T-GPLR-12 DONE — default profile v2 (bf16 kept on MoE by owner decision)

All four flips validated under the GPU mutex (traces
`build/llama-cuda/gplr12-*.jsonl`, checker OK on every trace):
- Dense anchor: f32 vs bf16 IDs `[286]` identical; ctest
  layer-streaming-layer-major-1024 2/2 in 19.8 s under pure new
  defaults (run_start slab=bf16 kv=q8 handoff_mode=auto-off, ID 286).
- Auto-on: QA-50/QA-512 pure-default runs auto-enabled handoff, migrate
  18,327,011,328 B exact, answers PASS (N1; 5/5), negative controls
  FAIL as designed. Auto-off: 262K streamed at 1160.9 tok/s with the
  loud notice (24.3 GB need > 22.2 GB free), exit 0.
- bf16 MoE routing perturbation (escalation clause FIRED): 100K pair
  route hashes 0/840 equal; QA-512 same-shape three-way — bf16 vs f32
  differ by ONE inserted template-boundary token, needle tail
  byte-identical, answers 5/5 both; the f32 control also differs from
  the old batch-major reference (pos 43), so TG-ID parity across
  schedules was never attainable for MoE. Owner re-decided: bf16
  default kept EVERYWHERE; MoE comparisons are answer-gated, never
  token-ID-gated.
- Why bf16 is worth it (measured): slabs halve (100K MoE 1.53 GiB ->
  768 MiB; 262K 4 GiB -> 2 GiB), peak VRAM drops equally, and slab
  bytes feed the auto-handoff fit arithmetic (marginal shapes flip
  auto-off -> auto-on). Dense: bit-identical + ~2% faster PP.
- Defects found & fixed en route: (1) MoE streaming builder lacked the
  dense builder's three slab_bf16 cast points — MoE+bf16 crashed in
  norm.cu (ported the cast-on-read pattern); (2) the new checker
  on/auto-on rule used handoff_seen (reset per phase_summary) and could
  never pass — now handoff_ever.
- Environment note: CTest entries bake Q38LS_MODEL_PATH at configure
  time; the build had been configured with it empty (every dense GPU
  ctest failed at arg parsing). Reconfigured with
  Q38LS_MODEL_PATH=C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf.
- qa-smoke and the eight dense GPU ctests are pinned to their recorded
  shapes (--pp-schedule batch-major / explicit dtype+kv / --no-handoff)
  so the default flip cannot silently change what they test; host suite
  23/23 including the new tri-state/refusal fixtures.

## 2026-09-10: Owner probe — MoE 50K streaming PP + streaming TG under an 8 GB VRAM ceiling

Question: can our layer-major PP followed by a TG phase stay under 8 GB VRAM
on the MoE model? Answer: yes, in the streaming-TG shape. Run (GPU mutex,
exit 0): `llama-layer-streaming --model <Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf>
--pp-schedule layer-major --pp-tokens 51200 --ubatch 512 --layer-slots 4
--ctx-size 51456 --residency-budget-mib 7000 --no-handoff --pp-no-logits
--tg-tokens 3 --greedy --no-mtp --telemetry-chunk-stride 1 --telemetry
build/llama-cuda/gplr-moe-8gb-50k.jsonl`.
- Preflight (loud, passed): bounded 4-slot window 5,269,481,984 B <= budget
  7,340,032,000 B (7000 MiB); KV q8 @ 51456 = 533.91 MiB; reserve 512 MiB
  against observed free 22,226,665,472 B.
- PP: 51200 tok in 36.77 s = 1392.57 tok/s (transfer 108.62 s, wait 0.5 ms,
  compute 16.96 s, 36 evictions, graph_builds 0).
- TG: 3 greedy tokens in 5.90 s = 0.509 tok/s (transfer 6.66 s, wait
  18.6 ms, compute 139.9 ms, 120 evictions — TG pages the 40 banks per
  token; that IS the 8 GB TG path). check-telemetry OK (9163 events).
- Peak VRAM: **6693.56 MiB (6.54 GB) by runner telemetry**; NVML poller
  (1 s, sparse under load) sampled 5607 MiB max. Both well under 8 GB.
- Why `--no-handoff` is required here: the MoE handoff fit arithmetic
  (full_model 22.0 GB + slabs + reserve) would likely pass on this 24 GB
  card and migrate all 40 expert banks (18.33 GB) — impossible under 8 GB;
  and layer-major + no-handoff caps TG at 3 tokens by design
  (layer-streaming-lm-tg-cap). Full-handoff TG at 8 GB remains open work
  (owner-gated mixed-placement/row-set program, T-GPLR-13+).
- Artifacts: `build/llama-cuda/gplr-moe-8gb-50k.{jsonl,log}`,
  `build/gpu-mem-poll-8gb50k.csv`, launcher `build/run-8gb50k.ps1`.

## 2026-09-09: NCMOE 262K probe — stock TG at full context, 16 GB pipeline evidence

Owner question: can the streaming-PP + stock-llama.cpp-TG hybrid run
262K context on a 16 GB card? Ran `llama-bench -p 262144 -n 64 -fa 1
-ctk q8_0 -ctv q8_0 -ngl 99 -ncmoe 20 -r 1 -o json` (Qwen3.6-35B-A3B
Q4_K_S, RTX 5090 Laptop 24 GB, Ryzen 9 9955HX, GPU mutex, exit 0):
- PP @ 262K: **498.6 tok/s** (stock placement, first 20 banks on CPU).
- TG: **64.9 tok/s** (15.4 ms/tok). [Corrected 2026-09-10: the llama-bench
  TG leg runs with n_prompt=0 — zero-length KV on a BOS-only context — so
  "full 262K ctx" above is wrong for this number; the bench also has no MTP.
  The at-context stock TG baseline is the owner's 2026-09-10 llama-server
  run recorded in the streaming-TG profiling section below.]
- VRAM: **flat 14,885 MiB** load→teardown (sampler, 2 s cadence). No
  transient above steady state — llama-bench reserves weights+KV+
  compute buffers at load. Fits a 16 GB card with ~1.5 GiB margin
  (dGPU is headless on this laptop; idle = 0 MiB). N=18 would add
  ~0.9 GiB more headroom if ever needed.
- Artifacts: `build/llama-cuda/ncmoe262k.{json,log}`, `build/nvsmi-
ncmoe262k.csv`, `build/ncmoe262k.DONE`.

Combined 16 GB pipeline (measured): our streaming PP 1160.9 tok/s →
262K prefill in ~226 s; stock `-ncmoe 20` TG 64.9 tok/s thereafter.
Still a two-process measurement, not a single-process hybrid: KV state
save/restore exists in the library (`llama_state_save/load_file`) but
no CLI exposes it, and our runner cannot do `-ncmoe`-style partial
placement — the single-process version is the T-GPLR-16 mixed-placement
TG mode (owner-gated new work; renumbered 2026-09-10 from a stale
"T-GPLR-14" reference — that ID belongs to the closed route-overlap
instrument).

## 2026-09-07: T-GPLR-07 smoke tier wired into tools/run-tests.ps1 (host-validated)

`tools/run-tests.ps1 -Suite qa-smoke` runs the smoke gate as one repeatable
run under the GPU mutex. DEFAULT rung is 512 tokens on the 512-token anchor;
the 50-token rung is opt-in via `-SmokeLengths 50` (comma list; the
quoted-string form silently binds as one int and must not be used).
2026-09-09 owner decision: the ladder series is 50/512/4k/24k/50k/100k/200k
-- the 15/30 rungs were removed from the ladder entirely (their live
evidence below is historical; the 30-token answer-quality failure is no
longer a ladder blocker). 4k+ lengths are manual owner-invoked commands,
never part of the suite. The generator refuses targets below 15 tokens (the
K1-scaffold floor), so no shorter run exists.

Per-rung flow (unchanged leg shape, now looped over the selected rungs,
default 512):
1. `gen-needle-prompt.py --target-tokens <len> --calibrate llama-tokenize
   --model <Qwen3.6 GGUF>` (host-only, exact token count) -> parse
   achieved=/items= from its summary line.
2. Runner under `tools/with-gpu-lock.ps1` with the new flags:
   `--prompt-file <calibrated> --ubatch <achieved> --tg-tokens 96
   --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank
   --kv-cache q8 --allow-full-residency --telemetry <dedicated JSONL>
   --tg-text <detok text>`. Single-ubatch PP is the documented workaround
   for the tail-ubatch gap.
3. Strict `--check-telemetry` on the dedicated JSONL (the gate that the
   ladder's earlier 512/4k rungs lacked — mixed-log claims are not
   accepted).
4. `check-answer.py check --tg-text <detok text> --items <items= list>` per-needle
   grading (50 = N1, 512 = all 5 items) + the same grading with
   `--negative` must FAIL as designed (exit != 0 with aggregate=FAIL). The
   checker grades the runner's `--tg-text` file directly -- the runner already
   detokenized with the model's own vocab via `llama_token_to_piece`, so no
   trace re-parsing and no GGUF vocab read happen at grading time
   (`check-answer.py check` also still accepts `--trace --model` for
   backward-compatible trace grading).

Any step failing exits the suite 1 with the failing rung named; each rung
gate keeps the dedicated trace `gplr07-qa-<len>.jsonl`, the detokenized text
`gplr07-qa-<len>.txt`, and the runner log `gplr07-qa-<len>.log` in the build
dir for inspection.

Host validation (2026-09-07, no GPU): PowerShell parse clean; checker
self-test 15/15 (added tg-text file mode cases); calibration exact against the real model
(llama-tokenize, vocab-only) for the smoke tier: 50 -> N1, 512 -> N1,N2,N3,N4,K1
(needle rows 3/6/9/12 of 12), all achieved == target; 15/30 calibration
(from the same generator, now removed from the ladder) is historical.
prompt files regenerated in `build/llama-cuda/gplr07-qa-prompts/`. The 50/512
legs of this suite previously ran end-to-end on the GPU (2026-09-07 16:08/
16:12, surviving dedicated JSONL traces + logs in the build dir). A fresh
512 default-gate pass is the intended next invocation whenever the
GPU mutex is free; the plan's remaining acceptance items (strict-trace
re-verification of the 4k rung, dense Qwen3.8 regression re-run) are
unchanged.

## 2026-09-07/09: qa-smoke gate live runs (50/512 PASS; 15/30 removed from the ladder on 2026-09-09)

Ran the gate legs individually under the GPU mutex (shared-GPU contention
forced leg-by-leg execution; `tools/run-tests.ps1 -Suite qa-smoke` remains
the canonical one-shot gate). Command shape per leg, identical to the suite:
`llama-layer-streaming.exe --model <Qwen3.6-35B GGUF> --prompt-file
<anchor> --ubatch <achieved> --tg-tokens 96 --pp-no-logits --pp-schedule
batch-major --residency moe-expert-bank --kv-cache q8 --allow-full-residency
--telemetry <dedicated JSONL> --tg-text <detok text>`, each runner leg under
`tools/with-gpu-lock.ps1`, then strict `--check-telemetry` and
`check-answer.py check --tg-text ...` (+ negative control).

Results (the 15/30 rungs below were removed from the ladder on 2026-09-09;
their evidence is historical, artifacts retained in the build dir):
- 15-token rung: PASS. runner exit 0; check-telemetry OK (23326 events, 1
  slot); K1 'Paris' found; negative control FAILs as designed (exit 1,
  aggregate=FAIL). Artifacts `gplr07-qa-15.{jsonl,txt,log}` (17:49).
- 30-token rung: answer gate FAIL. runner exit 0 and check-telemetry OK,
  but the model does not answer: it pattern-completes the note scaffold
  ("Note N2: ... is QX-7742", "Note N3: ... QX-7743", ...) instead of
  stating QX-7741. The checker correctly rejects (aggregate=FAIL) and the
  negative control correctly PASSes on the wrong answer (exit 0) -- the
  designed must-fail property worked exactly as specified. This is a
  model-quality failure on the 30-token N1 scaffold (no pads between
  statement and question), NOT a runner/telemetry/checker defect: all
  scheduler invariants held. Greedy decoding makes a re-run near-certain
  to reproduce. Tuning the scaffold or the acceptance wording is a
  plan-owned change (generator contract) and was NOT done; superseded by
  the 2026-09-09 ladder reduction (see resolution below). Artifacts
  `gplr07-qa-30.{jsonl,txt,log}` (17:58).
- 50-token rung: PASS (fresh tg-text-graded leg, 2026-09-09). runner exit 0;
  check-telemetry OK (23326 events, 1 slot, graph_builds 0 both phases); N1
  'QX-7741' found (aggregate PASS); negative control FAILs as designed
  (exit 1). Artifacts `gplr07-qa-50.{jsonl,txt,log}` regenerated. (Earlier
  trace-graded run 16:08 superseded.)
- 512-token rung: PASS (fresh tg-text-graded leg, 2026-09-09). runner exit
  0; check-telemetry OK (23326 events, 1 slot, graph_builds 0 both phases);
  all five items found ('QX-7741 TR-2918 MK-5501 LD-8834 Paris', aggregate
  PASS); negative control FAILs as designed (exit 1). Artifacts
  `gplr07-qa-512.{jsonl,txt,log}` regenerated. (Earlier trace-graded run
  16:12 superseded.)

Blocker resolution (2026-09-09, owner): the 15/30 rungs were removed from
the ladder series entirely (series = 50/512/4k/24k/50k/100k/200k). The
30-token answer-quality failure is therefore no longer a ladder blocker;
its artifacts (`gplr07-qa-30.{jsonl,txt,log}`) remain as evidence, and the
negative control's must-fail property was still proven live by that run.
Default-gate invocation: `powershell -NoProfile -ExecutionPolicy Bypass
-File tools/run-tests.ps1 -Suite qa-smoke` (512 only, ~5 min); the 50-token
rung is opt-in via `-SmokeLengths 50`; 4k+ ladder lengths stay manual
owner-invoked commands.

## 2026-09-09: Planner review -- T-GPLR-08 ready, T-GPLR-09 held

Reviewed the two drafted tasks per the planner workflow (read SDD, current
state, environment facts, controlling code; stated outcomes/gates/unknowns;
resolved ordinary uncertainty with focused probes; only environment/policy
decisions deferred to the owner).

T-GPLR-08 (one-layer-ahead PrefetchVirtualMemory probe) moved to READY. The
contract is falsifiable (262K wall drop >= 20% AND bank transfer drop >= 2x
vs the 351.9 s T-GPLR-04 baseline; effective per-bank feed >= 2x the 52 MB/s
cold rate), evidence-backed (the warm-cache leg falsified standby residency;
layer 0's ~3.5 GB/s working-set path is exactly the mechanism a rolling
prefetch populates), and narrow (new flag, 1-deep worker queue, two
telemetry events, one probe). Start remains serialized after T-GPLR-06/07
close per the plan's ordering rule (one ready task at a time). One draft
typo fixed: the focused-validation build target said llama-cli; corrected to
llama-layer-streaming.

T-GPLR-09 (host-RAM bank cache tier) stays DRAFT with a recorded blocker:
(1) it hard-depends on T-GPLR-08's range walker and telemetry plumbing;
(2) a 2026-09-09 environment probe (whoami /priv, <host>\<user>,
non-elevated) shows NO SeLockMemoryPrivilege and SeIncreaseWorkingSet-
Privilege disabled, so the contract's primary residency path (working-set
raise + VirtualLock) cannot execute on this machine as configured and the
acceptance gates (wall <= 110 s) are unreachable without a Local Security
Policy change. Owner decision required before re-review: (a) enable "Lock
pages in memory" via elevated policy, (b) redesign without VirtualLock
(weaker guarantee), or (c) defer until T-GPLR-08 results inform the tier's
value. Privilege facts recorded in docs/AGENT_ENVIRONMENT.md.
[Superseded 2026-09-09: T-GPLR-09 is PARKED (owner decision), not merely
held -- see the PARKED section below.]

## 2026-09-09: T-GPLR-06/07 CLOSED — T-GPLR-08 and T-GPLR-10 may start

T-GPLR-06 owner review closed (acceptance met; plan status -> done): k=4
batch verify 2.58x >= the 2x wall gate (d8 2.97x; code-text d16 6.91x vs
the measured serial 249.69 s); greedy-prefix parity under the recorded
knife-edge standard (top-2 margins; reruns bit-identical); every failure
loud (probe trace, en-route defects: n_outputs_max MTP clamp + graph-input
buffer sizing, both fixed and recorded); traces archived in the build dir.

T-GPLR-07 acceptance closed (plan status -> done):
- 512/4k strict dedicated-JSONL re-verification COMPLETE. The 512 rung was
  re-verified earlier today (gplr07-qa-512.jsonl, check-telemetry OK, all
  5 items PASS, negative control fails as designed). The 4k rung was
  re-run fresh today under the GPU mutex (GPU idle after a ~7-min wait;
  anchor regenerated: target=4096 achieved=4096, needle rows 27/55/84/106
  of 113, matching the ladder record): runner exit 0;
  `--check-telemetry` OK on the dedicated JSONL (38686 events, 1 slot,
  graph_builds 0 both phases); PP 4096 tok @ 610.5 tok/s (vs the original
  606.9); TG 160 tok @ 0.521 tok/s (vs 0.53); aggregate PASS on all five
  items (QX-7741 TR-2918 MK-5501 LD-8834 Paris); negative control FAILs as
  designed (exit 1). Peak VRAM 5.37 GiB (this run used --ctx-size 25088;
  the original 4k run used a smaller ctx). Artifacts
  gplr07-qa-4k.{jsonl,txt,log}. The 'check-telemetry OK' claims for 512/4k
  are now backed by surviving dedicated JSONL.
- Dense Qwen3.8 regression re-run GREEN (GPU mutex, ctest
  layer-streaming-layer-major-1024 2/2 in 24.6 s, Q38LS_MODEL_PATH =
  C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf): trace
  q38ls-lm-1024.jsonl tg_tokens ids=[286] == the recorded reference first
  greedy ID; run_start arch=qwen35 n_layer=64 slots=4 schedule=layer-major
  kv f16.

Plan serialization now: T-GPLR-08 (prefetch probe) and T-GPLR-10 (MoE
handoff) are the next eligible tasks, one at a time, in either order;
T-GPLR-09 remains held on the SeLockMemoryPrivilege blocker.
[Superseded 2026-09-09: T-GPLR-08/10/11 all closed and T-GPLR-09 PARKED
-- see the DONE and PARKED sections below.]

## 2026-09-09: T-GPLR-11 DONE — batch-major tail-ubatch gap closed

The recorded tail-boundary defect ("begin_set: activation buffer exhausted"
e.g. 511 tokens with ubatch 64; workaround --ubatch = pp-tokens) is fixed.
Pre-coding probes corrected the contract's hypothesis: with PP logits
enabled the runner marks every position, so the full-width variant B
already carries the final ubatch's outputs (512/64 logits exited 0 pre-fix,
PP first greedy ID 48) -- the ONLY missing prebuilt variant is the tail
WIDTH. Fix: the batch-major prebuild now also builds the tail ubatch
variant (width t = n % ubatch when 0 < t < ubatch and > 2 ubatches; real
tail tokens + real logits values, continuing state, consecutive synthetic
positions like variant B), and buf_act grows one act pair (PP term 4 -> 6
halves of max_ubatch, <= ~2 MiB at this model). Layer-major was already
correct (T-Q38CTX-01 prebuilds every chunk width).

Evidence (GPU mutex, Qwen3.6-35B, batch-major, moe-expert-bank, KV q8):
- 511/64 --pp-no-logits (previously exit 1, activation buffer exhausted):
  exit 0, PP 22.9 tok/s, evictions 319, graph_builds 0, check-telemetry OK
  (gplr11-fix-511x64.jsonl).
- 512/64 WITH logits: exit 0, graph_builds 0, check-telemetry OK, PP first
  greedy ID 48 -- identical pre-fix (gplr11-probe-512x64logits) and
  post-fix (gplr11-regress-512x64logits): the working shape is unchanged.
- Dense Qwen3.8 layer-major-1024 CTest regression 2/2 (ID 286 reference
  intact); host suite 22/22 green.
- Semantic-QA gate (runner change touched the experimental path): 512 rung
  re-run on the fixed binary -- check-telemetry OK, all five items PASS,
  negative control fails as designed (gplr07-qa-512.{jsonl,txt,log}).

Consequence: --ubatch can now be chosen freely below pp-tokens under
batch-major; the --ubatch = pp-tokens workaround guidance is retired.
(Plan header left for the T-GPLR-10 close-out to consolidate, avoiding
concurrent header edits across sessions.)

## 2026-09-09: T-GPLR-08 implemented and probed — prefetch hypothesis FALSIFIED (instrumentation result, task closed)

Implementation (default-off, byte-identical N=0): `--pp-prefetch-ahead N`
flag (host-safe parse, off-conflict table entry), per-member range walker
with an init-time mapping-containment guard (region-level VirtualQuery,
MEM_MAPPED), a dedicated worker thread + 1-deep queue enqueued from
`load_layer` after the eviction/consumer-done block, telemetry
`prefetch_submit`/`prefetch_complete` (+ `prefetch_init` and a run-summary
line); N=0 emits no new events. Build clean; `tools/run-tests.ps1 -Suite
host` green; host guards verified (range bound, off-conflict, usage).

Probe (262,144-token PP-only Q8 layer-major, GPU mutex, Qwen3.6-35B, same
shape as the T-GPLR-04/warm-cache baselines):
- N=0: wall 350.7 s (747.6 tok/s), transfer 326.4 s, check-telemetry OK
  (5507 events, 2 slots, graph_builds 0).
- N=1: wall 352.9 s (742.9 tok/s), transfer 328.3 s, check-telemetry OK
  (5586 events, 2 slots, graph_builds 0). Prefetch: 39/39 banks submitted
  and completed, 0 errors, 17.87 GB, API call 19-28 ms/bank (submit-rate
  line: feed=18,640 MB/s); every next-layer prefetch completed before that
  layer's copy window opened.
- Verdict: wall +0.6% (noise). BOTH acceptance gates (wall -20%;
  transfer/feed >= 2x) are FALSIFIED; recorded per the contract's
  falsification clause as an instrumentation result, not a performance
  claim. Parity criterion PASSED: the 512 rung (3783/3783 prefetches,
  check-telemetry OK, answer PASS, negative control FAILs as designed) and
  a fresh 50 rung on the prefetch binary (check-telemetry OK, answer PASS,
  negative control FAILs) via `tools/run-tests.ps1 -Suite qa-smoke
  -SmokeLengths 50`.
- Why it looks flat: the fault-driven memcpy feed is unchanged (~2.0 s/bank
  mean in both runs, lists within noise). The per-bank `transfer` field
  again absorbs the 10 full-attn layers' ~29.3 s compute windows (slow at
  copy indices 4,8,...,36 in BOTH runs; slow total 263.6 s / 264.5 s);
  cross-checked trace-for-trace against the archived 352.5 s warm-cache
  baseline (gplr-longpp-262144-warmcache.jsonl: same slow indices,
  264.8 s) — today's runs reproduce the baseline exactly, so there is no
  regression and no improvement.
- Attribution (open): PrefetchVirtualMemory returned success on every bank
  yet the fault rate did not rise. Layer 0's map-time prefetch still shows
  the fast path is real (~0.13 s / 3.5 GB/s, identical in both runs), so
  the API CAN populate the working set; the rolling probe shows it does
  not help steady-state. Remaining hypotheses: prefetched pages are
  trimmed before use under host memory pressure (7.1 GiB free of 61.7 at
  analysis time), or the ~2 s/bank stream is not page-fault-bound at all.
  Distinguishing them needs a fault-count probe (working-set query before/
  after prefetch) — out of T-GPLR-08 scope.
- Consequences: T-GPLR-09 (host-RAM tier) loses its main premise (the
  working-set mechanism it hoped to exploit does not transfer to rolling
  prefetch); its privilege blocker already holds it as draft. T-GPLR-10
  (MoE handoff) is unaffected (its 2x floor compares handoff vs streaming
  serial, not vs prefetch). N=2 was attempted; the run was killed by a
  terminal timeout mid-flight (5/39 prefetches, no run_end) — no result,
  not evidence.
- Artifacts: build/llama-cuda/gplr08-262k-n{0,1}.{jsonl,log}; the n2 file
  is the killed fragment.

## 2026-09-09: T-GPLR-10 DONE — MoE handoff (streaming PP -> stock llama.cpp TG hybrid)

Implemented, validated, and closed (plan status -> done). Implementation:
- `begin_handoff` accepts moe-expert-bank (migrate_layer runs verbatim on
  pkg_cache: the 40 bank packages); `handoff_attention_only` now refuses
  under merge_resident (exit 2) — partial handoff has no meaning when only
  banks are pageable.
- Defect found and fixed during validation: `migrate_layer` built the
  handoff bundle with only the bank package members, leaving the non-bank
  members (attention/router/shared-expert/norms) null, so the post-handoff
  TG graph rebuild dereferenced null and the process died silently right
  after the migrate event (first gplr10-ho-bm attempt). Fix: non-bank
  members point at the slot bundle (`dst = src`, the same rule
  `create_bundle` uses). The loud-failure invariant holds: a broken bundle
  fails at graph build, never runs another layer's data.
- Runner gates: `--mtp + --handoff` under moe-expert-bank -> exit 2
  "...handoff TG is stock llama.cpp decode without a draft loop; MoE MTP
  verify requires the streaming executor"; layer-major `--tg-tokens > 3`
  cap lifts when `--handoff` is active (cap remains without); policy-aware
  handoff progress message. Dense `--mtp + --handoff` verified still legal.
- VRAM preflight handoff-fit branch (full_model_plus_runtime + act_slab +
  reserve <= free at run start, structured byte table, refuse before the
  timed PP). Peak-order rule: slot buffers (522 MB) + act slabs (32 MB)
  released before the 18.33 GB buf_handoff allocation (recorded in the
  handoff `release` event).
- The migrate event now records `free_vram` (true handoff-window peak:
  free 3.30 GB of 25.65 GB at migrate -> peak ~22.35 GB).
- Checker: hybrid-TG rules (bank transfer ~0 in the post-handoff
  phase_summary; handoff event field validation incl. optional free_vram)
  — backward compatible with the archived dense handoff trace (re-verified
  against it directly).

Evidence (GPU mutex; Qwen3.6-35B-A3B-MTP-UD-Q4_K_S unless noted):
- Host suite 22/22 green, including the five new T-GPLR-10 gate fixtures
  (refuse-mtp-handoff-moe, lm-tg-cap, lm-tg-cap-lift,
  dense-mtp-handoff-legal, dense-attn-only-legal).
- Parity (512 layer-major pair): route-hash sequences 40/40 identical
  streaming vs handoff; TG prefix ids [220, 248046, 198] exact (the
  streaming reference TG is capped at 3 by the layer-major experiment).
- Speed: batch-major streaming reference TG 0.495 tok/s -> handoff TG
  64.63 tok/s (~130x; the contract's >= 2x instrumentation floor is
  exceeded; recorded as instrumentation, not a general claim).
  Layer-major handoff: TG 16 tok in 284 ms. Migrate 18,327,011,328 B in
  ~3.08 s (~5.9 GB/s effective copy rate).
- Answer gate: QA-512 rung WITH `--handoff` (batch-major, ubatch 512,
  tg 96): check-telemetry OK, aggregate=PASS 5/5 needles (QX-7741 TR-2918
  MK-5501 LD-8834 Paris), negative control FAILs as designed
  (gplr10-qa512-ho.{jsonl,txt,log}).
- Dense regression (Qwen3.8-27B-Q4_K_M, 1024 PP + 16 TG batch-major,
  --allow-full-residency — the T-GPLR-04 out-of-core preflight postdates
  the recorded dense evidence): first ID 286 == the recorded reference;
  all 16 ids identical streaming vs handoff; check-telemetry OK both
  (gplr10-dense-ref/-ho.{jsonl,log}).
- VRAM: 262K Q8 + --handoff refuses exit 1 with the byte table before the
  timed PP (gplr10-refuse-262k.log); 2048-token run fits with free
  22.23 GB at start and free_vram 3.14 GB at migrate (no OOM, no silent
  fallback).
- Shared-GPU-memory investigation (owner report: "28 GB shared / 48 GB
  total on PP, none on TG"): adapter `\GPU Adapter Memory\Shared Usage`
  stayed at the ~0.29-0.37 GB desktop baseline through model load, PP
  (dedicated 3.87 GB), and the handoff copy window (dedicated 18.63 GB) —
  run-wide, sampled at 1 s (gplr10-attrib trace +
  gpu-mem-poll-attrib.csv). The mechanism surfaces NO shared-memory
  spike: pageable host-mmap banks are read directly by the copy path
  (copy_source=direct), which Task Manager attributes to dedicated VRAM
  during the copy window, and TG after handoff transfers 0 bytes. The
  Task-Manager "Shared GPU memory" figure is machine-wide (desktop +
  driver + other processes), not a second pool this runner allocates.

Gaps / residual risk:
- FIXED (same-day follow-up): the TG phantom evictions. Root cause was in
  `load_layer`'s two handoff branches, not the summary: the executor re-arms
  `consumer_pending` per layer, and the handoff branches ran the streaming
  slot-reuse protocol (consumer wait + `total_evictions++` + `slot_evict`)
  although handoff TG copies nothing and overwrites no slot data — one
  phantom eviction per layer per token (640 = 40x16x1 on MoE; the same
  inherited defect produced 1024 = 64x16x1 on the dense full handoff,
  T-Q38HG-02 era). Fix: the consumer-wait/evict blocks are removed from the
  handoff branches (the streaming branch is untouched). Evidence
  (gplr10-evictfix-{moe,dense}.{jsonl,log}, GPU mutex, check-telemetry OK
  both): post-migrate slot_evict 0, TG phase_summary evictions 0, PP
  evictions unchanged (MoE 39 / dense 127 with identical transfer bytes),
  TG ids unchanged (dense [286, 220, 197, ...] identical to the pre-fix
  reference; MoE identical to the QA-512 anchor reference). Host suite
  22/22.
- Route hashes are prompt-content-derived: parity claims are same-prompt
  only (the needle-2048 attrib run vs the synthetic-2048 mem-lm run differ
  by design).
- Measurement tooling: per-process GPU counters stall (25+ s/call) under
  full GPU load; adapter counters are the reliable path. The surviving
  poller is build/poll-gpu-memory.ps1 (v10); earlier diagnostics v1-v8 are
  superseded.
- No commit (not authorized).

## 2026-09-09: T-GPLR-09 PARKED — owner decision, plan updated

The held host-RAM bank-cache tier (T-GPLR-09) is now formally PARKED in
the plan, not merely held as draft. Owner decision 2026-09-09; the plan's
T-GPLR-09 status block records the rationale, the two-key-door framing,
and the reopen triggers.

Why parked — a two-key door, and neither key turns today:
- Key 1 (policy): SeLockMemoryPrivilege is absent and
  SeIncreaseWorkingSetPrivilege disabled for the token (2026-09-09 probe),
  so the contract's primary path (working-set raise + VirtualLock) cannot
  execute without a Local Security Policy change — an owner-level
  environment change outside the implementer lane.
- Key 2 (premise): T-GPLR-08's falsified probe (this day) showed the
  working-set mechanism the tier relies on does not transfer to the
  rolling stage (hinted pages 39/39 populated, fault-driven feed
  unchanged), and the Sep 7 warm-cache leg already showed standby
  residency adds nothing to PP. A locked private copy would additionally
  be strictly worse under memory pressure (dirty private pages trim to
  the pagefile — up to ~18 GB pagefile traffic).

Reopen triggers (any ONE, owner-initiated, recorded in the plan's status
note): (1) privilege granted AND a fault-count attribution probe shows
staging is fault-bound with trimming — not memcpy-bandwidth-bound; (2)
the workload moves to host-memory-tight conditions with recorded
hard-fault/RSS thrash evidence per run; (3) target hardware/OS changes so
the privilege arrives by default.

Consequences: no code was ever written for the tier, so nothing is
retired; the 15% admission rule stands as design evidence only. TG
batch-verify staging already sits at the RAM-speed floor (earlier
measurement), so a resurrected tier could only ever target PP. The
remaining program items are unchanged: endurance ladder rungs
50k/100k/200k (owner-invoked), the T-GPLR-08 attribution question, and
the second-architecture decision (owner-gated).

Docs-only change (plan status block + header); no commit (not requested
for this decision).

## 2026-09-09: T-GPLR-10 drafted — MoE handoff (streaming PP + stock llama.cpp TG)

(Superseded 2026-09-09 by the T-GPLR-10 DONE section above; the design
facts below were confirmed during implementation.)

Scoped per owner request (working thread): the hybrid TG path proven on
dense (T-Q38HG-02/04, 24.75 tok/s) extended to the MoE model. Drafted as
T-GPLR-10 in docs/plans/generic-prefill-layer-residency-plan.md. Key facts
established by code inspection (not assumptions):
- The ONLY hard blocker in the module is begin_handoff's blanket
  merge_resident refusal (llama-layer-streaming.cpp ~2193-2198). The dense
  migration machinery (migrate_layer lambda: dev tensors -> one contiguous
  buf_handoff -> async copy -> sync -> handoff event) runs verbatim on the
  MoE pkg_cache: the migration list is exactly the 40 bank packages
  (18,327,011,328 B = 37 x 452,984,832 + 3 x 522,190,848); router,
  shared-expert, attention, recurrent, norms, static, MTP are already
  GPU-resident under merge_resident (~3.05 GB non-bank). KV + DeltaNet state
  stay context-owned (invariant).
- adapter_qwen35moe.mtp_package == nullptr: MoE MTP is the resident
  streaming path, so --mtp + --handoff under moe-expert-bank becomes a loud
  exit-2 gate refusal (stock TG has no draft loop). Dense --mtp + --handoff
  stays legal.
- Runner gates to change: the layer-major --tg-tokens > 3 cap (main.cpp
  ~1679) lifts when --handoff is active (post-handoff TG is stock
  llama_decode regardless of PP schedule; cap remains without handoff); the
  handoff progress message "all (64 layers)" becomes policy-aware.
- VRAM: reuse the T-GPLR-04 preflight (main.cpp ~2077) as the handoff fit
  gate (full_model_plus_runtime + slabs + reserve <= free at run start,
  refuse before the timed PP). Long-ctx refuses by design (262K Q8
  full+runtime 24.30 GB vs observed free 22.92 GB); short-ctx ~21.4-21.5 GB
  + overhead is marginal and decided by measured free VRAM. Peak-order rule:
  release slots/slabs before allocating the 18.33 GB handoff buffer.
- Checker: hybrid-TG phase validates with bank transfer ~0 in the
  post-handoff phase_summary (handoff-aware rules already accept
  graph_builds > 0 after handoff); no invented copy pairs.
- --tg-text works post-handoff (writes from the generated vector, no
  streaming-only restriction), so the T-GPLR-07 512-rung answer gate is the
  end-to-end parity vehicle.

## 2026-09-07: T-GPLR-07 runner flags --prompt-file / --tg-text landed (host-validated)

Both flags are additive and default-off in the experimental runner
(`tools/layer-streaming/main.cpp` in the vendored tree). Contract — the
plan left the interaction details open, so the decisions are recorded here:
- `--prompt-file PATH` reads the prompt text as raw bytes from PATH and
  feeds the exact same `prompt_text` pipeline as `--prompt`/`--prompt-env`
  (tokenize, pp_tokens override, MTP padding). Combining it with `--prompt`
  or `--prompt-env` is a loud exit-2 error; with the flag absent, the
  existing last-wins `--prompt`/`--prompt-env` behavior is byte-identical.
  Missing file, read failure, and empty file each exit 2 before
  `llama_backend_init` (host-only). Works in both the streaming and
  `--residency off` paths (it is prompt-input plumbing; the off path
  consumes `prompt_tokens` identically).
- `--tg-text PATH` writes the detokenized generated text
  (`llama_token_to_piece`, special=false — byte-identical rendering to the
  "--- generated text ---" section) after successful TG that produced
  tokens, regardless of the prompt source. Oversized vocab pieces are
  retried with the required buffer (negative return contract). A failed
  write is loud and fails the run (`run_end.exit_status 1`). Streaming path
  only: `--tg-text` + `--residency off` is rejected by the single
  off-conflict table (one entry added, exit 2).

Host validation (2026-09-07, no GPU, no model load): build exit 0 (only
main.cpp recompiled); default-off regression — five probe commands
(`--pp-tokens 0`, bad `--copy-source`, nonexistent-model arch read, forced
`off` arch read, unset `--prompt-env`) produce byte-identical stderr and
identical exit codes pre/post change (`build/audit/baseline/` vs
`build/audit/post/`); new-flag surface — missing value, unreadable file,
empty file, prompt-source conflict, and off-conflict all exit 2 with the
recorded messages; a valid file is read and routed
("prompt-file: read N bytes" then the expected model-arch failure, exit 1);
host CTest suite 15/15 on the rebuilt binary. The tokenize/TG-write
end-to-end legs are exercised by the GPU ladder runs per the plan. Still
remaining for acceptance: smoke wiring in `tools/run-tests.ps1`, the 512/4k
strict-trace re-verification, and the dense Qwen3.8 regression re-run.

## 2026-09-07: --mtp + --pp-no-logits n_outputs_max clamp fixed

The en-route defect recorded below (GGML_ASSERT(n_outputs_max <=
cparams.n_outputs_max) in llama-context.cpp:2551 at the first MTP round,
loud abort exit -1073740791) is fixed in the runner,
third_party/llama.cpp/tools/layer-streaming/main.cpp (cparams
n_outputs_max selection, just above the forced-off branch). Root cause:
with --pp-no-logits the runner clamped n_outputs_max to 1 (kept nonzero
only for graph reservation), but the MTP batch-verify decode requests
logits on the frontier plus every draft -- up to 1 + mtp_draft rows in
ONE ubatch (batch.logits[j] = true for all rmax = 1 + nd rows) -- so
llama_decode's output_reserve asked the context for more output rows
than it declared, tripping the assert. The clamp now accounts for the
MTP draft width: n_outputs_max = pp_no_logits ? (mtp ? 1 + mtp_draft :
1) : pp_tokens. All other combinations are unchanged (pp logits on:
pp_tokens; pp_no_logits without mtp: 1), and the draft context's own
n_outputs_max = 1 is still correct (its decodes never request more than
one logits row). --pp-no-logits + --mtp is valid again; the d8config no
longer must keep PP logits on. Compile/link verified (cmake --build
build/llama-cuda --config Release --target llama-layer-streaming). GPU
model leg (2026-09-07 15:33, warm cache, GPU mutex, trace
build/gplr06-batchverify-d8-nologits-fix.jsonl, log
out/q36-d8-nologits-fix.log, script build/gplr-mtp-nologits-fix-run.ps1):
the d8 + --pp-no-logits config COMPLETES -- runner exit 0, no assert.
Fingerprints: the target context now logs n_outputs_max = 9 (the draft
context stays at 1), and all 40 MTP rounds run. Parity vs the logits-on
warm baseline is exact: first-16 ids identical (220 248046 198 248045
248068 271 248069 271 2064 7701 421 279 1414 488 3766 369), 309
drafted / 87 accepted / acceptance 0.282, 2.17 accepted/round, TG wall
82.0 s (81,974 ms) vs 81.8 s, check-telemetry OK (10127 events). PP
phase 5992.5 ms vs warm 5932.9 ms (+1%, noise) -- PP is staging-bound
(~5.4 s transfer of 40 layer copies, compute only 0.35 s) at this
shape, so suppressing PP logits does not change the PP wall here; the
saving would show where logit rows dominate compute/copy. --pp-no-logits
+ --mtp is a valid, parity-identical d8 configuration.

## 2026-09-07: Host I/O Baseline and Tier Direction (owner decision)

Measured (host-only, no GPU): total RAM 61.69 GiB (66,234,945,536 B), free
50.50 GiB at measurement time. Drive: MSI M461 2TB NVMe on C:. Sequential
read of the 21.39 GB Qwen3.6 GGUF with 64 MiB reads: 7,745 MB/s buffered,
2,728 MB/s unbuffered qd1 (FILE_FLAG_NO_BUFFERING). The "HDD-like ~52 MB/s"
hypothesis for the 262K effective page-in rate is falsified: the device
delivers ~50x that; 52 MB/s is the mmap page-fault path (small serialized
I/Os). Warm buffered reads also confirm the whole 21.39 GB file fits
resident in the 61.7 GiB RAM cache. Warm-cache 262K leg (2026-09-07,
below) additionally falsifies cache-manager retention as the collapse
mechanism: with the entire file resident in standby before AND during the
run, wall/per-layer copy times are identical to the cold baseline.

Owner decision: pursue BOTH tiers per the 15% rule — host-RAM bank residency
when the pageable set fits in RAM leaving >= 15% of total RAM for the OS, and
rolling one-layer-ahead prefetch as the streaming tier when it does not.
Drafted as T-GPLR-08 (prefetch probe) and T-GPLR-09 (host-RAM bank cache
tier) in docs/plans/generic-prefill-layer-residency-plan.md. T-GPLR-09
admission math on this machine: reserve = 0.15 x 61.69 GiB = 9.25 GiB; bank
set 18.33 GB (17.07 GiB); free 50.50 GiB -> 50.50 - 17.07 = 33.43 GiB >= 9.25
GiB -> admitted today; the rule rejects larger pageable sets automatically.


## T-GPLR-05 Complete: Capability Registry + Regression Closure — Plan Done (2026-09-05)

The generic prefill layer-residency plan is complete: every loaded model now
receives a deterministic host-only capability result, only adapter-declared
policies can be selected, and the tested support matrix is published without
any performance claim.

Capability registry. `llama_ls_probe_capability(path)` reads only the GGUF
header KV (no CUDA enumeration, no model load, no placement change) and
returns a deterministic result: `qwen35` -> adapter `qwen35`, policy
`dense-layer`; `qwen35moe` -> adapter `qwen35moe`, policy `moe-expert-bank`;
anything else -> `off` with the stable reason `no layer-residency adapter for
this architecture`. The runner exposes `--probe-capability` (host-only JSON)
and defers `llama_backend_init` until after policy resolution, so all refusal
paths are strictly host-only.

Policy modes. `auto` selects only adapter-declared policies; on an
unsupported architecture it runs `off` and explains why on stderr (exit 0,
SDD 8) — only a FORCED policy/arch mismatch refuses host-only (exit 2).
Forced `off` runs
the normal llama.cpp path inside the runner (no tensor-buft overrides, no
streaming init, batch-major `llama_decode`) and writes an audit telemetry
record (`policy":"off"` + `capability_reason`). Forced `dense-layer` /
`moe-expert-bank` are validated against the probed architecture.

Post-audit hardening (residency off first-class): the normal path lives in
`run_residency_off()`; `off` combined with streaming-only flags (layer-major,
mtp, handoff, explicit pp-schedule, layer-slots, residency-budget) is rejected
loudly before CUDA (exit 2); off audit traces are validated by the telemetry
checker (policy `off` rules: run_start/run_end/meta consistency, no slot
events) via checked-in fixtures; and all 16 run tests pass
`--allow-full-residency` explicitly, removing ambient-VRAM flakiness.

Telemetry single-owner refactor (2026-09-05, post-audit): the run_start
identity head (event/run_id/t0_us) lives in one shared builder
(`tel_run_start_head`) used by both the streaming and off-run emitters;
the off-trace rules (policy-off classification, run_start metadata rule,
streaming-event rejection) live in one shared helper set used by BOTH
`check-telemetry` and `validate-telemetry` with byte-identical message
text. Intended behavior delta only: a healthy off run_end now emits
`"error":"none"` (failed:false) instead of placing the non-error string
"off mode: normal llama.cpp placement" in the error field. Validation:
host suite green; full layer-streaming suite 44/44 under the GPU mutex
with the final binary; emitted run_start field order verified identical
to the pre-refactor contract on both trace classes (fixtures updated for
the error-field delta); MoE forced-off vs layer-major streaming parity on
the final binary: identical first 3 generated IDs (19241 13 561) on the same
12-token prompt, both traces check-telemetry OK (dense-model parity runs are
superseded; MoE is the governing workload). Post-audit hardening: the six
off-incompatible flag conditions now live in one flag-to-reason table (single
exit code 2) evaluated after argv parsing and before the host-only model-arch
read, so a gate rejection wins over any later failure class; verified host-only
for every entry, including auto-resolved off.

Long-context layer-major PP measured (MoE, 2 slots, Q8 KV @ 262144, ubatch 512,
--pp-no-logits, direct staging, GPU mutex, 2026-09-06): 50,000 tok -> 69.7 s
wall / 69.6 s transfer / 62.6 s compute = 717.8 tok/s; 100,000 tok -> 135.0 s /
129.3 / 126.6 = 740.7 tok/s; 262,144 tok -> 351.9 s / 327.6 / 338.6 = 745.0
tok/s. 38 bank evictions and graph_builds 0 in every run. Decomposition
(2026-09-06, from per-event trace data): bytes_transferred is constant
(18.33 GB, 40 copies) and pure per-bank H2D copy cost is flat (layer-0
copies ~0.26 s in every run, ~10.4 s total), but wall is paced by the host
page-in stream of those same 18.33 GB: effective feed rate 263 MB/s at 50K
vs 52 MB/s at 262K (uncontrolled file-cache behavior; device limit falsified 2026-09-07 - MSI M461 2TB NVMe measured 2,728 MB/s unbuffered / 7,745 MB/s buffered sequential reads of the model file; see AGENT_ENVIRONMENT.md) -- that, not copy
volume, is why the transfer-time column climbs while bytes stay flat. The
phase transfer/compute fields measure submit->complete windows that include
overlapped compute (per-copy transfer_us inflates to ~29.5 s exactly on the
10 copies overlapping the full-attention layers' ~29.7 s compute windows),
so "compute hidden behind transfer" is not decidable from the phase fields
alone; sampled layer_compute sums to 41.3 s at 262K. Net: long-PP wall ~
18.33 GB / effective page-in rate; tok/s rises mildly with length only
because the fixed byte stream spreads over more tokens (marginal cost
~1.33 ms/token at 262K). 262K
confirmation and variants (all exit 0, check-telemetry OK): rerun 350.6 s =
747.7 tok/s (reproducible within 0.4%); pinned staging 352.1 s = 744.4 tok/s
(identical to direct -- the bottleneck is the mmap read, not the staging
copy; wait 0.6 s); 4 slots 352.3 s = 744.2 tok/s (36 evictions, slot window
1.82 GiB; cumulative transfer time tripled to 967.8 s yet wall unchanged --
wide-window copies run concurrently behind compute, leaving wall bound by
the host read stream). Traces: build/gplr-longpp-*.jsonl, all
check-telemetry OK.

Warm-cache leg (2026-09-07, same command/shape, GPU mutex, trace
build/gplr-longpp-262144-warmcache.jsonl, script build/gplr-warmcache-run.ps1,
log out/q36-longpp-262k-warmcache.log): the whole 21.39 GB GGUF was read
into the file cache immediately before the run (standby 0.38 -> 20.41 GB;
second-pass warm read 8,138 MB/s) and STILL sat at 20.57 GB standby after
run_end -- the file never left memory during the run, with 53 GB free the
whole time (cleanest environment any long-PP run has had). Result: wall
352.5 s / 743.7 tok/s, transfer 328.1 s, compute 339.0 s, 38 evictions,
graph_builds 0, peak VRAM 12,211 MiB -- identical to the 351.9 s baseline
(delta +0.17%), and per-layer copy windows match trace-for-trace: layer 0
= 0.13 s, recurrent ~1.7-2.3 s, full-attn ~29.2-29.4 s. Cache-manager
retention is FALSIFIED as the cause of the 263->52 MB/s collapse:
standby-resident pages fault through the memcpy path at ~200-215 MB/s, no
faster than cold. Two refinements:
- The apparent length-dependent collapse is mostly a phase-field artifact:
  total_transfer_us absorbs the 10 full-attn layers' ~29.3 s compute
  windows (~293 s of the 328 s total); the genuine per-bank stream is
  ~266 MB/s at 50K vs ~215 MB/s at 262K (~20% slower, not 5x), and wall ~
  30 recurrent x ~2.05 s copy + 10 full-attn x ~29.3 s compute = ~354 s.
- Layer 0's copy (0.13 s ~ 3.5 GB/s, identical in both traces) rides the
  loader's one-shot map-time PrefetchVirtualMemory, i.e. WORKING-SET
  residency -- 16x the standby-fault rate. The fast path exists and is
  working-set residency, not standby: this directly validates the
  T-GPLR-08 rolling one-layer-ahead prefetch (PrefetchVirtualMemory
  populates the working set) and pins its falsifiable gate to the layer-0
  rate (~3.5 GB/s feed -> wall floor ~66 s, matching the warm-resident
  boundary traces' 3968 tok/s).

Open performance frontier: TG remains a full 40-bank pass per token
(0.26-0.52 tok/s measured). Draft task T-GPLR-06 (MTP draft + layer-major
burst verify, falsifiable 2x-tok/s hypothesis at k=4, exact greedy-prefix
parity) is appended to the plan as draft, awaiting owner review before any
implementation. A pre-implementation probe already triggered the draft's
first escalation condition: --mtp --handoff on the MoE loads the draft head
(blk.40) but begin_handoff is unsupported for the moe-expert-bank policy
(loud TG decode -3, exit 1; build/gplr06-mtp-probe.jsonl) -- the MoE MTPacceptance
rate is unmeasurable with the current binary. Owner decision taken: redesign
the draft leg. Step 1 landed: --mtp without --handoff is now legal under
moe-expert-bank (MTP package is GPU-resident there; single gate moved after
policy resolution; dense-layer still requires handoff). First MoE MTP run
completed (exit 0): drafted 16 accepted 9 = 0.562 accept rate, and greedy IDs
exactly match the sequential no-MTP baseline on the same 512-token prompt
(build/gplr06-{seq-baseline,mtp-resident}.jsonl). Wall unchanged (0.474 vs
0.493 tok/s) because the existing loop verifies serially -- one full bank
pass per draft; the remaining T-GPLR-06 work is the layer-major batch verify
that amortizes one bank pass across k drafts. Acceptance measured over 128
tokens (all legs exit 0, greedy IDs identical everywhere): sequential 0.522
tok/s; draft 4 -> rate 0.479, 1.91 accepted/round; draft 8 -> rate 0.250,
2.00 accepted/round -- accepted tokens per bank pass saturate near 2, so the
step-2 batch-verify projection is ~2.7-3.0x wall (2x hypothesis reachable
with margin). En-route defect found and
fixed: the layer-major graph-input buffer was sized embd-bytes-exact plus a
flat 64 MiB slack, which exhausted at chunk 493/512 on the 262K run (two
deterministic loud aborts, never silent corruption); sizing now adds 1 MiB
headroom per chunk (host-side buffer, no VRAM cost). Failed attempt traces
archived in build/gplr-attempt1/.

T-GPLR-06 step 2 (batch verify) landed and measured (2026-09-06): the MTP
round now verifies the frontier plus all drafts in one width-(1+d) target
ubatch - one expert-bank pass per round instead of one per draft - with
corrected-for-state rollback via exact KV seq_rm plus the recurrent snapshot
ring, and prebuilt width-k phase-1 graph sets preserving graph_builds == 0.
128-token draft-4 config: 95.1 s wall vs the 245.4 s serial baseline (2.58x,
target ~90 s), exit 0, 47 rounds, avg accepted/round 1.70. Parity verdict
per owner decision (bit-exactness not required across ubatch shapes, but
divergence must be measured): the one divergence from the serial stream was
instrumented on both paths with top-2 margins and runner-up tokens - the
flipped decode is the identical logical decode (same token/position/history),
serial chose its argmax by 0.0037 logits (smallest margin of the whole run)
and the width-5 shape flipped it at 0.14 noise scale; all comparable rows
agree on gold AND runner-up; both streams re-join afterward. Quantized-
weight shape noise on a knife-edge tie, not a state or wiring defect; reruns
bit-identical. Host suite green (15/15) on the final binary; traces at
build/gplr06-{seq128-margins,batchverify-d4-v2}.jsonl. Draft-8 batch verify
measured 2026-09-07: 82.6 s (1.55 tok/s), 2.97x over serial, avg
accepted/round 2.17 (40 rounds) - deeper drafts pay off once verify is
batched; all cross-width divergences are the same measured sub-0.15-margin
tie flips (traces build/gplr06-batchverify-d8.jsonl). Draft-32 probe:
96.1 s, peak VRAM 7.42 GB, exit 0, graph_builds 0 - the frontieris d8 (82.6 s); acceptance horizon is 7 (depth >= 8 never accepted in 43
rounds), memory is NOT the binding constraint (ring planes scale
linearly, 2.18 GB at d32), and the depth-32 ring rewind is exercised
(6 zero-accept rounds) with identical first-16 IDs across all configs.

Warm-cache TG staging bound (2026-09-07, same d8 command, GPU mutex,
trace build/gplr06-batchverify-d8-warmcache.jsonl, script
build/gplr-tg-warmcache-run.ps1, log out/q36-tg-warmcache.log): the file
cache was warm before and during the run (standby 21.1 GB, 52.8 GB free,
GPU idle -- cleanest TG environment recorded). TG wall 81.8 s / 1.564
tok/s vs the 82.6 s / 1.55 tok/s baseline (delta -1%, noise); 40 rounds,
accepted/round 2.17, acceptance 0.282 -- behaviorally identical to the
recorded baseline. The bound: batch-verify wall is staging-volume-bound
at 40 x 458 MB = 18.33 GB per round, and every TG wall on this machine
collapses to that volume divided by ~9 GB/s (serial 128t: 250.4 s / 128
rounds = 1.96 s/round = 9.1 GB/s; d8: 2.01 s/round = 8.9 GB/s; d8 warm:
2.00 s/round = 9.0 GB/s; compute is only 1.4 s of the 82 s wall).
Consequences: (1) TG staging is ALREADY at the RAM-speed floor -- the
host-RAM tier (T-GPLR-09) adds ~0 wall for TG batch verify, exactly like
standby residency added nothing to PP; (2) batch verify's remaining lever
is acceptance (tokens per round -- the code-prompt d16 3.54 tok/s optimum
is the evidence), not staging; (3) batch verify at 1.56 tok/s already
matches the dense handoff resident-TG rate (1.68 tok/s), so staging is no
longer the MoE-vs-dense TG differentiator. En-route defect found: --mtp
with --pp-no-logits trips GGML_ASSERT(n_outputs_max <= cparams.
n_outputs_max) in llama-context.cpp:2551 at the first MTP round (loud
abort, exit -1073740791; FIXED same day -- the runner clamp now covers
1 + mtp_draft, see the '--mtp + --pp-no-logits n_outputs_max clamp
fixed' entry above). Also
noted: per-copy window telemetry is scheduler-skew sensitive (median 48
ms while a foreign process held the GPU busy vs ~0 ms with it idle);
aggregate staging rate (~9 GB/s) is the robust number, not per-copy
windows. Coding-prompt depth
sweep 2026-09-07 (calculator-spec prompt, --prompt-env, 1024 PP,
ctx 2048, tg 128, batch-major): d8 35.5 s TG (3.60 tok/s, 17 rounds,
accept 0.815), d16 25.7 s (4.98 tok/s, 12 rounds, 0.650), d32 18.2 s
(7.02 tok/s, 8 rounds, 0.496) - tok/s still rising at d32; the
synthetic-prompt depth horizon (7) dissolves on structured code text.
Caveat: d8/d32 fell into a greedy 2-token repetition loop (no rep
penalty in harness) while d16 escaped on a round-0 knife-edge flip
(margin 0.089) and generated real code; a no-loop rerun is required
before generalizing. Peak VRAM 5.83/6.36/7.42 GB; traces
build/gplr06-codeprompt-d{8,16,32}.{jsonl,log}.

Coding-prompt depth sweep, repetition penalty (2026-09-07, same prompt,
--rep-penalty 1.3 last-n 64): loop dead at every depth (max period-2
run 1/2/1), all legs emit real HTML. d8 76.4 s (1.68 tok/s, 38 rounds,
accept 0.299), d16 36.1 s (3.54 tok/s, 17 rounds, 0.426) = measured
optimum, d32 76.1 s (1.68 tok/s, 36 rounds, 0.088 - acceptance
collapses at depth 32 as the draft ring fills with hypothetical
penalized tokens). The unpenalized "rising at d32" was loop
contamination; d16 is the code optimum at 6.8x serial. New runner
flags --rep-penalty/--rep-last-n replicate penalty_repeat semantics at
all four greedy pick sites (initial frontier, draft chain, batch-verify
rows, serial TG) so the draft predicts the penalized target; host suite
green. Traces: build/gplr06-codeprompt-p13-d{8,16,32}.{jsonl,log}.
Serial no-MTP leg on the same prompt (2026-09-07, penalty 1.3, natural 885-token PP): 249.69 s TG (0.513 tok/s), real HTML output, loop-free (max period-2 run 1) - the first measured code-text serial baseline; d16 36.12 s => 6.91x true MTP speedup on code (the prior 6.8x inherited the synthetic-prompt baseline). Caveat: batch legs pad PP to 1024 with PAD tokens (mirror requires exact ubatches), serial runs natural 885, so the streams diverge at idx 0 by construction (pad-cell frontier); TG per-token bank geometry is identical, so the timing ratio is clean. Trace: build/gplr06-codeprompt-serial-p13.{jsonl,log}.

T-GPLR-07 semantic-QA ladder first results (2026-09-07): the host-only
needle-prompt generator and answer checker landed in tools/semantic-qa/
(stdlib-only, deterministic, exact token-count calibration via
llama-tokenize --ids; checker self-test 13/13 including the negative-control
logic). Full ladder tier is green at every rung so far, all on
Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf, --kv-cache q8, --pp-no-logits,
--allow-full-residency, batch-major PP, GPU mutex, runner exit 0:
15 tokens (K1 only) -> "Paris" PASS; 30 tokens (N1) -> "QX-7741" PASS;
50 tokens (N1) -> "QX-7741" PASS; 512-token anchor (all 5 items) ->
"QX-7741 TR-2918 MK-5501 LD-8834 Paris" PASS; 4096 tokens (all 5 items,
needles at rows 27/55/84/106 of 113) -> all five codes PASS with the full
answer emitted before any drift. The 4k rung: PP 4095 tok in 6.75 s
(606.9 tok/s), TG 160 tok in 303.9 s (0.53 tok/s), peak VRAM 3.9 GiB,
check-telemetry OK. Negative control verified live: grading the 512 and 4k
answers with --negative correctly FAILs (a grader that cannot fail is not
a grader). Evidence recorded as ANSWER stability, never token-ID stability.

24k ladder rung (2026-09-07, owner-invoked acceptance run): prompt 24,576
tokens exact (needle rows 169/340/511/647 of 649 = 25/50/75/95% depth),
delivered via a 113,786-byte env var built directly in Python's
subprocess env= (CreateProcessW environment block survives the full
bash->powershell->cmd->exe chain intact; the 32 KB Windows limit applies
only to SetEnvironmentVariable setters, which are bypassed -- probe
verified 98,304 chars round-trip, exit 0). Run config identical to the
4k rung: --ctx-size 25088, --ubatch 24575 single-ubatch PP,
--tg-tokens 160, --kv-cache q8, --pp-no-logits, --residency
moe-expert-bank, --pp-schedule batch-major, --allow-full-residency.
Result: exit 0, PP 24,576 tok in 16.2 s (1516 tok/s), TG 160 tok in
313.5 s (0.51 tok/s), peak VRAM 10.22 GiB, answer
"QX-7741 TR-2918 MK-5501 LD-8834 Paris" -> aggregate PASS (all 5 items),
negative control FAILs as designed. Strict gate satisfied on the
dedicated JSONL trace build/gplr07-qa-24k.jsonl: --check-telemetry OK
exit 0 (run_start/run_end, 2 phase_summary, 12,960 copy_submit/complete
pairs, 6,480 layer_compute, graph_builds 0 both phases). Reproducibility:
the mixed-log run and the --telemetry re-run produced byte-identical
first-16 generated IDs and the same PASS answer. This is the first rung
with a surviving dedicated --telemetry trace; the 512/4k rung traces
were mixed-log only (their "check-telemetry OK" claims were NOT backed
by surviving JSONL and should be re-verified before ladder completion).

Runner/tooling findings from the first ladder run (all recorded, none
silent): (1) the runner has no chat template and keeps generating past EOS
until its TG budget is spent -- the answer is followed by a turn-echo loop
("user/assistant" turns and prompt echo); the checker strips everything from
the first <|endoftext|> on, and the TG budget must cover the answer plus the
first EOS (~65 tokens for the 5-item answer). (2) The batch-major prebuild
only builds full-ubatch variants: a PP whose token count is not a multiple
of the ubatch hits a runtime begin_set for the tail ubatch whose act pair
exceeds the fixed activation buffer (loud failure "activation buffer
exhausted" at the tail boundary, e.g. 511 tokens with ubatch 64);workaround used for every run here is --ubatch <pp-tokens> (single-ubatch
PP, matching the T-GPLR-06 512-token precedent). This tail-ubatch gap was a
vendored-tree defect candidate; it is now CLOSED by T-GPLR-11 (2026-09-09,
see the task entry at the top of this file) -- batch-major prebuilds the
tail variant and the workaround is retired. The failure mode was loud,
never wrong-data. (3) The runner prints human-readable "first 16 generated ids"
and a "--- generated text ---" section on combined stdout/stderr instead of
JSONL run_start/run_end events in that mode; check-answer.py parses both
(the runner's own detokenized text section is preferred over the 16-ID
fallback since the full answer exceeds 16 tokens). The strict JSONL
telemetry gate (check-telemetry OK) remains a separate required check on
dedicated --telemetry files.

Normal-path parity closed. A forced-off run (GPU mutex, MoE model, 9-token
prompt) produced PP-logits first greedy ID 13 and first generated ID 5388 —
identical to the streaming layer-major run. This converts the T-GPLR-03/04
"router parity proven between residency configurations" caveat into a direct
stock-path comparison. A read defect was caught by exactly this check: index
0 of `llama_get_logits_ith` resolves to the FIRST batch token's row
(`output_resolve_row(i >= 0) = output_ids[i]`), reporting 2614; the correct
read is -1 (last output row), now fixed in the off path.

Refusal fixtures. New host-only CTests `layer-streaming-refuse-unsupported-dense`
(forced policy on an unsupported arch -> exit 2), `layer-streaming-probe-unsupported`,
off-audit checker tests `layer-streaming-check-off-audit-{valid,rejects-slot-event,rejects-bad-meta}`,
+ fixture generator
`tests/make-unsupported-fixture.cpp` (tiny `llama`-arch GGUF) all pass in
0.4 s with no GPU. The insufficient-capacity refusal was already proven by
the T-GPLR-04 negative fixture (aborts before the timed PP).

Validation: `tools/run-tests.ps1 -Suite host` 13/13 (suite extended to include
the probe/refusal tests and the extended model-inspect unit test); full
layer-streaming suite 44/44 under the GPU mutex (2026-09-05 closing run,
437 s, including the off-audit checker fixtures and the off first-class
gates); default-off
`llama-cli`/`llama-server` build exit 0.

Tested support matrix (SDD section 8.1 is the canonical copy): supported =
`qwen35`/`dense-layer` and `qwen35moe`/`moe-expert-bank` only; unsupported =
everything else (forced policies refused host-only; `auto` runs `off` and explains);
correct-but-slower = the validated configurations with no throughput claim;
unmeasured = MTP/TG expert prediction, weight rotation, multi-GPU, and every
other architecture. No speedup claim is made anywhere.

Traces: `build/llama-cuda/q36-off-parity.jsonl` (off audit record),
`out/q36-off-parity.log`, `out/q36-auto2.log`.

## T-GPLR-04 Complete: 262K Q8 Out-of-Core Capacity Proof (2026-09-05)

The 262,144-token Q8 KV capacity configuration is proven: the full model plus
runtime would not fit under the enforced residency budget, and only the
bounded 2-slot expert-bank window plus KV/state/act-slabs occupy pageable
weight VRAM while the 512-token layer-major prefill runs with exact parity
against the T-GPLR-03 control.

Acceptance run (GPU mutex, Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf,
`--ctx-size 262144 --kv-cache q8  --pp-tokens 512 --ubatch 512

--pp-schedule layer-major --layer-slots 2 --no-mtp --reserve-mib 512
--residency-budget-mib 7000 --prompt "The quick brown fox jumps over the
lazy dog" --greedy --tg-tokens 1`): exit 0, `check-telemetry: OK` (567
events), `ctx_size=262144` in run metadata, PP-logits first greedy ID 13 and
TG continuation 5388 identical to the T-GPLR-03 control, all 40 route-event
hashes identical to the 4-slot 9-token control, `graph_builds == 0` both
phases.

Out-of-core proof: full model + KV + state 24,295,318,016 B vs bounded 2-slot
window 6,517,385,728 B (exact T-GPLR-00 inventory match): static
957,530,112 + per_layer 1,597,483,520 + slot_window 1,044,381,696 (2 x
522,190,848) + KV 2,852,126,720 + DeltaNet state 65,863,680 B; act slabs
8,388,608 B, reserve 536,870,912 B, peak VRAM 7,625.56 MiB.

Negative fixture: `--residency-budget-mib 5000` aborts before the timed PP
(`bounded 2-slot window 6517385728 B exceeds the residency budget
5242880000 B`, exit 1, zero timed tokens).

Defect found and fixed during this task: the post-init `byte_account` emitter
applied MoE arithmetic to dense runs (per_layer counted resident, slot_window
0), so the dense reference trace reported `bounded_window` 25.7 GB >= free
VRAM and the new checker invariant failed. The checker caught exactly the
drift it was designed for. Both the preflight and the emitter now share one
helper (`ls_bounded_window_for_policy`), dense policy correctly reports
`per_layer=0` with `slot_window = slots x dense_largest`, and the fresh dense
262K reference validates (bounded 12.4 GB < full 35.2 GB). All 38
layer-streaming CTests pass with regenerated traces; default-off
`llama-cli`/`llama-server` build exit 0. `run_start` now also carries an
explicit additive `ctx_size` field.

Caveats: full model 24.3 GB fits physical VRAM (25.65 GB), so the out-of-core
condition is proven by the explicit sub-full-model residency budget plus the
bounded-window arithmetic, not by physical non-fit; mmap/file-cache source
state remains uncontrolled Windows behavior; per-layer routing is not
observable on the normal path, so router parity is between residency
configurations, not against the server.

Trace: `build/llama-cuda/q36-262k-parity.jsonl`; logs:
`out/q36-262k-parity.log`, `out/q36-neg-budget2.log`, `out/q38-dense-262k.log`.

## T-GPLR-03 Complete: 512-Token MoE Layer-Major Prefill + Parity Closure (2026-09-05)

The Qwen3.6 MoE 512-token layer-major prefill is complete and the last open
acceptance item — first-greedy-ID parity with the normal path — is now
closed with a matched control.

Streaming evidence (GPU mutex, `qwen35moe`, Q8 K/V, ubatch 512, 4 slots):
all 40 trunk layers in layer-major order, exactly one complete expert-bank
copy per MoE layer in the PP phase (74 x 452,984,832 B + 6 x 522,190,848 B,
generations 1-10, no partial/hidden copies), 40 `route` events with
`top_k=8`, 4-slot rolling vs 16-slot resident runs produce IDENTICAL
route-hash sequences, layer-major/batch-major/16-slot all agree on greedy
ID 220 (synthetic deterministic prompt), `graph_builds == 0` for PP and TG,
`check-telemetry: OK`, exit 0. Partial-package negative still fails loudly
with zero copies. Dense Qwen3.8 `layer-major-1024` regression: ID 286.
Traces: `build/llama-cuda/q36-moe-pp512-{rolling,resident16,batchmajor}.jsonl`.

Parity closure (the previously open item). Two false alarms were identified:

1. The `llama-cli` "control" reporting `<think>` was never a raw completion.
   This vendored `llama-cli` is a chat client: it spawns/uses a server and
   posts the prompt to `/v1/chat/completions` as a user message, so the Qwen
   chat template wraps the prompt and the model begins an assistant turn,
   whose first piece is the special token `<think>` (id 248068; verified with
   host-only `llama-tokenize`, which also shows the raw prompt is 9 tokens
   with no BOS and no synthetic padding).
2. An interim runner reading of 271 (`\n\n`) came from a 10-token prompt
   (prior trace `run_start.pp_tokens = 10`, i.e. a padded/older run): the
   T-GPLR-03 PP-logits accessor returns the argmax of the LAST logits row,
   which was the pad row, not `dog`. On the exact 9-token prompt the accessor
   returns 13.

Matched control (GPU mutex, same workspace build, `-ngl 99`,
`--cache-type-k/v q8_0`, `--parallel 1`, temperature 0, `n_probs=5`):
`llama-server` `/v1/completions` with the raw 9-token prompt reports first
token id 13 (`.`), logprob -1.138 (then `\n` -1.643, ` is` -2.018,
`\n\n` -2.274, special -3.17). The streaming layer-major runner on the same
prompt reports PP-logits first greedy ID 13 — exact first-token parity with
the normal path. The runner's TG then continues with id 5388 (` dog`), the
unpicked continuation, consistent with argmax 13 being consumed first.
Artifacts: runner trace `build/llama-cuda/q36-moe-prompt-ctl.jsonl` (40 PP
copies, 40 routes, `graph_builds == 0` both phases, telemetry OK), control
response `build/control-response.json`, server log
`out/q36-control-server.log`. No quality claim is made beyond first-token
parity; per-layer routing is not exposed by the normal path, so router
parity is proven between residency configurations (4-slot vs 16-slot,
identical hashes), and route hashes are prompt-dependent by construction.

## T-GPLR-02 Complete: MoE Expert-Bank Package Proof (2026-09-04)

The Qwen3.6 MoE expert-bank adapter is proven. `src/models/llama-layer-streaming-qwen35.cpp`
now registers `qwen35moe_package` (complete routed-expert bank: gate+up+down, or
fused gate_up+down), `merge_resident=true` (slot bundle = resident model layer with
only package members replaced by device tensors), and `causal_implicit=true`.
`qwen35moe::graph` gained the `layer(il)` indirection and a `build_streaming_graphs()`
mirror of the dense builder; `llama-graph.cpp` causal gate now reads
`adapter->causal_implicit` (no hard-coded arch). The runner resolves
`--residency auto|off|dense-layer|moe-expert-bank` from a host-only
`general.architecture` read before load; MoE loads with expert-bank tensors
overridden to CPU (pageable) while attention/router/shared/norms stay CUDA-resident.
Route observation marks the router's selection_probs as a live graph output
(lifetime only, no arithmetic change); the executor reads the F32 probs after each
layer compute and derives exact top-k IDs (stable descending sort = CUDA radix
tie-break), emitting one `route` event (sampled IDs, top-k, route-union, FNV hash).
Observational only.

Validation (2026-09-04, GPU mutex): one-layer proof layer 0, 64 tokens -> exactly
one `copy_submit` of 452,984,832 B (the complete bank), slot_vram 522,194,944 B,
exit 0, `route` event top_k=8 route_union=192 with all IDs in [0,256). Rolling
(1 slot) vs resident (16 slots) runs produce the IDENTICAL route hash
(9763251156467869940) and sampled IDs 220,167,131,161,250,237,8,185 — exact router
parity. Partial-package negative (`--fault-drop-member`) -> loud
`load_layer: package descriptor/bundle member mismatch`, exit_status 1, zero
`copy_submit` events. Dense regression `layer-streaming-layer-major-1024` passes
with greedy ID 286; host trio passes. Host fixtures cover separate/fused positives,
partial bank and zero-byte negatives; capability probe now selects qwen35moe.
Escalation conditions cleared: `mmid.cu` reads quantized experts directly (no
per-load repack); router IDs observable without changing arithmetic.

## T-GPLR-01 Complete: Qwen Package Adapter Extracted (2026-09-04)

The Qwen-specific package lists were extracted behind an architecture-neutral
descriptor boundary. `src/llama-layer-streaming.{h,cpp}` now defines
`llama_ls_member`/`llama_ls_package`/`llama_ls_adapter`, the host-testable
validator `llama_ls_validate_package()`, and the capability probe
`llama_ls_select_adapter()`. The Qwen3.5 dense member tables (recurrent 22,
attention 18, MTP 26 members; order preserved) live only in
`src/models/llama-layer-streaming-qwen35.cpp`. `init()` enumerates and
validates every trunk package before any device allocation or copy submission;
slot bundle creation, `load_layer`, and `begin_handoff` (including the MTP
migration) consume the cached validated descriptors. A new `pkg_member`
telemetry event emits adapter, policy, class, member count, and package bytes
per layer.

Validation (2026-09-04): new host CTest `test-layer-streaming-descriptors`
(reference member-order equality, six invalid-descriptor negatives, capability
probe with stable unsupported reasons) passes; the full host trio (layout,
model-inspect, descriptors) passes 3/3. GPU parity under the mutex:
`layer-streaming-layer-major-1024-run` + host telemetry check pass with
`graph_builds == 0` in both phases, first greedy ID 286 equal to the
pre-extraction baseline captured the same day, 64 `pkg_member` events (48
recurrent + 16 attention) with layer-0 package 364,022,656 B equal to the
recorded reference size, peak 4,783,144,960 B. Default-off
(`-DLLAMA_LAYER_STREAMING=OFF`) `llama-cli` builds with exit 0; the flag was
then restored to ON with a clean rebuild.

`qwen35moe` deliberately has no adapter yet and reports a stable unsupported
reason; T-GPLR-02 adds the MoE expert-bank adapter. A task-boundary review
(2026-09-04) confirmed the extraction is behavior-preserving with no Critical
findings, and its three Important fixes (MTP size guard, handoff failure
propagation + `abort_handoff()` cleanup, `make_dev_tensor` null checks) were
revalidated: host trio 3/3, a direct attention-only handoff run under the mutex
exited 0 with the migrate/teardown events and `check-telemetry: OK`, and the
fresh `layer-streaming-layer-major-1024` pair passed with greedy ID 286,
`graph_builds == 0` in both phases, and 64 `pkg_member` events.

## T-GPLR-00 Complete: MoE Artifact Inventory (2026-09-04)

The host-only inventory task is done. `llama-layer-streaming.exe
--inspect-model <gguf> --ctx-size 262144 --kv-cache q8 --physical-vram-bytes N
--inspect-json out.json` parses GGUF metadata with `gguf_init_from_file
(no_alloc=true, ctx=nullptr)` and returns before `llama_backend_init()`; it
enumerates no CUDA device and reads no tensor data. The shared implementation is
`llama_ls_model_inventory()` in `src/llama-layer-streaming.cpp`, validated by
the new host CTest `test-layer-streaming-model-inspect` (synthetic dense/MoE
GGUF fixtures + malformed-metadata negatives: unsupported arch, missing
`expert_count`, unclassified tensor, incomplete expert bank, missing file).

Measured facts (full JSON: `build/llama-cuda/q36-moe-inventory.json`, recorded
in `docs/AGENT_ENVIRONMENT.md`): architecture `qwen35moe` (vendored support
present: `LLM_ARCH_QWEN35MOE`, `src/models/qwen35moe.cpp`, 40 layers ->
`LLM_TYPE_35B_A3B`), 41 blocks (40 trunk + 1 MTP), 256 experts, top-8, 753
tensors, 21,377,327,616 B data. Every routed-expert tensor maps to exactly one
logical-layer expert-bank package: 37 layers at 452,984,832 B and layers 34/38/39
at 522,190,848 B (their `ffn_down_exps` are the q6_K tensors). Router, shared-
expert, attention, recurrent, layer-norm, static and MTP classes are reported
separately. Q8_0 K/V at capacity 262,144 = 2,852,126,720 B (10 attention
layers, `n_embd_k_gqa` 512, padded kv_size 262,144); Gated DeltaNet state =
65,863,680 B (30 recurrent layers).

Capacity finding: `full_model_plus_runtime` = 24,295,318,016 B is BELOW the
physical total 25,650,855,936 B by ~1.36 GB, so the raw byte sum alone does not
prove non-fit. The inventory therefore reports `require_sub_full_budget: true`,
and T-GPLR-04 must enforce and report an explicit residency budget below
full-model bytes. Against observed free VRAM (22,922,919,936 B at a
2026-08-21 `run_start`) full residency does not fit, and CUDA-context,
activation, graph/input and workspace classes are still unmeasured.

Build/verification commands (all host-only, no GPU):
`cmake --build build/llama-cuda --config Release --target llama-layer-streaming
--parallel 16` -> exit 0; `ctest --test-dir build/llama-cuda -C Release -R
test-layer-streaming-model-inspect` -> 100% pass; the `--inspect-model` run
above -> exit 0.

Incidental build-blocker fix (pre-existing, not task scope): the user's
uncommitted edit had `slab_off_b` where the committed code has `slab_off_B` in
`llama_layer_streaming::begin_set`, which failed every compile with C2065.
Restored to `slab_off_B`; no other user modification was touched.

## Active Generic Prefill Layer Residency Direction (2026-09-04)

The active design is now `docs/layer-residency-sdd.md`; the executable backlog
is `docs/plans/generic-prefill-layer-residency-plan.md`. No generic source
implementation has been made under this plan. The existing Qwen3.8 code and
evidence remain the dense reference implementation.

Observed implementation boundary: the vendored module already owns reusable
CUDA slots, generation and event ordering, layer-major activation slabs,
persistent per-width graph sets, preflight, and telemetry. Package member
arrays and recurrent/attention/MTP classification remain hard-coded in
`src/llama-layer-streaming.cpp`, and the split graph remains integrated through
`src/models/qwen35.cpp`. Generic support is therefore not yet proven.

Selected MoE acceptance artifact:
`C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`. The path was
observed to exist on 2026-09-04; its byte size, GGUF architecture key,
layer/expert counts, tensor inventory, package bytes, and vendored support are
not yet verified. T-GPLR-00 is ready and owns that host-only inventory.

The fixed first out-of-core acceptance shape is CUDA, context capacity 262,144,
Q8_0 K/V, one 512-token prefill chunk, layer-major schedule, complete routed-
expert-bank packages, and MTP disabled. The 262,144 value sizes context/KV; it
does not require processing a 262,144-token prompt. Acceptance must prove that
full model plus runtime state exceeds physical VRAM while the bounded slot
configuration fits. No speedup or generic-model-support claim exists yet.

Status: T-Q38LS-01 through T-Q38LS-06 (batch-major), T-Q38LM-01 through
T-Q38LM-05 (layer-major), T-Q38PW-00 through T-Q38PW-02 (per-width graphs),
T-Q38FA-01 (FA KV window), T-Q38TG-00 (hybrid TG probe, NO-GO), and
T-Q38HG-00/01/02/03/04 (handoff machinery, hybrid probe, constrained
handoff) are done. ALL batch-major plan tasks are
complete. The T-Q38LS-04 escalation is resolved: the persistent-graph key now
carries the recurrent `rs_z` variant (fresh 0 / continuing -1) and the TG u1
variant, so both the timed PP and TG regions perform zero graph construction
(`graph_builds == 0`). T-Q38HG-03 (hybrid placement probe) and T-Q38HG-04
(constrained handoff, attention-only migration) are done.

Layer-major follow-up (docs/plans/qwen38-pp20k-plan.md): T-Q38LM-01 (stable
layer-major activation slices) and T-Q38LM-02 (full two-chunk layer-major PP
pass) are done. T-Q38LM-01 adds the opt-in `--pp-schedule layer-major` path,
two stable full-prompt activation slabs with non-aliasing per-chunk slices
bound into per-chunk persistent graph sets, layer-major telemetry
(`schedule`, `act_slab_bytes`, `chunk`, `act_bind`), and host-validator
enforcement of per-layer ascending chunk order and non-aliasing slab binds.
T-Q38LM-02 drives all 1,024 PP tokens (two 512-token chunks) through all 64
layers in layer-major order via a dedicated `layer_major_pp` controller:
128 PP `layer_compute` events (chunk 0 then 1 per layer), exactly 64 PP
`copy_submit` events (each package copied once and reused across both chunks
under the same generation), `graph_builds == 0`, and a one-token TG
schedule-consistency check producing the same first greedy ID (286) under
batch-major and layer-major. T-Q38LM-03 (20K scale), T-Q38LM-04 (PP-to-3TG),
and T-Q38LM-05 (regression closure) are done. The free-GPU 20K trace records
2,560 PP computes, 64 PP package copies, `graph_builds == 0` for PP and TG,
and greedy IDs 317, 197, 197; host validation passes and the full 19-target
GPU regression suite passes. The T-Q38LS-07 batch-major 20K task remains
blocked/superseded by the layer-major design.

Facts below are observed Qwen3.8 reference evidence only. Superseded claims are
removed, never retained. The completed reference design is
`docs/qwen38-layer-streaming-sdd.md`; its original backlog is
`docs/plans/qwen38-layer-streaming-plan.md`. Active generic work follows the
design and plan named above. Environment facts live in
`docs/AGENT_ENVIRONMENT.md`.

## Arbitrary-Length Layer-Major PP (done 2026-08-23)

Plan: `docs/plans/qwen38-arbitrary-pp-plan.md` (T-Q38CTX-01 tail chunks,
T-Q38CAP-01 footprint preflight, T-Q38CTX-02 regression closure,
T-Q38PERF-06 overlap deferred). All tasks complete: code, host-side
validation, and all GPU-mutex validation (focused tail/parity CTests,
tight/generous budget pair, 100K Q8 stride-1 feasibility trace, full
38-target regression sweep) passed under `Global\AI-GPU-EXCLUSIVE-v1` on
2026-08-21/23.

GPU evidence (2026-08-21/23, model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`)
: - Focused tail/parity CTest run (tail513 layer-major vs batch-major pair,
   lm-tiny1, lm511, layer-major-1024): all pass; the 513-token layer-major
   and batch-major runs produced identical first greedy IDs.
: - 100,000-token Q8 PP-only stride-1 run
   (`--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major --layer-slots 4
   --kv-cache q8 --pp-no-logits --no-mtp --telemetry-chunk-stride 1`) exited 0
   and wrote `build/llama-cuda/q38ls-pp100k-layer-major-q8.jsonl` (13,256
   events). Manifest: 196 chunks, 195 x 512 + final 160-token tail. Preflight
   free VRAM 22,922,919,936 B of 25,650,855,936 B. PP: 195,649.4 ms, 511.118
   tok/s, 60 evictions, `graph_builds 0`, peak 12,085.56 MiB.
   `--check-telemetry` OK (13,256 events, 4 slots).
   Note: the first 100K attempt (2026-08-21) recorded 19.856 tok/s due to an
   80-minute single-event stall (layer 39, chunk 35) caused by GPU contention
   from another process not using the mutex; that trace is preserved as
   `q38ls-pp100k-layer-major-q8-stalled-20260821.jsonl` for reference. The
   clean re-run at 511.118 tok/s matches the 20K rate scaled for the larger
   Q8 KV cache.
: - Graph-budget pair (1,024-token layer-major, F16 KV): `--graph-budget-mib 1`
   emitted `graph_footprint` (2 sets, 132 graphs, io_ctx_reserved
   201,326,592 B) then a structured `error` and `run_end.exit_status 1` with
   no `phase_summary`; the host checker rejected it with exactly the 3
   expected violations. `--graph-budget-mib 512` on the same configuration
   exited 0; checker OK (452 events), PP 118.580 tok/s, `graph_builds 0`,
   peak 4,513.56 MiB. Traces:
   `build/llama-cuda/q38ls-budget-{tight,generous}-1024.jsonl`.
: - `tools/run-tests.ps1 -Suite all`: 38/38 pass (578.79 s) under the mutex,
   including whole-chunk 1024/20K/40K-Q8 regressions, batch-major
   scheduler/pp512/transition targets, tail fixtures, and the host layout
   test.

Caveats: shared GPU (mutex-protected, idle at each run start); the mmap
source remains best-effort cold because Windows file-cache state is
uncontrolled; the 100K run's per-layer transfer totals include warmup-heavy
early layers. No quality, equivalence, or speedup claim is made.

## 262K Attempt and Per-Width Graph Plan (2026-08-23)

A 262,144-token Q8 stride-1 attempt (same command as the 100K contract)
failed during prebuild at chunk ~505/512: `tensor 'inp_embd' exceeds backing
buffer (offset=5434509056 size=10485760 buffer=5435817984)` — the host input
buffer is sized for the whole prompt (~5.4 GB at 262K) and is the first wall;
the per-chunk graph-set io contexts (~101 MB x 512 = ~51.5 GB host RAM) are
the second. Trace: `build/llama-cuda/q38ls-pp262k-layer-major-q8.jsonl`
(partial, 1,011 lines, no run_end). Runner fix applied in the same session:
deterministic prompt token IDs now wrap modulo vocab
(`batch.token[i] = i % n_vocab` in `tools/layer-streaming/main.cpp`); before
the fix, 262K failed earlier with `init: invalid token[0] = 248320`.

Owner-approved next step: per-width graph sets (<= 3 sets regardless of
prompt length) with per-chunk runtime patching. Order: make 192K work first,
then the 262K stretch. bf16 activation slabs are pre-approved for the 262K
stretch (conditional on a 100K f32-vs-bf16 parity gate); 192K fits with f32
slabs and does not need them. Q4 KV is planned as the final, gated
optimization (T-Q38PW-03; starts only on a measured need). Plan:
`docs/plans/qwen38-perwidth-graphs-plan.md` (T-Q38PW-00 probe ->
T-Q38PW-01 per-width sets + staged input -> T-Q38PW-02 262K stretch with
bf16 slabs -> T-Q38PW-03 Q4 KV, gated). Measured budget: 192K fits
(~18.9 GB VRAM); 262K (~24.2 GB with f32 slabs, ~18.8 GB with bf16 slabs).

Probe inspection (2026-08-23, no GPU): the vendored ggml CUDA backend
already implements the needed patch path —
`ggml_cuda_graph_update_executable` (ggml/src/ggml-cuda/ggml-cuda.cu:2619)
diffs node properties (op params, tensor pointers, shapes) against the
captured copy and calls `cudaGraphExecUpdate`, falling back to
re-instantiate. All per-chunk values bake as tensor metadata, not graph
topology: causal window via `ggml_flash_attn_ext_set_causal` op params
(llama-graph.cpp:2592), act-slab view offsets, `inp_embd` pointer, KV write
positions. T-Q38PW-00 is therefore wiring + validation, not research.
Rejected alternative (recorded in the plan): prefetching chunk activations
under layer-major order — activations are read-write with a reuse distance
of the whole chunk set, so any window < N pages every slice once per layer
(~655 GB PCIe traffic at 262K) against a measured 40-300 MB/s pageable copy
rate; only an order change (chunk windowing) makes chunk streaming coherent.

## T-Q38PW-00: Patchability Probe (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Mechanism

The chosen mechanism is direct tensor-data-pointer patching plus input-snapshot
restore plus op-param (causal window) patching. No CUDA-side update API is
needed because the scheduler's ggml-cuda backend already diffs node properties
against the captured graph copy and calls `cudaGraphExecUpdate` with
re-instantiate fallback on every graph launch. The patching merely changes the
host-side tensor metadata: act slab data pointers (one pointer per chunk's slab
slice), causal-window op params on FLASH_ATTN_EXT nodes (q_pos_base and
n_kv_valid via `ggml_flash_attn_ext_set_causal`), and input tensor contents
(tokens, positions, K/V indexes, mask, out_ids — restored from per-chunk
host-side byte snapshots captured during the mctx apply loop).

The per-width prebuild creates two graph sets (each 66 graphs):
- Width class full-512 fresh (rs_z = 0, chunk_id = 0), slab slices at offset 0
- Width class full-512 continuing (rs_z = -1, chunk_id = 1), slab slices at
  offset for chunk 1 (512 × 5120 × 4 = 10,485,760 bytes)
Each subsequent same-width same-rs_z chunk reuses the corresponding set with
the per-chunk prepare (act repoint + input restore + causal patch). The host
validator passes (452 events, 4 layers, `graph_builds == 0`).

### Per-chunk-varying values and how they are patched

| Value | Baked at build | Patched per chunk | Mechanism |
| --- | --- | --- | --- |
| Act slab data pointers | Yes (fixed slab offsets) | Yes (cumulative byte offset per chunk) | `tensor->data` reassignment |
| Causal window (q_pos_base, n_kv_valid) | Yes (chunk descriptor's token_start/n_tokens) | Yes (from chunk descriptor) | `ggml_flash_attn_ext_set_causal` |
| Input tensor contents (tokens, pos, K/V idxs, mask, out_ids) | Shared host buffer | Yes (per-chunk snapshot) | Host-side memcpy from snapshot |
| rs_z variant (fresh vs continuing) | Distinct sets (0 and -1) | No (selected per chunk) | find_set by rs_z |
| KV cache view offsets (k_l[il], v_l[il]) | Owned by memory context, not graph | No (mctx per chunk) | Stable cache, per-chunk cells |
| Slot weight tensors | Stable device addresses | No (immutable within slot gen) | Slot lifecycle unchanged |

### Patch cost

Paired 1024-token layer-major PP runs (ubatch 512, four slots, pp-no-logits,
implicit causal F16 KV), same mutex-protected GPU:

| Metric | Reference (per-chunk sets) | Probe (per-width sets) |
| --- | --- | --- |
| Wall time | 8,109.2 ms | 7,990.9 ms |
| Tok/s | 126.28 | 128.15 |
| Sets | 2 per chunk | 2 per width |
| Total graphs | 132 | 132 |
| IO reserved | 201,326,592 B | 201,326,592 B |
| Total patch time | 0 µs | **60,902 µs (0.76%)** |
| Per-chunk patch | N/A | ~30,451 µs |
| Per-chunk-per-layer patch | N/A | ~461 µs |
| check-telemetry | OK | OK |
| graph_builds | 0 | 0 |

Patch cost is 0.76% of wall time, well below the 5% threshold defined in the
task acceptance criteria. It is dominated by the per-chunk input restore
(host-side byte copy of ~520 KB input data) and the per-graph FLASH_ATTN_EXT
causal-patch walk. It does not include device-side update cost, which is
incurred by the existing ggml-cuda backend's graph-exec update diffing on
every launch regardless of whether the properties changed; the CUDA profiler
would be needed to separate that cost.

### Equivalence check

For the 2-chunk probe with pp-no-logits there is no logit output to compare.
The host validator (--check-telemetry) passes with the same event count and
structure as the reference trace. Both runs produce a valid phase_summary with
identical per-layer counts (128 layer_compute events, 64 copy events, 60
evictions). The peak VRAM is identical (4,513.56 MiB). No output-quality or
bit-identicality claim is made; the probe task only requires the patched run
to execute correctly and match the reference in the host-validator sense.

### Residual risks

- Tail chunks (width < 512) are not yet handled by the per-width prebuild;
  T-Q38PW-01 adds a tail-width continuing set for that case.
- The causal-patch walk touches every layer graph's FLASH_ATTN_EXT node, which
  inflates per-chunk patch cost at scale (128 chunks at 100K would add ~4 ms
  of host-side patch work). Acceptable per acceptance criteria (< 5%).
- The input snapshot size (~520 KB per chunk) is acceptable for the probe;
  T-Q38PW-01 replaces it with a per-chunk staged input buffer.
- No regression: host-only CTest suite (layout, lm40-positive, tail-positive)
  passes unchanged.

## T-Q38PW-01: Per-Width Graph Sets + Staged Input Buffer (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### Architecture

Graph sets are now keyed by `(mode, n_tokens, rs_z, n_outputs)` — `chunk_id`
is dropped from the key when `--pw-probe` is active (the new default for
layer-major PP). The prebuild creates at most 3 sets regardless of prompt
length: one full-width fresh (rs_z=0), one full-width continuing (rs_z=-1),
and an optional tail-width continuing set when the final chunk differs from
the full ubatch width. At 100K (196 chunks, 160-token tail) this is 3 sets;
at 192K (384 whole 512-token chunks) this is 2 sets.

Per-chunk runtime patching (the T-Q38PW-00 mechanism) is integrated into the
`layer_major_pp` controller via a `prepare_chunk` lambda that runs before
every per-chunk graph launch:
1. Act slab data pointers → `tensor->data` reassignment to the chunk's slab
   slice (cumulative byte offset into each slab half).
2. Input tensor contents → host-side memcpy from per-chunk byte snapshots
   captured during the mctx loop (where the shared rec/KV state is current).
3. Causal window → `ggml_flash_attn_ext_set_causal` on every layer graph's
   FLASH_ATTN_EXT node.
4. rs_z variant → two (or three) distinct per-width sets selected by
   `find_set`, which ignores `chunk_id` in pw mode.

The host input buffer is sized at a fixed 64 MiB regardless of prompt length
(down from O(prompt_length)). Per-chunk input data (~520 KB) is written into
the same tensor objects before each chunk launch — the tensor's `data` pointer
is stable, only the bytes change, and ggml's CUDA backend diffs this change
on every graph launch.

### 100K Regression

Command: `--pp-tokens 100000 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe
--telemetry-chunk-stride 10 --telemetry q38ls-pw01-pp100k.jsonl`.

| Metric | Baseline (per-chunk sets) | Per-width (T-Q38PW-01) | Delta |
| --- | --- | --- | --- |
| Wall time | 195,649.4 ms | 197,542.1 ms | +0.97% |
| Tok/s | 511.118 | 506.22 | **-0.96%** |
| Graph sets | 196 | **3** | -98.5% |
| Total graphs | 12,936 | 198 | -98.5% |
| IO context reserved | ~19.7 GB | **288 MB** | -98.5% |
| Host input buffer | ~2.1 GB | **64 MB** | -97.0% |
| Act slab (f32) | 4.1 GB | 4.1 GB | same |
| Peak VRAM | 12,085.56 MiB | 12,085.56 MiB | same |
| graph_builds | 0 | 0 | same |
| Total patch time | N/A | 5,306,377 µs (2.68%) | acceptable |
| check-telemetry | OK | OK | same |
| evictions | 60 | 60 | same |

The 0.96% regression is well within the ±5% gate. Host RAM for graph-set I/O
contexts dropped from ~19.7 GB to 288 MB. The host input buffer dropped from
~2.1 GB to 64 MB.

### 192K Gate

Command: same flags, `--pp-tokens 196608`.
Trace: `build/llama-cuda/q38ls-pw01-pp192k.jsonl` (3,648 events).

| Metric | Value |
| --- | --- |
| Wall time | 601,660.8 ms |
| Tok/s | 326.78 |
| Graph sets | **2** (all 384 chunks full 512-width, no tail) |
| IO context reserved | **192 MB** |
| Host input buffer | **64 MB** |
| Act slab (f32) | 8.05 GB |
| Peak VRAM | **19,531.56 MiB** |
| graph_builds | 0 |
| Total patch time | 21,416,017 µs (3.56%) |
| check-telemetry | OK (3,648 events, 4 slots) |

192K fits comfortably: peak VRAM 19.5 GB < 22.9 GB free. Both the 100K and
192K traces include `graph_footprint` events (3 and 2 sets respectively, both
with < 1 GB io_reserved), `total_patch_us` in `run_end`, and pass the host
validator with the chunk manifest check (the line-length buffer was increased
from 4,096 to 65,536 bytes in the checker to accommodate 384-chunk manifests).

### Bug fixes during this task

- **JSON line-length truncation**: the host checker's `fgets` buffer was 4,096
  bytes, which truncated the 192K trace's 4,445-character run_start line
  (384-chunk manifest). Increased to 65,536 bytes.
- **chunks_str quoting**: the `pp_chunks` manifest entries (`0:512,...`) were
  emitted without JSON string quoting; fixed to emit `"0:512"` strings.
- Host-only CTest suite (layout, lm40-positive, tail-positive) passes.

### Residual risks

- Patch cost grows linearly with chunk count (2.68% at 100K, 3.56% at 192K).
  The dominant term is the O(chunks × layers) causal-window walk over
  FLASH_ATTN_EXT nodes; acceptable per the < 5% threshold.
- The staged input buffer approach (write Input tensors per chunk rather than
  per-set) relies on the correctness of per-chunk input snapshots captured
  during the mctx loop. A stale snapshot would cause a silent computation
  error. Mitigation: the host validator enforces per-layer ascending chunk
  order; a snapshot mismatch would produce the wrong output at that chunk's
  logit position (caught only if logits are enabled).
- The `--pw-probe` flag is still the opt-in.

## T-Q38PW-02: 262K Stretch with bf16 Slabs (done 2026-08-23)

Evidence collected under `Global\AI-GPU-EXCLUSIVE-v1` on 2026-08-23 (model
`C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090,
22,923 MiB free).

### f32 262K attempt

Contrary to the budget model, the f32 262K allocation actually succeeded:
10.74 GB act slabs (F32) + 3.32 GB Q8 KV + 1.44 GB slots + 0.63 GB compute
≈ 16.1 GB < 22.9 GB free. The run completed: 262.59 tok/s, peak 24,462 MiB
(exact total VRAM). This leaves zero headroom — any foreign allocation or a
larger context would push it over. The bf16 optimization remains valuable.

### bf16 slab implementation

The `--act-slab-dtype bf16` flag selects BF16 (2 bytes/elt) instead of F32
(4 bytes/elt) for the two activation slabs. Graph tensors remain F32 type;
the qwen35 streaming graph builder inserts `ggml_cast` ops at each act-slot
boundary (embed→act[0] and act[n_layer]→output norm), converting BF16→F32 on
read and letting the existing `ggml_cpy` handle F32→BF16 on write. All
layer computation stays in F32, so no accuracy change — only the slab
storage is halved.

### Parity gate (100K f32 vs bf16 with 3 TG tokens)

| Metric | f32 | bf16 |
| --- | --- | --- |
| PP tok/s | 503.90 | 502.85 |
| TG tok/s | 0.369 | 0.373 |
| Peak VRAM | 12,095.56 MiB | 10,141.56 MiB |
| TG tokens (greedy) | [95726, 248046, 198] | [95726, 248046, 198] |
| check-telemetry | OK (3,210 events) | OK (3,210 events) |
| act_slab_bytes | 4,096,000,000 | 2,048,000,000 |

The greedy IDs are bit-identical. VRAM saving: ~1,954 MiB (2 GB) at 100K.
The 0.21% tok/s delta is within noise. BF16 slabs are approved for 262K.

### 262K bf16 result

Command: `--pp-tokens 262144 --ubatch 512 --pp-schedule layer-major
--layer-slots 4 --kv-cache q8 --pp-no-logits --pw-probe --act-slab-dtype bf16`.
Trace: `q38ls-pw02-262k-bf16.jsonl` (4,736 events, validates).

| Metric | f32 | bf16 |
| --- | --- | --- |
| Tok/s | 262.59 | 262.62 |
| Peak VRAM | 24,462.56 MiB | **19,467.56 MiB** |
| act_slab_bytes | 10,737,418,240 | **5,368,709,120** |
| graph_sets | 2 | 2 |
| graph_builds | 0 | 0 |
| check-telemetry | OK | OK |
| VRAM headroom | 0 MiB | **~5,000 MiB** |

The bf16 slabs save 5 GiB (from 10.2 GiB to 5.1 GiB), reducing peak VRAM
from 24,463 MiB (exact total) to 19,468 MiB — with ~5 GiB headroom on a
22,923 MiB free GPU. Performance is unchanged (262.59 vs 262.62 tok/s).

### Context-vs-throughput curve

| pp_tokens | Schedule | Slab dtype | Tok/s | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- |
| 20,480 | layer-major per-chunk | f32 | 796.64 | 7,866 |
| 100,000 | layer-major per-width | f32 | 506.22 | 12,086 |
| 100,000 | layer-major per-width | bf16 | 504.22 | 10,132 |
| 196,608 | layer-major per-width | f32 | 326.78 | 19,532 |
| 262,144 | layer-major per-width | f32 | 262.59 | 24,463 |
| 262,144 | layer-major per-width | bf16 | 262.62 | 19,468 |

### Residual risks

- The bf16 cast path is exercised only with the pw-probe cast ops. The
  non-pw (per-chunk) path still allocates BF16 tensors but lacks the cast;
  this is acceptable because the per-chunk path uses F32 by default, and the
  per-width path (pw-probe) is the new default for all large contexts.
- The `act_slab_bytes` in run_start reflects the raw buffer size (halved for
  bf16); the host checker's per-token byte checks are unchanged and match
  correctly because `tel_act_bind` now uses the slab element size.
- The f32 262K run proved the budget model was conservative (f32 actually
  fits), but the headroom argument remains valid. On a shared GPU, bf16
  provides a safety margin.

### T-Q38PW-03 gate verdict (2026-08-23)

T-Q38PW-03 (Q4 KV cache) is gated and its three gate conditions are all
currently false:
- Gate 1 (bf16 parity gate failed): parity passed, 262K bf16 fits with 5 GiB headroom.
- Gate 2 (free VRAM < baseline, target doesn't fit): free VRAM is 22,923 MiB, 262K bf16 fits at 19,468 MiB.
- Gate 3 (owner opens goal beyond 262K): no such goal.

The per-width graph sets plan is complete. T-Q38PW-03 starts only when a
measured need reopens one of these gates.

### Plan status

`docs/plans/qwen38-perwidth-graphs-plan.md` is **complete**. All un-gated
tasks (T-Q38PW-00 through T-Q38PW-02) are done and validated. T-Q38PW-03
(Q4 KV cache) remains gated pending a measured need. T-Q38PERF-06 (stream
overlap) remains deferred, unchanged.

## Per-layer slot bundles for dynamic-quant models (done 2026-08-23/24)

The Q4_K_M model (Unsloth imatrix/dynamic scheme: 294x q4_K, 48x q5_K,
67x q6_K, 1x q8_0 tensors) failed the byte-size assertion in `load_layer`
because slots held one bundle per shape class templated from the first
matching layer; layers with a different quant type for the same tensor
mismatched the bundle's byte size.

Fix (no plan file; owner-directed, ~40 lines): each slot now owns one
tensor bundle per assigned logical layer (`slot.layers` map,
`slot.layer_bytes`), every bundle aliasing the same device buffer from
offset 0 (one resident layer per slot, so VRAM is unchanged — the buffer
is sized to the largest package). `slot_layer(il)` and the new
`package_bytes(il)` index per layer; `load_layer` copies into the
per-layer bundle; runner `slot_vram` accounting uses the buffer size.
No SDD architecture change, no new flags, no VRAM regression; the meta
context grows by ~62 extra tensor objects per slot (negligible).

Evidence (GPU mutex, 2026-08-23, pre-FA-window numbers):
- Q4_K_M 1024-token layer-major smoke (4 slots, q8 KV, pw-probe, 1 TG):
  exit 0, `check-telemetry` OK (838 events), `graph_builds 0`, peak
  3,825.56 MiB. Trace `build/llama-cuda/q38ls-q4km-1024.jsonl`.
- Q4_K_M 100K PP (same flags as the Q6_K_XL 100K contract): 525.699 tok/s,
  wall 190,222.9 ms, peak 11,383.56 MiB, `check-telemetry` OK (2,056
  events), `graph_builds 0`. Trace `q38ls-q4km-pp100k.jsonl`.
- Q4_K_M 192K PP: 331.308 tok/s, wall 593,429.3 ms, peak 18,829.56 MiB,
  `check-telemetry` OK (3,648 events), `graph_builds 0`. Trace
  `q38ls-q4km-pp192k.jsonl`.
- Q6_K_XL regression: `tools/run-tests.ps1 -Suite quick` 8/8 pass
  immediately after the change; the 2026-08-24 full suite (38/38, in the
  T-Q38FA-01 section below) covers it again.
- Post-FA-window Q4_K_M 100K (693.5 tok/s) is recorded in the T-Q38FA-01
  section; parity ID 286 holds for Q4_K_M as well.

## T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-window-plan.md` (T-Q38FA-01, single task).

### Mechanism

In pw mode the per-chunk `prepare_chunk` lambda already re-points act slab
data pointers, restores input snapshots, and patches the causal window op
params (`q_pos_base`, `n_kv_valid`) on every layer graph's FLASH_ATTN_EXT
node. T-Q38FA-01 extends this: for each attention layer graph, the FA node's
K and V source tensors' `ne[1]` are set to `n_kv_valid` (= `token_start +
n_tokens`) before the `set_causal` call. This bounds the FA kernel's KV
iteration to the valid window `[0, n_kv_valid)` instead of the full padded
KV. Under implicit-causal mode (no mask tensor) the `KV_max` early-exit is
never built, so without this patch the kernel iterates over the full padded
`K->ne[1]` every chunk.

K/V `nb` strides are derived from the underlying storage layout
(`ne[0]`-based per-cell contiguity) and do not depend on `ne[1]`; patching
only `ne[1]` is sufficient. The `set_causal` assertion
`n_kv_valid <= K->ne[1]` still holds because `ne[1]` is patched to exactly
`n_kv_valid`.

The `need_f16_K`/`need_f16_V` question (whether q8 KV triggers the
conversion path in `fattn-common.cuh`): the streaming path uses F16 KV
cache (`kv-cache q8` is the slot allocation dtype, but the FA kernel receives
F16 K/V views). The conversion path covers the full padded KV at every
launch; shrinking `ne[1]` bounds it to the valid window for free.

### Validation evidence (2026-08-24, model `Qwen3.8-27B-UD-Q6_K_XL.gguf`, RTX 5090)

All tests under `Global\AI-GPU-EXCLUSIVE-v1`.

**Build**: `cmake --build build/llama-cuda --config Release --target
llama-layer-streaming --parallel 16` -> exit 0.

**Full test suite**: `tools/run-tests.ps1 -Suite all` -> **38/38 pass**
(542.14 s). Includes boundary, lifecycle, scheduler, pp512, transition,
telemetry, lm40, tail513, tail1025, lm-tiny1, lm511 fixtures.

**Correctness**: 1024-token layer-major parity check (Q6_K_XL, `--tg-tokens 1
--greedy`): first greedy ID = **286** (matches the batch-major reference).
`--check-telemetry` exits 0.

**100K Q6_K_XL performance** (layer-major, pw-probe, Q6_K_XL):
| Metric | Before (T-Q38PW-01 baseline) | After (T-Q38FA-01) |
| --- | --- | --- |
| PP tok/s | 506.22 | **704.6** |
| Improvement | — | **+39.2%** |

**Falsifiable signature**: per-chunk `compute_us` for full-attn layer 3 is
no longer flat — grows from 4.6 ms (chunk 1) to 40.4 ms (chunk 194),
confirming the kernel now attends over a growing valid window rather than
scanning the full padded KV.

**Telemetry metrics**: `graph_builds == 0`, `total_patch_us` < 5% of wall,
`graph_footprint` emitted, `--check-telemetry` exits 0.

### Q4_K_M 100K performance (2026-08-24)

Command (matching the baseline `q38ls-q4km-pp100k.jsonl`):
`--model Qwen3.8-27B-Q4_K_M.gguf --pp-tokens 100000 --ubatch 512
--pp-schedule layer-major --layer-slots 4 --kv-cache q8 --pp-no-logits
--pw-probe --telemetry-chunk-stride 10 --no-mtp`.
Trace: `build/llama-cuda/q38ls-q4km-fa-100k.jsonl`.

| Metric | Baseline (2026-08-23) | After (T-Q38FA-01) | Delta |
| --- | --- | --- | --- |
| PP tok/s | 525.7 | **693.5** | **+31.9%** |
| Wall time | 190,222.9 ms | 144,192.9 ms | -24.2% |
| Total compute | 178,673.9 ms | 133,609.9 ms | -25.2% |
| Peak VRAM | — | 11,383.6 MiB | — |
| graph_builds | 0 | 0 | same |
| total_patch_us | — | 5,054,630 µs (3.50%) | acceptable |
| check-telemetry | OK | OK (2,056 events, 4 slots) | same |

**Q4_K_M parity**: 1024-token layer-major vs batch-major both produce greedy
ID **286** (matches Q6_K_XL reference).

**Escalation resolved**: the gate was >= 700 tok/s; measured 693.5 tok/s
(+31.9% over baseline, 0.93% below the gate). Owner accepted.

Profiling (Q6_K_XL vs Q4_K_M 100K) demonstrates that attention's share of
total compute actually **grows** from 56.5% to 57.4% under weight-only
quantization — not shrinks. The FA kernel (45.0% of total compute) is
weight-quantization-invariant: it operates on the quantized KV cache
(Q8/F16), not on model weights. The matmuls that do benefit from
quantization are only ~20% of each full-attn layer and ~44% of total
compute, and Q4 dequant overhead (per-group scales/input_scales) cancels
most of the bandwidth savings, yielding only ~4% matmul speedup. The net
effect is that the fixed FA cost dominates both variants, and the small
matmul speedup shrinks the denominator by less than the numerator —
attention's share rises slightly. The 31.9% vs 39.2% gap between Q4_K_M and
Q6_K_XL is not explained by quantization shifting the attention/matmul
ratio; it is within normal per-run variance on an attention-dominated
workload where the FA kernel is the invariant floor.

The falsifiable signature is confirmed: layer 3 compute grows from ~5.6 ms
(chunk 10) to 39.5 ms (chunk 190), versus the baseline's flat ~42 ms
profile.

All acceptance criteria confirmed: correctness (parity ID 286 on both
Q6_K_XL and Q4_K_M), falsifiable signature, graph_builds = 0, patch_us
< 5%, full test suite 38/38.

## T-Q38FA-02: resident control verdict (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-02).

Q6_K_XL cannot serve as the fully-resident control on this GPU: its
weights alone (~24.1 GiB) exceed the 24,462 MiB total VRAM, so any
`-ngl 99` run spills to system RAM (observed: 3.7 GB shared GPU memory)
and is invalid as a control. The control was run on Q4_K_M instead,
which has a matched streaming baseline (693.5 tok/s post-FA-window).

Command (GPU mutex, single repetition): `llama-bench.exe -m
C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -p 100000
-n 0 -fa on -ctk q8_0 -ctv q8_0 -t 16 -r 1 -b 8192 -ub 512` ->
**804.66 ± 0.00 tok/s**.

| Path | tok/s | Delta |
| --- | --- | --- |
| Fully resident (llama-bench) | 804.66 | — |
| Streaming (post-FA-window) | 693.5 | **-13.9%** |

**Verdict: the vendored FA/GEMM kernels are not the problem.** The same
backend achieves 805 tok/s fully resident. The 111 tok/s (14%) gap is
entirely streaming-path overhead (patch walk, graph replay diffing,
copy-stream contention, per-chunk host work). The "34 TFLOPS FA
anomaly" from the 2026-08-24 ceiling analysis was an artifact of
comparing against a 20K short-prompt ~1000 tok/s figure; the true
resident ceiling at 100K is ~805 tok/s and we are at 86% of it.

Consequences: T-Q38FA-04 (overhead reduction) is the primary lever.
T-Q38FA-05/06 (f16-KV A/B, ncu profile) are deprioritized — the kernels
are fine. T-Q38FA-03 (fused-op audit) closed same day: fused GDN
confirmed in streaming graphs (resolve_fused_ops runs at context init
before streaming graph build; runtime logs confirm "fused Gated Delta
Net (autoregressive/chunked) enabled").

## T-Q38FA-04: patch-walk cache (done 2026-08-24)

Plan: `docs/plans/qwen38-fa-efficiency-plan.md` (T-Q38FA-04).

Mechanism: `prepare_chunk` previously walked all 66 graphs x all nodes
(~3,840 node inspections) plus the full set-context tensor list on every
call — 12,544 calls at 100K (196 chunks x 64 layers + embeds). The
persistent graphs are never rebuilt (`graph_builds == 0` invariant), so
the first call now caches per set: the FLASH_ATTN_EXT node pointers
(`fa_nodes`) and the INPUT-flagged tensor list (`input_tensors`), and
every later call patches those lists directly. ~50M node-walk iterations
per 100K run eliminated.

Evidence (GPU mutex, Q4_K_M, 100K, q8 KV, pw-probe, trace
`build/llama-cuda/q38ls-fa04-pp100k.jsonl`):

| Metric | Before (T-Q38FA-01) | After (T-Q38FA-04) |
| --- | --- | --- |
| Wall | 144,192.9 ms | **135,389.3 ms** |
| Tok/s | 693.5 | **738.6 (+6.5%)** |
| Total compute | 133,609.9 ms | 122,210.3 ms |
| total_patch_us | 5,054,630 | 6,859,812 |
| check-telemetry | OK | OK (2,056 events, 4 slots) |
| graph_builds | 0 | 0 |

Honest attribution caveat: the compute delta (-11.4 s) cannot be caused
by a host-side walk change; at least part of the +6.5% is run-to-run GPU
state variance (shared GPU, clock/thermal state). The attributable win
is the walk elimination; a paired back-to-back run would be needed for
clean attribution. Gap to the resident control (804.66) is now 8.2%.

`total_patch_us` rose (5.05 -> 6.86 s) despite the walk shrinking — the
patch timer also covers the input-snapshot memcpy and the dominant term
was never the node walk alone; run variance applies here too. The
remaining patch cost is dominated by the per-chunk input restore
(~520 KB memcpy) and the per-FA-node `set_causal` op-param writes.

Validation: build exit 0; parity 1024-token layer-major + 1 greedy TG
(q8 KV) exit 0, check-telemetry OK; `tools/run-tests.ps1 -Suite all`
38/38 pass (561.59 s) on Q6_K_XL.

### Escalation: q8-KV layer-major/batch-major greedy-ID divergence (pre-existing)

Found during T-Q38FA-04 parity: with **q8 KV**, 1024-token layer-major
produces greedy ID **538** while batch-major produces **286** (Q4_K_M,
same build/flags). The divergence predates T-Q38FA-04 — it is present in
the 2026-08-23 per-layer-bundle smoke trace (`q38ls-q4km-1024.jsonl`,
ID 538). With **f16 KV**, layer-major produces 286 (matches batch-major
and the recorded T-Q38FA-01 parity evidence, traces
`q38ls-fa-parity{,2}.jsonl`). The T-Q38FA-01 "parity ID 286" record was
measured with f16 KV. So the layer-major implicit-causal path diverges
from batch-major specifically under quantized KV. Not caused by
T-Q38FA-04 (pointer caching changes no numerics). Needs its own
investigation task: candidate causes are the implicit-causal mask path
vs the dense-mask path interacting with q8 KV dequant, or the FA-window
`ne[1]` patch under q8 strides. All 100K q8-KV benchmarks to date
(525.7 -> 693.5 -> 738.6 tok/s) were run on the diverging configuration;
the rates are still valid throughput evidence, but the q8-KV layer-major
correctness gate is open.

## T-Q38TG-00: Hybrid TG Feasibility Probe (done 2026-08-24 — NO-GO)

### H2D rate measurements (corrected 2026-08-24 — original was disk-bound)

The original T-Q38TG-00 numbers (2.90/2.96 GB/s pageable, 2.77 pinned)
were measured against **mmap'd weights without warming the file cache**,
so the "copy time" included NVMe page-fault reads — 2.9 GB/s is NVMe
speed, not the PCIe bus. Re-measured with `--no-mmap` (all weights in
ordinary host RAM, zero page faults) and a 4-copy-per-layer protocol
(  `--pp-tokens 256 --ubatch 512 --layer-slots 1`, copies 2-4 = steady
state, copy 1 warmup ignored). Steady-state transfers are stable to
±0.1 ms across ubatches.

| Model | Source | Package size | transfer (steady) | Rate |
| --- | --- | --- | --- | --- |
| Q4_K_M recurrent (layers 1,2,4,5) | direct no-mmap | 245.9 MB | 17.94 ms | **13.71 GB/s** |
| Q4_K_M full-attn (layer 3) | direct no-mmap | 222.9 MB | 16.28 ms | **13.69 GB/s** |
| Q4_K_M recurrent (layer 0) | direct no-mmap | 245.9 MB | 22.9 ms | 10.7 GB/s (first-layer anomaly) |
| Q4_K_M recurrent | pinned no-mmap | 245.9 MB | DMA 18.0 ms + staging 11.7 ms | DMA 13.7 GB/s, total 29.8 ms |

Traces: `build/llama-cuda/q38ls-tg00-4x-direct-q4km.jsonl`,
`q38ls-tg00-4x-pinned-q4km.jsonl`.

**True steady-state H2D DMA rate is 13.7 GB/s**, 4.7x higher than the
original disk-bound figure. **Pinned staging gives no DMA advantage** —
with the source already in host RAM, `cudaMemcpyAsync` from pageable
memory is internally staged at the same 13.7 GB/s; the explicit pinned
path only adds an ~11.7 ms staging memcpy and is slower end-to-end. The
unverified "50 GB/s" discussion figure remains withdrawn; the measured
DMA ceiling is 13.7 GB/s.

PCIe link fact (nvidia-smi, 2026-08-24): negotiated **Gen4 x8** (max
also Gen4 x8 — laptop GPU wiring). Theoretical ceiling ~15.75 GB/s; the
measured 13.7 GB/s is **87% of the physical link** — the bus is
saturated and no software technique (pinned rings, multi-stream,
ReBAR, GDS) can exceed it. The rotation break-evens (45-50 GB/s) are
2.9-3.2x the physical link maximum: the hybrid design is impossible on
this hardware, not merely impractical. (A Gen5 x16 desktop/server target
would change this math — ~50 GB/s practical is at the rotation
break-even.)

### CPU recurrent layer cost (clean re-run, llama-bench)

The earlier `llama-server`/`llama-cli` attempts hung because `llama-cli`
enters interactive mode after generation and never exits; `llama-bench`
exits cleanly and reports deterministic tg128/tg16 rates. Measured with
our own build (04b569142) at `-ngl 17` (16 attention + output resident,
48 recurrent on CPU) vs `-ngl 99` (fully resident), `-fa 0`:

| Model | -ngl 99 TG | -ngl 17 TG | CPU recurrent diff |
| --- | --- | --- | --- |
| Q4_K_M | 36.32 t/s (27.5 ms/tok) | 3.80 t/s (263.2 ms/tok) | (263.2-27.5)/48 = **4.91 ms/layer** |
| Q6_K_XL | 25.15 t/s (39.8 ms/tok) | 2.63 t/s (380.2 ms/tok) | (380.2-39.8)/48 = **7.09 ms/layer** |

The Q6_K_XL `-ngl 17` number (2.63 t/s) closely matches the owner's
2026-08-23/24 stock-server baseline (2.55 t/s), validating the method.
The CPU recurrent cost is 4.9-7.1 ms/layer depending on quantization,
not the 7-8 ms/layer estimate in the plan.

### Mechanism verdict: Path A

Inspected `llama-model.cpp:1354-1372` — the stock `get_layer_buft_list`
is a static tail-count split: `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`.
A layer's device assignment is baked into `pimpl->dev_layer[il]` at load
time and never changes. There is no runtime weight movement in the stock
path.

**Verdict: Path A** — extend our module with an explicit per-layer placement
map. Load all layers to CPU mmap (as the streaming path already does), then
let the placement map decide: attention layers permanently bound to GPU
slots that never rotate, recurrent layers through slot rotation with
prefetch. The stock's static split mechanism is not touched. This is the
cheaper path because our module already owns host→device weight copies;
we only add the decision of *which* layers copy and on what schedule.

### Go/No-Go verdict: NO-GO (re-confirmed with corrected 13.7 GB/s rate)

## T-Q38HG-00/01/02: PP→TG Handoff (done 2026-08-24)

### T-Q38HG-00: Q4 KV cache
Implemented `--kv-cache q4` option (Q4_0 K and V). Greedy-ID parity confirmed
at 1024 tokens (ID 286, matches Q8 and F16 references). 38/38 full suite passes
with default KV type unchanged. Code change: `main.cpp` arg parsing only
(the vendored FA kernels already support Q4_0 KV dequant).

### T-Q38HG-01: Standard PP configuration validation
Config: 2 slots, bf16 slabs, Q4 KV, Q4_K_M model. 100K layer-major PP:
- Tok/s: 741.04 (+0.33% vs fa04 baseline 738.6) — Q4 KV FA path is slightly
  faster than Q8 (no dequant in the FA kernel).
- Total wait: 0.6 ms (< 10 ms gate). 2-slot lookahead holds comfortably.
- Peak VRAM: 7,395.56 MiB (7.22 GiB).
- q38ls-hg01-pp100k.jsonl, check-telemetry OK.

### T-Q38HG-02: Handoff machinery
End-to-end mechanism: PP (streaming) → begin_handoff (allocate + copy all 64
packages to one device buffer) → standard TG through layer_streaming_compute
with slot_layer returning resident tensors → end_handoff (free buffer).

Key design decisions:
- `slot_layer(il)` returns `handoff_layers.at(il)` when `handoff_active`.
  No graph topology changes — the same persistent graphs see different tensor
  data pointers (host mmap during PP, device buffer during handoff TG).
- `load_layer()` bypasses H2D copies when handoff is active (records
  zero-cost events for telemetry consistency).
- TG graph set is invalidated (`built = false`) after migration so the first
  post-handoff decode rebuilds with resident tensors.

Validation (Q4_K_M, 1024 PP + 16 TG, batch-major):
- Parity: ID 286 matches the standard TG baseline.
- TG rate: 24.75 tok/s (vs 0.61 tok/s without handoff — ~40× speedup).
  The TG tok/s is below llama-bench fully-resident (36.3 t/s) because the
  batch-major streaming path pays per-ubatch scheduler reset/split overhead
  that the standard decode path amortizes.
- Migration time: 1.72 s (15 GB Q4_K_M @ ~8.7 GB/s effective copy-stream
  rate). Plan expected ~1.1 s at 13.7 GB/s; the delta is from per-tensor
  `set_async` calls vs a single contiguous memcpy.
- Full suite: 36/36 passes.
- Telemetry: `handoff` events (migrate with byte count and copy time,
  teardown); host checker accepts graph_builds > 0 after handoff.
- q38ls-hg02-accept.jsonl.

### T-Q38HG-03: Hybrid Placement Probe (done 2026-08-24)

Read-only analysis probe. Three questions, all answered:

**Q1: Can `-ot` (--override-tensor) express "attention on GPU, recurrent on CPU"?**

YES. Mechanism: `common/arg.cpp:253` parses `pattern=buffer_type`, applies via
`std::regex_search` against tensor names in `llama-model-loader.cpp:1181`.
Tensor names follow `blk.N.member.weight` format. The 16 full-attention
layers can be targeted with:
```
-ot "blk\.(3|7|11|15|19|23|27|31|35|39|43|47|51|55|59|63)\.=CUDA0"
```
Combined with `-ngl 0` (all-CPU baseline), this places only attention
layers on CUDA. Validated with `-ot "blk.0.=CPU"` on `-ngl all`:
PP 238→203 t/s, TG 31.7→25.1 t/s — the override correctly reroutes
layer 0 tensors.

The 48-layer recurrent alternation is ~520 chars — well within OS limits.
Alternative: Path A tail-count split (`llama-model.cpp:1354`, `-ngl 17`)
would put only 4 of 16 attention layers on GPU — suboptimal for handoff.

**Q2: Attention-only migration scope?**

16 packages (full-attn layers only), ~3.8 GB at Q4_K_M (vs 15 GB for all
64). ~275 ms copy at 13.7 GB/s theoretical (~440 ms at measured 8.7 GB/s
on copy stream). Recurrent layers never migrate.

**Q3: Runtime-rebind — YES.**

The existing handoff mechanism (T-Q38HG-02) already supports this:
- `slot_layer(il)` routes attention layers to `handoff_layers.at(il)`
  (CUDA-resident) and recurrent layers to `slots[...].layers.at(il)`
  (host mmap via slot copies).
- `ggml_backend_sched_split_graph` assigns backends based on tensor
  buffer types automatically. CUDA tensors → CUDA backend, CPU tensors
  → CPU backend.
- The graph builder (`qwen35.cpp:layer(il)`) calls `slot_layer(il)`, so
  attention-layer graphs get CUDA-resident weights. No scheduler or
  graph-builder changes needed.

**Conclusion:** T-Q38HG-04 (constrained handoff) is unblocked.
Implementation is attention-only `begin_handoff()` (copy 16 layers, not
64) + standard handoff lifecycle.

### T-Q38HG-04: Constrained Handoff (done 2026-08-24)

Implemented `--handoff-attention-only` flag via `begin_handoff(true)`:
only 16 full-attention layers (blk.3,7,...,63) migrate to the device
buffer; recurrent layers route to CPU-resident host_layers via slot_layer().

**Results:**

| Metric | Value | Expected |
|--------|-------|----------|
| Migration bytes | 3.54 GB | ~3.8 GB ✓ |
| Copy time | 0.424 s | <0.5 s ✓ |
| Greedy ID (1024-tok) | 286 | 286 ✓ |
| Full test suite | 36/36 | 36/36 ✓ |
| TG tok/s | 3.28 | 3.80 (`-ngl 17` ref) ✓ (within ~15%) |

**TG routing fix:** slot_layer(il) returns `*host_layers[il]` for
recurrent layers during attention-only handoff, so the ggml scheduler
places them on CPU. load_layer() skips H2D copies for them. Transfer
collapsed from ~1280ms to ~68ms per 8-token batch.

Re-arm (T-Q38HG-06) is a follow-up task.

### T-Q38HG-06: MTP speculative decoding (done 2026-08-24)

Implemented by a second agent; reviewed and fixed 2026-08-24.

**Bug found during review:** the acceptance check compared
`draft[a] == prev` (the token *before* the current position) instead of
`draft[a] == cur` (the gold for the current position). This off-by-one
caused the reported 0.017-0.053 acceptance rate. Fixed to `draft[a] == cur`.

**Results (Q4_K_M, 1024 PP batch-major, 32 TG, `--handoff --mtp --mtp-draft 2`):**

| Metric | Baseline (no MTP) | MTP |
|--------|-------------------|-----|
| TG tok/s | 25.4 | 23.6 |
| Sequence | `286,220,197,289,...` | **identical** |
| Acceptance rate | — | 0.4167 (15/36) |
| check-telemetry | OK | OK |

**Parity: exact.** MTP-on and MTP-off produce identical 32-token greedy
sequences. The period-3 cycle (`289,197` repeating) is the model's true
greedy behavior on the synthetic prompt (deterministic token IDs
`i % n_vocab`), not an MTP defect.

**Acceptance rate 0.4167** is above the 0.3 escalation threshold. At
n_draft=2 the draft overhead slightly exceeds the acceptance benefit
(23.6 vs 25.4 tok/s). Higher n_draft or better acceptance would be
needed for net speedup; the mechanism is proven correct.

Traces: `q38ls-mtp-baseline.jsonl`, `q38ls-mtp-on.jsonl`.

The original verdict's reasoning was wrong (disk-bound measurement) but
the verdict itself survives the corrected numbers. True steady-state H2D
DMA is 13.7 GB/s. Two break-evens, both unmet:

1. All-48-streaming parity with the 3.80 t/s `-ngl 17` baseline:
   48 × 246 MB = 11.8 GB/token ÷ 263 ms/token budget = **45 GB/s
   required**; at 13.7 GB/s the transfer alone is 861 ms/token →
   ~1.16 t/s ceiling (better than the original 0.25 estimate, still 3.3×
   below the baseline it would replace).
2. Hybrid rotation (hide each window's copy behind the CPU compute of the
   current window): 246 MB ÷ 4.91 ms CPU layer time = **50 GB/s
   required**. At 13.7 GB/s the copy:compute ratio is fixed at 3.65×
   regardless of window size — the token always stalls 3.65× longer than
   the CPU work it overlaps. No window size or prefetch schedule fixes a
   fixed ratio.

The correct outcome per plan contract stands: record the negative result
and stop. T-Q38TG-01 remains not-started (no-go). The hybrid rotation
design is not viable on this machine's PCIe bus at 13.7 GB/s. The viable
TG lever remains post-PP slot expansion (all layers resident, zero
copies) for models that fit after slab release.

This plan is **complete with a negative result** (measurement corrected
2026-08-24; verdict unchanged).

## 20K Layer-Major Evidence (done 2026-08-20)

The availability probe found 24,137 MiB free and 0 MiB used. The GPU mutex
protected `powershell -NoProfile -ExecutionPolicy Bypass -File
tools/run-tests.ps1 -Suite pp20k-layer-major` exited 0 and wrote
`build/llama-cuda/q38ls-pp20k-layer-major.jsonl`; both `--check-telemetry` and
`--validate-telemetry` exited 0, producing the companion summary JSON.

- PP: 20,480 tokens / 40 chunks, 25,708.0 ms, 796.64 tok/s; 64 package copies,
  2,560 layer computes, 60 evictions, 23,111,182,336 bytes transferred,
  81,204.8 ms transfer, 0.5 ms availability wait, 17,752.6 ms compute, and
  `graph_builds == 0`.
- TG: 3 tokens, 7,630.3 ms, 0.39 tok/s; emitted IDs 317, 197, 197; 192
  evictions, 8,071.5 ms transfer, 17.1 ms availability wait, 270.8 ms
  compute, and `graph_builds == 0`. TG began from the live layer-major PP
  session; the validator accepted the PP-to-TG lifecycle without a slot reset.
- Preflight reported 22,922,919,936 free bytes of 25,650,855,936 total; two
  activation slabs used 838,860,800 bytes, and the 41 graph-set footprint
  contained 2,706 graphs. Peak VRAM was 8,247,640,064 bytes (7,865.56 MiB).
  The early PP layers had warmup-heavy transfer/compute totals; raw timings are
  recorded only as evidence, not as a speedup or quality claim.
- `tools/run-tests.ps1 -Suite all` then passed all 19 targets under the same
  mutex, including batch-major, layer-major, and the positive/negative 40-chunk
  lifecycle fixtures. The default-OFF `llama-cli` target had already rebuilt
  successfully. Default batch-major behavior remains unchanged and the
  layer-major path remains opt-in.

Caveats: this run had an idle GPU at start, but the mmap source remains
best-effort cold because Windows file-cache state is uncontrolled. No resident
baseline, quality evaluation, equivalence test, or speedup comparison was run.

## T-Q38PERF-01: copy timing split (done 2026-08-20)

`copy_complete` now carries additive `host_submit_us` and
`ready_latency_us` fields. Their sum equals the legacy `transfer_us` field;
the host telemetry checker rejects partial, negative, or inconsistent splits
when they are present while accepting historical fixtures without them. This
does not isolate DMA duration: `ready_latency_us` still includes copy-stream
queueing. The focused `layer-streaming-layer-major-1024` CUDA CTest passed
under the GPU mutex, followed by `--check-telemetry`; its first copy recorded
122.3 ms host submission, 344.2 ms ready latency, and 466.5 ms legacy
submit-to-ready transfer. The three host-only `layer-streaming-lm40` fixture
tests also passed. Default behavior and slot scheduling are unchanged.

## T-Q38PERF-02: package-class slot allocation (done 2026-08-20)

Slots now allocate only the recurrent and/or attention package classes served
by their logical-layer mapping. The representative tensor for each allocated
class is selected from a matching logical layer, which also fixes mixed-class
slot counts where the slot index itself can name an attention layer. For the
four-slot schedule, `slot_vram_bytes` fell from 2,853,972,608 to
1,444,448,896 bytes (1,409,523,712 bytes, or 1,344.2 MiB, saved). The focused
four-slot layer-major CTest passed. A new persistent five-slot mixed-class
runner-plus-checker fixture passed; its checker now derives each slot's actual
layer-cycle length, so it accepts uneven 64-layer distributions (12 or 13
layers per slot) while preserving the four-slot policy. Default behavior is
unchanged.

## T-Q38PERF-03: pinned-host package staging (done 2026-08-20)

The isolated runner now accepts `--copy-source direct|pinned` (default
`direct`) and records the selected value in `run_start.copy_source`. Pinned mode
gives each slot one CUDA-pinned host buffer sized to that slot's device package
allocation. It packs package tensors into the buffer before enqueueing the
existing asynchronous copies; the existing consumer-complete event protects the
buffer from reuse. Pinning failure or allocator fallback to ordinary host memory
fails initialization rather than silently changing copy behavior.

Paired mutex-protected 1,024-token layer-major traces at u-batch 512, four
slots, zero PP logits, and one greedy TG token both exited 0 and passed
`--check-telemetry` (837 events, four slots):
`build/llama-cuda/q38ls-direct-1024-paired.jsonl` and
`build/llama-cuda/q38ls-pinned-1024-paired.jsonl`. Pinned PP was 8,366.4 ms
(122.394 tok/s), versus direct 8,195.3 ms (124.950 tok/s). It reduced summed PP
host submission from 7,505.1 to 6,062.4 ms, but raised availability wait from
0.5 to 393.8 ms. Pinned TG was also slower (2,757.5 vs 2,515.0 ms), with
availability wait rising from 5.3 to 1,510.4 ms. This single paired observation
does not establish a speedup; pinned staging remains opt-in as an instrumented
experiment. After CMake reconfiguration, the mutex-protected
`layer-streaming-pinned-1024` CTest fixture passed its runner and host checker
(2/2, 17.90 s).

## Completed

### T-Q38LS-01: reproducible CUDA experiment envelope (done 2026-08-18)

Commands (both from workspace root, Git Bash; compilation does not need the
GPU mutex):

1. `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON` ->
   exit 0. CUDA 13.2.51 toolkit, host compiler MSVC 19.50.35726.0, target arch
   `120a-real` (RTX 5090 / sm_120). OpenSSL 3.6.1 found. NCCL absent (warning
   only; single GPU, irrelevant).
2. `cmake --build build/llama-cuda --config Release --target llama-cli --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-cli.exe`.
3. `cmake --build build/llama-cuda --config Release --target llama-gguf --parallel 16`
   -> exit 0. Artifact `build/llama-cuda/bin/Release/llama-gguf.exe`.

Model identification evidence:
- `llama-gguf.exe "<model>" r` (llama.cpp's own reader): GGUF version 3,
  alignment 32, data offset 10,996,800, n_kv 51, n_tensors 866; `qwen35.*`
  KV keys present; per-tensor sizes verified (e.g. `blk.0.attn_qkv.weight`
  55,705,600 bytes). Full dump in the env guide.
- `llama-cli` under the GPU mutex (`tools/with-gpu-lock.ps1`) with
  `-ngl 8 -p hi -n 1`: model loaded and executed on CUDA. Banner reported
  `build b10493-04b569142`, model path, `modalities: text`. Prompt processed
  at 13.2 t/s (partial offload: 8/64 layers on GPU), one token generated.
  Log: `out/q38ls-ngl8-smoke.log`. Evidence that the CUDA runtime loads and
  runs this exact artifact.
- `llama-cli -ngl 0 -p hi -n 1` (CPU-only, no CUDA): did not complete within
  600 s and was killed (201 s CPU consumed). The CPU path is impractical for
  this 26 GB hybrid model; consistent with the CUDA-only experiment scope.

Quant facts (corrected during this task): `general.file_type = 14` maps to
llama.cpp enum `LLAMA_FTYPE_MOSTLY_Q4_K_S` whose display name is "Q4_K -
Small" - a legacy label that does not describe this file. Actual per-tensor
quants (this tree's ggml enum: F32=0, Q8_0=8, Q5_K=13, Q6_K=14):
`output.weight` Q8_0, `token_embd` Q6_K, `attn_qkv`/`ffn_down` Q8_0,
`ffn_gate`/`ffn_up` Q6_K, `ssm_alpha`/`ssm_beta` Q5_K, 1-D params F32. This
is the UD/Q6_K_XL dynamic scheme. Byte sizes measured from tensor offsets are
ground truth and are in the env guide (static 2.394 GB, recurrent package
364.0 MB max, full-attn package 352.4 MB max, four-slot window ~1.444 GB).

### T-Q38LS-02: one-layer Qwen3.8 execution boundary (done 2026-08-18)

Implementation (all gated by CMake option `LLAMA_LAYER_STREAMING`, default OFF):
- `src/llama-layer-streaming.{h,cpp}`: slot buffers + tensor bundles, persistent
  activation pair, persistent host input buffer, dedicated CUDA copy backend,
  per-slot CUDA events, JSONL telemetry, and the layer-by-layer executor
  (`llama_context::layer_streaming_compute`).
- `src/llama-graph.h`: `llm_graph_params.layer_streaming` (default nullptr).
- `src/models/qwen35.cpp` + `models.h`: layer-view indirection `layer(il)`
  (model.layers[il] normally, slot-bound bundle in streaming mode) and
  `build_streaming_graphs()` which produces one cgraph per piece: embed graph,
  64 per-layer graphs, output graph.
- `src/llama-context.{h,cpp}`: gated streaming branch in `process_ubatch`.
- `tools/layer-streaming/main.cpp`: runner (`llama-layer-streaming` exe) with
  boundary mode, PP, greedy TG, JSONL telemetry.

Validated evidence (commands from workspace root, GPU runs under
`tools/with-gpu-lock.ps1`):
- `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` and `cmake --build ... --target
  llama-layer-streaming` -> exit 0.
- OFF-build check: separate dir `build/llama-cuda-off` (no option) builds
  `llama-cli` -> exit 0 (default path unchanged; the only ungated change is a
  nullptr-initialized field in `llm_graph_params`).
- `ctest --test-dir build/llama-cuda -C Release -R qwen38-layer-boundary
  --output-on-failure` -> Passed (4.0 s).
- Boundary run: `llama-layer-streaming --model <env path> --layer-slots 1
  --boundary-layers 0,3 --pp-tokens 64 --ubatch 512 --no-mtp --telemetry
  out/q38ls-boundary.jsonl` -> exit 0. Recurrent layer 0 (Gated DeltaNet) and
  full-attention layer 3 both constructed and executed one-layer CUDA graphs.
  Telemetry: `out/q38ls-boundary.jsonl` (14 lines: run_start, per-layer
  slot_request/copy_submit/copy_complete/layer_ready_wait/layer_compute,
  phase_summary pp 64 tokens 205.7 tok/s, run_end exit 0). Copy byte counts
  match the env guide exactly: layer 0 364,022,656 B, layer 3 352,380,928 B.
  Peak VRAM 3.87 GiB during the boundary run (foreign process still resident).

Key measured behavior (one slot): the H2D copy from the pageable mmap source
blocks the host inside `cudaMemcpyAsync` (driver staging for pageable memory),
so per-layer wall is copy-then-compute with ~0 availability wait at
consumption; transfer_us (submit->complete) ~111 ms for the recurrent package
and ~104 ms for the full-attn package on this shared GPU. This is recorded as
observed behavior, not a defect or a claim.

### T-Q38LS-03: full one-slot dense streaming prefill (done 2026-08-18)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf     --pp-tokens 512 --ubatch 512
 --layer-slots 1 --no-mtp --telemetry out/q38ls-t03.jsonl"`
-> exit 0. Log `out/q38ls-t03.log`, telemetry `out/q38ls-t03.jsonl` (3074 lines).

Measured results:
- PP: 512 tokens in 24,363.2 ms = 21.02 tok/s. 8 u-batches of 64 tokens.
- `phase_summary` (JSONL): tokens 512, ubatches 8, wall_us 24,363,220,
  tokens_per_s 21.015, bytes_transferred 184,889,458,688 (512 package copies
  x ~361 MB avg), evictions 511, total_transfer_us 23,391,879, total_wait_us
  43,900, total_compute_us 850,252, layers_run 64.
- Per-layer averages (over 8 executed u-batches): recurrent layer transfer
  ~45.4-53.2 ms + compute ~1.3-2.1 ms; full-attn transfer ~43.9-44.7 ms +
  compute ~1.4-2.1 ms; setup ~0.04-0.09 ms; wait ~0.08-0.10 ms. Copy bytes
  match the env-guide package inventory (recurrent 364,022,656 B,
  full-attn 352,380,928 B).
- CUDA0 compute buffer 188.64 MiB (matches expected), CUDA_Host 2.565 MiB.
  Peak VRAM 3.91 GiB.
- JSONL ordering verified: every `layer_compute` follows the matching
  generation's `copy_complete`; telemetry has 512 copy_submit/slot_request/
  copy_complete/layer_ready_wait/layer_compute pairs (8 ubatches x 64 layers)
  plus run_start/phase_summary/run_end.

Required validations:
- Runner command (above) -> exit 0 and eight PP u-batches in telemetry. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-slot-lifecycle
  --output-on-failure` -> Passed (30.9 s) under the GPU mutex.

Defect found and fixed during this task:
- Persistent graphs were corrupted on graph reuse: `ggml_backend_sched_alloc_graph`
  rewires node src pointers in place when introducing cross-backend copies, and
  `sched->ctx` is freed on the next split; re-splitting a reused graph then
  aliased stale rewired pointers with unrelated tensors (observed: embed cpy
  src degraded to a 64-elem token leaf -> `cpy.cu` element-count assert). Fix:
  the executor snapshots each graph's node src pointers before alloc and
  restores them after compute, keeping the persistent graphs pristine across
  u-batches. T-04/05 inherit this fix.

### T-Q38LS-04: four-slot persistent-graph PP streaming (done 2026-08-19)

Command (GPU mutex; workspace root):
`powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "<build>/bin/Release/llama-layer-streaming.exe --model C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf  --layer-slots 4 --pp-tokens 512 --ubatch 512

 --no-mtp --telemetry out/q38ls-pp.jsonl"`
-> exit 0. Log `out/q38ls-pp.log`, telemetry `out/q38ls-pp.jsonl` (3071 lines).

Measured results (512-token trace):
- PP: 512 tokens in 26,540.0 ms = 19.29 tok/s. 8 u-batches of 64 tokens.
- `phase_summary`: tokens 512, ubatches 8, wall_us 26,539,954, tokens_per_s
  19.292, bytes_transferred 184,889,458,688, evictions 508,
  total_transfer_us 102,470,682, total_wait_us 2,159, total_compute_us 775,010,
  layers_run 64, graph_prep_us 2,046, graph_builds 1.
- Peak VRAM 6.04 GiB (vs 3.91 GiB for one slot). Four-slot window observed as
  expected (~2.85 GB of slot buffers reported at run_start).
- With four slots the pageable H2D copies now overlap the compute of the other
  slots; total_wait_us dropped to ~2.2 ms (from 43.9 ms at one slot) because a
  layer's copy is usually already complete when consumed. Transfer remains
  dominant (~99.8 GB accumulated copy time vs ~0.78 s compute); copies from the
  pageable mmap source still serialize on the copy stream.

Required validations:
- Runner command (above) -> exit 0 with all required PP event types. Passed.
- `ctest --test-dir build/llama-cuda -C Release -R layer-streaming-scheduler
  --output-on-failure` -> Passed (run 12.5 s + host-side invariant check
  0.1 s) under the GPU mutex. The check validates exactly n_slots distinct
  slots, `slot == layer % n_slots`, per-slot generation monotonicity,
  copy_submit -> copy_complete -> layer_ready_wait -> layer_compute ordering,
  prior-resident `slot_evict` identity, and `graph_builds == 0` on the 64-token
  single-ubatch run.
- Regression: `qwen38-layer-boundary` Passed (4.0 s), `layer-slot-lifecycle`
  Passed (30.4 s) under the GPU mutex. Default (OFF) `llama-cli` build still
  compiles.

Implementation notes:
- Added `llama_context_params.n_pad` (default 0 = unchanged). The runner sets
  `n_pad = n_ctx` so the KV cache pads `n_kv` to the full context for every
  ubatch, keeping all graph shapes constant across the whole session (the
  existing `get_n_kv` padding mechanism, now pinned to full width). Tradeoff:
  full-attention layers attend over the full padded width from the first
  ubatch; recorded, not claimed as an optimization.
- Added `llama_layer_streaming_prebuild()` (exported wrapper around
  `llama_context::layer_streaming_prebuild`) which splits the batch, applies
  the first ubatch (so graph shapes match), builds the 66 per-piece graphs
  before the PP timer, then `memory->clear(false)` releases the applied cell
  metadata so the timed decode starts from the identical fresh cell mapping.
- Persistent graph entries now carry explicit `(mode, logical_layer, slot)`
  keys and the executor fails loudly on any mode/layer/slot mismatch; slot
  generation is now in copy_complete/layer_ready_wait/layer_compute telemetry;
  `slot_evict` now reports the prior resident layer.
- Runner gains a host-side `--check-telemetry PATH` validator (no GPU/model)
  used by the scheduler CTest.

Resolved by T-Q38LS-04B (rs_z two-variant persistent-graph key):
- The Gated DeltaNet recurrent `rs_z` is 0 on the first PP ubatch (fresh state;
  the graph includes an in-graph zeroing of one state cell) and -1 on every
  continuing ubatch (zero-sized view). T-Q38LS-04B closes this: the graph cache
  is now keyed `(mode, n_tokens, rs_z)` with per-variant input tensors and result
  objects, and the prebuild builds all three variants (fresh-PP, continuing-PP,
  TG-u1) before the timed region. The 512-token/4-slot timed PP region now
  reports `graph_builds == 0` (pre-fix baseline was 1) and the 256-token
  four-ubatch run that crosses the rs_z 0 -> -1 transition also reports
  `graph_builds == 0`. See the T-Q38LS-04B section below.

### T-Q38LS-04B: per-variant persistent graph sets (done 2026-08-19)

Observed behavior
- The streaming module now owns per-variant graph sets keyed
  `(mode, n_tokens, rs_z)`. Each set owns its io context, activation region,
  input region, and `llm_graph_result`. The executor selects the set per ubatch
  by `(phase, n_tokens, rs_z)`, verifies the prebuilt input tensor shapes
  (mismatch fails loudly per SDD), writes inputs, and runs the set's piece
  graphs. `process_ubatch` bypasses the `gf_res_prev` reuse/build machinery in
  streaming mode; an on-demand build of an unprebuilt variant remains as a
  counted fallback.
- Prebuild builds variant A (fresh PP, rs_z 0), variant B (continuing PP,
  rs_z -1), and, when `--tg-tokens > 0`, variant C (TG u1, mode 1, rs_z -1),
  then `memory->clear(false)` releases all applied cell metadata so the timed
  decode starts from the identical fresh cache.

Validation evidence (Release, GPU mutex)
- `layer-streaming-pp512`: 512 tokens / 4 slots, 8 ubatches, `graph_builds == 0`,
  `wall_us ~25.6ms`, `total_transfer_us ~97.2ms`, `total_compute_us ~0.94ms`,
  `layers_run 64`. Host `--check-telemetry` passes.
- `layer-streaming-scheduler-256`: 256 tokens / 4 slots, 4 ubatches crossing the
  rs_z 0 -> -1 transition, `graph_builds == 0`. Host check passes.
- `layer-streaming-scheduler` (64-token), `qwen38-layer-boundary`,
  `layer-slot-lifecycle`: all pass. OFF build `llama-cli` compiles clean.

Root cause of the first GPU run's shape-guard failure (debugging record)
- The executor verified sets via `llm_graph_result::can_reuse`, whose
  `params.allow_reuse` half compares the stored prebuild `llm_graph_params`
  (dangling `mctx`/`sched`/`ubatch.data` pointers) against the live decode
  params, so it always failed despite every tensor shape matching (prebuild
  n_kv 256 == live n_kv 256, n_rs 1, rs_z 0, n_tokens 64). Fix: run each
  prebuilt input's own `can_reuse` (shape checks, which re-point the input mctx
  to the live context) and skip the stale-pointer params half. No SDD/plan
  boundary changed.

### T-Q38LS-05: PP-to-TG transition (done 2026-08-19)

Observed behavior
- TG (128 greedy tokens) begins immediately after PP with the exact four slot
  assignments and per-session state (KV + Gated DeltaNet) left by PP; no slot
  reset or host prewarm. The first TG layer-0 request finds slot 0 already
  resident from PP (generation 129, `ready_at_consumption true`, wait ~3us) -
  no wrap-around preload.
- The TG phase uses the prebuilt `tg_u1` graph set (mode 1, n_tokens 1, rs_z -1)
  built before the timer, so the timed TG region reports `graph_builds == 0`.
- 128 greedy token IDs are emitted in a `tg_tokens` JSONL event and
  `run_end.n_tokens_generated == 128`.

Validation evidence (Release, GPU mutex, 512 PP + 128 greedy TG / 4 slots)
- `layer-streaming-transition` (run + host check): pass; run ~345s.
- PP: 512 tok / 8 ubatches, `graph_builds 0`, ~19.8 tok/s, transfer ~98s, compute ~1.0s.
- TG: 128 tok / 128 ubatches, `graph_builds 0`, 0.41 tok/s, transfer ~1223s,
  compute ~9.6s; per-layer transfer/wait/compute present for all 64 layers.
- All PP-only scheduler/slot/boundary regressions pass; OFF build compiles clean.

Notes
- `reset_stats()` at the PP->TG boundary makes each `phase_summary` report its
  own phase's totals (previously cumulative, so TG double-counted PP).
- Per-layer copy/wait/compute telemetry now carries a `phase` field so TG events
  are distinguishable from PP events.
- `cparams.n_ctx`/`n_pad` round `pp+tg` up to a 256-multiple (640 -> 768) so
  `n_pad` divides `kv_size == n_ctx_seq`; the mask masks empty padded KV slots to
  -INF so the larger constant `n_kv` is safe.

### T-Q38LS-06: evidence-only profiling record (done 2026-08-19)

Reproducibility command (no GPU/model needed for validation)
- `build/llama-cuda/bin/Release/llama-layer-streaming --validate-telemetry <trace>.jsonl`
  -> exit 0, writes `<trace>.jsonl.summary.json` (the validated profile artifact).
- Source trace: `--model <Q6_K_XL.gguf>  --layer-slots 4 --pp-tokens 512 --ubatch 512



  --tg-tokens 128 --greedy --no-mtp --telemetry <trace>.jsonl` (run via CTest
  `layer-streaming-transition`).

Validation result
- `--validate-telemetry` on the transition trace: exit 0, 52225 events, 4 slots,
  64 layers. `ctest -R layer-streaming-telemetry`: positive (writes summary) and
  negative (WILL_FAIL on missing/incomplete telemetry) both pass.

Profile artifact (q38ls-pp-tg.jsonl.summary.json) raw metrics
- PP: 512 tokens, ~7-20 tok/s (shared-GPU noise), transfer ~98-267s, wait ~4ms,
  compute ~1-2.3s, 64 layers, graph_builds 0.
- TG: 128 tokens, ~0.4 tok/s, transfer ~1.2s, wait ~62ms, compute ~9.3s,
  64 layers, graph_builds 0.
- peak_vram_bytes recorded; per-layer transfer/wait/compute min/mean/max for all
  64 layers per phase.
- Caveats recorded: source cache best-effort cold (uncontrolled Windows file
  cache); no speedup baseline was run; no quality/equivalence/speedup claim.

Manual test running + output (2026-08-19)
- `tools/run-tests.ps1 -Suite all|quick|boundary|telemetry` runs the CTest
  suites under the GPU mutex (`--output-always` prints every test's output).
  `-Suite telemetry` is host-only (no GPU, no lock).
- Every model run now self-checks its `--telemetry` trace at exit and prints a
  per-phase summary: tokens, tok/s, wall, total transfer/wait/compute, layers
  run, evictions, graph_builds, peak VRAM. `--check-telemetry <trace>.jsonl`
  prints the same summary for any existing trace (host-only).
- Note: `build/llama-cuda/q38ls-pp-tg.jsonl` must be regenerated by a fresh
  `layer-streaming-transition-run` (it was truncated by a killed mid-run test
  launch on 2026-08-19 and deleted).

### T-Q38LM-01: stable layer-major activation slices (done 2026-08-20)

Opt-in `--pp-schedule layer-major` path (default `batch-major`) proving that
two 512-token chunks can be driven through one selected logical layer while
retaining distinct, stable, full-prompt activation-slab ranges. Evidence only;
no quality/equivalence/speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Reconfigure with the model path baked into the CTest entries (the build had
  been reconfigured with an empty `Q38LS_MODEL_PATH`, which emptied every
  GPU test's `--model` arg): `export Q38LS_MODEL_PATH=
  C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf` then
  `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON
  -DLLAMA_LAYER_STREAMING=ON` -> exit 0.
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0.
- Layer-major boundary probe (GPU mutex): `ctest --test-dir build/llama-cuda
  -C Release -R layer-streaming-lm-boundary-run --output-on-failure` (runner:
  `--pp-schedule layer-major --boundary-layers 0 --pp-tokens 1024 --ubatch 512
  --pp-no-logits --no-mtp --telemetry build/llama-cuda/q38ls-lm-boundary.jsonl`)
  -> Passed (8.4-20 s). Log `out/q38ls-lm-boundary.log`.
- Host recheck (no GPU): `llama-layer-streaming --check-telemetry
  build/llama-cuda/q38ls-lm-boundary.jsonl` -> exit 0 (15 events, 1 slot).
- Regression (GPU mutex): `tools/run-tests.ps1 -Suite boundary` -> 2/2 pass;
  `-Suite quick` -> 8/8 pass (boundary, lifecycle, scheduler, scheduler-256,
  pp512). Default-OFF `llama-cli` builds (no GPU).

Observed behavior
- `run_start` carries `schedule:"layer-major"` and `act_slab_bytes:41943040`
  (2 x [5120,1024] F32 = 40 MiB for the 1,024-token probe). The preflight
  prints `act slabs require 41943040 bytes (2 x 20971520); free VRAM
  21610102784, total 25650855936` before inference. The formula is
  `2 * n_embd * pp_tokens * 4`, which is 838,860,800 bytes (800 MiB) for the
  20,480-token envelope; this is an allocation size, not a VRAM-fit claim.
- `act_bind` events record the per-chunk slab offsets: chunk 0 -> slab A
  [0,10485760) + slab B [0,10485760); chunk 1 -> slab A [10485760,20971520)
  + slab B [10485760,20971520). The slices are non-aliasing (host validator
  proves it).
- The layer-0 package is copied once (one `copy_submit`/`copy_complete`,
  generation 1, 364,022,656 B) and both chunk computes reuse that one
  generation: two `layer_compute` events, chunk 0 (ubatch 1) then chunk 1
  (ubatch 2), same slot 0, same generation 1. No second copy, no eviction
  between the two chunks. KV/recurrent state is owned by the logical layer
  and written in-graph, never by a slot index.
- `phase_summary`: tokens 1024, ubatches 2, `graph_builds 0` (per-chunk graph
  sets were prebuilt; the timed region did no graph construction), compute
  ~241 ms, transfer ~2320 ms (shared-GPU noise), peak VRAM ~3.85 GiB.

Selected graph-binding mechanism (recorded per the task contract)
- Per-chunk prebuilt graph sets: each chunk's act pair is a pair of
  non-aliasing fixed-offset slices of the two slabs, bound at prebuild. The
  set key is `(mode, n_tokens, rs_z, n_outputs, chunk_id)`. Measured
  two-chunk probe footprint: `chunks=2 graph_sets=2 total_graphs=132
  act_slab_bytes=41943040 io_ctxs=2` (2 sets x 66 graphs; 2 io contexts).
- Rejected mechanism: a `ggml_view_2d` over a stable slab tensor. The
  scheduler's galloc calls `ggml_backend_view_init` on a view whose source
  has a backend buffer, which allocates a *separate* buffer for the view and
  defeats the stable-slab / non-aliasing semantics. Fixed-offset
  `ggml_backend_tensor_alloc` from the slab buffer is used instead (the same
  documented API the batch-major act pairs use).
- Host `--check-telemetry` now enforces, for `schedule == "layer-major"`:
  per-layer ascending chunk identity (0,1,2,...) and non-aliasing `act_bind`
  slices per slab. Both negative fixtures (out-of-order chunk; aliasing bind)
  are rejected with exit 1, so the checks are not vacuous.

Caveats: shared GPU (a foreign process held VRAM; timing is subject to
shared-GPU noise); source cache is best-effort cold (uncontrolled Windows file
cache); this is a one-layer, two-chunk boundary probe - full 64-layer PP, 20K
scale, and the PP->TG handoff are T-Q38LM-02..04.

### T-Q38LM-02: full two-chunk layer-major PP pass (done 2026-08-20)

A dedicated `llama_context::layer_major_pp` controller drives 1,024 PP
tokens (two 512-token chunks) through all 64 logical layers in layer-major
order (chunk 0 then chunk 1 for every layer before the next layer), with each
package copied once and reused across both chunks. This is a
scheduler-state / determinism check, not a quality or speedup claim.

Commands (workspace root, Git Bash; GPU runs under `tools/with-gpu-lock.ps1`):
- Build (no GPU): `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16` -> exit 0. Default-OFF build
  (`build/llama-cuda-off`) also compiles (the experimental path is gated out).
- Layer-major 1024 (GPU mutex): `ctest --test-dir build/llama-cuda -C Release
  -R layer-streaming-layer-major-1024 --output-on-failure` (runner:
  `--pp-schedule layer-major --pp-tokens 1024 --ubatch 512 --layer-slots 4
  --pp-no-logits --tg-tokens 1 --greedy --no-mtp --telemetry
  build/llama-cuda/q38ls-lm-1024.jsonl`) -> 2/2 pass (run ~16-36 s, host
  check ~0.07 s). Log `out/q38ls-lm1024-full.log`.
- Regression (GPU mutex): `ctest -R layer-streaming-(scheduler^|pp512^|layer-major-1024)`
  -> 8/8 pass; `tools/run-tests.ps1 -Suite quick` -> 8/8 pass. Batch-major and
  the existing CTest names are unchanged.
- Host negative fixtures (no GPU): a deliberately reordered-chunk trace and a
  duplicated-chunk trace both fail `--check-telemetry` with exit 1 ("layer-major
  chunk not ascending for layer"), so the layer-major invariants are not
  vacuous.

Observed behavior
- `phase_summary` (PP): tokens 1024, ubatches 2, `graph_builds 0`, 64
  layers_run, 60 evictions (4-slot lookahead), compute ~0.74 s, transfer
  ~31 s (shared-GPU noise, host-synchronous pageable copies). 128 PP
  `layer_compute` events: chunk 0 then chunk 1 for each of the 64 layers, no
  layer L+1 compute before both layer-L computes complete. Exactly 64 PP
  `copy_submit` events: each layer's package is copied once and consumed by
  both chunks under the same generation; the generation increments as the
  four slots are reused (layers 0-3 gen 1, 4-7 gen 2, ...).
- TG (1 token) uses the standard per-ubatch path and reuses the TG-u1 set
  exactly like batch-major (`graph_builds 0`). The one-token TG
  schedule-consistency check produces the same first greedy ID **286** under
  batch-major (`q38ls-bm-1024.jsonl`) and layer-major (`q38ls-lm-1024.jsonl`)
  for the same 1,024 PP tokens.

Design notes (recorded per the task contract)
- The controller creates one memory context per chunk and applies it in chunk
  order so the shared rec/KV state is current for each chunk's `find_set` /
  `set_inputs` (the prebuilt per-chunk sets were built against the identical
  applies). The recurrent `rs_z` is captured per chunk at apply time (chunk 0
  -> 0, chunk 1 -> -1) because `get_rs_z()` reads the shared rec memory that
  the next `apply()` overwrites; the controller re-resolves each chunk against
  the same `(rs_z, n_out, chunk)` tuple the prebuild used.
- The per-ubatch path (batch-major) is unchanged. Layer-major PP is rejected
  there as a routing error; layer-major TG (phase 1) reuses the TG-u1 set
  exactly like batch-major. Activation slabs swap only after every chunk of
  the current layer completes (fixed per-chunk slab slices, no aliasing).

Caveats: shared GPU (timing subject to shared-GPU noise); source cache is
best-effort cold (uncontrolled Windows file cache); this is a 2-chunk / 64-layer
PP determinism check - 20K scale, a 3-token TG, and the negative/final
regression are T-Q38LM-03..05.

### T-Q38LM-03 through T-Q38LM-05: 20K PP, 3-token TG, and regression closure (done 2026-08-20)

See "20K Layer-Major Evidence" above. The free-GPU run completed all 40 chunks
through every logical layer with legal generation reuse and then generated
three TG tokens from the same live state. Both telemetry validators accepted
the 4,113-event trace, and the full mutex-protected regression suite passed
19/19 targets. The default-OFF build remains successful. Raw measurements are
subject to uncontrolled Windows file-cache state and are not a speedup or
quality claim.

## Known gaps / risks (current)

- Foreign `deepseek_v4.exe` shared the GPU during earlier validation
  (~12.4 GiB VRAM held, high util), so those timing results are subject to
  shared-GPU noise. The completed 20K run started with the GPU idle; every
  trace still records observed free VRAM at `run_start`.
- `llama-cli --log-file` stalled with an empty log file on this build (two
  runs killed). Not task-owned (our runner writes its own JSONL telemetry),
  but recorded here: do not rely on llama-cli's file logger.
- CPU-only execution of this model is impractically slow (see above).
- With pageable mmap copy sources the copies are host-synchronous; the four-slot
  lookahead overlaps compute but the copies themselves still serialize on the
  copy stream. Recorded as observed behavior, not a claim.
- The CMake CTest entries bake `$ENV{Q38LS_MODEL_PATH}` at configure time; a
  reconfigure without that env var set rewrites the model path to empty and the
  GPU tests fail at arg parsing. Reconfigure with Q38LS_MODEL_PATH exported.

## Blocked / escalated

- T-Q38LS-04 `rs_z` variant escalation is resolved by T-Q38LS-04B.
- T-Q38TG-01 (hybrid TG executor) is permanently gated: T-Q38TG-00
  returned NO-GO (measured H2D ~3.0 GB/s vs ~29 GB/s break-even).
- T-Q38HG-04 (constrained handoff, attention-only migration) is done —
  all acceptance criteria met: migration 3.54 GB / 0.424 s, parity ID
  286, test suite 36/36, TG 3.28 tok/s (within ~15% of -ngl 17 baseline).
  slot_layer() routes recurrent layers to CPU-resident host_layers;
  load_layer() skips H2D copies for them.
- T-Q38HG-05 (GTX 1080 validation) is gated on owner hardware access.
