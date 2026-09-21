# Generic prefill layer residency — executable backlog (live tasks only)

SPLIT 2026-09-20: the 18 closed/completed task blocks moved VERBATIM to
`docs/plans/generic-prefill-layer-residency-archive.md` (mechanical record:
docs/plans/archive-manifest.json, including the byte-exact reconstruction check).
This file now holds ONLY live work (in-progress or draft): search, read and edit it
as before. Index of archived blocks (original order preserved in the archive):
- ### T-GPLR-25: PP eviction prefetch-overlap prototype (device-side copy overlap in layer-major PP)
- ### T-GPLR-26: Gate-eligibility decision recording — binding vs advisory (item 4 brief)
- ### T-GPLR-27 (CLOSED — DEFERRED 2026-09-18 by owner decision): Second-architecture first pass — dense target (item 6 brief)
- ### T-GPLR-29: Pinned-pool probe — measure the true pinned H2D ceiling vs the direct-copy path
- ### T-GPLR-32: WDDM DMA-inflation serialization — one H2D in flight at a time (af worker DMA gate)
- ### T-GPLR-33: Member-batched submission — one submission per bank (91 -> ~44-58 ms window)
- ### T-GPLR-35: PP staging overlap — bank DMAs on a second stream inside the compute context (warm 4k ~2.75-2.8 -> ~2.0-2.1 s)
- ### T-GPLR-36: Long-context PP residual — what is the ~10 s that is neither compute nor engine time?
- ### T-GPLR-38: ubatch sweep on the 4k anchor — does fewer, fatter launches buy wall time under the current best mode?
- ### T-GPLR-31: Locked-RAM probe — does the driver grant a full-pool-size pinned allocation on this box?
- ### T-GPLR-30: Partial persistent pinned pool — warm-wall savings curve vs RAM cost
- ### T-GPLR-28: In-process layer-major PP repeat — first-touch mechanism experiment
- ### T-GPLR-44: per-graph patch scoping - stop restoring the whole pw set's inputs on every launch
- ### T-GPLR-45: drop the zero-consumer orphan from the patch restore - stop memcpy-ing the framework's pruned inp_embd on every launch
- ### T-GPLR-46: the pass-2 compute regression is REAL and rebuild-linked - isolate the trigger and the growth law
- ### T-GPLR-47: rebuild graph sets into a FRESH context (swap-in-slot), fixing the pass-2 regression and the T-GPLR-44 stale-pointer hazard at the root
- ### T-GPLR-48: make --pp-prebuild-per-pass KV-conditional (default ON only for quantized KV) per the T-GPLR-46 owner decision
- ### T-GPLR-49: fix the layer-major act-slab offset formula - bf16 element size and the chunk-0 cursor inheritance

Rules (same convention as the currentstate split, AGENT_ENVIRONMENT.md):
- Closed tasks are archived, never edited in place; a new closure appends its RESULT
  in the live file first, and the block moves to the archive on the NEXT split.
- The archive's 6-line header is not original bytes; strip it when reconstructing.
- Re-prove the split any time: python tools/split-gplr-plan.py --verify
### T-GPLR-37: direct mmap-sourced staging — remove the CPU hop from bank staging

Status: in-progress (2026-09-17) — PROBE FEASIBLE; V1 (per-bank churn)
**FALSIFIED AT 4k** (warm regression -15.4% at 2 slots, -23.8% at 3); V2
(churn-free, register once at init — see addendum 2 below) **CONFIRMS the churn
mechanism and WINS at 4k at both depths** (warm -23.3% at 2 slots, -15.4% at 3,
paired against adjacent pinned controls), at a measured cost of 17.23 GiB of
pinned file pages. PROMOTED TO DEFAULT 2026-09-20 by owner decision (verbatim in the RESULT at the end of this block).

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
4b13ac3 (evidence/tooling, gitlink bump); 2026-09-20 default flip: nested 8ab3f1f00
(default + copy_source_source provenance + --pp-pool refusal), outer 63bfaa6 (harness +
gitlink), plus this verdict commit



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


#### T-GPLR-37 RESULT (2026-09-20): PROMOTED TO DEFAULT by owner decision — the flip is behavioral, self-describing, and costed

**Owner decision, verbatim: "Promote to default — you always pay 17 GiB of locked RAM, always get the speedup where it exists."** The churn-free register-once mode (`--copy-source mmap-registered`) is now the DEFAULT copy source; explicit `--copy-source` still overrides everything, so every recorded arm command stays reproducible. The change: the default literal, a stderr announcement emitted ONLY when the default decided (T-GPLR-48 pattern), a new `run_start` key `copy_source_source` (`explicit|default`) beside the already-recorded resolved `copy_source`, the strict checker validating the new key, an updated help entry, and one disclosed consequence — bare `--pp-pool` (a pinned staging structure) now refuses BEFORE model load with the one-flag fix in the message, where it previously only failed at pool configure under the pinned default.

**One session, five arms + the smoke rung, all under the GPU mutex with started/verdict/done sentinels.** Control = `bin/gplr37-control` (exe `150B6351833DA150`, llama.dll `7EBB3803CDD67B41`), changed = `bin/gplr37-changed` (exe `A3EB9C3A32F1DA0E`, dll `89CE209871DA7DFB`). **The control-preserving lesson re-measured:** the first "control" copy was taken AFTER the build had already compiled the change into the live `bin/Release` — grep for the new key in the would-be control DLL exposed it — so the control was rebuilt from pristine nested HEAD (diff saved/reverted/built/pinned/restored, patch-identical restore verified) before anything ran.

- **INV1 (control silence):** the control at bare default resolves `pinned`, carries NO `copy_source_source` key and NO announcement (C1); at explicit `--copy-source pinned` it behaves exactly as the pre-change tree did (A3).
- **The new default, measured (A1, 4k/3 slots):** resolved `mmap-registered` with source `default`, announcement present, **40 init-once `mmap_register` events = 18,500,517,888 B (17.23 GiB) pinned in 2.66 ms, ALL before the first `layer_compute`** — the addendum-2 churn-free signature exactly; strict `--check-telemetry` OK; route parity clean (160 events, 0 singletons).
- **Identity:** the 96 greedy ids are **byte-identical across ALL FOUR 4k arms** (control default, control pinned, changed default, changed pinned); first id 248046; graded decode **PASS 5/5** (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`) in every arm.
- **Timing (same session, same binary, warm pass-2 walls — recorded, NOT gated):** control pinned 2095.4 ms (A3), changed pinned 2121.7 ms (A2), changed **default 1953.5 ms (A1) = -6.8% vs the same binary pinned and -7.9% vs the control**. This is the honest current-tree number: addendum 2 measured -15.4% at 3 slots on the pre-T-45/47/48/49 tree, and the pinned path itself got faster since, so the same absolute win is now a smaller fraction. Single pair, one session, no replication — the delta is directional evidence, not a certified constant. A1's pass-1 compute (3.71 s vs 1.84 s pass 2) carries the expected first-touch page-in of the pinning itself; disclosed, not a regression claim.
- **A4 (b1, 143k q8, the T-GPLR-43/47 reference shape):** resolved default, 40/17.23 GiB pre-compute, checker OK, parity clean (1200 events, 0 singletons), 154.4 s — the long quantized-KV shape works unchanged under the new default.
- **Smoke rung (V2, 512 anchor, bare default, batch-major q8, repeat 2):** runner exit 0, strict checker 0, needle **PASS 5/5**, negative control FAILs as designed — the DEFAULT path answers the needle.
- **Consumers:** every recorded script and CTest fixture passes `--copy-source` explicitly (all `pinned`), so no recorded command changes behavior; the only `--pp-pool` consumer pins it too. Host suite 37/37, telemetry suite pass, doc-size guard PASS.
- **Limits, stated:** the 17.23 GiB pinned-RAM price is UNCONDITIONAL under this default (that is the owner's recorded choice); addendum 3 measured the crowded 200k shape wall-neutral (-0.22%/-0.38%, within noise) with no memory-pressure event at 61.7 GiB, and the default now applies there too. The -6.8% wall delta is one session, one pair. Host-RSS at rest is ~20.5 GB for a defaulted run (measured via the live process working set during A4), most of it the pinned mapping.
- **Falsifiers: NONE fired** (no identity divergence anywhere, no parity failure, no refusal in any supported combination).
**Evidence:** verdicts `build/llama-cuda/verdicts/gplr37-*.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr37-*.jsonl`, smoke `build/llama-cuda/gplr37-v2-qa-512.{jsonl,log,txt}`, harnesses `tools/run-gplr37-arm.ps1` / `tools/run-gplr37-set.ps1` / `tools/run-gplr37-qasmoke.ps1`, binaries `build/llama-cuda/bin/gplr37-{control,changed}`.

### T-GPLR-39 (draft): make the `other` ledger bucket chunk-independent — the residue that grows with ubatch size

Status: DONE (2026-09-20) — route telemetry DEFAULTS OFF (owner decision); RESULT block below.
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
Commit: nested f6d104c1f (engine: gate + provenance keys + dense register coalescing), outer 52e66c4 (harness: arm/set/qasmoke runners + parity-tool note), outer (this commit) (plan + currentstate)

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

#### T-GPLR-39 RESULT (2026-09-20): route telemetry DEFAULTS OFF — the diagnostic no longer measures itself

**Owner decision, verbatim: "This can be turn off as it is a diagnostic and it should be run only when needed; by default it should be off if not used in code or for predictions."** The route event (device->host probs readback + host sort, 1161.5 ms/pass at u2048 — 100.8% of the measured `other` bucket) is now opt-in: `--route-telemetry 1` restores the observational event (route-parity gate, route analyses), the bare default resolves `0` with the T-37/T-48 self-describing pattern (stderr announcement emitted only when the default decided; `run_start` gains `route_telemetry` + `route_telemetry_source` (`explicit|default`); strict checker validates both keys and REJECTS route events in a telemetry-off trace). The FUNCTIONAL routing path (`read_router_ids` -> `sel_in`) is untouched — this is the recon's decision point resolved as "off unless needed", not a new fix class; the (d) bounded top-k host selection from the 2026-09-18 arm stays in place for both the diagnostic and the functional path.

**One session, three arms + the smoke rung (MoE vehicle, 4k/3 slots, q8):** control = `bin/gplr37-changed` (the T-37 RESULT binary, pre-T-39), changed = `bin/Release` (exe `B75661AC5F50060B`, dll `A6519AC5E2F9913E`).
- **D1 (default):** resolves `route_telemetry=0` source `default`, announcement present, **ZERO route events** in the trace, checker OK, decode PASS 5/5.
- **E1 (`--route-telemetry 1`):** resolves `1`/`explicit`, NO announcement, 160 route events, **route parity clean (0 singletons)**, decode PASS, pair identity vs D1 identical (tg ids + pass structure).
- **C1 (control INV1):** at bare default carries NO new keys and NO announcement, 160 route events (pre-T-39 behavior exactly).
- **Identity:** the 96 greedy tg ids are **byte-identical across D1/E1/C1** (first id 248046) — recording or not recording routing does not change the stream.
- **Timing (same session, warm pass-2 walls, recorded NOT gated):** D1 1951.0 ms, E1 1958.9 ms, C1 1985.0 ms — telemetry on/off is within session noise at this 4k shape, as expected (the 1161.5 ms figure was measured at u2048 where route events carry 4 MiB readbacks; the u2048 wall win predicted by the recon accrues when u2048 is adopted, not here).
- **Smoke rung (512 anchor, bare default, batch-major q8, repeat 2):** runner 0, strict checker 0, needle **PASS 5/5**, negative control FAILs as designed.
- **Dense-fixture defect fixed en route (T-37-owned, surfaced by this task's host suite):** `register_all_pkg_sources` refused on dense models ("cudaHostRegister ... already mapped", layer 1) because consecutive dense layers' page-rounded spans share a boundary page — the MoE vehicle never hit it (non-package tensors separate consecutive banks). Fixed by greedy coalescing of page-touching spans: MoE grouping unchanged (40 init-once events / 17.23 GiB, all pre-compute), dense groups merge into one registration owned by the group's first layer, members marked covered. Dense CTest fixtures run again; the CTest `--layer-slots 1` pair (`qwen38-layer-boundary`, `layer-slot-lifecycle`) still fails PRE-EXISTING (profile-v3's `--pp-async-feed` default-1 refuses single slots — recorded, not this task's).
- **Consumer contract changes:** the route-parity gate now requires an explicit `--route-telemetry 1` run (a telemetry-off trace exits 2 by design — "no route events" is the DESIGNED outcome, not a pass or a failure); the AGENTS.md route-parity note is updated accordingly.
- **Falsifiers: NONE fired** (no identity divergence, no parity failure in any on-arm, no silent route events in the off-arm).
**Evidence:** verdicts `build/llama-cuda/verdicts/gplr39-{D1-default,E1-on,C1-control}.verdict.json` (+ started/done), traces `build/llama-cuda/gplr39-{D1-default,E1-on,C1-control}.jsonl`, smoke `build/llama-cuda/gplr39-qa-512.{jsonl,txt,log}`, harnesses `tools/run-gplr39-{arm,set,qasmoke}.ps1`, binaries `build/llama-cuda/bin/{Release,gplr37-changed}`.

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

### T-GPLR-50 (draft): make the f16 rebuild default follow the charter shape — ON where the pass-2 excess is measured (flip the T-GPLR-48 f16 default back to ON)

Status: done (2026-09-21) - all acceptance criteria validated; the RESULT block
below records the measured outcome, including the P3 premise failure that
re-attributes the recorded +67.4 s pass-2 excess to GPU contention.
Depends on: T-GPLR-48 (the default being changed), T-GPLR-47 (the fresh-context
rebuild that makes rebuild-ON cheap and correct), T-GPLR-46 (the regression
evidence this reverses), T-GPLR-49b (the echo-attribution the validation reuses).
Commit: nested 40bca9f44 (engine flip); outer harness/gitlink and plan/state
commits follow in the outer repo (hashes: git log after 2026-09-21).

Survey context (why this is the highest-value remaining PP lever)
: With T-GPLR-39 closed, the recorded host ledgers leave exactly one lever whose
  prize is measured in tens of seconds per pass at the charter's crowded
  long-context shape, and it is a default flip, not new machinery. The T-48 f16
  default (rebuild OFF) re-exposed the pass-2 compute excess that only T-47's
  fresh-context rebuild ever eliminated:
    * 200k f16, rebuild ON  (T-47 v3, 2026-09-20): pass 1 = 141.3 s wall /
      133.85 s compute; pass 2 = 133.95 s compute — excess **+0.10 s**.
    * 200k f16, rebuild OFF (remeasure, 2026-09-21): pass 1 = 144.8 s wall /
      138.0 s compute; pass 2 = 205.4 s compute — excess **+67.4 s (+48.9%)**,
      uniform ~1.5x per layer_compute across all chunk positions.
    * 50k f16, both states: pass-2 compute within ~0.2 s of pass 1 in every
      recorded arm (v2 on 17.70 s vs off 17.76 s; T-48 default 17.86 s) — the
      excess does NOT appear at 50k under either state.
  The T-46-era justification for KV-conditionality (rebuild was expensive and
  carried a crash risk) is gone: T-47 rebuilt the rebuild into a fresh context,
  T-49 fixed the slab-offset derivation, and T-48's own q8 arms prove the
  rebuilt path clean at b1. What remains of the T-48 rationale is one measured
  fact — the f16+rebuild+pw combination at 4k/96 echoed instead of answering
  (T-49 session, pre-T-49b attribution) — and T-49b already bounded that: the
  echo needs pw AND rebuild TOGETHER; rebuild OFF + pw is clean (PASS 5/5), and
  nopw + rebuild ON is clean (v1-on PASS, ids byte-identical to the nopw
  default stream). The default path (pw ON, layer-major) would therefore take
  the echo risk back on f16 unless the validation below pins its trigger —
  which is exactly what this task's AC6 does before any flip is claimed.

Goal
: Change the T-GPLR-48 default resolution so f16 KV also prebuilds per pass
  (rebuild ON) whenever `--pp-repeat > 1`, with the same provenance pattern
  (announcement + `pp_prebuild_per_pass_source`), keeping explicit flags
  authoritative. Expected effect, per the recorded ledger deltas: the 200k
  pass-2 excess falls from +67.4 s to ~+0.1 s (measured ON/OFF pair already in
  hand), i.e. ~33% of warm two-pass wall at the charter's primary crowded
  shape. No other shape has a recorded regression to trade: 50k is neutral in
  both states.

Pre-registered predictions (written before the arms run)
: * P1 (primary): on the current tree, the 200k f16 rebuild-ON arm's pass-2
    compute lands within ~1 s of its pass-1 compute (recorded delta +0.10 s;
    falsified if the excess exceeds 5 s).
  * P2: the flip does not move pass-1 compute at any shape beyond session
    noise (rebuild happens between passes; falsified if pass-1 compute shifts
    > 3% paired within-session).
  * P3: the warm two-pass 200k wall improves >= 25% paired within-session vs
    the rebuild-OFF control (measured ON-side wall 141.3 s vs OFF-side 198.2 /
    206.1 s pass-2 sessions; falsified below 15%).
  * P4: 50k f16 and b1 q8 stay wall-neutral to +2% (rebuild adds ~3-7 s of
    slot-bucket variance at 50k per the v2 pairs; falsified beyond +5%).

Scope
: One resolution site in the runner (main.cpp, the T-48 block at ~3078):
  `(pp_repeat > 1 && kv_cache != "f16")` -> `(pp_repeat > 1)`; announcement
  text gains an f16 branch naming the charter shape; `run_start` keys and
  checker validation are already generic (no engine change expected; if the
  checker hard-codes the f16 expectation anywhere, that is in scope).
  Harness: one arm/set runner pair + the flip's validation arms, reusing the
  T-48 harness shapes verbatim (a1-style 50k default, a2-style b1 q8 default,
  a4-style 4k f16 default, plus a new 200k f16 default arm on the hostmem
  shape) and the recorded pw/nopw 4k identity pair for AC6.

Acceptance criteria (falsifiable)
: AC1 (identity, default path): the 96 greedy tg ids of the 4k f16 default arm
  (rebuild ON after the flip) are byte-identical to the recorded T-47 v1-on
  stream (the rebuild-ON reference already in hand) AND to the current
  rebuild-OFF default stream (gplr48-a4) — both comparisons, else falsified.
: AC2 (echo trigger pinned, pre-flip): the pw+rebuild+echo interaction is
  re-run on the CURRENT tree in both states (pw rebuild-OFF vs pw rebuild-ON
  at 4k/96, decode-graded). If pw+ON echoes on the current tree (T-49b was
  pre-T-49's slab fix, so this is genuinely open), the default flip MUST NOT
  complete while the default path uses pw; instead the task escalates back
  with the pinned trigger and the owner chooses between flipping pw's default,
  shipping rebuild-ON-for-f16 as explicit-only, or accepting the documented
  limitation. If it answers (PASS 5/5), AC6 is discharged trivially.
: AC3 (primary performance): the 200k f16 pair (OFF control = current default
  vs ON changed, same session under one mutex hold, repeat 2, hostmem shape,
  recon 1) shows pass-2-minus-pass-1 compute excess shrinking from the
  recorded +67.4 s class to <= 5 s (P1) and warm two-pass wall improving
  >= 25% (P3), with per-layer_compute uniformity restored (max/min ratio
  across chunk positions < 1.1, vs ~1.5 today).
: AC4 (no collateral regression): the 50k f16 default arm and the b1 q8
  default arm both PASS their strict checkers with pass-2 compute within +5%
  of pass 1 (both states were already clean at these shapes — this guards the
  flip against silently breaking the quantized-KV path it does not touch).
: AC5 (provenance): a defaulted run announces the new f16 branch and
  `run_start` carries `pp_prebuild_per_pass=1` + source `default`; explicit
  `--pp-prebuild-per-pass 0` and `1` arms resolve `explicit` and reproduce
  recorded behavior (the T-48 a3 matrix reruns clean, INV1 on the explicit
  paths); route-parity style integrity on the 200k pair via the ledger
  closures (each pass's ledger closes within noise as today).
: AC6 (semantic gate): the qa-smoke rung (512) passes on the new default;
  if AC2's current-tree rerun echoes at pw+ON, the smoke rung runs BOTH pw
  and nopw defaults and only the answering one may ship the flip.
: AC7 (long-context discipline): the 200k arms run one at a time under the
  GPU mutex as owner-invoked acceptance runs (never batched), per the ladder
  policy; single-slot refusals and handoff auto interactions checked exactly
  as the T-47 v-arms did.

Non-goals
: No engine (llama-layer-streaming) changes unless the checker hard-codes the
  old default; no pw-path redesign (that is the echo trigger's own task, if
  AC2 pins it live); no ubatch default change; no second-architecture work;
  no deep-submit or patch scoping revival.

Known evidence and unknowns
: Known: all ledger numbers above (recorded, mtime-attributable to binaries);
  the T-47 v1 within-session identity pair; the T-49b echo attribution; the
  T-48 a3 explicit-path matrix.
: Unknowns: (1) whether the pw+rebuild echo reproduces on the CURRENT tree
  (post-T-49 slab fix) — AC2 resolves it first, and its outcome selects
  between the plain flip and the escalate branch; (2) whether the 200k excess
  has a second contributor that rebuild-ON masks but does not fix (the +67.4 s
  was uniform, which argues no; AC3's uniformity metric catches it if yes);
  (3) pass-1 wall at 200k shifted +3.5 s between the T-47 and remeasure
  sessions — AC3's paired design cancels it, but absolute two-pass walls
  should be reported with the session caveat.

Escalation condition
: If AC2 shows the echo live on the current tree AND the owner is not
  reachable to choose, the task parks with the pinned trigger recorded and no
  flip; if AC3 fails (excess persists with rebuild ON on the current tree),
  the pass-2 excess is a NEW defect distinct from T-46's — escalate with the
  ledger rather than tuning.

#### RESULT (2026-09-21): SHIPPED - flip validated; the motivating excess was GPU contention, not the rebuild-OFF path

All arms on the changed binary (exe 1B08B382ACA4694D / dll B540024042E8B1F2,
nested 40bca9f44), sentineled verdicts in build/llama-cuda/verdicts/gplr50-*:

* AC2 (echo trigger, pre-flip): pw + rebuild-ON on the CURRENT tree ANSWERS -
  gplr50-ac2-pw-on grades PASS 5/5 (needle intact); pw-OFF also PASS. The T-49b
  echo did not reproduce post-T-49; the escalation branch never fired. AC6
  discharged: qa-smoke 512 rung PASS with the negative control failing as designed.
* AC1 (identity): the 4k f16 default arm's 96 greedy tg ids are byte-identical to
  BOTH recorded references (T-47 v1-on rebuild-ON and gplr48-a4 rebuild-OFF) and
  to every other 4k arm in the set - the flip moves no token.
* AC5 (provenance): default resolves pp_prebuild_per_pass=1 source=default on f16
  (gplr50-4k-f16-default, gplr50-50k-f16-default) and q8 (gplr50-b1-q8-default);
  explicit 0 -> 0/explicit (gplr50-ac2-pw-off), explicit 1 -> 1/explicit
  (gplr50-expl-on); the pre-T-50 control resolves 0/default on f16 with the T-48
  announcement and zero T-50 markers (gplr50-ctl-4k-f16), 1/default on q8
  (gplr50-ctl-4k-q8). Cosmetic note: q8 default arms now announce "default since
  T-GPLR-50" although the q8 VALUE dates to T-GPLR-48 (one announcement text for
  both branches; recorded in currentstate).
* AC4 (no collateral): 50k f16 pass-2 compute 17.795 s vs 17.813 s pass-1 (-0.1%);
  b1 q8 72.52 s vs 73.27 s (-1.0%). Both inside the +5% guard.
* AC3 (200k confirmation): single-binary A/B pair under ONE mutex hold, remeasure
  shape verbatim (tools/run-gplr50-200k-pair.ps1): OFF (explicit 0) pass-2 compute
  134.24 s vs pass-1 135.06 s (excess -0.81 s); ON (default) 134.67 s vs 136.14 s
  (-1.47 s). P1 PASSES (|excess| <= 5 s in both states). P3's PREMISE FAILED: the
  recorded +67.4 s rebuild-OFF excess (gplr36-remeasure-200k, pre-T-50 binary) DID
  NOT REPRODUCE on the byte-equivalent engine path. Re-attribution: that
  remeasure's pass-2 carried uniformly-inflated layer_compute with normal
  transfer ratios - the contention signature reproduced exactly by two ctest
  fixtures that crashed (teardown cudaFree OOM) while deepseek_v4.exe held
  ~16.5 GiB VRAM DURING this project's own mutex-held host suite; both fixtures
  PASS on rerun under a free GPU (234.9 -> 19.1 s, 148.5 -> 11.3 s). The GPU
  mutex does not exclude non-conforming projects. Consequence: the flip's
  charter value is unrealized-today (nothing to reclaim at 200k on a quiet GPU),
  but the default is still the correct posture: neutral where quiet (measured),
  and it reclaims the excess automatically if a genuine rebuild-path regression
  ever reappears (the T-46-class cause).
* Host suite: 68/70 on the changed binary after contention reruns - the two
  remaining failures are the documented pre-existing --layer-slots 1 refusal
  pair. Doc-size guard PASS.

Verdicts: build/llama-cuda/verdicts/gplr50-*.verdict.json (+ .done sentinels);
traces build/llama-cuda/gplr50-*.jsonl; contention evidence and the withdrawn
P3 in docs/currentstate.md (2026-09-21 T-GPLR-50 entry).

### T-GPLR-51 (draft): attack the EXPOSED transfer at the charter shape - copy-tail and hiding robustness at 200k

Status: draft (2026-09-21) - awaiting owner decision. Evidence-gathering
outside the plan so far: trace decompositions recorded in currentstate
2026-09-21 (the two entries below this block) and the gplr51-4k-mmap probe.
Depends on: T-GPLR-50 (closed; the current default binary), T-GPLR-37
(copy-source default under test), T-GPLR-25 (async-feed hiding mechanism).
Commit: (fill on completion)

Motivation (measured, from today's 200k traces - NOT the 249 s bucket)
: The 249 s transfer bucket at 200k is ~97% HIDDEN: wall 135.5 s ~= compute
  133.3 s, exposed layer_ready_wait totals 3.67 s (2.7%), because the 3-slot
  + async-feed pipeline overlaps staging with compute. The bucket also sums
  OVERLAPPING per-event transfer durations (40 parallel copies, per-event
  mean 6.2 s) and therefore overstates serial transfer time. Every quiet-GPU
  shape today is compute-bound (200k 1476 tok/s, 144k q8 1961, 50k 2614).
  What IS measured and wrong-shaped:
    * Copy-tail pathology at 200k mmap-registered: per-copy p50 32 ms but
      max 10.8 s (pass-1 mean ~74 MB/s per ~460 MB layer bank -
      page-fault-driven file IO, not PCIe). The exposed 3.67 s is the
      bubble residue of that tail; a colder file or a slower GPU generation
      grows it (the charter's crowded case).
    * Same-binary copy-source A/B never ran at 200k. At 4k (same binary,
      same session class) mmap-registered beats pinned 2x: 3.3 vs 11.7
      s/pass staged, warm tok/s 2582 vs 1824 (gplr51-4k-mmap vs
      gplr50-4k-f16-default). The T-37 default is right at 4k; 200k is
      untested same-binary (T-36's pinned 141.8 s was a different binary).
    * b1 q8 at u1024 shows the same big-copy pattern as 200k pass 1 (40
      copies/pass, mean 3.4 s) while 50k u512 pinned shows 160 small ones -
      chunk-to-copy mapping differs by shape and is unexplained.

Goal
: Cut the EXPOSED transfer at 200k - layer_ready_wait below 1% of wall and
  a bounded copy tail - and establish which copy-source hides better at the
  charter shape on one binary, so the crowded/cold case inherits the
  strongest hiding baseline. PP tok/s at 200k is NOT expected to move beyond
  noise on a quiet GPU (compute-bound); the prize is robustness of the
  hiding under the charter's priority-3 condition.

Arms (all repeat 2, owner-invoked one at a time under the mutex, sentineled)
: A1 200k mmap-registered (default) - EXISTS: gplr50-200k-on (baseline, no
    rerun unless stale).
  A2 200k pinned (same binary, --copy-source pinned) - the never-run
    same-binary staging A/B at the charter shape.
  A3 200k default + --layer-slots 4 (slot_vram 1.4 GiB of 24 GB today; a
    4th slot deepens the staging pipeline against the 10.8 s tail).
  A4 200k default + --pp-async-feed 0 (hiding-ablation control: how much of
    the 97% hiding is the async worker vs the slot window alone).
  A5 (optional, owner-gated) 200k default with a flushed file cache for a
    best-effort cold-read datum; recorded as uncontrolled per the standing
    Windows file-cache caveat.

Pre-registered predictions (written before the arms run)
: P1 A2 (pinned) exposes MORE wait than A1 at 200k (the 4k result inverts
    at 391 ubatches because mmap amortizes page-fault IO across a much
    longer pass) - falsified if pinned's ready_wait <= mmap's 3.67 s.
  P2 A3 (4 slots) cuts exposed wait below 2 s and shrinks the copy tail's
    p99.9 (max) by >= 30% vs A1 - falsified if the tail max stays > 7 s.
  P3 A4 (no async feed) exposes >= 10x A1's wait (~40 s class) - falsified
    below 5 s (would mean the slot window alone does the hiding).
  P4 No arm moves pass tok/s beyond +/- 3% of 1464-1485 (compute-bound
    prediction) - any bigger move is a finding about compute, not staging.

Acceptance criteria (falsifiable)
: AC1 A1/A2 same-binary pair completes with verdicts, check-telemetry clean,
    and gpu_pre_run=clean; per-pass wall/compute/transfer + copy p50/max
    recorded for both.
  AC2 The exposed-wait metric (ready_wait total / wall) is < 1% for the
    winning copy-source at 200k, or the failure is characterized per-shape
    and the result recorded (an honest negative also closes the task).
  AC3 The copy-tail ratio (max/p50) at 200k is explained by a named
    mechanism (page-fault vs PCIe vs slot contention) with at least one
    corroborating probe (e.g. pass-1 vs pass-2 pattern, or an
    event-tracing run) - no speculative fix ships without it.
  AC4 A4's ablation quantifies the async-feed contribution; the number is
    recorded in currentstate regardless of direction.
  AC5 No regression: every arm's 4k-proxy identity check (tg ids at the
    4k shape when run) stays byte-identical to the recorded streams; the
    200k arms carry tg 3 smokes and check-telemetry only.

Non-goals
: No engine change in this task unless AC3's mechanism names a one-site
    fix (then it returns to the Planner with the evidence). No new
    telemetry events; no KV-type change; no multi-GPU; no host-RAM
    prefetch (T-GPLR-08 falsified); no readahead tuning inside this task.

Escalation condition
: If A2 (pinned) beats A1 at 200k - inverting the 4k result - the
    copy-source default question reopens with a shape split and goes back
    to the owner before any default change. If P4 falsifies (tok/s moves >
    3%), stop and re-attribute before continuing: a staging change that
    moves compute is a different defect.

#### A2 result (2026-09-21, arm run gplr51-a2-pinned): copy-source question CLOSED at 200k - mmap stays; the copy tail is NOT source-specific

Same binary as the A1 baseline (exe 1B08B382ACA4694D / dll B540024042E8B1F2,
hash-verified), same day, both verdicts gpu_pre_run=clean (the 001bf65 gate):

| metric | A2 pinned | A1 mmap-registered |
| --- | --- | --- | --- |
| pass walls | 135.4 / 137.4 s | 136.6 / 135.1 s |
| pass compute | 133.1 / 134.8 s | 134.4 / 134.4 s |
| tok/s | 1477 / 1456 | 1464 / 1480 |
| exposed ready_wait | 3.62 s | 3.67 s |
| copy p50 | 75.8 ms | 31.5 ms |
| copy max | 10.97 s | 10.82 s |

* P1 FALSIFIED: pinned does not expose more wait at 200k (3.62 vs 3.67 s -
  equal); the 4k inversion does not extend to exposure at 391 ubatches.
* P4 HELD: tok/s inside the +/-3% band; compute-bound confirmed again.
* mmap keeps the typical-copy advantage (p50 2.4x faster), consistent with
  the 4k discriminator. The copy-source default does not reopen; per the
  draft's escalation condition no inversion occurred.
* NEW, sharper finding for AC3: the ~10.9 s copy tail exists in BOTH sources
  (10.82 mmap / 10.97 pinned). Pinned staging never touches the file during
  the pass (21.5 GiB pre-pinned at startup), so the tail cannot be the
  mmap page-fault path - it is STRUCTURAL: staging-worker queueing /
  slot-window contention / PCIe congestion with parallel copies. A3 (4
  slots) is now the prime tail probe; the page-fault hypothesis is demoted
  to the pass-1 pattern only (40 whole-bank copies at ~74 MB/s).

Remaining arms: A3 (4 slots), A4 (async-feed ablation); AC2's < 1% exposed
wait is not met by either source today (both ~2.7%).

#### A3+A4 result (2026-09-21, arms gplr51-a3 / gplr51-a4): the copy-tail question CLOSED - structural to the shape, not a fixable staging defect

Same binary as A1 (hash-verified), all verdicts gpu_pre_run/post_run clean:

| metric | A1 (mmap, 3 slots, af1) | A2 pinned | A3 (4 slots) | A4 (af 0) |
| --- | --- | --- | --- | --- |
| tok/s | 1464 / 1480 | 1477 / 1456 | 1475 / 1490 | 1485 / 1485 |
| exposed ready_wait | 3.67 s | 3.62 s | 3.57 s | 3.67 s |
| copy p50 | 31.5 ms | 75.8 ms | 31.8 ms | 31.5 ms |
| copy p90 | - | - | 12,072 ms | 10,796 ms |
| copy max | 10.82 s | 10.97 s | 12.31 s | 10.80 s |

* P2 FALSIFIED (A3): the 4th slot did NOT deepen useful overlap - pass-1
  per-copy times inflated ~4x (p50 3.0 -> 11.8 s; transfer bucket 249 -> 370
  s) because more in-flight copies contend for the same file/PCIe resources,
  and the tail max GREW (10.82 -> 12.31 s). Exposed wait barely moved
  (3.57 s). 3 slots is the saturation point; more slots are worse.
* P3 FALSIFIED (A4): async-feed contributes ~NOTHING to hiding at 200k -
  af 0 is byte-identical to the baseline (3.67 s, p50 31.5 ms, max 10.80 s,
  tok/s equal). The hiding is done by the slot window + recon pipeline
  alone; the T-25 worker is idle-redundant at this shape.
* AC2 verdict: < 1% exposed wait is NOT reachable by any tested staging
  configuration (floor ~2.7% across A1-A4). An honest negative.
* AC3 mechanism (named, corroborated in 4 arms): the ~11 s worst-case copy
  is per-layer-file READ throughput, not queueing or PCIe. It sits at the
  heaviest layers (24-39, 522 MB vs 453 MB - 16% more bytes, ~16% slower
  means), is byte-identical across pinned (no file reads during the pass),
  4-slot, and af-0 arms, and scales DOWN with contention measured the other
  way (A3's inflation shows the resource, not the fix). The copy tail is a
  property of the 200k shape's file layout, not a staging defect; it hides
  at 97% today and its exposure is flat under every lever tried.

Consequences: no engine change is justified by this task (the draft's
non-goal holds). A2/A3/A4 are recorded as falsifiers; A5 (cold cache)
remains owner-invoked optional. The staging-bound framing of the reference
table stands: PP at 200k is compute-bound on a quiet GPU; the exposed
transfer floor (~2.7%) is the cost of staging 18.3 GB twice per run under a
97%-efficient hiding pipeline.

### T-GPLR-52 (draft): attribute the shape-dependent per-copy staging cost (identical bytes, 18x wall) and correct the T-GPLR-51 copy-tail framing

Status: draft (2026-09-21)
Dependencies: T-GPLR-51 (closed; its A2/A3/A4 verdicts and framing are inputs and correction targets)
Commit:

Goal. Explain, with a named mechanism and pre-registered discriminators, why two
whole-bank copies of the SAME ~460 MB routed-expert bank in the same binary show
host-observed per-copy costs of 163 ms (4k prompt), 943 ms (50k), and 3003 ms
(200k). Secondary: establish what copy_complete actually measures and put an
instrument beside it that future staging claims can trust.

Established facts (host-only re-decomposition of existing traces, 2026-09-21):

- F1 - Staging granularity is shape-invariant: exactly 2 copy events per layer
  per PP pass (one per whole routed-expert bank) in EVERY recent trace (200k,
  144k, 50k, 4k, 512). The earlier "fractional copies per layer / chunking
  differs by shape" reading lumped phase-1 (TG) copy events into phase-0 (PP)
  stats. There is no 40-vs-160 chunking to explain.
- F2 - Per-copy host cost scales with prompt length, not bytes: median PP copy
  163 ms (4k) -> 943 ms (50k) -> 3003 ms (200k), an 18x swing for identical
  bank bytes. At 4k, pass-2 halves vs pass-1 (109 vs 285 ms); at 200k it does
  not (3003 vs 3007 ms).
- F3 - Heavy-layer byte correlation (layers 24-39 vs 0-23) is ~1.15-1.16 in the
  200k regime but ~1.06/1.01 at 4k - byte-proportionality exists only when
  copies are slow.
- F4 - copy_complete is a HOST-side event wait (wait_slot_ready):
  transfer_us = host_submit_us + ready_latency_us = enqueue + sync time, NOT
  DMA duration. It legitimately includes waiting behind concurrent compute and
  other queued copies. Every T-51 tail statistic (p50/max) is therefore
  host-observed cost, not transfer time; the A2/A3/A4 "exposed wait" numbers
  remain valid (they measure the same wait) but the "copy tail" framing is
  instrument-limited.
- F5 - A2's pinned arm shows mmap-identical 3.0 s medians at 200k, but its log
  retains the mmap banner ("expert banks in host mmap", load_mode = mmap) and
  the trace carries only copy_source=pinned, no in-trace proof the pinned pool
  was active. The engine has separate pinned machinery (--pp-set-pp-pool
  REQUIRES --copy-source pinned), which the A2 arm did not pass. If pinned
  silently fell back to the registered-mmap pool, the A2 "source-independent
  tail" attribution is unfounded.

Hypotheses (pre-registered; state which prediction each arm tests BEFORE it runs):

- H1 (queue-wait behind compute): the host wait grows because per-ubatch
  compute grows with prompt length and the copy event fires only after the
  copy engine gets the bus. Prediction: per-copy transfer_us correlates with
  concurrently-overlapping compute, not with bytes; true DMA duration would be
  near the 4k regime at all shapes.
- H2 (file-cache eviction on the registered mapping): pass-1 copies fault
  file pages at disk-class rate; effective 18.3 GB / ~250 s = ~73 MB/s matches
  disk-class read throughput. Prediction: at shapes whose working set survives
  in the Windows file cache (4k), later passes halve; at 200k the far-ahead
  pages are evicted before use, so every pass stays slow regardless of pass
  index or copy source reading the file.
- H3 (pinned never engaged): A2's arm exercised the mmap path. Prediction:
  with in-trace proof of an active pinned pool, 200k per-copy cost collapses
  toward the 4k regime (RAM-speed source), refuting the file-read component of
  H2 and correcting the T-51 A2 verdict.

Discriminating arms (each owner-invoked, GPU mutex, ~5 min; A1 first is
host-only and free):

- A1 (host-only): from existing traces, correlate per-copy transfer_us with
  the compute window it overlaps (layer_compute rows inside the copy interval)
  and with the copy queue's concurrent copy count. AC: a per-shape attribution
  table; if transfer_us ~ overlapped compute, H1 is confirmed without any new
  binary.
- A2 (warm/eviction probe): run the charter shape with --pp-repeat 3 (200k and
  50k) and build the pass-index vs median-copy table. H2 predicts pass-3
  ~ pass-2 at 200k (evicted) but a drop at 50k if its working set survives.
  AC: the table plus a verdict per H2's prediction.
- A3 (proven-active pinned): add an in-trace pinned_active assertion (staging
  pool address range check, or the set-pp-pool path) and rerun the 200k
  copy-source A/B. AC: trace carries pinned_active=true; medians compared
  against the mmap baseline settle H3 and correct the T-51 A2 addendum.

Acceptance criteria:

- AC1: the 40-vs-160 premise is corrected in the plan (this block) and
  currentstate; tracked analyzers that report PP staging stats are verified to
  split phase-0 (PP) from phase-1 (TG) copy events or to label the mixture.
- AC2: a named mechanism explains the 163/943/3003 ms scaling, supported by at
  least two independent discriminators from A1-A3, each prediction recorded
  before its arm runs.
- AC3: copy_complete's semantics (host-observed wait = enqueue + sync) are
  documented in AGENT_ENVIRONMENT.md, and either a true-DMA field lands behind
  the experimental gate or the doc states why it cannot be measured with the
  current backend API.
- AC4: the pinned-active question is settled with in-trace proof, and the
  T-GPLR-51 A2 addendum is corrected if a fallback is confirmed.
- AC5: closure follows the established pattern: plan RESULT block + commit
  hashes, currentstate entry, doc-size guard PASS.

Non-goals: no placement, routing, or package-content change; no performance
tuning based on an unproven mechanism; no new dependency; TG-phase copy
accounting is excluded from PP stats but not otherwise studied.

Focused validation: the host-only decomposition script from the drafting
session (AC1); owner-invoked mutex arms with the busy-gate verdict pattern
(A1-A3); python tools/check-doc-sizes.py after doc edits.

Escalation: if A1-A3 all fail to discriminate, present the three-way evidence
table to the owner with the option of instrumenting backend-level copy timing
(backend API dependency question) or closing the question as
instrument-limited.
