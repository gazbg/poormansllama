# Generic prefill layer residency — executable backlog (live tasks only)

SPLIT 2026-09-20: the 18 closed/completed task blocks moved VERBATIM to
`docs/plans/generic-prefill-layer-residency-archive.md` (mechanical record:
docs/plans/archive-manifest.json, including the byte-exact reconstruction check).
RESPLIT 2026-09-21: the blocks closed since then (T-GPLR-39, T-GPLR-50,
T-GPLR-51) moved to the archive the same way; mechanical record:
docs/plans/resplit-manifest-2026-09-21.json
(python tools/resplit-gplr-plan.py --verify re-proves the CURRENT state;
the 2026-09-20 tool's --verify is historical and no longer applies).
This file holds ONLY live work (in-progress or draft). Index of archived
blocks (original order preserved in the archive):
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
- ### T-GPLR-39 (draft): make the `other` ledger bucket chunk-independent — the residue that grows with ubatch size
- ### T-GPLR-50 (draft): make the f16 rebuild default follow the charter shape — ON where the pass-2 excess is measured (flip the T-GPLR-48 f16 default back to ON)
- ### T-GPLR-51 (draft): attack the EXPOSED transfer at the charter shape - copy-tail and hiding robustness at 200k

Rules (same convention as the currentstate split, AGENT_ENVIRONMENT.md):
- Closed tasks are archived, never edited in place; a new closure appends its RESULT
  in the live file first, and the block moves to the archive on the NEXT split
  (python tools/resplit-gplr-plan.py --apply, which also refreshes the index).
- The archive's 6-line header is not original bytes; strip it when reconstructing.
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
