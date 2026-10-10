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
- ### T-GPLR-37: direct mmap-sourced staging — remove the CPU hop from bank staging
- ### T-GPLR-52 (draft): attribute the shape-dependent per-copy staging cost (identical bytes, 18x wall) and correct the T-GPLR-51 copy-tail framing
- ### T-GPLR-54: single-chunk prefill identity - is the u4096 7-id tail insertion a numerics tie-break or a 1-chunk shape defect? (falsifiable ID-CHECK + a logit-margin discriminator)

Rules (same convention as the currentstate split, AGENT_ENVIRONMENT.md):
- Closed tasks are archived, never edited in place; a new closure appends its RESULT
  in the live file first, and the block moves to the archive on the NEXT split
  (python tools/resplit-gplr-plan.py --apply, which also refreshes the index).
- The archive's 6-line header is not original bytes; strip it when reconstructing.
### T-GPLR-53 (draft): attribute the pass-1 embed-step excess - the whole of the sweep's "unattributed" gap - and pre-register the first-touch decay discriminator (host-only first)

Status: CLOSED (owner decision, 2026-09-25) - A1-A4 evidence is complete and retained. The first-prefill excess is accepted as a startup/registration cost rather than a blocker for the long-lived multi-turn server target; no T-GPLR-53 lever is adopted, no A5 is authorized, and no further GPU run is required. The pass-1 wall remains conserved; the pre-registered excess-based reporting correction and all recorded verdicts remain unchanged.
Dependencies: T-GPLR-52 (its A1/A2/A3 verdicts are inputs AND correction targets: the
queue-behind-compute law stands, but "staging and compute are flat across passes"
describes the LAYER sweep only - the pass-1 excess lives before it); T-GPLR-37 (the
mmap-registered default is the probed mode, and its promoted flip is why every real
first prefill now pays this); T-GPLR-36 (the ledger found double-booked here); the
2026-09-21 anomaly sweep (currentstate item 5 is the input this task adjudicates).
Commit: nested 043fceedc (instruments: A4 pre-warm modes, recon-gated embed_chunk, recon generation join) + 36f0ae090 (whitespace fixup); outer 050f8f9 (analyzers, checker, A3/A4 arm drivers), ea1ab5d (host-suite registrations + plan-split guard), f2c385f (plan/env/state records), b716289 (plan whitespace); T-GPLR-53 band settlement outer ccdd0b7 (A4 reporting fix: the band is the excess, the wall printed as the relocation instrument with its ceiling)

Goal. Name what the "unaccounted" pass-1 gap is (wall - sum layer compute - sum layer
wait = 2.35 s at 50k pass 1, 2.83-2.88 s at 200k pass 1, 0.16/0.64 s later), explain
why it is pass-1-only, why it changes bucket with --copy-source, and what its mechanism
is - with the decay-with-chunk-index discriminator pre-registered, host-only first, and
no default change implied by this block.

Scope. Attribution + mechanism of the pass-1 excess in the layer-major PP loop; one
--pp-recon-gated per-chunk telemetry field; optionally one pre-warm probe flag; the
T-36 ledger-identity correction. Non-goals: no placement, package, routing or KV
change; no default change; no fix to the pinned path's own first-pass cost (recorded,
not owned); no cold-cache claim (Windows file cache is uncontrolled); no new dependency;
TG-phase accounting is not revisited.

Established facts (host-only re-decomposition of existing traces, 2026-09-21; no GPU
time; gap = phase_summary.wall_us - total_compute_us - total_wait_us; all values ms):

- F1 - The pass-1 excess IS the embed step (`host_ledger.embed_us`), not a missing
  bucket. mmap-registered arms, pass 1 -> pass 2 gap vs embed: 50k a2 2350.8 -> 161.4
  (+2189.4) with embed 2246.3 -> 62.5 (+2183.8 = 99.7% of the excess); 200k a2 2877.9
  -> 637.4 (+2240.5) with embed 2482.8 -> 249.2 (+2233.6 = 99.7%); 200k a3-mmap
  2835.3 -> 651.3 (+2184.0) with embed 2438.2 -> 251.0 (+2187.2); 4k gplr51-4k-mmap
  1790.1 -> 21.2 (+1768.9) with embed 1774.1 -> 5.6 (+1768.5). The residual rest = gap
  - embed - preload - output is PASS-INVARIANT in every mmap trace (200k a2
  372.0/367.4/373.6 across three passes; 50k a2 97.8/93.3/94.6; 4k 15.3/15.0), so
  nothing else in the pass scales with pass index.
- F2 - The excess is inside the embed graphs' SYNC-BOUNDED execution, not host prep.
  At 200k a2 the ledger's host buckets are flat pass 1 vs pass 2: setup 193.75/192.98,
  patch 18.78/18.53, graph_call 16.34/15.89, slot 70.37/70.35; the ledger compute
  bucket minus phase total_compute (= the embed step's own `compute_us` = launch +
  execute + sync, `run_graph` measures t_after_alloc..t_after_sync) is 2498.7/265.0.
  Timeline corroboration: pass_begin -> first copy_submit is 2.60 s in pass 1 and 0.37 s
  in passes 2/3 (a2-200k); 2.56 vs 0.37 (a3-mmap). The excess is entirely before any
  staging.
- F3 - Source contrast, same session, adjacent arms, same binary (a3-mmap 17:44 vs
  a3-pinned 17:49, 200k/3 slots): the pinned arm has NO embed excess (251.1/248.6) and
  pays its first-pass cost in the COPY path instead - slot 1827.8/322.4, preload
  327.8/123.2, copy_complete host_submit p50 110.0 ms/bank pass 1 vs 42.6 pass 2 (the
  mmap arm's host_submit is 0.124/0.100 ms). Net pass-1 excess over pass 2: mmap
  +2184 ms (wall 136,798.5/134,672.3) vs pinned +1711 ms (137,043.0/135,534.6). Both
  sources pay a per-run first-pass cost of the same order; the bucket differs.
- F4 - The excess is a per-run one-off, only weakly dependent on chunk count: 4k (8
  chunks) 1.77 s, 50k (98 chunks) 2.18 s, 200k (391 chunks) 2.19-2.23 s; every traced
  mmap arm pays it, including arms run minutes after another mmap arm had read the
  whole file. Pinned 4k pays ~3.0 s the same way (slot 3667.4/581.9).
- F5 - Registration is deferred and its work lands INSIDE pass 1, invisible to the DMA
  brackets: `mmap_register_init` reports 18,500,517,888 B in 2345 us pre-timed, while
  the process file-backed working set goes 2.10 GB at pass_begin 1 -> 20.43 GB at
  pass_end 1 and is dead flat in passes 2/3 (sys_free 53.8 -> 35.2 GB); the pinned arm
  shows the same +18.3 GB WS move over its memcpy path (2.09 -> 20.43 GB, 52.3 ->
  33.6 GB). No DMA stalls anywhere: recon_attr copy_gpu_ms p50 31.5 ms / max 36.3 ms in
  pass 1 as in pass 2 (85 windows; the first three preload banks 31.4-33.6 ms).
- F6 - The T-36 ledger double-books embed and output: led_other = led_sweep -
  (patch+setup+compute+graph_call+slot), and those buckets accumulate the embed (step 1)
  and output (step 4) launches while led_sweep starts AFTER the embed step. Measured
  identity over four traces: `loop_us + embed_us + output_us` = +0.10/+0.16/+0.13 ms at
  50k a2 passes 1/2/3 and +0.73/+0.73/+0.70 ms at 200k a2 passes 1/2/3. T-36's
  "closes exactly" reading was made when route telemetry supplied a +4.5 s `other` that
  masked the -embed; since T-39/T-45 the same algebra reads other_us ~= -embed_us. So
  the sweep's "unaccounted" is not an unnamed bucket.
- F7 - Code fact that kills the obvious page-in story for the embed step: the runner
  loads with n_gpu_layers = -1 and overrides ONLY the pageable class to the CPU buffer
  (tools/layer-streaming/main.cpp:9-11, 3559-3578), so token_embd/output_norm/output
  are CUDA-resident (LS_CLS_STATIC at llama-layer-streaming.cpp:6676 is an accounting
  class only, llama-layer-streaming.cpp:7059). The embed step reads no host-mapped
  model page; what is left of the registration is the 18.5 GB registered SPAN itself
  (F5), whose completion is what pass 1 is paying for.

Hypotheses (pre-registered; each arm states which prediction it tests BEFORE it runs):

- H1 (deferred registration completion, front-loaded): cudaHostRegister defers the real
  work (populate/lock 18,500,517,888 B + GPU-side mapping of the span) and the first GPU
  submission sequence after it - the embed step - absorbs it. Predictions: (a) the
  per-chunk embed profile is front-loaded and decays to the warm band (200k warm =
  0.64 ms/chunk; 50k 0.63; 4k 0.70); (b) a pre-timed CPU touch of the registered span
  before the timed pass removes >= 70% of it; (c) its absolute size tracks the
  registered bytes, not the prompt (already ~flat, F4); (d) no host bucket inflates
  (already measured flat, F2).
- H2 (background pinning contention): the driver pins on host threads right after
  registration, contending with the host's pass-1 launch/sync path. Predictions: (a)
  the decay is smooth and spread over much of the embed step rather than a spike; (b)
  the host-adjacent windows inflate a little too - F2 measures the ledger setup bucket
  flat to 0.77 ms across 391 embed launches, so this prediction is already effectively
  disconfirmed; record it rather than assume it; (c) a CPU pre-touch removes it only if
  the trigger is page-side.
- H3 (one-time driver/module state at the first submissions after registration):
  Predictions: (a) the excess sits in the FIRST launch(es) - chunk 0 within noise of the
  total, the rest at the warm band; (b) a pre-timed dummy GPU launch + sync removes it
  with no page touched; (c) a CPU pre-touch alone does NOT remove it.
- H4 (host-IO starvation), kept as a falsifier target only: predicts the host buckets
  AND the layer sweep's first-pass compute inflate - both contradicted by F2 (flat) and
  by the layer compute being flat across passes; revisit only if A3's profile is flat
  and spread.

Discriminating arms (A1 host-only and free; A2 host-validated before any GPU time; A3
and A4 owner-invoked under the GPU mutex, one session each):

- A1 (host-only, no GPU): `tools/analyze-gplr53-gap.py` - the F1-F6 decomposition as a
  checked tool over the named trace set (gplr52-a2-50k, a2-200k, a3-mmap, a3-pinned,
  gplr50-200k-on/off, gplr50-ctl-4k-f16, gplr51-4k-mmap, gplr36-remeasure-200k,
  gplr51-s2-2slot, gplr37c-{pin,ron}-s{2,3}), with `--selftest` fixtures and exit 2
  (NOT a pass) when a trace carries no host_ledger. AC: the per-pass table
  [wall, gap, embed, preload, output, rest] for every trace; the identity
  |gap - (embed+preload+output+rest)| <= 2% and |loop_us + embed_us + output_us| <= 5 ms
  on every ledger pass; the source-contrast row set (mmap vs pinned, same session); a
  written verdict on which of H1/H2/H4 survives the existing data (H4 should
  not; state it). Deliverable also: the placement check (F7 re-derived from the code,
  quoted with file:line). No new GPU time.
- A2 (instrument, host-validated first): emit the per-chunk embed timeline as a new
  event `embed_chunk` {chunk, n_tokens, setup_us, compute_us, span_us, t_us} under the
  existing `--pp-recon` gate ONLY (default path byte-identical in behavior; no new
  flag, no ledger change - F6 makes the identity derivable). Concrete steps: emit at
  the step-1 loop in `llama_layer_major_pp` beside the existing per-launch brackets;
  keep it inside the recon gate so a non-recon run stays silent (the T-GPLR-49
  lesson). AC: a synthetic fixture trace carrying `embed_chunk` passes BOTH validators
  (unknown-event tolerance re-proven, not assumed); the fixture's per-chunk sums
  reconcile to `embed_us` within 1 ms; host CTest suite green; no arithmetic, routing
  or package change (no identity arm needed for a telemetry-only field, but the
  checker must be run on the A3 gear anyway).
- A3 (the decay discriminator, GPU mutex, owner-invoked, one session): 200k and 50k,
  explicit `--copy-source mmap-registered` (not defaulted - the resolved source must be
  unambiguous), 3 slots, af=1, kv f16, ubatch 512, `--pp-repeat 3`, `--pp-recon 1`.
  Pre-registered decision rule, computed per pass: (i) share of the pass-1 embed excess
  in the first 10% of chunks and Spearman rho(chunk, per-chunk time - pass-2 median);
  (ii) rho <= -0.7 AND first-decile share >= 50% -> front-loaded: H1 or H3 (H1-vs-H3
  by A4's modality); (iii) |rho| < 0.3 or a uniform ~5-6.4 ms/chunk profile -> the
  page/driver stories are refuted and the question escalates; (iv) a chunk-0-only spike
  with the rest at the warm band -> H3. Artifacts: traces + verdicts as usual,
  gpu_pre/post_run recorded, strict --check-telemetry OK.
- A4 (modality discriminator + the payoff measurement, owner-invoked, conditional on
  A3): at 4k - where the excess is 1.77 s of a 3.55 s pass-1 wall, i.e. the shape where
  a real single-prompt prefill IS pass 1 - three pre-timed pre-warm variants behind one
  new experimental flag (0 = off default): (i) CPU read-touch of the registered spans
  before the timed pass; (ii) a dummy GPU launch + sync before the timed pass; (iii)
  both. Pre-registered: (i) removes >= 70% -> page-side completion (H1); (ii) removes
  it -> submission-side (H3); neither -> the cost attaches to the first execution of
  the embed graph set and the hypothesis set is incomplete (escalate; do not tune).
  The same-session pinned control and a repeat-2 route-parity arm accompany it; no
  package bytes, routing or graph content change.
- **SETTLED 2026-09-23 (owner directive; host-only, no GPU, no arm re-run): the `>= 70%`
  band's QUANTITY is the per-chunk embed EXCESS removal against the same-session off
  control** - `(ctrl_excess - arm_excess) / ctrl_excess` - i.e. option A of the ESCALATION
  section below. That is what the falsifier paragraph and the branch wording always
  described; it is what the code did not compute. The pass-1 WALL is not a band and cannot
  be one: pass 1 = pass 2 + the one-off, so its removable share IS the excess and its ceiling
  is `100*(1 - pass2/pass1)` - 56.2% at 4k, 10.5% at 50k, 0.86% at 200k. A wall band is
  therefore unreachable at EVERY shape by construction. The wall is kept as the RELOCATION
  instrument (pre-pay + pass-1 wall read as a SUM) and its removal % may only be printed
  beside that ceiling. Option B (a shape-relative wall band) was rejected as option A with
  an extra indirection - the ceiling is itself derived from the excess; option C (drop the
  band, prose per run) was rejected as the least automatable of the three. No threshold value
  changed, no recorded run was re-scored, and the A3/A4 conclusions and the 12:44Z/13:10Z
  withdrawal stand unchanged: they rest on the excess table, not on the wall line.
  The reporting fix that lands this is in `tools/run-gplr53-a4-arm.ps1`: both quantities per
  arm on ONE line (excess first, wall second), the ceiling printed once before any branch
  reads a wall number, and the FALSIFIER verdict evaluated on the excess.

Acceptance criteria:

- AC1 (attribution, host-only): A1's tool and table land and REPRODUCE the F1-F6
  identities within tolerance; both corrections below already landed in the drafting
  commit and only need re-derivation, not re-writing: `docs/currentstate.md` item 5 is
  marked superseded with the embed-step attribution (the T-36 ledger double-booking is
  recorded in `docs/AGENT_ENVIRONMENT.md`, "host_ledger closure identity", with the read
  rule: never interpret other_us/loop_us alone - add embed_us + output_us back, or read
  `loop_us + embed_us + output_us` as the closure residual).
- AC2 (mechanism): at least two INDEPENDENT pre-registered discriminators support the
  named mechanism (profile from A3 + modality from A4 + the F4/F5 magnitudes), each
  prediction recorded before its arm runs. A well-evidenced "front-loaded but the two
  page/driver variants remain unseparated" is an acceptable AC2 outcome ONLY if A4's
  removal tests actually ran.
- AC3 (payoff stated or explicitly not claimed): if A4(i) removes the excess, record
  the pass-1 wall delta at 4k against the same-session control and state the real-PP
  implication; if nothing removes it, record the non-removability and claim NO
  performance result (the T-37/T-52 discipline: no claim from a confounded arm, no
  default change from this block alone).
- AC4 (no behavior change): the instrumentation is recon-gated; every recorded command
  resolves identically; strict checker + route parity (on any repeat-2 arm) + host
  CTest suite + doc-size guard.
- AC5 (closure): RESULT block with the arm verdicts against the pre-registered bands,
  commit hashes, currentstate entry, and the usual resplit hygiene if a block is
  archived at the same time.

Focused validation:
: * `python tools/analyze-gplr53-gap.py --selftest` (fixtures: a clean mmap pass pair,
  a pinned pass pair, a trace with no host_ledger -> exit 2, not a pass)
* `python tools/analyze-gplr53-gap.py <traces...>` re-deriving F1-F6 on the record
* `python tools/check-doc-sizes.py` after the doc edits; host CTest suite after A2
* A3/A4 through the standard arm harness (verdict files with gpu_pre/post_run, started/
  done sentinels, strict --check-telemetry)

State/document updates:
: `docs/currentstate.md` (corrected item 5 + the T-53 draft note); `docs/AGENT_ENVIRONMENT.md`
  (the ledger identity rule); this block's Status and Commit fields; plan status only -
  no default changes are proposed by this task.

Escalation condition:
: If A3's profile is flat (|rho| < 0.3) or both A4 pre-warm modalities fail to remove a
  meaningful share, present the profile + modality + host_mem evidence table to the
  owner with the two options that remain: the A5-style probe (defer registration to
  the first slot use so its completion lands in the first copy instead of the embed
  step - deliberately NOT in this contract, because it reorders init work), or closing
  the question as an unavoidable per-run registration cost with the charter note that
  it is a fixed ~2 s off-timed-path candidate for a future pre-warm feature.

#### A1 result (2026-09-21, host-only, no GPU): F1-F6 REPRODUCED as checked expectations; the gap is the embed step at every shape and BOTH sources; H4 disfavored, H1/H2/H3 wait on A2-A4

Tool `tools/analyze-gplr53-gap.py` (`--selftest`, `--placement`, `--json`), owned
bands in its docstring, table captured at
`build/llama-cuda/gplr53-a1-gap-table.txt`. 14 traces, tool exit 0; selftest 8/8
with the negative controls actually firing: mmap pair whose excess sits in slot ->
CHK-embed FAIL; pinned pair with an embed-carried excess -> CHK-pin FAIL; 50 ms
ledger drift -> CHK-led FAIL; trace without `host_ledger` -> REFUSED (exit 2 on the
real pre-ledger trace `gplr35-slot3`, exit 2 for a missing tag); pre-T-39 ledger ->
CHK-led SKIP. `--placement` PASS, F7 re-derived from the code, not restated:
main.cpp:3559 `mparams.n_gpu_layers = -1;`, :3567 the routed-expert-bank override
pattern, :3575 the `tensor_buft_overrides` wiring, llama-layer-streaming.cpp:6676
`LS_CLS_STATIC = 0,` (accounting class), :7059 the token_embd/output classification.

Measured, all 14 traces: CHK-gap structural residual **0.000 us**; CHK-led
|loop+embed+output| **<= 0.86 ms** on every ledger that carries `loop_us` (11 of 14)
- F6 is exact, and the four `gplr37c-*` traces predate the field, so F6 is NOT
evaluable there: that is a refinement of the drafting wording ("four traces"), not
a failure, and the tool now encodes the skip.

F1, embed share of the pass1->pass2 gap excess (mmap): 0.9969 (a2-200k), 0.9974
(a2-50k, gplr50-200k-off), 1.0014 (a3-mmap), 0.9958 (gplr50-200k-on), 0.9915
(s2-2slot), 0.9998 (4k gplr51-4k-mmap), 1.0264 / 0.9710 (gplr37c-ron-s2/-s3).
Current-binary band **0.9915-1.0014**; the two T-37-era 200k ron traces sit at
0.971/1.026 because their patch/slot buckets still carried T-44-era host work. F1's
worded claim is therefore "the gap excess IS the embed step to >= 95%" (one-sided,
the tool's band), with 99.6-100% the mid-band on the current tree.

F3, pinned: embed share 0.0015 (a3-pinned), 0.0247 / -0.0008 (gplr37c-pin-s2/-s3),
0.0004 (4k pinned) - all <= 0.05, with slot+preload carrying +1710.0 ms of
(a3-pinned's) +1710.8 ms gap excess and +3280.8 ms of the 4k pinned +3215.7. So the
source contrast is not "pinned is spared": both sources pay a ~1.7-3.3 s per-run
first-pass one-off; the bucket differs (embed vs copy path). F1b rest drift <= 5.2%
on every mmap trace (a2-200k 1.7%, a2-50k 4.7%, a3-mmap 1.3%).

gplr36-remeasure-200k is CONFOUNDED (pass 2 = the T-46/T-47 compute regression,
wall 206 s): table only, checks skipped, and its 1.087 embed share is NOT a
falsification of F1.

Verdicts against the pre-registered hypotheses: **H4 (host-IO starvation)
DISFAVORED** - A1's own test was "the host buckets inflate too", and they are flat
pass-over-pass on every trace while the excess is isolated to the embed step's
sync-bounded span; the tool now enforces flatness as an expectation rather than a
claim. **H1 / H2 / H3 are NOT separable here**: no existing trace carries per-chunk
embed timing (all 782 `act_bind` events are pre-pass, `layer_compute` covers layer
launches only), so the decay discriminator requires A2's `embed_chunk` event and the
page-vs-submission split requires A4's modality arms. H2's host-adjacent-inflation
prediction is already disconfirmed (setup flat to 0.77 ms across 391 embed launches).

AC1: the attribution half is DISCHARGED (tool + table + identities + placement
check); the doc half landed in the drafting commit (currentstate item 5 superseded;
AGENT_ENVIRONMENT "host_ledger closure identity"). The selftest is wired into
`tools/run-tests.ps1` for every host-only suite, beside the doc-size guard, and was
observed there: `-Suite host` -> doc-size PASS, `[t53-a1]` SELFTEST PASS, 37/37 CTest.
Evidence: `tools/analyze-gplr53-gap.py`,
`build/llama-cuda/gplr53-a1-gap-table.txt`, and the F7/`--placement` anchors above;
selftest fixtures are transient (`build/llama-cuda/gplr53-a1-selftest-*.jsonl`,
removed on exit). Next: A2 (the recon-gated `embed_chunk` event), then A3/A4 under
one owner-invoked mutex hold.

#### A2 result (2026-09-21, host-validated): the recon-gated `embed_chunk` event lands, and the rider the A2 gate needed - the recon generation off-by-one - is fixed and pre-characterized

Instrument (engine, one event, one gate). `tel_embed_chunk(chunk, n_tokens,
setup_us, compute_us, span_us)` in `llama-layer-streaming.{h,cpp}`, called from the
step-1 loop of `llama_context::layer_major_pp` beside the existing per-launch
brackets, emitting
`{"event":"embed_chunk","run_id","chunk","n_tokens","setup_us","compute_us","span_us","t_us"}`.
`span_us` is the chunk's own wall (prepare + graph call), so a pass's sum of
`span_us` reproduces `host_ledger.embed_us` instead of estimating it. The emitter
self-gates (`if (!pp_recon) return;`), so a non-recon run emits nothing: no new
flag, no ledger field, and no arithmetic/routing/package change anywhere. A1's
tool still PASSES on the recorded traces after the change (re-ran gplr52-a2-50k +
a3-mmap; F1 1.0014, F6 0.86 ms, F1b 1.3%).

Rider - the recon generation join (plan A2 gate, item 4 of the sweep):
`stage_members` now receives the generation the copies below will be tagged with
and passes it to `recon_begin_bank`, instead of reading `slot.generation` before
its increment (both call sites already computed it; `llama-layer-streaming.cpp`).
Measured before the fix (existing traces, host-only): `recon_attr.generation` minus
the generation of the copy it brackets is **-1 on all 85 paired samples** of
`gplr52-a3-mmap` (and -1 x63 plus +12 x2 on `gplr50-ctl-4k-f16`, an unresolved
pre-fix curiosity RESOLVED 2026-09-21, see the A2 result addendum: it was the
checker's positional pairing, not a second stamp defect). A cross-event join on
(layer, generation) therefore matched NOTHING before this change - the sweep's
item 4 - and is exact by construction after it. Positive proof of the fix needs a
post-fix trace: A3 must re-run `tools/check-gplr53-embed-chunks.py <trace>
--join` and expect drift 0.

Fixture + both validators (measured, not assumed).
`tools/layer-streaming/gen-gplr53-fixtures.py` builds
`tools/layer-streaming/tests/gplr53-embed-chunk.jsonl` from the recorded
`gplr52-a2-50k` metadata and ledger numbers: 4 chunks carrying the front-loaded
pass-1 profile (0.62/0.18/0.13/0.07 of the measured 2,246,290 us embed) whose
`span_us` sums to `embed_us` exactly, plus a `loop_us = -(embed + output)` ledger
so the A1 identity holds on it too. Two new CTest targets
(`layer-streaming-check-gplr53-embed-chunk`, `-validate-gplr53-embed-chunk`) run
BOTH validators on a build-dir copy: both exit 0, i.e. the validators tolerate the
new event (forward compatibility re-proven for this event, and the fixture is a
schema-valid streaming trace: `check-telemetry: OK (8 events, 3 slots)`,
`validate-telemetry: OK ... -> gplr53-embed-chunk.jsonl.summary.json`). The
fixture is one pass, not two: `--validate-telemetry` permits one phase_summary per
phase and a multi-pass run emits one per pass (measured: a second pp summary is
rejected as a duplicate), so a two-pass fixture could not satisfy both validators;
multi-pass shapes are covered by the checker's own fixtures.

Checker (the A2 acceptance tool). `tools/check-gplr53-embed-chunks.py`:
CHK-shape (fields, non-negative times, setup+compute <= span), CHK-order (chunk ids
contiguous 0..k-1 per pass, `t_us` monotonic), CHK-cover (equal pass sizes and
`pp_n_chunks` when declared), CHK-sum (per pass `|sum(span_us) - embed_us| <=
1000 us`, the pre-registered tolerance), CHK-gate (no `embed_chunk` in a trace whose
run_start declares the streaming path off), plus `--join` for the generation join.
`--selftest` == 15/15 with seven negative controls firing (bad sum, index gap,
duplicate index, span below setup+compute, off-run event, count-vs-manifest,
join drift -1) and four exit-2 refusals (no event, no ledger, no run_start,
group/ledger mismatch: "pairing is refused, never guessed"). On the checked-in
fixture: `CHK-sum diff 0.0 us` for the pass. Wired into `tools/run-tests.ps1` for
every host-only suite (`[t53-a2]`), and the two CTest targets joined the `host`
suite's include list.

`--join` pairing hardening (2026-09-21 addendum, host-only): the join is now
KEYED by (layer, ubatch) instead of positional-within-layer. The positional
form was wrong in a multi-pass, handoff-enabled trace: submits span both PP
passes AND the TG re-staging burst (98 per layer on the 2-pass 4k control), so
a layer whose sampled banks skip a pass paired its sample against a foreign
copy and reported a phantom drift equal to the inter-pass stride minus the
stamp offset. That is exactly the recorded "+12 x2" on `gplr50-ctl-4k-f16`
(13 - 1 at two layers whose only samples were pass-2-only). Under the keyed
pairing, ALL 65 of its samples sit at the expected drift: -1 x64 (the known
pre-fix stamp) plus -15 x1, and gplr52-a2-50k shows -1 x119, -14 x3, -15 x3.
The -14/-15 class is the lazy-ring LABEL artifact, not a stamp defect: a pass's
last sampled banks sit in the ring until the next pass's first bank flushes
them, and the emitter prints the CURRENT `n_ubatches` as the sample's label
(src/llama-layer-streaming.cpp:1220 prints `n_ubatches`, not the sampled
event's), so those samples name the WRONG ubatch while their generation sits
at exactly -1 from their TRUE pass-1 copy (proven per-sample: every outlier's
generation matches its earlier-pass copy at -1, in both traces; the outliers
cluster on the last sampled layers at pass boundaries, the stale-ring
fingerprint). Consequences: (a) no second stamp defect exists on the record;
(b) `CHK-join`'s expectation is UNCHANGED (drift {0} post-fix, any nonzero
fails) and now fires on the true signal instead of cross-pass phantoms; (c)
the (layer, ubatch) -> one-submit uniqueness was verified on both traces (max
1 submit per key over 4,185 submits), so the keyed join is well-defined; (d)
if A3 wants per-ubatch sample attribution, the emitter's label should be the
SAMPLED bank's ubatch (one-line engine change, telemetry-only, defer to a
future task unless A3 needs it). `--selftest` is now 18/18: the three new
fixtures encode the measured gplr50 shape as negative controls (a TG-burst
layer pairing positionally onto its pass-1 copy must FAIL at the TRUE drift -1,
not the phantom +12; a genuinely wrong stamp fails; a missing-ubatch event is
REFUSED exit 2, not guessed). Re-measured on the real traces: gplr50 --join
FAILs naming "-15 x1, -1 x64 (65 paired, 0 unpaired)" (the TRUE pre-fix
signal), gplr52-a2-50k names "-15 x3, -14 x3, -1 x119".

AC mapping. AC (instrument, host-validated first) is DISCHARGED: both validators
accept the fixture (exit 0), the fixture reconciles within 1 ms (0.0 us), the host
CTest suite is green (39/39, was 37/37 + the 2 new targets), and no arithmetic,
routing or package byte changed (telemetry-only emitter + one generation argument;
A1 re-ran clean). NOT claimed here: any timing, any mechanism, any default. The
A3/A4 mutex session is the next and only GPU spend, and A3 must now also carry the
`--join` drift-0 check.

Evidence: `src/llama-layer-streaming.{h,cpp}` (nested),
`tools/layer-streaming/gen-gplr53-fixtures.py`, `tests/gplr53-embed-chunk.jsonl`
(+ its `.summary.json`), `tools/layer-streaming/CMakeLists.txt` (2 CTest targets),
`tools/check-gplr53-embed-chunks.py`, `tools/run-tests.ps1` (`[t53-a2]` + host
suite include). Commit: pending (no commit authorized).

#### Recon generation off-by-one join: DECIDED CLOSED, host-only from the 126 recorded traces (2026-09-22, no GPU, no arm) - the rider P0's A2 gate left unowned

T-GPLR-52 forwarded this as an unowned residue. It is decided here **from the traces on disk**, not
from the prose that claimed the fix. 126 traces in `build/llama-cuda/` carry `recon_attr` samples: 97
recorded before the A2 stamp fix, 28 after it.

| era | traces | `recon_attr` samples | exact `(layer, generation)` match vs `copy_submit` | match after a +1 generation shift | matched sample whose copy postdates it |
|---|---|---|---|---|---|
| pre-fix (gplr35 ... gplr52) | 97 | 7,379 | **0** | **7,379 (100%)** | 0 |
| post-fix (gplr53-*, gplr54-*, gplr56-*) | 28 | 2,281 | **2,281 (100%)** | 2,281 | 0 |

- **The pre-fix stamp was exactly one generation behind, and nothing else.** Every pre-fix sample
  matches a real copy of its own layer once the generation is shifted by +1 (7,379/7,379) and none
  matches unshifted - the numeric signature of `recon_begin_bank` being called with
  `slot.generation` before the increment. It also reproduces the record's sample counts
  (`gplr51-a3` 0/89, `gplr52-a2-50k` 0/125, `gplr52-a3-mmap` 0/85).
- **The live emitter is clean.** All 2,281 post-fix samples join exactly with **zero** ordering
  violations (no matched copy postdates the sample that names it). The re-keyed `(layer, generation)`
  join is what makes this decisive: the old `(layer, ubatch)` key produced a phantom one-pass-stride
  drift on the 6 samples per trace flushed across a pass boundary (layers 38/39), because `ubatch` is
  written from the LIVE `n_ubatches` at flush time rather than from the ring entry describing the
  measured copy. Under the current key that class is **0 across all 126 traces** - dead, not merely
  explained.

**The gate agrees on real inputs** (`tools/check-gplr53-embed-chunks.py <tag> --join`):
`gplr53-a3-50k` -> **PASS, `CHK-join ok`**; `gplr51-a3` -> **FAIL**, "89 sample(s) carry a generation NO
copy of that layer ever had - the pre-T-GPLR-53 stamp offset (first: layer 0 generation 0)". Pre-fix
traces therefore FAIL rather than being silently re-keyed, which is the required behaviour: **no tool
in `tools/` adds 1 to a recon generation** (checked), so an old trace stays honestly unbucketable
instead of being "repaired" into agreement.

**Disposition: CLOSED, with one standing negative requirement.** The concern is not open - the fix is
complete for the emitter and verified on every trace on disk. The only thing owed is that a future
pass must not add a +1 shim to a checker to make pre-fix traces pass; if a pre-fix trace is ever read,
the offset belongs in the analysis prose, never applied silently by a tool.

**Own-error note, recorded because it is the same class as the thing under audit:** this scan's first
run reported 271 pre-fix samples unmatched even after the +1 shift. That was a falsy-zero bug in the
scan itself (`(generation or -99) + 1` sends generation 0 to -98), not a second stamp defect;
corrected, the residual is **0**. Uncorrected, it would have written a phantom 3.7% residue into the
record as a finding.

#### A3 session attempt (2026-09-22): BLOCKED at preflight by the GPU busy gate - no arm ran, no verdict claimed; the two host-side prerequisites the blocker exposed are now in place

Owner-invoked A3 session attempt. It did NOT start: no arm ran, no trace was
produced, and nothing in this block's Status changes because of it. The blockers
are named, not inferred:

- **GPU busy, the documented poison condition.** `nvidia-smi --query-compute-apps`
  at 2026-09-22T10:35:12Z and again at 10:36:12Z showed the same foreign compute
  process resident and ACTIVE: pid **2224**,
  `C:\Development\AI\colibri\deepseek-v4-baseline\c\deepseek_v4.exe`,
  **11,929 MiB** of 24,463 MiB, GPU utilization **38%**. `with-gpu-lock.ps1`'s
  busy gate refuses (exit 125) while any compute process is visible, and AGENTS.md
  names this exact executable as the poison case, so the attempt was stopped
  rather than overridden: no `-AllowBusy`, no mutex acquire, and no CUDA process
  started by this attempt (zero GPU time).
- **A3 and A4 could not have started even on a free card** - both host-side, and
  both since fixed in the same change: `tools/run-gplr53-a3-arm.ps1` and
  `tools/run-gplr53-a4-arm.ps1` did not exist, and A4's pre-warm flag did not
  exist in the runner or the engine. The third item of the original preflight - a
  binary predating A2 - was NOT real: `bin/Release` was re-probed at 2026-09-22
  10:49 with `embed_chunk` present in `llama.dll`, so the session would have run
  a binary that carries A2's emitter.

Landed in this attempt (telemetry-only; no performance claim, no default change):

- `tools/analyze-gplr53-decay.py` - the A3 pre-registered decision rule as a
  checked tool (per-chunk excess against the later-pass median; FRONT_LOADED /
  FLAT / CHUNK0_SPIKE / MIXED / NO_EXCESS; exit 2 = nothing to evaluate and NEVER
  a pass). `--selftest` 6/6 with the negative controls firing (uniform -> FLAT,
  chunk-0-only -> CHUNK0_SPIKE, one pass -> REFUSED exit 2, no `embed_chunk` ->
  REFUSED exit 2). One honest root-cause fix during validation: the chunk-0 guard
  compared warm medians against THEMSELVES (always true), which made FRONT_LOADED
  unreachable and mislabelled the front-loaded fixture; the guard now tests the
  PASS-1 span against the warm band and the fixture reads FRONT_LOADED.
- `tools/run-gplr53-a3-arm.ps1` - the A3 driver: 50k + 200k in ONE mutex hold,
  arm-verdict handling (`Clear-ArmVerdict` + `Write-ArmStart`) before the run,
  strict `--check-telemetry`, `check-gplr53-embed-chunks.py <tag> --join` (the A2
  rider), then the decay rule; `-ValidateOnly` is a no-GPU dry run.
- `tools/run-gplr53-a4-arm.ps1` - the A4 driver: the pre-warm modalities at 4k
  (mode 0 control, 1, 2, 3) plus the same-session pinned control and the repeat-2
  route-parity arm, in one mutex hold; the driver asserts that mode 0 emits no
  pre-warm line and that a non-zero mode does.
- `tools/run-gplr53-session.ps1` - the single documented command:
  `tools/run-gplr53-session.ps1 [-Session a3|a4|both]` launches A3 then A4 (one
  mutex hold each, the contract's "one session each"), invalidates every tag in the
  LAUNCHER before `Start-Process` per the detached-launch rule, then waits on the
  sentinels and prints a roll-up.
- Engine + runner: the A4 flag `--pp-prewarm N` (0 = off DEFAULT, 1 = CPU
  read-touch of the package source spans, 2 = dummy GPU launch + event sync,
  3 = both), executed by the runner BEFORE the timed PP region and on pass 1 only,
  so A4 measures what moved OUT of the pass-1 wall;
  `llama_layer_streaming::{set_pp_prewarm,pp_prewarm_run}`. Off (0) runs nothing (no
  touch, no allocation, no submission) and every existing shape stays byte-identical;
  a non-zero mode requires `--pp-schedule layer-major` (refused, exit 2, outside it)
  and `--residency off` refuses it through the existing off-conflict table.

Host-side validation of the pieces (no GPU, no CUDA init, no mutex - all observed):

| check | result |
| --- | --- |
| `python tools/analyze-gplr53-decay.py --selftest` | PASS 6/6 |
| `python tools/check-gplr53-embed-chunks.py --selftest` | 18/18 |
| `python tools/analyze-gplr53-gap.py --selftest` | PASS (A1 unaffected) |
| `python tools/check-doc-sizes.py` | PASS |
| `tools/run-gplr53-session.ps1 -ValidateOnly` | DRY-RUN OK: both sessions, 8 arms, exact commands printed |
| `--pp-prewarm 4` | refused, exit 2, message observed |
| `--pp-prewarm 1 --pp-schedule batch-major --residency moe-expert-bank` (real GGUF) | refused, exit 2, message observed |
| the same shape with `--pp-schedule layer-major` | NOT refused for prewarm (negative control); the run then refuses later on an unrelated arch mismatch |
| `--residency off --pp-prewarm 1` | refused by the off-conflict table, message observed |
| `cmake --build ... --target llama-layer-streaming` | green; exe FE3B0A108C1D9BC9, llama.dll 13016F511FAB7F6D |
| `cmake --build ... --target test-layer-streaming-layout` (consumer rebuild) | green; final pair exe FE3B0A108C1D9BC9, llama.dll **8A9F3D83F24D0FEC** - the hashes the drivers record today |
| `ctest -R ^test-layer-streaming-layout$` with `CUDA_VISIBLE_DEVICES=""` | **Passed 1/1** after the consumer rebuild (it was the 1 failure in the earlier host-suite run, and its cause is section 2 of the hardening note below, not the GPU holder) |
| `grep -c -- --pp-prewarm` over the A4 `-ValidateOnly` output | **0 before** the `$a`/`$A` fix, then exactly `--pp-prewarm 1`, `2`, `3` and `--route-telemetry 1`, with neither flag on `pw0`/`pin` |

The pre-change binary pair was side-copied to
`build/llama-cuda/bin/pre-gplr53-prewarm/` before the rebuild (the streaming code
lives in `llama.dll`, so an exe-only copy would have been a silent control), and
every arm driver records both hashes in its verdict.

**A3 and A4 remain NOT RUN.** The next session is one command, and the busy gate -
not this driver - decides whether it may start.

#### A3/A4 driver hardening (2026-09-22, later): the A4 dry run was silently dropping the flag, and the earlier CTest "foreign process" attribution was WRONG

Owner-directed follow-up on the same attempt: the two host-side prerequisites the
blocker exposed, plus honest correction of one wrong attribution. Still no arm ran;
Status is unchanged and no performance claim is made.

**1. Defect found in my own A4 driver, fixed (host-only, observed).** In
`tools/run-gplr53-a4-arm.ps1`'s `Get-ArmArgs`, the parameter `$A` and the
accumulator `$a` are THE SAME VARIABLE - PowerShell variable names are
case-insensitive. The array literal's reads (`$A.repeat`, `$A.source`) are correct
because the right-hand side is evaluated before assignment, but every
POST-assignment read tested the array instead of the hashtable:
`if ($A.prewarm -gt 0)` and `if ($A.route)` were therefore always false, so
`--pp-prewarm` and `--route-telemetry` were NEVER appended to any A4 command. The
whole A4 session would have run six identical mode-0 arms; the driver's own
"mode set but no pre-warm line" assertion would have failed all three modality
arms - loudly, but only after spending the mutex hold. The source reads correctly,
which is exactly why the dry run found it and inspection did not: measured before
the fix, `grep -c -- --pp-prewarm` over `-ValidateOnly` output was **0**; after
renaming the accumulator to `$out`, the same command yields exactly
`--pp-prewarm 1`, `--pp-prewarm 2`, `--pp-prewarm 3`, `--route-telemetry 1`, and
neither flag on the two off arms (`pw0`, `pin`).

**2. Wrong attribution corrected: the layout-test CTest failure was NOT the
foreign process - it was my own partial build.** The earlier note in this record
read the `test-layer-streaming-layout` teardown crash (exit `-1073740791`,
0xC0000409, AFTER "all layout checks passed") as the documented poison-process
signature. That was wrong. The test is host-only C++ (`main()` only; no backend,
no CUDA), so an A/B was run with `CUDA_VISIBLE_DEVICES=""` and the side-copied
pre-change DLL: **pre-change dll exit 0, current dll crash**. Cause:
`tests/test-layer-streaming-layout.cpp:341` constructs `llama_layer_streaming ls;`
on the stack, and my build used `--target llama-layer-streaming`, which skipped
that consumer's translation unit - the Sep-21 exe allocated the pre-member-size
object while the current dll's constructor writes the new member. Rebuilding the
consumer (`--target test-layer-streaming-layout`) gives **exit 0 with BOTH dlls**,
so there is no defect in the streaming change. Lesson recorded because it is
generic: a header change invalidates every consumer, and a target-scoped build is
not a build - and a crash that survives a deliberate CUDA-off run is never
evidence about a GPU holder.

**3. Guard re-verification against the current binary (host, no CUDA init).**
Range guard `--pp-prewarm 4` -> refused, exit 2, message observed; residency-off
guard `--residency off --pp-prewarm 1` -> refused by the off-conflict table, exit
2; off default (no flag) -> parses through to model load, no pre-warm chatter.
One honest limit: the `--pp-schedule layer-major` requirement sits AFTER model
load, so it could not be re-exercised while the card is busy; it was observed
earlier in this attempt with the real GGUF, and the A4 driver always passes
`--pp-schedule layer-major` anyway, so no session depends on it.

**4. Second preflight probe (not an arm).** The card is busy again, and with a
different instance than the A3 attempt: pid **13720**,
`C:\Development\AI\colibri\pf06-wt\c\deepseek_v4.exe`, **18,751 MiB** of 24,463
MiB, **88%** utilization (2026-09-22, mutex not acquired, `-AllowBusy` not passed,
zero GPU time spent). **A3 and A4 remain NOT RUN**; the busy gate, not this
driver, decides when the one-command session may start.

#### P0 session attempt #2 (2026-09-22, 11:13-11:16Z) - still BLOCKED by an ACTIVE foreign compute cycle, plus a fail-fast hazard found before it could bite

Owner-directed attempt of the documented one-command session. It did NOT start: no
arm ran, no trace exists, no verdict is claimed, and nothing in this block's Status
changes.

- **Bounded re-probe (read-only `nvidia-smi`, no CUDA, no mutex, zero GPU time):**
  five samples 45 s apart from **2026-09-22T11:13:00Z to 11:16:10Z**, never clean.
  Two DISTINCT foreign instances: pid **13720**
  (`C:\Development\AI\colibri\pf06-wt\c\deepseek_v4.exe`) resident and active
  11:13:00 -> 11:14:40 at **11,893-11,919 MiB**, 36-67% utilization, then a handoff
  to pid **8676**
  (`C:\Development\AI\colibri\deepseek-v4-baseline\c\deepseek_v4.exe`) at
  11:15:25 (**9,171 MiB**, 58%) still active and climbing at 11:16:10
  (**18,719 MiB**, 84%). A PID handoff with rising VRAM is an active workload
  cycle, not the 30 s teardown grace - so this is the documented poison condition,
  and it outlasted the bounded window.
- **Not started, deliberately:** the session was not launched on a busy card,
  `-AllowBusy` was not passed, the mutex was not acquired, and no CUDA process was
  started by this attempt.
- **Fail-fast hazard, code-verified 2026-09-22T11:16:35Z (not a blocker - a
  hardening item).** Both arm drivers acquire the mutex with `WaitOne(3600 s)`
  BEFORE the busy probe (`run-gplr53-a3-arm.ps1:103` then `:115`; the A4 driver
  has the same order), which is also the recorded T-GPLR-52 house pattern. With no
  other project taking this mutex, attempting the documented command on a busy
  card would therefore NOT refuse in seconds - it would block for up to an hour and
  only then exit 125 with `gpu_pre_run` recorded. That is why this attempt probed
  first instead of launching.
- **LANDED the same day (2026-09-22): the read-only pre-launch probe is now in the
  session launcher, and the drivers keep the recorded T-GPLR-52 order unchanged.**
  `tools/run-gplr53-session.ps1` probes once with the shared helper
  (`Get-GpuComputeApps` + `Format-GpuBusy`, reusing the `GPU_BUSY_PROBE_CMD` stub
  the helper already documents) BEFORE it clears a verdict or starts a driver, and
  refuses fast: any visible compute app -> print the list, exit **125**, start no
  driver, write no arm artifact, acquire NO mutex. `probe-unavailable` warns and
  proceeds (the drivers probe again under the mutex, mirroring
  `with-gpu-lock.ps1`'s UNVERIFIED), and a new `-PreflightOnly` switch reports the
  device state and exits without launching, which is how the clean path can be
  proven host-side. The probe deliberately sits AFTER the `-ValidateOnly` branch so
  the dry run keeps working on a busy card. Host-side proof, no GPU, no mutex
  (all observed 2026-09-22):

| case | result |
| --- | --- |
| one fake compute app (`GPU_BUSY_PROBE_CMD`) | **exit 125 in 286 ms**, no `launched pid`/`SESSION:` line, no arm artifact written |
| two fake apps | exit 125 in **267 ms**, list printed as `13720 deepseek_v4.exe; 19768 test-opt.exe` |
| clean stub + `-PreflightOnly` | exit 0 in 299 ms, `PREFLIGHT OK: clean`, nothing launched |
| `-ValidateOnly` on the busy live card | unchanged: dry run OK, exit 0, all 8 arms |
| pre-existing `gplr53-layout-ctest.*` artifacts | untouched across the refused run (same mtimes) |
| driver process after every case | 0 `llama-layer-streaming` processes |

  The refusal line states the contract explicitly (no driver, no mutex, no
  artifact), and the drivers' own busy gate is unchanged, so the fast path is an
  optimisation of the launcher and never a replacement for the in-hold probe.
- Everything host-side is ready and was re-verified in this attempt: the session
  dry run is OK for all 8 arms, and the binary pair the drivers would record is
  exe **FE3B0A108C1D9BC9** / llama.dll **8A9F3D83F24D0FEC**.
- **Card state afterwards (2026-09-22T11:21:32Z): CLEAN.** An independent
  `nvidia-smi --query-compute-apps` probe returned no rows and `memory.used` read
  **0 MiB at 1% utilization** - the foreign cycle and the workspace's
  `test-opt.exe` run both finished, so the P0 session is now immediately runnable
  with the one documented command. The blocked verdict below still stands for the
  11:13-11:17Z window: no arm ran in it.
- **Final gate probe at 2026-09-22T11:17:08Z: still busy, and not only foreign.**
  `nvidia-smi --query-compute-apps` listed TWO compute processes: the foreign pid
  **8676** (`deepseek-v4-baseline\c\deepseek_v4.exe`, 18,867 MiB, 74%) and
  **pid 19768 `build/llama-cuda/bin/Release/test-opt.exe`** - a CUDA process from
  THIS checkout, i.e. another thread in this workspace is mid-run. The attempt is
  stopped here rather than queued: the gate would refuse on a two-app probe
  anyway, and racing a concurrent run from our own tree is exactly the
  contention case AGENTS.md tells the wrapper to refuse. No arm ran; A3 remains
  NOT RUN and P0 remains blocked on the shared GPU.

#### A3 session, third attempt (2026-09-22, 11:41:11Z - 11:49:18Z): BOTH ARMS RAN on a clean card - the A2 stamp fix is CONFIRMED, but two tool-input defects fail both arms, so A3 has no mechanism verdict yet

One mutex hold, the documented command (`tools/run-gplr53-session.ps1 -Session a3`).
The card was clean for the whole run and both arms started:

| | 50k | 200k |
| --- | --- | --- |
| gpu_pre_run | clean | clean |
| arm_exit | 0 | 0 |
| PP pass walls | 20.13 / 18.20 / 18.22 s | 136.05 / 134.64 / 134.82 s |
| pass tok/s | 2484.0 / 2746.6 / 2744.7 | 1470.0 / 1485.5 / 1483.5 |
| `--check-telemetry` | exit 0 | exit 0 |
| check-gplr53-embed-chunks | FAIL (exit 1) | FAIL (exit 1) |
| analyze-gplr53-decay | NOT_EVALUABLE (exit 2) | NOT_EVALUABLE (exit 2) |
| arm verdict | FAIL | FAIL |

Binary pair: exe **FE3B0A108C1D9BC9**, llama.dll **8A9F3D83F24D0FEC**; arm_exit 0 on
both, so the failures are the two TOOL checks, not the run. No performance claim is
made anywhere below - the numbers quoted are trace contents.

**Finding 1 - the A2 join fix is confirmed, and a DIFFERENT pre-existing defect is
now isolated to 6 samples.** Drift histogram: current 50k/200k
`+0 x118, -14 x3, -13 x3` (124 paired, 0 unpaired). The `-1` stamp class that every
pre-A2 trace carries is GONE, which is exactly A2's claim: `gplr51-a3 -1 x85`,
`gplr52-a2-50k -1 x119`, `gplr52-a3-mmap -1 x81` -> `+0 x118` today. What remains is
a stale-generation stamp at a pass boundary: exactly six samples, layers **38 and
39**, at ubatch **196 / 294** (the LAST ubatch of PP passes 2 and 3) and **295** (the
FIRST TG step) - the same coordinates at both shapes (782/1173/1174 at 200k by the
same rule), which is why the drift histogram is byte-identical at 50k and 200k.
Copies-per-key is exactly 1 for all 240 keys, so the checker's keyed pairing is
sound and this is a stamp defect, not a mispairing. It is **not** an A2 regression:
`gplr52-a2-50k` carries the same six coordinates at `-14` (layer 38) and `-15`
(layer 39), i.e. the magnitude fell by exactly one when the `-1` offset was removed.
One trace from today passes cleanly (`gplr54-a3-run1-u512-on: CHK-join ok`), so the
class is boundary- and shape-specific rather than universal. CHK-join was therefore
right to fail the arm at run time, and the arm could not PASS until this class was
attributed - it needed a read of the emitters at the pass boundary and the PP->TG
re-stage.

**SUPERSEDED later the same day by the classification section at the end of this
block: the class is a CHECKER KEYING artifact (recon_attr's `ubatch` is the live
flush stamp, not the copy's chunk), the checker is re-keyed to `(layer,
generation)`, and A3's three gates are now green on these traces. The wording above
is kept as the state AT RUN TIME, which is what the arms' FAIL verdicts record.**

**Finding 2 - the decay discriminator cannot evaluate its own input.** `embed_chunk`
is emitted and populated (294 = 98 x 3 chunks at 50k; 1173 = 391 x 3 at 200k) with
`setup_us` / `compute_us` / `span_us`, but it carries **no pass index**. The per-pass
boundaries that DO exist in the trace are `host_ledger` x3 (one per pass, carrying
the per-pass `embed_us`) and `pp_pass` x2 only - the last pass has no `pp_pass`, a
second small telemetry gap. `tools/analyze-gplr53-decay.py` buckets per-event by a
pass field, so it refuses with exit 2 "fewer than two passes with per-chunk
timing", which is NEVER a pass and is reported as such. Two small, host-testable
repairs exist (emit the pass index on `embed_chunk`, or bucket by the `host_ledger`
boundary); the rule must then be RE-RUN, because today's traces cannot yield a rule
verdict. A3 therefore has no mechanism verdict, and F1/H1/H3 remain undiscriminated
by the pre-registered rule.

**SUPERSEDED later the same day** by the section below: the loader was repaired
(bucketing by chunk-sequence reset), and the pre-registered rule then produced
CHUNK0_SPIKE from these very traces with no second GPU hold. This paragraph is kept
as the state AT RUN TIME, which is what the arms' FAIL verdicts record.

**Finding 3 - raw-trace corroboration of the pass-1 shape (telemetry only).**
The per-pass `embed_us` from `host_ledger` is 1,937.4 / 65.7 / 63.9 ms at 50k and
2,042.8 / 256.6 / 257.3 ms at 200k, and the per-chunk stream localizes essentially
all of the pass-1 excess in **chunk 0**: 1873.3 of 1936.2 ms (96.8%) at 50k and
1786.2 of 2037.4 ms (87.7%) at 200k, with chunk 0 the ONLY chunk above 10 ms in any
pass at either shape. The excess is thus a single-chunk, ~1.8 s, SIZE-INSENSITIVE
event (1.87 s at 50k vs 1.79 s at 200k), i.e. the chunk-0-spike shape rather than a
decaying front-loaded spread - consistent with a fixed-size completion such as F5's
18.5 GB registered span, not with a size-proportional cost. Stated as what it is:
telemetry-only corroboration of the pre-registered F1 shape, not a rule verdict,
and no speed claim follows.

**A3 status (as of the run):** both arms executed and both FAILed on the two
tool-input defects above; A4 was NOT run (its contract makes it conditional on a
clean second hold, and it was out of that invocation's scope). BOTH halves were run
later the same day - see the A3 session PASS (2/2) and the A4 session PASS (6/6)
sections at the end of this block.

#### A3 mechanism verdict (2026-09-22, decided from the traces already recorded - no new GPU hold): CHUNK0_SPIKE at both shapes (pre-registered mapping -> H3)

The blocker was a defect in the rule's own loader, and fixing it was enough: the
verdict is now produced from the same two traces the arms wrote, with no second
mutex hold.

**What was actually wrong.** `tools/analyze-gplr53-decay.py` bucketed passes by a
per-event `phase` field. In this format `phase` means **PP-vs-TG**, not the pass
index (values 0/1 on the per-op events, `'pp'`/`'tg'` on `phase_summary`), and
`embed_chunk` does not carry it at all - so every chunk landed in pass 0 and the
rule refused its own input on BOTH arms. It was never missing data: the per-chunk
spans and the per-pass boundaries were both in the trace. Bucketing is now by
**chunk-sequence reset** (a pass boundary is where the chunk index stops
increasing), with ambiguity REFUSED rather than guessed: blocks must be exactly
`0..n-1`, all blocks must be the same length, and a `host_ledger` boundary count of
>= 2 must agree with the block count. `pp_pass` is advisory only - it fires at pass
TRANSITIONS and is measured at n-1 (2 markers for 3 passes). The recorded traces
bucket cleanly: three equal blocks each (98 x 3 at 50k, 391 x 3 at 200k), monotone
`t_us`, `host_ledger` 3 = block count 3. **No emitter change is required for the rule
to be evaluable**, so no arm was re-run. Fixtures were converted to the recorded
shape (no `phase`/pass field) - the old fixtures carried a `phase` field, which is
exactly why the loader defect survived them - and the bucketing now has its own
controls: truncated pass, gapped chunk ids and boundary-count mismatch all REFUSE
(exit 2), plus a 3-pass positive control. `--selftest` 10/10.

**The verdict, with the per-pass numbers it was decided on:**

| | 50k | 200k |
| --- | --- | --- |
| pass totals (ms) | 1937.01 / 63.40 / 63.41 | 2040.50 / 254.62 / 255.26 |
| warm median (ms) | 0.638 | 0.641 |
| pass-1 chunk-0 span (ms) | 1873.331 | 1786.308 |
| excess at chunk 0 (ms) | 1872.39 | 1785.27 |
| largest excess EXCLUDING chunk 0 (ms) | 0.114 | 0.263 |
| chunk-0 share of positive excess | 0.9991 | 0.9965 |
| first-decile share | 0.9991 | 0.9967 |
| rho (Spearman) | +0.315 | +0.110 |
| chunks per pass | 98 | 391 |
| **rule verdict (exit 0)** | **CHUNK0_SPIKE** | **CHUNK0_SPIKE** |

CHUNK0_SPIKE is the rule's second branch and is evaluated before the rho branches:
chunk 0 carries >= 50% of the positive excess AND every later chunk sits inside 2x
the warm band. Both hold overwhelmingly - no chunk other than chunk 0 exceeds
0.263 ms of excess at 200k, or 0.114 ms at 50k. Honest limit: rho is POSITIVE
(+0.315 / +0.110), so on the 200k trace the FLAT branch (|rho| < 0.3) would have
fired had CHUNK0_SPIKE not taken precedence; the verdict therefore rests on the
chunk-0 concentration and the warm tail, not on the trend statistic. The
pre-registered mapping for CHUNK0_SPIKE is H3 (submission-side); H1-vs-H3 remains
A4's modality question and is never this rule's to answer.

**Independent corroboration, telemetry only.** The pass-1 excess is 1.872 s at 50k
and 1.785 s at 200k - essentially FLAT while the prompt grew 4x, and while the warm
embed per pass grew with it (63.4 -> 254.6 ms, i.e. the per-chunk registration cost
does scale). A cost that scales with touched pages would have grown with the prompt;
a fixed one-off completion does not. That favours H3 over H1 without deciding it. No
speed claim follows from any of this: these are trace contents, not measurements of
an improvement.

**Gate re-derivation on the recorded traces** (a re-derivation, NOT a new arm run -
the arms' recorded verdicts stay FAIL): `--check-telemetry` exit 0 on both; the decay
rule exit 0 on both (a verdict was produced); `check-gplr53-embed-chunks --join`
exit 1 on both. **The only remaining arm blocker is CHK-join**, which is the
defect below.

#### The layer-38/39 "boundary stamp" CLASSIFIED: it is a CHECKER KEYING artifact, not a stale stamp - checker re-keyed, A3's three gates re-derived green, emitter caveat recorded rather than patched

Host-only, no GPU, no mutex. The measured facts that preceded the classification
(kept because they are what made it decidable):

- 6 samples per trace, layers **38 and 39**, at ubatch **196 / 294** (the LAST ubatch
  of PP passes 2 and 3) and **295** (the FIRST TG step); identical coordinates at
  both shapes (782/1173/1174 at 200k), which is why the drift histogram was
  byte-identical at 50k and 200k.
- Drift was one pass-stride: `-14 x3, -13 x3` today, `-15 x3, -14 x3` on
  `gplr52-a2-50k`, and equal to each slot's per-pass generation stride (13 for layer
  38 in slot 2, 14 for layer 39 in slot 0).
- Copies-per-key was exactly 1 for all 240 keys, so a mispairing across copies was
  ruled out; and one trace from today passed the join cleanly
  (`gplr54-a3-run1-u512-on`), so the class was boundary- and shape-specific.

**The defect is in the EMITTER'S FIELD SEMANTICS, and the checker was asserting
something the emitter never promised.** `recon_flush_all()` writes the sample's
`ubatch` from the LIVE `n_ubatches` at flush time
(`third_party/llama.cpp/src/llama-layer-streaming.cpp:1323-1325`), while `layer` and
`generation` come from the ring entry that describes the copy actually measured
(`e.il` / `e.gen`, set in `recon_begin_bank`, `:1291-1292`). So a sample's
`(layer, generation)` identifies the copy while its `ubatch` names the chunk current
when it was flushed. Keyed by `(layer, ubatch)` - the old CHK-join - the samples
flushed across a pass boundary therefore appeared to carry a one-pass-stale
generation, when in fact their generation was correct all along.

The decisive evidence, all on the recorded traces: matching by `(layer, generation)`
gives **124/124 matched, 0 unmatched** on both A3 traces, and exactly 6 of them were
flushed after the live counter advanced past their copy's chunk - with **zero**
samples naming a flush earlier than their copy, and `copy.t_us <= sample.t_us`
holding for all 124. Against the pre-A2 traces the same key fails **100%**: 0/89
(`gplr51-a3`), 0/125 (`gplr52-a2-50k`), 0/125 (`gplr52-a2-200k`), 0/85
(`gplr52-a3-mmap`) - the `-1` stamp points at a generation no copy of that layer
ever carried, which is exactly the defect the rider exists to catch.

**Fix (checker-side, host-only, landed).** `CHK-join` now keys on
`(layer, generation)` - the identity the emitter captures together - and keeps two
ordering conditions so it is not vacuous: a sample may not name a chunk BEFORE the
copy it identifies, and when both carry `t_us` the copy must precede the sample. The
count of samples flushed after their copy is reported in the check's detail line
rather than silently ignored. `--selftest` is **20/20**: the `-1` controls still
fail (2 sample(s), 1 sample(s)), a new `-13` control that identifies no copy fails,
a new flush-earlier-than-copy control fails, the missing-ubatch case is still
refused at exit 2, and one control was RE-SPECIFIED: `join-tg-burst-bad` (-14, used
to expect a failure) lands exactly on that layer's pass-1 generation, so it is
byte-for-byte the flush-later shape the corrected keying must accept; it is now
`join-tg-burst-stale-legal` expecting PASS, with the failure path covered by the
`-1`/`-13` controls and by the four pre-A2 traces above.

**A3's three gates, re-derived on the traces already on disk** (a re-derivation, NOT
a new arm run - the arms' recorded verdicts stay FAIL and no GPU time was spent):
`--check-telemetry` exit 0; `check-gplr53-embed-chunks --join` **PASS** (CHK-join
ok); the decay rule exit 0 = CHUNK0_SPIKE. The recorded arm verdicts remain the
run-time record of what the tools said then.

**Does this invalidate the pass-boundary attribution the verdict rests on? NO.**
The bucketing uses the `embed_chunk` stream only (chunk-sequence resets, validated
by equal blocks / `0..n-1` ids / `host_ledger` count agreement), while this defect
lives in `recon_attr`/`copy_submit` - disjoint event streams. The 6 samples were
never part of the bucketing, and no `embed_chunk` field or pass boundary is
affected. The pre-registered rule and its thresholds were NOT touched after the
fact; only the loader that feeds it (and this checker) changed.

**Recorded, deliberately NOT patched (needs a GPU run to validate, which this task
forbade).** The emitter-side improvement is to stamp each sample with the ubatch of
the copy it describes, by storing the ubatch in the ring entry at
`recon_begin_bank` and writing `e.ubatch` instead of the live `n_ubatches` at
`llama-layer-streaming.cpp:1323-1325`. It is telemetry-only, but it cannot be shown
to work host-side: no recorded trace contains the new field, so validating it would
require producing a fresh trace, i.e. another mutex hold. The checker fix makes the
join sound without it, so it stays a recorded option rather than a landed change.

#### P0 session, A3 half (2026-09-22, 12:04-12:12Z): SESSION VERDICT PASS (2/2) - the run-time FAILs are replaced by real arm verdicts, and CHUNK0_SPIKE REPRODUCES on a second independent run at both shapes

One mutex hold, the documented command, card verified clean before and after
(`-PreflightOnly` returned clean in 0.34 s before launch). Same binary pair as the
first run - exe **FE3B0A108C1D9BC9**, llama.dll **8A9F3D83F24D0FEC** - so this is a
genuine repeat, not a different build.

| per-arm gate | 50k (dur 67.6 s) | 200k (dur 484.9 s) |
| --- | --- | --- |
| `--check-telemetry` | exit 0 | exit 0 |
| `check-gplr53-embed-chunks --join` | exit 0, `CHK-join ok` | exit 0, `CHK-join ok` |
| decay rule | exit 0, **CHUNK0_SPIKE** | exit 0, **CHUNK0_SPIKE** |
| `chunk0_share` / first-decile share | 0.9997 / 0.9997 | 0.9987 / 0.9987 |
| rho (Spearman) | +0.0159 | +0.2924 |
| excess total (ms) | 2001.91 | 1871.71 |
| PP pass walls (s) | 20.25 / 18.12 / 18.16 | 135.84 / 134.67 / 135.54 |
| gpu_pre_run | clean | clean |
| arm verdict | **PASS** | **PASS** |

**Reproducibility.** CHUNK0_SPIKE is now the verdict of FOUR independent arm-runs
(both shapes, twice), and the excess stays size-insensitive in the repeat as well:
2001.91 ms at 50k vs 1871.71 ms at 200k, against a warm per-pass embed that again
grows with the prompt. One honest nuance the repeat exposes: the DECIDING
statistics are stable (chunk-0 share and first-decile share ~0.999 in all four
runs), while `rho` is not (+0.315 and +0.0159 at 50k; +0.110 and +0.2924 at 200k),
which is exactly the limit already on the record - the verdict rests on the chunk-0
concentration and the warm tail, never on the trend statistic.

> SUPERSEDED (see "A4 half, second attempt" below): the A4 arms were subsequently RUN on a clean card in one
> mutex hold (2026-09-22, 12:28:01-12:29:31Z, 6/6 PASS) and H1-vs-H3 now has modality evidence. The blocker
> and the UNDISTINGUISHED statement below stand as the truth for the 12:12-12:17Z window only, and the arms
> named as "never started" here are exactly the arms that ran in the second attempt.

#### P0 session, A4 half: NOT RUN - blocked by a foreign compute process, refused in 394 ms by the new preflight, and H1-vs-H3 remains UNDISTINGUISHED

- **Blocker, named not inferred.** `nvidia-smi --query-compute-apps` showed pid
  **21092** (`C:\Development\AI\colibri\pf06-wt\c\deepseek_v4.exe`) resident and
  ACTIVE across the whole bounded re-probe: 12:12:20Z 18,705 MiB / 91%, 12:13:10Z
  18,695 MiB / 83%, 12:14:00Z 18,337 MiB / 51%, 12:14:50Z 18,391 MiB / 66%, and
  still resident at 12:15:25Z. Four samples over ~3 minutes, never clean, so the
  attempt stopped instead of waiting. `-AllowBusy` was NOT passed and no CUDA
  process was started by this half.
- **The new preflight did its job in production:** `run-gplr53-session.ps1
  -Session a4 -PreflightOnly` returned **exit 125 in 394 ms**, naming the PID. Under
  the pre-launch-probe section's recorded hazard, the same attempt would have sat
  on `WaitOne(3600 s)` for up to an hour before refusing.
- **Arms that did NOT run** (no artifact of any kind, so nothing is inferred from
  them): `gplr53-a4-pw0`, `-pw1`, `-pw2`, `-pw3`, `-pin`, `-r2` - all "never
  started", no `.started`, no verdict, no `.done`.
- **H1 vs H3: UNDISTINGUISHED by this session.** A3's rule maps CHUNK0_SPIKE to H3
  (submission-side), and the size-insensitivity is *consistent* with a fixed one-off
  completion rather than a page-proportional cost - but A3 cannot separate H1 from
  H3 by construction, because that separation is A4's modality question (mode 1 =
  CPU read-touch, H1's page-side candidate; mode 2 = dummy GPU launch + sync, H3's
  submission-side candidate; mode 3 = both). With all six A4 arms unrun, the
  question stays open and no claim either way is made.
#### P0 session, A4 half, second attempt (2026-09-22, 12:28:01Z - 12:29:31Z): SESSION VERDICT PASS (6/6) - the pre-warm arms ran, and the modality evidence points at the HOST PAGE/RESIDENCY side (H1), not at per-submission overhead (H3)

> **WITHDRAWN by the third attempt below (2026-09-22, 12:37:33Z - 12:39:42Z).** The H1 reading this
> section draws is FALSIFIED: a duration-matched control that touches ZERO pages removes 93.2% of the
> same excess. Withdrawn claims, each measured on the run recorded here: "H3 as per-submission overhead
> is REFUTED" (the submission side is not merely refuted, it is also not the payer - the payer is
> elapsed time), "H1 (page/residency side) is supported and dominates" (it does not: ~93% of the excess
> needs no page touch at all), and "mode 1 ... removes 99.3%" as evidence of page residency (mode 1's
> removal is explained by its own ~1.7 s DURATION, which the delay control matches without touching
> anything). What survives from this section: the raw per-arm numbers, the size-insensitivity, the
> pass-2/3 invariance, the pinned-source observation, the "relocation, not a saving" framing, and the
> fact that both pre-registered branches fired. Do NOT cite the H1 conclusion without the third-attempt
> section.

The card was clean at 12:27:53Z (0 MiB, 1%), so the half ran in ONE mutex hold. All six arms PASS all
three gates (`--check-telemetry` exit 0, `check-gplr53-embed-chunks --join` exit 0, decay rule exit 0 =
CHUNK0_SPIKE) on the same binary pair as A3 (`exe FE3B0A108C1D9BC9` / `llama.dll 8A9F3D83F24D0FEC`),
`gpu_pre_run` and `gpu_post_run` clean, `resolved_copy_source` matching what each arm requested.

| arm | mode | pre-warm pre-pay (ms) | pass-1 chunk-0 excess (ms) | excess total (ms) | pass 1 / 2 / 3 wall (s) | warm median (ms) | arm dur (s) |
|---|---|---|---|---|---|---|---|
| `pw0` | 0 off (control) | - | 1881.94 | 1882.17 | 3.65 / 1.57 / 1.58 | 0.641 | 14.7 |
| `r2` | 0 off, `--pp-repeat 2` | - | 1831.38 | 1831.29 | 3.63 / 1.61 / - | 0.663 | 12.8 |
| `pw1` | 1 CPU read-touch | 1673.5 | 13.52 | 13.74 | 1.79 / 1.57 / 1.57 | 0.649 | 15.3 |
| `pw2` | 2 dummy launch + sync | 1832.1 | 1.18 | 1.18 | 1.77 / 1.58 / 1.58 | 0.651 | 15.3 |
| `pw3` | 3 both | 1673.7 (+18.2) | 1.11 | 1.23 | 1.77 / 1.58 / 1.58 | 0.644 | 15.1 |
| `pin` | 0, PINNED source | - | 1.28 | 1.34 | 5.44 / 1.69 / 1.77 | 0.632 | 17.2 |

- **Measured: the one-off is a HOST-side residence/commit of the 17.23 GiB package source span, and it is
  removable without a single weight byte being copied to the GPU.** Mode 1 is a read-only CPU page walk
  (`pp-prewarm(1): CPU read-touch 17.068 GiB / 4474368 pages -> sink 166 in 1673.5 ms`), i.e. 0.374 us per
  4 KiB page (~10.2 GB/s). It costs 1673.5 ms and the pass-1 chunk-0 excess collapses 1881.94 -> 13.52 ms
  (-99.3%), for a pass-1 wall of 3.65 s -> 1.79 s. A pure CPU action with no GPU submission cannot remove a
  cost that lives on the submission side, so **H3 as "per-submission overhead" is REFUTED**.
- **Measured: the excess does not scale with submissions, bytes, or shape.** 1882.17 ms (4k, 8 chunks/ubatches)
  vs 1881.94 ms (50k) vs 1871.71 ms (200k) on the earlier A3 session, and 1831.29 ms on the 2-pass `r2` arm -
  one fixed quantity. Passes 2 and 3 are IDENTICAL across every arm (1.57-1.58 s, warm median 0.632-0.663 ms),
  so nothing here is per-copy or per-byte, and no mode helps or hurts the warm steady state.
- **Measured: a pinned source removes it from the embed bracket with NO pre-warm at all** (`pin` excess
  1.34 ms, mode 0). It does not remove the COST - `pin`'s pass-1 wall is 5.44 s, i.e. 1.79 s WORSE than `pw0`'s
  - so the cost is relocated into pass-1 setup, which is where a real pinned mirror has to be populated.
- **Measured: the two modalities compete for ONE one-off, they do not attack two.** Mode 1 alone costs
  1673.5 ms; mode 2 alone costs 1832.1 ms (a 64 B H2D + event sync is never 1.8 s of its own work); in mode 3
  the walk costs 1673.7 ms and the dummy submission that follows costs only **18.2 ms** - it finds the cost
  already paid. An independent GPU-side init would still have charged ~1.8 s there. Registration itself is
  NOT the cost: `mmap-registered staging: 18500517888 B (17.23 GiB) pinned in 1.5-1.6 ms`, identical in every
  mmap arm - `cudaHostRegister` is bookkeeping, and the residency work happens at first touch/first DMA.
- **Inferred (not directly traced): the first H2D DMA is the TRIGGER for that host-side residency commit.**
  That is the only mechanism consistent with both mode 1 (CPU-only) and mode 2 (submission-only) removing the
  same one-off and with mode 3 not compounding. The arms expose no driver-internal page-commit trace, so the
  trigger half is a labelled inference, not a measurement.
- **H1 vs H3, stated as the evidence supports it: H1 (page/residency side) is supported and dominates.
  H3-alone is refuted. The arms CANNOT cleanly separate "host page residency" from "host page residency
  triggered by the first submission"** - precisely because `pw2`'s dummy submission is not a clean H3 probe:
  it pays the same 1.7-1.8 s rather than isolating submission latency.
- **Against A4's OWN pre-registered rule: branch (i) FIRES, and branch (ii) fires too.** The contract says
  "(i) removes >= 70% -> page-side completion (H1); (ii) removes it -> submission-side (H3); neither -> the
  cost attaches to the first execution of the embed graph set (escalate; do not tune)". Mode 1 removes 99.3%
  (>= 70%, so branch (i) = H1 fires) AND mode 2 removes 99.9% (so branch (ii) = H3 fires). The disjunction
  was written for mutually exclusive modalities; the measurement shows they are not, so **the rule does not
  discriminate here** and this session does not pick a branch. The escalation branch does NOT apply: the
  cost did not survive both modalities.
- **This also DISAGREES with A3's pre-registered rule mapping, and that is recorded rather than reconciled.**
  A3's rule maps CHUNK0_SPIKE -> H3 (submission-side); A4's branch (i) maps its removal -> H1. Both mappings
  were pre-registered; the extra discriminating facts A4 measured (a CPU-only page walk removes 99.3%; mode 3
  shows the second modality adds only 18.2 ms, so it is one cost not two; registration is 1.6 ms not 1.8 s;
  a pinned source removes it with no pre-warm) point at H1. Neither rule, nor any threshold, nor the accepted
  A3 verdicts were changed: the disagreement is escalated to the Planner as the task's remaining open item.
- **No performance claim, and specifically: pre-warm is a RELOCATION, not a saving.** The pre-pay (1673-1832 ms)
  is the same magnitude as the excess it removes (1831-1882 ms), and arm wall times are flat to slightly worse
  (14.7 s `pw0` -> 15.3 / 15.3 / 15.1 s). The pass-1 wall does drop by ~1.8 s, but the cost is still paid, once,
  in the same arm. Nothing above is a speedup measurement and none is claimed.
- **The evidence hazard is proven closed in production, not just in the lab.** The launcher's archive step
  fired on this run before anything was invalidated
  (`archived predecessor evidence -> build/llama-cuda/runs/20260922T122800Z/`) and it captured the FAIL
  artifacts of the earlier shape-bug attempt (six `FAIL 1` verdicts, 12:25:32-12:25:50Z) side by side with the
  new PASS run's files, with a `manifest.json` naming each file's source, size and truncated SHA-256. The
  accepted A3 evidence was untouched by this session (its traces and verdict files keep their 14:04 / 14:10
  mtimes and their `PASS 0` contents).
- **Mode-0 vs the off default: now verified at run time as well as by host diff.** The off arms (`pw0`, `pin`)
  carry no `pp-prewarm(` line at all and `prewarm_engaged: false`, while modes 1-3 each carry their own line
  and `prewarm_engaged: true` - the driver's own "mode set but no pre-warm line" assertion passed for both
  classes, closing the run-time half of the assertion left open in the durability section above.

#### A4 half, third attempt - THE FALSIFIER (2026-09-22, 12:37:33Z - 12:39:42Z): SESSION VERDICT PASS (8/8), and the no-touch delay CLEARS 93.2% of the excess, so the page-residency mechanism claim COLLAPSES

The audit found the previous reading unfalsifiable: no arm separated the mode-1 CPU page walk from any
~1.7 s of elapsed time before pass 0, so "the walk faults in the source pages and pays the one-off" could
not be told apart from "the one-off is a lazy completion that any ~1.7 s wait absorbs". Two control modes
were added (engine `pp_prewarm` is now a bitfield 0..15; 4 = duration-matched no-touch delay, 8 =
duration-matched wrong-region touch on a private 64 MiB scratch), default-off, and the whole A4 set was
re-run in ONE mutex hold on a clean card (preflight 12:37:19Z, `gpu_post_run` clean on every arm).

| arm | mode | pre-pay (ms) | embed excess (ms) | removal vs control | pass-1 wall (s) | wall removal | pass 2 / 3 wall (s) |
|---|---|---|---|---|---|---|---|
| `pw0` | 0 off (control) | - | 1866.72 | - | 3.62 | - | 1.57 / 1.57 |
| `pw1` | 1 CPU read-touch | 1691.1 | 13.69 | **99.3%** | 1.78 | 50.8% | 1.57 / 1.57 |
| `pw2` | 2 dummy launch + sync | 1845.2 | 1.42 | **99.9%** | 1.77 | 51.1% | 1.57 / 1.57 |
| `pw3` | 3 both | 1690.3 | 1.51 | **99.9%** | 1.77 | 51.1% | 1.57 / 1.57 |
| `delay` | **4 no-touch delay** | 1710.7 | 126.85 | **93.2%** | 1.90 | 47.5% | 1.58 / 1.58 |
| `wrongtouch` | **8 wrong-region touch** | 1706.1 | 125.57 | **93.3%** | 1.89 | 47.8% | 1.57 / 1.57 |
| `pin` | 0, pinned source | - | 1.71 | 99.9% | 5.34 | -47.5% | 1.74 / 1.73 |
| `r2` | 0, `--pp-repeat 2` | - | 1834.68 | 1.7% | 3.63 | -0.3% | 1.60 / - |

- **THE FALSIFIER FIRED, and the mechanism claim collapses.** `pp-prewarm(4): no-touch delay 1700 ms / 0
  pages touched in 1710.6 ms` removes **93.2%** of the excess (1866.72 -> 126.85 ms). A control that
  touches NOTHING cannot pay a page-residency cost, so the excess is not "the first read of the 17.23 GiB
  source span": it is something that completes on its own given ~1.7 s. The wrong-region touch agrees to
  within 1.3 ms (125.57 ms, 93.3%, `64 MiB x 19970 loop(s) / 327188480 page touches` at 1700.0 ms), so page
  touching in ANY region buys nothing either. Mode 1's removal is therefore explained by its own duration
  (1691.1 ms, duration-matched to the delay's 1710.6 ms within 1.1%), not by the pages it walked.
- **What actually survives, stated as the small residue it is:** after the time-only effect, the source
  walk still leaves 13.69 ms where the delay leaves 126.85 ms - a **~113 ms** page-side residue, i.e. ~6% of
  the original 1.87 s and two orders below the "18.5 GB registered span completion" this block was opened
  to explain. The pinned source remains the only arm with a clean pass 1 and NO wait at all (1.71 ms), and
  it still pays ~1.7 s elsewhere (pass-1 wall 5.34 s, -47.5% against the control), so the pending thing is
  tied to the mmap-registered setup rather than to time as such.
- **The payer is neither the source read nor the submission, so the pre-registered ESCALATE branch applies.**
  H1 (page side, as this block's mode 1 defined it) is falsified; H3 (submission side) is not the payer
  either - the delay makes no submission at all. The hypothesis set as written is INCOMPLETE, and per the
  contract's own disposition that means ESCALATE, do not tune. **Pre-warm is not adopted, no mode is
  recommended, and no performance claim follows**: pre-pay still equals the excess it removes and arm walls
  are flat (pw0 14.9 s vs delay/pw1/pw2 15.2-15.3 s), so every mode remains a relocation at best.
- **A defect in the pre-registered threshold, recorded and NOT edited.** The contract's comparison quantity
  is the pass-1 WALL with the excess as the corroborating term, and it sets the branch at ">= 70%". On this
  shape the one-off is 1.84 s of a 3.62 s wall, so the ceiling for ANY wall-based removal is ~51%: `pw1`,
  `pw2` and `pw3` all land at 50.8-51.1% (i.e. branch (i) could never fire on the quantity it names), while
  the excess-based reading is 99.3%. The second attempt's "branch (i) fires (H1)" therefore rested on the
  CORROBORATING term, and the honest statement is that the wall-based threshold was unsatisfiable. The
  pre-registered text and its thresholds are untouched; this is a recorded defect for the Planner.
- **A3's rule mapping is left standing as its own output.** CHUNK0_SPIKE -> H3 was never a measurement, and
  the third attempt does not vindicate it either: the one-off is a time-gated completion, a third thing.
  Both pre-registered mappings (A3's -> H3, A4's branch (i) -> H1) are now unsupported by the arms, and the
  disagreement recorded in the status line resolves to "the hypothesis set was incomplete", not to H1.
- **n basis, stated explicitly (the audit's second provenance gap).** Every A4 mode has now run TWICE - the
  second attempt (12:28Z) and this one (12:37Z) - but on DIFFERENT binaries (`llama.dll 8A9F3D83F24D0FEC`
  then `5DE4FCD26921E6E8`; the mode-4/8 branches and the `gpu_post_run` field were added in between), so this
  is a same-session-shape reproduction across builds, NOT a same-binary repeat. The off control reproduces
  within 0.8% across the two (1882.17 vs 1866.72 ms; walls 3.65 vs 3.62 s) and modes 1-3 within 1.6%, which
  is what licenses reading the two runs together; each individual mode still has n=2 total and no repeat
  inside a single hold. A dose-response sweep would be the arm that raises this.
- **Provenance fixes landed and verified in the artifacts.** `gpu_post_run` is now persisted per arm in BOTH
  GPLR-53 drivers (parity with the gplr51/gplr52 pattern, which the GPLR-53 drivers had dropped by printing
  it only); all 8 verdicts of this run carry `gpu_post_run: clean`, so the "card clean after" claim the
  record makes is now backed by evidence rather than by a later manual probe. Verified: 8/8 verdict files
  contain the field and every arm's three gates are green (`--check-telemetry` exit 0, `--join` exit 0,
  decay exit 0 = CHUNK0_SPIKE; `r2` route parity clean).
- **New modes host-validated before the GPU was touched** (no mutex, no CUDA): `--pp-prewarm 16` refused
  (exit 2, message lists 1/2/4/8); modes 4 and 8 accepted and reaching the same refusal table as 1-3
  (`--residency off` -> exit 2); off-state BYTE-IDENTICAL again (flag absent vs `--pp-prewarm 0`); the
  driver dry run prints 8 arms with exactly `--pp-prewarm 1/2/3/4/8` and neither flag on the three off arms.
  Modes 4/8's RUNTIME behavior is first exercised by this run, exactly as modes 1-3's was by the second.
- **Recorded gap, not fixed here:** the session launcher does not persist its child driver's stdout, so the
  driver's own computed `FALSIFIER ...` line is not in `gplr53-a4-session4.out`; the percentages above are
  recomputed from the per-arm verdicts, which carry the raw `embed_excess_total_ms` and `pass1_wall_s`.
- **Durability fired again in production:** the third attempt archived the second attempt's 8-arm evidence
  into `build/llama-cuda/runs/20260922T123732Z/` (30 manifest entries) before overwriting those paths, so
  both A4 runs and the original shape-bug FAILs are all still on disk side by side.

#### Dose curve (2026-09-22, 12:44:13Z - 12:47:35Z): PASS 13/13 - absorption is a ~1:1 ramp in the delay, the wall is CONSERVED, and no dose shortens anything

The falsifier raised the follow-up question the walls implied: if the one-off is paid by elapsed time, can the
pre-pay be made SHORT (a net win) instead of merely relocating it? The mode-4 no-touch delay now takes its
duration from a new opt-in flag `--pp-prewarm-delay-ms N` (0..60000; default 1700 ms = the compiled control
duration, so an un-flagged run is unchanged; refused pre-model-load unless `--pp-prewarm` has bit 4 set), and
five dose arms (0/200/600/1200/1700 ms) ran in one mutex hold against the same-session off control.

| arm | delay ms | pre-pay ms | excess ms | absorbed | pass-1 wall s | total arm s | pre-pay + pass-1 s |
|---|---|---|---|---|---|---|---|
| `pw0` (off control) | 0 | 0.0 | 1905.95 | - | 3.67 | 14.5 | 3.67 |
| `d000` (mode 4, sleeps 0) | 0 | 0.0 | 1834.11 | 3.8% | 3.60 | 14.8 | 3.60 |
| `d200` | 200 | 208.4 | 1622.54 | 14.9% | 3.39 | 14.7 | 3.60 |
| `d600` | 600 | 606.6 | 1054.58 | 44.7% | 2.82 | 14.6 | 3.43 |
| `d1200` | 1200 | 1206.9 | 634.56 | 66.7% | 2.40 | 15.2 | 3.61 |
| `d1700` | 1700 | 1713.0 | 131.35 | 93.1% | 1.90 | 14.8 | 3.61 |
| `delay` (flag ABSENT) | compiled | 1701.5 | 139.04 | 92.7% | 1.91 | 15.2 | 3.61 |
| `pw1` source walk | - | 1693.8 | 14.18 | 99.3% | 1.78 | 15.2 | 3.47 |
| `pw2` dummy launch | - | 1856.0 | 1.10 | 99.9% | 1.76 | 14.7 | 3.62 |
| `pw3` both | - | 1689.7 | 1.29 | 99.9% | 1.77 | 15.1 | 3.46 |
| `wrongtouch` | - | 1706.8 | 154.51 | 91.9% | 1.93 | 14.8 | 3.64 |
| `pin` (no wait) | - | 0.0 | 1.37 | 99.9% | 5.39 | 17.3 | 5.39 |
| `r2` | - | 0.0 | 1833.56 | 3.8% | 3.70 | 13.7 | 3.70 |

- **Measured - absorption is a monotone ramp, NOT a step, and it is ~1:1 in the delay.** Absorbed excess
  vs delay: 200 ms -> 283 ms, 600 -> 851, 1200 -> 1271, 1700 -> 1775 (1.42, 1.42, 1.06, 1.04 ms absorbed
  per ms waited). There is no characteristic duration at which the one-off "finishes" and the curve goes
  flat; the residual simply tracks (one-off - delay), so full absorption would need ~1.8-2.0 s of pre-pay -
  i.e. about as much waiting as the one-off is worth.
- **Measured - the wall is CONSERVED, so no dose shortens anything.** pre-pay + pass-1 wall is 3.43-3.70 s at
  every dose (control 3.67, d200 3.60, d600 3.43, d1200 3.61, d1700 3.61), and the arm's own total wall is
  flat (14.5 control vs 14.6-15.2 across the doses, no trend). **Answer to the shortening question: the
  pre-pay can be made arbitrarily short, but only by leaving exactly that much excess behind - it is a 1:1
  purchase of pass-1 time, never a rebate.** There is therefore no dose at which pre-warm buys net wall
  time, at any length.
- **The curve also kills the residual hope that a SHORT dose is cheap:** the most favourable point, 200 ms,
  costs 0.21 s and absorbs 0.28 s - a 0.07 s difference, which is inside this metric's run-to-run noise
  floor (measured: `d000`, which sleeps nothing, differs from the off control by 3.8%, 1834.11 vs
  1905.95 ms; `r2` differs by 3.8% too). Nothing on this curve exceeds that floor, which is itself part of
  the result.
- **Measured - modality adds only a small term on top of duration, and the curve explains the earlier
  anomaly.** `pw2`'s cleaner result than `d1700` (1.10 vs 131.35 ms residual) is accounted for by its LONGER
  pre-pay (1856.0 vs 1713.0 ms = +143 ms), which the 1:1 ramp predicts as ~+143 ms of absorption - observed
  +130 ms. So mode 2's advantage over a sleep is duration, not submission. The source walk retains a real
  extra: 14.18 ms residual against `d1700`'s 131.35 ms is a ~117 ms page-side term (run 4's pair gave
  ~113 ms), reproducible across two runs - that is the entire surviving page-side signal.
- **`pin` remains the only configuration that avoids the wait, and it is worse overall.** Residual 1.37 ms
  with zero pre-pay, but its pass-1 wall is 5.39 s and its total wall 17.3 s against the control's 14.5 s.
  No configuration on this curve removes the work; only its location changes.
- **Reading of the curve, stated as measurement and nothing more:** the pass-1 one-off behaves like an
  irreducible quantity of pending work that pass 1 pays serially unless something waits it out first; the
  exchange rate is 1:1, so pre-warm cannot be a wall-time win. This CONFIRMS and sharpens the falsifier
  section's reading rather than superseding it - the withdrawn H1 section stays withdrawn and its history
  is untouched. **No performance claim is made from any of this**, and the pre-registered contract and its
  thresholds were not edited.
- **n basis for the curve, explicit:** each dose arm ran ONCE (one hold, one binary pair
  `exe 72C6DFE1C5F5FA6B` / `llama.dll 2D2EA6B41E0A22E9` at that moment - see the arms' own `.started`
  records for the exact commands); the off control has been run five times across four holds and the
  spread of its excess is 1866.72-1905.95 ms, which is the noise the curve is read against. The dose arms
  are dose-only: the flags actually sent were verified from the `.started` commands (`d000/d200/d600/d1200/
  d1700` carry `--pp-prewarm 4 --pp-prewarm-delay-ms N`; `delay` carries mode 4 with the flag ABSENT, which
  is the compiled-default equivalence check - 1701.5 ms vs `d1700`'s 1713.0 ms, i.e. the flag's default is
  the compiled default within 12 ms).
- **Host-validated before the GPU was touched** (no mutex, no CUDA): `--pp-prewarm-delay-ms 99999` refused;
  the flag refused for modes 0/1/3/8 (exit 2, "requires --pp-prewarm with bit 4 set") and accepted for mode
  4; off-state byte-identity re-confirmed (flag absent vs `--pp-prewarm 0`); both edited drivers parse; the
  dry run prints 13 arms with exactly the five dose values on the five dose arms.
- **Caveat on the driver's own curve printout, stated rather than glossed.** The driver now prints this table
  itself, but the launcher starts it with `Start-Process -WindowStyle Hidden` and neither captures its stdout
  nor inspects its exit code, so that printout is NOT in any artifact and is the one piece of this change a
  run cannot demonstrate. It was validated by (a) parsing clean and (b) re-running its exact data-access and
  formatting expressions host-only against the real verdict files, which reproduced this table and the 0.19 s
  spread. Every number in this record is independently reproducible from the verdicts, which carry
  `prewarm_delay_ms`, `prewarm_total_ms`, `embed_excess_total_ms`, `pass1_wall_s` and `duration_s`. The
  launcher-side capture / exit-code gap is recorded as a follow-up, not fixed here.
- **Durability fired a third time:** the falsifier run's 13-arm evidence was archived to
  `build/llama-cuda/runs/20260922T124413Z/` before this run overwrote those paths.

#### WHAT THE ONE-OFF IS (2026-09-22, host-only, from the recorded traces - no GPU, no new arms): the driver-side commit of the 17.23 GiB registered mapping, absorbed by chunk 0's embed step BEFORE any staging copy runs

The dose curve left one question open: what actually serializes the first pre-pass-1 consumer. It is answerable
from traces already on disk, because `host_mem` (T-GPLR-41 recon) samples the process working set at pass
boundaries and splits it into FILE-backed and private bytes, and `embed_chunk` times the layer-major embed step
per chunk.

- **Where it is: chunk 0's embed step, and nowhere else.** 4k control `embed_chunk` chunk 0 = **1907.1 ms**
  (span 1907.1, setup 0.05, compute 1907.0) while chunks 1-7 are 0.6-0.9 ms each; `host_ledger` pass-0
  `embed_us` = 1911.8 ms with `loop_us` -1912.3 (the negative residue IS this stall). Same law at the other
  shapes and on a different binary: 50k chunk 0 = 2002.7 ms, 200k chunk 0 = 1875.4 ms, other chunks 0.63 ms
  median in both.
- **The staging copies are NOT serialized - this corrects the framing.** The layer sweep's first
  `copy_complete` transfer is 31.6 ms in the control, 31.5 ms in the 1700 ms delay arm and 31.5 ms in the walk
  arm (median 31.4-31.5 ms over 240 copies in all three). Whatever the one-off is, it is paid BEFORE the first
  staging copy, by the embed step, which is why every downstream op looks normal.
- **What its magnitude equals: the not-yet-committed part of the mapped span.** `ws_file_bytes` (file-backed
  working set) is 1.95 GiB at pass-1 begin and 19.03 GiB at pass-1 end in EVERY arm and at EVERY shape
  (+17.08 GiB, i.e. the whole 17.23 GiB file span minus what the load already touched), and chunk 0's embed
  time is exactly the outstanding remainder at ~8.7-10.3 GiB/s: control 17.30 GiB -> 1907 ms (9.3 GB/s),
  d200 -> 1623 ms, d600 -> 1055 ms, d1200 -> 635 ms, d1700 (1.52 GiB left) -> 132 ms, walk (0.24 GiB left) ->
  15 ms, 50k 17.08 GiB -> 2003 ms (8.7 GB/s), 200k -> 1875 ms (9.3 GB/s). `sys_free` corroborates: 50.61 ->
  33.19 GiB across pass 1.
- **It is NOT our engine code doing the paging.** In mode 4 the process only sleeps, and the prefetch worker
  is not even created (these arms never pass `--pp-prefetch-ahead`), yet the file working set still grows at
  ~10 GiB/s during the wait: 2.00 GiB in 200 ms (10.0 GB/s), 6.18 in 600 (10.3), 10.96 in 1200 (9.1), 15.78 in
  1700 (9.3). The registration call itself is 1.6 ms for all 17.23 GiB (`mmap_register_init`, mode
  `init-once`, 40 merged ranges) and `commit_bytes` stays 5.58 GiB, so registration ENQUEUES the work and does
  not perform it.
- **Inferred, labelled:** the enqueued work is the CUDA driver's page-lock/residency commit of the range handed
  to `ggml_backend_cuda_register_host_range` (T-GPLR-37; its own probe measured 0.16 ms per 452 MiB window to
  register and a pinned-ceiling 14.44 GB/s DMA afterwards). The trace cannot see driver internals, so "driver
  worker" vs "OS cache manager servicing the lock" is NOT separated here; what IS excluded is that any runner
  thread performs the touch.
- **`pin` is the control that proves the source path is the cause:** `eng_bytes` 1.46 GiB (no mapped span),
  chunk-0 embed 2.2 ms - and its file working set still grows to 19.03 GiB because building the pinned mirror
  reads the same file, which is the ~1.7 s its pass-1 wall (5.39 s) pays instead.
- **Practical consequence, and the only shape of a win left:** the cost is a fixed ~17 GiB commit at ~9.5 GB/s
  that starts when the span is registered - which happens AFTER model load and after the graph prebuild
  (registration at +65.6 ms on a trace whose t0 is already post-load), leaving pass 1 to absorb the remainder.
  If registration (and therefore the commit) can be moved EARLIER - to the first point where the package spans
  are known, i.e. before the remaining load-time and VRAM setup - the commit would run concurrently with work
  the run already does, and the mechanism predicts pass-1 wall falls by the hidden amount with NO waiting at
  all. This is the first experiment the dose curve's failure points to rather than away from. Risk to measure,
  not assume: the commit competes for host RAM bandwidth with load I/O.
- Evidence: `build/llama-cuda/gplr53-a4-*.jsonl` (`host_mem`, `embed_chunk`, `copy_complete`), the 50k/200k A3
  traces, and the per-arm verdicts. No new arm ran for this section, nothing was rebuilt, and no performance
  claim is made.

#### Launcher driver capture (2026-09-22, host-only, no GPU): the arm drivers' own output and exit codes are now artifacts

The dose-curve turn exposed a provenance hole in the documented entry point. `run-gplr53-session.ps1` started
its drivers with `Start-Process -WindowStyle Hidden` and NO redirection, so everything a driver printed was
discarded and its exit code was never read. Only the per-arm verdict files survived - which meant a driver's
OWN computed verdict lines (the pre-registered removal table, the `FALSIFIER ...` line, the dose curve) existed
nowhere on disk, and a driver whose post-arm computation threw after writing PASS verdicts was
indistinguishable from a clean run.

- **What changed:** each driver's stdout and stderr are redirected to `<session>-driver-<a3|a4>.out` / `.err`
  in `build/llama-cuda/`; the last 20 non-empty lines are echoed into the session log; both files are copied
  into the run's archived evidence (`Save-RunEvidence` gained a `LogPrefix` glob, so the PREVIOUS run's driver
  output survives the run that replaces it); and the driver's exit code is reported, tabulated in the roll-up
  and folded into the session verdict - a driver that exits non-zero, or that never exits, FAILS the session
  even when every arm file reads PASS.
- **A measured PowerShell trap, found by probing rather than assuming, and it would have failed every clean
  run.** With `-RedirectStandardOutput`/`-RedirectStandardError`, a `Start-Process -PassThru` process object
  reports `HasExited` correctly but returns **NULL `ExitCode`** after a bounded `WaitForExit` - and
  `$null -ne 0` is TRUE, so the naive check would have marked every session FAIL. Probe (host-only, four
  shapes): timed wait without redirects -> 7; `-Wait` without redirects -> 7; timed wait then no-arg
  `WaitForExit()` without redirects -> 7; **redirects + timed wait -> empty**; redirects + `-Wait` -> 7;
  redirects + `EnableRaisingEvents = $true` before the wait -> **9** (the fix). `-Wait` also works but blocks
  UNBOUNDED, which is rejected on purpose: this block's own recorded hazard was exactly an unbounded wait
  (the drivers' `WaitOne(3600 s)` before their probe). Retaining the handle keeps the wait bounded at 15 s.
- **Proof, host-only (no GPU, no mutex, no CUDA).** Capture harness, using the launcher's exact flag set:
  `[1]` redirected non-zero exit -> `exit=7`, stdout and stderr files both exist, and the driver's own
  `FALSIFIER` line survived (`True`); `[2]` clean exit -> `exit=0`; `[3]` a stub that outlives a 1 s bounded
  wait -> `exit=not-exited` (the guard never reads a live process's code; the stub was then killed);
  `[4]` NEGATIVE CONTROL, the OLD shape with no redirect -> `captured=False`, i.e. the output is lost, which
  is the defect being fixed; `[5]` the launcher's own fold -> `badDrivers=3` over exits `[7,0,not-exited,5]`,
  so a clean run passes and every anomaly fails. Archive side: `-ArchiveOnly` produced a manifest whose
  entries now include `a4-driver-a4.out` and `.err` (byte count matching the source, source hash unchanged),
  and the test removed the run directory it created - the four real run directories are untouched.
- **Limits, stated rather than glossed:** the capture path inside the launcher itself executes on the next real
  session, so what is proven here is the flag set and the fold, not an end-to-end GPU run; and the two proof
  scripts live in `build/` (gitignored), which is the already-recorded "proof tooling outside tools/" gap -
  noted, not fixed. No arm behavior, tag, timeout, pre-registered rule, threshold or retention state changed,
  and no GPU time was spent.

#### Driver capture - PRODUCTION EXERCISE (2026-09-22 13:10Z, one mutex hold, 13/13 arms PASS): proven end-to-end, and it immediately put an unreachable pre-registered band on disk

The capture path above could not be exercised host-side (it needs real drivers), so the next A4 session was
used as its production test.

- **The run:** `tools/run-gplr53-session.ps1 -Session a4`, `gpu_pre_run` clean, 13 arms, `SESSION VERDICT:
  PASS (13/13 arms)`, driver `exit 0`, arm verdicts 13:10:07Z-13:13:33Z. Binary pair
  `exe 1BDE9D5AFF326CB7` / `llama.dll BE55C90F0A5D796E`, read from the ARCHIVED predecessor's own verdicts:
  the same pair the 12:44Z dose run used, so that comparison below is same-binary. `gpu_post_run = clean`
  (now persisted in all 13 verdicts) and `mutex released` are printed only by the driver, so before this
  change those two claims existed nowhere.
- **Artifacts:** `build/llama-cuda/a4-driver-a4.out` (3782 B, 67 lines) and `a4-driver-a4.err` (9265 B, the 13
  per-arm `check-telemetry: OK (1927 events, 3 slots)` summaries). The stdout is exactly what was being thrown
  away: the exe/dll identity and mutex lines, every arm's `pass1 wall ... excess ... ms`, the pre-registered
  removal list, the `FALSIFIER` line, the dose table, the conserved-sum line, `gpu_post_run`, `mutex released`,
  `gplr53-a4 VERDICT PASS`.
- **Archive pickup, in production:** `-ArchiveOnly` (host-only - no GPU, no CUDA, no mutex, no clearing)
  archived the live files into `runs/20260922T131541Z`: 67 entries of which 2 are the driver logs, copy and
  source `sha256_16` both `3A3230F71F5BE161` (`a4-driver-a4.out`) and `FADC9B6F9A1BA3B4` (`.err`), sources
  untouched, and the archived copy still contains `VERDICT PASS` / `FALSIFIER` / `DOSE CURVE` / `gpu_post_run` /
  `mutex released`.
- **No double run:** one launch line, one predecessor archive directory in that window (13:10:06Z), one
  monotone sequence of 13 arm verdicts, one driver log. What the operator's Stop killed was the POLLING
  command, not the session - the session finished at 13:13:38Z and its log is complete. No
  `llama-layer-streaming.exe` survived, the named mutex `Global\AI-GPU-EXCLUSIVE-v1` does not exist (a 0 ms
  `WaitOne(0)` probe finds no live holder), and the card was clean before and after.
- **Same-binary reproduction against the archived 12:44Z dose run:** the qualitative pattern repeats exactly -
  `pw1` walk 99.3% -> 99.3%, `pw2` 99.9% -> 99.9%, `pw3` 99.9% -> 99.9%, `pin` 99.9% -> 99.9%, `delay` 92.7% ->
  93.0%, `wrongtouch` 91.9% -> 86.2%, control repeat ~0 - while magnitudes drift: the control itself moved
  1906.0 -> 1804.6 ms and the five dose arms moved +110.5 to +312.5 ms, so this metric's run-to-run floor is
  ~+-110-130 ms (~6-7% of the control; arm `d000`, which waits 0 ms, reads -7.8% "removal"). The dose law is
  stable: absorbed per ms waited is 1.00/0.95/0.96/0.98 at 200/600/1200/1700 ms and the conserved sum holds
  (3.70-3.76 s, spread 0.06 s). No performance claim is made from any of these numbers.
- **DEFECT RECORDED 2026-09-22, FIXED 2026-09-23 (see the SETTLED section; the fix is the owner's
  settlement, not an implementer retune).** It was left unfixed on the day because the band is
  pre-registered. The driver stated its band on the pass-1 WALL
  (`tools/run-gplr53-a4-arm.ps1:11-15`: "the comparison quantity is the pass-1 wall against the same-session
  off control"), its FALSIFIER paragraph measures against the control's EXCESS (`:25-27`), and its code applies
  the 70% test to wall removal (`:246-264`). The two disagree, and the wall band is **unreachable by
  construction**: pass 1 has a ~1.75 s floor, so the walk - which removes 99.3% of the excess - scores only
  **50.3%** on the wall, and the driver itself prints `pw1/pw2/pw3 pass-1 wall removal = 50.3/50.6/50.8%`, all
  below its own 70% gate. Consequence on this run: the driver printed `FALSIFIER SURVIVED ... removes only
  47.2% (< 70%)` (the `delay` arm's WALL) while the SAME arm's EXCESS removal is **93.0%** (126.2 ms against
  1804.6 ms) - the quantity this record's withdrawal rests on. The printed line and the recorded conclusion
  therefore point opposite ways, and neither is wrong arithmetic: the metrics are. Escalates to the Planner,
  because the readable instrument is the excess term and the wall band cannot express (i) at all.
- **FIXED 2026-09-23.** The escalation was settled as option A (the band is the excess) and the driver now
  prints, per arm, `excess removal {e}% (BAND) | pass-1 wall removal {w}% (relocation instrument)`, then the
  `WALL CEILING` line once, then an excess-based `FALSIFIER FIRED/SURVIVED`, with an explicit
  `NOT EVALUABLE ... ceiling {c}% < 70%` note beside the wall number whenever the ceiling cannot express the
  band. On the recorded 13:10Z run that line now reads FIRED at **93.0%** (excess) instead of SURVIVED at
  47.2% (wall) - no number was recomputed, only the quantity the branch reads was made the one the contract
  always named.

#### ESCALATION - SETTLED 2026-09-23 BY OWNER DIRECTIVE: OPTION A (the band is the per-chunk embed EXCESS); the analysis below is the decision record as it stood (2026-09-22, host-only, no GPU, no arm re-run): which quantity does the pre-registered `>= 70%` band actually test?

**The question.** `run-gplr53-a4-arm.ps1:11-15` states the comparison quantity as the pass-1 WALL; `:25-27`
states the falsifier against the control's EXCESS; `:246-264` applies the 70% test to wall removal. One of those
three is the band's quantity and the other two are not, and the answer changes what the printed line means - on
the 13:10Z run it printed `FALSIFIER SURVIVED ... 47.2%` while the same arm's excess removal is 93.0%. Nothing
below retunes a threshold, re-scores a past run, or moves a gate; it separates a measurement from a decision.

**Measured: the wall band is unreachable at every shape, by construction.** Pass 1 = the same work as pass 2
plus the one-off, so the wall's removable part is the excess and its floor is the pass-2 wall. Both
formulations, from the verdicts already on disk (the generous one is the pass-2-difference):

| shape | pass-1 wall | pass-2 wall | most generous wall ceiling | excess share of pass-1 wall |
|---|---|---|---|---|
| 4k, 13:10Z | 3.56 s | 1.56 s | **56.2%** | 50.7% |
| 4k, 12:44Z (archived) | 3.67 s | 1.57 s | **57.2%** | 51.9% |
| 50k (A3) | 20.25 s | 18.12 s | **10.5%** | 9.9% |
| 200k (A3) | 135.84 s | 134.67 s | **0.86%** | 1.4% |

Even the most generous reading cannot reach 70% at any shape, and the ceiling FALLS as the prompt grows,
because the one-off is shape-invariant (~1.8-2.0 s) while the pass-1 wall grows with the token count. The
charter's priority shape - long context - is exactly where the wall band is most hopeless (0.86% at 200k). The
driver's own output already carries the proof: it printed `pw1/pw2/pw3 pass-1 wall removal = 50.3/50.6/50.8%`,
so the WALK - the arm that branch (i) is *defined* by - would fail its own 70% wall band while removing 99.3% of
the excess.

**What each branch can actually test.**

| branch | on the excess | on the wall |
|---|---|---|
| (i) removes >= 70% -> page-side (H1) | testable: walk 99.3%, delay 93.0%, wrongtouch 86.2% | **not testable** - ceiling 56%, so it can never fire |
| (ii) removes it -> submission-side (H3) | testable | **not testable** - same ceiling |
| neither -> escalate | testable | reachable, but reached by *every* arm including one that clears 99.3% of the excess - so it carries no information |

The wall band therefore cannot discriminate (i) from (ii) from "neither"; it returns "neither" for any input. Its
legitimate role is a different measurement entirely: the RELOCATION question, where the pre-pay plus the pass-1
wall is read as a **sum** (3.70-3.76 s across the doses, spread 0.06 s) and never as a removal fraction.

**What the A3 and A4 conclusions genuinely rest on (so the escalation cannot be read as reversing them).**
A3's `CHUNK0_SPIKE` rests on the per-chunk embed EXCESS bucketed by pass - chunk-0 share 0.9997/0.9987, excess
2001.91/1871.71 ms against a 0.63-0.64 ms warm median - with the three pass walls as corroboration only. A4's
H1-vs-H3 reading rests on the excess removal table (walk 99.3/99.3%, dummy 99.9/99.9%, both 99.9/99.9%, pin
99.9%, delay 92.7/93.0%, wrongtouch 91.9/86.2%, control repeat ~0 in both runs). Neither conclusion is
wall-band-derived, so neither is weakened by this finding; what is weakened is any impression that the printed
`FALSIFIER ...` line is the evidence. It is not - the excess table is.

**The zero-cost reporting fix (specified here, NOT applied - it touches no gate, no band value, and re-scores
nothing).** In `tools/run-gplr53-a4-arm.ps1`, inside the block at `:246-264` that already has `$ctrl` and
`$armRows`:

1. compute `$excessPct[$k] = (($ctrlExcess - [double] $armRows[$k].embed_excess_total_ms) / $ctrlExcess) * 100`
   next to the existing wall computation, and print both per arm on ONE line: `wall {w}% | excess {e}%`;
2. print the ceiling once, before the branches: `WALL CEILING: the excess is {c1}% of the control's pass-1 wall
   and the pass-2 wall is {c2}% of it, so no arm can remove more than {c}% of the wall at this shape - the 70%
   wall band is unreachable HERE by construction`;
3. when `$d -lt 70.0 -and $ceiling -lt 70.0`, print `FALSIFIER: NOT EVALUABLE on the wall (ceiling {c}% <
   70%); the excess removal for this arm is {e}%` instead of `FALSIFIER SURVIVED`. The threshold value, the two
   branches and every gate stay byte-identical; only the false impression that a 47.2% wall number is a failed
   test disappears, and the past runs need no re-scoring because nothing is recomputed retroactively.

**The decision itself is not an implementer edit.** Three options, with what each costs:

- **A - bind the band to the per-chunk embed excess** (the only quantity that can express (i) and (ii) at every
  shape). Cheapest and most sensitive; requires re-stating the band's unit in the contract, and keeps the wall
  as the relocation instrument it already is. *Recommended.*
- **B - keep the wall as the band's quantity and make it shape-relative** (e.g. `>= 70% of the measured
  ceiling`). Preserves the original text's letter, but the ceiling is itself derived from the excess, so it is
  option A with an extra indirection.
- **C - drop the band and report the two quantities plus the ceiling**, recording the (i)/(ii)/neither reading
  in prose per run. Most honest, least automatable, and no arm gate changes character either way.

Escalated because it changes product intent for the block (what an arm's PASS means), and the record's own
withdrawal rests on the quantity the printed line does not use. No GPU, no mutex, no arm re-run, no threshold
retuning, no re-scoring; retention state and the T-GPLR-55 shapes untouched.

**SETTLED 2026-09-23 - the owner chose A.** The band's quantity is the per-chunk embed excess removal against
the same-session off control; the wall stays the relocation instrument, printed with its ceiling and never as a
band. B and C are recorded as considered-and-rejected (B: option A with an indirection, since the ceiling is
derived from the excess; C: least automatable). What landed with the decision: the three-piece reporting fix in
`tools/run-gplr53-a4-arm.ps1` (both quantities per arm on one line, the ceiling once before the branches, an
excess-based verdict with the wall's non-evaluability stated), the contract text above, and this record. What did
NOT change: the 70% threshold value, the two branches' meaning, every arm gate, and every previously recorded
verdict - the past runs are read, never re-scored. Host-only: no GPU, no mutex, no arm re-run.

#### Evidence durability (2026-09-22, host-only, no GPU): re-runs can no longer destroy the previous run's artifacts

The hazard this fixes is measured, not hypothetical: the arm tags are FIXED, so the
second A3 run overwrote the first run's traces and verdict triple at the same paths,
and the original FAIL evidence survived only in this record's prose and in
`build/llama-cuda/gplr53-a3-session.out`.

- `tools/run-gplr53-session.ps1` now ARCHIVES the artifacts a run is about to
  replace into `build/llama-cuda/runs/<utc-stamp>/` - the `.jsonl` trace, `.log`,
  and the `.started` / `.verdict.json` / `.done` triple for every tag in the run -
  plus a `manifest.json` naming each file's source path, size and truncated SHA-256.
  It COPIES and never moves, renames or modifies anything, so the accepted A3
  evidence and every documented analysis path are untouched.
- Ordering is what makes it safe: the archive runs AFTER the busy preflight (a
  refused run writes nothing) and BEFORE `Clear-ArmVerdict` (a real run cannot lose
  its predecessor). A new `-ArchiveOnly` switch snapshots the current evidence and
  exits without launching, so the step is provable and usable on a busy card.
- Proof (host-only, no mutex, no CUDA): `-ArchiveOnly` produced
  `runs/20260922T122245Z/` with all 10 artifacts; `python build/check-archive.py`
  reports **copies identical: True, sources intact: True, ARCHIVE PROOF: PASS**
  (byte-for-byte comparison plus size and hash match against the manifest); and the
  source files' SHA-256 values were identical before and after the snapshot
  (`b4106aee...`, `da9cd9db...`, `cc..`, `48..`). The launcher's other paths were
  re-checked unchanged: `-ValidateOnly` exit 0 (DRY-RUN OK) and `-PreflightOnly` on
  a busy card exit 125.
- **Production proof (2026-09-22, 12:28:01Z):** the archive step then fired for real on the second A4
  attempt, capturing the displaced shape-bug attempt into `runs/20260922T122800Z/` (24 artifacts plus
  `manifest.json`) BEFORE the six PASS arms overwrote those paths. Verified after the fact: all 24 archived
  files DIFFER from their live counterparts (0 identical), the archived `.done` files are the six `FAIL 1`
  verdicts (12:25:32-12:25:50Z) while the live ones are `PASS 0`, and the two per-run directories have unique
  UTC stamps (`20260922T122245Z` for A3, `20260922T122800Z` for A4) so runs sit side by side instead of
  overwriting each other. The accepted A3 evidence is untouched by both sessions.
- Scope note: this is tooling only. No arm action, no arm tag, no pre-registered
  rule, no engine or runner code, and no accepted A3 evidence was changed.

- **Mode-0 identical to the off default: host-verified, not assumed, and the
  run-time half is now closed.** The off default (flag absent) and explicit
  `--pp-prewarm 0` produce BYTE-IDENTICAL runner output (`diff` clean) with zero
  `pp-prewarm(` chatter in either, and the code path agrees: `pp_prewarm == 0` makes
  `pp_prewarm_run()` return before touching a byte, and the runner only calls it when
  `pp_prewarm > 0`. The run-time half landed with the second A4 attempt: the off arms
  (`pw0`, `pin`) log no `pp-prewarm(` line and report `prewarm_engaged: false`, modes
  1-3 each log their own line and report `prewarm_engaged: true`.

### T-GPLR-56 (draft): take the registered-span commit OFF the first prefill's critical path - start it earlier and overlap it, or shrink the span - judged on a relocation-proof sum

Status: CLOSED (owner decision, 2026-09-25) - A1 and A2 remain host-validated evidence; A3 will not be run. The first-token registration cost is accepted as startup-only for the long-lived multi-turn server target, so no further mutex hold, 4k/50k/200k arm, default change, or performance claim is authorized by this block. The implementation checkpoint and refusal/measurement record remain historical evidence.

DECIDED (owner, 2026-09-22, both change product intent and are recorded here verbatim):

1. FIRST-TOKEN PP LATENCY IS IN SCOPE, BOUNDED: A3 decides. One 4k mutex hold with the
   pre-registered Q gates unchanged; the 50k/200k holds stay owner-invoked per length
   (the charter tension - 50.7% of the 4k wall vs 1.4% at 200k - is exactly why the
   ladder is not pre-authorized). A negative A3 closes the block charter-correct:
   "overlappable only at short shapes" is a decision-useful result under this reading.
2. REGISTRATION-TIMING AUTHORITY = A2's NARROWED FORM, RATIFIED: paced issuance from
   the EXISTING call site (p1-pace) plus the incremental-with-prebuild fallback
   (p1-overlap, only if p1-early shows contention) are the authorized mechanisms.
   Tensor-layout span derivation is DEAD per A2's negative finding. Registration
   DURING the model load is explicitly FORBIDDEN (the loader interaction is unmeasured;
   making it an arm would broaden the block) - recorded here so no later arm adds it
   by implication. p1-early stays as the pre-measured negative control it became.

OWNER CALLS - these decide whether the block is worth its mutex holds, so answer before
marking it ready (both change product intent, neither is an implementer choice):

1. IS FIRST-TOKEN PP LATENCY IN SCOPE? The one-off is ~1.9 s of LATENCY at every shape,
   and by throughput it is 50.7% of the 4k pass-1 wall, 9.9% at 50k, 1.4% at 200k - the
   payoff is LARGEST where the charter's long-context priority is NOT. Under a latency
   reading this block is worth running at every shape; under a strict long-context tok/s
   reading it is a short-shape win and possibly not worth a hold.
2. IS EARLIER REGISTRATION ACCEPTABLE? Today the call sits AFTER the package prebuild
   (`tools/layer-streaming/main.cpp:4540`, following `llama_layer_streaming_prebuild` at
   `:4472`, itself after `tel_run_start` at `:4362` and `llama_model_load_from_file` at
   `:3637`), so the spans come from the prebuild's `pkg_cache`. Moving it earlier means
   deriving the spans from the model tensor layout instead, or registering incrementally
   as packages are built, or registering while the loader is still reading the file
   (pages the loader itself touches - interaction unmeasured). The AGENTS rule that
   package spans must already be known is why this is an owner call.

Dependencies: T-GPLR-37 (the mmap-registered source whose commit this pays); T-GPLR-53
(the attribution and dose curve - this block's premise); T-GPLR-52 (the copy-path laws the
span arm must not disturb). Not blocking: T-GPLR-54/T-GPLR-55 (their shapes stay untouched).
Commit: nested 8633f8099 (pre-sink process-life markers + tel-mark test); outer 19d7e33 (acceptance-quantity Q checker), ea1ab5d (host-suite registration), f2c385f (records); T-GPLR-56 A3 IMPLEMENTATION CHECKPOINT nested d8d7253a7 + outer 9a012b5, pin c075671 (pace flags, analyzer, arms, launcher) - the 4k hold has NOT run, so no verdict commit yet

Goal. Make the first prefill stop paying the ~1.9 s commit of the 18,500,517,888 B
registered span IN ITS CRITICAL PATH - by issuing registration earlier and overlapping the
commit with load/prebuild work, by sharding it so no single consumer absorbs the whole
remainder, or by shrinking the span the first chunk actually needs - and prove any gain on
a quantity a relocation cannot satisfy. Stripping an "excess" is explicitly NOT the aim:
the dose curve showed the one-off is real pending work with a ~1:1 exchange rate, so only
overlap or reduction can win.

Scope. Opt-in, CUDA-only, DEFAULT OFF: one variant flag on the existing registration path
(0 = off, then early / pace; `p1-overlap` exists only as the pre-authorized fallback
mode), the span-map precondition, and the telemetry the
acceptance quantity needs. The flag is refused BEFORE model load when out of range (the
T-GPLR-53 prewarm-flag lesson: a no-op refusal placed after the load is unobservable).
Registration during the model load is forbidden (owner decision, 2026-09-22).
Non-goals: no change to package contents, routing, KV/recurrent state, slot policy, or the
copy-source default; no change to the pre-warm/dose modes; no default behaviour change; no
new dependency; no claim about the pinned path's first-pass cost (recorded, not owned); no
cold-cache claim (Windows file cache is uncontrolled); no re-tuning of any A4 threshold.

PRE-REGISTERED ACCEPTANCE QUANTITY (required - chosen so a relocation FAILS it).
Q_arm = `pass1_end_ts - proc_start_ts` (ms), per arm, from the trace's own clock, i.e. the
SUM from process start to the end of the first prefill pass - never the timed bracket
alone. An arm PASSES only if BOTH (a) its Q falls by MORE THAN 250 ms against the
same-session control AND (b) its `pass1_wall` falls.

Q reads EXACTLY two events, and refuses rather than guessing:

- `proc_start` - a pre-sink marker written as the trace's FIRST line, carrying `t_us`,
  `seq: 0`, `pre_run: 1` and NO `run_id`; emitted at `main()` entry (after
  `ggml_time_init()`, see the A1 result's measured trap).
- `host_mem` with `label == "pass_end"` and `pass == 1`, carrying `t_us`. These are
  recon-gated, so EVERY Q arm must run with `--pp-recon 1`: that is a precondition of the
  quantity, not a fallback.

`python tools/check-gplr56-q.py` exits 2 and prints NO value when either marker is absent,
when the marker order is broken, or when the control cannot yield Q; with `--control` it
applies the rule above and prints PASS / FAIL / FAIL-RELOCATION. An absent marker is never
read as a zero - the T-GPLR-53 decay rule once refused its own input, and Q must not repeat
that class of defect from the other side. Margin and sum rationale: A4 measured
the pilot instance of this exact quantity (pre-pay + pass-1 wall) at 3.43-3.76 s across
five doses (spread 0.19 s) and 3.70-3.76 s in the repeat (spread 0.06 s), with flat arm
walls (14.5-15.2 s). A mode that merely starts waiting sooner leaves the sum unchanged and
FAILS by construction - the precise failure the pre-warm family exhibited.
Reported but NEVER gating: `pass1_wall_ms`, the T-GPLR-53 excess, `registered_bytes`,
`register_init_us` (the ~1.6 ms call vs the ~1.9 s commit), `ws_file_bytes` at pass-1 begin.
EXPLICITLY NOT INHERITED: A4's `>= 70%` band (on the per-chunk embed excess since the
2026-09-23 settlement; its earlier wall-band form was unreachable at every shape - ceiling
56.2%/10.5%/0.86% at 4k/50k/200k, see the ESCALATION section). Each arm writes
`quantity: q_procstart_pass1end` into its own verdict, so no arm can be scored on a number
it did not name.

Arms (each states its own quantity BEFORE it runs). REVISED by the A2 result below: the
original `p1-shard` and `p1-span` are MERGED into `p1-pace`, because A2 measured that they
name the same mechanism and that `p1-span` as first written is a no-op.
- `p1-off` - registration at its current site (`main.cpp:4540`), nothing else changed -> Q.
- `p1-early` - registration moved to the earliest legal point (`main.cpp:4332`, right after
  `exec_layers` is assigned) -> Q. A2 measured this head start as 66.4 ms, so it is expected
  to FAIL the 250 ms margin; it is kept as the negative control for "earlier is enough".
- `p1-pace` (was `p1-shard` + `p1-span`) - layer 0's group registered before the pass, each
  later layer's registration issued a bounded distance ahead of its first staging copy, so
  pending (unlocked) bytes stay near one group instead of 17.23 GiB -> Q + how many groups
  were issued before their first touch + `copy_complete` transfer medians (contention).
- Fallback only if `p1-early` shows contention: `p1-overlap` - registration issued
  concurrently with the prebuild, load untouched -> Q. Host RAM bandwidth contention is to
  be MEASURED, not assumed.

Known evidence (host-only, from recorded traces and the code - no GPU):
- The call returns long before the work is done: `mmap_register_init` reports
  18,500,517,888 B in 2345 us (1.6 ms in the 4k arms), 40 merged groups; the file-backed
  working set then grows 1.95 -> 19.03 GiB during pass 1 in EVERY arm at ~9.5 GB/s, and in
  mode 4 - which only sleeps - it grows 2.00 GiB/200 ms, 6.18/600, 10.96/1200, 15.78/1700,
  so the process performs no paging of its own.
- The first consumer is chunk 0's embed step: 1907.1 ms at 4k, 2002.7 at 50k, 1875.4 at
  200k against 0.63-0.9 ms warm, and no staging copy inflates (first `copy_complete`
  31.5-31.6 ms, median 31.4-31.5 ms over 240 copies).
- The payer is TIME, not pages: a no-touch delay removes 93.0% of the excess while touching
  zero pages, and absorption is ~1:1 with the wait at 200/600/1200/1700 ms.
- Call-site sequence today: load `:3637` -> run_start `:4362` -> prebuild `:4472` ->
  register `:4540` -> first-token PP `:4744`; the registration itself is
  `register_all_pkg_sources()` (`llama-layer-streaming.cpp:660-740`, two-pass coalescing,
  per-group `mmap_register`, `cudaHostRegister` at `:707`).

Failure contract. Every mode fails LOUDLY with structured telemetry: a registration failure
unpins and refuses (existing behaviour); a `register_call_ts` that lands after the first
consumer's stall is a defect, not a result; an arm with no `register_init`, a stale span
map, or a working-set law that does not appear must refuse rather than silently fall back
to another copy source. An arm whose `pass1_wall` falls while Q stays flat is a FAIL and
must be reported as one, in the session verdict and in the record.

Implementation steps:
- A1 (host-only, no GPU, no mutex): make Q computable - a process-start timestamp plus the
  pass-1 end (or a documented derivation from `pass_begin`/`pass_end`/`phase_summary`), and
  a `register_call_ts` so an arm can show WHERE in the sequence registration was issued.
  Validate by re-deriving Q on the recorded A4 13:10Z arms: the control's Q must land below
  its own `duration_s` by the post-pass-1 work the verdict's pass walls already bound.
- A2 (host-only): the span-map precondition - show the registration's merged groups are
  derivable from the model tensor layout without the prebuild, or state plainly that they
  are not, which kills `p1-early`/`p1-span` for free before any GPU time is spent.
- A3 (one mutex hold, 4k): `p1-off` + `p1-early` + `p1-pace` (the A2-revised set;
  `p1-overlap` is the fallback only if `p1-early` shows contention), reporting Q,
  `pass1_wall`, excess, `register_init_us` and the working-set law per arm.
- A4 (one mutex hold per length, only for an arm whose Q fell > 250 ms at 4k): 50k, then
  200k, one length at a time, where the commit competes with an 18 s/135 s pass.

Acceptance criteria (falsifiable):
- A1: Q computed for >= 3 recorded traces, host-only, reproducible on a second invocation,
  with the derivation documented in the tool's own docstring.
- A2: the merged-group table reproduced without a GPU, or an explicit negative finding.
- A3: at least one arm's Q < control Q - 250 ms AND its `pass1_wall` < control
  `pass1_wall`, in one session, with the control's own Q stable across the two archived
  control runs within 250 ms; every arm's quantity present in its own verdict field.
- Relocation falsifier: any arm whose `pass1_wall` falls while Q does not is reported as
  FAILING, explicitly, in the session verdict and the record.
- A4: the winning arm's Q falls > 250 ms at 50k and at 200k; if it does not, the recorded
  finding is "overlappable only at short shapes" and the block closes NEGATIVE for the
  charter's long-context priority - itself the decision-useful result.
- Default preservation: with the new mode absent or 0, the runner output is byte-identical
  to today (`diff` clean) and the existing `test-layer-streaming-*` family stays green.

Focused validation commands and expected results:
- `cmake --build build/llama-cuda --target llama-layer-streaming` -> green; record the
  exe/dll hashes, because a new pair means the control must be re-recorded.
- `powershell -File tools/run-gplr56-session.ps1 -Session p1 -PreflightOnly` -> exit 0 on a
  clean card, or 125 within seconds naming the PID when busy; never `-AllowBusy`.
- `python tools/check-gplr56-q.py --selftest` -> **PASS 7/7 (measured)**, including the
  relocation pair that must be reported FAILING; `--trace <arm>` prints Q, `pass1_wall`, the
  load span and the arm's declared quantity, and refuses (exit 2) with the missing marker
  named when `proc_start` or the pass-1 `pass_end` is absent.
- `ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-tel-mark` -> **PASS
  (measured)**, and `tools/run-tests.ps1 -Suite host` -> **42/42 PASS (measured)**.
- GPU arms only through `tools/with-gpu-lock.ps1`, via the session launcher, which mirrors
  `tools/run-gplr53-session.ps1`: fail-fast preflight, predecessor archive, per-run driver
  capture with the exit-code fold.
- `python tools/check-doc-sizes.py` and `python tools/resplit-gplr-plan.py --verify` -> PASS.

Required state/document updates: a `docs/currentstate.md` entry per executed arm; the SDD's
mechanism section only if a mode is adopted (owner decision); this block's status lines,
including the arms that did NOT run, named rather than inferred.

Escalation condition: any mode needing the package-span knowledge in a form the SDD does not
define; any contention result where moving registration earlier grows total load time by
more than the commit it hides; any arm where the commit turns out to be already off the
critical path, which would refute the T-GPLR-53 attribution and belongs back with it.

#### A1 result (2026-09-22, host-only, no GPU): Q was not computable from ANY recorded trace - the telemetry that fixes it is now written and host-proven, and the real-trace shape is left to an arm

Evidence: `build/llama-cuda/gplr53-a4-pw0.jsonl` (the 13:10Z control arm, 1,927 events),
read with no GPU and no mutex.

- **Measured - the trace begins at `run_start`.** The earliest `t_us` in the whole file is
  `run_start`'s own `t0_us` = 2,974,883 (the first later event is 3 us after it). So the
  model load, `ls.init`, the capacity preflight and the boundary-layer selection
  (`main.cpp:3637 -> :4106 -> :4332 -> :4362`) are ALL absent from the trace, and the
  clock's zero is before them with nothing recording where.
- **Consequence: the criterion is unmet.** Only an arm's own wall bounds the missing span -
  14.6 s (`duration_s`) against a last event at 12,282,418 us leaves ~2.3 s containing BOTH
  `exec -> clock zero` and the teardown tail, indivisible. An error bar of order 1 s against
  the 250 ms margin makes Q unusable as written, so A1's acceptance criterion ("Q computed
  for >= 3 recorded traces") is NOT satisfied and is replaced by this finding.
- **What IS derivable today (measured, and enough to design the arms):** the pass-1 bracket
  from `host_mem` (`pass_begin` 3,041,260 -> `pass_end` 6,600,050 = 3,559 ms, matching the
  verdict's `pass1_wall_s` 3.56); the file-backed working set 2,104,336,384 ->
  20,428,955,648 B inside it and flat in passes 2/3; the register loop ending at 3,041,238;
  and the first `copy_submit` at 4,855,496 (layer 0). The stall that matters is therefore
  **1,814.3 ms** from the end of registration to the first staging copy, with the pass's
  first `embed_chunk` completing 49 us before it.
- **The required telemetry is now WRITTEN (same day, still host-only, still no GPU).**
  `llama_ls_tel_mark()` / `llama_ls_tel_flush_early()` buffer markers taken before the sink
  exists and flush them, in call order, as the trace's first lines - necessary because the
  sink is `fopen(..., "w")` inside `llama_layer_streaming::init`, so anything earlier had
  nowhere to go. `main.cpp` marks `proc_start` at entry and `load_begin`/`load_end` around the
  load (the model path in the first, the outcome in the second). Default-on, no flag, no
  behaviour change on any path.
- **A MEASURED TRAP, and it would have killed every run.** `ggml_time_us()` on Windows is
  `((QueryPerformanceCounter - timer_start) * 1000000) / timer_freq`, and both statics are set
  by `ggml_time_init()`. A marker at the top of `main()` therefore divided by
  `timer_freq == 0` and died with `0xC0000094` - the new host test caught it on its first run
  (exit `-1073741676`). Fix: `ggml_time_init()` is called at main entry, and defensively
  inside the marker, so no caller can crash the process from telemetry; it is InitOnce-guarded,
  so `llama_backend_init()`'s later call (`src/llama.cpp:122`, reached from `main.cpp:3613`)
  stays a no-op. **Consequence to record:** every `t_us` in a P1 trace is now measured from
  process entry instead of from backend init, so P1 traces are not comparable to older ones on
  absolute `t_us` (all checkers use differences; `run_id` remains per-run).
- **Host proof, all measured (no GPU, no model):** `test-layer-streaming-tel-mark` (new CTest
  target, registered in `tests/CMakeLists.txt` and added to the host suite's Include pattern)
  asserts the emitted line shape - three markers in call order, `seq` 0/1/2, `pre_run` present,
  no `run_id`, monotonic `t_us`, the escaped Windows path, a NULL flush consuming the buffer,
  and an empty/NULL event name writing nothing. `tools/check-gplr56-q.py --selftest` = 7/7,
  including a relocation pair that must FAIL and four refusals; the tool REFUSES the recorded
  13:10Z trace (exit 2, `proc_start` missing) instead of printing a number for it; and
  `tools/run-tests.ps1 -Suite host` = 42/42 PASS with both new checks green.
- **Still GPU-side, named as an arm rather than assumed:** that a REAL run's trace carries the
  three markers in order ahead of `run_start`, with a load span that makes the pre-`run_start`
  time divisible at the 250 ms scale. That is `p1-off` replayed at the next clean card, and the
  fixtures above cannot stand in for it - they prove the emitter's shape, not one trace.

#### A1 real-trace confirmation RAN (2026-09-22, one mutex hold, PASS) - the three markers are lines 1-3 of a real trace, Q is a measured baseline, and the t_us rebase is clean on a real rebased trace

**Result, all of it measured.** On the retry window the card cleared at 13:43:26Z and the mutex
was free, so the single control arm ran under `tools/with-gpu-lock.ps1` in one hold:

| item | measured |
|---|---|
| wrapper | exit 0, console `Verdict: PASS (0) gpu_pre=clean gpu_post=clean`, mutex released |
| trace | `build/llama-cuda/gplr56-p1off-4k.jsonl`, **1,930 events** vs the 13:10Z control's 1,927 - exactly the three markers |
| arm verdict | `verdicts/gplr56-p1off.verdict.json`: status PASS, exit_code 0, duration_s 19.1 |
| lines 1-3 | `proc_start` (seq 0, `pre_run` 1, NO `run_id`, detail `""`, t_us **3**), `load_begin` (seq 1, detail = the model path, t_us **197,673**), `load_end` (seq 2, detail `"ok"`, t_us **5,471,465**) - call order, as pre-registered |
| line 4 / run_start | line 4 is `pkg_member`; `run_start` moved from line 43 to **line 46**, and its `t0_us` = **5,540,566** where the old base read 2,974,883 |
| **Q (baseline)** | **9,370.7 ms** (`proc_start` 3 us -> `pass1_end` 9,370,662 us), pass-1 wall **3,762.9 ms** |
| where Q goes | pre-load **197.7 ms**, model load **5,273.8 ms** - the remaining ~3.9 s is post-load setup, registration and the 1.9 s stall |
| gates on the new trace | `--check-telemetry` **exit 0**; `check-gplr53-embed-chunks.py --join` **exit 0**; `analyze-gplr53-decay.py` **exit 0, CHUNK0_SPIKE**; pass-1 file-backed WS **19.03 GiB** (same plateau as every earlier arm) |
| tool | `check-gplr56-q.py --trace <it>` **exit 0**; with the older 13:10Z control it **REFUSES, exit 2** (`proc_start` missing) - exactly as pre-registered |

The rebase is therefore confirmed in production, not just on the shifted fixture: `proc_start`
lands at t_us = 3 while the old base's `t0_us` was 2,974,883, and every existing gate still reads
the trace green. **Q's baseline is stated for this base only.**

**One uncontrolled variable, recorded rather than explained away:** the load here is
**5,273.8 ms**, where the 13:10Z run's pre-`run_start` span (2.97 s from backend init) implied a
much shorter load. Different clock bases make the two numbers non-comparable, and the Windows
file cache is uncontrolled in both - an arm run after ten minutes of a foreign process's GPU
work cannot be assumed warm. That is the repo's standing caveat on the cold-mmap source state,
not a code difference, and no claim is made about which figure is "the" load time.

**A provenance gap was claimed here, and the claim was WRONG (corrected 2026-09-22).** The
paragraph that stood here said `tools/with-gpu-lock.ps1` "does not persist either - nor a tag -
into `gplr56-p1off.verdict.json`, which carries only status/exit_code/duration_s/command". Read
the artifact back: that file carries `fields = {gpu_pre_run: clean, gpu_post_run: clean,
mutex_acquired: true, timeout_seconds: 7200}`, because the wrapper has written exactly that since
commit `001bf65` (2026-09-21) - `timeout_seconds`/`mutex_acquired` are produced by nothing else
in `tools/`, and `tools/test-arm-verdict.ps1` has asserted `fields.mutex_acquired` on a wrapped
run since then, so the earlier reading and the file cannot both be right. The probe evidence and
the tag were therefore never actually missing; `arm` IS the tag. **What was genuinely missing was
smaller - a probe that could not be taken was recorded as a clean card - and it is landed in the
subsection below, which supersedes this paragraph.**

#### Wrapper verdict evidence (2026-09-22, host-only, no GPU, no CUDA, no arm re-run): a probe that could not be taken was being recorded as a CLEAN card, and the wrapper's verdict now carries the tag and a fixed status vocabulary

**Original task / why it mattered.** Every arm in this repo - the A3/A4 arms, the P1 arms still to
come - is launched through `tools/with-gpu-lock.ps1`, so anything its verdict fails to record is
lost for the whole programme. The request was to make that verdict carry the evidence the wrapper
already prints, additively and honestly.

**Ground truth first, because the premise was half wrong.** The record above claimed the wrapper
persisted neither probe value on `gplr56-p1off.verdict.json`. It does persist them (and always
has, since `001bf65`): the file's `fields` carries `gpu_pre_run: clean`, `gpu_post_run: clean`,
`mutex_acquired: true`, `timeout_seconds: 7200`, and the tag is present as `arm`. So the probe
fields were NOT added by this change; claiming they were would have been a second false entry.

**The defect the request was actually pointing at, measured.** `Get-GpuComputeApps` treated
any output from `nvidia-smi` as either a parseable row list or nothing: a nonempty output with no
parseable row (a dropped `--format` flag, a driver banner such as `WARNING: infoROM is corrupted
at gpu 0000:01:00.0`, a changed output shape) fell through to an EMPTY list, and
`Format-GpuBusy` reported that as **`clean`**. Measured before the fix, same fixture, host-only
(`build/probe-class/fixture-probe-class.ps1`) - the fixture now measures BOTH sides in one run
(the pre-fix helper is extracted from `git HEAD` into a temp file and dotted in a child process,
the post-fix helper from the working tree), so the negative control survives the fix instead of
living only in this record: `build/probe-class/before-after.log`.

| stubbed probe output | `Get-GpuComputeApps` | `Format-GpuBusy` BEFORE | `Format-GpuBusy` AFTER |
|---|---|---|---|
| empty (no compute apps) | empty array | `clean` | `clean` |
| **nonempty, UNPARSEABLE** | **empty array** | **`clean`** | **`probe-unavailable`** |
| one parseable row | 1 row | `1234 deepseek_v4.exe` | `1234 deepseek_v4.exe` |

So a probe that answered nothing was persisted as evidence that the card was idle - the exact
"defaulted to clean" failure the request names - in every wrapped arm's verdict ever written.

**What landed (all additive).** `Get-GpuComputeApps` returns `$null` (UNVERIFIED) for nonempty
unparseable output; a new `Get-GpuProbeStatus` maps a summary to a fixed vocabulary; the
wrapper's `fields` gains `tag`, `tag_source` (`explicit` when the caller passed `-VerdictTag`,
`derived` when the wrapper guessed it from the command), `gpu_probe_source` (`nvidia-smi` vs
`stub` - a probe driven by `GPU_BUSY_PROBE_CMD` is not evidence about the card, and the gate
harness drives it that way), and `gpu_pre_status`/`gpu_post_status` = `clean` | `busy` |
`unavailable` | `not-probed`. **The pre-existing fields keep their names, values and meanings** -
including `gpu_pre_run`/`gpu_post_run` on the exit-125 and mutex-timeout paths, where
`not-probed` is now the readable name of the old `__unprobed__` absence.

**Proof, all host-only (~8 s, no CUDA, no model, no mutex on the shared lock).**
`tools/test-arm-verdict.ps1` grew a section 12 that runs the REAL wrapper through stubbed probes
and private per-case mutex names: **70 checks PASS, 0 FAIL** (`build/tav-fast.log`, exit 0). It
asserts the three properties the request asked for - (a) a wrapped run's verdict carries the
probe fields, the status vocabulary, the tag and its source and the probe source; (b) a consumer
reading ONLY the pre-existing fields sees an unchanged top-level key set and the identical legacy
projection (`PASS 0 ev-clean gpu_pre=clean gpu_post=clean mutex_held=True timeout=30`) - backed by
an audit of the in-repo consumers rather than only by that assertion: every read of a verdict in
`tools/*.ps1` (`$v.status`, `$v.exit_code`, `$cv.fields.<name>`) is of a field that already
existed, and the field-bag comparisons are guarded with `-contains`, so they cannot trip over a
key that is not theirs; (c) busy,
unavailable, **unparseable** and never-taken are each distinguishable from `clean`, with
`not-probed` produced end-to-end by holding a private mutex in another process and taking the
wrapper with `-TimeoutSeconds 0` (exit 124). Both machine states are on disk: while another
project held the lock, three consecutive runs printed the section 6 SKIP line and still exited 0
(`build/tav-clean-1..3.log`); on a free card the integration section takes the real
`Global\AI-GPU-EXCLUSIVE-v1` and probes `clean` (`build/tav-run3.log`). **Revision honesty on
that last log:** it was produced before the two harness-efficiency edits (sections 10/11 probe
stubs, `BusyGraceSeconds 0` on the stub cases), neither of which touches section 6 - and a fresh
free-card run could not be obtained to supersede it, because a bounded re-probe
(13:57:15Z-14:00:48Z, 8 samples) found the lock held by `deepseek_v4.exe` pid 21208 in every
sample, so section 6's real-lock path has evidence from the earlier revision only. The harness is
registered in the host suite as `[gpu-lock]`, and `tools/run-tests.ps1 -Suite host` is green on it
(`build/host-suite-gpulock.log`: exit 0, 42/42 CTest).

**Not re-run, named rather than inferred:** `build/gpu-gate-tests/run-gate-cases.ps1` (the
six-branch gate proof) was NOT re-run - its cases take the REAL shared lock with the default
7200 s timeout and that lock is held by another project right now (`deepseek_v4.exe`), so
re-running it would block the pass for up to two hours. Its branch conditions are untouched by
this change: empty CSV, one-row CSV, a failing `.cmd` and a flipping `.cmd`, none of which can
produce nonempty-unparseable output. The branches therefore have no un-run risk from this edit,
but they also have no FRESH evidence today, and saying so is the point.

**Glossary.** Wrapper = `tools/with-gpu-lock.ps1`, which holds the machine GPU mutex around an arm.
Verdict = the `<tag>.verdict.json` + `.done` pair a later pass reads. Probe = one read-only
`nvidia-smi --query-compute-apps` query telling whether a foreign compute process is visible.
`GPU_BUSY_PROBE_CMD` = the env-var hook that stubs that query for host-only tests. Tag = the name
the verdict is written under.

**No performance claim** (nothing here is timing evidence), **no GPU time spent, A4's
pre-registered band and its saved scoring untouched, the retention state and T-GPLR-55's shapes
untouched, and the GPLR-53 drivers' own field contracts unchanged.**

#### A1 first attempt: BLOCKED (card held, nothing started) - and the t_us rebase checked against every existing gate, host-only (2026-09-22)

On the first window the 4k control arm did not run: the card was held by a foreign process for the
whole window and no mutex was taken.

- **The blocker, with the probe recorded:** pid **22436** (`C:\Development\AI\colibri\
deepseek-v4-baseline\c\deepseek_v4.exe`) visible in every sample - 13:34:08Z, 13:34:14Z,
  13:34:54Z, 13:35:34Z, 13:36:14Z, 13:36:55Z, 13:37:35Z, 13:38:15Z, 13:38:55Z, 13:40:23Z
  (nine samples over ~6 min, never clean). No GPU time was spent, no CUDA process was started,
  no `--AllowBusy`, and no trace file exists for this arm - it is NOT STARTED, not failed.
- **The arm is one command, recorded here ready for the next clean window:**
  `powershell -File tools/with-gpu-lock.ps1 -VerdictDir build/llama-cuda/verdicts -VerdictTag
  gplr56-p1off -Command "<exe> --model C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf
  --prompt-file build/llama-cuda/gplr07-qa-prompts/anchor-4096.txt --pp-tokens 4096 --ctx-size 4352
  --ubatch 512 --pp-no-logits --tg-tokens 3 --residency moe-expert-bank --kv-cache f16
  --act-slab-dtype bf16 --pp-schedule layer-major --layer-slots 3 --allow-full-residency
  --residency-budget-mib 0 --pp-repeat 3 --copy-source mmap-registered --pp-async-feed 1
  --pp-recon 1 --telemetry build/llama-cuda/gplr56-p1off-4k.jsonl"`, with `<exe>` =
  `build/llama-cuda/bin/Release/llama-layer-streaming.exe` - the same argument set the A4 driver's
  `p1-off`-equivalent control (`gplr53-a4-pw0`) uses, minus the tag, so it is a control and not a
  new mode. `--pp-recon 1` is not optional: Q reads the recon-gated pass-1 `pass_end`.
- **What it must show, pre-registered:** the trace's first three lines are `proc_start`,
  `load_begin`, `load_end` in call order with `seq` 0/1/2 and no `run_id`; then the existing
  stream, in which `run_start` currently sits at LINE 43 with 40 `pkg_member` lines ahead of it;
  and `python tools/check-gplr56-q.py --trace <it> --control build/llama-cuda/runs/20260922T131006Z/
  gplr53-a4-pw0.jsonl` refuses on the control (it predates the markers) - so Q is stated as a
  baseline from the new trace alone until a second new-base control exists.
- **The rebase WAS checked, host-only, because the base change is exactly what breaks a green
  gate silently.** Model of the rebase: every `t_us`/`t0_us`/`run_id` in the recorded 13:10Z
  control trace shifted DOWN by a constant 2,000,000 us (a uniform shift is the faithful model -
  the base moved, no event's relative timing changed), giving
  `build/llama-cuda/rebase-check/shifted.jsonl`. Measured, original vs rebased: runner
  `--check-telemetry` exit 0 vs 0; `check-gplr53-embed-chunks.py --join` exit 0 vs 0;
  `analyze-gplr53-decay.py` exit 0 with `CHUNK0_SPIKE` both; `analyze-gplr53-gap.py` exit 2 both
  (its own refusal, identical before and after, so not rebase-sensitive). No python tool gates on
  an absolute `t_us` (grep: none), and `run_id` is only ever compared for equality. So the rebase
  moves the base without disturbing any existing gate - stated as measured for `E = 2,000,000 us`,
  which brackets the real offset (`t0_us` 2,974,883 in the recorded trace, captured at `init`).
- **Why `run_start` sits behind the marker block, measured in code:** `t0_us` is captured in
  `llama_layer_streaming::init` (`src/llama-layer-streaming.cpp:2194`, the same place the sink
  opens) and `run_start` is written later (`main.cpp:4362`), with 40 `pkg_member` events emitted
  in between. The markers flush at sink-open, so they precede that block - CONFIRMED by the run
  above (lines 1-3, `run_start` line 46).

No latency or throughput win is claimed anywhere in this block; A4's pre-registered band and its
saved scoring are untouched, and so are the retention state and T-GPLR-55's shapes.

#### A2 result (2026-09-22, host-only, from code and the recorded traces): spans need the loaded model, "earlier" is worth 66.4 ms, and `p1-span` as written is a no-op

**Where the spans originate (all measured from code, file and line).**

- `pkg_cache` is built in `llama_layer_streaming::init` (`src/llama-layer-streaming.cpp:1862`;
  the per-layer enumeration is `:1936-1950`) from `model.layers[il]` through the architecture
  adapter - NOT by the prebuild, and NOT from a file-offset table.
- `register_all_pkg_sources` (`:635-747`) derives each span from the members' MAPPED tensor
  addresses `m.src->data` (`:671`), page-rounds them and coalesces the touching groups
  (`:686-732`); it refuses unless `copy_src_registered` (`:641-644`) is set. Both the
  descriptors and the mapping therefore exist only after `llama_model_load_from_file`
  (`tools/layer-streaming/main.cpp:3637`) and `ls.init` (`:4106`).
- Runner order: load `:3637` -> `ls.init` `:4106` -> `exec_layers = boundary_layers` `:4332`
  -> `run_start` `:4362` -> prebuild `:4472` -> register `:4540`.

**Owner call 2 is answerable, and the draft's premise was wrong.** Nothing can be registered
before the load returns, so the ~2.9 s pre-`run_start` window is NOT available to hide a
commit: the loader's mapping addresses do not exist yet. The earliest legal point today is
right after `exec_layers` (`:4332`, the filter the loop reads), and moving the call there is
worth the measured `t0 -> register-loop-end` gap of **66.4 ms** - 3.4% of the 1.9 s commit.
`p1-early` is therefore a micro-optimisation, kept only as the negative control for "earlier
is enough".

**Span geometry (measured, 4k control):** 40 groups, one per logical layer, every one
single-layer (`merged = 0`) with exactly 3 members; 457,322,496 B (layer 0) to 526,528,512 B
(layer 39), mean 462,512,947 B, total 18,500,517,888 B; all 40 calls complete in **1,363 us**
(t_us 3,039,875 -> 3,041,238). So the whole 17.23 GiB is queued within 1.4 ms, and the first
consumer waits for the entire outstanding remainder: 18,500,517,888 B / 1.8143 s =
**9.70 GB/s**, which also equals the working-set growth rate measured in the pure-sleep arm
(2.00 GiB/200 ms). One group at that rate is **47.7 ms**.

**The draft's `p1-span` is a NO-OP (measured):** in layer-major the first pass visits all 40
layers (first `copy_submit` is layer 0 and the sweep covers 0..39), so "the first chunk's
packages" IS all 40 groups. The quantum that matters is one LAYER, not one chunk - hence the
merged `p1-pace` arm above.

**What `p1-pace` would entail (predicted, explicitly NOT measured):** register layer 0's
group before the pass and issue each later group a bounded distance before its first staging
copy, keeping pending unlocked bytes near one group. Predicted first stall ~47.7 ms instead
of 1,814.3 ms, with the total commit unchanged (1.907 s) but spread across the sweep; at 4k
the pass is only 3.56 s, so the spread may not fully hide, while 50k/200k passes (18.1 s /
134.7 s) have room.

**Questions that need GPU evidence - named as arms, never answered by reasoning:**

- Does a paced registration actually bound the wait, or does the driver serialize
  registrations device-wide so the next submission still blocks for the whole remainder? This
  is `p1-pace`'s whole discriminator: a ~48 ms first stall says paced; ~1.8 s says global.
- Does the lock's own traffic inflate staging? Compare `copy_complete` transfer medians
  against the control in the same session (31.4-31.5 ms over 240 copies today).
- Does a later group's registration complete before that layer is staged at the long shapes?
  A4-shape arms only, after A3.

**Escalated to the Planner, not an arm:** registering during the load window would need
mapping addresses before `load_from_file` returns, or a runner-owned mapping that the staging
path then uses. Both change where the copy source's addresses come from - an SDD question,
not a mode flag.

**Owner calls, with these findings folded in.** Call 2 is now narrow: since "earlier" is
66.4 ms, the real decision is whether the block may PACE registration during the timed sweep
(today registration happens exactly once, before the pass, and the timed region is
deliberately free of it) and whether the SDD's copy-source ownership is worth amending. Call
1 is unchanged and still gates the block's value. No latency or throughput win is claimed
anywhere in this result: the numbers are mechanism and geometry, and the win is what the arms
would have to demonstrate.

#### A3 implementation landed (2026-09-23, host-only, no GPU spent): `p1-pace` exists behind three default-off flags, its reading has its own refusing analyzer, and the ONE 4k hold is BLOCKED on the card

**What was asked.** A3: the mechanism that takes the registered-span commit off the first
prefill's critical path, implemented and validated, then measured in ONE 4k mutex hold with the
pre-registered Q gates.

**What landed (all of it default-off; nothing on the default path changes).**

| where | what |
|---|---|
| `src/llama-layer-streaming.{h,cpp}` | `set_pp_span_pace(mode, ahead)`, `register_paced_begin()`, `register_paced_ahead(il)`, and the read-only accounting `paced_reg_group_count()/bytes_total()/bytes_issued()`. The paced path derives the same page-rounded, address-coalesced spans the one-shot path registers, keeps them in GROUP order, pins the FIRST group before the pass and issues each later group from `stage_members()` when the sweep reaches `ahead` groups before it. Ownership convention is the one-shot path's (the group's first layer owns the registration, the rest are marked covered), so teardown reuses `unregister_all_pkg_sources()` unchanged. `register_all_pkg_sources()` itself is NOT modified: with `pp_span_pace == 0` nothing below the paced block is ever reached. |
| telemetry | three new event names, so no existing parser sees a new shape: `span_pace_init` (once per paced run: mode, ahead, groups, bytes_total, head-start bytes), `span_pace` (one per group issued), `span_pace_stage` (one per layer at its FIRST staging copy: `covered` plus how many groups were already issued), `span_pace_end` (the commit is complete). |
| `tools/layer-streaming/main.cpp` | `--pp-span-pace 0|1` (default 0), `--pp-span-pace-ahead 0..64` (default 1) and `--pp-register-site post-prebuild|early` (default post-prebuild = today's site) with help text. The flag-level preconditions are refused BEFORE the model load, so a misconfigured arm fails in seconds with no GPU and no trace: pace without `--copy-source mmap-registered`, pace combined with the early site, and `--pp-span-pace-ahead` without pace. The paced mode prints its own arming line (`PACED registration armed - X of Y B in N groups`), because a paced call returns only the FIRST group's bytes and the one-shot wording would understate the span; the off path's line is byte-identical to before. |
| `tools/check-gplr56-pace.py` | NEW, host-only, no GPU: the paced reading and its refusals (below). |
| `tools/layer-streaming/gen-gplr56-fixtures.py` (nested) | NEW: derives a paced trace from a RECORDED one, so the `span_pace_*` shapes are proved against real-shaped data (and a partial commit is derivable for the refusal path) instead of only against hand-made fixtures. |
| `tools/run-gplr56-a3-arm.ps1` | NEW: the three arms (`off` = control at today's site, `early` = earliest legal site, `pace`) inside ONE mutex hold, each writing `quantity: q_procstart_pass1end` into its own verdict, with the Q rule applied by `check-gplr56-q.py` (its single owner) and the relocation falsifier printed explicitly. |
| `tools/run-gplr56-session.ps1` | NEW: the one documented entry point (`-Session p1`, `-ValidateOnly`, `-PreflightOnly`, `-ArchiveOnly`), with the read-only busy preflight, evidence archiving and driver-output capture the T-GPLR-53 launcher established. |
| `tools/run-tests.ps1` | the pace analyzer's selftest is registered as `[t56-a3]`, so a missing or broken reading tool is a FAILED suite, not a skipped check. |

**Measured host evidence (no GPU, no mutex, no model).**

| check | result |
|---|---|
| `python tools/check-gplr56-pace.py --selftest` | **13/13 PASS**: a clean paced run evaluated; the same run against a control; a layer staged with `covered = 0` -> INTEGRITY FAIL naming the layer; an unfinished commit -> REFUSED; groups left unissued -> REFUSED; a NON-paced trace -> REFUSED; a group record with no `bytes` -> REFUSED; `groups = 0` -> REFUSED; the refusal prints no reading; the pacing lead reads back exactly as the configured distance for `ahead` 1 and 3, and as 0 for `ahead` 0 |
| the analyzer against the RECORDED A1 control trace | **exit 2, `p1-off` refused** - a real non-paced trace cannot be scored as paced |
| a PACED VARIANT of that same real trace, reproducible with `python third_party/llama.cpp/tools/layer-streaming/gen-gplr56-fixtures.py --real build/llama-cuda/gplr56-p1off-4k.jsonl --out build/tmp-pace/synth-real-pace.jsonl --groups 40 --ahead 1` (one-shot event replaced by `span_pace_init`, 40 groups issued across the pass-1 window, one coverage record per layer, `span_pace_end` closing it; every other line of the recorded trace untouched) | runner `--check-telemetry` **exit 0** (the new names are inert to the scheduler-invariant validator), analyzer **EXIT 0 EVALUATED** printing the whole reading: 40/40 groups, lead min 0 / median 1 / max 1 of the configured 1, 40 layers covered at their first copy, 240 `copy_complete` events with per-pass transfer medians, the pass-1 stall proxy, and a control comparison |
| the same fixture with only 20 of 40 groups issued (`--unfinished 20`, which also omits `span_pace_end`) -> `build/tmp-pace/synth-real-pace-partial.jsonl` | analyzer **exit 2** (`the paced commit is UNFINISHED - 20 of 40 groups`) - the refusal path is exercised on a real-shaped trace, not only on hand-made fixtures |
| flag refusals, host-only (`--model <nonexistent>` so no load can run) | `--pp-span-pace 2` -> exit 2; pace + `--copy-source pinned` -> exit 2; pace + `--pp-register-site early` -> exit 2; `--pp-span-pace-ahead 3` without pace -> exit 2; `--pp-register-site bogus` -> exit 2. All five refuse before any CUDA work |
| `tools/run-gplr56-a3-arm.ps1 -ValidateOnly` | exit 0, all three commands printed with the resolved flags (the dry run is the only arm artifact that exists) |
| `tools/run-gplr56-session.ps1 -Session p1 -ValidateOnly` | exit 0 |
| `tools/run-tests.ps1 -Suite host` | **SUCCESS 42/42 CTest**, digest `9E94BA192ABBA4C3`, nested head `3de21074f5b8`, `llama.dll A8DFF5ED2A451F83`; `[t56-a3]` green |
| `python tools/check-doc-sizes.py`, `python tools/resplit-gplr-plan.py --verify` | PASS |

**A MEASURED TRAP, recorded because it costs a suite failure to anyone who edits the engine
header:** adding members to `llama_layer_streaming` changes that type's layout, so every test
target linked against it must be rebuilt. The host suite's CTest step does not build, so with a
stale `test-layer-streaming-layout.exe` the test printed every assertion `ok` and then died in
teardown with `0xC0000409` (`Exit code 0xc0000409`), which reads exactly like a new defect in the
change. Rebuilding the test targets (`cmake --build build/llama-cuda --config Release`) fixed it -
measured, 3.65 s crash vs 0.08 s pass. A full build also fails on `test-unicode`
(`LNK2019 unicode_regex_split`), which is pre-existing and unrelated.

**The hold is BLOCKED on the shared card, and nothing was started.** The busy preflight refused
with `GPU busy - 2312 deepseek_v4.exe` at 11:27Z, and bounded re-probes across the next
ten minutes (11:31:57Z/11:32:05Z/11:32:13Z pid 18524; 11:35:59Z/11:36:09Z/11:36:20Z/11:36:30Z
pid 2376; 11:40:54Z/11:41:01Z/11:41:08Z pid 9368 - the pid moves every window, so the foreign
process is actively cycling) found `deepseek_v4.exe` in all ten samples. No driver was started, no mutex was acquired, no arm artifact was
written, and `-AllowBusy` was never used. This is the same class of blockage A1's first attempt
recorded, with the same handling.

**The arm is one command, recorded here ready for the next clean window:**
`powershell -File tools/run-gplr56-session.ps1 -Session p1` (it runs
`tools/run-gplr56-a3-arm.ps1` under the shared mutex with `build/llama-cuda/p1-driver-a3.out`
captured, preflights fast, archives the artifacts it is about to replace, and writes the arm
verdicts plus the derived `*-qrule` and `gplr56-a3-control-stability` records).

**What this does NOT prove.** No arm has run: the mechanism's effect on Q is UNMEASURED, the
`span_pace_*` shapes are confirmed only against a synthesised paced variant of a real trace (the
validator accepts them; no real emitter has produced them yet), and the early arm's trace-order
property (registration events ahead of `run_start`) is argued from the call-site sequence, not
observed. The pre-registered Q margin, the acceptance rule and the closure language are unchanged
by this subsection: A3 decides, and A4's 50k/200k holds stay owner-invoked one length at a time.

**A3 and A4 are implemented and host-validated; both owner-authorized holds have run. A fresh K=8 coact admission leg is also recorded below.**

### T-GPLR-58: wire the TG expert predictor into the row-set path - three pre-registerable arms (bar-0.9 safety, unbarred coact, fusion), judged on a NULL-CORRECTED, VRAM-normalized miss share under the exact-demand-priority invariant

Status: CLOSED (owner decision, 2026-09-25; evidence-complete; no predictor adoption) - the three owner calls were ANSWERED the same day and are recorded  A0 (the speculative-placement MECHANISM) is IMPLEMENTED and host-validated as of 2026-09-23 - see the A0 IMPLEMENTATION LANDED subsection below; A2 (the verdict surface) is IMPLEMENTED and host-validated the same day - see the A2 IMPLEMENTATION LANDED subsection below; A1 (the proposer families, the count-matched null, the speculative planner and its port comparator) is IMPLEMENTED and host-validated the same day - see the A1 IMPLEMENTATION LANDED subsection below; the A3/A4 holds are no longer gated on any unbuilt piece: the ENGINE-CONSUMES-THE-PROPOSER WIRING (mode 2) is IMPLEMENTED and host-validated as of 2026-09-23 - see the A1 WIRING (MODE 2) LANDED subsection below - A3 and A4 have now run; A4's seven-leg result is recorded below. A3 HAS SINCE RUN (2026-09-23, one hold): the arm proposed on 10,178 of 10,240 layer-stages and the planner placed ZERO, because the fill-room ceiling at K=2 is 0.00% - M1/M3 FAIL ON CAPACITY, the block's M3 negative fires, and the same run exposed (and fixed) a bring-up defect that had made every speculative mode inert - see the A3 SESSION RUN subsection below.
in the DECIDED section below; both mutex-held sessions were PRE-AUTHORIZED (owner chose the
"both up front" option, so A4 was not conditional on A3's share outcome). GPU time has now been spent on the recorded A3, A4 and fresh K=8 evidence; A0/A1/A2 and the
mode-2 wiring are IMPLEMENTED and host-validated. Nothing here re-opens T-GPLR-15 (closed DONE-falsified-by-evidence), T-GPLR-19 (closed
on the K ladder) or T-GPLR-26 (closed ADVISORY: no gate binds on (b)). The offline screening this contract
consumes is `docs/findings/predictor-screen-gplr.md` + `tools/predictor-screen-gplr.py`
(2026-09-23, host-only, no engine change). Every number quoted below is labeled with where it
was measured: **ours** (this repo, recorded), **colibri** (the DeepSeek-V4 baseline project,
read this pass), or **PROJECTION** (arithmetic on measured shares, never a result).

Dependencies: T-GPLR-19 (the row cache and the K ladder this rides on, and the engine/offline
(b) identity criterion 2 established there); T-GPLR-15 (the reuse ceiling and the explicit
"predictive hypothesis not falsified, the budgeted variant is" close - this block tests the
FIRST half of that sentence, not the second); T-GPLR-26 (gate 9's (b) reading is ADVISORY, so
this block must carry its own quantities rather than inherit that gate); T-GPLR-53/T-GPLR-56
(the pre-registration, falsifier and relocation discipline this contract copies). Not
blocking: T-GPLR-54, T-GPLR-55 (their shapes are untouched).
Commit: nested c861ab0d4 (A0 - speculative TG expert-row placement behind a default-off mode, the A2 engine datum it carries, tests/test-layer-streaming-tg-spec.cpp, the runner flag surface) + 7480fb5b9 (A1 - the four proposer families, the merge operators, the count-matched null and the speculative planner as pure functions, plus the --dump port surface) + e3627d5f3 (mode 2 - the engine CONSUMES the A1 proposer: arm/null source selection, the prop + arm_count telemetry in the spec event, the five runner refusals, and the scripted arm fixture) + 8b3ea428 (A2/A3 replayable admission telemetry, in-flight occupancy, bring-up reset ordering, and tests); outer 8e41a52 (uncovered-margin instrument + guard), 8c01a96 (A2 checker + guard), 10b0798 (A1 port comparator + guard), db1366e (fill-room analyzer + guard), 876ef83 (mode-2 flag/refusal checker + guard), 7a00d53 + db937e4 (pin bumps) + f8c5a48 (cache counterfactual, admission checker, A3/K-probe drivers, records, and pin bump); T-GPLR-07 b784e20 (semantic-QA checker/guard, separate slice). A3 and the fresh K=8 admission leg are committed evidence; the owner authorized the outer A4 launcher/roll-up and evidence commit after validation. Outer A4 tooling/evidence commit: `4e34b85`. This plan closure is docs-only after f8c5a48.

Closure (owner decision, 2026-09-25): T-GPLR-58 is closed as an evidence-complete negative/capacity result. The runtime and host framework remain available, but no predictor in this family is adopted and no additional A3/A4 work is authorized.

Goal. Decide, on TWO pre-authorized mutex-held sessions (owner, 2026-09-23), whether a TG expert predictor that stages
SPECULATIVE rows into the existing per-layer row cache produces a demand-hit gain that
**survives the null control** and **costs no extra VRAM** - i.e. whether a predictor at a
shallow retention depth K_low delivers what retention-only needs a deeper K_high to deliver -
with the exact-demand-priority invariant as a BLOCKING gate rather than a design aspiration.
Charter placement, stated honestly up front: this is a TG-phase residency task, so it CANNOT
move the program's first objective (measured in-repo, T-GPLR-26: "no depth changes prefill").
It is a goals-2-and-3 task - minimize VRAM, and hold up when VRAM is crowded - because the
row cache's private VRAM is bought per depth (546.8/1093.0/2185.4/3277.8/4370.1 MiB at
K=1/2/4/6/8, recorded) and a predictor that buys hit rate at K_low is a VRAM instrument, which
is exactly the instrument the charter names.

Scope. Opt-in, CUDA-only, DEFAULT OFF: a prediction mode on the existing row-set path (0 =
today's path, then the arm modes), a proposal budget B per (token+1, layer), a speculative
staging path on the copy backend with its OWN completion event, the invariant's counters, and
the telemetry the acceptance quantities need. Flag-level preconditions are refused BEFORE the
model load (the T-GPLR-53/T-GPLR-56 lesson: a refusal placed after the load is unobservable):
prediction without `--residency-tg`, B out of range, a bar/source combination the port does not
define, and prediction of any kind without the row cache actually fitted.
Non-goals: no change to routing, expert ids, sampling, or logits; no change to the reference
(exact-demand) fetch path's bytes, order or barriers; no change to K's fit rules or the K=8
default; no change to prefill (this is TG-only by construction, and the block must not claim
otherwise); NO SAMPLING REGIME IN THIS TASK (owner call 2, decided: the verdict runs greedy +
`--rep-penalty`; a sampler chain, temperature/top-p/seed and their determinism gates are a
separate capability, not an implied part of this one); no new dependency (colibri's operators
are ports, not links - their project stays
a read-only reference); no per-cell weight tuning (the portable vector is the only fusion
parameterization allowed); no speculative drafting, no trained sidecar, no learned head; no
re-tuning of any recorded gate.

DECIDED (owner, 2026-09-23) - the three calls that were open in this block's first draft,
answered before any implementation or GPU time, and binding on every leg below:

1. CHARTER SCOPE - YES, AND BOTH SESSIONS ARE PRE-AUTHORIZED ("both up front"). The mechanism
   cannot move prefill (measured: the row cache is a TG residency class), so the block's
   footing is goals 2 and 3 - VRAM per unit of TG hit rate, and the crowded-VRAM case. The
   owner authorized A3 AND A4 together rather than gating A4 on A3's share outcome, so A4 runs
   even if A3's share band fails; what HALTS A4 is a FAIL-INVARIANT leg or a refused leg
   (correctness, not performance). The offline margin is accepted as small, and a negative
   close remains written into the acceptance criteria.
2. DECODE REGIME - GREEDY + `--rep-penalty`, ON A DECLARED NON-ECHOING PROMPT. The runner has
   no sampler chain at all (measured: no `--temp`/`--top-p`/`--seed`; TG is argmax under
   `--greedy`, `main.cpp:2882`), but it does have `--rep-penalty F` (default 1 = off), documented
   as "applied identically at every greedy pick site" - the closest existing lever to colibri's
   `V4_SAMPLE_NOREPEAT=256` guard, which is what they used against the argmax repetition
   degeneracy (their T-opt-33: repeated tokens give "zero cross-token structure; history
   trivially wins"). PRE-REGISTERED: `--rep-penalty 1.15` on EVERY leg of both sessions, chosen
   as an experimental parameter and never tuned per arm, with the value recorded in every arm
   verdict; the 885-token non-echoing prompt is the corpus (the conservative one). CONSEQUENCE
   RECORDED: the recorded K-ladder anchors were taken at F=1 greedy, so they are CROSS-CHECKS
   ONLY and are not any band's denominator (M3 below is self-contained for exactly this
   reason). A sampler chain stays out of scope: if a positive result ever needs generalizing to
   a sampled regime, that is a new task with its own determinism gates.
3. NULL DEFINITION - THE COUNT-MATCHED, STRUCTURE-FREE NULL GATES; there is NO duplicate-
   incumbent leg. The null leg publishes the same NUMBER of ids per (token+1, layer) as the arm
   under test published at that row (a scalar read, nothing else), drawn by a fixed-seed rule
   that reads no ids, no demand and no router output; M2 still prints the pool-growth share
   beside it. A duplicate-incumbent (colibri-comparable) leg is deliberately NOT added: it
   measures added volume rather than signal, and colibri's own offline construction is
   therefore cited as the design input for M2's existence rather than reproduced as a gate.

TARGET OF RECORD (what "the predictor" means in this block, and nothing more). For each
(token t+1, layer L) the predictor publishes at most B expert ids, using ONLY information a
faithful online predictor could have at that moment: tokens < t complete (all layers), token t
at layers < L, and the caches/table state built from exactly that history. Faithfulness is not
a promise here - it is already a tested property of the ported harness
(`tools/predictor-screen-gplr.py --selftest`, PASS, plants known history/markov structure and
asserts it is detected while `predict()` precedes `observe()`).
Candidate families are the ported set: history (the incumbent, = what retention already does),
markov-1 (per-layer id-bigram), coact (normalized per-layer co-activation), coupling-copula
(cross-layer, lift-weighted). The bar is colibri's rank-geometric operator (publish rank r
while `bar^r >= 0.5`); fusion is their vote operator, with their weighted operator allowed only
with the portable weight vector `h1 m2 c1 k2`.

EXACT-DEMAND-PRIORITY INVARIANT (pre-registered, BLOCKING, each clause testable). The name is
colibri's failure, not a theory: their UPDATE101 root-caused a 36-slot thrash (read this pass)
in which "predicted fills are placed into the SAME per-layer hot window as exact demands", the
victim picker's barrier set was `in_use OR undrained OR fill_pending`, and a window of 30
in-use plus in-flight speculative fills left `pick_victim` nothing - 88 no-slot events, 84
exhausted, 11 demands failing past the pool. Their existing precedent for the fix class is
`V4_CUDA_PP_SPEC_GUARD` (default ON: refuse predicted proposals during PP) and their
T-opt-49a `V4_CUDA_TG_GEN_PROTECT` ("a predicted admission is refused when only protected
slots exist; exact demand keeps the full path"). Ported to THIS engine's structure
(`load_row_set`: `uniq` = the exact routed union, `fetch` = the non-resident part, async H2D
into the per-layer cache, D2D refresh of the whole union into the slot bundle, then a
host-block on `slot.ev_copy_ready` before the FFN may consume):

- **P1 - a demand never waits on a speculative copy.** Speculative copies are issued on the
  copy backend with their own completion event, and the demand's `ev_copy_ready` wait covers
  ONLY the demand's own copies. Falsifier: any demand leg whose stage wait includes a
  speculative copy's completion. Counter: `spec_waited_by_demand` (must be 0 always).
- **P2 - a speculative row is resident only AFTER its own copy completed.** `row_set_have[]`
  is set for a predicted id only on that id's own completion; a demand for a staged-but-not-
  ready id falls back to a normal demand fetch and is telemetried as such. A half-copied row is
  never served - the slot-sharing and generation guards that protect layer identity today
  (`row_set_owner` / `row_set_gen`) must cover the speculative path identically.
- **P3 - speculation is a HARD BARRIER below demand capacity.** Speculative admissions are
  refused when the layer's cache has no row free beyond what the current and next generation's
  demand can need; speculative rows are the FIRST victims and are never counted as "resident
  margin" for any coverage claim. Counter: `spec_refused_capacity`.
- **P4 - cancellation is exercised, not assumed.** Queued speculative fills are cancelled at
  the generation boundary and before any demand's stage. Colibri's UPDATE101 recorded
  `predictions_cancelled=0` in every leg - a cancel hook that never fires is UNTESTED, so this
  block requires `spec_cancelled > 0` in at least one leg of an arm (and names that leg in the
  record). Counter: `spec_cancelled`.
- **P5 - no displacement between sources.** A new source may not reduce the incumbent
  (history) proposal set: the arm's `history` proposals must equal the `off` leg's, field for
  field. The precedent is measured (colibri T-opt-27, logit-band: "those 78 band candidates
  displaced higher-recall HISTORY candidates (`history_predicted` 256->178), regressing
  coverage 1.56%->0.87%") - the arm that loses to a displaced incumbent is a FAIL, not a result.
- **P6 - the path is inert when off, and fails loudly when broken.** With the mode at 0 the
  runner output is byte-identical to today (`diff` clean) and the existing
  `test-layer-streaming-*` family stays green; a stale generation, a failed speculative copy, a
  missing completion event or an admission refusal is structured telemetry plus a refusal -
  never a silent fallback to another generation's rows (the SDD's existing rule).

PRE-REGISTERED ACCEPTANCE QUANTITIES (NULL-CORRECTED; the denominator lesson is ported, not
invented). Colibri's T-pf-06 was re-scoped by owner decision because its first band measured
the wrong denominator: their probe's hits were real, but "the live union already covers
43,625 of 43,904 demands = 99.36%", so the probe's entire unique contribution was bounded by a
279-demand margin and the original `unique_vs_live/hits >= 5%` band was unreachable by
construction. This block therefore states BOTH forms and gates on the margin form, and every
band below reads `null@B` as the leg per decision 3: COUNT-MATCHED PER ROW to the arm under
test (the arm's published count is the only thing the null reads), NOT a fixed budget - which
is what makes the bar-0.9 arm's null valid, since that arm publishes fewer rows than B by
design and a fixed-B null would not be count-matched to it.

- **The engine quantity, and why it is trustworthy: the share is the same number the offline
  tool predicts.** Per leg, `share = rows_reused / rows_demanded` with
  `rows_demanded = top_k x layers x TG tokens` (8 x 40 x T), from `rowset_round`/`rowset_fetch`.
  T-GPLR-19 criterion 2 measured that the runtime `rows_retained` equals the offline
  `hits_total` EXACTLY (52,930 at K=6 and 59,432 at K=8, 40/40 layers, 0 refusals), so the
  offline (b) table and this engine quantity are the same statistic on the same stream - the
  honest ladder is K=1 0.2468, K=2 0.3778, K=4 0.5355, K=6 0.6487, K=8 0.7283 (recorded).
- **M1 - STRUCTURE-ATTRIBUTABLE SHARE GAIN (PRIMARY, and the only band that can carry a
  PASS).** On the UNCOVERED MARGIN: `margin_share(arm) = unique_arm / uncovered`, where
  `uncovered` = demands the `off` leg did not serve as hits in the same session. Band:
  `margin_share(arm) > margin_share(null@B)` AND `share(arm) - share(null@B) >= the session's
  own control spread` (the spread is MEASURED in-session by a repeated `off` leg, so the margin
  is not invented). Both numbers print; a PASS requires both.
- **M2 - THE POOL-GROWTH SHARE, reported and NEVER credited.** `share(null@B) - share(off)` is
  printed beside M1. Offline this share is large and must not be confused with the finding:
  the null-source control lifted held history 22.7 -> 28.0 (ours; the structure-attributable
  part was +1.6pp honest / +6.3pp echo). Any arm whose M1 is within noise of zero FAILS
  regardless of how large M2 is.
- **M3 - VRAM-NORMALIZED EQUIVALENCE (the charter band), SELF-CONTAINED.** Same session,
  same prompt, same allowlist, same decode regime: `share(arm @ K=2) >= share(off @ K=4)` - the
  predictor at 1,093.0 MiB must reach what retention-only delivers at 2,185.4 MiB, with BOTH
  numbers coming from the SAME session's legs. The recorded honest K=4 share (0.5355) is kept
  as a CROSS-CHECK only, per decision 2: the recorded ladder was taken at F=1 greedy while
  these legs run at `--rep-penalty 1.15`, so it cannot be a denominator. Report the arm's own
  peak VRAM beside the pair, and `share(off @ K=6)` as a reported reference (not a band).
- **M4 - NO REGRESSION (non-inferiority; the safety arm's whole reason to exist).** For every
  arm, `share(arm) >= share(off) - <session spread>` and the demanded-copy transfer medians are
  inside the `off` leg's spread. The bar-0.9 arm's offline case is STRICT dominance on both
  axes (25.1/29.0 vs 24.6/24.7 honest, i.e. +0.5 recall at +4.3 ppb - ours, held-out), so a
  regression there is a port or invariant defect, not a finding.
- **M5 - PROPOSAL-VOLUME DISCIPLINE (ported from colibri's L5, and the same lesson as M2).**
  Total proposals + staged rows growth off/on <= +5%. Their T-pf-04 merge was falsified for
  exactly this: its proposals "rode on history's ~98k, were absorbed by drops, and were charged
  staging at history's own waste rate" (their +13%). An arm that wins only by adding volume
  FAILS this band and M1 together, by construction.
- **M6 - TIMING HONESTY (reported; gating only in the regression direction).** TG tok/s and
  `rowset_fetch` transfer medians per leg, against the same-session `off` leg. The honest prior
  is colibri's own end-to-end lesson (T-pf-06: their fusion advantage "shrank to noise on
  larger samples"), so this block does NOT promise a tok/s win: it requires that a share win
  does not COST tok/s beyond the session spread, and reports the direction.
- **M7 - REACHABILITY: the lead must be measured, not assumed.** A proposal can only become a
  hit if its copy lands before the demand. Colibri measured this quantity (`lead_avg=2.0,
  lead_max=2`, "the k=1 staging lead the horizon screen promised") and their S3 failure was a
  margin problem, not a lead problem. **RESTATED 2026-09-23 (Planner call, taken after A2
  measured the wording below vacuous - see the M7 RESTATEMENT section of this block).** M7 is
  now three parts, and only the first two are gates:
  - **M7a - the floor (invariant, integrity).** `lead_min >= 2` on any leg with a consumption.
    The reap ordering makes a lower lead impossible, so a lower value is wrong bookkeeping
    rather than a clever schedule: FAIL-INVARIANT, no share read, and the series HALTS.
  - **M7b - the late-fill rate (arm band, quality).** `matched_by_demand / (consumed +
    matched_by_demand) <= 0.5`, equivalently `matched <= consumed`. A fill a demand asked for
    while it was still PENDING is cancelled and the demand fetches its own copy, so the guess
    is paid for twice and returns nothing: the rate IS the arm's wasted-guess rate. Gated on
    `arm` legs ONLY - the count-matched `null` control draws its ids structure-free, so gating
    its targeting would fail the control by construction; its rate is reported instead, as
    pool-growth evidence.
  - **M7c - the distribution (reported, never gated).** `lead_rows` min/median/max/mean and the
    8-bucket histogram. Not a band: M7a already forces the floor, so the median follows from it
    and any median criterion is true of every leg that is not already FAIL-INVARIANT.
  - M7b is SCORED only with `consumed + matched >= 8` resolved fills; a thinner leg reports its
    rate UNSCORED (its other bands still score) and cannot satisfy M7's clause, so a session
    with no qualifying arm leg reports M7 UNMEASURED - never PASS. An arm whose guesses never
    land fails M7b even when it consumed nothing (rate 1.000), which the pre-restatement
    wording could not see at all.

Arms (each declares its own quantity BEFORE it runs; each writes `quantity:` into its own verdict):

- `off` - today's row-set path, no prediction (`--residency moe-expert-bank --residency-tg
  --residency-tg-retain K`). The control AND the M1/M2/M4 denominator, run TWICE per session
  (the second leg is the control spread).
- `null@B` - predictions on, proposals COUNT-MATCHED PER ROW and STRUCTURE-FREE: the null
  reads the ARM's published COUNT for each (token+1, layer) - a scalar - and publishes exactly
  that many ids from a fixed-seed xorshift over [0, n_expert), duplicates rejected, consulting
  no ids, no demand and no router output (asserted by construction and by a host test that
  feeds it a reordered router stream and requires an identical proposal sequence). Per-row
  count-matching is what makes the bar-0.9 arm's null valid: that arm publishes FEWER rows than
  B by design, so a fixed-B null would not be count-matched to it.
- `arm-bar09` - same-layer markov-1/coact ranked, published while `0.9^rank >= 0.5` (the SAFETY
  arm: offline it strictly dominates history on both axes on both streams).
- `arm-coact` - unbarred co-activation, top-B (the SIMPLE arm: offline +3.5pp recall at equal
  precision-per-byte on the honest stream).
- `arm-fusion` - vote fusion (param-free) over {history, markov-1, coact, coupling-copula} (the
  AMBITIOUS arm: offline +6.9/+6.9 over history honest, but only +1.6pp of it structure-
  attributable - so M1's margin form is the ONLY way this arm can PASS). The tuned weighted
  variant is allowed ONLY with the portable vector, with the tune/held split preserved.
- `retref@K4` - retention-only at the deeper depth, the SELF-CONTAINED M3 denominator (the
  same-session `share(off @ K=4)` the arm at K=2 must reach). `retref@K6` is an optional
  REPORTED reference only, never a band. No flag other than `--residency-tg-retain` differs.

Sessions (mutex-held, one per group, one length at a time, BOTH PRE-AUTHORIZED by the owner
decisions above), each at the declared short shape - the 885-token non-echoing prompt,
`--tg-tokens 256`, `--rep-penalty 1.15` on every leg:
- **A3/pilot**: `off@K2`, `off@K2` (the second leg IS the control spread), `null@8@K2`,
  `arm-coact@K2`, `retref@K4` (the self-contained M3 denominator; optionally `retref@K6` as a
  reported reference).
- **A4**: the same set with `arm-bar09@K2` and `arm-fusion@K2` in place of `arm-coact@K2`
  (its own null leg is count-matched per row to the arm under test).
50k/200k are NOT pre-authorized: this mechanism's own quantity does not scale with context the
way prefill's does, and the charter ranks the crowded case second - so a longer-context leg
would need its own owner call.

Known evidence (measured, with provenance - and the honest headroom, stated first):

- **The headroom is small and bounded.** Colibri's T-32e bounds the live UNCOVERED margin at
  ~6.7% of demands, and their live union covered 99.36% of demands in the saturated cell. Our
  own conservative stream is the same shape: history (the incumbent = what retention does)
  already holds 0.2458 (b) at K=1 and 0.7283 at K=8. A predictor's absolute room is therefore
  the region between the incumbent's share and 1.0 - which is why M1 gates on the margin and
  why a "big" raw hit count is explicitly not evidence.
- **The offline frontier is measured on OUR model** (`docs/findings/predictor-screen-gplr.md`,
  held-out where noted): unbarred coact 28.1/28.4 vs history 24.6/24.7 at B=8 honest (+3.5pp);
  bar-0.9 same-layer 25.1/29.0 (+0.5 recall, +4.3 ppb - the risk-free point); vote fusion
  29.6/29.6 with the null source at 28.0 (structure-attributable +1.6pp); echo stream roughly
  doubles every margin (42.6/43.0 coact) - the echo bias is the reason the honest stream is the
  conservative one. The port is anchored: history@B=8 reproduces the recorded (b) K=1 to ~1e-3
  on both vehicles.
- **The offline-to-live shrink is the prior, not a surprise.** Colibri T-pf-06's end-to-end A/B
  reduced their fusion advantage to noise at larger samples, and their strongest shadow result
  was a MARGIN result (probe caught 90/279 = 32.3% of the uncovered margin) - the exact shape
  M1 is written in.
- **The invariant has a measured failure precedent and a measured fix class** (their UPDATE101 +
  T-opt-49a, quoted above); their T-opt-36 (Markov-2) is a measured NULL in a related direction
  (emissions all fused onto already-admitted jobs, `bigram_matches=0` while `hist_matches=231`),
  which is why M1 requires UNIQUE-ON-MARGIN attribution rather than any match counter.
- **The decode regime is a declared limit, not a neutral choice** (decision 2): the runner is
  argmax-only, so `--rep-penalty 1.15` breaks repetition without sampling a distribution, and
  the recorded K-ladder anchors (F=1 greedy) are cross-checks only. A verdict here is evidence
  about a repetition-broken greedy demand stream - named in every record, never extended to a
  sampled regime.
- **Our engine's insertion point is known and single.** `load_row_set` owns the demand set
  (`uniq`), the non-resident part (`fetch`), the H2D, the D2D union refresh and the blocking
  wait; `row_set_have/owner/gen` are the residency bookkeeping. A speculative path that does not
  touch `uniq`, does not enter `fetch`, and marks `have[]` only on its own completion is
  additive by construction - which is what makes P1-P3 testable host-side.

Failure contract. Every clause above has a named counter, and a missing counter is a FAILED
check, not a skipped one. A3 and A4 are pre-authorized, so a FAILED share band in A3 does NOT
cancel A4; a FAIL-INVARIANT leg (P1/P4/M7a as specified below) or a refused leg HALTS the
series at that point, because the remaining legs would be measuring a broken mechanism. An arm that cannot show P1 (`spec_waited_by_demand == 0`), P4
(`spec_cancelled > 0` in at least one leg) or M7a (`lead_min >= 2` wherever a fill was
consumed) is reported FAIL-INVARIANT with no share read at all. M7b is an ARM BAND in the
M1/M4/M5 family: a FAILED late-fill rate fails that leg's M7 line and does NOT halt the
session or cancel A4. A direction flip between the offline table and the engine is a
faithfulness bug and ESCALATES (colibri's T-pf-03 rule: escalate, do not retry). A leg whose
`share` cannot be computed (missing `rowset_fetch`/`rowset_round`, missing `retain_k`, a
refusal in the leg) is refused with no number printed - never read as zero. An arm that changes
routing, greedy ids or the scheduler's staged bytes FAILS the identity gate and is reverted,
not tuned.

#### A2 IMPLEMENTATION LANDED (2026-09-23, host-only: no GPU, no mutex, no arm run, no model loaded)

**What landed.** The verdict surface this block judges arms with - `tools/check-tg-predict.py` - plus
the two engine additions its bands needed and could not read from any existing recording.

| piece | where | what it does |
|---|---|---|
| the checker | `tools/check-tg-predict.py` (new, registered as the `[tg-predict]` host guard) | reads one leg per `--leg ROLE=PATH` (or `--trace PATH`), computes share / margin / M2 / M4 / M4n / M3 / M5 / M7 and every P-counter, prints the per-leg evidence, or REFUSES the leg |
| M7's datum | `llama_ls_spec_lead` / `_lead_bucket` (pure, declared in the header) + `spec_issue_ord` + the `lead_hist` counters in `tel_spec` | stamps each fill with the stage that issued it and, when a DEMAND is later served from it, records the lead in demand stages of that layer: the min / median / max and the 8-bucket histogram the band asks for |
| the resolved mode, in the trace | `run_start` gains `row_spec_mode` / `row_spec_budget` / `row_spec_reserve`, present only when the mode is on | a leg is recognisable as an arm leg from the trace ALONE, never from a filename or a command history (the T-GPLR-48/T-GPLR-52 lesson); mode 0 traces are byte-unchanged |
| the fixture coverage | `tests/test-layer-streaming-tg-spec.cpp`, new M7 section (11 checks) | proves the lead arithmetic and the bucket shape the checker reads: the two-stage floor, the same-stage non-consumption, the overflow bucket, a full histogram with min/median/max, and that a matched-in-flight fill is NOT a consumption |

**The refusals, and why each is a refusal rather than a zero.** Exit 2 prints NO number for a refused
leg, because a leg that cannot be justified is not evidence: `no-retention` (no `rowset_init`),
`no-retain-k` (absent, or 0 = the reactive baseline, which has no per-layer residency and therefore
no margin), `no-rowset-fetch` (no demand stage), `arm-without-spec` (a leg declared arm/null with no
`spec` events at all), `spec-missing-keys` (including a `spec` that lacks `cancelled`, the schema the
block names), `cancel-hook-never-fired` (submitted > 0 with `cancelled == 0`: P4 is UNTESTED there, so
the leg is refused rather than scored - the acceptance fixture), `off-leg-with-speculation` (the run
resolved a mode while the leg is declared the no-prediction control), `accounting-negative`
(submitted below consumed + cancelled + live = double counting), `lead-inconsistent` (the histogram
and `consumed` disagree, or the sum is unreachable).

**Invariants the checker falsifies (a FAIL blocks reporting; it never authorizes).** P1
`waited_by_demand == 0`; P2 the fill accounting closes (a positive residual is a READY fill evicted
before any demand used it, and is printed as the uninstrumented quantity it is); P4 the named
cancellation points sum to `cancelled`; P6 an off leg carries no speculation; and a NEW one this
pass measured - **`lead_min >= 2`**, because the only hot-path reap runs at the END of a demand
stage, so a fill issued after stage t cannot serve a demand before stage t+2. A lead below that floor
is not a good schedule, it is wrong bookkeeping, and the checker fails the leg.

**Validation (all of it, no GPU).**
- `python tools/check-tg-predict.py --selftest` -> **PASS, 56 checks**: a scored off/arm/null session
  whose every band is asserted exactly (share, M2, M4, M4n, M7 median, the four invariant PASSes),
  then one fixture per refusal above asserting BOTH the exit code AND that no band number is printed,
  then two invariant-violation legs asserting exit 1 with the failing invariant NAMED.
- **Emitter <-> checker cross-check, no GPU**: the selftest reads `tel_spec()`'s own format string out
  of `src/llama-layer-streaming.cpp` and fails if a key the checker REQUIRES is not emitted - the
  failure mode that would otherwise refuse every arm leg at the first hold.
- **Cross-tool agreement on real recordings**: the five recorded retention legs
  (`gplr19-k{2,4,6,8}-ab/arm-B1.jsonl`, `gplr19-honest-k6-ab/arm-B1.jsonl`) score
  share 0.4326 / 0.6260 / 0.7141 / 0.7745 / 0.6461, identical to `tools/analyze-tg-margin.py`'s own
  booking of the same traces - two independently written readers, one recording.
- `ctest -C Release -R test-layer-streaming-tg-spec` -> **PASS** (36 + 11 checks);
  `run-tests.ps1 -Suite host` -> **SUCCESS 43/43** with the new guard running.

**MEASURED FINDING, recorded because it changes how M7 must be read (and NOT applied to the band).**
The engine's only hot-path reap runs at the end of a demand stage (the demand-side ordering is what
keeps P1 true), so a fill issued at the end of stage t first counts as READY at the end of stage t+1
and is consumable no earlier than stage t+2. **The lead floor is therefore 2, not 0**, and the
pre-registered M7 band as written ("lead median > 0") is VACUOUS for any leg that consumes a single
fill: the median cannot be 1. The informative reading of the same datum is the balance between
`consumed` and `matched_by_demand` - a guess aimed at the next stage of its layer ALWAYS arrives after
that stage was classified, so it pays only if the ids repeat across the two-stage gap (or if a deeper
cache would have held them anyway), and `matched` is exactly the count of fills whose own row was
demanded while they were still in flight. **PLANNER CALL TAKEN 2026-09-23**: M7 is restated as
M7a / M7b / M7c in the M7 RESTATEMENT section that follows - the median gate is DROPPED, because
`median >= 2` is vacuous for exactly the same structural reason as `median > 0`, the reap floor
becomes the invariant, and the late-fill rate becomes the arm band with a threshold chosen before
any measurement.

**What is NOT proven, stated plainly.** No real `spec` event has ever been emitted: the mechanism has
not executed against a real cache, so the engine-to-checker integration - a recorded instrumented leg
read back by this tool - remains unmeasured and is what the first A3 hold must show. A1's proposer and
the port comparator do not exist, so no arm can be run at all (mode 2 is refused). The session driver
the block's validation commands name (`tools/run-gplr58-session.ps1`) does not exist yet and is an A3
prerequisite. Every M-band above is therefore a TOOL result, not a predictor result: nothing here
says a predictor wins anything.

#### M7 RESTATEMENT - SETTLED 2026-09-23 (Planner call; contract text plus the A2 checker, no engine code path, no GPU, no arm run)

**The defect.** M7 as pre-registered read "an arm whose lead median is 0 is a scheduling defect and
FAILS before any share band is read". A2 measured that criterion VACUOUS: the engine's only hot-path
reap runs at the END of a demand stage, so a fill issued at the end of stage t cannot be READY before
the end of t+1 and cannot serve a demand before t+2. Every consumed fill therefore has lead >= 2, the
median of any non-empty set of consumed fills is >= 2, and "lead median > 0" can never fail. The
obvious repair is no repair at all: **`median >= 2` is equally vacuous**, because it restates exactly
the floor M7a now enforces directly, and it is true of every leg that is not already FAIL-INVARIANT.
Worse, the old wording was SILENT on the case M7 exists for: a leg that consumed NOTHING published no
M7 reading at all, so an arm whose guesses never land on time failed nothing.

**The settled rule: M7a / M7b / M7c.**
- **M7a - the floor (invariant, integrity).** `lead_min >= 2` on any leg with a consumption. A lower
  value means the ordinal bookkeeping is wrong, not that the schedule was clever: FAIL-INVARIANT, no
  share read, and the series HALTS. (Implemented in A2 as the `M7-floor` check; unchanged by this
  call.)
- **M7b - the late-fill rate (arm band, quality).** `matched_by_demand / (consumed +
  matched_by_demand) <= 0.5`, equivalently `matched <= consumed`. `matched_by_demand` counts fills a
  demand asked for while they were still PENDING: the fill is cancelled and the demand fetches its own
  copy, so the guess is paid for twice and returns nothing. The rate is therefore the arm's own
  wasted-guess rate, and the band asks one question - is the proposer aiming PAST the reap floor, or at
  the very next stage of its own layer, which can never be served in time? It is gated on `arm` legs
  ONLY: the count-matched `null` control draws its ids structure-free, so gating its targeting would
  fail the control by construction, and its rate is reported instead, as pool-growth evidence.
- **M7c - the distribution (reported, never gated).** `lead_rows` min/median/max/mean plus the
  8-bucket histogram. Reported because it is how a reader sees the schedule; not gated because M7a
  already forces the floor and the median follows from it.
- **The sample floor.** M7b is SCORED only with `consumed + matched >= 8` resolved fills. A thinner
  leg reports its rate as UNSCORED - its other bands still score - and cannot satisfy M7's clause, so
  a session in which no arm leg reaches the floor reports M7 **UNMEASURED**, never PASS.

**The threshold, and what it is not.** 0.5 - "more guesses land in time than arrive too late" - was
chosen A PRIORI from the mechanism's own ordering, not fitted to any observation, because no real
`spec` event has ever been emitted. It is FIXED for A3 and A4 and is NOT re-registered after a
session's data: an arm that fails it is failed, not the threshold re-tuned, and the only legitimate
change is a Planner call taken BEFORE the affected session runs. What the band can and cannot decide
is worth stating plainly: an arm whose guesses all arrive late cannot win the margin anyway (a late
fill is cancelled and the demand fetches its own copy), so M1/M4n expose that OUTCOME while M7b says
WHY - no room to win, versus targeting aimed inside the floor.

**Consequence for the failure contract.** M7a stays in the FAIL-INVARIANT list with P1 and P4. M7b
joins the M1/M4/M5 band family: a FAILED late-fill rate fails that leg's M7 line, does NOT halt a
pre-authorized session, and does NOT cancel A4. No other band, arm, session or length moves.

**Validation (host-only, no GPU, no mutex).** `python tools/check-tg-predict.py --selftest` covers all
three outcomes of M7b on fixtures: an arm at `matched 9 / consumed 3` (rate 0.750 -> FAIL), an arm at
`matched 4 / consumed 4` (rate 0.500 -> PASS, the boundary is inclusive) and an arm that consumed
NOTHING with `matched 9` (rate 1.000 -> FAIL, the case the old wording could not see at all) - and
asserts in every case that the verdict stays `scored`, because a band is not an invariant. The scored
session's own arm leg resolves 7 fills and is asserted UNSCORED, pinning the sample floor; the `null`
leg's rate is asserted reported-not-gated.

#### OWNERSHIP AND PLACEMENT - SETTLED 2026-09-23 (Planner call; contract text only, no code changed, no GPU, no arm run)

**The defect this settles.** As first drafted, A0 and A1 both owned the P1-P5 decision tests
(near-verbatim duplicates), and A1 said the predictor is "ported to the runner side". The second
was an architectural claim the code does not support; the first violated the rule this plan uses
everywhere else - one rule, one owner.

**What the code actually allows (measured 2026-09-23, source inspection only).** The demand - the
router's exact top-k ids for one (ubatch, layer) - is computed INSIDE the module
(`src/llama-layer-streaming.cpp` ~3625, where `union_ids` is assembled and handed to
`load_row_set(il, union_ids)`); the resident-row state is module members (`row_set_have`,
`row_set_owner`, `row_set_gen`, `row_set_ids`); the per-layer retention cache and its recency
policy are module-owned; and the module exposes NO callback, `std::function` or hook interface of
any kind (grep for `std::function` and `_hook` in `src/llama-layer-streaming.h`: zero hits). The
runner (`tools/layer-streaming/main.cpp`) configures the module before init/prebuild and reads its
telemetry; it never sees a per-layer routing decision, and it could only be handed one if the
module grew an inversion-of-control interface - a new architectural pattern in this codebase, and
not this block's business.

**The split, as settled.**

| concern | owner | where it lives | how it is proven |
|---|---|---|---|
| proposal budget + mode, flag surface, preconditions refused before the model load | A0 | `tools/layer-streaming/main.cpp` | each refusal fires before any model work |
| speculative staging: its own completion event, `row_set_have[]` marked only on that copy's completion, capacity-boundary refusal, cancellation points | A0 | `src/llama-layer-streaming.{h,cpp}` | engine-behavior fixtures with a SCRIPTED proposal sequence |
| P1-P6 counters and the per-id residency state the checker reads | A0 | same | a fixture issuing a spec copy that shows zero demand waits |
| proposer families, merge operators, the count-matched structure-free null | A1 | same module, PURE functions declared in the header | determinism, exact proposal sets, budget, null demand-blindness under a reordered stream |
| the planner's ordering, cancellation point and capacity-refusal POLICY decisions | A1 | same | pure-function tests: same inputs -> same decisions, no clock, no I/O |
| equivalence to the offline screen (the port must not be a re-implementation with its own opinion) | A1 | nested `--dump` mode + outer `tools/check-tg-predict-port.py` | per-row equality on the eight recorded legs; a mismatch blocks A1 |

**Rules that follow, and bind later steps.** (1) A0's tests may only assert engine behavior under
a scripted proposal sequence; a test that needs the proposer to DECIDE something belongs to A1.
(2) A1 may not add a runner flag and may not alter the staging path's semantics; needing a
mechanism the engine does not have is an A0 amendment, returned to the Planner. (3) A2 owns
neither: it reads what A0 and A1 emit. (4) A runner-side proposer is explicitly NOT part of this
block. (5) The engine-side placement is what makes P1-P3 testable host-side at all, which is why
it is a contract clause and not an implementation detail.

**What would overturn it.** A decision to run this predictor under a different harness that drives
layers itself would need the module to hand out routing, i.e. a callback interface; that is a
separate architectural task with its own contract, and it would have to argue against the measured
fact that no such interface exists in the module today.

Implementation steps (the holds are AUTHORIZED, not scheduled: A3/A4 run only after A0-A2 pass
their host validation, because a hold taken against an unvalidated spec path measures nothing):
- A0 (the ONLY source change; opt-in, default-off) - **owns the MECHANISM: enforcement, state
  transitions, events and counters, all inside the streaming module.** A prediction mode on the
  row-set path (0 = today's path), a proposal budget, and the speculative staging path - copies
  issued on the copy backend against the per-layer row cache with their OWN completion event,
  `row_set_have[]` marked for a predicted id only on that id's own completion, admission refused
  at the capacity boundary, cancellation at the generation boundary and before every demand
  stage, and the P1-P6 counters (`spec_waited_by_demand`, `spec_cancelled`,
  `spec_refused_capacity`, plus the per-id residency state the checker reads). Flag-level
  preconditions refused BEFORE the model load. Placement: the mechanism in
  `src/llama-layer-streaming.{h,cpp}`, the flag surface and reporting in
  `tools/layer-streaming/main.cpp` only - the module's own convention is that the runner sets
  policy BEFORE init/prebuild and the engine owns everything after it. Host validation, no GPU:
  engine-behavior fixtures driven by a SCRIPTED proposal sequence (no policy involved, so no test
  here depends on a proposer decision) - a demand never queues behind a spec fill; a
  staged-but-not-ready id falls back to a demand fetch; speculation is refused at the capacity
  boundary; the incumbent's proposal set is not reduced; cancellation fires; plus a fixture that
  issues a spec copy and observes its completion with no demand waiting on it, and the
  byte-identity check that mode 0 leaves the runner's output and the existing
  `test-layer-streaming-*` family exactly as they are. New nested target for these:
  `tests/test-layer-streaming-tg-spec.cpp` (CTest `test-layer-streaming-tg-spec`).
- A1 (host-only, no GPU, no mutex) - **owns the POLICY: pure functions over a demand history,
  plus their equivalence to the offline screen.** Placement: the proposer families (history /
  markov-1 / coact / coupling-copula), the two merge operators, the count-matched structure-free
  null and the speculative-fill PLANNER live in the same module as PURE functions declared in the
  header - the demand site, the routing ids and the cache state are all engine-side (see the
  OWNERSHIP AND PLACEMENT subsection below), so there is NO runner-side proposer and NO new
  runner flag in this step; if the policy needs a mechanism the engine does not have, that is an
  A0 amendment returned to the Planner. Validate: (i) policy unit tests - the proposer is
  deterministic, emits exactly the ids it claims, respects the budget, and the null proposer's
  sequence is invariant under a reordered router stream (demand-blind); (ii) pure-function tests
  of the planner (same inputs -> same issue order, cancellation point and capacity-refusal
  decision; no clock, no I/O); (iii) the replay equivalence: the module's per-row proposal
  sequence must EQUAL `tools/predictor-screen-gplr.py`'s on all eight recorded legs and the three
  arms' held-out tables must reproduce `docs/findings/predictor-screen-gplr.md` - a mismatch
  BLOCKS A1 and is enumerated, never tolerated as a tie-break.
  The reference is that tool AT A RECORDED REVISION (its file hash, written into the A1 result),
  because it is live code: taking the comparison against whatever the tool says later would let the
  target move after the fact, which is the same defect the pinned-binary rule exists to prevent.
  New nested target for (i)/(ii):
  `tests/test-layer-streaming-tg-predict.cpp` (CTest `test-layer-streaming-tg-predict`), which
  also gains a `--dump <trace.jsonl>` mode writing its proposal sequence in the same JSON-lines
  shape; (iii) is run by the outer host tool `tools/check-tg-predict-port.py`, registered as a
  host guard, so the nested test never depends on build-tree artifacts.
- A2 (host-only): `tools/check-tg-predict.py` - computes share, margin_share, M2, the
  per-leg VRAM and transfer medians, `lead_rows`, and every P-counter; REFUSES (exit 2, no
  numbers) a non-instrumented trace, a trace missing `spec_cancelled`, or a leg with no
  `retain_k`. `--selftest` covers each refusal with a fixture, including the "cancel hook never
  fires" leg that must be refused rather than scored.
- A3 (one mutex hold, pilot shape): `off` x2 + `null@8` + `arm-coact` + `retref@K4`, at K=2 for
  the prediction arms (the shallow depth the charter case needs) - decides M1/M3/M4/M5 for the
  simple arm, and produces the session's own control spread.
- A4 (one mutex hold, PRE-AUTHORIZED with A3 - it is not conditional on A3's share outcome;
  only a FAIL-INVARIANT or a refused leg halts it): `off` x2 + a per-row count-matched `null` +
  `arm-bar09` + `arm-fusion` at the same shape and depth, `retref@K4` included for M3.

Acceptance criteria (falsifiable):
- A0 (mechanism only - every claim below is an ENGINE behavior under a scripted proposal
  sequence, so no claim here depends on a proposer decision): mode 0 leaves the runner
  byte-identical to today and the existing host tests green; a demand never queues behind a
  speculative fill; a staged-but-not-ready id falls back to a demand fetch rather than being
  served; speculation is refused at the capacity boundary; the incumbent's proposal set is not
  reduced; cancellation fires at the generation boundary and before a demand stage; the spec-copy
  fixture shows a speculative copy completing with ZERO demand waits
  (`spec_waited_by_demand == 0`); and every flag precondition is refused before the model load.
- A1 (policy only): the proposer is deterministic and budget-exact (it emits exactly the ids it
  claims, never more than the budget); the null proposer's sequence is provably demand-blind
  (invariant under a reordered router stream); the planner's issue order / cancellation point /
  capacity-refusal decisions are pure and reproducible; and the port's per-row proposal sequence
  equals the offline screen's on all eight recorded legs with the three arms' held-out tables
  reproduced.
- A2: every refusal fires on its fixture and the tool prints NO number for a refused leg.
- A3/A4 (both pre-authorized; A4 runs regardless of A3's share result, and is halted only by
  an A3 FAIL-INVARIANT or a refused leg): for the arm under test, M1 holds (margin gain AND >= control spread), M4 holds (no
  regression beyond the spread), M5 holds (<= +5% volume), M7a holds (`lead_min >= 2` wherever
  a fill was consumed) and M7b holds (late-fill rate <= 0.5 with >= 8 resolved fills), and every
  P-counter reads as specified (`spec_waited_by_demand == 0`, `spec_cancelled > 0` in a named
  leg, `spec_refused_capacity` present).
- The charter band M3: at least one arm reaches `share >= 0.5355` at K=2 with its peak VRAM
  reported. If no arm does, the recorded finding is "predictors do not buy depth-equivalent
  hits at this headroom" and the block closes NEGATIVE - itself the decision-useful result,
  exactly as T-GPLR-56's contract handles its own negative.
- Default preservation: mode 0 (and every precondition-refused invocation) leaves the runner's
  output byte-identical to today, with the existing host tests and the semantic-QA smoke rung
  green on the changed binary.
- Identity: route digest and greedy ids identical across all legs of a session (the blocking
  gate every TG task in this repo carries).

Focused validation commands and expected results:
- `cmake --build build/llama-cuda --config Release --target llama-layer-streaming` -> green;
  record the exe/dll hashes (a new pair means the control must be re-recorded).
- `python tools/predictor-screen-gplr.py --selftest` -> PASS (recorded); the A1 replay mode
  reproduces the findings doc's held-out rows for all three arms.
- `python tools/check-tg-predict.py --selftest` -> PASS including the absent-counter and
  never-fired-cancellation refusals; `--trace <leg>` prints the full band set or refuses.
- `ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-tg-spec` -> PASS (the
  A0 engine-behavior fixtures under a scripted proposal sequence).
- `ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-tg-predict` -> PASS (the
  A1 policy unit tests), and `python tools/check-tg-predict-port.py --selftest` -> PASS with the
  port-equivalence verdict printed for each of the eight recorded legs (the `--dump` mode of the
  A1 target is what it compares against; CTest instance names in this build keep the `test-`
  prefix, measured with `ctest -N`).
- `powershell -File tools/run-gplr58-session.ps1 -Session p1 -ValidateOnly` -> exit 0 (all legs
  printed with resolved flags); `-PreflightOnly` -> exit 0 on a clean card or 125 within seconds
  naming the PID when busy; never `-AllowBusy`.
- `powershell -File tools/run-tests.ps1 -Suite host` -> PASS with the new guard registered;
  the T-GPLR-07 smoke rung -> check-telemetry OK, answer PASS, negative control failing as
  designed (runner change).
- `python tools/check-doc-sizes.py` and `python tools/resplit-gplr-plan.py --verify` -> PASS.

Required state/document updates: a `docs/currentstate.md` entry per executed session (with the
legs that did NOT run named rather than inferred); the SDD's mechanism section ONLY if a mode is
adopted (owner decision) - and then the invariant goes in as a contract clause, not as prose;
this block's status lines; the findings doc gains the engine-vs-offline comparison table.
AGENTS.md gains a gotcha only if a measured, transferable trap appears (the cancel-hook lesson
is a candidate once it is demonstrated here, not before).

Escalation condition: any need for a routing-visible or sampling-visible change (that is a
different task, and owner call 2 governs it); any arm where the demanded-copy transfer medians
regress beyond the session spread (bandwidth contention is a measurement, not an assumption);
any leg where `spec_cancelled == 0` while the arm claims a share win; any case where the
incumbent's proposal set is reduced by a new source (a design defect, never a result); any
direction flip between offline and engine numbers (faithfulness bug: escalate, do not retry);
and any result whose M1 gain is inside the control spread while M2 is large - that is the
"pool growth dressed as prediction" outcome and belongs back with the screening as its verdict.

#### A0 IMPLEMENTATION LANDED (2026-09-23, host-only: no GPU, no mutex, no arm run, no model loaded)

**What landed.** The mechanism of A0, default-off, on the RETENTION path (`row_set_retain`),
because that is the only path with a per-layer cache to speculate into - the slot-keyed reactive
path is refused with that reason rather than silently degraded.

| piece | where | what it does |
|---|---|---|
| the placement RULES, pure and host-testable | `llama_ls_spec_*` in `src/llama-layer-streaming.{h,cpp}` | `is_hit` (P2), `admit` (P3/P5), `free_pos`, `may_evict` / `victim_better` (P3 eviction law), `cancel_plan` (P4) |
| the engine's own placement state | `spec_state` / `spec_flag` / `spec_gen` / `spec_tick`, per layer and position | speculative state ONLY: `cache_expert` / `cache_recency` remain the sole owners of which row is where |
| the fills | `spec_issue_next` / `spec_issue_one` / `spec_reap` / `spec_cancel` / `spec_reset` | admit a scripted guess, issue its own H2D on the copy backend, record its OWN completion event, claim residency only at the reap, release cancelled positions only after their copy completed |
| the demand-side guards | inside `load_row_set`'s retention branch | a PENDING position is not a hit and not a victim; a demand that finds its row in flight fetches its own copy (P1/P2) and the fill is cancelled as a duplicate; an eviction clears the row's placement state with it |
| the counters and telemetry | `tel_spec`, one `spec` event per demand stage | submitted / ready / cancelled (+ per-point capacity, demand, generation) / refused_capacity / dup_refused / matched_by_demand / waited_by_demand / reap_wait_us / issued_bytes, plus mode, budget, reserve, cap, pending and ready_spec |
| the driver | `--residency-tg-spec <0|1|2>`, `--residency-tg-spec-budget N`, `--residency-tg-spec-script <path>` | mode 1 = scripted (`<layer> <ordinal> <expert>...` per line, the "speculate these ids for this layer's ordinal-th demand" contract); mode 2 is RESERVED for A1's proposer and refused |

**The reservations the design rests on, stated plainly.** (1) A demand's own row wait never
covers a fill: the reap runs only on the speculation path, after the demand's `ev_copy_ready`
wait, and `spec_in_demand_window` counts any reap that happens inside that window - so
`spec_waited_by_demand` is a falsifier, not an assumption. (2) The admission rule keeps `top_k`
positions FREE at all times, which is what makes (1) achievable: the demand's complement always
fits without reclaiming, or waiting on, a speculative position. (3) `ggml` exposes only
`synchronize` (no non-blocking query) in this tree, so a reap can block; the honest measurable
for "did the guess arrive early" is therefore `reap_wait_us` per stage, not a per-fill latency,
and A1's real proposer inherits that limitation.

**Teardown owns an in-flight fill.** A demand stage always synchronizes its own copies before
returning, so no copy can be outstanding at teardown - but a speculative fill has no such owner,
and `end_row_cache()` frees the per-layer cache buffers the fill is writing into. `spec_reset()`
therefore DRAINS every fill event before freeing it. This is the one place where waiting on a
speculative copy is correct, and it is not a demand: `spec_waited_by_demand` counts only reaps
inside a demand window, and the drain runs after the last one.

**Host validation (all of it, no GPU).**
- `ctest -C Release -R test-layer-streaming-tg-spec` -> **PASS** (new nested target,
  `tests/test-layer-streaming-tg-spec.cpp`): 36 scripted-sequence checks covering P2 (pending is
  never a hit, residence only on completion), P3 (the reserve barrier, the budget, no
  displacement, spec-first eviction), P4 (oldest-first cancellation, READY never in the plan),
  P6 (null states = today's behaviour), plus a three-step guess/demand walk with the hit, the
  matched-but-pending and the wrong-guess outcomes counted.
- **The mode-0 differential PASSES**: the pre-A0 retention algorithm is kept verbatim in the test
  as the oracle and agrees with the new decision functions over **1200 random windows** across
  three capacities - identical hits, fetch sets, victim choices, eviction counts, placement
  positions and resulting cache. This is the strongest host-side statement available for P6; the
  run-level byte-identity check still belongs to a GPU arm.
- **The differential found a real defect while being written**: evicting a speculative row left
  its placement state READY on a position whose row was gone, which would have made the demand's
  free-position search skip a free position and shrink its guaranteed capacity. Fixed in the code
  (an eviction clears the row's placement state) and the free-position predicate was simplified to
  key on physical occupancy.
- `run-tests.ps1 -Suite host` -> **SUCCESS, 43/43** (the new target is registered in the suite's
  include pattern; the whole family was rebuilt after the header change, per the recorded
  stale-layout trap). Parse-time flag refusals verified against the built runner
  (`--residency-tg-spec 3` -> exit 2 before any model work); the remaining flag-precondition
  refusals sit with the other residency preconditions, i.e. after the architecture metadata read
  and before any model load.

**What is NOT proven, and stays with A1/A3.** A1's proposers do not exist (mode 2 is refused);
the mechanism has never executed against a real CUDA cache (no fill has ever been issued, no
event ever recorded), so every wiring claim here is a code-level claim with a host proof of the
RULES behind it; the run-level byte-identity of mode 0 on runner output, and every M-band, remain
unmeasured. A2's checker does not exist yet either, so the `spec` event's shape is defined by the
emitter and not yet read back by any tool.

#### BOUNDING MEASUREMENT (2026-09-23, host-only, no GPU, no arm run): the incumbent's uncovered margin and what an id-stream predictor can reach - the ceiling M1 and M3 are read inside

**What was asked.** Bound the room every T-GPLR-58 arm must fit inside, before any hold: how
much of the demand stream the incumbent row cache does NOT already serve, and how much of that
room is reachable BY ANY predictor built from observed expert ids. Tool:
`tools/analyze-tg-margin.py` (selftest 23 cases; registered as the `[tg-margin]` host guard);
measured on eight recorded retention legs, no GPU and no mutex.

**The retention semantics had to be identified first, and the documented reading is wrong.**
Rebuilding each trace's OWN `n_rows_resident` from `union_ids`, per layer, under four candidate
models (deviation of the rebuilt hit count from the engine's booking). Reproduce with
`python tools/analyze-tg-margin.py --identify gplr19-k2-ab/arm-B1.jsonl
gplr19-honest-k6-ab/arm-B1.jsonl qa-depth-k4.jsonl` (from `build/llama-cuda`):

| model | gplr19 honest K=6 | qa-depth K=4 | gplr19 echoed K=2 |
|---|---|---|---|
| last K windows, per LAYER | -16.92% | -17.15% | -7.53% |
| last K windows, per SLOT | -61.90% | -69.44% | -86.60% |
| capacity-LRU, per SLOT | -56.54% | -68.58% | -85.86% |
| **capacity-LRU, per LAYER, `K x top_k` rows** | **-0.21%** | **-0.17%** | **-1.14%** |

So the cache is PER LAYER (per-slot sharing is decisively excluded under BOTH candidate models,
even though `slot == layer % n_slots` holds in every one of these runs and a slot demonstrably
carries more than one layer) and its semantics are a capacity-LRU over `K x top_k` ROWS: the
effective retention horizon EXCEEDS K tokens whenever ids repeat inside the window, because a
row is evicted only under capacity pressure, never merely for age. The incumbent is therefore
stronger than the "last K windows" wording implies, and a predictor's room is smaller than a
naive reading of K suggests. A residual of 0.06-1.14% remains unexplained (eviction tie-breaking
or a first-window special case); the tool prints it and refuses any trace deviating more than
`--max-dev` (default 2%). The four-model comparison is a first-class mode of the tool rather
than a side experiment, because a margin is only a measurement if the cache model behind it
reproduces the run it is computed from.

**The margin, and its decomposition (the ceiling).** `share` = engine hits / demanded; each
margin row is classified once as C1/C2/C3/C4 (see the tool's docstring for the definitions).
Legs: echoed = `gplr19-k{2,4,6,8}-ab/arm-B1.jsonl`, honest =
`gplr19-honest-k6-ab/arm-B1.jsonl` and `gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl`,
qa-depth = `qa-depth-k4.jsonl` / `qa-depth-k8.jsonl` (all under `build/llama-cuda`):

| trace | K | demanded | share | margin | C1 same token, earlier layer | C2 same layer, >K .. <=W | C3 seen elsewhere earlier | C4 FIRST VISIT (unreachable) | ceiling for id streams |
|---|---|---|---|---|---|---|---|---|---|
| echoed | 2 | 81,920 | 0.4326 | 0.5674 | 43.4% | 49.1% | 6.9% | 0.5% | 0.9969 |
| echoed | 4 | 81,920 | 0.6260 | 0.3740 | 42.5% | 46.1% | 10.6% | 0.8% | 0.9969 |
| echoed | 6 | 81,920 | 0.7141 | 0.2859 | 42.0% | 43.1% | 13.8% | 1.1% | 0.9969 |
| echoed | 8 | 81,920 | 0.7745 | 0.2255 | 40.9% | 40.2% | 17.5% | 1.4% | 0.9969 |
| honest | 6 | 81,920 | 0.6461 | 0.3539 | 41.2% | 38.7% | 19.2% | 0.9% | 0.9969 |
| honest | 8 | 81,920 | 0.7255 | 0.2745 | 40.7% | 33.3% | 24.8% | 1.1% | 0.9969 |
| qa-depth | 4 | 30,720 | 0.6319 | 0.3681 | 42.7% | 34.5% | 20.5% | 2.3% | 0.9917 |
| qa-depth | 8 | 30,720 | 0.7521 | 0.2479 | 42.1% | 24.0% | 30.5% | 3.4% | 0.9917 |

The hit numerator is the recorded (b) ladder's own (52,930 at honest K=6, 59,432 at K=8); this
tool's denominator additionally counts the warm-up window (81,920 rows = 256 tokens x 40 layers x
top_k 8 against the record's 81,600 = 255 tokens), so the honest K=6 share here reads 0.6461
where the record reads 0.6487 - the same residency, a slightly colder denominator, and the
reason M3 compares within a session instead of against the record.

**Three readings that change how the arms should be read (none of them retunes a band).**

1. **There is no information wall worth the name: C4 is 0.5-3.4% of the margin (0.3-0.8% of
   demands).** So an id-stream predictor is NOT limited by rows it has never seen - it is limited
   by PRECISION: what the offline screen actually achieves is 24.6% (history) to 28.1% (coact) of
   demands at B=8, against a margin of 22-57% of demands. M1's fate therefore rests on proposal
   precision inside a margin that is only ~2-3 rows per (token, layer) at the depths that matter
   (2.945 at qa K=4, 1.983 at qa K=8), while an arm may propose B=8 - which is why M5's volume
   discipline (`<= +5%`) is load-bearing and not a formality.
2. **The margin is not one lever but three, and they need different mechanisms**: C1 (~41-43%
   everywhere, stable across K and vehicles) is the cross-layer region and it is reachable only
   with an INTRA-TOKEN layer lead (a few layers ahead, not a token ahead) - it is the coupling
   family's target, and an arm that stages coupling proposals a full token early is aiming at the
   wrong horizon. C2 (24-49%, shrinking as K rises) is precisely the region a DEEPER CACHE would
   also capture, which is the VRAM-vs-prediction trade M3 measures. C3 (7-31%, growing with K and
   at longer context) is what only a learned cross-layer model reaches.
3. **The incumbent is stronger than documented** (capacity-LRU, reading above), so every arm's
   baseline is higher than the "last K windows" model would predict: at honest K=6 the incumbent
   already serves 64.6% of demands, and honest K=8 serves 72.6% - the M3 comparison (arm at K=2
   against retention-only at K=4 same-session) is therefore a comparison against a deeper-than-K
   baseline, which is the harder, correct reading.

**Status of this subsection:** supporting evidence for the bands, not an acceptance. It changes
no pre-registered quantity, re-scores no recorded verdict, and was taken with no GPU. What it
does NOT establish: nothing about any predictor's actual effect (no arm has run), and nothing
about shapes beyond these eight legs (the 50k/200k TG shapes are not covered here).

#### A1 IMPLEMENTATION LANDED (2026-09-23, host-only: no GPU, no mutex, no arm run, no model loaded)

**What landed.** The POLICY half of the predictor block, as pure functions in the same module,
plus the comparator that proves it faithful to the offline screen. The reference is PINNED BY FILE
HASH, as this block's A1 step requires, because that tool is live code:
`tools/predictor-screen-gplr.py`
`sha256 81c9a4afa5333bad1599d774b6488c33f9d97156c0a3cc8a826267e7150caef2`, findings doc
`sha256 f414fc8f53b9456f...`.

| piece | where | what it does |
|---|---|---|
| the proposer families | `src/llama-layer-streaming.{h,cpp}`, pure `llama_ls_pred_*` | history / markov-1 / coact / coupling-copula - the screen's four - over an EXPLICIT caller-owned state (`llama_ls_pred_state`, allocated per family mask), so a test can hold two copies of one stream and prove determinism and demand-blindness |
| the arms | same | `llama_ls_pred_arm_predict`: `coact` (unbarred), `bar09` = `Barred(Markov1, 0.9)`, `fusion` = vote fusion over the four sources (param-free, equal weights, in-source dedup, rank-decay tiebreak) |
| the count-matched null | same (`llama_ls_pred_null_seed` / `_null_fill`) | xorshift32 over [0, n_expert) publishing exactly the ARM's count for the row, duplicates rejected, consulting no ids, no demand and no router output |
| the planner and the cancellation point | same (`llama_ls_spec_plan`, `llama_ls_spec_cancel_policy`) | the issue order, the stop reason (budget / pending cap / reserve / no free), what it declined and why, and when a cancel is needed ahead of a demand |
| the policy tests + the dump | `tests/test-layer-streaming-tg-predict.cpp` (new, CTest `test-layer-streaming-tg-predict`, 28 checks) | determinism, budget-exactness, range/uniqueness, no-peeking, null demand-blindness, planner purity AND replay equivalence against `llama_ls_spec_admit`, the cancellation policy; `--dump <trace.jsonl>` writes the per-row proposal sequence as JSON Lines |
| the port comparator | `tools/check-tg-predict-port.py` (new, registered as the `[tg-predict-port]` host guard) | IMPORTS the screen, drives its own classes row by row in the trace's order, and requires per-row SEQUENCE equality for the four families and the three arms; `--tables` re-derives the findings doc's recorded rows from that same reference |

**The port is anchored row by row (measured 2026-09-23).** On the eight recorded legs -
`gplr19-k{2,4,6,8}-ab/arm-B1.jsonl`, `gplr19-honest-k6-ab/arm-B1.jsonl`,
`gplr19-honest-k6-ab/gplr19-honest-k8-ab/arm-B1.jsonl`, `qa-depth-k4.jsonl`, `qa-depth-k8.jsonl` -
the module's per-row proposal sequence EQUALS the screen's for all four families and all three
arms: **483,840 row comparisons, zero mismatches**, with the null count-exact against its matched
arm on every row. The comparator also verifies the doc's own "same stream" claim from the demand
rows themselves: the four echo legs share one stream (`1c99b7fc6cf1`), the two honest legs another
(`da834f794b31`), the two qa legs a third (`57fb9aaa6d9e`).

**The doc's recorded rows are re-derived, not restated.** `--tables` computes each row from the
reference and requires the findings doc to state it: echo 30.8/30.9 history, 39.5/39.9 markov-1,
42.6/43.0 coact, 36.1/41.7 bar09; honest 24.6/24.7, 27.4/27.7, 28.1/28.4, 25.1/29.0; qa
33.3/33.7, 35.5/36.4, 36.8/37.7 - all reproduced to the doc's tenth.

**A measured artifact that had to be DECIDED, not smoothed over.** The screen observes
`list(frozenset(ids))`, so ITS history family carries CPython's set-iteration order while the
engine sees `union_ids` in the trace's order. Measured on three streams
(`build/tmp-tg-a1/orderexp.py`, seven candidates each): the order changes ONLY the fusion arm
(<= 0.09pp) and can move coupling by one rounded tenth (0.02pp echo, 0.01pp qa), while every
family and bar number the doc records reproduces EXACTLY under either order. So the comparator
drives the reference in the TRACE's order - the order the engine actually has - and holds those
two rows to the doc's tenth plus that measured artifact. That choice is recorded here because the
alternative (making the port imitate CPython's set order) would have been a port of an
implementation detail, not of the screen's method.

**What is NOT proven, stated plainly.** (1) The engine does not CONSUME the proposer yet: mode 2
is still reserved and refused, so no arm can run and A3 cannot start - that wiring is the
remaining A1 work and the next step in this block. **[SUPERSEDED the same day: that wiring
LANDED - see the A1 WIRING (MODE 2) LANDED subsection below, which this clause predates; the
remaining work is the hold RUN, not a code gap.]** (2) No real `spec` event has ever been emitted,
so nothing here says any arm wins anything; M7b's 0.5 and every M-band remain pre-registrations.
(3) The doc's fusion row (29.6/29.6 held-out vote fusion) is a tune/held quantity over a 4-source
pool and is deliberately NOT gated by `--tables`; the port's equivalence is exact because both
sides are driven identically. (4) `bar09` is implemented over MARKOV-1 because this block quotes
the markov-1@bar0.9 row (25.1/29.0) as the safety point; coact@bar0.9 (25.8/29.8) is the stronger
barred row and switching the arm to it is a Planner call, not a port detail.

**Validation (all of it host-only, no GPU, no mutex).** `ctest -C Release -R
test-layer-streaming-tg-predict` PASS, 28 checks; `python tools/check-tg-predict-port.py
--selftest` PASS, 13 checks - including the NEGATIVE control (one flipped id is CAUGHT, and a
dropped row, a short null row and a dropped candidate are each caught) and the refusals (a trace
with no `rowset_init`, a leg out of `(ubatch, layer)` order, a missing dump binary); `python
tools/check-tg-predict-port.py --tables` EQUAL; `run-tests.ps1 -Suite host` SUCCESS 44/44
(revision digest `3FE379D5EE294043`, llama.dll `09437EE03CE61125`). Committed as nested 7480fb5b9 + outer 10b0798.
#### A1 WIRING (MODE 2) LANDED (2026-09-23, host-only: no GPU, no mutex, no arm run, no model loaded)

**What landed.** Mode 2 is IMPLEMENTED. Until now the policy existed and was proven row-equal
to the offline screen, but nothing consumed it - mode 2 was reserved and refused - which is
exactly why A3 could not start. The engine now asks the arm itself for the rows to guess.

| piece | where | what it does |
|---|---|---|
| the orchestration | `src/llama-layer-streaming.{h,cpp}` (`llama_ls_spec_propose`, pure) | one call per layer-stage: OBSERVE the row the demand just served -> publish from the arm (or from the null, count-matched to it) -> PLAN under the budget/pending/reserve rules -> ISSUE. It is the SAME downstream path mode 1 lands in: the two modes differ only in where the proposals come from |
| the bring-up and members | same (`set_row_set_spec_arm`, arm / null / proposer state, the per-stage counters) | the arm's learning state is allocated BEFORE `begin_row_cache()`, so every refusal is loud before a context or graph exists |
| the telemetry | the `spec` event and `run_start` | mode 2 adds `"prop"` (what the SOURCE published this stage) and `"arm_count"` (the arm's own count the null was matched to, emitted so the count match is checkable from the trace alone). Both are guarded by `row_set_spec_mode == 2`, so a mode-0/1 event stays byte-identical to every recording made before this change. `run_start` echoes the resolved arm and `row_spec_null` |
| the runner | `tools/layer-streaming/main.cpp` | `--residency-tg-spec 2` (now implemented, was a refusal), `--residency-tg-spec-arm <coact|bar09|fusion>`, `--residency-tg-spec-null <0|1>`, and FIVE new refusals: mode 2 with no arm; mode 2 with a script (mode 1's source); an arm outside mode 2; a null outside mode 2; an unknown arm name |
| the scripted arm leg | `tests/test-layer-streaming-tg-spec.cpp` +20 checks (68 total) | the host fixture that drives the wiring: arm proposes on a periodic stream, the plan's accounting is real (already-resident proposals skipped, the cache refuses the rest), a saturated cache still PROPOSES but issues nothing, a cache with room DOES issue and holds the reserve back, no peeking, determinism, and the null publishing exactly the arm's count with ids that are not the arm's |
| the refusal contract checker | `tools/check-gplr58-mode2.py` (new, `[tg58-mode2]` host guard) | runs the REAL binary on the unsupported-arch fixture and asserts the exact exit code AND message for all ten cases; its selftest is hermetic stub runners |
| the fill-room analyzer | `tools/analyze-gplr58-fill-room.py` (new, `[tg58-fill-room]` host guard) | bounds the opportunity BEFORE a mutex hold is spent: how many layer-stages ever leave a free position beyond the demand reserve |

**Why the gates are testable without a GPU, stated once.** Every mode-2 gate sits AFTER the
host-only metadata read (`llama_ls_model_arch`) and BEFORE `llama_backend_init`, which this
runner defers deliberately. So the unsupported-arch fixture (1,280 bytes) drives the whole flag
surface with no CUDA, no model load and no mutex - verified by `nvidia-smi --query-compute-apps`
being empty before and after a full checker run. A legal shape is proven ACCEPTED by REACHING the
next stable refusal (the adapter's architecture gate), not by a green run.

**The finding that decides how the arms must be read (measured 2026-09-23, host-only).** A fill is
admitted only while its layer has a free position beyond the demand reserve, and the demand path
fills the cache to `cap = min(n_expert, K * top_k)` and evicts to place every later miss - so
every post-warm-up miss is a net-zero swap. Rebuilt from the eight recorded retention legs
(capacity-LRU over the demanded unions, the retention model the bounding pass identified):

| leg | K | cap | stages | admittable | % | mean free | full by stage |
|---|---|---|---|---|---|---|---|
| `gplr19-k2-ab` | 2 | 16 | 10240 | 0 | 0.00 | 0.05 | 4 |
| `gplr19-k4-ab` | 4 | 32 | 10240 | 181 | 1.77 | 0.35 | 12 |
| `gplr19-k6-ab` | 6 | 48 | 10240 | 413 | 4.03 | 1.01 | 20 |
| `gplr19-k8-ab` | 8 | 64 | 10240 | 678 | 6.62 | 2.09 | 30 |
| `gplr19-honest-k6-ab` | 6 | 48 | 10240 | 503 | 4.91 | 1.18 | 29 |
| `gplr19-honest-k8-ab` | 8 | 64 | 10240 | 771 | 7.53 | 2.41 | 37 |
| `qa-depth-k4.jsonl` | 4 | 32 | 3840 | 181 | 4.71 | 0.94 | 12 |
| `qa-depth-k8.jsonl` | 8 | 64 | 3840 | 678 | 17.66 | 5.58 | 30 |

**3,405 of 69,120 layer-stages (4.93%) ever leave room for a fill, and every one of them is in
the layer's WARM-UP** (the cache reaches full occupancy by stage ~K and keeps it). **PLANNER CALL
(recorded here, not left as a footnote):** a predictor's demand-hit gain therefore has a ceiling
set by CAPACITY, not by the arm's precision - and an arm run that reports little or no gain has
falsified nothing about prediction. Two consequences bind A3/A4: (1) every arm leg's session must
report the fill-room datum alongside `share`/`margin_share`, and the tool that computes it now
exists; (2) the pre-registered M-bands keep their meaning, but a null result must be read against
this ceiling before any arm is called rejected. The `null` control is unaffected - it is a
PROPOSAL-source control at the same budgets, not a claim about capacity.

**What is NOT proven, stated plainly.** (1) No real `spec` event has been emitted from a GPU run:
the wiring is exercised host-only, by a scripted demand leg and by fixtures, so nothing here says
an arm wins anything or that the mechanism behaves under a real router. (2) The fill-room table is
reconstructed from the INCUMBENT's recorded occupancy (the admission decision read after each
stage's placement); it is a bound on the opportunity, not a measurement of a speculative run,
because no speculative run exists yet. (3) The starvation the fixture found (a saturated cache
admits nothing) is a property of the incumbent's capacity rule, so mode 2 cannot lift it - only an
admitted fill in the warm-up window can have any effect at all.

**Validation (all of it host-only, no GPU, no mutex).** Both nested tests PASS
(`test-layer-streaming-tg-spec` 68 checks, `test-layer-streaming-tg-predict` 28 checks);
`python tools/check-gplr58-mode2.py` PASS 10/10 cases and `--selftest` PASS 7/7 - the selftest's
negative controls include a wrong exit code, a wrong message, a SILENT exit scored as agreement,
and a broken binary's model-missing exit 1 which must never read as a refusal match;
`python tools/analyze-gplr58-fill-room.py --selftest` PASS 8 checks;
`python tools/check-tg-predict-port.py` still **EQUAL on the wired binary** - 8 legs, 483,840 row
comparisons, so the mode-2 work did not perturb the policy; `run-tests.ps1 -Suite host` SUCCESS
44/44 with both new guards registered (revision digest `4D08CFCD186C9074`, llama.dll
`38386D5B495877F5`, record `20260923T171456Z-host`).
Committed as a task-split series: nested `e3627d5f3` (the wiring, its scripted arm fixture and the
runner flags), then outer `db1366e` (fill-room analyzer + guard), `876ef83` (mode-2 flag/refusal
checker + guard), `db937e4` (pin bump to `e3627d5f3`), and the docs commit carrying these records.

#### A3 SESSION RUN (2026-09-23, ONE mutex hold, five legs, GPU-clean): the arm PROPOSED on 10,178 of 10,240 layer-stages and the planner PLACED ZERO - the fill-room ceiling at K=2 is 0.00% - and the first real run also exposed a BRING-UP DEFECT that had made every speculative mode inert

**What ran.** `tools/run-gplr58-session.ps1` (new: one hold, the contract's five legs in order, busy preflight, evidence archiving, detached launch, verdict roll-up) driving `tools/run-gplr58-a3-arm.ps1` (new): `off@K2`, `off@K2`, `null@8@K2` (mode 2, count-matched), `arm-coact@K2` (mode 2), `retref@K4`, each at the declared short shape - the 885-token non-echoing prompt (`build/calc-prompt.txt`), `--tg-tokens 256`, `--rep-penalty 1.15`, `--kv-cache q8`, `--ubatch 512`, the retained row-set path. Binary: exe `668969213A334632` / llama.dll `6045AC4C71514BEA` (the FIXED dll; the first attempt ran `38386D5B495877F5`). Engine digest `DF2818188A47003F`; `gpu_pre_run` and `gpu_post_run` both `clean`; leg durations 22.2-24.0 s (no contention); all five legs exit 0 and pass their own integrity gates; the provenance gate is 5/5 PASS. Evidence: `build/llama-cuda/gplr58-a3-{off1,off2,null,coact,retref4}.jsonl`, their `-bands.json`, `gplr58-a3-summary.json`, `build/llama-cuda/verdicts/gplr58-a3-*.verdict.json`, `build/llama-cuda/p1-driver-a3.out`; the first (defective-binary) attempt is preserved under `build/llama-cuda/runs/20260923T175746Z/`.

**THE DEFECT THE FIRST REAL RUN EXPOSED, and the fix (engine, 16 lines).** In the first attempt the mode was configured correctly at bring-up - the log reads `mode 2 (proposer arm) budget 8, reserve 8 of cap 16` and `proposer arm coact` - yet the trace carried **no `row_spec_mode` in `run_start` and ZERO `spec` events**, and A2's checker REFUSED the leg (`arm-without-spec`), which is how the failure surfaced instead of being scored as a zero. Cause: `begin_row_cache()` called `spec_reset()` - the TEARDOWN, which clears the mode, the arm and the null source - AFTER the runner's `set_row_set_spec()`/`set_row_set_spec_arm()` and BEFORE the allocation gated on `if (row_set_spec_on)`, so a configured speculative run silently became a plain retention run. Fix: the reset is taken only on the off path (`if (row_set_spec_mode == 0) { spec_reset(); }`); a configured run's per-layer arrays are assigned fresh, and `end_row_cache()` still clears everything at teardown. **Why no host proof could catch it:** `tests/test-layer-streaming-tg-spec.cpp` drives the pure functions and never calls `begin_row_cache()` at all, so the bring-up ORDERING was outside every host fixture. A host-only guard for that ordering is an OPEN ITEM, not something this session delivered.

**The measured session (five usable legs; every pre-registered invariant PASS).**

| leg | role | K | share | staged bytes | admittable stages (fill room) |
|---|---|---|---|---|---|
| `gplr58-a3-off1` | off | 2 | 0.37705 | 85.084 GiB | 0 / 10,240 (0.00%) |
| `gplr58-a3-off2` | off | 2 | 0.37705 | 85.084 GiB | 0 / 10,240 (0.00%) |
| `gplr58-a3-null` | null (mode 2) | 2 | 0.37705 | 85.084 GiB | 0 / 10,240 (0.00%) |
| `gplr58-a3-coact` | arm (mode 2) | 2 | 0.37705 | 85.084 GiB | 0 / 10,240 (0.00%) |
| `gplr58-a3-retref4` | retref | 4 | 0.53486 | 63.560 GiB | 189 / 10,240 (1.85%) |

Bands (all from A2's single owner except the two labelled below): **M1 +0.0000** - the arm's share is the off baseline's to the digit (0.37705 both), and the session's control spread is **0** -> FAIL; **M2 +0.0000** (the count-matched null adds nothing, because it also placed nothing); **M4 +0.0000** and **M4n +0.0000**; **M3 -0.1578** (arm@K2 0.37705 vs retref@K4 0.53486) -> FAIL; **M7a/M7b/M7c UNMEASURED** - no fill was ever issued, so no lead exists to score; P1/P4/P6 PASS (trivially: nothing speculative ran, so `spec_waited_by_demand == 0` and the off legs carry no `row_spec_mode`).

**Why the zero is a CAPACITY result and not a precision result.** In the arm leg the `spec` telemetry records **10,178 of 10,240 layer-stages with `prop > 0`** (the arm published, `arm_count` reaching the full budget 8) while **`submitted == 0` on every single stage** - the planner placed nothing. It could place nothing: at K=2 `cap = min(256, K*top_k) = 16` while admission requires `n_free > reserve = top_k = 8`, and the demand path reaches occupancy 8 on each layer's FIRST token, so the rule can never hold again. The fill-room analyzer, run on THIS session's own traces, independently reports **0 of 10,240 admittable stages at K=2 (each layer full by stage 4)** and 189/10,240 (1.85%) at K=4 - the analyzer's model and the engine's own zero fills AGREE, which is the first end-to-end validation of the bound it was built to express. The arm's ceiling on this shape is therefore exactly ZERO, and per the Planner call recorded above an arm at this headroom has **falsified nothing about prediction**: it was never given one stage with room.

**Reproducibility cross-check (independent of the arm).** This session's retention legs reproduce the recorded honest K-ladder within ~0.1pp: off@K2 **0.37705** against the recorded 0.3778, retref@K4 **0.53486** against the recorded 0.5355. The baseline is not a broken-baseline artifact, and the two legs of the control are bit-identical (spread 0).

**What A3 decides, and what it does NOT.** The charter band M3 (`share >= 0.5355` at K=2) is NOT MET (0.37705), so the block's pre-registered negative fires: **"predictors do not buy depth-equivalent hits at this headroom"** - at K=2, on this shape, under this admission rule. It does NOT bear on the bar09/fusion arms (A4's), on K=4+ (where room exists but is still <= ~2%), or on any shape whose reserve-to-capacity ratio is less hostile; at K=2 that ratio leaves no room at all, so no arm could have differed from another. **A4 was still pre-authorized and NOT RUN at this point**: it faced the same 0.00% room at K=2 with different arms, and a different arm could not create capacity. A4 has since run; its result is below.

**Two A2 (checker) findings this session produced - both FIXED the same day (see the addendum below).** (1) **M5's ratio was a SUM over legs**, so with two `off` legs and one arm leg it printed exactly `0.500x (-50.0%)`: a leg-COUNT artifact, not a measurement (the honest per-leg comparison was 85.084 GiB against 85.084 GiB = **1.000x, no volume regression**). (2) **M3 keyed its denominator on an `off`-ROLE leg at K=4**, while this contract names that leg `retref@K4`, so the checker printed `NOT COMPUTABLE` on a session that HAD its reference and the driver had to difference the two shares by hand; the M3 value above was driver-computed for that reason.

**NOT committed.** The engine fix, the two new drivers and these records are uncommitted in the working tree.

#### A2 BAND-ARITHMETIC FIX (2026-09-23, host-only): M5 is now a per-leg MEAN and M3 accepts the contract's own `retref@K4` as its denominator - both defects were found by the A3 session above

**What changed, and why the old forms were wrong rather than merely awkward.** (1) **M5 was a ratio of SUMS** (`sum(arms)/sum(offs)`). Every session supplies `off` as a PAIR - that pair IS the control spread - while an arm is supplied once, so the band reported the leg COUNT: on the A3 session it printed exactly `0.500x (-50.0%)`, a 50% volume "saving" that was pure arithmetic while all three legs staged 85.084 GiB. It is now the **mean staged bytes per arm leg over the mean staged bytes per off leg** (identical to the old number whenever the roles have equal counts), and the line prints the leg counts and every per-leg staged volume beside it so the reader can see the raw data the mean came from. (2) **M3 keyed its denominator on an `off`-ROLE leg at K=4**; this contract names its K=4 retention reference `retref@K4` and declares that leg's checker role `retref`, so the band printed `NOT COMPUTABLE` on a session holding exactly the reference it needed. The denominator is now the session's K=4 retention reference, preferring an `off`-role leg when a session supplies one (so an existing session's number is unchanged) and accepting the `retref` leg otherwise; the printed label names the ROLE that supplied it.

**Validation (host-only; no GPU, no mutex, no model).** `tools/check-tg-predict.py --selftest` **PASS 76 checks** (was 65; the new fixtures are the A3 session's own shapes: two `off` legs + one arm leg with EQUAL per-leg volumes must read **1.000x**, the artifact string `0.500x` must be absent, the leg counts and the explanatory note must be printed; an arm + `retref@K4` must compute M3 with the `retref` label, an `off`-role K=4 leg must still WIN the denominator when both are supplied - with DIFFERENT shares, so the preference shows in the value and not just the label - and an arm with only a K=2 baseline must still print NOT COMPUTABLE and name both accepted roles). On the recorded A3 traces the checker now prints the session's bands itself: **M3 = 0.3771 vs 0.5349 = -0.1578** (identical to the value the driver had differenced by hand) and **M5 = 1.000x (+0.0%)** with `1 arm / 2 off leg(s)` and the three 91,357,790,208-byte volumes. `tools/check-tg-predict-port.py` still **EQUAL** (8 legs, 483,840 row comparisons); `run-tests.ps1 -Suite host` **SUCCESS 44/44** (digest `DF2818188A47003F`, llama.dll `6045AC4C71514BEA`). The arm driver no longer differences M3 privately: it PARSES the checker's own M3 line, recomputes the same difference from the shares it parsed, records both and warns if they disagree - a regex validated against the real band output before the change was accepted.

**NOT committed**, like the rest of this session's work.

#### A2 FILL-ID REPLAY + PER-STAGE SUBMISSION (2026-09-24, host-only plus one owner-authorized null K=8 GPU recording): ZERO runner violations, the budget barrier is measured directly, and replay exposed a real engine occupancy defect

**Goal and falsifiable hypothesis.** A2's `spec.submitted` was run-cumulative under a per-stage event, so summing it produced 15,189,015 apparent placements and a maximum increment of 306 against budget 8. The old null-K8 checker's 216 retention mismatches looked like a checker-only blindness because `spec` named no fill ids. The narrow hypothesis was: make `submitted` per-stage, emit the ids, and replay them in the cache model; then the same null shape should close with zero violations. The first replayable recording falsified "checker-only": the model found repeated ids, and `spec_issue_one()` was never writing `expert` into `cache_expert[p]` when it marked the slot PENDING. The engine could therefore reuse a position while its prior copy was in flight and could not reject a duplicate row.

**Mechanism and failure contract.** `tel_spec()` now emits `submitted` (successful fills in this stage), `submitted_total` (run-wide cumulative), and `fill_ids` (successful ids in issue order). The runner stages the list before the same stage's `rowset_fetch`, then applies it after the demand window. Its cache model mirrors the engine's state transitions and pure rules: only READY is a hit; PENDING/CANCELLED bytes are not evictable; generation and phase changes cancel/release in-flight rows; a demand for PENDING work fetches its own copy and cancels the fill; speculative rows are the first eviction victims; and a union is stamped in ascending expert-id order. A pre-A2 trace with fills and no ids is UNVERIFIABLE: it gets one leg-level diagnostic and no per-fetch blame. A per-stage trace must have `submitted <= budget` and `sum(submitted) == submitted_total` (P7). The issue path now writes `cache_expert[p]` at the same moment it marks the slot PENDING, so occupancy, duplicate admission, and the demand reserve see the physical row.

**Final GPU evidence (fresh null K=8, not the legacy trace).** Model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` (19.91 GiB), RTX 5090 Laptop GPU (24,462 MiB), 885 PP / 256 TG, K=8, mode 2, count-matched null, budget 8. Trace `build/llama-cuda/gplr58-null-k8-fillids-verified.jsonl`; verdict `gplr58-null-k8-fillids-verified.verdict.json` PASS (0), gpu pre/post clean, engine digest `BE4839804E1F2ADA`, llama.dll `55BF1B63828BB562`. `check-telemetry: OK (31689 events, 4 slots)`, **zero invariant violations**. A2: 10,240 stages; 1,186 fills; maximum 8 per stage; per-stage sum equals the run total; 155 consumed; 16 matched-in-flight / demand cancellations; zero waited; lead minimum 2; share 0.7292; 37.020 GiB staged; peak 10,169.56 MiB. P1/P2/P4/P6/P7/M7a PASS. The null late-fill rate is 0.094, reported and not gated. The legacy `gplr58-kprobe-null-k8.jsonl` is not claimed as passing: with the new checker it has one named `UNVERIFIED` cause covering 10,200 fill-bearing stages because it has no `fill_ids`.

**Validation.** Build targets `llama`, `llama-layer-streaming`, `test-layer-streaming-tg-spec`, and `test-layer-streaming-tg-predict` succeed. Both nested tests PASS; `tools/check-tg-predict.py --selftest` PASS 87 checks, including per-stage over-budget FAIL, stage-sum mismatch FAIL, legacy-cumulative no-evidence, and emitter `fill_ids` / `submitted_total`; `tools/run-gplr58-kprobe.ps1 -ValidateOnly` PASS; `tools/run-tests.ps1 -Suite host` SUCCESS 44/44. The mandatory T-GPLR-07 smoke tier passes on the final binary: 50 and 512 calibrate exactly, both report `check-telemetry: OK (30547 events, 3 slots)`, 50 passes N1, 512 passes N1/N2/N3/N4/K1, and both negative controls FAIL as designed; GPU pre/post are clean. Suite records: `20260924T083429Z-qa-smoke` and `20260924T083722Z-qa-smoke`. The probe's former `-AllowSpecCheckerGap` is restricted to traces carrying neither new key and cannot downgrade a current-schema failure.

**What this does NOT decide.** This is an instrumentation and lifecycle result on a structure-free NULL, not a predictor result, answer-equivalence result, or speedup claim. It does not explain the coact arm's zero-fill behavior at K=4/8/12; the fresh K=8 recording below now explains K=8, while K=4 and K=12 still need fresh legs after the occupancy fix. P2's positive 1,015 residual is the contract's allowed never-consumed READY remainder, not proof that every fill was useful. Gate results remain soft; the owner decides adoption.

**NOT committed.** The nested engine/runner/test edits, outer checker/driver edits, and these records are uncommitted.

#### A2 MODE-2 ADMISSION TELEMETRY (2026-09-24, host-only): every proposal is named and every non-placement class has a per-stage reason

**Mechanism and failure contract.** A mode-2 `spec` event now adds the source's full ordered `candidate_ids`; the planner's `skipped_present`, `skipped_dup`, and capacity-refusal counts; and its terminal `stop` enum plus readable `stop_reason` (`proposals`, `budget`, `pending`, `reserve`, `no_free`). The values come from the existing plan result and do not change admission. All additions are inside the mode-2 guard, leaving mode-0/1 event shape unchanged. Runner validation and `check-tg-predict.py` require a complete record when any admission field appears, cross-check the enum/reason and candidate/fill cardinalities, reject duplicates or impossible values, and report per-leg skip/refusal/stop totals. Partial-schema and contradictory-schema fixtures are refusal cases, not silently downgraded legacy evidence.

**Validation and limit (host-only; no model, CUDA, GPU, or mutex).** The engine, runner, layout test, and both TG tests were rebuilt; the linked tests PASS. Python compilation PASS; predictor selftest PASS 92 checks; host suite SUCCESS 44/44 at engine digest `63280E460286C26F`, `llama.dll` `AAF1E97736CED8DF`, record `build/llama-cuda/suite-runs/20260924T090856Z-host/revision.json`. The pre-admission null K=8 A2 recording is now intentionally refused with `spec-missing-admission-keys`; its earlier zero-violation fill-ID result remains evidence only for the older schema. At the time of this host-only pass no fresh coact trace carried the fields; the later owner-authorized recording below is the first current-schema runtime evidence and establishes the real-world admission explanation.

#### HOST-ONLY CANDIDATE-SELECTION COUNTERFACTUAL (2026-09-24): history is redundant after the demand, coact has no missing candidate while room exists, and cache-aware backfills only after the cache is full

**Design.** The existing predictor screen gained an opt-in cache mode over the K=4/8/12 retention twins. It first requires one identical ordered demand stream and refuses a K mismatch; then it replays each leg's union-protected capacity-LRU and proves its baseline hit count against every recorded stage. Each selector runs on its own copy of that replay with ideal completion: history is the engine's post-observation family, coact is ordinary top-8, and cache-aware takes coact's full ranking, removes rows resident in the live layer cache, and backfills eight without reading future demand. PENDING rows cannot hit or be evicted, a demand for one is cancelled and fetched, other fills become READY after that demand, and the first possible consumption is lead 2. The 189/771/1,588 warm-up room counts independently equal the existing fill-room analyzer.

**Result (10,240 stages per K, budget 8, demand signature `2613f366136ca8ac`).** History skips all 81,920 proposals as present at every K. Coact skips 58,423/73,654/78,686 present and refuses 23,001/7,770/2,738 for capacity; although it publishes at 136/710/1,526 room stages, it has **zero cache-missing candidates at any room stage**. Cache-aware publishes 77,728/72,470/63,306 backfilled IDs, but has zero proposals at room stages: all of its missing candidates arise at `no_free` stages. All three place and consume zero rows, exactly preserve shares 0.53486/0.72726/0.82175, and gain 0.0000pp. Evidence is `build/llama-cuda/gplr58-candidate-counterfactual.json`; Python compile/selftest and port selftest PASS, the K-set negative control exits 2, and the host suite PASS 44/44 with the new lifecycle guard (`20260924T092021Z-host`). No new model or GPU run was performed.

**Boundary.** This is an offline opportunity/ordering result on fixed recorded routes, not a transfer-time or CUDA measurement and not a new pre-registered engine arm. It falsifies cache-aware BACKFILL as a repair for these three traces, but it does not test a source that learns other candidates earlier or changes the capacity/lead schedule. The fresh admission recording below supplies runtime confirmation for the central opportunity result, with its explicit row-level ordering caveat.

#### FRESH K=8 COACT ADMISSION RECORDING (2026-09-24, one owner-authorized GPU leg)

**Evidence.** `build/llama-cuda/gplr58-fresh-coact-k8-admission.jsonl` is a fresh mode-2 `coact` recording at K=8: 885 PP / 256 TG, ctx 1280, q8 KV, four slots, batch-major PP, greedy, no MTP, Qwen3.6-35B-A3B (19.91 GiB), RTX 5090 Laptop GPU (24,462 MiB). The complete command and identity are in `build/llama-cuda/verdicts/gplr58-fresh-coact-k8-admission.verdict.json`; wrapper PASS (exit 0), mutex acquired, GPU pre/post `clean`, engine digest `63280E460286C26F`, runner `D2BD131C866EDFFC`, `llama.dll` `AAF1E97736CED8DF`.

**Validation and result.** The runner's `--check-telemetry` and `--validate-telemetry` both report `OK (31689 events, 4 slots)`. The host admission checker reports `VERDICT: scored`; P1/P2/P4/P6/P7 pass, while M7a is unmeasured because no fill was consumed. Across 10,240 stages, 81,424 candidates were published; runtime classified 73,652 present, zero duplicate, and 7,772 capacity-refused, with zero submissions, fills, cancellations, waits, or consumptions. Stop reasons were `no_free` on 3,764 stages and `proposals` on 6,476; no pending/reserve/budget stop fired. The room analyzer measures 771/10,240 room stages, 710 with candidates, and zero missing candidates at any room stage. The exact-trace host replay agrees on 81,424 proposals, 771 room stages, 710 proposal-at-room stages, zero room misses, zero fills, and 0.0000pp gain; its present/refused split differs by two (73,654/7,770), so this is structural confirmation, not exact row-level predictor parity. The likely boundary is the recorded sorted `union_ids` versus the engine's pre-sort predictor input order; runtime telemetry owns the admission conclusion.

**Boundary.** This is one instrumented arm, not a paired performance A/B: there is no same-session `off` leg, so it cannot establish a causal hit-rate gain, speedup, or predictor quality. It confirms the admission explanation and the zero-fill ceiling for this shape. No model code or admission policy was changed for this recording; all work remains uncommitted.

#### A4 SESSION RUN (2026-09-24, ONE mutex hold, seven legs, GPU-clean): `bar09` and `fusion` both hit the same zero-room K=2 ceiling - no fill, no M1/M7 score, and M3 fails at -0.1578

**What ran and identity.** The owner-authorized A4 command was `tools/with-gpu-lock.ps1 -VerdictTag gplr58-a4-session -Command 'powershell -File tools/run-gplr58-session.ps1 -Session a4 -ExternalMutex'`. It held `Global\AI-GPU-EXCLUSIVE-v1` for the complete seven-leg series; GPU pre/post were `clean`. The legs were `off1`, `off2`, `null-bar09`, `bar09`, `null-fusion`, `fusion`, then `retref4`, at the declared 885-token non-echoing prompt, 256 TG tokens, ctx 1280, q8 KV, four slots, batch-major PP, greedy, no MTP, and `--rep-penalty 1.15` on every leg. Model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`; runner `D2BD131C866EDFFC`; `llama.dll` `AAF1E97736CED8DF`; engine digest `821E503F27DF18B8`; nested HEAD `8b3ea428112b`. All seven legs exited 0, passed runner telemetry validation and their individual A2 checkers, and have complete PASS verdicts with that binary provenance. Ordered route identity is one value (`5de08495940289905ac9a0d229aa552f4cced9627c83ed78e6e514e70e394798`) and greedy-token identity is one value (`f10d47289b73baffbe660dbc80cddf1d4052faf51c51bbc257063dc0ff32c9f3`).

| leg | role | K | share | staged GiB | admittable stages |
|---|---|---:|---:|---:|---:|
| `gplr58-a4-off1` | off | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-off2` | off | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-null-bar09` | null (mode 2) | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-bar09` | arm (mode 2) | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-null-fusion` | null (mode 2) | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-fusion` | arm (mode 2) | 2 | 0.37705078125 | 85.084 | 0 / 10,240 (0.00%) |
| `gplr58-a4-retref4` | retref | 4 | 0.53486328125 | 63.560 | 189 / 10,240 (1.85%) |

**Paired results and opportunity ceiling.** Each arm was scored with its own count-matched null in a five-leg checker call. `bar09`: off/null/arm `0.37705078125`, paired share gain `0`, paired null-corrected margin gain `0`, M3 `-0.1578125`, M5 `1.000x`. `fusion`: the same values. These are measured equality and volume results, but M1 and M7 are **not scoreable**: all four speculative legs submitted, consumed, cancelled, and waited on zero fills. P1/P2/P6/P7 hold; P4 accounting closes, but cancellation is not exercised. The independent room analyzer reports 0/10,240 room stages for every K=2 leg and 189/10,240 at K=4, matching the runtime zero. Capacity, not predictor precision, owns this result.

**Admission evidence.** `null-bar09` records 4,467 present skips, 66,779 capacity refusals, and stops `no_free=10,129/proposals=62/reserve=49`; `bar09` records 37,067/34,179 and `no_free=9,591/proposals=649`; `null-fusion` records 5,196/76,724 and `no_free=10,151/reserve=89`; `fusion` records 61,881/20,039 and `no_free=8,488/proposals=1,752`. Duplicate refusals are zero. Raw traces, paired reports, summary and verdicts are `build/llama-cuda/gplr58-a4-*.jsonl`, `gplr58-a4-bands*.json`, `gplr58-a4-summary.json`, and `build/llama-cuda/verdicts/gplr58-a4-*.verdict.json`.

**Host roll-up failure and correction (no second GPU run).** The first A4 driver roll-up built its off-tag list as `@($prefix + '-off1', $prefix + '-off2')`; PowerShell bound this as ONE combined string, the pair checker received only null/arm/retref, and the driver rejected its own incomplete result and exited 1. This was tooling-owned, not a model or invariant failure. The driver now selects the two controls by role, asserts the exact `off,off,null,arm,retref` shape in `-ValidateOnly`, and has a host-only `-RefreshPairReportsOnly` path. That path re-verified 7/7 inner verdicts, reran both paired checkers, and rewrote only the derived pair fields in `gplr58-a4-summary.json`; it exited 0 with no model, CUDA, GPU or mutex use. The outer `gplr58-a4-session.verdict.json` remains the historical `UNCERTIFIED (10)`, because the wrapper saw a nested PowerShell command (`binary_state=not-supplied`) and the original command exited 1. It was deliberately not rewritten. The seven inner verdicts are the attributable binary evidence.

**What A4 decides, and what it does NOT.** A4 supplies the two missing runtime arms and confirms the same pre-registered negative as A3: at K=2 on this shape neither arm reaches the K=4 retention share, and the K=2 admission rule leaves zero opportunity. It does NOT establish that `bar09` or `fusion` lacks predictive precision, does not score M1/M7, and does not cover a less hostile reserve/capacity ratio, a sampled regime, or a changed admission schedule. The evidence-complete task result is negative for the declared charter test; final acceptance remains the owner's decision. The owner authorized the outer A4 tooling/evidence commit after validation: `4e34b85`.

### T-GPLR-59 (draft): the two predictor source classes T-GPLR-58 excludes - an MTP draft-preview source (PROPOSED FUTURES at zero learned weights) and a hidden-state probe head (measured offline against its OWN ceiling gate) - each with its own pre-registered gate

Status: D1 evidence-complete, D-gate FAIL (2026-09-25). The same-binary MTP session ran D1 plus count-matched null and all three available past-only arms (`coact`, `bar09`, `fusion`) with their count-matched nulls. D1 cleared the 60-percent pairing rule and beat its null on proposal coverage, but no arm admitted or consumed a speculative row. Every arm therefore tied at 0.451838 demand-hit share, so D1 did not beat the measured past-only frontier and adds no demonstrated runtime demand-hit value. D2 and Class E remain parked; this evidence does not authorize adoption.

Dependencies: T-GPLR-58 (the invariant, the null and the bands - this block is a source-class
companion to it, not a second mechanism); T-GPLR-19 (the row cache and the K ladder both
classes feed); the MTP draft head (T-GPLR-12 / T-Q38HG-06: `--mtp`, `--mtp-draft`, the `mtp`
telemetry event) for class D; the recorded model inventory `build/llama-cuda/q36-moe-inventory.json`
for the shape facts below. Not blocking: T-GPLR-53/T-GPLR-56 (untouched).
Commit: nested e62b32427 + outer 31b07ab (shared prediction-pool/validator substrate); D1 implementation and MTP frontier wiring nested e7795905d; final D-gate tools/evidence/docs outer b57e5ff.

Goal. Answer the two questions T-GPLR-58 deliberately leaves out, each with a gate that can
FAIL on its own terms: (D) does a predictor that reads a PROPOSED FUTURE - the tokens the MTP
draft head proposes before the target model routes them - add demand-hit value that the
past-only families cannot, at zero learned weights in the engine; and (E) does the model's own
hidden state carry more routing signal than the id stream, measured as a CEILING offline
before anyone argues about shipping it. Neither class is allowed to be judged by the other's
evidence, and neither is allowed to inherit T-GPLR-58's bands without its own gate.

Scope. Class D: one online table keyed by the DRAFTED token id (D1, no new weights) plus the
single nextn layer's own routed set reported as a carried, low-expectation source (D2); the
checker's class-D refusals; the void rule read from the EXISTING `mtp` event. Class E: ONE
default-off, capture-only engine hook for the pre-attention residual state per (token, layer),
an offline trainer with a tune/held split by token, and the class gate scored offline. Both
classes feed their proposals through T-GPLR-58's speculative path; neither adds a second one.
Non-goals: no change to drafting or speculative-acceptance logic, and no change to the drafter
itself; no training inside the engine; no engine-side learned weights (a sidecar) - that is a
separate owner decision with colibri's T-opt-45 no-go as precedent; no new dependency before
owner approval (measured: neither numpy nor torch is installed here); no PP-side use of either
class; no re-derivation of T-GPLR-58's bands or invariant; no comparison between MTP and
plain-TG traces; and no inference of runtime benefit from proposal coverage when the planner
admits zero rows.

**D1 HOST SLICE (2026-09-25).** The selected D1 source has a pure C++ table keyed by `(drafted token, layer)`. It predicts before observation, updates only accepted draft/target pairs, ranks by accepted count then ascending expert id, and exposes the 60-percent pairing gate. `llama_ls_spec_propose_d1()` feeds the existing T-GPLR-58 planner and count-matched null. `tools/tg59_d1.py` replays explicit `d1_pair` records and is exposed through `tools/check-tg-predict.py --class D`; it refuses missing shape, missing pairs, duplicate demand ids, and malformed fields. This slice does not infer pairs from aggregate `mtp` counters and does not claim a D-gate result.

Focused host validation: `test-layer-streaming-tg-predict` passed; the D1 host selftest passed **8/8** after the adversarial replay pass; the existing T-GPLR-58 checker selftest remained **92/92**. The replay now scores null traces from their recorded runtime proposals, requires one leading `d1_shape`, rejects out-of-order pairs, and checks the declared budget. The row-set runtime now refuses router-row/ubatch width mismatches and missing captured layer vectors before indexing them. No model, CUDA, GPU, mutex, or MTP handoff run was used.

**D1 RUNTIME SLICE + SAME-SESSION FRONTIER (2026-09-25, final D-gate FAIL).** The runner executes the supported MoE MTP row-set path and emits real `d1_shape`/`d1_pair`/`d1_summary`. The first owner-authorized D1 arm/null pair produced 87 rounds, 6,960 pair rows, 6,720 accepted rows and a 0.965517 pairing rate. D1 replay coverage was 0.1664 overall / 0.1723 paired versus null 0.0153 / 0.0158 at the same 29,120 published candidates. That is proposal precision, not runtime value: both legs admitted and consumed zero rows and tied on row-cache hits, transfer and peak VRAM.

The final same-binary frontier removed the stale MTP/non-D1 flag and prebuild restrictions, then ran D1 plus `coact`, `bar09`, and `fusion`, each with its count-matched null, at 885 PP / 256 TG, ctx 1,536, MTP draft 2, Q8 KV, K=4, budget 8. All eight wrapper verdicts passed with clean GPU pre/post and engine digest `20AE6C20E23C6B77`. Every leg produced 30,387 resident hits / 36,865 fetched rows = 0.451838 share, 65,975,820,288 TG staged bytes, and 8,658,681,856 peak VRAM. Every arm recorded zero submitted, consumed, cancelled, matched and waited rows. D1's proposal coverage remained +0.1511 over its count-matched null, but its runtime gain and every past-only runtime gain were exactly 0. The host D-gate therefore returns FAIL: proposal precision cannot substitute for cache demand-hit value. Canonical report: `build/llama-cuda/gplr59-d-gate-1536-r4.json`; traces/verdicts use the `gplr59-frontier-*-1536-r4` prefix. The first coact r3 attempt failed before PP at the stale `llama-context.cpp` width guard and is superseded, not evidence.

**D1 adversarial closure (2026-09-25).** Host probes found accepted out-of-order D1 pairs, duplicate `tg_tokens`, incomplete runtime fixtures, partial D1 observation on a late shape error, and retained D1 teardown state. The replay/runtime readers now enforce the explicit ordering and lifecycle contract; D1 observation validates before mutating; teardown releases D1 state; and new `d1_shape` events carry `n_layer`. The D-gate now requires a session manifest for a PASS, validates all eight clean wrapper verdicts and one source/binary/library/normalized-command identity, and requires a measured non-negative spread. The canonical manifest records `spread: null` because this session had no repeated-control spread, so the repaired gate remains FAIL with an explicit `session spread is unmeasured` reason. D1 replay self-test (18), class-D and general checker self-tests (96), mode-2 (7/7), focused CTest (3/3), Python compilation, and diff checks pass. No GPU/model run was repeated. Semantic-QA remains an open task-level gap.

- **Goal.** Make the D1 source executable in an opt-in MTP handoff session: after the draft head has produced a round, publish D1 guesses before target verification, settle them outside the demand wait, and record the exact target routes with the accepted/rejected outcome after verification.
- **Scope.** One MTP row-set window width (`1 + mtp_draft`); per-layer route capture already performed by the functional row-set path; D1 prediction/observation through the existing real-cache planner and count-matched null; explicit `d1_shape`/`d1_pair` JSONL; host validator/replay updates; a source label that distinguishes D1 from T-GPLR-58 past-only arms.
- **Non-goals.** No D2, no Class E, no hidden-state capture, no MTP acceptance/drafting change, no new pool, no learned weights, no claim that MTP and plain-TG arms are comparable, and no model run without owner authorization.
- **Design/failure contract.** D1 publication is at the pre-verify boundary. Every issued copy is completed before the target decode begins, so a speculative completion can never extend a demand's wait. Rejected drafts contribute to overall coverage and the 60-percent pairing denominator but never train the table. `d1_pair` carries both exact `demand_ids` and runtime `predicted_ids`; a runtime trace with `predicted_ids` required by `d1_shape` is replay-checked against the online table. Missing route rows, a stale/mismatched capture, an occupied/invalid planner result, or an unsettled copy fails the run with structured telemetry; it never falls back to a past-only proposal. The MTP row-set width must be declared, and retention depth must be at least the full verify width plus one free row so the functional demand union fits while the D1 planner can admit a non-zero speculative set; an exact-capacity D1 configuration is refused.
- **Implementation steps.** (1) Generalize only the MTP row-set width guard/validators and declare the width in `rowset_init`; keep the old width-1 path unchanged. (2) Capture `[token][layer]` routes from the existing router read. (3) Add the D1 source configuration and pre-verify issue/settle path. (4) Observe accepted/rejected pairs after verification and emit the explicit schema. (5) Add C++ and Python host tests, then run focused build/CTest, the D1 checker selftests, the host suite, and doc split checks.
- **Acceptance criteria.** Focused targets compile and pass; host tests prove predict-before-observe, accepted-only training, rejected-route coverage, runtime prediction/schema cross-checks, pre-verify issue ordering, settle-before-demand, width/capacity refusal, and exact JSONL schema/refusals; `git diff --check` and current-state split verification pass. These are feasibility/instrumentation results only, not a D-gate or performance result.
- **Validation commands.** `cmake --build build/llama-cuda --config Release --target test-layer-streaming-tg-predict llama-layer-streaming llama-layer-streaming --parallel 16`; focused CTest for `test-layer-streaming-tg-predict` and the existing layout/spec tests; `python tools/check-tg-predict.py --class D --selftest`; `python tools/check-tg-predict.py --selftest`; `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-tests.ps1 -Suite host`; `python tools/resplit-currentstate.py --verify`; `git diff --check`.
- **Required state/docs.** Record the final same-binary D1/frontier traces and verdicts, separate proposal precision from runtime fill/hit/transfer/VRAM evidence, state the measured D-gate FAIL and its capacity cause, and leave adoption to the owner. Remove superseded claims that no GPU arm ran or that the D-gate remains incomplete.
- **Escalation.** Any future width-generic execution that needs a change to routing arithmetic, MTP acceptance, graph topology beyond the two stale D1-only restrictions removed here, or a new dependency returns to the Planner. A capacity change that merely creates speculative room would be a new pre-registered experiment, not a repair hidden inside this completed negative gate.



- **The draft head has NO depth, and that is the fact that reshapes class D.** The recorded
  inventory (`q36-moe-inventory.json`, model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`) reads
  `trunk_layers` 40, `total_block_count` 41, **`nextn_predict_layers` 1**, and byte class
  `mtp_layer` **4,863,651 B**; `load_block_mtp` builds that one block as a full MoE layer
  (ffn_gate_inp 2048x256, ffn_down_exps, gate_up_exps, plus a shared expert). So colibri's
  formulation - "the draft model's per-layer routing selections preview the target model's
  expert sets" (their drafter had depth) - **does not port as written**: our draft's OWN routed
  set exists at ONE layer out of 40. The axes that DO exist are the drafted token IDS (D1) and
  that single nextn route (D2), and the block is designed around exactly those two.
- **Class D's void-rule input already exists.** The `mtp` telemetry event carries `tokens`,
  `mtp_drafted`, `mtp_accepted`, `accept_rate` (llama-layer-streaming.cpp:5843-5850), so the
  pairing/acceptance arithmetic needs no new counter.
- **No MTP arm has ever been recorded in this build directory** (no `mtp` event in the recorded
  top-level traces), so class D starts from zero evidence: its first run's acceptance rate is a
  MEASUREMENT and a void-rule input, not a result.
- **MTP is an execution-mode change.** `--mtp` requires `--handoff (-attention-only)` and
  `--tg-tokens` (main.cpp:2965-2970), is refused with `--pp-repeat > 1` (:3173) and with
  `--pp-prebuild-per-pass` (:3241); and the recorded AGENTS constraint stands - `--pp-schedule
  layer-major` refuses `--tg-tokens > 3` when handoff auto does not activate. Consequence,
  stated so no reader pools the two: class D's arms run in a HANDOFF session and T-GPLR-58's
  arms in a PLAIN TG session, and legs from the two sessions are NOT comparable to each other.
- **Class E has no numerical library available.** `import numpy` and `import torch` both FAIL in
  this environment (measured). So the trainer is either pure Python or an owner-approved new
  dependency - which AGENTS gates explicitly. This is owner call 1, and it is blocking.
- **The ceiling these classes aim inside is now measured** (T-GPLR-58's BOUNDING MEASUREMENT,
  below its arms): the incumbent's uncovered margin is 22-57% of demands depending on K, only
  0.5-3.4% of that margin is unreachable by observed ids (C4), and it splits ~41% same-token
  cross-layer (C1, needs an intra-token lead), 24-49% same-layer beyond K (C2, also reachable by
  a deeper cache) and 7-31% cross-layer long-range (C3). Class D and class E must therefore beat
  the past-only frontier INSIDE that composition; there is no information wall to exploit.
- **The capture budget is bounded and known.** `embedding_length` 2048, 40 trunk layers (30
  recurrent + 10 attention - a hybrid), `expert_count` 256, `expert_used_count` 8. At int8 plus
  a per-row scale the state row is 2,052 B, so a 256-token x 40-layer TG cell is **21.0 MB**
  captured (bf16 would be 41.9 MB); the hook therefore needs a sample-every rule for larger
  cells, not a format change.
- **Faithfulness holds for both classes by construction, and differently from T-GPLR-58's
  families.** A draft for position t+1 is produced from tokens <= t; a probe at (t, L)
  predicting (t+1, L) reads state computed a full token sweep earlier. Neither can leak the
  demand it predicts - which is why these two classes are the ones the past-only families
  cannot imitate.
- **The trap that governs D's table is measured, not hypothetical:** colibri's Atlas record -
  "rejected drafts' routes are noise by construction: speculative drafts count pre-verification
  routes". The table is therefore built from ACCEPTED drafts ONLY, and coverage is reported
  BOTH overall and paired-only, so the class cannot claim coverage its live form would not have.

CLASS D - MTP DRAFT PREVIEW (proposed futures; no learned weights in the engine):

- **D1 (PRIMARY).** An online table P(ranked expert ids | drafted token id, layer) learned from
  observed (token, exact route) pairs - the same class of online table as coact/markov, but
  keyed by the token the DRAFTER proposes instead of tokens already seen. At draft time (one
  round before the demand) the drafted token's ranked ids are staged speculatively through
  T-GPLR-58's path. This is the SP-MoE "proposed futures" axis and it needs no new weights, no
  sidecar and no dependency.
- **D2 (carried for completeness; low expectation, named as such).** The draft head's own
  routed expert set - ONE layer of 40 - as a feature about the future position. The block
  requires D2's reach to be printed beside its number and does NOT expect it to move a band.
- **D-GATE (its own, pre-registered, and the reason this class exists).** D1's margin share
  must beat the BEST PAST-ONLY ARM measured in the SAME session (T-GPLR-58's strongest available
  arm at that shape) by more than the session's own spread, under T-GPLR-58's count-matched
  per-row null. **Beating history alone is explicitly NOT a pass**: the class claims that
  proposed futures add what the past cannot, so the comparison must be against the past-only
  FRONTIER, and a result that only re-measures history is recorded as such.
- **VOID RULE (pre-registered).** If fewer than 60% of target positions pair with an ACCEPTED
  draft, the class is recorded NULL-UNPAIRED with the measured acceptance rate from the `mtp`
  event, and no share band is read (colibri's join-rate void rule; their T-pf-06 also records
  that a low-pairing candidate is void rather than degraded).
- **LEAD, different in kind.** A draft arrives one ROUND ahead, not one layer ahead, so the lead
  is reported in rounds and draft-depth (with the drafted -> accepted -> demanded ordering
  visible in the trace), and T-GPLR-58's layer-lead band is not reused as-is.

CLASS E - HIDDEN-STATE PROBE HEAD (offline ceiling first; engine form explicitly deferred):

- **E1 capture (the ONLY engine code in this block).** A default-off, capture-only hook emitting
  the pre-attention residual state entering layer L, one row per (token, layer), int8-quantized
  with a per-row scale, budget-bounded with an explicit sample-every rule; byte-identical off;
  no effect on routing, ids or output. This is colibri's T-pf-01-class discipline: capture-only,
  default off, no admission wiring.
- **Trainer.** A per-layer linear ranking map fit on a TUNE window only (80/20 split BY TOKEN),
  with a ranking-aware loss; per-layer recall reported, never pooled (colibri's surveyed
  failures concentrate in shallow layers, and a pooled number would hide exactly that).
- **E-GATE (its own, pre-registered, comparative - not absolute).** On held-out data and at the
  SAME proposal budget on the SAME stream, the probe's per-layer recall must beat the BEST
  id-stream family; and an int8-state probe must agree with a float-state probe within 1pp
  recall, else widen to int16 and record why (their quantization-sanity rule). A FAIL is a
  recorded NULL: "this class does not transfer to this model" - decision-useful, and the whole
  point of measuring a ceiling before arguing about shipping it.
- **ENGINE FORM IS OUT OF SCOPE.** Putting the probe in the engine means shipping learned
  weights - a sidecar - and colibri's T-opt-45 was an explicit no-go on trained sidecars; plus
  the trainer carries the dependency question. Nothing in this block authorizes either, and
  owner call 2 decides whether that is ever a separate task.
- **TWO COMPARABILITY LIMITS, recorded so the number is not over-read:** (a) this ceiling is
  measured on OUR model with OUR capture, so it is NOT comparable to colibri's reported
  "93%-class" probe signal, which came from a different model, capture and task; (b) recall@B
  on a held-out split is not a VRAM-normalized quantity, so a probe WIN here justifies a
  follow-up block - it does not justify adoption.

OWNER CALLS (two blocking, one shaping):
1. DEPENDENCY, BLOCKING: may the class-E trainer use an approved numerical library (numpy) as a
   tools-only dependency, or must it stay pure Python? Measured: neither numpy nor torch is
   installed, and AGENTS requires owner approval before any install. A pure-Python fallback is
   feasible at this size but slower; the block must state which one it is before B3 starts.
2. LEARNED SIDECARS, BLOCKING FOR CLASS E's future only: is a trained sidecar in the engine ever
   in scope? If no, class E's E-gate is its terminal artifact and that is a complete, useful
   result; if yes, that is a separate block with its own policy (T-opt-45 no-go as precedent).
3. EXECUTION MODE, SHAPING: may class D force handoff to get MTP arms at short shapes? MTP
   requires `--handoff*` and the recorded constraint refuses long TG under layer-major without
   it, so this answer sets which shapes class D can be measured at - and it must be recorded,
   because it makes class D's legs non-comparable with T-GPLR-58's plain-TG legs.

Implementation steps:
- B0 (host-only): extend T-GPLR-58's checker with `--class D` - the class-D refusals (an MTP
  trace with no `mtp` event; a paired fraction below the void bar) and the D-gate comparison
  (D1 against the session's best past-only arm). A missing `mtp` event is a FAILED check, never
  a zero, and a refused leg prints no number.
- B1 (host-only): the D1 table plus paired/overall coverage reporting, validated by replaying a
  recorded route stream CONDITIONED on drafted tokens (a fixture synthesises the pairing; the
  real pairing needs an MTP arm, and the fixture proves the table and the coverage arithmetic,
  not the model).
- B2 (engine, default-off, capture-only): the class-E capture hook and its host test - row
  shape, per-row scale, sample-every honoured, mode 0 byte-identical, no routing/output change.
- B3 (host-only): the trainer (pure Python or the approved dependency, per owner call 1) and the
  E-gate scoring on a held-out split, including the int8-vs-float agreement check.
- B4 (one mutex hold, HANDOFF session): the class-D arms, void rule read from the `mtp` event;
  the class-E capture may ride along in the same hold once B2 is host-validated, since it is
  capture-only and changes no demand path. Class E's gate itself is scored offline afterwards.

Acceptance criteria (falsifiable):
- B0: the checker refuses an MTP trace lacking the `mtp` event and one below the void bar, and
  prints no number in either case; the D-gate fails a leg whose gain exists only against
  history (a fixture constructs exactly that leg and must be reported NOT A PASS).
- B1: the replay reproduces the fixture's paired table and the both-forms coverage.
- B2: capture row shape/scale asserted, sample-every respected, mode 0 byte-identical, and the
  existing `test-layer-streaming-*` family green.
- B3: the E-gate is computed on held-out data with PER-LAYER reporting and the agreement check;
  a FAIL is recorded as a named NULL, not as a rerun.
- B4: class D yields either a scored D-gate result or NULL-UNPAIRED with the measured acceptance
  rate; T-GPLR-58's identity gate (route digest + greedy ids across legs) applies unchanged.
- Default preservation: with both classes' features absent, the runner is byte-identical to
  today; `run-tests.ps1 -Suite host`, `check-doc-sizes.py` and `resplit-gplr-plan.py --verify`
  all PASS.

Focused validation commands and expected results:
- `cmake --build build/llama-cuda --config Release --target llama-layer-streaming` -> green;
  exe/dll hashes recorded (a new pair means the control must be re-recorded).
- `python tools/check-tg-predict.py --selftest --class D` -> PASS including the two class-D
  refusals; `--class E` refuses a capture-less analysis.
- `ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-predict` -> PASS,
  including the new capture-shape test.
- `python tools/train-probe-head.py --selftest` -> PASS (deterministic fixture with a planted
  per-layer signal and a planted NULL layer, so the gate is shown to fail when it should).
- `powershell -File tools/run-gplr59-session.ps1 -Session d1 -ValidateOnly` -> exit 0;
  `-PreflightOnly` -> 0 on a clean card or 125 naming the PID when busy; never `-AllowBusy`.
- `python tools/check-doc-sizes.py`, `python tools/resplit-gplr-plan.py --verify` -> PASS.

Required state/document updates: a `docs/currentstate.md` entry per executed arm, naming the
legs that did NOT run and the execution mode each leg ran in (handoff vs plain TG, because the
two are not comparable); the findings doc gains the D-gate and E-gate tables; T-GPLR-58's block
gains a pointer ONLY if either class changes its arms (otherwise it is untouched); the SDD only
if a mechanism is adopted (owner decision).

Escalation condition: any proposal to change drafting or acceptance logic (that is the drafter's
task, not this block's); any dependency install or sidecar (owner calls 1 and 2); any result
where D1's advantage exists only against history and not against the past-only frontier (that is
the "we re-measured history" outcome and belongs back with the screening as its verdict); any
capture that perturbs the run's output or routing (a broken hook, not a measurement); any
offline-vs-engine direction flip (faithfulness bug: escalate, do not retry).

### T-GPLR-16 (CLOSED 2026-09-22 — the option (a) revert LANDED): the `--handoff-partial` MIN_BATCH=1 override is out of the tree, and a host-only guard keeps it out

T-GPLR-16 was the mixed-placement TG mode (partial handoff: the first N layers' expert banks
stay host-resident). Its numeric fix forced `GGML_OP_OFFLOAD_MIN_BATCH=1` so CUDA's `MUL_MAT_ID`
would keep taking those host banks, with an env-var bypass for value probes. The 2026-09-12
characterization showed that route deadlocks the device stream on the third TG token whichever
way it is reached, and the owner closed the task with **option (a): revert to the pre-fix partial
TG**, accepting the known step-1 gold flip. That decision was recorded then
(`docs/currentstate/2026-09-12..2026-09-13.md`); the revert itself had not been landed. It is now.

**What changed (vendored tree, one site).** `tools/layer-streaming/main.cpp`: the
`_putenv_s`/`setenv` force, the `bypass_minbatch_probe` static, the `gplr16:` stderr announce line
and the explanatory block around them are gone, replaced by a short comment that records the
revert, the deadlock and the accepted mixed-arithmetic cost, so the idea is not retried blindly.
`--handoff-partial` itself is untouched: the mode, its help text and all four compatibility
refusals stay. Nothing else in the tree referenced either string.

**Closure evidence, all host-only (no GPU time was spent).**

| check | result |
|---|---|
| `python tools/check-gplr16-revert.py` | **PASS**: 0 setters across 10 set-patterns (C `_putenv_s`/`setenv`/`putenv`, Win32, python, powershell, shell, cmd), the bypass flag **absent from source entirely**, no `gplr16:` announce line |
| reader allowlist | 5 reads of the variable, **all upstream** (`ggml-cuda.cu:5645`, `ggml-vulkan.cpp:18704`, `ggml-sycl.cpp:6733`, `ggml-cann.cpp:2996`, `ggml-metal-device.m:883`, each the `? atoi(getenv(...)) : 32` default); a new read on this project's own path FAILS the check |
| `--selftest` | **10/10** — a negative control for every way the revert can be undone (override back, bypass back, announce back, launcher `$env:` set, python `os.environ` set, read outside the upstream backends, mode surface removed) plus two refusals (mode file absent, empty tree) |
| host suite | `[t16-revert]` registered (selftest + real scan); `tools/run-tests.ps1 -Suite host` exit 0, `build/host-suite-t16.log` |
| compiles | `cmake --build build/llama-cuda --config Release --target llama-layer-streaming` exit 0 |
| mode behaviour | the mode's own refusal tests pass on the reverted binary: `ctest -R refuse-partial` -> `layer-streaming-refuse-partial-dense` and `-zero` both **Passed** (0.19 s total, so neither initialises CUDA). They are NOT in the host suite's CTest Include filter, so they were run explicitly - a suite-run-only report would have missed them |

**New binary identity, recorded because a rebuild makes earlier A/B controls a different
binary:** `llama-layer-streaming.exe 38B39C06229BF15B`, `llama.dll 2930A7D0B30F967E`,
`ggml-cuda.dll C04D744E9A4213F1` (2026-09-22 16:07Z). **SUPERSEDED (16:32Z, and this is the
finding the arm-provenance block below exists for):** a concurrent edit landed on
`src/llama-layer-streaming.{h,cpp}` at **16:31:13** and the tree was rebuilt at **16:31:58 /
16:32:01**, so this identity - and any claim leaning on it - describes a revision that no longer
exists. The current tree is `llama-layer-streaming.exe D01874A4235542A8` / `llama.dll
46628B7CDA8FAFA1` (`ggml-cuda.dll` unchanged since 09-17). Nothing in the record noticed at the
time, which is why evidence binding is now mechanical rather than editorial.

**The guard caught its own registration on first use (measured).** The host-suite comment that
named the bypass string made the real scan FAIL with 1 reference until the comment was
paraphrased — `tools/` is in scope, so a mention there is a surface. That is the check working,
and it is recorded because it also shows the rule's strength: it does not care that a comment is
harmless.

**Smoke regression RAN, PASS (2026-09-22 14:28:28Z-14:28:47Z, one mutex hold).** Four bounded probe
windows were refused first - 13:57:15Z-14:00:48Z, 14:10:13Z-14:13:15Z, 14:14:58Z-14:16:19Z and
14:22:54Z-14:26:41Z, 22 samples in all, the shared lock held by `deepseek_v4.exe` (pids 21208, 23744
then 17352) in every one - and the fifth cleared at **14:28:20Z** (`lock_free=True`, no compute app),
which is the fail-fast preflight the rule asks for before a window is spent.

| item | measured |
|---|---|
| chain verdict | `build/llama-cuda/gpu-smoke-ctxshell.verdict.json` **PASS**, exit 0, 17.9 s, `fetch_events 320`, `tg_text_bytes 21`, `checker_exit 0` |
| arm verdict | `build/llama-cuda/gpu-smoke-arm.verdict.json` **PASS**, exit 0, 13.4 s, `gpu_pre_run`/`gpu_post_run` clean, `mutex_acquired true` |
| arm log | `build/llama-cuda/gpu-smoke-ctxshell.log`: arm exit 0 after 16.0 s, `check-telemetry: OK (1049 events, 4 slots)`, `RESULT: CHAIN OK (ctx_shell -> mutex -> CUDA, no CUDA_PATH)` |
| arm command | `--residency moe-expert-bank --residency-tg --pp-schedule batch-major --pp-tokens 64 --ubatch 512 --tg-tokens 8 --layer-slots 4 --kv-cache q8 --greedy --no-mtp --pp-no-logits --allow-full-residency` on the 512-token anchor - **no `--handoff-partial`**, i.e. the default path the revert had to leave alone |
| binary | `llama-layer-streaming.exe 38B39C06229BF15B` (16:07Z), `llama.dll 2930A7D0B30F967E` (15:32Z), `ggml-cuda.dll C04D744E9A4213F1` (10:14Z) - unchanged since the revert's rebuild, so this smoke is same-binary with the reverted tree **as of 16:07. CORRECTED by the arm-provenance block below:** the engine sources were edited at 16:31:13 and the tree rebuilt at 16:31:58/16:32:01, so this holds for the 16:07 revision only. The exe hash is also the weaker anchor - the streaming code lands in `llama.dll` - and no saved verdict records the sources, so the claim cannot be re-certified without a re-run |
| after | card clean, `lock_free=True`, no stray process |

**What the smoke proves, and what it does not.** It proves the chain still runs on the reverted binary:
mutex, CUDA init, the residency path, telemetry, and the strict checker. It does **not** exercise
`--handoff-partial`, and it is not a measurement - its own header calls it a connectivity check, so the
PP/TG rates in its summary (129.7 / 15.3 tok/s over a 64-token PP and an 8-step TG) are context only,
never a result. One incidental finding: `gpu-smoke-arm.verdict.json` is the **first real arm verdict
carrying the wrapper's new provenance fields** (`tag_source: explicit`, `gpu_probe_source: nvidia-smi`,
both probe statuses `clean`), so the previous pass's wrapper work is now exercised in production, not
only by stubs. Also noted, not claimed: `cuda_path_empty: false` - this shell had CUDA_PATH set, so the
smoke's original "works without CUDA_PATH" property is not what this run tested.

**Out of scope, named:** the other T-GPLR-16-era probe guards (`GGML_CUDA_NO_GRAPHS_PROBE` and
friends) are untouched — this closure is the forced variable and its bypass. `tools/gplr16-dump.cpp`
(the hang stack-dumper) stays: it is a diagnostic and sets no flag. The historical narrative in the
dated currentstate files keeps its original text; the 2026-09-12 disposition carries a pointer that
the revert landed. **No latency or throughput win is claimed**, and nothing here re-opens the
option (a) decision.

### Arm provenance: a verdict must name the SOURCE TREE and the BINARY it describes (mechanism landed 2026-09-22; host-only, no GPU spent)

**The defect, measured.** This thread's own 4k control arm passed at **16:28** against the **16:07**
build. At **16:31:13** a concurrent edit landed on `third_party/llama.cpp/src/llama-layer-streaming.{h,cpp}`
and the tree was rebuilt at **16:31:58 / 16:32:01**. The verdict named only an exe hash, no consumer
compared that hash to anything, and the record's "same-binary with the reverted tree" claim stopped
being true with nobody touching it. Two weaker variants of the same hole are in this tree as well:
`tools/layer-streaming/main.cpp` was modified at **16:17**, after the 16:07 build, so that smoke ran a
binary which cannot have contained it; and the exe hash is the wrong anchor anyway, since the streaming
code lands in **`llama.dll`** and a relink churns the exe hash while the engine is unchanged.

**What a verdict now carries** (`fields.provenance`, plus `provenance_post` and `provenance_delta` from a
wrapped run). The mechanism lives in `tools/source-revision.ps1` (digest + the decision rule) and
`tools/arm-verdict.ps1` (`Write-ArmVerdict` attaches it), **not** in the wrapper alone:

| field / group | meaning |
|---|---|
| `source_scope`, `source_root`, `source_head` | which tree, and its nested HEAD (`third_party/llama.cpp` at `23e5d1b6e617`) |
| `source_digest` | 16-hex SHA-256 over HEAD + the full content diff of tracked changes + the CONTENT of untracked files. Content, not mtimes - reverting an edit returns the earlier digest, which is the correct reading |
| `source_changed_tracked` / `_untracked`, `source_newest_change_path/_utc` | what moved, and the newest change |
| `source_changed_paths` (+ `_truncated`) | a bounded manifest (50) of the changed and untracked paths, so a drill can be NAMED: "something changed" is a lead, "these paths appeared" is a decision |
| `binary_exe`, `binary_exe_sha`, `streaming_lib_sha`, `modules`, `binary_dir`, `binary_built_utc` | the binary the command actually names (its exe plus `llama.dll` and the `ggml*.dll` beside it) |
| `sources_newer_than_binary` + `sources_newer_detail` | the newest changed source vs the NEWEST artifact of the build - the flag that catches "edited but not rebuilt" |
| `origin` | `launch` (wrapper, before the command) / `post-run` (wrapper, after) / `verdict-write` (any other writer) |
| `provenance_delta.source_changed_during_run` / `binary_changed_during_run` | the tree or the binary moved WHILE the arm ran - the same defect class, caught mid-flight |

**Placement is the finding.** The wrapper is not the only writer: the arm drivers write their own verdicts
for their own tags - measured, `gplr53-a4-pin.verdict.json` carries none of the wrapper's fields - and
those ARE the arm verdicts the record cites. So the capture sits in `Write-ArmVerdict`, and the arms'
binaries are found from the run's own `.started` marker, which already records the arm's command. **No
driver needed a change.** A writer that names no binary records `binary_state=not-supplied`: absence is a
state, never a guess. Nothing here blocks an arm: a capture that cannot be taken is recorded as data
(`source_state=not-a-repo` / `git-missing` / `missing` / `error` + text) and the run proceeds unchanged.

**The checker refuses instead of passing silently.** `tools/check-arm-provenance.ps1` re-derives the digest
from the tree now and re-hashes the recorded modules, then applies `Compare-ArmProvenance`:

| exit | verdicts | meaning |
|---|---|---|
| 0 | `CURRENT` | this arm still describes this tree, and the binary it ran is unchanged |
| 1 | `STALE-SOURCES`, `SOURCES-NEWER-THAN-BINARY`, `SOURCES-CHANGED-DURING-RUN`, `BINARY-CHANGED-DURING-RUN`, `STALE-BINARY` | do not quote this arm as current evidence |
| 2 | `NO-PROVENANCE`, `NO-VERDICT`, `CANNOT-EVALUATE`, `BINARY-UNKNOWN` | **NOT A PASS** - cannot certify is not "fine" |

**Proof, host-only, two layers (no GPU, no mutex, no model).**

| layer | evidence |
|---|---|
| `tools/check-arm-provenance.ps1 -Selftest` | **22/22** over throwaway git repos and throwaway binaries; the real tree is never mutated. The measured failure is the central case: build, save a verdict, **edit a source -> STALE-SOURCES (exit 1)**; revert the edit -> `CURRENT` again (content, not mtimes); a NEW untracked source -> STALE; rebuild the engine dll -> STALE-BINARY; a source edited after the build -> SOURCES-NEWER-THAN-BINARY; a mid-run change -> SOURCES-CHANGED-DURING-RUN; a pre-provenance verdict -> `NO-PROVENANCE` exit 2 labelled "NOT A PASS"; a non-repo tree -> CANNOT-EVALUATE |
| `tools/test-arm-verdict.ps1` section 13 | the real-tree half: the arm command resolves to its exe + 5 modules; a verdict written through the shared writer carries the real digest (equal to an independent capture) and `llama.dll 46628B7CDA8FAFA1`; the checker CLI certifies it (**exit 0**); a wrapped stub run records `origin=launch`, a post-run capture and a delta agreeing with both captures; a command naming no binary records `binary_state=not-supplied` |
| host suite | `[arm-provenance]` registered; `tools/run-tests.ps1 -Suite host` exit 0 (`build/suite-provenance.log`) |
| **the mechanism caught a REAL drift while this pass was being written** | the nested tree's digest moved twice inside one hour - `CE58F2F7DCB62B67` -> `C33A0E40BC23F9BF` -> `BC5E1BBE8D0B30A0` - the second and third because another actor's scratch files appeared under a new `.tmp-split/` at 16:48:11. Re-running the rule against the pre-drift digest returned `STALE-SOURCES` and **named them**: `now changed: .tmp-split/n.ccm.diff, .tmp-split/n.cpp.diff, .tmp-split/n.h.diff, ...`. That is the defect the block was written for, observed live and correctly classified |

**The boundary, measured - and what it means for the evidence this record already holds.** Of the **101**
saved verdicts in `build/llama-cuda/verdicts/`: **0 certify (exit 0), 101 cannot-certify (exit 2), 0
stale.** None was written after the mechanism landed, so no saved arm - including the accepted A3/A4 arms
and the 16:28 smoke - can be retro-certified. They are not wrong; they are **uncertifiable**, and the
two claims above that leaned on a binary identity now carry that correction. A re-run is what produces a
certifiable verdict.

**Limits, named.** The digest covers the nested engine tree, which is what produces the binary - the
outer repo's tooling is not hashed, so a driver edit does not invalidate an arm. The mtime relation is
sensitive by design (any artifact rewritten after a source edit flags); the digest relation is not
(mtimes never enter it). A long arm's source state is captured at its start and its end but not
continuously, so an edit that lands and reverts entirely inside one arm leaves the digest unchanged.
**Whole-tree sensitivity cuts both ways, measured:** ANY new file in the nested tree reads as drift,
including scratch - the live `.tmp-split/` event above is exactly that, and it is why the path manifest
exists (a reader can see the drift is scratch in one line). Narrowing the untracked set to
build-relevant paths is a real refinement, but WHICH paths are build inputs is a judgement about this
tree, not a mechanical choice, so it is flagged here for the Planner rather than taken.

**Owner/Planner call, not an implementer's:** nothing wires the checker into an existing gate yet. Its
natural enforcement points are (a) quoting any arm's numbers into this record, and (b) the session
launchers' own post-run summary - both are process decisions, so they are named here rather than taken.
One hygiene item rides along: `docs/currentstate.md` is at 254,486 B (WARN band, 53% of FAIL) and the
convention's split is due before more evidence lands.

**BINDING AT THE LAUNCHERS (same day, 2026-09-22).** A capture nothing requires is a suggestion, so the
requirement now sits where this program's own arms launch. `Assert-ArmProvenance`
(`tools/arm-verdict.ps1`, beside the verdict convention) requires per arm: a `provenance` capture,
`source_state=git` with a 16-hex digest, `binary_state=hashed`, and none of the capture's own anomalies
(`sources_newer_than_binary`, `sources_changed_during_run`, `binary_changed_during_run`). Freshness
against the current tree stays the CHECKER's verdict - printed as a `NOTE` at session end, never a
reason to fail a run whose evidence was sound when it was written. An empty selection REFUSES: a gate
that shrugs there is the silent pass the mechanism exists to remove.

| where | what happens |
|---|---|
| `run-gplr53-a3-arm.ps1`, `run-gplr53-a4-arm.ps1` | gate their own arms at exit; an uncertifiable arm sets **exit 10** (distinct from an arm FAIL `1` and from mutex/busy `124`/`125`). A4's band and every arm field are untouched - the gate re-scores nothing |
| `run-gplr53-session.ps1` | gates each step's tags, passes `-AllowUnprovenanced` through to the drivers, and fails the session with the same `10` before the roll-up's PASS line |
| `tools/smoke-gpu-lock-ctxshell.ps1` | gates its arm BEFORE writing its chain verdict, and carries `arm_source_head` / `arm_source_digest` / `arm_streaming_lib_sha` / `arm_binary_exe_sha` / `arm_provenance_complete` INTO it, so the smoke's result is self-describing |
| escape | `-AllowUnprovenanced`: the run continues and `WAIVED label=... arms=... note=...` is appended to `<verdictDir>/provenance-waivers.log`. Explicit and recorded, never implied by a missing check. **No escape has been taken** (no such file exists) |
| host proof of the refusal path | `tools/test-arm-verdict.ps1` section 14, 16 cases, all PASS: complete verdict passes; capture STRIPPED is refused and named; no binary identity refused; binary-predates-sources refused; a tag with no verdict refused; empty selection refused; the waiver returns true AND writes the log; and a **static** guard asserts all four call sites still contain the gate call, so the wiring cannot be deleted silently (labelled static - a driver cannot be run host-only) |

**A mechanism correction this required, measured.** The staleness flag compared the newest changed file of
ANY kind, so the other actor's scratch `.tmp-split/n.*.diff` (written 16:48:11, later than the 16:32
build) made every arm read `sources_newer_than_binary` and would have refused a smoke that was in fact
fine. The DIGEST keeps whole-tree coverage (nothing hides); the MTIME relation now uses only
compile-relevant paths (`.c .cc .cpp .cxx .cu .cuh .h .hh .hpp .hxx .m .mm .s .asm .inc .cmake .def .rc`,
`CMakeLists.txt`, `Makefile`), recorded as `source_newest_input_path/_utc` beside the informational
`source_newest_change_path/_utc`. That list is the mechanism's one judgement call: a Planner who
disagrees with the boundary changes it and nothing else.

**ONE REAL ARM IS NOW CERTIFIED (2026-09-22, one mutex hold).** The boundary above (0 of 101 verdicts
certifiable) moved by exactly one, on real hardware, through the wrapper and the driver-side capture
path rather than through stubs.

| item | measured |
|---|---|
| window | card held by `deepseek_v4.exe` pid **24104** from 14:56:26Z with the mutex FREE (non-conforming); bounded probe, cleared **15:06:18Z** (sample 14, ~10 min of 45 s samples) |
| run | `tools/smoke-gpu-lock-ctxshell.ps1` **15:06:23Z-15:06:45Z**, one mutex hold, `arm exit 0 after 19.2 s` |
| chain verdict | `build/llama-cuda/gpu-smoke-ctxshell.verdict.json` **PASS** (exit 0): `check-telemetry: OK (1049 events, 4 slots)`, 8 rowset rounds, 320 fetch events, TG text 21 bytes, probes clean |
| arm verdict | `build/llama-cuda/gpu-smoke-arm.verdict.json` **PASS** |
| **gate** | `PROVENANCE GATE: 1 smoke arm verdict(s) checked - 1 complete, 0 refused` - the first real arm to pass it |
| **capture** | `origin=launch`, head **23e5d1b6e617**, digest **BC5E1BBE8D0B30A0** (5 tracked / 7 untracked), `llama-layer-streaming.exe D01874A4235542A8`, `llama.dll 46628B7CDA8FAFA1`; post-capture `origin=post-run`; both deltas false |
| **checker** | `tools/check-arm-provenance.ps1 -Dir build/llama-cuda -Tag gpu-smoke-arm` -> **CURRENT, exit 0**. The first arm on disk whose evidence is tied to a revision |
| rates | PP 133.957 tok/s, TG 15.280 tok/s at 512/8 - the smoke's own header calls it a connectivity check, so these are context only, never a result |
| after | card clean, lock free, no waiver file, no stray process |

**Not done, named:** no historical verdict was retro-certified (forbidden by the task, and the 101 remain
uncertifiable - only a re-run produces a certifiable verdict); the P1 arm set was not started; the gate is
NOT yet wired into any analysis or quoting path, so "don't quote an uncertified arm" remains a process
convention rather than a mechanical one. That last item is now the highest-value next step in this block.

### T-GPLR-57 (P3 host half, LANDED 2026-09-22 - the never-taken/zero distinction for per-part TG timings, and what the recorded traces can actually attribute)

**Status:** landed (host half). **Dependencies:** T-GPLR-13 (row-set TG + its ladder), the F3/H8 emitter
(2026-09-14). **Commit:** unassigned. **GPU half:** named as arms below, not run - the card was held by a
foreign compute process at every probe.

**Goal.** T-GPLR-55's transport attribution needs per-part TG timings as its input. This block establishes
what actually exists for that input, and removes the class of defect that made the ladder's `0/120` read as
"TG compute is free": a timing that was never TAKEN was indistinguishable from a real ZERO.

**Premise correction, measured (the request's premise was half right).** The per-part TG telemetry was
**already landed** - F3/H8, `docs/gplr13-bottlenecks-hypotheses.md` section 3.6, 2026-09-14: the row-set TG
executor (`third_party/llama.cpp/src/llama-layer-streaming.cpp:3327-3406`) times each graph part around the
staging and `tel_layer_compute` (line 5316) appends `attn_compute_us`/`attn_setup_us` (part A = attention +
residual + post-norm + router) and `ffn_compute_us`/`ffn_setup_us` (part B = FFN + shared experts on the
staged rows), with `compute_us`/`setup_us` keeping their exact prior meaning (the A+B sums). So the missing
piece is **not** the emitter. It is that (a) nothing mechanically distinguishes an absent timing from a zero
one, and (b) the ladder rungs the record cites were recorded BEFORE that fix.

**Measured over every trace on disk** (`python tools/check-tg-part-timings.py --census`, 444 traces carrying
`layer_compute`; log `build/tg-parts-census.log`):

| classification | traces | meaning |
|---|---|---|
| phase-1 MEASURED | **31** | the split is live and usable as attribution input |
| phase-1 NOT-TAKEN | **353** | part fields ABSENT: a batch-major TG path (no split by construction) or a pre-2026-09-14 binary |
| PARTS-ALL-ZERO defects | **0** | no trace claims a split and reports nothing - the emitter is correct today |

The row-set-path traces split **31 MEASURED / 64 NOT-TAKEN**: every one of the 64 is pre-fix, including all
four `gplr13-ladder` rungs (`50/512/4k/200k`, mtime 2026-09-13T13:38) - which is exactly the recorded
`0/120`. The first producer of the split is `gplr21-f3-parts/rowset-pp64-tg3-parts.jsonl` (120/120 measured,
2026-09-14T17:30); the newest is `gpu-smoke-ctxshell.jsonl` (320/320, 2026-09-22). **No re-run of the ladder
rungs with the split has ever been recorded.**

**What landed (host-only, telemetry-only, NO engine change).** `tools/check-tg-part-timings.py` derives the
path from the trace's own signals (a trace carrying `rowset_fetch` events ran the two-part row-set
executor), classifies each `(trace, phase)` group as MEASURED / NOT-TAKEN / PARTS-ALL-ZERO / MIXED, and
refuses rather than guessing.

| exit | verdict | meaning |
|---|---|---|
| 0 | MEASURED, or NOT-TAKEN on a path with no split by construction | classified; a batch-major trace says so instead of saying `compute = 0` |
| 1 | PARTS-ALL-ZERO | a defect on ANY path: a group that claims a split and reports all zero. Checked BEFORE the requirement, so a defect is never downgraded to "cannot certify" |
| 2 | a required group is NOT-TAKEN or MIXED | **NOT A PASS** - cannot evaluate is never "fine" |

No engine edit was made, deliberately: the split already exists on the row-set path, and the batch-major TG
path runs ONE fused layer graph, so it has no attn/FFN boundary to time at all - adding a "parts not
measurable" marker field to that record would have broken the byte-identity the F3 amendment preserved on
every non-row-set path (its own note: "Event cardinality is unchanged, so no consumer of the old fields
moves"). The distinction is therefore carried by the CHECKER, from signals the trace already has.

**Validation, all host-only (no GPU, no mutex, no model).** `--selftest` **10/10** on hermetic fixtures: the
recorded ladder shape (120 phase-1 events, `compute_us=0`, part fields absent) reads **NOT-TAKEN** and
refuses; the same 120 events WITH all-zero part fields reads **PARTS-ALL-ZERO** - the two are asserted to
differ, and that assertion is the point of the block; a MEASURED group containing one legitimately zero part
stays MEASURED (a real zero is not a defect); a batch-major trace classifies NOT-TAKEN at exit 0 and refuses
under `--require-parts`; a half-parted group reads MIXED and refuses; and the all-zero defect is exercised on both a row-set and a batch-major trace, so exit 1 is reachable rather than a documented-only code. Registered as `[tg-parts]` in
`tools/run-tests.ps1`; `-Suite host` green (42/42 CTest plus the new block). On the real tree,
`gplr13-ladder/{50,200k}/rowset-*.jsonl` -> REFUSED exit 2, `gplr21-f3-parts/*` and `gpu-smoke-ctxshell` ->
MEASURED exit 0, and `gplr53-a4-pw0` / `gplr56-p1off-4k` -> REFUSED exit 2 (batch-major, no split).

**A1, as far as the traces on disk allow (measured; no performance claim).** Distinct runs: **27** - 4 of
the 31 traces are duplicate copies of another trace (same `run_id` under `tmp-verdict-test/k6` and
`gplr19-honest-k6-ab`). Four TG-stream shapes exist: 120 / 320 / 3840 / 10240 fetches, i.e. 3 / 8 / 96 / 256
TG tokens. Transport (submit + ready-wait) is the majority at every one of them: transfer share **72.9%**
(n=1), **68.5-70.7%** (n=3), **55.5-89.4%** (n=9), **64.6-82.8%** (n=14); part A 5.0-28.3%, part B
2.3-12.0%, setup <= 4%. That spread is the first honesty statement: **a shape is not a measurement** - only a
within-session pair is.

The only same-session, same-shape pair that carries the split is the honest K=6 A/B
(`gplr19-honest-k6-ab` and its `-repro`: 256 TG tokens, 40 layers, 4 slots; A = reactive, B = retained K=6
per the 2026-09-18 entry), four runs per group, expressed per TG token:

| term | A (reactive) | B (retained K=6) | reading |
|---|---|---|---|
| submit | ~30.0 ms | ~23.1 ms | the host-side staging loop is still paid |
| ready-wait (consumer blocking) | **~42.1 ms** | **~7.7 ms** | the fault-driven wait is what retention removes |
| transport total | ~72.1 ms | ~30.8 ms | -57% |
| compute, part A (attn+router) | ~8.4 ms | ~8.0 ms | unchanged within the run spread |
| compute, part B (FFN) | ~5.2 ms | ~4.2 ms | unchanged within the run spread |
| setup (both parts) | ~1.6 ms | ~1.6 ms | unchanged |

So the gain in that pair is a TRANSPORT attribution, not a compute change: ~41 ms/token is removed, and
~34 of those 41 ms come out of the ready-wait. **Cannot be concluded from disk, named rather than implied:**
(i) nothing at any ladder rung - all four are NOT-TAKEN, so no per-part timing exists at 50/512/4k/200k;
(ii) nothing for the transport stack's own arm shapes - `gplr52-a2-{50k,200k}`, every `gplr53-a4-*` arm and
`gplr56-p1off-4k` run batch-major TG, where no split exists by construction, so no attribution is possible
there on any binary; (iii) `gplr24-ha-4096-retention-ARCHIVE.jsonl` is an unexplained outlier at 89.4%
transfer / 18.76 s submit and must not be pooled with its shape; (iv) the smoke's 8-token shape shows the
OPPOSITE signature (ready 344.1 ms vs submit 6.4 ms) and is n=1 - a lead for the first-touch work, not
evidence; (v) these numbers are a re-attribution of arms already banked for a tok/s result, so they do not
add a second independent performance reading.

**Named, not inferred:** `T-GPLR-55` has **no block anywhere in this repository** (the id appears only in
sentences forbidding edits to its shapes - 7 occurrences across the plan and currentstate), so its A1
wording, its arms and its thresholds could not be read and none were invented or altered here. **GPU arms
that would close the gap, both blocked on the shared card** (`nvidia-smi --query-compute-apps` showed
`deepseek_v4.exe` pid 14016 holding compute with the mutex FREE at 15:16:33Z, after probes at 13:57, 14:00,
14:10, 14:13, 14:22, 14:26 and 14:28): **A1-ladder** - re-run each ladder rung at its recorded shape
(`--residency moe-expert-bank --residency-tg --pp-tokens {50,512,4096,200000} --ubatch 64 --tg-tokens 3
--layer-slots 4 --allow-full-residency`; the split-producing producer of that same shape is
`gplr21-f3-parts/rowset-pp64-tg3-parts.jsonl`), which would produce the first per-part TG timings at ladder
lengths; **A2-shape** - one row-set retention/reactive pair at 200k, since the only clean pair on disk is at
256 TG tokens. Neither is claimed as run.

**Escalation.** If the Planner wants the attribution extended to the transport arms themselves (mmap vs
pinned, pre-warm vs not), that is a SHAPE decision, not a telemetry one: those arms must be row-set-shaped,
or a two-graph TG path has to be built for them - a behavior change, and the Planner's call.
### T-GPLR-57 A1-ladder + the marker-after-sink silent loss (same pass, 2026-09-22; the arms were blocked all through the window, THEN RAN once the card cleared - see the addendum at the end of this block)

**The two sessions the task calls for did not run in the window this pass spent first - blocker, measured (they ran later the same day; see the addendum at the end of this block).** A1-ladder's rungs are 512 and
200k, one mutex hold each. The card was held by a foreign compute process at every probe today:
`nvidia-smi --query-compute-apps` showed `deepseek_v4.exe` pid 21208 -> 23744 -> 17352 -> 14016 (13:57,
14:00, 14:10, 14:13, 14:22, 14:26, 14:28, 15:16:33Z) and then pid **5420** for the whole bounded window this
pass (**15:29:42Z** preflight, then 8 samples 15:31:17Z-15:37:17Z, every one `5420
C:\Development\AI\colibri\deepseek-v4-baseline\c\deepseek_v4.exe`, mutex FREE - a non-conforming holder).
No arm was launched, no `-AllowBusy` was passed, and **no result is inferred for either rung**. A separate
`llama-layer-streaming` arm did run in this shared checkout at **15:21-15:23Z** (verdict
`build/llama-cuda/verdicts/llama-layer-streaming.verdict.json`, PASS, 133.7 s, batch-major `gplr07-qa-512`
shape, `gpu_pre_run`/`gpu_post_run` clean) - **not this pass's run**; its `fields` carry only the four
pre-provenance keys, so it was written by an older tools revision than the one on disk now, and it is
recorded here only so nobody reads it as this session's evidence.

**The launcher is landed and host-proven** (`tools/run-gplr57-a1-ladder.ps1`), so the sessions cost one
command and ~3 minutes of holds when the card frees: recorded walls for these very rungs are **512 = 12 s
and 200k = 2.6 min** (`build/llama-cuda/gplr13-ladder/driver.log`), against lock timeouts of 2400 s and
7600 s. It takes ONE rung per invocation (one hold), re-runs the rung at its RECORDED shape and reuses the
recorded anchor prompt byte-for-byte (`gplr13-ladder/{512/anchor-512.txt,
200k/anchor-200000.txt}`, sha256 recorded in the chain verdict), writes into a per-run directory
(`build/llama-cuda/runs/<utc>-gplr57-a1-<rung>/`) so a re-run cannot overwrite the previous evidence, gates
its own arm with `Assert-ArmProvenance` (exit 10; `-AllowUnprovenanced` is the only escape and it writes a
`provenance-waivers.log` line), and then requires BOTH `--check-telemetry` and
`check-tg-part-timings.py --require-parts` to exit 0 before it can report PASS.

| host proof | result |
|---|---|
| parse | OK |
| `-PreflightOnly` against the BUSY card | **refuses in <1 s with exit 125**, naming `5420 deepseek_v4.exe`; starts no arm |
| `-PreflightOnly` with `GPU_BUSY_PROBE_CMD` stubbed clean | exit 0, "nothing launched" (the clean path, exercised without touching the card) |

**The audit's remaining correctness gap is fixed: a marker emitted after the sink opened was silently lost.**
`llama_ls_tel_mark` pushed such a marker into `g_tel_early`, and the only production flush is the one at
sink-open (`llama-layer-streaming.cpp:2267`), so nothing ever reached it again: no error, no flag, no line.
Fixed telemetry-only, with **no stashed `FILE*`** - the first design I wrote did stash the sink and write
late markers live, and the shared host test's own first case (open a sink, flush it, `fclose`, then mark)
would then have written to a CLOSED FILE, which is undefined behaviour; the test caught that, and the
design was replaced by one emission path:

| change | why |
|---|---|
| the three close sites (`teardown:1879`, `run_end:4721`, `refused:4764`) now call `llama_ls_tel_flush_early(tel_out)` right BEFORE their `fclose` | a marker taken after the sink opened is written at close instead of dying in the buffer; this is the whole fix |
| `tel_early_rec.late` + `g_tel_early_sink_seen` | a late line must say so, or it is indistinguishable from a second emission of the same event name (`load_end` twice) |
| `seq` continues across flushes | call order stays provable across both phases |
| shape of every other line is UNCHANGED (`pre_run:1`, no `run_id`, same field set/order) | the documented pre-sink shape and the F3-style byte-identity are preserved; `late:1` is additive and appears only on a line that did not exist before |

**Validation, host-only (no GPU, no mutex, no model).** `test-layer-streaming-tel-mark` extended with the
post-flush case: **ALL CHECKS PASSED**, `ctest -C Release -R layer-streaming-tel-mark` PASS. The new case
asserts a marker taken after a flushed sink is WRITTEN (not lost), carries `late:1`, keeps `pre_run` and
carries no `run_id`, and that a mark taken with no sink is still buffered and emitted by the next flush.
The four pre-existing cases (null-flush drop, the three-marker shape and order, the empty/NULL-name
refusal) still pass unchanged - which is what makes the addition additive rather than a redefinition.

**This pass rebuilt the engine, and the provenance mechanism said so - measured.** `llama.dll`
**46628B7CDA8FAFA1 -> 261AA683E10E97AA** (exe unchanged, `D01874A4235542A8`).
`check-arm-provenance.ps1 -Verdict build/llama-cuda/gpu-smoke-arm.verdict.json` now reports
**STALE-SOURCES (exit 1)**: digest recorded `BC5E1BBE8D0B30A0` vs now `9399BF96289C63E1`, and "module
llama.dll was REBUILT since this arm". So the one certified arm on disk is **no longer quotable as current
evidence** - the exact class this mechanism exists to catch, catching this pass's own change.

**Limits, stated rather than implied.** (i) A late marker is written by the NEXT flush, i.e. at the close
of the sink, so its LINE POSITION is not chronological - its `t_us` is, and that is what a reader must use;
whether to emit late markers at their point of occurrence is a design question this fix does not take.
(ii) With no sink at all there is no telemetry file to write to, so those markers are still buffer-only.
(iii) The rungs were blocked for the whole window this pass tried and then ran once the card cleared, so
**A1 now has per-part input at two ladder lengths - see the addendum below**, which is the ladder-length
attribution; the earlier T-GPLR-57 block's reading (banked arms, one same-session pair) stands as a separate
re-attribution and is still the only same-session transport-lever comparison on disk. No latency or
throughput claim is made anywhere in this block.
**Addendum (same day, the card cleared at ~15:42Z): BOTH RUNGS RAN - this is the A1-ladder evidence, and it
supersedes the "not run" status above.**

The bounded window this pass spent first was refused as recorded (8 samples 15:31:17Z-15:37:17Z, every one
pid **5420**, mutex FREE). The card then cleared, and both rungs were launched - one mutex hold each, at
their recorded shapes with their recorded anchor prompts, each into its own per-run directory.

| rung | run dir (UTC) | arm | chain | provenance | waiver | per-part guard |
|---|---|---|---|---|---|---|
| 512 | `runs/20260922T154620Z-gplr57-a1-512` | PASS, 10 s | **PASS exit 0** | digest `9399BF96289C63E1`, `llama.dll` `261AA683E10E97AA`, exe `D01874A4235542A8`, `sources_newer_than_binary=false` | **none taken** | `--require-parts` exit 0 -> **MEASURED** |
| 200k | `runs/20260922T154633Z-gplr57-a1-200k` | PASS, 147.7 s | **PASS exit 0** | same digest / same `llama.dll` / same exe; `provenance_delta` = no source or binary change DURING the run | **none taken** | `--require-parts` exit 0 -> **MEASURED** |

Re-verified from the artifacts rather than from the drivers' own output, because this record has carried
false claims before: `check-arm-provenance.ps1` reports **CURRENT (exit 0)** on all four verdicts (both arms
and both chains) against the tree as it stands now; `--check-telemetry` exits 0 on both traces; 120/120
phase-1 `layer_compute` events carry the part split; the wrapper fields read `gpu_pre_run=gpu_post_run=clean`
with `mutex_acquired=true` and `gpu_probe_source=nvidia-smi`; and **no waiver log exists at any level** (no
`provenance-waivers.log` in the repo root or in any run directory). The recorded rungs these replace measure
`phase1=120, with-parts=0, compute_us_sum=0.0 ms` - the recorded `0/120` reproduced exactly - so these two
runs are the first ladder-rung per-part input that exists.

**A1's reading at these two shapes (measured; no performance claim; 3 TG tokens per rung = a COLD window).**

| rung | PP wall | TG wall | TG tok/s | transfer | ready-latency | compute | part A (attn+router) | part B (FFN on staged rows) |
|---|---|---|---|---|---|---|---|---|
| 512 | 3,423.5 ms (149.56 tok/s) | **212.7 ms** | 14.10 | 131.8 ms (**62.0%** of TG wall) | 129.4 ms | 67.1 ms (31.5%) | 43.6 ms (20.5%) | 23.4 ms (11.0%) |
| 200k | 139,463.9 ms (1434.06 tok/s) | **243.5 ms** | 12.32 | 132.6 ms (**54.5%** of TG wall) | 130.1 ms | 95.1 ms (39.1%) | 71.9 ms (29.5%) | 23.2 ms (9.5%) |

Per TG token: ready-latency **43.1 vs 43.4 ms**, part A **14.55 -> 23.96 ms**, part B **7.81 vs 7.74 ms**.
`total_compute_us` equals part A plus part B exactly (67.074 ms vs 43.6 + 23.4), so the compute term is
internally consistent rather than assembled from two sources. Transport is **131.8 vs 132.6 ms - flat across
a 390x longer prefill**: 120 fetches of the same 8 experts x 1.77 MB = 14.16 MB per fetch, i.e. 573 MB/token
at **~13.0 GB/s** effective.

**What it supports.** The TG transport term is context-invariant, and the context-dependent growth is
**COMPUTE, entirely inside part A** (attention + router): across 512 -> 200k, part A gains **+9.4 ms/token**
while part B - the FFN consuming the freshly staged expert rows - is flat to 0.07 ms/token and holds ~10% of
the wall. The transport SHARE falls (62.0% -> 54.5%) only because part A grows. In the PP leg the same
transfer is overlapped (transfer 4610 ms against a 3423 ms wall at 512; 379,687 ms against 139,464 ms at
200k), whereas the TG leg's ready-latency is 98% of its transfer time - so the blocking is a TG-leg property,
not a staging-cost property. **QUALIFIED by the T-GPLR-55 decision below:** these rungs are the ONLY row-set
arms on disk whose row fetches are SERIALIZED (`ready_latency/transfer_us` = 0.982); the banked row-set arms
overlap theirs (0.067-0.598) and their per-token transport is 4-10x smaller, so the 62% share is a property
of the rungs' reactive / no-prefetch configuration, NOT of the path. What survives unqualified is the copy
rate itself: 13.03 GB/s (512) vs 12.96 GB/s (200k), a 0.5% spread.

**What the pair cannot distinguish.** (i) Three TG tokens is the first tokens after a cold prefill, and no
ladder shape on disk carries a token-index trend, so **no per-token TG rate at ladder lengths may be quoted
from this** - it is a shape attribution, not a rate. (ii) It therefore cannot separate a context-length
effect on transport from a warmup effect. (iii) It says nothing about any transport lever: no
mmap/pinned/pre-warm arm exists at ladder shapes, and the batch-major shapes those arms run on cannot carry
the split at all (the guard refuses them, by design). (iv) The 512-vs-200k part-A difference is confounded
with KV length AND with sample size, so no attention cost model is claimed from it. (v) No comparison against
the pre-fix rungs is made: those carried no compute term to compare against, and today's binary also differs
(the marker fix landed this pass).

**Two defects in this pass's OWN launcher, found and fixed in-pass - both the class the audits keep finding.**
(1) The gate step used `& exe *> file`, and PowerShell 5.1 turns a native command's STDERR into a
TERMINATING error under `ErrorActionPreference='Stop'` - the engine prints `rowset_round` to stderr, so the
FIRST rung PASSED and the launcher then died inside its own gate step, writing no chain verdict at all. Both
gate calls now run under `cmd.exe /d /s /c`, with cmd owning the redirection. (2)
`Write-ArmVerdict -ExitCode ([int]($status -eq 'PASS'))` records **1** for a PASSING run, and the writer
derives status FROM the exit code, so two chain verdicts read **FAIL while their own arms read PASS**; the
integer is now passed rather than the truth value. Superseded run directories, named so nobody cites them:
`runs/20260922T154241Z-gplr57-a1-512` (no chain verdict - died in the gate step),
`runs/20260922T154311Z-gplr57-a1-512` and `runs/20260922T154328Z-gplr57-a1-200k` (complete arms, chain
verdict mislabelled FAIL by the inverted exit code). Because the rung directories are per-run by
construction, this cost no other arm's evidence.
### T-GPLR-55 (decision, 2026-09-22, host-only, no GPU spent): the transport-remainder premise is REFUTED

**The premise as drafted:** the row-set transport ships at **7.3-7.4 GB/s** and there is a residual to
attribute to it. Decided from the traces and verdicts already on disk - 453 traces carry `copy_complete`,
450 usable, plus all 100 row-set traces with at least 20 `rowset_fetch` events. No arm was run, no mutex was
taken, and no threshold or drafted shape was touched.

**Finding 1, and it decides most of the question: `transfer_us` is a LATENCY, not a duration.** For a copy
event, `transfer_us` is exactly `copy_complete.t_us - copy_submit.t_us`, and `ready_latency_us` records how
much of that any consumer waited for. So `bytes / transfer_us` is a copy's **speed** only when the copy is
**serialized** (`ready_latency / transfer_us ~ 1`); when many copies are in flight it is a **queue-inclusive
latency** and it understates the speed. `ready_latency / transfer_us` is therefore the discriminator, and it
is available on every arm.

**Finding 2: every "transport GB/s" in this repository is one of FOUR different quantities.** They have been
read as one number, and that is what produced the premise.

| quantity | how it is formed | measured | arms |
|---|---|---|---|
| **copy link rate** (serialized) | bytes / latency, ready/transfer ~1 | **13.03 GB/s** (512 rung) and **12.96 GB/s** (200k rung), 13.65 MiB per fetch; **14.34 GB/s** for ONE 347.2 MiB full-bank copy at ratio **0.999** | `runs/20260922T154620Z-...`, `runs/20260922T154633Z-...`, `q38ls-lm-boundary.jsonl` |
| **queued copies** | bytes / latency, ready/transfer 0.004-0.62 | 9.6-9.8 GB/s (`direct`, 3880-19480 copies at ratio ~0.005), **7.37** (`gplr10-mem-m1`), **7.58** (`gplr24-hd-ab/hd-pinned`) | the batch-major arms - **this is where 7.3-7.4 GB/s came from** |
| **first-touch mmap-registered copies** | bytes / latency, ratio 1.000 but ~3.3 s per 436.95 MiB copy | **0.07-0.14 GB/s** | `gplr37-*`, `gplr41-*`, `gplr42-*`, `gplr48-a2-*` - the page-lock/residency commit P0 already closed |
| **wall rate** | bytes / WALL | 4.4-5.8 GB/s (row-set), 3.2-3.3 GB/s (whole-bank) | `docs/gplr13-bottlenecks-hypotheses.md` sections 3.7, H5, and the H1 estimate |

**Verdict: the premise is REFUTED, and no residual is left to attribute at that level.** On the only path
where a clean (serialized) copy measurement exists, the staged expert rows move at **13.0 GB/s, identical at
512 and 200k** (13.03 vs 12.96 - a 0.5% spread across a 390x longer prefill), and a serialized full-bank copy
of 347.2 MiB measures **14.34 GB/s**. Those are one link class, ~13-14 GB/s. There is no 7.4 GB/s transport
ceiling with a recoverable remainder above it: the 7.3-7.4 family is a **queue-inclusive latency on queued
copies**, the 4.4-5.8 family is a **wall rate**, and the 0.07-0.14 family is the **page-commit** already
attributed by P0. The context-dependent TG growth is not staging at all - it is attention + router (part A),
as the A1-ladder addendum above measures.

**Withdrawn, marked in place so a reader cannot be misled.** (i) `docs/gplr13-bottlenecks-hypotheses.md`
sections 3.7 / H5 / H1 carry a superseding note: their 4.4-5.8 and 5.4-5.8 GB/s are WALL rates and must not
be read as transport speeds. (ii) The A1-ladder addendum's transport share (62% of TG wall) is qualified: the
gplr57 rungs are the ONLY row-set arms on disk whose row copies are serialized (ready/transfer 0.98), while
the banked row-set arms overlap them (0.067-0.598, and their per-token transport is 4-10x smaller) - so that
share is a property of the rungs' reactive/no-prefetch configuration, not of the path. A row-set trace with
`ready/transfer < 0.9` has no per-copy speed in it at all.

**What genuinely remains (the re-scoped task).** The copy speed itself is answered. What is not answered is
the **source** effect on the row-set path: identical 13.65 MiB row fetches measure 13.03 GB/s
(`mmap-registered`, the rungs), 9.16 (`direct`, `gplr07-qa-4096-reactive`) and 7.85 (`pinned`,
`gplr19-honest-k6-ab/arm-A1`) - but those arms DIFFER in overlap ratio (0.982 / 0.067 / 0.578) and in
retention/prefetch configuration, so the three numbers are confounded and none of the three may be credited
as a source comparison.

- **The single arm that would settle it:** one row-set TG arm at ONE ladder shape, in the rungs' serialized
  configuration (reactive, no prefetch window, so `ready/transfer ~ 1` by construction), run once per
  `--copy-source` value - `mmap-registered` (already have: 13.03 GB/s) vs `pinned` vs `direct`. That puts all
  three sources on the same quantity for the first time, at the same fetch size, with no queue in the way. It
  is 2 runs and settles whether the source is worth a lever. Its precondition is the same as the rungs'
  (`--pp-recon 1`), and it names its own quantity (serialized copy rate) rather than inheriting A4's band.
- **Named tooling gap, not built here:** nothing on disk refuses to call a queued copy set a "rate". A guard
  that reports a rate ONLY when `ready_latency / transfer_us >= 0.9` (and otherwise says `QUEUED-LOWER-BOUND`)
  would have prevented this premise from being drafted. Recorded as a candidate; it is a new tool, so it is
  the Planner's call, not an implementer's.

**Evidence:** `build/llama-cuda/runs/20260922T154620Z-gplr57-a1-512/`, `...154633Z-gplr57-a1-200k/`,
`build/llama-cuda/q38ls-lm-boundary.jsonl`, `build/llama-cuda/gplr10-mem-m1.jsonl`,
`build/llama-cuda/gplr24-hd-ab/hd-pinned.jsonl`, `build/llama-cuda/gplr07-qa-4096-reactive.jsonl`,
`build/llama-cuda/gplr19-honest-k6-ab/arm-A1.jsonl`, `build/llama-cuda/gplr37-a4-b1-default.jsonl`, and the
census over all 450 rate-bearing traces. No latency or throughput win is claimed from any of it.
### T-GPLR-55 re-scope RUN (2026-09-22, one mutex hold per arm): the copy-source A/B is DECIDED - the source is NOT a bandwidth lever, it selects the submission mode, so only ONE of the three rates is a rate

**What was run.** The re-scoped question had one arm: same shape, same config, one run per remaining
`--copy-source` value, so the three sources sit on the same quantity at the same fetch size. Two arms, one
mutex hold each, at the rung's recorded shape (`-Rung 512 -CopySource pinned` 15:57:38Z-15:57:53Z and
`-CopySource direct` 15:57:57Z-15:58:12Z; the mmap-registered rung 15:46:20Z is the third).

| arm | run dir (UTC) | arm | chain | provenance | waiver | per-part |
|---|---|---|---|---|---|---|
| 512 **pinned** | `runs/20260922T155738Z-gplr57-a1-512` | PASS exit 0 | **PASS exit 0** | **CURRENT exit 0** - digest `9399BF96289C63E1`, `llama.dll` `261AA683E10E97AA`, exe `D01874A4235542A8` | **none taken** | MEASURED |
| 512 **direct** | `runs/20260922T155757Z-gplr57-a1-512` | PASS exit 0 | **PASS exit 0** | **CURRENT exit 0** - same digest / lib / exe | **none taken** | MEASURED |

Both traces read `gpu_pre_run=gpu_post_run=clean`, `copy_source`/`staging_source` matched the request
(`staging_source_match=1`), and the shape is identical across all three: 120 fetches of **13.65 MiB**, and TG
compute **67.1 / 68.2 / 68.0 ms** (mmap-registered / pinned / direct) - so the model work did not move and the
comparison is at the same fetch size. **No waiver log exists anywhere**, and the card was left free.

**The result, and it is not the expected one: the source changes the CONCURRENCY, not just the bytes.**

| source | `ready_latency/transfer_us` | bytes / transfer_us | is that a copy RATE? |
|---|---|---|---|
| **mmap-registered** | **0.982** (consumer waits the whole copy) | **13.03 GB/s** (per-fetch median 13.06; range 12.54-13.37) | **YES - the only one** |
| pinned | 0.597 | 8.14 GB/s | **no** - queue-inclusive lower bound |
| direct | 0.067 | 8.91 GB/s | **no** - queue-inclusive lower bound |

The mechanism is visible in the event census: `--copy-source mmap-registered` is the only value that
**registers the mapping** (40 `mmap_register` + 1 `mmap_register_init`; pinned and direct emit **0/0**) and the
only one whose consumer waits for the copy to land. Pinned and direct run the same 40 `copy_submit` /
`copy_complete` and the same 36x3 `async_feed_*` pairs but overlap the row copies instead of blocking on them.
So the field selects a staging PATH (registered/synchronous vs unregistered/async), and bytes/latency is a
rate only on the first.

**Verdict: the three sources are NOT distinguishable as bandwidth at the same fetch size - and they never
were.** The earlier figures reproduce under the new arms and are explained by them:

| source | earlier arm (recorded) | its ratio | new arm, source forced, same config | its ratio |
|---|---|---|---|---|
| direct | 9.16 GB/s (`gplr07-qa-4096-reactive`) | 0.067 | **8.91 GB/s** | **0.067** |
| pinned | 7.85 GB/s (`gplr19-honest-k6-ab/arm-A1`) | 0.578 | **8.14 GB/s** | **0.597** |

The ratio reproduces to three decimals and the value to 3-4%, on different arms and different days: those
numbers are governed by the **overlap mode**, not by any intrinsic bandwidth of the source. The "confound"
recorded in the decision block is therefore **not a nuisance to be controlled away - it is the mechanism**.
`direct`'s 8.91 GB/s is a **lower bound** and may be at or above 13.03 GB/s: with copies overlapped in time,
summing their latencies double-counts the same wall interval, which is exactly why a queued set cannot yield a
rate. **The single genuinely measured transport quantity for this path remains 13.03 GB/s, for
mmap-registered, serialized.** Nothing in this block credits 8.14 or 8.91 as a speed, and the earlier
13.03/9.16/7.85 comparison is recorded as withdrawn rather than resolved.

**What would actually settle a source comparison (named, NOT built - it is a different lever):** hold the
overlap fixed and vary the source, i.e. a serialization / prefetch-depth control applied to the pinned and
direct paths so they too reach `ready_latency/transfer_us ~ 1`. Copy-source alone cannot do it, because the
source value IS what moves the ratio. That is a Planner call, as is the rate guard named in the decision block.

**Launcher change (this pass, additive and default-neutral):** `tools/run-gplr57-a1-ladder.ps1` gained
`-CopySource <mmap-registered|pinned|direct>`. The default is EMPTY, which reproduces the recorded rung
exactly, so every existing tag, shape and piece of evidence is untouched; a source run gets a tag suffix
(`gplr57-a1-512-pinned`) and its own per-run directory. Host-validated before spending a run: parse OK, an
invalid value is refused by parameter validation with nothing launched, `-PreflightOnly` on a free card exits
0 having launched nothing, and the arm's own `.started` marker and `run_start` record the requested source.

**No performance claim.** The wall/tok-s spread across the three arms (212.7 / 274.9 / 294.7 ms; 14.10 / 10.91
/ 10.18 tok/s) is **n=1 at 3 cold TG tokens**, differing in both overlap mode and staging path, and is
recorded as context only - it is not a measurement of steady state and nothing is attributed to it.

**Evidence:** `build/llama-cuda/runs/20260922T155738Z-gplr57-a1-512/`,
`build/llama-cuda/runs/20260922T155757Z-gplr57-a1-512/`,
`build/llama-cuda/runs/20260922T154620Z-gplr57-a1-512/`, `tools/run-gplr57-a1-ladder.ps1`.
### Arm provenance: the gate moved to the CHOKE POINT, and a missing checker now FAILS the suite (2026-09-22, host-only, no GPU spent)

**The defect, as the audit found it: a gate that was not a gate.** Capture was already systematic (the verdict
writer records it on every path), but ENFORCEMENT lived in four callers - the A3 arm, the A4 arm, the GPLR-53
session and the GPLR-57 ladder - plus the smoke. The wrapper that every arm actually runs under recorded a
capture and asserted nothing, so a driver that forgot its own gate shipped unattestable evidence SILENTLY, and
the previous block's phrase "binding at the launchers" was narrower than it read. The same shape appeared in
the suite: all twelve host-suite blocks did `if (Test-Path <tool>) { run } else { WARNING ... skipped }`, so a
renamed or deleted checker REDUCED COVERAGE while the suite printed SUCCESS.

**Half 1 - the assertion is now inside `tools/with-gpu-lock.ps1`.**

| property | how it is preserved |
|---|---|
| same rule as the callers | the wrapper calls the same `Assert-ArmProvenance` with the same failure taxonomy; an unattestable verdict is refused, never waved through |
| same escape | `-AllowUnprovenanced` was ADDED to the wrapper and is FORWARDED by the ladder and the smoke, so the documented waiver reaches the gate instead of the caller and the wrapper disagreeing about one run (the waiver line is written by `Assert-ArmProvenance` itself) |
| taxonomy preserved | 125 = GPU busy, 124 = mutex timeout, the command's own code = the command's result, and a NEW 10 = the run happened but cannot be attributed to a revision |
| verdict agrees with the exit code | on refusal the verdict is REWRITTEN at exit 10 rather than left claiming PASS beside a refusal - the same contradiction class this repo recorded once before (an inverted exit code once made a chain verdict say FAIL while its arm said PASS); the command's own code is preserved as `command_exit` and the refusal is flagged `refused_unprovenanced` |
| only runs that EXECUTED are gated | a busy refusal (125) and a mutex timeout (124) describe the WRAPPER's decision, not the arm. MEASURED DEFECT IN MY OWN FIRST VERSION: without that scope the gate rewrote 125 and 124 into 10, destroying the taxonomy the task told me to preserve - the harness caught it (10 cases turned red), and `$commandRan` now scopes the gate |

**Half 2 - a missing checker FAILS the suite.** All twelve "not found; ... skipped" sites now print `FAILED ...
this guard is REQUIRED. A missing checker is a FAILED check, not a skipped one; this suite is green only if
every guard RAN.` and `exit 1`.

**Host-side proof, both halves, no GPU and no shared mutex** (private `...-selftest-*` mutex names; the fresh
`[gpu-lock]` block runs the harness in every host suite).

| proof | result |
|---|---|
| `test-arm-verdict.ps1` section 15 (new, 8 cases) - a throwaway copy of the three tools at a temp root with NO engine tree, so the capture cannot name a revision | **(a)** the WRAPPER refuses **exit 10** - not only its caller; **(b)** the verdict AGREES (`status=FAIL`); **(c)** `command_exit=0` is preserved, not lost; **(d)** the refusal is flagged in the verdict; **(e)** it NAMES what could not be attested; **(f)** `-AllowUnprovenanced` still passes AT THE WRAPPER (exit 0); **(g)** the waiver is RECORDED; **(h)** a waived run reads PASS and the waiver file names the arm |
| a suite run with a checker deliberately HIDDEN (`check-doc-sizes.py` moved aside for one run, then restored with sha256 verified identical) | `[doc-size] FAILED: ... not found - this guard is REQUIRED` and **suite exit 1** - previously this printed a WARNING and the suite reported SUCCESS |
| the whole harness after the change | **ALL ARM-VERDICT CASES PASS** (was 10 failing before the `$commandRan` scope was added) |
| the normal path, all checkers present | `-Suite host` **exit 0, 42/42 CTest, all 12 blocks ran, none skipped** (`build/host-suite-after-gate.log`) |

**One trap worth recording (measured, cost one red harness):** splatting a HASHTABLE onto a NATIVE command
(`& powershell -File x.ps1 @a`) expands every entry as `-Key value`, so a switch entry becomes the literal
positional argument `True` and the child fails to bind it. Splatting onto a SCRIPT binds switches correctly.
Section 15 now uses an explicit argument array with a bare `-AllowUnprovenanced` token.

**Contract change to note for future arms (not a defect):** because the gate now runs at the wrapper, a wrapper
run whose command names NO binary (a bare `exit 0`, a shell fragment) is refused. Every real arm names its exe
and is unaffected; the harness's own contrived stub cases carry the explicit recorded waiver, and section 15
is where the refusal itself is proven.

**Residual gap, stated rather than implied:** the A3/A4 drivers do NOT use the wrapper - they take the named
mutex themselves - so "the choke point" is the wrapper for the ladder, the smoke and every older runner, while
those two drivers enforce via their own `Assert-ArmProvenance` calls (verified by the harness's `[static]`
cases). A FUTURE driver that takes the mutex itself and forgets the gate would bypass both. Making
`Write-ArmVerdict` refuse to write an unattested PASS would close that last door, but it changes the writer's
documented "a capture failure still writes a verdict" behaviour, so it is the Planner's call, not this pass's.

**No verdict's evidence changed.** The refused-verdict rewrite and the new `command_exit`/
`refused_unprovenanced` fields exist only for runs the gate refuses; the passing path writes the same fields it
wrote before, and no arm's recorded evidence was touched. No latency or throughput claim is made.
### Arm provenance: WHICH paths can still record evidence without a gate - inventory, per-path decision, and the rule that keeps the list from growing (2026-09-22, host-only, no GPU spent)

**The inventory.** 28 files in `tools/` reference `Write-ArmVerdict`; four of them are tooling (the writer
itself, the checker, the harness, and the wrapper - the last of which now calls the gate), leaving **24 arm
runners**. Each was classified by whether it calls `Assert-ArmProvenance` - the stricter test than "does it
invoke the wrapper", because a runner can use the wrapper and STILL write an ungated verdict of its own.

| path | type | decision |
|---|---|---|
| `with-gpu-lock.ps1` | the shared wrapper | **GATED** this session - every arm that runs through it is asserted at the choke point (proven in section 15) |
| `run-gplr53-a3-arm.ps1`, `run-gplr53-a4-arm.ps1` | live (P0/P3 arms; take the named mutex themselves) | **GATED** by their own `Assert-ArmProvenance` call, exit 10 propagated (harness `[static]` cases) |
| `run-gplr53-session.ps1` | live (the documented session entry point) | **GATED** per step, forwards `-AllowUnprovenanced` to its drivers |
| `run-gplr57-a1-ladder.ps1` | live (A1-ladder) | **GATED** at the wrapper (it forwards the waiver) and at its own gate |
| `smoke-gpu-lock-ctxshell.ps1` | live (smoke regression) | **GATED** at the wrapper and at its own gate before writing its chain verdict |
| **20 historical runners** (named below) | completed tasks T-GPLR-19..52 | **NAMED GAP, not rewritten** - see the reason |

**The 20, by name, with the reason.** `run-gplr19-rowcache-ab`, `run-gplr36-remeasure-200k`, `run-gplr37-arm`,
`run-gplr37-set`, `run-gplr39-arm`, `run-gplr44-ab-50k`, `run-gplr44-dump-50k`, `run-gplr45-arm`,
`run-gplr46-rebuild-ab`, `run-gplr47-arm`, `run-gplr48-arm`, `run-gplr49-arm`, `run-gplr49b-echo-attrib`,
`run-gplr50-200k-pair`, `run-gplr50-arm`, `run-gplr51-a2-pinned`, `run-gplr51-s2-arm`, `run-gplr51-tail-arm`,
`run-gplr52-a2-arm`, `run-gplr52-a3-arm`.

They are **historical reproducers for tasks that are closed**: none is referenced by the active plan or by
`tools/run-tests.ps1` (checked: 0 hits for each sampled name), and 18 of the 20 take the shared mutex
themselves, so the wrapper they do not use cannot gate them. Rewriting 20 completed runners is out of scope by
instruction and would put churn through files whose evidence is already recorded. **The honest residual is
therefore stated rather than fixed: these 20 can write a verdict that no gate ever checked** - and because the
writer now captures provenance automatically, such a verdict would look certifiable. That is the gap.

**Two of the 20 were found by the census and not by the first hand-scan** - `run-gplr19-rowcache-ab.ps1` and
`run-gplr44-dump-50k.ps1` - precisely because they *do* invoke the wrapper while writing their own verdict, so
"uses the wrapper" would have marked them gated when their own verdict is not. The census uses the stricter
rule for that reason.

**The rule that keeps the list from growing (durable, and in the suite).** `test-arm-verdict.ps1` section 16
walks every `tools/*.ps1` that writes a verdict and requires it to call the gate, with the 20 historical names
as an explicit dated allow-list plus three tooling exemptions (the writer, the checker whose verdict records a
CHECK rather than an arm, and the harness). Four cases:

| case | result |
|---|---|
| (a) no ungated verdict writer exists beyond the named historical set | PASS (unexpected: none) |
| (b) the named set is not stale - a writer that becomes gated must leave the list | PASS (0 stale) |
| (c) a NEW ungated writer is caught - proven hermetically on a temp dir containing a fake `run-gplr99-new.ps1` | PASS (caught: `run-gplr99-new.ps1`) |
| (d) a gated writer and a no-evidence runner are both correctly ignored | PASS |

So the 19th ungated runner cannot join silently: it fails the harness in every host suite run.

**What the enforcement claim may now say, exactly.** Provenance is binding (i) at the wrapper, which every arm
that uses it passes through, (ii) at the four live call sites that manage the mutex themselves
(`run-gplr53-a3-arm`, `run-gplr53-a4-arm`, `run-gplr53-session`, and the smoke), and (iii) for the ladder, at
both. It is **NOT binding for the 20 historical runners named above**, and that list is tested rather than
assumed. Anything stronger than that - for instance "every arm in this repository is gated" - would be false.

**Residual, and the structural fix if it is wanted:** making `Write-ArmVerdict` refuse to write an unattested
PASS closes the last door for all 20 at once, but it changes the writer's documented behaviour that a capture
failure still writes a verdict (the harness asserts that behaviour today). That is the Planner's call, and it
is recorded here rather than taken.

**Evidence:** `tools/test-arm-verdict.ps1` sections 15 and 16 (host-only, private mutex names, no GPU work),
`tools/with-gpu-lock.ps1`, and the census over `tools/*.ps1`. No latency or throughput claim is made.
### Arm provenance: the WRITER refuses to leave an unattested PASS (2026-09-22, host-only, no GPU spent)

**What this closes, exactly.** The inventory left 20 historical runners that never call the gate. Because the
verdict writer captures provenance automatically, a verdict from one of them could read PASS while naming a
tree and binary - and, in the worse case the rule targets, could read PASS while naming *nothing usable*. The
rule now lives at the one place every verdict is created:

| property | how |
|---|---|
| one owner for the rule | the per-capture test moved into **`Get-AttestationFailure`**, used by BOTH the gate that refuses a run and the writer that records one, so the two can no longer disagree about what "can attest" means (the gate's inline copy is gone) |
| an unattested PASS is never recorded | if the status would be PASS and the capture fails the test, the verdict is written **`UNCERTIFIED` with `exit_code` 10** - the SAME contract the live gates use - plus `uncertified=true`, `would_be_status='PASS'`, `attestation_reasons`, and the command's own code as `command_exit` |
| the evidence is kept, not dropped | the files stay on disk with the capture intact; only the status changes. The `.done` sentinel is written from the same status/code, so a reader of either agrees |
| one name for one condition | status derivation now maps exit 10 to **`UNCERTIFIED`**, so a refusal reads the same at the writer, at the wrapper and at a launcher instead of FAIL in one place and uncertified in another |
| the escape, preserved and singular | `-AllowUnattested` is the writer-level opt-out, and the wrapper passes it FROM the operator's existing `-AllowUnprovenanced` - one escape at the CLI, honoured by both layers, or a waived run would still be recorded uncertified and the documented escape would be half a lie |
| runs with nothing to attest still work | a tool's own bookkeeping verdict or a harness fixture opts out explicitly; nothing was weakened to accommodate them |

**Proof - `test-arm-verdict.ps1` section 17, nine cases, host-only, no GPU, no shared mutex**

| case | result |
|---|---|
| (a) an unattested run that WOULD end PASS | **recorded `UNCERTIFIED`, not PASS** |
| (b) the same exit contract the live gates use | **exit 10** |
| (c) preserved | `would_be_status=PASS` and `command_exit=0` |
| (d) reasons named | `no binary identity was captured (binary_state=not-supplied)` |
| (e) a genuine CAPTURE FAILURE still produces a verdict that says so | the verdict exists, the capture is present, and the reason names the failure - evidence kept, degraded, named |
| (f) the sentinel agrees | `UNCERTIFIED 10 <ts> unattested` |
| (g)/(h) an ATTESTABLE run | still written **PASS**, no uncertified marker - no collateral damage |
| (i) the explicit opt-out | `-AllowUnattested` -> PASS |

**And the rule caught a subtler real condition while proving itself.** Case (g) first failed, correctly: passing
the exe ALONE made the capture read *"the binary PREDATES its sources"* (`src/llama-layer-streaming.cpp` edited
15:40:46Z; newest artifact of that set, the exe, built 14:32:01Z) - because the engine code lands in
**`llama.dll`** (15:41:00Z) while the exe relinks on its own schedule. That is this repository's own documented
gotcha, detected here by the rule rather than by a reader. The fixture now passes the real artifact set.

**Deliberate harness updates, not deletions.** Two fixtures whose subjects are NOT attestation (section 1's
status-derivation case and section 7's run-state case) now carry `-AllowUnattested` with the reason written
beside them; the wrapper cases in sections 6/10/11/12 keep their subjects unchanged because the waiver now
forwards to the writer. The whole harness is **ALL ARM-VERDICT CASES PASS**, and `-Suite host` is **exit 0,
42/42 CTest, all 12 blocks ran, none skipped** (`build/host-suite-writer-rule.log` - that log PREDATES the suite-revision binding, so it is a TIMESTAMP claim, not a revision claim; see the suite-revision block). **No historical runner
needed rewriting** - the rule reaches all 20 from the writer.

**The enforcement claim, restated to exactly what is true now.** (i) **No verdict on disk can read PASS unless
its capture attests the tree and the binary it ran** - writer-enforced, on every path, including the 20. (ii)
The GATE - refusing the run - binds at the wrapper and at the four live call sites that manage the mutex
themselves. (iii) The 20 historical runners still write verdicts without a gate, so their residue is now
*"recorded, attributable, but never gate-checked"* rather than *"recorded while unable to attest"*. That
supersedes the "would look certifiable" sentence in the inventory block above. Anything stronger - "every arm
is gate-checked" - remains false, and is not claimed.

**No latency or throughput claim.** Nothing in the retention state, A4's band or T-GPLR-55's shapes was
touched, and no historical runner was modified. Evidence: `tools/arm-verdict.ps1` (`Get-AttestationFailure`,
the writer rule), `tools/with-gpu-lock.ps1` (waiver forwarding), `tools/test-arm-verdict.ps1` section 17.
### Suite results bound to a revision (2026-09-22, host-only, no GPU spent)

**What was wrong.** The audit's last evidence gap. An ARM's evidence is bound to the source tree and the
binary it ran - a content digest plus the `llama.dll` hash - while a SUITE-GREEN claim was bound to nothing.
The suite's own log demonstrated the failure: the newest one on disk (14:09:57Z) described a revision that
no longer existed, so "42/42" was a statement about a tree nobody had named.

**How it works, reusing the arm convention rather than growing a second one.** `tools/run-tests.ps1` now
dot-sources `tools/source-revision.ps1` and captures, at suite START and again at suite END, the engine tree
(`third_party/llama.cpp`: nested HEAD, the whole tracked diff, every non-ignored untracked file by content)
and the binaries the suite exercises (runner + `llama.dll` + `ggml*.dll`, via `Get-ProvenanceArtifacts`).
Both captures plus the delta go to `build/llama-cuda/suite-runs/<UTC>-<suite>/revision.json`
(`schema: suite-revision/1`), and the result line NAMES the digest, so a suite-green claim is a claim about
a revision. The comparison is not re-implemented either: `-VerifyRevision <record>` re-derives the tree now
and settles a saved record with the SAME `Compare-ArmProvenance` rule the arm checker uses - exit 0 CURRENT,
1 stale, 2 cannot-evaluate - and it runs no suite.

**A result the tree moved under is not reported green.** `Complete-Suite` is the ONE place a suite result is
stated, so no caller can print a green the revision binding then undercuts:

| condition | result |
|---|---|
| tree digest and module hashes unchanged across the run | `SUCCESS: suite 'X' passed. revision digest <d> (nested head <h>, <n> tracked / <m> untracked changed), llama.dll <sha> - record <path>` (exit = the suite's own code) |
| sources or the binary moved WHILE it ran, tests passed | **`SUITE RESULT FLAGGED`**, exit **10** - the same contract the arm gates use - with the digest pair printed: the tests passed, but the result is not attributable to ONE revision |
| the tree could not be digested at all | also flagged (exit 10): a green that cannot name a revision is not a green |
| tests failed | `FAILED` with the suite's own exit code, revision line printed beside it |

The record carries `result` (`green` / `flagged` / `failed`), `tests_passed`, the effective `exit_code`, the
`flags`, and both captures - so the file and the process exit code agree, and a reader can tell "the tests
passed" from "the result is attributable".

**A defect found while proving it, not after.** The first version wrote `result=green, exit_code=0` into the
record of a run it then flagged at exit 10 - the artifact contradicted the process. It was caught by reading
the written record instead of the printed line, and fixed by computing the flag and the effective exit code
BEFORE the record is built. The same pass also fixed a cosmetic mis-keying (`changed_tracked` vs the flat
record's `source_changed_tracked`), which had printed the changed counts as blanks.

**Proofs - host-only, no GPU, no shared mutex, no arm re-run** (logs: `build/host-suite-revision-bind.log`,
`build/host-suite-revision-bind2.log`, `build/host-suite-inrun-drift2.log`, `build/suite-telemetry-revbind.log`)

| case | result |
|---|---|
| a stable tree | `-Suite host` **exit 0**: `SUCCESS: suite 'host' passed. revision digest 9399BF96289C63E1 (nested head 23e5d1b6e617, 5 tracked / 7 untracked changed), llama.dll 261AA683E10E97AA - record .../20260922T164140Z-host/revision.json`; the record reads `result=green, tests_passed=true, exit_code=0, flags=0`, delta clean, 6 module hashes |
| a source changes AFTER the run | `-VerifyRevision` -> **STALE-SOURCES exit 1**, naming `recorded digest 9399BF96289C63E1 (head 23e5d1b6e617, 5 tracked / 7 untracked changed) but the tree now digests to CB614B0FBCA7FA8D` and `now changed: suite-revision-probe.probe` |
| and it is residue-free | the probe deleted -> `-VerifyRevision` -> **CURRENT exit 0**: the digest is content-based, so a reverted tree returns to its earlier value and the record is again a description of it |
| the tree moves DURING the run | a probe file created 14 s into a real `-Suite host` -> **`SUITE RESULT FLAGGED`, exit 10**, `digest 9399BF96289C63E1 -> CB614B0FBCA7FA8D`; the record reads `result=flagged, tests_passed=true, exit_code=10`, `provenance_delta.source_changed_during_run=true`; and verifying THAT record reads **SOURCES-CHANGED-DURING-RUN exit 1** even though the tree now matches its start capture |
| a record with no revision | `-VerifyRevision build/nope.json` -> **CANNOT-EVALUATE exit 2**: an absent record is never a pass |
| the other finisher | `-Suite telemetry` **exit 0** with the same revision line and its own record - the suite's tail path and the telemetry branch both route through `Complete-Suite` (the `qa-smoke` green exit does too), so no branch can print an unbound green |

The probe was an untracked, uniquely named, non-ignored file inside the nested tree, present for seconds and
deleted in the same command; `git status` in the nested repo shows no trace of it afterwards and the digest
returns to `9399BF96289C63E1`. (Note for anyone repeating this: a `*.tmp` probe is SILENTLY NOT digested -
the nested `.gitignore` line 22 ignores `*.tmp`, and the digest covers non-ignored untracked files by design.
The first attempt used a `.tmp` name and correctly reported no drift, which is what exposed the distinction.)

**Scope: what this rebinds, and what it cannot.** Suite-green claims made BEFORE this landing name no digest
and are therefore **timestamp claims, not revision claims** - including `build/host-suite-writer-rule.log`
(18:22:28Z) and `build/host-suite-after-gate.log`, both cited in the blocks above, and the audit's own
14:09Z note. They are not retracted - each described the tree at the moment it ran - but none of them can be
verified against a tree now, and the two citations above are qualified in place so a reader cannot read them
as revision-bound. From this landing, every suite run writes a record and any later reader can settle it with
`-VerifyRevision`.

**Limits, stated rather than implied.** The binding is a claim about the ENGINE tree and the exercised
binaries, not about the whole repo: edits outside `third_party/llama.cpp` (a checker, a doc, a launcher) do
not move the digest, so a suite green can be "current" while the tooling around it has moved. Ignored files
are excluded by design (they are not sources), which is why a `.tmp` probe does not register. The capture is
a point-in-time reading: a source edited between the END capture and the record write is not seen by that
run (it is seen by any later `-VerifyRevision`). Nothing here changes what any suite RUNS - `Complete-Suite`
replaces the old print-and-exit pair and adds no work to the test path. **No latency or throughput claim.**
The nested sources under concurrent edit by another thread were not touched; the only writes inside that tree
were the probe files, created and removed inside this pass.

### T-GPLR-59 (draft): separate prediction-only expert-row pool - remove the real-cache capacity ceiling without changing its retained rows

Status: implemented, host-validated (2026-09-24, uncommitted; no GPU/model run authorized). Owner direction: use a separate pool with 12 prediction positions per logical layer and 12 proposals per stage; a READY prediction may serve a later exact demand, while a PENDING prediction is ignored and the demand falls back immediately. This is a new follow-up mechanism, not a reinterpretation of T-GPLR-58. T-GPLR-58's completed zero-fill/capacity evidence and its former no-extra-VRAM framing remain historical facts.

Result (2026-09-24, host-only). Steps A-F implemented; the focused host suite is green. Syntax: `--residency-tg-spec-pool 12` (the only legal nonzero value; requires `--residency-tg-spec 2`, a named arm, and the retention path; `--residency-tg-spec-budget` resolves to 12). Byte cost: layers x 12 x expert-row slab; host-derived 810 MiB on Qwen3.6-35B-A3B Q4_K_S (40 x 12 x 1,769,472 B), with the actual allocation recorded in `spec_pool_init.bytes` / `byte_account.prediction_pool`. Evidence: 20 T-GPLR-59 host policy/lifecycle checks (they caught a real inverted type guard in `llama_ls_expert_slab_bytes`), 26 pool telemetry validator cases (2 positives accepted and 11 single-rule negatives rejected by BOTH host validators), host suite 70/70. Known limits: capacity and proposal budget are fixed at 12 and refused otherwise (never silently reduced); pool hits are temporary slot input and never enter the real cache; the CUDA runtime path (pool buffers/events, H2D/D2D) has host-fixture evidence only; feasibility and performance remain unmeasured.

Dependencies: T-GPLR-58 mode-2 proposer and completion-event lifecycle (evidence and host fixtures complete); T-GPLR-19 per-layer row-cache allocation and retention telemetry; the runner's strict telemetry/retention validators.
Commit: unassigned.

Goal. Determine whether the existing predictor families can create useful demand hits when they are not restricted by the real row cache's occupied capacity. The first experiment fixes a pool capacity of 12 rows for every logical layer and asks each selected mode-2 predictor for at most 12 candidates after each completed demand stage.

Scope. Add one opt-in, CUDA-only experimental pool adjacent to the per-layer real cache. It owns separate GPU row storage, per-position lifecycle state, completion events, an admission policy, demand-read path, allocation byte accounting, telemetry, validators, and host tests. The pool is per logical layer; it is not 12 global positions, not 12 positions per predictor, and does not enlarge or replace the real cache. It supports the existing single selected mode-2 arm (coact, bar09, or fusion) and the count-matched null; it does not introduce a runtime ensemble.

Non-goals. No default-path behavior or telemetry change when the option is absent. No change to expert IDs, router output, quantization bytes, KV/recurrent state, graph arithmetic, PP scheduling, or T-GPLR-58 records/checkers. No GPU/model acceptance run is part of implementation; any later arm requires owner authorization, the shared GPU mutex, and a fresh attributable verdict. No claim that prediction is faster, has positive precision, or improves PP.

Design and failure contract.

1. Configuration and preflight. The runner exposes an explicit zero-default pool capacity (the first experiment uses `--residency-tg-spec-pool 12`) and records the resolved value in `run_start`. A nonzero value is legal only with `--residency-tg-spec 2`, an already configured arm, and the existing per-layer retention path. It is refused before CUDA/model work when malformed, incompatible, or unable to fit the *sum* of real-cache and prediction-pool bytes under the selected residency cap/allowance. A named pool capacity is never silently reduced.
2. Ownership. Each logical layer gets exactly the resolved number of private prediction-row positions and matching per-member CUDA tensors, independent of the real cache's buffers/tensors and of temporary shared slot bundles. The pool has its own expert/position map, state (`FREE`, `PENDING`, `READY`, `CANCELLED`), generation/phase marker, recency stamp, and copy-completion events. Teardown synchronizes then frees its own in-flight work. Pool state must never be encoded in the real cache's `cache_expert`, `cache_recency`, or `spec_state` arrays.
3. Admission after demand. The predictor observes the just-served demand only after the demand's own ready wait has closed, just as mode 2 does now. Its published candidates are deduplicated in proposal order. Before an admission, skip a candidate that is READY or PENDING in the real cache, READY or PENDING in the prediction pool, or duplicated earlier in the same proposal list. Every skip has a named telemetry reason. A candidate is copied H2D from its exact quantized expert slab into a free pool position, becomes PENDING immediately, and owns a completion event.
4. Capacity. The pool admits at most its proposal budget (12 in the first arm) and never overwrites PENDING bytes. When full, it may evict only the least-recently-issued READY prediction; the replacement copy starts only after the prior position is known complete. If no admissible READY victim exists, refuse the candidate with a structured capacity reason. No demand waits for pool completion.
5. Demand read. Exact router demand is unchanged. The real cache is authoritative first. For each real-cache miss, a READY matching prediction may be copied D2D into the current temporary execution slot as part of the normal whole-union slot refresh; it is counted as a pool hit and then retired from the pool after the demand's ready event. A matching PENDING or CANCELLED prediction is not a hit: the engine performs the ordinary real-cache miss path and does not wait for the pool copy. A pool hit is not inserted into the real cache, so it cannot inflate real-cache hit/retention accounting.
6. Lifecycle. A phase/generation boundary marks every in-flight pool row cancelled and reaps it before reuse; no byte is overwritten while its H2D can still be in flight. A cancelled row never produces a residency claim. Normal pool LRU eviction, generation cancellation, and demand consumption are separately counted.
7. Telemetry and accounting. Existing `spec` and real-cache retention fields retain their old meaning. A new `spec_pool` event, emitted only when the pool is enabled, carries run/layer/stage identity, capacity and proposal budget, candidate ids, each skip/refusal class, admitted/fill ids, pool pending/ready counts, PENDING demand fallbacks, READY demand hits and ids, consumption, cancellations by cause, evictions, completion-wait total, H2D bytes, D2D refresh bytes, and pool allocation bytes. `rowset_fetch` continues to describe the real cache; a feature trace must make the pool-hit D2D path separately attributable. `byte_account` adds separately named prediction-pool bytes and the validator requires the sum of real and pool bytes under the cap. Legacy traces remain valid when no pool is enabled; a pool-enabled trace missing/contradicting new fields is refused, not downgraded.
8. Safety. Quantized source spans and destination offsets use the same per-member slab arithmetic as the real cache. Missing source/member, buffer allocation failure, event failure, stale/mismatched generation, or an impossible D2D source is a loud structured refusal. No wrong expert, partial row, stale row, or untracked in-flight copy may reach the FFN graph.

Implementation steps.

A. Host-testable policy and lifecycle helpers. Add small pure helpers for cross-pool admission filtering, READY-only LRU victim choice, pool demand lookup, and lead/retirement bookkeeping. Keep their state explicit and free of CUDA handles so the fixtures can test boundary cases without a model.
B. Runner surface and preflight. Parse and validate the opt-in capacity; resolve it beside `--residency-tg-spec` / `--residency-tg-spec-budget`; include it in the run-start record and all pre-sink refusal records. Extend the exact per-member byte arithmetic used by `row_cache_bytes_for()` with a pool-capacity query, then include it in the preflight and actual-allocation cross-checks.
C. Engine storage and lifecycle. Allocate/deallocate pool buffers and tensors only when enabled; wire the mode-2 post-demand proposal path to the separate admission/issue/reap functions; keep the existing no-pool branch byte-for-byte in behavior and event meaning.
D. Demand path. Integrate READY pool lookup into the retention path's missing-row handling and whole-union D2D refresh. Record PENDING fallback separately and do not call a pool completion wait from a demand window.
E. Telemetry and validators. Add the `spec_pool` emitter, JSONL parsing, per-stage conservation checks, byte-account checking, and focused synthetic fixtures. Keep legacy schema acceptance explicit and test it.
F. Tests and records. Extend the existing TG-spec/TG-predict host fixtures and CMake registration. Update this task's result and `docs/currentstate.md` only after host validation; do not rewrite archived or completed T-GPLR-58 blocks.

Acceptance criteria (all are task-owned; each is necessary but none is owner release authorization).

- Host pure-policy tests prove: capacity is exactly 12 per layer; a 12-item publication admits no more than 12; duplicate candidates are admitted once; candidates present in real READY/PENDING or pool READY/PENDING are skipped with the correct reason; a full pool evicts only its LRU READY row; a full all-PENDING pool refuses without a wait; and PENDING demand never becomes a hit.
- Host lifecycle fixtures prove: READY demand selection retires exactly one pool row after the D2D path; a PENDING demand falls back without waiting; phase/generation cancellation blocks reuse until its event is reaped; and a stale slot/phase cannot satisfy a row.
- Allocation/refusal fixtures prove the pool uses the same per-member slab-size arithmetic as the real cache, counts its actual bytes separately, includes both pools in the cap/preflight, and refuses an explicit capacity that does not fit rather than reducing it.
- Telemetry fixtures prove per-stage proposal/admission/skip/refusal/eviction accounting closes; READY hits, PENDING fallbacks, cancellation causes, completion waits, H2D/D2D bytes, and pool allocation bytes are unambiguous; a feature trace with missing or contradictory pool fields is rejected; a legacy no-pool trace still passes unchanged.
- Mode-off regression proves no pool allocation, proposal, event, demand-read branch, or byte-account field changes behavior. The existing TG-spec, TG-predict, layout, and host-suite checks remain green after all consumers of a changed public header are rebuilt.
- No model or GPU run is started by this task. A later performance arm must separately record the exact command, model SHA/identity, hardware, resolved flags, mutex/verdict, raw JSONL path, and semantic-QA smoke result. A pool hit or telemetry PASS alone is not a predictor-quality or speedup result.

Focused validation commands (implementation task; no GPU).

```powershell
cmake --build build/llama-cuda --config Release --target test-layer-streaming-tg-spec test-layer-streaming-tg-predict test-layer-streaming-layout
ctest --test-dir build/llama-cuda -C Release -R "^(test-layer-streaming-tg-spec|test-layer-streaming-tg-predict|test-layer-streaming-layout)$" --output-on-failure
python tools/check-tg-predict.py --selftest
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-tests.ps1 -Suite host
```

Expected results: all listed builds/tests and checker self-test pass; the host suite reports SUCCESS with a saved engine revision record. If a source or binary moves during the suite, report its mandated `SUITE RESULT FLAGGED`, not a green result.

Required state/document updates.

- This task receives a measured result only after the focused host evidence above; add a dated entry at the top of `docs/currentstate.md` that separates policy correctness, instrumentation correctness, and any later feasibility/performance result.
- Keep T-GPLR-58's completed observations unchanged. If a later model run uses this pool, cite this task and its raw evidence rather than rewriting T-GPLR-58's real-cache claims.
- Update runner help and the active task status with the final opt-in syntax, resolved capacity, actual byte cost, and known limits. Do not update a generated build, trace, model, or telemetry artifact into version control.

Escalate to the owner before proceeding if any of these becomes necessary: changing the 12-per-layer or 12-proposal first arm; a shared/global pool or per-predictor allocation; a dynamic multi-predictor ensemble; a dependency; an API/flag that breaks an existing accepted trace; promotion of a pool hit into the real cache; a GPU/model run; or a capacity/VRAM policy that silently reduces the named pool size. Also escalate if a correct implementation cannot avoid demand waiting on PENDING pool work, or if telemetry cannot distinguish pool activity from real-cache activity.

### T-GPLR-59 follow-up design: host-only counterfactual replay of the separate prediction pool

Status: implemented, host-validated and scored over recorded traces (2026-09-24; see the result note at the end of this block). This is an evidence tool, not a runtime change and not a new performance arm. It remains separate from the T-GPLR-58 real-cache replay because a pool hit is not a real-cache retention hit.

Goal and hypothesis. Replay the ordered expert demand and published predictor candidates from an existing mode-2 JSONL trace through a 12-position, per-logical-layer prediction pool. The falsifiable hypothesis is that the separate pool creates useful later demand hits which the occupied real cache could not admit. The replay can measure lifecycle opportunity and logical bytes under an explicit ideal-completion model. It cannot establish CUDA behavior, transfer latency, PP speed, or a predictor-quality result.

Input contract. A scored leg must have one `rowset_init`, one `rowset_fetch`, and one mode-2 `spec` record for every `(ubatch, layer)` stage. Preserve file order and expert-id order. The `spec` record is semantically the proposal made after that stage's demand; because the emitter writes the two records in the opposite line order, pair them by `(ubatch, layer)` and apply the proposal to the next occurrence of that logical layer. The last proposal is unused if there is no later occurrence.

The primary replay consumes `spec.candidate_ids` as the raw proposal stream and records the source budget (`row_spec_budget`) separately from the pool budget. The first pool contract is 12 proposals, while the existing T-GPLR-58 legs normally record 8. A replay over an 8-item source is an explicit `recorded-8` lower bound, not a 12-proposal result. Never pad, repeat, or silently call it a 12-item arm. A full 12-item result requires 12 recorded candidates or a separately versioned host predictor replay; a null source additionally needs its RNG state or seed to reproduce IDs beyond the recorded list.

Two evidence modes stay distinct:

1. `overlay` (primary when `fill_ids` exist): replay the real cache, including recorded real speculative fills, PENDING/READY/CANCELLED transitions, generation boundaries, and capacity-LRU evictions. Require real hit and retention parity before applying the pool. The pool cannot alter that real-cache state.
2. `demand-only` (diagnostic): ignore real speculative fills and rebuild a demand-only real cache. This permits old traces without `fill_ids`, but the result is labeled a clean-baseline diagnostic and cannot be presented as an overlay measurement.

The replay is host-only and uses no model, CUDA, GPU, mutex, or network. It must not infer a role from a filename; arm/null/off roles are explicit session declarations.

Real-cache overlay. Each logical layer owns the real capacity-LRU with `cap_real = min(n_expert, retain_k * top_k)`. At demand, classify real READY rows first, cancel matching real PENDING rows without waiting, protect the demanded union from eviction, place its complement, and stamp recency in ascending expert-id order. Apply recorded `fill_ids` only after that stage's demand, as the real speculative path does. Require:

`route_union == real_hits + real_fetched`
`n_rows_resident == real_hits`
`n_rows_retained == real_hits`

A mismatch refuses the pool result. Never insert a pool hit into `cache_expert`, `cache_recency`, or real speculative state, and never increase real retention counters with a pool hit.

Pool state and ordering. Each logical layer owns exactly 12 private positions. Each position has its own expert id, state (`FREE`, `PENDING`, `READY`, `CANCELLED`), issue ordinal, recency, and generation. The default completion assumption is `ideal-next-reap`: a copy issued after stage `t` is PENDING during demand `t+1`, becomes READY after the end of `t+1`, and can first serve demand `t+2`. This is the measured lead floor, not a tunable convenience.

For every demand stage, in this order:

1. Apply the layer's generation transition. Old READY rows from another generation retire; old PENDING rows become CANCELLED and stay occupied until modeled completion. A stale row never satisfies demand.
2. Classify the real cache. For each real miss, look up the pool. A current-generation READY row is a pool hit; a current-generation PENDING row is cancelled and takes the ordinary real-miss path immediately; CANCELLED and stale rows are never hits. PENDING never causes a wait.
3. Refresh the temporary execution slot from real and pool hits. Retire a READY pool hit only after the demand ready point and never promote it into the real cache.
4. After the demand ready point, reap the pool: PENDING becomes READY and CANCELLED becomes FREE. Then apply the proposal from the stage just completed. In proposal order, reject invalid ids, publication duplicates, real READY/PENDING/CANCELLED duplicates, and pool READY/PENDING/CANCELLED duplicates. Prefer FREE; otherwise evict only the least-recently-issued READY row. Never overwrite PENDING or CANCELLED bytes. A full pool without a READY victim refuses with a capacity reason and no wait.

Because the proposal is issued after stage `t`'s reap, a candidate demanded at `t+1` is normally a PENDING fallback, not a hit. A candidate not demanded at `t+1` may become READY and serve a later exact demand. This carryover is a first-class result.

Accounting and metrics. Close candidate identity per stage:

`proposals = admitted + skipped_real + skipped_pool + skipped_dup + skipped_invalid + refused_budget + refused_capacity`

Close each admitted position's terminal fate over the replay:

`admitted = hits + pending_fallbacks + generation_cancellations + evictions + live_at_end`

Separately report real hits/fetches, pool hits, PENDING fallbacks, stale retirements, duplicate suppression, READY evictions, capacity refusals, modeled completion waits, H2D prediction bytes, D2D refresh bytes, and final occupancy. A PENDING match is never folded into hits.

Report at least:

- `demand_rows`, `demand_ids`, `real_hits`, `real_fetched`, and real share;
- `pool_proposals`, `pool_admitted`, `pool_hits`, hit rate over admitted rows, and hit rate over resolved rows (`hits + pending_fallbacks`);
- `extra_share = pool_hits / demand_ids` and total counterfactual share `(real_hits + pool_hits) / demand_ids`;
- next-demand fallback versus later-carryover hits;
- logical H2D bytes (`admitted * row_bytes`), D2D bytes (`hits * row_bytes`), and `net_h2d_delta = (admitted - hits) * row_bytes`;
- consumed-hit lead histogram, with every observed lead at least 2;
- per-layer/stage occupancy maxima proving `occupancy <= 12` and no PENDING/CANCELLED reuse.

Derive `row_bytes` from `bytes_full_bank / n_expert` and cross-check `bytes_staged / n_rows_fetched` where both exist. This is logical payload accounting, not actual CUDA allocation. Exact allocation bytes require a future `spec_pool_init` or runner byte-account field. Refuse byte metrics if the two trace ratios disagree.

Arm/null comparison. Require the same ordered demand-stream signature (including `(ubatch, layer, union_ids)`), model shape, top-k, retain-k, and source budget. Roles are explicit. For an arm/null pair, report each leg and paired deltas for pool hits, extra share, admissions, pending fallbacks, evictions, and net H2D bytes. Never pool sums across unequal leg counts. A null with a different source budget is not a count-matched control for a 12-item arm.

Refusals and integrity. Exit 2 with no measurement for no retention policy, incomplete stage pairing, missing raw candidates, invalid/out-of-range ids, W != 1, contradictory real-cache identity, missing `fill_ids` in overlay mode, mixed traces, mismatched arm/null demand streams, or inconsistent byte arithmetic. A closed trace with no opportunity is a measured zero; a missing source is `UNVERIFIED`, never zero. Unknown future events may be ignored, but contradictory known fields are rejected.

Output and fixtures. Use a new schema such as `gplr59-pool-counterfactual/1` with input paths, stream signature, source/pool budgets, completion assumption, real-replay status, per-leg summaries, paired deltas, and evidence limitations. Print lifecycle counters before the attractive hit rate.

The host self-test covers: 12-position capacity; 12-item admission bound; real/pool/list duplicate suppression; READY-only LRU eviction; all-PENDING refusal; a READY hit first at lead 2; a PENDING next-demand fallback without wait; generation cancellation; no real-cache promotion; candidate/admitted identity closure; occupancy bounds; overlay hit parity; and refusals for missing candidates/fill IDs, mixed streams, and inconsistent bytes. Wire it into the non-GPU host suite only after the fixtures pass. No model or GPU run is part of this design.

Non-goals. This replay does not claim a 12-row result from 8-row traces, infer actual transfer completion, estimate wall-clock speedup, change runtime admission, or alter T-GPLR-58's completed real-cache evidence. A positive replay is a reason to authorize a fresh owner-run pool arm, not acceptance of that arm.

Result (2026-09-24, host-only, no GPU/model run): implemented as `tools/analyze-gplr59-pool.py` (schema `gplr59-pool-counterfactual/1`); selftest 45/45 and wired into the host suite as `[tg59-pool-counterfactual]` (70/70). Replayed at OVERLAY PARITY 10,240/10,240 stages on all five candidate-bearing legs (validating the validator's retention model externally, evictions included). Measured: coact arm +2.18pp extra share (1,784 hits, every lead >= 2); A4 bar09 arm 6,483 hits (+7.91pp) vs null 1,123 (+1.37pp); fusion arm 5,082 (+6.20pp) vs null 921 (+1.12pp) - 4-5x the null's hits at ~1/7-1/8 its admission churn. `row_bytes` is layer-constant but not model-constant (37 x 1,769,472 B, layers 34/38/39 x 2,039,808 B; 12-row pool cost ~819 MiB, correcting the earlier 810 MiB figure). The pre-A2 null trace refuses (missing candidate_ids). Authorizing a fresh pool arm remains the owner's call.

### T-GPLR-60 (draft, planning only): context-conditioned, miss-aware, cost-aware prediction research

Status: CLOSED (owner decision, 2026-09-25; host-only G0/G1 complete; STOP). G2/G3 are not authorized and this predictor family is not advanced. Commit: nested e62b32427 + outer 31b07ab (shared substrate and closure metadata); scientific result already complete. Dependencies: T-GPLR-58/T-GPLR-59 evidence, the recorded Qwen3.6 demand stream, and the existing host replay; this block does not supersede the T-GPLR-59 MTP/hidden-state draft.

Goal. Test whether a predictor can improve as generated-token history accumulates by predicting only cache misses, conditioning on the current request/layer state, and charging for wasted prefetch. The falsifiable hypothesis is that a context-conditioned miss predictor beats an equal-row retention-only cache and a count-matched null on marginal miss hits without reducing net throughput; a failure is a useful stop result.

Scope. First host-only, on recorded demand: (1) learning curves by token prefix/window; (2) recency/frequency baseline; (3) request-local expert-map/cluster matching from already observed routes; (4) miss-only proposal accounting, Belady opportunity ceiling, equal-row K/P comparisons, and lead/readiness fates. No new learned sidecar or hidden-state capture is authorized here; those remain the existing T-GPLR-59 class-D/E owner decisions.

Non-goals. No change to routing, model arithmetic, expert IDs, quantization, PP, or the default-off path; no new dependency; no padding/repeating candidate IDs; no claim that offline recall, host replay, logical H2D bytes, or cache hit share equals CUDA tok/s.

Design contract. For each arm, report real hits, marginal pool hits on real misses, proposals, duplicates, refused/no-room candidates, READY-before-demand hits, PENDING fallbacks, lead histogram, useful/wasted H2D, equal-row retention baseline, count-matched null, and a Belady/oracle ceiling. Compare K+P against retention-only K+P/top_k at identical row budget. Evaluate at token prefixes 32/64/128/256 and, only with owner-authorized fresh runs, longer generations.

Staged gates. G0 provenance/no-leak closure; G1 host learning curve must show a repeatable miss-only opportunity above null and report whether it grows with history; G2 only then a cost-aware admission replay may claim a candidate for GPU testing; G3 owner-authorized mutex-held A/B must measure tok/s, transfer overlap, peak VRAM, and semantic-QA. A host failure at G1 stops the current family; a positive host result is authorization to seek a GPU run, never acceptance.

Validation commands. `python tools/analyze-gplr59-pool.py --selftest`; focused predictor-screen selftest; any new replay selftest; `python tools/resplit-currentstate.py --verify`; `python tools/check-doc-sizes.py --top 5`; `git diff --check`. A future GPU run must record exact command, model identity, hardware, flags, mutex/verdict, raw JSONL, and smoke result.

Required state update. Record the learning-curve table, null/oracle margins, equal-row comparison, and stop/continue decision in `docs/currentstate.md`; do not rewrite T-GPLR-58/59 historical results.

Escalate before changing the task boundary if the experiment requires hidden-state/router-logit capture, a learned sidecar, a new dependency, MTP/handoff mode, a multi-predictor runtime ensemble, routing changes, or a GPU/model run. The current evidence supports researching miss-aware/context-conditioned prediction, not assuming that more history or more fusion is automatically better.

Result (2026-09-24, host-only): G0 closed. Source trace SHA-256 `c38feedc5e20a0da47a03ae88cf667295a6fe05d5c35c2d58d6446135d566196`; prompt SHA-256 `e29c2fcc0da0815554c45e49026d4f3e24df6401fa492fe1b26e6286b00011d9`; model `Qwen3.6-35B-A3B`, 256 experts, top_k 8, 10,240 stages and 81,920 demand IDs. The coact, bar09, and fusion traces have identical stream signature `2613f366136ca8ac` and demand-row signature `c284cd11b2d9dbc5`. Prefix replay used token ranks 32/64/128/256, with 8,960/7,680/5,120/0 held-out rows; H=256 is `UNVERIFIED_NO_HELDOUT`. Predictor state was trained only on rows before the prefix and frozen on the suffix; the null seed is SHA-256-derived and stable across processes.

G1 result: miss-only pool hits are repeatable and far above the count-matched null (for example coact K=4: H32 2,096 vs 155; H64 1,809 vs 139; H128 1,179 vs 88), but they do not improve monotonically with history and every reported arm loses to equal-row retention. Coact K=4 total hit share is 55.90/56.45/54.87% for H32/64/128 versus 59.45/59.80/58.28% retention-only K+1; the corresponding marginal pool shares are 2.56/2.36/2.30%. The K=4 coact Belady ceiling is 77.21/77.32/76.77%, so prediction captures only a small fraction of the available oracle headroom. H32 K=4 coact also records 71,360 proposals, 2,605 admissions, 2,096 READY hits, 291 PENDING fallbacks, 0 evictions, 218 live-at-end rows, 3.50 GiB useful H2D, and 0.85 GiB wasted H2D. `tools/analyze-tg60-learning.py --selftest` is 9/9; the JSON replay is byte-repeatable after removing its timestamp. G1 therefore fails its required above-null-and-equal-row gate: STOP this family; no G2/G3 authorization follows.

Adversarial follow-up (2026-09-24): focused CLI probes found and fixed four task-owned boundary defects: missing/malformed trace and the fixed calculator-prompt input now refuse with exit 2 instead of leaking an exception; the free-form prompt override was removed so an unrelated prompt cannot be recorded as provenance; the command's `--trace` is included in the required equal-demand provenance set instead of silently checking only the default; and H2D accounting now uses each stage/layer's recorded `row_bytes` (the source has 1,769,472 B rows except layers 34/38/39 at 2,039,808 B). The analyzer selftest is now 15 checks, including fixed prompt identity/mismatch refusal, empty/out-of-order rejection, layer-byte mapping, transfer closure, prefix isolation, per-stage budget, and lead-2. The corrected H32/K4 coact values are 3.50 GiB useful and 0.85 GiB wasted H2D; the hit-share decision is unchanged. No async path exists in this host-only analyzer, and no imaginary phase/async hardening was added; the existing ideal-next-reap lifecycle and GPLR-59 generation tests remain the relevant state-synchronization coverage. Final decision remains **STOP**.

---

## Layer-major vs stock at the owner's real configurations (2026-10-01) - three DRAFT tasks

These three blocks were drafted on 2026-10-01 from the head-to-head sessions recorded in
`docs/tg-expert-path-reconciliation.md` §12.6 and summarised in `docs/currentstate.md`. **All
three are `draft`: none is authorized, none has a Planner contract, and no GPU time is
committed by any of them.** IDs continue after T-GPLR-62 (the highest used in this plan);
T-GPLR-63 and -64 are taken on another branch, so numbering starts at 65. Task numbering is
the Planner's to confirm at contract time.

Shared context for all three. Model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` (qwen35moe, 40 layers,
256 experts top-8, 19.91 GiB), RTX 5090 Laptop 24 GB, stock = anchor `llama-server` build 10892
at `C:\Development\AI\llamacppCUDA`, LM = this tree's `tools/layer-streaming` with
`--residency moe-expert-bank --pp-schedule layer-major`. Measured baseline this work must not
regress: LM PP **4,079.2 tok/s @ 10.056 GiB** against stock **2,099.6 @ 10.193 GiB** at ctx
262,144 (1.943x, 0.137 GiB apart), and LM TG **0.640-0.741x** stock. Every ratio is same-session
- cross-session LM variance reaches 16% - and VRAM is always the WDDM per-process peak from the
arm's own `*.mem.csv`, never the wrapper's `peak VRAM:` stderr line.

### T-GPLR-61 (ready): `moe-split-expert-ffn` — per-expert host/device split at token generation, shaped for the 8 GB target

**Status: READY (owner-directed 2026-10-01).** Promoted from the draft at
`docs/plans/qwen36-cpu-expert-placement-draft.md`, which is retained as the
full derivation, the withdrawn-and-corrected cost measurements, and the
provenance of the shape. The two blockers that held it at DRAFT are both
discharged:

- **The SDD amendment is written.** `docs/layer-residency-sdd.md` section 4 now
  carries the **`moe-row-split` class** — a *placement* class, not a storage
  class — with membership, structural exactness, trigger, cost gate, padded
  shape, the execution-shape constraint and its reasoning, the capacity
  treatment, the failure contract and the telemetry set.
- **The execution-shape question is resolved (2026-09-30, recorded in the SDD).**
  The class sits **OUTSIDE** section 4's row-set prohibition. That prohibition's
  subject is `MUL_MAT_ID` *in a row-set graph* consuming CUDA-resident weights,
  and what it forbids is the scheduler's expert row-copy plus any op-offload path
  that **rewires a weight source**. A CPU-backend sub-op reading tensors at their
  declared host placement is none of those. The T-GPLR-16 deadlock is specific to
  CUDA `MUL_MAT_ID` consuming scheduler-staged *host-bank weights*. The residual
  hazard — activation cross-backend staging — is **named in the SDD, permitted by
  both letter and evidence, and not proven safe**; it is where the per-layer sync
  cost lands, so step D's guard must assert specifically that no *weight* source
  is ever rewired, and step E must *measure* the activation round-trip.

**Owner direction (2026-10-01), which is this task's reason for existing:** on a
card with only 8 GB of VRAM, the full context must be used without sacrificing
prefill or token generation, and the answer is not stock-parameter tuning but a
reactive hybrid policy — *when an expert is in flight, wait; when it is merely
queued and not in flight, compute it on the CPU instead of waiting for the
transfer.*

**The shape is the 8 GB card, and ab14 (2026-10-01) is what makes it concrete.**
The one measured session at the target condition
(`build/llama-cuda/logs/ab14-lm8g-a1-u4096.jsonl`) gives, at 3 slots / ubatch
4096 / ctx 262,144 / KV q8: `static` 913.2, `per_layer` 1,523.5, `slot_window`
1,494.0, `kv` 2,720.0, `state` 62.8 MiB — **`bounded_window` 6,713.5 MiB, which
already contains KV and state** (`main.cpp:585`; see reconciliation §10.2 for why
adding them again is the double count that once produced a bogus "8,500 > 8,192"
refusal) — plus `act_slab` 1,125.0 MiB (ubatch-sized, **not** the 390.6 MiB a
ubatch-512 estimate suggests) and `reserve` 512.0 MiB.

At the 8,192 MiB budget the engine's own admission test is
`bounded <= budget` (`main.cpp:4730`), which **passes with 1,478.5 MiB to spare**.
But the test that decides whether a *physical* 8 GB card can run it is
`bounded + act_slab + reserve <= free_start` (`main.cpp:4741`), and
**6,713.5 + 1,125.0 + 512.0 = 8,350.5 MiB > 8,192 — this configuration would
have been REFUSED.** ab14 ran anyway because the budget flag never feeds that
third check. **That is the measured form of the owner's standing warning that
budget emulation over-reports what a real card can hold, and it is why this task
must be planned against a configuration that was never emulated.**

What actually fits at 8 GB, derived from the same `byte_account` with the act
slab scaled linearly to ubatch 512 (**derived, not measured at ubatch 512**):

| slots | bounded | + act slab (ub512) | + reserve | vs 8,192 | cache headroom (`budget - bounded`) | deepest K that fits |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5,717.7 | 140.6 | 512.0 | **6,370.3 fits** | 2,474.3 | **K=4** (2,185.4) |
| 2 | 6,215.3 | 140.6 | 512.0 | **6,867.9 fits** | 1,976.7 | K=2 (1,093.0) |
| 3 | 6,713.5 | 140.6 | 512.0 | **7,366.1 fits** | 1,478.5 | K=2 |
| 3 | 6,713.5 | 1,125.0 (ub4096) | 512.0 | **8,350.5 REFUSES** | — | — |

**So the 8 GB target condition is 1-2 slots at ubatch 512, and at those depths at
most 32 of a layer's 256 expert rows can be device-resident (K=4 x W=1 x top_k=8)
— 12.5% of a bank.** At least 87.5% of demanded experts have no device residency
available at any budget. **That is why this task is a feasibility mechanism at 8
GB and not a TG tuning knob.** On a 24 GB card the same block is worth an
estimated ~1.14x serial / ~1.33x overlapped (draft §8); on an 8 GB card the
alternative to a host-computed expert is a link fetch the measurement says is
3.4-6.7x more expensive than computing the row where it already lies.

**Cost gate: measured, and not to be re-derived** (owner directive 2026-10-01).
`tools/expert-cpu-bench`, iteration-disjoint (`--pool 256 --id-stride 8`), real
Q4_K bytes, TG shape `n_tokens=1`, 300 iterations, medians: marginal host cost
**~36.5 us/expert** against **~171 us/expert** for the link at the best measured
end-to-end feed of 10.35 GB/s (row bytes 1,769,472). Margin **3.4-6.7x, constant
in k**. A first measurement (2026-09-29, ~8.7-9.8 us/expert) was **withdrawn**: it
used a hard-coded expert-id stride of 1, so 7 of every 8 slabs were still
cache-warm and the marginal cost measured was a cache cost, not a DRAM cost —
visible in the run's own `gbps` column rising 41.9 -> 128.6 GB/s as k went 1 -> 8.
The corrected figure **raised** CPU cost ~3.9x and **lowered** the transfer
break-even hit rate from ~88.8% to 83.2%, so it moved *against* the GPU path.
**The rule is unchanged and is not a tunable threshold: a demanded row that is not
device-resident computes on the host.** The estimator stays explicit and host-side
so it can be re-derived when the host or the model changes, and a named gate
policy that cannot be evaluated **refuses the run** rather than defaulting.

Goal. Let a token-generation layer begin its expert FFN before every demanded
expert row has landed, by computing the still-**PENDING** rows on the CPU and the
already-resident rows on the GPU, summing both into the layer output. The layer
never waits on a row copy that has not completed. The split is decided at demand
time from actual residency, per layer and per token; nothing about it is fixed at
run start. The predictor's role is to raise the resident share, not to choose the
split.

Scope. Opt-in, CUDA+CPU, **default off**, single process, **TG only**. A
residency-driven split of the expert FFN into a device sub-op and a host sub-op;
a trigger and cost gate; a fixed-width padded execution shape so the persistent
graph set stays static; a non-finite guard; telemetry separating device experts,
host experts, and rows the split skipped. The identity probe is the blocking gate.
The reference project's `V4_CUDA_TG_CPU_DIRECT` is a **read-only source of the
mechanism**, not a dependency.

Non-goals.

- **No placement-fixed CPU banks.** A per-layer static CPU/GPU split is a
  different mechanism.
- **No change to the demand set.** The router's executed IDs are authoritative
  and untouched; the split never drops, substitutes, delays or reorders an expert.
- **No change to the predictor.** This consumes whatever is resident; it adds,
  retunes and re-gates nothing. T-GPLR-58/59/60 stand.
- **No `GGML_OP_OFFLOAD_MIN_BATCH` override and no scheduler expert-rowcopy** for
  the affected layers.
- No prefill change. In layer-major PP the bank arrives whole; this is a TG class
  by construction.
- No change to routing, router arithmetic, top-k, quantization bytes, KV or
  recurrent state, logits pattern, sampling, the add-chain reduction, the shared
  expert, or any default path.
- **No performance claim.** The first deliverable is feasibility/correctness.
- **Not a substitute for an 8 GB capacity configuration.** This task changes where
  a demanded expert computes; it does not make ab14's ubatch-4096 3-slot
  configuration fit an 8,192 MiB card.

Design and failure contract. As written into `docs/layer-residency-sdd.md`
section 4 (`moe-row-split`); the full reasoning, the bounded bounds of the
microbenchmark and the reference project's three mandatory test cases are in the
retained draft.

Implementation steps.

- **A. Policy and shape (host-only).** Pure functions: residency classification
  at demand time (READY / PENDING / absent), the cost gate, the fixed-width
  padding plan, and the host/device sub-op descriptor. No CUDA handles. CTest
  fixtures: all-READY (must not fire), all-PENDING with the gate open, gate
  closed by estimate, a mixed set, an absent row (refuse), padding contributing
  zero, an unevaluable policy (refuse).
- **B. Engine.** Extend the row-set path to build the two sub-ops against the
  existing add-chain reduction, with the host sub-op's weights bound to the
  in-flight source rows on the host. Slot, generation, copy-event and
  consumer-done lifecycles are unchanged; in-flight copies keep running and
  retire normally — the split must not cancel or consume a copy the mechanism
  still owns.
- **C. Guards.** Extend the host-only T-GPLR-16 guard to assert the MIN_BATCH
  override and the scheduler rowcopy are unreachable, and **specifically that no
  weight source on this path is ever rewired**; add the non-finite abort and the
  absent / never-executed aborts.
- **D. Focused host validation, then the identity probe.** No performance arm.

Falsifiable acceptance criteria.

- **AC1 (policy and shape, host-only, blocking).** All-READY never splits; a
  PENDING row goes to the host only when the gate opens; a closed gate returns it
  to the ordinary wait path unchanged; padding contributes exactly zero; an absent
  demanded row refuses; an unevaluable policy refuses; the emitted sub-op
  descriptors are constant-width for every input.
- **AC2 (IDENTITY, blocking, the real gate).** On the 512-token anchor with the
  split enabled and the gate forced open: greedy token IDs **byte-identical** to
  the recorded all-GPU streaming arm for the same shape, `check-telemetry` OK,
  aggregate PASS 5/5, negative control FAILing as designed. **If AC2 fails the
  task stops there** and records the falsification — no placement tuning, no gate
  tuning, no rescue. Per AC1's structure the failure can only be fp-order or
  kernel-domain divergence, and that distinction must be stated.
- **AC3 (no hang).** The arm completes inside the recorded wall band with no TDR
  and no `cudaErrorLaunchTimeout`. A hang is a hard fail even if AC2 passed.
- **AC4 (instrumentation, not performance).** `experts_device + experts_host ==
  experts_total` at every (layer, token); in-flight copies complete and retire
  normally; `staged_bytes_avoided` is consistent with `experts_host` and the
  measured row bytes.
- **AC5 (inertness is a reportable result, not a failure).** `experts_host == 0`
  for the whole run is a valid recorded outcome — the mechanism did not fire —
  and must be reported as such, never as a pass with an implied benefit.
- **AC6 (no default change).** Flag absent: trace byte-identical to the recorded
  control, host suite green, dense regression unchanged.
- **AC7 (the falsifier the owner should read first).** The recorded
  `experts_host / experts_total` share at the 512 anchor. The T-GPLR-59 frontier
  run measured a demand-hit share of 0.4518, so the prior is that the host
  carries a large minority-to-majority of expert work. **At the 8 GB shape the
  prior is far stronger**: K=4 caps device residency at 12.5% of a bank, so most
  demanded rows are host-eligible by construction. A large host share is not a
  defect to tune down — it is the input to the performance arithmetic, and until
  an arm is authorized no speed claim of any kind may be made.
- **AC8 (the 8 GB shape, when an arm is authorized).** Any arm run at the 8 GB
  target must be at a configuration whose `bounded + act_slab + reserve` is
  **measured** to be under 8,192 MiB. ab14 is the cautionary case: it was
  admitted by a budget that does not feed the check that would have refused it.

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
a non-zero host share, or a loud refusal — never a silent degradation; step 2 only
after step 1 passes.

Required state and document updates.

- A dated `docs/currentstate.md` entry separating **feasibility** (does it run, is
  the output identical), **instrumentation** (are the counters and the in-flight
  lifecycle true) and **performance** (not measured here — say so).
- The owner summary: goal, plain-language mechanism, glossary, pros/cons with
  every claim labeled measured / expected / unproven, and a recommendation.
- No build, trace, verdict or telemetry artifact into version control.
- If the block later changes, `docs/plans/qwen36-cpu-expert-placement-draft.md`
  keeps the derivation and the withdrawn measurements; this block is the contract.

Escalate to the owner before proceeding if: any measurement shows the scheduler
rewiring a **weight** source on this path (immediate stop — this is the named
residual hazard); AC2 fails; any hang or TDR appears; the mechanism turns out to
need the MIN_BATCH override, the scheduler rowcopy, or a variable-width graph; a
demanded row cannot be classified as READY, PENDING or absent; a new dependency is
needed; a flag would change an accepted trace's meaning; or a performance arm is
proposed before AC2 passes.

### T-GPLR-65 (draft): adopt the V1 causal-flash-attention dispatch fix into the vendored engine

Status: DRAFT (2026-10-01). Not authorized. The fix exists and is measured **only** in the side
copy `C:\Development\fafix\src\ggml\src\ggml-cuda\fattn.cu` behind an env toggle
(`FAFIX_CAUSAL_GQA=1`); the vendored tree is untouched.
Dependencies: none inside this plan; requires a Planner task contract before any vendored edit,
because `third_party/llama.cpp` is a gitlink carrying another workstream's uncommitted
`src/models/qwen35moe.cpp` edit. Blocked on that coordination, not on evidence.
Commit: (none - not started)

Goal. Move the measured V1 dispatch fix from the side copy into the vendored
`ggml/src/ggml-cuda/fattn.cu`, and prove it is a pure kernel-selection change: same
quantization, same routing, same expert IDs, same answers, less time.

Scope. The one gate that failed and the kernels it gates. A causal `flash_attn_ext` op with
`mask == nullptr` fails `use_gqa_opt` / `gqa_opt_applies` (`ggml-cuda/fattn.cu:58` and `:386`),
so it dispatches `<256,256,64,1>`: no cp.async pipeline, ~8x K/V re-reads, and the KV_max skip
at `fattn-common.cuh:~1094` is lost. The K/V `ne[1] % 256` gate misses because `prepare_chunk`
shrinks that dimension (`src/llama-layer-streaming.cpp:5579-5587`). Note the mechanism has two
independent gate failures, and only the first is the V1 edit.
Non-goals. No change to attention math, mask handling, KV quantization, or any non-causal path;
no new dependency; no default-on behavior change without the semantic-QA ladder; **not** a fix
for the pre-existing Ada+ q8_0 KV hazard where `n_q == 2` selects VEC and ignores the causal
window - that is a separate defect and is explicitly out of scope here.

Evidence already on disk (measured, 2026-10-01). VRAM-neutral: ON and OFF both peak at
**10.056 GiB** at ctx 262,144 (and likewise at ctx 128,000/f16 and 512,000/q8). Contribution at a
50k prompt is **1.186x / 1.182x / 1.182x** across those three contexts - a 0.4% spread over a 4x
context range - and it **grows** with prompt length: **1.342x** at 144,001 and **1.377x** at
200,000 tokens. Sources: `build/llama-cuda/verdicts/ab9.verdict.json`, traces
`build/llama-cuda/logs/ab9-{on,off}*.jsonl`; the length points are ab6/ab7. Side-copy fidelity on
record: logits within a 0.1-1.3 noise floor, answer-QA PASS, ktest2 332 cases, bench 1.37-2.09x
at long `nkv`.

Acceptance criteria (falsifiable). (1) With the vendored fix and the side copy carrying identical
source, a same-session ON/OFF pair at ctx 262,144 / 50k reproduces **>= 1.15x** PP with WDDM peaks
equal to within 0.01 GiB. (2) PP at 200,000 prompt tokens improves by **>= 1.30x** versus the same
binary with the gate restored, confirming the length dependence. (3) `tools/check-answer.py`
reports `aggregate=PASS` on the 512-token anchor under the **default-off** path, and the must-fail
negative control still fails. (4) The full host suite passes with no new failures. Failing (1)
means the gate diagnosis is wrong and the task stops.

Validation commands. Build the vendored engine under the project CMake, then run the 50k/512
semantic-QA smoke tier through `tools/with-gpu-lock.ps1` (mutex
`Global\AI-GPU-EXCLUSIVE-v1`, one workload per hold, `-WaitCpuQuiet`), then one owner-invoked
50k PP arm at ctx 262,144 with the gate on and off. **No compile may run inside a timing arm.**
Side-copy the exe and every DLL before rebuilding if an A/B control across the vendored change is
wanted - the streaming code lives in `llama.dll`, so copying the exe alone silently loads the new
library.

Required state/document updates. A `docs/currentstate.md` entry recording the vendored commit and
the reproduced ON/OFF pair; an `AGENTS.md` gotcha if the gate behaves differently in the vendored
build than in the side copy (in particular if the `% 256` dimension differs after
`prepare_chunk`).

Escalate before proceeding if the vendored edit would collide with the other workstream's
`qwen35moe.cpp` change, if the fix cannot be made default-off, or if reproducing >= 1.15x requires
touching anything outside `fattn.cu`. A PP win that needs a mask-upload or padding change is a
different task.

### T-GPLR-66 (draft): fix the `alloc_spill` alignment failure that crashes LM at ubatch 2048 and high context

Status: **DONE (owner-directed 2026-10-01).** Fixed, built, host-tested and validated on GPU. Commit: (none - owner has authorized no commit).

**ROOT CAUSE (found and fixed).** `alloc_spill` sized its spill buffer as a **flat
sum** of the pending tensors' alloc sizes, rounded up once to 32 B. But
`alloc_tensors` **realigns every tensor to the buffer alignment as it walks**, so a
pending set containing odd-sized tensors needs per-tensor padding the flat sum
omits. The buffer therefore came up short by exactly the padding, and because the
overflow always lands on the **last** tensor of the set, the only thing the runner
ever printed was the generic `finish_build: input buffer exhausted and spill
allocation failed`.

**Measured proof, from the ab15 failing arm's own log:** `layer-streaming: tensor
'leaf_3' exceeds backing buffer (offset=802848 size=4 buffer=802848 align=32)`.
`buffer=802848` **is the spill buffer**: it was allocated at precisely the cursor
the walk had already reached, for a 4-byte tensor. The `exceeds backing buffer`
line the run DID emit was misread as evidence against this cause - it also appears
in **working** arms (5x at ubatch 4096), because hitting the *primary* buffer's
limit is normal and is exactly what the spill exists to handle. (Reading these
logs needs `tr -d '\000'`; they are UTF-16 **with BOM** and `iconv -f UTF-16LE`
mis-parses them and silently finds nothing.)

**The fix** sizes the spill by walking it the way `alloc_tensors` will, using
`ggml_backend_buft_get_alignment` rather than a hardcoded 32. `alloc_spill`'s three
silent-false paths now each set a distinguishing `error_msg`.

**Why it looked non-monotonic in ubatch** (512 and 4096 clean, 1024 and 2048
crashing): at ubatch 4096 the pending set is dominated by 32 MiB-aligned `inp_embd`
slices where the padding is zero and a flat sum happens to be correct; at 1024 and
2048 the set carries small tensors (`leaf_*`, 4 B) whose padding is not.

**Validation (ab16, one mutex hold, 5 arms, `gpu_pre`/`gpu_post` clean, mutex
released).** ubatch **1024 and 2048 now complete at exit 0** where ab15 aborted
with 0xC0000409 on the identical configuration. The control is internal and
same-binary: ubatch 512 and 4096 in the same session on the same binary. Host
suite green first: `test-layer-streaming-layout` and `test-layer-streaming-tg-spec`
both PASS. Side-copied binary `build/llama-cuda/bin/gplr66-fixed/` (exe
`70e35f47acf27e17`, llama.dll `bcd5ddac455ca2b6`).

**What the fix is NOT.** It does not change prefill speed on the widths that already
worked - it only sizes a host allocation that happens after the primary input
buffer overflows. Its effect is **crash -> completion**, and the recovered points
are worth +18.3% (ubatch 1024) and +27.0% (ubatch 2048) over ubatch 512 in tok/s
simply because those widths now run at all.

**REPRODUCED TWICE, AND IT IS IN OUR TREE (ab15, 2026-10-01).** The "never
re-exercised" honesty note below is now **superseded**: an ab15 sweep at slots 2 /
ctx 262,144 / KV q8 / 50,000 tokens crashed at **ubatch 1024 AND ubatch 2048**,
both with exit `-1073740791` (0xC0000409), while **ubatch 512 and 4096 both
completed**. Both failures abort at the same message, and the string lives in
**our** vendored tree, not only in the fafix vehicle: `src/llama-layer-streaming.cpp:3867`
sets `error_msg = "finish_build: input buffer exhausted and spill allocation failed"`
and `src/models/qwen35moe.cpp:495` turns it into `GGML_ABORT`. The signature is
**non-monotonic in ubatch and strikes in pass 2/3, never pass 1** (u1024 got
3,460.6 tok/s through pass 2 before dying; u2048 completed pass 1), which points
at an alignment/allocation edge case rather than a resource limit - consistent
with this task's hypothesis. Traces `build/llama-cuda/logs/ab15-s2u{1024,2048}-a1-u*.log`.

**What this changes for the task.** The title says "ubatch 2048"; the defect is
**at least a two-point set {1024, 2048} with 512 and 4096 clean**, so the
alignment hypothesis must explain a non-monotonic pattern, not a single shape.
ACs that assume ubatch 2048 is the only failing width are wrong as written.
Dependencies: none; self-contained inside `src/llama-layer-streaming.cpp`.
Commit: (none - not started)

Goal. Make `alloc_spill` allocate at an alignment its consumers actually require, so a layer-major
prefill at ubatch 2048 with a large context completes instead of aborting with
`finish_build: input buffer exhausted and spill allocation failed` and exit `0xC0000409`.

Scope. `alloc_spill` at `src/llama-layer-streaming.cpp:1173-1202` and the 64 MiB host input
buffer it draws from. The suspected cause is a 32-byte alignment assumption that does not hold
for the spill buffer's consumers.
Non-goals. No change to the residency window size, the slot budget, KV placement, or the
preflight; no raising of VRAM limits to make the shape fit; no change to any ubatch that currently
works (4096 is clean).

Honesty note, which is part of the task's evidence. The crash was observed once, at LM ubatch
2048 / ctx 262,144 with `--pp-repeat >= 2`. **It was never re-exercised**: every LM arm in the
2026-10-01 head-to-head sessions ran ubatch 4096. So the alignment theory is a hypothesis, not a
diagnosis, and the first acceptance step is to reproduce rather than to fix. Do not let this block
be read as "known root cause".

Acceptance criteria (falsifiable). (1) The failing shape (LM, ubatch 2048, ctx 262,144,
`--pp-repeat 2`) completes with exit 0 and a `check-telemetry: OK` trace - reproduced FIRST, on the
unmodified tree, to confirm the shape still fails. (2) After the fix, that same arm passes and its
`phase_summary` PP is within 10% of the same run at ubatch 4096 (a larger ubatch is not free, so
this is a sanity band, not an equality). (3) The alignment actually required is *demonstrated* -
a host-only test that asserts the spill pointer satisfies the consumer's alignment - so the fix is
justified by evidence rather than by a crash disappearing. (4) The host suite passes unchanged.
If (1) does not reproduce, the task is re-scoped or closed as not-reproducible; that outcome is a
legitimate result and must be recorded rather than papered over.

Validation commands. Reproduce with one mutex-held arm at ubatch 2048 / ctx 262,144 /
`--pp-repeat 2` through `tools/with-gpu-lock.ps1`. Then the focused host CTest target, then the
50/512 semantic-QA smoke tier.

Required state/document updates. A `docs/currentstate.md` entry stating whether the shape
reproduced, the confirmed alignment, and the commit; remove or keep the risk bullet in
`docs/currentstate.md` accordingly.

Escalate before proceeding if reproduction requires a larger context than 262,144, if the fix
would change the residency window size, or if the spill buffer turns out to be sized rather than
aligned incorrectly. Report a NOT-REPRODUCED result upward rather than iterating on a fix for a
crash that no longer occurs.

### T-GPLR-67 (draft): `--handoff-partial N` is inert because its handoff-on assignment sits inside a comment

Status: DRAFT (2026-10-01). Recorded, deliberately **NOT fixed** - the vendored tree is another
workstream's and this needs a Planner contract. One-token change.
Dependencies: coordination with the workstream holding `third_party/llama.cpp` (it carries an
uncommitted `src/models/qwen35moe.cpp` edit).
Commit: (none - not started)

Goal. Make `--handoff-partial N` do what its own comment says, so passing it alone activates a
handoff instead of leaving `handoff_mode_str` at `auto`.

The defect, verbatim. `tools/layer-streaming/main.cpp:3454` reads:

        // stay host-resident (CPU) for mixed-placement TG. Implies an
        // explicit handoff-on request.            handoff_mode_str = "on";

The assignment is **inside the `//` comment**. `--handoff-partial N` parses `N` and sets
`handoff_partial_n`, but `handoff_mode_str` stays `"auto"`. Because `auto` resolves OFF whenever
full model+KV+state does not fit, the flag alone can never activate a handoff at a large context,
and every `--tg-tokens > 3` run then refuses at `main.cpp:4871`. Note the `--help` text
(`main.cpp:67-68`) describes only which banks migrate and never claims a handoff - so `--help` and
the code agree, and what the comment swallowed is the **documented intent**. A reader of `--help`
has no way to learn the flag is inert alone.

Measured impact (2026-10-01). All four partial-handoff arms of session **ab11** (`lmh20`, `lmh14`,
`lmh28`, `lmh20b`) failed this way - `ARM EXIT 2 (4.7s)` each, with
`handoff: auto -> OFF` then the cap. Evidence: `build/stock-levers/ab11.wrapper.log` and
`ab11-lmh20.tail.txt:2234-2236`. Workaround used for all reported numbers: pass `--handoff`
explicitly alongside `--handoff-partial N`.

Scope. One line in the vendored `tools/layer-streaming/main.cpp`, plus the `--help` text if the
Planner wants the implication stated there.
Non-goals. No change to the partial-handoff policy itself (which banks migrate for a given N), to
the preflight, or to the tg-token cap; no default change to `--handoff-partial`'s current
effective behavior without an owner call, since making the flag work may change what existing
recorded commands do.

Acceptance criteria (falsifiable). (1) `--handoff-partial 20` alone, at a shape where `auto` would
resolve OFF, activates the handoff - i.e. the run proceeds past the `main.cpp:4871` cap instead of
exiting 2. (2) `--handoff-partial 20 --handoff` behaves **identically** to today's recorded
workaround arm (same banks migrated, same `phase_summary` shape). (3) The refusal still fires for
`--no-handoff --tg-tokens > 3`, so the cap is not accidentally removed. (4) Host suite passes.
Criterion (2) is the important one: the fix must make the flag behave as its comment always
claimed, not introduce a new policy.

Validation commands. Host-only first: extend the existing flag-surface fixture to assert the
assignment is reachable. Then one mutex-held partial-handoff arm via `tools/with-gpu-lock.ps1` at
ctx 262,144, `--handoff-partial 20`, and the mirrored `--handoff-partial 20 --handoff` arm.
**Fidelity caveat that must be resolved by this task:** in ab12, `lmh20` and `lmh20b` - identical
flags - produced DIVERGENT sequences at 128 generated tokens despite a bit-identical token 0
(early margins 0.0093/0.1617/0.1099/0.0595 against 1.6988/2.0027/1.1112/3.2644). Any token-level
claim about the partial-handoff path is UNPROVEN until that divergence is explained; this task
should report PP/TG aggregate behavior and explicitly decline to certify token identity.

Required state/document updates. Remove the `AGENTS.md` gotcha about passing `--handoff`
explicitly once the flag works alone; record the outcome in `docs/currentstate.md`; update the
`docs/tg-expert-path-reconciliation.md` §12.6.8 defect note from "recorded, not fixed" to the
resolution.

Escalate before proceeding if making the flag effective would change an existing recorded
command's behavior in a way the owner has not called, or if the vendored coordination is
unavailable. This is a one-token fix, and the temptation to land it outside a contract is exactly
what the vendored-tree rule exists to prevent.

Architecture pass (2026-09-24, shape only; STOP unchanged). The T-GPLR-59/60 code now has three explicit boundaries. (1) `src/llama-layer-streaming-pool.{h,cpp}` owns the pure prediction-row state enum, exact expert-slab arithmetic, admission plan, generation transition, lookup, retirement, and completion policy. It has no CUDA or runner state. `llama_layer_streaming` remains the single runtime owner of pool buffers, completion events, package sources, and counters, and the large streaming implementation only orchestrates those resources through the policy API. (2) `tools/layer-streaming/telemetry-pool.{h,cpp}` owns all stateful `spec_pool*` validation; `main.cpp` only creates one validator per telemetry pass and forwards events. The minimal JSONL readers used by both validators live once in `telemetry-json.{h,cpp}`. (3) `tools/tg_pool_core.py` owns trace parsing, shared real-cache replay, pool lifecycle, fate closure, and report-data construction. `analyze-gplr59-pool.py` is now the CLI/rendering/selftest adapter, while T-GPLR-60 imports the shared real-cache fact builder, `PoolLayer`, and `replay_pool_stages`; its former duplicate replay lifecycle is deleted. The shared core emits its complete fate ledger; no policy, event schema, capacity, scientific result, or experimental decision changed in this pass. Data flow is runner flags -> engine policy/CUDA adapter -> JSONL -> state-owning validator, and recorded trace -> shared replay core -> CLI-specific scoring/presentation. Adversarial entry-point validation also added negative fixtures for duplicate `run_start`, `spec_pool` before `rowset_init`, and quoted/malformed numeric fields; these exercise the real `--check-telemetry` and `--validate-telemetry` paths. The shared reader preserves valid decimal serialization used by phase summaries while rejecting quoted, partial, non-finite, overflow, and malformed-array tokens. Runtime teardown remains event-synchronized before pool buffers/state are released; no extra async/cleanup path was added. This is validator robustness evidence only: no pool-enabled CUDA run was authorized, and the T-GPLR-60 decision remains **STOP**.

### T-GPLR-70 (FALSIFIED 2026-10-02 by ab19): narrowing the TG KV window makes TG 18.9% SLOWER

Status: **IMPLEMENTED, MEASURED, REJECTED, REVERTED.** The change built clean and passed the
host suite (76/76), ran correctly, and **regressed token generation by 18.9%**. The patch was
reverted and the reverted source is proven digest-identical to the validated pre-T-GPLR-70 tree
(`source_digest 28382EF598129F94`, the same value the ab16 verdict recorded). **Do not re-apply
this patch.** The rejected binary set is retained at `build/llama-cuda/bin/gplr70-changed/` for
reproducibility only, carries a `DO-NOT-RUN.txt`, and must not be reported as a candidate.
Dependencies: none. Evidence: `build/stock-levers/ab19.plan`, `ab19.manifest.jsonl`,
`build/llama-cuda/logs/ab19-{ctl38x,g70x}-a1-u1024.*`, `build/llama-cuda/verdicts/ab19.verdict.json`.
Commit: (none - reverted; nothing to commit)

**What was built.** In `llama_context::layer_streaming_compute`, before each token-generation launch,
assign every `GGML_OP_FLASH_ATTN_EXT` node's K and V view `ne[1]` to `max(ubatch.pos)+1`, mirroring
the PP patch (`T-Q38FA-01` in `prepare_chunk`). Gated to `ls->phase != 0`, so PP was untouched by
construction. Causal was re-stated only where the node was already causal.

**A bug in the patch itself, found and fixed before the arm ran.** The first version shrank the view
only when it was longer than the live window ("never grow"). Because the graph set is built once
(`graph_builds == 1`) and reused for every token while the live window grows one token per step,
that guard **froze the view at token 1's window** and would have silently stopped reading the newest
tokens from token 2 onward. The PP patch has no such guard; it assigns unconditionally. This is the
single most dangerous shape of bug in this mechanism and it was caught by reading the code, not by
a test.

**Measured result (ab19: one mutex hold, both arms re-run in session, side-copied binaries,
llama.dll the only differing artifact, `route=1` on both).**

| arm | PP tok/s | peak VRAM | TG tok/s | TG compute/token |
| --- | --- | --- | --- | --- |
| `ctl38x` (bin=fixed, pre-fix) | 3,089.6 | 7.600 GiB (re-derived) | **23.096** | 39,996.6 us |
| `g70x` (bin=g70, narrowed) | 3,089.2 | 7.600 GiB (re-derived) | **18.733** | 50,000.2 us |

PP and VRAM are unchanged, as designed. **TG regressed 18.9%.** The patch demonstrably fired:
`T-GPLR-70: TG attention KV view set 262144 -> 50001 (q_pos_base 50000, layer 3); 1 fa node(s)
per layer graph`.

**AC1 FAILED (the primary criterion): the narrowing did not reduce the decode attention cost.**
**AC2 FAILED in the worst direction: TG fell instead of rising 15%.**
**AC3 FIRED - this is the falsifier, and it fired.** The claim "the TG decode attention waste is the
KV view length" is **not supported by the measurement that was supposed to confirm it.**

**The regression is not where the hypothesis said it would be, and that is the real finding.**

| group | control | narrowed | delta |
| --- | --- | --- | --- |
| host-resident banks (il < 38) | 38,222.5 us | 47,195.5 us | **+23.5%** |
| device-migrated banks (il >= 38) | 1,774.1 us | 2,804.7 us | **+58%** |

The 30 gated-DeltaNet layers own **no `FLASH_ATTN_EXT` node** and therefore **could not have been
narrowed at all** - yet they regressed too. So the damage is a **side effect of mutating the
persistent graph set's KV view tensors**, not of narrowing them. The TG graph set is built once and
reused for all 128 tokens; its K/V views are shared across the layer graphs in the set, so changing
`ne[1]` on them perturbs the whole set's shape-dependent state - the scheduler's allocation plan and
the recurrent layers' view of the shared cache - for every subsequent token. **Mirroring the PP
patch's tensor mutation is therefore NOT safe on the TG path, because PP rebuilds per-chunk sets
that each own their views while TG reuses one set.**

**Corrected understanding of the ctx -> cost law (the host-only evidence that motivated this).**
The law is real - same engine, same 50,000-token prompt, ctx 50,176 -> 128,000 is 2.551x and the
decode FA kernel goes 228.3 -> 557.6 us (2.442x, 4.2% off). But **it is not evidence that the kernel
scans beyond `n_kv_valid`**, because narrowing the scan length made things worse, not better. The
consistent reading is that cost tracks `ctx` through the kernel's **launch geometry**
(`fattn-common.cuh:1085-1190`: `ntiles_KV`, `parallel_blocks`, `stream_k` all derive from
`K->ne[1]`), not through the amount of KV actually read. That points the next experiment at
**parallelism and occupancy**, not at scan length: the decode kernel is the non-tensor-core
`flash_attn_ext_vec<256,1,Q8_0,Q8_0>` launched at **grid (1,10,16) = 160 CTAs x 128 threads,
regs=255** - about 5.9% of a 5090's resident thread slots.

**AC5 NOT EVALUABLE as run, and the reason is a harness mistake, not a result.** `check-route-parity.py`
reported 40 diverging layers per chunk in **both** arms, including the untouched pre-fix control. The
parity contract is defined for a clean `--pp-repeat 2` trace; `ab19.plan` omitted `rep=`, so the
harness defaulted to **3 passes**, and a 3-pass trace cannot satisfy the exactly-twice contract. The
divergence is an artifact of the plan, and it is identical in both arms, so it says nothing about the
patch either way. **Any future T-GPLR-70 arm must set `rep=2` explicitly.**

**A second, independent finding, because it undercuts how this project records build identity.**
**Locally built `llama.dll` is NOT reproducible.** Two consecutive no-op rebuilds of byte-identical
source produced `0d98d81c0189634d` and `e131f822f999d9b0`. So a DLL SHA-256 identifies *a
particular file that was side-copied*, and nothing more. Statements like "changed arm is
llama.dll bd4718f45b186fd7" remain accurate as descriptions of preserved artifacts - the side-copy
rule is what makes the A/B valid - but a hash must never be presented as a reproducible build
identity. **The reproducible identity is the SOURCE digest.** That is what actually proved the
revert: `revision digest 28382EF598129F94` equals the `source_digest` the ab16 verdict recorded for
the pre-T-GPLR-70 tree.

**What survives from the original investigation.** The measured facts stand and are unaffected by
this falsification: KV is on CUDA0 for exactly the 10 attention layers and `flash_attn` is enabled,
so host/PCIe KV reads and non-FA paths are eliminated; `n_embd_k_gqa = 512` over `n_head_kv = 2`
gives head_dim 256; the attention-layer penalty is 1,435 us host vs 1,375 us device (+4%) and so is
independent of expert placement; and the decode kernel runs at ~5.9% occupancy.
**CORRECTION (prof3, 2026-10-02): the sentence that followed here - "Only the interpretation 'the
kernel reads 262,144 slots instead of 50,128' is withdrawn" - was itself wrong and is now reversed.**
The scan waste is real and independently measured; only the **patch mechanism** was falsified.

### T-GPLR-71 (MEASURED, owner-directed 2026-10-02): size the TG attention view from live n_kv at graph-build time, via `n_pad` + a post-prefill TG set rebuild

**Status: MEASURED** (promoted by owner 2026-10-02; steps 1-5 + ab20-ab22 executed 2026-10-02; host gates + ab23 cross-session replication revalidated 2026-10-08). Owner acceptance pending.

**Dependencies.**
- **T-GPLR-47** — the fresh-context swap-in-slot rebuild, i.e. the `needs_rebuild` / `rebuild_replacement` / `begin_set` replacement branch this task REUSES rather than reimplements.
- **T-GPLR-62 S2** — `mark_row_set_sets_stale()` and the flagged-set routing in `layer_streaming_compute` (`:4330-4350`).
- **T-GPLR-70 + its prof3 addendum** — the analysis being fixed (decode FA 5.116x across a 5.22x ctx sweep; **+9,000.4 us/token PREDICTED** = 20.8% of the TG wall; not re-derived this pass) and the falsified mechanism this task must not repeat.
- **T-GPLR-68 / T-GPLR-69** — VRAM attribution, and the `peak_vram_bytes` defect that makes AC4 nvidia-smi-only.
- **ab19 `ctl38x`** — the baseline binary set and the 23.096 tok/s / 3,089.6 tok/s PP reference numbers.

**Commit:** _(unfilled — no commit is authorised; leave blank until the owner asks, and record the hash here only if one is made)_

**Goal.** Remove the **9,000.4 us/token PREDICTED** dead-KV scan of the T-GPLR-70 addendum above (measured recovery: **9,026.2 us/token** of compute), **without mutating a reused graph**, and without regressing PP (charter priority 1).

**Root cause (code facts, read from source).** The attention view length is chosen at graph-build time by `llama_kv_cache::get_n_kv()` (`src/llama-kv-cache.cpp:1233`):
`n_pad_cur = max(n_pad, 256)`, `n_kv = min(cells.size(), max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur)))`.
`cells.used_max_p1()` is the live position, so **upstream already sizes from live**. Our runner pins the override at `tools/layer-streaming/main.cpp:4470`: `cparams.n_pad = (uint32_t) n_ctx_total;` which forces `n_kv = 262144` for every graph. Stock uses the default `n_pad = 0` (`src/llama-context.cpp:3936`) and therefore gets live-sized views.
**The handoff executor is not involved and needs no change.** Any fix aimed at it is aimed at the wrong layer.

**Scope.**
- (a) Replace the `n_pad = n_ctx_total` pin with a granularity that keeps graphs reusable (must satisfy `GGML_ASSERT(kv_size % n_pad == 0)`, `llama-kv-cache.cpp:98`; `kv_size = 262144`, so 256 is legal).
- (b) Build/rebuild the **phase-1** set **after prefill**, when `get_n_kv()` is truthful, and key the TG set on `n_kv` so later growth forces a rebuild rather than silent staleness. Reuse the existing `needs_rebuild` / `rebuild_replacement` / `mark_row_set_sets_stale()` path (T-GPLR-62 S2, T-GPLR-47).

**Non-goals.** No change to the handoff executor. No tensor mutation of a prebuilt graph (T-GPLR-70's mechanism, falsified). No change to routing, expert IDs, arithmetic or checkpoint metadata. No change to `--pp-schedule`, the act slab, or slot policy.

**Design / failure contract.**
- A set whose built view can no longer cover the live position MUST be rebuilt, never silently reused. An under-reading view is a correctness failure (wrong tokens attended), so this fails loudly or not at all.
- Recurrent state is NOT re-initialised by the rebuild: the prebuild must not be simply re-run after prefill, because it advances recurrent session state (`llama-context.cpp:1748-1750`). A post-prefill rebuild must reuse the T-GPLR-47 replacement semantics.
- KV cache and recurrent state stay owned by logical layer; nothing here may move them into a weight slot.

**Why (a) alone is forbidden.** TG sets are built at bring-up (`llama_context::layer_streaming_prebuild`, TG branch `:1746`), before any cells are applied, so `get_n_kv()` would return the 256 floor and generation would read only the last 256 positions. **(a) without (b) is a regression, not a fix.**

**Known PP risk that must be gated.** PP already narrows its own view per chunk (`T-Q38FA-01`, `llama-layer-streaming.cpp:5602-5610`), and in phase 0 `use_implicit_causal_mask()` is true (`llama-graph.cpp:1490`) so **no `kq_mask` tensor is allocated** (`llama-graph.cpp:2798`) — `n_pad` therefore does not change PP VRAM. The open risk is layout: the per-chunk sets assume constant shapes for "non-aliasing act-slab slices" (`llama-context.cpp:1537-1548`) and PP currently reports `graph_builds: 0`. Any extra timed build is a direct PP loss.

**Implementation steps (B1 — reuse the T-GPLR-47 replacement path).** Ordered; each step is independently checkable.
1. **Side-copy the control BEFORE rebuilding.** Copy the exe **and all DLLs** from `build/llama-cuda/bin/gplr66-fixed` to `build/llama-cuda/bin/t-gplr71-control`. Rebuilding destroys the control, and the streaming code lives in `llama.dll`.
2. **`n_pad` granularity.** At `tools/layer-streaming/main.cpp:4470`, replace `cparams.n_pad = (uint32_t) n_ctx_total;` with a granularity constant (default **256**, which divides `kv_size = 262144` and satisfies `GGML_ASSERT(kv_size % n_pad == 0)`, `src/llama-kv-cache.cpp:98`). Expose it as `--kv-graph-pad` so a later arm can sweep it **without a rebuild**.
3. **Rebuild the phase-1 set after prefill.** At handoff, mark the phase-1 (TG) sets stale so `layer_streaming_compute` rebuilds them on the first TG token through the existing replacement branch (`needs_rebuild` + `rebuild_replacement` + `begin_set`). **Do not add new lifecycle code and do not re-run `layer_streaming_prebuild`** — re-running it would advance recurrent session state (`src/llama-context.cpp:1748-1750`). This step is what makes `mctx->get_n_kv()` truthful: at bring-up it returns the 256 floor.
4. **Key the set on `n_kv`.** Add the `n_kv` observed at build time to the graph-set key beside `chunk_key` (`src/llama-layer-streaming.cpp:4330`), so a set whose built view can no longer cover the live position is **rebuilt, never silently reused**. With granularity 256 and `tg=128` this must fire **at most once**; assert it in telemetry.
5. **Make the built view observable.** The `tel_kv_view` / `tel_mctx_lifecycle` emitters are pw-probe-gated, so a defaulted trace cannot answer AC1. Emit the per-phase built view length in `phase_summary` (one field) so AC1 is checkable from the trace alone.
6. **Host validation** (steps 2-5), then **one** GPU A/B. Do not spend more than one mutex hold.

**Falsifiable acceptance criteria.**
1. `phase_summary phase=tg` shows decode FA cost flat in ctx across a sweep (today it is 5.116x over a 5.22x ctx change) — target <=1.05x.
2. TG tok/s improves by >=15% at the 8 GB operating point vs `ctl38x` (23.096 tok/s), with PP unchanged within run-to-run noise.
3. **PP does not regress**: `graph_builds` stays **0** in `phase_summary phase=pp`, and PP tok/s >= 3,089.6 x 0.98. (State the observed session-to-session spread alongside the result; a bare "within noise" is not evidence.)
4. VRAM peak does not increase (`gpu-mem-poll` nvidia-smi only; `run_end.peak_vram_bytes` is not a measurement, see T-GPLR-69).
5. Semantic-QA smoke tier passes (T-GPLR-07 ladder), since generation output must be unchanged.
6. **The TG rebuild fires at most once** for `tg=128` (step 4), and **zero** times in PP.

**Exact validation commands and expected results.**

Host only (no mutex needed; CPU-only compile and checks are exempt from the GPU mutex).
```
# 1. Build. CUDA_PATH is REQUIRED to build and not to run (AGENT_ENVIRONMENT.md addendum F).
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
cmake --build C:/Development/AI/qwen38GZI/build/llama-cuda --config Release
```
Expected: exit 0, no new warning in `llama-kv-cache.cpp` / `llama-layer-streaming.cpp`.
```
# 2. Host suite. Emits the SOURCE DIGEST - the reproducible build identity.
tools/run-tests.ps1 -Suite host
```
Expected: exit 0. Record the printed `revision digest`; it must be recorded in this block.
```
# 3. Focused CTest (host-capable targets only). -C Release is mandatory: this tree is
#    multi-config and without it every test reports "Not Run" (exit 1) - a usage error,
#    not a failure.
ctest --test-dir C:/Development/AI/qwen38GZI/build/llama-cuda -C Release -R "test-layer-streaming-(route-topk|host-mem)$"
```
Expected: exit 0, **2/2**.
**Do NOT widen this regex.** `-R "layer-streaming|route-topk|host-mem"` selects **111** tests, of which **25 are model-backed `-run` targets** that fail with `missing or empty model environment variable: Q38LS_MODEL_PATH`. They initialise CUDA, so they belong to the GPU rung with the env var set inside the mutex hold, not to host validation. Measured 2026-10-02: the wide regex gives 86/111 with all 25 failures environmental, which reads like a regression and is not one.

GPU (exactly one mutex hold; `run-ab-session.ps1` does NOT take the mutex itself).
```
# 4. Write build/stock-levers/ab20.plan first:
#      ctl38x  lm ub=1024 ctx=262144 kv=q8 slots=2 pp=50000 bin=fixed route=1 tg=128 ho=1 hpart=38
#      g71x    lm ub=1024 ctx=262144 kv=q8 slots=2 pp=50000 bin=g71  route=1 tg=128 ho=1 hpart=38
#    and add g71 -> build/llama-cuda/bin/t-gplr71-changed to $lmBinDir in run-ab-session.ps1.
tools/with-gpu-lock.ps1 -Command "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Development\AI\qwen38GZI\build\stock-levers\run-ab-session.ps1 -Session ab20 -PlanFile C:\Development\AI\qwen38GZI\build\stock-levers\ab20.plan"
```
Expected: verdict PASS, `gpu_pre`/`gpu_post` clean, both arms exit 0. **No compiles inside this hold.**
```
# 5. AC1 ctx sweep, reusing the prof3 machinery (arms lm@50176, lm@128000, lm@262144),
#    then re-derive with the tool instead of by hand:
python build/stock-levers/tg-decode-scan-length.py
```
Expected: the "ours" row reads <=1.05x across the 5.22x ctx sweep (today: 5.116x).
```
# 6. Semantic-QA smoke tier (T-GPLR-07): 512-token default, aggregate checker PASS
#    including the negative control failing as designed.
```
If route parity is evaluated, the plan must set `rep=2` explicitly - `run-ab-session.ps1` defaults to 3, which **cannot** pass and made ab19's AC5 unevaluable.

**Required state/doc updates.** `docs/currentstate.md`, this block, and the AGENTS.md gotcha for `n_pad`.

**Execution record -- steps 1-2, 2026-10-02 (host-only; NO GPU window).** Control side-copied to `build/llama-cuda/bin/t-gplr71-control/` (16 files, hashed, re-verified after every later rebuild; identity `llama.dll bcd5ddac455ca2b6` / exe `70e35f47acf27e17` = ab19 `ctl38x`). `--kv-graph-pad N` added to `main.cpp` as **opt-in**, not defaulted: with 256 as the default and steps 3-4 absent, generation would attend to a few hundred positions. A positive `N` therefore refused with exit 2. Host suite 76/76, digest `E1AA5A5DDB92ECB7`. **Both of those are superseded by the steps 3-4 record below: the refusal is lifted and the digest has moved.**

**Execution record -- steps 3-4, 2026-10-02 (host-only; NO GPU window; nothing measured).**

*Hypothesis, stated before the first edit.* If each phase-1 set records the `n_kv` it was built against and the executor arms the T-GPLR-47 replacement branch whenever that view can no longer cover the live `get_n_kv()`, the bring-up TG set is rebuilt once on the first TG token, and the legacy `n_pad = n_ctx_total` pin makes both sides the same constant so nothing fires.

*Cheapest disproof, run first, passed* (three numbers from `llama-kv-cache.cpp:1233`). Layer-major implies `pw_probe` (`main.cpp:3201`), so the prebuild builds WIDTH CLASSES -- chunk 0 at [0,1024), chunk 1 at [1024,2048), tail(848) at [2048,2896) -- then the phase-1 widths at `pos_c = applied = 2896`.

| config | bring-up set | required, 1st TG | required, tg=128 | arms? | refires? |
| --- | --- | --- | --- | --- | --- |
| legacy `n_pad = n_ctx` | 262,144 | 262,144 | 262,144 | no | no |
| opt-in pad 256 | **3,072** | **50,176** | 50,176 | **once** | no |

*CORRECTION: "Why (a) alone is forbidden" was right for the wrong reason.* It claimed `used_max_p1()` is 0 at bring-up, giving a 256 floor. False: `build_variant` calls `vmctx->apply()` BEFORE `model.build_graph()`, and layer-major's width classes apply real positions, so cells exist. The floor is **3,072**, not 256. Conclusion unchanged; mechanism corrected.

*Step 3 done:* `arm_tg_view_rebuild()` sets exactly T-GPLR-62 S2's two flags, so `begin_set` takes the identical T-GPLR-47 fresh-context branch. No new lifecycle code; the prebuild is not re-run. Recurrent safety is structural, not argued: the replacement re-derives graphs and applies nothing, whereas the prebuild calls `vmctx->apply()`.

*Step 4 done, deliberately NOT via `find_set`'s key.* A narrower view must REPLACE its set; putting width in the variant identity would allocate a second set beside the stale one. `graph_set::n_kv` is recorded in `finish_build()` and compared by `view_covers_live()`.

*Deviation from the plan, deliberate:* the granularity is **not** `cparams.n_pad`. That pin also sizes the phase-0 sets, which T-Q38FA-01 only ever SHRINKS; under the pw width classes the continuing class is built at ~2,048 columns while chunk 48 drives it with a 50,000 window, so a context-wide 256 pad would force an EXPAND. The pin is unchanged and the granularity is a phase-scoped override (`llama_memory_i::set_view_pad` -> `llama_kv_cache::n_pad_view`, 0 = off) armed at the prefill -> generation boundary. `--kv-graph-pad 0` never calls it.

*Refusal lifted* because its own named prerequisite now exists, wired to a branch already proven in situ, and shown by the table to fire only when the built view cannot cover the live one. This establishes nothing about hardware behaviour.

**Execution record -- four-dimension audit follow-up, 2026-10-02 (host-only).**

The audit found the fail-safe ordering self-defeating and it is fixed. `view_covers_live()` read an unrecorded width as "safe to reuse", so a set that could not prove coverage opted out of the check that prevents under-reading. Now: `required <= 0` -> nothing to check, reuse; `built <= 0` -> **coverage UNPROVEN, rebuild**; else `built >= required`. Second, `cur_build_n_kv` was never reset, so an un-instrumented build inherited the PREVIOUS build's width and stamped a plausible but unverified number; `finish_build()` now records unconditionally and then clears to -1. Unknown costs at most one extra rebuild, because the rebuild goes through `layer_streaming_compute`, which records the real width.

*All five `model.build_graph` sites, read individually:*
1. `llama-context.cpp:1526`, `layer_streaming_prebuild`'s `build_variant` -- **records** (set after `vmctx->apply()`, from the same context the builder reads).
2. `llama-layer-streaming.cpp:4386`, `layer_streaming_compute`'s build-on-miss -- **records**.
3. `llama-context.cpp:1392`, `process_ubatch` -- **does not record, and CAN reach a live set.** The streaming branch returns at `:1363` only for `LLM_GRAPH_TYPE_DEFAULT`, so a non-default type (the MTP draft head on `ctx_dft`, which shares the same `llama_layer_streaming`) falls through here with `p.layer_streaming` set, because `graph_params()` attaches it unconditionally at `:2881`.
4. `llama-context.cpp:2836`, `graph_reserve` -- reaches `begin_build` only if `layer_streaming != nullptr`, but it runs at context init before the runner attaches it, so in practice it cannot touch a live set. Uninstrumented regardless.
5. `llama-context.cpp:3845`, `opt_epoch_iter` -- **does not record and can reach a live set** when speculative graph optimisation is enabled.

Sites 3-5 are therefore covered by the fail-safe, not by instrumentation, which is the reason for that choice: instrumenting them would mean casting a hybrid memory context inside three generic llama.cpp paths, and a site that records the WRONG width is worse than one that records none.

*Storm bound.* In the ab20 operating point (`--no-mtp`, no speculative opt) sites 3 and 5 are unreachable, so the unknown-width branch cannot fire there at all and the count is the predicted single rebuild. With `--mtp` it is unmeasured: if the MTP path rebuilt a phase-1 set on every token, each such build would stamp -1 and each TG token would rebuild. `arm_tg_view_rebuild` emits `built_n_kv`/`required_n_kv` per firing, so a storm is directly countable from a trace rather than inferred. **That event is kept deliberately:** plan step 4 states the rebuild "must fire at most once; assert it in telemetry", so it is the contract's own requirement for AC6 and the only storm detector -- not an added reporting surface. Step 5's `phase_summary` field remains unwritten and gets its own pass.

*Audit's other findings, not fixed here:* the avoidable `llama_memory_i` virtual (three forwards for one integer; its vtable hazard already cost one test failure), duplicated prose, and that no PP/TG behaviour has been measured.

*Validation after the fail-safe fix:* build exit 0 (no new warning); `tools/run-tests.ps1 -Suite host` 76/76; focused ctest 2/2; `resplit-currentstate.py --verify` PASS; `git diff --check` clean. Source digest **`381E65E7B85DE7E7`** (11 tracked changed; `llama.dll 2FFB3AEA3A1BCFCA`; record `build/llama-cuda/suite-runs/20261002T123710Z-host/revision.json`). It supersedes `F4C398FB032C2646`, which was the steps 3-4 pre-audit-fix digest. **The two pre-existing uncommitted mods** (`src/llama-layer-streaming.cpp` lines ~1185-1219 and `src/models/qwen35moe.cpp`) were left untouched apart from this task's own hunks.

*INCIDENT, task-owned: this plan file crossed the doc-size FAIL band (480,000 B) during these records and failed the host suite's doc-size guard.* These two execution records were compressed in response (they had duplicated source comments and `currentstate.md`), which is also the audit's QUALITY finding. **The file remains close to the cap and needs a Planner-level split before the next task adds to it** -- moving closed task blocks into `docs/plans/generic-prefill-layer-residency-archive.md` is the documented mechanism. That is a structural decision, not one this pass should take unilaterally.

*AC1-AC5 measured by ab20/ab21 below; AC6 by ab22. `--kv-graph-pad` HAS now been executed on hardware.*

**Step 5 (2026-10-02, host-only).** `phase_summary` now carries `kv_view_built`, `kv_view_live` and `view_rebuilds_total` unconditionally (pw-probe gating left on the per-kernel emitters); counters are cumulative because PP passes share ONE `graph_builds` baseline. Score AC1 on the **absolute** `kv_view_built` -- the legacy pin makes both sides 262,144, so a ratio test would pass the control it fails in substance. Observability only. Scored in ab20/ab22 below.

**AB20/AB21 EXECUTION RECORD, 2026-10-02 (first hardware execution of `--kv-graph-pad`, 2 mutex holds).** Control `build/llama-cuda/bin/t-gplr71-control` (ab19 `ctl38x`, re-verified 16/16); candidate `build/llama-cuda/bin/t-gplr71-changed` (`llama.dll 14f4e04d98f9bd6e`). `pp_prebuild_per_pass` read as RESOLVED from `run_start`: **1, source `default`** (rep=2 + KV q8). Score AC1 on absolute `kv_view_built`. Telemetry `build/llama-cuda/logs/ab2{0,1}-*.jsonl` (+`.mem.csv`), `gplr07-qa-512{,-kvpad256}.*`.

| AC | Result |
|---|---|
| AC1 | **PASS.** `kv_view_built` = **50,176** at ctx 262144 / 131072 / 65536 vs the control's 262,144. TG across a 4.00x ctx sweep: control **0.827x**, candidate **1.000x** (target <=1.05). Kernel-level (nsys) flatness NOT re-measured. |
| AC2 | **PASS.** TG 23.407 -> **29.609 tok/s = +26.5%** same-hold (+28.2% vs the 23.096 baseline), target >=15%. Compute 39,281.8 -> 30,255.6 us/token: **9,026 recovered** vs the 9,000.4 predicted (0.3%). |
| AC3 | **PASS.** `graph_builds` **0** on every PP line, both engines, all ctx. PP 3,089.0 -> 3,096.7 (+0.25%); n=1 per cell. |
| AC4 | **PASS, figures corrected 2026-10-02 on re-derivation.** At ctx 262144 both arms peak at **8,160,768,000 B = 7,782.71 MiB = 7.600 GiB** (poller only) and are **byte-identical**; ctx 65536 at 5,039.94 MiB; the n=2 interleaved replication reads **byte-identical at all three contexts, 6/6 arms** (the ab21 +270,336 B at ctx 131072 was a transient). The 7,778.4 MiB / 7.596 GiB this row previously carried appears in no artefact and is withdrawn. |
| AC5 | **PASS on both paths.** qa-smoke 512: 5/5 needles, negative control FAIL exit 1, check-telemetry OK (30,547 events); repeated by hand **with** `--kv-graph-pad 256`, same PASS, `kv_view_built`=`kv_view_live`=768. |
| AC5b | Route parity **clean on all 6 arms** (480 events / 240 distinct / every tuple x2 / 0 singletons), exit 0. Required `rep=2` + `--route-telemetry 1`; the wrapper default `rep=3` cannot pass. |
| AC6 | **Superseded by ab22 below.** At `tg=128` the guard correctly had nothing to do; it fires at `tg>=177`. |

**CORRECTED (was: "the rule was never reached").** ab20 read `view_rebuilds_total: 0` because `tg=128` stops 49 tokens short of the first pad boundary, NOT because the decode set is a build-on-miss. The set IS found and built (`rs_z -1`) and the guard is evaluated on every TG token; it has nothing to do until live outgrows the built width at token 177 (see ab22). ab20's speedup still came from the step-2 `n_pad_view` override sizing that build. **Do not read `view_rebuilds_total: 0` at `tg<=176` as unreachability.**

**Withdrawn.** The plan's predicted ~16x-too-narrow bring-up set (3,072 vs 50,176) is not what runs. AC1's flatness is scored from end-to-end TG tok/s, NOT the nsys FA-kernel scan, so prof3's ratios are **not** re-measured.

**Confounds.** TG ids differ between arms, but the same control config also differs across sessions (pre-existing nondeterminism, not this change); cache uncontrolled; n=1 per cell; candidate dips 6.5% at ctx 131072 (unexplained); **provenance never supplied, not refused** (no `REFUSED` token exists in any verdict; `ab20` carries `binary_state=not-supplied` and there is no session-level `ab21` verdict), identity resting on hashing both binary directories before and after. Detail in `docs/currentstate.md`.

**GUARD PROOF, 2026-10-02 (host test + ONE hardware arm).** Steps 3-4 are no longer reviewed-but-unexecuted.

*Host.* New `tests/test-layer-streaming-tg-view.cpp`, 39 checks, in `-Suite host` (now **77/77**, was 76). It drives the **production** symbols, so two were extracted to be reachable: `should_arm_view_rebuild` (the shipped guard `if`) and `record_built_view` (the stamp+reset inside `finish_build`, whose sanity returns hid the reset). Covers all three directions, the reset, the flag pair, the storm arithmetic, and the legacy pin staying inert.

*It found a real latent defect.* Driving `arm_tg_view_rebuild` for the first time **ever** killed the process with `0xC0000094` -- the divide-by-zero already measured and fixed at `llama_ls_tel_mark` (`:1312`): `ggml_time_us()` divides by the static `timer_freq` that `ggml_time_init()` sets, and `arm_tg_view_rebuild` called it without that guard. Harmless on the runner, invisible to review **because the function had never run**. Fixed with the repo's own one-liner.

*Configuration derived, not guessed.* Case (f) models the runner's real initial condition (decode set built on-miss at `GGML_PAD(50001,256)`=50,176) and predicts the guard's **only** firing at generated **token 177**, live 50,432.

*ab22, one hold, one arm, `tg=256`: prediction exact.* One `tg_view_rebuild` event, `built_n_kv 50176`/`required_n_kv 50432`; `view_rebuilds_total` **0/0/1**; `kv_view_built`=`kv_view_live`=**50,432**; `graph_builds` 0/0/2; VRAM **7.600 GiB** = 8,160,768,000 B (re-derived; unchanged); route parity clean; TG **29.513** vs 29.609 tok/s at `tg=128`, so the mid-generation rebuild costs nothing measurable. **Steps 3-4 are exercised, not dead code.** Provenance was never supplied, not refused (`binary_state=not-supplied`); the hash attests a directory, not the running process. Detail in `docs/currentstate.md`.

**Escalation condition.** If any extra PP graph build cannot be avoided, or if the act-slab layout cannot be made chunk-shape-independent without new VRAM, **stop and escalate to the owner**: the charter ranks PP above TG, and a TG gain bought with PP graph builds or added VRAM is not a win.

### T-GPLR-70 ADDENDUM (prof3, 2026-10-02, host-only): stock's scan length is ESTABLISHED, and the "42x" gap was a warmup artifact

Tool `build/stock-levers/tg-decode-scan-length.py`; evidence `prof3-{anchsm,anch,anchbig,lm,lmbig}.sqlite`.
Method: identical decode kernel `flash_attn_ext_vec<256,1,Q8_0,Q8_0,0>` on `grid (1,10,16)` /
`block (32,4,1)` in **both** engines, one decode pass per trace (10 attention layers = one token),
`--ctx-size` swept 50,176 / 128,000 / 262,144 = **5.22x**.

| engine | 50,176 | 128,000 | 262,144 | ratio over the 5.22x ctx sweep |
| --- | --- | --- | --- | --- |
| stock | 2,695.6 us | 2,661.4 us | 2,677.8 us | **0.993x - flat** |
| ours | 2,282.7 us | 5,966.3 us | 11,678.2 us | **5.116x - proportional** |

- **Stock scans live tokens only; ours scans the whole allocated cache.** Stock's scan length is
  pinned by elimination: reading 262,144 slots would require 1,065 GB/s, while the same kernel at
  ctx 50,176 over the same 546 MB achieves 202 GB/s. The only self-consistent value is ~50,128.
- **Both engines run in the same efficiency band (stock 203-205 GB/s, ours 233-244 GB/s)**, so the
  gap is *bytes scanned*, not kernel efficiency. That also closes the "~230 GB/s is suspicious"
  worry: stock is equally far from the card's ~1,792 GB/s.
- **The defect below is CONFIRMED; +9,000.4 us/token at ctx 262,144 is the PREDICTED size, not a**
  **measurement** (the measured recovery is 9,026.2 us/token of compute, 8,949.4 of wall), 20.8% of the
  43,297 us/token TG WALL measured by ab19 `ctl38x` (23.096 tok/s) at the validated 8 GB operating
  point.** (Its layer-compute sum, 39,996.6 us/token, is a different quantity and must not be
  substituted for the wall.)
- **WITHDRAWN: "stock's decode FA is 13.2 us" and the "42x" gap.** That row used
  `flash_attn_ext_vec<256,**2**,..>` on `grid (1,1,16)` = llama.cpp's **2-query warmup at
  `n_past ~ 0`**, not decode. Stock's real decode cost is 2,661 us. Correct ratios, which must
  **always carry their ctx**: **0.847x / 2.242x / 4.361x** at 50,176 / 128,000 / 262,144. At ctx
  50,176, where allocated ~= live, **we are faster than stock**.
- **CORRECTED byte width:** Q8_0 packs 34 B per 32 values, so KV is **1,088 B/token/layer**, not the
  1,024 B an f16 assumption gives. **Q8_0 KV is 1.0625x LARGER than f16 KV**, not smaller - relevant
  to any future f16-vs-q8 size argument. The "51.33 MB of real work" figure is withdrawn.
- **The ~29.8 tok/s / 0.79x projection stays withdrawn** - the premise is confirmed but no working
  fix exists. **Next probe:** whether the handoff executor can size the TG FA view from the live
  position at *graph-build* time, or whether a per-token graph rebuild is required (which would cost
  the `graph_builds == 0` property the handoff path relies on). **Do not retry the tensor mutation** -
  that mechanism is falsified independently of the premise.

**The defect.** `llama-layer-streaming.cpp:5602-5610` (tag `T-Q38FA-01`) narrows the attention KV
view to the valid window on the **layer-major PP** path:
`n_kv_valid = chunks[c].token_start + chunks[c].n_tokens`, applied as
`t->src[1]->ne[1] = n_kv_valid; t->src[2]->ne[1] = n_kv_valid;` plus
`ggml_flash_attn_ext_set_causal(t, q_pos_base, n_kv_valid)`. **There is no equivalent on the TG
path**, whose graphs come from the handoff executor and are sized by `mctx->get_n_kv()` - the
allocated, padded cache. At ctx 262,144 with a 50,000-token prompt, **every generated token scans
262,144 KV slots instead of ~50,128: a 5.23x waste**, concentrated in the 10 full-attention layers,
which cost ~2,075 us each against ~640 us for the 30 gated-DeltaNet layers.
**Confirmed by the prof3 addendum above** (5.116x cost across a 5.22x ctx sweep, against stock's
flat 0.993x on the identical kernel).

**Measured evidence (host-only, existing nsys exports, no GPU, no rebuild).**
- `lm50` at ctx 50,176 -> decode FA 228.3 us; `lm` at ctx 128,000 -> **557.6 us**. Same engine, same
  50,000-token prompt (`run-prof.ps1` `$PpTokens = 50000`), same KV type, same launch geometry
  `flash_attn_ext_vec<256,1,Q8_0,Q8_0,0>` grid (1,10,16) = 160 CTAs x 128 threads. **2.551x ctx ->
  2.442x cost**, 4.2% off; cost per KV slot 4.551 vs 4.356 ns, constant to 4.5%.
- `prof2-v1on` at ctx 128,000 -> 595.5 us, reproducing `lm`.
- Extrapolation to ctx 262,144 -> **~1,166 us**, 81% of the measured 1,435 us attention penalty.

**Scope.** Narrow the TG decode attention's KV view to the live window, mirroring the PP patch, and
prove it. **Non-goals.** No change to the PP path (it is already correct); no change to the upstream
FA kernels or the `ncols1=1` vec specialization, whose 160-CTA geometry is a separate, upstream
occupancy limit; no VRAM trade; no default-path change.

**Falsifiable acceptance criteria.**
- **AC1.** With a decode profile at ctx 262,144 and a 50,000-token prompt, the decode FA kernel's
  average drops from the predicted ~1,166 us to **< 350 us**. Anything in between is a partial fix
  and must be reported as one.
- **AC2.** The same arm's TG `tokens_per_s` rises from 23.26 tok/s by **at least 15%**, and the
  measured peak VRAM does not increase by more than the compute-buffer drift (T-GPLR-68: ~78 MiB per
  ubatch doubling).
- **AC3 (falsifier).** If the decode FA kernel time does **not** track `n_ctx` when re-profiled after
  the patch, the waste is not in the KV view length and the whole hypothesis is wrong. The existing
  2-point ctx law makes this the sharpest available falsifier.
- **AC4 (stock parity check, mandatory before any ratio is quoted).** Establish stock's actual scan
  length at the same ctx and prompt. Stock's recorded decode FA is 13.2 us at ctx 128,000, which for
  51.33 MB would be ~3.9 TB/s and is therefore physically impossible - so stock's `n_kv` is
  **unverified** and the "42x" gap must not be quoted until it is.
- **AC5.** Route parity stays clean on the patched arm (`--route-telemetry 1`), proving the patch
  changes no arithmetic.

**Validation.** One GPU arm under `tools/with-gpu-lock.ps1`, decode-profiled, plus a host-side
trace check. No compiles during timing arms. Semantic-QA smoke tier if the change touches output.

**Escalate** if narrowing the TG KV view changes generated tokens - that would mean the PP patch and
the TG path disagree about what the causal window is, which is a correctness finding far more
important than the speedup.

### T-GPLR-68 (RESOLVED 2026-10-02, host-only): the unaccounted VRAM term is llama.cpp's own compute-buffer reserve

Status: **MEASURED (2026-10-02, host-only, no GPU, no rebuild)**. AC1 met, AC2 met, AC3 met by an
independent tool rather than by a falsifying arm, AC4 met **out of sample** for the residual and
**NOT met** for the compute-buffer law (stated below, not papered over). The fix - adding the term
to the capacity predicate - is **not started and not authorized**; see "What is left" at the end.
Dependencies: none. Re-derivation tool: `build/stock-levers/vram-gap-attrib.py`.
Commit: (none - measurement only, no source change)

**Why it matters, and why it is not a footnote.** The capacity predicate
(`llama-layer-streaming.cpp:6563`) accounts for `bounded_window + row_cache + prediction_pool +
act_slab + reserve`. Measured on ab16 (slots 2, ctx 262,144, KV q8, 50,000 tokens, our tree), that
`accounted` total is **6.451 GiB at ubatch 512, 1024, 2048 AND 4096 - constant** - while the
**measured per-process peak runs 7.293 / 7.596 / 8.214 / 9.568 GiB**. Everything the ubatch lever
buys or costs lives outside the predicate:

| ubatch | accounted GiB | measured peak GiB | unaccounted GiB |
| --- | --- | --- | --- |
| 512 | 6.451 | 7.293 | +0.842 |
| 1024 | 6.451 | 7.596 | +1.145 |
| 2048 | 6.451 | 8.214 | +1.763 |
| 4096 | 6.451 | 9.568 | +3.117 |

The unaccounted term is **4.6x larger than the whole accounted term's spread**, and it grows roughly
linearly in graph-set width. ab12 shows the same structure at slots 3 / ubatch 4096 with partial
handoff: accounted 6.938 GiB, measured 13.434-19.341 GiB, unaccounted +6.497 to +12.403 GiB.

**Consequences that are already established and should not be re-derived.**
- **The 8 GB question can never be answered by `--residency-budget-mib` or by the capacity
  predicate.** Both reason over `accounted`; the binding term is outside it. This is a second,
  independent reason (the first is that the flag never sees the act slab or the reserve) and it is
  why ab14 accepted a configuration that a real 8 GB card refuses.
- **The only trustworthy statement of fit is a MEASURED per-process peak**, which is why ab16's
  8 GB claim is stated against 7,778.3 MiB measured rather than against any predicted total.
- **Any capacity model that omits it will be wrong in the direction that matters** - it will admit
  configurations that do not fit.

**Scope.** Account for the unaccounted term. Candidates to separate, none yet proven: graph-set
compute buffers whose width scales with ubatch; the CUDA context (largely fixed); allocator slack;
the per-width graph-set retention under `--pp-prebuild-per-pass`. Only the first is expected to be
ubatch-dependent, and that expectation is currently unverified.

**RESOLUTION 2026-10-02 (host-only: `byte_account` + runner `.log` + `mem.csv`, no GPU, no
rebuild).** The unaccounted term is **llama.cpp's own scheduler compute-buffer reserve** — the
`sched_reserve: CUDA0 compute buffer size` line the engine already prints at load time and that
`byte_account` never counts. The name of the `graph_footprint` fields pointed at the right file
but neither field is the term:

| `graph_footprint` field | what the code actually computes | is it the term? |
| --- | --- | --- |
| `io_ctx_reserved_bytes` | `sum ggml_get_mem_size(io ctx)` (`main.cpp:5380`) — **host** tensor-descriptor metadata. 384.0 MiB, identical in all four arms. | **No.** Host RAM, and constant in ubatch. |
| `io_ctx_used_bytes` | `sum ggml_nbytes(t)` over **every tensor of every graph** (`main.cpp:5381-5383`) — a LOGICAL sum. 104,561 / 183,688 / 289,191 / 500,196 MiB at ubatch 512/1024/2048/4096. | **No.** ~50x the physical footprint; the gallocr reuses across graphs. |
| `slot_vram_bytes` | the slot window | **No** — already inside `byte_account.slot_window` (996.0 MiB, an 8,192 B alignment difference). |

**The attribution (ab16, MiB; every "actual" term is a number a tool printed, nothing modelled):**

| ubatch | measured peak | `byte_account` plan | GAP | weights | kv | state | **COMPUTE** | slab | slot | sum of actuals | residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 7,472.2 | 6,606.1 | 866.1 | 1,921.3 | 2,720.0 | 62.8 | **937.0** | 390.6 | 996.0 | 7,027.8 | 444.4 |
| 1024 | 7,782.7 | 6,606.1 | 1,176.6 | 1,921.3 | 2,720.0 | 62.8 | **1,218.0** | 390.6 | 996.0 | 7,308.8 | 473.9 |
| 2048 | 8,415.8 | 6,606.1 | 1,809.7 | 1,921.3 | 2,720.0 | 62.8 | **1,780.0** | 390.6 | 996.0 | 7,870.8 | 545.0 |
| 4096 | 9,801.8 | 6,606.1 | 3,195.8 | 1,921.3 | 2,720.0 | 62.8 | **3,032.0** | 390.6 | 996.0 | 9,122.8 | 679.1 |

- **AC1 (decomposition) MET.** The compute buffer carries **90.5% / 89.3% / 89.9%** of the growth
  from ubatch 512. What survives is a bounded, characterized residual — not a mystery.
- **AC2 (law) MET for the residual.** `residual = 411 + 0.0655 x ubatch` MiB; per-arm errors
  +0.0 / −4.0 / −0.0 / +0.0 MiB. Over the whole ab16 shape, `measured peak ~= 7,139 + 0.650 x
  ubatch` MiB — exact at both ends, −22 to −55 MiB in the middle (a sub-linear term, not noise).
- **AC3 (falsifier) MET by an independent witness, not by a toggling arm.** The claim was
  "compute buffers scale with graph-set width". llama.cpp sized that buffer itself and printed
  the answer; the per-width deltas are monotonic in ubatch (+281 / +562 / +1,252 MiB per
  doubling). **The `--pp-prebuild-per-pass` leg of AC3 is NOT exercised** — every ab16 arm ran
  with it at its default of 1, so there is no contrast arm. That sub-test remains open.
- **AC4 (predictive) MET OUT OF SAMPLE for the residual, NOT MET for the compute buffer.** The
  ab5 arms (`ctx 512,000`, slots 3, KV 5,312.5 MiB — a different session and a different shape)
  predict a residual of `411 + 0.0655 x 4096 = 679.3` MiB and measure **686.4 / 686.4 / 682.4**
  MiB across three repeats: **+7.3 MiB, 1.1%**. The compute buffer, however, is **not** a function
  of ubatch alone: at the *same* ubatch 4096 it is 3,032.0 MiB in ab16 and **5,472.0 MiB** in ab5.
  The whole of that difference is ctx/slots/KV. So the four-point ab16 law is a law **for a fixed
  ctx and slot count**, and a predictive compute-buffer law needs the ctx axis measured.

**The VRAM shape, which rules out two candidates outright.** From `mem.csv`, every arm is a single
**step at setup** that reaches its plateau by t≈10.9 s and then holds **flat** for the entire run
(20-29 of 24-32 samples at plateau): u1024 24/30, u2048 23/27, u4096 23/27. Nothing accumulates
per chunk, per layer or per pass. Therefore:
- **allocator slack / fragmentation from repeated PP passes: ELIMINATED as the growth term** (no
  growth exists to explain);
- **`--pp-prebuild-per-pass` retention: ELIMINATED as the growth term** (same reason);
- the CUDA context is inside the constant part of the residual, not the ubatch-dependent part.

**What the residual is, honestly.** ~411 MiB constant plus a slow 0.0655 MiB/token drift. A
constant of that order is the right size for the CUDA context plus driver and poller overhead on
Windows; the drift is **not separately attributed**. Stated as unattributed rather than guessed.

**Two corrections this produced, one of them a withdrawal-grade one.**

1. **`byte_account` also OVER-counts weights by +515.3 MiB.** It plans `static + per_layer =
   2,436.7 MiB`; llama.cpp allocated `CUDA0 model buffer = 1,921.3 MiB`. So the predicate is wrong
   in **both** directions at once, and the over-count happens to mask part of the under-count. `kv`
   and `state` agree exactly with `llama_kv_cache` / `llama_memory_recurrent` (2,720.0 / 62.8 MiB).
2. **`run_end.peak_vram_bytes` is not a measurement and has never been one on the default path.**
   It is `vram.total - vram.free_min` (`main.cpp:6424`), and `free_min` is updated **only** by
   `ls_vram_budget_refresh`, whose three post-start call sites are all inside
   `if (ls.row_set_retain)` (`:4956`, `:5036`) or `if (residency_tg)` (`:5742`). With the row cache
   and `--residency-tg` off — the ab16 shape — it is **never called again after
   `ls_vram_budget_measure_start`**, so `free_min == free_start` forever and the field is the
   free-VRAM-at-startup number echoed back: **3,424,190,464 B = 3,265.6 MiB, byte-identical in all
   five ab16 arms**, understating the true peak by 4,206.6-6,536.3 MiB. The console prints the same
   fiction (`peak VRAM: 3265.56 MiB`) and `q38ls-layer-streaming-profile` carries it into its
   report. Every peak quoted in this project has therefore come from the nvidia-smi poller, which
   is the right source. Filed as **T-GPLR-69**; not fixed here.

**Re-derived number, recorded because the charter requires it.** ab16 ubatch 1024's measured peak
re-derives to **7,782.7 MiB** (the plateau is exactly 8,160,768,000 B in 24 of 30 samples), not the
7,778.3 MiB carried in the PP-curve table — a 4.4 MiB (0.06%) discrepancy whose provenance is not
recoverable from the stored artifacts. Every downstream statement ("fits in 8 GB", the 413.7 MiB
headroom, the 3.72x / 0.62x comparison) is unchanged at that magnitude, but **the table figure
should be corrected to 7,782.7 MiB** and the 8 GB headroom is **409.3 MiB, not 413.7**.

**Act-slab suspect: still eliminated, and the general lesson stands.** The 2026-10-01 check stands
(`build/stock-levers/act-slab-residency.py`, `RATIO resident/reported: 1.000`, both halves bound,
`n_embd_io = n_embd_inp() = 2048` not 4096). It was never the term. **Lesson, now twice earned:
an accounting suspect is not a finding until the source's own unit is checked** — here 4096 vs
2048 manufactured an apparent 2x under-report that the data never contained, and **an accounting
suspect must be checked against the allocator that actually made the allocation, not against
another field of the same estimate.** `byte_account` was never the allocation log; `sched_reserve`
was.
**Non-goals.** No change to the predicate's *semantics* until the term is understood; no attempt to
make the number smaller in this task; no new dependency; no default-path change. Reporting an
unknown term as unknown is this task's deliverable, not hiding it.

**Falsifiable acceptance criteria.**
- **AC1 (decomposition).** At a fixed shape, the measured peak minus `accounted` is attributed to
  named components with per-component measurements, and the residual is either zero or bounded.
- **AC2 (law).** The ubatch dependence is characterized across at least the four widths already
  measured, with a stated form and its residuals. A claim of "roughly linear" is not accepted
  without the residuals.
- **AC3 (falsifier).** The claim "graph/compute buffers scale with graph-set width" is FALSIFIED if
  the per-width delta does not follow ubatch monotonically, or is FALSIFIED if it does not change
  when `--pp-prebuild-per-pass` is toggled while width is held fixed. Either outcome is a result.
- **AC4 (predictive value).** The characterization predicts a measured peak for a shape not in the
  training set, within a stated tolerance. A model that cannot predict an unseen shape has not
  solved AC2, however well it fits the four points it was built from.

**Validation.** Host-only first (byte accounting arithmetic over recorded `mem.csv` + `byte_account`
pairs); then at most one GPU arm per claim, through the mutex, no compiles during timing arms. All
existing traces (`ab16-*`, `ab12-*`) are legitimate inputs and are re-readable now that the UTF-16
log-reading trap is recorded in `AGENTS.md`.

**Escalate** if the unaccounted term turns out to be dominated by something the predicate is
*supposed* to own (which would mean the predicate is wrong, not merely incomplete), or if any part
of it is a per-run leak rather than a steady-state cost.

**Escalation condition: FIRED, and the answer is the first branch.** The term is dominated by
something the predicate is *supposed* to own — it is the model's own compute buffers, the same
family as the `static`/`per_layer`/`slot_window` terms already in `bounded`. So the predicate is
**wrong, not merely incomplete**, and additionally over-counts weights by 515.3 MiB. The VRAM is a
setup-time step, not a per-run leak, so the second branch did not fire.

**What is left (not done here, not authorized).**
1. Add the compute-buffer term to the capacity predicate — but it is **not known at preflight
   time**, because the predicate runs before the context exists and `sched_reserve` prints after
   it. This needs a *prediction* of `compute(ctx, slots, ubatch)`, which the current evidence
   does not yet support (see AC4). Predicting it by sweeping ubatch, as ab16 did, is the discovery
   method; predicting it from `ctx`/`slots` is the deliverable.
2. Fix the +515.3 MiB weight over-count, which is a pure accounting bug and needs no measurement.
3. The `T-GPLR-69` `peak_vram_bytes` defect.
4. The `--pp-prebuild-per-pass` leg of AC3 (one A/B arm; cheap, and it also pins the last open
   candidate from the original list).

### T-GPLR-69 (draft, found 2026-10-02 by T-GPLR-68): `run_end.peak_vram_bytes` reports free-VRAM-at-startup, not a peak

Status: DRAFT. Found during T-GPLR-68 attribution. **Not started, not authorized.** This is a
report-only defect in the experimental path; it is not a capacity bug, but every future capacity
claim that trusts the field would be wrong by 4-6 GiB.
Dependencies: none.
Commit: (none)

**The defect.** `tel_run_end(exit_status, vram.total - vram.free_min, ...)` (`main.cpp:6424`)
reports the device low-water mark. `free_min` is advanced **only** inside
`ls_vram_budget_refresh` (`main.cpp:692-698`), which is called from exactly four places:
`ls_vram_budget_measure_start` (`:4478`, before anything is allocated), and three later sites
guarded by `if (ls.row_set_retain)` (`:4956`, `:5036`) or `if (residency_tg)` (`:5742`). On the
default path — no row cache, no `--residency-tg` — **none of the three later sites runs**, so
`free_min` never moves below `free_start` and the field is `total - free_start`.

**Evidence.** All five ab16 arms emit `peak_vram_bytes = 3,424,190,464` byte-for-byte identical
across ubatch 512/1024/2048/4096 and the V1 reference arm, while their measured peaks span
7,472.2-9,801.8 MiB. `total - free_start = 25,650,855,936 - 22,226,665,472 = 3,424,190,464`,
which is exactly `byte_account.free_vram` subtracted from `byte_account.total_vram`. The T-GPLR-24
"low-water mark" fix is therefore correct **only** for row-cache / `--residency-tg` runs — which
are exactly the runs the fix was written for — and silently inert everywhere else. The console
line `peak VRAM: 3265.56 MiB` and the `q38ls-layer-streaming-profile` report field inherit it.

**Falsifiable acceptance criteria.**
- **AC1.** With row cache and `--residency-tg` both off, `run_end.peak_vram_bytes` tracks a real
  change in device usage across two arms with different ubatch (today it cannot: they are equal).
- **AC2.** The fix samples the device at least once after the compute buffer is allocated, so the
  field is never below the sum llama.cpp reports it allocated.
- **AC3 (falsifier).** If adding the post-prebuild refresh moves `peak_vram_bytes` to within
  ~2% of the nvidia-smi plateau on ab16's four arms, the field becomes a usable cross-check on
  the poller. If it does not, the discrepancy is the driver's own accounting and the field stays
  advisory.
- **AC4.** Every existing fixture that pins a `peak_vram_bytes` value keeps passing unchanged —
  the fixtures are synthetic and must not be rewritten to match.

**Escalate** if sampling the device inside the timed region costs measurable PP throughput; the
correct response would then be a post-run out-of-band sample, not a loop of in-loop reads.

---

## Appendix: re-derived ledger for T-GPLR-68 (2026-10-02)

`build/stock-levers/vram-gap-attrib.py` reads three independent sources per arm and prints the
ledger above. It is the re-derivation path for every number in this block. One trap it encodes:
`mem.csv` `dedicated_bytes` is in **bytes** while `byte_account` is compared in **MiB**, and a
first cut of this script compared them directly and reported every arm as "peak reached at
t=0.00s". The unit bug produced a *correct-looking* conclusion for the wrong reason — the same
class of error as the 4096-vs-2048 act-slab trap, and the reason the shape line in the tool now
carries an explicit warning comment.
