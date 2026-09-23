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

Status: in-progress (2026-09-21) - A1 executed host-only; A2 executed (instrument +
recon generation join fix, host-validated); A3 executed (PASS 2/2, CHUNK0_SPIKE on four
independent arm-runs); the one-off itself is now ATTRIBUTED host-only (the driver-side commit
of the 17.23 GiB registered mapping, absorbed by chunk 0's embed step before any staging copy;
see the WHAT THE ONE-OFF IS section); A4 executed THREE times (PASS 6/6, PASS 8/8 with two added control
modes, PASS 13/13 with a dose series). The dose curve is in: absorption of the pass-1
excess is ~1:1 in the length of the wait, the wall is conserved at every dose, and no dose
shortens the total - so the pre-pay can be shortened but only by leaving equally much excess
behind, and pre-warm cannot be a wall-time win at any length. The A4 controls SETTLED the mechanism question the wrong way for this block's
first reading: a duration-matched NO-TOUCH delay removes 93.2% of the pass-1 excess, so
the one-off is a time-gated completion rather than the source-span page residency the
second attempt inferred, and both pre-registered mappings (A3's CHUNK0_SPIKE -> H3 and
A4's branch (i) -> H1) are unsupported. The pre-registered ESCALATE disposition applies:
the hypothesis set is incomplete, nothing is adopted, and the remaining work is a
Planner decision plus a dose-response arm, not a re-run. No rule, threshold or accepted
verdict was changed by any A4 session.
Dependencies: T-GPLR-52 (its A1/A2/A3 verdicts are inputs AND correction targets: the
queue-behind-compute law stands, but "staging and compute are flat across passes"
describes the LAYER sweep only - the pass-1 excess lives before it); T-GPLR-37 (the
mmap-registered default is the probed mode, and its promoted flip is why every real
first prefill now pays this); T-GPLR-36 (the ledger found double-booked here); the
2026-09-21 anomaly sweep (currentstate item 5 is the input this task adjudicates).
Commit:

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
- **NEW OPEN DEFECT - recorded, deliberately NOT fixed (the band is pre-registered and is not the
  implementer's to retune).** The driver states its band on the pass-1 WALL
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

#### ESCALATION - decision-ready for the Planner (2026-09-22, host-only, no GPU, no arm re-run): which quantity does the pre-registered `>= 70%` band actually test?

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

Status: ready (2026-09-22) - owner calls DECIDED below; A1 landed and A2 answered, both
host-only, 2026-09-22; no GPU spent on this block, no arm run. A3 is ONE 4k mutex hold;
its conditional 50k/200k follow-ups remain owner-invoked, one length at a time. A1 was first recorded as UNMET - Q could not be derived from any trace
the block would produce - and the telemetry it needed is now WRITTEN: a `proc_start` marker at
main entry plus `load_begin`/`load_end` around the model load, default-on, buffered until the
telemetry sink opens. Q's read is host-proven (`test-layer-streaming-tel-mark` in CTest, and
`tools/check-gplr56-q.py --selftest` 7/7), but no arm has run, so the marker shape in a REAL
trace is confirmed only by the arm named at the end of the A1 result - and that arm is currently
BLOCKED on the card (foreign pid 22436 held it across nine samples on 2026-09-22; nothing was
started, no mutex taken), while the t_us rebase it would exercise has already been checked
against every existing gate host-only (see the A1 real-trace section). A2 answered owner call 2
and CORRECTED the draft's premise: nothing can be registered before the model load returns, so
"earlier" is worth a measured 66.4 ms and the real lever is pacing the registration during the
sweep. The ESCALATION section's band question is NOT a dependency here: no arm cites that
band, and each arm names its own quantity instead.

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
Commit:

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
EXPLICITLY NOT INHERITED: A4's `>= 70%` band on pass-1 wall (unreachable at every shape -
ceiling 56.2%/10.5%/0.86% at 4k/50k/200k, see the ESCALATION section). Each arm writes
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
