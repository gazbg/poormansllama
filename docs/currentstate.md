# Current State

## 2026-09-23: T-GPLR-53 BAND-QUANTITY ESCALATION SETTLED (owner directive, OPTION A) AND THE REPORTING FIX LANDED - host-only, no GPU, no mutex, no arm re-run

- **Decision (A): the pre-registered `>= 70%` band's QUANTITY is the per-chunk embed EXCESS removal** against the
  same-session off control - `(ctrl_excess - arm_excess) / ctrl_excess` - NOT the pass-1 wall. The wall cannot
  carry a band: pass 1 = pass 2 + the one-off, so its ceiling is `100*(1 - pass2/pass1)` - 56.2% at 4k, 10.5% at
  50k, 0.86% at 200k - unreachable at every shape by construction. Rejected and recorded: B (a shape-relative wall
  band = A with an indirection, because the ceiling is itself derived from the excess) and C (drop the band, prose
  per run - least automatable). The wall is retained as the RELOCATION instrument (pre-pay + pass-1 wall read as a
  SUM), which is the only role its numbers support.
- **What changed.** `tools/run-gplr53-a4-arm.ps1`: the header's pre-registered reading names the excess; the
  reporting block prints per arm `excess removal {e}% (BAND) | pass-1 wall removal {w}% (relocation instrument)`,
  then the `WALL CEILING` line ONCE before any branch reads a wall number, then an excess-based
  `FALSIFIER FIRED/SURVIVED` with an explicit `wall band NOT EVALUABLE ... ceiling {c}% < 70%` note when the
  ceiling cannot express the band; the dry-run summary line was corrected the same way (it still promised
  "pass-1 wall deltas"). Plan: the A4 contract bullet states the settlement, the 2026-09-22 defect bullet is marked
  RECORDED/FIXED with the fix addendum, the ESCALATION heading records the settlement and the section appends the
  decision, and T-GPLR-56's "EXPLICITLY NOT INHERITED" cross-reference now describes the settled band.
- **What did NOT change:** the 70% threshold value, the two branches' meaning, every arm gate, and every recorded
  verdict. No past run was re-scored, and the A3/A4 conclusions and the 12:44Z/13:10Z withdrawal stand unchanged -
  they rest on the excess table, never on the wall line it printed.
- **Validation (host-only, no GPU, no mutex, no model):** PowerShell parse OK; `-ValidateOnly` dry run exit 0; the
  LANDED reporting block was extracted verbatim from the driver and replayed against the 13 recorded 13:10Z arm
  verdicts - it reproduces the record's own numbers (pw1 99.3%/50.3%, pw2 99.9%/50.6%, pw3 99.9%/50.8%, delay
  93.0%/47.2%, wrongtouch 86.2%, ceiling 56.2% with the excess at 50.7% of the control wall) and now prints
  `FALSIFIER FIRED ... removes 93.0% of the EXCESS` where the 13:10Z run printed `FALSIFIER SURVIVED ... 47.2%`;
  `resplit-gplr-plan.py --verify` PASS; `check-doc-sizes.py` PASS; `run-tests.ps1 -Suite host` 42/42 PASS
  (revision digest 971B2356EABA4322); `git diff --check` clean.
- **Residual/limits:** the settlement changes what an A4-family arm's PASS means (the band's unit); it moves no
  measured number and predicts nothing new. The mechanism question stays exactly where the A4 controls left it -
  the dose-response arm named in the block is still the arm that could move it. Uncommitted: the two edited files
  on `research/layer-residency`.

## 2026-09-23: PREDICTOR SCREENING PORTED FROM THE COLIBRI FUNNEL - coact/markov-1 beat the history baseline at every budget on every substantive stream, and coupling-copula leads unique coverage and high-budget recall (host-only, no GPU, no engine change; feasibility result, NOT a wire-in verdict). The T-GPLR-15 close left the predictive hypothesis explicitly open (ceiling 0.9385 vs retention 0.3088 at K=1); the colibri/deepseek-v4-baseline predictor-funnel (docs/findings/predictor-funnel-findings.md, DeepSeek-V4) supplied the candidate families and the replay methodology, now ported to our rowset_fetch schema as `tools/predictor-screen-gplr.py` (selftest for online faithfulness included, PASS; a predictor for (t,L) sees only tokens <t and layers <L, matching what the live engine could know). Corpus: 74 traces collapse to 7 distinct demand streams; screened one per stream (the two 256-token vehicles echo/honest, 50k-context, QA-depth; the two gplr57 rungs carry 3 TG tokens each and are reported but unconcludable). **Anchors:** history@B=8 reproduces the recorded (b) K=1 to ~1e-3 on BOTH vehicles (echo 0.3076 vs 0.3088; honest 0.2458 vs 0.2468) - the port is validated against the repo's own recordings; other shapes print anchor N/A (no recorded K=1; the first run mis-flagged 50k and the anchor was scoped to the 10240-row vehicle). **Results (recall@B / precision-per-byte / extra-hits-vs-history):** echo stream coact 42.6/43.0/16573 at B=8 vs history 30.8/30.9/0, markov-1 39.5, coupling-copula 31.9 but highest unique coverage and 77.2 recall at B=32; honest stream coact 28.1/28.4 at B=8 vs history 24.6 (same-precision +3.5pp), coupling 59.0 at B=32; 50k and QA streams repeat the ranking (coact/markov above history at every B, coupling wins B=32). **Reading:** colibri's H1 (same-layer transition structure beats history) and H2 (cross-layer coupling adds coverage, favors long context) both reproduce on Qwen3.6-35B-A3B; projection (labeled arithmetic, NOT measurement): a predictor feeding a K=2-sized window plausibly reaches what K=4-6 retention delivers - the charter's VRAM objective has a candidate lever. **Limits:** n=1 prompt per stream, no fresh capture; the (b)-projection is not a measurement; colibri's own T-pf-06 end-to-end A/B shrank fusion gains to noise on larger samples and its slot-thrash incident (UPDATE101) showed predicted fills must yield to exact demand - any wire-in needs that invariant plus a pre-registered acceptance quantity. Evidence: `docs/findings/predictor-screen-gplr.md` (method, full tables, reproduce commands); tool `tools/predictor-screen-gplr.py`; deterministic re-run verified. **Same-day extension (confidence-bar frontier + fusion, all offline):** colibri's barsweep/fusion operators ported (`--mode bar`, `--mode fusion`; selftest extended and PASS). Bar sweep: a conservative bar (0.9) strictly DOMINATES history on both axes on both streams (markov-1@0.9 = 25.1/29.0 vs history 24.6/24.7 honest; 36.1/41.7 vs 30.8/30.9 echo), but loosening to bar 0.7 collapses recall ~2.7x for a precision jump (41-57pp) - colibri's H3-disconfirm reproduces, so the bar is risk control, not frontier-pushing. Fusion (tune/held 80/20 split, grid at B=8): fusion dominates every single candidate on BOTH axes on both streams (best held: 29.6/29.6 honest vote-fusion vs history 22.7/22.7; 48.8/48.8 tuned weighted vs 27.9/27.9 echo); the colibri portable weight vector lands within 0.1pp (honest) / 2.4pp (echo) of per-stream tuning, so no per-cell tuning is needed; and the NULL-SOURCE CONTROL exposes the pool-growth share - a history-clone alone lifts held history 22.7 -> 28.0, so the structure-attributable gain is +1.6pp (honest) / +6.3pp (echo), and any wire-in case must argue against the null row, not the history row. Three measured operating points now exist for a wire-in decision: unbarred coact (+3.5pp honest, equal ppb), bar-0.9 (+0.5/+4.3, risk-free), fusion (+6.9/+6.9 honest with the null caveat). The T-pf-06 lesson stands: an offline win is a hypothesis; the engine A/B remains the verdict instrument.

## 2026-09-22: THE A1-LADDER SESSIONS RAN AFTER ALL (both rungs PASS, both CERTIFY, both MEASURED) - the card was held for the whole bounded window this pass tried (pid 5420, mutex FREE) and then cleared at ~15:42Z, so the two rungs ran and A1 has per-part input at ladder lengths for the first time. The pass also closed the audit's marker-after-sink silent loss.

**Blocker, measured:** `nvidia-smi --query-compute-apps` showed `deepseek_v4.exe` at every probe today - pid 21208 -> 23744 -> 17352 -> 14016 (13:57, 14:00, 14:10, 14:13, 14:22, 14:26, 14:28, 15:16:33Z) and then pid **5420** for the whole window this pass (preflight 15:29:42Z; 8 samples 15:31:17Z-15:37:17Z, every one `5420 deepseek_v4.exe`, mutex free - a non-conforming holder). No arm was launched, `-AllowBusy` was not passed, and **no result is inferred for either rung**. A separate `llama-layer-streaming` arm ran in this shared checkout at 15:21-15:23Z (PASS, 133.7 s, batch-major `gplr07-qa-512` shape) - **not this pass's run**, and its verdict carries only the four pre-provenance field keys, so it was written by an older tools revision than the one on disk; recorded so nobody reads it as this session's evidence.

**The two rungs then ran** (one mutex hold each, the recorded shape and the recorded anchor prompt reused byte-for-byte, sha256 in the chain verdict): **512** in `runs/20260922T154620Z-gplr57-a1-512` (arm PASS 10 s, chain PASS exit 0) and **200k** in `runs/20260922T154633Z-gplr57-a1-200k` (arm PASS 147.7 s, chain PASS exit 0). Both certify - `check-arm-provenance.ps1` reports **CURRENT (exit 0)** on all four verdicts against the tree as it stands, digest `9399BF96289C63E1`, `llama.dll` `261AA683E10E97AA`, exe `D01874A4235542A8`, `sources_newer_than_binary=false`, `provenance_delta` clean - **no waiver was taken** (no waiver log exists at any level), `--check-telemetry` exits 0 on both traces, the wrapper fields read `gpu_pre_run=gpu_post_run=clean` with `mutex_acquired=true`, and `--require-parts` classifies both traces **MEASURED** with 120/120 phase-1 `layer_compute` events carrying the split. The recorded rungs they supersede measure `phase1=120, with-parts=0, compute_us_sum=0.0 ms` - the `0/120` reproduced exactly.

**A1's reading at these shapes (measured, no performance claim, 3 TG tokens per rung - a COLD window, not a steady state):** TG wall **212.7 ms (512) / 243.5 ms (200k)**, of which transport (the H2D transfer of the staged rows) is **131.8 / 132.6 ms (62.0% / 54.5%)**, ready-latency 129.4 / 130.1 ms, compute **67.1 / 95.1 ms (31.5% / 39.1%)** - part A (attention + router) 43.6 / 71.9 ms, part B (FFN on the staged expert rows) 23.4 / 23.2 ms. Per TG token: ready-latency **43.1 vs 43.4 ms**, part A **14.55 -> 23.96 ms**, part B **7.81 vs 7.74 ms**. **So the transport term is context-invariant (131.8 vs 132.6 ms across a 390x longer prefill - 120 fetches of the same 8 experts x 1.77 MB = 14.16 MB, 573 MB/token at ~13.0 GB/s) and the context-dependent growth is COMPUTE, entirely inside part A; the transport SHARE only falls because part A grows.** In the PP leg the same transfer is overlapped (transfer 4610 ms against a 3423 ms wall at 512; 379,687 ms against 139,464 ms at 200k), so the blocking is a TG-leg property (98% of its transfer time is wait). **Cannot distinguish:** cold-window versus steady state (no token-index trend exists at any ladder shape), a context effect on transport versus warmup, anything about a transport lever (no mmap/pinned/pre-warm arm exists at ladder shapes and the batch-major shapes their arms use cannot carry the split), and no attention cost model from the part-A difference (confounded with KV length and sample size).

**T-GPLR-55 DECIDED, host-only, no GPU: its transport-remainder premise is REFUTED.** Everyone reading "transport GB/s" in this repo has been reading four different quantities as one. `transfer_us` on a copy event is exactly `copy_complete.t_us - copy_submit.t_us` - a **latency**, so `bytes/transfer_us` is a copy's speed only when the copy is **serialized** (`ready_latency/transfer_us ~ 1`); when copies are queued it is queue-inclusive and understates the speed. Measured: (i) **copy link rate** (serialized) **13.03 GB/s** at 512 and **12.96** at 200k for the 13.65 MiB row fetches, and **14.34 GB/s** for ONE 347.2 MiB full-bank copy at ratio 0.999 (`q38ls-lm-boundary`) - one link class, ~13-14 GB/s; (ii) **queued copies** at ratio 0.004-0.62 read 9.6-9.8 GB/s (`direct`, 3880-19480 copies), **7.37** (`gplr10-mem-m1`), **7.58** (`gplr24-hd-ab/hd-pinned`) - **this is where the drafted 7.3-7.4 GB/s came from, and it is not a transport speed**; (iii) **mmap-registered first-touch** copies at ratio 1.000 but ~3.3 s per 436.95 MiB read 0.07-0.14 GB/s (the page-residency commit P0 already closed); (iv) **wall rates** 4.4-5.8 GB/s row-set and 3.2-3.3 GB/s whole-bank (`docs/gplr13-bottlenecks-hypotheses.md` sections 3.7/H5, now carrying superseding notes). **So there is no 7.4 GB/s ceiling with a recoverable remainder above it**, and the context-dependent TG growth is attention+router, not staging. **Withdrawn:** any use of 4.4-5.8 / 5.4-5.8 / 7.3-7.4 GB/s as a transport speed, and the A1-ladder addendum's 62% transport share as a path property (those rungs are the only serialized row-set arms on disk; the banked arms overlap, ratio 0.067-0.598). **Re-scoped, and it names its single arm:** the only thing left open is the **source** effect on row fetches (mmap-registered 13.03 vs direct 9.16 vs pinned 7.85 GB/s on identical 13.65 MiB fetches - confounded by overlap ratio 0.982/0.067/0.578, so none may be credited), settled by one row-set TG arm at ONE ladder shape in the rungs' serialized config, run once per `--copy-source` value. **That arm RAN the same day** (512, `-CopySource pinned` 15:57:38Z and `-CopySource direct` 15:57:57Z, one mutex hold each, both PASS + **CERTIFIED** + **MEASURED**, no waiver, card left free), **and the answer is that the source is NOT a bandwidth lever**: forcing the source moved `ready_latency/transfer_us` along with it - **0.982** mmap-registered / **0.597** pinned / **0.067** direct - and only the registered path registers the mapping (40 `mmap_register` + 1 init vs **0/0**), so the field selects a staging PATH (registered/synchronous vs unregistered/async) and `bytes/transfer_us` is a RATE **only** for mmap-registered: **13.03 GB/s** (per-fetch median 13.06, range 12.54-13.37), while pinned **8.14** and direct **8.91 GB/s** are queue-inclusive **LOWER BOUNDS** (direct's may be at or above 13.03 - overlapping copies double-count the same interval). The earlier figures reproduce under the new arms on different arms and days - 9.16 at ratio 0.067 -> **8.91 at 0.067**, 7.85 at 0.578 -> **8.14 at 0.597** - so the confound **IS** the mechanism and the earlier three-way source comparison is **withdrawn, not resolved**; shape held fixed throughout (120 fetches of 13.65 MiB, TG compute 67.1/68.2/68.0 ms). Settling a source comparison needs a serialization/prefetch-depth control applied to the pinned and direct paths - a different lever, and a Planner call. Named, not built: nothing refuses to call a queued copy set a "rate" - a guard reporting a rate only at `ready_latency/transfer_us >= 0.9` is the Planner's call. No latency or throughput win claimed.

**SUITE RESULTS ARE BOUND TO A REVISION (same day, host-only, no GPU) - the audit's last evidence gap.** An arm's evidence named the tree and the binary it ran; a suite-green claim named NOTHING, and the suite's own newest log (14:09:57Z) described a revision that no longer existed. `tools/run-tests.ps1` now dot-sources `tools/source-revision.ps1` and captures the engine tree plus the binaries the suite exercises (runner, `llama.dll`, `ggml*.dll`) at suite START and again at END, writes both captures and the delta to `build/llama-cuda/suite-runs/<UTC>-<suite>/revision.json` (`schema suite-revision/1`), names the digest in the result line, and routes every terminating exit through ONE `Complete-Suite` that **FLAGS (exit 10 - the arm gates' contract) instead of reporting green** when the sources or the binary moved under the run, or when the tree cannot be digested at all; the record carries `result` (green/flagged/failed), `tests_passed`, the effective `exit_code` and the `flags`, so the artifact and the process exit code agree. `-VerifyRevision <record>` runs NO suite: it re-derives the tree now and settles a saved record with the SAME `Compare-ArmProvenance` rule (0 CURRENT / 1 stale / 2 cannot-evaluate), carrying the run's own delta flags into the comparison. **Proofs (no GPU, no shared mutex, no arm re-run):** stable tree - `-Suite host` **exit 0**, `revision digest 9399BF96289C63E1 (nested head 23e5d1b6e617, 5 tracked / 7 untracked changed), llama.dll 261AA683E10E97AA - record build/llama-cuda/suite-runs/20260922T164140Z-host/revision.json`, record `green / tests_passed=true / exit 0 / flags 0`, delta clean, 6 module hashes; **a source change AFTER the run** - `-VerifyRevision` **STALE-SOURCES exit 1** naming `9399BF96289C63E1 -> CB614B0FBCA7FA8D` and `now changed: suite-revision-probe.probe`, then **CURRENT 0** once the probe is deleted (content digests: a reverted tree returns to its earlier value); **the tree moving DURING the run** - a probe created 14 s into a real `-Suite host` -> **SUITE RESULT FLAGGED, exit 10**, record `flagged / tests_passed=true / exit 10 / delta source_changed_during_run=true`, and verifying THAT record reads **SOURCES-CHANGED-DURING-RUN exit 1**; a missing record - **CANNOT-EVALUATE exit 2**; and the other finisher exercised too (`-Suite telemetry` exit 0 with its own record, so no branch can print an unbound green). **A defect found while proving it:** the first version wrote `result=green, exit_code=0` into the record of a run it then flagged at exit 10 - caught by reading the WRITTEN RECORD rather than the printed line, and fixed by computing the flag and the effective exit code BEFORE the record is built (the same pass fixed a mis-keyed changed-count print). **Trap:** a `*.tmp` probe is SILENTLY not digested - the nested `.gitignore` ignores `*.tmp` and the digest covers non-ignored untracked files by design; the first attempt reported no drift for exactly that reason. **Scope: suite-green claims made BEFORE this landing name no digest and are TIMESTAMP claims, not revision claims** - including `build/host-suite-writer-rule.log` and `build/host-suite-after-gate.log` (both qualified in place) and the audit's own 14:09Z note; none is retracted, but none can be verified against a tree now. **Limits:** the binding covers the ENGINE tree and the exercised binaries, not the whole repo, so a checker or doc edited outside `third_party/llama.cpp` does not move the digest; ignored files are excluded by design; the capture is a point-in-time reading, so an edit landing between the END capture and the record write is seen only by a later `-VerifyRevision`. Nothing changes what a suite RUNS. No latency or throughput win is claimed; the nested sources under concurrent edit by another thread were not touched (the only writes inside that tree were the two probe files, created and removed inside this pass).

**THE WRITER REFUSES TO LEAVE AN UNATTESTED PASS - the inventory's last structural hole is closed at the one place every verdict is created (same day, host-only, no GPU).** The 20 historical runners write verdicts no gate ever checked, and because the writer captures provenance automatically such a verdict could read PASS while naming nothing usable. The rule now lives in `Write-ArmVerdict`: the per-capture test was extracted to ONE owner, **`Get-AttestationFailure`**, used by BOTH the gate that refuses a run and the writer that records one, so the two can no longer disagree about what "can attest" means (the gate's inline copy is gone). If the status would be PASS and the capture fails, the verdict is written **`UNCERTIFIED` with `exit_code` 10 - the same contract the live gates use** - plus `uncertified=true`, `would_be_status='PASS'`, `attestation_reasons`, and the command's own code preserved as `command_exit`; the evidence stays on disk with its capture intact and the `.done` sentinel is written from the same status/code so a reader of either agrees, and status derivation now maps exit 10 to `UNCERTIFIED` everywhere instead of FAIL in one place and uncertified in another. **The escape is singular and preserved:** `-AllowUnattested` is the writer-level opt-out, and the wrapper passes it FROM the operator's existing `-AllowUnprovenanced`, so one CLI switch is honoured by both layers - otherwise a waived run would still be recorded uncertified and the documented escape would be half a lie. **A genuine capture failure still produces a verdict**, degraded and named, so the evidence is never dropped. **Proof, host-only, no GPU and no shared mutex:** `test-arm-verdict.ps1` **section 17** (nine cases) - an unattested run that would end PASS is recorded `UNCERTIFIED` at exit 10 with `would_be_status=PASS`/`command_exit=0`, the reasons named (`no binary identity was captured (binary_state=not-supplied)`), the sentinel reading `UNCERTIFIED 10 <ts> unattested`, an ATTESTABLE run still PASS with no uncertified marker (no collateral damage), and `-AllowUnattested` the explicit escape; the whole harness reads **ALL ARM-VERDICT CASES PASS** (re-run this pass, exit 0); `-Suite host` is **exit 0, 42/42 CTest, all 12 blocks ran, none skipped** (`build/host-suite-writer-rule.log`, 18:22:28Z - after the newest of the three tool edits, 18:21:10Z; that log PREDATES the suite-revision binding, so it is a TIMESTAMP claim, not a revision claim - see the SUITE entry above). **The rule caught a real condition while proving itself:** case (g) first failed CORRECTLY - passing the exe ALONE made the capture read "the binary PREDATES its sources", because the engine code lands in `llama.dll` (15:41:00Z) while the exe relinks on its own schedule, this repository's own documented gotcha detected by the rule rather than by a reader; the fixture now passes the real artifact set. **Deliberate harness updates, not deletions:** the two fixtures whose subjects are NOT attestation (section 1 status derivation, section 7 run state) now carry `-AllowUnattested` with the reason written beside them. **The enforcement claim, restated to exactly what is true now:** (i) no verdict on disk can read PASS unless its capture attests the tree and the binary it ran - writer-enforced on every path, including the 20; (ii) the GATE that refuses a run binds at the wrapper and at the four live call sites that manage the mutex themselves; (iii) the 20 still write verdicts without a gate, so their residue is now "recorded, attributable, but never gate-checked" - which supersedes the inventory's "would look certifiable". "Every arm is gate-checked" remains false and is not claimed. **No historical runner was rewritten**, nothing in the retention state, A4's band or T-GPLR-55's shapes was touched, and no latency or throughput win is claimed. Evidence: `tools/arm-verdict.ps1` (`Get-AttestationFailure`, the writer rule, status derivation), `tools/with-gpu-lock.ps1` (waiver forwarding), `tools/test-arm-verdict.ps1` section 17.

**UNGATED-PATH INVENTORY, and what the enforcement claim may now say exactly (same day, host-only).** 28 files reference `Write-ArmVerdict`; four are tooling, leaving **24 arm runners**, classified by the stricter test (does it CALL the gate, not merely use the wrapper - a runner can use the wrapper and still write its own ungated verdict). **GATED:** the wrapper (every arm through it), `run-gplr53-a3-arm`, `run-gplr53-a4-arm`, `run-gplr53-session`, the smoke, and the ladder (at both the wrapper and its own gate). **NAMED GAP, not rewritten: 20 historical runners** from closed tasks T-GPLR-19..52 - `run-gplr19-rowcache-ab`, `run-gplr36-remeasure-200k`, `run-gplr37-arm`, `run-gplr37-set`, `run-gplr39-arm`, `run-gplr44-ab-50k`, `run-gplr44-dump-50k`, `run-gplr45-arm`, `run-gplr46-rebuild-ab`, `run-gplr47-arm`, `run-gplr48-arm`, `run-gplr49-arm`, `run-gplr49b-echo-attrib`, `run-gplr50-200k-pair`, `run-gplr50-arm`, `run-gplr51-a2-pinned`, `run-gplr51-s2-arm`, `run-gplr51-tail-arm`, `run-gplr52-a2-arm`, `run-gplr52-a3-arm` - none referenced by the active plan or the suite, 18 of them taking the shared mutex themselves, so the wrapper cannot gate them; **they can still write a verdict no gate ever checked, and because the writer captures automatically such a verdict would LOOK certifiable**. Two (`run-gplr19-rowcache-ab`, `run-gplr44-dump-50k`) were found by the census rather than the hand-scan precisely because they use the wrapper while writing their own verdict. **The list is frozen and tested:** `test-arm-verdict.ps1` **section 16** requires every verdict-writing `tools/*.ps1` to call the gate, with the 20 names as a dated allow-list (plus three tooling exemptions), and its four cases prove (a) no ungated writer beyond the set, (b) the set is not stale, (c) a NEW ungated writer IS caught (hermetic fake `run-gplr99-new.ps1`), (d) gated/no-evidence runners are correctly ignored - so a 19th cannot join silently. **Scoped enforcement claim: binding at the wrapper and at the four live call sites; NOT binding for the 20 named above.** Anything stronger (e.g. "every arm is gated") would be false. **That structural fix LANDED the same day** (see the WRITER entry above): `Write-ArmVerdict` now refuses to leave an unattested PASS, so the 20's residue is "recorded, attributable, but never gate-checked" rather than "recorded while unable to attest", and **no historical runner was rewritten** to reach them. The record's list was verified equal to the live census (20 = 20) after writing it.

**Enforcement moved to the CHOKE POINT, and a missing checker now FAILS the suite (audit follow-up, host-only).** Capture was systematic while enforcement lived only in four callers plus the smoke: `with-gpu-lock.ps1` - the wrapper every arm actually runs under - recorded a capture and asserted nothing, so a driver that forgot its own gate shipped unattestable evidence SILENTLY. The wrapper now calls the same `Assert-ArmProvenance` (same rule, same escape: `-AllowUnprovenanced` added there and forwarded by the ladder and the smoke so caller and wrapper cannot disagree about one run), keeps the taxonomy - **125** busy, **124** timeout, the command's own code, and a new **10** = ran but cannot be attributed - and on refusal REWRITES the verdict at 10 so verdict and exit code AGREE, with the command's code preserved as `command_exit`. A measured defect in my own first version: with no `$commandRan` scope the gate rewrote **125 and 124 into 10**; the harness caught it (10 cases red) and the scope fixed it. All twelve suite blocks that used to print `WARNING ... skipped` on a missing checker now print FAILED and `exit 1`. **Proofs, no GPU and no shared mutex:** `test-arm-verdict.ps1` **section 15** (8 new cases, hermetic temp root, private mutex names) - the WRAPPER refuses exit 10 with the verdict agreeing, `command_exit` preserved, the refusal named, and the waiver still passing and RECORDED; a suite run with `check-doc-sizes.py` deliberately hidden -> **`[doc-size] FAILED` and suite exit 1** (checker restored, sha256 verified identical); the whole harness **ALL CASES PASS**; the normal path **`-Suite host` exit 0, 42/42, all 12 blocks ran, none skipped** (`build/host-suite-after-gate.log` - likewise a TIMESTAMP claim: it predates the suite-revision binding). A wrapper run whose command names **no binary** is now refused - every real arm names its exe, and the harness's contrived stubs carry the recorded waiver. **Residual CLOSED the same day, at the writer:** the A3/A4 drivers take the mutex themselves and enforce via their own gate calls, so a FUTURE self-mutexed driver would bypass both - that door is now shut one layer down, because `Write-ArmVerdict` itself refuses to leave an unattested PASS (recorded `UNCERTIFIED` at exit 10, evidence kept and named; see the WRITER entry above). Trap recorded: splatting a hashtable onto a NATIVE command (`& powershell -File x.ps1 @a`) expands every entry as `-Key value`, so a switch becomes the literal argument `True` - section 15 uses an argument array instead.

- **Launcher landed:** `tools/run-gplr57-a1-ladder.ps1` - one rung per invocation (ONE mutex hold), **plus `-CopySource <mmap-registered|pinned|direct>` for the T-GPLR-55 source comparison (added 2026-09-22, host-validated: parse OK, an invalid value refused with nothing launched, `-PreflightOnly` exits 0 having launched nothing; default EMPTY reproduces the recorded rung exactly, so every existing tag, shape and evidence is untouched, and a source run gets a tag suffix and its own per-run directory)**, the rung's RECORDED shape and its recorded anchor prompt reused byte-for-byte (sha256 in the chain verdict), a per-run directory (`build/llama-cuda/runs/<utc>-gplr57-a1-<rung>/`) so a re-run cannot overwrite earlier evidence, `Assert-ArmProvenance` gating its own arm (exit 10; `-AllowUnprovenanced` the only escape and it writes a waiver line), and PASS withheld unless BOTH `--check-telemetry` and `check-tg-part-timings.py --require-parts` exit 0. Host proof: parse OK; against the busy card it **refuses in under a second with exit 125** naming the pid; with the probe stubbed clean it exits 0 having launched nothing. Recorded rung walls (the 2026-09-13 ladder driver): **512 = 12 s, 200k = 2.6 min**.
- **Correctness gap fixed, telemetry-only:** a marker taken AFTER the sink opened was pushed into `g_tel_early`, and the only flush is the one at sink-open, so nothing ever reached it again - no error, no flag, no line. The three close sites (teardown, run_end, refusal) now call `llama_ls_tel_flush_early(tel_out)` immediately BEFORE their `fclose`, which is the whole fix; a late line also carries **`late:1`** (otherwise it is indistinguishable from a second `load_end`) while every other field keeps its documented shape, and `seq` continues across flushes so call order stays provable.
- **A design error of mine, caught by the shared test:** my first fix stashed the `FILE*` and wrote late markers live; the test's own first case (flush a sink, `fclose` it, then mark) would then have written to a CLOSED FILE - undefined behaviour. Replaced by one buffer, one emission path, no stashed handle.
- **Validation, host-only:** `test-layer-streaming-tel-mark` extended with the post-flush case - **ALL CHECKS PASSED**, `ctest -C Release -R layer-streaming-tel-mark` PASS - and the four pre-existing cases (null-flush drop, the three-marker shape/order, the empty/NULL-name refusal) still pass unchanged, which is what makes the addition additive rather than a redefinition.
- **The rebuild was caught by the mechanism:** `llama.dll` 46628B7CDA8FAFA1 -> **261AA683E10E97AA** (exe unchanged `D01874A4235542A8`), and `check-arm-provenance.ps1` on `gpu-smoke-arm.verdict.json` now reports **STALE-SOURCES (exit 1)** naming "module llama.dll was REBUILT since this arm". The one certified arm on disk is therefore **no longer quotable as current** - the defect class this mechanism exists for, catching this pass's own change.
- **Limits:** a late marker is written by the next flush (i.e. at sink close), so its LINE POSITION is not chronological while its `t_us` is - emit-at-occurrence is a design question this fix does not take; with no sink at all there is no file to write to, so those markers remain buffer-only; and the rungs have since run (above), so **A1 now has per-part input at two ladder lengths - but still only 3 TG tokens per rung, so it is a cold window and not a steady-state rate**. No latency or throughput claim. Evidence: `tools/run-gplr57-a1-ladder.ps1`, `third_party/llama.cpp/{src,tests}/...tel*`, verdicts under `build/llama-cuda/runs/`, plan blocks `T-GPLR-57`.

## 2026-09-22: THE PER-PART TG TIMINGS WERE ALREADY LANDED (F3/H8, 2026-09-14) - WHAT WAS MISSING IS THAT A TIMING NEVER TAKEN COULD NOT BE TOLD FROM A REAL ZERO. The guard for that distinction is landed host-only, and the recorded traces now support one transport attribution and nothing at the ladder shapes.

The P3 prerequisite read as "the per-part TG timings read zero and must be populated". Measured, that is half right: the emitter has been in the tree since **2026-09-14** (F3/H8, `docs/gplr13-bottlenecks-hypotheses.md` section 3.6) - the row-set TG executor times each graph part around the staging (`third_party/llama.cpp/src/llama-layer-streaming.cpp:3327-3406`) and `tel_layer_compute` (line 5316) appends `attn_compute_us`/`attn_setup_us` (part A = attention + residual + post-norm + router) and `ffn_compute_us`/`ffn_setup_us` (part B = FFN + shared experts on the staged rows). What was NOT fixed is the reason the old reading was dangerous: the part fields are simply ABSENT when a path does not measure them, so a consumer doing `e.get("attn_compute_us", 0.0)` reads a never-taken timing as a zero-cost part - the same class as the decay rule that refused its own input.

- **The guard, host-only, no engine change:** `tools/check-tg-part-timings.py` derives the path from the trace's own signals (a trace carrying `rowset_fetch` ran the two-part row-set executor) and classifies every `(trace, phase)` group as **MEASURED** / **NOT-TAKEN** (fields absent: never measured - never a zero) / **PARTS-ALL-ZERO** (claims a split, reports nothing: the defect pattern) / **MIXED**. Exit 0 = classified, 1 = a PARTS-ALL-ZERO defect (checked before the requirement), **2 = a required group is not measurable - NOT A PASS**. No engine edit was made on purpose: the batch-major TG path runs ONE fused layer graph, so it has no attn/FFN boundary to time at all, and a marker field there would break the byte-identity the F3 amendment preserved on every non-row-set path.
- **Census over every trace on disk** (`build/tg-parts-census.log`, 444 traces carrying `layer_compute`): **31 MEASURED, 353 NOT-TAKEN, 0 defects**. The 95 row-set-path traces split **31 measured / 64 not-taken**, and every one of the 64 is pre-fix - including all four `gplr13-ladder` rungs (`50/512/4k/200k`, mtime 2026-09-13T13:38), which IS the recorded `0/120`. First producer: `gplr21-f3-parts/rowset-pp64-tg3-parts.jsonl` (120/120, 2026-09-14T17:30). **The ladder rungs have never been re-run with the split.**
- **Validation:** selftest **10/10** on hermetic fixtures - the fixture pair that matters is the SAME 120 all-zero phase-1 events reading NOT-TAKEN without part fields and PARTS-ALL-ZERO with them, asserted to differ; a MEASURED group with one legitimately zero part stays MEASURED (a real zero is not a defect); a batch-major trace reads NOT-TAKEN at exit 0 and refuses under `--require-parts`. On the real tree: `gplr13-ladder/{50,200k}` REFUSED exit 2, `gplr21-f3-parts/*` + `gpu-smoke-ctxshell` MEASURED exit 0, `gplr53-a4-pw0`/`gplr56-p1off-4k` REFUSED exit 2. Registered `[tg-parts]` in `tools/run-tests.ps1`; `-Suite host` green (42/42 CTest plus the block).
- **A1 from the traces, measured, no performance claim - and QUALIFIED by the T-GPLR-55 decision above:** the per-token transport figures in that pair are queue-inclusive LATENCIES (`ready_latency/transfer_us` = 0.578 and 0.260), so the 42.1 -> 7.7 ms/token ready-wait comparison stands as a **same-mechanism comparison**, but none of those numbers is a copy speed. 27 distinct runs (4 of the 31 traces are duplicate copies sharing a `run_id`), four TG-stream shapes (3/8/96/256 tokens). Transport share ranges 55.5-89.4% across runs at the same shape, so **a shape is not a measurement**. The one same-session pair carrying the split (honest K=6, reactive A vs retained B, 256 TG tokens, 4 runs per group) reads per token: submit 30.0 -> 23.1 ms, **ready-wait 42.1 -> 7.7 ms**, transport 72.1 -> 30.8 ms, part A 8.4 -> 8.0 ms, part B 5.2 -> 4.2 ms, setup ~1.6 ms both. So ~41 ms/token is removed and ~34 of those come out of the ready-wait: the effect is transport, not compute - a re-attribution of arms already banked for a tok/s result, not a second reading.
- **Cannot conclude, named not implied:** nothing at any ladder rung (all four are NOT-TAKEN); nothing for the transport arms themselves - `gplr52-a2-{50k,200k}`, every `gplr53-a4-*` and `gplr56-p1off-4k` run batch-major TG, where no split exists on any binary; `gplr24-ha-4096-retention-ARCHIVE.jsonl` is an unexplained outlier (89.4% transfer, 18.76 s submit) that must not be pooled with its shape; the smoke's 8-token shape shows the OPPOSITE signature (ready 344.1 ms vs submit 6.4 ms) at n=1 - a lead for the first-touch work, not evidence.
- **GPU steps:** the card was held by a foreign compute process (`deepseek_v4.exe` pid **14016**, mutex FREE) at every probe - 13:57, 14:00, 14:10, 14:13, 14:22, 14:26, 14:28 and 15:16:33Z. **A1-ladder** was blocked by THAT window and then ran anyway once the card cleared at ~15:42Z (both rungs PASS / CERTIFIED / MEASURED, recorded above). **A2-shape** (a row-set retention/reactive pair at 200k) was NOT run and is not claimed.
- **`T-GPLR-55` is not in this repository.** The id appears only in sentences forbidding edits to its shapes (7 occurrences, plan + currentstate) - there is no block, so its A1 wording, arms and thresholds could not be read. None were invented, and A4's band, the retention state and its shapes are untouched. Block recorded as `T-GPLR-57` (P3 host half) in the live plan. **Updated the same day:** a `T-GPLR-55` block NOW exists - the host-only DECISION recorded at the end of the plan's T-GPLR-57 material, which refutes the transport-remainder premise and re-scopes what remains; its drafted ARMS and THRESHOLDS still do not exist in the tree and were not invented. Evidence: `tools/check-tg-part-timings.py`, `build/tg-parts-census.log`, plan `T-GPLR-57`.

## 2026-09-22: PROVENANCE IS NOW BINDING AT THE LAUNCHERS, AND ONE REAL ARM IS CERTIFIED (scope corrected the same day: this was TRUE of the four callers and the smoke, but the WRAPPER every arm runs under asserted nothing - the gate now lives there too, see the entry at the top of this file) (smoke PASS 15:06:23Z-15:06:45Z, one mutex hold, digest BC5E1BBE8D0B30A0 / llama.dll 46628B7CDA8FAFA1)

A capture that nothing requires is a suggestion: after the mechanism landed, an arm could still write
a verdict with no capture and pass silently. The gate now sits where this program's own arms launch.

- **`Assert-ArmProvenance`** (`tools/arm-verdict.ps1`, beside the verdict convention) requires, per
  named arm: a `provenance` capture, `source_state=git` with a 16-hex digest, `binary_state=hashed`,
  and no self-recorded anomaly (`sources_newer_than_binary` / `*_changed_during_run`). Freshness vs
  the current tree stays the CHECKER's verdict - printed as a `NOTE`, never a reason to fail a run
  whose evidence was sound when written. An empty selection REFUSES: a gate that shrugs is the silent
  pass this mechanism exists to remove.
- **Wired where the arms launch:** `run-gplr53-a3-arm.ps1` and `run-gplr53-a4-arm.ps1` gate their own
  arms at exit and set **exit 10** (distinct from an arm FAIL 1 and from 124/125) when an arm cannot
  attest its revision; `run-gplr53-session.ps1` gates each step's tags, passes `-AllowUnprovenanced`
  through to the drivers and fails the session with the same 10; `tools/smoke-gpu-lock-ctxshell.ps1`
  gates its arm BEFORE writing its chain verdict and carries `arm_source_head`,
  `arm_source_digest`, `arm_streaming_lib_sha`, `arm_binary_exe_sha` and `arm_provenance_complete`
  INTO that verdict, so the smoke's result is self-describing. A4's band and every arm field are
  untouched: the gate re-scores nothing.
- **The escape is explicit and recorded:** `-AllowUnprovenanced` continues the run and appends
  `WAIVED label=... arms=... note=...` to `<verdictDir>/provenance-waivers.log`. **No escape was
  taken** (the file does not exist).
- **Host proof of the refusal path** (`tools/test-arm-verdict.ps1` section 14, 16 cases, all PASS): a
  complete verdict passes and `Assert` returns true; a verdict with its capture STRIPPED is refused
  and named; a capture with no binary identity refused; an arm whose binary predates its sources
  refused; a tag with no verdict refused; an empty selection refused; the waiver returns true AND
  writes the log; and a **static** guard asserts all four call sites still contain the gate call and
  the exclusion plumbing, so the wiring cannot be deleted silently (labelled static - a driver cannot
  be run host-only). `-Suite host` exit 0 (`build/suite-gate.log`).
- **A mechanism correction this required, measured:** the staleness flag compared the newest changed
  file of ANY kind, so the other actor's scratch `.tmp-split/n.*.diff` (16:48:11, newer than the
  16:32 build) made every arm read `sources_newer_than_binary` and would have refused a smoke that
  was in fact fine. The DIGEST keeps whole-tree coverage; the MTIME relation now uses only
  compile-relevant paths (`.c/.cc/.cpp/.cxx/.cu/.cuh/.h/.hh/.hpp/.m/.mm/.s/.asm/.inc/.cmake/.def/.rc`,
  `CMakeLists.txt`, `Makefile`), recorded as `source_newest_input_path/_utc` beside the informational
  `source_newest_change_path/_utc`. That extension set is the mechanism's one judgement call and is
  documented where a Planner would look.
- **One real arm, on hardware, certified.** The card cleared at **15:06:18Z** (sample 14 of a bounded
  ~10-minute probe: `deepseek_v4.exe` pid **24104** held compute from 14:56:26Z with the mutex FREE -
  non-conforming - and released between samples 13 and 14). The smoke ran **15:06:23Z-15:06:45Z in one
  mutex hold**: chain verdict `gpu-smoke-ctxshell.verdict.json` **PASS** (exit 0, `arm exit 0 after
  19.2 s`, `check-telemetry: OK (1049 events, 4 slots)`, 8 rowset rounds, 320 fetch events, TG text 21
  bytes, probes clean), arm verdict `gpu-smoke-arm.verdict.json` **PASS**. The GATE printed `1 smoke
  arm verdict(s) checked - 1 complete, 0 refused`; the arm's capture is `origin=launch`, head
  **23e5d1b6e617**, digest **BC5E1BBE8D0B30A0** (5 tracked / 7 untracked),
  `llama-layer-streaming.exe D01874A4235542A8` / `llama.dll 46628B7CDA8FAFA1`, post-capture
  `origin=post-run`, both deltas false - and `tools/check-arm-provenance.ps1 -Dir build/llama-cuda -Tag
  gpu-smoke-arm` returns **CURRENT, exit 0**. It is the first arm on disk whose evidence can be tied
  to a revision. Its rates (PP 133.957 tok/s, TG 15.280 tok/s at 512/8) are the smoke's context only:
  a connectivity check, never a measurement. Card clean and lock free after; no waiver file.
- **Not done, named:** no historical verdict was retro-certified (forbidden here, and the 101 remain
  uncertifiable); the P1 arm set was not started; the gate is not wired into any *analysis* path, so
  the record's own quoting rule is still a process convention. Cost: `docs/currentstate.md` is now
  ~262 KB (WARN band) - the split is still due before more evidence lands.

## 2026-09-22: ARM PROVENANCE LANDED - every verdict now names the SOURCE TREE and the BINARY it describes, and a checker refuses to certify an arm whose sources moved (host-only; no GPU, no mutex, no arm run)

Nothing bound an arm's evidence to the revision its binary was built from: this thread's smoke
passed at 16:28 against the 16:07 build, a concurrent edit landed on
`third_party/llama.cpp/src/llama-layer-streaming.{h,cpp}` at **16:31:13** and the tree was
rebuilt at **16:31:58 / 16:32:01**, and the record's "same-binary with the reverted tree" claim
became false without anyone touching it. The exe hash was the wrong anchor twice over: the
streaming code lands in `llama.dll`, and a relink churns the exe hash while the engine is
unchanged. The mirror failure is in this tree too: `tools/layer-streaming/main.cpp` was modified
at **16:17**, after the 16:07 build, so an arm run then would have executed a binary that could
not contain it, silently.

- **Mechanism.** `tools/source-revision.ps1` (new, functions only) digests the engine source
  tree - nested HEAD + the full content diff of tracked changes + the CONTENT of untracked files
  under it (content, not mtimes: reverting an edit returns the earlier digest) - and identifies
  the binary FROM THE COMMAND (its exe plus the `llama.dll`/`ggml*.dll` beside it) as 16-hex
  SHA-256 prefixes, the form the arm drivers already record.
- **Every writer path, not just the wrapper.** `Write-ArmVerdict` (`tools/arm-verdict.ps1`) now
  always attaches `fields.provenance`. That placement is the finding, not a convenience: the arm
  drivers write their OWN verdicts for their OWN tags (measured: `gplr53-a4-pin.verdict.json`
  carries none of the wrapper's fields), and those are the arm verdicts the record cites. Their
  binary is discovered from the run's own `.started` marker, which already records the arm's
  command - so no driver needed a change. The wrapper additionally captures BEFORE the command
  (`origin=launch`) and AFTER it (`origin=post-run`), recording `provenance_post` and
  `provenance_delta` (`source_changed_during_run` / `binary_changed_during_run`): a rebuild DURING
  a long arm is the same defect class as one after it.
- **The flag that would have caught it:** `sources_newer_than_binary` + `sources_newer_detail`,
  newest changed source vs the newest artifact of the build. First cut compared against the OLDEST
  artifact and flagged almost everything (ggml-cuda.dll has not been rebuilt since 2026-09-17);
  rejected on measurement.
- **New checker:** `tools/check-arm-provenance.ps1` re-derives the digest from the tree now,
  re-hashes the recorded modules by name, and applies `Compare-ArmProvenance` (one owner, beside
  the digest it uses). Exit **0** CURRENT; **1** STALE-SOURCES / SOURCES-NEWER-THAN-BINARY /
  SOURCES-CHANGED-DURING-RUN / BINARY-CHANGED-DURING-RUN / STALE-BINARY; **2** NO-PROVENANCE /
  NO-VERDICT / CANNOT-EVALUATE / BINARY-UNKNOWN. Absence is never a pass. A raw revision object
  passed where a flat record is expected is refused loudly (first cut mis-read it as "cannot
  digest now", a silent wrong answer).
- **Proof, two host-only layers.** `-Selftest` **22/22** over throwaway git repos and throwaway
  binaries (never the real tree): the measured failure reproduced - build, save a verdict, edit a
  source -> **STALE-SOURCES exit 1**; revert -> CURRENT; a NEW untracked source -> STALE; rebuild
  the engine dll -> STALE-BINARY; source edited after the build -> SOURCES-NEWER-THAN-BINARY; a
  mid-run change -> SOURCES-CHANGED-DURING-RUN; a pre-provenance verdict -> **NO-PROVENANCE exit
  2, "NOT A PASS"**; a non-repo tree -> CANNOT-EVALUATE. `tools/test-arm-verdict.ps1` section 13
  adds the real-tree half: the command resolves to its exe + 5 modules, a verdict through the
  shared writer carries the real digest (equal to an independent capture) and `llama.dll
  46628B7CDA8FAFA1`, the checker CLI certifies it (exit 0), a wrapped stub run records
  `origin=launch` + a post capture + a delta agreeing with both captures, and a command naming no
  binary records `binary_state=not-supplied` rather than a guess. Registered as `[arm-provenance]`
  in the host suite; `-Suite host` exit 0 (`build/suite-provenance.log`).
- **It caught a real drift while this entry was being written.** The nested tree's digest moved
  three times in one hour - `CE58F2F7DCB62B67` -> `C33A0E40BC23F9BF` -> `BC5E1BBE8D0B30A0` - as
  another actor's scratch files appeared under a new `.tmp-split/` (16:48:11). Re-running the rule
  against the pre-drift digest returned **`STALE-SOURCES` exit 1** and NAMED them
  (`now changed: .tmp-split/n.ccm.diff, ...`), because a bounded path manifest
  (`source_changed_paths`) rides beside the digest. Recorded as the mechanism's first live catch,
  and as its sensitivity limit: whole-tree coverage means scratch reads as drift too - the intended
  error direction, with the path list making the explanation one line.
- **The boundary, measured:** of the **101** saved verdicts in `build/llama-cuda/verdicts/`, **0
  certify (exit 0), 101 cannot-certify (exit 2), 0 stale**. None was written after the mechanism
  landed, so nothing in the accepted evidence is retro-certifiable - a re-run is what produces a
  certifiable verdict. Those verdicts are not wrong; they are UNCERTIFIABLE, and the entry below
  is corrected to say so.
- **Two of my own first-cut defects, found by the checks:** the module assumed a caller's
  `$ErrorActionPreference`, and PowerShell 5.1 turns a native command's stderr into a TERMINATING
  error under `Stop`, so `git rev-parse` outside a repo aborted the capture with `state=error`
  instead of reporting `not-a-repo`; and a provenance field arrives as a dictionary in memory but
  a PSCustomObject after a verdict round-trips through JSON, so enumeration has to accept both or
  `Count`/`Keys` come back as if they were module names.
- **Scope, named:** the digest covers the nested engine tree (`third_party/llama.cpp`), which is
  what produces the binary; the outer repo's tooling is not hashed. A verdict that names no binary
  is uncertifiable by design. No performance claim, and no GPU time was spent (the card was held
  by `deepseek_v4.exe` throughout; nothing here needed it). **Cost:** this file is now 254,486 B
  (WARN band, 53% of FAIL) - the split flagged in the entry below is due before more evidence
  lands.

## 2026-09-22: TWO OWNER-AUTHORIZED ONE-LINERS APPLIED - stale one-slot CTest tests fixed (--pp-async-feed 0), recon emitter ubatch label landed and proven on a fresh arm

- TEST #53/#54 FIX: `qwen38-layer-boundary` and `layer-slot-lifecycle` were refusing
  since the T-GPLR-25 async-feed default flip (one-slot tests never passed
  `--pp-async-feed 0`); both CTest definitions now carry it. Validated: 2/2 PASS
  under the mutex with `Q38LS_MODEL_PATH` set (11.3 s / 9.7 s). The whole ctest layer
  family is now 74/74 with the env var restored (72/74 before this fix).
- RECON LABEL FIX: `ls_recon_ev.ubatch` captured at enqueue; emitter prints the entry's
  own value instead of the flush-time global. Engine-side only, recon-gated path only
  (a non-recon run's trace is byte-identical - the emitter is the only reader). Validated
  on a fresh 4k anchor `--pp-repeat 2 --pp-recon 1` arm (77 samples, 77/77 join at their
  own key, zero stale-label outliers; run PASS, gpu probes clean). The -14/-15 label
  class seen on gplr50/gplr52 is gone at the source; old traces keep their labels as
  recorded history (the keyed checker's classifications are unchanged and remain valid).
- Guards: host suite SUCCESS, nested `git diff --check` clean, doc-size PASS,
  resplit --verify PASS. Commit: pending (no commit authorized).

## 2026-09-22: T-GPLR-52's forwarded recon off-by-one join concern - DECIDED CLOSED, host-only from the 126 recorded traces

The last unowned item P5 carried. Decided from the traces on disk, not from the record that claimed
the fix: 126 traces in `build/llama-cuda/` carry `recon_attr` samples, 97 recorded before the A2
stamp fix and 28 after it.

- **Pre-fix era (97 traces, 7,379 samples): 0 exact `(layer, generation)` matches against
  `copy_submit`, 7,379 (100%) match after a +1 generation shift, 0 ordering violations.** So the
  stamp was exactly one generation behind and nothing else - the signature of `recon_begin_bank`
  reading `slot.generation` before the increment. This also reproduces the record's counts
  (`gplr51-a3` 0/89, `gplr52-a2-50k` 0/125, `gplr52-a3-mmap` 0/85).
- **Post-fix era (28 traces, 2,281 samples): 2,281/2,281 exact, 0 ordering violations.** The re-keyed
  `(layer, generation)` join also kills the old phantom: under `(layer, ubatch)` the 6 samples per
  trace flushed across a pass boundary (layers 38/39) drifted by exactly the inter-pass stride,
  because `ubatch` is the LIVE `n_ubatches` at flush time, not the measured copy's chunk. Under the
  current key that class is **0 across all 126 traces**.
- **The gate agrees on real inputs:** `check-gplr53-embed-chunks.py gplr53-a3-50k --join` -> PASS
  (`CHK-join ok`); `gplr51-a3 --join` -> FAIL, "89 sample(s) carry a generation NO copy of that layer
  ever had - the pre-T-GPLR-53 stamp offset". Old traces FAIL rather than being silently re-keyed,
  which is correct: **no tool in `tools/` adds 1 to a recon generation** (checked), so a pre-fix
  trace stays honestly unbucketable.
- **Disposition: CLOSED, with one standing negative requirement** - a future pass must not add a +1
  shim to a checker to make pre-fix traces pass; the offset belongs in analysis prose if ever needed.
  **Own error, recorded:** this scan's first run reported 271 pre-fix samples unmatched after +1;
  that was a falsy-zero bug in the scan (`(generation or -99) + 1` sends generation 0 to -98), not a
  second stamp defect. Corrected: 0. Left uncorrected it would have entered the record as a phantom
  3.7% residue.

## 2026-09-22: T-GPLR-56 OWNER CALLS DECIDED - first-token PP latency in scope, bounded (A3 decides on one 4k hold); registration-timing authority = A2's narrowed form; block is READY, no arm run yet

The owner answered the two calls the draft block named as product-intent decisions
(documents only; the block's pre-registered gates are untouched):

- CALL 1 (scope): first-token PP latency IS in scope, BOUNDED - one 4k mutex hold
  with the pre-registered Q rules unchanged; the conditional 50k/200k holds stay
  owner-invoked per length. The charter tension (the ~1.9 s one-off is 50.7% of the
  4k pass-1 wall but 1.4% at 200k) is exactly why the ladder is not pre-authorized;
  a negative A3 closes the block charter-correct ("overlappable only at short
  shapes" is a decision-useful result).
- CALL 2 (timing authority): A2's narrowed form RATIFIED - paced issuance from the
  existing call site (p1-pace) plus the incremental-with-prebuild fallback
  (p1-overlap, only if p1-early shows contention) are the authorized mechanisms;
  tensor-layout span derivation is DEAD per A2's negative finding; registration
  DURING the model load is explicitly FORBIDDEN (unmeasured loader interaction).
  p1-early stays as the pre-measured negative control (66.4 ms, a designed FAIL of
  the 250 ms margin). Status: draft -> ready. No arm has run; the next step is
  implementing the pace mode behind the default-off flag (A3's implementation
  steps), host-validated, then the one 4k session.

## 2026-09-22: T-GPLR-16 CLOSED - the `--handoff-partial` MIN_BATCH=1 override is REVERTED out of the tree, with a host-only guard

The last already-authorized item from the 2026-09-12 option (a) decision is landed. The runner used to
force `GGML_OP_OFFLOAD_MIN_BATCH=1` under `--handoff-partial` (plus an env-var bypass for probes) so
CUDA's `MUL_MAT_ID` would keep taking host-resident expert banks; that route deadlocks the device
stream on the third TG token, so option (a) reverts it and accepts the known step-1 gold flip.

- **Reverted at one site**, `third_party/llama.cpp/tools/layer-streaming/main.cpp`: the
  `_putenv_s`/`setenv` force, the bypass static, the `gplr16:` stderr announce line and the block
  around them, replaced by a short comment recording the deadlock and the accepted mixed-arithmetic
  cost. `--handoff-partial` itself, its help text and all four compatibility refusals are intact -
  the revert removed the override, not the mode. Nothing else in the tree referenced either string.
- **Host-side proof, not a grep transcript.** `python tools/check-gplr16-revert.py` is now a tracked
  checker: 0 setters across 10 set-patterns, the bypass flag absent from source entirely, no
  announce line, and all 5 reads of the variable on the upstream ggml allowlist (cuda 5645, vulkan
  18704, sycl 6733, cann 2996, metal 883 - each the `? atoi(getenv(...)) : 32` default). A NEW read
  on this project's own path fails. `--selftest` **10/10**, with a negative control for every way
  the revert can be undone plus two refusals (mode file absent, empty tree). Registered in the host
  suite as `[t16-revert]`; `tools/run-tests.ps1 -Suite host` exit 0 (`build/host-suite-t16.log`).
- **The mode's own refusals still hold** on the reverted binary: `ctest -R refuse-partial` -> both
  `layer-streaming-refuse-partial-dense` and `-zero` Passed (0.19 s total, no CUDA init). Those two
  are outside the host suite's CTest Include filter, so they were run explicitly rather than assumed
  covered by the 42/42.
- **Compiles:** `cmake --build build/llama-cuda --config Release --target llama-layer-streaming`
  exit 0. **New binary identity** (`llama-layer-streaming.exe 38B39C06229BF15B`, `llama.dll
  2930A7D0B30F967E`,  `ggml-cuda.dll C04D744E9A4213F1`, 16:07Z) - recorded because a rebuild makes
  every earlier A/B control a different binary. **SUPERSEDED 16:32Z, and this is the finding
  the provenance entry above exists for:** a concurrent edit landed at 16:31:13 and the tree was
  rebuilt at 16:31:58/16:32:01, so this identity (and the smoke line below that leans on it)
  describes a revision that no longer exists. Nothing in the record noticed at the time -
  evidence binding is now mechanical.
- **The guard caught its own registration on first use:** the host-suite comment naming the bypass
  string made the real scan FAIL with 1 reference until the comment was paraphrased (`tools/` is in
  scope, so a mention there is a surface). Recorded as evidence the rule bites.- **Smoke regression RAN, PASS** (2026-09-22 14:28:28Z-14:28:47Z, ONE mutex hold, ~19 s): four
  bounded probe windows were refused first (13:57:15Z-14:00:48Z, 14:10:13Z-14:13:15Z,
  14:14:58Z-14:16:19Z, 14:22:54Z-14:26:41Z; 22 samples, the lock held by `deepseek_v4.exe` pids
  21208, 23744 then 17352 throughout) and the fifth cleared at **14:28:20Z** (`lock_free=True`, no
  compute app) - the fail-fast preflight, satisfied before the window was spent. Chain verdict
  `build/llama-cuda/gpu-smoke-ctxshell.verdict.json` **PASS** (exit 0, 17.9 s, `fetch_events 320`,
  `tg_text_bytes 21`, `checker_exit 0`); arm verdict `gpu-smoke-arm.verdict.json` **PASS** (13.4 s,
  probes clean, `mutex_acquired true`); log `gpu-smoke-ctxshell.log`: arm exit 0 after 16.0 s,
  `check-telemetry: OK (1049 events, 4 slots)`, `CHAIN OK`; the arm command uses
  **no `--handoff-partial`**, so this is the default path the revert had to leave alone. Binary:
  `llama-layer-streaming.exe 38B39C06229BF15B` / `llama.dll 2930A7D0B30F967E` /
  `ggml-cuda.dll C04D744E9A4213F1` - unchanged since the revert's rebuild, so the smoke is
  same-binary with the reverted tree **as of 16:07; CORRECTED in the provenance entry above:**
  the engine sources were edited at 16:31:13 and the tree rebuilt at 16:31:58/16:32:01, so this
  holds for the 16:07 revision only, NOT for the current tree (now `llama-layer-streaming.exe
  D01874A4235542A8` / `llama.dll 46628B7CDA8FAFA1`, ggml-cuda unchanged), and no saved verdict
  carries provenance, so it cannot be re-certified without a re-run. Card clean and lock free after.
  It is a connectivity check, not a measurement: the PP/TG rates in its summary are context only, and
  `cuda_path_empty: false` means the smoke's "without CUDA_PATH" property was not what this run
  tested. Incidental: `gpu-smoke-arm.verdict.json` is the **first real arm carrying the wrapper's new
  provenance fields** (`tag_source: explicit`, `gpu_probe_source: nvidia-smi`, probe statuses clean),
  so the wrapper work is exercised in production, not only by stubs.
- **Cost this pass carried, named:** `check-doc-sizes.py` now warns on TWO files - the plan archive
  (345.0 KiB, pre-existing) and **`docs/currentstate.md` at 246.6 KiB**, which these entries pushed
  past the 250,000 B WARN band. Both are inside the FAIL band, but the next pass should split
  currentstate per the convention in `docs/AGENT_ENVIRONMENT.md` before adding more.
- **Scope, named:** the other T-GPLR-16-era probe guards (`GGML_CUDA_NO_GRAPHS_PROBE` and friends)
  and `tools/gplr16-dump.cpp` are untouched - this closure is the forced variable and its bypass.
  The live plan carries no T-GPLR-16 line (the id only ever lived in the dated currentstate files),
  so the closure is recorded here and as a closed block appended to the live plan; the 2026-09-12
  disposition now carries a pointer that the revert landed. No latency or throughput win is
  claimed, and the option (a) decision is not re-opened.

## 2026-09-22: ESCALATION - the >= 70% band is stated on a quantity that cannot reach it (host-only, no GPU, no arm re-run)

Decision-ready for the Planner; five pages of it in the T-GPLR-53 block. The A4 driver states its band on
pass-1 WALL (`:11-15`), measures the falsifier against the EXCESS (`:25-27`) and tests the WALL (`:246-264`),
so the 13:10Z run printed `FALSIFIER SURVIVED ... 47.2%` while that arm's excess removal is 93.0%.

- **Measured - the wall band is unreachable at every shape by construction** (generous formulation: pass-2 wall
  is the floor, since pass 1 is the same work plus the one-off): **4k 56.2%** (13:10Z; 57.2% in the archived
  12:44Z run), **50k 10.5%**, **200k 0.86%**; on the excess-share formulation 50.7/51.9/9.9/1.4%. The ceiling
  FALLS with prompt length because the one-off is shape-invariant (~1.8-2.0 s) while the wall grows - so the
  band is worst exactly at the long-context shape the charter prioritises. The driver's own printed
  `pw1/pw2/pw3 = 50.3/50.6/50.8%` already shows the walk, the arm branch (i) is defined by, failing its own band.
- **What each branch can test:** (i) and (ii) are testable on the excess only; on the wall both are unreachable,
  so the `neither -> escalate` branch is hit by EVERY arm including one clearing 99.3% of the excess - the wall
  band carries no information. The wall's legitimate role is the RELOCATION sum (pre-pay + pass-1 wall,
  3.70-3.76 s, spread 0.06 s), never a removal fraction.
- **Nothing is reversed:** A3's CHUNK0_SPIKE rests on the per-chunk excess (chunk-0 share 0.9997/0.9987,
  2001.91/1871.71 ms vs a 0.63-0.64 ms warm median) and A4's H1/H3 reading on the excess table (walk 99.3/99.3%,
  delay 92.7/93.0%, wrongtouch 91.9/86.2%, pin 99.9%, control repeat ~0). Only the impression that the printed
  `FALSIFIER` line IS the evidence is retracted; the excess table is.
- **Zero-cost reporting fix - specified, NOT applied:** print wall and excess removal on one line per arm, print
  the ceiling once, and print `NOT EVALUABLE on the wall (ceiling {c}% < 70%)` instead of `SURVIVED` when the
  ceiling is below the band. Threshold, branches and gates byte-identical; past runs re-scored = no.
- **The decision is the Planner's/owner's:** (A) bind the band to the embed excess - recommended, the only
  quantity that can express the branches at every shape; (B) keep the wall but make the band shape-relative to
  its own ceiling; (C) drop the band, report both quantities and the ceiling. No GPU, no mutex, no arm re-run,
  no threshold retuning; retention state and the T-GPLR-55 shapes untouched.
- **Successor drafted (2026-09-22, plan only, no code):** `### T-GPLR-56 (draft)` - take the registered-span
  commit off the first prefill's critical path (start it earlier / overlap it / shard it / shrink the span).
  It pre-registers `Q = pass1_end_ts - proc_start_ts` with a >250 ms margin and a both-conditions PASS rule, so
  a relocation cannot satisfy it, and it explicitly does NOT inherit the band above; two owner calls gate it
  (is first-token PP latency in scope, and is earlier registration acceptable given the spans must be known).
- **T-GPLR-56 A1 real-trace confirmation RAN, PASS - Q is a measured baseline and the rebase is clean on a real
  rebased trace (2026-09-22, one mutex hold, 4k control, `--pp-recon 1`).** Card cleared at 13:43:26Z with the
  mutex free, so the single control arm ran under `tools/with-gpu-lock.ps1`: wrapper exit 0, `PASS`,
  `gpu_pre=clean gpu_post=clean`, mutex released, arm duration 19.1 s. Trace
  `build/llama-cuda/gplr56-p1off-4k.jsonl` carries **1,930 events** vs the 13:10Z control's 1,927 - exactly the
  three markers, and they are the file's **lines 1-3 in call order** (`proc_start` seq 0 / `pre_run` 1 / NO
  `run_id` / t_us 3, `load_begin` t_us 197,673 with the model path, `load_end` t_us 5,471,465 `ok`), ahead of
  `pkg_member` on line 4 and `run_start` now on **line 46**. **Q baseline = 9,370.7 ms** (pass-1 wall 3,762.9 ms),
  of which pre-load 197.7 ms and model load **5,273.8 ms**. Gates on the new base: `--check-telemetry` 0,
  `check-gplr53-embed-chunks.py --join` 0, `analyze-gplr53-decay.py` 0 with CHUNK0_SPIKE, pass-1 file-backed WS
  19.03 GiB; `check-gplr56-q.py --trace` exit 0 and with the older 13:10Z control **exit 2 refusal** as
  pre-registered. So the clock rebase (old base `t0_us` 2,974,883 -> this base's `proc_start` at t_us 3) breaks
  no existing gate in production, and P1 comparisons must be same-base. **Uncontrolled, not explained away:**
  the 5,273.8 ms load is not comparable with the 13:10Z run's implied shorter one - different bases and an
  uncontrolled Windows file cache (a run after ten minutes of a foreign process's GPU work cannot be assumed
  warm). **Superseded claim, corrected 2026-09-22:** an earlier entry here said `tools/with-gpu-lock.ps1`
  "prints `gpu_pre/gpu_post` but persists neither (nor a tag) into its verdict json". That is wrong,
  and the artifact says so: `gplr56-p1off.verdict.json` carries `fields = {gpu_pre_run: clean,
  gpu_post_run: clean, mutex_acquired: true, timeout_seconds: 7200}` and `arm` IS the tag - the
  wrapper has recorded that since `001bf65`, and `tools/test-arm-verdict.ps1` asserts
  `fields.mutex_acquired` on a wrapped run. The real gap underneath it is fixed below.
- **Wrapper verdict evidence landed (2026-09-22, host-only, no GPU, no CUDA, no arm re-run).** The
  measured defect: a probe that ANSWERED with nonempty unparseable output (a dropped `--format`
  flag, a driver banner) fell through `Get-GpuComputeApps` to an EMPTY list and was persisted by
  `Format-GpuBusy` as **`clean`** - a probe that never answered recorded as an idle card, in every
  wrapped arm's verdict. Before/after on one fixture: `clean` -> `probe-unavailable`. Landed
  additively: nonempty-unparseable now returns `$null` (UNVERIFIED); the wrapper's `fields` gains
  `tag` / `tag_source` (explicit vs derived) / `gpu_probe_source` (`nvidia-smi` vs `stub`) and
  `gpu_pre_status` / `gpu_post_status` = `clean|busy|unavailable|not-probed`; the pre-existing
  fields keep their names, values and meanings (the mutex-timeout and exit-125 absence is now
  readable as `not-probed`). Proof: `tools/test-arm-verdict.ps1` section 12 - **70 checks PASS,
  0 FAIL** (`build/tav-fast.log`), covering (a) the fields are persisted, (b) a consumer reading
  only the old fields is unaffected, (c) busy / unavailable / unparseable / never-taken are each
  distinguishable from clean (`not-probed` produced end-to-end via a private mutex held in another
  process and `-TimeoutSeconds 0`, exit 124), plus an audit of the in-repo consumers: every verdict
  read in `tools/*.ps1` is a pre-existing key, and field-bag comparisons are `-contains`-guarded.
  Both card states on disk: section 6 SKIP with exit 0
  while another project held the lock on three consecutive runs (`build/tav-clean-1..3.log`), and
  real lock + `clean` probe on a free card (`build/tav-run3.log` - produced by the revision before
  two harness-efficiency edits that do not touch section 6; a bounded re-probe 13:57:15Z-14:00:48Z
  found the lock held by `deepseek_v4.exe` pid 21208 in all 8 samples, so no fresh free-card run).
  Registered in the host suite as
  `[gpu-lock]`; `tools/run-tests.ps1 -Suite host` exit 0, 42/42 CTest (`build/host-suite-gpulock.log`).
  **Not re-run:** `build/gpu-gate-tests/run-gate-cases.ps1` (six gate branches) - it takes the REAL
  shared lock with a 7200 s default timeout and that lock is held by another project, so re-running
  it would block; its branch conditions are untouched by this change but have no fresh evidence
  today. No performance claim; A4's band and saved scoring, the retention state and T-GPLR-55's
  shapes untouched.
- **T-GPLR-56 A1 real-trace arm, FIRST ATTEMPT: BLOCKED (2026-09-22).** The 4k control that would prove
  `proc_start`/`load_begin`/`load_end` in a REAL trace did not run: a foreign process (pid **22436**,
  `deepseek_v4.exe`) was visible in nine samples from 13:34:08Z to 13:40:23Z, so no mutex was taken, no CUDA
  process started, no `--AllowBusy`, and no trace exists - the arm is NOT STARTED, not failed, and its exact
  command is recorded in the T-GPLR-56 block for the next clean window. Because the base change is the kind of
  thing that breaks a green gate silently, the rebase was modelled host-only instead: every `t_us`/`t0_us`/
  `run_id` in the recorded 13:10Z control shifted down by a constant 2,000,000 us, then re-checked. Measured
  identical on original vs rebased: `--check-telemetry` 0/0, `check-gplr53-embed-chunks.py --join` 0/0,
  `analyze-gplr53-decay.py` 0/0 with CHUNK0_SPIKE both, `analyze-gplr53-gap.py` 2/2 (its own refusal, not
  rebase-sensitive); no python tool gates on an absolute `t_us`. Related code fact: `t0_us` is captured at
  `llama_layer_streaming::init` (the sink-open site), which is why `run_start` sits at LINE 43 behind 40
  `pkg_member` lines, and why the markers will be the trace's first three lines.
- **T-GPLR-56 A1 telemetry LANDED (2026-09-22, host-only, no GPU): Q is now computable and refuses when it is not.**
  `llama_ls_tel_mark()` / `llama_ls_tel_flush_early()` buffer markers taken before the sink exists (the
  sink is `fopen(...,"w")` in `llama_layer_streaming::init`) and flush them as the trace's first lines;
  `main.cpp` marks `proc_start` at entry and `load_begin`/`load_end` around the load, default-on, no
  behaviour change. **A measured trap that would have killed every run:** `ggml_time_us()` divides by
  `timer_freq`, which only `ggml_time_init()` sets, so a marker at the top of `main()` died with
  `0xC0000094` (integer divide by zero, exit `-1073741676`) - caught by the new host test on its first
  run; fixed by calling `ggml_time_init()` at main entry and defensively inside the marker (InitOnce, so
  `llama_backend_init()`'s call stays a no-op). Consequence: P1 traces measure `t_us` from process entry,
  so they are not comparable to older traces on absolute `t_us` (checkers use differences). Host proof,
  measured: new CTest `test-layer-streaming-tel-mark` PASS, `tools/check-gplr56-q.py --selftest` 7/7 (incl. a
  relocation pair that must FAIL and four refusals, and it refuses the 13:10Z trace with exit 2 because that
  trace predates `proc_start`), `tools/run-tests.ps1 -Suite host` 42/42. The marker shape in a REAL trace is
  still an arm (`p1-off` at the next clean card); no latency or throughput win is claimed.
- **T-GPLR-56 A2 landed host-only (2026-09-22, no GPU, no code change, no arm run) - CORRECTS the draft.**
  A1: Q is NOT derivable at the pre-registered precision, because the trace begins at `run_start`
  (`t0_us` 2,974,883 is its first timestamp) so the load/init window is invisible and only a ~2.3 s bound
  exists; a `proc_start` event plus `load_begin`/`load_end` are required, and A1's criterion is recorded UNMET
  rather than met. A2: `pkg_cache` is built in `llama_layer_streaming::init`
  (`llama-layer-streaming.cpp:1862`, enumeration `:1936-1950`) from mapped tensor addresses (`:671`), so
  nothing can be registered before the load returns - "earlier" is worth a measured **66.4 ms** (3.4% of the
  1.9 s commit), and `p1-shard`+`p1-span` were merged into `p1-pace` because layer-major visits all 40 layers, so
  "the first chunk's packages" is all 40 groups. Geometry: 40 single-layer groups of ~462 MB, all queued in
  1,363 us, commit 9.70 GB/s, first stall 1,814.3 ms; one group ~47.7 ms. No latency or throughput win claimed.

## 2026-09-22: DRIVER CAPTURE PRODUCTION-PROVEN (one mutex hold, 13/13 arms PASS) - and the captured lines expose a pre-registered band that is unreachable by construction

The launcher change below was exercised end-to-end by a real A4 session at 13:10Z, because it cannot be
exercised host-side (it needs real drivers). `SESSION VERDICT: PASS (13/13 arms)`, driver `exit 0`, arms
13:10:07Z-13:13:33Z, binary `exe 1BDE9D5AFF326CB7` / `llama.dll BE55C90F0A5D796E` (the same pair the 12:44Z dose
run used, per its archived verdicts, so the comparison is same-binary).

- **Durable now:** `build/llama-cuda/a4-driver-a4.out` (3782 B, 67 lines) and `.err` (9265 B). The stdout holds
  what was previously thrown away - exe/dll identity, `mutex held`, every arm's `pass1 wall / excess`, the
  removal list, the `FALSIFIER` line, the dose table, the conserved sum, `gpu_post_run = clean`, `mutex
  released`, `VERDICT PASS`. `gpu_post_run` is also persisted per arm in all 13 verdicts.
- **Archive pick-up proven in production:** `-ArchiveOnly` (host-only, no clearing) archived the live files to
  `runs/20260922T131541Z` - 67 entries, 2 of them driver logs, copy and source `sha256_16` identical
  (`3A3230F71F5BE161`, `FADC9B6F9A1BA3B4`), sources untouched.
- **One run, not two:** one launch line, one predecessor archive (13:10:06Z), one monotone 13-arm verdict
  sequence, one driver log. The operator's Stop killed the POLLING command; the session finished at 13:13:38Z.
  No `llama-layer-streaming.exe` survived, `Global\AI-GPU-EXCLUSIVE-v1` has no live holder, card clean.
- **Reproduces on the same binary:** walk/dummy/both/pin 99.3-99.9% both runs, `delay` 92.7% -> 93.0%,
  `wrongtouch` 91.9% -> 86.2%, control repeat ~0; the control's own excess moved 1906.0 -> 1804.6 ms and the
  dose arms +110.5 to +312.5 ms, so the run-to-run floor here is ~+-110-130 ms (~6-7%). The dose law holds:
  1.00/0.95/0.96/0.98 absorbed per ms waited, conserved sum 3.70-3.76 s (spread 0.06 s).
- **OPEN DEFECT (recorded, not fixed - the band is pre-registered):** the driver states its band on pass-1
  WALL (`run-gplr53-a4-arm.ps1:11-15`) but its FALSIFIER paragraph and this record's conclusion use the
  EXCESS. The wall band is unreachable by construction - pass 1 has a ~1.75 s floor, so even the walk (99.3%
  excess removal) scores 50.3% on the wall, and the driver prints `pw1/pw2/pw3 = 50.3/50.6/50.8%`, below its own
  70% gate. On this run it printed `FALSIFIER SURVIVED ... 47.2% (< 70%)` (delay's wall) while that same arm's
  excess removal is 93.0% (126.2 ms vs 1804.6 ms). Escalates to the Planner; nothing was retuned.

## 2026-09-22: THE SESSION LAUNCHER NOW CAPTURES THE ARM DRIVERS' OUTPUT AND EXIT CODES (host-only change; no GPU spent)

`run-gplr53-session.ps1` used to start each arm driver with `Start-Process -WindowStyle Hidden` and no
redirection, so everything a driver printed was discarded and its exit code was never read - the driver's own
computed verdict lines (removal table, `FALSIFIER` line, dose curve) existed nowhere on disk, and a driver
that threw after writing PASS verdicts looked identical to a clean run.

- **Now:** `<session>-driver-<a3|a4>.out` / `.err` are written in `build/llama-cuda/`, the last 20 non-empty
  lines are echoed into the session log, both files are copied into the run's archived evidence (so the
  previous run's driver output survives), and the driver's exit code is reported, tabulated and folded into
  the session verdict - non-zero OR never-exited FAILS the session even when every arm says PASS.
- **A measured trap that would have failed every clean run:** with `-RedirectStandardOutput/Error`, a
  `Start-Process -PassThru` object reports `HasExited` but returns NULL `ExitCode` after a bounded
  `WaitForExit`, and `$null -ne 0` is true. Fix (probed, not guessed): retain the handle with
  `EnableRaisingEvents = $true` before the wait. `-Wait` also yields the code but blocks unbounded, which
  repeats the exact hazard this block already recorded and fixed elsewhere.
- **Host-only proof (no GPU, no mutex):** capture harness `[1]` exit=7 with stdout+stderr files and the
  driver's `FALSIFIER` line preserved, `[2]` clean exit=0, `[3]` still-running -> `not-exited` (live code
  never read), `[4]` negative control - the old shape captures nothing, `[5]` fold -> badDrivers=3 over
  `[7,0,not-exited,5]`; and `-ArchiveOnly` now archives the driver files (bytes matching, source hash
  unchanged), with the test cleaning up the run dir it created.
- **Limits:** the proof scripts live in `build/` (gitignored), the already-recorded tooling-location gap. The
  in-launcher capture itself is now production-proven (see the entry above). No arm behavior, tags, timeouts,
  rules or retention state changed.

## 2026-09-22: THE PASS-1 ONE-OFF IS ATTRIBUTED (host-only, from recorded traces) - it is the driver-side commit of the 17.23 GiB registered mapping, absorbed by chunk 0's embed step BEFORE any staging copy

No GPU, no mutex, no new arm, nothing rebuilt: this comes out of `host_mem` + `embed_chunk` in the traces
already on disk, and it corrects the framing of the whole T-GPLR-53 block.

- **Where it lives:** chunk 0's `embed_chunk` in pass 1 ONLY - 1907.1 ms at 4k, 2002.7 ms at 50k,
  1875.4 ms at 200k, against 0.63-0.9 ms for every other chunk and `host_ledger` pass-0 `embed_us`
  1911.8 ms with a -1912.3 ms `loop_us` residue.
- **The staging copies are NOT serialized.** The sweep's first `copy_complete` is 31.6 ms in the control,
  31.5 ms in the 1700 ms delay arm and 31.5 ms in the walk arm (median 31.4-31.5 ms over 240 copies in
  each). The one-off is paid before the first staging copy, by the embed step.
- **Its magnitude is the un-committed part of the mapped span.** File-backed working set runs
  1.95 GiB -> 19.03 GiB (+17.08 GiB) across pass 1 in EVERY arm and shape, and chunk 0's embed time is the
  outstanding remainder at 8.7-10.3 GiB/s (17.30 GiB -> 1907 ms; 1.52 GiB -> 132 ms; 0.24 GiB -> 15 ms);
  `sys_free` drops 50.61 -> 33.19 GiB to match.
- **Not our code doing it:** with mode 4 the process only sleeps and the prefetch worker is not even created,
  yet the file working set still grows ~10 GiB/s (2.00 GiB/200 ms, 15.78 GiB/1700 ms). Registration is 1.6 ms
  for all 17.23 GiB and `commit_bytes` stays 5.58 GiB - it enqueues the work, it does not do it.
- **Inferred (labelled):** the CUDA driver's page-lock/residency commit for the span registered by
  `ggml_backend_cuda_register_host_range` (T-GPLR-37; probe: 0.16 ms per 452 MiB window, then a pinned-ceiling
  14.44 GB/s DMA). Driver-internal vs OS servicing of the lock is NOT separated; that no runner thread touches
  the pages IS excluded.
- **`pin` proves the source path is the cause:** `eng_bytes` 1.46 GiB, chunk-0 embed 2.2 ms, but it reads the
  same file to build its mirror, so its pass-1 wall is 5.39 s instead.
- **Consequence:** the commit starts when the span is registered, which is AFTER model load and after the
  graph prebuild (+65.6 ms on a post-load t0), so pass 1 absorbs the remainder. Moving registration to the
  earliest point where the package spans are known would let the commit overlap work the run already does -
  the first candidate fix the dose curve points at, and the only shape of a win left. Its risk (host RAM
  bandwidth contention with load I/O) is to be measured, not assumed. No performance claim is made here.

## 2026-09-22: T-GPLR-53 A4 DOSE CURVE - PASS 13/13: absorption of the pass-1 excess is ~1:1 in the wait, the wall is CONSERVED, so pre-warm cannot be a wall-time win at any length

One mutex hold (preflight clean, 12:44:13-12:47:35Z), 13 arms all PASS with `gpu_post_run` clean on
every one. The mode-4 no-touch delay now takes its duration from a new opt-in flag
`--pp-prewarm-delay-ms N` (0..60000, default 1700 = the compiled control, refused unless mode 4 is set).

| dose | pre-pay ms | excess ms | absorbed | pass-1 s | total arm s | pre-pay + pass-1 s |
|---|---|---|---|---|---|---|
| `pw0` off control | 0.0 | 1905.95 | - | 3.67 | 14.5 | 3.67 |
| `d000` | 0.0 | 1834.11 | 3.8% | 3.60 | 14.8 | 3.60 |
| `d200` | 208.4 | 1622.54 | 14.9% | 3.39 | 14.7 | 3.60 |
| `d600` | 606.6 | 1054.58 | 44.7% | 2.82 | 14.6 | 3.43 |
| `d1200` | 1206.9 | 634.56 | 66.7% | 2.40 | 15.2 | 3.61 |
| `d1700` | 1713.0 | 131.35 | 93.1% | 1.90 | 14.8 | 3.61 |
| `delay` flag absent | 1701.5 | 139.04 | 92.7% | 1.91 | 15.2 | 3.61 |
| `pw1` walk | 1693.8 | 14.18 | 99.3% | 1.78 | 15.2 | 3.47 |
| `pw2` dummy | 1856.0 | 1.10 | 99.9% | 1.76 | 14.7 | 3.62 |
| `pin` no wait | 0.0 | 1.37 | 99.9% | 5.39 | 17.3 | 5.39 |
| `r2` | 0.0 | 1833.56 | 3.8% | 3.70 | 13.7 | 3.70 |

- **No step, a ramp:** 200/600/1200/1700 ms of waiting absorbs 283/851/1271/1775 ms of the excess
  (1.42, 1.42, 1.06, 1.04 ms per ms). Nothing "finishes" at a characteristic duration; the
  residual tracks (one-off - wait), so full absorption would need ~1.8-2.0 s of pre-pay.
- **The wall is conserved, which answers the shortening question:** pre-pay + pass-1 wall is
  3.43-3.70 s at every dose and the arm totals are flat (14.5 vs 14.6-15.2 s, no trend). The
  pre-pay can be made arbitrarily short, but only by leaving exactly that much excess behind - a
  1:1 purchase, never a rebate. No dose buys net wall time.
- **The most favourable point is inside the noise floor:** 200 ms costs 0.21 s and absorbs 0.28 s
  (0.07 s), while `d000` (sleeps nothing) differs from the control by 3.8% and `r2` by 3.8% -
  the metric's run-to-run floor is ~±4%, larger than any apparent gain on the curve.
- **Modality adds only a small term on top of duration:** `pw2`'s cleaner residual than `d1700`
  (1.10 vs 131.35 ms) is explained by its 143 ms longer pre-pay, which the ramp predicts as
  +143 ms absorbed (observed +130 ms). The source walk keeps a real ~117 ms page-side term
  (14.18 vs 131.35 ms; run 4 gave ~113 ms) - reproducible, and the whole surviving page signal.
- **`pin` still avoids the wait and is still worse overall:** 1.37 ms residual with no pre-pay,
  but pass-1 wall 5.39 s and total 17.3 s against the control's 14.5 s. Nothing removes the work.
- Reads as: the one-off is irreducible pending work that pass 1 pays serially unless something
  waits it out first, and the exchange rate is 1:1. It CONFIRMS the falsifier run's reading, so
  the withdrawn H1 section stays withdrawn and unchanged. **No performance claim**; the
  pre-registered contract and thresholds were not edited.
- n basis: each dose ran ONCE (`exe 72C6DFE1C5F5FA6B` / `llama.dll 2D2EA6B41E0A22E9`); the off
  control has five runs across four holds with excess 1866.72-1905.95 ms - the noise the curve is
  read against. Flags verified from the `.started` commands (five dose arms carry the flag;
  `delay` carries mode 4 with the flag absent = compiled default, 1701.5 vs 1713.0 ms).
- Host-validated before the GPU was touched: flag range refused, flag refused for modes 0/1/3/8
  and accepted for mode 4, off-state byte-identical, both drivers parse, dry run shows 13 arms
  with exactly the five dose values.
- Durability fired a third time: the falsifier run's evidence archived to
  `build/llama-cuda/runs/20260922T124413Z/` before this run overwrote those paths.

## 2026-09-22: T-GPLR-53 A4 FALSIFIER RUN - PASS 8/8, and the no-touch delay CLEARS 93.2% of the pass-1 excess, so the page-residency reading of the one-off COLLAPSES

Third A4 attempt, one mutex hold (preflight 12:37:19Z clean, 12:37:33-12:39:42Z), same shape
and gate regime as before; each arm's `gpu_post_run` is now recorded and **all 8 are clean**.
Two control modes were added for this run (engine `pp_prewarm` is a bitfield 0..15): 4 =
duration-matched no-touch delay, 8 = duration-matched wrong-region touch (64 MiB scratch).

| arm | mode | pre-pay ms | excess ms | removal vs control | pass-1 wall s |
|---|---|---|---|---|---|
| `pw0` | 0 off | - | 1866.72 | - | 3.62 |
| `pw1` | 1 source walk | 1691.1 | 13.69 | 99.3% | 1.78 |
| `pw2` | 2 dummy launch | 1845.2 | 1.42 | 99.9% | 1.77 |
| `pw3` | 3 both | 1690.3 | 1.51 | 99.9% | 1.77 |
| `delay` | **4 no-touch delay** | 1710.7 | 126.85 | **93.2%** | 1.90 |
| `wrongtouch` | **8 wrong-region touch** | 1706.1 | 125.57 | **93.3%** | 1.89 |
| `pin` | 0, pinned source | - | 1.71 | 99.9% | 5.34 |
| `r2` | 0, repeat 2 | - | 1834.68 | 1.7% | 3.63 |

- **FALSIFIER FIRED: the mechanism claim collapses.** A control that touches ZERO pages
  (`no-touch delay 1700 ms / 0 pages touched in 1710.6 ms`) removes 93.2% of the excess.
  The one-off is therefore NOT the first read of the 17.23 GiB source span; it is something
  that completes on its own given ~1.7 s. The wrong-region touch agrees within 1.3 ms, so
  page touching in any region buys nothing. Mode 1's 99.3% is explained by its own ~1.69 s
  DURATION, which the delay matches within 1.1%.
- **What survives is small:** the walk leaves 13.69 ms where the delay leaves 126.85 ms - a
  ~113 ms page-side residue (~6% of the 1.87 s). `pin` is still the only arm clean with no
  wait at all (1.71 ms) while paying ~1.7 s elsewhere (pass-1 wall 5.34 s).
- **The payer is neither the source read nor the submission** - a sleep pays it - so the
  contract's ESCALATE branch applies (hypothesis set incomplete; do not tune). Pre-warm is
  NOT adopted; no mode is recommended; no performance claim (pre-pay equals the excess it
  removes and arm walls are flat, so it is relocation at best).
- **Contract defect recorded, not edited:** the pre-registered comparison quantity is the
  pass-1 WALL with a ">= 70%" branch, but the one-off is only 1.84 s of a 3.62 s wall, so
  every real mode lands at ~51% and branch (i) could never fire on it. The second attempt's
  "branch (i) fires (H1)" used the corroborating excess term instead; that section now
  carries an explicit WITHDRAWN pointer.
- **n basis, explicit:** each mode has now run twice (12:28Z and 12:37Z) but on DIFFERENT
  binaries (`llama.dll 8A9F3D83F24D0FEC` then `5DE4FCD26921E6E8`), so it is a shape
  reproduction across builds, not a same-binary repeat; the off control agrees within 0.8%
  and modes 1-3 within 1.6%, which is what licenses reading them together.
- **Provenance fixes landed:** `gpu_post_run` is now persisted per arm in both GPLR-53
  drivers (it had been printed only), verified present in all 8 verdicts of this run; and
  the n=1 (now n=2) basis is stated in the record instead of implied.
- **Host-validated before the GPU was touched:** `--pp-prewarm 16` refused (exit 2), modes
  4/8 accepted and reaching the same refusal table as 1-3, off-state byte-identical again,
  and the driver dry run prints 8 arms with exactly `--pp-prewarm 1/2/3/4/8`.
- Durability fired again: the second attempt's 8-arm evidence is archived intact at
  `build/llama-cuda/runs/20260922T123732Z/` (30 manifest entries) beside the older
  `runs/20260922T122800Z/` and the accepted A3 evidence.
- Recorded gap, not fixed: the session launcher does not persist its child driver's stdout,
  so the driver's own `FALSIFIER` line is not in `session4.out`; the table above is
  recomputed from the per-arm verdict artifacts.

## 2026-09-22: T-GPLR-53 A4 CLOSES - PASS 6/6 in one mutex hold, and the modality evidence points at HOST PAGE RESIDENCY (H1), not per-submission overhead (H3)

> SUPERSEDED the same day by the FALSIFIER RUN entry above: the H1 reading below is withdrawn
> (a no-touch delay removes 93.2% of the same excess), and so is "H3 is refuted". The raw
> per-arm numbers, the size-insensitivity and the pinned-source observation still stand.

Second A4 attempt, one mutex hold, card clean at 12:27:53Z and again on `gpu_post_run`,
same binary pair as A3 (exe FE3B0A108C1D9BC9 / llama.dll 8A9F3D83F24D0FEC). All six arms
PASS all three gates (`--check-telemetry` exit 0, `--join` exit 0, decay exit 0 =
CHUNK0_SPIKE), and the `r2` arm's route parity is clean (160 events, 80 distinct, paired x2).

- **Measured, the decisive number:** mode 1 (a read-only CPU page walk, `17.068 GiB /
  4474368 pages in 1673.5 ms`) removes 99.3% of the pass-1 chunk-0 excess (1881.94 ->
  13.52 ms, pass-1 wall 3.65 -> 1.79 s) with NO GPU submission at all. Mode 2 (one dummy
  H2D + event sync) removes 99.9%, but costs 1832.1 ms ITSELF; in mode 3 the walk costs
  1673.7 ms and the dummy that follows costs only 18.2 ms. One shared one-off, not two
  costs. Registration is NOT it: `18500517888 B (17.23 GiB) pinned in 1.5-1.6 ms` in every
  mmap arm.
- **`pin` (mode 0, pinned source, no pre-warm) removes it from the embed bracket** (excess
  1.34 ms) - but its pass-1 wall is 5.44 s, 1.79 s WORSE than the off control, so the cost
  is RELOCATED into pass-1 setup, not eliminated. Warm state is untouched everywhere
  (passes 2/3 1.57-1.58 s; warm median 0.632-0.663 ms across all six arms).
- **No performance claim; pre-warm is relocation, not a saving.** Pre-pay (1673-1832 ms)
  is the same magnitude as the excess removed (1831-1882 ms) and arm wall times are flat to
  worse (14.7 s -> 15.3 / 15.3 / 15.1 s).
- **The pre-registered A4 rule does not discriminate here, and is not being reinterpreted:**
  its branch (i) "removes >= 70% -> page-side (H1)" AND branch (ii) "removes it ->
  submission-side (H3)" both fire, because the modalities turned out not to be mutually
  exclusive. A3's rule maps CHUNK0_SPIKE -> H3; A4's branch (i) points to H1. Both were
  pre-registered; neither rule, threshold nor accepted A3 verdict was changed. **This is
  the task's remaining open item and it is a Planner adjudication, not an arm.**
- Evidence durability proven in production: the launcher archived the earlier shape-bug
  attempt's six FAIL artifacts into `build/llama-cuda/runs/20260922T122800Z/` (with a
  manifest of source, size and truncated SHA-256) before overwriting them; the accepted A3
  traces, verdicts and their mtimes are untouched. Mode-0-equals-off is now verified at run
  time as well (off arms log no `pp-prewarm(` line, `prewarm_engaged: false`).

## 2026-09-22: T-GPLR-53 A3 CLOSES ON REAL ARM VERDICTS - PASS 2/2, CHUNK0_SPIKE reproduced on a second run; A4 half BLOCKED by a foreign process and refused in 394 ms

(A4's blocker below was cleared later the same day: see the entry at the top.)

Second A3 session (one mutex hold, card clean before and after, same binary pair
exe FE3B0A108C1D9BC9 / llama.dll 8A9F3D83F24D0FEC as the first run, so this is a
genuine repeat).

- **Both arms PASS** and all three gates are green on each: `--check-telemetry`
  exit 0, `check-gplr53-embed-chunks --join` exit 0 (`CHK-join ok`), decay exit 0.
  50k: CHUNK0_SPIKE, chunk-0 share 0.9997, excess 2001.91 ms, walls
  20.25/18.12/18.16 s, 67.6 s. 200k: CHUNK0_SPIKE, chunk-0 share 0.9987, excess
  1871.71 ms, walls 135.84/134.67/135.54 s, 484.9 s. `gpu_pre_run` clean on both.
- **CHUNK0_SPIKE is now the verdict of four independent arm-runs** (both shapes,
  twice) and the excess stays size-insensitive in the repeat (2001.91 ms at 50k vs
  1871.71 ms at 200k). Nuance the repeat exposes: the DECIDING statistics are stable
  (~0.999 in all four runs) while `rho` is not (+0.315 / +0.0159 at 50k; +0.110 /
  +0.2924 at 200k) - consistent with the recorded limit that the verdict rests on
  the chunk-0 concentration and warm tail, not on the trend statistic.
- **A4 NOT RUN.** Blocked by pid **21092**
  (`colibri\pf06-wt\c\deepseek_v4.exe`, 18.3-18.7 GiB, 51-91%) resident and active
  across a 4-sample re-probe 12:12:20Z -> 12:14:50Z and still resident at
  12:15:25Z. The attempt stopped rather than waiting; `-AllowBusy` not passed, no
  CUDA process started. The new preflight refused the A4 half with **exit 125 in
  394 ms**, naming the PID - the recorded one-hour `WaitOne` hazard did not
  materialise. All six A4 arms (`pw0/pw1/pw2/pw3/pin/r2`) have no artifact of any
  kind.
- **H1 vs H3 remains UNDISTINGUISHED.** A3's rule maps CHUNK0_SPIKE to H3, and
  size-insensitivity is consistent with a fixed one-off completion rather than a
  page-proportional cost, but A4 (mode 1 = CPU read-touch, H1's page-side
  candidate; mode 2 = dummy launch + sync, H3's submission-side candidate) is what
  separates them, and it did not run.
- **Mode-0 vs off default: host-verified byte-identical** runner output and zero
  pre-warm chatter, matching the code path (`pp_prewarm == 0` returns before
  touching anything); the driver's run-time assertion still needs the A4 run.
- **Evidence durability landed (host-only, no GPU).** The fixed arm tags meant a
  re-run overwrote the previous run's trace and verdict triple (the original A3 FAIL
  artifacts are gone; only the record prose and `gplr53-a3-session.out` remain).
  `tools/run-gplr53-session.ps1` now copies the artifacts it is about to replace
  into `build/llama-cuda/runs/<utc-stamp>/` with a hashed `manifest.json`, AFTER the
  busy preflight and BEFORE `Clear-ArmVerdict`, so a refused run writes nothing and
  a real run cannot lose its predecessor. New `-ArchiveOnly` snapshots and exits
  without launching. Proof: 10 artifacts archived, `python build/check-archive.py`
  = **ARCHIVE PROOF: PASS** (copies byte-identical, sources intact), source hashes
  unchanged before/after, `-ValidateOnly` still exit 0 and `-PreflightOnly` still
  exit 125 on a busy card. Tooling only: no arm action, tag, rule, engine or runner
  change.
- No performance claim is made anywhere above; the numbers quoted are trace and
  verdict contents.

## 2026-09-22: T-GPLR-54 A3 EXECUTED (owner-invoked) - VERDICT: BENIGN (H1); the u4096 escape at index 41 was a near-tie tie-break; one A2 wiring defect found and fixed by the session itself

The owner invoked the parked A3 discriminator session (one GPU mutex hold,
`tools/run-gplr54-a3-session.ps1` + detached launcher `tools/run-gplr54-a3-launch.ps1`;
wrapper verdict PASS, gpu_pre/post both clean). This DISCHARGES the neutrality-pair
obligation A2 had pre-registered and decides H1 vs H2 by the archived block's
pre-registered rule - now with measured evidence instead of the closure's stated prior.

- VERDICT (tools/analyze-gplr54-a3-verdict.py, written before the run, selftest 10/10):
  **BENIGN (H1)**, exit 0. `margin_u4096(step 40) = 1.281971 <= M_band + 1.0 = 21.539429`
  (M_band = 20.539429, max over 123 multi-chunk rows in the derived window 0..40,
  window derived from the ID-CHECK's verified D=41, not assumed). CONTEXT (not the
  rule): at the flip step itself the u4096 margin was **0.138065** vs the u512
  reference's **0.0127** - both near-ties over the same top-2 pair {248045, 846}; the
  1-chunk shape tipped a coin-flip decision the other way. The escape was NOT a
  confident different choice.
- ALL PRE-REGISTERED GATES GREEN: neutrality (ids(u512-on) == ids(u512-off)
  byte-identical, 96 ids), C4 (both repeats byte-identical), ID-CHECK (u2048-on
  IDENTICAL K=0; u4096-on reproduces the recorded shape EXACTLY: D=41, K=7, the same
  7-id inserted sequence, tail realigned, template-only), instrument integrity
  (depth>=96 declared, rows contiguous, top1==pick, margin arithmetic).
- A2 WIRING DEFECT (found by run 1's own integrity gate, fixed before run 2):
  the engine's run_start setter embedded `tg_logit_margin` into the run_start
  string but NEVER assigned the member the emitter gates on - run_start claimed
  the readout was on while the trace stayed silent (0 rows). Run 1 = pilot that
  falsified the A2 coverage claim; traces preserved as
  `build/llama-cuda/gplr54-a3-run1-*.jsonl`. Fix: one-line member assignment in
  the setter (`llama-layer-streaming.cpp`); validated by run 2 (96 real rows per
  margin arm, A2 checker PASS on all 6 real traces for the first time) and by
  CROSS-BINARY determinism: run-1 (pre-fix binary) ids == run-2 (post-fix binary)
  ids on all 4 distinct arms - the telemetry-only fix provably did not perturb
  the pick, and D=41/K=7 reproduced across two binaries.
- CHECKER BRING-UP (both bugs were analyzer-side, caught by the gates as
  designed): (1) the off-arm invariant wrongly demanded the run_start key be
  ABSENT; the engine's real off shape declares `tg_logit_margin: 0` (explicit
  provenance) - invariant is zero rows under absent-or-zero depth, with two new
  selftest cases. (2) the session-surprise gate compared agreement POSITIONALLY
  across all six arms, which the insertion offset makes meaningless after D; the
  rule's own text ("of the multi-chunk arms", indices 0..40) is now implemented
  verbatim, window derived from verified D.
- SUITE CLASSIFICATION (zero task-owned failures): with `Q38LS_MODEL_PATH`
  restored, the ctest layer family is 72/74 and QA smoke PASSES (5/5 answers,
  negative control fails as designed, MoE model). The two real failures are
  PRE-EXISTING stale test definitions, not from this session: #53
  qwen38-layer-boundary and #54 layer-slot-lifecycle run `--layer-slots 1`
  without `--pp-async-feed 0`, and the async-feed default flipped to 1 in the
  T-GPLR-25 commit (68cd1726d, long in HEAD) whose one-slot reuse gate refuses
  exactly that shape. Recorded, not applied: the fix is adding
  `--pp-async-feed 0` to those two CTest definitions - a vendored-tree test
  change outside this task's contract, awaiting owner authorization. All other
  failures this session were the env-var class (the Freebuff restart dropped
  `Q38LS_MODEL_PATH`; `tools/run-gplr54-suite.ps1` now sets it for detached
  suite runs). Guards: doc-size PASS, resplit --verify PASS, nested
  `git diff --check` clean; one pre-existing trailing-whitespace complaint at
  plan line 982 (T-GPLR-53 block, another thread's edit, left alone).
- Evidence: `build/llama-cuda/gplr54-a3-*.jsonl` (run 2) + `-run1-*` (pilot),
  `build/llama-cuda/verdicts/gplr54-a3{,-run1}.verdict.json`;
  `build/llama-cuda/verdicts/` wrapper verdicts PASS with clean GPU probes.
  NOT claimed: anything about timing (never the task), any u>=4096 config's
  fitness (soft gates), or that other prompts cannot flip differently - the
  rule (AGENT_ENVIRONMENT.md, "Greedy identity at u >= 4096") remains the
  standing arm gate; its benign reading is now measured, not assumed.

## 2026-09-22: T-GPLR-53 A3 MECHANISM VERDICT PRODUCED - the pass-1 embed excess is a chunk-0 spike (pre-registered mapping -> H3), decided from the traces already recorded, zero new GPU time

The A3 arms ran earlier today on a clean card (one mutex hold) and BOTH FAILed their
tool checks. The blocker turned out to be a defect in the rule's own loader, not
missing telemetry: `tools/analyze-gplr53-decay.py` bucketed passes by an event
`phase` field, which in this format means PP-vs-TG and does not exist on
`embed_chunk` at all, so every chunk landed in pass 0 and the rule refused its own
input (exit 2 = never a pass).

- FIX (tool-only, no emitter change, no runner change): bucketing is now by
  chunk-sequence reset, with ambiguity refused rather than guessed (blocks must be
  `0..n-1`, equal length, and agree with the `host_ledger` boundary count when one is
  present; `pp_pass` is advisory because it fires at pass transitions and is measured
  at n-1). Fixtures moved to the recorded trace shape - the old ones carried a
  `phase` field, which is why the defect survived them - and the bucketing now has
  its own controls: `--selftest` 10/10 including three refusals and a 3-pass
  positive control.
- VERDICT from the recorded traces, no second mutex hold: **CHUNK0_SPIKE at both
  shapes (exit 0)**. Pass totals 1937.01/63.40/63.41 ms at 50k and
  2040.50/254.62/255.26 ms at 200k; excess at chunk 0 = 1872.39 / 1785.27 ms; largest
  excess EXCLUDING chunk 0 = 0.114 / 0.263 ms; chunk-0 share 0.9991 / 0.9965. The
  pass-1 excess is ~1.8 s at BOTH shapes while the prompt grew 4x (and while the
  warm per-pass embed grew 63.4 -> 254.6 ms), i.e. size-insensitive - consistent
  with a fixed one-off completion (F5's 18.5 GB registered span) rather than a
  page-proportional cost. Labeled telemetry-only: no performance claim follows.
- Honest limit: rho is POSITIVE (+0.315 / +0.110), so the FLAT branch would have
  fired on 200k had CHUNK0_SPIKE not taken precedence; the verdict rests on the
  chunk-0 concentration plus the warm tail, not on the trend statistic. H1-vs-H3
  remains A4's modality question.
- A2's stamp fix is CONFIRMED by the same traces: the `-1` drift class present in
  every pre-A2 trace (`-1 x85..119`) is gone (`+0 x118`).
- RESOLVED (same day, host-only, no GPU): the 6-sample layer-38/39 "boundary stamp"
  is a CHECKER KEYING artifact, not a stale stamp. `recon_flush_all()` writes a
  sample's `ubatch` from the LIVE `n_ubatches` at flush time while its
  `(layer, generation)` come from the ring entry that describes the measured copy
  (`llama-layer-streaming.cpp:1323-1325` vs `e.il`/`e.gen` at `:1291-1292`), so the
  old `(layer, ubatch)` key reported a phantom one-pass-stride drift on exactly the
  samples flushed across a pass boundary. Re-keyed `CHK-join` to
  `(layer, generation)` plus two ordering conditions; matching is now 124/124 on
  both A3 traces (0 flushed before their copy, `copy.t_us <= sample.t_us` for all
  124), and the pre-A2 traces still fail **100%** (0/89, 0/125, 0/125, 0/85), so the
  rider kept its teeth. Checker `--selftest` 20/20, including a control that was
  RE-SPECIFIED (the old -14 case lands on a real copy and is the flush-later shape
  the corrected keying must accept) and two new hard controls. A3's three gates are
  now GREEN on the traces already on disk: check-telemetry 0, CHK-join ok, decay
  CHUNK0_SPIKE. The bucketing the verdict rests on uses the `embed_chunk` stream
  only, so this does NOT invalidate it; the pre-registered rule was not touched.
  RESOLVED (2026-09-22, owner-authorized one-liner): the emitter fix is LANDED -
  `ls_recon_ev` carries `ubatch` captured at ENQUEUE time and the emitter prints the
  entry's own label; proven on a fresh 4k `--pp-repeat 2 --pp-recon 1` arm where all
  77 samples join exactly at their own (layer, ubatch) key - zero stale-label
  outliers (the old code produced the -14/-15 class on the same shape family).
- Evidence: `build/llama-cuda/gplr53-a3-{50k,200k}.{jsonl,log}`,
  `build/llama-cuda/verdicts/gplr53-a3-*.verdict.json`; full write-up in
  `docs/plans/generic-prefill-layer-residency-plan.md` (T-GPLR-53). T-GPLR-19/26
  retention state and T-GPLR-55's drafted shapes were not touched.

## 2026-09-22: T-GPLR-54 CLOSED ON THE ARM RULE (owner decision) - id-equivalence at u >= 4096 is unavailable by design; the enforced ID-CHECK replaces it; A2 parked ready, A3/A4 not run, zero GPU time spent

The owner took the third option the block left open: close T-GPLR-54 without
the A3 GPU session. Documents only; no engine, runner, or default change.

- THE DECISION: the mechanism question (near-tie tie-break vs confidently
  different distribution) stays deliberately open, and the discriminator's
  arm rule becomes the standing equivalence contract: at u >= 4096 (and any
  future arm that changes prefill chunk shape, KV type, or the prefill
  path), 96-id identity is NEVER the bar - the enforced A1 ID-CHECK is
  (`python tools/analyze-gplr54-ubatch-identity.py <candidate> <reference>`;
  PASS = IDENTICAL or BOUNDED-INSERTION with answer span intact; FAIL =
  escalate before recording timings; cannot-evaluate = exit 2, never a pass).
  The rule is now a permanent environment fact in
  `docs/AGENT_ENVIRONMENT.md` ("Greedy identity at u >= 4096").
- WHY THIS IS SAFE: the divergence is one bounded template-region insertion
  with byte-exact re-synchronization, PAST the graded answer span (needle
  grading 5/5 at u4096), H1 (batch-shape numerics) stays the stated prior,
  and every session's own reference pair still verifies K = 0 - so a REAL
  derailment (content tokens, no realignment, inside the answer span,
  non-repeating) still fails the check loudly. What is given up: the
  margin-band evidence that would decide H1 vs H2, and per-step logit
  visibility, until someone re-opens it.
- DISPOSITION OF THE ARMS: A1 done (enforced tool, selftest 7/7, wired as
  [t54-a1]); A2 LANDED AND PARKED READY - `--tg-logit-margin` is default-off,
  host-validated (41/41 CTest, checker 11/11, [t54-a2] in the host suite),
  and its one self-named obligation (the on/off neutrality pair) remains
  pre-registered as the first arm of any future A3; A3/A4 not run by design.
  The plan block carries the full RESULT and the unchanged pre-registered
  A3 rule for whoever re-opens the question.
- LIVE BACKLOG: T-GPLR-53 only (in-progress; A1 + A2 done; A3/A4 mutex
  session is the only GPU spend in the family). T-GPLR-54 is done; the resplit
  moves it to the archive.

## 2026-09-21: T-GPLR-54 A2 EXECUTED - the default-off --tg-logit-margin readout lands (runner + engine, host-validated); the on/off neutrality pair is pre-registered as A3's first arm

- WHAT: `--tg-logit-margin N` (<= 0 = off, the default) records one
  `tg_logits {step, top1_id, top1_logit, top2_id, top2_logit, margin, t_us}`
  per greedy TG step for the first N steps, carrying the SAME penalized top-2
  values the pick consumed (the T-GPLR-06 stderr line, now in the trace).
  This is the instrument A3's margin-band decision rule consumes; without it
  the benign-tie-break vs shape-defect question has no input.
- DEFAULT-OFF INVARIANT, machine-checked: `run_start` carries
  `tg_logit_margin` only when the flag is passed (pre-A2 recordings
  byte-unchanged), and `tools/check-gplr54-logits.py` CHK-gate fails any
  trace with rows under an undeclared key. `--selftest` 11/11 (3 positives,
  6 firing negative controls, 2 exit-2 refusals).
- HOST VALIDATION: fixture `tests/gplr54-tg-logits.jsonl` (from the recorded
  gplr38-u512 shape + the recorded stderr margins of bare-default-k.log;
  fixture top-1 ids ARE the gold ids, so top1==pick holds by construction);
  both validators exit 0 (new CTest pair
  `layer-streaming-{check,validate}-gplr54-tg-logits`; host suite **41/41**,
  was 39; `[t54-a2]` wired into every host-only suite). The first fixture
  draft failed 3 strict-validator invariants and was repaired to the minimal
  valid shape - recorded because it re-proves the validators bite.
- HONEST CAVEAT: the AC's same-session on/off sampler-neutrality pair needs
  one GPU arm (a host cannot execute the sampler); structural neutrality is
  host-proven (read-only tap before sampler-state advance, no added sampler
  call) and the pair is PRE-REGISTERED as A3's first arm (u512 x2, flag off
  vs on, ids must be identical) - A3's verdict is void without it.
- NOT CLAIMED: any timing, any mechanism verdict, any default change. Next:
  A3 (one owner-invoked mutex session: the neutrality pair, the u-arms with
  `--tg-logit-margin 96`, the pre-registered margin-band rule).

## 2026-09-21: +12 x2 RECON-JOIN ANOMALY RESOLVED - not a second join defect: the checker's positional pairing mispaired across passes; the keyed (layer, ubatch) join removes the phantom, and one real label artifact is exposed and characterized instead

Host-only (traces + code reading), no GPU. Question: is the +12 x2 drift on
`gplr50-ctl-4k-f16` a real second stamp defect or an artifact? Answer: an
ARTIFACT of the checker's pairing, plus one real (harmless) telemetry-label
artifact underneath.

- The two +12 pairs both paired a PASS-2 sample (ub 16) against the same
  layer's PASS-1 submit (ub 8), because layers 17 and 34 were sampled only in
  pass 2 and the checker's per-layer positional index landed on their first
  (pass-1) submit. Measured inter-pass generation stride on the same layers:
  13. 13 - 1 (the known pre-fix stamp) = +12. The phantom is arithmetic, not a
  second defect.
- Keyed (layer, ubatch) pairing over ALL 65 samples: -1 x64, -15 x1, 0
  unpaired - the +12s vanish. The same pairing on `gplr52-a2-50k`: -1 x119,
  -14 x3, -15 x3 (0 unpaired). (layer, ubatch) identifies exactly one submit
  on both traces (max 1 per key over 4,185 submits), so the key is
  well-defined.
- The residual -14/-15 class is the lazy recon ring's LABEL artifact: a pass's
  last sampled banks sit in the ring until the next pass's first bank flushes
  them, and the emitter prints the CURRENT `n_ubatches` as the sample's ubatch
  label (src/llama-layer-streaming.cpp:1220 prints the global, not the sampled
  event's). Proof: each outlier's GENERATION sits at exactly -1 from its true
  earlier-pass copy (verified per-sample in both traces), and the outliers
  cluster on the last sampled layers at pass boundaries (lay 39 in gplr50;
  lays 38/39 in gplr52) - the stale-ring fingerprint. No arithmetic claim in
  any A1/A2/A3 result depends on the ubatch label of a recon_attr sample;
  the COPY-ATTRIBUTION numbers (copy_gpu_ms, lag_ms) belong to the stamped
  generation, which is correct everywhere on these traces.
- Checker hardened so A3's drift-0 gate cannot see phantoms:
  `tools/check-gplr53-embed-chunks.py --join` now keys on (layer, ubatch),
  refuses events missing the key (exit 2, not guessed), and its `--selftest`
  grew 15/15 -> **18/18** with the measured gplr50 shape encoded as negative
  controls (a TG-burst layer pairing positionally onto its pass-1 copy must
  FAIL at the TRUE drift -1, not the phantom +12; a genuinely wrong stamp
  fails; a missing-ubatch event is REFUSED). Re-measured: gplr50 --join now
  FAILs naming "-15 x1, -1 x64" (the true pre-fix signal), gplr52-a2-50k
  names "-15 x3, -14 x3, -1 x119". Selftest re-run 18/18; both validators
  unaffected (no engine change).
- What remains genuinely open from this: nothing for the join itself (A3's
  drift-0 expectation stands, on the keyed checker). The recon emitter's
  ubatch label is a known cosmetic defect (one-line engine fix, defer unless a
  future analysis needs per-ubatch sample attribution).

## 2026-09-21: T-GPLR-54 A1 EXECUTED - the ID-CHECK is now a checked tool; D = 41 / K = 7 reproduced host-only, no GPU

`tools/analyze-gplr54-ubatch-identity.py` landed: the pre-registered ID-CHECK
(C1 prefix D >= 41, C2 one bounded insertion with byte-exact tail realignment,
C3 template-only ids, C4 `--repeat` determinism gate for A3's x2 arms), with
`--selftest` 7/7 (two positives, three negative controls that fire, the
recorded-shape check, the missing-trace refusal) and the verified exit contract
(PASS 0 / FAIL 1 / no-`tg_tokens` refusal 2 - `gplr38-100k-*` refuses, exit 2,
NOT a pass). On the recorded arms (no GPU): u1024 and u2048 are IDENTICAL to
u512 over all 96 ids; u4096 is BOUNDED-INSERTION D = 41, K = 7, inserted
`[248045, 248045, 248045, 248045, 248046, 198, 248045]`, tail realigned,
template-only - the draft's F1-F3 reproduce exactly, and the corrected wording
below is now enforced by CHK expectations instead of being prose.
One honest root-cause fix during validation: the first C2 form hardcoded the
equal-length realignment (passes these arms, rejects a true insertion at any
other stream length); the general form (g = len(cand)-len(ref)) covers both and
still rejects substitutions - caught by the pre-registered fixture discipline.
Wired as `[t54-a1]` into every host-only suite: `-Suite host` observed SELFTEST
PASS, 39/39 CTest, doc-size + plan-split PASS, suite SUCCESS. A2 (the
default-off `--tg-logit-margin` instrument) is next; A3 needs an owner-invoked
mutex session.

## 2026-09-21: T-GPLR-54 DRAFTED - the last ownerless correctness observation is now a task, and its shape is measurably sharper than the record said

The u4096 greedy stream is not a "repeat loop": a host-only re-read of the four
recorded 4k arms (`gplr38-u512/u1024/u2048/u4096.jsonl`, same binary era, 4096-token
prompt, handoff auto-on, 3 slots, 96 greedy ids each) shows that u1024 and u2048 are
ID-IDENTICAL to u512 for all 96 ids, while u4096 equals u512 with exactly ONE
contiguous 7-id insertion at index 41
(`248045, 248045, 248045, 248045, 248046, 198, 248045`) and is byte-identical after
realignment (`u4096[48:96] == u512[41:89]`). One argmax flip at the first decision
after the template region both arms enter at index 35, then exact
re-synchronization - not a derailment region, and not a truncation (the needle answer
is complete and ends before index 41). The 2026-09-21 Oddity 2 entry is corrected
above accordingly.

T-GPLR-54 (draft) turns that into a falsifiable task: a named ID-CHECK (prefix
identity with D >= 41, exactly one bounded insertion with byte-exact realignment,
template-only inserted ids, and repeat determinism) plus one default-off runner-side
instrument (`--tg-logit-margin N`, top1/top2 margin per TG step) that decides whether
the flip is an ordinary near-tie (benign; the u>=4096 arm rule becomes "prefix +
single-insertion realignment, never 96-id identity") or a confidently different
distribution at the 1-chunk prefill shape (escalate with a minimal reproducer).
Negative controls must fire (content-token insertion, non-realigning tail,
divergence inside the answer region). No GPU time used and none needed for A1; A3 is
one owner-invoked mutex session at 4k. Live backlog is now T-GPLR-53 (in-progress) +
T-GPLR-54 (draft).

## 2026-09-21: T-GPLR-53 A2 EXECUTED (host-validated, no GPU) - the recon-gated `embed_chunk` event lands, the recon generation off-by-one is fixed, and the pass-1 profile is now measurable on a real first pass

- Engine (telemetry only): `tel_embed_chunk(chunk, n_tokens, setup_us, compute_us,
  span_us)` in `llama-layer-streaming.{h,cpp}`, called from the step-1 loop of
  `llama_context::layer_major_pp`; the event is self-gated on `pp_recon`, so a
  non-recon run emits nothing (no new flag, no ledger field, no arithmetic,
  routing or package change). `span_us` is the chunk's whole wall, so a pass's
  sum reproduces `host_ledger.embed_us` instead of estimating it.
  `tools/analyze-gplr53-gap.py` was re-run after the change and still PASSES on
  the record (F1 1.0014 on gplr52-a3-mmap, F6 0.86 ms, F1b 1.3%).
- RECON GENERATION JOIN FIXED (the rider the A2 gate needed): `stage_members` now
  receives the generation its copies will be tagged with and hands THAT to
  `recon_begin_bank`, instead of reading `slot.generation` before the increment.
  Measured host-only on the pre-fix recordings: the drift is **-1 on all 85
  paired samples** of `gplr52-a3-mmap` (and -1 x63 with +12 x2 on
  `gplr50-ctl-4k-f16` - RESOLVED 2026-09-21: it was the checker's positional
  pairing landing on a foreign pass's submit, not a second stamp defect; the
  keyed (layer, ubatch) join puts every one of its samples at the expected
  drift, and the residual -15 x1 is the lazy-ring ubatch-label artifact with a
correct generation stamp - see the plan's A2 addendum). A
  (layer, generation) join therefore matched nothing before; it is exact by
  construction now, and A3 must re-prove it with drift 0 on a post-fix trace.
- FIXTURE + BOTH VALIDATORS (measured, not assumed):
  `tools/layer-streaming/gen-gplr53-fixtures.py` builds
  `tests/gplr53-embed-chunk.jsonl` from the recorded `gplr52-a2-50k` metadata and
  ledger (4 chunks, front-loaded 0.62/0.18/0.13/0.07 of the measured 2,246,290 us
  pass-1 embed, `loop_us = -(embed + output)` kept). Two new CTest targets run
  `--check-telemetry` and `--validate-telemetry` on a build-dir copy: both exit 0,
  so the validators tolerate the new event. One pass only: the validator allows a
  single `phase_summary` per phase and a multi-pass run emits one per pass
  (measured - a second pp summary is rejected as a duplicate).
- CHECKER: `tools/check-gplr53-embed-chunks.py` (CHK-shape, CHK-order, CHK-cover,
  CHK-sum with the 1 ms tolerance, CHK-gate, plus `--join` for the generation
  join). `--selftest` 15/15 with seven negative controls firing and four exit-2
  refusals ("pairing is refused, never guessed"); on the checked-in fixture
  `CHK-sum` diff is **0.0 us**. Wired into `tools/run-tests.ps1` as `[t53-a2]`
  for every host-only suite; the two CTest targets joined the `host` include list.
- VALIDATION: build `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming` exit 0; `ctest -R gplr53-embed-chunk` 2/2 pass; host
  suite `SUCCESS` with **39/39** CTest (was 37) plus doc-size PASS, plan-split
  PASS `[plan-split]`, `[t53-a1]` and `[t53-a2]` selftests PASS. `git diff --check`
  clean in both trees.
- RESPLIT HYGIENE: the 2026-09-21 apply left the archive ending on a blank line
  (each block chunk carries its separator), which `git diff --check` flags as
  "new blank line at EOF". Stripped, and `--apply` now strips it itself
  (`archive_terminator_stripped`, which `--verify` restores before its span
  checks). Recorded as a third manifest correction; archive body hash re-recorded
  (6b60b61f... -> 2b8b1c3a...); `--verify` PASS with all three corrections printed.
- NOT CLAIMED: any timing, any mechanism, any default change. A3 (the decay
  discriminator, 200k/50k, `--copy-source mmap-registered`, `--pp-repeat 3`,
  `--pp-recon 1`) and A4 (the pre-warm modality arms at 4k) remain the only GPU
  spend, and A3 must also run the new `--join` check and expect drift 0.

## 2026-09-21: PLAN CLOSE-OUT SWEEP - T-GPLR-52 closed (AC5), T-GPLR-37 status fixed, resplit applied; ONLY T-GPLR-53 stays live, and the resplit tool's header bug (which silently ate the live index) is fixed + guarded

Host-only, no GPU, no mutex, no engine change.

- T-GPLR-52 AC5 CLOSED: `Status: done (2026-09-21)` + the `#### RESULT` block are
  in the block, and the block moved to the archive in this resplit (verify
  PASS). The closure commit is NOT made - no commit is authorized yet, so the
  block's `Commit:` field lists only the substantive commits (nested 23e5d1b6e;
  outer 59a0266, 0b42781, 25c5581, 828ef2f, 48e8901, 8f648fb, f5ed99c).
- T-GPLR-37's stale `Status: in-progress (2026-09-17)` line (contradicting its own
  PROMOTED-TO-DEFAULT RESULT, and the reason the last resplit kept it live) now
  reads `Status: CLOSED (2026-09-20) - PROMOTED TO DEFAULT by owner decision`
  with the earlier reading preserved verbatim; the block is archived.
- RESPLIT APPLIED: T-GPLR-37 + T-GPLR-52 moved to the archive (block spans
  byte-identical, 322514 B body, archive 4990 lines). Live file = **1 block,
  T-GPLR-53 only** (257 lines, apply-time span 257). Dry run confirms nothing
  else is movable; the live block's T-53 A1 work is untouched by the split.
- DEFECT FOUND AND FIXED (would otherwise have repeated on every future split):
  `main_apply` split the header at the first `"### "` occurrence, which sits
  INSIDE index line 1 (`- ### T-GPLR-25: ...`). The apply therefore dropped the
  whole 21-entry index AND the rules paragraph, glued the preamble's trailing
  `-` onto index line 1, and indented the surviving block heading - and the
  follow-up `--verify` died with StopIteration instead of reporting it.
  Repairs: (a) the index is now DERIVED from the archive's block order (never
  carried forward) and the rules paragraph is a constant; (b) `--verify` fails
  loudly on index drift or a missing rules paragraph, and exits 2
  (cannot-evaluate) when the live file has no column-0 block heading instead of
  crashing; (c) new `--refresh-header` mode rebuilds ONLY the header. The live
  header is restored byte-identical to the pre-apply header except the two added
  index entries (git diff: one 2-line hunk in the index, rules hunk unchanged).
- NEGATIVE CONTROL MEASURED: deleting one index line makes `--verify` FAIL
  (exit 1) with `header index drift: 22 live vs 23 archive entries` and the exact
  repair command; restoring the file returns PASS.
- TWO IN-PLACE ARCHIVE CORRECTIONS (both from the AC5 close-out edits, both
  recorded in the manifest's new `corrections` array and printed by `--verify`):
  T-GPLR-37's preserved sentence had been relabelled V1 (it is V2,
  register-once/churn-free); T-GPLR-52's status carried the garbled sentence
  "Members (T-A) moved to the archive on the next resplit" (now states that the
  block moved in the 2026-09-21 resplit). The frozen
  `archive_body_sha256_lf` was re-recorded in the same change
  (c422b10e... -> 6b60b61f...); no block was moved, reordered or resized.
- GUARDS: a `[plan-split]` step is wired into `tools/run-tests.ps1` beside
  `[doc-size]` and `[t53-a1]` for every host-only suite (exit 2 counts as a
  suite failure - a plan that cannot be checked is not a soft pass). Observed:
  `[doc-size]` PASS, `[plan-split]` PASS, `[t53-a1]` SELFTEST PASS,
  **37/37 CTest**, suite `host` SUCCESS. Doc-size WARN stands for the archive
  itself (320.1 KiB, 68.3% of the 468.8 KiB FAIL band) - watch it, per
  AGENT_ENVIRONMENT.md, before the next split.
- LIVE BACKLOG NOW: T-GPLR-53 (in-progress; A1 + A2 done) only.
  T-GPLR-54 CLOSED 2026-09-22 on the owner's arm-rule decision (see top
  entry). Unowned residue corrected here: the recon `generation` off-by-one
  join was FIXED by T-GPLR-53 A2; "u4096 TG output truncation" and
  "pw-vs-non-pw last-chunk route difference" were both RESOLVED on
  2026-09-21; the +12 x2 recon-join samples on `gplr50-ctl-4k-f16` were
  RESOLVED 2026-09-21 (checker pairing artifact + a benign label artifact,
  see that entry). Nothing unowned remains in the correctness ledger except
  the recon emitter's cosmetic ubatch label (deferred by T-GPLR-53's A2
  addendum).

## 2026-09-21: T-GPLR-53 A1 EXECUTED (host-only, no GPU) - F1-F6 reproduced as CHECKED expectations; the pass-1 gap is the embed step at every shape and BOTH sources

- Tool `tools/analyze-gplr53-gap.py` (`--selftest` 8/8 with the negative controls
  firing, `--placement`, `--json`), table `build/llama-cuda/gplr53-a1-gap-table.txt`,
  14 traces, tool exit 0. Exit-2 refusals verified on real inputs (pre-ledger
  `gplr35-slot3`; a missing tag): "cannot evaluate" is never a pass.
- F6 EXACT: |loop + embed + output| <= 0.86 ms on all 11 traces whose ledger
  carries `loop_us`; the four `gplr37c-*` traces predate the field, so the tool
  SKIPS the check there - a refinement of the drafting wording ("four traces"),
  not a failure, and the skip is fixture-covered.
- F1: embed share of the pass1->pass2 gap excess 0.9915-1.0014 on current
  binaries (a2-200k 0.9969, a2-50k 0.9974, a3-mmap 1.0014, gplr50-200k-on
  0.9958, off 0.9974, s2-2slot 0.9915, 4k-mmap 0.9998); 0.971/1.026 on the
  T-37-era ron pair (T-44-era patch/slot work still inside `rest`). The worded
  claim is tightened to "the gap excess IS the embed step to >= 95%".
- F3 pinned CONFIRMED and sharpened: embed share 0.0015 (a3-pinned), 0.0247 /
  -0.0008 (gplr37c-pin s2/s3), 0.0004 (4k pinned); slot+preload carry +1710.0 ms
  of a3-pinned's +1710.8 ms and +3280.8 ms of the 4k pinned +3215.7. BOTH sources
  pay a ~1.7-3.3 s per-run first-pass one-off - the bucket differs, not the cost.
- F1b: rest drift <= 5.2% on every mmap trace (a2-200k 1.7%, a2-50k 4.7%).
- H4 (host-IO starvation) DISFAVORED by its own pre-registered test (host buckets
  flat, the excess isolated to the embed step's sync-bounded span) and now
  enforced as an expectation; H1/H2/H3 stay open - the decay needs A2's
  `embed_chunk`, the page-vs-submission split needs A4's modality arms. H2's
  host-inflation prediction is already disconfirmed (setup flat to 0.77 ms across
  391 embed launches).
- F7 re-derived from the code by `--placement` (main.cpp:3559/3567/3575,
  llama-layer-streaming.cpp:6676/7059): the embed step reads no host-mapped model
  page, so the delayed completion of the 18.5 GB registered span remains the
  candidate.
- gplr36-remeasure-200k stays CONFOUNDED (pass 2 = the T-46/T-47 regression,
  wall 206 s); its 1.087 embed share is NOT an F1 falsification.
- The selftest is wired into `tools/run-tests.ps1` for every host-only suite,
  beside the doc-size guard. Next: A2 (recon-gated `embed_chunk`), then A3/A4 in
  one owner-invoked mutex hold.

## 2026-09-21: T-GPLR-53 DRAFTED - the sweep's pass-1 "unattributed" gap is attributed host-only (it is the EMBED STEP; the T-36 ledger double-books it), and the mechanism probe is pre-registered with the decay discriminator

Host-only re-decomposition of the recorded traces (no GPU, no code change),
drafted into the plan as T-GPLR-53:

- The sweep's item 5 (wall - summed layer compute - summed layer wait = 2.35 s
  at 50k pass 1, 2.83-2.88 s at 200k pass 1, ~0.16/0.64 s later) is the EMBED
  STEP, not a missing bucket: `host_ledger.embed_us` 2246.3 -> 62.5 (50k a2),
  2482.8 -> 249.2 (200k a2), 2438.2 -> 251.0 (200k a3-mmap), 1774.1 -> 5.6
  (4k gplr51-4k-mmap) accounts for 99.7-100% of the pass-1 excess at every mmap
  shape; the leftover `rest` is pass-invariant (200k 372.0/367.4/373.6).
- The excess is inside the embed graphs' SYNC-BOUNDED execution, not host prep:
  the ledger's setup/patch/graph_call/slot buckets are flat pass 1 vs 2
  (193.75/192.98, 18.78/18.53, 16.34/15.89, 70.37/70.35 at 200k a2) while
  ledger-compute minus phase-compute (the embed step's own launch+execute+sync)
  is 2498.7/265.0; pass_begin -> first copy_submit is 2.60 s vs 0.37 s
  (a2-200k) and 2.56 vs 0.37 (a3-mmap).
- Source contrast, adjacent same-binary arms (a3-mmap/a3-pinned, 200k/3 slots):
  pinned has NO embed excess (251.1/248.6) and pays its first-pass cost in the
  copy path instead (slot 1827.8/322.4, preload 327.8/123.2, host_submit p50
  110.0/42.6 ms per bank; the mmap arm's is 0.124/0.100). Per-run excess over
  pass 2: mmap +2184 ms, pinned +1711 ms - both sources pay a one-off of the
  same order, in different buckets (pinned 4k pays ~3.0 s the same way: slot
  3667.4/581.9).
- Per-run and shape-flat in chunk count (4k 1.77 s / 50k 2.18 / 200k
  2.19-2.23), re-paid by every mmap arm including ones that ran minutes after
  another mmap arm had read the whole file. Registration is deferred:
  mmap_register_init reports 18,500,517,888 B in 2345 us pre-timed, the
  file-backed working set moves 2.10 -> 20.43 GB during pass 1 (the pinned arm
  the same, via its memcpy) and is flat afterwards, while no DMA stalls
  (recon_attr copy_gpu_ms p50 31.5 / max 36.3 ms in both passes).
- INSTRUMENT CORRECTION: the T-36 ledger's other_us/loop_us are algebraically
  inclusive of embed+output, because the per-launch buckets are accumulated by
  the embed and output steps too while sweep_us starts after the embed step.
  Measured `loop_us + embed_us + output_us` = +0.10/+0.16/+0.13 ms (50k a2
  p1/p2/p3) and +0.73/+0.73/+0.70 ms (200k a2) - an exact identity. T-36's
  "closes exactly" reading was made while route telemetry's +4.5 s `other`
  masked it; since T-39/T-45 the same algebra reads other ~= -embed.
- CODE FACT that removes the obvious page-in story: the runner loads with
  n_gpu_layers = -1 and overrides only the pageable class to the CPU buffer
  (tools/layer-streaming/main.cpp:9-11, 3559-3578), so token_embd/output are
  CUDA-resident and the embed step reads no host-mapped model page; the delayed
  completion of the 18.5 GB REGISTERED SPAN is what is left as the candidate.
- T-GPLR-53 drafted with pre-registered H1 (front-loaded deferred-registration
  completion) / H2 (background pinning contention, smooth decay) / H3
  (submission-side one-time driver state, chunk-0 spike) / H4 (host-IO
  starvation, already disfavored by the flat host buckets), arms A1 (host-only
  gap tool `tools/analyze-gplr53-gap.py`, free; EXECUTED 2026-09-21, see the entry
  above) / A2 (a new `--pp-recon`-gated
  per-chunk `embed_chunk` event, host-validated first) / A3 (200k+50k
  `--pp-repeat 3` decay arm with a rho/share decision rule recorded before the
  run) / A4 (4k pre-warm modality discriminator: CPU touch vs dummy GPU launch,
  plus the payoff measurement). Charter note: at 4k the excess is 1.77 s of a
  3.55 s pass-1 wall, and a real single-prompt prefill IS pass 1.

## 2026-09-21: Anomaly sweep, host-only on existing traces: 4 of 5 explained, 1 open unknown (pass-1 exposed gap); recon gen off-by-one logged as a hygiene task

Sweep over gplr52-a3-mmap/pinned, gplr52-a2-50k/200k, gplr50-200k-on (no GPU
time). Findings, each pinned to numbers:

1. Pinned-arm 1485.6 ms GPU window (REAL, benign): 2 of 83 recon windows
   exceed 100 ms (layer 36 = 1485.6, layer 0 = 140.1); mmap max is 36.3 ms;
   both arms p50 31.5 ms. One window, one layer, one arm - likely queueing
   behind a compute kernel inside the bracketed window. Not a source
   signature; leave alone unless it reproduces.
2. recon 85/83 windows vs 80 PP copies: integrity OK. Every PP copy has
   exactly one window (zero duplicates); the remainder are TG-phase brackets
   plus one unbracketed first PP copy.
3. Layer-0 sub-100 ms host waits (mmap arm only - pinned arm min is 123.2 ms):
   31.4/33.7 ms == copy_gpu_ms 31.5, because layer 0 stages while the compute
   stream is still empty at the pass boundary. Not an anomaly: the H1 law
   (wait = queued compute + DMA) seen with zero queue.
4. HYGIENE TASK (new): recon_attr.generation and copy_complete.generation
   label the SAME physical copy off by exactly one - join exact=0, join at
   gen+1 = 79/85, 118/125, 79/87 across the three traces. Harmless for
   medians, silently breaks any cross-event join. Fix: align one side or add
   a shared copy_id; until then every cross-event join must pin the offset.
5. OPEN UNKNOWN - pass-1 exposed gap: phase_summary unaccounted (wall -
   compute - wait) is 2.35 s at 50k pass-1 (0.16 s later) and 2.88/2.13 s at
   200k pass-1 (0.64 s later). Closes the tok/s jump arithmetic exactly
   (50k 2419->2740 = +13.3%; 200k ~+1.1-1.7%). Both staging and compute are
   flat across passes, so it is neither; same order as the 4k first-touch
   submit transient (119->28 ms, A1). Candidate: first-touch
   registration/page-warming of the staging path. Draft T-GPLR-53 with the
   decay-with-chunk-index discriminator (host-only first). [SUPERSEDED
   2026-09-21 by the T-GPLR-53 drafting recon above: the gap is the EMBED STEP
   (`host_ledger.embed_us`, 99.7-100% of the excess at every mmap shape) and
   the T-36 ledger double-books it, so it was never an unnamed bucket; the open
   part is only the mechanism of the embed step's pass-1 sync-bounded excess,
   which T-GPLR-53 pre-registers. The "neither staging nor compute" reading
   holds for the LAYER SWEEP - the excess is before it.]

Honest refinements vs the sweep as first reported: the pinned arm shows NO
sub-100 ms host wait (layer-0 min 123.2 ms), so the collapse-to-DMA
signature is observed only in the mmap arm; and the off-by-one join was
re-verified exact=0 on all three traces, not just one.

## 2026-09-21: T-GPLR-52 AC3 - true-DMA timing was possible all along (recon copy_gpu_ms); DMA = 1% of host cost

- copy_complete semantics documented in AGENT_ENVIRONMENT.md: host-side event
  wait (submit + sync), phase-split discipline, staging_source assertion.
- The "true-DMA field or impossibility" either/or resolved: T-GPLR-35's recon
  (--pp-recon 1, default in recorded arms) already emits copy_gpu_ms from
  timing-enabled CUDA events on the DMA's own stream.
- Measured from the existing A3 traces (no new GPU time): DMA p50 31.5 ms
  (~14.6 GB/s), source-identical, heavy-layer 36.3 ms = byte ratio; DMA is
  1.0-1.1% of the 2984-3063 ms host-observed cost - ~99% is queueing behind
  compute (H1 corroborated from the GPU side; T-51's "copy tail" was
  host-wait, not transfer).
- Remaining T-52 item: AC5 closure (RESULT block, Status, commit hashes,
  index resplit).

## 2026-09-21: T-GPLR-52 A3 - H3 falsified with in-trace proof: pinned WAS engaged, no fallback ever existed (AC4 discharged)

- run_start now carries ENGINE-DERIVED staging_source + staging_source_match
  (nested 23e5d1b6e); both validators fail a mismatch/missing pair, legacy
  traces stay admissible. Fixture-proven both ways before the GPU arms.
- 200k A/B, same binary, clean gates: mmap p50 2984.4 ms / ready_wait 3.67 s
  vs pinned p50 3062.7 ms / ready_wait 3.61 s, both arms
  engine_staging_source match=1. H3's collapse prediction (pinned <= 0.5x
  mmap) FALSIFIED at parity 1.03x.
- F5 resolved: the staging path has no silent fallback by construction; the
  T-51 A2 addendum needs no correction. The mmap banner in pinned-arm logs is
  the model loader's message, orthogonal to staging.
- Corroboration: the new binary reproduces baseline numbers exactly (p50
  ~2984-3003 ms, ready_wait 3.67 s) - the assertion is performance-neutral.
- Remaining T-52 items: AC3 (document copy_complete's host-wait semantics in
  AGENT_ENVIRONMENT.md, or state why a true-DMA field is not measurable with
  the current backend API) and AC5 closure.

## 2026-09-21: T-GPLR-52 A2 - H2 (file-cache eviction) FALSIFIED as a mechanism; the 50k falsifier is the discriminating datum

- --pp-repeat 3 arms, baseline binary (hash-matches A1), clean GPU gates,
  mmap-registered explicit at BOTH shapes (the recorded 50k baseline ran
  pinned, possibly never engaged per F5): 200k pass p50 3002.9/2996.2/3007.7 ms
  (flat, tok/s 1458/1483/1480); 50k pass p50 771.2/758.5/760.3 ms (flat, tok/s
  2420/2741/2738).
- The 200k confirmation band is NON-DISCRIMINATING (H1 predicts flat there
  too - per-pass compute is flat and r(wait, compute) = +0.998); the 50k
  survivor prediction (pass-3 <= 0.85 x pass-2) is FALSIFIED. Combined with
  A1's first-touch re-explanation of 4k, H2 retains no discriminating
  evidence at any shape: eviction is not the mechanism behind per-copy cost;
  queued-behind-compute (H1) is.
- Scope note: A2 did NOT re-test copy source (both arms mmap); the
  pinned-engagement question (H3/AC4) remains A3's, owner-invoked.

## 2026-09-21: T-GPLR-52 A1 (host-only) - H1 CONFIRMED: the copy wait IS overlapped compute at long context

- The contract's literal metric: r(copy wait, overlapped compute) = +0.998
  at 200k (wait ~= 1.03x overlapped compute after correcting the stride-10
  sampling of layer_compute rows), +0.78 at 50k, +0.27 at 4k - where 4k
  pass-1 is a first-touch transient (wait decays with pass position,
  r = -0.78; submit component 119 -> 28 ms pass-over-pass).
- Concurrent copies are NOT the driver (r ~ 0 at 200k, median 2 in flight);
  submit/sync split corroborates (200k waits are 100% sync = queued-behind-
  compute; 4k has a material first-touch submit component).
- H2 (file-cache eviction) REFUTED for steady state: no position drift at
  200k/50k, pinned == mmap, only 4k pass-1 shows the predicted decay.
- Consequence: the 163/943/3003 ms scaling of identical 460 MB banks is
  the copy engine queueing behind compute - a direct consequence of the
  layer-major overlap design, NOT a staging defect. It is invisible in
  wall time while compute keeps the GPU busy (the 97%-hidden result).
- AC1 discharged: analyzer audit clean (8/8 phase-filtered);
  run-gplr51-tail-arm.ps1 quantile block fixed to phase-0 only.
- Remaining for T-52: AC3 (copy_complete semantics doc + DMA-time question)
  and AC4 (H3 pinned-engagement proof, arm A3). No engine change.

## 2026-09-21: T-GPLR-51 CLOSED - 3 slots measured-optimal (S2), default combo confirmed, plan re-split

- S2 (2 slots - the leg A1-A4 never measured; current binary, hash-matched,
  gpu_pre/post_run clean, 2-slot shape proven in-trace by distinct-slot
  census): exposed ready_wait 6.08 s = 1.66x the 3-slot 3.67 s; tok/s
  1456.9/1476.2 in band. With A3 (4 slots: copies inflate ~4x, tail max
  12.31 s), 3 slots is the saturation optimum in BOTH directions.
- Owner default CONFIRMED on every leg, current binary: layer-slots 3 +
  pp-async-feed 1 + copy-source mmap-registered (slots: S2+A3 bracket;
  async-feed: A4 neutral at 200k, documented hiding mechanism where slots
  are scarce; mmap: A2 equal exposure, 2.4x faster typical copies, 2x at 4k).
- Copy-tail question stays CLOSED as structural; framing correction owed to
  T-GPLR-52 stands: copy p50/max are host-observed waits, not DMA times.
- Plan index updated by the first RESPLIT: T-GPLR-39, T-GPLR-50, T-GPLR-51
  blocks moved verbatim to the archive (tools/resplit-gplr-plan.py, manifest
  docs/plans/resplit-manifest-2026-09-21.json, --verify PASS). Live plan now
  holds T-GPLR-37 (in-progress) and T-GPLR-52 (draft) only.

## 2026-09-21: Copy-tail follow-up - "40 vs 160" premise CORRECTED; the real anomaly is per-copy cost scaling with prompt length

Re-decomposition of copy_complete across all recent traces (host-only, no GPU):

- Staging granularity is shape-invariant: 2 whole-bank copies/layer/pass in
  every trace (200k/144k/50k/4k/512). The "fractional copies per layer" note in
  the T-GPLR-51 drafting session lumped phase-1 (TG) events into phase-0 (PP).
- Per-copy host cost scales with prompt length for IDENTICAL bytes: median PP
  copy 163 ms (4k) / 943 ms (50k) / 3003 ms (200k). 4k pass-2 halves vs pass-1;
  200k is pass-invariant. Heavy-layer ratio 1.15-1.16 at 200k, ~1.0 at 4k.
- copy_complete is a HOST event wait (enqueue + sync), not DMA time; the
  T-GPLR-51 "copy tail" statistics are host-observed cost, not transfer time.
  The exposed-wait (layer_ready_wait) numbers from A2/A3/A4 remain valid.
- A2's pinned arm shows mmap-identical costs and its log retains the mmap
  banner; pinned-active is unproven in-trace (the separate --pp-set-pp-pool
  machinery requires --copy-source pinned and was not passed). If pinned fell
  back to the registered pool, "source-independent tail" is unfounded.

T-GPLR-52 drafted with pre-registered H1 (queue-wait behind compute) /
H2 (file-cache eviction, ~73 MB/s effective read rate) / H3 (pinned never
engaged) and arms A1 (host-only overlap correlation) / A2 (--pp-repeat 3
pass-index table) / A3 (proven-active pinned with in-trace assertion). No
engine change until a mechanism is named.

## 2026-09-21: T-GPLR-51 A3/A4 verdict - copy-tail question CLOSED (structural); async-feed redundant at 200k; 4 slots worse

Arms (same binary, clean-gated): A3 (--layer-slots 4) FALSIFIED P2 - pass-1
per-copy times inflated ~4x (p50 3.0 -> 11.8 s, bucket 249 -> 370 s), tail
max grew to 12.31 s, exposed wait flat (3.57 s): 3 slots is the saturation
point. A4 (--pp-async-feed 0) FALSIFIED P3 - byte-identical to baseline
(ready_wait 3.67 s, p50 31.5 ms, max 10.80 s, tok/s equal): the T-25
async-feed worker contributes ~nothing to hiding at 200k; the slot window +
recon pipeline does it all.

AC3 mechanism, named and corroborated across 4 arms: the ~11 s worst-case
copy is per-layer-file READ throughput on the heaviest layers (24-39: 522 MB
vs 453 MB elsewhere; heavy/light mean-copy ratio 1.15-1.16 = the byte
ratio). It is source-independent (identical under pinned, which never reads
the file mid-pass), slot-count-independent in direction, and af-independent.
Not a staging defect; a property of the shape's file layout, hidden at 97%.
AC2's < 1% exposed-wait target is unreachable by staging configuration
(floor ~2.7%, honest negative). No engine change justified; A5 (cold cache)
remains optional owner-invoked. The charter-relevant bottom line: PP at 200k
is compute-bound on a quiet GPU, and the staging pipeline is already at its
saturation point - further PP wall gains must come from compute or
residency policy, not staging.

## 2026-09-21: T-GPLR-51 A2 verdict - copy-source stays mmap at 200k; the 10.9 s copy tail exists in BOTH sources (structural, not page-fault)

Arm A2 (gplr51-a2-pinned, 200k f16, 3 slots, repeat 2, tg 3, shape verbatim
vs the A1 baseline, only --copy-source pinned changed): PASS verdict,
check-telemetry clean, gpu_pre_run/post_run both clean. Exposure is EQUAL
(ready_wait 3.62 s pinned vs 3.67 s mmap; ~2.7% of wall) - P1 falsified, no
inversion, the copy-source default does not reopen. tok/s 1477/1456 vs band
1464-1485 +/- 3% (P4 held; compute-bound again). mmap keeps its typical-copy
advantage (p50 31.5 vs 75.8 ms).

The sharp finding: the ~10.9 s worst-case copy exists in BOTH sources (10.82
mmap / 10.97 pinned). Pinned never reads the file during the pass, so the
tail is not the mmap page-fault path - it is structural (staging queueing /
slot contention / PCIe congestion with parallel copies). This redirects
T-GPLR-51: A3 (4 slots) is the prime tail probe; the page-fault mechanism
remains only the pass-1 whole-bank-copy pattern (40 copies at ~74 MB/s).
AC2 (< 1% exposed wait) unmet by both sources today. A3/A4 remain
owner-invoked.

## 2026-09-21: Transfer-bucket decomposition at 200k + the 4k copy-source discriminator (T-GPLR-51 draft evidence)

Attacking "the 249 s transfer bucket" required decomposing it first, from the
T-GPLR-50 session traces (host-only, no new GPU time):

* The bucket is ~97% HIDDEN: wall 135.5 s ~= compute 133.3 s; exposed
  layer_ready_wait totals 3.67 s (2.7%) at 200k (3.61 s at 50k, 0.71 s at
  b1 q8). The bucket sums OVERLAPPING per-event copy durations (40 parallel
  copies in flight; per-event mean 6.2 s), so it is not serial wall time.
  On a quiet GPU every shape is compute-bound (200k 1476 tok/s, 144k q8
  1961, 50k 2614) - the transfer lever buys wall time only where hiding
  breaks (the tail, the cold file, fewer slots).
* Copy-tail pathology at 200k mmap-registered: per-copy p50 32 ms, max
  10.8 s; pass 1 runs 40 whole-bank copies (mean 6.2 s ~ 74 MB/s per 460 MB
  layer bank - page-fault-class throughput), pass 2 runs 160 smaller ones
  (p50 31 ms). The pass-1/pass-2 chunk-to-copy split differs by shape
  (b1 q8 u1024 shows the pass-1 pattern; 50k pinned shows the small-copy
  one) and is unexplained - recorded for T-GPLR-51 AC3.
* Same-binary copy-source discriminator at 4k (new probe arm
  gplr51-4k-mmap vs gplr50-4k-f16-default, same argv except --copy-source):
  mmap-registered beats pinned ~2x - staged 18.3 GB in 3.3 vs 11.7 s/pass,
  first-touch tok/s 1154 vs 708, warm 2582 vs 1824. At 200k the same-binary
  pair has never run (T-36's pinned number was a different binary); it is
  arm A2 of the draft.

T-GPLR-51 drafted in the plan: five arms (existing-mmap baseline, same-binary
pinned, 4 slots, async-feed ablation, optional cold-cache), predictions P1-P4
pre-registered, AC1-AC5 falsifiable. Honest target: exposed wait < 1% of wall
and a mechanism-explained copy tail - NOT the bucket total, which the
3-slot pipeline already hides.

## 2026-09-21: Clean PP/TG reference numbers for the current binary (T-GPLR-50 session)

All numbers from today's sentineled, GPU-gate-era traces (binary exe
1B08B382ACA4694D / dll B540024042E8B1F2, nested 40bca9f44), measured on a
verified-quiet GPU during the T-GPLR-50 arms - the honest reference table for
the Qwen3.6-35B-A3B vehicle. Shape: layer-major, moe-expert-bank, 3 slots,
--pp-async-feed 1; PP tok/s is the phase_summary `tokens_per_s` of each pass;
TG is the 96-token greedy needle decode at 4k unless stated.

PP (prompt processing, tok/s by shape):

| Shape | KV / ubatch | pass-1 | pass-2 | trace |
| --- | --- | --- | --- | --- |
| 200k (charter) | f16 / u512 | 1476 | 1485 | gplr50-200k-off (this is the explicit-0 arm; the default-ON pair run: 1464 / 1480) |
| 144k (b1) | q8 / u1024 | 1894 | 1961 | gplr50-b1-q8-default |
| 50k | f16 / u512 | 2374 | 2614 | gplr50-50k-f16-default |
| 4k | f16 / u512 | 708 first-touch | 1824 | gplr50-4k-f16-default |
| 512 (QA anchor) | q8 / u512 | 130 | 372 | gplr50-qa-512 |

Rebuild ON vs OFF at 200k: 1464/1480 (ON) vs 1476/1485 (OFF) tok/s - equal
within session noise (the AC3 pair result; wall ~270 s both).

TG (96-token greedy, 4k needle shape): 77.8-80.2 tok/s on the changed binary
across the four same-session arms (wall 1.20-1.23 s, compute 9.7 ms/token);
control binary 85.6 and recorded T-47/T-48 references 85.8-86.9. The
session-to-session spread is wider than any in-session difference, so the
honest TG reference is ~78-87 tok/s with no rebuild-state dependence. The
3-token TG smokes in the 50k/200k arms (0.5-0.8 tok/s) are transfer-dominated
and are NOT TG benchmarks.

Staging note (charter priority check): at 200k the transfer bucket is ~249 s
vs ~134 s compute - PP on this GPU remains staging-bound, not compute-bound.

Contamination rule applied when picking these: the gplr50-qa-512 trace's own
TG phase (0.8 tok/s, 55.7 s compute for 96 tokens) carries the deepseek
contention signature and is excluded; every number above comes from
pre-run-probe-clean sessions. Future reference numbers should carry
`gpu_pre_run: clean` in their verdicts (gpu-lock 001bf65).

## 2026-09-21: GPU-mutex wrapper hardened against non-conforming compute processes

Follow-up to the T-GPLR-50 contention re-attribution (deepseek_v4.exe holding
~16.5 GiB VRAM during mutex-held runs). `tools/with-gpu-lock.ps1` now probes
`nvidia-smi --query-compute-apps` AFTER acquiring the mutex: visible compute
processes get a 30 s teardown grace, then the wrapper refuses with exit 125
naming the process (`-AllowBusy` overrides, loudly). A post-run probe catches a
process that ARRIVED mid-run and annotates the verdict (`gpu_pre_run` /
`gpu_post_run` fields; `refused_busy` on the 125 path; `probe-unavailable`
proceeds unverified rather than blocking). Probe helpers
(`Get-GpuComputeApps`/`Format-GpuBusy`) live in `tools/arm-verdict.ps1` with a
stub point (`GPU_BUSY_PROBE_CMD`) so the gate is host-testable without a GPU.

Validation (all host-only, no CUDA): six branches through the REAL wrapper -
clean start, busy-refuse (exit 125), busy-allow, probe-unavailable,
busy-clears-in-grace (`build/gpu-gate-tests/run-gate-cases.ps1`, all PASS), and
the appeared-mid-run case caught by the post-run probe (`gpu_pre_run=clean`,
`gpu_post_run=1234 deepseek_v4.exe`) - plus one live run against real
nvidia-smi (`gategen-live`: clean/clean, PASS).

## 2026-09-21: T-GPLR-50 CLOSED - f16 rebuild default flipped ON; the +67.4 s 200k pass-2 excess RE-ATTRIBUTED to GPU contention

Shipped: --pp-prebuild-per-pass now defaults ON whenever --pp-repeat > 1 for
EVERY KV type (f16 included) - nested commit 40bca9f44. Announcement cites
T-GPLR-50; explicit 0/1 unchanged; --pp-repeat 1 stays rebuild-OFF. Validation:
AC1 identity (96 tg ids byte-identical to T-47 v1-on AND gplr48-a4), AC5 full
provenance matrix incl. the control INV1 (control shows 0/default on f16, no
T-50 marker), AC4 neutral at 50k f16 (-0.1%) and b1 q8 (-1.0%), AC2 pw+rebuild-ON
ANSWERS on the current tree (no echo; the escalation branch never fired), AC6
qa-smoke PASS with the negative control failing as designed, host suite 68/70
(= the T-48 baseline; the 2 fails are the documented --layer-slots 1 refusals).
Full evidence table in the plan's T-GPLR-50 RESULT block.

The AC3 finding that outlives the flip: in a single-binary A/B pair under one
mutex hold (explicit-0 vs default, remeasure shape verbatim, 200k/f16/tg 3),
BOTH states came back pass-2-neutral (excess -0.81 s OFF / -1.47 s ON, two-pass
wall ~270 s both) - the recorded +67.4 s (+48.9%) rebuild-OFF excess from
gplr36-remeasure-200k did not reproduce on the byte-equivalent engine path (the
pre-T-50 binary's explicit-0 resolves and executes identically to today's
explicit-0; the only difference is the session). Classification evidence:
(a) that remeasure's pass-2 shows uniformly-inflated layer_compute with normal
transfer ratios - the contention signature; (b) the same signature crashed two
ctest fixtures (teardown cudaFree OOM, 0xc0000409) while deepseek_v4.exe held
~16.5 GiB VRAM DURING this project's own mutex-held host suite - both fixtures
PASS on a free-GPU rerun (234.9 -> 19.1 s, 148.5 -> 11.3 s). Conclusion: the GPU
mutex Global\AI-GPU-EXCLUSIVE-v1 is cooperative and does not exclude
non-conforming projects; single-session 200k timings taken while another
project was active are contaminated. The 2026-09-21 remeasure entry's "rebuild
OFF costs +67.4 s" claim and T-GPLR-50's P3 wall prediction are WITHDRAWN as
confounded; the flip stands on P1/P2/P4 + the identity/provenance gates, and it
remains the correct posture (measured-neutral on a quiet GPU; reclaims the
excess automatically if a genuine rebuild-path regression ever reappears).

Also recorded: q8 default arms now announce "default since T-GPLR-50" although
the q8 VALUE dates to T-GPLR-48 (one announcement for both branches; cosmetic).
New harness: tools/run-gplr50-200k-pair.ps1 (one-mutex-hold A/B, P1/P3 gates).

## 2026-09-21: T-GPLR-50 DRAFTED — the survey answer to "highest-value remaining PP lever": flip the T-48 f16 rebuild default back to ON (owner decision pending)

Survey (plan + currentstate, post-T-39): the remaining named levers are
deep-submit (~0.9% of a 200k pass, ordering risk, recorded do-not-draft) and
the per-launch patch lever (closed, 26.1 ms = 0.013%). The one lever worth tens
of seconds per pass at the charter's 200k shape is the T-GPLR-48 f16 default
(rebuild OFF), which re-exposed the pass-2 compute excess that only T-47's
fresh-context rebuild eliminated: recorded ledger pairs — 200k f16 ON
(p1 133.85 s / p2 133.95 s compute, excess +0.10 s) vs OFF (p1 138.0 s / p2
205.4 s, excess +67.4 s, uniform ~1.5x per layer_compute); 50k neutral in both
states. The draft (plan block T-GPLR-50) flips the f16 branch of the
KV-conditional default to ON for repeat > 1, with pre-registered predictions
P1-P4, AC1-AC7, and a pre-flight echo-trigger rerun (AC2) because the T-49
session's pw+rebuild+echo observation predates T-49's slab fix and is genuinely
open on the current tree — if it reproduces, the task escalates instead of
flipping. Not implemented; nothing committed beyond this entry and the plan
block.

## 2026-09-21: BOTH PARKED ODDITIES RESOLVED FROM RECORDED TRACES (no new GPU arms)

Owner request: investigate (1) pw vs non-pw last-chunk routing difference and (2) the
u4096 "truncated answer" (262 vs 305 bytes). Both resolved entirely from existing
traces; the parked phrasings were both **wrong in their attribution**.

**Oddity 1 - RESOLVED: it was never pw-vs-nonpw, and it was a real bug that T-GPLR-47
already fixed at the root.** Re-analysis of every repeat-2 trace with route telemetry
(pass-1 vs pass-2 hash comparison, both passes in the same run):

| trace (binary era) | mode | kv | pass1v2 route diffs |
|---|---|---|---|
| gplr38-u512..u4096 (pre-T-43) | pw default | f16 | **0** at all sampled chunks |
| gplr45-4k/50k/200k (T-45 era) | pw | f16 | 40 (last chunk), 400, 1560 - every sampled chunk |
| gplr49-4k-pw (T-49 era) | pw explicit | f16 | **40/40 = last chunk only** |
| gplr49-4k-nopw (T-49 era) | nopw | f16 | 0 |
| gplr49-50k-pw (T-49 era) | pw explicit | f16 | **400/440 = all but chunk 0** |
| gplr49b-4k both (rebuild 0) | both | f16 | 0 in both |
| gplr48-a3-50k-pbp1 (T-48 session) | pw | f16 | 400 (changed AND control) |
| gplr48-a2-b1-q8-default | pw | q8 | 560/600 (all but chunk 0) |
| gplr47-v1-4k-**control** (same session, pre-T-47 binary) | pw | f16 | **40 (last chunk)** |
| gplr47-v1-4k-**changed** (T-47 binary) | pw | f16 | **0** |
| gplr47-v2-50k / v3-200k / v4-b1-q8 (T-47) | pw | f16/q8 | 0, 0, 0 |
| gplr39-C1/E1 (post-T-49) | pw | q8 | 0 at 4k |

Mechanism: pw + per-pass rebuild (ON) left a stale pointer/binding window in the
in-place rebuild path (pre-T-47 `build_patch_split` re-used accumulated contexts);
the first sampled chunks of pass 2 read slightly different probs. **The bug was REAL
in compute** (pw-changed and pw-control TG streams differ from nopw from token 0;
pw pass-2 chunk-7 differs deterministically and identically in both binaries) but
**NOT harmless to the graded answer after all** (corrected 2026-09-21 when the
T-GPLR-50 survey decoded the pw stream): the pw arm's TG stream is a
DOCUMENT-ECHO derailment - `gplr49-4k-pw-changed-dec.json` grades **FAIL 0/5**
(`The maintenance log for row ...` echo, no needle), not PASS as this entry
first claimed. The graded answer stays protected because (a) rebuild-OFF + pw
grades PASS 5/5 (`gplr49b-4k-pw-rebuild0`) and (b) nopw + rebuild-ON grades
PASS (T-47 v1-on, ids identical to the nopw default stream), so the echo
needs pw AND rebuild-ON TOGETHER (T-49b's attribution, now confirmed from the
other side) - and every post-T-47 default-era gate run is clean. The
T-GPLR-50 draft's AC2 re-runs this trigger on the current tree before any
f16 rebuild-default flip completes. T-GPLR-47's fresh-context
rebuild eliminated it: every post-T-47 trace is clean, including the within-session
T-47 v1 control-vs-changed pair, which is the cleanest proof available. The T-48
session's pbp1 arms are clean at 4k-default but anomalous at 50k/q8 because that
session ran a pre-T-47-era binary for the a3 matrix (commit 8ab3f1f00 landed after;
its pbp1 rows are the old engine). No open defect remains; the "pw vs non-pw"
framing was a red herring - nopw with rebuild ON was ALSO anomalous at 50k/200k
(T-45 era), it just sampled differently (4k samples chunks 0+7 only).

**Oddity 2 - RESOLVED: u4096 does NOT truncate the answer; it derails AFTER it.**
Decode of the u4096 greedy stream (`gplr38-u4096-dec.json`, decode-graded): the full
needle answer is present and intact - normalized answer
`imendimstartqx7741tr2918mk5501ld8834parisimend` contains QX-7741, TR-2918, MK-5501,
LD-8834, Paris, i.e. the semantic content is COMPLETE at u4096. What differs: after
the answer region (TG index 41 of 96), u4096 drifts into a template-token repeat loop
(`<|im_end|><|im_start|>...`) while u512/u1024/u2048 continue document echo. The 305
vs 262 byte difference is the STRIPPED-SPECIALS decode of that tail difference, not a
cut-off answer. The first 41 TG ids (the entire needle-answer span) are byte-identical
at every ubatch. **CORRECTED 2026-09-21 (host-only re-read of the four arm traces,
see the T-GPLR-54 draft block): the tail difference is not a "repeat loop" but ONE
bounded insertion.** u1024 and u2048 are id-identical to u512 for ALL 96 ids; the
u4096 stream equals u512 with exactly one contiguous 7-id insertion at index 41
(`248045, 248045, 248045, 248045, 248046, 198, 248045`) and is byte-identical to
u512 after realignment (`u4096[48:96] == u512[41:89]`) - one argmax flip at the
first decision after the template region both arms enter at index 35, followed by
exact re-synchronization. Classification: benign greedy tail drift under a 1-chunk prefill
(u4096 = 1 x 4096-token chunk vs 8/4/2 chunks elsewhere - a different graph shape,
not a bigger version of the same shape), no identity or semantic gate impacted.
Lifting the recorded "no arm at or above u4096 can be called output-equivalent"
caution: arms at u>=4096 may be called **answer-equivalent** (needle grading), not
greedy-id-equivalent; if id-equivalence at u4096 is ever needed, it needs its own
investigation (single-chunk shape vs multi-chunk shape).

**Nothing to implement; no code changed.** The only artifact change: the T-48
gotcha in AGENTS.md already carries the parity/rebuild lessons this confirms; the
u4096 equivalence caution is lifted here and supersedes the T-GPLR-38 closure text.
Evidence: `build/llama-cuda/gplr{38,45,47,48,49,49b}-*.jsonl` (routes + tg_tokens +
grade files), decode artifacts `gplr38-u{512,4096}-dec.json`, vendored commit
chronology via `git log` (T-47 = 4403c1097, 09-20 16:43).


## 2026-09-21: Per-launch patch lever RE-MEASURED at 200k -> CLOSE the idea; a NEW finding takes its place

One 200k arm on the current binary (`gplr36-remeasure-200k`, the recorded hostmem shape
with the tree's own defaults: copy mmap-registered (T-37 default, resolved
`default`), route telemetry OFF (T-39 default, resolved `default`), f16 KV rebuild
OFF (T-48 default, resolved `default`), af 1, 3 slots, repeat 2, recon 1).
Verdict PASS, checker OK, sentinels in `build/llama-cuda/verdicts/gplr36-remeasure-200k.*`.

**Per-launch patch lever: CLOSED.** Warm-pass `patch_us` = 26.1 ms = 0.013% of the
sweep (pass 1: 20.8 ms / 0.015%). The recorded pre-T-45 baseline was 3,402.3 ms/pass
(2.40%); T-45's zero-consumer filter already removed 99.4% of it, and nothing
worth drafting remains. The "shrank, needs re-measuring" item is settled: the lever
is gone, not merely shrunken.

**Deep-submit: the per-launch sync it targets now hides ~0.9% of a pass, not 6.5%.**
The remaining named host buckets at 200k (pass 2): slot 72.4 ms, other ~0 (negative
residual: the ledger's loop residue went negative because route telemetry no longer
runs), setup 371.9 ms, graph_call 30.9 ms, patch 26.1 ms. Its ordering risk in the
T-40/43 crash area now buys almost nothing. Recommendation recorded: do NOT draft.

**NEW measured finding (first rebuild-OFF 200k measurement ever):** pass-2 compute
excess is BACK under the T-48 f16 default: pass 1 compute 138.0 s -> pass 2 205.4 s
= **+67.4 s (+48.9%)**, and per-`layer_compute` ratios are ~1.5x UNIFORM across all
chunk positions (flat vs chunk index; pass-1 346 ms/ubatch -> pass-2 525 ms/ubatch).
Ladder for this shape: pre-T-43 single-prebuild +0.36 s; T-43 in-place rebuild
+59-62 s; T-47 fresh-context rebuild ON +0.10 s; T-48 f16 default (rebuild OFF)
+61.5/+61.3 s (T-45 arms, rebuild state equivalent) and +67.4 s today. The T-46
"rebuild is the trigger" conclusion and the "rebuild OFF is clean at 50k" result DO
NOT transfer to 200k: with rebuild OFF the excess is worse than with the in-place
rebuild. The T-47 mechanism (fresh-context rebuild) remains the only configuration
that eliminates the excess at this shape — but T-48's f16 default turned it OFF.
Escalated to the owner: the f16 KV-conditional default (T-48) costs ~+65 s per
pass-2 at 200k vs rebuild ON; a default re-decision (rebuild ON everywhere now that
T-47 fixed its cost) or a new investigation is an owner call, not a planner call.
Arm was run under the GPU mutex while another project held other windows; pass-2
wall 206.1 s reproduces the hostmem arm's 198.1 s slow-pass-2 signature, so this
is NOT the previously-suspected session noise.

Commits: none yet (evidence-only entry). Trace
`build/llama-cuda/gplr36-remeasure-200k.jsonl`, verdict
`build/llama-cuda/verdicts/gplr36-remeasure-200k.verdict.json`, harness
`tools/run-gplr36-remeasure-200k.ps1`.


## 2026-09-20: T-GPLR-39 CLOSED — route telemetry defaults OFF (owner decision)

The route event is a diagnostic that cost 1161.5 ms/pass at u2048 (100.8% of the
`other` ledger bucket); it now records only with explicit `--route-telemetry 1`.
The bare default resolves 0 with the self-describing announcement +
`route_telemetry`/`route_telemetry_source` run_start keys; the strict checker
rejects route events in a telemetry-off trace; the functional routing path is
untouched. All three arms PASS (default-off: zero route events; explicit-on:
160 events, parity clean; control: silent), 96 tg ids byte-identical across all
arms, 512 smoke rung 5/5 with the negative control failing as designed. Fixed
en route (T-37-owned, dense-only): mmap-register coalescing for page-touching
dense layer spans — dense CTest fixtures run again under the T-37 copy-source
default. Consumer contract: the route-parity gate needs an explicit
`--route-telemetry 1` run; exit 2 on a defaulted trace is by design.
Commits: nested f6d104c1f, outer 52e66c4 (harness), outer (this commit)
(plan + state). The AGENTS.md route-parity gotcha is updated.


## 2026-09-20: T-GPLR-47 CLOSED — the per-pass rebuild now swaps in a FRESH context per graph set (“swap-in-slot”), which removes the T-GPLR-46 defects at their root instead of only re-defaulting the flag. **The defect, root-caused:** the per-pass prebuild re-entered `begin_set` on an already-BUILT set and reset it IN PLACE — `res->reset()`, graph lists cleared, the rebuild written into the SAME ggml ctx — so the ctx pool accumulated every prior build’s tensor objects (measured T-GPLR-44 signature: patch_cache members 7 → 14 → 28) and the set’s act bindings survived only because the rebuild never re-allocated (the “address-stability accident”). **The mechanism now:** when the runner’s explicit `rebuild_replacement` flag is set (around pass-2+ prebuilds only), a re-entered BUILT set is replaced wholesale: a fresh 96 MiB no-alloc ctx + new `llm_graph_result` are constructed OUTSIDE the live set, the act tensors are REPLAYED onto the old set’s exact device regions (captured before teardown — never re-derived, because the cursors are frozen post-first-pass and the descriptor recompute is 2x wrong at bf16; asserted non-aliased and region-exact), the fresh state is move-assigned into the SAME vector slot (element addresses unchanged → every stored `graph_set*` valid by construction, closing the T-GPLR-44 stale-pointer hazard), and the old ctx/res are `ggml_free`+deleted (bounded per-set pool). First builds, handoff-invalidated TG sets and a refused replacement keep the old in-place path; act_bind telemetry stays silent on replacement (INV1). **The evidence, one session, seven arms + the smoke rung (all under the GPU mutex with started/verdict/done sentinels):** control = pinned post-T-GPLR-48 binary `bin/gplr47-control` (exe `A05738CE609618EF`, llama.dll `57E81B5664A7EC6D`), changed = `bin/gplr47-changed` (exe `77542EC1386641F5`, llama.dll `754D1AD5BFA13048`). (1) **The echo is gone with rebuild ON:** 4k/96 changed-ON ids are byte-identical to BOTH the same-session OFF arm AND the recorded cross-session `gplr46-4k-off`, needle-graded PASS 5/5 (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`); the control at the same flags emits the document-echo (ids start 760, decode FAIL) — the regression reproduced on the pinned binary, so the fix is attributable to this change. (2) **The pass-2 compute excess collapsed:** 50k +0.23 s (gate ≤ 0.5 s; was +3.72 s), 200k +0.10 s (was +59.2 s) with pass-2 wall within 1.0% of pass 1 — the growth law is gone at the charter shape. (3) **The T-GPLR-43 quantized-KV rescue survives the swap:** q8 B1 (143360 tokens, ubatch 1024) rebuild-ON runs both passes, exit 0, strict checker OK, route parity clean (1200 events, 0 singletons). (4) **Structural proof, AC5:** the changed arms’ `patch_cache` dumps hold a CONSTANT 6 members across 4 (4k) and 6 (50k/200k) rebuilds — 6 not 7 because T-GPLR-45’s zero-consumer filter removed the `inp_embd` orphan — while the control’s rebuild arms emit only two dumps total (the in-place path never re-walks), the old mechanism’s own signature. (5) **The decode-free pass-2 gate flipped, cleanly:** route parity is CLEAN on every changed rebuild-ON trace (0 singletons at 4k/50k/143k/200k) and DIVERGED on the control (80 of 160 at 4k) and the recorded `gplr46-50k-on` (800 of 880). (6) **No cost, no re-emission:** strict `--check-telemetry` OK everywhere; 4k warm-pass compute 1.20 s vs control 1.36 s / OFF 1.32 s (same session); total 4k wall 15.8 s ON vs 15.7 s OFF, so the rebuild overhead stays in the recorded 7.4 ms order — bounded, not directly instrumented (no dedicated rebuild-wall event exists; recorded as the task’s only residual). **Falsifiers: NONE fired.** CTest 115/117 with the same two disclosed pre-existing fixture failures (`--pp-async-feed` vs `--layer-slots 1`, identical on the pinned control). **Evidence:** plan T-GPLR-47 block (RESULT 2026-09-20, status done); verdicts `build/llama-cuda/verdicts/gplr47-*.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr47-*.jsonl`, smoke log `build/llama-cuda/gplr47-v5-qasmoke.log`, harnesses `tools/run-gplr47-arm.ps1` / `tools/run-gplr47-set.ps1` / `tools/run-gplr47-qasmoke.ps1`. **Open:** the pw-vs-non-pw last-chunk routing difference (explicitly not this task’s) still needs an owner if it ever matters; the plan file at ~268 KB is 52% of the doc-size fail band and will need the currentstate-style split eventually. **Ready front after this closure:** the trio is fully closed — T-GPLR-47, T-GPLR-48 and T-GPLR-49 are all done; next eligible work is whatever the Planner drafts next against the generic backlog.

## 2026-09-20: THE PLAN FILE WAS SPLIT — 18 closed task blocks moved VERBATIM out of
`docs/plans/generic-prefill-layer-residency-plan.md` into
`docs/plans/generic-prefill-layer-residency-archive.md`, leaving the live file with only the two in-flight blocks
(T-GPLR-37 in-progress, T-GPLR-39 draft) plus the index. **Why now:** the file was 274,413 B = 57.2% of the
doc-size FAIL band and every closure adds ~5 KB; the guard warned on each pass. The currentstate split
convention was followed exactly: verbatim moves, an index of archived blocks in the live file, a mechanical
`docs/plans/archive-manifest.json`, and a from-disk reconstruction check that reproduces the pre-split file
BYTE-FOR-BYTE (`python tools/split-gplr-plan.py --verify`; PASS vs the recorded worktree sha256
`f76c319ac3edbfc64116d2885512a8c0002b96df3695434f35b62769cac54966` = blob LF `00fa4b792cfe128ffe32124e2c8c293da3270d079e57eef30cde412d07aa68ee`, 270,157 B —
both hashes recorded because `core.autocrlf` makes them differ). **Two measured splitter defects the verifier caught before they could ship:**
(1) the archive header was built with one blank line too many, so stripping 6 lines cut into the first block;
(2) line-list JOIN semantics collapse a trailing blank line (`join(["x",""]) == "x
"`), which silently dropped
exactly one byte at the seam after the LAST live block — block tiling now cuts by string position so chunks
carry their own separator, and files are built by plain concatenation. Result: live file 55,984 B on disk, archive
221,619 B on disk (LF committed sizes differ), doc-size guard PASS with zero warnings. Convention recorded in AGENT_ENVIRONMENT.md
("Where current state lives"). Tool: `tools/split-gplr-plan.py` (dry-run / --apply / --verify).

## 2026-09-20: T-GPLR-37 CLOSED — mmap-registered staging is now the DEFAULT copy source (owner decision: "Promote to default — you always pay 17 GiB of locked RAM, always get the speedup where it exists"). **The change (nested 8ab3f1f00):** the runner's default `--copy-source` flips `pinned` -> `mmap-registered` (register every package source span ONCE at init, DMA straight from the model's own pinned file pages); a defaulted run announces itself on stderr and `run_start` gains `copy_source_source` (`explicit|default`) beside the resolved `copy_source`; the strict checker validates the new key; bare `--pp-pool` (a pinned staging structure) now refuses BEFORE model load with the one-flag fix. **Evidence, one session, five arms + the smoke rung under the GPU mutex:** control = pristine nested HEAD rebuilt into `bin/gplr37-control` AFTER the first control copy was found contaminated (the live Release build had already compiled the change — grep for the new key in the would-be control DLL proved it; the T-GPLR-49 control-preserving lesson now has a mechanical check), changed = `bin/gplr37-changed` (exe `A3EB9C3A32F1DA0E`, dll `89CE209871DA7DFB`). (1) **INV1:** the control at bare default resolves `pinned` with NO new key and NO announcement (C1, A3 PASS). (2) **The default measured (A1, 4k/3 slots):** resolved `mmap-registered`/`default`, announced, **40 init-once `mmap_register` events = 18,500,517,888 B (17.23 GiB) in 2.66 ms, ALL before the first `layer_compute`**, checker OK, parity clean (160 events, 0 singletons). (3) **Identity:** 96 greedy ids byte-identical across ALL FOUR 4k arms; decode PASS 5/5 (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`) in every arm. (4) **Timing (recorded, not gated, warm pass-2 walls, same session):** control pinned 2095.4 ms, changed pinned 2121.7 ms, changed default **1953.5 ms = -6.8%/-7.9%** — the honest current-tree win, smaller than addendum 2's -15.4% because the pinned path itself got faster with T-45/47/48/49; single pair, no replication. A1's pass-1 compute carries the expected first-touch page-in (3.71 s vs 1.84 s). (5) **A4 (b1 143k q8):** default resolves correctly on the reference long shape, 40/17.23 GiB pre-compute, checker OK, parity clean (1200 events), 154.4 s. (6) **Smoke rung (512, bare default):** runner 0, checker 0, needle PASS 5/5, negative control FAILs as designed. **Costs and limits:** the 17.23 GiB pinned price is unconditional under the default (the owner's recorded choice; ~20.5 GB process working set observed at rest), the crowded 200k shape stays wall-neutral per addendum 3 with no pressure event, and the -6.8% wall delta is one session, one pair. **Falsifiers: NONE fired.** CTest host 37/37, telemetry suite pass, doc-size guard PASS (the plan file now lives split: 55,984 B live + 221,619 B archive, `tools/split-gplr-plan.py --verify` PASS). **Evidence:** plan T-GPLR-37 block (RESULT 2026-09-20, status done); verdicts `build/llama-cuda/verdicts/gplr37-*.verdict.json`, traces `build/llama-cuda/gplr37-*.jsonl`, smoke `build/llama-cuda/gplr37-v2-qa-512.*`, binaries `build/llama-cuda/bin/gplr37-{control,changed}`. **Ready front:** the only remaining open task is T-GPLR-39 (route-telemetry fix class, owner decision (a)/(b)/(c) pending); the u4096 output-truncation and pw-vs-non-pw last-chunk observations remain ownerless.

## How this file is organised (split 2026-09-18)

This is the canonical current-state record: every completed task appends a dated entry at
the top, newest first. It was SPLIT on 2026-09-18 because it had grown to 557,144 B and
passed lean-ctx's per-file search cap, which made `ctx_search` skip it ENTIRELY while still
reporting "0 matches" (see `docs/AGENT_ENVIRONMENT.md`, addendum H, fact 1). Nothing was
rewritten: the older entries moved VERBATIM into contiguous archive shards under
`docs/currentstate/`, each well under the cap and therefore searchable again. The split is
recorded mechanically in `docs/currentstate/archive-manifest.json`: the original sha256,
every line range and its disposition, and the from-disk reconstruction check.

- **Recent entries** (2026-09-17 and 2026-09-18) stay in this file, below.
- **Everything older** (2026-08-18 .. 2026-09-16) is indexed below and lives in the shards.
  Shards: `2026-09-15..2026-09-16.md`, `2026-09-14.md`, `2026-09-12..2026-09-13.md`, `2026-08-20..2026-09-11.md`, `2026-08-18-and-earlier.md`.
- **Current state that is not an entry** -- "Known gaps / risks" and "Blocked / escalated" --
  stays at the bottom of this file.
- New entries are still added at the top of the recent block exactly as before; only the
  location of OLD entries changed. Prefer `rg` when searching across shards.

## Index of archived entries (2026-08-18 .. 2026-09-16)

Format: `<date> -- <shard>:<line> -- <entry>`, original order (newest first).

- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:9` -- 2026-09-16: T-GPLR-35 ADDENDUM 2 — DEPTH DOES NOT PAY AT LONG CONTEXT, AND THE BUD
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:11` -- 2026-09-16: T-GPLR-35 PIPELINE REGRESSION GATE ADDED — `tools/check-gplr35-pipelin
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:13` -- 2026-09-16: T-GPLR-35 ADDENDUM 1 — 3 SLOTS CONVERT THE SERIAL WAITS INTO OVERLAP:
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:15` -- 2026-09-16: T-GPLR-35 RECON RUN — the coupling is NOT the staging/compute stream t
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:17` -- 2026-09-16: T-GPLR-35 V1 IMPLEMENTED + BUILT (A/B armed, NOT yet run) — staging DM
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:19` -- 2026-09-16: T-GPLR-34 MID-KERNEL PROBE — WDDM cross-engine serialization FALSIFIED
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:20` -- 2026-09-16: SUBMIT_BATCH GAP CLOSED — the field now reports the ACTUALLY-TAKEN bra
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:22` -- 2026-09-16: T-GPLR-33 WHY-ANALYSIS — the decisive pair: warm GPU-side ready_latenc
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:24` -- 2026-09-16: T-GPLR-33 DONE (A/B RUN) — one-submission-per-bank FALSIFIED: windows
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:63` -- 2026-09-16: T-GPLR-32 FOLLOW-UP RESEARCH — the ~82 ms window decomposes as 3 membe
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:116` -- 2026-09-16: T-GPLR-32 DONE (A/B RUN) — serialize-the-worker-DMAs FALSIFIED as a wa
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:160` -- 2026-09-16: WDDM DMA-INFLATION RESEARCH PASS — asyncEngineCount=5 measured on this
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:200` -- 2026-09-16: BASELINE COMPARISON (out/llamacpp-baseline.log -- a local artifact und
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:233` -- 2026-09-16: CROWDED-VRAM CHARTER RUNG MEASURED — 200k-token PP, F16 KV, the af=1+p
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:263` -- 2026-09-16: GOAL CHARTER RECORDED (owner decision) — PP speed first, VRAM second,
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:277` -- 2026-09-16: DEFAULT PROFILE v3 GATED — af=1 + pinned defaults passed the qa-smoke
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:313` -- 2026-09-16: T-GPLR-31 DONE — locked-RAM probe: the driver grants 44.72 GiB of pinn
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:337` -- 2026-09-16: T-GPLR-30 DONE — partial persistent pinned pool (--pp-pool N[,fixed/lr
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:391` -- 2026-09-16: WAIT SIGNATURE IDENTIFIED — it is the per-layer deficit max(0, effecti
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:422` -- 2026-09-16: PASS-2 ANOMALY REPLICATED AND RESOLVED — two separate observations, op
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:450` -- 2026-09-16: T-GPLR-29 ADDENDUM — af overlap A/B with pinned: overlap converts ONLY
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:478` -- 2026-09-16: T-GPLR-29 DONE — pinned-pool probe + end-to-end pinned arm: DMA ceilin
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:523` -- 2026-09-16: ONLINE RESEARCH PASS — first-touch + staging cost layers identified fr
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:597` -- 2026-09-16: T-GPLR-28 DONE — in-process --pp-repeat 3 proves first-touch is pay-on
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:651` -- 2026-09-15: 4k LAYER-MAJOR PP MEASURED — 2.3x PP-wall gain vs fresh adjacent batch
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:715` -- 2026-09-15: DEFAULT PP SCHEDULE VERIFIED as layer-major (T-GPLR-12, no code flip n
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:759` -- 2026-09-15: AGENTS.md governance updated — MoE-Qwen focus is a measurement choice
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:798` -- 2026-09-15: Pending owner decisions (item 4 gate-eligibility, item 6 second-archit
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:821` -- 2026-09-15: T-GPLR-25 residue FIXED — async_feed_submit wired to the worker (batch
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:879` -- 2026-09-15: T-GPLR-25 COMPLETED: 4k pair run — af delta +3.0% is AT the noise floo
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:931` -- 2026-09-15: T-GPLR-25 paired A/B, 512 rung COMPLETE: NO af delta at 1 chunk/layer;
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:971` -- 2026-09-15: OWNER DECISION IMPLEMENTED — default --layer-slots 2, auto-3 on fit, l
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:1069` -- 2026-09-15: documented online research pass on H-C/H-D (no GPU runs, no code chang
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:1103` -- 2026-09-15 (latest): H-A fixed and proven — row-set TG phase_summary carries trans
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:9` -- 2026-09-14 (latest): full-length validation sweep 4k/50k/200k over the pruned tree
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:76` -- 2026-09-14: DECISION BRIEF (item 6, owner call) — which second architecture should
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:228` -- 2026-09-14: T-GPLR-22 PRE-TIMED ABORT — every abort in that family now leaves a tr
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:327` -- 2026-09-14: DECISION BRIEF (item 4, owner call) — should gate eligibility BIND pro
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:462` -- 2026-09-14: T-GPLR-22 PRE-SINK — a refusal raised before `ls.init` now leaves a re
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:533` -- 2026-09-14: Mission item 5 — the three opt-in TG modes measured at 4k and 50k; run
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:640` -- 2026-09-14: T-GPLR-22 — a pre-work refusal is now a structured, recoverable record
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:738` -- 2026-09-14: the non-echoing re-probe RAN — K=4 does NOT clear gate 9 on it (0.5355
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:843` -- 2026-09-14: F3/H8 implemented — TG row-set `layer_compute` carries real per-part t
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:913` -- 2026-09-14: T-GPLR-07 ladder rungs 4k and 50k PASS (owner-invoked, one at a time)
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:987` -- 2026-09-14 (later): the TOTAL residency cap is now provable from JSONL — `byte_acc
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1083` -- 2026-09-14: `--residency-budget-mib` is now a TOTAL residency cap — the row cache
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1190` -- 2026-09-14: the capacity preflight and the row-cache fit now share ONE VRAM budget
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1266` -- 2026-09-14: the per-layer row cache's DEFAULT depth is now fitted to the device —
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1355` -- 2026-09-14: K=4 set as the per-layer row cache's provisional DEFAULT depth (owner-
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:9` -- 2026-09-13: T-GPLR-19 deferred rungs RUN at K=2, K=4, K=6, K=8 — every rung passes
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:105` -- 2026-09-13: Vendored-fork commit convention DECIDED and applied — the vendored bac
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:200` -- 2026-09-13: T-GPLR-19 DONE — the four-arm K=1 A/B gate RAN and all six acceptance
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:301` -- 2026-09-13: Retention policy RESOLVED — the implementation moved to the row-capaci
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:402` -- 2026-09-13: T-GPLR-19 drafted (F2-M1 per-layer row residency at K=1) and its A/B d
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:438` -- 2026-09-13: The (b)-versus-K curve published as a standalone artifact — `docs/plot
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:562` -- 2026-09-13: T-GPLR-15 CLOSED DONE-falsified-by-evidence — the escalated retention
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:612` -- 2026-09-13: Closing-hash and tracked-status bookkeeping after ba1f31e
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:641` -- 2026-09-13: T-GPLR-18 pinned row-set staging CLOSED DONE-falsified-by-evidence — t
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:713` -- 2026-09-13: T-GPLR-15 Phase A probe RUN — the gate FAILS inside the SDD's referenc
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:788` -- 2026-09-13: T8 oracle retention ceiling added to the analyzer — on the ladder samp
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:822` -- 2026-09-13: Both remaining untracked docs brought under version control — the mode
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:869` -- 2026-09-13: T-GPLR-15 Phase A analyzer implemented — first real values of the SDD
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:941` -- 2026-09-13: Rowset telemetry schema draft brought into version control (tracked, n
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:975` -- 2026-09-13: Gate 9 decision rule now single-owned by the SDD (T-GPLR-15 metric cor
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1017` -- 2026-09-13: SDD amended with the F2 per-layer row cache (`moe-row-cache`) residenc
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1057` -- 2026-09-13: T-GPLR-18 (F1 pinned row-set staging) added to the plan as READY with
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1106` -- 2026-09-13: T-GPLR-15 contract rewritten gate-measurement-first (READY) — Phase A
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1140` -- 2026-09-13: T-GPLR-13 marked DONE (owner review) — reactive expert row-set residen
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1178` -- 2026-09-13: TG row-set staging analysis (section 6 of docs/gplr13-bottlenecks-hypo
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1205` -- 2026-09-13: T-GPLR-13 test ladder executed — all five sizes PASS (50/512/4k/200k r
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1264` -- 2026-09-13: T-GPLR-13 implementation COMPLETE — moe-row-set TG path runs end-to-en
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1330` -- 2026-09-13: Memory-validation probe (T-MEMPROBE-01, owner request) — no RAM/VRAM o
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1392` -- 2026-09-12: T-GPLR-13 gate (b) DECIDED — reactive row-set is the priority TG path;
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1443` -- 2026-09-12: Drafted model-run testing standard
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1467` -- 2026-09-12: moe-row-set SDD amendment written (owner-directed; T-GPLR-13 gate (a)
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1497` -- 2026-09-12: T-GPLR-16 MIN_BATCH=1 wedge fully characterized — device-side stream d
- 2026-09-11 -- `docs/currentstate/2026-08-20..2026-09-11.md:9` -- 2026-09-11: T-GPLR-16 root cause — mixed-arithmetic PP→TG switch, not corruption
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:60` -- 2026-09-10: T-GPLR-17 REJECTED by owner and fully reverted (two-process hybrid bri
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:88` -- 2026-09-10: T-GPLR-17 drafted READY — two-process hybrid bridge (streaming PP -> s
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:140` -- 2026-09-10: Streaming-PP + stock-TG hybrid (--handoff) at 50K — loud preflight ref
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:170` -- 2026-09-10: Streaming-TG slowness profiled against stock baselines (host-only anal
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:244` -- 2026-09-10: T-GPLR-14 numbering collision resolved (mixed-placement TG mode renumb
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:261` -- 2026-09-09: MoE TG row-set program drafted/renumbered; T-GPLR-14 DONE (route-overl
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:311` -- 2026-09-09: T-GPLR-12 DONE — default profile v2 (bf16 kept on MoE by owner decisio
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:348` -- 2026-09-10: Owner probe — MoE 50K streaming PP + streaming TG under an 8 GB VRAM c
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:376` -- 2026-09-09: NCMOE 262K probe — stock TG at full context, 16 GB pipeline evidence
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:406` -- 2026-09-07: T-GPLR-07 smoke tier wired into tools/run-tests.ps1 (host-validated)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:460` -- 2026-09-07/09: qa-smoke gate live runs (50/512 PASS; 15/30 removed from the ladder
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:512` -- 2026-09-09: Planner review -- T-GPLR-08 ready, T-GPLR-09 held
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:544` -- 2026-09-09: T-GPLR-06/07 CLOSED — T-GPLR-08 and T-GPLR-10 may start
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:581` -- 2026-09-09: T-GPLR-11 DONE — batch-major tail-ubatch gap closed
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:614` -- 2026-09-09: T-GPLR-08 implemented and probed — prefetch hypothesis FALSIFIED (inst
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:669` -- 2026-09-09: T-GPLR-10 DONE — MoE handoff (streaming PP -> stock llama.cpp TG hybri
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:763` -- 2026-09-09: T-GPLR-09 PARKED — owner decision, plan updated
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:802` -- 2026-09-09: T-GPLR-10 drafted — MoE handoff (streaming PP + stock llama.cpp TG)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:841` -- 2026-09-07: T-GPLR-07 runner flags --prompt-file / --tg-text landed (host-validate
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:878` -- 2026-09-07: --mtp + --pp-no-logits n_outputs_max clamp fixed
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:914` -- 2026-09-07: Host I/O Baseline and Tier Direction (owner decision)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:938` -- T-GPLR-05 Complete: Capability Registry + Regression Closure — Plan Done (2026-09-
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1252` -- T-GPLR-04 Complete: 262K Q8 Out-of-Core Capacity Proof (2026-09-05)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1303` -- T-GPLR-03 Complete: 512-Token MoE Layer-Major Prefill + Parity Closure (2026-09-05
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1351` -- T-GPLR-02 Complete: MoE Expert-Bank Package Proof (2026-09-04)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1382` -- T-GPLR-01 Complete: Qwen Package Adapter Extracted (2026-09-04)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1419` -- T-GPLR-00 Complete: MoE Artifact Inventory (2026-09-04)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1462` -- Active Generic Prefill Layer Residency Direction (2026-09-04)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1527` -- Arbitrary-Length Layer-Major PP (done 2026-08-23)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1574` -- 262K Attempt and Per-Width Graph Plan (2026-08-23)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1613` -- T-Q38PW-00: Patchability Probe (done 2026-08-23)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1699` -- T-Q38PW-01: Per-Width Graph Sets + Staged Input Buffer (done 2026-08-23)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1804` -- T-Q38PW-02: 262K Stretch with bf16 Slabs (done 2026-08-23)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1904` -- Per-layer slot bundles for dynamic-quant models (done 2026-08-23/24)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:1938` -- T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path (done 2026-08-24)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2043` -- T-Q38FA-02: resident control verdict (done 2026-08-24)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2078` -- T-Q38FA-04: patch-walk cache (done 2026-08-24)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2138` -- T-Q38TG-00: Hybrid TG Feasibility Probe (done 2026-08-24 — NO-GO)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2215` -- T-Q38HG-00/01/02: PP→TG Handoff (done 2026-08-24)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2382` -- 20K Layer-Major Evidence (done 2026-08-20)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2413` -- T-Q38PERF-01: copy timing split (done 2026-08-20)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2426` -- T-Q38PERF-02: package-class slot allocation (done 2026-08-20)
- (undated) -- `docs/currentstate/2026-08-20..2026-09-11.md:2440` -- T-Q38PERF-03: pinned-host package staging (done 2026-08-20)
- 2026-08-18 and earlier -- `docs/currentstate/2026-08-18-and-earlier.md:9` -- Completed

## Recent entries (2026-09-17 .. 2026-09-19)

## 2026-09-19: T-GPLR-48 CLOSED — `--pp-prebuild-per-pass` is now **KV-conditional**, so the default f16 path stops paying for a fix it does not need, and the fix stays exactly where it is load-bearing. **The task:** T-GPLR-46 measured that the per-pass rebuild (default ON for every `--pp-repeat > 1`) causes BOTH the quadratic pass-2 compute excess (+3.72 s at 50k, +31.55 s at 100k) and the 4k TG answer→echo, while its only justification is the T-GPLR-43 quantized-KV pass-2 crash — so the default was over-triggering in exactly the configuration where the rebuild is a pure cost. T-GPLR-49b then attributed the echo to the **pw × rebuild INTERACTION**, which made the drafted hedge do double duty: defaulting the rebuild OFF for f16 removes the answer-echo from the default path too. **The change, one condition:** the `-1` sentinel branch becomes `pp_prebuild_per_pass = (pp_repeat > 1 && kv_cache != "f16") ? 1 : 0;` (`kv_cache` is the runner's already-validated `"f16" | "q8" | "q4"` string — the single source of truth, no second KV-type inference path), plus the help text, plus run self-description: a stderr announcement emitted ONLY when the value came from the default, and two new `run_start` keys, `pp_prebuild_per_pass` (resolved int) and `pp_prebuild_per_pass_source` (`"explicit" | "default"`), following the T-GPLR-12 `handoff_mode` pattern. Explicit `0`/`1` overrides everything, so every recorded arm command stays reproducible. **Eleven repeat-2 arms plus one repeat-1 probe, one GPU session** (A1 f16 50k / A2 q8 B1 143360 / A4 f16 4k 96 / A5 f16 4k repeat-1, each with NO flag, plus the four explicit-flag control-changed pairs of AC4; three arms re-run in the same session after a harness-expectation fix, evidence fields identical), all under the mutex with started/verdict/done sentinels and strict `--check-telemetry` OK wherever a pass completed. **INV3 — THE f16 DEFAULT FLIPPED, AND THE FLIP IS BEHAVIORAL, NOT COSMETIC:** A1 resolves `0` with source `default`, announces `pp-prebuild-per-pass = 0 (default: f16 KV, no per-pass rebuild needed)`, its pass2−pass1 compute excess is **+0.02 s against the old default's +3.72 s**, and its 3 greedy ids `248046,198,248045` are byte-identical to BOTH the same-session explicit-OFF arm AND the recorded `gplr46-50k-off` (the ON arm gives `3702,279,2128`) — a telemetry key alone would prove nothing; the ids do. A4 at 4k/96 DECODE-GRADES **PASS 5/5** (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`, exit 0) with ids byte-identical to `gplr46-4k-off` including its trailing drift, so the new default reproduces the OFF stream exactly on a different binary; the contract's "9/9 pieces" wording predates the decoder's 5-item needle set and the measured claim is the tool's 5/5. **INV2 — QUANTIZED KEEPS THE REBUILD:** A2 (the T-GPLR-43 B1 shape, 143360 tokens, q8, no flag) resolves `1` with source `default`, announces `quantized KV q8, pass-2 rebuild active`, survives BOTH passes, checker OK, exit 0 in 190 s, and reproduces the recorded B1 routing signature exactly (1200 route events / 1120 singletons / diverging chunks 10..140). **INV1 — EXPLICIT FLAGS UNCHANGED, EXCEPT WHERE THE SHAPE CRASHES ON BOTH BINARIES:** each explicit command ran on the changed AND the control binary in one session; identity fields (greedy ids, pass count, per-pass launches, route events, route parity) are EXACTLY equal for the f16 pbp0, f16 pbp1 and q8 pbp1 pairs, timings within the ~3% session spread. The q8 **pbp0** pair cannot produce ids at all — **both binaries die with exit `-1073740791` (0xC0000409) in pass 2**, so the quantized-KV + rebuild-OFF crash reproduces at a mere **50k**, not only at the 143k T-GPLR-43 shape. That is identity in the only form the shape admits AND the strongest available support for INV2: the configuration the new default refuses to enter is one that crashes; both arms are graded `-ExpectCrash` rather than reported as harness failures. **REPEAT-1 PRIOR BEHAVIOR CONFIRMED (A5):** f16 4k, `--pp-repeat 1`, no flag — resolved `0` with source `default`, the announcement names the repeat-1 case (`pp-prebuild-per-pass = 0 (default: --pp-repeat 1, there is no pass boundary)`), strict checker OK, and the 96-id stream needle-grades **PASS 5/5**; identical to pre-change behavior by construction (both old and new default evaluate to 0 when `pp_repeat == 1`) and now also by measurement. **NEW REUSABLE, HOST-ONLY GATE: `tools/check-route-parity.py`** — under `--pp-repeat 2` every sampled route tuple must appear EXACTLY TWICE, so a pass-2 divergence is visible from a trace with no decode and at every length. A1 is clean (880 events / 0 singletons) while the old-default 50k arm `gplr46-50k-on` is DIVERGED (880 events / **800 singletons**, 40 layers in each of chunks 10..97) — that flip IS AC1's evidence. The tool was validated against SIX recorded traces with known ground truth before use (clean for every rebuild-OFF trace, DIVERGED for every pw + rebuild-ON trace) and exits **2, never 0, when a trace has no route events** — a run that crashed before routing is not a pass. Its generalization (pw AND rebuild ON are JOINTLY required, at 4k, 50k and 143k) is handed to T-GPLR-47 as a cheap no-decode gate. **STEP 1 CORRECTED THE DRAFT AND FOUND ONE REAL, DISCLOSED CONSEQUENCE:** the draft claimed the `f16/q8/q4` validation preceded the resolution; in the real tree only the `--inspect-model` path's check did, while the MAIN path's identical check sat AFTER it, so the announcement could have named an unvalidated value. Fixed here by hoisting the main-path check above the resolution — verified on both binaries: with two invalid flags the changed binary now reports `--kv-cache` first where the control reports `--copy-source` first; the accepted value set and all three valid paths are unchanged. That diagnostic reordering is the ONLY observable difference on the invalid-input path and it is disclosed rather than buried. **Falsifiers: NONE fired** — no f16 default run performs the rebuild, no q8 default run crashes, no explicit arm diverges. **CTest 68/70 on the changed binary**; the two failures (`qwen38-layer-boundary`, `layer-slot-lifecycle`) are a fixture-vs-default mismatch (`--pp-async-feed` defaults to 1, the fixtures pin `--layer-slots 1`) and were PROVED pre-existing by running the CONTROL binary with the fixture's exact argv — it fails with the identical message. **Evidence:** plan `T-GPLR-48` block (RESULT 2026-09-19, status done); verdicts `build/llama-cuda/verdicts/gplr48-*.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr48-*.jsonl`, summary `build/llama-cuda/gplr48-summary.json`, harnesses `tools/run-gplr48-arm.ps1` / `tools/run-gplr48-set.ps1`, gate `tools/check-route-parity.py`; binaries `build/llama-cuda/bin/gplr48-control` (exe `65047AA91D706E2D`, llama.dll `76017976E3478CF6` — byte-identical llama.dll to the recorded `gplr49-changed` binary, verified by hash AND by the absence of the new key in the DLL). **Ready front after this closure:** T-GPLR-47 (fresh-context rebuild — now with a named target, the pw width-class rebuild re-entry, and a decode-free pass-2 gate) is the remaining open task of the trio; T-GPLR-48 and T-GPLR-49 are done, and the T-GPLR-46 escalated owner decision is now actioned.

## 2026-09-19: THE 4k ECHO IS A **pw x REBUILD** INTERACTION, ATTRIBUTED — one two-arm follow-up closes the last open observation from T-GPLR-49 and names T-GPLR-47's target. **The question:** T-GPLR-49 recorded that at 4k/96 the pw arms emit document-echo while the repaired non-pw arm answers correctly, with the SAME rebuild default, and left it unowned. **The design, and why it was only two arms:** reading the recorded T-GPLR-46 arms first showed its `gplr46-4k-off` cell was **pw** (`--pp-schedule layer-major`, no `--no-pw-probe`, `pp_prebuild_per_pass 0`) on binary `9BA0AB55` — i.e. three of the four (schedule x rebuild) cells already existed, two of them on binaries older than today's, and T-GPLR-45 changed the very restore path the rebuild interacts with. So the cheap pair was the one cell at risk and the one missing cell, both on the current binary `gplr49-changed`: **pw + `--pp-prebuild-per-pass 0`** (re-takes T-GPLR-46's key cell across two later changes) and **nopw + `--pp-prebuild-per-pass 0`** (completes the 2x2). Both predictions were pre-registered and both held: exit 0, strict `--check-telemetry` OK, needle **PASS 5/5**, first id 248046. **THE 2x2, ALL FOUR CELLS ON ONE BINARY, same 4k anchor / ubatch 512 / tg 96:** pw+ON -> **echo** (ids start 760; the control and every pre-T-GPLR-49 4k/96 trace on disk agree byte-for-byte); pw+OFF -> **answer**; nopw+ON -> **answer**; nopw+OFF -> **answer**. So the echo needs the **INTERACTION** — the pw schedule alone is not enough (rebuild OFF restores the answer) and the rebuild alone is not enough (non-pw with the rebuild ON answers correctly). **A CHEAP MECHANICAL OBSERVABLE, found while attributing, and it generalizes:** in a clean `--pp-repeat 2` trace every sampled `route` tuple `(chunk, layer, hash, route_union)` appears **exactly twice**, because route telemetry samples every 10th chunk (`telemetry_chunk_stride` = 10). Measured per arm (host-only, no GPU, no decode): **pw+ON diverges between pass 1 and pass 2 at EVERY sampled chunk EXCEPT the first — chunk 7 of 8 at 4k (120 distinct tuples of 160) and chunks 10/20/.../90/97 at 50k (840 of 880), identically on BOTH binaries** — while pw+OFF, nopw+ON and nopw+OFF are exactly 2-per-tuple at 4k and nopw+ON is also clean at 50k; chunk 0 is clean in every arm. So the pass-2 corruption under pw+rebuild is **not 4k-specific**: 4k/96 is merely where it becomes visible in the answer, and the parity test gives a decode-free gate that also works at 50k. **TARGET, recorded in T-GPLR-47's block as PRE-START EVIDENCE (evidence only — no contract line changed):** the **pw rebuild path**, i.e. exactly what that task's fresh-context swap replaces — the per-pass re-entry of `begin_set` into the pw **width-class** sets, the continuing `rs_z = -1` class shared across chunks. NOT `prepare_chunk` (pw + rebuild OFF is clean, so the per-chunk patch alone does not break the answer) and not the rebuild's bookkeeping in general (non-pw + rebuild ON is clean). **Recorded as NOT this task's, so it is not lost:** pw and non-pw route the sampled LAST chunk differently in BOTH rebuild states — identical `(chunk, layer)` keys and event counts, different hashes and slightly different `route_union` — yet pw+OFF answers correctly with them, so that pre-existing schedule difference is not the echo's cause and needs its own owner if it ever matters. **Harness fact found the same way:** `host_ledger` and the `host_mem` `pass_end` markers exist only when the rebuild is ON, so a `--pp-prebuild-per-pass 0` trace carries no per-pass ledger — a wrapper that requires one will fail a working rebuild-OFF arm. **Evidence:** `build/llama-cuda/verdicts/gplr49b-4k-{pw,nopw}-rebuild0.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr49b-4k-{pw,nopw}-rebuild0.jsonl`, graded decodes `.grade.json`, same-binary reference cells `build/llama-cuda/gplr49-4k-{pw,nopw}-{control,changed}.jsonl`; harness `tools/run-gplr49b-echo-attrib.ps1` (self-contained: two arms, ids, stream class, graded decode, prediction check). **Open next:** T-GPLR-47 (fresh-context rebuild — now with a named target and a decode-free pass-2 route-parity gate) and T-GPLR-48 (KV-conditional rebuild default) remain the two `ready` tasks; the recommendation to add route parity to T-GPLR-47's AC1 evidence is written into its block for the Planner to accept or decline.
## 2026-09-19: T-GPLR-49 CLOSED — the latent `--no-pw-probe` act-slab offset defect is REAL, its pre-registered refusal chunks were measured EXACTLY, and the repair lands **without moving the pw default path at all**. **The task:** a read-only rehearsal of T-GPLR-47's invariant 2 surfaced a pre-existing bug in `begin_set`'s `!pw_probe && chunk_id > 0` branch — it derived a chunk's slab offset as `sum_{j<chunk} n_j * n_embd_io * sizeof(float)` while `slab_slice` sized slices and the slab half with `ggml_type_size(act_slab_dtype)`, exactly **2x too large** at bf16 (the default since T-GPLR-12), and chunk 0 skipped the derivation entirely and inherited whatever cursor state it found. Reachable through exactly one CLI combination (`--pp-schedule layer-major --no-pw-probe`, never run by any recorded arm, script, test or doc). The owner chose REPAIR over declaring the mode unsupported, so the mode had to work, not disappear. **The fix, one owner for the arithmetic:** new pure `llama_ls_slab_region_of(chunks, chunk_id, n_embd_io, act_dtype)` beside `llama_ls_chunk_layout` derives `{offset, bytes, half}` from the immutable descriptor for EVERY chunk (chunk 0 included, always offset 0 in a fresh run), always with `ggml_type_size(act_slab_dtype)`; `slab_slice` keeps its slot-only cursor role but sizes through the helper and **claims** the bound interval. `llama_layer_streaming::claim_slab_region` records `[start,end)` per slab half and refuses LOUDLY (error_msg naming both intervals) on any overlap — the mechanism contract 2 chose, because the descriptor cannot know how many live sets share a `chunk_id`. Step 1 established from the code that this is safe in BOTH modes: non-pw prebuilds one set per chunk (adjacent, non-aliasing by construction) and pw prebuilds one set per (width, rs_z) class with `slab_slice`'s cursor advancing once per class, so classes are sequential — the claim never fired in any of the eight arms. **THE DEFECT IS REAL AND THE PREDICTION WAS EXACT (pre-fix control, exit 1 by design):** 4k refused at **chunk 4 of 8** with `activation slab A exhausted (18874368>16777216)`; 50k at **chunk 49 of 98** with `(207618048>204800000)` — both exactly the pre-registered `2*sum_{j<c} n_j + n_c > pp`, re-derived independently before the arms ran, and the byte arithmetic (18,874,368 = (2*2048+512)*2048*2) identifies the doubled element size rather than merely correlating with it. Refusal happens in the PREBUILD (9.3 s / 3.2 s), so a pre-fix binary never executes a chunk at a wrong offset at all — which is why the AC's overlap-corruption question stays OPEN and is recorded as such. **THE REPAIR WORKS:** fixed binary at the same flags exits 0 with strict `--check-telemetry` OK, no slab-exhausted message, 96 ids (4k) / 3 ids (50k), act_bind 16 / 196, and at 4k the stream DECODE-GRADES CORRECT — `tools/decode-tg-ids.py` needle aggregate **PASS 5/5** (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`) — so the mode that never completed anywhere is now functionally right, not merely non-refusing. **THE DEFAULT PATH DID NOT MOVE (blocking gate, paired arms in one session):** 4k greedy ids 96/96 byte-identical, `act_bind` 16/16 order-sensitively identical, `route` 160/160 identical INCLUDING the per-layer hash, the T-GPLR-45 `patch_cache` dumps identical field-for-field (kept 18,436 B, `excluded_count` 8 / 8,407,044 B at the accumulated set, `union_matches` 1), `run_start`/`pkg_member`/`graph_footprint`/`byte_account` identical, every event-kind count identical except the timing-only `recon_attr` (74->75); 50k the same with ids 3/3, act_bind 196/196, route 880/880, dumps 3/3 (only `recon_attr` 86->84). `patch_us`/pass 0.61->0.64 ms (4k) and 4.96->5.48 ms (50k): unchanged, which is exactly what a task repairing an unexercised mode owes the default one. **HOST GATE (cheapest, blocking):** `test-layer-streaming-layout` gained the descriptor-derived offset/bytes/half table over 7 shapes, chunk-0-from-derivation, non-overlapping claims, the F32 case bit-identical to the pre-fix formula, the pre-fix bf16 offset exactly 2x the truth for every chunk >= 1, every refusal case, the half-open overlap predicate and the claim registry itself — **21 checks pass**, while restoring the pre-fix formula (probe) fails **519** assertions and disabling the claim check (probe) fails exactly the two aliasing cases, so both halves are load-bearing in the direction the task claims. Host suite 37/37, telemetry suite pass, doc-size guard PASS. **FIRST COST MEASUREMENT of the repaired mode (reported, not gated):** same-session 50k, non-pw pass-1 sweep 25,044.9 ms / compute 23,380.4 ms / patch ~0 ms vs pw 19,467.8 / 17,872.5 / 5.3 ms (pass 2 23,895 / 23,289 / ~0 vs 22,267 / 21,669 / 5.5) — the mode now has a baseline where it had none; whether the extra compute is inherent to one-set-per-chunk is NOT established. **TWO THINGS DECLARED RATHER THAN GLOSSED:** (1) `prepare_chunk` opens with `if (!ls->pw_probe) return;`, so in this mode there is no per-launch patch and no patch cache at all — the T-GPLR-45 composition dump physically cannot exist there, and the arm wrapper's demand for it wrongly verdicted the first two changed arms FAIL, which is recorded and then corrected by re-running them with a pw-only expectation; (2) at 4k/96 the pw arms emit the document-continuation stream (needle FAIL) while the repaired non-pw arm at the same shape emits the correct answer — and the pw stream is byte-identical to ALL FOUR pre-existing 4k/96 traces on disk (`gplr44-ab-4k-{unscoped,scoped}`, `gplr45-4k-{control,changed}`), so it long predates this task; both modes carry the T-GPLR-43 rebuild default, which is consistent with T-GPLR-46's rebuild-linked answer→echo finding EXCEPT that the non-pw arm still answers correctly, pointing at the pw rebuild path rather than at `--pp-repeat > 1` alone — hypothesis NOT tested here, handed to T-GPLR-47/T-GPLR-48. **Evidence:** plan `T-GPLR-49` block (RESULT 2026-09-19, status done); verdicts `build/llama-cuda/verdicts/gplr49-*.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr49-{4k,50k}-{pw,nopw}-{control,changed}.jsonl`, smoke `build/llama-cuda/gplr07-qa-512.{jsonl,txt,log}`; harnesses `tools/run-gplr49-pair.ps1`, `tools/run-gplr49-arm.ps1`, `tools/run-gplr49-nopw-rerun.ps1`; binaries `build/llama-cuda/bin/gplr49-{control,changed}` (exe + all DLLs). **Ready front after this closure:** T-GPLR-47 (fresh-context rebuild) and T-GPLR-48 (KV-conditional rebuild default) remain the two open `ready` tasks; T-GPLR-49 is done.
## 2026-09-19: T-GPLR-45 CLOSED — PERFORMANCE VERDICT MET: the per-launch patch restore now copies only the tensors a graph actually consumes, and `patch_us` fell **96x–166x per launch / 95.9x–165.5x per pass** with the identity gate CLEAN at all three shapes. **The task:** T-GPLR-44 measured that every launch restores 7 tensors / 4,212,740 B when the transitive minimum is ~4,099 B (**1,027x**), 99.6% of it one `inp_embd` orphan that ZERO of the 42 graphs consume (a `build_inp_embd` artifact: it allocates both `inp_tokens` and `inp_embd`, then `ggml_build_forward_select` prunes the embeddings branch in token mode). T-GPLR-45 drops every zero-consumer INPUT member at CACHE-CAPTURE time by consumption (never by name), with no per-graph scoping, no `begin_set`/lifecycle change and no re-walk — which is why it does not re-enter the hazard that reverted T-GPLR-44. **Mechanism (one walk owns both):** `build_patch_split` now classifies the set ctx's INPUT candidates once — kept iff at least one graph reaches them transitively (through views) — and that single classification feeds the snapshot capture, the launch restore and the recon dump, so the sites cannot drift; the cache build moved to one owner (`ensure_patch_cache`) called from the chunk-snapshot site before the first snapshot and guarded again in `prepare_chunk`. **Instrumentation, measured:** the dump gained `excluded_count`/`excluded_bytes`/`excluded_members` and the byte identity `kept + excluded == walked` holds at EVERY changed-arm dump, with `walked` EQUAL to the control's at every paired dump (the filter cannot create or lose bytes). Candidates/walked: 4,212,740 B at 4k set 0 (7 members) and 2,764,612 B at 50k set 97 / 2,632,964 B at 200k set 390 (7 each); the 4k arm's second dump shows the doubled list T-GPLR-44 recorded (**7 -> 14 members, 8,425,480 B**), all 8 extras zero-consumer (`inp_tokens:2048`, **two** `inp_embd:4194304`, four leaf views 4/4096/4096/8192 B plus a 0 B leaf) — the 50k/200k dumps do NOT show the doubling, a shape difference this task does not explain (T-GPLR-47 owns the accumulation). Kept: 18,436 B at the full-batch dumps, 12,100 B (50k) / 11,524 B (200k) at the partial final chunks. `union_matches` — the engine's own two-sided closure flag, 1 meaning every cached tensor has >= 1 consumer — reads **0 on the control, 1 on the changed arm at every dump**, and `redundancy_input` collapses **1,027.748x -> 4.498x**. Analyzer selftest 7/7 including the new consumed-but-excluded negative control failing as designed. **PERFORMANCE (the verdict), paired in ONE session under one mutex hold, 6 arms, all exit 0, strict `--check-telemetry` OK, changed binary `89A7E7E275CC4AFD` vs the preserved control `9BA0AB55A05D5A98` (exe plus all DLLs side-copied before the rebuild):** 4k `patch_us`/pass 106,742 -> **645 us** (165.5x; 5.168% -> 0.033% of the sweep), 50k 598,152 -> **6,234 us** (95.9x; 2.659% -> 0.028%), 200k 2,432,783 -> **19,033 us** (127.8x; 1.231% -> 0.010%); per launch 317.7 / 145.3 / 148.1 us -> **1.92 / 1.51 / 1.16 us**. The >= 50% bar is cleared by more than an order of magnitude at 200k (99.2%). **Secondary wall (not the verdict):** pass-2 sweep delta -82.8 ms (**-4.01%**) at 4k, -271.8 ms (**-1.21%**) at 50k, -737.9 ms (**-0.37%**) at 200k — so 4k and 50k clear the 1.0% secondary bar and 200k is reported per contract as "restore narrowed, wall not resolved at this spread": the 200k restore is only 1.23% of the sweep and the changed arm's compute bucket ran +1.5 s (inside the ~3% session spread T-GPLR-46 recorded), which offsets most of the 2.4 s patch saving — **the 200k wall lever remains the pass-2 rebuild excess (T-GPLR-46/T-GPLR-47), not this one.** **IDENTITY GATE CLEAN (blocking criterion):** greedy ids byte-identical (96/96 at 4k, 3/3 at 50k, 3/3 at 200k — note the 4k pair sits in the rebuild-ON echo regime, and the filter changes NOTHING about it); routing identical — all `route` events match field-for-field (160/160, 880/880, 3200/3200) INCLUDING the per-layer route `hash`, which is the routing digest this contract asked for (the engine has no separate digest event); staged bytes per round equal — `phase_summary.bytes_transferred` identical at every phase and shape (18,327,011,328 for both pp phases, 54,981,033,984 for the 50k/200k tg phase) with evictions 37/40/120, layers 40 and launches 336/4116/16422 all identical; semantic-QA smoke rung (512, retention mode) check-telemetry OK, 5/5 needles PASS (`QX-7741 TR-2918 MK-5501 LD-8834 Paris`), negative control fails as designed. **No falsifier fired:** no identity divergence, the orphan is served by 0 graphs at every dump (not load-bearing at this shape), and `patch_us` fell far past the 50% hand-back trap, so the memcpy cost model is confirmed. **AC expectation CORRECTED, not the AC:** the draft predicted the 50k excluded set at 4,194,304 B (extrapolated from 4k); measured, the excluded `inp_embd` scales with the FINAL CHUNK's token count because it is allocated per ubatch shape — 4k 4,194,304 B (4096 = 8x512, no partial chunk), 50k 2,752,512 B = 336x8192, 200k 2,621,440 B = 320x8192 — so the larger figure does not hold while the identity does. **RECORDED CONDITION, not task-owned:** `tools/run-tests.ps1` host and telemetry suites are green on the changed binary, but the 8 model-backed `-run` tests fail a handoff invariant (`handoff_mode on/auto-on but no handoff events`) **identically, with identical messages, on the preserved control** — pre-existing, needs an owner. **Evidence:** plan `T-GPLR-45` block (RESULT 2026-09-19, status done); verdicts `build/llama-cuda/verdicts/gplr45-*.verdict.json` (+ `.started`/`.done`), traces `build/llama-cuda/gplr45-{4k,50k,200k}-{control,changed}.jsonl`, smoke `build/llama-cuda/gplr07-qa-512.{jsonl,txt,log}`; harnesses `tools/run-gplr45-pair.ps1`, `tools/run-gplr45-arm.ps1`; analyzer `tools/analyze-gplr44-cache.py` (selftest 7/7). **Ready front after this closure:** T-GPLR-47 (fresh-context rebuild — the root fix for the pass-2 regression and the accumulation seen above) and T-GPLR-48 (KV-conditional `--pp-prebuild-per-pass` default) are both drafted `ready`; T-GPLR-45 is done.

## 2026-09-19: T-GPLR-46 CLOSED — the pass-2 compute regression is REAL and REBUILD-LINKED, and the same rebuild degrades TG answer semantics at 4k layer-major. **Supersedes two recorded claims with dated pointers (not silent edits):** the T-GPLR-44 closure's "pass-2 wall difference is session-level and UNEXPLAINED" and its "corrupted restore" mechanism narrative for the run-1 id divergence are both corrected here. **The experiment (five pre-registered arms + two identity probes, binary 9ba0ab55, GPU mutex held, all exit 0, strict `--check-telemetry` OK; evidence `build/llama-cuda/verdicts/gplr46-*`, traces `gplr46-*.jsonl`):** (1) TRIGGER GATE PASSED — 50k A/B same binary same session: pass-2 compute excess +3.72 s with the per-pass rebuild ON (inside the pre-registered 3.0–4.6 s band) vs −1.31 s (collapse to pass-1 noise) with `--pp-prebuild-per-pass 0`; the per-pass rebuild IS the trigger. (2) GROWTH LAW — 4k ~0 s (7 chunks, below detection), 50k +3.72 s (0.40 ms/chunk²), 100k +31.55 s (0.83 ms/chunk², OUTSIDE the pre-registered 8.7–15.6 s band, ~2x the quadratic prediction — recorded honestly, possible second mechanism at 100k), 200k +59.2 s (0.39 ms/chunk², prior session); monotone and reproducible across sessions, not one clean law. (3) SATURATION — the 50k 3-pass arm shows pass-3 compute 21,676 ms vs pass-2 21,684 ms: the excess is a ONE-TIME jump at the first rebuild, it does not compound per pass. (4) ESCALATION FIRED (pre-registered "identity changes across the flag") and RESOLVED BY DECODE, host-only, no new dependencies — the GGUF `tokenizer.ggml.tokens` array was read with a pure-struct Python walk (`build/llama-cuda/gplr46-vocab.json` cache): the rebuild-OFF arms and a repeat-1 probe emit the CORRECT needle answer (9/9 pieces, `QX-7741 TR-2918 MK-5501 LD-8834 Paris`, format identical to the passing smoke rung), while the rebuild-ON pair arms at 4k — byte-identical on BOTH binaries (control 7bad011e and revert 9ba0ab55) — emit document-echo (`The maintenance log for row 00114...`, 0/9); and EVERY pre-rebuild-binary QA trace on disk (gplr07..gplr39, 09-07→09-17, both schedules) decodes to the correct answer. **Conclusion:** the per-pass rebuild causes BOTH measured defects — the quadratic pass-2 compute excess and the TG answer→echo degradation at 4k layer-major; the batch-major q8 smoke rung with rebuild ON stays correct (the semantic regression is schedule/shape-conditional so far as measured). **T-GPLR-44 amendment:** the run-1 "degenerate" ids (248046,198,248045...) were the ANSWER-format stream (`<|im_end|>\n<|im_start|><think>\n\n</think>\n\nQX-...`), so the pre-registered falsifier fired correctly on divergence but the corruption narrative is withdrawn; the revert decision itself stands per contract. **OWNER DECISION ESCALATED, not actioned:** the default of `--pp-prebuild-per-pass` — the rebuild's only justification is the quantized-KV pass-2 crash, yet it defaults on for every `--pp-repeat > 1`; at f16 it buys the quadratic excess and the TG regression. Recommended follow-up (drafted, not implemented): make the default KV-conditional and/or fix the rebuild path. **Also produced:** the vocab-decode probe is reusable for any future id-stream question (248046 = `<|im_end|>`, 248044 = `<|im_start|>`-cluster, so special-token-leading streams are answer-scaffolding, not garbage). **AMENDED 2026-09-19:** the probe is now the owned tool `tools/decode-tg-ids.py` (trace → detokenize → check-answer needle grading, exit 0/1/2 = PASS/FAIL/error), committed with its host-only vocab cache `build/llama-cuda/vocab-cache.json`; it supersedes the one-off heredoc probe and the `gplr46-vocab.json` cache described here, and T-GPLR-47's AC1 now grades through it. **Evidence:** plan `T-GPLR-46` block (RESULT 2026-09-19, status done); harness `tools/run-gplr46-rebuild-ab.ps1`, `tools/run-gplr46-set.ps1`; traces `build/llama-cuda/gplr46-{50k-on,50k-off,100k-on,100k-off,50k-3pass,4k-off,4k-1pass}.jsonl`.

## 2026-09-18: T-GPLR-44 CLOSED -- REVERTED-BY-FALSIFIER (the instrument is kept, the behavior is not). **The task:** scope the per-launch patch (the `prepare_chunk` restore) to the graph about to run, to cut the measured 3,402.3 ms/pass of patch time at the 200k shape; the pre-registered identity check (greedy ids byte-identical) was the falsifier. **What happened:** step 1 (the composition dump) measured the redundancy honestly -- per launch, 7 tensors / 4,212,740 B are restored when a scoped launch needs mean 4,099 B (**1,027x**), and 99.6% of the bytes are one `inp_embd` orphan (exactly 4 MiB, referenced by ZERO of the 42 graphs -- a framework artifact: `build_inp_embd` always allocates both `inp_tokens` and `inp_embd` and prunes the embeddings branch in token mode). Step 2 (per-graph scoping + a `begin_set` cache reset) IMPLEMENTED, and the falsifier FIRED: greedy ids diverged at BOTH shapes (degenerate pattern 248046, 198, 248045...). The mechanism, from the dumps: the scoped build's pw sets REBUILD in pass 2 (T-GPLR-43's per-pass rebuild re-enters `begin_set`), the ctx ACCUMULATES across rebuilds (7 -> 14 -> 28 members, first copy all orphans), and the reset made the scoped launches re-walk the accumulated ctx while the pre-patch build survives the same rebuild only via address-stable stale pointers. Per the pre-registered rule -- identity failure means revert and report the shape, never tune the patch to match -- the launch-path scoping and the reset were reverted (nested `e045fd611`), keeping the dump and its analyzer as read-only telemetry. **Validation of the reverted binary (all measured):** A/B vs the preserved pre-patch control (`7bad011e`) in one session -- greedy ids BYTE-IDENTICAL (96/96 at 4k, 3/3 at 50k), `check_telemetry` 0 on all four arms, `patch_us` parity (4k 114.8 -> 110.4 ms, 50k 637.0 -> 648.2 ms, inside the ~2.9% session spread); semantic-QA smoke rung PASS (5/5 needles, negative control fails as designed); 200k confirmation exit 0 in 359.9 s with the dump firing 3x at redundancy ~1027x and `inp_embd` named as the sole closure break each time. Per-pass 200k ledgers: pass 1 sweep 140.9 s / patch 2,987.3 ms (2.121%); pass 2 sweep 198.2 s / patch 2,530.3 ms (1.277%) vs the T-GPLR-36 different-session baseline 3,402.3 ms (2.40%) -- **the pass-2 wall difference (198.2 vs 141.8 s, compute-bound 194.5/198.2) is session-level and recorded UNEXPLAINED.** **What stands:** the step-1 composition measurement is the deliverable -- any future attempt at this lever must first solve the begin_set/rebuild ctx accumulation and the stale-pointer hazard the per-pass rebuild introduces (the current build is one address-stability accident away from the same corruption), not re-tune the scoping walk; and dropping the `inp_embd` orphan alone (a framework fix, not a scoping one) would remove ~99.6% of the restore bytes. **Also learned, now recorded in AGENTS.md:** attribute INPUT tensors by transitive src walk, never direct sources (a view-consumed input sits one hop past the node's direct src); the 4k+96-id identity arm belongs at 4k because layer-major refuses `--tg-tokens 96` at 50k without an active handoff. Evidence: plan `T-GPLR-44` block (Closure 2026-09-18; Status CLOSED with commits nested `f53c1bbf0`/`e045fd611`, outer `970b7a4`/`55148ea`); traces `build/llama-cuda/gplr44-dump-50k.jsonl`, `gplr44-ab-200k.jsonl`, verdicts `build/llama-cuda/verdicts/gplr44-*`, `build/llama-cuda/gplr07-qa-512-retention.{jsonl,log,txt}`; analyzer `tools/analyze-gplr44-cache.py` (selftest 5/5).

## 2026-09-18: THREE TASKS CLOSED (T-GPLR-35, T-GPLR-36, T-GPLR-38) — and the answer to "what default does closing them set?" is NONE of the three. **Why this close-out:** all three had been sitting in-progress/run with their evidence already recorded, but NO task in the plan was `ready`, so the backlog was decisions and closures rather than new implementation. **What each closure decided, with the numbers it rests on:** (1) **T-GPLR-35 (PP staging overlap on a second CUDA stream) — DISCARD the stream-topology lever.** The overlap=1 arm measured a 3088.4 ms warm 4k wall against the copy-backend baseline's 3089.0 ms — measurable identity — with the copy windows already pure in BOTH arms (median 31.4 ms = 1.00x the 31.37 ms pure-rate ceiling) and lag >= 0 on 76/76 samples, so the contract's falsifier fired: no DMA ever began while the compute backlog drained, i.e. there was nothing for a second stream to overlap. The wall is host-side instead (of 3089 ms: compute 1350, pure DMA 1238; the 3988 ms host transfer window splits into a 1559 ms host hop and 2429 ms ready latency of which only 1238 ms is engine time and 1191 ms is queue wait). **Kept, because it is a defect fix and not a performance result:** `backend_compute` was published only inside `layer_streaming_compute`, which layer-major PP never calls, so on the primary workload the aux mechanism could not engage at all — it is now published in `layer_major_pp` with the loud aux refusal and the recon emits `recon_unavailable` instead of a silent no-op, which is exactly what made this falsification observable. **Default: `--pp-copy-overlap` stays 0, slot count stays 2.** (2) **T-GPLR-36 (the long-context ~10 s residual) — CLOSED; the residual is NAMED and it is host bookkeeping.** The ledger closes exactly (0.00%): of a 141,847.9 ms warm pass (1,409.96 tok/s) over a 140,807.7 ms sweep, `other` 4,516.3 ms (48.3%), `patch` 3,402.3 (36.4%), `slot` 1,065.1 (11.4%), `setup` 278.6 (3.0%), `graph_call` 22.1 (0.2%) across 16,422 launches. The owner's per-graph-setup hypothesis is FALSIFIED (3.0%) and both pre-registered predictions were wrong the same way — `graph_call` forecast largest and measured 22.1 ms, `other` forecast under ~1 s and measured 4,516 ms — so the pre-registered falsifier did NOT fire. The `other` bucket has since been NAMED by T-GPLR-39's recon: route telemetry (`route_us` 1161.5 of 1152.8 ms at u2048, >100% because the split closes by construction), i.e. the device→host probs readback stalling the compute stream plus a host sort, scaling with ubatch SIZE (237.9 → 484.5 → 962.1 → 959.0 ms at u512/1024/2048/4096). The host-RSS half of the family's question — the owner's 19 → 30 GB growth, previously uninstrumented — is answered too: at 200k the working set climbs 4.13 → 25.05 GiB of which **83.5% is the file-backed model mapping** while the engine's own host holdings are FLAT at 1,494.0 MiB, so it is not a leak, and 30 GB needs more ANONYMOUS demand rather than more mapping (cross-instrument agreement: external monitor 23,458 MiB vs this instrument 23,457.1 MiB). **Default: `--pp-recon` stays opt-in (0); nothing changes in a normal run.** **HANDED BACK TO THE PLANNER, not drafted here:** this task's own measured lever list is still un-drafted — (2) make `prepare_chunk` per-chunk instead of per-launch (3,402.3 ms/pass at 200k, ~97% redundant) and (3) deep-submit chunk graphs to drop the per-launch sync that exposes 0.564 ms/launch of host prep (duty 93.4%); and these two are NOT together worth the full ~9.3 s this task's own list quotes — that figure is the TOTAL per-launch host prep exposed by the per-launch sync (0.564 ms/launch x 16,422 launches = 9.26 s = `other` 0.275 + `patch` 0.207 + `slot` 0.065 + `setup` 0.017 ms/launch), of which lever (1) `other` = 4.5 s is route telemetry and belongs to T-GPLR-39's fix-class decision, lever (2) `patch` = 3.4 s is an independent reduction of real redundant work (~2.4% of a 200k pass), and lever (3) deep-submit is the MECHANISM that would hide the residual exposure rather than a bucket of its own, so the three are not additive into 9.3 s twice. **Attribution corrected at close** (see the plan closure's ATTRIBUTION note): drafting (2) alone is cheap and touches no ordering semantics, whereas (3) removes the per-launch ordering the slot-reuse invariant relies on (a released slot can still carry `consumer_pending`; the wait can span a pass boundary, `src/llama-layer-streaming.cpp:823`) in an area where this program already took a pass-2 crash (T-GPLR-40/43) — so (3) should be SEQUENCED AFTER the T-GPLR-39 decision rather than drafted alongside (2). With that correction, the largest measured PP lever currently open is the per-launch host prep exposure as a whole, and it sits directly inside the charter's first priority. (3) **T-GPLR-38 (ubatch sweep) — CLOSED; u1024 is the measured 4k sweet spot and u2048 is a regression under the current host loop.** Warm wall 2317.5 (u512) → 2224.3 (u1024, -4.0%) → 2510.6 (u2048, +8.3%) → 2435.2 ms (u4096); tok/s 1767.4 → 1841.4 → 1631.5 → 1682.0; consumer wait FLAT (67.7-68.0 ms, so the predicted granularity penalty never appeared) and peak VRAM identical (3265.56 MiB, the known defect reading — the sweep probes the shared 3-slot window). The mechanism is the interesting part: ledger `compute` fell monotonically (1862.6 → 1282.7 ms) while `other` exploded (237.9 → 959.0), so the pre-registered overhead model was launch-COUNT-based when the residue is ubatch-SIZE-based. **Default: `--ubatch` stays 512 (main.cpp:2534) — deliberately NOT raised to the 1024 sweet spot**, on three grounds: the win is a net -93 ms at ONE shape and the pipeline gate cannot model a moving ubatch axis (the paired runs correctly refuse same-config pairs, so 1.042x/0.923x/0.952x are recorded but ungated); the mechanism says bigger is not generically better because the growing bucket is a function of ubatch SIZE, so raising the default before the host loop is fixed would trade a 4k gain for a long-context loss; and the only long-context ubatch arm (144k) is INVALID — both arms crashed in pass 2 and the surviving pass-1 numbers are COLD and arm-order confounded (the crash was later bisected: tail chunk exonerated, q8 KV + pass 2 is the trigger). Choosing a default on one shape and a wrong assumption is exactly how the K=4 row-cache default came about, so it was not repeated. **Open observation carried forward, not dropped:** in the same sweep greedy TG text was byte-identical at u1024 and u2048 but **u4096 DIFFERS (262 vs 305 bytes — the answer is cut short)**; that is an identity sanity check, NOT the semantic-QA gate, and no arm at or above u4096 should be treated as output-equivalent until it is investigated. **Also disclosed:** T-GPLR-35's 3-slot arm showed a per-pass compute rise of 1369 → 1611 ms (+18%) at identical bytes and arithmetic, recorded then and still unexplained (it does not bear on the closure, which rests on the 2-slot pair); and T-GPLR-36's ledger `compute` exceeds the summary `total_compute_us` by 274 ms because the embed/output loops run graphs that no layer stat accumulates, so ~0.27 s of apparent residual is real compute. **Evidence:** plan blocks `Closure (2026-09-18)` under T-GPLR-35/36/38, with each status marked CLOSED and the historical status preserved beneath it; traces `build/llama-cuda/gplr36-ledger-200k.jsonl` (2 `host_ledger` lines, one per pass), `gplr36-hostmem-200k.jsonl`, `gplr38-u{512,1024,2048,4096}.jsonl`, `gplr39-u2048-recon.jsonl`. These closures are docs-only; the underlying code and evidence commits are already recorded in each task's Commit field, and T-GPLR-36's placeholder is now filled (nested 8e14abd36 — the first commit carrying the `host_ledger` emitter, measured before it was committed — plus outer a58fdee, outer 4b13ac3, and the host half nested c5d79ad5c / outer fe9c1cf). **Still open after this close-out:** ONLY T-GPLR-37 (promote V2 mmap-registered staging to default? — wins at 4k, does not transfer to the crowded shape, 17.23 GiB pinned) and T-GPLR-39 (which fix class cuts route telemetry) remain in progress, both blocked on owner decisions and nothing else. **Hygiene found while closing, not fixed here:** four DONE/CLOSED tasks still carry the `Commit: (fill on completion)` placeholder — T-GPLR-27 (line 360), T-GPLR-32 (647), T-GPLR-33 (768) and T-GPLR-31 (2293); the fifth, T-GPLR-39 (1709), is legitimately unfilled because that task is not done.

## 2026-09-18: A LAUNCH-TIME **START MARKER** CLOSES THE LAST AMBIGUITY IN ARM OBSERVATION — a missing verdict can no longer mean "never started" and "still running" at once. **Why:** the sentinel convention made completion readable but left its own gap, which was measured the same day -- a detached `ctx_shell` job given a RELATIVE `-File` path died in under a second and, from the files alone, looked exactly like a run in progress. The fix is to record the START, not just the end. **The mechanism:** every arm now writes up to three files with one tag, and WHICH EXIST IS THE STATE -- `<tag>.started` at launch (before the mutex wait and before any model load), `<tag>.verdict.json` on exit, `<tag>.done` LAST. `Get-ArmRunState -Dir <dir> -Tag <tag>` returns `never-started` (neither), `running` (marker, no sentinel, recorded pid alive), `died` (marker, no sentinel, pid GONE -- started and vanished without a verdict), or `finished` (sentinel present -- the only state that reports an outcome; if the marker is missing it is flagged as an `anomaly` rather than quietly trusted). `Clear-ArmVerdict` (called BEFORE each run) now deletes all three, so a previous run's state can never be read as the current run's; `Wait-ArmStart` is the bounded launch-confirmation wait, returning the marker the moment it appears or `$null` on timeout. **Written at the same three chokepoints as the verdict** -- `tools/with-gpu-lock.ps1` (so every `run-*-locked.ps1` wrapper gains it with no edit to the wrapper), the row-cache driver (which acquires the mutex itself), and the smoke wrapper (chain-level marker) -- with the tag and directory unchanged from the sentinel convention. **The two questions are now separate and both answerable:** the `job_id` says whether the LAUNCH was accepted (and why not, when the tool rejects it); the marker says whether the SCRIPT BODY BEGAN. Neither replaces the other. **Validation, all measured (34 test cases plus four real processes):** `tools/test-arm-verdict.ps1` gains sections 7-11 and passes in full -- the four states, the dead-pid case, the anomaly case, stale-state invalidation (a tag that reads `finished` returns to `never-started` and then to `running` rather than `finished`), both bounded waits (0.00 s when the file exists, 3.03 s timeout with `$null`), and an ORDERING PROOF that needs no timing assumption: the wrapped command checks for its own marker and reports through its exit code (exit 0 = the marker existed when the arm body started; exit 9 would have meant it came too late). The four real processes, same tag: the relative-path detached launch reads **`never-started`** while the job id independently reports `failed` exit 1 with the reason; the same command with an ABSOLUTE path checked ~5 s in reads **`running`** with `alive=True`, pid and start time in the marker and no sentinel yet; checked once at the end reads **`finished`** `PASS 0` 17.5 s (arm tag `PASS 0` 10.8 s, GPU back to 0 MiB); and a marker-holding process killed with `Stop-Process -Force` (no `finally`, no `trap`) reads **`died`** with `alive=False`. **Limits, stated so the marker is not over-read:** liveness is BEST-EFFORT -- `died` is claimed only when the recorded pid is known gone on the same host, so a REUSED pid can make a dead run read as `running`, and the authoritative answer for a completed run is always the verdict; and a hard-killed process still writes no verdict, which is exactly why its state is `died` and never `finished` (an absent sentinel must never be reported as a passing run). Nothing about what a run MEASURES changed: no arm flag, criterion or telemetry was touched. Evidence: `tools/arm-verdict.ps1`, `tools/test-arm-verdict.ps1`, `tools/with-gpu-lock.ps1`, `tools/run-gplr19-rowcache-ab.ps1`, `tools/smoke-gpu-lock-ctxshell.ps1`; artifacts `build/llama-cuda/gpu-smoke-{ctxshell,arm}.{started,done,verdict.json}`; recorded in `docs/AGENT_ENVIRONMENT.md` (the state table in "Arm verdicts, the start marker and the `.done` sentinel").

## 2026-09-18: THE BARE FLAG'S DEFAULT RETENTION DEPTH IS NOW **K=8** (owner-directed, superseding the K=4 default of 2026-09-14). **The question that produced this:** the owner asked whether "keep K=8" had made K=8 the default. It had NOT -- the pick was the T-GPLR-26 branch decision plus a per-run operating depth, and `--residency-tg-retain 8` lives in the run command, not in a default. Three separate things are called "default" here, and the answer to each is now recorded: (1) the row cache itself is **OFF** by default (`residency_tg_retain = 0`, opt-in); (2) the bare `--residency-tg-retain` resolved to **K=4** via `kRowSetRetainDefault` (`tools/layer-streaming/main.cpp`), which the device fitted DOWNWARD and never upward (`row_cache_fit_depth` searches `k = requested .. 1`, `main.cpp:193`, so K=8 was unreachable from the bare flag); (3) the A/B driver's own `-RetainK` default is **1**, kept so `-AnalyzeOnly` regenerates the recorded K=1 report, and the driver always passes an EXPLICIT depth. **Why the default was raised:** K=4 was chosen as "the shallowest depth whose (b) clears the 0.6 gate" on the strength of 0.6284 on the preserved probe stream -- a stream that echoes its own anchor. The non-echoing re-probe that comment named as its outstanding validation was executed and MISSES at K=4: honest (b) is **0.5355 at K=4** against **0.6487 at K=6** and **0.7283 at K=8** (K=1 0.2468, K=2 0.3778; echoed 0.3088/0.4343/0.6284/0.7169/0.7775). So the one measured depth that failed the gate it was chosen for was the DEFAULT itself, while already costing ~2.19 GB; K=8 carries the highest measured honest (b) AND the best measured honest TG gain (+11.414 tok/s) and is the owner's operating depth, so the bare flag now resolves to the operating point instead of to a superseded one. **Scope of the change, deliberately minimal:** ONE constant (`kRowSetRetainDefault` 4 -> 8) and its two prose sites in `main.cpp` -- the justification comment (rewritten to state the honest curve and that its 0.6284 was echo-inflated) and the `--help` text. No other behaviour moves: the depth is still fitted DOWNWARD to the cache allowance and never upward, and a NAMED depth is still never reduced. **This is a DEFAULT change, NOT a gate verdict: no gate PASS is established at K=8 or at any other depth** -- gate 9 is still read at the depth a claiming task declares, and the honest stream's non-echoing certification remains partial (16-gram criterion met, compression criterion not). **Validation, all measured 2026-09-18, each run leaving its own verdict + `.done` sentinel unattended:** build of the `llama-layer-streaming` target clean; host-only tests **#49 route-topk and #50 host-mem pass 2/2**; bare flag on the MoE vehicle (512-anchor prompt, `--layer-slots 4`) resolves to **K=8** -- `default depth K=8 fitted to the device (4582408192 B of a 17149460480 B cache allowance)`, `40 layers x 64 rows/layer = 4370.1 MiB private VRAM, K=8`, `rowset_init` `"retain_k": 8`, `--check-telemetry` OK, peak VRAM 10161.56 MiB, exit 0; the ceiling still fits DOWNWARD -- the same bare flag with `--reserve-mib 14000` provisioned `the largest fitting depth K=5 (2864250880 B)` at 2731.6 MiB, exit 0 (K=6 needed 3436314624 B against a 3006267392 B allowance, so K=5 is the correct largest fit); and a NAMED depth is still never reduced -- explicit `--residency-tg-retain 8` under that same allowance refused loudly with `run_refused` `reason: row_cache_depth`, `bytes_needed 4582408192` vs `bytes_allowed 3006267392`, exit 1. **Consumers checked:** no CTest target and no driver uses the bare flag; the only script that does is `tools/run-tests.ps1`'s `retention` semantic-QA smoke mode, whose depth therefore changes K=4 -> K=8 -- so a retention-mode rung recorded before 2026-09-18 ran at a DIFFERENT depth than one recorded after, and the two are not like-for-like (its comment now says so). **SEMANTIC-QA GATE CLOSED, and the depth change is now proven to alter only what is STAGED, not what is routed.** The default change is a RUNNER change, so the T-GPLR-07 smoke tier applied and was owed; it has now been run on the changed path plus a two-arm depth comparison, all three under one mutex hold each and all three leaving their own verdict + `.done` sentinel unattended. (1) **Smoke tier, 512-token anchor, retention mode** (retention mode uses the BARE flag, i.e. exactly the code that changed): `build/llama-cuda/gplr07-qa-512-retention.txt` -- all five needles found, checker OK, aggregate **PASS**, and the must-fail negative control **FAILS as designed**, so the rung line is the designed `512-token rung PASSED (check-telemetry OK, answer PASS, negative control FAILs)`. The rung confirms the resolved depth is **K=8** (the changed path itself, not a fitted-down depth). (2) **Depth A/B, same prompt, K=4 vs K=8** (`build/llama-cuda/qa-depth-k{4,8}.{jsonl,log,txt,verdict.json,done}`, both `PASS 0`, exit 0): **greedy text is byte-identical** (`sha256 ce75e0dca52da35b...`, 170 B in both) and the **route digest is identical** (`f9056fdc30a5c52b...` over 3,840 `rowset_fetch` events, 96 rounds, 0 refusals in both) -- so changing the depth changes neither the model's routing nor its output at this shape. (3) **What the depth DOES change, measured:** mean `bytes_staged` per `rowset_round` is **210,798,592 B at K=4 vs 141,943,808 B at K=8, i.e. 32.7% less staged per round at K=8** -- the depth buys its TG gain by staging less, which is the mechanism the SDD claims, now shown directly on a non-echoing-equivalent pair rather than inferred from the A/B. **Disclosed method defect, corrected in place:** the first pass of this comparison scripted the staging field as `staged_bytes` and divided by zero; the real field is **`bytes_staged`** on the `rowset_round` event, and the corrected read is the 32.7% above -- the identity checks were unaffected and had already returned True on the first pass. This is a default change with a satisfied smoke gate, **not** a gate verdict that K=8 is the right depth: gate 9 remains read at whatever depth a claiming task declares, K=8's honest TG gain still rests on two B samples (spread 2.013 tok/s), and the honest-vs-echoed comparison remains confounded by prompt length. Evidence: `build/llama-cuda/gplr07-qa-512-retention.{jsonl,log,txt}`, `build/llama-cuda/qa-depth-k{4,8}.{jsonl,log,txt,verdict.json,done}`. **Discarded, disclosed:** a first squeeze attempt with `--reserve-mib 19600` tripped the T-GPLR-04 capacity preflight before the row cache was fitted; its artifacts were deleted rather than kept, because it measures nothing about this default. Evidence: `third_party/llama.cpp/tools/layer-streaming/main.cpp`; `build/llama-cuda/bare-default-k.{jsonl,txt,log,done,verdict.json}`, `fitdown-A2.*`, `explicit-refuse-B.*`; records in `docs/layer-residency-sdd.md` (2026-09-14 default marked SUPERSEDED + new 2026-09-18 default amendment; section 4 shape (i) table) and `docs/plans/generic-prefill-layer-residency-plan.md` ("T-GPLR-26 FOLLOW-UP").

## 2026-09-18: DETACHED-JOB CONVENTIONS CONSOLIDATED into one section of `docs/AGENT_ENVIRONMENT.md` ("Detached runs and waiting"), so the next session does not rediscover them. **Why:** the facts were scattered across addenda E-G and two of them were rediscovered the expensive way the same day -- a detached `ctx_shell` job given a RELATIVE `-File` path failed in under a second (a background job does not take the call's `cwd`; it resolved against `$HOME`), and two unconditional `sleep 170` waits blocked a turn, achieved nothing, and were interrupted twice while the job they waited on needed no attention at all. **What the section records:** the two questions and their two sources -- *did it start* is answerable only by the launch handshake (the `job_id`, observed form `shell_<16 hex>`), *did it finish and how* only by the `.done` sentinel, because an absent sentinel looks identical in both cases; `run_in_background: true` with `timeout_ms` (max 3600000 ms) after the ~110 s foreground cap (documented tool behaviour, not measured here); ABSOLUTE paths required in background commands; status/cancel mechanics with the observed states `running`/`completed`/`failed` and the structured lookup error for an evicted job; the RULE THAT A WAIT MUST BE CONDITIONAL AND PRODUCTIVE, with the `sleep 170` case as the measured negative example and the bounded-loop and `Wait-ArmVerdict` forms as the accepted shapes; and a copy-paste launch-then-one-check sequence. It states plainly what it does NOT settle: the app terminal's `process_type: "BACKGROUND"` is not implemented in this build, a detached job survives interrupts of the agent's own commands (measured), and none of it makes a GPU run cheap or exempt from the recorded-command rule. Documentation only -- no code, no measurements, no behaviour change; the sentinel mechanism it references is the one committed as `3525f3e`. Evidence: `docs/AGENT_ENVIRONMENT.md` ("Detached runs and waiting", and the pointer added to Addendum F's `ctx_shell` paragraph).

## 2026-09-18: EVERY GPU ARM NOW LEAVES A MACHINE-READABLE VERDICT plus a `.done` SENTINEL, so a launched run does not have to be watched -- one file check replaces polling and log tailing. **Why:** a run's outcome lived only in a text log, and the absence of a result was indistinguishable from a run still in progress. That cost real time twice the same day -- a blind `sleep 170` poll that had to be interrupted twice, and a crash after the criteria that left no verdict line and no exit code. **The convention (`tools/arm-verdict.ps1`, dot-sourced):** every arm writes `<dir>/<tag>.verdict.json` (schema `arm-verdict/1`: status, exit code, start/end/duration, mutex name, command, log path, `error`, free-form `fields`) and `<dir>/<tag>.done`. **The two rules that make the sentinel trustworthy:** the verdict is written FIRST and the `.done` LAST, so a `.done` implies a complete verdict beside it; and the `.done` is DELETED BEFORE the run starts, so a PASS left by an earlier run at the same depth and vehicle can never be read as the current run's result -- the stale-sentinel case that makes completion files useless in practice. **Status is DERIVED from the exit code** (0 PASS, 124 TIMEOUT, else FAIL); only the termination path sets it explicitly (`ERROR`), and that path still carries a non-zero code plus the exception text. **Where it is wired (three chokepoints, no per-wrapper change):** (1) `tools/with-gpu-lock.ps1` writes the verdict from its `finally`, so EVERY `tools/run-*-locked.ps1` wrapper emits one with no edit to the wrapper itself -- the tag is derived from the command (inner `.ps1` leaf, else `.exe` leaf, else `gpu-command`) and the dir defaults to `build/llama-cuda/verdicts/`; (2) `tools/run-gplr19-rowcache-ab.ps1` acquires the mutex itself, so it writes its own, tagged `rowcache-<vehicle>-k<K>` in the rung root, on the normal path AND from a `trap` that records an uncaught failure as `ERROR` -- an early death can no longer look like "still running"; (3) `tools/smoke-gpu-lock-ctxshell.ps1` writes a CHAIN verdict (`gpu-smoke-ctxshell`, which the inner exit code cannot express: a chain can fail after the arm succeeded) and passes its own dir/tag to the arm it launches (`gpu-smoke-arm`), so the arm command is captured verbatim in the arm verdict. Readers: `Read-ArmVerdict` (returns the verdict or `$null`), `Clear-ArmVerdict`, `Wait-ArmVerdict` (BOUNDED wait that returns the moment the sentinel appears, or `$null` on timeout -- never an unconditional sleep), `Get-ArmVerdictPaths`, `Get-ArmTagFromCommand`. **Behaviour change to declare:** the row-cache driver now exits `2` when criteria 1 or 2 do not pass; a NOT-PASSED rung previously exited `0`, so a failed rung was indistinguishable from a successful one to any caller. No caller depended on the old code (grepped: only the forwarding shim, which already propagated non-zero). Nothing about what a run MEASURES changed -- no arm's flags, criteria or telemetry were touched. **Validation, all of it measured:** `tools/test-arm-verdict.ps1` (CPU-only; takes the mutex only around `cmd.exe /c exit N`, which initialises no CUDA context) passes all 24 cases -- round trip, JSON parse, stale invalidation, status derivation (124 -> TIMEOUT), ERROR text, bounded wait (returns in 0.01 s when the sentinel exists, `$null` at 3.02 s on timeout), all four tag-derivation cases, and integration through `with-gpu-lock.ps1` where the SAME tag flips FAIL 3 -> PASS 0 (proving the invalidate-first rule, not an in-place overwrite). Driver path, non-GPU: `-AnalyzeOnly -RetainK 6 -Vehicle honest` on a COPY of the recorded K=6 root (copied because `-AnalyzeOnly` deletes `driver.log`) left `rowcache-honest-k6.done` = `PASS 0`, exit 0, with `crit1/crit2 true`, `gain_tok_s 9.943`, `staged_bytes_per_round 202913280` in `fields`; the no-traces refusal left `rowcache-honest-k4.done` = `ERROR 1` carrying the FULL refusal text in `error` and `[FATAL]` in `driver.log`. **End to end on real CUDA, launched detached through `ctx_shell` and then deliberately UNWATCHED:** one file check at the end found `build/llama-cuda/gpu-smoke-arm.done` = `PASS 0` (10.8 s, the full arm command in the verdict) and `gpu-smoke-ctxshell.done` = `PASS 0` (15.4 s, `arm_exit 0`, `checker_exit 0`, `fetch_events 320`, `tg_text_bytes 21`, `cuda_path_empty true`), GPU back to 0 MiB -- no polling, no log tail. **Two limits, stated so the sentinel is not over-read:** absence is AMBIGUOUS -- a missing `.done` means "not finished" but cannot distinguish that from "never started", and only the launch handshake can, which was demonstrated the same day when a detached `ctx_shell` job given a RELATIVE `-File` path failed with exit 1 in under a second (its `cwd` does not apply to a `run_in_background` job; it resolved against `$HOME`) while the file check simply said ABSENT **[CLOSED LATER THE SAME DAY by the launch-time start marker -- see the entry at the top of this file: `Get-ArmRunState` now answers `never-started` / `running` / `died` / `finished` from the files alone, and what remains open is only that pid liveness is best-effort]**; and the verdict comes from a termination handler, not a watchdog, so a hard-killed process writes nothing -- correct, since an absent sentinel must never be reported as a passing run. Evidence: `tools/arm-verdict.ps1`, `tools/test-arm-verdict.ps1`, `tools/with-gpu-lock.ps1`, `tools/run-gplr19-rowcache-ab.ps1`, `tools/smoke-gpu-lock-ctxshell.ps1`; artifacts `build/llama-cuda/gpu-smoke-{arm,ctxshell}.{verdict.json,done}`, `build/llama-cuda/tmp-verdict-test/` (a scratch root created for this verification and left where it landed, since the verdict JSONs embed their own absolute paths); recorded in `docs/AGENT_ENVIRONMENT.md` ("Arm verdicts and the `.done` sentinel").

## 2026-09-18: THE NON-ECHOING VEHICLE IS NOW THE DEFAULT for row-cache rungs (one driver, `-Vehicle honest|echoed`), and two defects this surfaced are fixed: the offline analyzer rejected EVERY current-build trace, and a PowerShell 5.1 enumeration trap killed the driver silently after its criteria. **Why the default flipped:** every recorded K rung (K=2/4/6/8) and the K=8 operating depth were chosen on ECHOED gains, and the honest re-run measured 22-24% MORE bytes staged per round at the same shapes (entry below). A default that produces the optimistic reading is the wrong default, so a depth decision can no longer rest on an echoed rung by accident. **The change, one implementation — no second arm loop:** `tools/run-gplr19-rowcache-ab.ps1` takes `-Vehicle honest|echoed`, default **honest**, and the vehicle selects the prompt (`build/calc-prompt.txt` at 885 prefill tokens, vs the self-echoing `anchor-512.txt` at 512), the recorded baseline trace, the offline (b) report AND the output root TOGETHER, so a rung can never pair one vehicle's prompt with another's prediction. Roots: honest -> `gplr19-honest-k<K>-ab`; echoed -> `gplr19-rowcache-ab` (K=1) / `gplr19-k<K>-ab`, unchanged, so the recorded evidence paths and the historical `-AnalyzeOnly -RetainK 1` regeneration stay valid (now with `-Vehicle echoed` said out loud). `-AnalyzeOnly` now FAILS LOUDLY when the requested vehicle has no traces, naming the missing paths and both ways out. Criterion 1 gained IDENTITY: the A arms must reproduce the recorded run's route digest AND greedy ids, not merely its fetch/staged counts. Every run logs the OTHER vehicle's gain at the same K (finding the honest K=8 reference at its nested path), so a rung is never a single reading. K=1's pre-registered tok/s bounds apply only to the echoed vehicle (they were derived from that shape); the honest vehicle's pre-registered quantity is the staged-BYTE prediction. `tools/run-gplr19-rowcache-ab-honest.ps1` is now a forwarding shim (kept so the recorded evidence keeps a resolvable producer path). **DEFECT 1, pre-existing, found by this run: `tools/route_overlap/analyze.py` rejected every current-build trace** -- `unsupported event type: 'async_feed_init'` on all four arms -- because its known-event list predated the T-GPLR-25 async-feed telemetry (`async_feed_init/_submit/_ready`, default-on). Consequence: the offline-(b)-vs-runtime cross-check was UNAVAILABLE for the whole honest run, and would have been unavailable for any future row-cache rung; the failure was also reported as four bare `FAILED` lines that read like a measurement problem. Fixed: the three events are in the analyzer's ignored known list, and the driver now labels a schema-drift failure as ANALYZER/ENGINE SCHEMA DRIFT and prints an explicit `OFFLINE CROSS-CHECK UNAVAILABLE` line naming the cost. **DEFECT 2, introduced by this change and fixed: the cross-vehicle block used `@(Get-Content -Raw | ConvertFrom-Json)`**, which in Windows PowerShell 5.1 does NOT enumerate a top-level JSON array -- it yields ONE object whose properties are ARRAYS, so `[double]$_.tg_tok_s` threw. Under `$ErrorActionPreference='Stop'` the driver died AFTER logging the criteria and BEFORE its verdict line: a run that measured correctly and reported nothing, never writing an exit code. Now a function called inside try/catch with the enumeration done in the form the file's own `$prior` block already used, and a failure there is a labelled WARNING that cannot cost the run its verdict. Also fixed while here: `$outRoot`/`$OutRoot` are the SAME variable in PowerShell (case-insensitive), the collision that nested the honest K=8 evidence. **Validation, all non-GPU except the last item:** parse OK; default-vehicle `-AnalyzeOnly` refuses with guidance; echoed `-AnalyzeOnly -RetainK 1` on a COPY of the recorded root reproduces the recorded K=1 table exactly (criterion 2 **+0.142%**, 396,398,208 vs predicted 395,835,932; gain **+2.068**, spread 0.011; criterion 1 PASS including identity, digest `c4ba88afa5291e82...`); the analyzer cross-check now PASSES on modern traces (`arm B1/B2: offline (b) hits 52,930 = runtime retained rows 52,930`, agree exactly). **Honest K=6 re-run through the DEFAULT path (no `-Vehicle` flag, one mutex hold, root `gplr19-honest-k6-repro`): criteria 1-2 PASS; staged/round 202,913,280 = +0.840% of the pre-registered 201,223,734; `rows_retained` 52,930 = the offline `hits_total` exactly; gain +10.017 (A 10.880 = 10.902/10.858; B 20.897 = 21.126/20.668), within-arm spread 0.458; route digest identity `225c3303e2f94871...` reproduced.** **REPRODUCIBILITY ACROSS SESSIONS, which is the strongest result here:** the morning's honest K=6 (+9.944, spread 0.817) and this run (+10.017, spread 0.458) differ by **0.073 tok/s**, far below both spreads, on a **bit-identical byte stream** (staged/round 202,913,280 both times, same digest, same retained count). The honest K=6 gain is ~+10 tok/s and stable. **Remaining gap, stated plainly:** this is a DEFAULT plus a labelled opt-out, NOT a gate -- nothing mechanically prevents a future rung from being cited alone, and the rule is enforced by the default, the labelling, and the two doc statements (SDD section 4; plan T-GPLR-26 RESULT). Evidence: `tools/run-gplr19-rowcache-ab.ps1`, `tools/run-gplr19-rowcache-ab-honest.ps1` (shim), `tools/route_overlap/analyze.py`, `build/llama-cuda/gplr19-honest-k6-repro/`.

## 2026-09-18: ROW-CACHE A/B ON THE HONEST STREAM — the K=8 depth decision's echo caveat is now MEASURED. K=6 and K=8 re-run on the declared non-echoing prompt (`build/calc-prompt.txt`, 885 tokens); both rungs PASS criteria 1 and 2; the honest (b) prediction reproduces in BYTES within 1.2%; and the tok/s gain does NOT measurably compress (honest minus echoed: **-0.252 tok/s at K=6 and -1.195 at K=8**, against within-arm B spreads of **0.817** and **2.013**). **Why this run:** the owner's operating depth is K=8 and the +12.609 tok/s it rests on came from the ECHOED pairing (SDD section 4 records that as an upper estimate, never validated on an honest stream). **The declared change is exactly ONE: the prompt BODY** -- same substitution the 2026-09-14 re-probe made, every other flag held at the recorded T-GPLR-19 arm values (`--residency moe-expert-bank --residency-tg --pp-schedule batch-major --ubatch 512 --tg-tokens 256 --layer-slots 4 --kv-cache q8 --greedy --no-mtp --pp-no-logits --allow-full-residency`), plus `--pp-tokens 885` to take the prompt whole. One mutex acquisition per rung, four arms A,B,A,B inside each; explicit depths only (a named depth is a contract and is never reduced). **PRE-REGISTERED, and that is what makes criterion 2 a real test:** the honest statistic (b) was computed 2026-09-14 from this same stream, BEFORE this run -- K=6 0.648652 (52,930/81,600) and K=8 0.728333 (59,432/81,600) -- giving predicted staged/round **201,223,734 B** and **155,588,690 B**. **Measured (arm-level: tg tok/s, staged/round, peak VRAM):** K=6 A1 10.868, B1 21.371, A2 11.170, B2 20.554; staged/round 572,719,104 (A) and **202,913,280** (B) -- **+0.840%** of prediction; reactive mean 11.019, retained mean 20.963, **gain +9.944**, within-arm spread 0.817, conversion 8.59 GB/s; peak VRAM A 5.72 GiB, B 8.83 GiB. K=8 A1 11.118, B1 23.565, A2 11.171, B2 21.552; staged/round 572,719,104 (A) and **157,388,544** (B) -- **+1.157%** of prediction; reactive mean 11.145, retained mean 22.559, **gain +11.414**, within-arm spread 2.013, conversion 9.15 GB/s; peak VRAM A 5.72 GiB, B 9.93 GiB. **Criterion 1 PASS at both rungs:** all four arms exit 0 with `--check-telemetry` 0, route stream identical across the four arms, greedy token ids identical across the four arms, and the A arms reproduce the RECORDED honest run on route digest **AND** greedy ids (digest `225c3303e2f94871...`, region fetched 10,240, staged/round 572,719,104, retained 0). **Criterion 2 PASS at both rungs:** the prediction held inside 2% and `rows_retained` equals the offline `hits_total` EXACTLY (52,930 at K=6, 59,432 at K=8) with reuse on **40/40 layers** and **0 refusals** -- the runtime reproduces the honest offline statistic exactly, on a second stream, the same property the echoed rungs had. **Answer stability survives the honest stream too:** all EIGHT arms produced byte-identical greedy text (`sha256 10a0988c65635142...`, 844 B). **What the honest stream changes about the echo caveat, in two parts.** (1) The BYTE-level echo inflation is CONFIRMED and larger than the recorded (b) shift suggests: the honest stream stages **+23.98%** (K=6: 202,913,280 vs 163,659,552) and **+21.94%** (K=8: 157,388,544 vs 129,075,264) more bytes per round than the echoed rung did. (2) The tok/s consequence is **NOT RESOLVABLE at this sample size** -- both deltas sit INSIDE the within-arm spread, so the correct reading is "the honest gain is indistinguishable from the echoed one here", NOT "the echo bias does not matter". The K=8 caveat in the T-GPLR-26 decision is therefore downgraded from *upper estimate, unvalidated* to *measured at +11.414 tok/s and statistically indistinguishable from +12.609 at two samples per rung*. Resolving it needs more B samples, not a different arm design. **CONFOUND DISCLOSED:** honest and echoed differ in TWO ways, not one -- prompt CONTENT and prompt LENGTH (885 vs 512 PP tokens), so the honest reactive baseline is slower (11.019/11.145 vs 11.873/11.879) because TG attends a longer KV. Each rung's A-vs-B gain is a within-session, within-prompt difference and is unaffected, but the CROSS-STREAM gain delta is confounded by baseline token time and cannot be attributed to echo bias alone; a length-matched control (`--pp-tokens 512` on the same non-echoing prompt) is the arm that would separate them. **ARTIFACT-PATH DEFECT, disclosed not repaired:** rung 2 was written to `build/llama-cuda/gplr19-honest-k6-ab/gplr19-honest-k8-ab` because PowerShell variable names are case-INSENSITIVE and the per-rung `$outRoot` overwrote the `$OutRoot` PARAMETER, so rung 2 nested under rung 1. The evidence is left where it landed (not moved, so no artifact metadata was edited after the fact); the driver is fixed -- the per-rung variable is now `$rungRoot`, verified to resolve both rungs to sibling directories -- so a FUTURE `-Rungs 8` run writes to `build/llama-cuda/gplr19-honest-k8-ab`. Do not confuse those two paths. One earlier launch died harmlessly at the range guard because `-Rungs 6,8` binds the literal string to an int[] (comma consumed as a digit separator -> 68); the parameter now takes a string and splits it. Evidence: `tools/run-gplr19-rowcache-ab-honest.ps1`; `build/llama-cuda/gplr19-honest-k6-ab/{driver.log,summary.json,arm-*.jsonl,arm-*.txt}` and the same set under the nested k8 path; rung roll-up `build/llama-cuda/gplr19-honest-summary.json`.

## 2026-09-18: T-GPLR-26 CLOSED — ADVISORY (branch B, documents only, ZERO source changes), and the TG retention depth becomes OPERATOR-NAMED PER RUN at K=8 (owner: "i take the more t/s option"). **The pick:** the row-set gate-eligibility reading ((b) versus the 0.6 line) BINDS NOTHING — no run, no depth and no provisioning is admitted, refused or reduced by it. It stays advisory permanently until an admissible oracle exists (tracked (b)-vs-K surface for a DECLARED workload class with per-stream provenance; pre-registered verdict rule fixing which stream length/W/retention rule produce a verdict; stated behaviour for a depth the surface lacks). Branch A was NOT found wrong — it was found to have no work to do at the chosen depth: on the honest (non-echoing) curve 0.6 is cleared by K=6 (0.6487) and K=8 (0.7283) and missed by K=4 (0.5355), so a binding rule would refuse the old default, not the depth the owner now names. **The depth:** owner takes the most tok/s, which is **K=8** — `--residency-tg-retain 8` (requires `--residency-tg`; bare flag stays the device-fitted K=4 CEILING; a NAMED depth is a contract and refuses loudly via `row_cache_depth` rather than being silently reduced). Measured ladder (512-anchor vehicle, 256 TG tokens, `--layer-slots 4`, reactive baseline 11.863-11.967 tok/s held across five GPU sessions; `currentstate.md` 2026-09-13 T-GPLR-19 rungs, traces `build/llama-cuda/gplr19-k<K>-ab/`): **K=1 546.8 MiB +2.068; K=2 1093.0 MiB +3.853; K=4 2185.4 MiB +7.865; K=6 3277.8 MiB +10.196; K=8 4370.1 MiB +12.609 tok/s (24.481 vs 11.873 = ~2.06x)**. **Three limits recorded WITH the pick so it cannot be read as an endorsement:** (1) the gains are the OPTIMISTIC echo pairing — that A/B's `(b)off` column is the echoed series, so a non-echoing stream compresses them by roughly the 0.06-0.09 (b) shift and +12.609 is an upper estimate, unvalidated on an honest stream; (2) the row cache is a TG-phase residency class — no depth changes prefill, so this pick does not move the program's primary PP objective; (3) the byte-to-time conversion DEGRADES with depth (14.29/12.18/10.67/10.52/10.23 GB/s; marginal 8.93/8.35/9.55/7.69), so K=8 is the most tok/s and the least efficient per byte — only the first retained window converts above the ~13.7 GB/s link ceiling, and K=6 is the best trade if the owner ever prefers gain-per-MiB. **Why the 0.6 reading stays advisory rather than binding on the recorded numbers:** the only two streams on hand were measured at the wrong size for policy-making and (b) is monotone in stream length, so a bind could refuse a productive run after one prompt change — while an unproductive depth still shows visibly in the `slot`/`other` ledger buckets and `rows_reused`. **Deliverable:** documents only — `docs/layer-residency-sdd.md` (header amendment; new section 4 depth amendment with the full ladder and both limits; new section 10 provisioning-status note; section 11 gate 9 wording unchanged) and the active plan (T-GPLR-26 Status, the decision block kept as the record, plus a RESULT block). `git diff` for this task contains no source change, so the host suite is unaffected. Evidence: plan block `T-GPLR-26 RESULT (2026-09-18)`; SDD sections 4 and 10.

## 2026-09-18: HOST-MEMORY INSTRUMENT — the T-GPLR-36 open question now HAS an instrument (vendored tree COMMITTED c5d79ad5c), and the T-GPLR-39 route-selection fix is extended to the row-set ROUTING path. **Why:** the 200k ledger entry records "the owner's observed host-RSS growth (19 -> 30 GB) has NO instrument in this run and is recorded as an open question with a proposed monitor probe, not as a result"; the existing `tools/monitor-mem-gplr36.ps1` samples the process from OUTSIDE at 5 s, so it yields a time series and can never attribute. **What was built:** a new `host_mem` telemetry record, emitted twice per PP pass (`pass_begin` / `pass_end`) under `--pp-recon` exactly like the T-GPLR-36 ledger, carrying process working set, private (anonymous) resident share and the derived file-backed/shared share, private commit, the OS lifetime peak and this run's peak, deltas since the previous sample and since the run's first sample, system free/total RAM and commit, and `eng_bytes` — the engine's OWN accounted host holdings at that instant (per-slot pinned staging + persistent pinned pool + registered mmap spans, read from the live allocations). **The attribution axis is `ws - ws_private`:** the runner's own anonymous allocations versus the mapped model's file-backed pages; `eng_bytes` then says whether an anonymous delta is the residency machinery or something else. The baseline deliberately survives `reset_stats()` (the runner resets stats at every pass boundary; cross-pass growth is the point), and `llama_ls_host_mem_sample()` sits in its own TU (`src/llama-layer-streaming-host-mem.cpp`) because `<windows.h>` defines min/max/small as macros while the main source uses std::min/max and locals named `small`; a platform without the counters reports `ok == false` rather than a fabricated zero. **Validation (host-only, measured, no GPU):** `tests/test-layer-streaming-host-mem.cpp` CTest #50 PASS — field invariants plus the REACTIVITY property (a touched 256 MiB allocation moved the working set by exactly +256 MiB, commit +257 MiB), which is what a constant, a cached value, or a wrong-process query would fail; CTest #49 (`test-layer-streaming-route-topk`) still PASSes; llama.dll builds clean with the new TU. Checker safety is established, not assumed: unknown non-empty event types are ignored by both validators (`check_telemetry` has no unknown-event error path; `validate_telemetry` errors only on an EMPTY event field) — the same property `host_ledger` and the T-GPLR-39 sub-buckets already relied on, and empirically the T-GPLR-39 recon trace carrying the unknown `host_ledger` event passed the strict checker. **Still UNMEASURED:** no GPU arm has yet run with the instrument, so the 19 -> 30 GB climb remains unexplained; it is now measurable and attributable per pass. **Second change (T-GPLR-39):** `read_router_ids` — the row-set ROUTING decision memcpy'd into `sel_in`, not telemetry — now uses the same `llama_ls_route_topk_row`, so `stable_sort` is gone from the module entirely. Equivalence is argued (bit-identical ids; `probs_buf` is written but never read at that function's single call site, so `ids_per_token` is the only live channel) and host-tested (34/34 cases, incl. 6 row-set cases that also compare the caller's `sel_in` packing against the pre-change derivation); end-to-end GPU confirmation is unmeasured. Evidence: build logs `build/llama-cuda/gplr36mem-{cfg,build}.log`; plan blocks `T-GPLR-39 implementation arm` and `T-GPLR-36 host-memory instrument` in docs/plans/generic-prefill-layer-residency-plan.md. **Committed in the vendored tree as `c5d79ad5c`; the plan/currentstate/tooling set is committed in the outer change that also bumps the gitlink.** **Owner decisions the same day:** (1) **T-GPLR-27 CLOSED — DEFERRED** (owner verbatim: "I do not need to prove genaralization now, note it and close it") — recorded in the task's heading and Status line as intentionally not pursued, with no generic-architecture claim implied; (2) today's uncommitted work is **HELD until the host-memory arm reports**, so the commit carries measured evidence rather than host-only validation; (3) T-GPLR-26 and T-GPLR-37 were re-explained at the owner's request (they are not recalled from the 2026-09-14 brief) and their picks are still open. **THE HOST-MEMORY ARM RAN (tools/run-gplr36-hostmem.ps1 / -locked.ps1, one mutex hold, GPU otherwise idle -- the other project released first): runner exit 0, strict --check-telemetry OK, pipeline gate PASS (shape-mismatch skip), pass 1 139,073.4 ms / 1,438.090 tok/s, pass 2 199,426.3 ms / 1,002.877 tok/s, peak VRAM 3,265.56 MiB (the known defect reading again).** The 19 -> 30 GB host climb is now EXPLAINED at this shape and it is not a leak: working set 4.13 -> 25.05 GiB over the run (+20,927.1 MiB), of which **83.5% is file-backed/shared (+17,484.2 MiB) and 16.5% private (+3,442.9 MiB), while the engine's own accounted host holdings are FLAT at 1,494.0 MiB (d_eng = 0 at every sample)**. The file-backed share lands almost entirely in PASS 1 (+17,484.9) and is dead flat in pass 2 -- the pinned staging path memcpy's every package member out of the model mapping, so the first pass touches essentially the whole 19.92 GiB GGUF; those pages are reclaimable model pages. The anonymous residue (+1,845.3 in pass 1, a -1,589.6 SHRINK at the boundary, +3,186.5 in pass 2) is the sawtooth of a buffer freed and re-allocated across the boundary, in the same band as the host-SPILLED KV byte_account reports at this shape (3,910 MiB), which the engine deliberately does not account. **Cross-instrument check: the external monitor measured 23,458 MiB max working set on a 200k arm (T-GPLR-37c) and this instrument measured 23,457.1 MiB at pass 1's end -- two instrument families agreeing within ~1 MiB.** The 30 GB ceiling does not appear at this shape (measured max 25.05 GiB), so reaching it needs more ANONYMOUS demand (deeper KV spill, more slots, a pinned pool), not more mapping. Disclosed instrument gaps demonstrated by this arm: d_ws clamps negative deltas (the boundary shrink shows only in g_ws); two samples per pass bracket rather than resolve a within-pass climb; eng_bytes covers this module only. **Separate anomaly, recorded not explained:** the warm pass was 43% SLOWER than the cold pass here (199.4 s vs 139.1 s), the reverse of the recorded ledger reference for this shape (cold ~146 s, warm 141.8 s) -- needs replicated adjacent pairs before interpretation. Evidence: `build/llama-cuda/gplr36-hostmem-200k.jsonl` (4 host_mem samples), `gplr36-hostmem-wrapper.log`, `tools/analyze-gplr36-hostmem.py`; plan block `T-GPLR-36 host-memory instrument -- FIRST REAL ARM`.

## 2026-09-17: T-GPLR-43 QA-SMOKE GATE — the audit's CORRECTNESS gap on the --pp-repeat default rebuild is CLOSED; both rungs PASS all four steps. **Standard suite (tools/run-tests.ps1 -Suite qa-smoke, the recorded 512 parity shape, batch-major/no-handoff, q8 KV, MoE vehicle):** runner exit 0; strict --check-telemetry OK (30503 events, 3 slots); answer checker aggregate=PASS 5/5 (QX-7741 TR-2918 MK-5501 LD-8834 Paris); negative control aggregate=FAIL exit 1 as designed. **Repeat-2 arm (tools/run-gplr43-qasmoke-repeat2.ps1 — the SAME recorded shape + --pp-repeat 2, which is what actually engages the default per-pass rebuild):** runner exit 0; two pp phase summaries = both passes completed (pass 1 97.1 tok/s / pass 2 199.8 tok/s warm, graph_builds 0 as designed — the rebuild advances build_generation, not the timed-phase counter); strict checker OK (30819 events); answer PASS 5/5; negative control FAIL as designed. => answer stability holds across the pass boundary with quantized KV and the rebuild engaged. **Tooling finding (pre-existing, not fixed here):** the qa-smoke suite wrapper stalls at the strict-check step when run detached under a UTF-16 `*>` redirect (checker output never flushes to the buffer; the checks themselves are fine — every step graded equivalently by direct invocation). The suite's own recorded shape does NOT exercise --pp-repeat > 1; the repeat-2 mirror arm is the added coverage and is now part of the evidence. Gates are soft: this proves answer stability on the 512 rung only, not general fitness. Evidence: build/llama-cuda/gplr07-qa-512.{jsonl,txt,log}, gplr07-qa-512-repeat2.{jsonl,txt,log}, wrapper log build/llama-cuda/qasmoke-wrapper.log. Nothing committed beyond the runner script.
## 2026-09-17: T-GPLR-43 UNIFICATION — act_bind emission has a SINGLE owner. Follow-up to the audit's design finding on the fix: emission previously lived in begin_set (llama-layer-streaming.cpp, non-pw layer-major path) AND a second emitter in llama-context.cpp (pw path, gated by the pw_bind_emitted latch). **Change (3 files):** begin_set now owns emission in both modes, still guarded by set_created (a per-pass rebuild stays silent by construction); in pw mode — one set per width class whose static slices sit at offset 0 — it emits the per-chunk cumulative-offset walk on chunk 0's creation (always the first set the layer-major prebuild builds), preserving the trace contract exactly (every chunk bound once per slab at distinct non-aliasing offsets, width-proportional); the llama-context.cpp pw emitter is deleted; the pw_bind_emitted latch and its header member are removed (one fewer piece of process state). **Validation (tools/run-gplr43-diag.ps1, 4k x --pp-repeat 2, q8):** act_bind count 16 = 8 chunks x 2 slabs x ONE pass (was 32 across two passes under the old dual-emitter semantics); 2 binds per chunk, no slab-slice aliasing, cumulative offsets to 14,680,064 B, both pp passes completed; strict checker reports ZERO act_bind violations (its single remaining violation is the documented T-GPLR-39 rule that fires only in TG-less shapes: run_start declared handoff_mode auto-on but --tg-tokens 0 produced no handoff events). Build clean for the touched tree (the only failure in the full build sweep is the pre-existing, unrelated test-unicode LNK2019 in the vendored tests). Compute paths, the between-pass rebuild policy, and the checker are untouched. Evidence: build/llama-cuda/gplr43-diag.jsonl/.log. Nothing committed.
## 2026-09-17: T-GPLR-43 FIX — the q8 144k pass-2 crash is FIXED and the regression is CLEAN. Mechanism (proven by the T-GPLR-43 probe, adjacent arms): graph-reuse-across-clear — pass 2 fails only while executing prebuilt graphs whose executed state was derived at prebuild time against the PRE-clear quantized KV cache; rebuilding the per-chunk graph sets between passes makes pass 2 survive (7.4 ms cost), while the adjacent rebuilt-off q8 control crashed at the byte-identical signature. **Fix (default-on):** `--pp-repeat > 1` now re-runs the prebuild between passes (`--pp-prebuild-per-pass`, explicit `0` restores the old behavior for evidence reproducibility); refused under `--mtp`. **Telemetry artifact resolved (forensic correction to the probe's first explanation):** the 564 checker act_bind violations were NOT the begin_set rebuild re-emitting (that guard was inert: pw_probe is the layer-major default and begin_set's emission is pw-gated) — they came from the pw-mode act_bind emitter in llama-context.cpp (T-Q38PW-00), which fired on EVERY prebuild, i.e. every pass (141 chunks × 2 slabs × 2 passes = 564; the 4k diag showed 8×2×2 = 32). The pw emitter now latches once per process (`pw_bind_emitted`); the diag `[ls-set]` stderr probe removed. **Regression (one mutex hold, default rebuild): B1 q8 / C1 q8-tail / D1 f16 ALL SURVIVE both passes** (last launch chunk 140, layer 39, gen 28 in each) with strict checker OK and zero act_bind violations; B1 walls 78,102.6 ms / 1843.7 tok/s (pass 1) and 111,000.5 ms / 1297.3 tok/s (pass 2). The earlier practical guidance is RESCINDED: long-context --pp-repeat with quantized KV is safe again on this build. Evidence: wrapper log `build/llama-cuda/gplr43-fix-wrapper.log`, traces `build/llama-cuda/gplr43-fix-*.jsonl`, diag `gplr43-diag.jsonl`; scripts `tools/run-gplr43-fix*.ps1`. Nothing committed.
## 2026-09-17: T-GPLR-42 Q4 DISCRIMINATOR — the pass-2 crash is QUANTIZED-KV-GENERAL, not q8-specific. Both arms (B1 shape: 143360 = 140×1024, u1024, 3 slots, mmap-registered, af=1, --pp-repeat 2) under one mutex hold on the free GPU: **q4 arm crashed in pass 2** at chunk 130, layer 3, slot 0, generation 16 (pass 1 complete: 143,013.2 ms / 1006.9 tok/s), and the adjacent **q8 confirm arm crashed at the byte-identical position** (pass 1: 144,543.2 ms / 996.2 tok/s). KV placement CUDA0 in both (791.72 MiB q4 / 1495.47 MiB q8). Combined with T-GPLR-40 (stock survives q8) and T-GPLR-41 (lifecycle facts identical across passes): the trigger is machinery shared by Q4_0 and Q8_0 KV but absent for f16, engaged across the layer-major pass boundary — candidates: quantized-KV buffer/view handling across clear+refill (incl. the [TAG_V_CACHE_VARIABLE] transposed-V path), quantized attention/dequant kernels interacting with pass-2 graph reuse, buffer-clear semantics on quantized KV buffers. Sharpest known fact: **f16 survives, any quantized KV crashes at the same deterministic position.** Next discriminators (owner decision): rebuild per-chunk graph sets between passes (isolates graph-reuse-across-clear from memory-state mechanisms), or bisect the fixed crash position. Traces: `gplr42-q4-disc.*`, `gplr42-q8-confirm.*`; runner `tools/run-gplr42-q4disc-arms.ps1`. Nothing committed.

## 2026-09-17: T-GPLR-41 MCTX RECON — the stale-reference hypothesis is REFUTED (falsifier fired); crash reproduced under instrumentation with a complete pass-2 lifecycle picture. **Instrument:** new opt-in `--pp-mctx-recon 1` emits, per chunk per pass in the layer-major mctx loop (after apply, before the sweep): one `mctx` event (live n_kv via `get_n_kv()`, recurrent rs_z, selected set's chunk_id/rs_z/n_out) and one `mctx_kvview` event per exec layer (baked cache_k_l/cache_v_l view shapes ne[2] + KV storage/view device pointers). Refused under `--mtp`; unknown event types ignored by both checkers (f16 arm checker OK, 14878 events). **Run:** B1 shape (143360 = 140×1024, u1024, 3 slots, mmap-registered, af=1, --pp-repeat 2) q8 + f16 under one mutex hold. **q8 arm: crash reproduced exactly** (exit -1073740791, illegal access detected at layer 3 / slot 0 / chunk 130 in pass 2 — the recorded signature) WITH all 141 pass-2 mctx events already in the trace. **Measured: every lifecycle fact identical between passes** — live_n_kv (144128 padded at chunk 130 in both), rs_z (-1 in both), set selection (chunk_id 1, n_out 0 in both), all 5640 baked KV-view shapes, all KV storage/view pointers (no reallocation, no defrag, no cell relocation across the clear). => stale shapes/pointers/set-selection/changed-n_kv CANNOT explain the crash; the mechanism lives outside what this instrument reaches. **Also excluded:** gross timing sensitivity (the recon's ~40% wall overhead did not heal the run). **Discriminating observation:** pass 1 starts from fresh cudaMalloc garbage, pass 2 from memset-zeroed KV — a data-dependent mechanism predicts pass 1 fragile; the opposite happens. **Remaining suspects for owner prioritization:** recurrent/GDN-side state across the boundary, scheduler CUDA-graph capture state persisting across the clear, async-feed/copy-stream interaction; cheapest next discriminator = q4-KV arm (crash ⇒ quantized-KV generally; survive ⇒ q8-specific). No fix attempted (per pre-registered escalation). Traces: `gplr41-{q8,f16}-mctx.jsonl/.log`; analyzer `tools/analyze-gplr41.py`; runner `tools/run-gplr41-mctx-arms.ps1`. Nothing committed.

## 2026-09-17: T-GPLR-40 BISECT — the 144k pass-2 crash is ATTRIBUTED: q8 KV cache + long context + the pass-2 clear/reuse path; tail chunk and ubatch EXONERATED. **Outcome matrix (all --pp-repeat 2, u1024, 3 slots, mmap-registered, 144k file; traces `build/llama-cuda/gplr40-*.jsonl`):** B1 no-tail (pp-tokens 143360 = 140×1024 exactly) / q8 KV → CRASH in pass 2 at chunk 130, layer 3, slot 0, generation 16 — byte-identical signature to the tail-bearing original arms; C1 tail (144000, 640-token tail) / q8 KV → same crash, same position; **D1 no-tail / f16 KV → SURVIVES both passes** (generation reached 28 = complete pass 2; warm 140,810.8 ms / 1022.7 tok/s — the first clean warm 144k reference; cold 134,497.1 ms / 1070.7). **Conclusion:** the trigger is the quantized (q8) KV cache under the pass-2 clear/reuse path at long context — NOT the tail chunk (exonerated by B1), NOT ubatch size (exonerated by identical crash position at u1024 and u2048 in the original run), NOT the staging mode (f16 staging identical). Consistent with history: 4k repeat runs (q8, tiny ctx) survived; 200k repeat runs (f16 KV) survived. **Defect ownership: RESOLVED as RESIDENCY-OWNED (2026-09-17 stock-probe addendum).** The deciding experiment ran: a standalone stock-libllama probe (no layer-streaming, no residency module — `tools/gplr40-stock-probe.cpp`) filled a q8_0 KV cache with the same 144,001 tokens (u1024, ctx 144128), called the identical pass boundary `llama_memory_clear(mem, true)`, and refilled — and SURVIVED (exit 0); the f16 control survived too. Placement validity was checked against the failing configuration: the crashing residency runs' own logs show the KV fully VRAM-resident on CUDA0 (1495.47 MiB q8 / 2815.00 MiB f16) — the same placement the probe used, so the control matches on the axis that matters. By the pre-registered matrix, the defect is NOT upstream: it lives in the layer-residency pass-boundary reset or its q8-KV interaction (candidate mechanism, unproven: the controller holds per-chunk `chunk_mctx` memory contexts across the pass while the runner clears memory between passes — stale q8 cell references in pass-2 graph inputs fit the fixed-position async illegal access early in pass 2). **Practical consequence meanwhile (unchanged, now with a cause):** at long context, run --pp-repeat with f16 KV or --pp-repeat 1 until the reset path is fixed. Next step (owner decision): fix the residency pass-boundary reset, then re-run the B1/C1/D1 matrix as the regression. **Practical consequence meanwhile:** at long context, run --pp-repeat with f16 KV (the q8 savings matter most exactly where this crashes, so long-context repeat runs should either use f16 or --pp-repeat 1 until fixed). **Process notes:** two aborted attempts (ctx must cover the whole 144,001-token file regardless of --pp-tokens; the generator overshoot again) are documented in the runner. Nothing committed beyond the bisect scripts.

## 2026-09-17: T-GPLR-38 ADDENDUM — 144K UBOATCH A/B: INCOMPLETE, and it found a real pass-2 crash. **Requested as a 100k test; the generated anchor tokenized to 144,001 tokens (generator overshoot, achieved=unverified — flagged), so the arms ran a 144,000-token prefill (ctx 144256).** Two flag-traps hit first (both now documented): the needle generator's target-token count is NOT verified against the tokenizer, and at this KV size the handoff fit gate measures auto-OFF so `--tg-tokens 96` refuses (layer-major TG cap); arms ran `--tg-tokens 0`. **Result: BOTH arms crashed identically in pass 2 of `--pp-repeat 2`** — CUDA illegal memory access in `ggml_backend_cuda_synchronize`, at the same trace position in each: chunk 130/141 (u1024) and chunk 60/71 (u2048), both at layer 3, slot 0, generation 16, i.e. deterministic and ubatch-independent. Neither trace has a second phase_summary; the strict checker rejects both (no completed warm pass). **The partial data (pass 1 only, COLD, arm order confounded — not a valid A/B):** u1024 80,631.8 ms / 1785.9 tok/s vs u2048 76,972.0 ms / 1870.8 tok/s (−4.5% for u2048, unproven). Route telemetry cost at this shape: 4.39 s / 4.31 s per pass (~3%), consistent with the T-GPLR-39 scaling (80 events × 512×n_tok readback + sort). **What the crash means:** all previous `--pp-repeat 2` runs used chunk counts that divide the prompt exactly (8×512, 1×4096, 391×512, 141 chunks at u1024/200k? — no: 200k used ubatch 512 = 391 exact). The 144k arms are the FIRST with a tail chunk (640 tokens) under layer-major + repeat; both died at the same pass-2 slot-generation point, which smells like the pass-boundary reset interacting with the tail chunk's graph/act-slab binding, not with ubatch size. **Status:** the ubatch question at long context remains UNANSWERED; the crash is a new defect (candidate task: bisect exact-multiple pp-tokens 143360 vs the tail, then fix the reset path). Nothing committed; traces + logs preserved as crash evidence. [SUPERSEDED by the T-GPLR-40 bisect above: tail chunk exonerated; trigger is q8 KV + pass 2.]

## 2026-09-17: T-GPLR-39 RECON — the ubatch-size residue is ATTRIBUTED: `tel_route` (route telemetry: device→host probs readback + host sort), not the launch loop. **Instrument (nested tree, opt-in `--pp-recon`):** the T-GPLR-36 `host_ledger` now splits `other` into `route_us` / `tel_us` / `loop_us` (closure exact by construction; new fields, checker-agnostic). **Recon arm (tools/run-gplr39-recon.ps1, u2048, 3 slots, ron mode, 4k anchor, --pp-repeat 2, checker OK; trace `build/llama-cuda/gplr39-u2048-recon.jsonl`):** warm pass other = 1152.8 ms, of which **route 1161.5 ms (100.8%), tel 1.0 ms, loop −9.7 ms** (80 route events/pass at 2-chunk stride × 40 layers). The pre-registered prediction ("route dominates, ~500–800 ms") is CONFIRMED in mechanism and even exceeded; the falsifier (loop carries it) did not fire. **Cross-check vs u512 (T-GPLR-38):** u512's other was 237.9 ms with the SAME 80 route events but 4× smaller readbacks (512×512 f32 = 1 MiB vs 512×2048 = 4 MiB per event) and 4× fewer sort comparisons — the route cost scales with n_tok as predicted (≈14.5 ms/event at u2048). **Consequence (the fix is now obvious and does NOT touch the host loop):** tel_route is DIAGNOSTIC telemetry, not model work — its readback stalls on the compute stream (ggml_backend_tensor_get) and its host sort is O(n_tok × 512 log 512). Options in order of invasiveness: (a) route-sampling stride already exists (`--telemetry-chunk-stride`); a route-HEAD sampling (first 64 tokens per sampled chunk) or F16 readback would cut route_us ~32–64×; (b) move the sort to the GPU / drop the hash; (c) accept the cost in exchange for the route-integrity evidence (the route events back the exact-routing invariant). The honest read: the u2048 "regression" is partly a telemetry-measurement artifact — PP math itself got faster (compute 1862.6 → 1282.7 ms); ubatch 1024 remains the measured sweet spot with telemetry on. **Process note (disclosed):** the first recon attempt used `--tg-tokens 0`, which skips the handoff migration block entirely while the fit gate still reports auto-on → checker FAIL (`auto-on but no handoff events`) — a real flag-combination trap, now documented here and in the runner; the fixed arm passes. Nothing committed.

## 2026-09-17: T-GPLR-38 UBATCH SWEEP (4k anchor, 3 slots, ron mode) — the prediction inverted at 2048: warm wall 2317.5 (u512) -> **2224.3 (u1024, −4.0%)** -> **2510.6 (u2048, +8.3%)** -> 2435.2 (u4096); tok/s 1767.4 -> 1841.4 -> 1631.5 -> 1682.0. **Run (tools/run-gplr38-ubatch.ps1, four adjacent arms, one mutex hold, all four strict checkers OK; traces `build/llama-cuda/gplr38-u*.jsonl`):** consumer wait FLAT (67.7-68.0 ms at every ubatch — the pipelining-granularity concern did not materialize), peak VRAM IDENTICAL 3265.56 MiB (the sweep probes the 3-slot window; buf_act deltas of 24->192 MiB are inside the shared-pool slack), ledger `compute` falls monotonically 1862.6 -> 1525.1 -> 1375.0 -> 1282.7 ms (fatter matmuls, as expected) — but ledger `other` EXPLODES 237.9 -> 484.5 -> 962.1 -> 959.0 ms at u2048/u4096, swallowing the compute savings and then some. **Where the win went (measured):** u1024 keeps the best of both (−334 ms compute, +247 ms other, net −93 ms wall); at u2048 the other-bucket growth (+724 ms) exceeds the compute saving (−488 ms). The other bucket is the T-GPLR-36 "loop body + sampled telemetry + route lookup" residue, per-launch cost now clearly ubatch-dependent (larger per-launch graph inputs/outputs to snapshot and patch), not launch-COUNT-dependent. **Output check:** greedy TG text byte-identical at u1024/u2048; u4096 DIFFERS (262 vs 305 bytes — the answer is cut short), consistent with a ubatch-related TG interplay, uninvestigated; this is an identity sanity check, NOT the semantic-QA gate. **Gates:** absolute profile PASS at all four; paired runs printed but correctly refuse same-config pairs (the moving axis is ubatch, which the gate does not model) — the paired throughput numbers are still recorded (1.042x / 0.923x / 0.952x). **Verdict vs the pre-registered prediction:** WRONG at u2048 (predicted best ~2050-2200 ms there; measured 2510.6, the second-worst arm) and its falsifier "wall worsens at 1024+ -> overhead model wrong" fired in the informative direction: the per-launch overhead model was launch-COUNT-based, but the residue is ubatch-SIZE-based. **Consequence:** the current sweet spot is ubatch 1024 at this shape (−4.0% warm wall, −498 MB… no VRAM cost measured, identical wait); ubatch ≥2048 is a regression under the current host loop. The generic fix (make the other-bucket work batch- or chunk-independent) is the real lever, worth ~0.7 s/pass at u2048 — a candidate task, not done here. Nothing committed.

## 2026-09-17 (addendum 3): T-GPLR-37 AT THE CROWDED SHAPE — the 4k win does NOT transfer: 200k/262k-F16-KV warm walls are FLAT (−0.2% to −0.4%), exactly as pre-registered. **Run (tools/run-gplr37c.ps1, four adjacent arms pin/ron × 2/3 slots, one mutex hold, 4k→200k anchor, `--pp-tokens 262144`, ubatch 512, kv f16, layer-major, af=1, --pp-repeat 2; all four strict checkers OK; traces `build/llama-cuda/gplr37c-*.jsonl`):** warm pass pin-s2 142,165.5 ms / 1406.8 tok/s -> ron-s2 141,853.7 / 1409.9 (**−311.8 ms, −0.22%**); pin-s3 141,666.6 / 1411.8 -> ron-s3 141,134.4 / 1417.1 (**−532.2 ms, −0.38%**). The pre-registered prediction ("within ~1% at both depths, because each bank serves ~391 chunks and af=1 already hides the ~1.8 s pass hop behind 131 s of compute") is CONFIRMED; neither falsifier branch (>3% win, >3% regression) fired. **The mechanism still shows in the ledger — the hop is gone, the wall just doesn't care:** per-bank hop 44.01→0.08 ms (s2) and 44.47→0.13 ms (s3); `slot` bucket 1759.0→1280.6 ms (s2) and **637.3→70.3 ms (s3)**; bank DMA windows stay pure (31.5–31.6 ms = 1.00×); `compute`/`patch`/`other`/`setup` unchanged (setup 221.8→207.2 ms, still the same 200k magnitude as the non-churn 4k band + the chunk-count term). Warm walls are compute-dominated (131.2 s of 141.9 s = 92.5%). **RAM pressure, now MEASURED by the fixed monitor (process-anchored window + sys_free column; CSV `build/llama-cuda/gplr37c-mem-monitor.csv`, 102+ samples covering all four arms):** min free system RAM **19,447 MiB** (of 61.7 GiB) — the box never came within ~19 GiB of pressure with 17.23 GiB pinned AND the ~5 GiB spilled KV resident; max process working set 23,458 MiB; the monitor now covers the whole run including arm boundaries. **Verdict:** the churn-free mmap-registered mode is a SHORT/AMORTIZED-CONTEXT lever (≤ ~8-16k tokens per bank, where staging sits on the critical path), NOT a long-context one; at 200k it costs 17.23 GiB of pinned RAM for ~0.3% wall. The honest default posture remains: pinned staging + depth for long context, mmap-registered opt-in for short prefill; a shape-aware policy could later switch modes on measured prompt length. Nothing here is a fitness claim; gates soft, acceptance is the owner's. Nothing committed.

## 2026-09-17 (addendum 2): T-GPLR-37 CHURN-FREE VARIANT — CHURN CONFIRMED AS THE CAUSE, SETUP INFLATION GONE, AND A LARGE WARM WIN AT 4k AT BOTH DEPTHS. **What changed (code):** `--copy-source mmap-registered` is now the churn-FREE mode — one `register_all_pkg_sources` call runs after the prebuild and BEFORE the timed PP, registering the page-aligned source span of every executed layer once (cached per logical layer, read-only afterwards); staging then never touches page tables during a pass. The V1 per-bank churn stays reproducible as `--copy-source mmap-registered-churn`, and the mode refuses loudly (driver text, `abort_reason=mmap_register`, run_refused) if any registration fails or a pass finds an unregistered span. **A/B (4 adjacent arms = pin-s2, ron-s2, pin-s3, ron-s3; 4k anchor, ubatch 512, layer-major, kv q8, af=1, --pp-repeat 2, one mutex hold; all four strict checkers OK; analyzer `tools/analyze-gplr37.py`; traces `build/llama-cuda/gplr37b-*.jsonl`):** warm pass **pin-s2 3490.2 ms / 1173.6 tok/s -> ron-s2 2676.9 ms / 1530.1 tok/s (−813.3 ms, −23.3%, +30.4% tok/s)** and **pin-s3 2706.0 ms / 1513.7 tok/s -> ron-s3 2290.6 ms / 1788.2 tok/s (−415.5 ms, −15.4%, +18.1%)**. **The falsifier did NOT fire — the churn mechanism is confirmed:** ledger `setup` (sched reset + `alloc_graph`) returns to the clean band, 13.0 -> 8.2 ms (2 slots) and 12.1 -> 7.9 ms (3 slots), versus the V1 churn arms' 227.8 / 134.7 ms; `mmap_register` fires 40 times with `mode="init-once"`, **40/40 before the first `layer_compute`** (V1: 80 events, 34.46 GiB of churn, 2/80 before), and one `mmap_register_init` reports 18,500,517,888 B (17.23 GiB) pinned in 1.6 ms pre-timed. So the V1 regression was registration churn, not an inherent cost of file-backed registered pages. **Mechanism of the win (measured):** the PP per-bank hop collapses from median 45.24 -> 0.06 ms/bank (2 slots) and 23.43 -> 0.11 ms (3 slots), the `slot` bucket falls 1620.7 -> 991.2 ms (2 slots) and 582.3 -> 69.4 ms (3 slots), bank DMA windows stay pure (31.38-31.48 ms = 1.00x the 31.37 ceiling), compute is essentially unchanged (1363.6 -> 1297.2 at 2 slots), and TG is unaffected (76.6 vs 76.75 tok/s). **Output check (identity only, not the semantic-QA gate):** the 96-token greedy `--tg-text` is byte-identical between the pinned and mmap-registered arms at both depths. **Headline vs history:** ron-s3 2290.6 ms is the fastest warm 4k layer-major pass recorded (previous best 2706 / 2811.9 / 2843.2), and **ron-s2 2676.9 ms matches pin-s3 2706.0 ms while using one slot fewer (−498 MiB of VRAM)**, so depth-3 pinning is no longer needed for this shape if this mode is adopted. **My pre-registered prediction was WRONG in direction and size** (predicted within ±3% of the pinned control, gain ≤5% at 2 slots; measured −23.3% / −15.4%): the pre-registered reading that "the hop is already hidden at 3 slots" was false at this shape. **RAM price (measured by the runner, the mode's real cost):** 17.23 GiB of the model's file pages held pinned for the run, versus 0.9-1.4 GiB of pinned staging buffers for the pinned path; no staging buffers are allocated in this mode. The side RAM monitor (`tools/monitor-mem-gplr36.ps1`, CSV `build/llama-cuda/gplr37b-mem-monitor.csv`) is NOT usable evidence: it was started before the ~26-minute mutex wait and stopped after ~74 s, so its window misses the arms entirely (tooling defect to fix, not a measurement). **Gate defect found and fixed:** `tools/check-gplr35-pipeline.py` paired mode was depth-only (it FAILed an arm comparison at equal depth by construction) and its absolute `wait/wall` fraction false-FAILed the 30%-faster ron-s2 arm (37.0% of a *smaller* wall). Paired mode now (a) accepts a same-depth pair when the copy source differs, (b) replaces the shape-anchored absolute floors with control-anchored criteria (paired wait in ms ≤ control x 1.25 + 50 ms; paired exposure ≤ control + 350 ms), keeping the pure-DMA-window check always on and refusing identical-configuration pairs. Re-validated on six controls: depth pair 3-vs-2 PASS 1.188x, reversed FAIL (0.842x + wait + exposure), the FALSIFIED churn arms FAIL (0.846x), ron-s2 / ron-s3 pairs PASS (1.304x / 1.181x), identical config FAIL. **Consequence:** the as-implemented churn-free variant removes the hop and wins warm PP at both depths at a 17.23 GiB pinned-RAM cost; nothing here is a default change and nothing certifies fitness for use. **Also to note:** the cold-pass numbers in these traces are CONFOUNDED in the candidates' favour (candidate arms page in the whole model before their timed pass), so only the warm pairs are claimed. Nothing committed.

## 2026-09-17: T-GPLR-37 — DIRECT MMAP-SOURCED STAGING: FEASIBLE, HOP REMOVED FROM THE ACCOUNTING, BUT A REGRESSION AT 4k AS IMPLEMENTED. **Probe (measured, tools/gplr37-mmapreg-probe.cu, 452.5 MiB window of the real GGUF, all arms verified byte-for-byte):** pageable mmap DMA 49.47 ms (9.59 GB/s); `cudaHostRegister` on the FILE-BACKED mapping succeeds and its DMA runs 32.86 ms = **14.44 GB/s = the pinned ceiling**, registration 0.16 ms; the current path is CPU hop 20.22 ms + DMA 32.86 ms = 53.08 ms per bank. So the hop is 20.22 ms per 452 MiB bank (~809 ms per 40-bank pass) and registration is cheap *in isolation*. **Prototype (implemented, built clean, `--copy-source mmap-registered`):** the staging path now page-locks the page-aligned span covering a package's member sources, caches it on the slot, unregisters before that slot stages a different package, and DMAs straight from the model's mapping with no CPU hop; it allocates no staging buffers and refuses loudly on a registration failure (driver text in `error_msg`), and `--pp-pool` with it is refused. **A/B (4k anchor, ubatch 512, layer-major, kv q8, af=1, --pp-repeat 2, one mutex hold, all four strict checkers OK):** warm pass pin-s2 3123.9 ms / 1311.2 tok/s vs reg-s2 3692.9 ms / 1109.2 tok/s (**-15.4%**); pin-s3 2811.9 ms / 1456.6 tok/s vs reg-s3 3689.0 ms / 1110.3 tok/s (**-23.8%**). The pre-registered falsifier that fired is the REGRESSION branch (>=5% worse at either depth). **Where the regression comes from (host ledger, both arms):** the hop DID shrink (PP `copy_complete.host_submit_us` sum per pass 1587.8 -> 1222.3 ms at 2 slots; 1289.0 -> 1247.4 ms at 3) and the `slot` bucket fell (1315.6 -> 1153.3; 688.0 -> 590.3), but the per-launch host windows GREW far more: `setup` (sched reset + `alloc_graph`) 11.2 -> 227.8 ms (2 slots) and 11.9 -> 134.7 ms (3 slots), and `compute` 1329.5 -> 1845.1 / 1581.4 -> 2441.7 ms. Bank DMA windows stayed pure in every arm (median 31.48 ms = 1.00x the 31.37 ceiling) and 3-slot consumer wait stayed 36.5 ms, so nothing starved — the loss is host-side per-launch cost, not transfer rate. **Mechanism (indicated, not yet isolated):** every bank is a DIFFERENT 436 MiB file region, so the mode registers and unregisters ~111k pages per bank, 80 times per run (34.46 GiB cumulative registration churn, one `mmap_register` event per bank fill at 436.1 MiB, median 99.9 ms apart) — page-table/TLB churn that lands inside the per-launch `alloc_graph`/sync windows. The one-shot 0.16 ms probe number does not transfer to per-bank churn on freshly touched file regions. **Confound disclosed:** the arms ran in the order pin-s2, reg-s2, pin-s3, reg-s3, so the registered arms' COLD passes (reg 4659/4680 ms vs pin 6095/5658 ms, i.e. -23%/-17%) benefited from a file cache the earlier arms had warmed — the cold-pass win is recorded as CONFOUNDED, not as a result; the warm comparison (steady state, minutes apart) is the reliable one. **Consequence:** the as-implemented prototype is rejected; the data points at the churn-free variant (register the whole mapping, or all exec-layer spans, ONCE at init — T-GPLR-31 proved an 18 GiB grantable — paying ~17 GiB of pinned file pages and removing registration churn entirely) as the one remaining test of the family, with the falsifier that the `setup` inflation persists even without churn (which would mean the cost is inherent to file-backed registered pages and the family closes). Nothing committed.

## 2026-09-17: T-GPLR-36 — THE LONG-CONTEXT RESIDUAL IS PER-LAUNCH HOST WORK, NOT PER-GRAPH SETUP. Ledger run: one 200k-token arm (262k F16 KV, layer-major, ubatch 512, 3 slots, af=1, pinned, bf16 slabs, --pp-repeat 2, --pp-recon 1) under one mutex hold (tools/run-gplr36-ledger.ps1 / -locked.ps1; trace build/llama-cuda/gplr36-ledger-200k.jsonl, 7,854 events, strict checker OK, pipeline gate EXIT 0). Warm pass: wall 141,847.9 ms, compute 131,249.3 ms (summary), wait 36.5 ms, 1,409.96 tok/s, peak VRAM 3,265.56 MiB (the `peak VRAM` line again reads identically to the 2/3/4-slot arms, the known instrument defect). The new `host_ledger` event closes EXACTLY on the sweep (bucket sum - sweep = +0.0 ms, 0.00%): sweep 140,807.7 ms = compute 131,523.3 + patch 3,402.3 + slot 1,065.1 + other 4,516.3 + setup 278.6 + graph_call 22.1, and fn 141,813.0 = embed 318.4 + preload 126.1 + sweep 140,807.7 + output 77.2 + controller 483.6. **ANSWER TO THE OWNER'S QUESTION — IS IT PER-GRAPH SETUP ACROSS THE 15,640 LAUNCHES? NO.** `setup_us` (sched reset + `ggml_backend_sched_alloc_graph`, the exact quantity named) is 278.6 ms per pass = **0.017 ms/launch = 3.0% of the 9,343.8 ms residual**, with `graph_builds = 0` and `graph_prep_us` 15.4 ms (graphs prebuilt once, never rebuilt). The per-launch stage (16,422 launches = 391 embed + 15,640 layer + 391 output) is dominated by **host work outside the graph window**: `other` 4,516.3 ms (48.3% of the residual, 0.275 ms/launch — the chunk-loop body, sampled telemetry writes, `sets[c]->routes.count(il)` lookups; NOT yet split further), `patch` 3,402.3 ms (36.4%, 0.207 ms/launch — `prepare_chunk` input-snapshot memcpy + FA causal/KV-bound patching, repeated identically once per launch even though its inputs are per-chunk, so ~97% of it is redundant across the 40 layers of a chunk), `slot` 1,065.1 ms (11.4% — load_layer/wait_slot_ready/record_consumer_done/af enqueue/event wait), `setup` 3.0%, `graph_call` **22.1 ms (0.2%, 0.0013 ms/launch — the src-pointer snapshot save/restore is negligible)**. Pre-registered prediction outcome: **falsified on both halves** — `graph_call_us` was predicted to be the largest unattributed bucket (measured 0.022 s) and `other_us` under 1 s (measured 4.52 s); the falsifier (both under 1 s -> residual is not host bookkeeping) did NOT fire, so the residual IS host bookkeeping, with the largest single piece still unnamed (`other`). **Accounting correction found by the ledger:** the ledger's compute bucket exceeds the phase summary's `total_compute_us` by 274 ms because steps 1 (embed) and 4 (output) are executed but never accumulated as layer stats, so ~0.27 s of what the summary calls "residual" is really embed/output graph compute, not host time — the residual should be read as wall - (layer compute + embed/output compute) - pure DMA. **Mechanism for the 80-100% GPU-utilisation wave the owner observed:** the controller synchronises after EVERY launched graph (`ggml_backend_sched_graph_compute_async` then `ggml_backend_sched_synchronize`), so the host's 0.564 ms/launch of prep (patch 0.207 + other 0.275 + slot 0.065 + setup 0.017) is exposed GPU-idle time by construction -> duty = 8.009/8.573 = 93.4% on average, oscillating with the bimodal layer cost (the 10 full-attention layers take ~40 ms at 200k vs ~3.5 ms for the other 30, i.e. a 4-layer beat). Pre-staging the NEXT LAYER cannot remove this wave: at this shape staging is already fully hidden (wait 36.5 ms total, slot block 1.07 s = 11.4% of the residual), so 'waiting for the next layer' is not the limiter — per-launch host work is. Host RAM: peak VRAM 3.19 GiB is the run's own measure while byte_account reports KV 3.82 GiB (host-spilled), act_slab 1.53 GiB, slot_window 1.46 GiB, static 0.89 GiB, free 20.70 GiB; the owner's observed host-RSS growth (19 -> 30 GB) has NO instrument in this run and is recorded as an open question with a proposed monitor probe, not as a result. Nothing committed.

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
