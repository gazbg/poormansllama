# Goal: beat stock llama.cpp TG at 8 GB VRAM / 262k context (autonomous mode)

**Status: IN PROGRESS (state saved at a reboot, 2026-10-10 ~23:20).**
**Authority:** owner `/goal` message (2026-10-10): "work autonomously, make decision to achieve ... Beat the
`docs/llamacpp 8gb vram 262k base.md` in term of TG ... find, explore, implement, prove the best optimizations ...
ideas must be disproven only after full implementation and analysis, with the reason ... caching and scheduling
should yield better GPU utilization and therefore more TG." Autonomous mode per AGENTS.md "Autonomous Mode" is in
effect for this goal only. Binding rules still apply (charter, invariants, mutex, no push/PR, honest gates,
owner summary with decision log).
**Owner idea (mid-turn message):** reactive K-cache, no predictor. After a layer finishes token t, missing
experts are loaded in the background so token t+1 finds them resident; at token t+1 the router's misses are
computed from host weights without waiting. Example K=4 (32 rows/layer). This is the T-GPLR-82 S3 mechanism.

## Target (measured by the owner, not yet reproduced here)
Stock `llama-server` (build 10892, `C:\Development\AI\llamacppCUDA\llama-server.exe`), Qwen3.6-35B-A3B-MTP-UD-Q4_K_S,
`-ctk q8_0 -ctv q8_0 --flash-attn on -c 262000 --batch-size 8192 --ubatch-size 512 --fit-target 16000`, no speculation.
Fit result: 41 layers on GPU, 37 layers' MoE tensors on CPU, 6918 MiB VRAM (model 3317 + ctx 2780 + compute 820).
TG: 30.68 tok/s (918 tokens, 32.6 ms/token), 29.56 and 28.05 on shorter runs. **Beat = sustained > ~30.7 tok/s,
paired against a re-run of stock on the same machine/day** (cross-session drift is +-26%, see currentstate 2026-10-08).

## Environment layout (facts)
- Work tree for this goal: `C:\Development\AI\qwen38GZI\.claude\worktrees\attribution-handoff-doc-8ed1d1`
  (branch `claude/attribution-handoff-doc-8ed1d1`, HEAD `d167354` = owner's T-GPLR-82 checkpoint on
  `research/layer-residency`). The live tree `C:\Development\AI\qwen38GZI` is NOT edited by this work.
- Nested engine: `third_party/llama.cpp` here is a LOCAL CLONE of the live nested repo, detached at `8eb6ea378`
  (device-tail diagnostic). Its uncommitted state = the live probe files (`op-rows-probe.cpp`, `CMakeLists.txt`)
  + this goal's edits (below). `core.longpaths=true` was needed (MAX_PATH).
- Build dir `build/llama-cuda` (VS 18 2026 generator, `-DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON`, arch 120a-real).
  Full build ~13 min; incremental ~25 s. Build with `cmake --build build/llama-cuda --config Release --target
  llama-layer-streaming --parallel 16`. Binaries current as of 23:07 (include the edits below).
- NOT yet copied into this worktree: `build/llama-cuda/gplr07-qa-prompts/anchor-512.txt` (copy from the live tree's
  build dir; sha256_16 `d5bfd34a8a795e49`) and the recorded traces (read them from the live tree, read-only).
- The old freebuff scratch worktree `.freebuff/worktrees/b947db99-...` holds 3 commits not in the live line
  (T-GPLR-62/63 docs: `db54d30`, `13c0ff0`, `991dd53`). They stay on branch
  `freebuff/in-this-project-is-it-possible-to-run-in-hybrid-mo-b947db99-...`; nothing was deleted.
- Side-copy exe + ALL dlls before running arms that must survive a rebuild (a running exe locks files).

## Edits made in the nested tree (uncommitted, default-off paths only)
1. `src/llama-layer-streaming.h`: `row_set_fresh` (per layer, per expert: slot slab written for the next consume).
2. `src/llama-layer-streaming.cpp`:
   - bring-up sizes `row_set_fresh`;
   - retention path of `load_row_set`: resets it, sets 1 where the 4b slot refresh is enqueued;
   - **state fix**: a row admitted to the host leg (refresh skipped) is PENDING regardless of a later wait that
     clears `row_set_outstanding`; test hook `LLAMA_LS_TEST_LEGACY_S3_STATE=1` restores the old rule;
   - non-retain path: `fresh = (state == READY)`;
   - executor guard (after `sel_in` fill): every device-prefix row (first `n_used - experts_host_n` positions) must
     have `fresh == 1`, else loud failure naming layer/token/position/expert;
   - `ls_first_nonfinite()` + `scan_host_expert_outputs()` now also scans graph tensors `split_ffn_in` (the
     activation both legs read; the CPU q8_K path turns NaN blocks into zeros so output rows alone cannot see it) and
     `split_dev_out` (device head leg + shared expert).
3. `src/models/qwen35moe.cpp`: in the split branch names/marks outputs `split_ffn_in`, `split_dev_out`.

## Findings so far
| # | Finding | Label | Evidence |
|---|---|---|---|
| F1 | Control (K=2) is PCIe-bound: median 179 of 320 demanded rows miss per token, 320 MB staged/token, transfer 60% of wall, 16.6 tok/s | measured | `build/llama-cuda/gplr82-a2dev3-arm-A0.jsonl` phase_summary + rowset_round (live tree) |
| F2 | Offline LRU on that trace: hit 0.42/0.52/0.63/0.72/0.79 at 16/24/32/48/64 rows/layer (K=4 = 32 rows = 2265 MiB: 63%). Repetitive anchor text inflates locality | derived | scratch sim, this session |
| F3 | A2dev (device tail, 6/2 partition, zero CPU rows) PASSES semantic QA, 0 non-finite, but drifts: |dlogit| median 0.22, max 1.21 over 41 same-context steps, first token flip at step 41 where control margin was 1.90 | measured | `gplr82-a2dev3-verdict.json` + traces |
| F4 | Split changes summation association (head chain + shexp + tail chain vs 8-chain + shexp); the row-count probe did not cover that; handoff candidate list omits it | derived (code reading `qwen35moe.cpp`) | untested; shadow-reference experiment proposed |
| F5 | Candidate root cause of the NaN cascade (A2/B2): in the retention path a LEFTOVER outstanding row admitted by S3 skips its slot refresh, a wait in the same round then clears `outst`, the post-wait state reads READY, the plan puts it on the device prefix, the head leg reads a never-written slab. Step-C check cannot see it (state==READY) | derived, NOT yet confirmed | `llama-layer-streaming.cpp` ~10490-10531, 10910, 10970, 10992 |
| F6 | B1 (CPU tail, decoupling OFF, `s3_cap=0`) is garbage text from step 1 while step-0 logit is only 0.02 off: a SECOND defect independent of S3 | measured | `gplr82-s3off/` B1 text and logits |
| F7 | CPU-leg cost model: A2 compute ~30 ms/token vs ~15 ms control for 80 host rows/token => ~190 us per host row-layer; fixed per-layer overhead (~250 us) decides whether hybrid beats stock (stock itself ~ 37 x (475 + ~250) us + GPU dense) | derived | phase_summary A0/A2 |

## Decision log (autonomous mode)
- D1 Work location: this worktree (user said "move all the changes here"). Copied the owner's uncommitted work; owner
  then committed it live as `d167354`; branch ref fast-forwarded to it (`git reset --mixed`, old tip `991dd53`
  preserved on the freebuff branch). Reversible.
- D2 Add guard + fix as one A/B-able binary (test hook) so the root cause is proven before the fix is credited.
- D3 Use parallel agents for research/analysis only (read-only, scratch dir, mutex for any CUDA/timing run).
- D4 Success metric: paired TG vs re-measured stock, semantic QA PASS (answer stability, never token-id identity).

## Results log
- **E1 (2026-10-11, session `build/llama-cuda/e1-guard/`, binaries `bin/e1-guard`, tool `tools/run-tg-arms.py`, spec
  `build/llama-cuda/specs/e1-guard.json`)**: C0 rc=0; **L2 (legacy state rule) aborted loudly at TG step 1**:
  "expert 125 of layer 10 token 0 position 1 would be read by the device leg but its slot slab was not written this
  round" -> F5 CONFIRMED (measured). F2 (fixed rule) rc=0. B1 + QA results: read `e1-guard-verdict.json` (session was
  still finishing when this was written).
- D5: owner re-issued `/goal` after the reboot (autonomous mode, multi-agent, single GPU => all GPU work via mutex).
- E2 prepared, not run: `build/llama-cuda/specs/e2-shadow.json` (binaries `bin/e2-shadow`, env `LLAMA_LS_HOST_SHADOW=1`
  builds a device recompute of the CPU leg's suffix rows; log lines `host-shadow ubatch ... max_rel ... min_cos`).
  Run: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -VerdictTag e2-shadow
  -AllowUnprovenanced -Command "python <wt>/tools/run-tg-arms.py --spec <wt>/build/llama-cuda/specs/e2-shadow.json"`.
- Agents (6) were stopped at a usage limit; resume with SendMessage to their ids (research-caching, research-cpu,
  cache-sim, mtp-spec, stock-gap, branch-mining); scratch under the session scratchpad.

- **E1 FULL (session e1-guard)**: C0 reproduces step0 16.194481, QA PASS; L2 (legacy rule) aborts at TG step 1 with the
  stale-slab message (F5 CONFIRMED); F2 (fixed rule) finite over 512 steps, host share 22.8%, 5682 skipped waits, but QA FAIL and
  divergence from step 1 (same signature as B1) => a SECOND defect independent of S3.
- **E2/E3 (sessions e2-shadow, e3-ids, diagnostics behind env `LLAMA_LS_HOST_SHADOW`)** localized the second defect: at TG step 0 the
  CPU leg equals a device recompute of the same rows (cos 0.9996) and dev+host equals the unsplit reference FFN (cos 0.99999); from
  step 1 the unsplit reference itself collapses (|ref| 5e-5 vs 0.35) while residual, ids (0 mismatches) and the host bank bytes
  (slab_match 30/30) are identical to the control. Ruled out: CUDA graphs (`GGML_CUDA_DISABLE_GRAPHS`), op fusion
  (`GGML_CUDA_DISABLE_FUSION`), wrong ids, wrong bank bytes, stale shexp. Severity scales with host rows (1 row: divergence at step 3).
- **ROOT CAUSE of the CPU-leg garbage (derived from `ggml-alloc.c` + confirmed by A/B, E7)**: persistent graphs are re-run through
  `ggml_backend_sched_alloc_graph`; ggml-alloc treats tensors with `data != NULL` as externally allocated, so computed nodes keep the
  addresses of the first run, while the scheduler's CPU<->device copy tensors are planned fresh every run and can overlap them. Run 1
  plans everything together (correct), runs >= 2 corrupt `cur` etc. Only graphs with a CPU split are affected (device-only graphs
  have no per-run copy tensors). Fix: `ls_reset_graph_allocs()` clears data/buffer of computed nodes before each split FFN run
  (test hook `LLAMA_LS_TEST_NO_ALLOC_RESET=1` restores the old behaviour).
- **E7 (session e7-reset, 64 TG tokens, one binary)**: C0 17.24 tok/s; B1 (CPU tail, S3 off) 12.36 tok/s with NO divergence over 64
  tokens (dlogit median 0.316, same order as the device-tail drift); B1old (reset disabled) diverges at step 1 (reproduces the defect);
  F2 (S3 on) finite, host share 23.9%, first divergence step 41. Per-token ms (compute/transfer/wait): C0 15.3/34.2/27.8; B1
  29.9/34.7/28.2; F2 40.1/22.8/17.5. => the CPU leg through the ggml scheduler costs +326 us/layer (~163 us/row, of which ~59 us is
  expert math): the hybrid is CORRECT now but SLOWER than the PCIe-bound control. Host work must become cheap and overlapped.

## Decisions (continued)
- D6 Commit checkpoints locally in both repos (autonomous mode grants local commits); stage only task-owned files; nested
  commits use `Assisted-by` per nested AGENTS.md; no push anywhere.
- D7 Keep the diagnostics (`LLAMA_LS_HOST_SHADOW`: shadow device recompute, unsplit reference, id/bank/act/norm checks) in the tree,
  env-gated, because they are the instruments that found the root cause and will be needed again.
- D8 Next lever is host-leg cost and overlap, not correctness: fused executor-owned host expert job (no ggml sched), concurrent with
  the device head leg; then K=4 / admission policy from the cache simulation.

## Experiment queue (each needs prediction, check, result recorded here)
1. **E1 guard proof**: arm B2 (`--experts-host 2 --experts-host-budget 100`, K=2) with `LLAMA_LS_TEST_LEGACY_S3_STATE=1`.
   Prediction: loud abort "slot slab was not written this round" within the first ~10 steps. Then without the hook:
   prediction: finite over 512 steps. Runner: adapt `tools/run-gplr82-s3off.py` (supports `--session`; ARMS list
   hard-coded; needs env support) or write `tools/run-tg-arms.py`. Needs the anchor prompt copied (see above).
2. **E2 B1 oracle**: explain the step-1 garbage with the CPU leg and decoupling OFF (shadow device reference of the
   same rows, compare host_out vs device tail per layer/token).
3. **E3 reassociation control**: 8-wide matmuls with A2dev's association; expect bit-identical to A2dev.
4. **E4 hybrid at the 8 GB shape**: K=4, correct S3, measure TG vs stock paired; decompose per-layer fixed overhead.
5. **E5 async host executor** (host leg concurrent with GPU head leg, fused kernel) if F7 overhead dominates.
6. **E6 MTP self-speculation** payoff (stock baseline had no speculation).

## Parallel agents (launched 23:1x; all read-only; they die at reboot)
Scratch root: `C:\Users\<user>\AppData\Local\Temp\claude\C--Development-AI-qwen38GZI--claude-worktrees-attribution-handoff-doc-8ed1d1\0a80fa77-7086-4e1c-a6f9-e861ca8657c7\scratchpad\`
- `research-caching/` literature on expert caching/scheduling (KTransformers, Fiddler, MoE-Infinity, HOBBIT, ...).
- `research-cpu/` CPU expert kernels + overlap, ranks the ~250 us overhead sources, designs a fused host executor.
- `cache-sim/` offline policy simulator on all route traces (LRU/LFU/ARC, layer-adaptive capacity, admission, TG model).
- `mtp-spec/` MTP self-speculation payoff model + prior MTP work in worktree `5e3cb7b5-...`.
- NOT launched yet: **stock-gap** (reproduce stock paired with ours, decompose its 32.6 ms; use side-copied binaries,
  mutex, llama-bench/llama-cli, never bind 0.0.0.0) and **branch-mining** (idea ledger from branches `optimizaionV1`
  ("TG at 8/16 GB blocked" - find exactly what blocked), freebuff/*, docs/tg-expert-path-reconciliation.md, with a flag
  on any disproof that rests on a flawed implementation).

## Resume checklist
1. `git -C <worktree> status`; nested `git -C third_party/llama.cpp status` (expect the edits above).
2. Copy the anchor prompt; confirm `build/llama-cuda/bin/Release/*.exe|dll` exist (else rebuild, ~13 min full).
3. Check scratch dirs for agent output; relaunch the unfinished agents from the descriptions above.
4. Run E1 under `tools/with-gpu-lock.ps1 -AllowUnprovenanced` (dot-source arm-verdict.ps1 rules from AGENTS.md).
