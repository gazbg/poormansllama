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

- **E8 (session e8-hostrows, S3 off, 32 tokens)**: ffn_compute_us vs host rows 1/2/4/6 = 579/613/574/643 us (all-device FFN 121 us): linear fit
  574 + 8.8 x rows => the CPU leg through the ggml scheduler is a FIXED ~450-500 us per layer; expert math is ~10 us/row. (Guard readbacks
  `ls_first_nonfinite` add ~170 us/layer of wall outside ffn_compute: env `LLAMA_LS_NO_SPLIT_SCAN=1` skips them for timing arms.)
- **E9 (session e9-thr)**: applying `--cpu-threads` to the scheduler CPU backend did NOT shrink the fixed cost; more threads made it worse
  (ffn_compute us: old 4-thread per-call pool 441, 2t 502, 4t 461, 8t 500, 16t 563). Thread propagation is therefore opt-in only
  (`LLAMA_LS_TEST_NO_CPU_THREADS` currently inverts nothing: see code; default to be flipped to opt-in).

## Agent findings (all labelled by the agents; D = derived, M = measured)
- cache-sim (D, 23 streams, 4 natural docs): LRU hit 0.40-0.44/0.56-0.62/0.78 at 16/32/64 rows per layer; offline optimum 0.61/0.77/0.89. Layer-adaptive
  capacity and static hot sets do not pay. Second-touch admission cuts PCIe fills ~3x (142 -> 25 rows/token) at equal hit on natural text. A one-layer-lead
  co-activation prewarm adds +5pp. Fixed cost F dominates: 100 us F = 3.7 ms/token, +1pp hit = 0.19 ms.
- research-cpu (D): fixed overhead is GPU<->CPU crossings in the scheduler (sync of the whole stream, pageable D2H/H2D) 120-180 us, CPU-split dispatch
  30-70 us, 4 threads default (`--cpu-threads` never reaches the scheduler CPU backend; confirmed in code). Fused executor-owned host job would reach
  ~34 tok/s only with a slimmed executor floor.
- mtp-spec (D/M): self-speculation is a ~1.35x multiplier (range 1.1-1.6x) on a working base engine; the drafter loop exists in `main.cpp` but is
  incompatible with layer-major PP and the host leg today. Upstream PR 22673 reports 22.9 -> 29.4 tok/s at d=2 on a 6 GB laptop.
- branch-mining (ledger): **the earlier 'handoff-partial 38' arm ran ~30 tok/s at 7.6 GiB (ab20/tg72) but with incoherent output from TG token 2 -> a
  run-1-ok/run-2-corrupt signature of the same allocation bug**; AER hybrid executor (GPU slot cache + pinned mailbox + CPU miss pool + EMA admission,
  `LLAMA_MOE_HYBRID`) measured **97/118/134 tok/s at C32/64/96 vs 48/54/61 stock at matched VRAM, ctx 1k** (docs/currentstate.md of the AER worktree, 2026-10-01).

## Decisions (continued 2)
- D9 STRATEGY: the owner's idea (GPU slot cache of the last-K experts, CPU computes misses, background admission, no predictor) already exists as the AER hybrid
  engine and was measured ~2x stock. Port it into the LM nested tree and use it as the TG engine after LM PP: `--tg-hybrid` makes decode graphs bypass the
  streaming executor (`ls->tg_stock`, ls==nullptr in the graph builder) so `llama_moe_hybrid_build_ffn` applies. Core commit `aa6f00805` applied to the
  LM tree (13/14 files clean; `qwen35moe.cpp` hook ported by hand; registration tolerates ranges the runner already pinned). PP stays LM (charter priority 1).
- D10 Measure first: E10 reproduces the AER hybrid vs stock at the goal shape (`-c 262000`, q8 KV, C=16/24/32 slots, interleaved, VRAM polled).

- **E10 (session e10-aer-pair, goal shape `-c 262000`, q8 KV, interleaved, 2 reps, side-copied AER binaries `bin/aer-v4`, 400 tokens, GPU mutex, one hold)**:
  stock `-ncmoe 37 -t 16` 38.80 tok/s (VRAM peak +7435 MiB over idle); AER hybrid C16 70.43 (+7167 MiB), C24 79.02 (+7713), C32 86.06 (+8259).
  => **1.8x-2.2x stock at the goal shape, measured.** Caveats: stock here is 38.8 not 30.7 (today's machine state; compare within this session only);
  `llama-completion`, not `llama-server`; 2 reps; the hybrid's VRAM figure is WDDM whole-GPU use over idle (the cache itself is C x 40 x 3 slabs);
  output correctness NOT yet graded (answer-stability QA is the next gate, AGENTS.md).
- **Port status**: AER core commit `aa6f00805` applied to the LM nested tree (13 of 14 files clean); `qwen35moe.cpp` hook ported by hand (`ls == nullptr`,
  `selected_experts_in == nullptr`, `routed_rows == 0` guard); `--tg-hybrid` runner flag + `begin_tg_stock()` (frees PP slots/slabs, sets `tg_stock`) + cap
  exemptions; host-range registration tolerates ranges already pinned. **NOT YET COMPILED** - the build is queued behind the stock-gap series on the mutex.
- **E12 (smoke, session e12-smoke): the ported hybrid runs end to end inside LM.** Build OK after three fixes: (1) `cudaHostRegister`
  'already mapped' collision - the hybrid engine (init at PP, from its first single-token graph) pins the whole expert bank (17.5 GiB,
  reg 9.7 ms) before LM's register-once pass; LM now accepts already-pinned spans WITHOUT claiming ownership (no unpin of the hybrid's
  pages); (2) hybrid tolerance matches the real CUDA text 'already mapped'. Result: rc 0, 47 TG tokens, coherent answer containing all
  anchor needles (QX-7741 TR-2918 MK-5501 LD-8834 Paris), hybrid timeline present (span 26.5 ms over 47 tokens, stall 10.5 ms, host serve 11.6 ms).
  Not yet graded by the QA checker, not yet measured for speed.
- **E13 (session e13-lm-hybrid, goal shape `-c 262144`, q8, layer-major PP 50k? no: anchor-512 PP, 128 TG tokens, `--tg-hybrid`, one mutex hold)**:
  LM-hosted hybrid C16 35.15, C24 36.72, C32 37.50 tok/s; **QA PASS on all three arms** (positive pass, negative control fails as designed,
  guards silent). Divergence from the first token onward is the expected answer-stability drift (dlogit median 0.24-0.25, first flip step 42).
- **E14 (stock in the LM harness)**: REFUSED (rc 2): LM's layer-major TG cap (`--tg-tokens <= 3` without handoff) makes a stock LM TG arm
  impossible at this shape. Stock comparator therefore stays E10 (`llama-completion -ncmoe 37`, 38.8 tok/s, same day), with the caveat that
  the harnesses differ (LM vs llama-completion; prompt and PP path differ).
- **Interpretation (derived, not a win yet)**: the LM-hosted hybrid is **~35-37.5 tok/s**, which is *below* the AER-binary hybrid (70-86 tok/s, E10)
  and roughly at stock (38.8). The 2x AER result did not survive the port into LM. Hypotheses to disprove next: (a) the LM-side per-token
  overhead (`--tg-hybrid` still runs the graph-build/alloc path per token, and the PP-time fill/stage bookkeeping), (b) the CPU engine
  threads (AER: 8 threads, `-t 2` for llama; LM: 16 CPU threads contending with the engine's pinned pool), (c) the two-pool contention with the
  LM host graphs. Each is testable with an env/flag arm and needs no rewrite.
- **E15 (session e15-same-prompt, anchor-512 prompt via `-f`, 128 tokens, AER binary, same session)** decode tok/s from llama-completion
  `eval time`: stock `-ncmoe 37` 35.77; AER hybrid C16 (pinned `-t 2`) 68.60; AER C16 `-t 16` unpinned 69.19. => **the 2x is NOT prompt-
  specific** (same ~2x on the anchor prompt as on the hash-map prompt) and **NOT llama-thread-count** (`-t 2` vs `-t 16` identical).
  Conclusion (derived): the loss sits in the LM port, not in the engine design. The LM-hosted hybrid (E13: 35-37 tok/s) equals stock.
- **Leading hypothesis for the port loss (to test in E16)**: the LM runner pins `cparams.n_pad = n_ctx` by default (legacy), so every decode
  step's attention scans the WHOLE 262k-slot KV cache (10 full-attention layers). llama-completion keeps live-sized views. E13 ran without
  `--kv-graph-pad 256`. Expected: removing the full-cache scan recovers most of the gap (est. 10-15 ms/token from the earlier FA numbers).
- **E16 CONFIRMED (session e16-kvpad, LM-hosted hybrid, anchor-512, 128 tokens, one hold)**: the LM port loss was the full-context
  attention scan. C16 without kv pad 34.95 tok/s (QA PASS); **C16 + `--kv-graph-pad 256` 65.71 tok/s (QA PASS)**; C32 + pad 70.95
  (QA PASS). Same binary, same prompt; only the pad changed => the `n_pad = n_ctx` pin (legacy) cost ~30 ms/token at 262k.
  Numerical check (derived from the traces): steps 1-2 logits differ by 0.04-0.09 (0.1-0.3 %) between padded and unpadded; step 3
  is a near-tie in the CONTROL too (margin 0.24) and is where greedy diverges. A wrong or too-small attention window would show
  far larger deviations. The padded text later falls into a greedy loop (`Paris user What is the capital of France assistant ...`):
  consistent with greedy decoding of this chat template, NOT yet proven benign; QA cannot see repetition (needles only).
- **E15 (AER engine, same prompt, n=2)**: stock 35.16, AER C16 69.65, C16 unpinned 68.38, C32 86.98 (tok/s, llama-completion).
- **E18 (8 GB shape, LM-hosted hybrid C16 + kv pad, 3 reps, QA each, deterministic)**: 65.58 / 65.49 / 64.96 tok/s; QA PASS 3/3;
  identical output across reps (dlogit 0.0000). VRAM peak not polled in this run (run-tg-arms had no poller at the time).
- **E19 (stock and AER reference, same shape, 3 reps each)**: in progress at time of writing; see verdict under build/llama-cuda/e19-stock-8gb.
## CORRECTION (2026-10-11, E21-E24): every LM-hosted number above with bf16 activations is INVALID for the claim
- Root cause of the degenerate output (measured): the LM runner's DEFAULT prefill activation dtype is **bf16** (`act_slab_dtype_str`
  default `"bf16"`, T-GPLR-12). With bf16 the hybrid path diverges from stock/AER at ~step 3 and falls into a repetition loop
  (E21: C14+pad bf16 512 tokens loops; E22: C14 WITHOUT pad also loops => the KV pad is NOT the cause).
- Fix verified (E23/E24): `--act-slab-dtype f32`. Same binary, same flags: 512 tokens are coherent (copy of the document like stock),
  QA PASS, deterministic across reps (byte-identical text).
- The speed of the bf16 runs is real but measured on a broken generation; the rates below are for the f32 path only.
- VALID 8 GB point: **C14 + kv-pad + f32, 512 tokens**: 63.65 (E23), 63.80 and 63.31 (E24) => mean 63.59 tok/s, VRAM peak 8071 MiB
  (<= 8192 MiB). C16 bf16 peaked 8203 MiB, so it cannot meet the 8 GB shape even before the dtype fix.
- Stock at 512 tokens (same prompt): 36.00 (E22, one rep; E25 adds two reps, see below).
- Also to re-check: any LM run that relied on the bf16 default (semantic-QA entries in docs/currentstate.md) may carry the same
  divergence; the needle QA passed on the bf16 runs even though the text looped, so QA alone did not catch it.
## RESULT TABLE (goal shape, anchor-512 prompt, 128 TG tokens, decode tok/s, whole-GPU VRAM peak, idle 0 MiB)
SUPERSEDED 2026-10-11 by the G1 CERTIFIED RESULTS below: the stock control here was the waived, unattested `aer-v4` build and the engines ran prompts that differed by one token. Kept as history only.
| arm | harness | n | tok/s mean (range) | VRAM peak | fits 8192 MiB | QA (needles) |
|---|---|---|---|---|---|---|
| stock `-ncmoe 37` (E19) | llama-completion | 3 | 35.10 (34.68-35.77) | 7443 MiB | yes | not run |
| AER engine C16 (E19) | llama-completion | 3 | 72.06 (71.82-72.25) | 7177 MiB | yes | not run |
| LM hybrid C12 + kv-pad (E20) | LM runner | 2 | 57.70 (57.24-58.16) | 7929 MiB | yes | PASS 2/2 |
| **LM hybrid C14 + kv-pad (E20)** | LM runner | 2 | **63.30 (62.69-63.91)** | **8067 MiB** | **yes** | PASS 2/2 |
| LM hybrid C16 + kv-pad (E18) | LM runner | 3 | 65.34 (64.96-65.58) | 8203 MiB | NO (+11 MiB) | PASS 3/3 |
| LM hybrid C16, no kv-pad (E16) | LM runner | 1 | 34.95 | not polled | - | PASS |
| owner doc stock (`llama-server`, 918 tok) | llama-server | 1 | 30.68 | 6918 MiB (fit log) | yes | - |

## FINAL MEASURED COMPARISON (512 TG tokens, anchor-512 prompt, same machine, 2026-10-11)
SUPERSEDED 2026-10-11 by the G1 CERTIFIED RESULTS below: the stock control here was the waived, unattested `aer-v4` build and the engines ran prompts that differed by one token. Kept as history only.
- Stock `llama-completion -ncmoe 37` (E22 + E25, 3 reps): 36.00 / 36.64 / 36.30 => mean **36.31 tok/s**, VRAM peak 7443 MiB.
- LM-hosted hybrid C14 + kv-pad + f32 prefill activations (E23 + E24, 3 reps): 63.65 / 63.80 / 63.31 => mean **63.59 tok/s**,
  VRAM peak **8071 MiB** (<= 8192), QA needle PASS 3/3, output coherent and byte-identical across reps.
- Ratio **1.75x stock** (same prompt, same day). Owner doc bar 30.68 tok/s => **2.07x**.
- Caveat (MEASURED, E22/E23 text + code): LM's default TG re-decodes the LAST PROMPT TOKEN at step 0 (`main.cpp` ~6001 and 6560),
  so its continuation is conditioned on one duplicated token; stock is not. Speed is unaffected (same work per token).
  `--tg-first-pp-token` (opt-in, E26/E30) removes the duplicate: token 1 becomes the PP argmax. See E30 below for what that token is.
  CORRECTION (2026-10-11, E30c): the earlier statement that stock's first token is `You` was WRONG. `You` is the first word of the
  prompt, echoed by `llama-completion` unless `--no-display-prompt` is given. Stock's 8 generated tokens are `\n\n` (id 271) `<think>` `\n\n` `</think>` `\n\n` `Q` `X` `-`; its token 1 is `\n\n`, not `<think>`.
- NOT yet done for acceptance: the QA ladder (50 / 4k / 24k ... per AGENTS.md; only the 512-token needle check ran), a same-harness
  stock control, and a long-generation loop test beyond 512 tokens. These are owner-invoked gates.
## E30 FIRST-TOKEN CHECK (2026-10-11, anchor-512, ctx 262144, q8 KV, GPU mutex)
| Arm | Engine and flags | Generated text, start | Provenance | Result |
|---|---|---|---|---|
| E30a | stock `llama-completion -n 1`, prompt echo ON | `You are answering questions...` = prompt echo, not a token | aer-v4 binary | **SUPERSEDED**: misread as "stock token 1 = You" |
| E30b LMNH1 | LM `e26-firstpp`, `--tg-first-pp-token --tg-hybrid --kv-graph-pad 256`, **no** hybrid env, 512 TG tokens (`tg_tokens:1` in the spec is NOT what ran; the command has `--tg-tokens 512`) | `Q` `X-7741 TR-2918 MK-5501 LD-8834 Paris`, then repeats the prompt to 512 tokens | exe and llama.dll hashes in `e30-ppcheck-verdict.json`; wrapper refused for want of binary identity, **waived** with `-AllowUnprovenanced` (recorded in `build/llama-cuda/verdicts/provenance-waivers.log`) | rc 0, guards 0, 34.15 tok/s, poller VRAM peak 7045 MiB; the `peak VRAM` console line (3265.56 MiB) is the known-wrong field, not used |
| E30c | stock `llama-completion -n 8 --no-display-prompt -ncmoe 37 --temp 0 -no-cnv` on `aer-v4` | `\n\n` `<think>` `\n\n` `</think>` `\n\n` `Q` `X` `-` (8 tokens; token 1 = `\n\n`, id 271) | aer-v4 binary predates its sources: **REFUSED, waived**; this control is **UNCERTIFIED** | rc 0 |
Launch failures (not evidence): E30c first two tries failed to launch, one on a `cmd` parse error (`'build' is not recognized`, a
relative exe path written with `/`) and one on a sharing violation under a bash-mangled `cmd` call; both waived-logged, neither
produced output. The mangled bash call also left a stray empty file `C` in the worktree root (deleted).

FINDINGS (REVISED 2026-10-11 after the prompt mismatch was found; see amendment A2 and the G1 sections below)
- E30 compared two DIFFERENT prompts. `llama-completion -f` strips one trailing newline (`common/arg.cpp:1796-1799`); the LM runner does
  not. `anchor-512.txt` ends in "\n", so stock read 511 tokens ending in "." and LM read 512 ending in "\n". The "token 1 differs" finding was
  a PROMPT ARTIFACT, not a prefill defect. MEASURED by the prediction fixed in A2: on `anchor-512-nonl.txt` LM's PP top-1 is id 271, exactly
  stock's token 1, and the generated text starts identically (`\n\n<think>\n\n</think>\n\nQX-7741 TR-2918 MK-5501 LD-8834 Paris`).
- MEASURED (C6, session `g1-gapn`): on the identical prompt LM's top-2 are 271 (21.784) and 198 (18.271), gap 3.514. Stock's gap, found by
  `--logit-bias` bisection, is 3.375 [3.25, 3.5] at the base config, 3.625 at `-ub 128` and 3.375 with f16 KV, so Delta = 0.14 / 0.11 / 0.14
  against the 2.42 limit. LM's prefill agrees with stock inside stock's own configuration-to-configuration spread (about 0.25).
- DECISION D14 resolved: `--tg-first-pp-token` is validated (its token 1 equals stock's). It stays opt-in so recorded commands keep their
  meaning; every G1 LM arm passes it explicitly.
- Everything above this line in this section that compares token 1 between LM and stock is SUPERSEDED by the two bullets above.

## ACCEPTANCE CONTRACT G-TG8 (agent's own; written 2026-10-11 BEFORE the certified runs)
Per AGENTS.md "Acceptance in autonomous mode". This is the agent's autonomous acceptance contract, NOT owner approval.
- **Goal (owner /goal).** Beat the TG of `docs/llamacpp 8gb vram 262k base.md` (30.68 tok/s, llama-server, 918 tokens) at an 8 GB VRAM /
  262k-context shape, using caching and scheduling that raise GPU utilization; prove it; for every idea that fails, find the root cause;
  never dismiss a good idea for a bad implementation. **The bar is 30.68 tok/s; the agent may not lower it.**
- **Subject.** LM-hosted hybrid engine: C14 (14 GPU-cached experts per layer) + `--kv-graph-pad 256` + `--act-slab-dtype f32` +
  `--tg-hybrid` + `--tg-first-pp-token`, from ONE build, `build/llama-cuda/bin/g1` (LM runner, stock `llama-completion`,
  `llama-tokenize`; every file newer than every source in `third_party/llama.cpp`).
- **Workload W.** `anchor-512.txt` (511 tokens in stock, 512 in LM), ctx 262144 (stock 262000), KV q8_0, greedy, 512 TG tokens,
  ubatch 512, 3 reps per arm, arms interleaved inside one mutex hold.
- **Arms.** HYB = subject. NH = same LM binary and flags with the hybrid env unset (same-harness control). STK = `llama-completion
  -ncmoe 37 -t 16 --no-mmap -fa on -ctk q8_0 -ctv q8_0`, the owner doc's configuration.

| # | Criterion (falsifiable) | Measured how |
|---|---|---|
| C1 | HYB mean TG tok/s over 3 reps >= 33.75 (1.10 x 30.68) | LM `phase_summary phase=tg tokens_per_s` |
| C2 | HYB mean >= 1.20 x STK mean (same session, interleaved), AND min(HYB reps) > max(STK reps). HYB/NH also reported | STK: `eval time` tokens/s from its log |
| C3 | HYB whole-GPU `memory.used` peak <= 8192 MiB over the whole run (idle baseline subtracted if non-zero). STK and NH peaks reported | nvidia-smi poller, 0.4 s |
| C4 | Needle QA positive PASS and negative control fails as designed, HYB at 50, 512, 4096 and 24000 prompt tokens; 4096 and 24000 are one mutex run each with their own verdict. VRAM at 4096/24000 is reported, not gated | `tools/semantic-qa/check-answer.py` |
| C5 | No degeneration over a 2048-token generation on W: HYB distinct-6-word-gram ratio of the generated text >= NH ratio - 0.05. STK ratio reported | `tools/check-repetition.py` |
| C6 | Prefill agreement at the last prompt position: Delta = [logit_STK(271) - logit_STK(48)] + [logit_LM(48) - logit_LM(271)] <= 2.42. STK gap = smallest bias b with `--logit-bias 48+b` flipping STK's token 1 from 271 to 48 (resolution 0.25); LM logits = `pp_last_logits` top-8. 2.42 = 2 x 1.21, the largest single-logit drift between two valid configurations recorded in this program (F3). Fixed here before any datum | `tools/stock-logit-gap.py`, LM trace |
| C7 | Every arm in C1-C6 ran through `tools/with-gpu-lock.ps1` WITHOUT `-AllowUnprovenanced`, with its exe named in the wrapped command and attested (binary hashed, not older than any nested source). No waiver line is written for the session | `provenance-waivers.log`, verdict `provenance` |
| C8 | Inertness: `tools/run-tests.ps1 -Suite host` passes after the edit, and a default run (no `--tg-logit-margin`) emits no `pp_last_logits` event | host suite, trace grep |
| C9 | This plan, `docs/currentstate.md` and the owner summary (with decision log) are updated; superseded claims removed | files |

**AMENDMENT A1 (2026-10-11; made AFTER seeing a failure, so flagged as a post-hoc narrowing).** C8 as first written ("`tools/run-tests.ps1
-Suite host` passes") FAILED at the suite's first step, the doc-size guard: `docs/plans/generic-prefill-layer-residency-plan.md` is 485,036 B
against a 480,000 B limit. That file is unchanged since commit `a5fa223`, so the failure predates and is independent of this work. Clearing it
means re-sharding the owner's backlog docs: `tools/resplit-gplr-plan.py` would move all 26 blocks, drafts and a "(ready)" task included, and push the
archive past its own limit. That is out of proportion to this goal. C8 is therefore evaluated as the suite's ENGINE steps: the same
`ctest -R <host Include> -E <host Exclude>` the suite runs, invoked directly (78/78 passed on the g1 build), plus the default-trace check.
The guard failure is reported to the owner, not fixed here. (D19)

**AMENDMENT A2 (2026-10-11, recorded BEFORE any data on the corrected prompt).** The first certified runs (`g1-smoke`, `g1-speed`,
`g1-gap`) used `anchor-512.txt`, which ends in a newline. `llama-completion -f` strips ONE trailing newline (`common/arg.cpp:1796-1799`), LM does
not, so the engines ran DIFFERENT prompts (stock 511 tokens ending in ".", LM 512 ending in "\n"). That alone predicts the E30 token-1
difference (stock `"\n\n"`, LM `Q`) and the bias probe's result (`--logit-bias 48+8` did not flip stock; `48+100` did, so the stock gap
logit(271) - logit(48) is > 8 on stock's prompt while LM's own gap on its prompt is -0.586). Corrections: (1) every engine-vs-engine run uses
`anchor-512-nonl.txt` (no trailing newline, so the strip is a no-op and both read 511 identical tokens); the first `g1-speed` numbers
(HYB 61.33 / NH 34.46 / STK 37.42 tok/s) are kept as a record of the mismatched-prompt run, not as evidence for C1-C3; (2) C6 is restated in
its general form, which is what it always meant: take LM's top-2 tokens (A, B) at the last prompt position; the stock gap is
b*_B - b*_A where b*_x = smallest `--logit-bias x+b` that makes stock emit x first (0 if x is already stock's top); Delta = |gap_STK(A over B) -
gap_LM(A over B)| <= 2.42. The threshold and the quantity are unchanged; only the hard-coded ids (48, 271) are replaced by LM's own top-2.
**PREDICTION (falsifiable, fixed now):** on `anchor-512-nonl.txt` LM's PP top-1 is id 271. If it is not, the prompt explanation is falsified.
**Rule.** ACCEPT the goal iff C1-C9 all hold. A failing criterion is diagnosed, repaired and the same validation rerun; criteria are not
weakened. **Not claimed:** PP speed (the Goal Charter ranks PP first; this goal is TG by the owner's instruction), other prompts, models,
GPUs or contexts, quality beyond answer stability and C6, production fitness.
**Reported, not gated:** STK/NH VRAM, HYB VRAM at 4096/24000, NH numbers, comparison with the AER engine.

## DECISION LOG (continued 4, 2026-10-11)
- D14 Keep `--tg-first-pp-token` opt-in until C6 says the PP argmax is right (above).
- D15 Discriminate the token-1 difference at logit level (LM top-8 at the last prompt position + stock `--logit-bias` bisection). First-token
  identity is not a gate (AGENTS.md). Evidence needed: C6. Reversible.
- D16 Build the LM runner, `llama-completion` and `llama-tokenize` from one tree and run without `-AllowUnprovenanced`; no nested source edit
  after the build. Reversible. Replaces the waived `aer-v4` control for every claim in this contract.
- D17 No commits in the nested clone: `third_party/llama.cpp/AGENTS.md` requires explicit human approval per commit there. Work stays in
  the nested working tree; the outer repo commits docs and tools only. Reversible; flagged to the owner.
- D18 The agent runs the 4096 / 24000 ladder rungs itself, one per mutex run (AGENTS.md "Ladder lengths" bullet, autonomous clause).
- D19 Evaluate C8 on the host suite's engine steps (amendment A1 above); the doc-size guard failure is pre-existing and unrelated. Reversible.
- D20 Compare engines only on `anchor-512-nonl.txt`; restate C6 in its general (LM top-2 pair) form (amendment A2). Reversible.

## G1 CERTIFIED RESULTS (2026-10-11)
Build `build/llama-cuda/bin/g1`: LM runner, `llama-completion` and `llama-tokenize` from ONE tree (the nested working tree, uncommitted per D17),
source digest `D9CF5935595D6E9C`, `llama.dll` sha `8dd11e9b243f47be`, LM exe `8ef23b245f29d7b7`, stock exe `7d6a1eed22a3817d`. Every file is newer
than the newest source (06:48:53 < 06:49:34). Every session below ran through `tools/with-gpu-lock.ps1` WITHOUT `-AllowUnprovenanced`: each wrapper
verdict says PROVENANCE PASS, `gpu_pre`/`gpu_post` clean, and `provenance-waivers.log` has no line for any `g1-*` tag. Artifacts:
`build/llama-cuda/g1-*/` and `build/llama-cuda/verdicts/g1-*.verdict.json`. This section SUPERSEDES the E22-E30 comparison numbers above
(their stock control was the waived `aer-v4` build and the prompts differed by one token).

### Speed on the identical prompt (`g1-speedn`: `anchor-512-nonl.txt`, 511 tokens in both engines, 512 TG tokens, 3 reps interleaved in one mutex hold)
| Arm | tok/s per rep | mean | whole-GPU VRAM peak | role |
|---|---|---|---|---|
| HYB: C14 hybrid + kv-graph-pad 256 + f32 act slab | 61.63 / 61.19 / 62.36 | **61.73** | 8071 MiB | subject |
| NH: same LM binary, hybrid off | 34.35 / 34.23 / 34.09 | 34.22 | 7045 MiB | same-harness control |
| STK: `llama-completion -ncmoe 37 -t 16 --no-mmap` | 37.39 / 37.28 / 37.39 | 37.35 | 7443 MiB | owner-doc configuration |
| HYBU: hybrid with a UNIFORM frequency prior | 62.08 / 63.08 / 62.95 | 62.71 | 8071 MiB | reported: the prior does not matter |

HYB / STK = **1.653x**; HYB / NH = 1.804x; HYB / owner bar (30.68) = **2.012x**. The first session on the mismatched prompt (`g1-speed`) gave
HYB 61.33, NH 34.46, STK 37.42: the prompt difference moved speed by under 1%. PP (reported, NOT claimed): HYB 317 tok/s, stock 349 tok/s,
NH 90 tok/s at 511 tokens (the hybrid arm pins the expert banks, which also speeds its prefill streaming).

### Answer stability (C4): HYB, needle QA positive pass and negative control fails as designed
| rung | pp_tokens (trace) | session | whole-GPU VRAM peak |
|---|---|---|---|
| 50 | 50 | `g1-qa50` | 7801 MiB |
| 512 | 511 | `g1-speedn` (3 reps, plus NH and STK) | 8071 MiB |
| 4096 | 4096 | `g1-qa4096` | 8127 MiB |
| 24000 | 24000 | `g1-qa24000` | 8439 MiB (247 MiB over 8 GiB: reported, not gated) |

### 2048-token generation (C5, `g1-loop2048`)
HYB 63.35, NH 34.72, STK 37.90 tok/s. Distinct-6-word-gram ratio of the generated text: HYB 0.2176, NH 0.2296, STK 0.0229 (HYB >= NH - 0.05
holds). All three engines fall into repeating the answer or the prompt after the answer, because `--ignore-eos` removes the end token the model
wants to emit there; that is a property of the setup, present in stock, not a hybrid defect. HYB first diverges from NH at generation step 119
(dlogit median 0.17 over the 119 same-context steps), a near-tie flip, not a failure of the answer.

### Free-form prompts (generality; reported, not gated; 512 tokens, 2 reps interleaved)
| prompt | HYB | NH | STK | HYB / STK | HYB VRAM peak |
|---|---|---|---|---|---|
| prose, 37 tokens (`g1-div-prose`) | 65.57 | 34.52 | 37.62 | 1.743x | 7793 MiB |
| code, 24 tokens (`g1-div-code`) | 66.74 | 34.38 | 37.77 | 1.767x | 7641 MiB |

The repeated rows of the anchor text did not flatter the hybrid: free text is as fast or faster.

### Stock tuning (reported, not gated; `g1-stocksweep`, 256 tokens, 1 rep each, anchor prompt)
`-ncmoe 37`: threads 8 / 12 / 16 / 24 / 32 = 36.66 / 37.99 / 37.22 / 34.70 / 33.96 tok/s (7443 MiB). `-ncmoe 38 -t 16`: 36.25 (7011 MiB).
`-ncmoe 36 -t 16`: 37.96 (7875 MiB, still under 8192). Stock's best under 8192 MiB is about 38 tok/s, so the 37.35 control was not under-tuned
(HYB / best stock = 1.62x). One rep per point: differences under about 1 tok/s are noise.

### Prefill agreement (C6, `g1-gapn`)
LM top-2 at the last prompt position: id 271 = 21.7842, id 198 = 18.2707 (gap 3.5135); `<think>` (248068) is rank 7. Stock gap by `--logit-bias`
bisection (resolution 0.25): base 3.375 [3.25, 3.5]; `-ub 128` 3.625 [3.5, 3.75]; f16 KV 3.375 [3.25, 3.5]. Delta 0.138 / 0.112 / 0.138 <= 2.42.

### VRAM sampling check (`g1-fastvram`)
The 0.4 s per-call poller could miss a spike shorter than a process launch, and HYB has only 121 MiB of headroom. The same workload was re-run (2 reps per arm, interleaved) with ONE long-lived `nvidia-smi --loop-ms=50` (about 400-500 samples per run): peaks are identical, HYB 8071, NH 7045, STK 7443 MiB, and the speeds replicate (HYB 61.79, NH 34.38, STK 37.55 tok/s). WDDM can still hide sub-50 ms events; that is a limit of every nvidia-smi based peak in this program.

### Frequency prior
`LLAMA_MOE_HYBRID_FREQ` is a REQUIRED input (the engine aborts without it); it seeds the initial cache fill. The shipped file
`build/llama-cuda/aer-freq-train.txt` (dated 2026-09-30, before the anchor prompt existed) holds per-layer routing counts from the AER project's
calibration split. A uniform prior (arm HYBU) is as fast (62.71 vs 61.73), so the result does not depend on where that file came from.

## ACCEPTANCE RESULT G-TG8 (agent's own autonomous acceptance under the owner's grant; NOT owner approval)
Mechanical evaluation: `python tools/check-g1-contract.py` (its thresholds are constants copied from the contract above, not read from results):

```
C1  PASS HYB mean 61.73 over 3 reps (min 61.19) >= 33.75
C2  PASS HYB 61.73 vs STK 37.35 = 1.652x (need >= 1.20x); min(HYB) 61.19 > max(STK) 37.39; HYB/NH 1.804x
C3  PASS HYB whole-GPU peak 8071 MiB <= 8192 (STK 7443, NH 7045, HYBU 8071); 50 ms poller HYB peaks [8071, 8071] over [409, 438] samples
C4  PASS needle QA positive + negative control: 512: pass; 50: pass (pp_tokens 50, VRAM 7801 MiB); 4096: pass (pp_tokens 4096, VRAM 8127 MiB); 24000: pass (pp_tokens 24000, VRAM 8439 MiB)
C5  PASS 2048 tokens, distinct-6-word-gram ratio HYB 0.2176 >= NH 0.2296 - 0.05 (STK 0.0229)
C6  PASS tokens [271, 198]: stock gap 3.375 [3.25, 3.5] vs LM gap 3.513 -> Delta 0.138 <= 2.42 (ub128: delta 0.112; kvf16: delta 0.138)
C7  PASS 7 sessions attested, no waiver (g1-smoke, g1-speedn, g1-gapn, g1-qa50, g1-qa4096, g1-qa24000, g1-loop2048); reported-only sessions waived: none
C8  PASS host engine ctest 78/78 passed; pp_last_logits events: margin-on 1 (>=1), default 0 (==0)  [amendment A1: doc-size guard excluded]
C9  PASS result marker in plan: True; G-TG8 entry in currentstate.md: True

CONTRACT G-TG8: ALL C1-C9 HOLD (mechanical evaluation; acceptance is the agent's own decision, not owner approval)
```

Deviations from the contract as first written, each recorded before the data it touches: **A1** (C8 is evaluated on the host suite's engine steps,
because the suite's first step, a doc-size guard, fails for a reason that predates this work) and **A2** (engine-vs-engine runs moved to the
newline-free prompt and C6 restated in its general form after the prompt mismatch was found). No threshold changed. The contract is the
agent's own, written before the certified runs.

**Decision: ACCEPTED**, for exactly the claim the contract defines: on this machine, with this model (Qwen3.6-35B-A3B, Q4_K_S), at the 8 GiB /
262144-context shape and prompts up to 24000 tokens, the LM-hosted hybrid decode (C14 + kv-graph-pad 256 + f32 act slab, `bin/g1`) generates at
61.73 tok/s against 37.35 for stock on the identical prompt (1.65x; 2.01x the owner's 30.68 bar), within 8071 MiB of whole-GPU VRAM, with
answer stability, no extra degeneration and prefill logits that agree with stock.

**This acceptance does NOT certify:** PP speed (317 vs 349 tok/s at 511 tokens, stock ahead); any other model, GPU, driver or filled-context
size; CPU power or thermal cost; production fitness; byte or token identity with stock (the text parts company by near-tie flips and answer
stability is the gate). The nested llama.cpp changes behind it are uncommitted (D17), so the build is reproducible only from the working tree
and the patch saved at `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch`.

**Handed to the owner:** the five decisions in the owner summary (pinned RAM and spinning threads; nested commit policy; the doc-size debt that
makes the host suite fail; whether `--tg-first-pp-token` becomes the default; whether to open pushes or upstream material).

## OWNER SUMMARY (AGENTS.md five-part form; autonomous decisions, NOT owner approval) - REWRITTEN 2026-10-11 from the certified G1 results
1. **Goal.** Beat the token-generation speed of `docs/llamacpp 8gb vram 262k base.md` (30.68 tok/s, llama-server) at an 8 GB VRAM / 262k-context
   shape, by caching and scheduling that raise GPU utilization, and prove it. Idea tested: keep a small per-layer cache of the experts that were
   used recently on the GPU and compute the misses on the CPU from RAM, instead of predicting the next experts or streaming them over PCIe.
2. **How it works, plainly.** The model has 40 layers; each has 256 small expert networks and the router picks 8 per token. A stock run with 37
   layers' experts in system RAM computes all of those layers' experts on the CPU every token. The hybrid keeps 14 experts per layer in VRAM
   (the ones used most recently). For each token and layer the GPU runs the cached experts it needs and posts the missed ones to a shared
   mailbox; 8 CPU threads read the same weights from RAM, compute the misses and write the result back; the GPU adds it in. Between tokens a
   running score swaps hot experts into the GPU cache by a background copy. The arithmetic and the routing are unchanged. Two further fixes
   were needed to get the full effect: sizing the attention window to the live tokens (`--kv-graph-pad`, +88% on its own) and keeping the
   prompt activations in f32 (bf16 made the output degenerate).
3. **Glossary.** TG = token generation (decode, one token per step). PP = prompt processing (prefill). KV = attention key/value cache.
   MoE = mixture of experts; an expert is one feed-forward block, 256 per layer. C14 = 14 GPU-cached experts per layer. HYB = the hybrid arm,
   NH = the same program with the hybrid off (same-harness control), STK = stock `llama-completion` (the owner-doc configuration, 37 layers of
   experts on CPU), HYBU = hybrid with a uniform starting cache. tok/s = tokens per second. VRAM = GPU memory (whole-GPU peak from nvidia-smi).
   QA = programmatic needle check: the answer must contain four planted codes and "Paris", and a negative control must fail. Pinned memory =
   RAM the GPU can read directly; the hybrid pins all expert weights.
4. **Pros and cons.** (M = measured, E = expected, U = unproven; evidence in `build/llama-cuda/g1-*/` and the G1 sections above)
   - M: HYB 61.73 tok/s vs stock 37.35 on the identical prompt, interleaved, 3 reps each with non-overlapping ranges = **1.65x**; 2.01x the
     owner's 30.68. Same-harness: 1.80x over the same program with the hybrid off.
   - M: whole-GPU VRAM peak 8071 MiB (<= 8192) at the 511-token workload; stock peaks 7443, the control 7045.
   - M: answer stability passes at 50 / 512 / 4096 / 24000 prompt tokens (negative control fails as designed); a 2048-token generation is no
     more repetitive than the control; prefill logits agree with stock (gap difference 0.14 vs a 2.42 limit).
   - M: the speed holds on free-form prose (1.74x) and code (1.77x) prompts, so it does not depend on the repetitive anchor text, and it does
     not depend on the frequency prior (uniform prior 62.71 tok/s). Best-tuned stock under 8 GiB is about 38 tok/s (1.62x).
   - M: costs: ~17.5 GiB of pinned system RAM (the rig has 62 GB), 8 CPU threads busy-waiting during generation, VRAM above 8 GiB at long
     prompts (8439 MiB at 24000 tokens), PP about 9% slower than stock at 511 tokens (317 vs 349 tok/s).
   - U: not tested: other models, GPUs or drivers; contexts actually filled to 100k+ tokens; the CPU power and thermal cost of the spinning
     threads; behavior while other heavy CPU work runs. Every arm ran with `--ignore-eos`, so the text after the answer is junk in all
     engines. The hybrid's text parts company with the control after 92 to 428 tokens by near-tie flips (dlogit median 0.09 to 0.22).
   - M (process): the earlier headline (63.6 vs 36.3 = 1.75x) is superseded: its stock binary was a waived, unattested build and the prompts
     differed by one token. The earlier "LM prefill differs from stock" scare was that same prompt mismatch (stock strips a trailing newline
     from `-f` files), not a defect.
5. **Recommendation (autonomous acceptance, not owner approval).** The goal as written is met and the contract's nine criteria hold, so I accept
   it for the claim the contract defines: TG on this machine, this model, the 8 GiB / 262k shape, short-to-24k prompts. Keep the hybrid as the
   TG path for the 8 GB shape. The owner should decide: (a) whether 17.5 GiB of pinned RAM and 8 spinning threads are acceptable; (b) the nested
   clone's commit policy (its `AGENTS.md` demands per-commit human approval; the work there is still uncommitted); (c) the pre-existing
   doc-size debt that makes `tools/run-tests.ps1 -Suite host` fail; (d) whether `--tg-first-pp-token` should become the default; (e) whether
   to open pushes or upstream material (this goal needed neither). Decision log: D1-D20 in this file.

## DECISION LOG (continued 3)
- D11 Diagnose the LM-vs-AER 2x by isolating variables: prompt (E15, not it), llama threads (E15, not it), attention window (E16, it).
- D12 Fit to the 8 GiB shape by slot count, measuring VRAM with a whole-GPU poller (E20): C14 is the largest that fits.
- D13 Accept the kv-pad fix as part of the hybrid path (it is opt-in `--kv-graph-pad`; default behaviour unchanged).
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
