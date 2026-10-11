# Goal: beat stock llama.cpp TG at 8 GB VRAM / 262k context (autonomous mode)

**Status (2026-10-11): GOAL ACCEPTED by the agent (G1: 1.65x stock, certified) and improved by G2 (decode-phase cache resize, about 2x stock, agent-accepted); one reliability risk is open (IDEA LEDGER row 15). Sections below the Results log are history; read ACCEPTANCE RESULT G-TG8, G2 ACCEPTANCE RESULT and IDEA LEDGER first.**
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

**Commit:** docs and tools `6a8da3d` (outer repo, local only, never pushed). The nested llama.cpp working-tree changes are NOT committed (D17); `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch` (sha256 prefix a8e83ca67e44354f) is their record.

**This acceptance does NOT certify:** PP speed (317 vs 349 tok/s at 511 tokens, stock ahead); any other model, GPU, driver or filled-context
size; CPU power or thermal cost; production fitness; byte or token identity with stock (the text parts company by near-tie flips and answer
stability is the gate). The nested llama.cpp changes behind it are uncommitted (D17), so the build is reproducible only from the working tree
and the patch saved at `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch`.

**Handed to the owner:** the five decisions in the owner summary (pinned RAM and spinning threads; nested commit policy; the doc-size debt that
makes the host suite fail; whether `--tg-first-pp-token` becomes the default; whether to open pushes or upstream material).

## G2 CONTRACT (agent's own; written 2026-10-11 BEFORE any G2 run) - decode-phase cache resize
Why: the stop-hook asked for every disproof to be shown with its root cause, and doing that audit exposed an unexplained +1.0 GiB and ~19% between the
LM-hosted hybrid and the same engine in plain `llama-completion`. Diagnostics (`g1-aer`, `g1-tl`, `g1-aer-prose`, `g1-aer-code`, exploratory, not part of
this contract) found: (1) the hybrid's 1024 MiB cache is allocated at context creation, so it sits in VRAM together with the prefill weight slots (1044 MB)
and activation slab; the 8071 MiB whole-run peak is the PREFILL peak, decode runs near 7.07 GiB; (2) the ~19% was mostly a CONTENT confound (plain
`llama-completion` generates a repeating loop on the anchor prompt, which raises expert reuse); on non-repetitive text the harness effect is 2-4%;
(3) cache size is the real lever: plain C28 fits 7985-7995 MiB and is +13% (prose) to +18% (code) over C14.
Hypothesis H-R: allocating the decode cache AFTER prefill released its slots (`--tg-hybrid-slots 28`, opt-in) lets the LM-hosted path use C28 inside
8192 MiB and recovers most of that gain, with identical arithmetic.
Subject: LM runner + hybrid, `LLAMA_MOE_HYBRID=14` at context creation, `--tg-hybrid-slots 28` after prefill, build `bin/g2` (a NEW build; g1 stays the
certified record of G1). Workloads: `div-code.txt` and `div-prose.txt` (free text, no repetition; on code all engines generate the SAME text) and
`anchor-512-nonl.txt` (QA), 512 TG tokens, ctx 262144 (stock 262000), q8 KV, greedy, 2 reps interleaved in one mutex hold per prompt.
| # | Criterion (falsifiable) |
|---|---|
| G2-1 | On code AND prose: mean tok/s of HYB28 >= 1.10 x HYB14 (same LM binary, no resize) and >= 1.80 x STK (stock `-ncmoe 37`), min(HYB28 reps) > max(STK reps) |
| G2-2 | Whole-GPU `memory.used` peak of HYB28 <= 8192 MiB, read by the 50 ms poller, on every prompt |
| G2-3 | Needle QA at 511 tokens (anchor, nonl): HYB28 positive pass, negative control fails as designed |
| G2-4 | HYB28 vs HYB14 (same harness, same prompt): same-context dlogit median <= 0.30 up to the first divergence (the largest drift recorded between valid configurations was 0.22 median) |
| G2-5 | Default inertness: HYB14 on `bin/g2` within +-3% of the `bin/g1` HYB14 mean on the same prompt (anchor 61.73, prose 64.86, code 66.06), host-engine ctest still 100% |
| G2-6 | Every G2 session attested through `with-gpu-lock.ps1` WITHOUT `-AllowUnprovenanced`; `bin/g2` newer than every nested source |
| G2-7 | Plan, `currentstate.md` and the idea ledger updated |
Rule: ACCEPT G2 iff G2-1..G2-7 hold. If G2-1 fails the hypothesis is recorded as disproven with its root cause (measured), not hidden. G1 stays accepted
either way. Reported, not gated: C26/C29 neighbours, plain-path C28, parameter sweeps (B, lambda, threads), llama-server pairing.
Prediction (fixed now): HYB28 / HYB14 = +13% (prose) to +17% (code), i.e. 73-78 tok/s; resize cost under 0.5 s before the first decode.

## G2 RESULTS (2026-10-11)
Build `build/llama-cuda/bin/g2` (G1's tree plus the opt-in resize): source digest `0B5648EE9EC98D3C`, `llama.dll` `c53896ce2c042b01`, LM exe `e14b50e4f1764a8a`,
stock exe `2e553236ba0bbff3`; every file newer than the newest source (08:19 > 08:18:57). All sessions through `with-gpu-lock.ps1`, none waived, VRAM peaks from
one long-lived `nvidia-smi --loop-ms=50`. The reliability soak and the sweeps ran on `bin/g3` (g2 plus an opt-in watchdog, digest `6CCBC609EABFE355`).

### How the audit found it (exploratory sessions on the attested g1 build; not part of the contract)
- VRAM arithmetic from the HYB log: the hybrid's 1024.1 MiB cache is allocated at context creation (`sched_reserve` -> single-token graph reserve), when the prefill
  weight slots (1,044,389,888 B = 996.0 MiB) and slab (8.0 MiB) are also resident. The 8071 MiB whole-run peak is the PREFILL peak; decode runs near 7.07 GiB.
- Engine timelines (`tools/hybrid-timeline.py` on `LLAMA_MOE_HYBRID_LOG`), anchor prompt, microseconds per token:

| run | span | stall (GPU waits for host) | host gap | span+gap | layers with a miss |
|---|---|---|---|---|---|
| LM runner C14 | 14,020 | 9,107 | 2,019 | 16,039 | 39.98 |
| LM runner C14, `--tg-logit-margin 0` | 14,034 | 9,118 | 2,007 | 16,042 | 39.98 |
| plain `llama-completion` C14 | 11,289 | 6,289 | 2,129 | 13,418 | 38.69 |
| plain `llama-completion` C28 | 7,510 | 2,551 | 2,082 | 9,592 | 23.53 |

The host gap is the same everywhere, so it is not runner overhead. The difference is in the stall, and it follows the text: on the anchor prompt plain completion
loops (distinct-gram ratio 0.07 vs 0.62) and the loop raises the hit rate. On prompts where all engines generate the SAME text the harness effect is small:

| prompt | LM C14 | plain C14 | plain C28 | stock |
|---|---|---|---|---|
| code (identical text in every arm) | 66.06 | 67.61 (+2.3%) | 79.46 | 37.70 |
| prose | 64.86 | 67.66 (+4.3%) | 76.39 | 37.41 |

### Contract sessions (2 reps interleaved each; tok/s means; peaks from the 50 ms poller)
| prompt | STK | HYB14 (G1 config) | HYB28 (resize, G2) | plain C28 | HYB28/HYB14 | HYB28/STK | HYB28 peak |
|---|---|---|---|---|---|---|---|
| code, 24 tokens (`g2-code`) | 37.83 | 67.28 | **78.01** | 79.56 | 1.159x | 2.062x | 7641 MiB |
| prose, 37 tokens (`g2-prose`) | 37.91 | 66.35 | **76.86** | 78.68 | 1.158x | 2.027x | 7793 MiB |
| anchor, 511 tokens, needle QA (`g2-anchor`) | 37.80 | 62.21 | **72.92** | n/a | 1.172x | 1.929x | 8071 MiB |

On the anchor the whole-run peak is unchanged by the resize (8071 MiB = the prefill peak): the C28 decode footprint fits inside memory prefill already used.
The prediction fixed in advance was +13% (prose) to +17% (code), 73-78 tok/s, resize under 0.5 s: measured +15.8% / +15.9% / +17.2%, 214.5 ms at C30.
Same-context logit drift of HYB28 against HYB14 up to the first divergence: 0.104 (prose, step 92), 0.133 (code, step 428), 0.169 (anchor, step 119).

### Cache size curve (`g2-curve`, code prompt, identical text in every arm, 2 reps; C30 has one rep because the other hit the timeout trap, row 15)
| C | 14 | 28 | 30 | 36 | 42 | 48 |
|---|---|---|---|---|---|---|
| tok/s | 67.45 | 78.21 | 79.03 | 82.48 | 86.10 | 88.42 |
| peak, code prompt (MiB) | 7641 | 7641 | 7739 | 8149 | 8557 | 8967 |

VRAM on the 511-token shape (the one that sets the 8 GiB line; `g2-scan`, `g2-ubscan`): decode peak is about `8002 + 73.15 x (C - 28)` MiB: C30 8159, C32 8295,
C36 8569, C40 8841. The cap is reached at C30; the recommended operating point is C28 (8071 MiB, 121 MiB headroom).
Speeds in `g2-scan` are NOT comparable across C (the generated text, and so the hit rate, changes with C on that prompt); use the code-prompt curve.

### The doc's own tool: llama-server with the owner's exact flags (sessions `g3-server*`, `g3-serverw*`, `g3-serverw2*`; build bin/g3, attested)
Stock = `-ctk q8_0 -ctv q8_0 --flash-attn on -c 262000 --batch-size 8192 --ubatch-size 512 --fit-target 16000` (the doc's configuration). Hybrid arms add `--fit off -ngl 99
-ncmoe 40 -t 2 --cpu-strict 1 -Cr 30-31` and the engine env (C = 26 or 28, B 16, frequency prior). One greedy `/completion` per request (temperature 0, ignore_eos, 512
tokens, no prompt cache), three requests per server launch, two launches per arm; TG is the server's own `timings.predicted_per_second`; VRAM from one 50 ms nvidia-smi across
the server's life. "warm" = requests 2 and 3 (n = 4 per arm); "first" = request 1.

| prompt | arm | first request | warm | warm / stock | peak VRAM |
|---|---|---|---|---|---|
| code | stock | 27.60 | 33.77 (33.41-34.09) | 1.00x | 7201 MiB |
| code | hybrid C28 | 47.27 | 76.15 (75.07-76.93) | 2.26x | 8179 MiB |
| code | hybrid C26 | 47.52 | 74.96 (74.52-75.69) | 2.22x | 8043 MiB |
| prose | stock | 28.67 | 32.98 (32.79-33.06) | 1.00x | 7203 MiB |
| prose | hybrid C28 | 49.55 | 77.32 (75.45-78.79) | 2.34x | 8233 MiB (41 MiB over 8192) |
| prose | hybrid C26 | 48.67 | 75.05 (74.66-75.85) | 2.28x | 8097 MiB |

- The stock server reproduces the owner's measurement on this machine (doc 28.05-30.68; here 27.6-28.7 on the first request, 33.0-33.8 warm), so this is the doc's
  tool and the doc's regime. The hybrid at C26 is inside 8192 MiB for both prompts: **about 75 tok/s, 2.2-2.3x warm stock and about 2.4x the 30.68 bar.**
- The first request is slow for BOTH: the server loads the experts with mmap, and the first pass faults cold pages from disk (the 24-token prefill took 2.4 s, 99.7 ms per
  token). The hybrid's first request ran at about 20 tok/s for roughly 100 tokens and then at 81 tok/s (server `tg_3s` line). A first request is therefore a cold-start
  measure, not a decode-speed measure; the doc's own 918-token number mixes both.
- The server path was not the slower one: an earlier single-request pairing (`g3-server`) reported 49.2 / 48.9 tok/s for C28 only because the first request carried the cold start.

### Stress and soak
`g2-stress`: 1024 tokens (each run crosses three or four padded-window rebuilds), 8 unresized and 8 C30 runs interleaved: 0 failures; 67.36 (67.06-67.59) vs 78.73
(77.72-79.18) tok/s, +16.9%. `g3-soak` (the g3 build, watchdog on): 80 runs, 0 failures; HYB 68.01 (64.18-68.89) vs H30 79.81 (77.51-80.84) tok/s, +17.3%. Host-engine stalls above 20 ms hit 5 of 80 runs (about 103 ms, once 24 ms), 2 unresized and 3 resized, all within the first 22 runs; ledger row 15.

### Other measurements (reported, not gated)
- Parameter sweep on the plain path, C28, code prompt, 2 reps (`g3-sweep`): default 79.14; B8 79.62, B32 77.70, B64 76.41; lambda 0.90 78.50, 0.99 79.37;
  threads 6 / 10 / 12 = 77.56 / 79.27 / 79.53; poll 10 / 1000 = 79.50 / 79.33; backend sampling (`-bs`) 79.10. Nothing beats the defaults by more than noise.
- Prefill ubatch (`g2-ubscan`, 64-token runs, VRAM and prefill only): scheduler compute buffer 936.45 / 796.75 / 726.50 MiB at ubatch 511 / 256 / 128; prefill
  317 / 302 / 282 tok/s.

## G2 ACCEPTANCE RESULT (agent's own autonomous acceptance under the owner's grant; NOT owner approval)
Mechanical evaluation: `python tools/check-g2-contract.py` (thresholds are constants copied from the G2 contract above):

```
G2-1  PASS code: HYB28 78.01 / HYB14 67.28 = 1.159x, / STK 37.83 = 2.062x, min>maxSTK True; prose: HYB28 76.86 / HYB14 66.35 = 1.158x, / STK 37.91 = 2.027x, min>maxSTK True (need >= 1.10x and >= 1.80x)
G2-2  PASS HYB28 whole-GPU peaks (50 ms poller) {'code': 7641, 'prose': 7793, 'anchor': 8071} <= 8192 MiB
G2-3  PASS anchor (511 tokens) HYB28 needle QA positive pass and negative control fails as designed in 2/2 reps
G2-4  PASS HYB28 vs HYB14 same-context dlogit median up to the first divergence {'code': 0.13280150000000113, 'prose': 0.10409000000000113, 'anchor': 0.1688200000000002} <= 0.30
G2-5  PASS HYB14 g2 vs g1: {'anchor': '+0.8%', 'prose': '+2.3%', 'code': '+1.9%'} (need within +-3%); host engine ctest 78/78
G2-6  PASS 3 core sessions attested without waiver (g2-code, g2-prose, g2-anchor)
G2-7  PASS plan has G2 result + idea ledger: True / True; currentstate entry: True

RELIABILITY (reported, not gated): resized arms 34, non-zero exits 1 ['g2-curve/H30#2 rc=3221226505']
CONTRACT G2: ALL G2-1..G2-7 HOLD (mechanical evaluation; acceptance is the agent's own decision, not owner approval)
```

**Decision: ACCEPTED**, for exactly the claim G2 defines: with `--tg-hybrid-slots 28`, the LM-hosted hybrid decodes 15.8-17.2% faster than G1's C14 configuration and 1.93-2.06x
faster than stock on the same prompts, inside the same 8071 MiB whole-run peak, with unchanged answer stability and a logit drift under 0.17 against the unresized run.
G1 stays accepted on its own evidence. G2 was run on build `bin/g2`; the default path (flag off) reproduces G1 within +-3% (G2-5).

**Reliability caveat, stated plainly.** One of the resized runs died with a device-side timeout trap (a host reply later than 500 ms; ledger row 15). The
immediate cause is established; why the reply was late is not. An 80-run soak (40 unresized, 40 resized, interleaved) then ran clean: 0 failures. It showed the host engine can stall for about 100 ms (5 of 80 runs, in both arms), all of them early in the soak while other host activity was running, which is why the evidence points at contention, not resize. Treat the resized path as research-grade until that is closed.

**This acceptance does NOT certify:** a fix for the timeout; any cache size above C30 on the 511-token shape (it does not fit 8192 MiB; ledger row 12 sizes the next step);
other models, GPUs, drivers or filled contexts; PP speed; production fitness. The nested llama.cpp changes remain uncommitted (D17); the patch file is refreshed to the
current working tree.

## G3 CONTRACT (agent's own; written 2026-10-11 BEFORE any G3 run) - release the prefill-sized scheduler buffers; behavior under host contention
Why. G2 left two sized items. (a) Ledger row 12: the scheduler's CUDA compute buffer (936.45 MiB, reserved for the 511-token prefill ubatch) stays allocated during
decode, which only needs n_tokens = 1; releasing it should admit about C42 inside 8192 MiB (est. +9% over C28, from the clean curve 78.21 -> 86.10). (b) Ledger row 15
and the owner summary's "behavior while other heavy CPU work runs": the soak pointed at host contention but did not test it.
Mechanism for (a), opt-in, default off: `--tg-release-compute` calls `llama_context::reserve_for_decode()` after the prefill logits were read, which sets
`cparams.n_ubatch = 1` and re-runs `sched_reserve()` (new scheduler, old compute buffers freed), BEFORE `--tg-hybrid-slots` grows the cache, so the peak never holds both.
Build `bin/g4` = g3 + this + watchdog v2 (heartbeat, sub-phase, observed-pending summary; opt-in diagnostics). Workloads, arms and prompts as in G2; code and prose
are free text with identical output across engines on code; the anchor (511 tokens, nonl) sets the VRAM line and runs needle QA.
| # | Criterion (falsifiable) |
|---|---|
| G3-1 | The re-reserve releases >= 700 MiB (runner log line) and the whole-run peak (50 ms poller) of HYB42R = `--tg-hybrid-slots 42 --tg-release-compute` on the anchor is <= 8192 MiB |
| G3-2 | On code AND prose: mean tok/s of HYB42R >= 1.06 x HYB28 and >= 2.00 x STK, min(HYB42R reps) > max(HYB28 reps) |
| G3-3 | Anchor needle QA: HYB42R positive pass, negative control fails as designed; same-context dlogit median vs HYB28 <= 0.30 |
| G3-4 | No failure in >= 12 HYB42R runs (every non-zero exit is reported with its log) |
| G3-5 | Default inertness: HYB14 and HYB28 on `bin/g4` within +-3% of their `bin/g3` means; host engine ctest 100% |
| G3-6 | All G3 sessions attested without waiver; plan, currentstate and ledger updated |
Rule: ACCEPT iff G3-1..G3-6 hold; if G3-2 fails the hypothesis is recorded as disproven with its measured cause. G2 and G1 stand either way.
Prediction (fixed now): released about 880-900 MiB; C42 peak about 8130 MiB; HYB42R / HYB28 = +9..10% (code about 86 tok/s, prose about 84).
Contention test (reported, not gated; predictions fixed now). N busy-loop processes at normal priority (`load` arm key; N = 0, 8, 24 of 32 logical CPUs) around HYB28 runs
with the watchdog v2 at 20 ms: (i) tok/s falls less than 10% at N = 8 and less than 35% at N = 24; (ii) host-engine stalls above 20 ms appear in the loaded arms more often
than in the unloaded one; (iii) a stalled request shows a stagnant heartbeat (engine thread not running), not a polling thread that fails to see the request; (iv) no device
timeout trap at N = 8. A refuted prediction is recorded with its measured cause.

**Amendment A3 (2026-10-11, written BEFORE any G3 result was read) - baselines for G3-5.** The contract says "their `bin/g3` means", but no `bin/g3` session ran HYB28
in the LM runner, and the only `bin/g3` HYB14 run in it is `g3-soak` (code prompt, 40 reps). `bin/g3` is `bin/g2` plus the opt-in watchdog; neither touches the default path.
So G3-5 compares g4's HYB14 with the `g3-soak` HYB mean (code) AND the `g2-<prompt>` HYB means, and g4's HYB28 with the `g2-<prompt>` HYB28 means; the deviation scored
is the worst over the available baselines. The +-3% threshold is unchanged. Mechanical check: `python tools/check-g3-contract.py` (thresholds are constants copied from this contract).

## IDEA LEDGER (written 2026-10-11 to answer the stop-hook audit: "show that each idea was disproven only after full implementation, with its root cause")
Status key. CONFIRMED: the idea works as built. REPLACED: the idea is kept, the implementation was changed because of the measured cause. DISPROVEN: the
hypothesis was tested and is false, with the cause. OPEN: not implemented or not tested, so NOT disproven. Evidence names a session under
`build/llama-cuda/` or an entry in this file (E-numbers). Sessions named g1-*, g2-*, g3-* are attested (no waiver); earlier E-sessions are the
pre-contract exploration and were run with `-AllowUnprovenanced`, which is why G1/G2 re-measure what matters.

| # | Idea or hypothesis | How far it was taken | Result | Root cause or reason, and how it was shown | Status |
|---|---|---|---|---|---|
| 1 | Owner idea: reactive K-cache in the layer-streaming executor. After a layer finishes, missing experts load in the background; at token t+1 the misses are computed on the CPU from host weights; no predictor. | Built end to end (row-set staging, host leg, executor guard, scan instruments). Nine sessions E1-E9. | Correct after two bug fixes, but SLOWER than the PCIe-bound control: 12.36 vs 17.24 tok/s (E7). | A FIXED 450-500 us per layer for any CPU split through the ggml scheduler. Linear fit `ffn_compute = 574 + 8.8 x host rows` us (E8; expert math only about 10 us/row). More threads do not shrink it: 441 / 502 / 461 / 500 / 563 us at the 4-thread per-call pool / 2 / 4 / 8 / 16 threads (E9). Cause located by reading the scheduler: whole-stream sync, pageable D2H/H2D, split dispatch (research-cpu, 120-180 us + 30-70 us). | REPLACED by row 2 |
| 2 | The same idea with the cost removed: per-layer GPU slot cache, pinned device-mapped mailbox, 8 spinning CPU threads compute the misses, EMA admission by background DMA, no predictor (the "AER hybrid" engine). | Ported into the LM tree (D9), three build fixes (E12), then measured. | WORKS. G1: 61.73 vs 37.35 tok/s stock, 1.65x. G2: 78 tok/s, 2.0x. | n/a | CONFIRMED |
| 3 | The first CPU-leg runs produced garbage text. | Two hypotheses, each with an A/B hook in the binary. | Two independent defects, both fixed. (a) Stale slab: with the legacy state rule the guard aborts at TG step 1 with the exact message; with the fixed rule 512 steps are finite (E1). (b) Persistent-graph allocation overlap: ggml-alloc treats tensors with `data != NULL` as external, so computed nodes keep run-1 addresses while the scheduler plans its CPU<->device copy tensors fresh each run and can overlap them; disabling the reset reproduces divergence at step 1 (E7). | Proven by A/B | CONFIRMED |
| 4 | The LM-hosted hybrid equals stock instead of 2x (E13: 35-37 tok/s vs the AER binary's 70-86 in E10). | Five hypotheses, each tested. | (i) the prompt: DISPROVEN, the AER binary shows the same 2x on the anchor prompt (E15). (ii) llama thread count: DISPROVEN, `-t 2` and `-t 16` identical (E15). (iii) engine CPU affinity: DISPROVEN, 77.48 unpinned vs 76.66 pinned (g1-aer). (iv) per-token graph build: DISPROVEN, `graph_prep_us` is 17,383 in total over 512 tokens and the CUDA graph is reused every step; stock reuses 508 of 511. (v) full-context attention scan: CONFIRMED. | `n_pad = n_ctx` made every decode step scan all 262144 KV slots. Changing only `--kv-graph-pad 256`: 34.95 -> 65.71 tok/s (E16). | CONFIRMED / DISPROVEN as listed |
| 5 | bf16 prefill activations are harmless. | A/B f32 vs bf16 with QA and a repetition read. | DISPROVEN: bf16 output loops from step about 3; f32 is coherent and byte-identical across reps (E21-E24). The KV pad is not the cause (E22 loops without it too). | The LM default act-slab dtype is bf16 (T-GPLR-12); needle QA cannot see repetition. | CONFIRMED fix |
| 6 | LM prefill differs from stock (E30: first token `Q` vs `\n\n`). | Logit-level probe with a pre-registered prediction. | DISPROVEN: a prompt artifact. On the identical prompt LM's top-1 is 271 as predicted; stock gap(271 over 198) 3.375 vs LM 3.514, Delta 0.14 against a 2.42 limit. | `llama-completion -f` strips one trailing newline; the LM runner does not. | PROMPT ARTIFACT |
| 7 | The LM runner's per-token host work costs speed (telemetry, logits copy, stderr print). | A/B, the engine's own timeline, and a read-only code audit by a subagent. | DISPROVEN: `--tg-logit-margin 0` gives 61.06 vs 62.38 tok/s (noise). The engine's host gap is 2.0 ms in the runner and 2.1 ms in plain completion. The audit puts every runner-only item at 0.15-0.6 ms in total. | The 2 ms gap is common to both harnesses (lm_head, 1 MiB logits D2H and sync, decode path, input upload, launch). | NO EFFECT |
| 8 | "The same engine is 19% faster in plain llama-completion" (76.7 vs 62.2 tok/s on the anchor prompt). | Engine timelines on three prompts. | DISPROVEN as a harness effect: a CONTENT confound. Plain completion falls into a repeating loop on the anchor (distinct-gram ratio 0.07 vs 0.62), which raises the cache hit rate (0.5454 vs 0.3365). On identical text the harness effect is +2.3% (code, 67.61 vs 66.06) and +4.3% (prose). | Different generated text gives different routing; g1-tl, g1-aer-code, g1-aer-prose | CONFOUND |
| 9 | 8071 MiB is the hybrid's decode footprint, so C16 cannot fit 8 GiB. | Log arithmetic plus the 50 ms poller. | DISPROVEN: 8071 MiB is the PREFILL peak. The hybrid cache (1024.1 MiB, allocated at context creation), the prefill weight slots (996.0 MiB) and the activation slab (8.0 MiB) are all resident; decode runs near 7.07 GiB after `begin_tg_stock` releases 1,044,389,888 + 8,372,224 bytes. | Cache allocation order, not capacity | ROOT CAUSE (led to row 10) |
| 10 | Allocate the decode cache AFTER prefill released its slots (`--tg-hybrid-slots`, opt-in) and recover that VRAM. | Built (engine `resize`), prediction fixed in advance (+13..17%), 3 prompts x 2 reps, 8+8 stress pairs, host tests. | CONFIRMED: +15.9% code, +15.8% prose, +17.2% anchor; whole-run peak unchanged at 8071 MiB; QA passes; logit drift 0.10-0.17. | G2 sessions | CONFIRMED |
| 11 | A smaller prefill ubatch frees VRAM for more slots. | VRAM scan, 6 arms. | DISPROVEN (small): scheduler compute buffer 936.45 / 796.75 / 726.50 MiB at ubatch 511 / 256 / 128; at C40 the peaks are 8841 / 8689 / 8613 MiB; prefill falls 317 -> 282 tok/s. | About 660 MiB of the buffer does not depend on ubatch; the proportional part is the 262144-wide mask (about 190 MiB). | NOT WORTH IT |
| 12 | More cache slots keep paying. | Clean curve on the code prompt (identical text), 2 reps. | CONFIRMED, cap-bound: C14 67.45, C28 78.21, C30 79.03, C36 82.48, C42 86.10, C48 88.42 tok/s. Decode peak on the 511-token shape is about `8002 + 73.15 x (C-28)` MiB, so the cap is reached at C30 (8159 MiB). | The ~900 MiB scheduler compute buffer is sized for the 511-token prefill but decode needs n_tokens = 1; releasing it after prefill would admit about C42 (est. +9%). Not implemented. | OPEN (sized next step) |
| 13 | Tune the engine (promotion budget B, EMA lambda, thread count, poll level). | 12 arms x 2 reps on the plain path, C28, identical text. | NO GAIN: every setting within +-2% of the default; worse are B32 (-1.8%), B64 (-3.4%) and 6 threads (-2.0%). | A flat optimum near the shipped defaults; large B costs promotion DMA (inferred). | CLOSED |
| 14 | On-device greedy sampling (`-bs`) to shrink the 2 ms inter-token gap. | 2 reps, plain path. | NO GAIN: 79.10 vs 79.14 tok/s, +48 MiB VRAM. | The gap is not the logits copy or sampling. | DISPROVEN |
| 15 | Reliability of the resized hybrid, and what the one timeout trap was. | Watchdog added (opt-in `LLAMA_MOE_HYBRID_WATCHDOG_MS`), `tools/hybrid-stalls.py`, 16 stress runs, an 80-run soak (`g3-soak`: 40 unresized + 40 resized, interleaved, 512 tokens). | Soak: **80 runs, 0 failures**, and the resize gain replicates over 40 reps each: 68.01 -> 79.81 tok/s (+17.3%). One trap in about 29 earlier resized runs (second C30 rep, `seq 8743 timed out (res_seq 8742)`), none in about 150 other hybrid runs. The soak did show host-engine STALLS (worst single `serve` time 24-204 ms against a normal 3 ms): 5 of 80 runs, 2 unresized and 3 resized, so NOT resize-specific. | The immediate cause is a host reply later than the 500 ms device-side timeout. The stalls are about 103 ms (and once 204 ms), identical in unresized and resized runs, and ALL FIVE fell in the first 22 soak runs, while this session was running other tool calls and three read-only subagents; the last 58 runs, with the session idle (blocked on a usage limit), had none. So the cause is contention from other host activity on the engine's pinned, spinning threads, not the engine logic. The watchdog's first report: request 11171 unanswered 110.6 ms while the engine thread had been serving it only 18.4 ms. | CONTENTION-CAUSED, not resize-caused; the 500 ms device timeout is fragile on a loaded host (tested in G3) |
| 16 | MTP self-speculation (est. 1.35x). | NOT implemented. | n/a | This tree does not load the model's nextn tensors for qwen35moe (load log: "unused tensor blk.40.nextn.*"), and the hybrid hook serves only n_tokens == 1, so verification batches would take the slow path. | OPEN (not disproven) |
| 17 | Predictors and prewarm (second-touch admission cuts PCIe fills about 3x; co-activation prewarm +5 pp hit rate). | Analysis only (cache-sim agent), not built. | n/a | The mailbox engine's EMA admission already runs without a predictor. | OPEN (not disproven) |
| 18 | An async fused host executor inside the LM scheduler (E5). | Not built separately. | n/a | Superseded by row 2, which removes the same cost. | REPLACED |

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
## Experiment queue - OUTCOMES (the original queue was written at the start of the goal; every item is now closed or explicitly open)
| # | Queued item | Outcome |
|---|---|---|
| 1 | E1 guard proof (legacy state rule must abort loudly, fixed rule must run finite) | DONE, E1: the legacy rule aborts at TG step 1 with the exact message; the fixed rule is finite over 512 steps (ledger row 3). |
| 2 | E2 explain the step-1 garbage with the CPU leg and decoupling OFF | DONE, E2/E3/E7: persistent-graph allocation overlap, proven by A/B (ledger row 3). |
| 3 | E3 reassociation control (8-wide matmuls with the device-tail association) | NOT RUN: moot once the split-FFN path was dropped in favour of the mailbox engine (D9). Not disproven; the question it asked (does summation order explain the drift) was answered differently by E30/C6 (Delta 0.14). |
| 4 | E4 hybrid at the 8 GB shape, K=4, decompose the per-layer overhead | DONE through D9: overhead decomposed in E8/E9 (450-500 us fixed scheduler cost), engine timelines in G2; shape measured in G1/G2. |
| 5 | E5 async fused host executor inside the LM scheduler | REPLACED by the mailbox engine, which removes the same cost (ledger rows 1-2, 18). |
| 6 | E6 MTP self-speculation payoff | OPEN, not implemented (ledger row 16: nextn tensors not loaded for qwen35moe in this tree; the hybrid hook serves n_tokens == 1 only). |

## Parallel agents - what each returned
- `research-caching`, `research-cpu`, `cache-sim`, `mtp-spec`, `branch-mining`: findings are in "Agent findings" above (labelled D = derived, M = measured).
- `stock-gap`: replaced by the paired stock measurements in E10/E19 and, certified, G1/G2 (including the stock thread/offload sweep).
- 2026-10-11 (stop-hook audit): a read-only subagent audited the LM runner's per-token host work. Its report matched the engine timelines: the ~2 ms inter-token gap
  is common to both harnesses, every runner-only item costs 0.15-0.6 ms in total, and the one lever common to both (on-device greedy sampling) measured no gain (ledger row 14).

## Resume checklist (current)
1. `git -C <worktree> status`. Outer repo: AGENTS.md may be uncommitted (owner review). Nested `third_party/llama.cpp` has UNCOMMITTED edits (D17); their record is
   `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch` (refreshed at each acceptance).
2. Builds are side-copied and never reused after a source edit: `build/llama-cuda/bin/g1` (G1), `bin/g2` (G2), `bin/g3` (g2 + watchdog), later ones if listed in the G3 notes.
   Rebuild with `cmake --build build/llama-cuda --config Release --target llama-layer-streaming llama-completion llama-tokenize llama-server --parallel 12`
   (the full build also fails to link the unrelated upstream `test-unicode`).
3. Engine-vs-engine runs use `anchor-512-nonl.txt` (llama-completion `-f` strips a trailing newline), free-text prompts `div-code.txt` / `div-prose.txt`, never the
   repeating anchor text for hit-rate-sensitive comparisons.
4. Re-run the checks: `python tools/check-g1-contract.py`, `python tools/check-g2-contract.py`, `python tools/check-repetition.py --selftest`, the host engine ctest
   pattern in `build/llama-cuda/host-include.txt` / `host-exclude.txt`.
5. All GPU work through `tools/with-gpu-lock.ps1`, with the exe named on the wrapped command line (`--attest`), never `-AllowUnprovenanced`.
