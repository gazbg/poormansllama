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
| arm | harness | n | tok/s mean (range) | VRAM peak | fits 8192 MiB | QA (needles) |
|---|---|---|---|---|---|---|
| stock `-ncmoe 37` (E19) | llama-completion | 3 | 35.10 (34.68-35.77) | 7443 MiB | yes | not run |
| AER engine C16 (E19) | llama-completion | 3 | 72.06 (71.82-72.25) | 7177 MiB | yes | not run |
| LM hybrid C12 + kv-pad (E20) | LM runner | 2 | 57.70 (57.24-58.16) | 7929 MiB | yes | PASS 2/2 |
| **LM hybrid C14 + kv-pad (E20)** | LM runner | 2 | **63.30 (62.69-63.91)** | **8067 MiB** | **yes** | PASS 2/2 |
| LM hybrid C16 + kv-pad (E18) | LM runner | 3 | 65.34 (64.96-65.58) | 8203 MiB | NO (+11 MiB) | PASS 3/3 |
| LM hybrid C16, no kv-pad (E16) | LM runner | 1 | 34.95 | not polled | - | PASS |
| owner doc stock (`llama-server`, 918 tok) | llama-server | 1 | 30.68 | 6918 MiB (fit log) | yes | - |

## OWNER SUMMARY (AGENTS.md five-part form; autonomous decisions, NOT owner approval)
1. **Goal.** Beat stock token generation (TG) at 8 GB VRAM / 262k context. Bar = the owner's doc: 30.68 tok/s (`llama-server`).
   Idea under test: a per-layer GPU cache of hot experts; cache misses computed on the CPU from host RAM, no predictor.
2. **How it works, plainly.** Each token, the router picks 8 experts per layer. Experts already on the GPU run there. The rest run
   on 8 CPU threads that read the same weights from RAM; results are added back on the GPU. A small table tracks what is cached;
   between tokens a simple recency score moves hot experts into the GPU cache by DMA. Routing and math are unchanged.
3. **Glossary.** TG = token generation (one token per step, decode). PP = prompt processing (prefill). KV = attention key/value cache.
   MoE = mixture of experts. Expert = one of 256 feed-forward blocks per layer. Slot / C = GPU-cached experts per layer.
   n_pad = attention window pin (legacy: whole 262k cache scanned each step). kv-graph-pad = live-sized window. VRAM = GPU memory.
   QA = needle answer-stability check (programmatic). WDDM = Windows GPU driver model (its overhead counts in the VRAM peak).
4. **Pros / cons** (labelled):
   - MEASURED: C14 + kv-pad = 1.80x stock in the same session (63.3 vs 35.1), 2.06x the doc bar; fits the 8 GiB peak (8067 MiB).
   - MEASURED: the kv-pad fix is worth +88 % alone on the same engine (C16: 34.95 -> 65.7); the hybrid engine itself is worth ~2x stock.
   - MEASURED: stock here is 35.1, not the doc's 30.7, so the bar moved with the machine; the comparison is paired, not absolute.
   - MEASURED: C16 (65.3) is faster but peaks at 8203 MiB, 11 MiB over 8 GiB, so it does NOT meet the 8 GB shape.
   - MEASURED: the pinned host copy of all expert banks is ~17.5 GB of RAM; this needs a 64 GB host (the rig has 62 GB).
   - EXPECTED / UNPROVEN: the padded output enters a greedy loop after the answer (`Paris user What is the capital of France ...`).
     QA cannot see repetition. The first divergence (step 3) is a near-tie in the control too (margin 0.24); the logit drift at steps
     1-2 is 0.1-0.3 %, which argues against a wrong attention window, but a long-generation check is still owed.
   - CAVEATS: one prompt family, 128 tokens, 2-3 reps per arm; stock and hybrid run in different harnesses (llama-completion vs LM);
     QA is 128-token smoke, not the 50/512/4k ladder AGENTS.md requires; the tg-logit-margin readout is on in the LM arms.
5. **Recommendation (autonomous).** Keep C14 + kv-pad as the 8 GB operating point for TG. Before any acceptance: (a) run the QA ladder
   (50 and 512 tokens) on C14, (b) a 1024-token generation to test the loop, (c) re-measure stock in the LM harness or a same-harness
   control, (d) decide whether the 17.5 GB pinned host copy is acceptable (owner decision).

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
