# Current State

## How this file is organised (resplit 2026-10-10)

This is the canonical current-state record. The initial split on 2026-09-18
moved old entries into searchable shards after `docs/currentstate.md` passed
lean-ctx's per-file search cap. The 2026-10-10 resplit applies the same
rule again: recent dated entries stay live, older dated entries move verbatim
into per-day shards, and this file keeps a derived index plus the non-entry
risk tail.

- **Recent entries** (2026-10-03 .. 2026-10-10) stay in this file, below.
- **Everything older** (2026-08-18 .. 2026-10-02) lives in the indexed shards:
  `docs/currentstate/2026-08-18-and-earlier.md`, `docs/currentstate/2026-08-20..2026-09-11.md`, `docs/currentstate/2026-09-12..2026-09-13.md`, `docs/currentstate/2026-09-14.md`, `docs/currentstate/2026-09-15..2026-09-16.md`, `docs/currentstate/2026-09-17.md`, `docs/currentstate/2026-09-18.md`, `docs/currentstate/2026-09-19.md`, `docs/currentstate/2026-09-20.md`, `docs/currentstate/2026-09-21.md`, `docs/currentstate/2026-09-22.md`, `docs/currentstate/2026-09-23.md`, `docs/currentstate/2026-09-24.md`, `docs/currentstate/2026-09-25.md`, `docs/currentstate/2026-10-01.md`, `docs/currentstate/2026-10-02.md`.
- **Current state that is not an entry** -- "Known gaps / risks" and
  "Blocked / escalated" -- stays at the bottom of this file.
- New entries are added at the top of the "Recent entries" block. Existing
  entry text is never edited in a shard; an intentional correction must be
  explicit and must update the split manifest.
- The split is mechanically checked with
  `python tools/resplit-currentstate.py --verify`. The apply-time source hash,
  entry hashes, shard bodies and dispositions are recorded in
  `docs/currentstate/resplit-manifest-2026-10-10.json`. Prefer `rg` across
  `docs/currentstate/` when the date is unknown.

## Recent entries (2026-10-03 .. 2026-10-10)
## 2026-10-11 (GPU held for 67 sessions since the G2 entry through `tools/with-gpu-lock.ps1`, none with `-AllowUnprovenanced`; every wrapper verdict PASS with the binary attested): G3-G7 -- the G3 release the prefill-sized scheduler buffers, the first-pad-boundary deadlock root-caused and fixed, a lazily committed KV cache, auto-sized expert cache; the LM-hosted hybrid decodes 2.7-2.9x stock (final builds g11-g14)

**Asked.** Same standing goal: more TG at the owner's 8 GB / 262,144-token shape, every failed idea disproven only after a full implementation and with its root cause, the agent decides and accepts.

**Answer.** (Agent's own acceptance, NOT owner approval; contracts G3-G7, the amendments A3-A6 and the idea ledger rows 19-34 are in `docs/plans/tg-beat-stock-8gb-262k.md`; mechanical checks `tools/check-g3-contract.py`, `tools/check-g4-contract.py`.)

- **Headline (final build g11, same machine, interleaved reps in one mutex hold per prompt, code/prose 4 reps, anchor 2).** Stock `llama-completion` with the owner's flags 39.05 / 38.73 / 38.84 tok/s (code / prose / anchor). Hybrid with the cache resized after prefill and the prefill buffers released (HYB42R) 86.94 / 89.35 / 77.85. **With the lazily committed KV cache and 82 slots (L82) 103.80 / 106.26 / 102.34 = 2.66x / 2.74x / 2.64x stock; with input batching on top (L82B) 107.73 / 110.92 / 106.47 = 2.76x / 2.86x / 2.74x stock, 3.5x the owner's 30.68.** Whole-GPU nvidia-smi poller peak 8,087-8,097 MiB for L82/L82B, 7,435 for stock. 104 hybrid runs in the g11 sessions (36 of them the reliability session), 0 non-zero exits.
- **G3 (accepted).** `--tg-release-compute` frees the scheduler's prefill-sized compute buffers (1,194.3 MiB, 931.4 of them VRAM, in 40 ms), which lets C42 fit: HYB42R / HYB28 = 1.102x (code) and 1.148x (prose), 2.23x / 2.31x stock on the final build; needle QA passes; the "3% regression" of an intermediate build was a timer that included the resize (amendment A4).
- **G4 (NOT accepted as written).** Input batching `LLAMA_CUDA_INPUT_BATCH=1` (opt-in): four designs, each profiled before the next. What works records the ten per-token input uploads in pinned staging and issues them back to back at the start of the graph compute: +3.1% (code and prose, ranges separated, text identical), but the gap-idle drop is -299 us against the contract's -400 us (G4-3 fails; the prediction counted API time, not critical-path time).
- **G5 (accepted): the three rare crashes were ONE deadlock.** At the second host request of the first KV-pad-boundary token the GPU spins in `wait_add`; the launching thread is inside a device-synchronizing driver call (the lazy load of a kernel launched for the first time, `cudaFuncGetAttributes`); the engine thread is in its promotion-copy `cudaMemcpyAsync` and needs the same driver lock, so it cannot write the reply; the 5 s device timeout traps. Reproduced 16 of 16 with a forced `cudaMalloc`/`cudaFree` in the direct launch path, fixed 16 of 16 (and on the final build's defaults 8 of 8 against 0 of 8 with the old behaviour) by a dedicated copy thread: the engine thread now makes no CUDA call (default on, `LLAMA_MOE_HYBRID_DMA_THREAD=0` disables). The pool growth I first suspected does not happen at any boundary. The G2 entry's explanation of its one trap (host contention) is superseded by this. A 2,048-token soak on the final build (6 runs, 48 pad boundaries) completed with 0 failures; three ~111 ms host stalls were survived by the 5 s limit.
- **G6 (accepted): lazily committed KV cache `--kv-lazy`.** The 262,144-token window's 2,720 MiB q8 KV cache is only reserved as address space and the runner commits the rows the run uses (40 MiB for the code prompt), so 2.6 GiB buy expert cache: L78 102.0 tok/s (1.173x the same engine at C42, prediction 1.17x), L82 103.5 (1.19x). Byte-identical text at equal slots; needle QA passes at 4k and 24k tokens, where the non-lazy run overshoots 8,192 MiB (8,439) and the lazy one fits (8,063). The gain shrinks as the window fills (0.0146 slots per MiB of committed KV); measured at 24k rows, derived beyond.
- **G7 (NOT accepted as written): `--tg-hybrid-slots auto`.** Sizes the cache from the device's own reading; chose 76 slots (106.3 tok/s, -2.0% against the explicit C82) where the contract's band was 79-82. The first hypothesis for the over-reading was wrong; the cause is a unit difference: `cudaMemGetInfo` reads exactly 1,080 MiB above the nvidia-smi poller, NVML 319 MiB above (VRAM ACCOUNTING CAVEAT in the plan).
- **Disproven this round, each with its measured cause (ledger rows 20-25, 32, 23).** Software prefetch of expert rows, more engine threads, a non-uniform slot split, history-based miss predictors, promotion-policy tuning (all at or near the DRAM ceiling: 46 of 51 GB/s), LOCKed polling loads in the ggml barrier (implemented, <= 2 us of ~40, reverted), GPU-assisted miss service (the PCIe copy loop costs the CPU path +16.3%; bounded +4..7%). Open and bounded: MTP self-speculation on the hybrid (+5..15% net), the host work between tokens (+2.5..3.5% cheap, +10..13% as a GPU-resident loop).
- **New code, all default off except the copy thread.** `ggml_cuda_lazy_buffer` and `ggml_backend_cuda_lazy_commit_rows` (ggml-cuda.cu), the `LLAMA_KV_LAZY` hook in `llama_kv_cache`, runner flags `--kv-lazy`, `--tg-hybrid-slots auto`, `--tg-vram-cap-mib`, `--tg-vram-safety-mib`, `--tg-vram-reserved-mib`; `LLAMA_CUDA_INPUT_BATCH`; test knobs `LLAMA_CUDA_GRAPH_RESET_EVERY`, `LLAMA_CUDA_DEBUG_FREE_AT_NODE`, `LLAMA_CUDA_POOL_LOG`, `LLAMA_TG_VRAM_TRACE`; watchdog sub-states; `tools/check-g3-contract.py`, `tools/check-g4-contract.py`, `tools/gpu-clock-probe.py`, `tools/dram-share-probe.py`, `tools/nsys-gap-api.py` and the other analysis tools listed in the plan.
- **Known gaps and residual risk.** (1) Every "fits 8 GiB" is in nvidia-smi poller units on a 24 GiB card; a physical 8 GiB card has its own driver reserve, and in the CUDA runtime's view the explicit C82 configuration reads 9,167.6 MiB (the auto configuration that fits that view is 65 slots, 100.7 tok/s). (2) Lazy KV is measured at short contexts; at a filled 262k window it gains nothing. (3) The deadlock fix is established by a forced reproduction and by the structure of the handshake (the engine thread depends on no CUDA call); the natural rate (0.6% per resized run) cannot be re-measured with the runs available. (4) Reliability under full CPU saturation is bad (0.35 tok/s with all 32 logical CPUs busy; 34 tok/s with 4 engine threads). (5) The nested llama.cpp tree is uncommitted (D17); its record is `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch`.
- **For the owner.** The five items in the plan's owner summary, plus: the VRAM accounting caveat, and whether lazy KV (a driver-level VMM buffer for the KV cache) is wanted outside this experiment.

## 2026-10-11 (GPU held for 15 sessions through `tools/with-gpu-lock.ps1`, none with `-AllowUnprovenanced`; every wrapper verdict PASS with the binary attested): G2 decode-phase cache resize -- the LM-hosted hybrid reaches about 2x stock inside the same 8071 MiB peak; plus the idea ledger the stop-hook asked for

**Asked.** The stop-hook said the goal also requires every failed idea to be disproven after full implementation, with its root cause, and that the transcript did not show it. Re-reading the plan and auditing each disproof exposed two unexplained numbers (+1.0 GiB and about 19% between the LM-hosted hybrid and the same engine in plain `llama-completion`), which led to a further optimization.

**Answer.** (Agent's own acceptance, NOT owner approval; contract G2 pre-registered in `docs/plans/tg-beat-stock-8gb-262k.md`, mechanical check `python tools/check-g2-contract.py`.)

- **Root cause of the VRAM gap (measured).** The hybrid's 1024.1 MiB cache is allocated at context creation, while the prefill weight slots (996.0 MiB) and slab (8.0 MiB) are also resident. The 8071 MiB whole-run peak is the PREFILL peak; decode runs near 7.07 GiB.
- **The 19% was mostly a content confound (measured).** Plain `llama-completion` falls into a repeating loop on the anchor prompt (distinct-gram ratio 0.07 vs 0.62), which raises expert reuse (hit rate 0.545 vs 0.337). On prompts where every engine writes the same text the harness effect is +2.3% (code) and +4.3% (prose). The engine's own timeline gives the same host gap (about 2.0 ms) in both harnesses.
- **The optimization.** `--tg-hybrid-slots N` (opt-in) resizes the cache after prefill released its slots. Pre-registered prediction +13..17%; measured **+15.9% code (67.28 -> 78.01 tok/s), +15.8% prose (66.35 -> 76.86), +17.2% anchor (62.21 -> 72.92)**; 2.06x / 2.03x / 1.93x stock; whole-run peak unchanged at 8071 MiB; needle QA passes; logit drift 0.10-0.17.
- **Not worth it (measured).** Smaller prefill ubatch (peak -152/-228 MiB, prefill -5%/-11%), engine tuning (B, lambda, threads, poll: all within +-2%), backend sampling (79.10 vs 79.14 tok/s). Cache size keeps paying (C30 79.0, C36 82.5, C42 86.1, C48 88.4 on the code prompt) but the 8 GiB line stops it near C30.
- **Reliability (measured).** One device-side timeout trap (a host reply later than the 500 ms device limit) in about 29 early resized runs. An 80-run soak (40 unresized, 40 resized, interleaved, watchdog on) then ran with **0 failures** and replicated the gain (68.01 -> 79.81 tok/s, +17.3%). The soak did show ~103 ms host-engine stalls in 5 of 80 runs, in both arms, all inside the first 22 runs while other host activity ran; the last 58 runs, with the session idle, had none. So the ~103 ms stalls come from host contention on the engine's pinned spinning threads, not from resize. **CORRECTED by G5 (entry above): the one timeout trap was NOT a stall's tail; it was the first-pad-boundary deadlock between the GPU spin-wait, the launching thread's driver lock and the engine's copy submission.**
- **The doc's own tool, `llama-server` with the owner's exact flags (measured; warm = requests 2-3 on the same server instance, 2 launches).** Stock reproduces the doc here: first request 27.6 / 28.7, warm 33.8 / 33.0 tok/s (code / prose) against the doc's 28.05-30.68. The hybrid reaches **76.2 / 77.3 tok/s at C28 (2.26x / 2.34x stock; peaks 8179 / 8233 MiB, the prose run 41 MiB over 8192)** and **75.0 / 75.1 at C26 (2.22x / 2.28x; peaks 8043 / 8097 MiB, inside the cap)**, i.e. about 2.4x the 30.68 bar at 8 GiB. The first request is slower for both (stock 27.6, hybrid 47.5): the server loads the experts with mmap and the cold pages cost a one-time warm-up (a 24-token prefill took 2.4 s).
- **New code, all default off.** `llama_moe_hybrid_resize` / `--tg-hybrid-slots`, the watchdog, `tools/hybrid-timeline.py`, `tools/run-server-pair.py`, `tools/check-g2-contract.py`.
- **For the owner.** Same list as the G-TG8 entry below, plus: the one unexplained timeout trap.

## 2026-10-11 (GPU held for 14 sessions through `tools/with-gpu-lock.ps1`, none with `-AllowUnprovenanced`; every wrapper verdict PROVENANCE PASS, `gpu_pre`/`gpu_post` clean): G-TG8 -- the LM-hosted AER hybrid decode beats stock TG 1.65x within 8071 MiB; the agent's own acceptance against a contract written before the runs

**Asked.** (Owner `/goal`, autonomous mode) Beat the TG of `docs/llamacpp 8gb vram 262k base.md` (30.68 tok/s) at an 8 GB VRAM / 262k shape with caching and scheduling; prove it; find the root cause of any idea that fails.

**Answer.** Accepted by the agent (NOT owner approval) under `docs/plans/tg-beat-stock-8gb-262k.md`, "ACCEPTANCE CONTRACT G-TG8": all nine criteria hold (`python tools/check-g1-contract.py`).

- **Speed (measured).** Identical 511-token prompt, 512 TG tokens, 3 reps interleaved in one mutex hold: hybrid (C14 + `--kv-graph-pad 256` + `--act-slab-dtype f32`) **61.73 tok/s** (61.19-62.36); same-harness control (hybrid off) 34.22; stock `llama-completion -ncmoe 37` **37.35** (37.28-37.39). 1.65x stock, 1.80x the control, 2.01x the owner's bar. Whole-GPU peak 8071 MiB (stock 7443, control 7045).
- **Holds elsewhere (measured).** Free prose 1.74x and code 1.77x over stock; a uniform frequency prior is as fast (62.71), so nothing depends on where `aer-freq-train.txt` came from; the best-tuned stock under 8 GiB is about 38 tok/s (thread and offload sweep), so the control was not under-tuned.
- **Answer stability (measured).** Needle QA (positive passes, negative control fails) at 50 / 511 / 4096 / 24000 prompt tokens; the 24000-token run peaks at 8439 MiB (reported, not gated). A 2048-token generation is no more repetitive than the control (distinct-6-gram ratio 0.218 vs 0.230).
- **Prefill agreement (measured).** On the same prompt the LM last-position top-2 (ids 271, 198) differ by 3.514 logits; stock's gap, by `--logit-bias` bisection, is 3.375 (`-ub 128` 3.625, f16 KV 3.375): Delta 0.11-0.14 against a 2.42 limit.
- **Corrections to earlier entries.** The 63.6 vs 36.3 (1.75x) headline is superseded: its stock binary was the waived, unattested `aer-v4` build and the prompts differed by one token. The earlier "LM prefill differs from stock" (token 1 `Q` vs `\n\n`) was that same mismatch: `llama-completion -f` strips ONE trailing newline (`common/arg.cpp:1796-1799`), the LM runner does not. Use `anchor-512-nonl.txt` for any engine-vs-engine run.
- **New code, default off.** `pp_last_logits` telemetry (top-8 at the last prompt position, only with `--tg-logit-margin N>0`; a default trace has none, checked); `tools/check-repetition.py` (selftest 5/5), `tools/stock-logit-gap.py`, `tools/check-g1-contract.py`; `tools/run-tg-arms.py` gained interleaved reps and stock-engine arms.
- **Checks.** Host-suite engine steps 78/78 on the g1 build. The suite's own first step, the doc-size guard, FAILS (`docs/plans/generic-prefill-layer-residency-plan.md` 485,036 B vs 480,000 B, unchanged since `a5fa223`, predates this work; amendment A1 evaluates the engine steps directly).
- **Gaps / risk.** The nested llama.cpp changes are uncommitted because its `AGENTS.md` demands per-commit human approval; they are saved as `docs/plans/tg-beat-stock-8gb-262k-nested-worktree.patch` (sha256 prefix a8e83ca67e44354f). About 17.5 GiB of pinned RAM and 8 spinning CPU threads; PP is not claimed (317 vs 349 tok/s at 511 tokens, stock ahead); one model, one GPU, contexts filled only to 24k; text after the answer is junk in every engine because `--ignore-eos` is on.
- **For the owner.** Pinned RAM and spinning threads; the nested commit policy; the doc-size debt that fails the host suite; whether `--tg-first-pp-token` becomes the default; whether to open pushes (this goal needed none). Decisions D14-D20 are in the plan.

## 2026-10-10 (GPU held, `tools/with-gpu-lock.ps1` tag `gplr82-oprows-cuda`, `gpu_pre`/`gpu_post` clean; verdict UNCERTIFIED for the pre-existing dirty nested tree): T-GPLR-82 -- the per-op row-count hypothesis is REFUTED by measurement, and the handoff for the next agent is written

**Asked.** Test per-op whether the same expert rows produce different values when computed in a 2-wide versus an 8-wide `MUL_MAT_ID` batch, to explain the 1.3e-3 relative step-0 shift that survives with no CPU compute.

**Answer: they do not. The hypothesis is refuted.**

- Built `third_party/llama.cpp/tools/layer-streaming/op-rows-probe.cpp` (target `op-rows-probe`, CTest `layer-streaming-op-rows-cpu`): ONE synthesized Q4_K/Q6_K bank at the model's real geometry (k=2048, m=512, `n_expert`=256, `top_k`=8), running the 8-wide whole op twice (the noise floor), the same rows in 2-, 1- and 6-wide batches, the weighted 6+2 split against the weighted 8-wide whole, and both legs inside ONE graph against the same legs alone. It writes JSON per run; `tools/check-op-rows-probe.py` classifies that JSON against the pre-registered bands, has `--selftest` (5 bands, a failed shape, an unreadable file) and is registered in the host suite.
- Result over four runs -- CUDA0 tokens=1 Q4_K, CUDA0 tokens=8 Q4_K, CUDA0 tokens=1 Q6_K down, CPU tokens=1 Q4_K -- is **42/42 comparisons bit-equal, worst `max_rel` = 0.0, `n_diff` = 0** (`build/llama-cuda/op-rows-{cuda-t1,cuda-t8,cuda-q6,cpu}.json`; wrapper verdict `build/llama-cuda/verdicts/gplr82-oprows-cuda.done`). A row's value is bit-identical computed alone, in a 2-wide batch, in the 8-wide whole, or beside its sibling leg in the same graph, and the CPU backend is bit-exact across all of them too.
- So the split's remaining shift is NOT a batched-GEMM row-count artifact. What this does not cover: the model's own aggregation order (the probe sums both arms the same way, so it is insensitive to reassociation), the real router-weight derivation, and realistic activation data. The candidates left are the ones that change the DATA a leg sees rather than the kernel: the expert-id/router-weight pairing across the two legs, and the tail leg's activation buffer.
- Incidental: the probe looks its device up before calling `ggml_backend_load_all()`, but CUDA registration is static in this build, so even a CPU run enumerates the CUDA driver (no VRAM, no compute). Treat the new CPU ctest as CUDA-touching for mutex purposes.
- Checks: the host suite still exits 1 on the pre-existing doc-size FAIL for the plan file; `resplit-currentstate.py --verify` PASS; `git diff --check` clean; the new `layer-streaming-op-rows-cpu` CTest PASS (68 s). **Nothing committed.**
- **Handoff for the next agent: `docs/handoff-gplr82-attribution.md`** -- the question chain with its answers, the probe and its refutation, nine failure-and-fix entries, the still-open guard gap (a NaN on the CPU's q8_K path is silently zeroed, so `host_finite` cannot see it), the exact commands, and the ranked next steps.

## 2026-10-10 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`, `gpu_pre_run`/`gpu_post_run` clean): T-GPLR-82 ATTRIBUTION (an AC1 follow-up, plan `docs/plans/expert-compute-staging-decoupling.md`) -- the CPU tail leg is NOT the cause of the non-finite cascade; the S3 copy-wait decoupling is. Two effects, two root causes, separated in a 2x2

**Asked.** Find why the CPU tail leg drives the run to non-finite logits on 506/512 steps while a device tail stays finite, and whether the two effects share one root cause.

**Answer: they do NOT share a root cause.** The non-finite cascade is caused by the S3 copy-wait decoupling (budget-dependent admission), not by the CPU leg's arithmetic. The CPU leg has its OWN, separate effect: it stays finite but destroys the answer.

**The 2x2** (one mutex hold, session `gplr82-s3off`, exe `a9ddf1699801b235`, dll `c11ac969d739f56d`, anchor `d5bfd34a8a795e49`, 512-token anchor, `--ubatch 512 --residency-tg-retain 2`, 512 TG tokens, 165.2 s wall). `s3_cap = floor(budget / 36.5)` clamped to `experts_host_n`, so budget 0.001 => cap 0 (decoupling OFF) and budget 100 => cap 2 (same admission as `inf`).

| arm | flags | step-0 id | step-0 logit | non-finite | ffn_dev | host_shr (plan) | wait_skipped | QA |
|---|---|---|---|---|---|---|---|---|
| C0 | `--experts-host 0` | 248046 | **16.194481** | 0/512 | {1: 20480} | 0.00000 | 0 | PASS (neg. control FAILs) |
| B1 | `--experts-host 2 --experts-host-budget 0.001` | 239541 | 16.214243 | **0/512** | {2: 20480} | 0.00000 | **0** | FAIL (all 5 needles missing) |
| B2 | `--experts-host 2 --experts-host-budget 100` | 239541 | 16.167877 | **506/512 (from step 6)** | {2: 20480} | 0.02065 | **564** | FAIL (all 5 needles missing) |

- C0 reproduces the recorded control **exactly** (16.194481) on the new binary, and B2 reproduces the recorded A2 exactly -- so the comparison is valid.
- B1 runs the CPU tail leg on all 20,480 records (census part B: 40 VIEW + 120 MUL_MAT_ID + 40 GLU host-placed, the same shape as A2) and is **zero** non-finite. **So the CPU tail leg does not produce non-finite logits.** B2 -- whose only change is that the decoupling is ON -- is 506/512 non-finite with 564 skipped waits. The cascade is the decoupling's.
- B1's `host_shr` is 0 for a mechanism reason, not because the leg is idle: with the decoupling OFF every demanded row is already staged when the plan runs, so `ls_plan_split` binds no row to the host and the leg executes the plan's FIXED-WIDTH FILLERS (`ffn_dev = 2` on all 20,480 records, `experts_host_sum = 0`, census `part_b_host=3`). B1 therefore isolates exactly one variable -- CPU tail leg present, decoupling off -- and B2 differs from it only by `s3_cap` 0 -> 2.
- Cross-check with the device tail: session `a2dev2` (dll `eb878312`, before the `!experts_host_leg_device` conjunct existed) ran this same `--experts-host 2 --experts-host-budget inf` split with `--experts-host-leg device`, so its graph had NO CPU leg (`ffn_dev = 1` on all 20,480 records, `experts_host_sum = 0`), and was 512/512 non-finite from step 0 with all 512 logit lines unparsable; session `a2dev3` (dll `c11ac969`, after the conjunct) is that same arm with the waits taken and is 0/512. So the decoupling predicts finiteness in all four cells -- (ON, CPU tail) 506/512, (ON, device tail) 512/512, (OFF, CPU tail) 0/512, (OFF, device tail) 0/512 -- while tail placement does not. Session `a2dev` (dll `f8e54ffd`) is NOT a data point: it died in the graph build, rc 0xC0000409, zero records.
- B1 against session `a2dev3` isolates the CPU tail leg's OWN effect: both run the split in the same filler configuration (`host_shr` 0 / `experts_host_sum` 0), both are finite, and in a2dev3 the waits are taken by construction (the `!experts_host_leg_device` conjunct forces it in that mode) -- yet B1 (CPU tail) misses all five needles while a2dev3 (device tail) passes QA. The CPU tail leg is therefore a real second defect: finite, but answer-destroying.
- **Per-step mechanism (the decoupling is NECESSARY, and its effect is DELAYED).** Using the runner's own criterion -- a `rowset_fetch` with `n_rows_fetched > 0` whose `ready_latency_us` is 0, i.e. rows were demanded and no wait occurred (`run-gplr82-s3off.py:254`) -- the counts per arm are C0 **0**, B1 **0**, B2 **564**. Ordered by `t_us`, B2's `chunk` 1 is the 40-layer pass that produced step 0's readout (clean, 16.167877, 0 skipped waits) and chunk N produces step N-1's readout from there; the first skipped wait is in chunk 2 (3 of 40 layers) and the passes behind steps 1..5 contain 3, 12, 15, 21 and 17 skipped waits **while all five of those readouts stay finite** -- the first non-finite readout is step 6. So the corruption needs ~5 TG steps to surface. That is the signature of KV/hidden-state accumulation feeding back, not of an immediately-bad read, and it also explains the asymmetry with the device leg: `a2dev2` (device tail) was non-finite from **step 0**, i.e. its PREFILL readout was already corrupt, while B2's prefill readout is clean and finite at 16.167877. The candidate explanation is the same masking mechanism as below -- the CPU leg's q8_K path silently converts NaN contributions to zero, so the delay IS the masking. Flagged as cross-binary (a2dev2 is dll `eb878312`); the delayed onset in B2 vs the immediate one in a2dev2 is consistent-but-not-isolated evidence, since no single-binary run yet has both a device tail and a clean prefill.
- The CPU leg's own effect is instead a **finite** output change: step-0 logit moves +0.019762 and the answer is destroyed (0/5 needles) with no NaN at all. It is not a precision artifact: same magnitude, opposite sign to the device tail's -0.021538 (A2dev).

**Why the existing loud-failure guard cannot see it** (source-verified, not inferred). `scan_host_expert_outputs` posts `host_finite = 0` on all 2,024 host records **including B2's 506 non-finite steps**. Cause: the host leg's expert bank is Q4_K and ggml's CPU quantized matmul quantizes its F32 activations to q8_K first. `quantize_row_q8_K_ref` (`third_party/llama.cpp/ggml/src/ggml-quants.c:2768-2785`) computes `ax = fabsf(x[j]); if (ax > amax) { amax = ax; max = x[j]; }` -- for NaN that comparison is false, so `amax` stays 0 and the block is written `d = 0` with serialized zeros. **A NaN activation block therefore becomes a ZERO contribution on the CPU's Q4_K path and the per-row output stays finite.** The scan covers only `ff_host_experts[il]` -- the host leg's own per-row outputs -- so a corruption that instead lands in the F32 router logits or in the device leg sits outside its view. This is a guard gap, not evidence that the run is healthy.

**Tooling defect found while grading.** The runner's QA subprocess crashes with `UnicodeEncodeError` (cp1252) while PRINTING the non-ASCII tg-text, so the recorded lines `positive_pass=False` and `negative_fails_as_designed=False` for B1/B2 are crash artifacts, not grades. Re-run with `PYTHONIOENCODING=utf-8`: C0 positive PASS / negative FAIL (fails as designed); B1 and B2 positive FAIL (all five needles absent). The negative arm PASSES for B1/B2 -- correct behaviour, not a defect: it certifies the needles are ABSENT, and a destroyed answer is exactly that -- so the negative control is only informative on an arm whose answer is correct. The same crash affected the recorded A2 QA lines; their FAIL classification is semantically right, but the crash must not be read as a grade.

**Still unexplained (do not claim otherwise).** Why the CPU leg's two rows change the answer while the device leg's do not, given comparable step-0 shifts. Leading candidates: the CPU's q8_K activation quantization (256-element blocks; the CUDA MMQ path quantizes in a different format and distribution) and the cross-backend copy of the strided `w_full` suffix view. Cheapest next test: build both tail legs in one graph (device as a probe) and log the per-row max relative difference for the same rows and weights.

**Validation.** `test-layer-streaming-expert-split` PASS; all layer-streaming host unit tests PASS. The backend-op census was re-derived from the three arm traces with the `graph_backend_ops` / `layer_compute` reader as written in `tools/run-gplr82-a2dev.py`: C0 is `part_b_host=0 part_b_device=15` with `ffn_dev = 1` x20,480; B1 and B2 are both `part_b_host=3 part_b_device=15` = {VIEW 40, MUL_MAT_ID 120, GLU 40} with `ffn_dev = 2` x20,480; only B2 has host-bound records (2,024) and every one of them reports `host_finite = 0`. Runner `tools/run-gplr82-s3off.py`; verdict `build/llama-cuda/gplr82-s3off/gplr82-s3off-verdict.json`; arm traces/logs/tg-texts in `build/llama-cuda/gplr82-s3off/`; session log `build/llama-cuda/s3off-session.log`. **Nothing committed.** AC1 remains FAIL on the shipped split regardless of this attribution: both split arms miss all five QA needles and their step-0 top-1 id is 239541 against the control's 248046.


## 2026-10-10 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`): T-GPLR-82 falsifier -- the split's step-0 shift is NOT the CPU's; a device-only 6/2 partition still moves logits by 0.021538

**Asked.** Run the same 6/2 routed-row partition ENTIRELY on the device and check whether step-0 logits return to 16.194481 exactly, which would confirm the CPU accumulation as the cause of A2's corruption.

**It could not be run with the existing binary, and arm A2d was not it.** A2d is `--experts-host 2` at the budget's default 0.0, so `ls_split_in_play` is false and the builder takes the UNSPLIT FFN branch -- A2d is the inert arm (bit-identical to A0), not a device-only partition. The shipped host leg is CPU-placed BY CONSTRUCTION: `build_moe_host_leg` binds `model.layers[il]`'s host-placed banks, so no flag could move those rows. A diagnostic mode was therefore added: **`--experts-host-leg device`** (default `cpu`; absent = every existing command line unchanged). It builds the SAME constant-width tail leg -- same widths, same ONE `w_full` derivation, same positional reduction via the same add chain -- against the SLOT layer's staged tensors, so the tail runs on the device; it builds no host sub-op and no `hsel` input, and leaves `row_set_split_built` at 0 so every host-side executor path (plan reorder, id fill, per-row scan) stays off. It is a diagnostic partition, never a placement policy.

**Arms** (one mutex hold, gpu_pre/post clean; 512 anchor, `--ubatch 512`, `--residency-tg-retain 2`, 512 TG tokens; session `gplr82-a2dev3`, exe `a9ddf1699801b235`, dll `c11ac969d739f56d`):

| arm | step-0 id | step-0 logit | margin | delta vs A0 | host share | `ffn_dev` | non-finite steps | QA |
|---|---|---|---|---|---|---|---|---|
| A0 | 248046 | 16.194481 | 0.524198 | 0 | 0.00000 | 1 (all device) on 20,480 | 0/512 | PASS; negative control FAILs as designed |
| A2 | 248046 | 16.167877 | 0.511299 | -0.026604 | 0.02065 | 2 (mixed) on 20,480 | **506/512** | FAIL |
| **A2dev** | 248046 | **16.172943** | 0.492351 | **-0.021538** | **0.00000** | **1 (all device) on 20,480** | **0/512** | PASS; negative control FAILs as designed |

A0 reproduces the recorded control (16.194481 / 248046 / 0.524198) EXACTLY on the new binary, so the comparison is binary-to-binary valid and the new flag is inert when absent.

**The arm really computed no routed row on the CPU** (census is per run over 40 layers): A0 `part_b_host == {}`; A2 `part_b_host == {VIEW 40, MUL_MAT_ID 120, GLU 40}`; A2dev `part_b_host == {VIEW 80}` -- VIEWs only, which are metadata aliases with no compute -- while `part_b_device` carries `MUL_MAT_ID 240` (= head 120 + tail 120) and `GLU 120` (= 80 + 40). Ledger: `experts_host` 0 of 163,840, `host_finite` -1 (no host leg exists), `ffn_dev` 1 on every record.

**Consequence -- the falsifier's criterion did NOT hold.** With no CPU row in the partition, ~81% of the shipped arm's step-0 shift survives, so CPU accumulation is NOT the cause; it can account for at most the 0.005066 remainder (and less, since A2dev's membership also differs). The two-leg partition's OWN arithmetic is the dominant contributor. A 6+2 regrouping of F32 partial sums should be ~1e-7 relative, not 1.3e-3, so the mechanism is NOT simple reassociation and remains UNPROVEN -- that is now the open question, and the previous entry's "precision signature, therefore the CPU" reading is withdrawn. A second, separate finding: the CPU tail is what makes the run non-finite (506/512 readouts, QA FAIL) while the device tail stays finite (0/512, QA PASS), so the catastrophic divergence is a CPU-leg property distinct from the 0.02 shift.

**Two defects found while building the arm, both fixed in the experimental path only.**
1. A suffix view of an I32 `[top_k, n_tokens]` tensor with the parent row stride is ILLEGAL in ggml (`data_size + view_offs <= ggml_nbytes`): measured GGML_ASSERT at `ggml.c:1787`, rc 0xC0000409, while building the persistent graphs. The diagnostic swaps sides instead -- tail at offset 0, head at `host_rows * len` -- where the wider head view ends exactly at the parent's last byte.
2. The executor's `s3` decoupling predicate (`llama-layer-streaming.cpp`, `load_row_set`) consulted the flag and the budget but not WHICH graph was built, so with an unbounded budget it skipped waits for rows no host leg could absorb: measured 512/512 non-finite readouts with `ffn_dev` = 1 and a zero host ledger. Fixed by requiring a real host leg (`!experts_host_leg_device`). The default path is unchanged.

**Both refusals of the new flag were exercised end to end** (not asserted): `--experts-host-leg bogus` exits 2 with `error: --experts-host-leg expects 'cpu' or 'device' (got 'bogus')` at arg parse (no GPU); and `--experts-host-leg device` without `--experts-host` exits 1 under the mutex with `error: --experts-host-leg device re-places the --experts-host split's tail leg; without --experts-host no split is built...` and the `experts_host_leg_without_host` refusal event (`build/llama-cuda/a2dev-guard4.log`, `verdicts/gplr82-a2dev-guard4.done`).

**Evidence.** `build/llama-cuda/gplr82-a2dev3-verdict.json`; session log `a2dev-session3.log`; per-arm `gplr82-a2dev3-arm-{A0,A2,A2dev}.{jsonl,log,txt}`; wrapper verdict `verdicts/gplr82-a2dev3.done`. Control binary side-copied BEFORE the rebuild: `build/llama-cuda/bin/a2dev-control-4412cc4a/` (+ `CONTROL-MANIFEST.json`, exe 4412cc4a60309fd3 / dll 2e96ab55278a9c9c). Build logs `a2dev-build{,2,3,4}.log`; host test run `a2dev-host.log` vs `a2dev-host-control.log`.

**Known gaps.** (1) A2dev differs from A2 in membership as well (A2's is plan-chosen per token; A2dev keeps router order), so 81/19 is indicative, not an exact decomposition. (2) Timing here is NOT evidence: single runs, and A2's 19.96 tok/s comes from a numerically corrupt arm -- do not quote it as a speedup (A2dev 15.82, A0 16.56, all within run-to-run noise of one arm each). (3) `--experts-host-leg device` costs the mechanism its point (the tail rows must be resident by then) and is for attribution only. (4) AC1 remains FAIL on the shipped split; nothing here changes that. (5) The host suite exits 1 on the PRE-EXISTING doc-size FAIL (`docs/plans/generic-prefill-layer-residency-plan.md`, 485,036 B), which aborts its later checks; the layer-streaming ctest family fails the same set of `-run`/preflight tests under BOTH this binary and the preserved control, so those failures are environmental, not from this change.


## 2026-10-10 (host-only, no GPU hold): `docs/currentstate.md` resplit into verbatim shards -- search no longer skips it; `--verify` PASS, 69 entries hash-exact

**Why.** `docs/currentstate.md` was 499,217 B, past the guard's 480,000 B FAIL band and at/over the observed per-file `ctx_search` cap, so a search for text known to be in the file returned "1 large file skipped" instead of matches -- the canonical current-state record was invisible to search.

**What was done.** `tools/resplit-currentstate.py` with cutoff `2026-10-03`: entries dated 2026-10-03 .. 2026-10-10 stay live; **51 older entries moved verbatim** into three new per-day shards (`2026-09-25.md`, `2026-10-01.md`, `2026-10-02.md`); the live file keeps a derived index (281 rows) plus the non-entry "Known gaps / risks" and "Blocked / escalated" tail, unchanged. Manifest `docs/currentstate/resplit-manifest-2026-10-10.json`; method recorded in `docs/AGENT_ENVIRONMENT.md`, "Where current state lives".

**Leftover disposition (owner-approved; recorded because it touched history).** The live file still held two 2026-09-24 entries and `docs/currentstate/2026-09-24.md` already existed, so the archiver would either ABORT or (with `--overwrite`) regenerate that shard from only those two entries and destroy the other 16. Measured at body level first: the live `T-GPLR-07 CUDA SEMANTIC-QA SMOKE TIER PASS` copy was **byte-identical** (sha `577d1141bc9c8ba7`, 4,078 B) to the shard's copies, so the live duplicate was **dropped** and the shard's own duplicate copy removed with it; `2026-09-24 - Nested engine rebuild and host-suite provenance recheck` (sha `3ea1eb9535bd9080`, 3,116 B) existed **only** in the live file and was carried into the shard **verbatim**. Reconciliation: 57,602 B -> 56,640 B = **-4,078 + 3,116**, exact.

**Evidence.** `python tools/resplit-currentstate.py --verify` -> **PASS, exit 0**: apply-time source 492,021 B sha `982362a4a8464310c94b48bedbfce6b0296571617a998fe61ede875434cffeb4`; 13 preexisting shards byte-exact; 3 new shards body-exact; **69 entries present and hash-exact**; 281 index rows derived from every shard; live **171,620 B** (LF-normalised, the figure `--verify` prints). On disk the live file is 173,416 B in CRLF (1,796 lines, so the same content; the repo line-ending convention is preserved) -- **35.8% of the FAIL band**. **Search verified through the interface, not inferred:** `ctx_search` now returns the 2026-10-10 A2 lead paragraph at `docs/currentstate.md:28`, returns the moved `Nested engine rebuild` entry at `docs/currentstate/2026-09-24.md:185`, and an unscoped search no longer lists a skipped large file.

**Guard.** `python tools/check-doc-sizes.py` still exits 1, but no longer because of this file: `docs/currentstate.md` is out of the band (FAIL and WARN both cleared). The remaining **FAIL** is `docs/plans/generic-prefill-layer-residency-plan.md` (485,036 B) and the new **WARN** is `docs/plans/generic-prefill-layer-residency-archive.md` (353,234 B) -- both pre-existing and outside this change, both with a documented splitter (`tools/resplit-gplr-plan.py`).

**Known gaps / residual risk.** (1) The 3 new shards and both resplit manifests are **untracked** (only the 13 older shards are tracked), so the guard's 241-tracked-file scan does not cover them and a stray clean would delete them; no commit was authorised. (2) `docs/currentstate/2026-10-02.md` is **266,625 B** -- the largest shard, above the 250,000 B WARN band; a future re-cut should re-cut that day rather than assume it is safe. (3) The superseded `resplit-manifest-2026-10-02.json` still records `docs/currentstate/2026-09-24.md` at 57,602 B (stale after the dedupe) and stores **no** new-shard body hash, so its own `--verify` would fail on that row without naming the cause; `resplit-manifest-2026-10-10.json` is the live record.


## 2026-10-10: T-GPLR-82 arm A2 corruption LOCALIZED -- the host leg is FINITE everywhere; the split moves step-0 logits by 0.16% and greedy decoding amplifies that into total divergence

**Session** `gplr82-ac1-post` (exe `4412cc4a60309fd3`, `llama.dll` `2e96ab55278a9c9c` -- the binary with both halves of the S3 split-shape fix). Traces `build/llama-cuda/gplr82-ac1-post-arm-{A0,A0b,A2,A2d}.jsonl`, verdict `gplr82-ac1-post-verdict.json`. All from the recorded traces; **no new GPU arm was run**.

**1. The asked question -- which ops sit on which side (per layer, x40 layers, from the `graph_backend_ops` event).** A2's part B: HOST = `MUL_MAT_ID` x3 (gate/up/down for the 2 host rows) + `GLU` x1 + the host-side 6-element view of `sel_in` (`dst_off=24`, i.e. `dev_rows` ints); DEVICE = everything else, **including the whole weight derivation** (`GET_ROWS`, `SUM_ROWS`, `CLAMP`, `DIV` x1 each) and every weighting/sum (`MUL` x4, `ADD` x9, `VIEW` x10, `RESHAPE` x5, `RMS_NORM`, `SOFT_MAX`, `CPY`). `part_a_*` is **identical between A0 and A2**, so the attention path is untouched by the split. A0/A2d have `part_b_host == {}` -- no host leg at all.

**2. The host leg is not the source. `host_finite = 0` (= every host per-row output FINITE; `bad.empty() ? 0 : 1`) on all 20,480 `layer_compute` records**, and the `host_finite == 1` abort never fired. It is 0 even on the **18,456** records where `experts_host == 0` (the plan admitted no host row), which is direct proof the fixed-width host leg executed with filler rows on every record. So no NaN/inf enters through the host leg, and the corruption lies OUTSIDE the region that guard covers.

**3. The divergence is numeric and present at step 0, with a measured drift floor of 0.0.** Step-0 readouts are **bit-identical** across A0, A0b and A2d (`top1_id 248046`, `top1_logit 16.194481`, `margin 0.524198`); **A2** reads the same `top1_id` with `top1_logit 16.167877`, `margin 0.511299` -- a **0.026604 (0.164%)** logit shift. Step 1 then leaves the control entirely (A0 `id 198 @ 33.641617` vs A2 `id 220 @ 13.967248`), and 506 of 512 steps go non-finite. Magnitude ~1.6e-3 is a *precision* signature, not a wrong-row or NaN signature (a wrong expert row moves logits by O(10%)).

**4. The row-cache collapse is an EFFECT, not the cause (checked to avoid a wrong attribution).** A2 fetches only **3,893** rows against A0/A2d's **94,671**, and reports **159,947** resident against 69,169 -- but token 0's fetch pattern is identical to A0 (320 fetched / 0 resident), tokens 1-5 differ only mildly (175 vs 152, 146 vs 234, ...), and the zero-fetch pattern (A2 **19,186/20,480** vs A0 232/20,480) accumulates only after the logits degenerate. Routing collapses because the hidden state blew up.

**5. Two hypotheses killed cheaply (recorded so they are not re-derived).** (a) *Omitted router bias*: this model has **no** `ffn_gate_inp_b` / `ffn_exp_probs_b` (no loader key exists; qwen35moe.cpp never binds one), so the split's `h_probs` and the unsplit path's `probs` are the same softmax; (b) *different weight derivation*: the split's gather/norm/scale is line-for-line the same order and the same `6.103515625e-5` clamp as `build_moe_ffn`'s `norm_w` block and the same `expert_weights_scale`, and both legs view ONE `w_full`.

**Leading hypothesis (unproven, and the next check is named):** the split computes 2 of the 8 routed rows with the **CPU** dequant/accumulation and 6 with CUDA's, for a Q4_K_S bank; those are not bit-identical, so the split is not numerically equivalent to the unsplit FFN, and AC1's near-tie rule with a measured drift floor of **0.0** cannot excuse any of it. **Cheapest falsifier:** one arm executing the SAME 6/2 partition entirely on the device (or two device legs). Step-0 logits returning to `16.194481` exactly would confirm the CPU accumulation; a persistent shift would instead mean the host rows get wrong weights/ids. Note the contract consequence: if this holds, AC1 as written and a CPU host leg are in conflict.

## 2026-10-10 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`, gpu_pre/post clean): T-GPLR-82 -- the SPLIT POINT is fixed: the split graph is now built only when the mechanism can admit a row, restoring the closed-gate property (A2d becomes IDENTICAL to the control over 512 tokens); the remaining host-leg execution defect is thereby ISOLATED

**Task.** Owner-directed, following the AC1 FAIL of the previous entry and the falsification of the ordering/pairing hypothesis: make the graph's device/host split follow the plan's disposition instead of the configured width, so a closed gate reproduces the unsplit device leg, with a failing-first host-only test. Scope: the vendored tree, opt-in `--experts-host` path only.

**Mechanism, in one paragraph.** The split FFN graph fixed its widths at BUILD (`dev_rows = n_expert_used - host_rows`, `host_rows = ls->experts_host_n`, qwen35moe.cpp:435,441) and filled the host leg's id input UNCONDITIONALLY from the tail of the row. Whatever the plan decided, the shape committed `host_rows` rows per (layer, token) to the CPU. The plan needs `experts_host_budget_us > 0` before it can admit a single row (`s3`, load_row_set:10437-10443), so with the budget at its default 0 the split changed the arithmetic while the ledger honestly reported zero host rows. The fix makes the builder ask the same question the executor asks, via a new pure predicate.

**Changes.** (1) `src/llama-expert-split-policy.h`: new `ls_split_enable` + pure `ls_split_in_play(e)`, carrying exactly the executor's admission conjuncts (width, budget, recon, spec_on, spec_mode, spec_arm, spec_pool) minus `per_token != nullptr`, which is a call-site property unknowable at build. (2) `src/models/qwen35moe.cpp`: the split branch is gated on that predicate, records `row_set_split_built[il]` behind a loud bounds check, and emits one `LLAMA_LOG_WARN` when the flag is set but the mechanism is off. (3) `src/llama-layer-streaming.{h,cpp}`: `row_set_split_built` sized in `init()`; the executor's four `experts_host_n > 0` gates and five telemetry ternaries now read that flag, so the two halves cannot disagree. (4) `tests/test-layer-streaming-expert-split.cpp`: three new test functions.

**Failing-first evidence (recorded).** With the predicate transcribed as the shipped rule (`host_w > 0` alone): build 0, test **exit 1**, `AC1 FAIL (6 failures)` -- the zero-budget case and all five arm gates failed, while the two "in play" controls passed. After the fix: build 0, test **exit 0**, `AC1 PASS (0 failures)`. Contract form: `ctest --test-dir build/llama-cuda -C Release -R test-layer-streaming-expert-split --output-on-failure` -> `1/1 Passed`, `100% tests passed, 0 tests failed`. Both touched TUs compile with zero warnings (the `%d`/`size_t` warnings at llama-layer-streaming.cpp:5485-5602 are pre-existing and untouched).

**Two intermediate failures, both instructive and both fixed.** (a) With only the builder half in place, A2d **aborted loudly**: `layer_streaming_compute: host-leg id input missing or misshaped for layer 0 (expected [2, 1])` -- the executor still demanded an input an unsplit graph never creates. That is why the fix is two halves; it also confirmed the guard works. (b) Sizing `row_set_split_built` in `begin_row_set()` and again in `set_pp_prefetch_ahead()` produced an **access violation (0xC0000005) during "building persistent graphs"**: `begin_row_set()` runs AFTER the persistent graphs are built, and `set_pp_prefetch_ahead()` may never run at all. Sizing now lives in `init()` (line ~3181, immediately after `n_layer` is set) and the builder write is bounds-checked with a named `GGML_ABORT`.

**Post-fix measurement (session gplr82-ac1-post, exe 4412cc4a60309fd3, dll 2e96ab55278a9c9c; pre-fix evidence preserved as `gplr82-ac1-pre-*`).** A0 `--experts-host 0` and A0b (repeat) unchanged: QA PASS, agreement 1.0 over 512 steps, drift floor 0.0, host share 0.000000, ~15.6 tok/s. **A2d (`--experts-host 2`, budget at its default) is now IDENTICAL to the control: QA PASS (positive 0 / negative 1 as designed), top-1 agreement 1.0 over 512 steps, first divergence `None`, `ffn_dev = 1` on all 20,480 records** (pre-fix: `ffn_dev = 2` on all 20,480, agreement 0.00195, token salad from step 1). `check-telemetry` rc=0 on all four arms. The run states the outcome out loud, exactly once, in the A2d log and in no other arm: `T-GPLR-82: --experts-host 2 is NOT in play (budget 0.0 us/token, recon 0, ...) -- building the UNSPLIT FFN graph for every layer of this run, exactly as with the flag absent`. **A2 (`--experts-host 2 --experts-host-budget inf`) is UNCHANGED**: host share 0.020654, agreement 0.001953, 511 unexplained divergences, 506 non-finite logit steps from step 6, `ffn_dev = 2`. AC1 as written therefore still **FAILs**, on the split arm only.

**Why this matters beyond the flag.** The fix separates two defects that were previously indistinguishable. The *shape* defect (now fixed, measured) was: a graph that splits when the mechanism admits nothing. The *remaining* defect is the host leg's execution: with the split genuinely in play (2.07% of rows host-computed, 2,024 host-scan records all reporting `host_finite = 0`, i.e. every host per-row output FINITE), the logits still go non-finite at step 6 and stay non-finite for 506 of 512 steps, so the corruption is NOT inside the region the existing finiteness guard covers. That is now the single next thing to diagnose, and it has a clean A/B (A2 vs A0) with a same-binary control.

**Scope, risks, recommendation. Measured:** the RED/GREEN test, the ctest form, the A2d identity, the loud warning, `ffn_dev` histograms, and that A2 is unchanged. **Unproven:** the remaining host-leg execution defect (cause unknown; the 512-logit non-finite onset and the finite-host-output scan are the two facts any explanation must fit). **Not run:** AC5/AC6/AC7, the 50-token rung, the host suite beyond the split target. **Doc guard:** `python tools/check-doc-sizes.py` still FAILs -- `docs/currentstate.md` is now 499,917 B as measured by the guard at the time this entry was written (the guard reports the live size; it grows with each entry) and `docs/plans/generic-prefill-layer-residency-plan.md` 485,036 B, both past the 480,000 B band; this entry adds ~8 KB to a file that needs the documented shard split (pre-existing, not caused here). **Recommendation:** keep experimenting. The split-point fix is adoptable as written (it only makes the flag safer: with the budget at its default the run is now bit-identical to the flag-absent path, and it says so). Do NOT read A2's 20.64 tok/s as a speedup -- that arm is numerically corrupt. Evidence: `build/llama-cuda/gplr82-ac1-post-verdict.json`, `gplr82-ac1-post-launch.log`, `gplr82-ac1-post-arm-*.{jsonl,log,txt}`, `verdicts/gplr82-ac1-post.done`; pre-fix `gplr82-ac1-pre-*`; intermediate `gplr82-ac1-half-*`, `gplr82-ac1-crash*-*`.

## 2026-10-10 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`, gpu_pre/post clean): T-GPLR-82 AC1 run as written -- the gate FAILS, and it separates TWO defects: the split is wrong even with ZERO host rows, and with host rows it goes non-finite while the engine's own guard stays silent

**Task.** T-GPLR-82 **AC1**, owner-directed. AC1 has never had evidence: `tools/run-tests.ps1` never passes `--experts-host`, so the 2026-10-05 ladder that "passed for the host-leg fix" exercised the fix's regression test, not this mechanism. AC1 requires semantic-QA aggregate PASS at 50 and 512 tokens on the 512 anchor, `check-telemetry` OK, negative control FAILing as designed, top-1 agreement **>= 99%** against the control over **>= 512** tokens, and **every** divergence explained as a near-tie (top-2 gap below the measured drift).

**Recorded command.** `powershell -NoProfile -ExecutionPolicy Bypass -File tools/with-gpu-lock.ps1 -Command "\"C:/Program Files/Python313/python.exe\" C:/Development/AI/qwen38GZI/tools/run-gplr82-ac1.py" -VerdictTag gplr82-ac1 -TimeoutSeconds 2400 -AllowUnprovenanced`. New session runner `tools/run-gplr82-ac1.py` (host-only re-grade: `--grade-only`, no GPU, no mutex). Four arms, one binary, 512 anchor, `--tg-tokens 512`, `--tg-logit-margin 512`: **A0** `--experts-host 0`; **A2** `--experts-host 2 --experts-host-budget inf`; **A2d** `--experts-host 2` alone; **A0b** control repeat. Shape matches the T-GPLR-62 reference session (`--ubatch 512 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --allow-full-residency --no-handoff --residency-tg --residency-tg-retain 2`). Provenance: exe `66b4eb9eb3d1a1b6`, `llama.dll` `1eef7d4f24fa822d`, anchor `d5bfd34a8a795e49` (byte-identical to the calibrated copy whose item list `N1,N2,N3,N4,K1` the T-GPLR-62 session validated). Peak VRAM 6,483.56 MiB; session wall 210.4 s; `check-telemetry` rc=0 on all four arms.

**AC1 result: FAIL, on four independent counts.** Drift floor **0.0** (A0 vs A0b agree at 512/512 steps, top-1 agreement 1.0). A2 vs A0: **agreement 0.00195 over 512 steps, 511 divergent, 511 unexplained** -- the finite divergences (steps 1-5) carry control top-2 gaps of 16.49 / 17.10 / 1.04 / 0.34 / 18.44, i.e. confident disagreements, and a floor of 0.0 makes every one unexplainable. Semantic QA: A0 and A0b PASS with the negative control failing as designed; **A2 FAILs, A2d FAILs**. The 50-token rung was **not run**; AC1 requires PASS at both lengths, so the 512 failure already decides the gate.

**Two defects, separated by A2d.** (1) **The split corrupts the arithmetic with ZERO host rows.** A2d measured `host_share = 0.000000` and `sum(experts_device) == sum(experts_total) == 163,840` -- the ledger says every row went to the device -- yet its top-1 agreement vs A0 is **0.00195** (first divergence step 1, margins 1.70/3.19/4.39, confident) and it emits token salad from step 1 while `ffn_dev == 2` on **all 20,480** records (vs `ffn_dev == 1` on all 20,480 in A0). The graph is partitioned into device+host legs unconditionally when `experts_host_n > 0` (the host leg's id input is a positional copy of the router top-k suffix, `llama-layer-streaming.cpp` ~5176-5194), while the ledger reports the PLAN's disposition. So **AC2's `device + host == total` invariant holds in every arm (0 violations) while the arithmetic is wrong** -- exactly the failure `tools/run-gplr82-hostarm.ps1`'s own header warned about. (2) **With host rows, the logits go non-finite and nothing aborts.** A2 realised `host_share = 0.020654` (2024 layer/step records with host rows); its logits are `top1_logit: nan`, `top2_logit: -inf`, `top1_id: 0` from **step 6 onward -- 506 of 512 steps**. Those lines are written as bare `nan` (invalid JSON). Every one of those 2024 records reports `host_finite = 0`, which is `scan_host_expert_outputs`'s "no non-finite value found" (llama-layer-streaming.cpp:4626: `host_finite = bad.empty() ? 0 : 1`); the `host_finite == 1` abort path never ran, and the arm **exited 0**. So the corruption arises OUTSIDE the guarded region and the loud-failure contract does not cover it.

**Grader defect found and fixed (recorded because it changed the answer).** The first grading pass silently dropped the 506 non-finite `tg_logits` lines (`except Exception: continue` on invalid JSON) and reported "5 divergent of 6" instead of "511 of 512", i.e. it understated a collapse as a small disagreement. `tools/run-gplr82-ac1.py` now counts unparseable lines, maps them back to their step, and grades a missing or non-finite readout as a divergence that can never be a near-tie. The superseded verdict is kept at `build/llama-cuda/gplr82-ac1-verdict-graderbug.json`. A second field-semantics bug was caught by inspection: `host_finite` is **inverted** relative to a naive reading (0 = all host outputs finite), so both counts are now recorded separately.

**Scope, risks, and recommendation. Measured:** AC1's four criteria on four arms at 512 tokens; the A2d zero-host-row corruption; the 506-step non-finite span; the silent guard; the ledger/execution divergence via `ffn_dev`. **Derived but not proven:** the precise injection point in defect (1) -- the leading hypothesis is that the device leg keeps only `n_used - experts_host_n` rows while the host leg's fixed 2-wide partition is not reconciled with the gate's zero approvals, so the two partials no longer reconstruct the full top-k; a failing-first test asserting id/weight pairing and partial-sum completeness under a non-identity reorder would settle it. **The ORDERING/pairing hypothesis is FALSIFIED (same session, code read, no run).** (a) `w_full` is built by `ggml_get_rows(h_probs, sel_in)` (qwen35moe.cpp:461-462) and `hsel` is a memcpy of that SAME reordered vector's tail (llama-layer-streaming.cpp:5187-5188), so each leg's weights FOLLOW their ids; the split is a partition of positions and the sum is PERMUTATION-INVARIANT, so reordering `ids_per_token` cannot change any number. (b) A2d takes the `host_pos.empty() -> continue` path, i.e. it performs NO reorder at all, and is still wrong from step 1 -- so no ordering behaviour can explain the corruption. What the compaction IS load-bearing for is MEMBERSHIP: it moves the gate-approved rows into the graph's host tail, which is why removing it would change WHICH rows go host without changing any value. **The live cause is the SPLIT POINT.** `const int host_rows = ls->experts_host_n; const int dev_rows = n_expert_used - host_rows;` (qwen35moe.cpp:435,441) fix the split at GRAPH BUILD from the CONFIGURED width and never from the plan's disposition, so a closed gate with zero host rows still hands the tail to the host leg -- exactly the measured `ffn_dev = 2` (mixed, per `ls_classify_graph_backend`: 0=all host, 1=all device, 2=mixed) alongside `host_share = 0` in A2d. `ls_plan_split`'s own `device` descriptor is full-top_k width, so the plan's device descriptor and the graph's device leg disagree in width whenever fewer rows are host-bound than `--experts-host N`. That, not the ordering, is what a fix must address. **Not claimed:** the A2 tok/s figure (20.31 vs A0 16.12) is **not** a speedup -- the run is numerically corrupt, so its timing is meaningless. **Not run:** the 50-token rung; AC2's lifecycle checks beyond the ledger; AC5/AC6/AC7. **Recommendation:** keep experimenting, but treat the split as **defective, not slow** -- no optimisation of it can be measured until defect (1) is fixed, because a mechanism that is wrong with zero host rows cannot be benchmarked as a mechanism. Evidence: `build/llama-cuda/gplr82-ac1-verdict.json`, `gplr82-ac1-grade.log`, `gplr82-ac1-arm-{A0,A2,A2d,A0b}.{jsonl,log,txt}`, `verdicts/gplr82-ac1.done`.

**Workspace fact (new, 2026-10-10).** Freebuff agent edits land in a **git worktree** at `qwen38GZI/.freebuff/worktrees/b947db99-...` (branch `freebuff/in-this-project-...`, HEAD `991dd53`) which has **no `build/`**, the vendored `third_party/llama.cpp` incomplete, and a **different** `tools/semantic-qa/check-answer.py` (1,554 lines vs 1,618). The engine, model wiring, traces, anchors and the live research log are in the **main checkout** `/c/Development/AI/qwen38GZI` on branch `research/layer-residency` (HEAD `98773fe`). This AC1 session was authored and run there for that reason; anything run from the worktree would silently use a different checker.

## 2026-10-08 (ONE GPU hold, session gplr61-ac3, wrapper PASS): T-GPLR-61 SPEC-gap follow-up -- `--route-telemetry 1` now MEASURED (flag echoed; `check-route-parity` gate root-caused as structurally unsatisfiable for this arm shape; TG route parity CLEAN exactly-twice from `rowset_fetch`), session restructured so walls are comparable (warm-up + counterbalanced orders), AC2 FAIL reproduced from ac3's own artifacts.

**Why.** The audit of the ac2 closure named two SPEC gaps: the contract's
probe command specifies `--route-telemetry 1` (plan L4379) and the ac2 harness
omitted it, so routing had been ruled out by structural reasoning only; and
ac2's fixed A->B->C order left arm A's lone cold wall (157.8 s) incomparable.
This entry closes both. The gplr61-ac2 AC2 FAIL entry stays intact below -- ac3
corroborates it. No design refactors; ac2 evidence untouched (fresh session
name `gplr61-ac3`; `source_changed_during_run: False`).

**Session (MEASURED).** One `tools/with-gpu-lock.ps1` hold
(`-VerdictTag gplr61-ac3`), wrapper exit 0, `gpu_pre=gpu_post=clean`
(nvidia-smi), provenance source_digest `E505C29E1DEE04C4`, `llama.dll
C6855398617D24BC` (same binary as ac2 and the host-suite 78/78 record); harness
session wall 650.6 s (09:51:23 -> 10:02:18Z); 9/9 arms rc=0;
`--check-telemetry` rc 0 x9; anchor sha16 `d5bfd34a8a795e49` == ac2 ==
recorded. Post-release probe: 0 compute apps, 0 MiB used, 1% util. Harness
`tools/run-gplr61-ac3.py`, launcher `tools/gplr61-ac3-launch.ps1`.

**Structure (closes gap 2).** Cref-ref/Bref-ref replay the exact ac2 commands
(1 PP pass; ID anchors, untimed), then A-warm (untimed), then the timed arms,
all with `--pp-repeat 2 --route-telemetry 1` (2 PP passes + TG): p1 order
A,B,C; p2 order C,B,A (counterbalanced). Same structure for all three timed
arms, so their walls are mutually comparable within this session.

**ID comparison (MEASURED, from ac3's own traces;
`build/llama-cuda/gplr61-ac3-verdict-corrected.json`).** Refs byte-identical
to ac2 (`Cref==ac2C`, `Bref==ac2B`); `Ap1==Ap2`, `Bp1==Bp2`, `Cp1==Cp2`
(repeat determinism); `Ap1==ac2A`, `Bp1==ac2B`, `Cp1==ac2C` (repeat +
route flags ID-neutral); `Bp1==Ap1`; `A-warm==Ap1`; `Cp1 != Bp1` first
divergence index 1 (B=198, C=220) -- AC2 FAIL reproduced. Correction: the
original `gplr61-ac3-verdict.json` records `Bref_eq_ac2B`/`Cref_eq_ac2C` as
false because the harness compared `ids.get('Cref')`/`ids.get('Bref')` against
arm keys `Cref-ref`/`Bref-ref` (None-vs-list); harness key fixed, traces
unmodified, the corrected file carries the real comparisons.

**Gap 1 -- route telemetry, measured result (MEASURED).** `--route-telemetry
1` is on every p1/p2 and warm arm; run_start echoes `route_telemetry:1
source:explicit` on all 7 (refs 0/default by design). `tools/check-route-parity.py`
over the six p1/p2 traces (`build/gplr61-ac3-parity.json`) exits **2** -- zero
`route` events in every trace, NOT a pass. Root cause, measured not reasoned:
the sole `tel_route` call site is `llama-layer-streaming.cpp:6349`, inside
`llama_context::layer_major_pp`, which hard-requires `--pp-schedule
layer-major`; a repo-wide scan found **0 of 315 batch-major traces ever emitted
a route event, all 203 route-bearing traces are layer-major**, and the checker
itself exits 0 on the layer-major ab20 control traces (tool verified). The
T-GPLR-61 arms cannot be run layer-major: T-GPLR-73 (`main.cpp:5073`) refuses
every `--residency-tg*`/`--experts-host*` flag there, and layer-major caps
`--tg-tokens <= 3` without an active handoff (arms run 96). **Spec finding for
the Planner: the contract's step-2 gate (plan L4385, `check-route-parity
<trace>`) is structurally unsatisfiable for this arm shape.** Note for readers:
exit 2 here does NOT mean the run died before routing -- all 9 arms completed
(96 tokens, check-telemetry rc 0); it means batch-major never emits `route`
events by construction. Contract edit not made here (Planner-owned; plan file
at 183 B headroom).

**Routing evidence that exists in ac3's own artifacts (MEASURED).** TG routes
are in `rowset_fetch` (`union_ids`+`hash`, top_k 8, route_union 8, n_tokens 1):
3,840 fetches per arm. (a) **Cross-process TG route reproducibility**: B p1
tuple sequence == B p2, C p1 == C p2 (3,840/3,840 each; combined exactly-twice
3,840 distinct x2, 0 singletons, 0 over-counted) -- deterministic per arm across
independent processes. Labeled as a cross-process reproducibility check, NOT
the checker's pass-2 gate (that gate is PP-only; unreachable here per above).
(b) **First-divergence localization**: B vs C route tuples are identical for
the first 23 fetches; first divergence is TG round 1 (chunk 2), layer 23, a
single near-tie top-1 flip (B `19,47,62,...` vs C `39,47,62,...`, remaining
7 experts identical), with 2 layers flipped in round 1, 39/40 in round 2, all
40 from round 3 on (3,801/3,840 total). Ordering (MEASURED): route divergence
appears in the FIRST TG round, before the first divergent token id -- ids[0]
is identical (248046) despite the round-1 route flips, ids[1] is the first
divergent id (B=198, C=220), i.e. the route flip precedes and predicts the
token divergence rather than following it. Interpretation (labeled): the route
flip is the earliest observable symptom of the split's already-recorded
fp-order difference propagating through hidden states -- 7/8 experts unchanged
is a near-tie signature of ulp-level noise, not a routing-policy change; the
router mechanism itself (top-k code, top_k, union cardinality) is unchanged in
every fetch. What is NOT proven: PP-side route equality (batch-major emits
none), and a causal separation between "fp noise moved the near-tie" vs "split
altered routing logic" cannot be made from route tuples alone -- AC2's ID FAIL
already fixes the root cause at the split.

**Comparable timings (MEASURED walls, n=2 per arm, counterbalanced; walls only
-- no performance claim, AC2 forbids speed claims).** Excluded from timing:
Cref 29.8 s, Bref 29.3 s, A-warm 145.8 s (all rc 0). Timed walls (p1 order
A,B,C / p2 order C,B,A; every timed arm identical structure: 2 PP passes +
TG): **A 146.2 / 146.1 (mean 146.1); B 29.2 / 30.5 (mean 29.9); C 32.2 / 30.6
(mean 31.4)**. (a) Cold-start gap closed: ac3's arm A ran MORE work than ac2's
A (2 PP passes vs 1, phase_summary confirmed) yet finished faster (146.1 vs
ac2's lone 157.8) while A-warm/A-p1/A-p2 agree within 0.4 s across process
positions -- ac2's fixed-order first-run wall carried the cold premium.
(b) Order effect bounded: each arm's two runs (opposite positions) within
1.3-1.6 s. (c) B vs C: +1.5 s mean (5.0%), which is SMALLER than the
per-arm order spread (1.3-1.6 s) -- not separable from run-to-run noise at
n=2; no split-overhead number may be read from this. Phase detail (MEASURED,
phase_summary): the A-B wall gap is entirely TG-phase (A TG 123.7 s vs B TG
6.8 s; PP passes ~7.6 s + 1.4 s both arms) -- consistent with A having no
row-set TG path, expected by construction, not a new result.

## 2026-10-08 (ONE GPU hold, session gplr61-ac2, provenance PASS): T-GPLR-61 AC2 FAILED -- the per-expert split does not preserve byte-identical greedy IDs (first divergence at token index 1), while every instrumentation, inertness and falsifier check recorded exactly as designed. The task STOPS at AC2 per contract; no tuning was attempted.

**Task.** Mission step 3: T-GPLR-61 (per-expert TG host/device split,
`--experts-host` + `--experts-host-budget`), contract
`docs/plans/generic-prefill-layer-residency-plan.md` L4183+. Pre-edit reading,
the step-B engine edits (S3/S4), the build, the AC1 focused test and the host
suite were completed earlier; this entry records the step-C gap closure and
the AC2 identity session.

**Step C completed first (the probe must run the final binary).**

- **Non-finite abort.** `build_moe_ffn` gained a null-defaulted `experts_out`
  out-parameter that hands back the PRE-AGGREGATION per-row outputs;
  `build_moe_host_leg` stashes it per FFN graph (`graph_set::ff_host_experts`);
  the executor scans after part B (host buffer read directly, else
  `ggml_backend_tensor_get`) and aborts with a structured error naming layer,
  token, expert ID, row and column. Flag absent -> no stash, no read, no sync
  (AC6 inertness preserved by construction).
- **Telemetry.** `layer_compute` gained `host_finite` (-1 = not scanned / no
  host leg, 0 = scanned clean, 1 = caught -> recorded, then the run aborts);
  the optional-parts buffer went 512 -> 640.
- **MTP pre-load refusal (SDD failure contract).** `--mtp` + `--experts-host`
  refuses at bring-up (exit 2, `run_refused experts_host_with_mtp`):
  `graph_mtp` builds its routed FFN with plain `build_moe_ffn` (no host leg,
  no row-set state) and T-GPLR-61 proves identity only for the target
  forward, so draft-safety is unproven. Remove only with a draft-safety
  argument or an MTP identity run.
- **Validation (host-only).** Full build exit 0 (llama lib, tool, all ten
  layer-streaming test targets); `ctest -R test-layer-streaming-expert-split`
  PASS (AC1 unchanged); `tools/run-tests.ps1 -Suite host` **78/78 PASS**,
  revision digest `E505C29E1DEE04C4`, `llama.dll C6855398617D24BC`.

**AC2 session (fresh name `gplr61-ac2`, ONE `with-gpu-lock` hold, no recorded
evidence overwritten).** Probes: pre-run `0 MiB / 0% / no compute apps`;
post-run `0 MiB / no compute apps`. Wrapper verdict **PASS (0)**,
`gpu_pre=gpu_post=clean`, provenance gate **PASS** (digest
`E505C29E1DEE04C4`, `llama.dll C6855398617D24BC`, exe sha256-16
`a96cc12668baa0b3`). One fresh calibrated anchor: `achieved=512
items=N1,N2,N3,N4,K1`. Three arms, identical prompt/ubatch/tg-tokens except
the TG placement flags:

| arm | flags | rc | wall |
|---|---|---|---|
| A | `--no-handoff` (the recorded streaming shape) | 0 | 157.8 s (FIRST run, cold file cache) |
| B | `--no-handoff --residency-tg` (row-set, all rows device = paired control) | 0 | 28.5 s |
| C | B + `--experts-host 2 --experts-host-budget inf` (candidate) | 0 | 30.9 s |

Walls are NOT comparable across arms (A paid the cold start) and **no timing
claim is made**.

**AC2 = FAIL (blocking).** Greedy IDs: **A == B == the recorded
`gplr07-qa-512` control, byte-identical over all 96 tokens** -- so the paired
control is valid and the row-set path itself is bit-clean. **C diverges at
token index 1** (`B=198, C=220`); position 0 is identical. QA: A and B
`aggregate=PASS` with the negative control failing as designed (rc 1,
`aggregate=FAIL`); **C positive FAIL (rc 1)** -- the needle answer changed.
(C's negative control on an already-failing text exits 1 without printing
`aggregate=FAIL`; a consequence of the positive failure, not an independent
defect.) Per contract: **the task stops here -- no placement tuning, no gate
tuning, no rescue was attempted.**

**The fp-order vs kernel-domain distinction the contract requires.** The
session does not isolate the two mechanisms -- admission couples them -- but
it bounds them tightly:

- **Certain by construction: kernel-domain.** Host-admitted rows execute on
  the CPU backend (ggml CPU f32/BLAS) instead of the CUDA matmul kernels that
  computed them in arm B. Per-row outputs round differently from the first
  multiply; that is the mechanism the task exists to evaluate.
- **Also present by construction: fp-order.** The fixed-shape reorder changes
  the add-chain association (device rows first; host rows + fill in the
  suffix) relative to router order; f32 addition is non-associative, so
  ~1e-7-level perturbations exist even if every row computed bit-identically.
- **Ruled out: routing / weight / quantization change.** `ls_check_no_rewire`
  was OK at every host-leg build (no GGML_ABORT),
  `staged_bytes_avoided == experts_host x 1,769,472 B` on every record (plan
  and graph agree), the absent / never-executed guards stayed silent, and
  both row-set traces contain **zero** error or refusal events (8,181 events
  each). The split reorders the selected row set; it never changes which
  experts the router selects (SDD invariant; the graph consumes the same
  top-k set).
- **Not separable from these artifacts:** which mechanism crossed the argmax
  first at token index 1. Distinguishing them needs a diagnostic arm the
  contract does not authorize (reorder-with-all-rows-device, or per-row
  logit-gap telemetry). Escalated to the owner below.

**AC4 (instrumentation, PASS).** 3,840 row-set `layer_compute` records in C:
`experts_device + experts_host == experts_total` **3,840/3,840, 0
violations**; `staged_bytes_avoided` an exact multiple of 1,769,472 B on
every record, 0 unclamped mismatches; `host_finite=0` on every record (the
new scan ran and was clean); strict `--check-telemetry` rc 0 on all three
traces; in-flight copies completed and retired (arm C: `copy_submit` 40 ==
`copy_complete` 40); and `python tools/check-gplr82-ledger.py` on arm C's
trace returned **VERDICT PASS (ledger closed)** -- 7,680/30,720 = 0.2500
host, share histogram 0.25 x 3840, exit 0.

**AC5.** NOT a did-not-fire: the mechanism fired (see AC7).

**AC6 (inertness, PASS within session).** Arm B (flag absent):
`experts_host == 0` on all 3,840 records, gate fields `-1`,
`host_finite=-1`; ids byte-identical to arm A **and** to the recorded control.

**AC7 (the falsifier, recorded).** Host share **0.2500 exactly** (7,680 host /
30,720 total routed rows; `rows_gate_cpu=7,680` = exactly 2 of top_k 8 rows
per token per record; `rows_gate_gpu=23,040` = 6). With `HOST_ALWAYS` and
`--experts-host-budget inf` the gate adjudicated `gate_wait_us 171.0 >
gate_host_us 36.5` and opened on **every** one of the 3,840 records -- a
large host share is the input to the performance arithmetic, not a defect.
**No performance arm is proposed: AC2 failed and the contract forbids one.**

**AC3 (no hang).** All three arms rc 0; zero `cudaErrorLaunchTimeout` / TDR /
OOM / CUDA-error needles in any arm log.

**Harness attempts (recorded honestly).** Launch attempt 1 died before any
arm (a heredoc-mangled string literal in `tools/run-gplr61-ac2.py`), attempt
2 on a list/str unpack bug -- both released the mutex cleanly and ran NO
engine arm. Attempt 2 is also the first to pass the provenance gate (fix:
the exe path is now passed inside `-Command` so `Get-ProvenanceArtifacts`
can attest). The attempt-1 wrapper record is the `REFUSED provenance (exit
10)` verdict in `verdicts/gpu-command.*`; later attempts attest under
`verdicts/gplr61-ac2.*`. Evidence: `build/gplr61-ac2-launch-attempt{1,2}-failed.log`,
`build/gplr61-ac2-attempt2-verdict.json`.

**Artifacts (all outside version control).**
`build/llama-cuda/gplr61-ac2-*` (anchor, three arms x jsonl/txt/log,
verdict.json); `build/llama-cuda/verdicts/gplr61-ac2.{started,done,verdict.json}`;
harness `tools/run-gplr61-ac2.py`, `tools/gplr61-ac2-launch.ps1`,
`tools/gplr61-ac2-start.ps1`.

**Known gaps / residual risk.** (1) AC2 failed: the split changes generated
ids and broke the needle answer at the 512 anchor -- mechanism unisolated
(above), escalated. (2) Identity was measured at ONE shape (512 anchor,
N=2, budget inf, HOST_ALWAYS); no other N/budget/shape was run and none may
be run to "rescue" the gate. (3) MTP + `--experts-host` refuses by design
(unproven draft-safety). (4) The plan task's `Status:` line still reads
READY; flipping it is the Planner/owner's call. (5) The `host_finite=1`
abort path was never exercised at runtime -- host-only construction, tests
and review only.

## 2026-10-08 (T-GPLR-71 closure validation: recipe steps 1-3 re-run on today's tree, then ONE mutex hold as session ab23 = cross-session replication): the Oct-2 TG win reproduces at +25.2% paired, VRAM byte-identical, PP non-regressing -- and today's ABSOLUTE levels are ~26% lower for BOTH arms, matching the N5 machine-state finding

**Task.** Mission step 2: close T-GPLR-71 per its contract in
`docs/plans/generic-prefill-layer-residency-plan.md` (contract, recipe, SDD and this file
re-read before the first edit). The truthful starting state: T-GPLR-71 steps 1-5 were already
implemented and measured on 2026-10-02 (ab20/ab21/ab22, guard proof, consolidated owner
summary above), while the block's status header still read "READY ... **no code written yet**"
-- stale, and what mis-queued this task. Closure therefore = re-validate the recipe gates on
today's tree, replicate the headline A/B across a session, and reconcile the status. No
staging, no commit. Evidence kept under a NEW session name so ab20/21/22 stay byte-intact.

**Doc reconciliation (host-only).** Plan block: heading `READY -> MEASURED`, status line
rewritten to record steps 1-5 + ab20-ab22 (2026-10-02) and the host gates + ab23 replication
(2026-10-08), owner acceptance explicitly pending; recipe step 4's `-WaitCpuQuiet` removed --
the reworked `tools/with-gpu-lock.ps1` no longer has that parameter and the recorded command
FAILED at binding (`NamedParameterNotFound`) on first use today. Net plan size +28 B ->
**479,817 B, 183 B under the 480,000 B FAIL band; `tools/check-doc-sizes.py` PASS exit 0.**
The AGENTS.md `n_pad` gotcha (required by the contract) already exists (AGENTS.md L200-201).

**Host validation (recipe steps 1-3, no mutex):**
- **Build** (`CUDA_PATH` set, `cmake --build build/llama-cuda --config Release`): every
  task-relevant target produced (llama.dll `3E03583A4C1D5D67`, exe mtime 2026-10-08 00:43,
  test-layer-streaming-{tg-view,route-topk,host-mem,expert-split} all linked). **0 warnings**
  in the whole log. Exit 1 **solely** on the documented pre-existing `test-unicode`
  `unicode_regex_split` link failure -- classified environment/pre-existing in this file
  (2026-10-06 triage: "upstream target, in none of the host-suite regexes, zero unicode files
  modified"; also 2026-10-02 records). Not task-owned; not repaired here.
- **Host suite** `tools/run-tests.ps1 -Suite host`: **exit 0, 78/78** (Oct 2 was 77/77; the
  +1 is `test-layer-streaming-expert-split`, T-GPLR-61 step A already registered in the
  tree). Revision digest **`23DE7D243959F8C2`** (nested head `9b90025c4f9d`, 16 tracked / 3
  untracked changed), record `build/llama-cuda/suite-runs/20261007T225141Z-host/revision.json`.
- **Focused ctest** `-R "test-layer-streaming-(route-topk|host-mem)$"` **2/2 exit 0**, and the
  task's own test `-R test-layer-streaming-tg-view` **1/1 exit 0** (Test #54, 0.17 s).

**GPU, exactly one mutex hold.** External pre-probe `nvidia-smi`: 0 MiB used of 24,463 MiB,
zero compute apps; wrapper post-acquire probes `gpu_pre=clean gpu_post=clean`, mutex
`Global\AI-GPU-EXCLUSIVE-v1` released normally. Session **ab23** = the recipe's ab20 plan
byte-for-byte (`build/stock-levers/ab20.plan`, rep=2, route=1, tg=128, kvpad=256 on the
candidate) with the FROZEN directories: control `t-gplr71-control` (ab19 ctl38x), candidate
`t-gplr71-changed` -- identity confirmed in-arm: **exe `D5FD112C997C438F`, llama.dll
`14F4E04D98F9BD6E`**, exactly the plan's recorded hashes. Both arms **exit 0** (81 s / 72 s),
session report `ab23 VERDICT PASS`. The wrapper verdict is **UNCERTIFIED (exit 10)**,
`binary_state=not-supplied` -- the same pre-existing provenance gate that marked ab20/21/22;
evidence kept, only status changed; `-AllowUnprovenanced` deliberately NOT used and no second
hold spent on the word.

**Measured -- ab23 acceptance criteria, all re-derived from the artefacts on disk:**

| AC | ab23 result (2026-10-08) | vs Oct-2 (ab20/21/22) |
|---|---|---|
| AC1 view width | candidate TG `kv_view_built = kv_view_live = **50,176**`; PP lines 262,144 (phase-0 keeps the legacy pin, by design); control binary predates step-5 telemetry so carries no kv_view field | exact match; **ctx-flatness sweep NOT re-run** -- ab21 + dip n=2 stands |
| AC2 TG | 17.386 -> **21.765 tok/s = +25.2%** paired, target >=15% | +26.5% Oct-2 -- ratio reproduces |
| AC3 PP | `graph_builds` **0** on every PP line, both arms; paired 2,976.6 -> 2,960.9 = **-0.53%**; absolute floor 3,089.6x0.98=3,027.8 **missed by BOTH arms** (control itself 2,976.6) | see spread below |
| AC4 VRAM | peak dedicated **8,160,768,000 B byte-identical on both ab23 arms** = 7,782.71 MiB | and on **both ab20 arms** -- 4/4 exact |
| AC6 rebuilds | `view_rebuilds_total` 0 at tg=128 (guard's first firing is token 177, ab22); 0 `tg_view_rebuild` events; 0 in PP; decode-set `graph_builds` 1 (build-on-miss, same as Oct-2) | consistent |
| route parity | `check-route-parity.py` **clean both arms**: 480 events, 240 distinct, every tuple x2, 0 singletons, exit 0 | clean |

**The honest cross-session spread (required by AC3's contract).** Same binaries, different
day: control PP 3,089.0 -> **2,976.6 (-3.64%)**; control TG 23.407 -> **17.386 (-25.7%)**;
candidate TG 29.609 -> **21.765 (-26.5%)**. Session state dominates absolutes -- consistent
with the 2026-10-07 N5 entry, which measured today's host-expert floor +19% (12-thread) to
+149% (1-thread) slower under ~4.75 GB free RAM, and TG here runs 38/40 layers on the host
(52.2% of the wall). **The paired ratio is what replicates: +25.2% vs +26.5% (2 sessions),
and the AC3 paired delta -0.53% is 7x smaller than the -3.64% session drift of the control
against itself.** This is the interleaved-pairing lesson of the dip entry, observed again.

**Not re-run this session (Oct-2 evidence stands, labelled):** AC1's 4.00x ctx-flatness
sweep, AC5 semantic-QA smoke (5/5 both paths), the ab22 tg=256 guard arm, and nsys
kernel-level flatness. Nothing here re-measures them.

**Evidence.** `build/llama-cuda/logs/ab23-{ctl38x,g71x}-a1-u1024.{jsonl,log,mem.csv,cpu.csv}`;
`build/llama-cuda/verdicts/ab23-*.verdict.json` (+ `run-ab-session.done`);
`build/stock-levers/ab23-console.log`, `ab20.plan` (unchanged); host suite record
`build/llama-cuda/suite-runs/20261007T225141Z-host/revision.json`; build log of this session
in `/tmp/build-gplr71.log` (ephemeral; findings quoted above).

**Known gaps / residual risk.** (1) Provenance: both ab20 and ab23 are
`binary_state=not-supplied` -- timing rests on directory hashes, never on attestation of the
running process (handover S4). (2) n=1 per arm per session; cross-session replication
exists now, but within-session replicate spread is not re-measured today. (3) The plan file
sits 183 B under the doc-size cap -- the Planner-level split owed since the incident note
remains open, and this entry argues it more strongly: the next edit of any size may cross.
(4) The build's exit-1 on `test-unicode` means a bare `cmake --build` can never read exit 0;
it is pre-existing and out of T-GPLR-71's scope, but any recipe that says "Expected: exit 0"
without that caveat misleads.

## 2026-10-07 (HOST-ONLY: no GPU, no mutex, no engine run): T-GPLR-82 N5 -- forced same-window DRAM streaming slows the host expert chain +12-25% (two reproducible arms vs four same-day controls), but the engine never overlaps byte-carrying copies with host expert compute, so impact at the recorded operating point is ~nil; and the "~20x" engine-vs-bench gap is a unit mismatch (like-for-like ~1.6x)

**Task.** N5 of `docs/plans/expert-compute-staging-decoupling-work-packages.md` (contract, plan
and this file re-read before the first edit): run the CPU-vs-copy expert-bandwidth microbench
(`tools/expert-cpu-bench.cpp`), reconcile the "~20x" between the engine's ~682 us host layer leg
and the bench's 35.07 us/expert, and record the conclusion in the work-packages doc. Host-only
throughout: no GPU command ran, no mutex was taken. No staging, no commit -- everything left
uncommitted for owner review.

**What ran.** Same 72-cell grid as the 2026-10-02 revalidate (threads 1,2,4,6,8,12,16,24,32 x
experts 1,2,4,8 x tokens 1,8; `--pool 256 --id-stride 8 --iters 200 --warmup 20`, model
`C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`, layer 0; bench selftest
PASS exit 0, `ggml-cpu.dll` MD5 `80be38203b0081e3d2390aa0af7a35a2` identical to the
revalidation record, so the binary cannot have drifted):

- **4 controls**, nothing else competing: `build/expert-cpu-bench/n5-control{,2,3,4}-20261007.json`.
- **2 contention arms**: `n5-contend2-20261007.json`, `n5-contend3-20261007.json`, each with
  three `dd` readers looping the same layer-0 expert window (byte offsets 949-1385 MiB, all
  three `blk.0.ffn_*_exps` banks, 437 MiB/pass; 149-156 GB moved per arm; final pass 437 MiB in
  0.138-0.146 s = 3.1-3.3 GB/s per reader). Arm shape in `n5-contend-arm.sh`; reader logs
  `n5-contend{2,3}-dd{1,2,3}.log`.
- **Frequency probe**: `n5-freqprobe-20261007.*`, `n5-freq2-20261007.*` -- constant 2006 MHz
  during a bench run (clock scaling ruled out).

The three readers are a **host-DRAM surrogate for the H2D copy**: same pages, same read
traffic, but not the DMA engine.

**Measured -- contention delta vs the worst of the four same-day controls, per cell:**

| | arm 2 | arm 3 |
|---|---|---|
| median cell delta | **+11.9%** | **+14.5%** |
| p90 | +24.6% | +27.4% |
| worst cell | +46.7% | +51.6% |
| cells slower than every control | 69/72 | 72/72 |

Arms agree cell-by-cell (r = 0.68); **50/72 cells >+5% in both**. By shape (arm2/arm3): TG
shape 8exp/1tok -- 4t +14.1/+17.2%, 6t +25.4/+22.3%, 8t +18.4/+21.4%, 12t +16.3/+15.9%,
16t +15.4/+13.4%; **1 thread -0.0/+1.1%** (compute-bound, nothing to steal); PP shape 8exp/8tok
mostly +3-15% with a 12-thread outlier +45.4/+50.8%. Controls reproduce each other (control 4
vs mean of 1-3: +0.4% median across cells).

**Measured -- run-to-run drift (does not affect the deltas above).** Today's absolute floor is
worse than 2026-10-02 on the identical DLL: 8exp/1tok best cell 35.07 -> **41.70** us/expert
(+19%), 1-thread 58.13 -> **144.70** (+149%), 4t 38.59 -> 50.18 (+30%). Clock is constant
(probe above), machine had ~4.75 GB free RAM under resident load -- **machine state, not code
drift** (attributed, not proven). All contention deltas use same-day controls; **absolute
us/expert figures do not transfer across days.**

**Measured -- engine duty cross-check (why impact is ~nil today).** Re-verified from
`build/llama-cuda/logs/tg72-threads/tg72-thr16.jsonl`: PP staging = 40 copy submits totalling
**18,327 MB**, t_us 5,508,755..20,615,525; TG = 5,120 copy submits with **bytes = 0** (residency
hits), t_us 22,778,100..27,042,327 -- staging ends **2.2 s before** the first TG copy event, so
the phases never overlap. Host expert compute (the **681.0 us** leg, 4,864 `layer_compute`
events with `dev=0`) exists only in TG, where byte movement is zero; PP `layer_compute` is
`dev=-1` (240 events, device-side). `build/llama-cuda/gplr82/tim-host2-a/trace.jsonl` agrees:
all 40 copy_submits are phase 0, zero byte-carrying copies during TG. So at this operating
point the measured contention has nothing to act on: **mechanism present when forced
(measured), impact ~0 when the engine runs (measured).**

**Reconciliation -- the "~20x" is a unit mismatch (measured arithmetic).** 682 us / 35.07 us =
19.5 divides a **whole layer** (8 experts + attention + recurrent-state step + plumbing) by
**one expert**. Like-for-like, using the frame already in this file (2026-10-02 entries):
device base per layer 203.7 us + 8 x in-situ host expert 58.0 us (the +463.8 us/layer penalty
/ 8) = **667.7 us** predicted vs **674.6-681.0 us** measured (~1-2% residual); bench optimum
35.07 vs in-situ 58.0 = **1.65x** (this file records 1.59x against the then-current 36.5
figure), worth 6,527 us/token = 19.3% of the wall -- size of the prize, not a promised gain.
58.0 sits within 0.9% of the bench **1-thread** cell (58.13), and the thread lever is already
refuted (engine host layer thread-invariant, 1.8% spread over 1/4/12/16 threads), so the
residual ~1.65x is **parallelism the host dispatch path does not take plus ggml scheduler
overhead** (suspect, **unproven** -- no arm isolates it). **Honest per-expert overhead:
~1.6-1.7x, never 20x.**

**Recorded.** Work-packages doc updated: section 4 now marks the contention unknown answered,
and section 5 carries an `N5 result (measured 2026-10-07)` block with the full numbers,
cross-check and reconciliation.

**Known gaps / residual risk.** (1) The **first** contention arm's JSON was overwritten by the
replicate (filename reused before renaming); its in-session numbers (median +11.8%, p90 +25.4%)
are not file-backed and are excluded from every claim. (2) A reader-only baseline (120.5 GB in
15.13 s = 7.95 GB/s, no bench) exists in-session only, no file. (3) The surrogate is not DMA:
**+12-25% under real H2D overlap is expected, unproven** -- the authoritative test is the
worktree-only draft `docs/plans/qwen36-staging-contention-arm.md`, which needs
`expert-cpu-contend-probe.cu` that does not exist. (4) The ~0-impact finding covers only
recorded operating points (tg72, h2) where staging and host compute do not overlap; any config
that stages bytes *during* host compute (partial handoff, async-feed overlap) inherits the
+12-25% budget. (5) N5 does not answer N1's two open unknowns (scheduler acceptance,
activation-transfer cost); the engine-side split remains never-run.

## 2026-10-06 (host-only edit; GPU held for four timing arms, mutex `Global\AI-GPU-EXCLUSIVE-v1`): T-GPLR-82 audit close-out -- the `build_moe_ffn` clamp now has a RED/GREEN regression test, the two-session hunk map for a future commit is recorded, timing is measurable again on a clean GPU, and the 3847-node graph count is claimed with named sources

**Task.** Close the four limitations the last audit left open, without committing anything:
(1) the `build_moe_ffn` clamp fix had no automated regression test (the top gap); (2) `tools/run-tests.ps1`
carries hunks from two sessions with no map for a future commit boundary, and the
`tools/with-gpu-lock.ps1` diff is not from this work at all; (3) every GPU timing number was
unusable because `ChatGPT.exe` (PID 24036) held the device during the recorded runs, so the
QA results stand as correctness-only; (4) the `sched_reserve: graph nodes = 3847` observation
was corroborated in logs but never claimed in this file. All gates below are advisory
evidence, not release authorization. No commit was made: 0 staged, outer HEAD `98773fe`.

**1 -- Top gap closed: a regression test proven to FAIL on the defect and PASS on the fix.**

New CTest test `layer-streaming-host2-preflight` (Test #60) in
`third_party/llama.cpp/tools/layer-streaming/CMakeLists.txt` (+39 lines, inserted after the
`layer-slot-lifecycle` `TIMEOUT 900` block). It runs the exact boundary runner shape
(`--model-env Q38LS_MODEL_PATH --residency moe-expert-bank --pp-schedule batch-major
--residency-tg --experts-host 2 --residency-tg-retain 2 --layer-slots 2 --kv-graph-pad 256
--kv-cache q8 --ctx-size 262144 --ubatch 1024 --tg-tokens 16 --copy-source mmap-registered`)
plus `--graph-preflight-only`, which builds the persistent row-set graphs and exits before
PP/TG, so the out-of-bounds view assert surfaces in ~17 s instead of needing a full
workload. Properties: `TIMEOUT 300`, `PASS_REGULAR_EXPRESSION "persistent row-set graphs
BUILT OK"`, `FAIL_REGULAR_EXPRESSION "GGML_ASSERT"`. The two-pattern contract is documented
in the comment above the test from measured ctest semantics (2026-10-06): FAIL wins when it
matches; PASS is required even at exit 0, so a vacuous run that never builds the graphs
cannot pass; a silent non-zero exit fails both ways.

Red/green cycle, exact outputs (logs persisted beside the other evidence):

| Step | What ran | Result |
|---|---|---|
| green 1 | `ctest -C Release -R layer-streaming-host2-preflight` on the fixed binary | **PASS** -- `1/1 Test #60: layer-streaming-host2-preflight ... Passed 16.13 sec`, `100% tests passed`; verdict `gplr82-regress-green` PASS exit 0 (wrapper 22.2 s), gpu_pre/gpu_post clean |
| RED | clamp mutated at `src/llama-graph.cpp:2340` to `const int64_t n_rows = hparams.n_expert_used;` (byte-safe Python replace, CRLF preserved 3901/3901), rebuild exit 0 | **FAIL** -- `1/1 Test #60 ...Exit code 0xc0000409***Exception: 17.23 sec` with verbatim `ggml.c:1787: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed`, `0% tests passed, 1 tests failed out of 1`; ctest exit **8**; verdict `gplr82-regress-red` FAIL (8) (20.7 s), gpu clean |
| green 2 | restore from the pre-mutation snapshot, sha256 `4f8bfa46316d1af0506050c9faabad033204444416627fc6dc2d211832aa7c8a` matches, 0 `RED-RUN` markers, rebuild exit 0 | **PASS** -- `1/1 Test #60 ... Passed 8.13 sec`; verdict `gplr82-regress-green2` PASS exit 0 |

The red run is the point: the test discriminates the defect instead of merely observing the
fixed binary. Evidence: `build/llama-cuda/gplr82/regress-green.log`, `regress-red.log`,
`regress-green2.log` and `build/llama-cuda/verdicts/gplr82-regress-green.verdict.json`,
`gplr82-regress-red.verdict.json`, `gplr82-regress-green2.verdict.json`.

Suite wiring: `tools/run-tests.ps1` line 286, boundary `Include` is now
`qwen38-layer-boundary|layer-slot-lifecycle|layer-streaming-host2-preflight`. End-to-end
`-Suite boundary` with no flags (record
`build/llama-cuda/suite-runs/20261006T100804Z-boundary/revision.json`): **3/3 pass** -- #58
8.89 s, #59 19.71 s, #60 8.27 s, `100% tests passed, 0 tests failed out of 3`. Suite exit
is **10** again, the provenance gate only (`Verdict REWRITTEN: UNCERTIFIED (10)` -- a bare
`ctest -Command` yields `binary_state=not-supplied`, so there is no exe path to attest).
That refusal is pre-existing structure and was deliberately NOT weakened: no
`-AllowUnprovenanced` was added anywhere. Log: `build/llama-cuda/gplr82/boundary-suite-20261006.log`.
Guards run: cmake reconfigure exit 0, `ctest -N` lists Test #60, `run-tests.ps1` PowerShell
PARSE OK, both edited files LF with 0 tabs, GPU probes clean on every GPU run.

**2 -- Commit-boundary hunk map: which hunks are which session's.**

`git diff -U0` on `tools/run-tests.ps1` shows exactly **6 hunks**:

| Hunk | Content | Owner |
|---|---|---|
| `@@ -110,0 +111,9 @@` | `[switch] $AllowBusy` param block (default OFF) | 2026-10-05 T-GPLR-82 ladder session |
| `@@ -277 +286 @@` | boundary `Include` gains `layer-streaming-host2-preflight` | this session (2026-10-06) |
| `@@ -320 +329 @@` | host-suite `Include` gains `tg-view|expert-split` | **NOT ours -- pre-existing foreign hunk; do not stage with ours** |
| `@@ -956,2 +965,7 @@` | Direct suite: array splat + `-AllowBusy` forward | 2026-10-05 ladder session |
| `@@ -1040,2 +1054,7 @@` | QaSmoke suite: same splat change | 2026-10-05 ladder session |
| `@@ -1087,2 +1106,7 @@` | Gpu/ctest suite: same splat change | 2026-10-05 ladder session |

So the earlier shorthand "4 mine + 1 pre-existing" was incomplete: the AllowBusy work is 4
hunks (param + three splat sites), my test wiring is 1 more, and 1 hunk belongs to another
line of work. `tools/with-gpu-lock.ps1` has exactly **one** hunk, `@@ -149 +149,11 @@`, the
PATHEXT / `$LASTEXITCODE` guard dated "measured 2026-09-30" -- **not mine**, and it must not
be swept into a commit of this work. Nested tree: `third_party/llama.cpp/tools/layer-streaming/CMakeLists.txt`
`+39` lines is entirely the new test (mine); `src/llama-graph.cpp` is 174 insertions / 126
deletions = pre-existing S2 `moe_weights_in` plumbing hunks PLUS this session's single-line
clamp at line 2340 -- my edit introduced no hunk beyond that line. Nothing staged or
committed.

**3 -- Timing usable again: four clean-GPU arms, same binary, n=2 per arm.**

The old numbers were unusable: the recorded full arms read `h0 = 51.2 s / h2 = 51.9 s` with
`gpu_pre = gpu_post = 24036 ChatGPT.exe` in both verdicts. The owner authorized unrestricted
GPU use, and every probe today was clean, so both full arms were re-run twice each through
`with-gpu-lock.ps1` -- no `-AllowBusy` needed, no provenance waiver (0 waiver lines for
these tags), mutex held throughout. Every arm: `gpu_pre=clean gpu_post=clean`, exit 0,
identical launch digest `23DE7D243959F8C2` / `llama.dll 1A921B6109710233` (nested head
`9b90025c4f9d`), `check-telemetry: OK (1705 events, 2 slots)`, peak VRAM 9279.56 MiB. The
commands are byte-identical to the old arms except `--experts-host 2|0` and isolated
evidence paths.

| Arm | PP tok/s | PP wall | TG tok/s | TG wall | process wall |
|---|---|---|---|---|---|
| h2-a (`--experts-host 2`) | 71.327 | 7178.2 ms | 13.065 | 1224.6 ms | 27.4 s |
| h2-b | 72.187 | 7092.7 ms | 13.477 | 1187.2 ms | 27.1 s |
| h0-a (`--experts-host 0`) | 70.055 | 7308.6 ms | 15.012 | 1065.8 ms | 26.1 s |
| h0-b | 77.142 | 6637.1 ms | 14.630 | 1093.6 ms | 26.1 s |

Timing components stay distinct as the charter requires (example h2-a: PP transfer
12125.1 ms, wait 885.4 ms, compute 1233.8 ms against PP wall 7178.2 ms -- transfer is the
CUMULATIVE copy time across concurrent streams, legitimately larger than wall, and is never
added to it).

**Measured:** (a) the ChatGPT.exe confound inflated process wall roughly 2x -- 51.2/51.9 s
became 26.1/27.x s on the same binary and commands, so the old timing is superseded, not
reused; (b) wall and TG are repeatable within arm (h2 wall spread 0.3 s; h0 wall identical
twice) and the TG arms do not overlap (max h2 13.477 < min h0 14.630 -- h0 faster on this
16-token TG shape, n=2); (c) **PP is NOT resolvable at n=2**: h0's own spread
(70.055..77.142, 9.7%) exceeds the between-arm gap (means 73.6 vs 71.8, 2.6%), so no PP
arm claim is made. Evidence: `build/llama-cuda/gplr82/tim-host2-a/run.log`,
`tim-host2-b/run.log`, `tim-host0-a/run.log`, `tim-host0-b/run.log` and verdicts
`gplr82-tim-h2-a`, `gplr82-tim-h2-b`, `gplr82-tim-h0-a`, `gplr82-tim-h0-b` in
`build/llama-cuda/verdicts/`. The QA results remain correctness-only as disclosed
yesterday; no throughput number is adopted from them.

**4 -- The 3847-node count is now claimed, with named sources.**

`sched_reserve: graph nodes  = 3847` appears **11 times across 8 persisted logs**:
`build/llama-cuda/gplr07-qa-512.log` (1), `build/llama-cuda/gplr07-qa-50.log` (1) -- both
2026-10-05 QA-ladder rungs, recorded after the clamp fix; `build/llama-cuda/gplr82/boundary-suite-20261005.log`
(2, the 2-test suite); `build/llama-cuda/gplr82/boundary-suite-20261006.log` (3, one per
test of today's 3/3 suite); and exactly 1 in each of the four timing logs above. It exists
only as a process-log line -- no `sched_reserve` event is emitted in `trace.jsonl`, which is
why the earlier evidence pass could not find it there. Claim: **the persistent row-set graph
build for this model and shape constructs 3847 nodes**, measured on the clamp-fixed binary.

**Glossary (restated for this entry).** PP = prompt processing/prefill, reading the prompt;
TG = token generation, producing new tokens; tok/s = tokens per second; wall = elapsed time
of a phase or of the whole process; transfer/wait/compute = the three telemetry time
components, kept separate on purpose (bytes copied, waiting for a slot/layer, kernel time);
arm = one configured run in an A/B; verdict = the JSON result `with-gpu-lock.ps1` writes
beside every run (status, exit code, GPU probes, provenance); provenance gate = the check
that a run can name its source tree and binary, refusing with exit 10 (UNCERTIFIED, not a
pass) when it cannot; PASS/FAIL_REGULAR_EXPRESSION = CTest properties that decide a test's
verdict from its output text; red run = deliberately re-introducing the defect to prove the
test fails; digest = recorded identity hash of source/binary at launch; slots = the bounded
VRAM windows that hold one immutable weight package each; `--experts-host 2` = two expert
banks placed on the CPU host (the configuration that exposed the clamp defect); graph nodes
= nodes in the built prefill graph, printed once by `sched_reserve`; wrapper = the
`with-gpu-lock.ps1` script that owns the GPU mutex and writes verdicts.

**Pros / cons.** **Measured wins:** the regression gap is closed with a test proven in both
directions; a future commit can be staged hunk-by-hunk from the map; timing measurement is
restored (clean probes, reproducible walls); the 3847 claim now has 11 on-disk citations.
**Costs/risks:** the new test adds ~8-17 s to the boundary suite and, like its siblings,
fails loudly when `Q38LS_MODEL_PATH` is unset (intended); n=2 timing cannot support a PP
difference claim, and the TG result is a 16-token shape observation, not a benchmark
(measured, single shape); `ChatGPT.exe` will confound any future run that skips the
pre-probe, so timing must keep going through `with-gpu-lock.ps1`.
**Still open (unchanged):** boundary suite exit 10 (provenance structure, deliberately not
weakened); 4k/24k/50k/100k/200k ladder rungs remain owner-invoked manual runs; the two
pre-existing host-test teardown crashes (`test-layer-streaming-layout`,
`test-layer-streaming-tg-view`, 0xc0000409) are out of scope.

**Recommendation.** Adopt the regression test and the hunk map as they are; treat the timing
table as a clean baseline to extend with more repetitions before asserting any PP/TG
difference; keep all four items with the owner for final acceptance -- every gate here is
soft. No commit was made, per instruction.

## 2026-10-05 (GPU held; owner authorised unrestricted GPU use): T-GPLR-82 -- semantic-QA ladder executed for the host-leg fix

**Task.** Mission step 4 of 4: run the semantic-QA ladder for this branch after the
`build_moe_ffn` out-of-bounds fix, and record pass/fail per step with actual exit codes.
Steps 1-3 of the same mission (narrow the assert, fix the view, prove a closed ledger) were
completed earlier today and are recorded in the entry below.

**Tooling change required to run the ladder at all.** `tools/run-tests.ps1` had no
`-AllowBusy` surface, and it always drives `with-gpu-lock.ps1`, which REFUSES with exit 125
whenever *any* compute process is visible. A desktop app's GPU rendering context (PID 24036,
`ChatGPT.exe`) holds the device and never releases it, so **every GPU suite was unrunnable**.
Added a **default-off** `[switch] $AllowBusy` forwarded to the wrapper.

- First attempt used a **hashtable** splat and failed measurably:
  `Cannot process argument transformation on parameter 'AllowBusy'. Cannot convert value
  "System.String" to type "...SwitchParameter"`. Cause: splatting to a *native* command
  renders `-AllowBusy True` on a text command line, and the child's `-File` binding receives
  the string. Fixed by **array** splatting so the switch travels as a bare token.
- **Default behaviour verified unchanged**: `-Suite qa-smoke` with NO flag still REFUSES with
  `Verdict: FAIL (125) ... REFUSING to run: GPU busy`. Existing callers are unaffected.

**Ladder results (actual exit codes, captured directly):**

| Step | Command | Exit | Result |
|---|---|---|---|
| host-only suite | `-Suite host` | **8** | 5/7 pass. FAIL `test-layer-streaming-layout`, `test-layer-streaming-tg-view` (both `0xc0000409`) -- **KNOWN-FAIL, pre-existing, out of scope** |
| telemetry suite | `-Suite telemetry` | **0** | **PASS** -- selftest OK; positive `--validate-telemetry` OK (26003 events, 4 slots, 40 layers); negative correctly rejected (exit 2) |
| QA smoke 512 rung | `-Suite qa-smoke -AllowBusy` | **0** | **PASS** in 192 s -- `check-telemetry: OK (30547 events, 3 slots)`; N1 `QX-7741`, N2 `TR-2918`, N3 `MK-5501`, N4 `LD-8834`, K1 `Paris` all PASS; **negative control FAILs as designed**; record `20261005T162341Z-qa-smoke` |
| QA smoke 50 rung | `-Suite qa-smoke -AllowBusy -SmokeLengths 50` | **0** | **PASS** in 198 s -- `target=50 achieved=50 items=N1`; `check-telemetry: OK (30547 events, 3 slots)`; needle `mode=needle aggregate=PASS`; **negative control FAILs as designed**; record `20261005T162718Z-qa-smoke` |
| boundary (env unset) | `-Suite boundary -AllowBusy` | 10 | **harness error, not a test failure**: both tests failed identically with `missing or empty model environment variable: Q38LS_MODEL_PATH`. My invocation omitted the env var |
| boundary (env set) | `-Suite boundary -AllowBusy` | **10** | **tests 100% (2/2 pass)**; suite exit 10 is the **provenance gate**, not a test result: `no binary identity was captured (binary_state=not-supplied)` because the wrapper's `-Command` is a bare `ctest` with no exe path for `Get-ProvenanceArtifacts` to attest |

**Recorded revision for every suite run:** engine digest `33D97FC814749032`, nested head
`9b90025c4f9d`, `llama.dll 8ABB934026189E8E`.

**New vs pre-existing failures:**
- **Pre-existing / known-fail (not re-investigated, as instructed):**
  `test-layer-streaming-layout` and `test-layer-streaming-tg-view`, `0xc0000409` teardown
  crashes that print their full success line before dying. 5/7 still pass.
- **Pre-existing structural, not a code regression:** `boundary` suite exit 10 is the
  provenance gate refusing an unattestable `ctest` command. `run-tests.ps1` has never passed
  `-AllowUnprovenanced`, and my diff does not touch `-Command` content -- the same
  `$ctestCommand` string is passed before and after, so the refusal cannot be attributed to
  this change. **I did not add an escape hatch for it**: weakening a provenance gate to turn a
  suite green is exactly the wrong fix.
- **Not a failure:** the first `boundary` run failed only because `Q38LS_MODEL_PATH` was
  unset in my shell. With it set, both tests pass.
- **No new failures introduced** by the `build_moe_ffn` fix or by the `-AllowBusy` change.

**Not run, by contract:** the 4k/24k/50k/100k/200k ladder rungs are **owner-invoked manual
runs**, never part of `-Suite qa-smoke` (see `tools/run-tests.ps1` suite docs and T-GPLR-07).
They were therefore not executed and are NOT claimed.

**Confound, disclosed rather than hidden:** every GPU run in this ladder carried
`gpu_pre=gpu_post=24036 ChatGPT.exe` and the wrapper printed its own
`WARNING: ... a non-conforming compute process may have run alongside this arm; timing
suspect.` The owner authorised unrestricted GPU use. This makes **timing suspect** on these
runs; the QA verdicts are **correctness** results (telemetry integrity + programmatic answer
grading + must-fail negative control) and do not depend on timing. No throughput number from
this ladder should be quoted.

## 2026-10-05 (host-only for the edit; GPU held for the two proof runs, mutex `Global\AI-GPU-EXCLUSIVE-v1`): T-GPLR-82 -- `--graph-preflight-only` added to the layer-streaming runner, and the "minutes instead of seconds" premise is WRONG

**Task.** Owner-directed 2026-10-05: "add a preflight dry-run mode to the layer-streaming runner
that builds the persistent row-set graphs without a model, so graph-construction asserts surface in
seconds instead of minutes."

**What shipped.** `--graph-preflight-only` in `tools/layer-streaming/main.cpp` (help `:174`,
declaration `:3300`, parse `:3577`, early exit `:5621`). It runs the normal
`llama_layer_streaming_prebuild`, prints the graph footprint, then **returns 0 before any PP or TG
execution** -- placed after the footprint/`--graph-budget-mib` block (so a budget overrun still
reports itself) and before the mmap-registered page-locking and the timed PP. `exit_status` is
honoured, so a budget failure stays a failure rather than being reported as a clean prebuild.

**TWO CORRECTIONS TO THE REQUEST'S PREMISE, both measured:**

1. **It cannot skip the model, and the flag says so.** `llama_layer_streaming_prebuild(ctx, batch,
   tg_tokens, st)` needs a real `llama_context` built from real tensors -- there is no shape-only
   entry point. So this is a *dry-run*, not a model-free build. The help text states this outright
   ("The model IS still loaded ... only EXECUTION is skipped, not the load") so the next reader does
   not go looking for a shortcut that is not there.
2. **The assert was never slow.** It already surfaced ~18 s after launch, because it fires during
   the *prebuild* -- before execution ever starts. Measured wall times, this machine, same flags:

| | full arm | `--graph-preflight-only` |
|---|---|---|
| crashing case (`--experts-host 2`) | ~18 s | **21 s** (no faster) |
| passing case (`--experts-host 0`) | 23.7 s | **15 s** |

So the speedup is ~9 s and only on the PASSING path; on the crashing path -- the case the request
cares about -- there is **no** speedup, because model load (~6.5 s) dominates and the assert is
already early. **Do not sell this as a speedup.**

**What it IS good for** (all measured): it turns "do the row-set graphs build?" into a single
deterministic flag with exit 0/1, independent of prompt content, token count, sampling and the
ledger checker. It avoids the mmap page-locking and the timed PP, so a graph defect cannot be
masked or prolonged by execution, and it is safe to re-run in a tight edit/verify loop.

**Proof, both directions (same new binary, `digest 209BDF0BDC728818`, `llama.dll 6CB0A5F2F61A6A18`):**

- `--experts-host 2 --graph-preflight-only` -> reproduces the defect verbatim,
  `ggml.c:1787: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <=
  ggml_nbytes(view_src)) failed` during "layer-major preflight: building persistent graphs";
  verdict FAIL, exit `-1073740791`, 21 s wall.
- `--experts-host 0 --graph-preflight-only` -> `graph preflight: persistent row-set graphs BUILT OK
  (2 set(s), 2 build_generation, prep 18.5 ms)` plus the footprint line (`sets=2 total_graphs=84
  io_reserved=201326592 B io_used=46711366720 B host_input=1140850688 B`); verdict PASS, exit 0,
  15 s wall.

Both traces carry **0 `layer_compute` records** (47 and 46 records, ending at `run_start`), which
confirms the mode's own printed warning that the demand-ledger checker is not applicable here (it
would exit 2). That is stated in the mode's output rather than left for the next reader to trip on.

**Guard on the freeze.** The rebuild went to `build/llama-cuda/bin/Release`; the N2 treatment
binary in `build/llama-cuda/bin/unscoped-bin/s2-hostleg` was re-hashed before the build, after the
build, and after both runs -- `sha256` prefix `5a570b6dde40d659` **unchanged throughout**, so the
attribution evidence from the previous entry still points at the same executable.

**Build.** `cmake --build build/llama-cuda --target llama-layer-streaming --config Release`, exit
0. The four `C4477` `%d`/`size_t` warnings are PRE-EXISTING, in
`src/llama-layer-streaming.cpp:5230/5247/5341/5347` -- untouched by this change.

**Regression check (ran after the change).** `ctest -C Release -R`
`test-layer-streaming-(layout|model-inspect|descriptors|tel-mark|route-topk|tg-view|expert-split)$`
-> **5 passed, 2 failed**, `ctest_exit=8`: `test-layer-streaming-layout` and
`test-layer-streaming-tg-view`, both `Exit code 0xc0000409`. Binaries were checked FRESH first
(`main.cpp` 17:30:07 < `llama.dll` 17:31:00 < `llama-layer-streaming.exe` 17:31:11), so this is
not the stale-binary trap.

**Those two failures are PRE-EXISTING and NOT caused by this change**, established rather than
assumed, because the fastfail is the same family as the row-set assert and looked alarming:

1. `graph_preflight_only` / `--graph-preflight-only` appears **nowhere** in `third_party/llama.cpp/src/`
   -- the edit is confined to the runner's `tools/layer-streaming/main.cpp`, which is neither part of
   the `llama` library nor of either test target.
2. Both tests print their full success line (`test-layer-streaming-layout: all layout checks
   passed`) and **then** die, so this is a crash in TEARDOWN/static destruction, not a failed
   assertion. No check regressed.
3. A/B on the DLL: the same test exes were run in a scratch directory against the **frozen
   11:16 `llama.dll`** (`892b24ecbfe1e956`) and the **freshly built** one (`6cb0a5f2f61a6a18`).
   Both crash identically, so the rebuild did not introduce it either.

**Left open deliberately:** which teardown path asserts. Not diagnosed here; it predates this work
and is out of scope for a flag that touches no library code. Recorded so the next reader does not
re-derive it.

**Observation, not fixed:** invoking `with-gpu-lock.ps1` DIRECTLY on a fastfail run propagates
exit **127** rather than the runner's `-1073740791`; the verdict JSON carries the true code and the
`run-gplr82-hostarm.ps1` launcher reads it correctly. Read the verdict, not the wrapper's shell
exit, when driving the wrapper by hand.

## 2026-10-05 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`): T-GPLR-82 N1 attribution -- the GGML_ASSERT is CAUSED BY THE HOST LEG BEING ON. It is NOT pre-existing.

**Task.** Owner-directed 2026-10-05: "build a side-by-side control binary with the host expert leg
disabled so the GGML_ASSERT crash can be attributed to step 2 versus pre-existing."

**No second binary was built, and none is needed.** `--experts-host` is a **RUNTIME** gate, not a
compile-time one: `tools/layer-streaming/main.cpp` documents it as "Default 0 = off, which builds no
host leg at all", and `experts_host` defaults to `0`. So the FROZEN step-2 binary can be **both** arms
and `--experts-host` becomes the single controlled variable. That is a *stronger* control than a
rebuilt binary (no second build to drift, no rebuild risk, byte-identical executable), and it answers
the same question. The control binary was therefore not built -- a deliberate scope decision, not an
omission. `tools/run-gplr82-hostarm.ps1` gained an explicit `-AllowHostOff` switch for this; the
existing N1 guard still refuses `-ExpertsHost 0` without it (verified: it throws).

**The A/B.** Both arms: same frozen exe `sha256 5A570B6DDE40D659`, same `source_digest
21ECF0325891CC00`, same model / prompt / `--tg-tokens 16` / `--residency-tg-retain 2` / `--layer-slots
2` / `--kv-cache q8` / `--ctx-size 262144` / `--ubatch 1024` / `--copy-source mmap-registered`. The
ONLY difference is `--experts-host 2` vs `--experts-host 0`.

| | treatment `--experts-host 2` | **control `--experts-host 0`** |
|---|---|---|
| runner exit | `-1073740791` (`0xC0000409`) | **0** |
| verdict | FAIL, 17.9 s | **PASS, 23.7 s** |
| row-set `layer_compute` records | 0 | **640** |
| `experts_device + experts_host == experts_total` | n/a (no records) | **640/640, 0 violations** |
| host share (min/max) | unmeasured | **0.0 / 0.0** (host=0, total=5120) |
| checker | exit 2, NOTHING TO EVALUATE | **exit 0, PASS (ledger closed), mechanism DORMANT** |

**Conclusion (measured, single-variable):** the assert fires **only when the host expert leg is
enabled**. The same executable, with the leg off, loads the model, builds all 40 layer packages,
builds the persistent row-set graphs, runs PP and 16 TG tokens, and closes the ledger cleanly. So
this is **a step-2 defect, not a pre-existing row-set/row-cache defect** -- the exact question the
attribution was for. The likely locus is the host sub-op's view construction in
`qwen35moe.cpp` (`build_moe_host_leg`, `:892-949`) or the `moe_weights_in` view it binds; the assert
is a ggml view whose `data_size + view_offs` overruns its source. **Not yet narrowed to a line** --
that is the next task.

**Second, unplanned result: the ledger checker is now validated on REAL telemetry.** Until this arm,
`tools/check-gplr82-ledger.py` had only been proven against synthetic fixtures. On the control's 640
genuine `layer_compute` records it reports PASS / DORMANT with the identity holding on every record,
and it still reports exit 2 (NOT a pass) on the treatment trace. Its two important exits are now
confirmed against real runs in both directions.

**GPU coordination.** Pre-run probe: **no compute processes**. During the run the only compute
process was our own frozen `s2-hostleg\llama-layer-streaming.exe` (pid 22804, confirmed by path).
Post-run probe: **no compute processes**. No foreign process appeared, so no shared-GPU poisoning
confound applies.

**Standing gap (unchanged, now with evidence):** `run_start` does **not** carry an `experts_host`
field -- confirmed false on the control trace too, where 640 downstream records prove the leg was
requested-and-off. So a trace alone cannot show the requested value without reaching a
`layer_compute` record.

**Cost note.** The control ran 23.7 s of GPU work for a complete answer. The equivalent rebuilt
control binary would have cost a full CUDA rebuild of the vendored tree to prove something this run
proved exactly.

## 2026-10-05 (GPU held, mutex `Global\AI-GPU-EXCLUSIVE-v1`): T-GPLR-82 N1 -- the forced-on host-leg arm was finally FIRED, and it ABORTED. `experts_host` share is UNMEASURED, not zero.

**Task.** T-GPLR-82, **N1 pass 2 of 2**, owner-directed 2026-10-05. Contract
`docs/plans/expert-compute-staging-decoupling.md` (plan wins on any difference). One narrow
question: **did the host expert sub-op fire at all?** If `experts_host == 0` across an arm with
the host leg forced on, that is a valid answer ("mechanism did not fire") and placement / budget /
K must NOT be tuned to rescue it. An abort must be recorded verbatim and the pass stops.

**Headline: the question is UNANSWERED.** The arm aborted before a single row-set record existed,
so the share was never observed -- it is *not* `experts_host == 0`, and it must not be reported
as a zero.

**The arm had never actually run before this pass.** Three defects in
`tools/run-gplr82-hostarm.ps1` meant the previous two attempts died before the runner was ever
executed. Each was found by inspecting artifacts, not by inference:

1. **`Start-Process -ArgumentList` does not re-quote its elements.** The multi-token `-Command`
   string arrived at the child as `$Command = '&'` with every runner argument spilled into unbound
   positionals (proved with a parameter-dump probe: `RECEIVED[&]`, 8 stray positionals). The
   wrapper then died on a parse error *before taking the mutex* -- no `.started` marker, no trace.
   Symptom that would have fooled a reader: a healthy `[run ] wrapper pid N` line. Fixed by
   quote-wrapping the value with backslash-escaped inner quotes; verified to arrive intact.
2. **The wrapper runs `cmd.exe /d /s /c $Command` (`tools/with-gpu-lock.ps1:148`), so `$runnerCmd`
   must be CMD syntax.** It was built with PowerShell's `&` call operator, so cmd refused it
   verbatim: **`& was unexpected at this time.`** Runner exit 1 in 3.6 s. Fixed by dropping the
   leading `&`.
3. **`-WindowStyle Hidden` discarded the runner's stderr into an unreadable window**, leaving the
   verdict's `log` field empty. The runner's stderr is the ONLY place an abort reason appears, so
   a dying arm was undiagnosable -- exactly what happened. Fixed with `-NoNewWindow` +
   `-RedirectStandardOutput/-RedirectStandardError` into the arm's own output dir.

**Two further launcher reporting defects, found on the attribution pass and now FIXED.** Both
affected how BOTH arms above were *displayed*; every number quoted in this entry was read from
the verdict JSON and direct `nvidia-smi` probes, which is why they were correct even while the
launcher's own post-run block was not:

4. `[post] GPU compute apps:` rendered as `System.Object[]` at run time although the identical
   expression returns `System.String` in isolation. Now flattened through
   `ForEach-Object { [string]$_ }`; re-verified to return `System.String`.
5. **Worse, and silent: the post-run verdict block read property paths that do not exist.**
   `Read-ArmVerdict` returns the RAW parsed verdict JSON, in which the verdict word is `status`
   and the GPU probes live under `fields`. The launcher read `.verdict` / `.gpu_pre_run` /
   `.gpu_post_run`, so `[post] mutex verdict`, `[post] gpu_pre_run` and `[post] gpu_post_run`
   printed **empty on every arm** -- a report that looks like it reported something and did
   not. Now reads `$verdict.status` and `$verdict.fields.*`, verified against the real control
   verdict file (`status = PASS`, `gpu_pre_run = clean`, `gpu_post_run = clean`).

**The abort, verbatim** (`build/llama-cuda/gplr82/gplr82-n1-host2-tg16/runner.err.log`, tail):

```
row-cache (T-GPLR-19, SDD section 4 shape (i)): 40 layers x 16 rows/layer = 1093.0 MiB private VRAM, K=2, query-time retention
layer-major preflight: building persistent graphs
C:\Development\AI\qwen38GZI\third_party\llama.cpp\ggml\src\ggml.c:1787: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
```

Runner exit **`-1073740791` = `0xC0000409`** (Windows fastfail from the GGML assert): a ggml view
whose `data_size + view_offs` overruns its source tensor, raised **while building persistent
graphs for the row-set TG path** -- i.e. before any layer ever computed.

**How far it got (from the trace):** `load_end: ok` @6.46 s, `async_feed_init`, `byte_account`,
and all 40 `pkg_member` package records written. `run_start` confirms the flags took effect:
`rowset = on`, `tg_tokens = 16`, `schedule = batch-major`, `n_slots = 2`, `copy_source =
mmap-registered`. Then abort at ~6.75 s. 46 trace records total, ending at `run_start`;
**0 `layer_compute` records**, so no record could carry `experts_total`.

**Checker result** (`tools/check-gplr82-ledger.py` on that trace): **exit 2**,
`NOTHING TO EVALUATE`, `row-set records: 0 of 0 layer_compute records`. Per this checker's
contract, exit 2 means "nothing to evaluate" and is **NOT a pass**. The launcher correctly refused
to call it one: `ARM FAILED (runner exit -1073740791)`.

**Verdict / provenance.** Tag `gplr82-n1-host2-tg16`: `status = FAIL`, `exit_code =
-1073740791`, `duration_s = 17.9`, `mutex_acquired = True`, `gpu_pre_run = clean`,
`gpu_post_run = clean`, provenance gate **PASS** (digest `21ECF0325891CC00`, `llama.dll`
`892B24ECBFE1E956`, `binary_dir = build/llama-cuda/bin/unscoped-bin/s2-hostleg`, exe sha
`5A570B6DDE40D659`, `source_head = 9b90025c4f9dd4cd1cfe190f23e14fb7f75bf364`). Note the
wrapper needed **no** provenance waiver once the command reached it intact; the launcher's
`-AllowUnprovenanced` remained a recorded WAIVE, not a satisfied gate.

**GPU coordination.** Pre-run probe: **no compute processes** (launcher and wrapper agree). A
foreign `llama-layer-streaming.exe` from an unrelated `nwt2m-build` temp tree was seen on the GPU
during the FIRST aborted attempt; it was gone before this arm, did not recur, and did **not**
contaminate either arm reported here. Post-run probe: **no compute processes**. No shared-GPU
poisoning confound applies to this result.

**NOT ATTRIBUTED -- do not read this as a verdict on the mechanism.** The assert fires in the
row-set graph-construction path, which is where the host leg would live, but **no control exists**:
the only frozen binary (`s2-hostleg`) already contains the step-2 host-leg change, so there is no
pre-change binary to A/B against, and this pass was explicitly directed not to rebuild. The
assert is equally consistent with (a) a step-2 host-leg view built out of bounds and (b) a
pre-existing row-set/row-cache defect that `--experts-host 2` merely perturbs. Separating them
needs a side-by-side build, which is the next task, not this one.

**Also noted, unfixed, out of scope here:** the `run_start` telemetry record does not echo
`experts_host`, so a trace cannot prove from `run_start` alone that the host leg was requested;
`experts_total` appears only on `layer_compute` records, which this run never reached.

## 2026-10-05 (host-only, no GPU hold): T-GPLR-82 step 2 -- the host expert sub-op exists, is guarded, and is flagged off by default

**Task.** T-GPLR-82 **step 2 only** (the host sub-op), owner-directed 2026-10-05. Contract
`docs/plans/expert-compute-staging-decoupling.md`, S2. Build a CPU-backend expert sub-op that
computes routed rows against the weights where they already lie, add `--experts-host N`
(default 0 = off), and prove no weight source is rewired. **No GPU arm, no semantic QA, no
commit** was authorised for this pass, so this entry reports build and host-test evidence only.

**What was built.** One weight derivation over the FULL top-k, sliced into a device head leg
(`dev_rows = n_expert_used - N`) and a host leg (`N`), summed through the existing add chain
(`qwen35moe.cpp:435-486`). The host leg binds the MODEL LOADER's tensors
`model.layers[il].ffn_{gate,up,down}_exps` -- never `layer(il)`, which under
`LLAMA_LAYER_STREAMING` returns `ls->slot_layer(il)` and therefore the slot's compact CUDA
copies (`qwen35moe.cpp:892-949`, `build_moe_host_leg`). Host-leg ids are a separate INPUT
(`rowset_selected_experts_host`, `qwen35moe.cpp:472-476`), not a view of `sel_in`, and the
executor fills them from the last `N` top-k positions
(`llama-layer-streaming.cpp:4960-4996`). `build_moe_ffn` gained a null-defaulted
`moe_weights_in` (`llama-graph.h:1119`, `:1158`; `llama-graph.cpp:1924`, `:1977`) because
neither leg can normalise over its own subset without the pair summing to 2.0.

**The guard is on the real path.** `llama_ls_policy::ls_check_no_rewire`
(`llama-expert-split-policy.h:281`) is called from `build_moe_host_leg`
(`qwen35moe.cpp:934-942`), not only from tests, and a non-ok verdict `GGML_ABORT`s with the
refusal reason, the layer, and the counts. It now carries the two facts S2 names: the weight's
declared placement, and `GGML_BACKEND_BUFFER_USAGE_COMPUTE` on its buffer, which is what
separates a weight at its declared host placement from one the scheduler's `compute_splits`
path has staged -- placement alone cannot tell those apart. `min_batch_override_live` is passed
explicitly `false` because that route is not in this tree (reverted 2026-09-22); it exists so
re-introducing it has to touch this call site.

**Validated (measured, host-only).** `cmake --build build/llama-cuda --config Release --target
llama-layer-streaming` and `... --target test-layer-streaming-expert-split`: exit 0, zero
errors, final source fully built (both binaries up to date on a repeat build). ctest
`test-layer-streaming-expert-split` at 11:18:21 on a binary built 11:18:13 -- mtime checked
before believing it -- **PASS**, `AC1 PASS (0 failures)`, with five rewire-guard fixtures
(all-placed, one rewired, device leg ignored, compute-staging buffer, live MIN_BATCH). CLI
surface checked through the shipped binary: `--help` prints the flag, `--experts-host 99`
exits 2 with `expects 0..64 routed rows`. `load_row_set` verified **byte-identical**
(36,204 B before and after) against the pre-edit backup, so step 2 touched nothing in the
staging path. Scratch removed: no `g82*` file remains anywhere in the tree.

**Plan/code mismatches hit.** (1) The SDD's "zero router weight for the tail" pad is not
expressible in `build_moe_ffn`: SOFTMAX normalises, SOFTMAX_WEIGHT re-normalises, and
SIGMOID/SQRT_SOFTPLUS are strictly positive, so no gating path can exclude a padded row. S2
sidesteps it because the two positional widths sum to exactly `top_k` (no pad needed);
S3, where membership is per-row, will need an explicit mask this function does not have yet.
(2) S2 requires "weights bound to the in-flight source rows on the host", but the graph builder
runs once per graph set while residency changes per token; membership therefore had to move to
an executor input (`ff_input_hsel`), which is what makes S3's residency-aware rule possible
without a rebuild. (3) The policy header's namespace was `llama_layer_streaming`, which
collides with the class of the same name as soon as one translation unit includes both headers;
renamed to `llama_ls_policy` (function names unchanged). (4) The init-time input-buffer budget
for row-set companion inputs is computed from `row_set_requested`, which the runner assigns
AFTER `ls.init`, so those inputs already take the spill path today; left untouched, and the
third host-leg input behaves the same way rather than diverging.

**Not validated, and it matters.** No GPU arm ran, so nothing here proves the host sub-op
executes correctly: the scheduler's backend assignment for CPU-resident `MUL_MAT_ID` weights,
the cross-backend ACTIVATION round-trip the SDD names as permitted-but-unproven, and any
numerics drift from CPU accumulation are all unmeasured. `experts_host > 0` in a trace is
still the falsifier AC7 asks for, and it is unmeasured. Step 3 (removing the device wait) is
deliberately absent, so this pass buys no speedup by construction.

**Recommendation.** Keep experimenting. Treat step 2 as a compiled, guarded skeleton: the
placement argument is now checked at run time instead of asserted in a comment, but the
execution question is unanswered until an owner-authorised GPU arm reports a non-zero
`experts_host / experts_total` with balanced device+host accounting.

## 2026-10-03 (host-only, no GPU hold): T-GPLR-82 step 1 -- the decoupling is now observable; experts_* ledger, TG readiness, per-token window

**Task.** T-GPLR-82 **step 1 only** (instrumentation), owner-directed 2026-10-03. Make the
decoupled-compute mechanism *observable* before any behaviour changes, so a working host
sub-op can later be told apart from an inert one. Scope: `src/llama-layer-streaming.cpp` and
its header. **No host execution path, no `--experts-host` flag, prefill untouched.**

**What was added, and where.** All default-inert; every field is emitted from a site that
already existed.

| field | event | anchor | note |
|---|---|---|---|
| `experts_total`, `experts_host`, `experts_device`, `host_us` | `layer_compute` | `main.cpp:7023-7041` | inside the row-set-only `parts` block |
| `t_begin_us`, `round_wall_us` | `rowset_round` | `main.cpp:10857-10863` | pairs with the existing `t_us` |
| `ready_at_consumption` | `rowset_fetch` | `main.cpp:10836-10842` | TG had no readiness field at all |
| `rowset_round_t0_us` (member) | -- | header `:1711` | round-start stamp |
| `t_layer_compute` signature | -- | header `:1038` | +4 defaulted params |

**Inertness rests on the existing gate, not on new plumbing.** The optional `parts` block is
written only when `attn_compute_us >= 0.0`, and **only the row-set TG caller
(`main.cpp:4948`) passes that**. Both prefill callers pass `-1.0` and therefore keep the
byte-identical short form. The buffer grew `160 -> 320` bytes, which cannot affect a record
that never uses it. `ffn_dev` is emitted **unchanged**.

**Two plan/code mismatches, found and handled.**

1. **The plan's "per-token `t_us` on `rowset_round`" was already satisfied** -- `tel_rowset_round`
   has emitted `t_us` since before this task. The real gap is different: there was no round
   *start*, so the gap between consecutive stamps was unattributable and the 2,662.864 ms
   `tg82` outlier could not be placed (round stamps spanned 7.249 s of a 10.680 s wall).
   Implemented the start stamp instead, and added `round_wall_us` so each token's window is
   measurable end to end.
2. **`ready_at_consumption` on TG is a LATENCY PROXY, not an observed wait.** The row-set path
   has no consumption-time wait site to observe, so the field reuses the existing `<100.0 us`
   convention that `main.cpp:4961` already feeds to `add_layer_stats`, making the trace and the
   in-memory statistic agree by construction. This is recorded in a code comment precisely so
   it is not later read as the measured readiness gate it was meant to close.

**`host_us` is a real measurement but currently measures the wrong thing.** It times this
layer's host-side row-set consume (union build + `load_row_set`), which today is the only host
work on the path. It does **not** yet measure host expert arithmetic -- there is none until
step 2. From S2 the same timer covers that too. Anyone reading `host_us` before S2 must not
treat it as the 36.5 / 59.4 us/expert figure.

**Validation.** Host-only, no GPU, no mutex. `cmake --build ... --target
llama-layer-streaming` **exit 0**, 0 errors; `llama.dll` and `llama-layer-streaming.exe`
relinked, `llama.dll` sha256 now **`1c52865caf0befa126c1ec5955e858d8c0d00e54e1fca753383d5e855710e221`**
(previously `715ca377...`). `ctest -R test-layer-streaming-expert-split` **Passed**, exit 0.
Format specifiers were checked against their argument types by hand at all three emit sites,
since a mismatch compiles but corrupts the trace. The four `C4477` warnings in the build log are
`[ls-exec]` debug `fprintf`s at `main.cpp:5159/5176/5270/5276` -- outside every S1 hunk and
present in the pre-patch file, so pre-existing.

**Method note, because it nearly caused a wrong edit.** This pass first "verified" the vendored
files were CRLF using `grep -c $'\r'`, which in Git Bash matches **every line** and therefore
returned the line count for both files -- indistinguishable from a true "all CRLF" result. Both
files are **LF**. The patcher now detects newline style from the raw bytes and re-asserts it
after writing, instead of assuming it. This is the same failure shape as the earlier
`ready_at_consumption` `None is False` defect: a check that looks like it passed and did not.

**Not done, by instruction.** No `--experts-host`, no host sub-op, no prefill change, no GPU
arm, no semantic QA. Nothing committed.

## 2026-10-03 (no GPU hold, read-only analysis of the tg82 trace): where every demanded expert ran during token generation -- all of them on the GPU, 49.2% cache-resident, and no layer ever waited

**Task.** Owner, after the `tg82` run: "how was the impress [TG] happening?
When was an expert being executed on the CPU and when on the GPU? What happens
with experts that are waiting to be transferred? ... in flight? ... when the
router demands an expert that is not gonna be mapped?" Answered from the `tg82`
trace alone, no new hardware run.

**The headline is that this run has no CPU/GPU split at all.** Across all
**5,120** token-generation `layer_compute` records: `ffn_dev` = `{1: 5120}`,
`attn_dev` = `{1: 5120}`, and **`cpu_us > 0` in 0 of 5,120**. Not one demanded
expert row was executed on the host. So the three sub-questions resolve as
follows, and none of them has a host leg in this configuration.

**1. Where each expert ran.** `top_k` = 8 on every layer, `route_union`
median 8.0 -- at batch 1 the eight experts a layer demands are eight distinct
experts, so **40,960 rows** were demanded over the phase (40 layers x 128
tokens x 8). **Every one executed on the GPU.**

**2. What was already there.** The resident row cache
(`--residency-tg-retain 2`) served **20,132 of 40,960 = 49.2%**. The other
**20,828** came across PCIe. `fetched + reused = 40,960` exactly, so every
demanded row is accounted for with no unexplained residue. `n_rows_retired`
sums to 20,188, i.e. the cache is evicting continuously, not accumulating.
The first token is cold -- **320 fetched, 0 reused**; rounds 2-128 settle at a
**49.5%** reuse rate.

**3. In flight / waiting. TG readiness-at-consumption is UNMEASURED, and an
earlier draft of this entry claimed otherwise -- that claim is withdrawn.**
`ready_at_consumption` is **not a field on `rowset_fetch` at all**; it lives
only on `layer_ready_wait`, which this trace emits **1,960 times, all of them
phase 0**. There are **zero** TG-phase `layer_ready_wait` records. So "no layer
ever blocked on an unfinished transfer" has **no evidence behind it** and must
not be quoted. What TG *does* record per layer: median `host_submit_us`
**0.072 ms** (issuing), `transfer_us` **0.733 ms**, `ready_latency_us`
**0.662 ms**, `bytes_staged` median **6.750 MiB** (max 15.562). Roughly
40 x 6.75 MiB = **~276 MiB staged per token**, matching the per-round median
of 276.211 MiB.

**PP, for contrast, is measured and blocks constantly:** of the 1,960
`layer_ready_wait` records, **1,421 (72.5%) were NOT ready at consumption**,
median `wait_us` **31.472**, p90 31.715, max 123.590. The scheduler's
readiness signal is real and does fire -- it is simply not instrumented for
token generation. The indirect TG argument (median `ready_latency_us`
0.662 ms against a per-layer share of 51.784/40 = 1.29 ms) *suggests* the
copy wins the race every time, but that is an inference from two medians,
**not a measurement**, and it is exactly the class of claim this project's
gates exist to catch.

**4. Router demands an expert that is not mapped.** It is **staged from the
host mmap over PCIe and then executed on the GPU.** There is no host-execution
fallback in this configuration -- that is exactly the gap
[T-GPLR-61 step B](docs/plans/generic-prefill-layer-residency-plan.md) exists
to fill. Note this is the *opposite* of the cost-gate premise: the gate was
measured on host expert arithmetic at ~36.5 us/expert against ~171 us/expert
over the link, i.e. computing on the host is 3.4-6.7x cheaper than fetching,
so a not-mapped expert being fetched is the expensive branch by that model.

**Correcting a number that reads well and means little.** TG per-token *layer
compute* median is 16.395 ms = **60.992 tok/s**, but that **excludes the
staging**. The end-to-end steady-state figure is the round-to-round gap:
median **51.784 ms = 19.31 tok/s** (mean 57.082 ms = 17.52 tok/s). Against the
stock anchor's **40.96 tok/s** (24.41 ms/token) this configuration is
**0.47x, i.e. roughly 2.1x slower** -- not "quite good". The 60.99 figure must
never be quoted as a decode rate.

**An inconsistency I could not resolve, recorded rather than smoothed.** The
worst token is `ubatch` 50 at **2,662.864 ms** of summed layer compute (second
worst 247.927 ms), yet the `rowset_round` gap from round 49 to round 50 is only
**410.424 ms**, and the rounds span 7.249 s of a 10.680 s TG wall, leaving
**3.431 s outside the round loop**. The 2.66 s cannot be placed inside the
410 ms gap, so either `rowset_round.t_us` marks round submission rather than
completion, or the rebuild is attributed into `layer_compute` while it is paid
outside the loop. **Not established.** Resolving it needs a `t_us`-per-token
event, which is what the trimmed-median telemetry proposal would add anyway.

**New tool `tools/tg-expert-routing.py`** (5,749 B): prints the execution-side
census (`ffn_dev`/`attn_dev`/`cpu_us`), the demand-versus-residency ledger
(`route_union`, `n_rows_fetched/resident/retained/retired`), the in-flight
timings, and the per-token reuse rate. Two defects found and fixed while
writing it, both in the tool rather than the trace: the missing-trace path
raised a traceback instead of a clean message (the same defect
`tools/pad-delta.py` was tightened for), and `n_rows_reused` was read off
`rowset_fetch`, where the field does not exist -- it is a per-`rowset_round`
field, and reading it per-layer is what made the reuse ledger look empty.
Negative path verified exit 1 with a clean message; real trace exit 0.

**A third tool defect, found on the final verification pass, of the same family
as the second.** Section 3 asked `e.get("ready_at_consumption") is False` over
`rowset_fetch` records. That field does not exist there, so `.get()` returned
`None`, the predicate was False for all 5,120, and the tool printed a clean,
plausible **`0 of 5120`** -- which reads as "every layer was ready". A missing
field and a false field are different facts, and `None is False` silently
converts the first into the second. The tool now counts field *presence*
separately and prints an explicit `UNMEASURED` warning when it is absent.
**This is the third time in this task that reading a field off an event that
does not carry it produced a confident wrong answer** (`n_rows_reused`, then
`ready_at_consumption`). The rule worth adopting: a trace-analysis tool must
fail loudly on an absent key rather than default it.

## 2026-10-03 (GPU hold tg82, current binary): the 50k / 128-token shape re-run -- PP inside the band, and one 2,662.864 ms outlier token that poisons the mean

**Task.** Owner: "run some benchmark ... 50000 tokens and 128 generating
tokens." The shape already existed (`tg81-pad50k`, four arms), but **every one
of those arms ran on the previous `llama.dll`**: the current dll is
`715ca3779a072789861659fa0d129b709905c24a5180d95f5828df0ca7d5e59b`, mtime
2026-10-03 10:06, and the `tg81` arms ran 06:48-07:05. A fresh arm was
therefore the only way to get a **current-binary** point at this shape. No code
change, no rebuild.

**Design.** One arm, `j1-pad256-50k`, flag-identical to `i1`/`i4`
(`--residency moe-expert-bank --pp-schedule batch-major --residency-tg
--residency-tg-retain 2 --layer-slots 2 --kv-graph-pad 256 --kv-cache q8
--ctx-size 262144 --tg-tokens 128 --ubatch 1024 --copy-source mmap-registered
--route-telemetry 1`, `anchor-50000.txt`). Launched through
`tools/with-gpu-lock.ps1 -VerdictTag tg82-50k128 -AllowUnprovenanced`; mutex
acquired and released, `gpu_pre`/`gpu_post` both clean, verdict **PASS**,
command exit **0**, 160.9 s. Evidence
`build/llama-cuda/logs/tg82-50k128-current/`. The provenance gate **REFUSED**
the arm (`binary_state=not-supplied`) and the run passed only under the waiver;
the dll hash above is the binary identity, recorded here by hand.

**Prefill is unchanged and inside the `tg81` band.** 50,000 tokens, wall
**137.138 s**, mean **364.597 tok/s**; transfer 151.213 s, wait 45.620 s,
compute 56.821 s, overlap **1.850x**, 1,958 evictions, feed **5.939 GB/s**.
The four `tg81` arms read 348.8-379.9 tok/s, so this sits inside them. The
transfer+wait+compute overlap (measured, 1.78-1.90x across all five arms) is
**DMA against SM** -- it is not host arithmetic, and it must not be quoted as
evidence of CPU/GPU compute concurrency.

**Token generation: the mean is meaningless here, the median is not.** The
phase summary reads **mean 11.985 tok/s / 10.680 s wall** -- worse than `i1`'s
18.600 -- but that is the mean-contamination pathology this project already
knows. Per-token cost (sum over the 40 `layer_compute` records of
`compute_us + setup_us`, grouped by `ubatch`):

| arm | binary | median/token | median tok/s | mean/token | max token |
| --- | --- | --- | --- | --- | --- |
| `j1` | **current** | **16.395 ms** | **60.992** | 39.247 ms | **2662.864 ms** |
| `i1` | previous | 16.633 ms | 60.121 | 18.223 ms | 151.139 ms |
| `i4` | previous | 16.433 ms | 60.853 | 18.702 ms | 199.201 ms |

**The three medians agree within 1.4%**, so the binary change is not a TG
regression -- `j1`'s median is in fact the best of the three. The entire mean
gap is **one token**: `j1`'s single 2,662.864 ms outlier, the `view_rebuilds_total=1`
KV view rebuild at `required=50176`. It contributes (2662.9 - 16.4)/128 =
**20.7 ms/token** to the mean, which is the whole 18.600 -> 11.985 difference.
This is the **largest single-token outlier recorded at this shape**, ~17x `i1`'s
151 ms, and it is the strongest instance yet of the standing rule that
`phase_summary.tokens_per_s` must never be quoted as a TG result.

**Re-confirms T-GPLR-79 on the rebuilt dll.** Across all 5,120 TG layer records
`ffn_dev` is `{1: 5120}` and **0 of 5,120 ran host arithmetic** (`cpu_us > 0`).
The CPU+GPU *concurrent* case still does not exist on this binary; this run
re-measures the zero-host-compute result rather than the overlap model.

**New tool `tools/trace-summary.py`** (3,791 B): phase summaries with the
overlap ratio and achieved feed, plus per-token median/mean/min/max built from
`layer_compute` grouped by `ubatch`, plus the host-arithmetic and `ffn_dev`
census. It exists because the first per-token implementation in this session
filtered on `phase == "tg"` and let `phase_summary` through, reporting the
phase wall as a "median" of n=1; the corrected filter keys on
`event == "layer_compute"` and `phase == 1`.

**Not established.** No A/B against the `tg81` arms is available for PP/TG
*means*, because that binary boundary is unresolved in both directions: the
medians match, but n=1 arm per binary cannot separate a real small regression
from session drift. Nothing here tests answer stability, and the charter's
262,144/50k identity arms remain impossible without handoff.

## 2026-10-03 (host-only, no GPU hold): T-GPLR-61 step A -- the `moe-row-split` policy and shape are written and AC1 PASSES. The identity gate is untouched. Soft gate -- advisory evidence only.

**Task.** Owner, after asking what hybrid-inference results exist at K=2: *"create an experiment ... what you have on the GPU, you run on the GPU, what's in flight, you wait for it, what is not in flight, we will CPU in the there."* That is verbatim the recorded owner direction for **T-GPLR-61**, which the plan already carries as `ready`. Scope agreed with the owner before any code: **step A only** (host-only policy and shape), with the AC2 identity probe deferred and the 512-token anchor chosen when it is run. No GPU hold, no model load, no change to any default path.

**What the record already said, so nothing was re-derived.** Stock llama.cpp hybrid exists and is strong: all 120 expert-bank tensors on `CUDA_Host`, 16 whole layers on CPU, **39.204 tok/s TG at 10.193 GiB** (owner doc1, 2026-10-01 head-to-head). Our layer-residency path at the same context is **25.074 tok/s = 0.640x stock**, while winning prefill **1.94x**. So the hybrid the owner describes has never run *here*: `experts_host`, `rows_gate_cpu` and `split_expert_ffn` return **zero hits** across `third_party/llama.cpp/src` and `/tools`.

**What step A now contains.** A new standalone header
`third_party/llama.cpp/src/llama-expert-split-policy.h` (9,590 B) holding pure
decision functions in `namespace llama_layer_streaming`: `ls_classify_row`
(READY / PENDING / ABSENT), `ls_evaluate_gate` (named policy, refuses rather
than defaulting), `ls_plan_split` (partition + two constant-width descriptors +
the SDD's telemetry counters). It is deliberately its own header and adds **no
members to any class**, so it cannot change the layout of anything already
linked. Test `tests/test-layer-streaming-expert-split.cpp` (17,659 B), registered
in `tests/CMakeLists.txt` inside the `LLAMA_LAYER_STREAMING` block, and added to
the host suite's explicit CTest allowlist in `tools/run-tests.ps1`.

**AC1 PASSES -- all seven fixtures, plus the gate arithmetic.** Focused CTest
`test-layer-streaming-expert-split` passes; host suite `SUCCESS`, **78/78 CTest
tests and 0 FAIL**, exit 0. **The suite did not run this test at first.**
`run-tests.ps1 -Suite host` selects by an explicit allowlist, not by "all
registered": the new name was absent, so the suite passed 77/77 without ever
executing AC1 -- a green suite that silently omitted the one test added that
week. Fixed by adding `expert-split` to that allowlist, after which the count
went 77 -> 78 and the log contains the test's own `AC1 PASS` line. The gate reproduces the SDD's measured comparison rather than
a convenient one: `wait_us` lands at 170.96 us for a 1,769,472 B row at
10.35 GB/s against `host_us` 36.5, a **4.68x** margin, inside the recorded
3.4-6.7x band.

1. all-READY **never splits** -- `experts_host == 0`, `staged_bytes_avoided == 0`.
2. all-PENDING with the gate open sends every row to the host, 8 x 1,769,472 B avoided.
3. a **closed gate is not a refusal**: PENDING rows return to the ordinary wait
   path, and the device sub-op is asserted **slot-identical** to the no-split
   plan. This is the property that makes the flag safe to add.
4. a mixed set splits per row (4/4), and the router's executed id set is
   preserved in order on **both** sub-ops -- nothing dropped, substituted or reordered.
5. an **absent** demanded row refuses and emits no sub-op at all.
6. an **unevaluable** policy refuses: unknown policy, zero feed rate, and a
   *negative* feed rate (which must not sneak through as free transfer).
7. padding is inert (`is_pad`, no id, no backend) and both sub-ops are exactly
   `top_k` wide for **every k from 1 to 8** and **every residency mix**, with
   `experts_device + experts_host == experts_total` throughout.

**Two defects the fixtures caught, both fixed in the policy rather than in the test.**
(i) `rows_gate_gpu` counted only PENDING rows that stayed on the device, so the
gate's two counters did not sum to the demand in a mixed set. READY rows are
dispositions too, so it now counts every device-assigned row and the pair sums.
(ii) A slot not host-bound was emitted with `expert_id = -1`, which made the two
sub-ops **differently shaped** instead of **differently weighted**. The SDD's
padded shape requires both partials to be full-width terms carrying their own
id, selected out by a zero router weight; without that the add-chain reduction
would not sum correctly. Both sub-ops now carry the expert id for every demanded
position, and only tail padding uses -1.

**A build-hygiene cost worth recording.** Adding any file under
`third_party/llama.cpp` makes the provenance attestation fail with *"the binary
PREDATES its sources"*, because the checker compares the newest source against
the newest engine artifact. An ordinary incremental rebuild does **not** clear
it: MSBuild correctly declines to relink an engine whose inputs did not change,
so the artifact stays at its old mtime. The host suite failed **6 arm-verdict
cases** on this. The fix is a real rebuild, and `--clean-first` is what does it
-- it costs a full CUDA template rebuild (~15 min, 106 artifacts); only the known
pre-existing `test-unicode` `unicode_regex_split` link failure remains. **The
engine binary `llama.dll` hash changed** (715CA3779A072789) from the one behind
the 2026-10-03 pad arms, so **any A/B against those arms is now cross-binary**.

**NOT proven, and deliberately so.** Step A proves the partition and the refusal
contract **only**. Nothing here has executed on hardware, no runner reaches this
code, and no host sub-op exists yet. AC2 -- greedy token ids byte-identical to
the all-GPU arm at the 512 anchor -- is untouched and remains the blocking gate;
per AC5 a run in which the mechanism never fires is a recorded outcome, never a
pass with an implied benefit. **No performance claim of any kind is made.** The
SDD's named residual hazard is also untouched: activations crossing the backend
boundary are permitted by letter and evidence but **not proven safe**, and that
is where the per-layer sync cost lands.

**Tooling warning that affects every future doc search in this repo.**
`code_search` and `read_files` resolve to the Freebuff **worktree**, which is a
much older copy: `docs/tg-expert-path-reconciliation.md` is 131,204 B there
against **366,496 B** in the repo, and `docs/plans/qwen36-staging-contention-arm.md`
exists only in the worktree. Every finding in this entry came from `grep`/`sed`
in the terminal against the real tree. Searches run through the agent tools on
these documents are reading stale content.

**Evidence.** `third_party/llama.cpp/src/llama-expert-split-policy.h`,
`tests/test-layer-streaming-expert-split.cpp`, `tests/CMakeLists.txt`;
suite record `build/llama-cuda/suite-runs/20261003T081238Z-host/revision.json`
(digest `175CB7FDE0CFED8E`, nested head `9b90025c4f9d`, 12 tracked / 3 untracked
changed, `llama.dll` 715CA3779A072789). Commit: unassigned -- no commit
authorised; outer HEAD `98773fe` and nested `9b90025c4f9d` both unmoved.

## 2026-10-03 (2 GPU holds, 4 matched arms at the 50k shape): the KV graph pad is worth **+19.5% TG** at 50k, and the previous session's headline for that arm was an artifact. Soft gate -- advisory evidence only.

**Task.** Owner: "Measure the KV graph pad benefit at the 50k shape by running a
same-session no-pad control" -- `h3-pad256-50k` (12.568 tok/s / 79.57 ms per
token) had no matched comparison, so it was not a delta. Scope: the 50k shape
only. No code change, no rebuild; the binary is byte-identical to the one that
produced h3 (`llama-layer-streaming.exe` mtime 2026-10-02 23:55:01, unchanged).

**Design.** Four arms, two pad and two no-pad, at the charter 50k shape,
identical in every flag except `--kv-graph-pad 256`, split across **two separate
GPU mutex holds** with the order **reversed** in the second hold
(`i1-pad, i2-nopad` then `i3-nopad, i4-pad`) so a monotone session drift cannot
masquerade as the pad effect. All four exit 0, 128/128 tokens, 149-166 s,
gpu_pre/gpu_post clean. Evidence `build/llama-cuda/logs/tg81-pad50k/`,
verdicts `tg81-pad50k` / `tg81b-pad50k` (both PASS; both waved through the
provenance gate with `-AllowUnprovenanced`, so neither carries binary identity).
**Provenance gap, recorded rather than papered over:** `run_start` does not emit
`kv_graph_pad` at all, so the telemetry alone cannot say which of the four arms
carried the pad. Arm identity rests on `arms.txt` plus the launch script, and
the corroboration is indirect -- the two pad arms each show exactly 1
`tg_view_rebuild` at `required=50176` and each no-pad arm shows 0 and stays at
`kv_view_live=262144`, which is only reachable with the flag set. Adding
`kv_graph_pad` and its `_source` to the `run_start` event would make this
self-proving and is a one-line change for whoever picks it up.

**Result -- TG, measured on the median per-token latency, not the mean.**

| arm | pad | median ms | trim ms | tok/s (median) | graph builds |
|---|---|---|---|---|---|
| i1 | yes | 51.32 | 50.21 | 19.49 | 1 |
| i4 | yes | 50.80 | 49.40 | 19.68 | 1 |
| i2 | no | 59.19 | 58.09 | 16.90 | 0 |
| i3 | no | 62.87 | 61.39 | 15.91 | 0 |

Pad median-of-medians 51.06 ms vs no-pad 61.03 ms: **no-pad is 19.5% slower per
token** (tok/s 19.58 vs 16.39). The two arms' medians do not overlap -- gap
7.87 ms. Pooled over all 254+254 per-token samples, 254/254 pad samples are
faster than 253/254 no-pad samples (Mann-Whitney z = -13.85, p = 1.3e-43). All
four 2x2 pairings land between +15.3% and +23.8%. The pad arm also replicates
tightly (51.32 vs 50.80, 1.0% apart); the control is the noisier of the two
(59.19 vs 62.87, 6.2% apart), so the headline delta carries roughly that much
spread.

**Measured costs, same arms.** PP is untouched, as it must be -- the pad is a TG
mechanism: pad PP 374.2 tok/s vs no-pad 364.4 tok/s against a 348.8-379.9
spread across all four arms, i.e. inside run-to-run noise. Peak VRAM is
**bit-identical at 10,074,259,456 B in all four arms**: the pad costs nothing in
memory. Each pad arm still shows exactly 1 rebuild
(`built=262144 -> required=50176`, `mechanism=t-gplr-47-fresh-context-rebuild`);
each no-pad arm shows 0, staying pinned at 262,144-wide for all 128 tokens.

**Why the median and not `phase_summary.tokens_per_s`.** `h3-pad256-50k` and
`i1-pad256-50k` are the **same configuration, same binary, same prompt**, run
17 hours apart, and they disagree by 48% on the headline: 12.568 vs 18.600
tok/s. Their medians agree within 4% (53.56 vs 51.32 ms). The cause is a single
token: h3's first TG token took **419.39 ms** against a ~53 ms median, and it
carries the graph rebuild. One such token moves a 128-token mean by ~30%, and
it moved h3's reported `total_compute_us` (3382 ms) *above* the no-pad control
(3173 ms) -- which is why the attention-width model fitted against h3 returns a
**negative** implied attention cost of -258 ms, a physical impossibility that
would have read as a falsified mechanism had it not been checked. `i3` has the
same pathology in the control arm (max 1189.12 ms, mean 74.46 vs median 62.87).
`tools/pad-delta.py` now reports mean, median, 10%-trimmed mean and max side by
side so this is visible rather than buried in a single field.

**Mechanism cross-check.** If the only thing the pad changes is the live
attention view width, then TG compute = A*(live/262144) + R with A and R shared
by both arms. Solving each clean pair separately gives A = 1362 ms (i1 vs i2)
and A = 1241 ms (i4 vs i2); the previously recorded 512-shape pair gives
A = 1459 ms. The mechanism's falsifiable prediction is that three different live
widths (768, 50176, 262144) imply the SAME full-width attention constant, and
they do agree to within **14.9%** (1459 -> 1241 ms) -- but that figure should be
read honestly: the two 50k fits differ from *each other* by 8.9% despite
identical flags, so ~9% of the spread is arm-to-arm session noise, not model
error. The prediction is supported at the level the noise permits, not
confirmed, and it is the weakest of the claims in this entry. This is why the pad helps at all: with the whole
262,144-wide cache live during decode, attention reads 262,144 columns when only
~50,000 are real.

**Token output: identical.** All four new arms and h3 emit **the same 128 token
ids**, pad and no-pad alike, first divergence `None`. This is new information
about the open T-GPLR-80 question: the pad does **not** perturb output at the 50k
shape. The recorded 512-shape divergence at index 42 (pad 248045 vs no-pad 846)
is **not** reproduced at 50k over 128 tokens. That weakens hypothesis (b) -- "the
whole-cache view is genuinely wrong" -- because a systematic mask defect at
262,144-wide should show at both shapes, and it now shows at neither reliably.
It stays consistent with hypothesis (a), a near-tie resolved differently because
a 50,176-wide and a 262,144-wide softmax reduce in a different order. **This is
not settled**: 128 tokens is a short window, the 512-shape divergence is still
unexplained, and `--tg-logit-margin` -- the instrument that measures the margin
at the divergence point -- is still OFF on every 512-shape arm recorded so far.

**Charter-scale caveat, measured not assumed.** The pad's benefit is largest
exactly where the charter does *not* operate. It scales as 1 - live/262144, so
at `--ctx-size 262144` the projected compute saving is ~41% at a 500-token
prompt, 33% at 50k, 25% at 100k, 9.7% at 200k, and **0.0% at a 262,144-token
prompt by construction** -- the whole cache is genuinely live, so there is
nothing to narrow. The projection also assumes TG wall keeps its measured
~42%-transfer / ~22%-compute split, so it understates nothing and overstates the
wall-clock share if transfer dominates harder at longer contexts. The honest
reading is: `--kv-graph-pad 256` is a **short-prompt, long-ctx** optimisation.
It is worth having and it is free, but at the charter's own 262k context it buys
nothing, and this measurement does not support it as a lever for the 8 GB target.

**Not concluded.** Gates are soft: this entry reports what four arms measured,
not that the pad is correct or approved. Still open after this work: the 512-shape
token-42 divergence; whether the pad is safe at contexts where `live` crosses a
pad block during generation; the unattributed RAM commit hold; ctx ~= 125,700 on
K=2 (8 GiB route) never run.

## Index of archived entries (2026-08-18 .. 2026-10-02)

Format: `<date> -- <shard>:<line> -- <entry>`, newest first. The rows
are derived from the shard files; do not edit them by hand.

- 2026-10-02 -- `docs/currentstate/2026-10-02.md:9` -- ## 2026-10-02 (3 GPU holds, host suite): `--kv-graph-pad 256` token generation is FIXED in the row-set executor. The...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:107` -- ## 2026-10-02 (1 GPU hold + host analysis, owner-directed): attention in the K-cache TG path is ALREADY 100% on the G...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:192` -- ## 2026-10-02 (host-only, no GPU hold): T-GPLR-78 -- the T-GPLR-75/76 classifier is CORRECTED at the source. No-kerne...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:242` -- ## 2026-10-02 (1 GPU hold, 2 arms, owner-directed "maybe do not use mmap?"): the answer is NO -- `--no-mmap` ADDS ~th...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:326` -- ## 2026-10-02 (host-only, no GPU): T-GPLR-77 -- `docs/currentstate.md` re-split under the size guard (480,985 -> 351,...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:376` -- ## 2026-10-02 (1 GPU hold + host build): T-GPLR-76 answers "which nodes" — the ONLY host-side op in the attention gra...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:447` -- ## 2026-10-02 (3 GPU holds, 10 arms, bisect): the ~58 GiB commit hold is NOT ours and NOT context-driven — ctx is eli...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:539` -- ## 2026-10-02 (1 GPU hold + host build): T-GPLR-75 answers the owner's CPU-inference question BY MEASUREMENT — the K...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:625` -- ## 2026-10-02 (4 GPU holds, owner-directed TG focus): the K cache is MEASURED at last — TG 13.3-16.2 tok/s, peak VRAM...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:822` -- ## 2026-10-02 (host-only, code inspection, no GPU hold): the K-cache was NEVER tested at our shape and is not wired i...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:944` -- ## 2026-10-02 (ONE GPU hold, 4 arms, owner-authorised): the CPU thread lever is REFUTED on hardware — 1/4/12/16 threa...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1025` -- ## 2026-10-02 (host-only, NO GPU): NUMA is ELIMINATED (1 socket, 0 NUMA nodes) and first-touch is RULED OUT BY MAGNIT...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1116` -- ## 2026-10-02 (host-only analysis + one host build, NO GPU): the overlap question is ANSWERED from existing traces —...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1188` -- ## 2026-10-02 (HOST-ONLY microbenchmark, NO GPU, NO mutex): the 36.5us/expert figure REPRODUCES at 35.07 (12 threads)...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1274` -- ## 2026-10-02 (host-only, NO GPU): the 42.6%-vs-4.0% blocker is CLOSED and the policy question it gated is answered —...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1350` -- ## 2026-10-02 (host-only + one host build, NO GPU): pool hit-rate observability FIXED — the published rate's denomina...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1426` -- ## 2026-10-02 (host-only, NO GPU): per-layer K is CORRECT and maximally valuable at small K (+1.96 pp at K=1) — but K...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1500` -- ## 2026-10-02 (host-only, NO GPU): the owner's simpler rule ("in-flight wait, queued don't") is ALREADY implemented i...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1574` -- ## 2026-10-02 (host-only, NO GPU): the proposed byte-accurate PCIe ledger is WITHDRAWN as unnecessary — byte accounti...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1657` -- ## 2026-10-02 (host-only, NO GPU): the 42.6% vs 4.0% conflict was a CATEGORY ERROR — they are disjoint terms, both tr...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1741` -- ## 2026-10-02 (expert row-cache K, host-only, NO GPU): per-layer K is worth at most +2 pp over flat K, the 8 GB targe...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1836` -- ## 2026-10-02 (T-GPLR-71 dip REPLICATION, one mutex hold, 12 interleaved arms): the ctx-131,072 dip DOES NOT REPRODUC...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1892` -- ## 2026-10-02 (T-GPLR-71 dip investigation, host-only, NO GPU): the 6.5% ctx-131,072 dip is EXONERATED of the mechani...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:1957` -- ## 2026-10-02 (T-GPLR-71 CONSOLIDATED OWNER SUMMARY — this entry SUPERSEDES the ab20/ab21 owner summary below): the v...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2063` -- ## 2026-10-02 (T-GPLR-71 audit close-out, host-only, NO GPU): the test's own justification is now measured, `set_view...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2133` -- ## 2026-10-02 (T-GPLR-71 guard proof, host test + ONE hardware arm): steps 3-4 are EXERCISED, not dead code — and dri...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2228` -- ## 2026-10-02 (T-GPLR-71 ab20/ab21, TWO mutex holds, FIRST hardware execution of `--kv-graph-pad`): token generation...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2295` -- ## 2026-10-02 (T-GPLR-71 step 5, host-only, observability only): AC1, AC3 and AC6 are now scorable from a DEFAULT tra...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2320` -- ## 2026-10-02 (T-GPLR-71 audit follow-up, host-only): the view-coverage fail-safe pointed the wrong way — an unrecord...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2347` -- ## 2026-10-02 (T-GPLR-71 steps 3-4, host-only, NO GPU window): the TG set is rebuilt when its built view cannot cover...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2374` -- ## 2026-10-02 (T-GPLR-71 steps 1-2, host-only, NO GPU window): the control is preserved, `--kv-graph-pad` exists and...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2402` -- ## 2026-10-02 (host-only, source reading): YES the view can be sized at graph-build time — the handoff executor is in...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2437` -- ## 2026-10-02 (host-only): the CPU is NOT an extra additive cost inside the 43 ms — but host layer execution is 42.6%...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2469` -- ## 2026-10-02 (prof3, host-only analysis): stock's decode scan length is ESTABLISHED — our engine scans the whole all...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2508` -- ## 2026-10-02 (host-only): the decode attention kernel is already at ~97% of its block-wave occupancy, and the MMA sp...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2536` -- ## 2026-10-02: T-GPLR-70 FALSIFIED AND REVERTED -- narrowing the TG KV window made token generation 18.9% SLOWER, and...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2572` -- ## 2026-10-02 (host-only): the ten full-attention layers cost ~2,075 us because TG's decode attention scans the WHOLE...
- 2026-10-02 -- `docs/currentstate/2026-10-02.md:2641` -- ## 2026-10-02 (host-only): T-GPLR-68 ATTRIBUTED — the unaccounted VRAM is llama.cpp's own compute-buffer reserve, and...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:9` -- ## 2026-10-01 (host-only): the activation slab's second half IS resident, telemetry reports the sum exactly, and the...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:24` -- ## 2026-10-01 (ab18b, host-only): the per-layer token-generation decomposition — and a RETRACTION of the "788 us/laye...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:42` -- ## 2026-10-01 (ab18): the MATCHED 8 GB comparison — prefill **3.72x** stock, token generation **0.62x**, at the same...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:65` -- ## 2026-10-01 (ab17): the TG leg finally has numbers — the hybrid does BOTH legs inside 8 GB, prefill is 2.11x stock...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:92` -- ## 2026-10-01 (ab16): T-GPLR-66 CLOSED — the `alloc_spill` crash is fixed, the depth-2 ubatch curve is complete, and...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:122` -- ## 2026-10-01 (ab15): the depth-2 PP curve on the real card — ubatch 1024 AND 2048 CRASH, and the act-slab law retire...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:147` -- ## 2026-10-01 (ab14): the 8 GB TARGET CONDITION, budget-emulated — it ACCEPTS, it is VRAM-neutral, and it admits a co...
- 2026-10-01 -- `docs/currentstate/2026-10-01.md:191` -- ## 2026-10-01: THE OWNER'S TWO REAL CONFIGURATIONS, RUN HEAD TO HEAD — layer-major is a PP-ONLY win (1.94x / 1.83x at...
- 2026-09-25 -- `docs/currentstate/2026-09-25.md:9` -- ## 2026-09-25: T-GPLR-59 D1 adversarial closure; parser and teardown gaps fixed
- 2026-09-25 -- `docs/currentstate/2026-09-25.md:19` -- ## 2026-09-25: T-GPLR-59 D1 same-session MTP frontier completes; proposal precision positive, runtime D-gate FAIL
- 2026-09-25 -- `docs/currentstate/2026-09-25.md:37` -- ## 2026-09-25: T-GPLR-59 D1 authorized MTP handoff arm/null run; pairing gate PASS, D-gate not yet established
- 2026-09-25 -- `docs/currentstate/2026-09-25.md:51` -- ## 2026-09-25: T-GPLR-59 D1 adversarial replay/lifecycle fixes
- 2026-09-25 -- `docs/currentstate/2026-09-25.md:61` -- ## 2026-09-25: T-GPLR-59 D1 runtime wiring reaches host validation; real MTP run still pending
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:9` -- ## 2026-09-24: T-GPLR-59 D1 implementation slice starts (decision recorded 2026-09-25)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:21` -- ## 2026-09-24: Owner closes T-GPLR-53, T-GPLR-56, T-GPLR-58, and T-GPLR-60; T-GPLR-59 remains pending owner calls (de...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:31` -- ## 2026-09-24: T-GPLR-59/60 architecture pass separates engine pool policy, telemetry state, and host replay; behavio...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:45` -- ## 2026-09-24: T-GPLR-59/60 validator adversarial pass fixed duplicate-start, pre-init state, and malformed-number ac...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:57` -- ## 2026-09-24: T-GPLR-60 G0/G1 host learning-curve experiment is COMPLETE; G1 FAILS the equal-row gate and the family...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:73` -- ## 2026-09-24: T-GPLR-59 FOLLOW-UP: the host-only counterfactual 12-row pool replay is IMPLEMENTED and SCORED over re...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:85` -- ## 2026-09-24: T-GPLR-59 prediction pool is IMPLEMENTED and HOST-VALIDATED (policy + instrumentation green; no GPU/mo...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:97` -- ## 2026-09-24: THE ARM-ZERO-ADMISSION QUESTION IS RESOLVED (host-only diagnosis) - a history-only predictor cannot be...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:111` -- ## 2026-09-24: T-GPLR-58 A4 COMPLETED (owner-authorized, one mutex hold, seven legs) - both K=2 arms were capacity-bo...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:123` -- ## 2026-09-24: T-GPLR-58 FRESH K=8 COACT ADMISSION TRACE (owner-authorized, uncommitted)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:131` -- ## 2026-09-24: T-GPLR-58 CACHE-AWARE CANDIDATE COUNTERFACTUAL (host-only, uncommitted)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:139` -- ## 2026-09-24: T-GPLR-58 MODE-2 ADMISSION TELEMETRY (host-only, uncommitted)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:147` -- ## 2026-09-24: T-GPLR-07 CHECKER CLI SELF-TEST HARDENED (host-only)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:153` -- ## 2026-09-24: T-GPLR-07 `--tg-text` CHECKER MODE BEHAVIOR VERIFIED (host-only)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:159` -- ## 2026-09-24: T-GPLR-58 A2 FILL-ID REPLAY IS REPAIRED - a fresh null K=8 leg passes check-telemetry with ZERO violat...
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:171` -- ## 2026-09-24: T-GPLR-07 CUDA SEMANTIC-QA SMOKE TIER PASS (50/512)
- 2026-09-24 -- `docs/currentstate/2026-09-24.md:185` -- ## 2026-09-24 — Nested engine rebuild and host-suite provenance recheck
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:9` -- ## 2026-09-23: T-GPLR-58 K-SCALING PROBE (owner-directed) - retention scales cleanly to K=12, the null PLACED ~200 fi...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:25` -- ## 2026-09-23: T-GPLR-58 A3 RAN (one mutex hold, five legs, GPU-clean) - the arm PROPOSED on 10,178 of 10,240 layer-s...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:39` -- ## 2026-09-23: T-GPLR-58 MODE 2 LANDED - the engine CONSUMES the A1 proposer, and the fill-room bound says an arm's c...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:86` -- ## 2026-09-23: THE T-GPLR-58 WORK IS COMMITTED AS A TASK-SPLIT SERIES (four outer + two nested commits, no GPU time s...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:98` -- ## 2026-09-23: T-GPLR-58 A1 IMPLEMENTED - the predictor POLICY (four families, the bar and vote operators, the count-...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:112` -- ## 2026-09-23: M7 RESTATED (Planner call, T-GPLR-58) - the vacuous "lead median > 0" band becomes an invariant on the...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:126` -- ## 2026-09-23: T-GPLR-58 A2 IMPLEMENTED - the TG predictor VERDICT SURFACE is in the tree (host-validated, refuses wh...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:140` -- ## 2026-09-23: T-GPLR-58 A0 IMPLEMENTED - the speculative TG expert-row placement MECHANISM is in the tree, default-o...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:158` -- ## 2026-09-23: T-GPLR-58 A0/A1 SETTLED - MECHANISM IS A0 AND LIVES IN THE STREAMING MODULE, POLICY IS A1 AS PURE FUNC...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:172` -- ## 2026-09-23: T-GPLR-58 BOUNDING PASS - the incumbent row cache's uncovered margin is MEASURED at eight recorded sha...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:197` -- ## 2026-09-23: THE TWO PREDICTOR SOURCE CLASSES T-GPLR-58 EXCLUDES ARE NOW DRAFTED AS T-GPLR-59 - an MTP draft-previe...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:213` -- ## 2026-09-23: THE TG EXPERT-PREDICTION WIRE-IN IS DRAFTED AS T-GPLR-58 AND MARKED READY THE SAME DAY - three pre-reg...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:229` -- ## 2026-09-23: THE T-53 SETTLEMENT AND THE T-56 A3 WORK ARE COMMITTED AS A TASK-SPLIT SERIES (five outer + one nested...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:248` -- ## 2026-09-23: T-GPLR-56 A3 IS IMPLEMENTED AND HOST-VALIDATED (`p1-pace` behind three DEFAULT-OFF flags, its own refu...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:264` -- ## 2026-09-23: THE UNCOMMITTED T-GPLR-53/54/56 WORK IS COMMITTED AS A TASK-SPLIT SERIES (seven outer + five nested co...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:280` -- ## 2026-09-23: T-GPLR-53 BAND-QUANTITY ESCALATION SETTLED (owner directive, OPTION A) AND THE REPORTING FIX LANDED -...
- 2026-09-23 -- `docs/currentstate/2026-09-23.md:312` -- ## 2026-09-23: PREDICTOR SCREENING PORTED FROM THE COLIBRI FUNNEL - coact/markov-1 beat the history baseline at every...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:9` -- ## 2026-09-22: THE A1-LADDER SESSIONS RAN AFTER ALL (both rungs PASS, both CERTIFY, both MEASURED) - the card was hel...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:34` -- ## 2026-09-22: THE PER-PART TG TIMINGS WERE ALREADY LANDED (F3/H8, 2026-09-14) - WHAT WAS MISSING IS THAT A TIMING NE...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:46` -- ## 2026-09-22: PROVENANCE IS NOW BINDING AT THE LAUNCHERS, AND ONE REAL ARM IS CERTIFIED (scope corrected the same da...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:101` -- ## 2026-09-22: ARM PROVENANCE LANDED - every verdict now names the SOURCE TREE and the BINARY it describes, and a che...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:176` -- ## 2026-09-22: TWO OWNER-AUTHORIZED ONE-LINERS APPLIED - stale one-slot CTest tests fixed (--pp-async-feed 0), recon...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:193` -- ## 2026-09-22: T-GPLR-52's forwarded recon off-by-one join concern - DECIDED CLOSED, host-only from the 126 recorded...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:221` -- ## 2026-09-22: T-GPLR-56 OWNER CALLS DECIDED - first-token PP latency in scope, bounded (A3 decides on one 4k hold);...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:242` -- ## 2026-09-22: T-GPLR-16 CLOSED - the `--handoff-partial` MIN_BATCH=1 override is REVERTED out of the tree, with a ho...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:308` -- ## 2026-09-22: ESCALATION - the >= 70% band is stated on a quantity that cannot reach it (host-only, no GPU, no arm r...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:425` -- ## 2026-09-22: DRIVER CAPTURE PRODUCTION-PROVEN (one mutex hold, 13/13 arms PASS) - and the captured lines expose a p...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:453` -- ## 2026-09-22: THE SESSION LAUNCHER NOW CAPTURES THE ARM DRIVERS' OUTPUT AND EXIT CODES (host-only change; no GPU spent)
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:478` -- ## 2026-09-22: THE PASS-1 ONE-OFF IS ATTRIBUTED (host-only, from recorded traces) - it is the driver-side commit of t...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:508` -- ## 2026-09-22: T-GPLR-53 A4 DOSE CURVE - PASS 13/13: absorption of the pass-1 excess is ~1:1 in the wait, the wall is...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:558` -- ## 2026-09-22: T-GPLR-53 A4 FALSIFIER RUN - PASS 8/8, and the no-touch delay CLEARS 93.2% of the pass-1 excess, so th...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:611` -- ## 2026-09-22: T-GPLR-53 A4 CLOSES - PASS 6/6 in one mutex hold, and the modality evidence points at HOST PAGE RESIDE...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:648` -- ## 2026-09-22: T-GPLR-53 A3 CLOSES ON REAL ARM VERDICTS - PASS 2/2, CHUNK0_SPIKE reproduced on a second run; A4 half...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:698` -- ## 2026-09-22: T-GPLR-54 A3 EXECUTED (owner-invoked) - VERDICT: BENIGN (H1); the u4096 escape at index 41 was a near-...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:761` -- ## 2026-09-22: T-GPLR-53 A3 MECHANISM VERDICT PRODUCED - the pass-1 embed excess is a chunk-0 spike (pre-registered m...
- 2026-09-22 -- `docs/currentstate/2026-09-22.md:818` -- ## 2026-09-22: T-GPLR-54 CLOSED ON THE ARM RULE (owner decision) - id-equivalence at u >= 4096 is unavailable by desi...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:9` -- ## 2026-09-21: T-GPLR-54 A2 EXECUTED - the default-off --tg-logit-margin readout lands (runner + engine, host-validat...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:39` -- ## 2026-09-21: +12 x2 RECON-JOIN ANOMALY RESOLVED - not a second join defect: the checker's positional pairing mispai...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:83` -- ## 2026-09-21: T-GPLR-54 A1 EXECUTED - the ID-CHECK is now a checked tool; D = 41 / K = 7 reproduced host-only, no GPU
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:105` -- ## 2026-09-21: T-GPLR-54 DRAFTED - the last ownerless correctness observation is now a task, and its shape is measura...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:131` -- ## 2026-09-21: T-GPLR-53 A2 EXECUTED (host-validated, no GPU) - the recon-gated `embed_chunk` event lands, the recon...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:184` -- ## 2026-09-21: PLAN CLOSE-OUT SWEEP - T-GPLR-52 closed (AC5), T-GPLR-37 status fixed, resplit applied; ONLY T-GPLR-53...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:243` -- ## 2026-09-21: T-GPLR-53 A1 EXECUTED (host-only, no GPU) - F1-F6 reproduced as CHECKED expectations; the pass-1 gap i...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:279` -- ## 2026-09-21: T-GPLR-53 DRAFTED - the sweep's pass-1 "unattributed" gap is attributed host-only (it is the EMBED STE...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:334` -- ## 2026-09-21: Anomaly sweep, host-only on existing traces: 4 of 5 explained, 1 open unknown (pass-1 exposed gap); re...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:376` -- ## 2026-09-21: T-GPLR-52 AC3 - true-DMA timing was possible all along (recon copy_gpu_ms); DMA = 1% of host cost
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:391` -- ## 2026-09-21: T-GPLR-52 A3 - H3 falsified with in-trace proof: pinned WAS engaged, no fallback ever existed (AC4 dis...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:409` -- ## 2026-09-21: T-GPLR-52 A2 - H2 (file-cache eviction) FALSIFIED as a mechanism; the 50k falsifier is the discriminat...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:425` -- ## 2026-09-21: T-GPLR-52 A1 (host-only) - H1 CONFIRMED: the copy wait IS overlapped compute at long context
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:446` -- ## 2026-09-21: T-GPLR-51 CLOSED - 3 slots measured-optimal (S2), default combo confirmed, plan re-split
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:464` -- ## 2026-09-21: Copy-tail follow-up - "40 vs 160" premise CORRECTED; the real anomaly is per-copy cost scaling with pr...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:488` -- ## 2026-09-21: T-GPLR-51 A3/A4 verdict - copy-tail question CLOSED (structural); async-feed redundant at 200k; 4 slot...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:511` -- ## 2026-09-21: T-GPLR-51 A2 verdict - copy-source stays mmap at 200k; the 10.9 s copy tail exists in BOTH sources (st...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:530` -- ## 2026-09-21: Transfer-bucket decomposition at 200k + the 4k copy-source discriminator (T-GPLR-51 draft evidence)
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:561` -- ## 2026-09-21: Clean PP/TG reference numbers for the current binary (T-GPLR-50 session)
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:600` -- ## 2026-09-21: GPU-mutex wrapper hardened against non-conforming compute processes
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:620` -- ## 2026-09-21: T-GPLR-50 CLOSED - f16 rebuild default flipped ON; the +67.4 s 200k pass-2 excess RE-ATTRIBUTED to GPU...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:657` -- ## 2026-09-21: T-GPLR-50 DRAFTED — the survey answer to "highest-value remaining PP lever": flip the T-48 f16 rebuild...
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:675` -- ## 2026-09-21: BOTH PARKED ODDITIES RESOLVED FROM RECORDED TRACES (no new GPU arms)
- 2026-09-21 -- `docs/currentstate/2026-09-21.md:755` -- ## 2026-09-21: Per-launch patch lever RE-MEASURED at 200k -> CLOSE the idea; a NEW finding takes its place
- 2026-09-20 -- `docs/currentstate/2026-09-20.md:9` -- ## 2026-09-20: T-GPLR-39 CLOSED — route telemetry defaults OFF (owner decision)
- 2026-09-20 -- `docs/currentstate/2026-09-20.md:26` -- ## 2026-09-20: T-GPLR-47 CLOSED — the per-pass rebuild now swaps in a FRESH context per graph set (“swap-in-slot”), w...
- 2026-09-20 -- `docs/currentstate/2026-09-20.md:28` -- ## 2026-09-20: THE PLAN FILE WAS SPLIT — 18 closed task blocks moved VERBATIM out of
- 2026-09-20 -- `docs/currentstate/2026-09-20.md:46` -- ## 2026-09-20: T-GPLR-37 CLOSED — mmap-registered staging is now the DEFAULT copy source (owner decision: "Promote to...
- 2026-09-19 -- `docs/currentstate/2026-09-19.md:9` -- ## 2026-09-19: T-GPLR-48 CLOSED — `--pp-prebuild-per-pass` is now **KV-conditional**, so the default f16 path stops p...
- 2026-09-19 -- `docs/currentstate/2026-09-19.md:11` -- ## 2026-09-19: THE 4k ECHO IS A **pw x REBUILD** INTERACTION, ATTRIBUTED — one two-arm follow-up closes the last open...
- 2026-09-19 -- `docs/currentstate/2026-09-19.md:13` -- ## 2026-09-19: T-GPLR-49 CLOSED — the latent `--no-pw-probe` act-slab offset defect is REAL, its pre-registered refus...
- 2026-09-19 -- `docs/currentstate/2026-09-19.md:15` -- ## 2026-09-19: T-GPLR-45 CLOSED — PERFORMANCE VERDICT MET: the per-launch patch restore now copies only the tensors a...
- 2026-09-19 -- `docs/currentstate/2026-09-19.md:17` -- ## 2026-09-19: T-GPLR-46 CLOSED — the pass-2 compute regression is REAL and REBUILD-LINKED, and the same rebuild degr...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:9` -- ## 2026-09-18: T-GPLR-44 CLOSED -- REVERTED-BY-FALSIFIER (the instrument is kept, the behavior is not). **The task:**...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:11` -- ## 2026-09-18: THREE TASKS CLOSED (T-GPLR-35, T-GPLR-36, T-GPLR-38) — and the answer to "what default does closing th...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:13` -- ## 2026-09-18: A LAUNCH-TIME **START MARKER** CLOSES THE LAST AMBIGUITY IN ARM OBSERVATION — a missing verdict can no...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:15` -- ## 2026-09-18: THE BARE FLAG'S DEFAULT RETENTION DEPTH IS NOW **K=8** (owner-directed, superseding the K=4 default of...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:17` -- ## 2026-09-18: DETACHED-JOB CONVENTIONS CONSOLIDATED into one section of `docs/AGENT_ENVIRONMENT.md` ("Detached runs...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:19` -- ## 2026-09-18: EVERY GPU ARM NOW LEAVES A MACHINE-READABLE VERDICT plus a `.done` SENTINEL, so a launched run does no...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:21` -- ## 2026-09-18: THE NON-ECHOING VEHICLE IS NOW THE DEFAULT for row-cache rungs (one driver, `-Vehicle honest|echoed`),...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:23` -- ## 2026-09-18: ROW-CACHE A/B ON THE HONEST STREAM — the K=8 depth decision's echo caveat is now MEASURED. K=6 and K=8...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:25` -- ## 2026-09-18: T-GPLR-26 CLOSED — ADVISORY (branch B, documents only, ZERO source changes), and the TG retention dept...
- 2026-09-18 -- `docs/currentstate/2026-09-18.md:27` -- ## 2026-09-18: HOST-MEMORY INSTRUMENT — the T-GPLR-36 open question now HAS an instrument (vendored tree COMMITTED c5...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:9` -- ## 2026-09-17: T-GPLR-43 QA-SMOKE GATE — the audit's CORRECTNESS gap on the --pp-repeat default rebuild is CLOSED; bo...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:11` -- ## 2026-09-17: T-GPLR-43 UNIFICATION — act_bind emission has a SINGLE owner. Follow-up to the audit's design finding...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:13` -- ## 2026-09-17: T-GPLR-43 FIX — the q8 144k pass-2 crash is FIXED and the regression is CLEAN. Mechanism (proven by th...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:15` -- ## 2026-09-17: T-GPLR-42 Q4 DISCRIMINATOR — the pass-2 crash is QUANTIZED-KV-GENERAL, not q8-specific. Both arms (B1...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:17` -- ## 2026-09-17: T-GPLR-41 MCTX RECON — the stale-reference hypothesis is REFUTED (falsifier fired); crash reproduced u...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:19` -- ## 2026-09-17: T-GPLR-40 BISECT — the 144k pass-2 crash is ATTRIBUTED: q8 KV cache + long context + the pass-2 clear/...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:21` -- ## 2026-09-17: T-GPLR-38 ADDENDUM — 144K UBOATCH A/B: INCOMPLETE, and it found a real pass-2 crash. **Requested as a...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:23` -- ## 2026-09-17: T-GPLR-39 RECON — the ubatch-size residue is ATTRIBUTED: `tel_route` (route telemetry: device→host pro...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:25` -- ## 2026-09-17: T-GPLR-38 UBATCH SWEEP (4k anchor, 3 slots, ron mode) — the prediction inverted at 2048: warm wall 231...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:27` -- ## 2026-09-17 (addendum 3): T-GPLR-37 AT THE CROWDED SHAPE — the 4k win does NOT transfer: 200k/262k-F16-KV warm wall...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:29` -- ## 2026-09-17 (addendum 2): T-GPLR-37 CHURN-FREE VARIANT — CHURN CONFIRMED AS THE CAUSE, SETUP INFLATION GONE, AND A...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:31` -- ## 2026-09-17: T-GPLR-37 — DIRECT MMAP-SOURCED STAGING: FEASIBLE, HOP REMOVED FROM THE ACCOUNTING, BUT A REGRESSION A...
- 2026-09-17 -- `docs/currentstate/2026-09-17.md:33` -- ## 2026-09-17: T-GPLR-36 — THE LONG-CONTEXT RESIDUAL IS PER-LAUNCH HOST WORK, NOT PER-GRAPH SETUP. Ledger run: one 20...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:9` -- ## 2026-09-16: T-GPLR-35 ADDENDUM 2 — DEPTH DOES NOT PAY AT LONG CONTEXT, AND THE BUDGET REFUSAL IS VERIFIED AT THAT...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:11` -- ## 2026-09-16: T-GPLR-35 PIPELINE REGRESSION GATE ADDED — `tools/check-gplr35-pipeline.py` fails when PP staging sile...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:13` -- ## 2026-09-16: T-GPLR-35 ADDENDUM 1 — 3 SLOTS CONVERT THE SERIAL WAITS INTO OVERLAP: warm 4k layer-major 3377.8 -> 28...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:15` -- ## 2026-09-16: T-GPLR-35 RECON RUN — the coupling is NOT the staging/compute stream topology. Falsifier class reached...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:17` -- ## 2026-09-16: T-GPLR-35 V1 IMPLEMENTED + BUILT (A/B armed, NOT yet run) — staging DMAs can now run on an auxiliary s...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:19` -- ## 2026-09-16: T-GPLR-34 MID-KERNEL PROBE — WDDM cross-engine serialization FALSIFIED (a second stream overlaps the c...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:20` -- ## 2026-09-16: SUBMIT_BATCH GAP CLOSED — the field now reports the ACTUALLY-TAKEN branch threaded from stage_members...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:22` -- ## 2026-09-16: T-GPLR-33 WHY-ANALYSIS — the decisive pair: warm GPU-side ready_latency 74.1 ms (3 submissions) vs 74....
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:24` -- ## 2026-09-16: T-GPLR-33 DONE (A/B RUN) — one-submission-per-bank FALSIFIED: windows identical with 1 vs 3 submission...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:63` -- ## 2026-09-16: T-GPLR-32 FOLLOW-UP RESEARCH — the ~82 ms window decomposes as 3 members x (10.5 ms pure transfer + ~1...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:116` -- ## 2026-09-16: T-GPLR-32 DONE (A/B RUN) — serialize-the-worker-DMAs FALSIFIED as a wall lever: warm walls identical (...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:160` -- ## 2026-09-16: WDDM DMA-INFLATION RESEARCH PASS — asyncEngineCount=5 measured on this 5090 Laptop; NVIDIA's engine mo...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:200` -- ## 2026-09-16: BASELINE COMPARISON (out/llamacpp-baseline.log -- a local artifact under the gitignored out/ tree, not...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:233` -- ## 2026-09-16: CROWDED-VRAM CHARTER RUNG MEASURED — 200k-token PP, F16 KV, the af=1+pinned 2-slot optimum: 1,363 tok/...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:263` -- ## 2026-09-16: GOAL CHARTER RECORDED (owner decision) — PP speed first, VRAM second, speed must hold under crowded VR...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:277` -- ## 2026-09-16: DEFAULT PROFILE v3 GATED — af=1 + pinned defaults passed the qa-smoke semantic gate (512 rung); defaul...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:313` -- ## 2026-09-16: T-GPLR-31 DONE — locked-RAM probe: the driver grants 44.72 GiB of pinned memory incrementally (106 x 4...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:337` -- ## 2026-09-16: T-GPLR-30 DONE — partial persistent pinned pool (--pp-pool N[,fixed|lru]): hit mechanics perfect, warm...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:391` -- ## 2026-09-16: WAIT SIGNATURE IDENTIFIED — it is the per-layer deficit max(0, effective_DMA_window − compute_window),...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:422` -- ## 2026-09-16: PASS-2 ANOMALY REPLICATED AND RESOLVED — two separate observations, opposite verdicts: the 4.10 s wall...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:450` -- ## 2026-09-16: T-GPLR-29 ADDENDUM — af overlap A/B with pinned: overlap converts ONLY with pinned staging (warm 4k PP...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:478` -- ## 2026-09-16: T-GPLR-29 DONE — pinned-pool probe + end-to-end pinned arm: DMA ceiling 14.4 GB/s is ~90% of the real...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:523` -- ## 2026-09-16: ONLINE RESEARCH PASS — first-touch + staging cost layers identified from primary sources (pageable-mem...
- 2026-09-16 -- `docs/currentstate/2026-09-15..2026-09-16.md:597` -- ## 2026-09-16: T-GPLR-28 DONE — in-process --pp-repeat 3 proves first-touch is pay-once-per-process (warm state stabl...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:651` -- ## 2026-09-15: 4k LAYER-MAJOR PP MEASURED — 2.3x PP-wall gain vs fresh adjacent batch-major baseline (prediction of 3...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:715` -- ## 2026-09-15: DEFAULT PP SCHEDULE VERIFIED as layer-major (T-GPLR-12, no code flip needed); stale selector comment f...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:759` -- ## 2026-09-15: AGENTS.md governance updated — MoE-Qwen focus is a measurement choice (other models on owner request),...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:798` -- ## 2026-09-15: Pending owner decisions (item 4 gate-eligibility, item 6 second-architecture) recorded as plan task dr...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:821` -- ## 2026-09-15: T-GPLR-25 residue FIXED — async_feed_submit wired to the worker (batch-major + layer-major), async_fee...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:879` -- ## 2026-09-15: T-GPLR-25 COMPLETED: 4k pair run — af delta +3.0% is AT the noise floor, no performance claim; batch-m...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:931` -- ## 2026-09-15: T-GPLR-25 paired A/B, 512 rung COMPLETE: NO af delta at 1 chunk/layer; earlier prototype wall comparis...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:971` -- ## 2026-09-15: OWNER DECISION IMPLEMENTED — default --layer-slots 2, auto-3 on fit, loud refusal below 2; T-GPLR-25 a...
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:1069` -- ## 2026-09-15: documented online research pass on H-C/H-D (no GPU runs, no code changes)
- 2026-09-15 -- `docs/currentstate/2026-09-15..2026-09-16.md:1103` -- ## 2026-09-15 (latest): H-A fixed and proven — row-set TG phase_summary carries transfer/wait/bytes; first real F3 pa...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:9` -- ## 2026-09-14 (latest): full-length validation sweep 4k/50k/200k over the pruned tree — 4k and 50k PASS; 200k abort r...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:76` -- ## 2026-09-14: DECISION BRIEF (item 6, owner call) — which second architecture should be the first to pass the generi...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:228` -- ## 2026-09-14: T-GPLR-22 PRE-TIMED ABORT — every abort in that family now leaves a trace the checker reads as a refus...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:327` -- ## 2026-09-14: DECISION BRIEF (item 4, owner call) — should gate eligibility BIND provisioning or stay ADVISORY?
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:462` -- ## 2026-09-14: T-GPLR-22 PRE-SINK — a refusal raised before `ls.init` now leaves a recoverable `run_refused` trace in...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:533` -- ## 2026-09-14: Mission item 5 — the three opt-in TG modes measured at 4k and 50k; run `handoff` where it fits, `reten...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:640` -- ## 2026-09-14: T-GPLR-22 — a pre-work refusal is now a structured, recoverable record (`run_refused`) instead of a pa...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:738` -- ## 2026-09-14: the non-echoing re-probe RAN — K=4 does NOT clear gate 9 on it (0.5355 vs 0.6284 echoed); the first cl...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:843` -- ## 2026-09-14: F3/H8 implemented — TG row-set `layer_compute` carries real per-part timings, so the transfer term now...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:913` -- ## 2026-09-14: T-GPLR-07 ladder rungs 4k and 50k PASS (owner-invoked, one at a time) — and the 50k rung exposed a HAR...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:987` -- ## 2026-09-14 (later): the TOTAL residency cap is now provable from JSONL — `byte_account` carries the provisioned ro...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1083` -- ## 2026-09-14: `--residency-budget-mib` is now a TOTAL residency cap — the row cache must fit under it, and a NAMED d...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1190` -- ## 2026-09-14: the capacity preflight and the row-cache fit now share ONE VRAM budget model — two parallel arithmetic...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1266` -- ## 2026-09-14: the per-layer row cache's DEFAULT depth is now fitted to the device — K=4 where the device affords it...
- 2026-09-14 -- `docs/currentstate/2026-09-14.md:1355` -- ## 2026-09-14: K=4 set as the per-layer row cache's provisional DEFAULT depth (owner-directed) — bare-flag arm reprod...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:9` -- ## 2026-09-13: T-GPLR-19 deferred rungs RUN at K=2, K=4, K=6, K=8 — every rung passes, the runtime reproduces the off...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:105` -- ## 2026-09-13: Vendored-fork commit convention DECIDED and applied — the vendored backlog is committed, so the T-GPLR...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:200` -- ## 2026-09-13: T-GPLR-19 DONE — the four-arm K=1 A/B gate RAN and all six acceptance criteria passed; F2-M1 is now a...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:301` -- ## 2026-09-13: Retention policy RESOLVED — the implementation moved to the row-capacity LRU the gate already measured...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:402` -- ## 2026-09-13: T-GPLR-19 drafted (F2-M1 per-layer row residency at K=1) and its A/B driver's home DECIDED — tracked `...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:438` -- ## 2026-09-13: The (b)-versus-K curve published as a standalone artifact — `docs/plots/gplr15-b-vs-k.html`, generated...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:562` -- ## 2026-09-13: T-GPLR-15 CLOSED DONE-falsified-by-evidence — the escalated retention choice resolved in favour of the...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:612` -- ## 2026-09-13: Closing-hash and tracked-status bookkeeping after ba1f31e
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:641` -- ## 2026-09-13: T-GPLR-18 pinned row-set staging CLOSED DONE-falsified-by-evidence — the 512 A/B shows pinned only REL...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:713` -- ## 2026-09-13: T-GPLR-15 Phase A probe RUN — the gate FAILS inside the SDD's reference retention band (0.3088 / 0.434...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:788` -- ## 2026-09-13: T8 oracle retention ceiling added to the analyzer — on the ladder samples NO policy can reach the 0.6...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:822` -- ## 2026-09-13: Both remaining untracked docs brought under version control — the model-run testing standard and the 2...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:869` -- ## 2026-09-13: T-GPLR-15 Phase A analyzer implemented — first real values of the SDD gate-9 statistic (b), and K=1 re...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:941` -- ## 2026-09-13: Rowset telemetry schema draft brought into version control (tracked, not inlined) — the T-GPLR-15 cont...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:975` -- ## 2026-09-13: Gate 9 decision rule now single-owned by the SDD (T-GPLR-15 metric corrected); cited pilot numbers car...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1017` -- ## 2026-09-13: SDD amended with the F2 per-layer row cache (`moe-row-cache`) residency class — realized reuse and the...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1057` -- ## 2026-09-13: T-GPLR-18 (F1 pinned row-set staging) added to the plan as READY with a 512 A/B gate — counter-evidenc...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1106` -- ## 2026-09-13: T-GPLR-15 contract rewritten gate-measurement-first (READY) — Phase A measures the 0.6 hit-rate gate w...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1140` -- ## 2026-09-13: T-GPLR-13 marked DONE (owner review) — reactive expert row-set residency closed; residual risks and op...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1178` -- ## 2026-09-13: TG row-set staging analysis (section 6 of docs/gplr13-bottlenecks-hypotheses.md) — ready-latency is ~5...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1205` -- ## 2026-09-13: T-GPLR-13 test ladder executed — all five sizes PASS (50/512/4k/200k row-set + qa-smoke-512 regression...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1264` -- ## 2026-09-13: T-GPLR-13 implementation COMPLETE — moe-row-set TG path runs end-to-end, checker + negative controls g...
- 2026-09-13 -- `docs/currentstate/2026-09-12..2026-09-13.md:1330` -- ## 2026-09-13: Memory-validation probe (T-MEMPROBE-01, owner request) — no RAM/VRAM overcommit observed; shared GPU m...
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1392` -- ## 2026-09-12: T-GPLR-13 gate (b) DECIDED — reactive row-set is the priority TG path; T-GPLR-16 deprioritized with op...
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1452` -- ## 2026-09-12: Drafted model-run testing standard
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1476` -- ## 2026-09-12: moe-row-set SDD amendment written (owner-directed; T-GPLR-13 gate (a) resolved)
- 2026-09-12 -- `docs/currentstate/2026-09-12..2026-09-13.md:1506` -- ## 2026-09-12: T-GPLR-16 MIN_BATCH=1 wedge fully characterized — device-side stream deadlock, containment attempt rev...
- 2026-09-11 -- `docs/currentstate/2026-08-20..2026-09-11.md:9` -- ## 2026-09-11: T-GPLR-16 root cause — mixed-arithmetic PP→TG switch, not corruption
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:60` -- ## 2026-09-10: T-GPLR-17 REJECTED by owner and fully reverted (two-process hybrid bridge)
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:88` -- ## 2026-09-10: T-GPLR-17 drafted READY — two-process hybrid bridge (streaming PP -> state file -> CPU-expert llama-se...
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:140` -- ## 2026-09-10: Streaming-PP + stock-TG hybrid (--handoff) at 50K — loud preflight refusal on today's VRAM baseline
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:170` -- ## 2026-09-10: Streaming-TG slowness profiled against stock baselines (host-only analysis, no GPU run)
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:244` -- ## 2026-09-10: T-GPLR-14 numbering collision resolved (mixed-placement TG mode renumbered to T-GPLR-16)
- 2026-09-10 -- `docs/currentstate/2026-08-20..2026-09-11.md:348` -- ## 2026-09-10: Owner probe — MoE 50K streaming PP + streaming TG under an 8 GB VRAM ceiling
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:261` -- ## 2026-09-09: MoE TG row-set program drafted/renumbered; T-GPLR-14 DONE (route-overlap instrument)
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:311` -- ## 2026-09-09: T-GPLR-12 DONE — default profile v2 (bf16 kept on MoE by owner decision)
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:376` -- ## 2026-09-09: NCMOE 262K probe — stock TG at full context, 16 GB pipeline evidence
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:512` -- ## 2026-09-09: Planner review -- T-GPLR-08 ready, T-GPLR-09 held
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:544` -- ## 2026-09-09: T-GPLR-06/07 CLOSED — T-GPLR-08 and T-GPLR-10 may start
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:581` -- ## 2026-09-09: T-GPLR-11 DONE — batch-major tail-ubatch gap closed
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:614` -- ## 2026-09-09: T-GPLR-08 implemented and probed — prefetch hypothesis FALSIFIED (instrumentation result, task closed)
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:669` -- ## 2026-09-09: T-GPLR-10 DONE — MoE handoff (streaming PP -> stock llama.cpp TG hybrid)
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:763` -- ## 2026-09-09: T-GPLR-09 PARKED — owner decision, plan updated
- 2026-09-09 -- `docs/currentstate/2026-08-20..2026-09-11.md:802` -- ## 2026-09-09: T-GPLR-10 drafted — MoE handoff (streaming PP + stock llama.cpp TG)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:406` -- ## 2026-09-07: T-GPLR-07 smoke tier wired into tools/run-tests.ps1 (host-validated)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:460` -- ## 2026-09-07/09: qa-smoke gate live runs (50/512 PASS; 15/30 removed from the ladder on 2026-09-09)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:841` -- ## 2026-09-07: T-GPLR-07 runner flags --prompt-file / --tg-text landed (host-validated)
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:878` -- ## 2026-09-07: --mtp + --pp-no-logits n_outputs_max clamp fixed
- 2026-09-07 -- `docs/currentstate/2026-08-20..2026-09-11.md:914` -- ## 2026-09-07: Host I/O Baseline and Tier Direction (owner decision)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:938` -- ## T-GPLR-05 Complete: Capability Registry + Regression Closure — Plan Done (2026-09-05)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1252` -- ## T-GPLR-04 Complete: 262K Q8 Out-of-Core Capacity Proof (2026-09-05)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1303` -- ## T-GPLR-03 Complete: 512-Token MoE Layer-Major Prefill + Parity Closure (2026-09-05)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1351` -- ## T-GPLR-02 Complete: MoE Expert-Bank Package Proof (2026-09-04)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1382` -- ## T-GPLR-01 Complete: Qwen Package Adapter Extracted (2026-09-04)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1419` -- ## T-GPLR-00 Complete: MoE Artifact Inventory (2026-09-04)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1462` -- ## Active Generic Prefill Layer Residency Direction (2026-09-04)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1527` -- ## Arbitrary-Length Layer-Major PP (done 2026-08-23)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1574` -- ## 262K Attempt and Per-Width Graph Plan (2026-08-23)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1613` -- ## T-Q38PW-00: Patchability Probe (done 2026-08-23)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1699` -- ## T-Q38PW-01: Per-Width Graph Sets + Staged Input Buffer (done 2026-08-23)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1804` -- ## T-Q38PW-02: 262K Stretch with bf16 Slabs (done 2026-08-23)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1904` -- ## Per-layer slot bundles for dynamic-quant models (done 2026-08-23/24)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:1938` -- ## T-Q38FA-01: per-chunk K/V window on the implicit-causal FA path (done 2026-08-24)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2043` -- ## T-Q38FA-02: resident control verdict (done 2026-08-24)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2078` -- ## T-Q38FA-04: patch-walk cache (done 2026-08-24)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2138` -- ## T-Q38TG-00: Hybrid TG Feasibility Probe (done 2026-08-24 — NO-GO)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2215` -- ## T-Q38HG-00/01/02: PP→TG Handoff (done 2026-08-24)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2382` -- ## 20K Layer-Major Evidence (done 2026-08-20)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2413` -- ## T-Q38PERF-01: copy timing split (done 2026-08-20)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2426` -- ## T-Q38PERF-02: package-class slot allocation (done 2026-08-20)
- 2026-08-20 -- `docs/currentstate/2026-08-20..2026-09-11.md:2440` -- ## T-Q38PERF-03: pinned-host package staging (done 2026-08-20)
- 2026-08-18 -- `docs/currentstate/2026-08-18-and-earlier.md:9` -- ## Completed

## Index rules

Shard entry text is frozen. New entries go in the live Recent entries
block; the next resplit moves old entries and regenerates this index.

## 2026-10-10 (optimizaionV1 D7 PP sweep, 27 arms, one mutex hold, all arms uncertified by provenance): the FA causal-GQA opt-in gives +13% to +17% steady prefill at 50k with identical VRAM; job-sized ctx 51,456 saves 2.9 to 4.0 GB with no PP cost. Best measured 8 GB with headroom: ubatch 4096 + FA at job ctx (3,693 tok/s, 5,724 MiB peak). Best 8 GB keeping ctx 262k: ubatch 1024 + FA (3,310 tok/s, 7,750 MiB). Best 16 GB: ubatch 4096 + 3 slots + f16 KV + FA (3,820 tok/s, 12,460 MiB). The plan's expected 16 GB figure of about 4,800 tok/s is not reproduced. 22 pinned arms carry a gate FAIL whose only reason is the driver rule that accepts mmap-registered only (run-lm-ubatch-sweep.ps1 line 246). Replicate spreads 0.02% to 0.80%. Anchor arm on the ab16 binary runs 4.1% under ab16 (cross-session drift). FA answer path not tested: the qa-smoke suite is batch-major. Details and corrected expectations: docs/plans/optimizaionV1-pp-sweep-results.md. Nothing committed.

## Known gaps / risks (current)

- **The stock/LM ratio decays with prompt length and the mechanism is UNPROVEN.**
  1.943x -> 1.651x -> 1.533x at 50k / 144,001 / 200,000 prompt tokens. The
  explanation (stock copies only routed experts, improving as the route union
  saturates) has **no route telemetry at 144k/200k**. Do not quote 1.943x as a
  general multiplier; it is the 50k figure.
- **LM token generation is unusable at the two owner contexts** (short 1.93 GiB
  at ctx 262,144, 5.34 GiB at ctx 512,000) and 0.640-0.741x stock where a
  partial handoff does fit. Any LM throughput claim must state that TG was not
  measured, or state that it lost.
- **Partial-handoff TG token-level fidelity is UNPROVEN**: identical-flag arms
  `lmh20` and `lmh20b` produced divergent sequences at 128 generated tokens
  despite a bit-identical token 0. The LM driver emits no detokenized text, so
  no coherence check is possible.
- **An unexplained context-scaled VRAM term remains.** LM's *unaccounted* VRAM
  grows 3.115 -> 5.507 GiB from ctx 262,144 to 512,000 (~9.8 KiB per context
  token) and ~8 KiB per *prompt* token (7.95 / 8.09 KiB/token measured). The KV
  account is NOT the cause (it is byte-identical to stock's). Cause UNPROVEN.
- **`--handoff-partial N` is inert on its own** (`main.cpp:3454`, the
  `handoff_mode_str = "on";` sits inside a `//` comment). Always pass
  `--handoff` explicitly. Recorded, deliberately unfixed; T-GPLR-67.
- **`alloc_spill` crashes LM at ubatch 2048 / high context**: `finish_build:
  input buffer exhausted and spill allocation failed`, exit `0xC0000409`, with
  a 64 MiB host input buffer. Suspected 32 B alignment in `alloc_spill`
  (`src/llama-layer-streaming.cpp:1173-1202`). NOT re-exercised this session —
  every LM arm ran ubatch 4096. Do not read a ubatch-2048 LM number as a
  measurement without checking this first. T-GPLR-66.
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
