# optimizaionV1: best long-prompt prefill (PP) under 8 GB and 16 GB VRAM ceilings

**Status: PLAN ONLY, revision 2 (2026-10-10).** Nothing created, staged, committed, branched or pushed. The branch `optimizaionV1` does not exist yet.
**Commit:** _(unfilled. No commit authorised.)_
**Owner exclusion:** worktree `b947db99-52c9-4f5f-a98b-6e6d33096bd1` (hybrid-mode WIP) is excluded from every source (owner note, 2026-10-10).

Every number is labelled **measured** (read from an artifact), **expected** (arithmetic on measured quantities) or **unproven**. Rev 1 (inventory and candidate survey) is kept in Appendix A.

---

**Rev 3 (2026-10-10):** the D7 PP sweep ran (27 arms, one mutex hold). Measured results and corrections to section 2.1 are in [optimizaionV1-pp-sweep-results.md](optimizaionV1-pp-sweep-results.md). The 16 GB expectation of about 4,800 tok/s is not reproduced: the best measured 16 GB option is 3,820 tok/s. The 8 GB job-context option (ubatch 4096, FA on, ctx 51,456) measured 3,693 tok/s at 5,724 MiB. The 8 GB option that keeps 262k context (ubatch 1024, FA on) measured 3,310 tok/s at 7,750 MiB. Section 2.1 figures below are rev-2 expectations, kept for the record.

---

## 0. Objective (rev 2, owner direction 2026-10-10)

**Maximise steady-state prefill throughput (PP, tokens per second) for long prompts, at 50,000 tokens or more, under a fixed VRAM ceiling, using layer-major streaming (LM) as the engine.**

- **Primary regime:** 50,000-token prompt. **Secondary:** 144,001 and 200,000 tokens.
- **Ceilings:** 8 GB class = peak ≤ 8,192 MiB. 16 GB class = peak ≤ 16,384 MiB. Peak = WDDM dedicated memory, sampled by the poller.
- **Out of target:** short prompts. At 4k, per-layer compute (34 ms, measured) is about equal to per-layer bank copy (31.4 ms, measured), so the copy is not amortised. At 50k compute is about 10× the copy (0.3–0.37 s per layer). LM only pays when compute per layer is much larger than copy per layer, which matches the observed behaviour.
- **TG** is still reported beside PP. This branch must not regress TG, but PP comes first.

---

## 1. Selection criteria (rev 2)

A candidate must pass every row.

| # | Criterion | Test |
|---|---|---|
| **C-LM** | **Composes with layer-major PP (LM).** | (a) Changes only the LM prefill path, or leaves it untouched. The stock path is unchanged unless the change sits behind an opt-in flag. (b) Keeps the LM invariants in AGENTS.md: slot generation and copy-before-use; a slot is reused only after all its consumers finish; KV and recurrent state stay owned by logical layer; ascending token order within each layer; loud failure on a missing package, a stale generation or a failed event. (c) Measured with LM on, same binary and same session, paired A/B, n ≥ 2. (d) The VRAM ledger closes: bounded window + activation slab + compute buffers + KV + reserve reconcile with the capacity preflight. (e) PCIe budget: bank DMA (18.33 GB per 50k pass, ≈1.27 s at the measured 14.44 GB/s pinned ceiling) plus any new host traffic stays below per-pass compute (≈12.7–15 s at 50k). Consumer-wait share stays ≈ 0 (`tools/check-gplr35-pipeline.py`). |
| C-PP | Raises steady PP at ≥ 50k | Paired A/B. Steady = mean of passes 2–3. |
| C-VRAM | Fits its ceiling at the target prompt length | Peak ≤ 8,192 MiB (8 GB class) or ≤ 16,384 MiB (16 GB class). |
| C-CHARTER | Routing, expert IDs, quantisation bytes and checkpoint metadata stay exact | Numerical drift only with the semantic-QA answer gate and the owner's recorded exception. |
| C-LIVE | Not WIP, not closed by a gate | Checked in §3 and Appendix A. |

**Composition status (C-LM) of every candidate is in §3.**

---

## 2. Best implementation (decision)

### 2.1 Answer

- **Engine:** layer-major PP in the layer-streaming runner, at the **current defaults** (profile v3: `pp_async_feed=1`, `copy_source=pinned`, 2 slots, bf16 activation slab, set 2026-09-16). Stock, the hybrid deferral line (A3) and moe-prefetch are not competitive in this regime (§2.3).
- **Best measured today, 8 GB ceiling, 50k, ctx 262,144 kept:** **ubatch 1024 → 3,088–3,112 tok/s steady, peak 7,778 MiB.** Three sessions (ab16 3,087.5; ab17 3,109.0; ab18 3,111.7) and two binaries agree within 1%. This is the best configuration that fits 8 GB without new code.
- **Best matched comparison, 8 GB ceiling (ab18):** stock `--fit-target 16000` gives 837.1 tok/s at 7,395 MiB. LM is 3.72× at 5% more VRAM. **Caveat:** the stock arm is a single pass, the LM arm is steady state. A steady stock arm is not measured; the ratio is an upper bound until it is.
- **Best expected next, 8 GB ceiling** (each needs the P-items in §3):
  - keep ctx 262,144: ubatch 1024 + FA fix (P1) → ≈ 3,640 tok/s at 7,778 MiB (expected, 3,088 × 1.18).
  - keep ctx 262,144: ubatch 2048 + activation slab on host (P3) + FA fix → ≈ 3,900 tok/s at ≈ 8,020 MiB (expected). Margin ≈ 172 MiB, which is thin.
  - prefill job with ctx sized to the job (≈ 51,456, KV 534 MiB instead of 2,720 MiB; P4): ubatch 4096 + FA fix → ≈ 3,930 tok/s at ≈ 7,612 MiB (expected). Valid only if no later session needs the 262k KV. Compute-buffer change under smaller ctx is unmeasured.
- **Best measured, 16 GB ceiling:** ubatch 4096 + FA fix → 3,929 tok/s at 9,798 MiB (ab16 V1 row; **uncertified fafix build**). Without FA: 3,328 at 9,798 MiB. Ubatch 4096 is the knee: 2048 → 4096 gives +0.5% for +1.39 GB.
- **Best expected, 16 GB ceiling:** ubatch 4096 + 3 slots + FA fix → ≈ 4,800 tok/s at ≈ 10,296 MiB (expected). Basis: doc1 measured 4,079 tok/s at 10.05 GiB (ubatch 4096, 3 slots). That VRAM equals the 2-slot point (9,798 MiB) plus one slot (498 MiB), and doc1 is 22.6% above the 2-slot point (3,328). The cause of that gap is **unexplained**. The sweep (P4-S3) decides it.

### 2.2 The 50k table (steady PP unless marked)

| Config | Ceiling | PP tok/s | Peak MiB | Status |
|---|---|---|---|---|
| Stock `--fit-target 16000` (ab18 s16k) | 8 GB | 837.1 | 7,395.3 | measured, **single pass** |
| LM ubatch 512, 2 slots, mmap-registered (ab16) | 8 GB | 2,608.8 | 7,468.0 | measured |
| **LM ubatch 1024, 2 slots (ab16 / ab17 / ab18)** | **8 GB** | **3,087.5 – 3,111.7** | **7,778.3** | **measured, best that fits today** |
| LM ubatch 2048 (ab16) | over by 219 MiB | 3,312.1 | 8,411.1 | measured |
| LM ubatch 4096 (ab16) | over by 1,406 MiB | 3,327.8 | 9,797.6 | measured |
| LM ubatch 4096 + FA fix, V1 row (ab16) | over | 3,929.4 | 9,797.6 | measured, **uncertified build** |
| LM ubatch 1024 + FA fix | 8 GB | ≈ 3,640 | 7,778 | expected |
| LM ubatch 2048 + act slab on host + FA fix | 8 GB | ≈ 3,900 | ≈ 8,020 | expected, thin margin |
| LM ubatch 4096 + ctx 51,456 + FA fix | 8 GB | ≈ 3,930 | ≈ 7,612 | expected |
| LM ubatch 4096, 3 slots (doc1, campaign build) | 16 GB | 4,079.2 | 10,052–10,056 (10.05 GiB) | measured, campaign build |
| LM ubatch 4096, 3 slots + FA fix | 16 GB | ≈ 4,800 | ≈ 10,296 | expected, unmeasured combination |
| Stock `--fit-target 8000` (doc2, ctx 512k) | 16 GB | 2,228.1 | 15.04 GiB | measured, different ctx and prompt |

Notes: ab16 and ab17/18 are the same ctx (262,144), KV q8, 50,000-token prompt, `--pp-repeat 3`, steady = mean of passes 2–3. ab16 used mmap-registered staging. The current default is pinned (§2.5).

### 2.3 Why LM, and what does not compete in this regime

- **LM vs stock at equal VRAM:** 1.94× at 10.2 GiB (doc1, 50k, ctx 262,144); 1.83× at 15 GiB (doc2, ctx 512k); 3.72× at 8 GiB (ab18, single-pass stock, §2.1 caveat).
- **Stock under an 8 GB budget starves:** at ctx 262,144 `--fit-target 16000` leaves 30 of 42 layers on the GPU with KV partly on CPU (ab6/ab7, measured).
- **moe-prefetch (July, GTX 1070):** 185 tok/s at 3,475 tokens. Different model quant, card and prompt length, and no measurement at ≥ 50k. Not comparable.
- **Hybrid deferral (A3):** PP is the stock path (411–415 tok/s at 511 tokens, against 533–535 stock). Not a long-prompt PP candidate.

### 2.4 Length behaviour (ubatch 4096 campaign values, derived)

| Prompt | LM PP (derived or measured) | Activation slab, bf16 (expected from 8 KB/token) | KV q8 at ctx 262,144 | FA gain (measured) |
|---|---|---|---|---|
| 50,000 | 4,079 (doc1, measured) | 390.6 MiB (measured) | 2,720 MiB | 1.18–1.19× |
| 144,001 | ≈ 2,866 (1.651 × 1,735.35) | ≈ 1,124.8 MiB | 2,720 MiB | 1.342× |
| 200,000 | ≈ 2,373 (1.533 × 1,548.1) | 1,562.5 MiB (measured in the crowded-VRAM rung) | 2,720 MiB | 1.377× |
| 262,144 | not measured | 2,048 MiB | 2,720 MiB | — |

- At 200k with ubatch 512 (F16 KV, 2 slots, af=1, pinned), LM measured **1,363 tok/s**. Against the ubatch-4096 derived ≈ 2,373, the ubatch gap is about 1.7×. **Ubatch is the largest long-prompt lever after FA.**
- Activation slab grows with prompt length, not with ubatch. Beyond about 100k it, rather than KV, decides whether 8 GB fits, which is why P3 is mandatory for 8 GB at ≥ 144k.

### 2.5 Conflicts and gaps in the evidence (must be closed before adoption)

1. **doc1 vs ab16 at ubatch 4096:** 4,079 vs 3,328 (+22.6%). The VRAM gap equals one slot (498 MiB). Cause unexplained: slot count, build, or steady-state definition. Decided by sweep S3.
2. **Copy-source confound:** the 50k curve (ab16) used mmap-registered. The current default is pinned. No same-session pinned-vs-mmap A/B exists at 50k. Cross-session figures: mmap 2,741–2,759, pinned 2,614 (f16, different sessions). Sweep S4.
3. **Stock figures conflict:** `tg-expert-path-reconciliation` quotes 347 tok/s for `--fit-target 16000` at 52,819 tokens (pass 1). ab18 quotes 837.1 at 50,000 (single pass). Different length and pass. The matched comparison uses ab18, and only as an upper bound (§2.1).
4. **200k crowded-VRAM rung:** the 3,266 MiB figure is **`run_start` usage**, not a peak. The entry says so. F16 KV at 200k is ≈ 4.1 GB if fully allocated. The "saves a lot of memory" claim for 200k therefore lacks a sampled peak. Required before use.
5. **Provenance:** every ab16–18 arm is uncertified (`-AllowUnprovenanced`). The V1 row is an uncertified fafix build.

---

## 3. Candidates ranked for PP, with C-LM status

### 3.1 PP track

| Rank | ID | Item | Evidence | C-LM composes? | VRAM effect | Status |
|---|---|---|---|---|---|---|
| **P1** | A2 | Causal, mask-free flash-attention op may use the GQA-optimised kernel variants (`FAFIX_CAUSAL_GQA`, T-GPLR-65) | PP ×1.181 re-measured at ubatch 4096; ×1.186 at 50k (ab9); ×1.342 at 144k; ×1.377 at 200k. Context-independent at 50k (×1.182 at two other contexts) | **Yes.** It is the LM prefill attention op (phase 0 allocates no `kq_mask`, AGENTS.md) | 0 (measured on/off) | T-GPLR-65 draft. Needs a task contract and the answer gate. One hunk in `ggml/src/ggml-cuda/fattn.cu`. Do not take the rest of the fafix tree |
| **P2** | — | Ubatch to the ceiling | 512→1024 +18.3%; 1024→2048 +7.3%; 2048→4096 +0.5% for +1.39 GB | **Yes** (no code) | Compute buffers 937 / 1,218 / 1,780 / 3,032 MiB at ubatch 512 / 1024 / 2048 / 4096 (T-GPLR-68, measured at ctx 262,144 and 2 slots) | Measured |
| **P3** | B1 | Activation slab to pinned host memory | Slab 390.6 / 1,562.5 / 2,048 MiB at 50k / 200k / 262k | **Conditional.** Must keep the A/B double buffer and its event ordering. Adds a per-layer round trip of 409.6 MB at 50k: ≈ 16.4 GB per pass, ≈ 8.2 GB H2D on top of the 18.33 GB bank DMA. Hidden only if compute ≫ transfer (to measure) | −391 MiB at 50k (enables ubatch 2048 at 8 GB). Mandatory for ≥ 144k at 8 GB (−1.1 to −2.0 GiB) | Proposal. Prototype in P5 |
| **P4** | — | ctx sized to the job (prefill-first jobs only) | KV q8: 534 MiB at ctx 51,456; 2,720 MiB at 262,144 | **Yes** (no code) | −2,186 MiB at 50k | Sizes measured. PP at small ctx unmeasured. Owner decision D8 |
| P5 | — | Slots 2 vs 3 | 498 MiB per slot (measured). At 4k, 3 slots +18.8% (T-GPLR-35 addendum 1). At 200k, 2→4 slots +0.8% total (addendum 2). At 50k unknown (doc1 gap) | **Yes** | +498 MiB per slot | Sweep S3 |
| P6 | — | Copy source: pinned (default) vs mmap-registered | mmap ≈ 5% faster at 50k (cross-session). mmap copy-tail at 200k with cold file cache: max copy 10.8 s | **Yes** | 0 | Sweep S4. Decide per length (D9) |
| P7 | — | KV q8 vs f16 | f16 + 3 slots 2,759 vs q8 + 2 slots 2,609 at ubatch 512 (cross-session; KV type not isolated) | **Yes** | +2,400 MiB at 262k (q8 2,720 → f16 5,120) | Sweep S5 (16 GB ceiling only) |
| P8 | — | PP kernel profile at 50k (nsys) | At 200k compute is 92% of wall (addendum 2). Kernel split at 50k unmeasured | **Yes** (analysis) | 0 | Proposed (P7 in §6) |

### 3.2 TG track (must not regress PP, does not gate PP)

| ID | Item | Evidence | C-LM composes? | VRAM effect | Status |
|---|---|---|---|---|---|
| T1 (A1) | TG attention view from live n_kv (T-GPLR-71: remove the `n_pad = n_ctx_total` pin, add a coverage guard) | TG +26.5% at 262k (23.41 → 29.61 tok/s, 2026-10-02). Replicated +25.2% paired (ab23, 2026-10-08). PP +0.25%. VRAM identical (≈ 7,780 MiB, 7.60 GiB, at the 8 GB operating point) | **Yes**, TG path only. PP graph sets untouched | 0 | Measured. Uncommitted (H1). Widened upstream interface `set_view_pad` has no test |
| T2 (A3) | Expert deferral on the stock hybrid (`LLAMA_MOE_HYBRID`, `LLAMA_MOE_HYBRID_DEFER`, ncmoe 40) | TG +15.26% (C64), +16.60% (C32), +25.51% (C16), merged tree, 3 reps. Measured at ctx 1024 only | **Conditional.** Needs LM PP and deferral TG to coexist in the merged tree | C16 ≈ 1.06 GiB, C32 ≈ 2.11 GiB, C64 ≈ 4.22 GiB (1.77 MB × 40 layers × C, expected) | Needs merge (H2), a ctx-262k arm and a stock `-ncmoe 40` baseline |
| T3 (B3) | Stock MTP speculative decoding (`--mtp`) | Never run (handover §4) | **No** for LM. LM TG is capped at 3 tokens without handoff (AGENTS.md) | — | Stock measurement only |

### 3.3 Excluded

| ID | Item | Reason |
|---|---|---|
| X1 (A4) | Stock flags (`--fit-target`, ubatch) | Stock path. **No** for LM. Keep as the baseline comparator only |
| X2 (C1) | moe-prefetch cache-prior router bias (λ = 0.5): TG +91% on GTX 1070 (λ = 0: +6%) | **No.** Different engine. Changes routing (C-CHARTER). Its own doc reports 0% cache hits. Decision D3 |
| X3 (C2) | Host-leg expert decoupling (T-GPLR-82) | Parked. Forced-on arm aborted (GGML_ASSERT, 2026-10-05). N5: impact ≈ nil at operating point |
| X4 (C3) | Per-expert host/device split (T-GPLR-61) | AC2 failed 2026-10-08: greedy IDs diverge at token index 1. Violates C-CHARTER |
| X5 (B2) | Host expert kernel efficiency (60.7 µs in situ vs 36.5 µs microbench) | **No** for PP. Host expert path exists only in TG hybrid |
| X6 | D1 MTP-as-predictor; speculative row fills (T-GPLR-58); aer fused eviction; moe-prefetch VRAM cache and PP window | Closed or negative by gate (Appendix A §A.4) |

---

## 4. Hazards to clear before branching

- **H1 (updated, blocking).** **Two fixes are uncommitted in the nested working tree** (`third_party/llama.cpp`):
  - **T-GPLR-71** (TG view sizing, `tools/layer-streaming/main.cpp:4470`, `src/llama-layer-streaming.h:763`, `src/llama-memory.h:140`).
  - **T-GPLR-66** (`alloc_spill` sized by walking with `ggml_backend_buft_get_alignment`, `src/llama-layer-streaming.cpp` near the `alloc_spill` function). `git log --grep spill` finds no commit. The ab16 "CLOSED" status and the 8 GB ubatch 1024/2048 results **depend on this fix**.
  - Both must be checkpointed, each as its own commit, separated from the T-GPLR-61/82 scaffolding (`src/llama-expert-split-policy.h`, `tests/test-layer-streaming-expert-split.cpp`). Needs D2.
- **H2.** `%TEMP%/parentmerge`: half-finished merge (MERGE_HEAD c51662b, 72 staged entries). Needed only for T2. Owner call.
- **H3.** `C:/Development/fafix` is not under git. Archive the `fattn.cu` diff and `src.tar` (173 MB) to `C:/Development/AI/evidence/` before anything touches it. It is the only copy of the P1 hunk.
- **H4.** `llamacppPPTGOptim`: 86 dirty entries, including deleted `docs/moe-prefetch/*`. Archive the dirty diff.
- **H5.** Aer outer repo: `docs/currentstate.md` and `tools/tel-format-baseline.json` uncommitted. Never stage `third_party/llama-merge/` as a gitlink.
- **H6.** The outer repo lists no remote, but a dangling `gzi/research/layer-residency` ref exists. Check before any push.

---

## 5. Validation method (8 GB and 16 GB ceilings)

- **Ceilings:** 8,192 MiB and 16,384 MiB. The peak is WDDM dedicated memory from the poller's `mem.csv` (max `dedicated_bytes`). **Do not use the `peak VRAM:` console line**, which is a pre-run baseline (AGENTS.md).
- **Steady PP:** `--pp-repeat 3`, mean of passes 2–3. Report the cold pass beside it.
- **Same session, one mutex hold, interleaved arms, n = 2.** Check CPU contamination per arm (`over6`).
- **Stock comparator: steady, not single-pass.** Any stock figure used in a claim must be measured with the same `--pp-repeat 3` convention. Until then, state that the stock figure is single-pass.
- **Real-card caution.** An 8 GB card exposes less than 8,192 MiB to applications (display and driver reserve, amount per card unmeasured). The final 8 GB profile must also run under ballast that leaves 7,500 MiB usable, i.e. ballast = 24,462 − 7,500 = 16,962 MiB.
- **Ballast emulation:** 8 GiB class = 24,462 − 8,192 = 16,270 MiB. 16 GiB class = 24,462 − 16,384 = 8,078 MiB.
- **Limits:** the only GPU here is the RTX 5090 Laptop (24,462 MiB, sm_120). Compute and bandwidth differ from real 8 GB and 16 GB cards. Only VRAM fit and same-session ratios transfer.

---

## 6. Phased plan (PP first)

Each phase has gates. A failed gate stops that phase and goes to escalation.

**P0. Decisions and archives (no GPU, no commits).** Resolve D1–D9 (§7). Archive H3 and H4.
- Acceptance: decisions recorded in §7. Archives exist and hash-match.

**P1. Checkpoints (needs D2).** Commit T-GPLR-71 and T-GPLR-66 as separate checkpoints. Park the T-GPLR-61/82 scaffolding on its own branch. Record the hashes in §4.
- Acceptance: `git -C third_party/llama.cpp status --short` shows only intended files; `git diff --check` is clean in both repos.

**P2. Create `optimizaionV1` (needs D1, D2).** Outer from `research/layer-residency` after P1. Nested from the checkpoint branch. Host-only: `tools/run-tests.ps1 -Suite host`, then `python tools/check-doc-sizes.py`.
- Acceptance: host suite green with no GPU and no mutex.

**P3. Port A2 (T-GPLR-65) as an opt-in flag, default off.** Hunk from §3.1 P1. Remove the env toggle from the final design.
- Gate: same-session A/B at 50k, ubatch 1024 / 2048 / 4096, flag on vs off, n = 2. PP ≥ +10% at 4096 (prior +18%). Peak delta ≤ +50 MiB. Semantic QA PASS at 50 and 512 with the negative control failing as designed. Any drift must be a near-tie (AC1 rule of the T-GPLR-82 contract, reused).
- Escalate if: any route or ID divergence, or QA FAIL.

**P4. PP sweep (one mutex hold, ~30 arms, ~1 h; needs D7).** All arms: 50k prompt, LM, `--pp-repeat 3`, poller on.
- **S1:** ubatch {1024, 2048, 4096} × FA {off, on}; slots 2; pinned; ctx 262,144.
- **S2:** ctx {262,144; 51,456} × ubatch {1024, 4096}; FA on; slots 2.
- **S3:** slots {2, 3} × ubatch 4096; FA on; ctx 262,144. Plus FA off, slots 2 and 3, to reconcile doc1.
- **S4:** copy {pinned, mmap-registered} × ubatch {1024, 4096}; FA on; slots 2.
- **S5:** KV {q8, f16} × ubatch 4096; slots 3; FA on; ctx 262,144 (16 GB ceiling only).
- Output: best config per ceiling. Resolves the doc1 gap, the copy-source choice (D9), the slot count and the ctx policy (D8).

**P5. Activation slab on host (P3 prototype).**
- Gate: peak −≥ 380 MiB at 50k with ubatch 2048 and FA; PP ≥ −3% against the device slab; consumer-wait share unchanged (`check-gplr35-pipeline`); H2D budget logged.
- Then 144k and 200k at 8 GB: peak and PP reported. Pass or fail is recorded, not tuned.

**P6. Length curve.** At the chosen configs and at stock under the same ceiling: 10k, 20k, 50k, 100k, 144k, 200k. Locates the LM crossover (expected ≈ 10k) and documents the out-of-target regime.

**P7. PP kernel profile (nsys) at 50k.** Which kernels hold the compute share after FA: expert GEMM (`mul_mat_id`), attention, DeltaNet recurrence, dense GEMM. Only then choose the next compute lever. At 200k compute is 92% of the wall.

**P8. TG track.** Port T1 (A1), then T2 (A3) after H2 and D4. Measure T3 on stock. None of these may regress P-results.

**P9. Owner summary and acceptance.** Write §10 for each profile. Acceptance stays with the owner. Every gate in this repo is advisory.

---

## 7. Owner decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | Branch name. Typed `optimizaionV1`; this plan uses that spelling. | Confirm, or switch to `optimizationV1`. |
| D2 | Authorise checkpoint commits (P1) before branching. AGENTS.md forbids commits and branches without an explicit ask. | Authorise. H2 is the owner's call. |
| D3 | Routing-changing cache-prior bias (X2): in or out? | Out. Behaviour-change experiment only, if ever. |
| D4 | Deferral numerical exception (T2). | Re-confirm. Gate on the merged tree with QA. |
| D5 | Real 8 GB or 16 GB card reachable (the GTX 1070 machine)? If not, ballast (§5). | Ballast now. Real-card run if reachable. |
| D6 | Delete stale branches and `freebuff-snapshot/*` tags after the port. | Yes, after owner sign-off. |
| **D7** | **Authorise the PP sweep P4 (one mutex hold, ~30 arms, ~1 h).** | **Yes. This decides between the §2.1 candidates.** |
| **D8** | **For 8 GB prefill-first jobs: may ctx be sized to the job (saves 2.2 GB, P4), or must the 262k KV always be kept?** | Decide after S2. Keep 262k for any session that generates later. |
| **D9** | **Copy source default for PP: pinned (current default; safer at 200k) or mmap-registered (≈ 5% faster at 50k warm cache; copy-tail risk on cold cache)?** | Decide after S4. Default stays pinned unless mmap wins by > 3% same-session with no tail risk. |

---

## 8. Doc corrections

1. **AGENTS.md, "Q8_0 KV is LARGER than f16 KV".** Wrong. Q8_0 = 34 bytes per 32 values = 1.0625 bytes per element. f16 = 2 bytes per element. Q8_0 is about 53% of f16. Correct it after checking the `llama_kv_cache` buffer lines from a real run.
2. **`n_pad` status.** AGENTS.md says the `n_pad` pin decides whether decode scans live tokens. This is true and was measured by T-GPLR-71 (+25–26% TG). Owner summary §4 lists the `n_pad` hypothesis as withdrawn ("profiling falsified it"). That withdrawal predates T-GPLR-71 and its 2026-10-08 replication. Update both.
3. **Crowded-VRAM 200k rung** (`docs/currentstate/2026-09-15..2026-09-16.md`, entry at line 233). Its VRAM figure is `run_start`, and the entry says so. Add a line: no sampled peak exists for the 200k rung.
4. **Stock 52,819-token figure (347 tok/s)** vs **ab18 (837.1 at 50,000)**. Reconcile, or label as different length and pass.
5. **T-GPLR-66 status:** "CLOSED" in ab16, but the fix is uncommitted (H1). State both in the plan table.
6. **ab16 table, `v1ref4096` row:** uncertified (fafix build). Mark it in every table that cites it.
7. **`docs/llamacpp 8gb vram*.md`** (three files) are raw `llama-server` logs, not notes. Moved to `build/owner-logs/` on 2026-10-10 (generated output, not version controlled).
8. **Owner summary §5, item 4:** the "stay on stock at 262k" recommendation predates T-GPLR-71 and the ubatch-1024 result. Add a follow-up line.

---

## 9. Risks and limits (stated, not hidden)

- The 50k numbers are single-session (ab16–18) and uncertified by provenance. Each claim above carries its session and binary.
- Copy-source confound: the curve used mmap-registered staging, the default is pinned (§2.5 item 2).
- doc1's +22.6% three-slot gap is unexplained (§2.5 item 1).
- Thin margin for ubatch 2048 + activation slab on host (≈ 172 MiB expected).
- The compute buffer depends on ctx and slots (AGENTS.md: same ubatch 4096 gives 3,032 MiB at 262k/2 slots and 5,472 MiB at 512k/3 slots). The ctx-sized arm may not drop as predicted.
- Activation slab on host: extra PCIe and 1.6 GB of pinned host RAM at 200k are untested.
- The real-card reserve on an 8 GB card is unmeasured.
- No 8 GB or 16 GB card is available here. The only 8 GB data is from another machine and another model (moe-prefetch).
- Stock comparisons are single-pass (§2.1). The 3.72× ratio is an upper bound until a steady stock arm is measured.

---

## 10. Owner summary (for decision)

**1. Goal.** Long-prompt prefill (≥ 50k tokens) as fast as possible under 8 GB and 16 GB VRAM ceilings, using layer-major streaming, without breaking the charter.

**2. Best implementation (decision).** Layer-major PP at current defaults. Ubatch as large as the ceiling allows. The FA causal fix (P1) next, because it is the only compute-side gain measured on the LM path (1.18× at 50k, 1.38× at 200k) and it is VRAM-neutral. Activation slab on host (P3) for 8 GB at ≥ 144k and for ubatch 2048 at 8 GB. Slots, copy source, KV type and ctx policy decided by the sweep.

**3. Mechanism, plain terms.**
- Layer-major prefill loads one layer's expert weights into a VRAM slot, runs that layer for the whole prompt, then moves on. VRAM holds only a few layers at once, so the model size does not matter. The copy overlaps compute when compute per layer is much larger than the copy, which is why long prompts win and short ones do not.
- A bigger ubatch means bigger chunks, so the GPU does fewer, larger kernel launches. The cost is compute-buffer VRAM, which grows with ubatch.
- The FA fix changes a kernel choice on the no-mask causal path. Same mathematical operation, different kernel, so the answer gate is needed.
- The activation slab stores hidden states for every prompt token between layers. It grows with prompt length, and it can live on the host instead of the GPU.

**4. Glossary.**
- **PP (prefill):** processing the whole prompt, reported in tokens per second.
- **TG (token generation):** producing output one token at a time.
- **Steady PP:** mean of prefill passes 2–3 of a repeated run, after the cold first pass.
- **LM (layer-major) / layer streaming:** the scheduling order of §2.1 above, implemented in the layer-streaming runner.
- **Slot:** a VRAM place that holds one layer's expert weights during LM prefill. 498 MiB each at this model.
- **Ubatch:** the number of prompt tokens processed per chunk. Larger chunks use more compute-buffer VRAM.
- **Activation slab (act slab):** the hidden states for every prompt token, kept between layers. Double-buffered (two halves).
- **KV cache (KV):** stored attention history. Sized by context length (ctx), not prompt length. q8 = 8-bit, f16 = 16-bit.
- **ctx:** the context length the server allocates KV for.
- **Copy source:** where bank data is read from for the copy to VRAM. `pinned` = a pinned host mirror (current default). `mmap-registered` = the model file mapping, registered for direct copy.
- **af (async feed):** the worker thread that submits copies ahead of compute. `pp_async_feed=1` is the current default.
- **FA:** flash attention. The FA fix is the mask-free causal kernel choice (P1).
- **Ceiling:** the peak VRAM limit (8,192 or 16,384 MiB).
- **Peak (WDDM dedicated):** Windows' per-process VRAM high-water mark, sampled by the poller.
- **Ballast:** memory allocated up front to shrink the VRAM a process can use, to emulate a smaller card.
- **Mutex:** `Global\AI-GPU-EXCLUSIVE-v1`, held while GPU work runs on this shared machine.
- **Semantic QA:** answer-level grading ladder (needle and knowledge rungs). Not a bit or ID comparison.
- **Expected / measured / unproven:** labels used throughout this file.

**5. Pros and cons.**

| Option | Pros | Cons / limits |
|---|---|---|
| LM at ubatch 1024, 8 GB | **Measured** 3,088–3,112 tok/s at 7,778 MiB; stable across three sessions | Thin headroom to the 8,192 ceiling; ctx 262k kept; single-pass stock comparison |
| + FA fix (P1) | **Measured** 1.18× at 50k (ubatch 4096), 1.38× at 200k; VRAM-neutral | Kernel changed, answer gate needed; the measured binary is uncertified |
| + activation slab on host (P3) | **Expected** −391 MiB at 50k; mandatory for 8 GB at ≥ 144k | Untested; extra PCIe; extra pinned host RAM |
| ctx sized to job (P4) | **Expected** −2.2 GB at 50k | Only for jobs that do not need 262k KV later |
| Stock `--fit-target` | Works with no code | 3.72× slower at the same ceiling (single-pass arm); starves at 262k |
| moe-prefetch bias (X2) | **Measured** TG +91% on an 8 GB card | Changes routing: charter violation. Its gain is unexplained (0% cache hits) |

**6. Recommendation.**
1. Decide D1–D9. Authorise P1 (checkpoints) and D7 (sweep).
2. Run the sweep (P4). It decides slots, copy source, KV type and ctx policy, and resolves the doc1 gap.
3. Port the FA fix (P3). It is the only measured compute-side gain on the LM path.
4. Build the 8 GB profile from the sweep result. Candidate: ubatch 2048 + activation slab on host + FA, or ubatch 4096 with a job-sized ctx if D8 allows. Verify under ballast before any claim.
5. Build the 16 GB profile from the sweep result. Candidate: ubatch 4096 + FA, plus 3 slots if S3 confirms the doc1 gain.
6. Nothing here is approved. Every gate in this repo is advisory evidence.

---

## Appendix A. Inventory and rev-1 survey (inspected 2026-10-10)

### A.1 Outer repo worktrees

| Path | Branch | HEAD | State | Disposition |
|---|---|---|---|---|
| `C:/Development/AI/qwen38GZI` | `research/layer-residency` | 98773fe | 58 dirty entries; `docs/currentstate.md` and T-GPLR docs modified; owner summary and T-GPLR-82 plan untracked | Base |
| `C:/Development/AI/qwen38-aer-worktree` | `freebuff/activation-trace-worktree` | ec2abad | 4 tracked files modified; untracked `third_party/llama-merge/` (172 MB) | Source of A3 (Q36-AER line) |
| `.freebuff/worktrees/5e3cb7b5-…` | `freebuff/we-were-working-on-the-mtp-usage-as-a-predictor-ho-…` | a29ac11 | 24 ahead of research; 1 dirty | Excluded: MTP-as-predictor WIP; D1 closed |
| `.freebuff/worktrees/b78d8fc7-…` | `freebuff/implement-activation-trace-infrastructure-slice-…` | 1a676e3 | 2 ahead / 35 behind research; 1 dirty | Excluded: duplicates aer Phase 1 |
| `.freebuff/worktrees/b947db99-…` | `freebuff/in-this-project-is-it-possible-to-run-in-hybrid-mo-…` | 991dd53 | 3 ahead; 3 dirty | **Excluded (owner: WIP)** |
| `%TEMP%/parentmerge` | `merge/q36aer-into-layer-residency` | 98773fe | **Merge in progress** (MERGE_HEAD c51662b; 72 staged) | Parked (H2) |

### A.2 Branches (outer repo)

| Branch | Tip | Relation to research | Disposition |
|---|---|---|---|
| `research/layer-residency` | 98773fe | (checked out) | Base |
| `master` | f1daeae | research 35 ahead | Fork point |
| `merge/q36aer-into-layer-residency` | 98773fe | equal | Parked merge (H2) |
| `freebuff/activation-trace-worktree` | ec2abad | aer 28 past f1daeae | A3 source via nested merge |
| `freebuff/is-there-a-plan-already-for-this-idea-e0e7ffeb-…` | 5396fda | 1 ahead, 182 behind | Stale |
| `freebuff/can-you-switch-…-afea71ae-…` | f1daeae | = master | Stale |
| `freebuff/letds-work-on-another-tree-…-424dbd90-…` | f1daeae | = master | Stale |
| `freebuff/i-need-to-switch-…-211c0944-…` | 2b54f22 | 1 behind | Stale |
| `gzi/research/layer-residency` (remote-tracking) | 98773fe | equal | Dangling (H6) |

Tags: `layer-residency-plan-complete`; `freebuff-snapshot/*` ×5 (211c0944, 424dbd90, afea71ae, b947db99, e0e7ffeb). Stash: empty.

### A.3 Nested vendored llama.cpp and side copies

- Nested HEAD `9b90025c4` on `research/layer-residency`: 54 commits past `origin/master`, 18 past local `master`. Working tree: 16 modified and 3 untracked files, uncommitted (T-GPLR-71, T-GPLR-66, T-GPLR-61/82 scaffolding).
- Upstream candidates, unreviewed: `origin/jg/ggml-backend-async-copy` (2026-07-04, asynchronous scheduler copies; not in master; relevant to slot transfers), `origin/jg/cuda-fa-np-runtime` (2025-03; FA parallel blocks at runtime; not in master; likely superseded), `origin/gg/fa-no-kq-pad` (2025-09; KQ mask padding; not in master).
- Aer nested: `third_party/llama-merge` (branch `q36aer-x-gzi-merge`, tip 95072ec53, clean). Holds `src/llama-moe-hybrid.{h,cpp}` (A3).
- `C:/Development/fafix`: non-git copy of the research-era tree. `fattn.cu` holds the `FAFIX_CAUSAL_GQA` toggle (P1). `src.tar` 173 MB.
- `C:/Development/AI/llamacppPPTGOptim`: git worktree of `llamacpp-master`, branch `moe-prefetch` at 0fd1ebb8a (9 commits past master, 2026-07-12), 86 dirty.
- `C:/Development/AI/llamacppCUDA`: stock anchor build 10892 (e5a8d439c), the owner's stock baseline.
- Listed, not mined: `moe-inference-engine` (docs/experiments, decisions), `colibri` (other project, GLM / DeepSeek), `UpdaterLlamaCPP`, `ggufsingle` (git, no commits), `llamacppMAin`, `llamaBonsai`, `llamacppVulkan`, `llamacppCache`, `Inference`.

### A.4 Closed items (rev 1, unchanged)

- D1: MTP proposals as expert predictor (T-GPLR-59). Runtime D-gate FAIL, 2026-09-25: zero speculative rows, zero gain.
- Speculative TG row fills (T-GPLR-58): admitted only at warm-up (4.93% of stages). Capacity-bound.
- aer fused eviction policy: NEGATIVE. Hit delta −0.0243 against a +0.020 bar; 0 of 3 reps positive; cost 24.8% of span. Ranker line retired.
- moe-prefetch VRAM cache and promotion thread: 0 cache hits. Abandoned. PP window cache: superseded.
- ARC eviction, mmap + madvise, PreScope triple-stream: not pursued (`paths-not-taken.md`).
- `--ubatch-size 8192` at doc2 (stock): regression. Residency 26 → 2 of 42; KV to CPU; TG 0.44×.
