# Experiment ledger

This ledger lists what was tried, what the record says came of it, and why. It
covers this repository and the branches and forks consolidated from the old
folders (see [HISTORY.md](HISTORY.md)).

Read it before proposing an optimization. If an idea appears below as falsified or
rejected, the proposal needs a new hypothesis, not a repeat.

## How to read it

- **ID** is the workstream ID used in the record (G-series for the llama-server
  work, T-GPLR for layer residency, T-Q38 for dense Qwen3.8 streaming, T-Q36-AER for
  the activation expert ranker).
- **Result** uses the words in the source heading: ACCEPTED (contract met, opt-in),
  DONE, FALSIFIED or REFUTED (hypothesis wrong), REJECTED (owner decision, reverted),
  NO-GO (feasibility failed), REVERTED, OPEN (not closed).
- **Accepted** means the agent accepted its own work against a pre-registered
  contract. The record says this is not owner approval. Each entry names the
  contract.
- **Uncertified** means the run lacked a provenance record, or it used a waiver.
  Runs made before the provenance gate matured (about 2026-10-11) often carry this
  status. Check the entry, and treat the numbers as provisional when it does.
- Numbers are as stated in the source. Where a source says a result is advisory, the
  row says so.
- The reason column repeats the source's own explanation. Where the source gives no
  explanation, the cell is empty.

Every row points to a source. The full list of dated entries is generated in
[history/currentstate-entries.md](history/currentstate-entries.md). Plans are in
[history/plans.md](history/plans.md).

## 1. Current best and the workstreams that produced it

| ID | Date | Tried | Result | Why / what it taught | Source |
|---|---|---|---|---|---|
| SYNC-1 | 2026-10-30 | Update the vendored llama.cpp from upstream b10488 to b12004 (six waypoints, 1,516 commits) | ACCEPTED (agent's own contract, plan section 9; not owner approval). Default path equals pure upstream (perplexity identical); host 78/78; server and runner smokes match the control; prompt +3.4%, decode -1.8% in one cell | Upstream had done the fork's G30 and G31l itself, collided with the fork on `FLASH_ATTN_EXT` `op_params[4]`, replaced the batch API and the KV padding the runner pins, and the recorded upstream base was a fork commit. Untested: Linux/Pascal build, MTP + hybrid margin, long-context decode, 4k+ QA ladder | `docs/plans/upstream-sync-plan.md` |
| PASCAL-1 | 2026-10-19/20 | Linux build on a GTX 1070 (8 GiB, no tensor cores); prefill limits and optimisations; thread pinning for decode | Built, measured, committed locally. Not certified (no provenance gate, no GPU mutex). Causal attention bug fixed | The tile flash-attention kernel ignored the causal window, so `--lm-prefill` computed wrong attention on this GPU (fixed). Int8 QK (opt-in) makes attention 1.67x faster; cold prefill at 100k tokens 471 s to 332 s. Pinning the expert threads took decode from 5-8 to 30-35 tok/s | [guide/LINUX-PASCAL.md](../guide/LINUX-PASCAL.md); `currentstate.md`, 2026-10-19/20 (PASCAL-1) |
| G42 | 2026-10-18 | Fused prefill (`--lm-prefill-fuse`), idle preparation (`--lm-prefill-prearm auto`), canonical checkpoints, `-np 1` | Implemented, opt-in. One contract item missed (GA-5a, by 0.017 nats on one dynamic-cache check) | Request wall, fixture A: 4.84 s to 3.66 s (0.755x against BASE). Idle preparation 0.714x. Small turn 0.902x. Canonical 0.787x. Stability soaks to 138k context | `currentstate.md`, 2026-10-18 entry |
| G32 | 2026-10-15/16 | `--lm-prefill`: batches of 1,024 tokens or more run layer by layer in llama-server | ACCEPTED (own contract LM-1..LM-8) | The owner was right that the G31 estimate of about 1.35x was wrong. Batches below about 1,000 tokens are not taken. Attention-bound past about 50k. The default mode fails hit-versus-fresh checks on swapped fixtures; the canonical mode passes | `currentstate.md`, 2026-10-15/16 entry; `docs/plans/llama-server-integration-results.md` |
| G31 | 2026-10-14 | `--hybrid-moe -fitt 16000` in llama-server, against the unmodified server with the same flags | ACCEPTED (own contract SRV-1..8) | Token generation 3.0x to 3.7x at 4k, 24k and 100k context. Prompt processing 3.65x to 4.11x. Keep-free margin held (minimum 16,169 MiB). Two 160-request soaks, one with `-np 2`, no failure | `currentstate.md`, 2026-10-14 (G31); `docs/plans/llama-server-integration.md` |
| G30 | 2026-10-14 | MTP self-speculation on top of the optimal solution (research runner) | ACCEPTED as opt-in, default off, with a recipe per band | MTP needs no batch-major prefill. Its price is cache slots. Draft depth above 1 loses. The 2-row verify ran an unfused MoE router (found with nsys, `g94-lc4k`). MTP with `--lm-prefill` untested | `currentstate.md`, 2026-10-14 (G30) |
| G-TG8 | 2026-10-11 | LM-hosted AER hybrid decode (the activation-ranker engine ported into the fork) | ACCEPTED vs its contract | Beats stock token generation 1.65x within 8,071 MiB. Needle QA passed at 50, 511, 4,096 and 24,000 tokens | `currentstate.md`, 2026-10-11 (G-TG8) |
| G24-G29 | 2026-10-13 | Token generation, prefill and sampled decoding under the strict whole-run keep-free reading | Mixed; each sub-result is in its entry | Sampled decode costs +0.35 ms per token (G27). Persistent graphs keep their tensor addresses when the compute buffer grows (G28). Segmented layer-major prefill needs a recipe per fill (G28). The owner's document baseline reproduced, and the stock arm of this tree was 21% faster than it (G24) | `currentstate.md`, 2026-10-13 entry; AGENTS.md gotchas |
| G23 | 2026-10-12 night | Paced bank copies, CPU/GPU interleave, stock graph path | Built and measured; mixed | The slow CPU-only path was the PCIe link downshifting while the GPU idled (commit addendum "G23 addendum"). Gen4 falls to Gen2 and Gen1 after about 10 s of low use | `currentstate.md`, 2026-10-12 night; the commit "docs+tools: G23 addendum" in the outer history |
| G19-G22 | 2026-10-12 evening | Deep-fill validation of the final configuration, GQA8 decode attention, dedicated-GPU prefill recipe, short-prompt study | Measured. Several limits found | Prefill rate on a dedicated GPU comes from the ubatch, not from slots or slab type (G21). Layer-major floor about 1.3 s, stock wins below about 600 tokens (G22). Footprint grows with the prompt and breaks the keep-free margin on long fills (G19). The vector FA kernel reads GQA KV eight times (G19/G20) | `currentstate.md`, 2026-10-12 evening; AGENTS.md gotchas |
| G8-G18 | 2026-10-12 | Lean host leg, predictive cache prefetch, stall root cause, MTP, policy ideas, final configuration under the owner's 8 GB definition | See the entry for each sub-result | The WDDM driver over-commits VRAM, so a ballast process cannot emulate a smaller card (G8). Software prefetch instructions are mostly dropped under a burst (G15) | `currentstate.md`, 2026-10-12 entry; AGENTS.md gotchas |
| G3-G7 | 2026-10-11 | Prefill-sized scheduler buffers, first-pad-boundary deadlock, lazy KV cache, auto-sized expert cache, input batching | Root-caused and fixed; LM-hosted hybrid 2.7x to 2.9x stock (final builds g11-g14) | A GPU-spinning handshake must not depend on a thread that makes CUDA calls (G5). Input batching: three of four designs lose (G4) | `currentstate.md`, 2026-10-11 entry |
| G2 | 2026-10-11 | Decode-phase cache resize | About 2x stock inside the same 8,071 MiB peak | "VRAM in use" has three values on this machine, and they differ by constants (2026-10-11 gotcha) | `currentstate.md`, 2026-10-11 (G2) |
| KV graph pad | 2026-10-03 | Padded KV graph at the 50k shape, four matched arms | +19.5% token generation at 50k. Advisory only (soft gate). The earlier headline for that arm was an artifact | Check the arm definition before quoting a headline | `currentstate.md`, 2026-10-03 entry |
| optimizaionV1 D7 | 2026-10-10 | Flash attention causal-GQA opt-in, PP sweep over 27 arms | +13% to +17% steady prefill at 50k with identical VRAM. Best 8 GB: 3,693 tok/s (ubatch 4096 with flash attention at job context, 5,724 MiB peak). Best 16 GB: 3,820 tok/s. The plan's expected 4,800 tok/s was not reproduced. All arms UNCERTIFIED at the time | `currentstate.md`, 2026-10-10 (optimizaionV1 D7); branch `archive/optimizaionV1` |

## 2. Falsified, refuted, rejected: do not repeat without new evidence

| ID | Date | Hypothesis or approach | Result | Why |
|---|---|---|---|---|
| T-Q38TG-00 | 2026-08-24 | Hybrid token generation for the dense Qwen3.8 model | NO-GO | Host-to-device rate measured about 3.0 GB/s. Break-even about 29 GB/s. The dense hybrid path is gated |
| T-GPLR-17 | 2026-09-10 | Two-process hybrid bridge (streaming PP writes a state file, a CPU-expert llama-server does TG) | REJECTED by the owner; fully reverted | Owner decision |
| T-GPLR-08 | 2026-09-09 | Predictive expert prefetch | FALSIFIED by instrumentation; task closed | The hypothesis failed the measurement |
| T-GPLR-15 | 2026-09-13 | Row retention meets the 0.6 gate in the SDD reference band | CLOSED, falsified by evidence. Phase A: 0.3088 and 0.4343 against 0.6 on 10,200 samples. Non-echoing re-probe: K=4 gives 0.5355, and K=6 is the first depth that clears | K=8 became the default on 2026-09-18 (owner-directed) |
| T-GPLR-18 | 2026-09-13 | Pinned row-set staging | CLOSED, falsified by evidence (512 A/B) | Pinned only relocates the staging copy |
| T-GPLR-30 | 2026-09-16 | Partial persistent pinned pool (`--pp-pool`) | DONE, warm savings flat at zero | Hit mechanics perfect; the savings curve stayed at zero |
| T-GPLR-32 | 2026-09-16 | Serialize the worker DMAs as a wall lever | FALSIFIED. Warm walls identical | Measured, no wall change |
| T-GPLR-33 | 2026-09-16 | One submission per bank | FALSIFIED. Windows identical with 1 and 3 submissions | Measured, no window change |
| T-GPLR-34 | 2026-09-16 | WDDM cross-engine serialization (mid-kernel probe) | FALSIFIED. A second stream overlaps compute and can overtake it | Measured |
| T-GPLR-37 | 2026-09-17 | Direct mmap-sourced staging | Feasible. Regression at 4k as implemented. The 4k win is not visible at 200k and 262k (warm walls flat) | Churn confirmed as the cause of the 4k effect (addendum 2) |
| T-GPLR-38 | 2026-09-17 | Ubatch sweep at 4k | The prediction inverted at ubatch 2048 | See the entry for the measured walls |
| T-GPLR-16 | 2026-09-11 | Suspected corruption at the PP-to-TG switch | Root cause: a mixed-arithmetic switch, not corruption | |
| T-GPLR-44 | 2026-09-18 | Per-launch patch | REVERTED by falsifier (the instrument is kept) | |
| T-GPLR-61 | 2026-10-08 | Per-expert split of the row set (`moe-row-split`) | AC2 FAILED: the split does not preserve byte-identical output | Gate not met |
| T-GPLR-82 | 2026-10-05 to 10-10 | Host expert leg (CPU tail) as the cause of a non-finite cascade | The per-op row-count hypothesis REFUTED. The cascade comes from the S3 copy-wait decoupling. The CPU leg stays finite but destroys the answer. AC1 gate FAILS | Left flagged off by default |
| G-prefetch (AGENTS) | 2026-10-12 | Software prefetch instructions | Mostly dropped under a burst (G15) | Plain loads worked |
| G-ballast (AGENTS) | 2026-10-11 | Simulate a smaller card with a ballast process | Not possible: WDDM over-commits and pages instead of failing (G8) | The project limits VRAM with `-fitt` |
| G30 depth | 2026-10-14 | MTP draft depth above 1 | Loses (plan "G30 DEPTH STUDY", ledger row 81) | Keep depth 1 unless a new study shows otherwise |
| G-n_pad (AGENTS) | 2026-10-12 | Setting `n_pad` small on its own to shorten decode attention | Not a fix. Decode attention cost tracks `--ctx-size` through launch geometry, not KV bytes read | AGENTS gotcha |
| G-view (AGENTS) | 2026-10-12 | Narrowing the TG graph set's KV view tensors | 18.9% slower | Do not mutate the TG graph's KV views |
| PASCAL-1 tile | 2026-10-20 | A 128-row MMQ tile for the K-quant MoE matmuls (Q4_K, Q5_K) on Pascal | 14 to 16% slower than the 64-row tile | Only dense Q8_0 gains (24%). With about 64 tokens per expert, fewer larger tiles waste more |
| PASCAL-1 J cap | 2026-10-20 | Narrower column tiles for the MoE matmuls (J 48 down to 16) | Slower at every value (gate/up 7.4 ms to 10.1 ms, down 9.8 ms to 21.0 ms) | J=64, which the host heuristic already picks, is the best |
| PASCAL-1 int8 configs | 2026-10-20 | Int8 attention kernel with 128 threads or 64 K rows per iteration | 1.6 to 7.5 times slower than the 256-thread, 32-row layout | Register spills (stack up to 1,960 B per thread) |
| PASCAL-1 V tile | 2026-10-20 | A whole 32-row V tile in the int8 kernel (fewer barriers) | No change | The kernel is bound by instruction issue, not by barriers |
| PASCAL-1 force pass | 2026-10-20 | The layer-major pass for every prompt size (`--lm-prefill-fuse-min 1`) | An 8-token prompt 14 times slower (0.21 s to 3.07 s); no gain above 570 tokens | The pass has a fixed floor of about 3.1 s on this machine (PCIe). Never set `--lm-prefill-min 1`: it catches decode steps |
| PASCAL-1 KV type | 2026-10-20 | f16 instead of q8_0 KV to avoid dequantization in the attention kernel | Same kernel time at every depth | The q8_0 to f16 conversion runs outside the kernel and is small |
| PASCAL-1 load mode | 2026-10-20 | `--load-mode none` instead of mmap (the server's own log hint) | No faster; 60 s to load against 4 s | |

## 3. Transfer and staging measurements (facts for the next design)

| ID | Date | Measurement | Result |
|---|---|---|---|
| T-GPLR-29 | 2026-09-16 | Host-to-device ceiling, Gen4 x8 laptop link | 14.4 GB/s, about 90% of the link. Overlap converts only with pinned staging |
| T-GPLR-31 | 2026-09-16 | Locked (pinned) memory the driver grants | 44.72 GiB, granted in 432 MiB steps; fails beyond with a CUDA error |
| T-GPLR-35 | 2026-09-16 | Three slots instead of two | Warm 4k layer-major 3,377.8 ms to 2,843.2 ms (-15.8%). Depth does not pay at long context (addendum 2) |
| T-GPLR-36 | 2026-09-17 | Long-context residual cost | Per-launch host work, not per-graph setup |
| T-GPLR-28 | 2026-09-16 | First-touch cost | Paid once per process. Warm state is stable in-process |
| 4k layer-major | 2026-09-15 | Layer-major against fresh batch-major at 4k | 2.3x PP-wall gain. The 3.5x to 4x prediction was falsified as stated |
| MMAP copies | 2026-10-13 | Pageable mmap copy sources | Copies are host-synchronous. The lookahead overlaps compute, but the copies still serialize on the copy stream (observed) |

## 4. Workstreams closed as done (layer residency, T-GPLR)

| ID | Date | Scope | Result |
|---|---|---|---|
| T-GPLR-00 | 2026-09-04 | MoE artifact inventory | DONE |
| T-GPLR-01 | 2026-09-04 | Qwen package adapter extracted | DONE |
| T-GPLR-02 | 2026-09-04 | Expert-bank package proof | DONE |
| T-GPLR-03 | 2026-09-05 | 512-token MoE layer-major prefill and parity | DONE |
| T-GPLR-04 | 2026-09-05 | 262k Q8 out-of-core capacity proof | DONE |
| T-GPLR-05 | 2026-09-05 | Capability registry and regression closure | DONE |
| T-GPLR-10 | 2026-09-09 | MoE handoff: streaming PP, then stock llama.cpp TG | DONE |
| T-GPLR-11 | 2026-09-09 | Batch-major tail-ubatch gap | DONE |
| T-GPLR-12 | 2026-09-09 | Default profile v2 (bf16 kept on MoE by owner decision) | DONE |
| T-GPLR-13 | 2026-09-13 | Reactive expert row-set residency for TG | DONE (owner review). All five ladder sizes PASS |
| T-GPLR-14 | 2026-09-09 | Route-overlap instrument | DONE |
| T-GPLR-19 | 2026-09-13 | Per-layer row residency at K=1 | DONE. All six acceptance criteria passed |
| T-GPLR-26 | 2026-09-18 | Gate eligibility (advisory) | CLOSED: advisory, documents only |
| T-GPLR-28 | 2026-09-16 | First-touch cost analysis | DONE |
| T-GPLR-43 | 2026-09-17 | 144k pass-2 crash (q8 KV, long context, pass-2 reuse) | FIXED; regression clean |

## 5. Dense Qwen3.8 layer streaming (T-Q38)

These entries were written for the dense model. The layer-major path was later
generalized (see [../layer-residency-sdd.md](../layer-residency-sdd.md)).

| ID | Date | Scope | Result |
|---|---|---|---|
| Arbitrary-length layer-major PP | 2026-08-23 | Prompt length without a fixed limit | DONE |
| T-Q38PW-00 | 2026-08-23 | Patchability probe | DONE |
| T-Q38PW-01 | 2026-08-23 | Per-width graph sets and staged input buffer | DONE |
| T-Q38PW-02 | 2026-08-23 | 262k stretch with bf16 slabs | DONE (see entry for limits) |
| Per-layer slot bundles | 2026-08-23/24 | Dynamic-quant models | DONE |
| T-Q38FA-01 | 2026-08-24 | Per-chunk K/V window on the implicit-causal flash-attention path | DONE |
| T-Q38FA-02 | 2026-08-24 | Resident control verdict | DONE |
| T-Q38FA-04 | 2026-08-24 | Patch-walk cache | DONE |
| T-Q38TG-00 | 2026-08-24 | Hybrid TG feasibility probe | NO-GO (see section 2) |
| T-Q38HG-00/01/02 | 2026-08-24 | PP-to-TG handoff | DONE |
| T-Q38HG-04 | 2026-08-24 | Constrained handoff (attention-only migration) | DONE. Migration 3.54 GB in 0.424 s. Parity ID 286. Test suite 36/36. TG 3.28 tok/s, within about 15% of the `-ngl 17` baseline |
| T-Q38HG-05 | open | GTX 1080 validation | OPEN. Gated on owner hardware access |

## 6. Activation expert ranker and hybrid decode lineage (T-Q36-AER)

The lineage runs from the ranker plan to the hybrid decode in the server.

| Step | Date | What | Source |
|---|---|---|---|
| Plan | (dates in the file) | Phases 0 to 7: reference identity, trace instrumentation, raw dataset, feature projection, offline predictor lab, cache-aware decision, future runtime (only if authorized) | `docs/plans/qwen36-activation-expert-ranker-plan.md` |
| Branch work | 2026-10-02 | Step 5: beta=1 fused promotion policy, opt-in, default OFF. Shared hybrid eligibility test, and the ranker batches are proven | retired nested fork, branch `q36aer-activation-trace-phase1` (kept in `archive/source-repos/aer-llama-nested-all-refs.bundle`); index in [history/nested-aer.md](history/nested-aer.md) |
| Port | fork history | "layer-streaming: port AER hybrid MoE decode (LLAMA_MOE_HYBRID) as --tg-hybrid" | fork commit in `third_party/llama.cpp` (replayed) |
| Measured | 2026-10-11 | G-TG8: LM-hosted AER hybrid decode beats stock TG 1.65x | section 1 |
| Integrated | 2026-10-14 | G31: `--hybrid-moe` in llama-server | section 1 |

The branch-only commits from the other two nested clones are indexed in
[history/nested-freebuff-5e3cb7b5.md](history/nested-freebuff-5e3cb7b5.md) (5
commits) and [history/nested-freebuff-b78d8fc7.md](history/nested-freebuff-b78d8fc7.md)
(1 commit). Their branches are work in progress and carry no verdict.

## 7. Optimization v1 branch (outer `archive/optimizaionV1`)

| Commit | Subject (short) | Status |
|---|---|---|
| Branch tip (outer) | "docs: optimizaionV1 validation results (PP reproduced; causal-GQA answer gate PASS in device-TG config; TG at 8/16 GB blocked)" | Recorded. The D7 sweep above is the usable part |
| Nested commit | "optimizaionV1: opt-in causal-GQA FA flag; drop T-GPLR-82 falsifier; fix build" | Kept as `archive/llama-optimizaionV1` in this repo |

## 8. Measured facts that cost time (do not rediscover)

The full list with dates is in [../../AGENTS.md](../../AGENTS.md), section
"Vendored-engine gotchas". The ones that most often mislead:

- `nsys` SQLite `start` and `end` are nanoseconds.
- Q8_0 KV is about half the size of f16 KV, not larger.
- The stock server reads past its MoE scratch buffer at `-ub 2048` (found with
  `compute-sanitizer --tool memcheck`, G31l).
- Uninitialized memory passes every test until the memory is reused (G31i).
- Python text-mode writes on Windows convert LF to CRLF.
- `ctest -R layer-streaming` selects 111 tests, and 25 of them need
  `Q38LS_MODEL_PATH` and fail without it.
- A digest compared across wrapper invocations can differ for an unchanged tree
  (2026-10-12).
- A reverted nested-tree edit is still an edit for provenance.
- The machine can be on battery. Then every timing is a different machine.
- The PCIe link downshifts after about 10 s of low GPU utilisation and needs about
  0.25 s of traffic to recover.
- `llama-completion` and `llama-tokenize` process backslash escapes by default
  (`-e`). The layer-streaming runner does not.
- Until 2026-10-20 the tile flash-attention kernel (GPUs without tensor cores) did
  not honour the causal window that `--lm-prefill` passes, so tokens saw later
  tokens of their own chunk. Any `--lm-prefill` number from such a GPU taken before
  the fix used the wrong attention (PASCAL-1). Run
  `test-backend-ops test -o FLASH_ATTN_EXT` on a new GPU before trusting a pass.
- `LLAMA_MOE_HYBRID_LEAN_CPUS` has to match the CPU's SMT layout. On a CPU whose
  two threads per core are `i` and `i+4`, the default layout cost a factor of about
  5 in decode (PASCAL-1).
- `--lm-prefill-canonical` re-prefills up to one chunk on every follow-up: a
  3-token follow-up to a 4k-token conversation took 3.2 s with it and 0.12 s
  without it on a GTX 1070 (PASCAL-1).

## 9. Owner decisions that set defaults

| Date | Decision | Effect |
|---|---|---|
| 2026-09-09 | bf16 kept on MoE (T-GPLR-12) | Default profile v2 |
| 2026-09-10 | Two-process bridge rejected (T-GPLR-17) | Reverted |
| 2026-09-14 | K=4 as default row-cache depth | Superseded on 2026-09-18 |
| 2026-09-15 | Default `--layer-slots 2`, auto-3 on fit | Default |
| 2026-09-16 | Goal charter: PP speed first, VRAM second, speed must hold under crowded VRAM | Governs what is judged a win |
| 2026-09-18 | Default retention depth K=8 | Default |
| 2026-09-15 | AGENTS.md: the MoE-Qwen focus is a measurement choice; other models on owner request | Scope |

## 10. Open items

| ID | Status |
|---|---|
| Upstream sync follow-ups (2026-10-30) | Open: Linux GCC / CUDA 12.8 build and the GTX 1070 + dense Qwen3.8 cells; long-context decode (50k-200k) and the 4k QA rung on the synced tree (the KV `n_pad` pin was re-applied through `set_attn_n_pad`); hybrid server with MTP at `--fit-target 16000`; upstream's MoE cache against the fork's (decision D3); the cause of the erratic IQ1/2/3_S matmul failures on the RTX 5090 Laptop |
| T-GPLR-82 | Gate FAILS. Left flagged off by default |
| T-GPLR-61 | AC2 FAILED. Not enabled |
| T-Q38HG-05 | Open. Gated on GTX 1080 access |
| MTP with `--lm-prefill` | Untested |
| `--lm-prefill-bf16` | Not validated against f32 |
| `-np 2` at 70k context | About a fifth of big batches do not fit under the margin |
| Chunk fusion into the pass | Not done. About 1.3 s per turn left on the table |
| Older known gaps (2026-10-12 list) | See [../guide/DESIGN.md](../guide/DESIGN.md#6-limits) for which are still open |

## 11. Adding an entry

1. Add the dated entry at the top of the "Recent entries" block in
   `docs/currentstate.md`, with the result, including failures.
2. If the result changes the status of an item above, update the row here.
3. Rerun the generator: `python tools/gen-history-index.py`. It rebuilds the
   indexes from the sources.
4. If the entry is a new hypothesis that failed, add a row to section 2.
