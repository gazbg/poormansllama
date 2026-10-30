# Plan: sync the vendored llama.cpp with upstream ggml-org/llama.cpp

Status: draft (planning only, nothing merged, nothing built; owner decisions D1-D4 below are open)
Dependencies: none (does not touch the active GPLR plan; it changes the engine every GPLR result was measured on)
Commit: unfilled (plan written 2026-10-30; no vendored-tree change yet)
Authority: owner request 2026-10-30, "pull the latest, see how (plan) to update the current repo with the latest ggml-org/llama.cpp it is based on".

## 1. Where things stand (measured 2026-10-30)

**Pull.** `git fetch origin --prune`: local `main` equals `origin/main` (`0a7a0f4`, PR #7), 0 commits behind. Two stale remote branches pruned. Nothing to pull.

**The recorded upstream base is wrong.** `third_party/README.md` and `docs/reference/HISTORY.md` name `36f0ae09` as the "upstream base". That id is not on ggml-org: a fetch by id is refused with `not our ref`, and `docs/plans/generic-prefill-layer-residency-plan.md:52` calls `36f0ae090` a nested fork commit ("whitespace fixup"). The vendored copy is therefore "fork snapshot of 2026-09-23 + 26 commits", not "upstream + 26 commits".

**True fork point, recovered by file hashes** (`tools/find-upstream-base.py`, host-only, 0.7 s): the vendored base commit `6dae153` matches upstream **`04b569142da23d91beca090a99098d592d3f3c80`** = tag **b10488**, "common: share thread pools when `n_threads` differ (#27138)", 2026-08-18. 3,403 of 3,488 vendored files are byte-identical to that commit; the next best position scores 1 file lower. The 85 others are fork-added or fork-modified.

**Upstream now.** `master` = `e216f4ec8753` (2026-10-30, b12004 + 5 commits). 1,516 commits ahead of the fork point (236 / 582 / 698 in Aug / Sep / Oct), 2,201 files, +392,277 / -146,386 lines.

**Fork delta against the true base** (HEAD tree `2404a1e8`): 148 files, +58,337 / -314. 100 files are added (63 under `tools/layer-streaming`, `src/llama-layer-streaming.cpp` 623 KB, `tools/layer-streaming/main.cpp` 464 KB, `src/llama-moe-hybrid.cpp` 120 KB, `src/llama-lm-prefill.cpp`, `src/llama-layer-streaming.h`, the tests); 48 are modified, 45 of those also changed upstream. No fork-added path exists upstream (no add/add collision). The fork is overwhelmingly additive: 314 deleted lines in total.

**Merge dry run** (`git merge-tree --write-tree`, merge base = `04b569142da2`, ours = the vendored tree committed on top of it, theirs = `e216f4ec8753`; read-only, 0.3 s): **25 files conflict, 70 hunks, 1,402 fork-side and 776 upstream-side lines inside the hunks.** Everything else (including 20 of the 45 files both sides touched) merges textually clean.

| File | Hunks | Lines in hunks (fork / upstream) | Fork delta | Upstream delta | First conflicts at |
|---|---|---|---|---|---|
| `src/llama-graph.cpp` | 12 | 41 / 171 | +279 -127 | +837 -333 | b10991 |
| `src/llama-context.cpp` | 6 | 461 / 14 | +561 -6 | +718 -150 | b10991 |
| `tools/server/server-context.cpp` | 6 | 239 / 81 | +817 -13 | +2353 -549 | b10729 |
| `ggml/src/ggml.c` | 6 | 80 / 49 | +86 -2 | +280 -42 | b10991 |
| `tests/test-backend-ops.cpp` | 4 | 86 / 97 | +103 -4 | +3196 -317 | b10729 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 4 | 20 / 16 | +39 -13 | +307 -165 | b10991 |
| `src/llama-kv-cache.cpp` | 3 | 151 / 6 | +340 -2 | +481 -181 | b11311 |
| `ggml/include/ggml.h` | 3 | 40 / 25 | +43 -0 | +143 -15 | b10991 |
| `common/speculative.cpp` | 3 | 30 / 4 | +92 -1 | +666 -299 | b11311 |
| `src/llama-graph.h` | 2 | 44 / 13 | +85 -2 | +175 -35 | b11311 |
| `common/common.h` | 2 | 35 / 1 | +36 -0 | +198 -39 | b11311 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 2 | 10 / 7 | +888 -6 | +1130 -215 | b10729 |
| `src/models/qwen35moe.cpp` | 2 | 7 / 5 | +765 -36 | +27 -34 | b10991 |
| `src/models/qwen35.cpp` | 2 | 7 / 5 | +313 -22 | +48 -30 | b10991 |
| `ggml/src/ggml-cuda/topk-moe.cu` | 2 | 2 / 4 | +13 -0 | +91 -13 | b11850 |
| `src/llama-model.cpp` | 2 | 2 / 0 | +2 -2 | +1197 -215 | b12004 |
| `ggml/src/ggml-backend.cpp` | 1 | 101 / 10 | +4 -1 | +660 -194 | b11675 |
| `src/llama-kv-cache.h` | 1 | 12 / 0 | +58 -1 | +53 -17 | b12004 |
| `src/llama-context.h` | 1 | 8 / 1 | +128 -0 | +55 -2 | b11311 |
| `common/arg.cpp` | 1 | 7 / 14 | +237 -3 | +316 -105 | b11675 |
| `ggml/src/ggml-cuda/fattn.cu` | 1 | 5 / 161 | +142 -4 | +316 -105 | b10991 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 1 | 5 / 4 | +36 -2 | +76 -14 | b11850 |
| `ggml/src/ggml-cpu/vec.cpp` | 1 | 4 / 86 | +4 -0 | +110 -10 | b11850 |
| `ggml/src/ggml-cuda/mmq.cu` | 1 | 3 / 1 | +2 -1 | +199 -50 | b11675 |
| `ggml/src/ggml-backend-meta.cpp` | 1 | 2 / 1 | +3 -1 | +344 -67 | b11850 |

"First conflicts at" comes from dry runs of the same fork tree against five upstream waypoints (always the original base as merge base): conflicted files grow 3 (b10729, 236 commits) -> 11 (b10991, 498) -> 16 (b11311, 818) -> 19 (b11675, 1,182) -> 23 (b11850, 1,357) -> 25 (tip, 1,516). A file that conflicts early can stop conflicting later (`server-context.cpp` is clean at b11675), so the column is first sight, not a stable mapping. `ggml-backend.cpp` (101 fork-side lines for a +4 -1 fork change) and `vec.cpp` (86 upstream-side lines for +4) show a one-line fork edit sitting inside code upstream rewrote: read these as small, not as large.

Not read yet: the hunks themselves. The table ranks effort by size; triage is step T-SYNC-02.

## 2. What upstream changed that touches fork code (clean merges can still break)

A textual merge says nothing about meaning. These upstream changes sit under code the fork depends on; each becomes a named re-check in section 5.

1. **KV cache rework.** `cache_v_l` and the `cache_k_l%d`/`cache_v_l%d` tensor naming are gone from `src/llama-kv-cache.cpp`; `n_pad_cur` (the padding rule `get_n_kv()` uses, which the AGENTS gotchas call "the single switch" for decode attention scan length) is gone; graph params now carry `attn_v_trans`. The fork's layer-streaming code names `cache_k_l`/`cache_v_l`, pins `cparams.n_pad`, and patches K/V view extents (T-Q38FA-01). Highest semantic risk.
2. **MMQ restructure.** `ggml_cuda_mmq_get_J_max` and `args.ncols_max` are gone. The fork's G31l fix (slack of the quantized-activation buffer in `ggml_cuda_mul_mat_q()`'s ids branch, sized from `ne12`) must be re-derived, and the out-of-bounds read it fixed must be re-tested rather than assumed fixed or still present.
3. **Native MoE expert GPU cache** (`src/llama-moe-cache.{cpp,h}`: #29887 2026-10-07, multi-GPU #30112, SLRU #30637). Overlaps the fork's hybrid MoE decode and its VRAM expert cache (`src/llama-moe-hybrid.cpp`). Both hook `MUL_MAT_ID` and `ggml-backend`; see decision D3.
4. **Native MTP and speculative work** (#27694, #28549 "CUDA graph for MTP draft", #29761, #30017/#30020 nextn extraction, #30257, #31350 spec checkpoint restore, `--spec-draft-n-ubatch`). Overlaps the fork's MTP draft, mirror and catch-up paths and conflicts in `common/{arg.cpp,common.h,speculative.cpp}` and `server-context.cpp`.
5. **CUDA changes under fork kernels.** Chunked CUDA graphs (#30804, 2026-10-24) against the fork's GPU spin-wait handshake (`moe_hybrid_wait_add`) and input batching; MoE `MUL_MAT_ID` gate/up/GLU MMQ fusion (#31188, #29948), top-k MoE fusion for 384 experts (#31288) and grouped experts (#29181) against `LLAMA_CUDA_TOPK_MOE_SMALL` and `GGML_MOE_HYBRID_ID_FLAG_DUP`; the `flash_attn_ext_vec` shared-memory fix (#30573) under `fattn-gqa8.cuh`; "always MMVQ for `MUL_MAT_ID` on sm_60" (#27828); faster low-bit MMVQ (#30517).
6. **Pascal MMQ config split.** Upstream deleted `mmq-config-pascal.cuh` and split it into `mmq-config-pascal-dp4a.cuh` and a new `mmq-config-pascal-older.cuh` (#26264). Git detected the 99% rename and carried the fork's 128-row Q8_0/IQ3/IQ4_XS tile change into `-pascal-dp4a.cuh` cleanly (verified by content); still unverified: which of the two files the GTX 1070 (sm_61) selects, and whether the tile still wins there.
7. **Tensor parallel (`-sm tensor`, `ggml-backend-meta.cpp`)** and backend-sched changes: the fork is single-GPU; the risk is compile and scheduler behaviour, not features.

A cheap identifier check (fork-added code against identifiers upstream removed, confirmed with `git grep` on both trees) found only the removals in items 1 and 2 plus `attn_n_pad` and `batch_view`. It does not see changed signatures; the real check is a build (T-SYNC-03).

## 3. Strategy

| Option | What | Verdict |
|---|---|---|
| **A. Staged three-way merge** (recommended) | Real `git merge` of upstream tags into a scratch clone, six waypoints, one resolution session and one validation each; vendored tree replaced waypoint by waypoint | Conflicts are few (70 hunks) and arrive in groups of 3, 8, 5, 3, 4, 2, so each is resolved with its upstream context in mind; a failure bisects to one waypoint; merge base is exact |
| B. Replay the fork commits (`git am`) | Re-apply the 26 + 8 fork commits onto new upstream | Rejected. The Aug-Sep fork work exists only as the 6dae153 snapshot (the nested `.git` was removed); replay covers 34 of the commits and conflicts repeatedly on the same hunks |
| C. Fresh upstream, re-port the fork as one patch | Start from the tag, apply the 148-file delta by hand | Fallback for single files whose merge is unreadable (`llama-graph.cpp` is the candidate); not the default, 100 files are pure additions anyway |
| D. No sync, cherry-pick fixes | Pull only specific upstream commits | Fallback if A's cost proves too high; each cherry-pick still needs the same re-validation and the gap keeps growing |

Waypoints (tags exist on ggml-org; commits from the fork point): **W1 b10729** (236), **W2 b10991** (498), **W3 b11311** (818), **W4 b11675** (1,182), **W5 b11850** (1,357), **W6 target tag** (1,516 at `e216f4ec8753`; pin a tag, see D1).

## 4. Task contract

### T-SYNC-01 Scratch merge workspace and baselines (no vendored change)

- Goal: a reproducible place to merge, and a frozen control to measure against.
- Steps:
  1. Correct `third_party/README.md` and `docs/reference/HISTORY.md`: upstream base is `04b569142da2` (b10488), `36f0ae09` is a fork commit; keep the tree-check row. Add `third_party/UPSTREAM_BASE` with the base tag and sha so the next sync does not need detection.
  2. Short path (Windows `gc` fails with "Filename too long" in deep scratch paths), for example `C:/Development/AI/llama-sync`: `git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git C:/Development/AI/llama-sync`, then `git fetch <this repo> HEAD:poor-head`, then `OURS=$(git commit-tree poor-head:third_party/llama.cpp -p 04b569142da2 -m "vendored fork tree")`, `git checkout -b sync/w0 $OURS`. Do not push anything from it.
  3. Build the current tree (docs/AGENT_ENVIRONMENT.md for Windows MSVC + CUDA 13.2; docs/guide/LINUX-PASCAL.md for GCC + CUDA 12.8) and **side-copy the exe and all DLLs before the first merge** (AGENTS: rebuilding destroys the control). Also build a pure-upstream tree at the target tag with all fork flags absent, for the default-off comparison in T-SYNC-04.
  4. Record baselines with the existing harnesses on the same session, mains power, GPU mutex: hybrid decode and layer-major prefill cells of `docs/plans/llama-server-integration-results.md`, the dense Qwen3.8 GTX 1070 prefill cells of `docs/plans/dense-qwen38-iq3-residency.md`, host suite, `test-backend-ops` subset. The recorded history is a reference only; the side-copied control run in the same session is the comparison.
- Acceptance: README/HISTORY corrected, control binaries and baseline logs present with command, flags, exit status and telemetry path.
- Non-goals: any change under `third_party/llama.cpp`.

### T-SYNC-02 Waypoint merges W1 .. W6

- Per waypoint: `git merge --no-ff <tag>` in the scratch clone; triage hunks first (which of the table's entries are one-line edits inside rewritten code, which are real logic); resolve; compile CPU and CUDA; export the tree as one commit in this repo (`third_party: sync llama.cpp to bNNNN`), with the conflict list and every non-trivial resolution decision in the commit body and in this file.
- Resolution rules, binding: (1) upstream structure wins; re-apply the fork's hook on top of it instead of reverting an upstream refactor to keep a fork line; (2) API drift in fork-added files is fixed in the fork files, never by editing upstream signatures; (3) every fork behaviour stays behind its opt-in gate, the default path equals upstream; (4) no resolution is guessed: a hunk that cannot be understood from both sides goes to option C for that file and is recorded.
- Expected content per waypoint (from the dry runs, dates are upstream): W1 `ggml-cuda.cu`, `test-backend-ops.cpp`, `server-context.cpp`; W2 + `ggml.h`, `ggml.c`, `fattn.cu`, `fattn-mma-f16.cuh`, `llama-context.cpp`, `llama-graph.cpp`, `qwen35.cpp`, `qwen35moe.cpp`; W3 + `common.h`, `speculative.cpp`, `llama-context.h`, `llama-graph.h`, `llama-kv-cache.cpp`; W4 + `arg.cpp`, `ggml-backend.cpp`, `mmq.cu`; W5 + `ggml-backend-meta.cpp`, `ggml-cpu.c`, `vec.cpp`, `topk-moe.cu`; W6 + `llama-kv-cache.h`, `llama-model.cpp`. Re-run the dry run on the stepwise result before each waypoint; counts will differ because earlier resolutions become the new base.
- Acceptance per waypoint: both builds succeed; the cheap gates of T-SYNC-04 (G1-G3) pass; the waypoint is recorded with its upstream tag, sha, conflict list and gate output.

### T-SYNC-03 Semantic re-verification of the section 2 items

A named check for each, run after W6 and, for items 1 and 2, also where they first appear (W3 and W4):

1. KV rework: layer-streaming K/V lookup and view patching, `n_pad` pin and graph-set key; run the PP-vs-stock identity arms and the TG decode-scan-length sweep (`build/stock-levers/tg-decode-scan-length.py`: ctx 50,176 / 128,000 / 262,144). Treat the gotchas "never mutate TG KV views" and "graph sets carry no `n_kv`" as binding.
2. MMQ ids slack: `compute-sanitizer --tool memcheck` on the server at `-ub 2048` with `--lm-prefill` and `--hybrid-moe` (command shape in the G31l/G32q AGENTS entries); `ERROR SUMMARY: 0 errors` required.
3. `test-moe-hybrid-dup-ids` (GPU, 80/80 expected, was 38 failures when unfixed) and a `--hybrid-moe-slots 16` run on a Q8_K_XL model to force the miss path.
4. `fattn-gqa8` against the #30573 fix: `test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=2,nr23=\[8,1\].*type_K=q8_0"` and the `fa-bench` rate.
5. Chunked CUDA graphs against the GPU spin-wait handshake and the dedicated copy thread: the 16-of-16 reproducer (`LLAMA_CUDA_GRAPH_RESET_EVERY=4 LLAMA_CUDA_DEBUG_FREE_AT_NODE=110`) plus a 4k hybrid decode soak.
6. Pascal: which `mmq-config-pascal-*` file sm_61 selects; dense Qwen3.8 prefill cells (3.7k, 16k) against the control.
7. Upstream MoE cache and MTP: confirm neither is active in a fork run that does not ask for it, and that fork flags and upstream flags do not share a name (`common/arg.cpp`).

### T-SYNC-04 Validation gates

Per the Gate Policy these are soft evidence; each proves only what it names. All GPU runs go through `tools/with-gpu-lock.ps1`, one at a time, with the nvidia-smi probe and the battery check (`tools/run-tg-arms.py` `power_state()`).

| Gate | Run | Expected |
|---|---|---|
| G1 build | Windows CUDA 13.2 and Linux GCC/CUDA 12.8, both | link clean, no new warnings in fork files |
| G2 host | the ctest command printed by `tools/run-tests.ps1 -Suite host` (that script halts at its doc-size guard, run the ctest line directly; Release config) | 78/78 (or the recorded count, stated if it changed) |
| G3 default-off | fork build with every fork flag off vs the pure-upstream build at the same tag: greedy tokens and first-token logprobs on 2 prompts, `test-backend-ops` failure list | identical tokens; no failure that upstream does not also show |
| G4 equivalence | the existing server harnesses at an equal chunk shape (`server_firsttok_matrix.py`, `server_hitfresh.py` with a frozen cache), `--lm-prefill-canonical` suites | within the recorded noise floors (AGENTS: first-token 0.0002 nats at chunk 512; hit-vs-fresh rule 0.25 nats) |
| G5 memory | compute-sanitizer memcheck (T-SYNC-03 item 2) | 0 errors |
| G6 semantic-QA | smoke tier 512; the 4k rung is owner/agent invoked, one length per mutex hold | aggregate checker PASS, negative control fails as designed |
| G7 performance | same-session A/B, control vs synced build, on the T-SYNC-01 cells | no cell more than 3% below the control (a different bar is a dated decision made before the run); upstream gains reported separately and not claimed as fork gains |

Claims allowed: exactly the gates' results for this machine, model files and flags. No speedup or quality claim for the sync as such.

### T-SYNC-05 Landing

- Updated `third_party/README.md` (new base tag and sha, fork delta numbers, tree-check row), `third_party/UPSTREAM_BASE`, `docs/currentstate.md` (new engine identity: source digest plus the gates' verdicts, remove superseded claims), `docs/reference/LEDGER.md`, and a line in `docs/reference/HISTORY.md`.
- Note for readers of older evidence: every number recorded before the sync was measured on the b10488-based engine; the sync does not retroactively change them.
- No push, no upstream PR (AGENTS: private fork). Commits are local and need the owner's go-ahead unless autonomous mode is on.

## 5. Decisions needed from the owner

- **D1 target.** Pin the tag current when work starts (today b12004, `e216f4ec8753` is 5 commits past it) and ignore later commits until the sync lands. Recommended.
- **D2 strategy.** Option A, six waypoints. Recommended; the alternative is B-style single jump (one 25-file conflict session, no bisect).
- **D3 upstream MoE cache and MTP.** Recommended: keep the fork engines as they are, compile the upstream ones in but leave them off, and file a separate comparison task afterwards (upstream's cache may match or beat `llama-moe-hybrid.cpp` on the 16 GB/8 GB cases; that is a measurement, not something to decide inside a merge). The alternative, replacing the fork engine with upstream's, changes the architecture and belongs in an SDD update.
- **D4 cadence.** Recommended: sync when a specific upstream fix or model is needed, not on a schedule; the `UPSTREAM_BASE` file and `tools/find-upstream-base.py` keep the next sync cheap.

## 6. What this plan does not know

- No merged tree has been compiled or run; the 25 conflicts and the section 2 list come from text, hashes and names only.
- The 70 hunks were counted, not read; real resolution cost may differ from their size.
- Section 2 item 5 (chunked CUDA graphs against the spin-wait handshake) is a suspicion from names and dates; no code path was traced.
- Performance after the sync is unknown in both directions: fork kernels may lose to or merge with upstream's new fusions.
- The dry runs use a blobless clone and the merge-base `04b569142da2` found by hash matching; a wrong base would show up as an implausibly large conflict set, and 3,403 of 3,488 files matching exactly argues against it.

## 7. Progress log

Owner approved D1-D4 as recommended on 2026-10-30 (pin b12004 `3c809f44383b`, six waypoints, keep the fork MoE/MTP engines, sync on need) and then, by `/goal`, authorised the implementation with a review and a local commit at each milestone (no push, no pull request). Work happens in this worktree (`claude/sync-llama-cpp-upstream-abe892`) and in the scratch repo `C:/Development/AI/llama-sync` (blobless clone of ggml-org/llama.cpp; branch `sync/w0` = the vendored tree on the true base, `sync/w1`, `sync/w2`, `sync/w3` = the merges, `rerere` on; the W3 merge is also checked out in `C:/Development/AI/llama-sync-w3`). The pure-upstream control is the worktree `C:/Development/AI/llama-up-b12004` (detached at b12004, its own `build-cuda`).

Milestones: M0 plan and base correction (commit `eba0e9e`); M1 T-SYNC-01 (controls built, baselines recorded); M2 W1 + W2 (b10991) in the tree; M3 W3 (b11311); M4 W4 (b11675); M5 W5 (b11850); M6 W6 (b12004) and landing.

### M1 / T-SYNC-01: controls and baselines (2026-10-30)

- **Finding F1 (not caused by the sync): fork HEAD did not build on Windows MSVC + CUDA 13.2.** `ggml/src/ggml-cuda/mmvq.cu(982)` used a `constexpr` of the enclosing generic lambda as a template argument inside a nested lambda (error C2672, "did not evaluate to a constant"). Introduced by `5ca1025` (2026-10-29, "Pascal matvec rows per warp"), evidently only built with GCC. Repaired by naming the value through `decltype(ncols_tag)::value`; no behaviour change. The control binaries are HEAD plus that line.
- **Control build**: `build/llama-cuda`, all targets except one: `test-unicode` fails to LINK (`unicode_regex_split` is not exported from `llama.dll`; `src/unicode.*` and the test are unmodified by the fork, so this is the shipped state of upstream on Windows shared builds). The binaries are side-copied to `build/llama-cuda/bin/sync-control/` (112 files, `llama.dll` sha256 `fd420574c1e9804e...`, `llama-server.exe` `0c2d42d02459e94a...`).
- **Pure upstream b12004** built in `C:/Development/AI/llama-up-b12004/build-cuda` (llama-server, llama-completion, llama-bench, llama-cli, test-backend-ops, llama-perplexity): clean.
- **Host suite on the control: 78/78** (the ctest regex of the `host` entry of `tools/run-tests.ps1`, run directly because that script halts at its doc-size guard).
- **GPU baselines** (RTX 5090 Laptop, mains, one hold of the GPU mutex, `build/llama-cuda/sync-baselines/`): `test-moe-hybrid-dup-ids` 80 checks, 0 failures; the FLASH_ATTN_EXT head-256 q8_0 GQA8 pattern OK; **`test-backend-ops` full run: the control fails 34 of 13,302 cases, pure upstream b12004 fails 167 of 19,209, and every failing case is `MUL_MAT` or `MUL_MAT_ID` with an `iq1_s`, `iq2_s` or `iq3_s` weight type** (plus, in upstream only, one `q4_1` case and two IQ3_S `ADD` cases that fan out of an IQ3_S matmul). So the fork adds no failure the pure upstream does not have; the IQ1/IQ2/IQ3-S matmul failure on sm_120 with CUDA 13.2 / MSVC predates the sync and is not part of it. The gate for every waypoint is therefore: no failing case outside the `iq1_s`/`iq2_s`/`iq3_s` MUL_MAT(_ID) family, and no IQ case that fails on the synced build but passes on pure upstream b12004 (`tools` in the scratchpad normalise the case strings; upstream added the `src_overlap,m_v,pad` fields).
- Decision (2026-10-30, reversible): no performance cells recorded now. The G7 comparison is a same-session A/B with the side-copied control, so a history-style baseline would be GPU time that G7 repeats. The dense Qwen3.8 GTX 1070 cells need the Linux/Pascal machine and are listed for the owner; they cannot be produced here.
- **Finding F9 (greedy text is not a cross-build gate on this model).** Three builds (synced, control, pure upstream) given the same 2,000-token prompt and `--temp 0` diverge at the first generated token; near-tied logits flip with any kernel change. The gate used instead is perplexity over the same text (`build/sync-ppl.sh`).

### M2 / W1 (b10729) + W2 (b10991)

W1 had 3 conflicts, W2 9 files / 19 hunks (the fixed-base dry run had said 3 and 8 new). Resolutions and what each one meant:

- **F2 (W1) upstream already ships the fork's G30.** #27621: the fused top-k router now ends every row's reads with `__syncthreads()` and lets the logits alias the outputs for up to `TOPK_MOE_ROWS_PER_BLOCK` = 8 rows. The fork's `LLAMA_CUDA_TOPK_MOE_SMALL` / `sync_rows` / `ggml_cuda_topk_moe_small_rows()` are removed; `topk-moe.cu` and `topk-moe.cuh` equal upstream. The environment variable is now ignored (the "MF" recipe in AGENTS may still set it; harmless).
- **F3 (W2) slot collision in `FLASH_ATTN_EXT` `op_params`.** Upstream's sparse flash attention keeps `n_kv_max` in `op_params[4]`; the fork kept its causal-window flag, `q_pos_base` and `n_kv_valid` in `[4]`, `[5]`, `[6]`. A causal op would have been read as `n_kv_max = 1`, with no compile error. The fork's three slots moved to `GGML_FLASH_ATTN_EXT_CAUSAL_PARAM` (12) .. 14, a macro in `ggml.h`; the writer (`ggml.c`), both readers (`ggml-cpu/ops.cpp`, `fattn-common.cuh`) and the two FAFIX gates in `fattn.cu` follow it.
- `fattn-mma-f16.cuh`: upstream's loader signature (K offset argument, unshifted mask base, `use_sparse`, `indices`); the fork's mask-free causal block now reads the K offset from `k_VKQ_0`; its `|| stride_mask < 0` conditions stay; its extra mask load in the multi-stage branch is compiled only for the non-sparse instantiation.
- **F4 (W2) `hparams.n_expert_used` became a method** (`n_expert_used(il)`, per-layer). 18 reads in `llama-layer-streaming.cpp` and `llama-moe-hybrid.cpp` now call it (`il` defaults to 0; the Qwen MoE models have a uniform top_k); the T-GPLR-82 `n_rows` clamp in `build_moe_ffn` uses the per-layer value. `ggml_mul_mat_set_prec` is deprecated upstream: the fork's one call uses `ggml_prec_set_acc`.
- `qwen35.cpp`, `qwen35moe.cpp`: upstream's `build_qkv`, fed with the fork's `layer(il)` (the streamed layer package) instead of `model.layers[il]`; tensor names (`Qcur`, `Kcur`, `Vcur`) are unchanged.
- `server-context.cpp` (W1): upstream moved the draft/MTP memory reservation into `common_fit_extra_model`; `--hybrid-moe` turns the fit step off, so the fork's own measurement (and `spec_fit_added`) stays but runs only when `hybrid_moe` is set (otherwise the memory is counted twice). Needs a GPU check of the 16,000 MiB margin with `--spec-type mtp` (G7/G8).
- `test-backend-ops.cpp`: `test_flash_attn_ext` takes upstream's `kv_view, v_is_view_of_k, n_kv_max` at positions 13-15 and the fork's `causal, q_pos_base` after them. **Four fork call sites outside the conflict hunks passed `causal, q_pos_base` positionally** and would have bound to `kv_view` / `v_is_view_of_k` without a compile error; moved.
- **F5 `small` macro**: `llama-layer-streaming.cpp` has a lambda variable named `small`; an upstream header now reaches `<windows.h>` before the file's `WIN32_LEAN_AND_MEAN`, so rpcndr.h's `#define small char` is live. Renamed. **F6** the MSVC repair of F1 had to be applied again (the line moved to `mmvq.cu:1174`).
- **Trap T1: `src/llama-layer-streaming.cpp` is stored with CRLF line endings in the repository blob** (all 12,997 lines). Edit scripts that match multi-line text must use `\r\n` for that file.
- **Trap T2: never chain a destructive step behind a pipeline.** A `git fetch ... | tail` hid a failed fetch (the blobless scratch clone cannot serve its own history to another repository: "bad pack header") and the following `git rm -r third_party/llama.cpp` emptied the vendored directory; restored from HEAD at once, nothing lost. The export now goes through a parentless commit of the tree (all its blobs are present) and checks the tree id between steps.

Fork delta after W2 against b10991: 146 files, +58,376 / -307 (was 148, +58,337 / -314 against b10488); gone: `topk-moe.cu`, `topk-moe.cuh` (superseded), `mmq-config-pascal.cuh` (renamed upstream to `-pascal-dp4a.cuh`, content carried). 262 lines of fork code differ from before (the review below).

**M2 gates (W2 tree = vendored tree):** G1 build clean (MSVC 19.5, CUDA 13.2, sm_120, all targets, `LLAMA_BUILD_UI=OFF` in the scratch build); G2 host suite 78/78; G3 default-off equivalence: perplexity of the same 8 x 512 tokens of English text with Qwen3.6-35B-A3B Q4_K_S (`-ngl 99 -ncmoe 12`, no fork flag): **synced 10.8386, control 10.8056, pure upstream b12004 10.8482** (each +/- 0.65; spread 0.4%); `test-moe-hybrid-dup-ids` 80 checks 0 failures; the head-256 q8_0 GQA8 FLASH_ATTN_EXT cases OK; `test-backend-ops` 16,100 of 16,156 pass, all 56 failures in the IQ-quantisation family (below), none elsewhere. **Finding F10: the IQ-quantisation matmul failures are nondeterministic on this machine.** Three runs of the SAME control binary on `MUL_MAT type_a=iq1_s..iq3_s` fail 28, 26 and 26 cases with different sets, so the family (iq1_s, iq2_s, iq3_s, iq2_xs, iq3_xxs, iq4_nl weights; one `LIGHTNING_INDEXER` case also fails on the control and on pure upstream; one `q4_1` case fails on pure upstream only) is the unit of the gate, not the case: `tools/sync-tbo-compare.py` exits 1 if the synced build fails anything outside it that fails on no other build. (The cause is not investigated; it predates the sync, shows on pure upstream, and does not touch the Q4_K_S / Q8_K_XL models of the fork's work. For the dense Qwen3.8 UD-IQ3_XXS model on a Blackwell card it may matter: owner information.) G4-G8 (server equivalence, memory, QA ladder, performance) are deferred to the end of the sync, when the server and runner paths of W3 (batch API) are in.

Review of M2 (diff of the fork delta before and after the re-base, 262 differing lines in 14 files): every differing line is one of the resolutions listed above; nothing else in fork code changed. Not covered by a gate yet: the FAFIX / causal-window path at runtime (`ggml_flash_attn_ext_set_causal` with the moved slots has host-test coverage only through `test-backend-ops` cases, which pass), `--spec-type mtp` with `--hybrid-moe` (server, W3).

### M3 / W3 (b11311, 818 commits from the fork point)

8 files, 14 hunks (the fixed-base dry run said 5 new files); the tree builds after one further compile fix.

- **F7 `llama_batch_ext` (#24669, 2026-09-24).** `llama_context::encode/decode` take a `llama_batch_ext`; `decode(const llama_batch &)` is a wrapper (`llama_batch_compat`). The fork's hooks in `decode()` (profile counters, the layer-major pass `lm_decode_begin`, the hybrid prefill window hook) moved behind `balloc->init()` and look at the sanitised legacy view `balloc->get_batch()` (positions filled in; `lm_make_plan` checks `pos[i] == p0 + i` and the logits flags, which the view carries). The three fork call sites of `llama_batch_allocr::init` (`layer_streaming_prebuild`, `layer_major_pp`, the pass chunks in `llama-lm-prefill.cpp`) build an ext batch from their legacy batch with `llama_batch_compat`. `llama_decode` / `llama_batch_init` stay in the public API, so `tools/layer-streaming/main.cpp` is unchanged.
- **F8 double-buffered graph result.** `gf_res_prev` is `std::array<llm_graph_result_ptr, 2>` indexed by `n_outputs > 0`, with `gf_res_prev_active`. The fork's single-pointer uses (drop of the graph reuse in the dense ring and the lm-prefill pass, the layer-major pass publishing its last chunk's result tensors, the first-token / top-k readers) go through `reset_graph_reuse()` (both buffers + the marker) and `get_gf_res_prev()`.
- **Positional initialiser.** `llm_graph_params` takes upstream's `prec_policy` before the fork's `layer_streaming`; `llama_context::graph_params` follows (AGENTS: new members are positional).
- `common_batch` (server, speculative): `batch.view.size()` for the G42 hooks, `batch_in.tokens[i].pos[0]` in the MTP deferral. `common_params::hostname` became `hostnames` upstream; the fork's parameter block stays, its `hostname` line is dropped.
- The `mmvq.cu` repair of F1 and the `small` rename of F5 were applied again (the W2 branch had them as commits; W3 forked before them).

**M3 gates (W3 tree = vendored tree):** G1 build clean (all targets, MSVC/CUDA 13.2); G2 host suite 78/78; G3 perplexity of the same text: **synced 10.7845, control 10.8056, pure upstream b12004 10.8482** (each +/- 0.64; spread 0.6%); `test-moe-hybrid-dup-ids` 80 checks 0 failures; the head-256 q8_0 GQA8 cases OK; `test-backend-ops` fails only inside the IQ-quantisation family (`tools/sync-tbo-compare.py` exit 0). **Server smoke (the paths the batch API port touches):** `llama-server` with the hybrid decode environment (experts on the host, VRAM expert cache) and `--lm-prefill`, `tools/server_firsttok.py` on two prompts of 1,600 and 3,000 tokens, control vs synced: the layer-major pass ran in both (`lm-prefill: pass 1: 1084 tokens`, the 1,600-token prompt cut at the 516-token checkpoint tail), top-1 token identical in all 4 pairs, first-token logprob difference max 0.0435 (1,600 tokens) and 0.0036 (3,000 tokens), KL max 0.0019 / 0.00025 - inside the noise floors recorded in AGENTS (control vs synced hybrid median 0.019, max 0.124 nats). Observation, not a result: the first pass of a fresh process took 3.16 s in the synced server against 1.77 s in the control and reported a 331 MiB footprint beyond the window against 229 MiB; one cold run each, to be looked at in G7.

Review of M3 (74 lines of fork code differ from before, 8 files): every line is a port listed above; nothing else in fork code changed.


## 8. Escalation

If a waypoint cannot be built after the Implementer Workflow's classify-and-repair loop and one documented online pass (upstream issues and PRs for the touched files), stop that waypoint, record the root cause and the blocking files, and return to the owner with options B/C/D from section 3. In autonomous mode this becomes a recorded decision resolved by measurement or a probe, never by guessing.
