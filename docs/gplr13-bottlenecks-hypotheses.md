# T-GPLR-13 Ladder Bottlenecks and Solution Hypotheses

Status: active findings file (created 2026-09-13). Future runs of the
T-GPLR-13/T-GPLR-15 program append dated sections here; do not rewrite older
sections. Unit convention: host commit/available figures in this file are
decimal GB (Python /1e9); the PowerShell ladder driver's table reports the
same mem.csv bytes as GiB (2^30) — e.g. 200k commit 25.75 GB = 24.0 GiB =
25,754,857,472 B. Audit close-out 2026-09-13: fixes R1-R3 are closed —
dense ctest pair 2/2 under the mutex, validator parity (window_tokens rule
in both validators), dual-path negative controls 18/18; the audit ruled the
never-fired reuse branch NOT a T-GPLR-13 defect, so H1 remains the owner of
its first positive demonstration. Source data: `build/llama-cuda/gplr13-ladder/` (traces, `mem.csv`
per size, `summary.json`, `analysis-stats.json` produced by the one-off
aggregator `build/analyze_gplr13_ladder.py`). Read-only analysis of preserved
artifacts; no new GPU runs and no library changes were made for this file.

Scope note: three TG rounds per run (greedy, `--tg-tokens 3`), one prompt
family, one model (Qwen3.6-35B-A3B-MTP-UD-Q4_K_S), idle shared GPU (a foreign
deepseek_v4.exe exited before the recorded runs). Every "rate" below is a
wall-derived or trace-derived measurement, labeled by its denominator.

## 1. Cross-size statistics (row-set ladder, 2026-09-13)

| metric | 50 | 512 | 4k (4001 tok) | 200k | qa-smoke-512 (off) |
| --- | --- | --- | --- | --- | --- |
| run exit / checker | 0 / OK | 0 / OK | 0 / OK | 0 / OK | 0 / OK |
| events | 569 | 569 | 663 | 4,469 | 23,326 |
| PP wall s (tok/s) | 5.508 (9.1) | 5.675 (90.2) | 6.663 (600.5) | 147.51 (1,355.8) | 5.626 (91.0) |
| PP compute s / transfer-phase s | 0.13 / 21.08 | 0.22 / 21.50 | 1.04 / 24.24 | 132.67 / 411.36 | 0.23 / 5.40 |
| PP wall staging bandwidth GB/s | 3.33 | 3.23 | 2.75 | (compute-bound) | 3.39 |
| PP ready-latency share of transfer-phase | 74.6% | 75.3% | 78.1% | 98.7% | n/a (batch-major) |
| TG wall s (tok/s) | 0.2969 (10.10) | 0.3198 (9.38) | 0.3150 (9.52) | 0.3913 (7.67) | 182.54 (0.526, 96 tok) |
| TG ms/token | 99.0 | 106.6 | 105.0 | 130.4 | 1,901.5 |
| staged bytes/round (of 18,327,011,328 bank) | 572,719,104 | 572,719,104 | 572,719,104 | 572,719,104 | n/a (full-bank TG) |
| staged %/round | 3.125 | 3.125 | 3.125 | 3.125 | 100% |
| rows fetched/round (of 320 routed) | 320 | 320 | 320 | 320 | n/a |
| rows reused/round | 0 | 0 | 0 | 0 | n/a |
| rowset_fetch count | 120 | 120 | 120 | 120 | 0 |
| fetch transfer_us mean (sum) | 1,835 (220.2 ms) | 1,932 (231.8 ms) | 1,918 (230.2 ms) | 2,273 (272.8 ms) | n/a |
| fetch host_submit_us mean (sum) | 1,732 (207.8 ms) | 1,826 (219.1 ms) | 1,811 (217.3 ms) | 2,166 (259.9 ms) | n/a |
| fetch ready_us mean | 103.8 | 106.0 | 107.0 | 107.1 | n/a |
| TG wall staging bandwidth GB/s | 5.79 | 5.37 | 5.45 | 4.39 | n/a |
| runner peak_vram_bytes (MiB) | 5,949,161,472 (5,674) | 6,005,784,576 (5,728) | 6,077,087,744 (5,796) | 10,214,768,640 (9,742) | 4,560,846,848 (4,350) |
| poller NVIDIA dedicated peak (MiB) | 4,803,440,640 (4,581) | 4,860,080,128 (4,635) | 4,931,391,488 (4,703) | 9,069,494,272 (8,649) | 3,425,615,872 (3,267) |
| runner-minus-poller gap | 1,145,720,832 | 1,145,704,448 | 1,145,696,256 | 1,145,274,368 | 1,135,230,976 |
| shared (non-local) peak MB | 69.5 | 77.9 | 82.1 | 281.3 | 77.9 |
| host commit peak GB | 18.53 | 18.61 | 19.06 | 25.75 | 17.25 |
| avail-phys min GB (start) | 39.66 (51.29) | 41.21 (59.01) | 38.73 (59.00) | 35.86 (58.97) | 37.95 (59.00) |
| TG `layer_compute` events with nonzero timings | 0/120 | 0/120 | 0/120 | 0/120 | n/a |

Normalizations used: 572,719,104 B/round = 40 layers x 8 rows x 3 members
x 589,824 B/row (per-layer row bytes differ on 3 layers but sum to this
constant). Staged-vs-full-bank = 32.0x byte reduction vs the off-path
full-bank TG staging (qa-smoke: 1,759.4 GB / 96 tokens = 18.328 GB/token).

## 2. Route-overlap and resident-hit statistics (T-GPLR-15 gate input)

Method: per layer, consecutive TG-window `union_ids` sets (W=1, top_k=8,
40 layers x 2 consecutive pairs x ... = 80 pairs per run); hash-equality is
the exact lower bound, containment `|A n B| / |B|` is exact (full lists).

| statistic | 50 | 512 | 4k | 200k |
| --- | --- | --- | --- | --- |
| consecutive-window hash-equal rate | 0/80 | 0/80 | 0/80 | 0/80 |
| containment mean (consecutive windows) | 0.377 | 0.370 | 0.389 | 0.358 |
| rows_reused reported by runner | 0 | 0 | 0 | 0 |

Provenance caveat (2026-09-13; corrected the same day after verification):
the two statistics above are MACHINE-computed by
`build/analyze_gplr13_ladder.py`, the untracked one-off already named in
section 1. That script lives INSIDE the gitignored `/build/` tree
(`.gitignore:/build/`), so it does not survive a clean checkout and the
numbers are not reproducible from tracked tooling alone — the regeneration
command in section 5 works only while that untracked file is present. The
tracked replacement is the T-GPLR-15 Phase A re-derivation in
`tools/route_overlap/analyze.py`, which becomes the producer of record. The
containment direction is owned by the SDD section 10 gate 9 decision rule
(`|A n B| / |B|`, next window in the denominator).

Honest thinness: 80 pairs per size, 3 rounds, one prompt family, W=1 only.
This is the first measurement of T-GPLR-15's 0.6 temporal hit-rate gate and
it is **below the gate at every size (mean 0.358-0.389)** - but the 0.6 gate
is defined over a real generation stream, not 2 consecutive steps; treat this
as a negative pilot signal, not a gate verdict.

Structural finding (explains rows_reused=0 at nonzero containment): the
executor rotates 40 layers through 4 slots (`layer % 4`), so slot 0 hosts
layers 0,4,8,...,36 within each window and each layer's next visit comes 9
layers later - the (slot, owner) pair always changes between a layer's
consecutive visits, the tracked coverage resets, and reuse can never fire
regardless of overlap. Evidence: `rowset_fetch` stream shows slot 0 visited by
10 distinct layers per window at generation 11 (trace rows
`"ubatch":2..4,"slot":0` at 50); `rows_reused=0` in all 12 `rowset_round`
events across the ladder while mean containment is ~0.37 (37% of each
window's rows were resident somewhere a cycle ago, but not in that slot).
A reactive row-set with per-layer slots (or persistent per-layer row caches)
would have realized roughly 0.37 x 3.125% ~= 1.2% of full-bank bytes in
reuse at these 3 steps.

## 3. Discrepancy analysis (evidence-cited; explained vs open)

### 3.1 The ~1.09 GB runner-vs-poller VRAM gap - EXPLAINED to first order (mechanism detail open)

The gap is **constant to 0.05%** across four row-set runs whose device
footprint differs by 4.3 GB: 1,145,720,832 / 1,145,704,448 / 1,145,696,256 /
1,145,274,368 B (= 1,092.3-1,092.7 MiB), and 1,135.2 MB on the shape-different
qa-smoke run. T-MEMPROBE-01 measured 1,082.7 MiB with the same instruments on
a different day and model - agreement within ~1%. A gap that constant across
footprints is a **fixed-size allocation class that the free-VRAM delta counts
but the WDDM dedicated-usage counter does not report at sample time**, not a
scaling leak and not sampling jitter (sampling error would not track to
<0.5 MB). The leading candidate remains the T-MEMPROBE-01 explanation (CUDA
context/runtime class visible to `cudaMemGetInfo`, excluded or transient in
the adapter counter); its exact counter-level mechanism is OPEN.
Proposed measurement (not run): in-process `cudaMemGetInfo` at phase
boundaries plus a same-instant adapter-counter read, isolating the delta at
CUDA-init time before any model allocation.

### 3.2 TG tok/s decline 10.10 -> 7.67 - EXPLAINED: staging-bound, context-independent staging rate

Per-token TG wall grows 99.0 -> 130.4 ms while the staged bytes per token are
constant (572.7 MB). Decomposition per token (40 fetches, sums): fetch
transfer 73.4 -> 90.9 ms, host submit 69.3 -> 86.6 ms, ready latency ~4.2 ms.
Staging transfer alone is 74-70% of the token at 50 and 200k. Submit+transfer
sums exceed the wall at every size (142.7 vs 99.0 ms at 50; 177.5 vs 130.4 ms
at 200k), so submit and transfer already overlap across layers (the overlap
ratio is flat-to-improving, 1.44 -> 1.36); the slowdown is therefore carried
by the transfer term itself (+17.5 ms/token) plus submit (+17.3 ms/token)
against a near-flat residual (token minus fetch-transfer sum: 25.6 -> 39.5
ms): the **effective staging bandwidth declines 5.79 -> 4.39 GB/s** (fetch
transfer_us mean 1,835 -> 2,273 us for identical bytes). The
attention-over-longer-KV cost is not separately measurable because every TG
`layer_compute` carries zero timings (see 3.6); the residual growth bounds
any KV effect at ~1.4x smaller than the staging-rate decline. The 200k fetch
slowdown plausibly shares its cause with the 200k PP transfer-phase slowdown
(411 s for the same 18.33 GB the small runs move in 21-24 s of
transfer-phase time) - see 3.5.

### 3.3 Shared segment 66 -> 268 MiB vs dedicated +4.1 GB - CONFIRMED: no duplication

Dedicated grew 4,803.4 -> 9,069.5 MB (+4,266.1 MB, matching the ~200k KV
estimate + activation slack) while the shared (non-local) segment grew only
69.5 -> 281.3 MB (+211.8 MB, 5.0% of the dedicated growth). If the driver
mirrored device allocations, shared would have grown by the same ~4.3 GB.
The T-MEMPROBE-01 no-duplication conclusion extends to the 200k regime.

### 3.4 Commit growth 17.3 -> 24.0 GB at 200k - PARTLY EXPLAINED, residual OPEN

System commit peak grew +7.22 GB (18.53 -> 25.75 GB) while the q8 KV at
200k accounts for +2.18 GB (10 full-attn layers x 2 tensors x 544 B/token
x ~200,003 tokens). The ~5.0 GB residual cannot be attributed from these
artifacts, and cross-run commit deltas are polluted by background processes:
the avail-phys baselines differ between runs by up to 7.7 GB (51.29 vs
58.97 GB at start), i.e. other processes' commit moved between runs. The
927 KB anchor file itself is negligible; the mmap page cache is file-backed
and does not bill to commit. OPEN. Proposed measurement: rerun one size with
a per-process commit counter (`Win32_PerfFormattedData_PerfProc_Process`)
for the runner pid only, sampled at phase boundaries - this also serves
hypothesis H7.

### 3.5 New: 200k PP transfer-phase rate collapse - EXPLAINED as first-touch page-in, hidden by overlap

PP `copy_complete` ready-latency at 200k totals 405.9 s of the 411.4 s
transfer-phase time (98.7%), i.e. submits wait ~10 s each for the copy engine
- the signature of mmap first-touch page faults streaming the 18.33 GB of
banks from disk at ~45.2 MB/s, matching the recorded ~52 MB/s effective
page-in in AGENT_ENVIRONMENT.md. It does not extend wall time because PP is
compute-bound at 200k (compute 132.67 s of wall 147.51 s; staging fully
overlapped). Caveat per docs/model-run-testing-standard.md: the mmap source
state is uncontrolled Windows file cache; the later ladder runs were
progressively warmer, so treat the absolute page-in number as an upper
bound. The small-size runs show the same mechanism mildly (74.6-78.1%
ready share of transfer phase).

### 3.6 New: TG compute timings are all zero - OPEN (instrumentation gap)

All 120 TG `layer_compute` events per run carry `compute_us=0, setup_us=0`
(PP events are properly timed). The schema draft's open decision #3
anticipated the split-graph timing question. Consequence: the TG
decomposition in 3.2 cannot separate FFN/attention compute from scheduler
overhead; only staging-vs-residual attribution is possible. Fix belongs to
telemetry (time the two graph parts around `load_row_set`), not to this
analysis.

### 3.7 New: PP floor ~5.5 s is staging-bound; per-token PP cost is amortization of one full-bank pass

PP wall is 5.508 s (50), 5.675 s (512), 6.663 s (4k) while PP compute is
0.13/0.22/1.04 s: the 18.33 GB expert-bank pass (3.2-3.3 GB/s wall rate)
dominates until 200k where compute overtakes. Hence the PP tok/s pattern
9.1 -> 90.2 -> 600.5 -> 1,355.8: tokens are nearly free; the pass is the
cost. Note the TG row-set path stages the same byte volume per 3 tokens that
PP stages once - the row-set fetch pipeline sustains 4.4-5.8 GB/s wall
vs PP's 3.2-3.3 GB/s whole-bank pipeline, consistent with the row-set's
smaller, deeper-pipelined copies vs 453 MB whole-bank copies with one slot
of lookahead.

### 3.8 New: the off-path (batch-major, full-bank TG) contrast quantifies the row-set win on this shape

qa-smoke-512 (row-set off, 96 TG tokens) stages 1,759.4 GB in TG
(18.328 GB/token, exactly the full bank) at 1.901 s/token; the row-set path
stages 0.5727 GB/token at 0.099-0.130 s/token on the same model. Byte
reduction 32.0x; wall-per-token improvement 14.6-19.2x. Shapes differ
(batch-major vs layer-major PP, 96 vs 3 TG tokens), so cite as a same-model
contrast, not a controlled A/B.

## 4. Ranked solution hypotheses

Each: observation -> candidate cause -> how to prove/disprove -> candidate
fix. Ranked by (expected TG impact) x (evidence strength). No new GPU runs
were executed for this file; proposed measurements are listed, not run.

### H1 - Kill the per-layer slot rotation so reactive reuse can fire (highest value, strongly evidenced)

Observation: `rows_reused=0` at all sizes while 37% of each window's rows
were resident one window earlier (section 2); rotation makes reuse
structurally impossible (3.2 of section 2).
Candidate cause: `layer % n_slots` slot assignment forces an (owner change)
coverage reset between consecutive visits of the same layer.
Prove/disprove: trace-level - recompute the reuse the checker would credit if
coverage were per-(layer) instead of per-slot (pure offline re-analysis, no
GPU run); design-level - a 4-layer-strided schedule variant or per-layer row
caches. Disprove if per-layer simulation also shows ~0 reuse (it will not:
containment is 0.358-0.389).
Candidate fix: row-set generation keyed by logical layer, not slot - either(a) assign each MoE layer a fixed row-cache (VRAM cost 40 x 453 MB worst case ~= the full banks, defeating the purpose), or (b) keep 4 slot buffers but
make coverage tracking per-logical-layer with copy-on-revisit of only the
missing rows (the current buffers are already full-bank-shaped, so the
rows persist in the buffer; only the bookkeeping discards them). (b) is a
library change inside the row-set path - needs a task contract;expected gain at these 3 steps ~1.2% of full-bank bytes/round (14.3 -> ~9.0 MB staged per layer-window), i.e. staging 3.125% -> ~2.0%, token wall ~99 -> ~85 ms at the measured 5.4-5.8 GB/s.

### H2 - Restore/raise TG staging bandwidth at long context (staging-bound today; strongly evidenced, cause open)

Observation: fetch transfer mean 1,835 -> 2,273 us for identical 14.15 MB
batches (5.79 -> 4.39 GB/s) from 50 to 200k; staging is 70-74% of token wall
(3.2).
Candidate cause: (i) KV growth steals L2/copy-engine bandwidth or forces the
copy engine to contend with attention traffic; (ii) allocator fragmentation
of the staging source pages at long runs; (iii) power/clock state.
Prove/disprove: re-run 200k with `--layer-slots 1` and `--layer-slots 8`
(slot-count sweep isolates buffer pressure); compare fetch transfer_us
distributions; pair with in-process `cudaMemGetInfo`/clock sampling
(proposed, not run). Disprove (i) if rates recover with smaller KV at equal
run length (e.g. 200k prompt, q8 vs f16 KV).
Candidate fix: none in this pass; if (i) confirmed, the fix space is
scheduling (stage during the attention part of the same layer's graph,
which the two-part graph split already enables) rather than copy tuning.

### H3 - Explain why host submit grows with context at all (submit is CPU-side; it should be context-independent)

Observation: host_submit sum grows 207.8 -> 259.9 ms (+25%) from 50 to 200k,
tracking the transfer-term growth almost 1:1 - but submit is CPU-side
per-fetch work on identical 14.15 MB batches and should not depend on KV
length. Cross-layer overlap is already effective (3.2: submit+transfer sum /
wall = 1.44 at 50, 1.36 at 200k), so this is not a lost-overlap problem.
Candidate cause: submit timing absorbs copy-queue backpressure (submit waits
when the copy engine is saturated - e.g. contended by attention traffic at
long KV), so the +25% is a symptom of the H2 slowdown measured at a
different point, not an independent bottleneck.
Prove/disprove: reconstruct the per-fetch submit/transfer timeline offline
from `t_us` deltas already in the traces (proposed, no GPU run); if submit
growth appears only as queue-wait coincident with transfer growth, H3
merges into H2 and carries no separate fix.
Candidate fix: none independent of H2; deeper staging queues would only help
if the timeline shows genuine serialization (it currently does not).

### H4 - Widen the TG window (W>1) to amortize the fixed 40-layer staging cadence

Observation: W=1 pays 40 fetch round-trips per token; ready latency is
negligible (103-107 us) but per-fetch submit is ~1.8-2.2 ms x 40.
Candidate cause: fixed per-fetch overhead dominates over bytes at W=1.
Prove/disprove: schema draft 4.2 bounds W>1 (`top_k <= route_union <=
min(256, W*8)`); a W=4 probe would stage at most ~min(256,32)=32 rows/layer
vs 40 x 8 = 320 row-visits - expected ~4x fewer fetch calls at <=4x rows
each; verdict from `rowset_round` byte ratio vs token wall. Requires
implementing W>1 (currently fails loudly by design; schema 4.2 checker rule
exists).
Candidate fix: window_tokens>1 implementation (task-sized, not this pass).
Expected gain: bounded by staging share (~70%); if fetch-call overhead
scales with calls, up to ~2-3x TG at long context; the union volatility
data to justify it is exactly the section 2 stream (0% exact repeats,
0.358-0.389 containment - volatility is high, so W>1 trades byte growth
against call overhead; needs the probe).

### H5 - Reduce PP whole-bank paging cost for short prompts (PP-bound below ~10k tokens)

Observation: PP wall floor ~5.5 s for 18.33 GB (3.2-3.3 GB/s wall) while TG
row staging sustains 4.4-5.8 GB/s (3.7); PP tok/s at 50/512 is 9-90.
Candidate cause: 453 MB whole-bank copies with one-slot lookahead are
latency-bound (74.6-78.1% of transfer-phase is ready latency) vs the row
path's small deep-pipelined copies.
Prove/disprove: offline - compare `copy_complete` ready-latency distribution
vs copy size across PP (453 MB) and TG (14 MB); a run with `--layer-slots 8`
would double lookahead (proposed, not run).
Candidate fix: increase PP prefetch depth, or (aligned with the SDD) page
PP as expert subsets once routing is known per chunk - larger change, own
task.

### H6 - Attribute the constant 1.09 GB VRAM accounting class (accuracy, not speed)

Observation: constant 1,145.3-1,145.7 MB runner-vs-poller gap (3.1).
Candidate cause: CUDA context/runtime class counted by the free-VRAM delta
but not visible to the WDDM dedicated counter at sample time (T-MEMPROBE-01
candidate).
Prove/disprove: in-process `cudaMemGetInfo` sampled at CUDA-init (before
model allocs), after init, at phase boundaries, at teardown, paired with
same-instant adapter reads; the gap's onset timing isolates the class
(proposed; also serves the T-MEMPROBE-01 owner decision (a) vs (b)).
Candidate fix: relabel `peak_vram_bytes` (option (b), docs-only) or sample
true peaks at phase boundaries (option (a), small library change).

### H7 - Attribute the ~5 GB non-KV commit growth at 200k (safety margin question)

Observation: commit peak +7.22 GB from 50 to 200k vs +2.18 GB KV (3.4);
residual ~5 GB unattributed and cross-run baseline pollution admitted.
Candidate cause: background-process drift (baselines differ by up to 7.7
GB), plus possibly per-token logits/scratch at PP tail, KV metadata, or
allocator arena growth.
Prove/disprove: per-process commit counter on the runner pid across one
50-vs-200k pair (proposed; pairs with H6 instrumentation).
Candidate fix: none if background drift; if runner-owned, bound it before
larger-context runs (the T-GPLR-09 RAM admission rule works on avail-phys
and is unaffected either way - measured avail stayed >= 35.9 GB).

### H8 - Fix zero TG compute timings (telemetry debt; unblocks H2/H3 attribution)

Observation: all TG `layer_compute` timings zero (3.6).
Candidate cause: the split graph (attn+router part / FFN part around
`load_row_set`) never records per-part timings.
Prove/disprove: n/a (read the builder/executor code - out of scope here).
Candidate fix: emit per-part `layer_compute` (or `rowset_fetch`-adjacent)
timings per schema draft open decision #3; small telemetry change; required
before H2's KV-vs-staging decomposition can be made exact.

## 5. Method and reproducibility

- Statistics: `python build/analyze_gplr13_ladder.py` (stdlib-only, read-only)
  regenerates `build/llama-cuda/gplr13-ladder/analysis-stats.json`; the tables
  above are transcribed from its output plus targeted grep of the traces
  (slot-0 stream, `rowset_round` lines, `phase_summary` fields).
- VRAM/RAM: from each size's `mem.csv` (0.5 s sampling, adapter-level
  counters authoritative per the MS KB caveat recorded in T-MEMPROBE-01).
- Limitations: 3 TG rounds per run; W=1 only; one prompt family; greedy
  decoding; poller granularity 0.5 s; file-cache state uncontrolled
  (model-run-testing-standard).

## 6. TG row-set staging decomposition: ready-latency share is ~5%; the transfer term is a submit-side pageable-staging loop — fix proposal + H1(b) correction (2026-09-13)

Append-only analysis of the preserved ladder artifacts plus a read of the
executor code (`llama-layer-streaming.cpp:load_row_set`) and NVIDIA's
documentation. No GPU runs, no code changes. All "expected" figures below are
hypotheses, not results.

### 6.1 The ready-latency share question — TG and PP have different answers

- **TG row-set fetch: ready latency is NOT the story.** `ready_us` mean is
  103.8-107.1 µs against a `transfer_us` mean of 1,835-2,273 µs — a **~5-6%
  share**. `transfer_us` is defined as submit + ready, and submit carries
  **~94-95%** of it (host_submit 1,732/1,826/1,811/2,166 µs at 50/512/4k/200k).
  The PP table's 74.6-98.7% ready share belongs to a different path
  (`load_layer` whole-bank copies; first-touch page-in per 3.5). For TG the
  analogous share is the **submit share**, and it is the target.
- Decomposition at 50 (per token, 40 fetches): submit 69.3 ms + ready 4.2 ms
  of wall 99.0 ms (~74%); residual (part-A/part-B compute, D2H ID read, graph
  launch) ~25.5 ms. Same shape at 200k (submit 86.6 ms of wall 130.4 ms,
  ~70%; residual 39.5 ms).

### 6.2 What the submit loop actually is (code-verified)

`load_row_set` issues **24 `ggml_backend_tensor_set_async` calls per fetch**
(3 members × 8 expert slabs × ~589,824 B ≈ 14.15 MB) **directly from
mmap-backed source pointers** — it never uses `slot.buf_staging` (only the
whole-bank `load_layer` path consults it). The sources are file-backed
pageable memory. Documented CUDA behavior: an H2D `cudaMemcpyAsync` from
pageable host memory *may be synchronous with respect to the host*, with the
driver staging the pageable source through an internal pinned bounce buffer;
true asynchrony requires pinned sources (CUDA driver API "API synchronization
behavior > Memcpy > Asynchronous"; CUDA programming guide "Asynchronous
Execution"). Evidence fit: steady-state host rate = 14.15 MB / 1.73-2.27 ms ≈
**6.5-8.2 GB/s** (memcpy-class), and `ready_us` ~104 µs ≈ the tail DMA of the
last call (per-call DMA overlaps the next call's staging). The +25% submit
and transfer growth at 200k (H2/H3) rides this same term; whether it is
driver-path overhead growth or file-cache pressure on the mapped sources is
separable only by the F1 A/B below. H3 merges into this mechanism, as its
timeline reconstruction already suggested.

### 6.3 Ranked fixes (expected-gain = hypothesis; each needs its own task contract)

- **F1 — explicit pinned row staging (highest leverage for TG staging time).**
  Allocate one pinned host buffer of one row window (~14.2 MB; fetches are
  strictly serial because `load_row_set` host-blocks on `ev_copy_ready`, so a
  single shared buffer suffices; alternatively reuse the already-allocated
  full-bank-shaped `slot.buf_staging` when `use_pinned_staging` is on). Per
  member: `memcpy` the 8 slabs mmap → pinned (disjoint per-member slices),
  then issue the DMA from pinned with byte-exact offsets — one async copy per
  member (or one batched submission, CUDA 12.8+ `cudaMemcpyBatchAsync`),
  events and lifecycle unchanged, staged bytes and telemetry schema unchanged,
  quantization untouched. Expected effect: the host rate becomes an explicit
  memcpy (~10-15 GB/s from file cache) overlapping per-member DMAs at pinned
  PCIe rates (~20+ GB/s), against the driver's serialized ~6.5-8.2 GB/s
  bounce path → staging term 1.8-2.2 → ~1.0-1.4 ms/fetch, token wall ~99 →
  ~75-85 ms (≈1.2-1.4x TG tok/s) **if** the submit mechanism is confirmed as
  dominant. Falsifiable probe (cheapest first): 512-rung A/B under the GPU
  mutex (`tools/run-gplr13-rowset-probe.ps1` shape); success = submit+memcpy
  mean < ~1.0 ms at identical staged bytes, hash-recomputed route parity,
  check-telemetry OK, qa-smoke regression unchanged. A second 200k A/B
  decides H2: submit growth that recovers = driver-path overhead; growth that
  persists = file-cache page-in (then only VirtualLock-class remedies apply,
  and the T-GPLR-08 falsification stands).
- **F2 — per-layer row caches (H1, corrected design).** See the H1(b)
  correction in 6.4: the sound variant is a rows-only cache, ≤ 8 rows ×
  3 members × ~589,824 B ≈ **14.2 MB per layer → ≤ ~570 MB total for 40
  layers** (not the 40 × 453 MB ≈ full-bank figure section 4 assumed).
  Execution-shape options: (i) device-side copy cache → slot bundle before
  part B (D2D ~14 MB ≈ 50-70 µs/layer — cheap, adds a copy class), or (ii)
  point each layer's FFN bundle tensors into its own cache buffer at build
  time (zero D2D; build-order change; raises a new residency class under the
  SDD — needs the atomic ready/retire + generation treatment before any
  contract). Payoff: reuse ≈ measured containment (0.358-0.389) becomes
  realizable instead of structurally impossible (staged 3.125% → ~2.0%), AND
  it turns T-GPLR-15's 0.6 hit-rate gate from a 2-step pilot into a
  measurable stream. Task-sized; multiplies with F1 (fewer bytes through the
  same faster pipe).
- **F3 — nonzero TG per-part timings (H8).** Cheap telemetry change; required
  to attribute F1's gain exactly (compute vs staging) and to bound the KV
  term. Do alongside F1, not after.
- **F4 — W>1 window (H4).** Unchanged as a probe, but its leverage shrinks
  once F1/F2 cut the per-fetch and per-byte costs it amortizes.

### 6.4 Correction to section 4 H1(b): "only the bookkeeping discards them" is UNSOUND with the current slot-buffer layout

Slot bundles alias one device buffer at offset 0: the init loop resets
`bundle_off = 0` for every layer assigned to a slot (`llama-layer-streaming.cpp`,
slot-buffer init: "they overlap by design (one resident layer per slot at a
time)"). Between two consecutive visits of layer il (9 layers later per
section 2), layers il+4..il+8n patch THEIR slabs into the SAME byte ranges,
so a row tracked resident for il can be overwritten by an intervening owner
before il returns. Per-layer coverage over the current buffers could then
claim a clobbered row and silently execute another layer's weights — the
exact SDD invariant violation row-set exists to prevent. The sound fix is F2
(per-layer row caches with real per-layer storage), not coverage bookkeeping
alone. H1's offline containment numbers (0.358-0.389) remain valid as the
upper bound of what F2 could realize.

### 6.5 Rejected alternatives (recorded so successors do not re-derive)

- `cudaHostRegister` over the 18.33 GB mmap region: locks ~18 GB of RAM
  permanently against the T-GPLR-09 admission arithmetic; registering
  file-backed mappings on Windows WDDM is unsupported/undocumented territory.
- Deeper staging queues (H3's standalone fix): the traces show no genuine
  serialization loss — submit growth is the same mechanism measured at a
  different point.
- `cudaMemcpyBatchAsync` alone without pinned staging: collapses 24 API
  submissions to 1, but the driver still stages pageable sources — the
  1.7-2.2 ms term barely moves.

### 6.6 Method and sources for this section

- Code read (no edits): `load_row_set` copy loop and `slot.buf_staging`
  usage, slot-buffer init bundle placement, `ggml_backend_cuda_set_tensor_async`
  (single `cudaMemcpyAsync` on the copy backend's stream).
- Online pass (2026-09-13, via context7): CUDA programming guide
  (Asynchronous Execution — pinned-memory requirement for async transfers)
  and CUDA driver API synchronization-behavior table (pageable async memcpy
  may stage synchronously through pinned memory). Claims verified against the
  local tree before use; the measured ~6.5-8.2 GB/s host rate matches the
  documented staging model.
- Arithmetic: 24 copies/fetch × ~589,824 B = 14.15 MB (matches section 1's
  batch size); submit sums = 40 × submit mean; residual = wall − submit −
  ready sums.

### 6.7 Correction to 6.3: pinned staging cannot beat the measured link ceiling, and it already failed the dense A/B (2026-09-13)

6.3's expected-effect arithmetic assumed pinned DMAs at "pinned PCIe rates
(~20+ GB/s)". The repository's own measurements contradict that on this
machine, and the F1 contract must be read against them:

- **The link is saturated at 13.7 GB/s.** T-Q38TG-00 measured steady-state
  H2D DMA at 13.7 GB/s on a negotiated Gen4 x8 (nvidia-smi, 2026-08-24) —
  87% of the 15.75 GB/s physical ceiling. The same measurement found that
  `cudaMemcpyAsync` from **pageable** memory "is internally staged at the
  same 13.7 GB/s", so for a source already in host RAM the explicit pinned
  path "only adds an ~11.7 ms staging memcpy and is slower end-to-end"
  (245.9 MB package: direct 17.94 ms vs pinned 18.0 ms DMA + 11.7 ms
  staging = 29.8 ms).
- **T-Q38PERF-03 already ran the A/B on the dense path and pinned lost:**
  paired 1,024-token layer-major runs gave pinned PP 8,366.4 ms vs direct
  8,195.3 ms and pinned TG 2,757.5 ms vs direct 2,515.0 ms. Pinned did cut
  host submission (PP 7,505.1 → 6,062.4 ms) but raised availability wait
  (TG 5.3 → 1,510.4 ms). Pinned staging remains opt-in as an instrumented
  path, not a performance default.

**What survives of F1.** Only the narrow per-call-overhead hypothesis: the
row-set regime issues 24 small copies (~0.59 MB) per 14.16 MB, where driver
per-call cost could dominate. The measured submit (1,826 µs at 512) exceeds
the ~1,033 µs a 14.16 MB DMA at the 13.7 GB/s ceiling would take, leaving
~0.8 ms/fetch not explained by raw DMA. The pre-registered ceiling is
therefore per-fetch staging ≈ 1.03 ms (from 1.93 ms); a larger claim is
arithmetic error. F1 is now plan task **T-GPLR-18** (READY 2026-09-13) with a
paired 512 A/B gate, and a negative result is a legitimate close.

**Consequence for the ranking.** F1's upside is smaller than 6.3 implied and
it carries a measured precedent against it on this hardware, while F2
(per-layer row caches, fewer bytes across the link at all) is the more robust
lever; the two are complements, not substitutes. T-GPLR-15's gate work is
unchanged. F2 does not yet have a task ID.

No GPU runs were performed for this correction; it transcribes recorded
measurements (T-Q38TG-00, T-Q38PERF-03) that 6.3 did not account for.
