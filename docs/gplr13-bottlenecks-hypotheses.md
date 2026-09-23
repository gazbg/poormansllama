# T-GPLR-13 Ladder Bottlenecks and Solution Hypotheses

Status: active findings file (created 2026-09-13). A documented online research
pass on H-C/H-D landed 2026-09-15 (docs/gplr13-research-hcd-online.md section
8); it changed no measurement and no verdict in sections 1-6. Future runs of the
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

Provenance caveat (2026-09-13; corrected the same day after verification, then
superseded the same day by the T-GPLR-15 Phase A re-derivation): the two
statistics above were MACHINE-computed by `build/analyze_gplr13_ladder.py`,
the untracked one-off already named in section 1. That script lives INSIDE the
gitignored `/build/` tree (`.gitignore:/build/`), so it does not survive a clean
checkout, and at the time of writing it was the only source. It no longer is:
the T-GPLR-15 Phase A analyzer `tools/route_overlap/analyze.py` REPRODUCES the
table above exactly — 0/80 hash-equal at every size and containment 0.377 /
0.370 / 0.389 / 0.358 at 50/512/4k/200k — and adds the gate-9 statistic (b) at
K=2 (0.4031 / 0.4031 / 0.4266 / 0.4000, first measured values; capacity 16
rows/layer = 1,132.5 MB over 40 layers vs 566.2 MB at K=1). That analyzer is
now TRACKED (2026-09-13), so the producer of these numbers is in a clean
checkout; the untracked `build/` traces it reads are not, so re-deriving them
needs those traces or a fresh ladder run. Reproduce with:
`python tools/route_overlap/analyze.py "build/llama-cuda/gplr13-ladder/*/rowset-*.jsonl"`
(artifact `build/gplr15-phaseA.{txt,json}`). The containment direction is owned
by the SDD section 10 gate 9 decision rule (`|A n B| / |B|`, next window in the
denominator).

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

**A SECOND, different gap in the same statistic - MEASURED 2026-09-14
(lifetime, not counter).** `peak_vram_bytes` is sampled exactly once, at the
end of the run (main.cpp:4662-4663, `total - min(free_after, free_start)`), so
any allocation released before that reading is invisible to it regardless of
how much device memory it held. The T-GPLR-19 per-layer row cache is precisely
that case: `ls.end_row_set()` (main.cpp:4557 -> `end_row_cache()`,
llama-layer-streaming.cpp:3653) frees it before the sample. Two 4k qa-smoke
arms that differ ONLY by the retention flag report the SAME peak
(5,311,627,264 B; 5,204,672,512 B at 50k) while `byte_account` records
`row_cache` 2,291,531,776 B in one and 0 in the other - and with
`tools/gpu-mem-poll.ps1` attached, per-process dedicated peaks are 3,978.8 MiB
(reactive) vs 6,242.8 MiB (retention), +2,264.0 MiB (one run per arm, 500 ms
sampling; the two poll files overlap, so each run's process is separated by its
absolute window, not by file name). Cross-checked against 3.1: for the reactive
arm the JSONL figure exceeds the polled one by 1,086.8 MiB (the 1.09 GB class
above); for the retention arm it falls 1,177.3 MiB *below* (5,065.6 vs 6,242.8),
i.e. the cache is missing. So the
statistic has two independent blind spots of different kinds - a counter blind
spot (3.1) and this lifetime blind spot - and they must not be conflated: a
phase-boundary sample fixes the first and still misses the second unless the
sample lands while the cache is live.

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

### 3.6 New: TG compute timings are all zero - CLOSED 2026-09-14 (F3/H8 fixed)

[CLOSED 2026-09-14 - original text retained as written; amendment below]

All 120 TG `layer_compute` events per run carry `compute_us=0, setup_us=0`
(PP events are properly timed). The schema draft's open decision #3
anticipated the split-graph timing question. Consequence: the TG
decomposition in 3.2 cannot separate FFN/attention compute from scheduler
overhead; only staging-vs-residual attribution is possible. Fix belongs to
telemetry (time the two graph parts around `load_row_set`), not to this
analysis.

**Amendment 2026-09-14 (F3/H8 implemented; measurement, not a claim).** The
fix named above is now in the code: the row-set TG executor times each of the
two graph parts around the staging (`run_graph_plain` gained the same
setup/reset-alloc vs compute/launch-execute-sync split the other executors
record) and `tel_layer_compute_parts` emits ONE `layer_compute` per
layer/window whose `compute_us`/`setup_us` keep their exact prior meaning (the
A+B sums) plus four new fields `attn_compute_us`/`attn_setup_us` (part A =
attention+residual+post-norm+router) and `ffn_compute_us`/`ffn_setup_us`
(part B = FFN + shared experts on the staged rows). Event cardinality is
unchanged, so no consumer of the old fields moves.

Re-run of the recorded rowset probe shape (identical flags,
`--residency moe-expert-bank --residency-tg --pp-tokens 64 --ubatch 64
--tg-tokens 3 --layer-slots 4 --allow-full-residency`; fresh output path, the
2026-09-13 trace is preserved): 120 TG `layer_compute` events, and the totals
move from `compute_us=0, setup_us=0` to `compute_us=66,822,
setup_us=4,788` - attn 43,000 (setup 2,737) / ffn 23,822 (setup 2,051) us.
On the fuller 512/256-token batch-major stream with retention at the default
K=4 (`--residency-tg-retain 4`): 10,240 TG `layer_compute` events, **0/10,240
zero-valued in either part**, and the row split is

| term | per TG token | share of TG wall |
| --- | --- | --- |
| submit (pageable host loop) | 23,574.9 us | 48.9% |
| copy completion (ready latency) | 7,959.9 us | 16.5% |
| compute, part A (attn+router) | 8,019.6 us | 16.6% |
| compute, part B (FFN) | 4,224.2 us | 8.8% |
| graph setup (both parts) | 1,297.6 us | 2.7% |
| accounted above | 45,076.1 us | **93.5%** |
| not yet instrumented | 3,148.9 us | 6.5% |

TG wall for that trace is 48,225 us/token (256 tokens at 20.736 tok/s). So
`transfer` = submit + copy completion = 65.4% of the token and compute+setup =
28.1%, which is the split 3.2 could not make before. The residual 6.5% is
NOT attributed by this fix and is not claimed to be: the router's blocking
device-to-host probs read, sampling/argmax and the round boundaries all sit
outside the four timed windows. Its shares are this shape's own; do not read
them back onto the ladder rungs, which run a different schedule.

No behavior change accompanies the timing: the 10,240-event `rowset_fetch`
stream (layer, chunk, union hash, fetched/retained counts, staged bytes), the
256 `rowset_round` byte accounts, `byte_account` on every original field and
the greedy text sha256 are all identical to the recorded K=4 arm.

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

> **SUPERSEDED READING (2026-09-22, T-GPLR-55 decision).** The 4.4-5.8 GB/s above is a **WALL rate**
> (bytes / wall), **not a transport speed**, and it must not be read as one. The copy itself, measured on the
> only serialized row-set arms on disk (`ready_latency/transfer_us` = 0.98), moves **13.0 GB/s** - identical
> at 512 and 200k - and a serialized 347.2 MiB full-bank copy measures 14.34 GB/s. The 7.3-7.4 GB/s family is
> a queue-inclusive latency, not a rate. See the T-GPLR-55 block in
> `docs/plans/generic-prefill-layer-residency-plan.md`.

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

> **Both figures are WALL rates**, not transport speeds, and neither is a copy rate - superseded reading
> recorded in section 3.7 above (T-GPLR-55 decision, 2026-09-22). The candidate cause below ("latency-bound",
> 74.6-78.1% ready latency) is still a real observation: the same trace field
> (`ready_latency_us / transfer_us`) is what shows a copy set is QUEUED rather than fast.
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
Second, now-MEASURED defect in the same statistic (3.1, 2026-09-14): it is a
single post-run sample, so an allocation freed before it - the T-GPLR-19 row
cache, 2,291,531,776 B - is invisible (two 4k arms with and without it report
identical peaks). Relabelling is now the MINIMUM repair, not an option, and
option (a) must sample at the run's true peak (while the cache is live) or it
inherits the same blind spot; a cheap third option is to have the runner add
the high-water mark of its own allocations instead of inferring from free
VRAM.

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

### H8 - Fix zero TG compute timings (telemetry debt; unblocks H2/H3 attribution) - DONE 2026-09-14

[STATUS 2026-09-14: implemented and measured; see the 3.6 amendment above for
the numbers. The two graph parts are now timed around `load_row_set` and
`layer_compute` carries `attn_*`/`ffn_*` alongside the unchanged `compute_us`.
H2's KV-vs-staging decomposition can now be made exact; the router-read
residual noted in 3.6 is a separate, still-open gap.]

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
- **F3 — nonzero TG per-part timings (H8).** [DONE 2026-09-14: implemented and
  measured, see the 3.6 amendment. One `layer_compute` per layer/window now
  carries `attn_compute_us`/`attn_setup_us`/`ffn_compute_us`/`ffn_setup_us`
  beside the unchanged `compute_us`; 0/10,240 events zero-valued on the
  512/256-token K=4 stream and 93.5% of TG wall accounted vs 0% attribution
  before.] It was required to attribute F1's gain exactly (compute vs staging)
  and to bound the KV term; F2's staged-byte reduction can now be read against
  a real compute term rather than a zero.
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

## 7. T-GPLR-15 gate measured on a real 256-token generation stream (2026-09-13): fails the reference band, ceiling 0.94, crossing at 2.26 GB

The section 2 pilot was 80 pairs over 3 TG windows and was explicitly "never a
gate verdict". The gate has now been measured on the stream it is defined over:
one batch-major run, 256 greedy tokens, 10,200 pairs (255 per layer x 40
layers), runner exit 0, `--check-telemetry` OK (21,020 events, 4 slots).

| statistic (SDD section 10) | value | retention cost |
| --- | --- | --- |
| (b) K=1 (W=1) | 0.3088 | 8 rows/layer = 566 MB |
| (b) K=2 | 0.4343 | 16 rows/layer = 1.13 GB |
| (b) K=3 | 0.5330 | 1.70 GB |
| (b) K=4 | 0.6284 | 2.26 GB <- first reading at or above the 0.6 threshold |
| (b) K=8 | 0.7775 | 4.53 GB |
| (b) K=16 | 0.9376 | 9.06 GB |
| T8 oracle (unbounded) | 0.9385 | 18.12 GB = the full 256-row bank resident |

So the SDD's reference retention band (K=1-2) FAILS the 0.6 gate while the
honest ceiling is 0.94: the predictive hypothesis is not falsified, the
budgeted variant is. Per-layer (b) at K=2 spans 0.1946-0.5353, so no small
subset of layers is carrying the aggregate, and no layer reaches 0.6 there.

Two caveats that must travel with the table:

- The generated text is an echo loop of the anchor document (463 characters,
  its code phrase repeated 6x, `<think>` 7x; token periodicity 0.479 at lag 39,
  immediate repeats only 8.6%). Repetition can only INCREASE temporal reuse, so
  the in-band FAIL is CONSERVATIVE (a cleaner stream measures the same or
  lower) while the K=4 crossing and the 0.9385 ceiling are likely OPTIMISTIC
  and should not justify a VRAM commitment until re-measured on a stream that
  does not echo its own prompt.
- The oracle is a generous bound by construction: reaching it means the whole
  expert bank is resident, which is the outcome the row-set policy exists to
  avoid. The informative number is the retention cost of crossing 0.6.

Related instrument facts from the same run: T5 consecutive-window hash equality
is 0.0019 (19 of 10,200) — near zero, as the top_k=8 over n_expert=256 sparsity
predicts — and T7 reappearance coverage at retention 1/2/4/8 windows is
0.329 / 0.428 / 0.554 / 0.703. No throughput claim is made here: this section
measures routing predictability only. Whether 2.26 GB of retention is
acceptable is a product decision, escalated to the owner (see
`docs/currentstate.md`, 2026-09-13).

**DECIDED 2026-09-13 (owner-directed close).** The verdict is read at the SDD's
own reference sizing — one-to-two window retention, K=1-2, 566 MB-1.13 GB — where
(b) is 0.3088 / 0.4343, so the gate FAILS and T-GPLR-15 closes
falsified-by-evidence per gate 9. The 2.26 GB K=4 crossing is REJECTED as the
optimistically biased half of the measurement: it sits on a stream that echoes
its own anchor, and repetition can only inflate reuse, so it is not a sound
basis for committing VRAM. A non-echoing re-probe is recorded as a precondition
for any future retention above the band — by the same monotonicity it can only
LOWER (b), so it can confirm this FAIL but can never rescue the crossing. What
is falsified is the budgeted variant, not the predictor's existence (the oracle
ceiling is 0.9385), so a later task may declare a retention and re-read the same
statistic under gate 9.

## 8. T-GPLR-18 closed falsified-by-evidence (2026-09-13): pinned row-set staging relocates the copy, it does not speed it up

Section 6.7 predicted this result from T-Q38PERF-03 and T-Q38TG-00 and the
narrow surviving hypothesis was the only reason to test it: the row-set regime
issues 24 slabs of ~0.59 MB per 14.16 MB fetch, where per-call driver cost
could dominate. The paired 512 A/B says it does not. Arms differ only by
`--copy-source`; four arms (direct, pinned, direct, pinned) ran inside one
GPU-mutex session with the memory poller attached, 120 fetches each.

| arm | source | submit_us | ready_us | transfer_us | staged bytes | commit peak |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | direct | 1,937 | 110 | 2,046.6 | 1,718,157,312 | 19.95 GiB |
| B1 | pinned | 898 | 1,057 | 1,955.2 | 1,718,157,312 | 21.25 GiB |
| A2 | direct | 1,899 | 110 | 2,009.2 | 1,718,157,312 | 20.07 GiB |
| B2 | pinned | 955 | 1,058 | 2,013.9 | 1,718,157,312 | 22.09 GiB |

- Per-fetch means; direct mean 2,027.9 us vs pinned 1,984.6 us, a 43.3 us gap
  BELOW the largest within-arm repeat spread (58.7 us). Criterion 2 FAILS.
- The signature is T-Q38PERF-03's, reproduced on the row-set regime: submission
  falls (~1,050 us) and the availability wait rises by the same amount, so the
  pageable source is not being staged slower than pinned - pinned merely makes
  the same staging explicit as a host memcpy.
- The 6.7 pre-registered ~1.03 ms per-fetch floor (14.16 MB at the measured
  13.7 GB/s link ceiling) is not approached by either arm; both sit near 2.0 ms,
  so there was no per-call overhead to recover at 24 slabs per fetch.
- Pinned is also the more expensive arm: one staging buffer per slot
  (~458 MB x 4 slots = ~1.83 GB of non-pageable host memory), visible as the
  1.3-2.0 GiB higher host commit peak.
- Feasibility and immunity held: staged bytes, route streams (SHA-256 over the
  ordered `layer|union_ids|hash` stream) and greedy token IDs identical across
  arms; host suite 23/23, the dense pinned CTest fixture 4/4, and the qa-smoke
  512 default-off rung PASS with the negative control failing as designed.

**Ranking consequence.** With F1 closed, the TG lever for the not-fitting
regime is F2 (bytes across the link) or T-GPLR-15 hiding, which is what 6.7's
ranking already said once F1's upside collapsed - the difference is that the
collapse is now measured rather than predicted. No throughput claim is made:
wall time across the four arms (10.4-16.5 s) is dominated by shared-GPU noise
and the two `direct` arms are both the fastest and the slowest.

**F2-M1 ranking outcome (2026-09-13, T-GPLR-19).** F2 was ranked as the
remaining lever here, and it has now been RUN as a mechanism rather than
predicted. The four-arm A/B (A,B,A,B in one GPU session; reactive vs K=1
per-layer row retention) measured TG **11.953 -> 14.021 tok/s**, gain **+2.068**
against a **0.011** within-pair spread, with staged bytes falling to
**396,398,208 B/round** against the predicted 395,835,932 (**+0.142%**, i.e. the
byte prediction was met, not merely bounded). So this section's ranking is
CONFIRMED: with F1 dead, F2 is the lever that moves TG, and it moves it by
~17% at K=1. Two things this does NOT upgrade: the same-run projection envelope
(13.61-15.30 tok/s) still rests on a self-echoing anchor at 3x extrapolation,
so scaling K is not licensed by this result; and the 0.6 gate is read against
statistic (b), which at K=1 is 0.3088 - well below it - so F2-M1 is a real but
partial answer, not a pass on the gate the earlier Phase A reading failed.

[UPDATED 2026-09-13, later the same day: the first clause is superseded -- the
`K > 1` rungs WERE run (K=2/4/6/8, all PASS) and the scaling is now measured,
not extrapolated. Measured TG: 14.021 / 15.820 / 19.727 / 22.074 / 24.481 tok/s
at K=1/2/4/6/8 against reactive arms at 11.863-11.967, and the runtime's
retained-row count equals the offline statistic exactly at every depth. Two
qualifications survive and one is added. (a) The self-echoing anchor caveat
STANDS: these rungs measure internal consistency, and the non-echoing re-probe
is still the gate-bearing run, so the gate standing at K>=4 (measured fraction
0.6263 against 0.6284 projected) is NOT yet a non-echoing verdict. (b) The
byte-to-time conversion DEGRADES with depth -- 14.29, 12.18, 10.67, 10.52,
10.23 GB/s -- so scaling K buys steadily less time per byte, and the marginal
rate past the first window is ~8-9.5 GB/s, near this path's effective per-fetch
rate rather than the link ceiling. (c) New: the second clause above is now a
measured statement at K=4 rather than an inference, but only on the echoing
stream.]

### 6.8 New: the non-echoing re-probe has been RUN — K=4 does NOT clear the gate on it (2026-09-14)

One measurement, one GPU session, no code changed by this section. This closes
the "(a) caveat STANDS" clause immediately above: the gate-bearing run is no
longer outstanding.

**What was run, and the honest provenance of the definition.** The recorded
T-GPLR-15 Phase A probe invocation with exactly ONE change — the prompt BODY:
`build/calc-prompt.txt` (the natural-language code prompt already recorded for
the gplr06 code-prompt runs, 885 tokens) in place of the echoing needle
document `build/llama-cuda/gplr13-ladder/512/anchor-512.txt`. Everything else
held: `--residency moe-expert-bank --residency-tg --pp-schedule batch-major
--ubatch 512 --tg-tokens 256 --layer-slots 4 --kv-cache q8 --greedy --no-mtp
--pp-no-logits --allow-full-residency`. The reading definition and the decision
rule were written down BEFORE this run but do NOT predate the original
measurement — they are authored 2026-09-14, and that is stated rather than
implied.

*Reading definition.* Statistic (b) exactly as SDD section 10 / gate 9 defines
it (the tracked analyzer, row-capacity-lru policy, W=1, next window in the
denominator), read at the depth the SDD declares, plus the T8 oracle, T5 and T7
for context. *Decision rule.* K=4 SURVIVES only if (b) at K=4 >= 0.6 on the
non-echoing stream; otherwise the recorded 0.6284 is echo-inflated and no K=4
gate PASS is available. Predicted direction, fixed in advance: repetition can
only RAISE (b), so a non-echoing (b) <= the echoed (b) at every K.

**Non-echoing check — PARTIAL, and its defect is mine.** The declared
conjunction was `distinct 16-gram ratio >= 0.90` AND `zlib(id sequence)
compression ratio >= 0.50`, calibrated beforehand on the known-echoing stream.
Result: 16-gram ratio **1.0000** (echoed: 0.3734 — all 241 spans distinct) and
zlib ratio **0.4238** (echoed: 0.1152). So one criterion is at its maximum and
the other fails: zlib compresses structured HTML/CSS tokens well without any
repetition, so that criterion measured the entropy of a code stream rather than
repetition and was mis-chosen when declared. The stream is non-echoing on the
criterion that measures repetition and on the generated text (a single-pass
calculator document, no loop); the certification is recorded as partial rather
than repaired after the fact. Distinct-token ratio 0.4102 vs 0.0977 echoed.

**Result.**

| K | echoed (recorded probe) | non-echoing (re-probe) | delta | VRAM over 40 layers |
| --- | --- | --- | --- | --- |
| 1 | 0.3088 | 0.2468 | -0.0620 | 566.2 MB |
| 2 | 0.4343 | 0.3778 | -0.0565 | 1132.5 MB |
| 3 | 0.5330 | 0.4704 | -0.0626 | 1698.7 MB |
| 4 | 0.6284 | **0.5355** | **-0.0929** | 2264.9 MB |
| 5 | 0.6803 | 0.5982 | -0.0821 | 2831.2 MB |
| 6 | 0.7169 | 0.6487 | -0.0682 | 3397.4 MB |
| 7 | 0.7483 | 0.6926 | -0.0557 | 3963.6 MB |
| 8 | 0.7775 | 0.7283 | -0.0492 | 4529.8 MB |
| T8 oracle | 0.9385 | 0.9143 | -0.0242 | unbounded |

Every K falls, so the pre-registered direction holds and the bias is confirmed
rather than merely asserted. **The 0.6 crossing moves from ~K=3.6 to ~K=5.1:
the first depth clearing the threshold on the non-echoing stream is K=6
(0.6487, ~3.40 GB), not K=4 (2.26 GB).** T4 consecutive-window containment
0.3088 -> 0.2468 and T5 window-equality 0.0019 -> 0.0004 (exact window identity
was never the discriminator, as the schema draft says); rows seen once 739/5342
-> 1039/7311; cumulative coverage at retention 1/2/4/8 = 0.3291/0.4279/0.5535/
0.7025 -> 0.2699/0.3832/0.5160/0.6498.

**One correction to how the echo caveat has been quoted.** The re-probe's
temporal-reuse ceiling is 0.9143, nowhere near this file's 0.4000-0.4266 range
for the four preserved ladder rungs. Those rungs are 3-window streams, so their
ceiling is structurally capped at two prior windows and is monotone in stream
length — the 0.40-0.43 range is a STREAM-LENGTH artifact, not the non-echoing
expectation. The only like-for-like comparison is 256 windows against 256
windows: 0.9385 -> 0.9143.

**Consequence.** No default, no gate verdict and no implementation changed:
K=4 stays the mechanism's provisional operating default, and a K=4 gate PASS
remains UNAVAILABLE — now on measured evidence rather than on a caveat. The
first depth that would carry a gate reading on a non-echoing stream is K=6.
Artifacts: `build/llama-cuda/gplr15-nononchoing/probe-tg256-nononchoing.jsonl`
(10,240 `rowset_fetch`, 256 rounds, `--check-telemetry` OK, run exit 0) and the
tracked analyzer's report beside it; the launching one-off is untracked under
the gitignored `build/` root, so a clean checkout needs a fresh run.

## 2026-09-14 late: full-length sweep (4k/50k; 200k attempt), bottleneck structure, and the 200k abort fix

**Scope.** Owner-directed validation sweep of the pruned tree at 4k, 50k and
200k on the recorded streaming shape (batch-major PP, `--no-handoff`, paged
full-bank TG; `--ubatch 512` at 50k/200k). The 4k rung went through the
suite's qa-smoke gate; the rest were direct runs under the GPU mutex with
fresh artifact paths. Measurement pass; no gate verdict and no default
changed.

**Gate results.** 4k PASS (check-telemetry OK, 23,326 events; answer 5/5
PASS; negative control FAILs as designed). 50k PASS (46,606 events; answer
PASS; control FAILs). 200k crashed at graph build on the first attempt (root
cause and fix below); the retry is recorded in currentstate when it lands.

**Measured bottleneck structure (streaming TG = paged full-bank).**
- Warm-cache recorded shape: TG 8.12 s / 96 tok (11.8 tok/s) vs PP 6.26 s /
  4,096 tok (654 tok/s). Reactive TG stages 54.98 GB across 3,840
  `rowset_fetch` in 6.00 s (9.16 GB/s effective on the link).
- Retention K=4 arms: TG 5.11 s (18.8 tok/s, +59%); staged 21.05 GB (4k) /
  17.93 GB (50k); hits 3,741/3,840 and 3,743/3,840 (~97.4%).
- F3's part split exists in the writer but NO trace on disk carries the
  attn/ffn part fields yet: the split is requested only by the row-set TG
  executor, and every recorded streaming-mode run bypasses row-set. The
  first producer will be a `--residency-tg` run on the new binary.
  [2026-09-15: that prediction is now false in the good direction — the
  4k retention proof run IS the first producer: 3,840 of 3,880
  `layer_compute` events carry `attn_compute_us`/`ffn_compute_us` and the
  setup parts. First real reading at 4k retention: attention 30.3 ms vs
  staged FFN 1.1 ms per layer-window — TG compute is ~27x attention-dominated
  at 4k context, so expert staging is the transfer story but NOT the
  compute story.]
  [CORRECTED 2026-09-15 by the 50k width test: the "~27x" was the FIRST
  event (layer 0, window 0), which is the distribution MAX -- a first-window
  warmup outlier -- not a representative reading. Full distributions:
  4k attention mean 0.30 ms / median 0.19 / max 30.3; FFN mean 0.14 ms;
  ratio of means 2.2x. 50k (`gplr07-qa-50000-retention.jsonl`, warm cache,
  qa-smoke gate PASS): attention mean 0.48 ms / median 0.27 / max 26.8;
  FFN mean 0.20 ms; ratio of means 2.4x; per-layer attention means are
  < 2 ms on ALL 40 layers in both runs. So attention leads staged-FFN
  compute by ~2-3x, not ~27x -- and at 50k all three TG terms are
  comparable: transfer 3.63 s, wait 1.04 s, compute 2.63 s of the 7.21 s
  TG wall. Expert staging remains the single largest TG term at width;
  the earlier "compute story" framing was an artifact of reading one
  event. Per-window attention does NOT grow across the TG (last-10-window
  mean 0.77x of first-10) -- recorded as an unexplained flat reading,
  not fitted to a story.]

**Discrepancies found (three real, one environmental).**
1. COLD CACHE (environmental; reported per AGENTS.md, not a defect): the
   22:36 re-run of the identical 4k rung took TG 185.2 s (0.52 tok/s) vs
   8.1 s warm and PP 8.28 s vs 6.26 s. The 30x TG hit vs 1.3x PP hit is the
   model page-in dominating TG; the trace is structurally identical (checker
   OK, answer PASS) and the pruning touched no fetch path. Every cross-run
   timing comparison must hold file-cache state fixed.
2. TG phase_summary reports zeros: `total_transfer_us`, `total_wait_us`,
   `total_compute_us` and `bytes_transferred` are all 0 in the TG phase
   while 3,840 `rowset_fetch` events carry real values; the PP summary is
   populated. The TG emitter never sums the row-set terms, so a consumer
   reading phase_summary alone sees a free TG phase.
   [2026-09-15: FIXED and proven on the real arm. The row-set TG executor
   now feeds the same `add_layer_stats` pipeline as the streaming
   executors — transfer/bytes accumulate in the two `rowset_fetch` emit
   paths, the timing triple is stashed per layer (both hit and miss paths
   update it, so no stale-carry), and `reset_stats()` zeroes it per phase.
   The 4k retention proof run (`gplr07-qa-4096-retention.jsonl`) reports
   TG `total_transfer_us` 20.47 s, `total_wait_us` 1.71 s,
   `total_compute_us` 1.67 s, `bytes_transferred` 21.05 GB — non-zero,
   and consistent with the 3,840 `rowset_fetch` events. Cold-start
   caveat: that run is the cold branch of the file cache (machine
   restart), so the absolute 20.47 s is not comparable to the warm 3.04
   s; only the nonzero-ness and internal consistency are the claim.]
3. `run_end.peak_vram_bytes` is blind to the row cache: the reactive and
   retention 4k arms report the IDENTICAL 5,311,627,264 B peak although
   retention holds a 2,291,531,776 B row cache -- the cache is allocated
   outside the ggml pool the peak samples. byte_account carries `row_cache`,
   but the headline peak understates true device usage by exactly the cache
   size.
   [2026-09-15: FIXED and proven on the real arm. Root cause was not the
   ggml pool: main's peak was a TWO-SAMPLE max (free_start vs a free_after
   taken after end_row_cache released the cache), so any allocation freed
   before run_end vanished from it. Fix: the budget struct (the one owner
   of VRAM state) now tracks free_min across every refresh point --
   post-init, post-cache-bring-up, post-slot-window -- and tel_run_end
   reports total - free_min. Non-retention runs are unaffected: free_min
   equals the old two-sample max when no cache/slot allocation intervenes.
   4k retention re-run now reports peak 7,675,117,568 B = old 5.31 GB +
   row cache 2.29 GB + slot window, and check-telemetry/answer gate PASS
   on the same trace. The reactive contrast arm was blocked the whole
   window by another project's GPU-mutex hold (1,800 s lock timeout, no
   contamination -- its recorded trace is untouched); the retention arm
   alone proves the fix because the identical-peak artifact had reactive
   == retention by construction.]
4. Ready-latency inversion under retention: median `ready_latency_us` 198
   (hit path) vs 101 (reactive miss path) at 4k, 185 vs ~101 at 50k -- the
   cache DOUBLES the wait term even while halving the transfer term (766 vs
   1,563 us/fetch). Net wall still favors retention (transfer dominates),
   but the wait grows with retained-window bookkeeping.

**Hypotheses (each tied to a measured discrepancy).**
- H-A (TG zeros): the TG phase_summary emitter sums only ggml-scheduler
  stats and row-set fetches bypass the scheduler's accounting. Cheap fix:
  accumulate `transfer_us`/`ready_latency_us`/`bytes_staged` per phase in
  the rowset_fetch emit path and add them at phase_summary. Makes TG wall
  decomposable from the summary alone.
- H-B (peak VRAM): the cache is a separate cudaMalloc outside the sampled
  pool. Cheap fix: add `row_cache` (already in byte_account) into the
  reported peak, or sample free VRAM across the TG loop.
- H-C (ready-latency): the ~198 us hit-path wait is per-fetch slot event
  synchronization on the K>0 cache (bookkeeping + retained-set verification),
  not transfer. Predicts the wait grows sub-linearly with K and that the F3
  submit/copy/ready split on a `--residency-tg` run will show it as neither
  submit nor copy. Testable with the existing F3 writer, no new plumbing.
- H-D (PP bandwidth): PP full-bank staging (39 evictions, 18.33 GB in
  5.49 s = 3.34 GB/s) runs at barely a third of the TG reactive rate
  (9.16 GB/s) -- PP copies whole banks on eviction with effectively no
  overlap window, TG overlaps fetch with decode. Raising PP prefetch overlap
  is the largest single PP lever left (~2.7 s at 4k, ~90 s at 50k of pure
  transfer exposure).

**200k abort: root cause and fix.** First attempt died at graph build:
`attn_inp_kq_mask exceeds backing buffer (offset=421603680 size=128122880
buffer=477102080)` -> `finish_build` failed -> `GGML_ABORT` at
`qwen35moe.cpp:478`. byte_account PROVES this is not VRAM: bounded window
5.32 GB vs 22.23 GB free, `bounded_within_free=true`. The budgeted host
input buffer (n_masks * mask_bytes + embd + 64 MiB = 477,102,080 B exactly)
undercounts when more persistent sets bind full-context masks than n_masks
counts -- the same sizing-formula class T-GPLR-05 patched once before with
a bigger constant. Fix (T-GPLR-24 candidate): a host-side SPILL allocator in
`finish_build` -- tensors that no longer fit `buf_inputs` allocate into an
extra host buffer (`buf_inputs_spill`) instead of failing the build. No
VRAM, no placement change, and a shape that fits never creates a spill, so
the allocation path of every existing recorded shape is unchanged.
**Landed and proven:** the re-run COMPLETED at 200k (116,926 events,
check-telemetry OK, answer 5/5 PASS, control FAILs; PP 1,022.1 s, TG
185.1 s, peak 7.18 GB) and the 512 gate plus the host suite still pass.
The first spill build had a real bug, caught by the re-run: the failed
`alloc_tensors` pass had already allocated a prefix of the list into
`buf_inputs`, and spilling the full list re-allocated bound tensors
(`GGML_ASSERT(tensor->buffer == NULL)`); the spill now takes only the
unallocated remainder. Evidence: `gplr24-sweep/rung-200000-fix.*`.
