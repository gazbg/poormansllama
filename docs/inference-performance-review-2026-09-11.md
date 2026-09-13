# Inference Performance Review

Date: 2026-09-11.

Status: TRACKED HISTORICAL REVIEW (tracked 2026-09-13; owner decision — track,
do NOT inline). A dated snapshot of 2026-09-11, not open work: per its own
scope note below it proposes work and amends no task contract. It is uncited
elsewhere in the repo, but it is the origin analysis for the MoE row-set
program — its P1 transfer-granularity finding is the row-set hypothesis, and
its focused experiment 7 anticipated the T-GPLR-15 reuse gate by warning that
the 60% TG cache hit rate was unevidenced and that per-layer TG IDs had to be
collected first. Phase A has since measured that gate on the ladder stream:
0.37-0.43 against the 0.6 threshold, i.e. below it, matching this review's
skepticism. Its per-task status observations (which task is READY/DRAFT/
rejected) are as of 2026-09-11 and are superseded by `docs/currentstate.md` and
the active plan.

Scope: all nine project plans, the complete [currentstate.md](currentstate.md),
[AGENT_ENVIRONMENT.md](AGENT_ENVIRONMENT.md), the active
[layer-residency-sdd.md](layer-residency-sdd.md), selected controlling code,
and primary-source online research. No new GPU benchmark was run.

**Target clarification: run the same 35B-A3B MoE on an 8 GB-class GPU,
minimizing peak VRAM while improving useful-answer latency.** The 24 GiB-class
development GPU is a measurement platform, not the deployment requirement.
Calculations below use an 8 GiB (8192 MiB) capacity envelope; actual available
memory must leave room for the driver, display, and other consumers. Neither
an 8 GiB allocation on this GPU nor its throughput proves performance on
another 8 GB GPU. A smaller model is not a substitute for this goal.

This report proposes work; it does not amend task contracts or approve
implementation. Existing worktree changes were preserved. Source findings
describe the in-progress snapshot, not a released or validated implementation.
PP means prompt processing; TG means token generation. GB is decimal;
GiB and MiB are binary.

## Priority Findings

### P1: Whole-Bank TG Has The Wrong Transfer Granularity

The recorded MoE streaming path transfers all 18,327,011,328 bytes of routed
expert banks per model pass, although each token selects only 8 of 256 experts
per layer. The 50K trace reports 0.509 tokens/s over three generated tokens.
This is a useful bottleneck sample, not a stable long-generation benchmark.

The owning [load_layer implementation](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L1067)
copies every package member. Exact selected-row payload across all layers is
572,719,104 bytes/token before padding, caching, or partial residency: **32x
less weight traffic**. This is not a prediction of a 32x application speedup.

Keep whole-bank, layer-major PP: observed 512-token route unions commonly
cover 245-256 experts, and whole-prompt unions approach saturation. For TG,
prioritize selected-expert execution within a bounded memory budget, retaining
only the weights that fit after session state and workspace are accounted for.
Full residency of the current checkpoint is not the 8 GB solution. Faster
full-bank copies cannot remove the fundamental byte cost.
Evidence: [current TG analysis](currentstate.md#L111),
[route-overlap task](plans/generic-prefill-layer-residency-plan.md#L1848).

### P1: Partial Handoff Is Not Yet A Validated Optimization

T-GPLR-16 already exists. Do not start a second implementation of it.
Three source-level issues must be resolved before using its results:

- **Boundary mismatch:** the runner passes `--handoff-partial N` unchanged
  to `begin_handoff_partial`. Ordinary cases retain layers below N on the
  host and migrate layers at or above N. But N equal to the layer count
  invokes full handoff. On this 40-layer MoE, N=40 therefore subtracts all
  expert banks in preflight and then requests all-bank migration. N=0 is
  refused despite corresponding to all banks migrating under that rule.
  Check asymmetric N=1/39 and both endpoints; the common N=20 case hides
  inverted-count mistakes. This is a source finding, not a reproduced OOM.
- **Dispatch differs from the plan:** before backend initialization, the
  runner defaults `GGML_OP_OFFLOAD_MIN_BATCH` to 1 for partial handoff when
  the variable is unset. Host-resident experts then use CUDA selected-row
  execution, not necessarily CPU expert arithmetic. The default backend
  threshold is 32. Source comments describe a runner-specific CPU numerical
  divergence investigation; those comments are not fresh verification and
  do not establish that CPU execution is fundamentally unsuitable.
- **Unaccounted traffic:** partial handoff emits zero-byte streaming-module
  copy events, while the backend scheduler can still copy selected experts
  over PCIe. Zero module `bytes_transferred` is not zero physical H2D traffic.

Sources: [runner call](../third_party/llama.cpp/tools/layer-streaming/main.cpp#L2968),
[boundary branches](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L2673),
[partial preflight](../third_party/llama.cpp/tools/layer-streaming/main.cpp#L2517),
[dispatch override](../third_party/llama.cpp/tools/layer-streaming/main.cpp#L2181),
[copy bypass](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L1083).

The existing [backend row-copy path](../third_party/llama.cpp/ggml/src/ggml-backend.cpp#L1629)
already reads selected IDs, deduplicates them, coalesces consecutive expert
ranges, and preserves quantized bytes and required padding. Reuse it before
building T-GPLR-13. However, its destination
[duplicates the original tensor layout](../third_party/llama.cpp/ggml/src/ggml-backend.cpp#L1404):
sparse traffic does not imply compact allocation. Allocator reuse determines
the peak. Router-ID readback also synchronizes the backend. Compact buffers,
persistent expert caching, and fewer routing barriers remain distinct work.

### P1: Current Baselines Do Not Establish Full-Context TG Performance

The quoted 64.9 tokens/s llama-bench result has `n_prompt=0` for its TG test.
Supplying separate PP and TG lengths does not make the TG test a continuation
of that PP. The 32.44 tokens/s server result uses MTP and allocates a
262000-token context, but its recorded prompt contains only 372 tokens.
Neither proves that rate after processing a 262K prompt.

Do not use either as T-GPLR-16's long-context acceptance baseline. Match actual
populated history, tokenized prompt, sampler, draft policy, quantization,
placement, and output length. Also avoid deriving host DRAM bandwidth from
whole-model active bytes if some expert banks were GPU-resident.
Evidence: [current baseline discussion](currentstate.md#L111).

### P2: Copy Windows And Capacity Estimates Can Mislead Prioritization

In [wait_slot_ready](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L1227),
`ready_latency_us` ends when the host observes the event, not when DMA
actually finished. It can include time spent doing unrelated work before
the host reaches the wait. Summing overlapping copy windows is not a
critical-path decomposition. Likewise, 18.33 GB divided by an entire
351-second PP run is not an SSD or PCIe bandwidth measurement.

The [handoff fit calculation](../third_party/llama.cpp/tools/layer-streaming/main.cpp#L2490)
includes PP activation slabs although the handoff releases slabs and slots
before allocating migrated weights. That can cause conservative refusals.
The remedy is phase-aware peak accounting, not simply deleting the reserve.
Backend copy buffers, graph workspaces, migration temporaries, and WDDM
headroom must remain included.

An experimental residency-window budget is not a hard process-wide VRAM
limit. Stock [--fit-target](../third_party/llama.cpp/common/arg.cpp#L2851)
specifies a **target margin**, not maximum VRAM usage. Neither that flag nor
a coarse memory sampler proves an 8 GiB deployment fits.

### P2: Planning Status And Quality Evidence Need Reconciliation

The generic [plan header](plans/generic-prefill-layer-residency-plan.md#L16)
says T-GPLR-17 READY and T-GPLR-16 DRAFT; their bodies say rejected and
in-progress. The [latest owner decision](currentstate.md#L10) rejects the
two-process bridge. Historical FA task bodies also retain ready statuses
after recorded completion. Follow the latest task-local decisions, not
stale summaries; do not reopen rejected work.

BF16 slab storage is already the accepted default. It is not lossless:
the MoE 100K comparison matched 0/840 route hashes, while the focused
needle answers passed. The older dense Q4_K_M Q8-KV schedule-dependent ID
divergence is a separate historical issue, not closed by the MoE decision.
Three token IDs, finite tensors, or a throughput trace do not establish broad
answer quality. Sources: [default-profile evidence](plans/generic-prefill-layer-residency-plan.md#L1744),
[dense Q8 caveat](currentstate.md#L2060).

## Recommendation

**Make end-to-end 8 GB feasibility the first gate, then finish and measure
the existing single-process selected-expert handoff path.** Keep most expert
weights in host RAM and retain only a bounded subset or staging window in
VRAM. Preserve layer-major PP, while budgeting its activation slabs as well
as TG memory. Compare CUDA selected-row execution with CPU expert execution
only after the runner's correctness investigation is resolved.

Reuse stock row copying if its actual scheduler allocation fits. If full-shaped
copy buffers exceed the budget, compact expert staging becomes a feasibility
requirement, not an optional late optimization. That is T-GPLR-13's existing
design territory and requires its approval gate, not a second unplanned pager.

Quantization should reduce transfer bytes, KV storage, and cache footprint;
do not prioritize making this checkpoint fully resident on the 24 GiB GPU.
For ideas outside the existing plans, prioritize adaptive active-expert counts,
lower-bit MoE expert profiles, single-engine prefix reuse, and cost-aware
handoff of a fitting subset. Compact row sets, caching/prefetch, partial
handoff, and MTP are already planned and are not new discoveries.

| Rank | Opportunity | Expected benefit and evidence | Effort | Quality cost |
| --- | --- | --- | --- | --- |
| 1 | Phase-aware 8 GB peak budget | Includes PP slabs, KV, scheduler copies, transitions and headroom; prevents false fits | Medium | None intended |
| 2 | Close T-GPLR-16; bounded selected-expert execution | Removes whole-bank TG amplification; reuse stock row copies if allocation fits, otherwise advance approved compact staging | Medium to high | No intentional model change; correctness gate required |
| 3 | Q4 KV and supported lower-bit expert profiles | Frees scarce VRAM and reduces expert miss bytes; full residency is not required | Low code, model evaluation required | Quantization-dependent |
| 4 | Adaptive active-expert counts | Reduces expert arithmetic and selected-row transfers without replacing the model with a smaller one | Medium to high | Potentially substantial |
| 5 | Byte-budgeted expert cache, then predictive prefetch | Saves repeated selected-row transfers if TG locality exists; cache capacity comes only from measured headroom | High | None with exact miss handling |
| 6 | Tune existing MTP; later compose with partial handoff | Amortizes verification costs; gains are prompt/depth-dependent | Low tuning, high integration | Acceptance and sampler fidelity must be checked |
| 7 | Stop at answer completion; reuse exact prefixes/session state | Avoids unnecessary generation and repeated PP; must remain within the same memory envelope | Low to medium; lifecycle work needed | None for exact reuse; possible loss with shorter reasoning/context |
| 8 | Measure and reduce diagnostic/graph overhead | Local synchronous observation and flushing exist; wall benefit unmeasured | Low to medium | None intended |

Effort describes relative engineering scope, not a delivery estimate.

## Hardware And Quantitative Bounds

Recorded environment, not freshly benchmarked:

| Resource | Recorded value | Consequence |
| --- | --- | --- |
| CPU/RAM | Ryzen 9 9955HX, 16 cores, approximately 61.69 GiB RAM | CPU expert execution is plausible; actual bandwidth and thread scaling still need measurement |
| GPU | RTX 5090 Laptop, WDDM, 24,463 MiB physical VRAM | Do not use desktop RTX 5090 compute or memory-bandwidth ceilings |
| PCIe | Gen4 x8; 15.75 GB/s theoretical; 13.7 GB/s recorded steady H2D | Full-bank TG cannot become interactive through copy tuning alone |
| Storage | MSI M461; 2,728 MB/s unbuffered QD1, 7,745 MB/s warm buffered | Distinguish storage faults from DRAM staging and DMA |
| Toolchain | CUDA 13.2, MSVC, Release build targeting sm_120; driver 610.74 | Native CUDA targeting already exists; a compiler upgrade is not a demonstrated optimization |
| Available VRAM | Recorded free-start snapshots around 22.2-22.9 GB | Physical capacity is not the process's available budget |

Sources: [AGENT_ENVIRONMENT.md](AGENT_ENVIRONMENT.md),
[corrected PCIe probe](currentstate.md#L2081).

For the current 35B-A3B MoE artifact, 37 banks contain 452,984,832 bytes;
layers 34/38/39 contain 522,190,848 bytes. Use the inventory rather than
the filename's nominal quantization or one representative layer.

$$
B_{\mathrm{selected}} = 18{,}327{,}011{,}328\frac{8}{256}
= 572{,}719{,}104\ \mathrm{bytes/token}.
$$

At the recorded 13.7 GB/s link rate, transferring all banks takes at least
about 1.34 s of link service per pass; selected rows take about 41.8 ms
if every bank is host-resident, every selected expert misses, and that
bandwidth is attainable. Routing, small-copy overhead, compute, and KV reads
are additional costs; resident banks and cache hits reduce required traffic.
These are conditional transfer bounds, not measured token-rate forecasts.

The saved 50K trace contains 54,981,033,984 TG transfer bytes across three
tokens, 6,660,108 us of summed transfer windows, and 139,882 us of layer
compute. The latter is 46.6 ms/token against approximately 1.966 s/token wall.
Even removing all of that measured compute would improve throughput by only
about 2.4% if it were entirely on the critical path. Do not infer pure DMA
bandwidth from the summed windows.

At 262,144 context capacity, the inventory gives:

| Byte class | Bytes |
| --- | ---: |
| All model tensor data, including declared MTP tensors | 21,377,327,616 |
| Routed expert banks, included above | 18,327,011,328 |
| Q8_0 K/V | 2,852,126,720 |
| Recurrent state | 65,863,680 |
| Raw model + Q8 K/V + recurrent state | 24,295,318,016 |
| Q4_0 K/V, same dimensions | 1,509,949,440 |
| Raw model + Q4 K/V + recurrent state | 22,953,140,736 |

Q4_0 uses 18 bytes per 32 values versus Q8_0's 34, so this KV reduction is
1,342,177,280 bytes (1.25 GiB), not exactly half. Block layouts are defined in
[ggml-common.h](../third_party/llama.cpp/ggml/src/ggml-common.h#L195).
Neither raw total includes every runtime allocation. Even Q4 KV alone does
not fit the recorded 22.2 GB free-start snapshot before workspace/reserve.

At this context length, BF16 MoE PP slabs consume another 2 GiB, but are
released at handoff. A hypothetical 25% reduction in expert-bank bytes saves
4.582 GB without changing other classes, but leaves 13.745 GB of expert banks
alone. It cannot deliver 8 GiB full residency. This is an illustrative byte
reduction, not an assertion about an available Q3 artifact or its quality.

The 262K PP phase also needs explicit attention. Adding those BF16 slabs
to the inventoried bounded minima gives these lower bounds, before graph
workspace, additional scheduler copies, and driver/display headroom:

| PP configuration | Inventoried minimum + BF16 slabs |
| --- | ---: |
| One slot, Q8 KV | 8,142,678,528 bytes, about 7.58 GiB |
| Two slots, Q8 KV | 8,664,869,376 bytes, about 8.07 GiB |
| Two slots, Q4 KV at the same dimensions | 7,322,692,096 bytes, about 6.82 GiB |

Two-slot Q8 already exceeds an 8 GiB capacity envelope in this accounting;
one-slot Q8 leaves only about 427 MiB before the excluded costs. Q4 KV is
therefore a potentially important capacity lever, not merely a speed tweak.
These are inventory-derived bounds, not measured process peaks or a claim
that Q4 passes long-context QA. Shorter histories reduce state/slab demand,
but cannot stand in for a required long-context acceptance test.

Use 8 GiB as the primary envelope and record the lower actual usable budget.
Larger-budget runs are diagnostic controls only. Account for the maximum live
allocation in PP, transition, and TG, plus agreed headroom. Do not make a
24 GiB run stand in for validation on an 8 GiB GPU.

## Focused Experiments

### 1. Complete Partial Handoff Before Adding A New Pager

Resolve the endpoint, dispatch, byte-accounting, and allocation issues above
inside T-GPLR-16's owner-approved boundaries. If the intended CPU-compute
contract changes to CUDA selected-row execution, return that decision to the
Planner instead of silently declaring the old contract satisfied.

Measure per host layer: selected bytes, copy count, router readback/wait,
expert compute backend, and scheduler-buffer peak. Compare fitting splits
at the same real prompt history. Later tune CPU thread count only for a
correct CPU path; 8/12/16 threads are a small candidate set, not a claim that
all 16 physical cores are always best. More threads can contend for memory
bandwidth and package power.

Success: same task-level answer gate, process allocation within the usable
8 GB-class budget in every phase, and lower end-to-end latency than a matched
whole-bank baseline. Stop on wrong-state symptoms, undefined copy-buffer
lifetime, or unexplained answer failures. Do not call those acceptable
quality loss.

### 2. Use Quantization To Reduce The Working Set

For the dense reference, the existing Q4_K_M checkpoint and Q4-KV path are
already useful controls. For the MoE target, inventory an owner-approved
lower-bit expert profile as a way to reduce miss traffic and fit more useful
experts in the same bounded VRAM cache, not to require full residency. Prefer a
calibrated artifact over blindly requantizing the current quantized file.
Keep sensitive router/shared/recurrent tensors at suitable precision when
the candidate's format allows it; compare whole-artifact quality and bytes.

Success: the resident subset/staging window, actual populated-history KV,
workspace, and reserve fit within the usable 8 GB-class budget without WDDM
spill, and useful-answer latency improves at the agreed quality budget.
Stop if lower-bit kernels are slower or quality regresses without a useful
capacity or latency gain. The complete checkpoint can remain in host RAM.

Lower-bit weights do not inherently accelerate long PP. Historical dense
Q4 and Q6 PP rates were close because attention/recurrent compute remained
expensive. The argument here is reducing expert miss bytes and the bounded
working set, not assuming that halving nominal weight bits doubles tokens/s.

### 3. Account For Phase Peaks And Amortize Handoff

Model peak memory as the maximum of PP, transition, and TG live sets, not
their sum. Include stock scheduler copies and CUDA graph/workspace storage.
Use actual per-layer package bytes when choosing a split; N layers is not
a uniform-byte fraction on mixed-quant models.

Record migration time separately from TG. The historical MoE full handoff
copies 18.327 GB once at a recorded roughly 3.08 s cost; it is not an 8 GB
candidate. For this target, compare only partial migrations whose complete
live allocation fits. Their expected output length can determine whether
the migration pays off:

$$
N_{\mathrm{break\ even}} =
\frac{T_{\mathrm{handoff}}}{t_{\mathrm{baseline/token}}-t_{\mathrm{partial/token}}}.
$$

Use matched measurements and a positive per-token saving in that equation.
A migration that pays off for 512 output tokens may not pay off for a one-token
classification. If partial handoff is not faster per token, there is no
positive break-even length from generation savings.
Success: a formerly refused shape safely fits or a short-answer request
avoids a net-negative migration. Preserve loud preflight failure.

### 4. Avoid Unnecessary Generation And Repeated Prefill

The [serial TG loop](../third_party/llama.cpp/tools/layer-streaming/main.cpp#L3434)
runs to a fixed token budget and has no end-of-generation check. Preserve
that behavior for fixed-length benchmarks, but give a future usable inference
mode explicit EOG/stop handling and the model's correct chat template.
Suppressing hidden reasoning or shortening requested answers is a separate,
quality-changing profile. Do not stop at a delimiter inside an unfinished
answer or alter the QA gate merely to report a shorter run.

For repeated documents or conversations, reuse exact prefix/session state
inside one process. Reuse the available llama.cpp state facilities, but
capture both attention KV and DeltaNet recurrent state at the same position,
with compatible model, quantization, template, and session identity. A KV-only
cache is insufficient. The existing single-request handoff teardown and
unfinished re-arm checks make this lifecycle work, not an available flag.
The rejected two-process T-GPLR-17 bridge stays rejected.

Retrieval, prompt deduplication, and summarization can reduce PP, KV capacity,
and subsequent TG attention cost. They change the workload and can lose
cross-document or long-range evidence. Compare useful-answer latency
including retrieval and summary generation, not only the final decode rate.

### 5. Measure Observation Overhead And Current PP Bottlenecks

Sampled [route telemetry](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L2481)
copies F32 probabilities to the CPU and stable-sorts all experts for each
sampled token. At width 512 and 256 experts that is 512 KiB per observation.
It is [already stride-gated](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L2234),
so do not charge it to every chunk when analyzing a sampled trace.
The [JSONL writer](../third_party/llama.cpp/src/llama-layer-streaming.cpp#L227)
flushes each event, and serial TG also prints/flushed diagnostic margins.

Start with a short matched diagnostic-cost experiment that preserves required
lifecycle/failure telemetry and has a full-audit control. Reuse actual router
IDs if their lifetime can be observed safely; do not introduce a second
approximate router. Buffer optional output only with reliable error/exit
flushing. No wall-time benefit has been measured for these changes here.

Profile the current BF16/Q8 default, not the old 351-second PP profile.
Separate host patching, allocation/split work, CUDA graph updates, kernel
time, staging, actual copy activity, and exposed waits. `graph_builds=0`
counts GGML construction, not zero CUDA graph-update/reinstantiation cost.
Only then consider local snapshot-copy reduction, graph reuse, or ubatch
tuning. Changes above the approved ubatch envelope need a new task contract.

Dense historical PP improved from 506.2 to 704.6 tokens/s after bounding
attention to the valid causal prefix. That optimization is already done and
is not lossy sliding-window attention. The later 738.6 tokens/s patch-cache
result has an explicit run-variance caveat; do not attribute its full gain
to pointer caching. Likewise, the old 804.66 tokens/s resident control is
not a permanent hardware ceiling. Sources: [FA window](currentstate.md#L1879),
[patch-cache attribution](currentstate.md#L2019).

Success: a repeatable reduction in the identified critical-path component
and request wall time. Stop if only summed overlapping intervals improve.

### 6. Tune MTP For Useful Accepted Tokens

Existing MoE batch verification already amortizes whole-bank loads. The
synthetic 128-token series improved from 245.4 s serial to about 82.6 s at
draft depth 8; depth 32 was slower. On the loop-controlled code prompt,
depths 8/16/32 recorded about 1.68/3.54/1.68 tokens/s. Earlier approximately
7 tokens/s results were repetition-contaminated, and the quoted 6.91x
comparison used different prompt lengths. Neither is a general speedup.

Optimize accepted useful tokens divided by total draft, verify, rollback,
and synchronization time. Keep prompt and repetition policy identical.
Dense resident MTP at depth 2 was slower, 23.6 versus 25.4 tokens/s, despite
matching the recorded sequence. MTP is not a universal default-on win.

MoE handoff currently refuses MTP; combining it with partial residency is
new integration work. Sparse verification also changes the tradeoff: width
W may require up to `min(256, 8 * W)` experts per layer. At depth 32,
width 33 can cover the full bank. Measure route unions and transfer bytes
per accepted token before choosing large draft depths or speculative trees.
Sources: [T-GPLR-06](plans/generic-prefill-layer-residency-plan.md#L751),
[dense MTP result](currentstate.md#L2268).

### 7. Cache Experts Only After Measuring TG Reuse

T-GPLR-14 supplies an offline analyzer, not evidence for a 60% TG cache hit
rate. Whole-vector route-hash inequality does not mean zero shared experts;
sampled PP IDs do not establish per-token TG locality either.

First collect per-layer TG IDs and exact expert byte costs at the target
history. Compare a small byte-budgeted reactive cache against the existing
row-copy baseline. Keys must include model identity, logical layer, expert,
and representation; identical numeric IDs in different layers are unrelated.
Maintain generation/copy/consumer lifetime protection and exact miss handling.

Only then evaluate T-GPLR-15 prediction: useful prefetch bytes, wasted bytes,
late arrivals, eviction harm, and saved critical-path time. A hit-rate
threshold alone is not a performance gate. Stop if metadata, barriers, or
speculative copies cost more than the transfers saved. A compact row-set
policy changes the current complete-package SDD boundary and needs approval.

### 8. Deliberately Trade Quality For Less Work

After establishing a trustworthy baseline, compare these separately rather
than changing every variable at once:

| Candidate | What it can save | Main limit |
| --- | --- | --- |
| Q4 KV instead of Q8 | 1.25 GiB at the inventoried 262K MoE capacity; possible decode bandwidth savings | Long-context retrieval quality and backend conversion costs need measurement |
| Shorter context or retrieval-selected context | PP work, KV, and TG attention reads | Can remove evidence; existing valid-prefix bounding does not provide this approximation |
| Smaller replacement model, outside the stated goal | Weight residency, arithmetic, and often KV | Not a solution for running the same MoE; requires a separate owner decision |
| Top-8 to top-6 or top-4 routing | About 25% or 50% fewer selected expert rows and associated expert arithmetic | Not the same model; quality loss is task/layer-dependent |
| Calibrated physical expert pruning | Actual model/bank storage, potentially enabling residency | Requires a new artifact, router remapping and validation; merely lowering top-k does not shrink banks |
| Shorter reasoning/output budget | Generated-token count and answer latency | Premature truncation or weaker reasoning |

**Lowering top-k alone does not reduce whole-bank PP/TG transfer bytes.** It
only attacks weight traffic when combined with selected-row execution, or
when a new physically smaller expert bank is produced. Do not alter GGUF
metadata and pretend residency remains numerically unchanged. Keep shared
experts, router normalization, attention, and recurrent updates well-defined.
Layer skipping or stale recurrent/KV state is not recommended as an initial
quality tradeoff: it changes too much with no local calibration evidence.

## Plan Coverage

Every plan below was read in full. Status is reconciled against task-local
results and current-state evidence, not copied blindly from its header.

| Plan | Reviewed outcome | Performance implication |
| --- | --- | --- |
| [plans/qwen38-layer-streaming-plan.md](plans/qwen38-layer-streaming-plan.md) | Core slots, event lifecycle, persistent variants, PP-to-TG continuity and telemetry done | Foundational correctness exists; old batch-major repeated transfers are not the current long-PP strategy |
| [plans/qwen38-pp20k-plan.md](plans/qwen38-pp20k-plan.md) | Layer-major PP, timing split, class-specific allocation and optional pinned staging done | One package load per layer across prompt chunks is the major existing PP win; pinned A/B did not improve wall time |
| [plans/qwen38-arbitrary-pp-plan.md](plans/qwen38-arbitrary-pp-plan.md) | Real tails, graph-footprint preflight and regression closure done; overlap work deferred | Do not reintroduce fake padding or redesign streams without an exposed-wait profile |
| [plans/qwen38-perwidth-graphs-plan.md](plans/qwen38-perwidth-graphs-plan.md) | Per-width graph reuse, bounded inputs and BF16 slabs done; historical Q4-KV gate later superseded by handoff work | Host graph reservation fell from about 19.7 GB to 288 MB at 100K; this is not remaining low-hanging work |
| [plans/qwen38-fa-window-plan.md](plans/qwen38-fa-window-plan.md) | Valid-prefix K/V bound done | Recorded dense 100K gain of 39.2%; not an approximate context window |
| [plans/qwen38-fa-efficiency-plan.md](plans/qwen38-fa-efficiency-plan.md) | Resident control, fused-op audit and patch-walk cache completed in evidence; stale ready fields remain | Fused GDN already active; current-profile overhead attribution comes before new kernels |
| [plans/qwen38-hybrid-tg-plan.md](plans/qwen38-hybrid-tg-plan.md) | Corrected dense rotation feasibility remains NO-GO | Required 45-50 GB/s exceeds the 15.75 GB/s physical link; window-size tuning cannot fix that ratio |
| [plans/qwen38-pp-tg-handoff-plan.md](plans/qwen38-pp-tg-handoff-plan.md) | Q4 KV, full/attention-only handoff done; dense MTP done per state; re-arm and GTX 1080 validation remain gaps/gates | Reuse handoff; do not transfer short-context rates or dense MTP conclusions blindly to MoE |
| [plans/generic-prefill-layer-residency-plan.md](plans/generic-prefill-layer-residency-plan.md) | Generic dense/MoE PP support, MTP, QA, full handoff, tail fix and defaults implemented; later tasks mixed status | T-GPLR-16 is the immediate unfinished TG path; custom row sets/cache require evidence and approval |

Generic backlog reconciliation:

- T-GPLR-00 through 05: inventory, adapters, full MoE PP, capacity and
  default-off/capability evidence complete. The initial 262K gate allocated
  context for a 512-token PP, not a 262K-token prompt.
- T-GPLR-06/07: MTP and semantic-QA machinery exist. Long answer-quality
  rungs remain owner-invoked; throughput completion is not a quality pass.
- T-GPLR-08: prefetch probe completed with a negative performance result,
  350.7 s at N=0 versus 352.9 s at N=1. T-GPLR-09 host locking is parked.
- T-GPLR-10/11/12: full MoE handoff, batch-major tail-width repair, and
  BF16/Q8/layer-major/auto-handoff defaults done. The latest recorded 262K
  auto-off PP is 1160.9 tokens/s, not the older approximately 745 tokens/s.
- T-GPLR-13: compact reactive expert row sets remain draft. T-GPLR-14:
  offline route analysis done. T-GPLR-15: predictive prefetch remains draft,
  with its TG-locality premise unmeasured.
- T-GPLR-16: in-progress, with the source risks described above.
  T-GPLR-17: rejected and reverted; do not restart it.

Historical state also retains superseded bandwidth and support claims near
its bottom. Prefer dated corrections, exact artifact/flags, and surviving
traces. The accepted BF16 profile does not validate every quant or schedule.

## Online Research

Primary sources checked for this review. Paper speedups are not predictions
for this laptop, model, quantization, recurrent architecture, or Windows build.

| Source | Relevant mechanism | Local interpretation |
| --- | --- | --- |
| [Fiddler](https://arxiv.org/abs/2402.07033) | CPU/GPU expert execution to avoid transferring large weights when VRAM is insufficient | Supports comparing CPU expert arithmetic with GPU offload; not proof of runner correctness or a drop-in Qwen implementation |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Heterogeneous CPU/GPU inference and optimized expert kernels | Current project includes AVX2/AVX-512 support, not only Intel AMX; its server results do not establish Ryzen laptop performance or compatibility with this artifact/current Windows stack |
| [AdapMoE](https://arxiv.org/abs/2408.10284) | Sensitivity-aware active-expert reduction plus prefetch/cache scheduling | Motivation for calibrated top-k experiments, not evidence that top-4 is harmless on this model |
| [SpecExec](https://arxiv.org/abs/2406.02532) | Amortize offloaded-model verification across speculative candidates | Existing MTP is the smaller local starting point; speculative trees would need new recurrent-state rollback and memory proofs |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device) | Minimize transfers; batch small copies; use pinned memory judiciously | Supports byte reduction first, not blanket pinning of all expert banks |
| [CUDA synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) and [events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) | Pageable Async copies may stage/block; event elapsed timing differs from host observation | Use properly timed device activity plus a timeline; preserve host submission and exposed wait as separate metrics |
| [PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory) | Strong performance hint; cached pages are not inserted into the target process working set by prefetch | Successful hint calls do not establish pinned/resident DMA sources or explain a wall-time bottleneck |
| [VirtualLock](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtuallock), [SetProcessWorkingSetSize](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize), [AllocateUserPhysicalPages](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-allocateuserphysicalpages) | Distinct working-set and physical-page locking mechanisms | Correct the privilege inference before treating it as a platform impossibility; do not reopen the parked task automatically |

The environment/plan inference about `SeLockMemoryPrivilege` is too broad.
Microsoft explicitly requires it for `AllocateUserPhysicalPages` (AWE).
`VirtualLock` is constrained by committed-page accessibility and the process
working-set limit; increasing that limit uses `SeIncreaseWorkingSetPrivilege`.
A disabled privilege is not the same as an absent privilege. The missing AWE
privilege alone does not prove `VirtualLock` unavailable. No privilege change,
large lock attempt, or residency claim was made in this review. The parked
task still needs an owner decision and a demonstrated critical-path need.

CUDA-registered pinned memory, Windows working-set locking, and OS prefetch
are not interchangeable. Registering an existing region could avoid an
extra staging memcpy, but support, alignment, resource pressure, lifetime,
and actual wall benefit must be tested in a separately approved task.
The current negative pinned-staging result is not proof that every possible
registered-memory pipeline is useless; it is evidence against defaulting
the existing implementation on.

## What Not To Pursue First

- Rejected two-process engine switching, or parked whole-bank host locking
  without reopening their owner gates.
- Dense whole-layer TG rotation on this Gen4 x8 link. The corrected negative
  result survives even at the measured 13.7 GB/s.
- More PP slots or more streams when the required copy is already ready.
  Dense 100K two-slot PP recorded only 0.6 ms availability wait.
- Calling `PrefetchVirtualMemory` success a throughput improvement. Its local
  N=1 experiment missed both performance gates.
- Reimplementing selected-row copies that the backend already supplies.
  Compact VRAM allocation and persistent caching are different requirements.
- SSD or RAM upgrades based on mislabeled transfer timers. First establish
  hard faults, working-set pressure, or sustained host bandwidth saturation.
- Treating WDDM shared GPU memory as free extra VRAM, or assuming ReBAR,
  mapped zero-copy, GDS, or additional streams bypass physical PCIe limits.
- Switching frameworks, installing a new kernel stack, or assuming NVFP4 is
  a quick flag for these GGUF tensors. Verify format, architecture, Windows,
  and model support; new dependencies require approval.
- Broad kernel rewrites or global fast-math changes before critical-path
  profiling. They cannot remove the recorded full-bank TG transfer floor.

For existing hardware, record AC/power mode, sustained CPU/GPU clocks,
temperatures, actual free VRAM, and cooperating/foreign GPU work during
authorized measurements. CPU and GPU can share laptop thermal/power limits.
Do not change drivers, OS security settings, clocks, or power limits as part
of this report. Buying more VRAM would change the deployment constraint,
not satisfy the goal of running this MoE on an 8 GB-class GPU.

## Acceptance Protocol

These are proposed comparison requirements, not commands executed here and
not replacements for the governing task gates.

1. Pin build and artifact identity, actual tokenized prompt, populated
   history length, KV/slab precision, placement, sampler/repetition policy,
   draft depth, and output stopping policy. Distinguish context allocation
   from processed tokens. Reserve context space for the generated suffix.
2. Compare load/setup, PP, handoff, first-answer latency, TG, and total
   useful-answer latency. Keep cold-start and reused-session results separate.
   Use enough output tokens to assess TG, not only a three-token handoff
   probe. Report latency spread, not a single optimistic average.
3. Account for all H2D/D2H sources and peak device allocations, including
   scheduler copies, graph storage, workspaces and transients. Label host
   submission, event/timeline copy duration, compute, and exposed waits
   separately. Do not insert global synchronization into every production
   step just to simplify timing. CUDA events require timing-enabled events;
   concurrent activity can still affect their intervals.
4. For unchanged-model placement work, keep the existing semantic-QA smoke
   and negative control, lifecycle/refusal checks, default-off behavior,
   and dense Qwen regression. Preserve exact expert identity and staged bytes.
   Do not replace answer grading with token-ID equality or an LLM judge.
5. For quality-changing profiles, agree on the quality budget before testing.
   Retain correctness gates and add programmatic real-work checks: code tests,
   structured extraction, factual questions across document positions, and
   reasoning tasks. Report pass-rate loss alongside latency and repetition
   failures. Five needles alone cannot certify broad reasoning/code quality.
6. Run every GPU command through [tools/with-gpu-lock.ps1](../tools/with-gpu-lock.ps1).
   Start with the smallest task-scoped checks. The 4K-and-longer QA ladder
   stays manual; 24K/50K/100K/200K runs are owner-invoked one at a time,
   never an unattended matrix. Follow the exact applicable commands in the
   active plan and record results in state only when that task is performed.

## Review Verification

This review used source inspection, saved evidence, inventory arithmetic,
and public primary documentation. The saved 50K TG phase summary was parsed
host-only and confirmed the three-pass byte count and timing totals quoted
above. Report validation checks local links, coverage of all nine plans,
ASCII/whitespace, and byte arithmetic. No new inference rates, quality
results, GPU profiles, builds, dependencies, or runtime fixes are claimed.