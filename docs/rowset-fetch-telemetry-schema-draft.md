# rowset_fetch telemetry schema (T-GPLR-13)

Status: TRACKED design doc (tracked 2026-09-13; owner decision — track, do NOT
inline into the plan). It is the normative Phase A spec cited by the T-GPLR-15
contract in `docs/plans/generic-prefill-layer-residency-plan.md` (section 6 for
its analyzer extensions, section 3 for the `union_ids` rule), by
`docs/currentstate.md` (section 6) and by
`docs/gplr13-bottlenecks-hypotheses.md` (section 4.2 and open decision 3), and
it records SHIPPED T-GPLR-13 behavior rather than a proposal:
the `union_ids` canonical string of section 7 decision 2 is implemented
(`third_party/llama.cpp/src/llama-layer-streaming.cpp:3643`) and the section 8
validations ran under T-GPLR-13 (see the T-GPLR-13 entries in
`docs/currentstate.md`). It elaborates the amended SDD sections 4 and 10 for the
`moe-row-set` policy, and nothing here alters routing, placement, or default-off
behavior.

[SUPERSEDED 2026-09-13 — retained as written] Status: draft design note
(2026-09-12, owner-requested sketch before any implementation). No code is
changed by this note; the schema enters the T-GPLR-13 task contract only after
gate (b) (owner priority) resolves and the task is marked ready.

Design rule: one authoritative staging event per layer per TG window, reusing
the existing JSON Lines conventions (`run_id`, `t_us`, `ubatch`, `layer`,
`slot` = `il % n_slots`, `generation`, `phase`) and the existing field names of
the `route` event where the semantics match, so the T-GPLR-14 tool and the
checker need minimal new parsing.

## 1. Emission plan

In a `moe-row-set` TG phase, per generated window W (default one token), per
MoE layer, in causal order:

1. attn+router graph runs (existing `layer_compute`-style events may cover it;
   the split-point decision is T-GPLR-13 implementation review);
2. `rowset_fetch` — one per layer/window, at the staging act (below);
3. one aggregated `copy_submit` / `copy_complete` pair for the window's row
   batch, when any rows are copied (`bytes` = `bytes_staged`; zero-copy
   windows emit no copy pair);
4. FFN graph runs (existing `layer_ready_wait` / `layer_compute` events).

Per TG round (one decode step): one `rowset_round` byte-account aggregate.
Once per run: `rowset_init` at activation, and `run_start` gains
`"rowset":"on"` when the policy is active (absent when off — the default-off
immunity check stays a negative assertion).

Failure: `rowset_refused` (below) followed by the loud `run_end` `failed:true`
path (exit 2). Never a silent skip.

PP phases emit nothing new; `route` events at PP-chunk granularity are
unchanged. TG in row-set mode emits NO separate `route` events — `rowset_fetch`
is the single per-token route record (one source of truth; the T5 code path in
`tools/route_overlap/analyze.py` switches its per-token source to
`rowset_fetch`).

## 2. Event schemas

### rowset_init (once, at begin_row_set)

```json
{"event":"rowset_init","run_id":U64,"policy":"moe-row-set","n_slots":I,
 "slot_bytes":U64,"window_tokens":I,"n_expert":I,"top_k":I,"t_us":I64}
```

- `n_slots` / `slot_bytes`: allocated full-bank-shaped row-set slot buffers.
- `window_tokens`: declared W (1 unless the implementation-review decision
  selects a small token group).
- `n_expert` / `top_k`: from the model inventory (e.g. 256 / 8 on Qwen3.6);
  the checker scopes all ID range checks from this event.

### rowset_fetch (per layer, per TG window)

```json
{"event":"rowset_fetch","run_id":U64,"ubatch":U64,"layer":I,"slot":I,
 "generation":U32,"chunk":I64,"phase":1,"n_tokens":I,"top_k":I,
 "route_union":I,"union_ids":"e1,e2,...","hash":U64,
 "n_rows_fetched":I,"n_rows_resident":I,
 "bytes_staged":U64,"bytes_full_bank":U64,
 "host_submit_us":F64,"transfer_us":F64,"ready_latency_us":F64,"t_us":I64}
```

- `chunk`: TG window ordinal, 0-based at the window's first decode step
  (matches the `route` event's field name; TG windows sort with `tg_tokens`
  order). For W>1 this is the window-start ordinal.
- `n_tokens`: W, the window width (1 by default).
- `top_k`: router cardinality, repeated per event so each line is
  self-contained (same convention as `route`).
- `route_union`: deduplicated row-union width |U| — same name and meaning as
  `route.route_union`, so the T1 width statistics run over both unchanged.
- `union_ids`: the FULL deduplicated routed expert IDs of the window,
  ascending, comma-joined, no spaces (string form, matching
  `route.sampled_ids` style and the FNV canonical input exactly).
- `hash`: FNV-1a-64 over the comma-joined `union_ids` bytes — the same
  canonicalization as `_fnv1a64` in `tools/route_overlap/analyze.py`, so
  offline tools recompute and cross-check it. Distinct from the PP `route`
  hash (selection order with per-token repeats); never compared across event
  types.
- `n_rows_fetched`: rows copied in THIS window (missing from the slot).
- `n_rows_resident`: rows already resident before this fetch (skipped).
  Identity: `route_union == n_rows_fetched + n_rows_resident`.
- `bytes_staged`: quantization-exact bytes actually copied this window
  (gate/up/down across fetched rows, natural offsets).
- `bytes_full_bank`: this layer's complete expert-bank bytes — the constant
  denominator for the union-width saving (SDD: measurable, never assumed).- `host_submit_us` / `transfer_us` / `ready_latency_us`: the staging batch's
  distinct timings (SDD section 10); compute and graph-setup times stay on the
  existing `layer_compute` events for the FFN graph; wall time stays in
  `phase_summary`. No timing is double-booked.
  [2026-09-15: the row-set TG executor now feeds the same `add_layer_stats`
  pipeline the streaming executors use, so the TG `phase_summary` in
  `moe-row-set`  modes reports non-zero `total_transfer_us` (submit+ready of the
  window's staging), `total_wait_us` (ready-latency) and
  `bytes_transferred` (staged bytes) instead of zeros. Additive; no
  `rowset_*` field changed.]
  [2026-09-14: that `layer_compute` event is now emitted by the row-set TG
  executor through the single `tel_layer_compute` writer (the merge that removed
  the short-lived `tel_layer_compute_parts`). Still exactly one per
  layer/window, still with the unchanged `compute_us`/`setup_us` (the two-part
  sums), plus `attn_compute_us`/`attn_setup_us` and
  `ffn_compute_us`/`ffn_setup_us`. This field list is ADDITIVE to the schema
  above; no `rowset_*` field changed.]

### rowset_round (per TG round)

```json
{"event":"rowset_round","run_id":U64,"round":I64,"n_tokens":I,
 "bytes_staged":U64,"bytes_full_bank":U64,
 "n_rows_fetched":U64,"n_rows_reused":U64,"t_us":I64}
```

Sums across the round's 40 layer windows. `n_rows_reused` = sum of
`n_rows_resident` (rows the reactive policy got for free). This is the
byte-account symmetry point with `tel_byte_account` and the headline
"staged bytes vs full-bank" ratio per round.

### rowset_refused (failure; always followed by loud run_end)

```json
{"event":"rowset_refused","run_id":U64,"layer":I,"slot":I,"generation":U32,
 "reason":"...","detail":"...","phase":1,"t_us":I64}
```

`reason` enum: `missing_row` | `stale_generation` | `insufficient_vram` |
`bad_shape` | `coverage_gap`. Structured, never prose-only.

### run_refused (T-GPLR-22; run-level refusal, TERMINAL; added 2026-09-14)

```json
{"event":"run_refused","run_id":U64,"reason":"...","detail":"...","exit_status":I,
 "bytes_needed":U64,"bytes_allowed":U64,"cap_source":"free|budget|none",
 "phase":I,"t_us":I64}
```

A *run-level* refusal, distinct from `rowset_refused` (which refuses one
layer/window's staging inside a run that continues). `run_refused` is the LAST
event in the trace and closes the sink: nothing may follow it. No timed phase
may precede it either, but a `run_start` may — a pre-timed check that stops a
run which already began writing is a refusal, not a crash (see the third shape
below).

There are two trace shapes, and three reachable refusals:
- **PRE-SINK** (added 2026-09-14) — **before `ls.init` exists**: the T-GPLR-04
  capacity preflight and the T-GPLR-10 / T-GPLR-16 handoff fit gates. A static
  one-shot emitter opens the path, writes the record and closes it, so the
  trace is that ONE line and nothing else — no `pkg_member` catalog, because no
  catalog was ever written. No instance state and no init are involved, which
  is why the emitter is static; it mirrors the existing one-shot `tel_off_run`
  precedent.
- **POST-INIT** — **after `ls.init` opened the sink**: the trace carries the
  catalog `ls.init` wrote, so `run_refused` follows the 40 `pkg_member`
  records, and `byte_account` / `run_start` (and, for a graph-budget refusal,
  `graph_footprint`) appear in their normal order before it. Two refinements of
  this shape:
  - the pre-run gates (the row-cache and prefetch refusals) stop the run before
    `run_start`;
  - **PRE-TIMED ABORT** (added 2026-09-14) — a check between `run_start` and the
    first timed phase stops an already-writing run, so `run_refused` is emitted
    instead of the `run_end` it used to write. `--graph-budget-mib 1`, a failed
    `ls.init` and a failed PP graph prebuild are the reachable instances. This
    shape is why `run_start` is admissible without ever being required.

- `reason` enum (both the emit sites and the checkers use this one list, so a
  reason cannot be emitted without being checkable): `row_cache_derive` |
  `row_cache_fit` | `row_cache_depth` | `row_cache_alloc` |
  `row_cache_divergence` | `prefetch_configure` | `capacity_preflight` |
  `handoff_preflight` | `init_failed` | `prebuild_failed` | `graph_budget`.
  The two `*_preflight` reasons are the PRE-SINK ones; `detail` names the
  exact gate (inventory, window parse, full-fit override, budget, free VRAM for
  `capacity_preflight`; T-GPLR-10 full-handoff or T-GPLR-16 partial-handoff fit
  for `handoff_preflight`) and `cap_source` the limiting term, so the deciding
  condition stays readable from the JSONL alone. The last three are the
  PRE-TIMED ABORT reasons: `init_failed` (`ls.init` returned false),
  `prebuild_failed` (the PP persistent-graph prebuild failed) and
  `graph_budget` (reserved io bytes exceeded `--graph-budget-mib`, the only
  reason that carries a non-zero `bytes_needed`).
- `detail`: free text; JSON-unsafe characters (quote, backslash, control) are
  folded to spaces by the producer, so the line always parses.
- `bytes_needed` / `bytes_allowed` / `cap_source`: the arithmetic that decided,
  and which cap bound it; 0 / `none` where the refusal is not about bytes.
- Present on every refusal the runner can reach: the ones that previously left
  a partial trace (the row-cache and prefetch gates), since 2026-09-14 the ones
  raised before the sink existed (the capacity and handoff preflights), and
  since 2026-09-14 also the pre-timed aborts that used to leave a trace the
  checker REJECTED (the `--graph-budget-mib` abort) or wrote nothing at all (a
  failed `ls.init`). Still traceless: the load/setup failures (model load,
  context creation, capability probe, `--inspect-model`) and the exit-2 usage
  errors — none of those is a byte-arithmetic refusal, and the earliest happen
  before the telemetry path is parsed. A pre-timed abort reached through the
  `init_failed` path is emitted but cannot be persisted when the telemetry path
  itself is the unopenable thing: the one-shot emitter uses the same `fopen`,
  so it fails the same way and only warns on stderr.

Why it exists: before this event a refused run left 40 `pkg_member` records and
no `run_start`, so neither the checker nor a harness could tell "refused because
of X" from a crash. The event makes the refusal recoverable from the JSONL
alone, without stderr.

## 3. Canonicalization rules

- `union_ids`: sorted ascending, unique, comma-joined ASCII, no spaces.
- `hash`: FNV-1a-64 over exactly that byte string (== `_fnv1a64`).
- Window union membership is computed from the router's executed IDs only
  (reactive); the event never contains predicted IDs. If T-GPLR-15 later adds
  prediction, its prefetch decisions appear as timing/byte effects, never as
  altered `union_ids` — the reactive superset gate in the SDD stays checkable.

## 4. Checker (check-telemetry) obligations

Per event, all checkable offline from the JSONL + inventory:

1. `union_ids` parse: integers, strictly ascending, unique, all in
   `[0, n_expert)`; `len == route_union`.
2. Width bounds: W=1 windows have `route_union == top_k` exactly (top-k of
   distinct indices per token). W>1 windows have
   `top_k <= route_union <= min(n_expert, n_tokens*top_k)`.
3. Accounting identity: `route_union == n_rows_fetched + n_rows_resident`;
   `bytes_staged > 0` iff `n_rows_fetched > 0`.
4. Completeness (SDD gate 8): for W=1, per token
   `sum(route_union over layers) == n_moe_layers * top_k` — union lists sum to
   per-token route counts. For W>1, bounds only (a further reason the default
   W is 1).
5. Generation lifecycle: within one `(layer, slot, generation)` chain,
   `n_rows_resident(t+1) == n_rows_resident(t) + n_rows_fetched(t)`
   (full-bank-shaped slots never evict mid-generation; reuse requires a
   generation bump). First window per layer per generation:
   `n_rows_resident == 0`, `n_rows_fetched == route_union`.
6. Hash integrity: recomputed FNV-1a-64 of `union_ids` == `hash`.
7. Byte baseline: `bytes_full_bank` == the layer's bank bytes from the
   inventory/pkg-member records; aggregated `copy_complete.bytes` ==
   `bytes_staged` for windows that copied.
8. Ratio: per-round `bytes_staged / bytes_full_bank` reported in the summary;
   the acceptance gate reads this ratio, not a claimed speedup.
9. Default-off immunity: with the option off, no `rowset_*` events exist and
   `run_start` carries no `"rowset"` key.
10. Refusal trace (T-GPLR-22, added 2026-09-14): a trace carrying `run_refused`
   (whether it also carries the `pkg_member` catalog, as a post-`ls.init` refusal
   does; or is that single line alone, as a PRE-SINK refusal is; or begins with
   `run_start` and stops before any timed phase, as a PRE-TIMED ABORT does) must
   contain no `phase_summary` and no `run_end`, and its `reason` must be in the
   enum above. `run_start` is ADMISSIBLE and neither required nor forbidden: it
   is present exactly when the refusal was raised after the run began writing.
   The three "missing run_start/phase_summary/run_end" rules describe a run that
   started, so they are SUPPRESSED for a refusal trace and replaced by this one.
   A trace with no `run_refused` is untouched by this rule, so traces and
   fixtures written before it keep exactly their prior verdicts — including the
   pre-event refusal shape (a bare `pkg_member` fragment), which is still
   rejected rather than retroactively accepted. The one pre-existing negative
   fixture that carries `run_start` keeps its verdict through the
   no-`phase_summary` clause rather than through the old blanket `run_start`
   rejection: accepting a started run as an aborted refusal is narrower than
   accepting a started run that already completed a phase.

## 5. What T-GPLR-15's analysis consumes

| T-GPLR-15 input | Source in this schema |
| --- | --- |
| 0.6 gate (SDD gate 9) | statistic (b): containment of the NEXT window in the CUMULATIVE resident set, computed offline by simulating bounded retention over the per-layer `union_ids` streams (SDD section 10, gate 9 decision rule). Consecutive-window `hash` equality per layer (T5 over `rowset_fetch`) is reported as CONTEXT only — never the verdict |
| Exact overlap (upgrades T4 from advisory) | `union_ids` containment `|A n B| / |A|` between consecutive windows — full lists, not samples |
| Predictor benefit ceiling | `n_rows_reused` (reactive reuse already captured) vs exact containment of the next window in the cumulative resident set |
| Window-size W decision | per-window `route_union` volatility and byte ratios at W=1 vs W>1 probe runs |
| Slot-count decision | per-layer row reappearance distance, computed offline from the `union_ids` streams |
| Timing attribution | `host_submit_us` / `transfer_us` / `ready_latency_us`; per-part compute/graph-setup on the existing `layer_compute` (`compute_us` + `setup_us` unchanged, split by `attn_*` / `ffn_*` since 2026-09-14) |

## 6. tools/route_overlap/analyze.py extensions required

The gate verdict is SDD statistic (b) per the SDD section 10 gate 9 decision
rule; the remaining items are context or telemetry-integrity work.

IMPLEMENTED 2026-09-13 (T-GPLR-15 Phase A): every item below is in
`tools/route_overlap/analyze.py`. That file entered version control on
2026-09-13 (commit 1dbdaf3), as did this spec, so the producer of these numbers
is present in a clean checkout; the rest of the T-GPLR-14 harness (the `.ps1`
runners, the MEMPROBE tools) is still untracked, and the `build/` traces the
analyzer reads are gitignored, so re-deriving the numbers needs a fresh ladder
run. Verified by reproducing the
recorded ladder
pilot exactly (0/80 hash-equal; statistic (b) at K=1 = 0.3766 / 0.3703 /
0.3891 / 0.3578 at 50/512/4k/200k; first K=2 values 0.4031 / 0.4031 / 0.4266 /
0.4000). Note for the recompute cross-check: `tel_rowset_fetch` seeds its hash
with the literal `1469598103934665603`, which is NOT the standard FNV-1a offset
basis — the analyzer must use the vendor literal or every real event would
mismatch.

- Add the OFFLINE simulation that produces statistic (b): containment of the
  NEXT window's routed rows in the CUMULATIVE per-layer resident set, by
  simulating a declared bounded retention over the recorded per-layer
  `union_ids` streams (retention K windows of width W=1, capacity sized per
  SDD section 4: `min(n_expert, K*W*top_k)` rows x that layer's member slab
  bytes). Report a per-K curve across the reference retention band (K=1-2 on
  the Qwen3.6 target, ~0.57-1.13 GB across 40 MoE layers); the gate value is
  (b) at the retention the adopting F2 task declares, with the next window in
  the denominator (`|A n B| / |B|`).
- Add the per-layer reappearance-distance histogram, computed offline from
  the `union_ids` streams — the slot-count decision input, not a gate value.
- Add the ORACLE ceiling (T8; owner request 2026-09-13): statistic (b) under
  UNBOUNDED retention — the fraction of each window covered by the union of all
  prior windows for that layer. No cache policy of any size can exceed it, so it
  answers whether any policy could reach the 0.6 gate on the same stream. It is
  monotone in stream length, so the report must carry that caveat wherever the
  ceiling is compared against the threshold, and it is reported with pooled
  hits/required totals so exact saturation by a reported K is detectable.
- Accept `rowset_init`, `rowset_fetch`, `rowset_round`, `rowset_refused` as
  known events (today unknown types exit 2 with file:line — the list must
  grow, loudly validated as before).
- T5 MEASURED source becomes `rowset_fetch` (per layer, sorted by `chunk`,
  consecutive `hash` equality). CONTEXT ONLY, never the gate verdict — at
  top_k=8 over n_expert=256 exact window identity is structurally near zero.
  W=1 windows only for the strict per-token verdict; W>1 windows report
  window-level overlap with the width stated.
- T4 containment becomes exact from `union_ids` (replacing the sampled-ID
  advisory where row-set traces are present). CONTEXT: state the direction
  (`|A n B| / |A|` or `/ |B|`) wherever it is reported.
- Hash recompute cross-check on every `rowset_fetch` (exit 2 on mismatch) —
  a telemetry-integrity check, independent of which statistic the gate reads.
- Determinism and byte-stability rules of the tool are unchanged.

## 7. Open implementation-review decisions (inside T-GPLR-13, none owner-gating)

1. Aggregated `copy_submit`/`copy_complete` pair per window (recommended, as
   above) vs timing carried only inside `rowset_fetch`.
2. `union_ids` as comma-joined string (recommended; matches the FNV input)
   vs a JSON array (`tg_tokens` style).
3. Split point of the two-part graph and whether attn+router emits its own
   `layer_compute`-class event (SDD leaves the split point to review).
   **RESOLVED 2026-09-14:** the split point is unchanged (part A ends at the
   router; part B is the FFN graph) and part A does NOT emit its own event -
   both parts are timed and reported on the SAME per-layer `layer_compute`
   record via `attn_compute_us`/`attn_setup_us` and
   `ffn_compute_us`/`ffn_setup_us`, so the event count stays one per
   layer/window and every existing consumer of `compute_us` is unaffected.
   Chosen over two events per layer to keep the "one authoritative record per
   layer per window" rule and the trace-volume bound of decision 5.
4. `chunk` semantics for W>1 (window-start ordinal, as specified here).
5. Trace volume: ~40 events/token at ~230 B ~= 9 KB/token of JSONL
   (0.6 MB per 65-token QA run; ~9 MB per 1k-token TG run) — acceptable;
   gzip only if a future ladder run needs it.

## 8. Validation of this schema itself (when T-GPLR-13 starts)

- Checker fixtures: hand-built JSONL lines covering the W=1 identity, the
  generation-lifecycle chain, hash mismatch, out-of-range ID, and the
  default-off absence assertion — each must pass or fail exactly as stated.
- One 512-token row-set run: checker OK per section 4, `rowset_round` ratios
  present, `union_ids` streams parse.
- Dense ctest 2/2 and qa-smoke 512 stay green (no new events on the
  default-off path).
