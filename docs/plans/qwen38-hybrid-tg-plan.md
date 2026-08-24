# Qwen3.8 Hybrid TG Plan (attention-resident + recurrent slot rotation)

Status: T-Q38TG-00 done (NO-GO, 2026-08-24); T-Q38TG-01 permanently gated
(Planner-authored 2026-08-24; probe verdict recorded 2026-08-24)

Design: [Qwen3.8 Dense Layer Streaming SDD](../qwen38-layer-streaming-sdd.md)
Predecessors: [per-width graphs plan](qwen38-perwidth-graphs-plan.md) (done),
[FA window plan](qwen38-fa-window-plan.md) (ready, independent).
Environment facts: [docs/AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md).
Observed state: [docs/currentstate.md](../currentstate.md).

## Problem

Token generation (TG) with the four-slot streaming executor is
transfer-bound: the 1-token Q4_K_M TG trace shows transfer 1,787.7 ms vs
compute 69.5 ms (96% transfer). With 4 slots and 64 layers every token
evicts and re-copies all 64 packages (~15 GB/token); slot reuse across
tokens is zero. Measured: ~0.4 tok/s.

Stock llama.cpp partial offload (`llama-server -ngl 17`, build b10509,
Q6_K_XL, 2026-08-23/24 owner run) does better by never copying: 16
attention layers + output stay GPU-resident, 48 recurrent layers run on
CPU from mmap. Measured by the owner: PP 125.54 tok/s (412 tokens),
TG 2.55 tok/s (210 tokens, 392.76 ms/token), graphs reused 208/210.
GPU: model 6,451.28 MiB + KV 2,176 MiB + compute 1,980 MiB. CPU: model
17,872.19 MiB + KV 6,528 MiB + RS split 115.34 MiB CPU / 34.29 MiB CUDA.

Goal of this plan: beat the 2.55 tok/s hybrid baseline with our executor by
adding GPU weight slots for recurrent layers, rotated through during each
token's forward pass, prefetched during the PP tail. End design target is a
limited-VRAM machine (owner: GTX 1080 8 GB), where the model does not fit
and the "everything resident" option does not exist.

## Measured facts (evidence only)

- Pageable H2D copy rate: ~3.3 GB/s (T-Q38LS-02, one slot, 364 MB package,
  111 ms submit->complete, GPU otherwise idle).
- Pinned staging: no wall-time improvement at four slots (T-Q38PERF-03:
  pinned PP 122.39 vs direct 124.95 tok/s). Raw pinned DMA rate: UNKNOWN.
  Correction for the record: a "50 GB/s pinned" figure was used in
  2026-08-24 discussion without measurement; it is withdrawn. T-Q38TG-00
  measures the real number.
- CPU recurrent layer cost estimate: (392.76 ms/token - GPU attention
  share) / 48 layers ≈ 7-8 ms/layer. Estimate only; T-Q38TG-00 measures.
- llama.cpp's `-ngl` is a static tail-count split; no per-layer regex
  offload and no runtime weight movement in the stock path. Its RS memory
  already splits per-layer across CPU/CUDA (see log evidence above).

## Break-even arithmetic (why the probe gates everything)

If all 48 recurrent layers stream from host every token, the sustained
copy engine must deliver one package (~235 MB Q4_K_M / ~364 MB Q6_K_XL)
per layer per token. Copy-rate ceiling:

| Sustained H2D rate | Q4_K_M per-layer copy | 48-layer TG ceiling |
| --- | --- | --- |
| 3.3 GB/s (measured pageable) | 71 ms | 0.28 tok/s (worse than CPU) |
| 20 GB/s | 11.8 ms | 1.8 tok/s |
| 29 GB/s (break-even) | 8.1 ms | 2.55 tok/s (matches baseline) |
| 50 GB/s (unverified) | 4.7 ms | 4.4 tok/s |

Measured outcome (T-Q38TG-00 clean re-run, 2026-08-24): pageable
2.90-2.96 GB/s, pinned 2.77 GB/s — below the first table row. NO-GO.

Mixed schedules (some recurrent layers permanently CPU, a slot window
rotating ahead of the token) relax the requirement: a copy of the next
window only has to finish within the CPU time of the current window
(~4 x 7.7 ms ≈ 31 ms for a 4-layer window -> ~30 GB/s for 940 MB, same
order). The design is only viable if T-Q38TG-00 measures a sustained
pinned rate well above ~15 GB/s; below that, the correct outcome is to
record the negative result and stop.

## Dependency order

```text
T-Q38TG-00 (probe) -> T-Q38TG-01 (hybrid executor, gated)
Independent: qwen38-fa-window-plan.md (PP compute), MTP (later lever).
```

### T-Q38TG-00: hybrid TG feasibility probe

Status: done (2026-08-24 — NO-GO verdict)
Depends on: none
Commit: pending (no commit authorized; probes only, no code changes)

Goal
: Measure the three numbers the design hangs on, with commands and traces
  recorded, and produce a go/no-go verdict against the break-even
  arithmetic. Instrumentation task; no executor changes, no performance
  claim.

Scope
: `tools/layer-streaming/main.cpp` runs only (existing flags), stock
  `llama-server` runs (owner build `C:\Development\AI\llamacppcuda`),
  read-only source inspection of the vendored device-split path.

Non-goals
: Any code change to the executor, GTX 1080 portability (separate probe),
  FA-window work, MTP.

Steps and exact commands (all GPU runs under `tools/with-gpu-lock.ps1`)
: 1. Raw H2D rate, pageable vs pinned, no queueing, GPU idle:
     `llama-layer-streaming --model <Q4_K_M> --layer-slots 1
      --boundary-layers 0,1,2,3,4,5 --pp-tokens 64 --ubatch 64 --no-mtp
      --copy-source direct --telemetry q38ls-tg00-copy-direct.jsonl` and
     the same with `--copy-source pinned`. One slot => no copy queueing,
     so `copy_complete.transfer_us` is the true submit->ready DMA time.
     Repeat with Q6_K_XL. Record bytes/us per run.
: 2. CPU recurrent layer cost by differential (stock server, greedy for
     determinism: add `--temp 0`): `-ngl 17` vs `-ngl 66` (fully resident)
     at identical prompt/context, Q4_K_M and Q6_K_XL. Attribute
     (tg_ms_17 - tg_ms_66) / 48 to CPU recurrent layers. This also
     re-baselines 2.55 tok/s on Q4_K_M (the owner's number is Q6_K_XL).
: 3. Mechanism inspection (read-only): how the vendored tree decides
     per-layer device at load (`llama_model_loader` / device split),
     whether tensor device binding can change after load, and how
     `llama_memory_recurrent` splits RS cells per device (evidence: it
     already does). Record which of mechanism A (extend our module with
     per-layer device dispatch) or B (extend the stock split with runtime
     weight rotation) is the cheaper path for T-Q38TG-01.

Acceptance criteria
: - [x] Sustained pinned and pageable H2D rates (MB/us) recorded for both
      models with command, trace path, and free-VRAM start.
: - [x] CPU recurrent per-layer cost (ms) recorded for both models via the
      `-ngl` differential, greedy, same prompt. Stock build hangs (driver
      incompatibility); plan's documented baseline and per-layer TG trace
      substituted. Owner informed; accepted.
: - [x] Written mechanism verdict (A vs B) with the inspected code paths
      named, in `docs/currentstate.md`. Verdict: Path A.
: - [x] Go/no-go verdict against the break-even table, recorded in
      `docs/currentstate.md`. **NO-GO** (corrected 2026-08-24): the
      original 2.9 GB/s figures were disk-bound (cold mmap page faults).
      True steady-state H2D DMA (`--no-mmap`, 4-copy protocol, copies 2-4):
      **13.7 GB/s** direct; pinned DMA identical but +11.7 ms staging
      memcpy. Break-evens: 45 GB/s for all-streaming parity, 50 GB/s to
      hide rotation copies behind CPU compute. Shortfall 3.3-3.7× — no
      window size fixes a fixed copy:compute ratio. Plan permanently gated.

Required validation
: None beyond the measurements themselves (this task produces evidence,
  not code). Every run records command, model identity, exit status, and
  trace path per AGENTS.md.

Escalate if
: The pinned path fails to allocate on this platform (it then fails
  initialization by design — record and fall back to the pageable
  verdict), or the `-ngl` differential is not attributable (e.g. sampling
  or server overhead dominates; then measure with `llama-cli` instead).

### T-Q38TG-01: hybrid TG executor (gated)

Status: not-started (gated: T-Q38TG-00 go verdict required)
Depends on: T-Q38TG-00
Commit: pending (no commit authorized)

Goal
: TG faster than the stock `-ngl 17` baseline (rematched on Q4_K_M by
  T-Q38TG-00) at the same model/context on this machine, via: 16 attention
  layers permanently GPU-resident, a slot window of recurrent layers that
  sweeps ahead of each token, and prefetch of the first window during the
  PP tail so token 1 starts with a warm window. Design target remains the
  8 GB configuration; this task validates on the local GPU.

Scope
: `src/llama-layer-streaming.{h,cpp}` (explicit layer->placement map
  replacing modulo slot assignment for hybrid mode; PP-tail prefetch;
  per-layer device dispatch per the T-Q38TG-00 mechanism verdict),
  `src/llama-context.cpp` (TG routing into hybrid mode),
  `tools/layer-streaming/main.cpp` (flags, telemetry), host checker.

Non-goals
: MTP (VRAM cost ~+240 MB; later lever), GTX 1080 portability, PP
  schedule changes beyond tail prefetch, FA-window plan work, quality
  claims, batch-major changes.

Design and failure contract (sketch; finalizes from T-Q38TG-00)
: 1. Placement map: each logical layer is {GPU-resident | rotating-slot |
     CPU}. Attention layers (and embed/output/norms) are resident; the
     recurrent split comes from the measured rates and the VRAM budget.
     Modulo slot identity does NOT work for this mapping (attention layers
     3,7,...,63 collide modulo any small slot count) — the map is
     explicit, and the SDD's "logical vs slot identity" invariant carries
     over unchanged.
: 2. PP-tail prefetch: in layer-major PP a slot is free the moment its
     layer's last chunk completes (existing consumer-done event). From
     layer 60 onward, freed slots load the first TG window instead of
     idling. At PP end, any still-missing window layer loads before token
     1; the stall is telemetered, never silent.
: 3. Steady-state sweep: while the token traverses layers k..k+W-1, the
     copy engine fills the window for k+W..k+2W-1 (pinned staging). A
     layer whose weights did not arrive in time falls back to CPU for that
     token and emits a `tg_fallback` telemetry event — correctness never
     depends on the copy winning the race.
: 4. Recurrent state: a layer's RS cell lives where the layer executes;
     the stock tree already allocates RS per device per layer. A layer
     whose placement is steady-state-GPU keeps its RS on GPU; the
     PP->TG boundary migrates RS cells once where the map requires. No
     per-token state ping-pong.
: 5. Loud failure contract unchanged: failed copy, failed event, stale
     generation, or VRAM shortfall aborts with structured telemetry; never
     run another layer's data.

Implementation steps
: 1. Reuse T-Q38TG-00's mechanism verdict; if A, extend the module; if B,
     extend the stock split. Do not build both.
: 2. Placement map + hybrid slot lifecycle + TG routing.
: 3. PP-tail prefetch + steady-state sweep + CPU fallback.
: 4. Telemetry: per-token device timeline (per-layer device, copy us,
     fallback count), `tg_window` fields in run_start.
: 5. Validation battery below.

Acceptance criteria
: - [ ] TG tok/s on Q4_K_M (1024 PP + >= 64 TG, greedy) beats the
      T-Q38TG-00-remeasured `-ngl 17` baseline, or the task escalates
      with the per-layer device timeline explaining the miss.
: - [ ] First TG token starts with a warm window (no full-window copy on
      the token-1 critical path; telemetered).
: - [ ] `tg_fallback` rate recorded; steady state reaches the designed
      window coverage (fallbacks only at startup or under measured copy
      misses).
: - [ ] Greedy-ID parity vs the existing all-GPU TG path at a context
      where both run (1024 tokens): first TG IDs identical. Correctness
      gate — a scheduling bug must fail parity, never pass silently.
: - [ ] VRAM at or under the planned budget (preflight reports the map's
      footprint before the timed region); 8 GB target budget table updated
      with measured sizes (including the MTP-off note: MTP adds ~240 MB
      when enabled and is out of scope here).
: - [ ] `tools/run-tests.ps1 -Suite all` passes unchanged in non-hybrid
      modes; default-OFF build compiles; PP behavior (layer-major, pw)
      unchanged except the tail prefetch.

Required validation
: Build target -> exit 0; all GPU runs under `tools/with-gpu-lock.ps1`
  with command/model/VRAM/exit/trace recorded; host validator on every
  trace; parity pair; baseline-matched TG comparison; full suite.

State update
: Replace in-progress state with the measured TG rates, device timeline,
  fallback statistics, VRAM footprint, and the 8 GB budget table in
  `docs/currentstate.md`.

Escalate before implementation if
: T-Q38TG-00 returns no-go; the mechanism verdict requires changes outside
  the isolated experimental path (e.g. rewriting the stock scheduler);
  RS migration at the PP->TG boundary is not expressible with existing
  memory-context APIs; or parity fails (stop and re-derive — a hybrid
  scheduling bug that changes outputs is not a tuning problem).

## 8 GB target budget (Q4_K_M, MTP off, 100K context — preflight must confirm)

| Item | VRAM |
| --- | --- |
| 16 attention layers (~235 MB ea) | ~3.8 GB |
| KV q8 @ 100K (Q4 KV lever if needed: T-Q38PW-03 gate now has a measured need) | 3.3 GB (1.65 GB Q4) |
| Static (embed/output/norms) + compute buffer | ~1.5-2.0 GB |
| Recurrent slot window (4 x 235 MB) | ~0.94 GB |
| Total | ~9.5 GB q8-KV / ~7.9 GB Q4-KV |

Honest caveat: with q8 KV this is over 8 GB; the Q4-KV lever
(T-Q38PW-03, currently gated) or a shorter context is required on the
target machine. Q6_K_XL does not fit this config at 100K at all — it is
the stress fixture for "doesn't fit", not the 8 GB deliverable.
