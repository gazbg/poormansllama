# Handoff: 20K-prompt PP experiment (layer-streaming runner)

Date: 2026-08-19. Owner of record: the agent session that last touched this work.
Read this top-to-bottom before editing anything.

## 1. Background (all done and verified)

The Qwen3.8-27B CUDA layer-streaming plan (`docs/plans/qwen38-layer-streaming-plan.md`)
is **complete**: all 7 tasks (T-Q38LS-01/02/03/04/04B/05/06) are marked `done`,
all 28 acceptance-criteria boxes checked, full 12-target CTest suite green
(run under `tools/with-gpu-lock.ps1`, mutex `Global\AI-GPU-EXCLUSIVE-v1`).
State lives in `docs/currentstate.md`. No commits were made (not authorized).

Since then, in this session, two user-requested improvements landed:

1. **`tools/run-tests.ps1`** — manual test runner (4+1 suites: `all`, `quick`,
   `boundary`, `telemetry` (host-only, no GPU), `pp20k` (direct experiment run)).
   Bugs found and fixed along the way:
   - `|` in ctest regexes must be escaped `^|` (the lock wrapper runs via `cmd.exe`).
   - This CMake's ctest has no `--output-always`; use `--verbose`.
   - Run-mode self-check (added below) is skipped for `--boundary-layers` runs
     (deterministic slot-policy invariant doesn't apply to a reduced layer subset).
   - `quick` excludes the telemetry CTest targets (their fixture would pull in
     the 350s `layer-streaming-transition-run`).
2. **Runner telemetry output** (`third_party/llama.cpp/tools/layer-streaming/main.cpp`):
   - `--check-telemetry PATH` now prints a per-phase summary after "OK":
     tokens, tok/s, wall, total transfer/wait/compute, layers, evictions,
     graph_builds, peak VRAM.
   - Every model run re-validates its own `--telemetry` trace at exit (same
     summary printed inline; fails the run on invariant violation).

## 2. Current task (IN PROGRESS)

User request: *"I would like to see a test where we do PP for 20K with ubatch 512
and TG with 3 tokens only, I want to see how PP handles larger prompts."*
→ PP 20480 tokens, ubatch 512 (= 40 whole ubatches), TG 3 greedy tokens.

Feasibility (verified):
- `n_ctx = n_pad = ((20480+3+255)/256)*256 = 20736`; divides `kv_size`; mask width
  (`n_kv`) is **constant 20736** the whole run (cells vector is preallocated to
  `kv_size` at cache construction — `llama-kv-cache.cpp` line ~142), so the
  attention mask shape never changes. Confirmed by debug print: all variants
  report `live n_kv=20736`.
- VRAM fit: ~9-10 GiB of 24 GiB (static 2.4 GB + 4 slots 1.5 GB + KV ~1 GB +
  buffers). OK when the shared GPU isn't heavily occupied; preflight aborts loudly.
- The default runner sets `n_outputs_max = pp_tokens` and logits for **every** PP
  token → a 20480×151936×4B logits buffer (~12 GB) would not fit. Hence a flag was
  added: `--pp-logits-last` (logits row only for the final PP token,
  `n_outputs_max = 1`). **This flag is being replaced — see §5.**

## 3. Bug found while running the experiment (root cause, with evidence)

First 20K run failed at PP ubatch 2 with:
`graph set (mode 0, n_tokens 512, rs_z -1) does not match current ubatch shapes`.

Debug instrumentation (env `LLAMA_LAYER_STREAMING_DEBUG=1`, prints currently in
place) showed:
- prebuild variants: `mode=0 n=512 rs_z=0 n_kv=20736` (A), `rs_z=-1` (B), `mode=1 n=1` (C) — all fine.
- ubatch 1 (set A): inputs embd/mem_hybrid/pos/out_ids all `ok=1`.
- ubatch 2 (set B): `embd ok=1, mem_hybrid ok=1, pos ok=1, out_ids ok=0` ← culprit.

Root cause: `llm_graph_input_out_ids` bakes the ubatch's **n_outputs** (count of
logits rows) as its tensor width; `can_reuse` checks `n_outputs == params.n_outputs`.
The prebuilt set B had been built with `logits[i]=true` hardcoded for all 512
tokens (→ n_outputs=512), but with `--pp-logits-last` the real intermediate ubatch
has **0** output rows. The 512-token tests passed only because every ubatch there
had all logits on (n_outputs constant per set). `n_kv`/mask were never the issue.

## 4. Fix applied so far (code compiles; NOT yet validated end-to-end)

The graph-set key was extended from `(mode, n_tokens, rs_z)` to
`(mode, n_tokens, rs_z, n_outputs)`:

- `src/llama-layer-streaming.h`: `graph_set.n_outputs`; `begin_set/find_set/begin_build`
  take an extra `int32_t n_outputs`.
- `src/llama-layer-streaming.cpp`: key comparison/creation updated; executor looks up
  `ls->find_set(phase, ubatch.n_tokens, rs_z, (int32_t) gparams.n_outputs)`;
  on-demand fallback build also passes it; error messages include n_outputs.
- `src/models/qwen35.cpp`: `begin_build(n_tokens, rs_z, (int32_t) n_outputs)`.
- `src/llama-context.cpp` prebuild:
  - `build_variant` passes the computed `n_outputs_new` into `begin_set`.
  - Variant B now **copies the real logits pattern** from `batch_inp`
    (previously hardcoded `true`) instead of assuming all-logits.
  - A running `applied` counter tracks prebuild cell positions (consecutive
    positions are required by KV `find_slot`; position *values* don't affect shapes).
  - A **B' variant** (final PP ubatch, different n_outputs) was added — **it crashes**
    (ggml buffer-overflow assert at `ggml-backend.cpp:2060` during B' graph build,
    first set with n_outputs=1). B' is being removed (see §5).

`--pp-logits-last` is currently in `tools/layer-streaming/main.cpp` (flag parse,
usage line, `n_outputs_max = pp_logits_last ? 1 : pp_tokens`, per-token logits
assignment) and in `tools/run-tests.ps1` (`pp20k` suite command).

## 5. Decision made (pivot) — NOT yet implemented

Drop the per-ubatch logits pattern entirely. Replace `--pp-logits-last` with
**`--pp-no-logits`**: every PP ubatch has `n_outputs = 0` (constant), TG has 1.
Rationale: PP logits are never consumed by this experiment (TG starts its own
logits chain), a constant `n_outputs=0` needs only sets A(0), B(0), C(1) — all
already proven to build — and removes the crashing B' path. The set-key extension
(n_outputs) stays: it's the correct general fix and is what makes this safe.

## 6. Remaining steps (in order)

1. `src/llama-context.cpp`: **remove the B' block** from `layer_streaming_prebuild`
   (keep the `applied` counter; C already uses `applied` for its position).
   Update the stale comment above variant B (still mentions `--pp-logits-last`).
2. `tools/layer-streaming/main.cpp`: replace `--pp-logits-last` with
   `--pp-no-logits`: `batch.logits[i] = false` for all PP tokens,
   `cparams.n_outputs_max = pp_no_logits ? (tg_tokens > 0 ? 1 : 0) : pp_tokens`,
   update usage text.
3. `tools/run-tests.ps1`: `pp20k` suite command → `--pp-no-logits`.
4. Rebuild: `cmake --build build/llama-cuda --config Release --target llama-layer-streaming --parallel 16`
5. Sanity (small, ~15s, under GPU lock): 64-token PP + 3 TG with `--pp-no-logits`
   (expect: OK, both phases `graph_builds 0`, self-check summary prints).
6. Run the 20K experiment:
   `.\tools\run-tests.ps1 -Suite pp20k`  (≈10-20 min, ~25 tok/s PP expected under
   contention → ~15 min PP + load)
   Expected: 40 PP ubatches, phases pp(20480)/tg(3), `graph_builds 0` both phases,
   per-layer table, peak VRAM ~9-10 GiB, 3 greedy TG ids.
7. Regressions (must stay green — the n_outputs key touched shared paths):
   `.\tools\run-tests.ps1 -Suite all` (12 targets; regenerates the transition
   trace too) and OFF build:
   `cmake --build build/llama-cuda-off --config Release --target llama-cli --parallel 16`
8. **Remove debug instrumentation** (env-gated `LLAMA_LAYER_STREAMING_DEBUG`):
   - `src/llama-layer-streaming.cpp`: per-input type/ok dump + shape-guard header line.
   - `src/llama-context.cpp`: prebuild-variant print.
   (The older `ubatch N build_generation ... graphs` + per-graph dump in the
   executor is pre-existing T-04 instrumentation — may stay or go with owner's call.)
9. Docs: `docs/currentstate.md` — add a short "20K PP experiment" section
   (command, results, raw numbers, shared-GPU caveat); note `--pp-no-logits` in
   the runner notes. Plan file: note the set-key extension under T-Q38LS-04B
   (implementation note) — do **not** reopen task status unless owner asks.
10. Report results to the user (PP tok/s, transfer/wait/compute per phase,
    peak VRAM, the 3 TG ids, per-layer table highlights).

## 7. Commands / environment cheat-sheet

- Workspace: `C:\Development\AI\qwen38GZI` (Git Bash; **MSYS path-mangles exe args** —
  from bash, use backslash Windows paths in the GPU-lock command string; from
  PowerShell (user) this doesn't apply).
- GPU lock (mandatory for any CUDA command):
  `powershell -NoProfile -ExecutionPolicy Bypass -File tools\with-gpu-lock.ps1 -TimeoutSeconds N -Command "<cmdstring>"`
- Model: `C:\Models\ALL\other\qwen-3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf` (also
  `docs/AGENT_ENVIRONMENT.md`). Reconfigure needs `Q38LS_MODEL_PATH` exported
  or CTest GPU targets get an empty model path.
- Builds: `build/llama-cuda` (LLAMA_LAYER_STREAMING=ON, GGML_CUDA=ON),
  `build/llama-cuda-off` (default OFF).
- Traces: `build/llama-cuda/q38ls-*.jsonl`; host-side re-check any trace with
  `build\llama-cuda\bin\Release\llama-layer-streaming.exe --check-telemetry <trace>`.
- The 20K trace (current, **failed/incomplete**): `build/llama-cuda/q38ls-pp20k.jsonl`.
- ctest fixture gotcha: `-R layer-streaming-telemetry` pulls in the 350s
  `layer-streaming-transition-run` via `FIXTURES_REQUIRED`; avoid ad-hoc ctest
  runs for that pattern, use the script.
- `q38ls-pp-tg.jsonl` was regenerated by the last full-suite run (it had been
  truncated by a killed mid-run launch and deleted on 2026-08-19).

## 8. Risks / open questions

- If `--pp-no-logits` (n_outputs=0) exposes an issue in the output-graph path
  (e.g., `n_outputs_max=0` edge cases in `llama-context.cpp` line ~259, or
  `sched_reserve` with 0 outputs), the failure will be a loud assert/log at
  context creation or first ubatch — fix in the normal (non-streaming) path
  only if unavoidable; prefer clamping to ≥1 where the API requires it.
- Shared GPU: tok/s numbers carry contention noise; record observed free VRAM
  from `run_start` and state the caveat with any numbers.
- Do **not** commit (no authorization). Do not open upstream PRs (vendored dep,
  isolated experimental gate only).
