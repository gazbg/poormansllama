# llama.cpp Layer Residency Research Agent Guide

## Purpose

This repository is a private research workspace for an opt-in CUDA prefill
layer-residency mechanism in llama.cpp. The target mechanism stages one
complete immutable weight package per logical layer through a bounded VRAM
slot window while logical KV and recurrent state remain owned by the model
context.

The existing Qwen3.8-27B implementation is the measured dense reference, not
the permanent architecture boundary. Its design and completed plans remain
historical evidence in [docs/qwen38-layer-streaming-sdd.md](docs/qwen38-layer-streaming-sdd.md)
and `docs/plans/qwen38-*.md`.

The active generic design is
[docs/layer-residency-sdd.md](docs/layer-residency-sdd.md). The executable
backlog is
[docs/plans/generic-prefill-layer-residency-plan.md](docs/plans/generic-prefill-layer-residency-plan.md).
Do not silently redesign either active document while implementing a task.

`docs/Idea/` is historical research material. It is not the governing workflow.

## Goal Charter (owner decision, 2026-09-16)

The program's objective, in priority order:

1. **Maximize PP (prompt-processing/prefill) speed first.**
2. **Minimize VRAM usage second** — the slot-window budget, not model
   placement, is the design instrument.
3. **PP speed must hold when VRAM is crowded** — long-context KV (e.g.
   262k), or a lower-end GPU (16 GB-class) where the expert banks cannot fit
   even without KV. The mechanism must never depend on the model fitting in
   VRAM; that case is not the target condition.

Tasks are prioritized and judged against this charter; results that only
improve the uncrowded case are secondary.

## Scope and Boundaries

- The current implementation supports dense Qwen3.8 through Qwen-specific
	package lists and graph construction. Generic support is not complete until
	another architecture passes the active plan's acceptance gates.
- Active development currently targets the Qwen MoE vehicle
	(Qwen3.6-35B-A3B routed-expert banks), because it is the only on-disk model
	whose expert banks do not fit in VRAM, making residency actually measurable.
	This focus is a measurement choice, not an architecture boundary: it does
	not exclude other models. Testing any other architecture is legitimate
	work whenever the owner asks for it (see the plan's second-architecture
	draft, T-GPLR-27), and owner-requested model tests are never treated as
	scope creep.
- For dense models, a pageable package contains the complete immutable decoder
	layer. For MoE prefill, the first policy pages the complete routed-expert
	bank for one layer; router, attention, shared-expert, embedding, output, KV,
	and recurrent state placement is explicit in the architecture adapter.
- The primary workload is layer-major prefill: load layer L once, execute every
	prompt chunk for L in causal order, then advance. Token-generation expert
	prediction and weight rotation are separate experiments and are not implied
	by generic prefill support.
- The layer-residency path is opt-in and CUDA-only. An experimental run fails
	loudly when unsupported; disabling the option preserves normal llama.cpp
	placement and CPU fallback behavior.
- Default llama.cpp behavior, including its CPU path, must remain unchanged when the experimental option is disabled.
- Quantization bytes and tensor semantics are preserved exactly during staging.
	Residency policy must not alter routing, expert IDs, model arithmetic, or
	checkpoint metadata.
- Vision, MTP, YaRN, quality tuning, quantization comparison, multi-GPU, and
	non-CUDA backends remain out of scope unless an approved task adds them.
- Do not claim quality, equivalence, or speedup unless the task contract explicitly defines and validates that claim.

## Repository Layout

- Project-owned design and plans live under `docs/`.
- Canonical environment facts live in `docs/AGENT_ENVIRONMENT.md`.
- Canonical current-state facts live in `docs/currentstate.md`.
- The completed Qwen documents are reference evidence. New generic work updates
	the active generic SDD and plan rather than rewriting completed Qwen history.
- Generated build, trace, model, and telemetry output must stay out of version control.
- `third_party/llama.cpp/` is a vendored private dependency. Read its nested `AGENTS.md` before changing anything in that tree.

Changes to the vendored dependency are permitted only for the isolated experimental path. Do not create an upstream pull request, push, or write upstream-facing material. Do not modify normal execution paths when an opt-in experiment gate can isolate the work.

## Environment and Dependencies

Before building, testing, running a model, installing a package, or changing a dependency, read `docs/AGENT_ENVIRONMENT.md` if it exists.

Bootstrap exception: until that file exists, only the environment-inventory task may use non-mutating discovery and CMake configure commands to create it. Record observed facts only; do not infer unavailable tools or capabilities.

- Prefer existing C++, CMake, CUDA, and llama.cpp facilities over new dependencies.
- New external dependencies require owner approval before installation, vendoring, include-path changes, or package metadata changes.
- On Windows, use forward slashes in CMake `-D` tool-path values to avoid generated CMake escape errors.
- Treat model paths, driver versions, build commands, and hardware capacity as environment facts, not permanent source-code constants.

## Planner Workflow

The Planner owns end-to-end design, task order, and task contracts.

Before marking a task `ready`, the Planner must:

1. Read the SDD, current state, environment facts, controlling code, nearby tests, and relevant nested dependency instructions.
2. State the observable outcome, invariants, scope, non-goals, known evidence, and unknowns.
3. Resolve ordinary technical uncertainty through focused inspection or a reversible probe.
4. Ask the owner only when the answer changes product intent, hardware target, dependency policy, data policy, public compatibility, or an irreversible tradeoff.
5. Define the narrowest vertical task that can be implemented and validated independently.

Each task in `docs/plans/` must include:

- status, dependencies, and commit field;
- goal, scope, non-goals, and design/failure contract;
- concrete implementation steps;
- falsifiable acceptance criteria;
- exact focused validation commands and expected results;
- required state/document updates; and
- an escalation condition.

For experimental tasks, distinguish a feasibility result, an instrumentation result, and a performance result. Telemetry alone does not prove state or lifecycle correctness.

## Implementer Workflow

Work on one `ready` task at a time.

Before the first edit:

1. Read the complete task contract, SDD section, current state, environment facts, controlling code, and nearest test.
2. Inspect the worktree and preserve unrelated changes.
3. State one falsifiable local hypothesis and the cheapest check that could disprove it.
4. Mark the task `in-progress` only after its boundaries and validation are understood.

After the first substantive edit, run the task's cheapest focused validation before expanding the change.

For a failure:

1. Capture the command, input, error, and relevant telemetry.
2. Classify it as task-owned, environment, test-expectation, or unrelated.
3. Trace the owning code path and make the smallest repair.
4. Rerun the same focused validation first.
5. If the failure resists steps 2-4, treat it as a problem and follow
"Problems and Online Help" below before improvising a repair.

Do not broaden a task across an SDD or plan boundary without returning it to the Planner.

## Problems and Online Help

A problem is a failure the loop above cannot resolve -- an error that will
not classify, an unfamiliar API or toolchain failure, platform behavior that
contradicts `docs/AGENT_ENVIRONMENT.md`. For such a problem, consult
authoritative online sources before improvising:

- Prefer primary sources over forum answers: vendor documentation
	(Microsoft Win32, CUDA, CMake), upstream llama.cpp code, issues, and
	PRs, and official library references.
- Verify every online claim against the local tree before acting on it;
	the vendored sources, recorded environment facts, and the active task
	contract outrank any web page.
- Never post private material (model paths, telemetry, host names,
	internal docs) to public sites; search with generic terms instead.
- Time-box the lookup. Record what was consulted and what it changed in
	the task notes or `docs/currentstate.md`.
- An online answer that requires a new dependency goes through the owner
	approval gate like any other dependency change.
- A problem that survives a documented online pass escalates to the owner
	under the task's escalation condition; it is never resolved by guessing.

## Layer-Residency Invariants

- Logical layer identity and device-slot identity are separate.
- A slot contains one complete immutable architecture-declared package for one generation; package members become ready and retire atomically.
- A layer graph may consume a slot only after the matching copy event completes.
- A slot may be reused only after all CUDA consumers of its current generation complete.
- KV cache, recurrent state, and other mutable session state remain owned by logical layer, never by a weight slot.
- Layer-major prefill processes chunks for each logical layer in ascending token-position order.
- An MoE full-expert package contains every routed expert for that layer; cache residency never changes the router's exact output.
- A missing package, stale slot generation, failed CUDA event, or insufficient VRAM must fail loudly with structured telemetry. It must not silently run another layer's data.
- PP-to-TG retains session state and live weight-slot assignments unless a task explicitly studies reset behavior.

### Vendored-engine gotchas (measured, not documented upstream)

- `llm_graph_context::build_inp_embd` always allocates BOTH `inp_tokens` and `inp_embd` (F32, ~4 MiB at ubatch 512), then `ggml_build_forward_select` prunes the embeddings branch in token mode — leaving a tensor that is INPUT-flagged, allocated, and consumed by NO graph. Any cache built by walking a graph ctx for INPUT tensors inherits it; attribute inputs by TRANSITIVE src walk (an input consumed through a view sits one hop past the node's direct src), never direct sources alone.
- `--pp-schedule layer-major` refuses `--tg-tokens > 3` when handoff auto does not activate (exit 2). Long-id identity arms at the 50k shape are impossible without forcing handoff (which changes the execution mode); run them at 4k instead.
- `prepare_chunk` early-returns when `pw_probe` is off, so `--no-pw-probe` mode does NO per-launch patch: no act repointing, no snapshot restore, and **no `patch_cache` recon dump to inspect**. Any harness or checker that requires that dump will verdict a working non-pw run FAIL (measured 2026-09-19, T-GPLR-49). Treat it as a pw-only artifact.
- The act-slab region claims (`slab_claims`) live for the module's lifetime and are only consistent with the offset cursors while nothing clears one without the other; derive slab offsets from the immutable chunk descriptor, never from a cursor, or a reordered/repeated pass silently re-binds a slice over a live one.
- `--pp-prebuild-per-pass` defaults ON for EVERY KV type whenever `--pp-repeat > 1` (f16 included since T-GPLR-50; T-GPLR-48 first made it KV-conditional with f16 OFF). A recorded command that omits the flag therefore runs differently today at f16 (rebuild OFF) than when it was recorded — read the resolved value from the trace's `run_start` (`pp_prebuild_per_pass` + `..._source`), never from command history.
- Route parity is a pass-2 integrity signal: in a clean `--pp-repeat 2` trace every sampled `route` tuple appears EXACTLY TWICE. Since T-GPLR-39 route telemetry DEFAULTS OFF — a defaulted trace has NO route events and `check-route-parity.py` exits 2 ("nothing to evaluate", NOT a pass); the gate requires an explicit `--route-telemetry 1` run. Since T-GPLR-47 (fresh-context swap-in-slot rebuild, nested 4403c1097) it is CLEAN on every supported mode; a DIVERGED result now means a genuinely new pass-2 divergence (or a pre-T-GPLR-47 binary — check the llama.dll hash), not a known defect — `tools/check-route-parity.py`, host-only, no decode. Its exit 2 (no route events) means the run died before routing, which is NOT a pass. T-GPLR-48 addendum: `--pp-prebuild-per-pass` defaults KV-conditional (ON iff `--pp-repeat > 1` AND KV != f16), and `host_ledger`/`pass_end` markers exist ONLY when the rebuild runs — a rebuild-OFF trace has no per-pass ledger.
- Quantized KV with the rebuild OFF crashes in pass 2 with exit `-1073740791` (0xC0000409) at the **50k** shape, not just T-GPLR-43's 143k one — never pin `--pp-prebuild-per-pass 0` alongside `--kv-cache q8/q4` and read the result as a measurement.
- Editing `src/llama-layer-streaming.h` changes the `llama_layer_streaming` TYPE LAYOUT, so every test target linked against it must be REBUILT: the host suite's CTest step does not build, and with a stale `test-layer-streaming-layout.exe` the test prints every assertion `ok` and then dies in teardown with `Exit code 0xc0000409` - which reads exactly like a fresh defect in whatever you just changed (measured 2026-09-23: 3.65 s crash vs 0.08 s pass after `cmake --build build/llama-cuda --config Release`).
- The GPU mutex is cooperative: a non-conforming project (deepseek_v4.exe, observed 2026-09-21) can hold ~16.5 GiB VRAM DURING a mutex-held run and poison it - uniformly-inflated per-layer compute, teardown `cudaFree` OOM crashes (exit 0xc0000409), normal transfer ratios. The recorded "+67.4 s rebuild-OFF pass-2 excess at 200k" was THIS, not a code regression (re-attributed in T-GPLR-50). Before trusting any timing arm: check `nvidia-smi --query-compute-apps` before AND after, and rerun suspect arms under a free GPU before attributing anything to code.

## Validation and Evidence

Run the narrowest meaningful validation first:

1. Compile or static check the touched code.
2. Run a focused unit or lifecycle test.
3. Run the focused CTest integration target.
4. Run the smallest real-model trace required by the task.
5. Run a larger real-model trace only when the task contract requires it.

For CUDA/runtime work, use the project CMake build and CTest targets defined by the active plan. Do not substitute unrelated `make`, DeepSeek V4, or upstream CI commands. Run the Qwen reference regression before claiming that genericization preserved existing behavior.

- An A/B control across a vendored-tree change is a DIFFERENT BINARY, and the streaming code lives in `llama.dll`: side-copy the exe AND all DLLs (e.g. `build/llama-cuda/bin/unscoped-bin/`) BEFORE rebuilding. Rebuilding destroys the control, and copying the exe alone leaves it silently loading the new dll.

### Semantic-QA answer gate (T-GPLR-07)

Runner changes that touch the layer-streaming experimental
path (or anything that can alter generated output) are validated with the
semantic-QA ladder in addition to the checks above:

- Smoke tier = 50/512-token runs on the 512-token anchor prompt (the
	default `qa-smoke` gate runs 512 only; 50 is opt-in via `-SmokeLengths
	50`; `tools/semantic-qa/gen-needle-prompt.py`, checker
	`tools/semantic-qa/check-answer.py`). All rungs must report check-telemetry
	OK and aggregate checker PASS, including the negative control failing as
	designed. Ladder series: 50/512/4k/24k/50k/100k/200k; 4k+ lengths are
	owner-invoked manual runs, never part of the suite.
- The strict acceptance gate is programmatic grading only: normalized
	substring/regex verdicts plus the must-fail negative control. Never bit,
	byte, hash, or first-token identity, and never an LLM judge.
- Ladder lengths (24k, 50k, 100k, 200k) are never run as a batch or on
	automotion; the owner invokes one length at a time as an explicit
	GPU-mutex acceptance run.
- The ladder promises answer stability, never token-ID stability. Do not
	repair a rung by demanding exact ID agreement.
- Full commands, flags, and expected results are in the T-GPLR-07 task
	contract in `docs/plans/generic-prefill-layer-residency-plan.md`; record
	results in `docs/currentstate.md`.

## Gate Policy: All Gates Are Soft

Every gate in this repository -- semantic-QA ladder, trace checker, CTest
families, capacity preflight and refusal path, performance and hit-rate
measurements -- is advisory evidence, not release authorization. A gate PASS
proves only what that gate's contract defines (for example, answer stability
or trace integrity); it never certifies that the code is fit for use.

- Final acceptance is always the owner's decision, made from the summary
	required below, never from gate output alone.
- Report gate results honestly, including their limits: noise floors,
	stream biases, unmeasured expectations, and withdrawn confounded
	comparisons. Never present a gate PASS as "ready" or "approved".
- Soft gates concern acceptance, not diligence: the required validation of a
	task must still run and pass before its evidence is reported. A hard
	failure (loud refusal, checker rejection, violated invariant) still blocks
	reporting a task complete.

## Owner Summary Requirement

Because all gates are soft, every completed task and every significant
experiment must end with a summary the owner can use to decide whether the
code is acceptable for use. The summary must be self-contained and structured:

1. Original task / goal / idea explored -- what was asked, the hypothesis or
	idea behind it, and why it mattered.
2. How it works in simple terms -- the mechanism in plain language a
	non-specialist can follow.
3. Glossary -- every term and abbreviation used in the summary is defined or
	explained at first use (e.g. PP, TG, af, slots, hit rate). Do not assume
	the glossary from prior turns; restate it in the summary itself.
4. Pros and cons -- benefits, costs, risks, and limitations of the solution,
	with every claim labeled measured / expected / unproven and pointed at its
	evidence (telemetry path or docs section).
5. Recommendation -- a clear bottom line: adopt, keep experimenting, or
	discard, plus what the owner should validate or decide next.

A task is not reported complete while any of these parts is missing.

## Shared GPU Coordination

This machine shares one GPU across multiple projects. Before any command that initializes CUDA or uses the GPU, acquire the Windows named mutex `Global\AI-GPU-EXCLUSIVE-v1`.

- Use `tools/with-gpu-lock.ps1` to run GPU tests, model traces, benchmarks, profilers, and GPU-backed CTest targets.
- The wrapper probes nvidia-smi after acquiring the mutex and REFUSES with exit 125 while any compute process is visible (teardown grace 30 s; `-AllowBusy` overrides). Verdicts carry `gpu_pre_run`/`gpu_post_run`; `probe-unavailable` proceeds UNVERIFIED.
- The wrapper waits for the mutex and exits on timeout. It must never start a GPU workload while another holder owns the mutex.
- CPU-only configuration, compilation, source inspection, and static checks do not require the mutex.
- Do not use a sentinel file or delete another process's lock. Windows releases the named mutex if its owning process exits unexpectedly.
- Other projects sharing this GPU must use the same mutex name and equivalent acquire-wait-release behavior.
- Detached arm launches: the LAUNCHER must dot-source `tools/arm-verdict.ps1` and run `Clear-ArmVerdict` for the tag BEFORE `Start-Process`. The arm clears its own files only after the process starts, so an immediate `Wait-ArmStart`/`Wait-ArmVerdict` reads the PREVIOUS run's marker and verdict (measured 2026-09-18: stale pid and `started_utc`).
- NEVER launch a detached arm through `setsid` (e.g. `(setsid nohup bash script.sh &)`). Under it PowerShell does NOT wait for a native child: `& exe > log 2>&1` returns in 0 s with `$LASTEXITCODE` UNSET and the child's redirected output never reaching the log, so the arm verdicts a run that is still in flight and the engine finishes orphaned. Use plain `nohup` or `Start-Process` (measured 2026-09-19 with one 6 s child in three launcher shapes: direct and plain `nohup` both `exit=0 dur=5.1s logbytes=1056`; `setsid` `exit=<null> dur=0.0s logbytes=0`).

Expensive model runs require a recorded command, model artifact identity, hardware, relevant flags, exit status, and telemetry path. Do not run them merely for confidence.

The PP/TG experiment must emit JSON Lines plus a human-readable summary. Timing evidence must keep transfer time, availability-wait time, compute time, graph/setup time, and wall time distinct. The best-effort cold mmap source state must be reported as uncontrolled Windows file-cache behavior, not as a guaranteed cold-cache condition.

## State, Commits, and Completion

Update `docs/currentstate.md` after each completed task with observed behavior, validation evidence, known gaps, blockers, and residual risk. Remove superseded claims.

A task is done only when its acceptance criteria and required validation are demonstrated, its state/docs are updated, and no task-owned defect or unchecked criterion remains.

Do not stage, commit, push, create branches, or create pull requests unless the owner explicitly asks. If authorized to commit, stage only task-owned files, verify the staged diff and `git diff --check`, and record the resulting hash in the task plan.

When the owner authorizes a mid-task commit while validation is still pending (e.g. work parked behind a GPU-mutex wait), checkpoint the mechanism and tooling BEFORE any verdict exists (nested and outer commits separately), and leave the task's `Commit:` field unfilled until the verdict commit lands.

After a task finishes, continue with the next eligible `ready` task only when the owner has authorized continuing implementation. Otherwise report the completed evidence and any blockers.


## Tools
- Prefer using lean-ctx MCP instead of default tools , fall back to default if it does not work for a particular call, read the lean-ctx skill to know how
- Prefer using context7 MCP for library information on how it works or what it offers for a give case. to be used if the library has no source locally , example CUDA . This can also search for functionality instead of implementign all by yourself
- For web search ,research and other web related context prefer using mcp wigolo, also when you hit similar  problem more than 1 time , do a web research on it to help yourself out 