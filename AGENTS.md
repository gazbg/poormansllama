# Qwen3.8 Layer Streaming Agent Guide

## Purpose

This repository is a private research workspace for a CUDA-only Qwen3.8-27B dense decoder-layer streaming experiment. The current objective is to measure prompt-processing (PP) and token-generation (TG) behavior with four pageable decoder-layer weight slots.

The normative design is [docs/qwen38-layer-streaming-sdd.md](docs/qwen38-layer-streaming-sdd.md). The executable backlog is [docs/plans/qwen38-layer-streaming-plan.md](docs/plans/qwen38-layer-streaming-plan.md). Do not silently redesign either document while implementing a task.

`docs/Idea/` is historical research material. It is not the governing workflow.

## Scope and Boundaries

- The current model is dense Qwen3.8. MoE routing, expert prediction, vision, MTP, YaRN, quality tuning, and quantization comparison are out of scope unless a new approved plan adds them.
- The layer-streaming path is opt-in and CUDA-only. It does not require a CPU fallback.
- Default llama.cpp behavior, including its CPU path, must remain unchanged when the experimental option is disabled.
- Decoder-layer weights may be pageable. Token embeddings, final norm, LM head, KV cache, and Gated DeltaNet recurrent state remain outside the four weight slots as defined by the SDD.
- Do not claim quality, equivalence, or speedup unless the task contract explicitly defines and validates that claim.

## Repository Layout

- Project-owned design and plans live under `docs/`.
- Canonical environment facts live in `docs/AGENT_ENVIRONMENT.md`.
- Canonical current-state facts live in `docs/currentstate.md`.
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

Do not broaden a task across an SDD or plan boundary without returning it to the Planner.

## Layer-Streaming Invariants

- Logical layer identity and device-slot identity are separate.
- A slot contains one complete immutable decoder-layer package for one generation.
- A layer graph may consume a slot only after the matching copy event completes.
- A slot may be reused only after all CUDA consumers of its current generation complete.
- KV cache and Gated DeltaNet recurrent state remain owned by logical layer, never by the four weight slots.
- A missing package, stale slot generation, failed CUDA event, or insufficient VRAM must fail loudly with structured telemetry. It must not silently run another layer's data.
- PP-to-TG retains session state and live weight-slot assignments unless a task explicitly studies reset behavior.

## Validation and Evidence

Run the narrowest meaningful validation first:

1. Compile or static check the touched code.
2. Run a focused unit or lifecycle test.
3. Run the focused CTest integration target.
4. Run the smallest real-model trace required by the task.
5. Run a larger real-model trace only when the task contract requires it.

For CUDA/runtime work, use the project CMake build and CTest targets defined by the plan. Do not substitute unrelated `make`, DeepSeek V4, or upstream CI commands.

## Shared GPU Coordination

This machine shares one GPU across multiple projects. Before any command that initializes CUDA or uses the GPU, acquire the Windows named mutex `Global\AI-GPU-EXCLUSIVE-v1`.

- Use `tools/with-gpu-lock.ps1` to run GPU tests, model traces, benchmarks, profilers, and GPU-backed CTest targets.
- The wrapper waits for the mutex and exits on timeout. It must never start a GPU workload while another holder owns the mutex.
- CPU-only configuration, compilation, source inspection, and static checks do not require the mutex.
- Do not use a sentinel file or delete another process's lock. Windows releases the named mutex if its owning process exits unexpectedly.
- Other projects sharing this GPU must use the same mutex name and equivalent acquire-wait-release behavior.

Expensive model runs require a recorded command, model artifact identity, hardware, relevant flags, exit status, and telemetry path. Do not run them merely for confidence.

The PP/TG experiment must emit JSON Lines plus a human-readable summary. Timing evidence must keep transfer time, availability-wait time, compute time, graph/setup time, and wall time distinct. The best-effort cold mmap source state must be reported as uncontrolled Windows file-cache behavior, not as a guaranteed cold-cache condition.

## State, Commits, and Completion

Update `docs/currentstate.md` after each completed task with observed behavior, validation evidence, known gaps, blockers, and residual risk. Remove superseded claims.

A task is done only when its acceptance criteria and required validation are demonstrated, its state/docs are updated, and no task-owned defect or unchecked criterion remains.

Do not stage, commit, push, create branches, or create pull requests unless the owner explicitly asks. If authorized to commit, stage only task-owned files, verify the staged diff and `git diff --check`, and record the resulting hash in the task plan.

After a task finishes, continue with the next eligible `ready` task only when the owner has authorized continuing implementation. Otherwise report the completed evidence and any blockers.