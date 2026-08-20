# Agent Workflow

This repository uses two roles:

- **Planner**: higher-reasoning agent. Owns end-to-end understanding, decisions,
  dependency order, and precise vertical task definitions.
- **Implementer**: faster execution agent. Owns exactly one ready task at a
  time and drives it through implementation, verification, documentation, and
  an atomic commit.

The roles may be separate models or separate runs of one model. They must not
blur their responsibilities: the planner creates an executable contract; the
implementer does not silently redesign it.

Before any build, test, model run, package installation, or dependency decision,
read [Agent Environment and Dependency Guide](docs/AGENT_ENVIRONMENT.md).

## Repository Rules

- Preserve the dependency-free CPU path. CUDA, platform, and experimental paths
  must retain a correct CPU fallback when the task's contract requires one.
- Keep each change focused. Do not refactor unrelated code while implementing a
  task.
- `c/v4/` is the canonical DeepSeek V4 implementation. `c/deepseek_v4.c` and
  `c/Makefile.deepseek-v4.units` are generated compatibility/build outputs;
  never edit them by hand. After a V4 module edit, run
  `make -C c v4-amalgamate` followed by `make -C c v4-amalgamate-check`.
- Treat existing code, tests, SDDs, and measured logs as evidence with stated
  scope. Do not claim a feature is validated just because symbols exist.
- `main` is stable. Integration work targets `dev`; do not commit directly to
  `main` unless the task owner explicitly instructs otherwise.
- Never revert, stage, or commit unrelated user or agent changes. In a dirty
  tree, stage only files belonging to the completed task.
- Never use performance as a reason to abandon a planned correctness or
  architecture task. Record measurements as evidence, then finish the defined
  task or escalate a product decision.

## Environment and Dependency Gate

- Use the commands and resource limits in
  [Agent Environment and Dependency Guide](docs/AGENT_ENVIRONMENT.md). Do not
  infer installed tools or package availability from a different shell/session.
- Before using a library, first check the standard library, repository code,
  installed CUDA toolkit, and declared optional dependencies.
- A new external library is never pre-approved. If a task identifies one as
  necessary, stop and ask the owner before installing it, adding include paths,
  changing package metadata, or committing vendored code.
- The request must state the required task, alternatives considered, expected
  benefit, version/source, license, build/distribution impact, and CPU fallback
  implications. Record approved dependency decisions in the task plan and
  environment guide.

## Persistent Artifacts

| Artifact | Owner | Purpose |
|---|---|---|
| `docs/plans/<area>-plan.md` | Planner | End-to-end goal, decisions, vertical task backlog, dependencies, and task contracts |
| `currentstate.md` | Both | Facts only: completed behavior, known gaps/bugs, validation commands/results, blocked work, and commit references |
| Design documents in `docs/` | Planner, then Implementer | Normative target behavior and accepted technical decisions |
| `docs/AGENT_ENVIRONMENT.md` | Both | Verified build/test/model environment and dependency approval procedure |
| Tests and fixtures | Implementer | Executable evidence for the task contract |

Create a plan file before implementation begins. Extend an existing plan when
one already owns the area; do not create competing plans. `currentstate.md` is
not a speculative backlog. It records the current truth and links to the plan.

## Planner Role

### 1. Establish the End-to-End Outcome

Read the complete request and the nearest controlling implementation, tests,
documentation, current state, and relevant history. State:

1. User-visible or system outcome.
2. Invariants that must remain true.
3. In-scope and explicitly out-of-scope behavior.
4. Compatibility, security, lifecycle, failure, and observability constraints.
5. Existing evidence and unknowns.

Prefer repository behavior and existing designs over invention. Resolve
technical unknowns by reading, tracing, a focused experiment, or a small
reversible probe.

### 2. Ask Only for Non-Technical Ambiguity

Stop and ask the user before planning or implementing when the answer changes
product intent, policy, ownership, or an irreversible tradeoff, for example:

- Which user behavior, compatibility promise, data-retention rule, or security
  policy is desired.
- Which of two business priorities should win when both cannot be delivered.
- Whether a migration, public API break, new dependency, cost increase, or
  quality tradeoff is acceptable.
- Which model, hardware target, deployment environment, or benchmark outcome is
  the actual acceptance target when the request does not say.

Do not ask the user to resolve ordinary technical ambiguity. Investigate it and
record the chosen technical decision with evidence. Do not ask for a preference
that existing code, an SDD, or a repository convention already decides.

### 3. Slice the Work Vertically

Make tasks deliver thin, observable behavior through the whole affected path,
not horizontal layers such as "add structs", "add API", or "write tests".
Each task should be independently reviewable, testable, releasable behind an
existing or explicit feature gate when partial exposure would be unsafe, and
small enough for one implementation/validation loop.

Order tasks by dependency and risk:

1. Make ownership, lifecycle, input validation, or data-safety foundations
   explicit when they control later work.
2. Add one complete happy path with its failure semantics and focused test.
3. Add boundary cases, fallback behavior, concurrency, and observability.
4. Remove temporary boundaries or replace scaffolding with the target pipeline.
5. Run final integration, real-environment, migration, and documentation work.

Avoid giant "implement feature" tasks and isolated testing/documentation tasks.
Tests and docs belong to the task that changes the behavior.

### 4. Write Executable Task Contracts

Every task in `docs/plans/<area>-plan.md` uses this format:

```markdown
### T-<area>-NN: <outcome-oriented title>
Status: planned | ready | in-progress | blocked | done
Depends on: <task IDs or none>
Commit: <hash when done>

Goal
: <one observable outcome>

Scope
: <owning symbols/files and allowed interfaces>

Non-goals
: <nearby work deliberately excluded>

Design and failure contract
: <normal behavior, invalid/failure behavior, ownership/lifetime/concurrency
  rule, fallback rule, and telemetry or logging semantics where relevant>

Implementation steps
: 1. <small concrete step>
  2. <small concrete step>

Acceptance criteria
: - [ ] <observable behavior and exact invariant>
  - [ ] <failure or boundary behavior>
  - [ ] <no regression or compatibility condition>

Required validation
: - `<exact focused command or test>` -> <expected result>
  - `<integration/build command when warranted>` -> <expected result>

State update
: <facts to add/remove in currentstate.md and SDD/doc changes required>

Escalate before implementation if
: <non-technical ambiguity unique to this task, or "none">
```

Acceptance criteria must be falsifiable. "Works", "optimized", "clean", and
"no bugs" are not criteria. State exact inputs, outputs, error behavior,
concurrency/lifecycle result, telemetry meaning, or measurable invariant.

### 5. Plan Quality Gate

Before marking tasks `ready`, verify:

- The end-to-end goal is covered by the union of task acceptance criteria.
- Every task has one owner boundary, a smallest useful change, a test path, and
  no hidden dependency on an unplanned task.
- No task relies on a TODO, a hand-waved fallback, or telemetry as proof of
  correctness.
- A later task cannot invalidate a completed earlier task's contract.
- The plan identifies the last task that provides real-environment acceptance.

If this gate fails, revise the plan. Do not send an implementer into an
underspecified task.

## Implementer Role

### 1. Start One Ready Task

Work on one `ready` task only. Before editing:

1. Read its full task contract, `currentstate.md`, the controlling code path,
   the nearest existing test, and relevant design section.
2. Inspect the working tree. Preserve unrelated modifications.
3. State one falsifiable local hypothesis and the cheapest check that could
   disprove it.
4. Mark the task `in-progress` in the plan only after its dependencies and
   acceptance criteria are understood.

If the contract is incomplete or a non-technical decision is required, stop and
ask the user. If a technical detail is missing, investigate locally and record
the evidence; do not expand scope without updating the plan.

### 2. Implement the Smallest Complete Change

Follow the task contract and repository patterns. Add or update the smallest
focused test that fails without the new behavior, unless the task is a trivial
documentation-only correction.

For a behavior change, cover:

- the normal path;
- the stated error, fallback, or boundary behavior;
- ownership, cleanup, or concurrency semantics when touched;
- telemetry counter meaning when telemetry is part of acceptance.

After the first substantive edit, immediately run the task's cheapest focused
validation. Do not start adjacent work before interpreting that result.

### 3. Debug Loop Until the Task Is Actually Complete

Use this loop for every failed check, compiler warning, sanitizer finding,
unexpected log, or unmet acceptance criterion:

1. **Capture**: save the exact command, input, failure text, and relevant log.
2. **Classify**: decide whether it is a task defect, bad test expectation,
   environment failure, or an unrelated pre-existing failure.
3. **Trace**: find the owning code path and form one falsifiable root-cause
   hypothesis. Do not patch each caller when a shared owner controls it.
4. **Plan**: make the smallest repair that satisfies the task contract. If this
   changes the contract or crosses a task boundary, return the task to the
   Planner instead of silently broadening it.
5. **Repair**: edit only the necessary files.
6. **Revalidate**: rerun the same focused check first, then the next required
   check.

Repeat until every acceptance criterion is demonstrated. A passing build alone
does not close a behavior task. An empty error log alone does not prove
correctness. Do not mark a task complete with known task-owned bugs, unchecked
criteria, unexplained counter inconsistencies, or an untested failure path.

For an unrelated failure, do not fix it opportunistically. Record the command,
failure, and why it is outside the task in `currentstate.md`; continue only if
the task's own validation remains trustworthy. If it blocks trustworthy
validation, mark the task `blocked` and escalate.

### 4. Validation Ladder

Run the narrowest meaningful checks first, then widen only when the task
contract requires it:

1. Syntax, compiler, type, or static analysis for touched files.
2. Focused unit or regression test for the changed contract.
3. Focused integration test for the affected subsystem.
4. Repository baseline: `make check` from the repository root, or
   `make -C c check` for scripts running in `c/`.
5. CUDA changes: build and run the documented CUDA test path. On a CUDA-capable
   host, use `make -C c cuda-test CUDA_ARCH=native`; on the target Windows
   environment, use the documented `c/build-v4.ps1` workflow where applicable.
6. Model/runtime changes: run the smallest real-model smoke that proves the
   contract, then the plan's required larger validation. Record exact command,
   environment flags, model/hardware, exit status, and relevant telemetry.

Do not run expensive model or benchmark commands merely for confidence. Run
them when the task contract says they prove behavior. Do not treat latency or
throughput as a pass/fail gate unless the planner explicitly made it one.

## Completion, State, and Commit

A task is complete only when all are true:

1. Every acceptance checkbox is checked with linked or recorded evidence.
2. Required validation passed, or a documented environment limitation has an
   accepted alternative check and is listed as residual risk.
3. All task-owned failures found during the loop are fixed or explicitly
   rejected by the user as a non-technical decision.
4. `currentstate.md` reflects the new behavior, remaining gaps, validation
   result, and any changed risk. Remove superseded claims; never append a
   success claim that contradicts measured evidence.
5. The SDD and user-facing docs are updated when the contract, supported
   behavior, configuration, or acceptance evidence changed. Generated docs
   follow `docs/MAINTAINING-DOCS.md`.
6. The task plan records `done`, validation summary, residual risk, and commit
   hash.

Then make one atomic commit:

```text
<area>: <outcome>

Task: T-<area>-NN
Validation: <commands and result>
```

Stage only the task's files. Verify the staged diff and `git diff --check`
before committing. Do not amend or rewrite another agent's commit. If a commit
cannot be made because of branch policy, merge conflict, or user-owned changes
in the same files, mark the task `blocked` and ask the user how to proceed.

### 6. Continue Until the Plan Is Resolved

After each successful commit, inspect the plan for a `ready` task whose
dependencies are `done`.

1. If one exists, return to **Implementer Role, 1. Start One Ready Task** and
  complete it before ending the agent run.
2. If every task is `done`, run any plan-level final validation, update
  `currentstate.md` with the completed plan outcome, and report completion.
3. If no task is ready and any task is `blocked`, record the blocker and
  escalate; do not claim the plan is complete.
4. If no task is ready but tasks remain `planned` or `in-progress`, re-engage
  the Planner to repair the task graph or resolve the missing dependency.

Do not end an agent run after a completed task while another eligible `ready`
task remains. The Planner is re-engaged whenever a completed task changes
assumptions, a residual risk invalidates a later task, or implementation
discovers a new cross-cutting design decision.

## Definition of Done

"No bugs or gaps for this task" means no known defect or unverified acceptance
criterion remains within that task's declared scope. It does not mean the whole
project is done, unrelated failures are ignored, or a future task is silently
assumed complete. The plan and `currentstate.md` must make the remaining work
visible before the next task begins.