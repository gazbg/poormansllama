# Model-run testing standard (llama.cpp layer-residency workspace)

Status: TRACKED DRAFT (tracked 2026-09-13; owner decision — track, do NOT
inline into the plan). It is the standard the T-GPLR-15 contract defers to for
what counts as recorded evidence ("attached per the testing standard",
"recorded evidence per the model-run testing standard") and it is cited by path
in `docs/gplr13-bottlenecks-hypotheses.md`, so a clean checkout must contain
it. TRACKED DOES NOT MEAN APPROVED: this is still a draft and owner review is
still desired before it is baked into runners or CI.

[SUPERSEDED 2026-09-13 — retained as written] Status: draft standard for
agent-driven model runs and semantic-QA legs. Owner review desired before
baking it into runners or CI.

## Why this exists

The workspace keeps hitting the same class of test failure: model runs that
do not terminate cleanly, or that terminate in a mode that is hard to
reproduce and hard to grade. The fix is not more patience. It is a standard
_invocation shape_ that is:

- deterministic,
- non-interactive,
- finite by construction,
- reproducible from a stored prompt file, and
- explicitly terminated when the process is being stubborn.

That is why the phrase "llama-completion is the right vehicle: single-shot,
-n hard cap, -f prompt file" is pointing in the right direction. The phrasing
in that note is too informal to become the standard as written, but the
intent is sound.

## Standard shape

For any agent-driven model completion run used as a test or gradeable leg,
use this shape:

1. Single-shot completion, not conversation mode.
2. Prompt comes from a file, not from a live interactive buffer.
3. Generation is capped explicitly with a small hard limit suitable to the
   leg's purpose.
4. The runner is launched in a way that can be killed and retried without
   leaving a stuck conversation session behind.
5. The output used for grading is the detokenized generated-text artifact, not
   a fragile mixed log scrape, unless the task explicitly needs the trace.

In this workspace, that currently means the layer-streaming runner's
`--prompt-file` + capped `--tg-tokens` + `--tg-text` shape for the
experimental path, and the analogous stock llama.cpp completion shape (`llama-completion`
or `llama-cli` single-shot completion with a prompt file or `-p` and a hard `-n`)
for non-experimental legs. The exact flags per leg are a task contract detail;
the _principles_ above are the standard.

### Local single-shot completion vehicle

For this repo, the durable phrase "llama-completion is the right vehicle"
means: use a single-shot completion binary, not a long-running conversation
process, when the run is a test or a gradeable leg.

- Stock llama.cpp behavior: `llama-completion` (or `llama-cli` in its
  single-shot completion mode) with a prompt file or `-p`, a hard `-n`/`--n-predict`,
  and deterministic sampling flags.
- Layer-residency experimental path: `llama-layer-streaming.exe` with
  `--prompt-file`, a small explicit `--tg-tokens` cap, `--tg-text`, dedicated
  `--telemetry`, and `--check-telemetry`.

The point is not to mandate one exact binary for every task. It is to mandate
the invocation shape: single-shot, file-driven, capped, terminable, and
gradeable from a clean generated-text artifact.

## What "no loops" must mean in practice

Do not define "no loops" as a vague hope. Define it as something the test can
observe:

- the runner exits with a bounded exit code,
- the generated-text artifact exists and is non-empty when TG was expected,
- the telemetry/checker contract passes, and
- the negative control fails as designed.

If a run can spin, hang, or keep regenerating after the intended stop point,
that is a runner/leg problem, not a " rerun until it looks right" situation.

## Force-termination is a controlled fallback, not the primary path

It is reasonable to have a force-kill path for a stubborn process. It is not
reasonable to make that the normal way a test finishes.

The right hierarchy is:

1. Prefer a run shape that stops on its own.
2. Prefer a cap that makes "run forever" impossible for the leg.
3. Use force-termination only when the process has demonstrably stuck, and
   record that it happened.
4. If force-termination keeps becoming normal, the bug is the leg design, not
   the kill command.

## Existing tests are already close to this

The repo's semantic-QA smoke gate is a good model for the standard:

- prompt generated to a file with exact token calibration,
- runner invoked with `--prompt-file`,
- dedicated telemetry JSONL,
- strict telemetry check,
- grading on the runner's detokenized generated-text file,
- negative control that must fail.

When a runner supports both conversation mode and single-shot completion mode,
the standard prefers the single-shot shape for tests. If a runner lacks a clean
single-shot cap, that is a runner gap to close, not an excuse to run tests in a
mode that can wander.

That is exactly the kind of shape this standard wants to generalize: file-based
input, capped single-shot generation, explicit artifacts, and programmatic
grading.

## Canonical local command shapes

These are the approved shapes for agent-driven model runs in this repo. They are
examples of the standard, not the whole standard. A specific leg may add model,
context, quantization, scheduling, or residency flags; it must still keep the
single-shot / file-driven / capped / terminable shape.

### Stock llama.cpp completion leg

- Binary: `llama-completion` (or `llama-cli` in single-shot completion mode).
- Prompt: a prompt file or `-p`, where a file is preferred for reproducibility.
- Cap: explicit hard `-n`/`--n-predict` suitable to the leg.
- Sampling: fixed, deterministic, and leg-appropriate; do not leave a test run
  on a mode that can drift run to run unless the task explicitly studies that drift.
- Termination: the process should stop on its own because the cap is explicit.

### Layer-streaming experimental leg

- Binary: `llama-layer-streaming.exe`.
- Prompt: `--prompt-file <file>`.
- Cap: explicit `--tg-tokens <small N>`.
- Artifact: `--tg-text <file>` for grading, plus dedicated `--telemetry <jsonl>`.
- Gate: `--check-telemetry <jsonl>` on the dedicated telemetry file.
- Run under the shared-GPU mutex (`tools/with-gpu-lock.ps1`) when the leg touches CUDA.

That is the shape the repo's existing `qa-smoke` suite already uses, and the
reason it is the local model for this standard.

## Force-termination contract

Force-termination is a controlled fallback, not the normal finish line.

1. Design the leg so it stops on its own.
2. Add an explicit cap so "run forever" is structurally impossible for that leg.
3. If the process demonstrably sticks, force-terminate and record that it happened.
4. If force-termination keeps becoming routine, treat that as a leg-design bug,
   not as a testing-constant problem.

Do not use a force-kill as a substitute for writing a bounded invocation.

## Acceptance rules for "no loops"

Treat "no loops" as observable behavior, not a hope:

- the run exits with a bounded, expected exit code,
- the generated-text artifact exists and is non-empty when TG was expected,
- the telemetry/checker contract passes,
- and the negative control fails as designed.

If a run can spin, hang, or keep generating past the intended stop point, that
is a runner/leg problem and must be fixed there.

## What "llama-completion" means here

Do not leave that phrase ambiguous in a standard.

- For generic llama.cpp single-shot completion legs, "llama-completion" means
  the stock completion binary / single-shot completion CLI, used in non-interactive
  mode with a capped generation target and a file-based prompt when possible.
- For the layer-residency experimental path in this repo, the equivalent vehicle
  is the layer-streaming runner invoked with `--prompt-file`, a small explicit
  `--tg-tokens` cap, `--tg-text`, dedicated `--telemetry`, and `--check-telemetry`.

If a task says "use llama-completion," read it as "use a single-shot completion
invocation, not a conversation server or interactive loop."
