# D22 — setup wizard corrections (owner, 2026-11-07)

Status: done (validated 2026-11-07; AC-1..AC-10 all pass, see evidence below)
Branch: `claude/setup-wizard-d20-parallel` (the wizard's branch)
Dependencies: none (engine flags/env already exist; verified below)
Commit: f19ff144 (owner authorized 2026-11-07; one commit per the owner's
"commit all for the current tree": the 2026-11-06 evict-defaults work and the
D22 wizard corrections landed together; `.playwright-mcp/` left untracked)

## Validation evidence (2026-11-07)

- `bash -n setup.sh` clean; `git diff --check` clean.
- `tools/setup-wizard/test-d22.ps1`: all 53 checks PASS (all rows x cc
  {12.0, 8.9, 6.1, 5.2, 3.5, ''} x {prearm exactly once == always, int8
  iff cc<=6.1, idempotence}, env-already-set branch, hook).
- `tools/setup-wizard/test-d22.sh`: all 195 checks PASS (same matrix).
- PS dry run R8 (this machine, cc 12.0): `$port = 8090`,
  `--lm-prefill-prearm always`, no int8 flag, flag gate ok, exit 0 (AC-1).
- PS dry run R8 with `SETUP_WIZARD_GPU_CC=6.1`: both flags + `# D22:` note,
  exit 0 (AC-2). R7 (1070 row) under 6.1: prearm+int8 appended, keeps
  `--device CUDA0`; R3 (dense) under 6.1: prearm kept, int8 appended.
- Port: piped `9091` -> `$port = 9091`; `banana` -> warn + re-ask -> 9092;
  empty -> 8090; `-Port 7777` -> used, no prompt (AC-4/5).
- sh dry run R8: `PORT=8090`, prearm always, exit 0; with the cc hook +
  `--port 9090`: both flags + note + `PORT=9090`, exit 0; piped `9091` ->
  `PORT=9091` (AC-7).
- Doc sweep: no `8033` left except the D22 decision row's "was 8033" history
  note (AC-8). `smoke.ps1` default is 8090.
- AC-10: the pre-existing evict-defaults hunks are untouched (numstat
  identical before/after: arg.cpp 2+/1-, common.h 1+/1-, moe-hybrid 0+/3-,
  server-context.cpp 15+/2-). No GPU workload ran (dry runs + host-only
  checks; the flag gate's `--help` invocation is host-only).

## Goal

Three owner corrections to the setup wizard (setup.ps1 + its setup.sh mirror):

1. **D22-A (Pascal-and-lower int8 attention).** For a card whose compute
   capability is 6.x or lower, the emitted server config must enable the int8
   flash-attention tile: if the row's env already sets
   `LLAMA_FATTN_TILE_Q8QK=1`, that env is kept (it is already set — the
   "not different" case, nothing to add); otherwise the flag
   `--lm-prefill-attn-int8` is appended to the emitted command.
2. **D22-B (prearm always, all profiles).** Every recipe row emits
   `--lm-prefill-prearm always` (R8's `auto` becomes `always`; the rows
   without it get it appended; R3 already has it).
3. **D22-C (port input, default 8090).** The wizard asks the user for the
   server port in the settings phase; the default is **8090** (was 8033).
   An explicit `-Port` / `--port` suppresses the question, as before.

## Engine mechanics (verified in `third_party/llama.cpp`, not assumed)

- `--lm-prefill-attn-int8` is a server flag that literally does
  `setenv("LLAMA_FATTN_TILE_Q8QK", "1")` before the model load
  (`tools/server/server-context.cpp`), and the help text says
  *"Same as LLAMA_FATTN_TILE_Q8QK=1"* with the target stated as
  *"a GPU without fast FP16, such as Pascal"*. `ggml/src/ggml-cuda/
  fattn-tile.cuh` reads the env once at first flash-attention call. The two
  spellings are the same switch — that is what the owner's
  "if it is different from the env … if not different just make sure the
  env is set" branches on. It is an **approximation** (int8 Q per 32
  values); first-token KL vs the exact pass is recorded in
  `docs/plans/dense-qwen38-iq3-residency.md`.
- `--lm-prefill-prearm {off,always,auto}` (`common/arg.cpp`); the engine
  default is `always` (1), but the server AUTO-TURNS the implicit default
  OFF where it was not validated: **without `--hybrid-moe`** (i.e. the dense
  rows R1–R3), for `-np != 1`, or for speculative decoding except MTP with
  `--spec-mtp-defer`. An explicit flag is always honored (it only refuses on
  the `-np` and non-exempt-spec checks). Acceptance on every row was
  checked row by row:
  - R1, R2 (dense, `--spec-type draft-mtp --spec-mtp-defer`, `-np 1`) → the
    MTP-defer exemption holds (`has_spec && !(spec_mtp && !has_draft &&
    mtp_defer)` is false) → `always` runs.
  - R3 (dense, no spec) → `always` runs (already had it).
  - R4, R7 (hybrid, no spec) → `always` runs.
  - R5, R6 (hybrid, MTP defer) → exempt → `always` runs.
  - R8 (hybrid, no spec, `-np 1`) → `always` runs (replaces `auto`).
  All rows carry `-np 1`.
- No engine change is needed for any of the three corrections.

## Scope

- `setup.ps1`: param default, settings-phase port question, GPU compute-cap
  detection (+ test hook), row-flag completion, `Emit-RunScript` wiring.
- `setup.sh`: the same, mirrored (D22 decisions are OS-agnostic).
- `tools/setup-wizard/smoke.ps1`: default port 8090 + usage example.
- Docs: `docs/setup-wizard-sdd.md` (D22 decision row, §4.6/§4.7 text,
  measured-matrix table), `docs/SETUP.md` (8033 → 8090, port row),
  `docs/guide/USAGE.md` (port examples), `docs/currentstate.md` (entry).

## Non-goals

- No engine (vendored tree) changes.
- No new measured recipe rows; the rows' Measured/Note text is untouched.
- The generated `run-server.last.*` artifacts and `.playwright-mcp/` are
  untouched.
- The uncommitted 2026-11-06 evict-defaults work stays as-is (separate
  logical change, owner commits it separately).
- No GPU work of any kind (dry runs + host-only checks only).

## Design / failure contract

- Arch detection: `nvidia-smi --query-gpu=compute_cap` (sh already parses it
  for the build arch). The card the run will use = nvidia-smi's first line,
  consistent with the D12/D17 convention. A test hook
  `SETUP_WIZARD_GPU_CC=M.m` overrides on both OSes (the documented
  force-missing hook is the precedent). Detection failure is NOT fatal:
  the row is emitted as-is with a warning (the flag is a performance
  approximation, not a correctness requirement).
- Completion is a **pure function** of (row flags, row env, compute cap),
  applied once in the settings phase and consumed by the flag gate and the
  emitter, so both wizards keep the rows as data and the decision is
  unit-testable without a GPU:
  - `--lm-prefill-prearm <mode>`: an existing occurrence is replaced by
    `always`; otherwise ` --lm-prefill-prearm always` is appended.
  - int8 (only for cc ≤ 6.1): if the row env already contains
    `LLAMA_FATTN_TILE_Q8QK=1` → leave the env (it is emitted, nothing to
    add) and print one line saying so; otherwise append
    ` --lm-prefill-attn-int8`.
  - The emitted script additionally prints a `# D22:` note line when
    `--lm-prefill-attn-int8` was added, so a future reader sees it is an
    arch-added approximation, not part of the measured row.
- Port: empty `-Port`/`--port` = "ask"; the question sits with the other
  settings (after the recipe is known); input is validated to an integer in
  1–65535 and re-asked otherwise; non-interactive mode uses the default
  8090 without asking.
- Idempotence (SDD AC-2): all additions are deterministic, so a same-day
  re-run still produces a byte-identical `run-server.*`.

## Implementation steps

1. `setup.ps1`: `detect_gpu_arch` + `Complete-RowFlags` (pure) + port
   question + param default + `Emit-RunScript` wiring (D22 note,
   `$port`/`$bind` guards, health-comment port).
2. `setup.sh`: `detect_gpu_arch` + `complete_row_flags` (pure,
   say→stderr) + port question + `PORT` default + usage text +
   `emit_run_script`/`check_row_flags` wiring.
3. `smoke.ps1` default port + example.
4. Docs (SDD D22 row + §4.6/§4.7 + matrix table; SETUP.md; USAGE.md;
   currentstate entry).
5. Validation (below).

## Acceptance criteria (falsifiable)

- AC-1: PS dry-run (NI, qwen36/16g/262k → R8, this machine = cc 12.0) exits
  0 and the emitted command contains `--lm-prefill-prearm always`, does NOT
  contain `--lm-prefill-attn-int8`, and the script has `$port = 8090`.
- AC-2: same with `SETUP_WIZARD_GPU_CC=6.1`: the emitted command contains
  BOTH `--lm-prefill-prearm always` and `--lm-prefill-attn-int8`, a
  `# D22:` note line is present, and the exit code is 0.
- AC-3: same with `SETUP_WIZARD_GPU_CC=6.1` on an R7-shaped env (a row that
  HAS `LLAMA_FATTN_TILE_Q8QK=1` in its env): no flag appended, the "env
  already set" line is printed instead. (Proven by the pure-function
  harness, not a real row: no current row carries that env.)
- AC-4: PS interactive (non-DR, `-Vram 16g -Context 262k` so only the port
  prompt fires, `-SkipBuild`, piped input `9091\n\n`): emitted `$port =
  9091`. Piped `banana\n9092\n`: re-asked, then `$port = 9092`.
- AC-5: `-Port 7777` (DR): no port prompt in the output, `$port = 7777`.
- AC-6: every row R1–R8 in BOTH wizards emits `--lm-prefill-prearm always`
  (pure-function harness over all rows, cc 12.0 and cc 6.1).
- AC-7: sh `--port 9090` is honored (unit/NI path); sh dry-run with
  `SETUP_WIZARD_GPU_CC=6.1` on R8 emits both flags and `PORT=8090` by
  default (bash harness in Git Bash, env-check bypassed via the unit cut;
  `bash -n` passes on the whole file).
- AC-8: `smoke.ps1` default is 8090; no `8033` remains in setup.ps1,
  setup.sh, smoke.ps1, docs/SETUP.md, docs/setup-wizard-sdd.md,
  docs/guide/USAGE.md.
- AC-9: `--help` / `-DryRun` help text shows the new port default.
- AC-10: no GPU workload ran; the evict-defaults hunks in the worktree are
  byte-identical before/after (they touch files this task does not, except
  a separate currentstate.md region).

## Validation commands (exact)

```
# pure-function harness (PS): all rows x {12.0, 6.1, 5.2} + env-already-set case
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup-wizard/test-d22.ps1
# PS dry runs (R8 on this machine; R7 + cc hook for the 1070 shape)
powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -DryRun -NonInteractive -Model qwen36 -Quant q4k_m -Vram 16g -Context 262k -SkipBuild -NoStart
$env:SETUP_WIZARD_GPU_CC='6.1'; powershell ... (same, expect int8 flag + note); Remove-Item env:SETUP_WIZARD_GPU_CC
# interactive port prompt (pipe)
echo 9091 | powershell -NoProfile ... setup.ps1 -NonInteractive:$false -SkipBuild -SkipDownload -Model qwen36 -Quant q4k_m -Vram 16g -Context 262k -NoStart   (see notes: uses the local model file, no download)
# sh: syntax + unit cut
bash -n setup.sh
bash tools/setup-wizard/test-d22.sh
# doc sweep
rg -n "8033" setup.ps1 setup.sh tools/setup-wizard/ docs/SETUP.md docs/setup-wizard-sdd.md docs/guide/USAGE.md
```

Expected: AC-1..AC-10 all hold; the doc sweep is empty.

## State / doc updates

- `docs/currentstate.md`: D22 entry (asked/answer/validated/scope).
- `docs/setup-wizard-sdd.md`: D22 decision row; §4.6 (port default 8090,
  arch-added flag, prearm normalization); §4.7 (D22 note line); matrix table
  R8 `auto` → `always` note.
- `docs/SETUP.md`: flag table + FAQ ports.
- `docs/guide/USAGE.md`: port examples.

## Escalation condition

- The built binary's `--help` refuses `--lm-prefill-prearm` or
  `--lm-prefill-attn-int8` (stale build of the flag gate) → rebuild or
  report; do not emit rows the binary cannot run.
- A dry run on this machine triggers a real build or download (it must not:
  `-DryRun`, local model present, binary-fresh check) → stop and inspect.

## Decision log

- **D22-A interpretation.** The owner said: add `--lm-prefill-attn-int8`
  "if it is different from the env LLAMA_FATTN_TILE_Q8QK=1", else "make sure
  the env is set". Verified: the flag IS the env (the server sets the env
  from the flag). So "different" = the row's config does not already enable
  that switch. Implementation: env already set → keep the env as-is (it is
  the emitted config; nothing to add); otherwise append the flag. Reversible:
  drop the arch branch in `Complete-RowFlags`/`complete_row_flags`.
- **D22-B semantics.** "Add `--lm-prefill-prearm always` for all profiles"
  is implemented as "the emitted value is always `always`" — an existing
  `auto` (R8) is replaced, an existing `always` (R3) is kept, a missing
  occurrence is appended. Reason: the owner asked for the profiles to HAVE
  `always`; two occurrences of the flag would be a parse hazard. Side effect
  (intended): on the dense rows R1–R3 the idle preparation actually runs
  now, because the explicit flag overrides the server's auto-off
  (`no --hybrid-moe`) — previously the rows relied on the implicit default,
  which the server turns off for non-hybrid configs.
- **Port question placement.** After the recipe is picked, before the
  parallel phase (settings phase, D20 flow). Reason: it is a server setting
  like VRAM/context, and D20 requires every question before heavy work.
- **cc hook name.** `SETUP_WIZARD_GPU_CC` (both OSes), mirroring
  `SETUP_WIZARD_FORCE_MISSING`.
- **No commit** — owner has not authorized one; the worktree's existing
  evict-defaults hunks stay untouched (this task adds no hunk to any file
  they own, except a separate currentstate.md region).
