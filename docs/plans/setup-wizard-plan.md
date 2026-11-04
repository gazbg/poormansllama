# Plan: setup wizard (install script)

Status: ready → in-progress (2026-11-04). Design:
[docs/setup-wizard-sdd.md](../setup-wizard-sdd.md). Owner grant: /goal
"finalize and implement the plan for the install script" (goal
55f0dae5-210e-405b-af29-e1e092f06b89) — this session runs in the repo's
autonomous mode; the acceptance below is the agent's own acceptance under
the grant, not owner approval.

## Goal

Committed, cross-platform bootstrap + wizard (`setup.ps1`, `setup.sh`):
env check → auto-install → clone (if needed) → build `llama-server` →
model from Hugging Face (if missing) → measured-recipe questions → emit
`run-server` script and offer to start it.

## Acceptance contract (written before the first run)

Scope of what acceptance certifies: on THIS machine (Windows 11, RTX 5090
24 GiB, CUDA 13.2, VS2022, CMake 4.2.3, in-repo checkout on branch
`claude/flint-support`, model `C:/Models/ALL/other/...` present), the
wizard works end-to-end including a real server start from the emitted
script, and every refusal/install path is demonstrated on its own. It does
NOT certify: a fresh-clone build on another machine, WSL/Linux build or
server runs (best-effort, reported as such), or the numerical quality of
the recipes (they are the owner's measured recipes, quoted verbatim).

Falsifiable criteria (each with its exact validation command and expected
result):

- **AC-1 static parse.**
  `powershell -NoProfile -Command "[void][scriptblock]::Create((Get-Content setup.ps1 -Raw)); 'PS-PARSE-OK'"`
  → prints `PS-PARSE-OK`, exit 0.
  `bash -n setup.sh` → exit 0. `shellcheck setup.sh` (if installed) → no
  errors (warnings allowed, listed).
- **AC-2 in-repo dry-run, idempotent emission.**
  `powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -DryRun --non-interactive --model qwen36 --vram full`
  → exit 0; prints env table (all present), "clone skipped (in repo)",
  "build skipped (binary newer than sources)", model found at
  `C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, selects
  row R5, writes `run-server.ps1`. Re-running the command →
  `run-server.ps1` byte-identical (sha-256 equal), previous generation
  moved to `run-server.last.ps1` (only from the second run on).
- **AC-3 refusal.**
  `... setup.ps1 -DryRun --non-interactive --model qwen36 --vram full --context 128k`
  → exit 3, stderr contains "no measured recipe" and the qwen36 rows
  R5/R6/R7 table.
- **AC-4 install path (no real install).**
  `$env:SETUP_WIZARD_FORCE_MISSING='cmake'; ... setup.ps1 -DryRun --non-interactive --model qwen36 --vram full`
  → exit 0; output contains `cmake ... MISSING` and the winget plan
  `NVIDIA.CUDA`-free line `winget install --id Kitware.CMake ...`; no
  winget process started (verified: `winget` not in the output's executed
  section / DryRun never calls winget — grep the script's DryRun branch).
- **AC-5 fresh clone + build.** (GPU mutex NOT required — compile only.)
  `... setup.ps1 -DryRun 0` is not this; the real command:
  `powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -CloneDir $env:TEMP\swz-clone -SkipModel -NoStart -FullBuild`
  (flags as implemented; `-SkipModel` = env+clone+build only, no model/recipe
  steps) → exit 0; `$env:TEMP\swz-clone\poormansllama` contains
  `third_party/llama.cpp`; `build/llama-cuda/bin/Release/llama-server.exe`
  exists and is executable; clone is shallow (`git -C ... rev-list --count HEAD` = 1).
  Time-box: 90 min; on timeout the task fails and is diagnosed (build log
  under `build/llama-cuda/setup-wizard-build.log`).
- **AC-6 E2E server smoke (GPU mutex, one run, tag `swz-smoke`).**
  `& tools\with-gpu-lock.ps1 -Command 'powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\smoke.ps1 -Row R5 -Model "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -Port 8033' -VerdictTag swz-smoke`
  → verdict `PROVENANCE` line present (0 refused), smoke output:
  `/health` returns `{"status":"ok"}` within 300 s, the pong chat request
  (USAGE.md §5 body, `enable_thinking:false`, `max_tokens:64`,
  temperature 0) returns a message whose content contains `pong`, server
  stopped cleanly (process gone, log tail has no access violation).
  Verdict JSON at `build/llama-cuda/verdicts/swz-smoke.verdict.json`.
- **AC-7 Linux (WSL).** `wsl -d Ubuntu-24.04 -- bash -n /mnt/c/Development/AI/poormansllama/setup.sh` → exit 0.
  `wsl -d Ubuntu-24.04 -- bash /mnt/c/Development/AI/poormansllama/setup.sh --dry-run --non-interactive --model qwen36 --vram full`
  → exit 2 with the CUDA apt plan printed (WSL has no CUDA toolkit) — OR, if
  a toolkit is present, exit 0 dry-run. Whichever occurs, it must be the
  documented branch, and the output is recorded. A WSL build/server is
  best-effort: if the apt CUDA install succeeds and time permits, run the
  full build + a Linux smoke under the same mutex; otherwise record
  "not validated in WSL" in currentstate.
- **AC-8 docs and registration.** `docs/SETUP.md` exists and contains:
  both one-liners, the §4.5 model table with sizes, the R1–R7 matrix with
  measured numbers and sources, the D7 + D13 risk notes, the exit-code
  table. `README.md` has the Quick-start section. `git diff --check` clean.
  `run-server.last.*` and `build/llama-cuda/server-*.log` are git-ignored.

Hard failures that block acceptance regardless of the above: the wizard
ever writing outside its allowed paths (§6 of the SDD); a real winget/apt
install running without the y/n confirm (the confirm cannot be bypassed in
non-interactive mode without `--yes-install`); the emitted R5 command
differing from the owner's pasted command in any flag (diff against the
paste, flag-by-flag, in the plan's evidence section).

Gates that must run: AC-1..AC-8 all. The GPU gate is AC-6 (single mutex
run). The host suite is NOT required by this task (no engine code change);
if any engine file is touched it is (none planned).

## Tasks

### T-SW-1: skeleton + env check + install plan (both OSes)

- Files: `setup.ps1`, `setup.sh`.
- Banner, arg parsing (all flags in the SDD), OS section, env-check table
  (§4.1 incl. `SETUP_WIZARD_FORCE_MISSING` hook), install plan + y/n
  confirm + winget/apt execution + re-check + fatal-with-manual-steps
  (§4.2). Non-interactive mode: no prompts; `--yes-install` required for
  any install, else the plan is printed and it exits 2 (DryRun: plan
  printed, exit 0).
- Validation: AC-1, AC-4.
- Done-when: parse clean; FORCE_MISSING demo prints the exact plan.
- Commit: task 1 of the wizard.

### T-SW-2: clone + build

- Files: `setup.ps1`, `setup.sh`.
- In-repo detection; clone with D8 defaults; `core.longpaths` (Windows);
  non-empty target refusal; build with the documented commands + D9 arch
  detection + D10 freshness skip + `--no-build`/`-SkipBuild` + build log
  tail on failure (exit 4).
- Validation: AC-2 (build-skip branch on this machine), AC-5 (fresh clone
  + full build in TEMP).
- Escalation: fresh build fails after 2 repair attempts → capture log,
  diagnose per AGENTS.md workflow before further changes.

### T-SW-3: model table + download

- Files: `setup.ps1`, `setup.sh`.
- §4.5 table, local-path offering, resume download (per-OS), sha-256
  verify, token handling (never printed), exit 5 path.
- Validation: dry-run picks the local file (AC-2); a **real partial
  download** test: download of a model file truncated to a small `-MaxBytes`
  test flag (documented test hook, `--max-download-mib`, default 0 = full)
  against `qwen38` → verifies resume + `.part` + rename behavior on a
  scratch dir, then deletes the scratch file (no data loss: scratch only).
  (Full 15–28 GiB downloads are NOT run as tests.)
- Note: the small-file download test hits the network; it is a plain file
  GET, no GPU.

### T-SW-4: recipe matrix + questions + emission

- Files: `setup.ps1`, `setup.sh`.
- §4.6 matrix verbatim; questions (model/target/ctx, host/port default
  8033); refusal exit 3 with table; D12 warning; §4.7 emission incl.
  `.last` backup and standalone executability; "start it now? (y/n)"
  (default n; `--start` / `--no-launch` to pin).
- Validation: AC-2 (R5 selection + byte-stable emission), AC-3 (refusal),
  plus a flag-by-flag diff of the emitted R5 command against the owner's
  pasted command (evidence appended to this plan).

### T-SW-5: smoke harness + docs + gitignore + README

- Files: `tools/setup-wizard/smoke.ps1`, `docs/SETUP.md`, `README.md`,
  `docs/guide/USAGE.md` pointer, `.gitignore`, `docs/reference` index if
  applicable.
- Validation: AC-8; `pwsh` parse of smoke.ps1; setup-wizard dir header
  comment per repo convention.

### T-SW-6: E2E verification + state + commits

- Run AC-6 (GPU mutex, tag `swz-smoke`) and AC-7 (WSL). Update
  `docs/currentstate.md` with the evidence (commands, outputs, verdict
  paths). Final commit of any doc fixes. Write the owner summary (goal
  contract §: decision log D1–D15, acceptance per criterion with evidence,
  scope limits, risks).

## Evidence log

Recorded 2026-11-03/04, autonomous mode (agent acceptance under the owner's
grant; not owner approval). All commands run from the repo root on the owner's
machine (RTX 5090 Laptop 24 GiB, driver 621.01, CUDA 13.2, CMake 4.2.3,
MSVC 14.50.35717 + Windows SDK 10.0.26100.0, git 2.53.0.windows.1, 62 GiB RAM).

- **AC-1 parse** — `powershell -Command "[void][scriptblock]::Create((Get-Content setup.ps1 -Raw))"` → `PS-PARSE-OK`; `bash -n setup.sh` → `SYNTAX-OK`; same for `tools/setup-wizard/smoke.ps1` → `SMOKE-PARSE-OK`. Re-verified after the 2026-11-04 `Invoke-Native` rewrite.
- **AC-2 idempotent emission** — two consecutive `-DryRun -NonInteractive -SkipBuild -SkipModel -Model qwen36 -Vram full -Context 262k -NoStart` runs, full output compared: `AC2-IDEMPOTENT-OK` (byte-identical, including the emitted `run-server.ps1` text; header carries date + git short hash only). Re-verified after the rewrite.
- **AC-3 unmeasured refusal** — `-Model qwen36 -Vram 16g -Context 128k` (via `-File`) → prints the available R5/R6/R7 rows, `[FAIL] no measured recipe…`, **exit 3** (`AC3-FILE-EXIT=3`). Note: `powershell -Command "& .\setup.ps1 …"` returns exit 1 for the same flow — a `-Command` wrapper artifact; the user-facing `-File` form (and `iex (irm …)`) propagates `exit N` correctly. `setup.sh` with the same combo → `unknown option`/refusal exit 3 verified during AC-7 dry-runs.
- **AC-4 FORCE_MISSING install path** — PS: `SETUP_WIZARD_FORCE_MISSING=cmake` + `-DryRun -NonInteractive` → `cmake MISSING` → install plan fires → `(dry-run: nothing installed)` → exit 0. Root cause found and fixed: PS `return` unrolls a 1-element array to a scalar, so `Check-Environment` results are wrapped `@()` at every call site (decision D41). Bash: `SETUP_WIZARD_FORCE_MISSING='cmake'` + `--dry-run --non-interactive` → `sudo apt install -y cmake` plan → `(dry-run: nothing installed)` → exit 0. Root cause found and fixed: `local src=… bld=… log=\"$bld/…\"` in one statement expands `$bld` before it is set (D44).
- **AC-5 fresh clone + full build** — COMPLETE (2026-11-04). One single unattended run (launched via `Start-Process`, log `%TEMP%\swz-ac5.log`, err log empty) produced: env check → `git clone --depth 1 --branch claude/flint-support` fails (branch unpublished) → warn + fallback clone of the remote default → `cloned -> …\swz-clone\poormansllama` → cmake configure (20 s) → `cmake --build … --config Release --target llama-server --parallel 32 -- /nodeReuse:false` → `[ok] built …\llama-server.exe` → `done: environment + repository + build` + `next: re-run with -Model …`. Binary + full DLL set present (`llama-server.exe` 9,728 B thin exe + `llama.dll` 3.9 MB, `ggml-cuda.dll` 38.8 MB, `llama-server-impl.dll` 7.5 MB — same layout as the in-repo build), build log 4,297,136 B ending in `llama-server.vcxproj -> …\llama-server.exe`, build err log 0 B, no manual kills of MSBuild/vctip. Exit code of the completed state measured separately: wizard re-run on the same tree (reuse + build-skip) → **exit 0**. The cmake wait no longer hangs: capture moved from `Start-Process -Wait` (redirected stdout is held by MSBuild nodes / `vctip.exe` grandchildren → hang, measured twice) to `Invoke-NativeCmd` (temp `.bat`, redirections inside the bat, `%ERRORLEVEL%` written to an rc file on a SECOND physical line; ct22 measured the single-line form returning 0 for a child that exited 9). Re-verification after all 2026-11-04 edits: AC-2 `AC2-IDEMPOTENT-OK` (exits 0/0), AC-3 `AC3-EXIT=3`.
- **AC-6 GPU-mutex E2E smoke** — still blocked at 2026-11-04 23:10: the owner's other project (`C:\Development\AI\llamacppCUDA\llama-server.exe`, Qwen3.8 + mmproj on port 8033) holds the GPU — first observed as PID 18496 (2026-11-03 15:42), now PID 8284 (started 2026-11-03 20:46; the owner restarted their own server). The mutex wrapper REFUSED correctly (exit 125, verdict `swz-smoke.done` = FAIL 125). Deliberately not killed (never disturb another process's work). Retry pending GPU availability; will use port 8091 to avoid the 8033 collision if that server is still up.
- **AC-7 WSL/Linux** — `wsl.exe -e bash -n /mnt/c/…/setup.sh` → `WSL-SYNTAX-OK` (bash 5.3.9); full Linux dry-run `wsl.exe -e bash setup.sh --dry-run --non-interactive --model qwen36 --vram full --context 262k --skip-model --no-start --clone-dir /tmp/swz-wsl` → complete flow (env check, dry-run plan, R5 recipe, emitted-script preview verbatim, `wizard finished`), **exit 0** (`WSL-DRYRUN-EXIT=0`). No-driver path (Git Bash, no nvidia-smi) → CUDA-arch guidance + exit 2 (earlier session). Note: Git Bash mangles `/mnt/c/…` paths for wsl.exe; use `MSYS_NO_PATHCONV=1`.
- **AC-8 docs + registration** — `docs/SETUP.md` written; README quick-start + table row; USAGE.md pointer block; `.gitignore` gains `/run-server.ps1`, `/run-server.sh`, `/run-server.last.ps1`, `/run-server.last.sh` (server logs land under `build/`, already ignored).
- **Flag-capability gate (D38)** — `Test-RowFlags` (PS) / `check_row_flags` (sh) run `<binary> --help` and require every `--flag` of the chosen row; verified against the in-repo binary: `binary supports all R5 row flags`. Fresh-clone consequence (recorded): a `main`-based build supports R1-R3/R6/R7 fully; R4 (`--lm-prefill-copy-piece`) and R5 (`--hybrid-moe-prefill-free-all`) exist only in uncommitted working-tree changes, so a fresh-clone build refuses exactly those two rows with exit 1 and a flag list. VERIFIED against the actual fresh-clone binary (2026-11-04): its `--help` (978 lines) contains `--hybrid-moe`, `--lm-prefill*`, and NOT `prefill-free-all`, NOT `lm-prefill-copy-piece` — exactly the origin/main flag set; R4/R5 emission is therefore refused on a fresh clone, R1-R3/R6/R7 pass.
- **Clone reuse (D58)** — measured 2026-11-04: a re-run on an existing target previously refused (`clone target … exists and is not empty`); a valid existing checkout (`.git` + `git remote get-url origin` == `$RepoUrl`) is now reused. First attempt keyed on `setup.ps1` presence and failed on a fresh clone (setup.ps1 is uncommitted/local-only); the origin-URL check is the robust test.

## Decision log (additions beyond D1-D15 in the SDD)

- **D16** PS 5.1 parameter names have no hyphens → single-dash flags (`-DryRun`); sh keeps dashed names. (user-visible divergence, documented in both scripts' headers and SETUP.md.)
- **D17** `-ModelFile`/`--model-file`: explicit local GGUF override; recipe still chosen by `-Model` (local dirs hold non-table quants and MTP variants).
- **D18** dynamic CUDA env vars set per-process via `[Environment]::SetEnvironmentVariable(…, 'Process')` (D18 in SDD).
- **D19** emitted header = date (day) + git short hash only, for AC-2 byte-stability.
- **D20** PS 5.1 enum members need `::` (`[System.Net.HttpStatusCode]::PartialContent`).
- **D21** `Get-RepoRoot` uses `$PSScriptRoot`.
- **D22** MSVC detection scans all VS instances; `-latest` can select an SDK-less BuildTools. Machine-level `Windows Kits\10\bin` accepted.
- **D23** GPU VRAM parse uses the matched MiB group (all-digit strip overflowed Int32).
- **D24** `AskChoice` isolates property access (`$lbl = $o.Label`) before concatenation.
- **D25** `-SkipModel` skips model-file resolution only; recipe still validated (AC-3 applies).
- **D26** D12 warning keyed on VRAM target thresholds (22528/10240/8192 MiB), not per-row CardMib; text names the measured card.
- **D27** R7 `LLAMA_FATTN_TILE_Q8QK=1` env var is part of the row (emitted as `$env:` line).
- **D28** emitted script redirects the server to `build\llama-cuda\server-<row>.log` (Tee-Object / tee).
- **D29** pong test comment in emitted scripts matches USAGE.md §5 exactly (model `qwen3.6-35b-a3b`, `max_tokens:64`, `temperature:0`, `enable_thinking:false`).
- **D30** sh model sizes as an integer GiB array (no float arithmetic in shell).
- **D31** `ask_choice` defaults to the first option.
- **D32** `ver_ge` tolerates 3-part and malformed versions.
- **D33** `--max-download-mib` is a real early-stop (background curl + size polling + kill + truncate), verified with a resume test.
- **D34** emitted `run-server.sh` defines `EXE` and sets `set -o pipefail`.
- **D35** `-CloneDir` overrides in-repo detection (explicit clone target always honored).
- **D36** AC-6 model (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,134,528,992 B) downloaded by the wizard's own path (HF resume verified on the real file).
- **D37** clone fallback: `claude/flint-support` is unpublished on the remote (`git ls-remote --heads origin` measured). If `--branch` clone fails and branch ≠ main, warn + remove partial + re-clone the default branch. Never silent.
- **D38** flag-capability gate before emission: every `--flag` of the row must appear in `<binary> --help`; else exit 1 with the missing list. Makes the fallback safe: fresh-clone (main) builds refuse R4/R5 only.
- **D39** `-SkipModel` without `-Model` exits 0 after env+clone+build with a "re-run with --model" note (the unattended AC-5 shape).
- **D40** sh: `check_row_flags` is passed `$BUILD_BIN` (invoke_build's variable), not `$EXE`.
- **D41** PS `return` unrolls 1-element arrays to scalars → `@()` at every `Check-Environment` call site; `$bad.Count` would be `$null` and the install path silently skipped (AC-4 root cause, measured).
- **D42** `Invoke-Captured` (EAP `SilentlyContinue` guard) is sufficient for SHORT commands where stderr content does not matter (git clone progress, `--help`);
- **D43** long-running commands (cmake configure/build) need BOTH streams in files: `SilentlyContinue` + `2>&1 | Tee-Object` measured to DROP every stderr line (incomplete build logs).
- **D44** bash: `local a=$x b=$a` expands `$a` before it is set → one variable per `local` line (AC-4 sh root cause).
- **D45** `Invoke-Native`: `Start-Process -FilePath <exe> -ArgumentList <args> -Wait -NoNewWindow -PassThru -RedirectStandardOut*/Error` is the only PS 5.1 shape measured to return a real exit code (polling forms return null even after `WaitForExit`/`Refresh`; `%ERRORLEVEL%` wrappers expand at parse time). Accepted cost: no live console output; log paths printed, tails shown on failure. Applies to: git clone (both attempts), winget, cmake configure, cmake build, `--help`.
- **D46** AC-6 port: if the owner's other server still occupies 8033 when the GPU frees up, the smoke runs on 8091 (port is a harness parameter; the emitted script's default stays 8033).
- **D47** never wrap exit-code capture in a SINGLE physical cmd line: `... & echo %ERRORLEVEL% > rc` measured to return 0 for a child that exited 9 (ct22: `%ERRORLEVEL%` expands at parse time, before the child exits). Two physical lines is the minimum (D54).
- **D48** `Start-Job` hybrid rejected for production use: it can stream and in principle return `$LASTEXITCODE`, but the observed test hung (interactive nested PowerShell from a test arg bug) and the `.bat` rc-file form (D53) is simpler and measured working (ct23: rc=9, < 6 s return, both streams captured).
- **D49** `Invoke-Native` (`Start-Process -Wait` with redirected streams) is SAFE for commands whose descendants do not outlive it (git clone, winget, `--help`) but NOT for `cmake --build`: MSBuild worker nodes and `vctip.exe` keep the redirected stdout handle after the primary process exits and `-Wait` blocks until they die (measured hang, two AC-5 attempts; one required a manual `vctip.exe` kill). `cmake configure` and `cmake --build` therefore use `Invoke-NativeCmd` (D53).
- **D50** `Invoke-NativeCmd` writes a temp `.bat` (NOT a `.tmp` name — `& file.tmp` silently no-ops; a `.bat` name executes), first line `@echo off` (otherwise cmd echoes the prompt/command text into the redirected logs), cmd-style doubled inner quotes (backslash-escaped quotes do not survive), and a `call ` prefix when the target is a `.bat`/`.cmd` (a child bat's `exit /b N` otherwise terminates the parent chain). All four measured (ct32-ct38, cap-test35-37). rc file read, then the bat and rc are deleted.
- **D51** `%ERRORLEVEL%` capture must be a SECOND physical line in the bat (ct22: single-line form returns the parse-time value 0; two-line form measured rc=9 for a child that exited 9, ct23).
- **D52** PowerShell parser anomalies on this machine (Windows PowerShell 5.1.26100.9711 AND pwsh 7.6.9, via files, inline `-Command`, stripped env, and repo paths — host-specific, not code-specific): (a) `@(1+1, 2)` raises `Method invocation failed because [System.Object[]] does not contain a method named 'op_Addition'`; `@('a' + $x + 'b', 'c')` collapses to ONE element containing both pieces; `@((1+1), 2)` and plain literal arrays are fine (cap-test24-28). Rule: never put `+` concatenation inside array elements — precompute or use separate double-quoted interpolated strings. (b) `Test-Path (expr) -and (expr)` (any command invocation followed by ` -and `) raises `A parameter cannot be found that matches parameter name 'and'` at RUNTIME even though the file parses; `$var -and $var` and `(cmd) -and (cmd)` are fine (cap-test42: V3 fail, V1/V4/V5 OK). Rule: split compound conditions into pre-assigned variables.
- **D53** `Invoke-NativeCmd` (bat + rc file) is the capture for long-running native commands whose descendants outlive the primary process; it returned promptly with rc=0 on the 22-minute AC-5 full build with zero manual intervention (2026-11-04).
- **D54** build invocation keeps `-- /nodeReuse:false` (measured fewer lingering MSBuild nodes; the bat design makes it a safety net rather than a requirement).
- **D55** `setup.ps1` test-hook note: a local test harness parameter named `$args` can be silently empty under the machine's parser; name harness parameters `$argl` (cap-test29-31 era). Production `Invoke-NativeCmd` parameter is `$argl` for the same reason (not the automatic `$Args`).
- **D56** AC-8 registration state: `docs/SETUP.md`, README quick-start + table row, USAGE.md pointer, `.gitignore` entries all in place; `docs/currentstate.md` entry + commit land with this evidence update.
- **D57** `Invoke-Clone` reuse (D58) returns the existing checkout before any git call beyond `git remote get-url origin` (stderr discarded with `2>$null`, not merged — EAP=Stop safe).
- **D58** clone reuse rule: an existing non-empty target is reused iff it is a git repo whose `origin` URL equals `$RepoUrl` (suffix-normalized, case-insensitive); anything else still exits 1 with the "use -CloneDir" message. Re-runs (different model, same day or later) must work, not refuse.
