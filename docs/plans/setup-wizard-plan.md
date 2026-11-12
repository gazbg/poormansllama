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

### T-SW-7: settings-first flow + parallel build/download (2026-11-06)

- Status: in-progress (2026-11-06, owner request: "the user 1st chooses all
  settings and then at the end do the compiling and the model download in
  parallel"). Design: SDD D20 + SDD §4.8. Files: `setup.ps1`, `setup.sh`,
  `docs/setup-wizard-sdd.md` (flow §4, §4.8, D20), `docs/SETUP.md`,
  `docs/currentstate.md`. **2026-11-07 validation state:** S7-1..S7-6 all
  demonstrated on the LINUX twin over ssh-mcp (gogo-big, the 2x GTX 1070
  box) plus the real capped-download parallel run, the resume run and a
  fail-fast run (evidence below); three further defects found and fixed
  while validating (D61 cmd leading-`&` bat trap on Windows, D62 `set -u`
  unbound rc refs + EXIT-trap orphan curl on sh, D63 curl meter to log).
  Remaining: re-run the Windows real-parallel leg (S7-5) on the owner's
  machine once the other llama.cpp workload leaves it (the owner asked for
  Linux-only testing in the meantime); commit is pending the owner's
  instruction (working-tree change so far).
- Goal: all questions (model + quant, local path, disk check, VRAM target,
  context) are asked BEFORE any heavy work, so the unmeasured-combo refusal
  (exit 3) can no longer follow a paid build or a started download. The build
  and the model download then run in parallel as two legs with one combined
  status line; first hard failure kills the other leg (fail-fast). Exit-code
  table and the emitted script are unchanged.
- Non-goals: no new settings questions (device stays a flag, `start now?`
  stays the final action); no change to the recipe rows, the model table,
  the download protocol (resume/Range/sha-256), or the build commands.
- Acceptance (host-side; written before the runs):
  - **S7-1 static.** PS parse (`[scriptblock]::Create`) + `bash -n` +
    `wsl bash -n` all clean.
  - **S7-2 order + idempotence.** In-repo `-DryRun --non-interactive --model
    qwen36 --vram full` → exit 0; the `Settings` header is printed BEFORE any
    `(dry-run: cmake …)` / download line; two consecutive runs print
    byte-identical output.
  - **S7-3 refusal-first.** `--model qwen36 --vram full --context 128k` →
    exit 3, the R5/R6/R7 table printed, and NO build command line and NO
    download line anywhere in the output.
  - **S7-4 install path unchanged.** `SETUP_WIZARD_FORCE_MISSING=cmake` +
    `-DryRun --non-interactive` → exit 0 with the winget plan and
    `(dry-run: nothing installed)`.
  - **S7-5 real parallel leg (the load-bearing one).** Fresh clone + full
    build + a real capped download (`--max-download-mib 2048`) in ONE
    invocation: `-CloneDir <temp> -Model qwen36 --vram 16g --context 262k
    --max-download-mib 2048 -ModelDir <temp> --non-interactive -NoStart` →
    the combined `build: … | model: …` line is visible in the output, the
    real build completes (`[ok] built …llama-server.exe`, build log tail
    clean, err log empty), the 2048 MiB `.part` stop fires with its warning,
    and the run exits with the D33 test-hook semantics (`.part` kept, no
    rename). Fresh-clone caveats (D37/D18): the branch is unpublished →
    fallback to the remote default branch (loud warning); `--vram 16g`
    selects R6, whose flags are all in origin/main, so the fork-only flag
    gate passes on the fresh binary (verified against its `--help`).
  - **S7-6 sh twin.** `bash -n` + WSL `--dry-run --non-interactive` full
    flow → exit 0 (settings header before the dry cmake lines).
- Escalation: a hang of the `.bat` legs (rc file missing > 30 min while the
  process is alive) → capture logs, diagnose per the D49/D50 traps before
  any design change.

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
- **AC-6-L Linux E2E smoke — CONTRACT (pre-run, 2026-11-04, Linux GTX 1070 machine).** Different machine from the Windows AC-6: Ubuntu 22.04, 2x GTX 1070 8 GiB, i7-4770K (8 threads), 30 GB RAM, g++ 11.4, CUDA 12.8, CMake 4.4.4, bash 5.1; this repo's convention on this machine is no GPU mutex / no provenance gate (see the LINUX-PASCAL / dense entries in `docs/currentstate.md`). Goal: prove the published wizard's server-start leg on Linux against the published `origin/main` source (PR #12 merged; main checkout at `0fc0a52`). R7 is the row measured on exactly this hardware (LINUX-PASCAL.md) and its flags are all in the published main (verified by source grep: hybrid-moe, lm-prefill-fuse/prearm/segment, ctx-checkpoints, hybrid-moe-prefill-ubatch).
  - Run 1: `./setup.sh --non-interactive --yes-install --model qwen36 --vram 8g --context 262k --clone-dir /home/gzhel/Development/wiz-ac6-clone --model-file /home/gzhel/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --no-start` → expected: env check OK (nothing to install) → LOUD fallback warning (claude/flint-support unpublished) → shallow clone of the default branch → configure + build (sm_61) → flag gate passes R7 → emits `run-server.sh` (R7, bind 127.0.0.1, port 8033, `export LLAMA_FATTN_TILE_Q8QK=1`) → exit 0.
  - Run 2 (identical args): reuses the clone, skips the build (binary newer than sources), emits a BYTE-IDENTICAL `run-server.sh` (run 1's copy preserved as `run-server.last.sh`), exit 0.
  - Smoke: `bash run-server.sh` in the clone (background); poll `http://127.0.0.1:8033/health` until `{"status":"ok"` (timeout 600 s); the verbatim USAGE.md §5 pong request (model `qwen3.6-35b-a3b`); require `pong` in the response content; stop the server; `nvidia-smi` shows no residual compute process.
  - Falsifiable criteria: (1) both runs exit 0; (2) fallback warning present in the wizard log (not silent); (3) emitted flags equal the R7 row verbatim with only model path / bind / port substituted, plus the R7 env export; (4) run 2 output byte-identical to run 1's script; (5) health ok + `pong` in content; (6) no `CUDA error` / crash / assertion in `build/llama-cuda/server-R7.log`.
  - Claim scope: certifies the server-start leg on LINUX with a PUBLISHED-`main` build for R7 only. Does NOT certify the Windows AC-6 (still open there), speed or quality, a real apt install (dry-run only, as contracted), or the download leg (local file used). Provenance note: the local model file is 22,663,387,424 B vs the Windows table file 22,134,528,992 B (D36) — a different UD build of the same quant name; acceptable for this claim, recorded here so the size mismatch is not rediscovered.
- **AC-6-L Linux E2E smoke — RESULTS (2026-11-04, Linux GTX 1070 machine; agent-executed under the owner's "continue the work" instruction, owner review open).** Two `setup.sh` defects were found by the run itself and fixed before the pass (D59, D60); the passing runs used the FIXED script, the pre-fix behavior is the reproduction below.
  - **Repro (pre-fix run 1a).** `git clone --depth 1 --branch claude/flint-support` fails (branch unpublished) → `git … 2>&1 | sed` reported sed's rc (the script has `set -u` but no `set -o pipefail`) → the fallback block was DEAD CODE: no warning, unconditional `[ok] cloned` → cmake configure died on the empty tree (`source directory …/third_party/llama.cpp does not exist`), `[FAIL] cmake configure failed`, exit 4. The AC-7 dry-run could not catch it: `--dry-run` short-circuits before any real clone.
  - **D59 fix.** Capture clone output, test `$?` directly; plus a loud completeness guard (`third_party/llama.cpp` must exist after a clone or fail 1) so an incomplete tree never reaches the build step.
  - **Post-fix run 1.** `[warn] clone of branch 'claude/flint-support' failed (not on the remote?); falling back to the remote default branch` → clone of the remote default (HEAD `0fc0a52` = published main) → configure + build (detected CUDA arch 61, ~16 min on 8 threads, warnings only, `Built target llama-server`) → flag gate `[ok] binary supports all R7 row flags` → `[ok] wrote …/run-server.sh` → `wizard finished`. (Exit code not captured — background launch with no rc file; evidence is the complete success path with zero `[FAIL]` lines. Run 2 below is the measured exit code.)
  - **Repro 2 (pre-fix run 2).** Identical re-run → `[FAIL] clone target … exists and is not empty (use --clone-dir to pick another)`, **exit 1**: the bash twin had NO clone reuse at all — D58 (reuse) exists only in `setup.ps1`, so a "change model" re-run was impossible on Linux.
  - **D60 fix.** D58 ported to bash with the same validity rule (`.git` present + `git remote get-url origin` suffix-normalized equal to the repo URL).
  - **Post-fix run 2 (foreground, measured).** `RUN2-EXIT=0`; `[ok] reusing existing checkout` → `[ok] build skipped (binary newer than sources)` → `[ok] wrote run-server.sh` → `wizard finished`. Emission BYTE-IDENTICAL to run 1's script: md5 `34cc3351258c41cda81873424d16a570` on both (`run-server.sh` vs `run-server.last.sh`); the one-generation backup preserved run 1's copy.
  - **Verbatim (criterion 3, programmatic).** Python diff of the emitted flag string against the R7 table row: `IDENTICAL: True` (26 tokens, same order); only model path / bind `127.0.0.1` / port `8033` substituted; `export LLAMA_FATTN_TILE_Q8QK=1` present (R7 env).
  - **Smoke (criteria 5+6).** GPU before: both cards idle (7 MiB, no compute process). `bash run-server.sh` → **health `{"status":"ok"}` after 35 s** → verbatim USAGE.md §5 pong request → response `"content":"pong"` (19 prompt / 2 generated tokens — the same numbers USAGE.md §5 recorded for the Windows run), `system_fingerprint b1-0fc0a52` (published-main build), prompt 27.4 tok/s, predicted 19.7 tok/s. Server stopped; **GPU after: no residual compute process**. Server log `build/llama-cuda/server-R7.log` (62 lines): **zero** `CUDA error` / `assert` / `abort` / segfault matches; the R7 recipe is visibly active (`hybrid-moe: expert banks on the host, VRAM expert cache`, `lm-prefill: fused prompt batches … idle preparation … auto`). The `lean request … took 3.0-3.4 ms` WARN lines are the lean-team threshold diagnostic (default 3.0 ms) at its edge on this i7-4770K — informational, expected on this CPU, not errors. Verdict file `/tmp/wiz-ac6-verdict.txt` (`SMOKE-HEALTH=1 SMOKE-PONG=1`); wizard logs `/tmp/wiz-ac6-run1.log`, `/tmp/wiz-ac6-run2.log`; clone kept at `/home/gzhel/Development/wiz-ac6-clone` (the working emitted server is `bash run-server.sh` there; delete the clone to clean up).
  - **All six contract criteria hold** (run-1 exit code log-based as noted; every other datum measured above).
  - **Landed** (owner instruction, 2026-11-04): commit `b97f1aa` on branch `claude/setup-wizard-linux-fixes` (setup.sh fixes D59/D60 + this record), PR #13 against `main`; commit identity `GAZ <47818403+gazbg@users.noreply.github.com>` (repo-local config, per the 2026-10-19 history rewrite).
- **AC-7 WSL/Linux** — `wsl.exe -e bash -n /mnt/c/…/setup.sh` → `WSL-SYNTAX-OK` (bash 5.3.9); full Linux dry-run `wsl.exe -e bash setup.sh --dry-run --non-interactive --model qwen36 --vram full --context 262k --skip-model --no-start --clone-dir /tmp/swz-wsl` → complete flow (env check, dry-run plan, R5 recipe, emitted-script preview verbatim, `wizard finished`), **exit 0** (`WSL-DRYRUN-EXIT=0`). No-driver path (Git Bash, no nvidia-smi) → CUDA-arch guidance + exit 2 (earlier session). Note: Git Bash mangles `/mnt/c/…` paths for wsl.exe; use `MSYS_NO_PATHCONV=1`.
- **AC-8 docs + registration** — `docs/SETUP.md` written; README quick-start + table row; USAGE.md pointer block; `.gitignore` gains `/run-server.ps1`, `/run-server.sh`, `/run-server.last.ps1`, `/run-server.last.sh` (server logs land under `build/`, already ignored).
- **Flag-capability gate (D38)** — `Test-RowFlags` (PS) / `check_row_flags` (sh) run `<binary> --help` and require every `--flag` of the chosen row; verified against the in-repo binary: `binary supports all R5 row flags`. Fresh-clone consequence (recorded): a `main`-based build supports R1-R3/R6/R7 fully; R4 (`--lm-prefill-copy-piece`) and R5 (`--hybrid-moe-prefill-free-all`) exist only in uncommitted working-tree changes, so a fresh-clone build refuses exactly those two rows with exit 1 and a flag list. VERIFIED against the actual fresh-clone binary (2026-11-04): its `--help` (978 lines) contains `--hybrid-moe`, `--lm-prefill*`, and NOT `prefill-free-all`, NOT `lm-prefill-copy-piece` — exactly the origin/main flag set; R4/R5 emission is therefore refused on a fresh clone, R1-R3/R6/R7 pass.
- **Clone reuse (D58)** — measured 2026-11-04: a re-run on an existing target previously refused (`clone target … exists and is not empty`); a valid existing checkout (`.git` + `git remote get-url origin` == `$RepoUrl`) is now reused. First attempt keyed on `setup.ps1` presence and failed on a fresh clone (setup.ps1 is uncommitted/local-only); the origin-URL check is the robust test.
- **S7-1 static** (2026-11-06/07) — `PS-PARSE-OK` (re-verified 2026-11-07 after all edits), `bash -n` clean locally AND on gogo-big after every remote patch (`SH-PARSE-OK`), remote `setup.sh` md5 `eae53e9ff505ba116a3f2902e13b625f` byte-identical to the local file in the final state.
- **S7-2 order + idempotence (PS)** (2026-11-06) — in-repo `-DryRun --non-interactive --model qwen36 --vram 16g --context 262k` → **exit 0**; the `== Settings (asked before the build and the download) ==` header precedes every `(dry-run: cmake …)` line; two consecutive runs byte-identical (AC-2 style re-check). The emitted `run-server.ps1` carries `# Model: C:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (local file found in the settings phase) and the full R6 flag line.
- **S7-3 refusal-first (sh)** (2026-11-06, gogo-big) — `bash setup.sh --non-interactive --model qwen36 --vram full --context 128k` → the R5/R6/R7 table printed under `== Settings … ==`, `[FAIL] no measured recipe for qwen36 / full / 128k`, **EXIT=3**, and ZERO build/cmake/download lines in the output (grep-verified).
- **S7-4 install path (PS)** (2026-11-06) — `SETUP_WIZARD_FORCE_MISSING=cmake` + `-DryRun --non-interactive` → winget plan + `(dry-run: nothing installed)`, exit 0 (re-run after the restructure).
- **S7-5 real parallel leg — LINUX twin** (2026-11-06, gogo-big; the Windows S7-5 is PENDING machine contention, see D61 and the Status above). The contract's load-bearing scenario, executed on the Linux twin in the existing clone `/home/gzhel/Development/wiz-ac6-clone` (fresh-clone variant already proven on this box by AC-5/AC-6-L): `--non-interactive --model qwen36 --vram 16g --context 262k --model-dir /home/gzhel/models/d20-test --max-download-mib 2048 --full-build --no-start` → both legs in ONE invocation; combined line visible every tick (`build: configuring/building … N min   |   model: M MiB downloaded`); real 8-thread build to completion (~4 min, `Built target llama-server`); the 2048 MiB cap fired with its `[warn] test hook stopped transfer at 2048 MiB (expected full 20.6 GiB)` and `[ok] leaving .part in place to verify resume; NOT renaming`; flag gate `[ok] binary supports all R6 row flags` against the FRESH binary; `run-server.sh` emitted (with the 2048 MiB `.part` as MODEL — recorded deviation, see below); **EXIT=0**. Download speed observed ~90 MiB/s against the real HF URL (the Windows box showed a throttled ~32 KiB/s in the same run shape — box/network specific, not script).
- **S7-6 sh twin dry-run** (2026-11-06, gogo-big) — `--dry-run --non-interactive --model qwen36 --vram 16g --context 262k --model-dir <fresh> --no-start` → **EXIT=0**; Settings header before the dry cmake lines; emitted preview carries `MODEL="/home/gzhel/models/d20-test/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"` (the D20 dry-run fix: download-target recorded so the script is complete).
- **S7-resume** (2026-11-06, gogo-big) — with a 4 GiB `.part` in place and `--max-download-mib 4300`: `[ok] resuming: 4096 MiB already in .part`, download continues 4096 → 4362 MiB, cap truncates to EXACTLY 4,508,876,800 B = 4300 MiB, **EXIT=0** (interactive `y` past the disk check — see S7-disk). This run also caught the D62 `set -u` defect (pre-fix it died on the first tick and orphaned the curl, which then ran the full 20.6 GiB to completion unprompted — the orphan's file is the size-verified model now in `~/models/qwen-3.6/`).
- **S7-fail-fast** (2026-11-06, gogo-big) — fake `cmake` shim on PATH (valid `--version`, exit 3 at configure) + `--full-build` + fresh model dir: parallel phase started both legs, configure rc=3 detected → `[FAIL] cmake configure failed (exit 3; full log: …)`, **EXIT=4**, the download curl KILLED (no leftover process, no .part data file), and the fake cmake's invocation log shows BOTH configure and build attempts (the build stage ran because the subshell continues — the wizard's fail-fast then waited for and reported the configure failure; recorded: the rc files are checked in the poll, configure first). Note: the fake shim initially failed the ENV CHECK (`cmake TOO-OLD` → install path → exit 2), which is itself correct wizard behavior; the refined shim answers `--version`.
- **S7-disk (bonus)** (2026-11-06, gogo-big) — the settings-phase disk check (moved there by D20) is live on the 224 GiB disk with ~20 GiB free: `[warn] disk has 19 GiB free, the model plus build wants ~36 GiB` → non-interactive: `[FAIL] aborted: not enough disk space`, **EXIT=6** BEFORE the parallel phase; interactive: `proceed anyway? [Y/n]` and `y` proceeds (used by S7-resume/S7-fail-fast).

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
- **D59** (2026-11-04, Linux AC-6-L) `setup.sh` clone bug class: the script runs under `set -u` WITHOUT `set -o pipefail`, so `if ! git clone … 2>&1 | sed …` tests sed's rc — a failed clone printed `[ok]` and the fallback block was dead code (measured: empty tree reached cmake configure). Rules: in `setup.sh`, a native command's exit status must come from a captured `$?` (or `PIPESTATUS[0]`, as the cmake steps already do), never from a trailing filter; and a clone must be verified COMPLETE (`third_party/llama.cpp` exists) before the build step — an incomplete tree fails loudly, it does not build.
- **D60** (2026-11-04, Linux AC-6-L) `setup.sh` had NO clone reuse (D58 existed only in `setup.ps1`): a non-empty `--clone-dir` always refused, so a same-day "change model" re-run was impossible on Linux (measured exit 1). D58 is now ported to bash with the identical validity rule. The bash/PS twins must stay behaviorally paired — when one gains a flow (reuse, fallback, guard), the other must be checked for it; AC-7's dry-runs cannot catch clone-path gaps because `--dry-run` short-circuits before real clones.
- **D61** (2026-11-06, Windows, found by the first real S7-5 attempt) a
  `.bat` command line that STARTS with `&` (as in `& cmake … > log 2> err`) is
  a syntax error for cmd - it prints `& was unexpected at this time.` and the
  whole batch aborts BEFORE the `%ERRORLEVEL%` line runs, so no log and no rc
  file ever appear: the parallel build legs sat at "configuring" forever with
  no processes and no output. The proven `Invoke-NativeCmd` bat never had a
  leading `&` (that shape was a misremembering). `Start-BuildLeg`'s bats now
  write `<exe> <args> > log 2> err` with cmd-style exe quoting (same convention
  as `Invoke-NativeCmd`). Verified: the fixed bat shape runs via
  `Start-Process -WindowStyle Hidden` and reports a real rc (cmake failing on
  an empty fake repo wrote rc=1).
- **D62** (2026-11-06, Linux, found by the resume run) `poll_download` read
  `$BUILD_RC1`/`$BUILD_RC2` unguarded; on the download-only path (build fresh
  or `-SkipBuild`) those were never set, and `set -u` DIED the wizard mid-
  download (first tick), leaving the background curl ORPHANED: it then ran for
  ~97 s and finished the whole 20.6 GiB model file unprompted (the .part ended
  at exactly the full size). Fixed: rc refs go through `${BUILD_RC1:-}` guards
  (empty on the download-only path, where the status line now says
  `not running (fresh or -SkipBuild)`), plus an EXIT trap that kills a still-
  running download curl so no future internal death can orphan a multi-GB
  transfer. The rescued 20.6 GiB file was size-verified and moved to
  `~/models/qwen-3.6/` on gogo-big (kept, not deleted).
- **D63** (2026-11-06, Linux, measured in the d20-parallel run) curl's own
  progress meter (stderr) interleaved with the combined status line on the
  console. The background curl now redirects stderr to `<part>.curl-err.log`;
  on a download failure the wizard tails that log before the exit-5 message.

## Post-acceptance change (2026-11-05, owner instruction)

The owner replaced the 8g default recipe (R7) and required a dynamic CUDA
device index. Changes land in `setup.ps1` / `setup.sh` and are recorded as
SDD decisions D16-D18 (docs/setup-wizard-sdd.md):

- **R7** (qwen36 / 8g / 262k) now emits the owner's 2026-11-05 command:
  `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 32 -b 32768
  -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0
  --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all
  --lm-prefill-segment 32768 -sm none -mg 0`, bind 0.0.0.0, env
  `LLAMA_FATTN_TILE_Q8QK=1` kept. The old LINUX-PASCAL §2 fuse/prearm
  recipe stays in docs/guide/LINUX-PASCAL.md as historical evidence.
- **D17**: R5 and R7 no longer hard-code `CUDA0`; the wizard appends
  `--device CUDA<n>` at emission (`-Device N` / `--device N` overrides;
  auto = largest card for `full`, smallest at/above the row minimum for
  `16g`/`8g`). The AC-6-L "flags equal the R7 row verbatim" criterion was
  measured against the old row and is historical; the substitution set
  (model path, bind, port) is unchanged except the device index.
- **D18**: because R7 now carries `--hybrid-moe-prefill-free-all`
  (fork-only), a fresh-clone (origin/main) build's flag gate refuses R7
  like R4/R5. The AC-6-L smoke (published-main build, R7) does NOT apply
  to the new row; it certifies the old recipe only.

### SDD D19 addendum (2026-11-05, owner request: "add choice for 5k_xl models")

Owner asked for a Q5_K_XL build choice for qwen38 and qwen36 (flint was
already Q5_K_XL). Recorded as **SDD D19** in `docs/setup-wizard-sdd.md`
(the SDD's own decision series; NOT this plan's D19 "emitted header" —
different numbering spaces, disambiguated by the "SDD" prefix).

- The model table in `setup.ps1` / `setup.sh` is now **flat over
  (model, quant)** with 5 entries: (1) qwen38 Q4_K_M [default], (2) qwen36
  Q4_K_M [default], (3) flint Q5_K_XL [only], (4) qwen38 Q5_K_XL [alt],
  (5) qwen36 Q5_K_XL [alt]. A new `-Quant` / `--quant q4k_m|q5k_xl` option
  selects the build; numeric picks 1-5 cover the flat list.
- **Recipe rows stay keyed on MODEL only** (R1-R7 unchanged): the alt-quant
  rows run the model's measured row's flags, and picking a non-default
  quant prints a warning that the row's speed/VRAM numbers are for the
  default (measured) build. This is an [assumption]-style disclosure, not
  a new measured row (SDD D19; the wizard only *emits* measured rows, it
  does not fabricate numbers for the alt build).
- Two defects found and fixed while validating (both turn silent
  misconfiguration into loud failure):
  - (a) `setup.ps1` silently **DROPS** hyphenated dashed flags
    (`--dry-run`, `--non-interactive`, `--skip-model` land unbound in
    `$args` and PowerShell discards them) — a `--dry-run` invocation
    actually downloaded the 24.7 GiB model. `setup.ps1` now normalizes the
    known dashed names up front and refuses anything unrecognized (exit 1).
  - (b) `setup.sh`'s `resolve_gpu_index` cached its result in a global that
    dies with the `$(...)` subshell and its cache read died under `set -u`
    (emitted `--device CUDA ` empty). It is now a **pure function** echoing
    the index; the `--device N` option still wins.
- **Validated (host-only, Windows):** PS parse + the `mi()` 14-case and
  `ask_choice` 7-case harnesses; `bash -n` + the 7-case `ask_choice`
  harness; E2E dry-runs for (qwen36/8g → R7 + `--device CUDA0`),
  (qwen36/8g/q5k_xl → R7 + Q5_K_XL file + alt-quant note), (flint/q4k_m →
  loud refusal), (qwen38/full/q5k_xl → R1 + alt-quant note). No GPU run, no
  tok/s claim for the alt-quant rows.
- **Linux E2E** (`./setup.sh --dry-run --non-interactive --model qwen36
  --vram 8g --context 262k --skip-model --no-start` on the 1070 box) is the
  remaining live-hardware confirmation of the D17 auto-device selection on a
  real 2-GPU system; run via ssh-mcp when the box is reachable.

## Post-acceptance change (2026-11-09, VOD-6 of docs/plans/vision-on-demand-plan.md): vision support

SDD D26: the Qwen3.6 table entries list a projector from the same Hugging Face repo; the settings phase asks "Vision support?" (default yes); the projector is downloaded with the model and the emitted command carries `--mmproj <var> --mmproj-on-demand`. Not offered for the MTP rows, the dense rows and flint. Evidence: `tools/setup-wizard/test-vod.ps1` (125 checks) and `test-vod.sh` (123 checks), host-only, plus the unchanged D22 tests (53 / 195 checks); dry runs on Windows and under Git Bash. No download was run (the projector files already exist locally for the model tests; the download path is exercised by the dry-run announcement and the existing resume logic).
