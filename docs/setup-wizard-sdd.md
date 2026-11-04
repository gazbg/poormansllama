# Setup wizard (install script) — design

Status: final design, 2026-11-04. Executable task plan and acceptance
contract: [plans/setup-wizard-plan.md](plans/setup-wizard-plan.md).

A committed, cross-platform bootstrap + wizard: point at the script (from
GitHub or a checkout) → it checks the environment, installs missing tooling,
clones the repository if needed, builds `llama-server` with the fork's
options, downloads the chosen model from Hugging Face if absent, then asks
(model, VRAM target, context) and emits a ready-to-run server script based
only on **measured** recipes.

## 1. Goals and non-goals

Goals (owner decisions, 2026-11-04):

- Two native scripts, one per OS: PowerShell (Windows) and bash (Linux).
  No Python requirement (owner chose B in the design review).
- Model files are auto-downloaded from Hugging Face links when missing
  (owner chose B).
- Missing toolchain: attempt auto-install (owner chose B); on failure print
  exact manual instructions and abort cleanly.
- The profile matrix is the set of measured recipes, verbatim. A
  model/VRAM/context combination with no measured recipe is refused with the
  available table printed — no extrapolated flag combinations (owner chose A).
- End state: emit `run-server.ps1` / `run-server.sh` and ask "start it now?
  (y/n)" (owner chose A).

Non-goals: multi-GPU recipes, non-CUDA backends, TLS/auth, service
installation, model upload, Windows GPU-less builds, GUI.

## 2. Files

```
setup.ps1            Windows entry point (repo root)
setup.sh             Linux entry point (repo root, executable)
docs/SETUP.md        user-facing: one-liner commands, prerequisites, recipes
                     table, model table, FAQ
tools/setup-wizard/
  smoke.ps1          internal E2E: start emitted server, /health, pong, stop
  (no other files)
```

Both entry scripts are self-contained (stdlib only: PowerShell core cmdlets,
bash + curl + standard utils). They are mirrors of each other; every
divergence is noted in the file header.

## 3. Invocation

```powershell
# Windows, from GitHub (no local clone):
iex (irm 'https://raw.githubusercontent.com/gazbg/poormansllama/claude/flint-support/setup.ps1')
```

```bash
# Linux, from GitHub:
curl -fsSL https://raw.githubusercontent.com/gazbg/poormansllama/claude/flint-support/setup.sh | bash
```

The remote one-liner fetches the script from the `claude/flint-support`
branch — the only branch that contains the flint recipes' flags
(`--lm-prefill-copy-piece`, `--hybrid-moe-prefill-free-all`); the default
branch would build a server that refuses them. Recorded decision D8: the
clone default branch is `claude/flint-support`, overridable.

If the script runs inside a checkout (a `third_party/llama.cpp` directory
next to it), the clone step is skipped and that checkout is used.

## 4. Flow (identical on both OSes)

```
 1. Banner + OS detection
 2. Environment check          (table printed: present / missing / version)
 3. Auto-install missing       (winget on Windows, apt on Linux; y/n confirm;
                                PATH re-hashed after install)
 4. Clone (if not in a repo)   (shallow by default; branch claude/flint-support)
 5. Build llama-server         (skip when the target is newer than the sources)
 6. Model                      (pick model → local path check → HF download)
 7. Recipe                     (VRAM target × context × model → measured row)
 8. Emit run-server script     (+ "start it now? (y/n)")
```

Each step is idempotent: re-running never re-clones, re-downloads, or
re-builds work that is already valid.

### 4.1 Environment check (step 2)

| Tool | Windows check | Linux check | Fail action |
|---|---|---|---|
| git ≥ 2.30 | `git --version` | `git --version` | install; required for clone (not for build) |
| CMake ≥ 4.2 | `cmake --version` | `cmake --version` | install |
| CUDA toolkit | `nvcc --version` (else search `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*`) | `nvcc --version` (else `/usr/local/cuda*`) | install; required for build |
| MSVC v143 + Windows SDK | `vswhere` (VS install dir) → `VC\Tools\MSVC\14.*\bin\Hostx64\x64\cl.exe` + `Windows SDK\...\bin` | — | install VS Build Tools 2022 (VCTools workload) |
| gcc ≥ 9 / g++ | — | `gcc --version` | install `build-essential` |
| NVIDIA GPU + driver | `nvidia-smi` | `nvidia-smi` | **fatal**: print driver-install guidance, exit 2 (no auto-install of drivers) |
| RAM ≥ 48 GiB | WMI `Win32_ComputerSystem` | `MemTotal` in `/proc/meminfo` | warning only (MoE models keep experts in host RAM) |
| Disk ≥ model size + 15 GiB | `Get-PSDrive` on model dir | `df` | confirm to proceed (y/n), default no |

Test hook (documented, mirrors the repo's `LLAMA_LM_PEER_FAIL` pattern):
`$env:SETUP_WIZARD_FORCE_MISSING='cmake,git'` / `SETUP_WIZARD_FORCE_MISSING`
makes named tools report as missing so the install path is testable without
uninstalling anything. The hook only affects the report, never the install.

### 4.2 Auto-install (step 3)

Windows (winget, all with `--accept-source-agreements --accept-package-agreements`;
IDs verified 2026-11-04 against the local winget 1.29):

| Tool | Package ID | Notes |
|---|---|---|
| git | `Git.Git` | |
| CMake | `Kitware.CMake` | |
| CUDA | `Nvidia.CUDA` | sets `CUDA_PATH`/`CUDA_PATH_V<major>_<minor>` itself |
| MSVC | `Microsoft.VisualStudio.2022.BuildTools` | override `quiet add Microsoft.VisualStudio.Workload.VCTools --includeRecommended` |

Linux (apt, needs sudo; the script asks once):

| Tool | Package | Notes |
|---|---|---|
| git | `git` | |
| CMake | `cmake` | |
| gcc/g++ | `build-essential` | |
| CUDA | `nvidia-cuda-toolkit` | after install, re-check `nvcc`; if the distro version is
  < 12, print the NVIDIA CUDA repo instructions and abort (the fork's Linux
  build was measured on CUDA 12.8; the code supports CUDA < 13, see
  docs/guide/LINUX-PASCAL.md) |

Every install attempt is confirmed with y/n first (default yes), logged, and
its failure exits 2 with the manual equivalent printed. After winget/apt
success the PATH is re-hashed (`RefreshEnv` equivalent: re-`$env:Path` from
the registry / `hash -r`) and the check re-runs; a tool still missing
(arch mismatch, silent install) aborts with the manual steps.

### 4.3 Clone (step 4)

- Default: `git clone --depth 1 --branch claude/flint-support
  https://github.com/gazbg/poormansllama.git <dest>` — the repo's `.git` is
  274 MB full; shallow keeps the one-liner fast (recorded decision D8).
- `-FullClone` / `--full-clone`: full history.
- `-Branch B` / `--branch B`: override (default `claude/flint-support`).
- `-RepoUrl U` / `--repo-url U`: override (private mirror).
- Clone target default: `C:\llama-setup\poormansllama` / `~/poormansllama`
  (overridable). Existing non-empty target directory → refuse (exit 1),
  never merge.
- Windows only, after clone: `git config core.longpaths true` (per USAGE.md;
  the repo sets it locally on every checkout).

### 4.4 Build (step 5)

Exactly the documented commands (docs/guide/USAGE.md §3, LINUX-PASCAL §1):

```
Windows:
  set CUDA_PATH / CUDA_PATH_V<maj>_<min>   (same shell, per USAGE.md)
  cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON
  cmake --build build/llama-cuda --config Release --target llama-server --parallel <cores>
Linux:
  cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=<detected> -DCMAKE_BUILD_TYPE=Release
  cmake --build build/llama-cuda --target llama-server --parallel <cores>
```

- Forward slashes in `-D` values on Windows (USAGE.md).
- `<detected>` on Linux: `nvidia-smi` compute capability → arch list, e.g.
  6.1 → `61`, 7.5 → `75`, 8.6/8.9 → `86;89`, 12.0 → `120`. Unknown → refuse
  with the flag explained (recorded decision D9: do not guess archs).
- Skip condition (recorded decision D10): the output binary
  (`build/llama-cuda/bin/Release/llama-server.exe` or
  `build/llama-cuda/bin/llama-server`) is newer than every source file under
  `third_party/llama.cpp/{src,ggml,common,tools/llama-server}` (recursive
  mtime scan). Skip → step 6. The scan is the only freshness signal; there is
  no cache of source hashes (recorded: a digest would be a cross-process
  comparison, and this repo has measured such comparisons to be unreliable —
  AGENTS.md "A digest compared across wrapper invocations can differ for an
  UNCHANGED tree").
- `--no-build` / `-SkipBuild`: emit the build commands and stop (dry build).
- Build failure → exit 4 with the last 50 lines of the build log
  (`build/llama-cuda/setup-wizard-build.log`).

### 4.5 Model (step 6)

Model table (all links verified 2026-11-04 against the HF API; sizes are
`Content-Length` of the resolve endpoint):

| Key | Kind | HF file | Size |
|---|---|---|---|
| `qwen38` | dense hybrid | `unsloth/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-UD-Q4_K_M.gguf` | 15.4 GiB |
| `qwen36` | MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 20.6 GiB |
| `flint` | SWA MoE | `unsloth/flint-GGUF` / `flint-UD-Q5_K_XL.gguf` | 27.9 GiB |

Recorded decision D7: the owner's dense profile used
`Qwen3.8-27B-Q4_K_M.gguf`, which does **not** exist on HF under that name;
`Qwen3.8-27B-UD-Q4_K_M.gguf` is the closest available quant. The host-layer
plan depends only on layer count, but per-layer byte sizes differ slightly —
flagged as a known risk in docs/SETUP.md (unmeasured substitution).

Questions and rules:

- Model pick (1/2/3) with kind + file + size printed.
- Local path: default per key (`C:/Models/ALL/other/qwen-3.8|qwen-3.6|flint`
  on Windows, `~/models/...` on Linux; the owner's existing layout). Any
  existing `*.gguf` in the directory is offered (1-based list + "type a path"
  + "download").
- Download when missing (owner decision): resume-capable — Windows
  `Invoke-WebRequest -Resume` (falling back to full download if the server
  416s), Linux `curl -fL -C - --retry 3 --retry-delay 5 -o
  <name>.part` (`-C -` is the portable resume flag; `--continue` is not
  accepted by all curl builds, measured 2026-11-03 on curl 8.18.0). The resolved file is renamed from `.part` only after a
  complete, non-zero-length transfer. A second file of identical size is
  verified byte-identical before rename (sha-256); a size-mismatched file
  starts fresh.
- Optional `--hf-token` (env `HF_TOKEN` also honored): sent as
  `Authorization: Bearer` for gated repos; never printed.
- Download failure → exit 5 with the exact URL and a manual fallback line.

### 4.6 Recipe (step 7)

Every row is **verbatim** from its source command. The wizard substitutes
only: the model path (question 6), the bind address (default = the row's own
value: 127.0.0.1 for R1-R3/R6/R7, 0.0.0.0 for R4/R5), and the port (default
8033, all rows). When the resolved host is 0.0.0.0 the wizard prints the
USAGE.md §5 security note once. Recorded decision D11: the wizard does not
add, drop, or re-tune any flag of a row (`-t 16` appears only where the row
has it).

VRAM targets (question): `full` (use all card VRAM), `16g` (keep ~8 GB
class), `8g` (keep ~16 GB class on a 24 GB card). Context: `262k` / `128k`.

Measured matrix (rows are verbatim from the owner's 2026-10-29/11-02
commands and docs/guide; tok/s as measured in those runs):

| # | Model | Target | Ctx | KV | Distinctive flags | Measured |
|---|---|---|---|---|---|---|
| R1 | qwen38 | full | 262k | q8_0 | `--lm-prefill-host-layers 0-37 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0` | PP 1093 t/s, TG 14.95 t/s |
| R2 | qwen38 | 16g | 262k | q8_0 | as R1 with `--lm-prefill-host-layers 0-63` | PP 1008 t/s, TG 9.96 t/s |
| R3 | qwen38 | 8g | 128k | q4_0 | `--lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --lm-prefill-prearm always` (no spec flags) | TG 9.92 t/s |
| R4 | flint | full | 262k | q8_0 | `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 --lm-prefill-copy-piece 64` | PP 2423 t/s, TG 53.8 t/s |
| R5 | qwen36 | full | 262k | q8_0 | `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 --device CUDA0 --hybrid-moe-prefill-free-all` | (owner's working recipe; tok/s not recorded in the paste — printed as "owner recipe, unmeasured in wizard records") |
| R6 | qwen36 | 16g | 262k | q8_0 | G42 canonical (USAGE.md §4): `--hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32` | TG 3.0–3.7x stock, PP 3.65–4.11x stock (G31 record) |
| R7 | qwen36 | 8g | 262k | q8_0 | LINUX-PASCAL §2 recipe: `-dev CUDA0 -sm none -mg 0 --hybrid-moe --lm-prefill --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 --ctx-checkpoints 32` (no `-fitt`), plus env `LLAMA_FATTN_TILE_Q8QK=1` on pre-Volta | TG 32–34 t/s short / ~26.5 t/s @100k; PP 670 t/s @19k (GTX 1070) |

Refusal rule (owner decision A): any (model, target, ctx) triple not in the
matrix → the wizard prints the table of available rows for that model and
exits 3. Examples refused: qwen36 128k, flint 16g/8g, qwen38 16g+128k.

GPU sanity check (warning only, recorded decision D12): if `nvidia-smi`
total VRAM < the target implies (full ≥ 22 GiB / 16g ≥ 10 GiB / 8g ≥ 8 GiB
for these models) the row is still emitted with a warning that the recipe
was measured on a 24 GiB card (R7 excepted, measured on 8 GiB). No refusal:
`-fitt` and the memory guard fail loudly in the server itself, and the
owner explicitly wants the 16g/8g rows usable on smaller cards.

### 4.7 Emitted artifacts (step 8)

Written into the repo root (both OSes write both names only if the OS
matches; one script per OS):

`run-server.ps1` (Windows) / `run-server.sh` (Linux):

- Header comment: date, recipe row id (R1..R7), model path, measured
  numbers, the wizard version (git short hash or "uncommitted").
- The exact command (one per line, quoted paths).
- Windows: `Start-Process` not used — plain foreground invocation; the
  server writes to the console and to `build/llama-cuda/server-<row>.log`
  (redirection), so a dead server leaves a readable tail.
- A trailing comment block with: the `/health` curl one-liner, the pong
  chat request (USAGE.md §5, port 8033), and the stop command (Ctrl-C).
- `--no-launch` / `-NoStart` skips the "start it now?" prompt entirely.

The emitted script is executable standalone (no wizard needed to re-run it),
and re-running the wizard overwrites it (the previous one is moved to
`run-server.last.ps1` / `.sh`, one generation of history).

## 5. Exit codes (both OSes)

| Code | Meaning |
|---|---|
| 0 | success (or user answered "no" to start-now at the end) |
| 1 | usage error / non-empty clone target / unexpected state |
| 2 | environment: a required tool is still missing after the install attempt, or no NVIDIA GPU/driver |
| 3 | no measured recipe for the chosen combination (table printed) |
| 4 | build failure (last 50 log lines printed) |
| 5 | model download failure (URL + manual fallback printed) |
| 6 | user abort (Ctrl-C / "n" on a required confirm) |

## 6. Safety

- Downloads only from `huggingface.co` (the table in §4.5) and the
  `raw.githubusercontent.com` clone URL; the token, if given, is never
  printed and is not written to any file.
- The wizard never touches files outside: the clone target, the build tree,
  the model directory, and the repo root (`run-server.*` files).
- No service creation, no firewall changes, no driver installation
  (drivers are a fatal-with-guidance case, §4.1).
- Auto-install runs only behind a y/n confirm and only via winget/apt with
  the package IDs in §4.2.

## 7. Testing

See the acceptance contract in plans/setup-wizard-plan.md. In short:

1. Static: `pwsh -NoProfile -Command "[scriptblock]::Create((Get-Content setup.ps1 -Raw))"`
   (or `powershell -File` with a parse-only switch) and `bash -n setup.sh`
   (+ `shellcheck` if present).
2. Non-GPU E2E: `setup.ps1 -DryRun --non-interactive --model qwen36
   --vram full` on this machine → clone skipped (in repo), build skipped
   (binary fresh), model found at the local path, R5 selected, emitted
   script byte-stable across two runs (idempotence).
3. Refusal: `--model qwen36 --vram full --context 128k` → exit 3 + table.
4. Install path: `SETUP_WIZARD_FORCE_MISSING=cmake` + `-DryRun` → plan
   printed, nothing installed.
5. Fresh-clone path: `setup.ps1 -CloneDir <temp>` into an empty dir →
   shallow clone of the branch, configure + build run, server binary
   present. (GPU-mutex not needed: no CUDA init in cmake/build; the build
   is CPU work. Recorded: `nvcc` does not acquire the GPU.)
6. E2E server smoke (GPU mutex via `tools/with-gpu-lock.ps1`): start the
   emitted `run-server.ps1` (R5 recipe), `/health` ok within 300 s, pong
   chat request returns exactly `pong`, clean stop. Harness:
   `tools/setup-wizard/smoke.ps1`.
7. Linux: in WSL (`Ubuntu-24.04`): `bash -n setup.sh`, then
   `bash setup.sh --dry-run --non-interactive --model qwen36 --vram full`
   → clean abort at the CUDA check with the apt plan (WSL has no CUDA
   toolkit by default) or full dry-run success if one is present. A WSL
   build+smoke is best-effort and reported as such, never as a Windows
   substitute.

## 8. Documentation updates

- `docs/SETUP.md` (new): one-liner, prerequisites table, model table,
  recipe matrix with measured numbers and their sources, refusal behavior,
  D7 risk note, FAQ (moved-repo CMakeCache, `-fitt` meaning, port in use).
- `README.md`: "Quick start" section pointing at docs/SETUP.md and the two
  one-liners.
- `docs/guide/USAGE.md`: one-line pointer to docs/SETUP.md under §1.
- `docs/reference/` index: register docs/SETUP.md if the index lists guide
  docs.
- `.gitignore`: nothing new (build tree, run-server logs, models already
  excluded or outside the repo; `run-server.*.last.*` is committed-OK but
  git-ignored).

## 9. Decision log (this design)

| # | Decision | Why | Reversible |
|---|---|---|---|
| D1 | Two native scripts, mirrored logic | owner choice; zero Python dependency | yes (rewrite) |
| D2 | HF auto-download with resume + sha-256 verify | owner choice; 15–28 GiB files need resume | yes |
| D3 | Auto-install via winget/apt behind y/n, then re-check; fatal+guide on failure | owner choice; drivers excluded | yes |
| D4 | Recipe matrix = 7 measured rows, verbatim; unmeasured combos refused with table | owner choice (A); repo evidence culture | yes (add rows as they get measured) |
| D5 | Emit run-server script + "start now?" | owner choice (A) | yes |
| D6 | Emitted script keeps a one-generation `.last` backup | cheap safety against overwriting a tweaked command | yes |
| D7 | qwen38 default file = `Qwen3.8-27B-UD-Q4_K_M.gguf` (profile's exact name not on HF) | only available Q4_K_M-class quant for the dense model | yes (add alternate file to table) |
| D8 | Clone: `--depth 1`, branch `claude/flint-support` by default; `.git` is 274 MB full | speed of the one-liner; flint flags exist only on that branch | yes (`-FullClone`, `-Branch`) |
| D9 | Linux: detect CUDA arch from `nvidia-smi`, refuse unknown | LINUX-PASCAL requires an explicit arch list | yes |
| D10 | Build skip iff binary newer than all scanned sources (no digest) | digest across processes measured unreliable in this repo | yes |
| D11 | Rows emitted verbatim; wizard substitutes only model path, bind (default per row), port | owner's recipes are measured evidence; re-tuning is a new measurement | yes |
| D12 | Small-VRAM rows warn, never refuse | `-fitt`/memory guard fail loudly in-server; owner wants 16g/8g rows usable on smaller cards | yes |
| D13 | R6 model substitution: G42 row uses `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (the MTP file the G42 record used is not on HF) | auto-download requirement (owner); closest available quant; risk flagged in SETUP.md | yes |
| D14 | Exit codes 0/1/2/3/4/5/6 as in §5 | machine-readable outcomes for automation | yes |
| D15 | `SETUP_WIZARD_FORCE_MISSING` test hook | mirrors the repo's existing fault-injection test-hook pattern (`LLAMA_LM_PEER_FAIL`) | yes |
