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
 5. Settings (D20)             (ALL questions up front: model + quant → local
                                path → disk check → VRAM target × context →
                                the unmeasured-combo refusal (exit 3) happens
                                BEFORE any build or download; the D12 GPU VRAM
                                warning runs here)
 6. Build + download (parallel)(D20: the build leg and the model download
                                (when missing) run at the same time, one
                                combined status line, fail-fast; see §4.8)
 7. Flag-capability gate       (D38: reads the BUILT binary, so it stays
                                after the parallel phase)
 8. Emit run-server script     (+ "start it now? (y/n)")
```

Console step headers are unnumbered (the numbering above documents the flow,
it is not printed).

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

### 4.4 Build (step 6, parallel leg or sequential fallback)

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
  mtime scan). Skip → step 6 (the parallel phase simply runs no build leg).
  The scan is the only freshness signal; there is no cache of source hashes
  (recorded: a digest would be a cross-process comparison, and this repo has
  measured such comparisons to be unreliable — AGENTS.md "A digest compared
  across wrapper invocations can differ for an UNCHANGED tree").
- `--no-build` / `-SkipBuild`: no build leg (the download still runs).
- Build failure → exit 4 with the last 50 lines of the build log
  (`build/llama-cuda/setup-wizard-build.log`); in the parallel phase the
  polling loop reports it (same tail, same exit, D20 fail-fast).

### 4.5 Model (step 5: selection; step 6: download leg)

Model table, flat over (model, quant) — D19 (all links verified 2026-11-04
against the HF API, Q5_K_XL entries 2026-11-05; sizes are `Content-Length`
of the resolve endpoint):

| # | Key | Quant | Kind | HF file | Size |
|---|---|---|---|---|---|
| 1 | `qwen38` | `q4k_m` (default) | dense hybrid | `unsloth/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-UD-Q4_K_M.gguf` | 15.4 GiB |
| 2 | `qwen36` | `q4k_m` (default) | MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 20.6 GiB |
| 3 | `flint` | `q5k_xl` (only) | SWA MoE | `unsloth/flint-GGUF` / `flint-UD-Q5_K_XL.gguf` | 27.9 GiB |
| 4 | `qwen38` | `q5k_xl` (alt) | dense hybrid | `unsloth/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 19.4 GiB |
| 5 | `qwen36` | `q5k_xl` (alt) | MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` | 24.7 GiB |
| 6 | `qwen36` | `q8k_xl` (alt, MTP) | MoE + MTP | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` / `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | 36.4 GiB |
| 7 | `apex35` | `q5k_m` (default) | MoE + MTP | `Apexintelligence-AI/Apex-35B-V1.0-GGUF` / `Apex-35B-V1.0-Q5_K_M.gguf` | 23.6 GiB |
| 8 | `apex35` | `q6k_l` (alt) | MoE + MTP | `Apexintelligence-AI/Apex-35B-V1.0-GGUF` / `Apex-35B-V1.0-Q6_K_L.gguf` | 30.5 GiB |

`-Quant`/`--quant` selects the quant (`qwen38`/`qwen36`: `q4k_m`, `q5k_xl`,
`q8k_xl`; `apex35`: `q5k_m`, `q6k_l`; default = the model's default entry);
numeric picks `1-8` address the flat list directly.
Recipe rows are keyed on the MODEL only (a row's flags do not depend on the
quant); the rows were measured on the default entries, so picking an alt
quant prints an `[assumption]`-style note: the row's speed/VRAM numbers are
for the default build. An unknown quant for a model fails loudly with the
available list.

Recorded decision D7: the owner's dense profile used
`Qwen3.8-27B-Q4_K_M.gguf`, which does **not** exist on HF under that name;
`Qwen3.8-27B-UD-Q4_K_M.gguf` is the closest available quant. The host-layer
plan depends only on layer count, but per-layer byte sizes differ slightly —
flagged as a known risk in docs/SETUP.md (unmeasured substitution).

Questions and rules:

- Model pick (1/2/3) with kind + file + size printed.
- Local path: default per key (`C:/Models/ALL/other/qwen-3.8|qwen-3.6|flint|apex`
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

### 4.6 Recipe (step 5, inside the settings phase)

Every row is **verbatim** from its source command. The wizard substitutes
only: the model path (question 6), the bind address (default = the row's own
value: 127.0.0.1 for R1-R3/R6, 0.0.0.0 for R4/R5/R7), the port (default
8090, all rows; asked in the settings phase, D22), and - for the
device-pinned rows (R5, R7) - the CUDA device index, appended as
`--device CUDA<n>` (D17: `-Device N` / `--device N` overrides; auto = the
largest card for `full`, the smallest card at or above the row minimum for
`16g`/`8g`, else the smallest card). When the resolved host is 0.0.0.0 the
wizard prints the USAGE.md §5 security note once.
Recorded decision D11: the wizard does not add, drop, or re-tune any flag of
a row (`-t 16` appears only where the row has it); D17 and D22 are the
recorded exceptions (a device index; the `--lm-prefill-prearm always`
normalization and the Pascal-or-lower `--lm-prefill-attn-int8` addition -
performance switches, never a tuning value of a measured flag).

**D22 flag completion (2026-11-07, owner request).** Applied once, after
the recipe is picked, as a pure function of (row flags, row env, compute
capability of the card the run will use - nvidia-smi's first line, the
D12/D17 convention; `SETUP_WIZARD_GPU_CC` overrides for testing):

- `--lm-prefill-prearm always` in every emitted profile: an existing
  occurrence (R3's `always`, R8's `auto`) is replaced, otherwise the flag
  is appended. Consequence: on the dense rows R1-R3 the idle pass
  preparation actually runs now - the server auto-turns the IMPLICIT
  default off without `--hybrid-moe`, but an explicit flag is honored (all
  rows carry `-np 1`; R1/R2/R5/R6's MTP-with-`--spec-mtp-defer` is the
  exempt spec case).
- Pascal and lower (compute capability <= 6.1): the pass's flash attention
  takes the int8 Q.K^T tile. If the row's env already sets
  `LLAMA_FATTN_TILE_Q8QK=1`, that env is kept (it is the emitted config,
  nothing to add - the flag is literally `setenv(LLAMA_FATTN_TILE_Q8QK,1)`
  in the server, "Same as LLAMA_FATTN_TILE_Q8QK=1"); otherwise
  `--lm-prefill-attn-int8` is appended and the emitted script carries a
  `# D22:` note line marking it as an arch-added APPROXIMATION (int8 Q per
  32 values; KL vs the exact pass in docs/plans/dense-qwen38-iq3-residency.md).
  Compute-capability detection failure is not fatal: the row is emitted as
  measured with a warning.

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
| R5 | qwen36 | full | 262k | q8_0 | `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 65536 --lm-prefill-ubatch 2048 --lm-prefill-segment 65536 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all` + MTP draft (`--spec-type draft-mtp`, `--mtp-draft-vocab 32768`) + `--lm-prefill-fuse --no-warmup`; quant q8k_xl (MTP file); env `LLAMA_MOE_HYBRID_LEAN_STALL_US=10000 LLAMA_FATTN_GQA8_MAXROWS=4` | PP 3600 t/s @50k, 2350 @200k; TG 77 @50k, 66 @200k (owner, GTX 5090 Laptop, Windows, one MoE model) |
| R6 | qwen36 | 16g | 262k | q8_0 | as R5 with `-fitt 8 -b 32768 --lm-prefill-segment 32768` (no `--no-warmup`); quant q8k_xl (MTP file) | PP 2000 t/s, TG 51 t/s @200k (owner, GTX 5090 Laptop, Windows, one MoE model) |
| R7 | qwen36 | 8g | 262k | q8_0 | `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 -ub 512 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 16384`; quant q4k_m; env as R5 | PP 590 t/s, TG 33 t/s @20k (owner, GTX 1070 8 GiB, Windows, one MoE model) |
| R8 | qwen36 | 16g | 262k | q8_0 | G42 canonical (USAGE.md §4, the former R6); quant q4k_m | TG 3.0–3.7x stock, PP 3.65–4.11x stock (G31 record) |
| R9 | apex35 | full | 262k | q8_0 | the R5 flag shape without a row env (the owner's running command for Apex-35B, D23): `--hybrid-moe -fitt 16 --lm-prefill ... --spec-type draft-mtp ... --mtp-draft-vocab 32768 --lm-prefill-fuse --no-warmup`; quant q5k_m (q6k_l falls back to this row) | TG ~100–110 tok/s, PP ~3,900 t/s at 20.7k tokens (2026-11-09, RTX 5090 Laptop, Windows; measured on `Apex-35B-V1.0-Q5_K_S.gguf`, not on the offered quants) |

Refusal rule (owner decision A): any (model, target, ctx) triple not in the
matrix → the wizard prints the table of available rows for that model and
exits 3. Examples refused: qwen36 128k, flint 16g/8g, apex35 16g/8g, qwen38 16g+128k.

GPU sanity check (warning only, recorded decision D12): if the card the run
will use (nvidia-smi's first line; for the device-pinned rows R5/R7 the
card selected per D17) has total VRAM below the target implies (full ≥
22 GiB / 16g ≥ 10 GiB / 8g ≥ 8 GiB for these models) the row is still
emitted with a warning that the recipe was measured on a 24 GiB card (R7
excepted, measured on 8 GiB). No refusal: `-fitt` and the memory guard fail
loudly in the server itself, and the owner explicitly wants the 16g/8g rows
usable on smaller cards.

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
  chat request (USAGE.md §5, port 8090), and the stop command (Ctrl-C).
  - A `# D22:` note line when `--lm-prefill-attn-int8` was added for the
    card (see §4.6).
- `--no-launch` / `-NoStart` skips the "start it now?" prompt entirely.

The emitted script is executable standalone (no wizard needed to re-run it),
and re-running the wizard overwrites it (the previous one is moved to
`run-server.last.ps1` / `.sh`, one generation of history).

### 4.8 The parallel phase (decision D20)

```
settings done:
   model missing AND build needed  →  build leg ∥ download leg
   exactly one of the two          →  that leg alone (no combined line)
   neither                         →  nothing to do
```

- **Windows.** The build leg is the two documented cmake calls (configure,
  then build), each wrapped in the D50/D53 `.bat` + rc-file shape and started
  with `Start-Process -WindowStyle Hidden -PassThru` WITHOUT `-Wait`: both
  streams live inside the `.bat`, so no redirected handle is ever created by
  the wizard process and the long-lived MSBuild/vctip descendants (the D49
  trap) cannot block it. The rc files (`swz-parallel-cfg.rc` /
  `swz-parallel-build.rc` under `build/llama-cuda`) are the completion
  signal, not the process handle. The download keeps the foreground
  `HttpWebRequest` Range loop (resume / token / sha-256 verify unchanged);
  its 500 ms tick renders the combined status line and fail-fasts the moment
  a stage rc appears nonzero.
- **Linux.** cmake configure + build run in ONE background subshell writing
  the same two rc files; `curl -fL -C - --retry 3 --retry-delay 5` runs in
  the background. A single 1 s polling loop renders the combined line, enforces
  the `--max-download-mib` test hook (D33), and fail-fasts: a nonzero cmake
  stage rc kills curl and exits 4 (`.part` kept), a dead curl kills the cmake
  tree and exits 5. An INT/TERM trap kills both legs and exits 6 (`.part`
  kept for resume).
- **Failure semantics.** First hard failure wins; the other leg is killed.
  Exit codes are unchanged (4 build, 5 download); a re-run resumes both (the
  binary-freshness scan and `.part` resume). A download failure prints a
  resume hint.
- **Edges.** Model already local → build only (the documented sequential
  `Invoke-Build` / `invoke_build` form, with its live log tails). Binary
  fresh or `-SkipBuild` → download only. `-SkipModel` without `-Model`
  (the AC-5 unattended shape) → sequential build, no settings questions.
  `--dry-run` prints the documented command text and starts no processes.

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
| D16 | (2026-11-05, owner decision) R7 recipe replaced by the owner's current 8g command: `--hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 32 -b 32768 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 32768 -sm none -mg 0`, bind 0.0.0.0, env `LLAMA_FATTN_TILE_Q8QK=1` kept; the LINUX-PASCAL §2 fuse/prearm recipe becomes history in that doc | owner instruction; the old recipe's measured numbers stay in LINUX-PASCAL.md as evidence | yes (restore the old row) |
| D17 | (2026-11-05) device-pinned rows (R5, R7) no longer hard-code `CUDA0` in the row flags; the wizard appends `--device CUDA<n>` at emission: `-Device N` / `--device N` overrides, else auto (largest card for `full`; smallest card at/above the row minimum for `16g`/`8g`, else smallest card). The D12 VRAM warning runs against the selected card | `CUDA0` is not guaranteed on multi-GPU or reordered systems; the row's recipe is unchanged apart from which card runs it | yes (re-pin the index in the row) |
| D18 | Consequence of D16: R7 now uses `--hybrid-moe-prefill-free-all`, which exists only in the fork (not origin/main), so a fresh-clone build's flag gate refuses R7 like R4/R5 (SETUP.md troubleshooting updated) | recorded so the refusal is not read as a new defect | yes (follows D16) |
| D19 | (2026-11-05, owner request "add choice for 5k_xl models") the model table is flat over (model, quant) with a `q5k_xl` entry for qwen38 (19.4 GiB) and qwen36 (24.7 GiB); `-Quant`/`--quant q4k_m\|q5k_xl` plus numeric picks 1-5; recipe rows stay keyed on model only and picking an alt quant prints a measured-quant note. Two defects found and fixed while validating: (a) setup.ps1 silently DROPS hyphenated dashed flags (`--dry-run`, `--non-interactive`, `--skip-model` land unbound in `$args` and PowerShell discards them - a `--dry-run` invocation downloaded the model), so setup.ps1 now normalizes the known dashed names up front and refuses anything unrecognized; (b) setup.sh's `resolve_gpu_index` cached its result in a global that dies with the `$(...)` subshell and its cache read died under `set -u`, so the sh side is now a pure function echoing the index | owner asked for the quant choice; the two fixes turn silent misconfiguration into loud failure (measured 2026-11-05) | yes (drop the quant entries / the normalization block) |
| D20 | (2026-11-06, owner request "optimize the setup flow: the user chooses all settings first, then the compile and the model download run in parallel at the end") the flow is now env → install → clone → **SETTINGS** (model + quant, local path, disk check, VRAM target × context; the unmeasured-combo refusal with exit 3 and the D12 GPU VRAM warning now happen BEFORE any build or download) → **PARALLEL PHASE** (build leg ∥ model download when missing; one combined status line `build: configuring/building … N min | model: M MiB downloaded`; fail-fast on the first hard failure) → flag gate → emit → start. The disk check (SDD §4.1) moved into the settings phase (it needs the model size) and gained its PowerShell twin (sh had it, PS did not). See §4.8 for the per-OS mechanics (PS: non-blocking `.bat` build legs in the D50/D53 shape + the foreground download loop's tick as the combined line; sh: both legs backgrounded under one polling loop + an INT/TERM trap). Bat-writing trap (plan D61, measured 2026-11-06): a cmd line that STARTS with `&` (`& cmake …`) aborts the whole batch before the rc-file line — write `<exe> <args> > log 2> err` with cmd-style exe quoting, never the PowerShell call-operator form. The flag-capability gate (D38) still runs after the build because it reads the built binary's `--help`. Console step headers are unnumbered | a refused combo previously paid the full build (up to ~22 min) and possibly a 20+ GiB download before exit 3; build and download share no state, so the phase's wall time is max(build, download) instead of their sum | yes (restore the sequential build → download order) |
| D21 | (2026-11-07, owner request "apply these profiles for the setup wizard for the appropriate VRAM profiles and models") the three MoE profiles measured by the owner (8 GB GTX 1070; >16 GB; 16 GB, all on Windows with one MoE model, Qwen3.6-35B-A3B) replace the qwen36 rows R5/R6/R7 (`full`/`16g`/`8g`, all 262k). Recipe rows are now keyed on (model, VRAM, ctx, **quant**): the >16 GB and 16 GB profiles need the MTP build (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` / `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf`, 36.4 GiB, new model-table entry 6, Q8_K_XL), so keying on the model alone would run MTP flags on a Q4 file. A quant with no row of its own falls back to the model's default-quant row with a warning (the D19 note). The former G42 row is kept as R8 (q4k_m, 16g), so the default qwen36 16g still resolves. The former owner recipes R5 (full) and R7 (8g, with `LLAMA_FATTN_TILE_Q8QK=1`) are replaced; qwen36 full on q4k_m now refuses with the table. Every new row sets the env pair of the owner's MoE profile. Scope: measured on Windows only, one MoE model; Linux, other MoE models and flint (R4) are not covered by these profiles and were not changed. The stats are the owner's, not re-measured by the wizard | owner asked for the tested commands to be applied; the quant key is needed so MTP flags never run on a non-MTP file (measured 2026-11-07 by the dry-run selection on both OSes) | yes (restore the model-only keying and the old R5/R7 rows) |
| D22 | (2026-11-07, owner request: int8 attention for Pascal and lower, `--lm-prefill-prearm always` for all profiles, a port input with default 8090) the wizard completes every row at emission (pure over the row, §4.6): (1) `--lm-prefill-prearm always` everywhere - an existing occurrence (R3 `always`, R8 `auto`) is replaced, otherwise appended; on the dense rows this turns the idle pass preparation ON where the server would auto-turn the implicit default off; (2) on a card with compute capability <= 6.1 (nvidia-smi `compute_cap`, first line; `SETUP_WIZARD_GPU_CC` test hook), the int8 flash-attention tile is enabled: row env `LLAMA_FATTN_TILE_Q8QK=1` stays as-is when present (it is the emitted config), otherwise `--lm-prefill-attn-int8` is appended with a `# D22:` note in the emitted script (the flag is literally `setenv(LLAMA_FATTN_TILE_Q8QK,1)` in the server, "Same as LLAMA_FATTN_TILE_Q8QK=1", target "a GPU without fast FP16, such as Pascal"; it is an approximation - int8 Q per 32 values); detection failure emits the row as measured with a warning; (3) the server port is asked in the settings phase, default **8090** (was 8033), explicit `-Port`/`--port` suppresses the question; `smoke.ps1`'s default follows. All rows accepted by the engine's prearm checks were verified per row (`-np 1` everywhere; MTP `--spec-mtp-defer` is the exempt spec case) | owner instruction; the int8 tile is the measured remedy for old architectures (AGENTS.md dense Qwen3.8 note: 1.19x at 15.7k with the flag) | yes (drop the completion function; port default is a one-liner) |
| D23 | (2026-11-09, owner request: add `Apex-35B-V1.0-Q6_K_L.gguf` and `Apex-35B-V1.0-Q5_K_M.gguf` from `Apexintelligence-AI/Apex-35B-V1.0-GGUF` to the setup wizard) new model key `apex35` with two flat-table entries (7: `q5k_m`, the default, 23.6 GiB; 8: `q6k_l`, an alt, 30.5 GiB; kind `moe-mtp`; default dir `C:/Models/ALL/other/apex` on Windows, `~/models/apex` on Linux) and ONE recipe row, R9 (apex35, `full`, 262k, `q5k_m`): the R5 flag shape (the owner's running command for Apex-35B) with `--device` pinning and bind 0.0.0.0 like R5, and NO row env, because the measured runs set none. `q6k_l` reuses R9 through the D19 fallback and its warning; `16g`/`8g` are refused with exit 3 like flint's (no measured row). Numeric picks are now `1-8` (the PowerShell script still said `1-5`). Evidence (2026-11-09): sizes from the HF API and the resolve endpoint's Content-Length agree (25,347,532,640 and 32,775,817,376 bytes); the first 32 MiB of each file (ranged read, header only) shows architecture `qwen35moe`, 41 blocks, `nextn_predict_layers` 1 and the four `blk.40.nextn.*` tensors in both quants, the same as the owner's local `Apex-35B-V1.0-Q5_K_S.gguf`, and no bf16 expert banks (the hybrid `mmf` hazard of AGENTS.md does not apply). NOT done: a download and a server run of the two offered quants. The measured numbers (TG about 100-110 tok/s on a 600-token greedy reply, PP about 3,900 t/s on a 20.7k-token prompt, RTX 5090 Laptop 24 GiB, Windows) come from `Apex-35B-V1.0-Q5_K_S.gguf`; the row's Measured and Note texts say so. Choices made without asking (the owner asked not to be asked): default = `q5k_m` (smaller, nearest to the measured Q5_K_S); only the `full` row | the owner supplied the two file names and the repo; the wizard emits measured rows only (SDD §3), and the only measured Apex command is the full-VRAM one | yes (remove the two table entries and R9 from both scripts, the test row, and these docs) |
