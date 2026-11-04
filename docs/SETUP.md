# Setup wizard (`setup.ps1` / `setup.sh`)

A cross-platform setup wizard for this repository. It checks the local
environment, offers to install missing toolchain pieces, clones the repository
when run outside one, builds `llama-server` with the fork's required options,
asks for model / VRAM target / context, downloads the selected GGUF from
Hugging Face when it is not already on disk, and emits a ready-to-run
`run-server.ps1` / `run-server.sh` containing the **measured** launch command
for the chosen combination. It then offers to start the server.

Design: [docs/setup-wizard-sdd.md](setup-wizard-sdd.md) · Plan and acceptance:
[docs/plans/setup-wizard-plan.md](plans/setup-wizard-plan.md)

## Quick start

### Windows (PowerShell 5.1 or 7)

```powershell
git clone https://github.com/gazbg/poormansllama.git
cd poormansllama
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Interactive: it asks for model, VRAM target, and context, and confirms every
write (tool install, clone, download, run-script overwrite, server start).

Fully unattended (example: the R5 owner recipe, qwen3.6 / full VRAM / 262k):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -NonInteractive -YesInstall -Model qwen36 -Vram full -Context 262k
```

### Linux

```bash
git clone https://github.com/gazbg/poormansllama.git
cd poormansllama
./setup.sh                    # interactive
# or unattended:
./setup.sh --non-interactive --yes-install --model qwen36 --vram full --context 262k
```

Run the wizard from outside a checkout and it clones the repository itself
(`-CloneDir` / `--clone-dir` picks the target; default
`%LOCALAPPDATA%\llama-setup` on Windows, `~/poormansllama` on Linux).

## What it does, in order

1. **Environment check** — git, CMake, CUDA toolkit, C++ toolchain (MSVC on
   Windows / g++ on Linux), `nvidia-smi` (GPU + driver), RAM. Missing pieces
   are offered as an install (winget on Windows, apt on Linux) behind a single
   y/n, then re-checked. **The NVIDIA driver is never auto-installed** — a
   missing GPU/driver is fatal with manual guidance (exit 2).
2. **Repository** — uses the checkout the wizard is running in, or clones
   (default `--depth 1`, branch `claude/flint-support`). If the branch is not
   on the remote (in-flight work), it falls back to the remote default branch
   and tells you; the flag-capability check below then decides which rows the
   resulting binary can run.
3. **Build** — `llama-server` with `GGML_CUDA=ON` and `LLAMA_LAYER_STREAMING=ON`
   into `build/llama-cuda/` (Windows: MSVC via the detected VS instance;
   Linux: CUDA arch list detected from `nvidia-smi`, refused when unknown).
   Skipped when the binary is newer than the sources (`-FullBuild` forces a
   rebuild). Build logs: `build/llama-cuda/setup-wizard-cmake-*.log`.
4. **Model** — qwen38 / qwen36 / flint, from disk if the table file is already
   there, otherwise downloaded from Hugging Face (resumable; partial data kept
   on interruption; the `HF_TOKEN` env var is honored and never printed).
5. **Recipe** — exactly the measured combinations (table below). Anything else
   is refused with the available rows (exit 3). Small-VRAM rows print a
   warning on a card that is bigger than the one they were measured on; they
   are never refused.
6. **Emit** — `run-server.ps1` / `run-server.sh` in the repository root: the
   row's flags verbatim with only the model path, bind default, and port
   substituted; one-generation `run-server.last.*` backup; server log under
   `build/llama-cuda/server-<row>.log`. Re-running the wizard on the same day
   rewrites the file byte-identically.
7. **Start?** — asks whether to start the server now (or `-NoStart`).

## Models (SDD table D7/D13)

| key      | Hugging Face repo            | file                                | size    | type        |
|----------|------------------------------|-------------------------------------|---------|-------------|
| qwen38   | unsloth/Qwen3.8-27B-GGUF     | Qwen3.8-27B-UD-Q4_K_M.gguf          | 15.4 GiB| dense       |
| qwen36   | unsloth/Qwen3.6-35B-A3B-GGUF | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf      | 20.6 GiB| MoE (A3B)   |
| flint    | unsloth/flint-GGUF           | flint-UD-Q5_K_XL.gguf               | 27.9 GiB| SWA MoE     |

`-ModelFile` / `--model-file` points at a local GGUF that is not the table
file (e.g. a different quant); the recipe is still chosen by `-Model`.

## Recipes (measured rows only)

| row | model  | VRAM | ctx  | measured on                          | note |
|-----|--------|------|------|--------------------------------------|------|
| R1  | qwen38 | full | 262k | RTX 5090 Laptop 24 GiB               | spec-MTP, dense layer-major pass |
| R2  | qwen38 | 16g  | 262k | RTX 5090 Laptop 24 GiB               | 48 host layers |
| R3  | qwen38 | 8g   | 128k | RTX 5090 Laptop 24 GiB (8 g budget)  | 64 host layers, `--lm-prefill-prearm always` |
| R4  | flint  | full | 262k | RTX 5090 Laptop 24 GiB               | hybrid MoE + 3-slot pass, binds 0.0.0.0 |
| R5  | qwen36 | full | 262k | RTX 5090 Laptop 24 GiB               | owner working recipe, binds 0.0.0.0 |
| R6  | qwen36 | 16g  | 262k | RTX 5090 Laptop 24 GiB (G42)         | TG 3.0-3.7x stock, PP 3.65-4.11x stock (USAGE.md) |
| R7  | qwen36 | 8g   | 262k | GTX 1070 8 GiB (LINUX-PASCAL.md)     | TG 32-34 t/s short, ~26.5 t/s @100k; sets `LLAMA_FATTN_TILE_Q8QK=1` |

Full flag text per row: the recipe tables in `setup.ps1` / `setup.sh` and
[docs/setup-wizard-sdd.md](setup-wizard-sdd.md) section 3.

## Exit codes

| code | meaning |
|------|---------|
| 0    | success (wizard finished; run script emitted when a recipe was chosen) |
| 1    | hard error (clone, build output missing, flag gate, unexpected) |
| 2    | environment not ready after the install attempt (incl. missing driver) |
| 3    | no measured recipe for the chosen model/VRAM/context |
| 4    | build failure (configure or compile) |
| 5    | model not found and download refused/failed |
| 6    | user abort (Ctrl-C / declined mandatory step) |

## Flags

PowerShell parameter names cannot contain hyphens, so `setup.ps1` uses
single-dash names (a double leading dash, `--Model`, also works); `setup.sh`
uses conventional dashed names.

| purpose                    | setup.ps1      | setup.sh              |
|----------------------------|----------------|-----------------------|
| dry run (no writes)        | -DryRun        | --dry-run             |
| never ask                  | -NonInteractive| --non-interactive     |
| model key                  | -Model         | --model               |
| VRAM target                | -Vram          | --vram                |
| context                    | -Context       | --context             |
| port / bind                | -Port, -Bind   | --port, --host        |
| model dir / explicit file  | -ModelDir, -ModelFile | --model-dir, --model-file |
| clone target / branch / url| -CloneDir, -Branch, -RepoUrl | --clone-dir, --branch, --repo-url |
| full clone / full build    | -FullClone, -FullBuild | --full-clone, --full-build |
| skip build / model / download | -SkipBuild, -SkipModel, -SkipDownload | --skip-build, --skip-model, --skip-download |
| allow auto-install (NI)    | -YesInstall    | --yes-install         |
| start / do not start       | -Start, -NoStart | (start asked), --no-start |
| HF token                   | -HfToken (or env HF_TOKEN) | --hf-token (or env HF_TOKEN) |
| CUDA arch (Linux)          | —              | --cuda-arch           |
| stop download at N MiB     | -MaxDownloadMib| --max-download-mib    |

`-SkipModel` still validates the recipe (exit 3 on an unmeasured combo); it
only skips model-file resolution and download. With `-SkipModel` and no
`-Model`, the wizard stops after environment + clone + build (exit 0).

## Test hooks (not for daily use)

- `SETUP_WIZARD_FORCE_MISSING='cmake,git'` (env var, both platforms) makes the
  named tools report missing, to exercise the install path.
- `-MaxDownloadMib N` / `--max-download-mib N` stops the transfer at N MiB
  (the partial file is kept, so the next run resumes).

## Troubleshooting

- **"clone of branch 'claude/flint-support' failed … falling back to the
  remote default branch"** — the branch is unpublished (in-flight work). The
  wizard continues on `main` and the **flag-capability check** then refuses to
  emit rows whose flags the built binary does not know (`binary does not
  support row R4 flags: --lm-prefill-copy-piece …`). R1-R3, R6, R7 work on
  `main`; R4 and R5 need a build from a branch that contains the two
  in-flight flags.
- **VRAM warning on a big card** (e.g. choosing R7 on a 24 GiB card) — the row
  was measured on a smaller card with a keep-free margin; the wizard warns and
  proceeds. The `-fitt` value is MiB of VRAM the CUDA runtime must keep free.
- **Download interrupted** — re-run the wizard; the `.part` file resumes.
- **Linux: unknown GPU arch** — the wizard refuses to guess `-DGGL_CUDA_ARCH`;
  pass `--cuda-arch` (e.g. `--cuda-arch 75` for a GTX 1070).
- **Server start refused while another GPU workload runs on this machine** —
  the shared-GPU tools in this repository use the `Global\AI-GPU-EXCLUSIVE-v1`
  mutex (`tools/with-gpu-lock.ps1`). The emitted run script does not take it
  itself; run it when the card is free.

## FAQ

**How do I know the server works?** (USAGE.md section 5, verbatim)

```
curl http://127.0.0.1:8033/health
# {"status":"ok"}

curl http://127.0.0.1:8033/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

(The `model` field may be any string; the server serves the loaded model.)

**I changed the emitted command by hand — will re-running the wizard kill it?**
No: the current `run-server.*` is moved to `run-server.last.*` first.

## Safety notes

- Downloads happen only from `huggingface.co` (model files) — the clone URL
  defaults to this repository's GitHub.
- Auto-install runs only the verified package set (winget: `Git.Git`,
  `Kitware.CMake`, `Nvidia.CUDA`, `Microsoft.VisualStudio.2022.BuildTools`;
  apt: `git`, `cmake`, `build-essential`, `nvidia-cuda-toolkit`) and only
  behind the explicit y/n (or `-YesInstall`).
- No firewall changes, no services, no driver installation, no registry edits
  beyond `core.longpaths` in the cloned repo.
- `HF_TOKEN` is read from the environment or `-HfToken`, used only for the
  download, never printed or written to files.
