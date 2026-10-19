# Usage

This guide covers building the server, running it with the best measured
configuration, sending a first request, measuring, and fixing common problems.
For the meaning of each flag see [PARAMETERS.md](PARAMETERS.md). For why the
design is what it is, see [DESIGN.md](DESIGN.md).

## 1. Prerequisites

Tested on Windows 11 with:

- NVIDIA GPU with 24 GiB (RTX 5090 Laptop GPU, 82 SMs, Blackwell `sm_120a`).
  Any CUDA GPU may build; the numbers in the record come from this card only.
- CUDA Toolkit 13.2 (`nvcc`).
- CMake 4.2 or newer.
- Visual Studio Build Tools 2022 or 18 (MSVC 19.4x or newer).
- Git for Windows with `core.longpaths` enabled (this repository sets it locally).
- The model file. It is not in the repository (`*.gguf` is git-ignored). The
  record uses `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` (19.9 GiB, Unsloth quant).

## 2. Get the code

```bash
git clone <this repository> qwen38GZI
cd qwen38GZI
```

`third_party/llama.cpp/` is part of the repository. There is no submodule to
initialise. See [third_party/README.md](../../third_party/README.md) for where
the fork's commits came from.

## 3. Build the server

Set the CUDA variables in the same shell that runs CMake. CMake reads them at
configure time, and a shell without them fails with
`CUDA Toolkit directory '' does not exist`.

```bash
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
export CUDA_PATH_V13_2="$CUDA_PATH"

cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON
cmake --build build/llama-cuda --config Release --target llama-server --parallel 16
```

Two options matter:

- `GGML_CUDA=ON` builds the CUDA backend.
- `LLAMA_LAYER_STREAMING=ON` builds the experimental path. It is OFF by default,
  and without it `--hybrid-moe` refuses to load with
  `--hybrid-moe needs a build with LLAMA_LAYER_STREAMING (CUDA)`.

The server is produced as `build/llama-cuda/bin/Release/llama-server.exe` with a
companion `llama-server-impl.dll`. Keep both together.

Use forward slashes in `-D` paths on Windows. Backslashes break CMake's escaping.

## 4. Run the best configuration

This is the G42 configuration. It is the one the record measures against.

```bash
build/llama-cuda/bin/Release/llama-server.exe \
  -m "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf" \
  --host 127.0.0.1 --port 8034 \
  -c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 \
  --hybrid-moe \
  --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto \
  --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 \
  --hybrid-moe-prefill-ubatch 0 \
  -fitt 16000 --ctx-checkpoints 32
```

What each group does:

| Group | Flags | Why it is set |
|---|---|---|
| Context | `-c 262144 -ctk q8_0 -ctv q8_0 -fa on` | 262k context with q8_0 KV (half the size of f16 KV) and flash attention |
| Slots | `-np 1` | One slot. The fuse and prearm features need it |
| Decode | `--hybrid-moe` | Experts on the host, VRAM expert cache, cold experts on the CPU |
| Prefill | `--lm-prefill --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768` | Layer-major prefill for batches of 1,024 tokens or more |
| Canonical | `--lm-prefill-canonical` | A cache hit gives the same bits as a fresh run. A follow-up prefills up to one chunk more than its new tokens, about 1 s |
| Fuse and prearm | `--lm-prefill-fuse --lm-prefill-prearm auto` | Rest of a prompt in one pass; prepare the next pass while idle |
| Hybrid prefill | `--hybrid-moe-prefill-ubatch 0` | Off, because `--lm-prefill` takes over the big batches |
| Budget | `-fitt 16000` | Keep 16,000 MiB of the card free |
| Checkpoints | `--ctx-checkpoints 32` | Up to 32 prefix checkpoints per slot for follow-up turns |

`releases/g42-s53/start-server.cmd` in a local checkout has the same flags for
the earlier `s53` binary. That folder is git-ignored, so the command above is the
copy to use.

## 5. Send a first request

Check health, then send one chat request:

```bash
curl http://127.0.0.1:8034/health

curl http://127.0.0.1:8034/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

`/health` returns `{"status":"ok"}` once the model is loaded. The last smoke run
reported health after 7 to 8 seconds and answered `pong` with 19 prompt tokens
and 2 generated tokens.

The server listens on `127.0.0.1` only. Do not bind `0.0.0.0` without a reason.
The default CORS setting allows all origins, which the server logs as a warning.

## 6. Use it for agents and multi-turn work

- Keep the same slot and prompt prefix between turns. The prefix cache and the
  checkpoints are what make follow-up turns cheap.
- `--lm-prefill` only acts on batches of 1,024 tokens or more. Short turns and
  tool results of a few hundred tokens take the ordinary path. This is expected:
  the record measures no gain below about 1,000 tokens.
- Each big pass costs about 0.6 s in expert-cache resize (shrink plus regrow)
  unless the idle preparation hides it. See [DESIGN](DESIGN.md#6-limits).
- With `-np 2` the big batches can decline under concurrency. That is recorded
  behaviour, not a crash.

## 7. Measure

Every claim about speed or quality needs a recorded run. Use this sequence:

1. Pre-register the contract in a plan file under `docs/plans/`. Write the
   criteria, the expected result and the validation commands before the first run.
2. Run the arms as paired runs, for example `A B B A`, so drift shows up.
3. Run GPU work through the lock wrapper. Put the executable path in the command
   so the verdict can hash the binary:

   ```powershell
   & tools\with-gpu-lock.ps1 `
     -Command 'powershell -NoProfile -ExecutionPolicy Bypass -File "<your-script>.ps1" -Exe "build\llama-cuda\bin\Release\llama-server.exe"' `
     -VerdictTag 'my-run-tag'
   ```

   The wrapper waits for the mutex, refuses to start while another compute process
   is visible, and writes `build/llama-cuda/verdicts/<tag>.verdict.json` and a
   `.done` file. Both are git-ignored.
4. Check the verdict. A run with provenance reads `PROVENANCE PASS`. A run
   without it is recorded as UNCERTIFIED.
5. Record the outcome in `docs/currentstate.md`, including the failures.

Helpers in `tools/`:

| Tool | What for |
|---|---|
| `server_mt.py` | Multi-turn request driver used for the llama-server contracts. Run `python tools/server_mt.py --help` for its options |
| `check-g1-contract.py`, `check-g2-contract.py` | Checkers that compare recorded results with the G1 and G2 contracts (see their plan files) |
| `pair-stats.py` | Paired-run statistics |
| `run-tests.ps1` | Test suites, including the host suite and the model-backed ctest targets |

[docs/model-run-testing-standard.md](../model-run-testing-standard.md) defines
the standard for model runs.

`-AllowUnprovenanced` skips the provenance gate and records a waiver line in
`provenance-waivers.log`. Use it only when the reason is written down. A
waiver run is not certified.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--hybrid-moe needs a build with LLAMA_LAYER_STREAMING (CUDA)` | Built without the option | Reconfigure with `-DLLAMA_LAYER_STREAMING=ON` |
| `CUDA Toolkit directory '' does not exist` | `CUDA_PATH` not set in the build shell | Export `CUDA_PATH` and `CUDA_PATH_V13_2` before configure and build |
| `CMakeCache.txt directory is different` | The repository folder was moved or renamed | Delete `build/llama-cuda` and reconfigure |
| `Filename too long` during git operations | Windows path limit | `git config core.longpaths true` (already set in this repository) |
| Provenance refused with `not-a-repo` | The gate predates the commit "tools: let the provenance gate read a vendored llama.cpp subtree", which accepts the vendored `third_party/llama.cpp` | Use the current `tools/` |
| Provenance refused with `binary_state=not-supplied` | The command does not name the executable | Put the `.exe` path in the `-Command` string |
| `cannot meet free memory target` in the load log | `-fitt` asks for more free VRAM than the card can give | Lower `-fitt`, or reduce `-c` |
| Server exits during load | Read the stderr log; the last line gives the reason | Often a flag that needs the layer-streaming build |
| `-np 2` declines big batches | Expected under concurrent clients | Use `-np 1` for the fused path, or accept the stock path for those requests |

Running the stock server with the same flags is the usual comparison. The
speedups in the record name their baseline: G31 is against the stock server, and
G42 is against the previous configuration (G32). Check the baseline of any number
before you quote it.
