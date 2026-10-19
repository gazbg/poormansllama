# qwen38GZI

Layer-residency inference for llama.cpp: running Mixture-of-Experts Qwen models
on a GPU with less VRAM than the model needs, and faster prompt processing and
token generation than the stock `llama-server` on the same card.

This repository is the research record and the code. The llama.cpp fork is
vendored in `third_party/llama.cpp/` as plain files (no submodule). Experiment
history is in `docs/` and in the archive branches listed in
[docs/reference/HISTORY.md](docs/reference/HISTORY.md).

Status as of 2026-10-18. The record ends with entry G42 in
[docs/currentstate.md](docs/currentstate.md).

## What it does

- Weights of the target model stay in host RAM. The routed expert banks are
  large (about 432 MiB per layer for Qwen3.6-35B-A3B, about 17 GiB in total),
  so they cannot all live on an 8 GiB budget.
- `--hybrid-moe` keeps the experts on the host, keeps a VRAM cache of the
  experts used most, and runs the cold ones on the CPU.
- `--lm-prefill` runs a long prompt layer by layer, so each layer's expert bank is
  copied once per batch instead of once per micro-batch.
- A keep-free margin (`-fitt 16000`) leaves 16,000 MiB of the 24 GiB card free
  for other work. The rest is the budget the server may use.

## Status

Best configuration measured so far, called G42 in the record:

| Measurement | Result | Source |
|---|---|---|
| Token generation vs stock `llama-server`, same flags, 4k / 24k / 100k context | 3.0x to 3.7x | G31 entry |
| Prompt processing vs stock, same flags | 3.65x to 4.11x | G31 entry |
| Request wall time, fixture A, G42 against its baseline BASE (the configuration before G42) | 4.84 s to 3.66 s (0.755x) | G42 entry |

These numbers were measured on a 24 GiB RTX 5090 Laptop GPU with the 16,000 MiB
keep-free margin applied. That is not a physical 8 GiB card.

Limits:
- Only `-np 1` for the newest prefill features. `-np 2` works for the earlier
  layer-major prefill but with declines under concurrency.
- Qwen3.5/3.6 MoE only, q8_0 KV only, CUDA only. The record above was measured on
  Windows; a Linux build and measurements on a GTX 1070 are in
  [docs/guide/LINUX-PASCAL.md](docs/guide/LINUX-PASCAL.md).
- `--lm-prefill` acts only on prompt batches of 1,024 tokens or more, so small
  turns are not helped. Decoding is handled by `--hybrid-moe`. See
  [DESIGN](docs/guide/DESIGN.md#6-limits) for the full list.

## Start here

| Document | Read it for |
|---|---|
| [docs/guide/USAGE.md](docs/guide/USAGE.md) | Build, run, first request, measuring, troubleshooting |
| [docs/guide/LINUX-PASCAL.md](docs/guide/LINUX-PASCAL.md) | Building and running on Linux, and the measured prefill and decode work on a GTX 1070 (8 GiB, no tensor cores) |
| [docs/guide/DESIGN.md](docs/guide/DESIGN.md) | Why it is built this way, what is on by default, current state and limits |
| [docs/guide/PARAMETERS.md](docs/guide/PARAMETERS.md) | Every flag and environment variable, defaults, profiles |
| [docs/reference/LEDGER.md](docs/reference/LEDGER.md) | What was tried, what worked, what did not and why. Read before proposing an optimization |
| [docs/reference/HISTORY.md](docs/reference/HISTORY.md) | Where each older branch, fork and folder went |
| [docs/reference/README.md](docs/reference/README.md) | Index of the large reference material (generated indexes, plans, specs) |
| [AGENTS.md](AGENTS.md) | Rules for agents working in this repository (GPU lock, provenance, evidence) |

## Repository layout

```
AGENTS.md              rules for agents, measured gotchas
docs/guide/            main documentation (usage, design, parameters)
docs/reference/        ledger, history, generated indexes
docs/currentstate.md   dated record of every experiment (newest entries first)
docs/currentstate/     older entries, split into shards
docs/plans/            plans, contracts and results for each workstream
docs/*-sdd.md          design documents (layer residency, Qwen3.8 streaming)
tools/                 harnesses, checkers, GPU lock wrapper, provenance tools
third_party/llama.cpp/ the vendored fork (see third_party/README.md)
archive/               git-ignored: old branch bundles and evidence (not published)
releases/              git-ignored: prebuilt binaries from earlier builds
build/                 git-ignored: build trees and verdicts
```

## Rules that apply to every change

- GPU jobs run under `tools/with-gpu-lock.ps1`, which takes a shared named mutex
  so one project at a time uses the card.
- Default llama.cpp behaviour must not change unless a flag is given. Every new
  feature is opt-in.
- Speedups and quality claims need a pre-registered contract and a recorded run.
  Results without that are recorded as uncertified.
- Record every experiment in `docs/currentstate.md` with its outcome, including
  the ones that failed.
