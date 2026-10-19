# Reference

Large reference material. The main documentation is in [../guide/](../guide/). This
directory holds the record of what was tried and where everything lives.

## Written by hand

| File | What it is |
|---|---|
| [LEDGER.md](LEDGER.md) | Experiment ledger: what was tried, the result, the reason, and the source. Read before proposing an optimization |
| [HISTORY.md](HISTORY.md) | Where each old folder, branch, fork and backup went, and what was not kept |

## Generated

Built by [`tools/gen-history-index.py`](../../tools/gen-history-index.py). Do not
edit these by hand. Rerun the script.

| File | What it is |
|---|---|
| [history/currentstate-entries.md](history/currentstate-entries.md) | Every dated entry in the current-state record, newest first, with its source line |
| [history/plans.md](history/plans.md) | Every plan and specification file under `docs/plans/` |
| [history/outer-archive-branches.md](history/outer-archive-branches.md) | Commits on each `archive/*` branch that are not on `main` |
| [history/nested-aer.md](history/nested-aer.md) | Fork commits on the AER branches of a retired nested llama.cpp |
| [history/nested-freebuff-5e3cb7b5.md](history/nested-freebuff-5e3cb7b5.md) | Fork commits of the MTP predictor work |
| [history/nested-freebuff-b78d8fc7.md](history/nested-freebuff-b78d8fc7.md) | Fork commits of the activation trace work |

## Primary sources

| Source | What it holds |
|---|---|
| [../currentstate.md](../currentstate.md) | The current-state record. Newest entries first. Older entries are in `../currentstate/` shards |
| [../layer-residency-sdd.md](../layer-residency-sdd.md) | Design document for the layer-residency mechanism |
| [../qwen38-layer-streaming-sdd.md](../qwen38-layer-streaming-sdd.md) | Design document for dense Qwen3.8 streaming (completed; historical) |
| [../plans/](../plans/) | Plans, contracts and results for each workstream |
| [../plans/llama-server-integration-results.md](../plans/llama-server-integration-results.md) | Results of the llama-server integration (G31 and G32) |
| [../plans/tg-beat-stock-8gb-262k.md](../plans/tg-beat-stock-8gb-262k.md) | The 8 GB token-generation contracts (G-series) |
| [../plans/qwen36-activation-expert-ranker-plan.md](../plans/qwen36-activation-expert-ranker-plan.md) | The activation expert ranker plan (T-Q36-AER) |
| [../AGENT_ENVIRONMENT.md](../AGENT_ENVIRONMENT.md) | Environment facts: tools, versions, build commands, the GPU, MCP servers |
| [../model-run-testing-standard.md](../model-run-testing-standard.md) | How model runs are recorded |
| [../../AGENTS.md](../../AGENTS.md) | Rules for agents, measured gotchas, GPU coordination |
| [../../third_party/README.md](../../third_party/README.md) | Where the llama.cpp fork came from and how its commit ids map |

## Search

The record is large. Search it directly:

```bash
git grep -n "G42" -- docs
git grep -n -i "falsified" -- docs
```

The record notes that search with `rg` over `docs/currentstate/` is the fastest
way to find an entry when its date is unknown.
