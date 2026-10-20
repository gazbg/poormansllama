# Session scripts of the dense Qwen3.8-27B work (2026-10-19)

Run from anywhere; they `cd` to the repository root, write their logs and raw results to the git-ignored `build/dense/`
and use ONE GPU (`CUDA_VISIBLE_DEVICES=0`, `-dev CUDA0 -sm none -mg 0`). One GPU job at a time: do not build or analyse while a script runs
(host load perturbs the timings). Model: `~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf` (override with `MODEL=`).

| script | what it does |
|---|---|
| `bench.sh LABEL "<flags>" CHARS...` | starts a server, warms it, one cold prompt per CHARS (12000 / 30000 / 52000 characters = 3,659 / 10,037 / 17,940 tokens), prints PP, TG, peak VRAM, pass lines |
| `quick.sh LABEL "<flags>" CHARS NPRED` | start, one probe, stop |
| `collect.sh LABEL "<flags>"` | start, `tools/dense_equiv.py collect` (8 fixed prompts, token ids and logprobs), stop |
| `equiv-run.sh` | DQ-2: ordinary path against the layer-major pass at an equal chunk shape |
| `exact-run.sh` | DQ-5(a): ten streamed layers against the same layers resident |
| `pp-grid.sh` | DQ-3: LM against the stock grid at three lengths |
| `tg-grid.sh` | DQ-5(b): eight decode plans, 3 repeats each |
| `dq4.sh` | DQ-4: 16 and 6 resident layers, and the room guard with 36 |
| `dq6.sh` | DQ-6: the old (git HEAD) and the new Pascal MMQ tile, bit for bit (rebuilds `ggml-cuda` twice) |
| `q8qk.sh`, `q8qk-eq.sh` | the opt-in int8 QK attention: PP of both arms, and its first-token/continuation check against the exact pass |
| `mt.sh <server flags>` | multi-turn smoke (`tools/dense_multiturn.py`) |
| `capture2.sh`, `capture-pair.sh` | nsys capture of one prompt |

Tools next to them: `tools/dense_probe.py` (one request), `tools/dense_equiv.py` (collect / compare), `tools/dense_multiturn.py`,
`tools/dense_plan.py` (the placement planner and its cost model).
