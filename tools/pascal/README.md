# GTX 1070 measurement harness (Qwen3.6-35B-A3B, `--hybrid-moe`, MTP)

Scripts used for the 2026-10-24 decode and prefill work (results: `docs/guide/LINUX-PASCAL.md` section 3.7).
They run on Linux with one GPU (`CUDA_VISIBLE_DEVICES=0`) and refuse to start while another compute process
is on the GPUs. Output goes to the git-ignored `build/pascal/`.

| Script | What it does |
|---|---|
| `moe_bench.py` | One server arm. The server command is the owner's baseline (`BASE` in the script); `--add`, `--drop` and `--env` change it. Measures cold prefill at exact token counts (`--pp`, token arrays cut from `build/ctx100k/base.txt`), greedy decode on three fixed chats (`--chats`, `--tg`) and decode after a long prompt (`--deep`). Writes `results.jsonl`, `server.log`, the libraries the server mapped (from `/proc/PID/maps`) and a hash of every generated text. `--hlog` adds the hybrid engine's per-step log, `--wrap` a profiler prefix. `--bin` points at a side copy of `build/llama-cuda/bin` (the harness sets `LD_LIBRARY_PATH` to it; the RUNPATH names the build directory). |
| `abba.sh` | A, B, B, A arms of `moe_bench.py` (`ARGS_A` / `ARGS_B` for per-arm options) and the table of `abba_table.py`. |
| `abba_table.py` | Decode and prefill rates per arm, the A and B means, ms per verify round (eval time / (tokens - accepted drafts): the step cost apart from how many drafts the text lets through), draft acceptance, hit rate, cache slots, and whether the texts agree. |
| `queue.sh` | One `moe_bench.py` arm per line of a file (`LABEL \| options`), one after the other. |
| `arms_table.py` | The same table for single arms. |
| `bench_vecdot.cpp` | Single-thread cost of the CPU expert dot products (Q4_K rows of 2048, Q5_K / Q6_K rows of 512) hot in the caches and streaming from DRAM. |

A change that moves the expert-cache size (or anything else that changes which experts are resident) changes
the generated text, because a resident expert is computed on the GPU and a missed one on the CPU (different
activation quantisation). Then compare ms per round, not tok/s alone: tok/s also moves with the draft acceptance
of the new text.
