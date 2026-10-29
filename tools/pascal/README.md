# GTX 1070 measurement harness (Qwen3.6-35B-A3B, `--hybrid-moe`, MTP)

Scripts used for the 2026-10-24 and 2026-10-29 decode and prefill work (results: `docs/guide/LINUX-PASCAL.md` sections 3.7 to 3.10; contract and ledger: `docs/plans/gtx1070-moe-mtp-optimization.md`).
They run on Linux with one GPU (`CUDA_VISIBLE_DEVICES=0`) and refuse to start while another compute process
is on the GPUs. Output goes to the git-ignored `build/pascal/`.

| Script | What it does |
|---|---|
| `moe_bench.py` | One server arm. The server command is the owner's baseline (`BASE` in the script); `--add`, `--drop` and `--env` change it. Measures cold prefill at exact token counts (`--pp`, token arrays cut from `build/ctx100k/base.txt`), greedy decode on three fixed chats (`--chats`, `--tg`) and decode after a long prompt (`--deep`). Writes `results.jsonl`, `server.log`, the libraries the server mapped (from `/proc/PID/maps`) and a hash of every generated text. `--hlog` adds the hybrid engine's per-step log, `--wrap` a profiler prefix. `--bin` points at a side copy of `build/llama-cuda/bin` (the harness sets `LD_LIBRARY_PATH` to it; the RUNPATH names the build directory). |
| `abba.sh` | A, B, B, A arms of `moe_bench.py` (`ARGS_A` / `ARGS_B` for per-arm options) and the table of `abba_table.py`. |
| `abba_table.py` | Decode and prefill rates per arm, the A and B means, ms per verify round (eval time / (tokens - accepted drafts): the step cost apart from how many drafts the text lets through), draft acceptance, hit rate, cache slots, and whether the texts agree. |
| `queue.sh` | One `moe_bench.py` arm per line of a file (`LABEL \| options`), one after the other. |
| `arms_table.py` | The same table for single arms. |
| `check_mmq_remainder.cpp` | Bit-exactness of `GGML_CUDA_MMQ_REMAINDER`: the same `MUL_MAT_ID` on the CUDA backend with the mode off and on (modes 1, 2, 3), outputs compared bit for bit over six quantisations, uniform and skewed routing. Build line in the file's header. With the default mode it also prints one `HASH` line per case: compare them between two library builds (`LD_LIBRARY_PATH`) to see whether a kernel change moved any output bit. |
| `bench_vecdot.cpp` | Single-thread cost of the CPU expert dot products (Q4_K rows of 2048, Q5_K / Q6_K rows of 512) hot in the caches and streaming from DRAM. |
| `mmq_q8_inner_bench.cu`, `mmq_q4k_inner_bench.cu` | The inner loop of the Q8_0 and the Q4_K MMQ kernels in shared memory only (no global loads, no write-back): thread tile, hoisting and conversion variants, ablations of the scale arithmetic, bit-identical checksums, TOPS against the 26.9 TOPS register-only `dp4a` ceiling. Build lines in the file headers. Results: `docs/plans/gtx1070-moe-mtp-optimization.md` H29, H30. |
| `fa_pv8_check.cpp` | Causal-window attention at the Qwen3.6 prefill shape (hs 256, 16 heads, 2 KV heads, q8_0 K / V) on fixed random data, written to a file; `cmp` mode prints the NMSE, by query, head and head-dim block, of two files (e.g. with and without `LLAMA_FATTN_TILE_Q8PV=1`). It found the lane-mapping bug of the int8 PV that `test-backend-ops` could only report as a failure. |
| `ftk_arm.py` | First-token distributions (top 20) of the layer-major pass on prefixes of the harness text, for one attention configuration of a plain-decode server (repeats are bit-identical); `--compare` prints the KL of every arm against the first. |
| `bench_stream.cpp` | The CPU expert leg as a microbenchmark: a spin-barrier team reading m random experts (Q4_K gate / up, then Q5_K down) from a 1.9 GiB region on 4 KiB or 2 MiB pages, microseconds per request. Result: hugepages gain at most 1.4% (`docs/plans/gtx1070-moe-mtp-optimization.md` H14). Build line in the file header. |

A change that moves the expert-cache size (or anything else that changes which experts are resident) changes
the generated text, because a resident expert is computed on the GPU and a missed one on the CPU (different
activation quantisation). Then compare ms per round, not tok/s alone: tok/s also moves with the draft acceptance
of the new text.

Short arms while developing: `moe_bench.py --quick` (about 45 s: no deep prompt, three chats of at most 128 tokens). Keep the A-B-B-A for the final confirmation.
Side-copy the binaries of every arm (`cp -a build/llama-cuda/bin/*.so* build/llama-cuda/bin/llama-server build/pascal/NAME-bin/`) and pass `--bin`; a rebuild destroys the control. Run no build, test or host analysis while an arm runs.
