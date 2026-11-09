# Vision on demand: results record

Plan and contract: [vision-on-demand-plan.md](vision-on-demand-plan.md). Mode: autonomous run under the owner's /goal of 2026-11-09 (acceptance below is the agent's own, never owner approval). All gates are soft evidence (AGENTS.md Gate Policy).

## Conventions

- **Machine.** RTX 5090 Laptop GPU, 24,463 MiB (24,435 MiB in the CUDA view), Windows 11, mains power (`power_state()` at the start of the run: `ac_line mains`, no GPU power cap). Each wrapper verdict records `gpu_pre=clean gpu_post=clean` (`build/llama-cuda/verdicts/<tag>.verdict.json`). One server at a time under `tools/with-gpu-lock.ps1`.
- **Provenance.** The wrapper's own binary gate cannot see the server path (`PROVENANCE GATE ... 1 refused: binary_state=not-supplied`, waiver recorded in `build/llama-cuda/verdicts/provenance-waivers.log`, as in every G31/G32 server session). Instead every `report.json` carries `provenance` (SHA-256 prefixes of `llama-server.exe`, `llama.dll`, `llama-server-impl.dll`, `mtmd.dll`, `ggml-cuda.dll`, `llama-common.dll`) and each binary directory a `PROVENANCE.txt`. (Decision DL-5.)
- **VRAM units.** "cudaMemGetInfo" = the CUDA runtime's free memory read in-process (`ggml_backend_dev_memory`), the unit of `--fit-target`; "nvidia-smi" = the external `memory.used` poller (`FreeSampler` / one read 3 s after ready and 3 s after the last request). They differ by a constant offset on this machine (AGENTS.md "three values"); numbers of different units are never subtracted from each other.
- **Binaries (side-copied, then never rebuilt in place).** `build/llama-cuda/bin/vod-base/`: pristine control, engine source digest `1B3F05238E46FC48`, `llama-server.exe` SHA-256 prefix `4c243d01b26c6c64`, `llama.dll` `24f126624df014fc`. `build/llama-cuda/bin/vod0/`: vod-base plus the env-gated log-only `LLAMA_VOD_LOG=1` lines in `tools/server/server-context.cpp` (encode, ordinary-path decode of an image chunk, projector init with a cudaMemGetInfo reading), digest `82C1449B1E58B90E`. Build: `cmake -S third_party/llama.cpp -B build/llama-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_LAYER_STREAMING=ON -DCMAKE_BUILD_TYPE=Release`, `cmake --build build/llama-cuda --config Release --target llama-server --parallel 28` (logs `build/vod-build-base.log`, `build/vod-build-vod0.log`, both exit 0).
- **Arms and flags.** The R8 recipe of the wizard plus `--hybrid-moe-prefill-free-all` (the owner's setting, D5), `-lv 4`: `-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32 --hybrid-moe-prefill-free-all -lv 4`. Model `Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf`; projector `mmproj-qwen-35b-tq3-BF16.gguf` (SHA-256 `da63cb47...cfa`, 902,822,528 B). NOVIS = no `--mmproj`; BASE = `--mmproj` resident; CPUV = BASE plus `--no-mmproj-offload`.
- **Harness.** `tools/server_vision.py` (selftest 22/22 PASS), `tools/run-server-vision.sh`. Requests are chat completions with `cache_prompt:false` (every request is a cold prefill), greedy, 24 tokens max. Image token count = `usage.prompt_tokens` minus a text-only control of the same question.

## Sessions (all exit 0; reports under `build/llama-cuda/<tag>/report.json`)

| Tag | Arm | Binary | Env | Reps | Fixtures |
|-----|-----|--------|-----|------|----------|
| `vod0-novis` | NOVIS | vod-base | - | 1 (no images) | - |
| `vod0-base` | BASE | vod-base | - | 1 | digit trio at 724 px |
| `vod0-baselog1` / `vod0-baselog2` | BASE | vod0 | `LLAMA_VOD_LOG=1` | 2 | trio 724 / trio 1024 (digits, see F-6) |
| `vod0-cpuv1` / `vod0-cpuv2` | CPUV | vod0 | `LLAMA_VOD_LOG=1` | 2 | same as baselog1 / baselog2 |
| `vod0-fixchk` | BASE | vod0 | `LLAMA_VOD_LOG=1` | 1 | FINAL fixture set (colour trio) |

Command shape: `bash tools/run-server-vision.sh <tag> "--phase p0 --reps N" -- --arm base|novis|cpuv --exe build/llama-cuda/bin/<dir>/llama-server.exe --free-all [--env LLAMA_VOD_LOG=1]`.

## M1: what the resident projector costs the expert cache (hypothesis H1, F3)

| Arm | margin used by the hybrid engine | free VRAM at sizing (cudaMemGetInfo) | cache after sizing | cache at ready (idle) | startup pass arming |
|-----|---------------------------------|--------------------------------------|--------------------|-----------------------|---------------------|
| NOVIS (`vod0-novis`) | 16,000 + 160 MiB | 19,047 MiB | 54 slots (3,755.0 MiB, 69.5 MiB per slot) | **2 slots**, pass armed (free-all floor) | accepted |
| CPUV (`vod0-cpuv1`) | 16,000 + 160 MiB | 19,047 MiB | 54 slots | 2 slots, pass armed | accepted |
| BASE (`vod0-base`, `vod0-baselog1/2`) | **17,109** + 160 MiB | 19,047 MiB | **38 slots** (2,662.7 MiB) | **21 slots**, pass NOT armed | **declined** |

The declined arming, verbatim: `hybrid-moe: lm-prefill: no room for 2228 MiB + margin 17269 MiB: free 16317 -> 18747 MiB, expert cache 38 -> 2 slots (min 8), make_room true, shrunk 1` then `lm_acquire_n: lm-prefill: could not make 2228 MiB of device memory available; this batch takes the ordinary path`. The target is 2,228 + 17,269 = 19,497 MiB; the best reachable (cache at its 2-slot floor) is 18,747 MiB, 750 MiB short. With the margin NOVIS has (16,160) the target is 18,388 MiB and the same free memory suffices (NOVIS and CPUV both armed).

**H1 verdict: CONFIRMED, in effect and in mechanism.** `fit_params_target` carries the projector's 1,109 MiB device estimate (`server-context.cpp:1507-1531`) and `hybrid_size_cache` turns it into the permanent keep-free margin (`:1357`); unlike the speculative context's estimate it is never released once the projector is resident. The projector's 1,112 MiB therefore cost twice: once as resident weights and buffers, once as extra margin that every later guard (the cache sizing, the KV-pressure shrink, the pass guard `hybrid_lm_vram`) re-adds. Measured consequences on this card with the owner's flags: 16 fewer cache slots at sizing (16 x 69.5 MiB = 1,112 MiB, equal to the projector's measured device bytes), and the layer-major pass preparation refused at startup. Whether a long text prompt is also declined at request time is NOT measured here (VOD-3 measures it: pass count with `LLAMA_LM_LOG=1`, BASE vs OD).

## M2: projector device bytes

| Quantity | Value | Unit / source |
|----------|-------|---------------|
| weights | 861.0 MiB (902,822,528 B, BF16) | file size |
| clip compute buffer (warm-up at 1472 x 1472) | 248.10 MiB CUDA0 + 24.93 MiB CPU | `reserve_compute_meta` log line |
| estimate added to the margin | 1,109 MiB (device) / 1,134.0 MiB reported incl. host | `[mtmd] estimated worst-case memory usage of mmproj` |
| measured at init | **1,112.0 MiB** (17,429 -> 16,317 MiB free), 609.6 / 611.3 ms (`t_init_ms`, file read + upload + warm-up) | cudaMemGetInfo, `vod: projector init` |

The estimate was accurate to 3 MiB at init. After the requests nvidia-smi used memory rose by ~320 MiB on BASE (5,927 -> 6,251 MiB) and by ~2,036 MiB on CPUV (5,029 -> 7,065 MiB); the CPUV growth is the expert cache regrowing from its 2-slot floor at the first decode (cache 2 -> 52 slots in the log) and is not projector memory. A whole-server nvidia-smi difference between arms (BASE ready 5,927 vs NOVIS 5,029 MiB, +898) is NOT the projector's bytes: the arms also differ in cache slots (21 vs 2 at ready) and armed state. Idle-state comparisons for the acceptance criterion VOD-3 therefore use one fixed warm-up sequence per arm and both units (plan, VOD-2).

## M3 / M4: image encode, GPU vs CPU (median of the runs, ms)

| Image tokens | GPU encode A / A2 | CPU encode A / A2 | CPU / GPU |
|--------------|-------------------|-------------------|-----------|
| 529 | 85.3 / 86.6 (first ever 272) | 1,732.7 / 1,748.2 | 20.3x |
| 1,024 | 248.9 / 244.9 | 4,025.5 / 4,005.7 | 16.3x |
| 2,025 | 834.6 / 838.4 | 11,077.0 / 10,912.1 | 13.1x |

A-A spread of the GPU medians is below 2%, of the CPU medians below 1.5%. Whole image request, `prompt_ms` median (encode + ordinary-path decode + the few text tokens), GPU vs CPU: 531 tokens 1,573-1,984 vs 3,224-3,491; 1,026 tokens 2,793-2,869 vs 6,805-6,907; 2,027 tokens 5,857-5,902 vs 16,173-16,479 (the first colour request of a run is ~0.4 s slower: first-use costs). The CPU option costs 1.7-2.8x on the whole request.

## M5: the ordinary-path prefill of image embeddings (the number phase 2 must beat)

Median decode time of an image chunk (`mtmd_helper_decode_image_chunk`, `vod: decode_chunk`), tokens/s: 529 tokens 1,335-1,431 ms (370-396 tok/s); 1,024 tokens 2,393-2,409 ms (425-428 tok/s); 2,025 tokens 4,814 ms (421 tok/s). The CPUV rows are the same path (395-404 tok/s). The cause is the one AGENTS.md names for the window path: every ubatch copies all 40 expert banks (~1.2 s). Reference (not measured here): a layer-major pass streams the banks once, with a floor of ~1.3 s per pass whatever the length, and runs text at ~2,000-2,600 tok/s; so the expected gain of phase 2 is small for a 529-token image (~1.3 s either way), ~1.7x at 1,024 tokens, ~3x at 2,025 tokens. This expectation is UNPROVEN and is measured in VOD-4.

## Findings that change the plan

- **F-1.** H1 is confirmed and its worst consequence is the refused startup arming, not the 16 slots (M1). This strengthens goal (a): OD must not add the projector to the margin (already planned, D12), and VOD-3 adds a pass-engagement check on a long text prompt for BASE / OD / NOVIS.
- **F-2.** The projector's real cost (1,112 MiB, 16 slots) is inside the plan's expected band (0.3-1.5 GiB of cache), so the VOD-0 escalation condition does not fire. Decision DL-7: continue to VOD-1.
- **F-3.** Encode time on the GPU is small next to the ordinary-path decode (85-835 ms against 1.3-4.8 s), so the unpark cost (a 0.86 GiB upload at ~14 GB/s is ~60 ms plus a compute-buffer allocation) is small next to the request; the CPU option is 13-20x slower on the encode alone. These are report-only inputs, no threshold is claimed.
- **F-4.** The server's `batch_max_tokens` default (1,024) means three 531-token images are encoded in three separate batches (`chunks=1` each); VOD-2's up-front encode must hold the embeddings of all batches, not one `mtmd_batch`.
- **F-5.** At `-lv 4` the INFO lines of the hybrid sizing and of the clip load are visible; the scrapers in `tools/server_vision.py` read them (selftest-covered).
- **F-6 (fixtures, DL-6).** The digit trio was misread by the CONTROL itself (5206 -> 5246 at 724 px in `vod0-base` / `vod0-baselog1`, 3917 -> 291 at 1024 px in `vod0-baselog2`), so it cannot discriminate between arms. The trio fixture is now three colour squares at 724 px; the single-image colour and digit fixtures were read correctly in every run (7 of 7 in `vod0-fixchk`, negative control failing as designed). The `trio` rows of `vod0-base`, `vod0-baselog1/2` and `vod0-cpuv1/2` are excluded from all comparisons.

## VOD-0 status

Acceptance (plan): M1-M5 tabulated with commands, build digests, GPU probes, power state and VRAM units; H1 stated. Met. The log-only engine change is behind `LLAMA_VOD_LOG`; the default-off identity of the final build is re-checked in VOD-2 (criterion VOD-1) on the final binary. No speed or VRAM claim beyond the measurements above is made.

## VOD-1: park / unpark in mtmd and clip (library)

Change (engine, `third_party/llama.cpp`): `tools/mtmd/clip.{h,cpp}` (`clip_context_params::park_on_init`, `clip_park`, `clip_unpark`, `clip_is_parked`, `clip_release_device`, a parked-load branch in `load_tensors`, a guard in `clip_encode`), `tools/mtmd/mtmd.{h,cpp}` (`mtmd_context_params::park_on_init`, `mtmd_park`, `mtmd_unpark`, `mtmd_is_parked`), `tests/test-mtmd-park.cpp` (+ CMake: built, not registered with ctest). New struct members are appended LAST (both structs are initialised positionally). Behaviour with `park_on_init == false` is the previous code path (the resident branch of `load_tensors` is untouched; `clip_encode` only gained the parked guard).

How it works: a parked-init context reads the 861.0 MiB of weights straight into a page-locked host buffer (`CUDA_Host`), leaves every tensor of `ctx_data` unallocated, skips the warm-up, and frees its scheduler and its accelerated backend instance. `clip_unpark` re-creates the backend, allocates the device buffer for all tensors in one `ggml_backend_alloc_ctx_tensors_from_buft`, copies the weights with async sets from the host mirror, re-creates the scheduler; the first encode then reserves the compute buffers for ITS batch (the existing lazy `warmup()` of `clip_encode`). `clip_park` frees scheduler, device buffer and backend and nulls the tensors' data pointers. An audio-bearing projector is refused at init.

Test: `test-mtmd-park <mmproj> <text model> [cycles]` (the text model is loaded vocab-only), GPU, three fixtures (448x448, 672x512, 896x896 px = 196 / 336 / 784 tokens). Command: `pwsh -NoProfile -File tools/with-gpu-lock.ps1 -AllowUnprovenanced -VerdictTag <tag> -Command "build\llama-cuda\bin\Release\test-mtmd-park.exe C:/Models/ALL/other/qwen-3.6/mmproj-qwen-35b-tq3-BF16.gguf C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf <cycles>"` with `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"` exported (Git Bash otherwise rewrites the `C:/...` arguments: first run `vod1-park3` died on the mangled path). The wrapper attested the test binary (`PROVENANCE GATE ... 1 complete, 0 refused`, source digest `7D4B511BC3147E7B`, `llama.dll` hash prefix `EF1D618600F9C8A9`).

| Session | cycles | rounds | embeddings byte-identical (memcmp) | max residue after park (cudaMemGetInfo) | result |
|---------|--------|--------|------------------------------------|-----------------------------------------|--------|
| `vod1-park3` | 3 | 9 | 9 of 9 | 0.0 MiB | ALL PASS |
| `vod1-park100` | 100 | 300 | **300 of 300** | **0.0 MiB** | ALL PASS |
| `vod1-memcheck` (compute-sanitizer memcheck, `--print-limit 500`) | 2 | 6 | 6 of 6 | 0.0 MiB | ALL PASS, `ERROR SUMMARY: 0 errors` |
| `vod1-keepbe` (diagnostic, `LLAMA_MTMD_PARK_KEEP_BACKEND=1`) | 3 | 9 | 9 of 9 | **128.0 MiB** (106 / 124 / 128) | the < 64 MiB criterion FAILS as expected |

Timings (`vod1-park100`, 300 rounds, mean): parked init 490.5 ms (disk to pinned host), **unpark 76.7 ms**, encode 83.6 ms (includes the compute-buffer allocation of the batch), **park 6.0 ms**. Device memory while unparked (cudaMemGetInfo, baseline = no projector): 992.0 / 1,026.0 / 1,082.0 MiB for 196 / 336 / 784 tokens = 861.0 MiB weights + the per-batch compute buffer (22.98 / 39.40 / 91.92 MiB, from the `reserve_compute_meta` lines) + ~108-130 MiB of backend state and pool. A parked context holds exactly the baseline (23,121.0 MiB free before and after).

Findings:
- **H5 resolved (embeddings identical):** the parked/unparked context reserves a different compute buffer than the resident one (22.98-91.92 MiB per batch against the 248.10 MiB worst-case warm-up) and still produces byte-identical embeddings in every round: the kernel selection does not depend on the reserve shape here.
- **H3 resolved:** the worst-case compute buffer (248.1 MiB) is far above what a small request needs (23-92 MiB). The window can allocate per batch; for the room request (`need_bytes`) VOD-2 uses weights + the worst-case buffer + 128 MiB of backend state until it has a better bound.
- **H4 resolved for the CUDA allocator:** after `clip_park` the free memory returns to the baseline exactly (0.0 MiB residue over 300 rounds). Freeing the backend is REQUIRED for that: with the backend kept, 106-128 MiB stay (pool and handles).
- A parked projector has to be given back the same way after any failure: `clip_unpark` returns the context to the parked state on an allocation failure (not exercised on a real OOM here; the fault-injection arm of VOD-2 covers the server's reaction).
- Host tests: `ctest -R mtmd` 2/2 (`test-mtmd-c-api`, `test-mtmd-impl`). The full host ctest list (78/78) is run once at the closeout on the final tree.

VOD-1 acceptance (plan): VOD-2 (embeddings identical over 100 cycles) MET at library level; VOD-3 (i)/(ii) idle bytes and no leak MET at library level (0.0 MiB residue; server-level idle comparison is VOD-2/VOD-3's); VOD-1 default-off identity is checked at server level on the final binary (VOD-2). Memcheck on the library path: 0 errors.

## VOD-2: server flag, encode window, wait rule

Change (engine): `common/common.h` + `common/arg.cpp` (`--mmproj-on-demand`, `--mmproj-wait-timeout`), `include/llama-hybrid.h` + `src/llama-hybrid-api.cpp` (`llama_hybrid_vram_acquire` / `llama_hybrid_vram_release`, thin wrappers over the pass's own `hybrid_lm_vram`), `tools/server/server-context.cpp` (parked init, no margin add for the projector, `vod_try_room` / `vod_acquire_projector` / `vod_prepare_slot`, `slot.vod_batches`, the guard in `encode_mtmd_chunk`, the lookup in `process_mtmd_chunk`), `docs/guide/PARAMETERS.md`. Binary: `build/llama-cuda/bin/vod2a/` (engine digest `AB142234486D731C` on top of the VOD-1 commit `ff06b0e2`).

How it works: after a slot has decided how much of its prompt the cache holds (`keep_first(n_past)`), `vod_prepare_slot` lists the image chunks that start at or after the first uncached token. None: the device is not touched. Otherwise: disarm an idle-prepared pass (it holds ~2.2 GiB of the room the window needs), make the room (`llama_hybrid_vram_acquire`: the expert cache shrinks as for a pass, to its floor with `--hybrid-moe-prefill-free-all`; without the hybrid engine: plain free memory above the fit margin), unpark, encode every such chunk (several `mtmd_batch`es if `batch_max_tokens` requires), park, hand the room back (the cache regrows at the first decode of <= 16 tokens), keep the embeddings in `slot.vod_batches`. Waiting: poll every 0.5-1 s on the loop thread inside `yield_to_queue` (the queue's worker serves metrics / health meanwhile), a WARN every ~10 s, `--mmproj-wait-timeout` seconds at most (0 = no limit). The window runs on the loop thread (`yield_to_queue` runs its work on the calling thread), the same thread as the shrink and the regrow.

### Hard criteria on `vod2a` (soft evidence, all sessions `gpu_pre=clean gpu_post=clean`, mains power)

| Criterion | Evidence | Result |
|-----------|----------|--------|
| VOD-1 default-off identity, A B B A (pristine `vod-base` vs `vod2a` WITHOUT the flag, tool `acc`) | `vod2-id-ctlA1`, `-offB1`, `-offB2`, `-ctlA2`: text, `prompt_n` / `cache_n` / `predicted_n` identical in 13 of 13 steps in all three comparisons; PP of the long text step 463.9 / 463.9 / 463.9 / 463.5 tok/s, of the mid step 441.7 / 441.8 / 442.0 / 441.8; TG of the 64-token step 95.7 / 95.7 / 96.2 / 96.1 tok/s (all within 0.6%, bar 3%); idle nvidia-smi used 5,927 / 6,001 / 6,197 / 6,199 MiB and in-process free-watch lines byte-identical in every run | MET |
| VOD-2 embeddings identical | library level: 300 of 300 rounds (VOD-1). Server level: every fixture answer is the same and right in OD and BASE | MET |
| VOD-3 (i) idle OD vs NOVIS within 64 MiB | nvidia-smi used MiB at ready / after the text steps: NOVIS 5,029 / 7,015, OD 5,029 / 7,015 (both runs); after 10 image requests and the 3-turn conversation OD 7,045 / 7,047 (+30 / +32 MiB over the text-only idle; BASE for comparison 6,001 -> 6,199) | MET |
| VOD-3 (ii) no leak over many windows | library: 0.0 MiB residue over 300 rounds; server soak: 76 windows, idle 7,027 -> 7,031 MiB (+4) | MET |
| VOD-3 (iii) in-process free-watch minimum >= 16,000 MiB through the sessions, windows included | OD 16,297 MiB in `vod2-acc-od1`, `-od2` and the soak; NOVIS 16,297; BASE 16,317-16,567 (unit: cudaMemGetInfo, the watcher's `lowest free device memory` line written at a graceful stop) | MET |
| VOD-3 (iv) idle expert-cache size equals NOVIS's, not BASE's | the cache-resize sequences of NOVIS and OD are identical for the same text steps (54, 2, 52, 2, 52, 2, 51 slots) and the idle memory is equal (above); BASE: 38 -> 2 -> 21 -> 2 -> 20 slots at the same steps | MET (mechanism check, no speed claim) |
| VOD-4 answers | the 7 fixtures and both single-image turns of the conversation are answered correctly in every OD session (`vod2-od1`, `vod2-acc-od1`, `vod2-acc-od2`) and in BASE; `vod2-acc-od2` vs `vod2-acc-ctl2`: 13 of 13 steps identical in text, counts AND image chunks encoded per step | MET |
| VOD-4 multi-turn (DL-12 form) | image chunks encoded per turn, BASE vs OD: turn 1: 1 / 1, turn 2: 1 / 1, turn 3: 2 / 2; `prompt_n` 557 / 588 / 2,649 and `cache_n` 0 in both | MET |
| VOD-5 wait and failure | `vod2-fault-room` (N = 3): request 1 waits (4.17 s against 2.18 s for request 2), WARN logged, answer right, 31 `/health` polls all 200, slowest 24 ms; `vod2-fault-encode`: request 1 HTTP 500, request 2 right, window logged `FAILED`, projector parked again, server alive; `vod2-fault-timeout` (3 s): both image requests HTTP 500 after 3.08 s / 3.09 s, text request fine, `/health` all 200 | MET |
| VOD-6 soak | `vod2-soak-od`: 120 requests (44 text, 68 single image, 8 three-image), seed 11, 12 cancelled by closing the connection; 108 OK, 0 failed, 68 of 68 graded answers right, 76 windows (92 images), 0 failed windows, 0 waits, no CUDA error / assert in the log, 287 s | MET |
| VOD-6 memcheck on an image request (criterion: `ERROR SUMMARY: 0 errors`, kept literal, DL-13) | `vod2-memcheck-nograph`: OD under `compute-sanitizer --tool memcheck` with `GGML_CUDA_DISABLE_GRAPHS=1`: **`ERROR SUMMARY: 0 errors`**, 3 image requests answered right. The first run with CUDA graphs on (`vod2-memcheck`) and the pristine resident-projector control (`vod2-memcheck-ctl`, same requests, no new code) both report **16 errors of ONE class**: `Program hit cudaErrorGraphExecUpdateFailure (error 910)` at `cudaGraphExecUpdate` / `cudaGetLastError` (ggml-cuda's own graph-update fallback; no invalid access, race or uninitialised read), so they are not task-owned | MET (0 errors with the graph source removed; the 16 are pre-existing) |

### Report-only measurements (no threshold, no claim)

- **Window cost** (all OD sessions, ms, median; `room + unpark` includes the cache shrink to the floor): 529 tokens 119 (n = 48), encode 94, park 5.7; 1,024 tokens 117 (32), encode 251, park 6.1; 1,587 tokens (3 images) 119 (10), encode 272; 2,025 tokens 116 (4), encode 808, park 7.4; 2,554 tokens 116 (2), encode 911. Without the hybrid engine (stock placement smoke, `vod2-stock-od`): room + unpark 76-80 ms (nothing to shrink), 7 of 7 fixtures right.
- **Image request time, OD minus BASE** (`prompt_ms`, means of two sessions each, same session shape): +500 / +482 / +480 / +487 / +486 / +492 / +501 ms for the 7 fixtures, +504 / +492 / +519 ms for the 3 conversation turns, i.e. a fixed +0.48..0.52 s whatever the image size (1.6-7.7 s requests). Composition from the log: shrink ~40 ms + unpark ~76 ms + park ~6 ms inside the window, and ~370 ms for the expert-cache regrow that the first short text decode triggers right after the window (`cache resized to 52 slots`, 372-385 ms) because `--hybrid-moe-prefill-free-all` emptied the cache. With the resident projector BASE never shrinks the cache for an image request. The ordinary-path image decode (370-430 tok/s) is unchanged.
- **Text prompts, pass engagement** (DL-8; the long step is a 5,746-token prompt, `cache_prompt:false`): BASE (resident projector) takes NO pass (1 declined arming at start-up + 4 declined batches: `no room for ... margin 17269 MiB`), PP 464-465 tok/s; OD takes passes (1,403 tokens at 924-926 tok/s, 4,096 tokens at 2,181 tok/s, 1,650 tokens at 1,069 tok/s), PP 1,641-1,649 tok/s (3.5x BASE); NOVIS takes passes with the FUSED one-pass prefill (`lm-fuse: ... in one batch`), PP 2,611 tok/s. The gap between OD and NOVIS has a cause in the code, not in the memory: any server started with `--mmproj` sets `slot.prompt.tokens.has_mtmd`, and the fuse gate requires it to be false (`server-context.cpp:5631`), so with a projector loaded the fused prefill is OFF even for text-only prompts (the prompt is cut at its checkpoint, 4,096 + 1,650, two bank streams). Candidate for VOD-5 (phase 3): fuse text-only prompts on a projector server under `--mmproj-on-demand`.
- **TG after a pass**: the 64-token step decodes at 63.9-65.5 tok/s in OD and NOVIS (pass, cache regrown) against 90-96 tok/s in BASE (no pass, cache never shrunk): a property of the pass (the cache is re-learned), not of the projector. Short outputs (2-5 tokens) give meaningless TG numbers and are not compared.
- **Startup** (OD): the cache is sized to 54 slots as in NOVIS (BASE: 38), the pass is armed (134 ms; BASE: refused), the projector is parked (init 519-536 ms from disk to pinned memory, 0.0 MiB on the device), ready in 6.1 s with a warm file cache.

## VOD-3: phase-1 closeout and Owner Summary (the agent's own autonomous acceptance of phase 1, not owner approval)

Scope of this acceptance: it certifies exactly the criteria of the plan's contract for phase 1 (VOD-1 .. VOD-6 and VOD-9) on this machine (RTX 5090 Laptop, Windows, mains power), with Qwen3.6-35B-A3B Q4_K_S and its BF16 projector, with the flags listed under Conventions. It does NOT certify other models or cards, a physical 8 GiB card, `-np > 1`, audio, video, or any speed gain: performance is report-only (D10). Phase 1 leaves the image tokens on the ordinary prefill path.

### 1. Original goal

The owner asked whether the project's idea for model layers ("load the part that is needed when it is needed, then unload it") could also serve the vision projector, so that it does not occupy VRAM between image requests and the freed memory goes back to the expert cache, without making image requests slow (goal (a) text-turn VRAM, goal (b) image-request speed, weighted equally).

### 2. How it works, in plain words

The projector is the small neural network that turns a picture into the numbers the language model reads. It is about 0.84 GiB. Normally the server keeps it on the graphics card for as long as it runs, and (under `--hybrid-moe`) it also reserves another ~1.1 GiB as "keep free" margin, so the card holds it twice over. With `--mmproj-on-demand` the server keeps the projector's weights in ordinary (page-locked) RAM instead. When a request arrives that contains pictures the server has not seen in its cache, it makes room on the card (the same shrink of the expert cache that a layer-major prefill already uses), copies the projector in (about 80 ms), turns every picture of the request into numbers, throws the projector off the card again, gives the room back and continues with the normal text processing. Between requests nothing of the projector is on the card.

### 3. Glossary

- **Projector (mmproj):** the image encoder, a separate model file used for vision.
- **Expert cache:** the part of VRAM that holds the most used experts of the mixture-of-experts model so that the card can serve them without waiting for the host RAM.
- **Margin / keep-free:** the VRAM the server promises to leave unused (here 16,000 MiB, `-fitt 16000`).
- **Layer-major pass:** the fast prefill that loads each layer's weights once for the whole prompt. It needs temporary VRAM and is refused when the margin leaves no room.
- **PP / TG:** prompt processing speed and token generation speed, in tokens per second.
- **Window:** the short time in which the projector sits on the card for one request.
- **Park / unpark:** move the projector's device side away / back.
- **Free-watch:** the server's in-process measure of the lowest free VRAM ever seen (cudaMemGetInfo).
- **BASE / NOVIS / OD:** the server with the projector resident / without any projector / with `--mmproj-on-demand`.

### 4. Pros and cons (measured / expected / unproven)

Pros:
- Between image requests the projector costs 0 MiB of VRAM: idle used memory of OD equals NOVIS (7,015 = 7,015 MiB after the text steps; +30 MiB after image requests). **Measured**, `vod2-acc-od1/od2` against `vod2-acc-novis`.
- The expert cache keeps its full size: 54 slots at sizing and 52 at idle, against 38 and 20-21 with a resident projector. **Measured**, same sessions.
- With a resident projector this card, at `-fitt 16000` and free-all, REFUSES the layer-major prefill every time (the double-counted margin leaves 750 MiB too little); with OD it is taken. On a 5.7k-token text prompt PP goes 464 -> 1,641-1,649 tok/s (3.5x). **Measured**, one prompt, two sessions per arm; the cause (margin double count, 1,109 MiB) is confirmed from the code and the log lines (`no room for ... margin 17269 MiB`).
- Outputs are not changed: with the flag off the server is identical to the pristine build (13/13 steps, A B B A); with the flag on all image answers equal the resident projector's, per turn, including the number of images encoded. **Measured.**
- The CPU option (`--no-mmproj-offload`) is 13-20x slower on the encode and 1.7-2.8x on the whole image request; OD keeps the encode on the GPU. **Measured** (VOD-0).
- It survives the failure cases tried: no room (waits, health stays up), encode failure (HTTP 500, projector parked, next request fine), timeout, 12 cancelled requests in a 120-request soak, a stock-placement server. **Measured.**

Cons and limits:
- Every image request costs a fixed +0.48..0.52 s over a resident projector (0.12 s window + 0.37 s regrow of the expert cache, which `--hybrid-moe-prefill-free-all` empties). For a 1.6 s request that is +30%, for a 7.7 s request +6%. **Measured.** It is the price of the freed VRAM, and it is paid on EVERY image turn because an image conversation is re-processed and re-encoded on each turn (also with a resident projector). Without free-all the shrink would be smaller (**expected, unproven**).
- The image tokens themselves still prefill at 370-430 tok/s on the ordinary path (phase 2 targets this, unproven).
- With any projector loaded the fused one-pass prefill stays off for all prompts (text PP 1,641 vs 2,611 tok/s without a projector); not changed by phase 1 (phase 3 candidate).
- 0.84 GiB of page-locked RAM for the life of the server. **Measured size.**
- The flag is refused with `--no-mmproj-offload`/`-mmdev none` and for projectors with audio. `-np > 1`, other models and a real 8 GiB card are not tested.
- The memcheck run needs `GGML_CUDA_DISABLE_GRAPHS=1` to read `0 errors`; with CUDA graphs on, the sanitizer reports 16 API-level graph-update events in the pristine server too. **Measured.**
- A wait for room was exercised by fault injection only (no second process held the GPU).

### 5. Recommendation

Keep it: phase 1 is worth adopting for anyone who runs `--hybrid-moe` with a projector at a crowded VRAM margin, because it gives the VRAM and the layer-major prefill back at a fixed +0.5 s per image request. Keep experimenting with phases 2 and 3 (image tokens through the pass, fused prefill with projector loaded), which are what can remove the +0.5 s and the image-prefill gap. To decide for your own card, repeat `tools/run-server-vision.sh` with `SV_CMD=acc` on `--arm base` and `--arm od` and compare the idle memory and the pass lines.

### VOD-9 (no regression) at the phase-1 closeout, tree `69384eab`

- Full build of every target (`cmake --build build/llama-cuda --config Release`, `build/vod-build-all.log`, exit 0).
- The host ctest list (`ctest --test-dir build/llama-cuda -C Release -R <the host suite's Include> -E 'tail513-(lm|bm)(-run)?$'`, run directly because the suite halts at its doc-size guard): **78/78 passed**. `test-layer-streaming-(route-topk|host-mem)$`: **2/2**. `ctest -R mtmd`: **2/2** (`test-mtmd-c-api`, `test-mtmd-impl`). `python tools/server_vision.py selftest`: PASS.
- Default-off identity at server level: VOD-2 table above (A B B A, 13/13).

## VOD-6: setup wizard downloads the projector with the model and emits the flags

Change: `setup.ps1`, `setup.sh` (projector columns of the Qwen3.6 table entries, `Test-VisionOffered` / `vision_offered`, the vision append in `Complete-RowFlags` / `complete_row_flags`, the "Vision support?" question in the settings phase, `-NoVision` / `--no-vision`, `-MmprojFile` / `--mmproj-file`, `Download-Mmproj` / `download_mmproj`, `$mmproj` / `MMPROJ` in the emitted script, the flag gate checking `--mmproj` and `--mmproj-on-demand` in the built binary), tests `tools/setup-wizard/test-vod.ps1` and `test-vod.sh`, `docs/SETUP.md`, `docs/setup-wizard-sdd.md` (D23), `docs/plans/setup-wizard-plan.md`.

Table (read from the Hugging Face tree API on 2026-11-09): `unsloth/Qwen3.6-35B-A3B-GGUF` (Q4_K_M, Q5_K_XL rows) `mmproj-BF16.gguf` 902,822,624 B; `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (Q8_K_XL row) `mmproj-BF16.gguf` 902,822,528 B.

Validation (host-only; no GPU, no download; both shells):

| Command | Result |
|---------|--------|
| `powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\test-vod.ps1` | 125 checks passed (table columns, `Test-VisionOffered` over all 8 rows, `Complete-RowFlags` with vision: exactly once, idempotent, equals the vision-off flags plus the suffix, for 3 compute capabilities per row; dry runs: default, `-NoVision`, `-MmprojFile` missing / present / contradicting, an MTP row, the dense model, the dashed spelling) |
| `bash tools/setup-wizard/test-vod.sh` | 123 checks passed (same matrix under Git Bash) |
| `test-d22.ps1` / `bash test-d22.sh` (unchanged) | 53 / 195 checks passed |
| `bash -n setup.sh`, PowerShell parser on `setup.ps1` | clean |

Decision DL-14 (autonomous, recorded here and as SDD D23): vision is offered only for the Qwen3.6 table entries whose recipe row has no `--spec-type` (R7, R8). The MTP rows (R5, R6) were never measured with a projector, the dense Qwen3.8 rows and flint are outside the acceptance scope (D7), and the wizard "emits only measured combinations"; the vision flags are labelled in the emitted script as an addition to the measured row. Reversible: yes. A real download and a Linux machine run (`ssh-mcp`) were NOT done in this task; the dry runs announce the download and the existing resume / Range / token logic is reused unchanged.

VOD-6 acceptance (plan): for every `qwen36` row x vision yes / no x both shells the emitted script has the projector flags iff vision is yes and the download list gains exactly one file of the table's size (dry-run announcement); the D22 tests pass unchanged; dry-run exit 0. MET within the stated limits.
