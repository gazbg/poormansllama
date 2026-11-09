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
