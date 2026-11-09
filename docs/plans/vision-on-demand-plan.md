# Plan: vision projector on demand (park / unpark), VOD

Status: planned (2026-11-09). VOD-0 is `ready`; VOD-1 .. VOD-5 are `planned` and become `ready` in order.
Mode: NOT autonomous. The owner decided the points in "Owner decisions" below in chat on 2026-11-09; everything else is the Planner's reading and is marked as such.
Related: [llama-server-integration.md](llama-server-integration.md) (G31 hybrid server, G32 layer-major pass, G42 fuse), [setup-wizard-plan.md](setup-wizard-plan.md), [../layer-residency-sdd.md](../layer-residency-sdd.md).

## Goal

A `llama-server` that is started with a vision projector (`--mmproj`) keeps the projector OUT of VRAM except while it encodes the images of a request, and the image tokens it produces are prefilled through the layer-major pass instead of the slow ordinary path.

Two goals, weighted equally (owner, 2026-11-09):

- **(a) Text-turn VRAM.** Today the projector's weights and its worst-case compute buffer stay on CUDA0 for the server's whole life, and under `--hybrid-moe` the expert cache is sized around them. Between image requests that memory should be the expert cache's.
- **(b) Image-request speed.** The image request itself must not pay for the saved VRAM (encode on the GPU, not the CPU), and its image tokens must not run at the copy-bound ordinary-path rate.

Charter fit (AGENTS.md): this is VRAM minimisation (priority 2) that must not cost PP speed (priority 1), and it must keep working when VRAM is crowded (priority 3: the 16,000 MiB keep-free rule is the crowded condition).

## What the code does today (read 2026-11-09, vendored tree at the repo's pinned head)

| # | Fact | Where |
|---|------|-------|
| F1 | The projector is loaded once at start-up and freed only at server exit (or when the server sleeps). | `tools/server/server-context.cpp:2084` (`mtmd_init_from_file`), `:1294` (`destroy()`) |
| F2 | The weights live in a backend buffer and the clip scheduler's compute buffer is reserved at a worst-case warm-up size (`reserve_compute_meta`), so both are resident for the process lifetime. | `tools/mtmd/clip.cpp:4213`, `:4375-4390`, `:217` |
| F3 | The projector's estimated device bytes (`mtmd_get_memory_usage`) are ADDED to `fit_params_target`. The hybrid engine then reads `fit_params_target[0]` as its keep-free margin (`hybrid_size_cache`, `llama_hybrid_set_margin`). The speculative-context estimate is released from the margin once that context exists (`spec_fit_added`); the projector's is NOT. | `server-context.cpp:1507-1531`, `:1357`, `:1370`, `:2042-2053` |
| F4 | CPU vision already exists: `--no-mmproj-offload`, `-mmdev`. | `common/arg.cpp:2940-2964`, `common/common.h:608` |
| F5 | An image chunk is encoded when the slot reaches it, on the queue's worker thread (`yield_to_queue`), which serves only metrics tasks meanwhile; slot compute stays serialised. | `server-context.cpp:5320-5325`, `tools/server/server-queue.cpp:222` |
| F6 | The embeddings of an image chunk are decoded by `mtmd_helper_decode_image_chunk` as embedding batches. | `server-context.cpp:1046-1118` |
| F7 | The layer-major pass DECLINES any batch that carries embeddings (`b.token == nullptr \|\| b.embd != nullptr`). Image tokens therefore always take the ordinary path today. | `src/llama-lm-prefill.cpp:157` |
| F8 | `--lm-prefill-fuse`, canonical mode and checkpoints are all disabled for a batch that handles an image chunk (`!has_mtmd`), and the server never checkpoints after an image chunk. | `server-context.cpp:5365`, `:5387`, `:5397`, `:5565`, `:5584` |
| F9 | The pass makes its VRAM room through `hybrid_lm_vram(ctx, need_bytes, begin)`: target = need + keep-free margin + safety; the expert cache shrinks just enough (floor `max(2, min_slots/4)`), or all the way to the floor with `--hybrid-moe-prefill-free-all`; if it still does not fit the pass is declined and logged. The cache regrows at the first decode of <= 16 tokens. | `src/llama-hybrid-api.cpp:386-446` |
| F10 | `qwen35moe` has no deepstack (only `qwen3vl`, `qwen3vlmoe`, granite do), so the image path into the LLM is plain embeddings plus M-RoPE positions. | grep of `n_deepstack` in `src/models/` |
| F11 | The follow-up turn of a conversation keeps the image in the prefix cache (chunks are keyed in `server_tokens`), so only a NEW image, or a rollback to before an image, needs an encode. | `server-context.cpp:5338-5343` (placeholder push), prompt cache logic |

Test files on disk (checked 2026-11-09):

- Model: `C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf` (21,388,319,008 B, the file `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` publishes).
- Projector: `C:/Models/ALL/other/qwen-3.6/mmproj-qwen-35b-tq3-BF16.gguf`, 902,822,528 B (0.841 GiB), SHA-256 `da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa`, which is byte-identical to `mmproj-BF16.gguf` in that Hugging Face repo (LFS oid compared via the public tree API; nothing was downloaded). `mmproj-BF16-27b-tq3.gguf` has the same hash. The local `mmproj-F16.gguf` and `mmproj-BF16-35b-tq3.gguf` are NOT the MTP repo's files (they are hash-identical to the non-MTP repo's `mmproj-F16.gguf` / `mmproj-BF16.gguf`, next line); do not use them as the MTP test projector.
- The non-MTP repo `unsloth/Qwen3.6-35B-A3B-GGUF` (the wizard's Q4_K_M / Q5_K_XL source) publishes its own `mmproj-BF16.gguf` (902,822,624 B, SHA-256 `356dfaa3...334d` = the local `mmproj-BF16-35b-tq3.gguf`) and `mmproj-F16.gguf` (899,283,680 B, SHA-256 `8971ee4f...887f` = the local `mmproj-F16.gguf`): a different file from the MTP repo's.
- Pillow is not installed. Fixtures are generated with a pure-Python PNG writer (no new dependency).

## Owner decisions (chat, 2026-11-09)

| ID | Question | Choice | Reason / consequence |
|----|----------|--------|----------------------|
| D1 | Which pain? | Both: text-turn VRAM and image-request speed, equally | Phase 1 serves (a); phase 2 and 3 serve (b). |
| D2 | Mechanism | **B: park the weights in pinned host RAM**, free only the device side | No disk read per request; costs ~0.84 GiB of pinned RAM and a change inside mtmd. The CPU path stays available. |
| D3 | Image prefill through the pass | Include, as phase 2 | F7 is the blocker. |
| D4 | Unload policy | Immediately after the request's images are encoded | Maximum VRAM for text; a rollback or new image pays the unpark again. No idle timer. |
| D5 | Keep-free rule in the encode window | Strict, through the existing `hybrid_lm_vram` mechanism. The owner runs `--hybrid-moe-prefill-free-all`, so the window starts from the cache floor; without that flag the same function shrinks only as needed. | No new memory mechanism. Planner reading of the owner's question "will auto-eviction make room": yes, `hybrid_lm_vram` does exactly that (F9). |
| D6 | If room cannot be made | Wait for room, no timeout | At `-np 1` nothing in the process competes (F5), so only an outside process can cause it. Planner additions: a WARN line every ~10 s and a tunable timeout flag whose default is unlimited. `/health` keeps answering because the worker thread serves metrics. |
| D7 | Models in the acceptance contract | Qwen3.6-35B-A3B only | Phase 1 is architecture-generic by construction but is only CLAIMED for this model. |
| D8 | Switch | Own flag `--mmproj-on-demand`, independent of `--hybrid-moe`, default OFF | Default llama.cpp behaviour is unchanged with the flag off. The setup wizard turns it on. |
| D9 | Typical workload | 1-3 screenshots per request, ~500-2,000 image tokens each | Embeddings are tens of MiB at most. No video, no many-page documents in v1. |
| D10 | Acceptance bar | Performance is REPORT-ONLY. Correctness stays as hard gates. | No speed claim is made (AGENTS.md: no claim that was not defined and validated). |
| D11 | Fuse | Include `--lm-prefill-fuse` / prearm with image prompts, as phase 3 | Largest and last piece; depends on phase 2. |
| D12 | Margin double-count (F3), if phase 0 confirms it | Fix ONLY under `--mmproj-on-demand` | Existing `--mmproj` configs keep their sizing; the baseline must be recorded before and after. |
| D13 | Where the projector comes from | Same Hugging Face repo as the model; the setup wizard downloads it with the model | See VOD-5. |

## Design

### Request flow with the flag on

1. Tokenise and preprocess images on the CPU (unchanged).
2. After the slot has settled how much of the prompt the cache already holds (prefix reuse, checkpoint restore), find the media chunks that lie beyond it. If there are none, stop: no VRAM is touched.
3. Make room: `hybrid_lm_vram(need = projector weights + compute buffer, begin)`. The same shrink the layer-major pass needs; one shrink and one regrow per request when both are used.
4. Unpark: allocate the device weights buffer, copy the weights from the pinned host copy, build the clip scheduler and reserve for THIS request's image batch.
5. Encode every uncached image chunk of the request. Keep the embeddings in host RAM in the slot.
6. Park at once: free the scheduler, the device weights buffer and the clip backend's pool. `hybrid_lm_vram(end)` hands the room back; the cache regrows at the first small decode, as for a pass.
7. Prefill: text through the pass as today. Phase 1: image chunks through the ordinary path (embedding batches). Phase 2: image embeddings go through the pass too. Phase 3: also under fuse.

The projector's output is the LLM's input, so the two never need VRAM at the same time. That is what makes "unload immediately" cheap.

### Memory accounting

- With the flag on, the projector's bytes are NOT added to `fit_params_target` at start-up (it is not resident), so the cache is sized as for a server without `--mmproj`. The projector's bytes are used only as `need_bytes` of the encode window.
- The margin unit is the CUDA-runtime view (`cudaMemGetInfo`, the in-process free-watch), the unit of `--fit-target`. The external nvidia-smi poller reads a different constant offset (AGENTS.md, "VRAM in use has three values"); every VRAM figure in the results names its unit.
- Start-up must not put the projector on the device before the cache is sized: the context is initialised PARKED (weights straight into the pinned host buffer).

### Threading

The encode runs on the worker thread (`yield_to_queue`) as today. Cache shrink and regrow mutate the hybrid engine and must stay on the loop thread: do them before and after the yield. A GPU-spinning host handshake must never depend on a thread that makes CUDA calls (AGENTS.md, G5); park/unpark touches the CUDA allocator only while no decode is in flight.

### Failure contract

- No room (guard says no): wait (D6), log, never run another request's data, never encode on the CPU silently.
- Unpark or encode CUDA failure: HTTP 500 for that request, projector left parked, later requests unaffected (same shape as the pass's fault injection).
- `--mmproj-on-demand` without `--mmproj`, with `--no-mmproj-offload` / `-mmdev` CPU, or with a projector that has audio / realtime-audio contexts: refuse at start-up with a message that names the conflict (v1 is vision only).

## Acceptance contract (written before any run; criteria may not be weakened after a result)

Definitions.

- BASE: the vendored server at the pinned head, `--hybrid-moe --lm-prefill ...` (the R8 recipe of the wizard: `-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32`, plus `--hybrid-moe-prefill-free-all` where stated) with `--mmproj <projector>` and the projector resident.
- OD: the new build with BASE's flags plus `--mmproj-on-demand`.
- NOVIS: BASE without `--mmproj`.
- CPUV: BASE with `--no-mmproj-offload`.
- Fixtures: image set generated by `tools/server_vision.py` (VOD-0): solid colour blocks, big seven-segment digit strings, a 3-image mix; sizes giving ~500 / ~1,000 / ~2,000 image tokens; each with a programmatically graded question (normalised substring, plus a must-fail negative control, as the semantic-QA gate).
- One server at a time under the GPU mutex, mains power, `tools/with-gpu-lock.ps1`; the GPU is checked for foreign compute processes before and after.

Hard correctness criteria (each gates its task):

- **VOD-1 default-off identity.** The new build WITHOUT `--mmproj-on-demand` equals BASE: greedy texts and `prompt_n / cache_n / predicted_n` identical on the text smoke and on every fixture request; PP / TG within the A-A spread (A B B A, <= 3%).
- **VOD-2 embeddings identical.** The embedding floats of every fixture image produced after an unpark are byte-identical (`memcmp`) to the ones the resident projector produces, over 100 park / unpark cycles. If they are not, the root cause is found and fixed (graph shape, kernel variant); the criterion is not relaxed.
- **VOD-3 memory.** (i) Idle (no image request in flight) device memory with OD differs from NOVIS by <= 64 MiB (the projector is really gone, including its pool). (ii) 100 park / unpark cycles: idle free VRAM start == end within 64 MiB. (iii) The in-process free-watch minimum stays >= 16,000 MiB through multi-request image sessions, the encode window included. (iv) At idle the hybrid expert cache has the slot count of NOVIS (+-1), not BASE's smaller one (a MECHANISM check read from the sizing log line; not a speed claim).
- **VOD-4 answers.** Every fixture question is answered correctly by OD wherever BASE answers it correctly, and the negative control fails as designed. Multi-turn: a follow-up text turn on the image conversation has `prompt_n` equal to BASE's and ZERO encodes in OD's log; a new image encodes exactly once.
- **VOD-5 wait and failure.** With fault injection (`LLAMA_VOD_FAIL_ROOM=N`, first N room attempts say no): WARN lines appear, the request completes after the N-th attempt, `/health` answers during the wait. With `LLAMA_VOD_FAIL_ENCODE=1`: that request is HTTP 500, the projector is parked, the next request succeeds. (A ballast process is NOT used: WDDM over-commits and cannot emulate a smaller card, AGENTS.md G8.)
- **VOD-6 stability.** Soak of >= 100 mixed text / image requests at `-np 1`, with cancelled requests mid-encode: no crash, no hang, idle free VRAM start == end within 64 MiB. `compute-sanitizer --tool memcheck` on an image request: `ERROR SUMMARY: 0 errors`.
- **VOD-7 pass takes embeddings (phase 2).** The pass log (`LLAMA_LM_LOG=1`) shows the image-bearing batch TAKEN, not declined. At an equal chunk shape the pass and the ordinary path give, for the fixture prompts, a first-token top-1 logprob difference <= 0.25 nats and 64-token greedy continuations that are identical or flip first at a near tie (< 0.25 nats); the same comparison is run for the ordinary path's own shape change (512 vs 2,048) beside it, as the LM-3 work taught. The position (`pos_next`) after an image equals the ordinary path's.
- **VOD-8 fuse with images (phase 3).** The checkpoint blobs a fused image prompt writes at aligned positions outside image regions are byte-identical to the legacy path's (hash equality); `prompt_n / cache_n` and checkpoints restored / created / erased equal BASE's over the image multi-turn scenario; prearm does not change either.
- **VOD-9 no regression.** Host ctest list 78/78 plus `test-layer-streaming-(route-topk|host-mem)$` 2/2 (run directly: the suite halts at its doc-size guard, AGENTS.md); the default-off identity (VOD-1) re-run on the final build; the wizard host tests pass.

Report-only (measured, tabulated, NO threshold, NO claim): image-request time to first token and its breakdown (shrink, unpark, encode, park, regrow, prefill) for OD vs BASE vs CPUV with 1 / 2 / 3 screenshots at the three sizes; text-turn TG and PP for OD vs NOVIS vs BASE; GPU encode time vs CPU encode time; unpark cost from pinned RAM (and a reload from disk for reference); compute-buffer size; pass rate on image tokens vs the ordinary path.

Claims allowed: exactly what is measured, on this machine (RTX 5090 Laptop, Windows), this model, these flags. NOT claimed: other architectures, other projectors, audio, video, multi-GPU, `-np > 2`, a physical 8 GiB card, any speedup (D10), non-causal image attention (Gemma-style), deepstack models.

Acceptance in this plan is the OWNER's decision from the Owner Summary the final task produces; gates are soft evidence (AGENTS.md Gate Policy).

## Tasks

Order: VOD-0 -> VOD-1 -> VOD-2 -> VOD-3 (-> VOD-4 phase 2 -> VOD-5 phase 3) and VOD-6 (wizard) after VOD-3. Phases of the discussion: P0 = VOD-0, P1 = VOD-1..3, P2 = VOD-4, P3 = VOD-5. Wizard = VOD-6.

### VOD-0: measure and verify the hypotheses, build the harness

Status: ready. Dependencies: none. Commit: unfilled.

Goal. Turn the open questions into numbers before any engine edit, and create the harness every later task uses.

Scope. New: `tools/server_vision.py` (fixture generator with a pure-Python PNG writer, chat requests with images, grading, phase timings from the server log), `tools/run-server-vision.sh` (same shape as `tools/run-server-firsttok.sh`: one server under the GPU mutex, results in `build/llama-cuda/<tag>/`), `docs/plans/vision-on-demand-results.md`. Allowed engine change: log-only, env-gated (`LLAMA_VOD_LOG=1`) timing lines around the mtmd encode and the projector's device bytes; nothing else.

Non-goals. No park / unpark, no flag, no behaviour change.

Measurements.

- M1 (hypothesis H1, F3). Start BASE with and without `--mmproj` at identical flags. Read `hybrid-moe: expert cache A -> B slots ... margin M MiB` from each log. Prediction if there is NO double count: slots(NOVIS) - slots(BASE) ~= (projector weights + compute buffer) / bytes-per-slot (~67 MiB estimated: 432 MiB bank / 256 experts x 40 layers). Prediction if F3 double-counts: about twice that, and the logged margin exceeds `--fit-target`. Either result is recorded.
- M2. Projector device bytes: weights, clip compute buffer (`reserve_compute_meta` log), `mtmd_get_memory_usage` estimate, free-watch delta around the load.
- M3. GPU encode time per fixture size (warm projector), and the image token count per fixture.
- M4. CPU encode time (CPUV) for the same fixtures.
- M5. Time of an ordinary-path image prefill in BASE for the image tokens (the number phase 2 must beat).

Acceptance. M1-M5 tabulated in the results file with the exact commands, build digest, GPU pre / post probe, power state (mains), and the unit of every VRAM figure. H1 is stated as confirmed or falsified.

Validation. `bash tools/run-server-vision.sh <tag> "--phase p0" -- --exe=build/llama-cuda/bin/<build>/llama-server.exe ...` (arguments completed when the script exists). Expected: exit 0, `report.json` with the M1-M5 fields, no `CUDA error` in `server.log`. `python tools/server_vision.py --selftest` (no GPU): PNG writer round-trips, grader accepts the right answer and rejects the negative control.

State updates. `docs/currentstate.md` entry; results file section "P0".

Escalation. If M1 shows the projector costs far more than ~1.5 GiB of cache (or less than ~0.3 GiB), the value of goal (a) changes: report it and stop for the owner before VOD-1.

### VOD-1: park / unpark in mtmd and clip (library)

Status: planned. Dependencies: VOD-0. Commit: unfilled.

Goal. The projector can live parked: weights in a pinned host buffer, nothing on the device, and be brought to the device and back without changing its output.

Scope. `third_party/llama.cpp/tools/mtmd/clip.cpp`, `clip.h`, `mtmd.cpp`, `mtmd.h`: `park_on_init` in `mtmd_context_params`, `mtmd_park()`, `mtmd_unpark()`, `mtmd_is_parked()`, `mtmd_device_bytes()`; new GPU test `tests/test-mtmd-park.cpp`. Nested instructions: read `third_party/llama.cpp/AGENTS.md` first; this is a private fork, nothing is sent upstream.

Non-goals. Server changes; audio / realtime-audio contexts (park refuses them); the CPU backend path (unchanged).

Design / failure. Init parked loads the tensors straight into a pinned host buffer. Unpark allocates the device weights, copies, creates the scheduler and reserves for the batch it is given (the reserve shape must equal the resident projector's for the same images, or the embeddings can differ: VOD-2). Park frees the scheduler, the weights buffer and the clip backend's pool; the backend object may stay if its idle cost is <= 64 MiB (measured). A failed unpark leaves the context parked and returns an error.

Steps. (1) Read `clip_ctx` ownership (`buf`, `sched`, `backend_ptrs`, `mem_compute`). (2) Add the parked init and the pinned buffer. (3) Implement unpark / park. (4) Test: 100 cycles, `memcmp` of embeddings against a resident context, idle `cudaMemGetInfo` delta. (5) Time unpark and a free + reload from disk for reference (report-only).

Acceptance. VOD-2 (embeddings byte-identical over 100 cycles), the idle-bytes part of VOD-3 (i)-(ii) at library level, VOD-1 default-off identity (the new API unused changes nothing).

Validation. `cmake --build build/llama-cuda --config Release --target test-mtmd-park llama-server`; `pwsh -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag vod1 -Command "build/llama-cuda/bin/Release/test-mtmd-park.exe C:/Models/ALL/other/qwen-3.6/mmproj-qwen-35b-tq3-BF16.gguf"`. Expected: `ALL PASS`, identical embeddings in 100 / 100 cycles, idle delta <= 64 MiB. Host: ctest list 78/78, `test-mtmd-impl` and `test-mtmd-c-api` still pass.

State updates. results file "VOD-1", `docs/currentstate.md`.

Escalation. Embeddings differ after unpark: find the cause (fattn variant selection from a different reserve shape, a stale cuBLAS workspace) before touching the criterion.

### VOD-2: server flag, encode window, wait rule

Status: planned. Dependencies: VOD-1. Commit: unfilled.

Goal. `--mmproj-on-demand` runs the request flow above with a real server.

Scope. `common/common.h`, `common/arg.cpp` (flag, `--mmproj-wait-timeout`, default unlimited), `tools/server/server-context.cpp` (parked init; skip the F3 margin add when the flag is on; up-front encode window; slot-owned embedding storage; refusals), `src/llama-hybrid-api.cpp` only if `hybrid_lm_vram` needs an entry that is not tied to the pass; `docs/guide/PARAMETERS.md`, `docs/guide/USAGE.md`.

Non-goals. Pass or fuse changes (VOD-4, VOD-5); `-np > 2`; CPU fallback (D6: wait, never silent CPU).

Design. Hook point is the start of prompt processing after the cache decision (around `server-context.cpp:5295-5346`), not the lazy per-chunk encode. The existing `mtmd_batch` batching and `mtmd_batch_max_tokens` cap apply; more media chunks than one batch hold need several batches or a slot-owned vector. The shrink / regrow runs on the loop thread, unpark / encode / park inside `yield_to_queue`. Without `--hybrid-moe` the "room" is plain free device memory >= need + safety; there is no cache to shrink.

Acceptance. VOD-1 identity, VOD-3 (all four parts), VOD-4, VOD-5, VOD-6 (soak and memcheck), on `--hybrid-moe` with and without `--hybrid-moe-prefill-free-all`; a smoke on the stock placement (`-ngl 99 -ncmoe N -fit off`).

Validation. Build as above; `bash tools/run-server-vision.sh vod2-<arm> "--phase p1 ..." -- ...` once per arm (OD, BASE, NOVIS, CPUV), A B B A for the report-only timings; `tools/run-server-soak.sh` with the vision mix; the memcheck wrapper per `docs/plans/llama-server-integration.md` (G32 section). Expected: every VOD-1..6 criterion met, `ERROR SUMMARY: 0 errors`.

State updates. results file "VOD-2", `docs/currentstate.md`, guide rows.

Escalation. A fix that needs a change to a normal (flag-off) execution path is re-planned, not made.

### VOD-3: closeout of phase 1 (record and summary)

Status: planned. Dependencies: VOD-2. Commit: unfilled.

Goal. The phase-1 evidence is complete and the Owner Summary is written, so the owner can decide whether phase 1 ships before phase 2 starts.

Scope. Results file, `docs/currentstate.md`, Owner Summary (original goal, mechanism in plain words, glossary, pros and cons labelled measured / expected / unproven with evidence paths, recommendation).

Acceptance. All hard criteria VOD-1..VOD-6 and VOD-9 demonstrated with recorded commands; the report-only tables present; known gaps listed.

### VOD-4: the layer-major pass takes image embeddings (phase 2)

Status: planned. Dependencies: VOD-3. Commit: unfilled.

Goal. A prompt `[text][image][text]` is prefilled by the pass instead of the ordinary path for its image tokens.

Scope. `src/llama-lm-prefill.cpp` (the `lm_make_plan` decline at `:157`, the embed piece, positions), `src/llama-graph.cpp` / `src/models/qwen35moe.cpp` only where the piece builder needs the embedding input; `tools/server/server-context.cpp` batching of image chunks.

Non-goals. Fuse and checkpoints inside image regions (VOD-5); models other than `qwen35moe`; non-causal image attention.

Open question to resolve first (by inspection, recorded as a decision): whether `llama_batch_ext` can carry text tokens and image embeddings in ONE `llama_decode` for this model. If not, the pass takes a list of consecutive input pieces and sweeps each layer over all of them (the pass already runs several chunks per layer and segments of up to 16,384 tokens through a slab), so one bank stream still serves the whole prompt.

Other things to verify by test, not assume: M-RoPE positions of image tokens (`mtmd_image_tokens_get_decoder_pos`) reach the pass's graph unchanged; the implicit causal mask's cell addressing (`lm_block`, contiguous cells) holds across an image whose positions are compressed; image chunks beyond one `n_batch` split the same way.

Acceptance. VOD-7 and VOD-1 / VOD-3 / VOD-4 / VOD-6 re-run with the pass taking images.

Validation. Build; `LLAMA_LM_LOG=1` runs of the fixture prompts; `tools/run-server-firsttok.sh` and `tools/run-server-gen.sh` style comparisons (pass vs ordinary path at equal chunk shape, plus the ordinary path's own shape change as calibration); memcheck on an image pass. Expected as VOD-7.

Escalation. If the pass cannot take embeddings without changing the normal path, stop and bring the owner a decision with the measured cost of the ordinary path (VOD-0 M5) against the change.

### VOD-5: fuse and prearm with image prompts (phase 3)

Status: planned. Dependencies: VOD-4. Commit: unfilled.

Goal. `--lm-prefill-fuse` and `--lm-prefill-prearm` work for prompts with images: one pass per request, checkpoints taken inside it.

Scope. The server's simulated cut and checkpoint list (the `!has_mtmd` conditions of F8), the library's snapshot positions (chunk boundaries outside image regions), canonical mode (`--lm-prefill-canonical`) interaction.

Non-goals. Checkpoints INSIDE an image region (a rollback into the middle of an image cannot re-split its embeddings): the rule is boundary-only.

Acceptance. VOD-8.

Validation. Hash comparison of the checkpoint blobs (fused vs legacy path) at aligned positions on the image multi-turn scenario; `prompt_n / cache_n` and checkpoint counters equal BASE; the G42 suite on text-only prompts unchanged (hash 8 of 8). Expected as VOD-8.

Escalation. If checkpoint semantics around an image cannot be made equal to the legacy ones, fuse stays off for image prompts (as today) and the owner is told what was lost.

### VOD-6: setup wizard downloads the projector with the model and emits the flags

Status: planned. Dependencies: VOD-2 (the flag exists in the built binary). Commit: unfilled.

Goal. `setup.ps1` / `setup.sh` fetch `mmproj-BF16.gguf` from the SAME Hugging Face repo as the chosen Qwen3.6-35B-A3B file, in the parallel download phase, and the emitted server script carries `--mmproj <path> --mmproj-on-demand`.

Scope. `setup.ps1`, `setup.sh` (model table gets `Mmproj`, `MmprojBytes`; settings phase asks "vision support? [Y/n]", default yes for the `qwen36` rows, `-NoVision` / `--no-vision` suppresses it; download with the same resume logic; emission; the binary flag gate), `tools/setup-wizard/` tests (a host-only dry-run test in the style of `test-d22.ps1` / `.sh`), `docs/plans/setup-wizard-plan.md` decision entry.

Table (verified against the Hugging Face tree API on 2026-11-09): `unsloth/Qwen3.6-35B-A3B-GGUF` -> `mmproj-BF16.gguf` 902,822,624 B (Q4_K_M, Q5_K_XL rows); `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` -> `mmproj-BF16.gguf` 902,822,528 B (Q8_K_XL row).

Non-goals. `qwen38` dense and `flint` rows (no vision in v1; their rows and recipes are unchanged); changing any measured recipe's existing flags.

Acceptance. For every `qwen36` row x vision yes / no x both OSes: the emitted script has the projector flags iff vision is yes; the download list gains exactly one file of the table's size; the existing wizard tests still pass (53 + 195 checks of D22); dry run exit 0.

Validation. `pwsh -File tools/setup-wizard/test-vod.ps1` and `bash tools/setup-wizard/test-vod.sh` (to be created; host-only, no GPU, no download: `-MaxDownloadMib` / dry-run hooks as the wizard already has); `test-d22.ps1` / `.sh` unchanged results. Expected: all checks pass, `bash -n setup.sh` clean.

Escalation. A repo whose tree does not publish `mmproj-BF16.gguf`: the row falls back to `mmproj-F16.gguf` with a recorded decision; if neither exists the row gets no vision and the wizard says so.

## Risks and unknowns (stated, not resolved)

- H1 (F3) margin double-count: unverified; VOD-0 M1 decides. It would also mean today's always-resident vision loses about twice the projector's bytes of cache.
- H2 mixed token / embedding batch in one decode: unknown for this model (VOD-4 first step).
- H3 the worst-case compute buffer may be much larger than a real request needs; reserving per request in unpark may shrink the window (and cost a graph build per request). Measured in VOD-0 / VOD-1.
- H4 WDDM may not return freed device memory immediately; the idle check (VOD-3 i) is the test.
- H5 a different reserve shape can change the flash-attention variant and so the embeddings (VOD-2 catches it).
- H6 the projector may carry audio or video contexts; v1 refuses them loudly.
- H7 a rollback to before an image inside a recurrent (Gated DeltaNet) model re-encodes that image; the up-front encode must encode only chunks beyond what the cache holds.
- H8 `-np 2` with images and the contiguous-block rule (`lm_block`): not claimed in v1.
- Pinned RAM: ~0.84 GiB page-locked for the life of the server (owner accepted by choosing option B).

## Required state / document updates (every task)

`docs/currentstate.md` (observed behaviour, evidence, gaps, risk), the results file `docs/plans/vision-on-demand-results.md` (commands, build digest, GPU pre / post probe, power state, VRAM unit), `docs/guide/PARAMETERS.md` for the new flags, and each task's `Commit:` field once its verdict commit lands.

## Progress log

- 2026-11-09: discussion with the owner (four question rounds), decisions D1-D13 recorded above, code read as listed in F1-F11, public Hugging Face tree listings read (no download), local projector hashed. No engine file touched, no GPU workload run.
