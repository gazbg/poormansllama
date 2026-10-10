# private research: Qwen3.5 dense decoder-layer streaming runner.
# This target exists only when LLAMA_LAYER_STREAMING=ON.

add_executable(llama-layer-streaming
    main.cpp
    telemetry-json.cpp
    telemetry-pool.cpp
)
target_compile_definitions(llama-layer-streaming PRIVATE LLAMA_LAYER_STREAMING)
target_include_directories(llama-layer-streaming PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_features   (llama-layer-streaming PRIVATE cxx_std_17)
target_link_libraries(llama-layer-streaming PRIVATE llama ggml)

if (GGML_CUDA)
    target_link_libraries(llama-layer-streaming PRIVATE ggml-cuda)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama-layer-streaming PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# T-GPLR-82 attribution: per-op probe for row-count dependence of the MoE expert
# matmul. The split computes the same routed experts in a 6-wide head and a 2-wide
# tail where the shipped path computes one 8-wide op, so this measures whether an
# output row changes with the batch row count. No model, no llama context.
if (LLAMA_LAYER_STREAMING)
    add_executable(op-rows-probe op-rows-probe.cpp)
    target_compile_definitions(op-rows-probe PRIVATE LLAMA_LAYER_STREAMING)
    target_compile_features   (op-rows-probe PRIVATE cxx_std_17)
    target_link_libraries(op-rows-probe PRIVATE ggml)
    if (BUILD_SHARED_LIBS)
        set_target_properties(op-rows-probe PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    # Host control: the CPU backend has its own quantized kernels, so its row-count
    # answer is independent evidence and needs no GPU and no mutex.
    add_test(
        NAME layer-streaming-op-rows-cpu
        COMMAND op-rows-probe --device CPU --json ${CMAKE_BINARY_DIR}/op-rows-cpu.json
    )
    set_tests_properties(layer-streaming-op-rows-cpu PROPERTIES
        TIMEOUT 300
        PASS_REGULAR_EXPRESSION "op-rows-probe RESULT: OK"
        FAIL_REGULAR_EXPRESSION "FAILED"
    )
endif()

# focused boundary test: one recurrent layer and one full-attention layer
# through the one-layer execution boundary (T-Q38LS-02)
if (LLAMA_LAYER_STREAMING AND GGML_CUDA)
    add_test(
        NAME qwen38-layer-boundary
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 1
                --pp-async-feed 0
                --boundary-layers 0,3
                --pp-tokens 64
                --ubatch 512
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-boundary.jsonl
    )

    # T-Q38LS-03: full one-slot 512-token prefill lifecycle (all 64 layers,
    # one host-to-device package per ubatch; with default ubatch 512 this is
    # a single-ubatch PP unless a smaller --ubatch is passed explicitly)
    add_test(
        NAME layer-slot-lifecycle
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 1
                --pp-async-feed 0
                --pp-tokens 512
                --ubatch 512
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-one-slot.jsonl
    )
    set_tests_properties(layer-slot-lifecycle PROPERTIES TIMEOUT 900)

    # T-GPLR-82: regression guard for the build_moe_ffn aggregation-loop clamp.
    # The device leg of a split top-k builds `experts` with NE[1] = top_k -
    # host_rows (6 for --experts-host 2 on Qwen3.6-35B-A3B), while the
    # view/aggregation loops used to bound by hparams.n_expert_used (8), which
    # walked one row past the tensor and aborted during graph prebuild with
    #   GGML_ASSERT(view_src == NULL || data_size == 0 ||
    #               data_size + view_offs <= ggml_nbytes(view_src))
    # (ggml.c:1787, exit 0xC0000409). This test runs the exact prebuild with
    # --experts-host 2 and stops before execution via --graph-preflight-only.
    # Contract (empirical ctest semantics, measured 2026-10-06):
    #   - FAIL_REGULAR_EXPRESSION wins when it matches, naming the assert;
    #   - PASS_REGULAR_EXPRESSION is required for a pass even at exit 0, so a
    #     vacuous run that never builds the graphs cannot pass;
    #   - a silent non-zero exit fails both ways.
    add_test(
        NAME layer-streaming-host2-preflight
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --residency moe-expert-bank
                --pp-schedule batch-major
                --residency-tg
                --experts-host 2
                --residency-tg-retain 2
                --layer-slots 2
                --kv-graph-pad 256
                --kv-cache q8
                --ctx-size 262144
                --ubatch 1024
                --tg-tokens 16
                --copy-source mmap-registered
                --graph-preflight-only
                --telemetry ${CMAKE_BINARY_DIR}/gplr82-host2-preflight.jsonl
    )
    set_tests_properties(layer-streaming-host2-preflight PROPERTIES
        TIMEOUT 300
        PASS_REGULAR_EXPRESSION "persistent row-set graphs BUILT OK"
        FAIL_REGULAR_EXPRESSION "GGML_ASSERT"
    )

    # T-Q38LS-04: four-slot persistent-graph scheduler. The run test produces
    # JSONL (64 tokens, one ubatch, 4 slots); the check test validates the
    # scheduler invariants host-side (no GPU). Fixtures make the check depend
    # on the run completing first.
    add_test(
        NAME layer-streaming-scheduler-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 4
                --pp-tokens 64
                --ubatch 64
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-sched.jsonl
    )
    set_tests_properties(layer-streaming-scheduler-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_scheduler
    )

    add_test(
        NAME layer-streaming-scheduler
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-sched.jsonl
    )
    set_tests_properties(layer-streaming-scheduler PROPERTIES
        FIXTURES_REQUIRED layer_streaming_scheduler
    )

    # T-Q38LS-04B: the full 512-token / 4-slot PP trace. The run produces JSONL;
    # the host-side check validates the scheduler invariants and rejects any
    # phase_summary with graph_builds != 0 (the pre-fix baseline had graph_builds=1).
    add_test(
        NAME layer-streaming-pp512-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 4
                --pp-tokens 512
                --ubatch 512
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-pp512.jsonl
    )
    set_tests_properties(layer-streaming-pp512-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_pp512
    )

    add_test(
        NAME layer-streaming-pp512
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-pp512.jsonl
    )
    set_tests_properties(layer-streaming-pp512 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_pp512
    )

    # T-Q38LS-05: PP(512) -> TG(128 greedy) transition. The run produces JSONL
    # with PP and TG phase summaries and the generated-token count; the host
    # check validates the scheduler invariants across the transition and
    # rejects any phase_summary with graph_builds != 0 (TG must use the
    # prebuilt tg_u1 set, not build in the timed region).
    add_test(
        NAME layer-streaming-transition-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 4
                --pp-tokens 512
                --ubatch 512
                --tg-tokens 128
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-pp-tg.jsonl
    )
    set_tests_properties(layer-streaming-transition-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_transition
    )

    add_test(
        NAME layer-streaming-transition
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-pp-tg.jsonl
    )
    set_tests_properties(layer-streaming-transition PROPERTIES
        FIXTURES_REQUIRED layer_streaming_transition
    )

    # T-Q38LS-06: telemetry validator. The positive test re-validates the
    # transition trace with strict field/sequence checks and writes
    # q38ls-pp-tg.jsonl.summary.json (the raw profile artifact). The negative
    # test feeds a malformed file and requires exit != 0 (no partial report).
    add_test(
        NAME layer-streaming-telemetry
        COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/q38ls-pp-tg.jsonl
    )
    set_tests_properties(layer-streaming-telemetry PROPERTIES
        FIXTURES_REQUIRED layer_streaming_transition
    )

    # Negative test: incomplete/missing telemetry must fail validation (exit
    # non-zero, no report written). A nonexistent path exercises the
    # "incomplete telemetry fails validation" branch. WILL_FAIL makes this
    # CTest target pass only if the validator exits non-zero.
    add_test(
        NAME layer-streaming-telemetry-negative
        COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/q38ls-does-not-exist.jsonl
    )
    set_tests_properties(layer-streaming-telemetry-negative PROPERTIES
        WILL_FAIL TRUE
    )

    # T-Q38LM-01: layer-major two-chunk boundary probe. One selected logical
    # layer (layer 0) processes chunk 0 then chunk 1 through non-aliasing
    # full-prompt activation-slab slices, reusing one package generation
    # across both chunks. The run produces JSONL (with act_bind + chunk
    # fields); the host-side check validates the layer-major invariants
    # (ascending chunk order per layer, non-aliasing slab binds).
    add_test(
        NAME layer-streaming-lm-boundary-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --boundary-layers 0
                --pp-tokens 1024
                --ubatch 512
                --pp-no-logits
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-boundary.jsonl
    )
    set_tests_properties(layer-streaming-lm-boundary-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_lm_boundary
    )

    add_test(
        NAME layer-streaming-lm-boundary
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-boundary.jsonl
    )
    set_tests_properties(layer-streaming-lm-boundary PROPERTIES
        FIXTURES_REQUIRED layer_streaming_lm_boundary
    )

    # T-Q38LM-02: full two-chunk layer-major PP pass. All 64 logical layers are
    # processed in layer-major order (chunk 0 then chunk 1 per layer), each
    # package copied once and reused across both chunks, 4-slot lookahead in
    # logical-layer order, one greedy TG token for the schedule-consistency
    # check. The run produces JSONL; the host-side check validates the
    # layer-major invariants (128 computes chunk-ordered, 64 package lifecycles,
    # no timed graph builds).
    add_test(
        NAME layer-streaming-layer-major-1024-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --pp-tokens 1024
                --ubatch 512
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-1024.jsonl
    )
    set_tests_properties(layer-streaming-layer-major-1024-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_lm_1024
    )

    add_test(
        NAME layer-streaming-layer-major-1024
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-1024.jsonl
    )
    set_tests_properties(layer-streaming-layer-major-1024 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_lm_1024
    )

    # T-Q38PERF-03: pinned host staging uses one buffer per slot. The fixture
    # verifies the opt-in source path has the same layer-major lifecycle.
    add_test(
        NAME layer-streaming-pinned-1024-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --copy-source pinned
                --pp-tokens 1024
                --ubatch 512
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-pinned-1024.jsonl
    )
    set_tests_properties(layer-streaming-pinned-1024-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_pinned_1024
    )

    add_test(
        NAME layer-streaming-pinned-1024
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-pinned-1024.jsonl
    )
    set_tests_properties(layer-streaming-pinned-1024 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_pinned_1024
    )

    add_test(
        NAME layer-streaming-scheduler-256-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 4
                --pp-tokens 256
                --ubatch 512
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-sched256.jsonl
    )
    set_tests_properties(layer-streaming-scheduler-256-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_scheduler_256
    )

    add_test(
        NAME layer-streaming-slot-mixed-5-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --layer-slots 5
                --pp-tokens 64
                --ubatch 64
                --pp-no-logits
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-slots5.jsonl
    )
    set_tests_properties(layer-streaming-slot-mixed-5-run PROPERTIES
        TIMEOUT 900
        FIXTURES_SETUP layer_streaming_slot_mixed_5
    )

    add_test(
        NAME layer-streaming-slot-mixed-5
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-slots5.jsonl
    )
    set_tests_properties(layer-streaming-slot-mixed-5 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_slot_mixed_5
    )

    # T-Q38LS-04B: four-ubatch 256-token PP exercises the recurrent rs_z
    # variant transition (fresh 0 -> continuing -1) inside the timed region.
    # Same ubatch as the other -256 scheduler run; the checker consumes its
    # telemetry.
    add_test(
        NAME layer-streaming-scheduler-256
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-sched256.jsonl
    )
    set_tests_properties(layer-streaming-scheduler-256 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_scheduler_256
    )

    # T-Q38LM-05: host-only layer-major telemetry fixtures (no GPU, no lock).
    # A static 40-chunk / 64-layer / 4-slot layer-major PP trace and two
    # deliberately malformed variants (a swapped-chunk order and a premature
    # slot re-copy). The positive fixture must pass; the two negatives must
    # be rejected (WILL_FAIL passes only when the validator exits non-zero).
    # Generated by gen-lm40-fixtures.py (checked in; regenerate only when the
    # validator's 40-chunk ordering rule changes).
    add_test(
        NAME layer-streaming-lm40-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lm40-positive.jsonl
    )

    add_test(
        NAME layer-streaming-lm40-sampled-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lm40-sampled-positive.jsonl
    )

    add_test(
        NAME layer-streaming-lm40-badorder
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lm40-badorder.jsonl
    )
    set_tests_properties(layer-streaming-lm40-badorder PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-lm40-premature
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lm40-premature.jsonl
    )
    set_tests_properties(layer-streaming-lm40-premature PROPERTIES WILL_FAIL TRUE)

    # T-Q38CTX-01: 513-token manifest fixtures (host-only, no GPU): chunk
    # layout [512, 1]. The positive fixture must pass; a chunk id outside the
    # manifest and a missing final chunk at stride 1 must be rejected.
    add_test(
        NAME layer-streaming-tail513-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tail513-positive.jsonl
    )

    add_test(
        NAME layer-streaming-tail513-badchunk
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tail513-badchunk.jsonl
    )
    set_tests_properties(layer-streaming-tail513-badchunk PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-tail513-missing
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tail513-missing.jsonl
    )
    set_tests_properties(layer-streaming-tail513-missing PROPERTIES WILL_FAIL TRUE)

    # T-Q38CAP-01: stride-1 completeness on a 3-chunk manifest [512, 512, 1].
    # The positive fixture passes; an interior missing chunk (a hole) must be
    # rejected even though chunk 0 and the final chunk are present.
    add_test(
        NAME layer-streaming-tail1025-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tail1025-positive.jsonl
    )

    add_test(
        NAME layer-streaming-tail1025-hole
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tail1025-hole.jsonl
    )
    set_tests_properties(layer-streaming-tail1025-hole PROPERTIES WILL_FAIL TRUE)

    # T-GPLR-19: the byte_account TOTAL-RESIDENCY-CAP rule (host-only, no GPU).
    # byte_account now carries `row_cache` (the per-layer cache's provisioned
    # private VRAM) and `residency_cap` (--residency-budget-mib in bytes, 0 =
    # no explicit cap), so bounded residency + cache <= cap is checkable from
    # the JSONL alone and the cache figure must agree with the retention policy
    # the trace declares. The two positives are real recorded configurations
    # (arm C1's 16384 MiB budget with the K=4 cache; and a budget-free run, where
    # containment does not apply). The three negatives are tamper variants the
    # checker must reject: arm C4's configuration asserted as a completed run
    # (the runner refuses it at bring-up), a cap field that is not the flag it
    # claims, and cache bytes in a trace whose retention is OFF.
    add_test(
        NAME layer-streaming-byte-account-cap-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/byte-account-cap-positive.jsonl
    )

    add_test(
        NAME layer-streaming-byte-account-uncapped
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/byte-account-uncapped.jsonl
    )

    add_test(
        NAME layer-streaming-byte-account-cap-over
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/byte-account-cap-over.jsonl
    )
    set_tests_properties(layer-streaming-byte-account-cap-over PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-byte-account-cap-mismatch
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/byte-account-cap-mismatch.jsonl
    )
    set_tests_properties(layer-streaming-byte-account-cap-mismatch PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-byte-account-cache-noretain
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/byte-account-cache-noretain.jsonl
    )
    set_tests_properties(layer-streaming-byte-account-cache-noretain PROPERTIES WILL_FAIL TRUE)

    # T-GPLR-22: the REFUSAL trace rule (host-only, no GPU). A trace carrying a
    # structured `run_refused` is a refusal, not a run: it must have no
    # phase_summary and no run_end, and its reason must be one of the enum. A
    # `run_start` is ADMISSIBLE -- the PRE-TIMED ABORT shape (see the aborted
    # fixtures below) -- so it is neither required nor forbidden. The positive
    # fixture is the shape a refused run now writes (the package catalog
    # ls.init emits, then run_refused); the three negatives are a reason outside
    # the enum, a missing reason field, and a refusal contradicted by a
    # completed phase (run_start + phase_summary). The fourth negative is the
    # PRE-EVENT refusal shape (a bare pkg_member fragment): it must still be
    # REJECTED, which is the non-breaking claim made falsifiable -- old refusal
    # traces are not retroactively accepted, they are simply still unreadable.
    add_test(
        NAME layer-streaming-run-refused-positive
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-positive.jsonl
    )

    add_test(
        NAME layer-streaming-run-refused-badreason
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-badreason.jsonl
    )
    set_tests_properties(layer-streaming-run-refused-badreason PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-run-refused-noreason
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-noreason.jsonl
    )
    set_tests_properties(layer-streaming-run-refused-noreason PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-run-refused-with-run-start
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-with-run-start.jsonl
    )
    set_tests_properties(layer-streaming-run-refused-with-run-start PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME layer-streaming-run-refused-legacy
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-legacy.jsonl
    )
    set_tests_properties(layer-streaming-run-refused-legacy PROPERTIES WILL_FAIL TRUE)

    # T-GPLR-22 (PRE-SINK): the capacity/handoff preflights refuse BEFORE
    # ls.init opens the sink, so their trace is a SINGLE run_refused record --
    # no pkg_member catalog. This pins the shape against the EXISTING refusal
    # rule (no second rule was added) and carries the real 50k `--handoff`
    # refusal numbers from 2026-09-14. The unknown-reason direction of the
    # non-breaking claim is carried once, by run-refused-badreason: the enum
    # check is shape-independent and the enum is one shared array.
    add_test(
        NAME layer-streaming-run-refused-presink
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-presink.jsonl
    )

    # T-GPLR-22 (PRE-TIMED ABORT): the SECOND admissible refusal shape -- the run
    # started writing and a pre-timed check stopped it, so run_start is present
    # but no phase completed and run_refused is terminal. The positive carries the
    # real arm's arithmetic and its graph_footprint; no rule reads graph_footprint,
    # so the shape is pinned once rather than once per reason. init_failed is a
    # second real shape (the pre-work single line). The WILL_FAIL negative pins
    # that run_refused is TERMINAL: a run_end after it is still a contradiction,
    # which is the clause that replaced the old blanket run_start rejection.
    add_test(
        NAME layer-streaming-run-refused-aborted
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-aborted.jsonl
    )

    add_test(
        NAME layer-streaming-run-refused-init-failed
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-init-failed.jsonl
    )

    add_test(
        NAME layer-streaming-run-refused-with-run-end
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/run-refused-with-run-end.jsonl
    )
    set_tests_properties(layer-streaming-run-refused-with-run-end PROPERTIES WILL_FAIL TRUE)


    # T-Q38CTX-01: 1-token layer-major PP + 1 TG (single 1-token chunk, the
    # smallest legal layout). This is an opt-in small-ubatch override on top
    # of the new default (512), kept so the tiny-token multi-ubatch path is
    # still exercised instead of collapsing into one oversized chunk.
    add_test(
        NAME layer-streaming-lm-tiny1-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --pp-tokens 1
                --ubatch 64
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry-chunk-stride 1
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-tiny1.jsonl
    )
    set_tests_properties(layer-streaming-lm-tiny1-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_lm_tiny1
    )

    add_test(
        NAME layer-streaming-lm-tiny1
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-lm-tiny1.jsonl
    )
    set_tests_properties(layer-streaming-lm-tiny1 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_lm_tiny1
    )

    # T-Q38CTX-01: 511-token single sub-full chunk run (one chunk of 511, the
    # sub-full-tail shape when the prompt is under the ubatch size).
    add_test(
        NAME layer-streaming-lm511-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --pp-tokens 511
                --ubatch 511
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry-chunk-stride 1
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-lm511.jsonl
    )
    set_tests_properties(layer-streaming-lm511-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_lm511
    )

    add_test(
        NAME layer-streaming-lm511
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-lm511.jsonl
    )
    set_tests_properties(layer-streaming-lm511 PROPERTIES
        FIXTURES_REQUIRED layer_streaming_lm511
    )

    # T-Q38CTX-01: 513-token layer-major PP + 1 TG GPU run. Chunks [512, 1],
    # stride 1 (both chunk timings emitted), manifest cross-checked by the
    # host checker. The first greedy ID must equal the batch-major 513 run
    # (verified by comparing the tg_tokens events of both traces in the
    # GPU window; the CTest pair provides the two traces).
    add_test(
        NAME layer-streaming-tail513-lm-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule layer-major
                --pp-tokens 513
                --ubatch 512
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry-chunk-stride 1
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-lm513.jsonl
    )
    set_tests_properties(layer-streaming-tail513-lm-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_tail513_lm
    )

    add_test(
        NAME layer-streaming-tail513-lm
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-lm513.jsonl
    )
    set_tests_properties(layer-streaming-tail513-lm PROPERTIES
        FIXTURES_REQUIRED layer_streaming_tail513_lm
    )

    add_test(
        NAME layer-streaming-tail513-bm-run
        COMMAND llama-layer-streaming
                --model-env Q38LS_MODEL_PATH
                --pp-schedule batch-major
                --pp-tokens 513
                --ubatch 512
                --layer-slots 4
                --pp-no-logits
                --tg-tokens 1
                --greedy
                --no-mtp
                --allow-full-residency
                --telemetry ${CMAKE_BINARY_DIR}/q38ls-bm513.jsonl
    )
    set_tests_properties(layer-streaming-tail513-bm-run PROPERTIES
        TIMEOUT 1200
        FIXTURES_SETUP layer_streaming_tail513_bm
    )

    add_test(
        NAME layer-streaming-tail513-bm
        COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q38ls-bm513.jsonl
    )
    set_tests_properties(layer-streaming-tail513-bm PROPERTIES
        FIXTURES_REQUIRED layer_streaming_tail513_bm
    )
endif()

# T-GPLR-05: host-only capability probe and policy-refusal tests (no CUDA).
# The capability probe and every refusal path run before llama_backend_init,
# so these tests need no GPU lock and no real model.
if (LLAMA_LAYER_STREAMING)
    # Fixture generator: writes a tiny unsupported-architecture GGUF.
    add_test(
        NAME layer-streaming-make-unsupported-fixture
        COMMAND make-unsupported-fixture ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )

    # Deterministic capability result for an unsupported architecture.
    add_test(
        NAME layer-streaming-probe-unsupported
        COMMAND llama-layer-streaming --probe-capability --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )
    set_tests_properties(layer-streaming-probe-unsupported PROPERTIES
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # Forced dense-layer on an unsupported architecture must refuse loudly
    # (host-only, before any CUDA init). WILL_FAIL makes the test pass only
    # when the runner exits non-zero.
    # T-GPLR-12: explicit --mtp under the auto handoff default with a dense
    # (non-moe-expert-bank) policy must still refuse loudly: auto's fit
    # decision cannot silently satisfy the mtp-requires-handoff contract
    # (dense --mtp must pass --handoff explicitly, T-Q38HG-06).
    add_test(
        NAME layer-streaming-dense-mtp-auto-refusal
        COMMAND llama-layer-streaming --residency dense-layer --mtp
                --pp-schedule batch-major --tg-tokens 2
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )
    set_tests_properties(layer-streaming-dense-mtp-auto-refusal PROPERTIES
        PASS_REGULAR_EXPRESSION "--mtp requires --handoff"
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    add_test(
        NAME layer-streaming-refuse-unsupported-dense
        COMMAND llama-layer-streaming --residency dense-layer --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf --pp-tokens 4
    )
    set_tests_properties(layer-streaming-refuse-unsupported-dense PROPERTIES
        WILL_FAIL TRUE
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-10: --mtp + --handoff under moe-expert-bank is a loud exit-2
    # refusal (handoff TG is stock llama.cpp decode without a draft loop; the
    # MoE MTP verify path requires the streaming executor). Fires from the
    # resolved policy before the arch-match gates, so the unsupported fixture
    # reaches it. Host-only, before any CUDA init.
    add_test(
        NAME layer-streaming-refuse-mtp-handoff-moe
        COMMAND llama-layer-streaming --residency moe-expert-bank --mtp --handoff
                --tg-tokens 4
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )
    set_tests_properties(layer-streaming-refuse-mtp-handoff-moe PROPERTIES
        WILL_FAIL TRUE
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-10/T-GPLR-12: the layer-major --tg-tokens > 3 cap stays for an
    # explicitly handoff-free request (exit 2 before the arch read; auto
    # passes the parse gate and re-enforces the cap at the preflight).
    add_test(
        NAME layer-streaming-lm-tg-cap
        COMMAND llama-layer-streaming --pp-schedule layer-major --no-handoff --tg-tokens 4
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
                --pp-tokens 4 --ubatch 512
    )
    set_tests_properties(layer-streaming-lm-tg-cap PROPERTIES
        WILL_FAIL TRUE
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-10: ...and lifts when --handoff is active (post-handoff TG is
    # stock llama_decode regardless of the PP schedule). The cap gate sits
    # before the arch read, so passing it is proven by reaching the next
    # stable refusal (forced dense-layer on the unsupported arch).
    add_test(
        NAME layer-streaming-lm-tg-cap-lift
        COMMAND llama-layer-streaming --residency dense-layer --handoff
                --pp-schedule layer-major --tg-tokens 4
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
                --pp-tokens 4 --ubatch 512
    )
    set_tests_properties(layer-streaming-lm-tg-cap-lift PROPERTIES
        PASS_REGULAR_EXPRESSION "--residency dense-layer requires architecture"
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-10: dense --mtp + --handoff stays legal (T-Q38HG-06) -- proven
    # by reaching the arch check instead of a handoff/mtp refusal.
    add_test(
        NAME layer-streaming-dense-mtp-handoff-legal
        COMMAND llama-layer-streaming --residency dense-layer --mtp --handoff
                --pp-schedule batch-major
                --tg-tokens 4
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )
    set_tests_properties(layer-streaming-dense-mtp-handoff-legal PROPERTIES
        PASS_REGULAR_EXPRESSION "--residency dense-layer requires architecture"
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-10: dense attention-only handoff stays legal (T-Q38HG-04).
    add_test(
        NAME layer-streaming-dense-attn-only-legal
        COMMAND llama-layer-streaming --residency dense-layer --handoff
                --handoff-attention-only
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf
    )
    set_tests_properties(layer-streaming-dense-attn-only-legal PROPERTIES
        PASS_REGULAR_EXPRESSION "--residency dense-layer requires architecture"
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-16: partial handoff is an moe-expert-bank-only mode.
    add_test(
        NAME layer-streaming-refuse-partial-dense
        COMMAND llama-layer-streaming --residency dense-layer --handoff-partial 20
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf --pp-tokens 4
    )
    set_tests_properties(layer-streaming-refuse-partial-dense PROPERTIES
        PASS_REGULAR_EXPRESSION "--handoff-partial requires --residency moe-expert-bank"
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # T-GPLR-16: --handoff-partial 0 leaves no bank resident — loud refusal.
    add_test(
        NAME layer-streaming-refuse-partial-zero
        COMMAND llama-layer-streaming --residency moe-expert-bank --handoff-partial 0
                --model ${CMAKE_BINARY_DIR}/unsupported-arch-fixture.gguf --pp-tokens 4
    )
    set_tests_properties(layer-streaming-refuse-partial-zero PROPERTIES
        WILL_FAIL TRUE
        DEPENDS layer-streaming-make-unsupported-fixture
    )

    # auto mode on a real unsupported model must select `off` with the
    # capability reason (SDD section 8) and run the normal llama.cpp path.
    # This executes a real model load, so it registers only when a fixture
    # model path was exported at configure time (bake-at-configure convention:
    # Q38LS_UNSUPPORTED_MODEL_PATH, e.g. a small non-Qwen3.5/3.5-MoE GGUF).
    if (WIN32 AND DEFINED ENV{Q38LS_UNSUPPORTED_MODEL_PATH})
        add_test(
            NAME layer-streaming-auto-unsupported-off
            COMMAND llama-layer-streaming --residency auto
                    --model-env Q38LS_UNSUPPORTED_MODEL_PATH
                    --pp-tokens 8 --tg-tokens 1
                    --telemetry ${CMAKE_BINARY_DIR}/q36-auto-off.jsonl
        )
        add_test(
            NAME layer-streaming-auto-unsupported-off-check
            COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/q36-auto-off.jsonl
        )
        set_tests_properties(layer-streaming-auto-unsupported-off-check PROPERTIES
            DEPENDS layer-streaming-auto-unsupported-off
        )
    endif()

    # T-GPLR-05 follow-up: off-mode audit traces are first-class telemetry.
    # The host checker validates their schema; streaming events inside an off
    # trace are a loud failure, as is invalid run metadata.
    add_test(
        NAME layer-streaming-check-off-audit-valid
        COMMAND llama-layer-streaming --check-telemetry
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/off-audit-valid.jsonl
    )
    add_test(
        NAME layer-streaming-check-off-audit-rejects-slot-event
        COMMAND llama-layer-streaming --check-telemetry
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/off-audit-bad-slot-event.jsonl
    )
    add_test(
        NAME layer-streaming-check-off-audit-rejects-bad-meta
        COMMAND llama-layer-streaming --check-telemetry
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/off-audit-bad-meta.jsonl
    )
    set_tests_properties(layer-streaming-check-off-audit-rejects-slot-event PROPERTIES
        WILL_FAIL TRUE
    )
    set_tests_properties(layer-streaming-check-off-audit-rejects-bad-meta PROPERTIES
        WILL_FAIL TRUE
    )

    # T-GPLR-53 A2: a trace carrying the new recon-gated `embed_chunk` event must
    # pass BOTH host validators (forward-compatibility tolerance measured, not
    # assumed). The fixture is copied into the build dir first: validate-telemetry
    # writes <path>.summary.json beside its input.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/gplr53-embed-chunk.jsonl
        ${CMAKE_BINARY_DIR}/gplr53-embed-chunk.jsonl
        COPYONLY
    )
    add_test(
        NAME layer-streaming-check-gplr53-embed-chunk
        COMMAND llama-layer-streaming --check-telemetry
                ${CMAKE_BINARY_DIR}/gplr53-embed-chunk.jsonl
    )
    add_test(
        NAME layer-streaming-validate-gplr53-embed-chunk
        COMMAND llama-layer-streaming --validate-telemetry
                ${CMAKE_BINARY_DIR}/gplr53-embed-chunk.jsonl
    )

    # T-GPLR-54 A2: a trace carrying the default-off `tg_logits` readout must
    # pass BOTH host validators (forward-compatibility tolerance measured, not
    # assumed). Same build-dir copy pattern as the gplr53 fixture above.
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/gplr54-tg-logits.jsonl
        ${CMAKE_BINARY_DIR}/gplr54-tg-logits.jsonl
        COPYONLY
    )
    add_test(
        NAME layer-streaming-check-gplr54-tg-logits
        COMMAND llama-layer-streaming --check-telemetry
                ${CMAKE_BINARY_DIR}/gplr54-tg-logits.jsonl
    )
    add_test(
        NAME layer-streaming-validate-gplr54-tg-logits
        COMMAND llama-layer-streaming --validate-telemetry
                ${CMAKE_BINARY_DIR}/gplr54-tg-logits.jsonl
    )

    # T-GPLR-59: prediction-pool telemetry fixtures (host-only, no GPU). Both
    # host validators must accept the two positives -- the pool lifecycle trace
    # (issue, READY hit, PENDING demand fallback, terminal closure) and the
    # pool-less legacy trace (default-off immunity) -- and reject every
    # negative, each of which breaks exactly one pool rule: init capacity,
    # run_start without a valid spec_pool_init, proposal conservation,
    # stop/refusal pairing, cumulative counters, spec-vs-fetch hit agreement,
    # a pool hit overlapping a real resident, pool hits in a pool-off trace, a
    # missing pool_hit_ids field, a missing spec_pool key, and an open
    # submitted-total closure. Fixtures are copied into the build dir first:
    # validate-telemetry writes <path>.summary.json beside its input.
    set(POOL_FIXTURES_POSITIVE
        pool-positive
        pool-legacy-off
    )
    set(POOL_FIXTURES_NEGATIVE
        pool-init-bad-capacity
        pool-run-start-no-init
        pool-conservation
        pool-stop-mismatch
        pool-totals-break
        pool-hit-mismatch
        pool-hit-resident
        pool-off-with-hits
        pool-fetch-missing-hits
        pool-event-missing-key
        pool-final-open
        pool-duplicate-run-start
        pool-before-rowset-init
        pool-malformed-numeric
    )
    foreach(f IN LISTS POOL_FIXTURES_POSITIVE POOL_FIXTURES_NEGATIVE)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${f}.jsonl
            ${CMAKE_BINARY_DIR}/${f}.jsonl
            COPYONLY
        )
    endforeach()
    foreach(f IN LISTS POOL_FIXTURES_POSITIVE)
        add_test(
            NAME layer-streaming-check-${f}
            COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        add_test(
            NAME layer-streaming-validate-${f}
            COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
    endforeach()
    foreach(f IN LISTS POOL_FIXTURES_NEGATIVE)
        add_test(
            NAME layer-streaming-check-${f}
            COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        set_tests_properties(layer-streaming-check-${f} PROPERTIES WILL_FAIL TRUE)
        add_test(
            NAME layer-streaming-validate-${f}
            COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        set_tests_properties(layer-streaming-validate-${f} PROPERTIES WILL_FAIL TRUE)
    endforeach()

    # T-GPLR-63 A: the handoff-assertion fixtures. The T-GPLR-12 rule --
    # "run_start declared handoff_mode on/auto-on but no handoff events followed" --
    # was enforced in both validators and pinned by NOTHING: the only two violating
    # traces on disk (gplr43-diag.jsonl, gplr49-4k-nopw-control.jsonl) are recorded
    # artifacts of past GPU sessions, not CTest arms, so deleting both assertion sites
    # would have failed no test in this suite. These two arms are that pin.
    #
    # A positive/negative PAIR, and the positive half is not optional bookkeeping.
    # Both fixtures are the SAME recorded prefill-only trace (gen-gplr63-fixtures.py
    # changes handoff_mode "auto-off"->"auto-on", adds tg_tokens:0, and drops any
    # handoff event). The control must be ACCEPTED, which proves three things at once:
    # the negative is rejected because of the lie and not because of the fixture's
    # shape, each negative failure is a single isolated violation rather than an
    # unrelated rule firing first, and the T-GPLR-63 B `tg_tokens` key is
    # forward-compatible with both readers (measured forward-compatibility, not
    # assumed). Regenerate with the generator's docstring; the source trace must
    # itself pass both validators.
    set(HANDOFF_FIXTURES_POSITIVE
        handoff-declared-on-control
    )
    set(HANDOFF_FIXTURES_NEGATIVE
        handoff-declared-on-no-event
    )
    foreach(f IN LISTS HANDOFF_FIXTURES_POSITIVE HANDOFF_FIXTURES_NEGATIVE)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${f}.jsonl
            ${CMAKE_BINARY_DIR}/${f}.jsonl
            COPYONLY
        )
    endforeach()
    foreach(f IN LISTS HANDOFF_FIXTURES_POSITIVE)
        add_test(
            NAME layer-streaming-check-${f}
            COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        add_test(
            NAME layer-streaming-validate-${f}
            COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
    endforeach()
    foreach(f IN LISTS HANDOFF_FIXTURES_NEGATIVE)
        add_test(
            NAME layer-streaming-check-${f}
            COMMAND llama-layer-streaming --check-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        set_tests_properties(layer-streaming-check-${f} PROPERTIES WILL_FAIL TRUE)
        add_test(
            NAME layer-streaming-validate-${f}
            COMMAND llama-layer-streaming --validate-telemetry ${CMAKE_BINARY_DIR}/${f}.jsonl
        )
        set_tests_properties(layer-streaming-validate-${f} PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()
