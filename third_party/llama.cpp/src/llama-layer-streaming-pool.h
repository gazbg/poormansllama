#pragma once

// Host-testable T-GPLR-59 prediction-pool policy. This module owns admission
// and row lifecycle decisions, but no CUDA resources or runner state.

#include "llama.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>

enum llama_ls_spec_state : uint8_t {
    LLAMA_LS_SPEC_FREE      = 0,
    LLAMA_LS_SPEC_READY     = 1,
    LLAMA_LS_SPEC_PENDING   = 2,
    LLAMA_LS_SPEC_CANCELLED = 3,
};

struct llama_ls_pool_plan {
    int32_t issue[64];
    int32_t issue_pos[64];
    int     n_issue = 0;
    int     skipped_real = 0;
    int     skipped_pool = 0;
    int     skipped_dup = 0;
    int     skipped_invalid = 0;
    int     refused_budget = 0;
    int     refused_capacity = 0;
    int     evicted = 0;
    int     stop = 0; // 0 complete, 1 budget, 2 no evictable position
};

struct llama_ls_pool_lookup {
    int ready_pos   = -1;
    int pending_pos = -1;
    int stale_pos   = -1;
};

LLAMA_API size_t llama_ls_expert_slab_bytes(ggml_type type, int64_t ne0, int64_t ne1);

LLAMA_API llama_ls_pool_plan llama_ls_pool_plan_make(const int32_t * real_experts,
                                                     const uint8_t * real_states, int real_cap,
                                                     const int32_t * pool_experts,
                                                     const uint8_t * pool_states,
                                                     const int64_t * pool_recency, int pool_cap,
                                                     int n_expert, int budget, const int32_t * proposals,
                                                     int n_proposals);

LLAMA_API llama_ls_pool_lookup llama_ls_pool_lookup_expert(const int32_t * experts,
                                                            const uint8_t * states,
                                                            const uint32_t * generations,
                                                            int cap, uint32_t generation,
                                                            int32_t expert);

LLAMA_API bool llama_ls_pool_retire_ready(int32_t * experts, uint8_t * states,
                                          int64_t * recency, uint32_t * generations,
                                          int cap, uint32_t generation, int pos);

LLAMA_API int llama_ls_pool_cancel_pending(uint8_t * states, int cap);

LLAMA_API int llama_ls_pool_begin_generation(int32_t * experts, uint8_t * states,
                                             int64_t * recency, uint32_t * generations,
                                             int cap, uint32_t generation);

LLAMA_API int llama_ls_pool_complete(int32_t * experts, uint8_t * states,
                                     int64_t * recency, uint32_t * generations,
                                     int cap, int pos);
