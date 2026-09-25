#include "llama-layer-streaming-pool.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

size_t llama_ls_expert_slab_bytes(ggml_type type, int64_t ne0, int64_t ne1) {
    if ((unsigned) type >= (unsigned) GGML_TYPE_COUNT || ne0 <= 0 || ne1 <= 0) {
        return 0;
    }
    const size_t row = ggml_row_size(type, ne0);
    if (row == 0 || (size_t) ne1 > std::numeric_limits<size_t>::max() / row) {
        return 0;
    }
    return row * (size_t) ne1;
}

llama_ls_pool_plan llama_ls_pool_plan_make(const int32_t * real_experts,
                                           const uint8_t * real_states, int real_cap,
                                           const int32_t * pool_experts,
                                           const uint8_t * pool_states,
                                           const int64_t * pool_recency, int pool_cap,
                                           int n_expert, int budget, const int32_t * proposals,
                                           int n_proposals) {
    llama_ls_pool_plan plan;
    if (real_experts == nullptr || real_states == nullptr || pool_experts == nullptr ||
        pool_states == nullptr || pool_recency == nullptr || proposals == nullptr ||
        real_cap < 0 || pool_cap <= 0 || pool_cap > 64 || n_expert <= 0 || n_proposals <= 0) {
        return plan;
    }
    budget = budget < 0 ? 0 : budget > 64 ? 64 : budget;
    std::vector<int32_t> virtual_experts(pool_experts, pool_experts + pool_cap);
    std::vector<uint8_t> virtual_states(pool_states, pool_states + pool_cap);
    std::vector<int64_t> virtual_recency(pool_recency, pool_recency + pool_cap);
    std::unordered_set<int32_t> published;
    int64_t next_recency = 0;
    for (int64_t recency : virtual_recency) {
        next_recency = std::max(next_recency, recency + 1);
    }

    for (int i = 0; i < n_proposals; ++i) {
        const int32_t expert = proposals[i];
        if (expert < 0 || expert >= n_expert) {
            ++plan.skipped_invalid;
            continue;
        }
        if (!published.insert(expert).second) {
            ++plan.skipped_dup;
            continue;
        }
        bool in_pool = false;
        for (int pos = 0; pos < pool_cap; ++pos) {
            if (virtual_states[pos] != LLAMA_LS_SPEC_FREE && virtual_experts[pos] == expert) {
                in_pool = true;
                break;
            }
        }
        if (in_pool) {
            ++plan.skipped_pool;
            continue;
        }
        bool in_real = false;
        for (int pos = 0; pos < real_cap; ++pos) {
            if (real_states[pos] != LLAMA_LS_SPEC_FREE && real_experts[pos] == expert) {
                in_real = true;
                break;
            }
        }
        if (in_real) {
            ++plan.skipped_real;
            continue;
        }
        if (plan.n_issue >= budget) {
            ++plan.refused_budget;
            plan.stop = 1;
            continue;
        }

        int selected = -1;
        for (int pos = 0; pos < pool_cap; ++pos) {
            if (virtual_states[pos] == LLAMA_LS_SPEC_FREE && virtual_experts[pos] < 0) {
                selected = pos;
                break;
            }
        }
        if (selected < 0) {
            for (int pos = 0; pos < pool_cap; ++pos) {
                if (virtual_states[pos] == LLAMA_LS_SPEC_READY &&
                    (selected < 0 || virtual_recency[pos] < virtual_recency[selected])) {
                    selected = pos;
                }
            }
        }
        if (selected < 0) {
            ++plan.refused_capacity;
            plan.stop = 2;
            continue;
        }
        if (virtual_states[selected] == LLAMA_LS_SPEC_READY) {
            ++plan.evicted;
        }
        virtual_experts[selected] = expert;
        virtual_states[selected] = LLAMA_LS_SPEC_PENDING;
        virtual_recency[selected] = next_recency++;
        plan.issue[plan.n_issue] = expert;
        plan.issue_pos[plan.n_issue] = selected;
        ++plan.n_issue;
    }
    return plan;
}

llama_ls_pool_lookup llama_ls_pool_lookup_expert(const int32_t * experts,
                                                 const uint8_t * states,
                                                 const uint32_t * generations,
                                                 int cap, uint32_t generation,
                                                 int32_t expert) {
    llama_ls_pool_lookup result;
    if (experts == nullptr || states == nullptr || generations == nullptr ||
        cap <= 0 || expert < 0) {
        return result;
    }
    for (int pos = 0; pos < cap; ++pos) {
        if (experts[pos] != expert || states[pos] == LLAMA_LS_SPEC_FREE) {
            continue;
        }
        if (generations[pos] != generation) {
            result.stale_pos = pos;
            continue;
        }
        if (states[pos] == LLAMA_LS_SPEC_READY && result.ready_pos < 0) {
            result.ready_pos = pos;
        } else if (states[pos] == LLAMA_LS_SPEC_PENDING && result.pending_pos < 0) {
            result.pending_pos = pos;
        }
    }
    return result;
}

bool llama_ls_pool_retire_ready(int32_t * experts, uint8_t * states,
                                int64_t * recency, uint32_t * generations,
                                int cap, uint32_t generation, int pos) {
    if (experts == nullptr || states == nullptr || recency == nullptr || generations == nullptr ||
        pos < 0 || pos >= cap || states[pos] != LLAMA_LS_SPEC_READY ||
        generations[pos] != generation) {
        return false;
    }
    experts[pos] = -1;
    states[pos] = LLAMA_LS_SPEC_FREE;
    recency[pos] = -1;
    generations[pos] = 0;
    return true;
}

int llama_ls_pool_cancel_pending(uint8_t * states, int cap) {
    if (states == nullptr || cap <= 0) {
        return 0;
    }
    int count = 0;
    for (int pos = 0; pos < cap; ++pos) {
        if (states[pos] == LLAMA_LS_SPEC_PENDING) {
            states[pos] = LLAMA_LS_SPEC_CANCELLED;
            ++count;
        }
    }
    return count;
}

int llama_ls_pool_begin_generation(int32_t * experts, uint8_t * states,
                                   int64_t * recency, uint32_t * generations,
                                   int cap, uint32_t generation) {
    if (experts == nullptr || states == nullptr || recency == nullptr || generations == nullptr || cap <= 0) {
        return 0;
    }
    int changed = 0;
    for (int pos = 0; pos < cap; ++pos) {
        if (states[pos] == LLAMA_LS_SPEC_PENDING && generations[pos] != generation) {
            states[pos] = LLAMA_LS_SPEC_CANCELLED;
            ++changed;
        } else if (states[pos] == LLAMA_LS_SPEC_CANCELLED) {
            // The old event still owns these bytes until completion is fenced.
        } else if (states[pos] == LLAMA_LS_SPEC_READY && generations[pos] != generation) {
            experts[pos] = -1;
            states[pos] = LLAMA_LS_SPEC_FREE;
            recency[pos] = -1;
            generations[pos] = 0;
            ++changed;
        }
    }
    return changed;
}

int llama_ls_pool_complete(int32_t * experts, uint8_t * states,
                           int64_t * recency, uint32_t * generations,
                           int cap, int pos) {
    if (experts == nullptr || states == nullptr || recency == nullptr || generations == nullptr ||
        pos < 0 || pos >= cap) {
        return 0;
    }
    if (states[pos] == LLAMA_LS_SPEC_PENDING) {
        states[pos] = LLAMA_LS_SPEC_READY;
        return 1;
    }
    if (states[pos] == LLAMA_LS_SPEC_CANCELLED) {
        experts[pos] = -1;
        states[pos] = LLAMA_LS_SPEC_FREE;
        recency[pos] = -1;
        generations[pos] = 0;
        return 2;
    }
    return 0;
}
