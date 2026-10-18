#pragma once

#include "ggml.h"
#include "ggml-backend.h"

struct llm_graph_context;
struct llama_model;

// T-Q36-AER experimental hybrid MoE decode (LLAMA_LAYER_STREAMING builds only).
//
// Opt-in via env LLAMA_MOE_HYBRID=<VRAM cache slots per MoE layer>. For a
// single-token ubatch the routed experts that sit in a per-layer VRAM slot
// cache run on the GPU; the missed ones are posted to a pinned host mailbox and
// computed by a dedicated CPU thread pool over the model's own host expert
// tensors. Between tokens an EMA admission policy promotes experts into the
// cache on the CUDA auxiliary stream. Routing, expert ids and gate weights are
// never changed.
//
// build_ffn returns nullptr when the path does not apply (env unset, not a
// single-token decode, warmup, layer streaming active, dry-run model); the
// caller then builds the stock MoE FFN. Unsupported setups abort loudly.
// On success *pub receives the publish node that build_wait consumes.
ggml_tensor * llama_moe_hybrid_build_ffn(const llm_graph_context & g, const llama_model & model, int il,
                                         ggml_tensor * cur, ggml_tensor ** pub);

// cur + the host engine's weighted sum of the experts that missed the cache
ggml_tensor * llama_moe_hybrid_build_wait(const llm_graph_context & g, ggml_tensor * cur, ggml_tensor * pub);

// stop and free the engine bound to this backend, if any
void llama_moe_hybrid_release(ggml_backend_t backend);

// Resize the VRAM slot cache of the one engine in this process (between decodes only; opt-in). The old cache is freed
// before the new one is allocated and the new one is refilled from the frequency prior. Keeps env LLAMA_MOE_HYBRID in step.
// Returns false and changes nothing when there is no engine or the slot count is invalid.
bool llama_moe_hybrid_resize_impl(int n_slot);

// bytes the slot cache of the one engine in this process holds now and its slot count (the zero slot makes the size (n_slot + 1) x one slot row). False when there is no engine.
bool llama_moe_hybrid_cache_info_impl(int64_t * cache_bytes, int * n_slot);

// G31: shrink the cache so that need_free_bytes of device memory are free (and slack_bytes more when it has to shrink); synchronizes the device first. No-op when they already are free.
// *shrunk is set when the cache changed. False when there is no engine or the resize failed.
// the refill after a resize keeps the popularity statistics and takes the highest-scoring experts (the server, G31j); off (the default, the runner): statistics reset, frequency-table order, as recorded
void llama_moe_hybrid_set_keep_stats(bool keep);

// G31s: ids == nullptr: report the engine's geometry; otherwise fold the routing of K prompt tokens (ids [K][n_layer][n_used], -1 = none) into the popularity statistics. False when there is no engine.
bool llama_moe_hybrid_seed_impl(const int32_t * ids, int K, int * n_layer, int * n_used);

bool llama_moe_hybrid_make_room_impl(int64_t need_free_bytes, int64_t slack_bytes, int min_slots, bool * shrunk);

// G31: resize the cache so that keep_free_bytes + safety_bytes of device memory stay free (slots clamp to [min_slots, max_slots]); between decodes only. False when there is no engine or the resize failed.
bool llama_moe_hybrid_auto_resize_impl(int64_t keep_free_bytes, int64_t safety_bytes, int min_slots, int max_slots, int * slots_before, int * slots_after, int64_t * free_before_bytes);
