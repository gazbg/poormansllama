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
