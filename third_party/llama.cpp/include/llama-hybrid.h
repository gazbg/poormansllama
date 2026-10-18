#ifndef LLAMA_HYBRID_H
#define LLAMA_HYBRID_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private fork (G31): the hybrid MoE expert-cache decode, for frontends such as llama-server.
// The functions exist in every build; without LLAMA_LAYER_STREAMING (CUDA) they report "not available" and do nothing.

// true when this libllama was built with the hybrid MoE engine
LLAMA_API bool llama_hybrid_available(void);

// Set the engine's environment for variables that the caller did not set. Call BEFORE the context is created.
// initial_slots: expert-cache slots per layer at creation (<= 0: 14); freq_path: expert frequency table (NULL: uniform prior)
LLAMA_API void llama_hybrid_prepare_env(int32_t initial_slots, const char * freq_path);

// true when an engine exists in this process (the context was created with the environment above and expert banks on the host)
LLAMA_API bool llama_hybrid_active(void);

// Size the VRAM expert cache so that keep_free_mib + safety_mib of device memory stay free; slots clamp to [min_slots, max_slots].
// Call between decodes only (no request in flight). The context's cached decode graph is dropped. Returns false when there is no engine or the resize failed.
LLAMA_API bool llama_hybrid_auto_size(struct llama_context * ctx, int32_t keep_free_mib, int32_t safety_mib, int32_t min_slots, int32_t max_slots,
                                      int32_t * slots_before, int32_t * slots_after, int32_t * free_before_mib);

// Keep keep_free_mib + safety_mib of device memory free while the lazily committed KV cache grows (LLAMA_KV_LAZY): before every commit the expert cache shrinks as far as needed,
// by slack_mib more than the commit asks, so that the next commits do not shrink it again. ctx is the context whose decode graph is dropped when that happens outside a decode (a state restore).
// Call once, after the first llama_hybrid_auto_size.
LLAMA_API void llama_hybrid_set_margin(struct llama_context * ctx, int32_t keep_free_mib, int32_t safety_mib, int32_t slack_mib, int32_t min_slots);

// The prefill window: while a decode has a batch of >= min_tokens tokens, the physical batch size is ubatch_big (the stock-path prefill copies every expert bank once per ubatch, so a bigger one is much faster) and the
// expert cache gives the room for its compute buffers back afterwards. Call once, after llama_hybrid_set_margin. ubatch_big <= the context's ubatch: off.
LLAMA_API void llama_hybrid_set_prefill(struct llama_context * ctx, int32_t ubatch_big, int32_t min_tokens, int32_t max_slots);

// Seed the expert cache's popularity statistics from the routing of the last n_tokens tokens of a prompt (read from the 17 .. min_tokens-1 token decodes the server cuts before the end of a prompt), so that
// the first generated tokens already find the experts of the new prompt. n_tokens <= 0: off. Call once, after llama_hybrid_set_margin.
LLAMA_API void llama_hybrid_set_seed(struct llama_context * ctx, int32_t n_tokens);

// G32: layer-major prefill. A decode whose batch is >= min_tokens prompt tokens of one sequence runs layer by layer (every routed-expert bank is copied from the host ONCE into a VRAM slot window while all
// chunks of the batch go through that layer) instead of ubatch by ubatch (every ubatch copies all banks). The slot window, the activation slab and the pass's compute buffer exist only while a pass runs.
// Works with or without the hybrid engine (the decode after the prefill is whatever the context does: stock placement, or the hybrid expert cache). Batches it cannot take run the ordinary path.
struct llama_lm_prefill_params {
    int32_t n_slots;      // layer slots in the VRAM window; 2 overlaps the next bank's copy with the compute of this one
    int32_t ubatch;       // tokens per chunk (2048)
    int32_t segment;      // tokens per activation-slab segment, a multiple of ubatch (16384): bounds the slab, the layer sweep repeats per segment
    int32_t min_tokens;   // smallest batch that is taken (1024)
    int32_t act_bf16;     // 1: bf16 activation slab (half the memory), 0: f32
    int32_t device;       // CUDA device index (0)
};
LLAMA_API struct llama_lm_prefill_params llama_lm_prefill_default_params(void);

// Qwen3.5/3.6 MoE contexts only (flash attention on). Layers whose routed experts are on the device already (for example -ncmoe N) are computed in place. Call after the context exists and, with the hybrid
// engine, after llama_hybrid_set_margin. false + err: not enabled.
LLAMA_API bool llama_lm_prefill_enable(struct llama_context * ctx, const struct llama_lm_prefill_params * p, char * err, size_t err_len);
LLAMA_API void llama_lm_prefill_disable(struct llama_context * ctx);
LLAMA_API bool llama_lm_prefill_active(const struct llama_context * ctx);

// G42: checkpoints taken INSIDE the layer-major pass. For the NEXT llama_decode only: token counts (from the batch's first token, 0 < cut < n_tokens) at which the sequence's recurrent state is wanted.
// A pass that takes the batch cuts a chunk boundary at each of them and keeps the state there; llama_lm_prefill_snap_count says how many it made (0: the batch took the ordinary path, nothing was kept).
LLAMA_API void    llama_lm_prefill_snap_request(struct llama_context * ctx, const int32_t * cuts, int32_t n_cuts);
LLAMA_API int32_t llama_lm_prefill_snap_count(const struct llama_context * ctx);
LLAMA_API int32_t llama_lm_prefill_snap_cut(const struct llama_context * ctx, int32_t i);
// the bytes llama_state_seq_get_data_ext(ctx, dst, size, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) would have written when the sequence ended at that cut; dst == NULL: the size. Valid until the next llama_decode.
LLAMA_API size_t  llama_lm_prefill_snap_data(struct llama_context * ctx, int32_t i, uint8_t * dst, size_t size);

// G42: prepare the pass before the batch arrives. seq_id < 0: in the idle time, for a batch of about n_tokens (make room, allocate the slot window / slab / pass scheduler, copy the banks of the first layers);
// seq_id >= 0: for the batch of n_tokens that continues this sequence and that the caller is about to send whole (the sequence's cache layout is checked too; the batch is then taken whatever its size).
// true: the resources are held (armed) and the next llama_decode of a batch the pass can take uses them; any decode of <= 16 tokens, or of a big batch the pass cannot take, gives them back.
LLAMA_API bool    llama_lm_prefill_arm(struct llama_context * ctx, llama_seq_id seq_id, int32_t n_tokens);
LLAMA_API void    llama_lm_prefill_disarm(struct llama_context * ctx);
LLAMA_API bool    llama_lm_prefill_armed(const struct llama_context * ctx);
LLAMA_API int64_t llama_lm_prefill_n_pass(const struct llama_context * ctx);

// Stop the helper threads and the callbacks of this API; call before the context is freed.
LLAMA_API void llama_hybrid_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_HYBRID_H
