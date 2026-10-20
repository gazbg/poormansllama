#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

#ifdef LLAMA_LAYER_STREAMING
struct llama_layer_streaming;

// G32: the server's layer-major prefill (llama-lm-prefill.cpp)
struct llama_lm_prefill_params;
struct llama_lm_state;
struct llama_lm_state_deleter { void operator()(llama_lm_state * s) const; };
struct llama_dense_stream;
struct llama_dense_stream_deleter { void operator()(llama_dense_stream * s) const; };
struct llama_dense_stream_params;

// what decode() decided about a batch: taken by the layer-major pass or not
struct llama_lm_plan {
    bool                 ok          = false;
    llama_seq_id         seq         = 0;
    llama_pos            p0          = 0;     // position of the batch's first token
    int64_t              c0          = 0;     // KV cell of position 0 of the sequence (the pass addresses keys relative to it)
    int32_t              n           = 0;
    bool                 want_logits = false; // the last token's logits are wanted
    std::vector<int32_t> widths;              // chunk widths, in order
    std::vector<int32_t> snap_cuts;           // G42: token counts at which the recurrent state is kept (each one is a chunk boundary)
    std::vector<int32_t> snap_chunk;          // G42: the chunk that ends at each of them
    const char *         why         = nullptr; // when !ok: the reason (a static string)
    std::string          why_detail;                // when !ok because of the KV layout: which cell broke it
};
#endif

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // T-GPLR-71: arm the phase-scoped attention-view granularity; see
    // llama_memory_i::set_view_pad. Returns false when there is no memory, or
    // when the memory type cannot narrow its view -- so the caller can refuse
    // loudly instead of quietly running the whole-cache view it asked to remove.
    // 0 restores the constructor's n_pad exactly. Inline because llama_context
    // members are not exported from llama.dll (set_layer_streaming below is
    // inline for the same reason).
    bool set_attn_view_pad(uint32_t pad) {
        return memory ? memory->set_view_pad(pad) : false;
    }

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

#ifdef LLAMA_LAYER_STREAMING
    // experimental CUDA-only layer-streaming mode (see llama-layer-streaming.h)
    void set_layer_streaming(llama_layer_streaming * ls) { layer_streaming = ls; }

    // execute the per-piece graphs with slot orchestration (defined in
    // llama-layer-streaming.cpp)
    ggml_status layer_streaming_compute(llm_graph_result * res, const llm_graph_params & gparams);

    // pre-build the persistent layer-streaming graphs for `batch` without
    // executing them and without advancing KV/recurrent cell state. The next
    // llama_decode() of the same ubatch shape reuses these graphs, so the
    // timed region performs no graph construction. Returns false and sets
    // `ret` on failure.
    bool layer_streaming_prebuild(const llama_batch & batch, int64_t tg_tokens, ggml_status & ret);

    // Layer-major PP controller (SDD section 13, T-Q38LM-02). Drives the full
    // prompt in layer-major order: for each executed logical layer, every PP
    // chunk (in ascending token-position order) is run through that layer's
    // prebuilt per-chunk sub-graph before the next layer starts. Each chunk's
    // KV/recurrent inputs are rebound from its own memory context, so the
    // per-layer causal state advances chunk 0 then chunk 1 exactly as the
    // batch-major schedule does. Returns 0 on success.
    int layer_major_pp(const llama_batch & batch_inp, ggml_status & ret);

    // T-GPLR-03: argmax of the last output row of the PP logits (the first
    // token a normal-path greedy sampler would generate). layer_major_pp
    // bypasses decode(), so the host logits buffer is never populated; this
    // reads the exposed result tensor directly. Returns -1 when unavailable.
    int64_t layer_major_pp_first_token();

    // Opt-in diagnostics (--tg-logit-margin): top-k of the last PP output row,
    // the distribution P(.|prompt) a normal-path sampler reads for its first
    // token. Writes up to k (id, logit) pairs, best first. Returns the number
    // written, or -1 when unavailable.
    int layer_major_pp_topk(int k, int64_t * ids, float * vals);

    // Opt-in (--tg-release-compute): after prefill, release the scheduler's worst-case compute buffers (sized for the prefill
    // ubatch) and reserve again for single-token decode only. The caller must already have read whatever it needs from the
    // prefill graph results, and must not decode more than one token per ubatch afterwards. Returns the bytes released (sum
    // over backends), or -1 on failure.
    int64_t reserve_for_decode();

    // G30: the same with an explicit ubatch (the MTP draft context after its prompt catch-up decodes at most 2 + draft rows). Returns the bytes released, or -1.
    int64_t reserve_ubatch(uint32_t n_ubatch_new);

    // G31: forget the cached decode graph (it points at tensors that a hybrid cache resize freed); the next decode builds a new one
    void drop_graph_reuse();

    // G31s: install (or, with nullptr, remove) an evaluation callback for the next decodes; the scheduler's current one is replaced at once
    void set_eval_cb(ggml_backend_sched_eval_callback cb, void * user_data);

    // G32: layer-major prefill of one sequence's batch (llama-lm-prefill.cpp)
    bool lm_prefill_enable(const llama_lm_prefill_params & p, std::string & err);
    void lm_prefill_disable();
    bool lm_prefill_is_enabled() const { return lm_state != nullptr; }

    // dense decode streaming (llama-dense-stream.cpp)
    bool dense_stream_enable(const llama_dense_stream_params & p, std::string & err);
    void dense_stream_disable();
    bool dense_stream_is_enabled() const { return dstream != nullptr; }
    void dense_stream_stats(int64_t * v6) const;
    llama_lm_plan lm_make_plan(const llama_batch & b) const;
    void lm_note(const llama_batch & b, const llama_lm_plan & pl) const;   // LLAMA_LM_LOG=1: why a big enough batch was not taken
    bool lm_acquire(const llama_lm_plan & pl);   // false: nothing is held, the batch takes the ordinary path
    void lm_release();
    int  lm_run(const llama_lm_plan & pl, uint32_t n_outputs_all);   // 0 done, 1 nothing changed (ordinary path), < 0 failed
    bool lm_acquire_n(int64_t n_tokens, int32_t wmax);               // G42: the resources for a batch of n_tokens whose widest chunk is wmax
    bool lm_stage_first();                                           // G42: copy the banks of the first streamed layers into the slot window
    bool lm_decode_begin(const llama_batch & b, llama_lm_plan & plan);   // G42: what decode() does about the pass: plan, acquire (true: the resources are held for this decode), or give an armed pass back
    void lm_release_if_live();                                       // G42: the destructor's release of an armed pass
    bool lm_prearm(llama_seq_id seq, int64_t n_tokens);              // G42: see llama_lm_prefill_arm
    void lm_disarm();
    void lm_snap_request(const int32_t * cuts, int32_t n_cuts);
    int32_t lm_snap_count() const;
    int32_t lm_snap_cut(int32_t i) const;
    size_t  lm_snap_data(int32_t i, uint8_t * dst, size_t size) const;
    int64_t lm_n_pass() const;
    bool    lm_is_armed() const;
#endif

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

#ifdef LLAMA_LAYER_STREAMING
    // experimental layer-streaming module (owned by the caller)
    llama_layer_streaming * layer_streaming = nullptr;

    // G32: the server's layer-major prefill; declared last so it is destroyed before the backends and the memory it works on
    std::unique_ptr<llama_lm_state, llama_lm_state_deleter> lm_state;
    std::unique_ptr<llama_dense_stream, llama_dense_stream_deleter> dstream;
#endif

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
