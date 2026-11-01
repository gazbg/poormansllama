#pragma once

// G32: state of the server's layer-major prefill (llama-lm-prefill.cpp); internal to libllama

#include "llama-hybrid.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef LLAMA_LAYER_STREAMING

struct llama_layer_streaming;
struct llama_context;
class  llama_kv_cache;
class  llama_memory_recurrent_context;

// H94: what the two contexts of a sequence-split pass share while it runs
struct llama_lm_link {
    std::atomic<int>   layers_done{0};          // leader: layers finished for its chunks (the follower may take layer il once this is > il)
    std::atomic<bool>  abort{false};            // either side: stop waiting
    std::atomic<int>   fstate{0};               // follower: 0 running, 1 done, -1 failed
    int                kc          = 0;         // first chunk of the follower
    int64_t            lead_tokens = 0;         // tokens of the leader's chunks
    std::vector<int32_t> widths;                // the leader's chunk widths (the follower's plan must be the same)
    llama_kv_cache *   lead_kv     = nullptr;
    int64_t            lead_c0     = 0;         // KV cell of position 0 in the leader
    const llama_memory_recurrent_context * lead_rc = nullptr;   // the recurrent cell of the leader's last chunk (valid once layers_done > 0)
    float *            lead_logits   = nullptr;  // the leader's logits row, written by the follower at the end
    size_t             lead_logits_n = 0;
    std::mutex              m;
    std::condition_variable cv;
};

// the follower's persistent thread: it runs llama_decode(follower, batch) when the leader hands it a pass
struct llama_lm_peer_worker {
    ~llama_lm_peer_worker();
    llama_context *         ctx = nullptr;      // the follower
    std::thread             th;
    std::mutex              m;
    std::condition_variable cv;
    std::condition_variable cv_done;
    bool                    stop = false;
    bool                    have = false;
    std::atomic<int>        result{0};
    std::atomic<int>        done{1};            // 1 = idle / finished
    llama_lm_link *         link = nullptr;
    // the follower's copy of the batch (sequence 0, logits for the last token only)
    std::vector<llama_token>   tokens;
    std::vector<llama_pos>     pos;
    std::vector<int32_t>       n_seq_id;
    std::vector<llama_seq_id>  seq_ids;
    std::vector<llama_seq_id*> seq_id_ptr;
    std::vector<int8_t>        logits_flag;
    void start(llama_context * follower);
    void kick(llama_lm_link * l, const llama_batch & b, bool want_logits);
    int  wait();                                // blocks until the pass is over; its llama_decode result
};

struct llama_lm_state {
    ~llama_lm_state();

    llama_lm_prefill_params p = {};

    // the slot window / feed machinery of the runner; its device buffers exist only while a pass runs
    std::unique_ptr<llama_layer_streaming> ls;

    int                  n_layer  = 0;
    std::vector<uint8_t> native;                    // per layer: 1 = the routed experts are on the device already (no slot, no copy)
    ggml_type            act_type = GGML_TYPE_F32;

    // resources of the pass that is running
    bool                   live          = false;
    bool                   slots_live    = false;
    bool                   vram_hooked   = false;
    bool                   sched_swapped = false;
    ggml_backend_buffer_ptr slab;                   // two halves of [n_embd, slab_tokens] activations
    size_t                 slab_half     = 0;
    int64_t                slab_tokens   = 0;
    ggml_backend_sched_ptr sched_pp;                // while a pass runs: holds the context's own scheduler (the pass's is swapped in); otherwise empty
    size_t                 sched_est     = 0;       // compute buffer the last pass needed (for the next pass's room estimate)
    int64_t                free_pre      = 0;       // device free memory when the room was made for the running pass
    int64_t                slot_total    = 0;       // the running pass's slot window
    int32_t                cur_w         = 0;       // the running pass's widest chunk
    int64_t                var_meas      = 0;       // what the last pass really took besides slot window and slab (scheduler buffer, CUDA pools, temporaries)
    int32_t                var_w         = 0;       // the widest chunk of that pass
    int64_t                t_acquire_us  = 0;       // time the last lm_acquire took

    // G42: checkpoints taken inside the pass
    std::vector<int32_t>   snap_req;                // token counts (from the first token of the next batch) at which the recurrent state is wanted
    struct snap_t {
        int32_t              cut = 0;               // tokens of the batch before the snapshot
        std::vector<uint8_t> blob;                  // what llama_state_seq_get_data_ext(PARTIAL_ONLY) writes for the sequence at that point
    };
    std::vector<snap_t>    snaps;                   // the snapshots the last decode made (empty: that decode made none)
    int64_t                n_snap        = 0;

    // G42: the pass prepared in the idle time
    bool                   armed         = false;   // slot window, slab and pass scheduler are held for the next pass (live is true too)
    bool                   armed_batch   = false;   // ... for a batch the caller decided to send whole: taken whatever its size
    int32_t                armed_w       = 0;       // widest chunk the armed room was made for
    int64_t                armed_n       = 0;       // tokens of the batch it was armed for (armed_batch)
    int64_t                n_arm         = 0;
    int64_t                n_arm_used    = 0;
    int64_t                n_arm_wasted  = 0;
    int64_t                t_arm_us      = 0;       // time the last arm took

    // H94: the follower of a sequence split
    llama_context *                       peer_ctx = nullptr;
    int32_t                               peer_min = 0;
    std::unique_ptr<llama_lm_peer_worker> peer_worker;
    int64_t                               n_split = 0;      // passes that ran split
    int64_t                               n_split_fallback = 0;

    // statistics
    int64_t n_pass  = 0;
    int64_t n_attempt = 0;     // lm_run entries (fault injection)
    int64_t n_tok   = 0;
    double  t_total = 0.0;
};

#endif // LLAMA_LAYER_STREAMING
