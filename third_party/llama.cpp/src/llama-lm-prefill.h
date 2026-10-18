#pragma once

// G32: state of the server's layer-major prefill (llama-lm-prefill.cpp); internal to libllama

#include "llama-hybrid.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef LLAMA_LAYER_STREAMING

struct llama_layer_streaming;

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

    // statistics
    int64_t n_pass  = 0;
    int64_t n_attempt = 0;     // lm_run entries (fault injection)
    int64_t n_tok   = 0;
    double  t_total = 0.0;
};

#endif // LLAMA_LAYER_STREAMING
