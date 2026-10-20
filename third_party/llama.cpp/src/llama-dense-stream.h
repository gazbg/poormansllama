#pragma once

// Dense decode streaming (private fork, CUDA, dense Qwen3.5/3.8): a decode token walks the layers in order; the weights of a layer are read once. Layers whose weights are in host memory are computed by
// the CPU backend (the ordinary placement) -- except the STREAMED layers of the plan, which are computed on the device from a VRAM slot: while the CPU computes the layers between two streamed layers, the
// copy engine brings the next streamed layer's weights into the slot that the previous one freed. The slot protocol is the runner's (copy backend with its own stream, one `ready` and one `done` event per slot),
// but the waits and the records are not issued by a driver loop: two marker nodes per streamed layer in the ordinary decode graph (llama_dense_stream::mark, a SCALE by 1.0 flagged GGML_TENSOR_FLAG_STREAM_MARK)
// make the CUDA backend call mark_cb() at the right points of the stream, so the graph keeps the ordinary reuse / scheduling machinery.
//
// Ring protocol: the streamed layers are numbered 0..K-1 in layer order; layer k lives in slot k % n_slots. Every execution of a graph that has the markers runs layers 0..K-1 once, in order. Done(k) records the
// slot's `done` event on the compute stream and queues the copy of layer (k + n_slots) % K into the slot (the copy stream waits for the event; across the end of the token the copy is for the NEXT token). Wait(k)
// makes the compute stream wait for the slot's `ready` event; if the slot does not hold layer k (a graph that did not run to its end), the layer is copied synchronously first -- never silently wrong data.

#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <vector>

#ifdef LLAMA_LAYER_STREAMING

struct llama_dense_stream {
    ~llama_dense_stream();

    // layers: ascending, all dense layers whose package is in host memory. false + err: not usable.
    bool init(const llama_model & model, const std::vector<int> & layers, int n_slots, int device, ggml_backend_t backend_compute, std::string & err);

    // the streamed layer index (0..K-1) of decoder layer il, or -1
    int index_of(int il) const {
        return il >= 0 && (size_t) il < k_of_layer.size() ? k_of_layer[(size_t) il] : -1;
    }
    // the slot-bound weights of streamed layer k
    const llama_layer & bundle(int k) const { return layers[(size_t) k].bundle; }

    int n_streamed() const { return (int) layers.size(); }
    int n_slots() const { return (int) slots.size(); }

    // The ring gives its device memory to the layer-major prefill pass for the length of the pass (both want the same VRAM): suspend() waits for the streams and frees the slot buffers, resume() allocates them again
    // (same sizes, the bundle tensors are bound again) and queues the first layers of the ring. A marked graph that is computed while suspended resumes the ring first.
    size_t held_bytes() const { return suspended ? 0 : bytes_held; }
    bool   is_suspended() const { return suspended; }
    void   suspend();
    bool   resume();

    // a marker node after x in ctx: kind 0 = wait (before the first kernel that reads layer k's weights), 1 = done (after the last one)
    ggml_tensor * mark(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * x, int kind, int k) const;

    // the function ggml_backend_cuda_set_stream_mark calls
    static void mark_cb(void * user, const ggml_tensor * node, int phase);

    // statistics (the callbacks run on the thread that computes the graph)
    int64_t n_wait      = 0;
    int64_t n_done      = 0;
    int64_t n_recover   = 0;    // waits that found the wrong layer in the slot (a graph that did not run to its end)
    int64_t n_suspend   = 0;
    int64_t n_copies    = 0;
    int64_t bytes_copied = 0;
    double  t_issue_us  = 0.0;  // host time spent issuing the copies

private:
    struct member_t {
        ggml_tensor * dst = nullptr;       // slot-bound tensor of the bundle
        const void *  src = nullptr;       // the host tensor's data (page-locked)
        size_t        nbytes = 0;
        size_t        off = 0;             // offset inside the slot buffer
    };
    struct layer_t {
        int                   il = -1;
        int                   slot = 0;
        llama_layer           bundle;
        std::vector<member_t> members;
    };
    struct slot_t {
        ggml_backend_buffer_ptr buf;
        ggml_backend_event_t    ev_ready = nullptr;
        ggml_backend_event_t    ev_done  = nullptr;
        int                     holds = -1;     // the streamed layer whose copy was queued into this slot last (-1: none)
        ggml_context_ptr        ctx_meta;
        size_t                  bytes = 0;      // size of the slot buffer
    };

    void issue_copy(int k);                      // queue layer k's weights into its slot (waits for the slot's done event first)
    void on_wait(int k);
    void on_done(int k);

    ggml_backend_t        backend_compute = nullptr;
    ggml_backend_t        backend_copy    = nullptr;
    ggml_backend_dev_t    dev             = nullptr;
    std::vector<layer_t>  layers;
    std::vector<slot_t>   slots;
    std::vector<int>      k_of_layer;
    std::vector<const void *> pinned;            // page-locked spans this object owns
    std::vector<size_t>       pinned_bytes;
    bool                  mark_installed = false;
    bool                  suspended = false;
    size_t                bytes_held = 0;
};

#endif // LLAMA_LAYER_STREAMING
