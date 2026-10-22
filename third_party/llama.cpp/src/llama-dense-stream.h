#pragma once

// Dense decode streaming (private fork, CUDA, dense Qwen3.5/3.8): a decode token walks the layers in order; the weights of a layer are read once. Layers whose weights are in host memory are computed by
// the CPU backend (the ordinary placement) -- except the RING layers of the plan, which are computed on the device from a VRAM slot: while the CPU computes the layers between two ring layers, the
// copy engine brings the next ring layer's weights into the slot that the previous one freed. The slot protocol is the runner's (copy backend with its own stream, one `ready` and one `done` event per slot),
// but the waits and the records are not issued by a driver loop: two marker nodes per ring layer in the ordinary decode graph (llama_dense_stream::mark, a SCALE by 1.0 flagged GGML_TENSOR_FLAG_STREAM_MARK)
// make the CUDA backend call mark_cb() at the right points of the stream, so the graph keeps the ordinary reuse / scheduling machinery.
//
// Ring protocol: the ring layers are numbered 0..K-1 in layer order; ring layer k lives in slot k % n_slots. Every execution of a graph that has the markers runs ring layers 0..K-1 once, in order. Done(k) records
// the slot's `done` event on the compute stream and queues the copy of ring layer (k + n_slots) % K into the slot (the copy stream waits for the event; across the end of the token the copy is for the NEXT token). Wait(k)
// makes the compute stream wait for the slot's `ready` event; if the slot does not hold ring layer k (a graph that did not run to its end), the layer is copied synchronously first -- never silently wrong data.
//
// The plan (which host layers are ring layers) is fixed (an explicit list) or DYNAMIC (G-DYN): every host layer is a candidate, the statistics of the previous passes (copy time of a layer, CPU time between ring
// layers, GPU time of a ring layer; include/llama-dense-plan.h) choose the ring layers, and the plan changes only between passes (begin_pass() tells the context to rebuild the decode graph). The first n_slots ring
// layers are always the first n_slots host layers, so the copies queued at the end of a pass stay valid whatever plan the next pass uses.

#include "llama-model.h"

#include "llama-dense-plan.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef LLAMA_LAYER_STREAMING

struct llama_dense_stream {
    ~llama_dense_stream();

    // host: every dense layer whose package is in host memory (ascending). ring: the ring layers of a fixed plan (a subset of host, ascending); in a dynamic plan it is ignored and the planner chooses
    // the first ring (from priors; the statistics improve it). false + err: not usable.
    // dedicated (G-EVICT): host layers that get a VRAM slot of their own, never recycled: they are computed on the device in decode steps like resident layers, and the layer-major prefill pass takes the slot memory
    // (suspend()) and streams them from host memory like any host layer; the first decode step puts them back (resume()). Not ring candidates.
    bool init(const llama_model & model, const std::vector<int> & host, const std::vector<int> & ring, const std::vector<int> & dedicated, int n_slots, int device, ggml_backend_t backend_compute, bool dynamic, int percentile, std::string & err);

    // the ring ordinal (0..K-1) of decoder layer il in the active plan, or -1
    int index_of(int il) const {
        return il >= 0 && (size_t) il < k_of_layer.size() ? k_of_layer[(size_t) il] : -1;
    }
    // the slot-bound weights of ring layer k
    const llama_layer & bundle(int k) const { return cand[(size_t) cand_of_layer[(size_t) ring[(size_t) k]]].bundle[(size_t) (k % n_slots())]; }

    // the weights of dedicated layer il in its own slot, or nullptr
    const llama_layer * dedicated(int il) const {
        return il >= 0 && (size_t) il < ded_of_layer.size() && ded_of_layer[(size_t) il] >= 0 ? &ded[(size_t) ded_of_layer[(size_t) il]].bundle : nullptr;
    }
    int n_dedicated() const { return (int) ded.size(); }

    int n_streamed() const { return (int) ring.size(); }
    int n_slots() const { return (int) slots.size(); }
    bool is_dynamic() const { return dynamic; }
    int  plan_epoch() const { return epoch; }
    const std::vector<int> & ring_layers() const { return ring; }

    // Called by the context before the graph of a decode step is built or reused. Closes the statistics of the previous pass and, in a dynamic plan, re-plans when it is due. true = the plan changed: the
    // graph (its ring layers and markers are bound at build time) must be built again.
    bool begin_pass();

    // The ring gives its device memory to the layer-major prefill pass for the length of the pass (both want the same VRAM): suspend() waits for the streams and frees the slot buffers, resume() allocates them again
    // (same sizes, the bundle tensors are bound again) and queues the first layers of the ring. A marked graph that is computed while suspended resumes the ring first.
    size_t held_bytes() const { return suspended ? 0 : bytes_held; }
    bool   is_suspended() const { return suspended; }
    void   suspend();
    bool   resume();

    // a marker node after x in ctx: kind 0 = wait (before the first kernel that reads ring layer k's weights), 1 = done (after the last one)
    ggml_tensor * mark(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * x, int kind, int k) const;

    // the function ggml_backend_cuda_set_stream_mark calls
    static void mark_cb(void * user, const ggml_tensor * node, int phase);

    // statistics (the callbacks run on the thread that computes the graph)
    int64_t n_wait      = 0;
    int64_t n_done      = 0;
    int64_t n_recover   = 0;    // waits that found the wrong layer in the slot (a graph that did not run to its end)
    int64_t n_suspend   = 0;
    int64_t n_notready  = 0;    // waits that found the layer's copy still in flight
    int64_t n_copies    = 0;
    int64_t bytes_copied = 0;
    double  t_issue_us  = 0.0;  // host time spent issuing the copies
    double  t_stall_us  = 0.0;  // host time spent waiting for the paced worker to finish issuing a layer
    int64_t n_replans   = 0;    // plans that were adopted after the first
    int64_t n_passes    = 0;    // complete passes seen

private:
    struct member_t {
        const void *  src = nullptr;       // the host tensor's data (page-locked)
        size_t        nbytes = 0;
        size_t        off = 0;             // offset inside the slot buffer
        std::vector<ggml_tensor *> dst;    // [slot]: the slot-bound tensor of the bundle
    };
    struct cand_t {                        // one per host layer that can be a ring layer
        int                       il = -1;
        size_t                    bytes = 0;     // size of the package
        std::vector<llama_layer>  bundle;        // [slot]
        std::vector<member_t>     members;
    };
    struct slot_t {
        ggml_backend_buffer_ptr buf;
        ggml_backend_event_t    ev_ready = nullptr;
        ggml_backend_event_t    ev_done  = nullptr;
        int                     holds = -1;     // the decoder layer whose copy was queued into this slot last (-1: none)
        int                     rec = -1;       // the decoder layer whose copy and ready event are fully issued (paced worker; guarded by mu)
        ggml_context_ptr        ctx_meta;
        size_t                  bytes = 0;      // size of the slot buffer
    };
    struct req_t {
        int il;
        int slot;
    };
    struct ded_t {                         // a dedicated layer: one buffer, one set of slot-bound tensors, filled at start-up and after every layer-major pass
        int                       il = -1;
        size_t                    bytes = 0;
        ggml_context_ptr          ctx_meta;
        ggml_backend_buffer_ptr   buf;
        llama_layer               bundle;
        std::vector<member_t>     members;     // dst[0] is the tensor
    };

    void issue_copy(int il, int slot);           // queue layer il's weights into the slot (waits for the slot's done event first)
    void post_copy(int il, int slot);            // issue_copy, or hand the layer to the paced copy worker
    void copy_paced(int il, int slot);           // worker: the copy of a layer in pieces with at most `window` in flight
    void worker_main();
    void drain_worker();
    void stop_worker();
    void on_wait(int k);
    void on_done(int k);
    void bind_all();                             // bind every bundle tensor to its slot buffer
    void bind_ded();                             // bind the tensors of the dedicated layers to their buffers
    void fill_ded();                             // copy the dedicated layers' weights into their buffers (waits for the copies)
    void set_plan(const std::vector<int> & r);   // the active plan (ring layers, ordinals, first copies)
    void close_pass();                           // statistics of the pass that just ended
    bool replan();                               // dynamic: choose the ring from the statistics; true = changed

    // Paced copy (LLAMA_DS_PIECE_KIB > 0): a small host-to-device copy of the compute path (the activation hop of a CPU layer) waits behind every earlier command on the copy engine,
    // so one 347 MiB layer copy in a single piece blocks it for 22 ms. Pieces with a bounded number in flight let it through after one piece.
    size_t                    piece_bytes = 0;
    int                       window = 2;
    int                       poll_us = 0;
    bool                      hop_kernel = false;
    std::thread               worker;
    std::mutex                mu;
    std::condition_variable   cv;
    std::deque<req_t>         reqs;
    bool                      stopping = false;
    bool                      busy = false;
    std::vector<ggml_backend_event_t> piece_ev;

    ggml_backend_t        backend_compute = nullptr;
    ggml_backend_t        backend_copy    = nullptr;
    ggml_backend_dev_t    dev             = nullptr;
    int                   n_layer_model   = 0;
    std::vector<cand_t>   cand;
    std::vector<int>      cand_of_layer;         // decoder layer -> index in cand, -1
    std::vector<int>      ring;                  // the active plan: ring layers, ascending
    std::vector<int>      k_of_layer;            // decoder layer -> ring ordinal, -1
    std::vector<slot_t>   slots;
    std::vector<ded_t>    ded;                   // dedicated layers
    std::vector<int>      ded_of_layer;          // decoder layer -> index in ded, -1
    std::vector<const void *> pinned;            // page-locked spans this object owns
    std::vector<size_t>       pinned_bytes;
    bool                  mark_installed = false;
    bool                  suspended = false;
    size_t                bytes_held = 0;
    int                   epoch = 0;

    // dynamic plan: the statistics and the decisions live in the controller (include/llama-dense-plan.h, host-tested)
    bool                  dynamic = false;
    llama_dense_plan::controller ctl;
    std::vector<int64_t>  pass_t_wait, pass_t_done;       // host time (us) of the markers of the running pass, by ring ordinal
    std::vector<void *>   ev_t0, ev_t1;                   // CUDA timing events around the device compute of ring layer k (start after the copy wait, end at done)
    int                   pass_waits = 0, pass_dones = 0, pass_late = 0;
    int                   pass_k = 0;                     // K of the running pass
    std::vector<int>      pass_ring;                      // the ring of the running pass
    std::mutex            mu_stat;                        // the copy samples come from the worker
    int64_t               t_copy_start_us = 0;            // worker thread only: the start of the DMA of the copy it is issuing
    std::vector<double>   copy_samples;                   // pending, from the worker
};

#endif // LLAMA_LAYER_STREAMING
